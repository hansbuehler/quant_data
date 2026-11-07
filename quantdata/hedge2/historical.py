"""Fixed-instrument historical P&L from the saved licensed 0.1 SANOS surfaces."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, count_business_days_between, get_business_days
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.source import SourceArchive, surface_digitals
from quantdata.sanos.arbfree import ArbFreePricer


@dataclass(frozen=True, kw_only=True)
class HistoricalReturns:
    instrument_gains: np.ndarray
    raw_instrument_gains: np.ndarray
    claim_gains: np.ndarray
    removed_instrument_mean: np.ndarray
    weights: np.ndarray
    pair_start_dates: np.ndarray
    pair_end_dates: np.ndarray
    mapped_cash_strikes: np.ndarray
    mapped_target_strikes: np.ndarray
    source_hashes: np.ndarray
    curve_hashes: np.ndarray
    carry_inputs: np.ndarray
    source_first_business_dtes: np.ndarray
    short_expiry_extrapolation_counts: np.ndarray
    outside_source_strike_counts: np.ndarray
    starting_instrument_prices: np.ndarray
    raw_cash_gains: np.ndarray
    return_basis: str
    return_convention: str = "one-day fixed-instrument P&L, not premium returns"
    contract_mapping: str = "K_history=K_today*S_history_start/S_today; fixed through each pair"
    target_convention: str = "historical P&L with historical initial marks, not today's conditional payoff law"
    strike_coverage_convention: str = "global latent-density support envelope, not observed-quote or maturity-specific coverage"


def historical_pairs(*, dates: tuple[date, ...], as_of_date: date,
                     window_transitions: int = 252, history_start: date = date(2022, 1, 1)) -> tuple[tuple[date, date], ...]:
    """Use only complete exact sessions within history, including today's closing observation."""
    if window_transitions < 1 or history_start > as_of_date or any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Historical window, sorted unique dates and decision date must be valid")
    scoped = tuple(day for day in dates if history_start <= day <= as_of_date)
    sessions = {day: index for index, day in enumerate(get_business_days(history_start, as_of_date))}
    pairs = tuple((before, after) for before, after in zip(scoped, scoped[1:])
                  if before in sessions and after in sessions and sessions[after] - sessions[before] == 1)
    if len(pairs) < window_transitions:
        raise ValueError(f"Need {window_transitions} prior exact transitions, found {len(pairs)}")
    return pairs[-window_transitions:]


def licensed_curve(directory: Path, day: date) -> CashCurve:
    """Read source and financing forwards separately, without loading old quote books."""
    frame = pd.read_parquet(directory / "SPX" / f"{day.isoformat()}.pdq",
        columns=["date", "spot", "is_am", "bdte", "fwd", "fwd_ref", "df"])
    if not (frame["date"] == pd.Timestamp(day)).all() or frame["spot"].nunique() != 1:
        raise ValueError(f"Historical spot/date mismatch on {day}")
    live = frame.loc[(~frame["is_am"]) & (frame["bdte"] > 0.)].groupby("bdte", sort=True)
    if (live[["fwd", "fwd_ref", "df"]].nunique() > 1).any().any():
        raise ValueError(f"Inconsistent historical carry on {day}")
    values = live[["fwd", "fwd_ref", "df"]].first()
    return CashCurve(spot=float(frame["spot"].iloc[0]), business_dtes=values.index.to_numpy(dtype=float),
        forwards=values["fwd"].to_numpy(dtype=float), reference_forwards=values["fwd_ref"].to_numpy(dtype=float),
        discounts=values["df"].to_numpy(dtype=float))


def sanos_europeans(*, source: ArbFreePricer, curve: CashCurve, business_dtes: np.ndarray,
                    cash_strikes: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Use the saved surface's call pricer and put-call parity; zero DTE settles."""
    days, strikes, calls = np.broadcast_arrays(business_dtes, cash_strikes, is_call)
    if (days.ndim != 1 or not np.isfinite(days).all() or np.any(days < 0.)
            or not np.isfinite(strikes).all() or np.any(strikes < 0) or np.any((strikes == 0) & ~calls)):
        raise ValueError("Historical European strikes/remaining tenors must be finite and feasible")
    output = np.empty(len(days))
    settled = days < 1e-6
    difference = curve.spot - strikes[settled]
    output[settled] = np.maximum(np.where(calls[settled], difference, -difference), 0.)
    live = ~settled
    zero_calls = live & (strikes == 0.)
    if zero_calls.any():
        forward, _, discount = curve.at(days[zero_calls])
        output[zero_calls] = discount * forward
    live &= strikes > 0.
    if live.any():
        forward, _, discount = curve.at(days[live])
        normalized = strikes[live] / forward
        call_values = np.asarray(source.price(expiries=days[live] / DAYS_PER_YEAR, strikes=normalized, mode="surface"))
        output[live] = discount * forward * np.where(calls[live], call_values, call_values + normalized - 1.)
    if not np.isfinite(output).all() or np.min(output, initial=0.) < -1e-5 * curve.spot:
        raise ValueError("Saved SANOS historical European prices are invalid")
    return output


def historical_pair_returns(*, before_source: ArbFreePricer, after_source: ArbFreePricer,
                            before_curve: CashCurve, after_curve: CashCurve,
                            current_spot: float, episode_initial_spot: float,
                            book: pd.DataFrame, claims: tuple[Claim, ...],
                            elapsed_day: int = 0, discount_to_current: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """Map contracts once at the historical start, then preserve cash strike and side.

    Each pair starts at today's spot-relative strike and remaining expiry.
    Unit-cash digital returns are not multiplied by the spot notional scale.
    These are historical mark changes, not today's fixed-premium payoff law.
    """
    if (current_spot <= 0 or episode_initial_spot <= 0 or discount_to_current <= 0
            or not np.isfinite([current_spot, episode_initial_spot, discount_to_current]).all()
            or not claims or any(claim.kind != "digital" or claim.terminal_day <= elapsed_day for claim in claims)):
        raise ValueError("The historical European-return adapter requires live digital targets and positive scales")
    tenors = book["bdte"].to_numpy(dtype=float)
    if np.any(tenors < 1. - 1e-6):
        raise ValueError("A historical pair cannot start with an expired hedge instrument")
    strikes = book["strike"].to_numpy(dtype=float) * before_curve.spot / current_spot
    calls = (book["option_type"] == "call").to_numpy()
    start = sanos_europeans(source=before_source, curve=before_curve,
        business_dtes=tenors, cash_strikes=strikes, is_call=calls)
    following = sanos_europeans(source=after_source, curve=after_curve,
        business_dtes=np.maximum(tenors - 1., 0.), cash_strikes=strikes, is_call=calls)
    _, reference, one_day_discount = before_curve.at(1.)
    scale = discount_to_current * current_spot / (episode_initial_spot * before_curve.spot)
    gains = scale * np.r_[one_day_discount * (after_curve.spot - reference), one_day_discount * following - start]
    target_gains = np.empty(len(claims))
    for index, claim in enumerate(claims):
        remaining = claim.terminal_day - elapsed_day
        strike = claim.strike * before_curve.spot / current_spot
        forward, _, discount = before_curve.at(float(remaining))
        old_price = float(discount * surface_digitals(before_source, expiry_y=remaining / DAYS_PER_YEAR,
            strikes=np.array([strike / forward]), is_call=claim.is_call)[0])
        if remaining == 1:
            new_price = float(after_curve.spot > strike if claim.is_call else after_curve.spot <= strike)
        else:
            forward, _, discount = after_curve.at(float(remaining - 1))
            new_price = float(discount * surface_digitals(after_source, expiry_y=(remaining - 1) / DAYS_PER_YEAR,
                strikes=np.array([strike / forward]), is_call=claim.is_call)[0])
        target_gains[index] = discount_to_current * (one_day_discount * new_price - old_price)
    if not np.isfinite(gains).all() or not np.isfinite(target_gains).all():
        raise ValueError("Historical joint instrument/target returns are non-finite")
    return gains, target_gains


def build_historical_returns(*, archive: SourceArchive, licensed_fit_directory: Path,
                             as_of_date: date, current_spot: float,
                             book: pd.DataFrame, claims: tuple[Claim, ...],
                             episode_initial_spot: float | None = None,
                             elapsed_day: int = 0, discount_to_current: float = 1.,
                             window_transitions: int = 252,
                             history_start: date = date(2022, 1, 1),
                             return_basis: str = "spot_notional",
                             training_pairs: tuple[tuple[date, date], ...] | None = None,
                             progress: Callable[[str], None] | None = None) -> HistoricalReturns:
    """One common causal pair set, uniform demeaning of instruments only, full provenance."""
    available = archive.dates()
    if return_basis not in ("spot_notional", "premium"):
        raise ValueError("Choose spot-notional or historical entry-premium returns")
    pairs = (historical_pairs(dates=available, as_of_date=as_of_date,
        window_transitions=window_transitions, history_start=history_start)
        if training_pairs is None else training_pairs)
    if (not pairs or len(pairs) != window_transitions or tuple(sorted(set(pairs))) != pairs
            or any(before < history_start or after > as_of_date or before not in available or after not in available
                   for before, after in pairs)
            or np.any(count_business_days_between(np.array([before for before, _ in pairs], dtype="datetime64[D]"),
                np.array([after for _, after in pairs], dtype="datetime64[D]")) != 1)):
        raise ValueError("Historical training pairs must be unique, causal and exact sessions in the source")
    initial_spot = current_spot if episode_initial_spot is None else episode_initial_spot

    @lru_cache(maxsize=2)
    def load(day: date):
        source, metadata = archive.load(day)
        curve_path = licensed_fit_directory / "SPX" / f"{day.isoformat()}.pdq"
        return source, metadata, licensed_curve(licensed_fit_directory, day), hashlib.sha256(curve_path.read_bytes()).hexdigest()

    instruments, targets, hashes, curve_hashes, carry, old_spots, starting_prices = [], [], [], [], [], [], []
    first_tenors, extrapolated, outside = [], [], []
    started = time.monotonic()
    remaining_days = np.r_[book["bdte"].to_numpy(dtype=float), [claim.terminal_day - elapsed_day for claim in claims]]
    cash_strikes = np.r_[book["strike"].to_numpy(dtype=float), [claim.strike for claim in claims]]
    for index, (before, after) in enumerate(pairs):
        old_source, old_meta, old_curve, old_curve_hash = load(before)
        new_source, new_meta, new_curve, new_curve_hash = load(after)
        old_spots.append(old_curve.spot)
        gains, claim_gains = historical_pair_returns(before_source=old_source, after_source=new_source,
            before_curve=old_curve, after_curve=new_curve, current_spot=current_spot,
            episode_initial_spot=initial_spot, book=book, claims=claims, elapsed_day=elapsed_day,
            discount_to_current=discount_to_current)
        instruments.append(gains)
        targets.append(claim_gains)
        option_start = sanos_europeans(source=old_source, curve=old_curve,
            business_dtes=book["bdte"].to_numpy(dtype=float),
            cash_strikes=book["strike"].to_numpy(dtype=float) * old_curve.spot / current_spot,
            is_call=(book["option_type"] == "call").to_numpy())
        starting_prices.append(discount_to_current * current_spot / initial_spot *
            np.r_[1., option_start / old_curve.spot])
        hashes.append([old_meta["source_sha256"], new_meta["source_sha256"]])
        curve_hashes.append([old_curve_hash, new_curve_hash])
        pair_carry, pair_first, pair_extrapolated, pair_outside = [], [], [], []
        mapped = cash_strikes * old_curve.spot / current_spot
        for source, curve, days in ((old_source, old_curve, remaining_days),
                                    (new_source, new_curve, remaining_days - 1.)):
            forward, reference, discount = curve.at(np.maximum(days, 0.))
            pair_carry.append(np.stack([forward, reference, discount], axis=-1))
            minimum = float(np.min(source.expiries) * DAYS_PER_YEAR)
            pair_first.append(minimum)
            pair_extrapolated.append(np.sum((days > 0.) & (mapped > 0.) & (days < minimum - 1e-6)))
            lower = min(float(np.min(values)) for values in source.mstrikes)
            upper = max(float(np.max(values)) for values in source.mstrikes)
            pair_outside.append(np.sum((days > 0.) & (mapped > 0.) & ((mapped / forward < lower) | (mapped / forward > upper))))
        carry.append(pair_carry)
        first_tenors.append(pair_first)
        extrapolated.append(pair_extrapolated)
        outside.append(pair_outside)
        if progress is not None and (index == 0 or (index + 1) % 25 == 0 or index + 1 == len(pairs)):
            elapsed = time.monotonic() - started
            progress(f"historical returns asof={as_of_date} pairs={index+1}/{len(pairs)} "
                     f"elapsed_s={elapsed:.1f} remaining_s~{elapsed*(len(pairs)-index-1)/(index+1):.0f}")
    raw_cash, claim_gains, starting_prices = np.asarray(instruments), np.asarray(targets), np.asarray(starting_prices)
    if return_basis == "premium" and (not np.isfinite(starting_prices).all() or np.any(starting_prices <= 0.)):
        raise ValueError("Historical premium returns need positive starting prices; no premium floor is substituted")
    raw = raw_cash / starting_prices if return_basis == "premium" else raw_cash.copy()
    weights = np.full(len(pairs), 1. / len(pairs))
    removed = weights @ raw
    centered = raw - removed
    if np.max(np.abs(weights @ centered), initial=0.) > 1e-10:
        raise RuntimeError("Historical weighted asset demeaning failed")
    old_spots = np.asarray(old_spots)
    return HistoricalReturns(instrument_gains=centered, raw_instrument_gains=raw,
        claim_gains=claim_gains, removed_instrument_mean=removed, weights=weights,
        pair_start_dates=np.array([before.isoformat() for before, _ in pairs]),
        pair_end_dates=np.array([after.isoformat() for _, after in pairs]),
        mapped_cash_strikes=old_spots[:, None] / current_spot * book["strike"].to_numpy(dtype=float),
        mapped_target_strikes=old_spots[:, None] / current_spot * np.array([claim.strike for claim in claims]),
        source_hashes=np.asarray(hashes), curve_hashes=np.asarray(curve_hashes), carry_inputs=np.asarray(carry),
        source_first_business_dtes=np.asarray(first_tenors),
        short_expiry_extrapolation_counts=np.asarray(extrapolated), outside_source_strike_counts=np.asarray(outside),
        starting_instrument_prices=starting_prices, raw_cash_gains=raw_cash, return_basis=return_basis,
        return_convention="one-day entry-premium returns" if return_basis == "premium" else "one-day fixed-instrument P&L, not premium returns")
