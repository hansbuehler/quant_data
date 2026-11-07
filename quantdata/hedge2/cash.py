"""Cash-contract curves, licensed hedge books and fixed-numeraire daily gains."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS, count_business_days_between
from quantdata.hedge2.pricing import DlvPricer

CONTRACT_KEY = ["expiry", "strike", "option_type", "is_am"]


@dataclass(frozen=True, kw_only=True)
class CashCurve:
    spot: float
    business_dtes: np.ndarray
    forwards: np.ndarray
    reference_forwards: np.ndarray
    discounts: np.ndarray

    def __post_init__(self) -> None:
        arrays = (self.business_dtes, self.forwards, self.reference_forwards, self.discounts)
        if (not np.isfinite(self.spot) or self.spot <= 0 or len(self.business_dtes) == 0
                or any(array.shape != self.business_dtes.shape for array in arrays)
                or any(not np.isfinite(array).all() or np.any(array <= 0) for array in arrays)
                or not np.all(np.diff(self.business_dtes) > 0)):
            raise ValueError("Cash curves require finite positive, aligned and ordered inputs")

    def at(self, business_dte: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        days = np.asarray(business_dte, dtype=float)
        if not np.isfinite(days).all() or np.any(days < 0) or np.any(days > self.business_dtes[-1] + 1e-6):
            raise ValueError("Requested business DTE lies outside the observed carry curve")
        nodes = np.r_[0., self.business_dtes]
        forward = self.spot * np.exp(np.interp(days, nodes, np.r_[0., np.log(self.forwards / self.spot)]))
        reference = self.spot * np.exp(np.interp(days, nodes, np.r_[0., np.log(self.reference_forwards / self.spot)]))
        discount = np.exp(np.interp(days, nodes, np.r_[0., np.log(self.discounts)]))
        return forward, reference, discount

    def at_spot(self, spot: float) -> CashCurve:
        ratio = spot / self.spot
        return CashCurve(spot=spot, business_dtes=self.business_dtes,
                         forwards=ratio * self.forwards,
                         reference_forwards=ratio * self.reference_forwards, discounts=self.discounts)

    def roll(self, business_days: float, *, new_spot: float) -> CashCurve:
        """Causal deterministic-carry roll; source and reference forwards stay distinct."""
        if not np.isfinite(business_days) or business_days <= 0:
            raise ValueError("Carry roll must advance a positive number of business days")
        forward, reference, discount = self.at(business_days)
        keep = self.business_dtes > business_days + 1e-6
        return CashCurve(spot=new_spot, business_dtes=self.business_dtes[keep] - business_days,
                         forwards=new_spot * self.forwards[keep] / forward,
                         reference_forwards=new_spot * self.reference_forwards[keep] / reference,
                         discounts=self.discounts[keep] / discount)


@dataclass(frozen=True, kw_only=True)
class MarketDay:
    day: date
    quotes: pd.DataFrame
    curve: CashCurve


@dataclass(frozen=True, kw_only=True)
class MarketStore:
    market_directory: Path
    licensed_fit_directory: Path
    raw_quote_directory: Path | None = None

    @lru_cache(maxsize=8)
    def load(self, day: date) -> MarketDay:
        quotes = pd.read_parquet(self.market_directory / "days" / f"{day.isoformat()}.parquet")
        source = pd.read_parquet(self.licensed_fit_directory / "SPX" / f"{day.isoformat()}.pdq",
                                 columns=["date", "expiry", "strike", "is_am", "spot", "bdte", "fwd", "fwd_ref", "df"])
        if (not (quotes["date"] == pd.Timestamp(day)).all()
                or not (source["date"] == pd.Timestamp(day)).all()
                or source["spot"].nunique() != 1):
            raise ValueError(f"Market date or spot mismatch on {day}")
        reference = source[["expiry", "strike", "is_am", "fwd_ref"]]
        if reference.duplicated(["expiry", "strike", "is_am"]).any():
            raise ValueError(f"Duplicate source contracts on {day}")
        quotes = quotes.merge(reference, on=["expiry", "strike", "is_am"], how="left", validate="many_to_one")
        if not np.isfinite(quotes["fwd_ref"]).all():
            raise ValueError(f"Missing reference forwards on {day}")
        pm = source.loc[~source["is_am"]]
        grouped = pm.groupby("bdte", sort=True)
        if (grouped[["fwd", "fwd_ref", "df"]].nunique() > 1).any().any():
            raise ValueError(f"Inconsistent carry by PM expiry on {day}")
        curves = grouped[["fwd", "fwd_ref", "df"]].first()
        curve = CashCurve(spot=float(source["spot"].iloc[0]), business_dtes=curves.index.to_numpy(dtype=float),
                          forwards=curves["fwd"].to_numpy(dtype=float),
                          reference_forwards=curves["fwd_ref"].to_numpy(dtype=float),
                          discounts=curves["df"].to_numpy(dtype=float))
        quotes["observed_quote_origin"] = "licensed_fitted_input"
        if self.raw_quote_directory is not None:
            raw = pd.read_parquet(self.raw_quote_directory / "SPX" / f"{day.isoformat()}.pdq")
            if not (raw["date"] == pd.Timestamp(day)).all() or not np.allclose(raw["spot"], curve.spot, rtol=0., atol=1e-6):
                raise ValueError(f"Upstream licensed quote dates/spots disagree on {day}")
            raw = raw.loc[(~raw["is_am"]) & (raw["bdte"] > 0.) & (raw["bdte"] <= 25. + 1e-6)]
            additions = []
            for option_type in ("call", "put"):
                frame = raw[["date", "expiry", "strike", "is_am", "bdte", "fwd_ref", "df"]].copy()
                frame["option_type"] = option_type
                frame["observed_bid"] = raw[f"mkt_{option_type}_bid"].to_numpy(dtype=float)
                frame["observed_ask"] = raw[f"mkt_{option_type}_ask"].to_numpy(dtype=float)
                valid = (np.isfinite(frame["observed_bid"]) & np.isfinite(frame["observed_ask"])
                         & (frame["observed_bid"] >= 0.) & (frame["observed_ask"] > frame["observed_bid"]))
                frame = frame.loc[valid]
                frame["observed_mid"] = .5 * (frame["observed_bid"] + frame["observed_ask"])
                frame["observed_half_spread"] = .5 * (frame["observed_ask"] - frame["observed_bid"])
                frame["fwd"] = curve.at(frame["bdte"].to_numpy(dtype=float))[0]
                frame["observed_quote_origin"] = "licensed_raw_day"
                additions.append(frame)
            extra = pd.concat(additions, ignore_index=True)
            if extra.duplicated(CONTRACT_KEY).any():
                raise ValueError(f"Upstream licensed quotes have duplicate contracts on {day}")
            present = pd.MultiIndex.from_frame(quotes[CONTRACT_KEY])
            extra = extra.loc[~pd.MultiIndex.from_frame(extra[CONTRACT_KEY]).isin(present)]
            quotes = pd.concat([quotes, extra], ignore_index=True).sort_values(CONTRACT_KEY).reset_index(drop=True)
        return MarketDay(day=day, quotes=quotes, curve=curve)


def select_book(market: MarketDay, *, atm_vols: np.ndarray,
                co_terminal_business_dtes: tuple[int, ...] | None = None,
                option_side: str = "otm", include_co_terminal_forwards: bool = False) -> pd.DataFrame:
    """Select one PM side per strike; optional zero-strike calls are explicit synthetic assets."""
    if option_side not in ("otm", "itm"):
        raise ValueError("Choose the ITM or OTM side of the observed option book")
    frame = market.quotes.loc[~market.quotes["is_am"]].copy()
    days = frame["bdte"].to_numpy(dtype=float)
    if atm_vols.shape != (len(market.quotes),) or not np.isfinite(atm_vols).all() or np.any(atm_vols <= 0):
        raise ValueError("ATM volatilities must match the complete observed quote frame")
    vols = atm_vols[~market.quotes["is_am"].to_numpy()]
    frame["hedge_nms"] = NMS.to_nms(frame["strike"].to_numpy() / frame["fwd"].to_numpy(),
                                    days / DAYS_PER_YEAR, vol=vols, eps_vol=vols)
    if co_terminal_business_dtes is None:
        later = days[days >= 5. - 1e-6]
        if len(later) == 0:
            raise ValueError("No PM expiry at or after five business days")
        eligible = days <= float(later.min()) + 1e-6
    else:
        eligible = np.any(np.isclose(days[:, None], np.asarray(co_terminal_business_dtes)[None, :],
                                     atol=1e-6, rtol=0.), axis=1)
        for tenor in co_terminal_business_dtes:
            if not np.any(np.isclose(days, tenor, atol=1e-6, rtol=0.)):
                raise ValueError(f"No PM contracts at required co-terminal expiry {tenor}D")
    choose_call = (frame["strike"].to_numpy() >= frame["fwd"].to_numpy() if option_side == "otm"
                   else frame["strike"].to_numpy() <= frame["fwd"].to_numpy())
    selected_side = (frame["option_type"].to_numpy() == "call") == choose_call
    inside = (frame["hedge_nms"].to_numpy() > -2.) & (frame["hedge_nms"].to_numpy() < 2.)
    frame = frame.loc[eligible & selected_side & inside].sort_values(["expiry", "strike"]).reset_index(drop=True)
    if frame.empty or frame.duplicated(CONTRACT_KEY).any():
        raise ValueError("The eligible hedge book is empty or contains duplicate contracts")
    return add_zero_strike_calls(frame, market=market) if include_co_terminal_forwards else frame


def zero_strike_mask(book: pd.DataFrame) -> np.ndarray:
    """A zero-strike asset must be declared; never invent a missing ordinary quote."""
    synthetic = (book["instrument_kind"].to_numpy() == "zero_strike_call"
                 if "instrument_kind" in book else np.zeros(len(book), dtype=bool))
    zero = book["strike"].to_numpy(dtype=float) == 0.
    if np.any(synthetic != zero) or np.any(synthetic & (book["option_type"].to_numpy() != "call")):
        raise ValueError("Zero-strike calls require an explicit synthetic instrument identity")
    return synthetic


def add_zero_strike_calls(book: pd.DataFrame, *, market: MarketDay) -> pd.DataFrame:
    """One cost-free prepaid forward per expiry, valued at D(T)*F(T), settling to spot."""
    if book.empty or book["is_am"].any() or np.any(book["strike"].to_numpy(dtype=float) <= 0):
        raise ValueError("Add prepaid forwards to a nonempty ordinary PM option book only")
    regular = book.copy()
    regular["instrument_kind"] = "european"
    synthetic = regular.groupby("expiry", sort=True, as_index=False).first()
    forward, reference, discount = market.curve.at(synthetic["bdte"].to_numpy(dtype=float))
    values = forward * discount
    synthetic["instrument_kind"] = "zero_strike_call"
    synthetic["strike"], synthetic["option_type"] = 0., "call"
    synthetic["fwd"], synthetic["fwd_ref"], synthetic["df"] = forward, reference, discount
    for name in ("observed_mid", "observed_bid", "observed_ask"):
        synthetic[name] = values
    synthetic["observed_half_spread"] = 0.
    synthetic["observed_quote_origin"] = "synthetic_prepaid_forward_at_mid"
    for name in ("fit_price", "fit_vol", "hedge_nms", "nms"):
        if name in synthetic:
            synthetic[name] = np.nan
    output = pd.concat([regular, synthetic], ignore_index=True).sort_values(["expiry", "strike"]).reset_index(drop=True)
    if output.duplicated(CONTRACT_KEY).any():
        raise ValueError("Synthetic forwards created duplicate contract keys")
    return output


def observed_marks(book: pd.DataFrame, market: MarketDay) -> tuple[np.ndarray, np.ndarray]:
    """Mark fixed PM contracts at observed mids, or cash settlement with no exit spread."""
    if book["is_am"].any():
        raise ValueError("Closing-spot settlement cannot be used for AM contracts")
    expiry = pd.to_datetime(book["expiry"]).dt.date
    if (expiry < market.day).any():
        raise ValueError("Expired contracts must be removed before a later mark")
    settled = (expiry == market.day).to_numpy()
    synthetic = zero_strike_mask(book)
    prices = np.empty(len(book))
    spreads = np.zeros(len(book))
    difference = market.curve.spot - book["strike"].to_numpy(dtype=float)
    prices[settled] = np.maximum(np.where(book["option_type"] == "call", difference, -difference), 0.)[settled]
    synthetic_live = synthetic & ~settled
    if synthetic_live.any():
        days = count_business_days_between(market.day, expiry[synthetic_live].to_numpy(dtype="datetime64[D]"))
        forward, _, discount = market.curve.at(days.astype(float))
        prices[synthetic_live] = forward * discount
    quoted_live = ~settled & ~synthetic
    live = book.loc[quoted_live, CONTRACT_KEY].copy()
    live["_row"] = np.flatnonzero(quoted_live)
    joined = live.merge(market.quotes[CONTRACT_KEY + ["observed_mid", "observed_half_spread"]],
                        on=CONTRACT_KEY, how="left", validate="one_to_one")
    if joined[["observed_mid", "observed_half_spread"]].isna().any().any():
        missing = joined.loc[joined["observed_mid"].isna(), CONTRACT_KEY].to_dict("records")
        raise ValueError(f"Missing observed mids on {market.day}: {missing[:5]}")
    prices[joined["_row"].to_numpy(dtype=int)] = joined["observed_mid"]
    spreads[joined["_row"].to_numpy(dtype=int)] = joined["observed_half_spread"]
    return prices, spreads


def model_europeans(pricer: DlvPricer, curve: CashCurve, *, business_dtes: np.ndarray,
                    cash_strikes: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Cash European prices use the same source forward in moneyness and prefactor."""
    days, strikes, calls = np.broadcast_arrays(business_dtes, cash_strikes, is_call)
    if days.ndim != 1 or not np.isfinite(strikes).all() or np.any(strikes < 0) or np.any((strikes == 0) & ~calls):
        raise ValueError("European strikes must be positive, except for zero-strike calls")
    output = np.empty(len(days))
    for tenor in np.unique(days):
        selected = days == tenor
        forward, _, discount = curve.at(float(tenor))
        output[selected & (strikes == 0.)] = discount * forward
        selected &= strikes > 0.
        if not selected.any():
            continue
        output[selected] = discount * forward * pricer.european_at(
            expiry_y=float(tenor) / DAYS_PER_YEAR, strikes=strikes[selected] / forward,
            is_call=calls[selected])
    return output


def discounted_daily_gains(*, previous_option_marks: np.ndarray, next_option_marks: np.ndarray,
                           next_spot: float, previous_curve: CashCurve,
                           episode_initial_spot: float, discount_to_previous: float = 1.) -> np.ndarray:
    """Forward-financed spot and cash option gains in one episode's initial units."""
    if previous_option_marks.shape != next_option_marks.shape:
        raise ValueError("Daily option marks must refer to the same fixed contracts")
    if episode_initial_spot <= 0 or discount_to_previous <= 0 or next_spot <= 0:
        raise ValueError("Spot normalization and discount factors must be positive")
    _, reference, one_day_discount = previous_curve.at(1.)
    option_gains = discount_to_previous * (one_day_discount * next_option_marks - previous_option_marks)
    spot_gain = discount_to_previous * one_day_discount * (next_spot - reference)
    return np.r_[spot_gain, option_gains] / episode_initial_spot
