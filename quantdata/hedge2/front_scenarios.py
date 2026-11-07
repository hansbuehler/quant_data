"""Opt-in calibrated-front one-day scenarios with an explicit surface-shock law."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.hedge2.batch_pricing import decode_batch
from quantdata.hedge2.cash import CashCurve, model_europeans
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.front_joint import EmpiricalFrontCopula, GaussianFrontCopula
from quantdata.hedge2.front_spot import DiscreteFrontLaw
from quantdata.hedge2.scenarios import ScenarioSet
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal_market import cash_europeans_batch


@dataclass(frozen=True, kw_only=True)
class FrontScenarioSet(ScenarioSet):
    spot_law: str
    spot_law_identity: str
    forecast_identity: str
    diagnostic_only: bool
    forecast_scope: str
    next_spots: np.ndarray
    physical_spot_returns: np.ndarray
    factor_residuals: np.ndarray
    historical_residual_indices: np.ndarray
    population_innovation_covariance: np.ndarray
    front_mean: float
    front_variance: float
    front_relative_mean_residual: float


def _forecast_identity(*, state: DailyState, dynamics: FittedDynamics,
                       coupling: EmpiricalFrontCopula | GaussianFrontCopula,
                       curve: CashCurve, book: pd.DataFrame, claims: tuple[Claim, ...],
                       initial_spot: float, current_claim_marks: np.ndarray,
                       elapsed_day: int, draw_count: int, seed: int, diagnostic_only: bool) -> str:
    """Bind the numerical forecast to its complete dated inputs and local implementation."""
    digest = hashlib.sha256(b"hedge2.front_scenarios.v1")

    def add(value) -> None:
        if is_dataclass(value):
            add(asdict(value))
        elif isinstance(value, np.ndarray):
            add((str(value.dtype), value.shape))
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(key)
                add(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(f"items:{len(value)}".encode())
            for item in value:
                add(item)
        elif isinstance(value, date):
            add(value.isoformat())
        elif isinstance(value, np.generic):
            add(value.item())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        digest.update(b"\0")

    add((state, dynamics, curve, claims, initial_spot, current_claim_marks,
        elapsed_day, draw_count, seed, diagnostic_only, coupling.identity))
    add((tuple(book.columns), tuple(str(value) for value in book.dtypes)))
    add(pd.util.hash_pandas_object(book, index=True).to_numpy())
    names = ("front_scenarios.py", "front_joint.py", "front_spot.py", "front_scores.py", "batch_pricing.py",
        "cash.py", "claims.py", "dynamics.py", "pricing.py", "state.py", "terminal_market.py", "terminal.py",
        "dlv.py", "source.py", "objectives.py", "historical.py")
    for name in names:
        add(name)
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def generate_front_scenarios(*, state: DailyState, dynamics: FittedDynamics,
                             coupling: EmpiricalFrontCopula | GaussianFrontCopula,
                             curve: CashCurve, book: pd.DataFrame, claims: tuple[Claim, ...],
                             initial_spot: float, current_claim_marks: np.ndarray, elapsed_day: int = 0,
                             draw_count: int | None = None, seed: int = 1,
                             pricing_batch_size: int = 256, diagnostic_only: bool = False) -> FrontScenarioSet:
    """Sample current front spot and explicit joint innovations, then remove asset alpha.

    Empirical draws compress only identical (front atom, historical residual)
    pairs, retaining their actual Monte Carlo frequencies. Gaussian factors
    remain continuous and their paths are not replaced by cubature branches.
    Population front calibration is exact; finite-sample moments retain normal
    Monte Carlo error. Claim gains are not demeaned. Raw gains are recoverable
    by adding the recorded removed mean, with one fixed inception numeraire.
    """
    minimum_count = max(10000, 10 * (1 + len(book)))
    count = minimum_count if draw_count is None else draw_count
    previous_claims = np.asarray(current_claim_marks, dtype=float)
    if (not np.isfinite(initial_spot) or initial_spot <= 0. or not claims
            or isinstance(pricing_batch_size, bool) or not isinstance(pricing_batch_size, int)
            or pricing_batch_size < 1 or any(claim.terminal_day <= elapsed_day for claim in claims)):
        raise ValueError("Front scenarios require live claims, positive inception spot and integer batch size")
    if previous_claims.shape != (len(claims),) or not np.isfinite(previous_claims).all() or np.any(previous_claims < 0.):
        raise ValueError("Front scenarios require aligned finite nonnegative common-market target marks")
    if count < minimum_count and not diagnostic_only:
        raise ValueError("Production front scenarios require max(10000, 10 * instruments) draws; use diagnostic_only for a small test")
    if state.day != dynamics.as_of_date or len(state.native) != dynamics.directions.shape[0]:
        raise ValueError("Front scenario model and pricing state must have the same date and geometry")
    coupling.validate_model(dynamics)
    current = state.decode().pricer
    parity, reference, discount = (float(value) for value in curve.at(1.))
    front = DiscreteFrontLaw.from_pricer(pricer=current, parity_forward=parity)
    draws = coupling.sample(front=front, reference_forward=reference, count=count, rng=np.random.default_rng(seed))
    if draws.historical_indices is not None:
        atoms = np.searchsorted(front.cash_support, draws.next_spots)
        indices, selected, frequencies = np.unique(np.c_[atoms, draws.historical_indices],
            axis=0, return_index=True, return_counts=True)
        historical_indices = draws.historical_indices[selected]
    else:
        indices = selected = np.arange(count)
        frequencies = np.ones(count, dtype=np.int64)
        historical_indices = np.empty(0, dtype=np.int64)
    spots, physical, factors = (values[selected] for values in
        (draws.next_spots, draws.physical_returns, draws.factor_residuals))
    weights = frequencies.astype(float) / count
    tenors = book["bdte"].to_numpy(dtype=float)
    strikes = book["strike"].to_numpy(dtype=float)
    calls = (book["option_type"] == "call").to_numpy()
    remaining = tenors - 1.
    if np.any(remaining < -1e-6):
        raise ValueError("Expired hedge contracts cannot enter a front scenario step")
    previous_options = model_europeans(current, curve, business_dtes=tenors, cash_strikes=strikes, is_call=calls)
    next_template = curve.roll(1., new_spot=curve.spot)
    maximum_remaining = max(float(np.max(remaining, initial=0.)),
        max(claim.terminal_day - elapsed_day - 1 for claim in claims))
    forecast = state.native + dynamics.forecast_state_change(state.native)
    instrument_gains = np.empty((len(spots), 1 + len(book)))
    claim_gains = np.empty((len(spots), len(claims)))
    lower_clips = np.empty(len(spots), dtype=np.int64)
    upper_clips = np.empty(len(spots), dtype=np.int64)
    for begin in range(0, len(spots), pricing_batch_size):
        end = min(begin + pricing_batch_size, len(spots))
        next_spots = spots[begin:end]
        native = (forecast + physical[begin:end, None] * dynamics.spot_state_loading
                  + factors[begin:end] @ dynamics.directions.T)
        decoded = decode_batch(state, native, max_business_dte=maximum_remaining)
        option_values = cash_europeans_batch(decoded=decoded, template=next_template, spots=next_spots,
            business_dtes=remaining, cash_strikes=strikes, is_call=calls)
        instrument_gains[begin:end, 0] = discount * (next_spots - reference) / initial_spot
        instrument_gains[begin:end, 1:] = (discount * option_values - previous_options) / initial_spot
        lower_clips[begin:end], upper_clips[begin:end] = decoded.lower_clips, decoded.upper_clips
        for target, claim in enumerate(claims):
            tenor = claim.terminal_day - elapsed_day - 1
            if claim.kind == "digital":
                if tenor == 0:
                    values = next_spots > claim.strike if claim.is_call else next_spots <= claim.strike
                else:
                    forward, _, future_discount = next_template.at(float(tenor))
                    normalized = claim.strike / (next_spots[:, None] / next_template.spot * forward)
                    values = future_discount * decoded.europeans(business_dtes=np.array([float(tenor)]),
                        strikes=normalized, is_call=claim.is_call, digital=True)[:, 0]
            else:
                values = np.array([claim.observe(elapsed_day=elapsed_day + 1, spot=float(spot)).price(
                    state.decode(native[index]).pricer, curve.roll(1., new_spot=float(spot)),
                    elapsed_day=elapsed_day + 1) for index, spot in enumerate(next_spots)])
            claim_gains[begin:end, target] = discount * values - previous_claims[target]
    removed = weights @ instrument_gains
    instrument_gains -= removed
    if (not np.isfinite(instrument_gains).all() or not np.isfinite(claim_gains).all()
            or np.max(np.abs(weights @ instrument_gains), initial=0.) > 1e-10):
        raise ValueError("Front scenario gains are nonfinite or weighted asset-alpha removal failed")
    return FrontScenarioSet(instrument_gains=instrument_gains, claim_gains=claim_gains, weights=weights,
        removed_instrument_mean=removed, draw_count=count, distinct_innovations=len(spots),
        lower_clips=lower_clips, upper_clips=upper_clips,
        effective_innovation_sample_size=float(1. / (weights @ weights)),
        innovation_indices=indices, innovation_frequencies=frequencies,
        spot_law=coupling.law_kind, spot_law_identity=coupling.identity,
        forecast_identity=_forecast_identity(state=state, dynamics=dynamics, coupling=coupling,
            curve=curve, book=book, claims=claims, initial_spot=initial_spot,
            current_claim_marks=previous_claims, elapsed_day=elapsed_day, draw_count=count, seed=seed,
            diagnostic_only=diagnostic_only),
        diagnostic_only=diagnostic_only, forecast_scope="next-day marked risk; not multi-day terminal optimization",
        next_spots=spots,
        physical_spot_returns=physical, factor_residuals=factors,
        historical_residual_indices=historical_indices,
        population_innovation_covariance=coupling.innovation_covariance(front=front, reference_forward=reference),
        front_mean=front.mean, front_variance=front.variance, front_relative_mean_residual=(front.mean - parity) / parity)
