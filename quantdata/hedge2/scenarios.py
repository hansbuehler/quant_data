"""Nonlinear one-session repricing under the fitted empirical PCA/AR(1) law."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantdata.hedge2.cash import CashCurve, model_europeans
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.state import DailyState


@dataclass(frozen=True, kw_only=True)
class ScenarioSet:
    instrument_gains: np.ndarray
    claim_gains: np.ndarray
    weights: np.ndarray
    removed_instrument_mean: np.ndarray
    draw_count: int
    distinct_innovations: int
    lower_clips: np.ndarray
    upper_clips: np.ndarray
    effective_innovation_sample_size: float
    innovation_indices: np.ndarray
    innovation_frequencies: np.ndarray


def generate_scenarios(*, state: DailyState, dynamics: FittedDynamics,
                       curve: CashCurve, book: pd.DataFrame, claims: tuple[Claim, ...],
                       initial_spot: float, elapsed_day: int = 0,
                       draw_count: int | None = None, seed: int = 1) -> ScenarioSet:
    """Reuse identical simulated instruments for every target and optimization objective.

    Bootstrap draws of the paired historical residual rows are compressed
    to distinct rows with their exact empirical frequencies. Thus 10,000
    draws do not manufacture 10,000 independent historical observations.
    The full spot loading is included once for spot-orthogonal dynamics.
    Asset gains are demeaned with the same weights used by the optimizer.
    """
    count = max(10000, 10 * (1 + len(book))) if draw_count is None else draw_count
    if count <= 0 or initial_spot <= 0:
        raise ValueError("Scenario count and episode scale must be positive")
    if state.day != dynamics.as_of_date or len(state.native) != dynamics.directions.shape[0]:
        raise ValueError("Scenario model and pricing state must have the same date and geometry")
    rng = np.random.default_rng(seed)
    indices, frequencies = np.unique(rng.integers(0, len(dynamics.residuals), size=count), return_counts=True)
    weights = frequencies.astype(float) / count
    residuals = dynamics.residuals[indices]
    forecast = state.native + dynamics.forecast_state_change(state.native)
    current = state.decode().pricer
    tenors = book["bdte"].to_numpy(dtype=float)
    strikes = book["strike"].to_numpy(dtype=float)
    calls = (book["option_type"] == "call").to_numpy()
    previous_options = model_europeans(current, curve, business_dtes=tenors, cash_strikes=strikes, is_call=calls)
    previous_claims = np.array([claim.price(current, curve, elapsed_day=elapsed_day) for claim in claims])
    _, reference, discount = curve.at(1.)
    remaining = tenors - 1.
    if np.any(remaining < -1e-6):
        raise ValueError("Expired hedge contracts cannot enter a new scenario step")
    settled = np.abs(remaining) <= 1e-6
    instrument_gains = np.empty((len(indices), 1 + len(book)))
    claim_gains = np.empty((len(indices), len(claims)))
    lower_clips, upper_clips = [], []
    for index, residual in enumerate(residuals):
        spot_innovation = float(residual[0])
        if not np.isfinite(spot_innovation) or spot_innovation <= -1.:
            raise ValueError(f"Empirical innovation {indices[index]} implies nonpositive next spot")
        next_spot = float(reference * (1. + spot_innovation))
        native = forecast + dynamics.spot_state_loading * spot_innovation + dynamics.directions @ residual[1:]
        decoded = state.decode(native)
        next_curve = curve.roll(1., new_spot=next_spot)
        option_values = np.empty(len(book))
        differences = next_spot - strikes[settled]
        option_values[settled] = np.maximum(np.where(calls[settled], differences, -differences), 0.)
        if (~settled).any():
            option_values[~settled] = model_europeans(decoded.pricer, next_curve,
                business_dtes=remaining[~settled], cash_strikes=strikes[~settled], is_call=calls[~settled])
        next_claims = np.array([claim.observe(elapsed_day=elapsed_day + 1, spot=next_spot).price(
            decoded.pricer, next_curve, elapsed_day=elapsed_day + 1) for claim in claims])
        instrument_gains[index, 0] = discount * (next_spot - reference) / initial_spot
        instrument_gains[index, 1:] = (discount * option_values - previous_options) / initial_spot
        claim_gains[index] = discount * next_claims - previous_claims
        lower_clips.append(decoded.lower_clips)
        upper_clips.append(decoded.upper_clips)
    removed_mean = weights @ instrument_gains
    instrument_gains -= removed_mean
    if (not np.isfinite(instrument_gains).all() or not np.isfinite(claim_gains).all()
            or np.max(np.abs(weights @ instrument_gains), initial=0.) > 1e-10):
        raise ValueError("Scenario gains are non-finite or weighted drift removal failed")
    return ScenarioSet(instrument_gains=instrument_gains, claim_gains=claim_gains,
                        weights=weights, removed_instrument_mean=removed_mean,
                        draw_count=count, distinct_innovations=len(indices),
                        lower_clips=np.asarray(lower_clips), upper_clips=np.asarray(upper_clips),
                        effective_innovation_sample_size=float(1. / (weights @ weights)),
                        innovation_indices=indices, innovation_frequencies=frequencies)
