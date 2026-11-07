"""Separate fixed-state spot and rolling-PCA finite-shock hedge exposures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantdata.hedge2.cash import CashCurve, model_europeans
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.state import DailyState

if TYPE_CHECKING:
    from quantdata.hedge2.front_joint import EmpiricalFrontCopula, GaussianFrontCopula
    from quantdata.hedge2.front_spot import DiscreteFrontLaw


@dataclass(frozen=True, kw_only=True)
class GreekResult:
    instrument_prices: np.ndarray
    claim_prices: np.ndarray
    instrument_exposures: np.ndarray
    claim_exposures: np.ndarray
    pc_lower_clips: np.ndarray
    pc_upper_clips: np.ndarray
    spot_bump_fraction: float
    spot_is_state_conditioned: bool = False
    spot_loading_lower_clips: int = 0
    spot_loading_upper_clips: int = 0
    combined_spot_instrument_exposures: np.ndarray | None = None
    combined_spot_claim_exposures: np.ndarray | None = None


def _price_vector(*, state: DailyState, native: np.ndarray, curve: CashCurve,
                  book: pd.DataFrame, claims: tuple[Claim, ...], initial_spot: float,
                  elapsed_day: int) -> tuple[np.ndarray, int, int]:
    decoded = state.decode(native)
    options = model_europeans(decoded.pricer, curve,
                              business_dtes=book["bdte"].to_numpy(dtype=float),
                              cash_strikes=book["strike"].to_numpy(dtype=float),
                              is_call=(book["option_type"] == "call").to_numpy())
    targets = np.array([claim.price(decoded.pricer, curve, elapsed_day=elapsed_day) for claim in claims])
    values = np.r_[curve.spot / initial_spot, options / initial_spot, targets]
    if not np.isfinite(values).all():
        raise ValueError("Greek repricing produced a non-finite cash-contract value")
    return values, decoded.lower_clips, decoded.upper_clips


def calculate_greeks(*, state: DailyState, dynamics: FittedDynamics,
                     curve: CashCurve, book: pd.DataFrame, claims: tuple[Claim, ...],
                     initial_spot: float, elapsed_day: int = 0,
                     spot_bump_fraction: float = .01) -> GreekResult:
    """PC columns are central differences at +/- one daily PCA standard deviation.

    The spot column divides by the relative spot bump, holding native u and
    cash contracts fixed. All prices are in the episode's initial spot units.
    Atomic DLV digitals have finite-shock exposures, not smooth derivatives.
    """
    if not 0 < spot_bump_fraction < 1 or initial_spot <= 0:
        raise ValueError("Spot bump must be in (0,1) and episode scale positive")
    if dynamics.as_of_date != state.day or dynamics.directions.shape[0] != len(state.native):
        raise ValueError("Daily pricing state and rolling dynamics must refer to the same date and geometry")
    instrument_count = 1 + len(book)

    def prices(native: np.ndarray, shocked_curve: CashCurve) -> tuple[np.ndarray, int, int]:
        return _price_vector(state=state, native=native, curve=shocked_curve, book=book,
                              claims=claims, initial_spot=initial_spot, elapsed_day=elapsed_day)

    baseline = prices(state.native, curve)[0]
    exposures = np.empty((len(baseline), 1 + dynamics.directions.shape[1]))
    positive = prices(state.native, curve.at_spot(curve.spot * (1. + spot_bump_fraction)))[0]
    negative = prices(state.native, curve.at_spot(curve.spot * (1. - spot_bump_fraction)))[0]
    exposures[:, 0] = (positive - negative) / (2. * spot_bump_fraction)
    lower_clips, upper_clips = [], []
    for factor, scale in enumerate(dynamics.daily_factor_scales):
        shock = dynamics.directions[:, factor] * scale
        positive, lower_positive, upper_positive = prices(state.native + shock, curve)
        negative, lower_negative, upper_negative = prices(state.native - shock, curve)
        exposures[:, factor + 1] = .5 * (positive - negative)
        lower_clips.append(lower_positive + lower_negative)
        upper_clips.append(upper_positive + upper_negative)
    return GreekResult(instrument_prices=baseline[:instrument_count], claim_prices=baseline[instrument_count:],
                        instrument_exposures=exposures[:instrument_count], claim_exposures=exposures[instrument_count:],
                        pc_lower_clips=np.asarray(lower_clips), pc_upper_clips=np.asarray(upper_clips),
                        spot_bump_fraction=spot_bump_fraction)


def factor_risk_covariance(model: FittedDynamics, *, spot_orthogonal: bool) -> np.ndarray:
    """Covariance of spot and raw score innovations in the finite-shock exposure units."""
    residuals = model.residuals.copy()
    if spot_orthogonal:
        beta = model.spot_state_loading @ model.directions
        residuals[:, 1:] += residuals[:, :1] * beta
    residuals[:, 1:] /= model.daily_factor_scales
    return residuals.T @ residuals / len(residuals)


def calculate_parametric_greeks(*, state: DailyState, dynamics: FittedDynamics,
                               curve: CashCurve, book: pd.DataFrame, claims: tuple[Claim, ...],
                               initial_spot: float, elapsed_day: int = 0,
                               spot_bump_fraction: float = .01) -> GreekResult:
    """Add the separately repriced full beta*spot surface response to the fixed-u delta.

    The PCA partials remain unchanged. With orthogonal PCA, this represents
    the conditional innovation axes (spot, factor residuals), including beta
    components outside the retained PCA span, rather than projecting beta.
    """
    base = calculate_greeks(state=state, dynamics=dynamics, curve=curve, book=book, claims=claims,
                            initial_spot=initial_spot, elapsed_day=elapsed_day, spot_bump_fraction=spot_bump_fraction)
    if not np.any(dynamics.spot_state_loading):
        return base
    arguments = dict(state=state, curve=curve, book=book, claims=claims,
                     initial_spot=initial_spot, elapsed_day=elapsed_day)
    shock = spot_bump_fraction * dynamics.spot_state_loading
    positive, lower_positive, upper_positive = _price_vector(native=state.native + shock, **arguments)
    negative, lower_negative, upper_negative = _price_vector(native=state.native - shock, **arguments)
    response = (positive - negative) / (2. * spot_bump_fraction)
    instrument_exposures = base.instrument_exposures.copy()
    claim_exposures = base.claim_exposures.copy()
    count = len(instrument_exposures)
    instrument_exposures[:, 0] += response[:count]
    claim_exposures[:, 0] += response[count:]
    positive = _price_vector(native=state.native + shock, **(arguments | {
        "curve": curve.at_spot(curve.spot * (1. + spot_bump_fraction))}))[0]
    negative = _price_vector(native=state.native - shock, **(arguments | {
        "curve": curve.at_spot(curve.spot * (1. - spot_bump_fraction))}))[0]
    combined = (positive - negative) / (2. * spot_bump_fraction)
    return replace(base, instrument_exposures=instrument_exposures, claim_exposures=claim_exposures,
                    spot_is_state_conditioned=True, spot_loading_lower_clips=lower_positive + lower_negative,
                    spot_loading_upper_clips=upper_positive + upper_negative,
                    combined_spot_instrument_exposures=combined[:count],
                    combined_spot_claim_exposures=combined[count:])


def parametric_factor_covariance(model: FittedDynamics, *,
                                front_coupling: EmpiricalFrontCopula | GaussianFrontCopula | None = None,
                                front: DiscreteFrontLaw | None = None,
                                reference_forward: float | None = None) -> np.ndarray:
    """Covariance in fixed-u delta and unit daily-PC-shock exposure units.

    Without an explicit front coupling this retains the legacy historical
    covariance. A supplied coupling calibrates physical spot variance and
    spot/factor covariance to the current front law before scaling factors.
    Using this matrix in a Gaussian risk optimizer is still an approximation
    to the non-Gaussian quantile-transformed physical-return distribution.
    """
    supplied = (front_coupling is not None, front is not None, reference_forward is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("Front covariance needs its coupling, current front law and financing forward together")
    scales = np.r_[1., model.daily_factor_scales]
    covariance = model.joint_innovation_covariance
    if front_coupling is not None:
        front_coupling.validate_model(model)
        covariance = front_coupling.innovation_covariance(front=front, reference_forward=reference_forward)
    return covariance / np.outer(scales, scales)
