"""Static, Q, and P training stages for the Tangent Levy v2 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from cdxcore.err import verify, verify_inp

from .config import DynamicsConfig, StaticFitConfig
from .double_exponential import call_prices
from .dynamics import DynamicsFit, fit_reduced_dynamics
from .physical_measure import PhysicalMeasureFit, fit_constant_market_price_of_risk
from .pricing import density_call_prices
from .static_fit import (
    DayFit,
    MaturityFit,
    SymmetryCurve,
    fit_maturity,
    fitted_average_density,
    reconstruct_forward_density_positive_spline,
)
from .v2_config import TangentLevyV2Config
from .v2_market import V2MarketDay


@dataclass(frozen=True)
class V2ReconstructionAudit:
    """Loss of fit caused by projecting a day onto the daily codebook grid."""

    maximum_relative_projection: float
    relative_l1_projection: float
    repricing_rmse_half_spreads: float
    maximum_absolute_price_error: float
    passed: bool


@dataclass(frozen=True)
class V2HistoricalFit:
    """Fully reconstructed exact-population codebooks and their fit audit."""

    dates: tuple[date, ...]
    day_fits: tuple[DayFit, ...]
    symmetry_curve: SymmetryCurve
    forward_density: np.ndarray
    audits: tuple[V2ReconstructionAudit, ...]
    consecutive_mask: np.ndarray


@dataclass(frozen=True)
class V2MeasureFits:
    """Shared volatility factors with distinct risk-neutral and physical drifts."""

    q_dynamics: DynamicsFit
    p_market_price_of_risk: PhysicalMeasureFit


def fit_v2_market_day(
    market: V2MarketDay,
    *,
    static_config: StaticFitConfig = StaticFitConfig(),
    symmetry_curve: SymmetryCurve | None = None,
    warm_start: DayFit | None = None,
) -> DayFit:
    """Fit the paper's bid/ask interval loss to every retained maturity."""

    warm_maturities = (
        np.empty(0, dtype=np.float64)
        if warm_start is None
        else warm_start.maturities
    )
    fits: list[MaturityFit] = []
    for book in market.books:
        if symmetry_curve is not None and not (
            symmetry_curve.maturities[0] <= book.tau <= symmetry_curve.maturities[-1]
        ):
            continue
        target_symmetry = (
            None
            if symmetry_curve is None
            else float(np.asarray(symmetry_curve(book.tau)).item())
        )
        warm = None
        if warm_start is not None:
            nearest = int(np.argmin(np.abs(warm_maturities - book.tau)))
            warm = warm_start.fits[nearest].parameters
        fits.append(
            fit_maturity(
                tau=book.tau,
                strikes=book.strikes,
                target_calls=book.sanos_calls,
                bids=book.bids,
                asks=book.asks,
                mask=np.ones(book.quote_count, dtype=bool),
                config=static_config,
                target_symmetry_index=target_symmetry,
                warm_start=warm,
                loss="interval_excess",
                midpoint_weight=0.05,
            )
        )
    verify(
        len(fits) >= 2,
        lambda: f"fewer than two maturities calibrated on {market.market_date}",
    )
    return DayFit(
        market_date=market.market_date,
        fits=tuple(fits),
        initial_symmetry_fit=symmetry_curve is None,
    )


def _repricing_audit(
    market: V2MarketDay,
    day_fit: DayFit,
    projected_average_density: np.ndarray,
    *,
    jump_x: np.ndarray,
    static_config: StaticFitConfig,
) -> tuple[float, float]:
    normalized: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    books_by_tau = {round(book.tau, 12): book for book in market.books}
    for index, fit in enumerate(day_fit.fits):
        book = books_by_tau[round(fit.tau, 12)]
        log_strikes = np.log(book.strikes)
        original = call_prices(
            tau=fit.tau,
            log_strikes=log_strikes,
            parameters=fit.parameters,
            config=static_config.pricing,
        )
        reconstructed = density_call_prices(
            tau=fit.tau,
            log_strikes=log_strikes,
            jump_x=jump_x,
            average_density=projected_average_density[index],
            config=static_config.pricing,
        )
        error = reconstructed - original
        half_spread = np.maximum(
            0.5 * (book.asks - book.bids), static_config.half_spread_floor
        )
        normalized.append(error / half_spread)
        absolute.append(np.abs(error))
    normalized_values = np.concatenate(normalized)
    absolute_values = np.concatenate(absolute)
    return (
        float(np.sqrt(np.mean(np.square(normalized_values)))),
        float(np.max(absolute_values)),
    )


def reconstruct_v2_day(
    market: V2MarketDay,
    day_fit: DayFit,
    *,
    config: TangentLevyV2Config,
    static_config: StaticFitConfig = StaticFitConfig(),
) -> tuple[np.ndarray, V2ReconstructionAudit]:
    """Reconstruct and audit one daily-grid forward-density codebook."""

    tau = np.asarray(config.dynamics.maturity_grid, dtype=np.float64)
    jump_x = np.asarray(config.dynamics.jump_grid, dtype=np.float64)
    fitted_tau = day_fit.maturities
    fitted_density = fitted_average_density(day_fit, jump_x)
    if fitted_tau[-1] < tau[-1]:
        # A common Musiela grid is required for covariance estimation.  Listed
        # expiries do not always reach the configured 70-day edge, so extend
        # the last calibrated average Levy density as a transparent flat-tail
        # boundary condition.  Repricing below remains restricted to quotes.
        fitted_tau = np.append(fitted_tau, tau[-1])
        fitted_density = np.concatenate(
            (fitted_density, fitted_density[-1:]), axis=0
        )
    reconstruction = reconstruct_forward_density_positive_spline(
        fitted_tau=fitted_tau,
        fitted_average_density=fitted_density,
        target_tau=tau,
        jump_x=jump_x,
        relative_tolerance=config.dynamics.reconstruction_relative_tolerance,
        forward_density_floor=static_config.forward_density_floor,
        basis_count=config.dynamics.reconstruction_basis_count,
        curvature_penalty=config.dynamics.reconstruction_curvature_penalty,
        relative_weight_floor=(
            config.dynamics.reconstruction_relative_weight_floor
        ),
    )
    repricing_rmse, maximum_price_error = _repricing_audit(
        market,
        day_fit,
        reconstruction.projected_fitted_average_density[: len(day_fit.fits)],
        jump_x=jump_x,
        static_config=static_config,
    )
    passed = bool(
        reconstruction.passed_tolerance
        and repricing_rmse <= config.dynamics.reconstruction_repricing_tolerance
    )
    return reconstruction.forward_density, V2ReconstructionAudit(
        maximum_relative_projection=reconstruction.maximum_relative_projection,
        relative_l1_projection=reconstruction.relative_l1_projection,
        repricing_rmse_half_spreads=repricing_rmse,
        maximum_absolute_price_error=maximum_price_error,
        passed=passed,
    )


def fit_v2_measures(
    history: V2HistoricalFit,
    *,
    config: TangentLevyV2Config,
    require_reconstruction_pass: bool = True,
) -> V2MeasureFits:
    """Estimate reduced-node Q factors and the P factor risk premium."""

    verify_inp(
        len(history.dates) == len(history.audits)
        and history.forward_density.shape[0] == len(history.dates)
        and history.consecutive_mask.shape == (len(history.dates) - 1,),
        lambda: "historical fit arrays and dates are inconsistent",
    )
    if require_reconstruction_pass:
        verify(
            all(audit.passed for audit in history.audits),
            lambda: (
                "reconstruction gate failed; pass require_reconstruction_pass=False "
                "only for an explicitly misspecified-model experiment"
            ),
        )
    training_mask = np.asarray(
        [value <= config.dynamics.training_end for value in history.dates], dtype=bool
    )
    verify(np.count_nonzero(training_mask) >= 61, "insufficient pre-cutoff history")
    selected = history.forward_density[training_mask]
    selected_positions = np.flatnonzero(training_mask)
    verify(
        np.array_equal(selected_positions, np.arange(len(selected_positions))),
        "the dynamics training dates must be a contiguous prefix",
    )
    consecutive = history.consecutive_mask[: len(selected_positions) - 1]
    dynamics_config = DynamicsConfig(
        tau_grid=config.dynamics.maturity_grid,
        jump_x_grid=config.dynamics.jump_grid,
        num_factors=config.dynamics.factor_rank,
        covariance_shrinkage=config.dynamics.covariance_shrinkage,
        require_reconstruction_pass=True,
    )
    q_dynamics = fit_reduced_dynamics(
        selected,
        config=dynamics_config,
        consecutive_mask=consecutive,
        maturity_stride=config.dynamics.reduced_maturity_stride,
        jump_node_count=config.dynamics.reduced_jump_node_count,
    )
    p_fit = fit_constant_market_price_of_risk(
        selected,
        q_dynamics,
        ridge=config.dynamics.physical_ridge,
        consecutive_mask=consecutive,
    )
    return V2MeasureFits(q_dynamics=q_dynamics, p_market_price_of_risk=p_fit)
