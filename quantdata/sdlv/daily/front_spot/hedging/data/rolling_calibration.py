"""Rolling physical and risk-neutral DLV calibration for the hedge benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    FixedContractDlvHistory,
    RollingDlvFactorConfig,
    RollingDlvFactorFit,
    RollingFactorAlignment,
    RollingFactorAlignmentConfig,
    align_rolling_dlv_factors,
    build_fixed_contract_dlv_history,
    fit_dlv_factors,
)
from quantdata.sdlv.daily.front_spot.jax_model import (
    density0_from_support,
    risk_neutral_dlv_drift,
    solve_sanos_step,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@version(
    "1.3",
    dependencies=[RollingDlvFactorConfig, RollingFactorAlignmentConfig],
)
@dataclass(frozen=True, kw_only=True)
class RollingDlvCalibrationConfig:
    """Controls for one rolling SDLV calibration date."""

    factor: RollingDlvFactorConfig = RollingDlvFactorConfig(factor_rank=16)
    alignment: RollingFactorAlignmentConfig = RollingFactorAlignmentConfig()
    greek_factor_count: int = 8
    drift_projection_ridge: float = 1.0e-10
    rolling_drift_ridge_candidates: tuple[float, ...] = (
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
        10.0,
        100.0,
    )
    rolling_validation_fraction: float = 1.0 / 3.0
    maximum_daily_risk_price_norm: float = 3.0
    minimum_validation_improvement_fraction: float = 0.005
    minimum_validation_gain_t: float = 1.96
    validation_hac_lags: int = 5
    drift_batch_size: int = 32
    physical_ar_z_threshold: float | None = None
    physical_ar_hac_lags: int = 5

    def __post_init__(self) -> None:
        verify_inp(
            1 <= self.greek_factor_count <= self.factor.factor_rank,
            lambda: "greek_factor_count must lie within the fitted factor rank",
        )
        verify_inp(
            np.isfinite(self.drift_projection_ridge)
            and self.drift_projection_ridge >= 0.0,
            lambda: "drift_projection_ridge must be finite and non-negative",
        )
        verify_inp(
            self.drift_batch_size >= 1,
            lambda: "drift_batch_size must be positive",
        )
        verify_inp(
            len(self.rolling_drift_ridge_candidates) >= 2
            and all(
                np.isfinite(value) and value >= 0.0
                for value in self.rolling_drift_ridge_candidates
            )
            and all(
                left < right
                for left, right in zip(
                    self.rolling_drift_ridge_candidates,
                    self.rolling_drift_ridge_candidates[1:],
                )
            ),
            lambda: "rolling drift ridge candidates must be increasing and non-negative",
        )
        verify_inp(
            np.isfinite(self.rolling_validation_fraction)
            and 0.1 <= self.rolling_validation_fraction <= 0.5,
            lambda: "rolling validation fraction must lie in [0.1,0.5]",
        )
        verify_inp(
            np.isfinite(self.maximum_daily_risk_price_norm)
            and self.maximum_daily_risk_price_norm > 0.0,
            lambda: "maximum daily risk-price norm must be finite and positive",
        )
        verify_inp(
            np.isfinite(self.minimum_validation_improvement_fraction)
            and self.minimum_validation_improvement_fraction >= 0.0,
            lambda: "minimum validation improvement must be non-negative",
        )
        verify_inp(
            np.isfinite(self.minimum_validation_gain_t)
            and self.minimum_validation_gain_t >= 0.0,
            lambda: "minimum validation gain statistic must be non-negative",
        )
        verify_inp(
            self.validation_hac_lags >= 0,
            lambda: "validation_hac_lags must be non-negative",
        )
        verify_inp(
            self.physical_ar_z_threshold is None
            or (
                np.isfinite(self.physical_ar_z_threshold)
                and self.physical_ar_z_threshold >= 0.0
            ),
            lambda: "physical_ar_z_threshold must be non-negative or None",
        )
        verify_inp(
            self.physical_ar_hac_lags >= 0,
            lambda: "physical_ar_hac_lags must be non-negative",
        )


@dataclass(frozen=True, kw_only=True)
class CompatiblePhysicalDrift:
    """Projection of an empirical physical drift onto ``m_Q + xi lambda``."""

    empirical_log_drift: FloatArray
    achievable_log_drift: FloatArray
    market_price_of_risk: FloatArray
    weighted_target_rms: float
    weighted_residual_rms: float
    achievable_fraction: float

    def __post_init__(self) -> None:
        if (
            self.empirical_log_drift.shape != self.achievable_log_drift.shape
            or self.market_price_of_risk.ndim != 1
        ):
            raise ValueError("physical-drift projection shapes do not align")
        arrays = (
            self.empirical_log_drift,
            self.achievable_log_drift,
            self.market_price_of_risk,
        )
        scalars = (
            self.weighted_target_rms,
            self.weighted_residual_rms,
            self.achievable_fraction,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays) or any(
            not np.isfinite(value) for value in scalars
        ):
            raise ValueError("physical-drift projection must be finite")


@dataclass(frozen=True, kw_only=True)
class RollingPhysicalDriftFit:
    """One constant equivalent-measure risk price fitted over a rolling window."""

    training_origin_dates: tuple[date, ...]
    training_endpoint_dates: tuple[date, ...]
    empirical_log_drifts: FloatArray
    risk_neutral_log_drifts: FloatArray
    achievable_log_drifts: FloatArray
    coordinate_weights: FloatArray
    market_price_of_risk: FloatArray
    market_price_of_risk_standard_errors: FloatArray
    weighted_target_rms: float
    weighted_residual_rms: float
    achievable_fraction: float
    selected_ridge: float = 0.0
    validation_loss: float = 0.0
    validation_loss_standard_error: float = 0.0
    validation_baseline_loss: float = 0.0
    validation_improvement_fraction: float = 0.0
    validation_gain: float = 0.0
    validation_gain_standard_error: float = 0.0
    validation_gain_t: float = 0.0
    daily_risk_price_norm: float = 0.0
    accepted: bool = True

    def __post_init__(self) -> None:
        sample_count = len(self.training_origin_dates)
        if len(self.training_endpoint_dates) != sample_count:
            raise ValueError("rolling physical-drift dates do not align")
        expected_prefix = (sample_count,)
        grids = (
            self.empirical_log_drifts,
            self.risk_neutral_log_drifts,
            self.achievable_log_drifts,
            self.coordinate_weights,
        )
        if any(
            values.ndim != 3 or values.shape[:1] != expected_prefix
            for values in grids
        ) or any(values.shape != grids[0].shape for values in grids[1:]):
            raise ValueError("rolling physical-drift grids do not align")
        if (
            self.market_price_of_risk.ndim != 1
            or self.market_price_of_risk_standard_errors.shape
            != self.market_price_of_risk.shape
        ):
            raise ValueError("rolling market price of risk must be a vector")
        if any(
            not np.all(np.isfinite(values))
            for values in (
                *grids,
                self.market_price_of_risk,
                self.market_price_of_risk_standard_errors,
            )
        ):
            raise ValueError("rolling physical-drift fit must be finite")
        if (
            np.any(self.coordinate_weights < 0.0)
            or np.any(self.market_price_of_risk_standard_errors < 0.0)
        ):
            raise ValueError("rolling physical-drift weights must be non-negative")
        scalars = (
            self.weighted_target_rms,
            self.weighted_residual_rms,
            self.achievable_fraction,
            self.selected_ridge,
            self.validation_loss,
            self.validation_loss_standard_error,
            self.validation_baseline_loss,
            self.validation_improvement_fraction,
            self.validation_gain,
            self.validation_gain_standard_error,
            self.validation_gain_t,
            self.daily_risk_price_norm,
        )
        if any(not np.isfinite(value) for value in scalars) or any(
            value < 0.0
            for value in (
                self.selected_ridge,
                self.validation_loss,
                self.validation_loss_standard_error,
                self.validation_baseline_loss,
                self.validation_gain_standard_error,
                self.daily_risk_price_norm,
            )
        ):
            raise ValueError("rolling physical-drift diagnostics must be finite")


@dataclass(frozen=True, kw_only=True)
class RollingDlvCalibration:
    """Rolling factor fit and current-state physical/risk-neutral drifts."""

    as_of_date: date
    factors: RollingDlvFactorFit
    alignment: RollingFactorAlignment | None
    greek_factor_count: int
    physical_conditional_intercept: FloatArray
    physical_conditional_slope: FloatArray
    physical_ar_z_threshold: float
    retained_physical_ar_coordinates: int
    front_spot_market_price_of_risk: float
    front_density: FloatArray
    future_densities: FloatArray
    density_volatilities: FloatArray
    coordinate_weights: FloatArray
    risk_neutral_relative_dlvsq_drift: FloatArray
    risk_neutral_log_dlvsq_drift: FloatArray
    conditional_physical: CompatiblePhysicalDrift
    unconditional_physical: CompatiblePhysicalDrift
    rolling_physical: RollingPhysicalDriftFit | None
    conditional_unconditional_weighted_rms: float

    def __post_init__(self) -> None:
        state_shape = self.factors.daily_loadings.shape[:2]
        density_shape = (state_shape[0], state_shape[1] + 2)
        if self.as_of_date != self.factors.as_of_date:
            raise ValueError("calibration and factor-fit as-of dates differ")
        if not 1 <= self.greek_factor_count <= self.factors.factor_rank:
            raise ValueError("calibration Greek factor count is invalid")
        if (
            self.physical_conditional_intercept.shape != state_shape
            or self.physical_conditional_slope.shape != state_shape
            or not np.all(np.isfinite(self.physical_conditional_intercept))
            or not np.all(np.isfinite(self.physical_conditional_slope))
            or not np.isfinite(self.physical_ar_z_threshold)
            or not np.isfinite(self.front_spot_market_price_of_risk)
            or not 0
            <= self.retained_physical_ar_coordinates
            <= int(np.prod(state_shape))
        ):
            raise ValueError("calibration physical AR coefficients are invalid")
        if (
            self.front_density.shape != (state_shape[1] + 2,)
            or self.future_densities.shape != density_shape
            or self.density_volatilities.shape
            != (*density_shape, self.factors.factor_rank)
            or self.coordinate_weights.shape != state_shape
            or self.risk_neutral_relative_dlvsq_drift.shape != state_shape
            or self.risk_neutral_log_dlvsq_drift.shape != state_shape
        ):
            raise ValueError("rolling DLV calibration arrays do not align")
        arrays = (
            self.front_density,
            self.future_densities,
            self.density_volatilities,
            self.coordinate_weights,
            self.risk_neutral_relative_dlvsq_drift,
            self.risk_neutral_log_dlvsq_drift,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("rolling DLV calibration must be finite")
        if (
            np.any(self.front_density < -1.0e-12)
            or np.any(self.future_densities < -1.0e-12)
            or np.any(self.coordinate_weights < 0.0)
            or not np.isfinite(self.conditional_unconditional_weighted_rms)
        ):
            raise ValueError("rolling DLV calibration probabilities are invalid")


def fit_rolling_dlv_calibration(
    panel: PublicHedgingPanel,
    *,
    as_of_date: date,
    reference_factors: RollingDlvFactorFit | None = None,
    config: RollingDlvCalibrationConfig = RollingDlvCalibrationConfig(),
) -> RollingDlvCalibration:
    """Fit one trailing window and evaluate its current ``P`` and ``Q`` drifts."""
    require_public_hedging_panel(panel)
    origin_dates = panel.prior_one_day_origins(
        as_of_date,
        count=config.factor.window_sessions,
    )
    history = build_fixed_contract_dlv_history(panel, origin_dates=origin_dates)
    factors = fit_dlv_factors(history, as_of_date=as_of_date, config=config.factor)
    alignment = None
    if reference_factors is not None:
        alignment = align_rolling_dlv_factors(
            reference_factors,
            factors,
            config=config.alignment,
        )
        factors = alignment.fit
    validation_count = max(
        config.factor.factor_rank + 1,
        int(round(len(history.origin_dates) * config.rolling_validation_fraction)),
    )
    training_count = len(history.origin_dates) - validation_count
    if training_count <= config.factor.factor_rank:
        raise ValueError("rolling physical-drift split leaves too few training dates")
    selection_history = _history_prefix(history, training_count)
    selection_factors = fit_dlv_factors(
        selection_history,
        as_of_date=selection_history.endpoint_dates[-1],
        config=replace(config.factor, window_sessions=training_count),
    )
    current = calibrate_current_dlv_drifts(
        day=panel.market_day(as_of_date),
        history=history,
        factors=factors,
        alignment=alignment,
        greek_factor_count=config.greek_factor_count,
        ridge=config.drift_projection_ridge,
        physical_ar_z_threshold=config.physical_ar_z_threshold,
        physical_ar_hac_lags=config.physical_ar_hac_lags,
    )
    rolling_physical = fit_rolling_constant_physical_drift(
        panel,
        history=history,
        factors=factors,
        selection_factors=selection_factors,
        conditional_intercept=current.physical_conditional_intercept,
        conditional_slope=current.physical_conditional_slope,
        front_spot_market_price_of_risk=(
            current.front_spot_market_price_of_risk
        ),
        ridge_candidates=config.rolling_drift_ridge_candidates,
        validation_fraction=config.rolling_validation_fraction,
        maximum_daily_risk_price_norm=config.maximum_daily_risk_price_norm,
        minimum_validation_improvement_fraction=(
            config.minimum_validation_improvement_fraction
        ),
        minimum_validation_gain_t=config.minimum_validation_gain_t,
        validation_hac_lags=config.validation_hac_lags,
        batch_size=config.drift_batch_size,
    )
    return replace(current, rolling_physical=rolling_physical)


def calibrate_current_dlv_drifts(
    *,
    day: PublicMarketDay,
    history: FixedContractDlvHistory,
    factors: RollingDlvFactorFit,
    alignment: RollingFactorAlignment | None = None,
    greek_factor_count: int = 8,
    ridge: float = 1.0e-10,
    physical_ar_z_threshold: float | None = None,
    physical_ar_hac_lags: int = 5,
) -> RollingDlvCalibration:
    """Evaluate current-state ``Q`` drift and closest equivalent ``P`` drift."""
    if not jax.config.x64_enabled:
        raise RuntimeError("rolling DLV calibration requires JAX 64-bit mode")
    if day.trading_date != factors.as_of_date:
        raise ValueError("market day and factor fit must share an as-of date")
    if history.origin_log_dlvsq.shape[1:] != factors.daily_loadings.shape[:2]:
        raise ValueError("history and factor loading grids do not align")
    if not 1 <= greek_factor_count <= factors.factor_rank:
        raise ValueError("greek_factor_count must lie within the fitted rank")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")

    support = jnp.asarray(day.dlv_support_strikes, dtype=jnp.float64)
    dlvsq = jnp.asarray(day.dlvsq, dtype=jnp.float64)
    annualized_loadings = jnp.asarray(
        factors.annualized_loadings,
        dtype=jnp.float64,
    )
    expected_shape = tuple(day.dlvsq[1:].shape)
    if factors.daily_loadings.shape[:2] != expected_shape:
        raise ValueError(
            f"factor grid {factors.daily_loadings.shape[:2]} does not match "
            f"current future DLV grid {expected_shape}"
        )
    expiry_dt = 1.0 / DAYS_PER_YEAR
    front_density = solve_sanos_step(
        support,
        density0_from_support(support),
        dlvsq[0] * expiry_dt,
    )
    future_densities, density_volatilities, relative_drift = (
        risk_neutral_dlv_drift(
            support,
            front_density,
            dlvsq[1:],
            annualized_loadings,
            dt=expiry_dt,
        )
    )
    front_density_np = np.asarray(front_density)
    future_densities_np = np.asarray(future_densities)
    density_volatilities_np = np.asarray(density_volatilities)
    relative_drift_np = np.asarray(relative_drift)
    factor_variance = np.sum(factors.annualized_loadings**2, axis=-1)
    risk_neutral_log_drift = relative_drift_np - 0.5 * factor_variance

    interior_density = np.maximum(future_densities_np[..., 1:-1], 0.0)
    expiry_mass = np.sum(interior_density, axis=-1, keepdims=True)
    if np.any(expiry_mass <= np.finfo(np.float64).tiny):
        raise FloatingPointError("a future expiry has no interior probability mass")
    coordinate_weights = interior_density / expiry_mass
    (
        physical_intercept,
        physical_slope,
        ar_z_threshold,
        retained_ar_coordinates,
    ) = _screen_physical_ar(
        history,
        factors=factors,
        configured_threshold=physical_ar_z_threshold,
        hac_lags=physical_ar_hac_lags,
    )
    front_spot_risk_price = factors.front_spot_market_price_of_risk
    fixed_risk_price = np.full(factors.factor_rank, np.nan, dtype=np.float64)
    fixed_risk_price[0] = front_spot_risk_price
    current_log_dlvsq = np.log(day.dlvsq[1:])
    conditional_empirical = DAYS_PER_YEAR * (
        physical_intercept
        + physical_slope * current_log_dlvsq
        - current_log_dlvsq
    )
    unconditional_empirical = DAYS_PER_YEAR * np.mean(history.increments, axis=0)
    conditional_projection = project_compatible_physical_drift(
        empirical_log_drift=conditional_empirical,
        risk_neutral_log_drift=risk_neutral_log_drift,
        coordinate_weights=coordinate_weights,
        annualized_loadings=factors.annualized_loadings,
        ridge=ridge,
        fixed_market_price_of_risk=fixed_risk_price,
    )
    unconditional_projection = project_compatible_physical_drift(
        empirical_log_drift=unconditional_empirical,
        risk_neutral_log_drift=risk_neutral_log_drift,
        coordinate_weights=coordinate_weights,
        annualized_loadings=factors.annualized_loadings,
        ridge=ridge,
        fixed_market_price_of_risk=fixed_risk_price,
    )
    conditional_unconditional_rms = _weighted_rms(
        conditional_empirical - unconditional_empirical,
        coordinate_weights,
    )
    return RollingDlvCalibration(
        as_of_date=day.trading_date,
        factors=factors,
        alignment=alignment,
        greek_factor_count=greek_factor_count,
        physical_conditional_intercept=physical_intercept,
        physical_conditional_slope=physical_slope,
        physical_ar_z_threshold=ar_z_threshold,
        retained_physical_ar_coordinates=retained_ar_coordinates,
        front_spot_market_price_of_risk=front_spot_risk_price,
        front_density=front_density_np,
        future_densities=future_densities_np,
        density_volatilities=density_volatilities_np,
        coordinate_weights=coordinate_weights,
        risk_neutral_relative_dlvsq_drift=relative_drift_np,
        risk_neutral_log_dlvsq_drift=risk_neutral_log_drift,
        conditional_physical=conditional_projection,
        unconditional_physical=unconditional_projection,
        rolling_physical=None,
        conditional_unconditional_weighted_rms=conditional_unconditional_rms,
    )


def _screen_physical_ar(
    history: FixedContractDlvHistory,
    *,
    factors: RollingDlvFactorFit,
    configured_threshold: float | None,
    hac_lags: int,
) -> tuple[FloatArray, FloatArray, float, int]:
    """Shrink noisy coordinatewise AR coefficients toward a random walk."""
    origin = history.origin_log_dlvsq
    origin_mean = np.mean(origin, axis=0)
    centered_origin = origin - origin_mean
    denominator = np.sum(centered_origin**2, axis=0)
    scale = np.maximum(np.sum(origin**2, axis=0), 1.0)
    variable = denominator > np.finfo(np.float64).eps * scale * len(origin)
    raw_increment_slope = np.zeros_like(denominator)
    raw_increment_slope[variable] = factors.conditional_slope[variable] - 1.0
    increments = history.increments
    mean_increment = np.mean(increments, axis=0)
    increment_residual = (
        increments
        - mean_increment[None, ...]
        - raw_increment_slope[None, ...] * centered_origin
    )
    score = centered_origin * increment_residual
    long_run_variance = np.sum(score**2, axis=0)
    linked = np.asarray(
        [
            history.endpoint_dates[index] == history.origin_dates[index + 1]
            for index in range(len(origin) - 1)
        ],
        dtype=bool,
    )
    for lag in range(1, min(hac_lags, len(origin) - 1) + 1):
        chain = np.ones(len(origin) - lag, dtype=bool)
        for offset in range(lag):
            chain &= linked[offset : offset + len(chain)]
        if not np.any(chain):
            continue
        weight = 1.0 - lag / (hac_lags + 1.0)
        long_run_variance += 2.0 * weight * np.sum(
            score[lag:][chain] * score[:-lag][chain],
            axis=0,
        )
    long_run_variance = np.maximum(long_run_variance, 0.0)
    standard_error = np.full_like(denominator, np.inf)
    standard_error[variable] = (
        np.sqrt(long_run_variance[variable]) / denominator[variable]
    )
    z_threshold = (
        math.sqrt(2.0 * math.log(max(denominator.size, 2)))
        if configured_threshold is None
        else configured_threshold
    )
    screened_increment_slope = np.sign(raw_increment_slope) * np.maximum(
        np.abs(raw_increment_slope) - z_threshold * standard_error,
        0.0,
    )
    conditional_slope = 1.0 + screened_increment_slope
    conditional_intercept = mean_increment - screened_increment_slope * origin_mean
    return (
        conditional_intercept,
        conditional_slope,
        float(z_threshold),
        int(np.count_nonzero(screened_increment_slope)),
    )


def _history_prefix(
    history: FixedContractDlvHistory,
    count: int,
) -> FixedContractDlvHistory:
    """Return the first ``count`` transitions without sharing holdout arrays."""
    if not 1 <= count <= len(history.origin_dates):
        raise ValueError("history prefix count is out of range")

    def optional_prefix(values: FloatArray | None) -> FloatArray | None:
        return None if values is None else values[:count].copy()

    return FixedContractDlvHistory(
        origin_dates=history.origin_dates[:count],
        endpoint_dates=history.endpoint_dates[:count],
        origin_log_dlvsq=history.origin_log_dlvsq[:count].copy(),
        endpoint_log_dlvsq=history.endpoint_log_dlvsq[:count].copy(),
        front_cdf_values=history.front_cdf_values[:count].copy(),
        spot_gaussian_scores=history.spot_gaussian_scores[:count].copy(),
        interpolation_clip_fractions=(
            history.interpolation_clip_fractions[:count].copy()
        ),
        forward_tail_extrapolation_fractions=(
            history.forward_tail_extrapolation_fractions[:count].copy()
        ),
        front_supports=optional_prefix(history.front_supports),
        front_densities=optional_prefix(history.front_densities),
        terminal_normalized_spots=optional_prefix(
            history.terminal_normalized_spots
        ),
        background_annual_volatilities=optional_prefix(
            history.background_annual_volatilities
        ),
    )


@jax.jit
def _evaluate_historical_q_drift_batch(
    supports: jax.Array,
    dlvsq: jax.Array,
    annualized_loadings: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate Q densities and relative DLV drifts for one padded batch."""
    expiry_dt = 1.0 / DAYS_PER_YEAR

    def evaluate_state(
        support: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        front_density = solve_sanos_step(
            support,
            density0_from_support(support),
            state[0] * expiry_dt,
        )
        future_densities, _, relative_drift = risk_neutral_dlv_drift(
            support,
            front_density,
            state[1:],
            annualized_loadings,
            dt=expiry_dt,
        )
        return future_densities, relative_drift

    return jax.vmap(evaluate_state)(supports, dlvsq)


def _historical_risk_neutral_inputs(
    panel: PublicHedgingPanel,
    *,
    history: FixedContractDlvHistory,
    factors: RollingDlvFactorFit,
    batch_size: int,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate historical Q log drifts and density weights for one basis."""
    days = tuple(panel.market_day(value) for value in history.origin_dates)
    supports = np.stack([day.dlv_support_strikes for day in days])
    dlvsq = np.stack([day.dlvsq for day in days])
    if factors.daily_loadings.shape[:2] != history.origin_log_dlvsq.shape[1:]:
        raise ValueError("factor basis and rolling history coordinates do not align")
    if not np.allclose(
        np.log(dlvsq[:, 1:]),
        history.origin_log_dlvsq,
        rtol=0.0,
        atol=2.0e-10,
    ):
        raise ValueError("rolling history and public DLV states do not match")
    annualized_loadings = jnp.asarray(
        factors.annualized_loadings,
        dtype=jnp.float64,
    )
    density_blocks: list[FloatArray] = []
    relative_drift_blocks: list[FloatArray] = []
    for start in range(0, len(days), batch_size):
        stop = min(start + batch_size, len(days))
        valid_count = stop - start
        support_block = supports[start:stop]
        dlvsq_block = dlvsq[start:stop]
        if valid_count < batch_size:
            pad_count = batch_size - valid_count
            support_block = np.concatenate(
                (support_block, np.repeat(support_block[-1:], pad_count, axis=0))
            )
            dlvsq_block = np.concatenate(
                (dlvsq_block, np.repeat(dlvsq_block[-1:], pad_count, axis=0))
            )
        batch_densities, batch_relative_drifts = (
            _evaluate_historical_q_drift_batch(
                jnp.asarray(support_block, dtype=jnp.float64),
                jnp.asarray(dlvsq_block, dtype=jnp.float64),
                annualized_loadings,
            )
        )
        density_blocks.append(np.asarray(batch_densities[:valid_count]))
        relative_drift_blocks.append(
            np.asarray(batch_relative_drifts[:valid_count])
        )
    future_densities = np.concatenate(density_blocks, axis=0)
    relative_drifts = np.concatenate(relative_drift_blocks, axis=0)
    factor_variance = np.sum(factors.annualized_loadings**2, axis=-1)
    risk_neutral_log_drifts = relative_drifts - 0.5 * factor_variance[None]
    interior_density = np.maximum(future_densities[..., 1:-1], 0.0)
    expiry_mass = np.sum(interior_density, axis=-1, keepdims=True)
    if np.any(expiry_mass <= np.finfo(np.float64).tiny):
        raise FloatingPointError("a historical future expiry has no interior mass")
    return risk_neutral_log_drifts, interior_density / expiry_mass


def _linked_transitions(
    origins: tuple[date, ...],
    endpoints: tuple[date, ...],
) -> BoolArray:
    """Return whether consecutive transition observations share an endpoint."""
    if len(origins) != len(endpoints):
        raise ValueError("transition dates do not align")
    return np.asarray(
        [endpoints[index] == origins[index + 1] for index in range(len(origins) - 1)],
        dtype=bool,
    )


def _newey_west_mean_standard_error(
    values: FloatArray,
    *,
    linked: BoolArray,
    lags: int,
) -> float:
    """Return a linked-chain Bartlett/Newey--West error of the sample mean."""
    observations = np.asarray(values, dtype=np.float64)
    if (
        observations.ndim != 1
        or len(observations) < 2
        or linked.shape != (len(observations) - 1,)
        or lags < 0
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("invalid Newey--West mean-error inputs")
    centered = observations - np.mean(observations)
    long_run_sum = float(centered @ centered)
    for lag in range(1, min(lags, len(observations) - 1) + 1):
        chain = np.ones(len(observations) - lag, dtype=bool)
        for offset in range(lag):
            chain &= linked[offset : offset + len(chain)]
        if np.any(chain):
            weight = 1.0 - lag / (lags + 1.0)
            long_run_sum += 2.0 * weight * float(
                centered[lag:][chain] @ centered[:-lag][chain]
            )
    return float(np.sqrt(max(long_run_sum, 0.0)) / len(observations))


def fit_rolling_constant_physical_drift(
    panel: PublicHedgingPanel,
    *,
    history: FixedContractDlvHistory,
    factors: RollingDlvFactorFit,
    selection_factors: RollingDlvFactorFit,
    conditional_intercept: FloatArray,
    conditional_slope: FloatArray,
    front_spot_market_price_of_risk: float | None = None,
    ridge_candidates: tuple[float, ...] = (1.0e-2, 1.0e-1, 1.0, 10.0),
    validation_fraction: float = 1.0 / 3.0,
    maximum_daily_risk_price_norm: float = 3.0,
    minimum_validation_improvement_fraction: float = 0.005,
    minimum_validation_gain_t: float = 1.96,
    validation_hac_lags: int = 5,
    batch_size: int = 32,
) -> RollingPhysicalDriftFit:
    """Fit one constant Brownian risk price across the trailing window."""
    if not jax.config.x64_enabled:
        raise RuntimeError("rolling physical-drift fitting requires JAX 64-bit mode")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (
        conditional_intercept.shape != history.origin_log_dlvsq.shape[1:]
        or conditional_slope.shape != conditional_intercept.shape
    ):
        raise ValueError("physical AR coefficients do not align with history")
    del conditional_intercept, conditional_slope
    empirical_log_drifts = DAYS_PER_YEAR * history.increments
    fixed_risk_price = None
    if front_spot_market_price_of_risk is not None:
        if not np.isfinite(front_spot_market_price_of_risk):
            raise ValueError("front spot market price of risk must be finite")
        fixed_risk_price = np.full(factors.factor_rank, np.nan, dtype=np.float64)
        fixed_risk_price[0] = front_spot_market_price_of_risk
    validation_count = max(
        factors.factor_rank + 1,
        int(round(len(history.origin_dates) * validation_fraction)),
    )
    training_count = len(history.origin_dates) - validation_count
    if training_count <= factors.factor_rank:
        raise ValueError("rolling physical-drift split leaves too few training dates")
    if (
        selection_factors.training_origin_dates
        != history.origin_dates[:training_count]
        or selection_factors.training_endpoint_dates
        != history.endpoint_dates[:training_count]
        or selection_factors.factor_rank != factors.factor_rank
    ):
        raise ValueError("selection factors must use only the training block")
    selection_risk_neutral_log_drifts, selection_coordinate_weights = (
        _historical_risk_neutral_inputs(
            panel,
            history=history,
            factors=selection_factors,
            batch_size=batch_size,
        )
    )
    risk_neutral_log_drifts, coordinate_weights = _historical_risk_neutral_inputs(
        panel,
        history=history,
        factors=factors,
        batch_size=batch_size,
    )
    selection_fixed_risk_price = np.full(
        selection_factors.factor_rank,
        np.nan,
        dtype=np.float64,
    )
    selection_fixed_risk_price[0] = (
        selection_factors.front_spot_market_price_of_risk
    )
    standard_error_overrides = np.full(
        factors.factor_rank,
        np.nan,
        dtype=np.float64,
    )
    standard_error_overrides[0] = (
        factors.front_spot_market_price_of_risk_standard_error
    )
    return fit_constant_physical_drift_from_inputs(
        training_origin_dates=history.origin_dates,
        training_endpoint_dates=history.endpoint_dates,
        empirical_log_drifts=empirical_log_drifts,
        selection_risk_neutral_log_drifts=selection_risk_neutral_log_drifts,
        selection_coordinate_weights=selection_coordinate_weights,
        selection_annualized_loadings=selection_factors.annualized_loadings,
        risk_neutral_log_drifts=risk_neutral_log_drifts,
        coordinate_weights=coordinate_weights,
        annualized_loadings=factors.annualized_loadings,
        selection_fixed_market_price_of_risk=selection_fixed_risk_price,
        fixed_market_price_of_risk=fixed_risk_price,
        market_price_of_risk_standard_error_overrides=standard_error_overrides,
        ridge_candidates=ridge_candidates,
        validation_fraction=validation_fraction,
        maximum_daily_risk_price_norm=maximum_daily_risk_price_norm,
        minimum_validation_improvement_fraction=(
            minimum_validation_improvement_fraction
        ),
        minimum_validation_gain_t=minimum_validation_gain_t,
        validation_hac_lags=validation_hac_lags,
    )


def fit_constant_physical_drift_from_inputs(
    *,
    training_origin_dates: tuple[date, ...],
    training_endpoint_dates: tuple[date, ...],
    empirical_log_drifts: FloatArray,
    selection_risk_neutral_log_drifts: FloatArray,
    selection_coordinate_weights: FloatArray,
    selection_annualized_loadings: FloatArray,
    risk_neutral_log_drifts: FloatArray,
    coordinate_weights: FloatArray,
    annualized_loadings: FloatArray,
    selection_fixed_market_price_of_risk: FloatArray | None = None,
    fixed_market_price_of_risk: FloatArray | None = None,
    market_price_of_risk_standard_error_overrides: FloatArray | None = None,
    ridge_candidates: tuple[float, ...] = (1.0e-2, 1.0e-1, 1.0, 10.0),
    validation_fraction: float = 1.0 / 3.0,
    maximum_daily_risk_price_norm: float = 3.0,
    minimum_validation_improvement_fraction: float = 0.005,
    minimum_validation_gain_t: float = 1.96,
    validation_hac_lags: int = 5,
) -> RollingPhysicalDriftFit:
    """Select and refit one compatible constant measure change.

    The loading arrays may be either one fixed field or one field per date.
    Candidate selection uses only the leading training block; the trailing
    block is reserved for the one-standard-error choice and signed HAC gate.
    """
    sample_count = len(training_origin_dates)
    if len(training_endpoint_dates) != sample_count:
        raise ValueError("rolling physical-drift dates do not align")
    empirical = np.asarray(empirical_log_drifts, dtype=np.float64)
    if empirical.shape[:1] != (sample_count,):
        raise ValueError("empirical rolling drifts do not align with dates")
    factor_count = np.asarray(annualized_loadings).shape[-1]
    validation_count = max(
        factor_count + 1,
        int(round(sample_count * validation_fraction)),
    )
    training_count = sample_count - validation_count
    if training_count <= factor_count:
        raise ValueError("rolling physical-drift split leaves too few training dates")
    candidates = tuple(float(value) for value in ridge_candidates)
    if (
        len(candidates) < 2
        or any(not np.isfinite(value) or value < 0.0 for value in candidates)
        or any(left >= right for left, right in zip(candidates, candidates[1:]))
    ):
        raise ValueError("ridge candidates must be increasing and non-negative")
    candidate_fits: list[RollingPhysicalDriftFit] = []
    candidate_daily_losses: list[FloatArray] = []
    selection_loadings = _slice_state_loadings(
        selection_annualized_loadings,
        stop=training_count,
        sample_count=sample_count,
    )
    validation_loadings = _slice_state_loadings(
        selection_annualized_loadings,
        start=training_count,
        sample_count=sample_count,
    )
    for candidate in candidates:
        candidate_fit = project_constant_physical_drift(
            training_origin_dates=training_origin_dates[:training_count],
            training_endpoint_dates=training_endpoint_dates[:training_count],
            empirical_log_drifts=empirical[:training_count],
            risk_neutral_log_drifts=(
                selection_risk_neutral_log_drifts[:training_count]
            ),
            coordinate_weights=selection_coordinate_weights[:training_count],
            annualized_loadings=selection_loadings,
            ridge=candidate,
            fixed_market_price_of_risk=selection_fixed_market_price_of_risk,
        )
        candidate_fits.append(candidate_fit)
        candidate_daily_losses.append(
            _daily_projection_losses(
                empirical_log_drifts=empirical[training_count:],
                risk_neutral_log_drifts=(
                    selection_risk_neutral_log_drifts[training_count:]
                ),
                coordinate_weights=selection_coordinate_weights[training_count:],
                annualized_loadings=validation_loadings,
                market_price_of_risk=candidate_fit.market_price_of_risk,
            )
        )
    mean_losses = np.asarray([np.mean(values) for values in candidate_daily_losses])
    best_index = int(np.argmin(mean_losses))
    validation_linked = _linked_transitions(
        training_origin_dates[training_count:],
        training_endpoint_dates[training_count:],
    )
    best_standard_error = _newey_west_mean_standard_error(
        candidate_daily_losses[best_index],
        linked=validation_linked,
        lags=validation_hac_lags,
    )
    eligible = np.flatnonzero(
        mean_losses <= mean_losses[best_index] + best_standard_error
    )
    selected_index = int(eligible[-1])
    selected_ridge = candidates[selected_index]
    selected_standard_error = _newey_west_mean_standard_error(
        candidate_daily_losses[selected_index],
        linked=validation_linked,
        lags=validation_hac_lags,
    )
    baseline_risk_price = np.zeros(factor_count, dtype=np.float64)
    if selection_fixed_market_price_of_risk is not None:
        selection_fixed = np.asarray(
            selection_fixed_market_price_of_risk,
            dtype=np.float64,
        )
        fixed_mask = np.isfinite(selection_fixed)
        baseline_risk_price[fixed_mask] = selection_fixed[fixed_mask]
    baseline_daily_losses = _daily_projection_losses(
        empirical_log_drifts=empirical[training_count:],
        risk_neutral_log_drifts=selection_risk_neutral_log_drifts[training_count:],
        coordinate_weights=selection_coordinate_weights[training_count:],
        annualized_loadings=validation_loadings,
        market_price_of_risk=baseline_risk_price,
    )
    baseline_loss = float(np.mean(baseline_daily_losses))
    selected_loss = float(mean_losses[selected_index])
    validation_improvement = (
        baseline_loss - selected_loss
    ) / max(baseline_loss, np.finfo(np.float64).tiny)
    validation_daily_gain = (
        baseline_daily_losses - candidate_daily_losses[selected_index]
    )
    validation_gain = float(np.mean(validation_daily_gain))
    validation_gain_standard_error = _newey_west_mean_standard_error(
        validation_daily_gain,
        linked=validation_linked,
        lags=validation_hac_lags,
    )
    validation_gain_t = validation_gain / max(
        validation_gain_standard_error,
        np.sqrt(np.finfo(np.float64).tiny),
    )
    result = project_constant_physical_drift(
        training_origin_dates=training_origin_dates,
        training_endpoint_dates=training_endpoint_dates,
        empirical_log_drifts=empirical,
        risk_neutral_log_drifts=risk_neutral_log_drifts,
        coordinate_weights=coordinate_weights,
        annualized_loadings=annualized_loadings,
        ridge=selected_ridge,
        fixed_market_price_of_risk=fixed_market_price_of_risk,
    )
    standard_errors = result.market_price_of_risk_standard_errors.copy()
    if market_price_of_risk_standard_error_overrides is not None:
        overrides = np.asarray(
            market_price_of_risk_standard_error_overrides,
            dtype=np.float64,
        )
        if overrides.shape != standard_errors.shape or np.any(np.isinf(overrides)):
            raise ValueError("risk-price standard-error overrides are invalid")
        override_mask = np.isfinite(overrides)
        standard_errors[override_mask] = overrides[override_mask]
    daily_norm = float(
        np.linalg.norm(result.market_price_of_risk) / np.sqrt(DAYS_PER_YEAR)
    )
    accepted = bool(
        daily_norm <= maximum_daily_risk_price_norm
        and validation_improvement >= minimum_validation_improvement_fraction
        and validation_gain_t >= minimum_validation_gain_t
    )
    return replace(
        result,
        market_price_of_risk_standard_errors=standard_errors,
        selected_ridge=selected_ridge,
        validation_loss=selected_loss,
        validation_loss_standard_error=selected_standard_error,
        validation_baseline_loss=baseline_loss,
        validation_improvement_fraction=validation_improvement,
        validation_gain=validation_gain,
        validation_gain_standard_error=validation_gain_standard_error,
        validation_gain_t=validation_gain_t,
        daily_risk_price_norm=daily_norm,
        accepted=accepted,
    )


def _slice_state_loadings(
    annualized_loadings: FloatArray,
    *,
    sample_count: int,
    start: int = 0,
    stop: int | None = None,
) -> FloatArray:
    """Slice dated loadings while leaving one common field unchanged."""
    values = np.asarray(annualized_loadings, dtype=np.float64)
    if values.ndim == 4 and values.shape[0] == sample_count:
        return values[start:stop]
    return values


def project_constant_physical_drift(
    *,
    training_origin_dates: tuple[date, ...],
    training_endpoint_dates: tuple[date, ...],
    empirical_log_drifts: FloatArray,
    risk_neutral_log_drifts: FloatArray,
    coordinate_weights: FloatArray,
    annualized_loadings: FloatArray,
    ridge: float = 1.0e-10,
    fixed_market_price_of_risk: FloatArray | None = None,
) -> RollingPhysicalDriftFit:
    """Project all historical drifts onto one constant Brownian risk price."""
    empirical = np.asarray(empirical_log_drifts, dtype=np.float64)
    risk_neutral = np.asarray(risk_neutral_log_drifts, dtype=np.float64)
    weights = np.asarray(coordinate_weights, dtype=np.float64)
    loadings = np.asarray(annualized_loadings, dtype=np.float64)
    if empirical.ndim != 3 or risk_neutral.shape != empirical.shape:
        raise ValueError("historical empirical and risk-neutral drifts must align")
    if weights.shape != empirical.shape:
        raise ValueError("historical drift weights and DLV loadings must align")
    if empirical.shape[0] != len(training_origin_dates) or len(
        training_endpoint_dates
    ) != len(training_origin_dates):
        raise ValueError("historical drift dates and arrays must align")
    arrays = (empirical, risk_neutral, weights, loadings)
    if any(not np.all(np.isfinite(values)) for values in arrays) or np.any(weights < 0.0):
        raise ValueError("historical drift projection inputs are invalid")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")

    sample_count = len(empirical)
    coordinate_count = int(np.prod(empirical.shape[1:]))
    flat_weights = weights.reshape(sample_count, coordinate_count)
    weight_sums = np.sum(flat_weights, axis=1)
    if np.any(weight_sums <= np.finfo(np.float64).tiny):
        raise ValueError("each historical state needs positive projection weight")
    flat_weights *= coordinate_count / weight_sums[:, None]
    loading_matrix = _state_loading_matrix(loadings, empirical.shape)
    target = (empirical - risk_neutral).reshape(sample_count, coordinate_count)
    risk_price = _solve_weighted_risk_price(
        loading_matrix=loading_matrix,
        target=target,
        weights=flat_weights,
        ridge=ridge,
        fixed_market_price_of_risk=fixed_market_price_of_risk,
    )
    risk_price_standard_errors = _risk_price_standard_errors(
        loading_matrix=loading_matrix,
        target=target,
        weights=flat_weights,
        ridge=ridge,
        market_price_of_risk=risk_price,
        fixed_market_price_of_risk=fixed_market_price_of_risk,
    )
    projected = np.einsum("spf,f->sp", loading_matrix, risk_price)
    residual = target - projected
    target_square = float(np.sum(flat_weights * target**2))
    residual_square = float(np.sum(flat_weights * residual**2))
    weight_total = float(np.sum(flat_weights))
    achievable_fraction = 1.0
    if target_square > np.finfo(np.float64).tiny:
        achievable_fraction -= residual_square / target_square
    return RollingPhysicalDriftFit(
        training_origin_dates=training_origin_dates,
        training_endpoint_dates=training_endpoint_dates,
        empirical_log_drifts=empirical,
        risk_neutral_log_drifts=risk_neutral,
        achievable_log_drifts=(
            risk_neutral + projected.reshape(empirical.shape)
        ),
        coordinate_weights=weights,
        market_price_of_risk=risk_price,
        market_price_of_risk_standard_errors=risk_price_standard_errors,
        weighted_target_rms=float(np.sqrt(target_square / weight_total)),
        weighted_residual_rms=float(np.sqrt(residual_square / weight_total)),
        achievable_fraction=float(achievable_fraction),
    )


def _daily_projection_losses(
    *,
    empirical_log_drifts: FloatArray,
    risk_neutral_log_drifts: FloatArray,
    coordinate_weights: FloatArray,
    annualized_loadings: FloatArray,
    market_price_of_risk: FloatArray,
) -> FloatArray:
    empirical = np.asarray(empirical_log_drifts, dtype=np.float64)
    loading_matrix = _state_loading_matrix(annualized_loadings, empirical.shape)
    projected = np.einsum("spf,f->sp", loading_matrix, market_price_of_risk)
    residual = empirical - risk_neutral_log_drifts - projected.reshape(empirical.shape)
    flat_weights = coordinate_weights.reshape(len(coordinate_weights), -1)
    flat_residual = residual.reshape(len(residual), -1)
    weight_sums = np.sum(flat_weights, axis=1)
    if np.any(weight_sums <= np.finfo(np.float64).tiny):
        raise ValueError("daily validation weights must have positive mass")
    return np.sum(flat_weights * flat_residual**2, axis=1) / weight_sums


def project_compatible_physical_drift(
    *,
    empirical_log_drift: FloatArray,
    risk_neutral_log_drift: FloatArray,
    coordinate_weights: FloatArray,
    annualized_loadings: FloatArray,
    ridge: float = 1.0e-10,
    fixed_market_price_of_risk: FloatArray | None = None,
) -> CompatiblePhysicalDrift:
    """Project one empirical drift onto the equivalent-measure loading span."""
    empirical = np.asarray(empirical_log_drift, dtype=np.float64)
    risk_neutral = np.asarray(risk_neutral_log_drift, dtype=np.float64)
    weights = np.asarray(coordinate_weights, dtype=np.float64)
    loadings = np.asarray(annualized_loadings, dtype=np.float64)
    if empirical.shape != risk_neutral.shape or weights.shape != empirical.shape:
        raise ValueError("empirical, risk-neutral, and weight grids must align")
    if loadings.shape[:-1] != empirical.shape:
        raise ValueError("DLV loadings and drift coordinates must align")
    if (
        np.any(~np.isfinite(empirical))
        or np.any(~np.isfinite(risk_neutral))
        or np.any(~np.isfinite(weights))
        or np.any(~np.isfinite(loadings))
        or np.any(weights < 0.0)
    ):
        raise ValueError("drift projection inputs must be finite with non-negative weights")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")

    flat_weights = weights.reshape(-1)
    weight_sum = float(np.sum(flat_weights))
    if weight_sum <= np.finfo(np.float64).tiny:
        raise ValueError("drift projection weights must have positive mass")
    flat_weights = flat_weights * (len(flat_weights) / weight_sum)
    loading_matrix = loadings.reshape(-1, loadings.shape[-1])
    target = (empirical - risk_neutral).reshape(-1)
    risk_price = _solve_weighted_risk_price(
        loading_matrix=loading_matrix,
        target=target[None, :],
        weights=flat_weights[None, :],
        ridge=ridge,
        fixed_market_price_of_risk=fixed_market_price_of_risk,
    )
    projected = loading_matrix @ risk_price
    residual = target - projected
    target_square = float(np.sum(flat_weights * target**2))
    residual_square = float(np.sum(flat_weights * residual**2))
    weight_total = float(np.sum(flat_weights))
    achievable_fraction = 1.0
    if target_square > np.finfo(np.float64).tiny:
        achievable_fraction -= residual_square / target_square
    return CompatiblePhysicalDrift(
        empirical_log_drift=empirical,
        achievable_log_drift=risk_neutral + projected.reshape(empirical.shape),
        market_price_of_risk=risk_price,
        weighted_target_rms=float(np.sqrt(target_square / weight_total)),
        weighted_residual_rms=float(np.sqrt(residual_square / weight_total)),
        achievable_fraction=float(achievable_fraction),
    )


def _solve_weighted_risk_price(
    *,
    loading_matrix: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    ridge: float,
    fixed_market_price_of_risk: FloatArray | None,
) -> FloatArray:
    """Solve a pooled projection with optional identified factor values.

    Finite entries in ``fixed_market_price_of_risk`` are held fixed and
    ``NaN`` entries remain free.  The front-spot calibration uses this to pin
    factor zero to the value identified directly by the historical Gaussian
    front scores, rather than allowing the DLV cross-section to overwrite the
    observed spot law.
    """
    loading_matrix = _coerce_loading_matrix(loading_matrix, target.shape)
    factor_count = loading_matrix.shape[-1]
    if fixed_market_price_of_risk is None:
        fixed = np.full(factor_count, np.nan, dtype=np.float64)
    else:
        fixed = np.asarray(fixed_market_price_of_risk, dtype=np.float64)
        if fixed.shape != (factor_count,) or np.any(np.isinf(fixed)):
            raise ValueError(
                "fixed market price of risk must contain finite values or NaN"
            )
    fixed_mask = np.isfinite(fixed)
    free_mask = ~fixed_mask
    risk_price = np.zeros(factor_count, dtype=np.float64)
    risk_price[fixed_mask] = fixed[fixed_mask]
    residual_target = target.copy()
    if np.any(fixed_mask):
        fixed_projection = np.einsum(
            "spf,f->sp",
            loading_matrix[..., fixed_mask],
            risk_price[fixed_mask],
        )
        residual_target -= fixed_projection
    if not np.any(free_mask):
        return risk_price

    free_loadings = loading_matrix[..., free_mask]
    gram = np.einsum(
        "spf,sp,spg->fg",
        free_loadings,
        weights,
        free_loadings,
    )
    right_hand_side = np.einsum(
        "spf,sp,sp->f",
        free_loadings,
        weights,
        residual_target,
    )
    free_count = int(np.count_nonzero(free_mask))
    ridge_scale = ridge * max(float(np.trace(gram)) / free_count, 1.0)
    risk_price[free_mask] = np.linalg.solve(
        gram + ridge_scale * np.eye(free_count),
        right_hand_side,
    )
    return risk_price


def _risk_price_standard_errors(
    *,
    loading_matrix: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    ridge: float,
    market_price_of_risk: FloatArray,
    fixed_market_price_of_risk: FloatArray | None,
) -> FloatArray:
    """Return date-clustered sandwich errors for the pooled risk price."""
    loading_matrix = _coerce_loading_matrix(loading_matrix, target.shape)
    factor_count = loading_matrix.shape[-1]
    if fixed_market_price_of_risk is None:
        fixed = np.full(factor_count, np.nan, dtype=np.float64)
    else:
        fixed = np.asarray(fixed_market_price_of_risk, dtype=np.float64)
    fixed_mask = np.isfinite(fixed)
    free_mask = ~fixed_mask
    standard_errors = np.zeros(factor_count, dtype=np.float64)
    if not np.any(free_mask):
        return standard_errors
    free_loadings = loading_matrix[..., free_mask]
    residual_target = target.copy()
    if np.any(fixed_mask):
        residual_target -= np.einsum(
            "spf,f->sp",
            loading_matrix[..., fixed_mask],
            market_price_of_risk[fixed_mask],
        )
    gram_by_date = np.einsum(
        "spf,sp,spg->sfg",
        free_loadings,
        weights,
        free_loadings,
    )
    right_by_date = np.einsum(
        "spf,sp,sp->sf",
        free_loadings,
        weights,
        residual_target,
    )
    free_risk_price = market_price_of_risk[free_mask]
    score_by_date = right_by_date - np.einsum(
        "sfg,g->sf",
        gram_by_date,
        free_risk_price,
    )
    score_by_date -= np.mean(score_by_date, axis=0, keepdims=True)
    gram = np.sum(gram_by_date, axis=0)
    free_count = int(np.count_nonzero(free_mask))
    ridge_scale = ridge * max(float(np.trace(gram)) / free_count, 1.0)
    inverse = np.linalg.inv(gram + ridge_scale * np.eye(free_count))
    meat = score_by_date.T @ score_by_date
    covariance = inverse @ meat @ inverse
    standard_errors[free_mask] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return standard_errors


def _state_loading_matrix(
    annualized_loadings: FloatArray,
    empirical_shape: tuple[int, ...],
) -> FloatArray:
    """Return loadings as ``sample x coordinate x factor``."""
    values = np.asarray(annualized_loadings, dtype=np.float64)
    sample_count = empirical_shape[0]
    coordinate_shape = empirical_shape[1:]
    if values.shape[:-1] == coordinate_shape:
        values = np.broadcast_to(values, (sample_count, *values.shape))
    elif values.shape[:-1] != empirical_shape:
        raise ValueError("historical drift loadings do not align")
    return values.reshape(sample_count, int(np.prod(coordinate_shape)), values.shape[-1])


def _coerce_loading_matrix(
    loading_matrix: FloatArray,
    target_shape: tuple[int, int],
) -> FloatArray:
    """Broadcast one fixed loading matrix across projection dates."""
    values = np.asarray(loading_matrix, dtype=np.float64)
    if values.ndim == 2 and values.shape[0] == target_shape[1]:
        values = np.broadcast_to(values, (target_shape[0], *values.shape))
    if values.ndim != 3 or values.shape[:2] != target_shape:
        raise ValueError("risk-price loading matrix does not align with targets")
    return values


def _weighted_rms(values: FloatArray, weights: FloatArray) -> float:
    flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
    flat_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight_sum = float(np.sum(flat_weights))
    if weight_sum <= np.finfo(np.float64).tiny:
        raise ValueError("weighted RMS requires positive weight mass")
    return float(np.sqrt(np.sum(flat_weights * flat_values**2) / weight_sum))
