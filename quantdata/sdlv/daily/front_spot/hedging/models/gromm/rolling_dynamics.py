"""Accepted rolling physical dynamics for the 200-strike GROMM2 model."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.gromm2.bridge import front_density_volatility
from quantdata.gromm2.dlv import decode_density_tower, risk_neutral_log_drift
from quantdata.gromm2.estimation import (
    ConstantDiffusionEstimate,
    SelectiveStateSpaceEstimate,
)
from quantdata.gromm2.market import solve_sanos_step_numpy
from quantdata.gromm2.simulation import localize_loadings_by_density
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_calibration import (
    RollingPhysicalDriftFit,
    fit_constant_physical_drift_from_inputs,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    FixedContractDlvHistory,
    RollingDlvFactorConfig,
    RollingDlvFactorFit,
    fit_dlv_factors,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

GROMM2_DYNAMICS_SCHEMA = "hedging-gromm2-estimate-v3"


def estimate_rolling_gromm2_dynamics(
    *,
    history_path: Path,
    dataset: Path,
    output: Path,
    as_of_date: date,
    origin_dates: tuple[date, ...],
    endpoint_dates: tuple[date, ...],
    factor_count: int,
    minimum_projection_fraction: float,
    maximum_daily_risk_price_norm: float,
    batch_size: int = 16,
) -> dict[str, object]:
    """Estimate exact-mixture factors and the validated compatible P change."""
    if not jax.config.x64_enabled:
        raise RuntimeError("rolling GROMM2 dynamics estimation requires JAX x64")
    if factor_count != 16:
        raise ValueError("the hedging GROMM2 dynamics require 16 factors")
    panel = PublicHedgingPanel(Path(dataset))
    arrays = _load_affine_history(history_path)
    history, origin_indices, origin_states = _build_fixed_contract_history(
        panel=panel,
        arrays=arrays,
        as_of_date=as_of_date,
        origin_dates=origin_dates,
        endpoint_dates=endpoint_dates,
    )
    active_mask = np.broadcast_to(
        (arrays["nms_nodes"] >= -2.5) & (arrays["nms_nodes"] <= 1.5),
        history.origin_log_dlvsq.shape[1:],
    )
    factor_config = RollingDlvFactorConfig(
        window_sessions=len(origin_dates),
        factor_rank=factor_count,
    )
    factors = fit_dlv_factors(
        history,
        as_of_date=endpoint_dates[-1],
        config=factor_config,
        active_coordinate_mask=active_mask,
    )
    validation_count = max(
        factor_count + 1,
        int(round(len(origin_dates) / 3.0)),
    )
    training_count = len(origin_dates) - validation_count
    selection_history = _history_prefix(history, training_count)
    selection_factors = fit_dlv_factors(
        selection_history,
        as_of_date=selection_history.endpoint_dates[-1],
        config=replace(factor_config, window_sessions=training_count),
        active_coordinate_mask=active_mask,
    )
    selection_q, selection_weights, selection_effective = _historical_q_inputs(
        states=origin_states,
        supports=arrays["supports"],
        transports=arrays["transports"],
        expiries=arrays["expiries"],
        annualized_loadings=selection_factors.annualized_loadings,
        batch_size=batch_size,
        progress_label="selection",
    )
    full_q, full_weights, full_effective = _historical_q_inputs(
        states=origin_states,
        supports=arrays["supports"],
        transports=arrays["transports"],
        expiries=arrays["expiries"],
        annualized_loadings=factors.annualized_loadings,
        batch_size=batch_size,
        progress_label="full",
    )
    selection_fixed = np.full(factor_count, np.nan, dtype=np.float64)
    selection_fixed[0] = selection_factors.front_spot_market_price_of_risk
    fixed = np.full(factor_count, np.nan, dtype=np.float64)
    fixed[0] = factors.front_spot_market_price_of_risk
    standard_error_overrides = np.full(factor_count, np.nan, dtype=np.float64)
    standard_error_overrides[0] = (
        factors.front_spot_market_price_of_risk_standard_error
    )
    physical = fit_constant_physical_drift_from_inputs(
        training_origin_dates=origin_dates,
        training_endpoint_dates=endpoint_dates,
        empirical_log_drifts=DAYS_PER_YEAR * history.increments,
        selection_risk_neutral_log_drifts=selection_q,
        selection_coordinate_weights=selection_weights,
        selection_annualized_loadings=selection_effective,
        risk_neutral_log_drifts=full_q,
        coordinate_weights=full_weights,
        annualized_loadings=full_effective,
        selection_fixed_market_price_of_risk=selection_fixed,
        fixed_market_price_of_risk=fixed,
        market_price_of_risk_standard_error_overrides=standard_error_overrides,
        ridge_candidates=(1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0),
        validation_fraction=1.0 / 3.0,
        maximum_daily_risk_price_norm=maximum_daily_risk_price_norm,
        minimum_validation_improvement_fraction=0.005,
        minimum_validation_gain_t=1.96,
        validation_hac_lags=5,
    )
    physical_accepted = bool(
        physical.accepted
        and physical.achievable_fraction >= minimum_projection_fraction
    )
    constant = _constant_diffusion(
        history=history,
        factors=factors,
        origin_indices=origin_indices,
        physical=physical,
    )
    selective = _unsupported_selective_state_space(constant)
    _write_dynamics(
        output,
        constant=constant,
        selective=selective,
        physical=physical,
        origin_dates=origin_dates,
        endpoint_dates=endpoint_dates,
        physical_accepted=physical_accepted,
    )
    report: dict[str, object] = {
        "schema": GROMM2_DYNAMICS_SCHEMA,
        "history_path": str(Path(history_path).resolve()),
        "dataset": str(Path(dataset).resolve()),
        "factor_count": factor_count,
        "pair_count": len(origin_dates),
        "explained_variance_ratio": constant.explained_variance_ratio,
        "physical_baseline": (
            "exact front mixture plus pooled realized fixed-contract transitions"
        ),
        "fixed_contract_interpolation_clip_fraction": float(
            np.mean(history.interpolation_clip_fractions)
        ),
        "spot_market_price_of_risk": factors.front_spot_market_price_of_risk,
        "spot_market_price_of_risk_standard_error": (
            factors.front_spot_market_price_of_risk_standard_error
        ),
        "physical_projection_fraction": physical.achievable_fraction,
        "physical_projection_metric": (
            "pooled_realized_transition_achievable_fraction"
        ),
        "physical_selected_ridge": physical.selected_ridge,
        "physical_validation_loss": physical.validation_loss,
        "physical_validation_baseline_loss": physical.validation_baseline_loss,
        "physical_validation_improvement_fraction": (
            physical.validation_improvement_fraction
        ),
        "physical_validation_gain": physical.validation_gain,
        "physical_validation_gain_standard_error": (
            physical.validation_gain_standard_error
        ),
        "physical_validation_gain_t": physical.validation_gain_t,
        "physical_daily_risk_price_norm": physical.daily_risk_price_norm,
        "physical_validation_accepted": physical.accepted,
        "physical_accepted": physical_accepted,
        "selective_state_space": {
            "architecture": "constant unit factor amplitudes",
            "supported": False,
            "reason": (
                "partial front observation requires posterior second moments "
                "for a valid selective likelihood"
            ),
            "hidden_size": selective.hidden_size,
            "log_scale_bound": selective.log_scale_bound,
            "epochs": selective.epochs,
            "accepted": selective.accepted,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _load_affine_history(path: Path) -> dict[str, FloatArray | tuple[date, ...]]:
    with h5py.File(path, "r") as stream:
        if stream.attrs.get("schema") != "gromm2-affine-dlv-history-v1":
            raise ValueError("unexpected rolling GROMM2 affine-history schema")
        return {
            "dates": tuple(
                date.fromisoformat(value)
                for value in np.asarray(stream["dates"]).astype("U10").tolist()
            ),
            "supports": np.asarray(stream["supports"], dtype=np.float64),
            "transports": np.asarray(stream["transports"], dtype=np.float64),
            "expiries": np.asarray(stream["model_expiries_y"], dtype=np.float64),
            "business_dtes": np.asarray(
                stream["model_business_dtes"], dtype=np.float64
            ),
            "nms_nodes": np.asarray(stream["nms_nodes"], dtype=np.float64),
            "dlvsq": np.asarray(stream["dlvsq"], dtype=np.float64),
        }


def _build_fixed_contract_history(
    *,
    panel: PublicHedgingPanel,
    arrays: dict[str, FloatArray | tuple[date, ...]],
    as_of_date: date,
    origin_dates: tuple[date, ...],
    endpoint_dates: tuple[date, ...],
) -> tuple[FixedContractDlvHistory, IntArray, FloatArray]:
    dates = arrays["dates"]
    assert isinstance(dates, tuple)
    supports = np.asarray(arrays["supports"], dtype=np.float64)
    expiries = np.asarray(arrays["expiries"], dtype=np.float64)
    business_dtes = np.asarray(arrays["business_dtes"], dtype=np.float64)
    dlvsq = np.asarray(arrays["dlvsq"], dtype=np.float64)
    by_date = {value: index for index, value in enumerate(dates)}
    expected_dates = tuple(
        sorted(set(origin_dates) | set(endpoint_dates) | {as_of_date})
    )
    if expected_dates != dates:
        raise ValueError(
            "affine states do not equal the transition-date union plus as-of"
        )
    origin_indices: list[int] = []
    origin_values: list[FloatArray] = []
    endpoint_values: list[FloatArray] = []
    front_densities: list[FloatArray] = []
    terminal_spots: list[float] = []
    front_cdfs: list[float] = []
    clip_fractions: list[float] = []
    forward_tail_fractions: list[float] = []
    for origin_date, endpoint_date in zip(
        origin_dates,
        endpoint_dates,
        strict=True,
    ):
        transition = panel.one_day_transition(origin_date)
        if transition.next_date != endpoint_date:
            raise ValueError("manifest transition is not one released session")
        origin_index = by_date[origin_date]
        endpoint_index = by_date[endpoint_date]
        origin_indices.append(origin_index)
        origin_values.append(np.log(dlvsq[origin_index, 1:]))
        aligned, clipped, forward_tail_fraction = _fixed_contract_endpoint_surface(
            endpoint_log_dlvsq=np.log(dlvsq[endpoint_index]),
            supports=supports,
            reference_business_dtes=business_dtes,
            origin_day=transition.origin,
            endpoint_day=panel.market_day(endpoint_date),
            gross_spot_return=(
                transition.forward_over_spot * transition.terminal_normalized_spot
            ),
        )
        endpoint_values.append(aligned)
        clip_fractions.append(clipped / aligned.size)
        forward_tail_fractions.append(forward_tail_fraction)
        density = _front_density(
            supports[0],
            dlvsq[origin_index, 0],
            float(expiries[0]),
        )
        front_densities.append(density)
        terminal_spots.append(transition.terminal_normalized_spot)
        front_cdfs.append(
            _front_cdf(
                support=supports[0],
                density=density,
                terminal_spot=transition.terminal_normalized_spot,
                background_annual_volatility=(
                    transition.origin.backbone_annual_volatility
                ),
            )
        )
    front_cdf_values = np.clip(
        np.asarray(front_cdfs),
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    history = FixedContractDlvHistory(
        origin_dates=origin_dates,
        endpoint_dates=endpoint_dates,
        origin_log_dlvsq=np.stack(origin_values),
        endpoint_log_dlvsq=np.stack(endpoint_values),
        front_cdf_values=front_cdf_values,
        spot_gaussian_scores=ndtri(front_cdf_values),
        interpolation_clip_fractions=np.asarray(clip_fractions),
        forward_tail_extrapolation_fractions=np.asarray(forward_tail_fractions),
        front_supports=np.broadcast_to(
            supports[0],
            (len(origin_dates), supports.shape[1]),
        ).copy(),
        front_densities=np.stack(front_densities),
        terminal_normalized_spots=np.asarray(terminal_spots),
        background_annual_volatilities=np.asarray(
            [
                panel.market_day(value).backbone_annual_volatility
                for value in origin_dates
            ]
        ),
    )
    return history, np.asarray(origin_indices, dtype=np.int64), dlvsq[origin_indices]


def _fixed_contract_endpoint_surface(
    *,
    endpoint_log_dlvsq: FloatArray,
    supports: FloatArray,
    reference_business_dtes: FloatArray,
    origin_day: PublicMarketDay,
    endpoint_day: PublicMarketDay,
    gross_spot_return: float,
) -> tuple[FloatArray, int, float]:
    target_dtes = reference_business_dtes[1:] - 1.0
    result = np.empty_like(endpoint_log_dlvsq[1:])
    clipped = 0
    forward_tail_extrapolations = 0
    for row, target_dte in enumerate(target_dtes):
        upper = min(
            int(np.searchsorted(reference_business_dtes, target_dte, side="left")),
            len(reference_business_dtes) - 1,
        )
        lower = max(upper - 1, 0)
        if reference_business_dtes[upper] == target_dte:
            lower = upper
        lower_dte = float(reference_business_dtes[lower])
        upper_dte = float(reference_business_dtes[upper])
        weight = (
            0.0
            if lower == upper
            else (target_dte - lower_dte) / (upper_dte - lower_dte)
        )
        origin_forward, origin_extrapolated = _forward_over_spot(
            origin_day,
            target_business_dte=float(reference_business_dtes[row + 1]),
        )
        endpoint_forward, endpoint_extrapolated = _forward_over_spot(
            endpoint_day,
            target_business_dte=float(target_dte),
        )
        forward_tail_extrapolations += int(origin_extrapolated)
        forward_tail_extrapolations += int(endpoint_extrapolated)
        query = (
            supports[row + 1, 1:-1]
            * origin_forward
            / (gross_spot_return * endpoint_forward)
        )
        lower_support = supports[lower, 1:-1]
        upper_support = supports[upper, 1:-1]
        clipped += int(
            np.count_nonzero(
                (query < lower_support[0])
                | (query > lower_support[-1])
                | (query < upper_support[0])
                | (query > upper_support[-1])
            )
        )
        lower_values = np.interp(
            query,
            lower_support,
            endpoint_log_dlvsq[lower],
        )
        upper_values = np.interp(
            query,
            upper_support,
            endpoint_log_dlvsq[upper],
        )
        result[row] = (1.0 - weight) * lower_values + weight * upper_values
    return (
        result,
        clipped,
        forward_tail_extrapolations / (2.0 * len(target_dtes)),
    )


def _forward_over_spot(
    day: PublicMarketDay,
    *,
    target_business_dte: float,
) -> tuple[float, bool]:
    """Interpolate released forwards and hold the terminal ratio flat."""
    if target_business_dte > day.business_dtes[-1] + 1.0e-8:
        return float(day.forward_over_spot[-1]), True
    return day.forward_at_business_dte(int(round(target_business_dte))), False


def _front_density(
    support: FloatArray,
    dlvsq: FloatArray,
    expiry_y: float,
) -> FloatArray:
    calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(calls, prepend=1.0) / np.diff(support, prepend=0.0)
    initial = np.diff(slopes, append=0.0)
    density = solve_sanos_step_numpy(support, initial, dlvsq * expiry_y)
    density = np.maximum(density, 0.0)
    return density / np.sum(density)


def _front_cdf(
    *,
    support: FloatArray,
    density: FloatArray,
    terminal_spot: float,
    background_annual_volatility: float,
) -> float:
    variance = background_annual_volatility**2 / DAYS_PER_YEAR
    z = np.full_like(support, np.inf)
    positive = support > 0.0
    z[positive] = (
        np.log(terminal_spot / support[positive]) + 0.5 * variance
    ) / np.sqrt(variance)
    return float(density @ ndtr(z))


def _history_prefix(
    history: FixedContractDlvHistory,
    count: int,
) -> FixedContractDlvHistory:
    def prefix(values: FloatArray | None) -> FloatArray | None:
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
        front_supports=prefix(history.front_supports),
        front_densities=prefix(history.front_densities),
        terminal_normalized_spots=prefix(history.terminal_normalized_spots),
        background_annual_volatilities=prefix(
            history.background_annual_volatilities
        ),
    )


def _historical_q_inputs(
    *,
    states: FloatArray,
    supports: FloatArray,
    transports: FloatArray,
    expiries: FloatArray,
    annualized_loadings: FloatArray,
    batch_size: int,
    progress_label: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    intervals = np.diff(expiries, prepend=0.0)
    factor_count = annualized_loadings.shape[-1]

    @jax.jit
    def evaluate(batch: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        tower = decode_density_tower(
            jnp.asarray(supports, dtype=jnp.float64),
            jnp.asarray(transports, dtype=jnp.float64),
            batch,
            jnp.asarray(intervals, dtype=jnp.float64),
        )
        future_densities = tower[..., 1:, :]
        effective = localize_loadings_by_density(
            jnp.asarray(annualized_loadings, dtype=jnp.float64),
            future_densities,
        )
        front_loading = jnp.zeros(
            tower[..., 0, :].shape + (factor_count,),
            dtype=batch.dtype,
        )
        front_loading = front_loading.at[..., 0].set(
            front_density_volatility(
                tower[..., 0, :],
                remaining_time=jnp.asarray(expiries[0], dtype=jnp.float64),
            )
        )
        _, _, q_log_drift = risk_neutral_log_drift(
            jnp.asarray(supports, dtype=jnp.float64),
            jnp.asarray(transports, dtype=jnp.float64),
            tower[..., 0, :],
            batch[..., 1:, :],
            jnp.asarray(intervals[1:], dtype=jnp.float64),
            effective,
            front_loading,
        )
        interior = jnp.maximum(future_densities[..., 1:-1], 0.0)
        weights = interior / jnp.maximum(
            jnp.sum(interior, axis=-1, keepdims=True),
            jnp.finfo(jnp.float64).tiny,
        )
        return q_log_drift, weights, effective

    q_blocks: list[FloatArray] = []
    weight_blocks: list[FloatArray] = []
    loading_blocks: list[FloatArray] = []
    started = time.perf_counter()
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        q_drift, weights, effective = evaluate(
            jnp.asarray(states[start:stop], dtype=jnp.float64)
        )
        q_blocks.append(np.asarray(q_drift))
        weight_blocks.append(np.asarray(weights))
        loading_blocks.append(np.asarray(effective))
        if stop == len(states) or stop % (5 * batch_size) == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(states) - stop) / stop
            print(
                json.dumps(
                    {
                        "stage": f"gromm2-{progress_label}-q-drift",
                        "completed_pairs": stop,
                        "total_pairs": len(states),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    }
                ),
                flush=True,
            )
    return (
        np.concatenate(q_blocks),
        np.concatenate(weight_blocks),
        np.concatenate(loading_blocks),
    )


def _constant_diffusion(
    *,
    history: FixedContractDlvHistory,
    factors: RollingDlvFactorFit,
    origin_indices: IntArray,
    physical: RollingPhysicalDriftFit,
) -> ConstantDiffusionEstimate:
    levels = history.origin_log_dlvsq.reshape(len(origin_indices), -1)
    level_mean = np.mean(levels, axis=0)
    centered = levels - level_mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    level_rank = min(8, len(singular))
    level_basis = right[:level_rank].T
    level_scores = centered @ level_basis
    level_scale = np.maximum(np.std(level_scores, axis=0, ddof=1), 1.0e-8)
    coefficients = np.zeros(
        (1 + level_rank + factors.factor_rank, factors.factor_rank),
        dtype=np.float64,
    )
    coefficients[0] = physical.market_price_of_risk
    return ConstantDiffusionEstimate(
        log_dlvsq_loadings=factors.annualized_loadings,
        factor_scores=factors.factor_scores,
        physical_ar_intercept=factors.conditional_intercept,
        physical_ar_slope=factors.conditional_slope,
        explained_variance_ratio=factors.explained_variance_fraction,
        level_mean=level_mean.reshape(history.origin_log_dlvsq.shape[1:]),
        level_basis=level_basis,
        level_scale=level_scale,
        physical_factor_coefficients=coefficients,
        physical_market_prices_of_risk=np.broadcast_to(
            physical.market_price_of_risk,
            (len(origin_indices), factors.factor_rank),
        ).copy(),
        spot_market_price_of_risk=float(physical.market_price_of_risk[0]),
        physical_projection_fraction=physical.achievable_fraction,
        pair_origin_indices=origin_indices,
    )


def _unsupported_selective_state_space(
    constant: ConstantDiffusionEstimate,
) -> SelectiveStateSpaceEstimate:
    """Serialize a neutral constant-scale placeholder for the unsupported SSM."""
    input_size = constant.level_basis.shape[1] + constant.factor_scores.shape[1]
    factor_count = constant.factor_scores.shape[1]
    hidden_size = 1
    parameters = {
        "decay_weight": np.zeros((input_size, hidden_size)),
        "decay_bias": np.zeros(hidden_size),
        "candidate_weight": np.zeros((input_size, hidden_size)),
        "candidate_bias": np.zeros(hidden_size),
        "output_weight": np.zeros((hidden_size, factor_count)),
        "output_bias": np.zeros(factor_count),
    }
    return SelectiveStateSpaceEstimate(
        parameters=parameters,
        terminal_hidden=np.zeros(hidden_size),
        input_mean=np.zeros(input_size),
        input_scale=np.ones(input_size),
        hidden_size=hidden_size,
        log_scale_bound=0.0,
        train_negative_log_likelihood=0.0,
        validation_negative_log_likelihood=0.0,
        constant_validation_negative_log_likelihood=0.0,
        accepted=False,
        epochs=0,
    )
def _write_dynamics(
    output: Path,
    *,
    constant: ConstantDiffusionEstimate,
    selective: SelectiveStateSpaceEstimate,
    physical: RollingPhysicalDriftFit,
    origin_dates: tuple[date, ...],
    endpoint_dates: tuple[date, ...],
    physical_accepted: bool,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "log_dlvsq_loadings": constant.log_dlvsq_loadings.astype(np.float32),
        "factor_scores": constant.factor_scores.astype(np.float32),
        "physical_ar_intercept": constant.physical_ar_intercept.astype(np.float32),
        "physical_ar_slope": constant.physical_ar_slope.astype(np.float32),
        "level_mean": constant.level_mean.astype(np.float32),
        "level_basis": constant.level_basis.astype(np.float32),
        "level_scale": constant.level_scale.astype(np.float32),
        "physical_factor_coefficients": (
            constant.physical_factor_coefficients.astype(np.float32)
        ),
        "physical_market_price_of_risk": (
            physical.market_price_of_risk.astype(np.float64)
        ),
        "physical_market_price_of_risk_standard_errors": (
            physical.market_price_of_risk_standard_errors.astype(np.float64)
        ),
        "spot_market_price_of_risk": np.asarray(
            physical.market_price_of_risk[0], dtype=np.float64
        ),
        "pair_origin_indices": constant.pair_origin_indices,
        "training_origin_dates": np.asarray(
            [value.isoformat() for value in origin_dates], dtype="S10"
        ),
        "training_endpoint_dates": np.asarray(
            [value.isoformat() for value in endpoint_dates], dtype="S10"
        ),
        "physical_selected_ridge": np.asarray(physical.selected_ridge),
        "physical_daily_risk_price_norm": np.asarray(
            physical.daily_risk_price_norm
        ),
        "physical_projection_fraction": np.asarray(
            physical.achievable_fraction
        ),
        "physical_validation_improvement_fraction": np.asarray(
            physical.validation_improvement_fraction
        ),
        "physical_validation_gain_t": np.asarray(physical.validation_gain_t),
        "physical_accepted": np.asarray(physical_accepted),
        "ssm_input_mean": selective.input_mean.astype(np.float32),
        "ssm_input_scale": selective.input_scale.astype(np.float32),
        "ssm_terminal_hidden": selective.terminal_hidden.astype(np.float32),
    }
    for name, value in selective.parameters.items():
        arrays[f"ssm_{name}"] = np.asarray(value, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
