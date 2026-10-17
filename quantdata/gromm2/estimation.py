"""Estimate constant and selective state-space GROMM2 diffusions."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr, ndtri

from quantdata.gromm2.bridge import front_density_volatility
from quantdata.gromm2.dlv import decode_density_tower, risk_neutral_log_drift
from quantdata.gromm2.heston_background_pilot import (
    HestonBackground,
    heston_return_cdf_grid_fft,
)
from quantdata.gromm2.market import solve_sanos_step_numpy

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class ConstantDiffusionEstimate:
    """Low-rank joint spot/log-DLV diffusion and physical drift fit."""

    log_dlvsq_loadings: FloatArray
    factor_scores: FloatArray
    physical_ar_intercept: FloatArray
    physical_ar_slope: FloatArray
    explained_variance_ratio: float
    level_mean: FloatArray
    level_basis: FloatArray
    level_scale: FloatArray
    physical_factor_coefficients: FloatArray
    physical_market_prices_of_risk: FloatArray
    spot_market_price_of_risk: float
    physical_projection_fraction: float
    pair_origin_indices: NDArray[np.int64]


@dataclass(frozen=True, kw_only=True)
class SelectiveStateSpaceEstimate:
    """Bounded factor-amplitude model over fixed spatial loading fields."""

    parameters: dict[str, FloatArray]
    terminal_hidden: FloatArray
    input_mean: FloatArray
    input_scale: FloatArray
    hidden_size: int
    log_scale_bound: float
    train_negative_log_likelihood: float
    validation_negative_log_likelihood: float
    constant_validation_negative_log_likelihood: float
    accepted: bool
    epochs: int


def _front_density(
    support: FloatArray,
    dlvsq: FloatArray,
    expiry_y: float,
) -> FloatArray:
    calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(calls, prepend=1.0) / np.diff(support, prepend=0.0)
    initial = np.diff(slopes, append=0.0)
    return solve_sanos_step_numpy(support, initial, dlvsq * expiry_y)


def _spot_gaussian_score(
    *,
    normalized_spot: float,
    support: FloatArray,
    density: FloatArray,
    expiry_y: float,
    backbone_annual_volatility: float,
    background_cdf_grid: tuple[FloatArray, FloatArray] | None = None,
) -> float:
    if background_cdf_grid is not None:
        log_grid, cumulative_grid = background_cdf_grid
        log_ratios = np.log(normalized_spot / support)
        if np.min(log_ratios) < log_grid[0] or np.max(log_ratios) > log_grid[-1]:
            raise ValueError("front-spot ratios exceed the Heston CDF grid")
        conditional_cumulative = np.interp(
            log_ratios,
            log_grid,
            cumulative_grid,
        )
        cumulative = float(density @ conditional_cumulative)
    else:
        variance = backbone_annual_volatility**2 * expiry_y
        standard_deviation = math.sqrt(variance)
        if standard_deviation == 0.0:
            cumulative = float(np.sum(density[support <= normalized_spot]))
        else:
            z = (
                np.log(normalized_spot / support) + 0.5 * variance
            ) / standard_deviation
            cumulative = float(density @ ndtr(z))
    probability = float(np.clip(cumulative, 1.0e-7, 1.0 - 1.0e-7))
    return float(ndtri(probability))


def _positive_curve_value(
    business_dtes: FloatArray,
    values: FloatArray,
    target_business_dte: float,
) -> float:
    """Log-linearly interpolate a positive curve with flat tail extrapolation."""
    tenors = np.asarray(business_dtes, dtype=np.float64)
    curve = np.asarray(values, dtype=np.float64)
    if (
        tenors.ndim != 1
        or curve.shape != tenors.shape
        or np.any(np.diff(tenors) <= 0.0)
        or np.any(curve <= 0.0)
    ):
        raise ValueError("positive curve inputs are invalid")
    return float(
        np.exp(
            np.interp(
                target_business_dte,
                tenors,
                np.log(curve),
                left=np.log(curve[0]),
                right=np.log(curve[-1]),
            )
        )
    )


def _fixed_contract_endpoint_surface(
    *,
    endpoint_log_dlvsq: FloatArray,
    supports: FloatArray,
    reference_business_dtes: FloatArray,
    origin_forward_over_spot: FloatArray,
    endpoint_forward_over_spot: FloatArray,
    gross_spot_return: float,
) -> tuple[FloatArray, int]:
    """Align tomorrow's surface to today's fixed expiries and cash strikes."""
    origin_dtes = np.asarray(reference_business_dtes[1:], dtype=np.float64)
    target_dtes = origin_dtes - 1.0
    result = np.empty_like(endpoint_log_dlvsq[1:])
    clipped = 0
    for row, target_dte in enumerate(target_dtes):
        upper = int(np.searchsorted(reference_business_dtes, target_dte, side="left"))
        upper = min(upper, len(reference_business_dtes) - 1)
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
        endpoint_forward = _positive_curve_value(
            reference_business_dtes,
            endpoint_forward_over_spot,
            target_dte,
        )
        query = (
            supports[row + 1, 1:-1]
            * origin_forward_over_spot[row]
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
    return result, clipped


def load_estimation_panel(
    *,
    history_path: Path,
    dataset: Path,
    backbone_annual_volatility: float,
) -> dict[str, np.ndarray]:
    """Load exact one-session pairs and derive front-distribution spot scores."""
    with h5py.File(history_path, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        dlvsq = np.asarray(stream["dlvsq"], dtype=np.float64)
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        nms_nodes = np.asarray(stream["nms_nodes"], dtype=np.float64)
        call_kernel_metadata = stream.attrs.get("call_kernel_metadata")
    background_cdf_grid = None
    if call_kernel_metadata is not None:
        metadata = json.loads(str(call_kernel_metadata))
        if metadata.get("model") == "independent-Heston":
            background = HestonBackground(**metadata["background"])
            background_cdf_grid = heston_return_cdf_grid_fft(
                background,
                expiry_y=float(expiries[0]),
            )
    observations = pl.read_parquet(
        dataset / "observations.parquet",
        columns=[
            "date",
            "business_sessions_from_previous_observation",
            "forward_carry_from_previous_observation",
            "forward_adjusted_spot_return_into_date",
        ],
    )
    expiry_rows = pl.read_parquet(
        dataset / "expiries.parquet",
        columns=["date", "business_dtes", "forward_over_spot"],
    )
    observation_map = {
        row["date"].isoformat(): row
        for row in observations.iter_rows(named=True)
    }
    expiry_map = {
        row["date"].isoformat(): row
        for row in expiry_rows.iter_rows(named=True)
    }
    origin_indices: list[int] = []
    spot_scores: list[float] = []
    aligned_origins: list[FloatArray] = []
    aligned_endpoints: list[FloatArray] = []
    clipped_queries = 0
    total_queries = 0
    log_dlvsq = np.log(dlvsq)
    reference_business_dtes = 252.0 * expiries
    for endpoint in range(1, len(dates)):
        current = observation_map[str(dates[endpoint])]
        if current["business_sessions_from_previous_observation"] != 1:
            continue
        origin_expiries = expiry_map[str(dates[endpoint - 1])]
        endpoint_expiries = expiry_map[str(dates[endpoint])]
        origin_forward = np.asarray(
            [
                _positive_curve_value(
                    np.asarray(origin_expiries["business_dtes"], dtype=np.float64),
                    np.asarray(origin_expiries["forward_over_spot"], dtype=np.float64),
                    float(target),
                )
                for target in reference_business_dtes[1:]
            ],
            dtype=np.float64,
        )
        endpoint_forward = np.asarray(
            [
                _positive_curve_value(
                    np.asarray(endpoint_expiries["business_dtes"], dtype=np.float64),
                    np.asarray(endpoint_expiries["forward_over_spot"], dtype=np.float64),
                    float(target),
                )
                for target in reference_business_dtes
            ],
            dtype=np.float64,
        )
        gross_spot_return = (
            float(current["forward_carry_from_previous_observation"])
            * (1.0 + float(current["forward_adjusted_spot_return_into_date"]))
        )
        aligned_endpoint, clipped = _fixed_contract_endpoint_surface(
            endpoint_log_dlvsq=log_dlvsq[endpoint],
            supports=supports,
            reference_business_dtes=reference_business_dtes,
            origin_forward_over_spot=origin_forward,
            endpoint_forward_over_spot=endpoint_forward,
            gross_spot_return=gross_spot_return,
        )
        aligned_origins.append(log_dlvsq[endpoint - 1, 1:])
        aligned_endpoints.append(aligned_endpoint)
        clipped_queries += clipped
        total_queries += aligned_endpoint.size
        normalized_spot = 1.0 + float(current["forward_adjusted_spot_return_into_date"])
        density = _front_density(
            supports[0],
            dlvsq[endpoint - 1, 0],
            float(expiries[0]),
        )
        spot_scores.append(
            _spot_gaussian_score(
                normalized_spot=normalized_spot,
                support=supports[0],
                density=density,
                expiry_y=float(expiries[0]),
                backbone_annual_volatility=backbone_annual_volatility,
                background_cdf_grid=background_cdf_grid,
            )
        )
        origin_indices.append(endpoint - 1)
    origins = np.asarray(origin_indices, dtype=np.int64)
    origin_values = np.stack(aligned_origins)
    endpoint_values = np.stack(aligned_endpoints)
    return {
        "dates": dates,
        "supports": supports,
        "transports": transports,
        "expiries": expiries,
        "nms_nodes": nms_nodes,
        "dlvsq": dlvsq,
        "reference_business_dtes": reference_business_dtes[1:],
        "origin_indices": origins,
        "log_dlvsq_origins": origin_values,
        "log_dlvsq_changes": endpoint_values - origin_values,
        "spot_scores": np.asarray(spot_scores, dtype=np.float64),
        "fixed_contract_interpolation_clip_fraction": np.asarray(
            clipped_queries / total_queries,
            dtype=np.float64,
        ),
    }


def fit_constant_diffusion(
    panel: dict[str, np.ndarray],
    *,
    factor_count: int = 16,
    level_rank: int = 8,
    dt: float = 1.0 / 252.0,
    ridge: float = 1.0e-3,
    smoothing_expiry_sigma: float = 1.0,
    smoothing_strike_sigma: float = 2.0,
) -> ConstantDiffusionEstimate:
    """Estimate the DYSANOS AR(1), smoothed innovations and joint factors."""
    changes = np.asarray(panel["log_dlvsq_changes"], dtype=np.float64)
    spot_scores = np.asarray(panel["spot_scores"], dtype=np.float64)
    origins = np.asarray(panel["log_dlvsq_origins"], dtype=np.float64)
    endpoints = origins + changes
    observations = len(changes)
    coordinates = int(np.prod(changes.shape[1:]))
    if observations <= factor_count + 2:
        raise ValueError("too few one-day pairs for the requested factor count")

    origin_mean = np.mean(origins, axis=0)
    endpoint_mean = np.mean(endpoints, axis=0)
    centered_origin = origins - origin_mean
    centered_endpoint = endpoints - endpoint_mean
    denominator = np.sum(centered_origin**2, axis=0)
    scale = np.maximum(np.sum(origins**2, axis=0), 1.0)
    variable = denominator > np.finfo(np.float64).eps * scale * observations
    ar_slope = np.zeros_like(denominator)
    ar_slope[variable] = (
        np.sum(centered_origin * centered_endpoint, axis=0)[variable]
        / denominator[variable]
    )
    ar_intercept = endpoint_mean - ar_slope * origin_mean
    innovations = endpoints - (ar_intercept + ar_slope * origins)
    innovations -= np.mean(innovations, axis=0, keepdims=True)
    nms_nodes = np.asarray(panel.get("nms_nodes"), dtype=np.float64)
    if nms_nodes.shape != changes.shape[-1:]:
        raise ValueError("NMS nodes must align with the DLV strike coordinates")
    core = (nms_nodes >= -2.5) & (nms_nodes <= 1.5)
    smoothed = gaussian_filter(
        innovations,
        sigma=(0.0, smoothing_expiry_sigma, smoothing_strike_sigma),
        mode="nearest",
    )
    smoothed -= np.mean(smoothed, axis=0, keepdims=True)
    smoothed[..., ~core] = 0.0
    scaled = smoothed.reshape(observations, coordinates) / math.sqrt(dt)
    spot_scores = (spot_scores - np.mean(spot_scores)) / np.std(
        spot_scores, ddof=0
    )
    spot_loading = spot_scores @ scaled / (spot_scores @ spot_scores)
    residual = scaled - spot_scores[:, None] * spot_loading[None, :]
    left, singular, right_transpose = np.linalg.svd(residual, full_matrices=False)
    residual_rank = factor_count - 1
    residual_loadings = (
        right_transpose[:residual_rank].T
        * singular[:residual_rank][None, :]
        / math.sqrt(observations - 1)
    )
    residual_scores = left[:, :residual_rank] * math.sqrt(observations - 1)
    loadings = np.concatenate((spot_loading[:, None], residual_loadings), axis=1)
    factor_scores = np.concatenate((spot_scores[:, None], residual_scores), axis=1)
    loadings = _restrict_loadings_to_fitted_range(
        loadings.reshape(changes.shape[1:] + (factor_count,)),
        nms_nodes=nms_nodes,
    ).reshape(coordinates, factor_count)
    core_coordinates = np.broadcast_to(core, changes.shape[1:]).reshape(-1)
    total_variance = float(
        np.mean(np.sum(scaled[:, core_coordinates] ** 2, axis=1))
    )
    explained_variance = float(
        np.mean(
            np.sum(
                (factor_scores @ loadings[core_coordinates].T) ** 2,
                axis=1,
            )
        )
    )

    flat_levels = origins.reshape(observations, coordinates)
    level_mean = np.mean(flat_levels, axis=0)
    centered_levels = flat_levels - level_mean
    _, level_singular, level_right = np.linalg.svd(centered_levels, full_matrices=False)
    retained_level_rank = min(level_rank, len(level_singular))
    level_basis = level_right[:retained_level_rank].T
    level_scores = centered_levels @ level_basis
    level_scale = np.std(level_scores, axis=0, ddof=1)
    level_scale = np.maximum(level_scale, 1.0e-8)
    standardized_levels = level_scores / level_scale
    lagged_scores = np.vstack((np.zeros((1, factor_count)), factor_scores[:-1]))
    pair_indices = np.asarray(panel["origin_indices"], dtype=np.int64)
    is_contiguous = np.concatenate(([False], np.diff(pair_indices) == 1))
    lagged_scores[~is_contiguous] = 0.0
    features = np.concatenate(
        (
            np.ones((observations, 1)),
            standardized_levels,
            lagged_scores,
        ),
        axis=1,
    )
    gram = features.T @ features
    penalty = ridge * np.trace(gram) / len(gram)
    regularizer = penalty * np.eye(len(gram))
    regularizer[0, 0] = 0.0
    if {
        "dlvsq",
        "supports",
        "transports",
        "expiries",
    }.issubset(panel):
        (
            market_prices_of_risk,
            spot_market_price_of_risk,
            projection_fraction,
        ) = _fit_compatible_physical_risk_prices(
            panel=panel,
            log_dlvsq_loadings=loadings.reshape(
                changes.shape[1:] + (factor_count,)
            ),
            physical_ar_intercept=ar_intercept,
            physical_ar_slope=ar_slope,
            dt=dt,
            ridge=ridge,
        )
    else:
        market_prices_of_risk = np.zeros((observations, factor_count))
        spot_market_price_of_risk = float(np.mean(panel["spot_scores"]) / math.sqrt(dt))
        projection_fraction = float("nan")
    coefficients = np.linalg.solve(
        gram + regularizer,
        features.T @ market_prices_of_risk,
    )
    return ConstantDiffusionEstimate(
        log_dlvsq_loadings=loadings.reshape(changes.shape[1:] + (factor_count,)),
        factor_scores=factor_scores,
        physical_ar_intercept=ar_intercept,
        physical_ar_slope=ar_slope,
        explained_variance_ratio=explained_variance / total_variance,
        level_mean=level_mean.reshape(changes.shape[1:]),
        level_basis=level_basis,
        level_scale=level_scale,
        physical_factor_coefficients=coefficients,
        physical_market_prices_of_risk=market_prices_of_risk,
        spot_market_price_of_risk=spot_market_price_of_risk,
        physical_projection_fraction=projection_fraction,
        pair_origin_indices=pair_indices,
    )


def _restrict_loadings_to_fitted_range(
    loadings: FloatArray,
    *,
    nms_nodes: FloatArray,
    fit_range: tuple[float, float] = (-2.5, 1.5),
) -> FloatArray:
    """Keep stochastic DLV dynamics only on the quote-fitted strike range."""
    result = np.asarray(loadings, dtype=np.float64).copy()
    nodes = np.asarray(nms_nodes, dtype=np.float64)
    core = (nodes >= fit_range[0]) & (nodes <= fit_range[1])
    if not np.any(core):
        raise ValueError("fit range does not intersect the NMS grid")
    result[..., ~core, :] = 0.0
    return result


def _density_localized_loadings(
    log_dlvsq_loadings: jax.Array,
    future_densities: jax.Array,
    *,
    relative_floor: float = 1.0e-4,
) -> jax.Array:
    interior = jnp.maximum(future_densities[..., 1:-1], 0.0)
    floor = relative_floor * jnp.max(interior, axis=-1, keepdims=True)
    weight = interior / (interior + floor)
    return log_dlvsq_loadings * weight[..., None]


def _fit_compatible_physical_risk_prices(
    *,
    panel: dict[str, np.ndarray],
    log_dlvsq_loadings: FloatArray,
    physical_ar_intercept: FloatArray,
    physical_ar_slope: FloatArray,
    dt: float,
    ridge: float,
    batch_size: int = 32,
) -> tuple[FloatArray, float, float]:
    """Project the empirical AR drift onto one equivalent-measure factor span."""
    supports = np.asarray(panel["supports"], dtype=np.float64)
    transports = np.asarray(panel["transports"], dtype=np.float64)
    expiries = np.asarray(panel["expiries"], dtype=np.float64)
    intervals = np.diff(expiries, prepend=0.0)
    indices = np.asarray(panel["origin_indices"], dtype=np.int64)
    states = np.asarray(panel["dlvsq"], dtype=np.float64)[indices]
    origins = np.asarray(panel["log_dlvsq_origins"], dtype=np.float64)
    empirical_log_drift = (
        physical_ar_intercept
        + physical_ar_slope * origins
        - origins
    ) / dt
    loadings = jnp.asarray(log_dlvsq_loadings, dtype=jnp.float64)
    factor_count = log_dlvsq_loadings.shape[-1]
    spot_market_price_of_risk = float(
        np.mean(np.asarray(panel["spot_scores"], dtype=np.float64)) / math.sqrt(dt)
    )

    @jax.jit
    def risk_neutral_batch(
        dlvsq: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        tower = decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            dlvsq,
            jnp.asarray(intervals),
        )
        future_densities = tower[..., 1:, :]
        effective = _density_localized_loadings(loadings, future_densities)
        front_loading = jnp.zeros(
            tower[..., 0, :].shape + (factor_count,),
            dtype=dlvsq.dtype,
        )
        front_loading = front_loading.at[..., 0].set(
            front_density_volatility(
                tower[..., 0, :],
                remaining_time=jnp.asarray(expiries[0]),
            )
        )
        _, _, q_log_drift = risk_neutral_log_drift(
            jnp.asarray(supports),
            jnp.asarray(transports),
            tower[..., 0, :],
            dlvsq[..., 1:, :],
            jnp.asarray(intervals[1:]),
            effective,
            front_loading,
        )
        return q_log_drift, future_densities

    q_drift_blocks: list[FloatArray] = []
    density_blocks: list[FloatArray] = []
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        q_drift, densities = risk_neutral_batch(
            jnp.asarray(states[start:stop], dtype=jnp.float64)
        )
        q_drift_blocks.append(np.asarray(q_drift))
        density_blocks.append(np.asarray(densities))
    q_drift = np.concatenate(q_drift_blocks)
    future_densities = np.concatenate(density_blocks)
    interior = np.maximum(future_densities[..., 1:-1], 0.0)
    density_scale = np.maximum(
        np.max(interior, axis=-1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    localization = interior / (interior + 1.0e-4 * density_scale)
    effective_loadings = log_dlvsq_loadings[None] * localization[..., None]
    coordinate_weights = interior / np.maximum(
        np.sum(interior, axis=-1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    target = empirical_log_drift - q_drift
    risk_prices = np.empty((len(states), factor_count), dtype=np.float64)
    risk_prices[:, 0] = spot_market_price_of_risk
    fitted = np.empty_like(target)
    for sample in range(len(states)):
        flat_loadings = effective_loadings[sample].reshape(-1, factor_count)
        weights = coordinate_weights[sample].reshape(-1)
        residual_target = target[sample].reshape(-1) - (
            flat_loadings[:, 0] * spot_market_price_of_risk
        )
        if factor_count > 1:
            remaining = flat_loadings[:, 1:]
            weighted = remaining * weights[:, None]
            gram = remaining.T @ weighted
            penalty = ridge * max(float(np.trace(gram)) / len(gram), 1.0)
            values = np.linalg.solve(
                gram + penalty * np.eye(factor_count - 1),
                remaining.T @ (weights * residual_target),
            )
            risk_prices[sample, 1:] = values
        fitted[sample] = np.einsum(
            "jnf,f->jn",
            effective_loadings[sample],
            risk_prices[sample],
        )
    target_energy = float(np.sum(coordinate_weights * target**2))
    residual_energy = float(np.sum(coordinate_weights * (target - fitted) ** 2))
    projection_fraction = (
        1.0 - residual_energy / target_energy
        if target_energy > np.finfo(np.float64).tiny
        else 1.0
    )
    return risk_prices, spot_market_price_of_risk, projection_fraction


def _ssm_apply_with_state(
    parameters: dict[str, jax.Array],
    inputs: jax.Array,
    reset: jax.Array,
    *,
    log_scale_bound: float,
) -> tuple[jax.Array, jax.Array]:
    hidden_size = parameters["candidate_bias"].shape[0]

    def step(hidden: jax.Array, values: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        current_input, current_reset = values
        hidden = jnp.where(current_reset, jnp.zeros_like(hidden), hidden)
        decay = jax.nn.sigmoid(
            current_input @ parameters["decay_weight"] + parameters["decay_bias"]
        )
        candidate = jnp.tanh(
            current_input @ parameters["candidate_weight"]
            + parameters["candidate_bias"]
        )
        hidden = decay * hidden + (1.0 - decay) * candidate
        log_scale = jnp.clip(
            hidden @ parameters["output_weight"] + parameters["output_bias"],
            -log_scale_bound,
            log_scale_bound,
        )
        return hidden, log_scale

    terminal_hidden, log_scales = jax.lax.scan(
        step,
        jnp.zeros((hidden_size,), dtype=inputs.dtype),
        (inputs, reset),
    )
    return log_scales, terminal_hidden


def _ssm_apply(
    parameters: dict[str, jax.Array],
    inputs: jax.Array,
    reset: jax.Array,
    *,
    log_scale_bound: float,
) -> jax.Array:
    """Return bounded conditional factor log-scales."""
    log_scales, _ = _ssm_apply_with_state(
        parameters,
        inputs,
        reset,
        log_scale_bound=log_scale_bound,
    )
    return log_scales


def fit_selective_state_space(
    constant: ConstantDiffusionEstimate,
    panel: dict[str, np.ndarray],
    *,
    hidden_size: int = 32,
    log_scale_bound: float = math.log(3.0),
    max_epochs: int = 1_000,
    patience: int = 100,
    learning_rate: float = 2.0e-3,
    seed: int = 91027,
) -> SelectiveStateSpaceEstimate:
    """Fit bounded Mamba-style selective factor amplitudes chronologically."""
    factor_scores = constant.factor_scores
    levels = np.asarray(panel["log_dlvsq_origins"], dtype=np.float64)
    flat_levels = levels.reshape(len(levels), -1)
    level_scores = (flat_levels - constant.level_mean.reshape(-1)) @ constant.level_basis
    level_scores /= constant.level_scale
    lagged_scores = np.vstack(
        (np.zeros((1, factor_scores.shape[1])), factor_scores[:-1])
    )
    contiguous = np.concatenate(
        ([False], np.diff(constant.pair_origin_indices) == 1)
    )
    lagged_scores[~contiguous] = 0.0
    inputs = np.concatenate((level_scores, lagged_scores), axis=1)
    split = max(int(0.8 * len(inputs)), 1)
    input_mean = np.mean(inputs[:split], axis=0)
    input_scale = np.std(inputs[:split], axis=0, ddof=1)
    input_scale = np.maximum(input_scale, 1.0e-6)
    inputs = (inputs - input_mean) / input_scale
    resets = ~contiguous

    generator = np.random.default_rng(seed)
    input_size = inputs.shape[1]
    factor_count = factor_scores.shape[1]
    parameters = {
        "decay_weight": generator.normal(
            scale=0.03, size=(input_size, hidden_size)
        ),
        "decay_bias": np.full((hidden_size,), 1.5),
        "candidate_weight": generator.normal(
            scale=0.05 / math.sqrt(input_size), size=(input_size, hidden_size)
        ),
        "candidate_bias": np.zeros((hidden_size,)),
        "output_weight": np.zeros((hidden_size, factor_count)),
        "output_bias": np.zeros((factor_count,)),
    }
    parameters_jax = jax.tree.map(lambda value: jnp.asarray(value), parameters)
    inputs_jax = jnp.asarray(inputs)
    resets_jax = jnp.asarray(resets)
    targets_jax = jnp.asarray(factor_scores)

    def losses(current: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        log_scales = _ssm_apply(
            current,
            inputs_jax,
            resets_jax,
            log_scale_bound=log_scale_bound,
        )
        point = 0.5 * jnp.mean(
            targets_jax**2 * jnp.exp(-2.0 * log_scales) + 2.0 * log_scales,
            axis=1,
        )
        return jnp.mean(point[:split]), jnp.mean(point[split:])

    value_and_gradient = jax.jit(jax.value_and_grad(lambda value: losses(value)[0]))
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(parameters_jax)
    best_parameters = parameters_jax
    best_validation = math.inf
    best_epoch = 0
    started = perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_loss, gradients = value_and_gradient(parameters_jax)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, parameters_jax
        )
        parameters_jax = optax.apply_updates(parameters_jax, updates)
        if epoch == 1 or epoch % 10 == 0:
            _, validation_loss = losses(parameters_jax)
            validation = float(validation_loss)
            if validation < best_validation - 1.0e-7:
                best_validation = validation
                best_parameters = jax.tree.map(lambda value: value.copy(), parameters_jax)
                best_epoch = epoch
            if epoch % 100 == 0:
                print(
                    f"selective-ssm epoch={epoch}/{max_epochs} "
                    f"train_nll={float(train_loss):.6f} val_nll={validation:.6f} "
                    f"elapsed={perf_counter() - started:.1f}s",
                    flush=True,
                )
            if epoch - best_epoch >= patience:
                break
    train_loss, validation_loss = losses(best_parameters)
    _, terminal_hidden = _ssm_apply_with_state(
        best_parameters,
        inputs_jax,
        resets_jax,
        log_scale_bound=log_scale_bound,
    )
    constant_validation = float(0.5 * jnp.mean(targets_jax[split:] ** 2))
    return SelectiveStateSpaceEstimate(
        parameters=jax.tree.map(lambda value: np.asarray(value), best_parameters),
        terminal_hidden=np.asarray(terminal_hidden),
        input_mean=input_mean,
        input_scale=input_scale,
        hidden_size=hidden_size,
        log_scale_bound=log_scale_bound,
        train_negative_log_likelihood=float(train_loss),
        validation_negative_log_likelihood=float(validation_loss),
        constant_validation_negative_log_likelihood=constant_validation,
        accepted=float(validation_loss) < constant_validation,
        epochs=best_epoch,
    )


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def estimate_models(
    *,
    history_path: Path,
    dataset: Path,
    output: Path,
    factor_count: int = 16,
) -> dict[str, Any]:
    """Estimate and serialize both requested diffusion models."""
    panel = load_estimation_panel(
        history_path=history_path,
        dataset=dataset,
        backbone_annual_volatility=0.05,
    )
    constant = fit_constant_diffusion(panel, factor_count=factor_count)
    selective = fit_selective_state_space(constant, panel)
    arrays: dict[str, np.ndarray] = {
        "log_dlvsq_loadings": constant.log_dlvsq_loadings.astype(np.float32),
        "factor_scores": constant.factor_scores.astype(np.float32),
        "physical_ar_intercept": constant.physical_ar_intercept.astype(np.float32),
        "physical_ar_slope": constant.physical_ar_slope.astype(np.float32),
        "level_mean": constant.level_mean.astype(np.float32),
        "level_basis": constant.level_basis.astype(np.float32),
        "level_scale": constant.level_scale.astype(np.float32),
        "physical_factor_coefficients": constant.physical_factor_coefficients.astype(
            np.float32
        ),
        "physical_market_prices_of_risk": (
            constant.physical_market_prices_of_risk.astype(np.float32)
        ),
        "spot_market_price_of_risk": np.asarray(
            constant.spot_market_price_of_risk,
            dtype=np.float32,
        ),
        "pair_origin_indices": constant.pair_origin_indices,
        "ssm_input_mean": selective.input_mean.astype(np.float32),
        "ssm_input_scale": selective.input_scale.astype(np.float32),
        "ssm_terminal_hidden": selective.terminal_hidden.astype(np.float32),
    }
    for name, value in selective.parameters.items():
        arrays[f"ssm_{name}"] = np.asarray(value, dtype=np.float32)
    _atomic_save_npz(output, arrays)
    report = {
        "schema": "gromm2-estimate-v1",
        "history_path": str(history_path.resolve()),
        "dataset": str(dataset.resolve()),
        "factor_count": factor_count,
        "pair_count": int(len(constant.factor_scores)),
        "explained_variance_ratio": constant.explained_variance_ratio,
        "physical_baseline": "coordinatewise AR(1) with Gaussian-smoothed innovations",
        "fixed_contract_interpolation_clip_fraction": float(
            panel["fixed_contract_interpolation_clip_fraction"]
        ),
        "spot_market_price_of_risk": constant.spot_market_price_of_risk,
        "physical_projection_fraction": constant.physical_projection_fraction,
        "selective_state_space": {
            "architecture": "bounded selective state-space factor amplitudes",
            "hidden_size": selective.hidden_size,
            "log_scale_bound": selective.log_scale_bound,
            "epochs": selective.epochs,
            "train_negative_log_likelihood": selective.train_negative_log_likelihood,
            "validation_negative_log_likelihood": selective.validation_negative_log_likelihood,
            "constant_validation_negative_log_likelihood": (
                selective.constant_validation_negative_log_likelihood
            ),
            "accepted": selective.accepted,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-count", type=int, default=16)
    arguments = parser.parse_args()
    report = estimate_models(
        history_path=arguments.history,
        dataset=arguments.dataset,
        output=arguments.output,
        factor_count=arguments.factor_count,
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
