"""End-to-end historical calibration of the daily front-spot model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily import DATASET_PATH, load_raw_grids
from quantdata.sdlv.daily.front_spot.estimation import (
    AlignedDlvHistory,
    JointInnovationFit,
    PhysicalMeasureChangeFit,
    align_fixed_contract_history,
    decode_dlvsq,
    fit_achievable_physical_drift,
    fit_joint_innovations,
)
from quantdata.sdlv.daily.front_spot.jax_model import (
    density0_from_support,
    risk_neutral_dlv_drift,
    solve_sanos_step,
)


@dataclass(frozen=True)
class HistoricalRiskNeutralDrift:
    """Risk-neutral drift and probability relevance at historical states."""

    relative_dlvsq_drift: NDArray[np.float64]
    log_dlvsq_drift: NDArray[np.float64]
    future_densities: NDArray[np.float64]
    coordinate_weights: NDArray[np.float64]


@dataclass(frozen=True)
class FrontSpotCalibration:
    """All fitted P/Q components of the daily front-spot model."""

    history: AlignedDlvHistory
    innovations: JointInnovationFit
    risk_neutral: HistoricalRiskNeutralDrift
    physical: PhysicalMeasureChangeFit


def _one_state_risk_neutral_drift(
    support: jax.Array,
    dlvsq: jax.Array,
    annualized_loadings: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    front_density = solve_sanos_step(
        support,
        density0_from_support(support),
        dlvsq[0] * dt,
    )
    future_densities, _, relative_drift = risk_neutral_dlv_drift(
        support,
        front_density,
        dlvsq[1:],
        annualized_loadings,
        dt=dt,
    )
    return future_densities, relative_drift


def estimate_historical_risk_neutral_drift(
    history: AlignedDlvHistory,
    innovations: JointInnovationFit,
    *,
    fit_label: str = "full",
    batch_size: int = 64,
    dt: float = 1.0 / DAYS_PER_YEAR,
    dataset_path: Path = DATASET_PATH,
    raw_grids: NDArray[np.float32] | None = None,
) -> HistoricalRiskNeutralDrift:
    """Evaluate the Q drift at every aligned historical origin state."""
    if not jax.config.x64_enabled:
        raise RuntimeError("Historical Q-drift estimation requires JAX 64-bit mode")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, found {batch_size}")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be finite and positive, found {dt}")

    with np.load(dataset_path, allow_pickle=False) as dataset:
        supports = np.asarray(dataset["supports"], dtype=np.float64)
    dlvsq = decode_dlvsq(load_raw_grids(fit_label) if raw_grids is None else raw_grids)
    origins = history.origin_indices
    origin_supports = supports[origins]
    origin_dlvsq = dlvsq[origins]
    expected_logs = np.log(origin_dlvsq[:, 1:])
    if not np.allclose(expected_logs, history.origin_log_dlvsq, atol=2.0e-7):
        raise ValueError("History and fitted DLV representation do not match")

    loadings = jnp.asarray(innovations.annualized_loadings, dtype=jnp.float64)

    @jax.jit
    def evaluate_batch(
        batch_supports: jax.Array,
        batch_dlvsq: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return jax.vmap(
            lambda support, state: _one_state_risk_neutral_drift(
                support,
                state,
                loadings,
                dt,
            )
        )(batch_supports, batch_dlvsq)

    density_blocks: list[NDArray[np.float64]] = []
    drift_blocks: list[NDArray[np.float64]] = []
    sample_count = len(origins)
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        valid_count = stop - start
        support_block = origin_supports[start:stop]
        dlvsq_block = origin_dlvsq[start:stop]
        if valid_count < batch_size:
            pad_count = batch_size - valid_count
            support_block = np.concatenate(
                (support_block, np.repeat(support_block[-1:], pad_count, axis=0))
            )
            dlvsq_block = np.concatenate(
                (dlvsq_block, np.repeat(dlvsq_block[-1:], pad_count, axis=0))
            )
        batch_densities, batch_drift = evaluate_batch(
            jnp.asarray(support_block, dtype=jnp.float64),
            jnp.asarray(dlvsq_block, dtype=jnp.float64),
        )
        density_blocks.append(np.asarray(batch_densities[:valid_count]))
        drift_blocks.append(np.asarray(batch_drift[:valid_count]))

    densities = np.concatenate(density_blocks, axis=0)
    relative_drift = np.concatenate(drift_blocks, axis=0)
    factor_variance = np.sum(innovations.annualized_loadings**2, axis=-1)
    log_drift = relative_drift - 0.5 * factor_variance
    if np.any(~np.isfinite(densities)) or np.any(densities < -1.0e-12):
        raise FloatingPointError("Historical Q densities are invalid")
    if np.any(~np.isfinite(log_drift)):
        raise FloatingPointError("Historical Q log-DLV drift is non-finite")

    interior_density = np.maximum(densities[..., 1:-1], 0.0)
    expiry_weight = np.sum(interior_density, axis=-1, keepdims=True)
    if np.any(expiry_weight <= np.finfo(np.float64).tiny):
        raise FloatingPointError("A historical expiry has no interior probability mass")
    coordinate_weights = interior_density / expiry_weight
    return HistoricalRiskNeutralDrift(
        relative_dlvsq_drift=relative_drift,
        log_dlvsq_drift=log_drift,
        future_densities=densities,
        coordinate_weights=coordinate_weights,
    )


def calibrate_front_spot_model(
    *,
    fit_label: str = "full",
    factor_rank: int = 8,
    batch_size: int = 64,
    correct_lag1_fit_noise: bool = False,
    smoothing_sigma: tuple[float, float] | None = (1.0, 2.0),
    dataset_path: Path = DATASET_PATH,
    raw_grids: NDArray[np.float32] | None = None,
) -> FrontSpotCalibration:
    """Estimate joint innovations, Q drift, and achievable P drift."""
    history = align_fixed_contract_history(
        fit_label=fit_label, dataset_path=dataset_path, raw_grids=raw_grids,
    )
    innovations = fit_joint_innovations(
        history,
        factor_rank=factor_rank,
        correct_lag1_fit_noise=correct_lag1_fit_noise,
        smoothing_sigma=smoothing_sigma,
    )
    risk_neutral = estimate_historical_risk_neutral_drift(
        history,
        innovations,
        fit_label=fit_label,
        batch_size=batch_size,
        dataset_path=dataset_path,
        raw_grids=raw_grids,
    )
    physical = fit_achievable_physical_drift(
        history,
        innovations,
        risk_neutral.log_dlvsq_drift,
        risk_neutral.coordinate_weights,
    )
    return FrontSpotCalibration(
        history=history,
        innovations=innovations,
        risk_neutral=risk_neutral,
        physical=physical,
    )
