"""Risk-neutral six-month GROMM2 path simulation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from quantdata.gromm2.bridge import advance_front_density, front_density_volatility
from quantdata.gromm2.dlv import (
    decode_density_tower_from_front,
    risk_neutral_log_drift,
)

Array = jax.Array


class SimulationState(NamedTuple):
    """One batch of fixed-expiry risk-neutral states."""

    front_density: Array
    dlvsq: Array
    background_level: Array


class DailyStepResult(NamedTuple):
    """State and pricing densities after one simulated business day."""

    state: SimulationState
    density_tower: Array
    latent_spot: Array
    spot: Array
    daily_factor_score: Array
    drift_taming_fraction: Array


def stabilized_relative_drift_increment(
    relative_drift: Array,
    step_size: Array,
    *,
    explicit_limit: float = 0.5,
) -> tuple[Array, Array]:
    """Apply exact Euler drift ordinarily and logarithmic tail taming exceptionally."""
    raw = relative_drift * step_size
    absolute = jnp.abs(raw)
    excess = jnp.maximum(absolute - explicit_limit, 0.0)
    tamed = jnp.sign(raw) * (explicit_limit + jnp.log1p(excess))
    active = absolute > explicit_limit
    return jnp.where(active, tamed, raw), active


def _front_loading(
    front_density: Array,
    *,
    remaining_time: Array,
    factor_count: int,
) -> Array:
    loading = jnp.zeros(
        front_density.shape + (factor_count,), dtype=front_density.dtype
    )
    return loading.at[..., 0].set(
        front_density_volatility(
            front_density,
            remaining_time=remaining_time,
        )
    )


def _decode_active_tower(
    supports: Array,
    transports: Array,
    front_density: Array,
    future_dlvsq: Array,
    interval_years: Array,
) -> Array:
    return decode_density_tower_from_front(
        supports,
        transports,
        front_density,
        future_dlvsq,
        interval_years,
    )


def localize_loadings_by_density(
    log_dlvsq_loadings: Array,
    future_densities: Array,
    *,
    relative_floor: float = 1.0e-4,
) -> Array:
    """Make DLV diffusion vanish smoothly as the corresponding mass vanishes."""
    interior = jnp.maximum(future_densities[..., 1:-1], 0.0)
    scale = jnp.max(interior, axis=-1, keepdims=True)
    floor = relative_floor * scale
    weight = interior / (interior + floor)
    return log_dlvsq_loadings * weight[..., None]


def one_day_risk_neutral_step(
    supports: Array,
    transports: Array,
    interval_years: Array,
    state: SimulationState,
    active_log_dlvsq_loadings: Array,
    factor_normals: Array,
    background_normal: Array,
    *,
    remaining_front_days: Array,
    substeps: int,
    day_year_fraction: float = 1.0 / 252.0,
    background_annual_volatility: float = 0.05,
    density_loading_floor: float = 1.0e-4,
) -> DailyStepResult:
    """Advance one day using the exact front bridge and Q DLV drift."""
    if factor_normals.shape[0] != substeps:
        raise ValueError("factor_normals and substeps do not align")
    future_dlvsq = state.dlvsq[..., 1:, :]
    front_density = state.front_density
    factor_count = factor_normals.shape[-1]
    substep_size = day_year_fraction / substeps
    day_start_remaining = remaining_front_days * day_year_fraction
    def substep(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], Array]:
        current_front, current_dlvsq = carry
        substep_index, factor_normal = inputs
        remaining_time = day_start_remaining - substep_index * substep_size
        front_loading = _front_loading(
            current_front,
            remaining_time=remaining_time,
            factor_count=factor_count,
        )
        current_tower = _decode_active_tower(
            supports,
            transports,
            current_front,
            current_dlvsq,
            interval_years,
        )
        effective_loadings = localize_loadings_by_density(
            active_log_dlvsq_loadings,
            current_tower[..., 1:, :],
            relative_floor=density_loading_floor,
        )
        factor_variance = jnp.sum(effective_loadings**2, axis=-1)
        _, _, log_drift = risk_neutral_log_drift(
            supports,
            transports,
            current_front,
            current_dlvsq,
            interval_years,
            effective_loadings,
            front_loading,
        )
        diffusion = jnp.einsum(
            "...jnf,...f->...jn", effective_loadings, factor_normal
        )
        relative_drift = log_drift + 0.5 * factor_variance
        drift_increment, tamed = stabilized_relative_drift_increment(
            relative_drift, substep_size
        )
        next_dlvsq = current_dlvsq * jnp.exp(
            drift_increment
            - 0.5 * factor_variance * substep_size
            + jnp.sqrt(substep_size) * diffusion
        )
        final_step = (remaining_front_days == 1) & (substep_index == substeps - 1)
        next_front = advance_front_density(
            current_front,
            factor_normal[..., 0],
            remaining_time=remaining_time,
            step_size=substep_size,
            final_step=final_step,
        )
        tamed_fraction = (
            jnp.asarray(0.0, dtype=current_front.dtype)
            if tamed.size == 0
            else jnp.mean(tamed)
        )
        return (next_front, next_dlvsq), tamed_fraction

    (updated_front, updated_future), taming_fractions = jax.lax.scan(
        substep,
        (front_density, future_dlvsq),
        (jnp.arange(substeps), factor_normals),
    )
    tower = _decode_active_tower(
        supports,
        transports,
        updated_front,
        updated_future,
        interval_years,
    )
    if tower.shape[-2] > 1:
        next_front = jax.lax.cond(
            remaining_front_days == 1,
            lambda: tower[..., 1, :],
            lambda: tower[..., 0, :],
        )
    else:
        next_front = tower[..., 0, :]
    updated_all_dlvsq = state.dlvsq.at[..., 1:, :].set(updated_future)
    next_background = state.background_level * jnp.exp(
        -0.5 * background_annual_volatility**2 * day_year_fraction
        + background_annual_volatility
        * jnp.sqrt(day_year_fraction)
        * background_normal
    )
    latent_spot = jnp.einsum("...n,n->...", updated_front, supports[0])
    daily_factor_score = jnp.mean(factor_normals, axis=0) * jnp.sqrt(substeps)
    return DailyStepResult(
        state=SimulationState(
            front_density=next_front,
            dlvsq=updated_all_dlvsq,
            background_level=next_background,
        ),
        density_tower=tower,
        latent_spot=latent_spot,
        spot=latent_spot * next_background,
        daily_factor_score=daily_factor_score,
        drift_taming_fraction=jnp.mean(taming_fractions),
    )
