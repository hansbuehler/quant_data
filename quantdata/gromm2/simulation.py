"""Risk-neutral six-month GROMM2 path simulation."""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

from quantdata.gromm2.bridge import advance_front_density, front_density_volatility
from quantdata.gromm2 import hat_front
from quantdata.gromm2.dlv import (
    SparseTransport,
    apply_sanos_matrix,
    apply_transport,
    decode_density_tower_from_front,
    risk_neutral_log_drift,
    risk_neutral_relative_drift,
    solve_sanos_step,
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
    dlv_variation: Array | None = None
    log_dlvsq: Array | None = None


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
    *,
    linear_solver: str = "thomas",
    active_maturity_count: Array | None = None,
) -> Array:
    return decode_density_tower_from_front(
        supports,
        transports,
        front_density,
        future_dlvsq,
        interval_years,
        linear_solver=linear_solver,
        active_maturity_count=active_maturity_count,
    )


def localize_loadings_by_density(
    log_dlvsq_loadings: Array,
    future_densities: Array,
    *,
    relative_floor: float = 1.0e-4,
) -> Array:
    """Apply explicit legacy attenuation, or preserve raw loadings at zero floor."""
    if relative_floor == 0.:
        return log_dlvsq_loadings
    interior = jnp.maximum(future_densities[..., 1:-1], 0.0)
    scale = jnp.max(interior, axis=-1, keepdims=True)
    floor = relative_floor * scale
    denominator = jnp.maximum(interior + floor, jnp.finfo(interior.dtype).tiny)
    weight = interior / denominator
    return log_dlvsq_loadings * weight[..., None]


def fused_risk_neutral_increments(
    supports: Array, transports: Array | SparseTransport, front_density: Array,
    future_dlvsq: Array, interval_years: Array, loading_field: Array,
    front_loading: Array, factor_normal: Array, factor_scales: Array, *,
    density_loading_floor: float, linear_solver: str,
    active_maturity_count: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Stream the unchanged Q recursion, retaining scalar increments per node.

    The loading field is unbatched. Path-dependent amplitudes and density
    localization are applied inside each expiry, rather than materializing
    path-by-expiry-by-strike-by-factor tensors for the whole tower.
    """
    factor_count = loading_field.shape[-1]

    def maturity(carry, inputs):
        previous_density, previous_loading = carry
        support, transport, local_variance, interval, base_loading = inputs
        density = solve_sanos_step(support, apply_transport(transport, previous_density),
            local_variance * interval, linear_solver=linear_solver)
        raw_loading = base_loading * factor_scales[..., None, :]
        loading = localize_loadings_by_density(raw_loading,
            density, relative_floor=density_loading_floor)
        zeros = jnp.zeros(loading.shape[:-2] + (1, factor_count), dtype=loading.dtype)
        loading_density = jnp.concatenate((zeros, loading, zeros), axis=-2) * density[..., :, None]
        density_loading = solve_sanos_step(support,
            apply_transport(transport, previous_loading, matrix=True) + loading_density
            - apply_sanos_matrix(support, loading_density, local_variance * interval),
            local_variance * interval, linear_solver=linear_solver)
        relative_drift = risk_neutral_relative_drift(raw_loading, density_loading, density,
            density_loading_floor=density_loading_floor)
        variance = jnp.sum(loading**2, axis=-1)
        diffusion = jnp.einsum("...nf,...f->...n", loading, factor_normal)
        return (density, density_loading), (relative_drift, variance, diffusion)

    if active_maturity_count is not None:
        empty = jnp.zeros_like(future_dlvsq)

        def active_step(index, carry):
            previous, previous_loading, drift, variance, diffusion = carry
            transport = jax.tree.map(lambda value: value[index], transports)
            (current, current_loading), values = maturity((previous, previous_loading),
                (supports[index + 1], transport, future_dlvsq[..., index, :],
                 interval_years[index], loading_field[index]))
            return (current, current_loading, drift.at[..., index, :].set(values[0]),
                    variance.at[..., index, :].set(values[1]), diffusion.at[..., index, :].set(values[2]))

        result = jax.lax.fori_loop(0, jnp.maximum(active_maturity_count - 1, 0), active_step,
                                  (front_density, front_loading, empty, empty, empty))
        return result[2:]
    _, outputs = jax.lax.scan(maturity, (front_density, front_loading),
        (supports[1:], transports, jnp.moveaxis(future_dlvsq, -2, 0),
         interval_years, loading_field))
    return tuple(jnp.moveaxis(value, 0, -2) for value in outputs)


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
    drift_taming: bool = True,
    linear_solver: str = "thomas",
    factor_scales: Array | None = None,
    active_maturity_count: Array | None = None,
    collect_dlv_variation: bool = False,
    log_dlvsq: Array | None = None,
    front: str | None = None,
) -> DailyStepResult:
    """Advance a day, with legacy drift taming only when explicitly enabled.

    Untampered paper log-Euler uses ``drift_taming=False``. A zero density
    loading floor preserves the estimated diffusion without attenuation.
    When supplied, log_dlvsq is the authoritative log-Euler state; its
    exponential is only a disposable view for the SANOS matrix solve.
    """
    if factor_normals.shape[0] != substeps:
        raise ValueError("factor_normals and substeps do not align")
    front = front or os.environ.get("GROMM2_FRONT", "bridge")       # default: bridge; "hat" = barycentric hat front
    if front not in ("bridge", "hat"):
        raise ValueError(f"unknown front {front!r}")
    use_hat = front == "hat"
    if active_maturity_count is not None and factor_scales is None:
        raise ValueError("a padded active tower requires the fused factor-scales path")
    future_dlvsq = state.dlvsq[..., 1:, :] if log_dlvsq is None else jnp.exp(log_dlvsq[..., 1:, :])
    front_density = state.front_density
    factor_count = factor_normals.shape[-1]
    substep_size = day_year_fraction / substeps
    day_start_remaining = remaining_front_days * day_year_fraction
    def substep(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], Array]:
        current_front, current_dlvsq = carry[:2]
        substep_index, factor_normal = inputs
        remaining_time = day_start_remaining - substep_index * substep_size
        if use_hat:
            current_front, hat_w = current_front
            _, hat_loading = hat_front.hat_density(hat_interval, hat_w, elapsed=substep_index * substep_size)
            front_loading = jnp.zeros(current_front.shape + (factor_count,), current_front.dtype).at[..., 0].set(hat_loading)
        else:
            front_loading = _front_loading(
                current_front,
                remaining_time=remaining_time,
                factor_count=factor_count,
            )
        if factor_scales is not None:
            relative_drift, factor_variance, diffusion = fused_risk_neutral_increments(
                supports, transports, current_front, current_dlvsq, interval_years,
                active_log_dlvsq_loadings, front_loading, factor_normal, factor_scales,
                density_loading_floor=density_loading_floor, linear_solver=linear_solver,
                active_maturity_count=active_maturity_count)
        else:
            current_tower = _decode_active_tower(
                supports, transports, current_front, current_dlvsq,
                interval_years, linear_solver=linear_solver)
            effective_loadings = localize_loadings_by_density(
                active_log_dlvsq_loadings, current_tower[..., 1:, :],
                relative_floor=density_loading_floor)
            factor_variance = jnp.sum(effective_loadings**2, axis=-1)
            _, _, log_drift = risk_neutral_log_drift(
                supports, transports, current_front, current_dlvsq,
                interval_years, effective_loadings, front_loading,
                linear_solver=linear_solver, raw_log_dlvsq_loadings=active_log_dlvsq_loadings,
                density_loading_floor=density_loading_floor)
            diffusion = jnp.einsum(
                "...jnf,...f->...jn", effective_loadings, factor_normal)
            relative_drift = log_drift + 0.5 * factor_variance
        if drift_taming:
            drift_increment, tamed = stabilized_relative_drift_increment(relative_drift, substep_size)
        else:
            drift_increment = relative_drift * substep_size
            tamed = jnp.zeros_like(relative_drift, dtype=jnp.bool_)
        log_increment = (
            drift_increment
            - 0.5 * factor_variance * substep_size
            + jnp.sqrt(substep_size) * diffusion
        )
        if log_dlvsq is None:
            next_dlvsq = current_dlvsq * jnp.exp(log_increment)
        else:
            next_log_dlvsq = carry[2] + log_increment
            next_dlvsq = jnp.exp(next_log_dlvsq)
        final_step = (remaining_front_days == 1) & (substep_index == substeps - 1)
        if use_hat:
            hat_w = hat_w + jnp.sqrt(substep_size) * factor_normal[..., 0]
            next_front = (jax.lax.cond(
                final_step,
                lambda: hat_front.settle_density(hat_interval, hat_w),
                lambda: hat_front.hat_density(hat_interval, hat_w, elapsed=(substep_index + 1) * substep_size)[0],
            ), hat_w)
        else:
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
        if active_maturity_count is not None:
            path_count = current_front.size // current_front.shape[-1]
            active_elements = jnp.maximum(active_maturity_count - 1, 1) * current_dlvsq.shape[-1] * path_count
            tamed_fraction = jnp.sum(tamed) / active_elements
        next_carry = (next_front, next_dlvsq)
        if log_dlvsq is not None:
            next_carry += (next_log_dlvsq,)
        if collect_dlv_variation:
            integrated_noise, integrated_variance, integrated_log_drift = carry[2 if log_dlvsq is None else 3:]
            next_carry += (integrated_noise + jnp.sqrt(substep_size) * diffusion,
                           integrated_variance + substep_size * factor_variance,
                           integrated_log_drift + drift_increment - .5 * factor_variance * substep_size)
        return next_carry, tamed_fraction

    if use_hat:
        # The hat front fixes G for one interval; carrying it across days is not implemented. Multi-day fronts
        # raise when remaining_front_days is static and turn the front into NaN when it is traced.
        try:
            static_days = int(remaining_front_days)
        except TypeError:                                                     # traced or per-path array
            static_days = None
        if static_days is not None and static_days != 1:
            raise NotImplementedError("hat front supports one-day fronts only (G is not carried across days)")
        front_density = front_density + jnp.where(remaining_front_days == 1, 0., jnp.nan)
        hat_interval = hat_front.start_interval(front_density, supports[..., 0, :], remaining_time=day_start_remaining)
        initial_carry = ((front_density, jnp.zeros(front_density.shape[:-1], front_density.dtype)), future_dlvsq)
    else:
        initial_carry = (front_density, future_dlvsq)
    if log_dlvsq is not None:
        initial_carry += (log_dlvsq[..., 1:, :],)
    if collect_dlv_variation:
        initial_carry += tuple(jnp.zeros_like(future_dlvsq) for _ in range(3))
    final_carry, taming_fractions = jax.lax.scan(
        substep,
        initial_carry,
        (jnp.arange(substeps), factor_normals),
    )
    updated_front, updated_future = final_carry[:2]
    if use_hat:
        updated_front = updated_front[0]
    variation = None
    if collect_dlv_variation:
        noise, expected_variance, log_drift = final_carry[2 if log_dlvsq is None else 3:]
        log_change = (jnp.log(updated_future) - jnp.log(future_dlvsq) if log_dlvsq is None
                      else final_carry[2] - log_dlvsq[..., 1:, :])
        variation = jnp.stack((noise**2, expected_variance, log_change, log_change**2,
                               jnp.abs(log_change - log_drift - noise)), axis=-3)
    tower = _decode_active_tower(
        supports,
        transports,
        updated_front,
        updated_future,
        interval_years,
        linear_solver=linear_solver,
        active_maturity_count=active_maturity_count,
    )
    if tower.shape[-2] > 1:
        has_next_expiry = tower.shape[-2] > 1 if active_maturity_count is None else active_maturity_count > 1
        next_front = jax.lax.cond(
            (remaining_front_days == 1) & has_next_expiry,
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
        dlv_variation=variation,
        log_dlvsq=None if log_dlvsq is None else log_dlvsq.at[..., 1:, :].set(final_carry[2]),
    )
