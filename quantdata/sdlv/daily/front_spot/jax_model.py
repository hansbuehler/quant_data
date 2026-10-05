"""JAX implementation of the daily risk-neutral front-spot DLV step."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.estimation import BACKGROUND_ANNUAL_VOL

Array = jax.Array


class RiskNeutralState(NamedTuple):
    """One shrinking fixed-expiry DLV tower."""

    front_density: Array
    future_dlvsq: Array
    background_level: Array


class DailyStepResult(NamedTuple):
    """State and diagnostics returned by one daily risk-neutral step."""

    state: RiskNeutralState
    density_tower: Array
    relative_dlv_drift: Array
    density_volatility: Array
    sampled_front_index: Array
    latent_spot: Array
    spot: Array


class PhysicalDailyStepResult(NamedTuple):
    """State and measure-change diagnostics for one physical daily step."""

    state: RiskNeutralState
    density_tower: Array
    risk_neutral_log_drift: Array
    empirical_log_drift: Array
    achievable_log_drift: Array
    market_price_of_risk: Array
    sampled_front_index: Array
    latent_spot: Array
    spot: Array


class TerminalFrontResult(NamedTuple):
    """Terminal front sample after the DLV tower has no later expiry."""

    background_level: Array
    sampled_front_index: Array
    latent_spot: Array
    spot: Array


class SubsteppedDailyStepResult(NamedTuple):
    """State returned by an intraday-substepped risk-neutral update."""

    state: RiskNeutralState
    density_tower: Array
    sampled_front_index: Array
    latent_spot: Array
    spot: Array


def substep_brownian_weights(
    substep_count: int,
    *,
    time_power: float = 1.0,
    dtype: jax.typing.DTypeLike = jnp.float64,
) -> Array:
    """Return Brownian standard-deviation weights for an expiry-refined grid."""
    if substep_count < 1:
        raise ValueError("At least one DLV substep is required")
    if not time_power >= 1.0:
        raise ValueError("time_power must be at least one")
    index = jnp.arange(substep_count, dtype=dtype)
    remaining_fraction = (substep_count - index) / substep_count
    next_fraction = (substep_count - index - 1.0) / substep_count
    step_fraction = remaining_fraction**time_power - next_fraction**time_power
    return jnp.sqrt(step_fraction)


def stabilized_relative_drift_increment(
    relative_drift: Array,
    dt: float | Array,
    *,
    explicit_limit: float = 0.5,
) -> Array:
    """Keep ordinary log-Euler drift exact and tame exceptional tail freezes."""
    raw_increment = relative_drift * dt
    absolute_increment = jnp.abs(raw_increment)
    excess = jnp.maximum(absolute_increment - explicit_limit, 0.0)
    tamed_increment = jnp.sign(raw_increment) * (
        explicit_limit + jnp.log1p(excess)
    )
    return jnp.where(
        absolute_increment <= explicit_limit,
        raw_increment,
        tamed_increment,
    )


def density0_from_support(support: Array) -> Array:
    """Represent unit spot exactly on an ordered finite strike support."""
    calls = jnp.maximum(1.0 - support, 0.0)
    slopes = jnp.diff(calls, prepend=jnp.ones_like(calls[..., :1])) / jnp.diff(
        support,
        prepend=jnp.zeros_like(support[..., :1]),
    )
    return jnp.diff(slopes, append=jnp.zeros_like(slopes[..., :1]))


def _sanos_diagonals(support: Array, dlvsq_dt: Array) -> tuple[Array, Array, Array]:
    box = support[..., 1:-1] ** 2 / (support[..., 2:] - support[..., :-2])
    w_plus = dlvsq_dt * box / (support[..., 2:] - support[..., 1:-1])
    w_minus = dlvsq_dt * box / (support[..., 1:-1] - support[..., :-2])
    zeros = jnp.zeros_like(w_plus[..., :2])
    lower = jnp.concatenate((zeros, -w_plus), axis=-1)
    diagonal = jnp.concatenate(
        (
            jnp.ones_like(dlvsq_dt[..., :1]),
            1.0 + w_minus + w_plus,
            jnp.ones_like(dlvsq_dt[..., :1]),
        ),
        axis=-1,
    )
    upper = jnp.concatenate((-w_minus, zeros), axis=-1)
    return lower, diagonal, upper


def solve_sanos_step(support: Array, previous: Array, dlvsq_dt: Array) -> Array:
    """Solve one SANOS tridiagonal step for one or more right-hand sides."""
    lower, diagonal, upper = _sanos_diagonals(support, dlvsq_dt)
    vector_rhs = previous.ndim == dlvsq_dt.ndim
    rhs = previous[..., :, None] if vector_rhs else previous
    result = jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs)
    return result[..., 0] if vector_rhs else result


def apply_sanos_matrix(support: Array, value: Array, dlvsq_dt: Array) -> Array:
    """Apply the SANOS tridiagonal matrix to vector or matrix values."""
    lower, diagonal, upper = _sanos_diagonals(support, dlvsq_dt)
    vector_value = value.ndim == dlvsq_dt.ndim
    values = value[..., :, None] if vector_value else value
    result = diagonal[..., :, None] * values
    result = result.at[..., 1:, :].add(lower[..., 1:, None] * values[..., :-1, :])
    result = result.at[..., :-1, :].add(upper[..., :-1, None] * values[..., 1:, :])
    return result[..., 0] if vector_value else result


def decode_density_tower(support: Array, dlvsq: Array, *, dt: float) -> Array:
    """Decode an expiry-major DLV-squared tower from unit spot."""
    initial = density0_from_support(support)
    batch_shape = dlvsq.shape[:-2]
    initial = jnp.broadcast_to(initial, batch_shape + initial.shape)

    return decode_density_tower_from_front(
        support,
        initial,
        dlvsq,
        dt=dt,
    )


def decode_density_tower_from_front(
    support: Array,
    front_density: Array,
    dlvsq: Array,
    *,
    dt: float,
) -> Array:
    """Decode future fixed-expiry densities from an arbitrary front density."""

    def step(previous: Array, current_dlvsq: Array) -> tuple[Array, Array]:
        current = solve_sanos_step(support, previous, current_dlvsq * dt)
        return current, current

    _, time_major = jax.lax.scan(
        step,
        front_density,
        jnp.moveaxis(dlvsq, -2, 0),
    )
    return jnp.moveaxis(time_major, 0, -2)


def front_density_volatility(front_density: Array, *, dt: float) -> Array:
    """Return the exact initial volatility of the Gaussian front bridge."""
    cumulative = jnp.cumsum(front_density, axis=-1)
    interior = cumulative[..., :-1]
    probability_floor = jnp.finfo(front_density.dtype).eps
    safe_probability = jnp.clip(
        interior,
        probability_floor,
        1.0 - probability_floor,
    )
    thresholds = jsp.special.ndtri(safe_probability)
    threshold_density = jnp.exp(-0.5 * thresholds**2) / jnp.sqrt(2.0 * jnp.pi)
    threshold_density = jnp.where(
        (interior <= 0.0) | (interior >= 1.0),
        0.0,
        threshold_density,
    )
    boundary_zero = jnp.zeros_like(threshold_density[..., :1])
    phi = jnp.concatenate((boundary_zero, threshold_density, boundary_zero), axis=-1)
    return (phi[..., :-1] - phi[..., 1:]) / jnp.sqrt(dt)


def normal_interval_probabilities(boundaries: Array) -> Array:
    """Return stable Gaussian masses between increasing interval boundaries."""
    direct = jnp.diff(jsp.special.ndtr(boundaries), axis=-1)
    survival = jsp.special.ndtr(-boundaries)
    upper_tail = survival[..., :-1] - survival[..., 1:]
    probabilities = jnp.where(
        boundaries[..., :-1] >= 0.0,
        upper_tail,
        direct,
    )
    return jnp.maximum(probabilities, 0.0)


def advance_front_density(
    front_density: Array,
    spot_normal: Array,
    *,
    remaining_time: float | Array,
    step_size: float,
    final_step: bool | Array,
) -> Array:
    """Apply one exact Gaussian-information-bridge transition."""

    def sample_terminal(_: None) -> Array:
        sampled_index = sample_front_index(front_density, spot_normal)
        return jax.nn.one_hot(
            sampled_index,
            front_density.shape[-1],
            dtype=front_density.dtype,
        )

    def reveal_increment(_: None) -> Array:
        cumulative = jnp.cumsum(front_density, axis=-1)[..., :-1]
        probability_floor = jnp.finfo(front_density.dtype).eps
        safe_probability = jnp.clip(
            cumulative,
            probability_floor,
            1.0 - probability_floor,
        )
        thresholds = jsp.special.ndtri(safe_probability)
        thresholds = jnp.where(cumulative <= 0.0, -jnp.inf, thresholds)
        thresholds = jnp.where(cumulative >= 1.0, jnp.inf, thresholds)
        shifted = (
            jnp.sqrt(remaining_time) * thresholds
            - jnp.sqrt(step_size) * spot_normal[..., None]
        ) / jnp.sqrt(remaining_time - step_size)
        boundaries = jnp.concatenate(
            (
                jnp.full_like(shifted[..., :1], -jnp.inf),
                shifted,
                jnp.full_like(shifted[..., :1], jnp.inf),
            ),
            axis=-1,
        )
        next_density = normal_interval_probabilities(boundaries)
        return next_density / jnp.sum(next_density, axis=-1, keepdims=True)

    return jax.lax.cond(final_step, sample_terminal, reveal_increment, None)


def risk_neutral_dlv_drift(
    support: Array,
    front_density: Array,
    future_dlvsq: Array,
    annualized_loadings: Array,
    *,
    dt: float,
    front_horizon: float | Array | None = None,
) -> tuple[Array, Array, Array]:
    """Compute future densities, density volatilities and relative DLV drift."""
    factor_count = annualized_loadings.shape[-1]
    front_e = jnp.zeros(front_density.shape + (factor_count,), dtype=front_density.dtype)
    bridge_horizon = dt if front_horizon is None else front_horizon
    front_e = front_e.at[..., 0].set(
        front_density_volatility(front_density, dt=bridge_horizon)
    )

    def step(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        previous_density, previous_e = carry
        current_dlvsq, current_xi = inputs
        current_density = solve_sanos_step(
            support,
            previous_density,
            current_dlvsq * dt,
        )
        boundary_zeros = jnp.zeros(current_xi.shape[:-2] + (1, factor_count))
        extended_xi = jnp.concatenate(
            (boundary_zeros, current_xi, boundary_zeros), axis=-2
        )
        xi_density = extended_xi * current_density[..., :, None]
        matrix_value = apply_sanos_matrix(
            support,
            xi_density,
            current_dlvsq * dt,
        )
        current_e = solve_sanos_step(
            support,
            previous_e + xi_density - matrix_value,
            current_dlvsq * dt,
        )
        safe_density = jnp.maximum(
            current_density[..., 1:-1, None],
            jnp.finfo(current_density.dtype).tiny,
        )
        relative_density_vol = current_e[..., 1:-1, :] / safe_density
        relative_drift = -jnp.sum(current_xi * relative_density_vol, axis=-1)
        return (current_density, current_e), (
            current_density,
            current_e,
            relative_drift,
        )

    _, outputs = jax.lax.scan(
        step,
        (front_density, front_e),
        (
            jnp.moveaxis(future_dlvsq, -2, 0),
            jnp.moveaxis(annualized_loadings, -3, 0),
        ),
    )
    densities, density_volatility, relative_drift = outputs
    return (
        jnp.moveaxis(densities, 0, -2),
        jnp.moveaxis(density_volatility, 0, -3),
        jnp.moveaxis(relative_drift, 0, -2),
    )


def sample_front_index(front_density: Array, spot_normal: Array) -> Array:
    """Sample the exact discrete front law with a Gaussian copula coordinate."""
    probability = jsp.special.ndtr(spot_normal)
    cumulative = jnp.cumsum(front_density, axis=-1)
    index = jnp.sum(probability[..., None] > cumulative, axis=-1)
    return jnp.minimum(index, front_density.shape[-1] - 1).astype(jnp.int32)


def terminal_front_step(
    support: Array,
    front_density: Array,
    background_level: Array,
    spot_normal: Array,
    background_normal: Array,
    *,
    dt: float = 1.0 / DAYS_PER_YEAR,
) -> TerminalFrontResult:
    """Sample the last discrete front and advance the independent background."""
    sampled_index = sample_front_index(front_density, spot_normal)
    latent_spot = support[sampled_index]
    next_background = background_level * jnp.exp(
        -0.5 * BACKGROUND_ANNUAL_VOL**2 * dt
        + BACKGROUND_ANNUAL_VOL * jnp.sqrt(dt) * background_normal
    )
    return TerminalFrontResult(
        background_level=next_background,
        sampled_front_index=sampled_index,
        latent_spot=latent_spot,
        spot=latent_spot * next_background,
    )


def one_day_risk_neutral_substepped(
    support: Array,
    state: RiskNeutralState,
    annualized_loadings: Array,
    substep_factor_normals: Array,
    background_normal: Array,
    *,
    expiry_dt: float = 1.0 / DAYS_PER_YEAR,
    time_power: float = 1.0,
) -> SubsteppedDailyStepResult:
    """Advance one day while recomputing Q drift along the exact front bridge."""
    substep_count = substep_factor_normals.shape[0]
    if substep_count < 1:
        raise ValueError("At least one DLV substep is required")
    brownian_weights = substep_brownian_weights(
        substep_count,
        time_power=time_power,
        dtype=state.front_density.dtype,
    )
    substep_sizes = expiry_dt * brownian_weights**2
    remaining_times = jnp.cumsum(substep_sizes[::-1])[::-1]
    batch_shape = substep_factor_normals.shape[1:-1]
    initial_front = jnp.broadcast_to(
        state.front_density,
        batch_shape + state.front_density.shape[-1:],
    )
    initial_dlvsq = jnp.broadcast_to(
        state.future_dlvsq,
        batch_shape + state.future_dlvsq.shape[-2:],
    )

    def substep(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        front_density, future_dlvsq = carry
        remaining_time, substep_dt, final_step, factor_normal = inputs
        _, _, relative_drift = risk_neutral_dlv_drift(
            support,
            front_density,
            future_dlvsq,
            annualized_loadings,
            dt=expiry_dt,
            front_horizon=remaining_time,
        )
        factor_variance = jnp.sum(annualized_loadings**2, axis=-1)
        diffusion = jnp.einsum(
            "jnf,...f->...jn",
            annualized_loadings,
            factor_normal,
        )
        next_dlvsq = future_dlvsq * jnp.exp(
            stabilized_relative_drift_increment(relative_drift, substep_dt)
            - 0.5 * factor_variance * substep_dt
            + jnp.sqrt(substep_dt) * diffusion
        )
        next_front = advance_front_density(
            front_density,
            factor_normal[..., 0],
            remaining_time=remaining_time,
            step_size=substep_dt,
            final_step=final_step,
        )
        return (next_front, next_dlvsq), None

    (_, updated_dlvsq), _ = jax.lax.scan(
        substep,
        (initial_front, initial_dlvsq),
        (
            remaining_times,
            substep_sizes,
            jnp.arange(substep_count) == substep_count - 1,
            substep_factor_normals,
        ),
    )
    daily_spot_normal = jnp.einsum(
        "s,s...->...",
        brownian_weights,
        substep_factor_normals[..., 0],
    )
    sampled_index = sample_front_index(initial_front, daily_spot_normal)
    sampled_front = jax.nn.one_hot(
        sampled_index,
        support.shape[-1],
        dtype=state.front_density.dtype,
    )

    next_tower = decode_density_tower_from_front(
        support,
        sampled_front,
        updated_dlvsq,
        dt=expiry_dt,
    )
    next_background = state.background_level * jnp.exp(
        -0.5 * BACKGROUND_ANNUAL_VOL**2 * expiry_dt
        + BACKGROUND_ANNUAL_VOL * jnp.sqrt(expiry_dt) * background_normal
    )
    latent_spot = support[sampled_index]
    return SubsteppedDailyStepResult(
        state=RiskNeutralState(
            next_tower[..., 0, :],
            updated_dlvsq[..., 1:, :],
            next_background,
        ),
        density_tower=next_tower,
        sampled_front_index=sampled_index,
        latent_spot=latent_spot,
        spot=latent_spot * next_background,
    )


def one_day_physical_substepped(
    support: Array,
    state: RiskNeutralState,
    annualized_loadings: Array,
    market_price_of_risk: Array,
    substep_factor_normals: Array,
    background_normal: Array,
    *,
    expiry_dt: float = 1.0 / DAYS_PER_YEAR,
    time_power: float = 1.0,
) -> SubsteppedDailyStepResult:
    """Advance one day under P with a fixed fitted Brownian risk price."""
    substep_count = substep_factor_normals.shape[0]
    if substep_count < 1:
        raise ValueError("At least one DLV substep is required")
    brownian_weights = substep_brownian_weights(
        substep_count,
        time_power=time_power,
        dtype=state.front_density.dtype,
    )
    substep_sizes = expiry_dt * brownian_weights**2
    remaining_times = jnp.cumsum(substep_sizes[::-1])[::-1]
    batch_shape = substep_factor_normals.shape[1:-1]
    initial_front = jnp.broadcast_to(
        state.front_density,
        batch_shape + state.front_density.shape[-1:],
    )
    initial_dlvsq = jnp.broadcast_to(
        state.future_dlvsq,
        batch_shape + state.future_dlvsq.shape[-2:],
    )
    factor_variance = jnp.sum(annualized_loadings**2, axis=-1)
    risk_premium = jnp.einsum(
        "jnf,...f->...jn",
        annualized_loadings,
        market_price_of_risk,
    )

    def substep(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        front_density, future_dlvsq = carry
        remaining_time, substep_dt, final_step, factor_normal = inputs
        _, _, relative_drift = risk_neutral_dlv_drift(
            support,
            front_density,
            future_dlvsq,
            annualized_loadings,
            dt=expiry_dt,
            front_horizon=remaining_time,
        )
        diffusion = jnp.einsum(
            "jnf,...f->...jn",
            annualized_loadings,
            factor_normal,
        )
        next_dlvsq = future_dlvsq * jnp.exp(
            stabilized_relative_drift_increment(
                relative_drift + risk_premium,
                substep_dt,
            )
            - 0.5 * factor_variance * substep_dt
            + jnp.sqrt(substep_dt) * diffusion
        )
        q_spot_normal = (
            factor_normal[..., 0]
            + market_price_of_risk[..., 0] * jnp.sqrt(substep_dt)
        )
        next_front = advance_front_density(
            front_density,
            q_spot_normal,
            remaining_time=remaining_time,
            step_size=substep_dt,
            final_step=final_step,
        )
        return (next_front, next_dlvsq), None

    (_, updated_dlvsq), _ = jax.lax.scan(
        substep,
        (initial_front, initial_dlvsq),
        (
            remaining_times,
            substep_sizes,
            jnp.arange(substep_count) == substep_count - 1,
            substep_factor_normals,
        ),
    )
    daily_q_spot_normal = (
        jnp.einsum(
            "s,s...->...",
            brownian_weights,
            substep_factor_normals[..., 0],
        )
        + market_price_of_risk[..., 0] * jnp.sqrt(expiry_dt)
    )
    sampled_index = sample_front_index(initial_front, daily_q_spot_normal)
    sampled_front = jax.nn.one_hot(
        sampled_index,
        support.shape[-1],
        dtype=state.front_density.dtype,
    )

    next_tower = decode_density_tower_from_front(
        support,
        sampled_front,
        updated_dlvsq,
        dt=expiry_dt,
    )
    next_background = state.background_level * jnp.exp(
        -0.5 * BACKGROUND_ANNUAL_VOL**2 * expiry_dt
        + BACKGROUND_ANNUAL_VOL * jnp.sqrt(expiry_dt) * background_normal
    )
    latent_spot = support[sampled_index]
    return SubsteppedDailyStepResult(
        state=RiskNeutralState(
            next_tower[..., 0, :],
            updated_dlvsq[..., 1:, :],
            next_background,
        ),
        density_tower=next_tower,
        sampled_front_index=sampled_index,
        latent_spot=latent_spot,
        spot=latent_spot * next_background,
    )


def project_physical_log_drift(
    log_future_dlvsq: Array,
    risk_neutral_log_drift: Array,
    future_densities: Array,
    annualized_loadings: Array,
    conditional_intercept: Array,
    conditional_slope: Array,
    *,
    dt: float,
    ridge: float = 1.0e-10,
) -> tuple[Array, Array, Array]:
    """Project the empirical physical drift onto ``m_Q + xi lambda``."""
    empirical_log_drift = (
        conditional_intercept
        + conditional_slope * log_future_dlvsq
        - log_future_dlvsq
    ) / dt
    target = empirical_log_drift - risk_neutral_log_drift
    interior_density = jnp.maximum(future_densities[..., 1:-1], 0.0)
    expiry_mass = jnp.sum(interior_density, axis=-1, keepdims=True)
    weights = interior_density / jnp.maximum(
        expiry_mass,
        jnp.finfo(interior_density.dtype).tiny,
    )
    flat_target = target.reshape(target.shape[:-2] + (-1,))
    flat_weights = weights.reshape(weights.shape[:-2] + (-1,))
    flat_weights *= flat_weights.shape[-1] / jnp.sum(
        flat_weights, axis=-1, keepdims=True
    )
    loading_matrix = annualized_loadings.reshape(-1, annualized_loadings.shape[-1])
    weighted_loadings = flat_weights[..., :, None] * loading_matrix
    gram = jnp.einsum("pf,...pg->...fg", loading_matrix, weighted_loadings)
    right_hand_side = jnp.einsum(
        "pf,...p->...f",
        loading_matrix,
        flat_weights * flat_target,
    )
    factor_count = annualized_loadings.shape[-1]
    ridge_scale = ridge * jnp.maximum(
        jnp.trace(gram, axis1=-2, axis2=-1) / factor_count,
        1.0,
    )
    regularized_gram = gram + ridge_scale[..., None, None] * jnp.eye(
        factor_count,
        dtype=gram.dtype,
    )
    market_price_of_risk = jnp.linalg.solve(
        regularized_gram,
        right_hand_side[..., None],
    )[..., 0]
    projected = jnp.einsum(
        "jnf,...f->...jn",
        annualized_loadings,
        market_price_of_risk,
    )
    achievable_log_drift = risk_neutral_log_drift + projected
    return empirical_log_drift, achievable_log_drift, market_price_of_risk


def one_day_risk_neutral_step(
    support: Array,
    state: RiskNeutralState,
    annualized_loadings: Array,
    factor_normals: Array,
    background_normal: Array,
    *,
    dt: float = 1.0 / DAYS_PER_YEAR,
) -> DailyStepResult:
    """Advance the shrinking DLV tower by one front-expiry day."""
    future_densities, density_volatility, relative_drift = risk_neutral_dlv_drift(
        support,
        state.front_density,
        state.future_dlvsq,
        annualized_loadings,
        dt=dt,
    )
    del future_densities
    diffusion_variance = jnp.sum(annualized_loadings**2, axis=-1)
    diffusion = jnp.einsum("jnf,...f->...jn", annualized_loadings, factor_normals)
    updated_dlvsq = state.future_dlvsq * jnp.exp(
        stabilized_relative_drift_increment(relative_drift, dt)
        - 0.5 * diffusion_variance * dt
        + jnp.sqrt(dt) * diffusion
    )
    sampled_index = sample_front_index(state.front_density, factor_normals[..., 0])
    sampled_front = jax.nn.one_hot(
        sampled_index,
        support.shape[-1],
        dtype=state.front_density.dtype,
    )

    def reconstruct(previous: Array, current_dlvsq: Array) -> tuple[Array, Array]:
        current = solve_sanos_step(support, previous, current_dlvsq * dt)
        return current, current

    _, next_time_major = jax.lax.scan(
        reconstruct,
        sampled_front,
        jnp.moveaxis(updated_dlvsq, -2, 0),
    )
    next_tower = jnp.moveaxis(next_time_major, 0, -2)
    next_front = next_tower[..., 0, :]
    next_future_dlvsq = updated_dlvsq[..., 1:, :]
    next_background = state.background_level * jnp.exp(
        -0.5 * BACKGROUND_ANNUAL_VOL**2 * dt
        + BACKGROUND_ANNUAL_VOL * jnp.sqrt(dt) * background_normal
    )
    latent_spot = jnp.sum(sampled_front * support, axis=-1)
    spot = latent_spot * next_background
    return DailyStepResult(
        state=RiskNeutralState(next_front, next_future_dlvsq, next_background),
        density_tower=next_tower,
        relative_dlv_drift=relative_drift,
        density_volatility=density_volatility,
        sampled_front_index=sampled_index,
        latent_spot=latent_spot,
        spot=spot,
    )


def one_day_physical_step(
    support: Array,
    state: RiskNeutralState,
    annualized_loadings: Array,
    conditional_intercept: Array,
    conditional_slope: Array,
    factor_normals: Array,
    background_normal: Array,
    *,
    dt: float = 1.0 / DAYS_PER_YEAR,
    ridge: float = 1.0e-10,
    fixed_market_price_of_risk: Array | None = None,
) -> PhysicalDailyStepResult:
    """Advance one day under P with the closest equivalent-measure drift."""
    future_densities, _, relative_drift = risk_neutral_dlv_drift(
        support,
        state.front_density,
        state.future_dlvsq,
        annualized_loadings,
        dt=dt,
    )
    factor_variance = jnp.sum(annualized_loadings**2, axis=-1)
    risk_neutral_log_drift = relative_drift - 0.5 * factor_variance
    if fixed_market_price_of_risk is None:
        empirical_log_drift, achievable_log_drift, risk_price = (
            project_physical_log_drift(
                jnp.log(state.future_dlvsq),
                risk_neutral_log_drift,
                future_densities,
                annualized_loadings,
                conditional_intercept,
                conditional_slope,
                dt=dt,
                ridge=ridge,
            )
        )
    else:
        empirical_log_drift = (
            conditional_intercept
            + conditional_slope * jnp.log(state.future_dlvsq)
            - jnp.log(state.future_dlvsq)
        ) / dt
        risk_price = fixed_market_price_of_risk
        achievable_log_drift = risk_neutral_log_drift + jnp.einsum(
            "jnf,...f->...jn",
            annualized_loadings,
            risk_price,
        )
    diffusion = jnp.einsum("jnf,...f->...jn", annualized_loadings, factor_normals)
    achievable_relative_drift = achievable_log_drift + 0.5 * factor_variance
    updated_dlvsq = state.future_dlvsq * jnp.exp(
        stabilized_relative_drift_increment(achievable_relative_drift, dt)
        - 0.5 * factor_variance * dt
        + jnp.sqrt(dt) * diffusion
    )

    q_spot_normal = factor_normals[..., 0] + risk_price[..., 0] * jnp.sqrt(dt)
    sampled_index = sample_front_index(state.front_density, q_spot_normal)
    sampled_front = jax.nn.one_hot(
        sampled_index,
        support.shape[-1],
        dtype=state.front_density.dtype,
    )

    def reconstruct(previous: Array, current_dlvsq: Array) -> tuple[Array, Array]:
        current = solve_sanos_step(support, previous, current_dlvsq * dt)
        return current, current

    _, next_time_major = jax.lax.scan(
        reconstruct,
        sampled_front,
        jnp.moveaxis(updated_dlvsq, -2, 0),
    )
    next_tower = jnp.moveaxis(next_time_major, 0, -2)
    next_front = next_tower[..., 0, :]
    next_future_dlvsq = updated_dlvsq[..., 1:, :]
    next_background = state.background_level * jnp.exp(
        -0.5 * BACKGROUND_ANNUAL_VOL**2 * dt
        + BACKGROUND_ANNUAL_VOL * jnp.sqrt(dt) * background_normal
    )
    latent_spot = jnp.sum(sampled_front * support, axis=-1)
    spot = latent_spot * next_background
    return PhysicalDailyStepResult(
        state=RiskNeutralState(next_front, next_future_dlvsq, next_background),
        density_tower=next_tower,
        risk_neutral_log_drift=risk_neutral_log_drift,
        empirical_log_drift=empirical_log_drift,
        achievable_log_drift=achievable_log_drift,
        market_price_of_risk=risk_price,
        sampled_front_index=sampled_index,
        latent_spot=latent_spot,
        spot=spot,
    )


def black_mixture_call_prices(
    support: Array,
    density_tower: Array,
    strikes: Array,
    background_level: Array,
    remaining_days: Array,
) -> Array:
    """Value grid calls under the independent 5% Black background."""
    total_vol = BACKGROUND_ANNUAL_VOL * jnp.sqrt(remaining_days / DAYS_PER_YEAR)
    component_spot = background_level[..., None, None] * support
    strike = strikes[..., None, :]
    component = component_spot[..., :, None]
    safe_vol = jnp.maximum(total_vol[..., :, None, None], 1.0e-12)
    d1 = (jnp.log(component / strike) + 0.5 * safe_vol**2) / safe_vol
    d2 = d1 - safe_vol
    black = component * jsp.special.ndtr(d1) - strike * jsp.special.ndtr(d2)
    return jnp.einsum("...ti,...tik->...tk", density_tower, black)


def black_mixture_call_second_moments(
    support: Array,
    density_tower: Array,
    strikes: Array,
    background_level: Array,
    remaining_days: Array,
) -> Array:
    """Return second moments of grid-call payoffs under the Black mixture."""
    total_vol = BACKGROUND_ANNUAL_VOL * jnp.sqrt(remaining_days / DAYS_PER_YEAR)
    component_spot = background_level[..., None, None] * support
    strike = strikes[..., None, :]
    component = component_spot[..., :, None]
    safe_vol = jnp.maximum(total_vol[..., :, None, None], 1.0e-12)
    variance = safe_vol**2
    log_moneyness = jnp.log(component / strike)
    d0 = (log_moneyness - 0.5 * variance) / safe_vol
    d1 = (log_moneyness + 0.5 * variance) / safe_vol
    d2 = (log_moneyness + 1.5 * variance) / safe_vol
    second_moment = (
        component**2 * jnp.exp(variance) * jsp.special.ndtr(d2)
        - 2.0 * component * strike * jsp.special.ndtr(d1)
        + strike**2 * jsp.special.ndtr(d0)
    )
    return jnp.einsum(
        "...ti,...tik->...tk",
        density_tower,
        second_moment,
    )


def black_mixture_call_exercise_probabilities(
    support: Array,
    density_tower: Array,
    strikes: Array,
    background_level: Array,
    remaining_days: Array,
) -> Array:
    """Return exercise probabilities for every Black-mixture grid call."""
    total_vol = BACKGROUND_ANNUAL_VOL * jnp.sqrt(remaining_days / DAYS_PER_YEAR)
    component_spot = background_level[..., None, None] * support
    strike = strikes[..., None, :]
    component = component_spot[..., :, None]
    safe_vol = jnp.maximum(total_vol[..., :, None, None], 1.0e-12)
    variance = safe_vol**2
    d0 = (jnp.log(component / strike) - 0.5 * variance) / safe_vol
    digital = jsp.special.ndtr(d0)
    return jnp.einsum("...ti,...tik->...tk", density_tower, digital)
