"""Exact Gaussian information bridge for a discrete front density."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp

Array = jax.Array


def front_density_volatility(front_density: Array, *, remaining_time: Array) -> Array:
    """Return the exact instantaneous front-density Brownian loading."""
    cumulative = jnp.cumsum(front_density, axis=-1)[..., :-1]
    probability_floor = jnp.finfo(front_density.dtype).eps
    safe_probability = jnp.clip(
        cumulative,
        probability_floor,
        1.0 - probability_floor,
    )
    thresholds = jsp.special.ndtri(safe_probability)
    gaussian_density = jnp.exp(-0.5 * thresholds**2) / jnp.sqrt(2.0 * jnp.pi)
    gaussian_density = jnp.where(
        (cumulative <= 0.0) | (cumulative >= 1.0),
        0.0,
        gaussian_density,
    )
    boundary_zero = jnp.zeros_like(gaussian_density[..., :1])
    phi = jnp.concatenate(
        (boundary_zero, gaussian_density, boundary_zero), axis=-1
    )
    return (phi[..., :-1] - phi[..., 1:]) / jnp.sqrt(remaining_time)


def _normal_interval_probabilities(boundaries: Array) -> Array:
    direct = jnp.diff(jsp.special.ndtr(boundaries), axis=-1)
    survival = jsp.special.ndtr(-boundaries)
    upper_tail = survival[..., :-1] - survival[..., 1:]
    probabilities = jnp.where(boundaries[..., :-1] >= 0.0, upper_tail, direct)
    return jnp.maximum(probabilities, 0.0)


def sample_front_index(front_density: Array, normal: Array) -> Array:
    """Sample a terminal front node with a Gaussian copula coordinate."""
    probability = jsp.special.ndtr(normal)
    cumulative = jnp.cumsum(front_density, axis=-1)
    index = jnp.sum(probability[..., None] > cumulative, axis=-1)
    return jnp.minimum(index, front_density.shape[-1] - 1).astype(jnp.int32)


def advance_front_density(
    front_density: Array,
    normal: Array,
    *,
    remaining_time: Array,
    step_size: Array,
    final_step: Array,
) -> Array:
    """Apply one exact finite-step Gaussian-information-bridge transition."""

    def settle(_: None) -> Array:
        index = sample_front_index(front_density, normal)
        return jax.nn.one_hot(index, front_density.shape[-1], dtype=front_density.dtype)

    def reveal(_: None) -> Array:
        cumulative = jnp.cumsum(front_density, axis=-1)[..., :-1]
        floor = jnp.finfo(front_density.dtype).eps
        safe_probability = jnp.clip(cumulative, floor, 1.0 - floor)
        thresholds = jsp.special.ndtri(safe_probability)
        thresholds = jnp.where(cumulative <= 0.0, -jnp.inf, thresholds)
        thresholds = jnp.where(cumulative >= 1.0, jnp.inf, thresholds)
        shifted = (
            jnp.sqrt(remaining_time) * thresholds
            - jnp.sqrt(step_size) * normal[..., None]
        ) / jnp.sqrt(remaining_time - step_size)
        boundaries = jnp.concatenate(
            (
                jnp.full_like(shifted[..., :1], -jnp.inf),
                shifted,
                jnp.full_like(shifted[..., :1], jnp.inf),
            ),
            axis=-1,
        )
        density = _normal_interval_probabilities(boundaries)
        return density / jnp.sum(density, axis=-1, keepdims=True)

    return jax.lax.cond(final_step, settle, reveal, None)
