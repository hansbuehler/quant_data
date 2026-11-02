"""Exact Gaussian information bridge for a discrete front density."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp

Array = jax.Array


def _front_normal_thresholds(front_density: Array) -> Array:
    """Invert the smaller cumulative tail without subtracting it from one."""
    total = jnp.sum(front_density, axis=-1, keepdims=True)
    lower = jnp.cumsum(front_density, axis=-1)[..., :-1] / total
    upper = jnp.flip(jnp.cumsum(jnp.flip(front_density, axis=-1), axis=-1), axis=-1)[..., 1:] / total
    lower_is_smaller = lower <= upper
    tail_probability = jnp.where(lower_is_smaller, lower, upper)
    positive = tail_probability > 0.
    quantile = jsp.special.ndtri(jnp.where(positive, tail_probability, .5))
    signed = jnp.where(lower_is_smaller, quantile, -quantile)
    return jnp.where(positive, signed, jnp.where(lower_is_smaller, -jnp.inf, jnp.inf))


def _narrow_bin_quadrature(front_density: Array) -> tuple[Array, Array, Array, Array]:
    """Integrate narrow probability bins without subtracting rounded quantiles."""
    total = jnp.sum(front_density, axis=-1, keepdims=True)
    mass = front_density / total
    zero = jnp.zeros_like(mass[..., :1])
    lower = jnp.concatenate((zero, jnp.cumsum(mass, axis=-1)[..., :-1]), axis=-1)
    reverse = jnp.flip(jnp.cumsum(jnp.flip(mass, axis=-1), axis=-1), axis=-1)
    upper = jnp.concatenate((reverse[..., 1:], zero), axis=-1)
    narrow = (mass > 0.) & (mass <= .001 * jnp.minimum(lower, upper))
    fractions = jnp.asarray((.046910077030668, .2307653449471585, .5,
        .7692346550528415, .953089922969332), dtype=mass.dtype)
    weights = jnp.asarray((.1184634425280945, .2393143352496832, .2844444444444444,
        .2393143352496832, .1184634425280945), dtype=mass.dtype)
    lower_probability = lower[..., None] + mass[..., None] * fractions
    upper_probability = upper[..., None] + mass[..., None] * (1. - fractions)
    lower_is_smaller = lower_probability <= upper_probability
    probability = jnp.minimum(lower_probability, upper_probability)
    quantile = jsp.special.ndtri(jnp.where(narrow[..., None], probability, .5))
    quantile = jnp.where(lower_is_smaller, quantile, -quantile)
    return mass, narrow, quantile, weights


def front_density_volatility(front_density: Array, *, remaining_time: Array) -> Array:
    """Return the exact instantaneous front-density Brownian loading."""
    thresholds = _front_normal_thresholds(front_density)
    finite = jnp.isfinite(thresholds)
    safe_thresholds = jnp.where(finite, thresholds, 0.)
    gaussian_density = jnp.where(finite, jnp.exp(-0.5 * safe_thresholds**2) / jnp.sqrt(2.0 * jnp.pi), 0.)
    boundary_zero = jnp.zeros_like(gaussian_density[..., :1])
    phi = jnp.concatenate(
        (boundary_zero, gaussian_density, boundary_zero), axis=-1
    )
    direct = phi[..., :-1] - phi[..., 1:]
    mass, narrow, quantile, weights = _narrow_bin_quadrature(front_density)
    integrated = mass * jnp.sum(quantile * weights, axis=-1)
    return jnp.where(narrow, integrated, direct) / jnp.sqrt(remaining_time)


def _normal_interval_probabilities(boundaries: Array) -> Array:
    direct = jnp.diff(jsp.special.ndtr(boundaries), axis=-1)
    survival = jsp.special.ndtr(-boundaries)
    upper_tail = survival[..., :-1] - survival[..., 1:]
    probabilities = jnp.where(boundaries[..., :-1] >= 0.0, upper_tail, direct)
    return jnp.maximum(probabilities, 0.0)


def sample_front_index(front_density: Array, normal: Array) -> Array:
    """Sample a terminal front node with a Gaussian copula coordinate."""
    thresholds = _front_normal_thresholds(front_density)
    index = jnp.sum(normal[..., None] > thresholds, axis=-1)
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
        thresholds = _front_normal_thresholds(front_density)
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
        mass, narrow, quantile, weights = _narrow_bin_quadrature(front_density)
        shifted_quantile = (jnp.sqrt(remaining_time) * quantile
            - jnp.sqrt(step_size) * normal[..., None, None]) / jnp.sqrt(remaining_time - step_size)
        log_ratio = (.5 * jnp.log(remaining_time / (remaining_time - step_size))
            + .5 * quantile**2 - .5 * shifted_quantile**2)
        safe_mass = jnp.where(narrow, mass, 1.)
        log_width = (jnp.log(safe_mass) + .5 * jnp.log(remaining_time / (remaining_time - step_size))
            + .5 * quantile[..., 2]**2 + .5 * jnp.log(2. * jnp.pi))
        quadrature_allowed = narrow & (log_width <= jnp.log(.001))
        integrated = jnp.exp(jnp.log(safe_mass)
            + jsp.special.logsumexp(log_ratio + jnp.log(weights), axis=-1))
        density = jnp.where(quadrature_allowed, integrated, density)
        return density / jnp.sum(density, axis=-1, keepdims=True)

    return jax.lax.cond(final_step, settle, reveal, None)
