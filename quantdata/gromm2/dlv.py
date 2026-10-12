"""Variable-support DLV decoding and risk-neutral drift equations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

Array = jax.Array
FloatArray = NDArray[np.float64]


def density0_from_support(support: Array) -> Array:
    """Represent a unit forward exactly on an ordered finite support."""
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
    """Solve one variable-support SANOS step for vector or matrix RHS."""
    lower, diagonal, upper = _sanos_diagonals(support, dlvsq_dt)
    vector_rhs = previous.ndim == dlvsq_dt.ndim
    rhs = previous[..., :, None] if vector_rhs else previous
    result = jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs)
    return result[..., 0] if vector_rhs else result


def apply_sanos_matrix(support: Array, value: Array, dlvsq_dt: Array) -> Array:
    """Apply the variable-support SANOS matrix."""
    lower, diagonal, upper = _sanos_diagonals(support, dlvsq_dt)
    vector_value = value.ndim == dlvsq_dt.ndim
    values = value[..., :, None] if vector_value else value
    result = diagonal[..., :, None] * values
    result = result.at[..., 1:, :].add(lower[..., 1:, None] * values[..., :-1, :])
    result = result.at[..., :-1, :].add(upper[..., :-1, None] * values[..., 1:, :])
    return result[..., 0] if vector_value else result


def decode_density_tower(
    supports: Array,
    transports: Array,
    dlvsq: Array,
    interval_years: Array,
) -> Array:
    """Decode an expiry tower with an expanding support at each expiry."""
    initial_density = density0_from_support(supports[0])
    initial_density = jnp.broadcast_to(
        initial_density,
        dlvsq.shape[:-2] + initial_density.shape,
    )
    first = solve_sanos_step(
        supports[0],
        initial_density,
        dlvsq[..., 0, :] * interval_years[0],
    )

    return decode_density_tower_from_front(
        supports,
        transports,
        first,
        dlvsq[..., 1:, :],
        interval_years[1:],
    )


def decode_density_tower_from_front(
    supports: Array,
    transports: Array,
    front_density: Array,
    future_dlvsq: Array,
    interval_years: Array,
) -> Array:
    """Decode a variable-support tower from its current front density."""

    def step(previous: Array, inputs: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
        support, transport, local_variance, interval = inputs
        transported = jnp.einsum("ij,...j->...i", transport, previous)
        current = solve_sanos_step(
            support,
            transported,
            local_variance * interval,
        )
        return current, current

    if supports.shape[0] == 1:
        return front_density[..., None, :]
    _, remainder = jax.lax.scan(
        step,
        front_density,
        (
            supports[1:],
            transports,
            jnp.moveaxis(future_dlvsq, -2, 0),
            interval_years,
        ),
    )
    remainder = jnp.moveaxis(remainder, 0, -2)
    return jnp.concatenate((front_density[..., None, :], remainder), axis=-2)


def risk_neutral_log_drift(
    supports: Array,
    transports: Array,
    front_density: Array,
    future_dlvsq: Array,
    interval_years: Array,
    log_dlvsq_loadings: Array,
    front_density_loading: Array,
) -> tuple[Array, Array, Array]:
    """Return densities, density loadings and the Q drift of log-DLV-squared.

    ``supports`` contains the front support followed by one support per future
    interval. ``transports`` maps each preceding support to the next. The
    diffusion relation is solved exactly before applying the Itô correction.
    """
    factor_count = log_dlvsq_loadings.shape[-1]
    if front_density_loading.shape[-1] != factor_count:
        raise ValueError("front and DLV loadings must have the same factor count")

    def step(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        previous_density, previous_loading = carry
        support, transport, local_variance, interval, current_xi = inputs
        transported_density = jnp.einsum("ij,...j->...i", transport, previous_density)
        transported_loading = jnp.einsum(
            "ij,...jf->...if", transport, previous_loading
        )
        current_density = solve_sanos_step(
            support,
            transported_density,
            local_variance * interval,
        )
        boundary_zeros = jnp.zeros(
            current_xi.shape[:-2] + (1, factor_count),
            dtype=current_xi.dtype,
        )
        extended_xi = jnp.concatenate(
            (boundary_zeros, current_xi, boundary_zeros), axis=-2
        )
        xi_density = extended_xi * current_density[..., :, None]
        matrix_value = apply_sanos_matrix(
            support,
            xi_density,
            local_variance * interval,
        )
        current_loading = solve_sanos_step(
            support,
            transported_loading + xi_density - matrix_value,
            local_variance * interval,
        )
        safe_density = jnp.maximum(
            current_density[..., 1:-1, None],
            jnp.finfo(current_density.dtype).tiny,
        )
        relative_density_loading = current_loading[..., 1:-1, :] / safe_density
        relative_level_drift = -jnp.sum(
            current_xi * relative_density_loading,
            axis=-1,
        )
        log_drift = relative_level_drift - 0.5 * jnp.sum(current_xi**2, axis=-1)
        return (current_density, current_loading), (
            current_density,
            current_loading,
            log_drift,
        )

    _, outputs = jax.lax.scan(
        step,
        (front_density, front_density_loading),
        (
            supports[1:],
            transports,
            jnp.moveaxis(future_dlvsq, -2, 0),
            interval_years,
            jnp.moveaxis(log_dlvsq_loadings, -3, 0),
        ),
    )
    densities, density_loadings, log_drift = outputs
    return (
        jnp.moveaxis(densities, 0, -2),
        jnp.moveaxis(density_loadings, 0, -3),
        jnp.moveaxis(log_drift, 0, -2),
    )


def infer_dlvsq_from_densities(
    support: NDArray[np.floating],
    previous_density: NDArray[np.floating],
    current_density: NDArray[np.floating],
    *,
    interval_years: float,
) -> FloatArray:
    """Invert one exact DLV step by a stable banded least-squares solve."""
    x = np.asarray(support, dtype=np.float64)
    previous = np.asarray(previous_density, dtype=np.float64)
    current = np.asarray(current_density, dtype=np.float64)
    if (
        x.ndim != 1
        or previous.shape != x.shape
        or current.shape != x.shape
        or len(x) < 3
        or interval_years <= 0.0
    ):
        raise ValueError("invalid DLV inversion geometry")
    interior = len(x) - 2
    box = x[1:-1] ** 2 / (x[2:] - x[:-2])
    plus = box / (x[2:] - x[1:-1])
    minus = box / (x[1:-1] - x[:-2])
    operator = np.zeros((len(x), interior), dtype=np.float64)
    columns = np.arange(interior)
    operator[columns, columns] = -minus * current[1:-1]
    operator[columns + 1, columns] = (minus + plus) * current[1:-1]
    operator[columns + 2, columns] = -plus * current[1:-1]
    right_hand_side = (previous - current) / interval_years
    scale = np.maximum(np.sqrt(np.maximum(current, 0.0)), 1.0e-7)
    weighted_operator = operator / scale[:, None]
    weighted_rhs = right_hand_side / scale
    result, *_ = np.linalg.lstsq(weighted_operator, weighted_rhs, rcond=1.0e-11)
    return result
