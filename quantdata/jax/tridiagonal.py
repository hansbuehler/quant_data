"""Parallel cyclic reduction for batched tridiagonal linear systems."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def solve_tridiagonal_pcr(
    lower: jax.Array,
    diagonal: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Solve batched systems by parallel cyclic reduction, without pivot changes.

    Diagonals have shape ``(..., N)`` and matrix right-hand sides have shape
    ``(..., N, F)``.  Leading batch dimensions may broadcast.  All arrays must
    have the same floating dtype.  The algorithm performs ``ceil(log2(N))``
    simultaneous elimination stages and is suitable for positive-definite or
    diagonally dominant systems.  It neither clips coefficients nor
    regularizes singular pivots; zero or unstable pivots remain visible.
    """
    lower, diagonal, upper, rhs = map(jnp.asarray, (lower, diagonal, upper, rhs))
    if diagonal.ndim < 1 or diagonal.shape[-1] == 0 or lower.shape != diagonal.shape or upper.shape != diagonal.shape:
        raise ValueError("tridiagonal coefficients must have identical nonempty (..., N) shapes")
    if rhs.ndim < 2 or rhs.shape[-2] != diagonal.shape[-1]:
        raise ValueError("right-hand sides must have shape (..., N, F)")
    if any(value.dtype != rhs.dtype for value in (lower, diagonal, upper)) or not jnp.issubdtype(rhs.dtype, jnp.floating):
        raise ValueError("tridiagonal coefficients and right-hand sides must have one floating dtype")
    size = diagonal.shape[-1]
    indices = jnp.arange(size)
    for stage in range((size - 1).bit_length()):
        stride = 1 << stage
        left_diagonal = jnp.roll(diagonal, stride, axis=-1)
        right_diagonal = jnp.roll(diagonal, -stride, axis=-1)
        alpha = jnp.where(indices >= stride, -lower / left_diagonal, 0.0)
        beta = jnp.where(indices + stride < size, -upper / right_diagonal, 0.0)
        next_diagonal = diagonal + alpha * jnp.roll(upper, stride, axis=-1) + beta * jnp.roll(lower, -stride, axis=-1)
        next_rhs = rhs + alpha[..., None] * jnp.roll(rhs, stride, axis=-2) + beta[..., None] * jnp.roll(rhs, -stride, axis=-2)
        next_lower = alpha * jnp.roll(lower, stride, axis=-1)
        next_upper = beta * jnp.roll(upper, -stride, axis=-1)
        lower, diagonal, upper, rhs = next_lower, next_diagonal, next_upper, next_rhs
    return rhs / diagonal[..., None]
