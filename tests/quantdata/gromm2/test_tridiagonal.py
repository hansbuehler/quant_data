from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_banded

from quantdata.gromm2.dlv import _solve_tridiagonal_thomas
from quantdata.gromm2.tridiagonal import solve_tridiagonal_pcr

jax.config.update("jax_enable_x64", True)


def stable_system(nodes: int, columns: int, batched: bool) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(2853 + nodes + columns)
    batch = (2,) if batched else ()
    off_diagonal = -generator.uniform(0.0, 10.0, size=batch + (nodes - 1,))
    zero = np.zeros(batch + (1,))
    lower = np.concatenate((zero, off_diagonal), axis=-1)
    upper = np.concatenate((off_diagonal, zero), axis=-1)
    diagonal = 1.0 + np.abs(lower) + np.abs(upper)
    rhs = generator.normal(size=batch + (nodes, columns))
    return lower, diagonal, upper, rhs


def banded_reference(arguments: tuple[np.ndarray, ...]) -> np.ndarray:
    lower, diagonal, upper, rhs = arguments
    if diagonal.ndim > 1:
        return np.stack([banded_reference(tuple(value[index] for value in arguments)) for index in range(len(diagonal))])
    band = np.zeros((3, len(diagonal)))
    band[0, 1:] = upper[:-1]
    band[1] = diagonal
    band[2, :-1] = lower[1:]
    return solve_banded((1, 1), band, rhs)


@pytest.mark.parametrize("nodes", [1, 2, 3, 200, 202])
@pytest.mark.parametrize("columns", [1, 16])
@pytest.mark.parametrize("batched", [False, True])
def test_pcr_matches_scipy_thomas_and_dense_residual(nodes: int, columns: int, batched: bool) -> None:
    values = stable_system(nodes, columns, batched)
    arguments = tuple(jnp.asarray(value) for value in values)
    actual = np.asarray(jax.jit(solve_tridiagonal_pcr)(*arguments))
    np.testing.assert_allclose(actual, banded_reference(values), atol=2.0e-14, rtol=2.0e-13)
    thomas = np.asarray(_solve_tridiagonal_thomas(*arguments))
    np.testing.assert_allclose(actual, thomas, atol=2.0e-14, rtol=2.0e-13)
    lower, diagonal, upper, rhs = values
    product = diagonal[..., :, None] * actual
    product[..., 1:, :] += lower[..., 1:, None] * actual[..., :-1, :]
    product[..., :-1, :] += upper[..., :-1, None] * actual[..., 1:, :]
    np.testing.assert_allclose(product, rhs, atol=2.0e-14, rtol=2.0e-13)
    assert actual.dtype == np.float64


def test_pcr_gradients_match_dense_linear_solve() -> None:
    arguments = tuple(jnp.asarray(value) for value in stable_system(7, 3, False))
    def dense(lower, diagonal, upper, rhs):
        matrix = jnp.diag(diagonal) + jnp.diag(lower[1:], k=-1) + jnp.diag(upper[:-1], k=1)
        return jnp.linalg.solve(matrix, rhs)
    def objective(solver, lower, diagonal, upper, rhs):
        return jnp.sum(solver(lower, diagonal, upper, rhs) ** 2)
    pcr_gradients = jax.grad(lambda *args: objective(solve_tridiagonal_pcr, *args), argnums=(0, 1, 2, 3))(*arguments)
    dense_gradients = jax.grad(lambda *args: objective(dense, *args), argnums=(0, 1, 2, 3))(*arguments)
    for actual, expected in zip(pcr_gradients, dense_gradients, strict=True):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, atol=5.0e-13, rtol=5.0e-12)


def test_pcr_preserves_float32_and_broadcasts_unbatched_coefficients() -> None:
    values = tuple(jnp.asarray(value, dtype=jnp.float32) for value in stable_system(11, 2, False))
    lower, diagonal, upper, rhs = values
    batched_rhs = jnp.stack((rhs, 2.0 * rhs))
    result = solve_tridiagonal_pcr(lower, diagonal, upper, batched_rhs)
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result[1], 2.0 * result[0], atol=2.0e-6, rtol=2.0e-6)


def test_pcr_rejects_incompatible_shapes_or_dtypes() -> None:
    lower, diagonal, upper, rhs = tuple(jnp.asarray(value) for value in stable_system(5, 2, False))
    with pytest.raises(ValueError, match="identical nonempty"):
        solve_tridiagonal_pcr(lower[:-1], diagonal, upper, rhs)
    with pytest.raises(ValueError, match="right-hand sides"):
        solve_tridiagonal_pcr(lower, diagonal, upper, rhs[:-1])
    with pytest.raises(ValueError, match="one floating dtype"):
        solve_tridiagonal_pcr(lower, diagonal, upper, rhs.astype(jnp.float32))


def test_singular_pivots_are_not_silently_regularized() -> None:
    result = solve_tridiagonal_pcr(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.ones((1, 1)))
    assert not np.all(np.isfinite(result))
