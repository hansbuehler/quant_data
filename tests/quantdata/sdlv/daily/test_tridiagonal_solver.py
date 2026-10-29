from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot import jax_model

jax.config.update("jax_enable_x64", True)


def sanos_case(*, matrix: bool, batched: bool, nodes: int = 202) -> tuple[jax.Array, ...]:
    support = jnp.asarray(np.exp(np.linspace(np.log(0.25), np.log(2.5), nodes)))
    factors = (1.0, 1.3) if batched else (1.0,)
    variances = np.asarray([factor * np.linspace(0.01, 8.0, nodes - 2) / 252.0 for factor in factors])
    if matrix:
        rhs = np.random.default_rng(914).normal(size=(len(factors), nodes, 16))
    else:
        initial = np.asarray(jax_model.density0_from_support(support))
        rhs = np.broadcast_to(initial, (len(factors), nodes)).copy()
    return support, jnp.asarray(rhs if batched else rhs[0]), jnp.asarray(variances if batched else variances[0])


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_v1_pcr_matches_native_and_preserves_mass_and_mean(matrix: bool, batched: bool) -> None:
    support, previous, variance_dt = sanos_case(matrix=matrix, batched=batched)
    actual = jax.jit(lambda rhs, variance: jax_model.solve_sanos_step(support, rhs, variance, linear_solver="pcr"))(previous, variance_dt)
    expected = jax_model.solve_sanos_step(support, previous, variance_dt, linear_solver="native")
    np.testing.assert_allclose(actual, expected, atol=2.0e-11, rtol=2.0e-11)
    node_axis = -2 if matrix else -1
    np.testing.assert_allclose(actual.sum(axis=node_axis), previous.sum(axis=node_axis), atol=2.0e-9, rtol=2.0e-11)
    weights = support[:, None] if matrix else support
    np.testing.assert_allclose((actual * weights).sum(axis=node_axis), (previous * weights).sum(axis=node_axis), atol=2.0e-9, rtol=2.0e-11)
    reconstructed = jax_model.apply_sanos_matrix(support, actual, variance_dt)
    np.testing.assert_allclose(reconstructed, previous, atol=2.0e-9, rtol=2.0e-10)


def test_v1_pcr_matches_dense_system_and_dense_gradients() -> None:
    support, previous, variance_dt = sanos_case(matrix=True, batched=False, nodes=10)
    def dense(rhs, variances):
        lower, diagonal, upper = jax_model._sanos_diagonals(support, variances)
        operator = jnp.diag(diagonal) + jnp.diag(lower[1:], k=-1) + jnp.diag(upper[:-1], k=1)
        return jnp.linalg.solve(operator, rhs)
    def pcr(rhs, variances):
        return jax_model.solve_sanos_step(support, rhs, variances, linear_solver="pcr")
    np.testing.assert_allclose(pcr(previous, variance_dt), dense(previous, variance_dt), atol=2.0e-13, rtol=2.0e-13)
    pcr_gradient = jax.grad(lambda rhs, variances: jnp.sum(pcr(rhs, variances) ** 2), argnums=(0, 1))(previous, variance_dt)
    dense_gradient = jax.grad(lambda rhs, variances: jnp.sum(dense(rhs, variances) ** 2), argnums=(0, 1))(previous, variance_dt)
    for actual, expected in zip(pcr_gradient, dense_gradient, strict=True):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, atol=2.0e-9, rtol=2.0e-11)


def test_cpu_default_remains_native_and_gpu_selection_uses_common_pcr(monkeypatch) -> None:
    support, previous, variance_dt = sanos_case(matrix=True, batched=True, nodes=10)
    original_native = jax.lax.linalg.tridiagonal_solve
    original_pcr = jax_model.solve_tridiagonal_pcr
    used = []
    def native(*args):
        used.append("native")
        return original_native(*args)
    def pcr(*args):
        used.append("pcr")
        return original_pcr(*args)
    monkeypatch.setattr(jax.lax.linalg, "tridiagonal_solve", native)
    monkeypatch.setattr(jax_model, "solve_tridiagonal_pcr", pcr)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    jax_model.solve_sanos_step(support, previous, variance_dt)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    jax_model.solve_sanos_step(support, previous, variance_dt)
    assert used == ["native", "pcr"]


def test_invalid_v1_solver_choice_fails_explicitly() -> None:
    arguments = sanos_case(matrix=False, batched=False, nodes=10)
    with pytest.raises(ValueError, match="linear_solver"):
        jax_model.solve_sanos_step(*arguments, linear_solver="unknown")
