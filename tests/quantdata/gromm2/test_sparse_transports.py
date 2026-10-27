from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.dlv import (
    apply_transport,
    decode_density_tower,
    decode_density_tower_from_front,
    prepare_sparse_transports,
    risk_neutral_log_drift,
)
from quantdata.gromm2.geometry import Gromm2GeometryConfig, expanding_supports, transport_tower

jax.config.update("jax_enable_x64", True)


def sparse_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expiries = np.asarray([1.0, 3.0, 8.0]) / 252.0
    supports = expanding_supports(0.2**2 * expiries, config=Gromm2GeometryConfig(num_inner_strikes=16))
    return expiries, supports, transport_tower(supports)


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_sparse_operator_matches_dense_for_vectors_and_matrices(matrix: bool, batch_shape: tuple[int, ...]) -> None:
    _, supports, transports = sparse_geometry()
    generator = np.random.default_rng(413)
    shape = batch_shape + (supports.shape[1],) + ((4,) if matrix else ())
    values = jnp.asarray(generator.normal(size=shape))
    sparse = prepare_sparse_transports(transports[0])
    actual = jax.jit(lambda right: apply_transport(sparse, right, matrix=matrix))(values)
    expected = apply_transport(jnp.asarray(transports[0]), values, matrix=matrix)
    np.testing.assert_allclose(actual, expected, atol=2.0e-15, rtol=2.0e-15)


def test_sparse_density_and_q_drift_match_dense_expiry_scans() -> None:
    expiries, supports, transports = sparse_geometry()
    intervals = np.diff(expiries, prepend=0.0)
    local_variance = jnp.full((2, len(expiries), supports.shape[1] - 2), 0.18**2)
    support = jnp.asarray(supports)
    sparse = prepare_sparse_transports(transports)
    dense = jnp.asarray(transports)
    expected = decode_density_tower(support, dense, local_variance, jnp.asarray(intervals))
    actual = jax.jit(decode_density_tower)(support, sparse, local_variance, jnp.asarray(intervals))
    np.testing.assert_allclose(actual, expected, atol=1.0e-14, rtol=1.0e-13)
    future = local_variance[:, 1:]
    decoded = jax.jit(decode_density_tower_from_front)(support, sparse, expected[:, 0], future, jnp.asarray(intervals[1:]))
    np.testing.assert_allclose(decoded, expected, atol=1.0e-14, rtol=1.0e-13)
    generator = np.random.default_rng(15)
    loadings = jnp.asarray(generator.normal(scale=0.1, size=future.shape + (3,)))
    front_loading = jnp.asarray(generator.normal(scale=0.1, size=expected[:, 0].shape + (3,)))
    arguments = (expected[:, 0], future, jnp.asarray(intervals[1:]), loadings, front_loading)
    dense_result = risk_neutral_log_drift(support, dense, *arguments)
    sparse_result = jax.jit(risk_neutral_log_drift)(support, sparse, *arguments)
    for actual_value, expected_value in zip(sparse_result, dense_result, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, atol=1.0e-11, rtol=1.0e-11)


def test_sparse_map_does_not_drop_tiny_coefficients_or_duplicate_mass() -> None:
    dense = np.asarray([[1.0, 0.0, 0.0], [1.0e-30, 0.75, 0.0], [0.0, 0.25, 1.0]])
    sparse = prepare_sparse_transports(dense)
    np.testing.assert_array_equal(apply_transport(sparse, jnp.asarray([1.0, 0.0, 0.0])), dense[:, 0])
    np.testing.assert_array_equal(apply_transport(sparse, jnp.asarray([0.0, 1.0, 0.0])), dense[:, 1])
    with pytest.raises(ValueError, match="at most two"):
        prepare_sparse_transports(np.ones((3, 3)))


@pytest.mark.parametrize("dense", [np.ones((2, 3)), np.eye(3)[None, None], np.asarray([[np.nan]])])
def test_invalid_sparse_geometry_is_rejected(dense: np.ndarray) -> None:
    with pytest.raises(ValueError):
        prepare_sparse_transports(dense)
