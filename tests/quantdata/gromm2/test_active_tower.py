from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.dlv import (
    decode_density_tower,
    decode_density_tower_from_front,
    prepare_sparse_transports,
)
from quantdata.gromm2.geometry import Gromm2GeometryConfig, expanding_supports, transport_tower
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("active_count", [1, 2, 4])
@pytest.mark.parametrize("substeps", [1, 8])
def test_padded_active_step_matches_the_unpadded_model(active_count: int, substeps: int) -> None:
    supports = expanding_supports(.2**2 * np.arange(1, 5) / 252., config=Gromm2GeometryConfig(num_inner_strikes=16))
    transports = transport_tower(supports)
    local_variance = np.full((4, 16), .18**2)
    intervals = jnp.full((3,), 1. / 252.)
    densities = decode_density_tower(jnp.asarray(supports), jnp.asarray(transports), jnp.asarray(local_variance), jnp.full((4,), 1. / 252.))
    generator = np.random.default_rng(10)
    loadings = jnp.asarray(generator.normal(0., .1, (3, 16, 2)))
    normals = jnp.asarray(generator.normal(size=(substeps, 4, 2)))
    scales = jnp.asarray([[.8, 1.1], [1., .9], [.9, 1.2], [1.2, .8]])
    state = SimulationState(jnp.broadcast_to(densities[0], (4, 18)), jnp.broadcast_to(local_variance, (4, 4, 16)), jnp.ones(4))
    padded = jax.jit(lambda count: one_day_risk_neutral_step(
        jnp.asarray(supports), prepare_sparse_transports(transports), intervals, state, loadings, normals, jnp.zeros(4),
        remaining_front_days=jnp.asarray(1), substeps=substeps, factor_scales=scales,
        active_maturity_count=count, linear_solver="pcr"))(jnp.asarray(active_count))
    unpadded = one_day_risk_neutral_step(jnp.asarray(supports[:active_count]),
        prepare_sparse_transports(transports[:active_count - 1]), intervals[:active_count - 1],
        SimulationState(state.front_density, state.dlvsq[:, :active_count], state.background_level),
        loadings[:active_count - 1], normals, jnp.zeros(4), remaining_front_days=jnp.asarray(1),
        substeps=substeps, factor_scales=scales, linear_solver="pcr")
    np.testing.assert_allclose(padded.spot, unpadded.spot, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(padded.density_tower[:, :active_count], unpadded.density_tower, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(padded.state.front_density, unpadded.state.front_density, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(padded.state.dlvsq[:, :active_count], unpadded.state.dlvsq, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(padded.drift_taming_fraction, unpadded.drift_taming_fraction, atol=1e-14)
    np.testing.assert_array_equal(padded.state.dlvsq[:, active_count:], state.dlvsq[:, active_count:])
    decoded = decode_density_tower_from_front(jnp.asarray(supports), prepare_sparse_transports(transports),
        state.front_density, state.dlvsq[:, 1:], intervals, linear_solver="pcr", active_maturity_count=jnp.asarray(active_count))
    np.testing.assert_allclose(decoded[:, :active_count], np.broadcast_to(densities[:active_count], (4, active_count, 18)), rtol=1e-12, atol=1e-14)
