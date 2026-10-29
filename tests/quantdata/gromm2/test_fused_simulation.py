from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.dlv import prepare_sparse_transports
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step
from tests.quantdata.gromm2.test_forward_hedging import small_model


@pytest.mark.parametrize("solver", ["thomas", "pcr"])
@pytest.mark.parametrize("remaining", [1, 2])
def test_streamed_drift_matches_full_loading_tower(solver, remaining):
    model = small_model()
    generator = np.random.default_rng(146)
    model["loadings"] = generator.normal(scale=0.02, size=model["loadings"].shape)
    paths, factors, substeps = 4, 2, 3
    scales = jnp.asarray(generator.uniform(0.7, 1.3, (paths, factors)))
    loadings = jnp.asarray(model["loadings"])
    state = SimulationState(jnp.broadcast_to(model["densities"][0], (paths, model["supports"].shape[-1])),
        jnp.broadcast_to(model["dlvsq"], (paths,) + model["dlvsq"].shape), jnp.ones(paths))
    arguments = (jnp.asarray(model["supports"]), prepare_sparse_transports(model["transports"]),
        jnp.asarray(np.diff(model["expiries_y"])), state)
    normals = jnp.asarray(generator.normal(size=(substeps, paths, factors)))
    background = jnp.asarray(generator.normal(size=paths))
    options = dict(remaining_front_days=jnp.asarray(remaining), substeps=substeps, linear_solver=solver)
    original = one_day_risk_neutral_step(*arguments, loadings[None] * scales[:, None, None, :], normals, background, **options)
    fused = one_day_risk_neutral_step(*arguments, loadings, normals, background, factor_scales=scales, **options)
    np.testing.assert_allclose(fused.density_tower, original.density_tower, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(fused.state.dlvsq, original.state.dlvsq, atol=2e-12, rtol=2e-11)
    np.testing.assert_array_equal(fused.spot, original.spot)
    np.testing.assert_array_equal(fused.daily_factor_score, original.daily_factor_score)
