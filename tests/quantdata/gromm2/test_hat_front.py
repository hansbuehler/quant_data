from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from quantdata.gromm2 import hat_front
from quantdata.gromm2.dlv import prepare_sparse_transports
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step
from tests.quantdata.gromm2.test_forward_hedging import small_model


def wide_tail_front():
    """Dense core with one far node on each side, as on the one-day SPX front."""
    core = np.linspace(0.98, 1.012, 201)
    support = np.r_[0.49, core, 1.52]
    density = np.r_[1.4e-4, norm.pdf(core, 1.0, 0.006), 6.7e-7]
    density[1:-1] *= (1.0 - density[0] - density[-1]) / density[1:-1].sum()
    return jnp.asarray(support), jnp.asarray(density)


def test_max_entropy_matches_density_with_wide_tail_cells():
    support, density = wide_tail_front()
    interval = hat_front.start_interval(density, support, remaining_time=jnp.asarray(1.0 / 252.0))
    i0, i1, _ = hat_front._cell_integrals(interval.lam, interval.width)
    np.testing.assert_allclose(hat_front._project(i0 - i1, i1), density, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(i0), 1.0, atol=1e-12)


def test_settlement_spot_is_unbiased_with_wide_tail_cells():
    support, density = wide_tail_front()
    tau = 1.0 / 252.0
    interval = hat_front.start_interval(density, support, remaining_time=jnp.asarray(tau))
    probability = (np.arange(20000) + 0.5) / 20000
    w = jnp.asarray(np.sqrt(tau) * norm.ppf(probability))
    settled = jax.vmap(lambda x: hat_front.settle_density(interval, x))(w) @ support
    assert abs(float(jnp.mean(settled)) - float(density @ support)) < 1e-6


def test_front_argument_defaults_to_bridge_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("GROMM2_FRONT", raising=False)
    model = small_model()
    paths, factors, substeps = 2, model["loadings"].shape[-1], 2
    state = SimulationState(jnp.broadcast_to(model["densities"][0], (paths, model["supports"].shape[-1])),
        jnp.broadcast_to(model["dlvsq"], (paths,) + model["dlvsq"].shape), jnp.ones(paths))
    generator = np.random.default_rng(7)
    arguments = (jnp.asarray(model["supports"]), prepare_sparse_transports(model["transports"]),
        jnp.asarray(np.diff(model["expiries_y"])), state, jnp.asarray(model["loadings"]),
        jnp.asarray(generator.normal(size=(substeps, paths, factors))), jnp.asarray(generator.normal(size=paths)))
    options = dict(remaining_front_days=1, substeps=substeps, factor_scales=jnp.ones((paths, factors)))
    default = one_day_risk_neutral_step(*arguments, **options)
    bridge = one_day_risk_neutral_step(*arguments, front="bridge", **options)
    hat = one_day_risk_neutral_step(*arguments, front="hat", **options)
    np.testing.assert_array_equal(default.density_tower, bridge.density_tower)
    assert np.isfinite(np.asarray(hat.density_tower)).all()
    assert not np.allclose(hat.density_tower[..., 0, :], bridge.density_tower[..., 0, :])
    with pytest.raises(ValueError):
        one_day_risk_neutral_step(*arguments, front="other", **options)
