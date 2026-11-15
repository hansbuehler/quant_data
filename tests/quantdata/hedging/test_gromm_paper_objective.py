from dataclasses import fields
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.hedging.model_audit.gromm_paper_objective import observed_view, paper_view_loss


def test_paper_loss_is_linear_excess_with_five_percent_midpoint_term(monkeypatch):
    calibration = pytest.importorskip("openqdata.models.gromm.calibrate")
    import jax.numpy as jnp

    monkeypatch.setattr(calibration, "_view_prices", lambda *args: jnp.array([2., 3., 4.]))
    raw = jnp.full((1, 4), np.log(.04))
    result = paper_view_loss(raw, None, None, jnp.array([False, True, True, False]),
        jnp.empty(0), None, None, None, None, None, None, jnp.full(3, 2.), jnp.ones(3), jnp.ones(3))
    assert float(result) == pytest.approx((0. + .05 + 1.10) / 3., abs=1e-7)


def test_cap_supplies_restoring_gradient_without_curvature_penalties(monkeypatch):
    calibration = pytest.importorskip("openqdata.models.gromm.calibrate")
    import jax
    import jax.numpy as jnp

    monkeypatch.setattr(calibration, "_view_prices", lambda *args: jnp.ones(1))
    active = jnp.array([False, True, False])

    def objective(raw):
        return paper_view_loss(raw, None, None, active, jnp.empty(0), None, None, None,
            None, None, None, jnp.ones(1), jnp.full(1, 1e-8), jnp.ones(1))

    raw = jnp.full((1, 3), np.log(9. - 1e-4) + .2)
    gradient = np.asarray(jax.grad(objective)(raw))
    assert gradient[0, 1] > 0.
    np.testing.assert_array_equal(gradient[0, [0, 2]], 0.)


def test_observed_target_reuses_geometry_but_not_fitted_sanos_prices():
    calibration = pytest.importorskip("openqdata.models.gromm.calibrate")
    from openqdata.models.gromm.config import DAY

    day = {"tte": np.array([DAY, 2 * DAY]), "k": np.array([1., 1.05]),
        "bid": np.array([.4, .5]), "ask": np.array([.6, .9])}
    half = .5 * (day["ask"] - day["bid"])
    arguments = {field.name: np.zeros(2) for field in fields(calibration.View)}
    arguments.update(date=date(2024, 4, 1), kind="start", ratio=1., tte=day["tte"], k=day["k"],
        hs=half, sanos=np.array([.42, .68]), report=np.array([True, False]), weight=np.array([1., .01]))
    view = calibration.View(**arguments)
    geometry = SimpleNamespace(config=SimpleNamespace(model_days=2),
        strikes=np.broadcast_to([.8, 1., 1.2], (2, 3)))
    adjusted = observed_view(view, geometry, day)
    np.testing.assert_array_equal(adjusted.sanos, .5 * (day["bid"] + day["ask"]))
    np.testing.assert_array_equal(adjusted.weight, [1., 0.])
    assert adjusted.kernels0 is view.kernels0
    assert adjusted.kernels1 is view.kernels1
