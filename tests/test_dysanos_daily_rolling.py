"""Mathematical and transfer invariants for the daily rolling DYSANOS fit."""

from datetime import date, timedelta

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.dysanos.v11.rolling_daily import (
    DeviceWindow,
    RollingConfig,
    daily_dynamics,
    date_weights,
    initialize_daily,
)


def test_exponential_date_weights_have_requested_endpoints_and_ratio():
    weights = date_weights(250)
    assert weights[0] == pytest.approx(.25)
    assert weights[-1] == 1.
    np.testing.assert_allclose(weights[1:]/weights[:-1], 4**(1/249))


def test_window_advances_one_day_and_preserves_other_device_buffers():
    days = tuple(date(2022, 1, 1)+timedelta(days=i) for i in range(10))
    calls = []
    def loader(day):
        calls.append(day)
        return {"strikes": np.full((1, 3), day.day, dtype=np.float64),
                "mask": np.ones((1, 3), dtype=bool)}
    window = DeviceWindow(days, loader, batch_size=2)
    retained = window.days[1]["strikes"]
    retained_batch = window.batches[1]
    new_day = date(2022, 1, 11)
    replaced = window.advance(new_day)
    assert replaced == 0
    assert calls == [*days, new_day]
    assert window.transferred_days == 11
    assert window.days[1]["strikes"] is retained
    assert window.batches[1] is retained_batch
    assert tuple(window.slot_dates[i] for i in window.order) == (*days[1:], new_day)
    config = RollingConfig(window=10, state_size=2, batch_size=2)
    np.testing.assert_allclose(np.asarray(window.weights(config))[window.order], date_weights(10))


def test_weighted_regauging_preserves_all_decoded_states():
    rng = np.random.default_rng(42)
    affine = {"h": rng.normal(size=(10, 2))*[2, 7]+[4, -3],
              "B": rng.normal(size=(2, 8)), "b": rng.normal(size=8)}
    config = RollingConfig(window=10, state_size=2, batch_size=2)
    initial = initialize_daily(affine, config)
    np.testing.assert_allclose(initial["b"]+initial["h"]@initial["B"], affine["b"]+affine["h"]@affine["B"], atol=1e-10)
    w = date_weights(10)
    np.testing.assert_allclose(np.average(initial["h"], weights=w, axis=0), 0, atol=1e-12)
    np.testing.assert_allclose((initial["h"].T*w)@initial["h"]/w.sum(), np.eye(2), atol=1e-12)


def test_ar_uses_one_session_and_endpoint_weights():
    config = RollingConfig(window=4, state_size=2, batch_size=2)
    states = jnp.array([[0., 1.], [1., .5], [2., .25], [3., 0.]])
    damping = jnp.diag(jnp.sqrt(jnp.array([.1, .2])))
    params = {"h": states, "damping": damping, "rotation": jnp.zeros((2, 2)),
              "mean": jnp.zeros(2), "q_raw": jnp.zeros((2, 2))}
    weights = jnp.asarray(date_weights(4))
    (_, (nll, _)), grad = jax.value_and_grad(lambda p: daily_dynamics(p, jnp.arange(4), weights, config), has_aux=True)(params)
    transition = np.diag(np.exp(-np.array([.1, .2])-config.minimum_daily_rate))
    residual = np.asarray(states)[1:]-np.asarray(states)[:-1]@transition
    std = config.innovation_std_floor+np.log(2.)
    expected = .5*(np.average(np.sum((residual/std)**2, axis=1), weights=np.asarray(weights)[1:])
                   +4*np.log(std)+2*np.log(2*np.pi))/2
    assert float(nll) == pytest.approx(expected, rel=1e-5)
    assert all(np.isfinite(np.asarray(x)).all() for x in grad.values())
    # A return spanning a missing source session is not a one-session pair.
    valid = jnp.array([True, False, True])
    _, (masked_nll, _) = daily_dynamics(params, jnp.arange(4), weights, config, valid)
    expected_masked = .5*(np.average(np.sum((residual/std)**2, axis=1)[[0, 2]], weights=np.asarray(weights)[1:][[0, 2]])
                          +4*np.log(std)+2*np.log(2*np.pi))/2
    assert float(masked_nll) == pytest.approx(expected_masked, rel=1e-5)
