"""Recursive V2 diagnostics preserve the approved one-day factor law."""

from types import SimpleNamespace

import numpy as np
from cdxcore.bs import bs

from quantdata.hedging.model_audit.standalone_v2_diagnostics import invert_capped, recursive_states
from quantdata.hedging.model_audit.standalone_v2_forecast import sample_next


def test_recursive_first_day_matches_approved_one_day_law():
    fit = SimpleNamespace(cov=np.array([[.1, .02], [.02, .05]]), scores=np.array([[.3]]),
        A=np.array([[.8]]), alpha=np.array([.02, -.01]), beta=np.array([.1, -.3]),
        directions=np.array([[.2], [.4]]), r_mean=.001)
    current = np.array([.7, -.2])
    expected, spot = sample_next(current, fit, paths=8, seed=4)
    snapshots, history, first_spot = recursive_states(current, fit, paths=8, horizons=(1, 3),
        long_paths=4, long_steps=2, seed=4)
    np.testing.assert_allclose(snapshots[1], expected, rtol=0., atol=1e-15)
    np.testing.assert_allclose(first_spot, spot, rtol=0., atol=1e-15)
    np.testing.assert_allclose(history[0], expected[:4], rtol=0., atol=1e-15)
    assert snapshots[3].shape == (8, 2)


def test_capped_iv_is_retained_and_repricing_failure_remains_visible():
    prices = np.array([bs.price(k=1., vol=6., sqrtT=1.), np.nan, 1.2])
    diagnostics = []
    vols = invert_capped(prices, np.ones(3), np.ones(3), diagnostics)
    assert vols[0] == 5.
    assert np.isnan(vols[1:]).all()
    assert diagnostics[0]["library_max_vol_cases"] == 1
    assert diagnostics[0]["failed"] == 1
