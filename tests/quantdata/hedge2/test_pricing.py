"""Backward DLV induction must agree with forward marginals and payoff timing."""

import numpy as np
import pytest

from quantdata.hedge2.dlv import DlvConfig, decode_densities, shared_inner_strikes
from quantdata.hedge2.pricing import DlvPricer


def _pricer() -> DlvPricer:
    config = DlvConfig(num_inner_strikes=20, business_dtes=(1., 2., 3., 4., 5.))
    inner = shared_inner_strikes(np.full(5, 0.2), config=config)
    return DlvPricer(support=np.r_[0.5 * inner[0], inner, 1.5 * inner[-1]],
                     expiries_y=config.expiries_y, dlvsqs=np.full((5, 20), 0.04))


def test_adjoint_preserves_cash_and_spot_on_nonuniform_grid() -> None:
    pricer = _pricer()
    np.testing.assert_allclose(pricer.backward_interval(np.ones(len(pricer.support)), 2), 1., atol=1e-13)
    np.testing.assert_allclose(pricer.backward_interval(pricer.support, 2), pricer.support, atol=1e-13)


def test_backward_vanillas_equal_forward_density_prices() -> None:
    pricer = _pricer()
    densities = decode_densities(pricer.support, pricer.dlvsqs, pricer.expiries_y)
    strikes = np.array([0.98, 1., 1.02])
    expected = densities[-1] @ np.maximum(pricer.support[:, None] - strikes, 0.)
    np.testing.assert_allclose(pricer.european(expiry_index=4, strikes=strikes), expected, atol=1e-13)


def test_single_ko_with_certain_survival_equals_vanilla() -> None:
    pricer = _pricer()
    expected = pricer.european(expiry_index=4, strikes=np.array([1.]))[0]
    assert pricer.single_observation_ko(ko_index=2, terminal_index=4, strike=1.,
                                       barrier=0.01, is_call=True) == pytest.approx(expected)
    knocked = pricer.single_observation_ko(ko_index=2, terminal_index=4, strike=1.,
                                          barrier=1., is_call=True)
    assert 0 <= knocked <= expected


def test_forward_start_resets_to_observed_ratio() -> None:
    original = _pricer()
    variances = original.dlvsqs.copy()
    variances[3:] = 0.
    pricer = DlvPricer(support=original.support, expiries_y=original.expiries_y, dlvsqs=variances)
    assert pricer.forward_start_straddle(reset_index=2, terminal_index=4, strike_ratio=1.) == pytest.approx(0., abs=1e-14)
    assert pricer.forward_start_straddle(reset_index=2, terminal_index=4, strike_ratio=1.02) == pytest.approx(0.02, abs=1e-13)
    assert pricer.forward_start_straddle(reset_index=2, terminal_index=4, strike_ratio=1.,
                                         forward_ratio=1.03) == pytest.approx(0.03, abs=1e-13)


def test_european_interpolation_and_right_sided_digital_derivative() -> None:
    pricer = _pricer()
    strike = pricer.support[len(pricer.support) // 2]
    step = .1 * np.min(np.diff(pricer.support))
    calls = pricer.european_at(expiry_y=pricer.expiries_y[2], strikes=np.array([strike, strike + step]))
    digital = pricer.european_at(expiry_y=pricer.expiries_y[2], strikes=np.array([strike]), digital=True)[0]
    assert digital == pytest.approx((calls[0] - calls[1]) / step, abs=1e-12)
    puts = pricer.european_at(expiry_y=pricer.expiries_y[2], strikes=np.array([strike]), digital=True, is_call=False)
    assert digital + puts[0] == pytest.approx(1.)
    np.testing.assert_allclose(pricer.density_at(2.5 / 252),
                               .5 * (pricer.marginal_densities[1] + pricer.marginal_densities[2]))
