"""Check the standalone V2 conditional law and its spot-factor covariance."""

from types import SimpleNamespace

import numpy as np

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.standalone_v2_forecast import (
    common_strikes,
    forward_log_returns,
    sample_next,
)


def test_sample_next_preserves_predictable_mean_and_joint_spot_innovation():
    model = SimpleNamespace(alpha=np.array([.1, -.2]), beta=np.array([.5, -.1]),
        r_mean=.03, directions=np.array([[1.], [.25]]), A=np.array([[.7]]),
        scores=np.array([[.2]]), cov=np.array([[.04, .01], [.01, .09]]))
    states, returns = sample_next(np.array([1., 2.]), model, paths=100_000, seed=19)
    expected_mean = np.array([1., 2.]) + model.alpha + model.beta * model.r_mean + model.directions[:, 0] * .14
    np.testing.assert_allclose(states.mean(axis=0), expected_mean, atol=.003)
    np.testing.assert_allclose(returns.mean(), .03, atol=.003)
    expected_covariance = model.beta * .09 + model.directions[:, 0] * .01
    np.testing.assert_allclose(np.cov(states.T, returns, ddof=0)[:2, 2], expected_covariance, atol=.002)


def test_common_strikes_match_reference_nms_offset():
    atm = np.array([[.10, .15]])
    times = np.array([1., 5.]) / 252.
    nms = np.array([-1.5, 0., 1.5])
    strikes = common_strikes(atm, times, nms)
    expected = NMS.from_nms(nms[None, None, :], times[None, :, None], vol=atm[:, :, None])
    np.testing.assert_allclose(strikes, expected)
    naive = np.exp(nms[None, None, :] * atm[:, :, None] * np.sqrt(times)[None, :, None])
    assert not np.allclose(strikes, naive)


def test_forward_conversion_recovers_forward_normalized_return():
    spot, next_spot, forward, dividend = 100., 102., 100.04, .00007
    total_return = np.array([np.log(next_spot / spot) + dividend])
    result = forward_log_returns(total_return, dividend_per_session=dividend, spot=spot, forward=forward)
    np.testing.assert_allclose(result, [np.log(next_spot / forward)], rtol=0., atol=1e-15)
