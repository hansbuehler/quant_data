"""Fixed cash strikes, exact settlement and centered return-variance attribution."""

from datetime import date

import numpy as np
import pytest

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.straddle_explained import (
    ReturnBasis,
    fixed_straddle_values,
    variance_capture,
)


def _state() -> DailyState:
    config = DlvConfig(business_dtes=(1., 3.), num_inner_strikes=10)
    native = np.tile(np.r_[np.full(10, 2. * np.log(.2)), np.log(.2)], 2)
    return DailyState(day=date(2024, 4, 1), native=native,
                      reference_support=np.r_[.5, np.linspace(.8, 1.2, 10), 1.5], config=config)


def test_fixed_cash_strikes_settle_exactly_and_have_no_remaining_surface_risk() -> None:
    state = _state()
    strikes = np.array([[.98, 1., 1.02]])
    spots = np.array([.97, 1.04])
    natives = np.stack([state.native, state.native + .1])
    prices, _ = fixed_straddle_values(state, natives, strikes=strikes,
        business_dtes=np.zeros(1), spot_ratios=spots)
    np.testing.assert_allclose(prices[:, 0], np.abs(spots[:, None] - strikes), atol=1e-12)


def test_fixed_straddle_pricing_agrees_with_the_scalar_surface_and_spot_homogeneity() -> None:
    state = _state()
    strikes = np.array([[.99, 1., 1.01], [.95, 1., 1.05]])
    spots = np.array([1.03])
    days = np.array([1., 2.])
    prices, _ = fixed_straddle_values(state, state.native[None], strikes=strikes,
        business_dtes=days, spot_ratios=spots)
    pricer = state.decode().pricer
    for row, tenor in enumerate(days):
        query = strikes[row] / spots[0]
        expected = spots[0] * (pricer.european_at(expiry_y=tenor / 252., strikes=query)
                              + pricer.european_at(expiry_y=tenor / 252., strikes=query, is_call=False))
        np.testing.assert_allclose(prices[0, row], expected, atol=1e-12)


def test_variance_capture_distinguishes_centered_variance_from_bias() -> None:
    actual = np.array([1., 2., 4., 7.]).reshape(-1, 1, 1)
    assert variance_capture(actual, np.zeros_like(actual))["variance_explained"] == pytest.approx(0.)
    assert variance_capture(actual, actual)["variance_explained"] == pytest.approx(1.)
    shifted = variance_capture(actual, actual + 3.)
    assert shifted["variance_explained"] == pytest.approx(1.)
    assert shifted["residual_mean_rms"] == pytest.approx(3.)
    assert shifted["zero_benchmark_energy_explained"] < 1.
    assert variance_capture(actual, -actual)["variance_explained"] < 0.


def test_nested_reconstruction_retains_full_spot_loading_outside_factor_span() -> None:
    encoder = np.eye(4)[:, :2]
    model = ReturnBasis(encoder=encoder, decoder=encoder, spot_loading=np.array([1., 2., 3., 4.]), label="test")
    change, spot = np.array([.2, -.1, .3, -.2]), .02
    projected = model.projected_changes(change, spot)
    np.testing.assert_allclose(projected[0], model.spot_loading * spot)
    np.testing.assert_allclose(projected[-1, :2], change[:2])
    np.testing.assert_allclose(projected[-1, 2:], (model.spot_loading * spot)[2:])
