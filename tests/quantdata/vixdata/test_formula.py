from __future__ import annotations

import numpy as np
import pytest

from quantdata.vixdata.formula import (
    constant_maturity_vix,
    discrete_total_variance,
    select_vix_strip,
)


def test_cboe_selection_weights_correction_and_zero_bid_cutoff() -> None:
    strikes = np.arange(60, 141, 10, dtype=float)
    # An arbitrage-consistent parity pair places the forward at 102, K0 at 100.
    puts = np.asarray([0.01, 0.1, 0.2, 1, 4, 10, 19, 28, 38], dtype=float)
    calls = puts + 102 - strikes
    calls = np.maximum(calls, 0.02)
    cb, ca, pb, pa = calls.copy(), calls.copy(), puts.copy(), puts.copy()
    pb[:3] = [0.01, 0, 0]  # positive 60 is excluded beyond consecutive zeros 80, 70
    cb[6:] = [0, 0, 0.02]  # positive 140 is excluded beyond 120, 130
    strip = select_vix_strip(strikes=strikes, call_bids=cb, call_asks=ca,
                            put_bids=pb, put_asks=pa, discount_factor=1)
    assert strip.forward == pytest.approx(102)
    assert strip.k0 == 100
    np.testing.assert_array_equal(strip.strikes, [90, 100, 110])
    np.testing.assert_array_equal(strip.delta_strikes, [10, 10, 10])
    np.testing.assert_allclose(strip.prices, [1, 5, 2])
    assert strip.put_zero_bid_stop == 70
    assert strip.call_zero_bid_stop == 130
    expected = 2 * (10 / 90**2 + 50 / 100**2 + 20 / 110**2) - (102 / 100 - 1)**2
    assert discrete_total_variance(strip, discount_factor=1) == pytest.approx(expected)


def test_cboe_formula_is_invariant_to_currency_and_discount_scaling() -> None:
    strikes = np.array([80, 95, 100, 110, 140], dtype=float)
    puts = np.array([0.2, 2, 5, 12, 40.1])
    calls = puts + 100 - strikes
    base = select_vix_strip(strikes=strikes, call_bids=calls, call_asks=calls,
                           put_bids=puts, put_asks=puts, discount_factor=1)
    scaled = select_vix_strip(strikes=strikes * 60, call_bids=calls * 60 * 0.99,
                             call_asks=calls * 60 * 0.99, put_bids=puts * 60 * 0.99,
                             put_asks=puts * 60 * 0.99, discount_factor=0.99)
    np.testing.assert_allclose(base.delta_strikes, [15, 10, 7.5, 20, 30])
    assert discrete_total_variance(base, discount_factor=1) == pytest.approx(
        discrete_total_variance(scaled, discount_factor=0.99))


def test_calendar_interpolation_is_in_total_variance_not_volatility() -> None:
    result = constant_maturity_vix(near_days=25, next_days=32,
        near_total_variance=0.2**2 * 25 / 365, next_total_variance=0.3**2 * 32 / 365)
    expected = 100 * np.sqrt(((2 / 7) * 25 * 0.2**2 + (5 / 7) * 32 * 0.3**2) / 30)
    assert result == pytest.approx(expected)
    with pytest.raises(ValueError, match="bracket"):
        constant_maturity_vix(near_days=24, next_days=28, near_total_variance=0.01, next_total_variance=0.02)
