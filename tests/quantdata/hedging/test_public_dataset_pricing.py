from __future__ import annotations

import numpy as np

from quantdata.hedging.data.pricing import (
    decode_dlv_density_tower,
    implied_volatilities,
    price_dlv_surface,
    price_sanos_surface,
)


def test_sanos_prices_are_reconstructed_from_state_and_geometry() -> None:
    expiries = np.asarray([1.0 / 252.0, 2.0 / 252.0])
    support = (
        np.asarray([0.8, 1.0, 1.2]),
        np.asarray([0.7, 1.0, 1.3]),
    )
    densities = (
        np.asarray([0.25, 0.5, 0.25]),
        np.asarray([0.25, 0.5, 0.25]),
    )
    prices = price_sanos_surface(
        query_expiries_y=np.asarray([expiries[0], expiries[1]]),
        query_strikes=np.asarray([1.0, 1.0]),
        is_call=np.asarray([True, False]),
        model_expiries_y=expiries,
        atm_variances=np.asarray([0.04 * expiries[0], 0.04 * expiries[1]]),
        support_strikes=support,
        densities=densities,
        smoothness=0.0,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
    )
    np.testing.assert_allclose(prices, [0.05, 0.075], atol=1.0e-14)


def test_dlv_density_and_price_preserve_forward() -> None:
    support = np.asarray([0.6, 0.85, 1.0, 1.2, 1.6])
    expiries = np.asarray([1.0 / 252.0, 2.0 / 252.0])
    raw = np.log(np.asarray([[0.03, 0.04, 0.05], [0.04, 0.05, 0.06]]))
    tower = decode_dlv_density_tower(
        support_strikes=support,
        raw_log_excess_dlvsq=raw,
        model_expiries_y=expiries,
        minimum_dlvsq=0.01**2,
        maximum_dlvsq=3.0**2,
    )
    np.testing.assert_allclose(tower.sum(axis=1), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(tower @ support, 1.0, atol=1.0e-14)

    prices = price_dlv_surface(
        query_expiries_y=np.asarray([expiries[0], expiries[1]]),
        query_strikes=np.asarray([1.0, 1.0]),
        is_call=np.asarray([True, False]),
        support_strikes=support,
        raw_log_excess_dlvsq=raw,
        model_expiries_y=expiries,
        minimum_dlvsq=0.01**2,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )
    assert np.all(np.isfinite(prices) & (prices > 0.0))


def test_implied_volatility_round_trip() -> None:
    from cdxcore.bs import bs

    expiries = np.asarray([5.0 / 252.0, 20.0 / 252.0])
    strikes = np.asarray([1.0, 1.0])
    sides = np.asarray([False, True])
    expected = np.asarray([0.2, 0.3])
    prices = np.asarray(
        bs.price(
            k=strikes,
            vol=expected,
            sqrtT=np.sqrt(expiries),
            is_call=sides,
        )
    )
    actual = implied_volatilities(
        expiries_y=expiries,
        normalized_strikes=strikes,
        normalized_prices=prices,
        is_call=sides,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-7)
