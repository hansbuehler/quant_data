from __future__ import annotations

from datetime import date

import numpy as np
from cdxcore.bs import bs

from quantdata.gromm2.market import (
    PublicSanosDay,
    SavedSanosPricer,
    _black_call_kernel,
    _project_points_to_support,
    public_sanos_prices,
)
from quantdata.sanos.arbfree import ArbFreePricer


def test_point_projection_preserves_mass_and_mean_without_clipping() -> None:
    support = np.asarray([0.5, 0.8, 1.0, 1.3, 1.8])
    values = np.asarray([0.7, 0.95, 1.4])
    masses = np.asarray([0.2, 0.5, 0.3])
    density, clipped_mass = _project_points_to_support(values, masses, support)
    assert clipped_mass == 0.0
    assert np.isclose(np.sum(density), np.sum(masses), atol=2.0e-15)
    assert np.isclose(density @ support, masses @ values, atol=2.0e-15)


def test_black_kernel_zero_volatility_is_intrinsic() -> None:
    support = np.asarray([0.8, 1.0, 1.2])
    strikes = np.asarray([0.9, 1.1])
    actual = _black_call_kernel(
        support,
        strikes,
        annual_volatility=0.0,
        expiry_y=0.2,
    )
    expected = np.maximum(support[None, :] - strikes[:, None], 0.0)
    assert np.array_equal(actual, expected)


def test_saved_sanos_native_terminal_rule_reuses_density_without_fitting() -> None:
    pricer = ArbFreePricer(smoothness=0.25)
    pricer.expiries = np.asarray([0.1, 0.2])
    pricer.variances = 0.2**2 * pricer.expiries
    pricer.mstrikes = [np.asarray([0.8, 1.0, 1.2])] * 2
    pricer.mdensities = [np.asarray([0.25, 0.5, 0.25])] * 2
    pricer.fitsX = [np.zeros(3)] * 2
    strikes = np.asarray([0.8, 1.0, 1.2])
    expiry = 125.0 / 252.0
    expected = (bs.price(k=strikes[:, None] / pricer.mstrikes[-1][None], vol=0.1, sqrtT=np.sqrt(expiry)) * pricer.mstrikes[-1][None]) @ pricer.mdensities[-1]
    np.testing.assert_allclose(pricer.price(expiries=expiry, strikes=strikes, mode="surface"), expected, atol=1.0e-13)


def test_read_only_dataset_pricer_matches_saved_parameters() -> None:
    day = PublicSanosDay(
        trading_date=date(2024, 8, 28), model_expiries_y=np.asarray([0.1, 0.2]),
        kernel_total_variances=0.2**2 * np.asarray([0.1, 0.2]),
        support_strikes=(np.asarray([0.8, 1.0, 1.2]),) * 2,
        densities=(np.asarray([0.25, 0.5, 0.25]),) * 2,
        smoothness=0.25, one_minute_y=1.0e-6,
    )
    pricer = SavedSanosPricer(day=day)
    strikes = np.asarray([0.9, 1.0, 1.1])
    expected = public_sanos_prices(day, expiries_y=0.5, strikes=strikes)
    np.testing.assert_allclose(pricer.price(expiries=0.5, strikes=strikes), expected)
    np.testing.assert_allclose(pricer.interpolate_atm_vols(np.asarray([0.01, 0.15, 0.5])), 0.2)


def test_public_sanos_price_interpolates_node_laws_and_kernel_variance() -> None:
    first_support = np.asarray([0.8, 1.0, 1.2])
    second_support = np.asarray([0.6, 1.0, 1.4])
    first_density = np.asarray([0.25, 0.5, 0.25])
    second_density = np.asarray([0.25, 0.5, 0.25])
    day = PublicSanosDay(
        trading_date=date(2024, 5, 1),
        model_expiries_y=np.asarray([0.1, 0.2]),
        kernel_total_variances=np.asarray([0.004, 0.01]),
        support_strikes=(first_support, second_support),
        densities=(first_density, second_density),
        smoothness=0.5,
        one_minute_y=1.0e-6,
    )
    strike = 1.1
    query_expiry = 0.15
    annual_volatility = np.sqrt(0.007 / query_expiry) * np.sqrt(0.5)
    expected = 0.5 * sum(
        (
            bs.price(
                k=strike / support,
                vol=annual_volatility,
                sqrtT=np.sqrt(query_expiry),
                eps=1.0e-10,
            )
            * support
        )
        @ density
        for support, density in (
            (first_support, first_density),
            (second_support, second_density),
        )
    )
    actual = public_sanos_prices(
        day,
        expiries_y=np.asarray([query_expiry, 0.0]),
        strikes=np.asarray([strike, 0.9]),
    )
    np.testing.assert_allclose(actual, np.asarray([expected, 0.1]), atol=1.0e-13)
