"""Distribution repricing does not create a return horizon or Q measure."""

import numpy as np
import pytest
from cdxcore.bs import bs
from scipy.special import ndtri

from quantdata.dysanos.v11.statistical_smiles import (
    empirical_smile,
    implied_total_standard_deviation,
)


def test_black_total_std_round_trip() -> None:
    strikes = np.array([.7, .9, 1., 1.2, 1.7])
    target = np.array([.3, .35, .4, .45, .5])
    is_call = strikes >= 1
    prices = bs.price(strikes, vol=target, sqrtT=1., is_call=is_call)
    np.testing.assert_allclose(implied_total_standard_deviation(strikes, prices, is_call=is_call), target, atol=1e-6)


def test_centering_scale_invariance_and_put_call_parity() -> None:
    levels = np.array([10., 12., 17., 25., 35., np.nan])
    strikes = np.array([.6, .8, 1., 1.3, 2.])
    a = empirical_smile(levels, strikes=strikes, minimum_payoffs=1)
    b = empirical_smile(100 * levels, strikes=strikes, minimum_payoffs=1)
    assert a["sample_count"] == 5
    assert a["normalized_mean"] == pytest.approx(1)
    assert a["parity_max_error"] < 1e-14
    for name in ("call_prices", "put_prices", "total_implied_std", "supported"):
        np.testing.assert_allclose(a[name], b[name])
    assert not a["supported"][-1]  # Beyond the finite empirical support, not a measured zero-vol wing.


def test_lognormal_levels_produce_approximately_flat_total_vol() -> None:
    normal = ndtri((np.arange(100_000) + .5) / 100_000)
    values = 20 * np.exp(.35 * normal - .5 * .35**2)
    result = empirical_smile(values, strikes=np.array([.7, .85, 1., 1.2, 1.5]))
    np.testing.assert_allclose(result["total_implied_std"], .35, atol=5e-4)


def test_constant_levels_have_no_option_time_value() -> None:
    result = empirical_smile(np.full(20, 30.), strikes=np.array([.8, 1., 1.2]))
    np.testing.assert_array_equal(result["otm_prices"], 0)
    np.testing.assert_array_equal(result["total_implied_std"], 0)


@pytest.mark.parametrize("levels", [[1.], [1., np.inf], [1., 0.], [1., -1.], [np.nan, np.nan]])
def test_invalid_levels_rejected(levels: list[float]) -> None:
    with pytest.raises(ValueError):
        empirical_smile(np.array(levels), strikes=np.array([1.]))


@pytest.mark.parametrize("strikes", [[0., 1.], [1., 1.], [1.2, .8], [np.nan]])
def test_invalid_strikes_rejected(strikes: list[float]) -> None:
    with pytest.raises(ValueError):
        empirical_smile(np.array([10., 20.]), strikes=np.array(strikes))
