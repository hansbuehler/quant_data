"""Saved-source moment normalization must preserve prices to source precision."""

import numpy as np
import pytest
from cdxcore.bs import bs

from quantdata.hedge2.source import (
    implied_vols_from_time_values,
    normalize_surface_moments,
    surface_digitals,
    surface_implied_vols,
    surface_time_values,
)
from quantdata.sanos.arbfree import ArbFreePricer


def test_positive_tail_values_do_not_turn_into_floor_volatility() -> None:
    strikes = np.array([.98, 1., 1.02, 1.04])
    times = np.full(4, 1. / 252.)
    expected = np.full(4, .12)
    calls = strikes >= 1.
    prices = bs.price(k=strikes, vol=expected, sqrtT=np.sqrt(times), is_call=calls, eps=1e-16)
    assert 0. < prices[-1] < 1e-8
    default = bs.implied(k=strikes, prices=prices, sqrtT=np.sqrt(times), is_call=calls)
    assert default[-1] == .01
    actual = implied_vols_from_time_values(expiries_y=times, strikes=strikes, time_values=prices)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    reconstructed = bs.price(k=strikes, vol=actual, sqrtT=np.sqrt(times), is_call=calls, eps=1e-16)
    np.testing.assert_allclose(reconstructed, prices, rtol=1e-8, atol=0.)


def test_tail_inversion_keeps_zero_boundary_explicit_and_rejects_invalid_prices() -> None:
    result = implied_vols_from_time_values(expiries_y=np.array([1. / 252.]),
        strikes=np.array([1.02]), time_values=np.array([0.]))
    np.testing.assert_array_equal(result, [0.])
    for value in (-1e-9, 1., np.nan):
        with pytest.raises(ValueError):
            implied_vols_from_time_values(expiries_y=np.array([1. / 252.]),
                strikes=np.array([1.02]), time_values=np.array([value]))


def test_otm_put_tail_inversion_avoids_parity_cancellation() -> None:
    strikes = np.array([.9587939825833324, .9598935169670288, .9609930513507253])
    prices = np.array([2.537744274369052e-11, 3.993761483381949e-10, 4.138150626655784e-9])
    times = np.full(len(strikes), 1. / 252.)
    actual = implied_vols_from_time_values(expiries_y=times, strikes=strikes, time_values=prices)
    reconstructed = strikes * bs.price(k=1. / strikes, vol=actual, sqrtT=np.sqrt(times), is_call=True, eps=1e-16)
    np.testing.assert_allclose(reconstructed, prices, rtol=1e-8, atol=0.)
    assert np.all(actual > .01)


def test_otm_put_call_symmetry_preserves_implied_volatility() -> None:
    call_strikes = np.array([1.001, 1.02, 1.04, 1.08])
    put_strikes = 1. / call_strikes
    times = np.full(len(call_strikes), 1. / 252.)
    expected = np.full(len(call_strikes), .12)
    call_prices = bs.price(k=call_strikes, vol=expected, sqrtT=np.sqrt(times), is_call=True, eps=1e-16)
    call_vols = implied_vols_from_time_values(expiries_y=times, strikes=call_strikes, time_values=call_prices)
    put_vols = implied_vols_from_time_values(expiries_y=times, strikes=put_strikes,
        time_values=put_strikes * call_prices)
    np.testing.assert_allclose(call_vols, expected, rtol=1e-9)
    np.testing.assert_allclose(put_vols, call_vols, rtol=1e-9)


def test_saved_surface_iv_uses_same_prices_without_absolute_tail_tolerance() -> None:
    pricer = ArbFreePricer(smoothness=.1)
    pricer.expiries = np.array([1. / 252.])
    pricer.variances = np.array([.04 / 252.])
    pricer.fitsX = [np.zeros(1)]
    pricer.mstrikes = [np.array([1.])]
    pricer.mdensities = [np.array([1.])]
    strikes = np.array([.99, 1., 1.02])
    prices, coarse = pricer.price(expiries=1. / 252., strikes=strikes, and_vols=True, mode="surface")
    actual = surface_implied_vols(pricer, expiries_y=np.full(3, 1. / 252.), strikes=strikes)
    np.testing.assert_allclose(actual, .2 * np.sqrt(.1), rtol=1e-9)
    assert coarse[-1] == .01
    np.testing.assert_array_equal(pricer.price(expiries=1. / 252., strikes=strikes, mode="surface"), prices)


@pytest.mark.parametrize("reconstructed", [0., np.nan])
def test_positive_subnormal_iv_reconstruction_cannot_accept_zero_or_nan(monkeypatch, reconstructed) -> None:
    monkeypatch.setattr("quantdata.hedge2.source.brentq", lambda *args, **kwargs: .1)
    monkeypatch.setattr("quantdata.hedge2.source.bs.price", lambda **kwargs:
        np.array([1. if kwargs["vol"] >= 1. else reconstructed]))
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        implied_vols_from_time_values(expiries_y=np.array([1. / 252.]),
            strikes=np.array([1.02]), time_values=np.array([np.nextafter(0., 1.)]))


def test_source_moment_roundoff_does_not_create_tail_put_value() -> None:
    pricer = ArbFreePricer(smoothness=0.1)
    pricer.expiries = np.array([1. / 252.])
    pricer.variances = np.array([0.04 / 252.])
    pricer.fitsX = [np.zeros(1)]
    pricer.mstrikes = [np.array([0.8, 1., 1.2]) * (1. + 2e-9)]
    pricer.mdensities = [np.array([0.25, 0.5, 0.25]) * (1. - 1e-9)]
    query = np.array([0.1, 0.95, 1., 1.05, 2.])
    original = pricer.price(expiries=1. / 252., strikes=query)
    errors = normalize_surface_moments(pricer)
    normalized = pricer.price(expiries=1. / 252., strikes=query)
    assert errors["source_mass_error"] > 1e-10
    assert abs(pricer.mdensities[0].sum() - 1.) < 1e-15
    assert abs(pricer.mdensities[0] @ pricer.mstrikes[0] - 1.) < 1e-15
    assert abs(normalized[0] - 0.9) < 1e-14
    assert np.max(np.abs(original - normalized)) < 5e-9
    time_values = surface_time_values(pricer, expiries_y=np.array([1. / 252.]), strikes=query)[0]
    np.testing.assert_allclose(time_values + np.maximum(1. - query, 0.), normalized, atol=1e-14)
    strikes = np.array([.95, 1., 1.05])
    digital = surface_digitals(pricer, expiry_y=1. / 252., strikes=strikes)
    step = 1e-6
    derivative = (pricer.price(expiries=1. / 252., strikes=strikes - step)
                  - pricer.price(expiries=1. / 252., strikes=strikes + step)) / (2. * step)
    np.testing.assert_allclose(digital, derivative, atol=1e-7)
    np.testing.assert_allclose(digital + surface_digitals(pricer, expiry_y=1. / 252., strikes=strikes,
                                                        is_call=False), 1.)
