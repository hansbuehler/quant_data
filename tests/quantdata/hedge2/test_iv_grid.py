"""Deterministic IV state, interpolation and fixed-contract cash pricing checks."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from cdxcore.bs import bs

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.iv_grid import IVGrid, IVGridConfig, extract_iv_grid
from quantdata.sanos.arbfree import ArbFreePricer


def _source(vol: float = 0.2) -> ArbFreePricer:
    source = ArbFreePricer(smoothness=0.1)
    source.expiries = np.array([1., 25.]) / DAYS_PER_YEAR
    source.variances = source.expiries * vol**2 / source.smoothness
    source.fitsX = [np.zeros(1), np.zeros(1)]
    source.mstrikes = [np.ones(1), np.ones(1)]
    source.mdensities = [np.ones(1), np.ones(1)]
    return source


def _grid(*, vols: np.ndarray | float = 0.2) -> IVGrid:
    config = IVGridConfig(business_dtes=(1., 3.), num_nms_nodes=3, nms_range=(-1., 1.))
    values = np.broadcast_to(vols, (2, 3)).copy()
    return IVGrid(day=date(2024, 4, 19), config=config, reference_vols=np.full(2, 0.3),
                  log_implied_vols=np.log(values))


def _curve() -> CashCurve:
    return CashCurve(spot=100., business_dtes=np.array([1., 3., 25.]),
                     forwards=np.array([100.2, 100.8, 104.]),
                     reference_forwards=np.array([101., 103., 108.]),
                     discounts=np.array([0.999, 0.996, 0.98]))


def test_extract_flat_black_source_and_preserve_frozen_parameters() -> None:
    source = _source()
    before = [values.copy() for values in source.mdensities + source.mstrikes]
    extracted = extract_iv_grid(source=source, day=date(2024, 4, 19),
                                config=IVGridConfig(nms_range=(-0.5, 0.5)))
    assert extracted.grid.state.shape == (9 * 29,)
    np.testing.assert_allclose(np.exp(extracted.grid.log_implied_vols), 0.2, atol=2e-6)
    assert not extracted.inversion_failed.any()
    assert not extracted.inversion_floor.any()
    assert extracted.summary()["max_abs_roundtrip_error"] <= extracted.grid.config.price_tolerance
    for saved, current in zip(before, source.mdensities + source.mstrikes, strict=True):
        np.testing.assert_array_equal(saved, current)


def test_inversion_floor_is_explicit_and_does_not_claim_identified_wings() -> None:
    extracted = extract_iv_grid(source=_source(vol=0.001), day=date(2024, 4, 19))
    assert extracted.inversion_floor.all()
    assert extracted.intrinsic_within_tolerance.any()
    assert extracted.inversion_failed.any()
    np.testing.assert_allclose(extracted.grid.log_implied_vols, np.log(extracted.grid.config.vol_min))
    assert extracted.summary()["inversion_floor_count"] == 9 * 29


def test_inversion_upper_bound_is_reported() -> None:
    config = IVGridConfig(vol_max=0.15, nms_range=(-0.5, 0.5))
    extracted = extract_iv_grid(source=_source(), day=date(2024, 4, 19), config=config)
    assert extracted.inversion_cap.all()
    assert extracted.inversion_failed.all()
    assert extracted.summary()["max_abs_roundtrip_error"] > config.price_tolerance


def test_source_moment_roundoff_is_reported_without_mutating_the_source() -> None:
    source = _source()
    source.mdensities[0] *= 1. + 5e-8
    before = source.mdensities[0].copy()
    extracted = extract_iv_grid(source=source, day=date(2024, 4, 19),
                                config=IVGridConfig(nms_range=(-0.5, 0.5)))
    np.testing.assert_array_equal(source.mdensities[0], before)
    assert extracted.summary()["max_abs_source_parity_error"] > 1e-10
    assert extracted.summary()["max_abs_otm_roundtrip_error"] <= extracted.grid.config.price_tolerance
    np.testing.assert_allclose(extracted.roundtrip_errors,
                               extracted.otm_roundtrip_errors + extracted.source_parity_errors, atol=1e-15)


def test_nms_interpolation_is_logarithmic_and_expiry_interpolation_uses_total_variance() -> None:
    grid = _grid(vols=np.array([[0.1, 0.2, 0.4], [0.2, 0.4, 0.8]]))
    days = np.array([1., 2., 3.])
    strikes = NMS.from_nms(np.full(3, 0.5), days / DAYS_PER_YEAR, vol=0.3, eps_vol=0.3)
    result = grid.interpolate(business_dtes=days, strikes_over_forward=strikes)
    first_variance = 0.2 * 0.4
    last_variance = 0.4 * 0.8
    expected = np.sqrt([first_variance, (first_variance + 3. * last_variance) / 4., last_variance])
    np.testing.assert_allclose(result.implied_vols, expected, atol=1e-13)
    np.testing.assert_allclose(result.nms, 0.5, atol=1e-13)
    assert not result.nms_extrapolated.any()


def test_reference_geometry_interpolates_total_variance_and_is_unchanged_by_state_bumps() -> None:
    grid = replace(_grid(vols=np.array([[0.1, 0.2, 0.4], [0.1, 0.2, 0.4]])),
                   reference_vols=np.array([0.2, 0.4]))
    day = 2.
    reference = np.sqrt((0.2**2 + 3. * 0.4**2) / 4.)
    strike = NMS.from_nms(0.5, day / DAYS_PER_YEAR, vol=reference, eps_vol=reference)
    original = grid.interpolate(business_dtes=day, strikes_over_forward=strike)
    bumped = grid.with_state(grid.state + np.log(2.)).interpolate(
        business_dtes=day, strikes_over_forward=strike)
    np.testing.assert_allclose(original.nms, 0.5)
    np.testing.assert_array_equal(original.nms, bumped.nms)
    np.testing.assert_allclose(bumped.implied_vols, 2. * original.implied_vols)


def test_short_tenor_anchor_and_constant_wings_are_visible() -> None:
    grid = _grid(vols=np.array([[0.1, 0.2, 0.4], [0.2, 0.4, 0.8]]))
    days = np.array([0.5, 1., 3.])
    strikes = NMS.from_nms(np.array([0., -2., 2.]), days / DAYS_PER_YEAR, vol=0.3, eps_vol=0.3)
    result = grid.interpolate(business_dtes=days, strikes_over_forward=strikes)
    np.testing.assert_allclose(result.implied_vols, [0.2, 0.1, 0.8])
    np.testing.assert_array_equal(result.below_first_expiry, [True, False, False])
    np.testing.assert_array_equal(result.nms_extrapolated, [False, True, True])
    with pytest.raises(ValueError, match="horizon"):
        grid.interpolate(business_dtes=3.01, strikes_over_forward=1.)


def test_black_prices_preserve_cash_strikes_carry_parity_and_financing_separation() -> None:
    grid = _grid()
    curve = _curve()
    days = np.array([2., 2., 2., 2.])
    strikes = np.array([99., 99., 102., 102.])
    calls = np.array([True, False, True, False])
    strike_copy = strikes.copy()
    priced = grid.price_europeans(curve=curve, business_dtes=days, cash_strikes=strikes, is_call=calls)
    forwards, _, discounts = curve.at(days)
    expected = discounts * forwards * bs.price(k=strikes / forwards, vol=0.2,
        sqrtT=np.sqrt(days / DAYS_PER_YEAR), is_call=calls)
    np.testing.assert_allclose(priced.prices, expected, atol=1e-13)
    np.testing.assert_allclose(priced.prices[::2] - priced.prices[1::2],
                               discounts[::2] * (forwards[::2] - strikes[::2]), atol=1e-12)
    np.testing.assert_array_equal(strikes, strike_copy)
    alternate = replace(curve, reference_forwards=np.full(3, 200.))
    other = grid.price_europeans(curve=alternate, business_dtes=days, cash_strikes=strikes, is_call=calls)
    np.testing.assert_array_equal(other.prices, priced.prices)
    moved = grid.price_europeans(curve=curve.at_spot(103.), business_dtes=days,
                                 cash_strikes=strikes, is_call=calls)
    assert moved.prices[0] > priced.prices[0]
    assert moved.prices[1] < priced.prices[1]


def test_settlement_and_live_zero_strike_calls() -> None:
    grid = _grid()
    curve = _curve()
    result = grid.price_europeans(curve=curve, business_dtes=np.array([0., 0., 2., 0.]),
        cash_strikes=np.array([99., 101., 0., 100.]), is_call=np.array([True, False, True, True]))
    forward, _, discount = curve.at(2.)
    np.testing.assert_allclose(result.prices, [1., 1., discount * forward, 0.])
    assert np.isnan(result.implied_vols).all()
    assert not result.nms_extrapolated.any()


@pytest.mark.parametrize("changes", [
    {"business_dtes": ()}, {"business_dtes": (2., 1.)}, {"num_nms_nodes": 1},
    {"nms_range": (0., 1.)}, {"vol_min": 0.}, {"vol_max": np.inf},
    {"price_tolerance": 1e-9}, {"low_vega_threshold": np.nan},
])
def test_invalid_config(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IVGridConfig(**changes)


@pytest.mark.parametrize("state", [np.array([0.]), np.full(6, np.nan), np.full(6, 1000.), np.full(6, -1000.)])
def test_invalid_log_state_is_rejected_without_silent_clipping(state: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _grid().with_state(state)


@pytest.mark.parametrize("days,strike,side", [(-1., 100., True), (4., 100., True),
    (1., -1., True), (1., 0., False), (1., np.nan, True), (1., 100., 1)])
def test_invalid_contracts_are_rejected(days: float, strike: float, side: bool) -> None:
    with pytest.raises(ValueError):
        _grid().price_europeans(curve=_curve(), business_dtes=days, cash_strikes=strike, is_call=side)
