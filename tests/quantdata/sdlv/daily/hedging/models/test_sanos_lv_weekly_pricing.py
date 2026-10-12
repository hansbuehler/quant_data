from __future__ import annotations

import math

import numpy as np
import pytest

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.data.pricing import price_dlv_surface
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_weekly_market_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    _price_barriers,
    _price_realized_variance,
    _shift_uniform_mass,
    build_common_weekly_claims,
    price_weekly_sanos_lv_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_smoke import (
    run_weekly_pricing_smoke,
)


def test_weekly_sanos_lv_prices_are_finite_and_respect_vanilla_bounds() -> None:
    day = synthetic_weekly_market_day()
    result = price_weekly_sanos_lv_targets(day)
    prices = result.by_label()
    discount = day.discount_at_business_dte(5)
    assert len(prices) == 13
    assert prices["digital-call-5d-atm"] + prices["digital-put-5d-atm"] == (
        pytest.approx(discount)
    )
    assert prices["realized-variance"] > 0.0
    vanilla_call, vanilla_put = discount * price_dlv_surface(
        query_expiries_y=np.asarray([5.0 / 252.0, 5.0 / 252.0]),
        query_strikes=np.asarray([1.0, 1.0]),
        is_call=np.asarray([True, False]),
        support_strikes=day.dlv_support_strikes,
        raw_log_excess_dlvsq=day.dlv_raw_log_excess_dlvsq,
        model_expiries_y=day.dlv_model_expiries_y,
        minimum_dlvsq=day.minimum_dlvsq,
        maximum_dlvsq=day.maximum_dlvsq,
        backbone_annual_volatility=day.backbone_annual_volatility,
    )
    assert 0.0 <= prices["down-and-out-put"] <= vanilla_put
    assert 0.0 <= prices["up-and-out-call"] <= vanilla_call
    assert result.transition_mass_error < 1.0e-12
    assert result.transition_martingale_error < 1.0e-12


def test_weekly_variance_price_uses_exact_background_second_moment() -> None:
    intervals = np.full(5, 1.0 / 252.0)
    annual_volatility = 0.2
    price = _price_realized_variance(
        support=np.asarray([1.0]),
        initial_density=np.asarray([1.0]),
        transitions=np.ones((5, 1, 1)),
        intervals=intervals,
        background_annual_volatility=annual_volatility,
    )
    expected = math.exp(annual_volatility**2 / 252.0) - 1.0
    assert price == pytest.approx(expected, rel=1.0e-12)


def test_weekly_sanos_lv_pricing_is_deterministic() -> None:
    day = synthetic_weekly_market_day()
    first = price_weekly_sanos_lv_targets(day)
    second = price_weekly_sanos_lv_targets(day)
    np.testing.assert_array_equal(first.prices, second.prices)


def test_weekly_background_grid_is_conservative_and_convergent() -> None:
    mass = np.zeros((2, 3, 9))
    mass[:, :, 4] = 1.0
    for shift in (-1.25, 0.25, 1.75):
        shifted = _shift_uniform_mass(mass, shift_cells=shift)
        assert np.sum(shifted) == pytest.approx(np.sum(mass))

    support = np.asarray([0.9, 1.0, 1.1])
    density = np.asarray([0.2, 0.6, 0.2])
    transition = np.asarray(
        [
            [0.8, 0.1, 0.0],
            [0.2, 0.8, 0.2],
            [0.0, 0.1, 0.8],
        ]
    )
    transitions = np.repeat(transition[None, :, :], 5, axis=0)
    intervals = np.full(5, 1.0 / 252.0)
    medium = _price_barriers(
        support=support,
        initial_density=density,
        transitions=transitions,
        intervals=intervals,
        background_annual_volatility=0.05,
        grid_size=513,
        stddev_width=10.0,
    )
    fine = _price_barriers(
        support=support,
        initial_density=density,
        transitions=transitions,
        intervals=intervals,
        background_annual_volatility=0.05,
        grid_size=1025,
        stddev_width=10.0,
    )
    np.testing.assert_allclose(medium, fine, rtol=0.0, atol=2.0e-5)


def test_common_weekly_claims_have_explicit_market_prices() -> None:
    day = synthetic_weekly_market_day()
    claims = build_common_weekly_claims(day)
    assert len(claims) == 13
    assert all(
        claim.market_price is not None and claim.market_price >= 0.0
        for claim in claims
    )
    direct_atm_call = next(
        claim
        for claim in claims
        if claim.target.label == "digital-call-5d-atm"
    )
    expected = build_weekly_claim(
        day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    assert direct_atm_call.market_price == pytest.approx(
        expected.market_price
    )


def test_weekly_pricing_smoke_exposes_common_and_model_prices() -> None:
    result = run_weekly_pricing_smoke(synthetic_weekly_market_day())
    assert result["schema"] == "hedging-weekly-pricing-smoke-v1"
    assert result["background_grid_size"] == 2049
    assert len(result["common_prices"]) == 13
    assert len(result["dlv_model_prices"]) == 13
    assert result["transition_mass_error"] < 1.0e-12
    assert result["transition_martingale_error"] < 1.0e-12
