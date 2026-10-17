from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from quantdata.hedging.claims import DigitalClaim
from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.mean_reduction import MeanReduction
from quantdata.hedging.scenario_optimization import ScenarioObjective
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalSample,
    DailyDigitalTarget,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedDailyScenarios,
    GeneratedHedgeConfig,
    _integral_business_dte,
    _smile_consistent_digital_probability,
    build_generated_daily_scenarios,
    fit_generated_objective_book,
    generated_linear_transaction_costs,
    interpolate_generated_implied_volatility,
)


def _one_day_sample() -> DailyDigitalSample:
    transition = synthetic_one_day_transition()
    quote = transition.origin.hedge_snapshot().option_quotes[0]
    target = DailyDigitalTarget(
        maturity_business_days=1,
        right=OptionRight.CALL,
        nms=0.0,
    )
    return DailyDigitalSample(
        transition=transition,
        endpoint=SimpleNamespace(trading_date=transition.next_date),
        target=target,
        claim=DigitalClaim(
            right=OptionRight.CALL,
            strike=1.0,
            maturity_business_days=1,
        ),
        target_strike_over_forward=1.0,
        target_discount_factor=1.0,
        market_price=0.5,
        terminal_value=1.0,
        claim_return=0.5,
        hedge_quotes=(quote,),
        hedge_returns=np.asarray([0.05]),
        closing_half_spreads_pv=np.asarray([0.0]),
    )


def test_generated_settlement_reprices_fixed_contracts() -> None:
    sample = _one_day_sample()
    grid = GenerativeSurfaceGrid()
    normalized_spots = np.asarray([0.9, 1.1])
    surfaces = np.full(
        (2, len(grid.expiries_y), len(grid.strikes_over_forward)),
        np.log(0.2),
    )

    scenarios = build_generated_daily_scenarios(
        sample,
        forward_log_returns=np.log(normalized_spots),
        next_log_implied_volatilities=surfaces,
        grid=grid,
    )

    quote = sample.hedge_quotes[0]
    np.testing.assert_allclose(scenarios.spot_returns, [-0.1, 0.1])
    np.testing.assert_allclose(
        scenarios.option_returns[:, 0],
        np.maximum(normalized_spots - 1.0, 0.0) - quote.mark,
    )
    np.testing.assert_allclose(scenarios.claim_terminal_values, [0.0, 1.0])
    np.testing.assert_allclose(scenarios.claim_returns, [-0.5, 0.5])


def test_generated_volatility_interpolation_preserves_flat_surface() -> None:
    grid = GenerativeSurfaceGrid()
    surfaces = np.full(
        (4, len(grid.expiries_y), len(grid.strikes_over_forward)),
        np.log(0.27),
    )
    strikes = np.asarray([0.5, 0.85, 1.15, 1.5])

    volatility = interpolate_generated_implied_volatility(
        surfaces,
        query_expiry_y=2.0 / 252.0,
        query_strikes_over_forward=strikes,
        grid=grid,
    )

    np.testing.assert_allclose(volatility, 0.27, rtol=1.0e-14, atol=1.0e-14)


def test_generated_digital_includes_smile_slope() -> None:
    grid = GenerativeSurfaceGrid()
    path_count = 3
    annual_volatility = 0.25
    log_volatility_slope = 0.35
    log_volatility = (
        math.log(annual_volatility)
        + log_volatility_slope * np.log(grid.strikes_over_forward)
    )
    surfaces = np.broadcast_to(
        log_volatility,
        (path_count, len(grid.expiries_y), len(grid.strikes_over_forward)),
    ).copy()
    expiry_y = grid.expiries_y[0]
    strikes = np.ones(path_count)

    probability, clip_count = _smile_consistent_digital_probability(
        surfaces,
        strike=strikes,
        expiry_y=expiry_y,
        right=OptionRight.CALL,
        grid=grid,
    )

    standard_deviation = annual_volatility * math.sqrt(expiry_y)
    d1 = 0.5 * standard_deviation
    d2 = -0.5 * standard_deviation
    normal_density_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    expected = ndtr(d2) - (
        normal_density_d1
        * math.sqrt(expiry_y)
        * annual_volatility
        * log_volatility_slope
    )
    np.testing.assert_allclose(probability, expected, rtol=1.0e-7, atol=1.0e-9)
    assert clip_count == 0


def test_generated_costs_use_expiry_date_not_rounded_dte() -> None:
    sample = _one_day_sample()
    quote = replace(
        sample.hedge_quotes[0],
        business_dte=1.0000000577419996,
    )
    noisy_sample = replace(sample, hedge_quotes=(quote,))

    costs = generated_linear_transaction_costs(noisy_sample)

    assert quote.contract.expiry_date == sample.transition.next_date
    assert costs[1] == pytest.approx(quote.half_spread)


@pytest.mark.parametrize("mean_reduction", tuple(MeanReduction))
@pytest.mark.parametrize("objective", tuple(ScenarioObjective))
def test_generated_objective_book_supports_full_tensor(
    mean_reduction: MeanReduction,
    objective: ScenarioObjective,
) -> None:
    sample = _one_day_sample()
    spot = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    option = np.asarray([-0.2, 0.4, -0.5, 0.3, 0.1])
    target_one = -(0.7 * spot - 0.25 * option)
    target_two = 0.4 * spot + 0.1 * option

    def scenarios(target: np.ndarray) -> GeneratedDailyScenarios:
        return GeneratedDailyScenarios(
            spot_returns=spot,
            option_returns=option[:, None],
            claim_returns=target,
            claim_terminal_values=np.ones(len(spot)),
        )

    fits = fit_generated_objective_book(
        (scenarios(target_one), scenarios(target_two)),
        sample,
        config=GeneratedHedgeConfig(
            scenario_count=len(spot),
            objective=objective,
            risk_parameter=1.0,
            mean_reduction=mean_reduction,
            cost_multiplier=0.0,
        ),
        claim_quantities=np.asarray([1.0, 2.0]),
    )

    assert len(fits) == 2
    assert all(fit.converged for fit in fits)
    assert fits[0].mean_reduction == fits[1].mean_reduction
    assert fits[0].mean_reduction.method is mean_reduction
    np.testing.assert_allclose(fits[0].action, [0.7, -0.25], atol=5.0e-7)
    np.testing.assert_allclose(fits[1].action, [-0.8, -0.2], atol=5.0e-7)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0000000577419996, 1),
        (5.000000288709998, 5),
        (21.00000121258199, 21),
    ],
)
def test_integral_business_dte_accepts_float32_roundtrip_noise(
    value: float,
    expected: int,
) -> None:
    assert _integral_business_dte(value) == expected


@pytest.mark.parametrize("value", [0.0, 1.0001, 1.5])
def test_integral_business_dte_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _integral_business_dte(value)
