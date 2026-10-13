from __future__ import annotations

import json

import numpy as np
import pytest

from quantdata.hedging.metrics import (
    HedgeSmokeResult,
    assert_smoke_fixture,
    entropy_risk,
    maximum_drawdown_from_losses,
    mean_variance_risk,
    mean_volatility_risk,
    power_exponential_risk,
)


def test_smoke_result_round_trips_to_json(tmp_path) -> None:
    result = HedgeSmokeResult(
        model="test",
        measure="Q",
        origin_date="2025-01-02",
        terminal_date="2025-01-03",
        claim="atm-1d-digital-call",
        market_price=0.5,
        model_price=0.5,
        payoff=1.0,
        unhedged_pnl=0.5,
        hedged_pnl=0.1,
        transaction_cost=0.001,
        actions={"spot": -2.0, "option": 1.0},
        pricing_difference=0.0,
        pure_hedge_error=0.1,
        objective_value=0.2,
        converged=True,
        cap_hits=0,
    )
    path = tmp_path / "result.json"
    result.write_json(path)
    assert json.loads(path.read_text(encoding="utf-8")) == result.to_dict()


def test_smoke_fixture_uses_explicit_tolerance(tmp_path) -> None:
    result = HedgeSmokeResult(
        model="test",
        measure="Q",
        origin_date="2025-01-02",
        terminal_date="2025-01-03",
        claim="claim",
        market_price=0.5,
        model_price=0.5,
        payoff=1.0,
        unhedged_pnl=0.5,
        hedged_pnl=0.1,
        transaction_cost=0.001,
        actions={"spot": -2.0},
        pricing_difference=0.0,
        pure_hedge_error=0.1,
        objective_value=0.2,
        converged=True,
        cap_hits=0,
    )
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "absolute_tolerance": 1.0e-6,
                "relative_tolerance": 0.0,
                "expected": {"market_price": 0.5000005, "converged": True},
            }
        ),
        encoding="utf-8",
    )
    assert_smoke_fixture(result, fixture)
    fixture.write_text(
        json.dumps(
            {
                "absolute_tolerance": 1.0e-8,
                "relative_tolerance": 0.0,
                "expected": {"market_price": 0.5000005},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="market_price"):
        assert_smoke_fixture(result, fixture)


@pytest.mark.parametrize(
    "risk_function",
    (
        mean_variance_risk,
        mean_volatility_risk,
        entropy_risk,
        power_exponential_risk,
    ),
)
def test_risk_scores_equal_a_constant_loss(risk_function) -> None:
    losses = np.full(17, 0.37)
    assert risk_function(losses, risk_parameter=3.0) == pytest.approx(0.37)


def test_entropy_risk_matches_the_normal_formula() -> None:
    generator = np.random.default_rng(17)
    mean = 0.03
    standard_deviation = 0.2
    risk_parameter = 1.5
    losses = generator.normal(mean, standard_deviation, size=500_000)
    expected = mean + 0.5 * risk_parameter * standard_deviation**2
    assert entropy_risk(losses, risk_parameter=risk_parameter) == pytest.approx(
        expected,
        abs=1.5e-3,
    )


@pytest.mark.parametrize("risk_parameter", (0.1, 1.0, 10.0))
def test_power_exponential_risk_is_cash_invariant(risk_parameter: float) -> None:
    losses = np.asarray([-0.3, -0.1, 0.0, 0.2, 0.8])
    shift = 0.47
    shifted = power_exponential_risk(
        losses + shift,
        risk_parameter=risk_parameter,
    )
    base = power_exponential_risk(losses, risk_parameter=risk_parameter)
    assert shifted == pytest.approx(base + shift)


def test_maximum_drawdown_from_losses_uses_chronological_cumulative_pnl() -> None:
    losses = np.asarray([-2.0, 1.0, 3.0, -1.0, 2.0])

    assert maximum_drawdown_from_losses(losses) == pytest.approx(5.0)


def test_maximum_drawdown_from_losses_includes_initial_zero_peak() -> None:
    assert maximum_drawdown_from_losses([2.0, -1.0]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "risk_function",
    (
        mean_variance_risk,
        mean_volatility_risk,
        entropy_risk,
        power_exponential_risk,
    ),
)
def test_risk_scores_reject_nonpositive_risk_parameter(risk_function) -> None:
    with pytest.raises(ValueError, match="finite-positive"):
        risk_function([0.0, 1.0], risk_parameter=0.0)
