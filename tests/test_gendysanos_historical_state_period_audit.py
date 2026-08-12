from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.grid.gendysanos.historical_state_period_audit import (
    aggregate_historical_audits,
    execute_frozen_action,
    observed_surface_calls,
    project_l1_ball,
    solve_mean_variance_portfolio,
    solve_optimal_expected_return,
)


def _observed_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expiries = np.asarray([0.1, 0.2])
    extended_strikes = np.asarray(
        [
            [0.5, 0.8, 1.0, 1.2, 1.5],
            [0.5, 0.8, 1.0, 1.2, 1.5],
        ]
    )
    calls = np.asarray(
        [
            [0.22, 0.08, 0.02],
            [0.25, 0.12, 0.05],
        ]
    )
    return expiries, extended_strikes, calls


def test_observed_surface_calls_reproduce_grid_prices() -> None:
    expiries, extended_strikes, calls = _observed_surface()
    result = observed_surface_calls(
        grid_expiries=expiries,
        extended_strikes=extended_strikes,
        inner_calls=calls,
        query_expiries=expiries,
        query_strikes=extended_strikes[:, 1:-1],
    )
    assert result == pytest.approx(calls)


def test_observed_surface_calls_interpolate_from_intrinsic_and_in_maturity() -> None:
    expiries, extended_strikes, calls = _observed_surface()
    query_strikes = np.asarray([[1.0], [1.0]])
    result = observed_surface_calls(
        grid_expiries=expiries,
        extended_strikes=extended_strikes,
        inner_calls=calls,
        query_expiries=np.asarray([0.05, 0.15]),
        query_strikes=query_strikes,
    )
    assert result[:, 0] == pytest.approx([0.04, 0.10])


def test_observed_surface_calls_use_call_boundaries_outside_inner_grid() -> None:
    expiries, extended_strikes, calls = _observed_surface()
    result = observed_surface_calls(
        grid_expiries=expiries,
        extended_strikes=extended_strikes,
        inner_calls=calls,
        query_expiries=np.asarray([0.1, 0.2]),
        query_strikes=np.asarray([[0.4, 1.6], [0.4, 1.6]]),
    )
    assert result == pytest.approx(np.asarray([[0.6, 0.0], [0.6, 0.0]]))


def test_execute_frozen_action_marks_fixed_cash_contracts() -> None:
    result = execute_frozen_action(
        np.asarray([0.2, -0.8]),
        spot_ratio=1.1,
        entry_calls=np.asarray([[0.2]]),
        exit_calls=np.asarray([[0.25]]),
        cost_rate=1.0e-3,
    )
    expected_gross = 0.2 * 0.1 - 0.8 * (1.1 * 0.25 - 0.2)
    expected_cost = 1.0e-3 * (0.2 + 0.8 * 0.2)
    assert result["gross_pnl"] == pytest.approx(expected_gross)
    assert result["transaction_cost"] == pytest.approx(expected_cost)
    assert result["net_pnl"] == pytest.approx(expected_gross - expected_cost)
    assert result["numerical_band"] > 0.0


def test_optimal_expected_return_uses_best_signed_coordinate() -> None:
    returns = np.asarray(
        [
            [0.01, -0.05, 0.02],
            [0.03, -0.04, 0.04],
            [0.02, -0.03, 0.03],
        ]
    )
    action, result = solve_optimal_expected_return(
        returns,
        np.asarray([0.0, 0.005, 0.0]),
    )
    assert action == pytest.approx([0.0, -1.0, 0.0])
    assert result["selected_instrument"] == 1
    assert result["selected_side"] == "short"
    assert result["selected_expected_net_pnl"] == pytest.approx(0.035)


def test_optimal_expected_return_allows_zero_action() -> None:
    action, result = solve_optimal_expected_return(
        np.asarray([[0.001, -0.002], [0.001, -0.002]]),
        np.asarray([0.003, 0.003]),
    )
    assert action == pytest.approx([0.0, 0.0])
    assert result["selected_side"] == "none"


def test_l1_projection_preserves_inside_points_and_projects_outside_points() -> None:
    inside = np.asarray([0.2, -0.3, 0.1])
    assert project_l1_ball(inside) == pytest.approx(inside)
    outside = project_l1_ball(np.asarray([2.0, -1.0, 0.5]))
    assert np.abs(outside).sum() == pytest.approx(1.0)
    assert outside == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("risk_aversion", "expected_action"),
    [(1.0, 0.1), (10.0, 0.01), (100.0, 0.001)],
)
def test_mean_variance_portfolio_matches_one_instrument_solution(
    risk_aversion: float,
    expected_action: float,
) -> None:
    action, result = solve_mean_variance_portfolio(
        np.asarray([[-0.9], [1.1]]),
        risk_aversion=risk_aversion,
    )
    assert action == pytest.approx([expected_action], abs=1.0e-10)
    assert result["converged"]
    assert result["projected_gradient_residual"] <= 1.0e-10


def _aggregate_row(*, validation_mean: float, l1: float, pnl: float) -> dict:
    return {
        "portfolio": {"l1_norm": l1},
        "next_market_execution": {
            "net_pnl": pnl,
            "numerical_band": 1.0e-6,
        },
        "model_validation": {
            "mean_pnl": validation_mean,
            "loss_count_beyond_float32_band": 1,
        },
    }


def test_aggregate_historical_audits_keeps_model_and_realized_tests_separate() -> None:
    result = aggregate_historical_audits(
        [
            _aggregate_row(validation_mean=0.01, l1=1.0, pnl=-0.02),
            _aggregate_row(validation_mean=-0.01, l1=0.0, pnl=0.00),
            _aggregate_row(validation_mean=0.02, l1=1.0, pnl=0.03),
        ]
    )
    assert result["historical_states_executed"] == 3
    assert result["active_portfolios"] == 2
    assert result["positive_model_validation_mean_count"] == 2
    assert result["realized_next_market_losses_beyond_band"] == 1
