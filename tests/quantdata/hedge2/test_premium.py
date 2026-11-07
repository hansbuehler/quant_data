"""Premium limits preserve contract cash flows and original-inventory turnover."""

from dataclasses import replace

import numpy as np
import pytest

from quantdata.hedge2.parametric import gaussian_power_risk
from quantdata.hedge2.premium import (
    GreekBudgetInfeasible,
    PremiumParametricOptimizer,
    PremiumUnits,
    solve_premium_greek,
)


def test_carried_inventory_projection_is_an_exact_fee_identity() -> None:
    units = PremiumUnits(normalized_prices=np.array([1.2, .3, .1]),
        normalized_costs=np.array([0., .006, .003]), position_cap=10.)
    old = np.array([15., -50., 30.])
    clipped, mandatory = units.carried_inventory(old)
    np.testing.assert_allclose(clipped, [10., -10., 3.])
    rng = np.random.default_rng(54)
    for premium in rng.uniform(-10., 10., size=(20, 3)):
        actual = units.normalized_costs @ np.abs(units.to_contracts(premium) - old)
        reduced = units.costs @ np.abs(premium - clipped) + mandatory
        assert reduced == pytest.approx(actual, abs=1e-14)


@pytest.mark.parametrize("objective,risk", [("mean_variance", 10.), ("mean_variance", np.inf),
    ("power_exponential", 10.)])
def test_parametric_adapter_replays_cash_risk_cost_and_post_trade_cap(objective, risk) -> None:
    prices = np.array([1.3, .15])
    costs = np.array([0., .006])
    exposures = np.array([[1., 0.], [.4, .7]])
    covariance = np.array([[.02, .003], [.003, .01]])
    target = np.array([.6, -.3])
    old = np.array([12., -100.])
    units = PremiumUnits(normalized_prices=prices, normalized_costs=costs)
    optimizer = PremiumParametricOptimizer(exposures=exposures, covariance=covariance, units=units)
    result = optimizer.solve(target_exposures=target, objective=objective, risk_aversion=risk,
        previous_positions=old)
    hedge = result.result.hedge
    np.testing.assert_allclose(hedge.positions, result.contract_positions * prices)
    assert np.max(np.abs(hedge.positions)) <= 10. + 1e-8
    assert result.mandatory_turnover_cost == pytest.approx(.2)
    actual_cost = float(costs @ np.abs(result.contract_positions - old))
    residual = exposures.T @ result.contract_positions - target
    variance = float(residual @ covariance @ residual)
    assert hedge.entry_cost == pytest.approx(actual_cost, abs=1e-10)
    assert hedge.training_std == pytest.approx(np.sqrt(variance), abs=1e-10)
    assert hedge.training_mean == pytest.approx(-actual_cost, abs=1e-10)
    expected = (.5 * variance if np.isinf(risk) else .5 * risk * variance + actual_cost)
    if objective == "power_exponential":
        expected = gaussian_power_risk(np.sqrt(variance), risk_aversion=risk)[0] + actual_cost
    assert hedge.training_objective == pytest.approx(expected, abs=1e-8)
    assert hedge.convex_first_order_gap_bound < 1e-5


def test_greek_budget_accounts_for_mandatory_fee_before_reoptimization() -> None:
    units = PremiumUnits(normalized_prices=np.array([1., .1]),
        normalized_costs=np.array([0., .01]), position_cap=10.)
    arguments = dict(exposures=np.array([[1., 0.], [.5, .1]]), target_exposures=np.array([2., 1.]),
        units=units, previous_positions=np.array([0., 200.]))
    full = solve_premium_greek(**arguments)
    partial = solve_premium_greek(**arguments, budget_fraction=.95)
    assert full.result.full_entry_cost == pytest.approx(1.9)
    assert full.mandatory_turnover_cost == pytest.approx(1.)
    assert partial.result.entry_cost <= .95 * full.result.full_entry_cost + 1e-7
    for result in (full, partial):
        actual = units.normalized_costs @ np.abs(result.contract_positions - arguments["previous_positions"])
        assert result.result.entry_cost == pytest.approx(actual, abs=1e-8)
        assert np.max(np.abs(result.result.positions)) <= 10. + 1e-8
    with pytest.raises(GreekBudgetInfeasible, match="mandatory turnover") as captured:
        solve_premium_greek(**arguments, budget_fraction=.5)
    assert captured.value.full_entry_cost == pytest.approx(1.9)
    assert captured.value.mandatory_turnover_cost == pytest.approx(1.)
    assert captured.value.discretionary_full_cost == pytest.approx(.9)
    assert captured.value.turnover_budget == pytest.approx(.95)
    assert captured.value.shortfall == pytest.approx(.05)


def test_greek_budget_at_exact_mandatory_cost_boundary() -> None:
    units = PremiumUnits(normalized_prices=np.array([1., .1]),
        normalized_costs=np.array([0., .01]))
    arguments = dict(exposures=np.array([[1., 0.], [.5, .1]]), target_exposures=np.array([2., 1.]),
        units=units, previous_positions=np.array([0., 200.]))
    full = solve_premium_greek(**arguments)
    fraction = full.mandatory_turnover_cost / full.result.full_entry_cost
    partial = solve_premium_greek(**arguments, budget_fraction=fraction)
    assert partial.result.entry_cost == pytest.approx(1., abs=2e-6)
    assert partial.result.positions[1] == pytest.approx(10., abs=2e-5)


@pytest.mark.parametrize("fee", [0., .01, 1e-17])
def test_greek_zero_budget_handles_free_spot_and_arbitrarily_small_fees(fee) -> None:
    units = PremiumUnits(normalized_prices=np.ones(2), normalized_costs=np.array([0., fee]))
    result = solve_premium_greek(exposures=np.eye(2), target_exposures=np.ones(2),
        units=units, budget_fraction=0.)
    assert result.mandatory_turnover_cost == 0.
    assert result.result.entry_cost <= 1e-6 * result.result.full_entry_cost
    assert result.contract_positions[0] == pytest.approx(1., abs=1e-5)
    if fee:
        assert result.contract_positions[1] == pytest.approx(0., abs=1e-5)
    else:
        np.testing.assert_allclose(result.contract_positions, 1., atol=1e-8)


def test_greek_independent_budget_check_rejects_an_overspending_solver(monkeypatch) -> None:
    from quantdata.hedge2.greek_hedge import GreekHedge

    def overspending(**kwargs):
        return GreekHedge(positions=np.ones(2), entry_cost=1., full_entry_cost=1.,
            risk_residual_norm=0., cap_hits=0, solver="test", status="test", warnings=())

    monkeypatch.setattr("quantdata.hedge2.premium.solve_greek_hedge", overspending)
    units = PremiumUnits(normalized_prices=np.ones(2), normalized_costs=np.array([0., 1.]))
    with pytest.raises(RuntimeError, match="exceeds budget"):
        solve_premium_greek(exposures=np.eye(2), target_exposures=np.ones(2),
            units=units, budget_fraction=.5)


@pytest.mark.parametrize("price", [0., -1., np.nan, np.inf])
def test_invalid_prices_do_not_get_an_invented_premium_floor(price) -> None:
    with pytest.raises(ValueError):
        PremiumUnits(normalized_prices=np.array([1., price]), normalized_costs=np.zeros(2))


def test_original_contract_residual_recovers_without_changing_partial_budget_reference(monkeypatch) -> None:
    from quantdata.hedge2.greek_hedge import solve_greek_hedge

    calls = []

    def weak_original_solve(**kwargs):
        calls.append(dict(kwargs))
        solved = solve_greek_hedge(**kwargs)
        if not kwargs.get("whiten_equalities", False):
            positions = solved.positions.copy()
            positions[1] += 2e-8
            cost = float(kwargs["costs"] @ np.abs(positions - kwargs["previous_positions"]))
            return replace(solved, positions=positions, entry_cost=cost, full_entry_cost=cost)
        return solved

    monkeypatch.setattr("quantdata.hedge2.premium.solve_greek_hedge", weak_original_solve)
    units = PremiumUnits(normalized_prices=np.array([1., .01]), normalized_costs=np.array([0., .001]))
    exposures, target = np.diag([1., .001]), np.array([1., .001])
    full = solve_premium_greek(exposures=exposures, target_exposures=target, units=units)
    np.testing.assert_allclose(full.contract_positions @ exposures, target, rtol=1e-10, atol=1e-14)
    assert any("Original contract-row" in warning for warning in full.result.warnings)
    assert any("minimum-cost gap" in warning for warning in full.result.warnings)
    calls.clear()
    partial = solve_premium_greek(exposures=exposures, target_exposures=target, units=units,
        budget_fraction=.5)
    assert [call.get("whiten_equalities", False) for call in calls] == [False, True, True]
    assert partial.result.full_entry_cost == pytest.approx(full.result.full_entry_cost, abs=1e-14)
    assert partial.result.entry_cost <= .5 * full.result.full_entry_cost + 1e-9
    assert np.linalg.norm(partial.contract_positions @ exposures - target) > 1e-5


def test_failed_original_row_recovery_is_not_economic_infeasibility(monkeypatch) -> None:
    from quantdata.hedge2.greek_hedge import GreekHedge

    def mismatched(**kwargs):
        return GreekHedge(positions=np.array([1., .01 + 2e-8]), entry_cost=.001,
            full_entry_cost=.001, risk_residual_norm=0., cap_hits=0,
            solver="unverified test", status="test", warnings=())

    monkeypatch.setattr("quantdata.hedge2.premium.solve_greek_hedge", mismatched)
    units = PremiumUnits(normalized_prices=np.array([1., .01]), normalized_costs=np.array([0., .001]))
    with pytest.raises(RuntimeError, match="original-row recovery needs review"):
        solve_premium_greek(exposures=np.diag([1., .001]), target_exposures=np.array([1., .001]), units=units)


def test_partial_solver_cannot_overwrite_a_different_full_cost_reference(monkeypatch) -> None:
    from quantdata.hedge2.greek_hedge import GreekHedge

    def changed_reference(**kwargs):
        partial = "budget_fraction" in kwargs
        return GreekHedge(positions=np.array([1., .5 if partial else 1.]),
            entry_cost=.5 if partial else 1., full_entry_cost=2. if partial else 1.,
            risk_residual_norm=0., cap_hits=0, solver="test", status="test", warnings=())

    monkeypatch.setattr("quantdata.hedge2.premium.solve_greek_hedge", changed_reference)
    units = PremiumUnits(normalized_prices=np.ones(2), normalized_costs=np.array([0., 1.]))
    with pytest.raises(RuntimeError, match="full-cost reference"):
        solve_premium_greek(exposures=np.eye(2), target_exposures=np.ones(2), units=units, budget_fraction=.5)
