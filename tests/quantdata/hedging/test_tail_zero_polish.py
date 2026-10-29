from __future__ import annotations

import cvxpy as cp
import numpy as np

from quantdata.hedging import scenario_optimization
from quantdata.hedging.scenario_optimization import (
    ScenarioHedgeFit,
    ScenarioObjective,
    ScenarioOptimizationConfig,
    _fit_tail_epigraph,
    _polish_tail_zero_actions,
    fit_static_scenario_hedge,
)


def test_rounds_only_subgradient_feasible_numerical_positions() -> None:
    original = np.array([-3.56232e-5, 0.1])
    polished, count, bound = _polish_tail_zero_actions(
        original,
        instruments=np.array([[0.00717329, 0.4], [-0.00127221, -0.2]]),
        costs=np.array([2.0132e-5, 0.1]),
        dual_linear_gradient=np.array([1.66486e-5, 0.2]),
        raw_action_ridge=0.001,
        certificate_tolerance=2.0e-6,
    )

    assert original[0] != 0.0
    assert polished.tolist() == [0.0, 0.1]
    assert count == 1
    assert 0.0 < bound <= 0.5e-6


def test_tail_solver_error_retries_at_stricter_tolerance(monkeypatch) -> None:
    tolerances = []

    def solve_once(*, solver_tolerance, **kwargs):
        tolerances.append(solver_tolerance)
        if solver_tolerance == 1.0e-7:
            raise cp.SolverError("initial numerical failure")
        return ScenarioHedgeFit(
            action=np.zeros(1),
            cash_translation=0.0,
            objective_value=0.0,
            iterations=1,
            maximum_kkt_residual=1.0e-9,
            cap_hits=0,
            converged=True,
            solver_message="CLARABEL optimal; certificate=1e-9",
        )

    monkeypatch.setattr(scenario_optimization, "_solve_tail_epigraph", solve_once)
    fit = _fit_tail_epigraph(
        instruments=np.zeros((2, 1)),
        claim=np.zeros(2),
        weights=np.full(2, 0.5),
        costs=np.zeros(1),
        config=ScenarioOptimizationConfig(objective=ScenarioObjective.CVAR_OCE),
    )

    assert tolerances == [1.0e-7, 1.0e-9]
    assert fit.converged
    assert "initial_solve_error=SolverError: initial numerical failure" in fit.solver_message


def test_does_not_round_positions_outside_the_impact_budget() -> None:
    original = np.array([-3.56232e-5])
    polished, count, bound = _polish_tail_zero_actions(
        original,
        instruments=np.array([[0.1], [-0.1]]),
        costs=np.array([2.0132e-5]),
        dual_linear_gradient=np.array([1.66486e-5]),
        raw_action_ridge=0.001,
        certificate_tolerance=2.0e-6,
    )

    assert np.array_equal(polished, original)
    assert count == 0
    assert bound == 0.0


def test_tail_solve_certifies_fallback_after_clarabel_error(monkeypatch) -> None:
    original_solve = cp.Problem.solve

    def unreliable_solve(problem, *args, **kwargs):
        if kwargs.get("solver") == cp.CLARABEL:
            raise cp.SolverError("injected CLARABEL failure")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", unreliable_solve)
    instrument_returns = np.linspace(-1.0, 1.0, 500)
    fit = fit_static_scenario_hedge(
        instrument_returns=instrument_returns[:, None],
        claim_returns=0.3 * instrument_returns + 0.02 * np.sin(3.0 * instrument_returns),
        scenario_weights=np.full(500, 1.0 / 500),
        linear_transaction_costs=np.array([1.0e-4]),
        config=ScenarioOptimizationConfig(
            objective=ScenarioObjective.CVAR_OCE,
            risk_parameter=10.0,
            position_cap=1_000.0,
            raw_action_ridge=0.001,
            certificate_tolerance=2.0e-6,
        ),
    )

    assert fit.converged
    assert fit.solver_message.startswith(("OSQP optimal;", "HIGHS optimal;"))


def test_tail_solve_certifies_fallback_after_clarabel_limit(monkeypatch) -> None:
    original_solve = cp.Problem.solve

    def limited_solve(problem, *args, **kwargs):
        if kwargs.get("solver") == cp.CLARABEL:
            problem._status = cp.USER_LIMIT
            return None
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", limited_solve)
    instrument_returns = np.linspace(-1.0, 1.0, 500)
    fit = fit_static_scenario_hedge(
        instrument_returns=instrument_returns[:, None],
        claim_returns=0.3 * instrument_returns + 0.02 * np.sin(3.0 * instrument_returns),
        scenario_weights=np.full(500, 1.0 / 500),
        linear_transaction_costs=np.array([1.0e-4]),
        config=ScenarioOptimizationConfig(
            objective=ScenarioObjective.CVAR_OCE,
            risk_parameter=10.0,
            position_cap=1_000.0,
            raw_action_ridge=0.001,
            certificate_tolerance=2.0e-6,
        ),
    )

    assert fit.converged
    assert fit.solver_message.startswith(("OSQP optimal;", "HIGHS optimal;"))
    assert "CLARABEL at 1e-07: user_limit" in fit.solver_message


def test_tail_solve_retries_clarabel_at_certified_looser_tolerance(monkeypatch) -> None:
    original_solve = cp.Problem.solve

    def fail_only_at_strict_tolerance(problem, *args, **kwargs):
        if (
            kwargs.get("solver") == cp.CLARABEL
            and kwargs.get("tol_gap_abs", 1.0) < 1.0e-6
        ):
            raise cp.SolverError("injected strict-tolerance failure")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_only_at_strict_tolerance)
    instrument_returns = np.linspace(-1.0, 1.0, 500)
    fit = fit_static_scenario_hedge(
        instrument_returns=instrument_returns[:, None],
        claim_returns=0.3 * instrument_returns + 0.02 * np.sin(3.0 * instrument_returns),
        scenario_weights=np.full(500, 1.0 / 500),
        linear_transaction_costs=np.array([1.0e-4]),
        config=ScenarioOptimizationConfig(
            objective=ScenarioObjective.CVAR_OCE,
            risk_parameter=10.0,
            position_cap=1_000.0,
            raw_action_ridge=0.001,
            certificate_tolerance=2.0e-6,
        ),
    )

    assert fit.converged
    assert fit.solver_message.startswith("CLARABEL optimal;")
    assert "solver_tolerance=1e-06" in fit.solver_message
    assert "injected strict-tolerance failure" in fit.solver_message
