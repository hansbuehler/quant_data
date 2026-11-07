"""Convex scenario objectives have known one-asset solutions and stable OCE gradients."""

import warnings

import cvxpy as cp
import numpy as np
import pytest

from quantdata.hedge2.certification import certify_candidate
from quantdata.hedge2.objectives import power_oce
from quantdata.hedge2.optimization import ScenarioOptimizer


def test_power_oce_is_cash_invariant_and_its_gradient_is_the_envelope() -> None:
    values = np.array([-.2, -.05, .1, .3])
    weights = np.array([.1, .2, .3, .4])
    risk = power_oce(values, weights, risk_aversion=10.)
    shifted = power_oce(values + 3., weights, risk_aversion=10.)
    assert shifted.certainty_equivalent == pytest.approx(risk.certainty_equivalent + 3.)
    assert abs(risk.cash_first_order_residual) < 1e-10
    bump = np.zeros(4)
    bump[1] = 1e-6
    up = power_oce(values + bump, weights, risk_aversion=10.)
    down = power_oce(values - bump, weights, risk_aversion=10.)
    assert (up.certainty_equivalent - down.certainty_equivalent) / 2e-6 == pytest.approx(
        weights[1] * risk.marginal_utility[1], rel=1e-6)


def test_mean_variance_and_power_hedge_replicable_claim() -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    weights = np.full(4, .25)
    optimizer = ScenarioOptimizer(instrument_gains=gains, weights=weights, costs=np.zeros(1))
    for objective, risk in (("mean_variance", 1.), ("mean_variance", np.inf), ("power_exponential", 10.),
                            ("power_exponential", 1000.)):
        hedge = optimizer.solve(claim_gains=2. * gains[:, 0], objective=objective, risk_aversion=risk)
        assert hedge.positions[0] == pytest.approx(2., abs=1e-4)
        assert hedge.training_std < 1e-6
        assert hedge.status in ("optimal", "optimal_inaccurate")


def test_mv_transaction_cost_has_exact_soft_threshold_solution() -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    weights = np.full(4, .25)
    variance = float(weights @ gains[:, 0]**2)
    cost = .0001
    optimizer = ScenarioOptimizer(instrument_gains=gains, weights=weights, costs=np.array([cost]))
    hedge = optimizer.solve(claim_gains=2. * gains[:, 0], objective="mean_variance", risk_aversion=10.)
    assert hedge.positions[0] == pytest.approx(2. - cost / (10. * variance), abs=1e-5)


def test_cash_unit_power_formulation_keeps_disciplined_parameter_programming() -> None:
    gains = np.array([[-.02], [.02]])
    optimizer = ScenarioOptimizer(instrument_gains=gains, weights=np.full(2, .5), costs=np.zeros(1))
    assert optimizer.power_problem.is_dpp()
    result = optimizer.solve(claim_gains=2. * gains[:, 0], objective="power_exponential", risk_aversion=1000.)
    assert result.positions[0] == pytest.approx(2., abs=1e-4)
    certified, evidence = certify_candidate(optimizer=optimizer, candidate=result,
        claim_gains=2. * gains[:, 0], objective="power_exponential", risk_aversion=1000., target_side=-1)
    assert evidence["optimization_certified"]
    assert certified.training_objective < 1e-7


def test_power_conic_failure_uses_feasible_fallback_and_preserves_exact_warning(monkeypatch) -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    optimizer = ScenarioOptimizer(instrument_gains=gains, weights=np.full(4, .25), costs=np.zeros(1))

    def fail_with_warning(*args, **kwargs):
        warnings.warn("Solution may be inaccurate.", UserWarning)
        raise cp.error.SolverError("deliberate conic test failure")

    monkeypatch.setattr(optimizer.power_problem, "solve", fail_with_warning)
    result = optimizer.solve(claim_gains=2. * gains[:, 0], objective="power_exponential", risk_aversion=1000.)
    assert result.positions[0] == pytest.approx(2., abs=1e-4)
    assert result.solver == "SLSQP cash-eliminated fallback"
    assert result.warnings.count("Solution may be inaccurate.") == 2
    assert any("deliberate conic test failure" in item for item in result.warnings)
    certified, evidence = certify_candidate(optimizer=optimizer, candidate=result,
        claim_gains=2. * gains[:, 0], objective="power_exponential", risk_aversion=1000., target_side=-1)
    assert evidence["optimization_certified"]
    assert certified.training_objective < 1e-7
