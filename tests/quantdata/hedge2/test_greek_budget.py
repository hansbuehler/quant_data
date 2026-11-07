"""Equivalent Greek-budget recovery needs feasibility and an independent objective bound."""

import warnings
from decimal import Decimal, localcontext

import cvxpy as cp
import numpy as np
import pytest

from quantdata.hedge2 import greek_budget
from quantdata.hedge2.greek_budget import greek_projection_gap, solve_greek_budget_projection
from quantdata.hedge2.greek_hedge import solve_greek_hedge


def test_positive_budget_reoptimizes_free_spot_and_paid_option() -> None:
    matrix = np.array([[1., .5], [0., 1.]])
    target = np.array([2., 1.])
    result = solve_greek_budget_projection(matrix=matrix, target=target,
        costs=np.array([0., .1]), previous=np.zeros(2), budget=.05, position_cap=10.)
    np.testing.assert_allclose(result.positions, [1.75, .5], atol=2e-6)
    assert .1 * abs(result.positions[1]) <= .05 + 1e-14
    assert result.objective_gap_bound < 1e-7


def test_exact_zero_budget_freezes_even_tiny_positive_cost_assets() -> None:
    matrix = np.array([[1., .5], [0., 1.]])
    previous = np.array([.3, .4])
    result = solve_greek_budget_projection(matrix=matrix, target=np.array([2., 1.]),
        costs=np.array([0., 1e-30]), previous=previous, budget=0., position_cap=10.)
    assert result.positions[1] == previous[1]
    assert result.positions[0] == pytest.approx(1.8)
    assert result.objective_gap_bound < 1e-10


def test_zero_budget_optimizes_all_free_cross_exposures_with_a_binding_box() -> None:
    matrix = np.array([[1., 1., 0.], [1., -1., 1.]])
    target = np.array([3., 1.])
    result = solve_greek_budget_projection(matrix=matrix, target=target,
        costs=np.array([0., 0., .1]), previous=np.zeros(3), budget=0., position_cap=1.)
    assert np.max(np.abs(result.positions)) <= 1.
    assert result.positions[2] == 0.
    assert float(np.sum((matrix @ result.positions - target)**2)) == pytest.approx(2., abs=1e-10)
    assert result.objective_gap_bound < 1e-7


def test_zero_budget_with_no_free_assets_keeps_the_only_feasible_inventory() -> None:
    previous = np.array([.25, -.5])
    result = solve_greek_budget_projection(matrix=np.eye(2), target=np.array([2., 1.]),
        costs=np.ones(2), previous=previous, budget=0., position_cap=1.)
    np.testing.assert_array_equal(result.positions, previous)
    assert result.objective_gap_bound < 1e-10


def test_tangent_bound_covers_the_actual_quadratic_suboptimality() -> None:
    matrix = np.array([[1., .5], [0., 1.]])
    target, positions = np.array([2., 1.]), np.array([1., 0.])
    gap = greek_projection_gap(matrix=matrix, target=target, positions=positions,
        costs=np.array([0., .1]), previous=np.zeros(2), budget=.05, position_cap=10.)
    actual_difference = float(np.sum((matrix @ positions - target)**2) - .25)
    assert gap >= actual_difference


def test_recovery_preserves_the_full_objective_under_uniform_risk_rescaling() -> None:
    arguments = dict(matrix=np.array([[1., .5], [0., 1.]]), target=np.array([2., 1.]),
        costs=np.array([0., .1]), previous=np.zeros(2), budget=.05, position_cap=10.)
    baseline = solve_greek_budget_projection(**arguments)
    arguments.update(matrix=1e-4 * arguments["matrix"], target=1e-4 * arguments["target"])
    scaled = solve_greek_budget_projection(**arguments)
    np.testing.assert_allclose(scaled.positions, baseline.positions, atol=1e-7)
    assert scaled.objective_gap_bound == pytest.approx(baseline.objective_gap_bound * 1e-8, abs=1e-16)


def test_tangent_roundoff_covers_large_terms_before_cancellation() -> None:
    gap = greek_projection_gap(matrix=np.ones((1, 2)), target=np.array([1e10]),
        positions=np.zeros(2), costs=np.ones(2), previous=np.zeros(2), budget=1e-15, position_cap=10.)
    with localcontext() as context:
        context.prec = 80
        initial = Decimal("1e10")
        budget = Decimal.from_float(1e-15)
        actual_improvement = initial**2 - (initial - budget)**2
        assert np.isfinite(gap) and Decimal.from_float(gap) >= actual_improvement > 0


def test_tangent_overflow_cannot_be_clamped_into_a_zero_certificate() -> None:
    with pytest.raises(RuntimeError, match="Non-finite Greek tangent"):
        greek_projection_gap(matrix=np.array([[1e308]]), target=np.zeros(1),
            positions=np.ones(1), costs=np.ones(1), previous=np.zeros(1), budget=1., position_cap=1.)


def test_nonfinite_tangent_input_cannot_be_certified() -> None:
    with pytest.raises(ValueError, match="finite"):
        greek_projection_gap(matrix=np.array([[np.nan]]), target=np.zeros(1),
            positions=np.zeros(1), costs=np.ones(1), previous=np.zeros(1), budget=1., position_cap=1.)


def test_overflowed_normalization_cannot_be_accepted() -> None:
    with pytest.raises(RuntimeError, match="normalization scale"):
        solve_greek_budget_projection(matrix=np.eye(1), target=np.array([1e308]),
            costs=np.ones(1), previous=np.zeros(1), budget=.1, position_cap=1.)


@pytest.mark.parametrize("budget,previous", [(-1., [0., 0.]), (np.nan, [0., 0.]), (.1, [11., 0.])])
def test_recovery_rejects_invalid_budget_or_prior_box(budget, previous) -> None:
    with pytest.raises(ValueError, match="feasible prior"):
        solve_greek_budget_projection(matrix=np.eye(2), target=np.ones(2), costs=np.ones(2),
            previous=np.array(previous), budget=budget, position_cap=10.)


@pytest.mark.parametrize("budget_fraction", [.5, 0.])
def test_failed_original_solvers_recover_without_discarding_their_warnings(monkeypatch, budget_fraction) -> None:
    original_solve = cp.Problem.solve

    def fail_original(problem, *args, **kwargs):
        if kwargs.get("tol_gap_abs") == 1e-9 or kwargs.get("solver") == "OSQP":
            warnings.warn("Solution may be inaccurate.", UserWarning)
            raise cp.error.SolverError("Original numerical attempt failed")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_original)
    result = solve_greek_hedge(exposures=np.array([[1., 0.], [.5, 1.]]),
        target_exposures=np.array([2., 1.]), costs=np.array([0., .1]),
        budget_fraction=budget_fraction, position_cap=10.)
    np.testing.assert_allclose(result.positions, [2. - .5 * budget_fraction, budget_fraction], atol=2e-6)
    assert result.warnings.count("Solution may be inaccurate.") == 2
    assert any("Original numerical attempt failed" in message for message in result.warnings)
    assert any("Independent Greek squared-risk objective gap" in message for message in result.warnings)


def test_wide_tangent_bound_certifies_a_tiny_budget_without_discarding_roundoff() -> None:
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("This precision-improvement comparison requires wider arithmetic")
    gap = greek_projection_gap(matrix=np.array([[1., 0., 0.], [0., 1e5, 1e5]]),
        target=np.array([0., 1.]), costs=np.array([0., 1e-3, 1e-3]), previous=np.zeros(3),
        positions=np.array([0., 5e-10, 5e-10]), budget=1e-12, position_cap=10.)
    assert 0. < gap < 1e-8


def test_gradient_uncertainty_covers_exact_residual_lost_to_cancellation() -> None:
    gap = greek_projection_gap(matrix=np.array([[1e20, 1., -1e20]]), target=np.zeros(1),
        costs=np.ones(3), previous=np.zeros(3), positions=np.ones(3), budget=3., position_cap=1.)
    assert np.isfinite(gap) and gap >= 1.


def test_no_wider_platform_still_accounts_for_gradient_error(monkeypatch) -> None:
    monkeypatch.setattr(greek_budget.np, "longdouble", np.float64)
    gap = greek_projection_gap(matrix=np.array([[1e20, 1., -1e20]]), target=np.zeros(1),
        costs=np.ones(3), previous=np.zeros(3), positions=np.ones(3), budget=3., position_cap=1.)
    assert np.isfinite(gap) and gap >= 1.


def test_no_wider_platform_does_not_hide_large_support_arithmetic(monkeypatch) -> None:
    monkeypatch.setattr(greek_budget.np, "longdouble", np.float64)
    gap = greek_projection_gap(matrix=np.array([[1., 0., 0.], [0., 1e5, 1e5]]),
        target=np.array([0., 1.]), costs=np.array([0., 1e-3, 1e-3]), previous=np.zeros(3),
        positions=np.array([0., 5e-10, 5e-10]), budget=1e-12, position_cap=10.)
    assert np.isfinite(gap) and gap > 1e-8


def test_positive_bound_below_float64_range_is_rounded_outward() -> None:
    if np.finfo(np.longdouble).tiny >= np.finfo(float).tiny:
        pytest.skip("The subnormal conversion comparison requires wider exponent range")
    gap = greek_projection_gap(matrix=np.array([[1e-200]]), target=np.array([1e-200]),
        costs=np.zeros(1), previous=np.zeros(1), positions=np.zeros(1), budget=0., position_cap=1.)
    assert gap == np.nextafter(0., np.inf)


def test_recovery_evaluates_bound_before_dividing_original_inputs(monkeypatch) -> None:
    original = greek_budget.greek_projection_gap
    captured = []

    def capture(**arguments):
        captured.append((arguments["matrix"].copy(), arguments["target"].copy()))
        return original(**arguments)

    monkeypatch.setattr(greek_budget, "greek_projection_gap", capture)
    matrix, target = np.array([[.01, .005], [0., .01]]), np.array([.02, .01])
    result = solve_greek_budget_projection(matrix=matrix, target=target,
        costs=np.array([0., .1]), previous=np.zeros(2), budget=.05, position_cap=10.)
    assert result.objective_gap_bound < 1e-8
    np.testing.assert_array_equal(captured[0][0], matrix)
    np.testing.assert_array_equal(captured[0][1], target)
