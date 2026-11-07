"""A cost-constrained Greek frontier must re-solve, not shrink all positions."""

import numpy as np
import pytest
from scipy.optimize import linprog

from quantdata.hedge2.greek_hedge import _minimum_cost_certificate, solve_greek_hedge


def test_full_greek_hedge_matches_exposures_with_cheapest_option() -> None:
    exposures = np.array([[1., 0.], [.5, 1.], [.2, 1.]])
    hedge = solve_greek_hedge(exposures=exposures, target_exposures=np.array([2., 1.]),
                              costs=np.array([0., .2, .1]))
    np.testing.assert_allclose(hedge.positions, [1.8, 0., 1.], atol=1e-9)
    assert hedge.entry_cost == pytest.approx(.1)
    assert hedge.risk_residual_norm < 1e-10


def test_zero_cost_budget_keeps_free_spot_hedge_instead_of_scaling_to_zero() -> None:
    exposures = np.array([[1., 0.], [.5, 1.]])
    target = np.array([2., 1.])
    half = solve_greek_hedge(exposures=exposures, target_exposures=target,
                             costs=np.array([0., .1]), budget_fraction=.5)
    zero = solve_greek_hedge(exposures=exposures, target_exposures=target,
                             costs=np.array([0., .1]), budget_fraction=0.)
    np.testing.assert_allclose(half.positions, [1.75, .5], atol=1e-5)
    np.testing.assert_allclose(zero.positions, [2., 0.], atol=1e-5)
    assert half.entry_cost <= .05 + 1e-8
    assert zero.entry_cost < 1e-8


def test_whitened_equalities_preserve_the_minimum_cost_problem() -> None:
    exposures = np.array([[1., 0.], [.5, 1.], [.2, 1.]])
    arguments = dict(exposures=exposures, target_exposures=np.array([2., 1.]), costs=np.array([0., .2, .1]))
    original = solve_greek_hedge(**arguments)
    recovered = solve_greek_hedge(**arguments, whiten_equalities=True)
    np.testing.assert_allclose(recovered.positions, original.positions, atol=1e-9)
    assert recovered.entry_cost == pytest.approx(original.entry_cost, abs=1e-12)
    assert any("minimum-cost gap" in warning for warning in recovered.warnings)


def test_whitening_does_not_silently_drop_dependent_exposure_equations() -> None:
    with pytest.raises(RuntimeError, match="invertible"):
        solve_greek_hedge(exposures=np.array([[1., 1.], [2., 2.]]), target_exposures=np.ones(2),
            costs=np.ones(2), whiten_equalities=True)


@pytest.mark.parametrize("seed", [5, 14, 33, 82])
def test_minimum_cost_bound_matches_an_independent_inventory_lp(seed) -> None:
    rng = np.random.default_rng(seed)
    assets, risks, cap = 8, 3, 10.
    matrix = rng.normal(size=(risks, assets))
    previous = rng.uniform(-cap, cap, size=assets)
    previous[:2] = [-cap, cap]
    target = matrix @ rng.uniform(-cap, cap, size=assets)
    costs = rng.uniform(.01, .1, size=assets)
    costs[0] = 0.
    identity = np.eye(assets)
    result = linprog(c=np.r_[np.zeros(assets), costs],
        A_ub=np.r_[np.c_[identity, -identity], np.c_[-identity, -identity]],
        b_ub=np.r_[previous, -previous], A_eq=np.c_[matrix, np.zeros_like(matrix)],
        b_eq=target, bounds=[(-cap, cap)] * assets + [(0., None)] * assets, method="highs")
    assert result.success
    row_scale = np.array([1e-8, 3., 1e7])
    arguments = dict(matrix=matrix, target=target, costs=costs, previous=previous,
        positions=result.x[:assets], row_scale=row_scale, cost_scale=float(costs.max()), position_cap=cap)
    normalized_dual = result.eqlin.marginals * row_scale / costs.max()
    lower, gap, allowance = _minimum_cost_certificate(**arguments, normalized_dual=normalized_dual)
    assert abs(lower * costs.max() - result.fun) < 1e-10
    assert -allowance <= gap <= 1e-10
    for multiplier in (-10., 0., .3, 17.):
        bound, _, error = _minimum_cost_certificate(**arguments, normalized_dual=multiplier * normalized_dual)
        assert (bound - error) * costs.max() <= result.fun + 1e-12


@pytest.mark.parametrize("target", [-12.5, 12.5])
def test_whitened_minimum_cost_certificate_handles_box_endpoints_and_inventory(target) -> None:
    exposures = np.array([[1.], [.5], [-.25]])
    previous = np.array([8., -9., 2.])
    costs = np.array([0., .2, .1])
    full = solve_greek_hedge(exposures=exposures, target_exposures=np.array([target]),
        costs=costs, previous_positions=previous, position_cap=10., whiten_equalities=True)
    assert np.max(np.abs(full.positions)) <= 10. + 1e-10
    assert full.positions @ exposures[:, 0] == pytest.approx(target, abs=1e-10)
    assert full.entry_cost == pytest.approx(costs @ np.abs(full.positions - previous), abs=1e-12)
    assert any("minimum-cost gap" in warning for warning in full.warnings)


def test_small_cost_many_assets_does_not_fail_from_float64_bound_accumulation() -> None:
    count = 112
    exposures = np.c_[np.ones(count), np.linspace(-1., 1., count)]
    costs = np.full(count, .0999)
    costs[0] = 0.
    result = solve_greek_hedge(exposures=exposures, target_exposures=np.array([1e-4, 2e-4]),
        costs=costs, previous_positions=np.zeros(count), position_cap=10., whiten_equalities=True)
    np.testing.assert_allclose(result.positions @ exposures, [1e-4, 2e-4], rtol=1e-10, atol=1e-14)
    assert result.entry_cost < 1e-4


def test_whitened_solver_cannot_pass_with_an_uninformative_dual(monkeypatch) -> None:
    import quantdata.hedge2.greek_hedge as implementation

    original = implementation.linprog

    def bad_dual(**kwargs):
        result = original(**kwargs)
        result.eqlin.marginals[:] = 0.
        return result

    monkeypatch.setattr(implementation, "linprog", bad_dual)
    with pytest.raises(RuntimeError, match="minimum-cost dual check"):
        solve_greek_hedge(exposures=np.eye(2), target_exposures=np.ones(2),
            costs=np.ones(2), position_cap=10., whiten_equalities=True)


def test_minimum_cost_certificate_rejects_nonfinite_dual() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        _minimum_cost_certificate(matrix=np.eye(2), target=np.ones(2), costs=np.ones(2),
            previous=np.zeros(2), positions=np.ones(2), normalized_dual=np.full(2, np.nan),
            row_scale=np.ones(2), cost_scale=1., position_cap=10.)
