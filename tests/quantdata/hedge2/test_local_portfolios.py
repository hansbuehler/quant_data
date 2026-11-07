"""Local mixed price exposures span factors without historical regression or forced zeros."""

import numpy as np
import pytest

import quantdata.hedge2.local_portfolios as local_module
from quantdata.hedge2.factor_portfolios import FactorPortfolioConfig
from quantdata.hedge2.local_portfolios import (
    LocalPortfolioConfig,
    _extreme_portfolio,
    fit_local_portfolios,
    hedge_in_local_basis,
    lookthrough_positions,
    minimize_basis_costs,
)


def test_local_baskets_span_mixed_factors_and_obey_each_dollar_budget() -> None:
    rng = np.random.default_rng(3281)
    greeks = rng.normal(size=(35, 8))
    result = fit_local_portfolios(price_factor_greeks=greeks, proportional_costs=np.full(35, .01))
    assert np.linalg.matrix_rank(result.factor_loadings) == 8
    np.testing.assert_allclose(result.factor_loadings, result.premium_positions.T @ greeks)
    assert np.max(np.abs(result.premium_positions)) <= 10.
    assert np.max(result.gross_positions) <= 100.
    assert np.all(np.diff(result.log_volumes) >= -1e-9)
    assert np.max(np.abs(result.factor_loadings - np.diag(np.diag(result.factor_loadings)))) > 1.
    assert result.converged
    np.testing.assert_allclose(result.entry_costs, .01 * result.gross_positions)


def test_basket_span_reconstructs_current_payoff_greeks_without_unit_factor_payouts() -> None:
    greeks = np.array([[0., 0.], [1e-4, 2e-4], [3e-4, -1e-4], [-2e-4, 4e-4]])
    result = fit_local_portfolios(price_factor_greeks=greeks, proportional_costs=np.zeros(4),
        config=LocalPortfolioConfig(limits=FactorPortfolioConfig(instrument_cap=10., gross_cap=15.)))
    target = np.array([.1, -.2])
    holdings = np.linalg.solve(result.factor_loadings.T, target)
    actual = lookthrough_positions(basis_positions=result.premium_positions, basket_holdings=holdings)
    np.testing.assert_allclose(greeks.T @ actual, target, atol=1e-12)
    assert np.max(np.abs(actual)) > 10.
    assert np.max(np.abs(result.premium_positions)) <= 10.


def test_overlapping_basket_trades_are_netted_before_charging_costs() -> None:
    basis = np.array([[0., 0.], [10., 10.], [5., -5.]])
    holdings = np.array([1., -1.])
    actual = lookthrough_positions(basis_positions=basis, basket_holdings=holdings, extra_spot_position=3.)
    np.testing.assert_allclose(actual, [3., 0., 10.])
    costs = np.array([0., .01, .02])
    assert costs @ np.abs(actual) == pytest.approx(.2)
    assert (costs @ np.abs(basis)) @ np.abs(holdings) == pytest.approx(.4)


def test_rank_deficiency_is_not_reported_as_full_factor_coverage() -> None:
    with pytest.raises(ValueError, match="cannot span"):
        fit_local_portfolios(price_factor_greeks=np.ones((5, 2)), proportional_costs=np.zeros(5))


def test_local_coverage_iteration_limit_is_visible_not_silently_called_converged() -> None:
    rng = np.random.default_rng(1)
    result = fit_local_portfolios(price_factor_greeks=rng.normal(size=(70, 8)), proportional_costs=np.zeros(70),
        config=LocalPortfolioConfig(maximum_sweeps=1))
    assert not result.converged
    assert result.sweeps == 1
    assert result.block_improvement_bounds.max() > 1e-7


def test_cost_refinement_keeps_all_mixed_exposures_and_has_independent_lower_bound() -> None:
    greeks = np.array([[1., 1., 0.], [2., 2., 1.]])
    positions = np.array([[1., 2.], [0., 0.], [1., -1.]])
    costs = np.array([.1, .01, .03])
    result = minimize_basis_costs(exposure_map=greeks, premium_positions=positions, proportional_costs=costs)
    np.testing.assert_allclose(result.greek_loadings, greeks @ positions, atol=1e-9)
    np.testing.assert_allclose(result.premium_positions, [[0., 0.], [1., 2.], [1., -1.]], atol=1e-9)
    np.testing.assert_allclose(result.entry_costs, [.04, .05], atol=1e-9)
    assert np.all(result.cost_lower_bounds <= result.entry_costs + 1e-9)
    assert np.max(result.cost_gaps) < 1e-8
    assert np.max(result.scaled_exposure_residuals) < 1e-9


def test_support_uses_fractional_last_asset_budget() -> None:
    positions = _extreme_portfolio(np.array([3., -2., 1.]), instrument_cap=10., gross_cap=15.)
    np.testing.assert_array_equal(positions, [10., -5., 0.])


def test_cost_refinement_preserves_exposure_with_instrument_and_gross_caps_binding() -> None:
    result = minimize_basis_costs(exposure_map=np.array([[1., 2.]]),
        premium_positions=np.array([[1.], [.75]]), proportional_costs=np.array([0., 1.]),
        limits=FactorPortfolioConfig(instrument_cap=1., gross_cap=1.75))
    np.testing.assert_allclose(result.premium_positions, [[1.], [.75]], atol=1e-9)
    np.testing.assert_allclose(result.entry_costs, [.75], atol=1e-9)
    np.testing.assert_allclose(result.cost_gaps, 0., atol=1e-9)


def test_spot_and_mixed_baskets_offset_target_without_forcing_other_greeks_flat() -> None:
    exposures = np.array([[.02, .7, -.3, .5], [0., .01, .02, -.01], [0., .03, -.01, .02]])
    basis = np.array([[0., 0.], [3., -2.], [-1., 4.], [2., 1.]])
    target = np.array([.2, -.03, .04])
    long = hedge_in_local_basis(exposure_map=exposures, basis_positions=basis,
        target_exposures=target, target_side=1)
    short = hedge_in_local_basis(exposure_map=exposures, basis_positions=basis,
        target_exposures=target, target_side=-1)
    np.testing.assert_allclose(exposures @ long.premium_positions, -target, atol=1e-12)
    np.testing.assert_allclose(short.premium_positions, -long.premium_positions, atol=1e-12)
    np.testing.assert_allclose(long.greek_residual, 0., atol=1e-12)
    assert np.count_nonzero(exposures @ basis) == 6


def test_augmented_spot_basis_rank_failure_is_explicit() -> None:
    with pytest.raises(ValueError, match="cannot span"):
        hedge_in_local_basis(exposure_map=np.array([[0., 1.], [0., 2.]]),
            basis_positions=np.ones((2, 1)), target_exposures=np.ones(2), target_side=1)


def test_accuracy_warning_alone_does_not_reject_independently_accurate_cost_fit(monkeypatch) -> None:
    original = local_module.linprog

    def warned_result(**kwargs):
        result = original(**kwargs)
        result.success = False
        result.message = "Solution may be inaccurate."
        return result

    monkeypatch.setattr(local_module, "linprog", warned_result)
    result = minimize_basis_costs(exposure_map=np.array([[1., 2.]]),
        premium_positions=np.array([[1.], [.75]]), proportional_costs=np.array([0., 1.]),
        limits=FactorPortfolioConfig(instrument_cap=1., gross_cap=1.75))
    assert result.solver_messages == ("Solution may be inaccurate.",)
    assert result.cost_gaps[0] < 1e-9
    assert result.scaled_exposure_residuals[0] < 1e-9
