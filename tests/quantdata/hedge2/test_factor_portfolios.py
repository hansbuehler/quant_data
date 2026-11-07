"""Joint tracking is approximate, bounded and need not produce orthogonal portfolios."""

import numpy as np
import pytest

from quantdata.hedge2.factor_portfolios import (
    FactorPortfolioConfig,
    fit_factor_portfolios,
    premium_to_cash_holdings,
)


def test_joint_portfolios_replicate_feasible_correlated_shocks_without_orthogonality() -> None:
    returns = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    coefficients = np.array([[2., 1.], [-1., 3.]])
    target = returns @ coefficients
    result = fit_factor_portfolios(instrument_returns=returns, factor_shocks=target,
        weights=np.full(4, .25), proportional_costs=np.array([.01, .02]))
    np.testing.assert_allclose(result.premium_positions, coefficients, atol=1e-5)
    np.testing.assert_allclose(result.training_r_squared, 1., atol=1e-9)
    assert abs(result.portfolio_correlation_matrix[0, 1]) > .1
    np.testing.assert_allclose(result.entry_cost, [.04, .07], atol=1e-7)
    assert result.certified


def test_instrument_cap_gives_a_finite_approximate_portfolio_not_an_infeasibility() -> None:
    returns = np.array([[-1.], [1.]])
    result = fit_factor_portfolios(instrument_returns=returns, factor_shocks=20. * returns,
        weights=np.full(2, .5), proportional_costs=np.array([0.]))
    assert result.premium_positions[0, 0] == pytest.approx(10., abs=1e-6)
    assert result.training_r_squared[0] == pytest.approx(.75, abs=1e-7)
    assert result.instrument_cap_hits[0] == 1
    assert result.certified


def test_gross_cap_applies_to_each_factor_not_sum_across_all_eight() -> None:
    returns = np.repeat(np.array([[-1.], [1.]]), 4, axis=1)
    target = np.repeat(40. * returns[:, :1], 8, axis=1)
    result = fit_factor_portfolios(instrument_returns=returns, factor_shocks=target,
        weights=np.full(2, .5), proportional_costs=np.zeros(4),
        config=FactorPortfolioConfig(instrument_cap=10., gross_cap=15.))
    np.testing.assert_allclose(result.gross_positions, 15., atol=1e-6)
    assert result.gross_cap_binding.all()
    assert np.max(np.abs(result.premium_positions)) <= 10.
    assert result.certified
    np.testing.assert_allclose(result.training_rmse, 25., atol=1e-6)


def test_nonzero_asset_alpha_is_rejected_before_factor_portfolio_fit() -> None:
    with pytest.raises(ValueError, match="Remove instrument alpha"):
        fit_factor_portfolios(instrument_returns=np.array([[1.], [2.]]), factor_shocks=np.array([[-1.], [1.]]),
            weights=np.full(2, .5), proportional_costs=np.zeros(1))


def test_cash_holdings_use_cash_prices_not_spot_normalized_prices() -> None:
    allocations = np.array([[10., -3.], [4., 6.]])
    prices = np.array([5000., 25.])
    holdings = premium_to_cash_holdings(allocations, prices)
    np.testing.assert_allclose(holdings, [[.002, -.0006], [.16, .24]])
    np.testing.assert_allclose(np.abs(holdings).T @ prices, np.abs(allocations).sum(axis=0))
