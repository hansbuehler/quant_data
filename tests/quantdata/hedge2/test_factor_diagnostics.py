"""Payout normalization changes feasibility without changing the stated portfolio budgets."""

import numpy as np
import pytest

from quantdata.hedge2.factor_diagnostics import tracking_payout_curve, unconstrained_tracking


def test_smaller_payout_improves_tracking_without_relaxing_either_cap() -> None:
    returns = np.array([[-.01], [.01]])
    shocks = np.array([[-1.], [1.]])
    results, arrays = tracking_payout_curve(instrument_returns=returns, factor_shocks=shocks,
        weights=np.full(2, .5), proportional_costs=np.array([.02]), payouts=(1., .1, .01))
    np.testing.assert_allclose(results.training_r_squared, [.19, 1., 1.], atol=1e-6)
    assert np.all(results.gross_positions <= 100.)
    assert np.all(results.maximum_position <= 10.)
    assert results.certified.all()
    np.testing.assert_allclose(arrays["premium_positions_2"], [[1.]], atol=1e-5)
    np.testing.assert_allclose(results.entry_cost_per_payout, [.2, 2., 2.], atol=1e-4)
    for index, payout in enumerate((1., .1, .01)):
        np.testing.assert_allclose(results.entry_cost_per_payout.iloc[index],
            .02 * np.abs(arrays[f"premium_positions_{index}"]).sum() / payout)


@pytest.mark.parametrize("payouts", [(), (0.,), (-1.,), (np.inf,), (1., 1.)])
def test_invalid_payout_experiments_are_rejected(payouts) -> None:
    with pytest.raises(ValueError, match="payouts"):
        tracking_payout_curve(instrument_returns=np.array([[-1.], [1.]]), factor_shocks=np.array([[-1.], [1.]]),
            weights=np.full(2, .5), proportional_costs=np.zeros(1), payouts=payouts)


def test_unconstrained_fit_uses_minimum_l2_dollars_not_an_arbitrary_nullspace_portfolio() -> None:
    returns = np.array([[-.01, -.01], [.01, .01]])
    shocks = np.array([[-1.], [1.]])
    result = unconstrained_tracking(instrument_returns=returns, factor_shocks=shocks, weights=np.array([.5, .5]))
    np.testing.assert_allclose(result.premium_positions, [[50.], [50.]], atol=1e-10)
    np.testing.assert_allclose(result.training_r_squared, 1., atol=1e-12)
    assert result.rank == 1
    alternative = result.premium_positions + np.array([[10.], [-10.]])
    np.testing.assert_allclose(returns @ alternative, returns @ result.premium_positions)
    assert np.linalg.norm(result.premium_positions) < np.linalg.norm(alternative)


def test_unconstrained_fit_discloses_cutoff_sensitive_large_null_direction_positions() -> None:
    returns = np.array([[-1., -1e-9], [-1., 1e-9], [1., -1e-9], [1., 1e-9]])
    shocks = np.array([[-1.], [1.], [-1.], [1.]])
    full = unconstrained_tracking(instrument_returns=returns, factor_shocks=shocks, weights=np.full(4, .25))
    truncated = unconstrained_tracking(instrument_returns=returns, factor_shocks=shocks,
        weights=np.full(4, .25), relative_cutoff=1e-6)
    assert full.premium_positions[1, 0] == pytest.approx(1e9)
    assert full.training_r_squared[0] == pytest.approx(1.)
    assert truncated.rank == 1
    assert truncated.training_r_squared[0] == pytest.approx(0.)
