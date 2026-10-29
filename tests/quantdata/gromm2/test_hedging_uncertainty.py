from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2.hedging_policy import optimized_oce
from quantdata.gromm2.hedging_uncertainty import payoff_hedge_statistics, risk_path_scores


def test_risk_scores_have_the_negative_optimized_oce_as_their_mean() -> None:
    pnl = np.array([-0.08, 0.03, 0.01, -0.02, 0.04, -0.01])
    scores, _ = risk_path_scores(pnl, risk_aversion=100.0)
    oce, _, _ = optimized_oce(pnl, 100.0)
    assert scores.mean() == pytest.approx(-oce)


def test_expectation_error_clusters_adjacent_antithetic_partners() -> None:
    payoff = np.array([0.01, 0.03, 0.02, 0.06, 0.03, 0.09])
    result = payoff_hedge_statistics(payoff, np.zeros(6), np.zeros(6), risk_aversion=100.0)
    paired = payoff.reshape(-1, 2).mean(axis=1)
    assert result["expectation"]["standard_error"] == pytest.approx(paired.std(ddof=1) / np.sqrt(3))
    assert result["expectation"]["pair_count"] == 3
    assert result["DH_empty"]["standard_error"] == 0.0


def test_incremental_risk_error_retains_covariance_between_both_policies() -> None:
    gain = np.array([-0.03, 0.02, -0.01, 0.05, 0.01, -0.02])
    payoff = np.full(6, 0.02)
    result = payoff_hedge_statistics(payoff, gain, gain, risk_aversion=100.0)
    assert result["DH_empty"]["standard_error"] > 0.0
    assert result["DH_claim"]["standard_error"] > 0.0
    assert result["DH_price"]["mean"] == pytest.approx(0.02)
    assert result["DH_price"]["standard_error"] < 1.0e-10


def test_cash_constant_risk_has_zero_uncertainty() -> None:
    payoff = np.full(6, 0.02)
    result = payoff_hedge_statistics(payoff, np.zeros(6), np.zeros(6), risk_aversion=100.0)
    assert result["DH_unhedged_payoff"]["mean"] == pytest.approx(0.02)
    assert result["DH_unhedged_payoff"]["standard_error"] == 0.0


@pytest.mark.parametrize("values", [np.zeros(5), np.array([0.01, np.nan, 0.03, 0.02])])
def test_statistics_refuse_incomplete_or_nonfinite_antithetic_samples(values) -> None:
    with pytest.raises(ValueError):
        payoff_hedge_statistics(values, np.zeros_like(values), np.zeros_like(values), risk_aversion=100.0)
