"""Tests for the one-session statistical hedge objectives."""

from __future__ import annotations

import numpy as np

from quantdata.dysanos.flex.rolling_hedging import (
    mean_variance_hedge,
    oce_hedge,
    one_session_pairs,
    paired_block_bootstrap_improvement,
    summarize_hedge_estimates,
    summarize_realized_pnl,
)


def test_one_session_pairs_follow_full_business_calendar() -> None:
    origin, endpoint = one_session_pairs(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-08"]
    )

    np.testing.assert_array_equal(origin, [0, 1])
    np.testing.assert_array_equal(endpoint, [1, 2])


def test_one_session_pairs_skip_half_day_endpoints() -> None:
    origin, endpoint = one_session_pairs(
        ["2025-07-02", "2025-07-03", "2025-07-07", "2025-07-08"]
    )

    np.testing.assert_array_equal(origin, [2])
    np.testing.assert_array_equal(endpoint, [3])


def test_mean_variance_recovers_linear_hedge() -> None:
    generator = np.random.default_rng(17)
    spot = generator.normal(size=10_000)
    portfolio = 0.7 - 1.75 * spot + 0.2 * generator.normal(size=len(spot))
    estimate = mean_variance_hedge(portfolio, spot)
    assert abs(estimate.spot_position - 1.75) < 0.01
    assert abs(estimate.first_order_condition_hedge) < 1.0e-12


def test_oce_first_order_conditions_and_linear_hedge() -> None:
    generator = np.random.default_rng(31)
    spot = generator.normal(scale=0.01, size=50_000)
    independent = generator.normal(scale=0.002, size=len(spot))
    portfolio = -0.8 * spot + independent
    for utility in ("entropy", "power-exp"):
        estimate = oce_hedge(
            portfolio,
            spot,
            utility=utility,
            risk_aversion=50.0,
        )
        assert estimate.optimizer_success
        assert abs(estimate.spot_position - 0.8) < 0.01
        assert abs(estimate.first_order_condition_cash or 0.0) < 1.0e-7
        assert abs(estimate.first_order_condition_hedge) < 1.0e-7


def test_realized_summary_reports_tail_mean() -> None:
    result = summarize_realized_pnl(np.arange(-50.0, 50.0))
    assert result["observations"] == 100
    assert result["root_mean_square"] > result["mean_absolute"]
    assert result["lower_five_percent_mean"] <= result["q05"]


def test_hedge_estimate_audit_reports_convergence_and_bounds() -> None:
    generator = np.random.default_rng(113)
    spot = generator.normal(scale=0.01, size=20_000)
    portfolio = -0.6 * spot + generator.normal(scale=0.002, size=len(spot))
    estimates = [
        mean_variance_hedge(portfolio, spot),
        mean_variance_hedge(-portfolio, spot),
    ]

    audit = summarize_hedge_estimates(estimates)

    assert audit["estimates"] == 2
    assert audit["optimizer_failures"] == 0
    assert audit["spot_bound_hits"] == 0
    assert audit["maximum_absolute_hedge_first_order_condition"] < 1.0e-12
    assert "maximum_absolute_cash_first_order_condition" not in audit


def test_paired_block_bootstrap_detects_uniform_improvement() -> None:
    unhedged = np.linspace(-1.0, 1.0, 101)
    result = paired_block_bootstrap_improvement(
        unhedged,
        0.5 * unhedged,
        replications=999,
    )
    assert result["absolute"]["mean_loss_reduction"] > 0.0
    assert result["squared"]["bootstrap_95_percent_interval"][0] > 0.0
    assert result["squared"]["one_sided_p_no_improvement"] <= 0.002
