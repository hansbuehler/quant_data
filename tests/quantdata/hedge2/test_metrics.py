"""Tail definitions and paired date dependence remain explicit in benchmark metrics."""

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.metrics import (
    calendar_block_weights,
    lower_tail_mean,
    paired_risk_interval,
    summarize_products,
)


def test_fractional_tail_mean_and_sample_standard_deviation() -> None:
    pnl = np.array([-4., -1., 1., 2.])
    assert lower_tail_mean(pnl, fraction=.375) == pytest.approx(-3.)
    assert lower_tail_mean(pnl, fraction=1.) == pytest.approx(pnl.mean())
    frame = pd.DataFrame({"date": range(4), "claim": ["d"] * 4, "costed_pnl": pnl, "entry_cost": [.1] * 4})
    result = summarize_products(frame, group_columns=("claim",)).iloc[0]
    assert result.std_pnl == pytest.approx(pnl.std(ddof=1))
    assert result.mean_loss == pytest.approx(1.25)
    assert result.cost_over_std == pytest.approx(.1 / pnl.std(ddof=1))


def test_paired_bootstrap_keeps_identical_strategies_identical_with_date_gaps() -> None:
    values = np.random.default_rng(19).normal(size=(30, 2))
    sessions = np.arange(30) * 2
    result = paired_risk_interval(candidate=values, reference=values, session_indices=sessions,
                                  count=200, block_length=5, seed=18)
    assert result["std_difference"] == 0.
    assert result["std_difference_ci95_lower"] == 0.
    assert result["std_difference_ci95_upper"] == 0.
    assert result["mean_difference"] == 0.
    first = calendar_block_weights(session_indices=sessions, count=200, block_length=5, seed=18)
    second = calendar_block_weights(session_indices=sessions, count=200, block_length=5, seed=18)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(axis=0), 1., atol=.25)
