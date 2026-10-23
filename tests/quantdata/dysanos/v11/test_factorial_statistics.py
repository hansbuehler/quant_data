"""Tests for split-safe DYSANOS 1.1 reporting statistics."""

import numpy as np
import pytest

from quantdata.dysanos.v11.factorial_statistics import (
    block_bootstrap_leverage,
    downside_leverage,
    historical_observation_panels,
    lag_correlations,
    transition_sample_indices,
)


def test_transition_split_assigns_first_2025_change_to_forward_sample() -> None:
    dates = np.asarray(
        ["2024-12-27", "2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    )
    training, forward = transition_sample_indices(dates)
    np.testing.assert_array_equal(training, [1, 2])
    np.testing.assert_array_equal(forward, [3, 4])


def test_block_bootstrap_leverage_recovers_exact_linear_relation() -> None:
    spot = np.linspace(-1.0, 1.0, 64)
    changes = np.column_stack((-2.0 * spot, 0.5 * spot))
    result = block_bootstrap_leverage(
        spot, changes, block_length=5, replicates=200, seed=4
    )
    np.testing.assert_allclose(result.correlation, [-1.0, 1.0])
    np.testing.assert_allclose(result.slope, [-2.0, 0.5])
    np.testing.assert_allclose(result.covariance, np.asarray([-2.0, 0.5]) * np.var(spot))
    assert result.correlation_interval.shape == (2, 2)
    assert result.slope_interval[0, 0] <= -2.0 <= result.slope_interval[0, 1]


def test_lag_and_tail_panels_keep_tenors_separate() -> None:
    spot = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    changes = np.column_stack((-spot, spot))
    assert lag_correlations(np.arange(30.0), lags=(1,))[1] == [pytest.approx(1.0)]
    result = downside_leverage(
        spot,
        changes,
        spot_down_threshold=-2.0,
        iv_up_thresholds=np.asarray([2.0, 2.0]),
    )
    np.testing.assert_allclose(result["negative_spot_slope"], [-1.0, 1.0])
    np.testing.assert_allclose(result["nonnegative_spot_slope"], [-1.0, 1.0])
    np.testing.assert_allclose(result["joint_down_up_rate"], [2.0 / 6.0, 0.0])


def test_historical_panels_fix_tail_thresholds_before_forward_sample() -> None:
    dates = np.asarray(
        [f"2024-{month:02d}-{day:02d}" for month in (10, 11) for day in range(1, 26)]
        + [f"2025-{month:02d}-{day:02d}" for month in (1, 2) for day in range(1, 26)]
    )
    dtes = np.asarray([1, 5, 10, 20, 40, 70], dtype=float)
    steps = np.arange(len(dates), dtype=float)
    iv = 0.2 + 0.0005 * steps[:, None] + 0.0001 * dtes[None, :]
    spot = 0.01 * np.sin(steps[1:])
    result = historical_observation_panels(
        dates,
        dtes,
        iv,
        spot,
        bootstrap_replicates=100,
        seed=2,
    )
    train = result["pre_evaluation"]
    forward = result["forward_evaluation"]
    assert train["observations"] == 49
    assert forward["observations"] == 50
    assert forward["first_endpoint_date"] == "2025-01-01"
    assert (
        train["downside_leverage"]["spot_down_threshold"]
        == forward["downside_leverage"]["spot_down_threshold"]
    )
