from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2 import hedging_measure as measure


@pytest.fixture(autouse=True)
def _resources(monkeypatch):
    monkeypatch.setattr(measure, "check_resource_headroom", lambda: None)


def test_entropy_weights_preserve_pairs_and_zero_all_original_columns() -> None:
    generator = np.random.default_rng(123)
    base = generator.normal(size=(800, 3)) + [0.08, -0.05, 0.02]
    values = np.column_stack((base, 2.0 * base[:, 0], base[:, 1] * 1.0e-12, np.zeros(800)))
    weights, report = measure.entropy_path_weights(values)
    assert np.array_equal(weights[0::2], weights[1::2])
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(2.0 * weights[0::2] @ values, 0.0, atol=1.0e-7)
    assert report["passed"] and report["effective_pairs"] <= 800
    assert report["retained_dual_directions"] == 3


def test_tiny_nonconstant_returns_are_not_ignored() -> None:
    values = np.asarray([-2.0, -1.0, 1.0, 3.0])[:, None] * 1.0e-12
    weights, report = measure.entropy_path_weights(values)
    assert report["retained_dual_directions"] == 1
    assert abs(float(2.0 * weights[0::2] @ values[:, 0])) < 1.0e-18
    assert not np.allclose(weights, 1.0 / 8.0)


def test_impossible_constant_mean_is_not_silently_recentered() -> None:
    with pytest.raises(ValueError, match="residual checks"):
        measure.entropy_path_weights(np.full((10, 1), 0.01))


def test_zero_return_constraints_leave_uniform_weights() -> None:
    weights, report = measure.entropy_path_weights(np.zeros((8, 3)))
    np.testing.assert_allclose(weights, 1.0 / 16.0)
    assert report["relative_entropy"] == pytest.approx(0.0)


def test_constraints_count_settlement_once_and_exclude_spot_and_expired_returns() -> None:
    prices = np.ones((4, 4, 3))
    prices[:, :, 0] = [1.0, 1.1, 1.2, 1.3]
    prices[:, :, 1] = [0.1, 0.3, 0.0, 0.0]
    prices[:, :, 2] = [0.2, 0.25, 0.3, 0.4]
    original = prices.copy()
    values, coordinates = measure.option_gain_constraints(prices, np.asarray([1, 3]))
    assert values.shape == (2, 8)
    np.testing.assert_allclose(values[0, :4], [0.2, 0.05, 0.05, 0.1])
    np.testing.assert_allclose(values[0, 4:], [0.2, 0.05, 0.05 / 1.1, 0.1 / 1.2])
    assert len(coordinates) == 8
    np.testing.assert_array_equal(prices, original)
