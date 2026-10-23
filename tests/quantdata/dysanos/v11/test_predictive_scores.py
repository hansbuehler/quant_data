"""Tests for proper sample-based forecast scores."""

from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.v11.predictive_scores import (
    empirical_crps,
    predictive_panel,
)


def test_empirical_crps_matches_pairwise_formula() -> None:
    observed = np.array([[0.3, -0.4], [0.8, 0.1]])
    samples = np.array(
        [
            [[-0.2, -0.6], [0.0, -0.1], [0.5, 0.2]],
            [[0.1, -0.3], [0.7, 0.0], [1.0, 0.4]],
        ]
    )
    expected = np.mean(np.abs(samples - observed[:, None, :]), axis=1)
    expected -= np.mean(
        np.abs(samples[:, :, None, :] - samples[:, None, :, :]),
        axis=(1, 2),
    ) / 2.0
    np.testing.assert_allclose(empirical_crps(observed, samples), expected)


def test_predictive_panel_identifies_bias_and_calibration() -> None:
    generator = np.random.default_rng(4)
    draws = generator.normal(size=(2000, 100, 2))
    outcomes = generator.normal(size=(2000, 2))
    report = predictive_panel(outcomes, draws)
    assert report["origins"] == 2000
    assert report["paths_per_origin"] == 100
    assert np.max(np.abs(report["innovation_bias"])) < 0.08
    assert np.allclose(report["predictive_variance_ratio"], 1.0, atol=0.08)
    assert np.allclose(report["central_interval_coverage"]["0.80"], 0.8, atol=0.06)


def test_empirical_crps_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="observed must be"):
        empirical_crps(np.ones((2, 1)), np.ones((3, 2, 1)))
