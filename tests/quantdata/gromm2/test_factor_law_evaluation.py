from __future__ import annotations

import numpy as np

from quantdata.gromm2.factor_law_evaluation import (
    _moving_block_means,
    _skill_summary,
)


def test_paired_block_skill_detects_uniform_improvement() -> None:
    baseline = np.linspace(0.5, 1.5, 100)
    candidate = baseline - 0.2
    result = _skill_summary(
        baseline,
        candidate,
        block_length=10,
        samples=2_000,
        generator=np.random.default_rng(7),
    )

    assert np.isclose(result["mean_improvement"], 0.2)
    assert result["confidence_interval_95_lower"] > 0.0
    assert result["bootstrap_probability_positive"] == 1.0


def test_moving_block_means_preserve_constant_series() -> None:
    result = _moving_block_means(
        np.full(50, 3.0),
        block_length=7,
        samples=1_000,
        generator=np.random.default_rng(11),
    )

    np.testing.assert_allclose(result, 3.0)
