from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2.attenuation_fidelity import block_mean_interval, covariance_score


def test_second_moment_score_is_minimized_at_empirical_second_moment() -> None:
    generator = np.random.default_rng(12)
    observations = generator.normal(size=(20, 3))
    optimum = observations.T @ observations / len(observations)
    actual = np.mean([covariance_score(value, optimum) for value in observations])
    changed = np.mean([covariance_score(value, optimum + .1 * np.eye(3)) for value in observations])
    assert changed - actual == pytest.approx(.01 / 3.)


def test_paired_block_interval_is_reproducible_and_preserves_constant_difference() -> None:
    report = block_mean_interval(np.full(63, .25))
    assert report['mean'] == .25
    assert report['ci95_lower'] == .25
    assert report['ci95_upper'] == .25
    assert block_mean_interval(np.arange(63.)) == block_mean_interval(np.arange(63.))
    with pytest.raises(ValueError, match='finite paired'):
        block_mean_interval(np.array([np.nan] * 63))
