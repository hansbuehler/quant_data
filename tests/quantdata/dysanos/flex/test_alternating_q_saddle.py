from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.flex.run_alternating_q_saddle import (
    effective_sample_size,
    game_objective,
    mirror_ascent_density_step,
    project_relative_density,
)


def test_projection_preserves_positivity_normalization_and_ess_floor() -> None:
    base = np.array([0.5, 0.8, 1.1, 1.6], dtype=np.float64)
    candidate = np.array([-20.0, -5.0, 3.0, 25.0], dtype=np.float64)
    result = project_relative_density(
        candidate_log_relative_density=candidate,
        base_weights=base,
        minimum_ess_fraction=0.7,
        minimum_relative_density=0.05,
    )

    assert np.mean(result.combined_weights) == pytest.approx(1.0, abs=1.0e-12)
    assert np.min(result.relative_density) >= 0.05 - 1.0e-12
    assert effective_sample_size(result.combined_weights) >= 2.8 - 1.0e-8
    assert 0.0 <= result.temperature < 1.0


def test_mirror_ascent_increases_fixed_action_game_value() -> None:
    base = np.ones(6, dtype=np.float64)
    relative = np.ones(6, dtype=np.float64)
    log_loss = np.log(np.array([0.5, 0.8, 1.0, 1.2, 2.0, 3.0]))
    result = mirror_ascent_density_step(
        relative_density=relative,
        base_weights=base,
        log_path_loss=log_loss,
        learning_rate=0.1,
        entropy_strength=0.0,
        reward_clip=10.0,
        minimum_ess_fraction=0.5,
        minimum_relative_density=0.01,
    )

    assert result.objective_after > result.objective_before
    assert game_objective(log_loss, result.projection.combined_weights) == pytest.approx(
        result.objective_after
    )


def test_constant_loss_leaves_uniform_relative_density_unchanged() -> None:
    base = np.array([0.5, 1.0, 1.5, 1.0], dtype=np.float64)
    relative = np.ones(4, dtype=np.float64)
    result = mirror_ascent_density_step(
        relative_density=relative,
        base_weights=base,
        log_path_loss=np.full(4, -0.2),
        learning_rate=0.1,
        entropy_strength=0.0,
        reward_clip=10.0,
        minimum_ess_fraction=0.7,
        minimum_relative_density=0.01,
    )

    np.testing.assert_allclose(result.projection.relative_density, 1.0, atol=1.0e-12)
    assert result.objective_after == pytest.approx(result.objective_before)
