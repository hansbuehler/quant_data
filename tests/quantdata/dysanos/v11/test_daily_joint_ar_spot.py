from __future__ import annotations

import numpy as np

from quantdata.dysanos.v11.daily_joint_ar_spot import (
    fit_daily_joint_ar_spot,
    sample_daily_joint_ar_spot,
)


def _synthetic_daily_joint_data() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(17)
    dimension = 6
    count = 400
    transition = np.diag([0.85, 0.70, 0.55, 0.40, 0.25, 0.10])
    loadings = np.diag([0.30, 0.22, 0.18, 0.14, 0.10, 0.08])
    gamma = np.array([0.55, -0.25, 0.0, 0.0, 0.0, 0.0])
    states = np.zeros((count + 1, dimension))
    scores = np.empty(count)
    residual_scale = np.sqrt(1.0 - gamma @ gamma)
    for index in range(count):
        epsilon = generator.standard_normal(dimension)
        states[index + 1] = transition @ states[index] + loadings @ epsilon
        scores[index] = gamma @ epsilon + residual_scale * generator.standard_normal()
    return states, scores


def test_fit_daily_joint_ar_spot_is_stable_and_spot_aware() -> None:
    states, scores = _synthetic_daily_joint_data()
    fit = fit_daily_joint_ar_spot(states, scores, rank=3)

    assert fit.factor_basis.shape == (6, 3)
    assert np.allclose(fit.factor_basis.T @ fit.factor_basis, np.eye(3), atol=1e-10)
    assert fit.spectral_radius < 1.0
    assert np.linalg.norm(fit.gamma) > 0.3
    assert fit.spot_direction_fraction > 0.0


def test_diagonal_daily_ar_avoids_dense_transition_overfit() -> None:
    states, scores = _synthetic_daily_joint_data()
    fit = fit_daily_joint_ar_spot(
        states[:260],
        scores[:259],
        rank=6,
        transition_structure="diagonal",
    )
    assert np.allclose(fit.factor_transition, np.diag(np.diag(fit.factor_transition)))
    assert fit.spectral_radius < 1.0


def test_sample_daily_joint_ar_spot_is_antithetic() -> None:
    states, scores = _synthetic_daily_joint_data()
    fit = fit_daily_joint_ar_spot(states, scores, rank=3)
    next_states, next_scores = sample_daily_joint_ar_spot(fit, states[10:12], count=200, seed=4)

    centered = states[10:12] - fit.state_mean
    conditional_factor_mean = fit.factor_intercept + (centered @ fit.factor_basis) @ fit.factor_transition
    conditional_mean = fit.state_mean + conditional_factor_mean @ fit.factor_basis.T
    assert next_states.shape == (2, 200, 6)
    assert next_scores.shape == (2, 200)
    assert np.allclose(next_states.mean(axis=1), conditional_mean, atol=1e-12)
    assert np.allclose(next_scores.mean(axis=1), fit.spot_score_mean, atol=1e-12)
