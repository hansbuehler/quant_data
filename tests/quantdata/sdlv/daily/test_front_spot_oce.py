from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from quantdata.sdlv.daily.front_spot.oce import (
    FrontSpotOcePolicy,
    FrontSpotOcePolicyConfig,
    action_moments,
    log_exponential_loss,
    negative_exponential_statistics,
    paired_conditional_drift_statistics,
    regularized_transition_gain,
)
from quantdata.sdlv.daily.front_spot.run_oce import (
    FrontSpotOceTrainConfig,
    _evaluate_policy,
    _fit_policy,
    _spot_normalized_price_increments,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _policy() -> FrontSpotOcePolicy:
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = FrontSpotOcePolicyConfig(
        label="test-front-spot-oce",
        feature_count=3,
        instrument_count=2,
        network=FeedForwardConfig(activation="SiLU", num_depth=2, num_width=4),
    )
    return FrontSpotOcePolicy(
        environment,
        config=config,
        feature_mean=torch.zeros(3, dtype=environment.fdtype),
        feature_scale=torch.ones(3, dtype=environment.fdtype),
    )


def test_policy_starts_at_zero_and_masks_inactive_coordinates() -> None:
    policy = _policy()
    features = torch.randn(5, 3, 3, dtype=torch.float64)
    active = torch.tensor(
        [[True, True], [True, False], [False, False]],
        dtype=torch.bool,
    )
    action = policy(features, active)
    assert action.shape == (5, 3, 2)
    assert torch.count_nonzero(action) == 0

    with torch.no_grad():
        policy.affine.bias.fill_(2.0)
    action = policy(features, active)
    assert torch.all(action[:, 0] == 2.0)
    assert torch.all(action[:, 1, 0] == 2.0)
    assert torch.all(action[:, 1, 1] == 0.0)
    assert torch.all(action[:, 2] == 0.0)


def test_local_regularized_objective_has_expected_deterministic_optimum() -> None:
    returns = torch.tensor([[[0.2], [0.2]], [[0.2], [0.2]]], dtype=torch.float64)
    optimal_action = torch.full_like(returns, 0.2)
    transition_gain = regularized_transition_gain(
        optimal_action,
        returns,
        epsilon=1.0,
    )
    assert torch.allclose(
        transition_gain,
        torch.full((2, 2), 0.02, dtype=torch.float64),
    )
    assert float(log_exponential_loss(transition_gain)) == pytest.approx(-0.02)
    statistics = negative_exponential_statistics(transition_gain.numpy())
    assert statistics["negative_expectation"] == pytest.approx(-math.exp(-0.02))
    assert statistics["standard_error"] == pytest.approx(0.0)
    assert statistics["tilted_effective_sample_size"] == pytest.approx(4.0)


def test_exponential_statistics_keep_finite_log_when_level_overflows() -> None:
    statistics = negative_exponential_statistics(
        np.asarray([-1_000.0, 0.0], dtype=np.float64)
    )
    assert statistics["log_expectation"] == pytest.approx(1_000.0 - math.log(2.0))
    assert statistics["expectation"] == math.inf


def test_action_outer_moment_trace_equals_pointwise_energy() -> None:
    actions = np.asarray(
        [
            [[1.0, 2.0], [0.0, 3.0]],
            [[-1.0, 0.0], [2.0, -2.0]],
        ],
        dtype=np.float64,
    )
    summary, outer = action_moments(actions)
    expected_outer = np.einsum("pti,ptj->ij", actions, actions) / 4.0
    assert np.allclose(outer, expected_outer)
    assert summary["expected_pointwise_action_energy"] == pytest.approx(
        np.trace(expected_outer)
    )
    assert summary["expected_path_action_energy"] == pytest.approx(
        2.0 * np.trace(expected_outer)
    )


def test_conditional_drift_inference_uses_antithetic_pairs() -> None:
    actions = np.ones((4, 2, 1), dtype=np.float64)
    returns = np.asarray(
        [
            [[1.0], [3.0]],
            [[-1.0], [-1.0]],
            [[2.0], [2.0]],
            [[0.0], [2.0]],
        ],
        dtype=np.float64,
    )
    summary, path_average, pair_average = paired_conditional_drift_statistics(
        actions,
        returns,
    )
    assert np.allclose(path_average, [2.0, -1.0, 2.0, 1.0])
    assert np.allclose(pair_average, [0.5, 1.5])
    assert summary["mean_daily_conditional_drift"] == pytest.approx(1.0)
    assert summary["pair_based_standard_error"] == pytest.approx(0.5)
    assert summary["z_score"] == pytest.approx(2.0)


def test_option_returns_are_price_increments_scaled_by_pre_step_spot() -> None:
    current_spot = np.asarray([1.0, 2.0])
    next_spot = np.asarray([1.1, 1.8])
    current_options = np.asarray(
        [
            [[1.0e-6, 0.2], [0.3, 0.4]],
            [[0.1, 0.2], [0.3, 0.4]],
        ]
    )
    next_options = np.asarray(
        [
            [[0.003, 0.1], [9.0, 9.0]],
            [[0.2, 0.1], [9.0, 9.0]],
        ]
    )
    returns = _spot_normalized_price_increments(
        current_spot,
        next_spot,
        current_options,
        next_options,
        np.asarray([True, False]),
    )
    expected = np.asarray(
        [
            [0.1, 0.002999, -0.1, 0.0, 0.0],
            [-0.1, 0.05, -0.05, 0.0, 0.0],
        ]
    )
    assert np.allclose(returns, expected)


def test_exact_line_search_update_strictly_decreases_local_loss(tmp_path) -> None:
    generator = np.random.default_rng(41)
    features = generator.standard_normal((32, 3, 2)).astype(np.float32)
    returns = (0.08 * features[..., :1]).astype(np.float32)
    active_mask = np.ones((3, 1), dtype=np.bool_)
    policy, history = _fit_policy(
        features,
        returns,
        active_mask,
        label="synthetic-descent",
        train_config=FrontSpotOceTrainConfig(
            batch_size=16,
            maximum_epochs=1,
            learning_rate=1.0e-2,
            seed=7,
        ),
        output_path=tmp_path / "descent.pt",
        affine_only=True,
    )
    summary, _, _ = _evaluate_policy(
        policy,
        features,
        returns,
        active_mask,
        batch_size=16,
        epsilon=1.0,
    )
    assert history[0]["accepted_step_size"] > 0.0
    assert history[0]["candidate_log_expectation"] < history[0]["log_expectation"]
    assert summary["log_expectation"] < 0.0
    assert summary["expected_pointwise_action_energy"] > 0.0


def test_synthetic_conditional_signal_has_positive_heldout_pair_drift(tmp_path) -> None:
    generator = np.random.default_rng(73)

    def population(pair_count: int) -> tuple[np.ndarray, np.ndarray]:
        pair_state = generator.standard_normal((pair_count, 3, 1))
        pair_noise = 0.01 * generator.standard_normal((pair_count, 3, 1))
        features = np.repeat(pair_state, 2, axis=0)
        mean_return = 0.12 * features
        antithetic_noise = np.stack((pair_noise, -pair_noise), axis=1).reshape(
            2 * pair_count,
            3,
            1,
        )
        return features.astype(np.float32), (mean_return + antithetic_noise).astype(
            np.float32
        )

    training_features, training_returns = population(64)
    heldout_features, heldout_returns = population(64)
    active_mask = np.ones((3, 1), dtype=np.bool_)
    policy, _ = _fit_policy(
        training_features,
        training_returns,
        active_mask,
        label="synthetic-positive-heldout",
        train_config=FrontSpotOceTrainConfig(
            batch_size=64,
            maximum_epochs=12,
            learning_rate=2.0e-2,
            seed=11,
        ),
        output_path=tmp_path / "positive-heldout.pt",
    )
    summary, _, _ = _evaluate_policy(
        policy,
        heldout_features,
        heldout_returns,
        active_mask,
        batch_size=64,
        epsilon=1.0,
    )
    drift = summary["paired_conditional_drift"]
    assert drift["mean_daily_conditional_drift"] > 0.0
    assert drift["confidence_interval_95"][0] > 0.0
