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
    _active_mask,
    _artifact_stem,
    _evaluate_policy,
    _fit_policy,
    _instrument_labels,
    _population_cache_schema,
    _spot_normalized_price_increments,
    unconditional_spot_scaled_martingale_audit,
)
from quantdata.sdlv.daily.front_spot.run_surface_oce import (
    POLICY_SCHEMA,
    SurfacePopulation,
    _checkpoint_metadata_mismatches,
    _dynamic_active_mask,
    _write_returns,
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


def test_policy_accepts_predictable_pathwise_instrument_masks() -> None:
    policy = _policy()
    features = torch.randn(2, 3, 3, dtype=torch.float64)
    active = torch.tensor(
        [
            [[True, False], [True, True], [False, False]],
            [[False, True], [True, False], [True, True]],
        ],
        dtype=torch.bool,
    )
    with torch.no_grad():
        policy.affine.bias.fill_(2.0)
    action = policy(features, active)
    assert action.shape == active.shape
    assert torch.all(action[active] == 2.0)
    assert torch.all(action[~active] == 0.0)


def test_surface_policy_checkpoint_allows_extended_training_schedule() -> None:
    checkpoint = {
        "schema": POLICY_SCHEMA,
        "policy_code_version": FrontSpotOcePolicyConfig.version.unique_id48,
        "train_config": {"maximum_epochs": 30},
        "fold_start": 0,
        "fold_stop": 50_000,
        "affine_only": True,
    }
    assert not _checkpoint_metadata_mismatches(
        checkpoint,
        start=0,
        stop=50_000,
        affine_only=True,
    )
    checkpoint["fold_stop"] = 49_998
    assert _checkpoint_metadata_mismatches(
        checkpoint,
        start=0,
        stop=50_000,
        affine_only=True,
    ) == ["fold_stop"]


def test_full_surface_mask_is_pathwise_and_expires_strips() -> None:
    features = np.zeros((2, 25, 2), dtype=np.float32)
    features[1, :, 1] = math.log(1.1)
    population = SurfacePopulation(
        features=features,
        returns=np.empty((2, 25, 76), dtype=np.float32),
        strikes=np.asarray([0.9, 1.0, 1.1]),
        factor_rank=1,
        nms_annualized_atm_volatility=0.2,
        nms_lower_bound=-2.0,
        nms_upper_bound=1.0,
        initial_date="2025-08-29",
        generation_seconds=0.0,
    )
    mask = _dynamic_active_mask(features, population)
    assert mask.shape == (2, 25, 76)
    assert np.all(mask[..., 0])
    assert not np.array_equal(mask[0, 0], mask[1, 0])
    assert not np.any(mask[:, 24, 1 : 1 + 24 * 3])
    assert np.any(mask[:, 24, 1 + 24 * 3 :])


def test_full_surface_returns_apply_pre_step_eligibility_and_spot_scaling() -> None:
    target = np.empty((2, 5), dtype=np.float32)
    current_spot = np.asarray([1.0, 2.0])
    next_spot = np.asarray([1.1, 1.8])
    current_options = np.asarray([[[0.2, 0.3], [0.4, 0.5]]] * 2)
    next_options = current_options + 0.1
    eligible = np.asarray(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    _write_returns(
        target,
        current_spot=current_spot,
        next_spot=next_spot,
        current_options=current_options,
        next_options=next_options,
        eligible_options=eligible,
    )
    assert np.allclose(target[0], [0.1, 0.1, 0.0, 0.0, 0.1])
    assert np.allclose(target[1], [-0.1, 0.0, 0.05, 0.05, 0.0])


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


def test_model_artifacts_distinguish_one_and_eight_substeps() -> None:
    assert _artifact_stem(8, 1) == "rank8_one_step_local_oce"
    assert _artifact_stem(8, 8) == "rank8_eight_step_local_oce"
    assert _population_cache_schema(8, 1) != _population_cache_schema(8, 8)
    assert _artifact_stem(8, 1, "all_nodes") == "rank8_one_step_all_nodes_local_oce"
    assert _population_cache_schema(8, 1, "all_nodes") != _population_cache_schema(
        8,
        1,
    )


def test_all_node_universe_has_one_spot_and_four_full_strips() -> None:
    strike_count = 200
    labels = _instrument_labels(strike_count, instrument_universe="all_nodes")
    active_mask = _active_mask(strike_count)
    assert len(labels) == 1 + 4 * strike_count
    assert labels[0] == "spot"
    assert labels[1] == "5D interior node 0 call"
    assert labels[-1] == "20D interior node 199 call"
    assert active_mask.shape == (20, 1 + 4 * strike_count)
    assert int(np.sum(active_mask)) == 20 + strike_count * (5 + 10 + 15 + 20)


def test_unconditional_audit_uses_antithetic_pair_sampling_units() -> None:
    returns = np.asarray(
        [
            [[3.0, 2.0]],
            [[-1.0, 0.0]],
            [[5.0, 4.0]],
            [[-1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    audit = unconditional_spot_scaled_martingale_audit(
        returns,
        np.ones((1, 2), dtype=np.bool_),
        ("spot", "option"),
    )
    assert audit["antithetic_pair_count"] == 2
    assert audit["spot"]["top_absolute_z_coordinates"][0]["mean"] == pytest.approx(
        1.5
    )
    assert audit["spot"]["top_absolute_z_coordinates"][0][
        "pair_based_standard_error"
    ] == pytest.approx(0.5)
    assert audit["spot"]["max_absolute_z"] == pytest.approx(3.0)


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
