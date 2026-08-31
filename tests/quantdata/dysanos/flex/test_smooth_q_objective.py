from __future__ import annotations

import pytest
import torch

from quantdata.dysanos.flex.run_all_state_vector_smooth_q_existence import (
    _make_iid_statewise_draw,
)
from quantdata.dysanos.flex.smooth_q_objective import (
    SmoothPriceObjectiveConfig,
    smooth_price_objective,
    studentized_block_sign_flip_test,
    validation_confirmation_consistency,
)


def test_smooth_price_objective_derivative_matches_finite_difference() -> None:
    means = torch.tensor(
        [
            [[0.010, -0.006], [0.004, 0.008]],
            [[0.012, -0.004], [0.006, 0.007]],
            [[0.008, -0.008], [0.005, 0.009]],
        ],
        dtype=torch.float64,
    )
    prices = torch.tensor(
        [[0.20, 0.10], [0.15, 0.25]],
        dtype=torch.float64,
    )
    config = SmoothPriceObjectiveConfig(
        temperature=0.02,
        robust_block_weight=0.25,
        block_weight_decay=0.5,
    )
    loss, derivative, metrics = smooth_price_objective(
        means,
        prices,
        config=config,
        derivative=True,
    )
    assert derivative is not None
    assert metrics["num_constraints"] == 4
    assert metrics["option_price_floor_count"] == 0
    assert 1.0 <= metrics["constraint_weight_effective_sample_size"] <= 4.0

    epsilon = 1.0e-6
    plus = means.clone()
    minus = means.clone()
    plus[1, 0, 1] += epsilon
    minus[1, 0, 1] -= epsilon
    plus_loss, _, _ = smooth_price_objective(
        plus,
        prices,
        config=config,
        derivative=False,
    )
    minus_loss, _, _ = smooth_price_objective(
        minus,
        prices,
        config=config,
        derivative=False,
    )
    finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon)
    assert float(loss) > 0.0
    assert float(derivative[1, 0, 1]) == pytest.approx(
        float(finite_difference),
        rel=2.0e-6,
        abs=2.0e-8,
    )


def test_smooth_price_objective_geometrically_weights_sorted_blocks() -> None:
    means = torch.tensor(
        [
            [[0.1, 0.0]],
            [[0.3, 0.0]],
            [[0.2, 0.0]],
        ],
        dtype=torch.float64,
    )
    prices = torch.ones((1, 2), dtype=torch.float64)
    config = SmoothPriceObjectiveConfig(
        temperature=0.1,
        robust_block_weight=1.0,
        block_weight_decay=0.5,
    )
    loss, _, metrics = smooth_price_objective(
        means,
        prices,
        config=config,
        derivative=False,
    )
    block_losses = torch.tensor(metrics["block_loss"]["values"])
    expected_robust = torch.sum(
        torch.tensor([4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0])
        * torch.sort(block_losses, descending=True).values
    )
    assert metrics["block_order_worst_first"] == [1, 2, 0]
    assert metrics["robust_block_loss"] == pytest.approx(float(expected_robust))
    assert float(loss) == pytest.approx(
        metrics["full_loss"] + metrics["robust_block_loss"]
    )


def test_block_sign_flip_null_is_joint_and_exact_for_zero_observed_mean() -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0]],
            [[-1.0, -2.0]],
            [[0.5, 1.0]],
            [[-0.5, -1.0]],
        ],
        dtype=torch.float64,
    )
    result = studentized_block_sign_flip_test(values)
    assert result["num_sign_patterns"] == 16
    assert result["same_block_sign_across_all_constraints"] is True
    assert result["maximum_abs_t"] == pytest.approx(0.0)
    assert result["global_maximum_p_value"] == pytest.approx(1.0)
    assert result["global_tail_mean_p_value"] == pytest.approx(1.0)


def test_block_sign_flip_exact_p_value_includes_observed_pattern() -> None:
    values = torch.tensor(
        [
            [[1.2000001, -0.4]],
            [[0.7, 0.2]],
            [[-0.1, 0.8]],
            [[0.3, -0.2]],
        ],
        dtype=torch.float64,
    )
    result = studentized_block_sign_flip_test(values)
    minimum_exact_p_value = 1.0 / result["num_sign_patterns"]
    assert result["global_maximum_p_value"] >= minimum_exact_p_value
    assert result["global_tail_mean_p_value"] >= minimum_exact_p_value


def test_block_sign_flip_exact_p_value_survives_huge_studentization() -> None:
    shared = torch.tensor([1.0, -0.2, 0.3], dtype=torch.float64)
    values = torch.stack(
        [shared + index * 1.0e-10 for index in range(10)],
        dim=0,
    )[:, None]
    result = studentized_block_sign_flip_test(values)
    assert result["maximum_abs_t"] > 1.0e8
    minimum_absolute_p_value = 2.0 / result["num_sign_patterns"]
    assert result["global_maximum_p_value"] >= minimum_absolute_p_value
    assert result["global_tail_mean_p_value"] >= minimum_absolute_p_value


def test_validation_confirmation_consistency_is_zero_for_identical_blocks() -> None:
    values = torch.tensor(
        [
            [[0.01, -0.02]],
            [[0.02, -0.01]],
            [[-0.01, 0.01]],
        ],
        dtype=torch.float64,
    )
    result = validation_confirmation_consistency(values, values.clone())
    assert result["maximum_abs_difference_t"] == pytest.approx(0.0)
    assert result["worst_10_percent_mean_abs_difference_t"] == pytest.approx(0.0)


def test_iid_statewise_draw_is_one_to_one_balanced_and_reproducible() -> None:
    first = _make_iid_statewise_draw(
        front_state_size=3,
        num_paths=10,
        seed=17,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    second = _make_iid_statewise_draw(
        front_state_size=3,
        num_paths=10,
        seed=17,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert first.spot_contexts == 10
    assert first.surface_paths == 1
    assert first.num_paths == 10
    assert first.epsilon.shape == (10, 3)
    assert int(torch.sum(first.branch < 0.0)) == 5
    assert int(torch.sum(first.branch > 0.0)) == 5
    assert torch.equal(first.component_uniform, second.component_uniform)
    assert torch.equal(first.backbone_uniform, second.backbone_uniform)
    assert torch.equal(first.branch, second.branch)
    assert torch.equal(first.epsilon, second.epsilon)
