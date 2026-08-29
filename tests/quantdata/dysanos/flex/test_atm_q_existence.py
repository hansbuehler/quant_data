from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from quantdata import env as quantdata_env
from quantdata.dysanos.flex import run_all_state_joint_generic_q_existence as all_state_joint
from quantdata.dysanos.flex.run_all_state_joint_generic_q_existence import (
    _aggregate_state_summaries,
    _block_objective_summary,
    _bootstrap_means,
    _bootstrap_replication_counts,
    _global_objective,
    _joint_path_blocks,
    _make_joint_path_draw,
    _replace_draw_quarter,
    _replication_mean_statistics,
    _tail_excess_loss_and_derivative,
    _train_state_batches,
    _wandb_training_metrics,
)
from quantdata.dysanos.flex.run_all_state_vector_q_existence import (
    AllStateQExistenceLaw,
    _summary_quantiles,
    control_variate_noise_scale_by_state,
    make_common_q_draw,
    normalize_front_components,
    pooled_spot_control_beta_by_state,
    sample_front_components_common,
    spot_control_variate_by_state,
)
from quantdata.dysanos.flex.run_atm_q_existence import (
    _DEFAULT_JOINT_COMPLETION,
    _DEFAULT_OUTPUT_ROOT,
    AtmQExistenceLaw,
    AtmQExistenceLawConfig,
    FrontWeekExactMixtureSampler,
    make_q_existence_batch,
    spot_control_variate,
)
from quantdata.dysanos.flex.run_six_state_joint_generic_q_existence import (
    JointGenericFrontTransport,
    JointGenericTransportConfig,
)
from quantdata.dysanos.flex.run_state_generic_q_existence import (
    GenericConditionalFrontLaw,
    GenericStateQLawConfig,
    make_generic_q_draw,
)
from quantdata.dysanos.flex.run_state_generic_q_existence import (
    _summary as generic_state_summary,
)
from quantdata.dysanos.flex.run_vector_q_existence import (
    spot_control_variate_vector,
)
from quantdata.torch import BaseEnvironment


def test_default_artifacts_follow_quantdata_training_location() -> None:
    training_root = Path(quantdata_env.locations.training.path)

    assert _DEFAULT_JOINT_COMPLETION.is_relative_to(training_root)
    assert _DEFAULT_OUTPUT_ROOT.is_relative_to(training_root)


def test_exact_front_sampler_has_unit_mean_and_finite_mixture_rank() -> None:
    dtype = torch.float64
    sampler = FrontWeekExactMixtureSampler.from_sanos_components(
        support=torch.tensor((0.8, 1.2), dtype=dtype),
        density=torch.tensor((0.5, 0.5), dtype=dtype),
        sqrt_variance=torch.tensor(0.20, dtype=dtype),
    )
    gross, rank = sampler.sample(
        num_paths=24_000,
        generator=torch.Generator().manual_seed(7_891),
    )

    assert torch.all(torch.isfinite(gross) & (gross > 0.0))
    assert torch.all(torch.isfinite(rank))
    assert abs(float(torch.mean(gross)) - 1.0) < 5.0e-4
    assert abs(float(torch.mean(rank))) < 2.0e-2
    assert abs(float(torch.std(rank, correction=0)) - 1.0) < 2.0e-2


def test_q_existence_batch_balances_branches_and_surface_noise() -> None:
    sampler = FrontWeekExactMixtureSampler.from_sanos_components(
        support=torch.tensor((0.9, 1.1), dtype=torch.float64),
        density=torch.tensor((0.5, 0.5), dtype=torch.float64),
        sqrt_variance=torch.tensor(0.12, dtype=torch.float64),
    )
    batch = make_q_existence_batch(
        front_sampler=sampler,
        front_state_size=3,
        spot_contexts=5,
        surface_paths=8,
        seed=19,
    )

    assert batch.num_paths == 40
    assert batch.epsilon.shape == (40, 3)
    assert torch.count_nonzero(batch.branch < 0.0) == 20
    assert torch.count_nonzero(batch.branch > 0.0) == 20
    assert torch.allclose(torch.mean(batch.epsilon, dim=0), torch.zeros(3, dtype=torch.float64))


def test_q_existence_law_is_differentiable_and_has_positive_noise() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = AtmQExistenceLawConfig(
        front_state_size=3,
        minimum_state_noise_std=1.0e-4,
        initial_state_noise_std=1.0e-2,
    )
    law = AtmQExistenceLaw(
        env,
        current_front_state=torch.tensor((1.0, -2.0, 0.5), dtype=torch.float64),
        config=config,
        initialization_seed=23,
    )
    output = law(
        spot_rank=torch.tensor((-1.0, 1.0), dtype=torch.float64),
        branch=torch.tensor((-1.0, 1.0), dtype=torch.float64),
        epsilon=torch.tensor(((0.5, -0.5, 1.0), (-0.5, 0.5, -1.0)), dtype=torch.float64),
    )
    output.square().sum().backward()

    assert output.shape == (2, 3)
    assert float(law.noise_std.detach()) > config.minimum_state_noise_std
    assert all(parameter.grad is not None for parameter in law.parameters())
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in law.parameters())


def test_generic_state_law_starts_as_continuous_affine_gaussian() -> None:
    dtype = torch.float64
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    current = torch.tensor((1.0, -2.0, 0.5), dtype=dtype)
    config = GenericStateQLawConfig(
        front_state_size=3,
        network=GenericStateQLawConfig(front_state_size=3).network,
    )
    law = GenericConditionalFrontLaw(
        env,
        current_front_state=current,
        config=config,
    )
    spot_rank = torch.tensor((-1.0, 1.0), dtype=dtype)
    epsilon = torch.tensor(((0.5, -0.5, 1.0), (-0.5, 0.5, -1.0)), dtype=dtype)
    output = law(spot_rank=spot_rank, epsilon=epsilon)

    torch.testing.assert_close(
        output,
        current + law.noise_std.detach() * epsilon,
    )
    output.square().sum().backward()
    assert all(parameter.grad is not None for parameter in law.parameters())
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in law.parameters())


def test_joint_generic_transport_starts_as_shared_continuous_gaussian() -> None:
    dtype = torch.float64
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    current = torch.tensor(
        ((1.0, -2.0, 0.5), (-0.5, 0.25, 1.5)),
        dtype=dtype,
    )
    config = JointGenericTransportConfig(front_state_size=3)
    law = JointGenericFrontTransport(env, config=config)
    spot_rank = torch.tensor(
        ((-1.0, 1.0), (-0.25, 0.75)),
        dtype=dtype,
    )
    epsilon = torch.tensor(((0.5, -0.5, 1.0), (-0.5, 0.5, -1.0)), dtype=dtype)
    output = law(
        current_front_states=current,
        spot_rank=spot_rank,
        epsilon=epsilon,
    )

    expected = current[:, None] + config.base_noise_std * epsilon[None]
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert output.shape == (2, 2, 3)
    assert all(parameter.grad is not None for parameter in law.parameters())
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in law.parameters())


def test_all_state_joint_summary_preserves_worst_state_identity() -> None:
    summary = _aggregate_state_summaries(
        [
            {
                "state_index": 17,
                "raw_mean_z": [0.5, -1.0],
                "noise_scaled_adjusted_mean_z": [0.25, -0.75],
            },
            {
                "state_index": 29,
                "raw_mean_z": [-1.5, 0.1],
                "noise_scaled_adjusted_mean_z": [-0.2, 1.25],
            },
        ],
        z_target=2.0,
        paths_per_state=24_000,
    )

    assert summary["worst_raw_state_index"] == 29
    assert summary["worst_raw_instrument_index"] == 0
    assert summary["worst_state_index"] == 29
    assert summary["worst_instrument_index"] == 1
    assert summary["passes_target"] is True


def test_all_state_draw_refresh_replaces_exactly_one_context_quarter() -> None:
    original = make_generic_q_draw(
        front_state_size=3,
        spot_contexts=8,
        surface_paths=4,
        seed=101,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    replacement = make_generic_q_draw(
        front_state_size=3,
        spot_contexts=8,
        surface_paths=4,
        seed=102,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    refreshed = _replace_draw_quarter(original, replacement, quarter=2)

    torch.testing.assert_close(refreshed.component_uniform[:4], original.component_uniform[:4])
    torch.testing.assert_close(refreshed.component_uniform[4:6], replacement.component_uniform[4:6])
    torch.testing.assert_close(refreshed.component_uniform[6:], original.component_uniform[6:])
    torch.testing.assert_close(refreshed.epsilon[:16], original.epsilon[:16])
    torch.testing.assert_close(refreshed.epsilon[16:24], replacement.epsilon[16:24])
    torch.testing.assert_close(refreshed.epsilon[24:], original.epsilon[24:])


def test_all_state_block_bootstrap_reestimates_beta() -> None:
    values = {
        "path_sums_by_replication": torch.tensor([[[0.0]], [[4.0]]]),
        "spot_sums_by_replication": torch.tensor([[-1.0], [1.0]]),
        "spot_square_sums_by_replication": torch.tensor([[1.0], [1.0]]),
        "product_sums_by_replication": torch.tensor([[[1.0]], [[3.0]]]),
        "paths_per_replication": torch.tensor([2, 2]),
    }
    counts = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        dtype=torch.float64,
    )

    raw, adjusted = _bootstrap_means(values, counts)

    torch.testing.assert_close(raw[:, 0, 0], torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(adjusted[:, 0, 0], torch.ones(3, dtype=torch.float64))


def test_all_state_bootstrap_counts_preserve_replication_count() -> None:
    counts = _bootstrap_replication_counts(
        replications=10,
        samples=25,
        seed=103,
        device=torch.device("cpu"),
    )

    assert counts.shape == (25, 10)
    torch.testing.assert_close(
        torch.sum(counts, dim=1), torch.full((25,), 10.0, dtype=torch.float64)
    )


def test_joint_path_draw_uses_one_distinct_spot_and_surface_state_per_path() -> None:
    dtype = torch.float64
    draw = _make_joint_path_draw(
        front_state_size=3,
        num_paths=200,
        seed=107,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    support, density, variance = normalize_front_components(
        torch.tensor(((0.8, 1.2),), dtype=dtype),
        torch.tensor(((0.5, 0.5),), dtype=dtype),
        torch.tensor((0.20,), dtype=dtype),
    )
    gross, rank = sample_front_components_common(
        support=support,
        density=density,
        sqrt_variance=variance,
        draw=draw,
    )

    assert draw.spot_contexts == 200
    assert draw.surface_paths == 1
    assert draw.num_paths == 200
    assert draw.epsilon.shape == (200, 3)
    assert gross.shape == rank.shape == (1, 200)
    assert torch.unique(gross[0]).numel() == 200
    assert torch.unique(draw.epsilon, dim=0).shape[0] == 200


def test_joint_path_blocks_partition_requested_paths_without_overlap() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    blocks = _joint_path_blocks(
        env=env,
        front_state_size=2,
        total_paths=100,
        blocks=10,
        seed=109,
    )

    assert len(blocks) == 10
    assert all(block.num_paths == 10 for block in blocks)
    uniforms = torch.cat([block.backbone_uniform for block in blocks])
    assert torch.unique(uniforms).numel() == 100


def test_replication_mean_statistics_reports_ten_means_and_standard_error() -> None:
    values = torch.arange(20.0, dtype=torch.float64).reshape(10, 2) / 1.0e4

    summary = _replication_mean_statistics(values)

    assert len(summary["block_means_bps_spot"]) == 10
    torch.testing.assert_close(
        torch.tensor(summary["mean_bps_spot"], dtype=torch.float64),
        torch.mean(1.0e4 * values, dim=0),
    )
    torch.testing.assert_close(
        torch.tensor(summary["median_bps_spot"], dtype=torch.float64),
        torch.quantile(1.0e4 * values, 0.5, dim=0),
    )
    torch.testing.assert_close(
        torch.tensor(summary["mc_standard_error_bps_spot"], dtype=torch.float64),
        torch.std(1.0e4 * values, dim=0, correction=1) / math.sqrt(10),
    )


def test_block_objective_reports_each_non_overlapping_block() -> None:
    values = {
        "path_sums_by_replication": torch.tensor([[[0.0]], [[2.0]]]),
        "path_square_sums_by_replication": torch.tensor([[[2.0]], [[4.0]]]),
        "spot_sums_by_replication": torch.zeros((2, 1)),
        "spot_square_sums_by_replication": torch.full((2, 1), 2.0),
        "paths_per_replication": torch.tensor((2, 2)),
        "beta": torch.zeros((1, 1)),
    }

    summary = _block_objective_summary(
        [values],
        z_target=0.5,
        inner_tail_fraction=1.0,
        outer_tail_fraction=1.0,
        inner_tail_weight=1.0,
    )

    assert summary["num_blocks"] == 2
    assert summary["paths_per_block"] == [2, 2]
    torch.testing.assert_close(
        torch.tensor(summary["global_loss"]["values"], dtype=torch.float64),
        torch.tensor((0.0, 0.25), dtype=torch.float64),
    )
    assert math.isclose(summary["global_loss"]["mean"], 0.125)
    assert math.isclose(summary["global_loss"]["median"], 0.125)


def test_all_state_global_objective_gradient_is_invariant_to_state_batches() -> None:
    adjusted = torch.tensor(
        (
            ((-0.3,), (0.2,), (0.5,)),
            ((-0.1,), (0.4,), (0.7,)),
            ((0.1,), (0.6,), (0.9,)),
        ),
        dtype=torch.float64,
    )
    raw = adjusted + torch.tensor((-0.1, 0.0, 0.1), dtype=torch.float64)[:, None, None]
    spot = torch.tensor(
        ((-0.2, -0.1, 0.0), (0.0, 0.1, 0.2), (0.2, 0.3, 0.4)),
        dtype=torch.float64,
    )
    beta = torch.tensor(((0.5,), (-0.25,), (0.75,)), dtype=torch.float64)

    combined = {
        "adjusted_replication_means": adjusted,
        "raw_replication_means": raw,
        "spot_replication_means": spot,
        "beta": beta,
    }
    split = [
        {key: value[:, :2] if key != "beta" else value[:2] for key, value in combined.items()},
        {key: value[:, 2:] if key != "beta" else value[2:] for key, value in combined.items()},
    ]

    combined_loss, combined_gradients, combined_metrics = _global_objective(
        [combined],
        z_target=0.1,
        inner_tail_fraction=0.34,
        outer_tail_fraction=1.0,
        inner_tail_weight=0.5,
        dtype=torch.float64,
    )
    split_loss, split_gradients, split_metrics = _global_objective(
        split,
        z_target=0.1,
        inner_tail_fraction=0.34,
        outer_tail_fraction=1.0,
        inner_tail_weight=0.5,
        dtype=torch.float64,
    )
    adjusted_mean = torch.mean(adjusted, dim=0).detach().requires_grad_()
    raw_se = torch.std(raw, dim=0, correction=1) / math.sqrt(len(raw))
    spot_se = torch.std(spot, dim=0, correction=1) / math.sqrt(len(spot))
    noise_scale = torch.maximum(torch.abs(beta) * spot_se[:, None], raw_se)
    scaled = adjusted_mean / noise_scale
    expected_loss, derivative = _tail_excess_loss_and_derivative(
        scaled.reshape(-1),
        z_target=0.1,
        inner_tail_fraction=0.34,
        outer_tail_fraction=1.0,
        inner_tail_weight=0.5,
    )
    expected_gradient = (derivative.reshape_as(scaled) / noise_scale).detach()

    torch.testing.assert_close(split_loss, combined_loss)
    torch.testing.assert_close(combined_loss, expected_loss)
    torch.testing.assert_close(torch.cat(split_gradients), combined_gradients[0])
    torch.testing.assert_close(combined_gradients[0], expected_gradient)
    assert all(
        math.isclose(split_metrics[key], combined_metrics[key], rel_tol=1.0e-12)
        for key in combined_metrics
    )


def test_tail_excess_objective_spreads_gradient_across_nested_tails() -> None:
    scaled = torch.tensor((5.0, -4.0, 3.0, 1.0, 0.5), dtype=torch.float64)

    loss, derivative = _tail_excess_loss_and_derivative(
        scaled,
        z_target=2.0,
        inner_tail_fraction=0.2,
        outer_tail_fraction=0.6,
        inner_tail_weight=0.5,
    )

    assert float(loss) > 0.0
    assert torch.all(derivative[:3] != 0.0)
    assert torch.all(derivative[3:] == 0.0)
    assert derivative[0] > 0.0
    assert derivative[1] < 0.0


def test_all_state_training_accumulates_one_optimizer_step(
    monkeypatch: Any,
) -> None:
    dtype = torch.float64
    law = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
    law.weight.data.fill_(0.25)
    optimizer = torch.optim.Adam(law.parameters(), lr=1.0e-2)
    draws = [0, 1, 2]

    def evaluate_joint(**kwargs: Any) -> dict[str, torch.Tensor]:
        states = kwargs["current_states"]
        replication_means = torch.stack(
            [law(states) + offset for offset in (-0.1, 0.0, 0.1)]
        )
        return {
            "adjusted_replication_means": replication_means,
            "raw_replication_means": replication_means,
            "spot_replication_means": torch.tensor(
                ((-0.1,), (0.0,), (0.1,)), dtype=dtype
            ).expand(-1, len(states)),
            "beta": torch.zeros_like(replication_means[0]),
        }

    def roll_returns(**kwargs: Any) -> tuple[torch.Tensor, None, torch.Tensor]:
        states = kwargs["current_states"]
        draw = kwargs["draw"]
        returns = (law(states) + (-0.1, 0.0, 0.1)[draw])[:, None, :]
        gross = torch.ones((len(states), 1), dtype=dtype)
        return returns, None, gross

    monkeypatch.setattr(all_state_joint, "_evaluate_joint", evaluate_joint)
    monkeypatch.setattr(all_state_joint, "joint_roll_returns", roll_returns)
    states = torch.tensor(((1.0,), (2.0,), (3.0,)), dtype=dtype)
    placeholder = torch.zeros_like(states)

    metrics = _train_state_batches(
        model=None,
        law=law,  # type: ignore[arg-type]
        optimizer=optimizer,
        current_states=states,
        current_w2_calls=placeholder,
        normalized_strikes=torch.ones(1, dtype=dtype),
        option_is_call=torch.ones(1, dtype=torch.bool),
        support=placeholder,
        density=placeholder,
        sqrt_variance=placeholder[:, 0],
        draws=draws,  # type: ignore[arg-type]
        z_target=0.1,
        inner_tail_fraction=0.34,
        outer_tail_fraction=1.0,
        inner_tail_weight=0.5,
        order=torch.tensor((2, 0, 1)),
        state_batch_size=2,
    )

    assert int(optimizer.state[law.weight]["step"]) == 1
    assert metrics["gradient_norm"] > 0.0
    assert float(law.weight.detach()) != 0.25


def test_generic_draw_has_nonduplicated_antithetic_surface_noise() -> None:
    draw = make_generic_q_draw(
        front_state_size=3,
        spot_contexts=5,
        surface_paths=8,
        seed=31,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    epsilon = draw.epsilon.reshape(5, 8, 3)

    assert draw.num_paths == 40
    torch.testing.assert_close(epsilon[:, :4] + epsilon[:, 4:], torch.zeros_like(epsilon[:, :4]))
    assert torch.count_nonzero(draw.branch) == 0


def test_generic_summary_anchors_control_variate_at_exact_spot_expectation() -> None:
    dtype = torch.float64
    spot_means = torch.tensor((-0.02, -0.01, 0.01, 0.02), dtype=dtype)
    beta = torch.tensor((2.0, -3.0), dtype=dtype)
    intercept = torch.tensor((0.001, -0.002), dtype=dtype)
    raw_means = intercept + spot_means[:, None] * beta
    adjusted_means = raw_means - spot_means[:, None] * beta
    summary = generic_state_summary(
        {
            "raw_replication_means": raw_means,
            "spot_replication_means": spot_means,
            "adjusted_replication_means": adjusted_means,
            "beta": beta,
            "path_sum": 4.0 * intercept,
            "path_square_sum": 4.0 * intercept.square(),
            "cap_mean_square": torch.tensor(0.0, dtype=dtype),
            "num_paths": torch.tensor(4),
        }
    )

    torch.testing.assert_close(
        torch.tensor(summary["adjusted_mean_bps_spot"], dtype=dtype),
        1.0e4 * intercept,
    )


def test_raw_z_summary_reports_mean_of_worst_decile() -> None:
    summary = _summary_quantiles(torch.arange(1.0, 21.0))

    assert summary["maximum"] == 20.0
    assert summary["worst_10_percent_mean"] == 19.5


def test_spot_control_variate_removes_exact_linear_spot_component() -> None:
    gross = torch.tensor((0.8, 0.9, 1.1, 1.2), dtype=torch.float64)
    returns = 0.02 + 3.0 * (gross - 1.0)
    adjusted, mean, beta = spot_control_variate(returns, gross)

    assert torch.allclose(adjusted, torch.full_like(adjusted, 0.02))
    assert torch.allclose(mean, torch.tensor(0.02, dtype=torch.float64))
    assert torch.allclose(beta, torch.tensor(3.0, dtype=torch.float64))


def test_vector_spot_control_variate_removes_each_linear_spot_component() -> None:
    gross = torch.tensor((0.8, 0.9, 1.1, 1.2), dtype=torch.float64)
    intercept = torch.tensor((0.02, -0.03), dtype=torch.float64)
    loading = torch.tensor((3.0, -2.0), dtype=torch.float64)
    returns = intercept + (gross - 1.0)[:, None] * loading
    adjusted, mean, beta = spot_control_variate_vector(returns, gross)

    assert torch.allclose(adjusted, intercept.expand_as(adjusted))
    assert torch.allclose(mean, intercept)
    assert torch.allclose(beta, loading)


def test_common_draw_is_reused_exactly_across_identical_front_marginals() -> None:
    dtype = torch.float64
    support, density, variance = normalize_front_components(
        torch.tensor(((0.8, 1.2), (0.8, 1.2)), dtype=dtype),
        torch.tensor(((0.5, 0.5), (0.5, 0.5)), dtype=dtype),
        torch.tensor((0.20, 0.20), dtype=dtype),
    )
    draw = make_common_q_draw(
        front_state_size=3,
        spot_contexts=125,
        surface_paths=8,
        seed=101,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    gross, rank = sample_front_components_common(
        support=support,
        density=density,
        sqrt_variance=variance,
        draw=draw,
    )

    assert gross.shape == (2, 1_000)
    assert rank.shape == (2, 1_000)
    assert torch.equal(gross[0], gross[1])
    assert torch.equal(rank[0], rank[1])
    assert torch.allclose(torch.mean(draw.epsilon, dim=0), torch.zeros(3, dtype=dtype))


def test_all_state_law_and_control_variate_keep_origins_separate() -> None:
    dtype = torch.float64
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = AtmQExistenceLawConfig(front_state_size=2)
    law = AllStateQExistenceLaw(
        env,
        current_front_states=torch.tensor(((1.0, 2.0), (3.0, 4.0)), dtype=dtype),
        config=config,
        initialization_seed=103,
    )
    spot_rank = torch.tensor(((-1.0, 1.0), (-1.0, 1.0)), dtype=dtype)
    output = law(
        origin_indices=torch.tensor((0, 1)),
        spot_rank=spot_rank,
        branch=torch.tensor((-1.0, 1.0), dtype=dtype),
        epsilon=torch.tensor(((0.5, -0.5), (-0.5, 0.5)), dtype=dtype),
    )
    gross = torch.tensor(((0.8, 1.2), (0.9, 1.1)), dtype=dtype)
    intercept = torch.tensor(((0.02, -0.03), (0.04, 0.01)), dtype=dtype)
    loading = torch.tensor(((3.0, -2.0), (1.5, 0.5)), dtype=dtype)
    returns = intercept[:, None] + (gross - 1.0)[..., None] * loading[:, None]
    adjusted, mean, beta = spot_control_variate_by_state(returns, gross)
    output.square().sum().backward()

    assert output.shape == (2, 2, 2)
    assert torch.allclose(adjusted, intercept[:, None].expand_as(adjusted))
    assert torch.allclose(mean, intercept)
    assert torch.allclose(beta, loading)
    assert all(parameter.grad is not None for parameter in law.parameters())


def test_statewise_beta_and_mean_are_dataset_local() -> None:
    dtype = torch.float64
    training_spot = torch.tensor(
        ((-0.20, -0.10, 0.10, 0.20), (-0.30, -0.10, 0.20, 0.40)),
        dtype=dtype,
    )
    intercept = torch.tensor(((0.02, -0.03), (0.04, 0.01)), dtype=dtype)
    loading = torch.tensor(((3.0, -2.0), (1.5, 0.5)), dtype=dtype)
    training_returns = intercept[:, None] + training_spot[..., None] * loading[:, None]
    num_paths = training_spot.shape[1]
    training_beta = pooled_spot_control_beta_by_state(
        path_sum=torch.sum(training_returns, dim=1),
        spot_sum=torch.sum(training_spot, dim=1),
        spot_square_sum=torch.sum(training_spot.square(), dim=1),
        spot_return_product_sum=torch.sum(
            training_spot[..., None] * training_returns,
            dim=1,
        ),
        num_paths=num_paths,
    )

    validation_spot = torch.tensor(
        ((-0.10, 0.10, 0.30), (-0.20, 0.10, 0.40)),
        dtype=dtype,
    )
    validation_returns = intercept[:, None] + validation_spot[..., None] * loading[:, None]
    validation_returns = validation_returns + torch.tensor(
        (
            ((0.01, 0.00), (-0.02, 0.01), (0.03, -0.01)),
            ((-0.01, 0.02), (0.02, -0.01), (0.00, 0.01)),
        ),
        dtype=dtype,
    )
    validation_beta = pooled_spot_control_beta_by_state(
        path_sum=torch.sum(validation_returns, dim=1),
        spot_sum=torch.sum(validation_spot, dim=1),
        spot_square_sum=torch.sum(validation_spot.square(), dim=1),
        spot_return_product_sum=torch.sum(
            validation_spot[..., None] * validation_returns,
            dim=1,
        ),
        num_paths=validation_spot.shape[1],
    )
    adjusted, adjusted_mean, returned_beta = spot_control_variate_by_state(
        validation_returns,
        validation_spot + 1.0,
        beta=validation_beta,
    )
    expected_mean = torch.mean(validation_returns, dim=1) - validation_beta * torch.mean(
        validation_spot,
        dim=1,
        keepdim=True,
    )

    assert torch.allclose(training_beta, loading)
    assert not torch.allclose(validation_beta, training_beta)
    assert torch.allclose(returned_beta, validation_beta)
    assert torch.allclose(adjusted_mean, expected_mean)


def test_control_variate_reporting_noise_uses_larger_raw_component() -> None:
    beta = torch.tensor(((-2.0, 0.5), (1.0, -3.0)), dtype=torch.float64)
    spot_se = torch.tensor((0.10, 0.20), dtype=torch.float64)
    raw_se = torch.tensor(((0.30, 0.04), (0.10, 0.80)), dtype=torch.float64)

    noise_scale = control_variate_noise_scale_by_state(
        beta=beta,
        spot_mc_standard_error=spot_se,
        raw_mc_standard_error=raw_se,
    )

    assert torch.allclose(
        noise_scale,
        torch.tensor(((0.30, 0.05), (0.20, 0.80)), dtype=torch.float64),
    )


def test_wandb_metrics_include_validation_blocks_and_stable_names() -> None:
    metrics = _wandb_training_metrics(
        {
            "sweep": 12.0,
            "fit_global_loss": 3.5,
            "validation_block_loss_mean": 2.5,
            "validation_block_loss_median": 2.25,
            "validation_block_loss_mc_standard_error": 0.125,
            "validation_block_global_losses": [float(index) for index in range(10)],
            "learning_rate": 1.0e-4,
        }
    )

    assert metrics["sweep"] == 12
    assert metrics["training/global_loss"] == 3.5
    assert metrics["validation/block_loss_mean"] == 2.5
    assert metrics["validation/block_loss_median"] == 2.25
    assert metrics["validation/block_loss_mc_se"] == 0.125
    assert metrics["validation/block_loss_00"] == 0.0
    assert metrics["validation/block_loss_09"] == 9.0
    assert metrics["validation/block_loss_q10"] == 0.9
    assert metrics["validation/block_loss_q25"] == 2.25
    assert metrics["validation/block_loss_q75"] == 6.75
    assert metrics["validation/block_loss_q90"] == 8.1
    assert metrics["optimizer/learning_rate"] == 1.0e-4
