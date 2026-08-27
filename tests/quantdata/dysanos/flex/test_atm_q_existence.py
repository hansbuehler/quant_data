from __future__ import annotations

from pathlib import Path

import torch

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.run_all_state_joint_generic_q_existence import (
    _aggregate_state_summaries,
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
