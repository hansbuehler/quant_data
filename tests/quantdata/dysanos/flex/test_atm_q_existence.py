from __future__ import annotations

from pathlib import Path

import torch

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.run_atm_q_existence import (
    _DEFAULT_JOINT_COMPLETION,
    _DEFAULT_OUTPUT_ROOT,
    AtmQExistenceLaw,
    AtmQExistenceLawConfig,
    FrontWeekExactMixtureSampler,
    make_q_existence_batch,
    spot_control_variate,
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
