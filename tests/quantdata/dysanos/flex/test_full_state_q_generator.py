"""Tests for the unrestricted state-local FLEX Q generator."""

from __future__ import annotations

import torch

from quantdata.dysanos.flex.full_state_q_generator import (
    FullStateConditionalGenerator,
    FullStateGeneratorConfig,
    JointFullStateConditionalGenerator,
    JointFullStateGeneratorConfig,
    JointUncompressedFrontConditionalGenerator,
    JointUncompressedFrontGeneratorConfig,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_full_generator_has_the_expected_shape_and_base_noise() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = FullStateGeneratorConfig(
        front_state_size=3,
        base_noise_std=0.02,
        maximum_increment=4.0,
        network=FeedForwardConfig(activation="SiLU", num_depth=2, num_width=8),
    )
    current = torch.tensor((1.0, -2.0, 0.5), dtype=torch.float64)
    law = FullStateConditionalGenerator(
        env,
        current_front_state=current,
        config=config,
    )
    epsilon = torch.tensor(
        ((0.5, -1.0, 2.0), (-0.25, 0.75, -1.5)),
        dtype=torch.float64,
    )
    generated = law(
        current_front_states=current[None],
        spot_rank=torch.tensor(((0.1, -0.2),), dtype=torch.float64),
        epsilon=epsilon,
    )
    expected_increment = config.maximum_increment * torch.tanh(
        config.base_noise_std * epsilon / config.maximum_increment
    )
    assert generated.shape == (1, 2, 3)
    torch.testing.assert_close(generated[0], current + expected_increment)


def test_full_generator_allows_dense_cross_coordinate_gradients() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = FullStateGeneratorConfig(
        front_state_size=3,
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5),
    )
    current = torch.zeros(3, dtype=torch.float64)
    law = FullStateConditionalGenerator(
        env,
        current_front_state=current,
        config=config,
    )
    with torch.no_grad():
        law.network.I0.weight.fill_(0.2)
        law.network.I0.bias.fill_(0.1)
        law.network.In[-1].weight.fill_(0.3)
        law.network.In[-1].bias.zero_()
    epsilon = torch.tensor(((0.2, -0.1, 0.3),), dtype=torch.float64, requires_grad=True)
    output = law(
        current_front_states=current[None],
        spot_rank=torch.tensor(((0.4,),), dtype=torch.float64),
        epsilon=epsilon,
    )
    jacobian_rows = []
    for coordinate in range(3):
        gradient = torch.autograd.grad(
            output[0, 0, coordinate],
            epsilon,
            retain_graph=True,
        )[0]
        jacobian_rows.append(gradient[0])
    jacobian = torch.stack(jacobian_rows)
    assert torch.count_nonzero(jacobian - torch.diag(torch.diagonal(jacobian))) == 6


def test_joint_full_generator_starts_from_state_local_base_noise() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = JointFullStateGeneratorConfig(
        front_state_size=3,
        base_noise_std=0.02,
        maximum_increment=4.0,
        network=FeedForwardConfig(activation="SiLU", num_depth=2, num_width=8),
    )
    current = torch.tensor(
        ((1.0, -2.0, 0.5), (-0.5, 1.0, 2.0)),
        dtype=torch.float64,
    )
    law = JointFullStateConditionalGenerator(
        env,
        current_front_states=current,
        config=config,
    )
    epsilon = torch.tensor(
        ((0.5, -1.0, 2.0), (-0.25, 0.75, -1.5)),
        dtype=torch.float64,
    )
    generated = law(
        current_front_states=current,
        spot_rank=torch.tensor(((0.1, -0.2), (0.3, -0.4)), dtype=torch.float64),
        epsilon=epsilon,
    )
    expected_increment = config.maximum_increment * torch.tanh(
        config.base_noise_std * epsilon / config.maximum_increment
    )
    assert generated.shape == (2, 2, 3)
    torch.testing.assert_close(generated, current[:, None] + expected_increment[None])


def test_joint_full_generator_conditions_on_initial_state_and_noise() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = JointFullStateGeneratorConfig(
        front_state_size=2,
        maximum_increment=8.0,
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
    )
    current = torch.tensor(((-1.0, 0.5), (1.0, -0.5)), dtype=torch.float64)
    law = JointFullStateConditionalGenerator(
        env,
        current_front_states=current,
        config=config,
    )
    with torch.no_grad():
        law.network.I0.weight.zero_()
        law.network.I0.weight[:, 0].fill_(0.2)
        law.network.I0.weight[:, 3:].fill_(0.2)
        law.network.I0.bias.fill_(0.1)
        law.network.In[-1].weight.fill_(0.3)
        law.network.In[-1].bias.zero_()
    epsilon = torch.tensor(((0.2, -0.1),), dtype=torch.float64, requires_grad=True)
    output = law(
        current_front_states=current,
        spot_rank=torch.tensor(((0.4,), (0.4,)), dtype=torch.float64),
        epsilon=epsilon,
    )
    increments = output[:, 0] - current
    assert not torch.allclose(increments[0], increments[1])
    gradient = torch.autograd.grad(output[0, 0, 0], epsilon)[0]
    assert torch.count_nonzero(gradient) == 2


def test_uncompressed_front_generator_uses_complete_raw_context() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = JointUncompressedFrontGeneratorConfig(
        context_state_size=5,
        front_state_size=2,
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
    )
    current = torch.tensor(
        ((1.0, -2.0, 0.5, -1.0, 2.0), (-0.5, 1.0, 2.0, 1.0, -2.0)),
        dtype=torch.float64,
    )
    law = JointUncompressedFrontConditionalGenerator(
        env,
        current_raw_states=current,
        config=config,
    )
    epsilon = torch.tensor(((0.5, -1.0), (-0.25, 0.75)), dtype=torch.float64)
    generated = law(
        current_raw_states=current,
        normalized_spot=torch.tensor(
            ((1.01, 0.98), (1.03, 0.96)),
            dtype=torch.float64,
        ),
        epsilon=epsilon,
    )
    normalized = (
        (current - law.context_center) / law.context_scale
    )[:, None].expand(-1, 2, -1)
    noise = epsilon[None].expand(2, -1, -1)
    context = torch.cat(
        (
            normalized,
            torch.tensor(
                (((1.01,), (0.98,)), ((1.03,), (0.96,))),
                dtype=torch.float64,
            ),
            noise,
        ),
        dim=-1,
    )
    assert generated.shape == (2, 2, 2)
    torch.testing.assert_close(
        generated,
        law.output_center + law.output_scale * law.network(context),
    )

    with torch.no_grad():
        law.network.I0.weight.zero_()
        law.network.I0.weight[:, 4].fill_(0.2)
        law.network.I0.bias.fill_(0.1)
        law.network.In[-1].weight.fill_(0.3)
        law.network.In[-1].bias.zero_()
    conditioned = law(
        current_raw_states=current,
        normalized_spot=torch.ones((2, 2), dtype=torch.float64),
        epsilon=epsilon,
    )
    assert not torch.allclose(conditioned[0], conditioned[1])


def test_uncompressed_front_generator_is_direct_not_residual() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    current = torch.tensor(
        ((-2.0, 1.0, 0.5), (4.0, 3.0, -0.5)),
        dtype=torch.float64,
    )
    law = JointUncompressedFrontConditionalGenerator(
        env,
        current_raw_states=current,
        config=JointUncompressedFrontGeneratorConfig(
            context_state_size=3,
            front_state_size=2,
            network=FeedForwardConfig(
                activation="SiLU",
                num_depth=1,
                num_width=4,
            ),
        ),
    )
    with torch.no_grad():
        for parameter in law.network.parameters():
            parameter.zero_()
        law.network.In[-1].bias.copy_(
            torch.tensor((0.25, -0.5), dtype=torch.float64)
        )
    generated = law(
        current_raw_states=current,
        normalized_spot=torch.ones((2, 1), dtype=torch.float64),
        epsilon=torch.zeros((1, 2), dtype=torch.float64),
    )
    expected = law.output_center + law.output_scale * torch.tensor(
        (0.25, -0.5),
        dtype=torch.float64,
    )
    torch.testing.assert_close(generated[0, 0], expected)
    torch.testing.assert_close(generated[1, 0], expected)
    assert not torch.allclose(generated[:, 0], current[:, :2])


def test_uncompressed_context_rebase_preserves_values_and_input_gradients() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = JointUncompressedFrontGeneratorConfig(
        context_state_size=5,
        front_state_size=2,
        network=FeedForwardConfig(activation="SiLU", num_depth=2, num_width=7),
    )
    pilot_reference = torch.tensor(
        ((-2.0, 1.0, 0.5, 3.0, -1.0), (0.0, 2.0, 1.5, 5.0, 1.0)),
        dtype=torch.float64,
    )
    all_state_reference = torch.tensor(
        (
            (-4.0, -1.0, 0.0, 1.0, -3.0),
            (-1.0, 0.0, 2.0, 4.0, -1.0),
            (2.0, 3.0, 4.0, 7.0, 2.0),
            (5.0, 6.0, 8.0, 9.0, 6.0),
        ),
        dtype=torch.float64,
    )
    law = JointUncompressedFrontConditionalGenerator(
        env,
        current_raw_states=pilot_reference,
        config=config,
    )
    generator = torch.Generator().manual_seed(812)
    with torch.no_grad():
        for parameter in law.network.parameters():
            parameter.copy_(
                0.1
                * torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    generator=generator,
                )
            )
    evaluation_states = all_state_reference[:2].clone().requires_grad_()
    spot_rank = torch.tensor(((0.2, -0.4), (0.1, 0.3)), dtype=torch.float64)
    epsilon = torch.tensor(((0.5, -0.7), (-0.2, 0.6)), dtype=torch.float64)
    before = law(
        current_raw_states=evaluation_states,
        normalized_spot=torch.exp(0.02 * spot_rank),
        epsilon=epsilon,
    )
    before_gradient = torch.autograd.grad(before.sum(), evaluation_states)[0]

    law.rebase_context_normalization(all_state_reference)

    rebased_states = evaluation_states.detach().clone().requires_grad_()
    after = law(
        current_raw_states=rebased_states,
        normalized_spot=torch.exp(0.02 * spot_rank),
        epsilon=epsilon,
    )
    after_gradient = torch.autograd.grad(after.sum(), rebased_states)[0]
    torch.testing.assert_close(after, before.detach(), rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        after_gradient,
        before_gradient,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        law.context_center,
        torch.mean(all_state_reference, dim=0),
    )
    torch.testing.assert_close(
        law.context_scale,
        torch.std(all_state_reference, dim=0, correction=0),
    )


def test_two_layer_2048_uncompressed_generator_capacity() -> None:
    env = BaseEnvironment(precision=32, device="cpu", debug=True)
    config = JointUncompressedFrontGeneratorConfig(
        context_state_size=808,
        front_state_size=202,
        network=FeedForwardConfig(
            activation="SiLU",
            num_depth=2,
            num_width=2_048,
        ),
    )
    law = JointUncompressedFrontConditionalGenerator(
        env,
        current_raw_states=torch.stack(
            (
                torch.zeros(808, dtype=torch.float32),
                torch.ones(808, dtype=torch.float32),
            )
        ),
        config=config,
    )
    assert sum(parameter.numel() for parameter in law.parameters()) == 6_682_826
    assert law.network.I0.in_features == 1_011
    assert law.network.I0.out_features == 2_048
    assert law.network.In[-1].out_features == 202
