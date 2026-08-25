from __future__ import annotations

import torch

from quantdata.dysanos.flex.oce_transport import (
    ConditionalOCEConfig,
    SpotMarginalOCEAction,
    apply_first_line_transport_sequence,
    conditional_oce_objective,
    estimate_return_inverse_sqrt,
    solve_conditional_powerexp_oce,
    solve_conditional_powerexp_oce_l1,
    solve_powerexp_translation,
)
from quantdata.dysanos.flex.weekly_measure import WeeklyMeasureDraw
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_conditional_oce_gradients_are_density_mass_and_moments() -> None:
    returns = torch.tensor(
        (
            ((-0.03, 0.01), (0.02, -0.04), (0.01, 0.05), (0.04, -0.01)),
            ((-0.02, -0.03), (0.01, 0.02), (0.03, 0.01), (-0.01, 0.04)),
        ),
        dtype=torch.float64,
    )
    action = torch.tensor(((0.4, -0.2), (-0.1, 0.3)), dtype=torch.float64, requires_grad=True)
    objective, details = conditional_oce_objective(returns, action)
    objective.backward()

    contexts = returns.shape[0]
    torch.testing.assert_close(
        action.grad,
        details["action_residual"] / contexts,
    )
    torch.testing.assert_close(details["mass_residual"], torch.zeros(2, dtype=torch.float64))


def test_zero_action_has_zero_oce_and_unit_density() -> None:
    returns = torch.tensor((((-0.05,), (0.05,), (-0.02,), (0.02,)),), dtype=torch.float64)
    action = torch.zeros((1, 1), dtype=torch.float64)
    objective, details = conditional_oce_objective(returns, action)

    torch.testing.assert_close(objective, torch.zeros_like(objective))
    torch.testing.assert_close(details["log_marginal"], torch.zeros_like(details["log_marginal"]))
    torch.testing.assert_close(details["mass_residual"], torch.zeros_like(details["mass_residual"]))


def test_powerexp_translation_normalizes_each_conditional_density() -> None:
    pnl = torch.tensor(
        ((-2.0, -0.5, 0.2, 4.0), (-100.0, -0.1, 0.0, 0.3)),
        dtype=torch.float64,
    )

    translation = solve_powerexp_translation(pnl)
    shifted = translation[:, None] + pnl
    marginal = torch.where(shifted > 0.0, torch.exp(-shifted), 1.0 - shifted)

    torch.testing.assert_close(torch.mean(marginal, dim=1), torch.ones(2, dtype=torch.float64))


def test_conditional_powerexp_newton_solves_mass_and_option_moments() -> None:
    generator = torch.Generator().manual_seed(712)
    shocks = torch.randn((2, 4096, 2), generator=generator, dtype=torch.float64)
    returns = torch.stack(
        (
            0.04 + 0.12 * shocks[..., 0],
            -0.02 + 0.05 * shocks[..., 0] + 0.09 * shocks[..., 1],
        ),
        dim=-1,
    )
    inverse_sqrt = estimate_return_inverse_sqrt(
        returns,
        minimum_standard_deviation=1.0e-8,
    )

    _, _, details = solve_conditional_powerexp_oce(
        returns[:, None],
        inverse_sqrt,
        num_iterations=24,
        ridge=1.0e-10,
    )

    torch.testing.assert_close(
        details["mass_residual"],
        torch.zeros((2, 1), dtype=torch.float64),
        atol=1.0e-9,
        rtol=0.0,
    )
    torch.testing.assert_close(
        details["action_residual"],
        torch.zeros((2, 2), dtype=torch.float64),
        atol=1.0e-9,
        rtol=0.0,
    )


def test_conditional_powerexp_l1_solver_respects_raw_cap_and_cost() -> None:
    generator = torch.Generator().manual_seed(1_029)
    shocks = torch.randn((1, 4096, 2), generator=generator, dtype=torch.float64)
    returns = torch.stack(
        (
            0.04 + 0.12 * shocks[..., 0],
            -0.02 + 0.04 * shocks[..., 0] + 0.09 * shocks[..., 1],
        ),
        dim=-1,
    )
    inverse_sqrt = estimate_return_inverse_sqrt(
        returns,
        minimum_standard_deviation=1.0e-8,
    )

    action, _, details = solve_conditional_powerexp_oce_l1(
        returns[:, None],
        inverse_sqrt,
        maximum_l1_action=0.5,
        l1_cost_per_unit=1.0e-3,
        num_iterations=48,
    )

    assert float(torch.sum(torch.abs(action))) <= 0.5 + 1.0e-10
    assert float(details["cost_adjusted_objective"][0]) > 0.0
    assert bool(details["converged"][0])
    torch.testing.assert_close(
        details["mass_residual"],
        torch.zeros((1, 1), dtype=torch.float64),
        atol=1.0e-9,
        rtol=0.0,
    )


def test_spot_marginal_action_has_smooth_radial_exposure_budget() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    maximum_norm = 0.5
    model = SpotMarginalOCEAction(
        env,
        config=ConditionalOCEConfig(
            state_sizes_by_expiry=(2, 1),
            num_instruments=2,
            maximum_preconditioned_action_norm=maximum_norm,
            network=FeedForwardConfig(activation="SiLU", num_depth=0, num_width=4),
        ),
        inverse_sqrt_second_moment=torch.eye(2, dtype=torch.float64),
    )
    with torch.no_grad():
        model.network.I0.weight.fill_(10.0)
        model.network.I0.bias.fill_(10.0)

    state = torch.ones((3, 3), dtype=torch.float64, requires_grad=True)
    action, preconditioned = model(state)
    torch.sum(action).backward()

    assert bool(torch.all(torch.linalg.vector_norm(preconditioned, dim=-1) < maximum_norm))
    assert bool(torch.all(torch.isfinite(state.grad)))


def test_spot_marginal_action_hard_cap_preserves_admissible_output() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = SpotMarginalOCEAction(
        env,
        config=ConditionalOCEConfig(
            state_sizes_by_expiry=(2, 1),
            num_instruments=2,
            maximum_preconditioned_action_norm=1.0,
            preconditioned_action_bound_mode="hard_cap_penalty",
            action_cap_excess_penalty_weight=2.0,
            network=FeedForwardConfig(activation="SiLU", num_depth=0, num_width=4),
        ),
        inverse_sqrt_second_moment=torch.eye(2, dtype=torch.float64),
    )
    with torch.no_grad():
        model.network.I0.weight.zero_()
        model.network.I0.bias.copy_(torch.tensor((0.3, 0.4), dtype=torch.float64))

    action, preconditioned, penalty, maximum_excess = (
        model.forward_with_cap_penalty(torch.zeros((3, 3), dtype=torch.float64))
    )

    expected = torch.tensor(((0.3, 0.4),) * 3, dtype=torch.float64)
    torch.testing.assert_close(preconditioned, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(action, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(penalty, torch.zeros((), dtype=torch.float64))
    torch.testing.assert_close(maximum_excess, torch.zeros((), dtype=torch.float64))


def test_spot_marginal_action_hard_cap_penalizes_uncapped_excess() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = SpotMarginalOCEAction(
        env,
        config=ConditionalOCEConfig(
            state_sizes_by_expiry=(2, 1),
            num_instruments=2,
            maximum_preconditioned_action_norm=1.0,
            preconditioned_action_bound_mode="hard_cap_penalty",
            action_cap_excess_penalty_weight=2.0,
            network=FeedForwardConfig(activation="SiLU", num_depth=0, num_width=4),
        ),
        inverse_sqrt_second_moment=torch.eye(2, dtype=torch.float64),
    )
    with torch.no_grad():
        model.network.I0.weight.zero_()
        model.network.I0.bias.copy_(torch.tensor((3.0, 4.0), dtype=torch.float64))

    action, preconditioned, penalty, maximum_excess = (
        model.forward_with_cap_penalty(torch.zeros((3, 3), dtype=torch.float64))
    )
    penalty.backward()

    expected = torch.tensor(((0.6, 0.8),) * 3, dtype=torch.float64)
    torch.testing.assert_close(preconditioned, expected)
    torch.testing.assert_close(action, expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(preconditioned, dim=-1),
        torch.ones(3, dtype=torch.float64),
    )
    torch.testing.assert_close(penalty, torch.tensor(32.0, dtype=torch.float64))
    torch.testing.assert_close(maximum_excess, torch.tensor(4.0, dtype=torch.float64))
    assert model.network.I0.bias.grad is not None
    torch.testing.assert_close(
        model.network.I0.bias.grad,
        torch.tensor((9.6, 12.8), dtype=torch.float64),
    )


def test_spot_marginal_action_can_condition_on_current_log_spot() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = SpotMarginalOCEAction(
        env,
        config=ConditionalOCEConfig(
            state_sizes_by_expiry=(2, 1),
            num_instruments=2,
            include_current_log_spot=True,
            current_log_spot_mean=5.0,
            current_log_spot_std=2.0,
            network=FeedForwardConfig(activation="SiLU", num_depth=0, num_width=4),
        ),
        inverse_sqrt_second_moment=torch.eye(2, dtype=torch.float64),
    )
    with torch.no_grad():
        model.network.I0.weight.zero_()
        model.network.I0.weight[:, -1] = 1.0
        model.network.I0.bias.zero_()

    state = torch.zeros((2, 3), dtype=torch.float64)
    action, preconditioned = model(
        state,
        torch.tensor((3.0, 7.0), dtype=torch.float64),
    )

    expected = torch.tensor(((-1.0, -1.0), (1.0, 1.0)), dtype=torch.float64)
    torch.testing.assert_close(preconditioned, expected)
    torch.testing.assert_close(action, expected)


def test_first_line_transport_sequence_applies_newest_increment_first() -> None:
    class AffineKernel:
        def __init__(self, scale: float, shift: float):
            self.scale = scale
            self.shift = shift

        def step_first_line_with_log_abs_det(
            self,
            current_state: torch.Tensor,
            gross_spot_return: torch.Tensor,
            standard_normals: torch.Tensor,
            current_log_spot: torch.Tensor | None = None,
        ) -> tuple[WeeklyMeasureDraw, torch.Tensor]:
            del gross_spot_return, current_log_spot
            transformed = standard_normals.clone()
            transformed[..., 1:] = self.scale * transformed[..., 1:] + self.shift
            draw = WeeklyMeasureDraw(
                reduced_state=transformed[..., 1:],
                reference_reduced_state=current_state,
                transformed_standard_normals=transformed,
                reference_standard_normals=standard_normals,
            )
            logdet = torch.full(
                standard_normals.shape[:-1],
                standard_normals.shape[-1] - 1,
                dtype=standard_normals.dtype,
            ) * torch.log(torch.as_tensor(self.scale, dtype=standard_normals.dtype))
            return draw, logdet

    current_state = torch.zeros((1, 2), dtype=torch.float64)
    gross_spot_return = torch.ones(1, dtype=torch.float64)
    standard_normals = torch.tensor(((0.25, 3.0, -2.0),), dtype=torch.float64)
    accepted = AffineKernel(scale=2.0, shift=0.0)
    increment = AffineKernel(scale=1.0, shift=1.0)

    draw, logdet = apply_first_line_transport_sequence(
        (accepted, increment),
        current_state,
        gross_spot_return,
        standard_normals,
    )

    torch.testing.assert_close(
        draw.transformed_standard_normals,
        torch.tensor(((0.25, 8.0, -2.0),), dtype=torch.float64),
    )
    torch.testing.assert_close(
        logdet,
        2.0 * torch.log(torch.tensor((2.0,), dtype=torch.float64)),
    )
