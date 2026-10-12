"""Tests for the semi-analytic native DYSANOS martingale chart."""

from __future__ import annotations

from pathlib import Path

import torch

from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.martingale_coordinate_chart import (
    _price_cash_call_grid,
    complete_branch_dlvs_to_martingales,
    decode_martingale_coordinates,
    implicit_linear_system_directional_derivatives,
    recover_density_from_cash_calls,
)
from quantdata.dysanos.flex.run_local_martingale_chart_scaling_gate import (
    Config as ScalingGateConfig,
)
from quantdata.dysanos.flex.run_local_martingale_chart_scaling_gate import (
    _problem as scaling_gate_problem,
)
from quantdata.dysanos.flex.run_local_martingale_chart_scaling_gate import (
    _successor_coordinates,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.torch import BaseEnvironment


def _problem() -> tuple[FlexSanosDecoder, torch.Tensor, torch.Tensor, torch.Tensor]:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=3,
        explicit_model_business_dtes=(1.0, 5.0, 10.0),
    )
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlvsqs = torch.tensor(
        [[
            [0.16, 0.12, 0.09, 0.10, 0.14, 0.19],
            [0.18, 0.14, 0.11, 0.12, 0.16, 0.21],
            [0.20, 0.16, 0.13, 0.14, 0.18, 0.23],
        ]],
        dtype=env.fdtype,
    )
    forward_vols = torch.tensor([[0.22, 0.24, 0.25]], dtype=env.fdtype)
    eta = torch.tensor([[0.18, 0.24, 0.24]], dtype=env.fdtype)
    return decoder, dlvsqs, forward_vols, eta


def test_chart_exactly_recovers_a_native_surface() -> None:
    decoder, dlvsqs, forward_vols, eta = _problem()
    surface = decoder.forward_batch(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=eta,
    )
    spot = torch.ones(1, dtype=dlvsqs.dtype)
    strikes = surface.support_strikes[..., 1:-1]
    calls = _price_cash_call_grid(
        decoder,
        surface,
        spot=spot,
        cash_strikes=strikes,
    )
    recovered = decode_martingale_coordinates(
        decoder,
        spot=spot,
        forward_vols=forward_vols,
        eta=eta,
        cash_strikes=strikes,
        cash_calls=calls,
    )
    torch.testing.assert_close(recovered.surface.densities, surface.densities)
    torch.testing.assert_close(recovered.surface.dlvsqs, dlvsqs, atol=2.0e-11, rtol=2.0e-10)
    assert float(recovered.diagnostics.maximum_absolute_cash_repricing_error) < 2.0e-13


def test_density_inverse_enforces_mass_mean_and_cash_calls() -> None:
    decoder, dlvsqs, forward_vols, eta = _problem()
    surface = decoder.forward_batch(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=eta,
    )
    spot = torch.tensor([1.0], dtype=dlvsqs.dtype)
    strikes = surface.support_strikes[..., 1:-1]
    calls = _price_cash_call_grid(
        decoder,
        surface,
        spot=spot,
        cash_strikes=strikes,
    )
    density, condition = recover_density_from_cash_calls(
        spot=spot,
        support=surface.support_strikes,
        backbone_total_vols=surface.backbone_total_vols,
        cash_strikes=strikes,
        cash_calls=calls,
    )
    torch.testing.assert_close(density, surface.densities)
    assert bool(torch.all(torch.isfinite(condition) & (condition > 1.0)))


def test_bounded_completion_preserves_an_existing_martingale_surface() -> None:
    decoder, dlvsqs, forward_vols, eta = _problem()
    surface = decoder.forward_batch(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=eta,
    )
    spot = torch.ones(1, dtype=dlvsqs.dtype)
    strikes = surface.support_strikes[..., 3:5]
    calls = _price_cash_call_grid(
        decoder,
        surface,
        spot=spot,
        cash_strikes=strikes,
    )
    completion = complete_branch_dlvs_to_martingales(
        decoder,
        branch_probabilities=torch.ones(1, dtype=dlvsqs.dtype),
        spot=spot,
        support=surface.support_strikes,
        density0=surface.density0,
        backbone_total_vols=surface.backbone_total_vols,
        reference_densities=surface.densities,
        cash_strikes=strikes,
        target_mean_cash_calls=calls[0],
        density_floor=1.0e-10,
        smooth_quadratic_projection=True,
        dlvsq_projection_weight=100.0,
    )
    torch.testing.assert_close(
        completion.densities,
        surface.densities,
        atol=2.0e-9,
        rtol=2.0e-8,
    )
    torch.testing.assert_close(
        completion.dlvsqs,
        dlvsqs,
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    assert completion.maximum_equality_residual < 2.0e-9
    assert completion.maximum_inequality_residual < 2.0e-9
    assert completion.selection_method == "convex-dlv-chi-square"
    assert completion.projection_solver_status in ("optimal", "optimal_inaccurate")


def test_implicit_linear_system_derivatives_recover_known_solution_curve() -> None:
    dtype = torch.float64
    matrix = torch.tensor([[2.0, 0.3], [-0.2, 1.4]], dtype=dtype)
    matrix_first = torch.tensor([[[0.1, -0.4], [0.2, 0.3]]], dtype=dtype)
    matrix_second = torch.tensor([[[-0.3, 0.2], [0.1, -0.2]]], dtype=dtype)
    solution = torch.tensor([0.7, -0.4], dtype=dtype)
    solution_first = torch.tensor([[-0.2, 0.5]], dtype=dtype)
    solution_second = torch.tensor([[0.6, -0.1]], dtype=dtype)
    right_hand_side = matrix @ solution
    right_hand_side_first = torch.einsum(
        "fij,j->fi", matrix_first, solution
    ) + torch.einsum("ij,fj->fi", matrix, solution_first)
    right_hand_side_second = (
        torch.einsum("fij,j->fi", matrix_second, solution)
        + 2.0 * torch.einsum("fij,fj->fi", matrix_first, solution_first)
        + torch.einsum("ij,fj->fi", matrix, solution_second)
    )

    result = implicit_linear_system_directional_derivatives(
        matrix=matrix,
        right_hand_side=right_hand_side,
        matrix_first=matrix_first,
        matrix_second=matrix_second,
        right_hand_side_first=right_hand_side_first,
        right_hand_side_second=right_hand_side_second,
    )

    torch.testing.assert_close(result.solution, solution)
    torch.testing.assert_close(result.first, solution_first)
    torch.testing.assert_close(result.second, solution_second)
    assert float(result.maximum_first_order_equation_residual) < 1.0e-14
    assert float(result.maximum_second_order_equation_residual) < 1.0e-14


def test_fixed_grid_and_backbone_ablation_preserves_initial_support() -> None:
    output = Path("/tmp/quant_data/test-local-martingale-chart")
    moving_config = ScalingGateConfig(output=output)
    fixed_config = ScalingGateConfig(
        output=output,
        fixed_relative_grid=True,
        fixed_backbone_annualized_volatility=0.05,
    )
    _, moving_decoder, _, moving_parameters, _ = scaling_gate_problem(
        moving_config
    )
    env, fixed_decoder, _, fixed_parameters, _ = scaling_gate_problem(
        fixed_config
    )
    moving_surface = moving_decoder.forward_batch(
        dlvsqs=moving_parameters.dlvsqs,
        forward_vols=moving_parameters.forward_vols,
        eta=moving_parameters.eta,
    )
    fixed_surface = fixed_decoder.forward_batch(
        dlvsqs=fixed_parameters.dlvsqs,
        forward_vols=fixed_parameters.forward_vols,
        eta=fixed_parameters.eta,
    )

    torch.testing.assert_close(
        fixed_surface.support_strikes,
        moving_surface.support_strikes,
    )
    target_total = 0.05 * torch.sqrt(fixed_decoder.model_expiries_y)
    torch.testing.assert_close(
        fixed_surface.backbone_total_vols[0],
        target_total,
    )
    successor, forward, eta, _ = _successor_coordinates(
        env,
        fixed_decoder,
        fixed_surface,
        business_day_fraction=0.5,
        fixed_backbone_annualized_volatility=0.05,
    )
    torch.testing.assert_close(
        successor._total_vols(forward * eta)[0],
        0.05 * torch.sqrt(successor.model_expiries_y),
    )
