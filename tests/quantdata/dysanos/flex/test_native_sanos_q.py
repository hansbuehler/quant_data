"""Mathematical and device tests for the Torch-native SANOS Q roller."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.config import (
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.native_sanos_q import NativeSanosQRoller
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface
from quantdata.torch import BaseEnvironment


def _surface(
    device: str = "cpu",
    *,
    requires_grad: bool = False,
    batch_size: int = 1,
) -> tuple[
    BaseEnvironment,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    FlexSanosDecoder,
    SanosSurface,
    torch.Tensor,
]:
    env = BaseEnvironment(precision=64, device=device, debug=True)
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=3,
    )
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlvsqs = torch.tensor(
        [[
            [0.12, 0.09, 0.07, 0.08, 0.11, 0.14],
            [0.10, 0.08, 0.06, 0.07, 0.10, 0.13],
            [0.09, 0.07, 0.05, 0.06, 0.09, 0.12],
        ]],
        dtype=env.fdtype,
        device=env.device,
    ).repeat(batch_size, 1, 1)
    dlvsqs.requires_grad_(requires_grad)
    surface = decoder.forward_batch(
        dlvsqs=dlvsqs,
        forward_vols=torch.tensor(
            [[0.22, 0.24, 0.25]], dtype=env.fdtype, device=env.device
        ).repeat(batch_size, 1),
        eta=torch.tensor(
            [[0.35, 0.40, 0.42]], dtype=env.fdtype, device=env.device
        ).repeat(batch_size, 1),
    )
    return env, geometry, decoder, surface, dlvsqs


def test_transition_rows_have_exact_one_and_two_step_martingale_towers() -> None:
    env, geometry, _, surface, _ = _surface()
    roller = NativeSanosQRoller(env, geometry=geometry)
    support = surface.support_strikes[0, 0]
    nodes = torch.arange(len(support), dtype=torch.int64).unsqueeze(0)

    rows_w2 = roller.transition_rows(surface, nodes, target_expiry_index=1)[0]
    rows_w3 = roller.transition_density(
        surface,
        rows_w2.unsqueeze(0),
        target_expiry_index=2,
    )[0]

    torch.testing.assert_close(rows_w2.sum(dim=-1), torch.ones_like(support))
    torch.testing.assert_close(rows_w2 @ support, support)
    torch.testing.assert_close(rows_w3.sum(dim=-1), torch.ones_like(support))
    torch.testing.assert_close(rows_w3 @ support, support)
    torch.testing.assert_close(surface.densities[0, 0] @ rows_w2, surface.densities[0, 1])
    torch.testing.assert_close(surface.densities[0, 0] @ rows_w3, surface.densities[0, 2])


def test_revealed_node_call_curves_satisfy_exact_cash_strike_tower() -> None:
    env, geometry, decoder, surface, _ = _surface()
    roller = NativeSanosQRoller(env, geometry=geometry)
    quadrature_nodes, quadrature_weights = np.polynomial.hermite.hermgauss(48)
    normals = torch.tensor(
        np.sqrt(2.0) * quadrature_nodes,
        dtype=env.fdtype,
        device=env.device,
    )
    weights = torch.tensor(
        quadrature_weights / np.sqrt(np.pi),
        dtype=env.fdtype,
        device=env.device,
    )
    support = surface.support_strikes[0, 0]
    support_size = len(support)
    nodes = torch.arange(support_size, device=env.device).repeat_interleave(
        len(normals)
    )
    tiled_normals = normals.repeat(support_size)
    path_weights = surface.densities[0, 0].repeat_interleave(len(normals)) * weights.repeat(
        support_size
    )
    front_vol = surface.backbone_total_vols[0, 0]
    front_spot = support[nodes] * torch.exp(
        -0.5 * front_vol.square() + front_vol * tiled_normals
    )
    curves = roller.price_remaining_calls(
        surface,
        current_expiry_index=0,
        revealed_nodes=nodes.unsqueeze(0),
        normalized_strikes=(1.0 / front_spot).reshape(1, -1, 1),
    )
    cash_values = front_spot[:, None] * curves.normalized_call_prices[0, :, :, 0]
    tower_values = torch.sum(path_weights[:, None] * cash_values, dim=0)
    direct = decoder.price_calls_batch(
        surface,
        expiries_y=surface.expiries_y[:, 1:],
        strikes=torch.ones((1, 2), dtype=env.fdtype, device=env.device),
    )[0]

    torch.testing.assert_close(tower_values, direct, atol=2.0e-12, rtol=2.0e-12)


def test_conditional_custom_support_is_martingale_and_call_curves_are_static_arb_free() -> None:
    env, geometry, _, surface, _ = _surface()
    roller = NativeSanosQRoller(env, geometry=geometry)
    nodes = torch.tensor([[2, 4, 6]], dtype=torch.int64, device=env.device)
    strikes = torch.linspace(0.65, 1.35, 71, dtype=env.fdtype, device=env.device)

    curves = roller.price_remaining_calls(
        surface,
        current_expiry_index=0,
        revealed_nodes=nodes,
        normalized_strikes=strikes,
    )

    mass = curves.conditional_densities.sum(dim=-1)
    mean = torch.sum(
        curves.conditional_densities * curves.normalized_support.unsqueeze(-2),
        dim=-1,
    )
    torch.testing.assert_close(mass, torch.ones_like(mass))
    torch.testing.assert_close(mean, torch.ones_like(mean))
    calls = curves.normalized_call_prices
    assert bool(torch.all(torch.diff(calls, dim=-1) <= 2.0e-13))
    slopes = torch.diff(calls, dim=-1) / torch.diff(strikes)
    assert bool(torch.all(torch.diff(slopes, dim=-1) >= -2.0e-11))
    intrinsic = torch.relu(1.0 - strikes)
    assert bool(torch.all(calls >= intrinsic - 2.0e-13))
    assert bool(torch.all(calls <= 1.0 + 2.0e-13))


def test_fresh_native_paths_match_spot_and_call_moments() -> None:
    env, geometry, decoder, surface, _ = _surface()
    roller = NativeSanosQRoller(env, geometry=geometry)
    generator = torch.Generator(device=env.device).manual_seed(91)
    randoms = roller.make_randoms(
        batch_size=1,
        num_paths=80_000,
        num_steps=3,
        generator=generator,
        antithetic=True,
    )

    paths = roller.sample_paths(surface, randoms=randoms)

    spots = paths.normalized_spots[0]
    for expiry_index in range(3):
        sample = spots[:, expiry_index]
        standard_error = torch.std(sample, correction=1) / np.sqrt(len(sample))
        assert abs(float(torch.mean(sample) - 1.0)) <= 5.0 * float(standard_error)
    payoff = torch.relu(spots - 1.0)
    mean_payoff = payoff.mean(dim=0)
    payoff_se = payoff.std(dim=0, correction=1) / np.sqrt(len(payoff))
    direct = decoder.price_calls_batch(
        surface,
        expiries_y=surface.expiries_y,
        strikes=torch.ones((1, 3), dtype=env.fdtype, device=env.device),
    )[0]
    assert bool(torch.all(torch.abs(mean_payoff - direct) <= 5.0 * payoff_se))


def test_pricing_retains_torch_gradients_after_discrete_node_is_fixed() -> None:
    env, geometry, _, surface, dlvsqs = _surface(requires_grad=True)
    roller = NativeSanosQRoller(env, geometry=geometry)
    curves = roller.price_remaining_calls(
        surface,
        current_expiry_index=0,
        revealed_nodes=torch.tensor([[3, 4]], dtype=torch.int64, device=env.device),
        normalized_strikes=torch.tensor((0.9, 1.0, 1.1), dtype=env.fdtype, device=env.device),
    )

    curves.normalized_call_prices.sum().backward()

    assert dlvsqs.grad is not None
    assert bool(torch.all(torch.isfinite(dlvsqs.grad)))
    assert float(torch.sum(torch.abs(dlvsqs.grad))) > 0.0


def test_last_expiry_is_a_hard_boundary_without_far_tail() -> None:
    env, geometry, _, surface, _ = _surface()
    roller = NativeSanosQRoller(env, geometry=geometry)
    with pytest.raises(ValueError, match="last expiry"):
        roller.make_randoms(batch_size=1, num_paths=2, num_steps=4)
    with pytest.raises(ValueError, match="no far-end tail"):
        roller.transition_rows(
            surface,
            torch.tensor([[2]], dtype=torch.int64),
            target_expiry_index=3,
        )
    with pytest.raises(ValueError, match="no far-end tail"):
        roller.price_remaining_calls(
            surface,
            current_expiry_index=2,
            revealed_nodes=torch.tensor([[2]], dtype=torch.int64),
            normalized_strikes=torch.tensor((1.0,), dtype=env.fdtype),
        )


def test_live_front_and_advance_api_supports_surface_batches() -> None:
    env, geometry, _, surface, _ = _surface(batch_size=2)
    roller = NativeSanosQRoller(env, geometry=geometry)
    generator = torch.Generator(device=env.device).manual_seed(47)
    randoms = roller.make_randoms(
        batch_size=2,
        num_paths=16,
        num_steps=3,
        generator=generator,
        antithetic=True,
    )
    state = roller.sample_front_state(
        surface,
        categorical_uniform=randoms.categorical_uniforms[..., 0],
        backbone_normal=randoms.backbone_normals[..., 0],
    )
    assert state.normalized_spot.shape == (2, 16)
    assert state.revealed_node.shape == (2, 16)
    assert state.expiry_index == 0
    for step in (1, 2):
        state = roller.advance_state(
            surface,
            state,
            categorical_uniform=randoms.categorical_uniforms[..., step],
            backbone_normal=randoms.backbone_normals[..., step],
        )
        assert state.expiry_index == step
    with pytest.raises(ValueError, match="last expiry"):
        roller.advance_state(
            surface,
            state,
            categorical_uniform=randoms.categorical_uniforms[..., 2],
            backbone_normal=randoms.backbone_normals[..., 2],
        )


def test_public_density_step_is_torch_compile_compatible() -> None:
    env, geometry, decoder, surface, _ = _surface()
    support = surface.support_strikes[:, 1]
    dlvsqs = surface.dlvsqs[:, 1]
    rhs = surface.densities[:, 0]
    dt = decoder.model_dts_y[1]

    def solve(
        support_value: torch.Tensor,
        dlv_value: torch.Tensor,
        rhs_value: torch.Tensor,
    ) -> torch.Tensor:
        return decoder.solve_density_step_batch(
            support=support_value,
            dlvsqs=dlv_value,
            time_increment_y=dt,
            rhs_density=rhs_value,
        )

    expected = solve(support, dlvsqs, rhs)
    compiled = torch.compile(solve, backend="eager", fullgraph=True)
    actual = compiled(support, dlvsqs, rhs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_q_preserves_cuda_device_and_dtype() -> None:
    env, geometry, _, surface, _ = _surface("cuda:0")
    roller = NativeSanosQRoller(env, geometry=geometry)
    generator = torch.Generator(device=env.device).manual_seed(11)
    randoms = roller.make_randoms(
        batch_size=1,
        num_paths=32,
        num_steps=3,
        generator=generator,
        antithetic=True,
    )
    paths = roller.sample_paths(surface, randoms=randoms)
    curves = roller.price_remaining_calls(
        surface,
        current_expiry_index=0,
        revealed_nodes=paths.revealed_nodes[..., 0],
        normalized_strikes=torch.tensor((0.9, 1.0, 1.1), dtype=env.fdtype, device=env.device),
    )

    assert paths.normalized_spots.device.type == "cuda"
    assert paths.normalized_spots.dtype == env.fdtype
    assert curves.normalized_call_prices.device.type == "cuda"
    assert curves.normalized_call_prices.dtype == env.fdtype
