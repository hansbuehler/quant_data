from __future__ import annotations

import torch

from quantdata.dysanos.flex.config import (
    ForwardMuFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.model import (
    LatentStateSanosModel,
    NativeParameters,
    bound_native_raw,
    raw_from_native_parameters,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.dysanos.flex.training import (
    initialize_triangular_pca_decoder_,
)
from quantdata.torch import BaseEnvironment


def _small_geometry() -> ForwardMuFlexGeometryConfig:
    return ForwardMuFlexGeometryConfig(
        num_inner_strikes=5,
        num_model_expiries=3,
    )


def test_forward_mu_geometry_is_expiry_major() -> None:
    geometry = ForwardMuFlexGeometryConfig()
    assert geometry.expiry_block_state_size == 202
    assert geometry.native_state_size == 2_828

    small = _small_geometry()
    assert small.expiry_block_state_size == 7
    assert small.native_state_size == 21


def test_forward_mu_bounds_round_trip_in_expiry_blocks() -> None:
    geometry = _small_geometry()
    dlvsqs = torch.linspace(0.02, 0.20, 30, dtype=torch.float64).reshape(2, 3, 5)
    forward_vols = torch.tensor(
        [[0.15, 0.20, 0.25], [0.30, 0.35, 0.40]],
        dtype=torch.float64,
    )
    forward_mus = torch.tensor(
        [[0.10, 0.15, 0.20], [0.25, 0.30, 0.35]],
        dtype=torch.float64,
    )
    parameters = NativeParameters(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=forward_mus,
    )
    raw = raw_from_native_parameters(parameters, geometry=geometry)
    recovered = bound_native_raw(raw, geometry=geometry)
    torch.testing.assert_close(recovered.dlvsqs, dlvsqs)
    torch.testing.assert_close(recovered.forward_vols, forward_vols)
    torch.testing.assert_close(recovered.eta, forward_mus)

    probabilities = torch.sigmoid(raw).reshape(2, 3, 7)
    expected_forward_probability = (
        (forward_vols - geometry.min_forward_vol)
        / (geometry.max_forward_vol - geometry.min_forward_vol)
    )
    expected_mu_probability = (
        (forward_mus - geometry.min_eta)
        / (geometry.max_eta - geometry.min_eta)
    )
    torch.testing.assert_close(probabilities[..., 5], expected_forward_probability)
    torch.testing.assert_close(probabilities[..., 6], expected_mu_probability)


def test_forward_mu_backbone_accumulates_mu_times_forward_vol() -> None:
    geometry = _small_geometry()
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlvsqs = torch.full((1, 3, 5), 0.04, dtype=torch.float64, requires_grad=True)
    forward_vols = torch.tensor(
        [[0.20, 0.25, 0.30]],
        dtype=torch.float64,
        requires_grad=True,
    )
    forward_mus = torch.tensor(
        [[0.10, 0.20, 0.30]],
        dtype=torch.float64,
        requires_grad=True,
    )
    surface = decoder.forward_batch(
        dlvsqs=dlvsqs,
        forward_vols=forward_vols,
        eta=forward_mus,
    )
    sqrt_dts = torch.sqrt(decoder.model_dts_y)
    torch.testing.assert_close(
        surface.total_vols,
        torch.cumsum(forward_vols * sqrt_dts, dim=-1),
    )
    torch.testing.assert_close(
        surface.backbone_total_vols,
        torch.cumsum(forward_mus * forward_vols * sqrt_dts, dim=-1),
    )
    loss = surface.backbone_total_vols.sum() + surface.densities.square().sum()
    loss.backward()
    assert forward_vols.grad is not None
    assert forward_mus.grad is not None
    assert dlvsqs.grad is not None
    assert torch.count_nonzero(forward_vols.grad).item() == 3
    assert torch.count_nonzero(forward_mus.grad).item() == 3
    assert torch.count_nonzero(dlvsqs.grad).item() > 0


def test_triangular_decoder_has_exact_maturity_mask() -> None:
    geometry = _small_geometry()
    config = TriangularPcaDecoderConfig(
        label="triangular-mask",
        geometry=geometry,
        state_size=6,
        state_sizes_by_expiry=(3, 2, 1),
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    model = LatentStateSanosModel(env, num_days=1, config=config)
    block_width = geometry.expiry_block_state_size
    with torch.no_grad():
        model.raw_decoder.I0.weight.fill_(1.0)
        model.raw_decoder.I0.bias.zero_()
        model.daily_states.zero_()

    index = torch.tensor([0], dtype=torch.long)
    baseline = model.raw_states_for_days(index)
    with torch.no_grad():
        model.daily_states[0, -1] = 2.0
    far_change = model.raw_states_for_days(index) - baseline
    torch.testing.assert_close(
        far_change[:, : 2 * block_width],
        torch.zeros_like(far_change[:, : 2 * block_width]),
    )
    torch.testing.assert_close(
        far_change[:, 2 * block_width :],
        torch.full_like(far_change[:, 2 * block_width :], 2.0),
    )

    with torch.no_grad():
        model.daily_states.zero_()
        model.daily_states[0, 0] = 3.0
    front_change = model.raw_states_for_days(index) - baseline
    torch.testing.assert_close(front_change, torch.full_like(front_change, 3.0))


def test_conditional_pca_recovers_exact_triangular_surface() -> None:
    geometry = _small_geometry()
    allocation = (2, 1, 1)
    config = TriangularPcaDecoderConfig(
        label="triangular-pca",
        geometry=geometry,
        state_size=sum(allocation),
        state_sizes_by_expiry=allocation,
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    model = LatentStateSanosModel(env, num_days=40, config=config)
    generator = torch.Generator().manual_seed(1729)
    states = torch.randn(40, 4, generator=generator, dtype=torch.float64)
    weight = torch.randn(21, 4, generator=generator, dtype=torch.float64)
    weight = weight * model.decoder_weight_mask
    bias = torch.randn(21, generator=generator, dtype=torch.float64)
    target = states @ weight.transpose(0, 1) + bias

    diagnostics = initialize_triangular_pca_decoder_(model, target)
    reconstructed = model.raw_states_for_days(torch.arange(40))
    torch.testing.assert_close(reconstructed, target, atol=1.0e-9, rtol=1.0e-9)
    assert diagnostics["mse"] < 1.0e-18
    assert diagnostics["explained_variance_fraction"] > 1.0 - 1.0e-12

