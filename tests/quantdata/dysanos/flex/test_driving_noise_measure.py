from __future__ import annotations

import torch

from quantdata.dysanos.flex.driving_noise_measure import (
    DrivingNoiseTransport,
    DrivingNoiseTransportConfig,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_zero_initialized_transport_is_identity_with_zero_kl() -> None:
    env = BaseEnvironment(device="cpu", precision=64)
    config = DrivingNoiseTransportConfig(
        state_size=3,
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=8),
    )
    transport = DrivingNoiseTransport(
        env,
        config=config,
        state_mean=torch.zeros(3, dtype=torch.float64),
        state_scale=torch.ones(3, dtype=torch.float64),
    )
    state = torch.randn(5, 3, dtype=torch.float64)
    normals = torch.randn(5, 4, dtype=torch.float64)

    transformed, conditional_kl, shift, log_scale = transport(
        state,
        torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
        normals,
    )

    assert torch.equal(transformed, normals)
    assert torch.equal(shift, torch.zeros_like(shift))
    assert torch.equal(log_scale, torch.zeros_like(log_scale))
    assert torch.equal(conditional_kl, torch.zeros_like(conditional_kl))
