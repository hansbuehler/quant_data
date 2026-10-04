"""Sampleable state-conditional transports of daily Gaussian driving noise."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class DrivingNoiseTransportConfig:
    """Architecture and bounds for a conditional diagonal-Gaussian law."""

    state_size: int
    maximum_shift: float = 2.0
    maximum_log_scale: float = 1.0
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=256,
    )

    def __post_init__(self) -> None:
        verify_inp(self.state_size > 0, lambda: "transport state size must be positive")
        verify_inp(
            math.isfinite(self.maximum_shift) and self.maximum_shift > 0.0,
            lambda: "maximum transport shift must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.maximum_log_scale) and self.maximum_log_scale > 0.0,
            lambda: "maximum transport log scale must be finite-positive",
        )

    @property
    def noise_size(self) -> int:
        """Surface-state innovations plus one orthogonal spot innovation."""

        return self.state_size + 1


class DrivingNoiseTransport(BaseModel):
    """Map iid normals into a conditional Gaussian law with known path KL."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: DrivingNoiseTransportConfig,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        if state_mean.shape != (config.state_size,) or state_scale.shape != (
            config.state_size,
        ):
            raise ValueError("transport state normalization has the wrong shape")
        if not bool(
            torch.all(torch.isfinite(state_mean))
            and torch.all(torch.isfinite(state_scale) & (state_scale > 0.0))
        ):
            raise ValueError("transport state normalization must be finite-positive")
        self.config = config
        self.register_buffer(
            "state_mean",
            state_mean.to(dtype=env.fdtype, device=env.device).detach().clone(),
        )
        self.register_buffer(
            "state_scale",
            state_scale.to(dtype=env.fdtype, device=env.device).detach().clone(),
        )
        self.network = FeedForward(
            env,
            config.state_size + 1,
            2 * config.noise_size,
            config.network,
        )
        final = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        current_state: torch.Tensor,
        normalized_time: torch.Tensor,
        base_normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw transformed noise and return per-path conditional KL to P."""

        if current_state.shape[-1] != self.config.state_size:
            raise ValueError("transport current state has the wrong width")
        if normalized_time.shape != current_state.shape[:-1]:
            raise ValueError("transport time coordinate has the wrong shape")
        expected_noise_shape = current_state.shape[:-1] + (self.config.noise_size,)
        if base_normals.shape != expected_noise_shape:
            raise ValueError(
                f"transport normals have shape {base_normals.shape}, expected "
                f"{expected_noise_shape}"
            )
        context = torch.cat(
            (
                (current_state - self.state_mean) / self.state_scale,
                normalized_time.unsqueeze(-1),
            ),
            dim=-1,
        )
        raw_shift, raw_log_scale = torch.chunk(self.network(context), 2, dim=-1)
        shift = self.config.maximum_shift * torch.tanh(
            raw_shift / self.config.maximum_shift
        )
        log_scale = self.config.maximum_log_scale * torch.tanh(
            raw_log_scale / self.config.maximum_log_scale
        )
        scale = torch.exp(log_scale)
        transformed = shift + scale * base_normals
        conditional_kl = 0.5 * torch.sum(
            shift.square() + scale.square() - 1.0 - 2.0 * log_scale,
            dim=-1,
        )
        return transformed, conditional_kl, shift, log_scale


__all__ = ["DrivingNoiseTransport", "DrivingNoiseTransportConfig"]
