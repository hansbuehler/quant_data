"""Unrestricted state-local conditional generator for FLEX Q experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@dataclass(frozen=True, kw_only=True)
class FullStateGeneratorConfig:
    """Geometry of one dense ``G(epsilon, spot)`` front-state generator."""

    front_state_size: int
    base_noise_std: float = 0.01
    maximum_increment: float = 16.0
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=4,
        num_width=512,
    )

    def __post_init__(self) -> None:
        if (
            self.front_state_size <= 0
            or self.base_noise_std <= 0.0
            or self.maximum_increment <= 0.0
        ):
            raise ValueError("full state generator dimensions and scales must be positive")


@dataclass(frozen=True, kw_only=True)
class JointFullStateGeneratorConfig:
    """Geometry of one dense ``G(h_0, spot, epsilon)`` generator."""

    front_state_size: int
    base_noise_std: float = 0.01
    maximum_increment: float = 16.0
    context_scale_floor: float = 1.0e-3
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=4,
        num_width=512,
    )

    def __post_init__(self) -> None:
        if (
            self.front_state_size <= 0
            or self.base_noise_std <= 0.0
            or self.maximum_increment <= 0.0
            or self.context_scale_floor <= 0.0
        ):
            raise ValueError("joint full generator dimensions and scales must be positive")


@dataclass(frozen=True, kw_only=True)
class JointUncompressedFrontGeneratorConfig:
    """Geometry of direct ``G(u_0, spot, epsilon)`` in raw SANOS coordinates."""

    context_state_size: int
    front_state_size: int
    context_scale_floor: float = 1.0e-3
    output_scale_floor: float = 1.0e-3
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=4,
        num_width=512,
    )

    def __post_init__(self) -> None:
        if (
            self.context_state_size <= 0
            or self.front_state_size <= 0
            or self.front_state_size > self.context_state_size
            or self.context_scale_floor <= 0.0
            or self.output_scale_floor <= 0.0
        ):
            raise ValueError(
                "uncompressed front generator dimensions and scales must be positive"
            )


class FullStateConditionalGenerator(BaseModel):
    r"""Generate a future front state with a dense non-invertible neural map.

    Every output may depend nonlinearly on every Gaussian coordinate and the
    front-month spot rank. The bounded residual protects the native decoder
    during early training; it does not impose triangular or Jacobian constraints.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        current_front_state: torch.Tensor,
        config: FullStateGeneratorConfig,
    ) -> None:
        super().__init__(env)
        if current_front_state.shape != (config.front_state_size,):
            raise ValueError("full generator received an invalid current front state")
        self.config = config
        self.register_buffer(
            "current_front_state",
            current_front_state.detach().clone(),
        )
        self.network = FeedForward(
            env,
            config.front_state_size + 1,
            config.front_state_size,
            config.network,
        )
        final = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        *,
        current_front_states: torch.Tensor,
        spot_rank: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the state-local dense generator to common Gaussian paths."""

        width = self.config.front_state_size
        if (
            current_front_states.shape != (1, width)
            or spot_rank.ndim != 2
            or spot_rank.shape[0] != 1
            or epsilon.shape != (spot_rank.shape[1], width)
        ):
            raise ValueError("full state generator inputs are inconsistent")
        context = torch.cat((epsilon, spot_rank[0, :, None]), dim=-1)
        raw_increment = self.config.base_noise_std * epsilon + self.network(context)
        increment = self.config.maximum_increment * torch.tanh(
            raw_increment / self.config.maximum_increment
        )
        return current_front_states[:, None] + increment[None]


class JointFullStateConditionalGenerator(BaseModel):
    r"""Generate future front states with a shared dense state-conditioned map."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        current_front_states: torch.Tensor,
        config: JointFullStateGeneratorConfig,
    ) -> None:
        super().__init__(env)
        if (
            current_front_states.ndim != 2
            or current_front_states.shape[1] != config.front_state_size
            or len(current_front_states) < 2
        ):
            raise ValueError("joint full generator received invalid current front states")
        self.config = config
        context_center = torch.mean(current_front_states, dim=0)
        context_scale = torch.std(
            current_front_states,
            dim=0,
            correction=0,
        ).clamp_min(config.context_scale_floor)
        self.register_buffer("context_center", context_center.detach().clone())
        self.register_buffer("context_scale", context_scale.detach().clone())
        self.network = FeedForward(
            env,
            2 * config.front_state_size + 1,
            config.front_state_size,
            config.network,
        )
        final = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        *,
        current_front_states: torch.Tensor,
        spot_rank: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the shared ``G(h_0, spot, epsilon)`` to every supplied state."""

        width = self.config.front_state_size
        if (
            current_front_states.ndim != 2
            or current_front_states.shape[1] != width
            or spot_rank.ndim != 2
            or spot_rank.shape[0] != len(current_front_states)
            or epsilon.shape != (spot_rank.shape[1], width)
        ):
            raise ValueError("joint full generator inputs are inconsistent")
        states = current_front_states[:, None].expand(-1, spot_rank.shape[1], -1)
        normalized_states = (
            (current_front_states - self.context_center) / self.context_scale
        )[:, None].expand_as(states)
        noise = epsilon[None].expand(len(current_front_states), -1, -1)
        context = torch.cat((normalized_states, spot_rank[..., None], noise), dim=-1)
        raw_increment = self.config.base_noise_std * noise + self.network(context)
        increment = self.config.maximum_increment * torch.tanh(
            raw_increment / self.config.maximum_increment
        )
        return states + increment


class JointUncompressedFrontConditionalGenerator(BaseModel):
    r"""Generate the next front block as unrestricted raw SANOS coordinates.

    The conditioning state is the complete current native raw surface, not the
    compressed front factor block.  The network directly represents
    ``u_t^1 = G(u_0, S_t / S_0, epsilon)``: there is no residual addition and no
    bounded-increment transform.  Output standardization is an invertible
    affine coordinate choice and therefore does not restrict the generated raw
    front state.  Only the next front raw block is generated; passing it through
    ``bound_native_raw`` and ``FlexSanosDecoder`` preserves the decoder's
    structural static-arbitrage guarantees.
    """

    generates_uncompressed_front_raw = True

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        current_raw_states: torch.Tensor,
        config: JointUncompressedFrontGeneratorConfig,
    ) -> None:
        super().__init__(env)
        if (
            current_raw_states.ndim != 2
            or current_raw_states.shape[1] != config.context_state_size
            or len(current_raw_states) < 2
        ):
            raise ValueError(
                "uncompressed front generator received invalid current raw states"
            )
        self.config = config
        context_center = torch.mean(current_raw_states, dim=0)
        context_scale = torch.std(
            current_raw_states,
            dim=0,
            correction=0,
        ).clamp_min(config.context_scale_floor)
        output_center = torch.mean(
            current_raw_states[:, : config.front_state_size],
            dim=0,
        )
        output_scale = torch.std(
            current_raw_states[:, : config.front_state_size],
            dim=0,
            correction=0,
        ).clamp_min(config.output_scale_floor)
        self.register_buffer("context_center", context_center.detach().clone())
        self.register_buffer("context_scale", context_scale.detach().clone())
        self.register_buffer("output_center", output_center.detach().clone())
        self.register_buffer("output_scale", output_scale.detach().clone())
        self.network = FeedForward(
            env,
            config.context_state_size + config.front_state_size + 1,
            config.front_state_size,
            config.network,
        )

    @torch.no_grad()
    def rebase_context_normalization(
        self,
        current_raw_states: torch.Tensor,
    ) -> None:
        """Re-estimate context moments without changing the represented map.

        The first affine layer is transformed together with the normalization
        buffers.  Therefore every raw-state, spot, and noise input has the same
        network output before and after rebasing, up to floating-point error.
        """

        if (
            current_raw_states.ndim != 2
            or current_raw_states.shape[1] != self.config.context_state_size
            or len(current_raw_states) < 2
        ):
            raise ValueError("normalization reference contains invalid raw states")
        first_layer = self.network.I0
        if first_layer.bias is None:
            raise ValueError("context normalization rebasing requires a first-layer bias")

        new_center = torch.mean(current_raw_states, dim=0)
        new_scale = torch.std(
            current_raw_states,
            dim=0,
            correction=0,
        ).clamp_min(self.config.context_scale_floor)
        old_center = self.context_center.detach().clone()
        old_scale = self.context_scale.detach().clone()
        context_width = self.config.context_state_size
        old_context_weight = first_layer.weight[:, :context_width].detach().clone()

        # old_z = (new_scale / old_scale) * new_z
        #         + (new_center - old_center) / old_scale.
        first_layer.weight[:, :context_width].copy_(
            old_context_weight * (new_scale / old_scale)[None]
        )
        first_layer.bias.add_(
            old_context_weight
            @ ((new_center - old_center) / old_scale)
        )
        self.context_center.copy_(new_center)
        self.context_scale.copy_(new_scale)

    def forward(
        self,
        *,
        current_raw_states: torch.Tensor,
        normalized_spot: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate direct ``G(u_0, S_t / S_0, epsilon)`` for every path."""

        context_width = self.config.context_state_size
        front_width = self.config.front_state_size
        if (
            current_raw_states.ndim != 2
            or current_raw_states.shape[1] != context_width
            or normalized_spot.ndim != 2
            or normalized_spot.shape[0] != len(current_raw_states)
            or epsilon.shape != (normalized_spot.shape[1], front_width)
        ):
            raise ValueError("uncompressed front generator inputs are inconsistent")
        paths = normalized_spot.shape[1]
        normalized_context = (
            (current_raw_states - self.context_center) / self.context_scale
        )[:, None].expand(-1, paths, -1)
        noise = epsilon[None].expand(len(current_raw_states), -1, -1)
        context = torch.cat(
            (normalized_context, normalized_spot[..., None], noise),
            dim=-1,
        )
        return self.output_center + self.output_scale * self.network(context)


__all__ = [
    "FullStateConditionalGenerator",
    "FullStateGeneratorConfig",
    "JointFullStateConditionalGenerator",
    "JointFullStateGeneratorConfig",
    "JointUncompressedFrontConditionalGenerator",
    "JointUncompressedFrontGeneratorConfig",
]
