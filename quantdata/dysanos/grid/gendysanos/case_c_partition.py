"""Parallel fixed-notional direction models for conditional-arbitrage searches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("1.4")
@dataclass(frozen=True, kw_only=True)
class ParallelDirectionConfig:
    """Logical geometry shared by the conditional-direction models."""

    label: str
    source_cache_uid: str
    architecture: str
    state_features: int
    notionals: tuple[float, ...]
    partitions: int = 16
    kernels: int = 64
    deterministic_time: int = 0
    conditional_time: int = 6
    expiries: int = 7
    instruments_per_expiry: int = 21
    learn_oce_translation: bool = False
    action_radius: float = 1.0
    group: str = ""
    user_version: str = "0.1"
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=96,
        activation="SiLU",
        use_bias=True,
    )

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.label, str) and is_filename(self.label),
            lambda: f"'label' must be a filename, got {self.label!r}",
        )
        verify_inp(
            self.architecture
            in {
                "feedforward",
                "radial-feedforward",
                "gated-feedforward",
                "kernel",
                "partition",
            },
            lambda: f"invalid direction architecture {self.architecture!r}",
        )
        verify_inp(
            isinstance(self.source_cache_uid, str) and bool(self.source_cache_uid),
            "'source_cache_uid' must be non-empty",
        )
        verify_inp(
            isinstance(self.group, str) and not self.group.startswith("/"),
            lambda: f"'group' must be relative, got {self.group!r}",
        )
        for name in (
            "state_features",
            "partitions",
            "kernels",
            "expiries",
            "instruments_per_expiry",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        verify_inp(
            isinstance(self.deterministic_time, int)
            and isinstance(self.conditional_time, int)
            and 0 <= self.deterministic_time < self.conditional_time,
            "trading-time indices must be ordered nonnegative integers",
        )
        verify_inp(
            isinstance(self.learn_oce_translation, bool),
            lambda: (
                "'learn_oce_translation' must be bool, got "
                f"{self.learn_oce_translation!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.action_radius) and 0.0 < self.action_radius <= 1.0,
            lambda: (
                "'action_radius' must lie in (0,1], got "
                f"{self.action_radius!r}"
            ),
        )
        verify_inp(
            bool(self.notionals)
            and all(math.isfinite(value) and value > 0.0 for value in self.notionals)
            and all(
                left < right
                for left, right in zip(self.notionals, self.notionals[1:])
            ),
            lambda: (
                "notionals must be finite-positive and strictly increasing, "
                f"got {self.notionals!r}"
            ),
        )

    @property
    def action_components(self) -> int:
        return 2 * self.expiries * self.instruments_per_expiry

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


def unit_gross_long_short(
    logits: Tensor,
    *,
    expiries: int,
    instruments_per_expiry: int,
) -> Tensor:
    """Map unconstrained logits to separate nonnegative unit-gross positions."""

    components = 2 * expiries * instruments_per_expiry
    verify_inp(
        logits.shape[-1] == components,
        lambda: f"expected {components} action logits, got {logits.shape}",
    )
    return torch.softmax(logits, dim=-1).reshape(
        *logits.shape[:-1], 2, expiries, instruments_per_expiry
    )


@torch.no_grad()
def project_rows_to_simplex_(value: Tensor) -> None:
    """Euclidean-project every final-axis row onto the probability simplex."""

    verify_inp(
        value.ndim >= 1 and value.shape[-1] >= 1,
        lambda: f"simplex tensor has invalid shape {value.shape}",
    )
    sorted_value = torch.sort(value, dim=-1, descending=True).values
    cumulative = torch.cumsum(sorted_value, dim=-1) - 1.0
    sequence = torch.arange(
        1,
        value.shape[-1] + 1,
        dtype=value.dtype,
        device=value.device,
    )
    active = sorted_value - cumulative / sequence > 0.0
    rho = active.sum(dim=-1, keepdim=True).clamp_min(1)
    threshold = cumulative.gather(dim=-1, index=rho - 1) / rho.to(value.dtype)
    value.copy_(torch.clamp_min(value - threshold, 0.0))


@torch.no_grad()
def project_rows_to_l1_ball_(value: Tensor) -> None:
    """Euclidean-project direct nonnegative rows onto the unit L1 ball."""

    verify_inp(
        value.ndim >= 1 and value.shape[-1] >= 1,
        lambda: f"L1-ball tensor has invalid shape {value.shape}",
    )
    value.clamp_(min=0.0)
    gross = value.sum(dim=-1, keepdim=True)
    outside = gross.squeeze(-1) > 1.0
    if torch.any(outside):
        projected = value[outside].clone()
        project_rows_to_simplex_(projected)
        value[outside] = projected


class _ParallelDirectionBase(BaseModel):
    """Shared deterministic directions and state whitening."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ParallelDirectionConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        super().__init__(env)
        verify_inp(
            state_mean.shape == (config.state_features,),
            lambda: f"invalid state mean shape {state_mean.shape}",
        )
        verify_inp(
            state_whitener.shape
            == (config.state_features, config.state_features),
            lambda: f"invalid state whitener shape {state_whitener.shape}",
        )
        self.config = config
        self.deterministic_action = torch.nn.Parameter(
            torch.full(
                (len(config.notionals), config.action_components),
                1.0 / config.action_components,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        if config.learn_oce_translation:
            # Each independently optimized (notional, trading-time) portfolio
            # has one global OCE cash translation.  It is not state dependent.
            self.oce_translation = torch.nn.Parameter(
                torch.zeros(
                    (len(config.notionals), 2),
                    dtype=env.fdtype,
                    device=env.device,
                )
            )
        else:
            self.register_parameter("oce_translation", None)
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_whitener", state_whitener.detach().clone())

    def factor(self, state: Tensor) -> Tensor:
        """Return orthogonalized states using training-fold moments only."""

        assert state.ndim == 2 and state.shape[1] == self.config.state_features
        result = (state - self.state_mean[None]) @ self.state_whitener
        self.verify(lambda: torch.isfinite(result), "state factors are non-finite")
        return result

    def deterministic(self, paths: int) -> Tensor:
        """Return one independent deterministic unit-gross direction per notional."""

        action = self.deterministic_action.reshape(
            len(self.config.notionals),
            2,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )
        return self.config.action_radius * action[None].expand(
            paths, -1, -1, -1, -1
        )

    @torch.no_grad()
    def project_parameters_(self) -> None:
        project_rows_to_simplex_(self.deterministic_action)


@version("1.2")
class ParallelFeedForwardDirections(_ParallelDirectionBase):
    """Independent feed-forward conditional directions for every notional."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ParallelDirectionConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        verify_inp(config.architecture == "feedforward", "invalid feed-forward config")
        super().__init__(
            env,
            config=config,
            state_mean=state_mean,
            state_whitener=state_whitener,
        )
        self.conditional = torch.nn.ModuleList(
            [
                FeedForward(
                    env,
                    num_input=config.state_features,
                    num_output=config.action_components,
                    config=config.network_config,
                )
                for _ in config.notionals
            ]
        )
        self.initialize_output_layers()

    @torch.no_grad()
    def initialize_output_layers(self) -> None:
        """Cold-start every notional at the same neutral unit-gross direction."""

        for network in self.conditional:
            output = network.In[-1] if network.In else network.I0
            output.weight.zero_()
            if output.bias is not None:
                output.bias.zero_()

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return actions with shape ``(S,N,2,Q,E,I)`` and raw logits."""

        factor = self.factor(state)
        logits = torch.stack([network(factor) for network in self.conditional], dim=1)
        conditional = self.config.action_radius * unit_gross_long_short(
            logits,
            expiries=self.config.expiries,
            instruments_per_expiry=self.config.instruments_per_expiry,
        )
        deterministic = self.deterministic(state.shape[0])
        action = torch.stack((deterministic, conditional), dim=2)
        self.verify(lambda: torch.isfinite(action), "feed-forward actions are non-finite")
        return action, logits


@version("1.0")
class ParallelRadialFeedForwardDirections(_ParallelDirectionBase):
    """Independent zero-capable feed-forward actions for every objective head.

    The network emits one signed position per instrument. Dividing by one plus
    its gross norm keeps the action strictly inside the unit L1 ball while
    retaining the identity derivative at the zero action. This avoids both a
    no-trade softmax coordinate and a hard clipping boundary.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ParallelDirectionConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        verify_inp(
            config.architecture == "radial-feedforward",
            "invalid radial feed-forward config",
        )
        super().__init__(
            env,
            config=config,
            state_mean=state_mean,
            state_whitener=state_whitener,
        )
        signed_components = config.expiries * config.instruments_per_expiry
        self.conditional = torch.nn.ModuleList(
            [
                FeedForward(
                    env,
                    num_input=config.state_features,
                    num_output=signed_components,
                    config=config.network_config,
                )
                for _ in config.notionals
            ]
        )
        self.initialize_zero_action()

    @torch.no_grad()
    def initialize_zero_action(self) -> None:
        """Start every deterministic and conditional head at no trade."""

        self.deterministic_action.zero_()
        for network in self.conditional:
            output = network.In[-1] if network.In else network.I0
            output.weight.zero_()
            if output.bias is not None:
                output.bias.zero_()

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return radial unit-ball actions and unconstrained signed outputs."""

        factor = self.factor(state)
        raw = torch.stack([network(factor) for network in self.conditional], dim=1)
        gross = raw.abs().sum(dim=-1, keepdim=True)
        signed = self.config.action_radius * raw / (1.0 + gross)
        long = 0.5 * (signed.abs() + signed)
        short = 0.5 * (signed.abs() - signed)
        conditional = torch.stack((long, short), dim=2).reshape(
            state.shape[0],
            len(self.config.notionals),
            2,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )
        deterministic = self.deterministic(state.shape[0])
        action = torch.stack((deterministic, conditional), dim=2)
        self.verify(lambda: torch.isfinite(raw), "radial raw actions are non-finite")
        self.verify(lambda: torch.isfinite(action), "radial actions are non-finite")
        return action, raw


@version("1.3")
class GatedParallelFeedForwardDirections(_ParallelDirectionBase):
    """Independent statewise directions with a learned zero-to-one activity."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ParallelDirectionConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        verify_inp(
            config.architecture == "gated-feedforward",
            "invalid gated feed-forward config",
        )
        super().__init__(
            env,
            config=config,
            state_mean=state_mean,
            state_whitener=state_whitener,
        )
        self.conditional = torch.nn.ModuleList(
            [
                FeedForward(
                    env,
                    num_input=config.state_features,
                    num_output=config.action_components,
                    config=config.network_config,
                )
                for _ in config.notionals
            ]
        )
        self.gate = torch.nn.ModuleList(
            [
                FeedForward(
                    env,
                    num_input=config.state_features,
                    num_output=1,
                    config=config.network_config,
                )
                for _ in config.notionals
            ]
        )
        self.initialize_output_layers()

    @torch.no_grad()
    def initialize_output_layers(self) -> None:
        """Start from neutral directions with small, recoverable activity."""

        for network in self.conditional:
            output = network.In[-1] if network.In else network.I0
            output.weight.zero_()
            if output.bias is not None:
                output.bias.zero_()
        for network in self.gate:
            output = network.In[-1] if network.In else network.I0
            output.weight.zero_()
            if output.bias is not None:
                output.bias.fill_(-3.0)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return unit directions multiplied by learned statewise activity."""

        factor = self.factor(state)
        raw_direction = torch.stack(
            [network(factor) for network in self.conditional], dim=1
        )
        direction = unit_gross_long_short(
            raw_direction,
            expiries=self.config.expiries,
            instruments_per_expiry=self.config.instruments_per_expiry,
        )
        raw_gate = torch.stack(
            [network(factor) for network in self.gate], dim=1
        ).squeeze(-1)
        activity = torch.sigmoid(raw_gate)
        conditional = (
            self.config.action_radius
            * direction
            * activity[..., None, None, None]
        )
        deterministic = self.deterministic(state.shape[0])
        action = torch.stack((deterministic, conditional), dim=2)
        self.verify(
            lambda: torch.isfinite(action),
            "gated feed-forward actions are non-finite",
        )
        return action, torch.cat((raw_direction, raw_gate[..., None]), dim=-1)


@version("1.5")
class ParallelKernelDirections(_ParallelDirectionBase):
    """Independent local Gaussian-kernel portfolios for every direction head."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ParallelDirectionConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
        centers: Tensor,
        inverse_bandwidth: Tensor,
    ) -> None:
        verify_inp(config.architecture == "kernel", "invalid kernel config")
        super().__init__(
            env,
            config=config,
            state_mean=state_mean,
            state_whitener=state_whitener,
        )
        verify_inp(
            centers.shape == (config.kernels, config.state_features),
            lambda: f"invalid kernel-center shape {centers.shape}",
        )
        verify_inp(
            inverse_bandwidth.shape == (config.kernels,),
            lambda: f"invalid inverse-bandwidth shape {inverse_bandwidth.shape}",
        )
        verify_inp(
            bool(torch.isfinite(centers).all().item())
            and bool(torch.isfinite(inverse_bandwidth).all().item())
            and bool(torch.all(inverse_bandwidth > 0.0).item()),
            "kernel basis must be finite with positive inverse bandwidths",
        )
        self.template_action = torch.nn.Parameter(
            torch.empty(
                len(config.notionals),
                config.kernels,
                config.action_components,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.centers = torch.nn.Parameter(centers.detach().clone())
        self.log_inverse_bandwidth = torch.nn.Parameter(
            torch.log(inverse_bandwidth.detach().clone())
        )
        self.register_buffer("initial_centers", centers.detach().clone())
        self.register_buffer(
            "initial_inverse_bandwidth", inverse_bandwidth.detach().clone()
        )
        self.initialize_kernel()

    @torch.no_grad()
    def initialize_kernel(self) -> None:
        """Cold-start distinct local directions with small recoverable activity."""

        self.template_action.exponential_(1.0)
        project_rows_to_simplex_(self.template_action)
        self.template_action.mul_(0.05)
        self.deterministic_action.mul_(0.05)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        project_rows_to_l1_ball_(self.deterministic_action)
        project_rows_to_l1_ball_(self.template_action)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return Gaussian-kernel mixtures and their normalized weights."""

        factor = self.factor(state)
        difference = factor[:, None, :] - self.centers[None, :, :]
        distance_square = difference.square().sum(dim=-1)
        inverse_bandwidth = torch.exp(self.log_inverse_bandwidth)
        log_kernel = (
            -0.5
            * distance_square
            * inverse_bandwidth.square()[None, :]
        )
        kernel_weight = torch.softmax(log_kernel, dim=-1)
        template = self.template_action.reshape(
            len(self.config.notionals),
            self.config.kernels,
            2,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )
        conditional = self.config.action_radius * torch.einsum(
            "sk,nkqei->snqei", kernel_weight, template
        )
        deterministic = self.deterministic(state.shape[0])
        action = torch.stack((deterministic, conditional), dim=2)
        self.verify(
            lambda: torch.isfinite(inverse_bandwidth),
            "kernel inverse bandwidths are non-finite",
        )
        self.verify(lambda: torch.isfinite(action), "kernel actions are non-finite")
        reported_weight = kernel_weight[:, None].expand(
            -1, len(self.config.notionals), -1
        ).clone()
        return action, reported_weight


@version("1.2")
class LearnedPartitionDirections(_ParallelDirectionBase):
    """Learned soft state partitions with independent local portfolio directions."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ParallelDirectionConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        verify_inp(config.architecture == "partition", "invalid partition config")
        super().__init__(
            env,
            config=config,
            state_mean=state_mean,
            state_whitener=state_whitener,
        )
        self.router = torch.nn.ModuleList(
            [
                FeedForward(
                    env,
                    num_input=config.state_features,
                    num_output=config.partitions,
                    config=config.network_config,
                )
                for _ in config.notionals
            ]
        )
        self.expert_action = torch.nn.Parameter(
            torch.empty(
                len(config.notionals),
                config.partitions,
                config.action_components,
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.initialize_partition()

    @torch.no_grad()
    def initialize_partition(self) -> None:
        """Start with a uniform router and distinct random feasible directions."""

        for network in self.router:
            output = network.In[-1] if network.In else network.I0
            output.weight.zero_()
            if output.bias is not None:
                output.bias.zero_()
        self.expert_action.exponential_(1.0)
        project_rows_to_simplex_(self.expert_action)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        super().project_parameters_()
        project_rows_to_simplex_(self.expert_action)

    def responsibilities(self, factor: Tensor) -> tuple[Tensor, Tensor]:
        """Return learned state-only soft assignments and their logits."""

        logits = torch.stack([network(factor) for network in self.router], dim=1)
        probability = torch.softmax(logits, dim=-1)
        self.verify(
            lambda: torch.isfinite(probability),
            "partition responsibilities are non-finite",
        )
        return probability, logits

    def expert_actions(self) -> Tensor:
        """Return local directions with shape ``(N,K,Q,E,I)``."""

        return self.expert_action.reshape(
            len(self.config.notionals),
            self.config.partitions,
            2,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return interpolated actions, responsibilities, and router logits."""

        factor = self.factor(state)
        responsibility, logits = self.responsibilities(factor)
        conditional = self.config.action_radius * torch.einsum(
            "snk,nkqei->snqei", responsibility, self.expert_actions()
        )
        deterministic = self.deterministic(state.shape[0])
        action = torch.stack((deterministic, conditional), dim=2)
        self.verify(lambda: torch.isfinite(action), "partition actions are non-finite")
        return action, responsibility, logits


__all__ = [
    "GatedParallelFeedForwardDirections",
    "LearnedPartitionDirections",
    "ParallelDirectionConfig",
    "ParallelFeedForwardDirections",
    "ParallelKernelDirections",
    "ParallelRadialFeedForwardDirections",
    "project_rows_to_l1_ball_",
    "project_rows_to_simplex_",
    "unit_gross_long_short",
]
