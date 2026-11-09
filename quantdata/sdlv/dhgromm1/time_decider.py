"""Causal floating hedge with a rich shared clock and no day-10 action node."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.dhgromm1.feedback import FeedbackHedge
from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    FloatingConfig,
    FloatingHedge,
    FloatingTriple,
    FloatingTripleConfig,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility


@version("0.1", dependencies=[FloatingConfig])
@dataclass(frozen=True, kw_only=True)
class TimeDeciderConfig(FloatingConfig):
    group: str = "sdlv/dhgromm1/time-decider"
    time_frequencies: int = 10
    action_deep: FeedForwardConfig = FeedForwardConfig(activation="Tanh", num_depth=2, num_width=64)
    time_gate: FeedForwardConfig = FeedForwardConfig(activation="Tanh", num_depth=2, num_width=32)

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(self.time_frequencies >= 10, lambda: "the shared clock must span all twenty daily decisions")


class TimeDeciderHedge(FloatingHedge):
    """Affine+deep decider on current state and a Fourier clock; only a0 is explicit.

    The same time features and weights act on EVERY date. There is no fixing
    indicator, fixed day-10 action, or scalar benchmark injected at that date.
    Fourier modes span any deterministic schedule on the daily observation
    grid, so capacity alone does not prevent learning a concentrated action.
    """

    def __init__(self, env: BaseEnvironment, *, config: TimeDeciderConfig,
                 price_mean: torch.Tensor, price_scale: torch.Tensor, return_mean: torch.Tensor) -> None:
        super().__init__(env, config=config, price_mean=price_mean, price_scale=price_scale, return_mean=return_mean)
        del self.day10
        width = self.trade_head.in_features + 2 * config.time_frequencies + 1
        self.trade_head = torch.nn.Linear(width, 21, device=env.device, dtype=env.fdtype)
        self.deep_head = FeedForward(env, num_input=width, num_output=21, config=config.action_deep)
        self.reference_trade_head = copy.deepcopy(self.trade_head).requires_grad_(False)
        self.reference_deep_head = copy.deepcopy(self.deep_head).requires_grad_(False)
        self.register_buffer("time_modes", torch.arange(1, config.time_frequencies + 1, device=env.device, dtype=env.fdtype))

    def clock(self, time: torch.Tensor) -> torch.Tensor:
        phase = 2 * torch.pi * time * self.time_modes
        return torch.cat((time.square(), phase.sin(), phase.cos()), dim=-1)

    def action(self, state: torch.Tensor) -> torch.Tensor:
        extended = torch.cat((state, self.clock(state[:, :1])), dim=-1)
        return FeedbackHedge.action(self, extended) + (state[:, :1] == 0).to(state.dtype) * self.day0


@version("0.1", dependencies=[FloatingTripleConfig, TimeDeciderConfig])
@dataclass(frozen=True, kw_only=True)
class TimeDeciderTripleConfig(FloatingTripleConfig):
    group: str = "sdlv/dhgromm1/time-decider"
    policy: TimeDeciderConfig = TimeDeciderConfig(label="policy")


class TimeDeciderTriple(FloatingTriple):
    """Three independent policies; unchanged OCE and recurrent accounting."""

    def __init__(self, env: BaseEnvironment, *, config: TimeDeciderTripleConfig,
                 mean: torch.Tensor, scale: torch.Tensor, return_mean: torch.Tensor, payoff_mean: float) -> None:
        BaseModel.__init__(self, env)
        self.config = config
        for label in LABELS:
            setattr(self, label, TimeDeciderHedge(env, config=replace(config.policy, label=label),
                price_mean=mean, price_scale=scale, return_mean=return_mean))
        self.register_buffer("payoff_mean", torch.tensor(payoff_mean, device=env.device, dtype=env.fdtype))
        self._utility, _, _ = torch_utility("powexp", config.policy.risk_aversion)


@version("0.1", dependencies=[TimeDeciderTripleConfig])
@dataclass(frozen=True, kw_only=True)
class TimeDeciderWithA10Config(TimeDeciderTripleConfig):
    """Matched rich-clock architecture, adding ONLY the deterministic a10 vector."""

    group: str = "sdlv/dhgromm1/time-decider-with-a10"


class TimeDeciderWithA10Hedge(TimeDeciderHedge):
    def __init__(self, env: BaseEnvironment, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.day10 = torch.nn.Parameter(torch.zeros(21, device=env.device, dtype=env.fdtype))

    def action(self, state: torch.Tensor) -> torch.Tensor:
        return super().action(state) + (state[:, :1] == .5).to(state.dtype) * self.day10


class TimeDeciderWithA10Triple(TimeDeciderTriple):
    def __init__(self, env: BaseEnvironment, *, config: TimeDeciderWithA10Config,
                 mean: torch.Tensor, scale: torch.Tensor, return_mean: torch.Tensor, payoff_mean: float) -> None:
        BaseModel.__init__(self, env)
        self.config = config
        for label in LABELS:
            setattr(self, label, TimeDeciderWithA10Hedge(env, config=replace(config.policy, label=label),
                price_mean=mean, price_scale=scale, return_mean=return_mean))
        self.register_buffer("payoff_mean", torch.tensor(payoff_mean, device=env.device, dtype=env.fdtype))
        self._utility, _, _ = torch_utility("powexp", config.policy.risk_aversion)
