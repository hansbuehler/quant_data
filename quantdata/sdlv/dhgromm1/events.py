"""Exact-date additive action networks, with output-preserving warm migration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.dhgromm1.feedback import FeedbackConfig, FeedbackHedge
from quantdata.sdlv.dhgromm1.joint import (
    JointConfig,
    JointHedges,
    load_joint_checkpoint,
    load_joint_model,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class EventActionConfig:
    label: str
    num_input: int
    num_output: int = 21
    deep: FeedForwardConfig = FeedForwardConfig(activation="Tanh", num_depth=2, num_width=32)
    group: str = "sdlv/dhgromm1/events"
    user_version: str = "0.1"

    def __post_init__(self) -> None:
        verify_inp(bool(self.label) and self.num_input > 0 and self.num_output > 0,
                   lambda: "event action requires a label and positive input/output dimensions")


class EventAction(BaseModel):
    """Uncapped affine plus deep map, minus frozen copies of both initial maps."""

    def __init__(self, env: BaseEnvironment, *, config: EventActionConfig) -> None:
        super().__init__(env)
        self.config = config
        self.affine = torch.nn.Linear(config.num_input, config.num_output, device=env.device, dtype=env.fdtype)
        self.deep = FeedForward(env, num_input=config.num_input, num_output=config.num_output, config=config.deep)
        self.reference_affine = copy.deepcopy(self.affine).requires_grad_(False)
        self.reference_deep = copy.deepcopy(self.deep).requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.affine(state) - self.reference_affine(state) + self.deep(state) - self.reference_deep(state)


@version("0.1", dependencies=[FeedbackConfig, EventActionConfig])
@dataclass(frozen=True, kw_only=True)
class EventFeedbackConfig(FeedbackConfig):
    event_days: tuple[int, ...] = (0, 10)
    event_deep: FeedForwardConfig = FeedForwardConfig(activation="Tanh", num_depth=2, num_width=32)

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(self.event_days == (0, 10), lambda: "this experiment adds independent day-0 and day-10 heads")


class EventFeedbackHedge(FeedbackHedge):
    """Add two independent networks at their exact dates inside the same scan.

    The hard date indicator cannot be learned away by the smooth shared time
    gate. Each added action reads current prices/state, updated LSTM memory,
    previous trades and the actual live book. Combined trades enter both the
    self-financing account and trade regularizer once, including their effect
    on all subsequent recurrent states and actions.
    """

    def __init__(self, env: BaseEnvironment, *, config: EventFeedbackConfig,
                 price_mean: torch.Tensor, price_scale: torch.Tensor) -> None:
        super().__init__(env, config=config, price_mean=price_mean, price_scale=price_scale)
        width = self.trade_head.in_features
        self.event_heads = torch.nn.ModuleDict({str(day): EventAction(env, config=EventActionConfig(
            label=f"{config.label}-day{day}", num_input=width, num_output=config.instruments, deep=config.event_deep))
            for day in config.event_days})

    def action(self, state: torch.Tensor) -> torch.Tensor:
        trade = super().action(state)
        # This static two-head loop is unrolled inside the compiled scan body;
        # there is no Python loop over trading dates or per-step host transfer.
        for day in self.config.event_days:
            active = (state[:, :1] == day / self.config.horizon_days).to(state.dtype)
            trade = trade + active * self.event_heads[str(day)](state)
        return trade


@version("0.1", dependencies=[JointConfig, EventFeedbackConfig])
@dataclass(frozen=True, kw_only=True)
class EventJointConfig(JointConfig):
    long: EventFeedbackConfig = EventFeedbackConfig(label="long")
    short: EventFeedbackConfig = EventFeedbackConfig(label="short")


class EventJointHedges(JointHedges):
    policy_type = EventFeedbackHedge


def initialize_events(source: Path, *, env: BaseEnvironment, config: EventJointConfig,
                      ) -> tuple[EventJointHedges, dict[str, int]]:
    """Explicit architecture migration from the selected base training pair.

    Never treat an older architecture as a matching restart. Shared weights,
    translations, feature scales and frozen references are copied unchanged;
    the two new heads each initially contribute exactly zero. The optimizer
    restarts; exact continuation is from the new event checkpoint instead.
    """
    old_config = JointConfig(label="bid-ask")
    saved = load_joint_checkpoint(source, config=old_config)
    old = load_joint_model(source, config=old_config, env=env)
    model = EventJointHedges(env, config=config,
        means={label: getattr(old, label).price_mean for label in ("long", "short")},
        scales={label: getattr(old, label).price_scale for label in ("long", "short")})
    result = model.load_state_dict(old.state_dict(), strict=False)
    if result.unexpected_keys or any(".event_heads." not in name for name in result.missing_keys):
        raise ValueError(f"incompatible event migration: {result}")
    return model, {label: saved["selected_epoch"] for label in ("long", "short")}


def exercise_event_heads(model: EventJointHedges) -> None:
    """Make both dated actions nonzero for the CUDA value/gradient regression."""
    with torch.no_grad():
        for policy in (model.long, model.short):
            for head in policy.event_heads.values():
                head.affine.weight.add_(torch.randn_like(head.affine.weight) * 1e-6)
                head.affine.bias.add_(.0001)
