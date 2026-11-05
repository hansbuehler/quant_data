"""Causal price/history-only hedging; no simulator latent state enters a policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch._higher_order_ops.scan import scan

from quantdata.sdlv.dhgromm1.events import EventFeedbackConfig, EventFeedbackHedge, EventJointConfig
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.sdlv.dhgromm1.joint import JointHedges, load_joint_checkpoint
from quantdata.torch import BaseEnvironment


def market_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack marks of already opened contracts only: 3,990 floats per path."""
    offsets = np.r_[0, np.cumsum(np.arange(20) * 21)]
    index = np.zeros((20, 420), dtype=np.int64)
    valid = np.zeros_like(index, dtype=bool)
    for day in range(1, 20):
        index[day, :day * 21] = np.arange(offsets[day], offsets[day + 1])
        valid[day, :day * 21] = True
    return offsets, index, valid


@version("0.1", dependencies=[EventFeedbackConfig])
@dataclass(frozen=True, kw_only=True)
class ObservableConfig(EventFeedbackConfig):
    # Parent's extra-feature slot now means 21 strike ratios, 21 tenors and
    # one current hedge P&L, NOT a simulator state or a learned latent factor.
    latent_width: int = 43
    strike_levels: tuple[float, ...] = (-2., -1.5, -1., -.5, -.25)

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(self.latent_width == 43 and len(self.strike_levels) == 5,
                   lambda: "observable policy requires 42 current contract descriptors plus current hedge P&L")


class ObservableHedge(EventFeedbackHedge):
    """Current quotes + causal contract book + LSTM, with exact-date heads.

    The carry retains all past contract units, including settled cohorts.
    Their marks freeze at settlement, so sum(q * (mark_t - opening_mark))
    gives current self-financing hedge P&L without accessing future returns.
    No explicit fixing is supplied. At day 10 its head sees S_10 as current
    spot; later LSTM memory can retain observations, but cannot see ahead.
    """

    def _body(self, carry: tuple[torch.Tensor, ...], inputs: tuple[torch.Tensor, ...]
              ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        hidden, cell, book, previous = carry
        features, marks, descriptors, changes, insert = inputs
        pnl = (book.flatten(1) * changes).sum(-1, keepdim=True)
        feedback = torch.cat((descriptors, pnl, book.flatten(1), previous), dim=-1)
        recurrent = torch.cat((features, feedback), dim=-1)
        gates = (torch.nn.functional.linear(recurrent, self.recurrence.weight_ih, self.recurrence.bias_ih)
                 + torch.nn.functional.linear(hidden, self.recurrence.weight_hh, self.recurrence.bias_hh))
        i, f, g, o = gates.chunk(4, dim=-1)
        next_cell = f.sigmoid() * cell + i.sigmoid() * g.tanh()
        next_hidden = o.sigmoid() * next_cell.tanh()
        state = torch.cat((features, next_hidden, feedback), dim=-1)
        modulation = 1 + torch.tanh(self.gate(features[:, :1]) - self.reference_gate(features[:, :1]))
        state = torch.cat((state[:, :1], state[:, 1:] * modulation), dim=-1)
        trade = self.action(state)
        next_book = book + insert[None, :, None] * (trade / marks[:, :1])[:, None, :]
        return (next_hidden.clone(), next_cell.clone(), next_book.clone(), trade.clone()), trade.clone()

    def forward(self, prices: torch.Tensor, changes: torch.Tensor, *, returns: torch.Tensor,
                eager: bool = False) -> HedgeOutput:
        batch = prices.shape[0]
        if prices.shape != (batch, 21, 21) or changes.shape != (batch, 20, 420):
            raise ValueError("expected 21-date quotes and 20-date historical contract mark changes")
        if returns.shape != (batch, 20, 21):
            raise ValueError("expiry returns have the wrong geometry")
        current = (prices[:, :20] - self.price_mean) / self.price_scale
        days = torch.arange(20, device=prices.device, dtype=prices.dtype)
        time = days[None, :, None] / 20
        features = torch.cat((time.expand(batch, -1, -1), current), dim=-1)
        # New-contract definitions are fixed ex ante or functions of CURRENT
        # spot. They do not use S_10, a future spot, or an expiry payoff.
        tenors = torch.stack((torch.ones_like(days), days * 0 + 5, torch.minimum(days * 0 + 10, 25 - days)), -1)
        levels = prices.new_tensor(self.config.strike_levels)
        ratios = torch.exp(.20 * (tenors[:, :, None] / 252).sqrt() * levels)
        strike = torch.cat((days[:, None] * 0, days[:, None] * 0 + 1, ratios.flatten(1), days[:, None].expand(-1, 4) * 0), -1)
        terminal = 20 - days
        dte = torch.cat((terminal[:, None].expand(-1, 2), tenors.repeat_interleave(5, -1), terminal[:, None], tenors), -1)
        descriptors = torch.cat((strike, dte / 20), -1)[None].expand(batch, -1, -1)
        inputs = tuple(x.transpose(0, 1).contiguous() for x in
                       (features, prices[:, :20], descriptors, changes)) + (self.insert,)
        initial = (prices.new_zeros((batch, self.config.hidden_size)), prices.new_zeros((batch, self.config.hidden_size)),
                   prices.new_zeros((batch, 20, 21)), prices.new_zeros((batch, 21)))
        if eager:
            carry, values = initial, []
            for day in range(20):
                carry, value = self._body(carry, tuple(x[day] for x in inputs))
                values.append(value)
            trades = torch.stack(values).transpose(0, 1)
        else:
            _, values = scan(self._body, initial, inputs)
            trades = values.transpose(0, 1)
        # These are labels used ONLY after all causal decisions are formed.
        gain = (trades * returns).sum((-1, -2))
        penalty = (self.config.trade_l1 * trades.abs() + self.config.trade_l2 * trades.square()).sum((-1, -2))
        return HedgeOutput(gain, penalty, trades, trades)


@version("0.1", dependencies=[EventJointConfig, ObservableConfig])
@dataclass(frozen=True, kw_only=True)
class ObservableJointConfig(EventJointConfig):
    long: ObservableConfig = ObservableConfig(label="long")
    short: ObservableConfig = ObservableConfig(label="short")


class ObservableJointHedges(JointHedges):
    policy_type = ObservableHedge


def initialize_observable(source: Path, *, env: BaseEnvironment, config: ObservableJointConfig,
                          ) -> tuple[ObservableJointHedges, dict[str, int]]:
    """Explicit partial warm start, removing every simulator-state input column.

    Keep price, memory, historical-position and previous-trade columns, plus
    all compatible deeper weights/translations. New observable columns start
    at zero weight. Dropping state dependence cannot preserve old outputs;
    record that fact rather than treating this as an exact continuation.
    """
    old_config = EventJointConfig(label="bid-ask")
    saved = load_joint_checkpoint(source, config=old_config)
    state = saved["best_state"]
    model = ObservableJointHedges(env, config=config,
        means={k:state[f"{k}.price_mean"] for k in ("long", "short")},
        scales={k:state[f"{k}.price_scale"] for k in ("long", "short")})
    target = model.state_dict()
    for label in ("long", "short"):
        old_width, new_width = old_config.long.latent_width, config.long.latent_width
        hidden = config.long.hidden_size
        for key, destination in target.items():
            if not key.startswith(label + "."):
                continue
            value = state[key].to(device=destination.device, dtype=destination.dtype)
            # Inputs: [time/prices, memory if action, extra, cohort units, previous trade].
            is_recurrence = key.endswith("recurrence.weight_ih")
            is_action = (key.endswith("trade_head.weight") or key.endswith("deep_head.I0.weight")
                         or (".event_heads." in key and (key.endswith("affine.weight") or key.endswith("deep.I0.weight"))))
            is_gate = (".gate.In.0." in key or ".reference_gate.In.0." in key)
            if is_recurrence or is_action:
                prefix = 22 if is_recurrence else 22 + hidden
                destination.zero_()
                destination[:, :prefix] = value[:, :prefix]
                destination[:, prefix + new_width:] = value[:, prefix + old_width:]
            elif is_gate:
                prefix = 21 + hidden  # Gate excludes the raw time coordinate.
                destination.zero_()
                destination[:prefix] = value[:prefix]
                destination[prefix + new_width:] = value[prefix + old_width:]
            elif destination.shape == value.shape:
                destination.copy_(value)
            else:
                raise ValueError(f"unmapped warm-start tensor: {key}: {value.shape} -> {destination.shape}")
    model.load_state_dict(target)
    return model, {k:saved["selected_epoch"] for k in ("long", "short")}
