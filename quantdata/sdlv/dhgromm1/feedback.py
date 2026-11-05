"""Full-state, contract-inventory feedback hedge with a differentiable scan."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch._higher_order_ops.scan import scan

from quantdata.sdlv.dhgromm1.contracts import RETURN_CONVENTION
from quantdata.sdlv.dhgromm1.hedge import HedgeConfig, HedgeOutput, RecurrentHedge
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def packed_geometry(*, horizon: int = 20, support: int = 202, future_rows: int = 24,
                    interior: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lossless ragged packing, with expired fixed-expiry rows masked out."""
    offsets = np.r_[0, np.cumsum([support + (future_rows - t) * interior + 1
                                 for t in range(horizon)])]
    width = support + future_rows * interior + 1
    index = np.zeros((horizon, width), dtype=np.int64)
    valid = np.zeros_like(index, dtype=bool)
    for t in range(horizon):
        slots = np.r_[np.arange(support),
                      np.arange(support + t * interior, width - 1), width - 1]
        index[t, slots] = np.arange(offsets[t], offsets[t + 1])
        valid[t, slots] = True
    return offsets, index, valid


def contract_expiries() -> np.ndarray:
    """Actual expiries by opening day and instrument, not rolling slot IDs."""
    expiry = np.full((20, 21), 20, dtype=np.int64)
    for t in range(20):
        tenors = [1, 5, min(10, 25 - t)]
        expiry[t, 2:17] = np.repeat(np.array(tenors) + t, 5)
        expiry[t, 18:] = np.array(tenors) + t
    return expiry


@version("0.2", dependencies=[HedgeConfig])
@dataclass(frozen=True, kw_only=True)
class FeedbackConfig:
    """A distinct architecture identity; old checkpoints require explicit migration."""

    label: str
    group: str = "sdlv/dhgromm1"
    user_version: str = "0.1"
    instruments: int = 21
    return_convention: str = RETURN_CONVENTION
    horizon_days: int = 20
    hidden_size: int = 32
    latent_width: int = 5003
    action_deep: FeedForwardConfig = FeedForwardConfig(activation="Tanh", num_depth=2, num_width=32)
    time_gate: FeedForwardConfig = FeedForwardConfig(activation="Tanh", num_depth=1, num_width=8)
    risk_aversion: float = 100.
    trade_l1: float = 1.e-4
    trade_l2: float = 1.e-8

    def __post_init__(self) -> None:
        verify_inp(bool(self.label) and self.instruments == 21 and self.horizon_days == 20,
                   lambda: "feedback hedge requires the existing 20-day, 21-instrument experiment")
        verify_inp(self.hidden_size > 0 and self.latent_width > 0,
                   lambda: "hidden and latent dimensions must be positive")
        verify_inp(self.return_convention == RETURN_CONVENTION, lambda: "feedback requires floating-contract gains")


class FeedbackHedge(BaseModel):
    """Scan LSTM, action, and live cohort book jointly, without Python time steps.

    A book entry is actual contract units, indexed by opening date and instrument.
    Settled contracts disappear before the next action. A floating quantity a
    converts to actual contract units a/S_open, never a/option_premium.
    Realized future gains enter only final P&L.
    """

    def __init__(self, env: BaseEnvironment, *, config: FeedbackConfig,
                 price_mean: torch.Tensor, price_scale: torch.Tensor) -> None:
        super().__init__(env)
        self.config = config
        self.register_buffer("price_mean", price_mean.to(env.device, env.fdtype).clone())
        self.register_buffer("price_scale", price_scale.to(env.device, env.fdtype).clone())
        extra = config.latent_width + 20 * 21 + 21
        self.recurrence = torch.nn.LSTMCell(22 + extra, config.hidden_size,
                                           device=env.device, dtype=env.fdtype)
        action_width = 22 + config.hidden_size + extra
        self.trade_head = torch.nn.Linear(action_width, 21, device=env.device, dtype=env.fdtype)
        self.deep_head = FeedForward(env, num_input=action_width, num_output=21, config=config.action_deep)
        self.reference_trade_head = copy.deepcopy(self.trade_head).requires_grad_(False)
        self.reference_deep_head = copy.deepcopy(self.deep_head).requires_grad_(False)
        self.gate = FeedForward(env, num_input=1, num_output=action_width - 1, config=config.time_gate)
        self.reference_gate = copy.deepcopy(self.gate).requires_grad_(False)
        self.translation = torch.nn.Parameter(torch.zeros((), device=env.device, dtype=env.fdtype))
        expiry = torch.as_tensor(contract_expiries(), device=env.device)
        days = torch.arange(20, device=env.device)
        # Entry [decision time, opening date, instrument]. Expiry settlement
        # precedes the decision at that date. The unopened rows remain zero.
        self.register_buffer("live", (expiry[None] > days[:, None, None]).to(env.fdtype))
        self.register_buffer("insert", torch.eye(20, device=env.device, dtype=env.fdtype))

    def _body(self, carry: tuple[torch.Tensor, ...], inputs: tuple[torch.Tensor, ...]
              ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        hidden, cell, book, previous = carry
        features, marks, latent, live, insert = inputs
        live_book = book * live[None]
        inventory = live_book.flatten(1)
        feedback = torch.cat((latent, inventory, previous), dim=-1)
        recurrent = torch.cat((features, feedback), dim=-1)
        gates = (torch.nn.functional.linear(recurrent, self.recurrence.weight_ih, self.recurrence.bias_ih)
                 + torch.nn.functional.linear(hidden, self.recurrence.weight_hh, self.recurrence.bias_hh))
        i, f, g, o = gates.chunk(4, dim=-1)
        next_cell = f.sigmoid() * cell + i.sigmoid() * g.tanh()
        next_hidden = o.sigmoid() * next_cell.tanh()
        state = torch.cat((features, next_hidden, feedback), dim=-1)
        time = features[:, :1]
        # Identity at migration, positive feature modulation; raw t bypasses
        # the gate. No sigmoid/tanh is applied to the final trade action.
        modulation = 1 + torch.tanh(self.gate(time) - self.reference_gate(time))
        state = torch.cat((state[:, :1], state[:, 1:] * modulation), dim=-1)
        trade = self.action(state)
        next_book = live_book + insert[None, :, None] * (trade / marks[:, :1])[:, None, :]
        # scan also saves cell/gate intermediates for its backward scan. Clone
        # every carry so AOT's saved intermediates cannot alias carry outputs.
        # Clones preserve derivatives; no feedback is detached.
        return (next_hidden.clone(), next_cell.clone(), next_book.clone(), trade.clone()), trade.clone()

    def action(self, state: torch.Tensor) -> torch.Tensor:
        """Shared all-date affine-plus-deep action; subclasses may add dated heads."""
        return (self.trade_head(state) - self.reference_trade_head(state)
                + self.deep_head(state) - self.reference_deep_head(state))

    def forward(self, prices: torch.Tensor, latent: torch.Tensor, *, returns: torch.Tensor,
                eager: bool = False) -> HedgeOutput:
        batch = prices.shape[0]
        if latent.shape != (batch, 20, self.config.latent_width):
            raise ValueError("latent state has the wrong batch/time/state geometry")
        current = (prices[:, :20] - self.price_mean) / self.price_scale
        time = torch.arange(20, device=prices.device, dtype=prices.dtype)[None, :, None] / 20
        features = torch.cat((time.expand(batch, -1, -1), current), dim=-1)
        inputs = (features.transpose(0, 1).contiguous(), prices[:, :20].transpose(0, 1).contiguous(),
                  latent.transpose(0, 1).contiguous(), self.live, self.insert)
        initial = (prices.new_zeros((batch, self.config.hidden_size)),
                   prices.new_zeros((batch, self.config.hidden_size)),
                   prices.new_zeros((batch, 20, 21)), prices.new_zeros((batch, 21)))
        if eager:
            carry, values = initial, []
            for t in range(20):
                carry, value = self._body(carry, tuple(x[t] for x in inputs))
                values.append(value)
            trades = torch.stack(values).transpose(0, 1)
        else:
            _, values = scan(self._body, initial, inputs)
            trades = values.transpose(0, 1)
        gain = (trades * returns).sum(dim=(-1, -2))
        penalty = (self.config.trade_l1 * trades.abs() + self.config.trade_l2 * trades.square()).sum(dim=(-1, -2))
        return HedgeOutput(gain, penalty, trades, trades)


def migrate_policy(old: RecurrentHedge, new: FeedbackHedge) -> None:
    """Explicit, output-preserving extension; never pretend schemas are identical."""
    if old.config.hidden_size != new.config.hidden_size or old.config.instruments != 21:
        raise ValueError("incompatible source architecture for feedback migration")
    source, target = old.state_dict(), new.state_dict()
    expanded = {"recurrence.weight_ih", "trade_head.weight", "reference_trade_head.weight",
                "deep_head.I0.weight", "reference_deep_head.I0.weight"}
    with torch.no_grad():
        for key, value in source.items():
            if key in expanded:
                target[key].zero_()
                target[key][:, :value.shape[1]].copy_(value)
            elif key in target:
                if target[key].shape != value.shape:
                    raise ValueError(f"incompatible migration tensor {key}")
                target[key].copy_(value)
        new.load_state_dict(target)
