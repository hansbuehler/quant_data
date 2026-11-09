"""Actual proportional cash fees inside ordinary OCE, with causal cost feedback.

The previous outside-utility L1/L2 regularizers are NOT used. Floating actions
are changes in contract quantities, not holdings or cash invested in premiums.
Fees are paid when a trade is opened; natural cash settlement is not a trade.
The fee basis is explicit because 1 bp of underlying notional and 1 bp of
option premium are financially different conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch._higher_order_ops.scan import scan

from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    FloatingConfig,
    FloatingHedge,
    FloatingTriple,
    FloatingTripleConfig,
    floating_quotes,
)
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.utility import torch_utility


@version("0.1", dependencies=[FloatingConfig])
@dataclass(frozen=True, kw_only=True)
class TransactionCostPolicyConfig(FloatingConfig):
    group: str = "sdlv/dhgromm1/transaction-cost"
    trade_l1: float = 0.
    trade_l2: float = 0.
    transaction_cost_rate: float = 1e-4
    transaction_cost_basis: str = "cash_value"
    transaction_cost_timing: str = "opening_trades_cash_settlement_free"

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(self.trade_l1 == self.trade_l2 == 0,
                   lambda: "L1/L2 regularizers and old cost coefficients must be zero")
        verify_inp(np.isfinite(self.transaction_cost_rate) and self.transaction_cost_rate >= 0,
                   lambda: "transaction cost rate must be finite and nonnegative")
        verify_inp(self.transaction_cost_basis in {"underlying_notional", "cash_value"},
                   lambda: "choose underlying_notional or cash_value explicitly")
        verify_inp(self.transaction_cost_timing == "opening_trades_cash_settlement_free",
                   lambda: "unsupported fee timing; no silent extra settlement charge")


def proportional_fee(trades: torch.Tensor, bases: torch.Tensor, *, rate: float) -> torch.Tensor:
    """Cash fee per path/date; bases are 1 or the contemporaneous price/S_open."""
    return (rate*trades.abs()*bases).sum(-1)


class TransactionCostHedge(FloatingHedge):
    """Same trainable tensors; an extra NONTRAINABLE scan carry tracks paid cash.

    Cash fees affect the running P&L input from the following decision onward.
    No future return/payoff or eventual fee enters the current action. They
    also enter terminal net wealth once, through HedgeOutput.penalty.
    """

    def _body(self, carry: tuple[torch.Tensor, ...], inputs: tuple[torch.Tensor, ...]
              ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        hidden, cell, book, previous, fees_paid = carry
        features, spot, descriptors, changes, insert, fee_basis = inputs
        marked_pnl = (book.flatten(1)*changes).sum(-1, keepdim=True)-fees_paid
        feedback = torch.cat((descriptors, marked_pnl, book.flatten(1), previous), -1)
        recurrent = torch.cat((features, feedback), -1)
        gates = (torch.nn.functional.linear(recurrent, self.recurrence.weight_ih, self.recurrence.bias_ih)
                 + torch.nn.functional.linear(hidden, self.recurrence.weight_hh, self.recurrence.bias_hh))
        i, f, g, o = gates.chunk(4, -1)
        next_cell = f.sigmoid()*cell+i.sigmoid()*g.tanh()
        next_hidden = o.sigmoid()*next_cell.tanh()
        state = torch.cat((features, next_hidden, feedback), -1)
        modulation = 1+torch.tanh(self.gate(features[:, :1])-self.reference_gate(features[:, :1]))
        state = torch.cat((state[:, :1], state[:, 1:]*modulation), -1)
        trade = self.action(state)
        next_book = book+insert[None, :, None]*(trade/spot)[:, None]
        fees = fees_paid+proportional_fee(trade, fee_basis, rate=self.config.transaction_cost_rate)[:, None]
        return (next_hidden.clone(), next_cell.clone(), next_book.clone(), trade.clone(), fees.clone()), trade.clone()

    def forward(self, prices: torch.Tensor, changes: torch.Tensor, *, returns: torch.Tensor,
                eager: bool = False) -> HedgeOutput:
        batch = len(prices)
        if prices.shape != (batch, 21, 21) or changes.shape != (batch, 20, 420) or returns.shape != (batch, 20, 21):
            raise ValueError("transaction-cost policy has incompatible price/mark/return geometry")
        quotes = floating_quotes(prices[:, :20])
        current = (quotes-self.price_mean)/self.price_scale
        days = torch.arange(20, device=prices.device, dtype=prices.dtype)
        time = days[None, :, None]/20
        features = torch.cat((time.expand(batch, -1, -1), current), -1)
        tenors = torch.stack((torch.ones_like(days), days*0+5, torch.minimum(days*0+10, 25-days)), -1)
        ratios = torch.exp(.20*(tenors[:, :, None]/252).sqrt()*prices.new_tensor(self.config.strike_levels))
        strike = torch.cat((days[:, None]*0, days[:, None]*0+1, ratios.flatten(1), days[:, None].expand(-1, 4)*0), -1)
        terminal = 20-days
        dte = torch.cat((terminal[:, None].expand(-1, 2), tenors.repeat_interleave(5, -1), terminal[:, None], tenors), -1)
        descriptors = torch.cat((strike, dte/20), -1)[None].expand(batch, -1, -1)
        # Actual trade units are a/S_open. For cash-value fees multiply a by
        # the contemporaneous instrument price/S_open, including spot -> 1.
        bases = (torch.ones_like(quotes) if self.config.transaction_cost_basis == "underlying_notional"
                 else prices[:, :20]/prices[:, :20, :1])
        inputs = tuple(x.transpose(0, 1).contiguous() for x in
            (features, prices[:, :20, :1], descriptors, changes))+(self.insert, bases.transpose(0, 1).contiguous())
        initial = (prices.new_zeros((batch, self.config.hidden_size)), prices.new_zeros((batch, self.config.hidden_size)),
            prices.new_zeros((batch, 20, 21)), prices.new_zeros((batch, 21)), prices.new_zeros((batch, 1)))
        if eager:
            carry, values = initial, []
            for day in range(20):
                carry, value = self._body(carry, tuple(x[day] for x in inputs))
                values.append(value)
            trades = torch.stack(values).transpose(0, 1)
        else:
            carry, values = scan(self._body, initial, inputs)
            trades = values.transpose(0, 1)
        gain = (trades*(returns-self.return_mean)).sum((-1, -2))
        return HedgeOutput(gain, carry[-1][:, 0], trades, trades)


@version("0.1", dependencies=[FloatingTripleConfig, TransactionCostPolicyConfig])
@dataclass(frozen=True, kw_only=True)
class TransactionCostTripleConfig(FloatingTripleConfig):
    group: str = "sdlv/dhgromm1/transaction-cost"
    policy: TransactionCostPolicyConfig = TransactionCostPolicyConfig(label="policy")


class TransactionCostTriple(FloatingTriple):
    """Three separate policies, ordinary OCE of wealth NET of cash fees."""

    def __init__(self, env: BaseEnvironment, *, config: TransactionCostTripleConfig,
                 mean: torch.Tensor, scale: torch.Tensor, return_mean: torch.Tensor, payoff_mean: float) -> None:
        BaseModel.__init__(self, env)
        self.config = config
        for label in LABELS:
            setattr(self, label, TransactionCostHedge(env, config=replace(config.policy, label=label),
                price_mean=mean, price_scale=scale, return_mean=return_mean))
        self.register_buffer("payoff_mean", torch.tensor(payoff_mean, device=env.device, dtype=env.fdtype))
        self._utility, _, _ = torch_utility("powexp", config.policy.risk_aversion)
