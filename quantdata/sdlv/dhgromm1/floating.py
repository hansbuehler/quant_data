"""Floating-contract gains, causal recurrent actions and deterministic dated vectors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch._higher_order_ops.scan import scan

from quantdata.sdlv.dhgromm1.contracts import canonical_gains, floating_gains, scenario_identity
from quantdata.sdlv.dhgromm1.feedback import FeedbackHedge
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.sdlv.dhgromm1.observable import ObservableConfig
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.utility import torch_utility

LABELS = ("empty", "long", "short")
SIGNS = (0, 1, -1)


def floating_quotes(prices: torch.Tensor) -> torch.Tensor:
    """Current spot level followed by current floating option/forward prices."""
    return torch.cat((prices[..., :1], prices[..., 1:] / prices[..., :1]), dim=-1)


def fit_return_means(gains: np.ndarray) -> np.ndarray:
    """Training-only means, separately for each opening date and control slot.

    Apply the same linear correction to spot and equivalent zero-strike
    claims; otherwise centering could create artificial deterministic spread
    trades. These marginal means are not conditional-drift estimates or a
    change to the simulation's Q law.
    """
    if gains.ndim != 3 or gains.shape[1:] != (20, 21) or len(gains) < 4 or len(gains) % 2 or not np.isfinite(gains).all():
        raise ValueError("fit centering on finite complete training antithetic pairs")
    return gains.mean(axis=0, dtype=np.float64)


def read_floating_sample(directory: Path, *, paths: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve source FP64 while converting existing paths; no simulation/replay."""
    quotes, gains, payoffs = [], [], []
    identity = scenario_identity(directory)
    completed = 0
    for path in sorted(directory.glob("[0-9]*.npz")):
        if int(path.stem) != completed:
            raise ValueError("scenario chunks must be contiguous in their original antithetic order")
        with np.load(path, allow_pickle=False) as saved:
            price, ret, payoff = (saved[k] for k in ("prices", "returns", "payoff"))
        if len(price) % 2 or payoff.shape != (len(price),) or not np.isfinite(payoff).all():
            raise ValueError("invalid original training population")
        quotes.append(price)
        gains.append(canonical_gains(price, ret, identity=identity))
        payoffs.append(payoff)
        completed += len(price)
    if completed != paths:
        raise ValueError(f"expected {paths} paths, received {completed}")
    return tuple(np.concatenate(x) for x in (quotes, gains, payoffs))


@version("0.1", dependencies=[ObservableConfig, torch_utility])
@dataclass(frozen=True, kw_only=True)
class FloatingConfig(ObservableConfig):
    group: str = "sdlv/dhgromm1/floating"
    gain_convention: str = "floating-contract-gain-centered-controls-v1"
    dated_actions: str = "deterministic-quantity-vectors"

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(self.gain_convention == "floating-contract-gain-centered-controls-v1"
                   and self.dated_actions == "deterministic-quantity-vectors",
                   lambda: "incompatible floating gain or dated action semantics")
        verify_inp(np.isfinite(self.trade_l1) and self.trade_l1 >= 0
                   and np.isfinite(self.trade_l2) and self.trade_l2 >= 0
                   and np.isfinite(self.risk_aversion) and self.risk_aversion > 0,
                   lambda: "costs must be finite/nonnegative and risk aversion finite/positive")


class FloatingHedge(FeedbackHedge):
    """Generic affine+deep action of current features/memory/book, plus a0/a10.

    Actions are floating-contract quantities. Thus the book adds a/S_open
    actual underlying-contract units, never a/option_premium. No future return
    or day-10 fixing is exposed to the policy. All expiry-return labels and
    their frozen training means enter only AFTER the recurrent decisions.
    """

    def __init__(self, env: BaseEnvironment, *, config: FloatingConfig,
                 price_mean: torch.Tensor, price_scale: torch.Tensor, return_mean: torch.Tensor) -> None:
        super().__init__(env, config=config, price_mean=price_mean, price_scale=price_scale)
        if return_mean.shape != (20, 21):
            raise ValueError("centering needs one frozen mean per opening day and instrument")
        self.register_buffer("return_mean", return_mean.to(env.device, env.fdtype).detach().clone())
        self.day0 = torch.nn.Parameter(torch.zeros(21, device=env.device, dtype=env.fdtype))
        self.day10 = torch.nn.Parameter(torch.zeros(21, device=env.device, dtype=env.fdtype))

    def action(self, state: torch.Tensor) -> torch.Tensor:
        generic = super().action(state)
        return (generic + (state[:, :1] == 0).to(state.dtype) * self.day0
                + (state[:, :1] == .5).to(state.dtype) * self.day10)

    def _body(self, carry: tuple[torch.Tensor, ...], inputs: tuple[torch.Tensor, ...]
              ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        hidden, cell, book, previous = carry
        features, spot, descriptors, changes, insert = inputs
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
        next_book = book + insert[None, :, None] * (trade / spot)[:, None, :]
        return (next_hidden.clone(), next_cell.clone(), next_book.clone(), trade.clone()), trade.clone()

    def forward(self, prices: torch.Tensor, changes: torch.Tensor, *, returns: torch.Tensor,
                eager: bool = False) -> HedgeOutput:
        batch = prices.shape[0]
        if prices.shape != (batch, 21, 21) or changes.shape != (batch, 20, 420) or returns.shape != (batch, 20, 21):
            raise ValueError("floating hedge received incompatible path/mark/return geometry")
        current = (floating_quotes(prices[:, :20]) - self.price_mean) / self.price_scale
        days = torch.arange(20, device=prices.device, dtype=prices.dtype)
        time = days[None, :, None] / 20
        features = torch.cat((time.expand(batch, -1, -1), current), dim=-1)
        tenors = torch.stack((torch.ones_like(days), days * 0 + 5, torch.minimum(days * 0 + 10, 25 - days)), -1)
        ratios = torch.exp(.20 * (tenors[:, :, None] / 252).sqrt() * prices.new_tensor(self.config.strike_levels))
        strike = torch.cat((days[:, None] * 0, days[:, None] * 0 + 1, ratios.flatten(1), days[:, None].expand(-1, 4) * 0), -1)
        terminal = 20 - days
        dte = torch.cat((terminal[:, None].expand(-1, 2), tenors.repeat_interleave(5, -1), terminal[:, None], tenors), -1)
        descriptors = torch.cat((strike, dte / 20), -1)[None].expand(batch, -1, -1)
        inputs = tuple(x.transpose(0, 1).contiguous() for x in
                       (features, prices[:, :20, :1], descriptors, changes)) + (self.insert,)
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
        gain = (trades * (returns - self.return_mean)).sum((-1, -2))
        penalty = (self.config.trade_l1 * trades.abs() + self.config.trade_l2 * trades.square()).sum((-1, -2))
        return HedgeOutput(gain, penalty, trades, trades)


@version("0.1", dependencies=[FloatingConfig, torch_utility])
@dataclass(frozen=True, kw_only=True)
class FloatingTripleConfig:
    label: str
    group: str = "sdlv/dhgromm1/floating"
    user_version: str = "0.1"
    policy: FloatingConfig = FloatingConfig(label="policy")


class FloatingTriple(BaseModel):
    """Separate empty/long/short policies; costs inside the same powexp OCE."""

    def __init__(self, env: BaseEnvironment, *, config: FloatingTripleConfig,
                 mean: torch.Tensor, scale: torch.Tensor, return_mean: torch.Tensor, payoff_mean: float) -> None:
        super().__init__(env)
        self.config = config
        for label in LABELS:
            setattr(self, label, FloatingHedge(env, config=replace(config.policy, label=label),
                price_mean=mean, price_scale=scale, return_mean=return_mean))
        self.register_buffer("payoff_mean", torch.tensor(payoff_mean, device=env.device, dtype=env.fdtype))
        self._utility, _, _ = torch_utility("powexp", config.policy.risk_aversion)

    def losses(self, results: tuple[HedgeOutput, ...], payoff: torch.Tensor) -> torch.Tensor:
        losses = []
        for label, sign, result in zip(LABELS, SIGNS, results, strict=True):
            cash = getattr(self, label).translation
            net = result.gain + sign * (payoff - self.payoff_mean) - result.penalty
            losses.append(-(cash + self._utility(net - cash, torch.zeros_like(net)).mean()))
        return torch.stack(losses)

    def policy_functions(self, *, compile_model: bool) -> tuple:
        # Approximate Triton division amplified roundoff in nearly constant
        # normalized forward quotes after feature scaling. Match eager IEEE
        # division without changing the model, its units, or its learned weights.
        return tuple(torch.compile(getattr(self, label), fullgraph=True,
                                   options={"eager_numerics.division_rounding":True}) if compile_model else getattr(self, label)
                     for label in LABELS)
