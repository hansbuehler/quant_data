"""Causal recurrent trading and power-exponential optimized certainty equivalents."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.dhgromm1.contracts import RETURN_CONVENTION
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility


@version("0.7", dependencies=[torch_utility])
@dataclass(frozen=True, kw_only=True)
class HedgeConfig:
    """Logical identity for the recurrent affine-plus-deep hedge."""

    label: str
    group: str = "sdlv/dhgromm1"
    user_version: str = "0.1"
    instruments: int = 21
    gain_mode: str = "expiry_trades"
    return_convention: str = RETURN_CONVENTION
    horizon_days: int = 20
    hidden_size: int = 32
    action_deep: FeedForwardConfig = FeedForwardConfig(activation="Tanh", num_depth=2, num_width=32)
    risk_aversion: float = 100.0
    trade_l1: float = 1.0e-4
    trade_l2: float = 1.0e-8

    def __post_init__(self) -> None:
        verify_inp(bool(self.label), lambda: "a hedge label is required")
        verify_inp(self.gain_mode in {"expiry_trades", "daily_positions"},
                   lambda: f"unknown gain mode {self.gain_mode!r}")
        verify_inp(self.return_convention == RETURN_CONVENTION, lambda: "hedge requires floating-contract gains")
        verify_inp(self.instruments > 0 and self.horizon_days > 0 and self.hidden_size > 0,
                   lambda: "instrument, horizon and hidden dimensions must be positive")
        verify_inp(np.isfinite(self.risk_aversion) and self.risk_aversion > 0,
                   lambda: "risk aversion must be finite and positive")
        verify_inp(np.isfinite(self.trade_l1) and self.trade_l1 >= 0
                   and np.isfinite(self.trade_l2) and self.trade_l2 >= 0,
                   lambda: "trade-penalty coefficients must be finite and nonnegative")


class HedgeOutput(NamedTuple):
    """Cash trading gain, regularizer, executed trades and held positions."""

    gain: torch.Tensor
    penalty: torch.Tensor
    trades: torch.Tensor
    positions: torch.Tensor


class RecurrentHedge(BaseModel):
    """Choose daily trade increments from current prices, time and LSTM state."""

    def __init__(self, env: BaseEnvironment, *, config: HedgeConfig,
                 price_mean: torch.Tensor, price_scale: torch.Tensor) -> None:
        super().__init__(env)
        if price_mean.shape != (config.instruments,) or price_scale.shape != (config.instruments,):
            raise ValueError("price normalization must have one value per traded instrument")
        if not bool(torch.isfinite(price_mean).all() and torch.isfinite(price_scale).all()
                    and (price_scale > 0).all()):
            raise ValueError("price normalization must be finite with strictly positive scales")
        self.config = config
        self.register_buffer("price_mean", price_mean.to(device=env.device, dtype=env.fdtype).detach().clone())
        self.register_buffer("price_scale", price_scale.to(device=env.device, dtype=env.fdtype).detach().clone())
        self.recurrence = torch.nn.LSTMCell(config.instruments + 1, config.hidden_size,
                                           device=env.device, dtype=env.fdtype)
        action_size = config.hidden_size + config.instruments + 1
        self.trade_head = torch.nn.Linear(action_size, config.instruments,
                                          device=env.device, dtype=env.fdtype)
        self.deep_head = FeedForward(env, num_input=action_size, num_output=config.instruments,
                                     config=config.action_deep)
        self.translation = torch.nn.Parameter(torch.zeros((), device=env.device, dtype=env.fdtype))
        if config.action_deep.num_depth < 1:
            raise ValueError("the deep action component must have at least one hidden layer")
        # Freeze an initial random head, not the input: derivatives through the
        # shared state remain mathematically consistent with the output formula.
        self.reference_trade_head = copy.deepcopy(self.trade_head).requires_grad_(False)
        self.reference_deep_head = copy.deepcopy(self.deep_head).requires_grad_(False)

    def _native_lstm(self) -> torch.nn.LSTM:
        """Execution-only fused view sharing the canonical checkpoint parameters.

        No new weights are owned here: keeping the view out of ``_modules``
        preserves the existing Cell parameter names and optimizer identities.
        Constructing on meta also avoids consuming RNG state during a restart.
        """
        view = self.__dict__.get("_native_lstm_view")
        if view is None:
            # Keep forward AND backward in IEEE FP32. A temporary forward-only
            # flag can leave cuDNN's backward using its default TF32 arithmetic.
            torch.backends.cudnn.allow_tf32 = False
            view = torch.nn.LSTM(self.config.instruments + 1, self.config.hidden_size,
                                 batch_first=True, device="meta", dtype=self.fdtype)
            self.__dict__["_native_lstm_view"] = view
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            parameter = getattr(self.recurrence, name)
            if getattr(view, f"{name}_l0") is not parameter:
                setattr(view, f"{name}_l0", parameter)
        # There is no dropout. cuDNN needs backward reserve space only when
        # autograd is enabled, including gradient checks outside train mode.
        view.train(torch.is_grad_enabled())
        return view

    def forward(self, prices: torch.Tensor, *, returns: torch.Tensor | None = None) -> HedgeOutput:
        """Execute the causal recurrence natively and batch both heads over time."""
        cfg = self.config
        if prices.ndim != 3 or prices.shape[1:] != (cfg.horizon_days + 1, cfg.instruments):
            raise ValueError("prices must have shape (paths, horizon_days+1, instruments)")
        if returns is not None and returns.shape != (prices.shape[0], cfg.horizon_days, cfg.instruments):
            raise ValueError("returns must have shape (paths, horizon_days, instruments)")
        if cfg.gain_mode == "expiry_trades" and returns is None:
            raise ValueError("expiry-trade gain requires explicit cohort returns")
        current = (prices[:, :-1] - self.price_mean) / self.price_scale
        time = torch.arange(cfg.horizon_days, dtype=prices.dtype, device=prices.device) / cfg.horizon_days
        time = time[None, :, None].expand(prices.shape[0], -1, -1)
        features = torch.cat((time, current), dim=-1)
        hidden, _ = self._native_lstm()(features)
        action_state = torch.cat((features, hidden), dim=-1)
        trades = ((self.trade_head(action_state) - self.reference_trade_head(action_state))
                  + (self.deep_head(action_state) - self.reference_deep_head(action_state)))
        positions = trades.cumsum(dim=1) if cfg.gain_mode == "daily_positions" else trades
        increments = prices[:, 1:] - prices[:, :-1] if returns is None else returns
        gain = (positions * increments).sum(dim=-1).sum(dim=-1)
        penalty = (cfg.trade_l1 * trades.abs().sum(dim=-1)
                   + cfg.trade_l2 * trades.square().sum(dim=-1)).sum(dim=-1)
        return HedgeOutput(gain=gain, penalty=penalty, trades=trades, positions=positions)

    def forward_eager(self, prices: torch.Tensor, *, returns: torch.Tensor | None = None) -> HedgeOutput:
        """Trade causally using returns for the actual selected contracts.

        In expiry mode each action opens a separate floating-contract quantity;
        the supplied return is realized at expiry or at day-20 liquidation.
        Future realized returns never enter the policy state. The legacy daily
        mode accumulates holdings before multiplying one-day returns. This
        Python-loop reference is retained only for numerical regression checks;
        production ``forward`` uses a native fused sequence recurrence.
        """
        cfg = self.config
        if prices.ndim != 3 or prices.shape[1:] != (cfg.horizon_days + 1, cfg.instruments):
            raise ValueError("prices must have shape (paths, horizon_days+1, instruments)")
        if returns is not None and returns.shape != (prices.shape[0], cfg.horizon_days, cfg.instruments):
            raise ValueError("returns must have shape (paths, horizon_days, instruments)")
        if cfg.gain_mode == "expiry_trades" and returns is None:
            raise ValueError("expiry-trade gain requires explicit cohort returns")
        batch = prices.shape[0]
        hidden = prices.new_zeros((batch, cfg.hidden_size))
        cell = prices.new_zeros((batch, cfg.hidden_size))
        position = prices.new_zeros((batch, cfg.instruments))
        gain = prices.new_zeros(batch)
        penalty = prices.new_zeros(batch)
        trades, positions = [], []
        for day in range(cfg.horizon_days):
            price_state = (prices[:, day] - self.price_mean) / self.price_scale
            time_state = prices.new_full((batch, 1), day / cfg.horizon_days)
            hidden, cell = self.recurrence(torch.cat((time_state, price_state), dim=-1), (hidden, cell))
            action_state = torch.cat((time_state, price_state, hidden), dim=-1)
            trade = ((self.trade_head(action_state) - self.reference_trade_head(action_state))
                     + (self.deep_head(action_state) - self.reference_deep_head(action_state)))
            position = position + trade if cfg.gain_mode == "daily_positions" else trade
            increment = prices[:, day + 1] - prices[:, day] if returns is None else returns[:, day]
            gain = gain + torch.sum(position * increment, dim=-1)
            penalty = penalty + cfg.trade_l1 * trade.abs().sum(dim=-1) + cfg.trade_l2 * trade.square().sum(dim=-1)
            trades.append(trade)
            positions.append(position)
        return HedgeOutput(gain=gain, penalty=penalty,
                           trades=torch.stack(trades, dim=1), positions=torch.stack(positions, dim=1))


def power_exponential_utility(value: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    """Power-exponential utility with exponential gains and quadratic losses."""
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError("risk aversion must be finite and positive")
    utility, _, _ = torch_utility("powexp", risk_aversion)
    return utility(value, torch.zeros_like(value))


def oce_value(pnl: torch.Tensor, translation: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    """OCE objective at a learned cash translation, before trade regularization."""
    return translation + power_exponential_utility(pnl - translation, risk_aversion).mean()


def indifference_quotes(*, empty_oce: float, long_oce: float, short_oce: float) -> tuple[float, float]:
    """Return cash-invariant bid and ask from unpenalized economic OCE values."""
    if not np.isfinite([empty_oce, long_oce, short_oce]).all():
        raise ValueError("OCE values must be finite")
    return long_oce - empty_oce, empty_oce - short_oce
