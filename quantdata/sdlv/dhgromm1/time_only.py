"""Deterministic affine-time trading, with transaction penalties INSIDE utility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.wide_audit import sha256
from quantdata.sdlv.dhgromm1.contracts import RETURN_CONVENTION, canonical_gains
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility

LABELS = ("empty", "long", "short")
SIGNS = (0, 1, -1)


@version("0.2", dependencies=[torch_utility])
@dataclass(frozen=True, kw_only=True)
class TimeOnlyConfig:
    label: str
    group: str = "sdlv/dhgromm1/time_only"
    user_version: str = "0.1"
    instruments: int = 21
    return_convention: str = RETURN_CONVENTION
    horizon_days: int = 20
    event_days: tuple[int, int] = (0, 10)
    time_scale: float = 20.
    affine: FeedForwardConfig = FeedForwardConfig(activation="Identity", num_depth=0, num_width=21)
    risk_aversion: float = 100.
    trade_l1: float = 1e-4
    trade_l2: float = 1e-8

    def __post_init__(self) -> None:
        verify_inp(bool(self.label) and self.instruments == 21 and self.horizon_days == 20,
                   lambda: "time-only experiment requires a label, 21 instruments and 20 trading days")
        verify_inp(self.return_convention == RETURN_CONVENTION, lambda: "time-only hedge requires floating-contract gains")
        verify_inp(self.event_days == (0, 10) and np.isfinite(self.time_scale) and self.time_scale > 0,
                   lambda: "invalid exact-date heads or time scale")
        verify_inp(self.affine.num_depth == 0 and self.affine.activation == "Identity" and self.affine.use_bias,
                   lambda: "background action must be affine with no nonlinear activation")
        verify_inp(np.isfinite(self.risk_aversion) and self.risk_aversion > 0
                   and np.isfinite(self.trade_l1) and self.trade_l1 >= 0
                   and np.isfinite(self.trade_l2) and self.trade_l2 >= 0,
                   lambda: "invalid utility or cost parameters")


def time_design(config: TimeOnlyConfig) -> np.ndarray:
    """Columns: intercept, normalized clock, day-0 and day-10 indicators."""
    days = np.arange(config.horizon_days)
    return np.column_stack((np.ones_like(days), days / config.time_scale,
                            days == config.event_days[0], days == config.event_days[1])).astype(np.float64)


def return_exposures(returns: np.ndarray, config: TimeOnlyConfig) -> np.ndarray:
    """Exactly compress sum(a(t)*return(t)) into four exposures per instrument."""
    if returns.ndim != 3 or returns.shape[1:] != (config.horizon_days, config.instruments):
        raise ValueError("invalid expiry-return geometry")
    if not np.isfinite(returns).all():
        raise ValueError("nonfinite expiry returns")
    return np.einsum("nti,tk->nki", returns.astype(np.float64), time_design(config)).reshape(len(returns), -1)


def load_training_exposures(scenarios: Path, *, config: TimeOnlyConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read existing paths and convert legacy returns; no states or replay."""
    import json

    identity = json.loads((scenarios / "identity.json").read_text())
    if identity["substeps"] != 1 or identity["instruments"] != config.instruments:
        raise ValueError("the accepted one-step/day, 21-instrument dataset is required")
    exposures, payoffs, records = [], [], []
    completed = 0
    for path in sorted((scenarios / "train").glob("[0-9]*.npz")):
        if int(path.stem) != completed:
            raise ValueError("training chunks are not contiguous in original antithetic order")
        with np.load(path, allow_pickle=False) as saved:
            ret, payoff = canonical_gains(saved["prices"], saved["returns"], identity=identity), saved["payoff"]
        if len(ret) % 2 or payoff.shape != (len(ret),) or not np.isfinite(payoff).all() or np.any(payoff < 0):
            raise ValueError("invalid training payoff or antithetic population")
        exposures.append(return_exposures(ret, config))
        payoffs.append(payoff.astype(np.float64))
        records.append(dict(file=str(path), sha256=sha256(path), paths=len(ret)))
        completed += len(ret)
    if completed != identity["paths_per_sample"]:
        raise ValueError("incomplete original training sample")
    return np.concatenate(exposures), np.concatenate(payoffs), dict(
        paths=completed, scenario_identity_sha256=sha256(scenarios / "identity.json"), chunks=records,
        data="existing training floating gains and payoffs; explicit legacy conversion if needed", no_replay=True, validation_used=False)


class TimeOnlyHedge(BaseModel):
    """A(t)=b+w*(day/20)+v0*1(day=0)+v10*1(day=10); all trades deterministic."""

    def __init__(self, env: BaseEnvironment, *, config: TimeOnlyConfig) -> None:
        super().__init__(env)
        self.config = config
        self.affine = FeedForward(env, num_input=1, num_output=config.instruments, config=config.affine)
        self.day0 = torch.nn.Parameter(torch.zeros(config.instruments, device=env.device, dtype=env.fdtype))
        self.day10 = torch.nn.Parameter(torch.zeros(config.instruments, device=env.device, dtype=env.fdtype))
        self.translation = torch.nn.Parameter(torch.zeros((), device=env.device, dtype=env.fdtype))
        self.register_buffer("design", torch.as_tensor(time_design(config), device=env.device, dtype=env.fdtype))
        # Affine maps have nonzero parameter gradients even at zero weights;
        # there are no deep hidden layers requiring a frozen random reference.
        with torch.no_grad():
            self.affine.I0.weight.zero_()
            self.affine.I0.bias.zero_()

    def coefficients(self) -> torch.Tensor:
        return torch.stack((self.affine.I0.bias, self.affine.I0.weight[:, 0], self.day0, self.day10))

    def actions(self) -> torch.Tensor:
        return self.design @ self.coefficients()

    def cost(self) -> torch.Tensor:
        action = self.actions()
        return (self.config.trade_l1 * action.abs() + self.config.trade_l2 * action.square()).sum()

    def forward(self, exposures: torch.Tensor) -> torch.Tensor:
        """Pathwise terminal trading gain, not an action conditioned on a path."""
        return exposures @ self.coefficients().flatten()


class TimeOnlyTriple(BaseModel):
    """Three independent affine policies; cash costs deducted BEFORE utility."""

    def __init__(self, env: BaseEnvironment, *, config: TimeOnlyConfig) -> None:
        super().__init__(env)
        self.config = config
        for label in LABELS:
            setattr(self, label, TimeOnlyHedge(env, config=config))
        self._utility, _, _ = torch_utility("powexp", config.risk_aversion)
        self.register_buffer("signs", torch.tensor(SIGNS, device=env.device, dtype=env.fdtype))

    def gains(self, exposures: torch.Tensor) -> torch.Tensor:
        coefficients = torch.stack([getattr(self, k).coefficients().flatten() for k in LABELS], -1)
        return exposures @ coefficients

    def costs(self) -> torch.Tensor:
        return torch.stack([getattr(self, k).cost() for k in LABELS])

    def cash(self) -> torch.Tensor:
        return torch.stack([getattr(self, k).translation for k in LABELS])

    def forward(self, exposures: torch.Tensor, payoff: torch.Tensor) -> torch.Tensor:
        net_terminal = self.gains(exposures) + payoff[:, None] * self.signs - self.costs()
        cash = self.cash()
        # This is literal net-P&L-inside-utility accounting. Do NOT add an
        # expected cost to the loss outside u, especially for stochastic policies.
        return -(cash + self._utility(net_terminal - cash, torch.zeros_like(net_terminal)).mean(0))
