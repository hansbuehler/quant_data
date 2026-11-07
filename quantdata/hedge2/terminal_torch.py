"""Differentiable terminal hedge accounting, checked against the independent NumPy replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from quantdata.hedge2.terminal import TerminalPaths
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class TensorTerminalPaths:
    """Exogenous paths copied once, not a separate fit or a future-information policy input."""

    env: BaseEnvironment
    gains: torch.Tensor
    costs: torch.Tensor
    features: torch.Tensor
    live_contracts: torch.Tensor
    terminate_after_step: torch.Tensor
    claim_gain: torch.Tensor
    weights: torch.Tensor
    initially_active: bool

    @classmethod
    def from_paths(cls, paths: TerminalPaths, *, env: BaseEnvironment) -> TensorTerminalPaths:
        def floating(value: np.ndarray) -> torch.Tensor:
            return torch.tensor(value, dtype=env.fdtype, device=env.device)

        return cls(env=env, gains=floating(paths.gains), costs=floating(paths.costs), features=floating(paths.features),
            live_contracts=torch.tensor(paths.live_contracts, dtype=torch.bool, device=env.device),
            terminate_after_step=torch.tensor(paths.terminate_after_step, dtype=torch.bool, device=env.device),
            claim_gain=floating(paths.claim_gain), weights=floating(paths.weights / paths.weights.sum()),
            initially_active=paths.initially_active)


@dataclass(frozen=True, kw_only=True)
class TensorTerminalEvaluation:
    wealth: torch.Tensor
    hedge_gain: torch.Tensor
    entry_cost: torch.Tensor
    positions: torch.Tensor
    step_gains: torch.Tensor
    step_costs: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class TensorTerminalRisk:
    value: torch.Tensor
    mean_wealth: torch.Tensor
    std_wealth: torch.Tensor
    cash_shift: torch.Tensor | None = None
    cash_first_order_residual: torch.Tensor | None = None


def evaluate_tensor_policy(*, paths: TensorTerminalPaths, initial_inventory: torch.Tensor,
                           policy: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
                           target_side: int, position_cap: float | None = None) -> TensorTerminalEvaluation:
    """Holdings are fixed-contract coefficients in the episode's original S0 units.

    Only current features, current inventory and decision time reach the
    policy. Future fees stay inside terminal wealth. Settlement/KO clears
    holdings after that period's gain without another turnover charge.
    This function does not turn an unrestricted path table into a causal
    policy: use a pointwise deterministic function with fixed normalization,
    never cross-path batch statistics, and verify
    its saved holdings with the NumPy history-aware replay.
    """
    count, steps, instruments = paths.gains.shape
    if (initial_inventory.shape != (instruments,) or target_side not in (-1, 1)
            or position_cap is not None and (not np.isfinite(position_cap) or position_cap <= 0)):
        raise ValueError("Tensor terminal inventory, side and optional position limit must align")
    initial = initial_inventory.to(dtype=paths.env.fdtype, device=paths.env.device)
    if not torch.isfinite(initial).all():
        raise ValueError("The current hedge inventory must be finite")
    if position_cap is not None and torch.any(torch.abs(initial) > position_cap):
        raise ValueError("The current hedge inventory exceeds the declared position limit")
    if torch.any(initial[~paths.live_contracts[0]] != 0.):
        raise ValueError("Expired assets cannot enter the current hedge inventory")
    inventory = initial.expand(count, instruments)
    active = torch.full((count,), paths.initially_active, dtype=torch.bool, device=paths.env.device)
    inventory = torch.where(active[:, None], inventory, 0.)
    positions, gains, costs = [], [], []
    for step in range(steps):
        proposed = policy(paths.features[:, step], inventory, step)
        if proposed.shape != (count, instruments) or proposed.dtype != paths.gains.dtype or proposed.device != paths.gains.device:
            raise ValueError(f"The tensor policy must preserve input precision, device and asset geometry: "
                f"expected={(count, instruments), paths.gains.dtype, paths.gains.device}, "
                f"actual={tuple(proposed.shape), proposed.dtype, proposed.device}")
        if paths.env.debug and not torch.isfinite(proposed).all():
            raise ValueError("The tensor terminal policy returned a non-finite position")
        live = active[:, None] & paths.live_contracts[step]
        proposed = torch.where(live, proposed, 0.)
        if position_cap is not None and torch.any(torch.abs(proposed) > position_cap * (1. + 1e-12)):
            raise ValueError("The tensor terminal policy exceeded the declared position limit")
        positions.append(proposed)
        costs.append(torch.sum(paths.costs[:, step] * torch.abs(proposed - inventory), dim=1))
        gains.append(torch.sum(proposed * paths.gains[:, step], dim=1))
        active = active & ~paths.terminate_after_step[:, step]
        inventory = torch.where(active[:, None] & paths.live_contracts[step + 1], proposed, 0.)
    step_gains, step_costs = torch.stack(gains, dim=1), torch.stack(costs, dim=1)
    total_gains, total_costs = step_gains.sum(dim=1), step_costs.sum(dim=1)
    wealth = target_side * paths.claim_gain + total_gains - total_costs
    return TensorTerminalEvaluation(wealth=wealth, hedge_gain=total_gains, entry_cost=total_costs,
        positions=torch.stack(positions, dim=1), step_gains=step_gains, step_costs=step_costs)


def tensor_terminal_risk(wealth: torch.Tensor, weights: torch.Tensor, *, objective: str,
                         risk_aversion: float) -> TensorTerminalRisk:
    """Population MV or exact cash-optimized power OCE of NET terminal wealth.

    Power cash is found by deterministic bisection without gradients.
    The envelope theorem then gives the exact FIRST derivative with
    respect to wealth. This is not a second-derivative implementation.
    """
    if (wealth.ndim != 1 or weights.shape != wealth.shape or len(wealth) == 0
            or weights.dtype != wealth.dtype or weights.device != wealth.device
            or objective not in ("mean_variance", "power_exponential")
            or risk_aversion <= 0 or np.isnan(risk_aversion)
            or objective == "power_exponential" and not np.isfinite(risk_aversion)):
        raise ValueError("Tensor risk needs aligned wealth/weights and a supported positive risk aversion")
    if (not torch.isfinite(wealth).all() or not torch.isfinite(weights).all()
            or torch.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Tensor wealth and weights must be finite with positive nonnegative weight mass")
    probabilities = weights / weights.sum()
    mean = torch.sum(probabilities * wealth)
    centered = wealth - mean
    variance = torch.sum(probabilities * centered.square())
    std = torch.sqrt(torch.clamp_min(variance, 0.))
    if objective == "mean_variance":
        finite = np.isfinite(risk_aversion)
        value = .5 * (risk_aversion if finite else 1.) * variance - (mean if finite else 0.)
        return TensorTerminalRisk(value=value, mean_wealth=mean, std_wealth=std)
    with torch.no_grad():
        values = centered.detach()
        lower, upper = values.min(), values.max()
        for _ in range(64):
            middle = .5 * (lower + upper)
            scaled = risk_aversion * (values - middle)
            marginal = torch.exp(-torch.clamp_min(scaled, 0.)) - torch.clamp_max(scaled, 0.)
            too_high = torch.sum(probabilities * marginal) > 1.
            lower = torch.where(too_high, lower, middle)
            upper = torch.where(too_high, middle, upper)
        shift = mean.detach() + .5 * (lower + upper)
    translated = wealth - shift
    positive, negative = torch.clamp_min(translated, 0.), torch.clamp_max(translated, 0.)
    utility = translated - torch.expm1(-risk_aversion * positive) / risk_aversion - positive - .5 * risk_aversion * negative.square()
    marginal = torch.exp(-risk_aversion * positive) - risk_aversion * negative
    value = -shift - torch.sum(probabilities * utility)
    residual = torch.sum(probabilities * marginal) - 1.
    return TensorTerminalRisk(value=value, mean_wealth=mean, std_wealth=std,
        cash_shift=shift, cash_first_order_residual=residual)
