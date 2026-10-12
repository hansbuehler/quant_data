"""Differentiable GROMM hedging objectives in common P&L units."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommObjective,
)
from quantdata.torch import torch


@dataclass(frozen=True, kw_only=True)
class GrommObjectiveValue:
    """Scalar loss and pathwise wealth diagnostics."""

    loss: torch.Tensor
    wealth: torch.Tensor
    transaction_cost: torch.Tensor


def power_exponential_utility(
    wealth: torch.Tensor,
    *,
    risk_parameter: float,
) -> torch.Tensor:
    """Return the smooth power-exponential utility used by the benchmark."""
    value = float(risk_parameter)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("risk_parameter must be finite-positive")
    positive = torch.clamp_min(wealth, 0.0)
    return torch.where(
        wealth >= 0.0,
        -torch.expm1(-value * positive) / value,
        wealth - 0.5 * value * wealth.square(),
    )


def gromm_hedging_objective(
    *,
    actions: torch.Tensor,
    instrument_returns: torch.Tensor,
    claim_returns: torch.Tensor,
    linear_transaction_costs: torch.Tensor,
    objective: GrommObjective,
    risk_parameter: float,
    cash_translation: torch.Tensor | None = None,
    initial_positions: torch.Tensor | None = None,
    settles_next: torch.Tensor | None = None,
    terminal_transaction_costs: torch.Tensor | None = None,
) -> GrommObjectiveValue:
    """Evaluate one objective with predictable turnover and liquidation costs."""
    if actions.shape != instrument_returns.shape or actions.ndim != 3:
        raise ValueError("actions and returns must be matching path-step tensors")
    if claim_returns.shape != actions.shape[:2]:
        raise ValueError("claim returns must align with path and step axes")
    if linear_transaction_costs.shape not in {
        actions.shape[1:],
        actions.shape,
    }:
        raise ValueError("transaction costs do not align with actions")
    if not math.isfinite(risk_parameter) or risk_parameter <= 0.0:
        raise ValueError("risk_parameter must be finite-positive")
    costs = linear_transaction_costs
    if costs.ndim == 2:
        costs = costs[None]
    if initial_positions is None:
        initial_positions = torch.zeros(
            actions.shape[-1], dtype=actions.dtype, device=actions.device
        )
    if initial_positions.shape not in {
        (actions.shape[-1],),
        (actions.shape[0], actions.shape[-1]),
    }:
        raise ValueError("initial positions do not align with actions")
    if settles_next is None:
        settles_next = torch.zeros(
            actions.shape[1:], dtype=torch.bool, device=actions.device
        )
    if settles_next.shape != actions.shape[1:]:
        raise ValueError("settlement mask does not align with actions")
    previous = (
        initial_positions[None]
        if initial_positions.ndim == 1
        else initial_positions
    )
    step_costs: list[torch.Tensor] = []
    for step in range(actions.shape[1]):
        current = actions[:, step]
        step_costs.append(
            torch.sum(costs[:, step] * torch.abs(current - previous), dim=-1)
        )
        previous = torch.where(
            settles_next[step][None],
            torch.zeros_like(current),
            current,
        )
    transaction_cost = torch.stack(step_costs, dim=1).sum(dim=1)
    if terminal_transaction_costs is not None:
        terminal = terminal_transaction_costs
        if terminal.ndim == 1:
            terminal = terminal[None]
        if terminal.shape not in {
            (1, actions.shape[-1]),
            (actions.shape[0], actions.shape[-1]),
        }:
            raise ValueError("terminal costs do not align with actions")
        transaction_cost = transaction_cost + torch.sum(
            terminal * torch.abs(previous), dim=-1
        )
    gain = torch.sum(actions * instrument_returns, dim=(-2, -1))
    claim = torch.sum(claim_returns, dim=-1)
    wealth = claim + gain - transaction_cost

    if objective is GrommObjective.MEAN_VARIANCE:
        centered = wealth - torch.mean(wealth)
        loss = -torch.mean(wealth) + 0.5 * risk_parameter * torch.mean(centered.square())
    elif objective is GrommObjective.MEAN_VOLATILITY:
        standard_deviation = torch.sqrt(
            torch.mean((wealth - torch.mean(wealth)).square())
            + torch.finfo(wealth.dtype).eps
        )
        loss = -torch.mean(wealth) + risk_parameter * standard_deviation
    elif objective is GrommObjective.ENTROPY:
        exponent = -risk_parameter * wealth
        loss = (
            torch.logsumexp(exponent, dim=0) - math.log(len(exponent))
        ) / risk_parameter
    elif objective is GrommObjective.POWER_EXPONENTIAL:
        if cash_translation is None or cash_translation.numel() != 1:
            raise ValueError("power-exponential OCE requires one cash translation")
        residual = wealth - cash_translation
        oce = cash_translation + torch.mean(
            power_exponential_utility(
                residual,
                risk_parameter=risk_parameter,
            )
        )
        loss = -oce
    else:
        raise ValueError(f"unsupported GROMM objective {objective!r}")
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("GROMM objective is non-finite")
    return GrommObjectiveValue(
        loss=loss,
        wealth=wealth,
        transaction_cost=transaction_cost,
    )
