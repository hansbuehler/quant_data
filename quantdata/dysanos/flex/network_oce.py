"""State-dependent entropic OCE objectives with bounded trading actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConditionalEntropicOceValues:
    """Conditional OCE objective and implied empirical density."""

    loss: torch.Tensor
    objective_by_origin: torch.Tensor
    weights: torch.Tensor
    weighted_return_mean: torch.Tensor
    effective_sample_size: torch.Tensor


@dataclass(frozen=True)
class EntropicDensityValues:
    """Empirical entropic OCE normalizer and mean-one density weights."""

    log_oce: torch.Tensor
    weights: torch.Tensor
    effective_sample_size: torch.Tensor


def l1_capped_action(raw_action: torch.Tensor, maximum_l1: float) -> torch.Tensor:
    """Apply the exact radial ``L1`` cap used by the utility experiments."""

    if raw_action.ndim != 2 or not math.isfinite(maximum_l1) or maximum_l1 <= 0.0:
        raise ValueError("a positive L1 cap and a two-dimensional action are required")
    norm = torch.sum(torch.abs(raw_action), dim=1, keepdim=True)
    return raw_action * maximum_l1 / torch.maximum(
        norm,
        torch.as_tensor(maximum_l1, dtype=raw_action.dtype, device=raw_action.device),
    )


def entropic_density_from_pnl(pnl: torch.Tensor) -> EntropicDensityValues:
    """Return the empirical entropic density induced by pathwise net PnL.

    The weights are ``dQ/dP`` values normalized to have sample mean one.  If
    the PnL is replayed with gradients enabled, the exact derivative of the
    empirical log-OCE is therefore obtained from ``-mean(weights * pnl)``
    while treating ``weights`` as detached.  This permits one optimizer batch
    to span several sequential simulation microbatches without retaining all
    their autograd graphs simultaneously.
    """

    if pnl.ndim != 1 or len(pnl) == 0 or not bool(torch.all(torch.isfinite(pnl))):
        raise ValueError("pathwise PnL must be a finite nonempty vector")
    log_oce = torch.logsumexp(-pnl, dim=0) - math.log(len(pnl))
    weights = torch.exp(-pnl - log_oce)
    effective_sample_size = len(pnl) ** 2 / torch.sum(weights.square())
    return EntropicDensityValues(
        log_oce=log_oce,
        weights=weights,
        effective_sample_size=effective_sample_size,
    )


def conditional_entropic_oce(
    returns: torch.Tensor,
    action: torch.Tensor,
    transaction_cost: torch.Tensor,
    *,
    action_ridge: float = 0.0,
) -> ConditionalEntropicOceValues:
    r"""Evaluate a conditional entropic OCE with proportional trading costs.

    The minimized criterion per origin is

    .. math::

       \log E[\exp(-a(h)'R)\mid h] + c'|a(h)|
       + \tfrac12\rho\|a(h)\|_2^2.

    Transaction costs are deterministic at the trading state and therefore
    do not enter the normalized marginal-utility density.  They change its
    first-order target from zero drift to the usual no-trade band.
    """

    if returns.ndim != 3 or returns.shape[1] < 2:
        raise ValueError("returns must have shape (origins,paths,instruments)")
    if action.shape != (returns.shape[0], returns.shape[2]):
        raise ValueError("action must have shape (origins,instruments)")
    if transaction_cost.shape != (returns.shape[2],):
        raise ValueError("transaction cost must contain one value per instrument")
    if (
        not math.isfinite(action_ridge)
        or action_ridge < 0.0
        or not bool(
            torch.all(torch.isfinite(returns))
            and torch.all(torch.isfinite(action))
            and torch.all(torch.isfinite(transaction_cost) & (transaction_cost >= 0.0))
        )
    ):
        raise ValueError("OCE inputs and penalties must be finite and valid")
    log_marginal = -torch.einsum("opi,oi->op", returns, action)
    log_normalizer = torch.logsumexp(log_marginal, dim=1) - math.log(returns.shape[1])
    cost = torch.sum(transaction_cost[None] * torch.abs(action), dim=1)
    ridge = 0.5 * action_ridge * torch.sum(action.square(), dim=1)
    objective = log_normalizer + cost + ridge
    weights = torch.exp(log_marginal - log_normalizer[:, None])
    weighted_mean = torch.mean(weights[..., None] * returns, dim=1)
    effective_sample_size = returns.shape[1] ** 2 / torch.sum(weights.square(), dim=1)
    return ConditionalEntropicOceValues(
        loss=torch.mean(objective),
        objective_by_origin=objective,
        weights=weights,
        weighted_return_mean=weighted_mean,
        effective_sample_size=effective_sample_size,
    )


__all__ = [
    "ConditionalEntropicOceValues",
    "EntropicDensityValues",
    "conditional_entropic_oce",
    "entropic_density_from_pnl",
    "l1_capped_action",
]
