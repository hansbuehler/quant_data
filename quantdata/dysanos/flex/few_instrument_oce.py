"""Entropic OCE root solves for small conditional trading books."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EntropicOceBatchFit:
    """One empirical entropic OCE solution per conditional origin."""

    action: torch.Tensor
    objective: torch.Tensor
    weights: torch.Tensor
    weighted_mean: torch.Tensor
    effective_sample_size: torch.Tensor
    maximum_weight: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor


def entropic_oce_values(
    returns: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Evaluate ``log E[exp(-a'R)]`` and its normalized density.

    ``returns`` has shape ``(origins, paths, instruments)`` and ``action`` has
    shape ``(origins, instruments)``.  The returned weights have sample mean
    one within each origin, so the weighted mean is the empirical OCE
    first-order condition without an additional denominator.
    """

    if returns.ndim != 3 or returns.shape[1] < 2:
        raise ValueError("returns must have shape (origins,paths,instruments)")
    if action.shape != (returns.shape[0], returns.shape[2]):
        raise ValueError("action must have shape (origins,instruments)")
    if not bool(torch.all(torch.isfinite(returns)) and torch.all(torch.isfinite(action))):
        raise ValueError("returns and action must be finite")
    log_marginal = -torch.einsum("opi,oi->op", returns, action)
    log_normalizer = torch.logsumexp(log_marginal, dim=1) - math.log(returns.shape[1])
    weights = torch.exp(log_marginal - log_normalizer[:, None])
    weighted_mean = torch.mean(weights[..., None] * returns, dim=1)
    return log_normalizer, weights, weighted_mean


def fit_batched_entropic_oce(
    returns: torch.Tensor,
    *,
    maximum_iterations: int = 50,
    mean_tolerance: float = 1.0e-9,
    covariance_eigenvalue_floor: float = 1.0e-10,
    minimum_step: float = 2.0**-20,
) -> EntropicOceBatchFit:
    r"""Solve ``E_Q[R]=0`` for several empirical conditional distributions.

    A covariance-Newton update is applied in parallel across origins.  Only a
    relative numerical eigenvalue floor is used when solving the Newton
    system; it is not an action penalty and does not alter the reported OCE
    density or first-order condition.
    """

    if returns.ndim != 3 or returns.shape[1] < 2:
        raise ValueError("returns must have shape (origins,paths,instruments)")
    if maximum_iterations <= 0 or mean_tolerance <= 0.0:
        raise ValueError("iterations and mean tolerance must be positive")
    if not 0.0 < covariance_eigenvalue_floor < 1.0 or minimum_step <= 0.0:
        raise ValueError("invalid Newton safeguards")
    if not bool(torch.all(torch.isfinite(returns))):
        raise ValueError("returns must be finite")

    origins, _, instruments = returns.shape
    action = torch.zeros(
        (origins, instruments), dtype=returns.dtype, device=returns.device
    )
    iterations = torch.zeros(origins, dtype=torch.int64, device=returns.device)
    converged = torch.zeros(origins, dtype=torch.bool, device=returns.device)
    objective, weights, weighted_mean = entropic_oce_values(returns, action)

    identity = torch.eye(instruments, dtype=returns.dtype, device=returns.device)
    for iteration in range(1, maximum_iterations + 1):
        maximum_mean = torch.amax(torch.abs(weighted_mean), dim=1)
        newly_converged = maximum_mean <= mean_tolerance
        converged |= newly_converged
        active = ~converged
        if not bool(torch.any(active)):
            break

        centered = returns - weighted_mean[:, None, :]
        covariance = torch.einsum(
            "op,opi,opj->oij", weights, centered, centered
        ) / returns.shape[1]
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))
        eigenvalues = torch.linalg.eigvalsh(covariance)
        floor = (
            torch.amax(eigenvalues, dim=1).clamp_min(torch.finfo(returns.dtype).tiny)
            * covariance_eigenvalue_floor
        )
        regularization = torch.clamp_min(floor - torch.amin(eigenvalues, dim=1), 0.0)
        direction = torch.linalg.solve(
            covariance + regularization[:, None, None] * identity,
            weighted_mean.unsqueeze(-1),
        ).squeeze(-1)
        direction = torch.where(active[:, None], direction, torch.zeros_like(direction))

        step = torch.ones(origins, dtype=returns.dtype, device=returns.device)
        accepted = ~active
        proposal_action = action
        proposal_values = (objective, weights, weighted_mean)
        while bool(torch.any(~accepted)) and float(torch.max(step[~accepted])) >= minimum_step:
            candidate_action = action + step[:, None] * direction
            candidate = entropic_oce_values(returns, candidate_action)
            improved = candidate[0] <= objective
            take = (~accepted) & improved
            proposal_action = torch.where(take[:, None], candidate_action, proposal_action)
            proposal_values = tuple(
                torch.where(
                    take.reshape((origins,) + (1,) * (value.ndim - 1)),
                    value,
                    previous,
                )
                for value, previous in zip(candidate, proposal_values, strict=True)
            )
            accepted |= take
            step = torch.where(accepted, step, 0.5 * step)
        if bool(torch.any(active & ~accepted)):
            break
        action = proposal_action
        objective, weights, weighted_mean = proposal_values
        iterations[active] = iteration

    converged |= torch.amax(torch.abs(weighted_mean), dim=1) <= mean_tolerance
    effective_sample_size = returns.shape[1] ** 2 / torch.sum(weights.square(), dim=1)
    return EntropicOceBatchFit(
        action=action,
        objective=objective,
        weights=weights,
        weighted_mean=weighted_mean,
        effective_sample_size=effective_sample_size,
        maximum_weight=torch.amax(weights, dim=1),
        iterations=iterations,
        converged=converged,
    )


__all__ = [
    "EntropicOceBatchFit",
    "entropic_oce_values",
    "fit_batched_entropic_oce",
]
