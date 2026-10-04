"""Entropic OCE root finding when Gaussian risk factors are directly traded."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExponentialOceValues:
    """Objective, density and first-order condition for one action."""

    objective: torch.Tensor
    weights: torch.Tensor
    weighted_return_mean: torch.Tensor
    effective_sample_size: torch.Tensor
    maximum_weight: torch.Tensor


@dataclass(frozen=True)
class ExponentialOceFit:
    """A fitted direct-factor action and its iteration history."""

    action: torch.Tensor
    values: ExponentialOceValues
    history: tuple[dict[str, float | int], ...]
    converged: bool


def exponential_oce_values(
    returns: torch.Tensor,
    action: torch.Tensor,
) -> ExponentialOceValues:
    r"""Evaluate ``log E[exp(-a'R)]`` and its normalized marginal density."""

    if returns.ndim != 2 or len(returns) < 2:
        raise ValueError("returns must have shape (paths, instruments)")
    if action.shape != (returns.shape[1],):
        raise ValueError("action must contain one coefficient per return")
    if not bool(torch.all(torch.isfinite(returns)) and torch.all(torch.isfinite(action))):
        raise ValueError("returns and action must be finite")
    log_marginal = -(returns @ action)
    log_mean_marginal = torch.logsumexp(log_marginal, dim=0) - math.log(len(returns))
    log_weight = log_marginal - torch.logsumexp(log_marginal, dim=0) + math.log(len(returns))
    weights = torch.exp(log_weight)
    weighted_mean = torch.sum(weights[:, None] * returns, dim=0) / len(returns)
    effective_sample_size = len(returns) ** 2 / torch.sum(weights.square())
    return ExponentialOceValues(
        objective=log_mean_marginal,
        weights=weights,
        weighted_return_mean=weighted_mean,
        effective_sample_size=effective_sample_size,
        maximum_weight=torch.max(weights),
    )


def fit_exponential_oce_action(
    returns: torch.Tensor,
    *,
    return_variance: float,
    maximum_iterations: int = 30,
    mean_tolerance: float = 1.0e-10,
    minimum_step: float = 2.0**-20,
) -> ExponentialOceFit:
    r"""Solve the entropic OCE first-order condition on one fixed path set.

    Direct standardized factor returns have population covariance
    ``return_variance * I``.  The corresponding inverse covariance is used as
    a natural-gradient/Newton preconditioner, with objective backtracking for
    finite-sample deviations from the population Hessian.
    """

    if not math.isfinite(return_variance) or return_variance <= 0.0:
        raise ValueError("return_variance must be finite and positive")
    if maximum_iterations <= 0 or mean_tolerance <= 0.0 or minimum_step <= 0.0:
        raise ValueError("solver iterations, tolerance and minimum step must be positive")
    action = torch.zeros(
        returns.shape[1],
        dtype=returns.dtype,
        device=returns.device,
    )
    history: list[dict[str, float | int]] = []
    converged = False
    values = exponential_oce_values(returns, action)
    for iteration in range(maximum_iterations + 1):
        maximum_mean = float(torch.max(torch.abs(values.weighted_return_mean)))
        history.append(
            {
                "iteration": iteration,
                "objective": float(values.objective),
                "weighted_return_mean_l2": float(
                    torch.linalg.vector_norm(values.weighted_return_mean)
                ),
                "maximum_absolute_weighted_return_mean": maximum_mean,
                "effective_sample_size": float(values.effective_sample_size),
                "maximum_weight": float(values.maximum_weight),
                "action_l2": float(torch.linalg.vector_norm(action)),
            }
        )
        if maximum_mean <= mean_tolerance:
            converged = True
            break
        if iteration == maximum_iterations:
            break

        direction = values.weighted_return_mean / return_variance
        step = 1.0
        accepted = False
        while step >= minimum_step:
            proposal = action + step * direction
            proposal_values = exponential_oce_values(returns, proposal)
            if float(proposal_values.objective) <= float(values.objective):
                action = proposal
                values = proposal_values
                accepted = True
                break
            step *= 0.5
        history[-1]["accepted_step"] = step if accepted else 0.0
        if not accepted:
            break
    return ExponentialOceFit(
        action=action,
        values=values,
        history=tuple(history),
        converged=converged,
    )


def antithetic_gaussian_factor_returns(
    *,
    num_paths: int,
    factor_drift: torch.Tensor,
    time_increment_y: float,
    generator: torch.Generator,
) -> torch.Tensor:
    r"""Generate paired returns ``lambda*h + sqrt(h)*epsilon``."""

    if num_paths <= 0 or num_paths % 2:
        raise ValueError("num_paths must be a positive even integer")
    if factor_drift.ndim != 1 or not bool(torch.all(torch.isfinite(factor_drift))):
        raise ValueError("factor_drift must be a finite vector")
    if not math.isfinite(time_increment_y) or time_increment_y <= 0.0:
        raise ValueError("time_increment_y must be finite and positive")
    half = torch.randn(
        (num_paths // 2, len(factor_drift)),
        dtype=factor_drift.dtype,
        device=factor_drift.device,
        generator=generator,
    )
    innovations = torch.stack((half, -half), dim=1).reshape(num_paths, -1)
    return time_increment_y * factor_drift[None] + math.sqrt(time_increment_y) * innovations


__all__ = [
    "ExponentialOceFit",
    "ExponentialOceValues",
    "antithetic_gaussian_factor_returns",
    "exponential_oce_values",
    "fit_exponential_oce_action",
]
