"""Hard-cutoff power-exponential OCE densities on empirical path sets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quantdata.dysanos.flex.direct_utility_frontier import (
    log_powerexp_marginal,
    powerexp_utility,
)


def _validate_geometry(value: torch.Tensor, cutoffs: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] < 2:
        raise ValueError("cutoff OCE values must have shape (heads,paths)")
    if cutoffs.shape != (value.shape[0],) or not bool(
        torch.all(torch.isfinite(cutoffs) & (cutoffs > 0.0))
    ):
        raise ValueError("one finite-positive cutoff is required per OCE head")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("cutoff OCE values must be finite")


def cutoff_powerexp_utility(value: torch.Tensor, cutoffs: torch.Tensor) -> torch.Tensor:
    r"""Evaluate :math:`\widetilde u(x)=u(\min(x,x_0))` per head."""

    _validate_geometry(value, cutoffs)
    return powerexp_utility(torch.minimum(value, cutoffs[:, None]))


def cutoff_powerexp_marginal(value: torch.Tensor, cutoffs: torch.Tensor) -> torch.Tensor:
    r"""Return the right derivative of the hard-cutoff utility.

    The derivative is the ordinary power-exponential marginal below the
    cutoff and zero above it. At the cutoff the concave subgradient is the
    interval from zero to the left derivative; the translation solver chooses
    the required boundary value explicitly.
    """

    _validate_geometry(value, cutoffs)
    log_marginal = log_powerexp_marginal(value)
    return torch.where(value < cutoffs[:, None], torch.exp(log_marginal), 0.0)


@dataclass(frozen=True)
class CutoffPowerexpDensity:
    """Cash-OCE translation and one normalized empirical density per head."""

    translation: torch.Tensor
    shifted_pnl: torch.Tensor
    weights: torch.Tensor
    mass_before_boundary_subgradient: torch.Tensor
    boundary_subgradient_weight: torch.Tensor


def solve_cutoff_powerexp_density(
    pnl: torch.Tensor,
    cutoffs: torch.Tensor,
    *,
    num_iterations: int = 80,
) -> CutoffPowerexpDensity:
    r"""Solve the empirical cash-OCE normalization for a hard cutoff.

    For a finite path set the derivative of ``u(min(x, cutoff))`` jumps when a
    path lands at the cutoff. Bisection approaches the optimizer from the
    right, after which the missing density mass is assigned to the closest
    boundary path. This selects a valid element of the cutoff utility's
    subgradient and makes sample-mean density normalization exact.
    """

    _validate_geometry(pnl, cutoffs)
    if num_iterations < 1:
        raise ValueError("cutoff OCE translation iterations must be positive")
    original_dtype = pnl.dtype
    values = pnl.detach().double()
    limits = cutoffs.detach().double()
    lower = -torch.amax(values, dim=1) - 1.0
    upper = limits - torch.amin(values, dim=1) + 1.0
    for _ in range(num_iterations):
        middle = 0.5 * (lower + upper)
        marginal = cutoff_powerexp_marginal(values + middle[:, None], limits)
        residual = torch.mean(marginal, dim=1) - 1.0
        lower = torch.where(residual > 0.0, middle, lower)
        upper = torch.where(residual > 0.0, upper, middle)

    translation = upper
    shifted = values + translation[:, None]
    weights = cutoff_powerexp_marginal(shifted, limits)
    mass_before = torch.mean(weights, dim=1)
    missing_total_weight = values.shape[1] * (1.0 - mass_before)
    if bool(torch.any(missing_total_weight < -1.0e-8)):
        raise FloatingPointError("cutoff OCE bisection ended above unit density mass")
    missing_total_weight = torch.clamp_min(missing_total_weight, 0.0)

    # Float32 path P&Ls can contain exact ties even when the underlying model
    # is continuous. Every tied boundary path has the same valid subgradient
    # interval, so distribute the missing mass over the whole tied atom.
    boundary_index = torch.argmin(torch.abs(shifted - limits[:, None]), dim=1)
    boundary_capacity = torch.exp(-limits)
    assigned_boundary_weight = torch.zeros_like(missing_total_weight)
    for head, index in enumerate(boundary_index.tolist()):
        tied = values[head] == values[head, index]
        available_per_path = torch.clamp_min(
            boundary_capacity[head] - weights[head, tied],
            0.0,
        )
        available = torch.sum(available_per_path)
        tolerance = 1.0e-8 * max(1.0, float(available))
        if float(missing_total_weight[head]) > float(available) + tolerance:
            raise FloatingPointError(
                "cutoff OCE boundary atom does not have enough subgradient capacity"
            )
        if float(missing_total_weight[head]) > 0.0:
            allocation = (
                missing_total_weight[head]
                * available_per_path
                / torch.clamp_min(available, torch.finfo(available.dtype).tiny)
            )
            weights[head, tied] += allocation
            assigned_boundary_weight[head] = torch.sum(allocation)
    mass_residual = torch.abs(torch.mean(weights, dim=1) - 1.0)
    if bool(torch.any(mass_residual > 1.0e-12)):
        raise FloatingPointError("cutoff OCE density did not normalize")
    if not bool(torch.all(torch.isfinite(weights) & (weights >= 0.0))):
        raise FloatingPointError("cutoff OCE density is not finite-nonnegative")
    return CutoffPowerexpDensity(
        translation=translation.to(dtype=original_dtype),
        shifted_pnl=shifted.to(dtype=original_dtype),
        weights=weights.to(dtype=original_dtype),
        mass_before_boundary_subgradient=mass_before.to(dtype=original_dtype),
        boundary_subgradient_weight=assigned_boundary_weight.to(dtype=original_dtype),
    )


__all__ = [
    "CutoffPowerexpDensity",
    "cutoff_powerexp_marginal",
    "cutoff_powerexp_utility",
    "solve_cutoff_powerexp_density",
]
