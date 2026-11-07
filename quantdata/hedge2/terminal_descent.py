"""Training-only finite descent probes for accepted full-terminal iterates."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, kw_only=True)
class TerminalDescentProbe:
    baseline_loss: float
    best_loss: float
    tolerance: float
    evaluations: int
    rounded_away: int
    direction_count: int

    @property
    def maximum_improvement(self) -> float:
        return self.baseline_loss - self.best_loss

    @property
    def no_detected_descent(self) -> bool:
        return self.maximum_improvement <= self.tolerance


def parameter_vector(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Copy parameters without rebinding their storage or modifying gradients."""
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def gradient_vector(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Return true gradients, retaining zeros for unused model parameters."""
    gradients = [torch.zeros_like(parameter).reshape(-1) if parameter.grad is None
                 else parameter.grad.detach().reshape(-1) for parameter in parameters]
    result = torch.cat(gradients)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Non-finite terminal-policy gradient")
    return result


@torch.no_grad()
def probe_terminal_descent(*, parameters: Iterable[torch.nn.Parameter],
                           evaluate_loss: Callable[[], torch.Tensor],
                           directions: Iterable[torch.Tensor],
                           relative_distances: tuple[float, ...],
                           tolerance: float,
                           to_probe_coordinates: Callable[[torch.Tensor], torch.Tensor] | None = None,
                           from_probe_coordinates: Callable[[torch.Tensor], torch.Tensor] | None = None) -> TerminalDescentProbe:
    """Probe both signs of fixed directions on the unchanged training sample.

    Distances are Euclidean parameter distances divided by max(1, ||theta||).
    Optional mutually inverse maps keep distances and supplied directions
    in the original physical coordinates of a reparameterized optimizer.
    Every evaluation retains the actual fees, caps and objective. Parameters
    are restored bit for bit even when evaluation raises. A passing result
    is only local numerical evidence, not a stationarity/global certificate.
    """
    parameters = tuple(parameters)
    if (not parameters or not relative_distances
            or any(not np.isfinite(distance) or distance <= 0 for distance in relative_distances)
            or len(set(relative_distances)) != len(relative_distances)
            or not np.isfinite(tolerance) or tolerance < 0
            or (to_probe_coordinates is None) != (from_probe_coordinates is None)):
        raise ValueError("Descent probes need parameters, distinct positive distances and a finite tolerance")
    original = parameter_vector(parameters)
    if not bool(torch.isfinite(original).all()):
        raise ValueError("Descent probes require finite model parameters")
    base = original if to_probe_coordinates is None else to_probe_coordinates(original)
    if (base.shape != original.shape or base.dtype != original.dtype or base.device != original.device
            or not bool(torch.isfinite(base).all())):
        raise ValueError("Descent-probe coordinate map must preserve finite parameter geometry")
    normalized = []
    for direction in directions:
        if direction.shape != original.shape or direction.device != original.device:
            raise ValueError("Descent directions must match the parameter vector and device")
        vector = direction.detach().to(torch.float64)
        norm = torch.linalg.vector_norm(vector)
        if not bool(torch.isfinite(norm)):
            raise ValueError("Descent directions must be finite")
        if float(norm) > 0:
            normalized.append(vector / norm)
    if not normalized:
        raise ValueError("Descent probes need at least one nonzero direction")
    scale = max(1., float(torch.linalg.vector_norm(base.to(torch.float64))))

    def assign(vector: torch.Tensor) -> None:
        offset = 0
        for parameter in parameters:
            parameter.copy_(vector[offset:offset + parameter.numel()].reshape(parameter.shape))
            offset += parameter.numel()

    baseline = float(evaluate_loss())
    if not np.isfinite(baseline):
        raise RuntimeError("Non-finite terminal objective before descent probes")
    best, evaluations, rounded_away = baseline, 1, 0
    try:
        for direction in normalized:
            for distance in relative_distances:
                for sign in (-1., 1.):
                    candidate = (base.to(torch.float64) + sign * distance * scale * direction).to(original.dtype)
                    if torch.equal(candidate, base):
                        rounded_away += 1
                        continue
                    if from_probe_coordinates is not None:
                        candidate = from_probe_coordinates(candidate)
                    if (candidate.shape != original.shape or candidate.dtype != original.dtype
                            or candidate.device != original.device or not bool(torch.isfinite(candidate).all())):
                        raise ValueError("Inverse descent-probe coordinate map changed finite parameter geometry")
                    if torch.equal(candidate, original):
                        rounded_away += 1
                        continue
                    if to_probe_coordinates is not None:
                        realized = to_probe_coordinates(candidate)
                        if (realized.shape != base.shape or realized.dtype != base.dtype or realized.device != base.device
                                or not bool(torch.isfinite(realized).all())):
                            raise ValueError("Realized descent-probe coordinates changed finite parameter geometry")
                        intended_change = sign * distance * scale * direction
                        error = torch.linalg.vector_norm((realized - base).to(torch.float64) - intended_change)
                        if float(error) > .01 * float(torch.linalg.vector_norm(intended_change)):
                            raise RuntimeError("Descent-probe coordinate roundoff obscures the intended original-coordinate movement")
                    assign(candidate)
                    value = float(evaluate_loss())
                    if not np.isfinite(value):
                        raise RuntimeError("Non-finite terminal objective during descent probes")
                    evaluations += 1
                    best = min(best, value)
    finally:
        assign(original)
    return TerminalDescentProbe(baseline_loss=baseline, best_loss=best, tolerance=tolerance,
        evaluations=evaluations, rounded_away=rounded_away, direction_count=len(normalized))
