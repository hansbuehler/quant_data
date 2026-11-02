"""Restart a numerical solve without changing its original absolute target."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from quantdata.gromm2.affine_audit_optimization import Progress
from quantdata.gromm2.lbfgs_audit_preconditioner import pcg_separate_residual_norm


@torch.no_grad()
def restart_pcg_original_target(operator: Callable, right: torch.Tensor, initial: torch.Tensor,
                                precondition: Callable, *, stopping_energy: Callable,
                                original_absolute_threshold: float, maximum_iterations: int,
                                progress: Progress | None = None) -> tuple[torch.Tensor, dict]:
    """Solve for a correction, then check the combined original true residual.

    This is restarted warm-start PCG, not continuation of a saved Krylov
    recurrence. Its absolute threshold remains the original right-hand-side
    target, not a new relative tolerance applied to the smaller residual.
    """
    if (right.ndim != 1 or right.dtype != torch.float64 or initial.shape != right.shape
        or initial.dtype != right.dtype or initial.device != right.device
        or not bool(torch.isfinite(right).all()) or not bool(torch.isfinite(initial).all())
        or not math.isfinite(original_absolute_threshold) or original_absolute_threshold <= 0.
        or type(maximum_iterations) is not int or maximum_iterations <= 0):
        raise ValueError('warm-start correction requires finite float64 vectors and the original positive target')

    def norm(value: torch.Tensor) -> float:
        energy = float(stopping_energy(value))
        if not math.isfinite(energy) or energy < 0.:
            raise FloatingPointError('invalid original-metric warm-start residual energy')
        return math.sqrt(energy)

    initial_residual = right-operator(initial)
    initial_norm = norm(initial_residual)
    if initial_norm <= original_absolute_threshold:
        direction = initial.clone()
        correction = {'iterations': 0, 'converged': True, 'no_correction_required': True}
    else:
        delta, correction = pcg_separate_residual_norm(operator, initial_residual, precondition,
            stopping_energy=stopping_energy, relative_tolerance=original_absolute_threshold/initial_norm,
            maximum_iterations=maximum_iterations, progress=progress)
        direction = initial+delta
    residual = right-operator(direction)
    residual_norm = norm(residual)
    if not bool(torch.isfinite(direction).all()):
        raise FloatingPointError('nonfinite combined warm-start direction')
    return direction, {'iterations': correction['iterations'], 'correction_solve': correction,
        'original_metric_initial_residual_norm': norm(right),
        'original_metric_warm_start_residual_norm': initial_norm,
        'original_metric_residual_norm': residual_norm, 'original_metric_threshold': original_absolute_threshold,
        'euclidean_residual_norm': float(torch.linalg.vector_norm(residual)),
        'converged': residual_norm <= original_absolute_threshold,
        'restarted_warm_start_not_uninterrupted_krylov': True,
        'outer_gradient_convergence_certified': False}
