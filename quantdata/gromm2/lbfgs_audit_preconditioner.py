"""Frozen positive raw-history preconditioning with separate residual geometry."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy

import torch

from quantdata.gromm2.affine_audit_optimization import Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_continuation import optimizer_state_sha256
from quantdata.gromm2.gradient_reliable_ggn import verify_actual_metric
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS


def metric_energy(metric: AffineBlockMetric, value: torch.Tensor) -> torch.Tensor:
    """Compute v'Mv in factor coordinates rather than a cancelling dot product."""
    affine = metric.affine_values(value)
    deep = value.clone()
    deep[metric.weight[0]], deep[metric.bias[0]] = 0., 0.
    energy = metric.deep_scale*deep.square().sum()
    for factor, columns in zip(metric.factors, metric.columns, strict=True):
        energy.add_((factor.T@affine[:, columns]).square().sum())
    return energy


class FrozenLBFGSPreconditioner:
    """Snapshot a positive inverse-BFGS operator; never mutate native history."""

    def __init__(self, optimizer: MetricLBFGS, policy: AffineDeepAction) -> None:
        verify_actual_metric(optimizer.metric, policy)
        self.source_sha256 = optimizer_state_sha256(optimizer)
        self.metric = deepcopy(optimizer.metric, {id(parameter): parameter for parameter in optimizer.metric.parameters})
        verify_actual_metric(self.metric, policy)
        self.history = tuple(tuple(value.detach().clone() for value in entry) for entry in optimizer.history)
        for step, change, reciprocal in self.history:
            curvature = torch.dot(step, change)
            if (not all(bool(torch.isfinite(value).all()) for value in (step, change, reciprocal))
                or not bool(curvature > optimizer.curvature_floor*torch.linalg.vector_norm(step)*torch.linalg.vector_norm(change))
                or not bool(reciprocal > 0.) or abs(float(reciprocal*curvature)-1.) > 1.e-8):
                raise ValueError('frozen preconditioning requires every original finite reliable positive secant')
        self.objects = (self.metric.factors, *self.metric.columns, *(value for entry in self.history for value in entry))
        self.versions = tuple(value._version for value in self.objects)
        self.controls = (self.metric.deep_scale, deepcopy(self.metric.slices), self.metric.weight, self.metric.bias)

    def check_fixed(self) -> None:
        """Reject operator or geometry mutation before any benchmark product."""
        actual = (self.metric.factors, *self.metric.columns, *(tensor for entry in self.history for tensor in entry))
        if (len(actual) != len(self.objects)
            or any(current is not saved for current, saved in zip(actual, self.objects, strict=True))
            or tuple(tensor._version for tensor in actual) != self.versions
            or (self.metric.deep_scale, self.metric.slices, self.metric.weight, self.metric.bias) != self.controls):
            raise ValueError('the frozen positive preconditioner changed during its solve')

    @torch.no_grad()
    def apply_and_energy(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.check_fixed()
        residual, coefficients = value.clone(), []
        energy = torch.zeros((), dtype=value.dtype, device=value.device)
        for step, change, reciprocal in reversed(self.history):
            projection = torch.dot(step, residual)
            coefficient = reciprocal*projection
            coefficients.append(coefficient)
            energy.add_((torch.sqrt(reciprocal)*projection).square())
            residual.sub_(coefficient*change)
        energy.add_(metric_energy(self.metric, residual))
        image = self.metric(residual)
        for (step, change, reciprocal), coefficient in zip(self.history, reversed(coefficients), strict=True):
            image.add_(step*(coefficient-reciprocal*torch.dot(change, image)))
        return image, energy


@torch.no_grad()
def pcg_separate_residual_norm(operator: Callable, right: torch.Tensor, precondition: Callable, *,
                              stopping_energy: Callable, relative_tolerance: float,
                              maximum_iterations: int, progress: Progress | None = None) -> tuple[torch.Tensor, dict]:
    """Use r'Pr for PCG recurrences and the original r'Mr only for stopping.

    Both operators must remain fixed and SPD. Every converged report uses a
    separately evaluated TRUE residual; recursive agreement alone is never a
    certificate. This solves only a numerical linear system, not the original
    nonlinear stationarity problem.
    """
    if (right.ndim != 1 or right.dtype != torch.float64 or not bool(torch.isfinite(right).all())
        or not 0. < relative_tolerance < 1. or type(maximum_iterations) is not int or maximum_iterations <= 0):
        raise ValueError('PCG requires a finite float64 vector and valid numerical controls')

    def norm(value: torch.Tensor) -> float:
        energy = float(stopping_energy(value))
        if not math.isfinite(energy) or energy < 0.:
            raise FloatingPointError('invalid original-metric residual energy')
        return math.sqrt(energy)

    solution, residual = torch.zeros_like(right), right.clone()
    initial = norm(right)
    threshold = max(torch.finfo(right.dtype).eps, relative_tolerance*initial)
    transformed, energy = precondition(residual)
    direction = transformed.clone()
    true_current = True
    iterations = 0
    while norm(residual) > threshold and iterations < maximum_iterations:
        image = operator(direction)
        denominator = torch.dot(direction, image)
        if (not bool(torch.isfinite(denominator)) or not bool(denominator > 0.)
            or not bool(torch.isfinite(energy)) or not bool(energy > 0.)):
            raise RuntimeError('PCG requires fixed finite positive recurrence curvature')
        rate = energy/denominator
        solution.add_(rate*direction)
        residual.sub_(rate*image)
        iterations += 1
        true_current = iterations % 10 == 0
        if true_current:
            residual = right-operator(solution)
        current_norm = norm(residual)
        restarted = False
        if current_norm <= threshold and not true_current:
            residual = right-operator(solution)
            current_norm = norm(residual)
            true_current, restarted = True, current_norm > threshold
        if progress is not None:
            progress({'stage': 'separate_norm_pcg', 'iteration': iterations, 'maximum_iterations': maximum_iterations,
                'original_metric_residual_norm': current_norm, 'original_metric_threshold': threshold,
                'residual_is_true': true_current, 'true_residual_restart': restarted})
        if current_norm <= threshold:
            break
        transformed, next_energy = precondition(residual)
        direction = transformed.clone() if restarted else transformed+(next_energy/energy)*direction
        energy = next_energy
    if not true_current:
        residual = right-operator(solution)
    final_norm = norm(residual)
    if not bool(torch.isfinite(solution).all()):
        raise FloatingPointError('nonfinite PCG solution')
    return solution, {'iterations': iterations, 'original_metric_initial_residual_norm': initial,
        'original_metric_residual_norm': final_norm, 'original_metric_threshold': threshold,
        'euclidean_residual_norm': float(torch.linalg.vector_norm(residual)), 'converged': final_norm <= threshold,
        'stopping_geometry': 'original fixed M', 'recurrence_geometry': 'fixed positive P',
        'outer_gradient_convergence_certified': False}
