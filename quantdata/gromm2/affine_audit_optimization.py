"""Deterministic, training-only affine optimization of the original diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

Progress = Callable[[dict], None]


@version('0.2')
@dataclass(frozen=True, kw_only=True)
class AffineSolveConfig:
    batch_size: int = 32768
    preconditioner_relative_floor: float = 1.e-12
    gradient_relative_tolerance: float = 1.e-6
    gradient_absolute_tolerance: float = 1.e-9
    cg_relative_tolerance: float = 1.e-8
    maximum_cg_iterations: int = 200
    maximum_newton_iterations: int = 30
    armijo: float = 1.e-4
    minimum_step: float = 2.**-30

    def __post_init__(self) -> None:
        verify_inp(self.batch_size > 0 and min(self.maximum_cg_iterations, self.maximum_newton_iterations) > 0,
            lambda: 'affine optimization requires positive batch and iteration limits')
        verify_inp(0. < self.preconditioner_relative_floor < 1. and 0. < self.cg_relative_tolerance < 1.,
            lambda: 'invalid preconditioner or CG tolerances')
        verify_inp(self.gradient_relative_tolerance > 0. and self.gradient_absolute_tolerance > 0.,
            lambda: 'gradient tolerances must be positive')
        verify_inp(0. < self.armijo < .5 and 0. < self.minimum_step < 1., lambda: 'invalid line search controls')


def preconditioned_cg(operator: Callable, right: torch.Tensor, precondition: Callable, *,
                      relative_tolerance: float, maximum_iterations: int,
                      residual_energy: Callable | None = None,
                      progress: Progress | None = None) -> tuple[torch.Tensor, dict]:
    """Solve the affine Hessian without discarding low-curvature directions."""
    solution = torch.zeros_like(right)
    residual = right.clone()
    def energy_of(value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value*precondition(value)) if residual_energy is None else residual_energy(value)

    initial = math.sqrt(max(float(energy_of(residual)), 0.))
    threshold = max(torch.finfo(right.dtype).eps, initial * relative_tolerance)
    direction = precondition(residual).clone()
    energy = energy_of(residual)
    iterations, norm = 0, initial
    while norm > threshold and iterations < maximum_iterations:
        image = operator(direction)
        denominator = torch.sum(direction * image)
        if not torch.isfinite(denominator) or float(denominator) <= 0. or float(energy) <= 0.:
            raise RuntimeError('CG encountered an unidentifiable or nonpositive affine direction')
        rate = energy / denominator
        solution.add_(rate * direction)
        residual.sub_(rate * image)
        iterations += 1
        if iterations % 10 == 0:
            residual = right - operator(solution)
        norm = math.sqrt(max(float(energy_of(residual)), 0.))
        if progress is not None:
            progress({'stage': 'affine_full_gradient_cg', 'iteration': iterations,
                'maximum_iterations': maximum_iterations, 'residual_norm': norm, 'threshold': threshold})
        if norm <= threshold:
            break
        transformed = precondition(residual)
        next_energy = energy_of(residual)
        direction = transformed + (next_energy / energy) * direction
        energy = next_energy
    true_residual = right-operator(solution)
    true_norm = math.sqrt(max(float(energy_of(true_residual)), 0.))
    return solution, {'iterations': iterations, 'residual_norm': true_norm,
        'euclidean_residual_norm': float(torch.linalg.vector_norm(true_residual)),
        'threshold': threshold, 'converged': true_norm <= threshold}


class AffineAuditProblem:
    """An unchanged empirical objective on explicitly selected training rows.

    Resident features and returns use the original training normalization.
    Parameters have shape (features+intercept, instruments). All optimization
    reductions are float64, even when retained observations are float32.
    A positive eigenvalue floor stabilizes only the preconditioner, not the
    objective or Hessian. No coefficient directions are discarded or capped.
    """

    def __init__(self, *, states: torch.Tensor, returns: torch.Tensor,
                 eligible: torch.Tensor, rows: torch.Tensor, action_scale: float,
                 squared_lambda: float = 1., config: AffineSolveConfig = AffineSolveConfig(),
                 progress: Progress | None = None) -> None:
        verify_inp(states.ndim == 2 and returns.ndim == 2 and len(states) == len(returns),
            lambda: 'affine state and return observations must agree')
        verify_inp(eligible.dtype == torch.bool and eligible.ndim == 2 and eligible.shape[1] == returns.shape[1],
            lambda: 'eligibility must contain every day and instrument')
        verify_inp(rows.ndim == 1 and len(rows) > 0 and rows.dtype in (torch.int32, torch.int64),
            lambda: 'explicit training-row indices are required')
        verify_inp(math.isfinite(action_scale) and action_scale > 0. and math.isfinite(squared_lambda) and squared_lambda >= 0.,
            lambda: 'invalid action scale or squared lambda')
        verify_inp(all(value.device == states.device for value in (returns, eligible, rows)),
            lambda: 'affine data must share one managed device')
        if int(rows.min()) < 0 or int(rows.max()) >= len(states):
            raise ValueError('training row lies outside resident observations')
        self.states, self.returns, self.eligible, self.rows = states, returns, eligible, rows
        self.action_scale, self.squared_lambda = action_scale, squared_lambda
        self.config, self.progress = config, progress
        self.parameter_shape = (states.shape[1]+1, returns.shape[1])
        self.group_masks, groups = torch.unique(eligible.T, dim=0, return_inverse=True)
        self.group_columns = tuple(torch.where(groups == group)[0] for group in range(len(self.group_masks)))
        self.grams = self.preconditioner_factors = self.cross = None
        self.rank_report = None

    def blocks(self, rows: torch.Tensor | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        selected = self.rows if rows is None else rows
        for start in range(0, len(selected), self.config.batch_size):
            indices = selected[start:start+self.config.batch_size]
            features = self.states[indices].to(torch.float64)
            augmented = torch.cat((features, torch.ones((len(indices), 1), device=features.device, dtype=features.dtype)), dim=1)
            yield augmented, self.returns[indices].to(torch.float64), self.eligible[indices % len(self.eligible)]
            if self.progress is not None:
                self.progress({'stage': 'affine_observation_pass', 'completed_observations': start+len(indices),
                    'total_observations': len(selected)})

    @torch.no_grad()
    def prepare(self) -> dict:
        width = self.parameter_shape[0]
        self.grams = torch.zeros((len(self.group_masks), width, width), dtype=torch.float64, device=self.states.device)
        self.cross = torch.zeros(self.parameter_shape, dtype=torch.float64, device=self.states.device)
        patterns, pattern_indices = torch.unique(self.eligible, dim=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            selected = self.rows[pattern_indices[self.rows % len(self.eligible)] == index]
            if not len(selected):
                continue
            gram = torch.zeros_like(self.grams[0])
            for augmented, returns, mask in self.blocks(selected):
                gram.add_(augmented.T @ augmented)
                self.cross.add_(augmented.T @ (returns * mask))
            for group, columns in enumerate(self.group_columns):
                if bool(pattern[columns[0]]):
                    self.grams[group].add_(gram)
        self.grams.div_(len(self.rows))
        self.cross.div_(len(self.rows))
        eigenvalues, eigenvectors = torch.linalg.eigh(self.grams)
        cutoff = torch.maximum(self.config.preconditioner_relative_floor * eigenvalues[:, -1:],
            torch.full_like(eigenvalues[:, -1:], torch.finfo(eigenvalues.dtype).eps))
        stabilized = torch.maximum(eigenvalues, cutoff)
        self.preconditioner_factors = eigenvectors / torch.sqrt(stabilized)[:, None]
        self.rank_report = {'eigenvalues_above_preconditioner_floor': (eigenvalues > cutoff).sum(dim=1).cpu().tolist(),
            'width': width, 'preconditioner_relative_floor': self.config.preconditioner_relative_floor,
            'maximum_eigenvalues': eigenvalues[:, -1].cpu().tolist(),
            'minimum_eigenvalues': eigenvalues[:, 0].cpu().tolist(),
            'objective_ridge_penalty_added': False, 'coefficient_directions_discarded': False}
        if not torch.isfinite(self.cross).all() or not torch.isfinite(self.preconditioner_factors).all():
            raise FloatingPointError('nonfinite affine moments; no observations may be discarded')
        return self.rank_report

    def precondition(self, value: torch.Tensor) -> torch.Tensor:
        if self.preconditioner_factors is None:
            raise RuntimeError('affine moments must be prepared first')
        result = torch.empty_like(value)
        for group, columns in enumerate(self.group_columns):
            factor = self.preconditioner_factors[group]
            result[:, columns] = factor @ (factor.T @ value[:, columns])
        return result

    def residual_energy(self, value: torch.Tensor) -> torch.Tensor:
        if self.preconditioner_factors is None:
            raise RuntimeError('affine moments must be prepared first')
        result = torch.zeros((), device=value.device, dtype=value.dtype)
        for group, columns in enumerate(self.group_columns):
            result.add_((self.preconditioner_factors[group].T @ value[:, columns]).square().sum())
        return result

    @torch.no_grad()
    def evaluate(self, weights: torch.Tensor, *, diagnostic: str, gradient: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        if diagnostic not in ('squared', 'entropy') or weights.shape != self.parameter_shape or weights.dtype != torch.float64:
            raise ValueError('invalid affine objective or parameter geometry/precision')
        loss = torch.zeros((), dtype=weights.dtype, device=weights.device)
        derivative = torch.zeros_like(weights) if gradient else None
        squared_scale = self.action_scale**2
        for augmented, returns, mask in self.blocks():
            action = (augmented @ weights) * mask
            gain = (action * returns).sum(dim=1)
            value = -gain+.5*action.square().sum(dim=1)
            if diagnostic == 'squared':
                loss.add_((value+.5*self.squared_lambda*squared_scale*gain.square()).sum())
                if gradient:
                    direction = action-returns+self.squared_lambda*squared_scale*gain[:, None]*returns
            else:
                exponent = squared_scale * value
                loss.add_((torch.expm1(exponent)/squared_scale).sum())
                if gradient:
                    direction = torch.exp(exponent)[:, None] * (action-returns)
            if gradient:
                derivative.add_(augmented.T @ (direction * mask))
        return loss/len(self.rows), None if derivative is None else derivative/len(self.rows)

    @torch.no_grad()
    def hessian(self, weights: torch.Tensor, direction: torch.Tensor, *, diagnostic: str) -> torch.Tensor:
        result = torch.zeros_like(direction)
        squared_scale = self.action_scale**2
        for augmented, returns, mask in self.blocks():
            delta = (augmented @ direction) * mask
            if diagnostic == 'squared':
                image = delta+self.squared_lambda*squared_scale*(delta*returns).sum(dim=1)[:, None]*returns
            elif diagnostic == 'entropy':
                action = (augmented @ weights) * mask
                residual = action-returns
                exponent = squared_scale * (-(action*returns).sum(dim=1)+.5*action.square().sum(dim=1))
                image = torch.exp(exponent)[:, None] * (delta+squared_scale*(delta*residual).sum(dim=1)[:, None]*residual)
            else:
                raise ValueError('unknown affine diagnostic')
            result.add_(augmented.T @ (image * mask))
        return result/len(self.rows)

    @torch.no_grad()
    def solve(self, *, diagnostic: str, initial: torch.Tensor | None = None,
              checkpoint: Callable[[torch.Tensor, dict], None] | None = None) -> tuple[torch.Tensor, dict]:
        if self.grams is None:
            self.prepare()
        weights = torch.zeros(self.parameter_shape, dtype=torch.float64, device=self.states.device) if initial is None else initial.clone()
        zero = torch.zeros_like(weights)
        _, baseline_gradient = self.evaluate(zero, diagnostic=diagnostic)
        baseline_norm = float(torch.linalg.vector_norm(baseline_gradient))
        threshold = self.config.gradient_absolute_tolerance+self.config.gradient_relative_tolerance*baseline_norm
        baseline_dual = math.sqrt(float(self.residual_energy(baseline_gradient)))
        dual_threshold = self.config.gradient_absolute_tolerance+self.config.gradient_relative_tolerance*baseline_dual
        history = []
        for iteration in range(self.config.maximum_newton_iterations+1):
            loss, derivative = self.evaluate(weights, diagnostic=diagnostic)
            norm = float(torch.linalg.vector_norm(derivative))
            dual_norm = math.sqrt(float(self.residual_energy(derivative)))
            converged = norm <= threshold and dual_norm <= dual_threshold
            entry = {'iteration': iteration, 'loss': float(loss), 'gradient_norm': norm, 'gradient_threshold': threshold,
                'preconditioned_gradient_norm': dual_norm, 'preconditioned_gradient_threshold': dual_threshold}
            history.append(entry)
            if checkpoint is not None:
                checkpoint(weights, {'diagnostic': diagnostic, 'history': history,
                    'gradient_converged': converged, 'rank_report': self.rank_report})
            if self.progress is not None:
                self.progress({'stage': 'affine_newton_iteration', 'diagnostic': diagnostic, **entry})
            if not math.isfinite(float(loss)) or not math.isfinite(norm):
                raise FloatingPointError('nonfinite deterministic affine objective')
            if converged or iteration == self.config.maximum_newton_iterations:
                break
            update, linear = preconditioned_cg(lambda value: self.hessian(weights, value, diagnostic=diagnostic),
                -derivative, self.precondition, relative_tolerance=self.config.cg_relative_tolerance,
                maximum_iterations=self.config.maximum_cg_iterations, residual_energy=self.residual_energy, progress=self.progress)
            entry['linear_solve'] = linear
            slope = float(torch.sum(update * derivative))
            if not slope < 0.:
                raise RuntimeError('deterministic affine update is not a descent direction')
            rate = 1.
            while rate >= self.config.minimum_step:
                proposed, _ = self.evaluate(weights+rate*update, diagnostic=diagnostic, gradient=False)
                if math.isfinite(float(proposed)) and float(proposed) <= float(loss)+self.config.armijo*rate*slope:
                    break
                rate *= .5
            if rate < self.config.minimum_step:
                raise RuntimeError('affine line search failed; no convergence claim is permitted')
            weights.add_(rate*update)
            entry['step_size'] = rate
        return weights, {'diagnostic': diagnostic, 'training_rows': len(self.rows), 'history': history,
            'converged': converged, 'zero_gradient_norm': baseline_norm,
            'zero_preconditioned_gradient_norm': baseline_dual, 'exact_global_optimum_certified': False,
            'rank_report': self.rank_report, 'unchanged_original_objective': True,
            'gradient_clipping': False, 'training_only': True, 'action_scale': self.action_scale,
            'squared_lambda': self.squared_lambda}
