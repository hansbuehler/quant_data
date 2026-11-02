"""Full-reorthogonalized minimum-residual proposals in fixed positive P geometry."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from quantdata.gromm2.affine_audit_optimization import Progress


@torch.no_grad()
def weighted_arnoldi_minimum_residual(*, right: torch.Tensor,
                                    operator: Callable[[torch.Tensor], torch.Tensor],
                                    precondition: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                                    stopping_energy: Callable[[torch.Tensor], torch.Tensor],
                                    relative_tolerance: float, maximum_iterations: int,
                                    orthogonality_tolerance: float = 1.e-7,
                                    progress: Progress | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Minimize a P-residual without assuming a positive Hessian.

    Covector V and primal Z=P V satisfy V'Z=I. Orthogonalizing H Z in
    this geometry is equivalent to applying Euclidean Arnoldi to the
    symmetric P^(1/2) H P^(1/2). Two reorthogonalization passes retain the
    measured full Hessenberg matrix; QR solves the projected least squares,
    never normal equations or a coefficient-truncating pseudoinverse.

    Original-M stopping is independent of the minimized P-norm. Reported
    convergence always uses an actual counted H product at the returned
    direction. A false recursive gate restarts at the actual residual.
    This is not textbook short-recurrence MINRES or outer stationarity.
    A 64-epsilon relative Krylov breakdown stops further expansion only;
    an incompatible actual residual remains explicitly nonconverged.
    A relative rank-unreliable projected QR refuses its new coefficients
    without a pseudoinverse; the previous point gets an actual H check.
    """
    if (right.ndim != 1 or right.dtype != torch.float64 or not bool(torch.isfinite(right).all())
        or not bool(right.ne(0.).any()) or not 0. < relative_tolerance < 1.
        or type(maximum_iterations) is not int or maximum_iterations <= 0
        or not math.isfinite(orthogonality_tolerance) or not 0. < orthogonality_tolerance < 1.):
        raise ValueError('weighted Arnoldi needs a finite nonzero float64 right side and valid numerical controls')

    def norm(value: torch.Tensor) -> float:
        energy = float(stopping_energy(value))
        if not math.isfinite(energy) or energy < 0.:
            raise FloatingPointError('invalid original-M residual energy')
        return math.sqrt(energy)

    def product(value: torch.Tensor) -> torch.Tensor:
        image = operator(value)
        if (image.shape != right.shape or image.dtype != right.dtype or image.device != right.device
            or not bool(torch.isfinite(image).all())):
            raise FloatingPointError('weighted Arnoldi requires a finite compatible operator image')
        return image

    def positive(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, energy = precondition(value)
        if (image.shape != right.shape or image.dtype != right.dtype or image.device != right.device
            or not bool(torch.isfinite(image).all()) or energy.ndim != 0
            or not bool(torch.isfinite(energy)) or bool(energy < 0.)
            or bool(energy == 0.) and bool(value.ne(0.).any())):
            raise FloatingPointError('weighted Arnoldi requires a fixed positive compatible preconditioner')
        return image, energy

    initial = norm(right)
    if not math.isfinite(initial) or initial <= 0.:
        raise FloatingPointError('unrepresentable original-M initial residual')
    threshold = relative_tolerance*initial
    covectors = right.new_zeros((len(right), maximum_iterations+1))
    primals = torch.zeros_like(covectors)
    images = right.new_zeros((len(right), maximum_iterations))
    hessenberg = right.new_zeros((maximum_iterations+1, maximum_iterations))
    base, residual = torch.zeros_like(right), right.clone()
    solution, true_current = base.clone(), True
    iterations, restarts, maximum_orthogonality_error = 0, 0, 0.
    stop_reason = 'iteration_cap_not_converged'
    while iterations < maximum_iterations:
        primal, energy = positive(residual)
        beta = torch.sqrt(energy)
        if not bool(beta > 0.):
            raise FloatingPointError('a nonzero original-M residual has zero preconditioned energy')
        covectors[:, 0], primals[:, 0] = residual/beta, primal/beta
        initial_error = abs(float(torch.dot(covectors[:, 0], primals[:, 0]))-1.)
        maximum_orthogonality_error = max(maximum_orthogonality_error, initial_error)
        if not math.isfinite(initial_error) or initial_error > orthogonality_tolerance:
            raise FloatingPointError('weighted Arnoldi initial P-normalization is inconsistent')
        base_residual = residual.clone()
        target = right.new_zeros(maximum_iterations+1)
        target[0] = beta
        hessenberg.zero_()
        for column in range(maximum_iterations-iterations):
            current = column+1
            image = product(primals[:, column])
            iterations += 1
            images[:, column] = image
            image_scale = torch.sqrt(positive(image)[1])
            remainder = image.clone()
            for _ in range(2):
                coefficients = primals[:, :current].T@remainder
                hessenberg[:current, column].add_(coefficients)
                remainder.sub_(covectors[:, :current]@coefficients)
            next_primal, next_energy = positive(remainder)
            next_beta = torch.sqrt(next_energy)
            hessenberg[current, column] = next_beta
            breakdown_scale = image_scale+torch.linalg.vector_norm(hessenberg[:current, column])
            breakdown = bool(next_beta <= 64*torch.finfo(right.dtype).eps*breakdown_scale)
            if not breakdown:
                covectors[:, current], primals[:, current] = remainder/next_beta, next_primal/next_beta
                cross = primals[:, :current].T@covectors[:, current]
                diagonal = torch.dot(covectors[:, current], primals[:, current])-1.
                error = float(torch.maximum(cross.abs().max(), diagonal.abs()))
                maximum_orthogonality_error = max(maximum_orthogonality_error, error)
                if not math.isfinite(error) or error > orthogonality_tolerance:
                    raise FloatingPointError('weighted Arnoldi lost measured P-orthogonality; no direction selected')
            orthogonal, triangular = torch.linalg.qr(hessenberg[:current+1, :current], mode='reduced')
            projected_right = orthogonal.T@target[:current+1]
            projected_scale = float(torch.linalg.matrix_norm(triangular, ord=float('inf')))
            if (not math.isfinite(projected_scale)
                or bool((triangular.diagonal().abs() <= 64*torch.finfo(right.dtype).eps*projected_scale).any())):
                residual = right-product(solution)
                true_current = True
                stop_reason = ('actual_original_M_residual_converged' if norm(residual) <= threshold
                    else 'rank_unreliable_projected_problem_not_converged')
                break
            coefficients = torch.linalg.solve_triangular(triangular, projected_right[:, None], upper=True)[:, 0]
            solution = base+primals[:, :current]@coefficients
            if not bool(torch.isfinite(solution).all()):
                raise FloatingPointError('nonfinite weighted Arnoldi solution')
            residual = base_residual-images[:, :current]@coefficients
            recursive_gate = norm(residual) <= threshold
            true_current = recursive_gate or breakdown or iterations == maximum_iterations
            if true_current:
                residual = right-product(solution)
            current_norm = norm(residual)
            if progress is not None:
                progress({'stage': 'weighted_arnoldi_minimum_residual', 'iteration': iterations,
                    'maximum_iterations': maximum_iterations, 'original_metric_residual_norm': current_norm,
                    'original_metric_threshold': threshold, 'residual_is_true': true_current,
                    'false_true_gate_restarts': restarts, 'maximum_P_orthogonality_error': maximum_orthogonality_error})
            if true_current and current_norm <= threshold:
                stop_reason = 'actual_original_M_residual_converged'
                break
            if breakdown and not recursive_gate:
                stop_reason = 'krylov_breakdown_not_converged'
                break
            if true_current and iterations < maximum_iterations:
                base = solution.clone()
                restarts += 1
                break
        if stop_reason != 'iteration_cap_not_converged' or iterations >= maximum_iterations:
            break
    if not true_current:
        residual = right-product(solution)
    final_norm = norm(residual)
    return solution, residual, {'iterations': iterations, 'initial_metric_residual_norm': initial,
        'true_metric_residual_norm': final_norm, 'threshold': threshold, 'converged': final_norm <= threshold,
        'stop_reason': stop_reason, 'residual_is_true': True, 'false_true_gate_restarts': restarts,
        'maximum_P_orthogonality_error': maximum_orthogonality_error,
        'minimized_residual_geometry': 'fixed positive P', 'stopping_geometry': 'original fixed M',
        'short_recurrence_MINRES': False, 'SPD_Hessian_assumed': False,
        'projected_coefficient_truncation': False,
        'outer_gradient_convergence_certified': False}
