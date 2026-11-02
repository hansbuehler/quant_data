"""Right-preconditioned GMRES minimizing the original fixed-M residual."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from quantdata.gromm2.affine_audit_optimization import Progress


@torch.no_grad()
def metric_right_gmres(*, right: torch.Tensor,
                                    operator: Callable[[torch.Tensor], torch.Tensor],
                                    precondition: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                                    metric_embedding: Callable[[torch.Tensor], torch.Tensor],
                                    relative_tolerance: float, maximum_iterations: int,
                                    orthogonality_tolerance: float = 1.e-7,
                                    progress: Progress | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Minimize and stop in the same original-M residual geometry.

    M=C C' and embedding(v)=C'v. Covectors V are M-orthonormal;
    primal directions Z=P V use the fixed positive right preconditioner.
    Arnoldi on H P need not be symmetric: this is weighted GMRES, not
    MINRES. Its exact-arithmetic solution subspace is the same as the
    P-weighted control, but its least-squares coefficients minimize M,
    the unchanged original stopping geometry.

    Two-pass reorthogonalization uses stable factor-coordinate products,
    not a cancelling materialized M bilinear form. Full Hessenberg QR,
    relative numerical rank/breakdown refusals, actual counted H residual
    checks and false-gate restarts preserve the control's safeguards.
    No coefficient truncation, shift or outer stationarity is implied.
    """
    if (right.ndim != 1 or right.dtype != torch.float64 or not bool(torch.isfinite(right).all())
        or not bool(right.ne(0.).any()) or not 0. < relative_tolerance < 1.
        or type(maximum_iterations) is not int or maximum_iterations <= 0
        or not math.isfinite(orthogonality_tolerance) or not 0. < orthogonality_tolerance < 1.):
        raise ValueError('original-M right-preconditioned GMRES needs a finite nonzero float64 right side and valid numerical controls')

    def embed(value: torch.Tensor) -> torch.Tensor:
        transformed = metric_embedding(value)
        if (transformed.shape != right.shape or transformed.dtype != right.dtype
            or transformed.device != right.device or not bool(torch.isfinite(transformed).all())):
            raise FloatingPointError('original-M embedding must return a finite compatible factor vector')
        return transformed

    def norm(value: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(embed(value)))

    def normalized_basis(value: torch.Tensor, embedded: torch.Tensor,
                         beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = value/beta
        expected = embedded/beta
        actual = embed(normalized)
        discrepancy = float(torch.linalg.vector_norm(actual-expected))
        scale = float(torch.linalg.vector_norm(expected))
        if (not math.isfinite(discrepancy) or not math.isfinite(scale) or scale <= 0.
            or discrepancy > orthogonality_tolerance*scale):
            raise FloatingPointError('normalized raw GMRES basis differs from its actual original-M embedding')
        return normalized, actual

    def product(value: torch.Tensor) -> torch.Tensor:
        image = operator(value)
        if (image.shape != right.shape or image.dtype != right.dtype or image.device != right.device
            or not bool(torch.isfinite(image).all())):
            raise FloatingPointError('original-M right-preconditioned GMRES requires a finite compatible operator image')
        return image

    def positive(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, energy = precondition(value)
        if (image.shape != right.shape or image.dtype != right.dtype or image.device != right.device
            or not bool(torch.isfinite(image).all()) or energy.ndim != 0
            or not bool(torch.isfinite(energy)) or bool(energy < 0.)
            or bool(energy == 0.) and bool(value.ne(0.).any())):
            raise FloatingPointError('original-M right-preconditioned GMRES requires a fixed positive compatible preconditioner')
        return image, energy

    initial = norm(right)
    if not math.isfinite(initial) or initial <= 0.:
        raise FloatingPointError('unrepresentable original-M initial residual')
    threshold = relative_tolerance*initial
    covectors = right.new_zeros((len(right), maximum_iterations+1))
    primals = torch.zeros_like(covectors)
    embedded_basis = torch.zeros_like(covectors)
    images = right.new_zeros((len(right), maximum_iterations))
    hessenberg = right.new_zeros((maximum_iterations+1, maximum_iterations))
    base, residual = torch.zeros_like(right), right.clone()
    solution, true_current = base.clone(), True
    iterations, restarts, maximum_orthogonality_error = 0, 0, 0.
    stop_reason = 'iteration_cap_not_converged'
    while iterations < maximum_iterations:
        embedded = embed(residual)
        beta = torch.linalg.vector_norm(embedded)
        if not bool(beta > 0.):
            raise FloatingPointError('a nonzero original-M residual has zero embedded energy')
        covectors[:, 0], embedded_basis[:, 0] = normalized_basis(residual, embedded, beta)
        initial_error = abs(float(embedded_basis[:, 0].square().sum())-1.)
        maximum_orthogonality_error = max(maximum_orthogonality_error, initial_error)
        if not math.isfinite(initial_error) or initial_error > orthogonality_tolerance:
            raise FloatingPointError('original-M GMRES initial normalization is inconsistent')
        primals[:, 0] = positive(covectors[:, 0])[0]
        base_residual = residual.clone()
        target = right.new_zeros(maximum_iterations+1)
        target[0] = beta
        hessenberg.zero_()
        for column in range(maximum_iterations-iterations):
            current = column+1
            image = product(primals[:, column])
            iterations += 1
            images[:, column] = image
            image_scale = torch.linalg.vector_norm(embed(image))
            remainder = image.clone()
            for _ in range(2):
                coefficients = embedded_basis[:, :current].T@embed(remainder)
                hessenberg[:current, column].add_(coefficients)
                remainder.sub_(covectors[:, :current]@coefficients)
            next_embedded = embed(remainder)
            next_beta = torch.linalg.vector_norm(next_embedded)
            hessenberg[current, column] = next_beta
            breakdown_scale = image_scale+torch.linalg.vector_norm(hessenberg[:current, column])
            breakdown = bool(next_beta <= 64*torch.finfo(right.dtype).eps*breakdown_scale)
            if not breakdown:
                covectors[:, current], embedded_basis[:, current] = normalized_basis(remainder, next_embedded, next_beta)
                cross = embedded_basis[:, :current].T@embedded_basis[:, current]
                diagonal = embedded_basis[:, current].square().sum()-1.
                error = float(torch.maximum(cross.abs().max(), diagonal.abs()))
                maximum_orthogonality_error = max(maximum_orthogonality_error, error)
                if not math.isfinite(error) or error > orthogonality_tolerance:
                    raise FloatingPointError('original-M right-preconditioned GMRES lost measured M-orthogonality; no direction selected')
                primals[:, current] = positive(covectors[:, current])[0]
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
                raise FloatingPointError('nonfinite original-M right-preconditioned GMRES solution')
            residual = base_residual-images[:, :current]@coefficients
            recursive_gate = norm(residual) <= threshold
            true_current = recursive_gate or breakdown or iterations == maximum_iterations
            if true_current:
                residual = right-product(solution)
            current_norm = norm(residual)
            if progress is not None:
                progress({'stage': 'original_M_right_preconditioned_GMRES', 'iteration': iterations,
                    'maximum_iterations': maximum_iterations, 'original_metric_residual_norm': current_norm,
                    'original_metric_threshold': threshold, 'residual_is_true': true_current,
                    'false_true_gate_restarts': restarts, 'maximum_M_orthogonality_error': maximum_orthogonality_error})
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
        'maximum_M_orthogonality_error': maximum_orthogonality_error,
        'minimized_residual_geometry': 'original fixed M', 'stopping_geometry': 'original fixed M',
        'MINRES': False, 'symmetric_HP_assumed': False, 'SPD_Hessian_assumed': False,
        'right_preconditioner_geometry': 'fixed positive P',
        'projected_coefficient_truncation': False,
        'outer_gradient_convergence_certified': False}
