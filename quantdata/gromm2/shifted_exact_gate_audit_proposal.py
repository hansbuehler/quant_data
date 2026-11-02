"""Shifted exact-H proposals with gate/end-only true-residual confirmation."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy

import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.full_hessian_audit import FullActionHessian
from quantdata.gromm2.ggn_audit_continuation import (
    _forward_geometry,
    action_point_sha256,
    optimizer_state_sha256,
)
from quantdata.gromm2.ggn_audit_optimization import InverseAffineMetric
from quantdata.gromm2.gradient_reliable_ggn import full_metric_norm, verify_actual_metric
from quantdata.gromm2.lbfgs_audit_preconditioner import FrozenLBFGSPreconditioner, metric_energy
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS
from quantdata.gromm2.shifted_exact_audit_proposal import (
    HessianWorkLimit,
    NegativeShiftedCurvature,
    ShiftedExactConfig,
)
from quantdata.gromm2.trust_region_audit_optimization import OriginalMetricCoordinates


@torch.no_grad()
def guarded_shifted_pcg(*, right: torch.Tensor,
                        operator: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                        precondition: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                        stopping_energy: Callable[[torch.Tensor], torch.Tensor],
                        relative_tolerance: float, maximum_iterations: int,
                        progress: Progress | None = None) -> tuple[torch.Tensor, dict]:
    """Use verified positive sampled curvature, or exit before a CG division.

    The operator must raise on nonpositive/ambiguous curvature. Its second
    return is the stable quadratic energy, not a recancelled dot product.
    Confirm the true residual only at a recursive gate or the iteration cap.
    A failed recursive gate replaces the residual and restarts the recurrence;
    no unconditional periodic replacement discards Krylov conjugacy.
    """
    if (right.ndim != 1 or right.dtype != torch.float64 or not bool(torch.isfinite(right).all())
        or not bool(right.ne(0.).any()) or not 0. < relative_tolerance < 1.
        or type(maximum_iterations) is not int or maximum_iterations <= 0):
        raise ValueError('guarded shifted PCG needs a finite nonzero float64 right side and valid controls')

    def norm(value: torch.Tensor) -> float:
        energy = float(stopping_energy(value))
        if not math.isfinite(energy) or energy < 0.:
            raise FloatingPointError('invalid original-M stopping energy')
        return math.sqrt(energy)

    def apply(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, curvature = operator(vector)
        if (image.shape != right.shape or image.dtype != right.dtype or image.device != right.device
            or not bool(torch.isfinite(image).all()) or curvature.ndim != 0
            or not bool(torch.isfinite(curvature)) or not bool(curvature > 0.)):
            raise FloatingPointError('guarded shifted operator must return finite compatible image and positive stable curvature')
        return image, curvature

    solution, residual = torch.zeros_like(right), right.clone()
    initial = norm(right)
    if not math.isfinite(initial) or initial <= 0.:
        raise FloatingPointError('unrepresentable original-M initial residual')
    threshold = relative_tolerance*initial
    transformed, energy = precondition(residual)
    direction = transformed.clone()
    iterations, true_current = 0, True
    for iterations in range(1, maximum_iterations+1):
        image, curvature = apply(direction)
        if not bool(torch.isfinite(energy)) or not bool(energy > 0.):
            raise FloatingPointError('invalid fixed-positive preconditioner recurrence energy')
        rate = energy/curvature
        solution.add_(rate*direction)
        residual.sub_(rate*image)
        recursive_norm = norm(residual)
        true_current = recursive_norm <= threshold
        if true_current:
            residual = right-operator(solution)[0]
        current_norm = norm(residual)
        if progress is not None:
            progress({'stage': 'guarded_shifted_exact_pcg', 'iteration': iterations,
                'maximum_iterations': maximum_iterations, 'original_metric_residual_norm': current_norm,
                'original_metric_threshold': threshold, 'true_residual_restart': true_current})
        if true_current and current_norm <= threshold:
            break
        transformed, next_energy = precondition(residual)
        direction = transformed.clone() if true_current else transformed+(next_energy/energy)*direction
        energy = next_energy
    if not true_current:
        residual = right-operator(solution)[0]
    final_norm = norm(residual)
    if not bool(torch.isfinite(solution).all()):
        raise FloatingPointError('nonfinite guarded shifted solution')
    return solution, {'iterations': iterations, 'initial_metric_residual_norm': initial,
        'true_metric_residual_norm': final_norm, 'threshold': threshold, 'converged': final_norm <= threshold,
        'residual_is_true': True, 'outer_gradient_convergence_certified': False,
        'SPD_spectrum_certified': False}


@torch.no_grad()
def propose_shifted_exact_step(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                              optimizer: MetricLBFGS, gradient: torch.Tensor, diagnostic: str,
                              point_sha256: str, config: ShiftedExactConfig = ShiftedExactConfig(),
                              progress: Progress | None = None) -> tuple[torch.Tensor | None, dict]:
    """Return a quality-checked shadow direction, or honestly refuse one.

    Frozen raw-history P preconditions a numerical shifted linear system;
    H is the exact original empirical Hessian. Shift trials share ONE product
    budget. Neither a shift nor a converged inner residual changes the loss,
    original outer gradient gates, policy, history, data or probability law.
    """
    metric = optimizer.metric
    verify_actual_metric(metric, policy)
    if (policy.affine_only or gradient.shape != (metric.size,) or gradient.dtype != torch.float64
        or gradient.device != metric.factors.device or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.ne(0.).any()) or action_point_sha256(policy) != point_sha256):
        raise ValueError('shifted exact-H proposal requires the actual original full-policy gradient and point')
    frozen = FrozenLBFGSPreconditioner(optimizer, policy)
    coordinates, inverse = OriginalMetricCoordinates(metric), InverseAffineMetric(metric)
    exact = FullActionHessian(policy, problem, diagnostic=diagnostic, compile=config.compile, progress=progress)
    snapshot = (*exact.parameters.items(), *exact.buffers.items())
    native = (*policy.named_parameters(), *policy.named_buffers())
    versions = tuple(value._version for name, value in (*snapshot, *native))
    source_geometry, source_config = _forward_geometry(policy), deepcopy(policy.config)

    def check_fixed() -> None:
        current = (*exact.parameters.items(), *exact.buffers.items(), *policy.named_parameters(), *policy.named_buffers())
        if (len(current) != len(versions)
            or any(name != saved_name or value is not saved for (name, value), (saved_name, saved)
                in zip(current, (*snapshot, *native), strict=True))
            or tuple(value._version for name, value in current) != versions
            or policy.affine_only or exact.functional_policy.affine_only
            or policy.config != source_config or exact.functional_policy.config != source_config
            or _forward_geometry(policy) != source_geometry or _forward_geometry(exact.functional_policy) != source_geometry
            or action_point_sha256(policy) != point_sha256):
            raise ValueError('shifted exact-H native point or frozen Hessian snapshot changed')
        frozen.check_fixed()
        coordinates.check_fixed()

    def raw_product(vector: torch.Tensor) -> torch.Tensor:
        check_fixed()
        if exact.full_curvature_passes >= config.maximum_hessian_products:
            raise HessianWorkLimit('shared complete exact-H product budget exhausted; not convergence')
        image = exact.product(vector)
        check_fixed()
        return image

    initial_norms = metric.diagnostics(gradient)
    initial_metric_norm = full_metric_norm(metric, gradient)
    trials, shift, selected, selected_report = [], config.initial_shift, None, None
    for attempt in range(config.maximum_shift_trials):
        solve_shift = shift
        before = exact.full_curvature_passes
        entry = {'trial': attempt, 'shift': solve_shift}

        def operator(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if not bool(vector.ne(0.).any()):
                return torch.zeros_like(vector), torch.zeros((), dtype=vector.dtype, device=vector.device)
            image = raw_product(vector)
            inverse_coordinates = coordinates.apply(vector, operation='inverse')
            inverse_energy = inverse_coordinates.square().sum()
            raw_curvature = torch.dot(vector, image)
            stable_curvature = raw_curvature+solve_shift*inverse_energy
            materialized = image+solve_shift*inverse(vector)
            measured_curvature = torch.dot(vector, materialized)
            scale = float(raw_curvature.abs()+solve_shift*inverse_energy)
            tolerance = 128*torch.finfo(vector.dtype).eps*scale
            if (not math.isfinite(scale) or not bool(torch.isfinite(materialized).all())
                or abs(float(measured_curvature-stable_curvature)) > tolerance):
                raise FloatingPointError('stable and materialized shifted curvature disagree; do not silently change shift')
            if abs(float(stable_curvature)) <= tolerance:
                raise FloatingPointError('ambiguous near-zero shifted curvature; not reliable negative-curvature evidence')
            if bool(stable_curvature < 0.):
                raise NegativeShiftedCurvature(generalized_rayleigh=float(raw_curvature/inverse_energy), shift=solve_shift)
            return materialized, stable_curvature

        try:
            direction, linear = guarded_shifted_pcg(right=-gradient, operator=operator,
                precondition=frozen.apply_and_energy, stopping_energy=lambda value: metric_energy(metric, value),
                relative_tolerance=config.relative_tolerance, maximum_iterations=config.maximum_iterations, progress=progress)
            entry['linear_solve'] = linear
            if not linear['converged']:
                entry['status'] = 'true_shifted_residual_not_converged'
            else:
                image = raw_product(direction)
                true_shifted = gradient+image+solve_shift*inverse(direction)
                predicted = gradient+image
                predicted_norms = metric.diagnostics(predicted)
                true_norm = full_metric_norm(metric, true_shifted)
                slope = float(torch.dot(gradient, direction))
                quality = (true_norm <= config.relative_tolerance*initial_metric_norm
                    and math.isfinite(slope) and slope < 0.
                    and predicted_norms['full_gradient_norm'] <= config.predicted_gradient_ratio_limit*initial_norms['full_gradient_norm']
                    and predicted_norms['affine_metric_gradient_norm'] <= config.predicted_gradient_ratio_limit*initial_norms['affine_metric_gradient_norm']
                    and full_metric_norm(metric, predicted) <= config.predicted_gradient_ratio_limit*initial_metric_norm)
                entry.update(status='quality_checked_proposal' if quality else 'unshifted_predicted_gradient_or_true_residual_not_acceptable',
                    actual_true_shifted_metric_residual_norm=true_norm, actual_unshifted_predicted_norms=predicted_norms,
                    actual_unshifted_predicted_full_metric_norm=full_metric_norm(metric, predicted), slope=slope)
                if quality:
                    selected, selected_report = direction, entry
        except NegativeShiftedCurvature as error:
            entry.update(status='negative_curvature_restart', generalized_original_metric_rayleigh=error.generalized_rayleigh)
            shift = max(config.shift_multiplier*solve_shift, -2.*error.generalized_rayleigh)
        except HessianWorkLimit:
            entry['status'] = 'shared_hessian_work_limit_not_converged'
        entry['full_curvature_passes'] = exact.full_curvature_passes-before
        trials.append(entry)
        if selected is not None or entry['status'] == 'shared_hessian_work_limit_not_converged':
            break
        if entry['status'] != 'negative_curvature_restart':
            shift = config.shift_multiplier*solve_shift
        if not math.isfinite(shift):
            raise FloatingPointError('nonfinite prospective shift; no proposal selected')
    check_fixed()
    if optimizer_state_sha256(optimizer) != frozen.source_sha256:
        raise RuntimeError('shifted exact-H proposal changed original raw optimizer history')
    verify_actual_metric(metric, policy)
    return selected, {'phase': 'guarded-frozen-history-shifted-exact-H-gate-end-v2', 'trials': trials,
        'proposal_selected': selected is not None, 'selected': selected_report,
        'full_curvature_passes': exact.full_curvature_passes, 'initial_norms': initial_norms,
        'point_sha256': point_sha256, 'source_optimizer_sha256': frozen.source_sha256,
        'optimizer_updates': 0, 'original_objective_unchanged': True,
        'original_thresholds_unchanged': True, 'outer_gradient_convergence_certified': False,
        'SPD_spectrum_certified': False, 'damping_is_objective_penalty': False}
