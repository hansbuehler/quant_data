"""Exact-H original-M proposals with separately identified layerwise bases."""

from __future__ import annotations

import json
import math
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
from quantdata.gromm2.gradient_reliable_ggn import full_metric_norm, verify_actual_metric
from quantdata.gromm2.kronecker_audit_preconditioner import FrozenKroneckerLBFGS
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS
from quantdata.gromm2.metric_gmres_audit import metric_right_gmres
from quantdata.gromm2.metric_gmres_audit_proposal import TransposeAffineMetric
from quantdata.gromm2.minimum_residual_audit_proposal import MinimumResidualConfig
from quantdata.gromm2.shifted_exact_audit_proposal import HessianWorkLimit


@torch.no_grad()
def propose_kronecker_gmres_step(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                                 optimizer: MetricLBFGS, preconditioner: FrozenKroneckerLBFGS,
                                 gradient: torch.Tensor, diagnostic: str, point_sha256: str,
                                 config: MinimumResidualConfig = MinimumResidualConfig(),
                                 progress: Progress | None = None) -> tuple[torch.Tensor | None, dict]:
    """Change only the right solve base; preserve original H/M and all gates.

    Factor collection/factorization costs occur before this helper and must be
    counted separately by its caller. Every exact-H product here, including
    actual residual confirmation, shares the unchanged declared work limit.
    This proposal never updates native weights/history or certifies an outer
    stationary point. Rejected proposals cannot become zero-policy artifacts.
    """
    metric = optimizer.metric
    verify_actual_metric(metric, policy)
    if (policy.affine_only or gradient.shape != (metric.size,) or gradient.dtype != torch.float64
        or gradient.device != metric.factors.device or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.ne(0.).any()) or action_point_sha256(policy) != point_sha256
        or preconditioner.policy is not policy or preconditioner.point_sha256 != point_sha256
        or preconditioner.source_sha256 != optimizer_state_sha256(optimizer)
        or preconditioner.frozen.metric.identity != metric.identity):
        raise ValueError('Kronecker proposal requires the same original full-policy point, gradient, metric and raw history')
    preconditioner.check_fixed()
    base_identity = json.dumps(preconditioner.identity, sort_keys=True, allow_nan=False)
    coordinates = TransposeAffineMetric(metric)
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
            or action_point_sha256(policy) != point_sha256
            or json.dumps(preconditioner.identity, sort_keys=True, allow_nan=False) != base_identity):
            raise ValueError('Kronecker proposal native point, base identity or exact-H snapshot changed')
        preconditioner.check_fixed()
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
    selected, linear = None, None
    predicted_norms, predicted_metric_norm, slope = None, None, None
    status = 'actual_residual_or_predicted_gradient_not_acceptable'
    try:
        direction, residual, linear = metric_right_gmres(right=-gradient,
            operator=raw_product, precondition=preconditioner.apply_and_energy,
            metric_embedding=coordinates, relative_tolerance=config.relative_tolerance,
            maximum_iterations=config.maximum_iterations, orthogonality_tolerance=config.orthogonality_tolerance,
            progress=progress)
        predicted = -residual
        predicted_norms = metric.diagnostics(predicted)
        predicted_metric_norm = full_metric_norm(metric, predicted)
        slope = float(torch.dot(gradient, direction))
        quality = (linear['converged'] and math.isfinite(slope) and slope < 0.
            and bool(torch.isfinite(direction).all()) and bool(direction.ne(0.).any())
            and predicted_norms['full_gradient_norm'] <= config.predicted_gradient_ratio_limit*initial_norms['full_gradient_norm']
            and predicted_norms['affine_metric_gradient_norm'] <= config.predicted_gradient_ratio_limit*initial_norms['affine_metric_gradient_norm']
            and predicted_metric_norm <= config.predicted_gradient_ratio_limit*initial_metric_norm)
        if quality:
            selected, status = direction, 'quality_checked_unshifted_proposal'
    except HessianWorkLimit:
        status = 'shared_hessian_work_limit_not_converged'
    check_fixed()
    if optimizer_state_sha256(optimizer) != preconditioner.source_sha256:
        raise RuntimeError('Kronecker proposal changed original raw optimizer history')
    verify_actual_metric(metric, policy)
    return selected, {'phase': 'frozen-Kronecker-base-raw-history-original-M-GMRES-exact-H',
        'status': status, 'linear_solve': linear, 'proposal_selected': selected is not None,
        'actual_unshifted_predicted_norms': predicted_norms,
        'actual_unshifted_predicted_full_metric_norm': predicted_metric_norm, 'slope': slope,
        'full_curvature_passes': exact.full_curvature_passes, 'initial_norms': initial_norms,
        'point_sha256': point_sha256, 'source_optimizer_sha256': preconditioner.source_sha256,
        'right_preconditioner_identity': deepcopy(preconditioner.identity),
        'factor_collection_work_requires_separate_accounting': True,
        'optimizer_updates': 0, 'original_objective_unchanged': True,
        'original_thresholds_unchanged': True, 'outer_gradient_convergence_certified': False,
        'original_stopping_metric_unchanged': True, 'SPD_Hessian_assumed': False, 'numerical_shift': 0.,
        'orthogonality_failure_is_numerical_inconclusiveness': True}
