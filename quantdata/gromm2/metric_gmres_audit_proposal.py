"""Unshifted exact-H proposals minimizing the original fixed-M residual."""

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
from quantdata.gromm2.gradient_reliable_ggn import full_metric_norm, verify_actual_metric
from quantdata.gromm2.lbfgs_audit_preconditioner import FrozenLBFGSPreconditioner
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.metric_gmres_audit import metric_right_gmres
from quantdata.gromm2.minimum_residual_audit_proposal import MinimumResidualConfig
from quantdata.gromm2.shifted_exact_audit_proposal import HessianWorkLimit


class TransposeAffineMetric:
    """Apply the original C' factor with guarded fixed geometry and no inverse."""

    def __init__(self, metric: AffineBlockMetric) -> None:
        self.metric = metric
        self.objects = (metric.factors, *metric.columns)
        self.versions = tuple(value._version for value in self.objects)
        self.controls = (metric.deep_scale, deepcopy(metric.slices), metric.weight, metric.bias, metric.size)

    def check_fixed(self) -> None:
        actual = (self.metric.factors, *self.metric.columns)
        if (len(actual) != len(self.objects)
            or any(value is not saved for value, saved in zip(actual, self.objects, strict=True))
            or tuple(value._version for value in actual) != self.versions
            or (self.metric.deep_scale, self.metric.slices, self.metric.weight, self.metric.bias, self.metric.size) != self.controls):
            raise ValueError('original-M transpose factor geometry changed')

    @torch.no_grad()
    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        self.check_fixed()
        metric = self.metric
        if (vector.shape != (metric.size,) or vector.dtype != torch.float64
            or vector.dtype != metric.factors.dtype or vector.device != metric.factors.device
            or not bool(torch.isfinite(vector).all())):
            raise ValueError('original-M transpose vector geometry or precision changed')
        affine = metric.affine_values(vector)
        transformed = torch.empty_like(affine)
        for factor, columns in zip(metric.factors, metric.columns, strict=True):
            transformed[:, columns] = factor.T@affine[:, columns]
        result = math.sqrt(metric.deep_scale)*vector
        result[metric.weight[0]] = transformed[:-1].T.reshape(-1)
        result[metric.bias[0]] = transformed[-1]
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('nonfinite original-M transpose factor vector')
        return result


@torch.no_grad()
def propose_metric_gmres_step(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                              optimizer: MetricLBFGS, gradient: torch.Tensor, diagnostic: str,
                              point_sha256: str, config: MinimumResidualConfig = MinimumResidualConfig(),
                              progress: Progress | None = None) -> tuple[torch.Tensor | None, dict]:
    """Propose an unshifted Newton direction without assuming an SPD Hessian.

    Frozen positive raw-history P right-preconditions GMRES; original fixed M
    defines BOTH minimized residual geometry and true-residual stopping. Require actual predicted
    raw/full-M/affine-M gradients no worse than their source values and strict
    canonical descent. A numerical refusal never selects a zero policy.
    Policy/history/RNG/objective/original gates and probability law are unchanged.
    """
    metric = optimizer.metric
    verify_actual_metric(metric, policy)
    if (policy.affine_only or gradient.shape != (metric.size,) or gradient.dtype != torch.float64
        or gradient.device != metric.factors.device or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.ne(0.).any()) or action_point_sha256(policy) != point_sha256):
        raise ValueError('minimum-residual proposal requires the actual original full-policy gradient and point')
    frozen = FrozenLBFGSPreconditioner(optimizer, policy)
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
            or action_point_sha256(policy) != point_sha256):
            raise ValueError('minimum-residual native point or frozen Hessian snapshot changed')
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
    selected, linear = None, None
    predicted_norms, predicted_metric_norm, slope = None, None, None
    status = 'actual_residual_or_predicted_gradient_not_acceptable'
    try:
        direction, residual, linear = metric_right_gmres(right=-gradient,
            operator=raw_product, precondition=frozen.apply_and_energy,
            metric_embedding=coordinates,
            relative_tolerance=config.relative_tolerance, maximum_iterations=config.maximum_iterations,
            orthogonality_tolerance=config.orthogonality_tolerance, progress=progress)
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
    if optimizer_state_sha256(optimizer) != frozen.source_sha256:
        raise RuntimeError('minimum-residual proposal changed original raw optimizer history')
    verify_actual_metric(metric, policy)
    return selected, {'phase': 'frozen-history-P-right-preconditioned-original-M-GMRES-exact-H',
        'status': status, 'linear_solve': linear, 'proposal_selected': selected is not None,
        'actual_unshifted_predicted_norms': predicted_norms,
        'actual_unshifted_predicted_full_metric_norm': predicted_metric_norm, 'slope': slope,
        'full_curvature_passes': exact.full_curvature_passes, 'initial_norms': initial_norms,
        'point_sha256': point_sha256, 'source_optimizer_sha256': frozen.source_sha256,
        'optimizer_updates': 0, 'original_objective_unchanged': True,
        'original_thresholds_unchanged': True, 'outer_gradient_convergence_certified': False,
        'SPD_Hessian_assumed': False, 'numerical_shift': 0.,
        'orthogonality_failure_is_numerical_inconclusiveness': True}
