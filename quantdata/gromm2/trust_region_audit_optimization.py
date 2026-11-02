"""Indefinite-safe exact-curvature proposals for the original audit loss."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.full_hessian_audit import FullActionHessian
from quantdata.gromm2.ggn_audit_continuation import (
    _forward_geometry,
    action_point_sha256,
    optimizer_state_sha256,
)
from quantdata.gromm2.gradient_reliable_ggn import verify_actual_metric
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class ExactTrustRegionConfig:
    """Inner numerical controls, not outer original stationarity thresholds."""

    relative_tolerance: float = .1
    maximum_iterations: int = 120
    coordinate_relative_tolerance: float = 1.e-9

    def __post_init__(self) -> None:
        verify_inp(math.isfinite(self.relative_tolerance) and 0. < self.relative_tolerance < 1.,
            lambda: 'exact trust-region residual tolerance must lie strictly between zero and one')
        verify_inp(type(self.maximum_iterations) is int and self.maximum_iterations > 0,
            lambda: 'exact trust-region iteration allowance must be a positive integer')
        verify_inp(math.isfinite(self.coordinate_relative_tolerance) and 0. < self.coordinate_relative_tolerance < 1.e-3,
            lambda: 'exact trust-region coordinate verification tolerance must be positive and small')


class OriginalMetricCoordinates:
    """Use s=Cz, M=CC', without forming a cancelling inverse-metric norm.

    The affine factor is the original saved Gram inverse factor; the deep
    factor is sqrt(deep_scale) times identity. No coefficient is discarded.
    """

    def __init__(self, metric: AffineBlockMetric) -> None:
        self.metric = metric
        self.objects = (metric.factors, *metric.columns)
        self.versions = tuple(value._version for value in self.objects)
        self.controls = (metric.deep_scale, metric.weight, metric.bias, metric.size)
        self.inverse_factors = torch.linalg.inv(metric.factors)
        if not bool(torch.isfinite(self.inverse_factors).all()):
            raise FloatingPointError('nonfinite original optimizer-metric inverse factors')

    def check_fixed(self) -> None:
        current = (self.metric.factors, *self.metric.columns)
        if (len(current) != len(self.objects)
            or any(actual is not saved for actual, saved in zip(current, self.objects, strict=True))
            or tuple(value._version for value in current) != self.versions
            or (self.metric.deep_scale, self.metric.weight, self.metric.bias, self.metric.size) != self.controls):
            raise ValueError('the original metric coordinate system changed during a proposal')

    @torch.no_grad()
    def apply(self, vector: torch.Tensor, *, operation: str) -> torch.Tensor:
        """Apply C, C' or C^-1 to an original-layout float64 vector."""
        self.check_fixed()
        metric = self.metric
        if (vector.shape != (metric.size,) or vector.dtype != torch.float64
            or vector.dtype != metric.factors.dtype or vector.device != metric.factors.device
            or not bool(torch.isfinite(vector).all()) or operation not in ('forward', 'transpose', 'inverse')):
            raise ValueError('original metric coordinate vector geometry or operation changed')
        affine = metric.affine_values(vector)
        transformed = torch.empty_like(affine)
        for index, (factor, columns) in enumerate(zip(metric.factors, metric.columns, strict=True)):
            operator = factor if operation == 'forward' else factor.T if operation == 'transpose' else self.inverse_factors[index]
            transformed[:, columns] = operator@affine[:, columns]
        scale = math.sqrt(metric.deep_scale)
        result = vector/scale if operation == 'inverse' else scale*vector
        result[metric.weight[0]] = transformed[:-1].T.reshape(-1)
        result[metric.bias[0]] = transformed[-1]
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('nonfinite original metric coordinate transform')
        return result


def _boundary_roots(point: torch.Tensor, direction: torch.Tensor, radius: float) -> tuple[float, float]:
    square = float(torch.dot(direction, direction))
    cross = float(torch.dot(point, direction))
    constant = float(torch.dot(point, point))-radius*radius
    tolerance = 128*torch.finfo(point.dtype).eps*radius*radius
    if square <= 0. or constant > tolerance:
        raise FloatingPointError('invalid trust-region boundary intersection')
    constant = min(constant, 0.)
    radical = math.sqrt(cross*cross-square*constant)
    positive = -constant/(cross+radical) if cross >= 0. and cross+radical > 0. else (-cross+radical)/square
    negative = -constant/(cross-radical) if cross < 0. and cross-radical < 0. else (-cross-radical)/square
    return negative, positive


@torch.no_grad()
def steihaug_step(*, gradient: torch.Tensor, hessian_product: Callable[[torch.Tensor], torch.Tensor],
                  radius: float, config: ExactTrustRegionConfig = ExactTrustRegionConfig(),
                  progress: Progress | None = None) -> tuple[torch.Tensor, dict]:
    """Truncate CG safely for an indefinite symmetric quadratic on a sphere.

    Inputs are ALREADY in original-M factor coordinates. Negative curvature
    and boundary/cap termination are not residual convergence. A complete
    true Hessian product verifies the final model reduction and residual.
    """
    if (gradient.ndim != 1 or gradient.dtype != torch.float64 or gradient.numel() == 0
        or not bool(torch.isfinite(gradient).all()) or not bool(gradient.ne(0.).any())
        or not math.isfinite(radius) or radius <= 0. or not math.isfinite(radius*radius) or radius*radius == 0.):
        raise ValueError('Steihaug requires a finite nonzero float64 gradient and positive finite squared radius')
    original_gradient = gradient.clone()
    products = 0

    def product(vector: torch.Tensor) -> torch.Tensor:
        nonlocal products
        image = hessian_product(vector)
        products += 1
        if (image.shape != gradient.shape or image.dtype != gradient.dtype or image.device != gradient.device
            or not bool(torch.isfinite(image).all())):
            raise FloatingPointError('nonfinite or incompatible complete exact-Hessian product')
        return image

    point = torch.zeros_like(gradient)
    residual = -gradient.clone()
    direction = residual.clone()
    energy = torch.dot(residual, residual)
    initial_norm = float(torch.linalg.vector_norm(residual))
    if not math.isfinite(initial_norm) or initial_norm == 0.:
        raise FloatingPointError('unrepresentable original trust-region gradient norm')
    threshold = config.relative_tolerance*initial_norm
    stop_reason, iterations = 'iteration_cap_not_residual_convergence', 0
    negative_curvature = False
    first_image = None
    for iterations in range(1, config.maximum_iterations+1):
        image = product(direction)
        if first_image is None:
            first_image = image.clone()
        curvature = float(torch.dot(direction, image))
        if not math.isfinite(curvature):
            raise FloatingPointError('nonfinite exact directional curvature')
        if curvature <= 0.:
            negative_curvature = True
            roots = _boundary_roots(point, direction, radius)
            candidates = []
            image_at_point = -gradient-residual
            for root in roots:
                candidate = point+root*direction
                candidate_image = image_at_point+root*image
                slope = float(torch.dot(gradient, candidate))
                model = slope+.5*float(torch.dot(candidate, candidate_image))
                if math.isfinite(model) and math.isfinite(slope) and slope < 0.:
                    candidates.append((model, candidate))
            if not candidates:
                raise FloatingPointError('neither negative-curvature boundary candidate is finite descent')
            point = min(candidates, key=lambda value: value[0])[1]
            stop_reason = 'negative_curvature_boundary'
            break
        rate = float(energy)/curvature
        candidate = point+rate*direction
        if float(torch.linalg.vector_norm(candidate)) >= radius:
            point = point+_boundary_roots(point, direction, radius)[1]*direction
            stop_reason = 'positive_curvature_boundary'
            break
        point = candidate
        residual.sub_(rate*image)
        current_norm = float(torch.linalg.vector_norm(residual))
        restarted = False
        if current_norm <= threshold:
            residual = -gradient-product(point)
            current_norm = float(torch.linalg.vector_norm(residual))
            if current_norm <= threshold:
                stop_reason = 'verified_true_residual'
                break
            restarted = True
        next_energy = torch.dot(residual, residual)
        direction = residual.clone() if restarted else residual+(next_energy/energy)*direction
        energy = next_energy
        if progress is not None:
            progress({'stage': 'exact_trust_region_iteration', 'iteration': iterations,
                'maximum_iterations': config.maximum_iterations, 'full_exact_hessian_products': products,
                'residual_norm': current_norm, 'residual_threshold': threshold, 'true_residual_restart': restarted})
    final_image = product(point)
    slope = float(torch.dot(gradient, point))
    predicted = -slope-.5*float(torch.dot(point, final_image))
    cauchy_rate = min(radius/initial_norm,
        initial_norm*initial_norm/float(torch.dot(-gradient, first_image))) if float(torch.dot(-gradient, first_image)) > 0. else radius/initial_norm
    cauchy = -cauchy_rate*gradient
    cauchy_image = cauchy_rate*first_image
    cauchy_reduction = -float(torch.dot(gradient, cauchy))-.5*float(torch.dot(cauchy, cauchy_image))
    fallback = not math.isfinite(predicted) or slope >= 0. or predicted < cauchy_reduction
    if fallback:
        point = cauchy
        final_image = product(point)
        slope = float(torch.dot(gradient, point))
        predicted = -slope-.5*float(torch.dot(point, final_image))
        stop_reason = 'cauchy_model_reduction_fallback'
    reduction_roundoff = 128*torch.finfo(gradient.dtype).eps*max(abs(predicted), abs(cauchy_reduction), torch.finfo(gradient.dtype).tiny)
    point_norm = float(torch.linalg.vector_norm(point))
    residual_norm = float(torch.linalg.vector_norm(gradient+final_image))
    if (not torch.equal(gradient, original_gradient) or not bool(torch.isfinite(point).all())
        or not math.isfinite(predicted) or predicted <= 0. or not slope < 0.
        or predicted < cauchy_reduction-reduction_roundoff
        or point_norm > radius*(1.+128*torch.finfo(gradient.dtype).eps)):
        raise FloatingPointError('exact trust-region proposal failed source, boundary or model-descent verification')
    return point, {'iterations': iterations, 'full_exact_hessian_products': products,
        'stop_reason': stop_reason, 'negative_curvature_encountered': negative_curvature,
        'cauchy_fallback': fallback, 'predicted_reduction': predicted,
        'cauchy_predicted_reduction': cauchy_reduction, 'slope': slope,
        'radius': radius, 'step_norm': point_norm, 'initial_residual_norm': initial_norm,
        'verified_residual_norm': residual_norm, 'residual_threshold': threshold,
        'inner_residual_converged': residual_norm <= threshold,
        'outer_gradient_convergence_certified': False, 'optimizer_updates': 0}


@torch.no_grad()
def propose_exact_trust_region_step(*, policy: AffineDeepAction, optimizer: MetricLBFGS,
                                   exact: FullActionHessian, gradient: torch.Tensor,
                                   radius: float, point_sha256: str,
                                   config: ExactTrustRegionConfig = ExactTrustRegionConfig(),
                                   progress: Progress | None = None) -> tuple[torch.Tensor, dict]:
    """Read-only exact-H proposal; canonical acceptance is still mandatory."""
    metric = optimizer.metric
    verify_actual_metric(metric, policy)
    if type(exact) is not FullActionHessian:
        raise ValueError('exact trust-region proposal requires a fixed Hessian snapshot')
    snapshot_parameters = tuple(exact.parameters.items())
    snapshot_buffers = tuple(exact.buffers.items())
    snapshot_versions = tuple(value._version for name, value in (*snapshot_parameters, *snapshot_buffers))
    native_objects = tuple(policy.named_parameters())+tuple(policy.named_buffers())
    native_versions = tuple(value._version for name, value in native_objects)
    source_config = deepcopy(policy.config)
    source_geometry = _forward_geometry(policy)

    def check_snapshot() -> None:
        current_snapshot = (*exact.parameters.items(), *exact.buffers.items())
        current_native = (*policy.named_parameters(), *policy.named_buffers())
        if (len(current_snapshot) != len(snapshot_versions) or len(current_native) != len(native_objects)
            or any(name != saved_name or value is not saved_value for (name, value), (saved_name, saved_value)
                in zip(current_snapshot, (*snapshot_parameters, *snapshot_buffers), strict=True))
            or tuple(value._version for name, value in current_snapshot) != snapshot_versions
            or any(name != saved_name or value is not saved_value for (name, value), (saved_name, saved_value)
                in zip(current_native, native_objects, strict=True))
            or tuple(value._version for name, value in current_native) != native_versions
            or exact.policy is not policy or policy.affine_only
            or exact.functional_policy.affine_only or policy.config != exact.functional_policy.config
            or policy.config != source_config
            or _forward_geometry(policy) != source_geometry or _forward_geometry(exact.functional_policy) != source_geometry
            or action_point_sha256(policy) != point_sha256
            or any(actual is not expected for actual, expected in zip(metric.parameters, policy.parameters(), strict=True))
            or any(not torch.equal(value.detach(), exact.parameters[name]) for name, value in policy.named_parameters())
            or any(not torch.equal(value.detach(), exact.buffers[name]) for name, value in policy.named_buffers())):
            raise ValueError('exact trust-region proposal requires the original complete policy and fixed Hessian snapshot')

    check_snapshot()
    original_optimizer = optimizer_state_sha256(optimizer)
    coordinates = OriginalMetricCoordinates(metric)
    transformed_gradient = coordinates.apply(gradient, operation='transpose')
    before = exact.full_curvature_passes
    last_raw_argument, last_raw_image = None, None

    def product(vector: torch.Tensor) -> torch.Tensor:
        nonlocal last_raw_argument, last_raw_image
        check_snapshot()
        coordinates.check_fixed()
        last_raw_argument = coordinates.apply(vector, operation='forward')
        last_raw_image = exact.product(last_raw_argument)
        image = coordinates.apply(last_raw_image, operation='transpose')
        coordinates.check_fixed()
        check_snapshot()
        return image

    step, report = steihaug_step(gradient=transformed_gradient, hessian_product=product,
        radius=radius, config=config, progress=progress)
    direction = coordinates.apply(step, operation='forward')
    slope = float(torch.dot(gradient, direction))
    check_snapshot()
    extra_products = 0
    if not torch.equal(last_raw_argument, direction):
        last_raw_image = exact.product(direction)
        extra_products = 1
        check_snapshot()
    mapped = coordinates.apply(direction, operation='inverse')
    mapped_norm = float(torch.linalg.vector_norm(mapped))
    round_trip_relative_error = float(torch.linalg.vector_norm(mapped-step))/float(torch.linalg.vector_norm(step))
    raw_predicted_reduction = -slope-.5*float(torch.dot(direction, last_raw_image))
    actual_residual = gradient+last_raw_image
    actual_metric_residual_norm = float(torch.linalg.vector_norm(coordinates.apply(actual_residual, operation='transpose')))
    actual_inner_converged = actual_metric_residual_norm <= report['residual_threshold']
    if (action_point_sha256(policy) != point_sha256 or optimizer_state_sha256(optimizer) != original_optimizer
        or not math.isfinite(slope) or not slope < 0.
        or not math.isclose(slope, report['slope'], rel_tol=config.coordinate_relative_tolerance, abs_tol=1.e-12)
        or not math.isfinite(round_trip_relative_error) or round_trip_relative_error > config.coordinate_relative_tolerance
        or mapped_norm > radius*(1.+config.coordinate_relative_tolerance)
        or not math.isfinite(raw_predicted_reduction) or raw_predicted_reduction <= 0.
        or not math.isfinite(actual_metric_residual_norm)
        or not math.isclose(raw_predicted_reduction, report['predicted_reduction'], rel_tol=config.coordinate_relative_tolerance, abs_tol=1.e-12)):
        raise RuntimeError('exact trust-region source, mapped boundary, prediction or canonical descent changed')
    verify_actual_metric(metric, policy)
    return direction, {**report, 'phase': 'exact-H-indefinite-safe-original-M-Steihaug',
        'coordinate_inner_residual_converged': report['inner_residual_converged'],
        'inner_residual_converged': actual_inner_converged,
        'coordinate_stop_reason': report['stop_reason'],
        'stop_reason': 'mapped_true_residual_not_converged' if report['stop_reason'] == 'verified_true_residual' and not actual_inner_converged else report['stop_reason'],
        'point_sha256': point_sha256, 'source_optimizer_sha256': original_optimizer,
        'preconditioner': 'original fixed M; raw history retained unchanged and unused',
        'full_curvature_passes': exact.full_curvature_passes-before,
        'mapping_confirmation_extra_products': extra_products,
        'mapped_direction_inverse_metric_norm': mapped_norm,
        'coordinate_round_trip_relative_error': round_trip_relative_error,
        'actual_raw_predicted_reduction': raw_predicted_reduction,
        'actual_raw_residual_norms': metric.diagnostics(actual_residual),
        'actual_original_metric_residual_norm': actual_metric_residual_norm,
        'boundary_geometry': 'norm(C^-1 s), M=C C_transpose',
        'stopping_geometry': 'norm(C_transpose r), original fixed M',
        'original_objective_unchanged': True, 'coefficient_directions_discarded': False,
        'policy_parameters_changed': False}
