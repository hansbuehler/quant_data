"""Exact empirical neural curvature diagnostics, not an SPD optimization step."""

from __future__ import annotations

import math

import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction, local_objective
from quantdata.gromm2.ggn_audit_continuation import _forward_geometry
from quantdata.gromm2.ggn_audit_optimization import FullActionGGN, InverseAffineMetric, _flatten
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric


class FullActionHessian(FullActionGGN):
    """Stream the exact full original loss Hessian at an immutable snapshot.

    This includes the neural second-derivative term omitted by GGN and can be
    indefinite. It must NOT be passed to the ordinary positive-definite CG
    optimizer. Products are diagnostics only until an independently reviewed
    indefinite-curvature-safe step and canonical acceptance are available.
    """

    def __init__(self, policy: AffineDeepAction, problem: AffineAuditProblem, *,
                 diagnostic: str, compile: bool = False, progress: Progress | None = None) -> None:
        def notify(entry: dict) -> None:
            if progress is not None:
                progress({**entry, 'stage': 'full_hessian_curvature_pass', 'exact_empirical_hessian': True})

        super().__init__(policy, problem, diagnostic=diagnostic, compile=compile, progress=notify)
        self.data_objects = (problem.states, problem.returns, problem.rows, problem.eligible)
        self.data_versions = tuple(value._version for value in self.data_objects)
        self.data_geometry = [(tuple(value.shape), value.dtype, value.device) for value in self.data_objects]
        self.objective = (problem.action_scale, problem.squared_lambda, problem.config.batch_size)

    def _block_product(self, features: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor,
                       direction: dict[str, torch.Tensor]) -> torch.Tensor:
        def scalar_loss(values: dict[str, torch.Tensor]) -> torch.Tensor:
            action = torch.func.functional_call(self.functional_policy,
                (values, self.buffers), (features, mask), strict=True)
            loss = local_objective(action, returns, diagnostic=self.diagnostic,
                action_scale=self.problem.action_scale, squared_lambda=self.problem.squared_lambda)
            return loss*(len(features)/len(self.problem.rows))

        _, image = torch.func.jvp(torch.func.grad(scalar_loss), (self.parameters,), (direction,))
        return _flatten(image).detach()

    def product(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply one complete Hessian product without changing native state."""
        def check() -> None:
            if (any(actual is not expected for actual, expected in zip(self.data_objects,
                (self.problem.states, self.problem.returns, self.problem.rows, self.problem.eligible), strict=True))
                or self.data_versions != tuple(value._version for value in self.data_objects)
                or self.data_geometry != [(tuple(value.shape), value.dtype, value.device) for value in self.data_objects]
                or self.objective != (self.problem.action_scale, self.problem.squared_lambda, self.problem.config.batch_size)):
                raise ValueError('exact-Hessian diagnostic data or objective changed')

        check()
        result = super().product(vector)
        check()
        return result


def _check_common_snapshot(exact: FullActionHessian, ggn: FullActionGGN,
                           metric: AffineBlockMetric) -> None:
    if (exact.policy is not ggn.policy or exact.problem is not ggn.problem
        or exact.diagnostic != ggn.diagnostic or exact.size != metric.size
        or type(ggn) is not FullActionGGN
        or exact.functional_policy.config != ggn.functional_policy.config
        or exact.functional_policy.affine_only != ggn.functional_policy.affine_only
        or _forward_geometry(exact.functional_policy) != _forward_geometry(ggn.functional_policy)
        or any(actual is not expected for actual, expected in zip(metric.parameters, exact.policy.parameters(), strict=True))
        or exact.parameters.keys() != ggn.parameters.keys()
        or exact.buffers.keys() != ggn.buffers.keys()
        or any(not torch.equal(value, ggn.parameters[name]) for name, value in exact.parameters.items())
        or any(not torch.equal(value, ggn.buffers[name]) for name, value in exact.buffers.items())):
        raise ValueError('curvature diagnostics require identical snapshots, objective, data and original metric')


def compare_full_curvature(*, exact: FullActionHessian, ggn: FullActionGGN,
                           direction: torch.Tensor, metric: AffineBlockMetric) -> dict:
    """Measure the complete omitted curvature vector, not just its projection.

    Both products include every original training row. This changes neither
    the action, loss, samples, fitted law nor any original stopping threshold.
    """
    _check_common_snapshot(exact, ggn, metric)
    actual = exact.product(direction)
    approximation = ggn.product(direction)
    omitted = actual-approximation
    return {'exact_hessian': metric.diagnostics(actual), 'ggn': metric.diagnostics(approximation),
        'omitted_neural_curvature': metric.diagnostics(omitted),
        'exact_directional_energy': float(torch.dot(direction, actual)),
        'ggn_directional_energy': float(torch.dot(direction, approximation)),
        'omitted_directional_energy': float(torch.dot(direction, omitted)),
        'full_training_observations': len(exact.problem.rows),
        'additional_exact_hessian_passes': 1, 'additional_ggn_passes': 1,
        'optimization_performed': False, 'outer_convergence_certified': False}


def decompose_gradient_change(*, initial_gradient: torch.Tensor, final_gradient: torch.Tensor,
                              direction: torch.Tensor, hessian_image: torch.Tensor,
                              ggn_image: torch.Tensor, damping: float,
                              metric: AffineBlockMetric) -> dict:
    """Account for the actual accepted gradient without certifying convergence.

    The caller supplies complete canonical gradients and products evaluated
    at the initial point in the actual rounded accepted increment. The final
    gradient equals the actual damped linear residual, minus damping, plus
    omitted neural curvature, plus the nonlinear gradient remainder. Closure
    checks arithmetic only: the remainder is defined by this identity.
    """
    vectors = (initial_gradient, final_gradient, direction, hessian_image, ggn_image)
    if (not math.isfinite(damping) or damping <= 0.
        or any(value.shape != (metric.size,) or value.dtype != metric.factors.dtype
            or value.device != metric.factors.device for value in vectors)):
        raise ValueError('gradient decomposition requires the original vector geometry and positive damping')
    if not all(bool(torch.isfinite(value).all()) for value in vectors):
        raise FloatingPointError('nonfinite full gradient decomposition')
    if float(torch.linalg.vector_norm(direction)) == 0.:
        raise ValueError('gradient decomposition requires a nonzero actual accepted increment')
    damping_image = damping*InverseAffineMetric(metric)(direction)
    components = {
        'actual_damped_linear_residual': initial_gradient+ggn_image+damping_image,
        'negative_damping': -damping_image,
        'omitted_neural_curvature': hessian_image-ggn_image,
        'nonlinear_gradient_remainder': final_gradient-initial_gradient-hessian_image,
    }
    all_vectors = {'initial_gradient': initial_gradient, 'final_gradient': final_gradient,
        'exact_hessian_image': hessian_image, 'ggn_image': ggn_image, **components}
    metrics = {}
    metric_coordinates = {}
    for name, value in all_vectors.items():
        norms = metric.diagnostics(value)
        norms['full_metric_gradient_norm'] = math.hypot(norms['affine_metric_gradient_norm'],
            math.sqrt(metric.deep_scale)*norms['deep_gradient_norm'])
        norms['actual_increment_projection'] = float(torch.dot(direction, value))
        norms['final_gradient_projection'] = float(torch.dot(final_gradient, value))
        metrics[name] = norms
        affine = metric.affine_values(value)
        transformed = torch.empty_like(affine)
        for factor, columns in zip(metric.factors, metric.columns, strict=True):
            transformed[:, columns] = factor.T@affine[:, columns]
        coordinates = math.sqrt(metric.deep_scale)*value
        coordinates[metric.weight[0]] = transformed[:-1].T.reshape(-1)
        coordinates[metric.bias[0]] = transformed[-1]
        metric_coordinates[name] = coordinates
    reconstruction = sum(components.values())
    error = reconstruction-final_gradient
    component_norms = [metrics[name] for name in components]
    cancellation = {}
    for field in ('full_gradient_norm', 'affine_metric_gradient_norm', 'full_metric_gradient_norm'):
        denominator = sum(value[field] for value in component_norms)
        cancellation[field] = None if denominator == 0. else metrics['final_gradient'][field]/denominator
    names = list(all_vectors)
    gram = torch.stack([metric_coordinates[name] for name in names])
    gram = gram@gram.T
    metric_gram = gram.cpu().tolist()
    cosines = [[None if metrics[left]['full_metric_gradient_norm'] == 0.
        or metrics[right]['full_metric_gradient_norm'] == 0. else metric_gram[left_index][right_index]
        /metrics[left]['full_metric_gradient_norm']/metrics[right]['full_metric_gradient_norm']
        for right_index, right in enumerate(names)] for left_index, left in enumerate(names)]
    affine_direction = metric.affine_values(direction)
    inverse_affine = torch.empty_like(affine_direction)
    inverse_energy = direction.clone()
    inverse_energy[metric.weight[0]], inverse_energy[metric.bias[0]] = 0., 0.
    inverse_energy = float(inverse_energy.square().sum())/metric.deep_scale
    for factor, columns in zip(metric.factors, metric.columns, strict=True):
        solved = torch.linalg.solve(factor, affine_direction[:, columns])
        inverse_energy += float(solved.square().sum())
        inverse_affine[:, columns] = torch.linalg.solve(factor.T, solved)
    stable_inverse = direction/metric.deep_scale
    stable_inverse[metric.weight[0]] = inverse_affine[:-1].T.reshape(-1)
    stable_inverse[metric.bias[0]] = inverse_affine[-1]
    actual_inverse = damping_image/damping
    inverse_error = metric.diagnostics(actual_inverse-stable_inverse)
    identity_error = metric.diagnostics(error)
    return {'vectors': metrics, 'cancellation_ratios': cancellation,
        'arithmetic_identity_error': identity_error,
        'arithmetic_identity_relative_raw_error': None if metrics['final_gradient']['full_gradient_norm'] == 0.
            else identity_error['full_gradient_norm']/metrics['final_gradient']['full_gradient_norm'],
        'full_metric_pairwise': {'names': names, 'gram': metric_gram, 'cosines': cosines},
        'inverse_metric_crosscheck': {'production_minus_factor_solve': inverse_error,
            'production_roundtrip_error': metric.diagnostics(metric(actual_inverse)-direction),
            'production_step_energy': float(torch.dot(direction, actual_inverse)),
            'factor_solve_step_energy': inverse_energy},
        'actual_increment_raw_norm': float(torch.linalg.vector_norm(direction)),
        'damping': damping, 'optimization_performed': False, 'outer_convergence_certified': False,
        'identity_is_arithmetic_not_independent_validation': True}


def diagnose_accepted_gradient_change(*, exact: FullActionHessian, ggn: FullActionGGN,
                                     initial_gradient: torch.Tensor, final_gradient: torch.Tensor,
                                     direction: torch.Tensor, damping: float,
                                     metric: AffineBlockMetric) -> dict:
    """Run two full read-only curvature products for an accepted-point audit."""
    _check_common_snapshot(exact, ggn, metric)
    report = decompose_gradient_change(initial_gradient=initial_gradient, final_gradient=final_gradient,
        direction=direction, hessian_image=exact.product(direction), ggn_image=ggn.product(direction),
        damping=damping, metric=metric)
    return {**report, 'full_training_observations': len(exact.problem.rows),
        'additional_exact_hessian_passes': 1, 'additional_ggn_passes': 1}
