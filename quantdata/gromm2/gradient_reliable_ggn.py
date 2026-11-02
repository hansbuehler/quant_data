"""Canonical line searches with a full-data gradient-model reliability check."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.full_hessian_audit import FullActionHessian
from quantdata.gromm2.ggn_audit_continuation import _forward_geometry
from quantdata.gromm2.ggn_audit_optimization import _flatten, _unpack
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric
from quantdata.gromm2.neural_audit_optimization import full_action_pass


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class GradientReliabilityConfig:
    """Numerical controls only; no replacement for either stationarity gate."""

    initial_rate: float = .25
    minimum_rate: float = 2.**-20
    armijo: float = 1.e-4
    nonlinear_fraction: float = .25
    expansion_fraction: float = .0625

    def __post_init__(self) -> None:
        verify_inp(0. < self.minimum_rate <= self.initial_rate <= 1.,
            lambda: 'gradient reliability needs positive bounded line-search rates')
        verify_inp(0. < self.armijo < .5 and 0. < self.expansion_fraction < self.nonlinear_fraction < 1.,
            lambda: 'gradient reliability needs valid Armijo and nonlinear-error fractions')


def full_metric_norm(metric: AffineBlockMetric, vector: torch.Tensor) -> float:
    """Factor-stable optimizer-coordinate norm, not an outer stopping norm."""
    norms = metric.diagnostics(vector)
    return math.hypot(norms['affine_metric_gradient_norm'], math.sqrt(metric.deep_scale)*norms['deep_gradient_norm'])


def verify_actual_metric(metric: AffineBlockMetric, policy: AffineDeepAction) -> None:
    """Bind actual numerical operators, not just their declared metadata."""
    shapes, slices, offset = {}, {}, 0
    for name, parameter in policy.named_parameters():
        shapes[name] = list(parameter.shape)
        slices[name] = (slice(offset, offset+parameter.numel()), parameter.shape)
        offset += parameter.numel()
    if (not math.isfinite(metric.deep_scale) or metric.deep_scale <= 0.
        or metric.deep_scale != metric.identity['deep_scale']
        or metric.size != offset or metric.slices != slices
        or list(shapes) != list(metric.identity['parameter_shapes']) or shapes != metric.identity['parameter_shapes']
        or metric.weight != slices['affine.I0.weight'] or metric.bias != slices['affine.I0.bias']
        or any(actual is not expected for actual, expected in zip(metric.parameters, policy.parameters(), strict=True))
        or any(columns.dtype != torch.int64 or columns.device != metric.factors.device for columns in metric.columns)
        or [columns.cpu().tolist() for columns in metric.columns] != metric.identity['columns']
        or list(metric.factors.shape) != metric.identity['factor_shape']
        or hashlib.sha256(metric.factors.cpu().contiguous().numpy().tobytes()).hexdigest() != metric.identity['factor_sha256']):
        raise ValueError('actual optimizer metric differs from its declared original identity')


def accept_gradient_reliable_direction(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                                       direction: torch.Tensor, exact: FullActionHessian,
                                       metric: AffineBlockMetric, diagnostic: str,
                                       config: GradientReliabilityConfig = GradientReliabilityConfig(),
                                       forward: Callable | None = None,
                                       progress: Progress | None = None) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Require canonical Armijo descent and a controlled nonlinear remainder.

    The exact Hessian is only a fixed first-order gradient predictor, never
    an SPD-CG operator. Every candidate gradient and product uses all original
    training rows. Work exhaustion or rejection is not convergence. Accepted
    gradients need not contract; neither original raw/affine gate is changed.
    """
    if (type(exact) is not FullActionHessian or exact.policy is not policy or exact.problem is not problem
        or diagnostic not in ('squared', 'entropy') or exact.diagnostic != diagnostic
        or policy.affine_only or policy.config != exact.functional_policy.config
        or _forward_geometry(policy) != _forward_geometry(exact.functional_policy)
        or any(actual is not expected for actual, expected in zip(metric.parameters, policy.parameters(), strict=True))):
        raise ValueError('gradient reliability requires the original complete policy, objective and fixed Hessian snapshot')
    parameters = dict(policy.named_parameters())
    if (parameters.keys() != exact.parameters.keys()
        or any(not torch.equal(value.detach(), exact.parameters[name]) for name, value in parameters.items())
        or any(not torch.equal(value.detach(), exact.buffers[name]) for name, value in policy.named_buffers())):
        raise ValueError('gradient reliability requires the unchanged initial parameter and buffer snapshot')
    verify_actual_metric(metric, policy)
    original = _flatten({name: value.detach() for name, value in parameters.items()}).clone()
    if (direction.shape != original.shape or direction.dtype != original.dtype or direction.device != original.device
        or not bool(torch.isfinite(direction).all())):
        raise ValueError('gradient reliability direction geometry or finiteness changed')
    entry_gradients = {name: None if value.grad is None else value.grad.detach().clone()
        for name, value in parameters.items()}
    objective_passes, initial_curvature_passes = 0, exact.full_curvature_passes
    trials = []

    def assign(values: torch.Tensor) -> None:
        unpacked = _unpack(values, parameters)
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(unpacked[name])

    def rollback() -> None:
        assign(original)
        for name, parameter in parameters.items():
            parameter.grad = None if entry_gradients[name] is None else entry_gradients[name].clone()

    def evaluate() -> tuple[float, torch.Tensor]:
        nonlocal objective_passes
        policy.zero_grad(set_to_none=True)
        loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True, forward=forward))
        objective_passes += 1
        gradient = _flatten({name: value.grad for name, value in parameters.items()}).detach().clone()
        if progress is not None:
            progress({'stage': 'gradient_reliable_complete_objective_pass',
                'full_objective_passes': objective_passes, 'completed_observations': len(problem.rows),
                'total_observations': len(problem.rows)})
        return loss, gradient

    def work() -> dict:
        return {'full_objective_passes': objective_passes,
            'full_exact_hessian_passes': exact.full_curvature_passes-initial_curvature_passes,
            'trials': trials, 'original_objective_unchanged': True,
            'outer_gradient_convergence_certified': False, 'gradient_contraction_guaranteed': False}

    try:
        initial_loss, initial_gradient = evaluate()
        if not math.isfinite(initial_loss) or not bool(torch.isfinite(initial_gradient).all()):
            raise FloatingPointError('nonfinite initial complete loss or gradient')
        slope = float(torch.dot(initial_gradient, direction))
        reference_norm = full_metric_norm(metric, initial_gradient)
        if not math.isfinite(slope) or not slope < 0. or not math.isfinite(reference_norm) or reference_norm == 0.:
            raise ValueError('gradient reliability requires a finite strictly descending nonstationary direction')
        rate = config.initial_rate
        while rate >= config.minimum_rate:
            assign(original+rate*direction)
            increment = _flatten({name: value.detach() for name, value in parameters.items()})-original
            actual_slope = float(torch.dot(initial_gradient, increment))
            trial = {'rate': rate, 'actual_increment_slope': actual_slope, 'armijo_passed': False,
                'gradient_reliability_passed': False}
            if (not bool(increment.ne(0.).any()) or not bool(torch.isfinite(increment).all())
                or not math.isfinite(actual_slope) or not actual_slope < 0.):
                trial['rejection'] = 'nonfinite_zero_or_nondescent_actual_increment'
            else:
                loss, gradient = evaluate()
                trial.update(training_loss=loss, finite_gradient=bool(torch.isfinite(gradient).all()))
                if math.isfinite(loss) and trial['finite_gradient'] and loss <= initial_loss+config.armijo*actual_slope:
                    trial['armijo_passed'] = True
                    hessian_image = exact.product(increment)
                    remainder = gradient-initial_gradient-hessian_image
                    remainder_norm = full_metric_norm(metric, remainder)
                    fraction = remainder_norm/reference_norm
                    trial.update(nonlinear_gradient_metric_norm=remainder_norm,
                        nonlinear_fraction_of_initial_gradient=fraction,
                        initial_gradient_metric_norm=reference_norm,
                        final_gradient_metric_norm=full_metric_norm(metric, gradient),
                        nonlinear_raw_norm=float(torch.linalg.vector_norm(remainder)))
                    if math.isfinite(fraction) and fraction <= config.nonlinear_fraction:
                        verify_actual_metric(metric, policy)
                        trial['gradient_reliability_passed'] = True
                        trials.append(trial)
                        result = {'phase': 'full-policy-GGN-gradient-reliable-Armijo', 'accepted': True,
                            'initial_loss': initial_loss, 'verified_training_loss': loss,
                            'rate': rate, 'next_initial_rate': min(1., rate*2.) if fraction <= config.expansion_fraction else rate,
                            'slope': slope, 'actual_increment_slope': actual_slope, **work()}
                        if progress is not None:
                            progress({'stage': 'gradient_reliable_candidate_accepted', **result})
                        return result, increment, gradient-initial_gradient
                    trial['rejection'] = 'nonlinear_gradient_model_unreliable'
                else:
                    trial['rejection'] = 'nonfinite_or_no_canonical_Armijo_descent'
            trials.append(trial)
            if progress is not None:
                progress({'stage': 'gradient_reliable_rejection', **trial, **work()})
            rate *= .5
        rollback()
        return {'phase': 'full-policy-GGN-gradient-reliable-Armijo', 'accepted': False,
            'stop_reason': 'gradient_reliable_line_search_not_converged',
            'original_state_restored': True, **work()}, torch.zeros_like(original), torch.zeros_like(original)
    except BaseException as error:
        rollback()
        error.gradient_reliable_work = work()
        raise
