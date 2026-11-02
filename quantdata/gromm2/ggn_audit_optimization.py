"""Coupled full-policy curvature for the unchanged conditional objectives."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import (
    AffineAuditProblem,
    Progress,
    preconditioned_cg,
)
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric
from quantdata.gromm2.neural_audit_optimization import full_action_pass


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class GGNStepConfig:
    """Inner linear-solve controls, never outer stationarity tolerances.

    Damping multiplies the inverse optimizer metric in the step equation.
    It is not a penalty in the objective and discards no action direction.
    """

    damping: float = .001
    cg_relative_tolerance: float = .1
    maximum_cg_iterations: int = 30
    compile: bool = False

    def __post_init__(self) -> None:
        verify_inp(math.isfinite(self.damping) and self.damping > 0.,
            lambda: 'GGN step damping must be finite and strictly positive')
        verify_inp(0. < self.cg_relative_tolerance < 1. and self.maximum_cg_iterations > 0,
            lambda: 'GGN requires positive linear-solve controls')
        verify_inp(isinstance(self.compile, bool), lambda: 'compile must be boolean')


def _flatten(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values.values()])


def _unpack(vector: torch.Tensor, template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    values, offset = {}, 0
    for name, parameter in template.items():
        values[name] = vector[offset:offset+parameter.numel()].reshape_as(parameter)
        offset += parameter.numel()
    if vector.ndim != 1 or offset != len(vector):
        raise ValueError('GGN parameter-vector geometry changed')
    return values


class FullActionGGN:
    """Stream J-transpose H-output J at one immutable accepted parameter state.

    Both objectives are convex in actions, giving positive-semidefinite GGN
    curvature. This is not the exact nonlinear Hessian or a convergence
    certificate. Only original training rows enter each complete product.
    Native tensors, ordinary gradients and RNG are not modified.
    """

    def __init__(self, policy: AffineDeepAction, problem: AffineAuditProblem, *,
                 diagnostic: str, compile: bool = False, progress: Progress | None = None) -> None:
        if diagnostic not in ('squared', 'entropy') or policy.affine_only:
            raise ValueError('GGN requires a known objective and the entire policy')
        if (policy.config.action_scale != problem.action_scale
            or policy.config.squared_lambda != problem.squared_lambda):
            raise ValueError('GGN policy and original objective scales differ')
        self.policy, self.problem, self.diagnostic = policy, problem, diagnostic
        self.functional_policy = deepcopy(policy)
        self.parameters = {name: value.detach().clone() for name, value in policy.named_parameters()}
        self.buffers = {name: value.detach().clone() for name, value in policy.named_buffers()}
        if any(value.dtype != torch.float64 or value.device != problem.states.device
            for value in self.parameters.values()):
            raise ValueError('GGN requires float64 parameters on the training device')
        self.size = sum(value.numel() for value in self.parameters.values())
        self.full_curvature_passes = 0
        self.progress = progress
        self.block_product = torch.compile(self._block_product) if compile else self._block_product

    def _block_product(self, features: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor,
                       direction: dict[str, torch.Tensor]) -> torch.Tensor:
        def action_function(values: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.func.functional_call(self.functional_policy, (values, self.buffers), (features, mask), strict=True)

        action, pullback = torch.func.vjp(action_function, self.parameters)
        _, tangent = torch.func.jvp(action_function, (self.parameters,), (direction,))
        squared_scale = self.problem.action_scale**2
        if self.diagnostic == 'squared':
            image = tangent+self.problem.squared_lambda*squared_scale*(tangent*returns).sum(dim=1)[:, None]*returns
        else:
            residual = action-returns
            exponent = squared_scale*(-(action*returns).sum(dim=1)+.5*action.square().sum(dim=1))
            image = torch.exp(exponent)[:, None]*(tangent+squared_scale*(tangent*residual).sum(dim=1)[:, None]*residual)
        contribution, = pullback(image/len(self.problem.rows))
        return _flatten(contribution).detach()

    def product(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply one complete, count-weighted matrix-free curvature product."""
        first = next(iter(self.parameters.values()))
        if vector.shape != (self.size,) or vector.dtype != first.dtype or vector.device != first.device:
            raise ValueError('GGN vector shape, dtype or device changed')
        direction = _unpack(vector, self.parameters)
        result = torch.zeros_like(vector)
        for augmented, returns, mask in self.problem.blocks():
            result.add_(self.block_product(augmented[:, :-1], returns, mask, direction))
        self.full_curvature_passes += 1
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('nonfinite full GGN product; no observations may be discarded')
        if self.progress is not None:
            self.progress({'stage': 'full_ggn_curvature_pass',
                'full_curvature_passes': self.full_curvature_passes,
                'completed_observations': len(self.problem.rows), 'total_observations': len(self.problem.rows)})
        return result


class InverseAffineMetric:
    """Invert the existing positive step metric, not the empirical Hessian."""

    def __init__(self, metric: AffineBlockMetric) -> None:
        self.metric = metric
        self.inverse_factors = torch.linalg.inv(metric.factors)
        if not bool(torch.isfinite(self.inverse_factors).all()):
            raise FloatingPointError('nonfinite inverse optimizer-metric factors')

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        affine = self.metric.affine_values(value)
        transformed = torch.empty_like(affine)
        for factor, columns in zip(self.inverse_factors, self.metric.columns, strict=True):
            transformed[:, columns] = factor.T@(factor@affine[:, columns])
        result = value/self.metric.deep_scale
        result[self.metric.weight[0]] = transformed[:-1].T.reshape(-1)
        result[self.metric.bias[0]] = transformed[-1]
        return result


def propose_full_ggn_step(*, curvature: FullActionGGN, gradient: torch.Tensor,
                          metric: AffineBlockMetric, config: GGNStepConfig = GGNStepConfig(),
                          precondition: Callable | None = None, progress: Progress | None = None) -> tuple[torch.Tensor, dict]:
    """Return a coupled damped direction without changing the accepted policy.

    An inexact inner solve may yield a useful descent direction but can never
    establish the original full-gradient convergence criteria. Acceptance must
    separately evaluate the unchanged canonical empirical objective.
    """
    if (gradient.shape != (curvature.size,) or metric.size != curvature.size
        or any(parameter is not original for parameter, original in
            zip(metric.parameters, curvature.policy.parameters(), strict=True))):
        raise ValueError('GGN gradient and metric do not share the original policy geometry')
    current = dict(curvature.policy.named_parameters())
    if (curvature.policy.affine_only or curvature.policy.config != curvature.functional_policy.config
        or list(current) != list(curvature.parameters)
        or not bool(torch.stack([(current[name].detach() == value).all()
            for name, value in curvature.parameters.items()]).all())):
        raise ValueError('the original policy no longer matches the immutable GGN snapshot')
    if not bool(torch.isfinite(gradient).all()) or float(torch.linalg.vector_norm(gradient)) == 0.:
        raise ValueError('a finite nonzero original gradient is required for a GGN proposal')
    inverse = InverseAffineMetric(metric)
    initial_passes = curvature.full_curvature_passes

    def operator(value: torch.Tensor) -> torch.Tensor:
        return curvature.product(value)+config.damping*inverse(value)

    direction, linear = preconditioned_cg(operator, -gradient, metric if precondition is None else precondition,
        relative_tolerance=config.cg_relative_tolerance, maximum_iterations=config.maximum_cg_iterations,
        progress=progress)
    slope = float(torch.dot(gradient, direction))
    if not math.isfinite(slope) or not slope < 0. or not bool(torch.isfinite(direction).all()):
        raise RuntimeError('the full GGN proposal is not finite strict descent')
    return direction, {'phase': 'coupled-full-policy-damped-GGN', 'linear_solve': linear,
        'damping': config.damping, 'slope': slope,
        'full_curvature_passes': curvature.full_curvature_passes-initial_passes,
        'working_memory_batch_size': curvature.problem.config.batch_size,
        'full_training_observations': len(curvature.problem.rows),
        'original_objective_unchanged': True, 'outer_gradient_convergence_certified': False,
        'coefficient_directions_discarded': False, 'damping_is_objective_penalty': False,
        'policy_parameters_changed': False}


def accept_full_ggn_direction(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                              direction: torch.Tensor, diagnostic: str,
                              minimum_step: float = 2.**-30, armijo: float = 1.e-4,
                              forward: Callable | None = None, progress: Progress | None = None) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Accept a proposal against the canonical loss, or restore the full state.

    Every trial includes all original training rows. Nonfinite trials are
    rejected, not clipped or censored. Only a finite verified full-gradient
    point is committed. This Armijo phase is not an existing Wolfe update
    and does not establish any outer numerical convergence criterion.
    """
    if diagnostic not in ('squared', 'entropy') or not 0. < minimum_step <= 1. or not 0. < armijo < .5:
        raise ValueError('known objective and valid Armijo controls are required')
    if (policy.affine_only or policy.config.action_scale != problem.action_scale
        or policy.config.squared_lambda != problem.squared_lambda):
        raise ValueError('canonical acceptance requires the original full policy and objective scales')
    parameters = dict(policy.named_parameters())
    original = _flatten({name: value.detach() for name, value in parameters.items()}).clone()
    entry_gradients = {name: None if value.grad is None else value.grad.detach().clone()
        for name, value in parameters.items()}
    if direction.shape != original.shape or direction.dtype != original.dtype or direction.device != original.device:
        raise ValueError('the GGN direction does not share the original parameter geometry')
    evaluations = 0

    def evaluate(backward: bool) -> float:
        nonlocal evaluations
        if backward:
            policy.zero_grad(set_to_none=True)
        value = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=backward, forward=forward))
        evaluations += 1
        if progress is not None:
            progress({'stage': 'full_ggn_acceptance_pass', 'full_objective_passes': evaluations})
        return value

    def assign(values: torch.Tensor) -> None:
        unpacked = _unpack(values, parameters)
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(unpacked[name])

    def rollback() -> None:
        assign(original)
        for name, parameter in parameters.items():
            parameter.grad = None if entry_gradients[name] is None else entry_gradients[name].clone()

    try:
        initial_loss = evaluate(True)
        initial_gradient = _flatten({name: value.grad for name, value in parameters.items()}).detach().clone()
    except BaseException:
        rollback()
        raise

    slope = float(torch.dot(initial_gradient, direction))
    if not math.isfinite(initial_loss) or not bool(torch.isfinite(initial_gradient).all()):
        rollback()
        raise FloatingPointError('the original accepted full loss or gradient is nonfinite')
    if not math.isfinite(slope) or not slope < 0.:
        rollback()
        raise ValueError('canonical full-gradient verification rejects a non-descent direction')
    rate, trials = 1., 0
    try:
        while rate >= minimum_step:
            assign(original+rate*direction)
            actual = _flatten({name: value.detach() for name, value in parameters.items()})-original
            actual_slope = float(torch.dot(initial_gradient, actual))
            trials += 1
            if not bool(actual.ne(0.).any()):
                break
            if not bool(torch.isfinite(actual).all()) or not math.isfinite(actual_slope) or not actual_slope < 0.:
                rate *= .5
                continue
            candidate = evaluate(False)
            if math.isfinite(candidate) and candidate <= initial_loss+armijo*actual_slope:
                verified = evaluate(True)
                gradient = _flatten({name: value.grad for name, value in parameters.items()}).detach().clone()
                if (math.isfinite(verified) and bool(torch.isfinite(gradient).all())
                    and verified <= initial_loss+armijo*actual_slope):
                    return {'phase': 'full-policy-GGN-canonical-Armijo', 'accepted': True,
                        'initial_loss': initial_loss, 'verified_training_loss': verified,
                        'initial_full_gradient_norm': float(torch.linalg.vector_norm(initial_gradient)),
                        'verified_full_gradient_norm': float(torch.linalg.vector_norm(gradient)),
                        'rate': rate, 'trials': trials, 'slope': slope,
                        'actual_increment_slope': actual_slope,
                        'full_objective_passes': evaluations, 'original_objective_unchanged': True,
                        'outer_gradient_convergence_certified': False}, actual, gradient-initial_gradient
            rate *= .5
        rollback()
    except BaseException:
        rollback()
        raise
    return {'phase': 'full-policy-GGN-canonical-Armijo', 'accepted': False,
        'stop_reason': 'line_search_not_converged', 'trials': trials,
        'full_objective_passes': evaluations, 'original_state_restored': True,
        'outer_gradient_convergence_certified': False}, torch.zeros_like(original), torch.zeros_like(original)
