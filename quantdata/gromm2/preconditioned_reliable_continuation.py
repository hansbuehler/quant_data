"""Strict-gradient continuation with frozen history preconditioning per solve."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_continuation import (
    AcceptedGGNWorkspace,
    GGNStoppingReference,
    _forward_geometry,
    _point_digest,
    action_point_sha256,
    optimizer_state_sha256,
)
from quantdata.gromm2.ggn_audit_optimization import GGNStepConfig, InverseAffineMetric, _flatten
from quantdata.gromm2.gradient_reliable_continuation import AcceptedHessianSnapshot
from quantdata.gromm2.gradient_reliable_ggn import (
    GradientReliabilityConfig,
    accept_gradient_reliable_direction,
    verify_actual_metric,
)
from quantdata.gromm2.lbfgs_audit_preconditioner import (
    FrozenLBFGSPreconditioner,
    metric_energy,
    pcg_separate_residual_norm,
)
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass


@version('0.1', dependencies=[GGNStepConfig, GradientReliabilityConfig])
@dataclass(frozen=True, kw_only=True)
class PreconditionedReliableConfig:
    """A numerical work window, not a replacement for original gradient gates."""

    step: GGNStepConfig = GGNStepConfig(damping=.0001, maximum_cg_iterations=120)
    acceptance: GradientReliabilityConfig = GradientReliabilityConfig()
    maximum_accepted_steps: int = 4

    def __post_init__(self) -> None:
        verify_inp(type(self.maximum_accepted_steps) is int and self.maximum_accepted_steps > 0,
            lambda: 'a positive bounded preconditioned reliable-GGN work window is required')


class AcceptedPreconditionedGGNWorkspace(AcceptedGGNWorkspace):
    """Preserve the accepted operator while freezing history for each proposal."""

    def propose(self, gradient: torch.Tensor, optimizer: MetricLBFGS, *,
                config: GGNStepConfig, expected_generation: int) -> tuple[torch.Tensor, dict]:
        if self.active or expected_generation != self.generation or config.compile != self.compile:
            raise ValueError('preconditioned proposal has an active, stale or differently compiled workspace')
        self._check_fixed_problem()
        metric = optimizer.metric
        verify_actual_metric(metric, self.policy)
        if (gradient.shape != (self.curvature.size,) or metric.size != self.curvature.size
            or not bool(torch.isfinite(gradient).all()) or float(torch.linalg.vector_norm(gradient)) == 0.
            or action_point_sha256(self.policy) != self.point_sha256
            or _point_digest(self.curvature) != self.point_sha256
            or self.curvature.functional_policy.affine_only
            or asdict(self.curvature.functional_policy.config) != self.model_config
            or _forward_geometry(self.curvature.functional_policy) != self.geometry):
            raise ValueError('preconditioned proposal requires the original finite nonzero gradient and accepted point')
        frozen = FrozenLBFGSPreconditioner(optimizer, self.policy)
        inverse = InverseAffineMetric(frozen.metric)
        before = self.curvature.full_curvature_passes
        self.active = True

        def operator(value: torch.Tensor) -> torch.Tensor:
            frozen.check_fixed()
            image = self.curvature.product(value)+config.damping*inverse(value)
            frozen.check_fixed()
            return image

        try:
            direction, linear = pcg_separate_residual_norm(operator, -gradient, frozen.apply_and_energy,
                stopping_energy=lambda value: metric_energy(frozen.metric, value),
                relative_tolerance=config.cg_relative_tolerance, maximum_iterations=config.maximum_cg_iterations,
                progress=self.curvature.progress)
            slope = float(torch.dot(gradient, direction))
            if not math.isfinite(slope) or not slope < 0. or not bool(torch.isfinite(direction).all()):
                raise RuntimeError('preconditioned proposal is not finite strict descent')
            if (optimizer_state_sha256(optimizer) != frozen.source_sha256
                or action_point_sha256(self.policy) != self.point_sha256
                or _point_digest(self.curvature) != self.point_sha256
                or self.curvature.functional_policy.affine_only
                or asdict(self.curvature.functional_policy.config) != self.model_config
                or _forward_geometry(self.curvature.functional_policy) != self.geometry):
                raise ValueError('native point or raw history changed during the frozen solve')
            frozen.check_fixed()
            verify_actual_metric(metric, self.policy)
            self._check_fixed_problem()
            return direction, {'phase': 'coupled-full-policy-frozen-LBFGS-damped-GGN',
                'linear_solve': linear, 'damping': config.damping, 'slope': slope,
                'preconditioner': 'frozen-positive-original-raw-LBFGS',
                'preconditioner_source_sha256': frozen.source_sha256,
                'preconditioner_pairs': len(frozen.history),
                'full_curvature_passes': self.curvature.full_curvature_passes-before,
                'working_memory_batch_size': self.problem.config.batch_size,
                'full_training_observations': len(self.problem.rows),
                'workspace_generation': self.generation, 'point_sha256': self.point_sha256,
                'original_objective_unchanged': True, 'outer_gradient_convergence_certified': False,
                'coefficient_directions_discarded': False, 'damping_is_objective_penalty': False,
                'policy_parameters_changed': False}
        finally:
            self.active = False


def continue_preconditioned_reliable_action(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                                      optimizer: MetricLBFGS, diagnostic: str, data_identity: str,
                                      reference: GGNStoppingReference, source_point: dict,
                                      config: PreconditionedReliableConfig = PreconditionedReliableConfig(),
                                      resume_state: dict | None = None, progress: Progress | None = None,
                                      checkpoint: Callable[[AffineDeepAction, MetricLBFGS, dict], None] | None = None) -> dict:
    """Preserve original stationarity gates and exact accepted-native lineage.

    Frozen positive raw-history PCG changes only the numerical inner solver.
    The damping and original-M residual target remain unchanged. The line search
    also controls the measured nonlinear gradient remainder. This is a bounded
    experiment, not a proof of gradient contraction or global convergence.
    """
    if not data_identity or diagnostic not in ('squared', 'entropy'):
        raise ValueError('the original training identity and objective are required')
    metric = optimizer.metric
    verify_actual_metric(metric, policy)
    if (any(actual is not expected for actual, expected in zip(metric.parameters, policy.parameters(), strict=True))
        or hashlib.sha256(metric.factors.cpu().contiguous().numpy().tobytes()).hexdigest() != metric.identity['factor_sha256']):
        raise ValueError('the original policy or fixed positive optimizer metric changed')
    identity = {'data_identity': data_identity, 'model_config': asdict(policy.config),
        'model_code_version': policy.config.version.unique_id48, 'model_user_version': policy.config.user_version,
        'continuation_version': PreconditionedReliableConfig.version.unique_id48,
        'proposal_preconditioner': 'frozen-original-positive-raw-LBFGS',
        'diagnostic': diagnostic, 'reference': asdict(reference), 'metric_identity': deepcopy(metric.identity),
        'step_config': asdict(config.step), 'acceptance_config': asdict(config.acceptance),
        'source_point': deepcopy(source_point), 'training_rows': len(problem.rows),
        'training_rows_sha256': hashlib.sha256(problem.rows.cpu().contiguous().numpy().tobytes()).hexdigest(),
        'eligibility_sha256': hashlib.sha256(problem.eligible.cpu().contiguous().numpy().tobytes()).hexdigest(),
        'working_memory_batch_size': problem.config.batch_size,
        'resident_shapes': [tuple(value.shape) for value in (problem.states, problem.returns, problem.eligible, problem.rows)],
        'input_precision': {'states': str(problem.states.dtype), 'returns': str(problem.returns.dtype), 'arithmetic': 'torch.float64'},
        'forward_geometry': _forward_geometry(policy),
        'optimizer_controls': {'history_size': optimizer.history_size, 'curvature_floor': optimizer.curvature_floor}}
    if resume_state is not None and (resume_state['identity'] != identity
        or resume_state['optimizer_state_sha256'] != optimizer_state_sha256(optimizer)):
        raise ValueError('reliable-GGN restart data, gates, source or actual raw optimizer history changed')
    history = [] if resume_state is None else deepcopy(resume_state['history'])
    if len(history) > config.maximum_accepted_steps:
        raise ValueError('reliable-GGN restart exceeds its declared work window')
    objective_passes = reference.prior_full_objective_passes if resume_state is None else resume_state['full_objective_passes']
    prior_curvature = reference.prior_full_curvature_passes if resume_state is None else resume_state['full_curvature_passes']
    next_rate = config.acceptance.initial_rate if resume_state is None else resume_state['next_initial_rate']
    forward = torch.compile(policy) if config.step.compile else policy
    policy.zero_grad(set_to_none=True)
    loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True, forward=forward))
    objective_passes += 1
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()}).detach().clone()
    norms = metric.diagnostics(gradient)
    expected = source_point if not history else history[-1]
    if (not math.isclose(loss, expected['training_loss'], rel_tol=1.e-10, abs_tol=1.e-11)
        or any(not math.isclose(norms[name], expected[name], rel_tol=1.e-9, abs_tol=1.e-10)
            for name in ('full_gradient_norm', 'affine_metric_gradient_norm'))
        or action_point_sha256(policy) != (source_point['point_sha256'] if resume_state is None else resume_state['point_sha256'])):
        raise ValueError('reliable-GGN did not restore the actual original accepted point')
    ggn = AcceptedPreconditionedGGNWorkspace(policy, problem, diagnostic=diagnostic, compile=config.step.compile,
        progress=progress, initial_generation=len(history))
    exact = AcceptedHessianSnapshot(policy, problem, diagnostic=diagnostic, compile=config.step.compile,
        progress=progress, initial_generation=len(history))
    converged, stop_reason = False, 'work_limit_not_converged'
    rejected_work = None

    def state() -> dict:
        return {'identity': identity, 'history': deepcopy(history), 'config': asdict(config),
            'full_objective_passes': objective_passes,
            'full_curvature_passes': prior_curvature+ggn.curvature.full_curvature_passes+exact.curvature.full_curvature_passes,
            'invocation_ggn_passes': ggn.curvature.full_curvature_passes,
            'invocation_exact_hessian_passes': exact.curvature.full_curvature_passes,
            'current_training_loss': loss, 'current_norms': norms, 'point_sha256': action_point_sha256(policy),
            'optimizer_state_sha256': optimizer_state_sha256(optimizer), 'next_initial_rate': next_rate,
            'converged': converged, 'stop_reason': stop_reason, 'rejected_work': deepcopy(rejected_work),
            'global_optimum_certified': False, 'gradient_contraction_guaranteed': False,
            'validation_used_for_steps_or_stopping': False,
            'work_scope': 'committed full passes; interrupted partial and replayed work are logged separately'}

    def interrupted_work(error: BaseException, candidate_work: dict | None = None) -> None:
        extra = getattr(error, 'gradient_reliable_work', None) or candidate_work or {}
        error.reliable_attempt_work = {
            'accepted_steps_before_interruption': len(history),
            'complete_objective_passes': objective_passes+extra.get('full_objective_passes', 0),
            'complete_curvature_passes': prior_curvature+ggn.curvature.full_curvature_passes+exact.curvature.full_curvature_passes,
            'invocation_ggn_passes': ggn.curvature.full_curvature_passes,
            'invocation_exact_hessian_passes': exact.curvature.full_curvature_passes,
            'candidate_work': deepcopy(extra), 'partial_products_excluded': True}

    for accepted in range(len(history), config.maximum_accepted_steps+1):
        converged = norms['full_gradient_norm'] <= reference.raw_threshold and norms['affine_metric_gradient_norm'] <= reference.affine_metric_threshold
        if converged:
            stop_reason = 'original_raw_and_affine_metric_gradient_tolerances'
        if checkpoint is not None:
            checkpoint(policy, optimizer, state())
        if progress is not None:
            progress({'stage': 'gradient_reliable_accepted_point', 'accepted_steps': accepted,
                'maximum_accepted_steps': config.maximum_accepted_steps, 'training_loss': loss, **norms,
                'gradient_threshold': reference.raw_threshold, 'affine_metric_gradient_threshold': reference.affine_metric_threshold})
        if converged or accepted == config.maximum_accepted_steps:
            break
        try:
            direction, proposal = ggn.propose(gradient, optimizer, config=config.step, expected_generation=ggn.generation)
        except BaseException as error:
            interrupted_work(error)
            raise
        original = {name: value.detach().clone() for name, value in policy.named_parameters()}
        original_gradients = {name: None if value.grad is None else value.grad.detach().clone()
            for name, value in policy.named_parameters()}
        original_optimizer = deepcopy(optimizer.state_dict())
        original_history = list(history)
        candidate_work = None
        exact.guard.active = True
        try:
            acceptance, increment, change = accept_gradient_reliable_direction(policy=policy, problem=problem,
                direction=direction, exact=exact.curvature, metric=metric, diagnostic=diagnostic,
                config=replace(config.acceptance, initial_rate=next_rate), forward=forward, progress=progress)
            candidate_work = acceptance
            if acceptance['accepted']:
                candidate_gradient = _flatten({name: value.grad for name, value in policy.named_parameters()}).detach().clone()
                candidate_norms = metric.diagnostics(candidate_gradient)
                retained = optimizer.accept(increment, change)
                ggn.bind_accepted_point()
                exact.guard.active = False
                exact.bind_accepted_point()
                verify_actual_metric(metric, policy)
                candidate_entry = {'iteration': reference.source_accepted_updates+accepted+1,
                    'training_loss': acceptance['verified_training_loss'], **candidate_norms,
                    'gradient_threshold': reference.raw_threshold, 'affine_metric_gradient_threshold': reference.affine_metric_threshold,
                    'full_objective_evaluations': objective_passes+acceptance['full_objective_passes'],
                    'full_curvature_passes': proposal['full_curvature_passes']+acceptance['full_exact_hessian_passes'],
                    'full_ggn_passes': proposal['full_curvature_passes'], 'full_exact_hessian_passes': acceptance['full_exact_hessian_passes'],
                    'optimizer_phase': 'GGN-LBFGS-preconditioned-gradient-reliable-canonical-Armijo',
                    'proposal': proposal, 'acceptance': acceptance, 'actual_positive_secant_retained': retained,
                    'point_sha256': action_point_sha256(policy)}
                history.append(candidate_entry)
        except BaseException as error:
            with torch.no_grad():
                for name, value in policy.named_parameters():
                    value.copy_(original[name])
                    value.grad = None if original_gradients[name] is None else original_gradients[name].clone()
            optimizer.load_state_dict(original_optimizer)
            history[:] = original_history
            interrupted_work(error, candidate_work)
            raise
        finally:
            exact.guard.active = False
        objective_passes += acceptance['full_objective_passes']
        if not acceptance['accepted']:
            rejected_work = {'proposal': proposal, 'acceptance': acceptance}
            stop_reason = acceptance['stop_reason']
            if checkpoint is not None:
                checkpoint(policy, optimizer, state())
            break
        loss, next_rate = acceptance['verified_training_loss'], acceptance['next_initial_rate']
        gradient, norms = candidate_gradient, candidate_norms
    return state()

