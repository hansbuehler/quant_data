"""Checkpointable full-data GGN continuation with original strict gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_optimization import (
    FullActionGGN,
    GGNStepConfig,
    _flatten,
    accept_full_ggn_direction,
    propose_full_ggn_step,
)
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass


def _point_digest(curvature: FullActionGGN) -> str:
    digest = hashlib.sha256()
    for name, value in (*curvature.parameters.items(), *curvature.buffers.items()):
        digest.update(name.encode())
        digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def action_point_sha256(policy: AffineDeepAction) -> str:
    """Identify exact native parameter and buffer values at an accepted point."""
    digest = hashlib.sha256()
    for name, value in (*policy.named_parameters(), *policy.named_buffers()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _forward_geometry(policy: AffineDeepAction) -> list:
    return [(name, type(module).__module__, type(module).__qualname__, module.training,
        getattr(module, 'use_final_activation', None)) for name, module in policy.named_modules()]


def optimizer_state_sha256(optimizer: MetricLBFGS) -> str:
    """Bind retained raw secants and controls without changing or dropping them."""
    digest = hashlib.sha256(json.dumps({'history_size': optimizer.history_size,
        'curvature_floor': optimizer.curvature_floor, 'metric_identity': optimizer.metric.identity}, sort_keys=True).encode())
    for entry in optimizer.history:
        for value in entry:
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AcceptedGGNWorkspace:
    """Reuse compiled tensor storage only between completed accepted-point solves.

    The Krylov operator is fixed throughout each proposal. Explicit refreshes
    copy new native weights into the same snapshot tensors, so compiled graphs
    see the new values without rebuilding the model or replacing tensor storage.
    This helper never changes native weights, gradients or an objective.
    """

    def __init__(self, policy: AffineDeepAction, problem: AffineAuditProblem, *,
                 diagnostic: str, compile: bool = False, progress: Progress | None = None,
                 initial_generation: int = 0) -> None:
        if type(initial_generation) is not int or initial_generation < 0:
            raise ValueError('a nonnegative accepted-point generation is required')
        self.curvature = FullActionGGN(policy, problem, diagnostic=diagnostic, compile=compile, progress=progress)
        self.policy, self.problem, self.compile = policy, problem, compile
        self.geometry = _forward_geometry(policy)
        self.model_config = asdict(policy.config)
        self.rows = problem.rows.detach().clone()
        self.eligible = problem.eligible.detach().clone()
        self.data_objects = (problem.states, problem.returns, problem.rows, problem.eligible)
        self.data_geometry = [(tuple(value.shape), value.dtype, value.device) for value in self.data_objects]
        self.data_versions = tuple(value._version for value in self.data_objects)
        self.objective = (problem.action_scale, problem.squared_lambda, problem.config.batch_size)
        self.generation = initial_generation
        self.point_sha256 = _point_digest(self.curvature)
        self.active = False

    def _check_fixed_problem(self) -> None:
        if (any(actual is not expected for actual, expected in zip(self.data_objects,
            (self.problem.states, self.problem.returns, self.problem.rows, self.problem.eligible), strict=True))
            or self.data_geometry != [(tuple(value.shape), value.dtype, value.device) for value in self.data_objects]
            or self.data_versions != tuple(value._version for value in self.data_objects)
            or not torch.equal(self.rows, self.problem.rows) or not torch.equal(self.eligible, self.problem.eligible)
            or self.objective != (self.problem.action_scale, self.problem.squared_lambda, self.problem.config.batch_size)
            or self.policy.affine_only or asdict(self.policy.config) != self.model_config
            or _forward_geometry(self.policy) != self.geometry):
            raise ValueError('GGN workspace data, objective or full-policy geometry changed')

    def bind_accepted_point(self) -> dict:
        """Refresh only when no Krylov solve or line-search proposal is active."""
        if self.active:
            raise RuntimeError('cannot refresh the GGN operator during an active Krylov solve')
        self._check_fixed_problem()
        current = dict(self.policy.named_parameters())
        buffers = dict(self.policy.named_buffers())
        if list(current) != list(self.curvature.parameters) or list(buffers) != list(self.curvature.buffers):
            raise ValueError('GGN workspace parameter or buffer names changed')
        values = [(destination, current[name] if name in current else buffers[name])
            for name, destination in (*self.curvature.parameters.items(), *self.curvature.buffers.items())]
        if any(destination.shape != value.shape or destination.dtype != value.dtype or destination.device != value.device
            for destination, value in values):
            raise ValueError('GGN workspace parameter shape, precision or device changed')
        if not all(bool(torch.isfinite(value).all()) for _, value in values):
            raise FloatingPointError('GGN workspace cannot bind nonfinite native parameters')
        with torch.no_grad():
            for destination, value in values:
                destination.copy_(value.detach())
        self.generation += 1
        self.point_sha256 = _point_digest(self.curvature)
        return {'generation': self.generation, 'point_sha256': self.point_sha256}

    def propose(self, gradient: torch.Tensor, metric: AffineBlockMetric, *,
                config: GGNStepConfig, expected_generation: int) -> tuple[torch.Tensor, dict]:
        """Use one fixed snapshot, rejecting stale generations and nested solves."""
        if self.active or expected_generation != self.generation or config.compile != self.compile:
            raise ValueError('GGN proposal has an active, stale or differently compiled workspace')
        self._check_fixed_problem()
        self.active = True
        try:
            direction, report = propose_full_ggn_step(curvature=self.curvature, gradient=gradient,
                metric=metric, config=config, progress=self.curvature.progress)
            if not all(torch.equal(value, dict(self.policy.named_parameters())[name].detach())
                for name, value in self.curvature.parameters.items()):
                raise ValueError('native parameters changed during the fixed GGN solve')
            if not all(torch.equal(value, dict(self.policy.named_buffers())[name].detach())
                for name, value in self.curvature.buffers.items()):
                raise ValueError('native buffers changed during the fixed GGN solve')
            self._check_fixed_problem()
            return direction, {**report, 'workspace_generation': self.generation, 'point_sha256': self.point_sha256}
        finally:
            self.active = False


@version('0.1', dependencies=[GGNStepConfig])
@dataclass(frozen=True, kw_only=True)
class GGNContinuationConfig:
    """Operational work window; only original external gradient gates can pass."""

    step: GGNStepConfig = GGNStepConfig()
    maximum_accepted_steps: int = 8

    def __post_init__(self) -> None:
        verify_inp(type(self.maximum_accepted_steps) is int and self.maximum_accepted_steps > 0,
            lambda: 'a positive bounded GGN continuation work window is required')


@dataclass(frozen=True, kw_only=True)
class GGNStoppingReference:
    """Immutable original tolerances and source work, independently source-bound."""

    source_checkpoint_sha256: str
    initial_gradient_norm: float
    raw_threshold: float
    affine_metric_threshold: float
    source_accepted_updates: int
    prior_full_objective_passes: int
    prior_full_curvature_passes: int

    def __post_init__(self) -> None:
        verify_inp(len(self.source_checkpoint_sha256) == 64
            and all(value in '0123456789abcdef' for value in self.source_checkpoint_sha256),
            lambda: 'an accepted source checkpoint SHA256 is required')
        verify_inp(math.isfinite(self.initial_gradient_norm) and self.initial_gradient_norm >= 0.
            and all(math.isfinite(value) and value > 0. for value in (self.raw_threshold, self.affine_metric_threshold)),
            lambda: 'original gradient reference and stopping thresholds must be finite')
        verify_inp(all(type(value) is int and value >= 0. for value in (
            self.source_accepted_updates, self.prior_full_objective_passes, self.prior_full_curvature_passes)),
            lambda: 'source accepted-update and work counters must be nonnegative integers')


def continue_full_ggn_action(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                             optimizer: MetricLBFGS, diagnostic: str, data_identity: str,
                             reference: GGNStoppingReference, source_point: dict,
                             config: GGNContinuationConfig = GGNContinuationConfig(),
                             resume_state: dict | None = None, progress: Progress | None = None,
                             checkpoint: Callable | None = None) -> dict:
    """Continue all original parameters, never select using validation outcomes.

    Checkpoints contain only coherent accepted points and their committed work.
    Interrupted in-flight CG is not resumed: discarded/replayed work remains in
    the persistent attempt log and must not be called committed optimization.
    A native restoration pass is counted on every invocation.
    """
    if not data_identity or diagnostic not in ('squared', 'entropy'):
        raise ValueError('bound original data and a known objective are required')
    metric = optimizer.metric
    if any(actual is not expected for actual, expected in zip(metric.parameters, policy.parameters(), strict=True)):
        raise ValueError('GGN optimizer does not share the original policy parameters')
    if hashlib.sha256(metric.factors.cpu().contiguous().numpy().tobytes()).hexdigest() != metric.identity['factor_sha256']:
        raise ValueError('the actual fixed optimizer metric differs from its source identity')
    identity = {'data_identity': data_identity, 'model_config': asdict(policy.config), 'diagnostic': diagnostic,
        'reference': asdict(reference), 'metric_identity': deepcopy(metric.identity), 'step_config': asdict(config.step),
        'source_point': deepcopy(source_point), 'training_rows_sha256': hashlib.sha256(
            problem.rows.cpu().contiguous().numpy().tobytes()).hexdigest(),
        'training_rows': len(problem.rows), 'working_memory_batch_size': problem.config.batch_size,
        'resident_shapes': [tuple(value.shape) for value in (problem.states, problem.returns, problem.eligible, problem.rows)],
        'input_precision': {'states': str(problem.states.dtype), 'returns': str(problem.returns.dtype), 'arithmetic': 'torch.float64'},
        'forward_geometry': _forward_geometry(policy),
        'optimizer_controls': {'history_size': optimizer.history_size, 'curvature_floor': optimizer.curvature_floor}}
    identity.update(model_code_version=policy.config.version.unique_id48,
        model_user_version=policy.config.user_version,
        continuation_version=GGNContinuationConfig.version.unique_id48,
        eligibility_sha256=hashlib.sha256(problem.eligible.cpu().contiguous().numpy().tobytes()).hexdigest())
    if resume_state is not None and resume_state['identity'] != identity:
        raise ValueError('GGN continuation data, original thresholds, source or solver identity changed')
    optimizer_digest = optimizer_state_sha256(optimizer)
    if resume_state is not None and resume_state['optimizer_state_sha256'] != optimizer_digest:
        raise ValueError('GGN continuation lost or changed its retained raw optimizer history')
    history = [] if resume_state is None else deepcopy(resume_state['history'])
    if len(history) > config.maximum_accepted_steps:
        raise ValueError('saved accepted work exceeds the declared GGN work window')
    objective_passes = reference.prior_full_objective_passes if resume_state is None else resume_state['full_objective_passes']
    prior_curvature = reference.prior_full_curvature_passes if resume_state is None else resume_state['full_curvature_passes']
    forward = torch.compile(policy) if config.step.compile else policy
    policy.zero_grad(set_to_none=True)
    loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True, forward=forward))
    objective_passes += 1
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()}).detach().clone()
    norms = metric.diagnostics(gradient)
    expected = source_point if not history else history[-1]
    if (not math.isclose(loss, expected['training_loss'], rel_tol=1.e-10, abs_tol=1.e-11)
        or any(not math.isclose(norms[name], expected[name], rel_tol=1.e-9, abs_tol=1.e-10)
            for name in ('full_gradient_norm', 'affine_metric_gradient_norm'))):
        raise ValueError('GGN continuation did not restore the actual accepted original objective and gradient')
    workspace = AcceptedGGNWorkspace(policy, problem, diagnostic=diagnostic, compile=config.step.compile,
        progress=progress, initial_generation=len(history))
    expected_point_sha = source_point['point_sha256'] if resume_state is None else resume_state['point_sha256']
    if expected_point_sha != workspace.point_sha256:
        raise ValueError('GGN continuation did not restore the exact accepted parameter state')
    converged = False
    stop_reason = 'work_limit_not_converged'

    def state() -> dict:
        return {'identity': identity, 'history': deepcopy(history), 'full_objective_passes': objective_passes,
            'full_curvature_passes': prior_curvature+workspace.curvature.full_curvature_passes,
            'current_training_loss': loss, 'current_norms': norms,
            'point_sha256': workspace.point_sha256,
            'optimizer_state_sha256': optimizer_digest,
            'converged': converged, 'stop_reason': stop_reason, 'config': asdict(config),
            'global_optimum_certified': False, 'validation_used_for_steps_or_stopping': False,
            'work_scope': 'committed full passes; interrupted partial and replayed work are logged separately'}

    for accepted in range(len(history), config.maximum_accepted_steps+1):
        converged = norms['full_gradient_norm'] <= reference.raw_threshold and norms['affine_metric_gradient_norm'] <= reference.affine_metric_threshold
        if converged:
            stop_reason = 'original_raw_and_affine_metric_gradient_tolerances'
        if checkpoint is not None:
            checkpoint(policy, optimizer, state())
        if progress is not None:
            progress({'stage': 'full_ggn_continuation_point', 'accepted_steps': accepted,
                'maximum_accepted_steps': config.maximum_accepted_steps, 'training_loss': loss, **norms,
                'gradient_threshold': reference.raw_threshold, 'affine_metric_gradient_threshold': reference.affine_metric_threshold})
        if converged or accepted == config.maximum_accepted_steps:
            break
        direction, proposal = workspace.propose(gradient, metric, config=config.step, expected_generation=workspace.generation)
        acceptance, increment, change = accept_full_ggn_direction(policy=policy, problem=problem,
            direction=direction, diagnostic=diagnostic, forward=forward, progress=progress)
        objective_passes += acceptance['full_objective_passes']
        if not acceptance['accepted']:
            stop_reason = 'line_search_not_converged'
            if checkpoint is not None:
                checkpoint(policy, optimizer, state())
            break
        loss = acceptance['verified_training_loss']
        gradient = _flatten({name: value.grad for name, value in policy.named_parameters()}).detach().clone()
        norms = metric.diagnostics(gradient)
        pair_retained = optimizer.accept(increment, change)
        optimizer_digest = optimizer_state_sha256(optimizer)
        history.append({'iteration': reference.source_accepted_updates+accepted+1, 'training_loss': loss, **norms,
            'gradient_threshold': reference.raw_threshold, 'affine_metric_gradient_threshold': reference.affine_metric_threshold,
            'full_objective_evaluations': objective_passes,
            'full_curvature_passes': proposal['full_curvature_passes'], 'optimizer_phase': 'coupled-GGN-canonical-Armijo',
            'proposal': proposal, 'acceptance': acceptance, 'actual_positive_secant_retained': pair_retained})
        workspace.bind_accepted_point()
        history[-1]['point_sha256'] = workspace.point_sha256
    return state()
