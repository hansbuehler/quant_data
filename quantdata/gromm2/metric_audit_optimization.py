"""Full-action L-BFGS with an invertible, training-only affine metric."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.neural_audit_optimization import NeuralSolveConfig, full_action_pass


@version('0.1', dependencies=[NeuralSolveConfig])
@dataclass(frozen=True, kw_only=True)
class MetricSolveConfig:
    neural: NeuralSolveConfig = NeuralSolveConfig()
    deep_metric_scale: float = 1.
    curvature_relative_floor: float = 1.e-12

    def __post_init__(self) -> None:
        verify_inp(math.isfinite(self.deep_metric_scale) and self.deep_metric_scale > 0.,
            lambda: 'deep metric must be finite and strictly positive')
        verify_inp(0. < self.curvature_relative_floor < 1., lambda: 'invalid curvature precision threshold')


@dataclass(frozen=True, kw_only=True)
class OptimizerMigration:
    accepted_checkpoint_sha256: str
    source_optimizer: str
    initial_gradient_norm: float
    absolute_gradient_threshold: float
    prior_accepted_updates: int
    prior_full_objective_passes: int
    curvature_history_reset: bool = True

    def __post_init__(self) -> None:
        verify_inp(len(self.accepted_checkpoint_sha256) == 64
            and all(character in '0123456789abcdef' for character in self.accepted_checkpoint_sha256),
            lambda: 'migration requires the original accepted checkpoint SHA256')
        verify_inp(bool(self.source_optimizer) and self.curvature_history_reset is True,
            lambda: 'optimizer migration must identify its source and explicitly reset curvature history')
        verify_inp(math.isfinite(self.initial_gradient_norm) and self.initial_gradient_norm >= 0.
            and math.isfinite(self.absolute_gradient_threshold) and self.absolute_gradient_threshold > 0.,
            lambda: 'migration gradient reference and original absolute threshold must be finite')
        verify_inp(type(self.prior_accepted_updates) is int and self.prior_accepted_updates >= 0
            and type(self.prior_full_objective_passes) is int and self.prior_full_objective_passes > 0,
            lambda: 'migration must preserve prior accepted-update and full-objective counts')


class AffineBlockMetric:
    """Apply positive Gram inverse factors without removing any coefficient.

    The deep block is a positive multiple of identity. This changes optimizer
    coordinates, never the action function, loss, observations or probability.
    """

    def __init__(self, policy: AffineDeepAction, problem: AffineAuditProblem, *, deep_scale: float = 1.) -> None:
        if problem.preconditioner_factors is None:
            problem.prepare()
        if not math.isfinite(deep_scale) or deep_scale <= 0.:
            raise ValueError('metric deep scale must be finite and positive')
        self.parameters = tuple(policy.parameters())
        self.slices = {}
        offset = 0
        for name, parameter in policy.named_parameters():
            self.slices[name] = (slice(offset, offset+parameter.numel()), parameter.shape)
            offset += parameter.numel()
        self.size, self.deep_scale = offset, deep_scale
        self.weight = self.slices['affine.I0.weight']
        self.bias = self.slices['affine.I0.bias']
        self.factors = problem.preconditioner_factors.detach().clone()
        self.columns = tuple(columns.clone() for columns in problem.group_columns)
        self.identity = {'parameter_shapes': {name: list(shape) for name, (_, shape) in self.slices.items()},
            'deep_scale': deep_scale, 'factor_shape': list(self.factors.shape),
            'factor_sha256': hashlib.sha256(self.factors.cpu().contiguous().numpy().tobytes()).hexdigest(),
            'columns': [columns.cpu().tolist() for columns in self.columns],
            'coefficient_directions_discarded': False, 'objective_ridge_added': False}

    def affine_values(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value[self.weight[0]].reshape(self.weight[1]).T,
            value[self.bias[0]].reshape(1, -1)), dim=0)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 1 or len(value) != self.size or value.dtype != self.factors.dtype or value.device != self.factors.device:
            raise ValueError('metric vector geometry, precision or device changed')
        affine = self.affine_values(value)
        transformed = torch.empty_like(affine)
        for factor, columns in zip(self.factors, self.columns, strict=True):
            transformed[:, columns] = factor @ (factor.T @ affine[:, columns])
        result = self.deep_scale*value
        result[self.weight[0]] = transformed[:-1].T.reshape(-1)
        result[self.bias[0]] = transformed[-1]
        return result

    def diagnostics(self, gradient: torch.Tensor) -> dict[str, float]:
        affine = self.affine_values(gradient)
        energy = torch.zeros((), dtype=gradient.dtype, device=gradient.device)
        for factor, columns in zip(self.factors, self.columns, strict=True):
            energy.add_((factor.T@affine[:, columns]).square().sum())
        deep = gradient.clone()
        deep[self.weight[0]], deep[self.bias[0]] = 0., 0.
        return {'full_gradient_norm': float(torch.linalg.vector_norm(gradient)),
            'affine_gradient_norm': float(torch.linalg.vector_norm(affine)),
            'affine_metric_gradient_norm': math.sqrt(max(float(energy), 0.)),
            'deep_gradient_norm': float(torch.linalg.vector_norm(deep))}


class MetricLBFGS:
    """Raw-coordinate curvature history with a fixed SPD initial operator."""

    def __init__(self, metric: AffineBlockMetric, *, history_size: int, curvature_floor: float) -> None:
        if history_size < 1 or not 0. < curvature_floor < 1.:
            raise ValueError('positive history size and valid relative curvature floor are required')
        self.metric, self.history_size, self.curvature_floor = metric, history_size, curvature_floor
        self.history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def direction(self, gradient: torch.Tensor) -> torch.Tensor:
        residual = gradient.clone()
        coefficients = []
        for step, change, reciprocal in reversed(self.history):
            coefficient = reciprocal*torch.dot(step, residual)
            coefficients.append(coefficient)
            residual.sub_(coefficient*change)
        result = self.metric(residual)
        for (step, change, reciprocal), coefficient in zip(self.history, reversed(coefficients), strict=True):
            result.add_(step*(coefficient-reciprocal*torch.dot(change, result)))
        return -result

    def accept(self, step: torch.Tensor, change: torch.Tensor) -> bool:
        curvature = torch.dot(step, change)
        threshold = self.curvature_floor*torch.linalg.vector_norm(step)*torch.linalg.vector_norm(change)
        if not torch.isfinite(curvature) or not torch.isfinite(threshold):
            raise FloatingPointError('nonfinite accepted L-BFGS curvature')
        if curvature <= threshold:
            return False
        self.history.append((step.clone(), change.clone(), curvature.reciprocal()))
        self.history = self.history[-self.history_size:]
        return True

    def state_dict(self) -> dict:
        return {'history': self.history, 'history_size': self.history_size,
            'curvature_floor': self.curvature_floor, 'metric_identity': self.metric.identity,
            'metric_factors': self.metric.factors}

    def load_state_dict(self, saved: dict) -> None:
        if (saved['history_size'] != self.history_size or saved['curvature_floor'] != self.curvature_floor
            or saved['metric_identity'] != self.metric.identity or len(saved['history']) > self.history_size):
            raise ValueError('metric optimizer history, geometry or source identity changed')
        if not torch.equal(saved['metric_factors'], self.metric.factors):
            raise ValueError('restored affine metric factors differ')
        self.history = deepcopy(saved['history'])


def _flatten_parameters(parameters: tuple[torch.Tensor, ...], *, gradients: bool = False) -> torch.Tensor:
    return torch.cat([(torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).reshape(-1)
        if gradients else parameter.detach().reshape(-1) for parameter in parameters]).clone()


@torch.no_grad()
def _assign_parameters(parameters: tuple[torch.Tensor, ...], values: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        parameter.copy_(values[offset:offset+parameter.numel()].reshape_as(parameter))
        offset += parameter.numel()
    if offset != len(values):
        raise ValueError('parameter-vector geometry changed')


def _assign_gradients(parameters: tuple[torch.Tensor, ...], values: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        parameter.grad = values[offset:offset+parameter.numel()].reshape_as(parameter).clone()
        offset += parameter.numel()


def fit_metric_action(*, policy: AffineDeepAction, problem: AffineAuditProblem, diagnostic: str,
                      data_identity: str, config: MetricSolveConfig = MetricSolveConfig(),
                      optimizer: MetricLBFGS | None = None, resume_state: dict | None = None,
                      initial_gradient_reference: float | None = None,
                      migration: OptimizerMigration | None = None,
                      progress: Progress | None = None,
                      checkpoint: Callable[[AffineDeepAction, MetricLBFGS, dict], None] | None = None) -> dict:
    """Fit the complete original policy using exact full-objective Wolfe steps.

    Accepted-point loss/gradients are reused, not the last line-search trial.
    Raw and affine-metric residuals both enter local-stationarity checks.
    Work limits, rejected trials or zero actions never establish a null pass.
    """
    if not data_identity or diagnostic not in ('squared', 'entropy'):
        raise ValueError('a bound training-data identity and known diagnostic are required')
    if initial_gradient_reference is not None and (not math.isfinite(initial_gradient_reference) or initial_gradient_reference < 0.):
        raise ValueError('a migrated gradient reference must be finite and nonnegative')
    if migration is not None:
        if initial_gradient_reference is not None and initial_gradient_reference != migration.initial_gradient_norm:
            raise ValueError('migration and supplied initial-gradient references differ')
        initial_gradient_reference = migration.initial_gradient_norm
    if policy.config.action_scale != problem.action_scale or policy.config.squared_lambda != problem.squared_lambda:
        raise ValueError('policy and empirical objective scales differ')
    policy.affine_only = False
    policy.requires_grad_(True)
    policy.train()
    parameters = tuple(policy.parameters())
    metric = AffineBlockMetric(policy, problem, deep_scale=config.deep_metric_scale)
    if optimizer is None:
        if resume_state is not None:
            raise ValueError('resuming requires the saved metric optimizer')
        optimizer = MetricLBFGS(metric, history_size=config.neural.history_size, curvature_floor=config.curvature_relative_floor)
    elif (any(actual is not expected for actual, expected in zip(optimizer.metric.parameters, parameters, strict=True))
        or optimizer.metric.identity != metric.identity or optimizer.history_size != config.neural.history_size
        or optimizer.curvature_floor != config.curvature_relative_floor):
        raise ValueError('supplied optimizer does not match this policy and declared metric')
    metric = optimizer.metric
    if migration is not None and resume_state is None and optimizer.history:
        raise ValueError('initial optimizer migration must reset its curvature history')
    identity = {'model_config': asdict(policy.config), 'diagnostic': diagnostic, 'data_identity': data_identity,
        'training_rows': len(problem.rows), 'metric_identity': metric.identity,
        'action_scale': problem.action_scale, 'squared_lambda': problem.squared_lambda,
        'optimizer_migration': None if migration is None else asdict(migration)}
    if resume_state is not None and resume_state['identity'] != identity:
        raise ValueError('metric restart objective, data or model changed')
    forward = torch.compile(policy) if config.neural.compile else policy
    history = [] if resume_state is None else deepcopy(resume_state['history'])
    evaluations = 0 if resume_state is None else resume_state['full_objective_evaluations']

    def evaluate() -> tuple[float, torch.Tensor]:
        nonlocal evaluations
        policy.zero_grad(set_to_none=True)
        loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True, forward=forward))
        gradient = _flatten_parameters(parameters, gradients=True)
        evaluations += 1
        if progress is not None:
            progress({'stage': 'metric_full_gradient_evaluation', 'diagnostic': diagnostic,
                'full_objective_evaluations': evaluations, 'completed_iterations': len(history)})
        return loss, gradient

    loss, gradient = evaluate()
    if resume_state is not None:
        initial_gradient = resume_state['initial_gradient_norm']
        if initial_gradient_reference is not None and initial_gradient_reference != initial_gradient:
            raise ValueError('a saved gradient reference cannot be changed on resume')
    else:
        initial_gradient = float(torch.linalg.vector_norm(gradient)) if initial_gradient_reference is None else initial_gradient_reference
    threshold = config.neural.gradient_absolute_tolerance+config.neural.gradient_relative_tolerance*initial_gradient
    if migration is not None and threshold > migration.absolute_gradient_threshold:
        raise ValueError('optimizer migration cannot relax the original absolute gradient threshold')
    affine_threshold = problem.config.gradient_absolute_tolerance+problem.config.gradient_relative_tolerance*math.sqrt(
        float(problem.residual_energy(problem.cross)))
    first = 0 if resume_state is None else history[-1]['iteration']
    if first > config.neural.maximum_iterations:
        raise ValueError('saved work exceeds the declared iteration limit')
    stop_reason = 'work_limit_not_converged'
    for iteration in range(first, config.neural.maximum_iterations+1):
        norms = metric.diagnostics(gradient)
        if not math.isfinite(loss) or not all(math.isfinite(value) for value in norms.values()):
            raise FloatingPointError('nonfinite accepted full-gradient state')
        converged = norms['full_gradient_norm'] <= threshold and norms['affine_metric_gradient_norm'] <= affine_threshold
        entry = {'iteration': iteration, 'training_loss': loss, **norms, 'gradient_threshold': threshold,
            'affine_metric_gradient_threshold': affine_threshold, 'full_objective_evaluations': evaluations}
        if history and history[-1]['iteration'] == iteration:
            history[-1] = entry
        else:
            history.append(entry)
        state = {'identity': identity, 'history': history, 'initial_gradient_norm': initial_gradient,
            'full_objective_evaluations': evaluations, 'empirical_local_stationarity': converged,
            'train_config': asdict(config), 'global_optimum_certified': False}
        if checkpoint is not None:
            checkpoint(policy, optimizer, state)
        if progress is not None:
            progress({'stage': 'metric_full_gradient_iteration', 'diagnostic': diagnostic, **entry})
        if converged:
            stop_reason = 'full_and_affine_metric_gradient_tolerances'
            break
        if iteration == config.neural.maximum_iterations:
            break
        original = _flatten_parameters(parameters)
        original_loss, original_gradient = loss, gradient
        direction = optimizer.direction(gradient)
        slope = float(torch.dot(gradient, direction))
        if not math.isfinite(slope) or not slope < 0.:
            raise RuntimeError('metric L-BFGS direction is not finite strict descent')
        lower, upper, lower_loss = 0., None, loss
        step = config.neural.initial_learning_rate
        accepted = False
        try:
            for trial in range(config.neural.maximum_line_search_evaluations):
                _assign_parameters(parameters, original+step*direction)
                candidate, candidate_gradient = evaluate()
                finite = math.isfinite(candidate) and bool(torch.isfinite(candidate_gradient).all())
                trial_slope = float(torch.dot(candidate_gradient, direction)) if finite else float('nan')
                finite = finite and math.isfinite(trial_slope)
                armijo = finite and candidate <= original_loss+1.e-4*step*slope
                if not armijo or (lower > 0. and candidate >= lower_loss):
                    upper = step
                elif abs(trial_slope) <= .9*abs(slope):
                    accepted = True
                    break
                elif trial_slope >= 0.:
                    upper = step
                else:
                    lower, lower_loss = step, candidate
                step = .5*(lower+upper) if upper is not None else 2.*step
            if accepted:
                _assign_parameters(parameters, original+step*direction)
                loss, gradient = evaluate()
                increment = _flatten_parameters(parameters)-original
                slope_before = float(torch.dot(original_gradient, increment))
                slope_after = float(torch.dot(gradient, increment))
                armijo = loss <= original_loss+1.e-4*slope_before
                curvature = abs(slope_after) <= .9*abs(slope_before)
                accepted = (math.isfinite(loss) and bool(torch.isfinite(gradient).all())
                    and slope_before < 0. and armijo and curvature)
                if accepted:
                    entry['line_search'] = {'evaluations': trial+1, 'armijo': armijo, 'strong_curvature': curvature,
                        'slope_before': slope_before, 'slope_after': slope_after, 'step_size': step,
                        'curvature_pair_retained': optimizer.accept(increment, gradient-original_gradient)}
        except BaseException:
            _assign_parameters(parameters, original)
            _assign_gradients(parameters, original_gradient)
            raise
        if not accepted:
            _assign_parameters(parameters, original)
            _assign_gradients(parameters, original_gradient)
            loss, gradient = original_loss, original_gradient
            stop_reason = 'line_search_not_converged'
            entry['rejected_step'] = stop_reason
            state['full_objective_evaluations'] = evaluations
            if checkpoint is not None:
                checkpoint(policy, optimizer, state)
            break
    return {'diagnostic': diagnostic, 'history': history, 'stop_reason': stop_reason,
        'converged': stop_reason == 'full_and_affine_metric_gradient_tolerances',
        'global_optimum_certified': False, 'unchanged_original_objective': True,
        'coefficient_directions_discarded': False, 'objective_ridge_added': False,
        'working_memory_batch_size': problem.config.batch_size, 'full_gradient_batch_observations': len(problem.rows),
        'full_objective_evaluations': evaluations, 'validation_used_for_steps_or_stopping': False,
        'optimizer_migration': identity['optimizer_migration']}
