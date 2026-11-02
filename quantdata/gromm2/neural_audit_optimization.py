"""Full-gradient deep-action fitting without stochastic clipping or zero fallback."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction, local_objective


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class NeuralSolveConfig:
    initial_learning_rate: float = 1.
    history_size: int = 30
    maximum_iterations: int = 300
    maximum_line_search_evaluations: int = 30
    gradient_relative_tolerance: float = 1.e-5
    gradient_absolute_tolerance: float = 1.e-9
    compile: bool = False

    def __post_init__(self) -> None:
        verify_inp(math.isfinite(self.initial_learning_rate) and self.initial_learning_rate > 0.,
            lambda: 'the full-gradient line search needs a positive initial learning rate')
        verify_inp(min(self.history_size, self.maximum_iterations, self.maximum_line_search_evaluations) > 0,
            lambda: 'neural solver iteration and history limits must be positive')
        verify_inp(self.gradient_relative_tolerance > 0. and self.gradient_absolute_tolerance > 0.,
            lambda: 'neural solver gradient tolerances must be positive')
        verify_inp(isinstance(self.compile, bool), lambda: 'compile must be boolean')


def full_action_pass(policy: AffineDeepAction, problem: AffineAuditProblem, *, diagnostic: str,
                     backward: bool, forward: Callable | None = None) -> torch.Tensor:
    """One exact full empirical pass; blocks only limit working memory.

    Backward accumulates observation-count-weighted block gradients into one
    complete objective gradient. No optimizer step or clipping occurs here.
    The caller clears parameter gradients once, not once per block.
    """
    parameters = tuple(policy.parameters())
    if not parameters or any(value.dtype != torch.float64 or value.device != problem.states.device for value in parameters):
        raise ValueError('deterministic neural fitting requires float64 parameters on the data device')
    total = torch.zeros((), dtype=torch.float64, device=problem.states.device)
    evaluate = policy if forward is None else forward
    with torch.enable_grad() if backward else torch.no_grad():
        for augmented, returns, mask in problem.blocks():
            action = evaluate(augmented[:, :-1], mask)
            value = local_objective(action, returns, diagnostic=diagnostic,
                action_scale=problem.action_scale, squared_lambda=problem.squared_lambda)
            fraction = len(augmented)/len(problem.rows)
            total.add_(value.detach()*fraction)
            if backward:
                (value*fraction).backward()
    return total


def full_gradient_norm(policy: AffineDeepAction) -> torch.Tensor:
    parameters = tuple(policy.parameters())
    total = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            total.add_(parameter.grad.square().sum())
    return torch.sqrt(total)


def fit_full_gradient_action(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                             diagnostic: str, config: NeuralSolveConfig = NeuralSolveConfig(),
                             optimizer: torch.optim.LBFGS | None = None,
                             resume_state: dict | None = None,
                             data_identity: str = '',
                             progress: Progress | None = None,
                             checkpoint: Callable[[AffineDeepAction, torch.optim.LBFGS, dict], None] | None = None) -> dict:
    """Refine the entire affine-plus-deep class with full-data Wolfe steps.

    This is an empirical local-stationarity check, not a global certificate
    for a nonconvex network. Validation does not drive steps or stopping.
    Exceeding the work limit or merely failing to improve cannot establish
    convergence. A failed/no-op line search is reported as such, not zero.
    """
    if diagnostic not in ('entropy', 'squared'):
        raise ValueError('unknown conditional diagnostic')
    if policy.config.action_scale != problem.action_scale or policy.config.squared_lambda != problem.squared_lambda:
        raise ValueError('neural action and empirical objective scales differ')
    policy.affine_only = False
    policy.requires_grad_(True)
    policy.train()
    forward = torch.compile(policy) if config.compile else policy
    if resume_state is not None and optimizer is None:
        raise ValueError('resuming full-gradient training requires its restored optimizer')
    if optimizer is None:
        optimizer = torch.optim.LBFGS(policy.parameters(), lr=config.initial_learning_rate,
            max_iter=1, max_eval=config.maximum_line_search_evaluations, history_size=config.history_size,
            tolerance_grad=0., tolerance_change=0., line_search_fn='strong_wolfe')
    parameters = tuple(policy.parameters())
    if type(optimizer) is not torch.optim.LBFGS or len(optimizer.param_groups) != 1:
        raise ValueError('full-gradient neural fitting requires one declared LBFGS parameter group')
    group = optimizer.param_groups[0]
    controls = {'lr': config.initial_learning_rate, 'max_iter': 1,
        'max_eval': config.maximum_line_search_evaluations, 'history_size': config.history_size,
        'tolerance_grad': 0., 'tolerance_change': 0., 'line_search_fn': 'strong_wolfe'}
    if (len(group['params']) != len(parameters) or any(actual is not expected
        for actual, expected in zip(group['params'], parameters, strict=True))
        or any(group[key] != value for key, value in controls.items())):
        raise ValueError('restored optimizer parameter identity or declared controls differ')
    identity = {'diagnostic': diagnostic, 'model_config': asdict(policy.config),
        'data_identity': data_identity, 'states_shape': tuple(problem.states.shape),
        'returns_shape': tuple(problem.returns.shape), 'training_rows': len(problem.rows),
        'action_scale': problem.action_scale, 'squared_lambda': problem.squared_lambda}
    if resume_state is not None and resume_state['identity'] != identity:
        raise ValueError('full-gradient restart objective, model or data identity changed')
    history = [] if resume_state is None else deepcopy(resume_state['history'])
    closure_calls = 0 if resume_state is None else resume_state['closure_evaluations']
    line_search_start = None

    class LineSearchBudgetExceeded(RuntimeError):
        pass

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        if line_search_start is not None and closure_calls-line_search_start >= config.maximum_line_search_evaluations:
            raise LineSearchBudgetExceeded('full-objective line-search evaluation budget exhausted')
        optimizer.zero_grad(set_to_none=True)
        value = full_action_pass(policy, problem, diagnostic=diagnostic, backward=True, forward=forward)
        closure_calls += 1
        if progress is not None:
            progress({'stage': 'neural_full_gradient_evaluation', 'diagnostic': diagnostic,
                'closure_evaluations': closure_calls, 'completed_iterations': len(history)})
        return value

    current_loss = float(closure())
    initial_loss = current_loss if resume_state is None else resume_state['initial_loss']
    initial_gradient = float(full_gradient_norm(policy)) if resume_state is None else resume_state['initial_gradient_norm']
    threshold = config.gradient_absolute_tolerance+config.gradient_relative_tolerance*initial_gradient
    first_iteration = 0 if resume_state is None else history[-1]['iteration']
    if first_iteration > config.maximum_iterations:
        raise ValueError('restored neural work exceeds the requested iteration limit')
    previous_loss = current_loss if resume_state is None else history[-1]['training_loss']
    stop_reason = 'work_limit_not_converged'
    for iteration in range(first_iteration, config.maximum_iterations+1):
        loss = current_loss if iteration == first_iteration else float(closure())
        norm = float(full_gradient_norm(policy))
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError('nonfinite full-objective neural loss or gradient')
        if loss > previous_loss+1.e-11*(1.+abs(previous_loss)):
            raise RuntimeError('full-gradient Wolfe step increased the actual training objective')
        entry = {'iteration': iteration, 'training_loss': loss, 'full_gradient_norm': norm,
            'gradient_threshold': threshold, 'closure_evaluations': closure_calls}
        if history and history[-1]['iteration'] == iteration:
            history[-1] = entry
        else:
            history.append(entry)
        if progress is not None:
            progress({'stage': 'neural_full_gradient_iteration', 'diagnostic': diagnostic, **entry})
        state = {'history': history, 'empirical_local_stationarity': norm <= threshold,
            'global_optimum_certified': False, 'initial_loss': initial_loss, 'initial_gradient_norm': initial_gradient,
            'gradient_threshold': threshold, 'closure_evaluations': closure_calls,
            'identity': identity, 'train_config': asdict(config)}
        if checkpoint is not None:
            checkpoint(policy, optimizer, state)
        if norm <= threshold:
            stop_reason = 'full_empirical_gradient_tolerance'
            break
        if iteration == config.maximum_iterations:
            break
        previous_loss = loss
        previous_parameters = [parameter.detach().clone() for parameter in parameters]
        previous_gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
        previous_optimizer = deepcopy(optimizer.state_dict())
        line_search_start = closure_calls
        rejected = None
        try:
            optimizer.step(closure)
        except LineSearchBudgetExceeded:
            rejected = 'line_search_evaluation_limit_not_converged'
        finally:
            line_search_start = None
        if rejected is None:
            proposed_loss = float(closure())
            increments = [parameter.detach()-previous for parameter, previous in zip(parameters, previous_parameters, strict=True)]
            slope_before = sum(float((gradient*increment).sum()) for gradient, increment in zip(previous_gradients, increments, strict=True) if gradient is not None)
            slope_after = sum(float((parameter.grad*increment).sum()) for parameter, increment in zip(parameters, increments, strict=True) if parameter.grad is not None)
            changed = any(bool(increment.ne(0.).any()) for increment in increments)
            armijo = proposed_loss <= loss+1.e-4*slope_before
            curvature = abs(slope_after) <= .9*abs(slope_before)
            entry['line_search'] = {'evaluations': closure_calls-state['closure_evaluations'],
                'armijo': armijo, 'strong_curvature': curvature, 'parameters_changed': changed}
            if not changed:
                rejected = 'line_search_no_step_not_converged'
            elif not (slope_before < 0. and armijo and curvature):
                rejected = 'line_search_conditions_not_met_not_converged'
        if rejected is not None:
            with torch.no_grad():
                for parameter, previous, gradient in zip(parameters, previous_parameters, previous_gradients, strict=True):
                    parameter.copy_(previous)
                    parameter.grad = gradient
            optimizer.load_state_dict(previous_optimizer)
            stop_reason = rejected
            entry['rejected_step'] = rejected
            state['closure_evaluations'] = closure_calls
            if checkpoint is not None:
                checkpoint(policy, optimizer, state)
            break
    return {'diagnostic': diagnostic, 'history': history, 'stop_reason': stop_reason,
        'converged': stop_reason == 'full_empirical_gradient_tolerance',
        'global_optimum_certified': False, 'validation_used_for_steps_or_stopping': False,
        'gradient_clipping': False, 'full_gradient_batch_observations': len(problem.rows),
        'working_memory_batch_size': problem.config.batch_size, 'initial_learning_rate': config.initial_learning_rate,
        'learning_rate_rule': 'strong-Wolfe full-objective line search', 'unchanged_original_objective': True}
