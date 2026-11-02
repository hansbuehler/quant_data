"""Explicit inner-validation selection for unchanged full-class action fitting.

This is an alternative selection protocol, not a relaxation of a numerical
convergence certificate. Independent confirmation remains necessary. The
convergence-only optimizer and its historical checkpoints are left unchanged.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, Progress
from quantdata.gromm2.conditional_audit import AffineDeepAction, LocalActionConfig
from quantdata.gromm2.full_gradient_audit import _atomic_torch_save
from quantdata.gromm2.metric_audit_optimization import (
    AffineBlockMetric,
    MetricLBFGS,
    MetricSolveConfig,
    fit_metric_action,
)

SELECTION_SOURCE = 'inner-validation selected genuinely trained full-class state'


@version('0.1')
@dataclass(frozen=True, kw_only=True)
class ValidationSelectionConfig:
    """Declared accepted-update cadence and patience, never a convergence claim.

    Improvement thresholds use log-mean-exponential units for entropy and
    normalized squared-loss units for squared. Exact best-checkpoint ranking
    is separate from the minimum improvement used to reset patience.
    """

    interval: int = 5
    patience: int = 10
    minimum_updates: int = 1
    improvement_relative: float = 1.e-4
    improvement_absolute: float = 1.e-8

    def __post_init__(self) -> None:
        verify_inp(all(type(value) is int and value > 0
            for value in (self.interval, self.patience, self.minimum_updates)),
            lambda: 'validation interval, patience and minimum updates must be positive integers')
        verify_inp(all(math.isfinite(value) and value >= 0.
            for value in (self.improvement_relative, self.improvement_absolute)),
            lambda: 'validation improvement thresholds must be finite and nonnegative')


@torch.no_grad()
def validation_action_value(policy: AffineDeepAction, problem: AffineAuditProblem, *, diagnostic: str) -> dict:
    """Evaluate every validation row without changing parameters, gradients or RNG.

    Entropy uses log(mean(exp(exponent))), exactly monotone in the original
    exponential objective. The log-domain reduction avoids overflow without
    clipping, dropping or reweighting any observation. Squared loss retains
    the original deterministic action-scale normalization.
    """
    if diagnostic not in ('squared', 'entropy'):
        raise ValueError('unknown validation diagnostic')
    if (policy.config.action_scale != problem.action_scale
        or policy.config.squared_lambda != problem.squared_lambda):
        raise ValueError('validation policy and empirical objective scales differ')
    total = torch.zeros((), dtype=torch.float64, device=problem.states.device)
    squared_action = torch.zeros_like(total)
    log_total = torch.full_like(total, -torch.inf)
    finite = torch.ones((), dtype=torch.bool, device=total.device)
    scale_squared = problem.action_scale**2
    for augmented, returns, eligible in problem.blocks():
        action = policy(augmented[:, :-1], eligible)
        gain = (action*returns).sum(dim=-1)
        norm_squared = action.square().sum(dim=-1)
        finite.logical_and_(torch.isfinite(action).all() & torch.isfinite(gain).all()
            & torch.isfinite(norm_squared).all())
        squared_action.add_(norm_squared.sum())
        if diagnostic == 'squared':
            total.add_((-gain+.5*norm_squared+.5*problem.squared_lambda*scale_squared*gain.square()).sum())
        else:
            exponent = scale_squared*(-gain+.5*norm_squared)
            log_total = torch.logaddexp(log_total, torch.logsumexp(exponent, dim=0))
    value = float(total/len(problem.rows) if diagnostic == 'squared' else log_total-math.log(len(problem.rows)))
    norm = float(squared_action/len(problem.rows))
    if not bool(finite) or not math.isfinite(value) or not math.isfinite(norm):
        raise FloatingPointError('nonfinite complete inner-validation objective or action')
    return {'value': value, 'mean_squared_unit_action': norm,
        'units': 'original normalized squared loss' if diagnostic == 'squared' else 'log exponential loss',
        'observations': len(problem.rows), 'observations_dropped': 0}


def _validate_separation(training: AffineAuditProblem, validation: AffineAuditProblem) -> None:
    if any(getattr(training, name) is not getattr(validation, name) for name in ('states', 'returns', 'eligible')):
        raise ValueError('selection requires explicit disjoint views of one development population')
    if training.action_scale != validation.action_scale or training.squared_lambda != validation.squared_lambda:
        raise ValueError('training and validation objective scales differ')
    used = torch.zeros(len(training.states), dtype=torch.bool, device=training.states.device)
    used[training.rows] = True
    if bool(used[validation.rows].any()):
        raise ValueError('inner training and validation rows overlap')


def fit_validation_selected_action(*, policy: AffineDeepAction, training: AffineAuditProblem,
                                   validation: AffineAuditProblem, diagnostic: str,
                                   data_identity: str, output: Path,
                                   optimizer_config: MetricSolveConfig = MetricSolveConfig(),
                                   selection_config: ValidationSelectionConfig = ValidationSelectionConfig(),
                                   progress: Progress | None = None) -> dict:
    """Fit the original full policy, select validation weights, retain native resume.

    This function must only be used under an explicitly agreed selection
    protocol. Both data views are development data, never confirmation. The
    selected state and terminal optimizer state have distinct files and labels;
    a work-limit exit is never relabelled numerical convergence. Literal zero
    and initial affine weights are not selection candidates. No policy cap,
    objective ridge, gradient clipping, return centering or Q reweighting is used.
    """
    _validate_separation(training, validation)
    if not data_identity or diagnostic not in ('squared', 'entropy'):
        raise ValueError('bound development-data identity and known diagnostic are required')
    if optimizer_config.neural.maximum_iterations < selection_config.minimum_updates:
        raise ValueError('optimizer work limit cannot precede eligible training work')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output/'resume.pt'
    identity = {'schema': 'gromm2-validation-selection-v1', 'data_identity': data_identity,
        'model_config': asdict(policy.config), 'diagnostic': diagnostic,
        'training_rows_sha256': hashlib.sha256(training.rows.cpu().contiguous().numpy().tobytes()).hexdigest(),
        'validation_rows_sha256': hashlib.sha256(validation.rows.cpu().contiguous().numpy().tobytes()).hexdigest(),
        'optimizer_config': asdict(optimizer_config), 'selection_config': asdict(selection_config),
        'code_version': LocalActionConfig.version.unique_id48,
        'selection_version': ValidationSelectionConfig.version.unique_id48,
        'user_version': policy.config.user_version,
        'implementation_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *(Path(__file__).with_name(name+'.py') for name in (
                'metric_audit_optimization', 'neural_audit_optimization', 'conditional_audit',
                'affine_audit_optimization', 'full_gradient_audit'))]}}
    selection = {'history': [], 'best_value': None, 'best_iteration': None, 'best_sha256': None,
        'patience_anchor': None, 'stale_checks': 0, 'last_checked_iteration': None,
        'initial_validation': None, 'stop_reason': None, 'best_filename': None}
    native = None
    optimizer = None
    last_state = None

    class ValidationStop(Exception):
        pass

    with FileLock(str(output/'fit.lock'), timeout=0):
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location=training.states.device, weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('validation-selected checkpoint identity or declared controls changed')
            policy.load_state_dict(saved['model'], strict=True)
            optimizer = MetricLBFGS(AffineBlockMetric(policy, training, deep_scale=optimizer_config.deep_metric_scale),
                history_size=optimizer_config.neural.history_size, curvature_floor=optimizer_config.curvature_relative_floor)
            optimizer.load_state_dict(saved['optimizer'])
            native, selection = saved['native_state'], saved['selection']
            if selection['best_sha256'] is not None:
                best_path = output/selection['best_filename']
                if not best_path.exists() or hashlib.sha256(best_path.read_bytes()).hexdigest() != selection['best_sha256']:
                    raise ValueError('selected action differs from its accepted resume ledger')
            torch.set_rng_state(saved['torch_rng'].cpu())
            if policy.device.type == 'cuda' and saved['cuda_rng'] is not None:
                torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        else:
            selection['initial_validation'] = validation_action_value(policy, validation, diagnostic=diagnostic)

        def persist(model: AffineDeepAction, current_optimizer: MetricLBFGS, state: dict) -> None:
            nonlocal last_state
            iteration = state['history'][-1]['iteration']
            event = None
            due = iteration >= selection_config.minimum_updates and (
                iteration == selection_config.minimum_updates or iteration % selection_config.interval == 0
                or iteration == optimizer_config.neural.maximum_iterations or state['empirical_local_stationarity'])
            if due and selection['last_checked_iteration'] != iteration:
                observed = validation_action_value(model, validation, diagnostic=diagnostic)
                record = {'iteration': iteration, **observed,
                    'training_loss': state['history'][-1]['training_loss'],
                    'empirical_local_stationarity': state['empirical_local_stationarity']}
                selection['history'].append(record)
                selection['last_checked_iteration'] = iteration
                eligible = observed['mean_squared_unit_action'] > 0.
                improved = eligible and (selection['best_value'] is None or observed['value'] < selection['best_value'])
                if improved:
                    best_path = output/f'selected-{iteration:06d}-{uuid4().hex}.pt'
                    if best_path.exists():
                        raise ValueError('uncommitted selection candidate exists; preserve and investigate it')
                    _atomic_torch_save(best_path, {'identity': identity, 'model_config': asdict(model.config),
                        'code_version': LocalActionConfig.version.unique_id48, 'user_version': model.config.user_version,
                        'state_dict': model.state_dict(), 'diagnostic': diagnostic, 'selected': record,
                        'input_normalization_dtype': 'float32', 'policy_state_source': SELECTION_SOURCE})
                    selection.update(best_value=observed['value'], best_iteration=iteration,
                        best_filename=best_path.name, best_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest())
                anchor = selection['patience_anchor']
                margin = selection_config.improvement_absolute+selection_config.improvement_relative*abs(anchor or 0.)
                meaningful = eligible and (anchor is None or observed['value'] < anchor-margin)
                if meaningful:
                    selection.update(patience_anchor=observed['value'], stale_checks=0)
                else:
                    selection['stale_checks'] += 1
                event = {'stage': 'inner_validation_selection', 'diagnostic': diagnostic, **record,
                    'selected_iteration': selection['best_iteration'], 'stale_checks': selection['stale_checks']}
                if selection['stale_checks'] >= selection_config.patience:
                    selection['stop_reason'] = 'declared_inner_validation_patience'
            last_state = deepcopy(state)
            _atomic_torch_save(checkpoint_path, {'identity': identity, 'model': model.state_dict(),
                'optimizer': current_optimizer.state_dict(), 'native_state': state, 'selection': selection,
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state() if model.device.type == 'cuda' else None})
            if event is not None and progress is not None:
                progress(event)
            if selection['stop_reason'] == 'declared_inner_validation_patience':
                raise ValidationStop

        try:
            numerical = fit_metric_action(policy=policy, problem=training, diagnostic=diagnostic,
                data_identity=data_identity, config=optimizer_config, optimizer=optimizer,
                resume_state=native, checkpoint=persist, progress=progress)
        except ValidationStop:
            numerical = {'stop_reason': 'validation_selected_not_a_stationarity_certificate',
                'converged': bool(last_state['empirical_local_stationarity']), 'history': last_state['history'],
                'full_objective_evaluations': last_state['full_objective_evaluations']}
        if selection['best_sha256'] is None:
            raise RuntimeError('no genuinely trained nonzero full-class validation candidate; no zero fallback')
        if numerical['stop_reason'] == 'line_search_not_converged':
            raise RuntimeError('line search failed; preserve the checkpoint, do not certify a selected fit')
        numerical['validation_used_for_steps_or_stopping'] = selection['stop_reason'] is not None
        numerical['validation_used_for_steps'] = False
        numerical['validation_used_for_stopping'] = selection['stop_reason'] is not None
        best_path = output/selection['best_filename']
        selected = torch.load(best_path, map_location=training.states.device, weights_only=False)
        policy.load_state_dict(selected['state_dict'], strict=True)
        return {'identity': identity, 'selection': deepcopy(selection), 'numerical': numerical,
            'selected_weights': str(best_path.resolve()), 'selected_weights_sha256': selection['best_sha256'],
            'policy_state_source': SELECTION_SOURCE, 'inner_validation_used_for_selection': True,
            'confirmation_read': False, 'global_optimum_certified': False,
            'numerical_stationarity_required_for_inference': False}
