"""Read-only strict-gradient verification with explicit optimizer-phase routing."""

from __future__ import annotations

import hashlib
import io
import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem
from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_continuation import (
    GGNContinuationConfig,
    GGNStoppingReference,
    action_point_sha256,
    continue_full_ggn_action,
    optimizer_state_sha256,
)
from quantdata.gromm2.ggn_audit_finalization import (
    _check_native_optimizer,
    _checked_bytes,
    _native_point_sha256,
)
from quantdata.gromm2.ggn_audit_optimization import GGNStepConfig
from quantdata.gromm2.gradient_reliable_continuation import (
    ReliableContinuationConfig,
    continue_gradient_reliable_action,
)
from quantdata.gromm2.gradient_reliable_ggn import GradientReliabilityConfig, verify_actual_metric
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS
from quantdata.gromm2.preconditioned_reliable_continuation import (
    PreconditionedReliableConfig,
    continue_preconditioned_reliable_action,
)

PHASES = {'gromm2-serial-strict-GGN-v1': 'ggn_continuation_state',
    'gromm2-gradient-reliable-GGN-v1': 'reliable_continuation_state',
    'gromm2-history-preconditioned-reliable-GGN-v1': 'preconditioned_reliable_state'}


def native_curvature_passes(native: dict) -> int:
    """Use the active phase, not retained old state keys or history sums."""
    phase = native.get('optimizer_phase_schema')
    if phase in PHASES:
        count = native[PHASES[phase]]['full_curvature_passes']
        if native['state'].get('full_curvature_evaluations', count) != count:
            raise ValueError('active native curvature counters differ')
    else:
        count = native['state'].get('full_curvature_evaluations',
            sum(entry.get('full_curvature_passes', 0) for entry in native['state']['history']))
    if type(count) is not int or count < 0:
        raise ValueError('native cumulative curvature work must be a nonnegative integer')
    return count


def verify_strict_action(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                         optimizer: MetricLBFGS, run_plan: Path,
                         run_plan_sha256: str, accepted_checkpoint: Path,
                         accepted_checkpoint_sha256: str) -> dict:
    """Recompute one full canonical gradient without optimizing or exporting.

    The original plan-bound fold, inner mask and normalization must be loaded
    by the caller. No fresh returns or opposite-fold arrays are opened. This
    separate receipt does not rewrite any native phase, loss history, raw
    optimizer state, model or normalization, and is not a four-policy freeze.
    """
    plan = json.loads(_checked_bytes(run_plan, run_plan_sha256))
    saved = torch.load(io.BytesIO(_checked_bytes(accepted_checkpoint, accepted_checkpoint_sha256)),
        map_location='cpu', weights_only=False)
    phase = saved.get('optimizer_phase_schema')
    if phase not in PHASES or phase != plan['schema'] or not plan['original_thresholds_unchanged']:
        raise ValueError('strict verification requires an explicit matching active optimizer phase')
    verify_actual_metric(optimizer.metric, policy)
    for filename, expected in plan['implementation_sha256'].items():
        _checked_bytes(Path(filename), expected)
    parent = torch.load(io.BytesIO(_checked_bytes(Path(plan['source_native']), plan['source_native_sha256'])),
        map_location='cpu', weights_only=False)
    state = saved[PHASES[phase]]
    if (not state['converged'] or not state['history']
        or state['stop_reason'] != 'original_raw_and_affine_metric_gradient_tolerances'
        or state['validation_used_for_steps_or_stopping'] or state.get('rejected_work') is not None):
        raise ValueError('active reliable action has not passed the original strict gates')
    baseline = parent['state']['history'][-1]
    reference = GGNStoppingReference(**plan['reference'])
    expected_reference = GGNStoppingReference(source_checkpoint_sha256=plan['source_native_sha256'],
        initial_gradient_norm=parent['state']['initial_gradient_norm'],
        raw_threshold=baseline['gradient_threshold'], affine_metric_threshold=baseline['affine_metric_gradient_threshold'],
        source_accepted_updates=baseline['iteration'], prior_full_objective_passes=parent['state']['full_objective_evaluations'],
        prior_full_curvature_passes=native_curvature_passes(parent))
    if reference != expected_reference or saved['state']['initial_gradient_norm'] != reference.initial_gradient_norm:
        raise ValueError('strict verification cannot rebase original gates, reference or source work')
    expected_history = deepcopy(parent['state']['history'])+deepcopy(state['history'])
    expected_history[-1]['full_objective_evaluations'] = state['full_objective_passes']
    if (saved['identity'] != parent['identity'] or saved['state']['identity'] != parent['state']['identity']
        or saved['state']['history'] != expected_history
        or saved['state']['full_objective_evaluations'] != state['full_objective_passes']
        or saved['state']['empirical_local_stationarity'] is not True
        or native_curvature_passes(saved) != state['full_curvature_passes']
        or state['identity']['diagnostic'] != parent['state']['identity']['diagnostic']):
        raise ValueError('strict native identity, objective, accepted history or active work changed')
    if any(saved.get(key) != parent[key] for key in PHASES.values() if key != PHASES[phase] and key in parent):
        raise ValueError('strict retained preceding optimizer-phase record changed')
    for name in ('torch_rng', 'cuda_rng'):
        if name in parent and (name not in saved or (saved[name] is None) != (parent[name] is None)
            or parent[name] is not None and not torch.equal(saved[name].cpu(), parent[name].cpu())):
            raise ValueError('strict deterministic continuation changed its native random state')
    _check_native_optimizer(saved['optimizer'], optimizer)
    if (hashlib.sha256(Path(plan['normalization']).read_bytes()).hexdigest() != parent['identity']['normalization_sha256']
        or optimizer.metric.identity != parent['optimizer']['metric_identity']):
        raise ValueError('strict original normalization or metric changed')
    source_point = {name: baseline[name] for name in ('training_loss', 'full_gradient_norm', 'affine_metric_gradient_norm')}
    if source_point != plan['source_point']:
        raise ValueError('strict source point differs from its original plan')
    source_point['point_sha256'] = _native_point_sha256(parent, policy)
    point_sha = action_point_sha256(policy)
    expected_phase = {'gromm2-serial-strict-GGN-v1': 'coupled-GGN-canonical-Armijo',
        'gromm2-gradient-reliable-GGN-v1': 'GGN-gradient-reliable-canonical-Armijo',
        'gromm2-history-preconditioned-reliable-GGN-v1': 'GGN-LBFGS-preconditioned-gradient-reliable-canonical-Armijo'}[phase]
    if (point_sha != state['point_sha256'] or point_sha != _native_point_sha256(saved, policy)
        or any(entry.get('optimizer_phase') != expected_phase
            or entry.get('acceptance', {}).get('accepted') is not True for entry in state['history'])):
        raise ValueError('strict actual parameter point or active phase history changed')
    settings = dict(plan['config'])
    settings['step'] = GGNStepConfig(**settings['step'])
    if phase != 'gromm2-serial-strict-GGN-v1':
        settings['acceptance'] = GradientReliabilityConfig(**settings['acceptance'])
        if phase == 'gromm2-history-preconditioned-reliable-GGN-v1':
            declared = PreconditionedReliableConfig(**settings)
            continuation = continue_preconditioned_reliable_action
        else:
            declared = ReliableContinuationConfig(**settings)
            continuation = continue_gradient_reliable_action
    else:
        declared = GGNContinuationConfig(**settings)
        continuation = continue_full_ggn_action
    if len(state['history']) > declared.maximum_accepted_steps:
        raise ValueError('strict accepted work exceeds the declared window')
    verification_config = replace(declared, maximum_accepted_steps=len(state['history']))
    original_state = deepcopy(state)
    optimizer_sha = optimizer_state_sha256(optimizer)
    gradients = [None if value.grad is None else value.grad.detach().clone() for value in policy.parameters()]
    devices = [] if next(policy.parameters()).device.type == 'cpu' else [next(policy.parameters()).device]
    try:
        with torch.random.fork_rng(devices=devices):
            verified = continuation(policy=policy, problem=problem, optimizer=optimizer,
                diagnostic=state['identity']['diagnostic'], data_identity=parent['state']['identity']['data_identity'],
                reference=reference, source_point=source_point, config=verification_config, resume_state=state)
    finally:
        for parameter, gradient in zip(policy.parameters(), gradients, strict=True):
            parameter.grad = gradient
    norms = verified['current_norms']
    if (not verified['converged'] or not math.isfinite(verified['current_training_loss'])
        or not verified['current_training_loss'] < 0. or not all(math.isfinite(value) for value in norms.values())
        or norms['full_gradient_norm'] > reference.raw_threshold
        or norms['affine_metric_gradient_norm'] > reference.affine_metric_threshold):
        raise ValueError('actual strict verification failed a full gradient gate or negative loss')
    if (state != original_state or verified['history'] != state['history']
        or verified['full_objective_passes'] != state['full_objective_passes']+1
        or verified['full_curvature_passes'] != state['full_curvature_passes']
        or verified.get('invocation_ggn_passes', 0) != 0 or verified.get('invocation_exact_hessian_passes', 0) != 0
        or action_point_sha256(policy) != point_sha or optimizer_state_sha256(optimizer) != optimizer_sha):
        raise RuntimeError('strict verification changed accepted state or performed optimization work')
    return {'schema': 'gromm2-strict-active-phase-verification-v1', 'active_optimizer_phase': phase,
        'run_plan': str(run_plan.resolve()), 'run_plan_sha256': run_plan_sha256,
        'accepted_checkpoint': str(accepted_checkpoint.resolve()), 'accepted_checkpoint_sha256': accepted_checkpoint_sha256,
        'normalization': plan['normalization'], 'normalization_sha256': saved['identity']['normalization_sha256'],
        'fold': saved['identity']['fold'], 'diagnostic': state['identity']['diagnostic'],
        'continuation_state_sha256': hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'original_reference': asdict(reference), 'point_sha256': point_sha, 'optimizer_state_sha256': optimizer_sha,
        'training_loss': verified['current_training_loss'], **norms,
        'verification_full_objective_passes': 1, 'additional_curvature_passes': 0,
        'cumulative_full_objective_passes': verified['full_objective_passes'],
        'cumulative_full_curvature_passes': verified['full_curvature_passes'],
        'accepted_history_preserved': True, 'original_thresholds_met': True,
        'validation_or_holdout_used': False, 'optimization_performed': False,
        'global_optimum_certified': False, 'four_policy_freeze_still_required': True}
