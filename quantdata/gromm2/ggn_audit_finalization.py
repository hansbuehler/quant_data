"""Verify accepted GGN actions without optimizing or rewriting phase history."""

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
from quantdata.gromm2.ggn_audit_optimization import GGNStepConfig
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS
from quantdata.torch import BaseEnvironment


def _checked_bytes(path: Path, expected_sha256: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f'finalization source changed: {path}')
    return content


def _native_point_sha256(saved: dict, policy: AffineDeepAction) -> str:
    if (saved['code_version'] != policy.config.version.unique_id48
        or saved['user_version'] != policy.config.user_version
        or saved['state']['identity']['model_config'] != asdict(policy.config)):
        raise ValueError('finalization native model version or geometry changed')
    with torch.random.fork_rng(devices=[]):
        restored = AffineDeepAction(BaseEnvironment(device='cpu', precision=64), config=policy.config)
        restored.load_state_dict(saved['model'], strict=True)
    return action_point_sha256(restored)


def _check_native_optimizer(saved: dict, optimizer: MetricLBFGS) -> None:
    actual = optimizer.state_dict()
    if (set(saved) != set(actual)
        or any(saved[name] != actual[name] for name in ('history_size', 'curvature_floor', 'metric_identity'))
        or len(saved['history']) != len(actual['history'])):
        raise ValueError('finalization native optimizer controls or raw history changed')
    pairs = [(saved['metric_factors'], actual['metric_factors'])]
    for retained, current in zip(saved['history'], actual['history'], strict=True):
        if len(retained) != len(current):
            raise ValueError('finalization native raw optimizer pair geometry changed')
        pairs.extend(zip(retained, current, strict=True))
    if any(expected.shape != current.shape or expected.dtype != current.dtype
        or expected.detach().cpu().contiguous().numpy().tobytes() != current.detach().cpu().contiguous().numpy().tobytes()
        for expected, current in pairs):
        raise ValueError('finalization native metric factors or raw optimizer pairs changed')


def verify_accepted_ggn_action(*, policy: AffineDeepAction, problem: AffineAuditProblem,
                               optimizer: MetricLBFGS, run_plan: Path,
                               run_plan_sha256: str, accepted_checkpoint: Path,
                               accepted_checkpoint_sha256: str) -> dict:
    """Re-evaluate one strictly converged native action; never take a step.

    The caller must load the original plan-bound training population using its
    original fold, inner-training mask and normalization. This function checks
    their saved identities and complete canonical gradient, not a holdout. It
    opens only declared plans/checkpoints and pinned implementation artifacts;
    it does not read fresh evaluation or opposite-fold return arrays.

    The returned receipt is separate from the immutable native state. All four
    independent policies still need verification before a phase-aware freezer
    may publish actions for independent confirmation.
    """
    plan = json.loads(_checked_bytes(run_plan, run_plan_sha256))
    if plan['schema'] != 'gromm2-serial-strict-GGN-v1' or not plan['original_thresholds_unchanged']:
        raise ValueError('finalization requires the declared original strict GGN protocol')
    for filename, expected in plan['implementation_sha256'].items():
        _checked_bytes(Path(filename), expected)
    parent = torch.load(io.BytesIO(_checked_bytes(Path(plan['source_native']), plan['source_native_sha256'])),
        map_location='cpu', weights_only=False)
    saved = torch.load(io.BytesIO(_checked_bytes(accepted_checkpoint, accepted_checkpoint_sha256)),
        map_location='cpu', weights_only=False)
    state = saved['ggn_continuation_state']
    if (not state['converged'] or state['stop_reason'] != 'original_raw_and_affine_metric_gradient_tolerances'
        or not state['history'] or state['validation_used_for_steps_or_stopping']):
        raise ValueError('accepted GGN action has not passed the original strict gates')
    reference = GGNStoppingReference(**plan['reference'])
    baseline = parent['state']['history'][-1]
    expected_reference = GGNStoppingReference(source_checkpoint_sha256=plan['source_native_sha256'],
        initial_gradient_norm=parent['state']['initial_gradient_norm'],
        raw_threshold=baseline['gradient_threshold'],
        affine_metric_threshold=baseline['affine_metric_gradient_threshold'],
        source_accepted_updates=baseline['iteration'],
        prior_full_objective_passes=parent['state']['full_objective_evaluations'],
        prior_full_curvature_passes=sum(entry.get('full_curvature_passes', 0) for entry in parent['state']['history']))
    if reference != expected_reference or saved['state']['initial_gradient_norm'] != reference.initial_gradient_norm:
        raise ValueError('finalization cannot rebase the original gradient thresholds or work')
    expected_history = deepcopy(parent['state']['history'])+deepcopy(state['history'])
    expected_history[-1]['full_objective_evaluations'] = state['full_objective_passes']
    if (saved['identity'] != parent['identity'] or saved['state']['identity'] != parent['state']['identity']
        or saved['state']['history'] != expected_history
        or saved['state']['full_objective_evaluations'] != state['full_objective_passes']
        or saved['state']['empirical_local_stationarity'] is not True):
        raise ValueError('finalization native source, fold, original history or committed work changed')
    if state['identity']['diagnostic'] != parent['state']['identity']['diagnostic']:
        raise ValueError('finalization diagnostic differs from its original source policy objective')
    _check_native_optimizer(saved['optimizer'], optimizer)
    if (hashlib.sha256(Path(plan['normalization']).read_bytes()).hexdigest() != parent['identity']['normalization_sha256']
        or optimizer.metric.identity != parent['optimizer']['metric_identity']):
        raise ValueError('finalization normalization or fixed metric changed')
    source_point = {name: baseline[name] for name in
        ('training_loss', 'full_gradient_norm', 'affine_metric_gradient_norm')}
    if source_point != plan['source_point']:
        raise ValueError('finalization source point differs from its immutable plan')
    source_point['point_sha256'] = _native_point_sha256(parent, policy)
    point_sha = action_point_sha256(policy)
    if (point_sha != state['point_sha256'] or point_sha != _native_point_sha256(saved, policy)
        or any(entry.get('optimizer_phase') != 'coupled-GGN-canonical-Armijo'
            or entry.get('acceptance', {}).get('accepted') is not True for entry in state['history'])):
        raise ValueError('finalization accepted parameter point or GGN phase history changed')
    settings = dict(plan['config'])
    settings['step'] = GGNStepConfig(**settings['step'])
    declared = GGNContinuationConfig(**settings)
    if len(state['history']) > declared.maximum_accepted_steps:
        raise ValueError('accepted GGN work exceeds its declared window')
    verification_config = replace(declared, maximum_accepted_steps=len(state['history']))
    optimizer_sha = optimizer_state_sha256(optimizer)
    original_state = deepcopy(state)
    gradients = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in policy.parameters()]
    devices = [] if next(policy.parameters()).device.type == 'cpu' else [next(policy.parameters()).device]
    try:
        with torch.random.fork_rng(devices=devices):
            verified = continue_full_ggn_action(policy=policy, problem=problem, optimizer=optimizer,
                diagnostic=state['identity']['diagnostic'], data_identity=parent['state']['identity']['data_identity'],
                reference=reference, source_point=source_point, config=verification_config, resume_state=state)
    finally:
        for parameter, gradient in zip(policy.parameters(), gradients, strict=True):
            parameter.grad = gradient
    norms = verified['current_norms']
    if (not verified['converged'] or not math.isfinite(verified['current_training_loss'])
        or not verified['current_training_loss'] < 0.
        or not all(math.isfinite(value) for value in norms.values())
        or not norms['full_gradient_norm'] <= reference.raw_threshold
        or not norms['affine_metric_gradient_norm'] <= reference.affine_metric_threshold):
        raise ValueError('actual full training verification failed an original strict gate or negative loss')
    if (verified['history'] != original_state['history'] or state != original_state
        or verified['full_objective_passes'] != state['full_objective_passes']+1
        or verified['full_curvature_passes'] != state['full_curvature_passes']
        or action_point_sha256(policy) != point_sha or optimizer_state_sha256(optimizer) != optimizer_sha):
        raise RuntimeError('verification altered accepted native state or took optimization work')
    return {'schema': 'gromm2-strict-GGN-verification-v1',
        'run_plan': str(run_plan.resolve()), 'run_plan_sha256': run_plan_sha256,
        'accepted_checkpoint': str(accepted_checkpoint.resolve()),
        'accepted_checkpoint_sha256': accepted_checkpoint_sha256,
        'continuation_state_sha256': hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'original_reference': asdict(reference), 'point_sha256': point_sha,
        'optimizer_state_sha256': optimizer_sha, 'training_loss': verified['current_training_loss'], **norms,
        'verification_full_objective_passes': 1, 'additional_curvature_passes': 0,
        'cumulative_full_objective_passes': verified['full_objective_passes'],
        'cumulative_full_curvature_passes': verified['full_curvature_passes'],
        'accepted_history_preserved': True, 'original_thresholds_met': True,
        'validation_or_holdout_used': False, 'optimization_performed': False,
        'global_optimum_certified': False, 'four_policy_freeze_still_required': True}
