from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from quantdata.gromm2 import ggn_audit_continuation, ggn_audit_finalization
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.ggn_audit_continuation import (
    GGNContinuationConfig,
    GGNStoppingReference,
    action_point_sha256,
    continue_full_ggn_action,
    optimizer_state_sha256,
)
from quantdata.gromm2.ggn_audit_finalization import verify_accepted_ggn_action
from quantdata.gromm2.ggn_audit_optimization import GGNStepConfig, _flatten
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.torch import BaseEnvironment


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, diagnostic='squared'):
    states = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float32).repeat(2, 1)
    problem = AffineAuditProblem(states=states, returns=torch.tensor([[.1, .2]], dtype=torch.float32).repeat(8, 1),
        eligible=torch.ones((2, 2), dtype=torch.bool), rows=torch.arange(8), action_scale=.2,
        config=AffineSolveConfig(batch_size=3))
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='strict-ggn-finalization-test', feature_count=2,
            instrument_count=2, action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    policy.zero_grad(set_to_none=True)
    loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True))
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.2),
        history_size=5, curvature_floor=1.e-12)
    norms = optimizer.metric.diagnostics(gradient)
    baseline = {'iteration': 0, 'training_loss': loss, **norms,
        'gradient_threshold': 1.e-7, 'affine_metric_gradient_threshold': 1.e-7,
        'full_objective_evaluations': 7}
    normalization = tmp_path/'normalization.npz'
    normalization.write_bytes(b'fixed-original-normalization')
    identity = {'fold': 0, 'normalization_sha256': hashlib.sha256(normalization.read_bytes()).hexdigest()}
    parent = {'model': deepcopy(policy.state_dict()), 'optimizer': deepcopy(optimizer.state_dict()),
        'identity': identity, 'code_version': policy.config.version.unique_id48,
        'user_version': policy.config.user_version,
        'state': {'identity': {'model_config': asdict(policy.config), 'data_identity': 'original-training-population',
                'diagnostic': diagnostic},
            'history': [baseline], 'initial_gradient_norm': norms['full_gradient_norm'],
            'full_objective_evaluations': 7, 'empirical_local_stationarity': False}}
    source = tmp_path/'original-native.pt'
    torch.save(parent, source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = GGNStoppingReference(source_checkpoint_sha256=source_sha,
        initial_gradient_norm=norms['full_gradient_norm'], raw_threshold=1.e-7,
        affine_metric_threshold=1.e-7, source_accepted_updates=0,
        prior_full_objective_passes=7, prior_full_curvature_passes=0)
    config = GGNContinuationConfig(step=GGNStepConfig(damping=.001,
        cg_relative_tolerance=1.e-6, maximum_cg_iterations=30), maximum_accepted_steps=10)
    source_point = {name: baseline[name] for name in
        ('training_loss', 'full_gradient_norm', 'affine_metric_gradient_norm')}
    source_point['point_sha256'] = action_point_sha256(policy)
    state = continue_full_ggn_action(policy=policy, problem=problem, optimizer=optimizer,
        diagnostic=diagnostic, data_identity='original-training-population',
        reference=reference, source_point=source_point, config=config)
    assert state['converged'] and state['history']
    saved = deepcopy(parent)
    saved['model'] = deepcopy(policy.state_dict())
    saved['optimizer'] = deepcopy(optimizer.state_dict())
    saved['ggn_continuation_state'] = state
    saved['state']['history'].extend(deepcopy(state['history']))
    saved['state']['history'][-1]['full_objective_evaluations'] = state['full_objective_passes']
    saved['state']['full_objective_evaluations'] = state['full_objective_passes']
    saved['state']['empirical_local_stationarity'] = True
    checkpoint = tmp_path/'accepted-native.pt'
    torch.save(saved, checkpoint)
    plan = {'schema': 'gromm2-serial-strict-GGN-v1', 'original_thresholds_unchanged': True,
        'source_native': str(source), 'source_native_sha256': source_sha,
        'implementation_sha256': {str(source): source_sha}, 'normalization': str(normalization),
        'reference': asdict(reference), 'config': asdict(config),
        'source_point': {name: baseline[name] for name in
            ('training_loss', 'full_gradient_norm', 'affine_metric_gradient_norm')}}
    run_plan = tmp_path/'run-plan.json'
    plan_sha = save_json(run_plan, plan)
    arguments = {'policy': policy, 'problem': problem, 'optimizer': optimizer,
        'run_plan': run_plan, 'run_plan_sha256': plan_sha, 'accepted_checkpoint': checkpoint,
        'accepted_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    return arguments, saved


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_strict_receipt_recomputes_once_without_updates_history_loss_or_rng_changes(tmp_path, monkeypatch, diagnostic):
    arguments, saved = fixture(tmp_path, diagnostic)
    original = deepcopy(saved)
    point = action_point_sha256(arguments['policy'])
    optimizer_sha = optimizer_state_sha256(arguments['optimizer'])
    rng = torch.get_rng_state().clone()
    gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in arguments['policy'].parameters()]

    def forbidden_proposal(*args, **kwargs):
        raise AssertionError('finalization must not optimize')

    monkeypatch.setattr(ggn_audit_continuation, 'propose_full_ggn_step', forbidden_proposal)
    receipt = verify_accepted_ggn_action(**arguments)
    assert receipt['original_thresholds_met']
    assert not receipt['optimization_performed']
    assert receipt['verification_full_objective_passes'] == 1
    assert receipt['additional_curvature_passes'] == 0
    assert receipt['cumulative_full_objective_passes'] == saved['ggn_continuation_state']['full_objective_passes']+1
    assert saved['ggn_continuation_state'] == original['ggn_continuation_state']
    assert action_point_sha256(arguments['policy']) == point
    assert optimizer_state_sha256(arguments['optimizer']) == optimizer_sha
    assert torch.equal(torch.get_rng_state(), rng)
    for parameter, before in zip(arguments['policy'].parameters(), gradients, strict=True):
        assert (parameter.grad is None) if before is None else torch.equal(parameter.grad, before)


@pytest.mark.parametrize('field', ['full_gradient_norm', 'affine_metric_gradient_norm', 'training_loss'])
def test_recomputed_bad_gate_or_nonnegative_loss_cannot_publish_receipt(tmp_path, monkeypatch, field):
    arguments, saved = fixture(tmp_path)
    actual = ggn_audit_finalization.continue_full_ggn_action

    def invalid_receipt(**kwargs):
        result = actual(**kwargs)
        if field == 'training_loss':
            result['current_training_loss'] = 0.
        else:
            result['current_norms'][field] = 1.
        return result

    monkeypatch.setattr(ggn_audit_finalization, 'continue_full_ggn_action', invalid_receipt)
    with pytest.raises(ValueError, match='actual full training verification'):
        verify_accepted_ggn_action(**arguments)


def test_changed_actual_training_data_or_parameter_point_fails_closed(tmp_path):
    arguments, saved = fixture(tmp_path)
    original = deepcopy(arguments['policy'].state_dict())
    with torch.no_grad():
        arguments['policy'].affine.I0.bias.add_(1.e-4)
    with pytest.raises(ValueError, match='accepted parameter point'):
        verify_accepted_ggn_action(**arguments)
    arguments['policy'].load_state_dict(original)
    arguments['problem'].returns.add_(.01)
    with pytest.raises(ValueError, match='restore the actual accepted original objective'):
        verify_accepted_ggn_action(**arguments)


def test_rebased_threshold_and_changed_normalization_are_rejected(tmp_path):
    arguments, saved = fixture(tmp_path)
    plan = json.loads(arguments['run_plan'].read_text())
    plan['reference']['raw_threshold'] *= 10.
    arguments['run_plan_sha256'] = save_json(arguments['run_plan'], plan)
    with pytest.raises(ValueError, match='cannot rebase'):
        verify_accepted_ggn_action(**arguments)
    plan['reference']['raw_threshold'] /= 10.
    arguments['run_plan_sha256'] = save_json(arguments['run_plan'], plan)
    Path(plan['normalization']).write_bytes(b'changed-normalization')
    with pytest.raises(ValueError, match='normalization'):
        verify_accepted_ggn_action(**arguments)


def test_failed_policy_changed_history_and_unmatched_checkpoint_hash_are_rejected(tmp_path):
    arguments, saved = fixture(tmp_path)
    original_sha = arguments['accepted_checkpoint_sha256']
    arguments['accepted_checkpoint_sha256'] = '0'*64
    with pytest.raises(ValueError, match='source changed'):
        verify_accepted_ggn_action(**arguments)
    arguments['accepted_checkpoint_sha256'] = original_sha
    saved['ggn_continuation_state']['converged'] = False
    torch.save(saved, arguments['accepted_checkpoint'])
    arguments['accepted_checkpoint_sha256'] = hashlib.sha256(arguments['accepted_checkpoint'].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='not passed the original strict gates'):
        verify_accepted_ggn_action(**arguments)
    saved['ggn_continuation_state']['converged'] = True
    saved['state']['history'][-1]['optimizer_phase'] = 'unrecorded-method'
    torch.save(saved, arguments['accepted_checkpoint'])
    arguments['accepted_checkpoint_sha256'] = hashlib.sha256(arguments['accepted_checkpoint'].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='original history'):
        verify_accepted_ggn_action(**arguments)


@pytest.mark.parametrize('change', ['history', 'factor', 'diagnostic'])
def test_native_embedded_optimizer_and_source_diagnostic_must_match(tmp_path, change):
    arguments, saved = fixture(tmp_path)
    if change == 'history':
        assert saved['optimizer']['history']
        saved['optimizer']['history'] = []
        message = 'native optimizer controls or raw history'
    elif change == 'factor':
        saved['optimizer']['metric_factors'][0, 0, 0] += 1.e-6
        message = 'native metric factors or raw optimizer'
    else:
        saved['ggn_continuation_state']['identity']['diagnostic'] = 'entropy'
        message = 'original source policy objective'
    torch.save(saved, arguments['accepted_checkpoint'])
    arguments['accepted_checkpoint_sha256'] = hashlib.sha256(arguments['accepted_checkpoint'].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=message):
        verify_accepted_ggn_action(**arguments)
