from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from quantdata.gromm2 import gradient_reliable_continuation
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
from quantdata.gromm2.ggn_audit_optimization import GGNStepConfig, _flatten
from quantdata.gromm2.gradient_reliable_continuation import (
    ReliableContinuationConfig,
    continue_gradient_reliable_action,
)
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.gromm2.preconditioned_reliable_continuation import (
    AcceptedPreconditionedGGNWorkspace,
    PreconditionedReliableConfig,
    continue_preconditioned_reliable_action,
)
from quantdata.gromm2.strict_action_verification import (
    native_curvature_passes,
    verify_strict_action,
)
from quantdata.torch import BaseEnvironment


def fixture(tmp_path, diagnostic='squared', phase='gromm2-gradient-reliable-GGN-v1'):
    states = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float32).repeat(2, 1)
    problem = AffineAuditProblem(states=states, returns=torch.tensor([[.1, .2]], dtype=torch.float32).repeat(8, 1),
        eligible=torch.ones((2, 2), dtype=torch.bool), rows=torch.arange(8), action_scale=.2,
        config=AffineSolveConfig(batch_size=3))
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='strict-reliable-verification-test', feature_count=2,
            instrument_count=2, action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    policy.zero_grad(set_to_none=True)
    loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True))
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.2), history_size=5, curvature_floor=1.e-12)
    norms = optimizer.metric.diagnostics(gradient)
    baseline = {'iteration': 0, 'training_loss': loss, **norms, 'gradient_threshold': 1.e-7,
        'affine_metric_gradient_threshold': 1.e-7, 'full_objective_evaluations': 7, 'full_curvature_passes': 3}
    normalization = tmp_path/'normalization.npz'
    normalization.write_bytes(b'fixed-original-normalization')
    identity = {'fold': 0, 'normalization_sha256': hashlib.sha256(normalization.read_bytes()).hexdigest()}
    parent = {'model': deepcopy(policy.state_dict()), 'optimizer': deepcopy(optimizer.state_dict()),
        'identity': identity, 'code_version': policy.config.version.unique_id48, 'user_version': policy.config.user_version,
        'optimizer_phase_schema': 'gromm2-serial-strict-GGN-v1', 'ggn_continuation_state': {'full_curvature_passes': 17},
        'torch_rng': torch.get_rng_state().clone(), 'cuda_rng': None,
        'state': {'identity': {'model_config': asdict(policy.config), 'data_identity': 'original-training-population',
                'diagnostic': diagnostic}, 'history': [baseline], 'initial_gradient_norm': norms['full_gradient_norm'],
            'full_objective_evaluations': 7, 'full_curvature_evaluations': 17, 'empirical_local_stationarity': False}}
    if phase == 'gromm2-history-preconditioned-reliable-GGN-v1':
        parent['optimizer_phase_schema'] = 'gromm2-gradient-reliable-GGN-v1'
        parent['reliable_continuation_state'] = {'full_curvature_passes': 17}
    source = tmp_path/'source-native.pt'
    torch.save(parent, source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = GGNStoppingReference(source_checkpoint_sha256=source_sha, initial_gradient_norm=norms['full_gradient_norm'],
        raw_threshold=1.e-7, affine_metric_threshold=1.e-7, source_accepted_updates=0,
        prior_full_objective_passes=7, prior_full_curvature_passes=17)
    step = GGNStepConfig(damping=.001, cg_relative_tolerance=1.e-6, maximum_cg_iterations=30)
    if phase == 'gromm2-history-preconditioned-reliable-GGN-v1':
        config = PreconditionedReliableConfig(step=step, maximum_accepted_steps=16)
        continuation, state_key = continue_preconditioned_reliable_action, 'preconditioned_reliable_state'
    elif phase == 'gromm2-gradient-reliable-GGN-v1':
        config = ReliableContinuationConfig(step=step, maximum_accepted_steps=16)
        continuation, state_key = continue_gradient_reliable_action, 'reliable_continuation_state'
    else:
        config = GGNContinuationConfig(step=step, maximum_accepted_steps=16)
        continuation, state_key = continue_full_ggn_action, 'ggn_continuation_state'
    source_point = {name: baseline[name] for name in ('training_loss', 'full_gradient_norm', 'affine_metric_gradient_norm')}
    state = continuation(policy=policy, problem=problem, optimizer=optimizer,
        diagnostic=diagnostic, data_identity='original-training-population', reference=reference,
        source_point={**source_point, 'point_sha256': action_point_sha256(policy)}, config=config)
    assert state['converged'] and state['history']
    saved = deepcopy(parent)
    saved.update(optimizer_phase_schema=phase, model=deepcopy(policy.state_dict()), optimizer=deepcopy(optimizer.state_dict()))
    saved[state_key] = state
    saved['state']['history'].extend(deepcopy(state['history']))
    saved['state']['history'][-1]['full_objective_evaluations'] = state['full_objective_passes']
    saved['state'].update(full_objective_evaluations=state['full_objective_passes'],
        full_curvature_evaluations=state['full_curvature_passes'], empirical_local_stationarity=True)
    checkpoint = tmp_path/'accepted-native.pt'
    torch.save(saved, checkpoint)
    plan = {'schema': phase, 'original_thresholds_unchanged': True,
        'source_native': str(source), 'source_native_sha256': source_sha, 'normalization': str(normalization),
        'implementation_sha256': {str(source): source_sha}, 'reference': asdict(reference),
        'config': asdict(config), 'source_point': source_point}
    run_plan = tmp_path/'run-plan.json'
    run_plan.write_text(json.dumps(plan))
    return dict(policy=policy, problem=problem, optimizer=optimizer, run_plan=run_plan,
        run_plan_sha256=hashlib.sha256(run_plan.read_bytes()).hexdigest(), accepted_checkpoint=checkpoint,
        accepted_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()), saved


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
@pytest.mark.parametrize('phase', ['gromm2-gradient-reliable-GGN-v1', 'gromm2-serial-strict-GGN-v1',
    'gromm2-history-preconditioned-reliable-GGN-v1'])
def test_active_verification_never_optimizes_or_uses_stale_state(tmp_path, monkeypatch, diagnostic, phase):
    arguments, saved = fixture(tmp_path, diagnostic, phase)
    point = action_point_sha256(arguments['policy'])
    optimizer_sha = optimizer_state_sha256(arguments['optimizer'])
    gradients = [value.grad.clone() for value in arguments['policy'].parameters()]
    rng = torch.get_rng_state().clone()

    def forbidden(*unused, **keywords):
        raise AssertionError('verification must not propose any optimizer step')

    monkeypatch.setattr(gradient_reliable_continuation.AcceptedGGNWorkspace, 'propose', forbidden)
    monkeypatch.setattr(AcceptedPreconditionedGGNWorkspace, 'propose', forbidden)
    receipt = verify_strict_action(**arguments)
    assert receipt['original_thresholds_met'] and not receipt['optimization_performed']
    assert receipt['active_optimizer_phase'] == saved['optimizer_phase_schema']
    assert receipt['verification_full_objective_passes'] == 1 and receipt['additional_curvature_passes'] == 0
    assert receipt['cumulative_full_curvature_passes'] == native_curvature_passes(saved)
    assert action_point_sha256(arguments['policy']) == point
    assert optimizer_state_sha256(arguments['optimizer']) == optimizer_sha
    assert torch.equal(torch.get_rng_state(), rng)
    for value, original in zip(arguments['policy'].parameters(), gradients, strict=True):
        torch.testing.assert_close(value.grad, original, rtol=0., atol=0.)


@pytest.mark.parametrize('field', ['phase', 'old_state', 'gates', 'rng'])
def test_phase_lineage_original_gates_and_native_rng_are_bound(tmp_path, field):
    arguments, saved = fixture(tmp_path)
    if field == 'phase':
        saved['optimizer_phase_schema'] = 'gromm2-serial-strict-GGN-v1'
        message = 'explicit matching active'
    elif field == 'old_state':
        saved['ggn_continuation_state']['full_curvature_passes'] += 1
        message = 'preceding optimizer-phase'
    elif field == 'gates':
        saved['reliable_continuation_state']['converged'] = False
        message = 'original strict gates'
    else:
        saved['torch_rng'][0] ^= 1
        message = 'native random state'
    torch.save(saved, arguments['accepted_checkpoint'])
    arguments['accepted_checkpoint_sha256'] = hashlib.sha256(arguments['accepted_checkpoint'].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=message):
        verify_strict_action(**arguments)


def test_curvature_uses_active_cumulative_counts_not_accepted_history_sum():
    native = {'optimizer_phase_schema': 'gromm2-gradient-reliable-GGN-v1',
        'ggn_continuation_state': {'full_curvature_passes': 17},
        'reliable_continuation_state': {'full_curvature_passes': 40},
        'state': {'full_curvature_evaluations': 40, 'history': [{'full_curvature_passes': 3}]}}
    assert native_curvature_passes(native) == 40
    native['state']['full_curvature_evaluations'] = 39
    with pytest.raises(ValueError, match='active native curvature'):
        native_curvature_passes(native)


@pytest.mark.parametrize('phase', ['gromm2-gradient-reliable-GGN-v1', 'gromm2-serial-strict-GGN-v1',
    'gromm2-history-preconditioned-reliable-GGN-v1'])
def test_both_phases_reject_inconsistent_native_cumulative_counters(tmp_path, phase):
    arguments, saved = fixture(tmp_path, phase=phase)
    saved['state']['full_curvature_evaluations'] += 100
    torch.save(saved, arguments['accepted_checkpoint'])
    arguments['accepted_checkpoint_sha256'] = hashlib.sha256(arguments['accepted_checkpoint'].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='active native curvature'):
        verify_strict_action(**arguments)


def test_preconditioned_phase_preserves_both_prior_optimizer_records(tmp_path):
    arguments, saved = fixture(tmp_path, phase='gromm2-history-preconditioned-reliable-GGN-v1')
    saved['reliable_continuation_state']['full_curvature_passes'] += 1
    torch.save(saved, arguments['accepted_checkpoint'])
    arguments['accepted_checkpoint_sha256'] = hashlib.sha256(arguments['accepted_checkpoint'].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='preceding optimizer-phase'):
        verify_strict_action(**arguments)
