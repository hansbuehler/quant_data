from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.ggn_audit_continuation import GGNStoppingReference, action_point_sha256
from quantdata.gromm2.ggn_audit_optimization import GGNStepConfig
from quantdata.gromm2.gradient_reliable_continuation import (
    AcceptedHessianSnapshot,
    ReliableContinuationConfig,
    continue_gradient_reliable_action,
)
from quantdata.gromm2.gradient_reliable_ggn import GradientReliabilityConfig
from quantdata.gromm2.metric_audit_optimization import (
    AffineBlockMetric,
    MetricLBFGS,
    _flatten_parameters,
)
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.torch import BaseEnvironment


def setup(diagnostic):
    torch.manual_seed(612)
    states = torch.randn(14, 2, dtype=torch.float32)
    returns = .3*states.double().square()+.1*states.double()
    states[12:], returns[12:] = float('nan'), float('nan')
    problem = AffineAuditProblem(states=states, returns=returns,
        eligible=torch.tensor([[True, True], [True, False]]), rows=torch.arange(12),
        action_scale=.4, config=AffineSolveConfig(batch_size=5))
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='reliable-continuation-test', feature_count=2, instrument_count=2,
            action_scale=.4, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(std=.3)
    loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True))
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    norms = metric.diagnostics(_flatten_parameters(tuple(policy.parameters()), gradients=True))
    reference = GGNStoppingReference(source_checkpoint_sha256='a'*64, initial_gradient_norm=norms['full_gradient_norm'],
        raw_threshold=1.e-14, affine_metric_threshold=1.e-14,
        source_accepted_updates=1612, prior_full_objective_passes=100, prior_full_curvature_passes=17)
    return dict(policy=policy, problem=problem, optimizer=MetricLBFGS(metric, history_size=5, curvature_floor=1.e-12),
        diagnostic=diagnostic, data_identity='original-full-training-rows', reference=reference,
        source_point={'training_loss': loss, **norms, 'point_sha256': action_point_sha256(policy)},
        config=ReliableContinuationConfig(step=GGNStepConfig(damping=.0001, maximum_cg_iterations=100),
            acceptance=GradientReliabilityConfig(initial_rate=.25), maximum_accepted_steps=2))


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_reliable_continuation_preserves_gates_counts_raw_history_and_resume(diagnostic):
    complete = setup(diagnostic)
    full = continue_gradient_reliable_action(**complete)
    partial = setup(diagnostic)
    first = continue_gradient_reliable_action(**{**partial, 'config': replace(partial['config'], maximum_accepted_steps=1)})
    old_history = deepcopy(first['history'])
    resumed = continue_gradient_reliable_action(**partial, resume_state=first)
    assert len(full['history']) == len(resumed['history']) == 2
    assert resumed['history'][:1] == old_history
    assert resumed['history'][-1]['iteration'] == 1614
    assert not full['converged'] and not resumed['converged']
    assert full['stop_reason'] == resumed['stop_reason'] == 'work_limit_not_converged'
    assert full['identity']['reference'] == resumed['identity']['reference']
    assert full['full_objective_passes']+1 == resumed['full_objective_passes']
    assert full['full_curvature_passes'] == resumed['full_curvature_passes']
    assert full['full_curvature_passes'] == 17+sum(entry['full_curvature_passes'] for entry in full['history'])
    assert full['full_objective_passes'] == 101+sum(entry['acceptance']['full_objective_passes'] for entry in full['history'])
    for name, value in complete['policy'].state_dict().items():
        torch.testing.assert_close(value, partial['policy'].state_dict()[name], rtol=1.e-11, atol=1.e-12)
    for entry in full['history']:
        assert entry['gradient_threshold'] == 1.e-14
        assert entry['affine_metric_gradient_threshold'] == 1.e-14
        assert entry['full_curvature_passes'] == entry['full_ggn_passes']+entry['full_exact_hessian_passes']
    assert len(partial['optimizer'].history) > 0
    assert not full['validation_used_for_steps_or_stopping']


def test_reliable_restart_rejects_changed_original_threshold_and_actual_optimizer_history():
    arguments = setup('squared')
    first = continue_gradient_reliable_action(**{**arguments, 'config': replace(arguments['config'], maximum_accepted_steps=1)})
    with pytest.raises(ValueError, match='restart'):
        continue_gradient_reliable_action(**{**arguments, 'reference': replace(arguments['reference'], raw_threshold=.1)}, resume_state=first)
    arguments['optimizer'].history = []
    with pytest.raises(ValueError, match='restart'):
        continue_gradient_reliable_action(**arguments, resume_state=first)


def test_hessian_storage_refresh_is_guarded_and_preserves_tensor_addresses():
    arguments = setup('squared')
    policy, problem = arguments['policy'], arguments['problem']
    snapshot = AcceptedHessianSnapshot(policy, problem, diagnostic='squared', compile=False)
    pointers = {name: value.data_ptr() for name, value in snapshot.curvature.parameters.items()}
    snapshot.guard.active = True
    with pytest.raises(RuntimeError, match='active'):
        snapshot.bind_accepted_point()
    snapshot.guard.active = False
    with torch.no_grad():
        for value in policy.parameters():
            value.add_(.001)
    snapshot.bind_accepted_point()
    assert pointers == {name: value.data_ptr() for name, value in snapshot.curvature.parameters.items()}
    for name, value in policy.named_parameters():
        torch.testing.assert_close(value, snapshot.curvature.parameters[name], rtol=0., atol=0.)


@pytest.mark.parametrize('field', ['deep_scale', 'columns', 'slices'])
def test_actual_metric_mutation_cannot_hide_behind_unchanged_identity(field):
    arguments = setup('squared')
    first = continue_gradient_reliable_action(**{**arguments, 'config': replace(arguments['config'], maximum_accepted_steps=1)})
    metric = arguments['optimizer'].metric
    if field == 'deep_scale':
        metric.deep_scale = 20.
    elif field == 'columns':
        metric.columns[0][0] += 1
    else:
        metric.slices['affine.I0.weight'] = (slice(0, 1), torch.Size([1]))
    with pytest.raises(ValueError, match='actual optimizer metric'):
        continue_gradient_reliable_action(**arguments, resume_state=first)


def test_interrupted_secant_commit_rolls_back_weights_gradients_and_optimizer(monkeypatch):
    arguments = setup('squared')
    policy, optimizer = arguments['policy'], arguments['optimizer']
    native = deepcopy(policy.state_dict())
    gradient = _flatten_parameters(tuple(policy.parameters()), gradients=True)

    def interrupt(*unused):
        raise InterruptedError('test interrupted secant commit')

    monkeypatch.setattr(optimizer, 'accept', interrupt)
    with pytest.raises(InterruptedError) as caught:
        continue_gradient_reliable_action(**arguments)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, native[name], rtol=0., atol=0.)
    torch.testing.assert_close(_flatten_parameters(tuple(policy.parameters()), gradients=True), gradient,
        rtol=1.e-13, atol=1.e-14)
    assert optimizer.history == []
    work = caught.value.reliable_attempt_work
    assert work['accepted_steps_before_interruption'] == 0
    assert work['complete_objective_passes'] > 101
    assert work['invocation_exact_hessian_passes'] > 0


def test_interrupted_history_digest_assembly_does_not_commit_a_candidate(monkeypatch):
    from quantdata.gromm2 import gradient_reliable_continuation

    arguments = setup('squared')
    policy, optimizer = arguments['policy'], arguments['optimizer']
    native = deepcopy(policy.state_dict())
    original_digest = gradient_reliable_continuation.action_point_sha256
    initial = original_digest(policy)

    def checked_digest(model):
        actual = original_digest(model)
        if actual != initial:
            raise InterruptedError('test history digest interruption')
        return actual

    monkeypatch.setattr(gradient_reliable_continuation, 'action_point_sha256', checked_digest)
    with pytest.raises(InterruptedError) as caught:
        continue_gradient_reliable_action(**arguments)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, native[name], rtol=0., atol=0.)
    assert optimizer.history == []
    assert caught.value.reliable_attempt_work['accepted_steps_before_interruption'] == 0


def test_zero_accepted_updates_report_nonconvergence_with_exact_failed_work():
    arguments = setup('squared')
    policy = arguments['policy']
    original = action_point_sha256(policy)
    acceptance = GradientReliabilityConfig(initial_rate=.25, minimum_rate=.25,
        nonlinear_fraction=1.e-12, expansion_fraction=1.e-13)
    result = continue_gradient_reliable_action(**{**arguments,
        'config': replace(arguments['config'], acceptance=acceptance)})
    assert result['history'] == [] and not result['converged']
    assert result['stop_reason'] == 'gradient_reliable_line_search_not_converged'
    assert result['point_sha256'] == action_point_sha256(policy) == original
    assert result['rejected_work']['acceptance']['original_state_restored']
    assert result['full_objective_passes'] == 101+result['rejected_work']['acceptance']['full_objective_passes']
    assert result['full_curvature_passes'] == 17+result['invocation_ggn_passes']+result['invocation_exact_hessian_passes']
