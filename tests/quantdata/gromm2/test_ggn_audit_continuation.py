from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from quantdata.gromm2 import ggn_audit_optimization
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.ggn_audit_continuation import (
    AcceptedGGNWorkspace,
    GGNContinuationConfig,
    GGNStoppingReference,
    action_point_sha256,
    continue_full_ggn_action,
)
from quantdata.gromm2.ggn_audit_optimization import FullActionGGN, GGNStepConfig, _flatten
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.torch import BaseEnvironment


def fixture():
    torch.manual_seed(351)
    states = torch.randn((28, 2), dtype=torch.float32)
    returns = .2*states.double()+.1*states.double().square()
    states[24:] = float('nan')
    returns[24:] = float('nan')
    problem = AffineAuditProblem(states=states, returns=returns,
        eligible=torch.tensor([[True, True], [True, False]]), rows=torch.arange(24),
        action_scale=.2, config=AffineSolveConfig(batch_size=7))
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='ggn-continuation-test', feature_count=2, instrument_count=2,
            action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(std=.2)
    return problem, policy


def setup(problem, policy, diagnostic):
    policy.zero_grad(set_to_none=True)
    loss = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True))
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    norms = metric.diagnostics(gradient)
    reference = GGNStoppingReference(source_checkpoint_sha256='a'*64,
        initial_gradient_norm=norms['full_gradient_norm'], raw_threshold=1.e-12,
        affine_metric_threshold=1.e-12, source_accepted_updates=1602,
        prior_full_objective_passes=100, prior_full_curvature_passes=17)
    return {'policy': policy, 'problem': problem,
        'optimizer': MetricLBFGS(metric, history_size=5, curvature_floor=1.e-12),
        'diagnostic': diagnostic, 'reference': reference, 'data_identity': 'original-training-only',
        'source_point': {'training_loss': loss, **norms, 'point_sha256': action_point_sha256(policy)},
        'config': GGNContinuationConfig(step=GGNStepConfig(damping=.01,
            cg_relative_tolerance=1.e-4, maximum_cg_iterations=30), maximum_accepted_steps=3)}


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_workspace_refresh_uses_new_values_without_retracing(monkeypatch, diagnostic):
    problem, policy = fixture()
    actual_compile = torch.compile
    traces = []

    def backend(graph, inputs):
        traces.append(graph)
        return graph.forward

    monkeypatch.setattr(ggn_audit_optimization.torch, 'compile',
        lambda function: actual_compile(function, backend=backend))
    workspace = AcceptedGGNWorkspace(policy, problem, diagnostic=diagnostic, compile=True)
    vector = torch.sin(torch.arange(workspace.curvature.size, dtype=torch.float64))
    initial = workspace.curvature.product(vector)
    initial_traces = len(traces)
    pointers = {name: value.data_ptr() for name, value in workspace.curvature.parameters.items()}
    for generation in range(1, 4):
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(.01)
        bound = workspace.bind_accepted_point()
        actual = workspace.curvature.product(vector)
        expected = FullActionGGN(policy, problem, diagnostic=diagnostic).product(vector)
        torch.testing.assert_close(actual, expected, rtol=1.e-12, atol=1.e-13)
        assert bound['generation'] == generation
        assert workspace.point_sha256 == action_point_sha256(policy)
        assert pointers == {name: value.data_ptr() for name, value in workspace.curvature.parameters.items()}
        assert len(traces) == initial_traces
    assert not torch.allclose(initial, actual)


def test_workspace_rejects_mid_solve_refresh_and_stale_generation():
    problem, policy = fixture()
    workspace = AcceptedGGNWorkspace(policy, problem, diagnostic='squared')
    gradient = torch.ones(workspace.curvature.size, dtype=torch.float64)
    metric = AffineBlockMetric(policy, problem)

    def refresh(entry):
        workspace.bind_accepted_point()

    workspace.curvature.progress = refresh
    with pytest.raises(RuntimeError, match='active Krylov'):
        workspace.propose(gradient, metric, config=GGNStepConfig(), expected_generation=0)
    assert not workspace.active
    assert workspace.generation == 0
    with pytest.raises(ValueError, match='stale'):
        workspace.propose(gradient, metric, config=GGNStepConfig(), expected_generation=1)


@pytest.mark.parametrize('field', ['states', 'returns', 'rows', 'eligible'])
def test_workspace_rejects_mutated_training_observations_or_partition(field):
    problem, policy = fixture()
    workspace = AcceptedGGNWorkspace(policy, problem, diagnostic='squared')
    value = getattr(problem, field)
    with torch.no_grad():
        value.flatten()[0] = 0
    with pytest.raises(ValueError, match='data'):
        workspace.bind_accepted_point()


def test_failed_refresh_does_not_partially_replace_snapshot():
    problem, policy = fixture()
    workspace = AcceptedGGNWorkspace(policy, problem, diagnostic='squared')
    original = {name: value.clone() for name, value in workspace.curvature.parameters.items()}
    with torch.no_grad():
        policy.affine.I0.weight.add_(1.)
        policy.deep.In[-1].bias.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        workspace.bind_accepted_point()
    assert workspace.generation == 0
    for name, value in workspace.curvature.parameters.items():
        torch.testing.assert_close(value, original[name], rtol=0., atol=0.)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_bounded_full_continuation_descends_original_loss_not_certifying_work_cap(diagnostic):
    problem, policy = fixture()
    arguments = setup(problem, policy, diagnostic)
    checkpoints = []
    report = continue_full_ggn_action(**arguments,
        checkpoint=lambda model, optimizer, state: checkpoints.append(deepcopy(state)))
    assert len(report['history']) == 3
    assert report['stop_reason'] == 'work_limit_not_converged'
    assert not report['converged']
    assert not report['validation_used_for_steps_or_stopping']
    assert report['history'][-1]['training_loss'] < arguments['source_point']['training_loss']
    assert all(entry['acceptance']['accepted'] for entry in report['history'])
    assert report['identity']['reference'] == arguments['reference'].__dict__
    assert report['point_sha256'] == action_point_sha256(policy)
    expected_passes = 101+sum(entry['acceptance']['full_objective_passes'] for entry in report['history'])
    assert report['full_objective_passes'] == expected_passes
    assert report['full_curvature_passes'] == 17+sum(entry['proposal']['full_curvature_passes'] for entry in report['history'])
    assert checkpoints[-1] == report


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_interrupted_accepted_restart_preserves_weights_curvature_and_strict_baselines(diagnostic):
    problem, policy = fixture()
    uninterrupted = deepcopy(policy)
    baseline = continue_full_ggn_action(**setup(problem, uninterrupted, diagnostic))
    arguments = setup(problem, policy, diagnostic)
    saved = {}

    def stop(model, optimizer, state):
        if len(state['history']) == 1:
            saved.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), state=deepcopy(state))
            raise InterruptedError('coherent accepted-point checkpoint')

    with pytest.raises(InterruptedError):
        continue_full_ggn_action(**arguments, checkpoint=stop)
    policy.load_state_dict(saved['model'])
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.2), history_size=5, curvature_floor=1.e-12)
    optimizer.load_state_dict(saved['optimizer'])
    resumed = continue_full_ggn_action(**{**arguments, 'optimizer': optimizer}, resume_state=saved['state'])
    for actual, expected in zip(policy.parameters(), uninterrupted.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert resumed['history'][-1]['training_loss'] == baseline['history'][-1]['training_loss']
    assert resumed['full_objective_passes'] == baseline['full_objective_passes']+1
    assert resumed['full_curvature_passes'] == baseline['full_curvature_passes']
    assert resumed['identity'] == baseline['identity']
    assert resumed['history'][-1]['proposal']['workspace_generation'] == baseline['history'][-1]['proposal']['workspace_generation']


def test_continuation_rejects_changed_thresholds_solver_rows_or_exact_parameter_point():
    problem, policy = fixture()
    arguments = setup(problem, policy, 'squared')
    saved = {}

    def initial_checkpoint(model, optimizer, state):
        saved.update(deepcopy(state))
        raise InterruptedError('initial verified state')

    with pytest.raises(InterruptedError):
        continue_full_ggn_action(**arguments, checkpoint=initial_checkpoint)
    with pytest.raises(ValueError, match='identity'):
        continue_full_ggn_action(**{**arguments, 'reference': replace(arguments['reference'], raw_threshold=1.e-6)}, resume_state=saved)
    with pytest.raises(ValueError, match='identity'):
        continue_full_ggn_action(**{**arguments, 'config': replace(arguments['config'], step=GGNStepConfig(damping=.1))}, resume_state=saved)
    problem.config = replace(problem.config, batch_size=5)
    with pytest.raises(ValueError, match='identity'):
        continue_full_ggn_action(**arguments, resume_state=saved)
    problem.config = replace(problem.config, batch_size=7)
    with torch.no_grad():
        policy.affine.I0.bias.add_(.1)
        policy.deep.In[-1].bias.sub_(.1)
    with pytest.raises(ValueError, match='exact accepted parameter'):
        continue_full_ggn_action(**arguments, resume_state=saved)


def test_only_both_original_gradient_gates_can_converge():
    problem, policy = fixture()
    arguments = setup(problem, policy, 'squared')
    reference = arguments['reference']
    easy = replace(reference, raw_threshold=1.e3, affine_metric_threshold=1.e3)
    report = continue_full_ggn_action(**{**arguments, 'reference': easy})
    assert report['converged']
    assert not report['history']
    assert report['stop_reason'] == 'original_raw_and_affine_metric_gradient_tolerances'
    with pytest.raises(ValueError, match='positive'):
        GGNContinuationConfig(maximum_accepted_steps=0)


def test_resume_rejects_lost_raw_curvature_and_changed_actual_metric():
    problem, policy = fixture()
    arguments = setup(problem, policy, 'squared')
    saved = {}

    def checkpoint(model, optimizer, state):
        if len(state['history']) == 1:
            saved.update(deepcopy(state))
            raise InterruptedError('accepted source state')

    with pytest.raises(InterruptedError):
        continue_full_ggn_action(**arguments, checkpoint=checkpoint)
    optimizer = arguments['optimizer']
    assert optimizer.history
    original_history = optimizer.history
    optimizer.history = []
    with pytest.raises(ValueError, match='retained raw optimizer history'):
        continue_full_ggn_action(**arguments, resume_state=saved)
    optimizer.history = original_history
    optimizer.metric.factors.add_(.01)
    with pytest.raises(ValueError, match='actual fixed optimizer metric'):
        continue_full_ggn_action(**arguments, resume_state=saved)
