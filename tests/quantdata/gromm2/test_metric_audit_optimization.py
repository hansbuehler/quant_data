from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from quantdata.gromm2 import metric_audit_optimization
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.metric_audit_optimization import (
    AffineBlockMetric,
    MetricLBFGS,
    MetricSolveConfig,
    OptimizerMigration,
    fit_metric_action,
)
from quantdata.gromm2.neural_audit_optimization import NeuralSolveConfig, full_action_pass
from quantdata.torch import BaseEnvironment


def fixture():
    generator = torch.Generator().manual_seed(61)
    states = torch.randn((64, 3), generator=generator, dtype=torch.float64)
    returns = .2*states[:, :2]+.1*states[:, :2].square()
    problem = AffineAuditProblem(states=states, returns=returns,
        eligible=torch.tensor([[True, True], [True, False]], dtype=torch.bool), rows=torch.arange(48),
        action_scale=.2, config=AffineSolveConfig(batch_size=13))
    torch.manual_seed(71)
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='test', feature_count=3, instrument_count=2, action_scale=.2,
            network=ActionNetworkConfig(num_depth=1, num_width=6)))
    return problem, policy


def test_factorized_metric_and_lbfgs_equal_dense_spd_inverse_bfgs():
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem)
    basis = torch.eye(metric.size, dtype=torch.float64)
    inverse = torch.stack([metric(vector) for vector in basis], dim=1)
    torch.testing.assert_close(inverse, inverse.T, rtol=1.e-12, atol=1.e-12)
    assert torch.linalg.eigvalsh(inverse).min() > 0.
    optimizer = MetricLBFGS(metric, history_size=5, curvature_floor=1.e-12)
    generator = torch.Generator().manual_seed(15)
    for _ in range(3):
        step = torch.randn(metric.size, generator=generator, dtype=torch.float64)
        change = step+.01*torch.randn(metric.size, generator=generator, dtype=torch.float64)
        reciprocal = torch.dot(step, change).reciprocal()
        transform = basis-reciprocal*torch.outer(step, change)
        inverse = transform@inverse@transform.T+reciprocal*torch.outer(step, step)
        assert optimizer.accept(step, change)
    gradient = torch.randn(metric.size, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(optimizer.direction(gradient), -inverse@gradient, atol=1.e-11, rtol=1.e-10)
    assert torch.dot(gradient, optimizer.direction(gradient)) < 0.


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_metric_fit_descends_full_original_objective_without_reading_unused_rows(diagnostic):
    problem, policy = fixture()
    problem.states[48:], problem.returns[48:] = float('nan'), float('nan')
    report = fit_metric_action(policy=policy, problem=problem, diagnostic=diagnostic, data_identity='fixture',
        config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=12)))
    assert report['history'][-1]['training_loss'] < -.01
    assert report['validation_used_for_steps_or_stopping'] is False
    assert not report['coefficient_directions_discarded']
    assert not report['objective_ridge_added']
    assert all(value['line_search']['armijo'] and value['line_search']['strong_curvature']
        for value in report['history'][:-1])
    verified = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=False))
    assert report['history'][-1]['training_loss'] == pytest.approx(verified, abs=1.e-14)
    assert report['full_objective_evaluations'] <= 1+3*12


def test_metric_fit_restart_reproduces_accepted_weights_and_objectives():
    problem, policy = fixture()
    uninterrupted = deepcopy(policy)
    config = MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=8,
        gradient_relative_tolerance=1.e-12, gradient_absolute_tolerance=1.e-15))
    baseline = fit_metric_action(policy=uninterrupted, problem=problem, diagnostic='squared',
        data_identity='fixture', config=config)
    saved = {}

    def interrupt(model, optimizer, state):
        if state['history'][-1]['iteration'] == 3:
            saved.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), state=deepcopy(state))
            raise InterruptedError('accepted-state checkpoint')

    with pytest.raises(InterruptedError):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
            config=config, checkpoint=interrupt)
    policy.load_state_dict(saved['model'])
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem), history_size=config.neural.history_size,
        curvature_floor=config.curvature_relative_floor)
    optimizer.load_state_dict(saved['optimizer'])
    resumed = fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
        config=config, optimizer=optimizer, resume_state=saved['state'])
    for actual, expected in zip(policy.parameters(), uninterrupted.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert resumed['history'][-1]['training_loss'] == baseline['history'][-1]['training_loss']


def test_incomplete_line_search_restores_weights_optimizer_and_gradients(monkeypatch):
    problem, policy = fixture()
    expected = deepcopy(policy.state_dict())
    policy.zero_grad(set_to_none=True)
    full_action_pass(policy, problem, diagnostic='squared', backward=True)
    expected_gradients = [parameter.grad.clone() for parameter in policy.parameters()]
    original = metric_audit_optimization.full_action_pass
    count = 0

    def failed_trial(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 1:
            return torch.tensor(float('inf'), dtype=torch.float64)
        return original(*args, **kwargs)

    monkeypatch.setattr(metric_audit_optimization, 'full_action_pass', failed_trial)
    report = fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
        config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_line_search_evaluations=2)))
    assert report['stop_reason'] == 'line_search_not_converged'
    assert not report['converged']
    for name, actual in policy.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0., atol=0.)
    for parameter, expected in zip(policy.parameters(), expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=0., atol=0.)


def test_metric_restart_rejects_changed_geometry_or_data_identity():
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem)
    optimizer = MetricLBFGS(metric, history_size=30, curvature_floor=1.e-12)
    saved = optimizer.state_dict()
    with pytest.raises(ValueError, match='identity changed'):
        MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=2.), history_size=30,
            curvature_floor=1.e-12).load_state_dict(saved)
    with pytest.raises(ValueError, match='positive'):
        MetricLBFGS(metric, history_size=0, curvature_floor=1.e-12)
    assert replace(MetricSolveConfig(), deep_metric_scale=2.).deep_metric_scale == 2.


def test_metric_retains_predictive_nearly_collinear_affine_direction():
    first = torch.tensor([-1., -1., 1., 1.], dtype=torch.float64).repeat(16)
    independent = torch.tensor([-1., 1., -1., 1.], dtype=torch.float64).repeat(16)
    states = torch.stack((first, first+1.e-7*independent), dim=1)
    problem = AffineAuditProblem(states=states, returns=.1*independent[:, None],
        eligible=torch.ones((1, 1), dtype=torch.bool), rows=torch.arange(64), action_scale=.2,
        config=AffineSolveConfig(batch_size=16))
    torch.manual_seed(17)
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='near-collinear', feature_count=2, instrument_count=1,
            action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    report = fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='near-collinear',
        config=MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=40)))
    assert report['history'][-1]['training_loss'] < -.0049
    assert report['history'][0]['affine_metric_gradient_norm'] > report['history'][0]['affine_metric_gradient_threshold']
    actual = policy(states, problem.eligible).detach().squeeze()
    torch.testing.assert_close(actual, .1*independent/(1.+.04*.01), atol=1.e-4, rtol=1.e-4)


def test_interrupt_during_trial_restores_last_accepted_parameters_and_gradients():
    problem, policy = fixture()
    expected = deepcopy(policy.state_dict())
    full_action_pass(policy, problem, diagnostic='squared', backward=True)
    expected_gradients = [parameter.grad.clone() for parameter in policy.parameters()]

    def interrupt(entry):
        if entry.get('stage') == 'metric_full_gradient_evaluation' and entry['full_objective_evaluations'] == 2:
            raise InterruptedError('trial not committed')

    with pytest.raises(InterruptedError):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture', progress=interrupt)
    for name, actual in policy.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0., atol=0.)
    for parameter, expected in zip(policy.parameters(), expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=0., atol=0.)


def test_explicit_optimizer_migration_preserves_original_gradient_reference():
    problem, policy = fixture()
    saved = {}

    def stop_before_step(model, optimizer, state):
        saved.update(deepcopy(state))
        raise InterruptedError('verified migration boundary')

    with pytest.raises(InterruptedError):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='migrated-fixture',
            initial_gradient_reference=.001, checkpoint=stop_before_step)
    assert saved['initial_gradient_norm'] == .001
    assert saved['history'][0]['gradient_threshold'] == pytest.approx(1.1e-8)


def test_migration_binds_prior_work_and_rejects_threshold_relaxation():
    problem, policy = fixture()
    migration = OptimizerMigration(accepted_checkpoint_sha256='a'*64, source_optimizer='torch.optim.LBFGS',
        initial_gradient_norm=.001, absolute_gradient_threshold=1.e-9+1.e-5*.001,
        prior_accepted_updates=300, prior_full_objective_passes=1201)
    captured = {}

    def interrupt(model, optimizer, state):
        captured.update(deepcopy(state))
        raise InterruptedError('bound migration')

    with pytest.raises(InterruptedError):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
            migration=migration, checkpoint=interrupt)
    assert captured['identity']['optimizer_migration']['prior_accepted_updates'] == 300
    assert captured['history'][0]['iteration'] == 0
    assert captured['initial_gradient_norm'] == .001
    with pytest.raises(ValueError, match='cannot relax'):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
            migration=migration, config=MetricSolveConfig(neural=NeuralSolveConfig(gradient_relative_tolerance=.01)))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem), history_size=30, curvature_floor=1.e-12)
    with pytest.raises(ValueError, match='restart objective'):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
            migration=replace(migration, accepted_checkpoint_sha256='b'*64), optimizer=optimizer, resume_state=captured)
    optimizer.history.append((torch.ones(optimizer.metric.size), torch.ones(optimizer.metric.size), torch.tensor(1.)))
    with pytest.raises(ValueError, match='reset its curvature'):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
            migration=migration, optimizer=optimizer)


def test_native_learning_rate_continuation_preserves_weights_history_and_gradient_reference():
    problem, policy = fixture()
    saved = {}
    original_config = MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=8,
        gradient_relative_tolerance=1.e-12, gradient_absolute_tolerance=1.e-15))

    def interrupt(model, optimizer, state):
        if state['history'][-1]['iteration'] == 3:
            saved.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), state=deepcopy(state))
            raise InterruptedError('accepted checkpoint before learning-rate continuation')

    with pytest.raises(InterruptedError):
        fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
            config=original_config, checkpoint=interrupt)
    policy.load_state_dict(saved['model'], strict=True)
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem), history_size=30, curvature_floor=1.e-12)
    optimizer.load_state_dict(saved['optimizer'])
    updated = replace(original_config, neural=replace(original_config.neural, initial_learning_rate=.001, maximum_iterations=5))
    report = fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='fixture',
        config=updated, optimizer=optimizer, resume_state=saved['state'])
    assert report['history'][3]['training_loss'] == saved['state']['history'][-1]['training_loss']
    assert report['history'][3]['gradient_threshold'] == saved['state']['history'][-1]['gradient_threshold']
    assert report['history'][:3] == saved['state']['history'][:3]
    assert report['history'][-1]['training_loss'] < report['history'][3]['training_loss']
    assert report['history'][-1]['iteration'] == 5
    assert len(optimizer.history) >= len(saved['optimizer']['history'])
