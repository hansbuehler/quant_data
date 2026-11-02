from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
    local_objective,
)
from quantdata.gromm2.neural_audit_optimization import (
    NeuralSolveConfig,
    fit_full_gradient_action,
    full_action_pass,
)
from quantdata.torch import BaseEnvironment


def fixture():
    generator = torch.Generator().manual_seed(61)
    states = torch.randn((64, 3), generator=generator, dtype=torch.float64)
    returns = .2*states[:, :2]+.1*states[:, :2].square()
    eligible = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
    problem = AffineAuditProblem(states=states, returns=returns, eligible=eligible, rows=torch.arange(48),
        action_scale=.2, config=AffineSolveConfig(batch_size=13))
    torch.manual_seed(71)
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='test', feature_count=3, instrument_count=2, action_scale=.2,
            network=ActionNetworkConfig(num_depth=1, num_width=6)))
    return problem, policy


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_full_gradient_accumulation_matches_one_canonical_objective(diagnostic):
    problem, policy = fixture()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(mean=0., std=.1)
    actual = full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    actual_gradients = {name: parameter.grad.clone() for name, parameter in policy.named_parameters()}
    policy.zero_grad(set_to_none=True)
    expected = local_objective(policy(problem.states[:48], problem.eligible[problem.rows % 2]),
        problem.returns[:48], diagnostic=diagnostic, action_scale=.2)
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(actual_gradients[name], parameter.grad, atol=1.e-14, rtol=1.e-12)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_full_network_training_improves_same_objective_without_reading_unused_rows(diagnostic):
    problem, policy = fixture()
    problem.states[48:] = float('nan')
    problem.returns[48:] = float('nan')
    report = fit_full_gradient_action(policy=policy, problem=problem, diagnostic=diagnostic,
        config=NeuralSolveConfig(maximum_iterations=15))
    assert report['history'][-1]['training_loss'] < -.01
    assert report['history'][0]['training_loss'] == 0.
    assert report['gradient_clipping'] is False
    assert report['validation_used_for_steps_or_stopping'] is False
    assert report['working_memory_batch_size'] == 13
    assert report['full_gradient_batch_observations'] == 48
    assert not policy.affine_only
    losses = [item['training_loss'] for item in report['history']]
    assert losses == sorted(losses, reverse=True)
    if not report['converged']:
        assert report['stop_reason'].endswith('not_converged')


def test_full_gradient_optimizer_restart_matches_uninterrupted_updates():
    problem, policy = fixture()
    uninterrupted = deepcopy(policy)
    config = NeuralSolveConfig(maximum_iterations=8, gradient_relative_tolerance=1.e-12,
        gradient_absolute_tolerance=1.e-15)
    baseline = fit_full_gradient_action(policy=uninterrupted, problem=problem, diagnostic='squared', config=config)
    saved = {}

    def interrupt(model, optimizer, state):
        if state['history'][-1]['iteration'] == 3:
            saved.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), state=deepcopy(state))
            raise InterruptedError('test accepted-state checkpoint')

    with pytest.raises(InterruptedError):
        fit_full_gradient_action(policy=policy, problem=problem, diagnostic='squared', config=config, checkpoint=interrupt)
    policy.load_state_dict(saved['model'])
    optimizer = torch.optim.LBFGS(policy.parameters(), lr=1., max_iter=1, max_eval=30,
        history_size=30, tolerance_grad=0., tolerance_change=0., line_search_fn='strong_wolfe')
    optimizer.load_state_dict(saved['optimizer'])
    resumed = fit_full_gradient_action(policy=policy, problem=problem, diagnostic='squared',
        config=config, optimizer=optimizer, resume_state=saved['state'])
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, dict(uninterrupted.named_parameters())[name], atol=0., rtol=0.)
    assert resumed['history'][-1]['training_loss'] == baseline['history'][-1]['training_loss']
    assert resumed['history'][-1]['gradient_threshold'] == baseline['history'][-1]['gradient_threshold']


def test_wrong_optimizer_parameter_binding_is_rejected():
    problem, policy = fixture()
    other = deepcopy(policy)
    optimizer = torch.optim.LBFGS(other.parameters(), lr=1., max_iter=1, max_eval=30,
        history_size=30, tolerance_grad=0., tolerance_change=0., line_search_fn='strong_wolfe')
    with pytest.raises(ValueError, match='parameter identity'):
        fit_full_gradient_action(policy=policy, problem=problem, diagnostic='squared', optimizer=optimizer)


def test_exhausted_line_search_is_not_convergence_and_preserves_accepted_state():
    problem, policy = fixture()
    original = deepcopy(policy.state_dict())
    result = fit_full_gradient_action(policy=policy, problem=problem, diagnostic='squared',
        config=NeuralSolveConfig(maximum_line_search_evaluations=1, maximum_iterations=2))
    assert result['converged'] is False
    assert result['stop_reason'] == 'line_search_evaluation_limit_not_converged'
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0., rtol=0.)
