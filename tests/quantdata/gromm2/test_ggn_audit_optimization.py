from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from quantdata.gromm2 import ggn_audit_optimization
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
    local_objective,
)
from quantdata.gromm2.ggn_audit_optimization import (
    FullActionGGN,
    GGNStepConfig,
    InverseAffineMetric,
    _flatten,
    _unpack,
    accept_full_ggn_direction,
    propose_full_ggn_step,
)
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.torch import BaseEnvironment


def fixture():
    torch.manual_seed(814)
    states = torch.randn(28, 3, dtype=torch.float32)
    returns = .3*states[:, :2].double().square()+.1*states[:, 1:].double()
    states[24:] = float('nan')
    returns[24:] = float('nan')
    problem = AffineAuditProblem(states=states, returns=returns,
        eligible=torch.tensor([[True, True], [True, False]]), rows=torch.arange(24),
        action_scale=.4, config=AffineSolveConfig(batch_size=7))
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='ggn-test', feature_count=3, instrument_count=2,
            action_scale=.4, network=ActionNetworkConfig(num_depth=2, num_width=3)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(std=.3)
    return problem, policy


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_streamed_full_ggn_matches_dense_curvature_and_preserves_native_state(diagnostic):
    problem, policy = fixture()
    before = deepcopy(policy.state_dict())
    rng = torch.get_rng_state().clone()
    parameters = {name: value.detach() for name, value in policy.named_parameters()}
    vector = _flatten(parameters)
    features = problem.states[problem.rows].double()
    eligible = problem.eligible[problem.rows % len(problem.eligible)]

    def action_function(values):
        return torch.func.functional_call(policy, _unpack(values, parameters), (features, eligible), strict=True)

    action = action_function(vector)
    jacobian = torch.func.jacrev(action_function)(vector).reshape(action.numel(), -1)
    output_hessian = torch.autograd.functional.hessian(lambda values: local_objective(
        values.reshape_as(action), problem.returns[problem.rows], diagnostic=diagnostic,
        action_scale=problem.action_scale, squared_lambda=problem.squared_lambda), action.flatten())
    expected = jacobian.T@output_hessian@jacobian
    direction = torch.sin(torch.arange(len(vector), dtype=vector.dtype))
    other = torch.cos(torch.arange(len(vector), dtype=vector.dtype))
    curvature = FullActionGGN(policy, problem, diagnostic=diagnostic)
    actual = curvature.product(direction)
    torch.testing.assert_close(actual, expected@direction, rtol=2.e-11, atol=3.e-12)
    torch.testing.assert_close(torch.dot(other, actual), torch.dot(direction, curvature.product(other)),
        rtol=1.e-11, atol=1.e-12)
    assert torch.dot(direction, actual) >= -1.e-12
    null = {name: torch.zeros_like(value) for name, value in parameters.items()}
    null['affine.I0.bias'].fill_(1.)
    null['deep.In.1.bias'].fill_(-1.)
    torch.testing.assert_close(curvature.product(_flatten(null)), torch.zeros_like(vector), atol=1.e-12, rtol=0.)
    assert curvature.full_curvature_passes == 3
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0., atol=0.)
    assert all(parameter.grad is None for parameter in policy.parameters())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0., atol=0.)


def test_positive_metric_inverse_preserves_all_directions():
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem, deep_scale=.03)
    inverse = InverseAffineMetric(metric)
    vector = torch.sin(torch.arange(metric.size, dtype=torch.float64))
    torch.testing.assert_close(inverse(metric(vector)), vector, rtol=1.e-11, atol=1.e-12)
    torch.testing.assert_close(metric(inverse(vector)), vector, rtol=1.e-11, atol=1.e-12)
    assert torch.dot(vector, inverse(vector)) > 0.


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_coupled_proposal_updates_hidden_layers_and_descends_unchanged_objective(diagnostic):
    problem, policy = fixture()
    initial_loss = full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    initial = {name: value.detach().clone() for name, value in policy.named_parameters()}
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    direction, report = propose_full_ggn_step(curvature=FullActionGGN(policy, problem, diagnostic=diagnostic),
        gradient=gradient, metric=metric, config=GGNStepConfig(damping=.1, cg_relative_tolerance=1.e-5,
            maximum_cg_iterations=60))
    assert report['slope'] < 0.
    assert report['outer_gradient_convergence_certified'] is False
    assert not report['damping_is_objective_penalty']
    assert torch.linalg.vector_norm(_unpack(direction, initial)['deep.I0.weight']) > 1.e-5
    for name, value in policy.named_parameters():
        torch.testing.assert_close(value, initial[name], rtol=0., atol=0.)
    rate = 1.
    for _ in range(30):
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                parameter.copy_(initial[name]+rate*_unpack(direction, initial)[name])
        candidate = full_action_pass(policy, problem, diagnostic=diagnostic, backward=False)
        if candidate <= initial_loss+1.e-4*rate*torch.dot(gradient, direction):
            break
        rate *= .5
    else:
        pytest.fail('canonical original-objective line search failed')
    assert candidate < initial_loss


def test_curvature_snapshot_is_independent_of_later_parameter_edits():
    problem, policy = fixture()
    curvature = FullActionGGN(policy, problem, diagnostic='squared')
    direction = torch.ones(curvature.size, dtype=torch.float64)
    expected = curvature.product(direction)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(10.)
    torch.testing.assert_close(curvature.product(direction), expected, rtol=0., atol=0.)
    with pytest.raises(ValueError, match='snapshot'):
        propose_full_ggn_step(curvature=curvature, gradient=direction, metric=AffineBlockMetric(policy, problem))


def test_curvature_snapshot_preserves_nonparameter_full_policy_configuration():
    problem, policy = fixture()
    curvature = FullActionGGN(policy, problem, diagnostic='squared')
    direction = torch.ones(curvature.size, dtype=torch.float64)
    expected = curvature.product(direction)
    policy.affine_only = True
    torch.testing.assert_close(curvature.product(direction), expected, rtol=0., atol=0.)
    with pytest.raises(ValueError, match='snapshot'):
        propose_full_ggn_step(curvature=curvature, gradient=direction, metric=AffineBlockMetric(policy, problem))


def test_nonfinite_entropy_is_rejected_without_filtering_or_model_changes():
    problem, policy = fixture()
    with torch.no_grad():
        policy.affine.I0.bias.fill_(1.e5)
    before = deepcopy(policy.state_dict())
    curvature = FullActionGGN(policy, problem, diagnostic='entropy')
    with pytest.raises(FloatingPointError, match='nonfinite'):
        curvature.product(torch.ones(curvature.size, dtype=torch.float64))
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0., atol=0.)


def test_work_limit_and_bad_geometry_never_certify_convergence():
    problem, policy = fixture()
    full_action_pass(policy, problem, diagnostic='squared', backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    curvature = FullActionGGN(policy, problem, diagnostic='squared')
    metric = AffineBlockMetric(policy, problem)
    _, report = propose_full_ggn_step(curvature=curvature, gradient=gradient, metric=metric,
        config=GGNStepConfig(maximum_cg_iterations=1, cg_relative_tolerance=1.e-12))
    assert not report['linear_solve']['converged']
    assert not report['outer_gradient_convergence_certified']
    with pytest.raises(ValueError, match='shape'):
        curvature.product(gradient[:-1])
    with pytest.raises(ValueError, match='nonzero'):
        propose_full_ggn_step(curvature=curvature, gradient=torch.zeros_like(gradient), metric=metric)
    with pytest.raises(ValueError, match='damping'):
        GGNStepConfig(damping=0.)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_canonical_acceptance_commits_only_verified_full_policy_state(diagnostic):
    problem, policy = fixture()
    full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    before = _flatten({name: value.detach() for name, value in policy.named_parameters()}).clone()
    report, increment, change = accept_full_ggn_direction(policy=policy, problem=problem,
        direction=-gradient, diagnostic=diagnostic)
    assert report['accepted']
    assert report['verified_training_loss'] < report['initial_loss']
    assert not report['outer_gradient_convergence_certified']
    torch.testing.assert_close(_flatten({name: value.detach() for name, value in policy.named_parameters()}),
        before+increment, rtol=1.e-14, atol=1.e-14)
    actual_gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    torch.testing.assert_close(actual_gradient, gradient+change)
    actual_loss = full_action_pass(policy, problem, diagnostic=diagnostic, backward=False)
    assert report['verified_training_loss'] == pytest.approx(float(actual_loss), abs=1.e-14)


def test_rejected_nonfinite_trials_restore_original_weights_and_gradient(monkeypatch):
    problem, policy = fixture()
    full_action_pass(policy, problem, diagnostic='squared', backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()}).clone()
    before = deepcopy(policy.state_dict())
    original = ggn_audit_optimization.full_action_pass
    calls = 0

    def nonfinite_trial(*arguments, **keywords):
        nonlocal calls
        calls += 1
        return original(*arguments, **keywords) if calls == 1 else torch.tensor(float('inf'), dtype=torch.float64)

    monkeypatch.setattr(ggn_audit_optimization, 'full_action_pass', nonfinite_trial)
    report, increment, change = accept_full_ggn_direction(policy=policy, problem=problem,
        direction=-gradient, diagnostic='squared', minimum_step=.5)
    assert not report['accepted']
    assert report['full_objective_passes'] == 3
    assert report['stop_reason'] == 'line_search_not_converged'
    assert not increment.any() and not change.any()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0., atol=0.)
    torch.testing.assert_close(_flatten({name: value.grad for name, value in policy.named_parameters()}),
        gradient, rtol=0., atol=0.)


def test_interrupted_trial_restores_native_accepted_state(monkeypatch):
    problem, policy = fixture()
    full_action_pass(policy, problem, diagnostic='squared', backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()}).clone()
    before = deepcopy(policy.state_dict())
    original = ggn_audit_optimization.full_action_pass
    calls = 0

    def interrupted_trial(*arguments, **keywords):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise InterruptedError('uncommitted GGN trial')
        return original(*arguments, **keywords)

    monkeypatch.setattr(ggn_audit_optimization, 'full_action_pass', interrupted_trial)
    with pytest.raises(InterruptedError):
        accept_full_ggn_direction(policy=policy, problem=problem, direction=-gradient, diagnostic='squared')
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0., atol=0.)
    torch.testing.assert_close(_flatten({name: value.grad for name, value in policy.named_parameters()}),
        gradient, rtol=0., atol=0.)


def test_interrupted_initial_gradient_restores_entry_gradient_buffers(monkeypatch):
    problem, policy = fixture()
    parameters = dict(policy.named_parameters())
    next(iter(parameters.values())).grad = torch.ones_like(next(iter(parameters.values())))
    entry = {name: None if value.grad is None else value.grad.clone() for name, value in parameters.items()}
    before = deepcopy(policy.state_dict())

    def interrupted_baseline(*arguments, **keywords):
        next(iter(parameters.values())).sum().backward()
        raise InterruptedError('partial baseline accumulation')

    monkeypatch.setattr(ggn_audit_optimization, 'full_action_pass', interrupted_baseline)
    with pytest.raises(InterruptedError):
        accept_full_ggn_direction(policy=policy, problem=problem, diagnostic='squared',
            direction=torch.ones(sum(value.numel() for value in parameters.values()), dtype=torch.float64))
    for name, value in parameters.items():
        torch.testing.assert_close(value, before[name], rtol=0., atol=0.)
        if entry[name] is None:
            assert value.grad is None
        else:
            torch.testing.assert_close(value.grad, entry[name], rtol=0., atol=0.)


def test_rounded_zero_increment_is_not_an_accepted_update():
    problem, policy = fixture()
    full_action_pass(policy, problem, diagnostic='squared', backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    before = deepcopy(policy.state_dict())
    report, increment, _ = accept_full_ggn_direction(policy=policy, problem=problem,
        diagnostic='squared', direction=-1.e-300*gradient)
    assert not report['accepted']
    assert not increment.any()
    assert report['full_objective_passes'] == 1
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0., atol=0.)
