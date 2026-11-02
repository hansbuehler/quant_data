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
from quantdata.gromm2.full_hessian_audit import (
    FullActionHessian,
    compare_full_curvature,
    decompose_gradient_change,
    diagnose_accepted_gradient_change,
)
from quantdata.gromm2.ggn_audit_optimization import FullActionGGN, _flatten, _unpack
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric
from quantdata.torch import BaseEnvironment


def fixture():
    torch.manual_seed(612)
    states = torch.randn(14, 2, dtype=torch.float32)
    returns = .3*states.double().square()+.1*states.double()
    states[12:], returns[12:] = float('nan'), float('nan')
    problem = AffineAuditProblem(states=states, returns=returns,
        eligible=torch.tensor([[True, True], [True, False]]), rows=torch.arange(12),
        action_scale=.4, config=AffineSolveConfig(batch_size=5))
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='exact-curvature-test', feature_count=2, instrument_count=2,
            action_scale=.4, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(std=.3)
    return problem, policy


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_full_hessian_matches_dense_autograd_symmetry_and_finite_differences(diagnostic):
    problem, policy = fixture()
    native = deepcopy(policy.state_dict())
    rng = torch.get_rng_state().clone()
    parameters = {name: value.detach() for name, value in policy.named_parameters()}
    point = _flatten(parameters)
    features = problem.states[problem.rows].double()
    eligible = problem.eligible[problem.rows % len(problem.eligible)]

    def loss(values):
        action = torch.func.functional_call(policy, _unpack(values, parameters), (features, eligible), strict=True)
        return local_objective(action, problem.returns[problem.rows], diagnostic=diagnostic,
            action_scale=problem.action_scale, squared_lambda=problem.squared_lambda)

    matrix = torch.autograd.functional.hessian(loss, point)
    direction = torch.sin(torch.arange(len(point), dtype=torch.float64))
    other = torch.cos(torch.arange(len(point), dtype=torch.float64))
    exact = FullActionHessian(policy, problem, diagnostic=diagnostic)
    actual = exact.product(direction)
    torch.testing.assert_close(actual, matrix@direction, rtol=1.e-11, atol=1.e-12)
    torch.testing.assert_close(torch.dot(other, actual), torch.dot(direction, exact.product(other)),
        rtol=1.e-11, atol=1.e-12)
    finite_difference = (torch.func.grad(loss)(point+1.e-5*direction)-torch.func.grad(loss)(point-1.e-5*direction))/(2.e-5)
    torch.testing.assert_close(actual, finite_difference, rtol=1.e-7, atol=1.e-9)
    assert torch.linalg.eigvalsh(matrix).min() < -1.e-4
    assert exact.full_curvature_passes == 2
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(parameter.grad is None for parameter in policy.parameters())
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, native[name], rtol=0., atol=0.)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_omitted_neural_curvature_is_a_full_vector_and_reports_exact_work(diagnostic):
    problem, policy = fixture()
    exact = FullActionHessian(policy, problem, diagnostic=diagnostic)
    ggn = FullActionGGN(policy, problem, diagnostic=diagnostic)
    direction = torch.sin(torch.arange(exact.size, dtype=torch.float64))
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    report = compare_full_curvature(exact=exact, ggn=ggn, direction=direction, metric=metric)
    assert report['omitted_neural_curvature']['full_gradient_norm'] > 1.e-3
    assert report['additional_exact_hessian_passes'] == exact.full_curvature_passes == 1
    assert report['additional_ggn_passes'] == ggn.full_curvature_passes == 1
    assert not report['optimization_performed']
    assert not report['outer_convergence_certified']
    with torch.no_grad():
        policy.affine.I0.bias.add_(.01)
    other = FullActionGGN(policy, problem, diagnostic=diagnostic)
    with pytest.raises(ValueError, match='identical snapshots'):
        compare_full_curvature(exact=exact, ggn=other, direction=direction, metric=metric)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_directional_agreement_can_hide_substantial_off_direction_neural_curvature(diagnostic):
    problem, policy = fixture()
    parameters = {name: value.detach() for name, value in policy.named_parameters()}
    point = _flatten(parameters)
    features = problem.states[problem.rows].double()
    eligible = problem.eligible[problem.rows % len(problem.eligible)]

    def actions(values):
        return torch.func.functional_call(policy, _unpack(values, parameters), (features, eligible), strict=True)

    def loss(action):
        return local_objective(action, problem.returns[problem.rows], diagnostic=diagnostic,
            action_scale=problem.action_scale, squared_lambda=problem.squared_lambda)

    action = actions(point)
    jacobian = torch.func.jacrev(actions)(point).reshape(action.numel(), -1)
    output_hessian = torch.autograd.functional.hessian(lambda value: loss(value.reshape_as(action)), action.flatten())
    exact_matrix = torch.autograd.functional.hessian(lambda values: loss(actions(values)), point)
    omitted_matrix = exact_matrix-jacobian.T@output_hessian@jacobian
    eigenvalues, vectors = torch.linalg.eigh(omitted_matrix)
    assert eigenvalues[0] < -1.e-4 and eigenvalues[-1] > 1.e-4
    direction = torch.sqrt(-eigenvalues[0])*vectors[:, -1]+torch.sqrt(eigenvalues[-1])*vectors[:, 0]
    direction /= torch.linalg.vector_norm(direction)
    omitted = (FullActionHessian(policy, problem, diagnostic=diagnostic).product(direction)
        -FullActionGGN(policy, problem, diagnostic=diagnostic).product(direction))
    assert abs(float(torch.dot(direction, omitted))) < 1.e-12
    assert torch.linalg.vector_norm(omitted) > .01


@pytest.mark.parametrize('field', ['states', 'returns', 'rows', 'eligible'])
def test_full_hessian_refuses_changed_observations_or_partition(field):
    problem, policy = fixture()
    exact = FullActionHessian(policy, problem, diagnostic='squared')
    value = getattr(problem, field)
    value.flatten()[0] = 0
    with pytest.raises(ValueError, match='data or objective changed'):
        exact.product(torch.ones(exact.size, dtype=torch.float64))


def test_compiled_exact_hessian_uses_functional_second_derivatives(monkeypatch):
    problem, policy = fixture()
    original_compile = torch.compile
    monkeypatch.setattr(torch, 'compile', lambda function: original_compile(function, backend='eager'))
    eager = FullActionHessian(policy, problem, diagnostic='squared')
    compiled = FullActionHessian(policy, problem, diagnostic='squared', compile=True)
    direction = torch.sin(torch.arange(eager.size, dtype=torch.float64))
    torch.testing.assert_close(compiled.product(direction), eager.product(direction), rtol=1.e-12, atol=1.e-13)


def test_same_weights_with_different_frozen_forward_functions_cannot_be_compared():
    problem, policy = fixture()
    exact = FullActionHessian(policy, problem, diagnostic='squared')
    policy.deep.use_final_activation = True
    changed_forward = FullActionGGN(policy, problem, diagnostic='squared')
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    with pytest.raises(ValueError, match='identical snapshots'):
        compare_full_curvature(exact=exact, ggn=changed_forward,
            direction=torch.ones(exact.size, dtype=torch.float64), metric=metric)
    assert exact.full_curvature_passes == changed_forward.full_curvature_passes == 0


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_accepted_gradient_decomposition_measures_full_vectors_without_mutating_policy(diagnostic):
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    parameters = {name: value.detach() for name, value in policy.named_parameters()}
    point = _flatten(parameters)
    features = problem.states[problem.rows].double()
    eligible = problem.eligible[problem.rows % len(problem.eligible)]

    def loss(values):
        action = torch.func.functional_call(policy, _unpack(values, parameters), (features, eligible), strict=True)
        return local_objective(action, problem.returns[problem.rows], diagnostic=diagnostic,
            action_scale=problem.action_scale, squared_lambda=problem.squared_lambda)

    direction = .02*torch.sin(torch.arange(len(point), dtype=torch.float64))
    initial_gradient = torch.func.grad(loss)(point)
    final_gradient = torch.func.grad(loss)(point+direction)
    exact = FullActionHessian(policy, problem, diagnostic=diagnostic)
    ggn = FullActionGGN(policy, problem, diagnostic=diagnostic)
    native, rng = deepcopy(policy.state_dict()), torch.get_rng_state().clone()
    report = diagnose_accepted_gradient_change(exact=exact, ggn=ggn,
        initial_gradient=initial_gradient, final_gradient=final_gradient,
        direction=direction, damping=.1, metric=metric)
    assert report['arithmetic_identity_error']['full_gradient_norm'] < 1.e-14
    assert report['vectors']['omitted_neural_curvature']['full_gradient_norm'] > 1.e-5
    assert report['vectors']['nonlinear_gradient_remainder']['full_gradient_norm'] > 1.e-6
    assert report['arithmetic_identity_relative_raw_error'] < 1.e-12
    gram = torch.tensor(report['full_metric_pairwise']['gram'], dtype=torch.float64)
    torch.testing.assert_close(gram, gram.T, rtol=1.e-13, atol=1.e-14)
    for index, name in enumerate(report['full_metric_pairwise']['names']):
        assert float(gram[index, index]) == pytest.approx(report['vectors'][name]['full_metric_gradient_norm']**2)
        assert report['full_metric_pairwise']['cosines'][index][index] == pytest.approx(1.)
    inverse = report['inverse_metric_crosscheck']
    assert inverse['production_minus_factor_solve']['full_gradient_norm'] < 1.e-14
    assert inverse['production_step_energy'] == pytest.approx(inverse['factor_solve_step_energy'])
    for norms in report['vectors'].values():
        assert norms['full_metric_gradient_norm']**2 == pytest.approx(
            norms['affine_metric_gradient_norm']**2+.2*norms['deep_gradient_norm']**2)
    assert report['additional_exact_hessian_passes'] == exact.full_curvature_passes == 1
    assert report['additional_ggn_passes'] == ggn.full_curvature_passes == 1
    assert report['identity_is_arithmetic_not_independent_validation']
    assert not report['optimization_performed'] and not report['outer_convergence_certified']
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(parameter.grad is None for parameter in policy.parameters())
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, native[name], rtol=0., atol=0.)


def test_gradient_decomposition_rejects_bad_geometry_nonfinite_values_and_zero_increment():
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem)
    arguments = {name: torch.ones(metric.size, dtype=torch.float64)
        for name in ('initial_gradient', 'final_gradient', 'direction', 'hessian_image', 'ggn_image')}
    with pytest.raises(ValueError, match='positive damping'):
        decompose_gradient_change(**arguments, damping=0., metric=metric)
    with pytest.raises(ValueError, match='vector geometry'):
        decompose_gradient_change(**{**arguments, 'direction': arguments['direction'].float()}, damping=.1, metric=metric)
    with pytest.raises(FloatingPointError, match='nonfinite'):
        decompose_gradient_change(**{**arguments, 'final_gradient': arguments['final_gradient']*float('nan')},
            damping=.1, metric=metric)
    with pytest.raises(ValueError, match='nonzero actual'):
        decompose_gradient_change(**{**arguments, 'direction': torch.zeros_like(arguments['direction'])},
            damping=.1, metric=metric)
