from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.full_hessian_audit import FullActionHessian
from quantdata.gromm2.ggn_audit_optimization import (
    FullActionGGN,
    GGNStepConfig,
    propose_full_ggn_step,
)
from quantdata.gromm2.gradient_reliable_ggn import (
    GradientReliabilityConfig,
    accept_gradient_reliable_direction,
)
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, _flatten_parameters
from quantdata.gromm2.neural_audit_optimization import full_action_pass
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
        config=LocalActionConfig(label='gradient-reliability-test', feature_count=2, instrument_count=2,
            action_scale=.4, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(std=.3)
    return problem, policy


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_complete_objective_acceptance_backtracks_for_actual_nonlinear_gradient_error(diagnostic):
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    initial = float(full_action_pass(policy, problem, diagnostic=diagnostic, backward=True))
    gradient = _flatten_parameters(tuple(policy.parameters()), gradients=True)
    direction, _ = propose_full_ggn_step(curvature=FullActionGGN(policy, problem, diagnostic=diagnostic),
        gradient=gradient, metric=metric, config=GGNStepConfig(damping=1.e-4, maximum_cg_iterations=100))
    exact = FullActionHessian(policy, problem, diagnostic=diagnostic)
    original = _flatten_parameters(tuple(policy.parameters()))
    rng = torch.get_rng_state().clone()
    report, increment, change = accept_gradient_reliable_direction(policy=policy, problem=problem,
        direction=direction, exact=exact, metric=metric, diagnostic=diagnostic,
        config=GradientReliabilityConfig(initial_rate=1., nonlinear_fraction=.01, expansion_fraction=.001))
    assert report['accepted'] and report['rate'] < 1.
    assert report['verified_training_loss'] < initial
    assert any(trial['armijo_passed'] and not trial['gradient_reliability_passed'] for trial in report['trials'])
    assert report['trials'][-1]['nonlinear_fraction_of_initial_gradient'] <= .01
    torch.testing.assert_close(increment, _flatten_parameters(tuple(policy.parameters()))-original, rtol=0., atol=0.)
    torch.testing.assert_close(change, _flatten_parameters(tuple(policy.parameters()), gradients=True)-gradient, rtol=0., atol=0.)
    assert exact.full_curvature_passes == report['full_exact_hessian_passes']
    assert report['full_objective_passes'] == len(report['trials'])+1
    assert torch.equal(rng, torch.get_rng_state())
    assert not report['outer_gradient_convergence_certified'] and not report['gradient_contraction_guaranteed']


@pytest.mark.parametrize('interrupt', [False, True])
def test_failed_or_interrupted_gradient_reliability_restores_exact_parameters_and_entry_gradients(interrupt):
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    full_action_pass(policy, problem, diagnostic='squared', backward=True)
    gradient = _flatten_parameters(tuple(policy.parameters()), gradients=True)
    direction = -metric(gradient)
    policy.zero_grad(set_to_none=True)
    native = deepcopy(policy.state_dict())
    exact = FullActionHessian(policy, problem, diagnostic='squared')

    def progress(entry):
        if interrupt:
            raise InterruptedError('test checkpoint stop')

    arguments = dict(policy=policy, problem=problem, direction=direction, exact=exact,
        metric=metric, diagnostic='squared', progress=progress,
        config=GradientReliabilityConfig(initial_rate=1., minimum_rate=1.,
            nonlinear_fraction=1.e-12, expansion_fraction=1.e-13))
    if interrupt:
        with pytest.raises(InterruptedError):
            accept_gradient_reliable_direction(**arguments)
    else:
        report, increment, change = accept_gradient_reliable_direction(**arguments)
        assert not report['accepted'] and report['original_state_restored']
        assert not increment.any() and not change.any()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, native[name], rtol=0., atol=0.)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_changed_initial_snapshot_rejected_before_any_full_pass():
    problem, policy = fixture()
    metric = AffineBlockMetric(policy, problem, deep_scale=.2)
    exact = FullActionHessian(policy, problem, diagnostic='squared')
    with torch.no_grad():
        policy.affine.I0.bias.add_(.01)
    with pytest.raises(ValueError, match='unchanged initial'):
        accept_gradient_reliable_direction(policy=policy, problem=problem,
            direction=torch.ones(metric.size, dtype=torch.float64), exact=exact, metric=metric, diagnostic='squared')
    assert exact.full_curvature_passes == 0
