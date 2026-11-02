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
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256, optimizer_state_sha256
from quantdata.gromm2.ggn_audit_optimization import _flatten
from quantdata.gromm2.gradient_reliable_ggn import (
    GradientReliabilityConfig,
    accept_gradient_reliable_direction,
)
from quantdata.gromm2.history_audit_proposal import propose_positive_history_step
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.torch import BaseEnvironment


def prepared(diagnostic):
    generator = torch.Generator().manual_seed(12)
    states = torch.randn((16, 2), generator=generator, dtype=torch.float32)
    returns = torch.randn((16, 2), generator=generator, dtype=torch.float32)*.1+.2
    problem = AffineAuditProblem(states=states, returns=returns, eligible=torch.ones((2, 2), dtype=torch.bool),
        rows=torch.arange(16), action_scale=.2, config=AffineSolveConfig(batch_size=5))
    with torch.random.fork_rng():
        torch.manual_seed(13)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='history-proposal', feature_count=2, instrument_count=2,
                action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.1), history_size=3, curvature_floor=1.e-12)
    full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    for unused in range(3):
        step = torch.randn(gradient.shape, generator=generator, dtype=torch.float64)
        change = step*.3
        assert optimizer.accept(step, change)
    return policy, problem, optimizer, gradient


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_direct_positive_history_matches_native_direction_without_mutating_any_state(diagnostic):
    policy, problem, optimizer, gradient = prepared(diagnostic)
    original_gradient = gradient.clone()
    point, optimizer_sha = action_point_sha256(policy), optimizer_state_sha256(optimizer)
    rng = torch.get_rng_state().clone()
    direction, report = propose_positive_history_step(policy=policy, optimizer=optimizer,
        gradient=gradient, point_sha256=point)
    torch.testing.assert_close(direction, optimizer.direction(gradient), rtol=1.e-12, atol=1.e-13)
    assert torch.dot(gradient, direction) < 0.
    assert report['full_curvature_passes'] == report['optimizer_updates'] == 0
    assert not report['outer_gradient_convergence_certified']
    assert optimizer_state_sha256(optimizer) == optimizer_sha and action_point_sha256(policy) == point
    assert torch.equal(gradient, original_gradient) and torch.equal(torch.get_rng_state(), rng)
    exact = FullActionHessian(policy, problem, diagnostic=diagnostic)
    result, increment, change = accept_gradient_reliable_direction(policy=policy, problem=problem,
        direction=direction, exact=exact, metric=optimizer.metric, diagnostic=diagnostic,
        config=GradientReliabilityConfig(initial_rate=.25))
    assert result['accepted'] and result['verified_training_loss'] < result['initial_loss']
    assert result['trials'][-1]['nonlinear_fraction_of_initial_gradient'] <= .25
    assert result['full_exact_hessian_passes'] == exact.full_curvature_passes
    assert bool(increment.ne(0.).any()) and bool(torch.isfinite(change).all())
    assert optimizer_state_sha256(optimizer) == optimizer_sha


@pytest.mark.parametrize('case', ['zero', 'nonfinite', 'stale_point', 'affine_only', 'wrong_precision', 'changed_metric'])
def test_positive_history_proposal_rejects_unbound_or_invalid_inputs(case):
    policy, problem, optimizer, gradient = prepared('squared')
    point = action_point_sha256(policy)
    if case == 'zero':
        gradient.zero_()
    elif case == 'nonfinite':
        gradient[0] = float('nan')
    elif case == 'stale_point':
        point = '0'*64
    elif case == 'affine_only':
        policy.affine_only = True
    elif case == 'wrong_precision':
        gradient = gradient.float()
    else:
        optimizer.metric.deep_scale *= 2.
    with pytest.raises(ValueError):
        propose_positive_history_step(policy=policy, optimizer=optimizer, gradient=gradient, point_sha256=point)


def test_positive_history_proposal_rejects_nonpositive_native_secants():
    policy, problem, optimizer, gradient = prepared('squared')
    step, change, reciprocal = deepcopy(optimizer.history[-1])
    optimizer.history[-1] = (step, -change, -reciprocal)
    with pytest.raises(ValueError, match='positive secant'):
        propose_positive_history_step(policy=policy, optimizer=optimizer, gradient=gradient,
            point_sha256=action_point_sha256(policy))
