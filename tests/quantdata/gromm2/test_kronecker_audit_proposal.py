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
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256, optimizer_state_sha256
from quantdata.gromm2.ggn_audit_optimization import _flatten
from quantdata.gromm2.kronecker_audit_preconditioner import FrozenKroneckerLBFGS
from quantdata.gromm2.kronecker_audit_proposal import propose_kronecker_gmres_step
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.minimum_residual_audit_proposal import MinimumResidualConfig
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.torch import BaseEnvironment


def fixture(diagnostic='squared'):
    generator = torch.Generator().manual_seed(1013)
    with torch.random.fork_rng():
        torch.manual_seed(1014)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='kronecker-proposal-test', feature_count=2, instrument_count=2,
                action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    problem = AffineAuditProblem(states=torch.randn((12, 2), generator=generator),
        returns=.1*torch.randn((12, 2), generator=generator)+.2,
        eligible=torch.ones((2, 2), dtype=torch.bool), rows=torch.arange(12),
        action_scale=.2, config=AffineSolveConfig(batch_size=4))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.1), history_size=3, curvature_floor=1.e-12)
    step = torch.randn(optimizer.metric.size, generator=generator, dtype=torch.float64)
    assert optimizer.accept(step, 1.3*step)
    moments = {name: (torch.eye(module.in_features+1, dtype=torch.float64),
        torch.eye(module.out_features, dtype=torch.float64)) for name, module in policy.named_modules()
        if name.startswith('deep.') and type(module) is torch.nn.Linear}
    preconditioner = FrozenKroneckerLBFGS(optimizer, policy, moments=moments, factor_data_identity='a'*64)
    full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    return policy, problem, optimizer, preconditioner, gradient


def install_quadratic(monkeypatch, multiplier, mutation=None):
    import quantdata.gromm2.kronecker_audit_proposal as implementation

    class QuadraticExact:
        def __init__(self, policy, problem, **kwargs):
            self.functional_policy = deepcopy(policy)
            self.parameters = {name: value.detach().clone() for name, value in policy.named_parameters()}
            self.buffers = {name: value.detach().clone() for name, value in policy.named_buffers()}
            self.full_curvature_passes = 0

        def product(self, value):
            self.full_curvature_passes += 1
            if mutation is not None:
                mutation(self)
            return multiplier*value

    monkeypatch.setattr(implementation, 'FullActionHessian', QuadraticExact)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_real_curvature_proposal_preserves_original_policy_history_gradients_and_rng(diagnostic):
    policy, problem, optimizer, preconditioner, gradient = fixture(diagnostic)
    point, history, rng = action_point_sha256(policy), optimizer_state_sha256(optimizer), torch.get_rng_state().clone()
    gradients = {name: value.grad.clone() for name, value in policy.named_parameters()}
    direction, report = propose_kronecker_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        preconditioner=preconditioner, gradient=gradient, diagnostic=diagnostic, point_sha256=point,
        config=MinimumResidualConfig(maximum_hessian_products=80, compile=False))
    assert report['full_curvature_passes'] <= 80
    assert optimizer_state_sha256(optimizer) == history and action_point_sha256(policy) == point
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value.grad, gradients[name]) for name, value in policy.named_parameters())
    assert report['original_stopping_metric_unchanged'] and report['factor_collection_work_requires_separate_accounting']
    assert not report['SPD_Hessian_assumed'] and not report['outer_gradient_convergence_certified']
    if direction is not None:
        assert report['proposal_selected'] and report['linear_solve']['converged'] and report['slope'] < 0.
    else:
        assert not report['proposal_selected']


def test_positive_dense_quadratic_produces_a_nonzero_actual_quality_checked_direction(monkeypatch):
    install_quadratic(monkeypatch, 1.)
    policy, problem, optimizer, preconditioner, gradient = fixture()
    direction, report = propose_kronecker_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        preconditioner=preconditioner, gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=MinimumResidualConfig(relative_tolerance=1.e-10, maximum_hessian_products=80, compile=False))
    assert direction is not None and report['linear_solve']['converged']
    torch.testing.assert_close(direction, -gradient, rtol=1.e-10, atol=1.e-10)
    assert report['slope'] < 0. and report['numerical_shift'] == 0. and report['optimizer_updates'] == 0


def test_negative_curvature_stationary_root_is_not_accepted_when_not_descent(monkeypatch):
    install_quadratic(monkeypatch, -1.)
    policy, problem, optimizer, preconditioner, gradient = fixture()
    direction, report = propose_kronecker_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        preconditioner=preconditioner, gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=MinimumResidualConfig(relative_tolerance=1.e-10, maximum_hessian_products=80, compile=False))
    assert direction is None and not report['proposal_selected']
    assert report['linear_solve']['converged'] and report['slope'] > 0.


def test_actual_residual_confirmation_stays_inside_the_shared_product_budget(monkeypatch):
    install_quadratic(monkeypatch, 1.)
    policy, problem, optimizer, preconditioner, gradient = fixture()
    direction, report = propose_kronecker_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        preconditioner=preconditioner, gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=MinimumResidualConfig(maximum_hessian_products=1, compile=False))
    assert direction is None and report['status'] == 'shared_hessian_work_limit_not_converged'
    assert report['full_curvature_passes'] == 1 and not report['outer_gradient_convergence_certified']


@pytest.mark.parametrize('case', ['point', 'history', 'gradient', 'affine_only'])
def test_changed_source_is_rejected_before_curvature_work(case):
    policy, problem, optimizer, preconditioner, gradient = fixture()
    if case == 'history':
        assert optimizer.accept(torch.ones_like(gradient), torch.ones_like(gradient))
    elif case == 'gradient':
        gradient.zero_()
    elif case == 'affine_only':
        policy.affine_only = True
    with pytest.raises(ValueError, match='same original full-policy'):
        propose_kronecker_gmres_step(policy=policy, problem=problem, optimizer=optimizer, preconditioner=preconditioner,
            gradient=gradient, diagnostic='squared', point_sha256='0'*64 if case == 'point' else action_point_sha256(policy),
            config=MinimumResidualConfig(compile=False))


@pytest.mark.parametrize('case', ['snapshot', 'base_identity'])
def test_mid_product_snapshot_or_factor_identity_mutation_is_rejected(monkeypatch, case):
    policy, problem, optimizer, preconditioner, gradient = fixture()

    def mutate(exact):
        if case == 'snapshot':
            exact.functional_policy.affine_only = True
        else:
            preconditioner.identity['factor_data_identity'] = 'b'*64

    install_quadratic(monkeypatch, 1., mutation=mutate)
    with pytest.raises(ValueError, match='snapshot changed'):
        propose_kronecker_gmres_step(policy=policy, problem=problem, optimizer=optimizer, preconditioner=preconditioner,
            gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
            config=MinimumResidualConfig(compile=False))
