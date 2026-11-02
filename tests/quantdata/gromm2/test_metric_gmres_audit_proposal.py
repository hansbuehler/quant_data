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
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.metric_gmres_audit_proposal import (
    MinimumResidualConfig,
    TransposeAffineMetric,
    propose_metric_gmres_step,
)
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.torch import BaseEnvironment


def prepared(diagnostic):
    generator = torch.Generator().manual_seed(31)
    problem = AffineAuditProblem(states=torch.randn((12, 2), generator=generator, dtype=torch.float32),
        returns=.1*torch.randn((12, 2), generator=generator, dtype=torch.float32)+.2,
        eligible=torch.ones((2, 2), dtype=torch.bool), rows=torch.arange(12),
        action_scale=.2, config=AffineSolveConfig(batch_size=4))
    with torch.random.fork_rng():
        torch.manual_seed(32)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='metric-gmres', feature_count=2, instrument_count=2,
                action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.1), history_size=3, curvature_floor=1.e-12)
    full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    step = torch.randn(gradient.shape, generator=generator, dtype=torch.float64)
    assert optimizer.accept(step, .3*step)
    return policy, problem, optimizer, gradient


def install_quadratic(monkeypatch, multiplier, mutate=False):
    import quantdata.gromm2.metric_gmres_audit_proposal as implementation

    class QuadraticExact:
        def __init__(self, policy, problem, **kwargs):
            self.policy = policy
            self.functional_policy = deepcopy(policy)
            self.parameters = {name: value.detach().clone() for name, value in policy.named_parameters()}
            self.buffers = {name: value.detach().clone() for name, value in policy.named_buffers()}
            self.full_curvature_passes = 0

        def product(self, vector):
            self.full_curvature_passes += 1
            if mutate:
                self.functional_policy.affine_only = True
            return multiplier*vector

    monkeypatch.setattr(implementation, 'FullActionHessian', QuadraticExact)



@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_minimum_residual_proposal_preserves_native_state_and_reports_actual_gradients(diagnostic):
    policy, problem, optimizer, gradient = prepared(diagnostic)
    point, history, rng = action_point_sha256(policy), optimizer_state_sha256(optimizer), torch.get_rng_state().clone()
    saved_gradients = {name: value.grad.clone() for name, value in policy.named_parameters()}
    direction, report = propose_metric_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic=diagnostic, point_sha256=point,
        config=MinimumResidualConfig(maximum_hessian_products=80, compile=False))
    assert report['full_curvature_passes'] <= 80
    assert optimizer_state_sha256(optimizer) == history and action_point_sha256(policy) == point
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value.grad, saved_gradients[name]) for name, value in policy.named_parameters())
    assert not report['SPD_Hessian_assumed'] and not report['outer_gradient_convergence_certified']
    if direction is not None:
        assert report['proposal_selected'] and report['linear_solve']['converged']
        assert report['slope'] < 0.
        assert report['actual_unshifted_predicted_norms']['full_gradient_norm'] <= report['initial_norms']['full_gradient_norm']
        assert report['actual_unshifted_predicted_norms']['affine_metric_gradient_norm'] <= report['initial_norms']['affine_metric_gradient_norm']
    else:
        assert not report['proposal_selected']


def test_positive_quadratic_produces_a_nonzero_quality_checked_direction(monkeypatch):
    install_quadratic(monkeypatch, 1.)
    policy, problem, optimizer, gradient = prepared('squared')
    direction, report = propose_metric_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=MinimumResidualConfig(relative_tolerance=1.e-10, maximum_hessian_products=80, compile=False))
    assert direction is not None and bool(direction.ne(0.).any())
    assert report['proposal_selected'] and report['linear_solve']['converged']
    assert report['slope'] < 0. and report['numerical_shift'] == 0.
    torch.testing.assert_close(direction, -gradient, atol=1.e-10, rtol=1.e-10)


def test_transpose_only_embedding_never_computes_a_factor_inverse(monkeypatch):
    from quantdata.gromm2.lbfgs_audit_preconditioner import metric_energy

    policy, problem, optimizer, gradient = prepared('squared')

    def forbidden_inverse(*args, **kwargs):
        raise AssertionError('the original-M factor embedding must not compute an inverse')

    monkeypatch.setattr(torch.linalg, 'inv', forbidden_inverse)
    embedding = TransposeAffineMetric(optimizer.metric)
    transformed = embedding(gradient)
    torch.testing.assert_close(transformed.square().sum(), metric_energy(optimizer.metric, gradient), atol=1.e-15, rtol=1.e-13)
    install_quadratic(monkeypatch, 1.)
    direction, report = propose_metric_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=MinimumResidualConfig(maximum_hessian_products=80, compile=False))
    assert direction is not None and report['proposal_selected']


def test_transpose_only_embedding_rejects_changed_factor_objects():
    policy, problem, optimizer, gradient = prepared('squared')
    embedding = TransposeAffineMetric(optimizer.metric)
    optimizer.metric.factors = optimizer.metric.factors.clone()
    with pytest.raises(ValueError, match='factor geometry changed'):
        embedding(gradient)


def test_negative_quadratic_newton_root_is_rejected_when_not_descent(monkeypatch):
    install_quadratic(monkeypatch, -1.)
    policy, problem, optimizer, gradient = prepared('squared')
    direction, report = propose_metric_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=MinimumResidualConfig(relative_tolerance=1.e-10, maximum_hessian_products=80, compile=False))
    assert direction is None and not report['proposal_selected']
    assert report['linear_solve']['converged'] and report['slope'] > 0.
    assert not report['outer_gradient_convergence_certified']


def test_actual_confirmation_shares_the_declared_hessian_budget(monkeypatch):
    install_quadratic(monkeypatch, 1.)
    policy, problem, optimizer, gradient = prepared('squared')
    direction, report = propose_metric_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=MinimumResidualConfig(maximum_hessian_products=1, compile=False))
    assert direction is None and not report['proposal_selected']
    assert report['full_curvature_passes'] == 1 and report['status'] == 'shared_hessian_work_limit_not_converged'


def test_frozen_forward_mutation_during_product_is_rejected(monkeypatch):
    install_quadratic(monkeypatch, 1., mutate=True)
    policy, problem, optimizer, gradient = prepared('squared')
    with pytest.raises(ValueError, match='snapshot'):
        propose_metric_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
            gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
            config=MinimumResidualConfig(compile=False))


@pytest.mark.parametrize('case', ['stale_point', 'zero_gradient', 'affine_only', 'wrong_dtype'])
def test_minimum_residual_rejects_changed_source_before_products(case):
    policy, problem, optimizer, gradient = prepared('squared')
    point = action_point_sha256(policy)
    if case == 'stale_point':
        point = '0'*64
    elif case == 'zero_gradient':
        gradient.zero_()
    elif case == 'affine_only':
        policy.affine_only = True
    else:
        gradient = gradient.float()
    with pytest.raises(ValueError):
        propose_metric_gmres_step(policy=policy, problem=problem, optimizer=optimizer,
            gradient=gradient, diagnostic='squared', point_sha256=point,
            config=MinimumResidualConfig(compile=False))
