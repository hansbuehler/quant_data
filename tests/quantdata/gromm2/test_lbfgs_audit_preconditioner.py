from __future__ import annotations

import pytest
import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.ggn_audit_optimization import FullActionGGN, InverseAffineMetric
from quantdata.gromm2.gradient_reliable_ggn import verify_actual_metric
from quantdata.gromm2.lbfgs_audit_preconditioner import (
    FrozenLBFGSPreconditioner,
    metric_energy,
    pcg_separate_residual_norm,
)
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.torch import BaseEnvironment


def fixture():
    torch.manual_seed(321)
    problem = AffineAuditProblem(states=torch.randn(12, 2), returns=torch.randn(12, 2),
        eligible=torch.ones(2, 2, dtype=torch.bool), rows=torch.arange(12),
        action_scale=.2, config=AffineSolveConfig(batch_size=5))
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='positive-preconditioner-test', feature_count=2, instrument_count=2,
            action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.2), history_size=5, curvature_floor=1.e-12)
    for _ in range(4):
        step = torch.randn(optimizer.metric.size, dtype=torch.float64)
        change = torch.arange(1, len(step)+1, dtype=torch.float64)*step
        assert optimizer.accept(step, change)
    return optimizer, policy, problem


def test_frozen_positive_energy_matches_dense_bfgs_and_native_two_loop():
    optimizer, policy, _ = fixture()
    frozen = FrozenLBFGSPreconditioner(optimizer, policy)
    verify_actual_metric(frozen.metric, policy)
    assert frozen.metric.factors is not optimizer.metric.factors
    assert all(actual is original for actual, original in zip(frozen.metric.parameters, policy.parameters(), strict=True))
    size = optimizer.metric.size
    basis = torch.eye(size, dtype=torch.float64)
    dense = torch.stack([optimizer.metric(value) for value in basis], dim=1)
    for step, change, reciprocal in optimizer.history:
        update = basis-reciprocal*step[:, None]*change[None, :]
        dense = update@dense@update.T+reciprocal*step[:, None]*step[None, :]
    vector = torch.sin(torch.arange(size, dtype=torch.float64))
    actual, energy = frozen.apply_and_energy(vector)
    torch.testing.assert_close(actual, dense@vector, rtol=1.e-11, atol=1.e-12)
    torch.testing.assert_close(actual, -optimizer.direction(vector), rtol=1.e-13, atol=1.e-14)
    torch.testing.assert_close(energy, vector@dense@vector, rtol=1.e-11, atol=1.e-12)
    assert torch.linalg.eigvalsh(dense).min() > 0.
    optimizer.history.clear()
    torch.testing.assert_close(frozen.apply_and_energy(vector)[0], actual, rtol=0., atol=0.)
    frozen.history[0][0].add_(.1)
    with pytest.raises(ValueError, match='frozen positive'):
        frozen.apply_and_energy(vector)


def test_separate_stopping_geometry_solves_the_same_spd_system():
    generator = torch.Generator().manual_seed(123)
    factor = torch.randn(8, 8, dtype=torch.float64, generator=generator)
    system = factor@factor.T+.2*torch.eye(8, dtype=torch.float64)
    stopping = torch.diag(torch.logspace(-3, 3, 8, dtype=torch.float64))
    inverse = torch.linalg.inv(system)
    right = torch.randn(8, dtype=torch.float64, generator=generator)
    operator_calls = []

    def operator(value):
        operator_calls.append(value.clone())
        return system@value

    result, report = pcg_separate_residual_norm(operator, right,
        lambda value: (inverse@value, value@inverse@value), stopping_energy=lambda value: value@stopping@value,
        relative_tolerance=1.e-10, maximum_iterations=20)
    torch.testing.assert_close(result, torch.linalg.solve(system, right), rtol=1.e-11, atol=1.e-12)
    assert report['converged'] and report['iterations'] == 1
    assert len(operator_calls) == 2
    assert report['original_metric_initial_residual_norm'] == pytest.approx(float(torch.sqrt(right@stopping@right)))
    assert not report['outer_gradient_convergence_certified']


def test_original_metric_true_residual_cannot_be_replaced_by_preconditioner_energy():
    system = torch.diag(torch.tensor([1., 10.], dtype=torch.float64))
    right = torch.ones(2, dtype=torch.float64)
    preconditioner = torch.diag(torch.tensor([1., 1.e-10], dtype=torch.float64))
    result, report = pcg_separate_residual_norm(lambda value: system@value, right,
        lambda value: (preconditioner@value, value@preconditioner@value), stopping_energy=lambda value: value.square().sum(),
        relative_tolerance=.1, maximum_iterations=1)
    assert not report['converged']
    assert report['original_metric_residual_norm'] > .9
    torch.testing.assert_close(report['euclidean_residual_norm'], float(torch.linalg.vector_norm(right-system@result)))


def test_factor_energy_and_invalid_recurrence_curvature():
    optimizer, policy, _ = fixture()
    vector = torch.cos(torch.arange(optimizer.metric.size, dtype=torch.float64))
    assert float(metric_energy(optimizer.metric, vector)) == pytest.approx(float(torch.dot(vector, optimizer.metric(vector))))
    with pytest.raises(RuntimeError, match='positive recurrence'):
        pcg_separate_residual_norm(lambda value: -value, torch.ones(2, dtype=torch.float64),
            lambda value: (value, value.square().sum()), stopping_energy=lambda value: value.square().sum(),
            relative_tolerance=.1, maximum_iterations=2)


@pytest.mark.parametrize('mutation', ['factor', 'scale', 'columns'])
def test_frozen_geometry_guard_also_protects_base_and_damping_operators(mutation):
    optimizer, policy, _ = fixture()
    frozen = FrozenLBFGSPreconditioner(optimizer, policy)
    frozen.check_fixed()
    if mutation == 'factor':
        frozen.metric.factors.add_(.1)
    elif mutation == 'scale':
        frozen.metric.deep_scale *= 2.
    else:
        frozen.metric.columns[0].add_(1)
    with pytest.raises(ValueError, match='frozen positive'):
        frozen.check_fixed()


def test_functional_ggn_products_remain_exact_under_no_grad_solver_context():
    optimizer, policy, problem = fixture()
    curvature = FullActionGGN(policy, problem, diagnostic='squared')
    inverse = InverseAffineMetric(optimizer.metric)
    direction = torch.sin(torch.arange(curvature.size, dtype=torch.float64))
    expected = curvature.product(direction)
    with torch.no_grad():
        actual = curvature.product(direction)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    frozen = FrozenLBFGSPreconditioner(optimizer, policy)
    result, report = pcg_separate_residual_norm(lambda value: curvature.product(value)+.1*inverse(value),
        direction, frozen.apply_and_energy, stopping_energy=lambda value: metric_energy(optimizer.metric, value),
        relative_tolerance=1.e-7, maximum_iterations=100)
    residual = direction-curvature.product(result)-.1*inverse(result)
    assert report['converged']
    assert float(torch.sqrt(metric_energy(optimizer.metric, residual))) <= report['original_metric_threshold']*1.0001
