from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quantdata.gromm2.affine_audit_optimization import (
    AffineAuditProblem,
    AffineSolveConfig,
    preconditioned_cg,
)
from quantdata.gromm2.conditional_audit import local_objective


def problem(*, singular: bool = False, scale: float = .2) -> AffineAuditProblem:
    generator = torch.Generator().manual_seed(17)
    states = torch.randn((240, 4), generator=generator, dtype=torch.float64)
    if singular:
        states[:, 3] = states[:, 2]
    returns = .1 * states[:, :3] + .1 * torch.randn((240, 3), generator=generator, dtype=torch.float64)
    eligible = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    return AffineAuditProblem(states=states, returns=returns, eligible=eligible,
        rows=torch.arange(200), action_scale=scale,
        config=AffineSolveConfig(batch_size=31, gradient_relative_tolerance=1.e-9))


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_affine_loss_gradient_and_hessian_equal_canonical_autograd(diagnostic):
    instance = problem(scale=.7)
    weights = .05 * torch.randn(instance.parameter_shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    weights.requires_grad_()
    states = instance.states[instance.rows]
    augmented = torch.cat((states, torch.ones((len(states), 1), dtype=states.dtype)), dim=1)
    mask = instance.eligible[instance.rows % len(instance.eligible)]
    expected = local_objective((augmented@weights)*mask, instance.returns[instance.rows],
        diagnostic=diagnostic, action_scale=instance.action_scale)
    gradient, = torch.autograd.grad(expected, weights, create_graph=True)
    direction = torch.randn(weights.shape, generator=torch.Generator().manual_seed(4), dtype=weights.dtype)
    hessian, = torch.autograd.grad(torch.sum(gradient*direction), weights)
    actual_loss, actual_gradient = instance.evaluate(weights.detach(), diagnostic=diagnostic)
    actual_hessian = instance.hessian(weights.detach(), direction, diagnostic=diagnostic)
    torch.testing.assert_close(actual_loss, expected)
    torch.testing.assert_close(actual_gradient, gradient)
    torch.testing.assert_close(actual_hessian, hessian)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
@pytest.mark.parametrize('singular', [False, True])
def test_deterministic_affine_fit_converges_without_ridge_or_clipping(diagnostic, singular):
    instance = problem(singular=singular)
    weights, report = instance.solve(diagnostic=diagnostic)
    assert report['converged']
    assert report['history'][-1]['loss'] < 0.
    assert report['history'][-1]['gradient_norm'] <= report['history'][-1]['gradient_threshold']
    assert report['gradient_clipping'] is False
    assert report['rank_report']['objective_ridge_penalty_added'] is False
    assert report['rank_report']['coefficient_directions_discarded'] is False
    assert max(report['rank_report']['eigenvalues_above_preconditioner_floor']) == (4 if singular else 5)
    losses = [item['loss'] for item in report['history']]
    assert losses == sorted(losses, reverse=True)
    assert torch.isfinite(weights).all()


def test_squared_affine_solution_matches_direct_dense_hessian_solve():
    instance = problem(scale=1.)
    instance.config = replace(instance.config, cg_relative_tolerance=1.e-12)
    weights, report = instance.solve(diagnostic='squared')
    zero = torch.zeros(instance.parameter_shape, dtype=torch.float64)
    _, gradient = instance.evaluate(zero, diagnostic='squared')
    basis = torch.eye(zero.numel(), dtype=zero.dtype)
    hessian = torch.stack([instance.hessian(zero, row.reshape_as(zero), diagnostic='squared').flatten()
        for row in basis], dim=1)
    expected = torch.linalg.solve(hessian, -gradient.flatten()).reshape_as(weights)
    torch.testing.assert_close(weights, expected, atol=1.e-10, rtol=1.e-9)
    assert report['history'][0]['linear_solve']['converged']


def test_masked_gram_preconditioner_and_training_only_rows():
    instance = problem()
    instance.states[200:] = float('nan')
    instance.returns[200:] = float('nan')
    instance.prepare()
    augmented = torch.cat((instance.states[:200], torch.ones((200, 1), dtype=torch.float64)), dim=1)
    for group, columns in enumerate(instance.group_columns):
        mask = instance.eligible[instance.rows % len(instance.eligible), columns[0]]
        expected = augmented[mask].T @ augmented[mask] / 200
        torch.testing.assert_close(instance.grams[group], expected)
    weights, report = instance.solve(diagnostic='squared')
    assert report['converged']
    assert torch.isfinite(weights).all()


def test_iteration_limit_is_not_reported_as_convergence():
    instance = problem(scale=2.)
    instance.config = replace(instance.config, maximum_cg_iterations=1, maximum_newton_iterations=1,
        gradient_relative_tolerance=1.e-14, gradient_absolute_tolerance=1.e-15)
    _, report = instance.solve(diagnostic='entropy')
    assert report['converged'] is False


def test_cg_reports_actual_residual_and_handles_zero_right_side():
    matrix = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    right = torch.tensor([1., 2.], dtype=torch.float64)
    actual, report = preconditioned_cg(lambda value: matrix@value, right, lambda value: value,
        relative_tolerance=1.e-12, maximum_iterations=10)
    torch.testing.assert_close(actual, torch.linalg.solve(matrix, right))
    assert report['converged']
    zero, report = preconditioned_cg(lambda value: matrix@value, torch.zeros_like(right), lambda value: value,
        relative_tolerance=1.e-12, maximum_iterations=10)
    assert report['converged']
    assert report['iterations'] == 0
    assert not zero.any()


@pytest.mark.parametrize('stored_dtype', [torch.float32, torch.float64])
def test_small_positive_feature_direction_is_not_discarded_or_falsely_converged(stored_dtype):
    first = torch.tensor([-1., -1., 1., 1.], dtype=stored_dtype).repeat(10)
    second = torch.tensor([-1., 1., -1., 1.], dtype=stored_dtype).repeat(10)
    states = torch.stack((first, first+1.e-7*second), dim=1)
    returns = (first+.1*second)[:, None]
    instance = AffineAuditProblem(states=states, returns=returns,
        eligible=torch.ones((1, 1), dtype=torch.bool), rows=torch.arange(len(states)), action_scale=.004,
        config=AffineSolveConfig(batch_size=16, maximum_cg_iterations=30))
    weights, report = instance.solve(diagnostic='squared')
    candidate = torch.tensor([[-1.e6], [1.e6+1.], [0.]], dtype=torch.float64)
    control, _ = instance.evaluate(candidate, diagnostic='squared')
    actual, _ = instance.evaluate(weights, diagnostic='squared')
    assert float(actual) <= float(control)+1.e-7
    assert report['rank_report']['coefficient_directions_discarded'] is False
    if report['converged']:
        assert report['history'][-1]['preconditioned_gradient_norm'] <= report['history'][-1]['preconditioned_gradient_threshold']
