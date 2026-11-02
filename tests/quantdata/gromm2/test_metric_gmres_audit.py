from __future__ import annotations

import pytest
import torch

from quantdata.gromm2.metric_gmres_audit import metric_right_gmres


@pytest.mark.parametrize('diagonal', [[1., 3., 8.], [-1., 3., 8.], [-1., -3., -8.]])
def test_positive_and_indefinite_symmetric_systems_use_true_M_residual(diagonal):
    matrix = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    right = torch.tensor([.3, -.2, .1], dtype=torch.float64)
    preconditioner = torch.tensor([[2., .1, 0.], [.1, 1., .2], [0., .2, 3.]], dtype=torch.float64)
    stopping = torch.diag(torch.tensor([.2, 4., 2.], dtype=torch.float64))
    calls = []

    def operator(value):
        calls.append(value.clone())
        return matrix@value

    solution, residual, report = metric_right_gmres(right=right, operator=operator,
        precondition=lambda value: (preconditioner@value, torch.dot(value, preconditioner@value)),
        metric_embedding=lambda value: torch.linalg.cholesky(stopping).T@value,
        relative_tolerance=1.e-10, maximum_iterations=6)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, right), atol=1.e-13, rtol=1.e-12)
    torch.testing.assert_close(residual, right-matrix@solution, atol=0., rtol=0.)
    assert report['converged'] and report['residual_is_true'] and len(calls) == report['iterations']+1
    assert not report['SPD_Hessian_assumed'] and not report['outer_gradient_convergence_certified']


def test_false_recursive_convergence_restarts_from_actual_residual():
    right = torch.tensor([1., -.25], dtype=torch.float64)
    calls = []

    def operator(value):
        calls.append(value.clone())
        return (1.1 if len(calls) == 1 else 1.)*value

    solution, residual, report = metric_right_gmres(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        metric_embedding=lambda value: value, relative_tolerance=1.e-10, maximum_iterations=5)
    torch.testing.assert_close(solution, right, atol=1.e-13, rtol=1.e-13)
    assert report['converged'] and report['false_true_gate_restarts'] == 1
    assert report['iterations'] == 2 and len(calls) == 4
    assert float(torch.linalg.vector_norm(residual)) <= report['threshold']


def test_iteration_cap_confirms_actual_residual_without_claiming_convergence():
    matrix = torch.diag(torch.tensor([1., 3., 8.], dtype=torch.float64))
    right = torch.ones(3, dtype=torch.float64)
    calls = []

    def operator(value):
        calls.append(value.clone())
        return matrix@value

    solution, residual, report = metric_right_gmres(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        metric_embedding=lambda value: value, relative_tolerance=1.e-10, maximum_iterations=1)
    assert len(calls) == 2 and torch.equal(calls[-1], solution)
    torch.testing.assert_close(residual, right-matrix@solution, atol=0., rtol=0.)
    assert not report['converged'] and report['stop_reason'] == 'iteration_cap_not_converged'


def test_singular_compatible_system_preserves_null_policy_directions():
    matrix = torch.diag(torch.tensor([1., 0., -2.], dtype=torch.float64))
    right = torch.tensor([1., 0., 1.], dtype=torch.float64)
    solution, residual, report = metric_right_gmres(right=right, operator=lambda value: matrix@value,
        precondition=lambda value: (value.clone(), value.square().sum()),
        metric_embedding=lambda value: value, relative_tolerance=1.e-10, maximum_iterations=4)
    torch.testing.assert_close(solution, torch.tensor([1., 0., -.5], dtype=torch.float64), atol=1.e-13, rtol=1.e-13)
    assert report['converged'] and float(torch.linalg.vector_norm(residual)) < report['threshold']


def test_zero_operator_is_numerical_refusal_not_a_zero_policy_pass():
    right = torch.ones(2, dtype=torch.float64)
    solution, residual, report = metric_right_gmres(right=right,
        operator=torch.zeros_like, precondition=lambda value: (value.clone(), value.square().sum()),
        metric_embedding=lambda value: value, relative_tolerance=.1, maximum_iterations=4)
    assert bool(solution.eq(0.).all()) and torch.equal(residual, right)
    assert not report['converged'] and report['stop_reason'] == 'rank_unreliable_projected_problem_not_converged'


def test_inconsistent_metric_embedding_dtype_is_rejected_before_products():
    calls = []
    with pytest.raises(FloatingPointError, match='embedding'):
        metric_right_gmres(right=torch.ones(2, dtype=torch.float64),
            operator=lambda value: calls.append(value),
            precondition=lambda value: (value.clone(), value.square().sum()),
            metric_embedding=lambda value: value.float(), relative_tolerance=.1, maximum_iterations=3)
    assert not calls


def test_cached_normalization_cannot_hide_raw_embedding_cancellation():
    factor = torch.tensor([[1.e16, 1.e16], [0., 1.]], dtype=torch.float64)
    right = torch.tensor([1., -1.+torch.finfo(torch.float64).eps], dtype=torch.float64)
    calls, preconditions = [], []
    with pytest.raises(FloatingPointError, match='normalized raw GMRES basis'):
        metric_right_gmres(right=right, operator=lambda value: calls.append(value),
            precondition=lambda value: preconditions.append(value), metric_embedding=lambda value: factor@value,
            relative_tolerance=.1, maximum_iterations=3)
    assert not calls and not preconditions


def test_near_null_nonzero_direction_is_not_discarded():
    matrix = torch.diag(torch.tensor([1.e-6, -1., 2.], dtype=torch.float64))
    right = torch.tensor([.3, -.2, .1], dtype=torch.float64)
    solution, residual, report = metric_right_gmres(right=right,
        operator=lambda value: matrix@value, precondition=lambda value: (value.clone(), value.square().sum()),
        metric_embedding=lambda value: value, relative_tolerance=1.e-8, maximum_iterations=6)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, right), atol=1.e-4, rtol=1.e-8)
    assert report['converged'] and float(torch.linalg.vector_norm(residual)) <= report['threshold']


def test_singular_incompatible_system_never_reports_a_false_pass():
    matrix = torch.diag(torch.tensor([1., 0., -2.], dtype=torch.float64))
    right = torch.tensor([1., 1., 1.], dtype=torch.float64)
    solution, residual, report = metric_right_gmres(right=right,
        operator=lambda value: matrix@value, precondition=lambda value: (value.clone(), value.square().sum()),
        metric_embedding=lambda value: value, relative_tolerance=1.e-10, maximum_iterations=6)
    torch.testing.assert_close(residual, right-matrix@solution, atol=0., rtol=0.)
    assert not report['converged'] and report['residual_is_true']
    assert report['true_metric_residual_norm'] >= 1.
    assert float(torch.linalg.vector_norm(solution)) < 1.e8
    assert report['stop_reason'] == 'rank_unreliable_projected_problem_not_converged'


def test_relative_well_conditioned_scalar_near_null_is_not_truncated():
    right = torch.ones(1, dtype=torch.float64)
    calls = []

    def operator(value):
        calls.append(value.clone())
        return 1.e-18*value

    solution, residual, report = metric_right_gmres(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        metric_embedding=lambda value: value, relative_tolerance=1.e-12, maximum_iterations=2)
    torch.testing.assert_close(solution, torch.full_like(right, 1.e18), atol=0., rtol=1.e-14)
    assert report['converged'] and len(calls) == 2 and torch.equal(calls[-1], solution)
    assert float(torch.linalg.vector_norm(residual)) <= report['threshold']


@pytest.mark.parametrize('case', ['zero', 'dtype', 'nan', 'iterations', 'tolerance'])
def test_invalid_inputs_fail_before_any_product(case):
    right = torch.ones(2, dtype=torch.float64)
    if case == 'zero':
        right.zero_()
    elif case == 'dtype':
        right = right.float()
    elif case == 'nan':
        right[0] = float('nan')
    calls = []
    with pytest.raises(ValueError):
        metric_right_gmres(right=right, operator=lambda value: calls.append(value),
            precondition=lambda value: (value, value.square().sum()),
            metric_embedding=lambda value: value,
            relative_tolerance=1. if case == 'tolerance' else .1,
            maximum_iterations=0 if case == 'iterations' else 3)
    assert not calls


def test_original_M_residual_is_minimized_in_the_same_right_preconditioned_subspace():
    from quantdata.gromm2.weighted_arnoldi_audit import weighted_arnoldi_minimum_residual

    matrix = torch.tensor([[-2., .3, .1, .2], [.3, 1., .2, .1],
        [.1, .2, 4., .3], [.2, .1, .3, .5]], dtype=torch.float64)
    positive = torch.diag(torch.tensor([2., 1., 3., .5], dtype=torch.float64))
    factor = torch.diag(torch.tensor([.01, 1., 10., 2.], dtype=torch.float64))
    right = torch.tensor([.3, -.2, .1, .4], dtype=torch.float64)
    common = dict(right=right, operator=lambda value: matrix@value,
        precondition=lambda value: (positive@value, torch.dot(value, positive@value)),
        relative_tolerance=1.e-14, maximum_iterations=2)
    old, old_residual, _ = weighted_arnoldi_minimum_residual(**common,
        stopping_energy=lambda value: (factor@value).square().sum())
    progress = []
    new, new_residual, report = metric_right_gmres(**common, metric_embedding=lambda value: factor@value,
        progress=progress.append)
    assert float(torch.linalg.vector_norm(factor@new_residual)) <= float(torch.linalg.vector_norm(factor@old_residual))+1.e-12
    assert progress[0]['original_metric_residual_norm'] <= float(torch.linalg.vector_norm(factor@right))+1.e-12
    assert progress[1]['original_metric_residual_norm'] <= progress[0]['original_metric_residual_norm']+1.e-12
    search = torch.stack((positive@right, positive@matrix@positive@right), dim=1)
    for direction in (old, new):
        coefficients = torch.linalg.lstsq(search, direction[:, None]).solution
        torch.testing.assert_close((search@coefficients)[:, 0], direction, atol=1.e-12, rtol=1.e-12)
    assert report['minimized_residual_geometry'] == report['stopping_geometry'] == 'original fixed M'
    assert not report['MINRES'] and not report['symmetric_HP_assumed']
