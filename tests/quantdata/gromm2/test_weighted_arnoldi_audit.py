from __future__ import annotations

import pytest
import torch

from quantdata.gromm2.weighted_arnoldi_audit import weighted_arnoldi_minimum_residual


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

    solution, residual, report = weighted_arnoldi_minimum_residual(right=right, operator=operator,
        precondition=lambda value: (preconditioner@value, torch.dot(value, preconditioner@value)),
        stopping_energy=lambda value: torch.dot(value, stopping@value),
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

    solution, residual, report = weighted_arnoldi_minimum_residual(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-10, maximum_iterations=5)
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

    solution, residual, report = weighted_arnoldi_minimum_residual(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-10, maximum_iterations=1)
    assert len(calls) == 2 and torch.equal(calls[-1], solution)
    torch.testing.assert_close(residual, right-matrix@solution, atol=0., rtol=0.)
    assert not report['converged'] and report['stop_reason'] == 'iteration_cap_not_converged'


def test_singular_compatible_system_preserves_null_policy_directions():
    matrix = torch.diag(torch.tensor([1., 0., -2.], dtype=torch.float64))
    right = torch.tensor([1., 0., 1.], dtype=torch.float64)
    solution, residual, report = weighted_arnoldi_minimum_residual(right=right, operator=lambda value: matrix@value,
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-10, maximum_iterations=4)
    torch.testing.assert_close(solution, torch.tensor([1., 0., -.5], dtype=torch.float64), atol=1.e-13, rtol=1.e-13)
    assert report['converged'] and float(torch.linalg.vector_norm(residual)) < report['threshold']


def test_zero_operator_is_numerical_refusal_not_a_zero_policy_pass():
    right = torch.ones(2, dtype=torch.float64)
    solution, residual, report = weighted_arnoldi_minimum_residual(right=right,
        operator=torch.zeros_like, precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=.1, maximum_iterations=4)
    assert bool(solution.eq(0.).all()) and torch.equal(residual, right)
    assert not report['converged'] and report['stop_reason'] == 'rank_unreliable_projected_problem_not_converged'


def test_inconsistent_positive_energy_is_rejected():
    with pytest.raises(FloatingPointError, match='normalization is inconsistent'):
        weighted_arnoldi_minimum_residual(right=torch.ones(2, dtype=torch.float64), operator=lambda value: value,
            precondition=lambda value: (2.*value, value.square().sum()),
            stopping_energy=lambda value: value.square().sum(), relative_tolerance=.1, maximum_iterations=3)


def test_near_null_nonzero_direction_is_not_discarded():
    matrix = torch.diag(torch.tensor([1.e-6, -1., 2.], dtype=torch.float64))
    right = torch.tensor([.3, -.2, .1], dtype=torch.float64)
    solution, residual, report = weighted_arnoldi_minimum_residual(right=right,
        operator=lambda value: matrix@value, precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-8, maximum_iterations=6)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, right), atol=1.e-4, rtol=1.e-8)
    assert report['converged'] and float(torch.linalg.vector_norm(residual)) <= report['threshold']


def test_singular_incompatible_system_never_reports_a_false_pass():
    matrix = torch.diag(torch.tensor([1., 0., -2.], dtype=torch.float64))
    right = torch.tensor([1., 1., 1.], dtype=torch.float64)
    solution, residual, report = weighted_arnoldi_minimum_residual(right=right,
        operator=lambda value: matrix@value, precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-10, maximum_iterations=6)
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

    solution, residual, report = weighted_arnoldi_minimum_residual(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-12, maximum_iterations=2)
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
        weighted_arnoldi_minimum_residual(right=right, operator=lambda value: calls.append(value),
            precondition=lambda value: (value, value.square().sum()),
            stopping_energy=lambda value: value.square().sum(),
            relative_tolerance=1. if case == 'tolerance' else .1,
            maximum_iterations=0 if case == 'iterations' else 3)
    assert not calls
