from __future__ import annotations

import pytest
import torch

from quantdata.gromm2.warmstart_audit_pcg import restart_pcg_original_target


def test_restarted_correction_keeps_original_absolute_target_and_counts_all_products():
    system = torch.diag(torch.tensor([1., 5., 100.], dtype=torch.float64))
    right = torch.tensor([1., 2., 3.], dtype=torch.float64)
    initial = torch.tensor([.7, .38, .029], dtype=torch.float64)
    inverse = torch.linalg.inv(system)
    calls = []

    def operator(value):
        calls.append(value.clone())
        return system@value

    result, report = restart_pcg_original_target(operator, right, initial,
        lambda value: (inverse@value, value@inverse@value), stopping_energy=lambda value: value.square().sum(),
        original_absolute_threshold=1.e-10, maximum_iterations=20)
    torch.testing.assert_close(result, torch.linalg.solve(system, right))
    assert len(calls) == 4
    assert report['converged'] and report['iterations'] == 1
    assert report['original_metric_threshold'] == 1.e-10
    assert report['restarted_warm_start_not_uninterrupted_krylov']


def test_initial_passing_direction_requires_no_new_relative_tolerance_or_cg():
    right = torch.ones(2, dtype=torch.float64)
    calls = []

    def operator(value):
        calls.append(value.clone())
        return value

    result, report = restart_pcg_original_target(operator, right, right,
        lambda value: pytest.fail('a passing warm start needs no preconditioned recurrence'),
        stopping_energy=lambda value: value.square().sum(), original_absolute_threshold=.01, maximum_iterations=1)
    torch.testing.assert_close(result, right, rtol=0., atol=0.)
    assert report['iterations'] == 0 and report['converged'] and len(calls) == 2


def test_work_cap_does_not_relax_original_target():
    system = torch.diag(torch.tensor([1., 100.], dtype=torch.float64))
    right = torch.ones(2, dtype=torch.float64)
    _, report = restart_pcg_original_target(lambda value: system@value, right, torch.zeros_like(right),
        lambda value: (value, value.square().sum()), stopping_energy=lambda value: value.square().sum(),
        original_absolute_threshold=1.e-8, maximum_iterations=1)
    assert not report['converged'] and report['original_metric_residual_norm'] > 1.e-8


def test_combined_original_residual_overrules_a_passing_correction():
    right = torch.ones(1, dtype=torch.float64)
    initial = torch.full_like(right, 2.**54)
    _, report = restart_pcg_original_target(lambda value: value, right, initial,
        lambda value: (value, value.square().sum()), stopping_energy=lambda value: value.square().sum(),
        original_absolute_threshold=.1, maximum_iterations=2)
    assert report['correction_solve']['converged']
    assert report['original_metric_residual_norm'] == 1. and not report['converged']


def test_invalid_warm_start_rejects_before_operator_work():
    with pytest.raises(ValueError, match='finite float64'):
        restart_pcg_original_target(lambda value: pytest.fail('invalid warm start performed a product'),
            torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float32), None,
            stopping_energy=None, original_absolute_threshold=.1, maximum_iterations=1)
