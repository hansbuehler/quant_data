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
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.gromm2.shifted_exact_gate_audit_proposal import (
    NegativeShiftedCurvature,
    ShiftedExactConfig,
    guarded_shifted_pcg,
    propose_shifted_exact_step,
)
from quantdata.torch import BaseEnvironment


def test_guarded_pcg_spd_uses_stable_energy_and_confirms_true_residual():
    matrix = torch.tensor([[2., .1], [.1, 1.]], dtype=torch.float64)
    right = torch.tensor([.2, -.3], dtype=torch.float64)
    calls = []

    def operator(vector):
        calls.append(vector.clone())
        image = matrix@vector
        return image, torch.dot(vector, image)

    solution, report = guarded_shifted_pcg(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-10, maximum_iterations=5)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, right), atol=1.e-13, rtol=1.e-12)
    assert report['converged'] and report['residual_is_true']
    assert len(calls) == 3 and torch.equal(calls[-1], solution)
    assert not report['outer_gradient_convergence_certified'] and not report['SPD_spectrum_certified']


def test_negative_curvature_is_rejected_before_a_cg_division():
    calls = []

    def operator(vector):
        calls.append(vector.clone())
        raise NegativeShiftedCurvature(generalized_rayleigh=-2., shift=.1)

    with pytest.raises(NegativeShiftedCurvature) as caught:
        guarded_shifted_pcg(right=torch.ones(2, dtype=torch.float64), operator=operator,
            precondition=lambda value: (value.clone(), value.square().sum()),
            stopping_energy=lambda value: value.square().sum(), relative_tolerance=.1, maximum_iterations=5)
    assert len(calls) == 1 and caught.value.generalized_rayleigh == -2.


def test_iteration_cap_is_not_linear_or_outer_convergence():
    matrix = torch.diag(torch.tensor([1., 4., 8.], dtype=torch.float64))
    solution, report = guarded_shifted_pcg(right=torch.ones(3, dtype=torch.float64),
        operator=lambda value: (matrix@value, torch.dot(value, matrix@value)),
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-10, maximum_iterations=1)
    assert not report['converged'] and report['residual_is_true']
    assert report['iterations'] == 1 and bool(torch.isfinite(solution).all())


def prepared(diagnostic):
    generator = torch.Generator().manual_seed(31)
    problem = AffineAuditProblem(states=torch.randn((12, 2), generator=generator, dtype=torch.float32),
        returns=.1*torch.randn((12, 2), generator=generator, dtype=torch.float32)+.2,
        eligible=torch.ones((2, 2), dtype=torch.bool), rows=torch.arange(12),
        action_scale=.2, config=AffineSolveConfig(batch_size=4))
    with torch.random.fork_rng():
        torch.manual_seed(32)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='shifted-exact', feature_count=2, instrument_count=2,
                action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.1), history_size=3, curvature_floor=1.e-12)
    full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    step = torch.randn(gradient.shape, generator=generator, dtype=torch.float64)
    assert optimizer.accept(step, .3*step)
    return policy, problem, optimizer, gradient


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_shifted_proposal_preserves_native_state_and_reports_true_predicted_gradients(diagnostic):
    policy, problem, optimizer, gradient = prepared(diagnostic)
    point, history, rng = action_point_sha256(policy), optimizer_state_sha256(optimizer), torch.get_rng_state().clone()
    saved_gradients = {name: value.grad.clone() for name, value in policy.named_parameters()}
    direction, report = propose_shifted_exact_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic=diagnostic, point_sha256=point,
        config=ShiftedExactConfig(initial_shift=1., maximum_hessian_products=80,
            predicted_gradient_ratio_limit=2., compile=False))
    assert report['full_curvature_passes'] <= 80
    assert optimizer_state_sha256(optimizer) == history and action_point_sha256(policy) == point
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value.grad, saved_gradients[name]) for name, value in policy.named_parameters())
    assert not report['SPD_spectrum_certified'] and not report['outer_gradient_convergence_certified']
    if direction is not None:
        selected = report['selected']
        assert report['proposal_selected'] and selected['linear_solve']['converged']
        assert selected['slope'] < 0.
        assert selected['actual_true_shifted_metric_residual_norm'] <= selected['linear_solve']['threshold']
        assert selected['actual_unshifted_predicted_norms']['full_gradient_norm'] <= 2.*report['initial_norms']['full_gradient_norm']
    else:
        assert not report['proposal_selected'] and report['selected'] is None


def test_all_shift_trials_share_one_hessian_budget_and_never_select_zero():
    policy, problem, optimizer, gradient = prepared('squared')
    direction, report = propose_shifted_exact_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=ShiftedExactConfig(maximum_hessian_products=1, compile=False))
    assert direction is None and not report['proposal_selected']
    assert report['full_curvature_passes'] == 1
    assert report['trials'][-1]['status'] == 'shared_hessian_work_limit_not_converged'


def install_quadratic(monkeypatch, multiplier, mutate=False):
    import quantdata.gromm2.shifted_exact_gate_audit_proposal as implementation

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


def test_known_spd_quadratic_produces_a_nonzero_quality_checked_direction(monkeypatch):
    install_quadratic(monkeypatch, 1.)
    policy, problem, optimizer, gradient = prepared('squared')
    direction, report = propose_shifted_exact_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=ShiftedExactConfig(relative_tolerance=1.e-10, maximum_hessian_products=80, compile=False))
    assert direction is not None and bool(direction.ne(0.).any())
    assert report['proposal_selected'] and len(report['trials']) == 1
    assert report['selected']['linear_solve']['converged']
    assert report['selected']['actual_unshifted_predicted_norms']['full_gradient_norm'] < float(torch.linalg.vector_norm(gradient))


def test_negative_curvature_restarts_shift_but_does_not_certify_a_worse_prediction(monkeypatch):
    install_quadratic(monkeypatch, -.02)
    policy, problem, optimizer, gradient = prepared('squared')
    direction, report = propose_shifted_exact_step(policy=policy, problem=problem, optimizer=optimizer,
        gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
        config=ShiftedExactConfig(maximum_hessian_products=80, compile=False))
    assert direction is None and not report['proposal_selected']
    first = report['trials'][0]
    assert first['status'] == 'negative_curvature_restart'
    assert first['generalized_original_metric_rayleigh'] < -first['shift']
    assert report['trials'][1]['shift'] == pytest.approx(max(10.*first['shift'], -2.*first['generalized_original_metric_rayleigh']))
    assert report['full_curvature_passes'] == sum(entry['full_curvature_passes'] for entry in report['trials'])


def test_materialized_curvature_disagreement_is_not_silently_repaired_by_shift(monkeypatch):
    import quantdata.gromm2.shifted_exact_gate_audit_proposal as implementation

    original_inverse = implementation.InverseAffineMetric

    class WrongInverse(original_inverse):
        def __call__(self, vector):
            return 2.*super().__call__(vector)

    monkeypatch.setattr(implementation, 'InverseAffineMetric', WrongInverse)
    policy, problem, optimizer, gradient = prepared('squared')
    with pytest.raises(FloatingPointError, match='disagree'):
        propose_shifted_exact_step(policy=policy, problem=problem, optimizer=optimizer,
            gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
            config=ShiftedExactConfig(compile=False))


def test_frozen_forward_mutation_during_product_is_rejected(monkeypatch):
    install_quadratic(monkeypatch, 1., mutate=True)
    policy, problem, optimizer, gradient = prepared('squared')
    with pytest.raises(ValueError, match='snapshot'):
        propose_shifted_exact_step(policy=policy, problem=problem, optimizer=optimizer,
            gradient=gradient, diagnostic='squared', point_sha256=action_point_sha256(policy),
            config=ShiftedExactConfig(compile=False))


@pytest.mark.parametrize('case', ['stale_point', 'zero_gradient', 'affine_only', 'wrong_dtype'])
def test_shifted_proposal_rejects_changed_source_before_products(case):
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
        propose_shifted_exact_step(policy=policy, problem=problem, optimizer=optimizer,
            gradient=gradient, diagnostic='squared', point_sha256=point,
            config=ShiftedExactConfig(compile=False))

def test_failed_recursive_gate_confirms_true_residual_and_restarts():
    right = torch.tensor([1., -.25], dtype=torch.float64)
    calls, progress = [], []

    def operator(vector):
        calls.append(vector.clone())
        image = (1.1 if len(calls) == 1 else 1.)*vector
        return image, torch.dot(vector, image)

    solution, report = guarded_shifted_pcg(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-12,
        maximum_iterations=5, progress=progress.append)
    assert report['converged'] and report['iterations'] == 2 and len(calls) == 4
    assert progress[0]['true_residual_restart']
    assert progress[0]['original_metric_residual_norm'] > report['threshold']
    assert progress[1]['original_metric_residual_norm'] <= report['threshold']
    torch.testing.assert_close(solution, right, atol=1.e-14, rtol=1.e-14)


def test_cap_confirmation_counts_actual_product_and_rejects_true_failure():
    matrix = torch.diag(torch.tensor([1., 4., 8.], dtype=torch.float64))
    right = torch.ones(3, dtype=torch.float64)
    calls = []

    def operator(vector):
        calls.append(vector.clone())
        return matrix@vector, torch.dot(vector, matrix@vector)

    solution, report = guarded_shifted_pcg(right=right, operator=operator,
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(), relative_tolerance=1.e-12,
        maximum_iterations=1)
    assert len(calls) == 2 and torch.equal(calls[-1], solution)
    true_norm = float(torch.linalg.vector_norm(right-matrix@solution))
    assert report['true_metric_residual_norm'] == pytest.approx(true_norm)
    assert not report['converged'] and true_norm > report['threshold']


def test_benign_spd_matches_previous_recurrence_without_periodic_restart():
    from quantdata.gromm2.shifted_exact_audit_proposal import guarded_shifted_pcg as original_pcg

    matrix = torch.tensor([[3., .2, .1], [.2, 2., -.1], [.1, -.1, 1.]], dtype=torch.float64)
    right = torch.tensor([.2, -.3, .4], dtype=torch.float64)
    arguments = dict(right=right,
        operator=lambda value: (matrix@value, torch.dot(value, matrix@value)),
        precondition=lambda value: (value.clone(), value.square().sum()),
        stopping_energy=lambda value: value.square().sum(),
        relative_tolerance=1.e-10, maximum_iterations=9)
    old, old_report = original_pcg(**arguments)
    new, new_report = guarded_shifted_pcg(**arguments)
    assert new_report == old_report
    torch.testing.assert_close(new, old, atol=0., rtol=0.)


def test_gate_end_confirmation_retains_conjugacy_beyond_ten_iterations():
    from quantdata.gromm2.shifted_exact_audit_proposal import guarded_shifted_pcg as original_pcg

    generator = torch.Generator().manual_seed(20261102)
    rotation = torch.linalg.qr(torch.randn(96, 96, dtype=torch.float64, generator=generator))[0]
    right = torch.randn(96, dtype=torch.float64, generator=generator)
    matrix = (rotation*torch.logspace(0., 3., 96, dtype=torch.float64))@rotation.T
    work = {}
    reports = {}
    for name, solver in (('periodic', original_pcg), ('gate_end', guarded_shifted_pcg)):
        calls = []

        def operator(vector):
            calls.append(1)
            image = matrix@vector
            return image, torch.dot(vector, image)

        solution, report = solver(right=right, operator=operator,
            precondition=lambda value: (value.clone(), value.square().sum()),
            stopping_energy=lambda value: value.square().sum(),
            relative_tolerance=.1, maximum_iterations=120)
        work[name], reports[name] = len(calls), report
        actual = float(torch.linalg.vector_norm(right-matrix@solution))
        assert report['true_metric_residual_norm'] == pytest.approx(actual, abs=1.e-12)
        assert report['converged'] and actual <= report['threshold']
    assert reports['gate_end']['iterations'] > 10
    assert work['gate_end'] < work['periodic']

