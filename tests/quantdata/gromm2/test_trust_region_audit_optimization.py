from __future__ import annotations

import math

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
from quantdata.gromm2.lbfgs_audit_preconditioner import metric_energy
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.gromm2.neural_audit_optimization import full_action_pass
from quantdata.gromm2.trust_region_audit_optimization import (
    ExactTrustRegionConfig,
    OriginalMetricCoordinates,
    _boundary_roots,
    propose_exact_trust_region_step,
    steihaug_step,
)
from quantdata.torch import BaseEnvironment


def test_steihaug_solves_interior_spd_model_with_true_residual_and_counted_products():
    matrix = torch.tensor([[2., .2], [.2, 1.]], dtype=torch.float64)
    gradient = torch.tensor([.3, -.2], dtype=torch.float64)
    calls = []

    def product(vector):
        calls.append(vector.clone())
        return matrix@vector

    source = gradient.clone()
    step, report = steihaug_step(gradient=gradient, hessian_product=product, radius=10.,
        config=ExactTrustRegionConfig(relative_tolerance=1.e-12, maximum_iterations=5))
    torch.testing.assert_close(step, torch.linalg.solve(matrix, -gradient), rtol=1.e-12, atol=1.e-14)
    assert report['stop_reason'] == 'verified_true_residual'
    assert report['inner_residual_converged'] and not report['outer_gradient_convergence_certified']
    assert report['full_exact_hessian_products'] == len(calls) == 4
    assert torch.equal(source, gradient) and report['optimizer_updates'] == 0


@pytest.mark.parametrize('diagonal, expected', [([-2., 1.], 'negative_curvature_boundary'),
    ([0., 0.], 'negative_curvature_boundary'), ([2., 1.], 'positive_curvature_boundary')])
def test_indefinite_and_positive_curvature_boundary_are_not_convergence(diagonal, expected):
    matrix = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    gradient = torch.tensor([1., .1], dtype=torch.float64)
    step, report = steihaug_step(gradient=gradient, hessian_product=lambda value: matrix@value, radius=.1)
    assert report['stop_reason'] == expected
    assert torch.linalg.vector_norm(step) == pytest.approx(.1)
    assert torch.dot(step, gradient) < 0.
    assert report['predicted_reduction'] >= report['cauchy_predicted_reduction']
    assert not report['inner_residual_converged'] and not report['outer_gradient_convergence_certified']
    assert report['full_exact_hessian_products'] == 2


def test_iteration_cap_retains_descent_without_certifying_residual_convergence():
    matrix = torch.diag(torch.tensor([1., 4., 8.], dtype=torch.float64))
    gradient = torch.tensor([1., .4, -.1], dtype=torch.float64)
    step, report = steihaug_step(gradient=gradient, hessian_product=lambda value: matrix@value, radius=10.,
        config=ExactTrustRegionConfig(relative_tolerance=1.e-12, maximum_iterations=1))
    assert report['stop_reason'] == 'iteration_cap_not_residual_convergence'
    assert report['iterations'] == 1 and report['full_exact_hessian_products'] == 2
    assert report['predicted_reduction'] > 0. and torch.dot(gradient, step) < 0.
    assert not report['inner_residual_converged']


def test_cauchy_fallback_verifies_the_returned_point_with_an_extra_true_product(monkeypatch):
    import quantdata.gromm2.trust_region_audit_optimization as implementation

    monkeypatch.setattr(implementation, '_boundary_roots', lambda *args: (-.001, .001))
    gradient = torch.tensor([1., .3], dtype=torch.float64)
    calls = []

    def product(vector):
        calls.append(vector.clone())
        return vector.clone()

    step, report = steihaug_step(gradient=gradient, hessian_product=product, radius=.1)
    assert report['cauchy_fallback'] and report['stop_reason'] == 'cauchy_model_reduction_fallback'
    assert report['full_exact_hessian_products'] == len(calls) == 3
    assert torch.equal(calls[-1], step)
    assert report['verified_residual_norm'] == pytest.approx(float(torch.linalg.vector_norm(gradient+step)))
    assert torch.linalg.vector_norm(step) == pytest.approx(.1)


def test_boundary_intersection_is_stable_near_positive_boundary():
    point = torch.tensor([1.-1.e-12], dtype=torch.float64)
    direction = torch.ones_like(point)
    negative, positive = _boundary_roots(point, direction, 1.)
    assert positive > 0. and positive == pytest.approx(1.-point.item(), rel=1.e-10)
    assert negative == pytest.approx(-1.-point.item())
    assert point.item()+positive == pytest.approx(1.)


@pytest.mark.parametrize('radius', [0., -1., float('nan'), float('inf'), 1.e-300, 1.e300])
def test_invalid_radius_is_not_silently_floored(radius):
    with pytest.raises(ValueError):
        steihaug_step(gradient=torch.ones(2, dtype=torch.float64), hessian_product=lambda value: value, radius=radius)


@pytest.mark.parametrize('kind', ['zero', 'nan', 'float32', 'bad_product'])
def test_invalid_gradient_or_hessian_fails_instead_of_discarding_directions(kind):
    gradient = torch.ones(2, dtype=torch.float64)
    if kind == 'zero':
        gradient.zero_()
    elif kind == 'nan':
        gradient[0] = float('nan')
    elif kind == 'float32':
        gradient = gradient.float()
    product = (lambda value: value*float('nan')) if kind == 'bad_product' else (lambda value: value)
    with pytest.raises((ValueError, FloatingPointError)):
        steihaug_step(gradient=gradient, hessian_product=product, radius=1.)


def prepared(diagnostic):
    generator = torch.Generator().manual_seed(21)
    problem = AffineAuditProblem(states=torch.randn((14, 2), generator=generator, dtype=torch.float32),
        returns=.1*torch.randn((14, 2), generator=generator, dtype=torch.float32)+.2,
        eligible=torch.ones((2, 2), dtype=torch.bool), rows=torch.arange(14),
        action_scale=.2, config=AffineSolveConfig(batch_size=4))
    with torch.random.fork_rng():
        torch.manual_seed(22)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='exact-trust-region', feature_count=2, instrument_count=2,
                action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.1), history_size=3, curvature_floor=1.e-12)
    full_action_pass(policy, problem, diagnostic=diagnostic, backward=True)
    gradient = _flatten({name: value.grad for name, value in policy.named_parameters()})
    step = torch.randn(gradient.shape, generator=generator, dtype=torch.float64)
    assert optimizer.accept(step, .2*step)
    return policy, problem, optimizer, gradient


def test_original_metric_transforms_preserve_all_directions_and_match_stable_norm():
    policy, problem, optimizer, gradient = prepared('squared')
    coordinates = OriginalMetricCoordinates(optimizer.metric)
    forward = coordinates.apply(gradient, operation='forward')
    torch.testing.assert_close(coordinates.apply(forward, operation='inverse'), gradient, rtol=1.e-12, atol=1.e-13)
    transpose = coordinates.apply(gradient, operation='transpose')
    assert float(transpose.square().sum()) == pytest.approx(float(metric_energy(optimizer.metric, gradient)), rel=1.e-12)
    assert torch.dot(gradient, forward) == pytest.approx(float(torch.dot(transpose, gradient)))
    torch.testing.assert_close(coordinates.apply(transpose, operation='forward'), optimizer.metric(gradient))
    optimizer.metric.factors.add_(0.)
    with pytest.raises(ValueError, match='changed'):
        coordinates.apply(gradient, operation='forward')


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_exact_neural_proposal_is_read_only_and_requires_full_canonical_acceptance(diagnostic):
    policy, problem, optimizer, gradient = prepared(diagnostic)
    point, history = action_point_sha256(policy), optimizer_state_sha256(optimizer)
    saved_gradients = {name: value.grad.clone() for name, value in policy.named_parameters()}
    rng = torch.get_rng_state().clone()
    exact = FullActionHessian(policy, problem, diagnostic=diagnostic)
    direction, report = propose_exact_trust_region_step(policy=policy, optimizer=optimizer,
        exact=exact, gradient=gradient, radius=.001, point_sha256=point)
    assert report['slope'] < 0. and report['predicted_reduction'] > 0.
    assert report['full_curvature_passes'] == exact.full_curvature_passes
    assert report['inner_residual_converged'] == (report['actual_original_metric_residual_norm'] <= report['residual_threshold'])
    assert report['mapped_direction_inverse_metric_norm'] <= .001*(1.+1.e-9)
    assert optimizer_state_sha256(optimizer) == history and action_point_sha256(policy) == point
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value.grad, saved_gradients[name]) for name, value in policy.named_parameters())
    result, increment, change = accept_gradient_reliable_direction(policy=policy, problem=problem,
        direction=direction, exact=exact, metric=optimizer.metric, diagnostic=diagnostic,
        config=GradientReliabilityConfig(initial_rate=1.))
    assert result['accepted'] and result['verified_training_loss'] < result['initial_loss']
    assert result['trials'][-1]['nonlinear_fraction_of_initial_gradient'] <= .25
    assert math.isfinite(float(torch.linalg.vector_norm(change))) and bool(increment.ne(0.).any())
    assert optimizer_state_sha256(optimizer) == history


def test_exact_neural_proposal_rejects_stale_snapshot_and_source():
    policy, problem, optimizer, gradient = prepared('squared')
    point = action_point_sha256(policy)
    exact = FullActionHessian(policy, problem, diagnostic='squared')
    with pytest.raises(ValueError, match='snapshot'):
        propose_exact_trust_region_step(policy=policy, optimizer=optimizer, exact=exact,
            gradient=gradient, radius=.001, point_sha256='0'*64)
    with torch.no_grad():
        next(iter(policy.parameters())).add_(.001)
    with pytest.raises(ValueError, match='snapshot'):
        propose_exact_trust_region_step(policy=policy, optimizer=optimizer, exact=exact,
            gradient=gradient, radius=.001, point_sha256=action_point_sha256(policy))
    assert action_point_sha256(policy) != point


@pytest.mark.parametrize('mutation', ['affine_only', 'final_activation', 'training_mode'])
def test_exact_neural_proposal_rejects_changed_frozen_forward_before_any_product(mutation):
    policy, problem, optimizer, gradient = prepared('squared')
    exact = FullActionHessian(policy, problem, diagnostic='squared')
    if mutation == 'affine_only':
        exact.functional_policy.affine_only = True
    elif mutation == 'final_activation':
        module = next(module for module in exact.functional_policy.modules() if hasattr(module, 'use_final_activation'))
        module.use_final_activation = not module.use_final_activation
    else:
        exact.functional_policy.eval()
    with pytest.raises(ValueError, match='snapshot'):
        propose_exact_trust_region_step(policy=policy, optimizer=optimizer, exact=exact,
            gradient=gradient, radius=.001, point_sha256=action_point_sha256(policy))
    assert exact.full_curvature_passes == 0


@pytest.mark.parametrize('mutation', ['snapshot_tensor', 'native_tensor', 'both_forward_modes'])
def test_exact_neural_proposal_rejects_mutation_between_products(mutation):
    policy, problem, optimizer, gradient = prepared('squared')
    exact = FullActionHessian(policy, problem, diagnostic='squared')
    original_product = exact.product

    def product(vector):
        image = original_product(vector)
        if mutation == 'snapshot_tensor':
            next(iter(exact.parameters.values())).add_(0.)
        elif mutation == 'native_tensor':
            with torch.no_grad():
                next(iter(policy.parameters())).add_(0.)
        else:
            policy.eval()
            exact.functional_policy.eval()
        return image

    exact.product = product
    with pytest.raises(ValueError, match='snapshot'):
        propose_exact_trust_region_step(policy=policy, optimizer=optimizer, exact=exact,
            gradient=gradient, radius=.001, point_sha256=action_point_sha256(policy))
    assert exact.full_curvature_passes == 1
