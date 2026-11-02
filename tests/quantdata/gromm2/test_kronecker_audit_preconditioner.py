from __future__ import annotations

from itertools import product

import pytest
import torch

from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
    local_objective,
)
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256, optimizer_state_sha256
from quantdata.gromm2.kronecker_audit_preconditioner import (
    FrozenKroneckerLBFGS,
    KroneckerBaseConfig,
    exact_head_ggn_factor,
    output_curvature_sqrt_probe,
    positive_inverse_factor,
)
from quantdata.gromm2.metric_audit_optimization import AffineBlockMetric, MetricLBFGS
from quantdata.torch import BaseEnvironment


def fixture():
    generator = torch.Generator().manual_seed(613)
    with torch.random.fork_rng():
        torch.manual_seed(614)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='kronecker-test', feature_count=3, instrument_count=2,
                action_scale=.3, network=ActionNetworkConfig(num_depth=1, num_width=2)))
    problem = AffineAuditProblem(states=torch.randn(9, 3, generator=generator),
        returns=torch.randn(9, 2, generator=generator), eligible=torch.tensor([[True, False], [True, True], [False, True]]),
        rows=torch.arange(9), action_scale=.3, config=AffineSolveConfig(batch_size=4))
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=.01), history_size=3, curvature_floor=1.e-12)
    for _ in range(3):
        step = torch.randn(optimizer.metric.size, dtype=torch.float64, generator=generator)
        assert optimizer.accept(step, step*torch.linspace(.2, 2., len(step), dtype=step.dtype))
    moments = {}
    for name, module in policy.named_modules():
        if name.startswith('deep.') and isinstance(module, torch.nn.Linear):
            width = module.in_features+int(module.bias is not None)
            left = torch.randn(width, width, dtype=torch.float64, generator=generator)
            right = torch.randn(module.out_features, module.out_features, dtype=torch.float64, generator=generator)
            moments[name] = (left@left.T+.1*torch.eye(width),
                right@right.T+.2*torch.eye(module.out_features))
    return optimizer, policy, moments


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_output_probes_and_exact_head_factor_match_the_original_dense_curvature(diagnostic):
    action = torch.tensor([[.2, -.4], [.7, -.1], [-.2, .3]], dtype=torch.float64, requires_grad=True)
    returns = torch.tensor([[.1, -.6], [-.8, .5], [.4, .3]], dtype=torch.float64)
    eligible = torch.tensor([[True, False], [True, True], [False, True]])
    states = torch.tensor(list(product((-1., 1.), repeat=2)), dtype=torch.float64)
    images = torch.stack([output_curvature_sqrt_probe(action*eligible, returns, eligible,
        probe[None, :].expand_as(action), diagnostic=diagnostic, action_scale=.3, squared_lambda=2.) for probe in states])
    assert not images.requires_grad
    observed = torch.einsum('pbi,pbj->bij', images, images)/len(states)
    expected = []
    for index in range(len(action)):
        def scalar(value):
            return local_objective(value[None, :]*eligible[index:index+1], returns[index:index+1],
                diagnostic=diagnostic, action_scale=.3, squared_lambda=2.)

        expected.append(torch.autograd.functional.hessian(scalar, action[index]))
    expected = torch.stack(expected)
    torch.testing.assert_close(observed, expected, rtol=1.e-13, atol=1.e-14)
    head = exact_head_ggn_factor(action*eligible, returns, eligible,
        diagnostic=diagnostic, action_scale=.3, squared_lambda=2.)
    torch.testing.assert_close(head, expected.mean(dim=0), rtol=1.e-13, atol=1.e-14)
    assert not head.requires_grad


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_head_factor_uses_actual_row_weighting_and_probe_has_no_mean_scale(diagnostic):
    generator = torch.Generator().manual_seed(813)
    action = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    returns = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    eligible = torch.ones_like(action, dtype=torch.bool)
    kwargs = dict(diagnostic=diagnostic, action_scale=.2)
    full = exact_head_ggn_factor(action, returns, eligible, **kwargs)
    first = exact_head_ggn_factor(action[:2], returns[:2], eligible[:2], **kwargs)
    second = exact_head_ggn_factor(action[2:], returns[2:], eligible[2:], **kwargs)
    torch.testing.assert_close(full, (2*first+5*second)/7, rtol=1.e-13, atol=1.e-14)
    probe = torch.ones_like(action)
    together = output_curvature_sqrt_probe(action, returns, eligible, probe, **kwargs)
    separate = torch.cat([output_curvature_sqrt_probe(action[index:index+1], returns[index:index+1],
        eligible[index:index+1], probe[index:index+1], **kwargs) for index in range(7)])
    torch.testing.assert_close(together, separate, rtol=0., atol=0.)


def test_rank_one_sqrt_is_well_defined_at_zero_returns_and_zero_lambda():
    action = torch.zeros((3, 2), dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, True], [False, True]])
    probe = torch.ones_like(action)
    for coefficient in (0., 1.):
        actual = output_curvature_sqrt_probe(action, action, mask, probe,
            diagnostic='squared', action_scale=.2, squared_lambda=coefficient)
        assert torch.equal(actual, probe*mask)


@pytest.mark.parametrize('exponent', [1000., -1000.])
def test_entropy_unrepresentable_weights_are_refused_not_clipped(exponent):
    action = torch.ones((1, 2), dtype=torch.float64)
    returns = torch.full_like(action, (1.-exponent)/2.)
    with pytest.raises(FloatingPointError, match='unrepresentable'):
        output_curvature_sqrt_probe(action, returns, torch.ones_like(action, dtype=torch.bool),
            action, diagnostic='entropy', action_scale=1.)


@pytest.mark.parametrize('field,value', [('relative_factor_floor', 0.), ('relative_factor_floor', 1.),
    ('absolute_factor_floor', 0.), ('absolute_factor_floor', float('nan'))])
def test_bad_numerical_floor_configs_are_rejected(field, value):
    with pytest.raises(ValueError):
        KroneckerBaseConfig(**{field: value})


def test_positive_factor_preserves_all_near_null_directions_without_objective_ridge():
    moment = torch.diag(torch.tensor([0., 1.e-20, 2.], dtype=torch.float64))
    config = KroneckerBaseConfig(relative_factor_floor=1.e-8, absolute_factor_floor=1.e-16)
    factor, report = positive_inverse_factor(moment, config=config)
    inverse = factor@factor.T
    torch.testing.assert_close(inverse, torch.diag(torch.tensor([5.e7, 5.e7, .5], dtype=torch.float64)))
    assert torch.linalg.eigvalsh(inverse).min() > 0.
    assert report['eigenvalues_floored_for_preconditioning'] == 2
    assert not report['objective_ridge_added'] and not report['coefficient_directions_discarded']


@pytest.mark.parametrize('case', ['negative', 'asymmetric', 'nan', 'dtype', 'nonsquare'])
def test_invalid_training_moments_fail_closed(case):
    matrix = torch.eye(2, dtype=torch.float64)
    if case == 'negative':
        matrix[0, 0] = -1.e-20
        matrix[1, 1] = 1.e-20
    elif case == 'asymmetric':
        matrix[0, 1] = .1
    elif case == 'nan':
        matrix[0, 0] = float('nan')
    elif case == 'dtype':
        matrix = matrix.float()
    else:
        matrix = matrix[:, :1]
    with pytest.raises((ValueError, FloatingPointError)):
        positive_inverse_factor(matrix, config=KroneckerBaseConfig())


def test_layerwise_base_matches_dense_kronecker_solve_including_biases():
    optimizer, policy, moments = fixture()
    operator = FrozenKroneckerLBFGS(optimizer, policy, moments=moments, factor_data_identity='a'*64)
    vector = torch.sin(torch.arange(optimizer.metric.size, dtype=torch.float64))
    image, energy = operator.base_and_energy(vector)
    for name, (activation, output) in moments.items():
        weight, bias = operator.layouts[name]
        block = torch.cat((vector[weight[0]].reshape(weight[1]), vector[bias[0]][:, None]), dim=1)
        expected = torch.linalg.solve(output, torch.linalg.solve(activation, block.T).T)
        torch.testing.assert_close(image[weight[0]].reshape(weight[1]), expected[:, :-1], rtol=1.e-12, atol=1.e-12)
        torch.testing.assert_close(image[bias[0]], expected[:, -1], rtol=1.e-12, atol=1.e-12)
    torch.testing.assert_close(energy, torch.dot(vector, image), rtol=1.e-12, atol=1.e-12)
    assert float(energy) > 0.


def test_every_raw_secant_replays_over_new_base_and_preserves_original_native_state():
    optimizer, policy, moments = fixture()
    point, history, rng = action_point_sha256(policy), optimizer_state_sha256(optimizer), torch.get_rng_state().clone()
    operator = FrozenKroneckerLBFGS(optimizer, policy, moments=moments, factor_data_identity='b'*64)
    size = optimizer.metric.size
    identity = torch.eye(size, dtype=torch.float64)
    dense = torch.stack([operator.base_and_energy(column)[0] for column in identity], dim=1)
    for step, change, reciprocal in optimizer.history:
        update = identity-reciprocal*step[:, None]*change[None, :]
        dense = update@dense@update.T+reciprocal*step[:, None]*step[None, :]
    vector = torch.cos(torch.arange(size, dtype=torch.float64))
    image, energy = operator.apply_and_energy(vector)
    torch.testing.assert_close(image, dense@vector, rtol=1.e-11, atol=1.e-12)
    torch.testing.assert_close(energy, vector@dense@vector, rtol=1.e-11, atol=1.e-12)
    torch.testing.assert_close(dense, dense.T, rtol=1.e-11, atol=1.e-12)
    assert torch.linalg.eigvalsh(dense).min() > 0.
    assert len(operator.frozen.history) == len(optimizer.history) == 3
    assert optimizer_state_sha256(optimizer) == history and action_point_sha256(policy) == point
    assert torch.equal(torch.get_rng_state(), rng)
    assert operator.identity['all_raw_secants_retained'] and operator.identity['original_stopping_metric_unchanged']
    assert not operator.identity['native_optimizer_changed']


@pytest.mark.parametrize('case', ['missing_layer', 'wrong_shape', 'bad_identity'])
def test_reduced_or_unidentified_factor_geometry_is_rejected(case):
    optimizer, policy, moments = fixture()
    if case == 'missing_layer':
        moments.pop(next(iter(moments)))
    elif case == 'wrong_shape':
        name = next(iter(moments))
        moments[name] = (torch.eye(1, dtype=torch.float64), moments[name][1])
    with pytest.raises(ValueError):
        FrozenKroneckerLBFGS(optimizer, policy, moments=moments,
            factor_data_identity='invalid' if case == 'bad_identity' else 'c'*64)


@pytest.mark.parametrize('case', ['factor', 'factor_object', 'native_factor', 'weight', 'weight_object', 'affine_only', 'buffer'])
def test_mutating_frozen_factor_or_source_policy_is_rejected(case):
    optimizer, policy, moments = fixture()
    operator = FrozenKroneckerLBFGS(optimizer, policy, moments=moments, factor_data_identity='d'*64)
    if case == 'factor':
        next(iter(operator.factors.values()))[0].add_(.1)
    elif case == 'factor_object':
        name = next(iter(operator.factors))
        activation, output = operator.factors[name]
        operator.factors[name] = (activation.clone(), output)
    elif case == 'native_factor':
        optimizer.metric.factors.add_(.1)
    elif case == 'weight':
        with torch.no_grad():
            policy.deep.I0.weight.add_(.1)
    elif case == 'weight_object':
        policy.deep.I0.weight = torch.nn.Parameter(policy.deep.I0.weight.clone())
    elif case == 'affine_only':
        policy.affine_only = True
    else:
        policy.register_buffer('unexpected', torch.zeros(1, dtype=torch.float64))
    with pytest.raises(ValueError):
        operator.apply_and_energy(torch.ones(optimizer.metric.size, dtype=torch.float64))


@pytest.mark.parametrize('case', ['dtype', 'shape', 'nan'])
def test_invalid_solve_vectors_are_rejected_before_raw_history_dot_products(case):
    optimizer, policy, moments = fixture()
    operator = FrozenKroneckerLBFGS(optimizer, policy, moments=moments, factor_data_identity='e'*64)
    vector = torch.ones(optimizer.metric.size, dtype=torch.float64)
    if case == 'dtype':
        vector = vector.float()
    elif case == 'shape':
        vector = vector[:-1]
    else:
        vector[0] = float('nan')
    with pytest.raises(ValueError, match='solve vector'):
        operator.apply_and_energy(vector)


def test_public_base_refuses_unrepresentable_energy_for_finite_input():
    optimizer, policy, moments = fixture()
    operator = FrozenKroneckerLBFGS(optimizer, policy, moments=moments, factor_data_identity='f'*64)
    vector = torch.full((optimizer.metric.size,), 1.e200, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match='unrepresentable Kronecker base'):
        operator.base_and_energy(vector)


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_curvature_helpers_reject_unmasked_logits_instead_of_changing_the_canonical_action(diagnostic):
    action = torch.ones((2, 2), dtype=torch.float64)
    eligible = torch.tensor([[True, False], [True, True]])
    with pytest.raises(ValueError, match='already eligibility-masked'):
        output_curvature_sqrt_probe(action, action, eligible, action, diagnostic=diagnostic, action_scale=.2)
    with pytest.raises(ValueError, match='already eligibility-masked'):
        exact_head_ggn_factor(action, action, eligible, diagnostic=diagnostic, action_scale=.2)
