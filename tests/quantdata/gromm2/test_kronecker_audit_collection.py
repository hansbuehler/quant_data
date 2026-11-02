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
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256
from quantdata.gromm2.kronecker_audit_collection import (
    KroneckerCollectionConfig,
    collect_kronecker_moments,
    layerwise_ggn_block,
)
from quantdata.gromm2.kronecker_audit_preconditioner import exact_head_ggn_factor
from quantdata.torch import BaseEnvironment


def fixture(batch_size=3):
    generator = torch.Generator().manual_seed(923)
    with torch.random.fork_rng():
        torch.manual_seed(924)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='kronecker-collection-test', feature_count=2, instrument_count=2,
                action_scale=.3, network=ActionNetworkConfig(num_depth=2, num_width=2)))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.copy_(torch.randn(parameter.shape, dtype=torch.float64, generator=generator)*.3)
            parameter.grad = torch.randn(parameter.shape, dtype=torch.float64, generator=generator)
    states = torch.randn((8, 2), dtype=torch.float32, generator=generator)
    returns = torch.randn((8, 2), dtype=torch.float32, generator=generator)
    problem = AffineAuditProblem(states=states, returns=returns,
        eligible=torch.tensor([[True, False], [True, True], [False, False]]),
        rows=torch.tensor([0, 1, 2, 6, 7]), action_scale=.3, config=AffineSolveConfig(batch_size=batch_size))
    return policy, problem


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_exhaustive_layer_probe_covariances_match_exact_masked_ggn_including_hidden_layers(diagnostic):
    policy, problem = fixture()
    features, returns = problem.states[:3].double(), problem.returns[:3].double()
    mask = problem.eligible
    signs = torch.tensor(list(product((-1., 1.), repeat=2)), dtype=torch.float64)
    block = layerwise_ggn_block(policy=policy, features=features, returns=returns,
        eligible=mask, probes=signs[:, None, :].expand(-1, len(features), -1), diagnostic=diagnostic)
    layers = [('deep.I0', policy.deep.I0), *[(f'deep.In.{index}', module) for index, module in enumerate(policy.deep.In)]]
    with torch.no_grad():
        action = policy(features, mask)
        activations, preactivations = [], []
        current = features
        for index, (name, module) in enumerate(layers):
            activations.append(torch.cat((current, torch.ones_like(current[:, :1])), dim=1))
            current = module(current)
            preactivations.append(current)
            if index+1 < len(layers):
                current = policy.deep.activation(current)
    expected = {name: torch.zeros((module.out_features, module.out_features), dtype=torch.float64) for name, module in layers}
    for row in range(len(features)):
        output_curvature = torch.autograd.functional.hessian(lambda value: local_objective(value[None, :],
            returns[row:row+1], diagnostic=diagnostic, action_scale=policy.config.action_scale), action[row])
        for index, (name, module) in enumerate(layers):
            def tail(preactivation):
                result = preactivation
                for next_index in range(index+1, len(layers)):
                    result = layers[next_index][1](policy.deep.activation(result))
                return result*mask[row]

            jacobian = torch.autograd.functional.jacobian(tail, preactivations[index][row])
            expected[name] += jacobian.T@output_curvature@jacobian
    for index, (name, module) in enumerate(layers):
        torch.testing.assert_close(block['probe_output_sums'][name].mean(dim=0), expected[name], rtol=1.e-12, atol=1.e-13)
        torch.testing.assert_close(block['activation_sums'][name], activations[index].T@activations[index], rtol=0., atol=0.)
        assert not block['probe_output_sums'][name].requires_grad
    torch.testing.assert_close(block['exact_head_sum'], expected[block['head_name']], rtol=1.e-12, atol=1.e-13)
    assert block['observations'] == 3
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in policy.modules())


@pytest.mark.parametrize('diagnostic', ['squared', 'entropy'])
def test_all_row_collection_is_reproducible_training_only_and_preserves_native_state_and_rng(diagnostic):
    policy, problem = fixture()
    problem.states[3:6] = float('nan')
    problem.returns[3:6] = float('nan')
    point, rng = action_point_sha256(policy), torch.get_rng_state().clone()
    gradients = {name: parameter.grad.clone() for name, parameter in policy.named_parameters()}
    kwargs = dict(policy=policy, problem=problem, diagnostic=diagnostic, data_identity='a'*64,
        source_point_sha256=point, config=KroneckerCollectionConfig(probe_seed=916, probe_count=2))
    first, first_report = collect_kronecker_moments(**kwargs)
    second, second_report = collect_kronecker_moments(**kwargs)
    assert first_report['factor_data_identity'] == second_report['factor_data_identity']
    for name in first:
        for left, right in zip(first[name], second[name], strict=True):
            assert torch.equal(left, right)
    selected = problem.states[problem.rows].double()
    augmented = torch.cat((selected, torch.ones_like(selected[:, :1])), dim=1)
    torch.testing.assert_close(first['deep.I0'][0], augmented.T@augmented/len(problem.rows), rtol=1.e-14, atol=1.e-14)
    mask = problem.eligible[problem.rows % len(problem.eligible)]
    action = policy(selected, mask)
    head = exact_head_ggn_factor(action, problem.returns[problem.rows].double(), mask,
        diagnostic=diagnostic, action_scale=policy.config.action_scale)
    torch.testing.assert_close(first['deep.In.1'][1], head, rtol=1.e-13, atol=1.e-14)
    assert first_report['identity']['training_observations'] == len(problem.rows)
    assert first_report['identity']['resident_dtypes'][:2] == ['torch.float32', 'torch.float32']
    assert first_report['identity']['arithmetic_dtype'] == 'torch.float64'
    assert first_report['complete_factor_collection_forward_passes'] == 1
    assert first_report['complete_probe_adjoint_passes'] == 2
    assert first_report['exact_hessian_vector_products'] == 0 and not first_report['outer_gradient_convergence_certified']
    assert action_point_sha256(policy) == point and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(parameter.grad, gradients[name]) for name, parameter in policy.named_parameters())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in policy.modules())


def test_chunking_and_private_probe_seed_are_explicit_factor_identity_fields():
    policy, problem = fixture()
    kwargs = dict(policy=policy, problem=problem, diagnostic='squared', data_identity='b'*64,
        source_point_sha256=action_point_sha256(policy))
    _, first = collect_kronecker_moments(**kwargs, config=KroneckerCollectionConfig(probe_seed=1))
    _, second = collect_kronecker_moments(**kwargs, config=KroneckerCollectionConfig(probe_seed=2))
    problem.config = AffineSolveConfig(batch_size=4)
    _, third = collect_kronecker_moments(**kwargs, config=KroneckerCollectionConfig(probe_seed=1))
    assert len({value['factor_data_identity'] for value in (first, second, third)}) == 3
    assert first['identity']['working_memory_batch_size'] == 3
    assert third['identity']['working_memory_batch_size'] == 4


@pytest.mark.parametrize('mutation', ['weight', 'rows', 'gradients', 'objective'])
def test_source_or_training_geometry_mutation_during_progress_fails_closed(mutation):
    policy, problem = fixture()

    def progress(event):
        with torch.no_grad():
            if mutation == 'weight':
                policy.deep.I0.weight.add_(.1)
            elif mutation == 'rows':
                problem.rows.add_(0)
            elif mutation == 'gradients':
                policy.deep.I0.weight.grad.add_(.1)
            else:
                problem.squared_lambda = 2.

    with pytest.raises(ValueError, match='changed during collection'):
        collect_kronecker_moments(policy=policy, problem=problem, diagnostic='squared', data_identity='c'*64,
            source_point_sha256=action_point_sha256(policy), progress=progress)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in policy.modules())


def test_observational_hooks_are_removed_when_the_canonical_forward_raises(monkeypatch):
    policy, problem = fixture()

    def invalid(*args, **kwargs):
        raise RuntimeError('deliberate forward failure')

    monkeypatch.setattr(policy.affine, 'forward', invalid)
    with pytest.raises(RuntimeError, match='deliberate forward failure'):
        layerwise_ggn_block(policy=policy, features=problem.states[:2].double(), returns=problem.returns[:2].double(),
            eligible=problem.eligible[:2], probes=torch.ones((1, 2, 2), dtype=torch.float64), diagnostic='squared')
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in policy.modules())


@pytest.mark.parametrize('case', ['source', 'data', 'objective'])
def test_invalid_original_source_identity_is_rejected_before_observation_work(case):
    policy, problem = fixture()
    events = []
    problem.progress = events.append
    with pytest.raises(ValueError):
        collect_kronecker_moments(policy=policy, problem=problem,
            diagnostic='other' if case == 'objective' else 'squared',
            data_identity='invalid' if case == 'data' else 'd'*64,
            source_point_sha256='0'*64 if case == 'source' else action_point_sha256(policy))
    assert not events


@pytest.mark.parametrize('field,value', [('probe_seed', -1), ('probe_seed', True),
    ('probe_count', 0), ('probe_count', 1025), ('probe_count', True)])
def test_invalid_private_probe_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        KroneckerCollectionConfig(**{field: value})
