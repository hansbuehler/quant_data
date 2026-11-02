from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.gromm2.conditional_audit import (
    DIAGNOSTICS,
    IndependentLocalActions,
    LocalActionConfig,
    LocalActionStep,
    fold_path_ranges,
    frozen_score_statistics,
    local_objective,
)
from quantdata.torch import BaseEnvironment


@pytest.mark.parametrize('diagnostic', DIAGNOSTICS)
@pytest.mark.parametrize('scale', [1., 1.e-3])
def test_scaled_local_loss_and_gradients_equal_the_original_objective(diagnostic: str, scale: float) -> None:
    generator = torch.Generator().manual_seed(29)
    unit_action = torch.randn((13, 4), generator=generator, dtype=torch.float64, requires_grad=True)
    scaled_returns = torch.randn((13, 4), generator=generator, dtype=torch.float64)
    actual = local_objective(unit_action, scaled_returns, diagnostic=diagnostic, action_scale=scale)
    action = scale * unit_action
    returns = scale * scaled_returns
    gain = (action * returns).sum(axis=-1)
    if diagnostic == 'squared':
        reference = (-gain + .5 * action.square().sum(axis=-1) + .5 * gain.square()).mean() / scale**2
    else:
        reference = torch.expm1(-gain + .5 * action.square().sum(axis=-1)).mean() / scale**2
    torch.testing.assert_close(actual, reference, rtol=2.e-12, atol=1.e-14)
    actual_gradient, = torch.autograd.grad(actual, unit_action, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference, unit_action)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=2.e-12, atol=1.e-14)


def test_squared_population_optimum_uses_the_uncentered_second_moment() -> None:
    returns = torch.tensor([[.3, -.1], [.5, .2], [-.2, .4], [.1, -.2]], dtype=torch.float64)
    mean = returns.mean(axis=0)
    second_moment = returns.T @ returns / len(returns)
    action = torch.linalg.solve(torch.eye(2, dtype=torch.float64) + second_moment, mean).requires_grad_()
    loss = local_objective(action.expand_as(returns), returns, diagnostic='squared', action_scale=1.)
    gradient, = torch.autograd.grad(loss, action)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1.e-14, rtol=0.)


def test_fold_split_and_pair_clustered_daily_average() -> None:
    first, second = fold_path_ranges(100000)
    assert (first.start, first.stop, second.start, second.stop) == (0, 50000, 50000, 100000)
    scores = np.arange(24.).reshape(8, 3)
    pairs = .5 * (scores[::2].mean(axis=1) + scores[1::2].mean(axis=1))
    report = frozen_score_statistics(scores)
    assert report['mean'] == pytest.approx(pairs.mean())
    assert report['standard_error'] == pytest.approx(pairs.std(ddof=1) / 2.)
    with pytest.raises(ValueError):
        fold_path_ranges(7)


@pytest.mark.parametrize('value,status', [(0., 'degenerate_zero_score'), (1., 'degenerate_nonzero_score')])
def test_zero_variance_is_not_reported_as_a_gaussian_test(value: float, status: str) -> None:
    report = frozen_score_statistics(np.full(8, value))
    assert report['test_status'] == status
    assert report['action_conditional_statistic'] is None
    assert report['standard_error'] == 0.


def test_local_cpu_updates_are_independent_and_start_with_zero_actions() -> None:
    env = BaseEnvironment(device='cpu', precision=64)
    config = LocalActionConfig(label='test', feature_count=3, instrument_count=2, action_scale=.01)
    model = IndependentLocalActions(env, config=config)
    features, returns, eligible = torch.ones((8, 3), dtype=torch.float64), torch.ones((8, 2), dtype=torch.float64), torch.ones((8, 2), dtype=torch.float64)
    eligible[:, 1] = 0.
    for policy in model.policies.values():
        assert torch.count_nonzero(policy(features, eligible)) == 0
    optimizers = {name: torch.optim.Adam(policy.parameters(), lr=.001) for name, policy in model.policies.items()}
    step = LocalActionStep(model, optimizers, gradient_clip=5., compile=False, use_cuda_graphs=False)
    before = {name: value.detach().clone() for name, value in model.policies['squared'].named_parameters()}
    result = step(features, returns, eligible, ('entropy',))
    assert set(result) == {'entropy'}
    assert torch.count_nonzero(model.policies['entropy'](features, eligible)[:, 0]) > 0
    assert torch.count_nonzero(model.policies['entropy'](features, eligible)[:, 1]) == 0
    for name, value in model.policies['squared'].named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
