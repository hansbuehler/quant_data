"""Causal fee-aware output coordinates retain price, cost and premium limits."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from quantdata.hedge2.terminal_policy import (
    PolicyBook,
    TerminalPolicy,
    TerminalPolicyConfig,
    fee_preconditioned_positions,
)
from quantdata.hedge2.terminal_torch import TensorTerminalPaths, evaluate_tensor_policy
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_tree import two_step_tree


def test_preconditioned_formula_preserves_the_pointwise_premium_box() -> None:
    raw = torch.tensor([-.2, 0., .7, 10000.], dtype=torch.float64, requires_grad=True)
    prices = torch.tensor([.01, .1, .8, 2.], dtype=torch.float64)
    costs = torch.tensor([.002, .001, 0., .1], dtype=torch.float64)
    positions = fee_preconditioned_positions(raw, prices, costs, instrument_cap=10.)
    expected = 10. * torch.tanh(raw * prices / (10. * (prices + costs))) / prices
    torch.testing.assert_close(positions, expected, rtol=1e-14, atol=1e-14)
    assert torch.all(torch.abs(positions * prices) <= 10.)
    positions.sum().backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_fee_reduces_mathematically_to_the_original_output(dtype) -> None:
    raw = torch.tensor([-30., -.2, 0., .7, 40.], dtype=dtype)
    prices = torch.tensor([1e-12, .1, 1., .2, 2.], dtype=dtype)
    actual = fee_preconditioned_positions(raw, prices, torch.zeros_like(prices), instrument_cap=10.)
    expected = 10. * torch.tanh(raw / 10.) / prices
    torch.testing.assert_close(actual, expected)


def test_tiny_price_has_no_hidden_floor_and_fee_derivative_is_finite_at_zero() -> None:
    raw = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    prices = torch.tensor([1e-30, 0., 1.], dtype=torch.float32)
    costs = torch.tensor([1e-4, 1e-4, 0.], dtype=torch.float32)
    positions = fee_preconditioned_positions(raw, prices, costs, instrument_cap=10.)
    positions.sum().backward()
    assert torch.isfinite(raw.grad).all()
    torch.testing.assert_close(raw.grad, torch.tensor([1e4, 0., 1.]), rtol=1e-6, atol=0.)
    assert torch.all(costs * raw.grad <= 1. + 1e-6)


def test_large_raw_output_can_still_reach_full_cap_when_fees_are_large() -> None:
    prices = torch.tensor([1e-30], dtype=torch.float32)
    costs = torch.tensor([1e-4], dtype=torch.float32)
    raw = 200. * (prices + costs) / prices
    positions = fee_preconditioned_positions(raw, prices, costs, instrument_cap=10.)
    assert torch.isfinite(positions).all()
    torch.testing.assert_close(positions * prices, torch.tensor([10.]), rtol=1e-6, atol=0.)


def test_tanhc_series_preserves_representable_holdings_when_allocation_underflows() -> None:
    raw = torch.tensor([1e-100], dtype=torch.float64, requires_grad=True)
    prices, costs = torch.tensor([1e-300], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    positions = fee_preconditioned_positions(raw, prices, costs, instrument_cap=10.)
    torch.testing.assert_close(positions, raw, rtol=1e-14, atol=0.)
    positions.sum().backward()
    torch.testing.assert_close(raw.grad, torch.ones_like(raw))


def test_preconditioned_gradient_matches_finite_differences() -> None:
    raw = torch.tensor([0., .2, -.3], dtype=torch.float64, requires_grad=True)
    prices = torch.tensor([.01, .3, .1], dtype=torch.float64)
    costs = torch.tensor([.001, .02, .05], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda value: fee_preconditioned_positions(value, prices, costs, instrument_cap=10.), (raw,))


def test_tanhc_series_matches_value_and_derivative_on_both_sides_of_cutoff() -> None:
    argument = torch.tensor([1e-4 * (1. - 1e-7), 1e-4 * (1. + 1e-7)], dtype=torch.float64)
    prices, costs = torch.full_like(argument, .3), torch.full_like(argument, .02)
    raw = (10. * argument * (prices + costs) / prices).requires_grad_()
    positions = fee_preconditioned_positions(raw, prices, costs, instrument_cap=10.)
    torch.testing.assert_close(positions, 10. * torch.tanh(argument) / prices, rtol=1e-14, atol=0.)
    positions.sum().backward()
    torch.testing.assert_close(raw.grad, (1. - torch.tanh(argument).square()) / (prices + costs), rtol=1e-14, atol=0.)


def test_nonzero_inventory_turnover_gradients_match_direct_two_step_accounting() -> None:
    paths = two_step_tree()
    paths = replace(paths, costs=np.full_like(paths.costs, .03))
    env = BaseEnvironment(device="cpu", precision=64)
    tensor_paths = TensorTerminalPaths.from_paths(paths, env=env)
    prices = torch.ones_like(tensor_paths.gains)
    prices[:, 1, 0] += .1 * tensor_paths.features[:, 1, 0]
    initial = torch.tensor([.2], dtype=env.fdtype)
    actual_raw = torch.tensor([.1, .3], dtype=env.fdtype, requires_grad=True)
    expected_raw = actual_raw.detach().clone().requires_grad_()

    def evaluate(parameters, preconditioned):
        def policy(features, inventory, step):
            raw = parameters[step] + .2 * inventory
            price, cost = prices[:, step], tensor_paths.costs[:, step]
            return (fee_preconditioned_positions(raw, price, cost, instrument_cap=10.) if preconditioned else
                10. * torch.tanh(raw * price / (10. * (price + cost))) / price)

        result = evaluate_tensor_policy(paths=tensor_paths, initial_inventory=initial, policy=policy, target_side=-1)
        return result, result.wealth.square().mean() + result.entry_cost.mean()

    actual, actual_loss = evaluate(actual_raw, True)
    expected, expected_loss = evaluate(expected_raw, False)
    torch.testing.assert_close(actual.wealth, expected.wealth, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual.entry_cost, expected.entry_cost, rtol=1e-13, atol=1e-13)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_raw.grad, expected_raw.grad, rtol=1e-12, atol=1e-12)
    for index in range(len(actual_raw)):
        higher, lower = actual_raw.detach().clone(), actual_raw.detach().clone()
        higher[index] += 1e-6
        lower[index] -= 1e-6
        difference = (evaluate(higher, True)[1] - evaluate(lower, True)[1]) / 2e-6
        torch.testing.assert_close(actual_raw.grad[index], difference, rtol=1e-6, atol=1e-8)


def test_optional_model_choice_is_explicit_and_zero_price_liquidation_is_unchanged() -> None:
    env = BaseEnvironment(device="cpu", precision=64, debug=True)
    base = TerminalPolicyConfig(label="fee-test", feature_count=1, use_deep=False)
    candidate = replace(base, fee_preconditioning=True)
    assert not base.fee_preconditioning and candidate.filename != base.filename
    policy = TerminalPolicy(env=env, config=candidate)
    with torch.no_grad():
        policy.affine.I0.bias.fill_(1.)
    book = PolicyBook(prices=np.array([[[1., 1e-30, 0.]]]),
        asset_descriptors=np.array([[1., 0., 0., 0.], [0., 0., 1., -.1], [0., 0., 0., .1]]),
        initial_business_dtes=np.array([0., 2., 2.]))
    positions = policy(torch.zeros((1, 1), dtype=env.fdtype), torch.zeros((1, 3), dtype=env.fdtype), 0,
        book=book.tensor(env), costs=torch.tensor([[0., 1e-4, 1e-4]], dtype=env.fdtype))
    assert positions[0, 2] == 0.
    torch.testing.assert_close(positions[0, 1], torch.tensor(1e4, dtype=env.fdtype))
    assert positions.dtype == env.fdtype


def test_preconditioner_cannot_be_enabled_with_an_ambiguous_configuration_value() -> None:
    with pytest.raises(ValueError, match="explicit boolean"):
        TerminalPolicyConfig(label="invalid-fee", feature_count=1, fee_preconditioning=1)
