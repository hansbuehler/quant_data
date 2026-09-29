from __future__ import annotations

import math

import pytest
import torch

from quantdata.dysanos.flex.run_convex_q_conditional_oce import (
    SharedExpiryOceConfig,
    SharedExpiryOcePolicy,
    _fixed_feature_pnl,
    assemble_expiry_slice_actions,
    weighted_log_oce,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_expiry_slice_actions_sum_spot_and_preserve_option_order() -> None:
    raw = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ]
    )

    action = assemble_expiry_slice_actions(raw, maximum_l1=100.0)

    assert action == pytest.approx(torch.tensor([[5.0, 2.0, 3.0, 5.0, 6.0]]))


def test_expiry_slice_actions_apply_one_global_l1_cap() -> None:
    raw = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    action = assemble_expiry_slice_actions(raw, maximum_l1=5.0)

    assert torch.sum(torch.abs(action), dim=1) == pytest.approx(torch.tensor([5.0]))
    assert action == pytest.approx(torch.tensor([[2.0, 1.0, 2.0]]))


def test_weighted_log_oce_matches_direct_weighted_expectation() -> None:
    pnl = torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64, requires_grad=True)
    density = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

    actual = weighted_log_oce(pnl, density)
    expected = torch.log(torch.sum(density * torch.exp(-pnl)) / torch.sum(density))
    actual.backward()

    assert float(actual.detach()) == pytest.approx(float(expected.detach()))
    expected_gradient = -density * torch.exp(-pnl.detach())
    expected_gradient /= torch.sum(density * torch.exp(-pnl.detach()))
    assert pnl.grad == pytest.approx(expected_gradient)


def test_zero_initialized_shared_policy_returns_zero_actions() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    config = SharedExpiryOceConfig(
        label="test-policy",
        state_size=3,
        num_expiries=2,
        num_strikes=2,
        maximum_action_l1=4.0,
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
    )
    policy = SharedExpiryOcePolicy(
        env,
        config=config,
        state_mean=torch.zeros(3, dtype=torch.float64),
        state_scale=torch.ones(3, dtype=torch.float64),
        log_price_mean=torch.zeros(2, dtype=torch.float64),
        log_price_scale=torch.ones(2, dtype=torch.float64),
    )

    action, raw_heads = policy(
        torch.randn(5, 3, dtype=torch.float64),
        torch.exp(torch.randn(5, 2, 2, dtype=torch.float64)),
    )

    assert action.shape == (5, 5)
    assert raw_heads.shape == (5, 2, 3)
    assert torch.count_nonzero(action) == 0
    assert math.isfinite(float(torch.sum(action).detach()))


def test_fixed_feature_pnl_matches_daywise_returns_and_costs() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    config = SharedExpiryOceConfig(
        label="test-fixed-pnl",
        state_size=3,
        num_expiries=2,
        num_strikes=2,
        maximum_action_l1=100.0,
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
    )
    policy = SharedExpiryOcePolicy(
        env,
        config=config,
        state_mean=torch.zeros(3, dtype=torch.float64),
        state_scale=torch.ones(3, dtype=torch.float64),
        log_price_mean=torch.zeros(2, dtype=torch.float64),
        log_price_scale=torch.ones(2, dtype=torch.float64),
    )
    final_layer = policy.network.In[-1]
    with torch.no_grad():
        final_layer.bias.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    returns = torch.tensor(
        [[[0.1, 0.2, -0.1, 0.3, -0.2], [0.0, 0.1, 0.2, -0.1, 0.4]]],
        dtype=torch.float64,
    )
    costs = torch.tensor([0.01, 0.02, 0.02, 0.02, 0.02], dtype=torch.float64)

    pnl, diagnostics = _fixed_feature_pnl(
        policy,
        states=torch.zeros((1, 2, 3), dtype=torch.float64),
        option_prices=torch.ones((1, 2, 2, 2), dtype=torch.float64),
        returns=returns,
        transaction_cost=costs,
    )

    action = torch.tensor([2.0, 2.0, 3.0, 2.0, 3.0], dtype=torch.float64)
    expected = torch.sum(returns[0] * action[None]) - 2.0 * torch.sum(costs * action)
    assert float(pnl.detach()) == pytest.approx(float(expected))
    assert float(diagnostics.mean_action_l1) == pytest.approx(12.0)
    assert float(diagnostics.maximum_absolute_action) == pytest.approx(3.0)
    assert float(diagnostics.cap_active_fraction) == 0.0
    pnl.sum().backward()
    assert final_layer.bias.grad is not None
