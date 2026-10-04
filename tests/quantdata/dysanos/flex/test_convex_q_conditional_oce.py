from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.run_convex_q_conditional_oce import (
    SharedExpiryOceConfig,
    SharedExpiryOcePolicy,
    _assemble_expiry_slice_action_details,
    _fixed_feature_pnl,
    _full_population_oce_probabilities,
    _trust_region_boundary_distance,
    assemble_expiry_slice_actions,
    weighted_log_oce,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_trust_region_boundary_distance_hits_radius() -> None:
    step = np.array([0.25, -0.5], dtype=np.float64)
    direction = np.array([3.0, 4.0], dtype=np.float64)
    radius = 2.0

    distance = _trust_region_boundary_distance(step, direction, radius)

    assert distance > 0.0
    assert np.linalg.norm(step + distance * direction) == pytest.approx(radius)


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


def test_long_short_actions_charge_gross_legs_and_apply_gross_cap() -> None:
    # Per expiry the heads are [long spot, long option, short spot, short option].
    raw = torch.tensor([[[2.0, 3.0, 1.0, 0.0], [4.0, 0.0, 2.0, 5.0]]])

    action, cost_basis, uncapped_gross = _assemble_expiry_slice_action_details(
        raw,
        maximum_l1=8.5,
        action_parameterization="long-short",
    )

    assert action == pytest.approx(torch.tensor([[1.5, 1.5, -2.5]]))
    assert cost_basis == pytest.approx(torch.tensor([[4.5, 1.5, 2.5]]))
    assert uncapped_gross == pytest.approx(torch.tensor([17.0]))
    assert torch.sum(cost_basis, dim=1) == pytest.approx(torch.tensor([8.5]))


def test_signed_checkpoint_logits_have_exact_long_short_embedding() -> None:
    signed_logits = torch.tensor(
        [[[1.0, -2.0, 0.5], [-3.0, 4.0, -0.25]]],
        dtype=torch.float64,
    )
    signed_action, signed_cost, _ = _assemble_expiry_slice_action_details(
        signed_logits,
        maximum_l1=100.0,
        action_parameterization="signed",
    )
    long_short_logits = torch.cat((signed_logits, -signed_logits), dim=-1)

    long_short_action, long_short_cost, _ = _assemble_expiry_slice_action_details(
        long_short_logits,
        maximum_l1=100.0,
        action_parameterization="long-short",
    )

    assert long_short_action == pytest.approx(signed_action)
    assert long_short_cost == pytest.approx(signed_cost)


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


def test_chunked_full_population_gradient_matches_direct_oce() -> None:
    features = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64)
    density = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)
    direct_parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    direct_objective = direct_parameter * features
    direct_loss = weighted_log_oce(
        direct_objective,
        torch.as_tensor(density, dtype=torch.float64),
    )
    direct_loss.backward()

    chunked_parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    log_loss, probabilities = _full_population_oce_probabilities(
        (chunked_parameter.detach() * features).numpy(),
        density,
    )
    for first, last in ((0, 2), (2, 4)):
        pnl = chunked_parameter * features[first:last]
        probability = torch.as_tensor(probabilities[first:last], dtype=torch.float64)
        (-torch.sum(probability * pnl)).backward()

    assert log_loss == pytest.approx(float(direct_loss.detach()))
    assert float(np.sum(probabilities)) == pytest.approx(1.0)
    assert float(chunked_parameter.grad) == pytest.approx(float(direct_parameter.grad))


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


def test_expiry_specific_linear_skip_directly_spans_normalized_features() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    config = SharedExpiryOceConfig(
        label="test-linear-skip-policy",
        state_size=3,
        num_expiries=2,
        num_strikes=2,
        maximum_action_l1=100.0,
        expiry_specific_linear_skip=True,
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
    assert policy.linear_skip is not None
    with torch.no_grad():
        policy.linear_skip[0].weight[0, 0] = 2.0
        policy.linear_skip[1].weight[1, 3] = -3.0
    states = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float64)
    prices = torch.tensor([[[1.0, 1.0], [math.exp(0.4), 1.0]]], dtype=torch.float64)

    _, raw_heads = policy(states, prices)

    assert float(raw_heads[0, 0, 0].detach()) == pytest.approx(1.0)
    assert float(raw_heads[0, 1, 1].detach()) == pytest.approx(-1.2)
    assert torch.count_nonzero(raw_heads) == 2
    raw_heads.sum().backward()
    assert policy.linear_skip[0].weight.grad is not None
    assert policy.linear_skip[1].weight.grad is not None


def test_calendar_time_is_an_explicit_policy_feature() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    config = SharedExpiryOceConfig(
        label="test-time-policy",
        state_size=3,
        num_expiries=2,
        num_strikes=2,
        maximum_action_l1=100.0,
        expiry_specific_linear_skip=True,
        include_calendar_time=True,
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
    assert policy.linear_skip is not None
    with torch.no_grad():
        policy.linear_skip[0].weight[0, -1] = 2.0

    _, raw_heads = policy(
        torch.zeros((1, 3), dtype=torch.float64),
        torch.ones((1, 2, 2), dtype=torch.float64),
        calendar_time=0.25,
    )

    assert float(raw_heads[0, 0, 0].detach()) == pytest.approx(0.5)
    assert torch.count_nonzero(raw_heads) == 1
    with pytest.raises(ValueError, match="calendar time is required"):
        policy(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.ones((1, 2, 2), dtype=torch.float64),
        )


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


def test_fixed_feature_pnl_adds_old_policy_before_one_global_cap() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    config = SharedExpiryOceConfig(
        label="test-additive-pnl",
        state_size=3,
        num_expiries=2,
        num_strikes=2,
        maximum_action_l1=8.0,
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
    )

    def make_policy(bias: list[float]) -> SharedExpiryOcePolicy:
        result = SharedExpiryOcePolicy(
            env,
            config=config,
            state_mean=torch.zeros(3, dtype=torch.float64),
            state_scale=torch.ones(3, dtype=torch.float64),
            log_price_mean=torch.zeros(2, dtype=torch.float64),
            log_price_scale=torch.ones(2, dtype=torch.float64),
        )
        with torch.no_grad():
            result.network.In[-1].bias.copy_(torch.tensor(bias, dtype=torch.float64))
        return result

    residual = make_policy([1.0, 2.0, 0.0])
    old = make_policy([2.0, 0.0, 3.0])
    old.requires_grad_(False)
    returns = torch.tensor([[[0.1, 0.2, -0.1, 0.3, -0.2]]], dtype=torch.float64)
    costs = torch.full((5,), 0.01, dtype=torch.float64)

    pnl, diagnostics = _fixed_feature_pnl(
        residual,
        states=torch.zeros((1, 1, 3), dtype=torch.float64),
        option_prices=torch.ones((1, 1, 2, 2), dtype=torch.float64),
        returns=returns,
        transaction_cost=costs,
        frozen_policies=(old,),
    )

    # The uncapped aggregate is [6,2,3,2,3], so one global L1 cap scales by 1/2.
    action = torch.tensor([3.0, 1.0, 1.5, 1.0, 1.5], dtype=torch.float64)
    expected = torch.sum(action * returns[0, 0]) - torch.sum(costs * action)
    assert float(pnl.detach()) == pytest.approx(float(expected))
    assert float(diagnostics.mean_action_l1) == pytest.approx(8.0)
    assert float(diagnostics.cap_active_fraction) == pytest.approx(1.0)
    pnl.sum().backward()
    assert residual.network.In[-1].bias.grad is not None
    assert old.network.In[-1].bias.grad is None
