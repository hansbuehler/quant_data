"""No dated fixing node, clock capacity, causality and scan gradients."""

import copy

import numpy as np
import torch

from quantdata.sdlv.dhgromm1.time_decider import (
    TimeDeciderConfig,
    TimeDeciderHedge,
    TimeDeciderWithA10Hedge,
)
from quantdata.torch import BaseEnvironment


def policy():
    torch.manual_seed(123)
    return TimeDeciderHedge(BaseEnvironment(device="cpu", precision=64),
        config=TimeDeciderConfig(label="test", hidden_size=3, trade_l1=0, trade_l2=0),
        price_mean=torch.zeros(21), price_scale=torch.ones(21), return_mean=torch.zeros(20, 21))


def test_no_a10_parameter_and_generic_clock_can_represent_every_daily_schedule():
    model = policy()
    assert not hasattr(model, "day10")
    assert not any("day10" in key or "event_heads" in key for key in model.state_dict())
    time = torch.arange(20, dtype=torch.float64)[:, None] / 20
    clock = torch.cat((torch.ones_like(time), time, model.clock(time)), dim=-1)
    assert torch.linalg.matrix_rank(clock).item() == 20
    target = torch.eye(20, dtype=torch.float64)[:, 10]
    weights = torch.linalg.lstsq(clock, target).solution
    torch.testing.assert_close(clock @ weights, target, atol=1e-12, rtol=1e-12)


def test_zero_action_initialization_has_nonzero_live_head_gradients():
    model = policy()
    state = torch.randn(4, model.trade_head.in_features - 21, dtype=torch.float64)
    state[:, 0] = .5
    action = model.action(state)
    torch.testing.assert_close(action, torch.zeros_like(action), atol=0, rtol=0)
    action.sum().backward()
    assert model.trade_head.weight.grad.abs().max() > 0
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.deep_head.parameters())
    assert all(p.grad is None for p in model.reference_deep_head.parameters())


def test_causal_no_a10_scan_matches_eager_values_and_gradients():
    model = policy()
    with torch.no_grad():
        model.trade_head.weight.add_(torch.randn_like(model.trade_head.weight) * 1e-4)
    reference = copy.deepcopy(model)
    prices = torch.ones(4, 21, 21, dtype=torch.float64)
    changes = torch.zeros(4, 20, 420, dtype=torch.float64)
    returns = torch.randn(4, 20, 21, dtype=torch.float64) * .01
    actual = model(prices, changes, returns=returns)
    expected = reference(prices, changes, returns=returns, eager=True)
    torch.testing.assert_close(actual.trades, expected.trades, atol=1e-12, rtol=1e-10)
    actual.gain.square().mean().backward()
    expected.gain.square().mean().backward()
    for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-11, rtol=1e-8, msg=name)
    future = prices.clone()
    future[:, 11:] *= 2
    before = model(prices, changes, returns=returns, eager=True)
    later = model(future, changes, returns=returns + 1, eager=True)
    np.testing.assert_array_equal(before.trades[:, :11].detach(), later.trades[:, :11].detach())


def test_matched_with_a10_changes_only_the_explicit_fixing_action():
    original = policy()
    other = TimeDeciderWithA10Hedge(BaseEnvironment(device="cpu",precision=64),config=original.config,
        price_mean=torch.zeros(21),price_scale=torch.ones(21),return_mean=torch.zeros(20,21))
    incompatible = other.load_state_dict(original.state_dict(),strict=False)
    assert incompatible.missing_keys==["day10"] and not incompatible.unexpected_keys
    with torch.no_grad():
        other.day10.fill_(1)
    state = torch.randn(20,original.trade_head.in_features-21,dtype=torch.float64)
    state[:,0] = torch.arange(20,dtype=torch.float64)/20
    difference = other.action(state)-original.action(state)
    expected = torch.zeros_like(difference)
    expected[10] = 1
    torch.testing.assert_close(difference,expected,atol=0,rtol=0)
