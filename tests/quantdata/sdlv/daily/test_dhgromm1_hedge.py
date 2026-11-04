"""Causality, accounting and OCE tests for the v1 LSTM hedge."""

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.hedge import (
    HedgeConfig,
    RecurrentHedge,
    indifference_quotes,
    oce_value,
    power_exponential_utility,
)
from quantdata.torch import BaseEnvironment


def _policy() -> RecurrentHedge:
    config = HedgeConfig(label="test", instruments=2, horizon_days=2, hidden_size=3, gain_mode="daily_positions")
    return RecurrentHedge(BaseEnvironment(device="cpu", precision=64), config=config,
                          price_mean=torch.zeros(2), price_scale=torch.ones(2))


def test_zero_strategy_is_admissible_and_has_zero_empty_oce() -> None:
    policy = _policy()
    prices = torch.tensor([[[1., 0.1], [1.2, 0.2], [1.1, 0.15]]], dtype=torch.float64)
    out = policy(prices)
    assert torch.count_nonzero(out.gain) == torch.count_nonzero(out.penalty) == 0
    assert oce_value(out.gain, policy.translation, 100).item() == 0.0


def test_trades_accumulate_and_penalty_charges_trades_not_positions() -> None:
    policy = _policy()
    with torch.no_grad():
        policy.trade_head.bias.add_(torch.tensor([0.5, -0.2]))
    prices = torch.tensor([[[1., 0.1], [1.2, 0.2], [1.1, 0.15]]], dtype=torch.float64)
    out = policy(prices)
    np.testing.assert_allclose(out.positions.detach(), [[[0.5, -0.2], [1.0, -0.4]]], atol=1e-8)
    assert out.gain.item() == pytest.approx(0.0, abs=1e-8)
    assert out.penalty.item() == pytest.approx(2 * (1e-4 * 0.7 + 1e-8 * 0.29), rel=1e-7)


def test_first_trade_is_causal_and_future_prices_do_not_change_it() -> None:
    policy = _policy()
    with torch.no_grad():
        policy.trade_head.weight.fill_(0.1)
    prices = torch.tensor([[[1., .1], [1.2, .2], [1.1, .15]], [[1., .1], [2., .5], [2.5, 1.]]],
                          dtype=torch.float64)
    out = policy(prices)
    torch.testing.assert_close(out.trades[0, 0], out.trades[1, 0])
    assert not torch.equal(out.trades[0, 1], out.trades[1, 1])


def test_utility_and_indifference_quote_signs() -> None:
    values = torch.tensor([-0.01, 0., 0.01], dtype=torch.float64, requires_grad=True)
    utility = power_exponential_utility(values, 100.0)
    assert utility[0].item() == pytest.approx(-0.015)
    assert utility[2].item() == pytest.approx((1 - np.exp(-1)) / 100)
    utility.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert indifference_quotes(empty_oce=0.001, long_oce=0.020, short_oce=-0.021) == pytest.approx((0.019, 0.022))


def test_action_reads_current_state_and_deep_branch_has_gradient() -> None:
    policy = _policy()
    with torch.no_grad():
        # The first two slots following time are current price features.
        policy.trade_head.weight[0, 1] += 1.0
    prices = torch.tensor([[[1., .1], [1., .1], [1., .1]],
                           [[2., .1], [2., .1], [2., .1]]], dtype=torch.float64)
    out = policy(prices)
    assert out.trades[1, 0, 0].item() - out.trades[0, 0, 0].item() == pytest.approx(1.0)
    out.gain.sum().backward()
    assert policy.deep_head.In[-1].weight.grad is not None
    assert torch.isfinite(policy.deep_head.In[-1].weight.grad).all()


def test_frozen_random_reference_initial_zero_and_hidden_gradient() -> None:
    policy = _policy()
    prices = torch.tensor([[[1., .1], [1.2, .2], [1.1, .15]]], dtype=torch.float64)
    out = policy(prices)
    torch.testing.assert_close(out.trades, torch.zeros_like(out.trades), rtol=0, atol=0)
    out.gain.sum().backward()
    assert torch.count_nonzero(policy.deep_head.I0.weight.grad) > 0
    assert all(p.grad is None for p in policy.reference_deep_head.parameters())
    assert all(p.grad is None for p in policy.reference_trade_head.parameters())


def test_expiry_gains_charge_each_new_trade_once_and_do_not_see_future_returns() -> None:
    config = HedgeConfig(label="expiry-test", instruments=2, horizon_days=2, hidden_size=3)
    policy = RecurrentHedge(BaseEnvironment(device="cpu", precision=64), config=config,
                            price_mean=torch.zeros(2), price_scale=torch.ones(2))
    with torch.no_grad():
        policy.trade_head.bias.add_(torch.tensor([0.5, -0.2]))
    prices = torch.tensor([[[1., .1], [1.2, .2], [1.1, .15]]], dtype=torch.float64)
    returns = torch.tensor([[[.2, -1.], [-.1, .25]]], dtype=torch.float64)
    out = policy(prices, returns=returns)
    assert out.gain.item() == pytest.approx(.5 * .2 + .2 - .5 * .1 - .2 * .25)
    changed = policy(prices, returns=returns * 2)
    torch.testing.assert_close(out.trades, changed.trades)
    torch.testing.assert_close(changed.gain, out.gain * 2)


@pytest.mark.parametrize("mode", ["expiry_trades", "daily_positions"])
@pytest.mark.parametrize("precision", [32, 64])
def test_native_sequence_matches_reference_values_gradients_and_checkpoint(mode: str, precision: int) -> None:
    torch.manual_seed(73)
    config = HedgeConfig(label="native-test", instruments=3, horizon_days=5, hidden_size=4, gain_mode=mode)
    env = BaseEnvironment(device="cpu", precision=precision)
    options = dict(config=config, price_mean=torch.zeros(3), price_scale=torch.ones(3))
    native, reference = RecurrentHedge(env, **options), RecurrentHedge(env, **options)
    with torch.no_grad():
        for parameter in native.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * .02)
    reference.load_state_dict(native.state_dict())
    keys = set(native.state_dict())
    prices = torch.rand((8, 6, 3), dtype=env.fdtype) + .5
    returns = torch.randn((8, 5, 3), dtype=env.fdtype) * .02
    rng = torch.get_rng_state().clone()
    actual = native(prices, returns=returns)
    expected = reference.forward_eager(prices, returns=returns)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert set(native.state_dict()) == keys
    assert all(
        getattr(native._native_lstm(), f"{name}_l0") is getattr(native.recurrence, name)
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
    tolerance = dict(rtol=1e-5, atol=1e-7) if precision == 32 else dict(rtol=1e-10, atol=1e-12)
    for result, baseline in zip(actual, expected):
        torch.testing.assert_close(result, baseline, **tolerance)
    (actual.gain.square() + actual.penalty).mean().backward()
    (expected.gain.square() + expected.penalty).mean().backward()
    for (name, parameter), (other_name, other) in zip(native.named_parameters(), reference.named_parameters()):
        assert name == other_name
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, **tolerance)
