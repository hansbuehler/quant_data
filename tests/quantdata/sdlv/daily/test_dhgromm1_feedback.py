"""Causal full-state/inventory feedback and scan differentiation checks."""

import numpy as np
import torch

from quantdata.sdlv.dhgromm1.feedback import (
    FeedbackConfig,
    FeedbackHedge,
    contract_expiries,
    migrate_policy,
    packed_geometry,
)
from quantdata.sdlv.dhgromm1.hedge import HedgeConfig, RecurrentHedge
from quantdata.torch import BaseEnvironment


def test_ragged_state_keeps_every_live_coordinate() -> None:
    offsets, index, valid = packed_geometry()
    assert offsets[-1] == 62060
    assert index.shape == (20, 5003)
    for t in range(20):
        np.testing.assert_array_equal(index[t, valid[t]], np.arange(offsets[t], offsets[t + 1]))
        assert valid[t].sum() == 5003 - 200 * t
    expiry = contract_expiries()
    np.testing.assert_array_equal(expiry[:, 0], 20)
    np.testing.assert_array_equal(expiry[:, 2], np.arange(1, 21))
    assert expiry[19, 16] == 25


def make_models() -> tuple[RecurrentHedge, FeedbackHedge]:
    torch.manual_seed(43)
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    old = RecurrentHedge(env, config=HedgeConfig(label="long", hidden_size=3),
                         price_mean=torch.zeros(21), price_scale=torch.ones(21))
    with torch.no_grad():
        for p in old.parameters():
            if p.requires_grad:
                p.add_(.01 * torch.randn_like(p))
    new = FeedbackHedge(env, config=FeedbackConfig(label="long", hidden_size=3, latent_width=4),
                        price_mean=old.price_mean, price_scale=old.price_scale)
    migrate_policy(old, new)
    return old, new


def test_migration_preserves_policy_and_exposes_new_input_gradients() -> None:
    old, new = make_models()
    prices = .8 + torch.rand(2, 21, 21, dtype=torch.float64)
    latent = torch.randn(2, 20, 4, dtype=torch.float64)
    returns = torch.randn(2, 20, 21, dtype=torch.float64) * .01
    expected = old.forward_eager(prices, returns=returns)
    actual = new(prices, latent, returns=returns, eager=True)
    torch.testing.assert_close(actual.trades, expected.trades, atol=2e-14, rtol=1e-12)
    actual.gain.sum().backward()
    # All three previously missing inputs are connected to the objective:
    # full latent state, live cohort book, previous trade.
    grad = new.recurrence.weight_ih.grad
    for region in (grad[:, 22:26], grad[:, 26:446], grad[:, 446:]):
        assert region.abs().sum() > 0
    assert new.gate.I0.weight.grad.abs().sum() > 0


def test_scan_matches_eager_values_and_all_parameter_gradients() -> None:
    _, eager = make_models()
    _, compiled = make_models()
    # Enable dependence on inventory and latent inputs, not just the old policy.
    with torch.no_grad():
        eager.recurrence.weight_ih[:, 22:].normal_(std=.001)
        eager.trade_head.weight[:, 25:].add_(.0001)
    compiled.load_state_dict(eager.state_dict())
    prices = .8 + torch.rand(2, 21, 21, dtype=torch.float64)
    latent = torch.randn(2, 20, 4, dtype=torch.float64)
    returns = torch.randn(2, 20, 21, dtype=torch.float64) * .01
    a = eager(prices, latent, returns=returns, eager=True)
    b = compiled(prices, latent, returns=returns)
    torch.testing.assert_close(a.trades, b.trades, atol=1e-12, rtol=1e-10)
    (a.gain.sum() + a.penalty.sum()).backward()
    (b.gain.sum() + b.penalty.sum()).backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), compiled.named_parameters(), strict=True):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=1e-11, rtol=1e-8, msg=name)


def test_future_states_and_returns_never_change_earlier_trades() -> None:
    _, policy = make_models()
    with torch.no_grad():
        policy.trade_head.weight[:, 25:29].normal_(std=.01)
    prices = torch.ones(2, 21, 21, dtype=torch.float64)
    latent = torch.ones(2, 20, 4, dtype=torch.float64)
    latent[1, 11:] *= 3
    returns = torch.randn(2, 20, 21, dtype=torch.float64)
    result = policy(prices, latent, returns=returns, eager=True)
    torch.testing.assert_close(result.trades[0, :11], result.trades[1, :11])
    assert not torch.equal(result.trades[0, 11:], result.trades[1, 11:])


def test_inventory_uses_cohort_units_and_clears_expired_contracts() -> None:
    _, policy = make_models()
    dtype = torch.float64
    # Explicit book: a 1-day call already settled at t=1, a 5-day call live.
    book = torch.zeros(1, 20, 21, dtype=dtype)
    book[0, 0, 2] = 7
    book[0, 0, 7] = 3
    carry = (torch.zeros(1, 3, dtype=dtype), torch.zeros(1, 3, dtype=dtype), book,
             torch.zeros(1, 21, dtype=dtype))
    args = (torch.ones(1, 22, dtype=dtype), torch.full((1, 21), 2., dtype=dtype),
            torch.ones(1, 4, dtype=dtype), policy.live[1], policy.insert[1])
    new, trade = policy._body(carry, args)
    assert new[2][0, 0, 2] == 0
    assert new[2][0, 0, 7] == 3
    torch.testing.assert_close(new[2][0, 1], trade[0] / 2)


def test_normalization_and_resident_gather_preserve_packed_state() -> None:
    from quantdata.sdlv.dhgromm1.feedback_data import state_normalization
    from quantdata.sdlv.dhgromm1.feedback_run import ResidentSample

    offsets, index, valid = packed_geometry()
    packed = np.stack([np.arange(offsets[-1], dtype=np.float32) * .0001 + i for i in range(4)])
    mean, scale = state_normalization(packed, chunk_size=2)
    np.testing.assert_allclose(mean, packed.mean(axis=0), atol=5e-7, rtol=1e-6)
    np.testing.assert_allclose(scale, packed.std(axis=0), atol=5e-7, rtol=1e-6)
    _, constant_scale = state_normalization(np.zeros((4, 3), dtype=np.float32))
    np.testing.assert_array_equal(constant_scale, np.full(3, .001, dtype=np.float32))
    arrays = (np.ones((4, 21, 21), dtype=np.float32),
              np.ones((4, 20, 21), dtype=np.float32), np.arange(4, dtype=np.float32))
    sample = ResidentSample(arrays=arrays, states=packed, mean=mean, scale=scale, device="cpu")
    _, _, payoff, states = sample.batch(torch.tensor([2, 0]))
    np.testing.assert_array_equal(payoff, [2, 0])
    expected = ((packed[[2, 0]] - mean) / scale)[:, index]
    expected[:, ~valid] = 0
    np.testing.assert_array_equal(states, expected)
