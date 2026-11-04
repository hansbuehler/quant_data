"""Exact-date support, warm preservation, causal feedback and event gradients."""

import copy

import pytest
import torch

from quantdata.sdlv.dhgromm1.events import (
    EventFeedbackConfig,
    EventFeedbackHedge,
    EventJointConfig,
    EventJointHedges,
    exercise_event_heads,
    initialize_events,
)
from quantdata.sdlv.dhgromm1.joint import JointConfig, JointHedges, load_joint_checkpoint
from quantdata.torch import BaseEnvironment


def policy() -> EventFeedbackHedge:
    torch.manual_seed(82)
    return EventFeedbackHedge(BaseEnvironment(device="cpu", precision=64),
        config=EventFeedbackConfig(label="long", latent_width=4, hidden_size=3),
        price_mean=torch.zeros(21), price_scale=torch.ones(21))


def test_new_heads_are_initially_zero_but_have_deep_parameter_gradients() -> None:
    model = policy()
    state = torch.randn(20, model.trade_head.in_features, dtype=torch.float64) * .1
    state[:, 0] = torch.arange(20, dtype=torch.float64) / 20
    action = model.action(state)
    torch.testing.assert_close(action, torch.zeros_like(action), rtol=0, atol=0)
    action.sum().backward()
    for head in model.event_heads.values():
        assert head.affine.weight.grad.abs().sum() > 0
        assert head.deep.I0.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in head.reference_affine.parameters())
        assert all(p.grad is None for p in head.reference_deep.parameters())
    # The background strategy is explicitly NOT frozen.
    assert model.trade_head.weight.grad.abs().sum() > 0
    assert model.deep_head.I0.weight.grad.abs().sum() > 0
    assert model.recurrence.weight_ih.requires_grad
    assert model.gate.I0.weight.requires_grad
    assert model.translation.requires_grad


def test_direct_event_actions_have_exact_date_support() -> None:
    model = policy()
    state = torch.randn(20, model.trade_head.in_features, dtype=torch.float64)
    state[:, 0] = torch.arange(20, dtype=torch.float64) / 20
    with torch.no_grad():
        model.event_heads["0"].affine.bias.add_(1)
        model.event_heads["10"].affine.bias.add_(2)
    actual = model.action(state)
    expected = torch.zeros_like(actual)
    expected[0] = 1
    expected[10] = 2
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert not ({p.data_ptr() for p in model.event_heads["0"].parameters()}
                & {p.data_ptr() for p in model.event_heads["10"].parameters()})


def test_event_scan_and_reference_agree_in_values_and_all_gradients() -> None:
    eager = policy()
    with torch.no_grad():
        eager.trade_head.weight.add_(torch.randn_like(eager.trade_head.weight) * 1e-4)
        eager.event_heads["0"].affine.bias.add_(.001)
        eager.event_heads["10"].deep.I0.weight.add_(.001)
    candidate = copy.deepcopy(eager)
    price = .8 + torch.rand(4, 21, 21, dtype=torch.float64)
    latent = .1 * torch.randn(4, 20, 4, dtype=torch.float64)
    returns = .01 * torch.randn(4, 20, 21, dtype=torch.float64)
    a = eager(price, latent, returns=returns, eager=True)
    b = candidate(price, latent, returns=returns)
    torch.testing.assert_close(a.trades, b.trades, rtol=1e-10, atol=1e-12)
    (a.gain.square() + a.penalty).mean().backward()
    (b.gain.square() + b.penalty).mean().backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), candidate.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-11, msg=name)


def test_day10_head_cannot_change_earlier_actions() -> None:
    model = policy()
    original = copy.deepcopy(model)
    with torch.no_grad():
        model.event_heads["10"].affine.bias.add_(.02)
    price = torch.ones(2, 21, 21, dtype=torch.float64)
    latent = torch.randn(2, 20, 4, dtype=torch.float64)
    returns = torch.randn(2, 20, 21, dtype=torch.float64) * .01
    a, b = (m(price, latent, returns=returns, eager=True) for m in (model, original))
    torch.testing.assert_close(a.trades[:, :10], b.trades[:, :10], rtol=0, atol=0)
    torch.testing.assert_close(a.trades[:, 10] - b.trades[:, 10], torch.full((2, 21), .02, dtype=torch.float64))


def test_explicit_migration_preserves_shared_policy_and_rejects_wrong_restart(tmp_path) -> None:
    from dataclasses import asdict

    torch.manual_seed(91)
    env = BaseEnvironment(device="cpu", precision=32)
    old_config = JointConfig(label="bid-ask")
    old = JointHedges(env, config=old_config, means={k:torch.zeros(21) for k in ("long", "short")},
                      scales={k:torch.ones(21) for k in ("long", "short")})
    with torch.no_grad():
        old.long.trade_head.bias.add_(.001)
        old.short.trade_head.bias.sub_(.001)
    source = tmp_path / "base.pt"
    torch.save(dict(model_code_version=type(old_config).version.unique_id48, user_version=old_config.user_version,
        model_config=asdict(old_config), best_state=old.state_dict(), selected_epoch=386), source)
    config = EventJointConfig(label="bid-ask")
    model, epochs = initialize_events(source, env=env, config=config)
    assert epochs == {"long":386, "short":386}
    price = .8 + torch.rand(2, 21, 21)
    latent = .1 * torch.randn(2, 20, 5003)
    ret = .01 * torch.randn(2, 20, 21)
    payoff = torch.tensor([.02, .03])
    torch.testing.assert_close(model(price, latent, ret, payoff, eager=True),
                               old(price, latent, ret, payoff, eager=True), rtol=0, atol=0)
    with pytest.raises(ValueError, match="objective, version or geometry differs"):
        load_joint_checkpoint(source, config=config)
    # Every shared trainable parameter remains trainable after migration.
    for name, p in old.named_parameters():
        assert dict(model.named_parameters())[name].requires_grad == p.requires_grad


def test_both_event_policies_and_all_four_heads_are_independent() -> None:
    config = EventJointConfig(label="test", long=EventFeedbackConfig(label="long", latent_width=4, hidden_size=3),
        short=EventFeedbackConfig(label="short", latent_width=4, hidden_size=3))
    model = EventJointHedges(BaseEnvironment(device="cpu", precision=64), config=config,
        means={k:torch.zeros(21) for k in ("long", "short")}, scales={k:torch.ones(21) for k in ("long", "short")})
    exercise_event_heads(model)
    price = .8 + torch.rand(4, 21, 21, dtype=torch.float64)
    latent = .1 * torch.randn(4, 20, 4, dtype=torch.float64)
    ret = .01 * torch.randn(4, 20, 21, dtype=torch.float64)
    model(price, latent, ret, torch.tensor([.01, .02, .03, .04], dtype=torch.float64), eager=True).sum().backward()
    assert not ({p.data_ptr() for p in model.long.parameters()} & {p.data_ptr() for p in model.short.parameters()})
    for policy in (model.long, model.short):
        for head in policy.event_heads.values():
            assert head.affine.weight.grad.abs().sum() > 0
            assert head.deep.I0.weight.grad.abs().sum() > 0
        assert policy.recurrence.weight_ih.grad.abs().sum() > 0
        assert policy.gate.I0.weight.grad.abs().sum() > 0
