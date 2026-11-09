"""Causality, marked-wealth feedback, migration and differentiable scan checks."""

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.events import EventFeedbackConfig, EventJointConfig, EventJointHedges
from quantdata.sdlv.dhgromm1.observable import (
    ObservableConfig,
    ObservableHedge,
    ObservableJointConfig,
    initialize_observable,
    market_geometry,
)
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.torch import BaseEnvironment


def policy() -> ObservableHedge:
    torch.manual_seed(69)
    model = ObservableHedge(BaseEnvironment(device="cpu", precision=64),
        config=ObservableConfig(label="long", hidden_size=3), price_mean=torch.zeros(21), price_scale=torch.ones(21))
    with torch.no_grad():
        model.trade_head.weight.add_(torch.randn_like(model.trade_head.weight) * .001)
        model.event_heads["0"].affine.bias.add_(.001)
        model.event_heads["10"].affine.bias.add_(.002)
    return model


def data():
    price = .8 + torch.rand(2, 21, 21, dtype=torch.float64)
    change = .01 * torch.randn(2, 20, 420, dtype=torch.float64)
    for day in range(20):
        change[:, day, day * 21:] = 0
    ret = .01 * torch.randn(2, 20, 21, dtype=torch.float64)
    return price, change, ret


@pytest.mark.parametrize("cutoff", [0, 9, 10, 18])
def test_future_observations_cannot_change_actions_at_or_before_cutoff(cutoff) -> None:
    model = policy()
    price, change, ret = data()
    future_price, future_change = price.clone(), change.clone()
    future_price[:, cutoff + 1:] *= 3
    future_change[:, cutoff + 1:] += 17
    a = model(price, change, returns=ret, eager=True)
    b = model(future_price, future_change, returns=ret + 100, eager=True)
    torch.testing.assert_close(a.trades[:, :cutoff + 1], b.trades[:, :cutoff + 1], rtol=0, atol=0)
    assert not torch.equal(a.trades[:, cutoff + 1:], b.trades[:, cutoff + 1:])


def test_all_expiry_return_labels_are_excluded_from_every_action() -> None:
    model = policy()
    price, change, ret = data()
    a = model(price, change, returns=ret, eager=True)
    b = model(price, change, returns=ret * 0 + 100, eager=True)
    torch.testing.assert_close(a.trades, b.trades, rtol=0, atol=0)


def test_current_mark_changes_affect_wealth_not_earlier_actions() -> None:
    model = policy()
    price, change, ret = data()
    changed = change.clone()
    changed[:, 10] += .1
    a = model(price, change, returns=ret, eager=True)
    b = model(price, changed, returns=ret, eager=True)
    torch.testing.assert_close(a.trades[:, :10], b.trades[:, :10], rtol=0, atol=0)
    assert not torch.equal(a.trades[:, 10], b.trades[:, 10])


def test_observable_scan_matches_reference_values_and_all_gradients() -> None:
    eager = policy()
    candidate = copy.deepcopy(eager)
    price, change, ret = data()
    a = eager(price, change, returns=ret, eager=True)
    b = candidate(price, change, returns=ret)
    torch.testing.assert_close(a.trades, b.trades, atol=1e-12, rtol=1e-10)
    (a.gain.square() + a.penalty).mean().backward()
    (b.gain.square() + b.penalty).mean().backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), candidate.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-11, rtol=1e-8, msg=name)


def test_market_pack_is_past_only_and_sample_does_not_expose_future_rows() -> None:
    offsets, _, valid = market_geometry()
    assert offsets[-1] == 3990
    assert not valid[0].any()
    packed = np.arange(2 * 3990, dtype=np.float32).reshape(2, 3990)
    arrays = (np.ones((2, 21, 21)), np.ones((2, 20, 21)), np.zeros(2))
    sample = ObservableSample(arrays=arrays, changes=packed, device="cpu")
    _, _, _, marks = sample.batch(slice(None))
    for day in range(20):
        np.testing.assert_array_equal(marks[:, day, :day * 21], packed[:, offsets[day]:offsets[day + 1]])
        assert torch.count_nonzero(marks[:, day, day * 21:]) == 0


def test_explicit_partial_migration_drops_latent_columns_and_keeps_price_weights(tmp_path) -> None:
    env = BaseEnvironment(device="cpu", precision=32)
    config = EventJointConfig(label="bid-ask")
    old = EventJointHedges(env, config=config, means={k:torch.zeros(21) for k in ("long", "short")},
        scales={k:torch.ones(21) for k in ("long", "short")})
    path = tmp_path / "event.pt"
    torch.save(dict(model_code_version=type(config).version.unique_id48, model_config=asdict(config),
        user_version=config.user_version, best_state=old.state_dict(), selected_epoch=16), path)
    target = ObservableJointConfig(label="bid-ask")
    new, epochs = initialize_observable(path, env=env, config=target)
    assert epochs == {"long":16, "short":16}
    for label in ("long", "short"):
        a, b = getattr(old, label), getattr(new, label)
        torch.testing.assert_close(a.recurrence.weight_ih[:, :22], b.recurrence.weight_ih[:, :22], rtol=0, atol=0)
        assert torch.count_nonzero(b.recurrence.weight_ih[:, 22:65]) == 0
        torch.testing.assert_close(a.trade_head.weight[:, 54 + 5003:], b.trade_head.weight[:, 54 + 43:], rtol=0, atol=0)
        assert all(p.requires_grad for p in b.recurrence.parameters())
        assert b.gate.I0.weight.requires_grad
        assert b.event_heads["10"].affine.weight.requires_grad
    assert type(config).version.unique_id48 != type(target).version.unique_id48
