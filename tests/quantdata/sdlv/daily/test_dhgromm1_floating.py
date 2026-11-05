"""Floating quantities, exact scalar-hedge inclusion, centering and causality."""

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.contracts import RETURN_CONVENTION, SCENARIO_SCHEMA, canonical_gains
from quantdata.sdlv.dhgromm1.floating import (
    FloatingConfig,
    FloatingHedge,
    FloatingTriple,
    FloatingTripleConfig,
    fit_return_means,
    floating_gains,
)
from quantdata.sdlv.dhgromm1.floating_run import (
    load_cost_continuation,
    load_floating_checkpoint,
    load_floating_model,
)
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.torch import BaseEnvironment


def data():
    rng = np.random.default_rng(81)
    prices = np.full((4, 21, 21), .02)
    spot = np.exp(.02 * rng.standard_normal((4, 21)))
    prices *= spot[:, :, None]
    prices[:, :, 0] = spot
    prices[:, :, 17:] = spot[:, :, None]
    ret = .01 * rng.standard_normal((4, 20, 21))
    ret[:, 10, 1] = np.maximum(spot[:, 20] - spot[:, 10], 0) / prices[:, 10, 1] - 1
    ret[:, 10, 17] = spot[:, 20] / spot[:, 10] - 1
    h = np.abs(spot[:, 20] / spot[:, 10] - 1)
    gains = floating_gains(prices, ret)
    changes = .01 * rng.standard_normal((4, 20, 420))
    for day in range(20):
        changes[:, day, day * 21:] = 0
    return prices, ret, gains, changes, h


def policy(*, active=False):
    torch.manual_seed(73)
    model = FloatingHedge(BaseEnvironment(device="cpu", precision=64),
        config=FloatingConfig(label="long", hidden_size=3),
        price_mean=torch.zeros(21), price_scale=torch.ones(21), return_mean=torch.zeros(20, 21))
    if active:
        with torch.no_grad():
            model.trade_head.weight.add_(torch.randn_like(model.trade_head.weight) * 1e-4)
            model.day0.fill_(.003)
            model.day10.fill_(.007)
    return model


def test_floating_return_is_payoff_minus_price_not_premium_percentage():
    prices, premium, gains, _, _ = data()
    s = prices[:, 10, 0]
    payoff = np.maximum(prices[:, 20, 0] / s - 1, 0)
    np.testing.assert_allclose(gains[:, 10, 1], payoff - prices[:, 10, 1] / s, atol=1e-15)
    np.testing.assert_allclose(gains[:, :, 0], premium[:, :, 0])
    assert not np.allclose(gains[:, 10, 1], premium[:, 10, 1])


def test_training_means_are_by_opening_date_and_control_with_identical_spot_forward_treatment():
    _, _, gains, _, _ = data()
    mean = fit_return_means(gains)
    np.testing.assert_allclose((gains-mean).mean(0), 0, atol=1e-16)
    duplicate = gains.copy()
    duplicate[:, :, 17] = duplicate[:, :, 0]
    duplicate_mean = fit_return_means(duplicate)
    np.testing.assert_array_equal(duplicate_mean[:, 17], duplicate_mean[:, 0])
    evaluation = gains * 7 + .03
    frozen = mean.copy()
    _ = evaluation - mean
    np.testing.assert_array_equal(mean, frozen)
    assert np.max(np.abs((evaluation-mean).mean(0)[:, 1:])) > .01


@pytest.mark.parametrize("sign", [1, -1])
def test_constant_day10_vector_contains_exact_centered_scalar_hedge(sign):
    prices, _, gains, changes, h = data()
    model = policy()
    model.return_mean.copy_(torch.tensor(fit_return_means(gains)))
    a = .9987
    with torch.no_grad():
        model.day10[1] = -2 * sign * a
        model.day10[17] = sign * a
    out = model(torch.tensor(prices), torch.tensor(changes), returns=torch.tensor(gains), eager=True)
    p = 1 + (2*prices[:, 10, 1]-prices[:, 10, 17])/prices[:, 10, 0]
    d = p-h
    np.testing.assert_allclose(out.gain.detach(), sign*a*(d-d.mean()), atol=3e-16)
    assert torch.count_nonzero(out.trades[:, :10]) == 0
    assert torch.count_nonzero(out.trades[:, 11:]) == 0
    assert model.day10.shape == (21,) and model.day0.shape == (21,)
    assert not any("event_heads" in name for name, _ in model.named_parameters())


def test_book_converts_quantity_to_actual_units_using_spot_not_option_premium():
    model = policy()
    with torch.no_grad():
        model.day10[1] = 2.
    batch = 2
    hidden = torch.zeros(batch, 3, dtype=torch.float64)
    book = torch.zeros(batch, 20, 21, dtype=torch.float64)
    previous = torch.zeros(batch, 21, dtype=torch.float64)
    features = torch.zeros(batch, 22, dtype=torch.float64)
    features[:, 0] = .5
    inputs = (features, torch.tensor([[1.], [2.]], dtype=torch.float64),
              torch.zeros(batch, 42, dtype=torch.float64), torch.zeros(batch, 420, dtype=torch.float64), model.insert[10])
    carry, trade = model._body((hidden, hidden.clone(), book, previous), inputs)
    torch.testing.assert_close(carry[2][:, 10, 1], torch.tensor([2., 1.], dtype=torch.float64))
    torch.testing.assert_close(trade[:, 1], torch.full((2,), 2., dtype=torch.float64))


@pytest.mark.parametrize("cutoff", [0, 9, 10, 18])
def test_future_quotes_marks_and_all_return_labels_cannot_change_past_actions(cutoff):
    prices, _, gains, changes, _ = data()
    model = policy(active=True)
    p, r, m = map(torch.tensor, (prices, gains, changes))
    future_p, future_m = p.clone(), m.clone()
    future_p[:, cutoff+1:] *= 3
    future_m[:, cutoff+1:] += 1
    a = model(p, m, returns=r, eager=True)
    b = model(future_p, future_m, returns=r+10, eager=True)
    torch.testing.assert_close(a.trades[:, :cutoff+1], b.trades[:, :cutoff+1], rtol=0, atol=0)


def test_scan_and_eager_values_and_all_gradients_agree():
    prices, _, gains, changes, _ = data()
    a = policy(active=True)
    b = copy.deepcopy(a)
    p, r, m = map(torch.tensor, (prices, gains, changes))
    x, y = a(p, m, returns=r, eager=True), b(p, m, returns=r)
    torch.testing.assert_close(x.trades, y.trades, atol=1e-12, rtol=1e-10)
    (x.gain.square()+x.penalty).mean().backward()
    (y.gain.square()+y.penalty).mean().backward()
    for (name, ap), (_, bp) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        if ap.grad is not None:
            torch.testing.assert_close(ap.grad, bp.grad, atol=1e-11, rtol=1e-8, msg=name)


def test_costs_are_inside_utility_and_policies_do_not_share_parameters():
    model = FloatingTriple(BaseEnvironment(device="cpu", precision=64),
        config=FloatingTripleConfig(label="test", policy=FloatingConfig(label="policy", hidden_size=3)),
        mean=torch.zeros(21), scale=torch.ones(21), return_mean=torch.zeros(20, 21), payoff_mean=.02)
    ids = [{id(p) for p in getattr(model, label).parameters()} for label in ("empty", "long", "short")]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    cost = torch.full((4,), .001, dtype=torch.float64)
    zero = torch.zeros_like(cost)
    out = HedgeOutput(zero, cost, zero, zero)
    loss = model.losses((out, out, out), torch.full((4,), .02, dtype=torch.float64))
    torch.testing.assert_close(loss, torch.full((3,), .00105, dtype=torch.float64))


def test_bad_gain_geometry_or_spot_is_rejected():
    prices, premium, _, _, _ = data()
    prices[:, 10, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        floating_gains(prices, premium)


def test_identified_legacy_conversion_is_once_only_and_both_loaders_agree(tmp_path):
    from quantdata.sdlv.dhgromm1.floating import read_floating_sample
    from quantdata.sdlv.dhgromm1.training import read_sample

    prices,premium,gains,_,h = data()
    sample = tmp_path/"train"
    sample.mkdir()
    np.savez(sample/"000000.npz",prices=prices,returns=premium,payoff=h)
    identity = {"schema":"dhgromm1-expiry-options-v3"}
    (tmp_path/"identity.json").write_text(json.dumps(identity))
    np.testing.assert_array_equal(read_floating_sample(sample,paths=4)[1],gains)
    np.testing.assert_allclose(read_sample(sample,paths=4)[1],gains,atol=1e-8,rtol=1e-7)
    current = dict(schema=SCENARIO_SCHEMA,return_convention=RETURN_CONVENTION)
    np.testing.assert_array_equal(canonical_gains(prices,gains,identity=current),gains)
    with pytest.raises(ValueError,match="ambiguous"):
        canonical_gains(prices,premium,identity={})
    with pytest.raises(ValueError,match="ambiguous"):
        canonical_gains(prices,premium,identity={**identity,"return_convention":RETURN_CONVENTION})


def test_read_only_checkpoint_load_and_semantic_version_rejection(tmp_path):
    env = BaseEnvironment(device="cpu",precision=64)
    config = FloatingTripleConfig(label="load",policy=FloatingConfig(label="policy",hidden_size=3))
    model = FloatingTriple(env,config=config,mean=torch.zeros(21),scale=torch.ones(21),
                          return_mean=torch.zeros(20,21),payoff_mean=.02)
    path = tmp_path/"floating.pt"
    payload = dict(model_code_version=type(config).version.unique_id48,model_config=asdict(config),
        user_version=config.user_version,cost_treatment="inside_utility",state_dict=model.state_dict(),
        best_states={k:getattr(model,k).state_dict() for k in ("empty","long","short")})
    torch.save(payload,path)
    loaded = load_floating_model(path,config=config,env=env)
    assert all(not p.requires_grad for p in loaded.parameters())
    payload["model_code_version"] = "old-premium-return-version"
    torch.save(payload,path)
    with pytest.raises(ValueError,match="version"):
        load_floating_checkpoint(path,config=config)


def test_explicit_cost_change_preserves_weights_optimizer_and_rejects_other_changes(tmp_path):
    env = BaseEnvironment(device="cpu",precision=64)
    old = FloatingTripleConfig(label="change",policy=FloatingConfig(label="policy",hidden_size=3))
    model = FloatingTriple(env,config=old,mean=torch.zeros(21),scale=torch.ones(21),
                          return_mean=torch.zeros(20,21),payoff_mean=.02)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    model.long.day10.square().sum().backward()
    optimizer.step()
    path = tmp_path/"old.pt"
    payload = dict(model_code_version=type(old).version.unique_id48,model_config=asdict(old),
        user_version=old.user_version,cost_treatment="inside_utility",state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),resume_epoch=138,resume_offset=94720,torch_rng=torch.get_rng_state())
    torch.save(payload,path)
    zero = replace(old,policy=replace(old.policy,trade_l1=0.,trade_l2=0.))
    with pytest.raises(ValueError,match="version"):
        load_floating_checkpoint(path,config=zero)
    loaded = load_cost_continuation(path,config=zero)
    assert loaded["resume_epoch"] == 138 and loaded["resume_offset"] == 94720
    assert loaded["model_config"]["policy"]["trade_l1"] == 1e-4
    torch.testing.assert_close(loaded["torch_rng"],payload["torch_rng"],atol=0,rtol=0)
    assert float(next(iter(loaded["optimizer"]["state"].values()))["step"]) == 1
    for key,value in model.state_dict().items():
        torch.testing.assert_close(loaded["state_dict"][key],value,atol=0,rtol=0)
    for wrong in (replace(zero,policy=replace(zero.policy,hidden_size=4)),
                  replace(zero,policy=replace(zero.policy,risk_aversion=90.))):
        with pytest.raises(ValueError,match="version"):
            load_cost_continuation(path,config=wrong)


def test_zero_cost_config_returns_exactly_zero_penalty_for_nonzero_trades():
    prices,_,gains,changes,_ = data()
    model = policy(active=True)
    model.config = replace(model.config,trade_l1=0.,trade_l2=0.)
    out = model(torch.tensor(prices),torch.tensor(changes),returns=torch.tensor(gains),eager=True)
    assert torch.count_nonzero(out.trades) > 0
    assert torch.count_nonzero(out.penalty) == 0


def test_compiled_policies_require_eager_equivalent_division(monkeypatch):
    model = FloatingTriple(BaseEnvironment(device="cpu",precision=64),
        config=FloatingTripleConfig(label="compile",policy=FloatingConfig(label="policy",hidden_size=3)),
        mean=torch.zeros(21),scale=torch.ones(21),return_mean=torch.zeros(20,21),payoff_mean=.02)
    calls = []
    def compile_spy(policy,**options):
        calls.append(options)
        return policy
    monkeypatch.setattr(torch,"compile",compile_spy)
    assert len(model.policy_functions(compile_model=True)) == 3
    assert all(c == {"fullgraph":True,"options":{"eager_numerics.division_rounding":True}} for c in calls)


def test_p_scalar_benchmark_keeps_q_means_instead_of_recentering_p():
    from quantdata.sdlv.dhgromm1.floating_run import scalar_benchmarks

    prices,_,gains,_,h = data()
    q_means = gains.mean(0)+.0002
    q_payoff_mean = float(h.mean()+.001)
    config = FloatingTripleConfig(label="P",policy=FloatingConfig(label="policy",trade_l1=0,trade_l2=0))
    report = scalar_benchmarks(prices=prices,returns=gains,payoff=h,means=q_means,config=config,payoff_mean=q_payoff_mean)
    assert report["replication_max_error"]<1e-12
    for key,sign in (("long",1),("short",-1)):
        selected = report["policies"][key]["net"]
        d = (gains-q_means)[:,10,17]-2*(gains-q_means)[:,10,1]
        net = sign*(h-q_payoff_mean+selected["alpha"]*d)
        from quantdata.sdlv.dhgromm1.training import _economic_utility
        assert _economic_utility(net,selected["cash"],100).mean()==pytest.approx(selected["oce"],abs=1e-12)
