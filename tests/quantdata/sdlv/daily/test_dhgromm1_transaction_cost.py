"""Pathwise cash costs, ordinary utility, causal fee feedback and scan gradients."""

from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.floating import FloatingConfig, FloatingHedge
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.sdlv.dhgromm1.transaction_cost import (
    TransactionCostHedge,
    TransactionCostPolicyConfig,
    TransactionCostTriple,
    TransactionCostTripleConfig,
    proportional_fee,
)
from quantdata.sdlv.dhgromm1.transaction_cost_run import load_transaction_cost_model
from quantdata.torch import BaseEnvironment


def policy(*, basis: str = "underlying_notional", rate: float = 1e-4) -> TransactionCostHedge:
    torch.manual_seed(79)
    return TransactionCostHedge(BaseEnvironment(device="cpu", precision=64),
        config=TransactionCostPolicyConfig(label="test", hidden_size=3,
            transaction_cost_basis=basis, transaction_cost_rate=rate),
        price_mean=torch.zeros(21), price_scale=torch.ones(21), return_mean=torch.zeros(20, 21))


def data():
    rng = np.random.default_rng(731)
    price = np.full((4, 21, 21), .02)
    spot = np.exp(.01*rng.standard_normal((4, 21)))
    price *= spot[:, :, None]
    price[:, :, 0], price[:, :, 17:] = spot, spot[:, :, None]
    marks = .01*rng.standard_normal((4, 20, 420))
    for day in range(20):
        marks[:, day, day*21:] = 0
    returns = .01*rng.standard_normal((4, 20, 21))
    return tuple(torch.tensor(v) for v in (price, marks, returns))


@pytest.mark.parametrize("basis,expected",[("underlying_notional",3e-4),("cash_value",1.04e-4)])
def test_one_day10_straddle_trade_pays_fee_once_on_trade_not_on_holdings(basis,expected):
    model = policy(basis=basis)
    with torch.no_grad():
        model.day10[1], model.day10[17] = -2., 1.
    p,m,r = data()
    out = model(p,m,returns=r,eager=True)
    torch.testing.assert_close(out.penalty,torch.full((4,),expected,dtype=torch.float64))
    assert torch.count_nonzero(out.trades[:,:10])==0
    assert torch.count_nonzero(out.trades[:,11:])==0


def test_cash_fee_gradients_use_signed_trade_and_current_price_basis():
    a = torch.tensor([[2.,-3.]],dtype=torch.float64,requires_grad=True)
    fee = proportional_fee(a,torch.tensor([[.02,1.]],dtype=torch.float64),rate=1e-4)
    fee.sum().backward()
    torch.testing.assert_close(a.grad,torch.tensor([[2e-6,-1e-4]],dtype=torch.float64))
    assert float(fee.detach())==pytest.approx(3.04e-4)


def test_cost_is_inside_utility_not_a_mean_regularizer():
    config = TransactionCostTripleConfig(label="objective",policy=TransactionCostPolicyConfig(label="p",hidden_size=3))
    model = TransactionCostTriple(BaseEnvironment(device="cpu",precision=64),config=config,
        mean=torch.zeros(21),scale=torch.ones(21),return_mean=torch.zeros(20,21),payoff_mean=.02)
    cost = torch.tensor([0.,.002],dtype=torch.float64)
    z = torch.zeros_like(cost)
    result = HedgeOutput(z,cost,z,z)
    loss = model.losses((result,result,result),torch.full_like(cost,.02))
    torch.testing.assert_close(loss,torch.full((3,),.0011,dtype=torch.float64))
    assert not torch.allclose(loss,torch.full_like(loss,float(cost.mean())))


def test_zero_fee_migration_preserves_original_values_and_weights_exactly():
    old = FloatingHedge(BaseEnvironment(device="cpu",precision=64),
        config=FloatingConfig(label="p",hidden_size=3,trade_l1=0.,trade_l2=0.),
        price_mean=torch.zeros(21),price_scale=torch.ones(21),return_mean=torch.zeros(20,21))
    with torch.no_grad():
        old.day0.fill_(.02)
        old.trade_head.weight.add_(torch.randn_like(old.trade_head.weight)*1e-4)
    new = policy(rate=0.)
    new.load_state_dict(old.state_dict(),strict=True)
    p,m,r = data()
    a,b = old(p,m,returns=r,eager=True),new(p,m,returns=r,eager=True)
    torch.testing.assert_close(a.trades,b.trades,rtol=0,atol=0)
    torch.testing.assert_close(a.gain,b.gain,rtol=0,atol=0)


def test_paid_cost_reduces_causal_running_pnl_before_next_action():
    model = policy()
    batch = 2
    hidden = torch.zeros(batch,3,dtype=torch.float64)
    book = torch.zeros(batch,20,21,dtype=torch.float64)
    previous = torch.zeros(batch,21,dtype=torch.float64)
    fees = torch.full((batch,1),.003,dtype=torch.float64)
    inputs = (torch.zeros(batch,22,dtype=torch.float64),torch.ones(batch,1,dtype=torch.float64),
        torch.zeros(batch,42,dtype=torch.float64),torch.zeros(batch,420,dtype=torch.float64),
        model.insert[1],torch.ones(batch,21,dtype=torch.float64))
    seen = []
    hook = model.trade_head.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    model._body((hidden,hidden.clone(),book,previous,fees),inputs)
    hook.remove()
    torch.testing.assert_close(seen[0][:,22+3+42],torch.full((batch,),-.003,dtype=torch.float64))


def test_cost_scan_and_eager_match_in_values_and_all_gradients():
    a = policy(basis="cash_value")
    with torch.no_grad():
        a.day0.fill_(.004)
        a.day10.fill_(.007)
        a.trade_head.weight.add_(torch.randn_like(a.trade_head.weight)*1e-4)
    b = copy.deepcopy(a)
    p,m,r = data()
    x,y = a(p,m,returns=r,eager=True),b(p,m,returns=r)
    torch.testing.assert_close(x.trades,y.trades,atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(x.penalty,y.penalty,atol=1e-12,rtol=1e-10)
    (x.gain.square()+x.penalty).mean().backward()
    (y.gain.square()+y.penalty).mean().backward()
    for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters(),strict=True):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad,pb.grad,atol=1e-11,rtol=1e-8,msg=name)


def test_cost_config_rejects_outside_regularizers_and_negative_costs():
    with pytest.raises(ValueError,match="L1/L2"):
        TransactionCostPolicyConfig(label="bad",trade_l2=1e-8)
    with pytest.raises(ValueError,match="nonnegative"):
        TransactionCostPolicyConfig(label="bad",transaction_cost_rate=-.01)


def test_cost_checkpoint_read_only_loader_preserves_semantics_and_rejects_version(tmp_path):
    config = TransactionCostTripleConfig(label="roundtrip",policy=TransactionCostPolicyConfig(label="p",hidden_size=3))
    model = TransactionCostTriple(BaseEnvironment(device="cpu",precision=64),config=config,
        mean=torch.zeros(21),scale=torch.ones(21),return_mean=torch.zeros(20,21),payoff_mean=.02)
    with torch.no_grad():
        model.long.day10[1] = 2.
    saved = dict(model_code_version=type(config).version.unique_id48,model_config=asdict(config),
        user_version=config.user_version,cost_treatment="inside_utility",state_dict=model.state_dict(),
        best_states={k:getattr(model,k).state_dict() for k in ("empty","long","short")})
    path = tmp_path/"cost.pt"
    torch.save(saved,path)
    loaded = load_transaction_cost_model(path,precision=64)
    assert loaded.config.policy.transaction_cost_basis=="cash_value"
    p,m,r = data()
    x,y = model.long(p,m,returns=r,eager=True),loaded.long(p,m,returns=r,eager=True)
    torch.testing.assert_close(x.penalty,y.penalty,rtol=0,atol=0)
    torch.testing.assert_close(x.trades,y.trades,rtol=0,atol=0)
    saved["model_code_version"] = "obsolete"
    torch.save(saved,path)
    with pytest.raises(ValueError,match="code version differs"):
        load_transaction_cost_model(path)
