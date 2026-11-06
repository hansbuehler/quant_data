"""Global tail semantics, unbiased batch decomposition, gradients and restarts."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    FloatingConfig,
    FloatingTriple,
    FloatingTripleConfig,
)
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.sdlv.dhgromm1.robust import (
    RobustTriple,
    RobustTripleConfig,
    exact_population_scores,
    exact_tail_size,
    lower_tail_value,
    migrate_selected_weights,
    profile_tail,
)
from quantdata.sdlv.dhgromm1.robust_training import load_robust_checkpoint, train_robust
from quantdata.sdlv.dhgromm1.training import TrainConfig, _economic_utility
from quantdata.torch import BaseEnvironment


def make_model():
    config = RobustTripleConfig(label="test",policy=FloatingConfig(label="policy",hidden_size=3,trade_l1=0.,trade_l2=0.))
    model = RobustTriple(BaseEnvironment(device="cpu",precision=32),config=config,
        mean=torch.zeros(21),scale=torch.ones(21),return_mean=torch.zeros(20,21),payoff_mean=.02)
    with torch.no_grad():
        for label in LABELS:
            getattr(model,label).day0.fill_(.003)
            getattr(model,label).tail_threshold.fill_(.001)
    return model


def test_shared_threshold_decomposes_over_batches_but_batch_trimming_does_not():
    values = torch.tensor([-10.]*10+[10.]*10,dtype=torch.float64)
    threshold = torch.tensor(10.,dtype=torch.float64)
    exact = values.sort().values[:18].mean()
    global_value = lower_tail_value(values,threshold,.9)
    minibatch_estimator = torch.stack([lower_tail_value(batch,threshold,.9) for batch in values.split(10)]).mean()
    wrong_batch_tail = torch.stack([batch.sort().values[:9].mean() for batch in values.split(10)]).mean()
    torch.testing.assert_close(global_value,exact)
    torch.testing.assert_close(minibatch_estimator,exact)
    assert wrong_batch_tail != pytest.approx(float(exact))


def test_variational_value_at_global_cutoff_is_exact_lower_tail():
    values = torch.tensor([-.08,-.03,-.01,.005,.01,.03,.05,.07,.09,.5],dtype=torch.float64,requires_grad=True)
    threshold = torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    value = lower_tail_value(values,threshold,.9)
    torch.testing.assert_close(value,values.sort().values[:9].mean())
    value.backward()
    torch.testing.assert_close(values.grad,torch.tensor([1/9]*9+[0.],dtype=torch.float64))
    assert threshold.grad == pytest.approx(0.,abs=1e-14)


def test_zero_hedge_has_zero_value_and_stationary_cash_and_threshold():
    model = make_model()
    with torch.no_grad():
        model.empty.translation.zero_()
        model.empty.tail_threshold.zero_()
    zero = torch.zeros(10)
    out = HedgeOutput(zero,zero,zero,zero)
    losses = model.losses((out,out,out),torch.full((10,),.02))
    assert losses[0] == 0
    losses[0].backward()
    assert model.empty.translation.grad == pytest.approx(0.,abs=2e-7)
    assert model.empty.tail_threshold.grad == pytest.approx(0.,abs=2e-7)
    assert profile_tail(np.zeros(100),fraction=.9)["oce"] == 0


def test_cash_profile_uses_the_global_lower_tail_and_utility_monotonicity():
    net = np.linspace(-.1,.2,100)
    tail = profile_tail(net,fraction=.9)
    full = profile_tail(net,fraction=1.)
    assert tail["retained_paths"] == 90
    assert tail["oce"] < full["oce"]
    assert tail["oce"] == pytest.approx(_economic_utility(net[:90],tail["cash"],100).mean())
    assert tail["derivative_weight"].mean() == pytest.approx(1.,abs=1e-9)
    # The winsorized pseudo-observations include the quantile influence.
    assert np.max(tail["variates"]) == pytest.approx(tail["cash"]+tail["threshold"])
    assert not np.allclose(tail["variates"],np.where(net<=net[89],_economic_utility(net,tail["cash"],100)/.9,0))


def test_bad_fraction_cost_or_population_is_rejected():
    for fraction in (0.,1.1,float("nan")):
        with pytest.raises(ValueError):
            RobustTripleConfig(label="bad",tail_fraction=fraction)
    with pytest.raises(ValueError,match="zero costs"):
        RobustTripleConfig(label="bad",policy=FloatingConfig(label="p"))
    with pytest.raises(ValueError,match="integer"):
        exact_tail_size(4,.9)


def test_full_population_scores_match_global_sort_not_batch_tail():
    model = make_model()
    gains = torch.stack([torch.linspace(-.02,.02,10)]*3)
    with torch.no_grad():
        scores = exact_population_scores(model,gains,torch.full((10,),.02))
        expected = _economic_utility(gains[0].numpy(),float(model.empty.translation),100)
    assert scores["empty"]["robust_oce"] == pytest.approx(np.sort(expected)[:9].mean(),abs=1e-8)
    assert scores["empty"]["threshold_gap"] >= -1e-8


def test_source_migration_retains_weights_and_only_adds_thresholds():
    model = make_model()
    original = FloatingTriple(BaseEnvironment(device="cpu",precision=32),
        config=FloatingTripleConfig(label="source",policy=model.config.policy),mean=torch.zeros(21),scale=torch.ones(21),
        return_mean=torch.zeros(20,21),payoff_mean=.02)
    saved = dict(state_dict=original.state_dict(),best_states={k:getattr(original,k).state_dict() for k in LABELS})
    migrate_selected_weights(model,saved)
    for key,value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
    assert set(model.state_dict())-set(original.state_dict()) == {f"{k}.tail_threshold" for k in LABELS}


def test_robust_scan_and_eager_values_and_all_gradients_agree():
    model = make_model()
    other = copy.deepcopy(model)
    torch.manual_seed(7)
    prices,marks = torch.ones(10,21,21),torch.zeros(10,20,420)
    returns,payoff = torch.randn(10,20,21)*.005,torch.linspace(.01,.03,10)
    def objective(which,eager):
        outputs = tuple(getattr(which,k)(prices,marks,returns=returns,eager=eager) for k in LABELS)
        return which.losses(outputs,payoff)
    x,y = objective(model,False),objective(other,True)
    torch.testing.assert_close(x,y,atol=1e-8,rtol=1e-5)
    x.sum().backward()
    y.sum().backward()
    for (name,a),(_,b) in zip(model.named_parameters(),other.named_parameters(),strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=2e-7,rtol=1e-4,msg=name)


def test_exact_restart_and_objective_version_rejection(tmp_path):
    model = make_model()
    rng = np.random.default_rng(25)
    sample = ObservableSample(arrays=(np.ones((10,21,21)),rng.normal(0,.001,(10,20,21)),np.linspace(.01,.03,10)),
        changes=np.zeros((10,3990)),device="cpu")
    monitor = SimpleNamespace(stage=lambda *args:None,record={})
    common = dict(sample=sample,monitor=monitor,stop={"requested":False})
    full,split = tmp_path/"full",tmp_path/"split"
    full.mkdir()
    split.mkdir()
    torch.manual_seed(90)
    train_robust(model=copy.deepcopy(model),train=TrainConfig(epochs=2,batch_size=10,learning_rate=1e-5),output=full,**common)
    torch.manual_seed(90)
    train_robust(model=copy.deepcopy(model),train=TrainConfig(epochs=1,batch_size=10,learning_rate=1e-5),output=split,**common)
    train_robust(model=copy.deepcopy(model),train=TrainConfig(epochs=2,batch_size=10,learning_rate=1e-5),output=split,**common)
    a = load_robust_checkpoint(full/"robust.checkpoint.pt",config=model.config)
    b = load_robust_checkpoint(split/"robust.checkpoint.pt",config=model.config)
    assert a["history"] == b["history"]
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key],b["state_dict"][key],atol=0,rtol=0)
    for key in a["optimizer"]["state"]:
        for name in ("step","exp_avg","exp_avg_sq"):
            torch.testing.assert_close(a["optimizer"]["state"][key][name],b["optimizer"]["state"][key][name],atol=0,rtol=0)
    torch.testing.assert_close(a["torch_rng"],b["torch_rng"],atol=0,rtol=0)
    with pytest.raises(ValueError,match="version"):
        load_robust_checkpoint(full/"robust.checkpoint.pt",config=replace(model.config,tail_fraction=.8))
