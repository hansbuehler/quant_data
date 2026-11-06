"""Outside-utility action penalties, strict warm starts and restart semantics."""

import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    FloatingTriple,
    FloatingTripleConfig,
)
from quantdata.sdlv.dhgromm1.floating_run import load_floating_checkpoint, train_floating
from quantdata.sdlv.dhgromm1.hedge import HedgeOutput
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.sdlv.dhgromm1.regularization import (
    ActionRegularizerConfig,
    action_regularizer,
    regularized_losses,
)
from quantdata.sdlv.dhgromm1.regularized_evaluation import evaluate_regularized, oce_statistics
from quantdata.sdlv.dhgromm1.regularized_run import load_zero_cost_source
from quantdata.sdlv.dhgromm1.time_decider import (
    TimeDeciderTriple,
    TimeDeciderTripleConfig,
    TimeDeciderWithA10Config,
    TimeDeciderWithA10Triple,
)
from quantdata.sdlv.dhgromm1.training import TrainConfig
from quantdata.torch import BaseEnvironment


def make_model(config_type=FloatingTripleConfig, model_type=FloatingTriple):
    template = config_type(label="test")
    config = replace(template, policy=replace(template.policy, hidden_size=3, trade_l1=0., trade_l2=0.))
    model = model_type(BaseEnvironment(device="cpu", precision=32), config=config,
        mean=torch.zeros(21), scale=torch.ones(21), return_mean=torch.zeros(20, 21), payoff_mean=.02)
    with torch.no_grad():
        for label in LABELS:
            getattr(model, label).day0.fill_(.003)
    return model, config


def payload(model, config):
    return dict(model_code_version=type(config).version.unique_id48, model_config=asdict(config),
        user_version=config.user_version, cost_treatment="inside_utility", state_dict=copy.deepcopy(model.state_dict()),
        best_states={k:copy.deepcopy(getattr(model,k).state_dict()) for k in LABELS},
        training_paths=4, torch_rng=torch.get_rng_state(), cuda_rng=[], selected_epochs=dict.fromkeys(LABELS, 5))


def test_penalty_reduction_and_gradient_are_on_executed_quantities():
    trades = torch.tensor([[[2., -3.]], [[-4., 0.]]], dtype=torch.float64, requires_grad=True)
    reg = ActionRegularizerConfig(l1=.1, l2=.02)
    value = action_regularizer(trades, reg)
    torch.testing.assert_close(value, torch.tensor([.76, .72], dtype=torch.float64))
    value.mean().backward()
    torch.testing.assert_close(trades.grad, (.1*trades.detach().sign()+.04*trades.detach())/2)


def test_penalty_is_outside_utility_not_deducted_from_wealth_or_translation():
    model, _ = make_model()
    trades = torch.ones(4, 20, 21)
    zero = torch.zeros(4)
    out = HedgeOutput(zero, zero, trades, trades)
    result = (out, out, out)
    payoff = torch.full((4,), .02)
    raw = model.losses(result, payoff)
    loss = regularized_losses(raw, result, ActionRegularizerConfig(l1=1e-4, l2=1e-8))
    torch.testing.assert_close(raw, torch.zeros(3))
    torch.testing.assert_close(loss, torch.full((3,), 420*(1e-4+1e-8)))
    loss.sum().backward()
    assert model.empty.translation.grad == 0
    # Deducting .042 from wealth would instead produce roughly .130 OCE loss.
    assert float(loss[0].detach()) < .05
    assert torch.count_nonzero(out.penalty) == 0


@pytest.mark.parametrize("coefficient", ["l1", "l2"])
@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_coefficients_rejected(coefficient, bad):
    with pytest.raises(ValueError):
        ActionRegularizerConfig(**{coefficient:bad})


@pytest.mark.parametrize("types", [(FloatingTripleConfig, FloatingTriple),
    (TimeDeciderTripleConfig, TimeDeciderTriple), (TimeDeciderWithA10Config, TimeDeciderWithA10Triple)])
def test_source_loader_retains_each_architecture_and_rejects_nonzero_costs(tmp_path, types):
    model, config = make_model(*types)
    path = tmp_path/"source.pt"
    saved = payload(model, config)
    torch.save(saved, path)
    restored, cfg, model_type = load_zero_cost_source(path)
    assert cfg == config and model_type is types[1]
    torch.testing.assert_close(restored["state_dict"]["empty.day0"], model.empty.day0)
    costly = replace(config, policy=replace(config.policy, trade_l1=1e-4))
    saved["model_config"] = asdict(costly)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="zero-cost"):
        load_zero_cost_source(path)


def test_plain_and_regularized_checkpoints_cannot_be_silently_interchanged(tmp_path):
    model, config = make_model()
    saved = payload(model, config)
    reg = ActionRegularizerConfig()
    saved.update(action_regularizer=asdict(reg), regularizer_code_version=reg.version.unique_id48)
    path = tmp_path/"regularized.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="regularizer"):
        load_floating_checkpoint(path, config=config)
    assert load_floating_checkpoint(path, config=config, regularizer=reg)["training_paths"] == 4
    with pytest.raises(ValueError, match="regularizer"):
        load_floating_checkpoint(path, config=config, regularizer=replace(reg, l1=2e-4))


def test_pair_mc_error_uses_paired_difference_not_sum_of_independent_errors():
    influence = np.array([1., -1., 2., 0.])
    a = oce_statistics(influence, influence)
    assert a["three_standard_errors"] == pytest.approx(1.5)
    same = oce_statistics(influence-influence, influence-influence)
    assert same["three_standard_errors"] == 0


def test_regularized_scan_and_eager_objectives_and_gradients_match():
    model, _ = make_model()
    other = copy.deepcopy(model)
    torch.manual_seed(19)
    prices = torch.ones(4, 21, 21)
    returns = torch.randn(4, 20, 21)*.01
    changes = torch.zeros(4, 20, 420)
    payoff = torch.tensor([.02, .03, .01, .04])
    reg = ActionRegularizerConfig()
    def objective(which, eager):
        outputs = tuple(getattr(which,k)(prices,changes,returns=returns,eager=eager) for k in LABELS)
        return regularized_losses(which.losses(outputs,payoff),outputs,reg)
    a, b = objective(model, False), objective(other, True)
    torch.testing.assert_close(a, b, atol=2e-8, rtol=1e-5)
    a.sum().backward()
    b.sum().backward()
    for (name,p), (_,q) in zip(model.named_parameters(),other.named_parameters(),strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=1e-4,msg=name)


def test_regularized_training_restart_retains_adam_rng_history_and_raw_oce(tmp_path):
    model, config = make_model()
    warm = payload(model, config)
    reg = ActionRegularizerConfig()
    rng = np.random.default_rng(18)
    sample = ObservableSample(arrays=(np.ones((4,21,21)), rng.normal(0,.001,(4,20,21)), np.array([.01,.02,.03,.04])),
        changes=np.zeros((4,3990)), device="cpu")
    monitor = SimpleNamespace(stage=lambda *args:None, record={})
    common = dict(sample=sample, config=config, monitor=monitor, stop={"requested":False},
                  initial_benchmark=None, regularizer=reg, warm_start=warm)
    full = tmp_path/"full"
    split = tmp_path/"split"
    full.mkdir()
    split.mkdir()
    train_floating(model=copy.deepcopy(model),train=TrainConfig(epochs=2,batch_size=2,learning_rate=1e-5),output=full,**common)
    train_floating(model=copy.deepcopy(model),train=TrainConfig(epochs=1,batch_size=2,learning_rate=1e-5),output=split,**common)
    train_floating(model=copy.deepcopy(model),train=TrainConfig(epochs=2,batch_size=2,learning_rate=1e-5),output=split,**common)
    a = load_floating_checkpoint(full/"floating.checkpoint.pt",config=config,regularizer=reg)
    b = load_floating_checkpoint(split/"floating.checkpoint.pt",config=config,regularizer=reg)
    assert a["history"] == b["history"]
    assert b["resume_epoch"] == 2 and b["resume_offset"] == 0
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key],b["state_dict"][key],atol=0,rtol=0)
    for label in LABELS:
        row = b["history"][-1]
        assert row[f"{label}_loss"] == pytest.approx(-row[f"{label}_raw_oce"]+row[f"{label}_regularizer"],abs=1e-8)
        assert row[f"{label}_mean_sum_abs_trades"] > 0
    torch.testing.assert_close(a["torch_rng"],b["torch_rng"],atol=0,rtol=0)
    for key in a["optimizer"]["state"]:
        for name in ("step","exp_avg","exp_avg_sq"):
            torch.testing.assert_close(a["optimizer"]["state"][key][name],b["optimizer"]["state"][key][name],atol=0,rtol=0)


def test_evaluation_reports_raw_and_penalized_values_separately_with_three_se(tmp_path, monkeypatch):
    model, _ = make_model()
    rng = np.random.default_rng(32)
    arrays = (np.ones((4,21,21)), rng.normal(0,.001,(4,20,21)), np.array([.01,.02,.03,.04]))
    sample = ObservableSample(arrays=arrays, changes=np.zeros((4,3990)), device="cpu")
    functions = tuple(lambda p,m,returns,policy=getattr(model,k):policy(p,m,returns=returns,eager=True) for k in LABELS)
    monkeypatch.setattr(model,"policy_functions",lambda **kwargs:functions)
    (tmp_path/"identity.json").write_text(json.dumps(dict(measure="Q")))
    reg = ActionRegularizerConfig()
    report, values, influence = evaluate_regularized(model=model,sample=sample,arrays=arrays,
        source_identity=dict(scenarios=str(tmp_path)),regularizer=reg,batch_size=2)
    assert values.shape == influence.shape == (4,3)
    for i,label in enumerate(LABELS):
        stats = report["statistics"][label]
        assert stats["raw_oce"]["oce"] == pytest.approx(values[:,i].mean())
        assert stats["regularized_objective"]["oce"] == pytest.approx(values[:,i].mean()-stats["mean_regularizer"])
        assert stats["raw_oce"]["three_standard_errors"] >= 0
    assert report["raw_long_minus_empty"]["oce"] == pytest.approx((values[:,1]-values[:,0]).mean())
