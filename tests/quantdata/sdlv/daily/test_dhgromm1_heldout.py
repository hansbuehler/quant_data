"""Validation tails, independent-population errors, and frozen-training means."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.heldout_evaluation import (
    evaluate_heldout,
    fixed_cash_tail,
    independent_mc,
)
from quantdata.sdlv.dhgromm1.heldout_run import robust_source_ready
from quantdata.sdlv.dhgromm1.robust import profile_tail


def test_user_stopped_source_requires_approved_immutable_checkpoint(tmp_path):
    checkpoint = b"frozen checkpoint bytes"
    (tmp_path/"robust.checkpoint.pt").write_bytes(checkpoint)
    (tmp_path/"status.json").write_text(json.dumps(dict(status="stopped")))
    with pytest.raises(RuntimeError,match="explicitly approve"):
        robust_source_ready(tmp_path)
    approved = hashlib.sha256(checkpoint).hexdigest()
    assert robust_source_ready(tmp_path,stopped_sha256=approved)
    (tmp_path/"robust.checkpoint.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError,match="differs"):
        robust_source_ready(tmp_path,stopped_sha256=approved)


def test_robust_source_gate_waits_for_running_and_rejects_failed_source(tmp_path):
    (tmp_path/"status.json").write_text(json.dumps(dict(status="running")))
    assert not robust_source_ready(tmp_path)
    (tmp_path/"status.json").write_text(json.dumps(dict(status="failed")))
    with pytest.raises(RuntimeError,match="failed"):
        robust_source_ready(tmp_path,stopped_sha256="not-an-approval-for-failure")
    (tmp_path/"status.json").write_text(json.dumps(dict(status="complete")))
    assert robust_source_ready(tmp_path)


def test_fixed_cash_tail_matches_profile_at_the_optimum_and_handles_ties():
    net = np.linspace(-.03,.04,20)
    for fraction in (1.,.9):
        p = profile_tail(net,fraction=fraction)
        fixed = fixed_cash_tail(net,cash=p["cash"],fraction=fraction)
        np.testing.assert_allclose(fixed["variates"],p["variates"],atol=1e-12)
        np.testing.assert_allclose(fixed["derivative_weight"],p["derivative_weight"],atol=1e-12)
    tied = fixed_cash_tail(np.zeros(20),cash=0.,fraction=.9)
    np.testing.assert_array_equal(tied["variates"],0)
    np.testing.assert_allclose(tied["derivative_weight"],1)


def test_independent_sampling_variances_add_without_false_cross_split_pairing():
    val,train = np.array([0.,2.,3.,5.]),np.array([2.,4.,6.,8.])
    actual = independent_mc(val,val,train)
    expected = 3*np.sqrt(np.var([1.,4.],ddof=1)/2+np.var([3.,7.],ddof=1)/2)
    assert actual["three_standard_errors"]==pytest.approx(expected)
    assert actual["three_standard_errors"]>actual["three_validation_only_errors"]
    with pytest.raises(ValueError):
        independent_mc(val[:3],val[:3],train)


def test_validation_gains_keep_training_means_even_if_validation_drift_differs():
    paths = 20
    ret = torch.full((paths,20,21),.003)
    training_mean = torch.full((20,21),.001)
    class Policy:
        translation = torch.tensor(0.)
        return_mean = training_mean.clone()
        def __call__(self,price,changes,*,returns,eager):
            trade = torch.full_like(returns,.1)
            return SimpleNamespace(trades=trade,gain=(trade*(returns-self.return_mean)).sum((1,2)),penalty=torch.zeros(len(price)))
    model = SimpleNamespace(empty=Policy(),long=Policy(),short=Policy(),payoff_mean=torch.tensor(.02))
    h = torch.full((paths,),.02)
    class Sample:
        size = paths
        def batch(self,selection):
            return torch.ones((paths,21,21))[selection],ret[selection],h[selection],torch.zeros((paths,20,420))[selection]
    train_r = np.full((paths,20,21),.001)
    arrays = (np.ones((paths,21,21)),ret.numpy().astype(float),h.numpy().astype(float))
    training = (arrays[0],train_r,np.full(paths,float(model.payoff_mean)))
    before = model.empty.return_mean.clone()
    result,_ = evaluate_heldout(model=model,sample=Sample(),arrays=arrays,training_arrays=training,batch_size=10)
    expected = 420*.1*(.003-.001)
    assert result["profiled_cash"]["ordinary"]["statistics"]["empty"]["oce"]==pytest.approx(expected,abs=1e-7)
    assert result["profiled_cash"]["ordinary"]["long_minus_empty"]["oce"]==pytest.approx(0,abs=1e-9)
    assert result["profiled_cash"]["ordinary"]["short_minus_empty"]["oce"]==pytest.approx(0,abs=1e-9)
    torch.testing.assert_close(before,model.empty.return_mean)
