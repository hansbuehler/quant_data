"""The immutable mark loader rejects changed or incomplete scenario identities."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.sdlv.daily.wide_audit import sha256
from quantdata.sdlv.dhgromm1.cache_load import load_retained_market_cache


def example(tmp_path):
    scenarios,cache = tmp_path/"scenarios",tmp_path/"cache"
    (scenarios/"train").mkdir(parents=True)
    cache.mkdir()
    (scenarios/"identity.json").write_text(json.dumps({"measure":"Q"}))
    np.savez(scenarios/"train"/"000000.npz",payoff=np.arange(4))
    identity = dict(schema="dhgromm1-causal-market-marks-v1",paths=4,sample="train",values_per_path=3990,
        scenario_identity_sha256=sha256(scenarios/"identity.json"))
    record = dict(start=0,count=4,scenario_chunk_sha256=sha256(scenarios/"train"/"000000.npz"))
    values = np.arange(4*3990,dtype=np.float32).reshape(4,3990)
    np.savez(cache/"000000.npz",changes=values,record=json.dumps(record))
    (cache/"identity.json").write_text(json.dumps(identity))
    (cache/"complete.json").write_text(json.dumps(dict(identity,status="complete",chunks=[record])))
    monitor = SimpleNamespace(stage=lambda *args:None)
    return scenarios,cache,monitor,values


def test_cache_is_read_only_and_reuses_complete_original_population(tmp_path):
    scenarios,cache,monitor,values = example(tmp_path)
    before = {p.name:sha256(p) for p in cache.iterdir()}
    actual,report = load_retained_market_cache(cache=cache,scenarios=scenarios,paths=4,monitor=monitor)
    np.testing.assert_array_equal(actual,values)
    assert report["reused_without_simulation"]
    assert before=={p.name:sha256(p) for p in cache.iterdir()}


def test_changed_source_chunk_or_identity_is_rejected(tmp_path):
    scenarios,cache,monitor,_ = example(tmp_path)
    np.savez(scenarios/"train"/"000000.npz",payoff=np.arange(4)+1)
    with pytest.raises(ValueError,match="original scenario changed"):
        load_retained_market_cache(cache=cache,scenarios=scenarios,paths=4,monitor=monitor)
    (scenarios/"identity.json").write_text(json.dumps({"measure":"P"}))
    with pytest.raises(ValueError,match="does not match"):
        load_retained_market_cache(cache=cache,scenarios=scenarios,paths=4,monitor=monitor)


def test_evaluation_cache_cannot_be_silently_used_as_training(tmp_path):
    scenarios,cache,monitor,values = example(tmp_path)
    (scenarios/"train").rename(scenarios/"evaluation")
    identity = json.loads((cache/"identity.json").read_text())
    complete = json.loads((cache/"complete.json").read_text())
    identity["sample"] = complete["sample"] = "evaluation"
    (cache/"identity.json").write_text(json.dumps(identity))
    (cache/"complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError,match="does not match"):
        load_retained_market_cache(cache=cache,scenarios=scenarios,paths=4,monitor=monitor)
    actual,_ = load_retained_market_cache(cache=cache,scenarios=scenarios,paths=4,monitor=monitor,sample="evaluation")
    np.testing.assert_array_equal(actual,values)
    with pytest.raises(ValueError,match="sample must be"):
        load_retained_market_cache(cache=cache,scenarios=scenarios,paths=4,monitor=monitor,sample="../train")


@pytest.mark.parametrize(("sample","seed_index"),[("train",0),("evaluation",1)])
def test_cache_replay_uses_original_split_seed_and_checks_saved_paths(tmp_path,monkeypatch,sample,seed_index):
    import jax.numpy as jnp

    from quantdata.sdlv.daily.front_spot import jax_model
    from quantdata.sdlv.dhgromm1 import rolling_simulation
    from quantdata.sdlv.dhgromm1.observable_data import cache_market_changes

    scenarios,output = tmp_path/"scenarios",tmp_path/"marks"
    (scenarios/sample).mkdir(parents=True)
    z = [-2.,-1.5,-1.,-.5,-.25]
    identity = dict(schema="dhgromm1-expiry-options-v4",return_convention="floating-payoff-minus-price-v1",
        substeps=1,instruments=21,batch_size=4,z=z,seed=17,calibration="/unused",calibration_sha256="fixture")
    (scenarios/"identity.json").write_text(json.dumps(identity))
    prices = np.zeros((4,21,21))
    prices[:,:,0] = 1
    prices[:,:20,17:] = 1
    for t in range(20):
        tenors = np.array([1,5,min(10,25-t)])
        prices[:,t,2:17] = (1-np.exp(.2*np.sqrt(tenors[:,None]/252)*np.array(z))).ravel()
    returns,payoff = np.zeros((4,20,21)),np.zeros(4)
    np.savez(scenarios/sample/"000000.npz",prices=prices,returns=returns,payoff=payoff)
    observed = []

    def generate(**kwargs):
        observed.append(kwargs["seed"].entropy)
        for day in range(20):
            kwargs["market_sink"](day,0,4,jnp.array([1.]),jnp.ones((4,25,1)),jnp.ones(4),jnp.ones(4))
        return prices.copy(),returns.copy(),payoff.copy()

    monkeypatch.setattr(rolling_simulation,"generate_chunk",generate)
    monkeypatch.setattr(jax_model,"black_mixture_call_prices",
        lambda support,density,strikes,background,remaining:jnp.maximum(1-strikes,0))
    monitor = SimpleNamespace(stage=lambda *args:None)
    actual = cache_market_changes(scenarios=scenarios,output=output,paths=4,monitor=monitor,sample=sample)
    assert observed==[[17,seed_index,0]]
    assert json.loads((output/"identity.json").read_text())["sample"]==sample
    assert actual.shape==(4,3990)
    np.testing.assert_allclose(actual,0,atol=2e-7)
