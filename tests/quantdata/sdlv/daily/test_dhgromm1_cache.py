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
