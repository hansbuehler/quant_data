"""Lossless sharing, canonical gains, causal marks, hashes and portable loading."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantdata.sdlv.dhgromm1.datasets.q1_20250829_100k.dataset import (
    Dataset,
    expand_changes,
    sha256,
    verify_arrays,
)
from quantdata.sdlv.dhgromm1.share_dataset import export_dataset


def sources(root: Path):
    scenarios,marks = root/"scenarios",root/"marks"
    (scenarios/"train").mkdir(parents=True)
    marks.mkdir()
    n = 4
    rng = np.random.default_rng(17)
    p = np.full((n,21,21),.02)
    p[:,:,0] = np.exp(rng.normal(0,.01,(n,21)))
    p[:,:,17:] = p[:,:,:1]
    h = np.abs(p[:,20,0]/p[:,10,0]-1)
    r = rng.normal(0,.01,(n,20,21))
    r[:,:,0] = p[:,20:21,0]/p[:,:20,0]-1
    r[:,10,1] = np.maximum(p[:,20,0]-p[:,10,0],0)/p[:,10,1]-1
    r[:,10,17] = p[:,20,0]/p[:,10,0]-1
    gains = r*(p[:,:20]/p[:,:20,:1])
    path = scenarios/"train/000000.npz"
    np.savez_compressed(path,prices=p,returns=r,payoff=h)
    identity = dict(schema="dhgromm1-expiry-options-v3",substeps=1,paths_per_sample=n,chunk_size=n,
                    instrument_order=[f"i{i}" for i in range(21)],z=[-2,-1.5,-1,-.5,-.25],
                    calibration_sha256="fixed-calibration",seed=17)
    (scenarios/"identity.json").write_text(json.dumps(identity))
    mark_identity = dict(scenario_identity_sha256=sha256(scenarios/"identity.json"),sample="train",paths=n)
    record = dict(start=0,count=n,scenario_chunk_sha256=sha256(path))
    changes = rng.normal(0,.01,(n,3990)).astype(np.float32)
    np.savez_compressed(marks/"000000.npz",changes=changes,record=json.dumps(record))
    (marks/"identity.json").write_text(json.dumps(mark_identity))
    (marks/"complete.json").write_text(json.dumps(dict(status="complete",**mark_identity,chunks=[record])))
    normalized = p[:,:20]/p[:,:20,:1]
    normalized[:,:,0] = p[:,:20,0]
    normalization = root/"normalization.npz"
    np.savez_compressed(normalization,return_mean=gains.mean(0),payoff_mean=h.mean(),
                       price_mean=normalized.mean((0,1)),price_scale=np.maximum(normalized.std((0,1)),1e-4))
    return scenarios,marks,normalization,dict(prices=p,gains=gains,payoff=h,changes_packed=changes)


def test_lossless_export_canonicalizes_once_and_checks_original_pair_order(tmp_path):
    scenarios,marks,normalization,expected = sources(tmp_path)
    output = tmp_path/"bundle"
    export_dataset(scenarios=scenarios,marks=marks,normalization=normalization,output=output)
    data = Dataset(output)
    records = list(data.chunks())
    assert len(records)==1 and records[0][0]["start"]==0
    for key in expected:
        np.testing.assert_array_equal(records[0][1][key],expected[key])
    normal = data.normalization()
    for key in ("return_mean","payoff_mean","price_mean","price_scale"):
        np.testing.assert_array_equal(normal[f"model_{key}"],normal[key].astype(np.float32))
    assert data.summary()["independent_antithetic_pairs"]==2
    assert not any(str(tmp_path) in json.dumps(v) for v in data.manifest.values())


def test_causal_marks_decode_only_earlier_cohorts_and_match_day_view():
    packed = np.arange(2*3990,dtype=np.float32).reshape(2,3990)
    expanded = expand_changes(packed)
    assert expanded.shape==(2,20,420)
    for day in range(20):
        np.testing.assert_array_equal(expanded[:,day],expand_changes(packed,day=day))
        assert not expanded[:,day,21*day:].any()
        begin,end = 21*day*(day-1)//2,21*day*(day+1)//2
        np.testing.assert_array_equal(expanded[:,day,:21*day],packed[:,begin:end])
    with pytest.raises(ValueError):
        expand_changes(packed,day=20)


def test_payoff_or_premium_return_mislabel_is_rejected(tmp_path):
    _,_,_,arrays = sources(tmp_path)
    verify_arrays(arrays,paths=4)
    wrong = {k:v.copy() for k,v in arrays.items()}
    wrong["gains"][:,10,1] *= 100
    with pytest.raises(AssertionError):
        verify_arrays(wrong,paths=4)
    wrong = {k:v.copy() for k,v in arrays.items()}
    wrong["payoff"] += .1
    with pytest.raises(AssertionError):
        verify_arrays(wrong,paths=4)


def test_changed_source_hash_is_rejected_before_exporting_wrong_data(tmp_path):
    scenarios,marks,normalization,_ = sources(tmp_path)
    with (scenarios/"train/000000.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError,match="scenario chunk changed"):
        export_dataset(scenarios=scenarios,marks=marks,normalization=normalization,output=tmp_path/"bundle")


def test_manifest_paths_cannot_escape_the_share_bundle(tmp_path):
    scenarios,marks,normalization,_ = sources(tmp_path)
    output = tmp_path/"bundle"
    export_dataset(scenarios=scenarios,marks=marks,normalization=normalization,output=output)
    manifest = json.loads((output/"MANIFEST.json").read_text())
    manifest["chunks"][0]["file"] = "../outside.npz"
    (output/"MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="escaped"):
        Dataset(output)
