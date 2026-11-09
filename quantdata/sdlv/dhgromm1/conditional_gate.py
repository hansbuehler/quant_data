"""Reconstruct untouched held-out contracts before changing their sampling law."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

import quantdata
from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower_from_front
from quantdata.sdlv.daily.wide_audit import sha256
from quantdata.sdlv.dhgromm1.conditional_run import FrozenScorer, calibration_arrays, plain_market
from quantdata.sdlv.dhgromm1.contracts import canonical_gains
from quantdata.sdlv.dhgromm1.persistence import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios",type=Path,required=True)
    parser.add_argument("--validation-cache",type=Path,required=True)
    parser.add_argument("--baseline-run",type=Path,required=True)
    parser.add_argument("--regularized-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if (quantdata.env.test_mode or jax.default_backend()!="cpu" or torch.cuda.is_initialized()
            or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root)
            or not args.output.resolve().is_relative_to(root)):
        raise ValueError("detached gate must discover the production root and remain CPU-only")
    print(json.dumps(dict(stage="conditional-gate-detached-preflight-passed",training_root=str(quantdata.env.locations.training.path))),flush=True)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    identity = json.loads((args.scenarios/"identity.json").read_text())
    if identity["substeps"]!=1 or identity["batch_size"]!=128 or identity["z"]!=[-2.,-1.5,-1.,-.5,-.25]:
        raise ValueError("expected the existing one-step ITM scenario geometry")
    rng = np.random.default_rng(np.random.SeedSequence([identity["seed"],1,0]))
    z = rng.standard_normal((20,1,64,8))
    z = np.stack((z,-z),axis=3).reshape(20,1,128,8)[:,0,:16]
    b = rng.standard_normal((20,64))
    b = np.stack((b,-b),axis=2).reshape(20,128)[:,:16]
    calibration = calibration_arrays(Path(identity["calibration"]))
    market,actual = plain_market(calibration,z,b)
    with np.load(args.scenarios/"evaluation/000000.npz") as f:
        price,ret,h = f["prices"][:16],f["returns"][:16],f["payoff"][:16]
    expected = price,canonical_gains(price,ret,identity=identity),h
    errors = {}
    for key,a,e in zip(("prices","floating_gains","payoff"),actual,expected,strict=True):
        np.testing.assert_allclose(a,e,rtol=2e-8,atol=2e-10)
        errors[key] = float(np.abs(a-e).max())
    with np.load(args.validation_cache/"000000.npz") as f:
        marks = f["changes"][:16]
    np.testing.assert_allclose(market.packed,marks,rtol=3e-6,atol=5e-8)
    errors["causal_market_marks"] = float(np.abs(market.packed-marks).max())
    scorer = FrozenScorer(args.baseline_run,args.regularized_run,args.output)
    net,_ = scorer(market,actual)
    replay,_ = scorer(type(market)(price,marks),expected)
    np.testing.assert_allclose(net,replay,atol=2e-7,rtol=2e-5)
    errors["frozen_policy_terminal_wealth"] = float(np.abs(net-replay).max())
    support,tower0,dlv0,_ = calibration
    fronts = jax.nn.one_hot(jnp.array([0,len(support)-1]),len(support),dtype=jnp.float64)
    dlv = jnp.broadcast_to(dlv0[1:],(2,*dlv0[1:].shape))
    decoded = decode_density_tower_from_front(support,fronts,dlv,dt=1/252)
    np.testing.assert_allclose(decoded, np.broadcast_to(np.asarray(fronts)[:,None],decoded.shape),atol=2e-13,rtol=0)
    errors["endpoint_absorption_all_future_expiries"] = float(np.abs(np.asarray(decoded)-np.asarray(fronts)[:,None]).max())
    report = dict(status="passed",paths=16,sample="original held-out evaluation",errors=errors,
        calibration_sha256=identity["calibration_sha256"],sources=scorer.sources,
        code_sha256={str(p):sha256(p) for p in (Path(__file__),*(Path(__file__).with_name(n) for n in
            ("conditional_paths.py","conditional_oce.py","conditional_run.py")))})
    atomic_json(args.output/"report.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
