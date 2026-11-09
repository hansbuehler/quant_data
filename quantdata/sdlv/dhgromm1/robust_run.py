"""Warm-start the one-step a10 case with global bottom-90% OCE and no penalties."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

import quantdata
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.allocation import claim_training_gpu
from quantdata.sdlv.dhgromm1.cache_load import load_retained_market_cache
from quantdata.sdlv.dhgromm1.floating import LABELS, FloatingTripleConfig, read_floating_sample
from quantdata.sdlv.dhgromm1.floating_run import load_floating_checkpoint, verify_compiled
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.sdlv.dhgromm1.persistence import atomic_json
from quantdata.sdlv.dhgromm1.regularization import ActionRegularizerConfig
from quantdata.sdlv.dhgromm1.regularized_run import restore_config
from quantdata.sdlv.dhgromm1.robust import (
    RobustTriple,
    RobustTripleConfig,
    migrate_selected_weights,
    profile_tail,
)
from quantdata.sdlv.dhgromm1.robust_evaluation import evaluate_robust, plot_robust_history
from quantdata.sdlv.dhgromm1.robust_training import collect_gains, train_robust
from quantdata.sdlv.dhgromm1.training import TrainConfig
from quantdata.torch import BaseEnvironment


def load_a10_source(path: Path) -> tuple[dict,FloatingTripleConfig]:
    """Accept ordinary/penalized source weights, not arbitrary architecture changes."""
    raw = torch.load(path,map_location="cpu",weights_only=False)
    if raw.get("model_code_version") != FloatingTripleConfig.version.unique_id48:
        raise ValueError("expected the original one-step a10 neural architecture/version")
    config = restore_config(FloatingTripleConfig(label=raw["model_config"]["label"]),raw["model_config"])
    source_penalty = ActionRegularizerConfig(**raw["action_regularizer"]) if raw.get("action_regularizer") else None
    saved = load_floating_checkpoint(path,config=config,regularizer=source_penalty)
    if config.policy.trade_l1 or config.policy.trade_l2 or "empty.day10" not in saved["state_dict"]:
        raise ValueError("source must use floating a0/a10 controls with zero transaction costs")
    return saved,config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=500)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    args = parser.parse_args()
    output,source = args.output.resolve(),args.source_run.resolve()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()) or output==source:
        raise ValueError("use a new external robust run directory")
    if quantdata.env.test_mode or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root):
        raise RuntimeError("detached environment did not discover the external data root")
    checks = resource_health()
    if not checks or any(c["health"] in {"STOP","ERROR","UNAVAILABLE"} for c in checks):
        raise RuntimeError("resource manager blocks robust launch")
    if any(c["health"]!="OK" for c in checks):
        print(json.dumps(dict(stage="robust-resource-warning",checks=checks)),flush=True)
    if json.loads((source/"status.json").read_text()).get("status")!="complete":
        raise ValueError("warm-start source must be completed and immutable")
    output.mkdir(parents=True,exist_ok=True)
    monitor,stop = AuditMonitor(output),{"requested":False}
    monitor.thread.start()
    signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True))
    try:
        with FileLock(output/"training.lock",timeout=0):
            torch.set_num_threads(4)
            torch.manual_seed(20261106)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            path = source/"floating.checkpoint.pt"
            digest = sha256(path)
            saved,source_config = load_a10_source(path)
            if digest!=sha256(path):
                raise RuntimeError("source checkpoint changed during migration")
            source_identity = json.loads((source/"identity.json").read_text())
            scenarios,cache = Path(source_identity["scenarios"]),Path(source_identity["cache"])
            scenario = json.loads((scenarios/"identity.json").read_text())
            if (scenario.get("measure","Q")!="Q" or scenario["substeps"]!=1 or scenario["paths_per_sample"]!=100_000
                    or sha256(scenarios/"identity.json")!=source_identity["scenario_identity_sha256"]
                    or sha256(Path(scenario["calibration"])/"diffusion.npz")!=scenario["calibration_sha256"]):
                raise ValueError("use the unchanged 100000-path Q one-step sample and calibration")
            config = RobustTripleConfig(label="Q1-a10-global-bottom90-no-penalties",policy=source_config.policy)
            train = TrainConfig(epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,
                                patience=0,compile_model=True)
            provenance = dict(objective="global-bottom90-OCE",model_config=asdict(config),
                model_code_version=RobustTripleConfig.version.unique_id48,train_config=asdict(train),
                measure="Q",simulation_substeps_per_day=1,hedge_decisions_per_day=1,paths=100_000,
                retained_individual_paths_per_policy=90_000,tail_fraction=.9,transaction_costs=0,l1=0,l2=0,weight_decay=0,
                objective_formula="SUM_s sup_(c,eta) [c+eta-(1/(0.9*N))*SUM_n relu(eta-u_100(G_s,n+s*(H_n-mQ)-c))]",
                tail_selection="lowest individual utility contributions per policy; full population, not per batch or pair",
                minibatches="unbiased additive estimator for one jointly learned global eta per policy",
                scoring="exact global bottom 90000 of all 100000 paths; logs variational threshold gap",
                source_run=str(source),source_checkpoint_sha256=digest,source_selected_epochs=saved["selected_epochs"],
                source_objective=source_identity["objective"],removed_source_penalties=saved.get("action_regularizer"),
                initialization="selected learned a0/a10 weights; globally reprofiled robust cash/cutoff; fresh Adam; saved RNG",
                scenarios=str(scenarios),scenario_identity_sha256=source_identity["scenario_identity_sha256"],cache=str(cache),
                centering="unchanged checkpoint Q gain/payoff means and feature scales; NO refitting",validation_used=False,
                q_warning="Same accepted one-step Q diagnostic warning; robust utility is not an arbitrage proof",
                code_sha256={str(p):sha256(p) for p in [Path(__file__),*(Path(__file__).with_name(n) for n in (
                    "robust.py","robust_training.py","robust_evaluation.py","floating.py","floating_run.py",
                    "observable_data.py","cache_load.py","allocation.py"))]})
            if (output/"identity.json").exists():
                old = json.loads((output/"identity.json").read_text())
                if {k:v for k,v in old.items() if k!="train_config"}!={k:v for k,v in provenance.items() if k!="train_config"}:
                    raise ValueError("robust logical run identity changed; use a new directory")
            atomic_json(output/"identity.json",provenance)
            atomic_json(output/"status.json",dict(status="running",stage="load-retained-sample"))
            monitor.stage("load-retained-one-step-training-sample")
            arrays = read_floating_sample(scenarios/"train",paths=100_000)
            changes,cache_report = load_retained_market_cache(cache=cache,scenarios=scenarios,paths=100_000,monitor=monitor)
            atomic_json(output/"cache-verification.json",cache_report)
            if stop["requested"]:
                atomic_json(output/"status.json",dict(status="stopped",stage="before-gpu-claim"))
                return
            if torch.cuda.is_initialized():
                raise RuntimeError("CUDA initialized before a managed GPU claim")
            claim_training_gpu(output)
            sample = ObservableSample(arrays=arrays,changes=changes,device="cuda")
            del changes
            gc.collect()
            state = saved["state_dict"]
            model = RobustTriple(BaseEnvironment(device="cuda",precision=32),config=config,
                mean=state["empty.price_mean"],scale=state["empty.price_scale"],
                return_mean=state["empty.return_mean"],payoff_mean=float(state["payoff_mean"]))
            migrate_selected_weights(model,saved)
            monitor.stage("profile-initial-global-cash-and-tail-threshold")
            functions = model.policy_functions(compile_model=True)
            gains = collect_gains(model,sample,functions=functions,batch_size=train.batch_size,monitor=monitor).cpu().numpy().astype(np.float64)
            with torch.no_grad():
                for i,(label,sign) in enumerate(zip(LABELS,(0,1,-1),strict=True)):
                    profile = profile_tail(gains[i]+sign*(arrays[2]-float(model.payoff_mean)),fraction=.9)
                    getattr(model,label).translation.fill_(profile["cash"])
                    getattr(model,label).tail_threshold.fill_(profile["threshold"])
            del gains
            monitor.stage("verify-robust-gpu-scan-values-and-gradients")
            verified = verify_compiled(model,sample,batch_size=train.batch_size)
            atomic_json(output/"gpu-verification.json",verified)
            print(json.dumps(dict(stage="robust-verification",**verified)),flush=True)
            if not (output/"initial-evaluation.json").exists():
                monitor.stage("evaluate-source-policy-global-tail")
                initial,values = evaluate_robust(model=model,sample=sample,arrays=arrays,batch_size=train.batch_size,monitor=monitor)
                atomic_json(output/"initial-evaluation.json",initial)
                np.savez_compressed(output/"initial-evaluation.npz",**values)
            # Setup/verification must not advance the actual training shuffle.
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            fit = train_robust(model=model,sample=sample,train=train,output=output,monitor=monitor,stop=stop)
            if fit["status"]=="complete":
                monitor.stage("evaluate-selected-global-tail-policies")
                report,values = evaluate_robust(model=model,sample=sample,arrays=arrays,batch_size=train.batch_size,monitor=monitor)
                report["fit"] = fit
                atomic_json(output/"report.json",report)
                np.savez_compressed(output/"final-evaluation.npz",**values)
                plot_robust_history(output)
                monitor.stage("complete",train.epochs,train.epochs)
            atomic_json(output/"status.json",fit)
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
