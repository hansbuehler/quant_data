"""Prepare held-out marks on CPU; score three frozen Q policies after robust training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

import quantdata
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.cache_load import load_retained_market_cache
from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    FloatingTriple,
    FloatingTripleConfig,
    read_floating_sample,
)
from quantdata.sdlv.dhgromm1.floating_run import load_floating_checkpoint
from quantdata.sdlv.dhgromm1.heldout_evaluation import evaluate_heldout
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample, cache_market_changes
from quantdata.sdlv.dhgromm1.persistence import atomic_json
from quantdata.sdlv.dhgromm1.regularization import ActionRegularizerConfig
from quantdata.sdlv.dhgromm1.regularized_run import restore_config
from quantdata.sdlv.dhgromm1.robust import RobustTriple, RobustTripleConfig
from quantdata.sdlv.dhgromm1.robust_training import load_robust_checkpoint
from quantdata.torch import BaseEnvironment


def check_resources() -> None:
    checks = resource_health()
    if any(c["health"] in {"STOP","ERROR","UNAVAILABLE"} for c in checks):
        raise RuntimeError(f"resource manager blocks held-out evaluation: {checks}")
    if any(c["health"]!="OK" for c in checks):
        print(json.dumps(dict(event="resource_warning",checks=checks)),flush=True)


def robust_source_ready(source: Path, *, stopped_sha256: str | None = None) -> bool:
    """Allow a user-stopped source only with its explicitly frozen checkpoint hash."""
    status = json.loads((source/"status.json").read_text()).get("status")
    if status=="stopped" and stopped_sha256 is None:
        raise RuntimeError("robust training stopped; explicitly approve its frozen checkpoint before validation")
    if status=="failed":
        raise RuntimeError("robust training failed; validation cannot use that source")
    if status not in {"complete","stopped"}:
        return False
    if stopped_sha256 is not None:
        with FileLock(source/"training.lock",timeout=0):
            if sha256(source/"robust.checkpoint.pt")!=stopped_sha256:
                raise ValueError("robust checkpoint differs from the explicitly approved stopped checkpoint")
    return True


def selected_model(source: Path, output: Path, *, robust: bool, expected_sha256: str | None = None):
    """Copy a complete checkpoint; independently load its TRAINING-best policies."""
    filename = "robust.checkpoint.pt" if robust else "floating.checkpoint.pt"
    frozen = output/filename
    shutil.copyfile(source/filename,frozen)
    if expected_sha256 is not None and sha256(frozen)!=expected_sha256:
        raise ValueError("copied checkpoint differs from the explicitly approved stopped checkpoint")
    raw = torch.load(frozen,map_location="cpu",weights_only=False)
    config_type,model_type = (RobustTripleConfig,RobustTriple) if robust else (FloatingTripleConfig,FloatingTriple)
    config = restore_config(config_type(label=raw["model_config"]["label"]),raw["model_config"])
    if robust:
        saved = load_robust_checkpoint(frozen,config=config)
    else:
        reg = ActionRegularizerConfig(**raw["action_regularizer"]) if raw.get("action_regularizer") else None
        saved = load_floating_checkpoint(frozen,config=config,regularizer=reg)
    if config.policy.trade_l1 or config.policy.trade_l2:
        raise ValueError("this validation compares the three zero-transaction-cost experiments")
    state = saved["state_dict"]
    model = model_type(BaseEnvironment(device="cpu",precision=32),config=config,
        mean=state["empty.price_mean"],scale=state["empty.price_scale"],return_mean=state["empty.return_mean"],
        payoff_mean=float(state["payoff_mean"]))
    model.load_state_dict(state)
    for label in LABELS:
        getattr(model,label).load_state_dict(saved["best_states"][label])
    model.eval()
    return model,dict(source=str(source),checkpoint_sha256=sha256(frozen),selected_epochs=saved["selected_epochs"],
                      completed_training_epoch=saved["resume_epoch"],resume_offset=saved.get("resume_offset",0),
                      source_training_status=json.loads((source/"status.json").read_text()),
                      weights_selected_on="TRAINING ONLY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run",type=Path,required=True)
    parser.add_argument("--regularized-run",type=Path,required=True)
    parser.add_argument("--robust-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--stopped-robust-sha256",help="Explicitly approved frozen checkpoint after a user-requested stop")
    parser.add_argument("--wait-for-validation-lock",action="store_true",
                        help="Continue this logical evaluation after its existing preparation process exits")
    args = parser.parse_args()
    # Limit THIS CPU diagnostic, not any running trainer or service.
    if hasattr(os,"sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0,set(allowed[-min(8,len(allowed)):]))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_initialized():
        raise RuntimeError("held-out evaluator must remain CPU-only")
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    output = args.output.resolve()
    if (quantdata.env.test_mode or not output.is_relative_to(root) or output.is_relative_to(Path.cwd())
            or any(output.is_relative_to(p.resolve()) for p in (args.baseline_run,args.regularized_run,args.robust_run))
            or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root)):
        raise ValueError("expected explicit production data root and a unique external run directory")
    output.mkdir(parents=True,exist_ok=True)
    check_resources()
    print(json.dumps(dict(stage="heldout-detached-preflight-passed",training_root=str(quantdata.env.locations.training.path),
        test_mode=quantdata.env.test_mode,cuda_initialized=torch.cuda.is_initialized(),output=str(output))),flush=True)
    monitor = AuditMonitor(output)
    try:
        if args.wait_for_validation_lock:
            print(json.dumps(dict(stage="await-existing-validation-preparation",output=str(output))),flush=True)
        with FileLock(output/"validation.lock",timeout=-1 if args.wait_for_validation_lock else 0):
            # The existing preparation owns its progress/status files until it
            # releases this lock. Never run two monitors against the same run.
            monitor.thread.start()
            check_resources()
            # Do not alter the original trainer's stopped status or claim that
            # all requested epochs completed. Freeze only its scored best policies.
            robust_source_ready(args.robust_run,stopped_sha256=args.stopped_robust_sha256)
            identity = json.loads((args.robust_run/"identity.json").read_text())
            scenarios = Path(identity["scenarios"])
            scenario = json.loads((scenarios/"identity.json").read_text())
            if (scenario.get("measure","Q")!="Q" or scenario["substeps"]!=1 or scenario["paths_per_sample"]!=100000
                    or sha256(scenarios/"identity.json")!=identity["scenario_identity_sha256"]):
                raise ValueError("expected the existing independent 100000-path one-step Q splits")
            for source in (args.baseline_run,args.regularized_run):
                record = json.loads((source/"identity.json").read_text())
                if record["scenario_identity_sha256"]!=identity["scenario_identity_sha256"]:
                    raise ValueError("comparison models used different scenario identities")
            provenance = dict(scenario_identity_sha256=identity["scenario_identity_sha256"],sample="evaluation",paths=100000,
                original_sample=str(scenarios/"evaluation"),training_sample=str(scenarios/"train"),
                split_innovations="SeedSequence([original_seed, 1, chunk_start]); training uses index 0",
                approved_stopped_robust_checkpoint_sha256=args.stopped_robust_sha256,
                device="CPU only; CUDA_VISIBLE_DEVICES empty; no GPU allocation",validation_used_for_training=False,
                code_sha256={str(p):sha256(p) for p in (Path(__file__),Path(__file__).with_name("heldout_evaluation.py"),
                    Path(__file__).with_name("observable_data.py"),Path(__file__).with_name("cache_load.py"))})
            atomic_json(output/"identity.json",provenance)
            atomic_json(output/"status.json",dict(status="running",stage="prepare-validation-marks"))
            cache = output/"validation-marks"
            # Replays only the original validation innovations to recover causal
            # quotes; every chunk must reproduce saved prices/gains/payoff.
            marks = cache_market_changes(scenarios=scenarios,output=cache,paths=100000,monitor=monitor,sample="evaluation")
            del marks
            gc.collect()
            while True:
                check_resources()
                if robust_source_ready(args.robust_run,stopped_sha256=args.stopped_robust_sha256):
                    break
                atomic_json(output/"status.json",dict(status="waiting",stage="await-final-robust-training-checkpoint"))
                monitor.stage("await-final-robust-training-checkpoint")
                time.sleep(30)
            check_resources()
            atomic_json(output/"status.json",dict(status="running",stage="load-independent-validation-sample"))
            arrays = read_floating_sample(scenarios/"evaluation",paths=100000)
            training = read_floating_sample(scenarios/"train",paths=100000)
            marks,cache_report = load_retained_market_cache(cache=cache,scenarios=scenarios,paths=100000,
                                                            monitor=monitor,sample="evaluation")
            atomic_json(output/"cache-verification.json",cache_report)
            sample = ObservableSample(arrays=arrays,changes=marks,device="cpu")
            del marks
            gc.collect()
            reports = {}
            for label,source,robust in (("ordinary",args.baseline_run,False),("regularized",args.regularized_run,False),
                                         ("robust",args.robust_run,True)):
                check_resources()
                folder = output/label
                folder.mkdir(exist_ok=False)
                model,checkpoint = selected_model(source,folder,robust=robust,
                    expected_sha256=args.stopped_robust_sha256 if robust else None)
                # Check frozen constants explicitly; no mean is fitted to validation.
                np.testing.assert_allclose(model.empty.return_mean.numpy(),training[1].mean(0).astype(np.float32),atol=1e-9,rtol=1e-6)
                monitor.stage(f"validate-frozen-{label}-policy",0,200000)
                report,retained = evaluate_heldout(model=model,sample=sample,arrays=arrays,training_arrays=training,
                    batch_size=args.batch_size,monitor=monitor)
                report["checkpoint"] = checkpoint
                report["sample"] = "original independent evaluation paths, not training"
                atomic_json(folder/"report.json",report)
                np.savez_compressed(folder/"utilities-and-influence.npz",**retained)
                reports[label] = report
                atomic_json(output/"results-so-far.json",reports)
                del model,retained
                gc.collect()
            atomic_json(output/"report.json",dict(status="complete",reports=reports,provenance=provenance))
            atomic_json(output/"status.json",dict(status="complete",models=list(reports),paths=100000))
            monitor.stage("complete",3,3)
    except Exception as error:
        atomic_json(output/"status.json",dict(status="failed",error=repr(error)))
        raise
    finally:
        monitor.done.set()


if __name__=="__main__":
    main()
