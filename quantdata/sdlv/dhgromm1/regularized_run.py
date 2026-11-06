"""Warm-start learned zero-cost policies with action penalties outside OCE."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

import quantdata
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.allocation import claim_training_gpu
from quantdata.sdlv.dhgromm1.cache_load import load_retained_market_cache
from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    FloatingTriple,
    FloatingTripleConfig,
    read_floating_sample,
)
from quantdata.sdlv.dhgromm1.floating_run import (
    load_floating_checkpoint,
    train_floating,
    verify_compiled,
)
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.sdlv.dhgromm1.persistence import atomic_json
from quantdata.sdlv.dhgromm1.regularization import ActionRegularizerConfig
from quantdata.sdlv.dhgromm1.regularized_evaluation import (
    evaluate_regularized,
    oce_statistics,
    plot_regularized_history,
)
from quantdata.sdlv.dhgromm1.time_decider import (
    TimeDeciderTriple,
    TimeDeciderTripleConfig,
    TimeDeciderWithA10Config,
    TimeDeciderWithA10Triple,
)
from quantdata.sdlv.dhgromm1.training import TrainConfig
from quantdata.torch import BaseEnvironment


def restore_config(template, record: dict):
    """Rebuild nested dataclasses without dropping unknown geometry fields."""
    if {f.name for f in fields(template)} != set(record):
        raise ValueError("source configuration fields changed")
    values = {}
    for key, value in record.items():
        default = getattr(template, key)
        values[key] = (restore_config(default, value) if is_dataclass(default) else
                       tuple(value) if isinstance(default, tuple) else value)
    return replace(template, **values)


def load_zero_cost_source(path: Path) -> tuple[dict, FloatingTripleConfig, type]:
    """Explicit objective migration, with strict original architecture/version checks."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    record = raw["model_config"]
    clock = "time_frequencies" in record["policy"]
    a10 = "empty.day10" in raw["state_dict"]
    config_type, model_type = ((TimeDeciderWithA10Config, TimeDeciderWithA10Triple) if clock and a10 else
                              (TimeDeciderTripleConfig, TimeDeciderTriple) if clock else
                              (FloatingTripleConfig, FloatingTriple))
    if raw.get("model_code_version") != config_type.version.unique_id48:
        raise ValueError("source model code version differs; no silent architecture migration")
    config = restore_config(config_type(label=record["label"]), record)
    saved = load_floating_checkpoint(path, config=config)
    if config.policy.trade_l1 != 0 or config.policy.trade_l2 != 0:
        raise ValueError("warm start must be a zero-cost baseline, not an old cost experiment")
    return saved, config, model_type


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500, help="additional regularized phase epochs")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-8)
    args = parser.parse_args()
    output, source = args.output.resolve(), args.source_run.resolve()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()) or output == source:
        raise ValueError("use a distinct external run directory; preserve the source baseline")
    if quantdata.env.test_mode or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root):
        raise RuntimeError("detached environment did not discover the external project data root")
    checks = resource_health()
    if not checks or any(c["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for c in checks):
        raise RuntimeError("resource manager blocks launch")
    if any(c["health"] != "OK" for c in checks):
        print(json.dumps(dict(stage="regularized-resource-warning", checks=checks)), flush=True)
    if json.loads((source/"status.json").read_text()).get("status") != "complete":
        raise ValueError("finish the source baseline before starting its regularized experiment")
    output.mkdir(parents=True, exist_ok=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    try:
        with FileLock(output/"training.lock", timeout=0):
            torch.set_num_threads(4)
            torch.manual_seed(20261105)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            path = source/"floating.checkpoint.pt"
            source_digest = sha256(path)
            saved, config, model_type = load_zero_cost_source(path)
            if source_digest != sha256(path):
                raise RuntimeError("source checkpoint changed while it was read")
            identity = json.loads((source/"identity.json").read_text())
            scenarios, cache = Path(identity["scenarios"]), Path(identity["cache"])
            if sha256(scenarios/"identity.json") != identity["scenario_identity_sha256"]:
                raise ValueError("source scenarios changed")
            scenario = json.loads((scenarios/"identity.json").read_text())
            if sha256(Path(scenario["calibration"])/"diffusion.npz") != scenario["calibration_sha256"]:
                raise ValueError("source calibration changed")
            regularizer = ActionRegularizerConfig(l1=args.l1, l2=args.l2)
            train = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                                learning_rate=args.learning_rate, patience=0, compile_model=True)
            provenance = dict(objective="oce_minus_action_regularizer", model_config=asdict(config),
                model_code_version=type(config).version.unique_id48,
                action_regularizer=asdict(regularizer), regularizer_code_version=ActionRegularizerConfig.version.unique_id48,
                transaction_costs=0, cost_treatment="zero; no wealth deduction",
                objective_formula="SUM_s[U_100(G_s+s*(H-mQ))-l1*E(sum(abs(a_s)))-l2*E(sum(a_s**2))]",
                penalty_units="executed floating contract quantities; sum 20 dates x 21 instruments; mean over paths",
                source_run=str(source), source_checkpoint_sha256=source_digest,
                source_selected_epochs=saved["selected_epochs"], source_resume_epoch=saved["resume_epoch"],
                initialization="independent source-best weights/cash; fresh Adam for changed objective; saved source RNG",
                selection="independent regularized training-best; raw OCE reported separately",
                scenarios=str(scenarios), scenario_identity_sha256=identity["scenario_identity_sha256"], cache=str(cache),
                measure=identity.get("measure", scenario.get("measure", "Q")),
                simulation_substeps_per_day=scenario["substeps"], hedge_decisions_per_day=1,
                paths=saved["training_paths"], epochs=args.epochs, train_config=asdict(train), validation_used=False,
                q_warning=identity.get("q_warning"),
                code_sha256={str(p): sha256(p) for p in [Path(__file__),
                    *(Path(__file__).with_name(n) for n in ("floating_run.py", "floating.py", "time_decider.py",
                        "regularization.py", "regularized_evaluation.py", "cache_load.py", "observable_data.py"))]})
            if (output/"identity.json").exists():
                old = json.loads((output/"identity.json").read_text())
                operational = {"epochs", "train_config"}
                if {k:v for k,v in old.items() if k not in operational} != {k:v for k,v in provenance.items() if k not in operational}:
                    raise ValueError("regularized run identity changed; use a new directory")
            atomic_json(output/"identity.json", provenance)
            atomic_json(output/"status.json", dict(status="running", stage="load-reusable-paths-and-marks"))
            monitor.stage("load-unchanged-floating-training-data")
            arrays = read_floating_sample(scenarios/"train", paths=saved["training_paths"])
            changes, cache_report = load_retained_market_cache(cache=cache, scenarios=scenarios,
                paths=saved["training_paths"], monitor=monitor)
            atomic_json(output/"cache-verification.json", cache_report)
            state = saved["state_dict"]
            # No price/return means are refitted, especially not under P.
            with np.load(source/"training-centering.npz", allow_pickle=False) as centering:
                for key, buffer in (("price_mean", "empty.price_mean"), ("price_scale", "empty.price_scale"),
                                    ("return_mean", "empty.return_mean"), ("payoff_mean", "payoff_mean")):
                    np.testing.assert_array_equal(centering[key].astype(np.float32), state[buffer].numpy())
            if stop["requested"]:
                atomic_json(output/"status.json", dict(status="stopped", stage="before-gpu-claim"))
                return
            if torch.cuda.is_initialized():
                raise RuntimeError("CUDA initialized before managed GPU claim")
            claim_training_gpu(output)
            sample = ObservableSample(arrays=arrays, changes=changes, device="cuda")
            del changes
            gc.collect()
            model = model_type(BaseEnvironment(device="cuda", precision=32), config=config,
                mean=state["empty.price_mean"], scale=state["empty.price_scale"],
                return_mean=state["empty.return_mean"], payoff_mean=float(state["payoff_mean"]))
            model.load_state_dict(state)
            for label in LABELS:
                getattr(model, label).load_state_dict(saved["best_states"][label])
            monitor.stage("verify-regularized-scan-values-and-all-gradients")
            atomic_json(output/"gpu-verification.json", verify_compiled(model, sample,
                batch_size=train.batch_size, regularizer=regularizer))
            baseline = output/"initial-evaluation.npz"
            if not baseline.exists():
                monitor.stage("evaluate-frozen-source-best")
                initial, values, influence = evaluate_regularized(model=model, sample=sample, arrays=arrays,
                    source_identity=identity, regularizer=regularizer, batch_size=train.batch_size, monitor=monitor)
                atomic_json(output/"initial-evaluation.json", initial)
                temporary = baseline.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, values=values, influence=influence)
                temporary.replace(baseline)
            fit = train_floating(model=model, sample=sample, config=config, train=train, output=output,
                monitor=monitor, stop=stop, initial_benchmark=None, regularizer=regularizer, warm_start=saved)
            if fit["status"] == "complete":
                monitor.stage("evaluate-regularized-training-best")
                report, values, influence = evaluate_regularized(model=model, sample=sample, arrays=arrays,
                    source_identity=identity, regularizer=regularizer, batch_size=train.batch_size, monitor=monitor)
                with np.load(baseline, allow_pickle=False) as before:
                    report["raw_oce_change_from_warm_start"] = {k:oce_statistics(values[:,i]-before["values"][:,i],
                        influence[:,i]-before["influence"][:,i]) for i,k in enumerate(LABELS)}
                report["fit"] = fit
                atomic_json(output/"report.json", report)
                plot_regularized_history(output)
                monitor.stage("complete", train.epochs, train.epochs)
            atomic_json(output/"status.json", fit)
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
