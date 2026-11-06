"""Warm-start ordinary Q1 DH with actual 1 bp cash-value trade fees inside u."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
from dataclasses import asdict, fields
from pathlib import Path

import torch
from filelock import FileLock

import quantdata
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.allocation import claim_training_gpu
from quantdata.sdlv.dhgromm1.cache_load import load_retained_market_cache
from quantdata.sdlv.dhgromm1.floating import (
    LABELS,
    FloatingConfig,
    FloatingTripleConfig,
    read_floating_sample,
)
from quantdata.sdlv.dhgromm1.floating_run import (
    load_floating_checkpoint,
    summarize,
    train_floating,
    verify_compiled,
)
from quantdata.sdlv.dhgromm1.observable_data import ObservableSample
from quantdata.sdlv.dhgromm1.persistence import atomic_json
from quantdata.sdlv.dhgromm1.regularization import ActionRegularizerConfig
from quantdata.sdlv.dhgromm1.regularized_run import restore_config
from quantdata.sdlv.dhgromm1.training import TrainConfig
from quantdata.sdlv.dhgromm1.transaction_cost import (
    TransactionCostPolicyConfig,
    TransactionCostTriple,
    TransactionCostTripleConfig,
)
from quantdata.sdlv.dhgromm1.triple_plot import plot_triple_history
from quantdata.torch import BaseEnvironment


def load_cost_source(path: Path) -> tuple[dict, FloatingTripleConfig]:
    """Strictly load old ordinary/regularized architecture before explicit migration."""
    raw = torch.load(path,map_location="cpu",weights_only=False)
    if raw.get("model_code_version") != FloatingTripleConfig.version.unique_id48:
        raise ValueError("source model code version differs; no silent architecture migration")
    old = restore_config(FloatingTripleConfig(label=raw["model_config"]["label"]),raw["model_config"])
    reg = ActionRegularizerConfig(**raw["action_regularizer"]) if raw.get("action_regularizer") else None
    saved = load_floating_checkpoint(path,config=old,regularizer=reg)
    if old.policy.trade_l1 or old.policy.trade_l2 or "empty.day10" not in saved["state_dict"]:
        raise ValueError("expected completed zero-cost floating a0/a10 source weights")
    return saved,old


def load_transaction_cost_model(path: Path, *, device: str = "cpu", precision: int = 32
                                ) -> TransactionCostTriple:
    """Read-only selected-weight load; never trains or claims a GPU implicitly."""
    raw = torch.load(path,map_location="cpu",weights_only=False)
    if raw.get("model_code_version") != TransactionCostTripleConfig.version.unique_id48:
        raise ValueError("transaction-cost checkpoint code version differs")
    config = restore_config(TransactionCostTripleConfig(label=raw["model_config"]["label"]),raw["model_config"])
    saved = load_floating_checkpoint(path,config=config)
    state = saved["state_dict"]
    model = TransactionCostTriple(BaseEnvironment(device=device,precision=precision),config=config,
        mean=state["empty.price_mean"],scale=state["empty.price_scale"],
        return_mean=state["empty.return_mean"],payoff_mean=float(state["payoff_mean"]))
    model.load_state_dict(state,strict=True)
    for label in LABELS:
        getattr(model,label).load_state_dict(saved["best_states"][label],strict=True)
    return model.eval()


def with_three_errors(value):
    """Retain old statistics while making the requested ±3 MC errors explicit."""
    if isinstance(value,dict):
        result = {k:with_three_errors(v) for k,v in value.items()}
        if "standard_error" in result:
            result["three_standard_errors"] = 3*result["standard_error"]
        return result
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=500)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    output,source = args.output.resolve(),args.source_run.resolve()
    if (quantdata.env.test_mode or not output.is_relative_to(root) or output.is_relative_to(Path.cwd()) or output==source
            or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root)):
        raise ValueError("expected explicit production root and unique external output")
    checks = resource_health()
    if not checks or any(c["health"] in {"STOP","ERROR","UNAVAILABLE"} for c in checks):
        raise RuntimeError("resource manager blocks the cost run")
    if any(c["health"]!="OK" for c in checks):
        print(json.dumps(dict(stage="transaction-cost-resource-warning",checks=checks)),flush=True)
    print(json.dumps(dict(stage="transaction-cost-detached-preflight-passed",training_root=str(quantdata.env.locations.training.path),
        test_mode=quantdata.env.test_mode)),flush=True)
    if json.loads((source/"status.json").read_text()).get("status")!="complete":
        raise ValueError("cost warm start must be completed and immutable")
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
            saved,old = load_cost_source(path)
            if digest!=sha256(path):
                raise RuntimeError("source checkpoint changed during objective migration")
            source_identity = json.loads((source/"identity.json").read_text())
            scenarios,cache = Path(source_identity["scenarios"]),Path(source_identity["cache"])
            scenario = json.loads((scenarios/"identity.json").read_text())
            if (scenario.get("measure","Q")!="Q" or scenario["substeps"]!=1 or saved["training_paths"]!=100000
                    or sha256(scenarios/"identity.json")!=source_identity["scenario_identity_sha256"]
                    or sha256(Path(scenario["calibration"])/"diffusion.npz")!=scenario["calibration_sha256"]):
                raise ValueError("expected the unchanged reference-date 100000-path Q1 sample")
            policy_values = {f.name:getattr(old.policy,f.name) for f in fields(FloatingConfig)}
            policy_values.update(group="sdlv/dhgromm1/transaction-cost",trade_l1=0.,trade_l2=0.)
            config = TransactionCostTripleConfig(label="Q1-a10-1bp-cash-value-inside-utility",
                policy=TransactionCostPolicyConfig(**policy_values,transaction_cost_rate=1e-4,transaction_cost_basis="cash_value"))
            train = TrainConfig(epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,
                patience=0,compile_model=True)
            provenance = dict(objective="ordinary-OCE-with-cash-transaction-costs",model_config=asdict(config),
                model_code_version=type(config).version.unique_id48,train_config=asdict(train),
                epochs=train.epochs,cost_treatment="inside_utility",checkpoint_selection="independent_policy_best",
                policy_kind="causal LSTM/current features + affine/deep + a0/a10; 1 bp cash-value fees",
                transaction_cost_rate=1e-4,transaction_cost_basis="cash_value",l1_regularizer=0,l2_regularizer=0,
                cost_formula="C=sum_(t,i) 0.0001*abs(a_(t,i))*P_(t,i)/S_t",
                objective_formula="SUM_s sup_c [c+E(u_100(G_s+s*(H-mQ)-C_s-c))]",
                cost_feedback="cash already paid reduces current marked-PNL feature; no forward-looking cost input",
                cost_timing="executed opening trades; natural cash settlement free; existing terminal marking convention unchanged",
                source_run=str(source),source_checkpoint_sha256=digest,source_selected_epochs=saved["selected_epochs"],
                removed_regularizer=saved.get("action_regularizer"),initialization="selected learned weights; fresh Adam; source RNG; selection reset for changed objective",
                scenarios=str(scenarios),scenario_identity_sha256=source_identity["scenario_identity_sha256"],cache=str(cache),
                measure="Q",simulation_substeps_per_day=1,hedge_decisions_per_day=1,paths=100000,
                training_sample="existing original 100000 training paths; conditional boundary sampling remains an evaluation method in this run",
                validation_used=False,q_warning=source_identity.get("q_warning"),
                code_sha256={str(p):sha256(p) for p in (Path(__file__),*(Path(__file__).with_name(n) for n in
                    ("transaction_cost.py","floating_run.py","floating.py","observable_data.py","cache_load.py")))})
            if (output/"identity.json").exists():
                existing = json.loads((output/"identity.json").read_text())
                if {k:v for k,v in existing.items() if k!="train_config"}!={k:v for k,v in provenance.items() if k!="train_config"}:
                    raise ValueError("run identity changed; use a new cost run")
            atomic_json(output/"identity.json",provenance)
            atomic_json(output/"status.json",dict(status="running",stage="reuse-paths-and-causal-marks"))
            arrays = read_floating_sample(scenarios/"train",paths=100000)
            marks,report = load_retained_market_cache(cache=cache,scenarios=scenarios,paths=100000,monitor=monitor)
            atomic_json(output/"cache-verification.json",report)
            if stop["requested"]:
                atomic_json(output/"status.json",dict(status="stopped",stage="before-gpu-claim"))
                return
            if torch.cuda.is_initialized():
                raise RuntimeError("CUDA initialized before a managed allocation")
            claim_training_gpu(output)
            sample = ObservableSample(arrays=arrays,changes=marks,device="cuda")
            del marks
            gc.collect()
            state = saved["state_dict"]
            model = TransactionCostTriple(BaseEnvironment(device="cuda",precision=32),config=config,
                mean=state["empty.price_mean"],scale=state["empty.price_scale"],
                return_mean=state["empty.return_mean"],payoff_mean=float(state["payoff_mean"]))
            model.load_state_dict(state,strict=True)
            for label in LABELS:
                getattr(model,label).load_state_dict(saved["best_states"][label],strict=True)
            monitor.stage("verify-cost-scan-values-and-gradients")
            atomic_json(output/"gpu-verification.json",verify_compiled(model,sample,batch_size=train.batch_size))
            monitor.stage("score-cost-warm-start")
            initial = with_three_errors(summarize(model,sample,payoff=arrays[2],batch_size=train.batch_size))
            initial["transaction_costs"] = dict(rate=1e-4,basis="cash_value",inside_utility=True,regularizers=False)
            atomic_json(output/"initial-evaluation.json",initial)
            fit = train_floating(model=model,sample=sample,config=config,train=train,output=output,monitor=monitor,
                stop=stop,initial_benchmark=None,warm_start=saved,allow_selected_cost_warm_start=True)
            if fit["status"]=="complete":
                monitor.stage("evaluate-cost-training-best")
                report = with_three_errors(summarize(model,sample,payoff=arrays[2],batch_size=train.batch_size))
                report.update(fit=fit,transaction_cost_rate=1e-4,transaction_cost_basis="cash_value",
                    regularizers=False,uncertainty="±3 antithetic-pair MC SE; original training constants and selected policies fixed; fitting/selection uncertainty excluded")
                atomic_json(output/"report.json",report)
                if not (output/"figures/losses.png").exists():
                    plot_triple_history(output,output/"figures")
                monitor.stage("complete",train.epochs,train.epochs)
            atomic_json(output/"status.json",fit)
    finally:
        monitor.done.set()


if __name__=="__main__":
    main()
