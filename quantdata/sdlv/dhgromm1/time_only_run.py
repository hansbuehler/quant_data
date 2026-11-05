"""Full-batch deterministic affine-time OCE training; no new simulation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

import quantdata
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.feedback_run import atomic_json
from quantdata.sdlv.dhgromm1.time_only import (
    LABELS,
    SIGNS,
    TimeOnlyConfig,
    TimeOnlyTriple,
    load_training_exposures,
)
from quantdata.sdlv.dhgromm1.training import (
    TrainConfig,
    _economic_utility,
    optimize_translation,
    pair_summary,
)
from quantdata.sdlv.dhgromm1.triple_plot import plot_triple_history
from quantdata.torch import BaseEnvironment


def load_time_only_checkpoint(path: Path, *, config: TimeOnlyConfig) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("model_code_version") != TimeOnlyConfig.version.unique_id48
            or saved.get("user_version") != config.user_version or saved.get("model_config") != asdict(config)
            or saved.get("cost_treatment") != "inside_utility"):
        raise ValueError("time-only checkpoint geometry, utility or cost placement differs")
    return saved


def load_time_only_model(path: Path, *, config: TimeOnlyConfig, env: BaseEnvironment) -> TimeOnlyTriple:
    """Read the retained policy without training or mutating its checkpoint."""
    saved = load_time_only_checkpoint(path, config=config)
    model = TimeOnlyTriple(env, config=config)
    model.load_state_dict(saved["best_state"])
    model.requires_grad_(False)
    return model


def train_time_only(*, config: TimeOnlyConfig, train: TrainConfig, exposures: np.ndarray,
                    payoff: np.ndarray, output: Path, monitor: AuditMonitor) -> TimeOnlyTriple:
    """One full-data gradient update per epoch, with no shuffle or minibatch noise."""
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    model = TimeOnlyTriple(env, config=config)
    x, h = torch.as_tensor(exposures, dtype=env.fdtype), torch.as_tensor(payoff, dtype=env.fdtype)
    checkpoint = output / "time-only.checkpoint.pt"
    saved = load_time_only_checkpoint(checkpoint, config=config) if checkpoint.exists() else None
    with torch.no_grad():
        for label, sign in zip(LABELS, SIGNS, strict=True):
            getattr(model, label).translation.fill_(optimize_translation(sign * payoff, config.risk_aversion))
    optimizer = torch.optim.Adam([dict(params=list(getattr(model, k).parameters()), name=k) for k in LABELS],
                                 lr=train.learning_rate)
    start = 0
    if saved:
        if saved["training_paths"] != len(payoff):
            raise ValueError("checkpoint belongs to a different training population")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = train.learning_rate
        start = saved["epoch"] + 1
        torch.set_rng_state(saved["torch_rng"])
    execute = torch.compile(model, fullgraph=True) if train.compile_model else model

    def snapshot():
        return {k:v.detach().clone() for k,v in model.state_dict().items()}

    def record(epoch, values, online):
        return dict(epoch=epoch, **{f"{k}_loss":float(v) for k,v in zip(LABELS, values, strict=True)},
                    sum_loss=float(values.sum()), online_sum_loss=float(online), learning_rate=train.learning_rate)

    with torch.no_grad():
        initial = execute(x, h).numpy()
    if not np.isfinite(initial).all():
        raise FloatingPointError("nonfinite initial time-only OCE")
    history = saved["history"] if saved else [record(0, initial, initial.sum())]
    best_loss = saved["best_loss"] if saved else float(initial.sum())
    best_state = saved["best_state"] if saved else snapshot()
    selected = saved["selected_epoch"] if saved else 0
    print(json.dumps(dict(stage="time-only-initial", **history[0], parameters=sum(p.numel() for p in model.parameters()),
        full_batch_paths=len(payoff), optimizer_updates_per_epoch=1, cost_treatment="inside_utility")), flush=True)
    started = time.monotonic()
    for epoch in range(start, train.epochs):
        monitor.stage("train-time-only-oce", epoch, train.epochs)
        optimizer.zero_grad(set_to_none=True)
        losses = execute(x, h)
        losses.sum().backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("nonfinite time-only gradient; prior checkpoint retained")
        optimizer.step()
        with torch.no_grad():
            values = execute(x, h).numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite time-only loss; prior checkpoint retained")
        if values.sum() < best_loss:
            best_loss, best_state, selected = float(values.sum()), snapshot(), epoch + 1
        history.append(record(epoch + 1, values, float(losses.detach().sum())))
        payload = dict(model_config=asdict(config), model_code_version=TimeOnlyConfig.version.unique_id48,
            user_version=config.user_version, train_config=asdict(train), cost_treatment="inside_utility",
            epoch=epoch, state_dict=model.state_dict(), best_state=best_state, best_loss=best_loss,
            selected_epoch=selected, optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            history=history, training_paths=len(payoff), sample_order="unchanged complete full batch; no RNG sampling")
        temporary = checkpoint.with_suffix(".tmp.pt")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        temporary_csv = output / "triple-losses.tmp.csv"
        with temporary_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        temporary_csv.replace(output / "triple-losses.csv")
        elapsed = time.monotonic() - started
        print(json.dumps(dict(stage="time-only-epoch", **history[-1], selected_epoch=selected,
            best_sum_loss=best_loss, completed=epoch+1-start, total=train.epochs-start,
            elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed*(train.epochs-epoch-1)/(epoch+1-start))), flush=True)
    model.load_state_dict(best_state)
    model.fit_summary = dict(initial=history[0], selected=history[selected], latest=history[-1],
        selected_epoch=selected, epochs_run=train.epochs, selection="minimum full-training summed negative net OCE")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    output, scenarios = args.output.resolve(), args.scenarios.resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("use a unique external run directory")
    if quantdata.env.test_mode or not Path(quantdata.env.locations.training.path).resolve().is_relative_to(root):
        raise RuntimeError("explicit environment did not discover the external project data root")
    checks = resource_health()
    if any(c["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for c in checks):
        raise RuntimeError("resource manager blocks launch")
    if any(c["health"] != "OK" for c in checks):
        print(json.dumps(dict(stage="resource-warning", checks=checks)), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(stage="environment-validated", interpreter=os.sys.executable, data_root=str(root),
        working_directory=str(Path.cwd()), device="cpu", precision=64, policy="deterministic affine time only")), flush=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    try:
        with FileLock(output / "training.lock", timeout=0):
            torch.set_num_threads(4)
            config = TimeOnlyConfig(label="time-only-empty-long-short")
            monitor.stage("load-existing-training-returns")
            exposures, payoff, data = load_training_exposures(scenarios, config=config)
            train = TrainConfig(epochs=args.epochs, batch_size=len(payoff), learning_rate=args.learning_rate,
                patience=0, gradient_clip=10., compile_model=args.compile)
            identity = dict(objective="oce", policy_kind="time-only affine", cost_treatment="inside_utility",
                epochs=args.epochs, model_config=asdict(config), model_code_version=TimeOnlyConfig.version.unique_id48,
                train_config=asdict(train), scenarios=str(scenarios), data=data, validation_used=False,
                initialization="all action coefficients zero; OCE cash initialized to unhedged training optimum",
                optimizer="full-batch Adam, constant LR, no gradient clipping, one update per epoch, no sampling",
                action="b+w*(day/20)+v0*1(day=0)+v10*1(day=10); 21-dimensional floating-contract quantity vector",
                objective_formula="SUM_s[-c_s-mean(u(s*H+G_s-C_s-c_s))], s=0,+1,-1; no external penalty",
                code_sha256={str(p):sha256(p) for p in (Path(__file__), Path(__file__).with_name("time_only.py"))})
            manifest = output / "identity.json"
            if manifest.exists() and json.loads(manifest.read_text()) != identity:
                raise ValueError("time-only run identity changed; use a new directory")
            atomic_json(manifest, identity)
            model = train_time_only(config=config, train=train, exposures=exposures, payoff=payoff, output=output, monitor=monitor)
            with torch.no_grad():
                gains = model.gains(torch.as_tensor(exposures)).numpy()
                costs, cash = model.costs().numpy(), model.cash().numpy()
            utilities, statistics = {}, {}
            for i, (label, sign) in enumerate(zip(LABELS, SIGNS, strict=True)):
                net = gains[:,i] + sign*payoff - costs[i]
                value = _economic_utility(net, cash[i], config.risk_aversion)
                profiled_cash = optimize_translation(net, config.risk_aversion)
                utilities[label] = value
                statistics[label] = dict(net_oce=pair_summary(value), cost=float(costs[i]), translation=float(cash[i]),
                    mean_gain=pair_summary(gains[:,i]),
                    net_oce_after_scalar_optimization=pair_summary(_economic_utility(net,profiled_cash,config.risk_aversion)),
                    optimized_scalar_for_fixed_actions=profiled_cash)
            atomic_json(output / "report.json", dict(status="complete", fit=model.fit_summary,
                training_payoff_expectation=pair_summary(payoff), statistics=statistics,
                bid=pair_summary(utilities["long"]-utilities["empty"]),
                ask=pair_summary(utilities["empty"]-utilities["short"]),
                interpretation="training-only OCEs of terminal P&L NET of costs; trained cash parameters; not validation"))
            np.savez(output / "selected-actions.npz", **{label:getattr(model,label).actions().detach().numpy() for label in LABELS})
            plot_triple_history(output, output / "figures")
            monitor.stage("complete", args.epochs, args.epochs)
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
