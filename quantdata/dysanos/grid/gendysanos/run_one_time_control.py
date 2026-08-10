"""Run an OCE control on coherent SANOS state/spot terminal paths."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from quantdata.dysanos.grid.rndysanos.run_experiment import (
    _baseline_split,
    _counter_difference,
    _dynamo_counters,
    _stage_report,
)
from quantdata.dysanos.grid.rndysanos.torch_drift_change import (
    DysanosConfig,
    TrainConfig,
    train,
)
from quantdata.torch import BaseEnvironment, DeviceLock

from .terminal_universe import TerminalPathDiagnostics
from .torch_prep import (
    GenDysanosExperiment,
    prepare_terminal_path_experiment,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    args: argparse.Namespace,
    *,
    prepared_universe: dict | None = None,
    sampling_diagnostics: TerminalPathDiagnostics | None = None,
) -> dict:
    """Generate the control universe, fit one OCE model, and report diagnostics."""

    if prepared_universe is None:
        preparation_started = time.monotonic()
        universe, sampling = prepare_terminal_path_experiment(
            GenDysanosExperiment(
                num_samples=args.paths,
                mode=args.universe_mode,
                num_steps=args.trading_state_steps,
                num_alpha=args.num_alpha,
                use_matrix_exp=args.use_matrix_exp,
                bandwidth=args.bandwidth,
                seed=args.seed,
                block_size=args.generator_block_size,
            )
        )
        preparation_seconds = time.monotonic() - preparation_started
    else:
        if sampling_diagnostics is None:
            raise ValueError(
                "'sampling_diagnostics' is required with a prepared universe"
            )
        universe = prepared_universe
        sampling = sampling_diagnostics
        preparation_seconds = 0.0

    with DeviceLock(fall_back=True) as device_lock:
        env = BaseEnvironment(
            precision=args.precision,
            device=device_lock.device,
            debug=args.debug,
        )
        if env.device.type == "cuda":
            # Some CUDA runtimes do not create per-device memory counters until
            # the first allocation, in which case an early reset is rejected.
            torch.empty((), dtype=env.fdtype, device=env.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(env.device)
        model_config = DysanosConfig(
            label=(
                f"gendysanos-{args.universe_mode}-{args.utility}-{args.action_type}-"
                f"{args.state_mode}-{args.paths}-{args.seed}"
            ),
            group=(
                f"gendysanos/{args.universe_mode}/{args.paths}/seed-{args.seed}/"
                f"{args.utility}/{args.action_type}-{args.state_mode}"
            ),
            user_version="2.0",
            utility=args.utility,
            lmbda=args.lmbda,
            action_type=args.action_type,
            state_mode=args.state_mode,
            delta_hedged=False,
            cash_returns=True,
            learn_cost=False,
            cost_cond=0.0,
            cost_uncond=0.0,
            cost_prop=args.cost_prop,
        )
        optimizer = "LBFGS" if args.action_type == "deterministic" else "AdamW"
        train_config = TrainConfig(
            trn_batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            validation_fraction=0.5,
            optimizer=optimizer,
            lr=args.lr,
            final_lr=args.lr,
            weight_decay=0.0,
            compile=args.compile,
            value_seconds=60.0,
            value_update=max(1, args.epochs // 10),
            checkpoint_seconds=120.0,
            checkpoint_update=max(1, args.epochs // 5),
            heavy_output_update=max(1, args.epochs),
            early_stopping=getattr(args, "early_stopping", False),
            minimum_epochs=getattr(args, "minimum_epochs", 0),
            require_cost_band=False,
            checkpoint_root=args.checkpoint_root,
        )
        before_compile = _dynamo_counters()
        training_started = time.monotonic()
        result = train(
            universe,
            config=model_config,
            train_config=train_config,
            epochs=args.epochs,
            moment_epochs=1,
            restart=args.restart,
            env=env,
        )
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        training_seconds = time.monotonic() - training_started
        stage = _stage_report(
            result=result,
            elapsed_seconds=training_seconds,
            device=env.device,
            compilation=_counter_difference(_dynamo_counters(), before_compile),
        )
        report = {
            "experiment": (
                f"{args.universe_mode} coherent-state-path SANOS terminal-payoff "
                "OCE control"
            ),
            "universe_mode": args.universe_mode,
            "paths": args.paths,
            "training_paths": args.paths // 2,
            "validation_paths": args.paths - args.paths // 2,
            "seed": args.seed,
            "epochs": args.epochs,
            "action_type": args.action_type,
            "state_mode": args.state_mode,
            "utility": args.utility,
            "lambda": args.lmbda,
            "cost_prop": args.cost_prop,
            "optimizer": optimizer,
            "device": str(env.device),
            "precision": args.precision,
            "preparation_seconds": preparation_seconds,
            "simulator": {
                "num_alpha": args.num_alpha,
                "use_matrix_exp": args.use_matrix_exp,
                "bandwidth": args.bandwidth,
                "trading_state_steps": args.trading_state_steps,
            },
            "sampling_diagnostics": asdict(sampling),
            "baseline": {
                "training": _baseline_split(result.training),
                "validation": _baseline_split(result.validation),
            },
            "stage": stage,
        }
    _atomic_json(Path(args.report), report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--paths", type=int, default=1_000)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--seed", type=int, default=500_000)
    result.add_argument("--generator-block-size", type=int, default=25_000)
    result.add_argument(
        "--universe-mode",
        choices=("one-step", "one-step-weekly", "path", "weekly"),
        default="one-step",
    )
    result.add_argument("--trading-state-steps", type=int, default=10)
    result.add_argument(
        "--num-alpha",
        type=lambda value: None if value.lower() == "none" else int(value),
        default=None,
    )
    result.add_argument(
        "--use-matrix-exp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument("--bandwidth", type=float, default=1.0)
    result.add_argument(
        "--action-type", choices=("deterministic", "network"), default="deterministic"
    )
    result.add_argument("--state-mode", choices=("raw", "full"), default="raw")
    result.add_argument(
        "--utility", choices=("entropy", "powexp"), default="entropy"
    )
    result.add_argument("--lmbda", type=float, default=1.0)
    result.add_argument("--cost-prop", type=float, default=0.0)
    result.add_argument("--batch-size", type=int, default=1_024)
    result.add_argument("--val-batch-size", type=int, default=4_096)
    result.add_argument("--lr", type=float, default=1.0)
    result.add_argument("--precision", type=int, choices=(32, 64), default=64)
    result.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--restart", action="store_true")
    result.add_argument("--early-stopping", action="store_true")
    result.add_argument("--minimum-epochs", type=int, default=0)
    result.add_argument("--debug", action="store_true")
    result.add_argument(
        "--checkpoint-root", default="~/.cache/rndysanos/oce-gendysanos-control"
    )
    result.add_argument(
        "--report",
        default=(
            "quantdata/dysanos/grid/gendysanos/reports/"
            "one-time-control-smoke.json"
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    compact = {
        "report": args.report,
        "device": report["device"],
        "sampling_diagnostics": report["sampling_diagnostics"],
        "baseline_training": report["baseline"]["training"]["martingale"],
        "baseline_validation": report["baseline"]["validation"]["martingale"],
        "training": report["stage"]["splits"]["training"]["martingale"],
        "validation": report["stage"]["splits"]["validation"]["martingale"],
        "objective_gradient": report["stage"]["splits"]["training"][
            "objective_gradient"
        ],
        "memory": report["stage"]["memory"],
        "compiled_graphs": report["stage"]["compiled_graphs"],
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
