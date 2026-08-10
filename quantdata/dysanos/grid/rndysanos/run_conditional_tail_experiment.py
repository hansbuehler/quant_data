"""Run a full raw-dC conditional-tail experiment on one simulated universe."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from quantdata.torch import BaseEnvironment, DeviceLock
from quantdata.torch.ffw import FeedForwardConfig

from .conditional_tail import ConditionalTailConfig
from .conditional_tail_experiment import (
    ConditionalTailTrainConfig,
    prepare_raw_option_tail_data,
    process_memory,
    split_tail_data,
    train_conditional_tail,
)
from .torch_prep import RNDysanosExperiment, prep_experiment


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dynamo_counters() -> dict[str, int]:
    try:
        from torch._dynamo.utils import counters

        return {
            "frames_total": int(counters["frames"]["total"]),
            "frames_ok": int(counters["frames"]["ok"]),
            "unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": int(sum(counters["graph_break"].values())),
        }
    except (ImportError, KeyError, TypeError):
        return {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.report)
    experiment = RNDysanosExperiment(
        num_samples=args.paths,
        pcnt_validation=0.5,
        cnt_universe=1,
        universe_seed_offset=args.universe_seed_offset,
    )
    preparation_started = time.monotonic()
    universe = prep_experiment(experiment, plot_sampler=False)[0]
    preparation_seconds = time.monotonic() - preparation_started
    report: dict[str, Any] = {
        "paths": args.paths,
        "training_paths": args.paths // 2,
        "validation_paths": args.paths - args.paths // 2,
        "validation_fraction": 0.5,
        "universe_seed_offset": args.universe_seed_offset,
        "epochs_requested": args.epochs,
        "minimum_epochs": args.minimum_epochs,
        "tail_path_counts": [1.0, 2.0, 10.0],
        "return": "raw option dC=C_tp1-C_t",
        "preparation_seconds": preparation_seconds,
        "status": "prepared",
    }
    _atomic_json(report_path, report)

    with DeviceLock() as device_lock:
        env = BaseEnvironment(device=device_lock.device, debug=False)
        if env.device.type == "cuda":
            torch.cuda.set_device(env.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        data, geometry = prepare_raw_option_tail_data(universe, env=env)
        training, validation = split_tail_data(data, validation_fraction=0.5)
        del data, universe
        gc.collect()
        config = ConditionalTailConfig(
            label=f"raw-dc-tail-{args.paths}-seed-{args.seed}",
            group=f"validation/raw-dc/{args.paths}-paths",
            user_version="1.0",
            state_features=geometry.state_features,
            instruments=geometry.expiries * geometry.strikes,
            training_paths=training["z"].shape[0],
            tail_path_counts=(1.0, 2.0, 10.0),
            seed=args.seed,
            network_config=FeedForwardConfig(
                num_depth=2,
                num_width=args.width,
                activation="LeakyReLU",
                use_bias=True,
            ),
        )
        train_config = ConditionalTailTrainConfig(
            batch_size=args.batch_size,
            valuation_batch_size=args.valuation_batch_size,
            lr=args.lr,
            final_lr=args.final_lr,
            weight_decay=args.weight_decay,
            compile=not args.no_compile,
            value_update=args.value_update,
            checkpoint_update=args.checkpoint_update,
            minimum_epochs=args.minimum_epochs,
            convergence_points=args.convergence_points,
            convergence_relative_tolerance=args.convergence_relative_tolerance,
            convergence_log_coverage_tolerance=args.convergence_log_coverage_tolerance,
            checkpoint_root=args.checkpoint_root,
        )
        report.update(
            device=str(env.device),
            geometry=asdict(geometry),
            model_config=asdict(config),
            train_config=asdict(train_config),
            memory_before_training=process_memory(env.device),
            status="training",
        )
        _atomic_json(report_path, report)
        before_compile = _dynamo_counters()
        training_started = time.monotonic()

        def progress(
            epoch: int,
            training_value: dict[str, Any],
            validation_value: dict[str, Any],
            convergence: dict[str, Any],
        ) -> None:
            report["progress"] = {
                "epoch": epoch,
                "training": training_value,
                "validation": validation_value,
                "convergence": convergence,
                "elapsed_seconds": time.monotonic() - training_started,
                "memory": process_memory(env.device),
            }
            _atomic_json(report_path, report)

        result = train_conditional_tail(
            training,
            validation,
            config=config,
            train_config=train_config,
            epochs=args.epochs,
            env=env,
            restart=args.restart,
            progress=progress,
        )
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        after_compile = _dynamo_counters()
        report.update(
            status="complete",
            training_seconds=time.monotonic() - training_started,
            checkpoint=str(result.checkpoint_path),
            restarted=result.restarted,
            epochs_completed=(result.tracker.epochs[-1] if result.tracker.epochs else 0),
            converged_epoch=result.tracker.converged_epoch,
            compiled_graphs={
                name: value - before_compile.get(name, 0)
                for name, value in after_compile.items()
            },
            training=result.training,
            validation=result.validation,
            tracker=asdict(result.tracker),
            memory_after_training=process_memory(env.device),
        )
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, required=True)
    parser.add_argument("--universe-seed-offset", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200_000)
    parser.add_argument("--minimum-epochs", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--valuation-batch-size", type=int, default=2_048)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--final-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--value-update", type=int, default=5_000)
    parser.add_argument("--checkpoint-update", type=int, default=10_000)
    parser.add_argument("--convergence-points", type=int, default=5)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--convergence-log-coverage-tolerance", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=84_771)
    parser.add_argument("--checkpoint-root", default="~/.cache/rndysanos/conditional-tail")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in (
        "paths",
        "epochs",
        "minimum_epochs",
        "batch_size",
        "valuation_batch_size",
        "width",
        "value_update",
        "checkpoint_update",
        "convergence_points",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.paths % 2:
        raise ValueError("--paths must be even for the required 50/50 split")
    report = run(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
