"""Compare simplified and global-offset power-exponential path OCE.

The simulated paths come from the exact Gaussian spot-tail proposal.  Every
training objective, density calibration, validation statistic, and martingale
error is likelihood weighted back to the fitted physical measure P.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from quantdata.dysanos.grid.gendysanos.torch_prep import (
    GenDysanosExperiment,
    prepare_terminal_path_experiment,
)
from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig
from quantdata.torch import BaseEnvironment, DeviceLock

from .run_experiment import (
    _atomic_json,
    _baseline_split,
    _counter_difference,
    _dynamo_counters,
    _stage_report,
)
from .run_v2_mask_comparison import (
    _attach_universe,
    _cleanup_shared_memory,
    _share_universe,
)
from .torch_drift_change import DysanosConfig, TrainConfig, train

_VARIANTS = ("simplified", "global_y")


def _job_args(
    args: argparse.Namespace,
    *,
    paths: int,
    variant: str,
) -> argparse.Namespace:
    name = f"{paths}-path-powexp-{variant}"
    return argparse.Namespace(
        paths=paths,
        variant=variant,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        batch_size=(
            args.million_batch_size
            if paths >= 1_000_000
            else args.batch_size
        ),
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        value_update=args.value_update,
        value_seconds=args.value_seconds,
        checkpoint_update=args.checkpoint_update,
        checkpoint_seconds=args.checkpoint_seconds,
        convergence_points=args.convergence_points,
        convergence_relative_tolerance=args.convergence_relative_tolerance,
        cost_prop=args.cost_prop,
        seed=args.seed + paths,
        compile=not args.no_compile,
        restart=args.restart,
        checkpoint_root=str(Path(args.checkpoint_root) / name),
        report=str(Path(args.output_root) / f"{name}.json"),
    )


def _run_job(
    universe: dict[str, Any],
    sampling_diagnostics: dict[str, Any],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    env = BaseEnvironment(precision=32, device=device, debug=False)
    torch.empty((), dtype=env.fdtype, device=env.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(env.device)
    config = DysanosConfig(
        label=(
            f"weighted-path-powexp-{args.variant}-{args.paths}-{args.seed}"
        ),
        group=(
            f"gendysanos/importance-path/{args.paths}/seed-{args.seed}/"
            f"powexp-{args.variant}"
        ),
        user_version="3.0",
        utility="powexp",
        lmbda=1.0,
        oce_variant=args.variant,
        action_type="network",
        state_mode="raw",
        delta_hedged=True,
        cash_returns=True,
        learn_cost=False,
        cost_cond=0.0,
        cost_uncond=0.0,
        cost_prop=args.cost_prop,
    )
    train_config = TrainConfig(
        trn_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        validation_fraction=0.5,
        optimizer="AdamW",
        lr=args.lr,
        final_lr=args.lr,
        weight_decay=0.0,
        compile=args.compile,
        value_seconds=args.value_seconds,
        value_update=args.value_update,
        checkpoint_seconds=args.checkpoint_seconds,
        checkpoint_update=args.checkpoint_update,
        heavy_output_update=args.epochs,
        early_stopping=True,
        minimum_epochs=args.minimum_epochs,
        convergence_points=args.convergence_points,
        convergence_relative_tolerance=(
            args.convergence_relative_tolerance
        ),
        require_cost_band=False,
        checkpoint_root=args.checkpoint_root,
    )
    before_compile = _dynamo_counters()
    started = time.monotonic()
    result = train(
        universe,
        config=config,
        train_config=train_config,
        epochs=args.epochs,
        moment_epochs=1,
        restart=args.restart,
        env=env,
    )
    torch.cuda.synchronize(env.device)
    stage = _stage_report(
        result=result,
        elapsed_seconds=time.monotonic() - started,
        device=env.device,
        compilation=_counter_difference(_dynamo_counters(), before_compile),
    )
    report = {
        "status": "complete",
        "experiment": "likelihood-weighted daily full-path powerexp OCE",
        "paths": args.paths,
        "training_paths": args.paths // 2,
        "validation_paths": args.paths // 2,
        "oce_variant": args.variant,
        "objective": (
            "sup_a E_P[u(Pi)]"
            if args.variant == "simplified"
            else "sup_{a,y} E_P[u(y+Pi)-y] with one global y"
        ),
        "return_semantics": (
            "expiry-matched delta-hedged cash P&L divided by gross "
            "option-plus-hedge notional; spot return divided by spot"
        ),
        "cost_semantics": (
            f"proportional rate {args.cost_prop:g} applied to every gross "
            "cash-normalized traded notional"
        ),
        "importance_semantics": (
            "proposal paths are weighted by p/q in optimization, density "
            "normalization, validation, martingale means, and MC errors"
        ),
        "sampling_diagnostics": sampling_diagnostics,
        "model_config": asdict(config),
        "train_config": asdict(train_config),
        "device": str(device),
        "baseline": {
            "training": _baseline_split(result.training),
            "validation": _baseline_split(result.validation),
        },
        "stage": stage,
    }
    _atomic_json(Path(args.report), report)
    return report


def _worker(
    descriptors: dict[str, dict[str, Any]],
    sampling_diagnostics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    # Reserve before CPU/GPU preparation so concurrent independent jobs cannot
    # finish preprocessing at different times and collapse onto one device.
    with DeviceLock() as device_lock:
        universe, handles = _attach_universe(descriptors)
        try:
            _run_job(
                universe,
                sampling_diagnostics,
                args,
                device=device_lock.device,
            )
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            for handle in handles:
                handle.close()


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    stage = report["stage"]
    splits = stage["splits"]
    return {
        "device": report["device"],
        "oce_variant": report["oce_variant"],
        "convergence": stage["convergence"],
        "memory": stage["memory"],
        "compiled_graphs": stage["compiled_graphs"],
        "training": {
            "loss": splits["training"]["loss"],
            "cash_offset": splits["training"]["cash_offset"],
            "cash_offset_first_order_residual": splits["training"][
                "cash_offset_first_order_residual"
            ],
            "objective_gradient": splits["training"]["objective_gradient"],
            "martingale": splits["training"]["martingale"],
        },
        "validation": {
            "loss": splits["validation"]["loss"],
            "cash_offset": splits["validation"]["cash_offset"],
            "cash_offset_first_order_residual": splits["validation"][
                "cash_offset_first_order_residual"
            ],
            "objective_gradient": splits["validation"]["objective_gradient"],
            "martingale": splits["validation"]["martingale"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    comparison_path = output_root / "weighted-path-oce-comparison.json"
    path_counts = tuple(args.path_counts)
    if any(paths <= 0 or paths % 2 for paths in path_counts):
        raise ValueError("path counts must be positive and even")
    report: dict[str, Any] = {
        "status": "starting",
        "path_counts": list(path_counts),
        "execution_order": list(path_counts),
        "variants": list(_VARIANTS),
        "epochs": args.epochs,
        "minimum_epochs": args.minimum_epochs,
        "jobs": {},
    }
    _atomic_json(comparison_path, report)

    for paths in path_counts:
        report["status"] = f"generating_{paths}"
        _atomic_json(comparison_path, report)
        generation_started = time.monotonic()
        universe, diagnostics = prepare_terminal_path_experiment(
            GenDysanosExperiment(
                num_samples=paths,
                mode="path",
                num_steps=args.trading_state_steps,
                num_alpha=None,
                use_matrix_exp=True,
                bandwidth=1.0,
                seed=args.seed + paths,
                block_size=args.generator_block_size,
                importance_sampling=GaussianSpotImportanceConfig(
                    spot_std_multiplier=args.importance_spot_std_multiplier,
                    base_probability=args.importance_base_probability,
                    target_action_count=args.importance_target_action_count,
                ),
            )
        )
        descriptors, handles, shared_bytes = _share_universe(universe)
        del universe
        gc.collect()
        atexit.register(_cleanup_shared_memory, handles)
        jobs = [
            _job_args(args, paths=paths, variant=variant)
            for variant in _VARIANTS
        ]
        report["jobs"].update(
            {f"{paths}-{job.variant}": job.report for job in jobs}
        )
        report.setdefault("preparation", {})[str(paths)] = {
            "seconds": time.monotonic() - generation_started,
            "shared_memory_bytes": shared_bytes,
            "sampling_diagnostics": asdict(diagnostics),
        }
        report["status"] = f"training_{paths}"
        _atomic_json(comparison_path, report)

        context = mp.get_context("spawn")
        processes = [
            context.Process(
                target=_worker,
                args=(descriptors, asdict(diagnostics), job),
                name=f"weighted-oce-{paths}-{job.variant}",
            )
            for job in jobs
        ]
        try:
            for process in processes:
                process.start()
            while any(process.is_alive() for process in processes):
                report["active_jobs"] = [
                    process.name for process in processes if process.is_alive()
                ]
                _atomic_json(comparison_path, report)
                time.sleep(5.0)
            for process in processes:
                process.join()
            exitcodes = {
                process.name: process.exitcode for process in processes
            }
            report.setdefault("exitcodes", {}).update(exitcodes)
            if any(code != 0 for code in exitcodes.values()):
                report.update(status="failed", active_jobs=[])
                _atomic_json(comparison_path, report)
                raise RuntimeError(
                    f"weighted path-OCE workers failed: {exitcodes}"
                )
            outputs = {
                job.variant: json.loads(
                    Path(job.report).read_text(encoding="utf-8")
                )
                for job in jobs
            }
            report.setdefault("summaries", {})[str(paths)] = {
                variant: _compact(outputs[variant])
                for variant in _VARIANTS
            }
            report.update(status=f"complete_{paths}", active_jobs=[])
            _atomic_json(comparison_path, report)
        finally:
            try:
                atexit.unregister(_cleanup_shared_memory)
            except ValueError:
                pass
            _cleanup_shared_memory(handles)

    report["status"] = "complete"
    _atomic_json(comparison_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-counts", type=int, nargs="+", default=[100_000, 1_000_000]
    )
    parser.add_argument("--epochs", type=int, default=200_000)
    parser.add_argument("--minimum-epochs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=600_000)
    parser.add_argument("--trading-state-steps", type=int, default=10)
    parser.add_argument("--generator-block-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--million-batch-size", type=int, default=8_192)
    parser.add_argument("--val-batch-size", type=int, default=8_192)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--cost-prop", type=float, default=1.0e-4)
    parser.add_argument("--value-update", type=int, default=2_000)
    parser.add_argument("--value-seconds", type=float, default=1_800.0)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--checkpoint-seconds", type=float, default=1_800.0)
    parser.add_argument("--convergence-points", type=int, default=7)
    parser.add_argument(
        "--convergence-relative-tolerance", type=float, default=1.0e-3
    )
    parser.add_argument("--importance-spot-std-multiplier", type=float, default=8.0)
    parser.add_argument("--importance-base-probability", type=float, default=0.1)
    parser.add_argument("--importance-target-action-count", type=int, default=3)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--checkpoint-root", default="~/.cache/rndysanos/weighted-path-oce"
    )
    parser.add_argument(
        "--output-root",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "weighted-path-oce-powerexp"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparison": str(
                    Path(args.output_root) / "weighted-path-oce-comparison.json"
                ),
                "summaries": report.get("summaries", {}),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
