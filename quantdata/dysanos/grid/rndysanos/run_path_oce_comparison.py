"""Run four complete-path OCE comparisons on one shared million-path universe."""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any

from .run_experiment import _atomic_json, run_experiment
from .run_v2_mask_comparison import (
    _attach_universe,
    _cleanup_shared_memory,
    _share_universe,
)
from .torch_prep import RNDysanosExperiment, prep_experiment


def _worker(descriptors: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    universe, handles = _attach_universe(descriptors)
    try:
        run_experiment(args, prepared_universes=[universe])
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.close()


def _job_args(
    args: argparse.Namespace,
    *,
    name: str,
    regime: str,
    utility: str,
    lmbda: float,
    masked: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        paths=args.paths,
        epochs=args.epochs,
        moment_epochs=args.moment_epochs,
        stages=1,
        validation_fraction=0.5,
        universe_seed_offset=args.universe_seed_offset,
        independent_split_universes=False,
        same_universe=True,
        regimes=regime,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        final_lr=args.final_lr,
        moment_lr=args.moment_lr,
        moment_final_lr=args.moment_final_lr,
        action_initialization_epochs=0,
        action_initialization_lr=1.0e-3,
        utility=utility,
        lmbda=lmbda,
        weight_decay=0.0,
        minimum_epochs=args.minimum_epochs,
        minimum_moment_epochs=args.minimum_moment_epochs,
        early_stopping=True,
        convergence_points=args.convergence_points,
        convergence_relative_tolerance=args.convergence_relative_tolerance,
        cost_band_tolerance=args.cost_band_tolerance,
        minimum_unsigned_improvement=0.0,
        continue_after_failed_gate=False,
        value_seconds=args.value_seconds,
        value_update=args.value_update,
        checkpoint_seconds=args.checkpoint_seconds,
        checkpoint_update=args.checkpoint_update,
        checkpoint_root=args.checkpoint_root,
        arb_mask_report=args.arb_mask_report if masked else None,
        report=str(Path(args.output_root) / f"{name}.json"),
        restart=args.restart,
        no_compile=args.no_compile,
    )


def _final_summary(output: dict[str, Any], regime: str) -> dict[str, Any]:
    regime_output = output["regimes"][regime]
    stage = regime_output["stages"][-1]
    return {
        "device": output["device"],
        "utility": output["utility"],
        "lmbda": output["lmbda"],
        "strategy_mask": output["strategy_mask"],
        "compiled_graphs": stage["compiled_graphs"],
        "elapsed_seconds": stage["elapsed_seconds"],
        "memory": stage["memory"],
        "convergence": stage["convergence"],
        "continuation_gate": stage["continuation_gate"],
        "training": {
            "loss": stage["splits"]["training"]["loss"],
            "objective_gradient": stage["splits"]["training"]["objective_gradient"],
            "density_selection": stage["splits"]["training"]["density_selection"],
            "martingale": stage["splits"]["training"]["martingale"],
        },
        "validation": {
            "loss": stage["splits"]["validation"]["loss"],
            "objective_gradient": stage["splits"]["validation"]["objective_gradient"],
            "density_selection": stage["splits"]["validation"]["density_selection"],
            "martingale": stage["splits"]["validation"]["martingale"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare once, run four GPU workers, and write a compact comparison."""

    output_root = Path(args.output_root)
    comparison_path = output_root / "path-oce-comparison.json"
    jobs = [
        (
            "powexp-unconditional-unmasked",
            "unconditional",
            _job_args(
                args,
                name="powexp-unconditional-unmasked",
                regime="unconditional",
                utility="powexp",
                lmbda=1.0,
                masked=False,
            ),
        ),
        (
            "powexp-unconditional-masked",
            "unconditional",
            _job_args(
                args,
                name="powexp-unconditional-masked",
                regime="unconditional",
                utility="powexp",
                lmbda=1.0,
                masked=True,
            ),
        ),
        (
            "powexp-conditional-unmasked",
            "conditional",
            _job_args(
                args,
                name="powexp-conditional-unmasked",
                regime="conditional",
                utility="powexp",
                lmbda=1.0,
                masked=False,
            ),
        ),
        (
            "cvar90-unconditional-unmasked",
            "unconditional",
            _job_args(
                args,
                name="cvar90-unconditional-unmasked",
                regime="unconditional",
                utility="cvar",
                lmbda=1.0 / 9.0,
                masked=False,
            ),
        ),
    ]
    report: dict[str, Any] = {
        "status": "preparing_shared_universe",
        "paths": args.paths,
        "training_paths": args.paths // 2,
        "validation_paths": args.paths // 2,
        "epochs": args.epochs,
        "moment_epochs": args.moment_epochs,
        "universe_seed_offset": args.universe_seed_offset,
        "jobs": {name: job.report for name, _, job in jobs},
    }
    _atomic_json(comparison_path, report)

    started = time.monotonic()
    experiment = RNDysanosExperiment(
        num_samples=args.paths,
        pcnt_validation=0.5,
        cnt_universe=1,
        universe_seed_offset=args.universe_seed_offset,
    )
    universe = prep_experiment(experiment, plot_sampler=False)[0]
    descriptors, handles, shared_bytes = _share_universe(universe)
    del universe
    gc.collect()
    atexit.register(_cleanup_shared_memory, handles)
    report.update(
        status="running_four_jobs",
        preparation_seconds=time.monotonic() - started,
        shared_memory_bytes=shared_bytes,
        shared_memory_segments=len(handles),
    )
    _atomic_json(comparison_path, report)

    context = mp.get_context("spawn")
    processes = [
        (
            name,
            context.Process(target=_worker, args=(descriptors, job), name=name),
        )
        for name, _, job in jobs
    ]
    worker_started = time.monotonic()
    try:
        for _, process in processes:
            process.start()
        for _, process in processes:
            process.join()
        exitcodes = {name: process.exitcode for name, process in processes}
        if any(code != 0 for code in exitcodes.values()):
            report.update(status="failed", exitcodes=exitcodes)
            _atomic_json(comparison_path, report)
            raise RuntimeError(f"one or more path-OCE workers failed: {exitcodes}")
        outputs = {
            name: json.loads(Path(job.report).read_text(encoding="utf-8"))
            for name, _, job in jobs
        }
        report.update(
            status="complete",
            worker_elapsed_seconds=time.monotonic() - worker_started,
            exitcodes=exitcodes,
            summaries={
                name: _final_summary(outputs[name], regime)
                for name, regime, _ in jobs
            },
        )
        _atomic_json(comparison_path, report)
        return report
    finally:
        atexit.unregister(_cleanup_shared_memory)
        _cleanup_shared_memory(handles)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=200_000)
    parser.add_argument("--moment-epochs", type=int, default=200_000)
    parser.add_argument("--minimum-epochs", type=int, default=100_000)
    parser.add_argument("--minimum-moment-epochs", type=int, default=100_000)
    parser.add_argument("--universe-seed-offset", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--val-batch-size", type=int, default=4_096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--final-lr", type=float, default=1.0e-3)
    parser.add_argument("--moment-lr", type=float, default=1.0e-3)
    parser.add_argument("--moment-final-lr", type=float, default=1.0e-6)
    parser.add_argument("--value-update", type=int, default=10_000)
    parser.add_argument("--value-seconds", type=float, default=3_600.0)
    parser.add_argument("--checkpoint-update", type=int, default=10_000)
    parser.add_argument("--checkpoint-seconds", type=float, default=3_600.0)
    parser.add_argument("--convergence-points", type=int, default=5)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--cost-band-tolerance", type=float, default=0.0)
    parser.add_argument("--checkpoint-root", default="~/.cache/rndysanos/oce-path")
    parser.add_argument(
        "--arb-mask-report",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "v2-1000000-fixed-simulator-cost-mask-comparison/"
            "cost-local-arbitrage-mask.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "path-oce-1000000-comparison"
        ),
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser


def main() -> None:
    report = run(_parser().parse_args())
    print(f"completed path-OCE comparison with status={report['status']}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
