"""Compare no-offset path-OCE optimizers on one shared simulated universe.

The comparison deliberately separates two questions:

* three neural strategies use a constant-rate AdamW optimizer and transaction
  costs; and
* one deterministic-time power-exponential strategy is a cost-free convex
  benchmark solved by SciPy L-BFGS-B over a GPU Torch objective and gradient.

All four workers receive the same read-only universe through shared memory and
are assigned independent CUDA devices by :class:`quantdata.torch.DeviceLock`.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from .run_experiment import _atomic_json
from .run_path_oce_comparison import _final_summary, _worker
from .run_v2_mask_comparison import (
    _cleanup_shared_memory,
    _share_universe,
)
from .torch_prep import RNDysanosExperiment, prep_experiment


def _job_args(
    args: argparse.Namespace,
    *,
    name: str,
    utility: str,
    regime: str,
    action_type: str,
    optimizer: str,
    epochs: int,
    lr: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        paths=args.paths,
        epochs=epochs,
        moment_epochs=1,
        stages=1,
        validation_fraction=0.5,
        universe_seed_offset=args.universe_seed_offset,
        independent_split_universes=False,
        same_universe=True,
        regimes=regime,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        optimizer=optimizer,
        action_type=action_type,
        lr=lr,
        final_lr=lr,
        moment_lr=1.0e-3,
        moment_final_lr=1.0e-3,
        action_initialization_epochs=0,
        action_initialization_lr=1.0e-3,
        utility=utility,
        lmbda=1.0 / 9.0 if utility == "cvar" else 1.0,
        weight_decay=(
            0.0 if optimizer in {"LBFGS", "SciPy"} else args.weight_decay
        ),
        minimum_epochs=(
            args.deterministic_minimum_epochs
            if action_type == "deterministic"
            else args.network_minimum_epochs
        ),
        minimum_moment_epochs=0,
        early_stopping=True,
        convergence_points=args.convergence_points,
        convergence_relative_tolerance=args.convergence_relative_tolerance,
        cost_band_tolerance=args.cost_band_tolerance,
        minimum_unsigned_improvement=0.0,
        continue_after_failed_gate=False,
        value_seconds=args.value_seconds,
        value_update=(
            args.deterministic_value_update
            if action_type == "deterministic"
            else args.network_value_update
        ),
        checkpoint_seconds=args.checkpoint_seconds,
        checkpoint_update=(
            args.deterministic_value_update
            if action_type == "deterministic"
            else args.network_value_update
        ),
        checkpoint_root=str(Path(args.checkpoint_root) / name),
        arb_mask_report=None,
        report=str(Path(args.output_root) / f"{name}.json"),
        restart=args.restart,
        no_compile=args.no_compile,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the four no-offset jobs concurrently on one simulated universe."""

    output_root = Path(args.output_root)
    comparison_path = output_root / "no-offset-comparison.json"
    specifications = [
        (
            "powexp-deterministic-costfree-scipy",
            "costfree",
            "powexp",
            "deterministic",
            "SciPy",
            args.deterministic_epochs,
            args.lbfgs_lr,
        ),
        (
            "powexp-network-unconditional-adamw",
            "unconditional",
            "powexp",
            "network",
            "AdamW",
            args.network_epochs,
            args.network_lr,
        ),
        (
            "entropy-network-unconditional-adamw",
            "unconditional",
            "entropy",
            "network",
            "AdamW",
            args.network_epochs,
            args.network_lr,
        ),
        (
            "cvar90-network-unconditional-adamw",
            "unconditional",
            "cvar",
            "network",
            "AdamW",
            args.network_epochs,
            args.network_lr,
        ),
    ]
    jobs = [
        (
            name,
            regime,
            _job_args(
                args,
                name=name,
                utility=utility,
                regime=regime,
                action_type=action_type,
                optimizer=optimizer,
                epochs=epochs,
                lr=lr,
            ),
        )
        for name, regime, utility, action_type, optimizer, epochs, lr in specifications
    ]
    report: dict[str, Any] = {
        "status": "preparing_shared_universe",
        "paths": args.paths,
        "training_paths": args.paths // 2,
        "validation_paths": args.paths // 2,
        "universe_seed_offset": args.universe_seed_offset,
        "simulator": {
            "num_alpha": args.num_alpha,
            "bandwidth": args.bandwidth,
            "use_matrix_exp": args.use_matrix_exp,
        },
        "jobs": {name: job.report for name, _, job in jobs},
        "optimizer_policy": {
            "neural": "constant-global-rate AdamW",
            "deterministic": (
                "SciPy L-BFGS-B with Torch GPU objective and exact autograd"
            ),
            "learning_rate_decay": False,
        },
    }
    _atomic_json(comparison_path, report)

    started = time.monotonic()
    experiment = RNDysanosExperiment(
        num_samples=args.paths,
        pcnt_validation=0.5,
        cnt_universe=1,
        universe_seed_offset=args.universe_seed_offset,
        num_alpha=args.num_alpha,
        bandwidth=args.bandwidth,
        use_matrix_exp=args.use_matrix_exp,
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
            raise RuntimeError(f"one or more no-offset workers failed: {exitcodes}")
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
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--universe-seed-offset", type=int, default=300_000)
    parser.add_argument(
        "--num-alpha",
        type=lambda value: None if value.lower() == "none" else int(value),
        default=5,
        help="PCA innovation factors; use 'none' for every xstate dimension",
    )
    parser.add_argument("--bandwidth", type=float, default=0.2)
    parser.add_argument("--use-matrix-exp", action="store_true")
    parser.add_argument("--network-epochs", type=int, default=100_000)
    parser.add_argument("--network-minimum-epochs", type=int, default=20_000)
    parser.add_argument("--deterministic-epochs", type=int, default=250)
    parser.add_argument("--deterministic-minimum-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--val-batch-size", type=int, default=4_096)
    parser.add_argument("--network-lr", type=float, default=1.0e-4)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--network-value-update", type=int, default=1_000)
    parser.add_argument("--deterministic-value-update", type=int, default=5)
    parser.add_argument("--value-seconds", type=float, default=900.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=900.0)
    parser.add_argument("--convergence-points", type=int, default=5)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--cost-band-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-root",
        default="~/.cache/rndysanos/oce-path-no-offset",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "path-no-offset-100000-comparison"
        ),
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser


def main() -> None:
    report = run(_parser().parse_args())
    print(f"completed no-offset comparison with status={report['status']}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
