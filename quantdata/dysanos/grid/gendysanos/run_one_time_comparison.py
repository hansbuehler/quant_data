"""Run four conditional one-time SANOS controls on one shared universe."""

from __future__ import annotations

import argparse
import atexit
import json
import multiprocessing as mp
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quantdata.dysanos.grid.rndysanos.run_experiment import _atomic_json
from quantdata.dysanos.grid.rndysanos.run_v2_mask_comparison import (
    _attach_universe,
    _cleanup_shared_memory,
    _share_universe,
)

from .run_one_time_control import run as run_control
from .terminal_universe import TerminalPathDiagnostics
from .torch_prep import (
    GenDysanosExperiment,
    prepare_one_time_terminal_experiment,
)


def _worker(
    descriptors: dict[str, dict[str, Any]],
    sampling: TerminalPathDiagnostics,
    args: argparse.Namespace,
) -> None:
    universe, handles = _attach_universe(descriptors)
    try:
        run_control(
            args,
            prepared_universe=universe,
            sampling_diagnostics=sampling,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.close()


def _job_args(
    args: argparse.Namespace,
    *,
    utility: str,
    state_mode: str,
) -> argparse.Namespace:
    name = f"{utility}-{state_mode}"
    return argparse.Namespace(
        paths=args.paths,
        epochs=args.epochs,
        seed=args.seed,
        generator_block_size=args.generator_block_size,
        universe_mode="one-step",
        trading_state_steps=10,
        num_alpha=None,
        use_matrix_exp=True,
        bandwidth=1.0,
        action_type="network",
        state_mode=state_mode,
        utility=utility,
        lmbda=1.0,
        cost_prop=0.0,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        precision=args.precision,
        compile=True,
        restart=args.restart,
        early_stopping=True,
        minimum_epochs=args.minimum_epochs,
        debug=False,
        checkpoint_root=args.checkpoint_root,
        report=str(Path(args.output_root) / f"{name}.json"),
    )


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": report["device"],
        "utility": report["utility"],
        "state_mode": report["state_mode"],
        "preparation_seconds": report["preparation_seconds"],
        "compiled_graphs": report["stage"]["compiled_graphs"],
        "memory": report["stage"]["memory"],
        "convergence": report["stage"]["convergence"],
        "training": {
            "loss": report["stage"]["splits"]["training"]["loss"],
            "objective_gradient": report["stage"]["splits"]["training"][
                "objective_gradient"
            ],
            "martingale": report["stage"]["splits"]["training"]["martingale"],
        },
        "validation": {
            "loss": report["stage"]["splits"]["validation"]["loss"],
            "objective_gradient": report["stage"]["splits"]["validation"][
                "objective_gradient"
            ],
            "martingale": report["stage"]["splits"]["validation"]["martingale"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare once, run four GPU workers, and aggregate their reports."""

    output_root = Path(args.output_root)
    comparison_path = output_root / "one-time-comparison.json"
    universe, sampling = prepare_one_time_terminal_experiment(
        GenDysanosExperiment(
            num_samples=args.paths,
            seed=args.seed,
            block_size=args.generator_block_size,
        )
    )
    descriptors, handles, shared_bytes = _share_universe(universe)
    del universe
    atexit.register(_cleanup_shared_memory, handles)
    jobs = [
        _job_args(args, utility=utility, state_mode=state_mode)
        for utility, state_mode in (
            ("entropy", "raw"),
            ("powexp", "raw"),
            ("entropy", "full"),
            ("powexp", "full"),
        )
    ]
    initial = {
        "status": "running",
        "paths": args.paths,
        "training_paths": args.paths // 2,
        "validation_paths": args.paths - args.paths // 2,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "shared_memory_bytes": shared_bytes,
        "sampling_diagnostics": asdict(sampling),
        "reports": [job.report for job in jobs],
    }
    _atomic_json(comparison_path, initial)

    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(descriptors, sampling, job),
            name=f"{job.utility}-{job.state_mode}",
        )
        for job in jobs
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        exitcodes = {process.name: process.exitcode for process in processes}
        if any(code != 0 for code in exitcodes.values()):
            failed = {**initial, "status": "failed", "exitcodes": exitcodes}
            _atomic_json(comparison_path, failed)
            raise RuntimeError(f"one or more controls failed: {exitcodes}")
        reports = [
            json.loads(Path(job.report).read_text(encoding="utf-8")) for job in jobs
        ]
        result = {
            **initial,
            "status": "complete",
            "exitcodes": exitcodes,
            "controls": {
                f"{report['utility']}-{report['state_mode']}": _compact(report)
                for report in reports
            },
        }
        _atomic_json(comparison_path, result)
        return result
    finally:
        _cleanup_shared_memory(handles)
        atexit.unregister(_cleanup_shared_memory)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--paths", type=int, default=100_000)
    result.add_argument("--epochs", type=int, default=20_000)
    result.add_argument("--seed", type=int, default=500_000)
    result.add_argument("--generator-block-size", type=int, default=25_000)
    result.add_argument("--batch-size", type=int, default=4_096)
    result.add_argument("--val-batch-size", type=int, default=8_192)
    result.add_argument("--lr", type=float, default=1.0e-4)
    result.add_argument("--precision", type=int, choices=(32, 64), default=32)
    result.add_argument("--restart", action="store_true")
    result.add_argument("--minimum-epochs", type=int, default=2_000)
    result.add_argument(
        "--checkpoint-root", default="~/.cache/rndysanos/oce-gendysanos-control"
    )
    result.add_argument(
        "--output-root",
        default="quantdata/dysanos/grid/gendysanos/reports/one-time-100000",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparison": str(Path(args.output_root) / "one-time-comparison.json"),
                "exitcodes": report["exitcodes"],
                "controls": report["controls"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
