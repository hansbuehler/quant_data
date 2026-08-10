"""Compare power-exponential OCE in one-step, daily-path, and weekly modes."""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from quantdata.dysanos.grid.rndysanos.run_experiment import _atomic_json
from quantdata.dysanos.grid.rndysanos.run_v2_mask_comparison import (
    _attach_universe,
    _cleanup_shared_memory,
    _share_universe,
)

from .run_one_time_control import run as run_control
from .terminal_universe import TerminalPathDiagnostics
from .torch_prep import GenDysanosExperiment, prepare_terminal_path_experiment

_TIME_FIELDS = {
    "dC",
    "dS",
    "expiry_dS",
    "C",
    "spot",
    "delta",
    "log_strikes",
    "xstate",
}


def _mode_view(
    universe: dict[str, np.ndarray],
    *,
    mode: str,
) -> dict[str, np.ndarray]:
    """Return the full path or its first-trade-date view without copying."""

    if mode in {"path", "weekly"}:
        return universe
    if mode not in {"one-step", "one-step-weekly"}:
        raise ValueError(f"unknown terminal universe mode {mode!r}")
    return {
        name: value[:, :1] if name in _TIME_FIELDS else value
        for name, value in universe.items()
    }


def _job_args(
    args: argparse.Namespace,
    *,
    paths: int,
    mode: str,
) -> argparse.Namespace:
    name = f"{paths}-{mode}-powexp"
    return argparse.Namespace(
        paths=paths,
        epochs=args.epochs,
        seed=args.seed + paths,
        generator_block_size=args.generator_block_size,
        universe_mode=mode,
        trading_state_steps=args.trading_state_steps,
        num_alpha=None,
        use_matrix_exp=True,
        bandwidth=1.0,
        action_type="network",
        state_mode="raw",
        utility="powexp",
        lmbda=1.0,
        cost_prop=0.0,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        precision=32,
        compile=True,
        restart=args.restart,
        early_stopping=True,
        minimum_epochs=args.minimum_epochs,
        debug=False,
        checkpoint_root=str(Path(args.checkpoint_root) / name),
        report=str(Path(args.output_root) / f"{name}.json"),
    )


def _worker(
    descriptors: dict[str, dict[str, Any]],
    diagnostics: TerminalPathDiagnostics,
    args: argparse.Namespace,
) -> None:
    universe, handles = _attach_universe(descriptors)
    try:
        view = _mode_view(universe, mode=args.universe_mode)
        mode_diagnostics = (
            diagnostics
            if not args.universe_mode.startswith("one-step")
            else replace(
                diagnostics,
                mode=args.universe_mode,
                trading_times=1,
                simulated_state_steps=(
                    max(
                        diagnostics.trading_step_stride,
                        diagnostics.expiry_step_offsets[-1],
                    )
                    + 1
                ),
            )
        )
        run_control(
            args,
            prepared_universe=view,
            sampling_diagnostics=mode_diagnostics,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.close()


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    splits = report["stage"]["splits"]
    return {
        "checkpoint": report["stage"]["checkpoint"],
        "device": report["device"],
        "universe_mode": report["universe_mode"],
        "paths": report["paths"],
        "sampling_diagnostics": report["sampling_diagnostics"],
        "convergence": report["stage"]["convergence"],
        "memory": report["stage"]["memory"],
        "compiled_graphs": report["stage"]["compiled_graphs"],
        "baseline_training": report["baseline"]["training"]["martingale"],
        "baseline_validation": report["baseline"]["validation"]["martingale"],
        "training": {
            "objective_gradient": splits["training"]["objective_gradient"],
            "martingale": splits["training"]["martingale"],
        },
        "validation": {
            "objective_gradient": splits["validation"]["objective_gradient"],
            "martingale": splits["validation"]["martingale"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Generate shared daily/weekly paths and schedule six GPU jobs."""

    path_counts = tuple(args.path_counts)
    if len(path_counts) != 2 or any(paths <= 0 or paths % 2 for paths in path_counts):
        raise ValueError("--path-counts requires two positive even path counts")
    output_root = Path(args.output_root)
    comparison_path = output_root / "terminal-mode-comparison.json"
    jobs = [
        _job_args(args, paths=paths, mode=mode)
        for paths in sorted(path_counts, reverse=True)
        for mode in args.modes
    ]
    report: dict[str, Any] = {
        "status": "preparing_universes",
        "path_counts": list(path_counts),
        "epochs": args.epochs,
        "training_fraction": 0.5,
        "utility": "powexp",
        "state_mode": "raw",
        "simulator": {
            "num_alpha": None,
            "use_matrix_exp": True,
            "bandwidth": 1.0,
            "trading_state_steps": args.trading_state_steps,
        },
        "jobs": [job.report for job in jobs],
    }
    _atomic_json(comparison_path, report)

    shared: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    diagnostics_by_source: dict[tuple[int, str], TerminalPathDiagnostics] = {}
    shared_bytes: dict[str, int] = {}
    preparation_seconds: dict[str, float] = {}
    all_handles: list[Any] = []
    try:
        source_modes = {
            "weekly" if mode in {"weekly", "one-step-weekly"} else "path"
            for mode in args.modes
        }
        for paths in path_counts:
            for source_mode in sorted(source_modes):
                source_key = (paths, source_mode)
                started = time.monotonic()
                universe, diagnostics = prepare_terminal_path_experiment(
                    GenDysanosExperiment(
                        num_samples=paths,
                        mode=source_mode,
                        num_steps=args.trading_state_steps,
                        num_alpha=None,
                        use_matrix_exp=True,
                        bandwidth=1.0,
                        seed=args.seed + paths,
                        block_size=args.generator_block_size,
                    )
                )
                descriptors, handles, total_bytes = _share_universe(universe)
                del universe
                gc.collect()
                shared[source_key] = descriptors
                diagnostics_by_source[source_key] = diagnostics
                all_handles.extend(handles)
                label = f"{paths}-{source_mode}"
                shared_bytes[label] = total_bytes
                preparation_seconds[label] = time.monotonic() - started
                report.update(
                    status=f"prepared_{label}",
                    preparation_seconds=preparation_seconds,
                    shared_memory_bytes=shared_bytes,
                    sampling_diagnostics={
                        f"{key[0]}-{key[1]}": asdict(value)
                        for key, value in diagnostics_by_source.items()
                    },
                )
                _atomic_json(comparison_path, report)
        atexit.register(_cleanup_shared_memory, all_handles)

        context = mp.get_context("spawn")
        processes = []
        for job in jobs:
            source_mode = (
                "weekly"
                if job.universe_mode in {"weekly", "one-step-weekly"}
                else "path"
            )
            source_key = (job.paths, source_mode)
            processes.append(
                context.Process(
                    target=_worker,
                    args=(
                        shared[source_key],
                        diagnostics_by_source[source_key],
                        job,
                    ),
                    name=f"{job.paths}-{job.universe_mode}",
                )
            )
        report["status"] = "running_six_jobs_on_four_gpus"
        _atomic_json(comparison_path, report)
        pending = list(processes)
        running: list[mp.Process] = []
        exitcodes: dict[str, int | None] = {}
        while pending or running:
            while pending and len(running) < 4:
                process = pending.pop(0)
                process.start()
                running.append(process)
            completed = [process for process in running if not process.is_alive()]
            if not completed:
                time.sleep(1.0)
                continue
            for process in completed:
                process.join()
                exitcodes[process.name] = process.exitcode
                running.remove(process)
            report.update(
                active_jobs=[process.name for process in running],
                pending_jobs=[process.name for process in pending],
                exitcodes=exitcodes,
            )
            _atomic_json(comparison_path, report)
        if any(code != 0 for code in exitcodes.values()):
            report.update(status="failed", exitcodes=exitcodes)
            _atomic_json(comparison_path, report)
            raise RuntimeError(f"one or more terminal-mode jobs failed: {exitcodes}")
        outputs = [
            json.loads(Path(job.report).read_text(encoding="utf-8"))
            for job in jobs
        ]
        result = {
            **report,
            "status": "complete",
            "exitcodes": exitcodes,
            "controls": {
                f"{output['paths']}-{output['universe_mode']}": _compact(output)
                for output in outputs
            },
        }
        _atomic_json(comparison_path, result)
        return result
    finally:
        if all_handles:
            try:
                atexit.unregister(_cleanup_shared_memory)
            except ValueError:
                pass
            _cleanup_shared_memory(all_handles)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--path-counts",
        nargs=2,
        type=int,
        default=(100_000, 1_000_000),
    )
    result.add_argument("--epochs", type=int, default=20_000)
    result.add_argument(
        "--modes",
        nargs="+",
        choices=("one-step", "one-step-weekly", "path", "weekly"),
        default=("path", "weekly", "one-step", "one-step-weekly"),
    )
    result.add_argument("--minimum-epochs", type=int, default=2_000)
    result.add_argument("--seed", type=int, default=600_000)
    result.add_argument("--trading-state-steps", type=int, default=10)
    result.add_argument("--generator-block-size", type=int, default=10_000)
    result.add_argument("--batch-size", type=int, default=4_096)
    result.add_argument("--val-batch-size", type=int, default=8_192)
    result.add_argument("--lr", type=float, default=1.0e-4)
    result.add_argument("--restart", action="store_true")
    result.add_argument(
        "--checkpoint-root",
        default="~/.cache/rndysanos/oce-gendysanos-terminal-modes",
    )
    result.add_argument(
        "--output-root",
        default=(
            "quantdata/dysanos/grid/gendysanos/reports/"
            "terminal-mode-powerexp"
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparison": str(Path(args.output_root) / "terminal-mode-comparison.json"),
                "exitcodes": report["exitcodes"],
                "controls": report["controls"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
