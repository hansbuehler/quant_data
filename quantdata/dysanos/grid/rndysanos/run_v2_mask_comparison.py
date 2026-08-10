"""Run million-path V2 full/raw and unmasked/masked comparisons on four GPUs."""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np

from .run_cost_local_arbitrage import run as run_cost_mask
from .run_direct_arbitrage import run as run_v2
from .torch_prep import RNDysanosExperiment, prep_experiment


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _share_universe(
    universe: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[SharedMemory], int]:
    descriptors: dict[str, dict[str, Any]] = {}
    handles: list[SharedMemory] = []
    total_bytes = 0
    try:
        for name, value in universe.items():
            array = np.ascontiguousarray(value)
            shared = SharedMemory(create=True, size=array.nbytes)
            handles.append(shared)
            output = np.ndarray(array.shape, dtype=array.dtype, buffer=shared.buf)
            np.copyto(output, array)
            descriptors[name] = {
                "name": shared.name,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
            total_bytes += array.nbytes
    except BaseException:
        _cleanup_shared_memory(handles)
        raise
    return descriptors, handles, total_bytes


def _attach_universe(
    descriptors: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[SharedMemory]]:
    universe: dict[str, np.ndarray] = {}
    handles: list[SharedMemory] = []
    try:
        for name, descriptor in descriptors.items():
            shared = SharedMemory(name=descriptor["name"], track=False)
            handles.append(shared)
            universe[name] = np.ndarray(
                tuple(descriptor["shape"]),
                dtype=np.dtype(descriptor["dtype"]),
                buffer=shared.buf,
            )
    except BaseException:
        for handle in handles:
            handle.close()
        raise
    return universe, handles


def _cleanup_shared_memory(handles: list[SharedMemory]) -> None:
    for handle in handles:
        try:
            handle.close()
        finally:
            try:
                handle.unlink()
            except FileNotFoundError:
                pass


def _cost_mask_worker(
    descriptors: dict[str, dict[str, Any]], args: argparse.Namespace
) -> None:
    universe, handles = _attach_universe(descriptors)
    try:
        run_cost_mask(args, prepared_universe=universe)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.close()


def _v2_worker(
    descriptors: dict[str, dict[str, Any]], args: argparse.Namespace
) -> None:
    universe, handles = _attach_universe(descriptors)
    try:
        run_v2(args, prepared_universe=universe)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.close()


def _complete_report(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load one CPU universe once, then attach four spawn workers to shared memory."""

    if args.paths <= 0 or args.paths % 2:
        raise ValueError("--paths must be a positive even number")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    overall_path = output_root / "v2-radial-mask-comparison.json"
    mask_report_path = output_root / "cost-local-arbitrage-mask.json"
    report: dict[str, Any] = {
        "status": "preparing_shared_universe",
        "paths": args.paths,
        "training_paths": args.paths // 2,
        "validation_paths": args.paths // 2,
        "epochs": args.epochs,
        "universe_seed_offset": args.universe_seed_offset,
        "shared_memory": (
            "one parent-loaded NumPy universe copied once into POSIX shared "
            "memory; clean spawn workers attach without CPU duplication and "
            "transfer only to their assigned GPU"
        ),
        "mask_report": str(mask_report_path),
    }
    _atomic_json(overall_path, report)

    preparation_started = time.monotonic()
    experiment = RNDysanosExperiment(
        num_samples=args.paths,
        pcnt_validation=0.5,
        cnt_universe=1,
        universe_seed_offset=args.universe_seed_offset,
    )
    universe = prep_experiment(experiment, plot_sampler=False)[0]
    shared_universe, shared_handles, shared_bytes = _share_universe(universe)
    del universe
    gc.collect()
    atexit.register(_cleanup_shared_memory, shared_handles)
    report.update(
        status="fitting_cost_adjusted_mask",
        shared_universe_preparation_seconds=time.monotonic() - preparation_started,
        shared_memory_bytes=shared_bytes,
        shared_memory_segments=len(shared_handles),
    )
    _atomic_json(overall_path, report)

    context = mp.get_context("spawn")
    if not _complete_report(mask_report_path) or args.restart_mask:
        mask_args = argparse.Namespace(
            paths=args.paths,
            universe_seed_offset=args.universe_seed_offset,
            neighbors=args.neighbors,
            query_batch_size=args.query_batch_size,
            instrument_batch_size=args.instrument_batch_size,
            valuation_batch_size=args.valuation_batch_size,
            roundoff_multiplier=args.roundoff_multiplier,
            cost_prop=args.cost_prop,
            progress_seconds=args.progress_seconds,
            mask_storage_root=args.mask_storage_root,
            conditional_report=args.conditional_report,
            report=str(mask_report_path),
        )
        mask_process = context.Process(
            target=_cost_mask_worker,
            args=(shared_universe, mask_args),
            name="cost-local-arbitrage-mask",
        )
        mask_process.start()
        mask_process.join()
        if mask_process.exitcode != 0:
            report.update(status="mask_failed", mask_exitcode=mask_process.exitcode)
            _atomic_json(overall_path, report)
            raise RuntimeError(
                f"cost-adjusted mask worker failed with exit code {mask_process.exitcode}"
            )

    jobs: list[tuple[str, argparse.Namespace]] = []
    for state_mode, seed in (("full", args.seed), ("raw", args.seed + 1)):
        for masked in (False, True):
            name = f"v2-radial-{state_mode}-{'masked' if masked else 'unmasked'}"
            jobs.append(
                (
                    name,
                    argparse.Namespace(
                        paths=args.paths,
                        epochs=args.epochs,
                        cost_mode="combined",
                        state_mode=state_mode,
                        arbitrage_mask_report=(
                            str(mask_report_path) if masked else None
                        ),
                        seed=seed,
                        universe_seed_offset=args.universe_seed_offset,
                        epsilon=args.epsilon,
                        delta=args.delta,
                        numerical_tolerance=args.numerical_tolerance,
                        smooth_temperature=args.smooth_temperature,
                        tail_fraction=args.tail_fraction,
                        downside_weight=args.downside_weight,
                        tail_weight=args.tail_weight,
                        overlap_weight=args.overlap_weight,
                        mean_reward_weight=args.mean_reward_weight,
                        discovery_fraction=args.discovery_fraction,
                        constraint_ramp_fraction=args.constraint_ramp_fraction,
                        initial_trade_fraction=args.initial_trade_fraction,
                        objective="radial-l1-v2",
                        cap_weight=args.cap_weight,
                        cost_prop=args.cost_prop,
                        batch_size=args.batch_size,
                        valuation_batch_size=args.valuation_batch_size,
                        lr=args.lr,
                        final_lr=args.final_lr,
                        weight_decay=args.weight_decay,
                        value_update=args.value_update,
                        checkpoint_update=args.checkpoint_update,
                        checkpoint_root=args.checkpoint_root,
                        conditional_report=args.conditional_report,
                        report=str(output_root / f"{name}.json"),
                        restart=args.restart_v2,
                        no_compile=args.no_compile,
                    ),
                )
            )

    report.update(
        status="running_four_v2_jobs",
        jobs={name: job.report for name, job in jobs},
    )
    _atomic_json(overall_path, report)
    processes = [
        (
            name,
            context.Process(
                target=_v2_worker,
                args=(shared_universe, job),
                name=name,
            ),
        )
        for name, job in jobs
    ]
    started = time.monotonic()
    for _, process in processes:
        process.start()
    for _, process in processes:
        process.join()
    exitcodes = {name: process.exitcode for name, process in processes}
    if any(code != 0 for code in exitcodes.values()):
        report.update(status="v2_failed", exitcodes=exitcodes)
        _atomic_json(overall_path, report)
        raise RuntimeError(f"one or more V2 workers failed: {exitcodes}")

    outputs = {
        name: json.loads(Path(job.report).read_text(encoding="utf-8"))
        for name, job in jobs
    }
    report.update(
        status="complete",
        four_job_elapsed_seconds=time.monotonic() - started,
        exitcodes=exitcodes,
        summaries={
            name: {
                "device": output["device"],
                "state_features": output["state_features"],
                "masked": output["arbitrage_mask_report"] is not None,
                "training": output["training"],
                "validation": output["validation"],
                "report": jobs[index][1].report,
            }
            for index, (name, output) in enumerate(outputs.items())
        },
    )
    _atomic_json(overall_path, report)
    atexit.unregister(_cleanup_shared_memory)
    _cleanup_shared_memory(shared_handles)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=200_000)
    parser.add_argument("--universe-seed-offset", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=120_771)
    parser.add_argument("--neighbors", type=int, default=10_240)
    parser.add_argument("--query-batch-size", type=int, default=2_048)
    parser.add_argument("--instrument-batch-size", type=int, default=10)
    parser.add_argument("--roundoff-multiplier", type=float, default=32.0)
    parser.add_argument("--cost-prop", type=float, default=0.0001)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--numerical-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--smooth-temperature", type=float, default=0.0025)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--downside-weight", type=float, default=10.0)
    parser.add_argument("--tail-weight", type=float, default=100.0)
    parser.add_argument("--overlap-weight", type=float, default=1.0)
    parser.add_argument("--mean-reward-weight", type=float, default=0.1)
    parser.add_argument("--discovery-fraction", type=float, default=0.05)
    parser.add_argument("--constraint-ramp-fraction", type=float, default=0.45)
    parser.add_argument("--initial-trade-fraction", type=float, default=0.01)
    parser.add_argument("--cap-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--valuation-batch-size", type=int, default=4_096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--final-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--value-update", type=int, default=20_000)
    parser.add_argument("--checkpoint-update", type=int, default=20_000)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument(
        "--mask-storage-root", default="/media/storage/tmp/rndysanos-v2-masks"
    )
    parser.add_argument(
        "--checkpoint-root", default="~/.cache/rndysanos/direct-arbitrage"
    )
    parser.add_argument(
        "--conditional-report",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "full-200000-e1000000-up-to10-per-cap-conditional-same.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "v2-1000000-cost-mask-comparison"
        ),
    )
    parser.add_argument("--restart-mask", action="store_true")
    parser.add_argument("--restart-v2", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser


def main() -> None:
    report = run(_parser().parse_args())
    print(
        f"completed V2 mask comparison with status={report['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
