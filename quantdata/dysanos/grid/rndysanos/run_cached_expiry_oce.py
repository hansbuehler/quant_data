"""Compare zero and no-arbitrage-basket warm starts on a cached expiry market."""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.dysanos.grid.gendysanos.required_runs import stratified_path_split
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

_VARIANTS = ("zero", "no-arb-warm")
_DEFAULT_MARKET_REPORT = (
    "quantdata/dysanos/grid/gendysanos/reports/required/"
    "dysanos-daily-1000000-expiry-delta-1e-04.json"
)
_DEFAULT_WARM_REPORT = (
    "quantdata/dysanos/grid/gendysanos/reports/required/"
    "dysanos-daily-1000000-expiry-raw-1e-04.json"
)
_SUPPORTED_TRADING_FREQUENCIES = frozenset({"daily", "weekly"})


def _validate_generated_market_report(
    report: dict[str, Any],
    *,
    expected_hedge: str,
    cost_prop: float,
) -> dict[str, Any]:
    """Validate the generated-market semantics required by this OCE runner."""

    configuration = report.get("configuration", {})
    frequency = configuration.get("trading_frequency")
    if (
        report.get("status") != "complete"
        or configuration.get("generator") != "dysanos"
        or frequency not in _SUPPORTED_TRADING_FREQUENCIES
        or configuration.get("return_horizon") != "expiry"
        or configuration.get("hedge") != expected_hedge
        or configuration.get("path_count") != 1_000_000
        or not math.isclose(
            float(configuration.get("cost_rate", math.nan)),
            cost_prop,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError(
            "the cached OCE runner requires a complete million-path daily or "
            f"weekly DySANOS expiry-{expected_hedge} market at cost {cost_prop:g}"
        )
    return configuration


def _load_cached_universe(
    market_report_path: str | Path,
    *,
    paths: int,
    cost_prop: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a stratified reference/audit subset directly from the block cache."""

    report_path = Path(market_report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    configuration = _validate_generated_market_report(
        report,
        expected_hedge="delta",
        cost_prop=cost_prop,
    )
    cache_directory = Path(report["base_cache"])
    manifest = json.loads(
        (cache_directory / "manifest.json").read_text(encoding="utf-8")
    )
    total_paths = int(manifest["num_samples"])
    if not 1 < paths <= total_paths:
        raise ValueError(f"paths must lie in [2,{total_paths}], got {paths}")

    components = np.empty(total_paths, dtype=np.int16)
    for item in manifest["blocks"]:
        with np.load(cache_directory / item["file"], allow_pickle=False) as block:
            components[item["start"] : item["stop"]] = block[
                "common__importance_component"
            ]
    split_seed = total_paths
    split = stratified_path_split(components, seed=split_seed)
    validation_paths = max(1, int(round(0.2 * paths)))
    training_paths = paths - validation_paths
    if training_paths > split.reference.size or validation_paths > split.audit.size:
        raise ValueError("requested subset exceeds the cached reference/audit split")
    selected = np.concatenate(
        (split.reference[:training_paths], split.audit[:validation_paths])
    )
    destination = np.full(total_paths, -1, dtype=np.int64)
    destination[selected] = np.arange(paths, dtype=np.int64)

    field_map = {
        "C": "common__C",
        "dC": "expiry__dC",
        "delta": "common__delta",
        "expiry_dS": "expiry__expiry_dS",
        "log_path_weight": "common__log_path_weight",
        "log_strikes": "common__log_strikes",
        "spot": "common__spot",
        "xstate": "common__xstate",
    }
    universe: dict[str, np.ndarray] = {}
    expiries: np.ndarray | None = None
    for item in manifest["blocks"]:
        start, stop = int(item["start"]), int(item["stop"])
        rows = destination[start:stop]
        retained = rows >= 0
        if not np.any(retained):
            continue
        target_rows = rows[retained]
        with np.load(cache_directory / item["file"], allow_pickle=False) as block:
            if expiries is None:
                expiries = np.asarray(block["common__expiries"]).copy()
            for name, stored_name in field_map.items():
                source = block[stored_name]
                if name not in universe:
                    universe[name] = np.empty(
                        (paths, *source.shape[1:]), dtype=source.dtype
                    )
                universe[name][target_rows] = source[retained]
    if expiries is None:
        raise RuntimeError("the cached expiry market contains no retained blocks")
    universe["expiries"] = expiries
    universe["dS"] = np.ascontiguousarray(universe["expiry_dS"][..., 0])
    for name, value in universe.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"cached universe field {name!r} is non-finite")
    diagnostics = {
        "market_report": str(report_path),
        "base_cache": str(cache_directory),
        "base_cache_uid": report["base_cache_uid"],
        "cached_total_paths": total_paths,
        "selected_paths": paths,
        "training_paths": training_paths,
        "validation_paths": validation_paths,
        "split_seed": split_seed,
        "split_semantics": (
            "the no-arbitrage benchmark's mixture-stratified reference paths "
            "precede its disjoint audit paths"
        ),
        "market_configuration": configuration,
        "generator_configuration": manifest["configuration"],
        "sampling": manifest["diagnostics"]["sampling"],
    }
    return universe, diagnostics


def _run_job(
    universe: dict[str, Any],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
    *,
    variant: str,
    device: torch.device,
) -> dict[str, Any]:
    env = BaseEnvironment(precision=32, device=device, debug=False)
    torch.empty((), dtype=env.fdtype, device=env.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(env.device)
    warm = variant == "no-arb-warm"
    market_configuration = diagnostics["market_configuration"]
    frequency = market_configuration["trading_frequency"]
    cost_label = f"{args.cost_prop:.0e}".replace("+", "")
    config = DysanosConfig(
        label=(
            f"cached-expiry-powexp-{frequency}-cost-{cost_label}-{variant}-"
            f"{args.paths}-{args.seed}"
        ),
        group=(
            f"gendysanos/cached-expiry/{frequency}/cost-{cost_label}/"
            f"{args.paths}/seed-{args.seed}/{variant}"
        ),
        user_version="1.0",
        utility="powexp",
        lmbda=1.0,
        oce_variant="simplified",
        action_type="network",
        state_mode="raw",
        zero_action_initialization=True,
        trade_spot=False,
        delta_hedged=True,
        cash_returns=False,
        learn_cost=False,
        cost_cond=0.0,
        cost_uncond=0.0,
        cost_prop=args.cost_prop,
        seed=args.seed,
    )
    train_config = TrainConfig(
        trn_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        validation_fraction=0.2,
        optimizer="AdamW",
        lr=args.lr,
        final_lr=args.lr,
        weight_decay=0.0,
        action_initialization_epochs=(args.initialization_epochs if warm else 0),
        action_initialization_lr=args.initialization_lr,
        action_initialization_report=(args.warm_start_report if warm else None),
        action_initialization_max_cap_fraction=args.maximum_cap_fraction,
        compile=not args.no_compile,
        value_seconds=args.value_seconds,
        value_update=args.value_update,
        checkpoint_seconds=args.checkpoint_seconds,
        checkpoint_update=args.checkpoint_update,
        heavy_output_update=args.epochs,
        early_stopping=True,
        minimum_epochs=args.minimum_epochs,
        convergence_points=args.convergence_points,
        convergence_relative_tolerance=args.convergence_relative_tolerance,
        require_cost_band=False,
        checkpoint_root=str(Path(args.checkpoint_root) / variant),
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
        "experiment": (
            f"cached million-path {frequency} expiry powerexp OCE at "
            f"cost {args.cost_prop:g}"
        ),
        "variant": variant,
        "paths": args.paths,
        "objective": "sup_a E_P[u(sum_t a(z_t)^T dH_t-|a(z_t)|^T c_t)]",
        "return_semantics": (
            "expiry-matched delta-hedged option cash P&L; the ambiguous single "
            "explicit spot output is disabled"
        ),
        "cost_semantics": (
            f"rate {args.cost_prop:g} times gross option-plus-delta-hedge notional"
        ),
        "warm_start_semantics": (
            "raw-option local basket option quantities fitted only on the "
            "no-arbitrage reference split; audit outcomes excluded; targets "
            "kept strictly inside tanh"
            if warm
            else "exact zero output, whose tanh derivative is one"
        ),
        "universe": diagnostics,
        "model_config": asdict(config),
        "train_config": asdict(train_config),
        "device": str(device),
        "baseline": {
            "training": _baseline_split(result.training),
            "validation": _baseline_split(result.validation),
        },
        "stage": stage,
    }
    _atomic_json(Path(args.output_root) / f"{variant}.json", report)
    return report


def _worker(
    descriptors: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
    variant: str,
) -> None:
    with DeviceLock() as device_lock:
        universe, handles = _attach_universe(descriptors)
        try:
            _run_job(
                universe,
                diagnostics,
                args,
                variant=variant,
                device=device_lock.device,
            )
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            for handle in handles:
                handle.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = output_root / "comparison.json"
    report: dict[str, Any] = {
        "status": "loading_cached_market",
        "paths": args.paths,
        "variants": list(_VARIANTS),
        "jobs": {},
    }
    _atomic_json(comparison_path, report)
    started = time.monotonic()
    universe, diagnostics = _load_cached_universe(
        args.market_report,
        paths=args.paths,
        cost_prop=args.cost_prop,
    )
    warm_report = json.loads(
        Path(args.warm_start_report).read_text(encoding="utf-8")
    )
    warm_configuration = _validate_generated_market_report(
        warm_report,
        expected_hedge="raw",
        cost_prop=args.cost_prop,
    )
    if (
        warm_report.get("base_cache_uid") != diagnostics["base_cache_uid"]
        or warm_configuration["trading_frequency"]
        != diagnostics["market_configuration"]["trading_frequency"]
    ):
        raise ValueError(
            "the warm-start report must use the same generated market as the "
            "delta-hedged OCE report"
        )
    descriptors, handles, shared_bytes = _share_universe(universe)
    del universe
    gc.collect()
    atexit.register(_cleanup_shared_memory, handles)
    report.update(
        status="training",
        preparation_seconds=time.monotonic() - started,
        shared_memory_bytes=shared_bytes,
        universe=diagnostics,
        jobs={
            variant: str(output_root / f"{variant}.json") for variant in _VARIANTS
        },
    )
    _atomic_json(comparison_path, report)

    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(descriptors, diagnostics, args, variant),
            name=f"cached-expiry-oce-{variant}",
        )
        for variant in _VARIANTS
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
        exitcodes = {process.name: process.exitcode for process in processes}
        report["exitcodes"] = exitcodes
        if any(code != 0 for code in exitcodes.values()):
            report.update(status="failed", active_jobs=[])
            _atomic_json(comparison_path, report)
            raise RuntimeError(f"cached expiry OCE workers failed: {exitcodes}")
        outputs = {
            variant: json.loads(
                (output_root / f"{variant}.json").read_text(encoding="utf-8")
            )
            for variant in _VARIANTS
        }
        report.update(
            status="complete",
            active_jobs=[],
            summaries={
                variant: {
                    "device": output["device"],
                    "action_initialization": output["stage"][
                        "action_initialization"
                    ],
                    "convergence": output["stage"]["convergence"],
                    "memory": output["stage"]["memory"],
                    "training": output["stage"]["splits"]["training"],
                    "validation": output["stage"]["splits"]["validation"],
                }
                for variant, output in outputs.items()
            },
        )
        _atomic_json(comparison_path, report)
        return report
    finally:
        try:
            atexit.unregister(_cleanup_shared_memory)
        except ValueError:
            pass
        _cleanup_shared_memory(handles)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=200_000)
    parser.add_argument("--minimum-epochs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1_700_000)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--val-batch-size", type=int, default=8_192)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--cost-prop", type=float, default=1.0e-4)
    parser.add_argument("--initialization-epochs", type=int, default=2_000)
    parser.add_argument("--initialization-lr", type=float, default=1.0e-3)
    parser.add_argument("--maximum-cap-fraction", type=float, default=0.25)
    parser.add_argument("--value-update", type=int, default=2_000)
    parser.add_argument("--value-seconds", type=float, default=1_800.0)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--checkpoint-seconds", type=float, default=1_800.0)
    parser.add_argument("--convergence-points", type=int, default=7)
    parser.add_argument(
        "--convergence-relative-tolerance", type=float, default=1.0e-3
    )
    parser.add_argument("--market-report", default=_DEFAULT_MARKET_REPORT)
    parser.add_argument("--warm-start-report", default=_DEFAULT_WARM_REPORT)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--checkpoint-root",
        default="/media/storage/tmp/quantdata/dysanos/oce-cached-expiry/checkpoints",
    )
    parser.add_argument(
        "--output-root",
        default="quantdata/dysanos/grid/rndysanos/reports/cached-expiry-oce",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparison": str(Path(args.output_root) / "comparison.json"),
                "variants": list(report["summaries"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
