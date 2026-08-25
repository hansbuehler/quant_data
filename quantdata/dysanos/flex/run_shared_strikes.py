"""Benchmark NMS strikes against the daily floating shared-strikes geometry."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from cdxcore.util import is_filename
from cdxcore.verbose import Context
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    NativeStateConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.lp import (
    DlvBehaviorStatistics,
    FlexLpConfig,
    dlv_behavior_statistics,
    fit_flex_density_lp,
)
from quantdata.dysanos.flex.market import FlexMarketDay, load_market_day
from quantdata.dysanos.flex.metrics import (
    expiry_excess_statistics,
    surface_fit_statistics,
)
from quantdata.dysanos.flex.training import fit_native_states

_DEFAULT_DATES = (date(2024, 3, 5), date(2025, 8, 29))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=[value.isoformat() for value in _DEFAULT_DATES],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("lp", "direct"),
        default=("lp", "direct"),
    )
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=2.0e-2)
    parser.add_argument("--fit-weight", type=float, default=0.05)
    parser.add_argument("--lp-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument(
        "--compile-sanos-solver",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--label", default="spx-two-date-shared-strikes-v2")
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _fit_statistics(
    days: tuple[FlexMarketDay, ...],
    fits: tuple[np.ndarray, ...],
) -> dict[str, object]:
    model_fit = np.concatenate(fits)
    bids = np.concatenate([day.bids for day in days])
    asks = np.concatenate([day.asks for day in days])
    market_fit = np.concatenate([day.fit_prices for day in days])
    business_dtes = np.concatenate(
        [day.quote_business_dtes for day in days]
    )
    return {
        "statistics": expiry_excess_statistics(
            model_fit,
            bids,
            asks,
            business_dtes,
        ).to_dict(),
        "surface_fit": surface_fit_statistics(
            model_fit,
            market_fit,
            bids,
            asks,
        ).to_dict(),
    }


def _day_fit_statistics(
    day: FlexMarketDay,
    fits: np.ndarray,
    *,
    fit_weight: float,
) -> dict[str, object]:
    outside = (
        np.maximum(fits - day.asks, 0.0)
        + np.maximum(day.bids - fits, 0.0)
    ) / day.half_spreads
    central = np.abs(fits - day.fit_prices) / day.half_spreads
    return {
        "objective": float(np.mean(outside + fit_weight * central)),
        "statistics": expiry_excess_statistics(
            fits,
            day.bids,
            day.asks,
            day.quote_business_dtes,
        ).to_dict(),
        "surface_fit": surface_fit_statistics(
            fits,
            day.fit_prices,
            day.bids,
            day.asks,
        ).to_dict(),
    }


def _mean_dlv_behavior(
    values: tuple[DlvBehaviorStatistics, ...],
) -> dict[str, int | float]:
    if not values:
        raise ValueError("At least one DLV diagnostic is required")
    integer_fields = frozenset(("num_nodes", "num_supported_nodes"))
    return {
        field: (
            int(sum(getattr(value, field) for value in values))
            if field in integer_fields
            else float(np.mean([getattr(value, field) for value in values]))
        )
        for field in values[0].__dataclass_fields__
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# FLEX shared-strikes two-date benchmark",
        "",
        f"Status: **{report['status']}**.",
        "",
        "`shared strikes` merges 100 front-expiry and 100 final-expiry "
        "canonical-NMS implied relative strikes. The spaced merged support "
        "is identical across the 14 expiries within a day and floats between "
        "days.",
        "",
    ]
    results = report.get("results", {})
    if not isinstance(results, dict):
        return "\n".join(lines)
    for stage in ("lp", "direct"):
        stage_results = results.get(stage)
        if not isinstance(stage_results, dict) or not stage_results:
            continue
        lines.extend(
            (
                f"## {stage.upper()}",
                "",
                "| Geometry | Options | Inside bid/ask | Positive excess "
                "mean | Positive excess median | Fit half-spread MAE |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for mode in ("nms_strikes", "shared_strikes"):
            result = stage_results.get(mode)
            if not isinstance(result, dict):
                continue
            aggregate = result["aggregate"]
            all_statistics = aggregate["statistics"]["all"]
            surface = aggregate["surface_fit"]
            lines.append(
                f"| {mode.replace('_', ' ')} | "
                f"{all_statistics['num_options']} | "
                f"{all_statistics['percent_zero']:.4f}% | "
                f"{all_statistics['positive_mean']:.6g} | "
                f"{all_statistics['positive_median']:.6g} | "
                f"{surface['half_spread_mae']:.6g} |"
            )
        lines.extend(
            (
                "",
                "DLV diagnostics use only inner nodes whose discrete density "
                "mass exceeds `1e-7`:",
                "",
                "| Geometry | Supported | At minimum | At maximum | Spatial "
                "log jump median / p90 | Expiry log change median / p90 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for mode in ("nms_strikes", "shared_strikes"):
            result = stage_results.get(mode)
            if not isinstance(result, dict):
                continue
            behavior = result["dlv_behavior"]
            lines.append(
                f"| {mode.replace('_', ' ')} | "
                f"{behavior['supported_percent']:.4f}% | "
                f"{behavior['minimum_bound_percent']:.4f}% | "
                f"{behavior['maximum_bound_percent']:.4f}% | "
                f"{behavior['spatial_log_jump_median']:.6g} / "
                f"{behavior['spatial_log_jump_p90']:.6g} | "
                f"{behavior['expiry_log_change_median']:.6g} / "
                f"{behavior['expiry_log_change_p90']:.6g} |"
            )
        lines.append("")
    return "\n".join(lines)


def _write_report(report: dict[str, object], *, path: Path) -> None:
    _atomic_write_json(report, path.with_suffix(".json"))
    _atomic_write_text(_markdown(report), path.with_suffix(".md"))


def _run_lp(
    days: tuple[FlexMarketDay, ...],
    *,
    geometry: CappedUnboundedDlvFlexGeometryConfig,
    objective_config: FlexObjectiveConfig,
    lp_config: FlexLpConfig,
    artifact_directory: Path,
    mode: str,
) -> dict[str, object]:
    fits: list[np.ndarray] = []
    behaviors: list[DlvBehaviorStatistics] = []
    day_results: list[dict[str, object]] = []
    for day in days:
        result = fit_flex_density_lp(
            day,
            geometry=geometry,
            objective_config=objective_config,
            config=lp_config,
            verbose=Context.quiet,
        )
        fits.append(result.fits)
        behaviors.append(result.dlv_behavior)
        artifact = artifact_directory / (
            f"lp-{mode}-{day.trading_date.isoformat()}.npz"
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            support_strikes=result.support_strikes,
            densities=result.densities,
            dlvsqs=result.dlvsqs,
            fits=result.fits,
        )
        day_results.append(
            {
                "date": day.trading_date.isoformat(),
                "status": result.status,
                "fell_back": result.fell_back,
                "elapsed_seconds": result.elapsed_seconds,
                "objective": result.objective,
                "statistics": result.statistics.to_dict(),
                "surface_fit": result.surface_fit.to_dict(),
                "dlv_behavior": result.dlv_behavior.to_dict(),
                "artifact": str(artifact),
            }
        )
    return {
        "aggregate": _fit_statistics(days, tuple(fits)),
        "dlv_behavior": _mean_dlv_behavior(tuple(behaviors)),
        "days": day_results,
    }


def _run_direct(
    days: tuple[FlexMarketDay, ...],
    *,
    geometry: CappedUnboundedDlvFlexGeometryConfig,
    objective_config: FlexObjectiveConfig,
    train_config: FlexTrainConfig,
    checkpoint_path: Path,
    mode: str,
) -> dict[str, object]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(checkpoint_path) + ".lock"):
        fit = fit_native_states(
            days,
            config=NativeStateConfig(
                label=f"shared-strikes-two-date-{mode}",
                group="flex/shared-strikes/two-date",
                geometry=geometry,
            ),
            objective_config=objective_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
        )
    indices = torch.arange(
        len(days),
        dtype=torch.long,
        device=fit.model.device,
    )
    with torch.no_grad():
        parameters, surfaces = fit.model.decode_days(indices)
    behaviors = tuple(
        dlv_behavior_statistics(
            parameters.dlvsqs[index].detach().cpu().numpy(),
            surfaces.densities[index].detach().cpu().numpy(),
            surfaces.support_strikes[index].detach().cpu().numpy(),
            geometry=geometry,
            density_mass_floor=1.0e-7,
        )
        for index in range(len(days))
    )
    day_results: list[dict[str, object]] = []
    for day, fits, behavior in zip(
        days,
        fit.evaluation.fits,
        behaviors,
        strict=True,
    ):
        result = _day_fit_statistics(
            day,
            fits,
            fit_weight=objective_config.fit_weight,
        )
        result.update(
            {
                "date": day.trading_date.isoformat(),
                "dlv_behavior": behavior.to_dict(),
            }
        )
        day_results.append(result)
    output = {
        "aggregate": {
            "objective": fit.evaluation.objective,
            "statistics": fit.evaluation.statistics.to_dict(),
            "surface_fit": fit.evaluation.surface_fit.to_dict(),
        },
        "dlv_behavior": _mean_dlv_behavior(behaviors),
        "days": day_results,
        "elapsed_seconds": fit.elapsed_seconds,
        "convergence": fit.convergence,
        "checkpoint": str(checkpoint_path),
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def main() -> None:
    """Run selected stages with durable reports after every geometry."""
    args = _parse_args()
    if not is_filename(args.label):
        raise ValueError(f"'label' must be a valid filename, found {args.label!r}")
    dates = tuple(date.fromisoformat(value) for value in args.dates)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("'dates' must be unique and strictly increasing")
    if args.device not in {"cuda:0", "cpu"}:
        raise ValueError("This experiment is restricted to 'cuda:0' or CPU")

    nms_geometry = CappedUnboundedDlvFlexGeometryConfig()
    shared_geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    market_config = FlexMarketConfig()
    objective_config = FlexObjectiveConfig(fit_weight=args.fit_weight)
    loaded = tuple(
        load_market_day(
            args.ticker,
            trading_date,
            geometry=nms_geometry,
            config=market_config,
            verbose=Context.quiet,
        )
        for trading_date in dates
    )
    if any(day is None for day in loaded):
        raise RuntimeError("At least one requested benchmark date has no market")
    days = tuple(day for day in loaded if day is not None)

    output_directory = args.output_directory or (
        Path(quantdata_env.locations.training.path)
        / "flex-shared-strikes"
        / "two-date-benchmark"
        / args.label
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / args.label
    lp_config = FlexLpConfig(timeout_seconds=args.lp_timeout_seconds)
    train_config = FlexTrainConfig(
        num_epochs=args.epochs,
        day_batch_size=len(days),
        learning_rate=args.learning_rate,
        precision=args.precision,
        device=args.device,
        checkpoint_every=25,
        report_every=25,
        compile_sanos_solver=args.compile_sanos_solver,
    )
    report: dict[str, object] = {
        "schema": "flex-shared-strikes-two-date-v2",
        "status": "running",
        "ticker": args.ticker,
        "dates": [value.isoformat() for value in dates],
        "quote_counts": [day.num_quotes for day in days],
        "objective_config": asdict(objective_config),
        "lp_config": asdict(lp_config),
        "train_config": asdict(train_config),
        "geometry": {
            "num_model_expiries": nms_geometry.num_model_expiries,
            "num_inner_strikes": nms_geometry.num_inner_strikes,
            "nms_range": list(nms_geometry.nms_range),
            "shared_candidate_strikes_per_endpoint": (
                shared_geometry.num_inner_strikes // 2
            ),
            "minimum_log_spacing_fraction": (
                shared_geometry.minimum_log_spacing_fraction
            ),
            "shared_within_day": True,
            "floats_between_days": True,
        },
        "results": {},
    }
    _write_report(report, path=report_path)
    geometries = {
        "nms_strikes": nms_geometry,
        "shared_strikes": shared_geometry,
    }
    for stage in args.stages:
        stage_results: dict[str, object] = {}
        report["results"][stage] = stage_results
        for mode, geometry in geometries.items():
            report["status"] = f"{stage}_{mode}_running"
            _write_report(report, path=report_path)
            if stage == "lp":
                result = _run_lp(
                    days,
                    geometry=geometry,
                    objective_config=objective_config,
                    lp_config=lp_config,
                    artifact_directory=output_directory / "artifacts",
                    mode=mode,
                )
            else:
                result = _run_direct(
                    days,
                    geometry=geometry,
                    objective_config=objective_config,
                    train_config=train_config,
                    checkpoint_path=(
                        output_directory / "checkpoints" / f"direct-{mode}.pt"
                    ),
                    mode=mode,
                )
            stage_results[mode] = result
            report["status"] = f"{stage}_{mode}_complete"
            _write_report(report, path=report_path)
    report["status"] = "complete"
    _write_report(report, path=report_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run_guarded_cli(main, program="flex-shared-strikes")
