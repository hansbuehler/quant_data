"""Fit 25 daily risk-neutral marginals on a shared 400-strike grid."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.util import is_filename
from cdxcore.verbose import Context

from quantdata import env as quantdata_env
from quantdata.biz_calendar import NMS
from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.lp import FlexLpConfig, fit_flex_density_lp
from quantdata.dysanos.flex.market import load_market_day, market_process_count
from quantdata.dysanos.flex.metrics import (
    expiry_excess_statistics,
    nominal_expiry_excess_statistics,
    surface_fit_statistics,
)

_MODEL_BUSINESS_DTES = tuple(float(value) for value in range(1, 26))
_VARIANT_ANNUAL_VOLS = {
    "zero-smoothing": 0.0,
    "fixed-5pct-smoothing": 0.05,
}


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


def _atomic_write_npz(payload: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_dates(path: Path) -> tuple[date, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("market_dates")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Report has no non-empty 'market_dates': {path}")
    dates = tuple(date.fromisoformat(str(value)) for value in values)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("Market dates must be unique and strictly increasing")
    return dates


def _geometry(
    num_inner_strikes: int,
) -> SharedStrikeCappedUnboundedDlvFlexGeometryConfig:
    return SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=num_inner_strikes,
        nms_range=(-2.5, 1.5),
        num_model_expiries=25,
        first_model_business_dte=1.0,
        model_expiry_step_business_days=1.0,
        explicit_model_business_dtes=_MODEL_BUSINESS_DTES,
    )


def _quote_nms(day: object, *, geometry: object) -> np.ndarray:
    expiry_index = day.expiry_index
    epsilon_annual_vol = day.model_atm_total_vols[0] / np.sqrt(
        geometry.model_expiries_y[0]
    )
    return np.asarray(
        NMS.to_nms(
            day.strikes,
            t=day.expiries_y[expiry_index],
            vol=day.atm_annual_vols[expiry_index],
            eps_vol=epsilon_annual_vol,
        ),
        dtype=np.float64,
    )


def _fit_date(
    trading_date: date,
    *,
    ticker: str,
    output_directory: Path,
    lp_timeout_seconds: float,
    solver_algorithms: dict[str, str],
    variants_to_fit: tuple[str, ...],
    num_inner_strikes: int,
) -> dict[str, object]:
    geometry = _geometry(num_inner_strikes)
    market_config = FlexMarketConfig(excluded_business_dtes=(0.2,))
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    day = load_market_day(
        ticker,
        trading_date,
        geometry=geometry,
        config=market_config,
        verbose=Context.quiet,
    )
    if day is None:
        return {"date": trading_date.isoformat(), "missing": True}

    quote_nms = _quote_nms(day, geometry=geometry)
    audit_mask = (quote_nms >= -2.0) & (quote_nms <= 1.0)
    variants: dict[str, object] = {}
    initial_densities: np.ndarray | None = None
    for variant in variants_to_fit:
        annual_vol = _VARIANT_ANNUAL_VOLS[variant]
        solver_algorithm = solver_algorithms[variant]
        artifact_path = (
            output_directory
            / "artifacts"
            / variant
            / f"{trading_date.isoformat()}.npz"
        )
        if artifact_path.exists():
            with np.load(artifact_path, allow_pickle=False) as artifact:
                fingerprint = str(artifact["training_fingerprint"].item())
                if fingerprint != day.training_fingerprint:
                    raise RuntimeError(
                        f"Stale artifact fingerprint for {trading_date}: "
                        f"{artifact_path}"
                    )
                fits = np.asarray(artifact["fits"], dtype=np.float64)
                fitted_densities = np.asarray(
                    artifact["densities"], dtype=np.float64
                )
                elapsed_seconds = float(artifact["elapsed_seconds"].item())
                objective = float(artifact["objective"].item())
                status = str(artifact["status"].item())
                fell_back = bool(artifact["fell_back"].item())
            resumed = True
        else:
            lp_config = FlexLpConfig(
                backbone_mu=None,
                fixed_backbone_annual_vol=annual_vol,
                solver_algorithm=solver_algorithm,
                central_fit_target="book_mid",
                enforce_dlv_bounds=False,
                timeout_seconds=lp_timeout_seconds,
            )
            result = fit_flex_density_lp(
                day,
                geometry=geometry,
                objective_config=objective_config,
                config=lp_config,
                initial_densities=initial_densities,
                verbose=Context.quiet,
            )
            fits = result.fits
            fitted_densities = result.densities
            elapsed_seconds = result.elapsed_seconds
            objective = result.objective
            status = result.status
            fell_back = result.fell_back
            _atomic_write_npz(
                {
                    "schema": np.asarray("flex-daily-25-lp-v1"),
                    "training_fingerprint": np.asarray(
                        day.training_fingerprint
                    ),
                    "annual_backbone_vol": np.asarray(annual_vol),
                    "support_strikes": result.support_strikes,
                    "densities": result.densities,
                    "dlvsqs": result.dlvsqs,
                    "fits": result.fits,
                    "elapsed_seconds": np.asarray(result.elapsed_seconds),
                    "objective": np.asarray(result.objective),
                    "status": np.asarray(result.status),
                    "fell_back": np.asarray(result.fell_back),
                },
                artifact_path,
            )
            resumed = False
        initial_densities = fitted_densities
        variants[variant] = {
            "fits": fits,
            "elapsed_seconds": elapsed_seconds,
            "objective": objective,
            "status": status,
            "fell_back": fell_back,
            "resumed": resumed,
            "artifact": str(artifact_path),
        }

    return {
        "date": trading_date.isoformat(),
        "missing": False,
        "num_calibration_quotes": day.num_quotes,
        "audit_mask": audit_mask,
        "bids": day.bids,
        "asks": day.asks,
        "market_mids": 0.5 * (day.bids + day.asks),
        "smooth_reference": day.fit_prices,
        "business_dtes": day.quote_business_dtes,
        "variants": variants,
    }


def _aggregate_variant(
    results: list[dict[str, object]],
    *,
    variant: str,
) -> dict[str, object]:
    masks = [np.asarray(result["audit_mask"], dtype=bool) for result in results]
    fits = np.concatenate(
        [
            np.asarray(result["variants"][variant]["fits"])[mask]
            for result, mask in zip(results, masks, strict=True)
        ]
    )
    bids = np.concatenate(
        [np.asarray(result["bids"])[mask] for result, mask in zip(results, masks, strict=True)]
    )
    asks = np.concatenate(
        [np.asarray(result["asks"])[mask] for result, mask in zip(results, masks, strict=True)]
    )
    market_mids = np.concatenate(
        [
            np.asarray(result["market_mids"])[mask]
            for result, mask in zip(results, masks, strict=True)
        ]
    )
    smooth_reference = np.concatenate(
        [
            np.asarray(result["smooth_reference"])[mask]
            for result, mask in zip(results, masks, strict=True)
        ]
    )
    business_dtes = np.concatenate(
        [
            np.asarray(result["business_dtes"])[mask]
            for result, mask in zip(results, masks, strict=True)
        ]
    )
    elapsed = np.asarray(
        [result["variants"][variant]["elapsed_seconds"] for result in results],
        dtype=np.float64,
    )
    return {
        "annual_backbone_vol": _VARIANT_ANNUAL_VOLS[variant],
        "num_audit_quotes": int(len(fits)),
        "bid_ask": expiry_excess_statistics(
            fits,
            bids,
            asks,
            business_dtes,
        ).to_dict(),
        "by_nominal_dte": list(
            nominal_expiry_excess_statistics(
                fits,
                bids,
                asks,
                business_dtes,
            )
        ),
        "market_mid_fit": surface_fit_statistics(
            fits,
            market_mids,
            bids,
            asks,
        ).to_dict(),
        "smooth_reference_fit": surface_fit_statistics(
            fits,
            smooth_reference,
            bids,
            asks,
        ).to_dict(),
        "solver": {
            "total_elapsed_seconds": float(np.sum(elapsed)),
            "median_elapsed_seconds": float(np.median(elapsed)),
            "p90_elapsed_seconds": float(np.quantile(elapsed, 0.9)),
            "fallback_days": int(
                sum(result["variants"][variant]["fell_back"] for result in results)
            ),
            "resumed_days": int(
                sum(result["variants"][variant]["resumed"] for result in results)
            ),
        },
    }


def _write_markdown(report: dict[str, object], path: Path) -> None:
    lines = [
        "# Daily 25-day shared-grid LP calibration",
        "",
        f"Status: **{report['status']}**.",
        "",
        f"The {report['geometry']['num_inner_strikes']} inner relative strikes "
        "merge front-day and 25th-day canonical-NMS candidates over "
        "`[-2.5, 1.5]`. The resulting "
        "grid is fixed across all 25 expiries within a market date and floats "
        "between dates with that date's ATM curve.",
        "",
        "The table audits observed call and put books in `[-2, 1]` NMS "
        "against their actual bid/ask and midpoint.",
        "",
        "| Backbone | Days | Quotes | Inside bid/ask | Mean excess | "
        "Midpoint MAE (half-spreads) | Normalized-price RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    aggregate = report.get("aggregate", {})
    if isinstance(aggregate, dict):
        for variant in _VARIANT_ANNUAL_VOLS:
            value = aggregate.get(variant)
            if not isinstance(value, dict):
                continue
            all_excess = value["bid_ask"]["all"]
            mid = value["market_mid_fit"]
            lines.append(
                f"| {variant} | {report['completed_days']} | "
                f"{value['num_audit_quotes']} | "
                f"{all_excess['percent_zero']:.4f}% | "
                f"{all_excess['surface_mean']:.6g} | "
                f"{mid['half_spread_mae']:.6g} | "
                f"{mid['normalized_price_rmse']:.6g} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--market-dates-report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--num-inner-strikes", type=int, default=400)
    parser.add_argument("--lp-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--zero-solver-algorithm",
        choices=(
            "simplex",
            "ipm",
            "clarabel",
            "direct-simplex",
            "direct-ipm",
        ),
        default="simplex",
    )
    parser.add_argument(
        "--fixed5-solver-algorithm",
        choices=(
            "simplex",
            "ipm",
            "clarabel",
            "direct-simplex",
            "direct-ipm",
        ),
        default="direct-simplex",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(_VARIANT_ANNUAL_VOLS),
        default=tuple(_VARIANT_ANNUAL_VOLS),
    )
    parser.add_argument("--label", default="spx-2022-2025-daily25-shared400-v1")
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    """Fit both backbone specifications and write a resumable audit."""
    arguments = _parse_args()
    if not is_filename(arguments.label):
        raise ValueError(f"Invalid label: {arguments.label!r}")
    if arguments.workers <= 0:
        raise ValueError("'workers' must be positive")
    if arguments.max_dates is not None and arguments.max_dates <= 0:
        raise ValueError("'max_dates' must be positive")
    if arguments.num_inner_strikes <= 0:
        raise ValueError("'num_inner_strikes' must be positive")
    dates = _load_dates(arguments.market_dates_report)
    if arguments.max_dates is not None:
        dates = dates[: arguments.max_dates]
    output_directory = arguments.output_directory or (
        Path(quantdata_env.locations.training.path)
        / "flex-shared-strikes"
        / "daily-25-lp"
        / arguments.label
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / f"{arguments.label}.json"
    markdown_path = report_path.with_suffix(".md")
    geometry = _geometry(arguments.num_inner_strikes)
    solver_algorithms = {
        "zero-smoothing": arguments.zero_solver_algorithm,
        "fixed-5pct-smoothing": arguments.fixed5_solver_algorithm,
    }
    variants_to_fit = tuple(dict.fromkeys(arguments.variants))
    report: dict[str, object] = {
        "schema": "flex-daily-25-lp-v1",
        "status": "running",
        "ticker": arguments.ticker,
        "requested_days": len(dates),
        "completed_days": 0,
        "missing_dates": [],
        "failed_dates": [],
        "market_dates_report": str(arguments.market_dates_report),
        "geometry": asdict(geometry),
        "market_config": asdict(
            FlexMarketConfig(excluded_business_dtes=(0.2,))
        ),
        "objective_config": asdict(FlexObjectiveConfig(fit_weight=0.05)),
        "central_fit_target": "book_mid",
        "enforce_dlv_bounds": False,
        "calibration_nms_range": [-2.5, 1.5],
        "audit_nms_range": [-2.0, 1.0],
        "backbone_annual_vols": {
            variant: _VARIANT_ANNUAL_VOLS[variant]
            for variant in variants_to_fit
        },
        "solver_algorithms": {
            variant: solver_algorithms[variant]
            for variant in variants_to_fit
        },
    }
    _atomic_write_json(report, report_path)

    completed: list[dict[str, object]] = []
    worker_count = market_process_count(arguments.workers)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fit_date,
                trading_date,
                ticker=arguments.ticker,
                output_directory=output_directory,
                lp_timeout_seconds=arguments.lp_timeout_seconds,
                solver_algorithms=solver_algorithms,
                variants_to_fit=variants_to_fit,
                num_inner_strikes=arguments.num_inner_strikes,
            ): trading_date
            for trading_date in dates
        }
        for index, future in enumerate(as_completed(futures), start=1):
            trading_date = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                report["failed_dates"].append(
                    {
                        "date": trading_date.isoformat(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(
                    f"daily25 {index}/{len(dates)} date={trading_date} "
                    f"failed={type(error).__name__}: {error}",
                    flush=True,
                )
            else:
                if result["missing"]:
                    report["missing_dates"].append(trading_date.isoformat())
                    print(
                        f"daily25 {index}/{len(dates)} date={trading_date} "
                        "missing",
                        flush=True,
                    )
                else:
                    completed.append(result)
                    report["completed_days"] = len(completed)
                    print(
                        f"daily25 {index}/{len(dates)} date={trading_date} "
                        f"quotes={result['num_calibration_quotes']}",
                        flush=True,
                    )
            if index % 5 == 0 or index == len(dates):
                _atomic_write_json(report, report_path)

    completed.sort(key=lambda value: str(value["date"]))
    if completed:
        report["aggregate"] = {
            variant: _aggregate_variant(completed, variant=variant)
            for variant in variants_to_fit
        }
    report["market_dates"] = [result["date"] for result in completed]
    report["status"] = (
        "complete"
        if len(completed) == len(dates)
        else "complete_with_missing_or_failed_dates"
    )
    _atomic_write_json(report, report_path)
    _write_markdown(report, markdown_path)
    print(json.dumps(report.get("aggregate", {}), indent=2), flush=True)


if __name__ == "__main__":
    run_guarded_cli(main, program="flex-daily-25-lp")
