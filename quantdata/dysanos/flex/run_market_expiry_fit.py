"""Reprice completed front/term-eta models by nominal market expiry."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.verbose import Context

from quantdata.dysanos.flex.config import (
    FlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    NativeStateConfig,
    PcaDecoderConfig,
)
from quantdata.dysanos.flex.market import (
    FlexMarketDay,
    load_market_day,
    market_process_count,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.metrics import (
    excess_statistics,
    nominal_expiry_excess_statistics,
    normalized_excess,
)
from quantdata.dysanos.flex.training import fit_latent_states, fit_native_states
from quantdata.pool import PoolConfig


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--day-batch-size", type=int, default=128)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tuple_fields(values: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    result = dict(values)
    for name in names:
        if result.get(name) is not None:
            result[name] = tuple(result[name])
    return result


def _load_completion(
    path: Path,
) -> tuple[
    dict[str, Any],
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    tuple[date, ...],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "native_complete":
        raise ValueError(f"native completion is not complete: {path}")
    if payload.get("geometry", {}).get("front_term_eta") is not True:
        raise ValueError("expiry reprice requires the front/term-eta geometry")
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        **_tuple_fields(
            payload["geometry"],
            ("nms_range", "explicit_model_business_dtes"),
        )
    )
    market_config = FlexMarketConfig(
        **_tuple_fields(payload["market_config"], ("excluded_business_dtes",))
    )
    objective_config = FlexObjectiveConfig(**payload["objective_config"])
    dates = tuple(date.fromisoformat(value) for value in payload["market_dates"])
    if dates != tuple(sorted(set(dates))):
        raise ValueError("completion market dates must be unique and sorted")
    return payload, geometry, market_config, objective_config, dates


def _market_cache_geometry(
    geometry: FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
) -> FlexGeometryConfig:
    """Match the market cache identity used by production fitting."""
    return FlexGeometryConfig(
        **{
            field.name: getattr(geometry, field.name)
            for field in fields(FlexGeometryConfig)
        }
    )


def _load_market_job(
    ticker: str,
    trading_date: date,
    *,
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
) -> FlexMarketDay | None:
    return load_market_day(
        ticker,
        trading_date,
        geometry=geometry,
        config=market_config,
        verbose=Context.quiet,
    )


def _load_market_days(
    *,
    ticker: str,
    dates: tuple[date, ...],
    geometry: FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    market_config: FlexMarketConfig,
    workers: int,
) -> tuple[FlexMarketDay, ...]:
    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()
    cache_geometry = _market_cache_geometry(geometry)
    jobs = {
        trading_date: pool.delayed(_load_market_job)(
            ticker,
            trading_date,
            geometry=cache_geometry,
            market_config=market_config,
        )
        for trading_date in dates
    }
    results = dict(pool.parallel(jobs))
    missing = [value.isoformat() for value in dates if results[value] is None]
    if missing:
        raise RuntimeError(
            "completed market universe no longer loads for dates: "
            + ", ".join(missing[:10])
        )
    days = tuple(results[value] for value in dates)
    if any(day is None for day in days):
        raise RuntimeError("market loader returned an unexpected empty day")
    return tuple(day for day in days if day is not None)


def _expiry_report(
    *,
    mode: str,
    evaluation: Any,
    days: tuple[FlexMarketDay, ...],
    source_report: Path,
    source_checkpoint: Path,
    excluded_business_dtes: tuple[float, ...],
) -> dict[str, object]:
    fits = np.concatenate(evaluation.fits)
    bids = np.concatenate([day.bids for day in days])
    asks = np.concatenate([day.asks for day in days])
    business_dtes = np.concatenate([day.quote_business_dtes for day in days])
    all_statistics = excess_statistics(normalized_excess(fits, bids, asks))
    by_expiry = nominal_expiry_excess_statistics(
        fits,
        bids,
        asks,
        business_dtes,
    )
    if sum(int(row["num_quotes"]) for row in by_expiry) != len(fits):
        raise RuntimeError("nominal-expiry buckets do not partition all quotes")
    return {
        "schema": "flex-market-expiry-fit-v1",
        "status": "complete",
        "model": mode,
        "source_report": str(source_report.resolve()),
        "source_report_sha256": _sha256(source_report),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "excluded_business_dtes": list(excluded_business_dtes),
        "all": {
            "num_quotes": all_statistics.num_options,
            "percent_inside_bid_ask": all_statistics.percent_zero,
            "mean_excess": all_statistics.surface_mean,
            "conditional_mean_excess": all_statistics.positive_mean,
            "conditional_median_excess": all_statistics.positive_median,
        },
        "by_nominal_market_dte": list(by_expiry),
    }


def main() -> None:
    arguments = _arguments()
    if arguments.workers <= 0:
        raise ValueError("'workers' must be positive")
    if arguments.day_batch_size <= 0:
        raise ValueError("'day_batch_size' must be positive")
    (
        completion,
        geometry,
        market_config,
        objective_config,
        dates,
    ) = _load_completion(arguments.native_completion)
    ticker = str(completion["ticker"])
    label = str(completion["label"])
    days = _load_market_days(
        ticker=ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        workers=arguments.workers,
    )
    train_config = FlexTrainConfig(
        num_epochs=1,
        day_batch_size=arguments.day_batch_size,
        device=arguments.device,
        precision=int(completion["train_config"]["precision"]),
        just_load=True,
    )
    native_checkpoint = arguments.checkpoint_directory / "native.pt"
    affine_checkpoint = arguments.checkpoint_directory / "pca-200.pt"
    native_fit = fit_native_states(
        days,
        config=NativeStateConfig(label=f"{label}-native", geometry=geometry),
        objective_config=objective_config,
        train_config=train_config,
        checkpoint_path=native_checkpoint,
    )
    affine_fit = fit_latent_states(
        days,
        native_fit,
        config=PcaDecoderConfig(
            label=f"{label}-pca-200",
            state_size=200,
            geometry=geometry,
        ),
        objective_config=objective_config,
        supervised_train_config=train_config,
        market_train_config=train_config,
        checkpoint_path=affine_checkpoint,
    )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for filename, mode, evaluation, checkpoint in (
        ("market-expiry-fit.json", "native", native_fit.evaluation, native_checkpoint),
        (
            "market-expiry-fit-affine-200.json",
            "affine-200",
            affine_fit.evaluation,
            affine_checkpoint,
        ),
    ):
        report = _expiry_report(
            mode=mode,
            evaluation=evaluation,
            days=days,
            source_report=arguments.native_completion,
            source_checkpoint=checkpoint,
            excluded_business_dtes=market_config.excluded_business_dtes,
        )
        write_json_atomic(report, arguments.output_directory / filename)
        print(json.dumps(report["all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
