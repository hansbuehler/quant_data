"""Fit and report the CPU-only Phase 3 half-spread attribution benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.verbose import Context
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from quantdata.dysanos.flex.config import FlexMarketConfig, ForwardMuFlexGeometryConfig
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import load_market_day, market_process_count
from quantdata.dysanos.flex.spread_cost import (
    FittedSpreadCostModel,
    SpreadAttributionConfig,
    SpreadObservations,
    equal_date_sample_weights,
    fit_spread_cost_models,
    observations_from_market_day,
    save_spread_cost_model,
)
from quantdata.pool import PoolConfig

TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 8, 29)
DEFAULT_DATES_REPORT = Path(
    "quantdata/dysanos/flex/reports/batch-descent/rebalanced-stronger/"
    "spx-2022-2025-forward-mu-v1/spx-2022-2025-forward-mu-v1.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path("quantdata/dysanos/flex/reports/phase3-spread")


def _load_observations(
    ticker: str,
    trading_date: date,
    *,
    geometry: ForwardMuFlexGeometryConfig,
    market_config: FlexMarketConfig,
) -> SpreadObservations | None:
    day = load_market_day(
        ticker,
        trading_date,
        geometry=geometry,
        config=market_config,
        verbose=Context.quiet,
    )
    return (
        None
        if day is None
        else observations_from_market_day(
            day,
            first_model_business_dte=geometry.first_model_business_dte,
        )
    )


def _dates_from_report(path: Path) -> tuple[date, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_dates = payload.get("market_dates")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError(f"No market_dates in {path}")
    dates = tuple(date.fromisoformat(value) for value in raw_dates)
    if dates != tuple(sorted(set(dates))):
        raise ValueError(f"market_dates must be unique and increasing in {path}")
    if dates[0].year < 2022 or dates[-1] > VALIDATION_END:
        raise ValueError(
            f"Expected only 2022 through {VALIDATION_END} data, found "
            f"{dates[0]} through {dates[-1]}"
        )
    if any(value.year >= 2026 for value in dates):
        raise ValueError("Phase 3 spread attribution must never read 2026 data")
    return dates


def _stack(
    observations: Sequence[SpreadObservations],
) -> dict[str, np.ndarray]:
    if len(observations) == 0:
        raise ValueError("Cannot stack an empty split")
    date_index = np.concatenate(
        [np.full(value.num_quotes, index) for index, value in enumerate(observations)]
    )
    return {
        "half_spread": np.concatenate([value.half_spread for value in observations]),
        "fit_price": np.concatenate([value.fit_price for value in observations]),
        "business_dte": np.concatenate([value.business_dte for value in observations]),
        "sqrt_tte": np.concatenate([value.sqrt_tte for value in observations]),
        "strike": np.concatenate([value.strike for value in observations]),
        "log_strike": np.concatenate([value.log_strike for value in observations]),
        "nms": np.concatenate([value.nms for value in observations]),
        "vega": np.concatenate([value.vega for value in observations]),
        "delta": np.concatenate([value.delta for value in observations]),
        "is_call": np.concatenate([value.is_call for value in observations]),
        "spot_ref": np.concatenate(
            [np.full(value.num_quotes, value.spot_ref) for value in observations]
        ),
        "date_index": date_index,
    }


def _predict(model: FittedSpreadCostModel, data: dict[str, np.ndarray]) -> np.ndarray:
    return model.predict_half_spread(
        vega=data["vega"],
        sqrt_tte=data["sqrt_tte"],
        nms=data["nms"],
        log_strike=data["log_strike"],
        delta=data["delta"],
    )


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    date_index: np.ndarray,
) -> dict[str, float | int]:
    """Return quote-pooled and equal-date regression metrics."""
    weights = equal_date_sample_weights(date_index)
    return {
        "num_quotes": int(len(target)),
        "num_dates": int(len(np.unique(date_index))),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
        "date_balanced_mae": float(
            mean_absolute_error(target, prediction, sample_weight=weights)
        ),
        "date_balanced_rmse": float(
            np.sqrt(mean_squared_error(target, prediction, sample_weight=weights))
        ),
        "date_balanced_r2": float(
            r2_score(target, prediction, sample_weight=weights)
        ),
        "prediction_zero_percent": float(np.mean(prediction == 0.0) * 100.0),
    }


def _quantiles(value: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)
    return {
        f"q{int(level * 100):02d}": float(result)
        for level, result in zip(levels, np.quantile(value, levels), strict=True)
    }


def _proportional_diagnostics(
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    *,
    price_floor: float,
) -> dict[str, object]:
    fit_price = data["fit_price"]
    denominator = np.maximum(fit_price, price_floor)
    actual = data["half_spread"] / denominator
    predicted = prediction / denominator
    itm = np.where(data["is_call"], data["strike"] <= 1.0, data["strike"] >= 1.0)

    def summarize(mask: np.ndarray) -> dict[str, object]:
        return {
            "num_quotes": int(np.sum(mask)),
            "actual_half_spread_over_fit": _quantiles(actual[mask]),
            "predicted_half_spread_over_fit": _quantiles(predicted[mask]),
            "absolute_ratio_error": _quantiles(np.abs(predicted[mask] - actual[mask])),
        }

    return {
        "price_floor": price_floor,
        "price_floor_active_percent": float(np.mean(fit_price < price_floor) * 100.0),
        "all": summarize(np.ones(len(fit_price), dtype=bool)),
        "itm_default_universe": summarize(itm),
    }


def _calibration_rows(
    target: np.ndarray,
    prediction: np.ndarray,
    date_index: np.ndarray,
    value: np.ndarray,
    edges: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    weights = equal_date_sample_weights(date_index)
    rows: list[dict[str, float | int | str]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (value >= lower) & (value < upper)
        if not np.any(mask):
            continue
        local_weights = weights[mask]
        local_weights /= np.sum(local_weights)
        rows.append(
            {
                "range": f"[{lower:g},{upper:g})",
                "num_quotes": int(np.sum(mask)),
                "target_mean": float(np.average(target[mask], weights=local_weights)),
                "prediction_mean": float(
                    np.average(prediction[mask], weights=local_weights)
                ),
                "date_balanced_mae": float(
                    np.average(np.abs(prediction[mask] - target[mask]), weights=local_weights)
                ),
            }
        )
    return rows


def _model_report(
    model: FittedSpreadCostModel,
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
) -> dict[str, object]:
    train_prediction = _predict(model, train_data)
    validation_prediction = _predict(model, validation_data)
    target = validation_data["half_spread"]
    date_index = validation_data["date_index"]
    report: dict[str, object] = {
        "feature_names": list(model.feature_names),
        "train": regression_metrics(
            train_data["half_spread"], train_prediction, train_data["date_index"]
        ),
        "validation": regression_metrics(target, validation_prediction, date_index),
        "validation_proportional_cost": _proportional_diagnostics(
            validation_data,
            validation_prediction,
            price_floor=model.proportional_price_floor,
        ),
        "validation_calibration": {
            "business_dte": _calibration_rows(
                target,
                validation_prediction,
                date_index,
                validation_data["business_dte"],
                (-np.inf, 5.0, 10.0, 20.0, 40.0, np.inf),
            ),
            "nms": _calibration_rows(
                target,
                validation_prediction,
                date_index,
                validation_data["nms"],
                (-np.inf, -2.0, -1.0, 0.0, 1.0, np.inf),
            ),
            "signed_delta": _calibration_rows(
                target,
                validation_prediction,
                date_index,
                validation_data["delta"],
                (-np.inf, -0.75, -0.25, 0.0, 0.25, 0.75, np.inf),
            ),
        },
    }
    if model.name == "linear":
        report["intercept"] = float(model.estimator.intercept_)
        report["coefficients"] = {
            name: float(value)
            for name, value in zip(
                model.feature_names, model.estimator.coef_, strict=True
            )
        }
    return report


def _markdown(report: dict[str, object]) -> str:
    counts = report["data"]
    lines = [
        "# Phase 3 half-spread attribution",
        "",
        "This CPU-only benchmark predicts normalized price half-spread. Every date has ",
        "equal total training weight, so quote-rich dates do not dominate. The temporal ",
        "split is frozen at training through 2024-12-31 and validation from 2025-01-02 ",
        "through 2025-08-29; no 2026 data is read.",
        "",
        "## Data",
        "",
        f"- Training: {counts['train_dates']} dates and {counts['train_quotes']:,} books.",
        f"- Validation: {counts['validation_dates']} dates and "
        f"{counts['validation_quotes']:,} books.",
        "- Calls and puts are separate real books. Prices and spreads are normalized by ",
        "  discounted forward in the upstream market pipeline.",
        "- Greeks use the smooth AF fitted call, with puts converted through normalized ",
        "  parity. Delta is `Phi(d1)` for calls and `Phi(d1)-1` for puts.",
        "",
        "## Temporal out-of-sample results",
        "",
        "| model | date-balanced MAE | date-balanced RMSE | date-balanced R2 | raw MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, model_report in report["models"].items():
        metrics = model_report["validation"]
        lines.append(
            f"| {name} | {metrics['date_balanced_mae']:.8g} | "
            f"{metrics['date_balanced_rmse']:.8g} | {metrics['date_balanced_r2']:.6f} | "
            f"{metrics['mae']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Model choices and limitations",
            "",
            "The linear design is intercept plus `vega/sqrt(TTE)`, signed delta, and NMS. ",
            "The nonlinear control is a deterministic histogram gradient booster using ",
            "vega, `sqrt(TTE)`, NMS, log strike, and signed delta. Its depth/leaf count, ",
            "learning rate, iteration count, minimum leaf size, and L2 penalty are ad-hoc ",
            "Phase 3 choices, not validation-selected hyperparameters. Squared-error loss ",
            "targets expected half-spread rather than a conditional median.",
            "",
            "The model predicts half-spread first and the simulator divides by current ",
            "option price. Reported proportional ratios therefore become unstable for ",
            "near-zero option values; the JSON records the explicit price floor and robust ",
            "quantiles. Zero-strike calls are spot positions and bypass this model in favor ",
            "of a separately configured spot cost. The fitted linear formula has a native, ",
            "gradient-preserving Torch inference method; the tree model remains a CPU ",
            "attribution benchmark because hard tree splits have no useful state gradient. ",
            "The model is an attribution benchmark, ",
            "not an execution guarantee, and 2025 is used only for temporal evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    *,
    ticker: str,
    dates_report: Path,
    output_directory: Path,
    workers: int,
    config: SpreadAttributionConfig,
) -> dict[str, object]:
    dates = _dates_from_report(dates_report)
    geometry = ForwardMuFlexGeometryConfig()
    market_config = FlexMarketConfig()
    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()
    jobs = {
        trading_date: pool.delayed(_load_observations)(
            ticker,
            trading_date,
            geometry=geometry,
            market_config=market_config,
        )
        for trading_date in dates
    }
    observations: list[SpreadObservations] = []
    for index, result in enumerate(pool.parallel(jobs), start=1):
        trading_date, value = result
        if value is None:
            raise RuntimeError(f"Missing previously valid market date {trading_date}")
        observations.append(value)
        if index % 25 == 0 or index == len(dates):
            print(
                f"spread market {index}/{len(dates)} date={value.trading_date} "
                f"quotes={value.num_quotes}",
                flush=True,
            )
    observations.sort(key=lambda value: value.trading_date)
    train = tuple(value for value in observations if value.trading_date <= TRAIN_END)
    validation = tuple(
        value
        for value in observations
        if VALIDATION_START <= value.trading_date <= VALIDATION_END
    )
    if len(train) + len(validation) != len(observations):
        raise RuntimeError("Loaded observations fall outside the frozen temporal split")

    print(
        f"fit spread models train_dates={len(train)} "
        f"train_quotes={sum(value.num_quotes for value in train)}",
        flush=True,
    )
    models = fit_spread_cost_models(train, config=config)
    train_data = _stack(train)
    validation_data = _stack(validation)
    report: dict[str, object] = {
        "schema": "flex-phase3-spread-attribution-v1",
        "ticker": ticker,
        "status": "complete",
        "source_dates_report": str(dates_report),
        "source_pipeline": (
            "flex.market.load_market_day -> dysanos.market.smooth_interpolated_market; "
            "ArbFreePricer.from_dict is used by FLEX eligibility"
        ),
        "split": {
            "train_start": train[0].trading_date.isoformat(),
            "train_end": train[-1].trading_date.isoformat(),
            "validation_start": validation[0].trading_date.isoformat(),
            "validation_end": validation[-1].trading_date.isoformat(),
            "no_2026": True,
        },
        "date_weighting": (
            "each quote receives Q/(D*n_quotes_on_date), so weights have mean one "
            "and every date has equal total mass"
        ),
        "target": "normalized price half-spread = (ask-bid)/2",
        "delta_convention": (
            "side-specific zero-carry forward delta: Phi(d1) for calls, "
            "Phi(d1)-1 for puts"
        ),
        "simulation_api": {
            "torch_native": "linear model only; preserves feature gradients",
            "cost_formula": "predicted_half_spread / max(option_price, price_floor)",
            "zero_strike_call": "bypass model and use separately configured spot cost",
        },
        "config": asdict(config),
        "data": {
            "all_dates": len(observations),
            "all_quotes": sum(value.num_quotes for value in observations),
            "train_dates": len(train),
            "train_quotes": sum(value.num_quotes for value in train),
            "validation_dates": len(validation),
            "validation_quotes": sum(value.num_quotes for value in validation),
            "year_date_counts": {
                str(year): sum(value.trading_date.year == year for value in observations)
                for year in range(2022, 2026)
            },
        },
        "models": {
            name: _model_report(model, train_data, validation_data)
            for name, model in models.items()
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        save_spread_cost_model(model, output_directory / f"{name}.joblib")
    json_path = output_directory / "phase3-spread.json"
    markdown_path = output_directory / "phase3-spread.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {json_path} and {markdown_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--dates-report", type=Path, default=DEFAULT_DATES_REPORT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    run_benchmark(
        ticker=args.ticker,
        dates_report=args.dates_report,
        output_directory=args.output_directory,
        workers=args.workers,
        config=SpreadAttributionConfig(),
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_spread_benchmark")
