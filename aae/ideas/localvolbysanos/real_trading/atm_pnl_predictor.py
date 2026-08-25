"""Predict next-session $1m strategy P&L from current and recent ATM volatility."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp

from .atm_loss_classifier import (
    FEATURE_NAMES,
    AtmLossClassifierConfig,
    build_atm_loss_panel,
)
from .investment_comparison import InvestmentComparisonConfig, _strategy_metrics


@dataclass(frozen=True, kw_only=True)
class AtmPnlPredictorConfig:
    """Expanding OLS and dollar P&L filter conventions."""

    median_window: int = 5
    minimum_training_days: int = 60
    capital_dollars: float = 1_000_000.0
    predicted_pnl_thresholds_dollars: tuple[float, ...] = (
        -5_000.0,
        -2_000.0,
        -1_000.0,
        -500.0,
        0.0,
    )
    tail_fraction: float = 0.10
    annual_sessions: int = 252

    def __post_init__(self) -> None:
        verify_inp(
            self.median_window >= 2,
            lambda: "median_window must be at least two",
        )
        verify_inp(
            self.minimum_training_days >= 30,
            lambda: "minimum_training_days must be at least 30",
        )
        verify_inp(
            np.isfinite(self.capital_dollars) and self.capital_dollars > 0.0,
            lambda: "capital_dollars must be finite and positive",
        )
        thresholds = np.asarray(self.predicted_pnl_thresholds_dollars)
        verify_inp(
            len(thresholds) >= 2
            and np.all(np.isfinite(thresholds))
            and np.all(np.diff(thresholds) > 0.0)
            and np.all(thresholds <= 0.0),
            lambda: (
                "predicted_pnl_thresholds_dollars must contain at least two "
                "strictly increasing, finite, nonpositive levels"
            ),
        )
        verify_inp(
            0.0 < self.tail_fraction <= 0.5,
            lambda: "tail_fraction must lie in (0, 0.5]",
        )
        verify_inp(
            self.annual_sessions >= 1,
            lambda: "annual_sessions must be positive",
        )


def expanding_pnl_predictions(
    pnl_dollars: np.ndarray,
    features: np.ndarray,
    *,
    minimum_training_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict P&L with expanding OLS using information before each date."""

    verify_inp(
        pnl_dollars.ndim == 1
        and features.ndim == 2
        and features.shape[1] == len(FEATURE_NAMES)
        and len(pnl_dollars) == len(features)
        and len(pnl_dollars) > minimum_training_days,
        lambda: "P&L and two-feature arrays are not aligned for expanding OLS",
    )
    predictions: list[float] = []
    baseline_predictions: list[float] = []
    coefficient_rows: list[np.ndarray] = []
    for index in range(minimum_training_days, len(pnl_dollars)):
        training_design = np.column_stack(
            (features[:index], np.ones(index, dtype=np.float64))
        )
        coefficients = np.linalg.lstsq(
            training_design,
            pnl_dollars[:index],
            rcond=None,
        )[0]
        prediction = np.concatenate((features[index], [1.0])) @ coefficients
        predictions.append(float(prediction))
        baseline_predictions.append(float(np.mean(pnl_dollars[:index])))
        coefficient_rows.append(coefficients)
    predictions_array = np.asarray(predictions)
    verify(
        np.all(np.isfinite(predictions_array)),
        lambda: "expanding ATM P&L predictions contain non-finite values",
    )
    return (
        predictions_array,
        np.asarray(baseline_predictions),
        np.vstack(coefficient_rows),
    )


def _prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    model_errors = actual - predicted
    baseline_errors = actual - baseline
    model_mse = float(np.mean(np.square(model_errors)))
    baseline_mse = float(np.mean(np.square(baseline_errors)))
    return {
        "prediction_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "rmse_dollars": float(np.sqrt(model_mse)),
        "baseline_rmse_dollars": float(np.sqrt(baseline_mse)),
        "mae_dollars": float(np.mean(np.abs(model_errors))),
        "baseline_mae_dollars": float(np.mean(np.abs(baseline_errors))),
        "out_of_sample_r_squared_vs_expanding_mean": 1.0
        - model_mse / baseline_mse,
        "mean_prediction_dollars": float(np.mean(predicted)),
        "mean_actual_pnl_dollars": float(np.mean(actual)),
    }


def _filter_trading_results(
    predictions: pl.DataFrame,
    *,
    config: AtmPnlPredictorConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    metric_config = InvestmentComparisonConfig(
        capital_dollars=config.capital_dollars,
        minimum_training_days=config.minimum_training_days,
        tail_fraction=config.tail_fraction,
        annual_sessions=config.annual_sessions,
    )
    path_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for objective, group in predictions.partition_by(
        "objective", as_dict=True
    ).items():
        group = group.sort("date")
        raw_returns = group["actual_pnl_dollars"].to_numpy() / config.capital_dollars
        forecast = group["predicted_pnl_dollars"].to_numpy()
        threshold_specs = (("always_on", float("-inf")),) + tuple(
            ("pnl_threshold", threshold)
            for threshold in config.predicted_pnl_thresholds_dollars
        )
        for approach, threshold in threshold_specs:
            invested = forecast >= threshold
            returns = raw_returns * invested
            cumulative_pnl = np.cumsum(returns) * config.capital_dollars
            traded_returns = raw_returns[invested]
            skipped_pnl = float(np.sum(raw_returns[~invested])) * config.capital_dollars
            summary_rows.append(
                {
                    "objective": objective[0],
                    "approach": approach,
                    "predicted_pnl_threshold_dollars": (
                        None if approach == "always_on" else threshold
                    ),
                    "num_evaluation_days": len(group),
                    "num_days_traded": int(np.sum(invested)),
                    "num_days_skipped": int(np.sum(~invested)),
                    "trade_fraction": float(np.mean(invested)),
                    "conditional_win_fraction": float(
                        np.mean(traded_returns > 0.0)
                    ),
                    "mean_abs_notional_dollars": float(np.mean(invested))
                    * config.capital_dollars,
                    "aggregate_skipped_pnl_dollars": skipped_pnl,
                    **_strategy_metrics(returns, config=metric_config),
                }
            )
            path_rows.extend(
                {
                    "objective": objective[0],
                    "approach": approach,
                    "predicted_pnl_threshold_dollars": (
                        None if approach == "always_on" else threshold
                    ),
                    "date": row["date"],
                    "predicted_pnl_dollars": row["predicted_pnl_dollars"],
                    "invested": bool(date_invested),
                    "spot_notional_dollars": (
                        config.capital_dollars if date_invested else 0.0
                    ),
                    "return": float(date_return),
                    "pnl_dollars": float(date_return * config.capital_dollars),
                    "cumulative_pnl_dollars": float(date_cumulative_pnl),
                }
                for row, date_invested, date_return, date_cumulative_pnl in zip(
                    group.to_dicts(),
                    invested,
                    returns,
                    cumulative_pnl,
                    strict=True,
                )
            )
    paths = pl.DataFrame(path_rows).sort(
        "objective", "approach", "predicted_pnl_threshold_dollars", "date"
    )
    summary = pl.DataFrame(summary_rows).sort(
        "objective", "approach", "predicted_pnl_threshold_dollars"
    )
    return paths, summary


def analyze_atm_pnl_predictor(
    legacy_comparison_file: Path,
    cvar_comparison_file: Path,
    daily_market_file: Path,
    output_dir: Path,
    *,
    config: AtmPnlPredictorConfig = AtmPnlPredictorConfig(),
) -> dict[str, object]:
    """Fit expanding ATM P&L models and test dollar loss cutoffs."""

    panel = build_atm_loss_panel(
        pl.read_parquet(legacy_comparison_file),
        pl.read_parquet(cvar_comparison_file),
        pl.read_parquet(daily_market_file),
        config=AtmLossClassifierConfig(
            median_window=config.median_window,
            minimum_training_days=config.minimum_training_days,
            capital_dollars=config.capital_dollars,
            tail_fraction=config.tail_fraction,
            annual_sessions=config.annual_sessions,
        ),
    ).filter(pl.col("variant") == "digital")
    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for objective, group in panel.partition_by(
        "objective", as_dict=True
    ).items():
        group = group.sort("date")
        pnl_dollars = (
            group["strategy_return"].to_numpy() * config.capital_dollars
        )
        features = group.select(*FEATURE_NAMES).to_numpy()
        predicted, baseline, coefficients = expanding_pnl_predictions(
            pnl_dollars,
            features,
            minimum_training_days=config.minimum_training_days,
        )
        evaluation = group.slice(config.minimum_training_days)
        actual = pnl_dollars[config.minimum_training_days :]
        summary_rows.append(
            {
                "objective": objective[0],
                "num_feature_observations": len(group),
                "num_evaluation_observations": len(evaluation),
                **_prediction_metrics(actual, predicted, baseline),
            }
        )
        prediction_rows.extend(
            {
                "objective": objective[0],
                "date": row["date"],
                "actual_pnl_dollars": float(date_actual),
                "predicted_pnl_dollars": float(date_predicted),
                "baseline_predicted_pnl_dollars": float(date_baseline),
                "current_atm_coefficient": float(date_coefficients[0]),
                "past_five_atm_median_coefficient": float(
                    date_coefficients[1]
                ),
                "intercept": float(date_coefficients[2]),
                "front_week_atm_volatility": row[
                    "front_week_atm_volatility"
                ],
                "past_five_atm_median": row["past_five_atm_median"],
            }
            for row, date_actual, date_predicted, date_baseline, date_coefficients in zip(
                evaluation.to_dicts(),
                actual,
                predicted,
                baseline,
                coefficients,
                strict=True,
            )
        )
    predictions = pl.DataFrame(prediction_rows).sort("objective", "date")
    prediction_summary = pl.DataFrame(summary_rows).sort("objective")
    trading_paths, trading_summary = _filter_trading_results(
        predictions,
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_dir / "atm_pnl_predictions.parquet")
    prediction_summary.write_parquet(
        output_dir / "atm_pnl_prediction_summary.parquet"
    )
    trading_paths.write_parquet(output_dir / "atm_pnl_filter_paths.parquet")
    trading_summary.write_parquet(
        output_dir / "atm_pnl_filter_summary.parquet"
    )
    report = {
        "legacy_comparison_file": str(legacy_comparison_file),
        "cvar_comparison_file": str(cvar_comparison_file),
        "daily_market_file": str(daily_market_file),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "model": (
            "expanding ordinary least squares of next-session signed $1m P&L "
            "on current ATM, the median ATM of the five preceding market dates, "
            "and an intercept"
        ),
        "validation": (
            "each forecast and coefficient vector uses only observations before "
            "the forecast date"
        ),
        "baseline": "expanding mean prior P&L",
        "trading_rule": (
            "hold cash when predicted P&L is below the specified dollar loss "
            "threshold; otherwise take the original signed $1m spot position"
        ),
        "threshold_selection": (
            "fixed dollar levels chosen to span the observed forecast range; "
            "not optimized using realized P&L and not independently validated"
        ),
        "methodology": (
            "P&L predictor timing is chronological; the underlying OCE state "
            "model remains the existing full-sample strategy diagnostic"
        ),
        "prediction_summary": prediction_summary.to_dicts(),
        "trading_summary": trading_summary.to_dicts(),
    }
    (output_dir / "atm_pnl_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("aae/ideas/localvolbysanos/real_trading/results")
    parser.add_argument(
        "--legacy-comparison-file",
        type=Path,
        default=base / "2025/one_day/spot_only_comparison.parquet",
    )
    parser.add_argument(
        "--cvar-comparison-file",
        type=Path,
        default=base / "2025_cvar_daily/one_day/spot_only_comparison.parquet",
    )
    parser.add_argument(
        "--daily-market-file",
        type=Path,
        default=Path(
            "aae/ideas/localvolbysanos/results/2025_full_reference_grid/"
            "daily_summary.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "2025/atm_pnl_predictor",
    )
    parser.add_argument("--minimum-training-days", type=int, default=60)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = analyze_atm_pnl_predictor(
        arguments.legacy_comparison_file,
        arguments.cvar_comparison_file,
        arguments.daily_market_file,
        arguments.output_dir,
        config=AtmPnlPredictorConfig(
            minimum_training_days=arguments.minimum_training_days
        ),
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
