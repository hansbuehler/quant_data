"""Classify next-session strategy losses from current and recent ATM volatility."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .investment_comparison import (
    OBJECTIVE_SPECS,
    InvestmentComparisonConfig,
    _strategy_metrics,
)

STRATEGY_RETURN_COLUMNS = {
    "continuous": "spot_only_return",
    "digital": "unit_gross_spot_return",
}
FEATURE_NAMES = (
    "front_week_atm_volatility",
    "past_five_atm_median",
)


@dataclass(frozen=True, kw_only=True)
class AtmLossClassifierConfig:
    """Chronological feature and logistic-classifier conventions."""

    median_window: int = 5
    minimum_training_days: int = 60
    logistic_l2_c: float = 1.0
    probability_threshold: float = 0.5
    capital_dollars: float = 1_000_000.0
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
            np.isfinite(self.logistic_l2_c) and self.logistic_l2_c > 0.0,
            lambda: "logistic_l2_c must be finite and positive",
        )
        verify_inp(
            0.0 < self.probability_threshold < 1.0,
            lambda: "probability_threshold must lie in (0, 1)",
        )
        verify_inp(
            np.isfinite(self.capital_dollars) and self.capital_dollars > 0.0,
            lambda: "capital_dollars must be finite and positive",
        )
        verify_inp(
            0.0 < self.tail_fraction <= 0.5,
            lambda: "tail_fraction must lie in (0, 0.5]",
        )
        verify_inp(
            self.annual_sessions >= 1,
            lambda: "annual_sessions must be positive",
        )


def build_atm_loss_panel(
    legacy_comparison: pl.DataFrame,
    cvar_comparison: pl.DataFrame,
    daily_market: pl.DataFrame,
    *,
    config: AtmLossClassifierConfig = AtmLossClassifierConfig(),
) -> pl.DataFrame:
    """Build the two-feature loss-classification panel for six strategies."""

    required = (
        "date",
        "universe",
        "utility",
        "risk_aversion",
        "spot_complete",
        *STRATEGY_RETURN_COLUMNS.values(),
    )
    comparison = pl.concat(
        (
            legacy_comparison.select(*required),
            cvar_comparison.select(*required),
        ),
        how="vertical_relaxed",
    )
    market = (
        daily_market.select("date", "front_week_atm_volatility")
        .unique("date")
        .sort("date")
        .with_columns(
            pl.col("front_week_atm_volatility")
            .shift(1)
            .rolling_median(window_size=config.median_window)
            .alias("past_five_atm_median")
        )
    )
    base = (
        comparison.filter(
            (pl.col("universe") == "normalised")
            & pl.col("spot_complete")
            & pl.any_horizontal(
                *[
                    (pl.col("utility") == utility)
                    & (pl.col("risk_aversion") == risk_aversion)
                    for utility, risk_aversion, _ in OBJECTIVE_SPECS
                ]
            )
        )
        .join(market, on="date", how="inner", validate="m:1")
        .filter(pl.col("past_five_atm_median").is_not_null())
    )
    frames: list[pl.DataFrame] = []
    for utility, risk_aversion, objective in OBJECTIVE_SPECS:
        objective_frame = base.filter(
            (pl.col("utility") == utility)
            & (pl.col("risk_aversion") == risk_aversion)
        )
        for variant, return_column in STRATEGY_RETURN_COLUMNS.items():
            frames.append(
                objective_frame.select(
                    "date",
                    pl.lit(objective).alias("objective"),
                    "utility",
                    "risk_aversion",
                    pl.lit(variant).alias("variant"),
                    pl.col(return_column).alias("strategy_return"),
                    *FEATURE_NAMES,
                ).with_columns(
                    (pl.col("strategy_return") < 0.0)
                    .cast(pl.Int8)
                    .alias("loss")
                )
            )
    panel = pl.concat(frames).sort("objective", "variant", "date")
    verify(
        panel.select(
            pl.all_horizontal(
                pl.col("strategy_return", *FEATURE_NAMES).is_finite()
            ).all()
        ).item(),
        lambda: "ATM loss-classification panel contains non-finite values",
    )
    return panel


def expanding_loss_probabilities(
    loss: np.ndarray,
    features: np.ndarray,
    *,
    config: AtmLossClassifierConfig = AtmLossClassifierConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Fit each prediction using only observations strictly before its date."""

    verify_inp(
        loss.ndim == 1
        and features.ndim == 2
        and len(loss) == len(features)
        and len(loss) > config.minimum_training_days,
        lambda: (
            "loss and features must be aligned arrays with more than "
            f"{config.minimum_training_days} rows"
        ),
    )
    probabilities: list[float] = []
    baseline_probabilities: list[float] = []
    for index in range(config.minimum_training_days, len(loss)):
        training_features = features[:index]
        means = np.mean(training_features, axis=0)
        scales = np.std(training_features, axis=0, ddof=0)
        verify(
            np.all(np.isfinite(scales)) and np.all(scales > 0.0),
            lambda: f"zero expanding-window feature scale at index {index}",
        )
        model = LogisticRegression(
            C=config.logistic_l2_c,
            solver="lbfgs",
            max_iter=2_000,
        ).fit((training_features - means) / scales, loss[:index])
        probability = model.predict_proba(
            ((features[index] - means) / scales)[None, :]
        )[0, 1]
        probabilities.append(float(probability))
        baseline_probabilities.append(float(np.mean(loss[:index])))
    return np.asarray(probabilities), np.asarray(baseline_probabilities)


def _classification_metrics(
    loss: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    return {
        "auc": float(roc_auc_score(loss, probabilities)),
        "accuracy": float(accuracy_score(loss, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(loss, predicted)),
        "brier_score": float(brier_score_loss(loss, probabilities)),
        "log_loss": float(log_loss(loss, probabilities, labels=[0, 1])),
        "num_predicted_losses": int(np.sum(predicted)),
    }


def build_classifier_trading_results(
    predictions: pl.DataFrame,
    *,
    config: AtmLossClassifierConfig = AtmLossClassifierConfig(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build always-on and classifier-filtered $1m digital trading ledgers."""

    digital = predictions.filter(pl.col("variant") == "digital").sort(
        "objective", "date"
    )
    metric_config = InvestmentComparisonConfig(
        capital_dollars=config.capital_dollars,
        minimum_training_days=config.minimum_training_days,
        tail_fraction=config.tail_fraction,
        annual_sessions=config.annual_sessions,
    )
    path_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for objective, group in digital.partition_by("objective", as_dict=True).items():
        group = group.sort("date")
        unfiltered_returns = group["strategy_return"].to_numpy()
        classifier_invested = ~group["predicted_loss"].to_numpy()
        for approach, invested in (
            ("always_on", np.ones(len(group), dtype=bool)),
            ("classifier_filter", classifier_invested),
        ):
            returns = unfiltered_returns * invested
            cumulative_pnl = np.cumsum(returns) * config.capital_dollars
            traded_returns = unfiltered_returns[invested]
            summary_rows.append(
                {
                    "objective": objective[0],
                    "approach": approach,
                    "num_evaluation_days": len(group),
                    "num_days_traded": int(np.sum(invested)),
                    "trade_fraction": float(np.mean(invested)),
                    "conditional_win_fraction": float(
                        np.mean(traded_returns > 0.0)
                    ),
                    "mean_abs_notional_dollars": float(np.mean(invested))
                    * config.capital_dollars,
                    **_strategy_metrics(returns, config=metric_config),
                }
            )
            path_rows.extend(
                {
                    "objective": objective[0],
                    "approach": approach,
                    "date": row["date"],
                    "loss_probability": row["loss_probability"],
                    "predicted_loss": row["predicted_loss"],
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
    paths = pl.DataFrame(path_rows).sort("objective", "approach", "date")
    summary = pl.DataFrame(summary_rows).sort("objective", "approach")
    return paths, summary


def analyze_atm_loss_classifier(
    legacy_comparison_file: Path,
    cvar_comparison_file: Path,
    daily_market_file: Path,
    output_dir: Path,
    *,
    config: AtmLossClassifierConfig = AtmLossClassifierConfig(),
) -> dict[str, object]:
    """Run and persist the expanding two-feature loss classifier."""

    panel = build_atm_loss_panel(
        pl.read_parquet(legacy_comparison_file),
        pl.read_parquet(cvar_comparison_file),
        pl.read_parquet(daily_market_file),
        config=config,
    )
    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for key, group in panel.partition_by(
        "objective",
        "utility",
        "risk_aversion",
        "variant",
        as_dict=True,
    ).items():
        group = group.sort("date")
        loss = group["loss"].to_numpy()
        features = group.select(*FEATURE_NAMES).to_numpy()
        probabilities, baseline_probabilities = expanding_loss_probabilities(
            loss,
            features,
            config=config,
        )
        evaluation = group.slice(config.minimum_training_days)
        evaluation_loss = evaluation["loss"].to_numpy()
        metrics = _classification_metrics(
            evaluation_loss,
            probabilities,
            threshold=config.probability_threshold,
        )
        baseline_metrics = _classification_metrics(
            evaluation_loss,
            baseline_probabilities,
            threshold=config.probability_threshold,
        )

        means = np.mean(features, axis=0)
        scales = np.std(features, axis=0, ddof=0)
        full_sample_model = LogisticRegression(
            C=config.logistic_l2_c,
            solver="lbfgs",
            max_iter=2_000,
        ).fit((features - means) / scales, loss)
        standardized_coefficients = full_sample_model.coef_[0]
        raw_coefficients = standardized_coefficients / scales
        raw_intercept = float(
            full_sample_model.intercept_[0] - raw_coefficients @ means
        )
        coefficient_rows.append(
            {
                "objective": key[0],
                "utility": key[1],
                "risk_aversion": key[2],
                "variant": key[3],
                "feature": "intercept",
                "raw_logit_coefficient": raw_intercept,
                "standardized_logit_coefficient": float(
                    full_sample_model.intercept_[0]
                ),
            }
        )
        for feature_name, raw_coefficient, standardized_coefficient in zip(
            FEATURE_NAMES,
            raw_coefficients,
            standardized_coefficients,
            strict=True,
        ):
            coefficient_rows.append(
                {
                    "objective": key[0],
                    "utility": key[1],
                    "risk_aversion": key[2],
                    "variant": key[3],
                    "feature": feature_name,
                    "raw_logit_coefficient": float(raw_coefficient),
                    "standardized_logit_coefficient": float(
                        standardized_coefficient
                    ),
                }
            )

        summary_rows.append(
            {
                "objective": key[0],
                "utility": key[1],
                "risk_aversion": key[2],
                "variant": key[3],
                "num_feature_observations": len(group),
                "num_evaluation_observations": len(evaluation),
                "evaluation_loss_fraction": float(np.mean(evaluation_loss)),
                **{f"classifier_{name}": value for name, value in metrics.items()},
                **{
                    f"baseline_{name}": value
                    for name, value in baseline_metrics.items()
                },
                "brier_score_improvement": (
                    float(baseline_metrics["brier_score"])
                    - float(metrics["brier_score"])
                ),
            }
        )
        prediction_rows.extend(
            {
                **row,
                "loss_probability": float(probability),
                "baseline_loss_probability": float(baseline_probability),
                "predicted_loss": bool(
                    probability >= config.probability_threshold
                ),
            }
            for row, probability, baseline_probability in zip(
                evaluation.to_dicts(),
                probabilities,
                baseline_probabilities,
                strict=True,
            )
        )

    predictions = pl.DataFrame(prediction_rows).sort(
        "objective", "variant", "date"
    )
    summary = pl.DataFrame(summary_rows).sort("objective", "variant")
    coefficients = pl.DataFrame(coefficient_rows).sort(
        "objective", "variant", "feature"
    )
    trading_paths, trading_summary = build_classifier_trading_results(
        predictions,
        config=config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(output_dir / "atm_loss_panel.parquet")
    predictions.write_parquet(output_dir / "atm_loss_predictions.parquet")
    summary.write_parquet(output_dir / "atm_loss_summary.parquet")
    coefficients.write_parquet(output_dir / "atm_loss_coefficients.parquet")
    trading_paths.write_parquet(
        output_dir / "atm_classifier_trading_paths.parquet"
    )
    trading_summary.write_parquet(
        output_dir / "atm_classifier_trading_summary.parquet"
    )
    report = {
        "legacy_comparison_file": str(legacy_comparison_file),
        "cvar_comparison_file": str(cvar_comparison_file),
        "daily_market_file": str(daily_market_file),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "outcome": "one when the next-session raw strategy return is negative",
        "features": {
            "front_week_atm_volatility": "current decision-date ATM volatility",
            "past_five_atm_median": (
                "median ATM volatility of the five preceding available market "
                "dates, excluding the decision date"
            ),
        },
        "validation": (
            "expanding logistic regression; each probability uses only earlier "
            "labels and feature rows"
        ),
        "baseline": "expanding frequency of losses before the prediction date",
        "trading_rule": (
            "take the original signed $1m digital spot position when predicted "
            "loss probability is below the configured threshold; otherwise hold "
            "cash; P&L is retained and not reinvested"
        ),
        "methodology": (
            "classifier timing is chronological; the underlying OCE state model "
            "remains the existing full-sample strategy diagnostic"
        ),
        "summary": summary.to_dicts(),
        "trading_summary": trading_summary.to_dicts(),
    }
    (output_dir / "atm_loss_report.json").write_text(
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
        default=base / "2025/atm_loss_classifier",
    )
    parser.add_argument("--minimum-training-days", type=int, default=60)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = analyze_atm_loss_classifier(
        arguments.legacy_comparison_file,
        arguments.cvar_comparison_file,
        arguments.daily_market_file,
        arguments.output_dir,
        config=AtmLossClassifierConfig(
            minimum_training_days=arguments.minimum_training_days
        ),
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
