"""Explain independent one-day spot-signal losses with volatility indicators."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp
from scipy.stats import norm, t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from quantdata.biz_calendar import DAYS_PER_YEAR

from .attribution import SERIES_KEYS

REQUESTED_FEATURES = (
    "next_day_spot_return_squared",
    "atm_variance_one_day",
    "today_spot_return",
    "today_spot_return_squared",
)
EX_ANTE_FEATURES = REQUESTED_FEATURES[1:]


@dataclass(frozen=True, kw_only=True)
class LossDiagnosticsConfig:
    """Inference, time-validation, and binning conventions."""

    minimum_training_days: int = 60
    logistic_l2_c: float = 1.0
    num_indicator_bins: int = 5

    def __post_init__(self) -> None:
        verify_inp(
            self.minimum_training_days >= 30,
            lambda: "minimum_training_days must be at least 30",
        )
        verify_inp(
            np.isfinite(self.logistic_l2_c) and self.logistic_l2_c > 0.0,
            lambda: "logistic_l2_c must be finite and positive",
        )
        verify_inp(
            self.num_indicator_bins >= 2,
            lambda: "num_indicator_bins must be at least two",
        )


def build_loss_observations(
    comparison: pl.DataFrame,
    daily_market: pl.DataFrame,
) -> pl.DataFrame:
    """Create one row per independent sign trade with dated explanatory data."""

    dated_market = (
        daily_market.select("date", "spot_ref", "front_week_atm_volatility")
        .unique("date")
        .sort("date")
        .with_columns(
            (pl.col("spot_ref") / pl.col("spot_ref").shift(1) - 1.0).alias(
                "today_spot_return"
            ),
            pl.col("front_week_atm_volatility")
            .pow(2)
            .alias("atm_variance_annualized"),
        )
        .with_columns(
            pl.col("today_spot_return")
            .pow(2)
            .alias("today_spot_return_squared"),
            (pl.col("atm_variance_annualized") / DAYS_PER_YEAR).alias(
                "atm_variance_one_day"
            ),
        )
    )
    observations = (
        comparison.filter(pl.col("spot_complete"))
        .join(dated_market, on="date", how="inner", validate="m:1")
        .with_columns(
            pl.col("horizon_spot_return")
            .pow(2)
            .alias("next_day_spot_return_squared"),
            pl.col("unit_gross_spot_return").alias("strategy_return"),
            (pl.col("unit_gross_spot_return") < 0.0)
            .cast(pl.Int8)
            .alias("loss"),
            pl.col("net_entry_delta").sign().cast(pl.Int8).alias("signal_sign"),
        )
        .filter(pl.col("today_spot_return").is_not_null())
        .sort(*SERIES_KEYS, "date")
    )
    verify(
        observations.select(
            pl.all_horizontal(
                pl.col(
                    "strategy_return",
                    *REQUESTED_FEATURES,
                ).is_finite()
            ).all()
        ).item(),
        lambda: "daily loss observation panel contains non-finite values",
    )
    return observations


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0, ddof=0)
    verify(
        np.all(np.isfinite(means))
        and np.all(np.isfinite(scales))
        and np.all(scales > 0.0),
        lambda: f"one or more diagnostic features has zero variance: {scales}",
    )
    return (features - means) / scales, means, scales


def _ols_return_diagnostics(
    returns: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit OLS with HC1 heteroskedasticity-robust standard errors."""

    standardized, _, _ = _standardize(features)
    design = np.column_stack((np.ones(len(returns)), standardized))
    coefficients = np.linalg.lstsq(design, returns, rcond=None)[0]
    residual = returns - design @ coefficients
    inverse_information = np.linalg.inv(design.T @ design)
    score = design * residual[:, None]
    covariance = inverse_information @ (score.T @ score) @ inverse_information
    covariance *= len(returns) / (len(returns) - design.shape[1])
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0.0,
    )
    centered = returns - np.mean(returns)
    total_variation = float(centered @ centered)
    residual_variation = float(residual @ residual)
    metrics = {
        "r_squared": 1.0 - residual_variation / total_variation,
        "adjusted_r_squared": 1.0
        - (residual_variation / (len(returns) - design.shape[1]))
        / (total_variation / (len(returns) - 1)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }
    return coefficients, standard_errors, statistics, metrics


def _logistic_loss_diagnostics(
    loss: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    standardized, _, _ = _standardize(features)
    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=2_000,
    ).fit(standardized, loss)
    probabilities = model.predict_proba(standardized)[:, 1]
    coefficients = np.concatenate((model.intercept_, model.coef_[0]))
    design = np.column_stack((np.ones(len(loss)), standardized))
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0.0,
    )
    return coefficients, standard_errors, statistics, probabilities


def _classification_metrics(
    loss: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predicted = probabilities >= 0.5
    return {
        "auc": float(roc_auc_score(loss, probabilities)),
        "accuracy": float(accuracy_score(loss, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(loss, predicted)),
        "brier_score": float(brier_score_loss(loss, probabilities)),
        "log_loss": float(log_loss(loss, probabilities, labels=[0, 1])),
    }


def _expanding_logistic_probabilities(
    loss: np.ndarray,
    features: np.ndarray,
    *,
    config: LossDiagnosticsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    verify_inp(
        len(loss) > config.minimum_training_days,
        lambda: (
            f"need more than {config.minimum_training_days} observations for "
            "expanding classification"
        ),
    )
    probabilities: list[float] = []
    baseline_probabilities: list[float] = []
    for index in range(config.minimum_training_days, len(loss)):
        training_features = features[:index]
        means = np.mean(training_features, axis=0)
        scales = np.std(training_features, axis=0, ddof=0)
        verify(
            np.all(scales > 0.0),
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


def _indicator_bins(
    observations: pl.DataFrame,
    *,
    config: LossDiagnosticsConfig,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in observations.partition_by(*SERIES_KEYS, as_dict=True).items():
        loss = group["loss"].to_numpy()
        strategy_return = group["strategy_return"].to_numpy()
        for feature_name in REQUESTED_FEATURES:
            values = group[feature_name].to_numpy()
            order = np.argsort(values, kind="stable")
            bins = np.empty(len(values), dtype=np.int64)
            bins[order] = np.minimum(
                np.arange(len(values)) * config.num_indicator_bins // len(values),
                config.num_indicator_bins - 1,
            )
            for bin_index in range(config.num_indicator_bins):
                selected = bins == bin_index
                rows.append(
                    {
                        "universe": key[0],
                        "utility": key[1],
                        "risk_aversion": key[2],
                        "feature": feature_name,
                        "bin": bin_index,
                        "num_observations": int(np.sum(selected)),
                        "mean_feature": float(np.mean(values[selected])),
                        "loss_fraction": float(np.mean(loss[selected])),
                        "mean_strategy_return": float(
                            np.mean(strategy_return[selected])
                        ),
                    }
                )
    return pl.DataFrame(rows).sort(*SERIES_KEYS, "feature", "bin")


def analyze_daily_losses(
    comparison_file: Path,
    daily_market_file: Path,
    output_dir: Path,
    *,
    config: LossDiagnosticsConfig = LossDiagnosticsConfig(),
) -> dict[str, object]:
    """Fit return regressions and loss classifiers for all daily sign strategies."""

    observations = build_loss_observations(
        pl.read_parquet(comparison_file),
        pl.read_parquet(daily_market_file),
    )
    coefficient_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    feature_sets = {
        **{
            f"individual_{feature_name}": (feature_name,)
            for feature_name in REQUESTED_FEATURES
        },
        "requested_with_future_magnitude": REQUESTED_FEATURES,
        "ex_ante_only": EX_ANTE_FEATURES,
    }
    for key, group in observations.partition_by(*SERIES_KEYS, as_dict=True).items():
        group = group.sort("date")
        returns = group["strategy_return"].to_numpy()
        loss = group["loss"].to_numpy()
        for feature_set_name, feature_names in feature_sets.items():
            features = group.select(*feature_names).to_numpy()
            ols_coef, ols_se, ols_stat, ols_metrics = _ols_return_diagnostics(
                returns,
                features,
            )
            logit_coef, logit_se, logit_stat, fitted_probabilities = (
                _logistic_loss_diagnostics(loss, features)
            )
            expanding_probabilities, baseline_probabilities = (
                _expanding_logistic_probabilities(
                    loss,
                    features,
                    config=config,
                )
            )
            expanding_loss = loss[config.minimum_training_days :]
            in_sample_metrics = _classification_metrics(
                loss,
                fitted_probabilities,
            )
            expanding_metrics = _classification_metrics(
                expanding_loss,
                expanding_probabilities,
            )
            baseline_metrics = _classification_metrics(
                expanding_loss,
                baseline_probabilities,
            )
            summary_rows.append(
                {
                    "universe": key[0],
                    "utility": key[1],
                    "risk_aversion": key[2],
                    "feature_set": feature_set_name,
                    "num_observations": len(group),
                    "loss_fraction": float(np.mean(loss)),
                    **{f"ols_{name}": value for name, value in ols_metrics.items()},
                    **{
                        f"classifier_in_sample_{name}": value
                        for name, value in in_sample_metrics.items()
                    },
                    **{
                        f"classifier_expanding_{name}": value
                        for name, value in expanding_metrics.items()
                    },
                    **{
                        f"baseline_expanding_{name}": value
                        for name, value in baseline_metrics.items()
                    },
                }
            )
            coefficient_names = ("intercept", *feature_names)
            ols_degrees_freedom = len(group) - len(ols_coef)
            for index, coefficient_name in enumerate(coefficient_names):
                coefficient_rows.extend(
                    (
                        {
                            "universe": key[0],
                            "utility": key[1],
                            "risk_aversion": key[2],
                            "feature_set": feature_set_name,
                            "model": "ols_daily_return",
                            "feature": coefficient_name,
                            "coefficient": float(ols_coef[index]),
                            "standard_error": float(ols_se[index]),
                            "statistic": float(ols_stat[index]),
                            "p_value": float(
                                2.0
                                * t.sf(
                                    abs(float(ols_stat[index])),
                                    df=ols_degrees_freedom,
                                )
                            ),
                            "coefficient_units": (
                                "daily return per one feature standard deviation"
                                if index > 0
                                else "daily return intercept"
                            ),
                        },
                        {
                            "universe": key[0],
                            "utility": key[1],
                            "risk_aversion": key[2],
                            "feature_set": feature_set_name,
                            "model": "logit_loss",
                            "feature": coefficient_name,
                            "coefficient": float(logit_coef[index]),
                            "standard_error": float(logit_se[index]),
                            "statistic": float(logit_stat[index]),
                            "p_value": float(
                                2.0 * norm.sf(abs(float(logit_stat[index])))
                            ),
                            "coefficient_units": (
                                "log odds of loss per one feature standard deviation"
                                if index > 0
                                else "log odds intercept"
                            ),
                        },
                    )
                )

    coefficients = pl.DataFrame(coefficient_rows).sort(
        *SERIES_KEYS,
        "feature_set",
        "model",
        "feature",
    )
    summary = pl.DataFrame(summary_rows).sort(*SERIES_KEYS, "feature_set")
    bins = _indicator_bins(observations, config=config)

    output_dir.mkdir(parents=True, exist_ok=True)
    observations.write_parquet(output_dir / "daily_loss_observations.parquet")
    coefficients.write_parquet(output_dir / "daily_loss_coefficients.parquet")
    summary.write_parquet(output_dir / "daily_loss_summary.parquet")
    bins.write_parquet(output_dir / "daily_loss_indicator_bins.parquet")
    report = {
        "comparison_file": str(comparison_file),
        "daily_market_file": str(daily_market_file),
        "output_dir": str(output_dir),
        "outcome_regression": "signed next-day SPX return of the sign-only strategy",
        "outcome_classification": "one when signed next-day SPX return is negative",
        "requested_future_feature": (
            "next_day_spot_return_squared is known only after the trade and is an "
            "ex-post magnitude diagnostic, not a tradable predictor"
        ),
        "atm_variance": (
            "front-week ATM implied volatility squared and divided by "
            f"DAYS_PER_YEAR={DAYS_PER_YEAR}"
        ),
        "today_return": (
            "same-date spot_ref divided by the preceding panel date spot_ref minus one"
        ),
        "regression_errors": "HC1 heteroskedasticity-robust",
        "classifier_validation": (
            f"expanding window after {config.minimum_training_days} observations; "
            "L2 logistic regularization only in the expanding fit"
        ),
        "methodology": "full-sample strategy diagnostic; not a look-ahead-free backtest",
        "summary": summary.to_dicts(),
    }
    (output_dir / "daily_loss_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("aae/ideas/localvolbysanos/real_trading/results/2025")
    parser.add_argument(
        "--comparison-file",
        type=Path,
        default=base / "one_day/spot_only_comparison.parquet",
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
        default=base / "loss_diagnostics",
    )
    parser.add_argument("--minimum-training-days", type=int, default=60)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = analyze_daily_losses(
        arguments.comparison_file,
        arguments.daily_market_file,
        arguments.output_dir,
        config=LossDiagnosticsConfig(
            minimum_training_days=arguments.minimum_training_days,
        ),
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
