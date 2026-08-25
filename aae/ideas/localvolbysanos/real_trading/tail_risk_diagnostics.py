"""Test entry-time overlays against realized one-day lower-tail risk."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp

from .attribution import SERIES_KEYS
from .experiment import _empirical_lower_tail_mean

REQUIRED_COMPARISON_COLUMNS = (
    "date",
    *SERIES_KEYS,
    "spot_complete",
    "spot_only_return",
    "net_entry_delta",
    "return_std",
    "return_p05",
)
RISK_TARGET_SIGNALS = {
    "atm_volatility": "front_week_atm_volatility",
    "today_absolute_return": "today_absolute_return",
    "scenario_return_std": "return_std",
    "scenario_downside_p05": "scenario_downside_p05",
}
AGREEMENT_SIGNAL = "cross_objective_sign_agreement"


@dataclass(frozen=True, kw_only=True)
class TailRiskDiagnosticsConfig:
    """Chronological overlay and lower-tail evaluation conventions."""

    minimum_training_days: int = 60
    tail_fraction: float = 0.10
    bootstrap_samples: int = 5_000
    bootstrap_block_days: int = 5
    annual_sessions: int = 252
    seed: int = 19_870_211

    def __post_init__(self) -> None:
        verify_inp(
            self.minimum_training_days >= 30,
            lambda: "minimum_training_days must be at least 30",
        )
        verify_inp(
            0.0 < self.tail_fraction <= 0.5,
            lambda: "tail_fraction must lie in (0, 0.5]",
        )
        verify_inp(
            self.bootstrap_samples >= 100,
            lambda: "bootstrap_samples must be at least 100",
        )
        verify_inp(
            self.bootstrap_block_days >= 1,
            lambda: "bootstrap_block_days must be positive",
        )
        verify_inp(
            self.annual_sessions >= 1,
            lambda: "annual_sessions must be positive",
        )


def build_tail_risk_panel(
    legacy_comparison: pl.DataFrame,
    cvar_comparison: pl.DataFrame,
    daily_market: pl.DataFrame,
) -> pl.DataFrame:
    """Combine capped OCE results with entry-time market and agreement signals."""

    panel = pl.concat(
        (
            legacy_comparison.filter(
                pl.col("utility").is_in(("entropy", "powexp"))
            ).select(*REQUIRED_COMPARISON_COLUMNS),
            cvar_comparison.filter(pl.col("utility") == "cvar10").select(
                *REQUIRED_COMPARISON_COLUMNS
            ),
        ),
        how="vertical_relaxed",
    )
    dated_market = (
        daily_market.select("date", "spot_ref", "front_week_atm_volatility")
        .unique("date")
        .sort("date")
        .with_columns(
            (pl.col("spot_ref") / pl.col("spot_ref").shift(1) - 1.0)
            .abs()
            .alias("today_absolute_return")
        )
    )
    panel = (
        panel.filter(pl.col("spot_complete"))
        .join(dated_market, on="date", how="inner", validate="m:1")
        .filter(pl.col("today_absolute_return").is_not_null())
        .with_columns(
            pl.col("net_entry_delta").sign().alias("strategy_sign"),
            pl.max_horizontal(-pl.col("return_p05"), 1.0e-8).alias(
                "scenario_downside_p05"
            ),
        )
    )
    agreement = panel.group_by("date", "universe").agg(
        pl.len().alias("num_objectives"),
        (pl.col("strategy_sign") > 0.0).sum().alias("num_long"),
        (pl.col("strategy_sign") < 0.0).sum().alias("num_short"),
        (pl.col("strategy_sign") == 0.0).sum().alias("num_flat"),
    )
    panel = (
        panel.join(agreement, on=("date", "universe"), validate="m:1")
        .with_columns(
            pl.when(pl.col("strategy_sign") > 0.0)
            .then(pl.col("num_long") / pl.col("num_objectives"))
            .when(pl.col("strategy_sign") < 0.0)
            .then(pl.col("num_short") / pl.col("num_objectives"))
            .otherwise(pl.col("num_flat") / pl.col("num_objectives"))
            .alias(AGREEMENT_SIGNAL)
        )
        .sort(*SERIES_KEYS, "date")
    )
    verify(
        panel.select(
            pl.all_horizontal(
                pl.col(
                    "spot_only_return",
                    *RISK_TARGET_SIGNALS.values(),
                    AGREEMENT_SIGNAL,
                ).is_finite()
            ).all()
        ).item(),
        lambda: "tail-risk panel contains non-finite values",
    )
    return panel


def expanding_inverse_median_exposure(
    values: np.ndarray,
    minimum_training_days: int,
) -> np.ndarray:
    """Target inverse risk using only each date's expanding history."""

    verify_inp(
        values.ndim == 1
        and len(values) > minimum_training_days
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0),
        lambda: "risk-target values must be a finite nonnegative vector",
    )
    exposure = np.empty(len(values) - minimum_training_days, dtype=np.float64)
    for output_index, date_index in enumerate(
        range(minimum_training_days, len(values))
    ):
        target = float(np.median(values[:date_index]))
        current = float(values[date_index])
        exposure[output_index] = min(1.0, target / max(current, 1.0e-12))
    return exposure


def _tail_metrics(
    returns: np.ndarray,
    *,
    config: TailRiskDiagnosticsConfig,
) -> dict[str, float | int]:
    verify(
        returns.ndim == 1 and len(returns) >= 2 and np.all(np.isfinite(returns)),
        lambda: f"invalid tail return vector with shape {returns.shape}",
    )
    mean = float(np.mean(returns))
    standard_deviation = float(np.std(returns, ddof=0))
    expected_shortfall = _empirical_lower_tail_mean(
        returns,
        config.tail_fraction,
    )
    retained_path = np.concatenate(([0.0], np.cumsum(returns)))
    wealth = 1.0 + retained_path
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "num_returns": len(returns),
        "mean_return": mean,
        "return_std": standard_deviation,
        "annualized_sharpe": (
            mean / standard_deviation * math.sqrt(config.annual_sessions)
            if standard_deviation > 0.0
            else float("nan")
        ),
        "expected_shortfall": expected_shortfall,
        "mean_to_abs_expected_shortfall": (
            mean / abs(expected_shortfall)
            if expected_shortfall != 0.0
            else float("nan")
        ),
        "worst_return": float(np.min(returns)),
        "maximum_drawdown": float(np.min(drawdown)),
    }


def _moving_block_tail_reductions(
    returns: np.ndarray,
    exposure: np.ndarray,
    *,
    config: TailRiskDiagnosticsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    num_returns = len(returns)
    block = min(config.bootstrap_block_days, num_returns)
    num_blocks = math.ceil(num_returns / block)
    reductions = np.empty(config.bootstrap_samples, dtype=np.float64)
    for bootstrap_index in range(config.bootstrap_samples):
        starts = rng.integers(0, num_returns - block + 1, size=num_blocks)
        indices = np.concatenate(
            [np.arange(start, start + block) for start in starts]
        )[:num_returns]
        sampled_returns = returns[indices]
        sampled_exposure = exposure[indices]
        overlay_tail = _empirical_lower_tail_mean(
            sampled_exposure * sampled_returns,
            config.tail_fraction,
        )
        constant_tail = _empirical_lower_tail_mean(
            float(np.mean(sampled_exposure)) * sampled_returns,
            config.tail_fraction,
        )
        reductions[bootstrap_index] = 1.0 - abs(overlay_tail) / abs(
            constant_tail
        )
    return reductions


def analyze_tail_risk_overlays(
    legacy_comparison_file: Path,
    cvar_comparison_file: Path,
    daily_market_file: Path,
    output_dir: Path,
    *,
    config: TailRiskDiagnosticsConfig = TailRiskDiagnosticsConfig(),
) -> dict[str, object]:
    """Evaluate entry-time risk overlays against matched constant exposure."""

    panel = build_tail_risk_panel(
        pl.read_parquet(legacy_comparison_file),
        pl.read_parquet(cvar_comparison_file),
        pl.read_parquet(daily_market_file),
    )
    result_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.seed)
    for key, group in panel.partition_by(*SERIES_KEYS, as_dict=True).items():
        group = group.sort("date")
        full_returns = group["spot_only_return"].to_numpy()
        evaluation_returns = full_returns[config.minimum_training_days :]
        signals: dict[str, np.ndarray] = {
            name: expanding_inverse_median_exposure(
                group[column].to_numpy(),
                config.minimum_training_days,
            )
            for name, column in RISK_TARGET_SIGNALS.items()
        }
        signals[AGREEMENT_SIGNAL] = group[AGREEMENT_SIGNAL].to_numpy()[
            config.minimum_training_days :
        ]
        evaluation_dates = group["date"].to_list()[
            config.minimum_training_days :
        ]
        for signal_name, exposure in signals.items():
            mean_exposure = float(np.mean(exposure))
            overlay_returns = exposure * evaluation_returns
            constant_returns = mean_exposure * evaluation_returns
            overlay = _tail_metrics(overlay_returns, config=config)
            constant = _tail_metrics(constant_returns, config=config)
            tail_reduction = 1.0 - abs(float(overlay["expected_shortfall"])) / abs(
                float(constant["expected_shortfall"])
            )
            result_rows.append(
                {
                    "universe": key[0],
                    "utility": key[1],
                    "risk_aversion": key[2],
                    "signal": signal_name,
                    "mean_exposure": mean_exposure,
                    **{f"overlay_{name}": value for name, value in overlay.items()},
                    **{
                        f"constant_{name}": value
                        for name, value in constant.items()
                    },
                    "expected_shortfall_reduction": tail_reduction,
                    "mean_return_change": float(overlay["mean_return"])
                    - float(constant["mean_return"]),
                    "annualized_sharpe_change": float(
                        overlay["annualized_sharpe"]
                    )
                    - float(constant["annualized_sharpe"]),
                    "mean_to_tail_change": float(
                        overlay["mean_to_abs_expected_shortfall"]
                    )
                    - float(constant["mean_to_abs_expected_shortfall"]),
                }
            )
            observation_rows.extend(
                {
                    "universe": key[0],
                    "utility": key[1],
                    "risk_aversion": key[2],
                    "signal": signal_name,
                    "date": market_date,
                    "base_return": float(base_return),
                    "exposure": float(date_exposure),
                    "overlay_return": float(date_exposure * base_return),
                    "constant_return": float(mean_exposure * base_return),
                }
                for market_date, base_return, date_exposure in zip(
                    evaluation_dates,
                    evaluation_returns,
                    exposure,
                    strict=True,
                )
            )
            if (
                key[0] == "normalised"
                and signal_name == "atm_volatility"
                and (
                    (key[1] in ("entropy", "powexp") and key[2] == 10.0)
                    or key[1] == "cvar10"
                )
            ):
                reductions = _moving_block_tail_reductions(
                    evaluation_returns,
                    exposure,
                    config=config,
                    rng=rng,
                )
                bootstrap_rows.append(
                    {
                        "universe": key[0],
                        "utility": key[1],
                        "risk_aversion": key[2],
                        "signal": signal_name,
                        "lower_95": float(np.quantile(reductions, 0.025)),
                        "median": float(np.median(reductions)),
                        "upper_95": float(np.quantile(reductions, 0.975)),
                        "fraction_positive": float(np.mean(reductions > 0.0)),
                    }
                )

    results = pl.DataFrame(result_rows).sort(*SERIES_KEYS, "signal")
    observations = pl.DataFrame(observation_rows).sort(
        *SERIES_KEYS,
        "signal",
        "date",
    )
    bootstrap = pl.DataFrame(bootstrap_rows).sort(*SERIES_KEYS)
    summary = (
        results.group_by("signal", "utility")
        .agg(
            pl.len().alias("num_series"),
            (pl.col("expected_shortfall_reduction") > 0.0)
            .sum()
            .alias("num_expected_shortfall_improvements"),
            pl.col("expected_shortfall_reduction")
            .median()
            .alias("median_expected_shortfall_reduction"),
            (pl.col("annualized_sharpe_change") > 0.0)
            .sum()
            .alias("num_sharpe_improvements"),
            (pl.col("mean_to_tail_change") > 0.0)
            .sum()
            .alias("num_mean_to_tail_improvements"),
            pl.col("mean_return_change").median().alias(
                "median_mean_return_change"
            ),
        )
        .sort("signal", "utility")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(output_dir / "tail_risk_panel.parquet")
    observations.write_parquet(output_dir / "tail_overlay_observations.parquet")
    results.write_parquet(output_dir / "tail_overlay_results.parquet")
    summary.write_parquet(output_dir / "tail_overlay_summary.parquet")
    bootstrap.write_parquet(output_dir / "tail_overlay_bootstrap.parquet")
    report = {
        "legacy_comparison_file": str(legacy_comparison_file),
        "cvar_comparison_file": str(cvar_comparison_file),
        "daily_market_file": str(daily_market_file),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "outcome": "continuous option-implied delta return over the next SPX session",
        "timing": "every overlay signal is known on the option decision date",
        "risk_target": (
            "exposure=min(1, expanding historical median/current signal); "
            "cross-objective agreement is used directly as exposure"
        ),
        "benchmark": (
            "constant exposure equal to the overlay's realized mean exposure"
        ),
        "tail_measure": (
            f"empirical mean of the worst {100 * config.tail_fraction:.0f}% "
            "of daily returns with fractional boundary mass"
        ),
        "methodology": (
            "chronological overlay evaluation on full-sample-fitted OCE signals; "
            "not a look-ahead-free model backtest"
        ),
        "summary": summary.to_dicts(),
        "representative_atm_bootstrap": bootstrap.to_dicts(),
    }
    (output_dir / "tail_overlay_report.json").write_text(
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
        default=base
        / "2025_delta_cap_1m/spot_1d/spot_only_comparison.parquet",
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
        default=base / "2025/tail_risk_diagnostics",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = analyze_tail_risk_overlays(
        arguments.legacy_comparison_file,
        arguments.cvar_comparison_file,
        arguments.daily_market_file,
        arguments.output_dir,
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
