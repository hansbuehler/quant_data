"""Compare $1m continuous and digital OCE signals with ATM targeting."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp

from .experiment import _empirical_lower_tail_mean
from .tail_risk_diagnostics import expanding_inverse_median_exposure

OBJECTIVE_SPECS = (
    ("entropy", 10.0, "Entropy, gamma 10"),
    ("powexp", 10.0, "Power-exp, gamma 10"),
    ("cvar10", 1.0, "Worst 90% CVaR"),
)
VARIANT_LABELS = {
    ("continuous", False): "Continuous",
    ("continuous", True): "Continuous, ATM-adjusted",
    ("digital", False): "Digital ±$1m",
    ("digital", True): "Digital, ATM-adjusted",
}
VARIANT_COLORS = {
    ("continuous", False): "#1f77b4",
    ("continuous", True): "#17becf",
    ("digital", False): "#ff7f0e",
    ("digital", True): "#d62728",
}


@dataclass(frozen=True, kw_only=True)
class InvestmentComparisonConfig:
    """Capital, timing, and risk conventions for the strategy comparison."""

    capital_dollars: float = 1_000_000.0
    minimum_training_days: int = 60
    tail_fraction: float = 0.10
    annual_sessions: int = 252

    def __post_init__(self) -> None:
        verify_inp(
            np.isfinite(self.capital_dollars) and self.capital_dollars > 0.0,
            lambda: "capital_dollars must be finite and positive",
        )
        verify_inp(
            self.minimum_training_days >= 30,
            lambda: "minimum_training_days must be at least 30",
        )
        verify_inp(
            0.0 < self.tail_fraction <= 0.5,
            lambda: "tail_fraction must lie in (0, 0.5]",
        )
        verify_inp(
            self.annual_sessions >= 1,
            lambda: "annual_sessions must be positive",
        )


def _strategy_metrics(
    returns: np.ndarray,
    *,
    config: InvestmentComparisonConfig,
) -> dict[str, float | int]:
    verify(
        returns.ndim == 1 and len(returns) >= 2 and np.all(np.isfinite(returns)),
        lambda: f"invalid investment return vector with shape {returns.shape}",
    )
    mean = float(np.mean(returns))
    standard_deviation = float(np.std(returns, ddof=0))
    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    wealth = 1.0 + cumulative
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "num_returns": len(returns),
        "terminal_pnl_dollars": float(np.sum(returns))
        * config.capital_dollars,
        "mean_daily_pnl_dollars": mean * config.capital_dollars,
        "daily_pnl_std_dollars": standard_deviation * config.capital_dollars,
        "annualized_sharpe": (
            mean / standard_deviation * math.sqrt(config.annual_sessions)
            if standard_deviation > 0.0
            else float("nan")
        ),
        "expected_shortfall_10_dollars": _empirical_lower_tail_mean(
            returns,
            config.tail_fraction,
        )
        * config.capital_dollars,
        "maximum_drawdown": float(np.min(drawdown)),
        "positive_fraction": float(np.mean(returns > 0.0)),
    }


def _plot_investment_paths(
    observations: pl.DataFrame,
    output_file: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(OBJECTIVE_SPECS),
        figsize=(15.0, 5.0),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.99, wspace=0.10)
    for axis, (utility, _, objective_label) in zip(
        axes,
        OBJECTIVE_SPECS,
        strict=True,
    ):
        objective = observations.filter(pl.col("utility") == utility)
        for variant, atm_adjusted in VARIANT_LABELS:
            series = objective.filter(
                (pl.col("variant") == variant)
                & (pl.col("atm_adjusted") == atm_adjusted)
            )
            axis.plot(
                series["date"].to_list(),
                series["cumulative_pnl_dollars"].to_numpy(),
                color=VARIANT_COLORS[(variant, atm_adjusted)],
                linewidth=1.8,
                label=VARIANT_LABELS[(variant, atm_adjusted)],
            )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_title(objective_label)
        axis.set_xlabel("Decision date")
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Cumulative retained P&L ($)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=4,
    )
    figure.suptitle(
        "$1m independent daily SPX signal: raw versus ATM-adjusted",
        y=0.98,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_investment_strategies(
    legacy_comparison_file: Path,
    cvar_comparison_file: Path,
    daily_market_file: Path,
    output_dir: Path,
    *,
    config: InvestmentComparisonConfig = InvestmentComparisonConfig(),
) -> dict[str, object]:
    """Persist raw and ATM-adjusted continuous/digital $1m strategies."""

    required = (
        "date",
        "universe",
        "utility",
        "risk_aversion",
        "spot_complete",
        "net_entry_delta",
        "spot_only_return",
        "unit_gross_spot_return",
    )
    legacy = pl.read_parquet(legacy_comparison_file).select(*required)
    cvar = pl.read_parquet(cvar_comparison_file).select(*required)
    comparison = pl.concat((legacy, cvar), how="vertical_relaxed")
    daily_market = (
        pl.read_parquet(daily_market_file)
        .select("date", "spot_ref", "front_week_atm_volatility")
        .unique("date")
        .sort("date")
        .with_columns(
            (pl.col("spot_ref") / pl.col("spot_ref").shift(1) - 1.0).alias(
                "today_spot_return"
            )
        )
    )
    panel = (
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
        .join(daily_market, on="date", how="inner", validate="m:1")
        .filter(pl.col("today_spot_return").is_not_null())
        .sort("utility", "risk_aversion", "date")
    )

    observation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for utility, risk_aversion, objective_label in OBJECTIVE_SPECS:
        group = panel.filter(
            (pl.col("utility") == utility)
            & (pl.col("risk_aversion") == risk_aversion)
        ).sort("date")
        atm_exposure = expanding_inverse_median_exposure(
            group["front_week_atm_volatility"].to_numpy(),
            config.minimum_training_days,
        )
        dates = group["date"].to_list()[config.minimum_training_days :]
        base_returns = {
            "continuous": group["spot_only_return"].to_numpy()[
                config.minimum_training_days :
            ],
            "digital": group["unit_gross_spot_return"].to_numpy()[
                config.minimum_training_days :
            ],
        }
        target_fraction = {
            "continuous": group["net_entry_delta"].to_numpy()[
                config.minimum_training_days :
            ],
            "digital": np.sign(group["net_entry_delta"].to_numpy())[
                config.minimum_training_days :
            ],
        }
        for variant, unadjusted_returns in base_returns.items():
            for atm_adjusted in (False, True):
                exposure = atm_exposure if atm_adjusted else np.ones_like(
                    atm_exposure
                )
                returns = exposure * unadjusted_returns
                cumulative_pnl = np.cumsum(returns) * config.capital_dollars
                summary_rows.append(
                    {
                        "objective": objective_label,
                        "utility": utility,
                        "risk_aversion": risk_aversion,
                        "variant": variant,
                        "atm_adjusted": atm_adjusted,
                        "mean_exposure": float(np.mean(exposure)),
                        "mean_abs_spot_notional_dollars": float(
                            np.mean(np.abs(exposure * target_fraction[variant]))
                        )
                        * config.capital_dollars,
                        **_strategy_metrics(returns, config=config),
                    }
                )
                observation_rows.extend(
                    {
                        "objective": objective_label,
                        "utility": utility,
                        "risk_aversion": risk_aversion,
                        "variant": variant,
                        "atm_adjusted": atm_adjusted,
                        "date": market_date,
                        "exposure": float(date_exposure),
                        "target_spot_fraction": float(
                            date_exposure * date_target
                        ),
                        "return": float(date_return),
                        "pnl_dollars": float(
                            date_return * config.capital_dollars
                        ),
                        "cumulative_pnl_dollars": float(date_cumulative),
                    }
                    for market_date, date_exposure, date_target, date_return, date_cumulative in zip(
                        dates,
                        exposure,
                        target_fraction[variant],
                        returns,
                        cumulative_pnl,
                        strict=True,
                    )
                )

    observations = pl.DataFrame(observation_rows).sort(
        "utility",
        "risk_aversion",
        "variant",
        "atm_adjusted",
        "date",
    )
    summary = pl.DataFrame(summary_rows).sort(
        "utility",
        "risk_aversion",
        "variant",
        "atm_adjusted",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.write_parquet(output_dir / "investment_paths.parquet")
    summary.write_parquet(output_dir / "investment_summary.parquet")
    figure_file = output_dir / "investment_comparison.png"
    _plot_investment_paths(observations, figure_file)
    report = {
        "legacy_comparison_file": str(legacy_comparison_file),
        "cvar_comparison_file": str(cvar_comparison_file),
        "daily_market_file": str(daily_market_file),
        "output_dir": str(output_dir),
        "figure_file": str(figure_file),
        "config": asdict(config),
        "universe": "normalised",
        "strategy": (
            "independent next-session SPX position with P&L retained and not "
            "reinvested"
        ),
        "digital": "sign of option-implied delta times $1m",
        "continuous": "option-implied delta times $1m",
        "atm_adjustment": (
            "multiply target by min(1, expanding pre-date median ATM vol / "
            "current ATM vol)"
        ),
        "summary": summary.to_dicts(),
    }
    (output_dir / "investment_report.json").write_text(
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
        default=base / "2025/investment_comparison",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = compare_investment_strategies(
        arguments.legacy_comparison_file,
        arguments.cvar_comparison_file,
        arguments.daily_market_file,
        arguments.output_dir,
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
