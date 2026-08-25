"""Analyze sequential capped spot holdings and their retained-P&L ledgers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp

from .attribution import SERIES_KEYS


@dataclass(frozen=True, kw_only=True)
class SequentialAnalysisConfig:
    """Capital, annualization, and numerical conventions for the spot ledger."""

    capital_dollars: float = 1_000_000.0
    annual_sessions: int = 252
    cap_tolerance_fraction: float = 1.0e-6

    def __post_init__(self) -> None:
        verify_inp(
            np.isfinite(self.capital_dollars) and self.capital_dollars > 0.0,
            lambda: "capital_dollars must be finite and positive",
        )
        verify_inp(
            self.annual_sessions >= 1,
            lambda: "annual_sessions must be positive",
        )
        verify_inp(
            self.cap_tolerance_fraction > 0.0,
            lambda: "cap_tolerance_fraction must be positive",
        )


def _performance(
    returns: np.ndarray,
    *,
    config: SequentialAnalysisConfig,
) -> dict[str, float | int]:
    verify(
        returns.ndim == 1 and len(returns) >= 2 and np.all(np.isfinite(returns)),
        lambda: f"invalid sequential return vector with shape {returns.shape}",
    )
    mean = float(np.mean(returns))
    standard_deviation = float(np.std(returns, ddof=0))
    retained_path = np.concatenate(([0.0], np.cumsum(returns)))
    wealth = 1.0 + retained_path
    running_peak = np.maximum.accumulate(wealth)
    drawdown = wealth / running_peak - 1.0
    return {
        "num_returns": len(returns),
        "mean_daily_return": mean,
        "median_daily_return": float(np.median(returns)),
        "daily_return_std": standard_deviation,
        "positive_fraction": float(np.mean(returns > 0.0)),
        "annualized_arithmetic_return": mean * config.annual_sessions,
        "annualized_volatility": standard_deviation
        * math.sqrt(config.annual_sessions),
        "annualized_sharpe": (
            mean / standard_deviation * math.sqrt(config.annual_sessions)
            if standard_deviation > 0.0
            else float("nan")
        ),
        "final_retained_pnl_dollars": float(np.sum(returns))
        * config.capital_dollars,
        "maximum_drawdown": float(np.min(drawdown)),
    }


def build_sign_only_ledger(
    comparison: pl.DataFrame,
    *,
    config: SequentialAnalysisConfig,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Recast independent signs as a sequential capped $1m holding."""

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    complete = comparison.filter(pl.col("spot_complete"))
    for key, group in complete.partition_by(*SERIES_KEYS, as_dict=True).items():
        group = group.sort("date")
        raw_signal = np.sign(group["net_entry_delta"].to_numpy())
        signal = raw_signal.copy()
        for index in np.flatnonzero(signal == 0.0):
            # No new direction means retain the preceding sign; an initial zero
            # remains flat until the first nonzero instruction arrives.
            if index > 0:
                signal[index] = signal[index - 1]
        spot_return = group["horizon_spot_return"].to_numpy()
        retained_pnl_fraction = 0.0
        carried_before_cap = 0.0
        absolute_trade = 0.0
        num_cap_clips = 0
        for index, row in enumerate(group.iter_rows(named=True)):
            clipped = float(np.clip(carried_before_cap, -1.0, 1.0))
            cap_clip_cash = carried_before_cap - clipped
            num_cap_clips += int(abs(cap_clip_cash) > config.cap_tolerance_fraction)
            target = float(signal[index])
            trade = target - carried_before_cap
            absolute_trade += abs(trade)
            daily_pnl_fraction = target * float(spot_return[index])
            retained_pnl_fraction += daily_pnl_fraction
            exit_holding_before_cap = target * (1.0 + float(spot_return[index]))
            rows.append(
                {
                    "universe": key[0],
                    "utility": key[1],
                    "risk_aversion": key[2],
                    "date": row["date"],
                    "horizon_date": row["horizon_date"],
                    "carried_spot_fraction_before_cap": carried_before_cap,
                    "carried_spot_fraction": clipped,
                    "cap_clip_cash_fraction": cap_clip_cash,
                    "raw_signal": float(raw_signal[index]),
                    "retained_previous_sign": bool(raw_signal[index] == 0.0),
                    "target_spot_fraction": target,
                    "spot_trade_fraction": trade,
                    "spot_return": float(spot_return[index]),
                    "daily_spot_pnl_fraction": daily_pnl_fraction,
                    "daily_spot_pnl_dollars": (
                        daily_pnl_fraction * config.capital_dollars
                    ),
                    "cumulative_retained_pnl_fraction": retained_pnl_fraction,
                    "cumulative_retained_pnl_dollars": (
                        retained_pnl_fraction * config.capital_dollars
                    ),
                    "exit_holding_fraction_before_cap": (
                        exit_holding_before_cap
                    ),
                }
            )
            carried_before_cap = exit_holding_before_cap

        final_clipped = float(np.clip(carried_before_cap, -1.0, 1.0))
        final_target = float(signal[-1])
        num_cap_clips += int(
            abs(carried_before_cap - final_clipped)
            > config.cap_tolerance_fraction
        )
        summaries.append(
            {
                "strategy": "sign_only",
                "universe": key[0],
                "utility": key[1],
                "risk_aversion": key[2],
                **_performance(spot_return * signal, config=config),
                "mean_abs_target_spot_dollars": float(np.mean(np.abs(signal)))
                * config.capital_dollars,
                "median_abs_target_spot_dollars": float(
                    np.median(np.abs(signal))
                )
                * config.capital_dollars,
                "cap_binding_fraction": float(
                    np.mean(
                        np.abs(signal)
                        >= 1.0 - config.cap_tolerance_fraction
                    )
                ),
                "long_fraction": float(np.mean(signal > 0.0)),
                "short_fraction": float(np.mean(signal < 0.0)),
                "num_zero_signals_retained": int(np.sum(raw_signal == 0.0)),
                "mean_abs_daily_trade_dollars": (
                    absolute_trade / len(group) * config.capital_dollars
                ),
                "num_cap_clips": num_cap_clips,
                "terminal_spot_dollars": final_target * config.capital_dollars,
            }
        )
    return pl.DataFrame(rows).sort(*SERIES_KEYS, "date"), summaries


def summarize_continuous_results(
    results: pl.DataFrame,
    *,
    config: SequentialAnalysisConfig,
) -> list[dict[str, object]]:
    """Summarize realized P&L from the sequential endowment-aware OCE run."""

    summaries: list[dict[str, object]] = []
    for key, group in results.partition_by(*SERIES_KEYS, as_dict=True).items():
        group = group.sort("date")
        daily_returns = group["daily_spot_pnl_fraction"].to_numpy()[1:]
        target = group["net_forward_delta_fraction"].to_numpy()
        carried_before_cap = group[
            "carried_spot_fraction_before_cap"
        ].to_numpy()
        spot_trade = target - carried_before_cap
        cap_clip_cash = group["cap_clip_cash_fraction"].to_numpy()
        verify(
            float(np.max(np.abs(target))) <= 1.0 + config.cap_tolerance_fraction,
            lambda: f"continuous series {key} breaches its spot cap",
        )
        summaries.append(
            {
                "strategy": "continuous_endowment_oce",
                "universe": key[0],
                "utility": key[1],
                "risk_aversion": key[2],
                **_performance(daily_returns, config=config),
                "mean_abs_target_spot_dollars": float(np.mean(np.abs(target)))
                * config.capital_dollars,
                "median_abs_target_spot_dollars": float(np.median(np.abs(target)))
                * config.capital_dollars,
                "cap_binding_fraction": float(
                    np.mean(
                        np.abs(target)
                        >= 1.0 - config.cap_tolerance_fraction
                    )
                ),
                "long_fraction": float(np.mean(target > 0.0)),
                "short_fraction": float(np.mean(target < 0.0)),
                "mean_abs_daily_trade_dollars": float(np.mean(np.abs(spot_trade)))
                * config.capital_dollars,
                "num_cap_clips": int(
                    np.sum(
                        np.abs(cap_clip_cash)
                        > config.cap_tolerance_fraction
                    )
                ),
                "terminal_spot_dollars": float(target[-1])
                * config.capital_dollars,
            }
        )
    return summaries


def analyze_sequential_strategies(
    continuous_results_file: Path,
    sign_comparison_file: Path,
    output_dir: Path,
    *,
    config: SequentialAnalysisConfig = SequentialAnalysisConfig(),
) -> dict[str, object]:
    """Persist sign-only and continuous sequential spot ledgers and summaries."""

    continuous_results = pl.read_parquet(continuous_results_file)
    sign_comparison = pl.read_parquet(sign_comparison_file)
    sign_ledger, sign_summary = build_sign_only_ledger(
        sign_comparison,
        config=config,
    )
    continuous_summary = summarize_continuous_results(
        continuous_results,
        config=config,
    )
    summary = pl.DataFrame(sign_summary + continuous_summary).sort(
        "strategy",
        *SERIES_KEYS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sign_ledger.write_parquet(output_dir / "sign_only_sequential_ledger.parquet")
    summary.write_parquet(output_dir / "sequential_spot_summary.parquet")
    report = {
        "continuous_results_file": str(continuous_results_file),
        "sign_comparison_file": str(sign_comparison_file),
        "output_dir": str(output_dir),
        "capital_dollars": config.capital_dollars,
        "accounting": (
            "daily P&L is previous target notional times observed spot_ref return; "
            "P&L is transferred to a retained ledger and is not reinvested"
        ),
        "cap_timing": (
            "market-move exposure is clipped to ±capital at each daily observation "
            "before the new target is selected"
        ),
        "zero_sign_signal": (
            "retain the preceding sign; remain flat only if the first signal is zero"
        ),
        "costs": "no spread, commissions, funding, borrow, or market impact",
        "summary": summary.to_dicts(),
    }
    (output_dir / "sequential_spot_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("aae/ideas/localvolbysanos/real_trading/results")
    parser.add_argument(
        "--continuous-results-file",
        type=Path,
        default=base / "2025_sequential_cap_1m/oce_results.parquet",
    )
    parser.add_argument(
        "--sign-comparison-file",
        type=Path,
        default=base / "2025/one_day/spot_only_comparison.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "2025_sequential_cap_1m",
    )
    parser.add_argument("--capital-dollars", type=float, default=1_000_000.0)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = analyze_sequential_strategies(
        arguments.continuous_results_file,
        arguments.sign_comparison_file,
        arguments.output_dir,
        config=SequentialAnalysisConfig(
            capital_dollars=arguments.capital_dollars,
        ),
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
