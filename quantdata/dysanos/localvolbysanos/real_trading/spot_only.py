"""Trade only the entry delta implied by each optimized option portfolio."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify_inp

from .attribution import (
    PORTFOLIO_KEYS,
    SERIES_KEYS,
    _entry_position_exposures,
    _hac_mean_standard_error,
    _tail_statistics,
)


@dataclass(frozen=True, kw_only=True)
class SpotOnlyConfig:
    """Holding-period and inference conventions for the spot-only test."""

    horizon_sessions: int = 5
    hac_lags: int = 4
    top_fraction: float = 0.10

    def __post_init__(self) -> None:
        verify_inp(
            self.horizon_sessions >= 1,
            lambda: "horizon_sessions must be positive",
        )
        verify_inp(self.hac_lags >= 0, lambda: "hac_lags must be non-negative")
        verify_inp(
            0.0 < self.top_fraction < 0.5,
            lambda: "top_fraction must lie between zero and one half",
        )


def _horizon_dates(dates: list[date], horizon_sessions: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": dates[:-horizon_sessions],
            "horizon_date": dates[horizon_sessions:],
        }
    )


def _summary(
    comparison: pl.DataFrame,
    *,
    config: SpotOnlyConfig,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in comparison.partition_by(*SERIES_KEYS, as_dict=True).items():
        complete = group.filter(pl.col("spot_complete"))
        spot_only_return = complete["spot_only_return"].to_numpy()
        horizon_spot_return = complete["horizon_spot_return"].to_numpy()
        net_delta = complete["net_entry_delta"].to_numpy()
        mean_net_delta = float(np.mean(net_delta))
        fixed_mean_delta_return = mean_net_delta * horizon_spot_return
        spot_std = float(np.std(horizon_spot_return, ddof=0))
        strategy_std = float(np.std(spot_only_return, ddof=0))
        traded = complete.filter(pl.col("gross_scheduled_delta") > 0.0)
        unit_gross_return = traded["unit_gross_spot_return"].to_numpy()
        unit_gross_standard_error = _hac_mean_standard_error(
            unit_gross_return,
            lags=config.hac_lags,
        )
        risk_matched_fixed_delta = strategy_std / spot_std
        risk_matched_spot_return = risk_matched_fixed_delta * horizon_spot_return
        standard_error = _hac_mean_standard_error(
            spot_only_return,
            lags=config.hac_lags,
        )
        common = complete.filter(pl.col("option_realized_return").is_not_null())
        rows.append(
            {
                "universe": key[0],
                "utility": key[1],
                "risk_aversion": key[2],
                "num_portfolios": group.height,
                "num_complete": complete.height,
                "num_traded_portfolios": traded.height,
                "coverage": complete.height / group.height,
                "mean_net_entry_delta": mean_net_delta,
                "median_net_entry_delta": float(np.median(net_delta)),
                "mean_gross_entry_delta": float(
                    complete["gross_entry_delta"].mean()
                ),
                "mean_spot_only_return": float(np.mean(spot_only_return)),
                "median_spot_only_return": float(np.median(spot_only_return)),
                "spot_only_return_std": strategy_std,
                "spot_only_mean_hac_se": standard_error,
                "spot_only_mean_hac_t": (
                    float(np.mean(spot_only_return)) / standard_error
                    if standard_error > 0.0
                    else float("nan")
                ),
                "mean_fixed_average_delta_spot_return": float(
                    np.mean(fixed_mean_delta_return)
                ),
                "fixed_average_delta_spot_return_std": float(
                    np.std(fixed_mean_delta_return, ddof=0)
                ),
                "risk_matched_fixed_delta": risk_matched_fixed_delta,
                "mean_risk_matched_spot_return": float(
                    np.mean(risk_matched_spot_return)
                ),
                "spot_only_return_to_risk": float(
                    np.mean(spot_only_return) / strategy_std
                ),
                "mean_unit_gross_spot_return": float(
                    np.mean(unit_gross_return)
                ),
                "median_unit_gross_spot_return": float(
                    np.median(unit_gross_return)
                ),
                "unit_gross_spot_return_std": float(
                    np.std(unit_gross_return, ddof=0)
                ),
                "unit_gross_spot_positive_fraction": float(
                    np.mean(unit_gross_return > 0.0)
                ),
                "unit_gross_spot_mean_hac_se": unit_gross_standard_error,
                "unit_gross_spot_mean_hac_t": (
                    float(np.mean(unit_gross_return)) / unit_gross_standard_error
                    if unit_gross_standard_error > 0.0
                    else float("nan")
                ),
                "fixed_spot_return_to_risk": float(
                    np.mean(horizon_spot_return) / spot_std
                ),
                "entry_delta_future_spot_correlation": float(
                    np.corrcoef(net_delta, horizon_spot_return)[0, 1]
                ),
                "num_common_option_portfolios": common.height,
                "mean_option_realized_return_common": (
                    float(common["option_realized_return"].mean())
                    if common.height
                    else float("nan")
                ),
                "mean_spot_only_return_common": (
                    float(common["spot_only_return"].mean())
                    if common.height
                    else float("nan")
                ),
                **_tail_statistics(
                    spot_only_return,
                    top_fraction=config.top_fraction,
                ),
            }
        )
    return pl.DataFrame(rows).sort(*SERIES_KEYS)


def run_spot_only_test(
    positions_file: Path,
    results_file: Path,
    realized_comparison_file: Path,
    spot_file: Path,
    output_dir: Path,
    *,
    config: SpotOnlyConfig = SpotOnlyConfig(),
) -> dict[str, object]:
    """Run one spot-delta cohort per option-optimization date."""

    positions = _entry_position_exposures(pl.read_parquet(positions_file))
    results = pl.read_parquet(results_file)
    realized = pl.read_parquet(realized_comparison_file).select(
        *PORTFOLIO_KEYS,
        pl.col("realized_return").alias("option_realized_return"),
    )
    dates = sorted(results["date"].unique().to_list())
    horizons = _horizon_dates(dates, config.horizon_sessions)
    spot = (
        pl.read_parquet(spot_file)
        .select("date", "spot_ref")
        .unique("date")
        .sort("date")
    )
    current_spot = spot.rename({"spot_ref": "entry_spot_ref"})
    close_spot = spot.rename({"date": "close_date", "spot_ref": "close_spot_ref"})
    horizon_spot = spot.rename(
        {"date": "horizon_date", "spot_ref": "horizon_spot_ref"}
    )

    position_spot = (
        positions.join(horizons, on="date", how="inner", validate="m:1")
        .with_columns(
            pl.min_horizontal("expiry_date", "horizon_date").alias("close_date")
        )
        .join(current_spot, on="date", how="left", validate="m:1")
        .join(close_spot, on="close_date", how="left", validate="m:1")
        .with_columns(
            pl.col("close_spot_ref").is_not_null().alias("spot_available"),
            (pl.col("close_spot_ref") / pl.col("entry_spot_ref") - 1.0).alias(
                "spot_return_to_close"
            ),
        )
        .with_columns(
            (
                pl.col("net_forward_delta_contribution")
                * pl.col("spot_return_to_close")
            ).alias("spot_only_return_contribution")
        )
    )
    close_slices = position_spot.group_by(*PORTFOLIO_KEYS, "close_date").agg(
        pl.col("horizon_date").first(),
        pl.col("entry_spot_ref").first(),
        pl.col("close_spot_ref").first(),
        pl.col("spot_available").all(),
        pl.col("net_forward_delta_contribution").sum().alias("net_delta_slice"),
        pl.col("spot_only_return_contribution")
        .sum()
        .alias("spot_only_return_slice"),
    )
    close_slices = close_slices.with_columns(
        pl.col("net_delta_slice").abs().alias("gross_scheduled_delta_slice")
    )
    portfolios = close_slices.group_by(*PORTFOLIO_KEYS).agg(
        pl.col("horizon_date").first(),
        pl.col("entry_spot_ref").first(),
        pl.col("spot_available").all().alias("spot_complete"),
        pl.len().alias("num_close_dates"),
        pl.col("net_delta_slice").sum().alias("net_entry_delta"),
        pl.col("gross_scheduled_delta_slice")
        .sum()
        .alias("gross_scheduled_delta"),
        pl.col("spot_only_return_slice").sum().alias("spot_only_return_partial"),
    )
    portfolios = portfolios.with_columns(
        pl.col("net_entry_delta").abs().alias("gross_entry_delta"),
        pl.when(pl.col("spot_complete"))
        .then(pl.col("spot_only_return_partial"))
        .otherwise(None)
        .alias("spot_only_return")
    )
    portfolio_panel = (
        results.select(*PORTFOLIO_KEYS)
        .join(horizons, on="date", how="inner", validate="m:1")
        .join(current_spot, on="date", how="left", validate="m:1")
        .join(
            portfolios.drop("horizon_date", "entry_spot_ref"),
            on=list(PORTFOLIO_KEYS),
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("spot_complete").fill_null(True),
            pl.col("num_close_dates").fill_null(0),
            pl.col(
                "net_entry_delta",
                "gross_scheduled_delta",
                "spot_only_return_partial",
                "gross_entry_delta",
                "spot_only_return",
            ).fill_null(0.0),
        )
    )
    comparison = (
        results.join(
            portfolio_panel,
            on=list(PORTFOLIO_KEYS),
            how="inner",
            validate="1:1",
        )
        .join(horizon_spot, on="horizon_date", how="left", validate="m:1")
        .join(realized, on=list(PORTFOLIO_KEYS), how="left", validate="1:1")
        .with_columns(
            (
                pl.col("spot_complete")
                & pl.col("horizon_spot_ref").is_not_null()
            ).alias("spot_complete"),
            (pl.col("horizon_spot_ref") / pl.col("entry_spot_ref") - 1.0).alias(
                "horizon_spot_return"
            ),
            (pl.col("spot_only_return") * 1_000_000.0).alias(
                "spot_only_pnl_per_million"
            ),
            (pl.col("net_entry_delta") * 1_000_000.0).alias(
                "net_spot_notional_per_million"
            ),
            pl.when(pl.col("gross_scheduled_delta") > 0.0)
            .then(pl.col("spot_only_return") / pl.col("gross_scheduled_delta"))
            .otherwise(None)
            .alias("unit_gross_spot_return"),
        )
        .sort(*PORTFOLIO_KEYS)
    )
    summary = _summary(comparison, config=config)

    output_dir.mkdir(parents=True, exist_ok=True)
    position_spot.write_parquet(output_dir / "spot_only_positions.parquet")
    close_slices.write_parquet(output_dir / "spot_only_close_slices.parquet")
    comparison.write_parquet(output_dir / "spot_only_comparison.parquet")
    summary.write_parquet(output_dir / "spot_only_summary.parquet")
    report = {
        "positions_file": str(positions_file),
        "results_file": str(results_file),
        "realized_comparison_file": str(realized_comparison_file),
        "spot_file": str(spot_file),
        "output_dir": str(output_dir),
        "strategy": (
            "replace every optimized option position with its signed entry forward "
            "delta in SPX"
        ),
        "entry": "SPX spot_ref midpoint on the option decision date",
        "exit": (
            "close each delta slice at its option expiry inside the holding week, "
            f"otherwise at the {config.horizon_sessions}-session horizon"
        ),
        "rebalancing": "none between entry and the assigned close date",
        "costs": "no spot spread, commissions, funding, borrow, or market impact",
        "notional": (
            "net delta is signed SPX notional per $1 risk-budget unit; values above "
            "one require leverage if implemented in cash spot"
        ),
        "horizon_sessions": config.horizon_sessions,
        "num_comparisons": comparison.height,
        "num_complete": int(comparison["spot_complete"].sum()),
        "summary": summary.to_dicts(),
    }
    (output_dir / "spot_only_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("quantdata/dysanos/localvolbysanos/real_trading/results/2025")
    parser.add_argument(
        "--positions-file", type=Path, default=base / "positions.parquet"
    )
    parser.add_argument(
        "--results-file", type=Path, default=base / "oce_results.parquet"
    )
    parser.add_argument(
        "--realized-comparison-file",
        type=Path,
        default=base / "realized_comparison.parquet",
    )
    parser.add_argument(
        "--spot-file",
        type=Path,
        default=Path(
            "quantdata/dysanos/localvolbysanos/results/2025_full_reference_grid/"
            "daily_summary.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    parser.add_argument("--horizon-sessions", type=int, default=5)
    parser.add_argument("--hac-lags", type=int, default=4)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_spot_only_test(
        arguments.positions_file,
        arguments.results_file,
        arguments.realized_comparison_file,
        arguments.spot_file,
        arguments.output_dir,
        config=SpotOnlyConfig(
            horizon_sessions=arguments.horizon_sessions,
            hac_lags=arguments.hac_lags,
        ),
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
