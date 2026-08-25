"""Split optimized option-position deltas into eight option categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify

from .attribution import PORTFOLIO_KEYS

SIDES = ("long", "short")
STRIKE_REGIONS = ("below_atm", "above_atm")
OPTION_TYPES = ("call", "put")
BUCKET_DEFINITIONS = tuple(
    (side, strike_region, option_type, f"{side}_{strike_region}_{option_type}")
    for side in SIDES
    for strike_region in STRIKE_REGIONS
    for option_type in OPTION_TYPES
)
REPRESENTATIVE_OBJECTIVES = (
    ("entropy", 10.0, "Entropy 10"),
    ("powexp", 10.0, "Power-exp 10"),
    ("cvar10", 1.0, "Worst 90% CVaR"),
)


def _common_position_columns(positions: pl.DataFrame) -> pl.DataFrame:
    return positions.select(
        *PORTFOLIO_KEYS,
        "side",
        "option_type",
        "expiry_date",
        "expiry_days",
        "strike_over_forward",
        "effective_nms",
        "allocation_fraction",
        "allocation_dollars",
        "entry_pure_forward_delta",
        "net_forward_delta_contribution",
        "spot_only_return_contribution",
    )


def _common_comparison_columns(comparison: pl.DataFrame) -> pl.DataFrame:
    return comparison.select(
        *PORTFOLIO_KEYS,
        "date_index",
        "spot_complete",
        "horizon_spot_return",
        "net_entry_delta",
        "spot_only_return",
    )


def annotate_delta_categories(positions: pl.DataFrame) -> pl.DataFrame:
    """Label every option position by side, forward strike region, and type."""

    return (
        positions.with_columns(
            pl.when(pl.col("strike_over_forward") < 1.0)
            .then(pl.lit("below_atm"))
            .otherwise(pl.lit("above_atm"))
            .alias("strike_region")
        )
        .with_columns(
            pl.concat_str(
                "side",
                "strike_region",
                "option_type",
                separator="_",
            ).alias("delta_bucket")
        )
        .sort(*PORTFOLIO_KEYS, "delta_bucket", "expiry_days", "effective_nms")
    )


def build_delta_bucket_panel(
    positions: pl.DataFrame,
    comparison: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return annotated positions, a complete eight-bucket panel, and audit."""

    annotated = annotate_delta_categories(positions)
    bucket_sums = annotated.group_by(
        *PORTFOLIO_KEYS,
        "side",
        "strike_region",
        "option_type",
        "delta_bucket",
    ).agg(
        pl.len().alias("num_option_positions"),
        pl.col("allocation_fraction").sum().alias("allocation_fraction_sum"),
        pl.col("allocation_dollars").sum().alias("allocation_dollars_sum"),
        pl.col("net_forward_delta_contribution")
        .sum()
        .alias("bucket_net_forward_delta"),
        pl.col("net_forward_delta_contribution")
        .abs()
        .sum()
        .alias("bucket_gross_forward_delta"),
        pl.col("spot_only_return_contribution")
        .sum()
        .alias("bucket_spot_only_return"),
    )
    definitions = pl.DataFrame(
        BUCKET_DEFINITIONS,
        schema=("side", "strike_region", "option_type", "delta_bucket"),
        orient="row",
    )
    panel = (
        comparison.join(definitions, how="cross")
        .join(
            bucket_sums,
            on=[
                *PORTFOLIO_KEYS,
                "side",
                "strike_region",
                "option_type",
                "delta_bucket",
            ],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("num_option_positions").fill_null(0),
            pl.col(
                "allocation_fraction_sum",
                "allocation_dollars_sum",
                "bucket_net_forward_delta",
                "bucket_gross_forward_delta",
                "bucket_spot_only_return",
            ).fill_null(0.0),
        )
        .sort(*PORTFOLIO_KEYS, "delta_bucket")
    )
    reconciliation = (
        panel.group_by(*PORTFOLIO_KEYS)
        .agg(
            pl.col("net_entry_delta").first(),
            pl.col("spot_only_return").first(),
            pl.col("bucket_net_forward_delta")
            .sum()
            .alias("reconstructed_net_entry_delta"),
            pl.col("bucket_spot_only_return")
            .sum()
            .alias("reconstructed_spot_only_return"),
        )
        .with_columns(
            (
                pl.col("reconstructed_net_entry_delta")
                - pl.col("net_entry_delta")
            ).alias("net_delta_error"),
            (
                pl.col("reconstructed_spot_only_return")
                - pl.col("spot_only_return")
            ).alias("spot_only_return_error"),
        )
        .sort(*PORTFOLIO_KEYS)
    )
    maximum_delta_error = reconciliation.select(
        pl.col("net_delta_error").abs().max()
    ).item()
    maximum_return_error = reconciliation.select(
        pl.col("spot_only_return_error").abs().max()
    ).item()
    verify(
        maximum_delta_error is not None
        and np.isfinite(maximum_delta_error)
        and maximum_delta_error <= 1.0e-10,
        lambda: f"delta buckets do not reconcile: max error {maximum_delta_error}",
    )
    verify(
        maximum_return_error is not None
        and np.isfinite(maximum_return_error)
        and maximum_return_error <= 1.0e-10,
        lambda: (
            "delta-bucket spot returns do not reconcile: max error "
            f"{maximum_return_error}"
        ),
    )
    return annotated, panel, reconciliation


def build_eight_category_attribution(
    panel: pl.DataFrame,
    *,
    capital_dollars: float = 1_000_000.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Select eight strike-region buckets for three objective attributions."""

    objective_map = pl.DataFrame(
        REPRESENTATIVE_OBJECTIVES,
        schema=("utility", "risk_aversion", "objective"),
        orient="row",
    )
    attribution = (
        panel.join(
            objective_map,
            on=["utility", "risk_aversion"],
            how="inner",
            validate="m:1",
        )
        .filter(
            (
                (pl.col("universe") == "ITM")
                & (
                    (
                        (pl.col("strike_region") == "below_atm")
                        & (pl.col("option_type") == "call")
                    )
                    | (
                        (pl.col("strike_region") == "above_atm")
                        & (pl.col("option_type") == "put")
                    )
                )
            )
            | (
                (pl.col("universe") == "normalised")
                & (
                    (
                        (pl.col("strike_region") == "below_atm")
                        & (pl.col("option_type") == "put")
                    )
                    | (
                        (pl.col("strike_region") == "above_atm")
                        & (pl.col("option_type") == "call")
                    )
                )
            )
        )
        .rename({"universe": "source_universe"})
        .sort("objective", "date", "delta_bucket")
    )
    rows_per_date = attribution.group_by("objective", "date").len()
    verify(
        rows_per_date.select((pl.col("len") == 8).all()).item(),
        lambda: "canonical category attribution does not have eight rows per date",
    )
    totals = (
        attribution.unique(["objective", "source_universe", "date"])
        .group_by("objective", "source_universe")
        .agg(
            (pl.col("spot_only_return").sum() * capital_dollars).alias(
                "source_portfolio_cumulative_pnl_dollars"
            )
        )
    )
    summary = (
        attribution.group_by(
            "objective",
            "source_universe",
            "delta_bucket",
            "side",
            "strike_region",
            "option_type",
        )
        .agg(
            pl.len().alias("num_portfolio_dates"),
            (pl.col("num_option_positions") > 0)
            .sum()
            .alias("num_nonempty_portfolio_dates"),
            pl.col("num_option_positions").sum().alias("num_option_positions"),
            pl.col("bucket_net_forward_delta").mean().alias("mean_bucket_delta"),
            pl.col("bucket_net_forward_delta")
            .abs()
            .mean()
            .alias("mean_abs_bucket_delta"),
            pl.col("bucket_net_forward_delta").median().alias("median_bucket_delta"),
            (pl.col("bucket_spot_only_return").sum() * capital_dollars).alias(
                "cumulative_bucket_pnl_dollars"
            ),
            (pl.col("bucket_spot_only_return").mean() * capital_dollars).alias(
                "mean_daily_bucket_pnl_dollars"
            ),
        )
        .join(
            totals,
            on=["objective", "source_universe"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("cumulative_bucket_pnl_dollars")
                / pl.col("source_portfolio_cumulative_pnl_dollars")
            ).alias("pnl_attribution_fraction")
        )
        .sort("objective", "delta_bucket")
    )
    return attribution, summary


def build_source_characteristic_attribution(
    panel: pl.DataFrame,
    *,
    capital_dollars: float = 1_000_000.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attribute delta by long/short and above/below ATM within each source."""

    objective_map = pl.DataFrame(
        REPRESENTATIVE_OBJECTIVES,
        schema=("utility", "risk_aversion", "objective"),
        orient="row",
    )
    representative = panel.join(
        objective_map,
        on=["utility", "risk_aversion"],
        how="inner",
        validate="m:1",
    )
    daily_cells = (
        representative.group_by(
            "objective",
            "universe",
            "date",
            "side",
            "strike_region",
        )
        .agg(
            pl.col("bucket_net_forward_delta").sum().alias("attributed_delta"),
            pl.col("bucket_spot_only_return")
            .sum()
            .alias("attributed_spot_only_return"),
        )
        .rename({"universe": "source_universe"})
        .sort("source_universe", "objective", "date", "side", "strike_region")
    )

    def summarize(
        daily: pl.DataFrame,
        group_columns: list[str],
        *,
        grouping: str,
        level: pl.Expr,
    ) -> pl.DataFrame:
        return (
            daily.group_by("source_universe", "objective", *group_columns)
            .agg(
                pl.len().alias("num_dates"),
                pl.col("attributed_delta").mean().alias("mean_attributed_delta"),
                pl.col("attributed_delta")
                .abs()
                .mean()
                .alias("mean_abs_attributed_delta"),
                (pl.col("attributed_spot_only_return").sum() * capital_dollars)
                .alias("cumulative_attributed_pnl_dollars"),
                (pl.col("attributed_spot_only_return").mean() * capital_dollars)
                .alias("mean_daily_attributed_pnl_dollars"),
            )
            .with_columns(
                pl.lit(grouping).alias("grouping"),
                level.alias("level"),
            )
            .select(
                "source_universe",
                "objective",
                "grouping",
                "level",
                pl.col("side") if "side" in group_columns else pl.lit(None).alias("side"),
                (
                    pl.col("strike_region")
                    if "strike_region" in group_columns
                    else pl.lit(None).alias("strike_region")
                ),
                "num_dates",
                "mean_attributed_delta",
                "mean_abs_attributed_delta",
                "cumulative_attributed_pnl_dollars",
                "mean_daily_attributed_pnl_dollars",
            )
        )

    cell_summary = summarize(
        daily_cells,
        ["side", "strike_region"],
        grouping="side_x_strike_region",
        level=pl.concat_str("side", "strike_region", separator="_"),
    )
    daily_side = daily_cells.group_by(
        "source_universe", "objective", "date", "side"
    ).agg(
        pl.col("attributed_delta").sum(),
        pl.col("attributed_spot_only_return").sum(),
    )
    side_summary = summarize(
        daily_side,
        ["side"],
        grouping="side",
        level=pl.col("side"),
    )
    daily_region = daily_cells.group_by(
        "source_universe", "objective", "date", "strike_region"
    ).agg(
        pl.col("attributed_delta").sum(),
        pl.col("attributed_spot_only_return").sum(),
    )
    region_summary = summarize(
        daily_region,
        ["strike_region"],
        grouping="strike_region",
        level=pl.col("strike_region"),
    )
    daily_total = daily_cells.group_by(
        "source_universe", "objective", "date"
    ).agg(
        pl.col("attributed_delta").sum(),
        pl.col("attributed_spot_only_return").sum(),
    )
    total_summary = summarize(
        daily_total,
        [],
        grouping="portfolio",
        level=pl.lit("total"),
    )
    summary = pl.concat(
        (cell_summary, side_summary, region_summary, total_summary),
        how="vertical_relaxed",
    )
    totals = total_summary.select(
        "source_universe",
        "objective",
        pl.col("cumulative_attributed_pnl_dollars").alias(
            "source_portfolio_cumulative_pnl_dollars"
        ),
    )
    summary = (
        summary.join(
            totals,
            on=["source_universe", "objective"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("cumulative_attributed_pnl_dollars")
                / pl.col("source_portfolio_cumulative_pnl_dollars")
            ).alias("pnl_attribution_fraction")
        )
        .sort("source_universe", "objective", "grouping", "level")
    )
    return daily_cells, summary


def split_delta_buckets(
    legacy_positions_file: Path,
    cvar_positions_file: Path,
    legacy_comparison_file: Path,
    cvar_comparison_file: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Persist the complete eight-category delta decomposition."""

    positions = pl.concat(
        (
            _common_position_columns(pl.read_parquet(legacy_positions_file)),
            _common_position_columns(pl.read_parquet(cvar_positions_file)),
        ),
        how="vertical_relaxed",
    )
    comparison = pl.concat(
        (
            _common_comparison_columns(pl.read_parquet(legacy_comparison_file)),
            _common_comparison_columns(pl.read_parquet(cvar_comparison_file)),
        ),
        how="vertical_relaxed",
    )
    annotated, panel, reconciliation = build_delta_bucket_panel(
        positions,
        comparison,
    )
    eight_category_attribution, eight_category_summary = (
        build_eight_category_attribution(panel)
    )
    source_characteristic_attribution, source_characteristic_summary = (
        build_source_characteristic_attribution(panel)
    )
    summary = (
        panel.group_by(
            "universe",
            "delta_bucket",
            "side",
            "strike_region",
            "option_type",
        )
        .agg(
            pl.len().alias("num_portfolio_dates"),
            (pl.col("num_option_positions") > 0)
            .sum()
            .alias("num_nonempty_portfolio_dates"),
            pl.col("num_option_positions").sum().alias("num_option_positions"),
            pl.col("bucket_net_forward_delta").mean().alias("mean_bucket_delta"),
            pl.col("bucket_net_forward_delta")
            .abs()
            .mean()
            .alias("mean_abs_bucket_delta"),
            pl.col("bucket_net_forward_delta").median().alias("median_bucket_delta"),
        )
        .sort("universe", "delta_bucket")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.write_parquet(output_dir / "delta_bucket_positions.parquet")
    panel.write_parquet(output_dir / "delta_bucket_panel.parquet")
    reconciliation.write_parquet(output_dir / "delta_bucket_reconciliation.parquet")
    summary.write_parquet(output_dir / "delta_bucket_summary.parquet")
    eight_category_attribution.write_parquet(
        output_dir / "eight_category_delta_attribution.parquet"
    )
    eight_category_summary.write_parquet(
        output_dir / "eight_category_delta_attribution_summary.parquet"
    )
    source_characteristic_attribution.write_parquet(
        output_dir / "source_characteristic_delta_attribution.parquet"
    )
    source_characteristic_summary.write_parquet(
        output_dir / "source_characteristic_delta_attribution_summary.parquet"
    )
    for source_universe, filename in (
        ("ITM", "itm_characteristic_delta_attribution.parquet"),
        ("normalised", "normalised_characteristic_delta_attribution.parquet"),
        ("actual", "actual_characteristic_delta_attribution.parquet"),
    ):
        source_characteristic_summary.filter(
            pl.col("source_universe") == source_universe
        ).write_parquet(output_dir / filename)
    maximum_delta_error = reconciliation.select(
        pl.col("net_delta_error").abs().max()
    ).item()
    maximum_return_error = reconciliation.select(
        pl.col("spot_only_return_error").abs().max()
    ).item()
    report = {
        "legacy_positions_file": str(legacy_positions_file),
        "cvar_positions_file": str(cvar_positions_file),
        "legacy_comparison_file": str(legacy_comparison_file),
        "cvar_comparison_file": str(cvar_comparison_file),
        "output_dir": str(output_dir),
        "strike_region": (
            "forward relative at entry: below ATM means K/F < 1 and above ATM "
            "means K/F >= 1"
        ),
        "side": "original optimized option position side, not the sign of delta",
        "delta": (
            "signed forward delta contribution including option side, optimized "
            "allocation fraction, and the universe-specific divider"
        ),
        "num_option_positions": len(annotated),
        "num_portfolios": len(reconciliation),
        "num_bucket_rows": len(panel),
        "buckets_per_portfolio": len(BUCKET_DEFINITIONS),
        "maximum_net_delta_reconciliation_error": maximum_delta_error,
        "maximum_spot_return_reconciliation_error": maximum_return_error,
        "universe_structure": (
            "ITM portfolios supply below-ATM calls and above-ATM puts; actual "
            "and normalised portfolios supply below-ATM puts and above-ATM calls"
        ),
        "canonical_eight_category_view": (
            "attribution only: each strike-region/type bucket comes from the ITM "
            "or normalised optimization that contains that option; source universes "
            "remain explicit and are not summed into one portfolio"
        ),
        "characteristic_attribution": (
            "separate source tables collapse synthetic put/call identity and "
            "attribute delta and P&L by long/short x below/above ATM, including "
            "long/short and strike-region marginal totals; no prediction"
        ),
        "summary": summary.to_dicts(),
        "eight_category_summary": eight_category_summary.to_dicts(),
        "source_characteristic_summary": source_characteristic_summary.to_dicts(),
    }
    (output_dir / "delta_bucket_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("aae/ideas/localvolbysanos/real_trading/results")
    parser.add_argument(
        "--legacy-positions-file",
        type=Path,
        default=base / "2025/one_day/spot_only_positions.parquet",
    )
    parser.add_argument(
        "--cvar-positions-file",
        type=Path,
        default=base / "2025_cvar_daily/one_day/spot_only_positions.parquet",
    )
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
        "--output-dir",
        type=Path,
        default=base / "2025/delta_buckets",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = split_delta_buckets(
        arguments.legacy_positions_file,
        arguments.cvar_positions_file,
        arguments.legacy_comparison_file,
        arguments.cvar_comparison_file,
        arguments.output_dir,
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
