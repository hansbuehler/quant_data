"""
Command-line runner for the single-option Kalman diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.biz_calendar import prev_business_day

from .data import expiry_summary, load_day_expiries, option_rank, validate_expiry_frame
from .plotting import plot_single_option_result
from .single_option import run_single_option_filter, sweep_rho


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--date", default="2025-01-23", help="Reference date; prev_business_day(..., count_start=True) is used.")
    parser.add_argument("--expiry-index", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--rho-points", type=int, default=13)
    parser.add_argument("--rho-low", type=float, default=-0.9)
    parser.add_argument("--rho-high", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, default=Path("kalman_filter_outputs"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ref_date = date.fromisoformat(args.date)
    d = prev_business_day(ref_date, exclude_half_days=True, count_start=True)

    results = load_day_expiries(args.ticker, d)
    expiry_info = expiry_summary(results)
    print(expiry_info)

    target_expiry = expiry_info["expiry"][args.expiry_index]
    df_expiry = results[target_expiry].sort(["opt_ticker", "t_d"])
    validation_summary, validation_issues = validate_expiry_frame(df_expiry)
    print(validation_summary.head(10))
    if len(validation_issues):
        print(validation_issues)

    ranked = option_rank(df_expiry)
    example_opt_tickers = ranked.head(max(args.examples, 1))["opt_ticker"].to_list()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for opt_ticker in example_opt_tickers[: args.examples]:
        result = run_single_option_filter(df_expiry, opt_ticker, rho=0.0, drop_intrinsic=True)
        safe_name = opt_ticker.replace(":", "_").replace("/", "_")
        output_path = args.output_dir / f"{safe_name}_rho0.png"
        plot_single_option_result(result, output_path=output_path)
        print(result.summary)
        print(f"wrote {output_path}")

    rho_grid = np.round(np.linspace(args.rho_low, args.rho_high, args.rho_points), 2)
    rho_test_tickers = ranked.head(max(8, args.examples))["opt_ticker"].to_list()
    rho_detail, rho_agg = sweep_rho(df_expiry, rho_test_tickers, rho_grid, drop_intrinsic=True)
    rho_detail.write_parquet(args.output_dir / "rho_detail.parquet")
    rho_agg.write_parquet(args.output_dir / "rho_agg.parquet")
    print(rho_agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
