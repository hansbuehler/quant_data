#!/usr/bin/env python
"""Build the SPY option ticker universe for one or more trade dates.

This is the .py counterpart of personal/personal_notebooks/viz_all_strikes_same_expiry.ipynb
(the upstream producer of the ``spy_options_tickers_<date>.csv`` files that
three_model_comparison.ipynb / build_iv_observations consume).

For each trade date it:
  1. fetches the contract universe as-of that date (filter_contracts_between),
  2. selects the single expiry whose business days-to-expiry is closest to
     --target-dte (default 2; the 2024-01-03 -> 2024-01-05 case),
  3. counts trades per contract on the trade date (dropping contracts that
     error or did not trade), and
  4. writes <out-dir>/spy_options_tickers_<date>.csv with columns
     ``ticker, trade_count, ticker_suffix`` (the schema downstream expects).

Examples
--------
    # single day
    python personal/build_ticker_universe.py 2024-01-03

    # explicit list of days
    python personal/build_ticker_universe.py 2024-01-03 2024-03-05

    # a contiguous range of trading days (weekends/holidays skipped)
    python personal/build_ticker_universe.py --start 2024-01-03 --end 2024-02-15
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path


def _find_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists() and (d / "quantdata").is_dir():
            return d
    raise RuntimeError(f"quant_data project root not found above {start}")


PROJECT_ROOT = _find_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from personal.shared_functions.utils import parse_option_ticker
from quantdata.biz_calendar import (
    count_business_days_between,
    get_business_days,
    to_date,
)
from quantdata.massive.client import MassiveClient
from quantdata.massive.options.contracts import (
    Context,
    ContractsFilter,
    DateRange,
    PoolConfig,
    filter_contracts_between,
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "personal" / "ticker_universes"


def select_expiry_tickers(
    df_contracts: pl.DataFrame, trade_date: date, target_dte: int
) -> tuple[date, list[str]]:
    """Pick the expiry with business-DTE closest to ``target_dte`` and return
    (chosen_expiry, tickers_for_that_expiry). Only expiries on/after the trade
    date are considered; ties break to the shorter-dated expiry."""
    tickers = df_contracts.get_column("ticker").unique().to_list()
    by_expiry: dict[date, list[str]] = {}
    for t in tickers:
        _, expiry_dt, _ = parse_option_ticker(t)
        by_expiry.setdefault(expiry_dt.date(), []).append(t)

    candidates = []
    for expiry, tk in by_expiry.items():
        dte = count_business_days_between(trade_date, expiry)
        if dte < 0:  # expiry before the trade date — not a tradable forward
            continue
        candidates.append((abs(dte - target_dte), dte, expiry, tk))
    if not candidates:
        raise RuntimeError(f"no expiries on/after {trade_date} in contract universe")

    _, _, expiry, tk = min(candidates, key=lambda c: (c[0], c[1]))
    return expiry, tk


def build_universe_for_date(
    client: MassiveClient,
    underlying: str,
    trade_date: date,
    target_dte: int,
    contracts_filter: ContractsFilter,
    pool_config: PoolConfig,
    min_trades: int,
) -> pl.DataFrame:
    """Return the ticker/trade_count/ticker_suffix frame for one trade date."""
    df_contracts = filter_contracts_between(
        underlying,
        DateRange(start=trade_date, end=trade_date, exclude_half_days=True),
        contracts_filter=contracts_filter,
        refresh=False,
        pool_config=pool_config,
        verbose=Context.quiet,
    )
    if df_contracts is None or df_contracts.is_empty():
        raise RuntimeError(f"no contracts returned for {underlying} as-of {trade_date}")

    expiry, tickers = select_expiry_tickers(df_contracts, trade_date, target_dte)
    print(
        f"  selected expiry {expiry} "
        f"(biz-DTE {count_business_days_between(trade_date, expiry)}, "
        f"{len(tickers)} contracts) ..."
    )

    rows = []
    for ticker in sorted(tickers):
        try:
            n = client.fetch_trades(ticker, trade_date).height
        except Exception as e:  # contract with no/broken data on this day
            print(f"    drop {ticker}: {e}")
            continue
        if n < min_trades:
            continue
        rows.append({"ticker": ticker, "trade_count": n})

    if not rows:
        raise RuntimeError(
            f"no contracts with >= {min_trades} trade(s) for expiry {expiry} on {trade_date}"
        )

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("ticker").str.slice(-6).cast(pl.Float64).alias("ticker_suffix"))
        .sort("ticker")
    )


def resolve_dates(args: argparse.Namespace) -> list[date]:
    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("--start and --end must be given together")
        return list(get_business_days(to_date(args.start), to_date(args.end)))
    if not args.dates:
        raise SystemExit("provide trade date(s) positionally, or use --start/--end")
    out = []
    for s in args.dates:
        d = to_date(s)
        if not get_business_days(d, d):
            print(f"skip {d}: not a trading day")
            continue
        out.append(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dates", nargs="*", help="trade date(s) YYYY-MM-DD")
    p.add_argument("--start", help="range start (YYYY-MM-DD), inclusive")
    p.add_argument("--end", help="range end (YYYY-MM-DD), inclusive")
    p.add_argument("--ticker", default="SPY", help="underlying (default SPY)")
    p.add_argument(
        "--target-dte", type=int, default=2, help="target business days-to-expiry (default 2)"
    )
    p.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR), help=f"output folder (default {DEFAULT_OUT_DIR})"
    )
    p.add_argument(
        "--min-trades", type=int, default=0, help="drop contracts with fewer trades (default 0)"
    )
    p.add_argument(
        "--max-dte", type=int, default=25, help="ContractsFilter.max_dte business days (default 25)"
    )
    p.add_argument(
        "--max-mns", type=float, default=3.0, help="ContractsFilter.max_mns (default 3.0)"
    )
    p.add_argument(
        "--min-opt-vol-k",
        type=float,
        default=1.0,
        help="ContractsFilter.min_opt_vol_k (default 1.0)",
    )
    p.add_argument("--workers", type=int, default=8, help="contract-fetch pool workers (default 8)")
    p.add_argument(
        "--overwrite", action="store_true", help="rebuild even if the CSV already exists"
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = resolve_dates(args)
    print(f"Building {args.ticker} ticker universe for {len(dates)} day(s) -> {out_dir}")

    client = MassiveClient()
    contracts_filter = ContractsFilter(
        max_dte=args.max_dte, max_mns=args.max_mns, min_opt_vol_k=args.min_opt_vol_k
    )
    pool_config = PoolConfig(num_workers=args.workers, threading=False)

    failures = []
    for d in dates:
        out_path = out_dir / f"spy_options_tickers_{d}.csv"
        if out_path.exists() and not args.overwrite:
            print(f"{d}: exists, skipping (use --overwrite)")
            continue
        print(f"{d}:")
        try:
            df = build_universe_for_date(
                client,
                args.ticker,
                d,
                args.target_dte,
                contracts_filter,
                pool_config,
                args.min_trades,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append((d, str(e)))
            continue
        df.write_csv(out_path)
        print(f"  wrote {df.height} contracts -> {out_path}")

    if failures:
        print("\nFailures:")
        for d, e in failures:
            print(f"  {d}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
