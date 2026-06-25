#!/usr/bin/env python
"""Pre-warm the quantdata cache for all days that have a ticker universe CSV.

Fetches the two data sources that three_model_comparison.ipynb needs but that
build_ticker_universe.py did NOT already cache:

  1. fetch_trades("SPY", date)   — underlying trades (one call per day)
  2. fetch_quotes(ticker, date)  — option quotes  (~200-400 calls per day, threaded)

fetch_trades for option tickers is already cached from build_ticker_universe.py.

Usage
-----
    python personal/warm_cache.py                          # all CSVs in ticker_universes/
    python personal/warm_cache.py 2024-01-03 2024-01-04   # specific dates
    python personal/warm_cache.py --start 2024-01-03 --end 2024-01-10
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path


def _find_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists() and (d / "quantdata").is_dir():
            return d
    raise RuntimeError(f"quant_data root not found above {start}")


PROJECT_ROOT = _find_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from quantdata.biz_calendar import get_business_days, to_date
from quantdata.massive.client import MassiveClient

TICKER_DIR = PROJECT_ROOT / "personal" / "ticker_universes"
UNDERLYING = "SPY"
QUOTE_WORKERS = 16  # I/O-bound; threads are fine


def dates_from_dir() -> list[date]:
    """All dates that have a ticker CSV in ticker_universes/."""
    out = []
    for f in sorted(TICKER_DIR.glob("spy_options_tickers_*.csv")):
        try:
            out.append(date.fromisoformat(f.stem.replace("spy_options_tickers_", "")))
        except ValueError:
            pass
    return out


def warm_day(client: MassiveClient, d: date) -> None:
    csv = TICKER_DIR / f"spy_options_tickers_{d}.csv"
    if not csv.exists():
        print(f"  {d}: no ticker CSV — skipping")
        return

    tickers = pl.read_csv(csv)["ticker"].to_list()
    t0 = time.monotonic()

    # 1. Underlying trades (single call)
    client.fetch_trades(UNDERLYING, d)

    # 2. Option quotes in parallel (not yet cached; the heavy part)
    errors = []
    with ThreadPoolExecutor(max_workers=QUOTE_WORKERS) as pool:
        futs = {pool.submit(client.fetch_quotes, t, d): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            done += 1
            ticker = futs[fut]
            try:
                fut.result()
            except Exception as e:
                errors.append(f"{ticker}: {e}")
            if done % 50 == 0 or done == len(tickers):
                print(f"  {d}: quotes {done}/{len(tickers)} ...", flush=True)

    elapsed = time.monotonic() - t0
    status = f"  {d}: done in {elapsed:.0f}s ({len(tickers)} tickers"
    if errors:
        status += f", {len(errors)} errors)"
        for e in errors[:5]:
            print(f"    {e}")
    else:
        status += ")"
    print(status)


def resolve_dates(args: argparse.Namespace) -> list[date]:
    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("--start and --end must both be given")
        return list(get_business_days(to_date(args.start), to_date(args.end)))
    if args.dates:
        return [to_date(s) for s in args.dates]
    return dates_from_dir()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "dates", nargs="*", help="trade date(s) YYYY-MM-DD (default: all in ticker_universes/)"
    )
    p.add_argument("--start", help="range start YYYY-MM-DD")
    p.add_argument("--end", help="range end YYYY-MM-DD")
    p.add_argument(
        "--workers",
        type=int,
        default=QUOTE_WORKERS,
        help=f"threads for fetch_quotes (default {QUOTE_WORKERS})",
    )
    args = p.parse_args()

    dates = resolve_dates(args)
    print(f"Warming cache for {len(dates)} day(s) ...")
    client = MassiveClient()
    t_total = time.monotonic()
    for d in dates:
        warm_day(client, d)
    print(f"\nAll done in {(time.monotonic() - t_total) / 60:.1f} min.")


if __name__ == "__main__":
    main()
