"""
run_days.py — multi-day driver for the single-strike IV-KF q_scale sweep.

Runs LINEARLY over trade dates (no cross-date parallelism, to bound peak memory);
WITHIN each date the q_scale grid is parallelized by iv_kf_sweep across --workers
processes.  Per date it:

  1. builds the Phase-1 cache if missing (the expensive network fetch + batched IV,
     paid exactly once per date — see phase1_cache.py), then
  2. runs the q_scale sweep, writing results/iv_kf_sweep_{date}.csv.

After all dates it concatenates the per-date CSVs into results/iv_kf_sweep_ALL.csv
with a leading `date` column.

Usage
-----
    # a contiguous range of trading days (weekends/holidays skipped)
    python run_days.py --start 2024-01-03 --end 2024-01-17

    # an explicit list
    python run_days.py --dates 2024-01-03 2024-01-04 2024-01-05

    # force-rebuild caches, fewer workers
    python run_days.py --start 2024-01-03 --end 2024-01-17 --rebuild-cache --workers 6
"""

import argparse
import gc
import os
import sys
from datetime import date, datetime

project_root = "/Users/elie/Documents/GitHub/quant_data/"
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (project_root, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import iv_kf_sweep
import pandas as pd
import phase1_cache

from quantdata.biz_calendar import get_business_days, to_date


def _resolve_dates(args: argparse.Namespace) -> list[date]:
    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("--start and --end must be given together")
        return list(get_business_days(to_date(args.start), to_date(args.end)))
    if not args.dates:
        raise SystemExit("provide --dates, or a --start/--end range")
    out = []
    for s in args.dates:
        d = to_date(s)
        if not get_business_days(d, d):
            print(f"skip {d}: not a trading day")
            continue
        out.append(d)
    return out


def _ensure_cache(d: date, rebuild: bool) -> str:
    """Build phase1_cache_{d}.pkl if missing (or rebuild). Returns the cache path.
    Frames are freed when phase1_cache.main returns, so the subsequent sweep starts
    from a low-memory parent."""
    cache_path = iv_kf_sweep._cache_path(str(d))  # _HERE/phase1_cache_{d}.pkl
    if rebuild or not os.path.exists(cache_path):
        print(f"[{d}] building phase1 cache → {cache_path}")
        phase1_cache.main(d, cache_path)
        gc.collect()
    else:
        print(f"[{d}] phase1 cache exists, reusing {cache_path}")
    return cache_path


def _aggregate(dates: list[date], results_dir: str) -> None:
    frames = []
    for d in dates:
        p = os.path.join(results_dir, f"iv_kf_sweep_{d}.csv")
        if not os.path.exists(p):
            print(f"  (no results for {d}, skipping in aggregate)")
            continue
        df = pd.read_csv(p)
        df.insert(0, "date", str(d))
        frames.append(df)
    if not frames:
        print("No per-date results to aggregate.")
        return
    all_path = os.path.join(results_dir, "iv_kf_sweep_ALL.csv")
    pd.concat(frames, ignore_index=True).to_csv(all_path, index=False)
    print(f"\nAggregated {len(frames)} date(s) → {all_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", help="range start YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="range end YYYY-MM-DD (inclusive)")
    p.add_argument("--dates", nargs="*", help="explicit trade dates YYYY-MM-DD")
    p.add_argument("--workers", type=int, default=iv_kf_sweep.mp.cpu_count() - 1)
    p.add_argument(
        "--rebuild-cache", action="store_true", help="rebuild phase1 caches even if present"
    )
    args = p.parse_args()

    dates = _resolve_dates(args)
    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Running IV-KF sweep over {len(dates)} day(s), {args.workers} workers/day:")
    print("  " + ", ".join(str(d) for d in dates))

    failures = []
    for d in dates:
        print(f"\n===== {d} =====")
        try:
            _ensure_cache(d, args.rebuild_cache)
            out = os.path.join(results_dir, f"iv_kf_sweep_{d}.csv")
            iv_kf_sweep.main(str(d), out, args.workers)
        except Exception as e:
            print(f"[{d}] FAILED: {e}")
            failures.append((d, str(e)))
        finally:
            gc.collect()

    _aggregate(dates, results_dir)

    if failures:
        print("\nFailures:")
        for d, e in failures:
            print(f"  {d}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
