"""
run_days.py — multi-day driver for the SSVI surface-KF sweep.

Runs LINEARLY over trade dates (no cross-date parallelism, to bound peak memory);
WITHIN each date the (Q_scale, P0_scale, Q_structure) grid is parallelized by
ssvi_sweep across --workers processes.

The Phase-1 cache is SHARED with the single-strike pipeline
(Kalman_filter_individual/phase1_cache_{date}.pkl) — both models use the identical
fetch + IV batch, so it is built once (here, only if missing) and reused, never
re-fetched per model.

Per date it:
  1. builds the shared Phase-1 cache if absent (expensive network fetch, once), then
  2. runs the SSVI sweep, writing results/ssvi_sweep_{date}.csv.

After all dates it concatenates the per-date CSVs into results/ssvi_sweep_ALL.csv
with a leading `date` column.

Usage
-----
    python run_days.py --start 2024-01-03 --end 2024-01-17
    python run_days.py --dates 2024-01-03 2024-01-04 2024-01-05
    python run_days.py --start 2024-01-03 --end 2024-01-17 --rebuild-cache --workers 6
"""

import argparse
import gc
import os
import sys
from datetime import date

project_root = "/Users/elie/Documents/GitHub/quant_data/"
_HERE = os.path.dirname(os.path.abspath(__file__))
_INDIV = os.path.join(project_root, "personal", "Kalman_filter_individual")
for _p in (project_root, _HERE, _INDIV):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The shared Phase-1 builder (single-strike pipeline). Imported as a module so we
# can call its main(trade_date, out_path) to populate the cache when missing.
import importlib.util

import pandas as pd
import ssvi_sweep

_spec = importlib.util.spec_from_file_location(
    "indiv_phase1_cache", os.path.join(_INDIV, "phase1_cache.py")
)
indiv_phase1_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indiv_phase1_cache)

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
    """Build the SHARED single-strike phase1 cache if missing (or rebuild)."""
    cache_path = ssvi_sweep._cache_path(str(d))  # Kalman_filter_individual/phase1_cache_{d}.pkl
    if rebuild or not os.path.exists(cache_path):
        print(f"[{d}] building shared phase1 cache → {cache_path}")
        indiv_phase1_cache.main(d, cache_path)
        gc.collect()
    else:
        print(f"[{d}] shared phase1 cache exists, reusing {cache_path}")
    return cache_path


def _aggregate(dates: list[date], results_dir: str) -> None:
    frames = []
    for d in dates:
        p = os.path.join(results_dir, f"ssvi_sweep_{d}.csv")
        if not os.path.exists(p):
            print(f"  (no results for {d}, skipping in aggregate)")
            continue
        df = pd.read_csv(p)
        df.insert(0, "date", str(d))
        frames.append(df)
    if not frames:
        print("No per-date results to aggregate.")
        return
    all_path = os.path.join(results_dir, "ssvi_sweep_ALL.csv")
    pd.concat(frames, ignore_index=True).to_csv(all_path, index=False)
    print(f"\nAggregated {len(frames)} date(s) → {all_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", help="range start YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="range end YYYY-MM-DD (inclusive)")
    p.add_argument("--dates", nargs="*", help="explicit trade dates YYYY-MM-DD")
    p.add_argument("--workers", type=int, default=ssvi_sweep.mp.cpu_count() - 1)
    p.add_argument(
        "--rebuild-cache", action="store_true", help="rebuild shared phase1 caches even if present"
    )
    args = p.parse_args()

    dates = _resolve_dates(args)
    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Running SSVI sweep over {len(dates)} day(s), {args.workers} workers/day:")
    print("  " + ", ".join(str(d) for d in dates))

    failures = []
    for d in dates:
        print(f"\n===== {d} =====")
        try:
            _ensure_cache(d, args.rebuild_cache)
            out = os.path.join(results_dir, f"ssvi_sweep_{d}.csv")
            ssvi_sweep.main(str(d), out, args.workers)
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
