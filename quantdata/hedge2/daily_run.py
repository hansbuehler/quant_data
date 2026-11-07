"""Run one-day Greek and Black baselines over every observable rolling origin."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.cash import MarketStore, observed_marks, select_book
from quantdata.hedge2.hedge_pilot import run_pilot
from quantdata.hedge2.source import SourceArchive


def run_daily(*, source_root: Path, licensed_fit_directory: Path, raw_quote_directory: Path,
              market_directory: Path, daily_directory: Path, dynamics_directory: Path,
              output_directory: Path) -> dict[str, object]:
    """Exclude unavailable marks explicitly, never filter the inception book by future prices."""
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    archive = SourceArchive(root=source_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    eligible, missing = [], []
    started = time.monotonic()
    paths = sorted((dynamics_directory / "days").glob("*.npz"))
    for index, path in enumerate(paths):
        day = date.fromisoformat(path.stem)
        next_day = get_business_days(day, day + timedelta(days=10))[1]
        try:
            current = store.load(day)
            following = store.load(next_day)
            source, _ = archive.load(day)
            vols = np.asarray(source.interpolate_atm_vols(current.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
            book = select_book(current, atm_vols=vols)
            observed_marks(book, following)
            eligible.append(day)
        except FileNotFoundError as error:
            missing.append({"date": day.isoformat(), "reason": "missing_source_or_next_session", "detail": str(error)})
        except ValueError as error:
            if not (str(error).startswith("Missing observed mids") or str(error).startswith("No PM expiry")):
                raise
            missing.append({"date": day.isoformat(), "reason": "incomplete_observed_hedge_book", "detail": str(error)})
        if (index + 1) % 50 == 0:
            print(f"daily coverage {index+1}/{len(paths)} eligible={len(eligible)} missing={len(missing)} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    coverage = {"candidate_dates": len(paths), "eligible_dates": len(eligible),
                "eligible_start": eligible[0].isoformat() if eligible else None,
                "eligible_end": eligible[-1].isoformat() if eligible else None,
                "excluded_dates": missing,
                "selection_note": "Causal inception books; complete realized marks required for all methods. Missing episodes are disclosed."}
    (output_directory / "observed_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if not eligible:
        raise ValueError("No rolling origins have complete observed hedge marks")
    result = run_pilot(source_root=source_root, licensed_fit_directory=licensed_fit_directory,
                       raw_quote_directory=raw_quote_directory, market_directory=market_directory,
                       daily_directory=daily_directory, dynamics_directory=dynamics_directory,
                       output_directory=output_directory / "hedges", dates=tuple(eligible), diagnostic_only=False)
    frame = pd.read_parquet(output_directory / "hedges" / "results.parquet")
    summary = frame.groupby(["method", "side"], sort=False).agg(
        observations=("costed_pnl", "size"), mean_pnl=("costed_pnl", "mean"),
        std_pnl=("costed_pnl", "std"), median_pnl=("costed_pnl", "median"),
        mean_entry_cost=("entry_cost", "mean"), mean_hypothetical_exit_cost=("hypothetical_exit_cost", "mean"),
        cap_hits=("cap_hits", "sum"), mean_gross_positions=("gross_positions", "mean"))
    summary.to_parquet(output_directory / "pooled_diagnostics.parquet")
    result.update(coverage=coverage, total_elapsed_seconds=time.monotonic() - started)
    (output_directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "licensed-fit-directory", "raw-quote-directory", "market-directory", "daily-directory", "dynamics-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_daily(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
