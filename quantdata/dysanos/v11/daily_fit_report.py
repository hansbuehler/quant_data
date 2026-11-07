"""Per-origin daily fit comparisons derived without interrupting GPU fitting."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Inverse weighted empirical CDF at 50%, without interpolation."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, .5 * cumulative[-1], side="left")
    return float(values[order[index]])


def daily_comparison(frame: pd.DataFrame, runtime_seconds: float, oldest_weight: float = .25) -> dict:
    """Compare today with previous dates under the same fitted window model.

    Each date's containment percentage is weighted once by its decay weight,
    not again by its quote count. Today's value is excluded from both previous
    day statistics. Date weights retain their ages within the full window.
    """
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) < 2 or frame.date.duplicated().any():
        raise ValueError("Need at least two distinct fitted dates")
    fits = frame.inside_percent.to_numpy(dtype=np.float64)
    if not np.isfinite(fits).all() or np.any((fits < 0) | (fits > 100)):
        raise ValueError("Fit percentages must be finite and between zero and 100")
    weights = oldest_weight ** (np.arange(len(frame)-1, -1, -1, dtype=np.float64)/(len(frame)-1))
    prior_fits, prior_weights = fits[:-1], weights[:-1]
    return {"date": str(frame.date.iloc[-1]), "today_fit_percent": float(fits[-1]),
            "previous_dates": len(prior_fits),
            "previous_weighted_mean_fit_percent": float(np.average(prior_fits, weights=prior_weights)),
            "previous_weighted_median_fit_percent": weighted_median(prior_fits, prior_weights),
            "runtime_seconds": float(runtime_seconds), "runtime_minutes": float(runtime_seconds/60)}


def refresh(root: Path, reported: set[str]) -> int:
    """Update the derived CSV atomically from completed, immutable origin files."""
    summary = root/"summary.jsonl"
    if not summary.exists():
        return 0
    manifest = json.loads((root/"manifest.json").read_text())
    output = []
    for line in summary.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame = pd.read_csv(root/"origins"/row["date"]/"daily-fit.csv")
        record = daily_comparison(frame, row["elapsed_seconds"], manifest["config"]["oldest_weight"])
        record.update(window_market_fit_percent=row["market_inside_percent"],
                      iterations=row["iterations"], stop_reason=row["stop_reason"])
        output.append(record)
        if row["date"] not in reported:
            print(json.dumps({"event": "daily_fit_report", **record}), flush=True)
            reported.add(row["date"])
    if output:
        temporary = root/"daily-fit-report.tmp.csv"
        pd.DataFrame(output).to_csv(temporary, index=False)
        temporary.replace(root/"daily-fit-report.csv")
    return len(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--unit")
    args = parser.parse_args()
    reported = set()
    while True:
        count = refresh(args.run, reported)
        if (args.run/"COMPLETE.json").exists():
            print(json.dumps({"event": "DAILY_REPORT_COMPLETE", "dates": count}), flush=True)
            return
        if args.unit is None:
            return
        active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", args.unit], check=False)
        if active.returncode:
            print(json.dumps({"event": "training_not_running", "reported_dates": count}), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
