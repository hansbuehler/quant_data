"""Extract matched-date diagnostics from the standalone DYSANOS v2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.hedging.model_audit.rolling_versions import origins
from quantdata.hedging.model_audit.run import _hash, _json

FACTORS = (0, 1, 2, 4, 8, 16)


def aligned_indices(actual: np.ndarray, expected: tuple[str, ...]) -> np.ndarray:
    """Map the exact comparison dates without silently skipping or duplicating any."""
    values = list(map(str, actual))
    if len(set(values)) != len(values):
        raise ValueError("Input contains duplicate dates")
    lookup = {day: index for index, day in enumerate(values)}
    missing = [day for day in expected if day not in lookup]
    if missing:
        raise ValueError(f"Standalone V2 lacks matched dates: {missing}")
    return np.array([lookup[day] for day in expected], dtype=int)


def summarize(*, reference: Path, run: Path, calibration: Path, output: Path) -> dict:
    """Compute only statistics already supported by the saved standalone V2 run."""
    selected = origins(reference)
    score_path = run / "straddle-density-per-day.npz"
    quote_path = run / "quotefit.npz"
    with np.load(score_path, allow_pickle=False) as saved:
        index = aligned_indices(saved["dates"], selected)
        data = saved["per_day"][index].copy()
    with np.load(quote_path, allow_pickle=False) as saved:
        index = aligned_indices(saved["dates"], selected)
        quotes, inside, excess = (saved[name][index].copy() for name in ("n", "inside", "excess_sum"))
    if (data.shape != (len(selected), len(FACTORS) + 1) or not np.isfinite(data).all()
            or np.any(data[:, 0] <= 0.) or np.any(data[:, 1:] < 0.)
            or np.any(quotes <= 0) or np.any(inside < 0) or np.any(inside > quotes)
            or not np.isfinite(excess).all()):
        raise ValueError("Standalone V2 input statistics are malformed")
    shares = 1. - data[:, 1:].sum(axis=0) / data[:, 0].sum()
    summary = {"model": "standalone DYSANOS v2 density", "dates": list(selected),
        "projection": {"factors": list(FACTORS), "shares": shares.tolist(),
            "scope": "out-of-sample realized straddle changes projected ex post onto spot and factor sensitivities",
            "score": "1 - sum residual squared / sum realized squared; coefficients fitted on each realized outcome",
            "not_a_forecast_covariance_test": True},
        "same_day_quotes": {"count": int(quotes.sum()), "inside_count": int(inside.sum()),
            "inside_rate": float(inside.sum() / quotes.sum()), "median_date_inside_rate": float(np.median(inside / quotes)),
            "scope": "standalone V2 quote book; same-day reconstruction, not next-day prediction"},
        "pending": ["matched observable forecast covariance", "tail calibration", "recursive stability", "static-arbitrage simulation"],
        "sources": {str(path): _hash(path) for path in (score_path, quote_path, calibration / "calibration.json")}}
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "STANDALONE_V2_METRICS.json", summary)
    frame = pd.DataFrame({"date": selected, "quotes": quotes, "inside_quotes": inside,
        "inside_rate": inside / quotes, "total_squared_straddle_change": data[:, 0],
        **{f"residual_squared_p{rank}": data[:, position + 1] for position, rank in enumerate(FACTORS)}})
    frame.to_csv(output / "standalone_v2_daily.csv", index=False)
    print(json.dumps({"dates": len(selected), "inside_rate": summary["same_day_quotes"]["inside_rate"],
        "span_share_p8": shares[4], "span_share_p16": shares[5], "output": str(output)}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "run", "calibration", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    summarize(reference=args.reference, run=args.run, calibration=args.calibration, output=args.output)


if __name__ == "__main__":
    main()
