"""Independent persisted-state audit and concise hedge2 preparation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE


def audit_daily_states(directory: Path) -> dict[str, object]:
    configuration = json.loads((directory / "config.json").read_text())
    tolerance = configuration["reconstruction_tolerance"]
    rows = []
    for index, path in enumerate(sorted((directory / "days").glob("*.npz"))):
        with np.load(path, allow_pickle=False) as saved:
            support = saved["support_strikes"]
            density = saved["decoded_densities"]
            source_calls = saved["source_call_prices"]
            calculated = density @ np.maximum(support[None, :] - support[:, None], 0.).T
            error = float(np.max(np.abs(calculated - source_calls)))
            mass_error = float(np.max(np.abs(density.sum(axis=1) - 1.)))
            mean_error = float(np.max(np.abs(density @ support - 1.)))
            expected_native = np.c_[np.log(saved["training_dlvsqs"]), np.log(saved["forward_vols"])].reshape(-1)
            if (error > tolerance or mass_error > 1e-8 or mean_error > 1e-8
                    or not np.isfinite(expected_native).all()
                    or not np.array_equal(expected_native, saved["raw_u"])
                    or not np.isfinite(density).all() or density.min() < -1e-12
                    or str(saved["date"].item()) != path.stem):
                raise ValueError(f"Independent stored-state audit failed on {path.stem}")
            front = float(saved["source_first_business_dte"])
            rows.append({
                "date": path.stem, "max_source_price_error": error,
                "mass_error": mass_error, "mean_error": mean_error,
                "min_probability": float(density.min()),
                "source_first_business_dte": front,
                "extrapolated_tenors": int(np.sum(saved["expiries_y"] + ONE_MINUTE < front / DAYS_PER_YEAR)),
                "regularization_weight": float(saved["density_regularization_weight"]),
                "source_calendar_correction": float(saved["source_calendar_correction"]),
                "low_mass_nodes": int(saved["low_mass_nodes"]),
                "undefined_nodes": int(saved["undefined_nodes"]),
                "floored_nodes": int(saved["floored_nodes"]), "capped_nodes": int(saved["capped_nodes"]),
                "extreme_nodes": int(saved["extreme_nodes"]),
                "max_raw_local_vol": float(saved["max_raw_local_vol"]),
            })
        if (index + 1) % 500 == 0:
            print(f"audit {index + 1} dates", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / "independent_audit.parquet", index=False)
    summary = {
        "dates": len(frame), "start": frame.date.min(), "end": frame.date.max(),
        "maximum_source_price_error": float(frame.max_source_price_error.max()),
        "median_source_price_error": float(frame.max_source_price_error.median()),
        "maximum_mass_error": float(frame.mass_error.max()),
        "maximum_mean_error": float(frame.mean_error.max()),
        "dates_with_price_regularization": int((frame.regularization_weight > 0).sum()),
        "largest_regularization_weight": float(frame.regularization_weight.max()),
        "dates_with_source_front_beyond_1d": int((frame.source_first_business_dte > 1. + ONE_MINUTE * DAYS_PER_YEAR).sum()),
        "dates_with_capping": int((frame.capped_nodes > 0).sum()),
        "dates_with_extreme_values_flagged": int((frame.extreme_nodes > 0).sum()),
    }
    (directory / "independent_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_daily_states(args.daily_directory), indent=2), flush=True)


if __name__ == "__main__":
    main()
