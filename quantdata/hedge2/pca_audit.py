"""Attribute rolling PCA variation to numerically guarded DLV coordinates."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.hedge2.rolling import load_dynamics


def audit_pca(*, daily_directory: Path, dynamics_directory: Path,
              output_directory: Path) -> dict[str, object]:
    """Keep PCA unchanged; split its measured variation by endpoint guard flags."""
    dynamics_config = json.loads((dynamics_directory / "config.json").read_text())
    daily_config = json.loads((daily_directory / "config.json").read_text())
    history_start = date.fromisoformat(dynamics_config["history_start"])
    requested_end = date.fromisoformat(dynamics_config["requested_end"])
    states, flags, dates = [], [], []
    for path in sorted((daily_directory / "days").glob("*.npz")):
        day = date.fromisoformat(path.stem)
        if not history_start <= day <= requested_end:
            continue
        with np.load(path, allow_pickle=False) as saved:
            if not bool(saved["reconstruction_passed"]):
                continue
            raw = saved["raw_dlvsqs"]
            unidentified = ~saved["identified"]
            floored = saved["dlvsqs"] < daily_config["min_local_vol"]**2
            capped = raw > daily_config["max_local_vol"]**2
            low_mass = saved["densities"][:, 1:-1] <= daily_config["density_floor"]
            extrapolated = (saved["expiries_y"] + ONE_MINUTE
                            < float(saved["source_first_business_dte"]) / DAYS_PER_YEAR)
            by_type = {"unidentified": unidentified, "floored": floored,
                       "capped": capped, "low_mass": low_mass,
                       "extrapolated": np.broadcast_to(extrapolated[:, None], raw.shape)}
            by_type["guarded"] = unidentified | floored | capped | low_mass
            flags.append({name: np.c_[mask, np.zeros(len(mask), dtype=bool)].reshape(-1)
                          for name, mask in by_type.items()})
            states.append(saved["raw_u"])
            dates.append(day)
    values = np.asarray(states)
    masks = {name: np.asarray([row[name] for row in flags]) for name in flags[0]}
    observations = pd.read_parquet(dynamics_config["observations"]).set_index("date")
    observations.index = pd.to_datetime(observations.index).date
    observations = observations.loc[dates]
    sessions = observations["session_index"].to_numpy(dtype=int)
    spot = observations["normalized_spot_return_into_date"].to_numpy(dtype=float)
    rows, factor_rows, feature_rows = [], [], []
    for path in sorted((dynamics_directory / "days").glob("*.npz")):
        model = load_dynamics(path)
        eligible = np.array([index for index in range(1, len(dates))
                             if dates[index] <= model.as_of_date and sessions[index] - sessions[index - 1] == 1])
        selected = eligible[-dynamics_config["window_transitions"]:]
        changes = values[selected] - values[selected - 1]
        centered = changes - changes.mean(axis=0)
        if dynamics_config["spot_orthogonal"]:
            centered -= (spot[selected] - spot[selected].mean())[:, None] * model.spot_state_loading
        energy = centered**2
        total_energy = float(energy.sum())
        row = {"date": model.as_of_date.isoformat(), "training_start": model.training_start.isoformat(),
               "explained_variance": float(model.explained_variance_fraction.sum())}
        for name, mask in masks.items():
            endpoints = mask[selected] | mask[selected - 1]
            row[f"{name}_transition_fraction"] = float(endpoints.mean())
            row[f"{name}_change_energy_fraction"] = float(energy[endpoints].sum() / total_energy)
        rows.append(row)
        endpoints = masks["guarded"][selected] | masks["guarded"][selected - 1]
        factor_changes = centered @ model.directions
        flagged_changes = (centered * endpoints) @ model.directions
        other_changes = factor_changes - flagged_changes
        frequency = masks["unidentified"][np.r_[selected[0] - 1, selected]].mean(axis=0)
        for factor in range(model.directions.shape[1]):
            denominator = float(np.sum(factor_changes[:, factor]**2))
            factor_rows.append({
                "date": model.as_of_date.isoformat(), "factor": factor + 1,
                "frequently_unidentified_loading_fraction": float(np.sum(model.directions[:, factor]**2 * frequency)),
                "guarded_score_energy_ratio": float(np.sum(flagged_changes[:, factor]**2) / denominator),
                "other_score_energy_ratio": float(np.sum(other_changes[:, factor]**2) / denominator),
                "cross_score_energy_ratio": float(2. * np.sum(flagged_changes[:, factor] * other_changes[:, factor]) / denominator),
            })
        feature_energy = energy.sum(axis=0)
        switches = np.count_nonzero(masks["unidentified"][selected] != masks["unidentified"][selected - 1], axis=0)
        stride = daily_config["num_inner_strikes"] + 1
        for feature in np.argsort(feature_energy)[-10:][::-1]:
            expiry_index, node = divmod(int(feature), stride)
            feature_rows.append({
                "date": model.as_of_date.isoformat(), "feature": int(feature),
                "business_dte": daily_config["business_dtes"][expiry_index],
                "coordinate": "forward_vol" if node == stride - 1 else "dlv_squared",
                "strike_node": None if node == stride - 1 else node,
                "change_energy_fraction": float(feature_energy[feature] / total_energy),
                "identification_switches": int(switches[feature]),
            })
        if len(rows) % 100 == 0:
            print(f"PCA guard audit {len(rows)} windows; date={model.as_of_date}", flush=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    windows = pd.DataFrame(rows)
    factors = pd.DataFrame(factor_rows)
    windows.to_parquet(output_directory / "windows.parquet", index=False)
    factors.to_parquet(output_directory / "factors.parquet", index=False)
    pd.DataFrame(feature_rows).to_parquet(output_directory / "top_features.parquet", index=False)
    summary = {
        "daily_directory": str(daily_directory), "dynamics_directory": str(dynamics_directory),
        "windows": len(rows), "start": windows.date.min(), "end": windows.date.max(),
        "median_guarded_transition_fraction": float(windows.guarded_transition_fraction.median()),
        "median_guarded_change_energy_fraction": float(windows.guarded_change_energy_fraction.median()),
        "maximum_guarded_change_energy_fraction": float(windows.guarded_change_energy_fraction.max()),
        "median_floored_change_energy_fraction": float(windows.floored_change_energy_fraction.median()),
        "median_unidentified_change_energy_fraction": float(windows.unidentified_change_energy_fraction.median()),
        "pca_is_unchanged": True,
        "score_attribution": "guarded energy + other energy + cross term = total factor-score energy",
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-directory", type=Path, required=True)
    parser.add_argument("--dynamics-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_pca(daily_directory=args.daily_directory, dynamics_directory=args.dynamics_directory,
                               output_directory=args.output_directory), indent=2), flush=True)


if __name__ == "__main__":
    main()
