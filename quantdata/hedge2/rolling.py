"""Resumable CPU estimation of a new floating-state PCA/AR(1) each date."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.hedge2.dynamics import DynamicsConfig, FittedDynamics, fit_dynamics
from quantdata.hedge2.period import ExperimentPeriod


def load_dynamics(path: Path) -> FittedDynamics:
    with np.load(path, allow_pickle=False) as saved:
        values = {field.name: saved[field.name] for field in fields(FittedDynamics)}
        for name in ("as_of_date", "training_start", "training_end", "initialisation_date"):
            value = str(values[name].item())
            values[name] = None if value == "" else date.fromisoformat(value)
        for name in ("unconstrained_spectral_radius", "constrained_spectral_radius", "training_rmse", "persistence_rmse",
                     "spot_conditioned_training_rmse", "spot_conditioned_persistence_rmse"):
            values[name] = float(values[name])
    return FittedDynamics(**values)


def save_dynamics(model: FittedDynamics, path: Path) -> None:
    values = asdict(model)
    for name in ("as_of_date", "training_start", "training_end", "initialisation_date"):
        values[name] = "" if values[name] is None else values[name].isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def run_rolling(*, daily_directory: Path, observations_path: Path,
                output_directory: Path, start: date, end: date,
                config: DynamicsConfig,
                history_start: date = date(2022, 1, 1)) -> dict[str, object]:
    period = ExperimentPeriod(history_start=history_start, evaluation_start=start, requested_end=end)
    observations = pd.read_parquet(observations_path).set_index("date")
    observations.index = pd.to_datetime(observations.index).date
    dates = []
    raw_states = []
    for path in sorted((daily_directory / "days").glob("*.npz")):
        day = date.fromisoformat(path.stem)
        if day < history_start or day > end:
            continue
        with np.load(path, allow_pickle=False) as saved:
            if not bool(saved["reconstruction_passed"]):
                continue
            dates.append(day)
            raw_states.append(saved["raw_u"])
    if not dates:
        raise ValueError("No validated daily states available")
    states = np.asarray(raw_states)
    dated_observations = observations.loc[dates]
    sessions = dated_observations["session_index"].to_numpy(dtype=int)
    spot_returns = dated_observations["normalized_spot_return_into_date"].to_numpy(dtype=float)
    configuration = json.loads(json.dumps({"version": DynamicsConfig.version.unique_id48, **asdict(config),
                                         "daily_directory": str(daily_directory), "observations": str(observations_path),
                                         "history_start": history_start.isoformat(),
                                         "evaluation_start": start.isoformat(), "requested_end": end.isoformat()}))
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = output_directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("The output directory belongs to different rolling dynamics")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    previous = None
    started = time.monotonic()
    rows = []
    skipped = []
    candidate_days = [day for day in dates if start <= day <= end]
    for index, day in enumerate(candidate_days):
        path = output_directory / "days" / f"{day.isoformat()}.npz"
        if path.exists():
            model = load_dynamics(path)
        else:
            try:
                model = fit_dynamics(
                    dates=tuple(dates), states=states, spot_returns_into_date=spot_returns,
                    session_indices=sessions, as_of_date=day, config=config, previous=previous,
                )
            except ValueError as error:
                if str(error).startswith("Insufficient exact one-session"):
                    skipped.append({"date": day.isoformat(), "reason": str(error)})
                    continue
                raise
            save_dynamics(model, path)
        previous = model
        rows.append({
            "date": day.isoformat(), "training_start": model.training_start.isoformat(),
            "explained_variance": float(model.explained_variance_fraction.sum()),
            "raw_radius": model.unconstrained_spectral_radius,
            "radius": model.constrained_spectral_radius,
            "training_rmse": model.training_rmse,
            "persistence_rmse": model.persistence_rmse,
            "spot_conditioned_training_rmse": model.spot_conditioned_training_rmse,
            "spot_conditioned_persistence_rmse": model.spot_conditioned_persistence_rmse,
        })
        if len(rows) % 25 == 0:
            elapsed = time.monotonic() - started
            print(f"rolling {len(rows)} fits; date={day} coverage={rows[-1]['explained_variance']:.3f} "
                  f"radius={model.constrained_spectral_radius:.4f} elapsed_s={elapsed:.1f} "
                  f"remaining_dates={len(candidate_days)-index-1}", flush=True)
    pd.DataFrame(rows).to_parquet(output_directory / "fit_diagnostics.parquet", index=False)
    coverage = period.coverage(tuple(dates), as_of_date=date.today())
    result = {"models": len(rows), "available_evaluation_dates": len(candidate_days),
              "fitted_start": rows[0]["date"] if rows else None,
              "fitted_end": rows[-1]["date"] if rows else None,
              "skipped_evaluation_dates": skipped,
              "coverage": coverage,
              "elapsed_seconds": time.monotonic() - started}
    (output_directory / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (output_directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-directory", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--history-start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--spot-orthogonal", action="store_true")
    args = parser.parse_args()
    print(run_rolling(daily_directory=args.daily_directory, observations_path=args.observations,
                      output_directory=args.output_directory, start=args.start, end=args.end,
                      config=DynamicsConfig(spot_orthogonal=args.spot_orthogonal),
                      history_start=args.history_start))


if __name__ == "__main__":
    main()
