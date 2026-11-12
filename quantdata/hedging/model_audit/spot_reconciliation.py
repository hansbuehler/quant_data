"""Check joint spot/state replay and native PCA energy in the retained DYSANOS fits."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedging.model_audit.run import _json, retained_dynamics_path
from quantdata.hedging.model_audit.tail import _training_pairs


def replay_training_changes(before: np.ndarray, after: np.ndarray,
                            fit: FittedDynamics) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted paired replay and its exact retained-projection identity."""
    directions = fit.directions
    scores = (before - fit.state_mean) @ directions
    forecast_changes = (fit.ar_intercept + scores @ fit.ar_matrix.T - scores) @ directions.T
    spot = fit.residuals[:, :1]
    replay = forecast_changes + spot * fit.spot_state_loading + fit.residuals[:, 1:] @ directions.T
    projected_spot = (fit.spot_state_loading @ directions) @ directions.T
    expected = ((after - before) @ directions) @ directions.T + spot * (fit.spot_state_loading - projected_spot)
    return replay, expected


def native_energy(before: np.ndarray, after: np.ndarray, fit: FittedDynamics,
                  *, config: DlvConfig) -> dict:
    """Use population centered-change energy; do not alter or rescale the shocks."""
    changes = after - before
    centered = changes - changes.mean(axis=0)
    spot_component = fit.residuals[:, :1] * fit.spot_state_loading
    projected = ((centered - spot_component) @ fit.directions) @ fit.directions.T + spot_component
    shape = (len(before), len(config.business_dtes), config.num_inner_strikes + 1)
    original = centered.reshape(shape)
    compressed = projected.reshape(shape)
    floor = 2. * np.log(config.min_local_vol)
    crossings = ((before.reshape(shape)[:, :, :-1] <= floor + 1e-8)
                 != (after.reshape(shape)[:, :, :-1] <= floor + 1e-8))
    total_energy = float(np.sum(original**2))
    forward_energy = float(np.sum(original[:, :, -1]**2))
    return {
        "centered_native_energy": total_energy / len(before),
        "retained_native_energy_fraction": float(np.sum(compressed**2) / total_energy),
        "forward_vol_energy_fraction_of_native": forward_energy / total_energy,
        "retained_forward_vol_energy_fraction": float(np.sum(compressed[:, :, -1]**2) / forward_energy),
        "floor_crossing_fraction_of_dlv_observations": float(crossings.mean()),
        "floor_crossing_energy_fraction_of_native": float(np.sum(original[:, :, :-1]**2 * crossings) / total_energy),
        "floor_definition": "stored log DLV squared at or below the configured minimum, before XOR after",
        "scope": "unweighted native-coordinate energy, not option-price or implied-volatility energy",
    }


def _summary(records: list[dict]) -> dict:
    result = {}
    for model in sorted({record["model"] for record in records}):
        selected = [record for record in records if record["model"] == model]
        metrics = {}
        for name in selected[0]["native_energy"]:
            if isinstance(selected[0]["native_energy"][name], str):
                continue
            values = [record["native_energy"][name] for record in selected]
            metrics[name] = {"minimum": float(np.min(values)), "median": float(np.median(values)),
                             "maximum": float(np.max(values))}
        result[model] = {"origins": len(selected),
            "max_training_replay_identity_error": max(record["training_replay_max_absolute_error"] for record in selected),
            "max_spot_history_alignment_error": max(record["spot_history_max_absolute_error"] for record in selected),
            "native_energy": metrics}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = arguments.root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    values = json.loads((source / "config.json").read_text())
    config = DlvConfig(**{field.name: values[field.name] for field in fields(DlvConfig) if field.name in values})
    observations = pd.read_parquet(source / "observations.parquet").set_index("date")
    observations.index = pd.to_datetime(observations.index).date
    pairs = _training_pairs(arguments.root)
    origins = [date.fromisoformat(path.stem) for path in sorted(
        (arguments.audit / "daily/DYSANOS_ordinary_(16_factors)").glob("*.json"))]

    @lru_cache(maxsize=512)
    def native(day: date) -> np.ndarray:
        with np.load(source / f"days/{day}.npz", allow_pickle=False) as saved:
            return saved["raw_u"]

    records = []
    started = time.monotonic()
    for model in ("DYSANOS ordinary (16 factors)", "DYSANOS spot-orthogonal (16 factors)"):
        for origin in origins:
            fit = load_dynamics(retained_dynamics_path(arguments.root, model, origin, output=arguments.audit))
            window = [pair for pair in pairs if pair[1] <= origin][-len(fit.residuals):]
            if window[0][0] != fit.training_start or window[-1][1] != fit.training_end:
                raise ValueError("Retained dynamics and native-state chronology disagree")
            before = np.stack([native(pair[0]) for pair in window])
            after = np.stack([native(pair[1]) for pair in window])
            actual_spot = observations.loc[[pair[1] for pair in window],
                                           "normalized_spot_return_into_date"].to_numpy(dtype=float, copy=True)
            actual_spot -= actual_spot.mean()
            spot_error = float(np.max(np.abs(actual_spot - fit.residuals[:, 0])))
            replay, expected = replay_training_changes(before, after, fit)
            error = float(np.max(np.abs(replay - expected)))
            if error > 1e-10 or spot_error > 1e-14:
                raise ValueError(f"Joint spot/state replay failed: state={error}, spot={spot_error}")
            spot = fit.residuals[:, 0]
            effective_loading = fit.spot_state_loading + fit.directions @ (
                fit.residuals[:, 1:].T @ spot / (spot @ spot))
            innovations = (fit.residuals[:, :1] * fit.spot_state_loading
                           + fit.residuals[:, 1:] @ fit.directions.T)
            measured_loading = innovations.T @ spot / (spot @ spot)
            np.testing.assert_allclose(effective_loading, measured_loading, atol=1e-11, rtol=1e-11)
            record = {"model": model, "origin_date": str(origin),
                "training_start": str(fit.training_start), "training_end": str(fit.training_end),
                "training_replay_max_absolute_error": error, "spot_history_max_absolute_error": spot_error,
                "explicit_spot_loading_norm": float(np.linalg.norm(fit.spot_state_loading)),
                "effective_joint_innovation_spot_loading_norm": float(np.linalg.norm(effective_loading)),
                "native_energy": native_energy(before, after, fit, config=config),
                "identity": "replay = P(delta u) + r*(I-P)*beta; P=U U^T, paired centered spot r",
                "ordinary_spot_policy": "spot/factor dependence remains in paired innovations even when explicit beta is zero"}
            _json(arguments.output / model.replace(" ", "_") / f"{origin}.json", record)
            records.append(record)
            if len(records) % 10 == 0 or len(records) == 2 * len(origins):
                elapsed = time.monotonic() - started
                print(f"spot reconciliation {len(records)}/{2*len(origins)} elapsed={elapsed:.1f}s "
                      f"eta={elapsed*(2*len(origins)-len(records))/len(records):.1f}s", flush=True)
    result = {"models": _summary(records), "shocks_rescaled": False,
              "model_parameters_changed": False, "replay_scope": "in-sample algebraic identity, not out-of-sample accuracy"}
    _json(arguments.output / "SUMMARY.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
