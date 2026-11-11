"""Reconcile native PCA energy with decoded IV variance without scaling any shocks."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.run import (
    _dlv_surface,
    _json,
    retained_dynamics_path,
)
from quantdata.hedging.model_audit.tail import _training_pairs


def reconcile(root: Path, *, day: date, model: str, output: Path | None = None) -> dict:
    """Decode the same 252 historic shocks before and after PCA and AR projection."""
    source = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    fit = load_dynamics(retained_dynamics_path(root, model, day, output=output))
    rank = fit.directions.shape[1]
    values = json.loads((source / "config.json").read_text())
    config = DlvConfig(**{field.name: values[field.name] for field in fields(DlvConfig) if field.name in values})

    @lru_cache(maxsize=512)
    def native(origin: date) -> np.ndarray:
        with np.load(source / f"days/{origin}.npz", allow_pickle=False) as saved:
            return saved["raw_u"]

    pairs = [pair for pair in _training_pairs(root) if pair[1] <= day][-252:]
    before = np.stack([native(pair[0]) for pair in pairs])
    after = np.stack([native(pair[1]) for pair in pairs])
    changes = after - before
    changes -= changes.mean(axis=0)
    spot_component = fit.residuals[:, :1] * fit.spot_state_loading
    pca = ((changes - spot_component) @ fit.directions) @ fit.directions.T + spot_component
    innovations = fit.residuals[:, 1:] @ fit.directions.T + spot_component
    with np.load(source / f"days/{day}.npz", allow_pickle=False) as saved:
        state = DailyState(day=day, native=saved["raw_u"], reference_support=saved["support_strikes"], config=config)
    groups = {"full_centered_native_changes": changes, f"pca{rank}_centered_changes": pca,
              f"pca{rank}_ar_innovations": innovations,
              f"pca{rank}_ar_with_current_drift": innovations + fit.forecast_state_change(state.native)}
    results = {}
    for name, shocks in groups.items():
        surfaces, guards = _dlv_surface(state, state.native[None] + shocks)
        rows = surfaces.reshape(len(surfaces), -1)
        valid = np.isfinite(rows).all(axis=1)
        results[name] = {"native_centered_energy": float(np.mean(np.sum((shocks - shocks.mean(axis=0))**2, axis=1))),
            "decoded_log_iv_variance_trace": float(np.sum(np.var(rows[valid], axis=0, ddof=0))) if valid.any() else None,
            "valid_scenarios": int(valid.sum()), "guards": guards}
    full = results["full_centered_native_changes"]
    for result in results.values():
        result["native_energy_fraction_of_full"] = result["native_centered_energy"] / full["native_centered_energy"]
        result["decoded_variance_fraction_of_full"] = (
            result["decoded_log_iv_variance_trace"] / full["decoded_log_iv_variance_trace"]
            if full["decoded_log_iv_variance_trace"] else None)
    return {"model": model, "origin_date": str(day), "training_start": str(fit.training_start),
            "training_end": str(fit.training_end), "historic_scenarios": 252, "groups": results,
            "scope": "same current decoder and historic rows; diagnostic ablation, not a fitted full-rank alternative model",
            "shocks_rescaled": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, default=[date(2024, 4, 1), date(2024, 4, 30), date(2024, 5, 31)])
    arguments = parser.parse_args()
    for model in ("DYSANOS ordinary", "DYSANOS spot-orthogonal"):
        for day in arguments.dates:
            result = reconcile(arguments.root, day=day, model=model)
            _json(arguments.output / "variance_reconciliation" / model.replace(" ", "_") / f"{day}.json", result)
            print(json.dumps({"model": model, "date": str(day), "variance_fractions": {
                name: item["decoded_variance_fraction_of_full"] for name, item in result["groups"].items()}}), flush=True)


if __name__ == "__main__":
    main()
