"""Controlled eight-versus-sixteen-factor audit of the retained floating-DLV fits."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import fields
from datetime import date
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedge2.dynamics import DynamicsConfig, FittedDynamics
from quantdata.hedge2.rolling import load_dynamics, run_rolling
from quantdata.hedging.model_audit.refresh import refresh_report
from quantdata.hedging.model_audit.run import _hash, _json, retained_dynamics_path
from quantdata.hedging.model_audit.variance_reconciliation import reconcile

BASE_MODELS = ("DYSANOS ordinary", "DYSANOS spot-orthogonal")
MODELS = (*BASE_MODELS, *(f"{model} (16 factors)" for model in BASE_MODELS))


def prepare(source: Path, output: Path) -> dict:
    """Snapshot the exact reference and completed eight-factor comparison records."""
    source_protocol = json.loads((source / "PROTOCOL.json").read_text())
    protocol = {"schema": "dysanos-rank-comparison-v1", "source_run": str(source),
        "source_protocol_sha256": _hash(source / "PROTOCOL.json"),
        "ranks": [8, 16], "models": list(MODELS), "rolling_transitions": 252,
        "paths": source_protocol["paths"], "sessions": source_protocol["sessions"],
        "seed": source_protocol["seed"], "spot_law": "same paired empirical spot innovations",
        "shocks_rescaled": False, "decoder_changed": False,
        "tail_diffusion": False, "eight_factor_fits_retrained": False}
    if (protocol["paths"], protocol["sessions"], protocol["seed"]) != (32, 115, 20_261_111):
        raise ValueError("The source does not use the common path protocol")
    path = output / "RANK_PROTOCOL.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("The rank-comparison directory belongs to another protocol")
        return protocol
    output.mkdir(parents=True, exist_ok=True)
    inputs = []
    for name in ("reference_history.npz", "frozen_observable_pca.npz", "observable_reference.json"):
        shutil.copy2(source / name, output / name)
        inputs.append({"path": str(source / name), "sha256": _hash(source / name)})
    dates = json.loads((source / "observable_reference.json").read_text())["dates"]
    if len(dates) != 44:
        raise ValueError("The comparison requires the same 44 April--May origins")
    for model in BASE_MODELS:
        for directory in ("daily", "persistence"):
            for day in dates:
                original = source / directory / model.replace(" ", "_") / f"{day}.json"
                record = json.loads(original.read_text())
                if directory == "daily" and record.get("status") != "scored":
                    raise ValueError(f"The eight-factor comparator is incomplete: {original}")
                if (directory == "persistence" and record.get("status") in ("scored", "model_invalid")
                        and (record["steps"], record["path_count"], record["seed"]) != (115, 32, 20_261_111)):
                    raise ValueError(f"The eight-factor path protocol differs: {original}")
                destination = output / directory / model.replace(" ", "_") / original.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
                if directory == "persistence" and record.get("status") not in ("scored", "model_invalid"):
                    _json(output / "superseded_path_errors" / model.replace(" ", "_") / original.name, record)
                inputs.append({"path": str(original), "sha256": _hash(original)})
    _json(output / "COMPARATOR_INPUTS.json", inputs)
    _json(path, protocol)
    return protocol


def validate_matched_fits(previous: FittedDynamics, current: FittedDynamics) -> None:
    """Rank is the treatment; dates, empirical spot shocks and native geometry must match."""
    if (previous.as_of_date, previous.training_start, previous.training_end) != (
            current.as_of_date, current.training_start, current.training_end):
        raise ValueError("Eight- and sixteen-factor training windows differ")
    if previous.directions.shape[1] != 8 or current.directions.shape[1] != 16:
        raise ValueError("The controlled comparison requires ranks eight and sixteen")
    if (not np.array_equal(previous.state_mean, current.state_mean)
            or not np.allclose(previous.spot_state_loading, current.spot_state_loading, atol=1e-13, rtol=1e-13)
            or not np.allclose(previous.residuals[:, 0], current.residuals[:, 0], atol=1e-14, rtol=1e-13)):
        raise ValueError("Native centering, spot loading or paired spot shocks changed")
    omitted = previous.directions - current.directions @ (current.directions.T @ previous.directions)
    if np.linalg.norm(omitted) > 1e-8:
        raise ValueError("The old eight-factor subspace is not contained in the new sixteen-factor span")


def _fit(root: Path, output: Path) -> None:
    source = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    for model in BASE_MODELS:
        variant = "orthogonal" if "spot-orthogonal" in model else "ordinary"
        baseline = retained_dynamics_path(root, model, date(2024, 4, 1))
        original = json.loads((baseline.parent.parent / "config.json").read_text())
        values = {field.name: original[field.name] for field in fields(DynamicsConfig)}
        values["rank"] = 16
        config = DynamicsConfig(**values)
        print(f"Fit start {model}: rank=16, 252 causal transitions, prior-origin alignment", flush=True)
        run_rolling(daily_directory=source, observations_path=source / "observations.parquet",
            output_directory=output / "fits" / variant, start=date(2024, 3, 28), end=date(2024, 5, 31),
            config=config, history_start=date(2022, 1, 1))
        dates = json.loads((output / "observable_reference.json").read_text())["dates"]
        for day in map(date.fromisoformat, dates):
            validate_matched_fits(load_dynamics(retained_dynamics_path(root, model, day)),
                load_dynamics(retained_dynamics_path(root, f"{model} (16 factors)", day, output=output)))
        print(f"Fit complete {model}: all 44 training windows and spot shocks match", flush=True)


def _run_module(module: str, output: Path, *arguments: str) -> None:
    subprocess.run([sys.executable, "-u", "-m", f"quantdata.hedging.model_audit.{module}",
        "--output", str(output), *arguments], check=True)


def comparison_results(output: Path) -> dict:
    """Summarize the entire fixed window, retaining missing and failed cases explicitly."""
    protocol = json.loads((output / "RANK_PROTOCOL.json").read_text())
    summaries = json.loads((output / "summary.json").read_text())
    rows = []
    for model in MODELS:
        records = [json.loads(path.read_text()) for path in sorted((output / "daily" / model.replace(" ", "_")).glob("*.json"))]
        scored = [record for record in records if record.get("status") == "scored"]
        paths = [json.loads(path.read_text()) for path in sorted((output / "persistence" / model.replace(" ", "_")).glob("*.json"))]
        completed = [record for record in paths if record.get("status") in ("scored", "model_invalid")]
        summary = summaries[model]
        if len(scored) != 44 or len(completed) != 44:
            raise ValueError(f"Incomplete comparison for {model}: daily={len(scored)}/44 recursive={len(completed)}/44")
        rows.append({"model": model, "rank": 16 if model.endswith("(16 factors)") else 8,
            "origins": len(scored), "recursive_origins": len(completed),
            "native_coverage_mean": float(np.mean([record["diagnostics"]["native_pca_coverage"] for record in scored])),
            "conditional_trace_ratio": summary["conditional_covariance"]["trace_ratio"],
            "pooled_trace_ratio": summary["pooled_covariance"]["trace_ratio"],
            "pooled_covariance_error": summary["pooled_covariance"]["cov_frob_rel"],
            "spectral_radius_max": max(record["diagnostics"]["stability"]["spectral_radius"] for record in scored),
            "forecast_rmse": summary["forecast_rmse"], "persistence_rmse": summary["persistence_rmse"],
            "pc1_tail_counts": [summary["tails"]["PC1"][str(probability)]["exceptions"] for probability in (.01, .05, .95, .99)],
            "invalid_path_rate": float(np.mean([record["diagnostics"]["invalid_path_rate"] for record in completed])),
            "extreme_surface_rate": float(np.mean([record["diagnostics"]["extreme_surface_rate"] for record in completed])),
            "daily_decoder_lower_clips": sum(record["diagnostics"]["lower_clips"] for record in scored),
            "daily_decoder_upper_clips": sum(record["diagnostics"]["upper_clips"] for record in scored)})
    reconciliation = [json.loads(path.read_text()) for path in sorted((output / "variance_reconciliation").glob("*/*.json"))]
    manifests = [{"path": str(path), "sha256": _hash(path)} for path in sorted((output / "fits").glob("*/config.json"))]
    manifests.extend({"path": str(path), "sha256": _hash(path)} for path in sorted((output / "fits").glob("*/days/*.npz")))
    return {"protocol": protocol, "run_directory": str(output), "models": rows,
        "reconciliation": reconciliation, "fit_files": manifests,
        "scope": "controlled factor-count experiment; original native decoder/guards and reference smoothness mismatch remain unchanged"}


def _write_markdown(output: Path, result: dict) -> None:
    rows = ["# DYSANOS: eight versus sixteen factors", "",
        "Same 44 April–May 2024 origins, 252 causal training transitions, unchanged native decoder and pricing conventions. "
        "Each new rolling PCA basis is aligned to its previous origin. March 28 is a causal alignment seed, not a scored origin. "
        "No tail multiplier or shock inflation is introduced. All recursive tests use 32 paths, 115 sessions and seed 20261111.", "",
        "| Model | Days | Native coverage | Conditional variance ratio | Pooled variance ratio | Covariance error | Max AR radius |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["models"]:
        rows.append(f"| {row['model']} | {row['origins']} | {row['native_coverage_mean']:.2%} | "
            f"{row['conditional_trace_ratio']:.4f} | {row['pooled_trace_ratio']:.4f} | "
            f"{row['pooled_covariance_error']:.4f} | {row['spectral_radius_max']:.4f} |")
    rows.extend(["", "Variance ratios compare decoded log-IV forecasts with the same observed reference. "
        "The conditional denominator is unconditional observed variance, so its ratio need not equal one. "
        "Pooled covariance also includes variation in conditional means. Native coverage is not an IV-variance guarantee. "
        "The model inputs retain smoothness 0.1 and the common observable reference smoothness 0.25; this comparison does not resolve that difference.", "",
        "The JSON retains covariance errors, tail counts, recursive invalid/extreme rates and decoder clipping counts. "
        "The three-origin same-shock reconciliation is a decoder/projection ablation, not a fitted full-rank alternative or held-out calibration test."])
    (output / "RANK_COMPARISON.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with FileLock(arguments.output / ".rank-comparison.lock"):
        prepare(arguments.source, arguments.output)
        _fit(arguments.root, arguments.output)
        _run_module("run", arguments.output, "--models", *MODELS)
        _run_module("persistence", arguments.output, "--models", *MODELS, "--paths", "32", "--steps", "115")
        for model in MODELS:
            for day in (date(2024, 4, 1), date(2024, 4, 30), date(2024, 5, 31)):
                destination = arguments.output / "variance_reconciliation" / model.replace(" ", "_") / f"{day}.json"
                if not destination.exists():
                    _json(destination, reconcile(arguments.root, day=day, model=model, output=arguments.output))
                print(f"Rank reconciliation {model} {day} completed elapsed={time.monotonic()-started:.1f}s", flush=True)
        result = comparison_results(arguments.output)
        _json(arguments.output / "rank_comparison.json", result)
        _write_markdown(arguments.output, result)
        if arguments.publish is not None:
            _json(arguments.publish / "rank_comparison.json", result)
            refresh_report(arguments.publish)
        print(f"Rank comparison complete: {arguments.output} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
