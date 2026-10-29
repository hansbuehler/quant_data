"""Run the one-day SANOS historical-PCA hedge objective grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import run_daily

OBJECTIVE_CELLS = (
    ("mean-variance", 1000),
    *(("entropy", risk) for risk in (1, 10, 100, 1000)),
    *(("power-exponential", risk) for risk in (1, 10, 100, 1000)),
    *(("cvar-oce", risk) for risk in (1, 10, 100, 1000)),
)


def cell_name(objective: str, risk_parameter: int) -> str:
    """Give each objective and risk setting an unambiguous output directory."""
    return f"{objective.removesuffix('-oce')}-lambda{risk_parameter}"


def certified_cell(cell_root: Path) -> bool:
    """Check persisted results independently of an earlier runner exit code."""
    summary_path = cell_root / "run/summary.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    completed_origins = summary.get("completed_origins", ())
    record_origins = {
        path.stem for path in (cell_root / "run/records").glob("2024-*.json")
    }
    return (
        len(completed_origins) == 44
        and set(completed_origins) == record_origins
        and summary.get("result_count") == 440
        and summary.get("nonconverged_results") == 0
        and summary.get("fit_cap_hits") == 0
        and summary.get("executed_cap_hits") == 0
    )


def run_grid(*, dataset: Path, manifest_sha256: str, output_root: Path) -> None:
    """Resume each cell and retain its exact numerical-gate result."""
    if not dataset.is_dir() or len(manifest_sha256) != 64:
        raise ValueError("dataset and manifest SHA-256 must identify the public dataset")
    if not default_public_candidate_path().is_dir() or not callable(run_daily.main):
        raise RuntimeError("detached environment cannot discover public data and runner")
    print("detached environment preflight OK", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "GRID_STATUS.json"
    identity = {
        "schema": "sanos-normal-statistical-pca-objective-grid-v1",
        "dataset": str(dataset.resolve()),
        "manifest_sha256": manifest_sha256,
        "evaluation_start": "2024-04-01",
        "evaluation_end": "2024-05-31",
        "target_maturity_business_days": 1,
        "cells": [cell_name(*cell) for cell in OBJECTIVE_CELLS],
    }
    identity_path = output_root / "GRID_IDENTITY.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("PCA objective grid identity differs from existing output")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")

    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    failures = []
    for completed, (objective, risk_parameter) in enumerate(OBJECTIVE_CELLS, start=1):
        name = cell_name(objective, risk_parameter)
        cell_root = output_root / name
        cell_root.mkdir(parents=True, exist_ok=True)
        command = (
            sys.executable,
            "-u",
            "-m",
            "quantdata.sdlv.daily.front_spot.hedging.runners.run_daily",
            "--model", "sanos-normal-statistical-pca",
            "--dataset", str(dataset),
            "--dataset-manifest-sha256", manifest_sha256,
            "--start", "2024-04-01",
            "--end", "2024-05-31",
            "--target-maturity", "1",
            "--objective", objective,
            "--risk-parameter", str(risk_parameter),
            "--mean-reduction", "demean",
            "--position-cap", "1000",
            "--label", f"sanos-normal-pca-{name}-aprmay-20261028",
            "--output", str(cell_root / "run"),
            "--progress-every", "1",
        )
        reused = certified_cell(cell_root)
        if reused:
            print(f"REUSE {completed}/{len(OBJECTIVE_CELLS)} {name}", flush=True)
            result = None
        else:
            print(f"START {completed}/{len(OBJECTIVE_CELLS)} {name}", flush=True)
            with (cell_root / "job.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        certified = certified_cell(cell_root)
        status[name] = {
            "status": "complete" if certified else "failed",
            "exit_code": result.returncode if result is not None else None,
            "completed_dates": len(tuple((cell_root / "run/records").glob("2024-*.json"))),
            "log": str(cell_root / "job.log"),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_source": "existing_certified_results" if reused else "runner",
        }
        temporary = status_path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)
        print(f"{name}: {status[name]['status']} ({status[name]['completed_dates']}/44)",
              flush=True)
        if not certified:
            failures.append(name)
    if failures:
        raise RuntimeError(f"PCA objective cells failed certification: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    run_grid(**vars(arguments))


if __name__ == "__main__":
    main()
