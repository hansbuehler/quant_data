"""Resume a numerical-only optimizer fix with explicit inherited-cell provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

from quantdata.hedge2.objective_run import load_checkpoint, run_objectives, save_checkpoint


def migrate_checkpoints(*, input_directory: Path, output_directory: Path,
                        numerical_fix: str = "parametric_zero_endpoint") -> dict[str, object]:
    """Keep accepted cells exactly; an old failed cell is never counted as completed.

    Only the named, reviewed numerical changes are permitted. Neither fix
    changes data, costs, caps, target payouts or the economic objective.
    """
    source_configuration = json.loads((input_directory / "config.json").read_text())
    if numerical_fix not in ("parametric_zero_endpoint", "cash_unit_power_cone"):
        raise ValueError("Choose a reviewed numerical-only checkpoint migration")
    if numerical_fix == "parametric_zero_endpoint" and source_configuration["method"] != "parametric":
        raise ValueError("This migration only covers the reviewed parametric zero-endpoint numerical fix")
    if source_configuration["method"] not in ("parametric", "statistical", "historical"):
        raise ValueError("The numerical migration requires a known scenario or parametric objective run")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("Choose a new, empty output directory for a documented implementation migration")
    output_directory.mkdir(parents=True, exist_ok=True)
    inherited = 0
    artifacts = {}
    for checkpoint in sorted(input_directory.glob("*/checkpoint.npz")):
        rows, positions = load_checkpoint(checkpoint)
        if not all(row["optimization_certified"] for row in rows):
            raise ValueError(f"Uncertified cells must not be inherited: {checkpoint}")
        source_manifest = checkpoint.parent / "completed.json"
        completed = json.loads(source_manifest.read_text()) if source_manifest.exists() else None
        if completed is not None:
            for name, digest in completed["artifacts_sha256"].items():
                if hashlib.sha256((checkpoint.parent / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Completed source artifact changed before migration: {checkpoint.parent / name}")
        directory = output_directory / checkpoint.parent.name
        directory.mkdir()
        for artifact in checkpoint.parent.iterdir():
            if artifact.is_file() and artifact.name != "checkpoint.npz":
                shutil.copy2(artifact, directory / artifact.name)
        for row in rows:
            row["inherited_implementation_run"] = str(input_directory)
        save_checkpoint(directory / "checkpoint.npz", rows, positions)
        if completed is not None:
            completed["inherited_completed_manifest_sha256"] = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
            completed["artifacts_sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in completed["artifacts_sha256"]}
            (directory / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
        artifacts[checkpoint.parent.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        inherited += len(rows)
    reasons = {"parametric_zero_endpoint": "Replace the artificial k=lambda*1e-8 QP with the exact k->0+ free-asset variance limit; unchanged economic optimization problem.",
        "cash_unit_power_cone": "Equivalent cash-unit power cone with scalar objective normalization and cash-eliminated fallback; unchanged returns, target, costs, caps and economic objective."}
    record = {"parent_run": str(input_directory), "inherited_certified_cells": inherited,
        "parent_configuration_sha256": hashlib.sha256((input_directory / "config.json").read_bytes()).hexdigest(),
        "parent_checkpoint_sha256": artifacts,
        "numerical_fix": numerical_fix, "reason": reasons[numerical_fix]}
    (output_directory / "migration.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--numerical-fix", choices=("parametric_zero_endpoint", "cash_unit_power_cone"),
                        default="parametric_zero_endpoint")
    args = parser.parse_args()
    parent = json.loads((args.input_directory / "config.json").read_text())
    record = migrate_checkpoints(input_directory=args.input_directory, output_directory=args.output_directory,
        numerical_fix=args.numerical_fix)
    print(json.dumps(record, indent=2), flush=True)
    arguments = {name: None if parent[name] is None else Path(parent[name]) for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
        "market_directory", "daily_directory", "dynamics_directory")}
    result = run_objectives(method=parent["method"], **arguments, output_directory=args.output_directory,
        dates=tuple(date.fromisoformat(value) for value in parent["dates"]),
        objective_gap_tolerance=parent["objective_gap_tolerance"], diagnostic_only=parent["diagnostic_only"],
        itm_premium_book=parent.get("itm_premium_book", False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
