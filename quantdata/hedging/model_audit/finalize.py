"""Finish every model before promoting sixteen-factor DYSANOS and retaining the deck."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from filelock import FileLock

from quantdata.hedging.model_audit.refresh import refresh_report
from quantdata.hedging.model_audit.run import _hash, _json

OTHER_MODELS = ("FuNVol", "VolGAN PCA8", "AD-Seq-Vol", "AD-Seq-Vol-FT", "GROMM-P", "GROMM-Q")
PRIMARY_MODELS = ("DYSANOS ordinary (16 factors)", "DYSANOS spot-orthogonal (16 factors)")
TAIL_MODELS = tuple(f"{model} (tail)" for model in PRIMARY_MODELS)
FINAL_MODELS = (*PRIMARY_MODELS, *OTHER_MODELS, *TAIL_MODELS)


def coverage_state(output: Path, models: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Count completed requested origins, including explicit model-law failures, not runner errors."""
    dates = json.loads((output / "observable_reference.json").read_text())["dates"]
    result = {}
    for model in models:
        result[model] = {}
        for kind in ("daily", "quote_forecasts", "persistence"):
            completed = 0
            for day in dates:
                path = output / kind / model.replace(" ", "_") / f"{day}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    accepted = ("scored", "model_invalid") if kind == "persistence" else ("scored",)
                    completed += record.get("status") in accepted
            result[model][kind] = completed
    return result


def repair_batch_error_records(source: Path, rank_run: Path) -> None:
    """Retain old whole-batch errors and replace them with the same paths' per-path diagnostics."""
    for model in ("DYSANOS_ordinary", "DYSANOS_spot-orthogonal"):
        for path in sorted((source / "persistence" / model).glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("status") != "error" or record.get("reason") != "Decoded DYSANOS ATM IV is unidentified":
                continue
            replacement = rank_run / "persistence" / model / path.name
            corrected = json.loads(replacement.read_text())
            if corrected.get("status") not in ("scored", "model_invalid"):
                raise ValueError(f"No complete per-path accounting for {replacement}")
            if (corrected["steps"], corrected["path_count"], corrected["seed"]) != (115, 32, 20_261_111):
                raise ValueError("The repaired diagnostic uses different paths")
            archive = source / "completion/superseded_batch_errors" / model / path.name
            _json(archive, record)
            corrected["superseded_batch_error"] = {"record": str(archive), "sha256": _hash(archive),
                "change": "per-path accounting of the same native paths; no replacement IVs or model changes"}
            shutil.copy2(replacement.with_suffix(".npz"), path.with_suffix(".npz"))
            _json(path, corrected)
            print(f"Recovered per-path diagnostics {model} {path.stem}; original error retained", flush=True)


def prepare(source: Path, rank_run: Path, output: Path) -> None:
    """Reuse completed model laws and paths; only the selected DYSANOS factor rank changes."""
    protocol = {"schema": "hedging-model-audit-final16-v3", "source_run": str(source),
        "rank_run": str(rank_run), "paths": 32, "sessions": 115, "seed": 20_261_111,
        "dysanos_rank": 16, "models": list(FINAL_MODELS), "other_models_retrained": False}
    destination = output / "PROTOCOL.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("The final output directory belongs to another protocol")
        return
    state = coverage_state(source, OTHER_MODELS)
    if any(value != 44 for counts in state.values() for value in counts.values()):
        raise ValueError("Other-model diagnostics must finish before promoting DYSANOS")
    output.mkdir(parents=True, exist_ok=True)
    inputs = []

    def copy_file(path: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        inputs.append({"path": str(path), "sha256": _hash(path), "destination": str(target)})

    for name in ("reference_history.npz", "frozen_observable_pca.npz", "observable_reference.json", "INFERENCE_MODELS.json"):
        copy_file(source / name, output / name)
    for directory in ("inference", "regenerated"):
        target = output / directory
        if not target.exists():
            target.symlink_to(source / directory, target_is_directory=True)
    fit_link = output / "fits"
    if not fit_link.exists():
        fit_link.symlink_to(rank_run / "fits", target_is_directory=True)
    for path in (source / "completion/ft-may/dates").glob("*.json"):
        copy_file(path, output / "completion/ft-may/dates" / path.name)
    for model in (*OTHER_MODELS, *PRIMARY_MODELS):
        origin = source if model in OTHER_MODELS else rank_run
        for kind in ("daily", "quote_forecasts", "persistence"):
            if model in PRIMARY_MODELS and kind == "quote_forecasts":
                continue
            for path in sorted((origin / kind / model.replace(" ", "_")).glob("*.json")):
                target = output / kind / model.replace(" ", "_") / path.name
                copy_file(path, target)
                arrays = path.with_suffix(".npz")
                if arrays.exists() and not target.with_suffix(".npz").exists():
                    target.with_suffix(".npz").symlink_to(arrays)
    for path in (rank_run / "variance_reconciliation").glob("*/*.json"):
        record = json.loads(path.read_text())
        if "(16 factors)" in record["model"]:
            copy_file(path, output / "variance_reconciliation" / path.parent.name / path.name)
    copy_file(rank_run / "rank_comparison.json", output / "rank_comparison.json")
    _json(output / "SOURCE_REUSE.json", {"source_checks": state, "files": inputs,
        "policy": "same origins, counts, horizons and seeds; no resampling or retraining of completed other-model results"})
    _json(output / "MODEL_SELECTION.json", {"dysanos_rank": 16, "models": list(FINAL_MODELS),
        "selection": "user-requested sixteen-factor primary comparison; eight-factor results remain in rank_comparison.json"})
    _json(destination, protocol)
    _json(output / "BASE_READY.json", {"status": "all reused native one-day baselines complete"})


def retain_deck(output: Path, repository_directory: Path) -> None:
    """Intentionally retain aggregates and report artifacts, never raw quotes or model arrays."""
    coverage = pd.read_csv(output / "coverage.csv")
    if len(coverage) != len(FINAL_MODELS) or not coverage[["surface_origins", "quote_forecast_origins", "recursive_origins"]].eq(44).all().all():
        raise ValueError("The final deck still has missing origin-level diagnostics")
    if not coverage[["recursive_min_sessions", "recursive_max_sessions"]].eq(115).all().all():
        raise ValueError("The final deck does not use identical path horizons")
    for model in FINAL_MODELS:
        for path in (output / "persistence" / model.replace(" ", "_")).glob("*.json"):
            record = json.loads(path.read_text())
            if (record["path_count"], record["steps"], record["seed"]) != (32, 115, 20_261_111):
                raise ValueError(f"The final deck has a mismatched path count, horizon or seed: {path}")
    weighted_path = output / "density_weighted_diagnostics.json"
    if weighted_path.exists():
        weighted = json.loads(weighted_path.read_text())
        for model in weighted["models"]:
            if any(model["coverage"][key] != 44 for key in ("surface_origins", "quote_forecast_origins", "recursive_origins")):
                raise ValueError("The weighted follow-up has missing origin-level diagnostics")
            if (weighted["paths"], weighted["sessions"], weighted["seed"]) != (32, 115, 20_261_111):
                raise ValueError("The weighted follow-up has mismatched paths or horizons")
    repository_directory.mkdir(parents=True, exist_ok=True)
    names = {path.name for pattern in ("*.pdf", "*.tex", "*.csv") for path in output.glob(pattern)}
    names.update({"STATS.md", "summary.json", "summary_matched_dates.json", "observable_reference.json",
        "TRAFFIC_LIGHT_RULES.json", "MODEL_SELECTION.json", "PROTOCOL.json", "SOURCE_REUSE.json",
        "SOURCE_INPUTS.json", "INFERENCE_MODELS.json", "rank_comparison.json", "MANIFEST.json"})
    names.update(name for name in ("covariance_extension.json", "covariance.png", "density_weighted_diagnostics.json",
                                  "DENSITY_WEIGHTED_INPUTS.json") if (output / name).is_file())
    retained = {}
    for name in sorted(names):
        original = output / name
        if not original.is_file():
            raise FileNotFoundError(original)
        target = repository_directory / name
        shutil.copy2(original, target)
        retained[name] = {"sha256": _hash(target), "bytes": target.stat().st_size}
    _json(repository_directory / "RETAINED_ARTIFACTS.json", {"external_run": str(output),
        "files": retained, "large_files_external": True, "raw_licensed_quotes_included": False})
    print(f"Retained complete deck and {len(retained)} report artifacts in {repository_directory}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rank-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retain", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / ".finalize.lock"):
        repair_batch_error_records(arguments.source, arguments.rank_run)
        started = time.monotonic()
        while True:
            state = coverage_state(arguments.source, OTHER_MODELS)
            print(json.dumps({"stage": "waiting-for-other-models", "elapsed_seconds": round(time.monotonic()-started, 1),
                "completed": state}), flush=True)
            if all(value == 44 for counts in state.values() for value in counts.values()):
                break
            time.sleep(60.)
        prepare(arguments.source, arguments.rank_run, arguments.output)
        stages = (("run", TAIL_MODELS), ("quotes", (*PRIMARY_MODELS, *TAIL_MODELS)), ("persistence", TAIL_MODELS))
        for module, models in stages:
            subprocess.run([sys.executable, "-u", "-m", f"quantdata.hedging.model_audit.{module}",
                "--output", str(arguments.output), "--models", *models], check=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
        refresh_report(arguments.output)
        retain_deck(arguments.output, arguments.retain)
        _json(arguments.output / "READY_TO_PUBLISH.json", {"complete": True, "retained_directory": str(arguments.retain),
            "models": list(FINAL_MODELS), "origins": 44, "paths": 32, "sessions": 115})


if __name__ == "__main__":
    main()
