"""Common 32-path, 115-session audit, direct cdxcore inversion and causal tail cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from quantdata.hedging.model_audit.queue_ft_forecasts import wait_for_gpu
from quantdata.hedging.model_audit.recursive import _save_arrays, summarize_paths
from quantdata.hedging.model_audit.refresh import refresh_report
from quantdata.hedging.model_audit.run import _hash, _json

TAIL_MODELS = ("DYSANOS ordinary (tail)", "DYSANOS spot-orthogonal (tail)")
BASE_MODELS = ("DYSANOS ordinary", "DYSANOS spot-orthogonal", "FuNVol", "VolGAN PCA8")


def reuse_gromm_forecasts(source: Path, output: Path) -> None:
    """The old extra mask was a no-op for these already-interior one-day samples."""
    reused, checked = [], 0
    for model in ("GROMM-P", "GROMM-Q"):
        for path in sorted((source / "daily" / model).glob("*.json")):
            record = json.loads(path.read_text())
            if (record.get("status") != "scored" or record["valid_scenario_count"] != record["scenario_count"]
                    or record["invalid_surface_rate"] != 0. or record["extreme_surface_rate"] != 0.):
                continue
            destination = output / "daily" / model / path.name
            if destination.exists():
                candidate = json.loads(destination.read_text())
                for name in ("surface_mean", "surface_covariance", "joint_covariance", "quantiles"):
                    if not np.array_equal(record[name], candidate[name]):
                        raise ValueError(f"Recomputed one-day GROMM forecast differs: {model} {path.stem} {name}")
                checked += 1
            else:
                record["diagnostics"]["inversion_reuse"] = {
                    "source_record": str(path), "source_sha256": _hash(path),
                    "reason": "all source scenarios were interior and accepted; removing the additional mask does not change the existing vector cdxcore estimates",
                    "model_paths_resampled": False}
                _json(destination, record)
                reused.append(str(destination))
    _json(output / "ONEDAY_GROMM_REUSE.json", {"reused_records": reused, "independently_recomputed_matching_records": checked,
        "scope": "unchanged one-day forecasts only; all long-path GROMM tests are recomputed under direct inversion"})
    print(f"Reused {len(reused)} unchanged one-day GROMM forecasts; {checked} independently recomputed records agree exactly", flush=True)


def prepare(source: Path, output: Path) -> None:
    """Preserve prior results and reuse identical neural path prefixes and inference models."""
    protocol = {"schema": "hedging-common-path-audit-v2", "source_run": str(source),
        "paths": 32, "sessions": 115, "seed": 20_261_111,
        "inversion": "vector cdxcore.bs.implied; no custom intrinsic-distance mask or caller-side price clipping",
        "tail_cases": list(TAIL_MODELS), "spot_policy": "normal spot for added tail cases",
        "path_policy": "same origins/count/horizon/seed; reuse existing identical prefixes; native model laws remain distinct"}
    destination = output / "PROTOCOL.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("Common-path output directory belongs to another protocol")
        return
    output.mkdir(parents=True, exist_ok=False)
    for name in ("reference_history.npz", "frozen_observable_pca.npz", "observable_reference.json", "INFERENCE_MODELS.json"):
        shutil.copy2(source / name, output / name)
    for directory in ("inference", "regenerated"):
        (output / directory).symlink_to(source / directory, target_is_directory=True)
    shutil.copytree(source / "quote_forecasts", output / "quote_forecasts")
    shutil.copytree(source / "completion/ft-may/dates", output / "completion/ft-may/dates")
    for directory in (source / "daily").iterdir():
        if directory.is_dir() and not directory.name.startswith(("DYSANOS", "GROMM")):
            shutil.copytree(directory, output / "daily" / directory.name)
    reuse_gromm_forecasts(source, output)
    with np.load(output / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    for model in ("AD-Seq-Vol", "AD-Seq-Vol-FT"):
        for path in sorted((source / "persistence" / model).glob("*.json")):
            record = json.loads(path.read_text())
            arrays = path.with_suffix(".npz")
            if not arrays.exists() or record.get("steps", 0) < 115:
                continue
            with np.load(arrays, allow_pickle=False) as saved:
                surfaces = saved["log_iv_surfaces"][:, :115].copy()
            migrated = summarize_paths(surfaces, basis)
            migrated["diagnostics"].update({key: value for key, value in record["diagnostics"].items()
                                           if key not in ("invalid_path_rate", "extreme_surface_rate")})
            target = output / "persistence" / model / path.name
            target_arrays = target.with_suffix(".npz")
            _save_arrays(target_arrays, log_iv_surfaces=surfaces)
            migrated.update(model=model, origin_date=record["origin_date"], path_count=32, steps=115,
                requested_steps=115, seed=20_261_111, fit_frozen_through_path=True,
                source_path_record=str(path), source_path_sha256=_hash(arrays),
                migration="first 115 sessions of the exact original simulated paths; no resampling",
                surface_artifact=str(target_arrays), surface_artifact_sha256=_hash(target_arrays))
            _json(target, migrated)
        for path in sorted((source / "completion/recursive" / model).glob("*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                values = {name: saved[name] for name in saved.files}
            identity = json.loads(str(values["identity_json"]))
            if int(values["completed"]) > 115:
                continue
            identity["steps"] = 115
            values["identity_json"] = json.dumps(identity)
            values["surfaces"] = values["surfaces"][:, :115].copy()
            _save_arrays(output / "completion/recursive" / model / path.name, **values)
    _json(output / "summary.json", {})
    _json(destination, protocol)
    _json(source / "SUPERSEDED_PROTOCOL.json", {"replacement_run": str(output),
        "reason": "user requested common horizons, direct cdxcore inversion and added Tail Diffusion cases",
        "original_results_preserved": True})
    print(f"Prepared common-path protocol: {output}", flush=True)


def run_module(module: str, output: Path, *extra: str, cpu: bool = True) -> None:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"} if cpu else os.environ
    subprocess.run([sys.executable, "-u", "-m", f"quantdata.hedging.model_audit.{module}",
        "--output", str(output), *extra], check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "baseline", "cpu", "gromm", "gpu", "tail"), required=True)
    parser.add_argument("--gpuctl", type=Path, default=Path.home() / ".local/bin/gpuctl")
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        if arguments.source is None:
            raise ValueError("Preparation requires the retained source run")
        prepare(arguments.source, arguments.output)
        return
    protocol = json.loads((arguments.output / "PROTOCOL.json").read_text())
    if protocol["paths"] != 32 or protocol["sessions"] != 115:
        raise ValueError("The requested common 32-path/115-session protocol was changed")
    if arguments.stage == "baseline":
        run_module("run", arguments.output, "--models", *BASE_MODELS[:2])
        run_module("gromm", arguments.output)
        _json(arguments.output / "BASE_READY.json", {"status": "native one-day baselines repriced under direct cdxcore inversion"})
    elif arguments.stage == "cpu":
        run_module("persistence", arguments.output, "--models", *BASE_MODELS, "--steps", "115", "--refresh-report")
    elif arguments.stage == "gromm":
        run_module("recursive", arguments.output, "--models", "GROMM-Q", "GROMM-P", "--steps", "115", "--refresh-report")
    elif arguments.stage == "tail":
        run_module("run", arguments.output, "--models", *TAIL_MODELS)
        run_module("quotes", arguments.output, "--models", *TAIL_MODELS)
        run_module("persistence", arguments.output, "--models", *TAIL_MODELS, "--steps", "115", "--refresh-report")
    else:
        wait_for_gpu(arguments.gpuctl, stage="Common 115-session diffusion paths")
        subprocess.run([str(arguments.gpuctl), "run", "--gpus", "1", "--task-name", "Hedging common-path audit",
            "--label", "32 paths x 115 sessions, base and FT diffusion", "--", sys.executable, "-u", "-m",
            "quantdata.hedging.model_audit.recursive", "--output", str(arguments.output),
            "--models", "AD-Seq-Vol", "AD-Seq-Vol-FT", "--steps", "115", "--device", "cuda", "--refresh-report"], check=True)
    refresh_report(arguments.output)


if __name__ == "__main__":
    main()
