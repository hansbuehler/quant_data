"""Frozen observable-PC persistence using a retained subset of the same v2 paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.legacy_v2_diagnostics import load_origin, native_snapshots
from quantdata.hedging.model_audit.persistence import path_acf
from quantdata.hedging.model_audit.rolling_versions import origins
from quantdata.hedging.model_audit.run import COMMON_INDICES, _dlv_surface, _hash, _json


def run(root: Path, reference: Path, output: Path) -> None:
    selected = origins(reference)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": "DYSANOS v2-P", "origins": list(selected), "simulation_paths": 1000,
        "retained_paths": 32, "steps": 115, "seed": 20_261_112,
        "same_population": "first 32 paths from the full 1000-path static audit stream, not new 32-path draws",
        "observables": "floating log-IVs and fixed pre-April observable PCA, not daily re-estimated PCs",
        "basis_sha256": _hash(reference / "frozen_observable_pca.npz"), "source_sha256": _hash(Path(__file__))}
    _json(output / "PROTOCOL.json", protocol)
    with np.load(reference / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    with np.load(reference / "reference_history.npz", allow_pickle=False) as saved:
        historical, dates = saved["surfaces"], saved["dates"].astype(str)
    started = time.monotonic()
    for index, origin in enumerate(selected):
        health_check()
        destination = output / f"daily/{origin}.json"
        if destination.exists():
            continue
        state, law, tail, training = load_origin(root, origin)
        snapshots, _ = native_snapshots(state, law, tail, paths=1000, retain_paths=32,
            horizons=tuple(range(1, 116)), seed=20_261_112)
        native = np.stack(list(snapshots.values()), axis=1).reshape(32 * 115, -1)
        values, guards = [], []
        for begin in range(0, len(native), 128):
            surface, diagnostics = _dlv_surface(state, native[begin:begin + 128])
            values.append(surface.reshape(len(surface), -1))
            guards.append(diagnostics)
            if begin % 512 == 0:
                health_check()
                print(f"v2 persistence {origin} decoded={min(begin+128,len(native))}/{len(native)}", flush=True)
        levels = np.concatenate(values).reshape(32, 115, -1)
        valid = np.isfinite(levels).all(axis=(1, 2))
        history = historical[(dates >= training["dates"][0]) & (dates <= origin)][:, :, COMMON_INDICES].reshape(-1, 49)
        record = {"model": "DYSANOS v2-P", "origin": origin, "paths": 32, "steps": 115,
            "invalid_surface_paths": int((~valid).sum()), "paths_removed": 0,
            "lower_clip_coordinates": sum(item["lower_clips"] for item in guards),
            "upper_clip_coordinates": sum(item["upper_clips"] for item in guards),
            "historical_dates": len(history), "historical_observable_acf": path_acf((history @ basis)[None]),
            "scope": "observable persistence is descriptive; analytic stationarity is tested on the floating PCA/VAR backbone only"}
        if valid.all():
            record["observable_acf"] = path_acf(levels @ basis)
            record["status"] = "complete"
        else:
            record.update(status="diagnostic_failure", exact_error="Nonfinite decoded observable paths; ACF not computed on a filtered sample")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.npz")
        np.savez_compressed(temporary, log_iv_paths=levels)
        temporary.replace(destination.with_suffix(".npz"))
        _json(destination, record)
        records = [json.loads(path.read_text()) for path in sorted(destination.parent.glob("*.json"))]
        _json(output / "SUMMARY.json", {"processed": len(records), "complete": sum(item["status"] == "complete" for item in records),
            "failures": [item["origin"] for item in records if item["status"] != "complete"]})
        elapsed = time.monotonic() - started
        print(f"v2 persistence completed={index+1}/{len(selected)} elapsed={elapsed:.1f}s eta={elapsed*(len(selected)-index-1)/(index+1):.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.reference, args.output)


if __name__ == "__main__":
    main()
