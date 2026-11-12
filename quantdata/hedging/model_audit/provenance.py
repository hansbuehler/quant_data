"""Checksummed external input registry; never copies licensed quotes or weights."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.model_audit.run import (
    _hash,
    _json,
    _record,
    retained_dynamics_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root, output = arguments.root, arguments.output
    dates = json.loads((output / "observable_reference.json").read_text())["dates"]
    registry: dict[str, dict] = {}

    def register(path: Path, role: str, model: str = "", day: str = "") -> None:
        if not path.is_file():
            return
        key = str(path)
        if key not in registry:
            registry[key] = {"path": key, "sha256": _hash(path), "bytes": path.stat().st_size,
                             "roles": [], "models": [], "origin_dates": []}
        for field, value in (("roles", role), ("models", model), ("origin_dates", day)):
            if value and value not in registry[key][field]:
                registry[key][field].append(value)

    public = root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1"
    selection_path = output / "MODEL_SELECTION.json"
    selected = json.loads(selection_path.read_text())["models"] if selection_path.exists() else []
    for name in ("MODEL_SELECTION.json", "PROTOCOL.json", "SOURCE_REUSE.json", "rank_comparison.json", "covariance_extension.json",
                 "density_weighted_diagnostics.json", "DENSITY_WEIGHTED_INPUTS.json"):
        register(output / name, "audit protocol and selected-model provenance")
    inference_catalog = output / "INFERENCE_MODELS.json"
    if inference_catalog.exists():
        register(inference_catalog, "verified inference replacement and historical source checksums")
        for entry in json.loads(inference_catalog.read_text())["models"].values():
            manifest = Path(entry["manifest"])
            register(manifest, "dated inference-only model", entry["variant"], entry["origin_date"])
            state = json.loads(manifest.read_text())
            for name in ("backbone", "adapter"):
                if state[name] is not None:
                    register(Path(state[name]["path"]), "shared exact inference tensors", entry["variant"], entry["origin_date"])
    for path in public.glob("*"):
        register(path, "reference history and common observable geometry")
    source = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    for name in ("config.json", "observations.parquet"):
        register(source / name, "DYSANOS source configuration/history")
    for day in dates:
        continued = output / f"completion/ft-may/dates/{day}.json"
        if continued.exists():
            record = json.loads(continued.read_text())
            register(continued, "new causal FT continuation lineage", "AD-Seq-Vol-FT", day)
            register(Path(record["checkpoint"]), "rolling neural checkpoint", "AD-Seq-Vol-FT", day)
            register(Path(record["base_checkpoint"]), "frozen FT base", "AD-Seq-Vol-FT", day)
        daily = source / f"days/{day}.npz"
        register(daily, "fitted DYSANOS daily state", "DYSANOS shared", day)
        with np.load(daily, allow_pickle=False) as saved:
            register(Path(str(saved["source_model"])), "original licensed SANOS source fit", "DYSANOS shared", day)
        for variant in ("ordinary", "orthogonal"):
            register(root / f"quant_data/hedge2/runs/rolling-2022-2025-{variant}-v2-20261106/days/{day}.npz",
                     "rolling PCA/AR(1) parameters", f"DYSANOS {variant}", day)
        for model in selected:
            if "(16 factors)" in model and not model.endswith("(tail)"):
                dynamics = retained_dynamics_path(root, model, date.fromisoformat(day), output=output)
                register(dynamics, "selected sixteen-factor PCA/AR(1) parameters", model, day)
                register(dynamics.parent.parent / "config.json", "selected rank and causal fitting protocol", model)
        for name in ("manifest.json", "affine-200-history.h5", "rank16-ar1-dynamics.npz", "rank16-ar1-dynamics.json"):
            register(root / f"training/hedging/gromm-current-public-20261024-v1/artifacts/{day}/{name}",
                     "shared GROMM P/Q fit", "GROMM P/Q", day)
        register(root / f"training/hedging/funvol-causal-chain-20261025-warning-v1/aprmay/checkpoints/{day}.pt",
                 "rolling neural checkpoint", "FuNVol", day)
        for name in ("volgan", "diffusion-base", "diffusion-ft", "gromm-P", "gromm-Q"):
            register(root / f"experiments/hedging/generated-objective-grid-20261026-v1/scenario-books/{name}/{day}.npz",
                     "raw retained price-return scenarios", name, day)
        for path in (output / "daily").glob(f"*/{day}.json"):
            row = json.loads(path.read_text())
            if row.get("status") != "scored":
                continue
            diagnostics = row["diagnostics"]
            if "checkpoint" in diagnostics:
                register(Path(diagnostics["checkpoint"]), "rolling neural checkpoint", row["model"], day)
            if "scenario_path" in diagnostics:
                register(Path(diagnostics["scenario_path"]), "one-day spot/IV scenarios", row["model"], day)
            if row["model"] in ("AD-Seq-Vol", "AD-Seq-Vol-FT"):
                variant = "base" if row["model"] == "AD-Seq-Vol" else "ft"
                parent = (root / f"experiments/hedging/short-end-regeneration-20261024/diffusion-{variant}-rolling-fixed-v2/{day}"
                          if day == "2024-04-01" else root / f"experiments/hedging/diffusion-{variant}-gpu-20261027-v1/{day}")
                record = parent / f"records/{day}.json"
                if record.exists():
                    register(record, "dated checkpoint authentication", row["model"], day)
                    diagnostics = _record(record)["diagnostics"]
                    register(Path(diagnostics["fit_checkpoint_path"]), "rolling neural checkpoint", row["model"], day)
                else:
                    register(output / f"completion/ft-may/dates/{day}.json", "new causal FT continuation lineage", row["model"], day)
        register(root / f"experiments/hedge2/market-full-20261106T0001/market/days/{day}.parquet",
                 "licensed observed quotes; external only", "all", day)
    endpoints = {json.loads(path.read_text()).get("endpoint") for path in (output / "quote_forecasts").glob("*/*.json")}
    for endpoint in endpoints - {None}:
        register(root / f"experiments/hedge2/market-full-20261106T0001/market/days/{endpoint}.parquet",
                 "licensed next-session quote observations; external only")
    for path in (output / "persistence").glob("*/*.json"):
        record = json.loads(path.read_text())
        if "surface_artifact" in record:
            register(Path(record["surface_artifact"]), "recursive observable IV paths", record["model"], record["origin_date"])
    for directory, role in (("tail_fits", "causal rolling Tail Diffusion fit"),
                            ("tail_solver_reviews", "independent optimizer accuracy review"),
                            ("variance_reconciliation", "same-shock native/observable variance ablation")):
        for path in (output / directory).glob("*/*.json"):
            record = json.loads(path.read_text())
            register(path, role, record["model"], record["origin_date"])
    _json(output / "SOURCE_INPUTS.json", {"schema": "hedging-model-audit-external-inputs-v1",
        "files": list(registry.values()), "file_count": len(registry),
        "total_unique_bytes": sum(item["bytes"] for item in registry.values()),
        "warning": "External paths/hashes are an inventory, not redistribution rights; licensed inputs and large model files stay external."})
    print(f"Authenticated {len(registry)} unique external input files", flush=True)


if __name__ == "__main__":
    main()
