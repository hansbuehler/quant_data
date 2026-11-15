"""Legacy eight-PC/STRT controls, not the paper-v1 or new SANOS-v2 models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import fields
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import DynamicsConfig
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.density_weighted import _load
from quantdata.hedging.model_audit.grid_shape import _score
from quantdata.hedging.model_audit.metrics import factor_innovation_covariance, linear_stability
from quantdata.hedging.model_audit.projection_metrics import (
    _save,
    fit_projected,
    projected_scenarios,
)
from quantdata.hedging.model_audit.run import _hash, _json, _summarize
from quantdata.hedging.model_audit.tail import _training_pairs

MODELS = ("DYSANOS v1", "DYSANOS v2", "FuNVol", "VolGAN PCA8",
          "AD-Seq-Vol", "AD-Seq-Vol-FT", "GROMM-P", "GROMM-Q")
SPECIFICATIONS = {
    "DYSANOS v1": {"variant": "ordinary", "projection": "affine200_unweighted", "rank": 8,
        "description": "Unweighted affine-200 level projection, then eight Euclidean change PCs and VAR(1)."},
    "DYSANOS v2": {"variant": "orthogonal", "projection": "straddle_sqrt_time", "rank": 16,
        "description": "Direct square-root-maturity-scaled straddle sensitivity metric, sixteen change PCs and VAR(1); full native spot loading and paired spot/factor innovations retained."},
}


def run(root: Path, source: Path, projections: Path, weighted: Path, output: Path) -> None:
    """Refit the specified VAR ranks rather than truncating higher-dimensional AR matrices."""
    raise RuntimeError("This legacy runner mislabels PCA surrogates as paper DYSANOS v1/v2; use paper_v1 for the authenticated affine-200 law. Existing surrogate artifacts remain controls only.")
    book = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    raw = json.loads((book / "config.json").read_text())
    config = DlvConfig(**{field.name: raw[field.name] for field in fields(DlvConfig) if field.name in raw})
    origins = json.loads((source / "observable_reference.json").read_text())["dates"]
    if len(origins) != 44:
        raise ValueError("The versioned comparison requires the original 44 origins")
    protocol = {"schema": "dysanos-two-versions-v1", "models": list(MODELS),
        "specifications": SPECIFICATIONS, "origins": origins, "source": str(source),
        "projections": str(projections), "weighted": str(weighted), "native_book": str(book),
        "source_smoothness": .1, "reference_smoothness": .25,
        "data_policy": "unchanged source states, original native training spot series, and original observable outcome/spot series; no alternative data or normalization",
        "training_transitions": 252, "source_code_sha256": _hash(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PROTOCOL.json").exists() and json.loads((output / "PROTOCOL.json").read_text()) != protocol:
        raise ValueError("Output already belongs to another versioned audit")
    _json(output / "PROTOCOL.json", protocol)
    for name in ("reference_history.npz", "frozen_observable_pca.npz", "observable_reference.json",
                 "INFERENCE_MODELS.json"):
        shutil.copy2(source / name, output / name)
    for model in MODELS[2:]:
        for category in ("daily", "quote_forecasts", "persistence"):
            target = output / category / model.replace(" ", "_")
            if not target.exists():
                shutil.copytree(source / category / model.replace(" ", "_"), target)
    with np.load(source / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        observable_basis = saved["basis"]
    inputs = {Path(__file__), book / "config.json", book / "observations.parquet",
              source / "observable_reference.json", source / "reference_history.npz"}
    pairs = _training_pairs(root)

    @lru_cache(maxsize=600)
    def native(day: date) -> np.ndarray:
        path = book / f"days/{day}.npz"
        inputs.add(path)
        with np.load(path, allow_pickle=False) as saved:
            return saved["raw_u"]

    previous, records = {}, {model: [] for model in MODELS[:2]}
    started = time.monotonic()
    for index, text in enumerate(["2024-03-28", *origins]):
        day = date.fromisoformat(text)
        window = [pair for pair in pairs if pair[1] <= day][-252:]
        if len(window) != 252:
            raise ValueError(f"Not enough original training transitions for {day}")
        selected_days = tuple(sorted({point for pair in window for point in pair[:2]}))
        session_lookup = {point: pair[2] - offset for pair in window
                          for point, offset in ((pair[0], 1), (pair[1], 0))}
        sessions = np.asarray([session_lookup[point] for point in selected_days])
        date_index = {point: position for position, point in enumerate(selected_days)}
        levels = np.stack([native(point) for point in selected_days])
        for model, specification in SPECIFICATIONS.items():
            variant, method = specification["variant"], specification["projection"]
            parent = projections / f"fits/{variant}/{method}/{text}"
            with np.load(parent / "projection.npz", allow_pickle=False) as saved:
                encoder, decoder = saved["encoder"], saved["decoder"]
                beta = saved["native_spot_loading"]
            if model == "DYSANOS v2":
                encoder, decoder = encoder[:, :specification["rank"]], decoder[:, :specification["rank"]]
            original = _load(weighted / f"fits/{variant}/{text}").weighted_fit
            spots = np.full(len(selected_days), np.nan)
            spots[[date_index[pair[1]] for pair in window]] = original.residuals[:, 0]
            checkpoint = output / f"fits/{model.replace(' ', '_')}/{text}"
            if (checkpoint / "dynamics.npz").exists():
                fit = load_dynamics(checkpoint / "dynamics.npz")
            else:
                fit = fit_projected(levels=levels, dates=selected_days, spots=spots, sessions=sessions,
                    encoder=encoder, decoder=decoder,
                    config=DynamicsConfig(rank=specification["rank"], spot_orthogonal=variant == "orthogonal"),
                    previous=previous.get(model))
                _save(checkpoint, fit, encoder, decoder, beta)
            previous[model] = (fit, decoder)
            inputs.update((parent / "projection.npz", parent / "dynamics.npz",
                           weighted / f"fits/{variant}/{text}/dynamics.npz"))
            if text not in origins:
                continue
            path = output / f"daily/{model.replace(' ', '_')}/{text}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                baseline_path = source / f"daily/DYSANOS_ordinary_(16_factors)/{text}.json"
                baseline = json.loads(baseline_path.read_text())
                inputs.add(baseline_path)
                with np.load(book / f"days/{text}.npz", allow_pickle=False) as saved:
                    state = DailyState(day=day, native=saved["raw_u"], reference_support=saved["support_strikes"], config=config)
                following = projected_scenarios(state.native, encoder, decoder, fit, native_spot_loading=beta)
                effective_loading = encoder.T @ beta
                innovation = factor_innovation_covariance(fit.joint_innovation_covariance,
                    fit.directions.T @ effective_loading)
                stability = linear_stability(fit.ar_matrix, innovation)
                diagnostics = {"specification": specification, "rank": specification["rank"],
                    "training_start": str(fit.training_start), "training_end": str(fit.training_end),
                    "initialisation_date": str(fit.initialisation_date), "fit": str(checkpoint),
                    "raw_spectral_radius": fit.unconstrained_spectral_radius,
                    "stability": {**stability, "scope": "floating-volatility PCA VAR backbone"},
                    "native_pca_coverage": float(fit.explained_variance_fraction.sum()),
                    "spot_policy": "original paired empirical shocks; full native loading, including outside the retained span",
                    "full_native_spot_loading_norm": float(np.linalg.norm(beta)),
                    "source_data_changed": False}
                record = _score(state, following, np.log1p(fit.residuals[:, 0]), baseline,
                    label=model, basis=observable_basis, diagnostics=diagnostics)
                _json(path, record)
            records[model].append(record)
        elapsed = time.monotonic() - started
        print(f"versioned fits {text} completed={index+1}/{len(origins)+1} elapsed={elapsed:.1f}s "
              f"eta={elapsed*(len(origins)-index)/(index+1):.1f}s", flush=True)
    for model in MODELS[2:]:
        records[model] = [json.loads((output / f"daily/{model.replace(' ', '_')}/{text}.json").read_text())
                          for text in origins]
    _json(output / "summary.json", {model: _summarize(rows, observable_basis) for model, rows in records.items()})
    _json(output / "MODEL_SELECTION.json", {"models": list(MODELS), "specifications": SPECIFICATIONS,
        "selection": "Only DYSANOS v1 and v2 are reported; older DYSANOS laws are not relabelled or substituted."})
    _json(output / "VERSIONED_INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in sorted(inputs)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    for name in ("source", "projections", "weighted", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(args.output / ".versioned.lock"):
        run(args.root, args.source, args.projections, args.weighted, args.output)


if __name__ == "__main__":
    main()
