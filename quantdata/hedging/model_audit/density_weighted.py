"""Causal grid-probability-weighted PCA/VAR comparison with retained DYSANOS fits."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, fields, replace
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import DynamicsConfig, FittedDynamics, fit_dynamics
from quantdata.hedge2.rolling import load_dynamics, save_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.metrics import covariance
from quantdata.hedging.model_audit.run import (
    QUANTILES,
    _backbone_stability,
    _dlv_surface,
    _hash,
    _json,
    _summarize,
)
from quantdata.hedging.model_audit.tail import _training_pairs


@dataclass(frozen=True, kw_only=True)
class DensityWeightedFit:
    weighted_fit: FittedDynamics
    weights: np.ndarray

    def __post_init__(self) -> None:
        if (self.weights.shape != self.weighted_fit.state_mean.shape
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0.)
                or not np.any(self.weights > 0.)):
            raise ValueError("Density metric must be finite, nonnegative and match the native state")

    @property
    def scale(self) -> np.ndarray:
        return np.sqrt(self.weights)

    def unscale_changes(self, changes: np.ndarray) -> np.ndarray:
        """Map weighted changes back to native units, freezing zero-weight coordinates."""
        return np.divide(changes, self.scale, out=np.zeros_like(changes), where=self.scale > 0.)

    def next_native(self, current: np.ndarray) -> np.ndarray:
        """Enumerate the same jointly paired empirical spot/factor innovation law."""
        fit = self.weighted_fit
        forecast = fit.forecast_state_change(current * self.scale)
        changes = forecast + fit.residuals[:, :1] * fit.spot_state_loading + fit.residuals[:, 1:] @ fit.directions.T
        return current + self.unscale_changes(changes)


def density_metric(origin_probabilities: np.ndarray) -> np.ndarray:
    """Average origin grid masses; weight squared errors, not their amplitudes."""
    values = np.asarray(origin_probabilities, dtype=float)
    if (values.ndim != 3 or values.shape[2] < 3 or not np.isfinite(values).all()
            or np.any(values < 0.) or not np.allclose(values.sum(axis=2), 1., atol=1e-8, rtol=0.)):
        raise ValueError("Grid probabilities must be nonnegative path-by-expiry-by-node unit masses")
    average = values[:, :, 1:-1].mean(axis=0)
    return np.c_[average, np.ones(average.shape[0])].reshape(-1)


def fit_weighted(*, weights: np.ndarray, dates: tuple[date, ...], states: np.ndarray,
                 spot_returns_into_date: np.ndarray, session_indices: np.ndarray,
                 as_of_date: date, config: DynamicsConfig,
                 previous: DensityWeightedFit | None = None) -> DensityWeightedFit:
    """Fit PCA and VAR in one fixed density metric per rolling training window."""
    states = np.asarray(states, dtype=float)
    metric = np.asarray(weights, dtype=float)
    if (metric.shape != (states.shape[1],) or not np.isfinite(metric).all()
            or np.any(metric < 0.) or not np.any(metric > 0.)):
        raise ValueError("Invalid density metric for the training states")
    scale = np.sqrt(metric)
    prior = None
    if previous is not None:
        native_columns = previous.unscale_changes(previous.weighted_fit.directions.T).T
        mapped = scale[:, None] * native_columns
        norms = np.linalg.norm(mapped, axis=0)
        if not np.isfinite(mapped).all() or np.any(norms <= 0.):
            raise ValueError("Previous factor geometry cannot be represented in the new density metric")
        prior = replace(previous.weighted_fit, directions=mapped / norms)
    fit = fit_dynamics(states=states * scale, previous=prior, dates=dates,
        spot_returns_into_date=spot_returns_into_date, session_indices=session_indices,
        as_of_date=as_of_date, config=config)
    return DensityWeightedFit(weighted_fit=fit, weights=metric)


def _save(model: DensityWeightedFit, directory: Path) -> None:
    save_dynamics(model.weighted_fit, directory / "dynamics.npz")
    temporary = directory / "metric.partial.npz"
    np.savez_compressed(temporary, weights=model.weights)
    temporary.replace(directory / "metric.npz")


def _load(directory: Path) -> DensityWeightedFit:
    with np.load(directory / "metric.npz", allow_pickle=False) as saved:
        return DensityWeightedFit(weighted_fit=load_dynamics(directory / "dynamics.npz"), weights=saved["weights"])


def load_weighted_model(output: Path, model: str, day: date) -> DensityWeightedFit:
    """Load the dated metric as well as its VAR; never substitute an unweighted fit."""
    if not model.endswith("density-weighted"):
        raise ValueError("A weighted checkpoint requires an explicitly density-weighted model label")
    extension = output / "covariance_extension.json"
    directory = Path(json.loads(extension.read_text())["experiment"]) if extension.exists() else output
    variant = "orthogonal" if "spot-orthogonal" in model else "ordinary"
    fitted = _load(directory / "fits" / variant / str(day))
    if fitted.weighted_fit.as_of_date != day or fitted.weighted_fit.directions.shape[1] != 16:
        raise ValueError("Weighted checkpoint date or factor count differs from its label")
    return fitted


def recursive_native_paths(model: DensityWeightedFit, current: np.ndarray, *, paths: int,
                           steps: int, seed: int) -> tuple[np.ndarray, dict]:
    """Simulate in fitted weighted coordinates, then invert that same fixed metric."""
    fit = model.weighted_fit
    weighted_current = current * model.scale
    current_scores = (weighted_current - fit.state_mean) @ fit.directions
    scores = np.broadcast_to(current_scores, (paths, len(current_scores))).copy()
    projected_spot = fit.spot_state_loading @ fit.directions
    perpendicular = fit.spot_state_loading - fit.directions @ projected_spot
    anchor = weighted_current - fit.directions @ current_scores
    accumulated_spot = np.zeros(paths)
    random = np.random.default_rng(seed)
    values = []
    for _ in range(steps):
        innovations = fit.residuals[random.integers(0, len(fit.residuals), paths)]
        scores = scores @ fit.ar_matrix.T + fit.ar_intercept + innovations[:, 1:] + innovations[:, :1] * projected_spot
        accumulated_spot += innovations[:, 0]
        following = anchor + scores @ fit.directions.T + accumulated_spot[:, None] * perpendicular
        values.append(current + model.unscale_changes(following - weighted_current))
    ratio = float(np.linalg.norm(perpendicular) / np.linalg.norm(fit.spot_state_loading)) if np.any(fit.spot_state_loading) else 0.
    return np.stack(values, axis=1), {
        "spot_loading_outside_factor_span_fraction": ratio,
        "spot_loading_fraction_metric": "fixed training-origin grid-probability norm",
        "full_native_unit_root_from_spot_loading": bool(ratio > 1e-8),
        "recursive_metric_policy": "same frozen metric as the dated fitted VAR; no reweighting along simulated paths"}


def _compression(model: DensityWeightedFit, before: np.ndarray, after: np.ndarray,
                 *, config: DlvConfig) -> dict:
    fit = model.weighted_fit
    centered = after - before
    centered -= centered.mean(axis=0)
    weighted = centered * model.scale
    spot = fit.residuals[:, :1] * fit.spot_state_loading
    compressed = spot + ((weighted - spot) @ fit.directions) @ fit.directions.T
    native_projection = model.unscale_changes(compressed)
    shape = (len(before), len(config.business_dtes), config.num_inner_strikes + 1)
    raw, reduced = centered.reshape(shape), native_projection.reshape(shape)
    crossings = ((before.reshape(shape)[:, :, :-1] <= 2. * np.log(config.min_local_vol) + 1e-8)
                 != (after.reshape(shape)[:, :, :-1] <= 2. * np.log(config.min_local_vol) + 1e-8))
    weighted_floor_energy = np.sum(weighted.reshape(shape)[:, :, :-1]**2 * crossings)
    return {"retained_native_forward_vol_energy_fraction": float(np.sum(reduced[:, :, -1]**2) / np.sum(raw[:, :, -1]**2)),
        "retained_weighted_energy_fraction": float(np.sum(compressed**2) / np.sum(weighted**2)),
        "floor_crossing_energy_fraction_of_weighted_total": float(weighted_floor_energy / np.sum(weighted**2))}


def _score(model: DensityWeightedFit, state: DailyState, baseline: dict, *, path: Path) -> dict:
    fit = model.weighted_fit
    current, _ = _dlv_surface(state, state.native[None])
    surfaces, guards = _dlv_surface(state, model.next_native(state.native))
    next_common = surfaces.reshape(len(surfaces), -1)
    returns = np.log1p(fit.residuals[:, 0])
    valid = np.isfinite(next_common).all(axis=1) & np.isfinite(returns)
    if valid.sum() < 4:
        raise ValueError("Fewer than four valid observable forecast scenarios")
    changes = next_common[valid] - current.reshape(-1)
    with np.load(path / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    targets = np.c_[changes @ basis, returns[valid]]
    joint = np.c_[returns[valid], changes @ basis]
    quantiles = np.quantile(targets, QUANTILES, axis=0)
    actual = np.asarray(baseline["actual_change"])
    diagnostics = {**baseline["diagnostics"], **guards,
        "stability": _backbone_stability(fit), "raw_spectral_radius": fit.unconstrained_spectral_radius,
        "metric_pca_coverage": float(fit.explained_variance_fraction.sum()),
        "zero_weight_coordinates": int(np.sum(model.weights == 0.)),
        "min_positive_weight": float(model.weights[model.weights > 0.].min()),
        "metric": "origin-window mean probability for log-DLV-squared; unit weight for log forward vols"}
    diagnostics.pop("native_pca_coverage", None)
    return {**baseline, "model": f"{baseline['model']} density-weighted", "scenario_count": len(surfaces),
        "valid_scenario_count": int(valid.sum()), "scenario_covariance_ddof": 0,
        "surface_mean": changes.mean(axis=0).tolist(), "surface_covariance": covariance(changes, ddof=0).tolist(),
        "joint_mean": joint.mean(axis=0).tolist(), "joint_covariance": covariance(joint, ddof=0).tolist(),
        "quantiles": quantiles.tolist(),
        "es_lower": [float(targets[targets[:, column] <= quantiles[1, column], column].mean()) for column in range(4)],
        "invalid_surface_rate": float(np.mean(~valid)),
        "extreme_surface_rate": float(np.mean(np.any(next_common[valid] > np.log(3.), axis=1))),
        "forecast_rmse": float(np.sqrt(np.mean((changes.mean(axis=0) - actual)**2))), "diagnostics": diagnostics}


def _write_results(output: Path, source: Path, records: dict[str, list[dict]]) -> None:
    with np.load(source / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    summaries = {}
    rows = []
    for model, entries in records.items():
        summaries[model] = _summarize(entries, basis)
        selected = [entry for entry in entries if entry.get("status") == "scored"]
        summary = summaries[model]
        rows.append({"model": model, "scored_origins": len(selected), "requested_origins": len(entries),
            "conditional_trace_ratio": summary.get("conditional_covariance", {}).get("trace_ratio"),
            "pooled_trace_ratio": summary.get("pooled_covariance", {}).get("trace_ratio"),
            "pooled_covariance_error": summary.get("pooled_covariance", {}).get("cov_frob_rel"),
            "forecast_rmse": summary.get("forecast_rmse"),
            "maximum_ar_radius": max((entry["diagnostics"]["stability"]["spectral_radius"] for entry in selected), default=None),
            "mean_metric_pca_coverage": (float(np.mean([entry["diagnostics"]["metric_pca_coverage"] for entry in selected]))
                                         if selected and "metric_pca_coverage" in selected[0]["diagnostics"] else None),
            "median_retained_forward_vol_energy_fraction": (float(np.median([
                entry["diagnostics"]["compression"]["retained_native_forward_vol_energy_fraction"] for entry in selected]))
                if selected and "compression" in selected[0]["diagnostics"] else None),
            "median_floor_crossing_share_of_weighted_energy": (float(np.median([
                entry["diagnostics"]["compression"]["floor_crossing_energy_fraction_of_weighted_total"] for entry in selected]))
                if selected and "compression" in selected[0]["diagnostics"] else None),
            "mean_invalid_scenario_rate": float(np.mean([entry["invalid_surface_rate"] for entry in selected])) if selected else None,
            "lower_clips": sum(entry["diagnostics"]["lower_clips"] for entry in selected),
            "upper_clips": sum(entry["diagnostics"]["upper_clips"] for entry in selected)})
    _json(output / "summary.json", summaries)
    _json(output / "COMPARISON.json", rows)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    lines = ["# Density-weighted DYSANOS PCA/VAR test", "",
        "Same 44 April–May 2024 origins and each origin's original 252 causal training pairs. "
        "Sixteen factors, unchanged daily states, decoder, spot law, VAR ridge and spectral-radius bound. "
        "Both ordinary and spot-orthogonal variants are fitted. March 28 seeds previous-date factor alignment.", "",
        "For each fitted origin, w[j,k] is the mean decoded grid probability mass at expiry j and node k "
        "over its 252 training origins. The loss metric is sum(w * error(log DLV squared)^2), "
        "with unit weight for each log-forward-volatility coordinate. PCA and VAR operate on sqrt(w)*u; "
        "forecast changes and innovations are mapped back to native u before pricing. "
        "Weights are fixed inside each fit, then recomputed causally next day. They are probabilities, "
        "not density heights on the nonuniform strike grid. No probability floor, shock rescaling or decoder change is introduced. "
        "Zero-weight coordinates, if any, remain at their current native values. "
        "Previous factor directions are mapped into the new metric for rolling orientation alignment.", "",
        "| Model | Scored origins | Conditional variance / observed | Pooled variance / observed | Covariance error | Forecast RMSE | Max AR radius |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["scored_origins"] < 2:
            lines.append(f"| {row['model']} | {row['scored_origins']}/{row['requested_origins']} | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(f"| {row['model']} | {row['scored_origins']}/{row['requested_origins']} | "
            f"{row['conditional_trace_ratio']:.3f} | {row['pooled_trace_ratio']:.3f} | "
            f"{row['pooled_covariance_error']:.3f} | {row['forecast_rmse']:.4f} | {row['maximum_ar_radius']:.4f} |")
    lines.extend(["", "Variance ratios use the same 49 observable floating-NMS log-IV changes as the published audit. "
        "The conditional denominator is unconditional observed variance; one is not a prescribed conditional target. "
        "Pooled covariance additionally includes variation of forecast means. "
        "Lower covariance error and RMSE are closer to the common observed reference. "
        "This is a one-day forecast/compression experiment, not a recursive-path or hedging rerun. "
        "The original DYSANOS input smoothness 0.1 versus observable-reference smoothness 0.25 difference remains unchanged.", "",
        "Per-origin records retain all decoder clips, invalid scenarios, quantile scores and fit provenance. "
        "The unweighted baseline records are copied without modifying their numerical results. "
        "Any incomplete or failed origins are explicit in daily records; weighted checkpoints and metric files are retained for reuse."])
    for row in rows:
        if row["median_retained_forward_vol_energy_fraction"] is not None:
            lines.extend(["", f"{row['model']}: median forward-volatility change energy retained before VAR residual extraction "
                f"{row['median_retained_forward_vol_energy_fraction']:.2%}; median floor-crossing share of the new weighted "
                f"squared-change energy {row['median_floor_crossing_share_of_weighted_energy']:.2%}."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run(*, root: Path, source: Path, output: Path) -> None:
    directory = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    values = json.loads((directory / "config.json").read_text())
    dlv_config = DlvConfig(**{field.name: values[field.name] for field in fields(DlvConfig) if field.name in values})
    observations = pd.read_parquet(directory / "observations.parquet").set_index("date")
    observations.index = pd.to_datetime(observations.index).date
    pairs = _training_pairs(root)
    origins = json.loads((source / "observable_reference.json").read_text())["dates"]
    evaluation = set(map(date.fromisoformat, origins))
    all_dates = tuple(sorted({day for pair in pairs if pair[1] <= max(evaluation) for day in pair[:2]}))

    @lru_cache(maxsize=600)
    def daily(day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(directory / f"days/{day}.npz", allow_pickle=False) as saved:
            return saved["raw_u"], saved["decoded_densities"], saved["support_strikes"]

    states = np.stack([daily(day)[0] for day in all_dates])
    aligned = observations.loc[list(all_dates)]
    returns = aligned["normalized_spot_return_into_date"].to_numpy(dtype=float)
    sessions = aligned["session_index"].to_numpy(dtype=int)
    candidates = [day for day in all_dates if date(2024, 3, 28) <= day <= max(evaluation)]
    protocol = {"schema": "density-weighted-pca-var-v1", "source_audit": str(source),
        "source_config_sha256": _hash(directory / "config.json"), "rank": 16, "rolling_pairs": 252,
        "scored_origins": origins, "weighting": "mean training-origin grid masses as squared-error weights",
        "forward_vol_weight": 1., "zero_probability_policy": "freeze native coordinate", "probability_floor": None,
        "fit_coordinates": "sqrt(weight)*native", "decoder_changed": False, "shock_rescaling": False}
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "PROTOCOL.json"
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError("Output directory belongs to a different density-weighting experiment")
    _json(existing, protocol)
    records = {}
    inputs = {directory / "config.json", directory / "observations.parquet", source / "frozen_observable_pca.npz"}
    inputs.add(Path(__file__))
    started = time.monotonic()
    completed, total = 0, 2 * len(candidates)
    for orthogonal in (False, True):
        variant = "orthogonal" if orthogonal else "ordinary"
        name = f"DYSANOS {'spot-orthogonal' if orthogonal else 'ordinary'} (16 factors)"
        config_values = json.loads((source / f"fits/{variant}/config.json").read_text())
        config = DynamicsConfig(**{field.name: config_values[field.name] for field in fields(DynamicsConfig)})
        previous = None
        records[name], records[f"{name} density-weighted"] = [], []
        for day in candidates:
            window = [pair for pair in pairs if pair[1] <= day][-config.window_transitions:]
            metric = density_metric(np.stack([daily(pair[0])[1] for pair in window]))
            fit_directory = output / "fits" / variant / str(day)
            if (fit_directory / "metric.npz").exists():
                model = _load(fit_directory)
                if not np.array_equal(metric, model.weights):
                    raise ValueError("Saved metric differs from the causal origin-window probabilities")
            else:
                model = fit_weighted(weights=metric, previous=previous, dates=all_dates, states=states,
                    spot_returns_into_date=returns, session_indices=sessions, as_of_date=day, config=config)
                _save(model, fit_directory)
            previous = model
            if model.weighted_fit.initialisation_date != (None if day == candidates[0] else candidates[candidates.index(day) - 1]):
                raise ValueError("Rolling initialization chain is not causal and continuous")
            inputs.update(directory / f"days/{pair[0]}.npz" for pair in window)
            inputs.update(directory / f"days/{pair[1]}.npz" for pair in window)
            if day in evaluation:
                baseline_path = source / "daily" / name.replace(" ", "_") / f"{day}.json"
                baseline = json.loads(baseline_path.read_text())
                original_fit_path = source / f"fits/{variant}/days/{day}.npz"
                original_fit = load_dynamics(original_fit_path)
                inputs.update((baseline_path, original_fit_path, source / f"fits/{variant}/config.json"))
                if ((original_fit.training_start, original_fit.training_end) != (
                        model.weighted_fit.training_start, model.weighted_fit.training_end)
                        or not np.array_equal(original_fit.residuals[:, 0], model.weighted_fit.residuals[:, 0])):
                    raise ValueError("Density-weighting treatment changed training dates or spot shocks")
                records[name].append(baseline)
                destination = output / "daily" / f"{name} density-weighted".replace(" ", "_") / f"{day}.json"
                if destination.exists():
                    record = json.loads(destination.read_text())
                else:
                    native, _, support = daily(day)
                    state = DailyState(day=day, native=native, reference_support=support, config=dlv_config)
                    try:
                        record = _score(model, state, baseline, path=source)
                    except (ValueError, RuntimeError, FloatingPointError) as error:
                        record = {"model": f"{name} density-weighted", "origin_date": str(day),
                                  "status": "error", "reason": f"{type(error).__name__}: {error}"}
                if record.get("status") == "scored":
                    record["diagnostics"]["compression"] = _compression(model,
                        np.stack([daily(pair[0])[0] for pair in window]),
                        np.stack([daily(pair[1])[0] for pair in window]), config=dlv_config)
                _json(destination, record)
                records[f"{name} density-weighted"].append(record)
                print(f"density-weighted {variant} {day} {record['status']}", flush=True)
            completed += 1
            elapsed = time.monotonic() - started
            print(f"density experiment {completed}/{total} elapsed={elapsed:.1f}s eta={elapsed*(total-completed)/completed:.1f}s", flush=True)
        _write_results(output, source, records)
    _json(output / "INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in sorted(inputs)])
    print(f"Density-weighted experiment complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / ".experiment.lock"):
        run(root=arguments.root, source=arguments.source, output=arguments.output)


if __name__ == "__main__":
    main()
