"""Separate ATM-derived floating-grid factors from density-weighted DLV shape."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, fields
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from filelock import FileLock
from scipy.optimize import linear_sum_assignment

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import DynamicsConfig
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.density_weighted import _load, density_metric
from quantdata.hedging.model_audit.metrics import covariance, linear_stability
from quantdata.hedging.model_audit.run import QUANTILES, _dlv_surface, _hash, _json, _summarize
from quantdata.hedging.model_audit.tail import _training_pairs


@dataclass(frozen=True, kw_only=True)
class GridShapeFit:
    as_of_date: str
    training_start: str
    training_end: str
    initialisation_date: str
    native_mean: np.ndarray
    encoder: np.ndarray
    decoder: np.ndarray
    native_spot_loading: np.ndarray
    ar_matrix: np.ndarray
    ar_intercept: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray
    grid_indices: np.ndarray
    shape_indices: np.ndarray
    grid_rank: int
    grid_coverage: float
    shape_coverage: float
    unconstrained_radius: float
    constrained_radius: float

    def forecast_change(self, current: np.ndarray) -> np.ndarray:
        scores = (current - self.native_mean) @ self.encoder
        return self.decoder @ (self.ar_intercept + self.ar_matrix @ scores - scores)

    def next_native(self, current: np.ndarray) -> np.ndarray:
        return (current + self.forecast_change(current) + self.residuals[:, :1] * self.native_spot_loading
                + self.residuals[:, 1:] @ self.decoder.T)

    def retained_changes(self, changes: np.ndarray) -> np.ndarray:
        projected_beta = self.decoder @ (self.native_spot_loading @ self.encoder)
        return ((changes @ self.encoder) @ self.decoder.T
                + self.residuals[:, :1] * (self.native_spot_loading - projected_beta))


def _directions(changes: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    gram = changes @ changes.T / (len(changes) - 1)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:rank]
    if values[order[-1]] <= np.finfo(float).eps * max(float(values.max()), 1.):
        raise ValueError("The requested grid/shape PCA rank exceeds the numerical data rank")
    directions = changes.T @ vectors[:, order] / np.sqrt((len(changes) - 1) * values[order])
    return directions, float(values[order].sum() / np.trace(gram))


def _align(directions: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is not None:
        old, new = linear_sum_assignment(-np.abs(previous.T @ directions))
        directions = directions[:, new[np.argsort(old)]]
        signs = np.sign(np.sum(previous * directions, axis=0))
    else:
        signs = np.sign(directions[np.argmax(np.abs(directions), axis=0), np.arange(directions.shape[1])])
    return directions * np.where(signs == 0., 1., signs)


def fit_grid_shape(*, before: np.ndarray, after: np.ndarray, spot: np.ndarray,
                   weights: np.ndarray, config: DynamicsConfig, native_config: DlvConfig,
                   as_of_date: str, training_start: str, training_end: str,
                   grid_rank: int = 3, previous: GridShapeFit | None = None) -> GridShapeFit:
    """Fit separate bases, conditional shape loadings and one joint stable VAR."""
    if (before.shape != after.shape or before.ndim != 2 or len(before) != config.window_transitions
            or weights.shape != (before.shape[1],) or spot.shape != (len(before),)
            or not all(np.isfinite(values).all() for values in (before, after, spot, weights))
            or np.any(weights < 0.) or not 0 < grid_rank < config.rank):
        raise ValueError("Grid/shape fitting requires aligned finite training pairs and a valid factor split")
    if previous is not None and previous.as_of_date >= as_of_date:
        raise ValueError("Rolling factor orientation must come from an earlier date")
    packed_indices = np.arange(before.shape[1]).reshape(len(native_config.business_dtes), native_config.num_inner_strikes + 1)
    grid_indices, shape_indices = packed_indices[:, -1], packed_indices[:, :-1].reshape(-1)
    changes = after - before
    changes -= changes.mean(axis=0)
    centered_spot = spot - spot.mean()
    beta = centered_spot @ changes / (centered_spot @ centered_spot) if config.spot_orthogonal else np.zeros(before.shape[1])
    residual = changes - centered_spot[:, None] * beta
    grid, grid_coverage = _directions(residual[:, grid_indices], grid_rank)
    grid = _align(grid, None if previous is None else previous.decoder[grid_indices, :grid_rank])
    grid_changes = residual[:, grid_indices] @ grid
    coefficient = np.linalg.lstsq(grid_changes, residual[:, shape_indices], rcond=None)[0]
    shape_changes = residual[:, shape_indices] - grid_changes @ coefficient
    scale = np.sqrt(weights[shape_indices])
    shape, shape_coverage = _directions(shape_changes * scale, config.rank - grid_rank)
    prior_shape = None
    if previous is not None:
        prior_shape = previous.decoder[shape_indices, grid_rank:] * scale[:, None]
        norms = np.linalg.norm(prior_shape, axis=0)
        if np.any(norms <= 0.) or not np.isfinite(prior_shape).all():
            raise ValueError("Previous shape factors cannot be mapped into the new density metric")
        prior_shape /= norms
    shape = _align(shape, prior_shape)
    decoder = np.zeros((before.shape[1], config.rank))
    encoder = np.zeros_like(decoder)
    decoder[grid_indices, :grid_rank] = grid
    decoder[shape_indices, :grid_rank] = coefficient.T
    decoder[shape_indices, grid_rank:] = np.divide(shape, scale[:, None], out=np.zeros_like(shape), where=scale[:, None] > 0.)
    encoder[grid_indices, :grid_rank] = grid
    encoder[shape_indices, grid_rank:] = scale[:, None] * shape
    encoder[grid_indices, grid_rank:] = -grid @ coefficient @ encoder[shape_indices, grid_rank:]
    np.testing.assert_allclose(encoder.T @ decoder, np.eye(config.rank), atol=1e-10, rtol=1e-10)
    mean = np.vstack((before, after)).mean(axis=0)
    old_scores, new_scores = (before - mean) @ encoder, (after - mean) @ encoder
    old_mean, new_mean = old_scores.mean(axis=0), new_scores.mean(axis=0)
    centered_old = old_scores - old_mean
    score_beta = beta @ encoder
    target = new_scores - new_mean - centered_spot[:, None] * score_beta
    cross = centered_old.T @ centered_old
    ridge_scale = max(float(np.trace(cross) / config.rank), np.finfo(float).tiny)
    matrix = np.linalg.solve(cross + config.regression_ridge * ridge_scale * np.eye(config.rank), centered_old.T @ target).T
    raw_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if raw_radius > config.max_spectral_radius:
        matrix *= config.max_spectral_radius / raw_radius
    intercept = new_mean - matrix @ old_mean
    errors = new_scores - old_scores @ matrix.T - intercept - centered_spot[:, None] * score_beta
    joint = np.c_[centered_spot, errors]
    joint -= joint.mean(axis=0)
    return GridShapeFit(as_of_date=as_of_date, training_start=training_start, training_end=training_end,
        initialisation_date="" if previous is None else previous.as_of_date,
        native_mean=mean, encoder=encoder, decoder=decoder, native_spot_loading=beta,
        ar_matrix=matrix, ar_intercept=intercept, residuals=joint, weights=weights,
        grid_indices=grid_indices, shape_indices=shape_indices, grid_rank=grid_rank,
        grid_coverage=grid_coverage, shape_coverage=shape_coverage, unconstrained_radius=raw_radius,
        constrained_radius=float(np.max(np.abs(np.linalg.eigvals(matrix)))))


def save_model(model: GridShapeFit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **asdict(model))
    temporary.replace(path)


def load_model(path: Path) -> GridShapeFit:
    with np.load(path, allow_pickle=False) as saved:
        values = {field.name: saved[field.name] for field in fields(GridShapeFit)}
    for name in ("as_of_date", "training_start", "training_end", "initialisation_date"):
        values[name] = str(values[name])
    for name in ("grid_coverage", "shape_coverage", "unconstrained_radius", "constrained_radius"):
        values[name] = float(values[name])
    values["grid_rank"] = int(values["grid_rank"])
    return GridShapeFit(**values)


def _score(state: DailyState, following: np.ndarray, returns: np.ndarray, baseline: dict,
           *, label: str, basis: np.ndarray, diagnostics: dict) -> dict:
    current, _ = _dlv_surface(state, state.native[None])
    surfaces, guards = _dlv_surface(state, following)
    surfaces = surfaces.reshape(len(following), -1)
    valid = np.isfinite(surfaces).all(axis=1) & np.isfinite(returns)
    if valid.sum() < 4:
        raise ValueError("Fewer than four valid grid/shape observable forecast scenarios")
    changes = surfaces[valid] - current.reshape(-1)
    joint = np.c_[returns[valid], changes @ basis]
    targets = np.c_[changes @ basis, returns[valid]]
    quantiles = np.quantile(targets, QUANTILES, axis=0)
    return {**baseline, "model": label, "scenario_count": len(following), "valid_scenario_count": int(valid.sum()),
        "surface_mean": changes.mean(axis=0).tolist(), "surface_covariance": covariance(changes, ddof=0).tolist(),
        "joint_mean": joint.mean(axis=0).tolist(), "joint_covariance": covariance(joint, ddof=0).tolist(),
        "scenario_covariance_ddof": 0, "quantiles": quantiles.tolist(),
        "es_lower": [float(targets[targets[:, column] <= quantiles[1, column], column].mean()) for column in range(4)],
        "invalid_surface_rate": float(np.mean(~valid)),
        "extreme_surface_rate": float(np.mean(np.any(surfaces[valid] > np.log(3.), axis=1))),
        "forecast_rmse": float(np.sqrt(np.mean((changes.mean(axis=0) - np.asarray(baseline["actual_change"]))**2))),
        "diagnostics": {**baseline["diagnostics"], **guards, **diagnostics}}


def run(root: Path, source: Path, weighted: Path, output: Path) -> dict:
    directory = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    values = json.loads((directory / "config.json").read_text())
    config = DlvConfig(**{field.name: values[field.name] for field in fields(DlvConfig) if field.name in values})
    dates = json.loads((source / "observable_reference.json").read_text())["dates"]
    if len(dates) != 44:
        raise ValueError("The controlled comparison requires the existing 44 origins")
    with np.load(source / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    pairs = _training_pairs(root)
    protocol = {"schema": "explicit-grid3-shape13-v1", "source": str(source), "weighted_comparator": str(weighted),
        "grid_factors": 3, "shape_factors": 13, "rolling_transitions": 252, "origins": dates,
        "grid_variables": "nine log forward volatilities derived from the ATM total-variance curve",
        "estimator": "separate closed-form PCA/conditional regressions and one joint ridge VAR; not joint JAX calibration",
        "metric": "same causal training-origin mean node probabilities as the mixed sixteen-factor comparator",
        "discarded_control": "paired omitted centered native changes restored; diagnostic, not a calibrated full-rank model",
        "source_config_sha256": _hash(directory / "config.json"), "price_decoder_changed": False,
        "rolling_forecast_residual_calibration_run": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PROTOCOL.json").exists() and json.loads((output / "PROTOCOL.json").read_text()) != protocol:
        raise ValueError("Output directory belongs to a different explicit-grid protocol")
    _json(output / "PROTOCOL.json", protocol)

    @lru_cache(maxsize=600)
    def daily(day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(directory / f"days/{day}.npz", allow_pickle=False) as saved:
            return saved["raw_u"], saved["decoded_densities"], saved["support_strikes"]

    all_records = {}
    started = time.monotonic()
    for variant in ("ordinary", "orthogonal"):
        ordinary_name = f"DYSANOS {'spot-orthogonal' if variant == 'orthogonal' else 'ordinary'} (16 factors)"
        mixed_name = f"{ordinary_name} density-weighted"
        split_name = f"{ordinary_name} grid3-shape13"
        all_records[mixed_name], all_records[split_name] = [], []
        all_records[f"{mixed_name} + discarded"], all_records[f"{split_name} + discarded"] = [], []
        previous = None
        for text in ["2024-03-28", *dates]:
            day = date.fromisoformat(text)
            mixed = _load(weighted / f"fits/{variant}/{text}")
            fit = mixed.weighted_fit
            window = [pair for pair in pairs if pair[1] <= day][-252:]
            before = np.stack([daily(pair[0])[0] for pair in window])
            after = np.stack([daily(pair[1])[0] for pair in window])
            metric = density_metric(np.stack([daily(pair[0])[1] for pair in window]))
            np.testing.assert_array_equal(metric, mixed.weights)
            fit_path = output / f"fits/{variant}/{text}.npz"
            if fit_path.exists():
                split = load_model(fit_path)
            else:
                split = fit_grid_shape(before=before, after=after, spot=fit.residuals[:, 0], weights=metric,
                    config=DynamicsConfig(rank=16, spot_orthogonal=variant == "orthogonal"), native_config=config,
                    as_of_date=text, training_start=str(fit.training_start), training_end=str(fit.training_end), previous=previous)
                save_model(split, fit_path)
            previous = split
            if text not in dates:
                continue
            baseline = json.loads((weighted / "daily" / mixed_name.replace(" ", "_") / f"{text}.json").read_text())
            all_records[mixed_name].append(baseline)
            native, _, support = daily(day)
            state = DailyState(day=day, native=native, reference_support=support, config=config)
            changes = after - before
            changes -= changes.mean(axis=0)
            mixed_spot = fit.residuals[:, :1] * fit.spot_state_loading
            weighted_changes = changes * mixed.scale
            mixed_projection = mixed.unscale_changes(mixed_spot + ((weighted_changes - mixed_spot) @ fit.directions) @ fit.directions.T)
            split_projection = split.retained_changes(changes)
            discarded_split = changes - split_projection
            discarded_mixed = changes - mixed_projection
            next_split = split.next_native(native)
            score_beta = split.native_spot_loading @ split.encoder
            score_innovations = split.residuals[:, 1:] + split.residuals[:, :1] * score_beta
            split_diagnostics = {"grid_pca_coverage": split.grid_coverage, "shape_pca_coverage": split.shape_coverage,
                "raw_spectral_radius": split.unconstrained_radius,
                "stability": linear_stability(split.ar_matrix, covariance(score_innovations, ddof=0)),
                "grid_change_energy_retained": float(np.sum(split_projection[:, split.grid_indices]**2) / np.sum(changes[:, split.grid_indices]**2)),
                "spot_loading_max_error_vs_mixed": float(np.max(np.abs(split.native_spot_loading - mixed.unscale_changes(fit.spot_state_loading)))),
                "encoder_decoder_identity_error": float(np.max(np.abs(split.encoder.T @ split.decoder - np.eye(16)))),
                "training_start": split.training_start, "training_end": split.training_end,
                "estimator": protocol["estimator"]}
            groups = {split_name: (next_split, split_diagnostics),
                f"{split_name} + discarded": (next_split + discarded_split, {**split_diagnostics, "diagnostic_ablation": True}),
                f"{mixed_name} + discarded": (mixed.next_native(native) + discarded_mixed, {"diagnostic_ablation": True})}
            for label, (following, diagnostics) in groups.items():
                path = output / "daily" / label.replace(" ", "_") / f"{text}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                else:
                    try:
                        record = _score(state, following, np.log1p(fit.residuals[:, 0]), baseline,
                                        label=label, basis=basis, diagnostics=diagnostics)
                    except (ValueError, RuntimeError, FloatingPointError) as error:
                        record = {"model": label, "origin_date": text, "status": "error", "reason": str(error)}
                    _json(path, record)
                all_records[label].append(record)
            print(f"grid/shape {variant} {text} completed={len(all_records[split_name])}/44 elapsed={time.monotonic()-started:.1f}s", flush=True)
    summaries = {name: _summarize(records, basis) for name, records in all_records.items()}
    _json(output / "summary.json", summaries)
    rows = []
    lines = ["# Explicit ATM-derived grid factors: first controlled test", "",
        "Legacy saved daily u states; separate PCA/VAR estimation, not the joint JAX DYSANOS fit. "
        "Same 44 origins, 252 historical pairs, 16 total factors, density metric and decoder. "
        "Three factors describe the log-forward-volatility curve that sets the floating grid; thirteen describe residual DLV shape. "
        "The shape regression removes retained grid moves and, for the orthogonal variant, spot jointly. "
        "One joint VAR retains cross-dynamics and paired spot/factor innovations. These grid variables are not pure decoded ATM-IV Greeks.", "",
        "| Model | Origins | Conditional trace / observed | Pooled trace / observed | Covariance error | Forecast RMSE | Variance / forecast MSE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        row = {"model": name, "origins": summary["surface_days"]}
        if row["origins"] >= 2:
            row.update(conditional=summary["conditional_covariance"]["trace_ratio"], pooled=summary["pooled_covariance"]["trace_ratio"],
                covariance_error=summary["pooled_covariance"]["cov_frob_rel"], rmse=summary["forecast_rmse"],
                variance_over_forecast_mse=summary["forecast_variance_calibration"]["conditional_variance_over_forecast_mse"])
            lines.append(f"| {name} | {row['origins']}/44 | {row['conditional']:.3f} | {row['pooled']:.3f} | "
                f"{row['covariance_error']:.3f} | {row['rmse']:.4f} | {row['variance_over_forecast_mse']:.3f} |")
        rows.append(row)
    lines.extend(["", "Rows marked '+ discarded' are same-row decoder ablations, not calibrated alternative forecasting models. "
        "They retain the original mean and paired innovations and restore omitted centered native changes. "
        "These counterfactual variances are not additive risk contributions. "
        "No volatility multiplier, new price interpolation or probability floor is introduced. "
        "Recursive/quote reruns and genuine historical rolling-forecast-residual calibration are not performed in this first test."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    _json(output / "COMPARISON.json", rows)
    _json(output / "INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in (
        directory / "config.json", weighted / "PROTOCOL.json", source / "observable_reference.json", Path(__file__))])
    return {"models": rows, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weighted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / ".experiment.lock"):
        run(arguments.root, arguments.source, arguments.weighted, arguments.output)


if __name__ == "__main__":
    main()
