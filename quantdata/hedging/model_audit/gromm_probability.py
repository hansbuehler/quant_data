"""Controlled probability-weighted innovation PCA for standalone GROMM v1."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from quantdata.hedging.model_audit.run import _hash, _json


def probability_metric(densities: np.ndarray, date_weights: np.ndarray) -> np.ndarray:
    """Causal date-average interior probability masses, not density heights."""
    densities, date_weights = np.asarray(densities, float), np.asarray(date_weights, float)
    if (densities.ndim != 3 or date_weights.shape != (len(densities),)
            or not np.isfinite(densities).all() or np.any(densities < 0)
            or not np.isfinite(date_weights).all() or np.any(date_weights <= 0)
            or not np.allclose(densities.sum(axis=-1), 1., atol=1e-10, rtol=0.)):
        raise ValueError("Expected normalized historical probability masses and positive date weights")
    average = np.einsum("d,djn->jn", date_weights / date_weights.sum(), densities[:, 1:, 1:-1])
    if np.any(average <= 0):
        raise ValueError("Zero-mass coordinates are unidentified by the probability metric; no arbitrary probability floor applied")
    return np.sqrt(average)


def to_native_loading(weighted_loading: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Undo the coordinate metric without rescaling the model's variance."""
    if (weighted_loading.shape[:-1] != metric.shape or not np.isfinite(metric).all() or np.any(metric <= 0)
            or not np.isfinite(weighted_loading).all()):
        raise ValueError("Loading and strictly positive coordinate metric must be compatible")
    return weighted_loading / metric[..., None]


def run(representation_path: Path, baseline_path: Path, output: Path, *, previous_path: Path | None = None,
        scope: str = "single saved origin pilot; not the older April-May hedging GROMM2 cohort") -> None:
    """Reuse the exact price fit and independent-Y likelihood; refit only diffusion."""
    import jax
    import jax.numpy as jnp
    from openqdata.models.gromm.artifacts import content_identity, load_archive, save_archive
    from openqdata.models.gromm.config import TrainingConfig
    from openqdata.models.gromm.dynamics import (
        LATENT_INITIALIZATION,
        load_dynamics,
        verify_representation_content,
        warm_start_latent,
    )
    from openqdata.models.gromm.estimation import fit_ar1, normalized_weights
    from openqdata.models.gromm.latent_estimation import LatentInitialization, fit_latent_diffusion
    from openqdata.models.gromm.laws import p_coefficients, q_coefficients
    from openqdata.models.gromm.numerics import decode, require_float64
    from openqdata.models.gromm.progress import Heartbeat
    from openqdata.models.gromm.rolling import _representation_from_snapshot

    require_float64()
    output.mkdir(parents=True, exist_ok=True)
    report, native = load_archive(representation_path)
    baseline = load_dynamics(baseline_path)
    verify_representation_content(baseline.metadata, content_identity(report, native))
    if baseline.metadata["model"] != "gromm1" or not baseline.metadata.get("observation_fit", {}).get("independent_Y_preserved"):
        raise ValueError("This ablation requires the source-bound, independent-Y-compatible GROMM1 baseline")
    representation = _representation_from_snapshot(report, native)
    training = TrainingConfig.from_dict(report.get("training_configuration", {}))
    representation.require_quality(training)
    provenance = report["provenance"]
    endpoint = provenance["endpoint_extraction"]
    for day, record in provenance["source"]["verified_native_files"].items():
        if _hash(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Historical native source changed on {day}")
    for day, record in provenance["source"]["saved_sanos"]["verified_daily_parameters"].items():
        if _hash(Path(record["path"])) != record["sha256"] or _hash(Path(record["certificate_path"])) != record["certificate_sha256"]:
            raise ValueError(f"Historical SANOS source changed on {day}")
    if endpoint["normalization"] != baseline.metadata["provenance"]["endpoint_extraction"]["normalization"]:
        raise ValueError("Endpoint normalization differs from the compatible baseline")
    lookup = {str(day): index for index, day in enumerate(representation.dates)}
    pairs = endpoint["pairs"]
    if any(after >= baseline.metadata["origin"] for before, after in pairs):
        raise ValueError("Origin or future observation entered the dynamics pairs")
    before_indices = np.asarray([lookup[before] for before, after in pairs])
    after_indices = np.asarray([lookup[after] for before, after in pairs])
    states = representation.fit.states[before_indices]
    weights = native["date_weights"][after_indices]
    gross = 1. / np.asarray(endpoint["endpoint_strike_scales"])
    regression = fit_ar1(states[:, 1:], representation.endpoints.states, weights)
    np.testing.assert_allclose(regression.slope, baseline.arrays["ar1_slope"], atol=1e-12, rtol=1e-12)
    geometry = representation.geometry
    rank = baseline.rank
    protocol = {"schema": "gromm1-probability-pca-ablation-v1", "origin": baseline.metadata["origin"],
        "representation_sha256": _hash(representation_path), "baseline_sha256": _hash(baseline_path),
        "baseline_identity": baseline.identity, "pairs": len(pairs), "rank": rank,
        "metric": "sqrt(date-weighted historical node probability mass), future expiries, interior nodes",
        "unchanged": ["market fit", "geometry", "fixed-contract returns", "date weights", "AR1", "rank", "independent Y likelihood"],
        "refitted": ["weighted initial PCA", "latent spot dependence", "weighted residual PCA", "compatible Q drift", "attainable P drift"],
        "previous_weighted_sha256": None if previous_path is None else _hash(previous_path),
        "initialization": "first causal origin" if previous_path is None else str(previous_path),
        "scope": scope,
        "source_sha256": _hash(Path(__file__))}
    if (output / "PROTOCOL.json").exists() and json.loads((output / "PROTOCOL.json").read_text()) != protocol:
        raise ValueError("Ablation identity changed; use a unique output directory")
    _json(output / "PROTOCOL.json", protocol)
    started = time.monotonic()

    def emit(event):
        print(json.dumps(dict(event, elapsed_seconds=time.monotonic() - started), allow_nan=False), flush=True)

    with Heartbeat(emit, interval=60.) as progress:
        decoder = jax.jit(jax.vmap(lambda value: decode(value, jnp.asarray(geometry.supports),
            jnp.asarray(geometry.transports), jnp.asarray(geometry.expiries), linear_solver=training.fit.linear_solver)))
        probabilities = []
        for position in range(0, len(states), 16):
            probabilities.append(np.asarray(decoder(jnp.asarray(states[position:position + 16]))))
            progress.update({"phase": "historical_probabilities", "completed": min(position + 16, len(states)), "total": len(states)})
        densities = np.concatenate(probabilities)
        metric = probability_metric(densities, weights)
        initialization = None
        if previous_path is not None:
            previous = load_dynamics(previous_path)
            native_initialization = warm_start_latent(previous, geometry, representation.model, representation.origin)
            initialization = LatentInitialization(native_initialization.basis * metric.ravel()[:, None],
                native_initialization.correlation, native_initialization.location, native_initialization.provenance)
        normalized = normalized_weights(weights, len(pairs))
        annual = regression.residuals / np.sqrt(regression.dt)
        annual -= np.einsum("d,djn->jn", normalized, annual)
        np.savez_compressed(output / "input-metric.npz", metric=metric,
            historical_probability_mean=metric**2, date_weights=weights,
            representation_identity=np.asarray(representation.identity))
        weighted_fit = fit_latent_diffusion(regression.residuals * metric, gross, densities[:, 0],
            geometry.supports[0], float(geometry.variance_at(regression.dt, backbone=True)), weights,
            rank=rank, dt=regression.dt, progress=progress.update, previous=initialization)
        loadings = to_native_loading(weighted_fit.diffusion.loadings, metric)
        current = representation.fit.states[-1]
        current_densities = decode(jnp.asarray(current), jnp.asarray(geometry.supports), jnp.asarray(geometry.transports),
            jnp.asarray(geometry.expiries), linear_solver=training.fit.linear_solver)
        current_densities, q_drift, front_loading = q_coefficients(current_densities[0], jnp.asarray(current[1:]),
            jnp.asarray(loadings), jnp.asarray(geometry.supports), jnp.asarray(geometry.transports),
            jnp.asarray(np.diff(geometry.expiries)), jnp.asarray(geometry.expiries[0]), linear_solver=training.fit.linear_solver)
        p_drift, risk_price, captured, residual = p_coefficients(current_densities[0], jnp.asarray(current[1:]), current_densities,
            q_drift, front_loading, jnp.asarray(loadings), jnp.asarray(geometry.supports),
            jnp.asarray(regression.intercept), jnp.asarray(regression.slope), regression.dt, baseline.metadata["spot_p_drift"])
        if not all(np.isfinite(value).all() for value in (q_drift, p_drift, risk_price)):
            raise ValueError("Weighted PCA produces nonfinite compatible drifts; no repair applied")
        arrays = dict(baseline.arrays)
        arrays.update(log_variance_loading=loadings, probability_pca_metric=metric,
            latent_pca_basis=weighted_fit.initialization.basis / metric.ravel()[:, None],
            latent_correlation=weighted_fit.initialization.correlation,
            current_q_log_drift=np.asarray(q_drift), current_p_log_drift=np.asarray(p_drift),
            current_risk_price=np.asarray(risk_price), residual_eigenvalues=weighted_fit.diffusion.residual_eigenvalues)
        weighted_marginal = np.einsum("d,djn->jn", normalized, annual**2) / (1 - normalized @ normalized)
        native_total = float(weighted_marginal.sum())
        weighted_total = float(np.sum(weighted_marginal * metric**2))
        baseline_loading = baseline.arrays["log_variance_loading"]
        diagnostics = {"weighted_metric_variance_retention_baseline": float(np.sum((baseline_loading * metric[..., None])**2) / weighted_total),
            "weighted_metric_variance_retention_new": float(np.sum((loadings * metric[..., None])**2) / weighted_total),
            "native_variance_retention_baseline": float(np.sum(baseline_loading**2) / native_total),
            "native_variance_retention_new": float(np.sum(loadings**2) / native_total),
            "probability_mass_min": float(np.min(metric**2)), "probability_mass_max": float(np.max(metric**2)),
            "p_drift_captured_energy_fraction": float(captured), "p_drift_weighted_residual_norm": float(residual),
            "q_drift_maximum_absolute": float(np.max(np.abs(q_drift))),
            "decoded_variance_and_forecast_mse": "not yet evaluated; innovation retention is not forecast calibration"}
        energies = np.sum(loadings**2, axis=(0, 1))
        arrays["cumulative_explained_variance"] = np.cumsum(energies) / native_total
        metadata = dict(baseline.metadata, observation_fit=weighted_fit.report, probability_pca_ablation=protocol,
            latent_initialization=LATENT_INITIALIZATION,
            p_drift_captured_energy_fraction=float(captured), p_drift_weighted_residual_norm=float(residual),
            explained_variance_fraction=diagnostics["native_variance_retention_new"],
            current_drift_finite=True, dynamic_no_arbitrage_audited=False)
        metadata.pop("identity", None)
        metadata["identity"] = content_identity(metadata, arrays)
        save_archive(output / "dynamics.npz", metadata, arrays)
        if load_dynamics(output / "dynamics.npz").identity != metadata["identity"]:
            raise ValueError("Weighted inference archive did not round-trip with its original identity")
        _json(output / "summary.json", dict(diagnostics, observation_fit=weighted_fit.report,
            model_identity=metadata["identity"], elapsed_seconds=time.monotonic() - started))
        progress.update({"phase": "weighted_dynamics_complete", "completed": 1, "total": 1, **diagnostics})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if not Path(os.environ["QUANTDATA_ROOT"]).is_dir():
        raise ValueError("QUANTDATA_ROOT is unavailable")
    run(args.representation, args.baseline, args.output, previous_path=args.previous)


if __name__ == "__main__":
    main()
