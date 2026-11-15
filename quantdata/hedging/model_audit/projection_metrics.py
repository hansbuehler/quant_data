"""Causal affine and observable-tangent projection comparisons on retained daily states."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields, replace
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from filelock import FileLock

from quantdata.hedge2.batch_pricing import decode_batch
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import DynamicsConfig, FittedDynamics, fit_dynamics
from quantdata.hedge2.rolling import load_dynamics, save_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.density_weighted import _load, density_metric
from quantdata.hedging.model_audit.grid_shape import _score
from quantdata.hedging.model_audit.metrics import factor_innovation_covariance, linear_stability
from quantdata.hedging.model_audit.run import _hash, _json, _summarize
from quantdata.hedging.model_audit.tail import _training_pairs

METRICS = ("itm_premium", "straddle_spot", "straddle_sqrt_time", "probability_changes")


def affine_projection(levels: np.ndarray, weights: np.ndarray, *, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted least-squares affine reconstruction; zero-weight coordinates remain frozen."""
    if (levels.ndim != 2 or weights.shape != (levels.shape[1],) or np.any(weights < 0.)
            or not np.isfinite(levels).all() or not np.isfinite(weights).all() or rank <= 0):
        raise ValueError("Affine projection needs finite levels, nonnegative coordinate weights and positive rank")
    scale = np.sqrt(weights)
    centered = (levels - levels.mean(axis=0)) * scale
    values, vectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(values)[::-1]
    order = order[values[order] > np.finfo(float).eps * max(float(values.max()), 1.)][:rank]
    if not len(order):
        raise ValueError("No nonzero weighted affine directions")
    directions = centered.T @ vectors[:, order] / np.sqrt(values[order])
    decoder = np.divide(directions, scale[:, None], out=np.zeros_like(directions), where=scale[:, None] > 0.)
    return scale[:, None] * directions, decoder, float(values[order].sum() / values.clip(min=0.).sum())


def metric_projection(changes: np.ndarray, span_encoder: np.ndarray, image_jacobian: np.ndarray,
                      *, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA of tangent observable changes, with native preimages and an exact linear encoder."""
    if (changes.ndim != 2 or span_encoder.shape[0] != changes.shape[1]
            or image_jacobian.ndim != 2 or image_jacobian.shape[0] != span_encoder.shape[1]
            or not all(np.isfinite(value).all() for value in (changes, span_encoder, image_jacobian))):
        raise ValueError("Metric images and native changes must have aligned finite coordinates")
    centered = changes - changes.mean(axis=0)
    images = (centered @ span_encoder) @ image_jacobian
    values, vectors = np.linalg.eigh(images @ images.T)
    order = np.argsort(values)[::-1][:rank]
    if len(order) != rank or values[order[-1]] <= np.finfo(float).eps * max(float(values.max()), 1.):
        raise ValueError("Observable projection has insufficient numerical rank")
    singular = np.sqrt(values[order])
    left = vectors[:, order]
    right = images.T @ left / singular
    normalization = np.sqrt(len(changes) - 1.)
    encoder = span_encoder @ (image_jacobian @ right) * (normalization / singular)
    decoder = centered.T @ left / normalization
    np.testing.assert_allclose(encoder.T @ decoder, np.eye(rank), atol=2e-7, rtol=2e-7)
    return encoder, decoder, float(values[order].sum() / values.clip(min=0.).sum())


def observable_measures(state: DailyState, natives: np.ndarray, *, strikes: np.ndarray,
                        business_dtes: np.ndarray, origin_support: np.ndarray,
                        origin_premium: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Surface partials at fixed origin spot, tenors and cash strikes; no time roll."""
    decoded = decode_batch(state, natives)
    times = np.asarray(business_dtes) / 252.
    query_days = np.repeat(business_dtes, strikes.shape[1])
    query_strikes = strikes.reshape(-1)
    calls = decoded.europeans(business_dtes=query_days, strikes=query_strikes)
    puts = calls + query_strikes - 1.
    itm = np.where(query_strikes < 1., calls, puts)
    straddles = calls + puts
    masses = []
    for tenor in business_dtes:
        prices = decoded.europeans(business_dtes=np.full(len(origin_support), tenor), strikes=origin_support)
        slopes = np.diff(prices, axis=1) / np.diff(origin_support)
        probabilities = np.c_[1. + slopes[:, :1], np.diff(slopes, axis=1), -slopes[:, -1:]]
        if probabilities.min() < -1e-8 or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-8, rtol=0.):
            raise ValueError("Origin-grid call slopes do not define nonnegative unit probability masses")
        masses.append(probabilities)
    return {"itm_premium": itm / origin_premium,
        "straddle_spot": straddles,
        "straddle_sqrt_time": straddles / np.repeat(np.sqrt(times), strikes.shape[1]),
        "probability_changes": np.concatenate(masses, axis=1)}, {
        "lower_clips": int(decoded.lower_clips.sum()), "upper_clips": int(decoded.upper_clips.sum()),
        "minimum_probability": float(min(value.min() for value in masses))}


def fit_projected(*, levels: np.ndarray, dates: tuple[date, ...], spots: np.ndarray, sessions: np.ndarray,
                  encoder: np.ndarray, decoder: np.ndarray, config: DynamicsConfig,
                  previous: tuple[FittedDynamics, np.ndarray] | None = None) -> FittedDynamics:
    """Estimate the same spot/PCA/VAR law in the supplied affine coordinates."""
    prior = None
    if previous is not None:
        old_fit, old_decoder = previous
        mapped = encoder.T @ old_decoder @ old_fit.directions
        norms = np.linalg.norm(mapped, axis=0)
        if mapped.shape != (encoder.shape[1], config.rank) or np.any(norms <= 0.) or not np.isfinite(mapped).all():
            raise ValueError("Previous rolling factors cannot be mapped into the new projection")
        prior = replace(old_fit, directions=mapped / norms)
    return fit_dynamics(states=levels @ encoder, dates=dates, spot_returns_into_date=spots,
        session_indices=sessions, as_of_date=dates[-1], config=config, previous=prior)


def projected_scenarios(current: np.ndarray, encoder: np.ndarray, decoder: np.ndarray,
                        fit: FittedDynamics, *, native_spot_loading: np.ndarray | None = None) -> np.ndarray:
    """Keep the unmodelled current-state component anchored, not replaced by the affine mean."""
    change = (fit.forecast_state_change(current @ encoder)
        + fit.residuals[:, :1] * fit.spot_state_loading + fit.residuals[:, 1:] @ fit.directions.T)
    following = current + change @ decoder.T
    if native_spot_loading is not None:
        perpendicular = native_spot_loading - (native_spot_loading @ encoder) @ decoder.T
        following += fit.residuals[:, :1] * perpendicular
    return following


def _save(directory: Path, fit: FittedDynamics, encoder: np.ndarray, decoder: np.ndarray,
          native_spot_loading: np.ndarray) -> None:
    save_dynamics(fit, directory / "dynamics.npz")
    temporary = directory / "projection.partial.npz"
    np.savez_compressed(temporary, encoder=encoder, decoder=decoder, native_spot_loading=native_spot_loading)
    temporary.replace(directory / "projection.npz")


def run(root: Path, source: Path, weighted: Path, output: Path) -> dict:
    directory = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    raw_config = json.loads((directory / "config.json").read_text())
    config = DlvConfig(**{field.name: raw_config[field.name] for field in fields(DlvConfig) if field.name in raw_config})
    origins = json.loads((source / "observable_reference.json").read_text())["dates"]
    if len(origins) != 44:
        raise ValueError("The controlled experiment requires the same 44 forecast origins")
    with np.load(source / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        observable_basis = saved["basis"]
    pairs = _training_pairs(root)
    protocol = {"schema": "affine-and-observable-tangent-v1", "origins": origins, "rolling_pairs": 252,
        "affine_rank": 200, "innovation_rank": 16, "metrics": list(METRICS), "source": str(source),
        "weighted_comparator": str(weighted), "source_config_sha256": _hash(directory / "config.json"),
        "source_code_sha256": _hash(Path(__file__)), "variants": ["ordinary", "orthogonal"],
        "time_weights": "equal, unchanged from the matched audit", "metric_weights": "causal origin-window average node masses",
        "option_grid": "all nine native tenors and all 200 native interior cash strikes, ITM sides; no hedge-book screen",
        "surface_partial": "fixed origin spot, tenors and cash strikes; separate learned spot loading",
        "density_metric": "absolute mass changes on origin support via call slopes, endpoint masses include tails",
        "tangent_step": 1e-4, "tangent_check": "repeat at half step and retain relative Jacobian change",
        "estimator": "separate affine/tangent projection and VAR, not joint nonlinear JAX calibration",
        "decoder_changed": False, "future_data_used_in_fit": False,
        "full_native_spot_loading_retained": True,
        "reconstruction_policy": "current component outside the affine span stays anchored",
        "source_smoothness": .1, "observable_reference_smoothness": .25}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "PROTOCOL.json"
    if destination.exists() and json.loads(destination.read_text()) != protocol:
        raise ValueError("The output directory belongs to a different projection protocol")
    _json(destination, protocol)
    used_days = set()

    @lru_cache(maxsize=600)
    def daily(day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        used_days.add(day)
        with np.load(directory / f"days/{day}.npz", allow_pickle=False) as saved:
            return saved["raw_u"], saved["decoded_densities"], saved["support_strikes"]

    records, previous = {}, {}
    started = time.monotonic()
    total = (len(origins) + 1) * 2 * (len(METRICS) + 2)
    completed = 0
    for text in ["2024-03-28", *origins]:
        day = date.fromisoformat(text)
        window = [pair for pair in pairs if pair[1] <= day][-252:]
        selected_days = tuple(sorted({entry for pair in window for entry in pair[:2]}))
        session_lookup = {entry: pair[2] - offset for pair in window for entry, offset in ((pair[0], 1), (pair[1], 0))}
        sessions = np.asarray([session_lookup[entry] for entry in selected_days])
        date_index = {entry: index for index, entry in enumerate(selected_days)}
        levels = np.stack([daily(entry)[0] for entry in selected_days])
        weights = density_metric(np.stack([daily(pair[0])[1] for pair in window]))
        native, _, support = daily(day)
        state = DailyState(day=day, native=native, reference_support=support, config=config)
        changes = np.stack([daily(pair[1])[0] - daily(pair[0])[0] for pair in window])
        changes -= changes.mean(axis=0)
        base_pricer = state.decode().pricer
        strikes = np.broadcast_to(base_pricer.support[1:-1], (len(config.business_dtes), config.num_inner_strikes))
        days = np.asarray(config.business_dtes)
        premiums = np.concatenate([base_pricer.european_at(expiry_y=tenor / 252., strikes=row, is_call=row < 1.)
                                  for tenor, row in zip(days, strikes, strict=True)])
        if np.any(premiums <= 0.) or not np.isfinite(premiums).all():
            raise ValueError("Every ITM normalization premium must be finite and strictly positive")
        affine_maps = {}
        for label, metric in (("affine200_unweighted", np.ones_like(weights)), ("affine200_probability", weights)):
            affine_maps[label] = affine_projection(levels, metric, rank=200)
        for variant in ("ordinary", "orthogonal"):
            mixed = _load(weighted / f"fits/{variant}/{text}")
            pair_spots = mixed.weighted_fit.residuals[:, 0]
            spots = np.full(len(selected_days), np.nan)
            spots[[date_index[pair[1]] for pair in window]] = pair_spots
            beta = pair_spots @ changes / (pair_spots @ pair_spots) if variant == "orthogonal" else np.zeros(len(native))
            residual = changes - pair_spots[:, None] * beta
            encoder_span, decoder_span, _ = affine_projection(residual, weights, rank=251)
            steps = 1e-4 / np.maximum(np.max(np.abs(decoder_span), axis=0), 1.)
            image_maps, tangent_checks, guard_totals = {}, {}, {}
            arguments = dict(strikes=strikes, business_dtes=days, origin_support=base_pricer.support, origin_premium=premiums)
            for fraction in (1., .5):
                bumps = decoder_span.T * (steps * fraction)[:, None]
                values, guards = observable_measures(state, np.r_[native + bumps, native - bumps], **arguments)
                guard_totals[str(fraction)] = guards
                images = {name: (value[:len(bumps)] - value[len(bumps):]) / (2. * steps[:, None] * fraction)
                          for name, value in values.items()}
                if fraction == 1.:
                    image_maps = images
                else:
                    tangent_checks = {name: float(np.linalg.norm(images[name] - image_maps[name])
                        / max(np.linalg.norm(images[name]), np.finfo(float).tiny)) for name in METRICS}
                    image_maps = images
            maps = dict(affine_maps)
            for metric in METRICS:
                maps[metric] = metric_projection(residual, encoder_span, image_maps[metric], rank=16)
            for label, (encoder, decoder, coverage) in maps.items():
                name = f"DYSANOS {variant} {label}"
                records.setdefault(name, [])
                checkpoint = output / f"fits/{variant}/{label}/{text}"
                if (checkpoint / "projection.npz").exists():
                    fit = load_dynamics(checkpoint / "dynamics.npz")
                    with np.load(checkpoint / "projection.npz", allow_pickle=False) as saved:
                        encoder, decoder = saved["encoder"], saved["decoder"]
                else:
                    fit = fit_projected(levels=levels, dates=selected_days, spots=spots,
                        sessions=sessions, encoder=encoder, decoder=decoder,
                        config=DynamicsConfig(rank=16, spot_orthogonal=variant == "orthogonal"), previous=previous.get(name))
                    _save(checkpoint, fit, encoder, decoder, beta)
                previous[name] = (fit, decoder)
                completed += 1
                if text in origins:
                    baseline_name = f"DYSANOS {'spot-orthogonal' if variant == 'orthogonal' else 'ordinary'} (16 factors) density-weighted"
                    baseline = json.loads((weighted / "daily" / baseline_name.replace(" ", "_") / f"{text}.json").read_text())
                    path = output / f"daily/{name.replace(' ', '_')}/{text}.json"
                    if path.exists():
                        record = json.loads(path.read_text())
                    else:
                        following = projected_scenarios(native, encoder, decoder, fit, native_spot_loading=beta)
                        diagnostics = {"training_start": str(fit.training_start), "training_end": str(fit.training_end),
                            "initialisation_date": str(fit.initialisation_date), "projection_metric": label,
                            "projection_rank": encoder.shape[1], "projection_coverage": coverage,
                            "encoder_decoder_identity_error": float(np.max(np.abs(encoder.T @ decoder - np.eye(encoder.shape[1])))),
                            "tangent_relative_half_step_error": tangent_checks.get(label), "tangent_decoder_guards": guard_totals,
                            "raw_spectral_radius": fit.unconstrained_spectral_radius,
                            "stability": linear_stability(fit.ar_matrix, factor_innovation_covariance(
                                fit.joint_innovation_covariance, fit.spot_state_loading @ fit.directions)),
                            "estimator": protocol["estimator"]}
                        try:
                            record = _score(state, following, np.log1p(fit.residuals[:, 0]), baseline,
                                label=name, basis=observable_basis, diagnostics=diagnostics)
                        except (ValueError, RuntimeError, FloatingPointError) as error:
                            record = {"model": name, "origin_date": text, "status": "error", "reason": str(error)}
                        _json(path, record)
                    records[name].append(record)
                elapsed = time.monotonic() - started
                print(f"projection {text} {variant} {label} completed={completed}/{total} elapsed={elapsed:.1f}s "
                      f"eta={elapsed * (total - completed) / completed:.1f}s", flush=True)
        _write_results(output, records, observable_basis)
    source_files = [directory / "config.json", weighted / "PROTOCOL.json", source / "observable_reference.json",
                    source / "frozen_observable_pca.npz", Path(__file__)]
    source_files.extend(directory / f"days/{entry}.npz" for entry in sorted(used_days))
    _json(output / "INPUTS.json", [{"path": str(path), "sha256": _hash(path)} for path in source_files])
    return {"output": str(output), "models": len(records)}


def _write_results(output: Path, records: dict, basis: np.ndarray) -> None:
    summaries = {name: _summarize(values, basis) for name, values in records.items()}
    _json(output / "summary.json", summaries)
    lines = ["# Affine and observable-tangent projection comparison", "",
        "Same 44 origins, 252 causal daily transitions and 16 innovation factors. This is not joint JAX calibration. "
        "Affine200 variants reconstruct levels before fitting dynamics; observable variants choose native change directions "
        "using the local origin-date option/density Jacobian. No volatility rescaling or new decoder is introduced.", "",
        "| Model | Scored | Conditional trace / observed | Pooled trace / observed | Covariance error | Forecast RMSE | Variance / forecast MSE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        count = summary["surface_days"]
        if count < 2:
            lines.append(f"| {name} | {count}/44 | — | — | — | — | — |")
        else:
            lines.append(f"| {name} | {count}/44 | {summary['conditional_covariance']['trace_ratio']:.3f} | "
                f"{summary['pooled_covariance']['trace_ratio']:.3f} | {summary['pooled_covariance']['cov_frob_rel']:.3f} | "
                f"{summary['forecast_rmse']:.4f} | "
                f"{summary['forecast_variance_calibration']['conditional_variance_over_forecast_mse']:.3f} |")
    lines.extend(["", "Forecasts are scored on the existing 49 common floating log-IV coordinates. "
        "Density inputs use smoothness 0.1 and the common market reference uses 0.25, unchanged from the prior audit. "
        "Weights are grid probability masses, not density heights or a measure change. "
        "Density changes are absolute differences, not division by tiny probabilities. "
        "Tangent losses and affine reconstruction are not evidence of predictive variance calibration. "
        "Per-origin Jacobian half-step discrepancies and every decoding clip are retained. "
        "This first comparison does not rerun recursive-path or quote tests."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


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
