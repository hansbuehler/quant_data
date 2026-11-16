"""Causal adapters for the original daily25 JAX trainer and front-spot model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.ndimage import gaussian_filter
from scipy.special import ndtri

from quantdata.biz_calendar import get_business_days
from quantdata.hedging.model_audit.legacy_rolling import Progress, market_from_cache
from quantdata.hedging.model_audit.rolling_dysanos import window_dates
from quantdata.hedging.model_audit.rolling_versions import INITIALIZATION_POLICY, origins
from quantdata.hedging.model_audit.run import _hash, _json


def prepare(root: Path, reference: Path, *, origin: str) -> Path:
    """Use the original immutable quote schema, restricted to this origin's window."""
    from quantdata.dysanos.flex.sanos import make_supports_from_total_vols
    from quantdata.sdlv.daily.train import _atomic_write_npz, _geometry

    origins(reference).index(origin)
    dates = window_dates(get_business_days(date(2020, 1, 1), date.fromisoformat(origin)), date.fromisoformat(origin))
    directory = root / f"fits/gromm1/{origin}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "market.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            if saved["dates"].astype(str).tolist() != list(map(str, dates)):
                raise ValueError("Saved GROMM input window changed")
        return destination
    geometry = _geometry(200)
    market = [market_from_cache(day, geometry) for day in dates]
    chunk = 512
    counts = np.asarray([day.num_quotes for day in market], dtype=np.int32)
    capacity = chunk * int(np.ceil(counts.max() / chunk))
    shape = (len(dates), capacity)
    data = {name: np.full(shape, 1. if name in ("strikes", "asks") else 0., dtype=np.float32)
        for name in ("strikes", "bids", "asks", "mids", "alpha", "business_dtes")}
    data.update({name: np.zeros(shape, dtype=np.int32) for name in ("lower", "upper")})
    data.update({name: np.zeros(shape, dtype=bool) for name in ("is_call", "before_first", "mask")})
    supports, initial = [], []
    times = np.asarray(geometry.model_expiries_y)
    for index, day in enumerate(market):
        count = day.num_quotes
        upper_raw = np.searchsorted(times, day.expiries_y[day.expiry_index], side="left")
        upper = np.minimum(upper_raw, len(times) - 1)
        lower = np.maximum(upper_raw - 1, 0)
        previous = np.where(upper_raw == 0, 0., times[lower])
        interval = times[upper] - previous
        alpha = np.divide(day.expiries_y[day.expiry_index] - previous, interval,
            out=np.ones_like(interval), where=interval > 0.)
        values = {"strikes": day.strikes, "bids": day.bids, "asks": day.asks,
            "mids": .5 * (day.bids + day.asks), "is_call": day.is_call, "lower": lower, "upper": upper,
            "before_first": upper_raw == 0, "business_dtes": day.quote_business_dtes,
            "alpha": np.clip(alpha, 0., 1.)}
        for name, value in values.items():
            data[name][index, :count] = value
        data["mask"][index, :count] = True
        support = make_supports_from_total_vols(day.model_atm_total_vols, geometry=geometry)
        if not np.allclose(support, support[:1], atol=1e-12, rtol=0.):
            raise ValueError("Original GROMM requires shared strikes")
        supports.append(support[0])
        annual = day.model_atm_total_vols / np.sqrt(times)
        initial.append(np.log(np.maximum(annual[:, None]**2 - .01**2, 1e-8)) * np.ones((1, 200)))
    _atomic_write_npz({**data, "schema": np.asarray("daily25-jax-market-v1"), "dates": np.asarray(list(map(str, dates))),
        "supports": np.asarray(supports, np.float32), "raw_initial": np.asarray(initial, np.float32),
        "model_times_y": times.astype(np.float32), "quote_counts": counts,
        "quote_chunk_size": np.asarray(chunk, np.int32), "fit_weight": np.asarray(.05, np.float32),
        "backbone_annual_vol": np.asarray(.05, np.float32)}, destination)
    _json(directory / "PROTOCOL.json", {"origin": origin, "dates": list(map(str, dates)),
        "implementation": "original quantdata.sdlv.daily.train and front_spot, not gromm2 or standalone models",
        "initialization": INITIALIZATION_POLICY, "native_geometry": "25 daily expiries by200, fixed5percent Black background",
        "source_smoothness": .1, "input_sha256": _hash(destination), "innovation_pca": "probability weighted,16 total Brownian factors",
        "window": "250 observed closes including origin,249 exact daily transitions,matching the current DYSANOS comparison",
        "market_fingerprints": [day.training_fingerprint for day in market], "source_sha256": _hash(Path(__file__))})
    return destination


def calibration(dataset: Path, checkpoint: Path, observations: Path):
    """Re-estimate the old fixed-contract law without its global archive defaults."""
    import jax
    import jax.numpy as jnp

    from quantdata.sdlv.daily.front_spot.calibration import _one_state_risk_neutral_drift
    from quantdata.sdlv.daily.front_spot.estimation import (
        AlignedDlvHistory,
        _front_density,
        _front_spot_cdf,
        _low_rank_loadings,
        decode_dlvsq,
        fit_achievable_physical_drift,
        fit_joint_innovations,
    )

    jax.config.update("jax_enable_x64", True)
    with np.load(dataset, allow_pickle=False) as saved:
        dates, supports = saved["dates"].astype(str), saved["supports"].astype(float)
    with np.load(checkpoint, allow_pickle=False) as saved:
        raw = saved["mean"][None] + np.einsum("dr,rtk->dtk", saved["codes"], saved["loadings"])
    dlvsq = decode_dlvsq(raw)
    frame = pd.read_parquet(observations)
    frame.index = pd.to_datetime(frame.date).dt.date
    selected = frame.loc[[date.fromisoformat(value) for value in dates]]
    if np.any(np.diff(selected.session_index.to_numpy()) != 1):
        raise ValueError("Original GROMM window must contain only exact daily pairs")
    gross = 1. + selected.normalized_spot_return_into_date.to_numpy(float)[1:]
    before = np.log(dlvsq[:-1, 1:])
    after, cdf = np.empty_like(before), np.empty(len(before))
    for index in range(len(before)):
        query = supports[index, 1:-1] / gross[index]
        for expiry in range(24):
            after[index, expiry] = np.interp(query, supports[index + 1, 1:-1], np.log(dlvsq[index + 1, expiry]))
        cdf[index] = _front_spot_cdf(gross[index], supports[index], _front_density(supports[index], dlvsq[index, 0]))
    epsilon = np.finfo(float).eps
    history = AlignedDlvHistory(dates=tuple(dates), origin_indices=np.arange(len(before)),
        endpoint_indices=np.arange(1, len(dates)), origin_log_dlvsq=before, endpoint_log_dlvsq=after,
        spot_gaussian_scores=ndtri(np.clip(cdf, epsilon, 1. - epsilon)), front_cdf_values=cdf)
    innovations = fit_joint_innovations(history, factor_rank=16, smoothing_sigma=(1., 2.))

    from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower

    density_function = jax.jit(jax.vmap(lambda support, values: decode_density_tower(support, values, dt=1. / 252.)))
    densities = np.concatenate([np.asarray(density_function(jnp.asarray(supports[start:start + 16]),
        jnp.asarray(dlvsq[start:start + 16]))) for start in range(0, len(before), 16)])[:len(before)]
    if not np.isfinite(densities).all() or np.min(densities) < -1e-12:
        raise ValueError("Original GROMM decoder produced invalid historical probabilities")
    metric = np.sqrt(np.maximum(densities[:, 1:, 1:-1], 0.).mean(axis=0))
    residual = gaussian_filter(innovations.centered_residuals.reshape(before.shape), sigma=(0., 1., 2.), mode="nearest")
    residual -= residual.mean(axis=0)
    flat = residual.reshape(len(before), -1)
    spot = (history.spot_gaussian_scores - innovations.spot_score_mean) / innovations.spot_score_std
    linked = np.mean(flat * spot[:, None], axis=0)
    orthogonal = flat - spot[:, None] * linked
    weighted, eigenvalues, scores = _low_rank_loadings(orthogonal * metric.ravel(), 15)
    native_loading = np.divide(weighted, metric.ravel()[:, None], out=np.zeros_like(weighted), where=metric.ravel()[:, None] > 0.)
    daily = np.column_stack((linked, native_loading))
    innovations = replace(innovations, annualized_loadings=daily.reshape(24, 200, 16) * np.sqrt(252.),
        daily_eigenvalues=eigenvalues, factor_scores=np.column_stack((spot, scores)),
        explained_variance_fraction=float(np.sum(daily**2) / np.mean(np.sum(flat**2, axis=1))))
    drift_function = jax.jit(jax.vmap(lambda support, values: _one_state_risk_neutral_drift(
        support, values, jnp.asarray(innovations.annualized_loadings), 1. / 252.)))
    relative = np.concatenate([np.asarray(drift_function(jnp.asarray(supports[start:start + 8]),
        jnp.asarray(dlvsq[start:start + 8]))[1]) for start in range(0, len(before), 8)])[:len(before)]
    q_drift = relative - .5 * np.sum(innovations.annualized_loadings**2, axis=-1)
    interior = np.maximum(densities[:, 1:, 1:-1], 0.)
    if np.any(interior.sum(axis=-1) <= 0.):
        raise ValueError("Every GROMM expiry needs positive interior probability mass")
    weights = interior / interior.sum(axis=-1, keepdims=True)
    physical = fit_achievable_physical_drift(history, innovations, q_drift, weights)
    return SimpleNamespace(history=history, innovations=innovations, physical=physical,
        supports=supports, dlvsq=dlvsq, coordinate_metric=metric, q_log_drift=q_drift)


def run(root: Path, reference: Path, *, origin: str, prepare_only: bool = False) -> None:
    """The original 300+700 affine recipe is fresh at every rolling origin."""
    from quantdata.sdlv.daily.train import fit_model

    directory = root / f"fits/gromm1/{origin}"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / ".training.lock"), Progress(directory, family="GROMM v1 P/Q", origin=origin) as progress:
        progress.phase = "prepare causal original-schema market inputs"
        dataset = prepare(root, reference, origin=origin)
        if prepare_only:
            return
        for phase, epochs, learning_rate, offset in (("initial", 300, .02, 0), ("continuation", 700, .005, 300)):
            output = directory / phase
            if (output / "report.json").exists():
                continue
            progress.phase = f"original affine200 {phase} market refinement"
            initial_checkpoint = None if phase == "initial" else directory / "initial/checkpoint.npz"
            fit_model(dataset=dataset, output_directory=output, mode="affine", rank=200,
                epochs=epochs, learning_rate=learning_rate, minimum_learning_rate_fraction=.05,
                cap_penalty_weight=1., seed=1729, checkpoint_every=5, initial_checkpoint=initial_checkpoint,
                epoch_offset=offset, gradient_tolerance=1e-5)
        report = json.loads((directory / "continuation/report.json").read_text())
        inside = float(report["final_statistics"]["percent_inside_bid_ask"])
        last_gradient = float(report["history"][-1]["gradient_l2_norm"])
        review = {"inside_bid_ask_percent": inside, "minimum_inside_bid_ask_percent": 80.,
            "last_gradient_l2_norm": last_gradient, "gradient_tolerance_reached": bool(report["stopped_at_gradient_tolerance"]),
            "original_budget_finished": not bool(report["stopped_at_gradient_tolerance"]),
            "acceptance": "surface-fit accuracy>=80percent, not a claim that budget exhaustion proves gradient convergence",
            "fit_accuracy_accepted": bool(np.isfinite(inside) and np.isfinite(last_gradient) and inside >= 80.)}
        _json(directory / "MARKET_FIT_REVIEW.json", review)
        if not review["fit_accuracy_accepted"]:
            raise RuntimeError(f"Original GROMM market fit requires review: {inside:.3f}% inside bid/ask, gradient={last_gradient:.6g}")
        progress.phase = "original fixed-contract probability PCA and compatible P/Q drift"
        fitted = calibration(dataset, directory / "continuation/checkpoint.npz", root / "observations.parquet")
        np.savez_compressed(directory / "dynamics.npz", support=fitted.supports[-1], dlvsq=fitted.dlvsq[-1],
            annualized_loadings=fitted.innovations.annualized_loadings,
            physical_market_price_of_risk=fitted.physical.constant_market_price_of_risk,
            conditional_intercept=fitted.innovations.conditional_intercept,
            conditional_slope=fitted.innovations.conditional_slope, coordinate_metric=fitted.coordinate_metric,
            q_log_drift=fitted.q_log_drift, dates=np.asarray(fitted.history.dates))
        progress.phase = "matched one-day P/Q observable evaluation"
        score(root, reference, origin=origin, fitted=fitted)
        _json(directory / "COMPLETE.json", {"origin": origin, "model": "original fixed-grid GROMM v1",
            "measures": ["P", "Q"], "initialization": INITIALIZATION_POLICY,
            "dynamics_sha256": _hash(directory / "dynamics.npz"),
            "forecast_statistics": "one-day P/Q observable evaluation saved; remaining diagnostics pending",
            "same_origin_restart": "original trainer checkpoints contain parameters only, not optimizer moments"})
        progress.phase = "complete"
        progress.completed = 4
        progress.emit()


def score(root: Path, reference: Path, *, origin: str, fitted, paths: int = 1000) -> None:
    """Use the original eight-substep bridge, common repricer and existing reference."""
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    import jax
    import jax.numpy as jnp

    from quantdata.hedging.model_audit.gromm import observable_surface
    from quantdata.hedging.model_audit.metrics import covariance
    from quantdata.hedging.model_audit.run import QUANTILES
    from quantdata.sdlv.daily.front_spot.jax_model import (
        RiskNeutralState,
        decode_density_tower,
        one_day_physical_substepped,
        one_day_risk_neutral_substepped,
    )

    support = jnp.asarray(fitted.supports[-1])
    values = jnp.asarray(fitted.dlvsq[-1])
    tower = decode_density_tower(support, values, dt=1. / 252.)
    state = RiskNeutralState(tower[0], values[1:], jnp.asarray(1.))
    market = SimpleNamespace(business_dtes=np.arange(1, 26), supports=np.broadcast_to(fitted.supports[-1], (25, 202)))
    current, _ = observable_surface(market, np.asarray(tower)[None], np.ones(1), np.ones(1), elapsed=0)
    remaining = SimpleNamespace(business_dtes=np.arange(2, 26), supports=np.broadcast_to(fitted.supports[-1], (24, 202)))
    random = np.random.default_rng(20_261_112)
    normals = random.standard_normal((8, paths, 16))
    background = random.standard_normal(paths)
    with np.load(reference / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"]
    common = json.loads((reference / f"daily/FuNVol/{origin}.json").read_text())
    for measure in ("P", "Q"):
        if measure == "P":
            operation = jax.jit(lambda shocks, black: one_day_physical_substepped(support, state,
                jnp.asarray(fitted.innovations.annualized_loadings),
                jnp.asarray(fitted.physical.constant_market_price_of_risk), shocks, black))
        else:
            operation = jax.jit(lambda shocks, black: one_day_risk_neutral_substepped(support, state,
                jnp.asarray(fitted.innovations.annualized_loadings), shocks, black))
        returns, surfaces, validities = [], [], []
        for start in range(0, paths, 32):
            result = operation(jnp.asarray(normals[:, start:start + 32]), jnp.asarray(background[start:start + 32]))
            spot = np.asarray(result.spot)
            surface, valid = observable_surface(remaining, np.asarray(result.density_tower),
                np.asarray(result.state.background_level), spot, elapsed=1)
            surfaces.append(surface.reshape(len(spot), -1))
            validities.append(valid.all(axis=(1, 2)) & np.isfinite(spot) & (spot > 0.))
            returns.append(np.log(spot))
            print(f"GROMM v1 {measure} forecast origin={origin} completed={min(start+32,paths)}/{paths}", flush=True)
        valid = np.concatenate(validities)
        if valid.sum() < 4:
            raise ValueError("Original GROMM has fewer than four valid forecast paths")
        following = np.concatenate(surfaces)[valid]
        changes = following - current.reshape(-1)
        spot_returns = np.concatenate(returns)[valid]
        targets = np.c_[changes @ basis, spot_returns]
        joint = np.c_[spot_returns, changes @ basis]
        quantiles = np.quantile(targets, QUANTILES, axis=0)
        record = {name: common[name] for name in ("origin_date", "endpoint", "actual_change", "actual_spot", "actual_targets")}
        record.update(model=f"GROMM v1-{measure}", status="scored", scenario_count=paths, valid_scenario_count=int(valid.sum()),
            surface_mean=changes.mean(axis=0).tolist(), surface_covariance=covariance(changes, ddof=0).tolist(),
            joint_mean=joint.mean(axis=0).tolist(), joint_covariance=covariance(joint, ddof=0).tolist(),
            quantiles=quantiles.tolist(), es_lower=[float(targets[targets[:, column] <= quantiles[1, column], column].mean())
                for column in range(4)], scenario_covariance_ddof=0, invalid_surface_rate=float(1. - valid.mean()),
            extreme_surface_rate=float(np.any(following > np.log(3.), axis=1).mean()),
            forecast_rmse=float(np.sqrt(np.mean((changes.mean(axis=0) - np.asarray(common["actual_change"]))**2))),
            diagnostics={"sampling": "original front_spot eight-substep bridge, paired P/Q normals", "seed": 20_261_112,
                "dynamics_path": str(root / f"fits/gromm1/{origin}/dynamics.npz")},
            bidask={"status": "not_scored", "reason": "Matched quote reconstruction audit pending"})
        _json(root / f"daily/GROMM_v1-{measure}/{origin}.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not Path(os.environ["QUANTDATA_ROOT"]).is_dir():
        raise ValueError("Configured data root is missing")
    run(args.root, args.reference, origin=args.origin, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
