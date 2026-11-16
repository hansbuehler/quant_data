"""Evaluate the actual joint affine-200 Gaussian paper law, without PCA refitting."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm, solve_discrete_lyapunov

from quantdata.biz_calendar import NMS
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.presentation_hedging import load_presentation_model
from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity
from quantdata.hedging.model_audit.run import COMMON_INDICES, COMMON_NMS, GRID, _hash, _json


def daily_law(generator: np.ndarray, weekly_factor: np.ndarray, *, sessions: int = 5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Factor a row-state weekly law, rejecting a non-PSD daily embedding."""
    generator, weekly_factor = np.asarray(generator, float), np.asarray(weekly_factor, float)
    if (generator.ndim != 2 or generator.shape[0] != generator.shape[1]
            or weekly_factor.shape != generator.shape or not isinstance(sessions, int) or sessions < 1
            or not np.isfinite(generator).all() or not np.isfinite(weekly_factor).all()):
        raise ValueError("Expected finite square generator/factor and a positive integer session count")
    weekly = expm(-generator)
    daily = expm(-generator / sessions)
    if np.max(np.abs(np.linalg.eigvals(weekly))) >= 1.:
        raise ValueError("The saved weekly law is not stationary")
    weekly_cov = weekly_factor @ weekly_factor.T
    stationary = solve_discrete_lyapunov(weekly.T, weekly_cov)
    covariance = stationary - daily.T @ stationary @ daily
    covariance = .5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() <= 0.:
        raise ValueError(f"Weekly law has no positive-definite equal daily embedding: min eigenvalue={eigenvalues.min()}")
    factor = np.linalg.cholesky(covariance)
    accumulated, propagation = np.zeros_like(covariance), np.eye(len(daily))
    for _ in range(sessions):
        accumulated += propagation.T @ covariance @ propagation
        propagation = propagation @ daily
    diagnostics = {"state_dimension": len(daily), "daily_min_eigenvalue": float(eigenvalues.min()),
        "weekly_transition_max_error": float(np.max(np.abs(np.linalg.matrix_power(daily, sessions) - weekly))),
        "weekly_covariance_relative_error": float(np.linalg.norm(accumulated - weekly_cov) / np.linalg.norm(weekly_cov)),
        "covariance_projection_applied": False}
    if diagnostics["weekly_transition_max_error"] > 1e-10 or diagnostics["weekly_covariance_relative_error"] > 1e-9:
        raise ValueError(f"Daily embedding does not reproduce the fitted weekly law: {diagnostics}")
    return daily, factor, diagnostics


def observable(model, states: torch.Tensor) -> np.ndarray:
    """Decode floating log-IVs on the same 49 coordinates as the existing audit."""
    with torch.no_grad():
        raw = model.raw_states_from_states(states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        surface = model.sanos_decoder.forward_batch(dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols, eta=parameters.eta)
        times = np.asarray(GRID.expiries_y)
        expiries = torch.as_tensor(np.broadcast_to(times, (len(states), len(times))).copy(), dtype=model.fdtype)
        atm_prices = model.sanos_decoder.price_calls_batch(surface, expiries_y=expiries, strikes=torch.ones_like(expiries)).numpy()
        atm, valid = implied_volatilities_with_validity(expiries_y=expiries.numpy(),
            normalized_strikes=np.ones_like(atm_prices), normalized_prices=atm_prices, is_call=np.ones_like(atm_prices, bool))
        if not valid.all():
            raise ValueError("Paper model has an invalid ATM IV; no path is discarded")
        strikes = NMS.from_nms(COMMON_NMS[None, None, :], times[None, :, None], vol=atm[:, :, None])
        query_times = np.broadcast_to(times[None, :, None], strikes.shape).reshape(len(states), -1).copy()
        strikes = strikes.reshape(len(states), -1)
        prices = model.sanos_decoder.price_calls_batch(surface,
            expiries_y=torch.as_tensor(query_times, dtype=model.fdtype),
            strikes=torch.as_tensor(strikes, dtype=model.fdtype)).numpy()
    vols, valid = implied_volatilities_with_validity(expiries_y=query_times,
        normalized_strikes=strikes, normalized_prices=prices, is_call=np.ones_like(prices, bool))
    if not valid.all():
        raise ValueError("Paper model has an invalid grid IV; no path is discarded")
    return np.log(vols)


def run(checkpoint: Path, targets: Path, reference: Path, output: Path, *, paths: int, origins: list[str] | None) -> None:
    """Saved all-history states are explicitly in-sample, never causal rolling fits."""
    if paths < 4 or paths % 2:
        raise ValueError("Use an even number of at least four antithetic paths")
    output.mkdir(parents=True, exist_ok=True)
    digest = _hash(checkpoint)
    history = torch.load(targets, map_location="cpu", weights_only=False)
    if history["joint_checkpoint_sha256"] != digest:
        raise ValueError("Target history does not authenticate this joint checkpoint")
    _, model = load_presentation_model(checkpoint, device="cpu", precision=64)
    with torch.no_grad():
        transition, factor, embedding = daily_law(model.mean_reversion_generator().numpy(), model.innovation_cholesky().numpy())
    if history["schema"] == "flex-all-history-joint-direct-targets-v1":
        dates, historical_states = history["dates"], history["states"]
    elif history["schema"] == "flex-joint-direct-target-history-v1":
        dates, historical_states = history["train_dates"], history["train_states"]
    else:
        raise ValueError(f"Unsupported authenticated state history: {history['schema']}")
    if len(dates) != model.num_days or not np.allclose(historical_states.numpy(), model.daily_states.detach().numpy()):
        raise ValueError("Saved historical states differ from the joint calibrated states")
    with np.load(reference / "reference_history.npz", allow_pickle=False) as saved:
        reference_dates = saved["dates"].astype(str).tolist()
        common = saved["surfaces"][:, :, COMMON_INDICES].reshape(len(reference_dates), -1)
    selected = json.loads((reference / "observable_reference.json").read_text())["dates"] if origins is None else origins
    protocol = {"schema": "paper-affine200-daily-v1", "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "targets_sha256": _hash(targets), "reference_sha256": _hash(reference / "reference_history.npz"),
        "origins": selected, "paths": paths, "seed": 20_261_112, "embedding": embedding,
        "evaluation_scope": "IN-SAMPLE parameters: saved checkpoint fitted on all 891 dates, including evaluation window",
        "law": "full 200-state joint market-calibrated weekly Gaussian VAR; exact equal daily embedding; no PCA refit",
        "source_sha256": _hash(Path(__file__))}
    if (output / "PROTOCOL.json").exists() and json.loads((output / "PROTOCOL.json").read_text()) != protocol:
        raise ValueError("Output identity changed; use another run directory")
    _json(output / "PROTOCOL.json", protocol)
    equilibrium = model.equilibrium_mean.detach().numpy()
    random = np.random.default_rng(protocol["seed"])
    half = random.standard_normal((paths // 2, model.config.state_size))
    shocks = np.stack((half, -half), axis=1).reshape(paths, -1)
    started = time.monotonic()
    rows = []
    for position, day in enumerate(selected):
        print(f"paper-v1 start={position+1}/{len(selected)} origin={day} paths={paths}", flush=True)
        destination = output / f"daily/{day}.json"
        if destination.exists():
            rows.append(json.loads(destination.read_text()))
            continue
        state = model.daily_states[dates.index(day)].detach().numpy()
        conditional_mean = equilibrium + (state - equilibrium) @ transition
        future = conditional_mean + shocks @ factor.T
        current = observable(model, torch.as_tensor(state[None], dtype=model.fdtype))[0]
        surfaces = np.concatenate([observable(model, torch.as_tensor(block, dtype=model.fdtype))
                                   for block in np.array_split(future, max(1, paths // 32))])
        changes = surfaces - current
        reference_index = reference_dates.index(day)
        actual = common[reference_index + 1] - common[reference_index]
        mean = changes.mean(axis=0)
        row = {"model": "DYSANOS paper affine-200", "origin_date": day,
            "evaluation_scope": protocol["evaluation_scope"], "status": "scored", "paths": paths,
            "surface_mean": mean.tolist(), "surface_covariance": np.cov(changes, rowvar=False).tolist(),
            "actual_change": actual.tolist(), "predicted_variance_trace": float(np.var(changes, axis=0, ddof=1).sum()),
            "forecast_squared_error": float(np.sum((actual - mean)**2))}
        _json(destination, row)
        rows.append(row)
        elapsed = time.monotonic() - started
        print(f"paper-v1 completed={position+1}/{len(selected)} origin={day} paths={paths} elapsed={elapsed:.1f}s "
              f"eta={elapsed*(len(selected)-position-1)/(position+1):.1f}s", flush=True)
    predicted = float(np.mean([row["predicted_variance_trace"] for row in rows]))
    error = float(np.mean([row["forecast_squared_error"] for row in rows]))
    _json(output / "summary.json", {"origins": len(rows), "evaluation_scope": protocol["evaluation_scope"],
        "predicted_variance_trace": predicted, "forecast_mse": error, "variance_over_forecast_mse": predicted / error,
        "embedding": embedding})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--origins", nargs="+")
    args = parser.parse_args()
    if not Path(os.environ["QUANTDATA_ROOT"]).is_dir():
        raise ValueError("QUANTDATA_ROOT is unavailable")
    run(args.checkpoint, args.targets, args.reference, args.output, paths=args.paths, origins=args.origins)


if __name__ == "__main__":
    main()
