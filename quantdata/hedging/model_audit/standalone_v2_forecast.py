"""Evaluate standalone DYSANOS v2 on the common one-day observable grid."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity
from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.metrics import covariance
from quantdata.hedging.model_audit.rolling_versions import origins
from quantdata.hedging.model_audit.run import QUANTILES, _hash, _json, _summarize


def sample_next(current: np.ndarray, fit, *, paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the standalone normal spot/factor law, without projecting out innovations."""
    random = np.random.default_rng(seed)
    innovations = random.multivariate_normal(np.zeros(fit.cov.shape[0]), fit.cov,
        size=paths, check_valid="raise", tol=1e-9)
    returns = fit.r_mean + innovations[:, -1]
    factors = fit.scores[-1] @ fit.A.T + innovations[:, :-1]
    following = current + fit.alpha + returns[:, None] * fit.beta + factors @ fit.directions.T
    if not np.isfinite(following).all() or not np.isfinite(returns).all():
        raise ValueError("Nonfinite standalone V2 state or spot innovations")
    return following, returns


def forward_log_returns(total_returns: np.ndarray, *, dividend_per_session: float,
                        spot: float, forward: float) -> np.ndarray:
    """Convert the standalone dividend-adjusted log return to the common forward numeraire."""
    if not all(np.isfinite(value) for value in (dividend_per_session, spot, forward)) or spot <= 0. or forward <= 0.:
        raise ValueError("A finite positive spot and one-day forward are required")
    return total_returns - dividend_per_session - np.log(forward / spot)


def make_pricer(configuration, indices: np.ndarray):
    import jax
    import jax.numpy as jnp
    from openqdata.models._dysanos import dlv
    from openqdata.models.dysanos2 import surface

    @jax.jit
    def decode(natives):
        nodes, density, scale = surface.decode(natives, configuration.surface)
        return nodes[:, indices], density[:, indices], scale[:, indices]

    @jax.jit
    def prices(nodes, density, scale, strikes):
        return dlv.price(nodes, density, scale, strikes, configuration.surface.eta)

    return decode, prices


def common_strikes(atm: np.ndarray, times: np.ndarray, nms: np.ndarray) -> np.ndarray:
    """Use the audit reference's NMS convention, including its short-end offset."""
    return NMS.from_nms(nms[None, None, :], times[None, :, None], vol=atm[:, :, None])


def observable(natives: np.ndarray, *, decode, prices, times: np.ndarray, nms: np.ndarray,
               batch_size: int) -> np.ndarray:
    import jax.numpy as jnp

    out = []
    for begin in range(0, len(natives), batch_size):
        states = natives[begin:begin + batch_size]
        nodes, density, scale = decode(jnp.asarray(states))
        atm_prices = np.asarray(prices(nodes, density, scale, jnp.ones((len(states), len(times), 1))))[..., 0]
        atm, valid = implied_volatilities_with_validity(expiries_y=np.broadcast_to(times, atm_prices.shape).ravel(),
            normalized_strikes=np.ones(atm_prices.size), normalized_prices=atm_prices.ravel(),
            is_call=np.ones(atm_prices.size, dtype=bool))
        if not valid.all():
            raise ValueError(f"Standalone V2 ATM inversion unusable: {int((~valid).sum())}/{len(valid)}")
        atm = atm.reshape(atm_prices.shape)
        strikes = common_strikes(atm, times, nms)
        call_prices = np.asarray(prices(nodes, density, scale, jnp.asarray(strikes)))
        time_grid = np.broadcast_to(times[None, :, None], strikes.shape)
        vols, valid = implied_volatilities_with_validity(expiries_y=time_grid.ravel(),
            normalized_strikes=strikes.ravel(), normalized_prices=call_prices.ravel(),
            is_call=np.ones(call_prices.size, dtype=bool))
        if not valid.all():
            raise ValueError(f"Standalone V2 common-grid inversion unusable: {int((~valid).sum())}/{len(valid)}")
        out.append(np.log(vols.reshape(len(states), -1)))
    return np.concatenate(out)


def run(*, states_path: Path, observations_path: Path, reference: Path, output: Path, paths: int,
        batch_size: int, selected: tuple[str, ...]) -> None:
    from openqdata.models._dysanos import dlv, precision
    from openqdata.models.dysanos2 import config, evaluate, factors, surface
    from openqdata.models.dysanos2.config import Config
    from openqdata.models.dysanos2.evaluate import driftless_returns
    from openqdata.models.dysanos2.factors import coordinate_weights, fit_factors

    configuration = Config()
    if configuration.factors.mode != "density":
        raise ValueError("This evaluation requires the standalone density-mode V2")
    with np.load(states_path, allow_pickle=False) as saved:
        states = {name: saved[name].copy() for name in ("U", "mass", "dates", "spot", "div_1d")}
    with np.load(reference / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis, nms, times = (saved[name].copy() for name in ("basis", "nms_nodes", "expiries_y"))
    observations = pd.read_parquet(observations_path)
    observations.index = pd.to_datetime(observations["date"]).dt.strftime("%Y-%m-%d")
    if observations.index.has_duplicates or not set(selected) <= set(observations.index):
        raise ValueError("The forward-curve observations do not uniquely cover every origin")
    model_days = np.asarray(configuration.surface.expiry_days)
    positions = [np.flatnonzero(model_days == round(day * 252.)) for day in times]
    if any(len(position) != 1 for position in positions):
        raise ValueError("The standalone model does not contain every common reference expiry")
    decode, prices = make_pricer(configuration, np.array([position[0] for position in positions]))
    dates = states["dates"].astype(str)
    if len(set(dates)) != len(dates):
        raise ValueError("Standalone V2 state dates are duplicated")
    lookup = {day: index for index, day in enumerate(dates)}
    returns = driftless_returns(states["spot"], states["div_1d"], dates)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "standalone-v2-density-observable-forecast-v1", "dates": list(selected),
        "states_sha256": _hash(states_path), "reference_sha256": _hash(reference / "observable_reference.json"),
        "observations_sha256": _hash(observations_path),
        "pca_sha256": _hash(reference / "frozen_observable_pca.npz"),
        "standalone_source_sha256": {Path(module.__file__).name: _hash(Path(module.__file__))
            for module in (config, evaluate, factors, surface, dlv, precision)},
        "configuration": json.loads(configuration.to_json()), "paths": paths, "batch_size": batch_size,
        "jax_precision": "float64" if precision.X64 else "float32",
        "measure": "P", "initialization": "fresh independent rolling fit; no previous-origin weights",
        "law": "standalone density-weighted factor VAR with jointly Gaussian spot and factor innovations",
        "spot_convention": "fit the standalone dividend-adjusted returns unchanged; convert sampled log returns to the reference one-day forward numeraire using the origin dividend and forward",
        "surface_reference": "standalone source smoothness 0.1; realized observable reference smoothness 0.25",
        "normalization": "floating common-NMS log-IV; vector cdxcore.bs inversion; no filtered failures"}
    identity = output / "PROTOCOL.json"
    if identity.exists() and json.loads(identity.read_text()) != protocol:
        raise ValueError("Output directory belongs to a different standalone V2 evaluation")
    _json(identity, protocol)
    records, started = [], time.monotonic()
    for position, origin in enumerate(selected):
        health_check()
        destination = output / f"daily/{origin}.json"
        if destination.exists():
            record = json.loads(destination.read_text())
        else:
            try:
                index = lookup[origin]
                window = configuration.factors.window
                if index < window or dates[index] != origin:
                    raise ValueError("Insufficient causal standalone V2 history")
                market = observations.loc[origin]
                if not np.isclose(float(market.spot), float(states["spot"][index]), rtol=1e-10, atol=1e-10):
                    raise ValueError("Standalone and common-reference spot values differ")
                fit = fit_factors(states["U"][index-window:index+1], returns[index-window:index],
                    coordinate_weights(states["mass"][index], configuration.surface, configuration.factors),
                    configuration.factors)
                following, total_spot = sample_next(states["U"][index], fit, paths=paths, seed=20_261_112)
                spot = forward_log_returns(total_spot, dividend_per_session=float(states["div_1d"][index]),
                    spot=float(market.spot), forward=float(market.forward_reference_1d))
                current = observable(states["U"][index:index+1], decode=decode, prices=prices,
                    times=times, nms=nms, batch_size=batch_size)
                surfaces = observable(following, decode=decode, prices=prices,
                    times=times, nms=nms, batch_size=batch_size)
                changes = surfaces - current
                targets = np.c_[changes @ basis, spot]
                joint = np.c_[spot, changes @ basis]
                quantiles = np.quantile(targets, QUANTILES, axis=0)
                baseline = json.loads((reference / f"daily/FuNVol/{origin}.json").read_text())
                record = {name: baseline[name] for name in
                    ("origin_date", "endpoint", "actual_change", "actual_spot", "actual_targets")}
                record.update(model="standalone DYSANOS v2 density-P", status="scored", scenario_count=paths,
                    valid_scenario_count=paths, surface_mean=changes.mean(axis=0).tolist(),
                    surface_covariance=covariance(changes, ddof=0).tolist(), joint_mean=joint.mean(axis=0).tolist(),
                    joint_covariance=covariance(joint, ddof=0).tolist(), quantiles=quantiles.tolist(),
                    es_lower=[float(targets[targets[:, column] <= quantiles[1, column], column].mean()) for column in range(4)],
                    invalid_surface_rate=0., extreme_surface_rate=float(np.any(surfaces > np.log(3.), axis=1).mean()),
                    forecast_rmse=float(np.sqrt(np.mean((changes.mean(axis=0) - np.asarray(record["actual_change"]))**2))),
                    diagnostics={"spectral_radius": float(np.max(np.abs(np.linalg.eigvals(fit.A)))),
                        "state_start": dates[index-window], "state_end": origin, "spot_law": "Gaussian log return",
                        "factor_law": "Gaussian VAR innovations", "source": "standalone openqdata_models_dysanos",
                        "forward_numeraire_shift": float(total_spot[0] - spot[0])})
            except (ValueError, FloatingPointError, RuntimeError) as error:
                record = {"model": "standalone DYSANOS v2 density-P", "origin_date": origin,
                    "status": "error", "exact_error": f"{type(error).__name__}: {error}"}
                print(json.dumps(record), flush=True)
            _json(destination, record)
        records.append(record)
        _json(output / "SUMMARY.json", {"processed": len(records), "scored": sum(item["status"] == "scored" for item in records),
            "failures": [item for item in records if item["status"] != "scored"], "statistics": _summarize(records, basis)})
        elapsed = time.monotonic() - started
        print(f"standalone V2 forecasts completed={position+1}/{len(selected)} elapsed={elapsed:.1f}s "
            f"eta={elapsed*(len(selected)-position-1)/(position+1):.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("states", "observations", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--origin")
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu" or args.paths < 4 or args.batch_size < 1:
        raise ValueError("Use the explicit CPU environment, at least four paths and a positive batch size")
    selected = origins(args.reference)
    if args.origin:
        selected = (selected[selected.index(args.origin)],)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(args.output / ".forecast.lock"):
        run(states_path=args.states, observations_path=args.observations, reference=args.reference, output=args.output, paths=args.paths,
            batch_size=args.batch_size, selected=selected)


if __name__ == "__main__":
    main()
