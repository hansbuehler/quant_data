"""Independent STRT-16 windows using the original native decoder and VAR code."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.linalg import solve_triangular

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    LogVarianceDiffusionConfig,
    filter_log_variance_diffusion,
    fit_log_variance_diffusion,
)
from quantdata.hedge2.dlv import DlvConfig, build_day
from quantdata.hedge2.dynamics import DynamicsConfig
from quantdata.hedge2.observations import daily_observation
from quantdata.hedge2.source import SourceArchive
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.density_weighted import density_metric
from quantdata.hedging.model_audit.grid_shape import _score
from quantdata.hedging.model_audit.legacy_rolling import Progress
from quantdata.hedging.model_audit.projection_metrics import (
    _save,
    affine_projection,
    fit_projected,
    metric_projection,
    observable_measures,
)
from quantdata.hedging.model_audit.rolling_dysanos import window_dates
from quantdata.hedging.model_audit.rolling_versions import (
    INITIALIZATION_POLICY,
    origins,
    resource_health,
)
from quantdata.hedging.model_audit.run import _hash, _json, _summarize
from quantdata.hedging.model_audit.tail import TailSampler, conditional_factor_innovations


def checked_native(path: Path, day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject missing, noncausal or unverified daily states instead of skipping dates."""
    with np.load(path, allow_pickle=False) as saved:
        if (str(saved["date"]) != str(day) or not bool(saved["reconstruction_passed"])
                or not np.isclose(float(saved["source_smoothness"]), .1)):
            raise ValueError(f"Unverified native market state: {path}")
        native, density, support = (saved[name].copy() for name in ("raw_u", "decoded_densities", "support_strikes"))
    if not all(np.isfinite(value).all() for value in (native, density, support)):
        raise ValueError(f"Nonfinite native market state: {path}")
    return native, density, support


def prepare_missing_inputs(root: Path, reference: Path, source: Path) -> None:
    """Recreate omitted dates from the identical saved source, never extend a window."""
    selected = origins(reference)
    calendar = get_business_days(date(2020, 1, 1), date.fromisoformat(selected[-1]))
    required = sorted({day for origin in selected for day in window_dates(calendar, date.fromisoformat(origin))})
    observations = pd.read_parquet(source / "observations.parquet")
    observations["date"] = pd.to_datetime(observations["date"]).dt.date
    known = set(observations.date)
    with np.load(source / f"days/{selected[0]}.npz", allow_pickle=False) as saved:
        source_path = Path(str(saved["source_model"]))
    archive = SourceArchive(root=source_path.parents[4], smoothness=.1)
    licensed = Path(observations.source_cache_file.iloc[0]).parent.parent
    configuration = json.loads((source / "config.json").read_text())
    configuration["source_smoothness"] = .1
    config = DlvConfig(**{field.name: configuration[field.name] for field in fields(DlvConfig) if field.name in configuration})
    recovered = []
    for day in required:
        if not (source / f"days/{day}.npz").exists():
            destination = root / f"native-overlay/{day}.npz"
            if not destination.exists():
                values = build_day(archive, day, config=config)
                if not values["reconstruction_passed"]:
                    raise ValueError(f"Recovered input {day} failed the original price reconstruction check")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".partial.npz")
                np.savez_compressed(temporary, **values)
                temporary.replace(destination)
            recovered.append(str(day))
        if day not in known:
            observations = pd.concat((observations, pd.DataFrame([daily_observation(licensed, day)])), ignore_index=True)
    observations = observations.sort_values("date").reset_index(drop=True)
    session_lookup = {day: index for index, day in enumerate(get_business_days(observations.date.iloc[0], observations.date.iloc[-1]))}
    observations["session_index"] = [session_lookup[day] for day in observations.date]
    exact = np.diff(observations.session_index, prepend=-2) == 1
    returns = np.r_[np.nan, observations.spot.to_numpy()[1:] / observations.forward_reference_1d.to_numpy()[:-1] - 1.]
    returns[~exact] = np.nan
    observations["normalized_spot_return_into_date"] = returns
    observations["exact_previous_session"] = exact
    observations.to_parquet(root / "observations.parquet", index=False)
    _json(root / "RECOVERED_INPUTS.json", {"dates": recovered, "source_archive": str(archive.root),
        "method": "original build_day and daily_observation; identical saved smoothness-0.1 fits and licensed carry cache",
        "window_extended": False, "source_files_modified": False})


def independent_tail(fit, *, sessions: np.ndarray, directory: Path) -> dict:
    """Fit the original scalar tail diffusion without a preceding-window initializer."""
    loading, factor, residuals = conditional_factor_innovations(fit)
    white = solve_triangular(factor, residuals.T, lower=True).T
    if len(sessions) != len(white):
        raise ValueError("Tail residuals must have exactly the retained VAR chronology")
    radii = np.mean(white**2, axis=1)
    starts = np.r_[True, np.diff(sessions) != 1]
    tail = fit_log_variance_diffusion(radii, starts, dimension=white.shape[1], initial_config=None)
    record = {"origin_date": str(fit.as_of_date), "training_start": str(fit.training_start),
        "training_end": str(fit.training_end), "initialisation_date": None, "fit": tail.to_dict(step_sessions=1),
        "initial_next_log_variance_mean": float(tail.filter.next_predicted_means[-1]),
        "initial_next_log_variance_variance": float(tail.filter.next_predicted_variances[-1]),
        "conditional_spot_loading": loading.tolist(), "conditional_factor": factor.tolist(),
        "spot_policy": "normal spot; conditional surface innovations use the original log-variance diffusion",
        "innovation_count": len(radii), "initialization": INITIALIZATION_POLICY}
    _json(directory / "tail.json", record)
    if not tail.success:
        names = ("persistence", "stationary_log_variance", "unconditional_scale")
        values = np.asarray([getattr(tail.config, name) for name in names])
        lower, upper = np.array([0., 1e-8, .05]), np.array([.999, 10., 20.])

        def objective(parameters: np.ndarray) -> float:
            config = LogVarianceDiffusionConfig(**dict(zip(names, parameters, strict=True)))
            return -float(np.mean(filter_log_variance_diffusion(radii, starts,
                dimension=white.shape[1], config=config).log_likelihood))

        gradients = []
        for step in (1e-4, 1e-5):
            gradient = []
            for coordinate in range(len(values)):
                above, below = values.copy(), values.copy()
                above[coordinate] = min(values[coordinate] + step, upper[coordinate])
                below[coordinate] = max(values[coordinate] - step, lower[coordinate])
                derivative = (objective(above) - objective(below)) / (above[coordinate] - below[coordinate])
                if values[coordinate] <= lower[coordinate] + 1e-10 and derivative > 0.:
                    derivative = 0.
                if values[coordinate] >= upper[coordinate] - 1e-10 and derivative < 0.:
                    derivative = 0.
                gradient.append(float(derivative))
            gradients.append(gradient)
        norm = float(np.max(np.abs(gradients)))
        record["numerical_review"] = {"exact_solver_message": tail.message, "parameters_changed": False,
            "projected_gradients": gradients, "steps": [1e-4, 1e-5], "tolerance": 1e-6,
            "maximum_projected_gradient": norm, "local_accuracy_verified": bool(np.isfinite(norm) and norm <= 1e-6)}
        _json(directory / "tail.json", record)
        if not record["numerical_review"]["local_accuracy_verified"]:
            raise RuntimeError(f"Tail optimizer requires numerical review: {tail.message}; projected gradient={norm:.6g}")
        print(f"Tail optimizer warning preserved: {tail.message!r}; independent projected gradient={norm:.3g}, "
            "local accuracy verified without changing parameters", flush=True)
    return record


def fit_origin(root: Path, reference: Path, source: Path, *, origin: str, paths: int = 1000) -> dict:
    """Fit one causal window and score its one-day observable law on the unchanged reference."""
    origins(reference).index(origin)
    day = date.fromisoformat(origin)
    dates = window_dates(get_business_days(date(2020, 1, 1), day), day)
    directory = root / f"fits/dysanos2/{origin}"
    directory.mkdir(parents=True, exist_ok=True)
    raw_config = json.loads((source / "config.json").read_text())
    raw_config["source_smoothness"] = .1
    config = DlvConfig(**{field.name: raw_config[field.name] for field in fields(DlvConfig) if field.name in raw_config})
    observations = pd.read_parquet(root / "observations.parquet")
    observations.index = pd.to_datetime(observations["date"]).dt.date
    selected = observations.loc[list(dates)]
    sessions = selected.session_index.to_numpy(int)
    if np.any(np.diff(sessions) != 1):
        raise ValueError("A 250-session window must contain 249 exact daily transitions")
    spots = selected.normalized_spot_return_into_date.to_numpy(float)
    inputs = [source / f"days/{value}.npz" if (source / f"days/{value}.npz").exists()
        else root / f"native-overlay/{value}.npz" for value in dates]
    with FileLock(directory / ".training.lock"), Progress(directory, family="DYSANOS v2", origin=origin) as progress:
        progress.phase = "validate original daily native states"
        loaded = [checked_native(path, value) for path, value in zip(inputs, dates, strict=True)]
        levels = np.stack([entry[0] for entry in loaded])
        weights = density_metric(np.stack([entry[1] for entry in loaded[:-1]]))
        state = DailyState(day=day, native=levels[-1], reference_support=loaded[-1][2], config=config)
        changes = np.diff(levels, axis=0)
        changes -= changes.mean(axis=0)
        spot = spots[1:] - spots[1:].mean()
        beta = spot @ changes / (spot @ spot)
        residual = changes - spot[:, None] * beta
        span_encoder, span_decoder, span_coverage = affine_projection(residual, weights, rank=len(residual) - 1)
        progress.phase = "original fixed-cash-strike straddle Jacobian"
        pricer = state.decode().pricer
        strikes = np.broadcast_to(pricer.support[1:-1], (len(config.business_dtes), config.num_inner_strikes))
        tenors = np.asarray(config.business_dtes)
        premiums = np.concatenate([pricer.european_at(expiry_y=tenor / 252., strikes=row, is_call=row < 1.)
            for tenor, row in zip(tenors, strikes, strict=True)])
        if not np.isfinite(premiums).all() or np.any(premiums <= 0.):
            raise ValueError("ITM premiums must be finite and positive")
        arguments = {"strikes": strikes, "business_dtes": tenors,
            "origin_support": pricer.support, "origin_premium": premiums}
        steps = 1e-4 / np.maximum(np.max(np.abs(span_decoder), axis=0), 1.)
        images, guards = [], []
        for fraction in (1., .5):
            bumps = span_decoder.T * (steps * fraction)[:, None]
            values, guard = observable_measures(state, np.r_[levels[-1] + bumps, levels[-1] - bumps], **arguments)
            images.append((values["straddle_sqrt_time"][:len(bumps)] - values["straddle_sqrt_time"][len(bumps):])
                / (2. * fraction * steps[:, None]))
            guards.append(guard)
            progress.emit()
        encoder, decoder, coverage = metric_projection(residual, span_encoder, images[-1], rank=16)
        progress.phase = "independent daily VAR and tail diffusion"
        fit = fit_projected(levels=levels, dates=dates, spots=spots, sessions=sessions, encoder=encoder, decoder=decoder,
            config=DynamicsConfig(window_transitions=len(dates) - 1, rank=16, spot_orthogonal=True), previous=None)
        _save(directory, fit, encoder, decoder, beta)
        protocol = {"schema": "original-code-independent-strt16-v1", "origin": origin, "dates": list(map(str, dates)),
            "initialization": INITIALIZATION_POLICY, "measure": "P", "native_dimension": levels.shape[1],
            "factor_rank": 16, "source_smoothness": .1, "reference_smoothness": .25, "full_native_spot_loading": True,
            "estimator": "original direct STRT tangent projection and separate daily VAR, not paper affine-200",
            "geometry": raw_config, "metric_coverage": coverage, "span_coverage": span_coverage,
            "jacobian_half_step_relative_change": float(np.linalg.norm(images[0] - images[1]) / np.linalg.norm(images[1])),
            "tangent_decoder_guards": guards, "source_sha256": _hash(Path(__file__)),
            "input_hashes": {str(path): _hash(path) for path in inputs}}
        _json(directory / "PROTOCOL.json", protocol)
        tail = independent_tail(fit, sessions=sessions[1:], directory=directory)
        progress.phase = "matched one-day forecast evaluation"
        sampler = TailSampler(fit, tail, paths=paths, seed=20_261_112)
        shocks = sampler.draw()
        changes = fit.forecast_state_change(levels[-1] @ encoder) + shocks[:, :1] * fit.spot_state_loading
        following = levels[-1] + (changes + shocks[:, 1:] @ fit.directions.T) @ decoder.T
        following += shocks[:, :1] * (beta - (beta @ encoder) @ decoder.T)
        with np.load(reference / "frozen_observable_pca.npz", allow_pickle=False) as saved:
            basis = saved["basis"]
        reference_record = json.loads((reference / f"daily/FuNVol/{origin}.json").read_text())
        baseline = {name: reference_record[name] for name in
            ("origin_date", "endpoint", "status", "actual_change", "actual_spot", "actual_targets")}
        baseline["diagnostics"] = {"sampling": "original TailSampler,1000 seeded paths from this origin's independent VAR",
            "seed": 20_261_112, "scenario_path": None, "model_directory": str(directory)}
        baseline["bidask"] = {"status": "not_scored", "reason": "Matched native bid/ask reconstruction audit pending"}
        packed = following.reshape(paths, len(config.business_dtes), config.num_inner_strikes + 1)
        lower = np.r_[np.full(config.num_inner_strikes, 2. * np.log(config.min_local_vol)), np.log(config.min_local_vol)]
        upper = np.r_[np.full(config.num_inner_strikes, 2. * np.log(config.max_local_vol)), np.log(config.max_local_vol)]
        clipped = (packed < lower) | (packed > upper)
        mass = np.sum(clipped[:, :, :-1] * loaded[-1][1][None, :, 1:-1], axis=-1)
        clipping = {"business_dtes": tenors.tolist(), "paths_with_any_clip_rate": float(clipped.any(axis=(1, 2)).mean()),
            "coordinate_clip_counts_by_tenor": clipped.sum(axis=(0, 2)).tolist(),
            "path_clip_rates_by_tenor": clipped.any(axis=2).mean(axis=0).tolist(),
            "forward_scale_clip_rates": clipped[:, :, -1].mean(axis=0).tolist(),
            "origin_mass_on_clipped_nodes_mean_by_tenor": mass.mean(axis=0).tolist(),
            "scope": "All native tenors; origin-density masses are descriptive, not an unguarded repricing comparison"}
        record = _score(state, following, np.log1p(shocks[:, 0]), baseline, label="DYSANOS v2-P", basis=basis,
            diagnostics={"training_start": str(dates[0]), "training_end": origin, "initialisation_date": None,
                "metric_coverage": coverage, "constrained_spectral_radius": fit.constrained_spectral_radius,
                "unconstrained_spectral_radius": fit.unconstrained_spectral_radius, "tail_fit": tail["fit"],
                "bounded_decoder_clipping": clipping})
        _json(root / f"daily/DYSANOS_v2-P/{origin}.json", record)
        _json(directory / "COMPLETE.json", {"model": "DYSANOS v2-P", "origin": origin,
            "paths": paths, "status": "trained and one-day scored with original bounded decoder; clipping review and remaining diagnostics pending",
            "dynamics_sha256": _hash(directory / "dynamics.npz")})
        progress.phase = "complete"
        progress.completed = 4
        progress.emit()
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--origin")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if not Path(os.environ["QUANTDATA_ROOT"]).is_dir():
        raise ValueError("Configured data root is missing")
    selected = (args.origin,) if args.origin else origins(args.reference)
    args.root.mkdir(parents=True, exist_ok=True)
    prepare_missing_inputs(args.root, args.reference, args.source)
    records, started = [], time.monotonic()
    for index, origin in enumerate(selected):
        if any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in resource_health()):
            raise RuntimeError("Managed resources require stopping; no new origin launched")
        destination = args.root / f"daily/DYSANOS_v2-P/{origin}.json"
        if (args.root / f"fits/dysanos2/{origin}/COMPLETE.json").exists() and not args.refresh:
            record = json.loads(destination.read_text())
        else:
            record = fit_origin(args.root, args.reference, args.source, origin=origin)
        records.append(record)
        with np.load(args.reference / "frozen_observable_pca.npz", allow_pickle=False) as saved:
            _json(args.root / "DYSANOS_v2-P-summary.json", _summarize(records, saved["basis"]))
        elapsed = time.monotonic() - started
        print(f"DYSANOS v2 completed={index+1}/{len(selected)} elapsed={elapsed:.1f}s "
            f"eta={elapsed*(len(selected)-index-1)/(index+1):.1f}s", flush=True)


if __name__ == "__main__":
    main()
