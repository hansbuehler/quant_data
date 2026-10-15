"""Calibrate and audit Tangent Levy v2 from a specified market date."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.biz_calendar import count_business_days_between

from .config import StaticFitConfig
from .double_exponential import call_prices
from .dynamics import DynamicsFit
from .pricing import average_density_from_forward, density_call_prices
from .run_v2 import _load_day, _save_day
from .static_fit import (
    DayFit,
    fitted_average_density,
    reconstruct_forward_density,
)
from .v2_audit import (
    RestrictedPopulation,
    crossfit_conditional_audit,
    unconditional_coordinate_audit,
)
from .v2_config import TangentLevyV2AuditConfig, TangentLevyV2Config
from .v2_market import exact_population_dates, load_market_day
from .v2_population import simulate_restricted_population
from .v2_training import (
    V2HistoricalFit,
    V2ReconstructionAudit,
    fit_v2_market_day,
    fit_v2_measures,
    reconstruct_v2_day,
)

TARGET_DATE = date(2025, 8, 29)
DEFAULT_OUTPUT = Path(
    "/home/hans/quantdata_data/training/dysanos/tangent-levy/v2/"
    "spx-2025-08-29-historical-diffusion-v2"
)
POPULATION_CHUNK_COUNT = 8


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _training_dates(config: TangentLevyV2Config) -> tuple[date, ...]:
    return tuple(
        value
        for value in exact_population_dates(config=config.market)
        if value <= config.dynamics.training_end
    )


def _consecutive_session_mask(dates: tuple[date, ...]) -> np.ndarray:
    return np.asarray(
        [
            count_business_days_between(left, right) == 1
            for left, right in zip(dates[:-1], dates[1:], strict=True)
        ],
        dtype=bool,
    )


def _target_checkpoint(output: Path, target_date: date) -> Path:
    return output / "initial-day" / f"{target_date}.npz"


def _history_exclusion_path(directory: Path, market_date: date) -> Path:
    return directory / f"{market_date}.excluded.json"


def _completed_history_dates(
    directory: Path,
    requested_dates: tuple[date, ...],
) -> tuple[date, ...]:
    return tuple(
        market_date
        for market_date in requested_dates
        if (directory / f"{market_date}.npz").exists()
    )


def _target_quote_metrics(
    *,
    target_date: date,
    target_fit: DayFit,
    target_density: np.ndarray,
    config: TangentLevyV2Config,
) -> dict[str, object]:
    market = load_market_day(target_date, config=config.market, verbose=None)
    if market is None:
        raise RuntimeError(f"missing target market {target_date}")
    tau_grid = np.asarray(config.dynamics.maturity_grid, dtype=np.float64)
    jump_x = np.asarray(config.dynamics.jump_grid, dtype=np.float64)
    projected_average = average_density_from_forward(
        tau=tau_grid,
        forward_density=target_density[None],
    )[0]
    static_config = StaticFitConfig()
    fitted_tau = target_fit.maturities
    fitted_density = fitted_average_density(target_fit, jump_x)
    if fitted_tau[-1] < tau_grid[-1]:
        fitted_tau = np.append(fitted_tau, tau_grid[-1])
        fitted_density = np.concatenate(
            (fitted_density, fitted_density[-1:]), axis=0
        )
    calendar_projection = reconstruct_forward_density(
        fitted_tau=fitted_tau,
        fitted_average_density=fitted_density,
        target_tau=tau_grid,
        jump_x=jump_x,
        relative_tolerance=config.dynamics.reconstruction_relative_tolerance,
        forward_density_floor=static_config.forward_density_floor,
    ).projected_fitted_average_density[: len(target_fit.fits)]
    books = {round(book.tau, 12): book for book in market.books}
    raw_prices = []
    calendar_projected_prices = []
    projected_prices = []
    bids = []
    asks = []
    for fit_index, fit in enumerate(target_fit.fits):
        book = books[round(fit.tau, 12)]
        log_strikes = np.log(book.strikes)
        raw_prices.append(
            call_prices(
                tau=fit.tau,
                log_strikes=log_strikes,
                parameters=fit.parameters,
            )
        )
        calendar_projected_prices.append(
            density_call_prices(
                tau=fit.tau,
                log_strikes=log_strikes,
                jump_x=jump_x,
                average_density=calendar_projection[fit_index],
            )
        )
        right = int(np.searchsorted(tau_grid, fit.tau, side="left"))
        right = min(max(right, 1), len(tau_grid) - 1)
        left = right - 1
        weight = (fit.tau - tau_grid[left]) / (tau_grid[right] - tau_grid[left])
        integrated_left = tau_grid[left] * projected_average[left]
        integrated_right = tau_grid[right] * projected_average[right]
        interpolated_average_density = (
            (1.0 - weight) * integrated_left + weight * integrated_right
        ) / fit.tau
        projected_prices.append(
            density_call_prices(
                tau=fit.tau,
                log_strikes=log_strikes,
                jump_x=jump_x,
                average_density=interpolated_average_density,
            )
        )
        bids.append(book.bids)
        asks.append(book.asks)

    bid = np.concatenate(bids)
    ask = np.concatenate(asks)
    half_spread = np.maximum(0.5 * (ask - bid), 1.0e-4)

    def metrics(values: list[np.ndarray]) -> dict[str, float | int]:
        price = np.concatenate(values)
        excess = np.maximum(price - ask, 0.0) + np.maximum(bid - price, 0.0)
        return {
            "quotes": int(price.size),
            "inside_bid_ask": int(np.count_nonzero(excess == 0.0)),
            "inside_bid_ask_fraction": float(np.mean(excess == 0.0)),
            "excess_rmse_half_spreads": float(
                np.sqrt(np.mean(np.square(excess / half_spread)))
            ),
            "maximum_absolute_excess": float(np.max(excess)),
        }

    return {
        "independent_maturity_fit": metrics(raw_prices),
        "calendar_consistent_projection": metrics(calendar_projected_prices),
        "daily_grid_reconstruction": metrics(projected_prices),
    }


def _fit_target(
    *,
    output: Path,
    target_date: date,
    config: TangentLevyV2Config,
) -> tuple[DayFit, np.ndarray, V2ReconstructionAudit]:
    checkpoint = _target_checkpoint(output, target_date)
    if checkpoint.exists():
        return _load_day(checkpoint)
    market = load_market_day(target_date, config=config.market, verbose=None)
    if market is None:
        raise RuntimeError(f"missing target market {target_date}")
    day_fit = fit_v2_market_day(market)
    density, audit = reconstruct_v2_day(market, day_fit, config=config)
    _save_day(checkpoint, day_fit, density, audit)
    return day_fit, density, audit


def calibrate_history(
    *,
    output: Path,
    target_date: date = TARGET_DATE,
) -> V2HistoricalFit:
    """Fit pre-cutoff history in the target date's deterministic codebook chart."""

    config = TangentLevyV2Config(label=f"spx-{target_date}-historical-diffusion-v2")
    target_fit, target_density, target_audit = _fit_target(
        output=output,
        target_date=target_date,
        config=config,
    )
    symmetry_curve = target_fit.symmetry_curve()
    requested_dates = _training_dates(config)
    directory = output / "history-days"
    directory.mkdir(parents=True, exist_ok=True)
    day_fits = []
    densities = []
    audits = []
    exclusions: list[dict[str, str]] = []
    previous_fit = None
    started = time.perf_counter()
    for index, market_date in enumerate(requested_dates):
        checkpoint = directory / f"{market_date}.npz"
        exclusion_path = _history_exclusion_path(directory, market_date)
        if exclusion_path.exists():
            exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
            exclusions.append(exclusion)
            print(
                f"tangent-levy same-date calibration excluded={market_date} "
                f"reason={exclusion['reason']}",
                flush=True,
            )
            continue
        if checkpoint.exists():
            day_fit, density, audit = _load_day(checkpoint)
        else:
            try:
                market = load_market_day(
                    market_date,
                    config=config.market,
                    verbose=None,
                )
            except RuntimeError as error:
                reason = str(error)
                if not reason.startswith("invalid retained quote book on "):
                    raise
                exclusion = {"date": str(market_date), "reason": reason}
                _atomic_json(exclusion, exclusion_path)
                exclusions.append(exclusion)
                print(
                    f"tangent-levy same-date calibration excluded={market_date} "
                    f"reason={reason}",
                    flush=True,
                )
                continue
            if market is None:
                raise RuntimeError(f"missing historical market {market_date}")
            day_fit = fit_v2_market_day(
                market,
                symmetry_curve=symmetry_curve,
                warm_start=previous_fit,
            )
            density, audit = reconstruct_v2_day(
                market,
                day_fit,
                config=config,
            )
            _save_day(checkpoint, day_fit, density, audit)
        day_fits.append(day_fit)
        densities.append(density)
        audits.append(audit)
        previous_fit = day_fit
        completed = index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed * (len(requested_dates) - completed) / completed
        print(
            "tangent-levy same-date calibration "
            f"requested={completed}/{len(requested_dates)} "
            f"retained={len(day_fits)} excluded={len(exclusions)} "
            f"gate_pass={sum(value.passed for value in audits)} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    dates = _completed_history_dates(directory, requested_dates)
    if len(dates) != len(day_fits):
        raise RuntimeError("completed history checkpoints are inconsistent")
    history = V2HistoricalFit(
        dates=dates,
        day_fits=tuple(day_fits),
        symmetry_curve=symmetry_curve,
        forward_density=np.stack(densities),
        audits=tuple(audits),
        consecutive_mask=_consecutive_session_mask(dates),
    )
    target_quote_metrics = _target_quote_metrics(
        target_date=target_date,
        target_fit=target_fit,
        target_density=target_density,
        config=config,
    )
    _atomic_json(
        {
            "schema": "tangent-levy-v2-same-date-history",
            "target_date": str(target_date),
            "training_start": str(dates[0]),
            "training_end": str(dates[-1]),
            "training_dates": len(dates),
            "requested_training_dates": len(requested_dates),
            "excluded_training_dates": exclusions,
            "consecutive_transitions": int(np.count_nonzero(history.consecutive_mask)),
            "target_static_fit": {
                "successful": target_fit.successful,
                "maturities": len(target_fit.fits),
                "quotes": sum(value.num_quotes for value in target_fit.fits),
                "mean_weighted_rmse": float(
                    np.mean([value.weighted_rmse for value in target_fit.fits])
                ),
                "maximum_weighted_rmse": max(
                    value.weighted_rmse for value in target_fit.fits
                ),
                "maximum_absolute_error": max(
                    value.maximum_absolute_error for value in target_fit.fits
                ),
            },
            "target_reconstruction": asdict(target_audit),
            "target_quote_metrics": target_quote_metrics,
            "historical_reconstruction_passed": sum(
                value.passed for value in audits
            ),
            "historical_reconstruction_failed": sum(
                not value.passed for value in audits
            ),
            "maximum_historical_relative_l1": max(
                value.relative_l1_projection for value in audits
            ),
            "maximum_historical_repricing_rmse_half_spreads": max(
                value.repricing_rmse_half_spreads for value in audits
            ),
            "target_density_shape": list(target_density.shape),
        },
        output / "history-report.json",
    )
    return history


def load_history(
    *,
    output: Path,
    target_date: date = TARGET_DATE,
) -> tuple[V2HistoricalFit, np.ndarray, V2ReconstructionAudit]:
    """Load a completed target calibration and all required historical states."""

    config = TangentLevyV2Config(label=f"spx-{target_date}-historical-diffusion-v2")
    target_fit, target_density, target_audit = _load_day(
        _target_checkpoint(output, target_date)
    )
    requested_dates = _training_dates(config)
    directory = output / "history-days"
    resolved_dates = {
        market_date
        for market_date in requested_dates
        if (directory / f"{market_date}.npz").exists()
        or _history_exclusion_path(directory, market_date).exists()
    }
    if len(resolved_dates) != len(requested_dates):
        raise RuntimeError(
            "historical calibration is incomplete: "
            f"resolved {len(resolved_dates)} of {len(requested_dates)} dates"
        )
    dates = _completed_history_dates(directory, requested_dates)
    if not dates:
        raise RuntimeError("no completed historical calibration checkpoints")
    loaded = [
        _load_day(directory / f"{market_date}.npz")
        for market_date in dates
    ]
    history = V2HistoricalFit(
        dates=dates,
        day_fits=tuple(value[0] for value in loaded),
        symmetry_curve=target_fit.symmetry_curve(),
        forward_density=np.stack([value[1] for value in loaded]),
        audits=tuple(value[2] for value in loaded),
        consecutive_mask=_consecutive_session_mask(dates),
    )
    return history, target_density, target_audit


def fit_measures(*, output: Path, target_date: date = TARGET_DATE) -> None:
    """Estimate target-local Q factors and the historical P risk premium."""

    config = TangentLevyV2Config(label=f"spx-{target_date}-historical-diffusion-v2")
    history, target_density, _ = load_history(
        output=output,
        target_date=target_date,
    )
    measures = fit_v2_measures(
        history,
        config=config,
        require_reconstruction_pass=False,
        reference_density=target_density,
    )
    dynamics = measures.q_dynamics
    physical = measures.p_market_price_of_risk
    _atomic_npz(
        output / "measure-fits.npz",
        target_date=str(target_date),
        training_start=str(history.dates[0]),
        training_end=str(history.dates[-1]),
        tau=dynamics.tau,
        jump_x=dynamics.jump_x,
        beta=dynamics.beta,
        eigenvalues_one_session=dynamics.eigenvalues_one_session,
        explained_variance_ratio=dynamics.explained_variance_ratio,
        mean_change=dynamics.mean_change,
        observations=dynamics.observations,
        maximum_constraint_residual=dynamics.maximum_constraint_residual,
        covariance_shrinkage=dynamics.covariance_shrinkage,
        dt=dynamics.dt,
        active_mask=dynamics.active_mask,
        p_theta=physical.theta,
        p_ridge=physical.ridge,
        p_residual_rmse=physical.residual_rmse,
        p_target_drift_norm=physical.target_drift_norm,
        p_explained_drift_fraction=physical.explained_drift_fraction,
    )
    _atomic_json(
        {
            "schema": "tangent-levy-v2-same-date-measures",
            "target_date": str(target_date),
            "training_start": str(history.dates[0]),
            "training_end": str(history.dates[-1]),
            "observations": dynamics.observations,
            "factor_count": dynamics.num_factors,
            "active_coordinates": int(np.count_nonzero(dynamics.active_mask)),
            "total_coordinates": int(dynamics.active_mask.size),
            "eigenvalues_one_session": dynamics.eigenvalues_one_session.tolist(),
            "explained_variance_ratio": dynamics.explained_variance_ratio.tolist(),
            "retained_explained_variance": float(
                np.sum(dynamics.explained_variance_ratio)
            ),
            "maximum_constraint_residual": dynamics.maximum_constraint_residual,
            "covariance_shrinkage": dynamics.covariance_shrinkage,
            "physical_residual_rmse": physical.residual_rmse,
            "physical_target_drift_norm": physical.target_drift_norm,
            "physical_explained_drift_fraction": physical.explained_drift_fraction,
        },
        output / "measure-report.json",
    )
    print(
        f"tangent-levy same-date measures factors={dynamics.num_factors} "
        f"observations={dynamics.observations} "
        f"active={np.count_nonzero(dynamics.active_mask)} "
        f"constraint_residual={dynamics.maximum_constraint_residual:.6g}",
        flush=True,
    )


def _load_dynamics(path: Path) -> DynamicsFit:
    with np.load(path, allow_pickle=False) as payload:
        return DynamicsFit(
            tau=np.asarray(payload["tau"], dtype=np.float64),
            jump_x=np.asarray(payload["jump_x"], dtype=np.float64),
            beta=np.asarray(payload["beta"], dtype=np.float64),
            eigenvalues_one_session=np.asarray(
                payload["eigenvalues_one_session"], dtype=np.float64
            ),
            explained_variance_ratio=np.asarray(
                payload["explained_variance_ratio"], dtype=np.float64
            ),
            mean_change=np.asarray(payload["mean_change"], dtype=np.float64),
            observations=int(payload["observations"].item()),
            maximum_constraint_residual=float(
                payload["maximum_constraint_residual"].item()
            ),
            covariance_shrinkage=float(payload["covariance_shrinkage"].item()),
            dt=float(payload["dt"].item()),
            active_mask=np.asarray(payload["active_mask"], dtype=bool),
        )


def generate_q_population(
    *,
    output: Path,
    target_date: date = TARGET_DATE,
) -> None:
    """Generate and audit the measured-factor restricted Q population."""

    config = TangentLevyV2AuditConfig()
    _, initial_density, reconstruction = _load_day(
        _target_checkpoint(output, target_date)
    )
    dynamics = _load_dynamics(output / "measure-fits.npz")
    population, universe, diagnostics = simulate_restricted_population(
        initial_density,
        dynamics,
        config=config,
        path_count=config.path_count,
        batch_size=500,
    )
    directory = output / "q-population"
    _atomic_npy(directory / "features.npy", population.features)
    _atomic_npy(directory / "returns.npy", population.returns)
    _atomic_npy(
        directory / "returns-training-f32.npy",
        population.returns.astype(np.float32),
    )
    _atomic_npy(directory / "active-mask.npy", population.active_mask)
    _atomic_npy(directory / "pair-ids.npy", population.pair_ids)
    unconditional = unconditional_coordinate_audit(
        population.returns,
        active_mask=population.active_mask,
    )
    _atomic_json(
        {
            "schema": "tangent-levy-v2-same-date-q-audit",
            "target_date": str(target_date),
            "paths": config.path_count,
            "pairs": config.path_count // 2,
            "factor_count": dynamics.num_factors,
            "factor_observations": dynamics.observations,
            "factor_constraint_residual": dynamics.maximum_constraint_residual,
            "reconstruction": asdict(reconstruction),
            "population_diagnostics": asdict(diagnostics),
            "unconditional": asdict(unconditional),
            "instrument_labels": universe.labels,
        },
        output / "q-audit.json",
    )


def generate_q_population_chunk(
    *,
    output: Path,
    chunk_index: int,
    chunk_count: int = POPULATION_CHUNK_COUNT,
    target_date: date = TARGET_DATE,
) -> None:
    """Generate one independent, pair-preserving Q population chunk."""

    config = TangentLevyV2AuditConfig()
    if not 0 <= chunk_index < chunk_count:
        raise ValueError(f"invalid chunk index {chunk_index} of {chunk_count}")
    if config.path_count % chunk_count != 0:
        raise ValueError("population chunk count must divide the path count")
    chunk_paths = config.path_count // chunk_count
    if chunk_paths % 4 != 0 or chunk_paths % 500 != 0:
        raise ValueError("each population chunk must preserve folds and full batches")
    directory = output / "q-population-chunks" / f"chunk-{chunk_index:03d}"
    report_path = directory / "chunk-report.json"
    required = (
        directory / "features.npy",
        directory / "returns.npy",
        report_path,
    )
    if all(path.exists() for path in required):
        print(f"tangent-levy-v2 Q chunk={chunk_index} already complete", flush=True)
        return
    _, initial_density, _ = _load_day(
        _target_checkpoint(output, target_date)
    )
    dynamics = _load_dynamics(output / "measure-fits.npz")
    chunk_seed = config.seed + chunk_index
    chunk_config = replace(config, seed=chunk_seed)
    population, universe, diagnostics = simulate_restricted_population(
        initial_density,
        dynamics,
        config=chunk_config,
        path_count=chunk_paths,
        batch_size=500,
    )
    _atomic_npy(directory / "features.npy", population.features)
    _atomic_npy(directory / "returns.npy", population.returns)
    _atomic_json(
        {
            "schema": "tangent-levy-v2-same-date-q-population-chunk",
            "target_date": str(target_date),
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "seed": chunk_seed,
            "diagnostics": asdict(diagnostics),
            "instrument_labels": universe.labels,
        },
        report_path,
    )


def merge_q_population_chunks(
    *,
    output: Path,
    chunk_count: int = POPULATION_CHUNK_COUNT,
    target_date: date = TARGET_DATE,
) -> None:
    """Merge complete Q chunks and run the unconditional coordinate audit."""

    config = TangentLevyV2AuditConfig()
    chunk_root = output / "q-population-chunks"
    reports = []
    feature_chunks = []
    return_chunks = []
    for chunk_index in range(chunk_count):
        directory = chunk_root / f"chunk-{chunk_index:03d}"
        report = json.loads(
            (directory / "chunk-report.json").read_text(encoding="utf-8")
        )
        if (
            report.get("schema")
            != "tangent-levy-v2-same-date-q-population-chunk"
            or report.get("chunk_index") != chunk_index
            or report.get("chunk_count") != chunk_count
            or report.get("target_date") != str(target_date)
        ):
            raise ValueError(f"incompatible population chunk {chunk_index}")
        reports.append(report)
        feature_chunks.append(np.load(directory / "features.npy", mmap_mode="r"))
        return_chunks.append(np.load(directory / "returns.npy", mmap_mode="r"))
    if sum(len(value) for value in feature_chunks) != config.path_count:
        raise ValueError("population chunks do not contain the configured path count")
    directory = output / "q-population"
    directory.mkdir(parents=True, exist_ok=True)

    def merge_arrays(name: str, chunks: list[np.ndarray], dtype: np.dtype) -> None:
        temporary = directory / f".{name}.tmp"
        shape = (config.path_count,) + chunks[0].shape[1:]
        merged = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        start = 0
        for values in chunks:
            stop = start + len(values)
            merged[start:stop] = values
            start = stop
        merged.flush()
        del merged
        os.replace(temporary, directory / name)

    merge_arrays("features.npy", feature_chunks, np.dtype(np.float32))
    merge_arrays("returns.npy", return_chunks, np.dtype(np.float64))
    merge_arrays("returns-training-f32.npy", return_chunks, np.dtype(np.float32))
    active_mask = np.zeros(
        (config.horizon_days, config.instrument_count), dtype=bool
    )
    active_mask[:, 0] = True
    for option_index, expiry in enumerate(config.option_expiry_days):
        start = 1 + option_index * config.options_per_expiry
        active_mask[:expiry, start : start + config.options_per_expiry] = True
    _atomic_npy(directory / "active-mask.npy", active_mask)
    _atomic_npy(
        directory / "pair-ids.npy",
        np.repeat(np.arange(config.path_count // 2, dtype=np.int64), 2),
    )
    returns = np.load(directory / "returns.npy", mmap_mode="r")
    unconditional = unconditional_coordinate_audit(
        returns,
        active_mask=active_mask,
    )
    diagnostics = [report["diagnostics"] for report in reports]
    total_paths = sum(int(value["path_count"]) for value in diagnostics)
    population_diagnostics = {
        "measure": "Q",
        "path_count": total_paths,
        "minimum_gamma": min(float(value["minimum_gamma"]) for value in diagnostics),
        "mean_gamma": sum(
            float(value["mean_gamma"]) * int(value["path_count"])
            for value in diagnostics
        )
        / total_paths,
        "boundary_limited_substeps": sum(
            int(value["boundary_limited_substeps"]) for value in diagnostics
        ),
        "gamma_below_0_1": sum(
            int(value["gamma_below_0_1"]) for value in diagnostics
        ),
        "gamma_below_0_01": sum(
            int(value["gamma_below_0_01"]) for value in diagnostics
        ),
        "gamma_below_0_0001": sum(
            int(value["gamma_below_0_0001"]) for value in diagnostics
        ),
        "frozen_paths": sum(int(value["frozen_paths"]) for value in diagnostics),
    }
    _, _, reconstruction = _load_day(_target_checkpoint(output, target_date))
    dynamics = _load_dynamics(output / "measure-fits.npz")
    _atomic_json(
        {
            "schema": "tangent-levy-v2-same-date-q-audit",
            "target_date": str(target_date),
            "paths": config.path_count,
            "pairs": config.path_count // 2,
            "population_chunks": chunk_count,
            "chunk_seeds": [int(report["seed"]) for report in reports],
            "factor_count": dynamics.num_factors,
            "factor_observations": dynamics.observations,
            "factor_constraint_residual": dynamics.maximum_constraint_residual,
            "reconstruction": asdict(reconstruction),
            "population_diagnostics": population_diagnostics,
            "unconditional": asdict(unconditional),
            "instrument_labels": reports[0]["instrument_labels"],
        },
        output / "q-audit.json",
    )


def conditional_q_audit(
    *,
    output: Path,
    objective: str,
    device: str,
) -> None:
    """Fit one cross-fitted conditional Q diagnostic."""

    config = TangentLevyV2AuditConfig()
    directory = output / "q-population"
    population = RestrictedPopulation(
        features=np.load(directory / "features.npy", mmap_mode="r"),
        returns=np.load(directory / "returns-training-f32.npy", mmap_mode="r"),
        active_mask=np.load(directory / "active-mask.npy"),
        pair_ids=np.load(directory / "pair-ids.npy"),
    )
    result = crossfit_conditional_audit(
        population,
        objective=objective,
        config=config,
        checkpoint_directory=output / "q-conditional-checkpoints",
        device=device,
    )
    _atomic_json(asdict(result), output / f"q-{objective}.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "calibration",
            "measures",
            "population",
            "population-chunk",
            "population-merge",
            "conditional",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-date", type=date.fromisoformat, default=TARGET_DATE)
    parser.add_argument("--objective", choices=("squared", "entropy"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--chunk-count", type=int, default=POPULATION_CHUNK_COUNT)
    arguments = parser.parse_args()
    if arguments.stage == "calibration":
        calibrate_history(output=arguments.output, target_date=arguments.target_date)
    elif arguments.stage == "measures":
        fit_measures(output=arguments.output, target_date=arguments.target_date)
    elif arguments.stage == "population":
        generate_q_population(output=arguments.output, target_date=arguments.target_date)
    elif arguments.stage == "population-chunk":
        if arguments.chunk_index is None:
            parser.error("population-chunk stage requires --chunk-index")
        generate_q_population_chunk(
            output=arguments.output,
            chunk_index=arguments.chunk_index,
            chunk_count=arguments.chunk_count,
            target_date=arguments.target_date,
        )
    elif arguments.stage == "population-merge":
        merge_q_population_chunks(
            output=arguments.output,
            chunk_count=arguments.chunk_count,
            target_date=arguments.target_date,
        )
    else:
        if arguments.objective is None:
            parser.error("conditional stage requires --objective")
        conditional_q_audit(
            output=arguments.output,
            objective=arguments.objective,
            device=arguments.device,
        )


if __name__ == "__main__":
    main()
