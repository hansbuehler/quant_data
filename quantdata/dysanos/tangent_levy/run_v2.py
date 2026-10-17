"""Resumable Tangent Levy v2 historical calibration and measure fitting."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np

from .config import StaticFitConfig
from .double_exponential import DoubleExponentialParameters
from .static_fit import DayFit, MaturityFit, SymmetryCurve
from .v2_config import TangentLevyV2Config
from .v2_market import exact_population_dates, load_market_day
from .v2_training import (
    V2HistoricalFit,
    V2ReconstructionAudit,
    fit_v2_market_day,
    fit_v2_measures,
    reconstruct_v2_day,
)

DEFAULT_OUTPUT = Path(
    "/home/hans/quantdata_data/training/dysanos/tangent-levy/v2/spx-891-v2"
)


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


def _save_day(
    path: Path,
    day_fit: DayFit,
    forward_density: np.ndarray,
    audit: V2ReconstructionAudit,
) -> None:
    parameters = np.asarray(
        [
            (
                fit.parameters.intensity,
                fit.parameters.positive_rate,
                fit.parameters.negative_rate,
                fit.parameters.positive_probability,
            )
            for fit in day_fit.fits
        ],
        dtype=np.float64,
    )
    _atomic_npz(
        path,
        market_date=str(day_fit.market_date),
        initial_symmetry_fit=day_fit.initial_symmetry_fit,
        maturities=day_fit.maturities,
        parameters=parameters,
        objective=np.asarray([fit.objective for fit in day_fit.fits]),
        success=np.asarray([fit.success for fit in day_fit.fits]),
        iterations=np.asarray([fit.iterations for fit in day_fit.fits]),
        selected_start=np.asarray([fit.selected_start for fit in day_fit.fits]),
        num_quotes=np.asarray([fit.num_quotes for fit in day_fit.fits]),
        weighted_rmse=np.asarray([fit.weighted_rmse for fit in day_fit.fits]),
        maximum_absolute_error=np.asarray(
            [fit.maximum_absolute_error for fit in day_fit.fits]
        ),
        forward_density=np.asarray(forward_density, dtype=np.float32),
        audit=np.asarray(
            [
                audit.maximum_relative_projection,
                audit.relative_l1_projection,
                audit.repricing_rmse_half_spreads,
                audit.maximum_absolute_price_error,
                float(audit.passed),
            ],
            dtype=np.float64,
        ),
    )


def _load_day(path: Path) -> tuple[DayFit, np.ndarray, V2ReconstructionAudit]:
    with np.load(path, allow_pickle=False) as payload:
        maturities = np.asarray(payload["maturities"], dtype=np.float64)
        parameters = np.asarray(payload["parameters"], dtype=np.float64)
        fits = tuple(
            MaturityFit(
                tau=float(tau),
                parameters=DoubleExponentialParameters(
                    intensity=float(parameters[index, 0]),
                    positive_rate=float(parameters[index, 1]),
                    negative_rate=float(parameters[index, 2]),
                    positive_probability=float(parameters[index, 3]),
                ),
                objective=float(payload["objective"][index]),
                success=bool(payload["success"][index]),
                iterations=int(payload["iterations"][index]),
                selected_start=int(payload["selected_start"][index]),
                start_objectives=np.empty(0, dtype=np.float64),
                num_quotes=int(payload["num_quotes"][index]),
                weighted_rmse=float(payload["weighted_rmse"][index]),
                maximum_absolute_error=float(
                    payload["maximum_absolute_error"][index]
                ),
                message="loaded v2 checkpoint",
            )
            for index, tau in enumerate(maturities)
        )
        day_fit = DayFit(
            market_date=date.fromisoformat(str(payload["market_date"].item())),
            fits=fits,
            initial_symmetry_fit=bool(payload["initial_symmetry_fit"].item()),
        )
        values = np.asarray(payload["audit"], dtype=np.float64)
        audit = V2ReconstructionAudit(
            maximum_relative_projection=float(values[0]),
            relative_l1_projection=float(values[1]),
            repricing_rmse_half_spreads=float(values[2]),
            maximum_absolute_price_error=float(values[3]),
            passed=bool(values[4]),
        )
        forward_density = np.asarray(payload["forward_density"], dtype=np.float64)
    return day_fit, forward_density, audit


def calibrate_history(
    *,
    output: Path,
    maximum_dates: int | None = None,
) -> V2HistoricalFit:
    """Calibrate every manifest date, resuming at per-day checkpoints."""

    config = TangentLevyV2Config(label="spx-891-v2")
    static_config = StaticFitConfig()
    dates = exact_population_dates(config=config.market)
    if maximum_dates is not None:
        dates = dates[:maximum_dates]
    day_directory = output / "days"
    day_directory.mkdir(parents=True, exist_ok=True)
    day_fits: list[DayFit] = []
    densities: list[np.ndarray] = []
    audits: list[V2ReconstructionAudit] = []
    symmetry_curve: SymmetryCurve | None = None
    previous_fit: DayFit | None = None
    started = time.perf_counter()
    quote_count = 0
    for index, market_date in enumerate(dates):
        checkpoint = day_directory / f"{market_date}.npz"
        market = load_market_day(market_date, config=config.market, verbose=None)
        if market is None:
            raise RuntimeError(f"missing exact-population market {market_date}")
        quote_count += market.quote_count
        if checkpoint.exists():
            day_fit, density, audit = _load_day(checkpoint)
        else:
            day_fit = fit_v2_market_day(
                market,
                static_config=static_config,
                symmetry_curve=symmetry_curve,
                warm_start=previous_fit,
            )
            density, audit = reconstruct_v2_day(
                market, day_fit, config=config, static_config=static_config
            )
            _save_day(checkpoint, day_fit, density, audit)
        if symmetry_curve is None:
            symmetry_curve = day_fit.symmetry_curve()
        day_fits.append(day_fit)
        densities.append(density)
        audits.append(audit)
        previous_fit = day_fit
        elapsed = time.perf_counter() - started
        completed = index + 1
        eta = elapsed * (len(dates) - completed) / completed
        print(
            f"tangent-levy-v2 calibration dates={completed}/{len(dates)} "
            f"quotes={quote_count} gate_pass={sum(value.passed for value in audits)} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    if len(dates) == config.market.expected_dates and quote_count != config.market.expected_quotes:
        raise RuntimeError(
            f"loaded {quote_count} quotes, expected {config.market.expected_quotes}"
        )
    assert symmetry_curve is not None
    history = V2HistoricalFit(
        dates=dates,
        day_fits=tuple(day_fits),
        symmetry_curve=symmetry_curve,
        forward_density=np.stack(densities),
        audits=tuple(audits),
        consecutive_mask=np.ones(len(dates) - 1, dtype=bool),
    )
    _atomic_json(
        {
            "schema": "tangent-levy-v2-history",
            "config_filename": config.filename,
            "dates": len(dates),
            "quotes": quote_count,
            "reconstruction_passed": sum(value.passed for value in audits),
            "reconstruction_failed": sum(not value.passed for value in audits),
            "maximum_relative_l1": max(
                value.relative_l1_projection for value in audits
            ),
            "maximum_repricing_rmse_half_spreads": max(
                value.repricing_rmse_half_spreads for value in audits
            ),
        },
        output / "history-report.json",
    )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-dates", type=int)
    parser.add_argument("--fit-measures", action="store_true")
    arguments = parser.parse_args()
    history = calibrate_history(
        output=arguments.output, maximum_dates=arguments.maximum_dates
    )
    if arguments.maximum_dates is None or arguments.fit_measures:
        config = TangentLevyV2Config(label="spx-891-v2")
        measures = fit_v2_measures(
            history,
            config=config,
            require_reconstruction_pass=False,
        )
        dynamics = measures.q_dynamics
        physical = measures.p_market_price_of_risk
        _atomic_npz(
            arguments.output / "measure-fits.npz",
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
            maturity_node_indices=dynamics.maturity_node_indices,
            jump_node_indices=dynamics.jump_node_indices,
            p_theta=physical.theta,
            p_ridge=physical.ridge,
            p_residual_rmse=physical.residual_rmse,
            p_target_drift_norm=physical.target_drift_norm,
            p_explained_drift_fraction=physical.explained_drift_fraction,
        )
        print(
            f"tangent-levy-v2 measures factors={dynamics.num_factors} "
            f"observations={dynamics.observations} "
            f"constraint_residual={dynamics.maximum_constraint_residual:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
