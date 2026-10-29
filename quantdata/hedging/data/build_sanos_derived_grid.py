"""Regrid public SANOS states through the original SANOS density fitter."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import torch
from cdxcore.verbose import Context

from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.lp import _recover_dlvsqs
from quantdata.dysanos.flex.model import (
    NativeParameters,
    bound_native_raw,
    raw_from_native_parameters,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, make_supports_from_total_vols
from quantdata.hedging.data.pricing import price_sanos_surface
from quantdata.sanos.arbfree import MkArbFree
from quantdata.torch import BaseEnvironment

BUSINESS_DTES = (1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0)
GEOMETRY = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
    num_model_expiries=len(BUSINESS_DTES),
    explicit_model_business_dtes=BUSINESS_DTES,
    num_inner_strikes=200,
    nms_range=(-4.0, 3.0),
)


@lru_cache(maxsize=1)
def _decoder() -> FlexSanosDecoder:
    return FlexSanosDecoder(
        BaseEnvironment(device="cpu", precision=64), geometry=GEOMETRY
    )


def _target_variances(model_times: np.ndarray, model_variances: np.ndarray) -> tuple[np.ndarray, int]:
    expiries = GEOMETRY.model_expiries_y
    variances = np.interp(expiries, model_times, model_variances)
    before_first = expiries < model_times[0]
    after_last = expiries > model_times[-1]
    variances[before_first] = (
        model_variances[0] * expiries[before_first] / model_times[0]
    )
    variances[after_last] = (
        model_variances[-1] * expiries[after_last] / model_times[-1]
    )
    increments = np.diff(variances, prepend=0.0)
    minimum_increments = (
        (GEOMETRY.min_forward_vol * (1.0 + 1.0e-6)) ** 2
        * np.diff(expiries, prepend=0.0)
    )
    repaired = int(np.count_nonzero(increments < minimum_increments))
    return np.cumsum(np.maximum(increments, minimum_increments)), repaired


def fit_day(
    geometry_row: dict,
    state_row: dict,
    *,
    timeout_seconds: float = 180.0,
) -> dict[str, object]:
    """Fit the specified shared-grid DYSANOS coordinates to SANOS prices."""
    if geometry_row["date"] != state_row["date"]:
        raise ValueError("SANOS state and geometry dates do not match")
    model_times = np.asarray(geometry_row["model_expiries_y"], dtype=np.float64)
    model_variances = np.asarray(state_row["kernel_total_variances"], dtype=np.float64)
    target_variances, repaired = _target_variances(model_times, model_variances)
    expiries = GEOMETRY.model_expiries_y
    time_increments = np.diff(expiries, prepend=0.0)
    forward_vols = np.sqrt(np.diff(target_variances, prepend=0.0) / time_increments)
    if np.any(forward_vols >= GEOMETRY.max_forward_vol):
        raise ValueError("SANOS forward volatility exceeds the DYSANOS bound")
    root_increments = forward_vols * np.sqrt(time_increments)
    total_vols = np.cumsum(root_increments)
    support = make_supports_from_total_vols(total_vols, geometry=GEOMETRY)[0]
    strikes = support[1:-1]

    source_calls = price_sanos_surface(
        query_expiries_y=np.repeat(expiries, len(strikes)),
        query_strikes=np.tile(strikes, len(expiries)),
        is_call=np.ones(len(expiries) * len(strikes), dtype=bool),
        model_expiries_y=model_times,
        atm_variances=model_variances,
        support_strikes=geometry_row["support_strikes"],
        densities=state_row["densities"],
        smoothness=0.25,
        one_minute_y=1.0 / (365.0 * 24.0 * 60.0),
    ).reshape(len(expiries), len(strikes))

    front_eta = 0.49
    target_kernel_vols = 0.5 * np.sqrt(target_variances)
    later_total = np.cumsum(root_increments[1:])
    later_eta = (
        target_kernel_vols[-1] - front_eta * root_increments[0]
    ) / later_total[-1]
    later_eta = float(np.clip(later_eta, GEOMETRY.min_eta + 1.0e-6, GEOMETRY.max_eta - 1.0e-6))
    backbone_vols = front_eta * root_increments[0] + later_eta * np.concatenate(
        (np.zeros(1), later_total)
    )
    fit_variances = backbone_vols**2 / 0.25

    epsilon_vol = total_vols[0] / np.sqrt(expiries[0])
    nms_scales = np.sqrt(
        (epsilon_vol**2 * GEOMETRY.nms_epsilon_y + total_vols**2)
        / (1.0 + GEOMETRY.nms_epsilon_y)
    )
    nms = np.log(strikes)[None, :] / nms_scales[:, None]
    core = (nms >= -3.5) & (nms <= 2.5)
    fit_half_spreads = np.where(core, 1.0e-5, 1.0e-4)
    fitter = MkArbFree(
        smoothness=0.25,
        mid_fit_weight=1.0,
        min_half_spread=1.0e-6,
        min_dlv=np.sqrt(GEOMETRY.min_dlvsq),
        max_dlv=np.sqrt(GEOMETRY.max_dlvsq),
        timeout_seconds=timeout_seconds,
    )
    result = fitter.surface_densities(
        expiries=expiries,
        variances=fit_variances,
        strikes=[strikes] * len(expiries),
        bids=[source_calls[index] - fit_half_spreads[index] for index in range(len(expiries))],
        asks=[source_calls[index] + fit_half_spreads[index] for index in range(len(expiries))],
        masks=None,
        xstrikes=[support] * len(expiries),
        verbose=Context.quiet,
    )
    densities = np.stack(result.xdensities)
    fitted_calls = np.stack(result.fits)
    fallback = np.broadcast_to(forward_vols[:, None] ** 2, (len(expiries), len(strikes)))
    dlvsqs = _recover_dlvsqs(
        densities,
        np.broadcast_to(support, (len(expiries), len(support))),
        geometry=GEOMETRY,
        density_mass_floor=1.0e-7,
        fallback_dlvsqs=fallback,
        enforce_dlv_bounds=True,
    )
    eta = np.r_[front_eta, np.repeat(later_eta, len(expiries) - 1)]
    native = NativeParameters(
        dlvsqs=torch.as_tensor(dlvsqs, dtype=torch.float64),
        forward_vols=torch.as_tensor(forward_vols, dtype=torch.float64),
        eta=torch.as_tensor(eta, dtype=torch.float64),
    )
    raw_state = raw_from_native_parameters(
        native, geometry=GEOMETRY, probability_floor=1.0e-12
    ).numpy()
    if not np.all(np.isfinite(raw_state)):
        raise FloatingPointError("Fitted raw DYSANOS coordinates are non-finite")
    with torch.no_grad():
        decoder = _decoder()
        decoded = bound_native_raw(
            torch.as_tensor(raw_state, dtype=torch.float64), geometry=GEOMETRY
        )
        surface = decoder.forward(
            dlvsqs=decoded.dlvsqs,
            forward_vols=decoded.forward_vols,
            eta=decoded.eta,
        )
        decoded_calls = decoder.price_calls(
            surface,
            expiries_y=torch.as_tensor(
                np.repeat(expiries, len(strikes)), dtype=torch.float64
            ),
            strikes=torch.as_tensor(
                np.tile(strikes, len(expiries)), dtype=torch.float64
            ),
        ).numpy().reshape(source_calls.shape)
    if not np.all(np.isfinite(decoded_calls)):
        raise FloatingPointError("Decoded DYSANOS prices are non-finite")
    return {
        "date": geometry_row["date"].isoformat(),
        "support_strikes": support,
        "source_call_prices": source_calls,
        "fitted_call_prices": fitted_calls,
        "decoded_call_prices": decoded_calls,
        "densities": densities,
        "dlvsqs": dlvsqs,
        "forward_vols": forward_vols,
        "eta_front": front_eta,
        "eta_term": later_eta,
        "raw_u": raw_state,
        "fit_max_abs_core": float(np.max(np.abs(fitted_calls - source_calls)[core])),
        "fit_mean_abs_core": float(np.mean(np.abs(fitted_calls - source_calls)[core])),
        "fit_max_weighted_error": float(np.max(np.abs(fitted_calls - source_calls) / fit_half_spreads)),
        "decode_max_abs_core": float(np.max(np.abs(decoded_calls - source_calls)[core])),
        "decode_max_weighted_error": float(np.max(np.abs(decoded_calls - source_calls) / fit_half_spreads)),
        "variance_increments_repaired": repaired,
    }


def _fit_and_save(
    geometry_row: dict,
    state_row: dict,
    output: str,
    timeout_seconds: float,
) -> tuple[str, float, float]:
    started = perf_counter()
    result = fit_day(geometry_row, state_row, timeout_seconds=timeout_seconds)
    destination = Path(output)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **result)
    os.replace(temporary, destination)
    return result["date"], result["fit_max_abs_core"], perf_counter() - started


def run(
    source: Path,
    output: Path,
    *,
    workers: int,
    timeout_seconds: float,
    allowed_dates: set[str] | None = None,
    excluded_dates: set[str] | None = None,
) -> None:
    """Fit all dates with resumable per-date outputs and flushed progress."""
    output.mkdir(parents=True, exist_ok=True)
    geometry_rows = pl.read_parquet(source / "sanos_geometry.parquet").to_dicts()
    state_rows = pl.read_parquet(source / "sanos_state.parquet").to_dicts()
    if len(geometry_rows) != len(state_rows):
        raise ValueError("SANOS geometry and state date counts differ")
    todo = [
        (geometry_row, state_row, str(output / f"{geometry_row['date'].isoformat()}.npz"))
        for geometry_row, state_row in zip(geometry_rows, state_rows, strict=True)
        if allowed_dates is None or geometry_row["date"].isoformat() in allowed_dates
        if excluded_dates is None or geometry_row["date"].isoformat() not in excluded_dates
        if not (output / f"{geometry_row['date'].isoformat()}.npz").exists()
    ]
    started = perf_counter()
    completed = 0
    failures: list[dict[str, str]] = []
    progress_path = output / (
        "progress.json" if allowed_dates is None else "progress-retry.json"
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fit_and_save, *arguments, timeout_seconds): arguments[0]["date"]
            for arguments in todo
        }
        for future in as_completed(futures):
            completed += 1
            try:
                day, error, elapsed = future.result()
                print(f"{completed}/{len(todo)} {day} max_core={error:.3g} seconds={elapsed:.1f}", flush=True)
            except Exception as exception:
                day = futures[future].isoformat()
                failures.append({"date": day, "error": repr(exception)})
                print(f"{completed}/{len(todo)} {day} FAILED {exception!r}", flush=True)
            if completed % 25 == 0:
                progress_path.write_text(json.dumps({
                    "completed": completed,
                    "total": len(todo),
                    "elapsed_seconds": perf_counter() - started,
                    "failures": failures,
                }, indent=2))
    progress_path.write_text(json.dumps({
        "completed": completed,
        "total": len(todo),
        "elapsed_seconds": perf_counter() - started,
        "failures": failures,
    }, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} SANOS regridding dates failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--dates-file", type=Path)
    parser.add_argument("--exclude-dates-file", type=Path)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    allowed_dates = (
        set(arguments.dates_file.read_text().splitlines())
        if arguments.dates_file is not None
        else None
    )
    excluded_dates = (
        set(arguments.exclude_dates_file.read_text().splitlines())
        if arguments.exclude_dates_file is not None
        else None
    )
    run(
        arguments.source,
        arguments.output,
        workers=arguments.workers,
        timeout_seconds=arguments.timeout_seconds,
        allowed_dates=allowed_dates,
        excluded_dates=excluded_dates,
    )


if __name__ == "__main__":
    main()
