"""Build date-varying Heston call kernels for historical GROMM2 fitting."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from numpy.typing import NDArray

from quantdata.gromm2.correlation_aware_estimation import _kernel_volatility_panel
from quantdata.gromm2.heston_background_pilot import (
    HestonBackground,
    heston_call_kernels,
    write_call_kernel_grid_bundle,
)

FloatArray = NDArray[np.float64]


def _atomic_npy(path: Path, values: FloatArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def historical_heston_variances(
    *,
    dataset: Path,
    dates: NDArray[np.str_],
    business_dtes: FloatArray,
    final_annual_volatility: float,
) -> FloatArray:
    """Scale released one-day SANOS kernel variance to the origin Heston level."""
    panel_dates, kernel_volatilities = _kernel_volatility_panel(
        dataset_path=dataset,
        target_business_dtes=business_dtes,
    )
    if not np.array_equal(panel_dates, dates):
        raise ValueError("released SANOS dates do not match prepared-book dates")
    if final_annual_volatility <= 0.0:
        raise ValueError("final annual volatility must be positive")
    scale = final_annual_volatility / kernel_volatilities[-1, 0]
    return (scale * kernel_volatilities[:, 0]) ** 2


def initial_variance_grid(
    date_variances: FloatArray,
    *,
    point_count: int,
) -> FloatArray:
    """Return a volatility-spaced grid spanning every historical state."""
    values = np.asarray(date_variances, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 2
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
        or point_count < 3
    ):
        raise ValueError("invalid historical variances or grid size")
    volatility_grid = np.linspace(
        math.sqrt(float(np.min(values))),
        math.sqrt(float(np.max(values))),
        point_count,
    )
    return volatility_grid**2


def build_heston_kernel_grid(
    *,
    work_directory: Path,
    supports: FloatArray,
    strikes: FloatArray,
    expiries_y: FloatArray,
    background: HestonBackground,
    variance_grid: FloatArray,
) -> FloatArray:
    """Build a restartable grid of call kernels, one variance state at a time."""
    work_directory.mkdir(parents=True, exist_ok=True)
    kernels: list[FloatArray] = []
    started = perf_counter()
    for grid_index, variance in enumerate(variance_grid):
        path = work_directory / f"kernel-{grid_index:03d}.npy"
        if path.exists():
            kernel = np.load(path, allow_pickle=False)
            expected_shape = (len(expiries_y), strikes.shape[1], supports.shape[1])
            if kernel.shape != expected_shape or np.any(~np.isfinite(kernel)):
                raise ValueError(f"invalid cached kernel {path}")
        else:
            kernel = heston_call_kernels(
                supports,
                strikes,
                expiries_y,
                background=replace(background, initial_variance=float(variance)),
            )
            _atomic_npy(path, kernel)
        kernels.append(kernel)
        elapsed = perf_counter() - started
        rate = (grid_index + 1) / max(elapsed, 1.0e-9)
        eta = (len(variance_grid) - grid_index - 1) / max(rate, 1.0e-9)
        print(
            f"heston-kernel-grid state={grid_index + 1}/{len(variance_grid)} "
            f"variance={variance:.8g} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    return np.stack(kernels)


def build_historical_kernel_bundle(
    *,
    prepared_books: Path,
    dataset: Path,
    heston_pilot: Path,
    work_directory: Path,
    output: Path,
    grid_point_count: int,
) -> dict[str, object]:
    """Build and persist the historical date-varying Heston kernel bundle."""
    with h5py.File(prepared_books, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        supports = np.asarray(stream["supports"], dtype=np.float64)
        strikes = np.asarray(stream["target_strikes"], dtype=np.float64)
    pilot = json.loads(heston_pilot.read_text())
    if pilot.get("schema") != "gromm2-independent-heston-one-date-pilot-v1":
        raise ValueError(f"unexpected Heston pilot schema in {heston_pilot}")
    background = HestonBackground(**pilot["dynamics_adjusted_background"])
    date_variances = historical_heston_variances(
        dataset=dataset,
        dates=dates,
        business_dtes=business_dtes,
        final_annual_volatility=math.sqrt(background.initial_variance),
    )
    variance_grid = initial_variance_grid(
        date_variances,
        point_count=grid_point_count,
    )
    kernel_grid = build_heston_kernel_grid(
        work_directory=work_directory,
        supports=supports,
        strikes=strikes,
        expiries_y=expiries_y,
        background=background,
        variance_grid=variance_grid,
    )
    identity = write_call_kernel_grid_bundle(
        output,
        kernel_grid=kernel_grid,
        variance_grid=variance_grid,
        date_variances=date_variances,
        dates=dates,
        supports=supports,
        strikes=strikes,
        expiries_y=expiries_y,
        background=background,
        reference_date=str(pilot["trading_date"]),
    )
    report = {
        "schema": "gromm2-historical-heston-kernel-grid-v1",
        "identity": identity,
        "date_count": len(dates),
        "grid_point_count": len(variance_grid),
        "minimum_variance": float(variance_grid[0]),
        "maximum_variance": float(variance_grid[-1]),
        "minimum_annual_volatility": float(math.sqrt(variance_grid[0])),
        "maximum_annual_volatility": float(math.sqrt(variance_grid[-1])),
        "output": str(output.resolve()),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def validate_historical_kernel_bundle(
    *,
    prepared_books: Path,
    bundle: Path,
    output: Path,
) -> dict[str, object]:
    """Compare every interpolation interval midpoint with fresh FFT kernels."""
    with h5py.File(prepared_books, "r") as stream:
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        supports = np.asarray(stream["supports"], dtype=np.float64)
        strikes = np.asarray(stream["target_strikes"], dtype=np.float64)
    with np.load(bundle, allow_pickle=False) as values:
        if str(values["schema"].item()) != "gromm2-call-kernels-v2":
            raise ValueError(f"unexpected call-kernel schema in {bundle}")
        metadata = json.loads(str(values["metadata"].item()))
        variance_grid = np.asarray(values["variance_grid"], dtype=np.float64)
        kernel_grid = np.asarray(values["call_kernels"], dtype=np.float64)
    background = HestonBackground(**metadata["background"])
    interval_reports: list[dict[str, float]] = []
    started = perf_counter()
    for interval_index, (lower, upper) in enumerate(
        zip(variance_grid[:-1], variance_grid[1:], strict=True)
    ):
        midpoint = 0.5 * (lower + upper)
        exact = heston_call_kernels(
            supports,
            strikes,
            expiries_y,
            background=replace(background, initial_variance=float(midpoint)),
        )
        interpolated = 0.5 * (
            kernel_grid[interval_index] + kernel_grid[interval_index + 1]
        )
        error = interpolated - exact
        interval_reports.append(
            {
                "lower_variance": float(lower),
                "upper_variance": float(upper),
                "midpoint_variance": float(midpoint),
                "maximum_absolute_price_error": float(np.max(np.abs(error))),
                "root_mean_square_price_error": float(np.sqrt(np.mean(error**2))),
            }
        )
        elapsed = perf_counter() - started
        rate = (interval_index + 1) / max(elapsed, 1.0e-9)
        eta = (len(variance_grid) - interval_index - 2) / max(rate, 1.0e-9)
        print(
            f"heston-kernel-validation interval={interval_index + 1}/"
            f"{len(variance_grid) - 1} max_error={np.max(np.abs(error)):.3g} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    report = {
        "schema": "gromm2-historical-heston-kernel-validation-v1",
        "bundle": str(bundle.resolve()),
        "maximum_absolute_price_error": max(
            row["maximum_absolute_price_error"] for row in interval_reports
        ),
        "maximum_root_mean_square_price_error": max(
            row["root_mean_square_price_error"] for row in interval_reports
        ),
        "intervals": interval_reports,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--prepared-books", type=Path, required=True)
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--heston-pilot", type=Path, required=True)
    build.add_argument("--work-directory", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--grid-point-count", type=int, default=17)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--prepared-books", type=Path, required=True)
    validate.add_argument("--bundle", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build_historical_kernel_bundle(
            prepared_books=arguments.prepared_books,
            dataset=arguments.dataset,
            heston_pilot=arguments.heston_pilot,
            work_directory=arguments.work_directory,
            output=arguments.output,
            grid_point_count=arguments.grid_point_count,
        )
    else:
        validate_historical_kernel_bundle(
            prepared_books=arguments.prepared_books,
            bundle=arguments.bundle,
            output=arguments.output,
        )


if __name__ == "__main__":
    main()
