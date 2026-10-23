"""Build leakage-free six-month GROMM2 historical DLV states."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from quantdata.gromm2.geometry import (
    Gromm2GeometryConfig,
    expanding_supports,
    project_total_variance,
    transport_tower,
)
from quantdata.gromm2.market import (
    PublicSanosDay,
    load_public_sanos_days,
    public_sanos_atm_total_variances,
    reconstruct_public_day,
)


def _atomic_hdf5_path(path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temporary), path


def _reference_geometry(
    reference_day: PublicSanosDay,
    *,
    geometry: Gromm2GeometryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expiries = geometry.model_business_dtes / 252.0
    raw_variances = public_sanos_atm_total_variances(reference_day, expiries)
    projected = project_total_variance(
        expiries,
        raw_variances,
        forward_volatility_floor=geometry.forward_volatility_floor,
    )
    supports = expanding_supports(projected, config=geometry)
    return expiries, raw_variances, projected, supports


def _reconstruct_task(
    day: PublicSanosDay,
    supports: np.ndarray,
    expiries: np.ndarray,
    geometry: Gromm2GeometryConfig,
) -> tuple[str, dict[str, np.ndarray | float]]:
    result = reconstruct_public_day(
        day,
        supports=supports,
        model_expiries_y=expiries,
        geometry=geometry,
    )
    return day.trading_date.isoformat(), {
        "raw_atm_total_variances": result.raw_atm_total_variances,
        "projected_atm_total_variances": result.projected_atm_total_variances,
        "dlvsq": result.dlvsq,
        "floored_raw_dlvsq_fraction": result.floored_raw_dlvsq_fraction,
        "capped_raw_dlvsq_fraction": result.capped_raw_dlvsq_fraction,
        "clipped_projection_mass": result.clipped_projection_mass,
        "maximum_density_error": result.maximum_density_error,
        "root_mean_square_price_error": result.root_mean_square_price_error,
        "maximum_price_error": result.maximum_price_error,
    }


def build_history(
    *,
    dataset: Path,
    output: Path,
    geometry: Gromm2GeometryConfig,
    workers: int,
    limit: int | None = None,
) -> None:
    """Reconstruct published SANOS states and write an atomic HDF5 panel."""
    days = load_public_sanos_days(dataset)
    if limit is not None:
        days = days[-limit:]
    if not days:
        raise ValueError("public SANOS dataset contains no dates")
    dates = [day.trading_date for day in days]
    if dates != sorted(set(dates)):
        raise ValueError("public SANOS dates must be unique and sorted")
    reference_day = days[-1]
    reference_date = reference_day.trading_date
    expiries, reference_raw, reference_projected, supports = _reference_geometry(
        reference_day,
        geometry=geometry,
    )
    completed: dict[str, dict[str, np.ndarray | float]] = {}
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _reconstruct_task,
                day,
                supports,
                expiries,
                geometry,
            ): day.trading_date
            for day in days
        }
        for count, future in enumerate(as_completed(futures), start=1):
            trading_date = futures[future]
            key, result = future.result()
            completed[key] = result
            if count == 1 or count % 10 == 0 or count == len(dates):
                elapsed = perf_counter() - started
                rate = count / max(elapsed, 1.0e-9)
                remaining = (len(dates) - count) / max(rate, 1.0e-9)
                print(
                    f"history {count}/{len(dates)} date={trading_date} "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                    flush=True,
                )
    ordered = [completed[trading_date.isoformat()] for trading_date in dates]
    temporary, destination = _atomic_hdf5_path(output)
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["schema"] = "gromm2-history-v1"
            stream.attrs["reference_date"] = reference_date.isoformat()
            stream.attrs["geometry_config"] = json.dumps(asdict(geometry), sort_keys=True)
            stream.attrs["source_dataset"] = str(dataset.resolve())
            stream.attrs["source_model"] = "published-sanos-parameters"
            stream.create_dataset(
                "dates",
                data=np.asarray([value.isoformat() for value in dates], dtype="S10"),
            )
            stream.create_dataset("model_expiries_y", data=expiries)
            stream.create_dataset("model_business_dtes", data=geometry.model_business_dtes)
            stream.create_dataset("nms_nodes", data=geometry.nms_nodes)
            stream.create_dataset("supports", data=supports)
            stream.create_dataset("transports", data=transport_tower(supports))
            stream.create_dataset("reference_raw_total_variances", data=reference_raw)
            stream.create_dataset(
                "reference_projected_total_variances", data=reference_projected
            )
            source_terminal_expiries = np.asarray(
                [day.model_expiries_y[-1] for day in days], dtype=np.float64
            )
            stream.create_dataset(
                "source_terminal_business_dte",
                data=252.0 * source_terminal_expiries,
            )
            stream.create_dataset(
                "terminal_extrapolated",
                data=expiries[None, :] > source_terminal_expiries[:, None],
            )
            for name in (
                "floored_raw_dlvsq_fraction",
                "capped_raw_dlvsq_fraction",
                "clipped_projection_mass",
                "maximum_density_error",
                "root_mean_square_price_error",
                "maximum_price_error",
            ):
                stream.create_dataset(
                    name,
                    data=np.asarray([result[name] for result in ordered], dtype=np.float64),
                )
            for name in (
                "raw_atm_total_variances",
                "projected_atm_total_variances",
                "dlvsq",
            ):
                stream.create_dataset(
                    name,
                    data=np.stack([np.asarray(result[name]) for result in ordered]).astype(
                        np.float32
                    ),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            stream.flush()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"wrote {destination} ({destination.stat().st_size / 2**20:.1f} MiB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-inner-strikes", type=int, default=200)
    parser.add_argument("--backbone-annual-volatility", type=float, default=0.05)
    arguments = parser.parse_args()
    if arguments.workers <= 0:
        raise ValueError("--workers must be positive")
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("--limit must be positive")
    build_history(
        dataset=arguments.dataset,
        output=arguments.output,
        geometry=Gromm2GeometryConfig(
            num_inner_strikes=arguments.num_inner_strikes,
            backbone_annual_volatility=arguments.backbone_annual_volatility,
        ),
        workers=arguments.workers,
        limit=arguments.limit,
    )


if __name__ == "__main__":
    main()
