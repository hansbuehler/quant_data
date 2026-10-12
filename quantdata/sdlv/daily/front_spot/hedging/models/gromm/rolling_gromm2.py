"""Leakage-free rolling artifacts for the 200-strike GROMM2 market model."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import h5py
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.calibration import (
    DlvCalibrationConfig,
    fit_affine_history,
    prepare_synthetic_books,
)
from quantdata.gromm2.estimation import estimate_models
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
from quantdata.hedging.data.public_dataset import PublicHedgingDataset
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
)

ROLLING_GROMM2_SCHEMA = "hedging-rolling-gromm2-v1"


@version("0.1", dependencies=[Gromm2GeometryConfig, DlvCalibrationConfig])
@dataclass(frozen=True, kw_only=True)
class RollingGromm2Config:
    """Precommitted market-model and rolling-estimation specification."""

    window_transitions: int = 252
    factor_count: int = 16
    substeps: int = 8
    minimum_inside_spread_percent: float = 80.0
    minimum_physical_projection_fraction: float = 0.5
    maximum_daily_risk_price_norm: float = 3.0
    geometry: Gromm2GeometryConfig = Gromm2GeometryConfig(num_inner_strikes=200)
    calibration: DlvCalibrationConfig = DlvCalibrationConfig(rank=200)

    def __post_init__(self) -> None:
        verify_inp(
            self.window_transitions >= self.calibration.rank + 1,
            lambda: "rolling window must contain more transitions than affine rank",
        )
        verify_inp(
            self.factor_count == 16,
            lambda: "the hedging benchmark requires 16 GROMM2 factors",
        )
        verify_inp(
            self.substeps == 8,
            lambda: "the hedging benchmark requires eight Brownian substeps",
        )
        verify_inp(
            self.geometry.num_inner_strikes == 200,
            lambda: "the accepted GROMM2 market model has 200 interior strikes",
        )
        verify_inp(
            self.calibration.rank == 200,
            lambda: "the accepted GROMM2 affine market fit has rank 200",
        )
        verify_inp(
            0.0 < self.minimum_inside_spread_percent <= 100.0,
            lambda: "inside-spread threshold must lie in (0,100]",
        )
        verify_inp(
            0.0 <= self.minimum_physical_projection_fraction <= 1.0,
            lambda: "physical projection threshold must lie in [0,1]",
        )
        verify_inp(
            np.isfinite(self.maximum_daily_risk_price_norm)
            and self.maximum_daily_risk_price_norm > 0.0,
            lambda: "daily physical risk-price norm threshold must be positive",
        )


@dataclass(frozen=True, kw_only=True)
class RollingGromm2Paths:
    """Stable file layout for one dated rolling GROMM2 fit."""

    directory: Path
    source_history: Path
    prepared_books: Path
    affine_history: Path
    affine_checkpoint: Path
    dynamics: Path
    manifest: Path


@dataclass(frozen=True, kw_only=True)
class RollingGromm2Artifacts:
    """Validated complete rolling GROMM2 market and dynamics artifacts."""

    paths: RollingGromm2Paths
    as_of_date: date
    training_dates: tuple[date, ...]
    percent_inside_spread: float
    explained_variance_ratio: float
    physical_projection_fraction: float
    maximum_daily_risk_price_norm: float
    dataset_manifest_sha256: str


def rolling_gromm2_paths(root: Path, *, as_of_date: date) -> RollingGromm2Paths:
    """Return the project-owned layout for one dated fit."""
    directory = Path(root) / as_of_date.isoformat()
    return RollingGromm2Paths(
        directory=directory,
        source_history=directory / "source-history.h5",
        prepared_books=directory / "prepared-books.h5",
        affine_history=directory / "affine-200-history.h5",
        affine_checkpoint=directory / "affine-200-checkpoint.npz",
        dynamics=directory / "rank16-ar1-dynamics.npz",
        manifest=directory / "manifest.json",
    )


def select_rolling_public_days(
    days: list[PublicSanosDay],
    *,
    as_of_date: date,
    window_transitions: int,
) -> list[PublicSanosDay]:
    """Select exactly one trailing state more than the transition count."""
    eligible = [day for day in days if day.trading_date <= as_of_date]
    required = window_transitions + 1
    if len(eligible) < required:
        raise ValueError(
            f"{as_of_date}: need {required} public states, found {len(eligible)}"
        )
    selected = eligible[-required:]
    dates = [day.trading_date for day in selected]
    if dates != sorted(set(dates)) or dates[-1] != as_of_date:
        raise ValueError("rolling GROMM2 dates must be unique and end at as-of")
    return selected


def prepare_rolling_gromm2(
    *,
    dataset: Path,
    artifact_root: Path,
    as_of_date: date,
    config: RollingGromm2Config = RollingGromm2Config(),
) -> RollingGromm2Paths:
    """Prepare leakage-free history and synthetic books on CPU."""
    paths = rolling_gromm2_paths(artifact_root, as_of_date=as_of_date)
    paths.directory.mkdir(parents=True, exist_ok=True)
    public_dataset = PublicHedgingDataset(Path(dataset))
    selected = select_rolling_public_days(
        load_public_sanos_days(public_dataset.directory),
        as_of_date=as_of_date,
        window_transitions=config.window_transitions,
    )
    _write_source_history(
        days=selected,
        dataset=public_dataset.directory,
        output=paths.source_history,
        geometry=config.geometry,
    )
    prepare_synthetic_books(
        source_history=paths.source_history,
        dataset=public_dataset.directory,
        output=paths.prepared_books,
    )
    _write_manifest(
        paths,
        {
            "schema": ROLLING_GROMM2_SCHEMA,
            "status": "prepared",
            "as_of_date": as_of_date.isoformat(),
            "training_start_date": selected[0].trading_date.isoformat(),
            "training_end_date": selected[-1].trading_date.isoformat(),
            "state_count": len(selected),
            "transition_count": len(selected) - 1,
            "dataset_manifest_sha256": file_sha256(
                public_dataset.directory / "manifest.json"
            ),
            "config": _config_snapshot(config),
            "source_history_sha256": file_sha256(paths.source_history),
            "prepared_books_sha256": file_sha256(paths.prepared_books),
        },
    )
    return paths


def fit_rolling_gromm2_market(
    *,
    dataset: Path,
    artifact_root: Path,
    as_of_date: date,
    config: RollingGromm2Config = RollingGromm2Config(),
) -> RollingGromm2Artifacts:
    """Run the managed-GPU affine fit, then estimate rolling AR(1) dynamics."""
    paths = rolling_gromm2_paths(artifact_root, as_of_date=as_of_date)
    dataset_manifest_sha256 = file_sha256(Path(dataset) / "manifest.json")
    prepared = _read_manifest(paths.manifest)
    _validate_prepared_manifest(
        prepared,
        paths=paths,
        as_of_date=as_of_date,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    fit_affine_history(
        prepared_books=paths.prepared_books,
        output=paths.affine_history,
        checkpoint=paths.affine_checkpoint,
        config=config.calibration,
    )
    report = estimate_models(
        history_path=paths.affine_history,
        dataset=Path(dataset),
        output=paths.dynamics,
        factor_count=config.factor_count,
    )
    with h5py.File(paths.affine_history, "r") as stream:
        percent_inside = float(stream.attrs["percent_inside_spread"])
    if percent_inside < config.minimum_inside_spread_percent:
        raise RuntimeError(
            "rolling GROMM2 market fit failed the accepted inside-spread gate: "
            f"{percent_inside:.3f}% < {config.minimum_inside_spread_percent:.3f}%"
        )
    physical_projection_fraction = float(report["physical_projection_fraction"])
    maximum_daily_risk_price_norm = _maximum_daily_risk_price_norm(paths.dynamics)
    _validate_physical_fit(
        physical_projection_fraction=physical_projection_fraction,
        maximum_daily_risk_price_norm=maximum_daily_risk_price_norm,
        config=config,
    )
    complete = {
        **prepared,
        "status": "complete",
        "affine_history_sha256": file_sha256(paths.affine_history),
        "dynamics_sha256": file_sha256(paths.dynamics),
        "dynamics_report_sha256": file_sha256(paths.dynamics.with_suffix(".json")),
        "percent_inside_spread": percent_inside,
        "explained_variance_ratio": float(report["explained_variance_ratio"]),
        "physical_projection_fraction": physical_projection_fraction,
        "maximum_daily_risk_price_norm": maximum_daily_risk_price_norm,
    }
    _write_manifest(paths, complete)
    return load_rolling_gromm2(
        artifact_root,
        as_of_date=as_of_date,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def load_rolling_gromm2(
    artifact_root: Path,
    *,
    as_of_date: date,
    config: RollingGromm2Config = RollingGromm2Config(),
    dataset_manifest_sha256: str,
) -> RollingGromm2Artifacts:
    """Load one complete fit and reject future-contaminated or stale artifacts."""
    paths = rolling_gromm2_paths(artifact_root, as_of_date=as_of_date)
    manifest = _read_manifest(paths.manifest)
    _validate_prepared_manifest(
        manifest,
        paths=paths,
        as_of_date=as_of_date,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    if manifest.get("status") != "complete":
        raise ValueError("rolling GROMM2 artifact is not complete")
    expected_hashes = {
        "affine_history_sha256": paths.affine_history,
        "dynamics_sha256": paths.dynamics,
        "dynamics_report_sha256": paths.dynamics.with_suffix(".json"),
    }
    for field, path in expected_hashes.items():
        if manifest.get(field) != file_sha256(path):
            raise ValueError(f"rolling GROMM2 {field} does not match {path}")
    with h5py.File(paths.affine_history, "r") as stream:
        if stream.attrs.get("schema") != "gromm2-affine-dlv-history-v1":
            raise ValueError("unexpected rolling GROMM2 affine-history schema")
        dates = tuple(
            date.fromisoformat(value)
            for value in np.asarray(stream["dates"]).astype("U10").tolist()
        )
        nms_nodes = np.asarray(stream["nms_nodes"], dtype=np.float64)
        percent_inside = float(stream.attrs["percent_inside_spread"])
    if (
        len(nms_nodes) != config.geometry.num_inner_strikes
        or len(dates) != config.window_transitions + 1
        or dates[-1] != as_of_date
        or any(value > as_of_date for value in dates)
    ):
        raise ValueError("rolling GROMM2 affine history violates dated geometry")
    if percent_inside < config.minimum_inside_spread_percent:
        raise ValueError("rolling GROMM2 affine history fails the market-fit gate")
    report = json.loads(paths.dynamics.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        report.get("schema") != "gromm2-estimate-v1"
        or report.get("factor_count") != config.factor_count
        or report.get("pair_count") != config.window_transitions
    ):
        raise ValueError("rolling GROMM2 dynamics report violates the specification")
    physical_projection_fraction = float(report["physical_projection_fraction"])
    maximum_daily_risk_price_norm = _maximum_daily_risk_price_norm(paths.dynamics)
    _validate_physical_fit(
        physical_projection_fraction=physical_projection_fraction,
        maximum_daily_risk_price_norm=maximum_daily_risk_price_norm,
        config=config,
    )
    return RollingGromm2Artifacts(
        paths=paths,
        as_of_date=as_of_date,
        training_dates=dates,
        percent_inside_spread=percent_inside,
        explained_variance_ratio=float(report["explained_variance_ratio"]),
        physical_projection_fraction=physical_projection_fraction,
        maximum_daily_risk_price_norm=maximum_daily_risk_price_norm,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def _write_source_history(
    *,
    days: list[PublicSanosDay],
    dataset: Path,
    output: Path,
    geometry: Gromm2GeometryConfig,
) -> None:
    reference_day = days[-1]
    expiries = geometry.model_business_dtes / 252.0
    raw_variances = public_sanos_atm_total_variances(reference_day, expiries)
    projected = project_total_variance(
        expiries,
        raw_variances,
        forward_volatility_floor=geometry.forward_volatility_floor,
    )
    supports = expanding_supports(projected, config=geometry)
    started = time.perf_counter()
    reconstructed = []
    for count, day in enumerate(days, start=1):
        reconstructed.append(
            reconstruct_public_day(
                day,
                supports=supports,
                model_expiries_y=expiries,
                geometry=geometry,
            )
        )
        if count == 1 or count % 10 == 0 or count == len(days):
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(days) - count) / count
            print(
                json.dumps(
                    {
                        "stage": "prepare-rolling-gromm2-history",
                        "completed_dates": count,
                        "total_dates": len(days),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    }
                ),
                flush=True,
            )
    temporary = _temporary_path(output)
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["schema"] = "gromm2-history-v1"
            stream.attrs["reference_date"] = reference_day.trading_date.isoformat()
            stream.attrs["geometry_config"] = json.dumps(
                asdict(geometry), sort_keys=True
            )
            stream.attrs["source_dataset"] = str(Path(dataset).resolve())
            stream.attrs["source_model"] = "published-sanos-parameters"
            stream.create_dataset(
                "dates",
                data=np.asarray(
                    [day.trading_date.isoformat() for day in days], dtype="S10"
                ),
            )
            stream.create_dataset("model_expiries_y", data=expiries)
            stream.create_dataset(
                "model_business_dtes", data=geometry.model_business_dtes
            )
            stream.create_dataset("nms_nodes", data=geometry.nms_nodes)
            stream.create_dataset("supports", data=supports)
            stream.create_dataset("transports", data=transport_tower(supports))
            stream.create_dataset("reference_raw_total_variances", data=raw_variances)
            stream.create_dataset(
                "reference_projected_total_variances", data=projected
            )
            terminal_expiries = np.asarray(
                [day.model_expiries_y[-1] for day in days], dtype=np.float64
            )
            stream.create_dataset(
                "source_terminal_business_dte", data=252.0 * terminal_expiries
            )
            stream.create_dataset(
                "terminal_extrapolated",
                data=expiries[None, :] > terminal_expiries[:, None],
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
                    data=np.asarray(
                        [getattr(value, name) for value in reconstructed],
                        dtype=np.float64,
                    ),
                )
            for name in (
                "raw_atm_total_variances",
                "projected_atm_total_variances",
                "dlvsq",
            ):
                stream.create_dataset(
                    name,
                    data=np.stack(
                        [np.asarray(getattr(value, name)) for value in reconstructed]
                    ).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            stream.flush()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_prepared_manifest(
    manifest: dict[str, object],
    *,
    paths: RollingGromm2Paths,
    as_of_date: date,
    config: RollingGromm2Config,
    dataset_manifest_sha256: str,
) -> None:
    if (
        manifest.get("schema") != ROLLING_GROMM2_SCHEMA
        or manifest.get("as_of_date") != as_of_date.isoformat()
        or manifest.get("training_end_date") != as_of_date.isoformat()
        or manifest.get("state_count") != config.window_transitions + 1
        or manifest.get("transition_count") != config.window_transitions
        or manifest.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or manifest.get("config") != _config_snapshot(config)
    ):
        raise ValueError("rolling GROMM2 manifest is stale or incompatible")
    for field, path in (
        ("source_history_sha256", paths.source_history),
        ("prepared_books_sha256", paths.prepared_books),
    ):
        if manifest.get(field) != file_sha256(path):
            raise ValueError(f"rolling GROMM2 {field} does not match {path}")


def _maximum_daily_risk_price_norm(path: Path) -> float:
    """Return the largest fitted one-day market-price-of-risk norm."""
    with np.load(path, allow_pickle=False) as values:
        risk_prices = np.asarray(
            values["physical_market_prices_of_risk"], dtype=np.float64
        )
    if risk_prices.ndim != 2 or not np.all(np.isfinite(risk_prices)):
        raise ValueError("rolling GROMM2 physical risk prices are invalid")
    return float(np.max(np.linalg.norm(risk_prices, axis=1) / np.sqrt(252.0)))


def _validate_physical_fit(
    *,
    physical_projection_fraction: float,
    maximum_daily_risk_price_norm: float,
    config: RollingGromm2Config,
) -> None:
    """Enforce the precommitted compatible-physical-drift acceptance gate."""
    if (
        not np.isfinite(physical_projection_fraction)
        or physical_projection_fraction < config.minimum_physical_projection_fraction
    ):
        raise ValueError(
            "rolling GROMM2 physical drift fails the projection gate: "
            f"{physical_projection_fraction:.6g} < "
            f"{config.minimum_physical_projection_fraction:.6g}"
        )
    if (
        not np.isfinite(maximum_daily_risk_price_norm)
        or maximum_daily_risk_price_norm > config.maximum_daily_risk_price_norm
    ):
        raise ValueError(
            "rolling GROMM2 physical drift fails the daily risk-price gate: "
            f"{maximum_daily_risk_price_norm:.6g} > "
            f"{config.maximum_daily_risk_price_norm:.6g}"
        )


def _config_snapshot(config: RollingGromm2Config) -> dict[str, object]:
    return json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rolling GROMM2 manifest must contain a JSON object")
    return payload


def _write_manifest(paths: RollingGromm2Paths, payload: dict[str, object]) -> None:
    paths.directory.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(paths.manifest)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(paths.manifest)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(filename)
