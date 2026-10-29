"""Leakage-free rolling artifacts for the 200-strike GROMM2 market model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import h5py
import jax
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.gromm2.calibration import (
    DlvCalibrationConfig,
    fit_affine_history,
    prepare_synthetic_books,
)
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
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_dynamics import (
    GROMM2_DYNAMICS_SCHEMA,
    MINIMUM_PHYSICAL_VALIDATION_GAIN_T,
    MINIMUM_PHYSICAL_VALIDATION_IMPROVEMENT,
    PHYSICAL_SELECTION_RULE,
    estimate_rolling_gromm2_dynamics,
)

ROLLING_GROMM2_SCHEMA = "hedging-rolling-gromm2-v5"
LEGACY_ROLLING_GROMM2_SCHEMA = "hedging-rolling-gromm2-v4"
COMPATIBLE_ROLLING_GROMM2_V5_SOURCE_IDENTITY = (
    "dbaaff6ac9bc36c8e08b202e56d5fc6ca6e9804cca1843aa027ca68d6a2df5bf"
)
COMPATIBLE_ROLLING_GROMM2_V5_SOURCE_IDENTITIES = frozenset({
    COMPATIBLE_ROLLING_GROMM2_V5_SOURCE_IDENTITY,
    "bae743d2dc838740cca7fbd0c9570864a618d58586ba160491d940c7d798855e",
    "c6e246c50d251f2a508b7cd6d067c563a4297d58f6acdc6f4be6b37516370f27",
    "4e8748d49d8940c75c7e617f57b776aab80e34b791d89b3b6502a95629097b3f",
})
LEGACY_PHYSICAL_PROJECTION_FRACTION = 0.5
_CALIBRATION_EPOCH_MULTIPLIERS = (1, 2, 4)
_ROLLING_GROMM2_SOURCE_FILES = (
    "biz_calendar.py",
    "gromm2/bridge.py",
    "gromm2/calibration.py",
    "gromm2/dlv.py",
    "gromm2/estimation.py",
    "gromm2/geometry.py",
    "gromm2/market.py",
    "gromm2/simulation.py",
    "hedging/data/pricing.py",
    "hedging/data/public_dataset.py",
    "sdlv/daily/front_spot/hedging/data/market_panel.py",
    "sdlv/daily/front_spot/hedging/data/rolling_calibration.py",
    "sdlv/daily/front_spot/hedging/data/rolling_factors.py",
    "sdlv/daily/front_spot/hedging/models/gromm/rolling_dynamics.py",
    "sdlv/daily/front_spot/hedging/models/gromm/rolling_gromm2.py",
)


@version("0.6", dependencies=[Gromm2GeometryConfig, DlvCalibrationConfig])
@dataclass(frozen=True, kw_only=True)
class RollingGromm2Config:
    """Precommitted market-model and rolling-estimation specification."""

    window_transitions: int = 252
    factor_count: int = 16
    substeps: int = 8
    minimum_inside_spread_percent: float = 80.0
    maximum_daily_risk_price_norm: float = 3.0
    geometry: Gromm2GeometryConfig = Gromm2GeometryConfig(num_inner_strikes=200)
    calibration: DlvCalibrationConfig = DlvCalibrationConfig(rank=200, epochs=2_000)

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
    transition_origin_dates: tuple[date, ...]
    transition_endpoint_dates: tuple[date, ...]
    calibration_epochs_completed: int
    calibration_retry_count: int
    percent_inside_spread: float
    explained_variance_ratio: float
    physical_projection_fraction: float
    candidate_physical_projection_fraction: float
    baseline_physical_projection_fraction: float
    maximum_daily_risk_price_norm: float
    candidate_daily_risk_price_norm: float
    candidate_physical_accepted: bool
    physical_fallback_used: bool
    physical_selected_model: str
    physical_accepted: bool
    dataset_manifest_sha256: str


@dataclass(frozen=True, kw_only=True)
class _ValidatedPhysicalSelection:
    """Recomputed physical-selection evidence from one dynamics artifact."""

    candidate_achievable_fraction: float
    baseline_achievable_fraction: float
    selected_achievable_fraction: float
    candidate_daily_risk_price_norm: float
    baseline_daily_risk_price_norm: float
    selected_daily_risk_price_norm: float
    candidate_accepted: bool
    baseline_accepted: bool
    selected_model: str
    fallback_used: bool
    selected_physical_accepted: bool


def rolling_gromm2_source_identity() -> str:
    """Hash all project sources that define one dated rolling artifact."""
    quantdata_directory = Path(__file__).resolve().parents[6]
    digest = hashlib.sha256()
    for relative_path in _ROLLING_GROMM2_SOURCE_FILES:
        path = quantdata_directory / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    transition_pairs: tuple[tuple[date, date], ...] | None = None,
) -> list[PublicSanosDay]:
    """Select the states supporting an exact trailing one-session-pair window."""
    if transition_pairs is not None:
        if len(transition_pairs) != window_transitions:
            raise ValueError("rolling GROMM2 transition-pair count is wrong")
        origin_dates = tuple(origin for origin, _ in transition_pairs)
        endpoint_dates = tuple(endpoint for _, endpoint in transition_pairs)
        if (
            origin_dates != tuple(sorted(set(origin_dates)))
            or endpoint_dates != tuple(sorted(set(endpoint_dates)))
            or endpoint_dates[-1] > as_of_date
            or any(origin >= endpoint for origin, endpoint in transition_pairs)
        ):
            raise ValueError("rolling GROMM2 transition pairs are not chronological")
        by_date = {day.trading_date: day for day in days}
        required_dates = tuple(
            sorted(set(origin_dates) | set(endpoint_dates) | {as_of_date})
        )
        missing = tuple(value for value in required_dates if value not in by_date)
        if missing:
            raise ValueError(
                "rolling GROMM2 public states are missing transition dates: "
                + ", ".join(value.isoformat() for value in missing[:3])
            )
        selected = [by_date[value] for value in required_dates]
        if selected[-1].trading_date != as_of_date:
            raise ValueError("rolling GROMM2 dates must end at as-of")
        return selected
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
    panel = PublicHedgingPanel(public_dataset.directory)
    origin_dates = panel.prior_one_day_origins(
        as_of_date,
        count=config.window_transitions,
    )
    endpoint_dates = tuple(
        panel.next_one_session_date(origin_date) for origin_date in origin_dates
    )
    transition_pairs = tuple(zip(origin_dates, endpoint_dates, strict=True))
    selected = select_rolling_public_days(
        load_public_sanos_days(public_dataset.directory),
        as_of_date=as_of_date,
        window_transitions=config.window_transitions,
        transition_pairs=transition_pairs,
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
    _invalidate_fitted_outputs(paths)
    _write_manifest(
        paths,
        {
            "schema": ROLLING_GROMM2_SCHEMA,
            "status": "prepared",
            "as_of_date": as_of_date.isoformat(),
            "training_start_date": origin_dates[0].isoformat(),
            "training_end_date": as_of_date.isoformat(),
            "state_count": len(selected),
            "transition_count": len(origin_dates),
            "training_origin_dates": [value.isoformat() for value in origin_dates],
            "training_endpoint_dates": [
                value.isoformat() for value in endpoint_dates
            ],
            "dataset_manifest_sha256": file_sha256(
                public_dataset.directory / "manifest.json"
            ),
            "source_identity": rolling_gromm2_source_identity(),
            "config": _config_snapshot(config),
            "source_history_sha256": file_sha256(paths.source_history),
            "prepared_books_sha256": file_sha256(paths.prepared_books),
        },
    )
    return paths


def load_prepared_rolling_gromm2(
    artifact_root: Path,
    *,
    as_of_date: date,
    config: RollingGromm2Config = RollingGromm2Config(),
    dataset_manifest_sha256: str,
) -> RollingGromm2Paths:
    """Load and authenticate the CPU-prepared portion of one rolling fit."""
    paths = rolling_gromm2_paths(artifact_root, as_of_date=as_of_date)
    manifest = _read_manifest(paths.manifest)
    if manifest.get("status") not in {"prepared", "complete"}:
        raise ValueError("rolling GROMM2 artifact is not prepared")
    _validate_prepared_manifest(
        manifest,
        paths=paths,
        as_of_date=as_of_date,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    return paths


def migrate_legacy_rolling_gromm2_affine_fit(
    *,
    dataset: Path,
    source_artifact_root: Path,
    target_artifact_root: Path,
    as_of_date: date,
    source_manifest_sha256: str,
    source_identity: str,
    config: RollingGromm2Config = RollingGromm2Config(),
) -> RollingGromm2Paths:
    """Seed a current prepared artifact from an authenticated legacy affine fit.

    The source history, prepared books, and accepted affine history are linked
    read-only by convention. The legacy dynamics and their physical-selection
    decision are deliberately discarded and recomputed by the current code.
    """
    source_paths = rolling_gromm2_paths(
        source_artifact_root,
        as_of_date=as_of_date,
    )
    target_paths = rolling_gromm2_paths(
        target_artifact_root,
        as_of_date=as_of_date,
    )
    if source_paths.directory.resolve() == target_paths.directory.resolve():
        raise ValueError("legacy GROMM2 migration requires a distinct target root")
    dataset_manifest_sha256 = file_sha256(Path(dataset) / "manifest.json")
    if target_paths.manifest.is_file():
        try:
            return load_prepared_rolling_gromm2(
                target_artifact_root,
                as_of_date=as_of_date,
                config=config,
                dataset_manifest_sha256=dataset_manifest_sha256,
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"current rolling GROMM2 target is incompatible: {target_paths.directory}"
            ) from error
    if file_sha256(source_paths.manifest) != source_manifest_sha256:
        raise ValueError("legacy rolling GROMM2 manifest hash is inconsistent")
    source_manifest = _read_manifest(source_paths.manifest)
    source_schema = source_manifest.get("schema")
    legacy_config = _config_snapshot(config)
    if source_schema == LEGACY_ROLLING_GROMM2_SCHEMA:
        legacy_config["minimum_physical_projection_fraction"] = (
            LEGACY_PHYSICAL_PROJECTION_FRACTION
        )
    elif source_schema != ROLLING_GROMM2_SCHEMA:
        raise ValueError("unsupported rolling GROMM2 migration source schema")
    source_config = _normalize_legacy_config(source_manifest.get("config"))
    origin_dates, endpoint_dates = _manifest_transition_dates(
        source_manifest,
        window_transitions=config.window_transitions,
    )
    expected_state_dates = tuple(
        sorted(set(origin_dates) | set(endpoint_dates) | {as_of_date})
    )
    if (
        source_manifest.get("status") != "complete"
        or (
            source_schema == ROLLING_GROMM2_SCHEMA
            and source_identity not in COMPATIBLE_ROLLING_GROMM2_V5_SOURCE_IDENTITIES
        )
        or source_manifest.get("as_of_date") != as_of_date.isoformat()
        or source_manifest.get("training_end_date") != as_of_date.isoformat()
        or source_manifest.get("training_start_date")
        != origin_dates[0].isoformat()
        or source_manifest.get("state_count") != len(expected_state_dates)
        or source_manifest.get("transition_count") != config.window_transitions
        or source_manifest.get("dataset_manifest_sha256")
        != dataset_manifest_sha256
        or source_manifest.get("source_identity") != source_identity
        or source_config != legacy_config
    ):
        raise ValueError("legacy rolling GROMM2 manifest is incompatible")
    source_files = {
        "source_history_sha256": source_paths.source_history,
        "prepared_books_sha256": source_paths.prepared_books,
        "affine_history_sha256": source_paths.affine_history,
    }
    authenticated_sha256_by_path: dict[Path, str] = {}
    for field, path in source_files.items():
        expected_sha256 = source_manifest.get(field)
        if not isinstance(expected_sha256, str) or file_sha256(path) != expected_sha256:
            raise ValueError(f"legacy rolling GROMM2 {field} is inconsistent")
        authenticated_sha256_by_path[path] = expected_sha256
    with h5py.File(source_paths.affine_history, "r") as stream:
        if stream.attrs.get("schema") != "gromm2-affine-dlv-history-v1":
            raise ValueError("unexpected legacy rolling GROMM2 affine schema")
        dates = tuple(
            date.fromisoformat(value)
            for value in np.asarray(stream["dates"]).astype("U10").tolist()
        )
        nms_nodes = np.asarray(stream["nms_nodes"], dtype=np.float64)
        percent_inside = float(stream.attrs["percent_inside_spread"])
        calibration_payload = json.loads(str(stream.attrs["calibration_config"]))
    (
        calibration_epochs_completed,
        calibration_retry_count,
        fitted_calibration,
    ) = _validate_affine_calibration_payload(
        calibration_payload,
        baseline=config.calibration,
    )
    manifest_percent_inside = source_manifest.get("percent_inside_spread")
    manifest_calibration = source_manifest.get("affine_calibration_config")
    if (
        dates != expected_state_dates
        or len(nms_nodes) != config.geometry.num_inner_strikes
        or percent_inside < config.minimum_inside_spread_percent
        or isinstance(manifest_percent_inside, bool)
        or not isinstance(manifest_percent_inside, int | float)
        or not math.isclose(
            float(manifest_percent_inside),
            percent_inside,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or source_manifest.get("calibration_epochs_completed")
        != calibration_epochs_completed
        or source_manifest.get("calibration_retry_count")
        != calibration_retry_count
        or _normalize_legacy_calibration(manifest_calibration)
        != json.loads(json.dumps(asdict(fitted_calibration), sort_keys=True))
    ):
        raise ValueError("legacy rolling GROMM2 affine fit violates the current gate")
    target_paths.directory.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (source_paths.source_history, target_paths.source_history),
        (source_paths.prepared_books, target_paths.prepared_books),
        (source_paths.affine_history, target_paths.affine_history),
    ):
        _link_authenticated_file(
            source=source,
            target=target,
            expected_sha256=authenticated_sha256_by_path[source],
        )
    _write_manifest(
        target_paths,
        {
            "schema": ROLLING_GROMM2_SCHEMA,
            "status": "prepared",
            "as_of_date": as_of_date.isoformat(),
            "training_start_date": origin_dates[0].isoformat(),
            "training_end_date": as_of_date.isoformat(),
            "state_count": len(expected_state_dates),
            "transition_count": len(origin_dates),
            "training_origin_dates": [value.isoformat() for value in origin_dates],
            "training_endpoint_dates": [
                value.isoformat() for value in endpoint_dates
            ],
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "source_identity": rolling_gromm2_source_identity(),
            "config": _config_snapshot(config),
            "source_history_sha256": file_sha256(target_paths.source_history),
            "prepared_books_sha256": file_sha256(target_paths.prepared_books),
            "legacy_affine_migration": {
                "source_artifact_root": str(Path(source_artifact_root).resolve()),
                "source_manifest_sha256": source_manifest_sha256,
                "source_identity": source_identity,
                "affine_history_sha256": file_sha256(target_paths.affine_history),
            },
        },
    )
    return load_prepared_rolling_gromm2(
        target_artifact_root,
        as_of_date=as_of_date,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


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
    exact_reuse = prepared.get("exact_input_affine_reuse")
    if exact_reuse is not None:
        if (
            not isinstance(exact_reuse, dict)
            or file_sha256(paths.affine_history)
            != exact_reuse.get("affine_history_sha256")
            or (
                exact_reuse.get("affine_checkpoint_sha256") is not None
                and file_sha256(paths.affine_checkpoint)
                != exact_reuse["affine_checkpoint_sha256"]
            )
        ):
            raise ValueError("exact-input affine reuse provenance is invalid")
        previous_date = None
        previous_paths = None
    else:
        origin_dates, _ = _manifest_transition_dates(
            prepared,
            window_transitions=config.window_transitions,
        )
        previous_date = origin_dates[-1]
        panel = PublicHedgingPanel(Path(dataset))
        try:
            panel.prior_one_day_origins(previous_date, count=config.window_transitions)
        except ValueError:
            previous_paths = None
        else:
            previous = load_rolling_gromm2(
                artifact_root,
                as_of_date=previous_date,
                config=config,
                dataset_manifest_sha256=dataset_manifest_sha256,
            )
            previous_paths = previous.paths
    (
        percent_inside,
        calibration_epochs_completed,
        calibration_retry_count,
        fitted_calibration,
    ) = _fit_affine_market_until_gate(
        paths=paths,
        calibration=config.calibration,
        minimum_inside_spread_percent=config.minimum_inside_spread_percent,
        warm_start_checkpoint=(
            previous_paths.affine_checkpoint if previous_paths is not None else None
        ),
        warm_start_prepared_books=(
            previous_paths.prepared_books if previous_paths is not None else None
        ),
    )
    with jax.enable_x64(True):
        origin_dates, endpoint_dates = _manifest_transition_dates(
            prepared,
            window_transitions=config.window_transitions,
        )
        report = estimate_rolling_gromm2_dynamics(
            history_path=paths.affine_history,
            dataset=Path(dataset),
            output=paths.dynamics,
            as_of_date=as_of_date,
            origin_dates=origin_dates,
            endpoint_dates=endpoint_dates,
            factor_count=config.factor_count,
            maximum_daily_risk_price_norm=(
                config.maximum_daily_risk_price_norm
            ),
        )
    physical_selection = _validate_physical_fit(
        report=report,
        dynamics_path=paths.dynamics,
        config=config,
    )
    complete = {
        **prepared,
        "prepared_source_identity": prepared["source_identity"],
        "source_identity": rolling_gromm2_source_identity(),
        "status": "complete",
        "affine_history_sha256": file_sha256(paths.affine_history),
        "affine_calibration_config": json.loads(
            json.dumps(asdict(fitted_calibration), sort_keys=True, allow_nan=False)
        ),
        "calibration_epochs_completed": calibration_epochs_completed,
        "calibration_retry_count": calibration_retry_count,
        "warm_start_date": (
            previous_date.isoformat() if previous_paths is not None else None
        ),
        "warm_start_checkpoint_sha256": (
            file_sha256(previous_paths.affine_checkpoint)
            if previous_paths is not None
            else None
        ),
        "dynamics_sha256": file_sha256(paths.dynamics),
        "dynamics_report_sha256": file_sha256(paths.dynamics.with_suffix(".json")),
        "percent_inside_spread": percent_inside,
        "explained_variance_ratio": float(report["explained_variance_ratio"]),
        "physical_projection_fraction": (
            physical_selection.selected_achievable_fraction
        ),
        "candidate_physical_projection_fraction": (
            physical_selection.candidate_achievable_fraction
        ),
        "baseline_physical_projection_fraction": (
            physical_selection.baseline_achievable_fraction
        ),
        "maximum_daily_risk_price_norm": (
            physical_selection.selected_daily_risk_price_norm
        ),
        "candidate_daily_risk_price_norm": (
            physical_selection.candidate_daily_risk_price_norm
        ),
        "candidate_physical_accepted": physical_selection.candidate_accepted,
        "physical_fallback_used": physical_selection.fallback_used,
        "physical_selected_model": physical_selection.selected_model,
        "physical_accepted": physical_selection.selected_physical_accepted,
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
        calibration_payload = json.loads(str(stream.attrs["calibration_config"]))
    (
        calibration_epochs_completed,
        calibration_retry_count,
        fitted_calibration,
    ) = _validate_affine_calibration_payload(
        calibration_payload,
        baseline=config.calibration,
    )
    origin_dates, endpoint_dates = _manifest_transition_dates(
        manifest,
        window_transitions=config.window_transitions,
    )
    expected_state_dates = tuple(
        sorted(set(origin_dates) | set(endpoint_dates) | {as_of_date})
    )
    if (
        len(nms_nodes) != config.geometry.num_inner_strikes
        or dates != expected_state_dates
        or expected_state_dates[-1] != as_of_date
    ):
        raise ValueError("rolling GROMM2 affine history violates dated geometry")
    if percent_inside < config.minimum_inside_spread_percent:
        raise ValueError("rolling GROMM2 affine history fails the market-fit gate")
    manifest_percent_inside = manifest.get("percent_inside_spread")
    if (
        isinstance(manifest_percent_inside, bool)
        or not isinstance(manifest_percent_inside, int | float)
        or not math.isclose(
            float(manifest_percent_inside),
            percent_inside,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or manifest.get(
            "calibration_epochs_completed",
            config.calibration.epochs,
        )
        != calibration_epochs_completed
        or manifest.get("calibration_retry_count", 0) != calibration_retry_count
    ):
        raise ValueError("rolling GROMM2 calibration evidence is inconsistent")
    manifest_calibration = manifest.get("affine_calibration_config")
    if manifest_calibration is not None and manifest_calibration != json.loads(
        json.dumps(asdict(fitted_calibration), sort_keys=True, allow_nan=False)
    ):
        raise ValueError("rolling GROMM2 fitted calibration config is inconsistent")
    report = json.loads(paths.dynamics.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        report.get("schema") != GROMM2_DYNAMICS_SCHEMA
        or report.get("factor_count") != config.factor_count
        or report.get("pair_count") != config.window_transitions
    ):
        raise ValueError("rolling GROMM2 dynamics report violates the specification")
    physical_selection = _validate_physical_fit(
        report=report,
        dynamics_path=paths.dynamics,
        config=config,
    )
    manifest_float_diagnostics = {
        "explained_variance_ratio": float(report["explained_variance_ratio"]),
        "physical_projection_fraction": (
            physical_selection.selected_achievable_fraction
        ),
        "candidate_physical_projection_fraction": (
            physical_selection.candidate_achievable_fraction
        ),
        "baseline_physical_projection_fraction": (
            physical_selection.baseline_achievable_fraction
        ),
        "maximum_daily_risk_price_norm": (
            physical_selection.selected_daily_risk_price_norm
        ),
        "candidate_daily_risk_price_norm": (
            physical_selection.candidate_daily_risk_price_norm
        ),
    }
    for field, expected in manifest_float_diagnostics.items():
        value = manifest.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isclose(
                float(value),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                f"rolling GROMM2 manifest {field} is inconsistent"
            )
    manifest_selection = {
        "candidate_physical_accepted": physical_selection.candidate_accepted,
        "physical_fallback_used": physical_selection.fallback_used,
        "physical_selected_model": physical_selection.selected_model,
        "physical_accepted": physical_selection.selected_physical_accepted,
    }
    for field, expected in manifest_selection.items():
        value = manifest.get(field)
        expected_type = bool if isinstance(expected, bool) else str
        if not isinstance(value, expected_type) or value != expected:
            raise ValueError(
                "rolling GROMM2 manifest physical selection is inconsistent"
            )
    return RollingGromm2Artifacts(
        paths=paths,
        as_of_date=as_of_date,
        training_dates=dates,
        transition_origin_dates=origin_dates,
        transition_endpoint_dates=endpoint_dates,
        calibration_epochs_completed=calibration_epochs_completed,
        calibration_retry_count=calibration_retry_count,
        percent_inside_spread=percent_inside,
        explained_variance_ratio=float(report["explained_variance_ratio"]),
        physical_projection_fraction=(
            physical_selection.selected_achievable_fraction
        ),
        candidate_physical_projection_fraction=(
            physical_selection.candidate_achievable_fraction
        ),
        baseline_physical_projection_fraction=(
            physical_selection.baseline_achievable_fraction
        ),
        maximum_daily_risk_price_norm=(
            physical_selection.selected_daily_risk_price_norm
        ),
        candidate_daily_risk_price_norm=(
            physical_selection.candidate_daily_risk_price_norm
        ),
        candidate_physical_accepted=physical_selection.candidate_accepted,
        physical_fallback_used=physical_selection.fallback_used,
        physical_selected_model=physical_selection.selected_model,
        physical_accepted=physical_selection.selected_physical_accepted,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def _fit_affine_market_until_gate(
    *,
    paths: RollingGromm2Paths,
    calibration: DlvCalibrationConfig,
    minimum_inside_spread_percent: float,
    warm_start_checkpoint: Path | None = None,
    warm_start_prepared_books: Path | None = None,
) -> tuple[float, int, int, DlvCalibrationConfig]:
    """Continue a borderline affine fit without weakening its market-fit gate."""
    last_percent_inside = float("nan")
    if paths.affine_history.is_file():
        with h5py.File(paths.affine_history, "r") as stream:
            if stream.attrs.get("schema") == "gromm2-affine-dlv-history-v1":
                percent_inside = float(stream.attrs["percent_inside_spread"])
                last_percent_inside = percent_inside
                payload = json.loads(str(stream.attrs["calibration_config"]))
                epochs, retries, fitted = _validate_affine_calibration_payload(
                    payload,
                    baseline=calibration,
                )
                if percent_inside >= minimum_inside_spread_percent:
                    return percent_inside, epochs, retries, fitted

    completed_epochs = _checkpoint_completed_epochs(paths.affine_checkpoint)
    last_calibration = calibration
    for retry_count, multiplier in enumerate(_CALIBRATION_EPOCH_MULTIPLIERS):
        epochs = calibration.epochs * multiplier
        if epochs < completed_epochs:
            continue
        fitted_calibration = replace(calibration, epochs=epochs)
        with jax.enable_x64(False):
            fit_affine_history(
                prepared_books=paths.prepared_books,
                output=paths.affine_history,
                checkpoint=paths.affine_checkpoint,
                config=fitted_calibration,
                warm_start_checkpoint=warm_start_checkpoint,
                warm_start_prepared_books=warm_start_prepared_books,
            )
        with h5py.File(paths.affine_history, "r") as stream:
            last_percent_inside = float(stream.attrs["percent_inside_spread"])
        last_calibration = fitted_calibration
        completed_epochs = epochs
        if last_percent_inside >= minimum_inside_spread_percent:
            return (
                last_percent_inside,
                completed_epochs,
                retry_count,
                fitted_calibration,
            )
    raise RuntimeError(
        "rolling GROMM2 market fit failed the accepted inside-spread gate after "
        f"{completed_epochs} epochs: {last_percent_inside:.3f}% < "
        f"{minimum_inside_spread_percent:.3f}% (last config "
        f"epochs={last_calibration.epochs})"
    )


def _checkpoint_completed_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with np.load(path, allow_pickle=False) as values:
        completed = int(values["completed_epochs"])
    if completed < 0:
        raise ValueError("rolling GROMM2 checkpoint has negative completed epochs")
    return completed


def _validate_affine_calibration_payload(
    payload: object,
    *,
    baseline: DlvCalibrationConfig,
) -> tuple[int, int, DlvCalibrationConfig]:
    if not isinstance(payload, dict):
        raise ValueError("rolling GROMM2 affine calibration config is invalid")
    allowed: list[tuple[int, DlvCalibrationConfig]] = []
    for retry_count, multiplier in enumerate(_CALIBRATION_EPOCH_MULTIPLIERS):
        candidate = replace(baseline, epochs=baseline.epochs * multiplier)
        candidate_payload = json.loads(json.dumps(asdict(candidate), sort_keys=True))
        if _normalize_legacy_calibration(payload) == candidate_payload:
            allowed.append((retry_count, candidate))
    if len(allowed) != 1:
        raise ValueError("rolling GROMM2 affine calibration config is incompatible")
    retry_count, fitted = allowed[0]
    return fitted.epochs, retry_count, fitted


def _normalize_legacy_calibration(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    normalized.setdefault("plateau_review_epochs", 0)
    return normalized


def _normalize_legacy_config(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    normalized["calibration"] = _normalize_legacy_calibration(
        payload.get("calibration")
    )
    return normalized


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
    parsed_origins, parsed_endpoints = _manifest_transition_dates(
        manifest,
        window_transitions=config.window_transitions,
    )
    unique_states = set(parsed_origins) | set(parsed_endpoints) | {as_of_date}
    if (
        manifest.get("schema") != ROLLING_GROMM2_SCHEMA
        or manifest.get("as_of_date") != as_of_date.isoformat()
        or manifest.get("training_end_date") != as_of_date.isoformat()
        or manifest.get("training_start_date") != parsed_origins[0].isoformat()
        or parsed_origins != tuple(sorted(set(parsed_origins)))
        or parsed_endpoints != tuple(sorted(set(parsed_endpoints)))
        or parsed_endpoints[-1] > as_of_date
        or any(origin >= endpoint for origin, endpoint in zip(
            parsed_origins,
            parsed_endpoints,
            strict=True,
        ))
        or manifest.get("state_count") != len(unique_states)
        or manifest.get("transition_count") != config.window_transitions
        or manifest.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or manifest.get("source_identity") not in {
            rolling_gromm2_source_identity(),
            "c6e246c50d251f2a508b7cd6d067c563a4297d58f6acdc6f4be6b37516370f27",
            "4e8748d49d8940c75c7e617f57b776aab80e34b791d89b3b6502a95629097b3f",
        }
        or manifest.get("config") != _config_snapshot(config)
    ):
        raise ValueError("rolling GROMM2 manifest is stale or incompatible")
    for field, path in (
        ("source_history_sha256", paths.source_history),
        ("prepared_books_sha256", paths.prepared_books),
    ):
        if manifest.get(field) != file_sha256(path):
            raise ValueError(f"rolling GROMM2 {field} does not match {path}")


def _invalidate_fitted_outputs(paths: RollingGromm2Paths) -> None:
    """Remove fit outputs whose prepared inputs have just been replaced."""
    for path in (
        paths.affine_history,
        paths.affine_checkpoint,
        paths.dynamics,
        paths.dynamics.with_suffix(".json"),
    ):
        path.unlink(missing_ok=True)


def _manifest_transition_dates(
    manifest: dict[str, object],
    *,
    window_transitions: int,
) -> tuple[tuple[date, ...], tuple[date, ...]]:
    """Return the exact dated transitions recorded by a rolling artifact."""
    origin_dates = manifest.get("training_origin_dates")
    endpoint_dates = manifest.get("training_endpoint_dates")
    if (
        not isinstance(origin_dates, list)
        or not isinstance(endpoint_dates, list)
        or len(origin_dates) != window_transitions
        or len(endpoint_dates) != window_transitions
        or any(not isinstance(value, str) for value in origin_dates)
        or any(not isinstance(value, str) for value in endpoint_dates)
    ):
        raise ValueError("rolling GROMM2 manifest lacks exact transition dates")
    return (
        tuple(date.fromisoformat(value) for value in origin_dates),
        tuple(date.fromisoformat(value) for value in endpoint_dates),
    )


def _validate_physical_fit(
    *,
    report: dict[str, object],
    dynamics_path: Path,
    config: RollingGromm2Config,
) -> _ValidatedPhysicalSelection:
    """Authenticate the candidate-versus-front-null physical selection."""
    if report.get("physical_selection_rule") != PHYSICAL_SELECTION_RULE:
        raise ValueError("rolling GROMM2 physical selection rule is incompatible")

    def finite_scalar(name: str) -> float:
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"rolling GROMM2 {name} is invalid")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"rolling GROMM2 {name} is invalid")
        return result

    def boolean(name: str) -> bool:
        value = report.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"rolling GROMM2 {name} is invalid")
        return value

    def vector(name: str) -> np.ndarray:
        values = np.asarray(report.get(name), dtype=np.float64)
        if values.shape != (config.factor_count,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(f"rolling GROMM2 {name} is invalid")
        return values

    candidate = vector("candidate_market_price_of_risk")
    baseline = vector("baseline_market_price_of_risk")
    selected = vector("selected_market_price_of_risk")
    candidate_standard_errors = vector(
        "candidate_market_price_of_risk_standard_errors"
    )
    candidate_norm = float(np.linalg.norm(candidate) / np.sqrt(252.0))
    baseline_norm = float(np.linalg.norm(baseline) / np.sqrt(252.0))
    selected_norm = float(np.linalg.norm(selected) / np.sqrt(252.0))
    reported_candidate_norm = finite_scalar("candidate_daily_risk_price_norm")
    reported_baseline_norm = finite_scalar("baseline_daily_risk_price_norm")
    reported_selected_norm = finite_scalar("selected_daily_risk_price_norm")
    if not (
        math.isclose(candidate_norm, reported_candidate_norm, abs_tol=1.0e-12)
        and math.isclose(baseline_norm, reported_baseline_norm, abs_tol=1.0e-12)
        and math.isclose(selected_norm, reported_selected_norm, abs_tol=1.0e-12)
        and math.isclose(
            selected_norm,
            finite_scalar("physical_daily_risk_price_norm"),
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("rolling GROMM2 physical risk-price norms are inconsistent")
    if (
        not math.isclose(
            baseline[0],
            finite_scalar("full_window_front_market_price_of_risk"),
            abs_tol=1.0e-12,
        )
        or not math.isclose(candidate[0], baseline[0], abs_tol=1.0e-12)
        or np.any(baseline[1:] != 0.0)
        or not math.isclose(
            candidate_standard_errors[0],
            finite_scalar("full_window_front_market_price_of_risk_standard_error"),
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("rolling GROMM2 front-only physical baseline is invalid")

    validation_improvement = finite_scalar(
        "physical_validation_improvement_fraction"
    )
    validation_gain_t = finite_scalar("physical_validation_gain_t")
    candidate_rejection_reasons: list[str] = []
    if candidate_norm > config.maximum_daily_risk_price_norm:
        candidate_rejection_reasons.append("daily-risk-price-norm")
    if validation_improvement < MINIMUM_PHYSICAL_VALIDATION_IMPROVEMENT:
        candidate_rejection_reasons.append("validation-improvement")
    if validation_gain_t < MINIMUM_PHYSICAL_VALIDATION_GAIN_T:
        candidate_rejection_reasons.append("validation-gain-t")
    candidate_accepted = not candidate_rejection_reasons
    baseline_rejection_reasons: list[str] = []
    if baseline_norm > config.maximum_daily_risk_price_norm:
        baseline_rejection_reasons.append("daily-risk-price-norm")
    baseline_accepted = not baseline_rejection_reasons
    selected_model = (
        "candidate"
        if candidate_accepted
        else "front-only-null"
        if baseline_accepted
        else "unavailable"
    )
    expected_selected = candidate if candidate_accepted else baseline
    fallback_used = selected_model == "front-only-null"
    selected_physical_accepted = selected_model != "unavailable"
    expected_reasons = report.get("candidate_rejection_reasons")
    expected_baseline_reasons = report.get("baseline_rejection_reasons")
    if (
        expected_reasons != candidate_rejection_reasons
        or expected_baseline_reasons != baseline_rejection_reasons
        or not np.array_equal(selected, expected_selected)
        or report.get("selected_model") != selected_model
        or boolean("candidate_accepted") is not candidate_accepted
        or boolean("physical_validation_accepted") is not candidate_accepted
        or boolean("baseline_accepted") is not baseline_accepted
        or boolean("selected_candidate") is not candidate_accepted
        or boolean("fallback_used") is not fallback_used
        or boolean("selected_physical_accepted") is not selected_physical_accepted
        or boolean("physical_accepted") is not selected_physical_accepted
    ):
        raise ValueError("rolling GROMM2 physical selection is inconsistent")

    candidate_fraction = finite_scalar("candidate_achievable_fraction")
    baseline_fraction = finite_scalar("baseline_achievable_fraction")
    selected_fraction = finite_scalar("selected_achievable_fraction")
    expected_fraction = (
        candidate_fraction if candidate_accepted else baseline_fraction
    )
    if not (
        math.isclose(selected_fraction, expected_fraction, abs_tol=1.0e-12)
        and math.isclose(
            finite_scalar("physical_projection_fraction"),
            selected_fraction,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("rolling GROMM2 physical fit fractions are inconsistent")

    with np.load(dynamics_path, allow_pickle=False) as values:
        selected_npz = np.asarray(
            values["physical_market_price_of_risk"], dtype=np.float64
        )
        selected_standard_errors = np.asarray(
            values["physical_market_price_of_risk_standard_errors"],
            dtype=np.float64,
        )
        candidate_npz = np.asarray(
            values["candidate_market_price_of_risk"], dtype=np.float64
        )
        candidate_standard_errors_npz = np.asarray(
            values["candidate_market_price_of_risk_standard_errors"],
            dtype=np.float64,
        )
        baseline_npz = np.asarray(
            values["baseline_market_price_of_risk"], dtype=np.float64
        )
        stored_rule = str(np.asarray(values["physical_selection_rule"]).astype("U"))
        stored_model = str(np.asarray(values["selected_model"]).astype("U"))
        stored_flags = {
            name: bool(np.asarray(values[name]).item())
            for name in (
                "candidate_accepted",
                "baseline_accepted",
                "selected_candidate",
                "fallback_used",
                "selected_physical_accepted",
                "physical_accepted",
            )
        }
    expected_selected_standard_errors = (
        candidate_standard_errors
        if candidate_accepted
        else np.concatenate(
            (
                candidate_standard_errors[:1],
                np.zeros(config.factor_count - 1, dtype=np.float64),
            )
        )
    )
    if (
        stored_rule != PHYSICAL_SELECTION_RULE
        or stored_model != selected_model
        or not np.array_equal(candidate_npz, candidate)
        or not np.array_equal(candidate_standard_errors_npz, candidate_standard_errors)
        or not np.array_equal(baseline_npz, baseline)
        or not np.array_equal(selected_npz, selected)
        or not np.array_equal(
            selected_standard_errors,
            expected_selected_standard_errors,
        )
        or stored_flags["candidate_accepted"] is not candidate_accepted
        or stored_flags["baseline_accepted"] is not baseline_accepted
        or stored_flags["selected_candidate"] is not candidate_accepted
        or stored_flags["fallback_used"] is not fallback_used
        or stored_flags["selected_physical_accepted"]
        is not selected_physical_accepted
        or stored_flags["physical_accepted"] is not selected_physical_accepted
    ):
        raise ValueError("rolling GROMM2 physical NPZ selection is inconsistent")
    return _ValidatedPhysicalSelection(
        candidate_achievable_fraction=candidate_fraction,
        baseline_achievable_fraction=baseline_fraction,
        selected_achievable_fraction=selected_fraction,
        candidate_daily_risk_price_norm=candidate_norm,
        baseline_daily_risk_price_norm=baseline_norm,
        selected_daily_risk_price_norm=selected_norm,
        candidate_accepted=candidate_accepted,
        baseline_accepted=baseline_accepted,
        selected_model=selected_model,
        fallback_used=fallback_used,
        selected_physical_accepted=selected_physical_accepted,
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


def _link_authenticated_file(
    *,
    source: Path,
    target: Path,
    expected_sha256: str,
) -> None:
    """Hard-link one immutable authenticated artifact without overwriting data."""
    if file_sha256(source) != expected_sha256:
        raise ValueError(f"authenticated source changed before linking: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        if file_sha256(target) != expected_sha256:
            raise ValueError(f"existing migration target is incompatible: {target}")
    if file_sha256(target) != expected_sha256:
        raise ValueError(f"linked migration target is inconsistent: {target}")


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(filename)
