"""Initialization-period target notionals for the weekly hedge benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WEEKLY_HORIZON,
    build_weekly_episode_path,
    overlapping_weekly_origins,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import (
    SanosLvConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    build_common_weekly_claims,
)

FloatArray = NDArray[np.float64]
WEEKLY_TARGET_NOTIONAL_SCHEMA = "hedging-weekly-target-notionals-v1"


@version("0.1", dependencies=[SanosLvConfig])
@dataclass(frozen=True, kw_only=True)
class WeeklyTargetNotionalConfig:
    """Precommitted initialization sample and weekly risk-unit convention."""

    initialization_start: date = date(2022, 1, 1)
    initialization_end: date = date(2022, 12, 31)
    reference_spot_risk_multiplier: float = 1.0
    minimum_episodes: int = 200
    minimum_target_standard_deviation: float = 1.0e-8
    background_grid_size: int = 2_049
    background_stddev_width: float = 10.0

    def __post_init__(self) -> None:
        verify_inp(
            self.initialization_start <= self.initialization_end,
            lambda: "weekly target-sizing initialization dates are reversed",
        )
        verify_inp(
            math.isfinite(self.reference_spot_risk_multiplier)
            and self.reference_spot_risk_multiplier > 0.0,
            lambda: "weekly reference spot-risk multiplier must be finite-positive",
        )
        verify_inp(
            self.minimum_episodes >= 20,
            lambda: "weekly target sizing requires at least twenty episodes",
        )
        verify_inp(
            math.isfinite(self.minimum_target_standard_deviation)
            and self.minimum_target_standard_deviation > 0.0,
            lambda: "weekly minimum target volatility must be finite-positive",
        )
        verify_inp(
            257 <= self.background_grid_size <= 4_097
            and self.background_grid_size % 2 == 1,
            lambda: "weekly sizing background grid must be odd and in [257,4097]",
        )
        verify_inp(
            math.isfinite(self.background_stddev_width)
            and self.background_stddev_width >= 6.0,
            lambda: "weekly sizing background width must be at least six",
        )


@dataclass(frozen=True, kw_only=True)
class WeeklyTargetNotionalCalibration:
    """Fixed weekly target quantities and their initialization diagnostics."""

    as_of_date: date
    episode_origin_dates: tuple[date, ...]
    episode_terminal_dates: tuple[date, ...]
    target_labels: tuple[str, ...]
    target_standard_deviations: FloatArray
    reference_spot_standard_deviation: float
    target_quantities: FloatArray

    def __post_init__(self) -> None:
        target_count = len(self.target_labels)
        episode_count = len(self.episode_origin_dates)
        if (
            episode_count == 0
            or len(self.episode_terminal_dates) != episode_count
            or tuple(sorted(set(self.episode_origin_dates)))
            != self.episode_origin_dates
        ):
            raise ValueError("weekly target-sizing episode dates are invalid")
        if any(
            terminal <= origin or terminal > self.as_of_date
            for origin, terminal in zip(
                self.episode_origin_dates,
                self.episode_terminal_dates,
                strict=True,
            )
        ):
            raise ValueError("weekly target-sizing episodes violate initialization timing")
        if self.target_standard_deviations.shape != (target_count,) or (
            self.target_quantities.shape != (target_count,)
        ):
            raise ValueError("weekly target-sizing vectors do not align")
        if (
            not math.isfinite(self.reference_spot_standard_deviation)
            or self.reference_spot_standard_deviation <= 0.0
            or np.any(
                ~np.isfinite(self.target_standard_deviations)
                | (self.target_standard_deviations <= 0.0)
            )
            or np.any(
                ~np.isfinite(self.target_quantities)
                | (self.target_quantities <= 0.0)
            )
        ):
            raise ValueError("weekly target-sizing risks and quantities are invalid")

    @property
    def quantity_by_label(self) -> dict[str, float]:
        """Return the fixed quantity keyed by target label."""
        return dict(zip(self.target_labels, self.target_quantities, strict=True))

    def to_dict(self) -> dict[str, object]:
        """Return the portable calibration payload."""
        return {
            "schema": WEEKLY_TARGET_NOTIONAL_SCHEMA,
            "as_of_date": self.as_of_date.isoformat(),
            "episode_origin_dates": [
                value.isoformat() for value in self.episode_origin_dates
            ],
            "episode_terminal_dates": [
                value.isoformat() for value in self.episode_terminal_dates
            ],
            "target_labels": list(self.target_labels),
            "target_standard_deviations": (
                self.target_standard_deviations.tolist()
            ),
            "reference_spot_standard_deviation": (
                self.reference_spot_standard_deviation
            ),
            "target_quantities": self.target_quantities.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> WeeklyTargetNotionalCalibration:
        """Reconstruct and validate a portable weekly sizing artifact."""
        if payload.get("schema") != WEEKLY_TARGET_NOTIONAL_SCHEMA:
            raise ValueError("weekly target-notional artifact has an incompatible schema")
        try:
            return cls(
                as_of_date=date.fromisoformat(str(payload["as_of_date"])),
                episode_origin_dates=tuple(
                    date.fromisoformat(str(value))
                    for value in payload["episode_origin_dates"]
                ),
                episode_terminal_dates=tuple(
                    date.fromisoformat(str(value))
                    for value in payload["episode_terminal_dates"]
                ),
                target_labels=tuple(str(value) for value in payload["target_labels"]),
                target_standard_deviations=np.asarray(
                    payload["target_standard_deviations"], dtype=np.float64
                ),
                reference_spot_standard_deviation=float(
                    payload["reference_spot_standard_deviation"]
                ),
                target_quantities=np.asarray(
                    payload["target_quantities"], dtype=np.float64
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("weekly target-notional artifact is incomplete") from error


def calibrate_weekly_target_notionals(
    panel: PublicHedgingPanel,
    *,
    config: WeeklyTargetNotionalConfig = WeeklyTargetNotionalConfig(),
) -> WeeklyTargetNotionalCalibration:
    """Fix weekly quantities from completed 2022 common-market P&L episodes."""
    require_public_hedging_panel(panel)
    initialization_dates = tuple(
        value
        for value in panel.dates
        if config.initialization_start <= value <= config.initialization_end
    )
    if not initialization_dates:
        raise ValueError("public panel contains no weekly sizing initialization dates")
    as_of_date = initialization_dates[-1]
    candidates = overlapping_weekly_origins(
        panel,
        evaluation_start=config.initialization_start,
        evaluation_end=config.initialization_end,
        rolling_window_business_days=1,
    )
    episode_origins: list[date] = []
    episode_terminals: list[date] = []
    pnl_rows: list[FloatArray] = []
    pricing_config = SanosLvConfig(
        label="weekly-target-notional-pricing",
        weekly_background_grid_size=config.background_grid_size,
        weekly_background_stddev_width=config.background_stddev_width,
    )
    targets = weekly_targets()
    expected_labels = tuple(target.label for target in targets)
    started = time.monotonic()
    for candidate_index, origin_date in enumerate(candidates, start=1):
        episode = build_weekly_episode_path(
            panel,
            origin_date=origin_date,
            schedule=HedgeSchedule.FROZEN_WEEKLY,
            rolling_window_business_days=1,
        )
        if episode.spec.terminal_date > config.initialization_end:
            continue
        claims = build_common_weekly_claims(
            episode.market_days[0],
            config=pricing_config,
        )
        labels = tuple(claim.target.label for claim in claims)
        if labels != expected_labels or any(
            claim.market_price is None for claim in claims
        ):
            raise RuntimeError("weekly sizing claims do not match the common target book")
        terminal_discount = float(episode.discount_to_origin[-1])
        pnl_rows.append(
            np.asarray(
                [
                    terminal_discount * claim.payoff(episode)
                    - float(claim.market_price)
                    for claim in claims
                ],
                dtype=np.float64,
            )
        )
        episode_origins.append(origin_date)
        episode_terminals.append(episode.spec.terminal_date)
        completed = len(episode_origins)
        if completed == 1 or completed % 20 == 0:
            elapsed = time.monotonic() - started
            rate = candidate_index / elapsed if elapsed > 0.0 else 0.0
            eta = (len(candidates) - candidate_index) / rate if rate > 0.0 else None
            print(
                json.dumps(
                    {
                        "stage": "weekly-target-sizing",
                        "completed_episodes": completed,
                        "candidate_index": candidate_index,
                        "candidate_count": len(candidates),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(pnl_rows) < config.minimum_episodes:
        raise ValueError(
            f"weekly target sizing has only {len(pnl_rows)} complete episodes, "
            f"below {config.minimum_episodes}"
        )
    reference_annual_volatility = panel.market_day(
        as_of_date
    ).atm_vol_at_business_dte(WEEKLY_HORIZON)
    reference_spot_standard_deviation = (
        config.reference_spot_risk_multiplier
        * reference_annual_volatility
        * math.sqrt(WEEKLY_HORIZON / DAYS_PER_YEAR)
    )
    return calibrate_weekly_target_notionals_from_pnl(
        np.stack(pnl_rows),
        as_of_date=as_of_date,
        episode_origin_dates=tuple(episode_origins),
        episode_terminal_dates=tuple(episode_terminals),
        target_labels=expected_labels,
        reference_spot_standard_deviation=reference_spot_standard_deviation,
        config=config,
    )


def calibrate_weekly_target_notionals_from_pnl(
    pnl: NDArray[np.floating],
    *,
    as_of_date: date,
    episode_origin_dates: tuple[date, ...],
    episode_terminal_dates: tuple[date, ...],
    target_labels: tuple[str, ...],
    reference_spot_standard_deviation: float,
    config: WeeklyTargetNotionalConfig = WeeklyTargetNotionalConfig(),
) -> WeeklyTargetNotionalCalibration:
    """Scale target P&L volatility to one common five-day ATM spot-risk unit."""
    values = np.asarray(pnl, dtype=np.float64)
    if (
        values.shape != (len(episode_origin_dates), len(target_labels))
        or len(episode_terminal_dates) != len(episode_origin_dates)
        or len(episode_origin_dates) < config.minimum_episodes
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("weekly target-sizing P&L sample is invalid")
    if (
        not math.isfinite(reference_spot_standard_deviation)
        or reference_spot_standard_deviation <= 0.0
    ):
        raise ValueError("weekly reference spot risk must be finite-positive")
    target_standard_deviations = np.std(values, axis=0, ddof=1)
    if np.any(
        ~np.isfinite(target_standard_deviations)
        | (
            target_standard_deviations
            < config.minimum_target_standard_deviation
        )
    ):
        raise ValueError("weekly target P&L standard deviation is too small")
    target_quantities = (
        reference_spot_standard_deviation / target_standard_deviations
    )
    return WeeklyTargetNotionalCalibration(
        as_of_date=as_of_date,
        episode_origin_dates=episode_origin_dates,
        episode_terminal_dates=episode_terminal_dates,
        target_labels=target_labels,
        target_standard_deviations=np.asarray(
            target_standard_deviations, dtype=np.float64
        ),
        reference_spot_standard_deviation=reference_spot_standard_deviation,
        target_quantities=np.asarray(target_quantities, dtype=np.float64),
    )


def load_or_calibrate_weekly_target_notionals(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    config: WeeklyTargetNotionalConfig = WeeklyTargetNotionalConfig(),
) -> WeeklyTargetNotionalCalibration:
    """Load or atomically build the dataset-bound weekly sizing artifact."""
    require_public_hedging_panel(panel)
    path, expected_header = _artifact_identity(
        panel, cache_directory=cache_directory, config=config
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            return _load_artifact(
                path,
                expected_header=expected_header,
                panel=panel,
                config=config,
            )
        calibration = calibrate_weekly_target_notionals(panel, config=config)
        calibration_payload = calibration.to_dict()
        _atomic_write_json(
            path,
            {
                "header": expected_header,
                "calibration_sha256": _payload_sha256(calibration_payload),
                "calibration": calibration_payload,
            },
        )
        return calibration


def load_authenticated_weekly_target_notionals(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    expected_artifact_sha256: str,
    config: WeeklyTargetNotionalConfig = WeeklyTargetNotionalConfig(),
) -> WeeklyTargetNotionalCalibration:
    """Load a CPU-preflighted weekly sizing artifact without recalibration."""
    require_public_hedging_panel(panel)
    verify_inp(
        len(expected_artifact_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_artifact_sha256.lower()
        ),
        lambda: "expected weekly target-notional hash must be a SHA-256",
    )
    path, expected_header = _artifact_identity(
        panel, cache_directory=cache_directory, config=config
    )
    with FileLock(str(path) + ".lock"):
        if file_sha256(path) != expected_artifact_sha256:
            raise RuntimeError("weekly target-notional artifact changed after preflight")
        calibration = _load_artifact(
            path,
            expected_header=expected_header,
            panel=panel,
            config=config,
        )
        if file_sha256(path) != expected_artifact_sha256:
            raise RuntimeError("weekly target-notional artifact changed while loading")
        return calibration


def load_authenticated_weekly_target_notionals_from_path(
    panel: PublicHedgingPanel,
    *,
    artifact_path: Path,
    expected_artifact_sha256: str,
    config: WeeklyTargetNotionalConfig = WeeklyTargetNotionalConfig(),
) -> WeeklyTargetNotionalCalibration:
    """Load the exact CPU-preflighted sizing artifact passed by production."""
    require_public_hedging_panel(panel)
    verify_inp(
        len(expected_artifact_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_artifact_sha256.lower()
        ),
        lambda: "expected weekly target-notional hash must be a SHA-256",
    )
    path = Path(artifact_path)
    _, expected_header = _artifact_identity(
        panel,
        cache_directory=path.parent,
        config=config,
    )
    with FileLock(str(path) + ".lock"):
        if file_sha256(path) != expected_artifact_sha256:
            raise RuntimeError("weekly target-notional artifact changed after preflight")
        calibration = _load_artifact(
            path,
            expected_header=expected_header,
            panel=panel,
            config=config,
        )
        if file_sha256(path) != expected_artifact_sha256:
            raise RuntimeError("weekly target-notional artifact changed while loading")
        return calibration


def weekly_target_notional_artifact_path(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    config: WeeklyTargetNotionalConfig = WeeklyTargetNotionalConfig(),
) -> Path:
    """Return the immutable artifact path for this dataset and sizing config."""
    path, _ = _artifact_identity(
        panel, cache_directory=cache_directory, config=config
    )
    return path


def weekly_target_notional_calibration_sha256(
    calibration: WeeklyTargetNotionalCalibration,
) -> str:
    """Return the canonical calibration hash used by weekly run identities."""
    return _payload_sha256(calibration.to_dict())


def _artifact_identity(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    config: WeeklyTargetNotionalConfig,
) -> tuple[Path, dict[str, object]]:
    header = {
        "schema": WEEKLY_TARGET_NOTIONAL_SCHEMA,
        "dataset_manifest_sha256": file_sha256(
            panel.dataset.directory / "manifest.json"
        ),
        "config_code_version": str(type(config).version),
        "config": {
            **asdict(config),
            "initialization_start": config.initialization_start.isoformat(),
            "initialization_end": config.initialization_end.isoformat(),
        },
        "pricing_code_version": str(SanosLvConfig.version),
        "target_labels": [target.label for target in weekly_targets()],
        "pnl_convention": "discounted-terminal-payoff-minus-common-market-price",
    }
    identity = hashlib.sha256(_canonical_json(header).encode()).hexdigest()
    return Path(cache_directory) / f"weekly-target-notionals-{identity}.json", header


def _load_artifact(
    path: Path,
    *,
    expected_header: dict[str, object],
    panel: PublicHedgingPanel,
    config: WeeklyTargetNotionalConfig,
) -> WeeklyTargetNotionalCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("header") != expected_header:
        raise ValueError("weekly target-notional cache identity is invalid")
    calibration_payload = payload.get("calibration")
    if not isinstance(calibration_payload, dict) or payload.get(
        "calibration_sha256"
    ) != _payload_sha256(calibration_payload):
        raise ValueError("weekly target-notional calibration hash is invalid")
    calibration = WeeklyTargetNotionalCalibration.from_dict(calibration_payload)
    _validate_calibration(panel, calibration=calibration, config=config)
    return calibration


def _validate_calibration(
    panel: PublicHedgingPanel,
    *,
    calibration: WeeklyTargetNotionalCalibration,
    config: WeeklyTargetNotionalConfig,
) -> None:
    initialization_dates = tuple(
        value
        for value in panel.dates
        if config.initialization_start <= value <= config.initialization_end
    )
    if not initialization_dates or calibration.as_of_date != initialization_dates[-1]:
        raise ValueError("weekly target-notional cache has an unexpected as-of date")
    if (
        len(calibration.episode_origin_dates) < config.minimum_episodes
        or calibration.episode_origin_dates[0] < config.initialization_start
        or calibration.episode_terminal_dates[-1] > config.initialization_end
        or calibration.target_labels
        != tuple(target.label for target in weekly_targets())
    ):
        raise ValueError("weekly target-notional cache violates initialization timing")
    expected_reference = (
        config.reference_spot_risk_multiplier
        * panel.market_day(calibration.as_of_date).atm_vol_at_business_dte(
            WEEKLY_HORIZON
        )
        * math.sqrt(WEEKLY_HORIZON / DAYS_PER_YEAR)
    )
    if not math.isclose(
        calibration.reference_spot_standard_deviation,
        expected_reference,
        rel_tol=1.0e-12,
        abs_tol=1.0e-14,
    ) or not np.allclose(
        calibration.target_quantities * calibration.target_standard_deviations,
        expected_reference,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("weekly target-notional cache has inconsistent risk scaling")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
