"""Initialization-period target notionals in a common ATM spot-risk unit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.config import (
    CovarianceHedgeConfig,
    CovarianceObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.hedger import (
    fit_covariance_book_actions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    COVARIANCE_RETURN_GEOMETRY_IDENTITY,
    HistoricalReturnPanel,
    build_historical_return_panel,
)

FloatArray = NDArray[np.float64]
TARGET_NOTIONAL_ARTIFACT_SCHEMA = "hedging-daily-target-notionals-v3"


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class TargetNotionalConfig:
    """Precommitted initialization sample and risk-unit convention."""

    initialization_start: date = date(2022, 1, 1)
    initialization_end: date = date(2022, 12, 31)
    reference_spot_risk_multiplier: float = 1.0
    mean_volatility_kappa: float = 1.0
    minimum_observations: int = 200
    numerical_position_cap: float = 1_000.0
    minimum_target_standard_deviation: float = 1.0e-8

    def __post_init__(self) -> None:
        verify_inp(
            self.initialization_start <= self.initialization_end,
            lambda: "target-sizing initialization dates are reversed",
        )
        for name, value in (
            ("reference_spot_risk_multiplier", self.reference_spot_risk_multiplier),
            ("mean_volatility_kappa", self.mean_volatility_kappa),
            ("numerical_position_cap", self.numerical_position_cap),
            (
                "minimum_target_standard_deviation",
                self.minimum_target_standard_deviation,
            ),
        ):
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name: f"{name} must be finite-positive",
            )
        verify_inp(
            self.minimum_observations >= 20,
            lambda: "minimum_observations must be at least twenty",
        )


@dataclass(frozen=True, kw_only=True)
class DailyTargetNotionalCalibration:
    """Fixed target quantities and initialization hedge-cost diagnostics."""

    as_of_date: date
    training_origin_dates: tuple[date, ...]
    target_labels: tuple[str, ...]
    target_maturities_business_days: NDArray[np.int64]
    target_standard_deviations: FloatArray
    reference_spot_standard_deviations: FloatArray
    target_quantities: FloatArray
    covariance_actions: FloatArray
    transaction_cost_fractions: FloatArray
    covariance_converged: bool
    covariance_cap_hits: int

    def __post_init__(self) -> None:
        target_count = len(self.target_labels)
        if not self.training_origin_dates or self.training_origin_dates[-1] >= self.as_of_date:
            raise ValueError("target-sizing dates are inconsistent")
        vectors = (
            self.target_maturities_business_days,
            self.target_standard_deviations,
            self.reference_spot_standard_deviations,
            self.target_quantities,
            self.transaction_cost_fractions,
        )
        if any(values.shape != (target_count,) for values in vectors):
            raise ValueError("target-sizing vectors do not align")
        if self.covariance_actions.shape[0] != target_count:
            raise ValueError("target-sizing action rows do not align")
        if not all(
            np.all(np.isfinite(values))
            for values in (*vectors, self.covariance_actions)
        ):
            raise ValueError("target-sizing arrays must be finite")
        if (
            np.any(self.target_maturities_business_days <= 0)
            or np.any(self.target_standard_deviations <= 0.0)
            or np.any(self.reference_spot_standard_deviations <= 0.0)
            or np.any(self.target_quantities <= 0.0)
            or np.any(self.transaction_cost_fractions < 0.0)
            or self.covariance_cap_hits < 0
        ):
            raise ValueError("target-sizing arrays contain invalid values")

    @property
    def median_transaction_cost_fraction(self) -> float:
        return float(np.median(self.transaction_cost_fractions))

    @property
    def p90_transaction_cost_fraction(self) -> float:
        return float(
            np.quantile(self.transaction_cost_fractions, 0.9, method="linear")
        )

    @property
    def quantity_by_label(self) -> dict[str, float]:
        """Return the fixed benchmark quantity keyed by target label."""
        return dict(zip(self.target_labels, self.target_quantities, strict=True))

    def to_dict(self) -> dict[str, object]:
        """Return a portable JSON representation of the complete calibration."""
        return {
            "schema": TARGET_NOTIONAL_ARTIFACT_SCHEMA,
            "as_of_date": self.as_of_date.isoformat(),
            "training_origin_dates": [
                value.isoformat() for value in self.training_origin_dates
            ],
            "target_labels": list(self.target_labels),
            "target_maturities_business_days": (
                self.target_maturities_business_days.tolist()
            ),
            "target_standard_deviations": self.target_standard_deviations.tolist(),
            "reference_spot_standard_deviations": (
                self.reference_spot_standard_deviations.tolist()
            ),
            "target_quantities": self.target_quantities.tolist(),
            "covariance_actions": self.covariance_actions.tolist(),
            "transaction_cost_fractions": self.transaction_cost_fractions.tolist(),
            "covariance_converged": self.covariance_converged,
            "covariance_cap_hits": self.covariance_cap_hits,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> DailyTargetNotionalCalibration:
        """Reconstruct and validate a portable calibration artifact."""
        if payload.get("schema") != TARGET_NOTIONAL_ARTIFACT_SCHEMA:
            raise ValueError("target-notional artifact has an incompatible schema")
        try:
            return cls(
                as_of_date=date.fromisoformat(str(payload["as_of_date"])),
                training_origin_dates=tuple(
                    date.fromisoformat(str(value))
                    for value in payload["training_origin_dates"]
                ),
                target_labels=tuple(str(value) for value in payload["target_labels"]),
                target_maturities_business_days=np.asarray(
                    payload["target_maturities_business_days"],
                    dtype=np.int64,
                ),
                target_standard_deviations=np.asarray(
                    payload["target_standard_deviations"],
                    dtype=np.float64,
                ),
                reference_spot_standard_deviations=np.asarray(
                    payload["reference_spot_standard_deviations"],
                    dtype=np.float64,
                ),
                target_quantities=np.asarray(
                    payload["target_quantities"],
                    dtype=np.float64,
                ),
                covariance_actions=np.asarray(
                    payload["covariance_actions"],
                    dtype=np.float64,
                ),
                transaction_cost_fractions=np.asarray(
                    payload["transaction_cost_fractions"],
                    dtype=np.float64,
                ),
                covariance_converged=_required_boolean(
                    payload,
                    "covariance_converged",
                ),
                covariance_cap_hits=_required_integer(
                    payload,
                    "covariance_cap_hits",
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("target-notional artifact is incomplete") from error


def load_or_calibrate_daily_target_notionals(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    config: TargetNotionalConfig = TargetNotionalConfig(),
) -> DailyTargetNotionalCalibration:
    """Load or atomically build the dataset- and config-specific calibration."""
    require_public_hedging_panel(panel)
    path, expected_header = _target_notional_artifact_identity(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            payload = _read_json_object(path)
            if payload.get("header") != expected_header:
                raise ValueError("target-notional cache identity does not match its path")
            calibration_payload = payload.get("calibration")
            if not isinstance(calibration_payload, dict):
                raise ValueError("target-notional cache lacks a calibration object")
            calibration_sha256 = payload.get("calibration_sha256")
            if not isinstance(calibration_sha256, str) or calibration_sha256 != (
                _calibration_payload_sha256(calibration_payload)
            ):
                raise ValueError("target-notional cache calibration hash is invalid")
            calibration = DailyTargetNotionalCalibration.from_dict(
                calibration_payload
            )
            _validate_target_notional_calibration(
                panel,
                config=config,
                calibration=calibration,
            )
            recomputed = calibrate_daily_target_notionals(panel, config=config)
            _require_equivalent_target_notional_calibrations(
                calibration,
                recomputed,
            )
            return calibration
        calibration = calibrate_daily_target_notionals(panel, config=config)
        calibration_payload = calibration.to_dict()
        _atomic_write_json(
            path,
            {
                "header": expected_header,
                "calibration_sha256": _calibration_payload_sha256(
                    calibration_payload
                ),
                "calibration": calibration_payload,
            },
        )
        return calibration


def load_authenticated_daily_target_notionals(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    expected_artifact_sha256: str,
    config: TargetNotionalConfig = TargetNotionalConfig(),
) -> DailyTargetNotionalCalibration:
    """Load a CPU-preflighted calibration without repeating its expensive fit.

    The caller pins the full artifact-file hash produced by an immediately
    preceding CPU preflight. Header, internal content hash, public-data timing,
    target geometry, accounting identities, and numerical gates are still
    validated here. A before/after hash check prevents concurrent replacement.
    """
    require_public_hedging_panel(panel)
    verify_inp(
        len(expected_artifact_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_artifact_sha256.lower()
        ),
        lambda: "expected target-notional artifact hash must be a SHA-256",
    )
    path, expected_header = _target_notional_artifact_identity(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    with FileLock(str(path) + ".lock"):
        if not path.is_file():
            raise FileNotFoundError(f"target-notional artifact is missing: {path}")
        if _file_sha256(path) != expected_artifact_sha256:
            raise RuntimeError("target-notional artifact changed after CPU preflight")
        payload = _read_json_object(path)
        if _file_sha256(path) != expected_artifact_sha256:
            raise RuntimeError("target-notional artifact changed while loading")
        if payload.get("header") != expected_header:
            raise ValueError("target-notional cache identity does not match its path")
        calibration_payload = payload.get("calibration")
        if not isinstance(calibration_payload, dict):
            raise ValueError("target-notional cache lacks a calibration object")
        calibration_sha256 = payload.get("calibration_sha256")
        if not isinstance(calibration_sha256, str) or calibration_sha256 != (
            _calibration_payload_sha256(calibration_payload)
        ):
            raise ValueError("target-notional cache calibration hash is invalid")
        calibration = DailyTargetNotionalCalibration.from_dict(calibration_payload)
        _validate_target_notional_calibration(
            panel,
            config=config,
            calibration=calibration,
        )
        return calibration


def daily_target_notional_calibration_sha256(
    calibration: DailyTargetNotionalCalibration,
) -> str:
    """Return the canonical content hash used in benchmark run identities."""
    return _calibration_payload_sha256(calibration.to_dict())


def daily_target_notional_artifact_path(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    config: TargetNotionalConfig = TargetNotionalConfig(),
) -> Path:
    """Return the immutable target-notional artifact path for this dataset."""
    path, _ = _target_notional_artifact_identity(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    return path


def daily_target_notional_file_sha256(path: Path) -> str:
    """Return the full artifact-file hash pinned between CPU and GPU phases."""
    return _file_sha256(Path(path))


def calibrate_daily_target_notionals(
    panel: PublicHedgingPanel,
    *,
    config: TargetNotionalConfig = TargetNotionalConfig(),
) -> DailyTargetNotionalCalibration:
    """Calibrate fixed daily-target quantities using initialization data only."""
    require_public_hedging_panel(panel)
    as_of_date, training_origin_dates = _target_notional_training_dates(
        panel,
        config=config,
    )
    training_origins = list(training_origin_dates)
    if len(training_origins) < config.minimum_observations:
        raise ValueError(
            f"target sizing has only {len(training_origins)} valid initialization "
            f"returns, below {config.minimum_observations}"
        )

    targets = daily_digital_targets()
    market_day = panel.market_day(as_of_date)
    hedge_quotes = market_day.hedge_snapshot().option_quotes
    returns = build_historical_return_panel(
        panel,
        origin_date=as_of_date,
        option_quotes=hedge_quotes,
        targets=targets,
        count=len(training_origins),
    )
    if returns.training_origin_dates != tuple(training_origins):
        raise RuntimeError("target sizing return panel escaped the initialization window")
    maturities = np.asarray(
        [target.maturity_business_days for target in targets],
        dtype=np.int64,
    )
    reference_annual_volatilities = np.asarray(
        [market_day.atm_vol_at_business_dte(int(value)) for value in maturities],
        dtype=np.float64,
    )
    costs = np.asarray(
        [
            2.0e-4,
            *(
                quote.half_spread
                * (1.0 if round(quote.business_dte) == 1 else 2.0)
                for quote in hedge_quotes
            ),
        ],
        dtype=np.float64,
    )
    return calibrate_daily_target_notionals_from_returns(
        returns,
        as_of_date=as_of_date,
        target_maturities_business_days=maturities,
        reference_annual_volatilities=reference_annual_volatilities,
        linear_transaction_costs=costs,
        config=config,
    )


def _validate_target_notional_calibration(
    panel: PublicHedgingPanel,
    *,
    config: TargetNotionalConfig,
    calibration: DailyTargetNotionalCalibration,
) -> None:
    expected_as_of_date, expected_training_origins = (
        _target_notional_training_dates(panel, config=config)
    )
    if calibration.as_of_date != expected_as_of_date:
        raise ValueError("target-notional cache has an unexpected as-of date")
    if calibration.training_origin_dates != expected_training_origins:
        raise ValueError("target-notional cache has unexpected training origins")

    targets = daily_digital_targets()
    expected_labels = tuple(target.label for target in targets)
    expected_maturities = np.asarray(
        [target.maturity_business_days for target in targets],
        dtype=np.int64,
    )
    if calibration.target_labels != expected_labels or not np.array_equal(
        calibration.target_maturities_business_days,
        expected_maturities,
    ):
        raise ValueError("target-notional cache does not match the target catalog")

    market_day = panel.market_day(expected_as_of_date)
    reference_annual_volatilities = np.asarray(
        [
            market_day.atm_vol_at_business_dte(int(maturity))
            for maturity in expected_maturities
        ],
        dtype=np.float64,
    )
    expected_reference_standard_deviations = (
        config.reference_spot_risk_multiplier
        * reference_annual_volatilities
        * np.sqrt(expected_maturities / DAYS_PER_YEAR)
    )
    if not np.allclose(
        calibration.reference_spot_standard_deviations,
        expected_reference_standard_deviations,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("target-notional cache has inconsistent reference risk")
    if not np.allclose(
        calibration.target_quantities * calibration.target_standard_deviations,
        calibration.reference_spot_standard_deviations,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("target-notional cache has inconsistent target quantities")

    hedge_quotes = market_day.hedge_snapshot().option_quotes
    costs = np.asarray(
        [
            2.0e-4,
            *(
                quote.half_spread
                * (1.0 if round(quote.business_dte) == 1 else 2.0)
                for quote in hedge_quotes
            ),
        ],
        dtype=np.float64,
    )
    expected_action_shape = (len(targets), len(costs))
    if calibration.covariance_actions.shape != expected_action_shape:
        raise ValueError("target-notional cache actions do not match the hedge book")
    expected_cost_fractions = (
        np.abs(calibration.covariance_actions)
        @ costs
        / calibration.reference_spot_standard_deviations
    )
    if not np.allclose(
        calibration.transaction_cost_fractions,
        expected_cost_fractions,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("target-notional cache has inconsistent transaction costs")
    if not calibration.covariance_converged or calibration.covariance_cap_hits:
        raise ValueError("target-notional cache failed its numerical gates")


def _target_notional_training_dates(
    panel: PublicHedgingPanel,
    *,
    config: TargetNotionalConfig,
) -> tuple[date, tuple[date, ...]]:
    initialization_dates = tuple(
        value
        for value in panel.dates
        if config.initialization_start <= value <= config.initialization_end
    )
    if not initialization_dates:
        raise ValueError("public panel contains no target-sizing initialization dates")
    as_of_date = initialization_dates[-1]
    training_origins: list[date] = []
    for origin_date in initialization_dates[:-1]:
        try:
            transition = panel.one_day_transition(origin_date)
        except ValueError:
            continue
        if transition.next_date <= as_of_date:
            training_origins.append(origin_date)
    return as_of_date, tuple(training_origins)


def _require_equivalent_target_notional_calibrations(
    cached: DailyTargetNotionalCalibration,
    recomputed: DailyTargetNotionalCalibration,
) -> None:
    if (
        cached.as_of_date != recomputed.as_of_date
        or cached.training_origin_dates != recomputed.training_origin_dates
        or cached.target_labels != recomputed.target_labels
        or cached.covariance_converged != recomputed.covariance_converged
        or cached.covariance_cap_hits != recomputed.covariance_cap_hits
    ):
        raise ValueError("target-notional cache does not match its public-data fit")
    cached_arrays = (
        cached.target_maturities_business_days,
        cached.target_standard_deviations,
        cached.reference_spot_standard_deviations,
        cached.target_quantities,
        cached.covariance_actions,
        cached.transaction_cost_fractions,
    )
    recomputed_arrays = (
        recomputed.target_maturities_business_days,
        recomputed.target_standard_deviations,
        recomputed.reference_spot_standard_deviations,
        recomputed.target_quantities,
        recomputed.covariance_actions,
        recomputed.transaction_cost_fractions,
    )
    if any(
        cached_values.shape != recomputed_values.shape
        or not np.allclose(
            cached_values,
            recomputed_values,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        for cached_values, recomputed_values in zip(
            cached_arrays,
            recomputed_arrays,
            strict=True,
        )
    ):
        raise ValueError("target-notional cache does not match its public-data fit")


def calibrate_daily_target_notionals_from_returns(
    returns: HistoricalReturnPanel,
    *,
    as_of_date: date,
    target_maturities_business_days: NDArray[np.integer],
    reference_annual_volatilities: NDArray[np.floating],
    linear_transaction_costs: NDArray[np.floating],
    config: TargetNotionalConfig = TargetNotionalConfig(),
) -> DailyTargetNotionalCalibration:
    """Calibrate notionals and the independent mean-volatility cost diagnostic."""
    maturities = np.asarray(target_maturities_business_days, dtype=np.int64)
    annual_volatilities = np.asarray(
        reference_annual_volatilities,
        dtype=np.float64,
    )
    costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    target_count = len(returns.target_labels)
    if (
        maturities.shape != (target_count,)
        or annual_volatilities.shape != (target_count,)
        or costs.shape != (returns.instrument_returns.shape[1],)
        or np.any(maturities <= 0)
        or np.any(~np.isfinite(annual_volatilities) | (annual_volatilities <= 0.0))
        or np.any(~np.isfinite(costs) | (costs < 0.0))
    ):
        raise ValueError("target-sizing inputs do not align or contain invalid values")
    if len(returns.training_origin_dates) < config.minimum_observations:
        raise ValueError("target-sizing return panel is too short")
    target_standard_deviations = np.std(
        returns.target_returns,
        axis=0,
        ddof=1,
    )
    if np.any(
        ~np.isfinite(target_standard_deviations)
        | (target_standard_deviations < config.minimum_target_standard_deviation)
    ):
        raise ValueError("target return standard deviation is too small")
    reference_standard_deviations = (
        config.reference_spot_risk_multiplier
        * annual_volatilities
        * np.sqrt(maturities / DAYS_PER_YEAR)
    )
    target_quantities = reference_standard_deviations / target_standard_deviations
    scaled_returns = replace(
        returns,
        target_returns=returns.target_returns * target_quantities[None, :],
    )
    fit = fit_covariance_book_actions(
        scaled_returns,
        linear_transaction_costs=costs,
        config=CovarianceHedgeConfig(
            label="target-notional-initialization",
            rolling_observations=len(returns.training_origin_dates),
            objective=CovarianceObjective.MEAN_VOLATILITY,
            risk_parameter=config.mean_volatility_kappa,
            position_cap=config.numerical_position_cap,
            maximum_solver_iterations=50_000,
        ),
    )
    if any(action.action is None for action in fit.actions):
        raise RuntimeError("target-sizing mean-volatility hedge is unbounded")
    actions = np.stack([action.action for action in fit.actions])
    cap_hits = sum(action.cap_hits for action in fit.actions)
    converged = all(action.converged for action in fit.actions)
    transaction_cost_fractions = (
        np.abs(actions) @ costs / reference_standard_deviations
    )
    if not converged or cap_hits:
        raise RuntimeError(
            "target-sizing covariance hedge failed its numerical or cap gate"
        )
    return DailyTargetNotionalCalibration(
        as_of_date=as_of_date,
        training_origin_dates=returns.training_origin_dates,
        target_labels=returns.target_labels,
        target_maturities_business_days=maturities,
        target_standard_deviations=np.asarray(
            target_standard_deviations,
            dtype=np.float64,
        ),
        reference_spot_standard_deviations=np.asarray(
            reference_standard_deviations,
            dtype=np.float64,
        ),
        target_quantities=np.asarray(target_quantities, dtype=np.float64),
        covariance_actions=np.asarray(actions, dtype=np.float64),
        transaction_cost_fractions=np.asarray(
            transaction_cost_fractions,
            dtype=np.float64,
        ),
        covariance_converged=converged,
        covariance_cap_hits=cap_hits,
    )


def _target_notional_artifact_identity(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    config: TargetNotionalConfig,
) -> tuple[Path, dict[str, object]]:
    manifest_sha256 = _file_sha256(panel.dataset.directory / "manifest.json")
    header = {
        "schema": TARGET_NOTIONAL_ARTIFACT_SCHEMA,
        "dataset_manifest_sha256": manifest_sha256,
        "return_geometry_identity": COVARIANCE_RETURN_GEOMETRY_IDENTITY,
        "config_code_version": str(type(config).version),
        "config": {
            "initialization_start": config.initialization_start.isoformat(),
            "initialization_end": config.initialization_end.isoformat(),
            "reference_spot_risk_multiplier": config.reference_spot_risk_multiplier,
            "mean_volatility_kappa": config.mean_volatility_kappa,
            "minimum_observations": config.minimum_observations,
            "numerical_position_cap": config.numerical_position_cap,
            "minimum_target_standard_deviation": (
                config.minimum_target_standard_deviation
            ),
        },
    }
    identity = hashlib.sha256(_canonical_json(header).encode()).hexdigest()
    return (
        Path(cache_directory) / f"daily-target-notionals-{identity}.json",
        header,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target-notional artifact must contain a JSON object")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _calibration_payload_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _required_boolean(payload: dict[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"target-notional field {name!r} must be boolean")
    return value


def _required_integer(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"target-notional field {name!r} must be an integer")
    return value
