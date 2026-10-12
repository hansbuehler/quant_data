"""Resumable model-neutral evaluation of the complete daily target book."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)

DAILY_EVALUATION_SCHEMA = "hedging-daily-evaluation-v2"
DAILY_RECORD_SCHEMA = "hedging-daily-record-v2"

DailyBookFunction = Callable[
    [PublicHedgingPanel, date],
    tuple[HedgeSmokeResult, ...],
]


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class DailyEvaluationSpec:
    """Immutable logical identity and gates for one daily benchmark run."""

    label: str
    model: str
    measure: str
    model_config_identity: str
    model_config_snapshot_json: str
    dataset_manifest_sha256: str
    evaluation_start: date
    evaluation_end: date
    rolling_observations: int
    target_labels: tuple[str, ...]
    target_quantities: tuple[float, ...]
    target_notional_identity: str
    require_convergence: bool = True
    user_version: str = "0.1"

    def __post_init__(self) -> None:
        labels = (
            self.label,
            self.model,
            self.measure,
            self.model_config_identity,
            self.model_config_snapshot_json,
            self.dataset_manifest_sha256,
            self.target_notional_identity,
            self.user_version,
        )
        verify_inp(
            all(value.strip() for value in labels),
            lambda: "daily evaluation labels and identities must be non-empty",
        )
        verify_inp(
            len(self.dataset_manifest_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in self.dataset_manifest_sha256.lower()
            ),
            lambda: "dataset_manifest_sha256 must be a hexadecimal SHA-256",
        )
        try:
            model_config_snapshot = json.loads(self.model_config_snapshot_json)
        except json.JSONDecodeError as error:
            raise ValueError("model_config_snapshot_json must be valid JSON") from error
        verify_inp(
            isinstance(model_config_snapshot, dict),
            lambda: "model_config_snapshot_json must contain a JSON object",
        )
        verify_inp(
            self.evaluation_start <= self.evaluation_end,
            lambda: "daily evaluation date range is reversed",
        )
        verify_inp(
            self.rolling_observations > 0,
            lambda: "rolling_observations must be positive",
        )
        verify_inp(
            bool(self.target_labels)
            and len(set(self.target_labels)) == len(self.target_labels)
            and all(label.strip() for label in self.target_labels),
            lambda: "daily target labels must be non-empty and unique",
        )
        verify_inp(
            len(self.target_quantities) == len(self.target_labels)
            and all(
                math.isfinite(quantity) and quantity > 0.0
                for quantity in self.target_quantities
            ),
            lambda: "daily target quantities must be finite-positive and aligned",
        )

    @property
    def run_identity(self) -> str:
        """Return a stable identity for all result-affecting run controls."""
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return the immutable JSON run specification."""
        return {
            "schema": DAILY_EVALUATION_SCHEMA,
            "code_version": str(type(self).version),
            "label": self.label,
            "model": self.model,
            "measure": self.measure,
            "model_config_identity": self.model_config_identity,
            "model_config_snapshot": json.loads(self.model_config_snapshot_json),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "evaluation_start": self.evaluation_start.isoformat(),
            "evaluation_end": self.evaluation_end.isoformat(),
            "rolling_observations": self.rolling_observations,
            "target_labels": list(self.target_labels),
            "target_quantities": list(self.target_quantities),
            "target_notional_identity": self.target_notional_identity,
            "require_convergence": self.require_convergence,
            "user_version": self.user_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DailyEvaluationSpec:
        """Reconstruct and authenticate an immutable JSON run specification."""
        if payload.get("schema") != DAILY_EVALUATION_SCHEMA:
            raise ValueError("daily run metadata has an incompatible schema")
        if payload.get("code_version") != str(cls.version):
            raise ValueError("daily run metadata has an incompatible code version")
        model_config_snapshot = payload.get("model_config_snapshot")
        if not isinstance(model_config_snapshot, dict):
            raise ValueError("daily model configuration snapshot must be a JSON object")
        spec = cls(
            label=_required_string(payload, "label"),
            model=_required_string(payload, "model"),
            measure=_required_string(payload, "measure"),
            model_config_identity=_required_string(
                payload,
                "model_config_identity",
            ),
            model_config_snapshot_json=_canonical_json(model_config_snapshot),
            dataset_manifest_sha256=_required_string(
                payload,
                "dataset_manifest_sha256",
            ),
            evaluation_start=_required_date(payload, "evaluation_start"),
            evaluation_end=_required_date(payload, "evaluation_end"),
            rolling_observations=_required_integer(payload, "rolling_observations"),
            target_labels=_required_string_tuple(payload, "target_labels"),
            target_quantities=_required_float_tuple(payload, "target_quantities"),
            target_notional_identity=_required_string(
                payload,
                "target_notional_identity",
            ),
            require_convergence=_required_boolean(payload, "require_convergence"),
            user_version=_required_string(payload, "user_version"),
        )
        if payload.get("run_identity") != spec.run_identity:
            raise ValueError("daily run metadata identity is not authentic")
        return spec


@dataclass(frozen=True, kw_only=True)
class DailyEvaluationSummary:
    """Completion and numerical-gate counts for one resumable daily run."""

    run_identity: str
    eligible_origins: tuple[str, ...]
    completed_origins: tuple[str, ...]
    reused_origins: tuple[str, ...]
    result_count: int
    nonconverged_results: int
    cap_hits: int

    def __post_init__(self) -> None:
        if not self.run_identity.strip():
            raise ValueError("daily summary run identity must be non-empty")
        if (
            self.eligible_origins != tuple(sorted(self.eligible_origins))
            or len(set(self.eligible_origins)) != len(self.eligible_origins)
        ):
            raise ValueError("daily eligible origins must be unique and chronological")
        if self.completed_origins != tuple(sorted(self.completed_origins)):
            raise ValueError("daily completed origins must be chronological")
        if self.reused_origins != tuple(sorted(self.reused_origins)):
            raise ValueError("daily reused origins must be chronological")
        if len(set(self.completed_origins)) != len(self.completed_origins):
            raise ValueError("daily completed origins must be unique")
        if len(set(self.reused_origins)) != len(self.reused_origins):
            raise ValueError("daily reused origins must be unique")
        if self.completed_origins != self.eligible_origins[: len(self.completed_origins)]:
            raise ValueError("daily completed origins must be an eligible prefix")
        if not set(self.reused_origins).issubset(self.completed_origins):
            raise ValueError("daily reused origins must be completed origins")
        if (
            self.result_count < 0
            or self.nonconverged_results < 0
            or self.cap_hits < 0
        ):
            raise ValueError("daily summary counts must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible summary."""
        payload = asdict(self)
        payload["schema"] = DAILY_EVALUATION_SCHEMA
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DailyEvaluationSummary:
        """Reconstruct and validate one JSON completion summary."""
        if payload.get("schema") != DAILY_EVALUATION_SCHEMA:
            raise ValueError("daily run summary has an incompatible schema")
        return cls(
            run_identity=_required_string(payload, "run_identity"),
            eligible_origins=_required_string_tuple(payload, "eligible_origins"),
            completed_origins=_required_string_tuple(payload, "completed_origins"),
            reused_origins=_required_string_tuple(payload, "reused_origins"),
            result_count=_required_integer(payload, "result_count"),
            nonconverged_results=_required_integer(
                payload,
                "nonconverged_results",
            ),
            cap_hits=_required_integer(payload, "cap_hits"),
        )


def dataset_manifest_sha256(panel: PublicHedgingPanel) -> str:
    """Return the exact released-dataset manifest identity."""
    require_public_hedging_panel(panel)
    identity = getattr(panel, "dataset_manifest_sha256", None)
    if identity is not None:
        return identity
    path = panel.dataset.directory / "manifest.json"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_daily_origins(
    panel: PublicHedgingPanel,
    *,
    spec: DailyEvaluationSpec,
) -> tuple[date, ...]:
    """Return chronological gap-free origins with a complete training window."""
    require_public_hedging_panel(panel)
    origins: list[date] = []
    for origin_date in panel.dates:
        if not spec.evaluation_start <= origin_date <= spec.evaluation_end:
            continue
        try:
            panel.prior_one_day_origins(
                origin_date,
                count=spec.rolling_observations,
            )
            panel.one_day_transition(origin_date)
        except ValueError:
            continue
        origins.append(origin_date)
    return tuple(origins)


def run_daily_evaluation(
    panel: PublicHedgingPanel,
    *,
    spec: DailyEvaluationSpec,
    run_book: DailyBookFunction,
    output_directory: Path,
    progress: Callable[[int, int, date, bool], None] | None = None,
    stop_after: int | None = None,
) -> DailyEvaluationSummary:
    """Run or resume one complete daily evaluation with atomic date records."""
    require_public_hedging_panel(panel)
    if dataset_manifest_sha256(panel) != spec.dataset_manifest_sha256:
        raise ValueError("daily evaluation dataset manifest identity does not match")
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive when supplied")
    output = Path(output_directory)
    records_directory = output / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    _initialize_run(output / "run.json", spec)

    origins = eligible_daily_origins(panel, spec=spec)
    selected_origins = origins if stop_after is None else origins[:stop_after]
    completed: list[str] = []
    reused: list[str] = []
    result_count = 0
    nonconverged = 0
    cap_hits = 0
    for index, origin_date in enumerate(selected_origins, start=1):
        record_path = records_directory / f"{origin_date.isoformat()}.json"
        was_reused = record_path.exists()
        if was_reused:
            results = read_daily_record(record_path, spec=spec, origin_date=origin_date)
            reused.append(origin_date.isoformat())
        else:
            results = tuple(run_book(panel, origin_date))
            _validate_results(results, spec=spec, origin_date=origin_date)
            _write_daily_record(
                record_path,
                spec=spec,
                origin_date=origin_date,
                results=results,
            )
        completed.append(origin_date.isoformat())
        result_count += len(results)
        nonconverged += sum(not result.converged for result in results)
        cap_hits += sum(result.cap_hits for result in results)
        if progress is not None:
            progress(index, len(selected_origins), origin_date, was_reused)

    summary = DailyEvaluationSummary(
        run_identity=spec.run_identity,
        eligible_origins=tuple(value.isoformat() for value in origins),
        completed_origins=tuple(completed),
        reused_origins=tuple(reused),
        result_count=result_count,
        nonconverged_results=nonconverged,
        cap_hits=cap_hits,
    )
    _atomic_write_json(output / "summary.json", summary.to_dict())
    return summary


def _initialize_run(path: Path, spec: DailyEvaluationSpec) -> None:
    payload = {**spec.to_dict(), "run_identity": spec.run_identity}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("daily output directory belongs to another run identity")
        return
    _atomic_write_json(path, payload)


def _write_daily_record(
    path: Path,
    *,
    spec: DailyEvaluationSpec,
    origin_date: date,
    results: tuple[HedgeSmokeResult, ...],
) -> None:
    _atomic_write_json(
        path,
        {
            "schema": DAILY_RECORD_SCHEMA,
            "run_identity": spec.run_identity,
            "origin_date": origin_date.isoformat(),
            "results": [result.to_dict() for result in results],
        },
    )


def read_daily_record(
    path: Path,
    *,
    spec: DailyEvaluationSpec,
    origin_date: date,
) -> tuple[HedgeSmokeResult, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != DAILY_RECORD_SCHEMA
        or payload.get("run_identity") != spec.run_identity
        or payload.get("origin_date") != origin_date.isoformat()
        or not isinstance(payload.get("results"), list)
    ):
        raise ValueError(f"invalid or incompatible daily record {path}")
    results = tuple(HedgeSmokeResult(**item) for item in payload["results"])
    _validate_results(results, spec=spec, origin_date=origin_date)
    return results


def _validate_results(
    results: tuple[HedgeSmokeResult, ...],
    *,
    spec: DailyEvaluationSpec,
    origin_date: date,
) -> None:
    if tuple(result.claim for result in results) != spec.target_labels:
        raise ValueError("daily model did not emit the precommitted target book")
    for result, expected_quantity in zip(
        results,
        spec.target_quantities,
        strict=True,
    ):
        if (
            result.model != spec.model
            or result.measure != spec.measure
            or result.origin_date != origin_date.isoformat()
        ):
            raise ValueError("daily result identity does not match its run specification")
        if not math.isclose(
            result.claim_quantity,
            expected_quantity,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError("daily result claim quantity does not match its run specification")
    if spec.require_convergence:
        failures = tuple(
            result.claim
            for result in results
            if not result.converged or result.cap_hits > 0
        )
        if failures:
            raise RuntimeError(
                "daily numerical gate failed for " + ", ".join(failures)
            )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _required_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"daily metadata field {name!r} must be a non-empty string")
    return value


def _required_date(payload: dict[str, object], name: str) -> date:
    value = _required_string(payload, name)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"daily metadata field {name!r} must be an ISO date") from error


def _required_integer(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"daily metadata field {name!r} must be an integer")
    return value


def _required_boolean(payload: dict[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"daily metadata field {name!r} must be a boolean")
    return value


def _required_string_tuple(
    payload: dict[str, object],
    name: str,
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"daily metadata field {name!r} must be a string array")
    return tuple(value)


def _required_float_tuple(
    payload: dict[str, object],
    name: str,
) -> tuple[float, ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in value
    ):
        raise ValueError(f"daily metadata field {name!r} must be a numeric array")
    return tuple(float(item) for item in value)
