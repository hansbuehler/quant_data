"""Resumable model-neutral evaluation of complete weekly target books."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.hedging.accounting import ClaimPnlBreakdown
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyHedgeResult,
    overlapping_weekly_origins,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyModelRun,
    WeeklyPriceRecord,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
    write_numerical_gate_failure,
)

WEEKLY_EVALUATION_SCHEMA = "hedging-weekly-evaluation-v4"


def weekly_evaluation_model(model: str) -> str:
    """Return the model identity persisted by the weekly adapter."""
    return {
        "diffusion-ad-seq-vol": "ad-seq-vol",
        "diffusion-ad-seq-vol-ft": "ad-seq-vol-ft",
    }.get(model, model)


WEEKLY_RECORD_SCHEMA = "hedging-weekly-record-v4"

WeeklyBookFunction = Callable[
    [PublicHedgingPanel, date],
    tuple[WeeklyModelRun, ...],
]


@version("0.4")
@dataclass(frozen=True, kw_only=True)
class WeeklyEvaluationSpec:
    """Immutable logical identity and numerical gates for one weekly run."""

    label: str
    model: str
    measure: str
    model_config_identity: str
    policy_config_identity: str
    configuration_snapshot_json: str
    dataset_manifest_sha256: str
    schedule: HedgeSchedule
    evaluation_start: date
    evaluation_end: date
    rolling_observations: int
    target_labels: tuple[str, ...]
    target_quantities: tuple[float, ...]
    target_notional_identity: str
    position_cap: float
    spot_cost_bps: float = 1.0
    require_no_cap_hits: bool = True
    user_version: str = "0.1"

    def __post_init__(self) -> None:
        identities = (
            self.label,
            self.model,
            self.measure,
            self.model_config_identity,
            self.policy_config_identity,
            self.configuration_snapshot_json,
            self.dataset_manifest_sha256,
            self.target_notional_identity,
            self.user_version,
        )
        verify_inp(
            all(value.strip() for value in identities),
            lambda: "weekly evaluation labels and identities must be non-empty",
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
            configuration_snapshot = json.loads(self.configuration_snapshot_json)
        except json.JSONDecodeError as error:
            raise ValueError("configuration_snapshot_json must be valid JSON") from error
        verify_inp(
            isinstance(configuration_snapshot, dict),
            lambda: "configuration_snapshot_json must contain a JSON object",
        )
        verify_inp(
            self.schedule
            in {HedgeSchedule.FROZEN_WEEKLY, HedgeSchedule.RESTARTED_WEEKLY},
            lambda: "weekly evaluation requires a weekly hedge schedule",
        )
        verify_inp(
            self.evaluation_start <= self.evaluation_end,
            lambda: "weekly evaluation date range is reversed",
        )
        verify_inp(
            self.rolling_observations > 0,
            lambda: "rolling_observations must be positive",
        )
        expected_targets = tuple(target.label for target in weekly_targets())
        verify_inp(
            self.target_labels == expected_targets,
            lambda: "weekly evaluation requires the complete ordered target book",
        )
        verify_inp(
            len(self.target_quantities) == len(self.target_labels)
            and all(
                math.isfinite(quantity) and quantity > 0.0
                for quantity in self.target_quantities
            ),
            lambda: "weekly target quantities must be finite-positive and aligned",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "weekly position cap must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.spot_cost_bps) and self.spot_cost_bps >= 0.0,
            lambda: "weekly spot cost must be finite and non-negative",
        )

    @property
    def run_identity(self) -> str:
        """Return the stable identity of every result-affecting control."""
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return the immutable JSON run specification."""
        return {
            "schema": WEEKLY_EVALUATION_SCHEMA,
            "code_version": str(type(self).version),
            "label": self.label,
            "model": self.model,
            "measure": self.measure,
            "model_config_identity": self.model_config_identity,
            "policy_config_identity": self.policy_config_identity,
            "configuration_snapshot": json.loads(
                self.configuration_snapshot_json
            ),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "schedule": self.schedule.value,
            "evaluation_start": self.evaluation_start.isoformat(),
            "evaluation_end": self.evaluation_end.isoformat(),
            "rolling_observations": self.rolling_observations,
            "target_labels": list(self.target_labels),
            "target_quantities": list(self.target_quantities),
            "target_notional_identity": self.target_notional_identity,
            "position_cap": self.position_cap,
            "spot_cost_bps": self.spot_cost_bps,
            "require_no_cap_hits": self.require_no_cap_hits,
            "user_version": self.user_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WeeklyEvaluationSpec:
        """Reconstruct and authenticate an immutable weekly specification."""
        if payload.get("schema") != WEEKLY_EVALUATION_SCHEMA:
            raise ValueError("weekly run metadata has an incompatible schema")
        if payload.get("code_version") != str(cls.version):
            raise ValueError("weekly run metadata has an incompatible code version")
        snapshot = payload.get("configuration_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("weekly configuration snapshot must be a JSON object")
        spec = cls(
            label=_required_string(payload, "label"),
            model=_required_string(payload, "model"),
            measure=_required_string(payload, "measure"),
            model_config_identity=_required_string(
                payload, "model_config_identity"
            ),
            policy_config_identity=_required_string(
                payload, "policy_config_identity"
            ),
            configuration_snapshot_json=_canonical_json(snapshot),
            dataset_manifest_sha256=_required_string(
                payload, "dataset_manifest_sha256"
            ),
            schedule=HedgeSchedule(_required_string(payload, "schedule")),
            evaluation_start=_required_date(payload, "evaluation_start"),
            evaluation_end=_required_date(payload, "evaluation_end"),
            rolling_observations=_required_integer(payload, "rolling_observations"),
            target_labels=_required_string_tuple(payload, "target_labels"),
            target_quantities=_required_float_tuple(payload, "target_quantities"),
            target_notional_identity=_required_string(
                payload, "target_notional_identity"
            ),
            position_cap=_required_float(payload, "position_cap"),
            spot_cost_bps=_required_float(payload, "spot_cost_bps"),
            require_no_cap_hits=_required_boolean(
                payload, "require_no_cap_hits"
            ),
            user_version=_required_string(payload, "user_version"),
        )
        if payload.get("run_identity") != spec.run_identity:
            raise ValueError("weekly run metadata identity is not authentic")
        return spec


@dataclass(frozen=True, kw_only=True)
class WeeklyEvaluationSummary:
    """Completion and numerical diagnostics for one resumable weekly run."""

    run_identity: str
    eligible_origins: tuple[str, ...]
    completed_origins: tuple[str, ...]
    reused_origins: tuple[str, ...]
    result_count: int
    fit_cap_hits: int
    executed_cap_hits: int
    nonconverged_fits: int
    unregularized_recession_direction_fits: int
    recession_direction_fits: int
    cap_hits: int

    def __post_init__(self) -> None:
        if not self.run_identity.strip():
            raise ValueError("weekly summary run identity must be non-empty")
        if (
            self.eligible_origins != tuple(sorted(self.eligible_origins))
            or len(set(self.eligible_origins)) != len(self.eligible_origins)
        ):
            raise ValueError("weekly eligible origins must be unique and chronological")
        if self.completed_origins != self.eligible_origins[: len(self.completed_origins)]:
            raise ValueError("weekly completed origins must be an eligible prefix")
        if self.reused_origins != tuple(sorted(set(self.reused_origins))):
            raise ValueError("weekly reused origins must be unique and chronological")
        if not set(self.reused_origins).issubset(self.completed_origins):
            raise ValueError("weekly reused origins must be completed origins")
        counts = (
            self.result_count,
            self.fit_cap_hits,
            self.executed_cap_hits,
            self.nonconverged_fits,
            self.unregularized_recession_direction_fits,
            self.recession_direction_fits,
            self.cap_hits,
        )
        if any(value < 0 for value in counts):
            raise ValueError("weekly summary counts must be non-negative")
        if self.cap_hits != self.fit_cap_hits + self.executed_cap_hits:
            raise ValueError("weekly summary cap counts do not reconcile")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible completion summary."""
        return {**asdict(self), "schema": WEEKLY_EVALUATION_SCHEMA}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WeeklyEvaluationSummary:
        """Reconstruct and validate one JSON completion summary."""
        if payload.get("schema") != WEEKLY_EVALUATION_SCHEMA:
            raise ValueError("weekly run summary has an incompatible schema")
        return cls(
            run_identity=_required_string(payload, "run_identity"),
            eligible_origins=_required_string_tuple(payload, "eligible_origins"),
            completed_origins=_required_string_tuple(payload, "completed_origins"),
            reused_origins=_required_string_tuple(payload, "reused_origins"),
            result_count=_required_integer(payload, "result_count"),
            fit_cap_hits=_required_integer(payload, "fit_cap_hits"),
            executed_cap_hits=_required_integer(payload, "executed_cap_hits"),
            nonconverged_fits=_required_integer(payload, "nonconverged_fits"),
            unregularized_recession_direction_fits=_required_integer(
                payload, "unregularized_recession_direction_fits"
            ),
            recession_direction_fits=_required_integer(
                payload, "recession_direction_fits"
            ),
            cap_hits=_required_integer(payload, "cap_hits"),
        )


@dataclass(frozen=True, kw_only=True)
class WeeklyNumericalDiagnostics:
    """Deduplicated fitted-policy and realized-action diagnostics."""

    fit_cap_hits: int
    executed_cap_hits: int
    nonconverged_fits: int
    unregularized_recession_direction_fits: int
    recession_direction_fits: int

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.fit_cap_hits,
                self.executed_cap_hits,
                self.nonconverged_fits,
                self.unregularized_recession_direction_fits,
                self.recession_direction_fits,
            )
        ):
            raise ValueError("weekly numerical diagnostic counts must be non-negative")

    @property
    def cap_hits(self) -> int:
        """Return fitted-policy plus realized-action cap hits."""
        return self.fit_cap_hits + self.executed_cap_hits


def eligible_weekly_origins(
    panel: PublicHedgingPanel,
    *,
    spec: WeeklyEvaluationSpec,
) -> tuple[date, ...]:
    """Return all chronological overlapping episodes admitted by the spec."""
    require_public_hedging_panel(panel)
    return overlapping_weekly_origins(
        panel,
        evaluation_start=spec.evaluation_start,
        evaluation_end=spec.evaluation_end,
        rolling_window_business_days=spec.rolling_observations,
    )


def run_weekly_evaluation(
    panel: PublicHedgingPanel,
    *,
    spec: WeeklyEvaluationSpec,
    run_book: WeeklyBookFunction,
    output_directory: Path,
    progress: Callable[[int, int, date, bool], None] | None = None,
    stop_after: int | None = None,
) -> WeeklyEvaluationSummary:
    """Run or resume overlapping episodes with one atomic record per origin."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".writer.lock"), timeout=0):
        return _run_weekly_evaluation_locked(
            panel,
            spec=spec,
            run_book=run_book,
            output_directory=output,
            progress=progress,
            stop_after=stop_after,
        )


def _run_weekly_evaluation_locked(
    panel: PublicHedgingPanel,
    *,
    spec: WeeklyEvaluationSpec,
    run_book: WeeklyBookFunction,
    output_directory: Path,
    progress: Callable[[int, int, date, bool], None] | None,
    stop_after: int | None,
) -> WeeklyEvaluationSummary:
    """Execute one weekly writer while its run-identity lock is held."""
    require_public_hedging_panel(panel)
    if dataset_manifest_sha256(panel) != spec.dataset_manifest_sha256:
        raise ValueError("weekly evaluation dataset manifest identity does not match")
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive when supplied")
    output = Path(output_directory)
    records_directory = output / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    _initialize_run(output / "run.json", spec)

    origins = eligible_weekly_origins(panel, spec=spec)
    selected_origins = origins if stop_after is None else origins[:stop_after]
    completed: list[str] = []
    reused: list[str] = []
    result_count = 0
    fit_cap_hits = 0
    executed_cap_hits = 0
    nonconverged_fits = 0
    unregularized_recession_direction_fits = 0
    recession_direction_fits = 0
    cap_hits = 0
    for index, origin_date in enumerate(selected_origins, start=1):
        record_path = records_directory / f"{origin_date.isoformat()}.json"
        was_reused = record_path.exists()
        if was_reused:
            runs = read_weekly_record(
                record_path,
                panel=panel,
                spec=spec,
                origin_date=origin_date,
            )
            reused.append(origin_date.isoformat())
        else:
            runs = tuple(run_book(panel, origin_date))
            diagnostics = _validate_runs(
                runs,
                panel=panel,
                spec=spec,
                origin_date=origin_date,
                enforce_numerical_gate=False,
            )
            if (
                (spec.require_no_cap_hits and diagnostics.cap_hits)
                or diagnostics.nonconverged_fits
                or diagnostics.recession_direction_fits
            ):
                write_numerical_gate_failure(
                    output_directory=output,
                    run_identity=spec.run_identity,
                    evaluation_schema=WEEKLY_EVALUATION_SCHEMA,
                    model=spec.model,
                    measure=spec.measure,
                    schedule=spec.schedule.value,
                    dataset_manifest_sha256=spec.dataset_manifest_sha256,
                    origin_date=origin_date,
                    diagnostics={
                        "fit_cap_hits": diagnostics.fit_cap_hits,
                        "executed_cap_hits": diagnostics.executed_cap_hits,
                        "nonconverged_fits": diagnostics.nonconverged_fits,
                        "unregularized_recession_direction_fits": (
                            diagnostics.unregularized_recession_direction_fits
                        ),
                        "recession_direction_fits": (
                            diagnostics.recession_direction_fits
                        ),
                        "cap_hits": diagnostics.cap_hits,
                    },
                    require_no_fit_cap_hits=spec.require_no_cap_hits,
                    require_no_executed_cap_hits=spec.require_no_cap_hits,
                )
                _enforce_weekly_numerical_gate(
                    diagnostics,
                    require_no_cap_hits=spec.require_no_cap_hits,
                )
            _write_weekly_record(
                record_path,
                spec=spec,
                origin_date=origin_date,
                runs=runs,
                diagnostics=diagnostics,
            )
        diagnostics = weekly_numerical_diagnostics(
            runs,
            position_cap=spec.position_cap,
        )
        completed.append(origin_date.isoformat())
        result_count += len(runs)
        fit_cap_hits += diagnostics.fit_cap_hits
        executed_cap_hits += diagnostics.executed_cap_hits
        nonconverged_fits += diagnostics.nonconverged_fits
        unregularized_recession_direction_fits += (
            diagnostics.unregularized_recession_direction_fits
        )
        recession_direction_fits += diagnostics.recession_direction_fits
        cap_hits += diagnostics.cap_hits
        if progress is not None:
            progress(index, len(selected_origins), origin_date, was_reused)

    summary = WeeklyEvaluationSummary(
        run_identity=spec.run_identity,
        eligible_origins=tuple(value.isoformat() for value in origins),
        completed_origins=tuple(completed),
        reused_origins=tuple(reused),
        result_count=result_count,
        fit_cap_hits=fit_cap_hits,
        executed_cap_hits=executed_cap_hits,
        nonconverged_fits=nonconverged_fits,
        unregularized_recession_direction_fits=(
            unregularized_recession_direction_fits
        ),
        recession_direction_fits=recession_direction_fits,
        cap_hits=cap_hits,
    )
    _atomic_write_json(output / "summary.json", summary.to_dict())
    return summary


def read_weekly_record(
    path: Path,
    *,
    panel: PublicHedgingPanel,
    spec: WeeklyEvaluationSpec,
    origin_date: date,
) -> tuple[WeeklyModelRun, ...]:
    """Read and revalidate one atomic weekly origin record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != WEEKLY_RECORD_SCHEMA
        or payload.get("run_identity") != spec.run_identity
        or payload.get("origin_date") != origin_date.isoformat()
        or not isinstance(payload.get("runs"), list)
    ):
        raise ValueError(f"invalid or incompatible weekly record {path}")
    runs = tuple(_weekly_model_run_from_dict(value) for value in payload["runs"])
    diagnostics = _validate_runs(
        runs,
        panel=panel,
        spec=spec,
        origin_date=origin_date,
    )
    expected_diagnostics = {
        "fit_cap_hits": diagnostics.fit_cap_hits,
        "executed_cap_hits": diagnostics.executed_cap_hits,
        "nonconverged_fits": diagnostics.nonconverged_fits,
        "unregularized_recession_direction_fits": (
            diagnostics.unregularized_recession_direction_fits
        ),
        "recession_direction_fits": diagnostics.recession_direction_fits,
        "cap_hits": diagnostics.cap_hits,
    }
    if payload.get("numerical_diagnostics") != expected_diagnostics:
        raise ValueError("weekly record numerical diagnostics are inconsistent")
    return runs


def _initialize_run(path: Path, spec: WeeklyEvaluationSpec) -> None:
    payload = {**spec.to_dict(), "run_identity": spec.run_identity}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("weekly output directory belongs to another run identity")
        return
    _atomic_write_json(path, payload)


def _write_weekly_record(
    path: Path,
    *,
    spec: WeeklyEvaluationSpec,
    origin_date: date,
    runs: tuple[WeeklyModelRun, ...],
    diagnostics: WeeklyNumericalDiagnostics,
) -> None:
    _atomic_write_json(
        path,
        {
            "schema": WEEKLY_RECORD_SCHEMA,
            "run_identity": spec.run_identity,
            "origin_date": origin_date.isoformat(),
            "numerical_diagnostics": {
                "fit_cap_hits": diagnostics.fit_cap_hits,
                "executed_cap_hits": diagnostics.executed_cap_hits,
                "nonconverged_fits": diagnostics.nonconverged_fits,
                "unregularized_recession_direction_fits": (
                    diagnostics.unregularized_recession_direction_fits
                ),
                "recession_direction_fits": diagnostics.recession_direction_fits,
                "cap_hits": diagnostics.cap_hits,
            },
            "runs": [_weekly_model_run_to_dict(run) for run in runs],
        },
    )


def _validate_runs(
    runs: tuple[WeeklyModelRun, ...],
    *,
    panel: PublicHedgingPanel,
    spec: WeeklyEvaluationSpec,
    origin_date: date,
    enforce_numerical_gate: bool = True,
) -> WeeklyNumericalDiagnostics:
    if tuple(run.result.target_label for run in runs) != spec.target_labels:
        raise ValueError("weekly model did not emit the precommitted target book")
    decision_dates, terminal_date = _expected_episode_dates(panel, origin_date)
    terminal_dates = {run.result.terminal_date for run in runs}
    if terminal_dates != {terminal_date}:
        raise ValueError("weekly results do not share one valid terminal date")
    for run, expected_quantity in zip(
        runs,
        spec.target_quantities,
        strict=True,
    ):
        result = run.result
        if result.schedule is not spec.schedule or result.origin_date != origin_date:
            raise ValueError("weekly result identity does not match its run specification")
        if not math.isclose(
            result.breakdown.claim_quantity,
            expected_quantity,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError("weekly claim quantity does not match its run specification")
        if not math.isclose(
            result.breakdown.payoff,
            result.terminal_value,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ) or not math.isclose(
            result.breakdown.model_price,
            float(run.fit_long_prices_origin_pv[0]),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError("weekly price or payoff audit trail does not reconcile")
        fit_dates = tuple(date.fromisoformat(value) for value in run.fit_as_of_dates)
        if spec.model in {"gromm", "volgan", "ad-seq-vol", "ad-seq-vol-ft"} and any(
            not fingerprints for fingerprints in run.market_artifact_fingerprints
        ):
            raise ValueError("weekly generated-model records require market artifacts")
        if spec.model == "gromm" and any(
            not fingerprints for fingerprints in run.policy_artifact_fingerprints
        ):
            raise ValueError("weekly GROMM records require policy checkpoints")
        first_market_price = (
            run.fit_price_records[0].common_market_price_per_fit_spot
        )
        if first_market_price is None or not math.isclose(
            first_market_price,
            result.breakdown.market_price,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError("weekly common market price does not reconcile")
        if spec.schedule is HedgeSchedule.FROZEN_WEEKLY:
            if (
                fit_dates != (origin_date,) * 5
                or len(set(run.market_fit_identities)) != 1
                or len(set(run.policy_fit_identities)) != 1
            ):
                raise ValueError("frozen-weekly run contains a restarted fit")
        elif (
            fit_dates != decision_dates
            or len(set(run.market_fit_identities)) != 5
            or len(set(run.policy_fit_identities)) != 5
        ):
            raise ValueError("restarted-weekly fit dates are incomplete or unordered")
    diagnostics = weekly_numerical_diagnostics(
        runs,
        position_cap=spec.position_cap,
    )
    if enforce_numerical_gate:
        _enforce_weekly_numerical_gate(
            diagnostics,
            require_no_cap_hits=spec.require_no_cap_hits,
        )
    return diagnostics


def _enforce_weekly_numerical_gate(
    diagnostics: WeeklyNumericalDiagnostics,
    *,
    require_no_cap_hits: bool,
) -> None:
    if diagnostics.recession_direction_fits:
        raise RuntimeError(
            "weekly numerical gate found "
            f"{diagnostics.recession_direction_fits} mean-volatility "
            "recession-direction fits"
        )
    if require_no_cap_hits and diagnostics.cap_hits:
        raise RuntimeError(
            f"weekly numerical gate found {diagnostics.cap_hits} position-cap hits"
        )
    if diagnostics.nonconverged_fits:
        raise RuntimeError(
            "weekly numerical gate found "
            f"{diagnostics.nonconverged_fits} nonconverged policy fits"
        )


def _expected_episode_dates(
    panel: PublicHedgingPanel,
    origin_date: date,
) -> tuple[tuple[date, ...], date]:
    """Return the exact five decision dates and shared settlement date."""
    try:
        origin_index = panel.dates.index(origin_date)
    except ValueError as error:
        raise ValueError("weekly origin is absent from the public panel") from error
    episode_dates = panel.dates[origin_index : origin_index + 6]
    if len(episode_dates) != 6:
        raise ValueError("weekly origin lacks five subsequent public sessions")
    return tuple(episode_dates[:5]), episode_dates[5]


def weekly_numerical_diagnostics(
    runs: tuple[WeeklyModelRun, ...],
    *,
    position_cap: float,
) -> WeeklyNumericalDiagnostics:
    fit_diagnostics: dict[str, tuple[bool, int, bool, bool]] = {}
    for run in runs:
        for (
            identity,
            converged,
            cap_hits,
            unregularized_recession_direction,
            recession_direction,
        ) in zip(
            run.policy_fit_identities,
            run.policy_fit_converged,
            run.policy_fit_cap_hits,
            run.policy_fit_unregularized_recession_directions,
            run.policy_fit_recession_directions,
            strict=True,
        ):
            diagnostics = (
                converged,
                cap_hits,
                unregularized_recession_direction,
                recession_direction,
            )
            previous = fit_diagnostics.setdefault(identity, diagnostics)
            if previous != diagnostics:
                raise ValueError("shared weekly policy fit diagnostics changed")
    threshold = position_cap * (1.0 - 1.0e-10)
    executed_cap_hits = sum(
        int(np.count_nonzero(np.abs(run.spot_positions) >= threshold))
        + int(np.count_nonzero(np.abs(run.option_positions) >= threshold))
        for run in runs
    )
    return WeeklyNumericalDiagnostics(
        fit_cap_hits=sum(value[1] for value in fit_diagnostics.values()),
        executed_cap_hits=executed_cap_hits,
        nonconverged_fits=sum(not value[0] for value in fit_diagnostics.values()),
        unregularized_recession_direction_fits=sum(
            value[2] for value in fit_diagnostics.values()
        ),
        recession_direction_fits=sum(
            value[3] for value in fit_diagnostics.values()
        ),
    )


def _weekly_model_run_to_dict(run: WeeklyModelRun) -> dict[str, object]:
    result = run.result
    return {
        "result": {
            "schedule": result.schedule.value,
            "target_label": result.target_label,
            "origin_date": result.origin_date.isoformat(),
            "terminal_date": result.terminal_date.isoformat(),
            "terminal_value": result.terminal_value,
            "hedge_gain": result.hedge_gain,
            "transaction_cost": result.transaction_cost,
            "spot_transaction_cost": result.spot_transaction_cost,
            "option_transaction_cost": result.option_transaction_cost,
            "spot_gain": result.spot_gain,
            "option_gain": result.option_gain,
            "step_transaction_costs": result.step_transaction_costs.tolist(),
            "step_hedge_gains": result.step_hedge_gains.tolist(),
            "breakdown": asdict(result.breakdown),
        },
        "market_fit_identities": list(run.market_fit_identities),
        "policy_fit_identities": list(run.policy_fit_identities),
        "market_artifact_fingerprints": [
            list(values) for values in run.market_artifact_fingerprints
        ],
        "policy_artifact_fingerprints": [
            list(values) for values in run.policy_artifact_fingerprints
        ],
        "fit_as_of_dates": list(run.fit_as_of_dates),
        "fit_price_records": [asdict(record) for record in run.fit_price_records],
        "policy_fit_converged": list(run.policy_fit_converged),
        "policy_fit_cap_hits": list(run.policy_fit_cap_hits),
        "policy_fit_unregularized_recession_directions": list(
            run.policy_fit_unregularized_recession_directions
        ),
        "policy_fit_recession_directions": list(
            run.policy_fit_recession_directions
        ),
        "fit_long_prices_origin_pv": run.fit_long_prices_origin_pv.tolist(),
        "spot_positions": run.spot_positions.tolist(),
        "option_positions": run.option_positions.tolist(),
    }


def _weekly_model_run_from_dict(payload: object) -> WeeklyModelRun:
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise ValueError("weekly model run must be a JSON object")
    result_payload = payload["result"]
    breakdown_payload = result_payload.get("breakdown")
    if not isinstance(breakdown_payload, dict):
        raise ValueError("weekly model run lacks a P&L breakdown")
    result = WeeklyHedgeResult(
        schedule=HedgeSchedule(_required_string(result_payload, "schedule")),
        target_label=_required_string(result_payload, "target_label"),
        origin_date=_required_date(result_payload, "origin_date"),
        terminal_date=_required_date(result_payload, "terminal_date"),
        terminal_value=_required_float(result_payload, "terminal_value"),
        hedge_gain=_required_float(result_payload, "hedge_gain"),
        transaction_cost=_required_float(result_payload, "transaction_cost"),
        spot_transaction_cost=_required_float(
            result_payload, "spot_transaction_cost"
        ),
        option_transaction_cost=_required_float(
            result_payload, "option_transaction_cost"
        ),
        spot_gain=_required_float(result_payload, "spot_gain"),
        option_gain=_required_float(result_payload, "option_gain"),
        step_transaction_costs=np.asarray(
            _required_float_tuple(result_payload, "step_transaction_costs"),
            dtype=np.float64,
        ),
        step_hedge_gains=np.asarray(
            _required_float_tuple(result_payload, "step_hedge_gains"),
            dtype=np.float64,
        ),
        breakdown=ClaimPnlBreakdown(
            **{
                name: _required_float(breakdown_payload, name)
                for name in ClaimPnlBreakdown.__dataclass_fields__
            }
        ),
    )
    price_records_payload = payload.get("fit_price_records")
    if not isinstance(price_records_payload, list) or any(
        not isinstance(value, dict) for value in price_records_payload
    ):
        raise ValueError("weekly price records must be an object array")
    return WeeklyModelRun(
        result=result,
        market_fit_identities=_required_string_tuple(
            payload, "market_fit_identities"
        ),
        policy_fit_identities=_required_string_tuple(
            payload, "policy_fit_identities"
        ),
        market_artifact_fingerprints=_required_string_matrix(
            payload, "market_artifact_fingerprints"
        ),
        policy_artifact_fingerprints=_required_string_matrix(
            payload, "policy_artifact_fingerprints"
        ),
        fit_as_of_dates=_required_string_tuple(payload, "fit_as_of_dates"),
        fit_price_records=tuple(
            WeeklyPriceRecord(**value) for value in price_records_payload
        ),
        policy_fit_converged=_required_boolean_tuple(
            payload, "policy_fit_converged"
        ),
        policy_fit_cap_hits=_required_integer_tuple(
            payload, "policy_fit_cap_hits"
        ),
        policy_fit_unregularized_recession_directions=_required_boolean_tuple(
            payload, "policy_fit_unregularized_recession_directions"
        ),
        policy_fit_recession_directions=_required_boolean_tuple(
            payload, "policy_fit_recession_directions"
        ),
        fit_long_prices_origin_pv=np.asarray(
            _required_float_tuple(payload, "fit_long_prices_origin_pv"),
            dtype=np.float64,
        ),
        spot_positions=np.asarray(
            _required_float_tuple(payload, "spot_positions"),
            dtype=np.float64,
        ),
        option_positions=np.asarray(
            _required_float_matrix(payload, "option_positions"),
            dtype=np.float64,
        ),
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
        raise ValueError(f"weekly metadata field {name!r} must be a non-empty string")
    return value


def _required_date(payload: dict[str, object], name: str) -> date:
    value = _required_string(payload, name)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"weekly metadata field {name!r} must be an ISO date"
        ) from error


def _required_integer(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"weekly metadata field {name!r} must be an integer")
    return value


def _required_boolean(payload: dict[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"weekly metadata field {name!r} must be a boolean")
    return value


def _required_float(payload: dict[str, object], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"weekly metadata field {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"weekly metadata field {name!r} must be finite")
    return result


def _required_string_tuple(
    payload: dict[str, object],
    name: str,
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"weekly metadata field {name!r} must be a string array")
    return tuple(value)


def _required_boolean_tuple(
    payload: dict[str, object],
    name: str,
) -> tuple[bool, ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"weekly metadata field {name!r} must be a boolean array")
    return tuple(value)


def _required_integer_tuple(
    payload: dict[str, object],
    name: str,
) -> tuple[int, ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"weekly metadata field {name!r} must be an integer array")
    return tuple(value)


def _required_float_tuple(
    payload: dict[str, object],
    name: str,
) -> tuple[float, ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"weekly metadata field {name!r} must be a finite numeric array")
    return tuple(float(item) for item in value)


def _required_string_matrix(
    payload: dict[str, object],
    name: str,
) -> tuple[tuple[str, ...], ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"weekly metadata field {name!r} must be a string matrix")
    rows: list[tuple[str, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in row
        ):
            raise ValueError(f"weekly metadata field {name!r} must be a string matrix")
        rows.append(tuple(row))
    return tuple(rows)


def _required_float_matrix(
    payload: dict[str, object],
    name: str,
) -> tuple[tuple[float, ...], ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"weekly metadata field {name!r} must be a numeric matrix")
    rows = tuple(
        _required_float_tuple({name: row}, name)
        for row in value
    )
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"weekly metadata field {name!r} must be rectangular")
    return rows
