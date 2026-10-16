"""Machine-readable evidence for precommitted numerical-gate failures."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]

NUMERICAL_GATE_FAILURE_SCHEMA = "hedging-numerical-gate-failure-v2"
LEGACY_NUMERICAL_GATE_FAILURE_SCHEMA = "hedging-numerical-gate-failure-v1"
NUMERICAL_GATE_FAILURE_FILENAME = "numerical-gate-failure.json"
NUMERICAL_GATE_CLASSIFICATIONS = frozenset(
    {
        "position-cap-hit",
        "policy-fit-cap-hit",
        "nonconverged-fit",
        "mean-volatility-recession-direction",
    }
)


def write_numerical_gate_failure(
    *,
    output_directory: Path,
    run_identity: str,
    evaluation_schema: str,
    model: str,
    measure: str,
    dataset_manifest_sha256: str,
    origin_date: date,
    diagnostics: JsonObject,
    schedule: str | None = None,
    require_no_fit_cap_hits: bool = True,
    require_no_executed_cap_hits: bool = True,
) -> Path:
    """Atomically record the exact strict gate rejected by an evaluation runner."""
    payload = numerical_gate_failure_payload(
        run_identity=run_identity,
        evaluation_schema=evaluation_schema,
        model=model,
        measure=measure,
        dataset_manifest_sha256=dataset_manifest_sha256,
        origin_date=origin_date,
        diagnostics=diagnostics,
        schedule=schedule,
        require_no_fit_cap_hits=require_no_fit_cap_hits,
        require_no_executed_cap_hits=require_no_executed_cap_hits,
    )
    path = Path(output_directory) / NUMERICAL_GATE_FAILURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def numerical_gate_failure_payload(
    *,
    run_identity: str,
    evaluation_schema: str,
    model: str,
    measure: str,
    dataset_manifest_sha256: str,
    origin_date: date,
    diagnostics: JsonObject,
    schedule: str | None = None,
    require_no_fit_cap_hits: bool = True,
    require_no_executed_cap_hits: bool = True,
) -> JsonObject:
    """Build and validate one deterministic numerical-gate failure payload."""
    _require_nonempty(run_identity, "run_identity")
    _require_nonempty(evaluation_schema, "evaluation_schema")
    _require_nonempty(model, "model")
    _require_nonempty(measure, "measure")
    _require_sha256(dataset_manifest_sha256, "dataset_manifest_sha256")
    if schedule is not None:
        _require_nonempty(schedule, "schedule")
    if (
        not isinstance(require_no_fit_cap_hits, bool)
        or not isinstance(require_no_executed_cap_hits, bool)
    ):
        raise ValueError("numerical cap-hit gate controls must be boolean")
    normalized_diagnostics = _normalized_diagnostics(diagnostics)
    gate_policy = {
        "require_no_fit_cap_hits": require_no_fit_cap_hits,
        "require_no_executed_cap_hits": require_no_executed_cap_hits,
    }
    triggered_gates = _triggered_gates(
        normalized_diagnostics,
        require_no_fit_cap_hits=require_no_fit_cap_hits,
        require_no_executed_cap_hits=require_no_executed_cap_hits,
    )
    if not triggered_gates:
        raise ValueError("numerical-gate failure has no triggered gate")
    classification = triggered_gates[0]
    summary = _failure_summary(normalized_diagnostics, triggered_gates)
    return {
        "schema": NUMERICAL_GATE_FAILURE_SCHEMA,
        "run_identity": run_identity,
        "evaluation_schema": evaluation_schema,
        "model": model,
        "measure": measure,
        "schedule": schedule,
        "dataset_manifest_sha256": dataset_manifest_sha256.lower(),
        "origin_date": origin_date.isoformat(),
        "classification": classification,
        "triggered_gates": list(triggered_gates),
        "gate_policy": gate_policy,
        "summary": summary,
        "diagnostics": normalized_diagnostics,
    }


def load_numerical_gate_failure(path: Path) -> JsonObject:
    """Load and independently validate one runner-emitted failure artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("numerical-gate failure artifact must be an object")
    schema = payload.get("schema")
    if schema not in {
        NUMERICAL_GATE_FAILURE_SCHEMA,
        LEGACY_NUMERICAL_GATE_FAILURE_SCHEMA,
    }:
        raise ValueError("unexpected numerical-gate failure schema")
    _require_nonempty(payload.get("run_identity"), "run_identity")
    _require_nonempty(payload.get("evaluation_schema"), "evaluation_schema")
    _require_nonempty(payload.get("model"), "model")
    _require_nonempty(payload.get("measure"), "measure")
    _require_sha256(
        payload.get("dataset_manifest_sha256"),
        "dataset_manifest_sha256",
    )
    schedule = payload.get("schedule")
    if schedule is not None:
        _require_nonempty(schedule, "schedule")
    origin = _require_nonempty(payload.get("origin_date"), "origin_date")
    date.fromisoformat(origin)
    diagnostics_value = payload.get("diagnostics")
    if not isinstance(diagnostics_value, dict):
        raise ValueError("numerical-gate diagnostics must be an object")
    diagnostics = _normalized_diagnostics(diagnostics_value)
    if diagnostics != diagnostics_value:
        raise ValueError("numerical-gate diagnostics are not canonical")
    gate_policy_value = payload.get("gate_policy")
    if (
        schema == LEGACY_NUMERICAL_GATE_FAILURE_SCHEMA
        and gate_policy_value is None
    ):
        gate_policy_value = {"require_no_cap_hits": True}
    if not isinstance(gate_policy_value, dict):
        raise ValueError("numerical-gate policy must be an object")
    if schema == LEGACY_NUMERICAL_GATE_FAILURE_SCHEMA:
        if set(gate_policy_value) != {"require_no_cap_hits"} or not isinstance(
            gate_policy_value["require_no_cap_hits"],
            bool,
        ):
            raise ValueError("legacy numerical-gate policy is not canonical")
        require_no_fit_cap_hits = gate_policy_value["require_no_cap_hits"]
        require_no_executed_cap_hits = gate_policy_value["require_no_cap_hits"]
    else:
        expected_policy_fields = {
            "require_no_fit_cap_hits",
            "require_no_executed_cap_hits",
        }
        if set(gate_policy_value) != expected_policy_fields or not all(
            isinstance(gate_policy_value[name], bool)
            for name in expected_policy_fields
        ):
            raise ValueError("numerical-gate policy is not canonical")
        require_no_fit_cap_hits = gate_policy_value["require_no_fit_cap_hits"]
        require_no_executed_cap_hits = gate_policy_value[
            "require_no_executed_cap_hits"
        ]
    triggered_gates = _triggered_gates(
        diagnostics,
        require_no_fit_cap_hits=require_no_fit_cap_hits,
        require_no_executed_cap_hits=require_no_executed_cap_hits,
    )
    if not triggered_gates:
        raise ValueError("numerical-gate failure has no triggered gate")
    if payload.get("triggered_gates") != list(triggered_gates):
        raise ValueError("numerical-gate triggered gates are inconsistent")
    if payload.get("classification") != triggered_gates[0]:
        raise ValueError("numerical-gate classification is inconsistent")
    if payload.get("summary") != _failure_summary(diagnostics, triggered_gates):
        raise ValueError("numerical-gate summary is inconsistent")
    return payload


def _normalized_diagnostics(diagnostics: JsonObject) -> JsonObject:
    allowed = {
        "cap_hits",
        "executed_cap_hits",
        "fit_cap_hits",
        "nonconverged_fits",
        "nonconverged_results",
        "recession_direction_fits",
        "unregularized_recession_direction_fits",
    }
    unknown = set(diagnostics) - allowed
    if unknown:
        raise ValueError(f"unsupported numerical-gate diagnostics: {sorted(unknown)}")
    normalized: JsonObject = {}
    for name in sorted(diagnostics):
        value = diagnostics[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"numerical-gate diagnostic {name} must be non-negative")
        normalized[name] = value
    daily = {"cap_hits", "nonconverged_results"}
    daily_with_recession = daily | {"recession_direction_fits"}
    daily_with_warnings = daily | {"unregularized_recession_direction_fits"}
    daily_with_recession_and_warnings = daily_with_recession | {
        "unregularized_recession_direction_fits"
    }
    daily_split = {
        "cap_hits",
        "executed_cap_hits",
        "fit_cap_hits",
        "nonconverged_results",
    }
    weekly = {
        "cap_hits",
        "executed_cap_hits",
        "fit_cap_hits",
        "nonconverged_fits",
        "recession_direction_fits",
        "unregularized_recession_direction_fits",
    }
    fields = set(normalized)
    if fields in (
        daily,
        daily_with_recession,
        daily_with_warnings,
        daily_with_recession_and_warnings,
    ):
        return normalized
    optional_daily_split = {
        "recession_direction_fits",
        "unregularized_recession_direction_fits",
    }
    if fields in tuple(
        daily_split | included
        for included in (
            set(),
            {"recession_direction_fits"},
            {"unregularized_recession_direction_fits"},
            optional_daily_split,
        )
    ):
        if normalized["cap_hits"] != (
            normalized["fit_cap_hits"] + normalized["executed_cap_hits"]
        ):
            raise ValueError("daily numerical-gate cap-hit counts do not reconcile")
        return normalized
    optional_weekly = {
        "recession_direction_fits",
        "unregularized_recession_direction_fits",
    }
    if fields in tuple(
        weekly - omitted
        for omitted in (
            set(),
            {"recession_direction_fits"},
            {"unregularized_recession_direction_fits"},
            optional_weekly,
        )
    ):
        if normalized["cap_hits"] != (
            normalized["fit_cap_hits"] + normalized["executed_cap_hits"]
        ):
            raise ValueError("weekly numerical-gate cap-hit counts do not reconcile")
        return normalized
    raise ValueError("numerical-gate diagnostics have an unsupported scope")


def _triggered_gates(
    diagnostics: JsonObject,
    *,
    require_no_fit_cap_hits: bool,
    require_no_executed_cap_hits: bool,
) -> tuple[str, ...]:
    triggered: list[str] = []
    if diagnostics.get("recession_direction_fits", 0):
        triggered.append("mean-volatility-recession-direction")
    if require_no_executed_cap_hits and diagnostics.get(
        "executed_cap_hits",
        diagnostics.get("cap_hits", 0),
    ):
        triggered.append("position-cap-hit")
    if require_no_fit_cap_hits and diagnostics.get("fit_cap_hits", 0):
        triggered.append("policy-fit-cap-hit")
    if diagnostics.get("nonconverged_fits", diagnostics.get("nonconverged_results", 0)):
        triggered.append("nonconverged-fit")
    return tuple(triggered)


def _failure_summary(
    diagnostics: JsonObject,
    triggered_gates: tuple[str, ...],
) -> str:
    counts = ", ".join(
        f"{name}={diagnostics[name]}" for name in sorted(diagnostics)
    )
    return f"strict numerical gate rejected {', '.join(triggered_gates)} ({counts})"


def _require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    text = _require_nonempty(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a SHA-256")
    return text
