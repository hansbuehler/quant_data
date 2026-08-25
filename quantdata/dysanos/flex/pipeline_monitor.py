"""Dependency-aware persistent supervisor for the FLEX Phase 3 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from cdxcore.err import verify_inp
from cdxcore.util import is_filename

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.monitor_services import (
    _atomic_write_json,
    _cpu_usage_ns,
    _has_recent_journal,
    _quantdata_resource_status,
    _run_systemctl,
    _service_properties,
)

_ACCEPTED_COMPLETION_STATUSES = frozenset({"complete", "native_complete"})
_DETERMINISTIC_EXCEPTION_TYPES = frozenset(
    {
        "AssertionError",
        "AttributeError",
        "KeyError",
        "NotImplementedError",
        "TrainingConvergenceError",
        "TypeError",
        "ValueError",
    }
)


@dataclass(frozen=True)
class DependencySpec:
    """An upstream stage and the proof which releases this dependency."""

    upstream: str
    proof_path: Path | None = None

    def __post_init__(self) -> None:
        verify_inp(
            is_filename(self.upstream),
            lambda: f"Invalid upstream stage name {self.upstream!r}",
        )
        verify_inp(
            self.proof_path is None or self.proof_path.is_absolute(),
            lambda: f"Dependency proof must be absolute: {self.proof_path}",
        )


@dataclass(frozen=True)
class PipelineStageSpec:
    """One systemd worker, its final proof, and its dependency gates."""

    name: str
    unit: str
    completion_path: Path
    dependencies: tuple[DependencySpec, ...] = ()
    worker_heartbeat_path: Path | None = None
    expected_generation: str | None = None

    def __post_init__(self) -> None:
        verify_inp(
            is_filename(self.name),
            lambda: f"Invalid pipeline stage name {self.name!r}",
        )
        verify_inp(
            self.unit.endswith(".service") and "/" not in self.unit,
            lambda: f"Invalid systemd service unit {self.unit!r}",
        )
        verify_inp(
            self.completion_path.is_absolute(),
            lambda: f"Completion path must be absolute: {self.completion_path}",
        )
        verify_inp(
            self.worker_heartbeat_path is None
            or self.worker_heartbeat_path.is_absolute(),
            lambda: (
                "Worker heartbeat path must be absolute: "
                f"{self.worker_heartbeat_path}"
            ),
        )
        verify_inp(
            self.expected_generation is None
            or (
                isinstance(self.expected_generation, str)
                and len(self.expected_generation) > 0
            ),
            lambda: "Expected generation must be a non-empty string",
        )


@dataclass(frozen=True)
class CompletionEvidence:
    """Parsed state of an immutable completion proof."""

    state: str
    error: str | None = None
    schema: str | None = None
    proof_sha256: str | None = None
    generation: str | None = None
    validated_artifacts: int = 0


@dataclass(frozen=True)
class StageObservation:
    """One poll's process and artifact evidence for a stage."""

    completion: CompletionEvidence
    active: bool
    properties: Mapping[str, str]
    recent_journal: bool
    cpu_usage_ns: int | None
    cpu_advanced: bool
    command_line: tuple[str, ...]
    newest_failure_path: Path | None
    failure_type: str | None = None
    failure_message: str | None = None
    progress_value: object | None = None
    last_progress_at: str | None = None
    progress_advanced: bool = False
    progress_recent: bool = False
    progress_source: str | None = None
    progress_error: str | None = None
    worker_heartbeat_at: str | None = None
    worker_generation: str | None = None
    worker_heartbeat_error: str | None = None
    worker_heartbeat_status_error: str | None = None


@dataclass(frozen=True)
class PipelineClassification:
    """Pure classification result before restart side effects."""

    stages: tuple[dict[str, Any], ...]
    restart_names: tuple[str, ...]
    stop_names: tuple[str, ...]
    terminal_names: tuple[str, ...]
    all_complete: bool


@dataclass(frozen=True)
class ProgressEvidence:
    """Checkpoint or worker-heartbeat progress observed in one monitor cycle."""

    value: object | None
    source: str | None
    error: str | None = None


@dataclass(frozen=True)
class WorkerHeartbeatEvidence:
    """Optional future worker-owned heartbeat and fencing evidence."""

    timestamp_utc: str | None
    generation: str | None
    progress_value: object | None
    error: str | None = None


def _event_log_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.events.jsonl")


def _heartbeat_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.heartbeat.json")


def _manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.manifest.json")


def _pipeline_manifest_payload(
    stages: tuple[PipelineStageSpec, ...],
    *,
    poll_seconds: int,
    stale_seconds: int,
    max_restarts: int,
) -> dict[str, object]:
    """Return the aggregate durable DAG and recovery-policy manifest."""
    return {
        "schema": "flex-pipeline-manifest-v1",
        "stages": [
            {
                "name": stage.name,
                "unit": stage.unit,
                "completion_path": str(stage.completion_path),
                "worker_heartbeat_path": (
                    None
                    if stage.worker_heartbeat_path is None
                    else str(stage.worker_heartbeat_path)
                ),
                "expected_generation": stage.expected_generation,
                "dependencies": [
                    {
                        "upstream": dependency.upstream,
                        "proof_path": (
                            None
                            if dependency.proof_path is None
                            else str(dependency.proof_path)
                        ),
                    }
                    for dependency in stage.dependencies
                ],
            }
            for stage in stages
        ],
        "policy": {
            "poll_seconds": poll_seconds,
            "stale_seconds": stale_seconds,
            "required_consecutive_stale_observations": 2,
            "max_restarts": max_restarts,
            "accepted_completion_statuses": sorted(_ACCEPTED_COMPLETION_STATUSES),
            "semantic_completion_validation": True,
            "checkpoint_progress_persistence": True,
        },
        "future_launch_requirements": {
            "worker_heartbeat_and_generation": (
                "use the five-field --stage form and write the same generation "
                "into worker heartbeat and completion proof"
            ),
            "systemd_durability": (
                "launch persistent supervisor and watchdog units with an OS-level "
                "Restart=on-failure policy; an agent session is not durability"
            ),
        },
    }


def _append_json_event(payload: Mapping[str, object], path: Path) -> None:
    """Append and fsync one compact event without rewriting prior history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, allow_nan=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o664,
    )
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def parse_stage_spec(value: str) -> PipelineStageSpec:
    """Parse a stage and optional future worker-heartbeat/fencing fields.

    Existing launches retain the three-field form. Future durable workers may
    add ``::ABSOLUTE_WORKER_HEARTBEAT::GENERATION_TOKEN``; the monitor then
    rejects heartbeats and completion proofs from an obsolete generation.
    """
    fields = value.split("::")
    verify_inp(
        len(fields) in {3, 5} and all(fields),
        lambda: (
            "Expected NAME::UNIT::ABSOLUTE_COMPLETION_PATH"
            "[::ABSOLUTE_WORKER_HEARTBEAT::GENERATION], found "
            f"{value!r}"
        ),
    )
    return PipelineStageSpec(
        name=fields[0],
        unit=fields[1],
        completion_path=Path(fields[2]),
        worker_heartbeat_path=(None if len(fields) == 3 else Path(fields[3])),
        expected_generation=(None if len(fields) == 3 else fields[4]),
    )


def parse_dependency_spec(value: str) -> tuple[str, DependencySpec]:
    """Parse ``STAGE::UPSTREAM[::ABSOLUTE_GATE_PROOF]``."""
    fields = value.split("::")
    verify_inp(
        len(fields) in {2, 3} and all(fields),
        lambda: f"Expected STAGE::UPSTREAM[::ABSOLUTE_PROOF], found {value!r}",
    )
    proof = None if len(fields) == 2 else Path(fields[2])
    verify_inp(
        is_filename(fields[0]),
        lambda: f"Invalid dependent stage name {fields[0]!r}",
    )
    return fields[0], DependencySpec(upstream=fields[1], proof_path=proof)


def attach_dependencies(
    stages: tuple[PipelineStageSpec, ...],
    dependencies: tuple[tuple[str, DependencySpec], ...],
) -> tuple[PipelineStageSpec, ...]:
    """Attach parsed dependency flags and validate the resulting DAG."""
    stage_by_name = {stage.name: stage for stage in stages}
    verify_inp(
        len(stage_by_name) == len(stages),
        lambda: "Pipeline stage names must be unique",
    )
    attached: dict[str, list[DependencySpec]] = {name: [] for name in stage_by_name}
    for dependent, dependency in dependencies:
        verify_inp(
            dependent in stage_by_name,
            lambda: f"Unknown dependent pipeline stage {dependent!r}",
        )
        verify_inp(
            dependency.upstream in stage_by_name,
            lambda: f"Unknown upstream pipeline stage {dependency.upstream!r}",
        )
        attached[dependent].append(dependency)
    result = tuple(
        PipelineStageSpec(
            name=stage.name,
            unit=stage.unit,
            completion_path=stage.completion_path,
            dependencies=tuple(attached[stage.name]),
            worker_heartbeat_path=stage.worker_heartbeat_path,
            expected_generation=stage.expected_generation,
        )
        for stage in stages
    )
    _topological_stages(result)
    return result


def _topological_stages(
    stages: tuple[PipelineStageSpec, ...],
) -> tuple[PipelineStageSpec, ...]:
    stage_by_name = {stage.name: stage for stage in stages}
    remaining = set(stage_by_name)
    ordered: list[PipelineStageSpec] = []
    completed: set[str] = set()
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if all(
                dependency.upstream in completed
                for dependency in stage_by_name[name].dependencies
            )
        )
        verify_inp(
            len(ready) > 0,
            lambda: f"Pipeline dependencies contain a cycle: {sorted(remaining)}",
        )
        for name in ready:
            ordered.append(stage_by_name[name])
            completed.add(name)
            remaining.remove(name)
    return tuple(ordered)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_json_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(name, str) and _finite_json_tree(child)
            for name, child in value.items()
        )
    if isinstance(value, list):
        return all(_finite_json_tree(child) for child in value)
    return value is None or isinstance(value, (str, int, bool))


def _validate_common_coverage(payload: Mapping[str, object]) -> str | None:
    """Validate common count/date conventions in reports and completion proofs."""
    count_to_rows = {
        "num_days": "market_dates",
        "num_train_dates": "train_dates",
        "num_validation_dates": "validation_dates",
    }
    for count_name, row_name in count_to_rows.items():
        count = payload.get(count_name)
        rows = payload.get(row_name)
        if count is not None and (
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
        ):
            return f"{count_name} must be a positive integer, found {count!r}"
        if rows is not None:
            if not isinstance(rows, list) or len(rows) == 0:
                return f"{row_name} must be a non-empty list"
            if len({str(value) for value in rows}) != len(rows):
                return f"{row_name} contains duplicate rows"
            if count is not None and len(rows) != count:
                return (
                    f"{count_name}={count} does not match "
                    f"len({row_name})={len(rows)}"
                )
    return None


def _validate_json_artifact(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"cannot read JSON artifact {path}: {error}"
    if not isinstance(payload, Mapping):
        return f"JSON artifact {path} is not an object"
    if not isinstance(payload.get("schema"), str) or not payload.get("schema"):
        return f"JSON artifact {path} has no schema"
    if not _finite_json_tree(payload):
        return f"JSON artifact {path} contains a non-finite number"
    coverage_error = _validate_common_coverage(payload)
    if coverage_error is not None:
        return f"JSON artifact {path}: {coverage_error}"
    if payload.get("schema") == "flex-joint-hierarchical-normal-q-sequence-v1":
        source_name = payload.get("source_q1_completion")
        source_digest = payload.get("source_q1_completion_sha256")
        stages = payload.get("stages")
        completed = payload.get("completed_roll_lines")
        if (
            payload.get("status") != "complete"
            or not isinstance(source_name, str)
            or not isinstance(source_digest, str)
            or not isinstance(stages, list)
            or not stages
            or completed != 1 + len(stages)
        ):
            return f"JSON artifact {path} has incomplete Q-sequence coverage"
        proof_names = [source_name]
        proof_digests = [source_digest]
        expected_roll_line = 2
        for stage in stages:
            if (
                not isinstance(stage, Mapping)
                or stage.get("roll_line") != expected_roll_line
                or stage.get("status") != "complete"
                or not isinstance(stage.get("completion"), str)
                or not isinstance(stage.get("completion_sha256"), str)
            ):
                return f"JSON artifact {path} has malformed Q stage {expected_roll_line}"
            proof_names.append(str(stage["completion"]))
            proof_digests.append(str(stage["completion_sha256"]))
            expected_roll_line += 1
        for proof_name, proof_digest in zip(
            proof_names, proof_digests, strict=True
        ):
            proof = Path(proof_name)
            if not proof.is_absolute() or not proof.is_file():
                return f"JSON artifact {path} references missing proof {proof}"
            if _sha256(proof) != proof_digest:
                return f"JSON artifact {path} has proof digest mismatch {proof}"
            evidence = _completion_evidence(proof)
            if evidence.state != "complete":
                return (
                    f"JSON artifact {path} has invalid nested proof {proof}: "
                    f"{evidence.error}"
                )
    return None


def _validate_torch_checkpoint(
    path: Path,
    *,
    allowed_phases: frozenset[str] = frozenset({"complete"}),
) -> str | None:
    """Validate completion-critical checkpoint provenance and tensor integrity."""
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - corrupted files have varied errors
        return f"cannot load checkpoint {path}: {type(error).__name__}: {error}"
    if not isinstance(payload, Mapping):
        return f"checkpoint {path} is not a mapping"
    if not isinstance(payload.get("schema"), str) and not isinstance(
        payload.get("schema_version"), int
    ):
        return f"checkpoint {path} has no schema identity"
    if payload.get("phase") not in allowed_phases:
        return (
            f"checkpoint {path} phase is {payload.get('phase')!r}, expected "
            f"one of {sorted(allowed_phases)}"
        )
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping) or len(model_state) == 0:
        daily_states = payload.get("daily_states")
        if isinstance(daily_states, torch.Tensor):
            model_state = {"daily_states": daily_states}
        else:
            return f"checkpoint {path} has no model or daily state"
    tensors = [value for value in model_state.values() if isinstance(value, torch.Tensor)]
    if len(tensors) != len(model_state):
        return f"checkpoint {path} model state contains non-tensor values"
    if any(value.numel() == 0 or not bool(torch.all(torch.isfinite(value))) for value in tensors):
        return f"checkpoint {path} contains empty or non-finite model tensors"
    daily = model_state.get("daily_state_embedding.weight")
    if daily is None:
        daily = model_state.get("daily_raw_embedding.weight")
    if daily is None:
        daily = model_state.get("daily_states")
    fingerprints = payload.get("market_fingerprints")
    if isinstance(daily, torch.Tensor) and isinstance(fingerprints, list):
        if daily.ndim != 2 or daily.shape[0] != len(fingerprints):
            return (
                f"checkpoint {path} daily-state coverage {tuple(daily.shape)} "
                f"does not match {len(fingerprints)} market fingerprints"
            )
    model_config = payload.get("model_config")
    if isinstance(daily, torch.Tensor) and isinstance(model_config, Mapping):
        state_size = model_config.get("state_size")
        if isinstance(state_size, int) and daily.shape[1] != state_size:
            return (
                f"checkpoint {path} daily-state width {daily.shape[1]} does "
                f"not match configured state_size={state_size}"
            )
    completed_epochs = payload.get("completed_epochs")
    history = payload.get("history")
    if payload.get("schema") == "flex-weekly-measure-kernel-v1":
        # Sequential Q checkpoints reset the within-line epoch counter after a
        # line is certified and advance ``line_index``. Their durable history
        # is therefore one completed record per line, not one record per epoch.
        line_index = payload.get("line_index")
        line_histories = payload.get("histories")
        if (
            not isinstance(completed_epochs, int)
            or completed_epochs != 0
            or not isinstance(line_index, int)
            or line_index <= 0
            or not isinstance(line_histories, list)
            or len(line_histories) != line_index
            or any(
                not isinstance(line, Mapping) or line.get("status") != "complete"
                for line in line_histories
            )
        ):
            return f"checkpoint {path} has inconsistent sequential-line coverage"
    elif completed_epochs is not None and (
        not isinstance(completed_epochs, int)
        or completed_epochs <= 0
        or not isinstance(history, list)
        or len(history) != completed_epochs
    ):
        return f"checkpoint {path} has inconsistent epoch/history coverage"
    return None


def _validate_hierarchical_continuation_base_checkpoint(
    path: Path,
    *,
    expected_digest: str,
    completion_payload: Mapping[str, object],
) -> str | None:
    """Validate the deliberately non-complete base of a completed continuation."""
    base_error = _validate_torch_checkpoint(
        path,
        allowed_phases=frozenset({"training"}),
    )
    if base_error is not None:
        return base_error
    final_name = completion_payload.get("validation_state_checkpoint")
    final_digest = completion_payload.get("validation_state_checkpoint_sha256")
    if not isinstance(final_name, str) or not isinstance(final_digest, str):
        return "hierarchical continuation proof has no final validation checkpoint"
    final_path = Path(final_name)
    if not final_path.is_absolute() or not final_path.is_file():
        return f"hierarchical continuation checkpoint is missing: {final_path}"
    if _sha256(final_path) != final_digest:
        return f"hierarchical continuation checkpoint digest mismatch: {final_path}"
    try:
        import torch

        base = torch.load(path, map_location="cpu", weights_only=False)
        final = torch.load(final_path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - corrupted files have varied errors
        return f"cannot load hierarchical continuation provenance: {error}"
    if not isinstance(base, Mapping) or not isinstance(final, Mapping):
        return "hierarchical continuation checkpoints are not mappings"
    if (
        final.get("schema")
        != "flex-quote-optimal-hierarchical-states-continuation-v1"
        or final.get("phase") != "complete"
    ):
        return "final hierarchical continuation checkpoint is not complete"
    provenance = final.get("continuation_provenance")
    if not isinstance(provenance, Mapping) or not _finite_json_tree(provenance):
        return "final hierarchical continuation checkpoint has invalid provenance"
    if provenance.get("source_state_checkpoint") != str(path) or (
        provenance.get("source_state_checkpoint_sha256") != expected_digest
    ):
        return "hierarchical continuation provenance does not bind the base checkpoint"
    completed_epochs = base.get("completed_epochs")
    if provenance.get("source_completed_epochs") != completed_epochs:
        return "hierarchical continuation provenance has inconsistent base epochs"
    history = base.get("history")
    if (
        not isinstance(history, list)
        or len(history) != completed_epochs
        or not history
        or provenance.get("source_final_metrics") != history[-1]
    ):
        return "hierarchical continuation provenance has inconsistent base history"
    convergence = provenance.get("source_convergence")
    if not isinstance(convergence, Mapping) or (
        convergence.get("converged") is not False
        or convergence.get("reason") != "trailing_window_not_converged"
        or convergence.get("best_epoch") != completed_epochs
    ):
        return "hierarchical continuation source was not a finite improving plateau failure"
    if provenance.get("optimizer_scheduler_reset") is not True:
        return "hierarchical continuation did not record an optimizer reset"
    return None


def _validate_torch_artifact(path: Path) -> str | None:
    """Validate a non-checkpoint Torch artifact's schema and finite tensors."""
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - corruption has varied errors
        return f"cannot load Torch artifact {path}: {type(error).__name__}: {error}"
    if not isinstance(payload, Mapping) or not isinstance(payload.get("schema"), str):
        return f"Torch artifact {path} has no schema identity"

    def tensors(value: object) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, Mapping):
            return [item for child in value.values() for item in tensors(child)]
        if isinstance(value, (list, tuple)):
            return [item for child in value for item in tensors(child)]
        return []

    values = tensors(payload)
    if not values:
        return f"Torch artifact {path} contains no tensors"
    if any(value.numel() == 0 or not bool(torch.all(torch.isfinite(value))) for value in values):
        return f"Torch artifact {path} contains empty or non-finite tensors"
    return None


def _completion_evidence(
    path: Path,
    *,
    expected_generation: str | None = None,
) -> CompletionEvidence:
    if not path.exists():
        return CompletionEvidence(state="missing")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as error:
        return CompletionEvidence(state="invalid", error=str(error))
    if not isinstance(payload, Mapping) or payload.get("status") not in (
        _ACCEPTED_COMPLETION_STATUSES
    ):
        status = payload.get("status") if isinstance(payload, Mapping) else None
        return CompletionEvidence(
            state="invalid",
            error=(
                f"completion status is {status!r}, expected one of "
                f"{sorted(_ACCEPTED_COMPLETION_STATUSES)}"
            ),
        )
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema:
        return CompletionEvidence(
            state="invalid",
            error="completion proof has no non-empty schema",
        )
    if not _finite_json_tree(payload):
        return CompletionEvidence(
            state="invalid",
            error="completion proof contains a non-finite number",
            schema=schema,
        )
    coverage_error = _validate_common_coverage(payload)
    if coverage_error is not None:
        return CompletionEvidence(
            state="invalid",
            error=coverage_error,
            schema=schema,
        )
    generation_value = next(
        (
            payload[name]
            for name in ("generation", "generation_token", "worker_generation")
            if isinstance(payload.get(name), str)
        ),
        None,
    )
    if expected_generation is not None and generation_value != expected_generation:
        return CompletionEvidence(
            state="invalid",
            error=(
                f"completion generation is {generation_value!r}, expected "
                f"{expected_generation!r}"
            ),
            schema=schema,
            generation=generation_value,
        )

    validated_artifacts = 0
    for digest_name, expected_digest in payload.items():
        if not digest_name.endswith("_sha256"):
            continue
        path_name = digest_name[: -len("_sha256")]
        artifact_name = payload.get(path_name)
        if artifact_name is None:
            continue
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            return CompletionEvidence(
                state="invalid",
                error=f"invalid {digest_name} in completion proof",
                schema=schema,
                generation=generation_value,
            )
        artifact = _resolve_completion_artifact(
            artifact_name,
            expected_digest,
        )
        if artifact is None:
            return CompletionEvidence(
                state="invalid",
                error=(
                    "completion artifact is missing or has a digest mismatch: "
                    f"{artifact_name}"
                ),
                schema=schema,
                generation=generation_value,
            )
        artifact_error = (
            _validate_json_artifact(artifact)
            if artifact.suffix == ".json"
            else (
                (
                    (
                        _validate_hierarchical_continuation_base_checkpoint(
                            artifact,
                            expected_digest=expected_digest,
                            completion_payload=payload,
                        )
                        if schema
                        == "flex-shared-strikes-joint-hierarchical-completion-v1"
                        and path_name == "validation_state_base_checkpoint"
                        else _validate_torch_checkpoint(artifact)
                    )
                    if "checkpoint" in path_name
                    else _validate_torch_artifact(artifact)
                )
                if artifact.suffix in {".pt", ".pth"}
                else None
            )
        )
        if artifact_error is not None:
            return CompletionEvidence(
                state="invalid",
                error=artifact_error,
                schema=schema,
                generation=generation_value,
            )
        validated_artifacts += 1
    # Older aggregate completion proofs name a JSON report without publishing
    # its digest. Keep them readable, but still validate the report semantically.
    for name in ("report", "comparison_report"):
        artifact_name = payload.get(name)
        if artifact_name is None or f"{name}_sha256" in payload:
            continue
        artifact = Path(str(artifact_name))
        if not artifact.is_absolute() or not artifact.is_file():
            return CompletionEvidence(
                state="invalid",
                error=f"completion artifact is missing or not absolute: {artifact}",
                schema=schema,
                generation=generation_value,
            )
        artifact_error = _validate_json_artifact(artifact)
        if artifact_error is not None:
            return CompletionEvidence(
                state="invalid",
                error=artifact_error,
                schema=schema,
                generation=generation_value,
            )
        validated_artifacts += 1
    return CompletionEvidence(
        state="complete",
        schema=schema,
        proof_sha256=hashlib.sha256(encoded).hexdigest(),
        generation=generation_value,
        validated_artifacts=validated_artifacts,
    )


def _resolve_completion_artifact(
    recorded_path: object,
    expected_sha256: str,
    *,
    training_root: Path | None = None,
) -> Path | None:
    """Resolve a checksummed completion artifact after training-root migration."""
    recorded = Path(str(recorded_path))
    root = (
        Path(quantdata_env.locations.training.path)
        if training_root is None
        else training_root
    )
    candidates = [recorded]
    for marker in ("flex-shared-strikes", "flex-fixed-weekly", "flex-phase3"):
        if marker in recorded.parts:
            marker_index = recorded.parts.index(marker)
            candidates.append(root.joinpath(*recorded.parts[marker_index:]))
            break
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and _sha256(candidate) == expected_sha256:
            return candidate
    return None


def _latest_failure(path: Path) -> Path | None:
    failure_directory = path.parent / "failures"
    if not failure_directory.is_dir():
        return None
    candidates = tuple(failure_directory.glob("*.failure.json"))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _failure_details(path: Path | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    exception = payload.get("exception") if isinstance(payload, Mapping) else None
    if not isinstance(exception, Mapping):
        return None, None
    exception_type = exception.get("type")
    message = exception.get("message")
    return (
        exception_type if isinstance(exception_type, str) else None,
        message if isinstance(message, str) else None,
    )


def _process_command_line(main_pid: int) -> tuple[str, ...]:
    if main_pid <= 0:
        return ()
    try:
        raw = Path(f"/proc/{main_pid}/cmdline").read_bytes()
    except OSError:
        return ()
    return tuple(
        field.decode("utf-8", errors="replace")
        for field in raw.split(b"\0")
        if field
    )


def _checkpoint_progress(
    command_line: tuple[str, ...],
    *,
    previous_value: object | None = None,
) -> ProgressEvidence:
    """Read the newest durable checkpoint's stage-specific progress fields."""
    directory_value = _option_value(command_line, "--checkpoint-directory")
    candidates: tuple[Path, ...] = ()
    if directory_value is not None:
        directory = Path(directory_value)
        if not directory.is_absolute():
            return ProgressEvidence(
                value=None,
                source="checkpoint",
                error=f"checkpoint directory is not absolute: {directory}",
            )
        if directory.is_dir():
            candidates = tuple(directory.glob("*.pt")) + tuple(
                directory.glob("*.pth")
            )
    checkpoint_value = _option_value(command_line, "--checkpoint")
    if checkpoint_value is not None:
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_absolute():
            return ProgressEvidence(
                value=None,
                source="checkpoint",
                error=f"checkpoint path is not absolute: {checkpoint}",
            )
        if checkpoint.is_file():
            candidates += (checkpoint,)
    if not candidates:
        return ProgressEvidence(value=None, source=None)
    try:
        checkpoint = max(candidates, key=lambda value: value.stat().st_mtime_ns)
        stat = checkpoint.stat()
    except OSError as error:
        return ProgressEvidence(
            value=None,
            source="checkpoint",
            error=f"cannot inspect checkpoint progress: {error}",
        )
    value: dict[str, object] = {
        "path": str(checkpoint),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }
    if isinstance(previous_value, Mapping) and all(
        previous_value.get(name) == value[name]
        for name in ("path", "mtime_ns", "size_bytes")
    ):
        # A durable atomic checkpoint with the same identity cannot contain new
        # progress. Reuse parsed metadata and avoid repeatedly loading tensors.
        return ProgressEvidence(value=dict(previous_value), source="checkpoint")
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - corruption has varied errors
        return ProgressEvidence(
            value=value,
            source="checkpoint",
            error=(
                f"cannot read checkpoint progress {checkpoint}: "
                f"{type(error).__name__}: {error}"
            ),
        )
    if not isinstance(payload, Mapping):
        return ProgressEvidence(
            value=value,
            source="checkpoint",
            error=f"checkpoint progress payload is not a mapping: {checkpoint}",
        )
    for name in (
        "schema",
        "schema_version",
        "phase",
        "line_index",
        "completed_epochs",
    ):
        item = payload.get(name)
        if item is not None and isinstance(item, (str, int, float, bool)):
            value[name] = item
    history = payload.get("history")
    if isinstance(history, list) and history and isinstance(history[-1], Mapping):
        last = {
            str(name): item
            for name, item in history[-1].items()
            if isinstance(item, (str, int, float, bool))
            and (not isinstance(item, float) or math.isfinite(item))
        }
        value["last_history"] = last
    line_histories = payload.get("histories")
    if (
        isinstance(line_histories, list)
        and line_histories
        and isinstance(line_histories[-1], Mapping)
    ):
        last_line = line_histories[-1]
        value["last_line"] = {
            str(name): item
            for name, item in last_line.items()
            if isinstance(item, (str, int, float, bool))
            and (not isinstance(item, float) or math.isfinite(item))
        }
    return ProgressEvidence(value=value, source="checkpoint")


def _worker_heartbeat_evidence(
    stage: PipelineStageSpec,
) -> WorkerHeartbeatEvidence:
    """Validate an opt-in worker heartbeat used by future fenced launches."""
    path = stage.worker_heartbeat_path
    if path is None:
        return WorkerHeartbeatEvidence(None, None, None)
    if not path.exists():
        return WorkerHeartbeatEvidence(
            None,
            None,
            None,
            f"worker heartbeat is missing: {path}",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(str(payload["timestamp_utc"]))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        return WorkerHeartbeatEvidence(None, None, None, str(error))
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("schema"), str)
        or timestamp.tzinfo is None
    ):
        return WorkerHeartbeatEvidence(
            None,
            None,
            None,
            "worker heartbeat has invalid schema or timestamp",
        )
    generation = payload.get("generation")
    if not isinstance(generation, str):
        return WorkerHeartbeatEvidence(
            timestamp.isoformat(),
            None,
            None,
            "worker heartbeat has no generation token",
        )
    if (
        stage.expected_generation is not None
        and generation != stage.expected_generation
    ):
        return WorkerHeartbeatEvidence(
            timestamp.isoformat(),
            generation,
            payload.get("last_progress_value"),
            (
                f"worker heartbeat generation {generation!r} does not match "
                f"expected {stage.expected_generation!r}"
            ),
        )
    return WorkerHeartbeatEvidence(
        timestamp.isoformat(),
        generation,
        payload.get("last_progress_value"),
    )


def _timestamp_is_recent(value: str | None, *, stale_seconds: int) -> bool:
    if value is None:
        return False
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return False
    return timestamp.tzinfo is not None and (
        datetime.now(UTC) - timestamp.astimezone(UTC)
    ).total_seconds() <= stale_seconds


def _option_value(command_line: tuple[str, ...], option: str) -> str | None:
    prefix = f"{option}="
    for index, argument in enumerate(command_line):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option and index + 1 < len(command_line):
            return command_line[index + 1]
    return None


def process_constraint_violations(command_line: tuple[str, ...]) -> tuple[str, ...]:
    """Check the GPU-0 and four-market-worker operational constraints."""
    violations: list[str] = []
    device = _option_value(command_line, "--device")
    if device is not None and device != "cuda:0":
        violations.append(f"configured device is {device!r}, expected 'cuda:0'")
    workers = _option_value(command_line, "--workers")
    if workers is not None:
        try:
            worker_count = int(workers)
        except ValueError:
            violations.append(f"invalid --workers value {workers!r}")
        else:
            if worker_count > 4:
                violations.append(
                    f"configured market workers are {worker_count}, maximum is 4"
                )
    return tuple(violations)


def _systemd_restart_count(properties: Mapping[str, str]) -> int:
    raw_value = properties.get("NRestarts", "0")
    try:
        value = int(raw_value)
    except ValueError:
        return 0
    return max(value, 0)


def _durable_stage_state(state: str) -> str:
    if state == "complete":
        return "SUCCEEDED"
    if state in {"waiting_dependency", "blocked_failed_dependency"}:
        return "BLOCKED_DEPENDENCY"
    if state in {"terminal_failure", "invalid_completion"}:
        return "FAILED_TERMINAL"
    return "RUNNING"


def classify_pipeline(
    stages: tuple[PipelineStageSpec, ...],
    observations: Mapping[str, StageObservation],
    *,
    restart_counts: Mapping[str, int],
    max_restarts: int,
    systemd_restart_counts: Mapping[str, int] | None = None,
    stale_observation_counts: Mapping[str, int] | None = None,
) -> PipelineClassification:
    """Classify one DAG poll without performing any service mutation."""
    ordered = _topological_stages(stages)
    stage_by_name = {stage.name: stage for stage in stages}
    state_by_name: dict[str, str] = {}
    payloads: list[dict[str, Any]] = []
    restart_names: list[str] = []
    stop_names: list[str] = []
    terminal_names: list[str] = []
    systemd_restart_counts = (
        {} if systemd_restart_counts is None else systemd_restart_counts
    )
    stale_observation_counts = (
        {} if stale_observation_counts is None else stale_observation_counts
    )

    for stage in ordered:
        observation = observations[stage.name]
        monitor_restart_count = restart_counts.get(stage.name, 0)
        systemd_restart_count = systemd_restart_counts.get(stage.name, 0)
        effective_restart_count = monitor_restart_count + systemd_restart_count
        prior_stale_observations = stale_observation_counts.get(stage.name, 0)
        next_stale_observations = prior_stale_observations
        reason: str | None = None
        failed_dependencies: list[str] = []
        waiting_dependencies: list[str] = []
        for dependency in stage.dependencies:
            proof_path = (
                stage_by_name[dependency.upstream].completion_path
                if dependency.proof_path is None
                else dependency.proof_path
            )
            proof = (
                observations[dependency.upstream].completion
                if proof_path == stage_by_name[dependency.upstream].completion_path
                else _completion_evidence(proof_path)
            )
            if proof.state == "complete":
                continue
            upstream_state = state_by_name[dependency.upstream]
            if proof.state == "invalid" or upstream_state in {
                "terminal_failure",
                "blocked_failed_dependency",
                "invalid_completion",
            }:
                failed_dependencies.append(dependency.upstream)
            else:
                waiting_dependencies.append(dependency.upstream)

        if observation.completion.state == "complete":
            state = "complete"
            next_stale_observations = 0
        elif observation.completion.state == "invalid":
            state = "invalid_completion"
            reason = observation.completion.error
            terminal_names.append(stage.name)
        elif failed_dependencies:
            state = "blocked_failed_dependency"
            reason = "terminal upstream failure: " + ", ".join(failed_dependencies)
            terminal_names.append(stage.name)
            if observation.active:
                stop_names.append(stage.name)
        elif waiting_dependencies:
            # Dependencies are represented in the durable DAG; an active
            # downstream wait loop is wasted work and obscures blocked state.
            state = "waiting_dependency"
            reason = "awaiting completion proof from: " + ", ".join(
                waiting_dependencies
            )
            if observation.active:
                stop_names.append(stage.name)
        else:
            violations = process_constraint_violations(observation.command_line)
            if violations:
                state = "terminal_failure"
                reason = "; ".join(violations)
                terminal_names.append(stage.name)
            elif observation.progress_error is not None:
                state = "terminal_failure"
                reason = observation.progress_error
                terminal_names.append(stage.name)
            elif observation.worker_heartbeat_error is not None:
                state = "terminal_failure"
                reason = observation.worker_heartbeat_error
                terminal_names.append(stage.name)
            elif (
                not observation.active
                and observation.failure_type in _DETERMINISTIC_EXCEPTION_TYPES
            ):
                state = "terminal_failure"
                reason = (
                    f"deterministic {observation.failure_type}: "
                    f"{observation.failure_message or 'no message'}"
                )
                terminal_names.append(stage.name)
            elif observation.active and (
                observation.progress_advanced
                or observation.progress_recent
                or (
                    stage.worker_heartbeat_path is not None
                    and observation.worker_heartbeat_status_error is None
                )
                or (
                    stage.worker_heartbeat_path is None
                    and (
                        observation.recent_journal or observation.cpu_advanced
                    )
                )
            ):
                state = "running"
                next_stale_observations = 0
            else:
                reason = (
                    "no journal or CPU progress"
                    if observation.active
                    else "service is not active"
                )
                if observation.active and prior_stale_observations < 1:
                    # A single quiet observation is not enough to interrupt a
                    # legitimate long solver step. Persist it and require the
                    # next consecutive observation to confirm staleness.
                    state = "slow"
                    reason = "first consecutive observation without progress"
                    next_stale_observations = prior_stale_observations + 1
                elif effective_restart_count < max_restarts:
                    state = "restart_pending"
                    restart_names.append(stage.name)
                    next_stale_observations = prior_stale_observations + 1
                else:
                    state = "terminal_failure"
                    terminal_names.append(stage.name)
                    next_stale_observations = prior_stale_observations + 1

        state_by_name[stage.name] = state
        payloads.append(
            {
                "name": stage.name,
                "unit": stage.unit,
                "state": state,
                "durable_state": _durable_stage_state(state),
                "reason": reason,
                "completion_path": str(stage.completion_path),
                "completion_state": observation.completion.state,
                "completion_schema": observation.completion.schema,
                "completion_proof_sha256": observation.completion.proof_sha256,
                "completion_generation": observation.completion.generation,
                "completion_validated_artifacts": (
                    observation.completion.validated_artifacts
                ),
                "dependencies": [
                    {
                        "upstream": dependency.upstream,
                        "proof_path": str(
                            stage_by_name[dependency.upstream].completion_path
                            if dependency.proof_path is None
                            else dependency.proof_path
                        ),
                    }
                    for dependency in stage.dependencies
                ],
                "recent_journal": observation.recent_journal,
                "cpu_usage_ns": observation.cpu_usage_ns,
                "cpu_advanced_since_poll": observation.cpu_advanced,
                "last_progress_value": observation.progress_value,
                "last_progress_at": observation.last_progress_at,
                "progress_advanced_since_poll": observation.progress_advanced,
                "progress_recent": observation.progress_recent,
                "progress_source": observation.progress_source,
                "progress_error": observation.progress_error,
                "consecutive_stale_observations": next_stale_observations,
                "worker_heartbeat_path": (
                    None
                    if stage.worker_heartbeat_path is None
                    else str(stage.worker_heartbeat_path)
                ),
                "worker_heartbeat_at": observation.worker_heartbeat_at,
                "worker_generation": observation.worker_generation,
                "worker_heartbeat_error": observation.worker_heartbeat_status_error,
                "expected_generation": stage.expected_generation,
                "generation_fencing": (
                    "enforced"
                    if stage.expected_generation is not None
                    else "unavailable_for_legacy_worker"
                ),
                "worker_identity": observation.properties.get("InvocationID"),
                "worker_identity_assurance": (
                    "worker_heartbeat_and_generation"
                    if stage.expected_generation is not None
                    else "legacy_systemd_invocation_only"
                ),
                "monitor_restart_count": monitor_restart_count,
                "systemd_restart_count": _systemd_restart_count(
                    observation.properties
                ),
                "observed_systemd_restart_count": systemd_restart_count,
                "effective_restart_count": effective_restart_count,
                "newest_failure_path": (
                    None
                    if observation.newest_failure_path is None
                    else str(observation.newest_failure_path)
                ),
                "failure_type": observation.failure_type,
                "failure_message": observation.failure_message,
                "systemd": dict(observation.properties),
                "command_line": list(observation.command_line),
            }
        )

    return PipelineClassification(
        stages=tuple(payloads),
        restart_names=tuple(restart_names),
        stop_names=tuple(stop_names),
        terminal_names=tuple(terminal_names),
        all_complete=all(payload["state"] == "complete" for payload in payloads),
    )


def _load_monitor_state(
    output: Path,
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, str],
    dict[str, str],
    dict[str, int],
    dict[str, object],
    dict[str, str],
    dict[str, int],
]:
    if not output.exists():
        return {}, {}, {}, {}, {}, {}, {}, {}
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}, {}, {}, {}, {}, {}, {}
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "flex-pipeline-monitor-v1"
    ):
        return {}, {}, {}, {}, {}, {}, {}, {}
    restart_counts: dict[str, int] = {}
    cpu_usage: dict[str, int] = {}
    prior_states: dict[str, str] = {}
    invocation_ids: dict[str, str] = {}
    systemd_restart_counts: dict[str, int] = {}
    progress_values: dict[str, object] = {}
    progress_at: dict[str, str] = {}
    stale_observation_counts: dict[str, int] = {}
    for stage in payload.get("stages", []):
        if not isinstance(stage, Mapping) or not isinstance(stage.get("name"), str):
            continue
        name = stage["name"]
        restart_counts[name] = int(stage.get("monitor_restart_count", 0))
        systemd_restart_counts[name] = int(
            stage.get("observed_systemd_restart_count", 0)
        )
        if isinstance(stage.get("state"), str):
            prior_states[name] = stage["state"]
        systemd = stage.get("systemd")
        if isinstance(systemd, Mapping) and isinstance(
            systemd.get("InvocationID"), str
        ):
            invocation_ids[name] = systemd["InvocationID"]
        value = stage.get("cpu_usage_ns")
        if isinstance(value, int):
            cpu_usage[name] = value
        if stage.get("last_progress_value") is not None:
            progress_values[name] = stage["last_progress_value"]
        if isinstance(stage.get("last_progress_at"), str):
            progress_at[name] = stage["last_progress_at"]
        stale_observation_counts[name] = max(
            int(stage.get("consecutive_stale_observations", 0)),
            0,
        )
    return (
        restart_counts,
        cpu_usage,
        prior_states,
        invocation_ids,
        systemd_restart_counts,
        progress_values,
        progress_at,
        stale_observation_counts,
    )


def _pipeline_payload(
    *,
    status: str,
    warnings: list[str],
    resource_error: str | None,
    resource_stop_latched: bool,
    terminal_names: list[str],
    resources: Mapping[str, Any],
    stage_payloads: list[dict[str, Any]],
    manifest_path: Path | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    terminal_stages = sorted(set(terminal_names))
    serious_error = resource_error
    if serious_error is None and resource_stop_latched:
        serious_error = (
            "resource stop remains latched pending explicit operator assessment"
        )
    if serious_error is None and terminal_stages:
        serious_error = "terminal pipeline stage(s): " + ", ".join(terminal_stages)
    durable_status = {
        "complete": "SUCCEEDED",
        "terminal_failure": "FAILED_TERMINAL",
        "resource_stop": "BLOCKED_DEPENDENCY",
    }.get(status, "RUNNING")
    return {
        "schema": "flex-pipeline-monitor-v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "durable_status": durable_status,
        "warnings": warnings,
        "serious_error": serious_error,
        "resource_stop_latched": resource_stop_latched,
        "terminal_stages": terminal_stages,
        "resources": dict(resources),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "stages": stage_payloads,
    }


def _resource_status() -> tuple[dict[str, Any], list[str], str | None]:
    return _quantdata_resource_status()


def monitor_pipeline(
    stages: tuple[PipelineStageSpec, ...],
    *,
    output: Path,
    poll_seconds: int,
    stale_seconds: int,
    max_restarts: int,
    acknowledge_resource_stop: bool = False,
) -> None:
    """Persistently supervise a dependency graph until every stage completes."""
    verify_inp(len(stages) > 0, lambda: "At least one stage is required")
    verify_inp(10 <= poll_seconds <= 600, lambda: "Invalid monitor poll interval")
    verify_inp(
        stale_seconds >= 2 * poll_seconds,
        lambda: "Stale interval must span at least two monitor polls",
    )
    verify_inp(max_restarts >= 0, lambda: "'max_restarts' cannot be negative")
    stages = _topological_stages(stages)
    verify_inp(
        isinstance(acknowledge_resource_stop, bool),
        lambda: "'acknowledge_resource_stop' must be bool",
    )
    (
        restart_counts,
        previous_cpu_usage,
        previous_states,
        previous_invocations,
        observed_systemd_restarts,
        previous_progress_values,
        last_progress_at,
        stale_observation_counts,
    ) = _load_monitor_state(output)
    manifest_path = _manifest_path(output)
    _atomic_write_json(
        _pipeline_manifest_payload(
            stages,
            poll_seconds=poll_seconds,
            stale_seconds=stale_seconds,
            max_restarts=max_restarts,
        ),
        manifest_path,
    )
    manifest_sha256 = _sha256(manifest_path)
    resource_stop_latched = False
    if output.exists() and not acknowledge_resource_stop:
        try:
            prior_payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_payload = {}
        resource_stop_latched = (
            isinstance(prior_payload, Mapping)
            and prior_payload.get("status") == "resource_stop"
        )
    for stage in stages:
        restart_counts.setdefault(stage.name, 0)
        observed_systemd_restarts.setdefault(stage.name, 0)
        stale_observation_counts.setdefault(stage.name, 0)

    completion_cache: dict[
        tuple[Path, str | None],
        tuple[tuple[int, int] | None, CompletionEvidence],
    ] = {}

    def completion_evidence(stage: PipelineStageSpec) -> CompletionEvidence:
        key = (stage.completion_path, stage.expected_generation)
        try:
            stat = stage.completion_path.stat()
            identity: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            identity = None
        cached = completion_cache.get(key)
        if cached is not None and cached[0] == identity:
            return cached[1]
        evidence = _completion_evidence(
            stage.completion_path,
            expected_generation=stage.expected_generation,
        )
        completion_cache[key] = (identity, evidence)
        return evidence

    while True:
        resources, warnings, resource_error = _resource_status()
        observations: dict[str, StageObservation] = {}
        for stage in stages:
            properties = _service_properties(stage.unit)
            observed_systemd_restarts[stage.name] = max(
                observed_systemd_restarts[stage.name],
                _systemd_restart_count(properties),
            )
            active = properties.get("ActiveState") == "active"
            cpu_usage = _cpu_usage_ns(properties)
            prior_cpu = previous_cpu_usage.get(stage.name)
            cpu_advanced = (
                active
                and cpu_usage is not None
                and prior_cpu is not None
                and cpu_usage > prior_cpu
            )
            if cpu_usage is not None:
                previous_cpu_usage[stage.name] = cpu_usage
            try:
                main_pid = int(properties.get("MainPID", "0") or 0)
            except ValueError:
                main_pid = 0
            command_line = _process_command_line(main_pid)
            checkpoint_progress = _checkpoint_progress(
                command_line,
                previous_value=previous_progress_values.get(stage.name),
            )
            worker_heartbeat = _worker_heartbeat_evidence(stage)
            progress_value = (
                worker_heartbeat.progress_value
                if worker_heartbeat.progress_value is not None
                else checkpoint_progress.value
            )
            progress_source = (
                "worker_heartbeat"
                if worker_heartbeat.progress_value is not None
                else checkpoint_progress.source
            )
            progress_advanced = (
                progress_value is not None
                and progress_value != previous_progress_values.get(stage.name)
            )
            if progress_advanced:
                previous_progress_values[stage.name] = progress_value
                last_progress_at[stage.name] = datetime.now(UTC).isoformat()
            worker_error = worker_heartbeat.error
            if (
                stage.worker_heartbeat_path is not None
                and worker_error is None
                and not _timestamp_is_recent(
                    worker_heartbeat.timestamp_utc,
                    stale_seconds=stale_seconds,
                )
            ):
                worker_error = "worker heartbeat timestamp is stale"
            worker_heartbeat_fencing_error = (
                worker_error
                if worker_error is not None and "generation" in worker_error
                else None
            )
            newest_failure_path = _latest_failure(stage.completion_path)
            failure_type, failure_message = _failure_details(newest_failure_path)
            observations[stage.name] = StageObservation(
                completion=completion_evidence(stage),
                active=active,
                properties=properties,
                recent_journal=active
                and _has_recent_journal(stage.unit, stale_seconds),
                cpu_usage_ns=cpu_usage,
                cpu_advanced=cpu_advanced,
                command_line=command_line,
                newest_failure_path=newest_failure_path,
                failure_type=failure_type,
                failure_message=failure_message,
                progress_value=previous_progress_values.get(stage.name),
                last_progress_at=last_progress_at.get(stage.name),
                progress_advanced=progress_advanced,
                progress_recent=_timestamp_is_recent(
                    last_progress_at.get(stage.name),
                    stale_seconds=stale_seconds,
                ),
                progress_source=progress_source,
                progress_error=checkpoint_progress.error,
                worker_heartbeat_at=worker_heartbeat.timestamp_utc,
                worker_generation=worker_heartbeat.generation,
                worker_heartbeat_error=worker_heartbeat_fencing_error,
                worker_heartbeat_status_error=worker_error,
            )
            prior_state = previous_states.get(stage.name)
            prior_invocation = previous_invocations.get(stage.name)
            current_invocation = properties.get("InvocationID")
            if (
                prior_state in {"terminal_failure", "blocked_failed_dependency"}
                and active
                and isinstance(current_invocation, str)
                and current_invocation != ""
                and current_invocation != prior_invocation
            ):
                # A new active invocation after a terminal state is explicit
                # operator recovery and receives a fresh bounded retry budget.
                restart_counts[stage.name] = 0
                observed_systemd_restarts[stage.name] = _systemd_restart_count(
                    properties
                )

        classification = classify_pipeline(
            stages,
            observations,
            restart_counts=restart_counts,
            max_restarts=max_restarts,
            systemd_restart_counts=observed_systemd_restarts,
            stale_observation_counts=stale_observation_counts,
        )
        stage_payloads = [dict(payload) for payload in classification.stages]
        payload_by_name = {payload["name"]: payload for payload in stage_payloads}
        terminal_names = list(classification.terminal_names)
        if resource_error is not None:
            resource_stop_latched = True
        for name in classification.stop_names:
            stage = next(stage for stage in stages if stage.name == name)
            print(
                "FLEX pipeline monitor stopping dependency-blocked service "
                f"{stage.unit}",
                flush=True,
            )
            result = _run_systemctl("stop", stage.unit, check=False)
            if result.returncode == 0:
                payload_by_name[name]["action"] = "stopped_blocked_service"
            else:
                payload_by_name[name]["state"] = "terminal_failure"
                payload_by_name[name]["durable_state"] = "FAILED_TERMINAL"
                payload_by_name[name]["reason"] = (
                    f"systemctl stop failed: {result.stderr.strip()}"
                )
                terminal_names.append(name)
        for name in (() if resource_stop_latched else classification.restart_names):
            stage = next(stage for stage in stages if stage.name == name)
            print(f"FLEX pipeline monitor restarting {stage.unit}", flush=True)
            result = _run_systemctl("restart", stage.unit, check=False)
            restart_counts[name] += 1
            payload_by_name[name]["monitor_restart_count"] = restart_counts[name]
            payload_by_name[name]["effective_restart_count"] = (
                restart_counts[name] + observed_systemd_restarts[name]
            )
            if result.returncode == 0:
                payload_by_name[name]["state"] = "restarting"
                payload_by_name[name]["durable_state"] = "RUNNING"
            else:
                payload_by_name[name]["state"] = "terminal_failure"
                payload_by_name[name]["durable_state"] = "FAILED_TERMINAL"
                payload_by_name[name]["reason"] = (
                    f"systemctl restart failed: {result.stderr.strip()}"
                )
                terminal_names.append(name)

        if resource_stop_latched:
            status = "resource_stop"
        elif terminal_names:
            status = "terminal_failure"
        elif classification.all_complete:
            status = "complete"
        else:
            status = "running"
        payload = _pipeline_payload(
            status=status,
            warnings=warnings,
            resource_error=resource_error,
            resource_stop_latched=resource_stop_latched,
            terminal_names=terminal_names,
            resources=resources,
            stage_payloads=stage_payloads,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        _atomic_write_json(payload, output)
        state_changes = [
            {
                "stage": str(stage_payload["name"]),
                "from": previous_states.get(str(stage_payload["name"])),
                "to": str(stage_payload["state"]),
            }
            for stage_payload in stage_payloads
            if previous_states.get(str(stage_payload["name"]))
            != str(stage_payload["state"])
        ]
        progress_changes = [
            {
                "stage": str(stage_payload["name"]),
                "last_progress_value": stage_payload["last_progress_value"],
                "last_progress_at": stage_payload["last_progress_at"],
            }
            for stage_payload in stage_payloads
            if bool(stage_payload["progress_advanced_since_poll"])
        ]
        if (
            state_changes
            or progress_changes
            or classification.restart_names
            or classification.stop_names
            or warnings
            or payload["serious_error"] is not None
        ):
            _append_json_event(
                {
                    "schema": "flex-pipeline-event-v1",
                    "timestamp_utc": payload["timestamp_utc"],
                    "status": status,
                    "state_changes": state_changes,
                    "progress_changes": progress_changes,
                    "restart_names": list(classification.restart_names),
                    "stop_names": list(classification.stop_names),
                    "warnings": warnings,
                    "serious_error": payload["serious_error"],
                },
                _event_log_path(output),
            )
        _atomic_write_json(
            {
                "schema": "flex-pipeline-monitor-heartbeat-v1",
                "timestamp_utc": payload["timestamp_utc"],
                "monitor_pid": os.getpid(),
                "status": status,
                "active_stages": [
                    str(stage_payload["name"])
                    for stage_payload in stage_payloads
                    if stage_payload["state"] in {"running", "slow", "restarting"}
                ],
                "terminal_stages": payload["terminal_stages"],
                "state_path": str(output),
                "event_log_path": str(_event_log_path(output)),
            },
            _heartbeat_path(output),
        )
        for warning in warnings:
            print(f"FLEX pipeline monitor warning: {warning}", flush=True)
        if resource_stop_latched:
            message = resource_error or (
                "resource stop remains latched pending explicit operator assessment"
            )
            print(f"FLEX pipeline monitor serious error: {message}", flush=True)
            for stage in stages:
                if observations[stage.name].active:
                    _run_systemctl("stop", stage.unit, check=False)
            # Safe headroom alone is not permission to resume. The supervisor
            # remains visible and stopped until explicitly acknowledged.
        elif terminal_names:
            print(
                "FLEX pipeline terminal failure: "
                + ", ".join(sorted(set(terminal_names))),
                flush=True,
            )
        elif classification.all_complete:
            print("FLEX pipeline monitor: all stages completed", flush=True)
            return
        previous_states = {
            str(payload["name"]): str(payload["state"])
            for payload in stage_payloads
        }
        previous_invocations = {
            str(payload["name"]): str(payload["systemd"].get("InvocationID", ""))
            for payload in stage_payloads
        }
        stale_observation_counts = {
            str(payload["name"]): int(
                payload.get("consecutive_stale_observations", 0)
            )
            for payload in stage_payloads
        }
        time.sleep(poll_seconds)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", required=True, type=parse_stage_spec)
    parser.add_argument(
        "--dependency",
        action="append",
        default=[],
        type=parse_dependency_spec,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=1_800)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument(
        "--acknowledge-resource-stop",
        action="store_true",
        help="resume only after the prior resource-stop cause was assessed",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    arguments = _parse_arguments()
    stages = attach_dependencies(tuple(arguments.stage), tuple(arguments.dependency))
    monitor_pipeline(
        stages,
        output=arguments.output,
        poll_seconds=arguments.poll_seconds,
        stale_seconds=arguments.stale_seconds,
        max_restarts=arguments.max_restarts,
        acknowledge_resource_stop=arguments.acknowledge_resource_stop,
    )


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="monitor_flex_pipeline",
        clean_sigterm=True,
        enable_ram_guard=False,
    )
