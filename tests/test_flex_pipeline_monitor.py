"""Dependency and failure-propagation tests for the FLEX DAG monitor."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from quantdata.dysanos.flex.pipeline_monitor import (
    CompletionEvidence,
    DependencySpec,
    PipelineStageSpec,
    StageObservation,
    _append_json_event,
    _checkpoint_progress,
    _completion_evidence,
    _event_log_path,
    _failure_details,
    _heartbeat_path,
    _load_monitor_state,
    _manifest_path,
    _pipeline_manifest_payload,
    _pipeline_payload,
    _sha256,
    _systemd_restart_count,
    _worker_heartbeat_evidence,
    attach_dependencies,
    classify_pipeline,
    parse_dependency_spec,
    parse_stage_spec,
    process_constraint_violations,
)


def _write_completion(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"schema": "test-completion-v1", "status": "complete"}\n',
        encoding="utf-8",
    )


def test_monitor_artifact_paths_and_event_log_are_durable(tmp_path: Path) -> None:
    state = tmp_path / "pipeline.json"
    assert _event_log_path(state) == tmp_path / "pipeline.events.jsonl"
    assert _heartbeat_path(state) == tmp_path / "pipeline.heartbeat.json"
    event_path = _event_log_path(state)
    _append_json_event({"sequence": 1}, event_path)
    _append_json_event({"sequence": 2}, event_path)
    assert [json.loads(line) for line in event_path.read_text().splitlines()] == [
        {"sequence": 1},
        {"sequence": 2},
    ]


@pytest.mark.parametrize(
    "failure_type",
    ["TrainingConvergenceError", "AttributeError"],
)
def test_failure_details_and_deterministic_failure_are_not_retried(
    tmp_path: Path,
    failure_type: str,
) -> None:
    failure = tmp_path / "failure.json"
    failure.write_text(
        json.dumps(
            {
                "exception": {
                    "type": failure_type,
                    "message": "plateau gate failed",
                }
            }
        ),
        encoding="utf-8",
    )
    assert _failure_details(failure) == (
        failure_type,
        "plateau gate failed",
    )
    completion = tmp_path / "worker.json"
    stage = PipelineStageSpec(
        name="worker",
        unit="worker.service",
        completion_path=completion,
    )
    observation = _observation(completion, active=False)
    observation = StageObservation(
        **{
            **observation.__dict__,
            "newest_failure_path": failure,
            "failure_type": failure_type,
            "failure_message": "plateau gate failed",
        }
    )
    result = classify_pipeline(
        (stage,),
        {"worker": observation},
        restart_counts={"worker": 0},
        max_restarts=2,
    )
    assert result.stages[0]["state"] == "terminal_failure"
    assert result.restart_names == ()


@pytest.mark.parametrize("status", ["complete", "native_complete"])
def test_completion_evidence_accepts_configured_producer_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    path = tmp_path / "proof.json"
    path.write_text(
        json.dumps({"schema": "test-completion-v1", "status": status}),
        encoding="utf-8",
    )
    evidence = _completion_evidence(path)
    assert evidence.state == "complete"
    assert evidence.schema == "test-completion-v1"
    assert evidence.proof_sha256 is not None


def test_completion_evidence_rejects_arbitrary_complete_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"status": "almost_complete"}), encoding="utf-8")
    evidence = _completion_evidence(path)
    assert evidence.state == "invalid"
    assert "almost_complete" in str(evidence.error)


def test_completion_evidence_validates_artifact_provenance_and_dimensions(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "test-report-v1",
                "status": "complete",
                "num_days": 2,
                "market_dates": ["2025-01-02", "2025-01-03"],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "schema": "test-checkpoint-v1",
            "phase": "complete",
            "completed_epochs": 1,
            "history": [{"epoch": 1.0, "objective": 0.5}],
            "market_fingerprints": ["a", "b"],
            "model_config": {"state_size": 3},
            "model_state_dict": {
                "daily_state_embedding.weight": torch.zeros((2, 3)),
            },
        },
        checkpoint,
    )
    proof = tmp_path / "complete.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "test-completion-v1",
                "status": "complete",
                "report": str(report),
                "report_sha256": _sha256(report),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    evidence = _completion_evidence(proof)
    assert evidence.state == "complete"
    assert evidence.validated_artifacts == 2

    report.write_text('{"schema":"changed"}', encoding="utf-8")
    invalid = _completion_evidence(proof)
    assert invalid.state == "invalid"
    assert "digest mismatch" in str(invalid.error)


def test_completion_evidence_accepts_completed_sequential_q_checkpoint(
    tmp_path: Path,
) -> None:
    report = tmp_path / "q-report.json"
    report.write_text(
        json.dumps({"schema": "test-q-report-v1", "status": "complete"}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "q.pt"
    torch.save(
        {
            "schema": "flex-weekly-measure-kernel-v1",
            "phase": "complete",
            "line_index": 1,
            "completed_epochs": 0,
            "histories": [{"line_index": 0, "status": "complete"}],
            "model_state_dict": {"weight": torch.zeros(2)},
        },
        checkpoint,
    )
    proof = tmp_path / "q-complete.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "test-q-completion-v1",
                "status": "complete",
                "report": str(report),
                "report_sha256": _sha256(report),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    evidence = _completion_evidence(proof)
    assert evidence.state == "complete"
    assert evidence.validated_artifacts == 2


def test_completion_evidence_validates_nested_q_sequence_proofs(
    tmp_path: Path,
) -> None:
    def q_proof(name: str, line_index: int) -> Path:
        report = tmp_path / f"{name}-report.json"
        report.write_text(
            json.dumps({"schema": f"test-{name}-report-v1", "status": "complete"}),
            encoding="utf-8",
        )
        checkpoint = tmp_path / f"{name}.pt"
        torch.save(
            {
                "schema": "flex-weekly-measure-kernel-v1",
                "phase": "complete",
                "line_index": line_index,
                "completed_epochs": 0,
                "histories": [
                    {"line_index": index, "status": "complete"}
                    for index in range(line_index)
                ],
                "model_state_dict": {"weight": torch.zeros(2)},
            },
            checkpoint,
        )
        proof = tmp_path / f"{name}-complete.json"
        proof.write_text(
            json.dumps(
                {
                    "schema": f"test-{name}-completion-v1",
                    "status": "complete",
                    "report": str(report),
                    "report_sha256": _sha256(report),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            ),
            encoding="utf-8",
        )
        return proof

    q1 = q_proof("q1", 1)
    q2 = q_proof("q2", 2)
    sequence_report = tmp_path / "sequence.json"
    sequence_report.write_text(
        json.dumps(
            {
                "schema": "flex-joint-hierarchical-normal-q-sequence-v1",
                "status": "complete",
                "source_q1_completion": str(q1),
                "source_q1_completion_sha256": _sha256(q1),
                "maximum_roll_lines": 13,
                "completed_roll_lines": 2,
                "stages": [
                    {
                        "roll_line": 2,
                        "status": "complete",
                        "completion": str(q2),
                        "completion_sha256": _sha256(q2),
                    }
                ],
                "stopping_rule": {},
                "termination_reason": "test",
            }
        ),
        encoding="utf-8",
    )
    sequence_proof = tmp_path / "sequence-complete.json"
    sequence_proof.write_text(
        json.dumps(
            {
                "schema": "test-sequence-completion-v1",
                "status": "complete",
                "report": str(sequence_report),
                "report_sha256": _sha256(sequence_report),
            }
        ),
        encoding="utf-8",
    )
    assert _completion_evidence(sequence_proof).state == "complete"
    q2.write_text('{}', encoding="utf-8")
    invalid = _completion_evidence(sequence_proof)
    assert invalid.state == "invalid"
    assert "proof digest mismatch" in str(invalid.error)


def test_completion_evidence_accepts_provenance_bound_continuation_base(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.pt"
    base_history = [{"epoch": 1.0, "objective": 0.5, "total": 0.5}]
    torch.save(
        {
            "schema": "flex-quote-optimal-hierarchical-states-v1",
            "phase": "training",
            "completed_epochs": 1,
            "history": base_history,
            "market_fingerprints": ["a", "b"],
            "model_config": {"state_size": 3},
            "daily_states": torch.zeros((2, 3)),
        },
        base,
    )
    base_digest = _sha256(base)
    continuation = tmp_path / "continuation.pt"
    torch.save(
        {
            "schema": (
                "flex-quote-optimal-hierarchical-states-continuation-v1"
            ),
            "phase": "complete",
            "completed_epochs": 1,
            "history": [{"epoch": 1.0, "objective": 0.4, "total": 0.4}],
            "market_fingerprints": ["a", "b"],
            "model_config": {"state_size": 3},
            "daily_states": torch.zeros((2, 3)),
            "continuation_provenance": {
                "source_state_checkpoint": str(base),
                "source_state_checkpoint_sha256": base_digest,
                "source_completed_epochs": 1,
                "source_final_metrics": base_history[-1],
                "source_convergence": {
                    "converged": False,
                    "reason": "trailing_window_not_converged",
                    "best_epoch": 1,
                },
                "optimizer_scheduler_reset": True,
            },
        },
        continuation,
    )
    proof = tmp_path / "complete.json"
    proof.write_text(
        json.dumps(
            {
                "schema": (
                    "flex-shared-strikes-joint-hierarchical-completion-v1"
                ),
                "status": "complete",
                "validation_state_checkpoint": str(continuation),
                "validation_state_checkpoint_sha256": _sha256(continuation),
                "validation_state_base_checkpoint": str(base),
                "validation_state_base_checkpoint_sha256": base_digest,
            }
        ),
        encoding="utf-8",
    )

    evidence = _completion_evidence(proof)
    assert evidence.state == "complete"
    assert evidence.validated_artifacts == 2

    payload = torch.load(continuation, map_location="cpu", weights_only=False)
    payload["continuation_provenance"]["source_state_checkpoint_sha256"] = "0" * 64
    torch.save(payload, continuation)
    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    proof_payload["validation_state_checkpoint_sha256"] = _sha256(continuation)
    proof.write_text(json.dumps(proof_payload), encoding="utf-8")
    invalid = _completion_evidence(proof)
    assert invalid.state == "invalid"
    assert "does not bind" in str(invalid.error)


def test_completion_generation_fences_obsolete_worker(tmp_path: Path) -> None:
    proof = tmp_path / "complete.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "test-completion-v1",
                "status": "complete",
                "generation": "old",
            }
        ),
        encoding="utf-8",
    )
    evidence = _completion_evidence(proof, expected_generation="current")
    assert evidence.state == "invalid"
    assert "expected 'current'" in str(evidence.error)


def test_future_worker_heartbeat_enforces_generation(tmp_path: Path) -> None:
    heartbeat = tmp_path / "worker-heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "schema": "test-worker-heartbeat-v1",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "generation": "generation-2",
                "last_progress_value": {"epoch": 7},
            }
        ),
        encoding="utf-8",
    )
    current = PipelineStageSpec(
        name="worker",
        unit="worker.service",
        completion_path=tmp_path / "complete.json",
        worker_heartbeat_path=heartbeat,
        expected_generation="generation-2",
    )
    evidence = _worker_heartbeat_evidence(current)
    assert evidence.error is None
    assert evidence.progress_value == {"epoch": 7}
    obsolete = PipelineStageSpec(
        name="worker",
        unit="worker.service",
        completion_path=tmp_path / "complete.json",
        worker_heartbeat_path=heartbeat,
        expected_generation="generation-3",
    )
    assert "does not match" in str(_worker_heartbeat_evidence(obsolete).error)


def test_checkpoint_progress_extracts_epoch_and_reuses_unchanged_metadata(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    checkpoint = directory / "model.pt"
    torch.save(
        {
            "schema": "progress-v1",
            "phase": "training",
            "completed_epochs": 5,
            "history": [{"epoch": 5.0, "objective": 1.25}],
        },
        checkpoint,
    )
    command = ("python", "worker", "--checkpoint-directory", str(directory))
    first = _checkpoint_progress(command)
    assert first.error is None
    assert first.source == "checkpoint"
    assert isinstance(first.value, dict)
    assert first.value["completed_epochs"] == 5
    assert first.value["last_history"] == {"epoch": 5.0, "objective": 1.25}
    second = _checkpoint_progress(command, previous_value=first.value)
    assert second == first

    torch.save(
        {
            "schema": "progress-v1",
            "phase": "training",
            "completed_epochs": 6,
            "history": [{"epoch": 6.0, "objective": 1.0}],
        },
        checkpoint,
    )
    stat = checkpoint.stat()
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    third = _checkpoint_progress(command, previous_value=first.value)
    assert isinstance(third.value, dict)
    assert third.value["completed_epochs"] == 6


def _observation(
    completion_path: Path,
    *,
    active: bool,
    recent_journal: bool = False,
    cpu_advanced: bool = False,
    command_line: tuple[str, ...] = (),
) -> StageObservation:
    completion = (
        CompletionEvidence(state="complete")
        if completion_path.exists()
        else CompletionEvidence(state="missing")
    )
    return StageObservation(
        completion=completion,
        active=active,
        properties={
            "ActiveState": "active" if active else "failed",
            "SubState": "running" if active else "failed",
            "MainPID": "123" if active else "0",
        },
        recent_journal=recent_journal,
        cpu_usage_ns=10 if active else 0,
        cpu_advanced=cpu_advanced,
        command_line=command_line,
        newest_failure_path=None,
    )


def _pipeline(tmp_path: Path) -> tuple[PipelineStageSpec, ...]:
    native = PipelineStageSpec(
        name="native",
        unit="native.service",
        completion_path=tmp_path / "native.json",
    )
    full = PipelineStageSpec(
        name="full",
        unit="full.service",
        completion_path=tmp_path / "full.json",
    )
    capacity = PipelineStageSpec(
        name="capacity",
        unit="capacity.service",
        completion_path=tmp_path / "capacity.json",
        dependencies=(DependencySpec("native"), DependencySpec("full")),
    )
    compressed = PipelineStageSpec(
        name="compressed",
        unit="compressed.service",
        completion_path=tmp_path / "compressed.json",
        dependencies=(DependencySpec("native"), DependencySpec("capacity")),
    )
    weekly_p = PipelineStageSpec(
        name="weekly-p",
        unit="weekly-p.service",
        completion_path=tmp_path / "weekly-p.json",
        dependencies=(DependencySpec("full"), DependencySpec("compressed")),
    )
    weekly_q = PipelineStageSpec(
        name="weekly-q",
        unit="weekly-q.service",
        completion_path=tmp_path / "weekly-q.json",
        dependencies=(
            DependencySpec("compressed"),
            DependencySpec("weekly-p", tmp_path / "weekly-p-primary.json"),
        ),
    )
    return native, full, capacity, compressed, weekly_p, weekly_q


def test_stage_and_dependency_cli_parsing(tmp_path: Path) -> None:
    stage = parse_stage_spec(f"capacity::capacity.service::{tmp_path / 'done.json'}")
    dependent, dependency = parse_dependency_spec(
        f"weekly-q::weekly-p::{tmp_path / 'primary.json'}"
    )
    attached = attach_dependencies((stage,), ())
    assert attached == (stage,)
    assert dependent == "weekly-q"
    assert dependency == DependencySpec(
        upstream="weekly-p", proof_path=tmp_path / "primary.json"
    )
    fenced = parse_stage_spec(
        "::".join(
            (
                "future",
                "future.service",
                str(tmp_path / "future-complete.json"),
                str(tmp_path / "future-heartbeat.json"),
                "generation-7",
            )
        )
    )
    assert fenced.worker_heartbeat_path == tmp_path / "future-heartbeat.json"
    assert fenced.expected_generation == "generation-7"


@pytest.mark.parametrize(
    "value",
    ["missing-fields", "name::bad-unit::/tmp/done", "name::unit.service::relative"],
)
def test_stage_cli_rejects_invalid_spec(value: str) -> None:
    with pytest.raises(ValueError):
        parse_stage_spec(value)


def test_wait_loop_heartbeat_is_not_pipeline_progress(tmp_path: Path) -> None:
    stages = _pipeline(tmp_path)
    _write_completion(tmp_path / "native.json")
    _write_completion(tmp_path / "full.json")
    observations = {
        "native": _observation(tmp_path / "native.json", active=False),
        "full": _observation(tmp_path / "full.json", active=False),
        "capacity": _observation(
            tmp_path / "capacity.json", active=True, cpu_advanced=True
        ),
        # These journal lines say only "waiting for completion proof".
        "compressed": _observation(
            tmp_path / "compressed.json", active=True, recent_journal=True
        ),
        "weekly-p": _observation(
            tmp_path / "weekly-p.json", active=True, recent_journal=True
        ),
        "weekly-q": _observation(
            tmp_path / "weekly-q.json", active=True, recent_journal=True
        ),
    }
    result = classify_pipeline(
        stages,
        observations,
        restart_counts={stage.name: 0 for stage in stages},
        max_restarts=2,
    )
    states = {stage["name"]: stage["state"] for stage in result.stages}
    assert states == {
        "full": "complete",
        "native": "complete",
        "capacity": "running",
        "compressed": "waiting_dependency",
        "weekly-p": "waiting_dependency",
        "weekly-q": "waiting_dependency",
    }
    assert result.restart_names == ()
    assert set(result.stop_names) == {"compressed", "weekly-p", "weekly-q"}
    assert result.terminal_names == ()


def test_terminal_failure_propagates_through_every_downstream_gate(
    tmp_path: Path,
) -> None:
    stages = _pipeline(tmp_path)
    _write_completion(tmp_path / "native.json")
    _write_completion(tmp_path / "full.json")
    observations = {
        "native": _observation(tmp_path / "native.json", active=False),
        "full": _observation(tmp_path / "full.json", active=False),
        "capacity": _observation(tmp_path / "capacity.json", active=False),
        "compressed": _observation(
            tmp_path / "compressed.json", active=True, recent_journal=True
        ),
        "weekly-p": _observation(
            tmp_path / "weekly-p.json", active=True, recent_journal=True
        ),
        "weekly-q": _observation(
            tmp_path / "weekly-q.json", active=True, recent_journal=True
        ),
    }
    restart_counts = {stage.name: 0 for stage in stages}
    restart_counts["capacity"] = 2
    result = classify_pipeline(
        stages,
        observations,
        restart_counts=restart_counts,
        max_restarts=2,
    )
    states = {stage["name"]: stage["state"] for stage in result.stages}
    assert states["capacity"] == "terminal_failure"
    assert states["compressed"] == "blocked_failed_dependency"
    assert states["weekly-p"] == "blocked_failed_dependency"
    assert states["weekly-q"] == "blocked_failed_dependency"
    assert set(result.terminal_names) == {
        "capacity",
        "compressed",
        "weekly-p",
        "weekly-q",
    }
    assert set(result.stop_names) == {"compressed", "weekly-p", "weekly-q"}


def test_systemd_restarts_count_toward_the_same_bounded_budget(
    tmp_path: Path,
) -> None:
    stage = PipelineStageSpec(
        name="worker",
        unit="worker.service",
        completion_path=tmp_path / "worker.json",
    )
    observation = _observation(tmp_path / "worker.json", active=False)
    result = classify_pipeline(
        (stage,),
        {"worker": observation},
        restart_counts={"worker": 1},
        systemd_restart_counts={"worker": 1},
        max_restarts=2,
    )
    assert result.stages[0]["state"] == "terminal_failure"
    assert result.stages[0]["effective_restart_count"] == 2
    assert result.restart_names == ()
    assert result.stop_names == ()


def test_active_stage_requires_two_consecutive_stale_observations(
    tmp_path: Path,
) -> None:
    stage = PipelineStageSpec(
        name="worker",
        unit="worker.service",
        completion_path=tmp_path / "worker.json",
    )
    observation = _observation(tmp_path / "worker.json", active=True)
    first = classify_pipeline(
        (stage,),
        {"worker": observation},
        restart_counts={"worker": 0},
        stale_observation_counts={"worker": 0},
        max_restarts=2,
    )
    assert first.stages[0]["state"] == "slow"
    assert first.stages[0]["consecutive_stale_observations"] == 1
    assert first.restart_names == ()

    second = classify_pipeline(
        (stage,),
        {"worker": observation},
        restart_counts={"worker": 0},
        stale_observation_counts={"worker": 1},
        max_restarts=2,
    )
    assert second.stages[0]["state"] == "restart_pending"
    assert second.stages[0]["consecutive_stale_observations"] == 2
    assert second.restart_names == ("worker",)


def test_monitor_state_restores_progress_time_and_stale_count(tmp_path: Path) -> None:
    output = tmp_path / "pipeline.json"
    output.write_text(
        json.dumps(
            {
                "schema": "flex-pipeline-monitor-v1",
                "stages": [
                    {
                        "name": "worker",
                        "state": "slow",
                        "monitor_restart_count": 1,
                        "observed_systemd_restart_count": 2,
                        "cpu_usage_ns": 10,
                        "last_progress_value": {"completed_epochs": 5},
                        "last_progress_at": "2026-08-23T10:00:00+00:00",
                        "consecutive_stale_observations": 1,
                        "systemd": {"InvocationID": "abc"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = _load_monitor_state(output)
    assert state[5] == {"worker": {"completed_epochs": 5}}
    assert state[6] == {"worker": "2026-08-23T10:00:00+00:00"}
    assert state[7] == {"worker": 1}


def test_manifest_contains_dag_and_monitor_policy(tmp_path: Path) -> None:
    stages = _pipeline(tmp_path)
    manifest = _pipeline_manifest_payload(
        stages,
        poll_seconds=60,
        stale_seconds=1_800,
        max_restarts=2,
    )
    assert _manifest_path(tmp_path / "pipeline.json") == (
        tmp_path / "pipeline.manifest.json"
    )
    assert manifest["schema"] == "flex-pipeline-manifest-v1"
    assert len(manifest["stages"]) == len(stages)
    assert manifest["policy"]["required_consecutive_stale_observations"] == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2", 2), ("0", 0), ("-1", 0), ("[not set]", 0)],
)
def test_systemd_restart_count(raw: str, expected: int) -> None:
    assert _systemd_restart_count({"NRestarts": raw}) == expected


def test_primary_gate_releases_q_before_final_p_completion(tmp_path: Path) -> None:
    stages = _pipeline(tmp_path)
    for name in ("native", "full", "capacity", "compressed"):
        _write_completion(tmp_path / f"{name}.json")
    _write_completion(tmp_path / "weekly-p-primary.json")
    observations = {
        "native": _observation(tmp_path / "native.json", active=False),
        "full": _observation(tmp_path / "full.json", active=False),
        "capacity": _observation(tmp_path / "capacity.json", active=False),
        "compressed": _observation(tmp_path / "compressed.json", active=False),
        "weekly-p": _observation(
            tmp_path / "weekly-p.json", active=True, cpu_advanced=True
        ),
        "weekly-q": _observation(
            tmp_path / "weekly-q.json", active=True, cpu_advanced=True
        ),
    }
    result = classify_pipeline(
        stages,
        observations,
        restart_counts={stage.name: 0 for stage in stages},
        max_restarts=2,
    )
    states = {stage["name"]: stage["state"] for stage in result.stages}
    assert states["weekly-p"] == "running"
    assert states["weekly-q"] == "running"


def test_process_constraints_enforce_gpu_zero_and_four_workers() -> None:
    assert process_constraint_violations(
        ("python", "-m", "worker", "--device", "cuda:0", "--workers", "4")
    ) == ()
    violations = process_constraint_violations(
        ("python", "-m", "worker", "--device=cuda:1", "--workers=5")
    )
    assert violations == (
        "configured device is 'cuda:1', expected 'cuda:0'",
        "configured market workers are 5, maximum is 4",
    )


def test_invalid_completion_is_terminal(tmp_path: Path) -> None:
    completion = tmp_path / "invalid.json"
    completion.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    stage = PipelineStageSpec(
        name="invalid",
        unit="invalid.service",
        completion_path=completion,
    )
    observation = StageObservation(
        completion=CompletionEvidence(
            state="invalid", error="completion status is 'running'"
        ),
        active=True,
        properties={"ActiveState": "active"},
        recent_journal=True,
        cpu_usage_ns=1,
        cpu_advanced=True,
        command_line=(),
        newest_failure_path=None,
    )
    result = classify_pipeline(
        (stage,), {"invalid": observation}, restart_counts={}, max_restarts=2
    )
    assert result.stages[0]["state"] == "invalid_completion"
    assert result.terminal_names == ("invalid",)


def test_pipeline_payload_has_one_latch_and_terminal_serious_error() -> None:
    payload = _pipeline_payload(
        status="terminal_failure",
        warnings=[],
        resource_error=None,
        resource_stop_latched=False,
        terminal_names=["capacity", "compressed"],
        resources={},
        stage_payloads=[],
    )
    assert list(payload).count("resource_stop_latched") == 1
    assert payload["resource_stop_latched"] is False
    assert payload["durable_status"] == "FAILED_TERMINAL"
    assert payload["serious_error"] == (
        "terminal pipeline stage(s): capacity, compressed"
    )
