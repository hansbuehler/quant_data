from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.runners import (
    execute_production_matrix,
)


def test_execute_production_queue_runs_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    script = tmp_path / "write_result.py"
    script.write_text(
        """
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
results = output / "results" / "results.json"
results.parent.mkdir(parents=True, exist_ok=True)
results.write_text(json.dumps({
    "schema": "hedging-daily-results-v2",
    "model": "historical-sanos",
    "result_count": 2,
    "run_identity": "test-run",
}), encoding="utf-8")
print("synthetic production complete", flush=True)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-daily-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "historical-sanos-test",
                        "model": "historical-sanos",
                        "resource": "cpu",
                        "readiness": "ready",
                        "command": [
                            "python",
                            str(script),
                            str(output),
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(execute_production_matrix, "_check_resources", lambda: None)
    state_path = tmp_path / "queue.json"

    first = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=state_path,
        interpreter=Path(sys.executable),
        case_ids=("historical-sanos-test",),
        case_prefix=None,
        resource="cpu",
        expected_result_count=2,
        heartbeat_seconds=1.0,
    )
    second = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=state_path,
        interpreter=Path(sys.executable),
        case_ids=("historical-sanos-test",),
        case_prefix=None,
        resource="cpu",
        expected_result_count=2,
        heartbeat_seconds=1.0,
    )

    assert first["status"] == "complete"
    assert second == first
    records = first["records"]
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["status"] == "complete"
    assert "synthetic production complete" in (output / "run.log").read_text()


def test_execute_production_queue_rejects_unready_case(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-daily-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "blocked",
                        "model": "gromm",
                        "resource": "gpu",
                        "readiness": "blocked",
                        "command": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not ready"):
        execute_production_matrix.execute_production_queue(
            matrix_path=matrix,
            state_path=tmp_path / "queue.json",
            interpreter=Path(sys.executable),
            case_ids=("blocked",),
            case_prefix=None,
            resource="gpu",
            expected_result_count=2,
        )


def test_resource_probe_warns_when_gpuctl_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        execute_production_matrix,
        "GPUCTL_PATH",
        tmp_path / "missing-gpuctl",
    )

    execute_production_matrix._check_resources()

    assert "could not be launched" in capsys.readouterr().out


def test_production_child_environment_preserves_only_managed_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    gpu = execute_production_matrix._child_environment(resource="gpu")
    cpu = execute_production_matrix._child_environment(resource="cpu")

    assert gpu["CUDA_VISIBLE_DEVICES"] == "2"
    assert "JAX_PLATFORMS" not in gpu or gpu["JAX_PLATFORMS"] != "cpu"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu["JAX_PLATFORMS"] == "cpu"


def test_gpu_child_environment_requires_gpuctl_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="managed gpuctl allocation"):
        execute_production_matrix._child_environment(resource="gpu")


def test_execute_weekly_production_queue_validates_complete_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "weekly-output"
    script = tmp_path / "write_weekly_result.py"
    script.write_text(
        """
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
(output / "run.json").write_text(json.dumps({
    "schema": "hedging-weekly-evaluation-v1",
    "model": "sanos-lv",
    "run_identity": "weekly-test",
}), encoding="utf-8")
(output / "summary.json").write_text(json.dumps({
    "schema": "hedging-weekly-evaluation-v1",
    "eligible_origins": ["2023-01-17"],
    "completed_origins": ["2023-01-17"],
    "result_count": 13,
    "cap_hits": 0,
    "run_identity": "weekly-test",
}), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    matrix = tmp_path / "weekly-matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "frozen-weekly-sanos-lv-reference",
                        "model": "sanos-lv",
                        "resource": "cpu",
                        "readiness": "ready",
                        "command": [
                            "python",
                            str(script),
                            str(output),
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(execute_production_matrix, "_check_resources", lambda: None)

    state = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=tmp_path / "weekly-queue.json",
        interpreter=Path(sys.executable),
        case_ids=("frozen-weekly-sanos-lv-reference",),
        case_prefix=None,
        resource="cpu",
        expected_result_count=13,
        heartbeat_seconds=1.0,
    )

    assert state["status"] == "complete"
