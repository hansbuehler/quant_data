"""Managed allocation identity and detached worker environment checks."""

import json
from types import SimpleNamespace

import pytest

from quantdata.hedging.model_audit import completion_queue


def test_worker_environment_does_not_override_managed_device(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unmanaged")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    path = tmp_path / "worker.env"
    path.write_text("QUANTDATA_ROOT=/data/quant_data\nPYTHONPATH=/source\n")
    environment = completion_queue.worker_environment(path)
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert "JAX_PLATFORMS" not in environment
    assert environment["QUANTDATA_ROOT"] == "/data/quant_data"
    assert completion_queue.worker_environment(path, cpu=True)["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("task_id,accepted", [(completion_queue.TASK_ID, True), ("another-thread", False)])
def test_allocation_requires_exact_thread_identity(monkeypatch, task_id, accepted):
    status = {"allocations": [{"ref": "owned-ref", "task_id": task_id}]}
    monkeypatch.setattr(completion_queue.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(stdout=json.dumps(status)))
    if accepted:
        completion_queue.verify_allocation("owned-ref")
    else:
        with pytest.raises(RuntimeError, match="exact hedging thread"):
            completion_queue.verify_allocation("owned-ref")


def test_queue_releases_gpu_after_both_fits_and_gromm_forecast_before_cpu_scores(tmp_path, monkeypatch):
    environment = tmp_path / "worker.env"
    environment.write_text("QUANTDATA_ROOT=/data/quant_data\nPYTHONPATH=/source\n")
    args = SimpleNamespace(reference=tmp_path, dysanos=tmp_path / "dysanos", gromm=tmp_path / "gromm",
        control=tmp_path / "control", market_inputs=tmp_path, dysanos_env=environment, gromm_env=environment,
        allocation="owned-ref", producer_unit="cpu-inputs.service")
    prepared = args.gromm / "gromm2/2024-04-01/prepared/COMPLETE.json"
    prepared.parent.mkdir(parents=True)
    prepared.write_text("{}")
    monkeypatch.setattr(completion_queue, "origins", lambda _: ["2024-04-01"])
    monkeypatch.setattr(completion_queue, "verify_allocation", lambda _: None)
    monkeypatch.setattr(completion_queue, "health_check", lambda: None)
    monkeypatch.setattr(completion_queue, "resource_health", lambda: ["OK"])
    events = []

    def run(command, *, environment):
        if "--score-only" in command:
            assert environment["CUDA_VISIBLE_DEVICES"] == ""
            events.append("cpu-score")
        elif "quantdata.hedging.model_audit.legacy_rolling" in command:
            events.append("dysanos-fit")
            marker = args.dysanos / "fits/dysanos1/2024-04-01/FIT_COMPLETE.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}")
        elif "--fit-only" in command:
            events.append("gromm-fit")
            (args.gromm / "gromm2/2024-04-01/ESTIMATION_COMPLETE.json").write_text("{}")
        else:
            events.append("gromm-gpu-forecast")

    monkeypatch.setattr(completion_queue, "run_owned", run)
    monkeypatch.setattr(completion_queue.subprocess, "run", lambda command, **kwargs:
        events.append("release") if command[:2] == ["gpuctl", "release"] else pytest.fail(str(command)))
    completion_queue.execute(args)
    assert events == ["dysanos-fit", "gromm-fit", "gromm-gpu-forecast", "release", "cpu-score"]
