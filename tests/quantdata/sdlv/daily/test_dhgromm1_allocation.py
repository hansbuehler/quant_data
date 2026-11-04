"""Delayed GPU reservation ownership and post-exit cleanup."""

import json
from types import SimpleNamespace

import pytest

from quantdata.sdlv.dhgromm1.allocation import claim_training_gpu, release_after_exit


def test_claim_uses_only_manager_assigned_devices(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RESOURCED_ALLOCATION", "")
    response = dict(allocation=dict(ref="managed-ref", gpus=["GPU-assigned"]),
                    env=dict(CUDA_VISIBLE_DEVICES="GPU-assigned", RESOURCED_ALLOCATION="managed-ref"))
    calls = []

    def run(command, **kwargs):
        if command[1] == "resources":
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(checks=[dict(health="OK")])), stderr="")
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr("quantdata.sdlv.dhgromm1.allocation.subprocess.run", run)
    result = claim_training_gpu(tmp_path)
    assert result["gpus"] == ["GPU-assigned"]
    assert "--owner-pid" in calls[0]
    assert json.loads((tmp_path / "gpu-allocation.json").read_text())["ref"] == "managed-ref"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-assigned"
    assert os.environ["RESOURCED_ALLOCATION"] == "managed-ref"


def test_post_exit_hook_refuses_to_release_live_owner(tmp_path, monkeypatch) -> None:
    (tmp_path / "gpu-allocation.json").write_text(json.dumps(dict(ref="managed-ref", owner_pid=1234)))
    monkeypatch.setattr("quantdata.sdlv.dhgromm1.allocation.os.kill", lambda *args: None)
    with pytest.raises(RuntimeError, match="owner 1234 is alive"):
        release_after_exit(tmp_path)


def test_post_exit_hook_releases_only_recorded_allocation(tmp_path, monkeypatch) -> None:
    (tmp_path / "gpu-allocation.json").write_text(json.dumps(dict(ref="managed-ref", owner_pid=1234)))
    calls = []

    def exited(*args):
        raise ProcessLookupError

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("quantdata.sdlv.dhgromm1.allocation.os.kill", exited)
    monkeypatch.setattr("quantdata.sdlv.dhgromm1.allocation.subprocess.run", run)
    release_after_exit(tmp_path)
    assert calls[0][calls[0].index("--allocation") + 1] == "managed-ref"
    assert "--force" in calls[0]


def test_post_exit_hook_is_harmless_before_any_claim(tmp_path) -> None:
    release_after_exit(tmp_path)


def test_claim_refuses_resource_stop_before_requesting_device(tmp_path, monkeypatch) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(checks=[dict(health="STOP")])), stderr="")

    monkeypatch.setattr("quantdata.sdlv.dhgromm1.allocation.subprocess.run", run)
    with pytest.raises(RuntimeError, match="blocks the GPU training launch"):
        claim_training_gpu(tmp_path)
    assert len(calls) == 1 and calls[0][1] == "resources"
