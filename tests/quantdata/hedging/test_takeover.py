"""Takeover orchestration must stop rather than ignore an owned worker failure."""

import subprocess

import pytest

from quantdata.hedging.model_audit import takeover


def test_resource_stop_prevents_launch(monkeypatch) -> None:
    monkeypatch.setattr(takeover, "resource_health", lambda: ["OK", "STOP"])
    with pytest.raises(RuntimeError, match="Resource stop"):
        takeover.check_resources()


def test_worker_failure_is_not_an_accepted_fit(monkeypatch) -> None:
    monkeypatch.setattr(takeover, "resource_health", lambda: ["OK"])
    monkeypatch.setattr(takeover.subprocess, "Popen", lambda *args, **kwargs: FailedWorker())
    with pytest.raises(subprocess.CalledProcessError):
        takeover.run_child(["owned-model-worker"])


class FailedWorker:
    returncode = 2

    def poll(self):
        return self.returncode
