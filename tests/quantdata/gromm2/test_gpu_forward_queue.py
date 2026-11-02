from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from quantdata.gromm2 import gpu_forward_queue as queue


def result(*, code: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout, stderr)


@pytest.mark.parametrize("health,blocked,warning", [("OK", False, False), ("WARN", False, True), ("STOP", True, True), ("UNAVAILABLE", True, True), ("ERROR", True, True)])
def test_resource_policy_uses_only_manager_health(monkeypatch, health: str, blocked: bool, warning: bool) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: result(stdout=json.dumps({"checks": [{"health": health}]})))
    report = queue.resource_report()
    assert report["blocked"] is blocked
    assert report["warning"] is warning


def test_resource_probe_failure_is_an_explicit_warning(monkeypatch) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: result(code=1, stderr="manager offline"))
    report = queue.resource_report()
    assert report["warning"] and not report["blocked"]
    assert "could not be read" in report["message"]


def timeout_check():
    return {"health": "ERROR", "name": "storage", "current": None,
        "error": "TimeoutError: probe did not finish within 10s"}


def test_transient_embedded_timeout_recovers_without_stopping(monkeypatch, capsys) -> None:
    replies = iter([result(stdout=json.dumps({"checks": [timeout_check()]})),
        result(stdout=json.dumps({"checks": [{"health": "OK"}]}))])
    delays = []
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(queue.time, "sleep", delays.append)
    report = queue.resource_report()
    assert not report["blocked"] and not report["warning"]
    assert report["probe_attempts"] == 2
    assert delays == [15.0]
    assert "resource-probe-timeout-retry" in capsys.readouterr().out


def test_persistent_embedded_timeout_blocks_only_after_bounded_retries(monkeypatch) -> None:
    calls, delays = [], []
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs:
        calls.append(args) or result(stdout=json.dumps({"checks": [timeout_check()]})))
    monkeypatch.setattr(queue.time, "sleep", delays.append)
    report = queue.resource_report()
    assert report["blocked"] and report["warning"]
    assert report["probe_attempts"] == len(calls) == 3
    assert delays == [15.0, 30.0]


@pytest.mark.parametrize("other", [{"health": "STOP", "name": "ram"},
    {"health": "ERROR", "name": "disk", "error": "permission denied"},
    {"health": "UNAVAILABLE", "name": "disk", "error": "not mounted"},
    {"health": "ERROR", "name": "disk", "current": {"free_pct": 1}, "error": "TimeoutError"}])
def test_timeout_never_delays_another_confirmed_resource_problem(monkeypatch, other) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs:
        result(stdout=json.dumps({"checks": [timeout_check(), other]})))
    def forbidden_sleep(_):
        raise AssertionError("confirmed resource problems must not be delayed")
    monkeypatch.setattr(queue.time, "sleep", forbidden_sleep)
    assert queue.resource_report()["blocked"]


def test_cli_timeout_is_retried_but_remains_a_warning_under_host_policy(monkeypatch) -> None:
    calls, delays = [], []
    def timeout(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired("gpuctl resources", 20)
    monkeypatch.setattr(queue.subprocess, "run", timeout)
    monkeypatch.setattr(queue.time, "sleep", delays.append)
    report = queue.resource_report()
    assert not report["blocked"] and report["warning"]
    assert report["probe_attempts"] == len(calls) == 3
    assert delays == [15.0, 30.0]


@pytest.mark.parametrize("code", [3, 4])
def test_only_explicit_gpu_busy_codes_retry(monkeypatch, code: int) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: result(code=code))
    assert queue.claim_gpu() is None


def test_claim_failure_does_not_silently_loop(monkeypatch) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: result(code=5, stderr="permission denied"))
    with pytest.raises(RuntimeError, match="permission denied"):
        queue.claim_gpu()


def test_claim_is_bound_to_queue_owner_pid(monkeypatch) -> None:
    seen = []
    def run(command, **kwargs):
        seen.append((command, kwargs))
        return result(stdout=json.dumps({"allocation": {"ref": "allocation-test"}}))
    monkeypatch.setattr(queue.subprocess, "run", run)
    assert queue.claim_gpu() == "allocation-test"
    command, kwargs = seen[0]
    assert command[command.index("--owner-pid") + 1] == str(queue.os.getpid())
    assert kwargs["timeout"] == 20


def test_release_requires_matching_explicit_authorization(tmp_path: Path) -> None:
    marker = tmp_path / "release-approved.json"
    assert not queue.release_approved(tmp_path, "current")
    marker.write_text(json.dumps({"approved": True, "allocation": "old"}))
    assert not queue.release_approved(tmp_path, "current")
    marker.write_text(json.dumps({"approved": True, "allocation": "current"}))
    assert queue.release_approved(tmp_path, "current")


def test_child_shutdown_targets_only_owned_session_and_escalates_after_timeout(monkeypatch) -> None:
    calls = []
    class Child:
        pid = 43210
        returncode = None
        def poll(self):
            return None
        def wait(self, *, timeout):
            calls.append(("wait", timeout))
            if timeout == 60:
                raise subprocess.TimeoutExpired("owned-child", timeout)
            self.returncode = -9
            return self.returncode
    monkeypatch.setattr(queue.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    status = queue.terminate_child(Child())
    assert calls == [(43210, signal.SIGTERM), ("wait", 60), (43210, signal.SIGKILL), ("wait", 5)]
    assert status["child_killed"]
    assert "periodic checkpoint only" in status["checkpoint_trust"]


def test_queue_handoff_and_release_never_touch_unrelated_service(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(queue, "resource_report", lambda: {"blocked": False, "warning": False})
    monkeypatch.setattr(queue, "available_gpu_status", lambda: {"available": 1, "driver": "ok"})
    monkeypatch.setattr(queue, "claim_gpu", lambda: calls.append("claim") or "owned-allocation")
    class Child:
        pid = 43210
        returncode = 0
        def poll(self):
            return self.returncode
    def launch(command, **kwargs):
        calls.append(("launch", command, kwargs))
        return Child()
    def run(command, **kwargs):
        calls.append(("run", command))
        return result()
    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    monkeypatch.setattr(queue.subprocess, "run", run)
    service = "quant-data-gromm-forward-owned.service"
    assert queue.run_queue(run_directory=tmp_path, command=["python", "owned-job.py"], cpu_service=service) == 0
    assert calls[0] == "claim"
    assert calls[1] == ("run", ["systemctl", "--user", "kill", "--signal=SIGTERM", service])
    assert calls[2] == ("launch", ["gpuctl", "run", "--allocation", "owned-allocation", "--", "python", "owned-job.py"], {"start_new_session": True})
    assert calls[3] == ("run", ["gpuctl", "release", "--allocation", "owned-allocation"])
    saved = json.loads((tmp_path / "gpu_queue_status.json").read_text())
    assert saved["allocation_released"] and saved["child_returncode"] == 0


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("gpuctl status", 20),
    queue.GpuProbeUnavailableError("driver health is temporarily unknown")])
def test_gpu_probe_failure_waits_without_claiming_and_recovers(monkeypatch, tmp_path: Path, capsys, failure) -> None:
    calls = []
    responses = iter([failure, {"available": 1, "driver": "ok"}])

    def status():
        value = next(responses)
        calls.append("probe")
        if isinstance(value, Exception):
            raise value
        return value

    class Child:
        pid = 43210
        returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(queue, "resource_report", lambda: {"blocked": False, "warning": False})
    monkeypatch.setattr(queue, "available_gpu_status", status)
    monkeypatch.setattr(queue, "claim_gpu", lambda: calls.append("claim") or "owned-allocation")
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: calls.append(command) or result())
    assert queue.run_queue(run_directory=tmp_path, command=["owned-job"], poll_seconds=.001) == 0
    assert calls[:3] == ["probe", "probe", "claim"]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["event"] == "waiting_for_gpu_status"
    assert events[0]["allocation"] is None
    assert events[-1]["allocation_released"]


def test_gpu_status_cli_timeout_is_transient_but_permission_errors_are_not(monkeypatch) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs:
        result(code=5, stderr="cannot talk to resourced: timed out"))
    with pytest.raises(queue.GpuProbeUnavailableError):
        queue.available_gpu_status()
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs:
        result(code=5, stderr="permission denied"))
    with pytest.raises(RuntimeError, match="permission denied") as caught:
        queue.available_gpu_status()
    assert not isinstance(caught.value, queue.GpuProbeUnavailableError)


def test_resource_stop_keeps_claim_until_explicit_release_approval(monkeypatch, tmp_path: Path, capsys) -> None:
    reports = iter([{"blocked": False, "warning": False}, {"blocked": False, "warning": False},
                    {"blocked": True, "warning": True}, {"blocked": False, "warning": False}])
    monkeypatch.setattr(queue, "resource_report", lambda: next(reports))
    monkeypatch.setattr(queue, "available_gpu_status", lambda: {"available": 1, "driver": "ok"})
    monkeypatch.setattr(queue, "claim_gpu", lambda: "owned-allocation")
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(queue.time, "monotonic", lambda: next(clock))
    class Child:
        pid = 43210
        returncode = None
        def poll(self):
            return self.returncode
    child = Child()
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: child)
    def terminate(owned):
        assert owned is child
        child.returncode = -15
        return {"child_returncode": -15, "child_killed": False, "checkpoint_trust": "last periodic checkpoint only"}
    monkeypatch.setattr(queue, "terminate_child", terminate)
    def sleep(seconds):
        saved = json.loads((tmp_path / "gpu_queue_status.json").read_text())
        assert saved["release_requires_user_approval"]
        (tmp_path / "release-approved.json").write_text(json.dumps({"approved": True, "allocation": "owned-allocation"}))
    monkeypatch.setattr(queue.time, "sleep", sleep)
    calls = []
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: calls.append(command) or result())
    assert queue.run_queue(run_directory=tmp_path, command=["owned-job"], poll_seconds=0.001) == 20
    assert calls == [["gpuctl", "release", "--allocation", "owned-allocation"]]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    stopped = next(event for event in events if event["event"] == "resource_stopped")
    assert stopped["release_requires_user_approval"]
    assert [event["event"] for event in events].index("awaiting_release_approval") < [event["event"] for event in events].index("allocation_released")


def test_unrelated_cpu_service_is_rejected_before_any_resource_action(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project-owned"):
        queue.run_queue(run_directory=tmp_path, command=["owned-job"], cpu_service="unrelated-user.service")


@pytest.mark.parametrize("report", [{"driver": "unknown", "available": 0}, {"driver": "ok"}, {"driver": "ok", "available": -1}])
def test_unknown_manager_gpu_state_fails_closed(monkeypatch, report: dict) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: result(stdout=json.dumps(report)))
    with pytest.raises((RuntimeError, ValueError)):
        queue.available_gpu_status()


def test_manager_status_command_failure_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: result(code=1, stderr="manager unavailable"))
    with pytest.raises(RuntimeError, match="status failed"):
        queue.available_gpu_status()


def test_busy_manager_never_attempts_claim(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(queue, "resource_report", lambda: {"blocked": False, "warning": False})
    monkeypatch.setattr(queue, "available_gpu_status", lambda: {"available": 0, "driver": "ok"})
    def forbidden_claim():
        raise AssertionError("claim must not run while manager says GPU busy")
    monkeypatch.setattr(queue, "claim_gpu", forbidden_claim)
    def stop_wait(seconds):
        signal.raise_signal(signal.SIGTERM)
        return True
    class Event:
        stopped = False
        def set(self):
            self.stopped = True
        def is_set(self):
            return self.stopped
        def wait(self, seconds):
            return stop_wait(seconds)
    monkeypatch.setattr(queue.threading, "Event", Event)
    assert queue.run_queue(run_directory=tmp_path, command=["owned-job"]) == 130


@pytest.mark.parametrize("active,load,expected", [("inactive", "loaded", False), ("failed", "loaded", False), ("inactive", "not-found", False), ("active", "loaded", None)])
def test_already_stopped_cpu_service_does_not_abort_gpu_handoff(monkeypatch, active: str, load: str, expected: bool | None) -> None:
    results = iter([result(code=1, stderr="kill failed"), result(stdout=f"ActiveState={active}\nLoadState={load}\n")])
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: next(results))
    if expected is None:
        with pytest.raises(RuntimeError, match="checkpoint request failed"):
            queue.request_cpu_checkpoint("quant-data-gromm-owned.service")
    else:
        assert queue.request_cpu_checkpoint("quant-data-gromm-owned.service") is expected
