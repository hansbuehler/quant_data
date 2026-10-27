"""Queue the GROMM2 forward-straddle job through gpuctl, without GPU imports."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def resource_report() -> dict[str, Any]:
    """Read only resource-manager thresholds; a failed probe is a warning."""
    try:
        result = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True, text=True, timeout=20, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"exit {result.returncode}")
        report = json.loads(result.stdout)
        if not isinstance(report.get("checks"), list) or not report["checks"]:
            raise ValueError("missing resource checks")
        health = [str(check["health"]).upper() for check in report["checks"]]
        blocked = any(value in {"STOP", "UNAVAILABLE", "ERROR"} for value in health)
        warned = any(value != "OK" for value in health)
        return {"blocked": blocked, "warning": warned, "report": report}
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, RuntimeError) as error:
        return {"blocked": False, "warning": True, "probe_error": str(error), "message": "gpuctl resources could not be read"}


def claim_gpu() -> str | None:
    """Retry only the manager's explicit busy/not-enough-device exit codes."""
    command = ["gpuctl", "claim", "--gpus", "1", "--owner-pid", str(os.getpid()), "--label", "gromm2-forward-straddle",
               "--task-name", "Risk-neutral SANOS GROMM2", "--json"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    if result.returncode in (3, 4):
        return None
    if result.returncode:
        raise RuntimeError(f"gpuctl claim failed ({result.returncode}): {result.stderr.strip()}")
    allocation = json.loads(result.stdout)["allocation"]["ref"]
    if not isinstance(allocation, str) or not allocation:
        raise ValueError("gpuctl claim returned no allocation reference")
    return allocation


def available_gpu_status() -> dict[str, Any]:
    """Fail closed unless the manager confirms both a healthy driver and free count."""
    result = subprocess.run(["gpuctl", "status", "--json"], capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        raise RuntimeError(f"gpuctl status failed ({result.returncode}): {result.stderr.strip()}")
    report = json.loads(result.stdout)
    available = report.get("available")
    if str(report.get("driver", "")).lower() != "ok":
        raise RuntimeError("GPU manager did not confirm a healthy driver; no claim attempted")
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise ValueError("GPU manager returned no valid available-device count; no claim attempted")
    return report


def request_cpu_checkpoint(service: str) -> bool:
    """Return whether SIGTERM was delivered; an already inactive unit is harmless."""
    result = subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", service], capture_output=True, text=True, timeout=20, check=False)
    if result.returncode == 0:
        return True
    state = subprocess.run(["systemctl", "--user", "show", service, "--property=ActiveState", "--property=LoadState", "--no-pager"],
                           capture_output=True, text=True, timeout=20, check=False)
    properties = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
    if state.returncode == 0 and (properties.get("ActiveState") in {"inactive", "failed"} or properties.get("LoadState") == "not-found"):
        return False
    raise RuntimeError(f"owned CPU service checkpoint request failed: {result.stderr.strip()}")


def terminate_child(child: subprocess.Popen, *, timeout: float = 60.0) -> dict[str, Any]:
    """Stop only the newly created child session, never another GPU owner."""
    if child.poll() is not None:
        return {"child_returncode": child.returncode, "child_killed": False, "checkpoint_trust": "last periodic checkpoint; final checkpoint not independently verified"}
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    killed = False
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        killed = True
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)
    return {"child_returncode": child.returncode, "child_killed": killed,
            "checkpoint_trust": "last periodic checkpoint only; final checkpoint not independently verified"}


class QueueStatus:
    """Persist the small queue state atomically and flush a JSON log line."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.started = time.monotonic()
        self.values: dict[str, Any] = {"allocation": None, "release_requires_user_approval": False}
        directory.mkdir(parents=True, exist_ok=True)

    def update(self, event: str, **values: Any) -> None:
        self.values.update(values)
        self.values.update(event=event, elapsed_seconds=time.monotonic() - self.started, timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, prefix=".gpu-status-", suffix=".json", delete=False) as stream:
            json.dump(self.values, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, self.directory / "gpu_queue_status.json")
        print(json.dumps(self.values, sort_keys=True, allow_nan=False), flush=True)


def release_approved(directory: Path, allocation: str) -> bool:
    """Accept an explicit, allocation-specific authorization, never a stale flag."""
    path = directory / "release-approved.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
        return value.get("approved") is True and value.get("allocation") == allocation
    except (OSError, ValueError):
        return False


def run_queue(*, run_directory: Path, command: list[str], cpu_service: str | None = None, poll_seconds: float = 60.0) -> int:
    """Claim once, checkpoint the owned CPU job, and supervise one GPU run."""
    if not command or not 0.0 < poll_seconds <= 60.0:
        raise ValueError("a child command and polling interval in (0, 60] are required")
    if cpu_service is not None and (not cpu_service.startswith("quant-data-gromm") or not cpu_service.endswith(".service")):
        raise ValueError("only the explicitly named project-owned GROMM CPU service may be stopped")
    status = QueueStatus(run_directory)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    allocation: str | None = None
    child: subprocess.Popen | None = None
    release_requires_approval = False
    try:
        while not stop.is_set():
            resources = resource_report()
            if resources["blocked"]:
                status.update("waiting_for_resource_headroom", resources=resources, completed_work_units=0, total_work_units=1)
                stop.wait(poll_seconds)
                continue
            if resources["warning"]:
                status.update("resource_warning", resources=resources)
            manager_status = available_gpu_status()
            if manager_status["available"] == 0:
                status.update("waiting_for_gpu", resources=resources, manager_status=manager_status, completed_work_units=0, total_work_units=1,
                              remaining_seconds=None, message="Manager reports no free GPU; no claim attempted and unrelated GPU jobs are untouched")
                stop.wait(poll_seconds)
                continue
            allocation = claim_gpu()
            if allocation is not None:
                break
            status.update("waiting_for_gpu", resources=resources, manager_status=manager_status, completed_work_units=0, total_work_units=1,
                          remaining_seconds=None, message="No free allocation; unrelated GPU jobs are untouched")
            stop.wait(poll_seconds)
        if allocation is None:
            status.update("cancelled_without_allocation")
            return 130
        status.update("allocation_obtained", allocation=allocation)
        if stop.is_set():
            status.update("cancelled_before_child_start")
            return 130
        if cpu_service is not None:
            delivered = request_cpu_checkpoint(cpu_service)
            status.update("cpu_checkpoint_requested" if delivered else "cpu_already_inactive", cpu_service=cpu_service)
        resources = resource_report()
        if resources["blocked"]:
            release_requires_approval = True
            status.update("resource_stop_before_child", resources=resources, release_requires_user_approval=True)
        else:
            if resources["warning"]:
                status.update("resource_warning", resources=resources)
            child = subprocess.Popen(["gpuctl", "run", "--allocation", allocation, "--", *command], start_new_session=True)
            status.update("running_gpu", child_pid=child.pid, command=command, completed_work_units=0, total_work_units=1)
            next_report = time.monotonic() + poll_seconds
            while child.poll() is None:
                if stop.is_set():
                    status.update("cancelled", **terminate_child(child))
                    return 130
                if time.monotonic() >= next_report:
                    resources = resource_report()
                    if resources["blocked"]:
                        release_requires_approval = True
                        termination = terminate_child(child)
                        status.update("resource_stopped", resources=resources, release_requires_user_approval=True, **termination)
                        break
                    status.update("running_gpu", resources=resources, completed_work_units=0, total_work_units=1,
                                  remaining_seconds=None, message="GPU child writes its own simulation/training progress")
                    next_report = time.monotonic() + poll_seconds
                stop.wait(min(1.0, poll_seconds))
            if not release_requires_approval:
                result_code = child.returncode
                status.update("completed" if result_code == 0 else "failed", child_returncode=result_code,
                              completed_work_units=1, total_work_units=1)
                return int(result_code) if result_code >= 0 else 128 - int(result_code)
        while not release_approved(run_directory, allocation):
            status.update("awaiting_release_approval", allocation=allocation, release_requires_user_approval=True,
                          resources=resource_report(), message="Child stopped. User authorization is required before releasing this allocation; no automatic restart.")
            time.sleep(poll_seconds)
        release_requires_approval = False
        status.update("release_authorized", release_requires_user_approval=False)
        return 20
    except Exception as error:
        if child is not None and child.poll() is None:
            status.update("failed_stopping_child", error=str(error), **terminate_child(child))
        else:
            status.update("failed", error=str(error))
        raise
    finally:
        if allocation is not None and not release_requires_approval:
            try:
                result = subprocess.run(["gpuctl", "release", "--allocation", allocation], capture_output=True, text=True, timeout=20, check=False)
                status.update("allocation_released" if result.returncode == 0 else "allocation_release_failed",
                              allocation_released=result.returncode == 0, release_error=result.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as error:
                status.update("allocation_release_failed", allocation_released=False, release_error=str(error))
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--cpu-service")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    raise SystemExit(run_queue(run_directory=arguments.run_directory, command=command, cpu_service=arguments.cpu_service))


if __name__ == "__main__":
    main()
