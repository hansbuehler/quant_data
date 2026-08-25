from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from quantdata.dysanos.flex.failure_reporting import (
    GIB,
    LowAvailableMemoryError,
    available_memory_bytes,
    failure_directory_from_argv,
    ram_limits_for_hostname,
    run_guarded_cli,
)


def _write_meminfo(path: Path, *, available_bytes: int) -> None:
    path.write_text(
        f"MemTotal:       {1024 * 1024 * 1024} kB\n"
        f"MemAvailable:   {available_bytes // 1024} kB\n",
        encoding="utf-8",
    )


def test_available_memory_uses_memavailable(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    _write_meminfo(path, available_bytes=321 * GIB)
    assert available_memory_bytes(path) == 321 * GIB


def test_ram_limits_are_host_specific() -> None:
    assert ram_limits_for_hostname("Sommer") == (30 * GIB, 20 * GIB)
    assert ram_limits_for_hostname("ws-artp-wrx900-01") == (200 * GIB, 100 * GIB)


def test_guarded_cli_warns_below_ram_warning_threshold(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "meminfo"
    _write_meminfo(path, available_bytes=25 * GIB)
    run_guarded_cli(
        lambda: None,
        program="ram-warning-test",
        failure_directory=tmp_path / "failures",
        ram_meminfo_path=path,
        ram_warning_available_bytes=30 * GIB,
        ram_terminate_available_bytes=20 * GIB,
    )
    assert "FLEX RAM warning" in capfd.readouterr().err
    assert not tuple((tmp_path / "failures").iterdir())


def test_guarded_cli_stops_and_records_ram_termination_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meminfo"
    _write_meminfo(path, available_bytes=20 * GIB)
    failure_directory = tmp_path / "failures"
    with pytest.raises(LowAvailableMemoryError, match="at or below"):
        run_guarded_cli(
            lambda: None,
            program="ram-termination-test",
            failure_directory=failure_directory,
            ram_meminfo_path=path,
            ram_warning_available_bytes=30 * GIB,
            ram_terminate_available_bytes=20 * GIB,
        )
    payload = json.loads(
        (failure_directory / "ram-termination-test-latest.failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["source"] == "ram_guard"
    assert payload["extra"]["available_bytes"] == 20 * GIB
    assert payload["extra"]["warning_available_bytes"] == 30 * GIB
    assert payload["extra"]["terminate_available_bytes"] == 20 * GIB
    assert not tuple(failure_directory.glob("*.fatal.log"))


def test_guarded_cli_allows_a_supervisor_to_own_the_ram_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meminfo"
    _write_meminfo(path, available_bytes=19 * GIB)
    called: list[bool] = []
    run_guarded_cli(
        lambda: called.append(True),
        program="external-ram-guard-test",
        failure_directory=tmp_path / "failures",
        ram_meminfo_path=path,
        enable_ram_guard=False,
    )
    assert called == [True]
    assert not tuple((tmp_path / "failures").iterdir())


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM unavailable")
def test_ram_monitor_terminates_a_running_activity(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    _write_meminfo(path, available_bytes=25 * GIB)
    failure_directory = tmp_path / "failures"

    def consume_memory_headroom() -> None:
        _write_meminfo(path, available_bytes=19 * GIB)
        time.sleep(2.0)

    with pytest.raises(SystemExit) as raised:
        run_guarded_cli(
            consume_memory_headroom,
            program="ram-monitor-test",
            failure_directory=failure_directory,
            ram_poll_seconds=0.01,
            ram_meminfo_path=path,
            ram_warning_available_bytes=30 * GIB,
            ram_terminate_available_bytes=20 * GIB,
        )
    assert raised.value.code == 128 + signal.SIGTERM
    payload = json.loads(
        (failure_directory / "ram-monitor-test-latest.failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["source"] == "ram_guard"
    assert payload["extra"]["available_bytes"] == 19 * GIB
    assert not tuple(failure_directory.glob("*.fatal.log"))


def test_guarded_cli_records_python_exception_and_preserves_failure(
    tmp_path: Path,
) -> None:
    def fail() -> None:
        raise RuntimeError("deliberate top-level failure")

    with pytest.raises(RuntimeError, match="deliberate top-level failure"):
        run_guarded_cli(
            fail,
            program="failure-test",
            failure_directory=tmp_path,
        )

    latest = json.loads(
        (tmp_path / "failure-test-latest.failure.json").read_text(encoding="utf-8")
    )
    unique = tuple(
        path
        for path in tmp_path.glob("failure-test-*.failure.json")
        if path.name != "failure-test-latest.failure.json"
    )
    assert len(unique) == 1
    assert latest["schema"] == "flex-process-failure-v1"
    assert latest["source"] == "main"
    assert latest["exception"]["type"] == "RuntimeError"
    assert latest["exception"]["message"] == "deliberate top-level failure"
    assert "raise RuntimeError" in latest["exception"]["traceback"]
    assert latest["process"]["pid"] == os.getpid()
    assert not tuple(tmp_path.glob("*.fatal.log"))


def test_guarded_cli_normal_completion_leaves_no_failure_artifact(
    tmp_path: Path,
) -> None:
    run_guarded_cli(
        lambda: None,
        program="success-test",
        failure_directory=tmp_path,
    )
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM unavailable")
def test_guarded_cli_can_treat_supervisor_sigterm_as_clean(
    tmp_path: Path,
) -> None:
    code = """
import sys
import time
from pathlib import Path
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli

run_guarded_cli(
    lambda: time.sleep(30.0),
    program="clean-signal-test",
    failure_directory=Path(sys.argv[1]),
    clean_sigterm=True,
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        text=True,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not tuple(tmp_path.glob("*.fatal.log")):
        if process.poll() is not None:
            raise AssertionError(f"Child exited early with code {process.returncode}")
        time.sleep(0.02)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=10.0) == 0
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (("--report-directory", "reports/run"), Path("reports/run/failures")),
        (("--output-directory=reports/out",), Path("reports/out/failures")),
        (("--output", "reports/result.json"), Path("reports/failures")),
    ],
)
def test_failure_directory_uses_cli_output_location(
    arguments: tuple[str, ...],
    expected: Path,
) -> None:
    assert failure_directory_from_argv(arguments) == expected


def _wait_for_path(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise AssertionError(f"Child exited early with code {process.returncode}")
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {path}")


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM unavailable")
def test_guarded_cli_records_sigterm_before_nonzero_exit(tmp_path: Path) -> None:
    code = """
import sys
import time
from pathlib import Path
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli

run_guarded_cli(
    lambda: time.sleep(30.0),
    program="signal-test",
    failure_directory=Path(sys.argv[1]),
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        text=True,
    )
    _wait_for_path(next_path := tmp_path, process)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not tuple(next_path.glob("*.fatal.log")):
        if process.poll() is not None:
            raise AssertionError(f"Child exited early with code {process.returncode}")
        time.sleep(0.02)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=10.0) == 128 + signal.SIGTERM

    payload = json.loads(
        (tmp_path / "signal-test-latest.failure.json").read_text(encoding="utf-8")
    )
    assert payload["source"] == "signal"
    assert payload["signal"] == {"number": signal.SIGTERM, "name": "SIGTERM"}
    assert not tuple(tmp_path.glob("*.fatal.log"))


def test_uncaught_thread_exception_is_persisted_before_process_exit(
    tmp_path: Path,
) -> None:
    code = """
import sys
import threading
from pathlib import Path
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli

def fail_thread():
    raise RuntimeError("deliberate worker-thread failure")

def main():
    worker = threading.Thread(target=fail_thread, name="failure-worker")
    worker.start()
    worker.join()

run_guarded_cli(
    main,
    program="thread-test",
    failure_directory=Path(sys.argv[1]),
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=10.0,
        check=False,
    )
    assert completed.returncode == 1
    payload = json.loads(
        (tmp_path / "thread-test-latest.failure.json").read_text(encoding="utf-8")
    )
    assert payload["source"] == "thread"
    assert payload["process"]["thread"] == "failure-worker"
    assert payload["exception"]["message"] == "deliberate worker-thread failure"
