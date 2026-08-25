"""Monitor checkpointed FLEX systemd workers for failure and staleness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cdxcore.err import verify_inp

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import (
    GIB,
    RAM_TERMINATE_AVAILABLE_BYTES,
    RAM_WARNING_AVAILABLE_BYTES,
    available_memory_bytes,
    run_guarded_cli,
)

_MEDIA_WARNING_FREE_FRACTION = 0.20
_MEDIA_STOP_FREE_FRACTION = 0.10
_TMP_WARNING_USED_FRACTION = 0.80
_TMP_STOP_USED_FRACTION = 0.90


@dataclass(frozen=True)
class WorkerSpec:
    """One systemd service and the artifact proving successful completion."""

    unit: str
    completion_path: Path

    def __post_init__(self) -> None:
        verify_inp(
            self.unit.endswith(".service") and "/" not in self.unit,
            lambda: f"Invalid systemd service unit {self.unit!r}",
        )
        verify_inp(
            self.completion_path.is_absolute(),
            lambda: f"Completion path must be absolute: {self.completion_path}",
        )


def parse_worker_spec(value: str) -> WorkerSpec:
    """Parse ``UNIT::ABSOLUTE_COMPLETION_PATH`` from the command line."""
    unit, separator, raw_path = value.partition("::")
    verify_inp(
        separator != "" and raw_path != "",
        lambda: f"Expected UNIT::ABSOLUTE_COMPLETION_PATH, found {value!r}",
    )
    return WorkerSpec(unit=unit, completion_path=Path(raw_path))


def _run_systemctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("systemctl", "--user", *arguments),
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _service_properties(unit: str) -> dict[str, str]:
    result = _run_systemctl(
        "show",
        unit,
        "--property=ActiveState",
        "--property=SubState",
        "--property=MainPID",
        "--property=NRestarts",
        "--property=ExecMainStatus",
        "--property=CPUUsageNSec",
        "--property=InvocationID",
        "--property=ExecMainStartTimestamp",
        check=False,
    )
    if result.returncode != 0:
        return {
            "ActiveState": "not-found",
            "SubState": "not-found",
            "MainPID": "0",
            "NRestarts": "0",
            "ExecMainStatus": str(result.returncode),
            "CPUUsageNSec": "0",
        }
    return dict(
        line.split("=", maxsplit=1)
        for line in result.stdout.splitlines()
        if "=" in line
    )


def _cpu_usage_ns(properties: dict[str, str]) -> int | None:
    """Return cumulative cgroup CPU time when systemd exposes it."""
    raw_value = properties.get("CPUUsageNSec")
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def _has_recent_journal(unit: str, stale_seconds: int) -> bool:
    result = subprocess.run(
        (
            "journalctl",
            "--user",
            "-u",
            unit,
            "--since",
            f"{stale_seconds} seconds ago",
            "-n",
            "1",
            "--no-pager",
            "-o",
            "cat",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip() != ""


def _filesystem_payload(path: Path) -> dict[str, int | float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_fraction": usage.used / usage.total,
        "free_fraction": usage.free / usage.total,
    }


def _quantdata_resource_status() -> tuple[dict[str, Any], list[str], str | None]:
    """Inspect the filesystems configured by ``quantdata.env.locations``."""
    memory_available = available_memory_bytes()
    persistent_paths = {
        name: Path(getattr(quantdata_env.locations, name).path)
        for name in ("raw", "processing", "training", "experiments")
    }
    persistent = {
        name: _filesystem_payload(path)
        for name, path in persistent_paths.items()
    }
    temporary_path = Path(quantdata_env.locations.temp.path)
    temporary = _filesystem_payload(temporary_path)
    warnings: list[str] = []
    serious_error: str | None = None

    if memory_available <= RAM_TERMINATE_AVAILABLE_BYTES:
        serious_error = (
            "MemAvailable reached the FLEX termination threshold: "
            f"{memory_available / GIB:.3f} GiB"
        )
    elif memory_available < RAM_WARNING_AVAILABLE_BYTES:
        warnings.append(
            "MemAvailable is below the FLEX warning threshold: "
            f"{memory_available / GIB:.3f} GiB"
        )
    for name, payload in persistent.items():
        path = str(payload["path"])
        if float(payload["free_fraction"]) <= _MEDIA_STOP_FREE_FRACTION:
            serious_error = (
                f"The configured quantdata {name} filesystem at {path} "
                "reached its stop threshold"
            )
        elif float(payload["free_fraction"]) < _MEDIA_WARNING_FREE_FRACTION:
            warnings.append(
                f"The configured quantdata {name} filesystem at {path} "
                "is below 20% free"
            )
    if float(temporary["used_fraction"]) >= _TMP_STOP_USED_FRACTION:
        serious_error = (
            "The configured quantdata temporary filesystem at "
            f"{temporary_path} reached its stop threshold"
        )
    elif float(temporary["used_fraction"]) >= _TMP_WARNING_USED_FRACTION:
        warnings.append(
            "The configured quantdata temporary filesystem at "
            f"{temporary_path} is at least 80% used"
        )
    return (
        {
            "mem_available_bytes": memory_available,
            "persistent_locations": persistent,
            "temporary_location": temporary,
        },
        warnings,
        serious_error,
    )


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _stop_workers(workers: tuple[WorkerSpec, ...]) -> None:
    for worker in workers:
        properties = _service_properties(worker.unit)
        if properties.get("ActiveState") in {"active", "activating", "reloading"}:
            _run_systemctl("stop", worker.unit, check=False)


def monitor(
    workers: tuple[WorkerSpec, ...],
    *,
    output: Path,
    poll_seconds: int,
    stale_seconds: int,
    max_restarts: int,
) -> None:
    """Monitor until all workers complete or a serious error is recorded."""
    verify_inp(len(workers) > 0, lambda: "At least one worker is required")
    verify_inp(10 <= poll_seconds <= 600, lambda: "Invalid monitor poll interval")
    verify_inp(
        stale_seconds >= 2 * poll_seconds,
        lambda: "Stale interval must span at least two monitor polls",
    )
    verify_inp(max_restarts >= 0, lambda: "'max_restarts' cannot be negative")
    restart_counts = {worker.unit: 0 for worker in workers}
    previous_cpu_usage_ns: dict[str, int] = {}

    while True:
        timestamp = datetime.now(UTC).isoformat()
        resources, warnings, serious_error = _quantdata_resource_status()

        worker_payloads: list[dict[str, Any]] = []
        all_complete = True
        for worker in workers:
            complete = worker.completion_path.exists()
            properties = _service_properties(worker.unit)
            active = properties.get("ActiveState") == "active"
            cpu_usage_ns = _cpu_usage_ns(properties)
            prior_cpu_usage_ns = previous_cpu_usage_ns.get(worker.unit)
            cpu_advanced = (
                active
                and cpu_usage_ns is not None
                and prior_cpu_usage_ns is not None
                and cpu_usage_ns > prior_cpu_usage_ns
            )
            if cpu_usage_ns is not None:
                previous_cpu_usage_ns[worker.unit] = cpu_usage_ns
            recent_journal = active and _has_recent_journal(worker.unit, stale_seconds)
            # Dense LAPACK operations can be silent for longer than the stale
            # interval. Advancing cgroup CPU time is genuine forward activity;
            # a worker is stale only when both its journal and CPU have stopped.
            recent = complete or recent_journal or cpu_advanced
            if not complete:
                all_complete = False
            systemd_restarts = int(properties.get("NRestarts", "0") or 0)
            restart_counts[worker.unit] = max(
                restart_counts[worker.unit],
                systemd_restarts,
            )

            reason: str | None = None
            if not complete and not active:
                reason = f"service state is {properties.get('ActiveState')}/{properties.get('SubState')}"
            elif not complete and not recent:
                reason = f"no journal heartbeat for {stale_seconds} seconds"
            if reason is not None and serious_error is None:
                if restart_counts[worker.unit] >= max_restarts:
                    serious_error = (
                        f"{worker.unit} remains unhealthy after "
                        f"{restart_counts[worker.unit]} restarts: {reason}"
                    )
                else:
                    print(f"FLEX monitor restarting {worker.unit}: {reason}", flush=True)
                    restart_result = _run_systemctl("restart", worker.unit, check=False)
                    restart_counts[worker.unit] += 1
                    if restart_result.returncode != 0:
                        serious_error = (
                            f"Could not restart {worker.unit}: "
                            f"{restart_result.stderr.strip()}"
                        )

            worker_payloads.append(
                {
                    "unit": worker.unit,
                    "completion_path": str(worker.completion_path),
                    "complete": complete,
                    "recent_heartbeat": recent,
                    "recent_journal": recent_journal,
                    "cpu_usage_ns": cpu_usage_ns,
                    "cpu_advanced_since_poll": cpu_advanced,
                    "restart_count": restart_counts[worker.unit],
                    "systemd": properties,
                }
            )

        status = "serious_error" if serious_error else "complete" if all_complete else "monitoring"
        payload: dict[str, Any] = {
            "schema": "flex-service-monitor-v1",
            "timestamp_utc": timestamp,
            "status": status,
            "warnings": warnings,
            "serious_error": serious_error,
            "resources": resources,
            "workers": worker_payloads,
        }
        _atomic_write_json(payload, output)
        for warning in warnings:
            print(f"FLEX monitor warning: {warning}", flush=True)
        if serious_error is not None:
            print(f"FLEX monitor serious error: {serious_error}", flush=True)
            _stop_workers(workers)
            return
        if all_complete:
            print("FLEX monitor: all workers completed", flush=True)
            return
        time.sleep(poll_seconds)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        action="append",
        required=True,
        type=parse_worker_spec,
        help="UNIT::ABSOLUTE_COMPLETION_PATH; repeat for each monitored worker",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=1_800)
    parser.add_argument("--max-restarts", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    arguments = _parse_arguments()
    monitor(
        tuple(arguments.worker),
        output=arguments.output,
        poll_seconds=arguments.poll_seconds,
        stale_seconds=arguments.stale_seconds,
        max_restarts=arguments.max_restarts,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="monitor_flex_services")
