"""Independent heartbeat watchdog for one persistent FLEX DAG monitor."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from cdxcore.err import verify_inp

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.monitor_services import (
    _atomic_write_json,
    _run_systemctl,
    _service_properties,
)
from quantdata.dysanos.flex.pipeline_monitor import (
    _append_json_event,
    _event_log_path,
    _heartbeat_path,
)


@dataclass(frozen=True)
class WatchdogDecision:
    """Pure watchdog classification before any service mutation."""

    state: str
    reason: str | None
    restart: bool
    consecutive_unhealthy_observations: int = 0


def classify_monitor_health(
    *,
    active: bool,
    heartbeat_age_seconds: float | None,
    heartbeat_valid: bool,
    monitor_status: str | None,
    restart_count: int,
    max_restarts: int,
    stale_seconds: int,
    consecutive_unhealthy_observations: int = 0,
) -> WatchdogDecision:
    """Classify a monitor from independent service and heartbeat evidence."""
    if heartbeat_valid and monitor_status == "complete":
        return WatchdogDecision(
            state="SUCCEEDED",
            reason="monitor certified pipeline completion",
            restart=False,
            consecutive_unhealthy_observations=0,
        )
    if heartbeat_valid and monitor_status == "terminal_failure":
        return WatchdogDecision(
            state="FAILED_TERMINAL",
            reason="monitor certified terminal pipeline failure",
            restart=False,
            consecutive_unhealthy_observations=0,
        )
    if (
        active
        and heartbeat_valid
        and heartbeat_age_seconds is not None
        and heartbeat_age_seconds <= stale_seconds
    ):
        return WatchdogDecision(
            state="RUNNING",
            reason=None,
            restart=False,
            consecutive_unhealthy_observations=0,
        )
    reasons: list[str] = []
    if not active:
        reasons.append("monitor service is not active")
    if not heartbeat_valid:
        reasons.append("monitor heartbeat is missing or invalid")
    elif heartbeat_age_seconds is None:
        reasons.append("monitor heartbeat age is unavailable")
    elif heartbeat_age_seconds > stale_seconds:
        reasons.append(
            f"monitor heartbeat is stale by {heartbeat_age_seconds:.1f} seconds"
        )
    reason = "; ".join(reasons) or "monitor health is indeterminate"
    unhealthy_count = consecutive_unhealthy_observations + 1
    if unhealthy_count < 2:
        return WatchdogDecision(
            state="SUSPECT",
            reason="first consecutive unhealthy observation: " + reason,
            restart=False,
            consecutive_unhealthy_observations=unhealthy_count,
        )
    if restart_count < max_restarts:
        return WatchdogDecision(
            state="RESTART_PENDING",
            reason=reason,
            restart=True,
            consecutive_unhealthy_observations=unhealthy_count,
        )
    return WatchdogDecision(
        state="FAILED_TERMINAL",
        reason=reason,
        restart=False,
        consecutive_unhealthy_observations=unhealthy_count,
    )


def _read_heartbeat(path: Path) -> tuple[bool, float | None, str | None]:
    if not path.exists():
        return False, None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(str(payload["timestamp_utc"]))
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False, None, None
    valid = (
        isinstance(payload, Mapping)
        and payload.get("schema") == "flex-pipeline-monitor-heartbeat-v1"
        and timestamp.tzinfo is not None
    )
    if not valid:
        return False, None, None
    return (
        True,
        max(
            (datetime.now(UTC) - timestamp.astimezone(UTC)).total_seconds(),
            0.0,
        ),
        str(payload.get("status")) if payload.get("status") is not None else None,
    )


def _load_state(output: Path) -> tuple[int, str | None, str | None, int]:
    if not output.exists():
        return 0, None, None, 0
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, None, None, 0
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "flex-pipeline-monitor-watchdog-v1"
    ):
        return 0, None, None, 0
    return (
        max(int(payload.get("restart_count", 0)), 0),
        str(payload.get("state")) if payload.get("state") is not None else None,
        (
            str(payload.get("monitor_invocation_id"))
            if payload.get("monitor_invocation_id") is not None
            else None
        ),
        max(int(payload.get("consecutive_unhealthy_observations", 0)), 0),
    )


def monitor_watchdog(
    *,
    unit: str,
    monitor_heartbeat: Path,
    output: Path,
    poll_seconds: int,
    stale_seconds: int,
    max_restarts: int,
) -> None:
    """Watch one DAG monitor until completion or an explicit operator stop."""
    verify_inp(
        unit.endswith(".service") and "/" not in unit,
        lambda: f"Invalid monitor service unit {unit!r}",
    )
    verify_inp(10 <= poll_seconds <= 600, lambda: "Invalid watchdog poll interval")
    verify_inp(
        stale_seconds >= 2 * poll_seconds,
        lambda: "Watchdog stale interval must span at least two polls",
    )
    verify_inp(max_restarts >= 0, lambda: "'max_restarts' cannot be negative")
    (
        restart_count,
        previous_state,
        previous_invocation,
        consecutive_unhealthy_observations,
    ) = _load_state(output)

    while True:
        properties = _service_properties(unit)
        active = properties.get("ActiveState") == "active"
        invocation = properties.get("InvocationID", "")
        heartbeat_valid, heartbeat_age, monitor_status = _read_heartbeat(
            monitor_heartbeat
        )
        if (
            previous_state == "FAILED_TERMINAL"
            and active
            and heartbeat_valid
            and invocation not in {"", previous_invocation}
        ):
            # A distinct healthy invocation is explicit operator recovery.
            restart_count = 0
            consecutive_unhealthy_observations = 0
        decision = classify_monitor_health(
            active=active,
            heartbeat_age_seconds=heartbeat_age,
            heartbeat_valid=heartbeat_valid,
            monitor_status=monitor_status,
            restart_count=restart_count,
            max_restarts=max_restarts,
            stale_seconds=stale_seconds,
            consecutive_unhealthy_observations=(
                consecutive_unhealthy_observations
            ),
        )
        consecutive_unhealthy_observations = (
            decision.consecutive_unhealthy_observations
        )
        action: str | None = None
        state = decision.state
        reason = decision.reason
        if decision.restart:
            result = _run_systemctl("restart", unit, check=False)
            restart_count += 1
            if result.returncode == 0:
                action = "restarted_monitor"
                state = "RESTARTING"
            else:
                state = "FAILED_TERMINAL"
                reason = f"systemctl restart failed: {result.stderr.strip()}"
        timestamp = datetime.now(UTC).isoformat()
        payload = {
            "schema": "flex-pipeline-monitor-watchdog-v1",
            "timestamp_utc": timestamp,
            "watchdog_pid": os.getpid(),
            "state": state,
            "reason": reason,
            "action": action,
            "restart_count": restart_count,
            "max_restarts": max_restarts,
            "consecutive_unhealthy_observations": (
                consecutive_unhealthy_observations
            ),
            "monitor_unit": unit,
            "monitor_invocation_id": invocation,
            "monitor_active": active,
            "monitor_heartbeat_path": str(monitor_heartbeat),
            "monitor_heartbeat_valid": heartbeat_valid,
            "monitor_heartbeat_age_seconds": heartbeat_age,
            "monitor_status": monitor_status,
            "durability_requirement": (
                "This watchdog must itself be launched by the OS with "
                "Restart=on-failure; returning from an agent does not keep it alive."
            ),
        }
        _atomic_write_json(payload, output)
        _atomic_write_json(
            {
                "schema": "flex-pipeline-monitor-watchdog-heartbeat-v1",
                "timestamp_utc": timestamp,
                "watchdog_pid": os.getpid(),
                "state": state,
                "state_path": str(output),
            },
            _heartbeat_path(output),
        )
        if previous_state != state or action is not None or state == "FAILED_TERMINAL":
            _append_json_event(
                {
                    "schema": "flex-pipeline-monitor-watchdog-event-v1",
                    "timestamp_utc": timestamp,
                    "from": previous_state,
                    "to": state,
                    "reason": reason,
                    "action": action,
                    "restart_count": restart_count,
                    "monitor_invocation_id": invocation,
                },
                _event_log_path(output),
            )
        if action is not None:
            print(f"FLEX monitor watchdog action: {action}: {reason}", flush=True)
        elif state == "FAILED_TERMINAL":
            print(f"FLEX monitor watchdog terminal failure: {reason}", flush=True)
        elif state == "SUCCEEDED":
            print("FLEX monitor watchdog: pipeline completed", flush=True)
            return
        previous_state = state
        previous_invocation = invocation
        time.sleep(poll_seconds)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--monitor-heartbeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=300)
    parser.add_argument("--max-restarts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    arguments = _parse_arguments()
    monitor_watchdog(
        unit=arguments.unit,
        monitor_heartbeat=arguments.monitor_heartbeat,
        output=arguments.output,
        poll_seconds=arguments.poll_seconds,
        stale_seconds=arguments.stale_seconds,
        max_restarts=arguments.max_restarts,
    )


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="monitor_flex_pipeline_supervisor",
        clean_sigterm=True,
        enable_ram_guard=False,
    )
