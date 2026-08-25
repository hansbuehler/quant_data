"""Tests for the independent FLEX pipeline-monitor heartbeat watchdog."""

from __future__ import annotations

import json

from quantdata.dysanos.flex.pipeline_monitor_watchdog import (
    WatchdogDecision,
    _load_state,
    classify_monitor_health,
)


def test_watchdog_accepts_only_active_fresh_monitor() -> None:
    assert classify_monitor_health(
        active=True,
        heartbeat_age_seconds=20.0,
        heartbeat_valid=True,
        monitor_status="running",
        restart_count=0,
        max_restarts=3,
        stale_seconds=300,
    ) == WatchdogDecision(state="RUNNING", reason=None, restart=False)


def test_watchdog_restarts_stale_monitor_then_fails_terminally() -> None:
    suspect = classify_monitor_health(
        active=True,
        heartbeat_age_seconds=301.0,
        heartbeat_valid=True,
        monitor_status="running",
        restart_count=2,
        max_restarts=3,
        stale_seconds=300,
    )
    assert suspect.state == "SUSPECT"
    assert suspect.restart is False
    assert suspect.consecutive_unhealthy_observations == 1
    retry = classify_monitor_health(
        active=True,
        heartbeat_age_seconds=301.0,
        heartbeat_valid=True,
        monitor_status="running",
        restart_count=2,
        max_restarts=3,
        stale_seconds=300,
        consecutive_unhealthy_observations=1,
    )
    assert retry.state == "RESTART_PENDING"
    assert retry.restart is True
    terminal = classify_monitor_health(
        active=False,
        heartbeat_age_seconds=None,
        heartbeat_valid=False,
        monitor_status=None,
        restart_count=3,
        max_restarts=3,
        stale_seconds=300,
        consecutive_unhealthy_observations=1,
    )
    assert terminal.state == "FAILED_TERMINAL"
    assert terminal.restart is False


def test_watchdog_accepts_completed_monitor_as_successful_terminal_state() -> None:
    completed = classify_monitor_health(
        active=False,
        heartbeat_age_seconds=900.0,
        heartbeat_valid=True,
        monitor_status="complete",
        restart_count=3,
        max_restarts=3,
        stale_seconds=300,
    )
    assert completed == WatchdogDecision(
        state="SUCCEEDED",
        reason="monitor certified pipeline completion",
        restart=False,
    )


def test_watchdog_does_not_restart_terminal_monitor() -> None:
    terminal = classify_monitor_health(
        active=False,
        heartbeat_age_seconds=10.0,
        heartbeat_valid=True,
        monitor_status="terminal_failure",
        restart_count=0,
        max_restarts=3,
        stale_seconds=300,
    )
    assert terminal == WatchdogDecision(
        state="FAILED_TERMINAL",
        reason="monitor certified terminal pipeline failure",
        restart=False,
    )


def test_watchdog_persists_consecutive_unhealthy_observations(tmp_path) -> None:
    state = tmp_path / "watchdog.json"
    state.write_text(
        json.dumps(
            {
                "schema": "flex-pipeline-monitor-watchdog-v1",
                "restart_count": 1,
                "state": "SUSPECT",
                "monitor_invocation_id": "abc",
                "consecutive_unhealthy_observations": 1,
            }
        ),
        encoding="utf-8",
    )
    assert _load_state(state) == (1, "SUSPECT", "abc", 1)
