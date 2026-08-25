"""Focused tests for the persistent FLEX service monitor."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.monitor_services import (
    WorkerSpec,
    _cpu_usage_ns,
    _quantdata_resource_status,
    parse_worker_spec,
)


def test_parse_worker_spec() -> None:
    worker = parse_worker_spec("flex-run.service::/media/storage/run/complete.json")
    assert worker == WorkerSpec(
        unit="flex-run.service",
        completion_path=Path("/media/storage/run/complete.json"),
    )


@pytest.mark.parametrize(
    "value",
    ["flex-run.service", "bad/unit.service::/tmp/done", "run::relative"],
)
def test_parse_worker_spec_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_worker_spec(value)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("123", 123), ("0", 0), ("[not set]", None), ("-1", None), (None, None)],
)
def test_cpu_usage_ns(raw_value: str | None, expected: int | None) -> None:
    properties = {} if raw_value is None else {"CPUUsageNSec": raw_value}
    assert _cpu_usage_ns(properties) == expected


def test_resource_status_uses_quantdata_locations() -> None:
    resources, _, _ = _quantdata_resource_status()
    persistent = resources["persistent_locations"]

    assert set(persistent) == {"raw", "processing", "training", "experiments"}
    for name, payload in persistent.items():
        assert Path(payload["path"]) == Path(getattr(quantdata_env.locations, name).path)
    temporary = resources["temporary_location"]
    assert Path(temporary["path"]) == Path(quantdata_env.locations.temp.path)
