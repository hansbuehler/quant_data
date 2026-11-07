"""Shared forecasts preserve exact arrays and do not duplicate completed simulations."""

import errno
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from filelock import FileLock

from quantdata.hedge2 import terminal_cache
from quantdata.hedge2.terminal_cache import ForecastCheckpointStop, saved_terminal_forecast
from quantdata.hedge2.terminal_market import TerminalSimulationConfig, generate_terminal_paths
from tests.quantdata.hedge2.test_terminal_market import inputs


@pytest.fixture
def forecast():
    return generate_terminal_paths(**inputs(), config=TerminalSimulationConfig(paths=4))


def assert_identical(left, right) -> None:
    for field in fields(left):
        np.testing.assert_array_equal(getattr(left, field.name), getattr(right, field.name))


def test_identical_forecasts_share_arrays_across_objective_directories(tmp_path: Path, forecast) -> None:
    calls = []

    def generate():
        calls.append(True)
        return forecast

    outputs = [tmp_path / name for name in ("mv1", "mv10", "power10", "long")]
    for directory in outputs:
        result = saved_terminal_forecast(directory, identity={"date": "2024-04-19", "claim": "digital"},
            generate=generate, cache_directory=tmp_path / "shared")
        assert_identical(result, forecast)
    assert calls == [True]
    assert len({(directory / "forecast.npz").stat().st_ino for directory in outputs}) == 1
    assert json.loads((outputs[0] / "forecast.json").read_text())["storage_mode"] == "hardlink"


def test_changed_market_inputs_or_claim_state_use_different_cache_entries(tmp_path: Path, forecast) -> None:
    calls = []
    for index, identity in enumerate(({"spot": 100., "reset": None}, {"spot": 101., "reset": None},
                                      {"spot": 101., "reset": 99.})):
        saved_terminal_forecast(tmp_path / str(index), identity=identity,
            generate=lambda: calls.append(True) or forecast, cache_directory=tmp_path / "shared")
    assert len(calls) == 3
    assert len(tuple((tmp_path / "shared").glob("*/forecast.json"))) == 3


def test_local_forecast_survives_removal_of_its_shared_cache_without_regeneration(tmp_path: Path, forecast) -> None:
    local = tmp_path / "local"
    saved_terminal_forecast(local, identity={"seed": 1}, generate=lambda: forecast, cache_directory=tmp_path / "shared")
    record = json.loads((local / "forecast.json").read_text())
    shared = Path(record["shared_cache_entry"])
    (shared / "forecast.npz").unlink()
    (shared / "forecast.json").unlink()
    result = saved_terminal_forecast(local, identity={"seed": 1}, generate=lambda: pytest.fail("regenerated"),
        cache_directory=tmp_path / "shared")
    assert_identical(result, forecast)


def test_existing_local_scope_cannot_silently_change(tmp_path: Path, forecast) -> None:
    saved_terminal_forecast(tmp_path / "local", identity={"seed": 1}, generate=lambda: forecast)
    with pytest.raises(ValueError, match="inputs changed"):
        saved_terminal_forecast(tmp_path / "local", identity={"seed": 2}, generate=lambda: forecast)


def test_corrupted_shared_forecast_is_not_regenerated(tmp_path: Path, forecast) -> None:
    first = tmp_path / "first"
    saved_terminal_forecast(first, identity={"seed": 1}, generate=lambda: forecast, cache_directory=tmp_path / "shared")
    record = json.loads((first / "forecast.json").read_text())
    shared = Path(record["shared_cache_entry"]) / "forecast.npz"
    shared.write_bytes(shared.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after completion"):
        saved_terminal_forecast(tmp_path / "second", identity={"seed": 1}, generate=lambda: pytest.fail("regenerated"),
            cache_directory=tmp_path / "shared")


def test_concurrent_requests_generate_exactly_once(tmp_path: Path, forecast) -> None:
    calls, mutex = [], threading.Lock()

    def generate():
        with mutex:
            calls.append(True)
        time.sleep(.15)
        return forecast

    def request(index):
        return saved_terminal_forecast(tmp_path / str(index), identity={"seed": 1}, generate=generate,
            cache_directory=tmp_path / "shared")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, range(2)))
    assert calls == [True]
    assert_identical(results[0], results[1])


def test_cross_filesystem_fallback_is_explicit_and_byte_identical(monkeypatch, tmp_path: Path, forecast) -> None:
    def different_filesystem(*args, **kwargs):
        raise OSError(errno.EXDEV, "different filesystems")

    monkeypatch.setattr(terminal_cache.os, "link", different_filesystem)
    local = tmp_path / "local"
    result = saved_terminal_forecast(local, identity={"seed": 1}, generate=lambda: forecast,
        cache_directory=tmp_path / "shared")
    assert_identical(result, forecast)
    record = json.loads((local / "forecast.json").read_text())
    assert record["storage_mode"] == "copy_cross_filesystem"
    assert (local / "forecast.npz").read_bytes() == (Path(record["shared_cache_entry"]) / "forecast.npz").read_bytes()


def test_waiting_for_a_writer_can_checkpoint_without_regenerating(tmp_path: Path, forecast) -> None:
    lock = FileLock(str(tmp_path / "forecast.lock"))
    messages = []
    with lock, pytest.raises(ForecastCheckpointStop):
        saved_terminal_forecast(tmp_path, identity={"seed": 1}, generate=lambda: pytest.fail("generated"),
            progress=messages.append, stop_requested=lambda: bool(messages))
    assert messages and "completed=0/1" in messages[0]
    assert not (tmp_path / "forecast.npz").exists()


def test_cache_outputs_cannot_be_written_inside_git(tmp_path: Path, forecast) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    (repository / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(ValueError, match="outside Git"):
        saved_terminal_forecast(tmp_path / "local", identity={"seed": 1}, generate=lambda: forecast,
            cache_directory=repository / "cache")
