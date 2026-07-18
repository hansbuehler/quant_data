from __future__ import annotations

import importlib
import tempfile
import uuid
from pathlib import Path

import pytest
from cdxcore.subdir import CacheMustExistError, SubDir

import openqai.core


def test_environment_defaults_use_root_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("QUANTDATA_TEST_MODE", raising=False)
    monkeypatch.delenv("QUANTDATA_CACHE_MODE", raising=False)
    monkeypatch.delenv("QUANTDATA_RAW_DATA", raising=False)
    monkeypatch.delenv("QUANTDATA_PROCESSING_DATA", raising=False)
    monkeypatch.delenv("QUANTDATA_TRAINING_DATA", raising=False)
    monkeypatch.delenv("QUANTDATA_TEMP_DATA", raising=False)
    monkeypatch.setenv("QUANTDATA_ROOT", str(tmp_path / "quantdata_root"))

    env = openqai.core.Environment()

    assert env.test_mode is False
    assert Path(env.locations.raw.path) == tmp_path / "quantdata_root" / "raw"
    assert Path(env.locations.processing.path) == tmp_path / "quantdata_root" / "processing"
    assert Path(env.locations.training.path) == tmp_path / "quantdata_root" / "training"
    assert Path(env.locations.temp.path) == Path(tempfile.gettempdir()) / "openqai.core"


def test_environment_test_mode_uses_stored_data(monkeypatch) -> None:
    monkeypatch.setenv("QUANTDATA_TEST_MODE", "1")
    monkeypatch.delenv("QUANTDATA_CACHE_MODE", raising=False)
    monkeypatch.delenv("QUANTDATA_RAW_DATA", raising=False)
    monkeypatch.delenv("QUANTDATA_PROCESSING_DATA", raising=False)
    monkeypatch.delenv("QUANTDATA_TRAINING_DATA", raising=False)
    monkeypatch.delenv("QUANTDATA_TEMP_DATA", raising=False)

    env = openqai.core.Environment()

    assert env.test_mode is True
    assert Path(env.locations.raw.path).name == "raw"
    assert Path(env.locations.processing.path).name == "processing"
    assert Path(env.locations.training.path).name == "training"
    assert Path(env.locations.temp.path).name == "quantdata_test_temp"


def test_test_mode_cached_function_does_not_create_files(monkeypatch) -> None:
    monkeypatch.setenv("QUANTDATA_TEST_MODE", "1")
    env = openqai.core.Environment()

    raw_root = Path(env.locations.raw.path)
    before = {path for path in raw_root.rglob("*") if path.is_file()}
    cache_uid = f"no_write_{uuid.uuid4().hex}"

    @env.locations.raw.cache(
        "0.1",
        uid=lambda: cache_uid,
        exclude_args=[],
    )
    def _cached_value() -> dict[str, int]:
        return {"value": 1}

    with pytest.raises(CacheMustExistError):
        _cached_value()

    after = {path for path in raw_root.rglob("*") if path.is_file()}
    assert after == before