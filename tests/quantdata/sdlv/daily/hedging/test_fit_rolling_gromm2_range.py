from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.runners import fit_rolling_gromm2_range
from quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2_range import (
    RollingGromm2RangeStage,
    load_completed_rolling_gromm2_range,
    run_rolling_gromm2_range,
)


class _Panel:
    def __init__(self, _directory: Path) -> None:
        start = date(2023, 1, 2)
        self.dates = tuple(start + timedelta(days=index) for index in range(3))

    def prior_one_day_origins(self, _origin: date, *, count: int) -> tuple[date, ...]:
        return tuple(date(2022, 1, 1) + timedelta(days=index) for index in range(count))

    def one_day_transition(self, _origin: date) -> object:
        return object()


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text('{"public": true}\n', encoding="utf-8")
    return dataset


def _dated_manifest(root: Path, as_of_date: date, *, label: str) -> Path:
    path = root / as_of_date.isoformat() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": label}), encoding="utf-8")
    return path


def test_fit_range_reuses_authenticated_origins_and_resumes_missing_ones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    artifact_root = tmp_path / "artifacts"
    state_file = tmp_path / "state.json"
    origins = _Panel(dataset).dates
    existing = _dated_manifest(artifact_root, origins[0], label="existing")
    fitted: list[date] = []

    def load(_root: Path, *, as_of_date: date, **_kwargs: object) -> object:
        if as_of_date != origins[0]:
            raise FileNotFoundError(as_of_date)
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=existing),
            physical_accepted=True,
        )

    def fit(*, as_of_date: date, **_kwargs: object) -> object:
        fitted.append(as_of_date)
        manifest = _dated_manifest(artifact_root, as_of_date, label="fitted")
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=False,
        )

    monkeypatch.setattr(fit_rolling_gromm2_range, "PublicHedgingPanel", _Panel)
    monkeypatch.setattr(fit_rolling_gromm2_range, "load_rolling_gromm2", load)
    monkeypatch.setattr(fit_rolling_gromm2_range, "fit_rolling_gromm2_market", fit)

    summary = run_rolling_gromm2_range(
        dataset=dataset,
        artifact_root=artifact_root,
        evaluation_start=origins[0],
        evaluation_end=origins[-1],
        stage=RollingGromm2RangeStage.FIT,
        state_file=state_file,
        progress_seconds=300.0,
    )

    assert fitted == list(origins[1:])
    assert summary.status == "complete"
    assert summary.physical_accepted_count == 1
    assert [record.reused for record in summary.records] == [True, False, False]
    loaded = load_completed_rolling_gromm2_range(
        state_file,
        dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
        artifact_root=artifact_root,
    )
    assert loaded.to_dict() == summary.to_dict()


def test_fit_range_recovers_after_mid_range_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    artifact_root = tmp_path / "artifacts"
    state_file = tmp_path / "state.json"
    origins = _Panel(dataset).dates
    fit_attempts: list[date] = []
    fail_second_origin = True

    def load(_root: Path, *, as_of_date: date, **_kwargs: object) -> object:
        manifest = artifact_root / as_of_date.isoformat() / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(as_of_date)
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=False,
        )

    def fit(*, as_of_date: date, **_kwargs: object) -> object:
        nonlocal fail_second_origin
        fit_attempts.append(as_of_date)
        if as_of_date == origins[1] and fail_second_origin:
            fail_second_origin = False
            raise RuntimeError("forced restart test")
        manifest = _dated_manifest(artifact_root, as_of_date, label="fitted")
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=False,
        )

    monkeypatch.setattr(fit_rolling_gromm2_range, "PublicHedgingPanel", _Panel)
    monkeypatch.setattr(fit_rolling_gromm2_range, "load_rolling_gromm2", load)
    monkeypatch.setattr(fit_rolling_gromm2_range, "fit_rolling_gromm2_market", fit)

    with pytest.raises(RuntimeError, match="forced restart test"):
        run_rolling_gromm2_range(
            dataset=dataset,
            artifact_root=artifact_root,
            evaluation_start=origins[0],
            evaluation_end=origins[-1],
            stage=RollingGromm2RangeStage.FIT,
            state_file=state_file,
            progress_seconds=300.0,
        )

    failed = json.loads(state_file.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["completed_count"] == 1
    assert failed["current_origin"] == origins[1].isoformat()

    resumed = run_rolling_gromm2_range(
        dataset=dataset,
        artifact_root=artifact_root,
        evaluation_start=origins[0],
        evaluation_end=origins[-1],
        stage=RollingGromm2RangeStage.FIT,
        state_file=state_file,
        progress_seconds=300.0,
    )

    assert resumed.status == "complete"
    assert [record.reused for record in resumed.records] == [True, False, False]
    assert fit_attempts == [origins[0], origins[1], origins[1], origins[2]]


def test_prepare_range_revalidates_output_after_material_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    artifact_root = tmp_path / "artifacts"
    state_file = tmp_path / "state.json"
    origin = _Panel(dataset).dates[0]
    prepared: list[date] = []
    load_attempts: dict[date, int] = {}

    def load(_root: Path, *, as_of_date: date, **_kwargs: object) -> object:
        load_attempts[as_of_date] = load_attempts.get(as_of_date, 0) + 1
        manifest = artifact_root / as_of_date.isoformat() / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(as_of_date)
        return SimpleNamespace(manifest=manifest)

    def prepare(*, as_of_date: date, **_kwargs: object) -> object:
        prepared.append(as_of_date)
        manifest = _dated_manifest(artifact_root, as_of_date, label="prepared")
        return SimpleNamespace(manifest=manifest)

    monkeypatch.setattr(fit_rolling_gromm2_range, "PublicHedgingPanel", _Panel)
    monkeypatch.setattr(
        fit_rolling_gromm2_range,
        "load_prepared_rolling_gromm2",
        load,
    )
    monkeypatch.setattr(fit_rolling_gromm2_range, "prepare_rolling_gromm2", prepare)

    summary = run_rolling_gromm2_range(
        dataset=dataset,
        artifact_root=artifact_root,
        evaluation_start=origin,
        evaluation_end=_Panel(dataset).dates[-1],
        stage=RollingGromm2RangeStage.PREPARE,
        state_file=state_file,
        stop_after=1,
        progress_seconds=300.0,
    )

    assert prepared == [origin]
    assert load_attempts[origin] == 2
    assert summary.status == "partial"
    assert summary.records[0].physical_accepted is None


def test_completed_range_loader_rejects_changed_dated_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    artifact_root = tmp_path / "artifacts"
    state_file = tmp_path / "state.json"
    origins = _Panel(dataset).dates

    def load(_root: Path, *, as_of_date: date, **_kwargs: object) -> object:
        manifest = _dated_manifest(artifact_root, as_of_date, label="complete")
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=True,
        )

    monkeypatch.setattr(fit_rolling_gromm2_range, "PublicHedgingPanel", _Panel)
    monkeypatch.setattr(fit_rolling_gromm2_range, "load_rolling_gromm2", load)
    run_rolling_gromm2_range(
        dataset=dataset,
        artifact_root=artifact_root,
        evaluation_start=origins[0],
        evaluation_end=origins[-1],
        stage=RollingGromm2RangeStage.FIT,
        state_file=state_file,
        progress_seconds=300.0,
    )
    changed = artifact_root / origins[1].isoformat() / "manifest.json"
    changed.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        load_completed_rolling_gromm2_range(
            state_file,
            dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
            artifact_root=artifact_root,
        )
