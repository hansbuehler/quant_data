from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.runners import fit_rolling_gromm2_range
from quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2_range import (
    LEGACY_ROLLING_GROMM2_RANGE_SCHEMA,
    LegacyRollingGromm2RangeEvidence,
    RollingGromm2RangeStage,
    load_completed_rolling_gromm2_range,
    load_legacy_rolling_gromm2_range,
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
    path.write_text(
        json.dumps({"label": label, "source_identity": "c" * 64}),
        encoding="utf-8",
    )
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
            candidate_physical_accepted=True,
            physical_fallback_used=False,
        )

    def fit(*, as_of_date: date, **_kwargs: object) -> object:
        fitted.append(as_of_date)
        manifest = _dated_manifest(artifact_root, as_of_date, label="fitted")
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=False,
            candidate_physical_accepted=False,
            physical_fallback_used=False,
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
    assert summary.candidate_physical_accepted_count == 1
    assert summary.physical_fallback_count == 0
    assert [record.reused for record in summary.records] == [True, False, False]
    records_by_date = {record.as_of_date: record for record in summary.records}

    def authenticated_load(
        _root: Path,
        *,
        as_of_date: date,
        **_kwargs: object,
    ) -> object:
        record = records_by_date[as_of_date]
        return SimpleNamespace(
            paths=SimpleNamespace(
                manifest=artifact_root / as_of_date.isoformat() / "manifest.json"
            ),
            physical_accepted=record.physical_accepted,
            candidate_physical_accepted=record.candidate_physical_accepted,
            physical_fallback_used=record.physical_fallback_used,
        )

    monkeypatch.setattr(
        fit_rolling_gromm2_range,
        "load_rolling_gromm2",
        authenticated_load,
    )
    loaded = load_completed_rolling_gromm2_range(
        state_file,
        dataset=dataset,
        dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
        artifact_root=artifact_root,
    )
    assert loaded.to_dict() == summary.to_dict()

    original_payload = state_file.read_text(encoding="utf-8")
    truncated = json.loads(original_payload)
    truncated["records"] = truncated["records"][:-1]
    truncated["eligible_origin_count"] -= 1
    truncated["selected_origin_count"] -= 1
    truncated["completed_count"] -= 1
    truncated["physical_accepted_count"] = sum(
        value["physical_accepted"] for value in truncated["records"]
    )
    truncated["candidate_physical_accepted_count"] = sum(
        value["candidate_physical_accepted"] for value in truncated["records"]
    )
    truncated["physical_fallback_count"] = sum(
        value["physical_fallback_used"] for value in truncated["records"]
    )
    state_file.write_text(json.dumps(truncated), encoding="utf-8")
    with pytest.raises(ValueError, match="eligible cohort"):
        load_completed_rolling_gromm2_range(
            state_file,
            dataset=dataset,
            dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
            artifact_root=artifact_root,
        )
    state_file.write_text(original_payload, encoding="utf-8")

    def mismatched_load(
        _root: Path,
        *,
        as_of_date: date,
        **_kwargs: object,
    ) -> object:
        return SimpleNamespace(
            paths=SimpleNamespace(
                manifest=artifact_root / as_of_date.isoformat() / "manifest.json"
            ),
            physical_accepted=True,
            candidate_physical_accepted=False,
            physical_fallback_used=True,
        )

    monkeypatch.setattr(
        fit_rolling_gromm2_range,
        "load_rolling_gromm2",
        mismatched_load,
    )
    with pytest.raises(ValueError, match="disagrees with range record"):
        load_completed_rolling_gromm2_range(
            state_file,
            dataset=dataset,
            dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
            artifact_root=artifact_root,
        )


def test_legacy_range_loader_authenticates_complete_manifest_set(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    artifact_root = tmp_path / "legacy"
    origins = _Panel(dataset).dates
    records = []
    for origin in origins:
        manifest = _dated_manifest(artifact_root, origin, label="legacy")
        records.append(
            {
                "as_of_date": origin.isoformat(),
                "reused": True,
                "manifest_sha256": file_sha256(manifest),
                "physical_accepted": False,
            }
        )
    config = fit_rolling_gromm2_range.RollingGromm2Config()
    legacy_config = json.loads(json.dumps(asdict(config), sort_keys=True))
    legacy_config["minimum_physical_projection_fraction"] = 0.5
    summary = tmp_path / "legacy-range.json"
    summary.write_text(
        json.dumps(
            {
                "schema": LEGACY_ROLLING_GROMM2_RANGE_SCHEMA,
                "status": "complete",
                "stage": RollingGromm2RangeStage.FIT.value,
                "dataset_manifest_sha256": file_sha256(
                    dataset / "manifest.json"
                ),
                "artifact_root": str(artifact_root.resolve()),
                "eligible_origin_count": len(origins),
                "selected_origin_count": len(origins),
                "completed_count": len(origins),
                "source_identity": "a" * 64,
                "config": legacy_config,
                "records": records,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_legacy_rolling_gromm2_range(
        summary,
        dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
        artifact_root=artifact_root,
    )

    assert loaded.source_identity == "a" * 64
    assert loaded.manifest_source_identity == "c" * 64
    assert tuple(loaded.manifest_sha256_by_date) == origins
    (artifact_root / origins[1].isoformat() / "manifest.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid"):
        load_legacy_rolling_gromm2_range(
            summary,
            dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
            artifact_root=artifact_root,
        )


def test_fit_range_uses_authenticated_legacy_affine_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    artifact_root = tmp_path / "current"
    origin = _Panel(dataset).dates[0]
    migrated: list[date] = []
    migrated_source_identities: list[str] = []

    def load(_root: Path, *, as_of_date: date, **_kwargs: object) -> object:
        raise FileNotFoundError(as_of_date)

    def migrate(
        *,
        as_of_date: date,
        source_identity: str,
        **_kwargs: object,
    ) -> object:
        migrated.append(as_of_date)
        migrated_source_identities.append(source_identity)
        return object()

    def fit(*, as_of_date: date, **_kwargs: object) -> object:
        manifest = _dated_manifest(artifact_root, as_of_date, label="migrated")
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=True,
            candidate_physical_accepted=False,
            physical_fallback_used=True,
        )

    monkeypatch.setattr(fit_rolling_gromm2_range, "PublicHedgingPanel", _Panel)
    monkeypatch.setattr(fit_rolling_gromm2_range, "load_rolling_gromm2", load)
    monkeypatch.setattr(
        fit_rolling_gromm2_range,
        "migrate_legacy_rolling_gromm2_affine_fit",
        migrate,
    )
    monkeypatch.setattr(fit_rolling_gromm2_range, "fit_rolling_gromm2_market", fit)
    legacy = LegacyRollingGromm2RangeEvidence(
        artifact_root=tmp_path / "legacy",
        source_identity="a" * 64,
        manifest_source_identity="c" * 64,
        manifest_sha256_by_date={origin: "b" * 64},
    )

    summary = run_rolling_gromm2_range(
        dataset=dataset,
        artifact_root=artifact_root,
        evaluation_start=origin,
        evaluation_end=_Panel(dataset).dates[-1],
        stage=RollingGromm2RangeStage.FIT,
        state_file=tmp_path / "state.json",
        legacy=legacy,
        stop_after=1,
        progress_seconds=300.0,
    )

    assert migrated == [origin]
    assert migrated_source_identities == ["c" * 64]
    assert summary.physical_fallback_count == 1


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
            physical_accepted=True,
            candidate_physical_accepted=False,
            physical_fallback_used=True,
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
            physical_accepted=True,
            candidate_physical_accepted=False,
            physical_fallback_used=True,
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
    assert resumed.candidate_physical_accepted_count == 0
    assert resumed.physical_fallback_count == 3
    assert [record.reused for record in resumed.records] == [True, False, False]
    assert fit_attempts == [origins[0], origins[1], origins[1], origins[2]]


def test_overlapping_ranges_serialize_each_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    artifact_root = tmp_path / "artifacts"
    origin = _Panel(dataset).dates[0]
    fit_started = Event()
    release_fit = Event()
    attempts_lock = Lock()
    fit_attempts: list[date] = []

    def load(_root: Path, *, as_of_date: date, **_kwargs: object) -> object:
        manifest = artifact_root / as_of_date.isoformat() / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(as_of_date)
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=True,
            candidate_physical_accepted=False,
            physical_fallback_used=True,
        )

    def fit(*, as_of_date: date, **_kwargs: object) -> object:
        with attempts_lock:
            fit_attempts.append(as_of_date)
        fit_started.set()
        if not release_fit.wait(timeout=5.0):
            raise TimeoutError("test did not release first fit")
        manifest = _dated_manifest(artifact_root, as_of_date, label="fitted")
        return SimpleNamespace(
            paths=SimpleNamespace(manifest=manifest),
            physical_accepted=True,
            candidate_physical_accepted=False,
            physical_fallback_used=True,
        )

    monkeypatch.setattr(fit_rolling_gromm2_range, "PublicHedgingPanel", _Panel)
    monkeypatch.setattr(fit_rolling_gromm2_range, "load_rolling_gromm2", load)
    monkeypatch.setattr(
        fit_rolling_gromm2_range,
        "fit_rolling_gromm2_market",
        fit,
    )

    def run(state_name: str):
        return run_rolling_gromm2_range(
            dataset=dataset,
            artifact_root=artifact_root,
            evaluation_start=origin,
            evaluation_end=_Panel(dataset).dates[-1],
            stage=RollingGromm2RangeStage.FIT,
            state_file=tmp_path / state_name,
            stop_after=1,
            progress_seconds=300.0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run, "first.json")
        assert fit_started.wait(timeout=5.0)
        second = executor.submit(run, "second.json")
        time.sleep(0.1)
        assert fit_attempts == [origin]
        release_fit.set()
        summaries = (first.result(timeout=5.0), second.result(timeout=5.0))

    assert fit_attempts == [origin]
    assert sorted(summary.records[0].reused for summary in summaries) == [False, True]


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
            candidate_physical_accepted=True,
            physical_fallback_used=False,
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
            dataset=dataset,
            dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
            artifact_root=artifact_root,
        )


def test_completed_range_loader_rejects_changed_source_identity(
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
            candidate_physical_accepted=False,
            physical_fallback_used=True,
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
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    payload["source_identity"] = "0" * 64
    state_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stale or incomplete"):
        load_completed_rolling_gromm2_range(
            state_file,
            dataset=dataset,
            dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
            artifact_root=artifact_root,
        )
