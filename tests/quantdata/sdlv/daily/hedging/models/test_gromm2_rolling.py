from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest

from quantdata.gromm2.market import PublicSanosDay
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_calibration import (
    project_constant_physical_drift,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.gromm import (
    rolling_gromm2 as rolling_gromm2_module,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_dynamics import (
    GROMM2_DYNAMICS_SCHEMA,
    PHYSICAL_SELECTION_RULE,
    _forward_over_spot,
    _select_physical_drift,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    LEGACY_PHYSICAL_PROJECTION_FRACTION,
    LEGACY_ROLLING_GROMM2_SCHEMA,
    ROLLING_GROMM2_SCHEMA,
    RollingGromm2Config,
    _fit_affine_market_until_gate,
    _invalidate_fitted_outputs,
    load_prepared_rolling_gromm2,
    load_rolling_gromm2,
    migrate_legacy_rolling_gromm2_affine_fit,
    rolling_gromm2_paths,
    rolling_gromm2_source_identity,
    select_rolling_public_days,
)


def _public_day(trading_date: date) -> PublicSanosDay:
    return PublicSanosDay(
        trading_date=trading_date,
        model_expiries_y=np.asarray([1.0 / 252.0]),
        kernel_total_variances=np.asarray([0.2**2 / 252.0]),
        support_strikes=(np.asarray([0.8, 1.0, 1.2]),),
        densities=(np.asarray([0.1, 0.8, 0.1]),),
        smoothness=1.0,
        one_minute_y=1.0 / (252.0 * 390.0),
    )


def _write_physical_selection_fixture(
    path,
    *,
    candidate: np.ndarray | None = None,
    candidate_accepted: bool = True,
    candidate_fraction: float = 0.1,
) -> None:
    candidate_values = (
        np.zeros(16, dtype=np.float64)
        if candidate is None
        else np.asarray(candidate, dtype=np.float64)
    )
    baseline = np.zeros(16, dtype=np.float64)
    baseline[0] = candidate_values[0]
    selected = candidate_values if candidate_accepted else baseline
    candidate_standard_errors = np.zeros(16, dtype=np.float64)
    baseline_fraction = 0.0
    selected_fraction = (
        candidate_fraction if candidate_accepted else baseline_fraction
    )
    candidate_norm = float(np.linalg.norm(candidate_values) / np.sqrt(252.0))
    baseline_norm = float(np.linalg.norm(baseline) / np.sqrt(252.0))
    selected_norm = float(np.linalg.norm(selected) / np.sqrt(252.0))
    reasons = [] if candidate_accepted else ["validation-improvement"]
    report = {
        "schema": GROMM2_DYNAMICS_SCHEMA,
        "factor_count": 16,
        "pair_count": 252,
        "explained_variance_ratio": 0.91,
        "physical_selection_rule": PHYSICAL_SELECTION_RULE,
        "physical_projection_fraction": selected_fraction,
        "candidate_achievable_fraction": candidate_fraction,
        "baseline_achievable_fraction": baseline_fraction,
        "selected_achievable_fraction": selected_fraction,
        "candidate_market_price_of_risk": candidate_values.tolist(),
        "candidate_market_price_of_risk_standard_errors": (
            candidate_standard_errors.tolist()
        ),
        "baseline_market_price_of_risk": baseline.tolist(),
        "selected_market_price_of_risk": selected.tolist(),
        "full_window_front_market_price_of_risk": float(baseline[0]),
        "full_window_front_market_price_of_risk_standard_error": 0.0,
        "physical_validation_improvement_fraction": (
            0.01 if candidate_accepted else 0.001
        ),
        "physical_validation_gain_t": 2.5,
        "physical_validation_accepted": candidate_accepted,
        "candidate_daily_risk_price_norm": candidate_norm,
        "baseline_daily_risk_price_norm": baseline_norm,
        "selected_daily_risk_price_norm": selected_norm,
        "physical_daily_risk_price_norm": selected_norm,
        "candidate_accepted": candidate_accepted,
        "candidate_rejection_reasons": reasons,
        "baseline_accepted": True,
        "baseline_rejection_reasons": [],
        "selected_model": "candidate" if candidate_accepted else "front-only-null",
        "selected_candidate": candidate_accepted,
        "fallback_used": not candidate_accepted,
        "selected_physical_accepted": True,
        "physical_accepted": True,
    }
    np.savez_compressed(
        path,
        physical_selection_rule=np.asarray(PHYSICAL_SELECTION_RULE, dtype="S32"),
        selected_model=np.asarray(report["selected_model"], dtype="S16"),
        physical_market_price_of_risk=selected,
        physical_market_price_of_risk_standard_errors=(
            candidate_standard_errors
            if candidate_accepted
            else np.zeros(16, dtype=np.float64)
        ),
        candidate_market_price_of_risk=candidate_values,
        candidate_market_price_of_risk_standard_errors=candidate_standard_errors,
        baseline_market_price_of_risk=baseline,
        candidate_accepted=np.asarray(candidate_accepted),
        baseline_accepted=np.asarray(True),
        selected_candidate=np.asarray(candidate_accepted),
        fallback_used=np.asarray(not candidate_accepted),
        selected_physical_accepted=np.asarray(True),
        physical_accepted=np.asarray(True),
    )
    path.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")


def test_rolling_gromm2_selection_ends_at_as_of_without_future_dates() -> None:
    days = [_public_day(date(2023, 1, 2) + timedelta(days=index)) for index in range(7)]

    selected = select_rolling_public_days(
        days,
        as_of_date=date(2023, 1, 6),
        window_transitions=2,
    )

    assert [day.trading_date for day in selected] == [
        date(2023, 1, 4),
        date(2023, 1, 5),
        date(2023, 1, 6),
    ]


def test_rolling_gromm2_selection_requires_exact_as_of_state() -> None:
    days = [_public_day(date(2023, 1, 2) + timedelta(days=index)) for index in range(4)]

    with pytest.raises(ValueError, match="end at as-of"):
        select_rolling_public_days(
            days,
            as_of_date=date(2023, 1, 8),
            window_transitions=2,
        )


def test_rolling_gromm2_selection_counts_exact_transition_pairs_across_gap() -> None:
    days = [_public_day(date(2023, 1, 2) + timedelta(days=index)) for index in range(6)]
    pairs = (
        (date(2023, 1, 2), date(2023, 1, 3)),
        (date(2023, 1, 4), date(2023, 1, 5)),
    )

    selected = select_rolling_public_days(
        days,
        as_of_date=date(2023, 1, 5),
        window_transitions=2,
        transition_pairs=pairs,
    )

    assert [day.trading_date for day in selected] == [
        date(2023, 1, 2),
        date(2023, 1, 3),
        date(2023, 1, 4),
        date(2023, 1, 5),
    ]


def test_rolling_gromm2_selection_appends_unpaired_current_state() -> None:
    days = [_public_day(date(2023, 1, 2) + timedelta(days=index)) for index in range(6)]
    pairs = (
        (date(2023, 1, 2), date(2023, 1, 3)),
        (date(2023, 1, 4), date(2023, 1, 5)),
    )

    selected = select_rolling_public_days(
        days,
        as_of_date=date(2023, 1, 6),
        window_transitions=2,
        transition_pairs=pairs,
    )

    assert [day.trading_date for day in selected] == [
        date(2023, 1, 2),
        date(2023, 1, 3),
        date(2023, 1, 4),
        date(2023, 1, 5),
        date(2023, 1, 6),
    ]


def test_rolling_gromm2_forward_uses_flat_released_tail() -> None:
    day = synthetic_one_day_transition().origin
    target = float(day.business_dtes[-1] + 1.0)

    value, extrapolated = _forward_over_spot(
        day,
        target_business_dte=target,
    )

    assert extrapolated
    assert value == pytest.approx(day.forward_over_spot[-1])


def test_physical_selection_uses_front_only_null_after_candidate_rejection() -> None:
    sample_count = 3
    origins = tuple(date(2023, 1, 2 + index) for index in range(sample_count))
    endpoints = tuple(value + timedelta(days=1) for value in origins)
    loadings = np.zeros((sample_count, 1, 2, 2), dtype=np.float64)
    loadings[:, 0, 0, 0] = 1.0
    loadings[:, 0, 1, 1] = 1.0
    risk_price = np.asarray([0.4, 0.8], dtype=np.float64)
    empirical = np.broadcast_to(risk_price, (sample_count, 1, 2)).copy()
    risk_neutral = np.zeros_like(empirical)
    weights = np.ones_like(empirical)
    candidate = project_constant_physical_drift(
        training_origin_dates=origins,
        training_endpoint_dates=endpoints,
        empirical_log_drifts=empirical,
        risk_neutral_log_drifts=risk_neutral,
        coordinate_weights=weights,
        annualized_loadings=loadings,
        fixed_market_price_of_risk=np.asarray([0.4, np.nan]),
    )
    candidate = replace(
        candidate,
        validation_improvement_fraction=0.001,
        validation_gain_t=0.5,
        daily_risk_price_norm=float(np.linalg.norm(risk_price) / np.sqrt(252.0)),
        accepted=False,
    )

    selection = _select_physical_drift(
        candidate=candidate,
        annualized_loadings=loadings,
        fixed_market_price_of_risk=np.asarray([0.4, np.nan]),
        maximum_daily_risk_price_norm=3.0,
    )

    assert not selection.candidate_accepted
    assert selection.baseline_accepted
    assert selection.fallback_used
    assert selection.selected_model == "front-only-null"
    assert np.array_equal(
        selection.selected.market_price_of_risk,
        np.asarray([0.4, 0.0]),
    )

    accepted = replace(
        candidate,
        validation_improvement_fraction=0.01,
        validation_gain_t=2.5,
        accepted=True,
    )
    selected_candidate = _select_physical_drift(
        candidate=accepted,
        annualized_loadings=loadings,
        fixed_market_price_of_risk=np.asarray([0.4, np.nan]),
        maximum_daily_risk_price_norm=3.0,
    )
    assert selected_candidate.selected_model == "candidate"
    assert selected_candidate.selected is accepted


def test_affine_market_fit_continues_until_fixed_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RollingGromm2Config()
    paths = rolling_gromm2_paths(tmp_path, as_of_date=date(2023, 1, 17))
    paths.directory.mkdir(parents=True)
    paths.prepared_books.write_bytes(b"prepared")
    percentages = iter((79.5, 84.0))
    fitted_epochs: list[int] = []

    def fit(*, output, checkpoint, config, **_kwargs) -> None:
        fitted_epochs.append(config.epochs)
        with h5py.File(output, "w") as stream:
            stream.attrs["schema"] = "gromm2-affine-dlv-history-v1"
            stream.attrs["percent_inside_spread"] = next(percentages)
            stream.attrs["calibration_config"] = json.dumps(
                asdict(config), sort_keys=True
            )
        np.savez_compressed(checkpoint, completed_epochs=config.epochs)

    monkeypatch.setattr(
        "quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2.fit_affine_history",
        fit,
    )

    percent, epochs, retries, fitted = _fit_affine_market_until_gate(
        paths=paths,
        calibration=config.calibration,
        minimum_inside_spread_percent=config.minimum_inside_spread_percent,
    )

    assert fitted_epochs == [2_000, 4_000]
    assert percent == pytest.approx(84.0)
    assert epochs == fitted.epochs == 4_000
    assert retries == 1


def test_affine_market_fit_resumes_partial_retry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RollingGromm2Config()
    paths = rolling_gromm2_paths(tmp_path, as_of_date=date(2023, 1, 17))
    paths.directory.mkdir(parents=True)
    paths.prepared_books.write_bytes(b"prepared")
    np.savez_compressed(paths.affine_checkpoint, completed_epochs=3_500)
    fitted_epochs: list[int] = []

    def fit(*, output, checkpoint, config, **_kwargs) -> None:
        fitted_epochs.append(config.epochs)
        with h5py.File(output, "w") as stream:
            stream.attrs["schema"] = "gromm2-affine-dlv-history-v1"
            stream.attrs["percent_inside_spread"] = 84.0
            stream.attrs["calibration_config"] = json.dumps(
                asdict(config), sort_keys=True
            )
        np.savez_compressed(checkpoint, completed_epochs=config.epochs)

    monkeypatch.setattr(
        "quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2.fit_affine_history",
        fit,
    )

    _, epochs, retries, _ = _fit_affine_market_until_gate(
        paths=paths,
        calibration=config.calibration,
        minimum_inside_spread_percent=config.minimum_inside_spread_percent,
    )

    assert fitted_epochs == [4_000]
    assert epochs == 4_000
    assert retries == 1


def test_repreparation_invalidates_all_fitted_outputs(tmp_path) -> None:
    paths = rolling_gromm2_paths(tmp_path, as_of_date=date(2023, 1, 17))
    paths.directory.mkdir(parents=True)
    derived = (
        paths.affine_history,
        paths.affine_checkpoint,
        paths.dynamics,
        paths.dynamics.with_suffix(".json"),
    )
    for path in derived:
        path.write_bytes(b"stale")

    _invalidate_fitted_outputs(paths)

    assert all(not path.exists() for path in derived)


def test_prepared_rolling_gromm2_loader_authenticates_source_files(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RollingGromm2Config()
    start = date(2022, 1, 1)
    origins = [start + timedelta(days=index) for index in range(252)]
    endpoints = [value + timedelta(days=1) for value in origins]
    as_of = endpoints[-1]
    paths = rolling_gromm2_paths(tmp_path, as_of_date=as_of)
    paths.directory.mkdir(parents=True)
    paths.source_history.write_bytes(b"source")
    paths.prepared_books.write_bytes(b"books")
    dataset_manifest_sha256 = "a" * 64
    manifest = {
        "schema": ROLLING_GROMM2_SCHEMA,
        "status": "prepared",
        "as_of_date": as_of.isoformat(),
        "training_start_date": origins[0].isoformat(),
        "training_end_date": as_of.isoformat(),
        "state_count": len(set(origins) | set(endpoints)),
        "transition_count": 252,
        "training_origin_dates": [value.isoformat() for value in origins],
        "training_endpoint_dates": [value.isoformat() for value in endpoints],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_identity": rolling_gromm2_source_identity(),
        "config": json.loads(json.dumps(asdict(config), sort_keys=True)),
        "source_history_sha256": file_sha256(paths.source_history),
        "prepared_books_sha256": file_sha256(paths.prepared_books),
    }
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_prepared_rolling_gromm2(
        tmp_path,
        as_of_date=as_of,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    assert loaded == paths
    with monkeypatch.context() as patch:
        patch.setattr(
            rolling_gromm2_module,
            "rolling_gromm2_source_identity",
            lambda: "b" * 64,
        )
        with pytest.raises(ValueError, match="stale or incompatible"):
            load_prepared_rolling_gromm2(
                tmp_path,
                as_of_date=as_of,
                config=config,
                dataset_manifest_sha256=dataset_manifest_sha256,
            )
    paths.prepared_books.write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared_books_sha256"):
        load_prepared_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )


def test_legacy_affine_migration_authenticates_and_hardlinks_fit(tmp_path) -> None:
    config = RollingGromm2Config()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        '{"public": true}\n', encoding="utf-8"
    )
    source_root = tmp_path / "legacy"
    target_root = tmp_path / "current"
    start = date(2022, 1, 1)
    origins = [start + timedelta(days=index) for index in range(252)]
    endpoints = [value + timedelta(days=1) for value in origins]
    as_of = endpoints[-1]
    state_dates = sorted(set(origins) | set(endpoints) | {as_of})
    source = rolling_gromm2_paths(source_root, as_of_date=as_of)
    source.directory.mkdir(parents=True)
    source.source_history.write_bytes(b"source")
    source.prepared_books.write_bytes(b"books")
    _write_affine_history(source.affine_history, dates=state_dates, config=config)
    source_identity = "a" * 64
    legacy_config = json.loads(json.dumps(asdict(config), sort_keys=True))
    legacy_config["minimum_physical_projection_fraction"] = (
        LEGACY_PHYSICAL_PROJECTION_FRACTION
    )
    source.manifest.write_text(
        json.dumps(
            {
                "schema": LEGACY_ROLLING_GROMM2_SCHEMA,
                "status": "complete",
                "as_of_date": as_of.isoformat(),
                "training_start_date": origins[0].isoformat(),
                "training_end_date": as_of.isoformat(),
                "state_count": len(state_dates),
                "transition_count": len(origins),
                "training_origin_dates": [value.isoformat() for value in origins],
                "training_endpoint_dates": [
                    value.isoformat() for value in endpoints
                ],
                "dataset_manifest_sha256": file_sha256(
                    dataset / "manifest.json"
                ),
                "source_identity": source_identity,
                "config": legacy_config,
                "source_history_sha256": file_sha256(source.source_history),
                "prepared_books_sha256": file_sha256(source.prepared_books),
                "affine_history_sha256": file_sha256(source.affine_history),
                "affine_calibration_config": json.loads(
                    json.dumps(asdict(config.calibration), sort_keys=True)
                ),
                "calibration_epochs_completed": config.calibration.epochs,
                "calibration_retry_count": 0,
                "percent_inside_spread": 83.5,
            }
        ),
        encoding="utf-8",
    )

    target = migrate_legacy_rolling_gromm2_affine_fit(
        dataset=dataset,
        source_artifact_root=source_root,
        target_artifact_root=target_root,
        as_of_date=as_of,
        source_manifest_sha256=file_sha256(source.manifest),
        source_identity=source_identity,
        config=config,
    )

    assert os.stat(source.affine_history).st_ino == os.stat(target.affine_history).st_ino
    manifest = json.loads(target.manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == ROLLING_GROMM2_SCHEMA
    assert manifest["status"] == "prepared"
    assert manifest["legacy_affine_migration"]["source_identity"] == source_identity
    loaded = load_prepared_rolling_gromm2(
        target_root,
        as_of_date=as_of,
        config=config,
        dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
    )
    assert loaded == target

    with pytest.raises(ValueError, match="manifest hash"):
        migrate_legacy_rolling_gromm2_affine_fit(
            dataset=dataset,
            source_artifact_root=source_root,
            target_artifact_root=tmp_path / "invalid",
            as_of_date=as_of,
            source_manifest_sha256="b" * 64,
            source_identity=source_identity,
            config=config,
        )


def test_complete_rolling_gromm2_loader_rejects_future_history(tmp_path) -> None:
    config = RollingGromm2Config()
    as_of = date(2023, 12, 29)
    paths = rolling_gromm2_paths(tmp_path, as_of_date=as_of)
    paths.directory.mkdir(parents=True)
    paths.source_history.write_bytes(b"source")
    paths.prepared_books.write_bytes(b"books")
    _write_physical_selection_fixture(paths.dynamics, candidate_fraction=0.1)
    dataset_manifest_sha256 = "a" * 64
    dates = [
        as_of - timedelta(days=config.window_transitions - index)
        for index in range(config.window_transitions + 1)
    ]
    _write_affine_history(paths.affine_history, dates=dates, config=config)
    manifest = {
        "schema": ROLLING_GROMM2_SCHEMA,
        "status": "complete",
        "as_of_date": as_of.isoformat(),
        "training_start_date": dates[0].isoformat(),
        "training_end_date": as_of.isoformat(),
        "state_count": 253,
        "transition_count": 252,
        "training_origin_dates": [value.isoformat() for value in dates[:-1]],
        "training_endpoint_dates": [value.isoformat() for value in dates[1:]],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_identity": rolling_gromm2_source_identity(),
        "config": json.loads(json.dumps(asdict(config), sort_keys=True)),
        "source_history_sha256": file_sha256(paths.source_history),
        "prepared_books_sha256": file_sha256(paths.prepared_books),
        "affine_history_sha256": file_sha256(paths.affine_history),
        "affine_calibration_config": json.loads(
            json.dumps(asdict(config.calibration), sort_keys=True)
        ),
        "calibration_epochs_completed": config.calibration.epochs,
        "calibration_retry_count": 0,
        "percent_inside_spread": 83.5,
        "dynamics_sha256": file_sha256(paths.dynamics),
        "dynamics_report_sha256": file_sha256(paths.dynamics.with_suffix(".json")),
    }
    _sync_physical_manifest(manifest, paths.dynamics.with_suffix(".json"))
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_rolling_gromm2(
        tmp_path,
        as_of_date=as_of,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    assert loaded.as_of_date == as_of
    assert loaded.training_dates[-1] == as_of
    assert loaded.percent_inside_spread == pytest.approx(83.5)
    assert loaded.dataset_manifest_sha256 == dataset_manifest_sha256
    assert loaded.maximum_daily_risk_price_norm == 0.0
    assert loaded.physical_accepted
    assert loaded.physical_projection_fraction == pytest.approx(0.1)
    assert not loaded.physical_fallback_used

    manifest["physical_accepted"] = 1
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest physical selection"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
    _sync_physical_manifest(manifest, paths.dynamics.with_suffix(".json"))
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stale or incompatible"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256="b" * 64,
        )

    _write_physical_selection_fixture(
        paths.dynamics,
        candidate=np.concatenate(([0.25], np.ones(15))),
        candidate_accepted=False,
    )
    report_path = paths.dynamics.with_suffix(".json")
    manifest["dynamics_sha256"] = file_sha256(paths.dynamics)
    manifest["dynamics_report_sha256"] = file_sha256(report_path)
    _sync_physical_manifest(manifest, report_path)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_rolling_gromm2(
        tmp_path,
        as_of_date=as_of,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    assert loaded.physical_accepted
    assert loaded.physical_fallback_used
    assert loaded.physical_selected_model == "front-only-null"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selected_model"] = "candidate"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["dynamics_report_sha256"] = file_sha256(report_path)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="physical selection is inconsistent"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )

    _write_physical_selection_fixture(paths.dynamics)
    manifest["dynamics_sha256"] = file_sha256(paths.dynamics)
    manifest["dynamics_report_sha256"] = file_sha256(report_path)
    _sync_physical_manifest(manifest, report_path)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    dates[-1] = as_of + timedelta(days=1)
    _write_affine_history(paths.affine_history, dates=dates, config=config)
    manifest["affine_history_sha256"] = file_sha256(paths.affine_history)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dated geometry"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )


def test_complete_rolling_gromm2_loader_accepts_gapped_state_window(
    tmp_path,
) -> None:
    config = RollingGromm2Config()
    start = date(2022, 1, 1)
    origins = [start + timedelta(days=index) for index in range(126)]
    origins.extend(start + timedelta(days=index) for index in range(127, 253))
    endpoints = [value + timedelta(days=1) for value in origins]
    as_of = endpoints[-1] + timedelta(days=2)
    state_dates = sorted(set(origins) | set(endpoints) | {as_of})
    paths = rolling_gromm2_paths(tmp_path, as_of_date=as_of)
    paths.directory.mkdir(parents=True)
    paths.source_history.write_bytes(b"source")
    paths.prepared_books.write_bytes(b"books")
    _write_physical_selection_fixture(paths.dynamics)
    _write_affine_history(paths.affine_history, dates=state_dates, config=config)
    dataset_manifest_sha256 = "a" * 64
    manifest = {
        "schema": ROLLING_GROMM2_SCHEMA,
        "status": "complete",
        "as_of_date": as_of.isoformat(),
        "training_start_date": origins[0].isoformat(),
        "training_end_date": as_of.isoformat(),
        "state_count": len(state_dates),
        "transition_count": 252,
        "training_origin_dates": [value.isoformat() for value in origins],
        "training_endpoint_dates": [value.isoformat() for value in endpoints],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_identity": rolling_gromm2_source_identity(),
        "config": json.loads(json.dumps(asdict(config), sort_keys=True)),
        "source_history_sha256": file_sha256(paths.source_history),
        "prepared_books_sha256": file_sha256(paths.prepared_books),
        "affine_history_sha256": file_sha256(paths.affine_history),
        "affine_calibration_config": json.loads(
            json.dumps(asdict(config.calibration), sort_keys=True)
        ),
        "calibration_epochs_completed": config.calibration.epochs,
        "calibration_retry_count": 0,
        "percent_inside_spread": 83.5,
        "dynamics_sha256": file_sha256(paths.dynamics),
        "dynamics_report_sha256": file_sha256(paths.dynamics.with_suffix(".json")),
    }
    _sync_physical_manifest(manifest, paths.dynamics.with_suffix(".json"))
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_rolling_gromm2(
        tmp_path,
        as_of_date=as_of,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    assert loaded.training_dates == tuple(state_dates)
    assert loaded.training_dates[-1] == as_of
    assert loaded.transition_origin_dates == tuple(origins)
    assert loaded.transition_endpoint_dates == tuple(endpoints)
    assert len(loaded.training_dates) == 255


def _write_affine_history(path, *, dates: list[date], config: RollingGromm2Config) -> None:
    with h5py.File(path, "w") as stream:
        stream.attrs["schema"] = "gromm2-affine-dlv-history-v1"
        stream.attrs["percent_inside_spread"] = 83.5
        stream.attrs["calibration_config"] = json.dumps(
            asdict(config.calibration), sort_keys=True
        )
        stream.create_dataset(
            "dates",
            data=np.asarray([value.isoformat() for value in dates], dtype="S10"),
        )
        stream.create_dataset("nms_nodes", data=config.geometry.nms_nodes)


def _sync_physical_manifest(
    manifest: dict[str, object],
    report_path: Path,
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "explained_variance_ratio": report["explained_variance_ratio"],
            "physical_projection_fraction": report[
                "selected_achievable_fraction"
            ],
            "candidate_physical_projection_fraction": report[
                "candidate_achievable_fraction"
            ],
            "baseline_physical_projection_fraction": report[
                "baseline_achievable_fraction"
            ],
            "maximum_daily_risk_price_norm": report[
                "selected_daily_risk_price_norm"
            ],
            "candidate_daily_risk_price_norm": report[
                "candidate_daily_risk_price_norm"
            ],
            "candidate_physical_accepted": report["candidate_accepted"],
            "physical_fallback_used": report["fallback_used"],
            "physical_selected_model": report["selected_model"],
            "physical_accepted": report["selected_physical_accepted"],
        }
    )
