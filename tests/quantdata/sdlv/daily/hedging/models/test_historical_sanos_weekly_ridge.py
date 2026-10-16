from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import date, timedelta

import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_ridge import (
    WeeklyRidgeBundle,
    WeeklyRidgeCandidateResult,
    WeeklyRidgeSelection,
    load_weekly_ridge_bundle,
    load_weekly_ridge_for_scope,
    load_weekly_ridge_selection,
    repository_source_commit,
    select_weekly_ridge,
    write_weekly_ridge_bundle,
    write_weekly_ridge_selection,
)


def _candidate(
    ridge: float | None,
    mean: float,
    standard_error: float,
    *,
    feasible: bool = True,
) -> WeeklyRidgeCandidateResult:
    scale = math.sqrt(2.0) * standard_error
    blocks = tuple(
        mean + scale * offset
        for offset in (-2.0, -1.0, 0.0, 1.0, 2.0)
    )
    return WeeklyRidgeCandidateResult(
        ridge=ridge,
        block_regrets=blocks,
        pricing_block_regrets=blocks,
        zero_block_regrets=blocks,
        execution_block_regrets=blocks,
        mean_regret=mean,
        standard_error=standard_error,
        feasible=feasible,
        selection_converged_components=40 if feasible else 39,
        final_converged_components=40,
        converged_components=80 if feasible else 79,
        component_count=80,
        selection_cap_hits=0,
        internal_cap_hits=0,
        executable_cap_hits=0,
        cap_hits=0,
        unregularized_recession_directions=0,
        recession_directions=0,
        maximum_objective_error=1.0e-10,
        maximum_constraint_violation=1.0e-9,
        maximum_latent_coordinate=42.0,
    )


def _selection() -> WeeklyRidgeSelection:
    first = date(2022, 1, 1)
    training = tuple(first + timedelta(days=index) for index in range(252))
    candidates = (
        _candidate(0.01, -0.20, 0.03),
        _candidate(0.1, -0.18, 0.02),
        _candidate(1.0, -0.15, 0.01, feasible=False),
    )
    return WeeklyRidgeSelection(
        dataset_manifest_sha256="1" * 64,
        target_notional_artifact_sha256="2" * 64,
        origin_date=date(2023, 1, 17),
        training_origin_dates=training,
        scenario_origin_dates=training[-230:],
        fit_origin_dates=training[-230:-46],
        validation_origin_dates=training[-46:],
        validation_blocks=(
            training[-46:-36],
            training[-36:-27],
            training[-27:-18],
            training[-18:-9],
            training[-9:],
        ),
        schedule_family=("frozen-weekly", "restarted-weekly"),
        target_labels=("target",),
        target_quantities=(1.0,),
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=0.1,
        validation_fraction=0.2,
        position_cap=1_000.0,
        maximum_latent_coordinate=1.0e5,
        candidates=candidates,
        selected_ridge=0.1,
        selected_by_one_standard_error=True,
        source_commit=repository_source_commit(),
        code_version=str(HistoricalSanosConfig.version),
    )


def test_select_weekly_ridge_uses_largest_feasible_candidate_within_one_se() -> None:
    candidates = (
        _candidate(0.001, -0.25, 0.05),
        _candidate(0.01, -0.22, 0.04),
        _candidate(0.1, -0.205, 0.01),
        _candidate(1.0, -0.19, 0.01),
        _candidate(10.0, -1.0, 0.01, feasible=False),
    )

    selected = select_weekly_ridge(candidates)

    assert selected.ridge == 0.1


def test_select_weekly_ridge_chooses_exact_null_when_finite_fits_overfit() -> None:
    candidates = (
        _candidate(0.1, 0.25, 0.10),
        _candidate(1.0, 0.08, 0.03),
        _candidate(None, 0.0, 0.0),
    )

    selected = select_weekly_ridge(candidates)

    assert selected.ridge is None


def test_select_weekly_ridge_rejects_null_when_every_finite_fit_failed() -> None:
    candidates = (
        _candidate(0.1, 0.25, 0.10, feasible=False),
        _candidate(1.0, 0.08, 0.03, feasible=False),
        _candidate(None, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="no feasible finite candidate"):
        select_weekly_ridge(candidates)


def test_weekly_ridge_candidate_cannot_hide_split_or_cap_failure() -> None:
    split_failure = _candidate(0.1, 0.0, 0.01, feasible=False)
    cap_failure = replace(
        _candidate(1.0, 0.0, 0.01),
        feasible=False,
        executable_cap_hits=1,
        cap_hits=1,
    )

    with pytest.raises(ValueError, match="failed gates"):
        replace(split_failure, feasible=True)
    with pytest.raises(ValueError, match="failed gates"):
        replace(cap_failure, feasible=True)


def test_weekly_ridge_selection_round_trip_is_authenticated(tmp_path) -> None:
    selection = _selection()
    path = tmp_path / "weekly-ridge.json"
    write_weekly_ridge_selection(selection, output=path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_weekly_ridge_selection(
        path,
        expected_sha256=file_sha256,
        expected_dataset_manifest_sha256="1" * 64,
        expected_target_notional_artifact_sha256="2" * 64,
        expected_source_commit=repository_source_commit(),
        expected_position_cap=1_000.0,
    )

    assert loaded == selection
    assert loaded.sha256 == selection.sha256


def test_weekly_ridge_selection_round_trips_exact_null_policy(tmp_path) -> None:
    selection = replace(
        _selection(),
        candidates=(
            _candidate(0.1, 0.1, 0.01),
            _candidate(None, 0.0, 0.0),
        ),
        selected_ridge=None,
    )
    path = tmp_path / "weekly-null-ridge.json"
    write_weekly_ridge_selection(selection, output=path)

    loaded = load_weekly_ridge_selection(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_dataset_manifest_sha256="1" * 64,
        expected_target_notional_artifact_sha256="2" * 64,
        expected_source_commit=repository_source_commit(),
        expected_position_cap=1_000.0,
    )

    assert loaded.selected_ridge is None
    assert loaded.to_dict()["selected_policy"] == "null-baseline"


def test_weekly_ridge_selection_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "weekly-ridge.json"
    write_weekly_ridge_selection(_selection(), output=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_labels"] = ["tampered"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest is invalid"):
        load_weekly_ridge_selection(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_dataset_manifest_sha256="1" * 64,
            expected_target_notional_artifact_sha256="2" * 64,
            expected_source_commit=repository_source_commit(),
            expected_position_cap=1_000.0,
        )


def test_weekly_ridge_bundle_selects_exact_objective_and_risk(tmp_path) -> None:
    entropy = _selection()
    mean_volatility = replace(
        entropy,
        objective=HistoricalSanosObjective.MEAN_VOLATILITY,
    )
    bundle = WeeklyRidgeBundle(
        dataset_manifest_sha256=entropy.dataset_manifest_sha256,
        target_notional_artifact_sha256=(
            entropy.target_notional_artifact_sha256
        ),
        quant_review_sha256="3" * 64,
        selections=(entropy, mean_volatility),
    )
    path = tmp_path / "bundle.json"
    write_weekly_ridge_bundle(bundle, output=path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_weekly_ridge_bundle(
        path,
        expected_sha256=digest,
        expected_dataset_manifest_sha256=entropy.dataset_manifest_sha256,
        expected_target_notional_artifact_sha256=(
            entropy.target_notional_artifact_sha256
        ),
        expected_source_commit=repository_source_commit(),
        expected_position_cap=1_000.0,
    )
    assert loaded.sha256 == bundle.sha256
    selected = load_weekly_ridge_for_scope(
        path,
        expected_sha256=digest,
        expected_dataset_manifest_sha256=entropy.dataset_manifest_sha256,
        expected_target_notional_artifact_sha256=(
            entropy.target_notional_artifact_sha256
        ),
        expected_source_commit=repository_source_commit(),
        expected_position_cap=1_000.0,
        expected_objective=HistoricalSanosObjective.MEAN_VOLATILITY,
        expected_risk_parameter=0.1,
    )
    assert selected.objective is HistoricalSanosObjective.MEAN_VOLATILITY
    with pytest.raises(ValueError, match="lacks the requested"):
        loaded.selection_for(
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=0.1,
        )


def test_weekly_ridge_bundle_rejects_mixed_calibration_designs() -> None:
    entropy = _selection()
    incompatible = replace(
        entropy,
        objective=HistoricalSanosObjective.MEAN_VOLATILITY,
        maximum_latent_coordinate=2.0e5,
    )

    with pytest.raises(ValueError, match="designs do not align"):
        WeeklyRidgeBundle(
            dataset_manifest_sha256=entropy.dataset_manifest_sha256,
            target_notional_artifact_sha256=(
                entropy.target_notional_artifact_sha256
            ),
            quant_review_sha256="3" * 64,
            selections=(entropy, incompatible),
        )


def test_weekly_ridge_loader_rejects_another_source_commit(tmp_path) -> None:
    path = tmp_path / "weekly-ridge.json"
    write_weekly_ridge_selection(_selection(), output=path)

    with pytest.raises(ValueError, match="source commit"):
        load_weekly_ridge_selection(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_dataset_manifest_sha256="1" * 64,
            expected_target_notional_artifact_sha256="2" * 64,
            expected_source_commit="f" * 40,
            expected_position_cap=1_000.0,
        )


def test_weekly_ridge_loader_rejects_another_position_cap(tmp_path) -> None:
    path = tmp_path / "weekly-ridge.json"
    write_weekly_ridge_selection(_selection(), output=path)

    with pytest.raises(ValueError, match="position cap"):
        load_weekly_ridge_selection(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_dataset_manifest_sha256="1" * 64,
            expected_target_notional_artifact_sha256="2" * 64,
            expected_source_commit=repository_source_commit(),
            expected_position_cap=999.0,
        )


def test_weekly_ridge_selection_rejects_wrong_one_se_policy() -> None:
    with pytest.raises(ValueError, match="one-SE rule"):
        replace(_selection(), selected_ridge=0.01)


def test_weekly_ridge_selection_rejects_inconsistent_validation_fraction() -> None:
    with pytest.raises(ValueError, match="validation fraction"):
        replace(_selection(), validation_fraction=0.5)
