"""Check strict dated checkpoint lineage in rolling diffusion runs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.runners.run_rolling_diffusion import (
    _carry_without_update,
    _completed_fit,
    _completed_pre_evaluation_fit,
    _fit_pre_evaluation_date,
    _has_fresh_diffusion_target,
    _selected_epochs,
)
from quantdata.torch import torch


def test_completed_fit_rejects_missing_or_wrong_parent(tmp_path: Path) -> None:
    origin = date(2024, 4, 1)
    checkpoint = tmp_path / "dated.pt"
    torch.save({"training_end_date": origin.isoformat()}, checkpoint)
    assert _completed_fit(tmp_path, origin_date=origin, expected_parent_sha256="parent") is None
    records = tmp_path / "records"
    records.mkdir()
    (records / "2024-04-01.json").write_text(
        json.dumps(
            {
                "origin_date": "2024-04-01",
                "results": [
                    {
                        "diagnostics": {
                            "warm_start_sha256": "parent",
                            "fit_checkpoint_path": str(checkpoint),
                            "fit_checkpoint_sha256": file_sha256(checkpoint),
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _completed_fit(
        tmp_path, origin_date=origin, expected_parent_sha256="parent"
    ) == (checkpoint, file_sha256(checkpoint))
    with pytest.raises(RuntimeError, match="parent hash mismatch"):
        _completed_fit(tmp_path, origin_date=origin, expected_parent_sha256="older")
    torch.save({"training_end_date": "2024-04-02"}, checkpoint)
    with pytest.raises(RuntimeError, match="changed"):
        _completed_fit(tmp_path, origin_date=origin, expected_parent_sha256="parent")


def test_completed_fit_authenticates_generated_scenarios(tmp_path: Path) -> None:
    origin = date(2024, 4, 1)
    checkpoint = tmp_path / "dated.pt"
    scenarios = tmp_path / "generated-scenarios.npz"
    torch.save({"training_end_date": origin.isoformat()}, checkpoint)
    scenarios.write_bytes(b"scenario paths")
    records = tmp_path / "records"
    records.mkdir()
    (records / "2024-04-01.json").write_text(
        json.dumps(
            {
                "origin_date": origin.isoformat(),
                "results": [
                    {
                        "diagnostics": {
                            "warm_start_sha256": "parent",
                            "fit_checkpoint_path": str(checkpoint),
                            "fit_checkpoint_sha256": file_sha256(checkpoint),
                            "scenario_artifact_path": str(scenarios),
                            "scenario_artifact_sha256": file_sha256(scenarios),
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _completed_fit(
        tmp_path, origin_date=origin, expected_parent_sha256="parent"
    ) == (checkpoint, file_sha256(checkpoint))
    scenarios.write_bytes(b"altered paths")
    with pytest.raises(RuntimeError, match="scenarios are absent or changed"):
        _completed_fit(tmp_path, origin_date=origin, expected_parent_sha256="parent")


def test_epoch_selection_requires_same_geometry_and_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    production = tmp_path / "production.pt"
    torch.save(
        {"base_compatibility_identity": "grid-a", "model_code_version": "v1"},
        source,
    )
    torch.save(
        {"base_compatibility_identity": "grid-a", "model_code_version": "v1"},
        production,
    )
    selection = tmp_path / "selection.json"
    payload = {
        "schema": "hedging-generated-epoch-selection-v1",
        "model": "diffusion-ad-seq-vol",
        "dataset_manifest_sha256": "a" * 64,
        "one_standard_error_rule": True,
        "selected_epochs": 30,
        "warm_start_checkpoint": str(source),
        "warm_start_sha256": file_sha256(source),
        "validation_origins": ["2022-07-06", "2022-12-30"],
    }
    selection.write_text(json.dumps(payload), encoding="utf-8")
    assert _selected_epochs(
        selection,
        model="diffusion-ad-seq-vol",
        dataset_manifest_sha256="a" * 64,
        production_checkpoint=production,
    ) == 30
    with pytest.raises(ValueError, match="missing or incompatible"):
        _selected_epochs(
            selection,
            model="diffusion-ad-seq-vol",
            dataset_manifest_sha256="b" * 64,
            production_checkpoint=production,
        )
    torch.save(
        {"base_compatibility_identity": "grid-b", "model_code_version": "v1"},
        production,
    )
    with pytest.raises(ValueError, match="another model geometry"):
        _selected_epochs(
            selection,
            model="diffusion-ad-seq-vol",
            dataset_manifest_sha256="a" * 64,
            production_checkpoint=production,
        )


def test_gap_day_carries_previous_fitted_weights_explicitly(tmp_path: Path) -> None:
    origin = date(2023, 5, 18)
    dates = (date(2023, 5, 15), date(2023, 5, 16), date(2023, 5, 17), origin)
    sessions = (0, 1, 1, 2)
    observations = SimpleNamespace(
        row=lambda index, named: {
            "business_sessions_from_previous_observation": sessions[index]
        }
    )
    panel = SimpleNamespace(dates=dates, observations=observations)
    assert not _has_fresh_diffusion_target(panel, origin)
    checkpoint = tmp_path / "previous.pt"
    torch.save({"training_end_date": "2023-05-17"}, checkpoint)
    parent_sha256 = file_sha256(checkpoint)
    lineage = tmp_path / "lineage.json"
    assert _carry_without_update(
        lineage,
        origin_date=origin,
        model="diffusion-ad-seq-vol",
        model_label="rolling-base",
        epochs=30,
        parent_path=checkpoint,
        parent_sha256=parent_sha256,
        dataset_manifest_sha256="a" * 64,
        epoch_selection_sha256="b" * 64,
    ) == (checkpoint, parent_sha256)
    assert _completed_pre_evaluation_fit(
        lineage,
        origin_date=origin,
        model="diffusion-ad-seq-vol",
        model_label="rolling-base",
        epochs=30,
        parent_sha256=parent_sha256,
        dataset_manifest_sha256="a" * 64,
        epoch_selection_sha256="b" * 64,
    ) == (checkpoint, parent_sha256)
    payload = json.loads(lineage.read_text(encoding="utf-8"))
    assert payload["status"] == "no-update"
    assert payload["reason"] == "missing exact-session return"


def test_fit_after_gap_inherits_last_fitted_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantdata.sdlv.daily.front_spot.hedging.runners import (
        run_rolling_diffusion as rolling,
    )

    prior = tmp_path / "prior.pt"
    torch.save({"training_end_date": "2023-05-17"}, prior)
    prior_sha256 = file_sha256(prior)
    fitted_checkpoint = tmp_path / "new-fit.pt"
    torch.save({"training_end_date": "2023-05-23"}, fitted_checkpoint)

    def fake_fit(panel: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["warm_start_checkpoint"] == prior
        assert kwargs["expected_warm_start_sha256"] == prior_sha256
        assert kwargs["as_of_date"] == date(2023, 5, 23)
        return SimpleNamespace(
            result=SimpleNamespace(
                converged=True,
                warm_start_sha256=prior_sha256,
                checkpoint_path=fitted_checkpoint,
                validation_loss=0.1,
            )
        )

    monkeypatch.setattr(rolling, "fit_public_diffusion", fake_fit)
    lineage = tmp_path / "2023-05-23.json"
    fitted = _fit_pre_evaluation_date(
        SimpleNamespace(dataset_manifest_sha256="a" * 64),
        origin_date=date(2023, 5, 23),
        model="diffusion-ad-seq-vol",
        model_label="rolling-base",
        parent_path=prior,
        parent_sha256=prior_sha256,
        base_checkpoint=None,
        base_checkpoint_sha256=None,
        checkpoint_directory=tmp_path,
        observation_count=252,
        epochs=30,
        epoch_selection_sha256="b" * 64,
        lineage_path=lineage,
    )
    assert fitted == (fitted_checkpoint, file_sha256(fitted_checkpoint))
    assert _completed_pre_evaluation_fit(
        lineage,
        origin_date=date(2023, 5, 23),
        model="diffusion-ad-seq-vol",
        model_label="rolling-base",
        epochs=30,
        parent_sha256=prior_sha256,
        dataset_manifest_sha256="a" * 64,
        epoch_selection_sha256="b" * 64,
    ) == fitted
