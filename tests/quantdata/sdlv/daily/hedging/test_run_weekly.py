from __future__ import annotations

import argparse

import pytest

from quantdata.sdlv.daily.front_spot.hedging.runners.run_weekly import (
    build_weekly_adapter,
)


def _arguments(model: str, **updates) -> argparse.Namespace:
    values = {
        "model": model,
        "schedule": "frozen-weekly",
        "label": None,
        "objective": "mean-variance",
        "risk_parameter": 1.0,
        "measure": "Q",
        "position_cap": 1_000.0,
        "market_artifact_root": None,
        "market_artifact_range_summary": None,
        "architecture_selection": None,
        "base_checkpoint": None,
        "warm_start_checkpoint": None,
        "observation_count": 252,
        "start": None,
        "end": None,
        "epochs": 2,
        "batch_size": 8,
        "scenario_count": 16,
        "training_progress_seconds": 240.0,
        "checkpoint_epochs": 1,
        "device": "cpu",
        "precision": None,
        "compile_model": False,
        "no_mixed_precision": False,
        "just_load": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("model", "expected_model", "expected_measure"),
    (
        ("sanos-lv", "sanos-lv", "Q"),
        ("historical-sanos", "historical-sanos", "historical-P"),
        ("volgan", "volgan", "historical-P"),
        ("diffusion-ad-seq-vol", "ad-seq-vol", "historical-P"),
    ),
)
def test_weekly_adapter_builds_are_fully_identified(
    tmp_path,
    model: str,
    expected_model: str,
    expected_measure: str,
) -> None:
    build = build_weekly_adapter(
        _arguments(model),
        checkpoint_directory=tmp_path,
        dataset_manifest_sha256="d" * 64,
    )

    assert build.adapter.model == expected_model
    assert build.adapter.measure == expected_measure
    assert build.rolling_observations == 252
    assert build.position_cap == 1_000.0
    assert "common_market_pricing" in build.configuration_snapshot_json


def test_weekly_fine_tuned_diffusion_requires_base_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --base-checkpoint"):
        build_weekly_adapter(
            _arguments("diffusion-ad-seq-vol-ft"),
            checkpoint_directory=tmp_path,
            dataset_manifest_sha256="d" * 64,
        )


def test_weekly_diffusion_precision_changes_configuration_identity(tmp_path) -> None:
    build32 = build_weekly_adapter(
        _arguments("diffusion-ad-seq-vol", precision=32),
        checkpoint_directory=tmp_path,
        dataset_manifest_sha256="d" * 64,
    )
    build64 = build_weekly_adapter(
        _arguments("diffusion-ad-seq-vol", precision=64),
        checkpoint_directory=tmp_path,
        dataset_manifest_sha256="d" * 64,
    )

    assert build32.configuration_snapshot_json != build64.configuration_snapshot_json
    assert '"precision":"torch.float32"' in build32.configuration_snapshot_json
    assert '"precision":"torch.float64"' in build64.configuration_snapshot_json


def test_weekly_gromm_requires_completed_range_evidence(tmp_path) -> None:
    with pytest.raises(ValueError, match="market-artifact-range-summary"):
        build_weekly_adapter(
            _arguments(
                "gromm",
                architecture_selection=tmp_path / "architecture.json",
            ),
            checkpoint_directory=tmp_path,
            dataset_manifest_sha256="d" * 64,
        )
