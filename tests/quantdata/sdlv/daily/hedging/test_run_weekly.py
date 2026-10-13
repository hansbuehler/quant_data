from __future__ import annotations

import argparse
from datetime import date
from types import SimpleNamespace

import pytest

from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_sizing import (
    WeeklyTargetNotionalConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import run_weekly
from quantdata.sdlv.daily.front_spot.hedging.runners.run_weekly import (
    _load_target_notionals,
    build_weekly_adapter,
    weekly_book_function,
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
        "base_checkpoint_sha256": None,
        "warm_start_checkpoint_sha256": None,
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


def test_weekly_sizing_loads_the_exact_preflighted_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "custom" / "notionals.json"
    captured = {}
    expected = object()

    def load(panel, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        run_weekly,
        "load_authenticated_weekly_target_notionals_from_path",
        load,
    )
    result = _load_target_notionals(
        object(),
        config=WeeklyTargetNotionalConfig(),
        artifact_path=artifact,
        expected_artifact_sha256="a" * 64,
    )

    assert result is expected
    assert captured["artifact_path"] == artifact
    assert captured["expected_artifact_sha256"] == "a" * 64


def test_weekly_conventional_gromm_builds_without_deep_policy_selection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "range.json"
    summary.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        run_weekly,
        "load_completed_rolling_gromm2_range",
        lambda *_args, **_kwargs: argparse.Namespace(
            evaluation_start=date(2023, 1, 1),
            evaluation_end=date(2025, 12, 31),
        ),
    )
    arguments = _arguments(
        "gromm-q-dlv-vega",
        start=date(2023, 1, 1),
        end=date(2025, 12, 31),
        market_artifact_root=tmp_path / "artifacts",
        market_artifact_range_summary=summary,
    )

    build = build_weekly_adapter(
        arguments,
        checkpoint_directory=None,
        dataset_manifest_sha256="d" * 64,
    )

    assert build.adapter.model == "gromm-q-dlv-vega"
    assert build.adapter.measure == "Q"
    assert build.rolling_observations == 252


def test_weekly_book_releases_generated_origin_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claims = (
        SimpleNamespace(target=SimpleNamespace(label="first")),
        SimpleNamespace(target=SimpleNamespace(label="second")),
    )

    class Adapter:
        model = "generated-test"
        measure = "P"
        release_count = 0

        def release_weekly_origin(self) -> None:
            self.release_count += 1

    adapter = Adapter()
    monkeypatch.setattr(
        run_weekly,
        "build_weekly_episode_path",
        lambda *args, **kwargs: SimpleNamespace(market_days=(object(),)),
    )
    monkeypatch.setattr(
        run_weekly,
        "build_common_weekly_claims",
        lambda *args, **kwargs: claims,
    )
    monkeypatch.setattr(
        run_weekly,
        "run_weekly_model",
        lambda *args, claim, **kwargs: claim.target.label,
    )
    run_book = weekly_book_function(
        adapter=adapter,
        schedule=run_weekly.HedgeSchedule.FROZEN_WEEKLY,
        rolling_observations=252,
        claim_quantities={"first": 1.0, "second": 2.0},
        spot_cost_bps=1.0,
    )

    assert run_book(object(), date(2025, 1, 2)) == ("first", "second")
    assert adapter.release_count == 1
