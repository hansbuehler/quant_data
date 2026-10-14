from __future__ import annotations

import argparse
import json
from datetime import date

import pytest

from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.selection import (
    GrommArchitectureStudyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import run_daily


def _arguments(*, model: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        label=None,
        objective="mean-variance",
        risk_parameter=10.0,
        position_cap=None,
        observation_count=252,
        epochs=2,
        batch_size=8,
        scenario_count=16,
        training_progress_seconds=60.0,
        checkpoint_epochs=1,
        device="cpu",
        precision=None,
        compile_model=False,
        no_mixed_precision=False,
        just_load=False,
        base_checkpoint=None,
        base_checkpoint_sha256=None,
        warm_start_checkpoint=None,
        warm_start_checkpoint_sha256=None,
        retain_fit_checkpoints=False,
        measure="Q",
        market_artifact_root=None,
        market_artifact_range_summary=None,
        architecture_selection=None,
        start=date(2023, 1, 1),
        end=date(2025, 12, 31),
    )


def _write_gromm_architecture(path, *, dataset_manifest_sha256: str) -> None:
    candidates = [
        {
            "label": f"candidate-{index}",
            "hidden_size": hidden_size,
            "recurrent_layers": 1,
            "deep_activation": "SiLU",
            "deep_depth": 1,
            "deep_width": hidden_size,
            "parameter_count": hidden_size * 10,
            "best_epoch": 10,
            "completed_epochs": 20,
            "training_loss": 0.2,
            "validation_loss": 0.1 + 0.01 * index,
            "validation_standard_error": 0.02,
            "cap_hit_fraction": 0.0,
            "converged": True,
        }
        for index, hidden_size in enumerate((8, 16, 32, 64))
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "gromm-architecture-selection-v2",
                "code_version": str(GrommArchitectureStudyConfig.version),
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "origin_date": "2022-12-30",
                "measures": ["Q", "P"],
                "targets": [target.label for target in daily_digital_targets()],
                "case_count": 2 * len(daily_digital_targets()),
                "factor_rank": 16,
                "physical_drift_accepted": True,
                "selected_label": "candidate-0",
                "best_validation_loss": 0.1,
                "best_validation_standard_error": 0.02,
                "one_standard_error_threshold": 0.12,
                "training_feature_count": 21,
                "instrument_count": 17,
                "candidates": candidates,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_volgan_daily_builder_pins_training_and_quantity_controls(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def run_book(panel, **kwargs):
        captured.update(kwargs)
        return (panel,)

    monkeypatch.setattr(run_daily, "run_volgan_daily_book", run_book)
    model, measure, observations, config, book = run_daily._build_model(
        _arguments(model="volgan"),
        claim_quantities={"target": 0.25},
        default_position_cap=1_000.0,
        checkpoint_directory=tmp_path,
    )
    assert (model, measure, observations) == ("volgan", "historical-P", 252)
    assert config.model_config.label == "volgan-rolling-public-settings"
    assert config.train_config.epochs == 2
    assert config.hedge_config.scenario_count == 16
    assert config.hedge_config.position_cap == 1_000.0
    assert json.loads(run_daily._config_snapshot_json(config))["fields"][
        "observation_count"
    ] == 252

    panel = object()
    assert book(panel, date(2023, 1, 17)) == (panel,)
    assert captured["claim_quantities"] == {"target": 0.25}
    assert captured["checkpoint_directory"] == tmp_path
    assert not captured["retain_fit_checkpoint"]


def test_diffusion_fine_tuning_requires_base_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --base-checkpoint"):
        run_daily._build_model(
            _arguments(model="diffusion-ad-seq-vol-ft"),
            claim_quantities={"target": 0.25},
            default_position_cap=1_000.0,
            checkpoint_directory=tmp_path,
        )


def test_covariance_daily_builder_returns_its_book(tmp_path, monkeypatch) -> None:
    captured = {}

    def run_book(panel, **kwargs):
        captured.update(kwargs)
        return (panel,)

    monkeypatch.setattr(run_daily, "run_covariance_daily_book", run_book)
    model, measure, observations, config, book = run_daily._build_model(
        _arguments(model="covariance"),
        claim_quantities={"target": 0.25},
        default_position_cap=1_000.0,
        checkpoint_directory=tmp_path,
    )

    assert (model, measure, observations) == (
        "robust-covariance",
        "historical-P",
        252,
    )
    panel = object()
    assert book(panel, date(2023, 1, 17)) == (panel,)
    assert captured["config"] is config
    assert captured["claim_quantities"] == {"target": 0.25}


def test_gromm_daily_builder_pins_dated_market_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    captured = {}

    def run_book(panel, **kwargs):
        captured.update(kwargs)
        return (panel,)

    arguments = _arguments(model="gromm")
    arguments.measure = "P"
    arguments.market_artifact_root = tmp_path / "market"
    arguments.architecture_selection = tmp_path / "architecture.json"
    dataset_manifest_sha256 = "a" * 64
    _write_gromm_architecture(
        arguments.architecture_selection,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    monkeypatch.setattr(run_daily, "run_gromm2_daily_book", run_book)

    model, measure, observations, config, book = run_daily._build_model(
        arguments,
        claim_quantities={"target": 0.25},
        default_position_cap=1_000.0,
        checkpoint_directory=tmp_path / "policy",
        expected_dataset_manifest_sha256=dataset_manifest_sha256,
    )

    assert (model, measure, observations) == ("gromm", "P", 252)
    assert config.model_config.path_count == 16
    assert config.model_config.batch_size == 8
    assert config.rolling_config.factor_count == 16
    assert config.policy_architecture.hidden_size == 8
    assert len(config.architecture_selection_sha256) == 64
    panel = object()
    assert book(panel, date(2023, 1, 17)) == (panel,)
    assert captured["artifact_root"] == tmp_path / "market"
    assert captured["claim_quantities"] == {"target": 0.25}
    assert captured["policy_config"].hidden_size == 8


def test_gromm_daily_builder_requires_frozen_architecture(tmp_path) -> None:
    with pytest.raises(ValueError, match="architecture-selection"):
        run_daily._build_model(
            _arguments(model="gromm"),
            claim_quantities={"target": 0.25},
            default_position_cap=1_000.0,
            checkpoint_directory=tmp_path,
            expected_dataset_manifest_sha256="a" * 64,
        )


def test_conventional_gromm_daily_builder_uses_q_price_and_dlv_hedge(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def run_book(panel, **kwargs):
        captured.update(kwargs)
        return (panel,)

    summary = tmp_path / "range.json"
    summary.write_text("{}\n", encoding="utf-8")
    arguments = _arguments(model="gromm-q-dlv-vega")
    arguments.dataset = tmp_path / "dataset"
    arguments.market_artifact_root = tmp_path / "market"
    arguments.market_artifact_range_summary = summary
    monkeypatch.setattr(
        run_daily,
        "load_completed_rolling_gromm2_range",
        lambda *_args, **_kwargs: argparse.Namespace(
            evaluation_start=arguments.start,
            evaluation_end=arguments.end,
        ),
    )
    monkeypatch.setattr(run_daily, "run_gromm2_conventional_daily_book", run_book)

    model, measure, observations, config, book = run_daily._build_model(
        arguments,
        claim_quantities={"target": 0.25},
        default_position_cap=1_000.0,
        checkpoint_directory=None,
        expected_dataset_manifest_sha256="a" * 64,
    )

    assert (model, measure, observations) == (
        "gromm-q-dlv-vega",
        "Q-price/historical-transition",
        252,
    )
    panel = object()
    assert book(panel, date(2023, 1, 17)) == (panel,)
    assert captured["artifact_root"] == tmp_path / "market"
    assert captured["claim_quantities"] == {"target": 0.25}
    assert config.dlv_config.rolling_factors.factor_rank == 8


@pytest.mark.parametrize(
    ("model", "field"),
    (("volgan", "gradient_matching_epochs"), ("diffusion-ad-seq-vol", "warmup_steps")),
)
def test_generated_rolling_fit_skips_initializer_warmup(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    field: str,
) -> None:
    source = tmp_path / "initializer.pt"
    source.write_bytes(b"initializer")
    arguments = _arguments(model=model)
    arguments.warm_start_checkpoint = source
    arguments.model_label = "authenticated-initializer"
    captured = {}

    def run_book(panel, **kwargs):
        captured.update(kwargs)
        return (panel,)

    runner_name = (
        "run_volgan_daily_book"
        if model == "volgan"
        else "run_diffusion_daily_book"
    )
    monkeypatch.setattr(run_daily, runner_name, run_book)

    _, _, _, config, book = run_daily._build_model(
        arguments,
        claim_quantities={"target": 0.25},
        default_position_cap=1_000.0,
        checkpoint_directory=tmp_path,
    )

    assert getattr(config.train_config, field) == 0
    assert config.model_config.label == "authenticated-initializer"
    assert config.warm_start_checkpoint_sha256 is not None
    panel = object()
    assert book(panel, date(2023, 1, 17)) == (panel,)
    assert (
        captured["expected_warm_start_sha256"]
        == config.warm_start_checkpoint_sha256
    )
