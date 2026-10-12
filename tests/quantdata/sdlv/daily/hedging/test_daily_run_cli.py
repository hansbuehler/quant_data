from __future__ import annotations

import argparse
import json
from datetime import date

import pytest

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
        warm_start_checkpoint=None,
        retain_fit_checkpoints=False,
        measure="Q",
        market_artifact_root=None,
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
    monkeypatch.setattr(run_daily, "run_gromm2_daily_book", run_book)

    model, measure, observations, config, book = run_daily._build_model(
        arguments,
        claim_quantities={"target": 0.25},
        default_position_cap=1_000.0,
        checkpoint_directory=tmp_path / "policy",
    )

    assert (model, measure, observations) == ("gromm", "P", 252)
    assert config.model_config.path_count == 16
    assert config.model_config.batch_size == 8
    assert config.rolling_config.factor_count == 16
    panel = object()
    assert book(panel, date(2023, 1, 17)) == (panel,)
    assert captured["artifact_root"] == tmp_path / "market"
    assert captured["claim_quantities"] == {"target": 0.25}


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
    assert config.warm_start_checkpoint_sha256 is not None
    panel = object()
    assert book(panel, date(2023, 1, 17)) == (panel,)
    assert (
        captured["expected_warm_start_sha256"]
        == config.warm_start_checkpoint_sha256
    )
