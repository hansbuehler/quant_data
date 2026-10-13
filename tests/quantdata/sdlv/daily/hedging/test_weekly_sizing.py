from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import weekly_targets
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_sizing import (
    WeeklyTargetNotionalCalibration,
    WeeklyTargetNotionalConfig,
    calibrate_weekly_target_notionals_from_pnl,
    weekly_target_notional_calibration_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import prepare_weekly_inputs


def _dates(count: int) -> tuple[tuple[date, ...], tuple[date, ...]]:
    origins = tuple(date(2022, 1, 3) + timedelta(days=index) for index in range(count))
    terminals = tuple(value + timedelta(days=7) for value in origins)
    return origins, terminals


def test_weekly_target_sizing_equalizes_realized_target_risk() -> None:
    config = WeeklyTargetNotionalConfig(minimum_episodes=20)
    origins, terminals = _dates(40)
    labels = tuple(target.label for target in weekly_targets())
    generator = np.random.default_rng(20261013)
    scales = np.linspace(0.01, 0.13, len(labels))
    pnl = generator.normal(size=(len(origins), len(labels))) * scales[None, :]

    calibration = calibrate_weekly_target_notionals_from_pnl(
        pnl,
        as_of_date=date(2022, 12, 30),
        episode_origin_dates=origins,
        episode_terminal_dates=terminals,
        target_labels=labels,
        reference_spot_standard_deviation=0.025,
        config=config,
    )

    assert np.allclose(
        calibration.target_quantities * calibration.target_standard_deviations,
        0.025,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert calibration.quantity_by_label.keys() == set(labels)
    restored = WeeklyTargetNotionalCalibration.from_dict(calibration.to_dict())
    assert restored.as_of_date == calibration.as_of_date
    assert restored.episode_origin_dates == calibration.episode_origin_dates
    assert restored.episode_terminal_dates == calibration.episode_terminal_dates
    assert restored.target_labels == calibration.target_labels
    assert np.array_equal(
        restored.target_standard_deviations,
        calibration.target_standard_deviations,
    )
    assert (
        restored.reference_spot_standard_deviation
        == calibration.reference_spot_standard_deviation
    )
    assert np.array_equal(restored.target_quantities, calibration.target_quantities)
    assert len(weekly_target_notional_calibration_sha256(calibration)) == 64


def test_weekly_target_sizing_rejects_degenerate_target() -> None:
    config = WeeklyTargetNotionalConfig(
        minimum_episodes=20,
        minimum_target_standard_deviation=1.0e-6,
    )
    origins, terminals = _dates(20)
    labels = tuple(target.label for target in weekly_targets())
    pnl = np.ones((len(origins), len(labels)))

    with pytest.raises(ValueError, match="standard deviation is too small"):
        calibrate_weekly_target_notionals_from_pnl(
            pnl,
            as_of_date=date(2022, 12, 30),
            episode_origin_dates=origins,
            episode_terminal_dates=terminals,
            target_labels=labels,
            reference_spot_standard_deviation=0.025,
            config=config,
        )


def test_weekly_target_sizing_rejects_post_initialization_terminal() -> None:
    config = WeeklyTargetNotionalConfig(minimum_episodes=20)
    origins, terminals = _dates(20)
    labels = tuple(target.label for target in weekly_targets())
    generator = np.random.default_rng(7)

    with pytest.raises(ValueError, match="initialization timing"):
        calibrate_weekly_target_notionals_from_pnl(
            generator.normal(size=(len(origins), len(labels))),
            as_of_date=terminals[-1] - timedelta(days=1),
            episode_origin_dates=origins,
            episode_terminal_dates=terminals,
            target_labels=labels,
            reference_spot_standard_deviation=0.025,
            config=config,
        )


def test_weekly_input_preflight_emits_authenticated_handoff(
    tmp_path,
    monkeypatch,
) -> None:
    config = WeeklyTargetNotionalConfig(minimum_episodes=20)
    origins, terminals = _dates(20)
    labels = tuple(target.label for target in weekly_targets())
    calibration = calibrate_weekly_target_notionals_from_pnl(
        np.random.default_rng(3).normal(size=(len(origins), len(labels))),
        as_of_date=date(2022, 12, 30),
        episode_origin_dates=origins,
        episode_terminal_dates=terminals,
        target_labels=labels,
        reference_spot_standard_deviation=0.025,
        config=config,
    )
    artifact = tmp_path / "weekly-target-notionals.json"
    artifact.write_text("authenticated weekly sizing\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_weekly_inputs,
        "load_or_calibrate_weekly_target_notionals",
        lambda panel, *, cache_directory, config: calibration,
    )
    monkeypatch.setattr(
        prepare_weekly_inputs,
        "weekly_target_notional_artifact_path",
        lambda panel, *, cache_directory, config: artifact,
    )
    monkeypatch.setattr(
        prepare_weekly_inputs,
        "dataset_manifest_sha256",
        lambda panel: "d" * 64,
    )

    payload = prepare_weekly_inputs.prepare_weekly_inputs(
        SimpleNamespace(),
        cache_directory=tmp_path,
        config=config,
    )

    assert payload["dataset_manifest_sha256"] == "d" * 64
    assert payload["target_notional_artifact"] == str(artifact)
    assert len(payload["target_notional_artifact_sha256"]) == 64
    assert payload["target_count"] == len(labels)
    assert payload["episode_count"] == 20
    assert payload["target_quantities"] == calibration.quantity_by_label
