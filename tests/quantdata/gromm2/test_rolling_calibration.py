from __future__ import annotations

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest

from quantdata.gromm2 import rolling_calibration
from quantdata.gromm2.rolling_calibration import (
    RollingFitConfig,
    ar1_estimate,
    estimate_window,
    fit_summary,
    joint_innovation_loadings,
    saved_parameter_inventory,
    select_origin_inventory,
    source_inventory,
    training_prefix,
)


def test_training_prefix_excludes_current_and_future_dates() -> None:
    dates = [f"2024-01-{day:02d}" for day in (2, 3, 4, 5, 8, 9, 10, 11)]
    assert training_prefix(dates, 7, window=4) == [3, 4, 5, 6]
    with pytest.raises(ValueError, match="complete historical window"):
        training_prefix(dates, 2, window=4)


def test_history_window_does_not_reach_further_back_to_replace_missing_days() -> None:
    dates = [f"2024-01-{day:02d}" for day in (2, 3, 4, 5, 8, 10, 11)]
    assert training_prefix(dates, 6, window=4) == [3, 4, 5]


def test_single_origin_inventory_excludes_unneeded_dates_and_their_failures() -> None:
    dates = [f"2024-01-{day:02d}" for day in (2, 3, 4, 5, 8, 10, 11, 12)]
    inventory = {
        "dates": {day: {"terminal_business_dte": 60.0 if index == 0 else 140.0} for index, day in enumerate(dates)},
        "required_terminal_business_dte": 125.0, "ready": False,
    }
    result = select_origin_inventory(inventory, origin=dates[6], window=4)
    assert list(result["dates"]) == dates[3:7]
    assert result["requested_origin"] == dates[6]
    assert result["ready"]
    assert result["date_count"] == 4
    with pytest.raises(ValueError, match="absent"):
        select_origin_inventory(inventory, origin="2024-01-15", window=4)


def test_single_origin_history_warm_starts_chronologically(tmp_path, monkeypatch) -> None:
    dates = [f"2024-01-{day:02d}" for day in (2, 3, 4, 5, 8, 9)]
    origin = dates[-1]
    own = tmp_path / "own_dates" / origin
    own.mkdir(parents=True)
    np.savez(own / "inputs.npz", supports=np.asarray([[0.5, 0.9, 1.1, 1.5]] * 2), model_expiries_y=np.asarray([1.0, 2.0]) / 252.0)
    calls = []

    def fit_task(source, output, config, fixed_grid, initializer):
        calls.append((source, initializer))
        directory = tmp_path / output
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "fit.npz", dlvsq=np.ones((2, 2)), densities=np.asarray([[0.0, 0.5, 0.5, 0.0]] * 2),
                 spot_ref=np.asarray(1.0), forward_over_spot=np.ones(2))
        return {"date": source, "stop_reason": "maximum_epochs", "fit": {"all": {"inside_percent": 95.0}}}

    monkeypatch.setattr(rolling_calibration, "_fit_task", fit_task)
    monkeypatch.setattr(rolling_calibration, "fit_summary", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setattr(rolling_calibration, "estimate_window", lambda **kwargs: {
        "innovations": np.zeros((3, 1, 2)), "spot_gaussian_scores": np.zeros(3),
        "pair_end_dates": np.asarray(dates[2:5]), "fixed_contract_clipped_query_fraction": np.asarray(0.0),
        "slope": np.zeros((1, 2)), "innovation_variance": np.zeros((1, 2)),
    })
    monkeypatch.setattr(rolling_calibration, "joint_innovation_loadings", lambda *args, **kwargs: {"q_log_dlvsq_loadings": np.zeros((1, 2, 2))})
    config = replace(RollingFitConfig(), history_business_days=4, rolling_start_date="2024-01-01", diffusion_factors=2)
    rolling_calibration.rolling_stage(
        inventory={"dates": {day: {"path": day} for day in dates}}, run_directory=tmp_path,
        config=config, status={}, origin_date=origin,
    )
    assert [day for day, _ in calls] == dates[1:-1]
    assert calls[0][1] is None
    for index in range(1, len(calls)):
        assert calls[index][1] == str(tmp_path / "rolling" / origin / "history" / calls[index - 1][0])
    assert json.loads((tmp_path / "rolling_complete.json").read_text())["completed_origins"] == 1


def test_ar1_recovers_known_conditional_mean_and_constant_coordinate() -> None:
    origins = np.stack((np.linspace(-1.0, 1.0, 25), np.full(25, 2.0)), axis=1)
    endpoints = 0.3 + 0.8 * origins
    result = ar1_estimate(origins, endpoints)
    np.testing.assert_allclose(result["slope"], [0.8, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(result["intercept"], [0.3, 1.9], atol=1.0e-14)
    np.testing.assert_allclose(result["innovations"], 0.0, atol=1.0e-14)
    np.testing.assert_array_equal(result["identified_slope"], [True, False])
    assert np.isnan(result["slope_standard_error"][1])


def test_inventory_requires_six_months_for_every_date(tmp_path) -> None:
    def source(label: str, day: str, horizon: float) -> None:
        directory = tmp_path / label / "SPX"
        directory.mkdir(parents=True, exist_ok=True)
        with h5py.File(directory / f"{day}.h5", "w") as stream:
            stream.create_dataset("root/af_pricer/expiries", data=np.asarray([1.0, horizon]) / 252.0)
            stream.create_dataset("root/af_pricer/_1_Nsmoothness", data=0.25)

    source("old", "2024-01-02", 63.0)
    source("old", "2024-01-03", 63.0)
    source("full", "2024-01-03", 138.2)
    config = replace(RollingFitConfig(), start_date="2024-01-02", end_date="2024-01-03")
    result = source_inventory(tmp_path, config=config)
    assert result["date_count"] == 2
    assert result["complete_horizon_dates"] == 1
    assert result["incomplete_horizon_dates"] == ["2024-01-02"]
    assert not result["ready"]
    source("full", "2024-01-02", 138.2)
    assert source_inventory(tmp_path, config=config)["ready"]


def test_pinned_saved_parameters_do_not_require_new_longer_horizon_fits(tmp_path) -> None:
    dates = [f"2024-01-{day:02d}" for day in (2, 3, 4, 5, 8, 9)]
    entries = {}
    for day in dates:
        path = tmp_path / f"{day}.h5"
        with h5py.File(path, "w") as stream:
            stream.create_dataset("root/af_pricer/expiries", data=np.asarray([1.0, 60.0]) / 252.0)
            stream.create_dataset("root/af_pricer/_1_Nsmoothness", data=0.25)
        entries[day] = {"path": str(path), "terminal_business_dte": 60.0}
    manifest = tmp_path / "pinned.json"
    manifest.write_text(json.dumps({"dates": entries}))
    inventory = saved_parameter_inventory(manifest, config=RollingFitConfig())
    assert inventory["ready"]
    assert not inventory["source_refitting_permitted"]
    assert inventory["dates_using_native_terminal_pricing"] == dates
    selected = select_origin_inventory(inventory, origin=dates[-1], window=4)
    assert selected["ready"]
    assert selected["incomplete_horizon_dates"] == []
    assert selected["dates_using_native_terminal_pricing"] == dates[1:]
    assert selected["dates"][dates[-1]]["path"] == str(tmp_path / f"{dates[-1]}.h5")
    (tmp_path / f"{dates[-1]}.h5").unlink()
    with pytest.raises(FileNotFoundError, match="saved SANOS parameters are missing"):
        saved_parameter_inventory(manifest, config=RollingFitConfig())


def test_estimation_aligns_fixed_expiries_before_ar1() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    supports = np.asarray([[0.5, 0.9, 1.1, 1.5], [0.5, 0.9, 1.1, 1.5]])
    levels = np.broadcast_to(np.arange(4)[:, None, None] * 0.1, (4, 2, 2))
    result = estimate_window(
        dates=dates, log_dlvsq=levels, supports=supports,
        model_expiries_y=np.asarray([1.0, 2.0]) / 252.0,
        forward_over_spot=np.ones((4, 2)), spot_ref=np.ones(4),
        front_densities=np.tile([0.0, 0.5, 0.5, 0.0], (4, 1)), backbone_vol=0.05,
    )
    np.testing.assert_allclose(result["slope"], 1.0, atol=1.0e-14)
    np.testing.assert_allclose(result["intercept"], 0.1, atol=1.0e-14)
    assert len(result["pair_end_dates"]) == 3
    assert float(result["fixed_contract_clipped_query_fraction"]) == 0.0


def test_unfinished_good_fit_does_not_unlock_ar1() -> None:
    report = {
        "date": "2024-01-02", "stop_reason": "checkpointed_stop",
        "fit": {"all": {"inside_percent": 95.0, "quotes": 20}},
        "density": {"maximum_mass_error": 0.0, "maximum_forward_error": 0.0, "minimum_probability": 0.0},
    }
    summary = fit_summary([report], threshold=90.0)
    assert not summary["ready"]
    assert summary["unfinished_dates"] == ["2024-01-02"]


def test_joint_diffusion_preserves_spot_covariance_and_rank_two_variance() -> None:
    generator = np.random.default_rng(15)
    scores = generator.normal(size=30)
    scores -= np.mean(scores)
    scores /= np.std(scores, ddof=1)
    residual = generator.normal(size=30)
    residual -= np.mean(residual)
    residual -= scores * np.dot(scores, residual) / np.dot(scores, scores)
    loading = generator.normal(size=(2, 3, 2))
    innovations = (scores[:, None, None] * loading[None, ..., 0] + residual[:, None, None] * loading[None, ..., 1]) / np.sqrt(252.0)
    result = joint_innovation_loadings(innovations, scores, factor_count=2)
    actual = result["q_log_dlvsq_loadings"].reshape(6, 2)
    np.testing.assert_allclose(actual[:, 0], loading[..., 0].reshape(6), atol=1.0e-13)
    np.testing.assert_allclose(actual @ actual.T, np.cov(innovations.reshape(30, 6), rowvar=False) * 252.0, atol=1.0e-12)
    assert float(result["diffusion_explained_variance_fraction"]) == pytest.approx(1.0)
