"""Standalone CPU checks for FuNVol's public-data research core."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
    SanosFixedGridHistory,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol import (
    FunvolConfig,
    FunvolTrainConfig,
    fit_functional_pca,
    fit_funvol,
    sample_funvol_one_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.arbitrage import (
    screen_generated_iv_surfaces,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.rolling import _validate_records
from quantdata.torch import BaseEnvironment


def _config() -> FunvolConfig:
    return FunvolConfig(
        grid=GenerativeSurfaceGrid(
            strikes_over_forward=(),
            nms_nodes=(-1.0, -0.5, 0.0, 0.5, 1.0),
            expiries_y=(1.0 / 252.0, 5.0 / 252.0, 10.0 / 252.0),
        ),
        functional_order=2,
        factor_count=2,
        lag_count=4,
        gru_layers=1,
        hidden_size=5,
    )


def _history(config: FunvolConfig, count: int = 36) -> SanosFixedGridHistory:
    grid_expiries = np.asarray(config.grid.expiries_y)
    grid_nms = np.asarray(config.grid.nms_nodes)
    expiry_shape = np.sqrt(grid_expiries / grid_expiries[-1])[None, :, None]
    smile_shape = grid_nms[None, None, :] ** 2
    calendar = np.arange(count, dtype=np.float64)[:, None, None]
    volatilities = 0.18 + 0.025 * expiry_shape + 0.01 * smile_shape
    volatilities = volatilities + 0.005 * np.sin(calendar * 0.37) * grid_nms[None, None, :]
    volatilities = volatilities + 0.002 * np.cos(calendar * 0.22)
    return SanosFixedGridHistory(
        dates=tuple(date(2022, 1, 1) + timedelta(days=index) for index in range(count)),
        log_implied_volatilities=np.log(volatilities),
        implied_vol_is_interior=np.ones_like(volatilities, dtype=bool),
        forward_log_returns_into_date=np.r_[np.nan, 0.01 * np.sin(np.arange(1, count))],
        sessions_from_previous_observation=np.r_[0, np.ones(count - 1, dtype=np.int64)],
    )


def test_functional_pca_positive_reconstruction_and_training_only_cutoff() -> None:
    config = _config()
    history = _history(config)
    original = np.exp(history.log_implied_volatilities)
    fpc = fit_functional_pca(original[:28], config=config)
    scores = fpc.encode(original[28:])
    reconstruction = fpc.decode(scores)
    assert scores.shape == (8, config.factor_count)
    assert reconstruction.shape == original[28:].shape
    assert np.all(reconstruction > 0.0)
    assert 0.0 <= np.sum(fpc.explained_variance_ratio) <= 1.0 + 1.0e-12
    changed = original.copy()
    changed[28:] *= 3.0
    heldout_changed_fit = fit_functional_pca(changed[:28], config=config)
    np.testing.assert_array_equal(fpc.components, heldout_changed_fit.components)
    with pytest.raises(ValueError, match="positive"):
        fpc.encode(np.zeros_like(original[:1]))


def test_funvol_cpu_fit_sampling_and_no_validation_leakage() -> None:
    config = _config()
    history = _history(config)
    train_config = FunvolTrainConfig(
        epochs=2,
        batch_size=8,
        validation_fraction=0.2,
        seed=17,
    )
    env = BaseEnvironment(precision=64, device="cpu")
    fit = fit_funvol(history, config=config, train_config=train_config, environment=env)
    assert fit.training_target_count > fit.validation_target_count > 0
    assert np.isfinite(fit.training_loss)
    assert np.isfinite(fit.validation_loss)
    changed_surfaces = history.log_implied_volatilities.copy()
    changed_surfaces[fit.last_training_surface_index + 1 :] += 0.25
    changed_history = replace(history, log_implied_volatilities=changed_surfaces)
    changed_fit = fit_funvol(
        changed_history,
        config=config,
        train_config=train_config,
        environment=env,
    )
    np.testing.assert_allclose(fit.fpc.components, changed_fit.fpc.components, atol=0.0)
    np.testing.assert_allclose(fit.state_center, changed_fit.state_center, atol=0.0)
    first_returns, first_volatilities = sample_funvol_one_day(
        fit,
        history,
        path_count=12,
        seed=19,
    )
    second_returns, second_volatilities = sample_funvol_one_day(
        fit,
        history,
        path_count=12,
        seed=19,
    )
    np.testing.assert_array_equal(first_returns, second_returns)
    np.testing.assert_array_equal(first_volatilities, second_volatilities)
    assert first_returns.shape == (12,)
    assert first_volatilities.shape == (12, 3, 5)
    assert np.std(first_returns) > 0.0
    assert np.all(first_volatilities > 0.0)
    assert all(parameter.device.type == "cpu" for parameter in fit.model.parameters())
    assert torch.isfinite(torch.as_tensor(first_returns)).all()


def test_funvol_checkpoint_resume_and_causal_weight_inheritance(tmp_path: Path) -> None:
    config = _config()
    training = FunvolTrainConfig(epochs=1, batch_size=8, seed=31)
    previous_history = _history(config, 36)
    previous_checkpoint = tmp_path / "2022-02-05.pt"
    previous = fit_funvol(
        previous_history,
        config=config,
        train_config=training,
        checkpoint_path=previous_checkpoint,
    )
    resumed = fit_funvol(
        previous_history,
        config=config,
        train_config=training,
        checkpoint_path=previous_checkpoint,
    )
    assert resumed.completed_epochs == 1
    for name, weight in previous.model.state_dict().items():
        torch.testing.assert_close(weight, resumed.model.state_dict()[name], rtol=0, atol=0)

    next_history = _history(config, 37)
    next_checkpoint = tmp_path / "2022-02-06.pt"
    current = fit_funvol(
        next_history,
        config=config,
        train_config=training,
        checkpoint_path=next_checkpoint,
        warm_start_checkpoint=previous_checkpoint,
    )
    assert current.warm_start_date == previous_history.dates[-1].isoformat()
    assert current.warm_start_sha256 is not None
    np.testing.assert_array_equal(current.fpc.components, previous.fpc.components)
    np.testing.assert_array_equal(current.state_center, previous.state_center)
    np.testing.assert_array_equal(current.state_scale, previous.state_scale)
    assert current.completed_epochs == 1
    current_resume = fit_funvol(
        next_history,
        config=config,
        train_config=training,
        checkpoint_path=next_checkpoint,
        warm_start_checkpoint=previous_checkpoint,
    )
    assert current_resume.warm_start_sha256 == current.warm_start_sha256
    shifted_history = replace(
        next_history,
        dates=next_history.dates[1:],
        log_implied_volatilities=next_history.log_implied_volatilities[1:],
        implied_vol_is_interior=next_history.implied_vol_is_interior[1:],
        forward_log_returns_into_date=np.r_[np.nan, next_history.forward_log_returns_into_date[2:]],
        sessions_from_previous_observation=np.r_[
            0, next_history.sessions_from_previous_observation[2:]
        ],
    )
    shifted = fit_funvol(
        shifted_history,
        config=config,
        train_config=training,
        warm_start_checkpoint=previous_checkpoint,
    )
    assert shifted.spot_log_anchor == pytest.approx(
        previous_history.forward_log_returns_into_date[1]
    )

    with pytest.raises(ValueError, match="earlier date"):
        fit_funvol(
            previous_history,
            config=config,
            train_config=training,
            warm_start_checkpoint=next_checkpoint,
        )
    changed = replace(
        previous_history,
        log_implied_volatilities=previous_history.log_implied_volatilities + 0.01,
    )
    with pytest.raises(ValueError, match="identity is incompatible"):
        fit_funvol(
            changed,
            config=config,
            train_config=training,
            checkpoint_path=previous_checkpoint,
        )


def test_funvol_rolling_fit_uses_entire_available_history() -> None:
    config = _config()
    history = _history(config)
    fit = fit_funvol(
        history,
        config=config,
        train_config=FunvolTrainConfig(epochs=1, validation_fraction=0.0),
    )
    assert fit.last_training_surface_index == len(history.dates) - 1
    assert fit.validation_target_count == 0
    assert fit.validation_loss is None
    assert fit.training_target_count == len(history.dates) - config.lag_count


def test_funvol_static_arbitrage_screen_detects_calendar_violation() -> None:
    grid = _config().grid
    flat_volatilities = np.full((4, len(grid.expiries_y), grid.strike_count), 0.2)
    strikes = np.linspace(-0.005, 0.005, 9)
    clean = screen_generated_iv_surfaces(flat_volatilities, grid=grid, log_strikes=strikes)
    assert clean.interior_paths == 4
    assert clean.any_violation_fraction == 0.0

    declining_variance = flat_volatilities.copy()
    declining_variance[:, 1:, :] = 0.05
    breached = screen_generated_iv_surfaces(declining_variance, grid=grid, log_strikes=strikes)
    assert breached.calendar_fraction == 1.0
    assert breached.strike_monotonicity_fraction == 0.0


def test_funvol_dispersion_can_be_recorded_without_waiving_numerical_checks() -> None:
    record = {
        "claim": "digital-call-1d-atm",
        "converged": True,
        "cap_hits": 0,
        "actions": {"spot": 1.0},
        "diagnostics": {
            "warm_start_date": "2024-03-28",
            "generated_return_std": 0.00333,
            "historical_return_std": 0.00778,
        },
    }
    records = tuple(record.copy() for _ in range(10))
    with pytest.raises(RuntimeError, match="dispersion gate failed"):
        _validate_records(records, previous_date="2024-03-28")
    assert not _validate_records(
        records,
        previous_date="2024-03-28",
        allow_spot_dispersion_warning=True,
    )
    bad_records = (dict(record, converged=False),) + records[1:]
    with pytest.raises(RuntimeError, match="numerical gate"):
        _validate_records(
            bad_records,
            previous_date="2024-03-28",
            allow_spot_dispersion_warning=True,
        )
