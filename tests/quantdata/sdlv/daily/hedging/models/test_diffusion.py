from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    SanosFixedGridHistory,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion import (
    adapter as diffusion_adapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion import (
    training as diffusion_training,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    diffusion_live_context_from_history,
    diffusion_sequences_from_history,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    sample_diffusion_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.model import (
    SequentialVolatilityDenoiser,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.process import (
    static_arbitrage_penalty,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.training import (
    fit_diffusion,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.weekly_paths import (
    sample_diffusion_weekly_paths,
)
from quantdata.torch import BaseEnvironment


def _history() -> SanosFixedGridHistory:
    count = 32
    dates = tuple(date(2022, 1, 3) + timedelta(days=index) for index in range(count))
    surfaces = np.log(0.2 + 0.001 * np.arange(count)[:, None, None]) * np.ones(
        (count, 9, 11)
    )
    returns = np.concatenate((np.asarray([np.nan]), np.linspace(-0.01, 0.01, count - 1)))
    return SanosFixedGridHistory(
        dates=dates,
        log_implied_volatilities=surfaces,
        implied_vol_is_interior=np.ones_like(surfaces, dtype=bool),
        forward_log_returns_into_date=returns,
        sessions_from_previous_observation=np.concatenate(
            (np.asarray([0]), np.ones(count - 1, dtype=np.int64))
        ),
    )


def _prefix_history(history: SanosFixedGridHistory, count: int) -> SanosFixedGridHistory:
    return replace(
        history,
        dates=history.dates[:count],
        log_implied_volatilities=history.log_implied_volatilities[:count],
        implied_vol_is_interior=history.implied_vol_is_interior[:count],
        forward_log_returns_into_date=history.forward_log_returns_into_date[:count],
        sessions_from_previous_observation=(
            history.sessions_from_previous_observation[:count]
        ),
    )


def test_diffusion_training_checkpoint_and_sampling(tmp_path: Path) -> None:
    data = diffusion_sequences_from_history(_history())
    config = DiffusionModelConfig(
        label="diffusion-unit",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
        dropout=0.0,
    )
    fit = fit_diffusion(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=DiffusionTrainConfig(
            epochs=2,
            batch_size=4,
            warmup_steps=1,
            checkpoint_epochs=1,
            progress_seconds=300.0,
            mixed_precision=False,
        ),
        environment=BaseEnvironment(precision=32, device="cpu", debug=True),
    )
    generated = sample_diffusion_transitions(
        fit,
        observed_sequence=data.sequences[-1, :-1],
        path_count=3,
        seed=4,
    )

    assert fit.converged
    assert fit.checkpoint_path.exists()
    assert generated.forward_log_returns.shape == (3,)
    assert generated.next_log_implied_volatilities.shape == (3, 9, 11)
    assert generated.return_channel_max_dispersion >= 0.0


def test_diffusion_weekly_recursion_matches_one_step_and_advances(
    tmp_path: Path,
) -> None:
    full = _history()
    history = _prefix_history(full, len(full.dates) - 1)
    data = diffusion_sequences_from_history(full)
    live = diffusion_live_context_from_history(
        history,
        as_of_date=history.dates[-1],
        forecast_date=full.dates[-1],
    )
    config = DiffusionModelConfig(
        label="diffusion-weekly-unit",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
        dropout=0.0,
    )
    fit = fit_diffusion(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=DiffusionTrainConfig(
            epochs=1,
            batch_size=4,
            warmup_steps=0,
            checkpoint_epochs=1,
            progress_seconds=300.0,
            mixed_precision=False,
        ),
        environment=BaseEnvironment(precision=32, device="cpu", debug=True),
    )
    one_step = sample_diffusion_transitions(
        fit,
        observed_sequence=live.sequence,
        path_count=3,
        seed=23,
    )
    weekly = sample_diffusion_weekly_paths(
        fit,
        live=live,
        path_count=3,
        step_count=3,
        seed=23,
        return_dispersion_tolerance=1.0,
    )

    np.testing.assert_allclose(
        weekly.forward_log_returns[:, 0],
        one_step.forward_log_returns,
    )
    np.testing.assert_allclose(
        weekly.next_log_implied_volatilities[:, 0],
        one_step.next_log_implied_volatilities,
    )
    assert weekly.forward_log_returns.shape == (3, 3)
    assert weekly.next_log_implied_volatilities.shape == (3, 3, 9, 11)


def test_diffusion_live_context_ends_at_declared_as_of_date() -> None:
    full = _history()
    training = diffusion_sequences_from_history(full)
    history = replace(
        full,
        dates=full.dates[:-1],
        log_implied_volatilities=full.log_implied_volatilities[:-1],
        implied_vol_is_interior=full.implied_vol_is_interior[:-1],
        forward_log_returns_into_date=full.forward_log_returns_into_date[:-1],
        sessions_from_previous_observation=full.sessions_from_previous_observation[:-1],
    )
    live = diffusion_live_context_from_history(
        history,
        as_of_date=history.dates[-1],
        forecast_date=full.dates[-1],
    )

    assert live.as_of_date == history.dates[-1]
    assert live.forecast_date == training.target_dates[-1]
    np.testing.assert_allclose(live.sequence, training.sequences[-1, :-1])

    with pytest.raises(ValueError, match="not as-of date"):
        diffusion_live_context_from_history(
            history,
            as_of_date=full.dates[-1],
            forecast_date=full.dates[-1] + timedelta(days=1),
        )


def test_static_arbitrage_penalty_is_differentiable() -> None:
    config = DiffusionModelConfig(
        label="diffusion-penalty",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
    )
    surface = torch.full((2, 9, 11), np.log(0.2), dtype=torch.float64, requires_grad=True)
    penalty = static_arbitrage_penalty(surface, config=config)
    penalty.backward()

    assert torch.isfinite(penalty)
    assert surface.grad is not None
    assert torch.all(torch.isfinite(surface.grad))


def test_diffusion_fine_tuning_loads_base_and_trains_only_adapter(
    tmp_path: Path,
) -> None:
    data = diffusion_sequences_from_history(_history())
    common_geometry = {
        "model_dimension": 16,
        "transformer_layers": 1,
        "attention_heads": 4,
        "diffusion_steps": 4,
        "dropout": 0.0,
    }
    train_config = DiffusionTrainConfig(
        epochs=1,
        batch_size=4,
        warmup_steps=0,
        checkpoint_epochs=1,
        progress_seconds=300.0,
        mixed_precision=False,
    )
    base = fit_diffusion(
        data,
        model_config=DiffusionModelConfig(
            label="diffusion-base-unit",
            **common_geometry,
        ),
        checkpoint_directory=tmp_path,
        train_config=train_config,
        environment=BaseEnvironment(precision=32, device="cpu", debug=True),
    )
    fine_tuned = fit_diffusion(
        data,
        model_config=DiffusionModelConfig(
            label="diffusion-ft-unit",
            variant=DiffusionVariant.AD_SEQ_VOL_FT,
            **common_geometry,
        ),
        checkpoint_directory=tmp_path,
        train_config=train_config,
        base_checkpoint=base.checkpoint_path,
        environment=BaseEnvironment(precision=32, device="cpu", debug=True),
    )

    trainable_names = {
        name
        for name, parameter in fine_tuned.model.named_parameters()
        if parameter.requires_grad
    }
    assert fine_tuned.converged
    assert trainable_names == {
        "output_adapter.down.weight",
        "output_adapter.up.weight",
    }
    assert torch.any(fine_tuned.model.output_adapter.up.weight != 0.0)


def test_diffusion_fine_tuning_rebases_historical_base_and_rejects_future(
    tmp_path: Path,
) -> None:
    history = _history()
    historical_data = diffusion_sequences_from_history(_prefix_history(history, 31))
    current_data = diffusion_sequences_from_history(history)
    geometry = {
        "model_dimension": 16,
        "transformer_layers": 1,
        "attention_heads": 4,
        "diffusion_steps": 4,
        "dropout": 0.0,
    }
    train_config = DiffusionTrainConfig(
        epochs=1,
        batch_size=4,
        warmup_steps=0,
        checkpoint_epochs=1,
        progress_seconds=300.0,
        mixed_precision=False,
    )
    environment = BaseEnvironment(precision=32, device="cpu", debug=True)
    historical_base = fit_diffusion(
        historical_data,
        model_config=DiffusionModelConfig(label="historical-base", **geometry),
        checkpoint_directory=tmp_path / "historical-base",
        train_config=train_config,
        environment=environment,
    )
    fine_tuned = fit_diffusion(
        current_data,
        model_config=DiffusionModelConfig(
            label="current-ft",
            variant=DiffusionVariant.AD_SEQ_VOL_FT,
            **geometry,
        ),
        checkpoint_directory=tmp_path / "current-ft",
        train_config=train_config,
        base_checkpoint=historical_base.checkpoint_path,
        environment=environment,
    )
    validation_count = max(1, int(round(len(current_data.sequences) * 0.2)))
    current_mean = np.mean(
        current_data.sequences[:-validation_count],
        axis=(0, 1, 3, 4),
    )

    np.testing.assert_allclose(fine_tuned.model.channel_mean, current_mean)

    with pytest.raises(RuntimeError, match="changed before fitting"):
        fit_diffusion(
            current_data,
            model_config=DiffusionModelConfig(
                label="wrong-base-sha-ft",
                variant=DiffusionVariant.AD_SEQ_VOL_FT,
                **geometry,
            ),
            checkpoint_directory=tmp_path / "wrong-base-sha-ft",
            train_config=train_config,
            base_checkpoint=historical_base.checkpoint_path,
            expected_base_checkpoint_sha256="0" * 64,
            environment=environment,
        )

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        fit_diffusion(
            current_data,
            model_config=DiffusionModelConfig(
                label="wrong-base-geometry-ft",
                variant=DiffusionVariant.AD_SEQ_VOL_FT,
                attention_heads=2,
                **{name: value for name, value in geometry.items() if name != "attention_heads"},
            ),
            checkpoint_directory=tmp_path / "wrong-base-geometry-ft",
            train_config=train_config,
            base_checkpoint=historical_base.checkpoint_path,
            environment=environment,
        )

    future_base = fit_diffusion(
        current_data,
        model_config=DiffusionModelConfig(label="future-base", **geometry),
        checkpoint_directory=tmp_path / "future-base",
        train_config=train_config,
        environment=environment,
    )
    with pytest.raises(ValueError, match="future training data"):
        fit_diffusion(
            historical_data,
            model_config=DiffusionModelConfig(
                label="historical-ft",
                variant=DiffusionVariant.AD_SEQ_VOL_FT,
                **geometry,
            ),
            checkpoint_directory=tmp_path / "historical-ft",
            train_config=train_config,
            base_checkpoint=future_base.checkpoint_path,
            environment=environment,
        )


def test_diffusion_checkpoint_identity_includes_exact_training_data() -> None:
    data = diffusion_sequences_from_history(_history())
    sequences = data.sequences.copy()
    sequences[0, 0, 0, 0, 0] += 1.0e-8
    changed = replace(data, sequences=sequences)

    assert diffusion_training._diffusion_data_fingerprint(
        data
    ) != diffusion_training._diffusion_data_fingerprint(changed)


def test_diffusion_normalization_rebase_preserves_economic_map() -> None:
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = DiffusionModelConfig(
        label="diffusion-rebase",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
        dropout=0.0,
    )
    old_mean = torch.tensor([-1.2, 0.03], dtype=torch.float64)
    old_scale = torch.tensor([0.5, 0.02], dtype=torch.float64)
    new_mean = torch.tensor([-1.0, -0.01], dtype=torch.float64)
    new_scale = torch.tensor([0.7, 0.04], dtype=torch.float64)
    source = SequentialVolatilityDenoiser(
        environment,
        config=config,
        channel_mean=old_mean,
        channel_scale=old_scale,
    ).eval()
    current = SequentialVolatilityDenoiser(
        environment,
        config=config,
        channel_mean=new_mean,
        channel_scale=new_scale,
    ).eval()
    economic = torch.randn(2, 22, 2, 9, 11, dtype=torch.float64)
    steps = torch.tensor([1, 2], dtype=torch.long)
    source_economic = source.denormalize(source(source.normalize(economic), steps))

    current.load_state_dict(
        diffusion_training._rebase_denoiser_state(
            source.state_dict(),
            current_state=current.state_dict(),
        )
    )
    current_economic = current.denormalize(current(current.normalize(economic), steps))

    torch.testing.assert_close(current_economic, source_economic)


def test_diffusion_warm_start_is_historical_and_auditable(tmp_path: Path) -> None:
    history = _history()
    source_data = diffusion_sequences_from_history(_prefix_history(history, 31))
    current_data = diffusion_sequences_from_history(history)
    config = DiffusionModelConfig(
        label="diffusion-rolling-warm-start",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
        dropout=0.0,
    )
    train_config = DiffusionTrainConfig(
        epochs=1,
        batch_size=4,
        warmup_steps=0,
        checkpoint_epochs=1,
        progress_seconds=300.0,
        mixed_precision=False,
    )
    environment = BaseEnvironment(precision=32, device="cpu", debug=True)
    source = fit_diffusion(
        source_data,
        model_config=config,
        checkpoint_directory=tmp_path / "source",
        train_config=train_config,
        environment=environment,
    )
    current = fit_diffusion(
        current_data,
        model_config=config,
        checkpoint_directory=tmp_path / "current",
        train_config=train_config,
        warm_start_checkpoint=source.checkpoint_path,
        environment=environment,
    )
    validation_count = max(1, int(round(len(current_data.sequences) * 0.2)))
    expected_mean = np.mean(
        current_data.sequences[:-validation_count],
        axis=(0, 1, 3, 4),
    )

    assert current.warm_start_checkpoint == source.checkpoint_path
    assert current.warm_start_sha256 is not None
    assert current.training_end_date == current_data.target_dates[-1]
    np.testing.assert_allclose(current.model.channel_mean, expected_mean)
    payload = torch.load(current.checkpoint_path, weights_only=False)
    assert payload["warm_start_source_sha256"] == current.warm_start_sha256
    assert payload["training_end_date"] == current_data.target_dates[-1].isoformat()

    with pytest.raises(ValueError, match="must predate"):
        fit_diffusion(
            source_data,
            model_config=config,
            checkpoint_directory=tmp_path / "invalid",
            train_config=train_config,
            warm_start_checkpoint=source.checkpoint_path,
            environment=environment,
        )


def test_diffusion_epoch_checkpoint_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Interrupted(RuntimeError):
        pass

    data = diffusion_sequences_from_history(_history())
    config = DiffusionModelConfig(
        label="diffusion-exact-restart",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
        dropout=0.0,
    )
    train_config = DiffusionTrainConfig(
        epochs=2,
        batch_size=4,
        warmup_steps=1,
        checkpoint_epochs=1,
        progress_seconds=300.0,
        mixed_precision=False,
        seed=19,
    )
    environment = BaseEnvironment(precision=32, device="cpu", debug=True)
    uninterrupted = fit_diffusion(
        data,
        model_config=config,
        checkpoint_directory=tmp_path / "uninterrupted",
        train_config=train_config,
        environment=environment,
    )
    save_checkpoint = diffusion_training._save_checkpoint

    def interrupt_after_first_checkpoint(*args: object, **kwargs: object) -> None:
        save_checkpoint(*args, **kwargs)
        if kwargs["epoch"] == 0:
            raise Interrupted

    monkeypatch.setattr(
        diffusion_training,
        "_save_checkpoint",
        interrupt_after_first_checkpoint,
    )
    with pytest.raises(Interrupted):
        fit_diffusion(
            data,
            model_config=config,
            checkpoint_directory=tmp_path / "resumed",
            train_config=train_config,
            environment=environment,
        )
    monkeypatch.setattr(diffusion_training, "_save_checkpoint", save_checkpoint)
    load_locations: list[object] = []
    torch_load = torch.load

    def recording_load(*args: object, **kwargs: object) -> object:
        load_locations.append(kwargs.get("map_location"))
        return torch_load(*args, **kwargs)

    monkeypatch.setattr(diffusion_training.torch, "load", recording_load)
    resumed = fit_diffusion(
        data,
        model_config=config,
        checkpoint_directory=tmp_path / "resumed",
        train_config=train_config,
        environment=environment,
    )

    assert load_locations == ["cpu"]
    for name, value in uninterrupted.model.state_dict().items():
        assert value.equal(resumed.model.state_dict()[name]), name
    for name, value in uninterrupted.ema_model.state_dict().items():
        assert value.equal(resumed.ema_model.state_dict()[name]), name


def test_diffusion_checkpoint_rejects_changed_split(tmp_path: Path) -> None:
    data = diffusion_sequences_from_history(_history())
    config = DiffusionModelConfig(
        label="diffusion-split-identity",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
        dropout=0.0,
    )
    common = {
        "epochs": 1,
        "batch_size": 4,
        "warmup_steps": 0,
        "checkpoint_epochs": 1,
        "progress_seconds": 300.0,
        "mixed_precision": False,
    }
    environment = BaseEnvironment(precision=32, device="cpu", debug=True)
    fit_diffusion(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=DiffusionTrainConfig(
            validation_fraction=0.2,
            seed=13,
            **common,
        ),
        environment=environment,
    )

    with pytest.raises(ValueError, match="seed, validation_fraction"):
        fit_diffusion(
            data,
            model_config=config,
            checkpoint_directory=tmp_path,
            train_config=DiffusionTrainConfig(
                validation_fraction=0.3,
                seed=99,
                **common,
            ),
            environment=environment,
        )


def test_diffusion_changed_optimizer_requires_opt_in_and_more_epochs(
    tmp_path: Path,
) -> None:
    data = diffusion_sequences_from_history(_history())
    config = DiffusionModelConfig(
        label="diffusion-explicit-warm-start",
        model_dimension=16,
        transformer_layers=1,
        attention_heads=4,
        diffusion_steps=4,
        dropout=0.0,
    )
    common = {
        "batch_size": 4,
        "warmup_steps": 0,
        "checkpoint_epochs": 1,
        "progress_seconds": 300.0,
        "mixed_precision": False,
        "seed": 19,
    }
    environment = BaseEnvironment(precision=32, device="cpu", debug=True)
    fit_diffusion(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=DiffusionTrainConfig(
            epochs=1,
            learning_rate=1.0e-3,
            **common,
        ),
        environment=environment,
    )

    with pytest.raises(ValueError, match="allow_changed_config_warm_start"):
        fit_diffusion(
            data,
            model_config=config,
            checkpoint_directory=tmp_path,
            train_config=DiffusionTrainConfig(
                epochs=2,
                learning_rate=2.0e-3,
                **common,
            ),
            environment=environment,
        )
    with pytest.warns(UserWarning, match="learning_rate"):
        resumed = fit_diffusion(
            data,
            model_config=config,
            checkpoint_directory=tmp_path,
            train_config=DiffusionTrainConfig(
                epochs=2,
                learning_rate=2.0e-3,
                allow_changed_config_warm_start=True,
                **common,
            ),
            environment=environment,
        )

    assert resumed.completed_epochs == 2
    assert resumed.converged


def test_diffusion_daily_book_fits_and_generates_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"fit": 0, "generate": 0, "evaluate": 0}
    forecast_date = date(2025, 1, 3)
    monkeypatch.setattr(
        diffusion_adapter,
        "require_public_hedging_panel",
        lambda panel: None,
    )
    monkeypatch.setattr(
        diffusion_adapter,
        "build_daily_digital_sample",
        lambda panel, *, origin_date, target: SimpleNamespace(
            target=target,
            hedge_quotes=(),
            transition=SimpleNamespace(next_date=forecast_date),
        ),
    )

    def fit(*args, **kwargs):
        calls["fit"] += 1
        checkpoint = tmp_path / "fit.pt"
        checkpoint.write_bytes(b"diffusion")
        return SimpleNamespace(
            result=SimpleNamespace(checkpoint_path=checkpoint),
            data=SimpleNamespace(start_dates=()),
        )

    monkeypatch.setattr(diffusion_adapter, "fit_public_diffusion", fit)
    monkeypatch.setattr(
        diffusion_adapter,
        "build_diffusion_live_context",
        lambda *args, **kwargs: SimpleNamespace(
            forecast_date=forecast_date,
            sequence=np.zeros((2, 9, 11)),
        ),
    )

    def generate(*args, **kwargs):
        calls["generate"] += 1
        return SimpleNamespace(
            forward_log_returns=np.zeros(4),
            next_log_implied_volatilities=np.zeros((4, 9, 11)),
            return_channel_max_dispersion=0.0,
        )

    monkeypatch.setattr(diffusion_adapter, "sample_diffusion_transitions", generate)

    def evaluate(sample, **kwargs):
        calls["evaluate"] += 1
        return sample.target.label

    monkeypatch.setattr(diffusion_adapter, "_evaluate_diffusion_target", evaluate)
    results = diffusion_adapter.run_diffusion_daily_book(
        object(),
        origin_date=date(2025, 1, 2),
        model_config=DiffusionModelConfig(
            label="diffusion-book",
            model_dimension=16,
            transformer_layers=1,
            attention_heads=4,
            diffusion_steps=4,
        ),
        checkpoint_directory=tmp_path,
    )
    assert len(results) == 20
    assert calls == {"fit": 1, "generate": 1, "evaluate": 20}
