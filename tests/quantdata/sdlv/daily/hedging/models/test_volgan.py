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
from quantdata.sdlv.daily.front_spot.hedging.models.volgan import adapter as volgan_adapter
from quantdata.sdlv.daily.front_spot.hedging.models.volgan import training as volgan_training
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    volgan_examples_from_history,
    volgan_live_condition_from_history,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.hedger import (
    sample_volgan_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.model import (
    VolganDiscriminator,
    VolganGenerator,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.training import fit_volgan
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.weekly_paths import (
    sample_volgan_weekly_paths,
)
from quantdata.torch import BaseEnvironment


def _history() -> SanosFixedGridHistory:
    count = 30
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


def test_volgan_training_checkpoint_and_sampling(tmp_path: Path) -> None:
    data = volgan_examples_from_history(_history())
    config = VolganModelConfig(label="volgan-unit")
    fit = fit_volgan(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=VolganTrainConfig(
            epochs=3,
            gradient_matching_epochs=1,
            batch_size=4,
            checkpoint_epochs=1,
            progress_seconds=300.0,
        ),
        environment=BaseEnvironment(precision=64, device="cpu", debug=True),
    )
    generated = sample_volgan_transitions(
        fit,
        condition=data.conditions[-1],
        path_count=5,
        seed=9,
    )

    assert fit.converged
    assert fit.checkpoint_path.exists()
    assert generated.forward_log_returns.shape == (5,)
    assert generated.next_log_implied_volatilities.shape == (5, 9, 11)


def test_volgan_weekly_recursion_matches_one_step_and_advances(tmp_path: Path) -> None:
    full = _history()
    history = _prefix_history(full, len(full.dates) - 1)
    data = volgan_examples_from_history(full)
    live = volgan_live_condition_from_history(
        history,
        as_of_date=history.dates[-1],
        forecast_date=full.dates[-1],
    )
    config = VolganModelConfig(label="volgan-weekly-unit")
    fit = fit_volgan(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=VolganTrainConfig(
            epochs=1,
            gradient_matching_epochs=1,
            batch_size=4,
            checkpoint_epochs=1,
            progress_seconds=300.0,
        ),
        environment=BaseEnvironment(precision=64, device="cpu", debug=True),
    )
    one_step = sample_volgan_transitions(
        fit,
        condition=live.values,
        path_count=4,
        seed=17,
    )
    weekly = sample_volgan_weekly_paths(
        fit,
        live=live,
        path_count=4,
        step_count=3,
        seed=17,
    )

    np.testing.assert_allclose(
        weekly.forward_log_returns[:, 0],
        one_step.forward_log_returns,
    )
    np.testing.assert_allclose(
        weekly.next_log_implied_volatilities[:, 0],
        one_step.next_log_implied_volatilities,
    )
    assert weekly.forward_log_returns.shape == (4, 3)
    assert weekly.next_log_implied_volatilities.shape == (4, 3, 9, 11)


def test_volgan_live_condition_ends_at_declared_as_of_date() -> None:
    full = _history()
    training = volgan_examples_from_history(full)
    history = replace(
        full,
        dates=full.dates[:-1],
        log_implied_volatilities=full.log_implied_volatilities[:-1],
        implied_vol_is_interior=full.implied_vol_is_interior[:-1],
        forward_log_returns_into_date=full.forward_log_returns_into_date[:-1],
        sessions_from_previous_observation=full.sessions_from_previous_observation[:-1],
    )
    live = volgan_live_condition_from_history(
        history,
        as_of_date=history.dates[-1],
        forecast_date=full.dates[-1],
    )

    assert live.as_of_date == training.condition_dates[-1]
    assert live.forecast_date == training.target_dates[-1]
    np.testing.assert_allclose(live.values, training.conditions[-1])
    np.testing.assert_allclose(
        live.trailing_forward_log_returns,
        history.forward_log_returns_into_date[-21:],
    )

    with pytest.raises(ValueError, match="not as-of date"):
        volgan_live_condition_from_history(
            history,
            as_of_date=full.dates[-1],
            forecast_date=full.dates[-1] + timedelta(days=1),
        )


def test_volgan_checkpoint_identity_includes_exact_training_data() -> None:
    data = volgan_examples_from_history(_history())
    targets = data.targets.copy()
    targets[0, 0] += 1.0e-8
    changed = replace(data, targets=targets)

    assert volgan_training._volgan_data_fingerprint(
        data
    ) != volgan_training._volgan_data_fingerprint(changed)


def test_volgan_normalization_rebase_preserves_economic_map() -> None:
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = VolganModelConfig(label="volgan-rebase")
    old_condition_mean = torch.linspace(-0.2, 0.2, config.condition_dimension)
    old_condition_scale = torch.linspace(0.7, 1.3, config.condition_dimension)
    old_target_mean = torch.linspace(-0.1, 0.1, config.output_dimension)
    old_target_scale = torch.linspace(0.8, 1.2, config.output_dimension)
    new_condition_mean = old_condition_mean + 0.07
    new_condition_scale = old_condition_scale * 1.15
    new_target_mean = old_target_mean - 0.03
    new_target_scale = old_target_scale * 0.9
    source_generator = VolganGenerator(
        environment,
        config=config,
        condition_mean=old_condition_mean,
        condition_scale=old_condition_scale,
        target_mean=old_target_mean,
        target_scale=old_target_scale,
    )
    source_discriminator = VolganDiscriminator(environment, config=config)
    current_generator = VolganGenerator(
        environment,
        config=config,
        condition_mean=new_condition_mean,
        condition_scale=new_condition_scale,
        target_mean=new_target_mean,
        target_scale=new_target_scale,
    )
    current_discriminator = VolganDiscriminator(environment, config=config)
    conditions = torch.randn(3, config.condition_dimension, dtype=torch.float64)
    noise = torch.randn(3, config.noise_dimension, dtype=torch.float64)
    economic_targets = torch.randn(3, config.output_dimension, dtype=torch.float64)
    source_generated = source_generator.decode(source_generator(noise, conditions))
    source_logits = source_discriminator(
        (conditions - old_condition_mean) / old_condition_scale,
        (economic_targets - old_target_mean) / old_target_scale,
    )

    current_generator.load_state_dict(
        volgan_training._rebase_generator_state(
            source_generator.state_dict(),
            current_state=current_generator.state_dict(),
            noise_dimension=config.noise_dimension,
        )
    )
    current_discriminator.load_state_dict(
        volgan_training._rebase_discriminator_state(
            source_discriminator.state_dict(),
            source_generator_state=source_generator.state_dict(),
            current_generator_state=current_generator.state_dict(),
            condition_dimension=config.condition_dimension,
        )
    )
    current_generated = current_generator.decode(current_generator(noise, conditions))
    current_logits = current_discriminator(
        (conditions - new_condition_mean) / new_condition_scale,
        (economic_targets - new_target_mean) / new_target_scale,
    )

    torch.testing.assert_close(current_generated, source_generated)
    torch.testing.assert_close(current_logits, source_logits)


def test_volgan_warm_start_is_historical_and_auditable(tmp_path: Path) -> None:
    history = _history()
    source_data = volgan_examples_from_history(_prefix_history(history, 29))
    current_data = volgan_examples_from_history(history)
    config = VolganModelConfig(label="volgan-rolling-warm-start")
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    manifest_sha256 = "a" * 64
    source = fit_volgan(
        source_data,
        model_config=config,
        checkpoint_directory=tmp_path / "source",
        train_config=VolganTrainConfig(
            epochs=1,
            gradient_matching_epochs=1,
            batch_size=4,
            checkpoint_epochs=1,
            progress_seconds=300.0,
        ),
        dataset_manifest_sha256=manifest_sha256,
        environment=environment,
    )
    current = fit_volgan(
        current_data,
        model_config=config,
        checkpoint_directory=tmp_path / "current",
        train_config=VolganTrainConfig(
            epochs=1,
            gradient_matching_epochs=0,
            batch_size=4,
            checkpoint_epochs=1,
            progress_seconds=300.0,
        ),
        warm_start_checkpoint=source.checkpoint_path,
        dataset_manifest_sha256=manifest_sha256,
        environment=environment,
    )
    expected_mean = np.mean(
        current_data.conditions[: len(current_data.conditions) - 2],
        axis=0,
    )

    assert current.warm_start_checkpoint == source.checkpoint_path
    assert current.warm_start_sha256 is not None
    assert current.training_end_date == current_data.target_dates[-1]
    np.testing.assert_allclose(current.generator.condition_mean, expected_mean)
    payload = torch.load(current.checkpoint_path, weights_only=False)
    assert payload["warm_start_source_sha256"] == current.warm_start_sha256
    assert payload["training_end_date"] == current_data.target_dates[-1].isoformat()
    assert payload["dataset_manifest_sha256"] == manifest_sha256

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        fit_volgan(
            current_data,
            model_config=config,
            checkpoint_directory=tmp_path / "wrong-public-manifest",
            train_config=VolganTrainConfig(
                epochs=1,
                gradient_matching_epochs=0,
                batch_size=4,
                checkpoint_epochs=1,
                progress_seconds=300.0,
            ),
            warm_start_checkpoint=source.checkpoint_path,
            dataset_manifest_sha256="b" * 64,
            environment=environment,
        )

    with pytest.raises(RuntimeError, match="changed before fitting"):
        fit_volgan(
            current_data,
            model_config=config,
            checkpoint_directory=tmp_path / "wrong-source-sha",
            train_config=VolganTrainConfig(
                epochs=1,
                gradient_matching_epochs=0,
                batch_size=4,
                checkpoint_epochs=1,
                progress_seconds=300.0,
            ),
            warm_start_checkpoint=source.checkpoint_path,
            expected_warm_start_sha256="0" * 64,
            environment=environment,
        )

    with pytest.raises(ValueError, match="must predate"):
        fit_volgan(
            source_data,
            model_config=config,
            checkpoint_directory=tmp_path / "invalid",
            train_config=VolganTrainConfig(
                epochs=1,
                gradient_matching_epochs=0,
                batch_size=4,
                checkpoint_epochs=1,
                progress_seconds=300.0,
            ),
            warm_start_checkpoint=source.checkpoint_path,
            dataset_manifest_sha256=manifest_sha256,
            environment=environment,
        )


def test_volgan_epoch_checkpoint_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Interrupted(RuntimeError):
        pass

    data = volgan_examples_from_history(_history())
    config = VolganModelConfig(label="volgan-exact-restart")
    train_config = VolganTrainConfig(
        epochs=3,
        gradient_matching_epochs=1,
        batch_size=4,
        checkpoint_epochs=1,
        progress_seconds=300.0,
        seed=13,
    )
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    uninterrupted = fit_volgan(
        data,
        model_config=config,
        checkpoint_directory=tmp_path / "uninterrupted",
        train_config=train_config,
        environment=environment,
    )
    save_checkpoint = volgan_training._save_checkpoint

    def interrupt_after_first_checkpoint(*args: object, **kwargs: object) -> None:
        save_checkpoint(*args, **kwargs)
        if kwargs["epoch"] == 0:
            raise Interrupted

    monkeypatch.setattr(
        volgan_training,
        "_save_checkpoint",
        interrupt_after_first_checkpoint,
    )
    with pytest.raises(Interrupted):
        fit_volgan(
            data,
            model_config=config,
            checkpoint_directory=tmp_path / "resumed",
            train_config=train_config,
            environment=environment,
        )
    monkeypatch.setattr(volgan_training, "_save_checkpoint", save_checkpoint)
    resumed = fit_volgan(
        data,
        model_config=config,
        checkpoint_directory=tmp_path / "resumed",
        train_config=train_config,
        environment=environment,
    )

    for name, value in uninterrupted.generator.state_dict().items():
        assert value.equal(resumed.generator.state_dict()[name]), name
    for name, value in uninterrupted.discriminator.state_dict().items():
        assert value.equal(resumed.discriminator.state_dict()[name]), name


def test_volgan_checkpoint_rejects_changed_split(tmp_path: Path) -> None:
    data = volgan_examples_from_history(_history())
    config = VolganModelConfig(label="volgan-split-identity")
    common = {
        "epochs": 1,
        "gradient_matching_epochs": 1,
        "batch_size": 4,
        "checkpoint_epochs": 1,
        "progress_seconds": 300.0,
    }
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    fit_volgan(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=VolganTrainConfig(
            validation_fraction=0.2,
            seed=13,
            **common,
        ),
        environment=environment,
    )

    with pytest.raises(ValueError, match="seed, validation_fraction"):
        fit_volgan(
            data,
            model_config=config,
            checkpoint_directory=tmp_path,
            train_config=VolganTrainConfig(
                validation_fraction=0.3,
                seed=99,
                **common,
            ),
            environment=environment,
        )


def test_volgan_changed_optimizer_requires_opt_in_and_more_epochs(
    tmp_path: Path,
) -> None:
    data = volgan_examples_from_history(_history())
    config = VolganModelConfig(label="volgan-explicit-warm-start")
    common = {
        "gradient_matching_epochs": 1,
        "batch_size": 4,
        "checkpoint_epochs": 1,
        "progress_seconds": 300.0,
        "seed": 13,
    }
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    fit_volgan(
        data,
        model_config=config,
        checkpoint_directory=tmp_path,
        train_config=VolganTrainConfig(
            epochs=1,
            generator_learning_rate=1.0e-4,
            **common,
        ),
        environment=environment,
    )

    with pytest.raises(ValueError, match="allow_changed_config_warm_start"):
        fit_volgan(
            data,
            model_config=config,
            checkpoint_directory=tmp_path,
            train_config=VolganTrainConfig(
                epochs=2,
                generator_learning_rate=2.0e-4,
                **common,
            ),
            environment=environment,
        )
    with pytest.warns(UserWarning, match="generator_learning_rate"):
        resumed = fit_volgan(
            data,
            model_config=config,
            checkpoint_directory=tmp_path,
            train_config=VolganTrainConfig(
                epochs=2,
                generator_learning_rate=2.0e-4,
                allow_changed_config_warm_start=True,
                **common,
            ),
            environment=environment,
        )

    assert resumed.completed_epochs == 2
    assert resumed.converged


def test_volgan_daily_book_fits_and_generates_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"fit": 0, "generate": 0, "evaluate": 0}
    forecast_date = date(2025, 1, 3)
    monkeypatch.setattr(
        volgan_adapter,
        "require_public_hedging_panel",
        lambda panel: None,
    )
    monkeypatch.setattr(
        volgan_adapter,
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
        checkpoint.write_bytes(b"volgan")
        return SimpleNamespace(
            result=SimpleNamespace(checkpoint_path=checkpoint),
            data=SimpleNamespace(condition_dates=()),
        )

    monkeypatch.setattr(volgan_adapter, "fit_public_volgan", fit)
    monkeypatch.setattr(
        volgan_adapter,
        "build_volgan_live_condition",
        lambda *args, **kwargs: SimpleNamespace(
            forecast_date=forecast_date,
            values=np.zeros(2),
        ),
    )

    def generate(*args, **kwargs):
        calls["generate"] += 1
        return SimpleNamespace(
            forward_log_returns=np.zeros(4),
            next_log_implied_volatilities=np.zeros((4, 9, 11)),
        )

    monkeypatch.setattr(volgan_adapter, "sample_volgan_transitions", generate)

    def evaluate(sample, **kwargs):
        calls["evaluate"] += 1
        return sample.target.label

    monkeypatch.setattr(volgan_adapter, "_evaluate_volgan_target", evaluate)
    results = volgan_adapter.run_volgan_daily_book(
        object(),
        origin_date=date(2025, 1, 2),
        model_config=VolganModelConfig(label="volgan-book"),
        checkpoint_directory=tmp_path,
    )
    assert len(results) == 20
    assert calls == {"fit": 1, "generate": 1, "evaluate": 20}
