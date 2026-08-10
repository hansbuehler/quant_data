from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from quantdata.dysanos.grid.rndysanos.conditional_tail import (
    ConditionalTailConfig,
    ConditionalTailQuantileModel,
    empirical_lower_cvar,
    lower_cvar_score,
    lower_quantile_pinball,
    moment_signal_statistics,
    signed_returns,
)
from quantdata.dysanos.grid.rndysanos.conditional_tail_experiment import (
    ConditionalTailTrainConfig,
    prepare_raw_option_tail_data,
    split_tail_data,
    train_conditional_tail,
    valuate_conditional_tail,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def _config(*, precision_paths: int = 50_000) -> ConditionalTailConfig:
    return ConditionalTailConfig(
        label="test-tail",
        state_features=3,
        instruments=2,
        training_paths=precision_paths,
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=8,
            activation="LeakyReLU",
            use_bias=True,
        ),
    )


def _universe(*, samples: int = 8, steps: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(41)
    expiries, strikes, state_variables = 1, 2, 3
    shape = (samples, steps, expiries, strikes)
    return {
        "C": np.full(shape, 8.0, dtype=np.float32),
        "dC": rng.normal(0.0, 0.2, shape).astype(np.float32),
        "dS": rng.normal(0.0, 0.1, (samples, steps)).astype(np.float32),
        "delta": np.full(shape, 0.2, dtype=np.float32),
        "expiries": np.array([[[0.5]]], dtype=np.float32),
        "log_strikes": rng.normal(2.3, 0.1, shape).astype(np.float32),
        "spot": np.full((samples, steps), 10.0, dtype=np.float32),
        "xstate": rng.normal(size=(samples, steps, state_variables)).astype(np.float32),
    }


def test_tail_levels_have_explicit_finite_sample_resolution() -> None:
    config = _config()

    assert config.requested_tail_probabilities == pytest.approx(
        (0.1 / 50_000, 1.0 / 50_000, 2.0 / 50_000, 10.0 / 50_000)
    )
    assert config.effective_tail_path_counts == (1.0, 1.0, 2.0, 10.0)
    assert config.requested_to_unique_level == (0, 0, 1, 2)
    assert len(config.unique_effective_tail_probabilities) == 3
    metadata = config.level_metadata()
    assert not metadata[0]["empirically_distinct"]
    assert metadata[0]["confidence_level"] == pytest.approx(1.0 - 0.1 / 50_000)


def test_empirical_lower_cvar_uses_fractional_boundary_mass() -> None:
    values = torch.tensor([1.0, 2.0, 5.0, 10.0])

    assert empirical_lower_cvar(values, tail_probability=0.25).item() == pytest.approx(1.0)
    assert empirical_lower_cvar(values, tail_probability=0.10).item() == pytest.approx(1.0)
    assert empirical_lower_cvar(values, tail_probability=0.50).item() == pytest.approx(1.5)
    assert empirical_lower_cvar(values, tail_probability=0.375).item() == pytest.approx(4.0 / 3.0)


def test_pinball_and_cvar_scores_cover_long_and_short_with_gradients() -> None:
    returns = torch.tensor([[-3.0], [-1.0], [1.0], [3.0]])
    quantiles = torch.zeros(4, 2, 2, 1, requires_grad=True)
    alpha = torch.tensor([0.25, 0.50])

    loss, by_output = lower_quantile_pinball(
        quantiles=quantiles,
        returns=returns,
        tail_probabilities=alpha,
        normalize_levels=False,
    )
    scores = lower_cvar_score(
        quantiles=quantiles,
        returns=returns,
        tail_probabilities=alpha,
    )
    loss.backward()

    assert by_output.shape == (2, 2, 1)
    assert torch.allclose(by_output[0], by_output[1])
    assert scores.shape == quantiles.shape
    assert quantiles.grad is not None
    assert torch.isfinite(quantiles.grad).all()
    assert torch.count_nonzero(quantiles.grad) > 0
    assert torch.equal(signed_returns(returns)[..., 0, :], returns)
    assert torch.equal(signed_returns(returns)[..., 1, :], -returns)


def test_quantdata_feedforward_model_is_monotone_tied_and_uses_environment() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    config = _config()
    model = ConditionalTailQuantileModel(
        env,
        config=config,
        state_mean=torch.zeros(config.state_features, dtype=env.fdtype),
        state_scale=torch.ones(config.state_features, dtype=env.fdtype),
    )
    z = torch.randn(5, 4, config.state_features, dtype=env.fdtype)
    returns = torch.randn(5, 4, config.instruments, dtype=env.fdtype)
    quantiles = model(z)
    unique = model.unique_quantiles(z)
    loss, _ = model.loss(z, returns)
    training_loss, _ = model.training_loss(z, returns)
    assert torch.allclose(loss, training_loss)
    loss.backward()

    assert isinstance(model, BaseModel)
    assert isinstance(model.network, FeedForward)
    assert next(model.parameters()).dtype == torch.float64
    assert quantiles.shape == (5, 4, 2, 4, config.instruments)
    assert unique.shape == (5, 4, 2, 3, config.instruments)
    assert torch.equal(quantiles[..., 0, :], quantiles[..., 1, :])
    assert torch.all(unique[..., 1:, :] >= unique[..., :-1, :])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_moment_signal_statistics_uses_abs_mean_and_flags_zero_variance() -> None:
    drift = torch.tensor([[[4.0, -4.0, 1.0, 2.0]]])
    second = torch.tensor([[[17.0, 17.0, 2.0, 4.0]]])

    output = moment_signal_statistics(drift, second, g=3.0)

    assert output["hit_comparisons"] == 3
    assert output["positive_hit_comparisons"] == 2
    assert output["negative_hit_comparisons"] == 1
    assert output["positive_sigma_hit_comparisons"] == 2
    assert output["raw_negative_variance_comparisons"] == 0
    assert output["states_with_any_hit"] == 1
    assert output["instruments_with_any_hit"] == 3
    assert output["zero_sigma_comparisons"] == 1
    assert output["zero_sigma_nonzero_mean_comparisons"] == 1
    assert output["max_finite_abs_mean_over_sigma"] == pytest.approx(4.0)


def test_raw_option_preparation_and_split_preserve_training_isolation() -> None:
    env = BaseEnvironment(device="cpu", debug=False)
    universe = _universe()
    data, geometry = prepare_raw_option_tail_data(universe, env=env)
    training, validation = split_tail_data(data, validation_fraction=0.5)

    assert geometry.state_features == 6
    assert data["z"].shape == (8, 2, 6)
    assert data["returns"].shape == (8, 2, 2)
    assert torch.allclose(
        data["returns"], torch.from_numpy(universe["dC"]).reshape(8, 2, 2)
    )
    expected_roundoff = 32.0 * torch.finfo(torch.float32).eps * 16.0
    assert torch.allclose(data["roundoff"], torch.full_like(data["roundoff"], expected_roundoff))
    assert training["z"].shape[0] == validation["z"].shape[0] == 4
    assert torch.equal(training["returns"], data["returns"][:4])
    assert torch.equal(validation["returns"], data["returns"][4:])


def test_one_sided_valuation_counts_roundoff_and_realized_counter_sign() -> None:
    env = BaseEnvironment(device="cpu", debug=False)
    config = ConditionalTailConfig(
        label="one-sided",
        state_features=3,
        instruments=2,
        training_paths=4,
        tail_path_counts=(1.0, 2.0),
        network_config=FeedForwardConfig.linear_config(),
    )
    model = ConditionalTailQuantileModel(
        env,
        config=config,
        state_mean=torch.zeros(3),
        state_scale=torch.ones(3),
    )
    raw = torch.tensor(
        [
            [[0.10, 0.10], [0.20, 0.20]],
            [[-0.10, -0.10], [-0.20, -0.20]],
        ]
    )
    with torch.no_grad():
        model.network.I0.weight.zero_()
        assert model.network.I0.bias is not None
        model.network.I0.bias.copy_(raw.reshape(-1))
    returns = torch.full((4, 2, 2), 0.25)
    returns[0, 0, 0] = -0.25
    data = {
        "z": torch.zeros(4, 2, 3),
        "returns": returns,
        "values": torch.ones_like(returns),
        "roundoff": torch.full_like(returns, 1.0e-6),
    }

    output = valuate_conditional_tail(
        model,
        data,
        batch_size=3,
        split="validation",
        full=True,
        example_count=2,
    )

    assert output["one_sided_count_by_direction_level"] == [[16, 16], [0, 0]]
    assert output["states_with_any_one_sided_instrument_by_direction_level"] == [
        [8, 8],
        [0, 0],
    ]
    assert output["realized_counter_sign_count_by_direction_level"] == [[1, 1], [0, 0]]
    assert np.allclose(
        output["coverage_by_direction_level"],
        [[1.0 / 16.0, 1.0 / 16.0], [15.0 / 16.0, 15.0 / 16.0]],
    )
    assert output["both_sides_positive_count_by_level"] == [0, 0]
    assert output["has_any_roundoff_one_sided_state"]
    assert len(output["top_examples"][0][0]) == 2


def test_small_conditional_tail_checkpoint_restart(tmp_path) -> None:
    env = BaseEnvironment(device="cpu", debug=False)
    data, geometry = prepare_raw_option_tail_data(_universe(), env=env)
    training, validation = split_tail_data(data, validation_fraction=0.5)
    config = ConditionalTailConfig(
        label="checkpoint-tail",
        state_features=geometry.state_features,
        instruments=geometry.expiries * geometry.strikes,
        training_paths=4,
        tail_path_counts=(1.0, 2.0),
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=6,
            activation="LeakyReLU",
            use_bias=True,
        ),
    )
    train_config = ConditionalTailTrainConfig(
        batch_size=2,
        valuation_batch_size=2,
        lr=1.0e-2,
        final_lr=1.0e-3,
        compile=False,
        value_update=1,
        checkpoint_update=1,
        minimum_epochs=10,
        convergence_points=2,
        checkpoint_root=str(tmp_path),
    )
    first = train_conditional_tail(
        training,
        validation,
        config=config,
        train_config=train_config,
        epochs=2,
        env=env,
        restart=True,
    )
    second = train_conditional_tail(
        training,
        validation,
        config=config,
        train_config=train_config,
        epochs=4,
        env=env,
    )

    assert first.checkpoint_path.exists()
    assert second.restarted
    assert second.tracker.epochs[-1] == 4
    assert math.isfinite(second.training["loss"])


@pytest.mark.parametrize("bad_probability", [0.0, -0.1, 1.1, float("nan")])
def test_empirical_lower_cvar_rejects_invalid_probability(bad_probability: float) -> None:
    with pytest.raises(ValueError):
        empirical_lower_cvar(
            torch.tensor([1.0, 2.0]),
            tail_probability=bad_probability,
        )
