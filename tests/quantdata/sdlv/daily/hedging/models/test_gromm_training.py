from __future__ import annotations

import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    GrommPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.training import (
    GrommTrainingData,
    _paired_split,
    fit_gromm_policy,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _synthetic_training_data() -> GrommTrainingData:
    generator = np.random.default_rng(71)
    pair_count = 128
    half = generator.normal(size=(pair_count, 2))
    returns = np.stack((half, -half), axis=1).reshape(2 * pair_count, 1, 2)
    true_action = np.asarray([-0.7, 0.35])
    noise_half = 0.02 * generator.normal(size=pair_count)
    noise = np.stack((noise_half, -noise_half), axis=1).reshape(-1)
    claim = -(returns[:, 0] @ true_action) + noise
    features = np.ones((2 * pair_count, 1, 3))
    return GrommTrainingData(
        features=features,
        instrument_returns=returns,
        claim_returns=claim[:, None],
        active_mask=np.ones((1, 2), dtype=bool),
        linear_transaction_costs=np.zeros((1, 2)),
    )


def test_paired_split_keeps_antithetic_paths_together() -> None:
    training, validation = _paired_split(
        40,
        validation_fraction=0.25,
        seed=3,
    )
    training_set = set(training)
    validation_set = set(validation)
    for first in range(0, 40, 2):
        pair = {first, first + 1}
        assert pair <= training_set or pair <= validation_set
    assert training_set.isdisjoint(validation_set)


def test_gromm_training_recovers_linear_hedge_and_restarts(tmp_path) -> None:
    data = _synthetic_training_data()
    policy_config = GrommPolicyConfig(
        label="linear-recovery",
        feature_count=3,
        instrument_count=2,
        hidden_size=4,
        decision_network=FeedForwardConfig(
            activation="SiLU",
            num_depth=1,
            num_width=4,
        ),
        position_cap=10.0,
    )
    train_config = GrommTrainConfig(
        maximum_epochs=300,
        minimum_epochs=30,
        patience_epochs=40,
        batch_pairs=128,
        learning_rate=0.03,
        checkpoint_epochs=50,
        progress_seconds=300.0,
        seed=11,
    )
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)

    result = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=20.0,
        checkpoint_directory=tmp_path,
        train_config=train_config,
        environment=environment,
    )
    loaded = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=20.0,
        checkpoint_directory=tmp_path,
        train_config=GrommTrainConfig(
            maximum_epochs=300,
            minimum_epochs=30,
            patience_epochs=40,
            batch_pairs=128,
            learning_rate=0.03,
            checkpoint_epochs=50,
            progress_seconds=300.0,
            seed=11,
            just_load=True,
        ),
        environment=environment,
    )

    assert result.checkpoint_path.is_file()
    assert result.validation_loss < 0.01
    assert loaded.validation_loss == pytest.approx(result.validation_loss)
    assert loaded.best_epoch == result.best_epoch


def test_gromm_checkpoint_identity_includes_objective_and_risk(tmp_path) -> None:
    data = _synthetic_training_data()
    policy_config = GrommPolicyConfig(
        label="logical-policy",
        feature_count=3,
        instrument_count=2,
    )
    train_config = GrommTrainConfig(
        maximum_epochs=1,
        minimum_epochs=1,
        patience_epochs=1,
        batch_pairs=128,
        checkpoint_epochs=1,
        seed=4,
    )
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)

    mean_variance = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
        checkpoint_directory=tmp_path,
        train_config=train_config,
        environment=environment,
    )
    entropy = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.ENTROPY,
        risk_parameter=2.0,
        checkpoint_directory=tmp_path,
        train_config=train_config,
        environment=environment,
    )

    assert mean_variance.checkpoint_path != entropy.checkpoint_path


def test_gromm_checkpoint_identity_includes_exact_training_data(tmp_path) -> None:
    data = _synthetic_training_data()
    changed_claims = data.claim_returns.copy()
    changed_claims[0, 0] += 1.0e-8
    changed = GrommTrainingData(
        features=data.features,
        instrument_returns=data.instrument_returns,
        claim_returns=changed_claims,
        active_mask=data.active_mask,
        linear_transaction_costs=data.linear_transaction_costs,
    )
    policy_config = GrommPolicyConfig(
        label="data-identity",
        feature_count=3,
        instrument_count=2,
    )
    train_config = GrommTrainConfig(
        maximum_epochs=1,
        minimum_epochs=1,
        patience_epochs=10,
        batch_pairs=128,
        checkpoint_epochs=1,
        seed=4,
    )
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)

    original = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
        checkpoint_directory=tmp_path,
        train_config=train_config,
        environment=environment,
    )
    perturbed = fit_gromm_policy(
        changed,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
        checkpoint_directory=tmp_path,
        train_config=train_config,
        environment=environment,
    )

    assert original.checkpoint_path != perturbed.checkpoint_path


def test_gromm_budget_extension_matches_uninterrupted_training(tmp_path) -> None:
    data = _synthetic_training_data()
    policy_config = GrommPolicyConfig(
        label="exact-restart",
        feature_count=3,
        instrument_count=2,
        hidden_size=4,
        decision_network=FeedForwardConfig(
            activation="SiLU",
            num_depth=1,
            num_width=4,
        ),
        position_cap=10.0,
    )
    common = {
        "minimum_epochs": 1,
        "patience_epochs": 100,
        "batch_pairs": 128,
        "learning_rate": 0.01,
        "checkpoint_epochs": 1,
        "progress_seconds": 300.0,
        "seed": 17,
    }
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)

    uninterrupted = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=5.0,
        checkpoint_directory=tmp_path / "uninterrupted",
        train_config=GrommTrainConfig(maximum_epochs=4, **common),
        environment=environment,
    )
    partial = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=5.0,
        checkpoint_directory=tmp_path / "resumed",
        train_config=GrommTrainConfig(maximum_epochs=2, **common),
        environment=environment,
    )
    assert not partial.converged
    resumed = fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=5.0,
        checkpoint_directory=tmp_path / "resumed",
        train_config=GrommTrainConfig(maximum_epochs=4, **common),
        environment=environment,
    )

    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.validation_loss == pytest.approx(
        uninterrupted.validation_loss,
        rel=0.0,
        abs=0.0,
    )
    for name, value in uninterrupted.policy.state_dict().items():
        assert value.equal(resumed.policy.state_dict()[name]), name


def test_gromm_checkpoint_rejects_changed_split(tmp_path) -> None:
    data = _synthetic_training_data()
    policy_config = GrommPolicyConfig(
        label="split-identity",
        feature_count=3,
        instrument_count=2,
    )
    common = {
        "maximum_epochs": 1,
        "minimum_epochs": 1,
        "patience_epochs": 10,
        "batch_pairs": 128,
        "checkpoint_epochs": 1,
        "progress_seconds": 300.0,
    }
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
        checkpoint_directory=tmp_path,
        train_config=GrommTrainConfig(
            validation_fraction=0.2,
            seed=13,
            **common,
        ),
        environment=environment,
    )

    with pytest.raises(ValueError, match="seed, validation_fraction"):
        fit_gromm_policy(
            data,
            policy_config=policy_config,
            objective=GrommObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            checkpoint_directory=tmp_path,
            train_config=GrommTrainConfig(
                validation_fraction=0.3,
                seed=99,
                **common,
            ),
            environment=environment,
        )


def test_gromm_changed_optimizer_requires_opt_in_and_more_epochs(
    tmp_path,
) -> None:
    data = _synthetic_training_data()
    policy_config = GrommPolicyConfig(
        label="explicit-warm-start",
        feature_count=3,
        instrument_count=2,
    )
    common = {
        "minimum_epochs": 1,
        "patience_epochs": 100,
        "batch_pairs": 128,
        "checkpoint_epochs": 1,
        "progress_seconds": 300.0,
        "seed": 17,
    }
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    fit_gromm_policy(
        data,
        policy_config=policy_config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
        checkpoint_directory=tmp_path,
        train_config=GrommTrainConfig(
            maximum_epochs=1,
            learning_rate=0.01,
            **common,
        ),
        environment=environment,
    )

    with pytest.raises(ValueError, match="allow_changed_config_warm_start"):
        fit_gromm_policy(
            data,
            policy_config=policy_config,
            objective=GrommObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            checkpoint_directory=tmp_path,
            train_config=GrommTrainConfig(
                maximum_epochs=2,
                learning_rate=0.02,
                **common,
            ),
            environment=environment,
        )
    with pytest.warns(UserWarning, match="learning_rate"):
        resumed = fit_gromm_policy(
            data,
            policy_config=policy_config,
            objective=GrommObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            checkpoint_directory=tmp_path,
            train_config=GrommTrainConfig(
                maximum_epochs=2,
                learning_rate=0.02,
                allow_changed_config_warm_start=True,
                **common,
            ),
            environment=environment,
        )

    assert resumed.completed_epochs == 2
