from __future__ import annotations

from dataclasses import replace

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.hedger import (
    fit_gromm_claim_minus_zero,
    gromm_pure_hedge_actions,
    select_gromm_architecture,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    GrommPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.selection import (
    GrommArchitectureStudyConfig,
    combine_gromm_training_data,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.training import (
    GrommTrainingData,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _data() -> GrommTrainingData:
    generator = np.random.default_rng(18)
    half = generator.normal(size=(32, 2))
    returns = np.stack((half, -half), axis=1).reshape(64, 1, 2)
    claim = -(returns[:, 0] @ np.asarray([0.4, -0.2]))
    return GrommTrainingData(
        features=np.ones((64, 1, 3)),
        instrument_returns=returns,
        claim_returns=claim[:, None],
        active_mask=np.ones((1, 2), dtype=bool),
        linear_transaction_costs=np.zeros((1, 2)),
    )


def _train_config(*, maximum_epochs: int = 50) -> GrommTrainConfig:
    return GrommTrainConfig(
        maximum_epochs=maximum_epochs,
        minimum_epochs=min(10, maximum_epochs),
        patience_epochs=10,
        batch_pairs=32,
        learning_rate=0.02,
        checkpoint_epochs=maximum_epochs,
        seed=9,
    )


def test_claim_minus_zero_policy_returns_capped_pure_action(tmp_path) -> None:
    data = _data()
    config = GrommPolicyConfig(
        label="claim-minus-zero",
        feature_count=3,
        instrument_count=2,
        hidden_size=4,
        decision_network=FeedForwardConfig(num_depth=1, num_width=4),
        position_cap=2.0,
    )
    fit = fit_gromm_claim_minus_zero(
        data,
        pure_data=data,
        policy_config=config,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=5.0,
        checkpoint_directory=tmp_path,
        train_config=_train_config(),
        environment=BaseEnvironment(precision=64, device="cpu", debug=True),
    )

    action, cap_hits = gromm_pure_hedge_actions(
        fit,
        features=data.features[:1],
        active_mask=data.active_mask,
    )

    assert action.shape == (1, 1, 2)
    assert np.all(np.isfinite(action))
    assert np.max(np.abs(action)) <= 2.0
    assert cap_hits >= 0
    assert fit.claim.checkpoint_path != fit.zero.checkpoint_path
    assert fit.short.checkpoint_path != fit.zero.checkpoint_path
    assert fit.pure.checkpoint_path != fit.zero.checkpoint_path
    assert np.isfinite(fit.buyer_indifference_price)
    assert np.isfinite(fit.seller_indifference_price)


def test_architecture_selection_trains_four_candidates(tmp_path) -> None:
    data = _data()
    candidates = tuple(
        GrommPolicyConfig(
            label=f"candidate-{index}",
            feature_count=3,
            instrument_count=2,
            hidden_size=hidden,
            decision_network=FeedForwardConfig(num_depth=1, num_width=hidden),
            position_cap=2.0,
        )
        for index, hidden in enumerate((2, 3, 4, 5))
    )

    selection = select_gromm_architecture(
        data,
        candidates=candidates,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=5.0,
        checkpoint_directory=tmp_path,
        train_config=_train_config(maximum_epochs=1),
        environment=BaseEnvironment(precision=64, device="cpu", debug=True),
    )

    assert len(selection.candidates) == 4
    assert selection.selected in candidates
    assert selection.best_validation_standard_error >= 0.0


def test_architecture_study_combines_independent_case_blocks() -> None:
    first = _data()
    second = replace(
        first,
        features=first.features + 1.0,
        claim_returns=first.claim_returns - 0.1,
    )

    combined = combine_gromm_training_data((first, second))

    assert combined.path_count == first.path_count + second.path_count
    np.testing.assert_array_equal(
        combined.features[: first.path_count],
        first.features,
    )
    np.testing.assert_array_equal(
        combined.features[first.path_count :],
        second.features,
    )
    np.testing.assert_array_equal(combined.active_mask, first.active_mask)


def test_architecture_study_rejects_inconsistent_cost_geometry() -> None:
    first = _data()
    changed_costs = first.linear_transaction_costs.copy()
    changed_costs[0, 0] = 0.01
    second = replace(first, linear_transaction_costs=changed_costs)

    with np.testing.assert_raises_regex(ValueError, "policy geometry"):
        combine_gromm_training_data((first, second))


def test_architecture_study_is_precommitted_to_2022_p_and_q() -> None:
    config = GrommArchitectureStudyConfig(paths_per_case=8)

    assert config.origin_date.isoformat() == "2022-12-30"
    assert config.calibration_window_sessions == 200
    assert tuple(measure.value for measure in config.measures) == ("Q", "P")
    assert config.position_cap == 1_000.0
