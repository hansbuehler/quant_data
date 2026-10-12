from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.objectives import (
    gromm_hedging_objective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    GrommPolicy,
    GrommPolicyConfig,
    gromm_architecture_candidates,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _policy() -> GrommPolicy:
    environment = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = GrommPolicyConfig(
        label="unit",
        feature_count=3,
        instrument_count=2,
        hidden_size=4,
        decision_network=FeedForwardConfig(
            activation="SiLU",
            num_depth=1,
            num_width=5,
        ),
        position_cap=10.0,
    )
    return GrommPolicy(
        environment,
        config=config,
        feature_mean=torch.zeros(3, dtype=torch.float64),
        feature_scale=torch.ones(3, dtype=torch.float64),
    )


def test_gromm_policy_has_zero_initial_action_and_gradients() -> None:
    policy = _policy()
    features = torch.randn(6, 3, 3, dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False], [False, True]])

    actions = policy(features, mask)
    actions.square().sum().backward()

    assert actions.shape == (6, 3, 2)
    assert actions.detach().numpy() == pytest.approx(np.zeros((6, 3, 2)))
    assert actions[:, 1, 1].detach().numpy() == pytest.approx(np.zeros(6))
    assert policy.affine.weight.grad is not None


@pytest.mark.parametrize("objective", tuple(GrommObjective))
def test_gromm_objectives_are_finite_and_differentiable(
    objective: GrommObjective,
) -> None:
    actions = torch.zeros((8, 1, 2), dtype=torch.float64, requires_grad=True)
    returns = torch.tensor(
        [[[1.0, -0.5]], [[-1.0, 0.5]]] * 4,
        dtype=torch.float64,
    )
    claim = torch.linspace(-0.2, 0.2, 8, dtype=torch.float64)[:, None]
    costs = torch.tensor([[0.01, 0.02]], dtype=torch.float64)
    cash_translation = (
        torch.zeros((), dtype=torch.float64, requires_grad=True)
        if objective is GrommObjective.POWER_EXPONENTIAL
        else None
    )

    result = gromm_hedging_objective(
        actions=actions,
        instrument_returns=returns,
        claim_returns=claim,
        linear_transaction_costs=costs,
        objective=objective,
        risk_parameter=1.0,
        cash_translation=cash_translation,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert actions.grad is not None
    assert torch.all(torch.isfinite(actions.grad))


def test_four_architecture_candidates_are_distinct() -> None:
    candidates = gromm_architecture_candidates(
        label_prefix="architecture",
        feature_count=5,
        instrument_count=7,
        position_cap=100.0,
    )

    assert len(candidates) == 4
    assert len({candidate.filename for candidate in candidates}) == 4
    assert [candidate.hidden_size for candidate in candidates] == [16, 32, 64, 64]


def test_gromm_objective_charges_turnover_and_free_settlement() -> None:
    actions = torch.tensor(
        [[[2.0, 3.0], [4.0, 5.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = gromm_hedging_objective(
        actions=actions,
        instrument_returns=torch.zeros_like(actions),
        claim_returns=torch.zeros((1, 2), dtype=torch.float64),
        linear_transaction_costs=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64
        ),
        initial_positions=torch.tensor([1.0, 1.0], dtype=torch.float64),
        settles_next=torch.tensor(
            [[False, True], [False, False]], dtype=torch.bool
        ),
        terminal_transaction_costs=torch.tensor(
            [5.0, 6.0], dtype=torch.float64
        ),
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
    )

    expected = (
        1.0 * abs(2.0 - 1.0)
        + 2.0 * abs(3.0 - 1.0)
        + 3.0 * abs(4.0 - 2.0)
        + 4.0 * abs(5.0 - 0.0)
        + 5.0 * abs(4.0)
        + 6.0 * abs(5.0)
    )
    assert result.transaction_cost.item() == pytest.approx(expected)
    assert result.wealth.item() == pytest.approx(-expected)
