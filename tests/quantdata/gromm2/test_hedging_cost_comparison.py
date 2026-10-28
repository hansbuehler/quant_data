from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from quantdata.gromm2.hedging_cost_comparison import frozen_market_model, transaction_cost_quotes
from quantdata.gromm2.market_hedging import (
    MarketBookConfig,
    MarketBookPolicy,
    UnitTurnoverBookConfig,
    UnitTurnoverBookPolicy,
)
from quantdata.gromm2.portfolio_hedging import LIABILITIES, IndependentHedges
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_constant_costs_shift_risks_and_quotes_by_exact_cash_amounts():
    payoff = np.linspace(0.01, 0.05, 8)
    gains = {name: np.zeros(8) for name in LIABILITIES}
    costs = {name: np.full(8, value) for name, value in {"zero": 0.001, "short": 0.003, "long": 0.002}.items()}
    result = transaction_cost_quotes(payoff, gains, costs, risk_aversion=100.0)
    before, after = (result[key]["statistics"] for key in ("with_costs", "costs_removed"))
    assert before["ask"]["mean"] - after["ask"]["mean"] == pytest.approx(0.002)
    assert after["bid"]["mean"] - before["bid"]["mean"] == pytest.approx(0.001)
    assert after["ask"]["standard_error"] == pytest.approx(before["ask"]["standard_error"])
    assert after["bid"]["standard_error"] == pytest.approx(before["bid"]["standard_error"])


def test_random_path_costs_are_not_replaced_by_their_average():
    payoff = np.linspace(0.0, 0.1, 8)
    gains = {name: np.zeros(8) for name in LIABILITIES}
    costs = {name: np.zeros(8) for name in LIABILITIES}
    costs["short"] = 0.05 * payoff
    exact = transaction_cost_quotes(payoff, gains, costs, risk_aversion=100.0)
    averaged = transaction_cost_quotes(payoff, gains,
        {name: np.full(8, value.mean()) for name, value in costs.items()}, risk_aversion=100.0)
    assert abs(exact["costs_removed"]["statistics"]["ask"]["mean"]
        - averaged["costs_removed"]["statistics"]["ask"]["mean"]) > 1e-5


def test_negative_or_incomplete_cost_vectors_are_rejected():
    gains = {name: np.zeros(8) for name in LIABILITIES}
    with pytest.raises(ValueError, match="all three"):
        transaction_cost_quotes(np.ones(8), gains, {}, risk_aversion=100.0)
    costs = {name: np.zeros(8) for name in LIABILITIES}
    costs["long"][0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        transaction_cost_quotes(np.ones(8), gains, costs, risk_aversion=100.0)


@pytest.mark.parametrize("unit_costs", [False, True])
def test_best_weights_normalization_and_actions_are_loaded_unchanged_without_optimizer(unit_costs):
    config_class = UnitTurnoverBookConfig if unit_costs else MarketBookConfig
    policy_class = UnitTurnoverBookPolicy if unit_costs else MarketBookPolicy
    config = config_class(label="frozen-cost-test", kind="factors", feature_count=3,
        option_expiry_days=(20,) * 3, option_start_days=(0,) * 3, basket_weights=((1.0,),),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    source = IndependentHedges(env, config=config,
        normalization=(np.ones(3), np.full(3, 2.0)), policy_class=policy_class)
    with torch.no_grad():
        for index, policy in enumerate(source.policies.values()):
            policy.affine.I0.bias.fill_(0.1 * (index + 1))
    checkpoint = dict(identity=dict(config=asdict(config), code_version=config.version.unique_id48,
        user_version=config.user_version), report=dict(train_config=dict(precision=64)),
        state=dict(policies={name: dict(best_state=copy.deepcopy(policy.state_dict()))
            for name, policy in source.policies.items()}))
    restored = frozen_market_model(checkpoint)
    features = torch.zeros((4, 20, 3), dtype=torch.float64)
    prices = torch.ones((4, 21, 3), dtype=torch.float64)
    with torch.no_grad():
        for name, policy in restored.policies.items():
            assert not policy.training
            torch.testing.assert_close(policy(features, prices).positions, source.policies[name](features, prices).positions,
                rtol=0.0, atol=0.0)
            torch.testing.assert_close(policy.normalization_mean, torch.ones(3, dtype=torch.float64))
            torch.testing.assert_close(policy.normalization_scale, torch.full((3,), 2.0, dtype=torch.float64))
