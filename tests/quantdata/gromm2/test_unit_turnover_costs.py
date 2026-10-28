from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.gromm2.market_books import primary_contracts
from quantdata.gromm2.market_hedging import (
    MarketBookConfig,
    UnitTurnoverBookConfig,
    UnitTurnoverBookPolicy,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _policy(kind: str, **overrides) -> UnitTurnoverBookPolicy:
    if kind == "cohorts":
        book = primary_contracts()
        geometry = dict(option_expiry_days=tuple(map(int, book["expiries"])),
            option_start_days=tuple(map(int, book["starts"])), locked=tuple(map(bool, book["locked"])),
            call_sides=tuple(map(bool, book["calls"])))
    else:
        geometry = dict(option_expiry_days=(20,) * 4, option_start_days=(0,) * 4,
            basket_weights=((1.0, -1.0),))
    config = UnitTurnoverBookConfig(label="unit-turnover-test", kind=kind, feature_count=3,
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5),
        **(geometry | overrides))
    return UnitTurnoverBookPolicy(BaseEnvironment(device="cpu", precision=64), config=config,
        normalization=(np.zeros(3), np.ones(3)))


def _bias(policy: UnitTurnoverBookPolicy, index: int, units: float) -> None:
    with torch.no_grad():
        policy.affine.I0.bias[index] = policy.config.position_cap * np.arctanh(units / policy.config.position_cap)


def test_cohort_cost_has_no_spot_scaling_and_settlement_is_free():
    policy = _policy("cohorts")
    _bias(policy, 14, 2.0)
    features = torch.zeros((4, 20, 3), dtype=torch.float64)
    prices = torch.zeros((4, 21, 169), dtype=torch.float64)
    prices[:, :, 0] = torch.linspace(0.5, 2.0, 21)
    prices[:, 1:12, 15] = 0.02
    prices[:, 1:, 99] = 0.95
    first = policy(features, prices)
    prices[:, :, 0] *= 3.0
    second = policy(features, prices)
    torch.testing.assert_close(first.transaction_cost, torch.full((4,), 2e-5, dtype=torch.float64))
    torch.testing.assert_close(first.transaction_cost, second.transaction_cost, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.positions, second.positions, rtol=0.0, atol=0.0)
    first.trading_gain.mean().backward()
    assert torch.isfinite(policy.affine.I0.bias.grad).all()


def test_pca_cost_charges_each_basket_even_if_underlying_legs_cancel():
    policy = _policy("factors")
    _bias(policy, 0, 2.0)
    _bias(policy, 1, 2.0)
    features = torch.zeros((4, 20, 3), dtype=torch.float64)
    prices = torch.ones((4, 21, 4), dtype=torch.float64)
    first = policy(features, prices)
    prices[:, :, 0] *= 7.0
    second = policy(features, prices)
    torch.testing.assert_close(first.transaction_cost, torch.full((4,), 8e-5, dtype=torch.float64))
    torch.testing.assert_close(first.transaction_cost, second.transaction_cost, rtol=0.0, atol=0.0)
    first.trading_gain.mean().backward()
    assert torch.isfinite(policy.affine.I0.bias.grad).all()


def test_daily_call_cost_counts_entries_not_cash_settlement_or_spot_level():
    policy = _policy("factors")
    _bias(policy, 3, 2.0)
    features = torch.zeros((4, 20, 3), dtype=torch.float64)
    prices = torch.ones((4, 21, 4), dtype=torch.float64)
    prices[:, :, 0] = torch.linspace(1.0, 2.0, 21)
    output = policy(features, prices)
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 4e-4, dtype=torch.float64))


def test_new_cost_convention_has_a_distinct_checkpoint_identity():
    policy = _policy("factors")
    legacy = MarketBookConfig(label="unit-turnover-test", kind="factors", feature_count=3,
        option_expiry_days=(20,) * 4, option_start_days=(0,) * 4, basket_weights=((1.0, -1.0),),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    assert legacy.version.unique_id48 != policy.config.version.unique_id48
    assert legacy.filename != policy.config.filename
    with pytest.raises(ValueError, match="unit_l1"):
        _policy("factors", turnover_cost_basis="spot_notional")
