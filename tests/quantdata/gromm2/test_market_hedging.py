from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from quantdata.gromm2.market_books import factor_baskets, primary_contracts
from quantdata.gromm2.market_hedging import MarketBookConfig, MarketBookPolicy, MarketView
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def cohort_policy(**overrides):
    book = primary_contracts()
    config = MarketBookConfig(label="cohort-test", kind="cohorts", feature_count=3,
        option_expiry_days=tuple(map(int, book["expiries"])), option_start_days=tuple(map(int, book["starts"])),
        locked=tuple(map(bool, book["locked"])), call_sides=tuple(map(bool, book["calls"])),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5), **overrides)
    policy = MarketBookPolicy(BaseEnvironment(device="cpu", precision=64), config=config,
        normalization=(np.zeros(3), np.ones(3)))
    return policy, book


def inputs(book):
    features = torch.zeros((4, 20, 3), dtype=torch.float64)
    prices = torch.zeros((4, 21, 169), dtype=torch.float64)
    prices[:, :, 0] = 1
    for index, (start, expiry, level) in enumerate(zip(book["starts"], book["expiries"], book["levels"], strict=True)):
        prices[:, start:expiry + 1, index + 1] = torch.linspace(0.02, 0.05, expiry - start + 1)
        prices[:, start:, 85 + index] = float(level)
    return features, prices


def bias(policy, index, units):
    with torch.no_grad():
        policy.affine.I0.bias[index] = policy.config.position_cap * np.arctanh(units / policy.config.position_cap)


def test_cohort_is_locked_and_settles_without_birth_gain_or_exit_cost():
    policy, book = cohort_policy()
    features, prices = inputs(book)
    bias(policy, 14, 2)
    output = policy(features, prices)
    torch.testing.assert_close(output.positions[:, :1, 14], torch.zeros((4, 1), dtype=torch.float64))
    torch.testing.assert_close(output.positions[:, 1:11, 14], torch.full((4, 10), 2.0, dtype=torch.float64))
    torch.testing.assert_close(output.positions[:, 11:, 14], torch.zeros((4, 9), dtype=torch.float64))
    torch.testing.assert_close(output.gross_gain, torch.full((4,), 0.06, dtype=torch.float64))
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 2e-5, dtype=torch.float64))
    output.trading_gain.mean().backward()
    assert torch.isfinite(policy.affine.I0.bias.grad).all()


def test_duplicate_last_cohort_is_not_a_second_static_contract():
    policy, book = cohort_policy()
    features, prices = inputs(book)
    bias(policy, 80, 2)
    output = policy(features, prices)
    assert torch.count_nonzero(output.positions[:, :, 80]) == 0
    assert torch.count_nonzero(output.transaction_cost) == 0


def test_day_ten_call_can_be_rebalanced_but_other_cohorts_remain_locked():
    policy, book = cohort_policy()
    features, prices = inputs(book)
    features[:, :, 0] = torch.arange(20)
    prices[:, 10:, 85 + 80] = 1.003
    with torch.no_grad():
        policy.affine.I0.weight[80, 0] = 0.1
    enabled = policy(features, prices)
    disabled = MarketBookPolicy(policy.env, config=replace(policy.config, with_fixing_call=False),
        normalization=(np.zeros(3), np.ones(3)))
    disabled.load_state_dict(policy.state_dict())
    locked = disabled(features, prices)
    assert torch.all(enabled.positions[:, 11, 80] > enabled.positions[:, 10, 80])
    torch.testing.assert_close(locked.positions[:, 11, 80], locked.positions[:, 10, 80])


def test_daily_atm_calls_are_new_contracts_with_one_free_settlement_each():
    config = MarketBookConfig(label="factor-test", kind="factors", feature_count=3,
        option_expiry_days=(20, 20, 20), option_start_days=(0, 0, 0), basket_weights=((1.0,),),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    policy = MarketBookPolicy(BaseEnvironment(device="cpu", precision=64), config=config,
        normalization=(np.zeros(3), np.ones(3)))
    bias(policy, 2, 2)
    features = torch.zeros((4, 20, 3), dtype=torch.float64)
    prices = torch.zeros((4, 21, 3), dtype=torch.float64)
    prices[:, :, 0] = torch.arange(21) * 0.001 + 1
    prices[:, :20, 2] = 0.0015
    output = policy(features, prices)
    torch.testing.assert_close(output.gross_gain, torch.full((4,), -0.02, dtype=torch.float64))
    expected = 2e-5 * prices[:, :20, 0].sum(dim=1)
    torch.testing.assert_close(output.transaction_cost, expected)


def test_basket_costs_are_netted_on_actual_option_positions():
    config = MarketBookConfig(label="netting-test", kind="factors", feature_count=3,
        option_expiry_days=(20,) * 4, option_start_days=(0,) * 4, basket_weights=((1.0, -1.0),),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    policy = MarketBookPolicy(BaseEnvironment(device="cpu", precision=64), config=config,
        normalization=(np.zeros(3), np.ones(3)))
    bias(policy, 0, 2)
    bias(policy, 1, 2)
    output = policy(torch.zeros((4, 20, 3), dtype=torch.float64), torch.ones((4, 21, 4), dtype=torch.float64))
    torch.testing.assert_close(output.transaction_cost, torch.zeros(4, dtype=torch.float64))


def test_feature_views_match_across_books_and_hide_future_birth_strikes():
    book = primary_contracts()
    arrays = dict(spots=np.ones((4, 21)), backgrounds=np.ones((4, 21)),
        core_features=np.zeros((4, 20, 5)), option_prices=np.zeros((4, 21, 84)),
        factor_prices=np.zeros((4, 21, 2)), daily_calls=np.ones((4, 20)))
    first = MarketView(arrays, book, kind="cohorts", features=True)
    second = MarketView(arrays, book, kind="factors", features=True)
    before = first[:]
    np.testing.assert_array_equal(before, second[:])
    assert before.shape == first.shape
    arrays["spots"][:, 11:] *= 2
    np.testing.assert_array_equal(first[:][:, :11], before[:, :11])


def test_factor_portfolios_are_real_fixed_weights_not_normal_shock_payoffs():
    exposure = np.asarray([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    weights, report = factor_baskets(exposure, residual_factors=1)
    np.testing.assert_allclose(exposure.T @ weights, [[0], [2], [0]])
    np.testing.assert_allclose(np.abs(weights).sum(axis=0), 1)
    assert report["maximum_exposure_residual"] == pytest.approx(0)
