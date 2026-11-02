from __future__ import annotations

import copy
import os
from dataclasses import asdict

import numpy as np
import pytest
import torch

from quantdata.gromm2.hedging_cost_comparison import frozen_market_model
from quantdata.gromm2.hedging_policy import ScenarioData
from quantdata.gromm2.market_batching import assemble_market_batch
from quantdata.gromm2.market_books import primary_contracts
from quantdata.gromm2.market_hedging import (
    MarketView,
    SpotCohortBookConfig,
    SpotCohortBookPolicy,
)
from quantdata.gromm2.market_training_runtime import IndependentPortfolioStep
from quantdata.gromm2.portfolio_hedging import LIABILITIES, IndependentHedges
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def configuration(**overrides):
    book = primary_contracts()
    return SpotCohortBookConfig(**(dict(label="spot-cohort-test", kind="cohorts", feature_count=3,
        option_expiry_days=tuple(map(int, book["expiries"])) + (20,),
        option_start_days=tuple(map(int, book["starts"])) + (0,),
        locked=tuple(map(bool, book["locked"])) + (False,),
        call_sides=tuple(map(bool, book["calls"])) + (False,),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5)) | overrides))


def policy_inputs(*, device="cpu", precision=64):
    env = BaseEnvironment(device=device, precision=precision)
    policy = SpotCohortBookPolicy(env, config=configuration(), normalization=(np.zeros(3), np.ones(3)))
    features = torch.zeros((4, 20, 3), dtype=env.fdtype, device=env.device)
    prices = torch.zeros((4, 21, 171), dtype=torch.float64, device=env.device)
    prices[:, :, 0] = torch.linspace(1, 1.2, 21, device=env.device)
    prices[:, :, 85] = prices[:, :, 0]
    return policy, features, prices


def set_units(policy, index, units):
    with torch.no_grad():
        policy.affine.I0.bias[index] = policy.config.position_cap * np.arctanh(units / policy.config.position_cap)


def test_spot_trades_all_days_earns_own_increments_and_pays_liquidation():
    torch.set_num_threads(1)
    policy, features, prices = policy_inputs()
    set_units(policy, 84, 2)
    output = policy(features, prices)
    torch.testing.assert_close(output.positions[:, :, -1], torch.full((4, 20), 2.0, dtype=torch.float64))
    torch.testing.assert_close(output.gross_gain, 2 * (prices[:, -1, 0] - prices[:, 0, 0]))
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 4e-5, dtype=torch.float64))
    torch.testing.assert_close(output.trading_gain, output.gross_gain - output.transaction_cost)
    output.trading_gain.mean().backward()
    assert torch.isfinite(policy.affine.I0.bias.grad).all()
    assert policy.affine.I0.bias.grad[-1].abs() > 0


def test_spot_rebalances_daily_without_spot_scaled_turnover():
    policy, features, prices = policy_inputs()
    features[:, :, 0] = torch.arange(20) - 10
    with torch.no_grad():
        policy.affine.I0.weight[-1, 0] = 0.2
    first = policy(features, prices)
    units = first.positions[:, :, -1]
    previous = torch.cat((torch.zeros_like(units[:, :1]), units[:, :-1]), dim=1)
    expected = 1e-5 * ((units - previous).abs().sum(dim=1) + units[:, -1].abs())
    torch.testing.assert_close(first.transaction_cost, expected)
    prices[:, :, (0, 85)] *= 3
    second = policy(features, prices)
    torch.testing.assert_close(first.positions, second.positions, rtol=0, atol=0)
    torch.testing.assert_close(first.transaction_cost, second.transaction_cost, rtol=0, atol=0)
    torch.testing.assert_close(second.gross_gain, 3 * first.gross_gain)


def test_spot_extension_preserves_locked_cohort_and_free_option_settlement():
    policy, features, prices = policy_inputs()
    set_units(policy, 14, 2)
    set_units(policy, 84, 2)
    prices[:, 1:12, 15] = torch.linspace(0.02, 0.05, 11)
    prices[:, 1:, 100] = 0.95
    output = policy(features, prices)
    torch.testing.assert_close(output.positions[:, 1:11, 14], torch.full((4, 10), 2.0, dtype=torch.float64))
    assert not torch.count_nonzero(output.positions[:, 11:, 14])
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 6e-5, dtype=torch.float64))
    torch.testing.assert_close(output.gross_gain, 2 * (prices[:, -1, 0] - prices[:, 0, 0]) + 0.06)


@pytest.mark.parametrize("rows", [slice(None), slice(0, 0), np.array([3, 1, 3])])
def test_spot_views_preserve_features_and_shared_assembly_exactly(rows):
    book = primary_contracts()
    generator = np.random.default_rng(17)
    arrays = dict(spots=np.exp(generator.normal(size=(4, 21)) * 0.1), backgrounds=np.ones((4, 21)),
        core_features=generator.normal(size=(4, 20, 5)).astype("f4"),
        option_prices=generator.normal(size=(4, 21, 84)), factor_prices=np.zeros((4, 21, 2)),
        daily_calls=np.zeros((4, 20)))
    features = MarketView(arrays, book, kind="cohorts", features=True, with_spot=True)
    prices = MarketView(arrays, book, kind="cohorts", features=False, with_spot=True)
    original_features = MarketView(arrays, book, kind="cohorts", features=True)
    data = ScenarioData(features, prices, np.ones(4), tuple(map(int, book["expiries"])) + (20,), {}, "source")
    expected = (features[rows], prices[rows], data.payoff[rows])
    np.testing.assert_array_equal(features[rows], original_features[rows])
    assert prices.shape == (4, 21, 171)
    np.testing.assert_array_equal(prices[rows][:, :, 85], prices[rows][:, :, 0])
    assert not np.count_nonzero(prices[rows][:, :, -1])
    actual = assemble_market_batch(data, rows)
    for original, prepared in zip(expected, actual, strict=True):
        assert original.tobytes() == prepared.tobytes()


def test_frozen_spot_policies_load_with_correct_geometry_and_identity():
    config = configuration()
    env = BaseEnvironment(device="cpu", precision=64)
    model = IndependentHedges(env, config=config, normalization=(np.zeros(3), np.ones(3)), policy_class=SpotCohortBookPolicy)
    checkpoint = dict(identity=dict(config=asdict(config), code_version=config.version.unique_id48,
        user_version=config.user_version), report=dict(train_config=dict(precision=64)),
        state=dict(policies={name: dict(best_state=copy.deepcopy(policy.state_dict()))
            for name, policy in model.policies.items()}))
    restored = frozen_market_model(checkpoint)
    assert restored.config.instrument_count == 85
    assert restored.config.spot_in_cohort_book
    for name, policy in restored.policies.items():
        assert isinstance(policy, SpotCohortBookPolicy)
        for key, value in model.policies[name].state_dict().items():
            torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
def test_captured_spot_update_matches_eager_and_final_partial_batch():
    torch.set_num_threads(1)
    policy, features, prices = policy_inputs(device="cuda", precision=32)
    payoff = torch.linspace(0.01, 0.04, 4, device="cuda", dtype=torch.float64)
    model = IndependentHedges(policy.env, config=policy.config, normalization=(np.zeros(3), np.ones(3)),
        policy_class=SpotCohortBookPolicy)
    reference = IndependentHedges(policy.env, config=policy.config, normalization=(np.zeros(3), np.ones(3)),
        policy_class=SpotCohortBookPolicy)
    reference.load_state_dict(model.state_dict())
    optimizers = {name: torch.optim.Adam(child.parameters(), lr=2.5e-4) for name, child in model.policies.items()}
    reference_optimizers = {name: torch.optim.Adam(child.parameters(), lr=2.5e-4) for name, child in reference.policies.items()}
    captured = IndependentPortfolioStep(model, optimizers, gradient_clip=5, compile=True)
    eager = IndependentPortfolioStep(reference, reference_optimizers, gradient_clip=5, compile=False, use_cuda_graphs=False)
    for count in (4, 4, 2):
        captured(features[:count], prices[:count], payoff[:count], tuple(LIABILITIES))
        eager(features[:count], prices[:count], payoff[:count], tuple(LIABILITIES))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=2e-5, atol=2e-7)
