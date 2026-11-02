from __future__ import annotations

import copy
import io
import os
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

from quantdata.gromm2.combined_hedging import (
    CombinedBookConfig,
    CombinedBookPolicy,
    CombinedMarketView,
    embed_pca_policy,
)
from quantdata.gromm2.hedging_cost_comparison import frozen_market_model
from quantdata.gromm2.hedging_policy import HedgingNetworkConfig, ScenarioData
from quantdata.gromm2.market_batching import GPU_CAPTURE_LOCK, assemble_market_batch
from quantdata.gromm2.market_books import primary_contracts
from quantdata.gromm2.market_hedging import (
    MarketView,
    UnitTurnoverBookConfig,
    UnitTurnoverBookPolicy,
)
from quantdata.gromm2.market_training_runtime import IndependentPortfolioStep
from quantdata.gromm2.portfolio_hedging import (
    LIABILITIES,
    IndependentHedges,
    portfolio_evaluation_forward,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


def configuration(**overrides):
    book = primary_contracts()
    weights = ((0.5, 0.0), (-0.5, 0.5), (0.0, -0.5))
    return CombinedBookConfig(**(dict(label="combined-test", feature_count=3, hidden_size=4,
        option_expiry_days=tuple(map(int, book["expiries"])) + (20,) * 4,
        option_start_days=tuple(map(int, book["starts"])) + (0,) * 4,
        locked=tuple(map(bool, book["locked"])), call_sides=tuple(map(bool, book["calls"])),
        basket_weights=weights, deep=FeedForwardConfig(activation="SiLU", num_depth=2, num_width=5)) | overrides))


def policy_inputs(*, device="cpu", precision=64):
    env = BaseEnvironment(device=device, precision=precision)
    config = configuration()
    policy = CombinedBookPolicy(env, config=config, normalization=(np.zeros(3), np.ones(3)))
    features = torch.zeros((4, 20, 3), dtype=env.fdtype, device=env.device)
    prices = torch.zeros((4, 21, 1 + config.instrument_count + config.option_count), dtype=torch.float64, device=env.device)
    prices[:, :, 0] = torch.linspace(1.0, 1.2, 21, dtype=torch.float64, device=env.device)
    prices[:, :, config.instrument_count - 1] = prices[:, :, 0]
    return policy, features, prices


def set_units(policy, index, units):
    with torch.no_grad():
        policy.affine.I0.bias[index] = policy.config.position_cap * np.arctanh(units / policy.config.position_cap)


def test_combined_has_exactly_one_spot_and_correct_unit_costs():
    policy, features, prices = policy_inputs()
    assert policy.config.instrument_count == 88
    set_units(policy, -2, 2)
    output = policy(features, prices)
    torch.testing.assert_close(output.gross_gain, 2 * (prices[:, -1, 0] - prices[:, 0, 0]))
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 4e-5, dtype=torch.float64))
    assert not torch.count_nonzero(output.positions[:, :, :-2])
    assert not torch.count_nonzero(output.positions[:, :, -1])


def test_combined_daily_call_settles_freely_and_is_bought_anew_each_day():
    policy, features, prices = policy_inputs()
    set_units(policy, -1, 2)
    prices[:, :-1, policy.config.instrument_count] = 0.005
    output = policy(features, prices)
    expected = 2 * (torch.relu(torch.diff(prices[:, :, 0], dim=1)) - 0.005).sum(dim=1)
    torch.testing.assert_close(output.gross_gain, expected)
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 40e-5, dtype=torch.float64))


def test_combined_cohorts_remain_locked_and_cash_settlement_has_no_exit_cost():
    policy, features, prices = policy_inputs()
    set_units(policy, 14, 2)
    prices[:, 1:12, 15] = torch.linspace(0.02, 0.05, 11)
    output = policy(features, prices)
    torch.testing.assert_close(output.positions[:, 1:11, 14], torch.full((4, 10), 2.0, dtype=torch.float64))
    assert not torch.count_nonzero(output.positions[:, 11:, 14])
    torch.testing.assert_close(output.gross_gain, torch.full((4,), 0.06, dtype=torch.float64))
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 2e-5, dtype=torch.float64))


def test_combined_basket_costs_are_per_basket_without_a_spot_multiplier():
    policy, features, prices = policy_inputs()
    set_units(policy, 84, 2)
    prices[:, :, 85] = torch.linspace(0.1, 0.2, 21, dtype=torch.float64)
    output = policy(features, prices)
    torch.testing.assert_close(output.gross_gain, torch.full((4,), 0.2, dtype=torch.float64))
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 4e-5, dtype=torch.float64))
    prices[:, :, (0, policy.config.instrument_count - 1)] *= 3
    scaled = policy(features, prices)
    torch.testing.assert_close(output.transaction_cost, scaled.transaction_cost, rtol=0, atol=0)


def test_matching_day10_call_and_spot_replicate_the_payoff_after_fixing():
    policy, features, prices = policy_inputs()
    fixing = 1.0133
    features[:, 10:, 0] = 1
    with torch.no_grad():
        policy.affine.I0.weight[80, 0] = 10 * np.arctanh((2 / fixing) / 10)
        policy.affine.I0.weight[-2, 0] = 10 * np.arctanh((-1 / fixing) / 10)
    prices[:, 10, 0] = fixing
    prices[:, :, policy.config.instrument_count - 1] = prices[:, :, 0]
    strike_start = 1 + policy.config.instrument_count
    prices[:, 10:, strike_start + 80] = fixing
    prices[:, 10:21, 81] = torch.linspace(0.02, 1.2 - fixing, 11, dtype=torch.float64)
    output = policy(features, prices)
    target = (prices[:, -1, 0] / fixing - 1).abs()
    torch.testing.assert_close(2 * prices[:, 10, 81] / fixing + output.gross_gain, target)


@pytest.mark.parametrize("rows", [slice(None), slice(0, 0), np.array([3, 1, 3])])
def test_combined_views_share_unchanged_information_and_exact_batch_assembly(rows):
    book = primary_contracts()
    generator = np.random.default_rng(7)
    arrays = dict(spots=np.exp(generator.normal(size=(4, 21)) * 0.1), backgrounds=np.ones((4, 21)),
        core_features=generator.normal(size=(4, 20, 5)).astype("f4"),
        option_prices=generator.normal(size=(4, 21, 84)), factor_prices=generator.normal(size=(4, 21, 2)),
        daily_calls=generator.uniform(size=(4, 20)))
    features = CombinedMarketView(arrays, book, features=True)
    prices = CombinedMarketView(arrays, book, features=False)
    original = MarketView(arrays, book, kind="factors", features=True)
    np.testing.assert_array_equal(features[rows], original[rows])
    assert prices.shape == (4, 21, 173)
    np.testing.assert_array_equal(prices[rows][:, :, 0], prices[rows][:, :, 87])
    data = ScenarioData(features, prices, np.ones(4), configuration().option_expiry_days, {}, "source")
    expected = (features[rows], prices[rows], data.payoff[rows])
    actual = assemble_market_batch(data, rows)
    for reference, prepared in zip(expected, actual, strict=True):
        assert reference.tobytes() == prepared.tobytes()


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_pca_embedding_preserves_recurrent_positions_gains_costs_and_translations(depth):
    policy, features, prices = policy_inputs()
    config = configuration(deep=FeedForwardConfig(activation="SiLU", num_depth=depth, num_width=5))
    source_config = UnitTurnoverBookConfig(label="source-pca", kind="factors", feature_count=3, hidden_size=4,
        option_expiry_days=(20,) * 4, option_start_days=(0,) * 4, basket_weights=config.basket_weights,
        spot_cost_bps=0.1, deep=config.deep)
    source = IndependentHedges(policy.env, config=source_config, normalization=(np.zeros(3), np.ones(3)),
        policy_class=UnitTurnoverBookPolicy)
    generator = torch.Generator().manual_seed(39)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.02, 0.02, generator=generator)
    target = IndependentHedges(policy.env, config=config, normalization=(np.zeros(3), np.ones(3)), policy_class=CombinedBookPolicy)
    target.load_state_dict(embed_pca_policy(source, config=config))
    features.normal_(generator=generator)
    pca_prices = torch.cat((prices[:, :, :1], prices[:, :, 85:87], prices[:, :, 88:89]), dim=-1)
    for name in LIABILITIES:
        expected = source.policies[name](features, pca_prices)
        actual = target.policies[name](features, prices)
        assert not torch.count_nonzero(actual.positions[:, :, :84])
        torch.testing.assert_close(actual.positions[:, :, 84:], expected.positions, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual.trading_gain, expected.trading_gain, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual.transaction_cost, expected.transaction_cost, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(target.policies[name].translation, source.policies[name].translation, rtol=0, atol=0)
    target.policies["short"](features, prices).trading_gain.sum().backward()
    assert torch.isfinite(target.policies["short"].affine.I0.bias.grad).all()


def test_frozen_combined_checkpoint_restores_the_new_geometry():
    policy, _, _ = policy_inputs()
    model = IndependentHedges(policy.env, config=policy.config, normalization=(np.zeros(3), np.ones(3)), policy_class=CombinedBookPolicy)
    checkpoint = dict(identity=dict(config=asdict(model.config), code_version=model.config.version.unique_id48,
        user_version=model.config.user_version), report=dict(train_config=dict(precision=64)),
        state=dict(policies={name: dict(best_state=copy.deepcopy(child.state_dict())) for name, child in model.policies.items()}))
    restored = frozen_market_model(checkpoint)
    assert restored.config.kind == "combined" and restored.config.instrument_count == 88
    for name in LIABILITIES:
        for key, value in model.policies[name].state_dict().items():
            torch.testing.assert_close(restored.policies[name].state_dict()[key], value, rtol=0, atol=0)


def test_embedding_accepts_identical_plain_and_versioned_network_configurations():
    config = configuration(deep=HedgingNetworkConfig(num_width=5))
    source_config = UnitTurnoverBookConfig(label="source-plain-config", kind="factors", feature_count=3, hidden_size=4,
        option_expiry_days=(20,) * 4, option_start_days=(0,) * 4, basket_weights=config.basket_weights,
        spot_cost_bps=0.1, deep=FeedForwardConfig(**asdict(config.deep)))
    source = IndependentHedges(BaseEnvironment(device="cpu", precision=64), config=source_config,
        normalization=(np.zeros(3), np.ones(3)), policy_class=UnitTurnoverBookPolicy)
    embedded = embed_pca_policy(source, config=config)
    assert embedded["policies.short.affine.I0.weight"].shape[0] == 88


def test_command_line_forwards_requested_minimum_epochs(monkeypatch, tmp_path):
    from quantdata.gromm2 import combined_hedging

    captured = {}

    def experiment(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(combined_hedging, "run_combined_hedges", experiment)
    monkeypatch.setattr(sys, "argv", ["combined_hedging", "--run-directory", str(tmp_path),
        "--output-directory", str(tmp_path / "policies"), "--minimum-epochs", "100", "--maximum-epochs", "2000",
        "--host-cache-training"])
    combined_hedging.main()
    assert captured["train_config"].minimum_epochs == 100
    assert captured["train_config"].maximum_epochs == 2000
    assert captured["train_config"].host_cache_training is True


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
def test_captured_combined_updates_match_eager_and_keep_partial_batches():
    policy, features, prices = policy_inputs(device="cuda", precision=32)
    payoff = torch.linspace(0.01, 0.04, 4, dtype=torch.float64, device="cuda")
    model = IndependentHedges(policy.env, config=policy.config, normalization=(np.zeros(3), np.ones(3)), policy_class=CombinedBookPolicy)
    reference = IndependentHedges(policy.env, config=policy.config, normalization=(np.zeros(3), np.ones(3)), policy_class=CombinedBookPolicy)
    reference.load_state_dict(model.state_dict())
    optimizers = {name: torch.optim.Adam(child.parameters(), lr=2.5e-4) for name, child in model.policies.items()}
    reference_optimizers = {name: torch.optim.Adam(child.parameters(), lr=2.5e-4) for name, child in reference.policies.items()}
    captured = IndependentPortfolioStep(model, optimizers, gradient_clip=5, compile=True, captured_batch_size=4)
    eager = IndependentPortfolioStep(reference, reference_optimizers, gradient_clip=5, compile=False, use_cuda_graphs=False)
    for count in (4, 4, 2, 4):
        captured(features[:count], prices[:count], payoff[:count], tuple(LIABILITIES))
        eager(features[:count], prices[:count], payoff[:count], tuple(LIABILITIES))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=2e-5, atol=2e-7)
    assert captured.capture_count == 1
    assert captured.tail_step_count == 1


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an explicit gpuctl allocation")
def test_production_geometry_training_graph_survives_evaluation_and_checkpoint():
    """Isolate production geometry from earlier small-shape compiler specializations."""
    torch.compiler.reset()
    generator = np.random.default_rng(20261101)
    weights = tuple(map(tuple, generator.normal(size=(44, 15)) * 0.02))
    book = primary_contracts()
    config = configuration(label="graph-workspace-regression", feature_count=584, hidden_size=32,
        option_expiry_days=tuple(map(int, book["expiries"])) + (20,) * 17,
        option_start_days=tuple(map(int, book["starts"])) + (0,) * 17,
        basket_weights=weights, risk_aversion=100,
        deep=FeedForwardConfig(activation="SiLU", num_depth=2, num_width=64))
    env = BaseEnvironment(device="cuda", precision=32, debug=False)

    def instance(state=None):
        candidate = IndependentHedges(env, config=config,
            normalization=(np.zeros(config.feature_count), np.ones(config.feature_count)), policy_class=CombinedBookPolicy)
        optimizers = {name: torch.optim.Adam(policy.parameters(), lr=2.5e-4)
            for name, policy in candidate.policies.items()}
        if state is not None:
            candidate.load_state_dict(state["model"])
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(copy.deepcopy(state["optimizers"][name]))
        return candidate, optimizers

    model, optimizers = instance()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.as_tensor(generator.normal(size=tuple(parameter.shape)) * 0.005,
                dtype=parameter.dtype, device=env.device))
    features = torch.as_tensor(generator.normal(size=(256, 20, config.feature_count)) * 0.1,
        dtype=env.fdtype, device=env.device)
    prices = torch.as_tensor(generator.normal(size=(256, 21, 1 + config.instrument_count + 84)) * 0.001 + 0.01,
        dtype=torch.float64, device=env.device)
    prices[:, :, 0] += 0.99
    prices[:, :, -84:] = 1
    payoff = (prices[:, 20, 0] / prices[:, 10, 0] - 1).abs()
    inputs = features, prices, payoff
    runtime = IndependentPortfolioStep(model, optimizers, gradient_clip=5, compile=True, captured_batch_size=256)
    evaluator = portfolio_evaluation_forward(model, compile=True)
    for cycle in range(2):
        runtime(*inputs, tuple(LIABILITIES))
        runtime(*(value[:64] for value in inputs), tuple(LIABILITIES))
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        snapshot = dict(model=copy.deepcopy(model.state_dict()),
            optimizers={name: copy.deepcopy(value.state_dict()) for name, value in optimizers.items()})
        model.eval()
        with torch.no_grad():
            for count in (4096, 576, 4096, 576, 4096, 576):
                expanded = tuple(value.repeat((int(np.ceil(count / len(value))), 1, 1))[:count] for value in inputs[:2])
                torch.compiler.cudagraph_mark_step_begin()
                with GPU_CAPTURE_LOCK:
                    packed = evaluator(*expanded)
                assert torch.isfinite(packed.cpu()).all()
                del packed, expanded
        torch.save(snapshot, io.BytesIO())
        with GPU_CAPTURE_LOCK:
            torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        pressure = torch.full((16 * 1024 * 1024,), float("nan"), device=env.device)
        del pressure
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        actual = runtime(*inputs, tuple(LIABILITIES))
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        reference, reference_optimizers = instance(snapshot)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        eager = IndependentPortfolioStep(reference, reference_optimizers,
            gradient_clip=5, compile=False, use_cuda_graphs=False)
        expected = eager(*inputs, tuple(LIABILITIES))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=3e-5, atol=2e-7)
        for name, parameter in model.named_parameters():
            other = dict(reference.named_parameters())[name]
            assert (parameter.grad is None) == (other.grad is None)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-4, atol=5e-7)
        for name, optimizer in optimizers.items():
            actual_states = optimizer.state_dict()["state"]
            for parameter_id, state in reference_optimizers[name].state_dict()["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(actual_states[parameter_id][key], value, rtol=3e-4, atol=5e-7)
            assert actual[name]["loss"] == pytest.approx(expected[name]["loss"], rel=2e-5, abs=5e-8)
            assert actual[name]["gradient_norm"] == pytest.approx(expected[name]["gradient_norm"], rel=3e-4, abs=5e-7)
        assert runtime.capture_count == 1
        assert runtime.tail_step_count == cycle + 1
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        del eager, reference, reference_optimizers, snapshot
