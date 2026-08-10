from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from quantdata.dysanos.grid.rndysanos.direct_arbitrage import (
    DirectArbitrageConfig,
    DirectArbitrageTrainConfig,
    LongShortSimplexPolicy,
    RadialL1Policy,
    direct_arbitrage_loss,
    effective_cost,
    net_return,
    train_direct_policy,
    unconditional_cost,
    v2_arbitrage_loss,
    valuate_policy,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _env() -> BaseEnvironment:
    return BaseEnvironment(device="cpu", debug=False)


def _config(label: str, *, instruments: int = 2) -> DirectArbitrageConfig:
    return DirectArbitrageConfig(
        label=label,
        state_features=3,
        instruments=instruments,
        epsilon=0.01,
        delta=0.01,
        numerical_tolerance=1.0e-6,
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=8,
            activation="LeakyReLU",
            use_bias=True,
        ),
    )


def _policy(config: DirectArbitrageConfig) -> LongShortSimplexPolicy:
    return LongShortSimplexPolicy(
        _env(),
        config=config,
        state_mean=torch.zeros(config.state_features),
        state_scale=torch.ones(config.state_features),
    )


def _radial_policy(config: DirectArbitrageConfig) -> RadialL1Policy:
    return RadialL1Policy(
        _env(),
        config=config,
        state_mean=torch.zeros(config.state_features),
        state_scale=torch.ones(config.state_features),
    )


def test_long_short_policy_is_state_only_and_respects_simplex() -> None:
    config = _config("simplex")
    policy = _policy(config)
    z = torch.randn(4, 2, config.state_features)
    long, short, no_trade = policy(z)

    assert long.shape == short.shape == (4, 2, config.instruments)
    assert no_trade.shape == (4, 2)
    assert torch.all(long >= 0.0) and torch.all(short >= 0.0)
    assert torch.allclose(
        (long + short).sum(dim=-1) + no_trade,
        torch.ones_like(no_trade),
    )

    # Returns are deliberately absent from the policy API: identical state
    # always gives identical allocations irrespective of the later outcome.
    repeated = z[:1].expand(2, -1, -1)
    first = policy(repeated)[0]
    assert torch.allclose(first[0], first[1])


def test_long_short_policy_assigns_zero_weight_to_excluded_directions() -> None:
    config = replace(
        _config("masked-simplex"), mask_arbitrage_candidates=True
    )


def test_radial_l1_policy_clips_norm_and_retains_tangential_gradients() -> None:
    config = replace(_config("radial", instruments=3), objective="radial-l1-v2")
    policy = _radial_policy(config)
    final = policy.network.In[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias[:] = torch.tensor([2.0, 1.0, 0.0])

    z = torch.zeros(1, 1, config.state_features)
    action, raw_l1 = policy(z)
    assert raw_l1.item() == pytest.approx(3.0)
    assert torch.allclose(action, torch.tensor([[[2.0 / 3.0, 1.0 / 3.0, 0.0]]]))
    assert action.abs().sum().item() == pytest.approx(1.0)

    action[..., 1].sum().backward()
    assert final.bias.grad is not None
    assert final.bias.grad[0].item() != 0.0
    assert final.bias.grad[1].item() != 0.0


def test_radial_l1_policy_masks_only_the_excluded_sign() -> None:
    config = replace(
        _config("masked-radial", instruments=2),
        objective="radial-l1-v2",
        mask_arbitrage_candidates=True,
    )
    policy = _radial_policy(config)
    final = policy.network.In[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias[:] = torch.tensor([2.0, -3.0])
    excluded = torch.tensor([[[True, False, False, True]]])

    action, raw_l1 = policy(
        torch.zeros(1, 1, config.state_features), excluded
    )
    assert torch.equal(action, torch.zeros_like(action))
    assert raw_l1.item() == 0.0

    with torch.no_grad():
        final.bias[:] = torch.tensor([-2.0, 3.0])
    action, raw_l1 = policy(
        torch.zeros(1, 1, config.state_features), excluded
    )
    assert torch.allclose(action, torch.tensor([[[-0.4, 0.6]]]))
    assert raw_l1.item() == pytest.approx(5.0)
    policy = _policy(config)
    z = torch.randn(4, 2, config.state_features)
    excluded = torch.zeros(4, 2, 2 * config.instruments, dtype=torch.bool)
    excluded[..., 1] = True
    excluded[..., config.instruments] = True

    long, short, no_trade = policy(z, excluded)

    assert torch.all(long[..., 1] == 0.0)
    assert torch.all(short[..., 0] == 0.0)
    assert torch.allclose(
        (long + short).sum(dim=-1) + no_trade,
        torch.ones_like(no_trade),
    )


def test_separate_long_short_net_return_includes_both_costs() -> None:
    long = torch.tensor([[[0.4, 0.1]]])
    short = torch.tensor([[[0.2, 0.3]]])
    dH = torch.tensor([[[2.0, -1.0]]])
    cost = torch.tensor([[[0.1, 0.2]]])

    actual = net_return(long=long, short=short, dH=dH, cost=cost)
    expected = (0.4 - 0.2) * 2.0 + (0.1 - 0.3) * -1.0
    expected -= (0.4 + 0.2) * 0.1 + (0.1 + 0.3) * 0.2
    assert actual.item() == pytest.approx(expected)


def test_direct_loss_prefers_a_uniformly_profitable_trade() -> None:
    config = _config("loss", instruments=1)
    shape = (8, 2, 1)
    zero = torch.zeros(shape)
    no_trade_loss, _ = direct_arbitrage_loss(
        long=zero,
        short=zero,
        dH=torch.full(shape, 0.03),
        cost=zero,
        config=config,
    )
    profitable_loss, diagnostics = direct_arbitrage_loss(
        long=torch.ones(shape),
        short=zero,
        dH=torch.full(shape, 0.03),
        cost=torch.full(shape, 0.001),
        config=config,
    )

    assert profitable_loss < no_trade_loss
    assert diagnostics["pnl"].min().item() == pytest.approx(0.029)
    assert diagnostics["downside_mean_scaled"].item() == 0.0


def test_v2_loss_uses_exact_downside_tail_and_cap_penalty() -> None:
    config = replace(
        _config("v2-loss", instruments=1),
        objective="radial-l1-v2",
        downside_weight=2.0,
        tail_weight=3.0,
        cap_weight=4.0,
        tail_fraction=0.5,
    )
    action = torch.tensor([[[1.0]], [[-1.0]]])
    dH = torch.tensor([[[0.2]], [[0.1]]])
    cost = torch.zeros_like(dH)
    raw_l1 = torch.tensor([[2.0], [0.5]], requires_grad=True)

    loss, diagnostics = v2_arbitrage_loss(
        action=action,
        raw_l1=raw_l1,
        dH=dH,
        cost=cost,
        config=config,
    )
    # pnl=(0.2,-0.1): -mean + 2*mean downside + 3*worst-half + 4*mean cap^2
    assert loss.item() == pytest.approx(-0.05 + 0.1 + 0.3 + 2.0)
    assert diagnostics["downside_mean"].item() == pytest.approx(0.05)
    loss.backward()
    assert raw_l1.grad is not None
    assert raw_l1.grad[0].item() == pytest.approx(4.0)
    assert raw_l1.grad[1].item() == 0.0


def test_costs_use_training_population_error_and_conditional_maximum() -> None:
    dH = torch.tensor(
        [
            [[1.0, -2.0], [3.0, 0.0]],
            [[5.0, 2.0], [7.0, 4.0]],
        ]
    )
    data = {
        "z": torch.zeros(2, 2, 3),
        "dH": dH,
        "H": torch.full_like(dH, 10.0),
    }
    bar_c = unconditional_cost(data, cost_uncond=1.0, batch_size=1)
    expected = dH.reshape(-1, 2).std(dim=0, correction=0) / 2.0
    assert torch.allclose(bar_c, expected)

    class ConditionalModel:
        @staticmethod
        def moment_outputs(z: torch.Tensor, returns: torch.Tensor):
            del z
            variance = torch.full_like(returns, 0.16)
            return torch.zeros_like(returns), variance, variance

    cost = effective_cost(
        data,
        bar_c=bar_c,
        cost_prop=0.001,
        conditional_variance_model=ConditionalModel(),
        n_cost=4,
        batch_size=1,
    )
    expected_conditional = torch.full_like(cost, 0.2)
    expected_effective = torch.maximum(
        expected_conditional,
        torch.maximum(
            bar_c.reshape(1, 1, -1),
            torch.full_like(cost, 0.01),
        ),
    )
    assert torch.allclose(cost, expected_effective)


def test_independent_validation_rejects_a_training_only_opportunity() -> None:
    config = _config("validation-isolation", instruments=1)
    policy = _policy(config)
    final = policy.network.In[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias[:] = torch.tensor([20.0, -20.0, 20.0])

    z = torch.zeros(100, 2, config.state_features)
    cost = torch.zeros(100, 2, 1)
    training = {
        "z": z,
        "dH": torch.full((100, 2, 1), 0.02),
        "cost": cost,
    }
    validation_returns = torch.full((100, 2, 1), 0.02)
    validation_returns[0, 0, 0] = -0.02
    validation = {"z": z, "dH": validation_returns, "cost": cost}

    training_value = valuate_policy(
        policy, training, batch_size=17, split="training"
    )
    validation_value = valuate_policy(
        policy, validation, batch_size=17, split="validation"
    )
    assert training_value["qualifies_empirically"]
    assert training_value["negative_observations_beyond_tolerance"] == 0
    assert not validation_value["qualifies_empirically"]
    assert validation_value["negative_observations_beyond_tolerance"] == 1
    assert validation_value["losing_paths"] == 1


def test_small_training_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _config("checkpoint", instruments=1)
    z = torch.zeros(20, 2, config.state_features)
    data = {
        "z": z,
        "dH": torch.full((20, 2, 1), 0.03),
        "cost": torch.zeros(20, 2, 1),
    }
    train_config = DirectArbitrageTrainConfig(
        batch_size=5,
        valuation_batch_size=7,
        lr=1.0e-2,
        final_lr=1.0e-3,
        compile=False,
        value_update=2,
        checkpoint_update=2,
        checkpoint_root=str(tmp_path),
    )
    first = train_direct_policy(
        data,
        data,
        config=config,
        train_config=train_config,
        epochs=2,
        env=_env(),
        restart=True,
    )
    second = train_direct_policy(
        data,
        data,
        config=config,
        train_config=train_config,
        epochs=4,
        env=_env(),
    )

    assert first.checkpoint_path.exists()
    assert second.restarted
    assert second.tracker["epochs"][-1] == 4
    assert torch.isfinite(torch.tensor(second.training["pnl_mean"]))
