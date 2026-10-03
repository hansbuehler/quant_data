from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from quantdata.dysanos.flex.dynamic_option_hedging import (
    DynamicOptionPolicy,
    DynamicOptionPolicyConfig,
    five_day_straddle_claim,
    interpolated_option_returns,
    marginal_utility_density,
    oce_value,
    power_exponential_utility,
)
from quantdata.dysanos.flex.run_dynamic_option_deep_hedging import (
    _antithetic_validation_chunks,
)
from quantdata.torch import BaseEnvironment


class _Decoder:
    def __init__(self) -> None:
        self.model_expiries_y = torch.tensor([1.0, 5.0, 10.0, 25.0]) / 255.0

    def price_calls_batch(
        self,
        surface: SimpleNamespace,
        *,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
    ) -> torch.Tensor:
        del surface
        # Smooth, positive test prices; the exact financial model is tested elsewhere.
        return 0.08 + 0.2 * expiries_y + 0.03 * (strikes - 1.0).square()


def _model_and_surface(paths: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    decoder = _Decoder()
    model = SimpleNamespace(sanos_decoder=decoder)
    supports = torch.tensor([0.8, 1.0, 1.2])[None, None].expand(paths, 4, -1)
    densities = torch.tensor([0.25, 0.5, 0.25])[None, None].expand(paths, 4, -1)
    surface = SimpleNamespace(
        total_vols=torch.tensor([0.1, 0.2, 0.3, 0.5])[None].expand(paths, -1),
        backbone_total_vols=torch.tensor([0.05, 0.1, 0.15, 0.25])[
            None
        ].expand(paths, -1),
        support_strikes=supports,
        densities=densities,
        density0=densities[:, 0],
    )
    return model, surface


def test_policy_is_initially_zero_and_respects_l1_cap() -> None:
    env = BaseEnvironment(precision=32, device="cpu", debug=True)
    config = DynamicOptionPolicyConfig(
        label="test", state_size=3, hidden_size=4, option_slots=2, maximum_action_l1=1.5
    )
    policy = DynamicOptionPolicy(
        env,
        config=config,
        state_mean=torch.zeros(3),
        state_scale=torch.ones(3),
    )
    states = torch.randn(7, 3)
    decision = policy(states, torch.zeros(7), 0.25, policy.initial_hidden(7))
    assert torch.equal(decision.quantities, torch.zeros(7, 3))
    assert not torch.equal(decision.nms[:, 0], decision.nms[:, 1])
    assert not torch.equal(
        decision.maturity_fraction[:, 0], decision.maturity_fraction[:, 1]
    )

    with torch.no_grad():
        policy.affine.bias[:3].fill_(10.0)
    decision = policy(states, torch.zeros(7), 0.25, policy.initial_hidden(7))
    assert torch.all(torch.sum(torch.abs(decision.quantities), dim=1) <= 1.5 + 1.0e-6)


def test_calendar_interpolation_uses_same_weights_for_price_and_payoff() -> None:
    paths = 2
    model, surface = _model_and_surface(paths)
    spots = torch.tensor(
        [[1.0, 1.01, 1.03, 1.02, 1.04, 1.05], [1.0, 0.99, 0.98, 1.01, 1.0, 0.97]]
    )
    maturity = torch.tensor([[0.375], [0.375]], requires_grad=True)
    nms = torch.zeros((paths, 1), requires_grad=True)
    option_return, details = interpolated_option_returns(
        model,
        surface,
        issue_day=0,
        remaining_days=5,
        path_spots=spots,
        nms=nms,
        maturity_fraction=maturity,
        put_weight=torch.zeros_like(nms),
    )
    assert torch.allclose(details["continuous_days"], torch.full((paths, 1), 2.5))
    assert torch.equal(details["lower_days"], torch.full((paths, 1), 2.0))
    assert torch.equal(details["upper_days"], torch.full((paths, 1), 3.0))
    assert torch.allclose(details["alpha"], torch.full((paths, 1), 0.5))
    option_return.sum().backward()
    assert maturity.grad is not None and torch.all(torch.isfinite(maturity.grad))
    assert nms.grad is not None and torch.all(torch.isfinite(nms.grad))


def test_tiny_premium_option_is_an_exact_no_trade() -> None:
    model, surface = _model_and_surface(1)
    model.sanos_decoder.price_calls_batch = (
        lambda surface, *, expiries_y, strikes: torch.full_like(strikes, 1.0e-8)
    )
    option_return, details = interpolated_option_returns(
        model,
        surface,
        issue_day=0,
        remaining_days=5,
        path_spots=torch.ones((1, 6)),
        nms=torch.zeros((1, 1)),
        maturity_fraction=torch.full((1, 1), 0.5),
        put_weight=torch.zeros((1, 1)),
        price_floor=1.0e-6,
    )
    assert not bool(details["eligible"].item())
    assert option_return.item() == 0.0


def test_claim_and_oce_normalization() -> None:
    spots = torch.ones((3, 26))
    spots[0, 5] = 1.1
    spots[0, 10:] = 1.1
    claim = five_day_straddle_claim(spots)
    assert claim.tolist() == pytest.approx([0.1, 0.0, 0.0])

    wealth = torch.tensor([-0.2, 0.0, 0.3], dtype=torch.float64)
    entropy = oce_value(wealth, utility="entropy", risk_aversion=1.0)
    assert torch.isfinite(entropy)
    density = marginal_utility_density(wealth, utility="entropy", risk_aversion=1.0)
    assert torch.mean(density).item() == pytest.approx(1.0)

    cash = torch.tensor(0.1, dtype=torch.float64)
    power = oce_value(
        wealth, utility="power-exp", risk_aversion=2.0, cash_translation=cash
    )
    assert torch.isfinite(power)
    power_density = marginal_utility_density(
        wealth,
        utility="power-exp",
        risk_aversion=2.0,
        cash_translation=cash,
    )
    assert torch.mean(power_density).item() == pytest.approx(1.0)
    assert torch.all(torch.diff(power_exponential_utility(wealth, 2.0)) > 0.0)


def test_power_exponential_large_loss_has_finite_gradient() -> None:
    value = torch.tensor([-1.0e6, 0.2], dtype=torch.float64, requires_grad=True)
    utility = power_exponential_utility(value, 10.0)
    utility.sum().backward()
    assert torch.all(torch.isfinite(utility))
    assert value.grad is not None and torch.all(torch.isfinite(value.grad))


def test_validation_chunks_keep_antithetic_pairs_together() -> None:
    chunks = _antithetic_validation_chunks(2_000, batch_size=500, block_size=1_000)
    assert len(chunks) == 4
    for chunk in chunks:
        left, right = chunk[:250], chunk[250:]
        assert (right - left == 500).all()
