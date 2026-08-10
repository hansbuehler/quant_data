from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.dysanos.grid.rndysanos.quadratic_action import (
    DirectActionNetwork,
    QuadraticActionConfig,
    QuadraticActionModel,
    conditional_gaussian_entropy_action,
    downside_oce_utility,
    objective_gradient,
    valuate_quadratic_action,
)
from quantdata.dysanos.grid.rndysanos.torch_drift_change import ScaledTanhNetwork
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _environment() -> BaseEnvironment:
    return BaseEnvironment(precision=64, device="cpu", debug=True)


def _constant_data(
    env: BaseEnvironment,
    *,
    paths: int = 8,
    steps: int = 2,
    value: float = 0.4,
) -> dict[str, torch.Tensor]:
    return {
        "z": torch.zeros(
            (paths, steps, 1), dtype=env.fdtype, device=env.device
        ),
        "dH": torch.full(
            (paths, steps, 1), value, dtype=env.fdtype, device=env.device
        ),
        "sampling_weight": torch.ones(
            paths, dtype=env.fdtype, device=env.device
        ),
    }


def _linear_model(
    env: BaseEnvironment,
    *,
    utility: str = "entropy",
    bounded: bool = False,
) -> QuadraticActionModel:
    return QuadraticActionModel(
        env,
        num_input=1,
        num_instruments=1,
        config=QuadraticActionConfig(
            utility=utility,
            quadratic_coefficient=0.5,
            bounded_action=bounded,
            network_config=FeedForwardConfig.linear_config(),
        ),
    )


def _output_bias(model: QuadraticActionModel) -> torch.nn.Parameter:
    network = model.action.network
    assert network.I0.bias is not None
    return network.I0.bias


def test_quadratic_config_rejects_nonpositive_penalty() -> None:
    with pytest.raises(ValueError, match="quadratic_coefficient"):
        QuadraticActionConfig(quadratic_coefficient=0.0)


def test_quadratic_config_rejects_invalid_downside_power() -> None:
    with pytest.raises(ValueError, match="downside_power"):
        QuadraticActionConfig(utility="downside", downside_power=0)


def test_direct_action_uses_feed_forward_and_zero_initializes() -> None:
    env = _environment()
    model = _linear_model(env)
    assert isinstance(model.action, DirectActionNetwork)
    data = _constant_data(env)
    action = model.action(data["z"])
    assert torch.count_nonzero(action).item() == 0

    loss = model(data)
    loss.backward()
    bias = _output_bias(model)
    assert bias.grad is not None
    assert torch.isfinite(bias.grad).all()
    assert bias.grad.item() < 0.0


@pytest.mark.parametrize("utility", ["entropy", "powexp", "downside"])
def test_all_utilities_learn_the_oce_cash_offset(utility: str) -> None:
    env = _environment()
    model = _linear_model(env, utility=utility)
    assert isinstance(model.cash_offset, torch.nn.Parameter)

    loss = model(_constant_data(env))
    loss.backward()
    assert model.cash_offset.grad is not None
    assert torch.isfinite(model.cash_offset.grad)


@pytest.mark.parametrize("utility", ["entropy", "powexp"])
def test_deterministic_return_optimum_has_zero_action_derivative(utility: str) -> None:
    env = _environment()
    model = _linear_model(env, utility=utility)
    with torch.no_grad():
        _output_bias(model).fill_(0.4)
    data = _constant_data(env, value=0.4)

    loss = model(data)
    loss.backward()
    bias = _output_bias(model)
    assert bias.grad is not None
    assert bias.grad.item() == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize("utility", ["entropy", "powexp"])
def test_quadratic_penalty_restores_large_action(utility: str) -> None:
    env = _environment()
    model = _linear_model(env, utility=utility)
    with torch.no_grad():
        _output_bias(model).fill_(10.0)
    data = _constant_data(env, value=0.4)

    loss = model(data)
    loss.backward()
    bias = _output_bias(model)
    assert bias.grad is not None
    # Gradient descent subtracts this positive gradient and reduces the action.
    assert bias.grad.item() > 0.0
    assert torch.isfinite(bias.grad).all()


def test_conditional_gaussian_entropy_action() -> None:
    mean = torch.tensor([0.3, -0.4], dtype=torch.float64)
    covariance = torch.tensor(
        [[0.2, 0.05], [0.05, 0.1]], dtype=torch.float64
    )
    action = conditional_gaussian_entropy_action(
        mean,
        covariance,
        lmbda=2.0,
        quadratic_coefficient=0.5,
    )
    expected = torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) + 2.0 * covariance,
        mean,
    )
    assert torch.allclose(action, expected)


@pytest.mark.parametrize(
    ("power", "expected_utility", "expected_marginal"),
    [
        (1, [-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        (2, [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]),
        (3, [-2.0, 0.0, 0.0], [6.0, 0.0, 0.0]),
    ],
)
def test_downside_oce_utility_and_marginal(
    power: int,
    expected_utility: list[float],
    expected_marginal: list[float],
) -> None:
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    utility, marginal = downside_oce_utility(
        x,
        torch.zeros((), dtype=torch.float64),
        scale=2.0,
        power=power,
    )
    assert torch.allclose(
        utility, torch.tensor(expected_utility, dtype=torch.float64)
    )
    assert torch.allclose(
        marginal, torch.tensor(expected_marginal, dtype=torch.float64)
    )


def test_downside_cvar_offset_selects_requested_tail_mass() -> None:
    x = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
    _, marginal = downside_oce_utility(
        x,
        torch.zeros((), dtype=torch.float64),
        scale=2.0,
        power=1,
    )
    # The OCE offset derivative is E[u'(X+y)]-1.
    assert marginal.mean().item() - 1.0 == pytest.approx(0.0)


def test_downside_power_two_is_c1_but_not_c2() -> None:
    scale = 2.0
    epsilon = 1.0e-6

    def derivatives(value: float, power: int) -> tuple[float, float]:
        x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        utility, _ = downside_oce_utility(
            x,
            torch.zeros((), dtype=torch.float64),
            scale=scale,
            power=power,
        )
        first = torch.autograd.grad(utility, x, create_graph=True)[0]
        second = torch.autograd.grad(first, x)[0]
        return first.item(), second.item()

    left_first, left_second = derivatives(-epsilon, 2)
    right_first, right_second = derivatives(epsilon, 2)
    assert left_first == pytest.approx(0.0, abs=1.0e-5)
    assert right_first == pytest.approx(0.0)
    assert left_second == pytest.approx(-2.0 * scale)
    assert right_second == pytest.approx(0.0)

    _, cubic_left_second = derivatives(-epsilon, 3)
    _, cubic_right_second = derivatives(epsilon, 3)
    assert cubic_left_second == pytest.approx(0.0, abs=1.0e-4)
    assert cubic_right_second == pytest.approx(0.0)


def test_valuation_exposes_nonmartingale_density_identity() -> None:
    env = _environment()
    model = _linear_model(env)
    with torch.no_grad():
        _output_bias(model).fill_(0.4)
    data = _constant_data(env, paths=12, value=0.4)

    valuation = valuate_quadratic_action(model, data, batch_size=5)
    assert valuation["first_order_residual_max_abs_z"] == pytest.approx(0.0)
    assert valuation["marginal_density_identity_max_abs"] == pytest.approx(0.0)
    assert valuation["marginal_density_martingale_max_abs"] == pytest.approx(0.4)
    assert np.asarray(valuation["action_mean"]).shape == (2, 1)


def test_bounded_action_reports_saturation() -> None:
    env = _environment()
    model = _linear_model(env, bounded=True)
    assert isinstance(model.action, ScaledTanhNetwork)
    with torch.no_grad():
        assert model.action.network.I0.bias is not None
        model.action.network.I0.bias.fill_(10.0)
    valuation = valuate_quadratic_action(
        model,
        _constant_data(env),
        batch_size=3,
    )
    assert valuation["action_cap_saturation_rate_max"] == pytest.approx(1.0)


def test_exact_objective_gradient_accumulates_path_batches() -> None:
    env = _environment()
    model = _linear_model(env)
    data = _constant_data(env, paths=11)
    diagnostics = objective_gradient(model, data, batch_size=4)
    assert diagnostics["loss"] == pytest.approx(float(model(data).item()))
    assert diagnostics["gradient_norm"] > 0.0
    assert diagnostics["nonfinite_gradient_count"] == 0
