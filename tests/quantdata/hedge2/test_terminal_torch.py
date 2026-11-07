"""Terminal hedge values and policy gradients use the same full-horizon economics."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from quantdata.hedge2.terminal import evaluate_terminal_policy, terminal_risk
from quantdata.hedge2.terminal_torch import (
    TensorTerminalPaths,
    evaluate_tensor_policy,
    tensor_terminal_risk,
)
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_tree import two_step_tree


def test_device_alias_is_resolved_by_the_actual_input_tensor() -> None:
    env = BaseEnvironment(device="cpu:0", precision=64)
    paths = TensorTerminalPaths.from_paths(two_step_tree(), env=env)
    assert paths.gains.device != env.device
    result = evaluate_tensor_policy(paths=paths, initial_inventory=torch.zeros(1, dtype=env.fdtype),
        policy=lambda features, inventory, step: inventory, target_side=-1)
    assert result.wealth.device == paths.gains.device


@pytest.mark.parametrize("side", [-1, 1])
def test_tensor_accounting_matches_independent_replay_with_settlement_and_ko(side) -> None:
    original = two_step_tree()
    gains = np.repeat(original.gains, 2, axis=2)
    gains[:, 1, 0] = 0.
    stops = np.zeros((4, 2), dtype=bool)
    stops[:2, 0] = True
    paths = replace(original, gains=gains, costs=np.full_like(gains, .1),
        live_contracts=np.array([[True, True], [False, True], [False, False]]), terminate_after_step=stops)
    env = BaseEnvironment(device="cpu", precision=64, debug=True)
    initial = np.array([.4, -.2])

    def numpy_policy(features, inventory, step):
        return .25 + .1 * features + .2 * inventory + .03 * step

    def torch_policy(features, inventory, step):
        return .25 + .1 * features + .2 * inventory + .03 * step

    expected = evaluate_terminal_policy(paths=paths, initial_inventory=initial,
        policy=numpy_policy, target_side=side, position_cap=None)
    actual = evaluate_tensor_policy(paths=TensorTerminalPaths.from_paths(paths, env=env),
        initial_inventory=torch.tensor(initial), policy=torch_policy, target_side=side)
    for name in ("wealth", "hedge_gain", "entry_cost", "positions", "step_gains", "step_costs"):
        np.testing.assert_allclose(getattr(actual, name).detach().numpy(), getattr(expected, name), atol=1e-14)
    assert actual.wealth.dtype == env.fdtype


@pytest.mark.parametrize("objective,risk", [("mean_variance", 1.), ("mean_variance", 1000.),
    ("mean_variance", np.inf), ("power_exponential", 1.), ("power_exponential", 1000.)])
def test_tensor_terminal_risk_value_and_first_derivative_match_numpy(objective, risk) -> None:
    values = np.array([-.08, -.01, .03, .2])
    probabilities = np.array([.1, .2, .3, .4])
    expected = terminal_risk(values, probabilities, objective=objective, risk_aversion=risk)
    wealth = torch.tensor(values, requires_grad=True)
    actual = tensor_terminal_risk(wealth, torch.tensor(probabilities), objective=objective, risk_aversion=risk)
    assert float(actual.value.detach()) == pytest.approx(expected.value, abs=1e-12)
    actual.value.backward()
    np.testing.assert_allclose(wealth.grad.numpy(), expected.wealth_gradient, atol=1e-11)
    if objective == "power_exponential":
        assert float(actual.cash_shift) == pytest.approx(expected.cash_shift, abs=1e-12)
        assert abs(float(actual.cash_first_order_residual.detach())) < 1e-12


@pytest.mark.parametrize("objective", ["mean_variance", "power_exponential"])
def test_policy_gradient_includes_future_inventory_turnover_and_terminal_payoff(objective) -> None:
    env = BaseEnvironment(device="cpu", precision=64)
    paths = TensorTerminalPaths.from_paths(two_step_tree(), env=env)

    def loss(parameters):
        def policy(features, inventory, step):
            return parameters[step] + parameters[2] * features + parameters[3] * inventory

        evaluated = evaluate_tensor_policy(paths=paths, initial_inventory=torch.tensor([.3]),
            policy=policy, target_side=-1)
        return tensor_terminal_risk(evaluated.wealth, paths.weights, objective=objective, risk_aversion=10.).value

    parameters = torch.tensor([.2, -.1, .3, .2], dtype=env.fdtype, requires_grad=True)
    assert torch.autograd.gradcheck(loss, (parameters,), eps=1e-6, atol=2e-6, rtol=2e-5)


def test_constant_terminal_wealth_has_exact_cash_shift_and_unit_marginal() -> None:
    wealth = torch.full((4,), .03, dtype=torch.float64, requires_grad=True)
    result = tensor_terminal_risk(wealth, torch.full((4,), .25, dtype=torch.float64),
        objective="power_exponential", risk_aversion=1000.)
    assert float(result.value.detach()) == pytest.approx(-.03)
    assert float(result.cash_shift) == pytest.approx(.03)
    result.value.backward()
    np.testing.assert_allclose(wealth.grad.numpy(), -.25)


def test_initial_inventory_must_respect_the_same_cap_as_the_policy() -> None:
    paths = TensorTerminalPaths.from_paths(two_step_tree(), env=BaseEnvironment(device="cpu", precision=64))
    with pytest.raises(ValueError, match="current hedge inventory exceeds"):
        evaluate_tensor_policy(paths=paths, initial_inventory=torch.tensor([3.], dtype=torch.float64),
            policy=lambda features, inventory, step: torch.zeros_like(inventory), target_side=1, position_cap=2.)


@pytest.mark.parametrize("risk", [1., 10., 100., 1000.])
def test_float32_cash_residual_and_gradient_have_precision_appropriate_error(risk) -> None:
    wealth = torch.tensor([-.08, -.01, .03, .2], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([.1, .2, .3, .4], dtype=torch.float32)
    result = tensor_terminal_risk(wealth, weights, objective="power_exponential", risk_aversion=risk)
    expected = terminal_risk(wealth.detach().numpy().astype(float), weights.numpy().astype(float),
        objective="power_exponential", risk_aversion=risk)
    result.value.backward()
    tolerance = 16. * np.finfo(np.float32).eps * (1. + risk * float(wealth.detach().max()-wealth.detach().min()))
    assert abs(float(result.cash_first_order_residual.detach())) < tolerance
    assert float(result.value.detach()) == pytest.approx(expected.value, abs=tolerance)
    np.testing.assert_allclose(wealth.grad.numpy(), expected.wealth_gradient, atol=tolerance)
