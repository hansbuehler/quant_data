from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from scipy.stats import norm

from quantdata.dysanos.grid.gendysanos.cell_measure_change import (
    CellMeasureChangeConfig,
    IndependentCellAction,
    IndependentFeedForward,
    cell_pnl,
    cell_utility_and_marginal,
)
from quantdata.dysanos.grid.gendysanos.conditional_static_arbitrage import (
    frozen_payoff_cone_lower_bound,
    solve_conditional_support_lp,
    terminal_portfolio_pnl,
    weighted_mean_standard_error,
)
from quantdata.dysanos.grid.gendysanos.expiry_arbitrage import (
    ExpiryArbitrageConfig,
    ExpiryArbitrageModel,
    exact_payoff_cone_lower_bound,
    expiry_portfolio_pnl,
    payoff_cone_numerical_band,
    payoff_cone_values,
    powerexp_oce_by_expiry,
    propagated_expiry_band,
    smooth_payoff_cone_lower_bound,
    solve_payoff_cone_lp,
    weighted_expiry_mean,
)
from quantdata.dysanos.grid.gendysanos.multi_expiry_arbitrage import (
    multi_expiry_lower_bound,
    multi_expiry_terminal_pnl,
    solve_local_multi_expiry_portfolio,
    solve_multi_expiry_static_arbitrage_lp,
    weighted_quantile,
)
from quantdata.dysanos.grid.gendysanos.required_runs import propagated_basket_band
from quantdata.dysanos.grid.gendysanos.run_batched_multi_expiry_arbitrage import (
    _audit_batches,
    _weighted_stats_with_quantiles,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility


def _config(*, method: str = "negative-part") -> ExpiryArbitrageConfig:
    return ExpiryArbitrageConfig(
        label=f"test-{method}",
        state_features=2,
        trading_times=2,
        expiries=2,
        instruments_per_expiry=3,
        method=method,
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=5,
            activation="LeakyReLU",
            use_bias=True,
        ),
    )


def _model(*, method: str = "negative-part") -> ExpiryArbitrageModel:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    return ExpiryArbitrageModel(
        env,
        config=_config(method=method),
        state_mean=torch.zeros(2, dtype=env.fdtype),
        state_scale=torch.ones(2, dtype=env.fdtype),
    )


def test_expiry_network_uses_project_feedforward_and_keeps_expiries_separate() -> None:
    model = _model()
    assert isinstance(model, BaseModel)
    assert all(isinstance(network, FeedForward) for network in model.action.networks)
    assert all(network.env is model.env for network in model.action.networks)

    state = torch.randn(4, 2, 2, dtype=model.fdtype, requires_grad=True)
    action, raw_l1, cap_penalty = model(state)
    assert action.shape == (4, 2, 2, 3)
    assert raw_l1.shape == cap_penalty.shape == (4, 2, 2)
    assert torch.equal(action, torch.zeros_like(action))

    # A loss on expiry zero must not create gradients in expiry one's network.
    (action[:, :, 0] * torch.arange(1, 4, dtype=model.fdtype)).sum().backward()
    assert any(parameter.grad is not None for parameter in model.action.networks[0].parameters())
    assert all(
        parameter.grad is None or torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        for parameter in model.action.networks[1].parameters()
    )


def test_radial_l1_map_has_an_exterior_recovery_gradient() -> None:
    model = _model()
    with torch.no_grad():
        for network in model.action.networks:
            output = network.In[-1] if network.In else network.I0
            output.weight.zero_()
            assert output.bias is not None
            output.bias.copy_(torch.tensor([2.0, -1.0, 0.5], dtype=model.fdtype))

    action, raw_l1, cap_penalty = model(torch.zeros(3, 2, 2, dtype=model.fdtype))
    assert torch.allclose(action.abs().sum(dim=-1), torch.ones_like(raw_l1))
    assert torch.allclose(raw_l1, torch.full_like(raw_l1, 3.5))
    cap_penalty.mean().backward()
    for network in model.action.networks:
        output = network.In[-1] if network.In else network.I0
        assert output.bias is not None and output.bias.grad is not None
        assert torch.linalg.vector_norm(output.bias.grad) > 0.0
        assert torch.isfinite(output.bias.grad).all()


def test_expiry_path_pnl_never_combines_expiry_axes() -> None:
    action = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, -0.5], [0.25, 0.75]]]],
        dtype=torch.float64,
    )
    returns = torch.tensor(
        [[[[2.0, 9.0], [8.0, 3.0]], [[4.0, 2.0], [1.0, 5.0]]]],
        dtype=torch.float64,
    )
    costs = torch.full_like(returns, 0.1)
    time_pnl, path_pnl = expiry_portfolio_pnl(action, returns, costs)
    expected_time = torch.tensor([[[1.9, 2.9], [0.9, 3.9]]], dtype=torch.float64)
    assert torch.allclose(time_pnl, expected_time)
    assert torch.allclose(path_pnl, torch.tensor([[2.8, 6.8]], dtype=torch.float64))


def test_torch_propagated_band_matches_numpy_formula_time_by_time() -> None:
    rng = np.random.default_rng(29)
    action_np = rng.normal(size=(2, 3, 2, 4))
    returns_np = rng.normal(size=action_np.shape)
    costs_np = np.abs(rng.normal(scale=0.01, size=action_np.shape))
    band_np = np.abs(rng.normal(scale=1.0e-6, size=action_np.shape))
    time, path = propagated_expiry_band(
        torch.tensor(action_np),
        torch.tensor(returns_np),
        torch.tensor(costs_np),
        torch.tensor(band_np),
    )
    expected = np.empty((2, 3, 2))
    for sample in range(2):
        for trading_time in range(3):
            for expiry in range(2):
                expected[sample, trading_time, expiry] = propagated_basket_band(
                    action_np[sample, trading_time, expiry],
                    returns_np[sample, trading_time, expiry][None, :],
                    costs_np[sample, trading_time, expiry][None, :],
                    band_np[sample, trading_time, expiry][None, :],
                )[0]
    assert np.allclose(time.numpy(), expected)
    assert np.allclose(path.numpy(), expected.sum(axis=1))


def _cone_inputs(*, delta_values: tuple[float, float] = (0.0, 0.0)) -> dict[str, torch.Tensor]:
    return {
        "spot": torch.tensor([[1.0]], dtype=torch.float64),
        "option_price": torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64),
        "delta": torch.tensor([[[[delta_values[0], delta_values[1]]]]], dtype=torch.float64),
        "cash_strike": torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64),
        "costs": torch.zeros((1, 1, 1, 3), dtype=torch.float64),
    }


def test_generic_payoff_cone_detects_a_wrongly_ordered_call_pair() -> None:
    inputs = _cone_inputs()
    action = torch.tensor([[[[0.0, 0.5, -0.5]]]], dtype=torch.float64)
    values, tail_slope = payoff_cone_values(action, hedge="raw", **inputs)
    time_worst, path_worst, exact_slope = exact_payoff_cone_lower_bound(
        action, hedge="raw", **inputs
    )
    assert values.shape == (1, 1, 1, 3)
    assert torch.allclose(tail_slope, torch.zeros_like(tail_slope))
    assert torch.equal(tail_slope, exact_slope)
    assert time_worst.item() == pytest.approx(0.5)
    assert path_worst.item() == pytest.approx(0.5)

    smooth_time, smooth_path, _ = smooth_payoff_cone_lower_bound(
        action, hedge="raw", temperature=0.01, **inputs
    )
    assert torch.isfinite(smooth_time).all()
    assert smooth_path.item() > 0.49

    lp_action, lp_margin = solve_payoff_cone_lp(
        spot=1.0,
        option_price=np.asarray([1.0, 2.0]),
        delta=np.asarray([0.0, 0.0]),
        cash_strike=np.asarray([1.0, 2.0]),
        costs=np.zeros(3),
        hedge="raw",
    )
    assert np.abs(lp_action).sum() == pytest.approx(1.0)
    assert lp_margin == pytest.approx(0.5)
    numerical_band = payoff_cone_numerical_band(
        lp_action,
        spot=1.0,
        option_price=np.asarray([1.0, 2.0]),
        delta=np.asarray([0.0, 0.0]),
        cash_strike=np.asarray([1.0, 2.0]),
        costs=np.zeros(3),
        hedge="raw",
    )
    expected_scale = max(
        1.0,
        float(np.sum(np.abs(lp_action) * np.asarray([1.0, 1.0, 2.0]))),
    )
    assert numerical_band == pytest.approx(
        32.0 * np.finfo(np.float32).eps * expected_scale
    )


def test_payoff_cone_broadcasts_spot_across_multiple_expiries() -> None:
    inputs = _cone_inputs()
    inputs["option_price"] = inputs["option_price"].expand(-1, -1, 2, -1).clone()
    inputs["delta"] = inputs["delta"].expand(-1, -1, 2, -1).clone()
    inputs["cash_strike"] = inputs["cash_strike"].expand(-1, -1, 2, -1).clone()
    inputs["costs"] = inputs["costs"].expand(-1, -1, 2, -1).clone()
    action = torch.tensor(
        [[[[0.0, 0.5, -0.5], [0.0, 0.5, -0.5]]]], dtype=torch.float64
    )
    values, slope = payoff_cone_values(action, hedge="raw", **inputs)
    assert values.shape == (1, 1, 2, 3)
    assert slope.shape == (1, 1, 2)


def test_delta_hedged_cone_includes_the_matched_spot_leg() -> None:
    inputs = _cone_inputs(delta_values=(0.4, 0.6))
    # The spot action cancels the aggregate entry-delta hedge.  Scaling to the
    # L1 boundary preserves every payoff sign.
    scale = 1.0 / 1.1
    action = scale * torch.tensor([[[[-0.1, 0.5, -0.5]]]], dtype=torch.float64)
    _, path_worst, slope = exact_payoff_cone_lower_bound(
        action, hedge="delta", **inputs
    )
    assert slope.item() == pytest.approx(0.0, abs=1.0e-14)
    assert path_worst.item() == pytest.approx(0.5 * scale)
    assert action.abs().sum().item() == pytest.approx(1.0)


def test_negative_upper_tail_slope_is_rejected_as_unbounded_loss() -> None:
    inputs = _cone_inputs()
    action = torch.tensor([[[[0.0, 0.0, -1.0]]]], dtype=torch.float64)
    time_worst, path_worst, slope = exact_payoff_cone_lower_bound(
        action, hedge="raw", **inputs
    )
    assert slope.item() < 0.0
    assert torch.isneginf(time_worst).all()
    assert torch.isneginf(path_worst).all()


def test_weighted_mean_uses_raw_importance_density_without_self_normalizing() -> None:
    value = torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.float64)
    weight = torch.tensor([0.5, 1.5], dtype=torch.float64)
    result = weighted_expiry_mean(value, weight)
    assert torch.allclose(result, torch.tensor([1.75, 4.5], dtype=torch.float64))


def test_powerexp_oce_finite_sample_mass_correction_matches_offset_gradient() -> None:
    model = _model(method="powerexp-oce")
    with torch.no_grad():
        model.cash_offset.copy_(torch.tensor([0.2, -0.1], dtype=model.fdtype))
    pnl = torch.tensor([[-1.0, 0.5], [0.25, -0.4]], dtype=model.fdtype)
    weight = torch.tensor([0.4, 1.2], dtype=model.fdtype)
    value, residual, marginal = powerexp_oce_by_expiry(model, pnl, weight)
    value.sum().backward()
    assert model.cash_offset.grad is not None
    assert torch.allclose(model.cash_offset.grad, residual, atol=1.0e-12, rtol=1.0e-12)
    assert torch.isfinite(value).all() and torch.isfinite(marginal).all()
    assert torch.all(marginal > 0.0)

    _, utility_derivative, _ = torch_utility("powexp", model.config.lmbda)
    expected = (weight[:, None] * utility_derivative(pnl, model.cash_offset)).mean(dim=0) - 1.0
    assert torch.allclose(residual, expected)


def test_smooth_cone_and_powerexp_have_finite_action_gradients() -> None:
    inputs = _cone_inputs()
    action = torch.zeros((1, 1, 1, 3), dtype=torch.float64, requires_grad=True)
    _, path, _ = smooth_payoff_cone_lower_bound(
        action, hedge="raw", temperature=0.02, **inputs
    )
    (-path.mean()).backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert torch.linalg.vector_norm(action.grad) > 0.0

    utility, _, _ = torch_utility("powexp", 1.0)
    x = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    utility(x, torch.zeros((), dtype=torch.float64)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.all(x.grad > 0.0)
    assert math.isclose(x.grad[1].item(), 1.0)


def test_terminal_portfolio_pnl_and_conditional_lp_use_observed_support() -> None:
    terminal_spot = np.asarray([0.0, 1.0, 2.0, 3.0])
    option_price = np.asarray([1.0, 2.0])
    delta = np.zeros(2)
    cash_strike = np.asarray([1.0, 2.0])
    costs = np.zeros(3)
    action, margin = solve_conditional_support_lp(
        terminal_spot,
        spot=1.0,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge="raw",
    )
    pnl = terminal_portfolio_pnl(
        action,
        terminal_spot,
        spot=1.0,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge="raw",
    )
    assert np.abs(action).sum() == pytest.approx(1.0)
    # The finite support admits a stronger bounded-tail trade than the exact
    # all-terminal-spot cone; its reported margin must equal the worst sample.
    assert margin > 0.5
    assert pnl.min() == pytest.approx(margin)
    exact_margin, tail_slope = frozen_payoff_cone_lower_bound(
        action,
        spot=1.0,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge="raw",
    )
    assert tail_slope < 0.0
    assert exact_margin == -math.inf


def test_terminal_delta_hedge_and_gross_cost_are_applied_without_netting() -> None:
    pnl = terminal_portfolio_pnl(
        np.asarray([0.25, -0.5]),
        np.asarray([0.8, 1.3]),
        spot=1.0,
        option_price=np.asarray([0.1]),
        delta=np.asarray([0.4]),
        cash_strike=np.asarray([1.0]),
        costs=np.asarray([0.01, 0.03]),
        hedge="delta",
    )
    spot_return = np.asarray([-0.2, 0.3])
    option_return = np.maximum(np.asarray([0.8, 1.3]) - 1.0, 0.0) - 0.1
    option_return -= 0.4 * spot_return
    expected = 0.25 * spot_return - 0.5 * option_return
    expected -= 0.25 * 0.01 + 0.5 * 0.03
    assert np.allclose(pnl, expected)


def test_weighted_mean_standard_error_matches_uniform_sample_formula() -> None:
    values = np.asarray([1.0, 2.0, 5.0, 8.0])
    result = weighted_mean_standard_error(values)
    assert result["mean"] == pytest.approx(values.mean())
    assert result["standard_error"] == pytest.approx(
        values.std(ddof=1) / math.sqrt(values.size)
    )
    weighted = weighted_mean_standard_error(values, np.asarray([0.0, 1.0, 2.0, 1.0]))
    assert weighted["mean"] == pytest.approx(5.0)
    assert weighted["effective_sample_size"] == pytest.approx(8.0 / 3.0)


def test_batched_audit_serializes_zero_variance_positive_payoff() -> None:
    result = _weighted_stats_with_quantiles(
        np.ones(8, dtype=np.float64),
        np.ones(8, dtype=np.float64),
    )
    assert result["mean"] == pytest.approx(1.0)
    assert result["standard_error"] == pytest.approx(0.0)
    assert "mean_over_standard_error" not in result
    json.dumps(result, allow_nan=False)


def _multi_expiry_inputs() -> dict[str, object]:
    return {
        "spot": 1.0,
        "option_price": np.asarray([[0.20, 0.10, 0.00], [0.35, 0.15, 0.05]]),
        "delta": np.zeros((2, 3)),
        "cash_strike": np.asarray([[0.8, 1.0, 1.2], [0.7, 1.0, 1.3]]),
        "costs": np.zeros((2, 4)),
        "hedge": "raw",
    }


def test_multi_expiry_pnl_sums_cashflows_at_their_own_expiries() -> None:
    inputs = _multi_expiry_inputs()
    action = np.asarray(
        [
            [0.0, 0.25, -0.50, 0.25],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    terminal_spot = np.asarray([[0.9, 0.8], [1.1, 1.4], [1.5, 0.5]])
    pnl = multi_expiry_terminal_pnl(action, terminal_spot, **inputs)
    first_expiry = (
        0.25 * (np.maximum(terminal_spot[:, 0] - 0.8, 0.0) - 0.20)
        - 0.50 * (np.maximum(terminal_spot[:, 0] - 1.0, 0.0) - 0.10)
        + 0.25 * np.maximum(terminal_spot[:, 0] - 1.2, 0.0)
    )
    assert np.allclose(pnl, first_expiry)
    lower, per_expiry, slope = multi_expiry_lower_bound(action, **inputs)
    assert lower == pytest.approx(0.0)
    assert np.allclose(per_expiry, 0.0)
    assert np.allclose(slope, 0.0)


def test_multi_expiry_lp_detects_zero_floor_nonzero_payoff() -> None:
    inputs = _multi_expiry_inputs()
    result = solve_multi_expiry_static_arbitrage_lp(**inputs)
    assert result.static_arbitrage_candidate
    assert not result.strict_certificate
    assert result.lower_bound >= -result.numerical_band
    assert result.witness_value > result.numerical_band
    assert np.abs(result.action).sum() == pytest.approx(1.0)
    terminal = np.asarray(
        [[0.0, 0.0], [0.9, 0.8], [1.0, 1.0], [1.1, 1.2], [2.0, 2.0]]
    )
    pnl = multi_expiry_terminal_pnl(result.action, terminal, **inputs)
    assert np.min(pnl) >= -result.numerical_band
    assert np.max(pnl) > result.numerical_band


def test_multi_expiry_lp_does_not_turn_tolerance_into_profit() -> None:
    inputs = _multi_expiry_inputs()
    inputs["option_price"] = np.asarray(
        [[0.25, 0.15, 0.08], [0.31, 0.16, 0.06]]
    )
    result = solve_multi_expiry_static_arbitrage_lp(**inputs)
    assert not result.static_arbitrage_candidate
    assert np.allclose(result.action, 0.0)


def test_multi_expiry_lp_accepts_black_scholes_call_surfaces() -> None:
    spot = 1.0
    strikes = np.asarray([[0.7, 1.0, 1.3], [0.7, 1.0, 1.3]])
    expiries = np.asarray([0.1, 0.5])[:, None]
    volatility = 0.25
    sigma_root_t = volatility * np.sqrt(expiries)
    d1 = (np.log(spot / strikes) + 0.5 * volatility**2 * expiries) / sigma_root_t
    d2 = d1 - sigma_root_t
    option_price = spot * norm.cdf(d1) - strikes * norm.cdf(d2)
    result = solve_multi_expiry_static_arbitrage_lp(
        spot=spot,
        option_price=option_price,
        delta=norm.cdf(d1),
        cash_strike=strikes,
        costs=np.full((2, 4), 1.0e-4),
        hedge="delta",
    )
    assert not result.static_arbitrage_candidate
    assert np.allclose(result.action, 0.0)


def test_local_multi_expiry_portfolio_keeps_training_chain_explicit() -> None:
    scenario_return = np.asarray(
        [
            [0.10, -0.20, 0.00],
            [0.20, -0.10, 0.10],
            [0.30, 0.00, 0.20],
            [0.40, 0.10, 0.30],
        ]
    )
    result = solve_local_multi_expiry_portfolio(
        scenario_return,
        np.zeros(3),
        weights=np.asarray([1.0, 1.0, 2.0, 2.0]),
    )
    assert result.training_candidate
    assert np.abs(result.action).sum() == pytest.approx(1.0)
    assert np.all(result.training_pnl >= -result.training_band)
    assert float(result.training_statistics["mean"]) > 0.0


def test_local_multi_expiry_portfolio_repairs_solver_scale_budget_error() -> None:
    scenario_return = np.asarray(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
        ]
    )
    result = solve_local_multi_expiry_portfolio(
        scenario_return,
        np.zeros(2),
    )
    assert np.abs(result.action).sum() <= 1.0
    assert np.all(result.training_pnl >= -result.training_band)


def test_local_constraint_generation_matches_full_lp() -> None:
    rng = np.random.default_rng(61)
    scenario_return = rng.normal(scale=0.05, size=(257, 8))
    scenario_return[:, 0] = 0.02 + rng.uniform(0.0, 0.03, size=257)
    costs = np.full(8, 0.002)
    weights = rng.uniform(0.25, 2.0, size=257)
    generated = solve_local_multi_expiry_portfolio(
        scenario_return,
        costs,
        weights=weights,
        initial_constraints=16,
    )
    full = solve_local_multi_expiry_portfolio(
        scenario_return,
        costs,
        weights=weights,
        initial_constraints=None,
    )
    generated_mean = float(generated.training_statistics["mean"])
    full_mean = float(full.training_statistics["mean"])
    assert generated_mean == pytest.approx(full_mean, abs=2.0e-8)
    assert np.min(generated.training_pnl) >= -1.1e-8
    assert np.min(full.training_pnl) >= -1.1e-8
    assert generated.active_constraints < scenario_return.shape[0]


def test_weighted_quantile_uses_normalized_nonnegative_weights() -> None:
    values = np.asarray([10.0, 1.0, 4.0, 8.0])
    weights = np.asarray([1.0, 4.0, 2.0, 1.0])
    assert weighted_quantile(values, weights, 0.0) == 1.0
    assert weighted_quantile(values, weights, 0.5) == 1.0
    assert weighted_quantile(values, weights, 0.75) == 4.0
    assert weighted_quantile(values, weights, 1.0) == 10.0


def test_audit_batches_are_disjoint_reproducible_path_clusters() -> None:
    component = np.arange(100) % 3
    first = _audit_batches(component, batches=5, batch_paths=10, seed=17)
    second = _audit_batches(component, batches=5, batch_paths=10, seed=17)
    assert first.shape == (5, 10)
    assert np.array_equal(first, second)
    assert np.unique(first).size == first.size


def _cell_model() -> IndependentCellAction:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = CellMeasureChangeConfig(
        label="test-cell-measure-change",
        source_cache_uid="unit-test-cache",
        state_features=2,
        risk_aversion=10.0,
        utility="exponential",
        trading_times=2,
        expiries=2,
        instruments_per_expiry=3,
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=5,
            activation="SiLU",
            use_bias=True,
        ),
    )
    return IndependentCellAction(
        env,
        config=config,
        state_mean=torch.zeros((2, 2), dtype=env.fdtype),
        state_scale=torch.ones((2, 2), dtype=env.fdtype),
        return_rms=torch.tensor(
            [[[2.0, 4.0, 5.0], [2.0, 4.0, 5.0]]] * 2,
            dtype=env.fdtype,
        ),
    )


def test_independent_cell_networks_have_no_parameter_or_gradient_sharing() -> None:
    model = _cell_model()
    assert isinstance(model, BaseModel)
    assert isinstance(model.network, IndependentFeedForward)
    assert model.network.cells == 4
    with torch.no_grad():
        model.network.biases[-1][2].copy_(
            torch.tensor([1.0, -2.0, 0.5], dtype=model.fdtype)
        )
    action = model(torch.zeros((4, 2, 2), dtype=model.fdtype))
    assert action.shape == (4, 2, 2, 3)
    assert torch.allclose(
        action[:, 1, 0],
        torch.tensor([0.05, -0.05, 0.01], dtype=model.fdtype).expand(4, -1),
    )
    action[:, 1, 0].sum().backward()
    for index in range(4):
        gradient_norm = sum(
            float(torch.linalg.vector_norm(parameter.grad[index]))
            for parameter in (*model.network.weights, *model.network.biases)
            if parameter.grad is not None
        )
        assert (gradient_norm > 0.0) is (index == 2)


def test_cell_pnl_and_all_utility_marginals_are_finite_and_differentiable() -> None:
    action = torch.tensor(
        [[[[1.0, -0.5]], [[0.25, 0.5]]], [[[0.5, 0.25]], [[-0.5, 1.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    returns = torch.tensor(
        [[[[0.2, -0.1]], [[0.3, 0.4]]], [[[0.1, 0.5]], [[-0.2, 0.1]]]],
        dtype=torch.float64,
    )
    costs = torch.full_like(returns, 0.01)
    pnl = cell_pnl(action, returns, costs)
    assert pnl.shape == (2, 2, 1)
    total = torch.zeros((), dtype=torch.float64)
    for utility in ("powerexp", "exponential", "mean-variance"):
        value, marginal = cell_utility_and_marginal(
            pnl,
            utility=utility,
            risk_aversion=100.0,
            weight=torch.ones(2, dtype=torch.float64),
        )
        assert value.shape == marginal.shape[1:] == (2, 1)
        assert torch.isfinite(value).all() and torch.isfinite(marginal).all()
        total = total + value.sum()
    total.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert torch.linalg.vector_norm(action.grad) > 0.0


@pytest.mark.parametrize("utility", ["powerexp", "exponential", "mean-variance"])
def test_uncapped_cell_objective_has_exact_risk_aversion_scaling(
    utility: str,
) -> None:
    base_gamma = 100.0
    base_pnl = torch.tensor(
        [[[-0.002, 0.001]], [[0.003, -0.001]], [[0.001, 0.002]]],
        dtype=torch.float64,
    )
    weight = torch.ones(3, dtype=torch.float64)
    base_value, base_marginal = cell_utility_and_marginal(
        base_pnl,
        utility=utility,
        risk_aversion=base_gamma,
        weight=weight,
    )
    for risk_aversion in (1.0, 10.0, 1_000.0, 10_000.0):
        scaled_pnl = base_pnl * (base_gamma / risk_aversion)
        value, marginal = cell_utility_and_marginal(
            scaled_pnl,
            utility=utility,
            risk_aversion=risk_aversion,
            weight=weight,
        )
        assert torch.allclose(
            risk_aversion * value,
            base_gamma * base_value,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert torch.allclose(marginal, base_marginal, atol=1.0e-12, rtol=1.0e-12)
