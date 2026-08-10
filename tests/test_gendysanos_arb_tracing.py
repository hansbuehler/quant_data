from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.dysanos.grid.gendysanos.arb_tracing import (
    ArbTracingConfig,
    OrthogonalArbDirection,
    fit_physical_state_whitener,
    traced_scale_statistics,
    tracing_numerical_band,
    tracing_pnl,
    tracing_pnl_right_derivative,
    tracing_utility_objective,
)
from quantdata.dysanos.grid.gendysanos.required_runs import derive_case_arrays
from quantdata.dysanos.grid.gendysanos.run_arb_tracing import _select_stage
from quantdata.dysanos.grid.gendysanos.run_conditional_arb_replay import (
    _conditional_delta_pnl,
    _deterministic_terminal_spot_lower_bound,
    _portfolio_pnl_band,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def _config() -> ArbTracingConfig:
    return ArbTracingConfig(
        label="test-arb-tracing",
        source_cache_uid="test-cache",
        state_features=2,
        trading_times=2,
        expiries=1,
        instruments_per_expiry=3,
        risk_aversion=2.0,
        utility="powerexp",
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=5,
            activation="SiLU",
            use_bias=True,
        ),
    )


def _model() -> OrthogonalArbDirection:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    return OrthogonalArbDirection(
        env,
        config=_config(),
        state_mean=torch.zeros((2, 2), dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype).expand(2, -1, -1).clone(),
    )


def test_physical_whitener_orthogonalizes_each_trading_time() -> None:
    rng = np.random.default_rng(3107)
    normal = rng.normal(size=(20_000, 2, 2))
    loading = np.asarray(
        [
            [[2.0, 0.0], [1.2, 0.5]],
            [[0.4, 0.0], [-0.7, 1.5]],
        ]
    )
    mean = np.asarray([[3.0, -1.0], [-2.0, 4.0]])
    state = np.einsum("stf,tgf->stg", normal, loading) + mean[None, :, :]
    state_tensor = torch.tensor(state, dtype=torch.float64)
    weight = torch.ones(state.shape[0], dtype=torch.float64)
    indices = torch.arange(state.shape[0])
    fitted_mean, whitener, diagnostics = fit_physical_state_whitener(
        state_tensor,
        weight,
        indices,
    )
    factor = torch.einsum(
        "stf,tfg->stg", state_tensor - fitted_mean[None, :, :], whitener
    )
    assert torch.allclose(
        factor.mean(dim=0), torch.zeros((2, 2), dtype=torch.float64), atol=1.0e-12
    )
    covariance = torch.einsum("stf,stg->tfg", factor, factor) / factor.shape[0]
    assert torch.allclose(
        covariance,
        torch.eye(2, dtype=torch.float64).expand(2, -1, -1),
        atol=1.0e-10,
    )
    assert diagnostics["maximum_absolute_weighted_factor_covariance_error"] < 1.0e-10


def test_direction_uses_project_feedforward_and_keeps_times_independent() -> None:
    model = _model()
    assert isinstance(model, BaseModel)
    assert all(isinstance(network, FeedForward) for network in model.networks)
    assert all(network.env is model.env for network in model.networks)
    state = torch.randn(7, 2, 2, dtype=model.fdtype)
    direction, raw_l1 = model(state)
    assert direction.shape == (7, 2, 1, 3)
    assert raw_l1.shape == (7, 2)
    assert torch.equal(direction, torch.zeros_like(direction))

    direction[:, 0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.networks[0].parameters())
    assert all(
        parameter.grad is None or torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        for parameter in model.networks[1].parameters()
    )


def test_direction_map_has_no_flat_exterior_and_preserves_l1_budget() -> None:
    model = _model()
    with torch.no_grad():
        for network in model.networks:
            output = network.In[-1] if network.In else network.I0
            output.weight.zero_()
            assert output.bias is not None
            output.bias.copy_(torch.tensor([20.0, 10.0, 5.0], dtype=model.fdtype))
    direction, raw_l1 = model(torch.zeros(4, 2, 2, dtype=model.fdtype))
    direction_l1 = direction.abs().sum(dim=(-1, -2))
    assert torch.all(direction_l1 < 1.0)
    assert torch.all(direction_l1 > 0.97)
    assert torch.allclose(raw_l1, torch.full_like(raw_l1, 35.0))

    direction.sum().backward()
    for network in model.networks:
        output = network.In[-1] if network.In else network.I0
        assert output.bias is not None and output.bias.grad is not None
        assert torch.linalg.vector_norm(output.bias.grad) > 0.0
        assert torch.isfinite(output.bias.grad).all()


def test_combined_position_cost_and_right_derivative_are_exact_at_sign_crossing() -> None:
    base = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    direction = torch.tensor([[[[-2.0]]]], dtype=torch.float64)
    returns = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    costs = torch.tensor([[[[0.1]]]], dtype=torch.float64)

    action = base + 0.75 * direction
    assert tracing_pnl(action, returns, costs).item() == pytest.approx(-1.05)
    derivative = tracing_pnl_right_derivative(
        base, direction, 0.75, returns, costs
    )
    assert derivative.item() == pytest.approx(-4.2)
    kink_right_derivative = tracing_pnl_right_derivative(
        base, direction, 0.5, returns, costs
    )
    assert kink_right_derivative.item() == pytest.approx(-4.2)


def test_tracing_band_uses_all_expiries_and_instruments() -> None:
    action = torch.tensor([[[[1.0, -2.0], [0.5, -0.25]]]], dtype=torch.float64)
    returns = torch.tensor([[[[0.2, -0.3], [0.4, 0.1]]]], dtype=torch.float64)
    costs = torch.full_like(action, 0.01)
    instrument_band = torch.full_like(action, 1.0e-7)
    result = tracing_numerical_band(action, returns, costs, instrument_band)
    absolute = action.abs()
    expected = (absolute * instrument_band).sum() + 32.0 * np.finfo(
        np.float32
    ).eps * (absolute * (returns.abs() + costs)).sum()
    assert result.item() == pytest.approx(float(expected))


def test_conditional_delta_replay_matches_canonical_case_derivation() -> None:
    terminal_spot = np.asarray([[0.85, 1.25], [1.10, 0.95]], dtype=np.float64)
    spot = 1.0
    option_price = np.asarray([[0.08, 0.03], [0.15, 0.07]])
    delta = np.asarray([[0.6, 0.3], [0.7, 0.4]])
    cash_strike = np.asarray([[0.9, 1.1], [0.8, 1.2]])
    action = np.asarray([[0.2, -0.3, 0.5], [-0.1, 0.4, -0.2]])
    cost_rate = 1.0e-4
    d_spot = terminal_spot - spot
    d_call = (
        np.maximum(terminal_spot[:, :, None] - cash_strike[None], 0.0)
        - option_price[None]
    )
    universe = {
        "dC": d_call[:, None].astype(np.float32),
        "C": np.broadcast_to(option_price, d_call.shape)[:, None].astype(np.float32),
        "delta": np.broadcast_to(delta, d_call.shape)[:, None].astype(np.float32),
        "spot": np.full((2, 1), spot, dtype=np.float32),
        "expiry_dS": d_spot[:, None].astype(np.float32),
        "log_strikes": np.broadcast_to(
            np.log(cash_strike / spot), d_call.shape
        )[:, None].astype(np.float32),
        "option_expired": np.ones((1, 1, 2), dtype=bool),
    }
    derived = derive_case_arrays(universe, hedge="delta", cost_rate=cost_rate)
    selection = np.asarray([[0, 2, 3], [1, 4, 5]])
    returns = np.take(derived.returns[:, 0], selection.reshape(-1), axis=1).reshape(
        2, 2, 3
    )
    costs = np.take(derived.costs[:, 0], selection.reshape(-1), axis=1).reshape(
        2, 2, 3
    )
    bands = np.take(
        derived.numerical_band[:, 0], selection.reshape(-1), axis=1
    ).reshape(2, 2, 3)
    expected_pnl, expected_band = _portfolio_pnl_band(
        action, returns, costs, bands
    )

    actual_pnl, actual_band = _conditional_delta_pnl(
        terminal_spot,
        action=action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        cost_rate=cost_rate,
    )

    assert actual_pnl == pytest.approx(expected_pnl, abs=2.0e-8)
    assert actual_band == pytest.approx(expected_band, rel=2.0e-6)


def test_deterministic_terminal_spot_bound_matches_piecewise_linear_grid() -> None:
    spot = 1.0
    option_price = np.asarray([[0.15, 0.04]])
    delta = np.asarray([[0.65, 0.25]])
    cash_strike = np.asarray([[0.9, 1.1]])
    action = np.asarray([[0.1, -0.4, 0.3]])
    result = _deterministic_terminal_spot_lower_bound(
        action=action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        cost_rate=1.0e-4,
    )
    terminal_spot = np.linspace(0.0, 4.0, 40_001)[:, None]
    pnl, _ = _conditional_delta_pnl(
        terminal_spot,
        action=action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        cost_rate=1.0e-4,
    )

    assert result["unbounded_below"] is False
    assert result["lower_bound"] == pytest.approx(float(np.min(pnl)), abs=1.0e-12)


@pytest.mark.parametrize("utility", ["powerexp", "entropy"])
def test_utility_objective_has_finite_correct_directional_gradient(utility: str) -> None:
    pnl_unit = torch.tensor(
        [[0.2], [-0.1], [0.3], [-0.05]], dtype=torch.float64
    )
    weight = torch.tensor([1.0, 0.5, 1.2, 0.8], dtype=torch.float64)
    notional = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    objective = tracing_utility_objective(
        notional * pnl_unit,
        weight,
        utility=utility,
        risk_aversion=2.0,
    ).sum()
    objective.backward()
    assert notional.grad is not None and torch.isfinite(notional.grad)
    step = 1.0e-6
    upper = tracing_utility_objective(
        (0.7 + step) * pnl_unit,
        weight,
        utility=utility,
        risk_aversion=2.0,
    ).item()
    lower = tracing_utility_objective(
        (0.7 - step) * pnl_unit,
        weight,
        utility=utility,
        risk_aversion=2.0,
    ).item()
    assert notional.grad.item() == pytest.approx((upper - lower) / (2.0 * step), rel=1.0e-7)


@pytest.mark.parametrize("utility", ["powerexp", "entropy"])
def test_scale_statistics_marginal_matches_objective_finite_difference(
    utility: str,
) -> None:
    unit = np.asarray([0.2, -0.1, 0.3, -0.05], dtype=np.float64)
    weight = np.asarray([1.0, 0.5, 1.2, 0.8], dtype=np.float64)
    component = np.asarray([0, 0, 1, 1])
    notional = 0.7
    report = traced_scale_statistics(
        (notional * unit)[:, None],
        np.zeros((unit.size, 1)),
        unit[:, None],
        weight,
        component,
        utility=utility,
        risk_aversion=2.0,
        loss_probability_epsilon=1.0e-6,
    )[0]
    step = 1.0e-6

    def objective(scale: float) -> float:
        value = torch.tensor((scale * unit)[:, None], dtype=torch.float64)
        return tracing_utility_objective(
            value,
            torch.tensor(weight, dtype=torch.float64),
            utility=utility,
            risk_aversion=2.0,
        ).item()

    finite_difference = (objective(notional + step) - objective(notional - step)) / (
        2.0 * step
    )
    assert report["objective_derivative"] == pytest.approx(
        finite_difference, rel=1.0e-7
    )


def test_statistics_separate_near_arbitrage_from_a_good_risky_trade() -> None:
    rng = np.random.default_rng(811)
    paths = 20_000
    weight = np.ones(paths)
    component = np.zeros(paths, dtype=np.int16)
    band = np.full((paths, 1), 1.0e-12)

    arbitrage_pnl = rng.uniform(0.01, 0.03, size=(paths, 1))
    arbitrage = traced_scale_statistics(
        arbitrage_pnl,
        band,
        arbitrage_pnl,
        weight,
        component,
        utility="powerexp",
        risk_aversion=10.0,
        loss_probability_epsilon=1.0 / paths,
    )[0]
    assert arbitrage["near_nonnegative_within_mc_noise"] is True
    assert arbitrage["statistically_positive_mean"] is True
    assert arbitrage["P_loss_probability"] == 0.0

    risky_pnl = rng.normal(0.02, 0.05, size=(paths, 1))
    risky = traced_scale_statistics(
        risky_pnl,
        band,
        risky_pnl,
        weight,
        component,
        utility="powerexp",
        risk_aversion=10.0,
        loss_probability_epsilon=1.0 / paths,
    )[0]
    assert risky["near_nonnegative_within_mc_noise"] is False
    assert risky["statistically_positive_mean"] is True
    assert risky["P_loss_probability"] > 0.30


def test_residual_continuation_is_selected_on_training_not_validation() -> None:
    def item(
        objective: float,
        derivative: float,
        *,
        positive: bool,
        near_nonnegative: bool,
    ) -> dict[str, float | bool]:
        return {
            "objective": objective,
            "normalized_marginal_derivative": derivative,
            "normalized_marginal_derivative_standard_error": 0.01,
            "marginal_density_effective_sample_size": 100.0,
            "near_nonnegative_within_mc_noise": near_nonnegative,
            "statistically_positive_mean": positive,
        }

    training = {
        "trace": [
            {"notional": 0.0, "times": [item(0.0, 1.0, positive=False, near_nonnegative=False)]},
            {"notional": 2.0, "times": [item(1.0, 0.1, positive=True, near_nonnegative=False)]},
            {"notional": 4.0, "times": [item(0.5, -0.2, positive=True, near_nonnegative=False)]},
        ]
    }
    validation = {
        "trace": [
            {"notional": 0.0, "times": [item(0.0, 1.0, positive=False, near_nonnegative=False)]},
            {"notional": 2.0, "times": [item(-1.0, -0.2, positive=True, near_nonnegative=False)]},
            {"notional": 4.0, "times": [item(-2.0, -0.3, positive=True, near_nonnegative=False)]},
        ]
    }
    selection, base_scale, next_active = _select_stage(
        training,
        validation,
        notionals=[0.0, 2.0, 4.0],
        active_time=[True],
        confidence_z=3.0,
        minimum_marginal_ess=20.0,
        probe_notional=2.0,
    )
    assert selection[0]["classification"] == "training-finite-risk-trade-not-validated"
    assert base_scale == [2.0]
    assert next_active == [True]
