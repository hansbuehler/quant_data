"""Tests for the simple two-period OU/OCE experiment."""

from __future__ import annotations

import numpy as np
import pytest

from quantdata.oce.simple_ou_oce import (
    MarketConfig,
    atm_black_scholes_call,
    black_scholes_call,
    build_trading_design,
    simulate_market,
    solve_exponential_oce,
    solve_exponential_oce_adam,
)


def test_atm_black_scholes_call_matches_known_formula() -> None:
    price = atm_black_scholes_call(
        volatility=np.asarray([0.2]),
        time_to_expiry=1.0,
    )
    assert price[0] == pytest.approx(0.07965567455405798)


def test_general_black_scholes_call_respects_intrinsic_bounds() -> None:
    strikes = np.asarray([0.95, 1.0, 1.05])
    prices = black_scholes_call(
        strike=strikes,
        volatility=np.full(3, 0.2),
        time_to_expiry=0.25,
    )
    assert np.all(prices >= np.maximum(1.0 - strikes, 0.0))
    assert np.all(prices <= 1.0)


def test_simulation_shapes_and_exact_ou_second_moment() -> None:
    config = MarketConfig(kappa=3.0, xi=0.7, paths=100_000, seed=17)
    paths = simulate_market(config)

    expected_variance = config.xi**2 * (
        1.0 - np.exp(-2.0 * config.kappa * config.step_years)
    ) / (2.0 * config.kappa)
    assert paths.v.shape == (config.paths, 3)
    assert paths.stock_return.shape == (config.paths, 2)
    assert paths.option_return.shape == (config.paths, 2, 5)
    assert np.var(paths.v[:, 1]) == pytest.approx(expected_variance, rel=0.015)
    assert np.mean(paths.stock_return[:, 0]) == pytest.approx(0.0, abs=2.0e-4)
    assert np.all((paths.implied_volatility > 0.0) & (paths.implied_volatility < 1.0))


def test_trading_design_uses_day10_state_dependent_positions() -> None:
    paths = simulate_market(MarketConfig(paths=128, seed=11))
    design, names = build_trading_design(paths)

    assert design.shape == (128, 12)
    assert len(names) == design.shape[1]
    np.testing.assert_allclose(design[:, 0], paths.stock_return[:, 1])
    np.testing.assert_allclose(design[:, 2], paths.option_return[:, 1, 0])
    state_scale = np.sqrt(np.mean(np.square(paths.v[:, 1])))
    normalized_v = paths.v[:, 1] / state_scale
    np.testing.assert_allclose(design[:, 1], normalized_v * paths.stock_return[:, 1])
    np.testing.assert_allclose(design[:, 3], normalized_v * paths.option_return[:, 1, 0])
    np.testing.assert_allclose(design[:, 10], paths.option_return[:, 1, 4])
    np.testing.assert_allclose(design[:, 11], normalized_v * paths.option_return[:, 1, 4])


def test_oce_density_enforces_the_traded_moment() -> None:
    design = np.asarray([[-1.0], [1.0]], dtype=np.float64)
    result = solve_exponential_oce(
        design,
        base_density=np.asarray([0.8, 0.2]),
        gradient_tolerance=1.0e-8,
    )

    np.testing.assert_allclose(result.density, [0.5, 0.5], atol=1.0e-10)
    assert result.gradient_maximum_absolute < 1.0e-8
    assert result.converged


def test_adam_oce_density_enforces_the_traded_moment() -> None:
    design = np.asarray([[-1.0], [1.0]], dtype=np.float64)
    result = solve_exponential_oce_adam(
        design,
        base_density=np.asarray([0.8, 0.2]),
        learning_rate=0.05,
        maximum_updates=5_000,
        gradient_tolerance=1.0e-7,
    )

    np.testing.assert_allclose(result.density, [0.5, 0.5], atol=1.0e-6)
    assert result.gradient_maximum_absolute < 1.0e-6
    assert result.converged


def test_adam_trace_reports_stationarity_and_curvature() -> None:
    paths = simulate_market(MarketConfig(paths=256, seed=29))
    design, _ = build_trading_design(paths)
    result = solve_exponential_oce_adam(
        design,
        maximum_updates=20,
        gradient_tolerance=1.0e-30,
        trace_interval=5,
    )

    assert [point["update"] for point in result.trace] == [0, 5, 10, 15, 20]
    final = result.trace[-1]
    assert final["raw_gradient_l2_norm"] > 0.0
    assert final["scaled_gradient_l2_norm"] > 0.0
    assert final["newton_decrement_squared"] >= 0.0
    assert final["effective_sample_size"] > 0.0
    assert final["hessian_condition_number"] >= 1.0


def test_small_market_oce_converges_and_fresh_problem_is_zero() -> None:
    paths = simulate_market(MarketConfig(paths=2_000, seed=23))
    design, _ = build_trading_design(paths)
    result = solve_exponential_oce(design, gradient_tolerance=1.0e-8)
    certificate = solve_exponential_oce(
        design,
        base_density=result.density,
        initial_coefficients=result.coefficients,
        gradient_tolerance=1.0e-8,
    )

    assert result.gradient_maximum_absolute < 2.0e-7
    assert certificate.oce == pytest.approx(0.0, abs=1.0e-10)
    assert np.linalg.norm(certificate.coefficients) < 1.0e-4
