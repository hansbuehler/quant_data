from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from cdxcore.bs import bs

from quantdata.sdlv.tangent_levy.config import (
    DynamicsConfig,
    MarketConfig,
    PricingConfig,
    SimulationConfig,
    StaticFitConfig,
)
from quantdata.sdlv.tangent_levy.constraints import (
    constraint_residuals,
    hjm_cell_drift,
    hjm_characteristic_residual,
    hjm_drift,
    hjm_integrated_characteristic_residual,
    project_factor_constraints,
)
from quantdata.sdlv.tangent_levy.double_exponential import (
    DoubleExponentialParameters,
    call_prices,
    characteristic_function,
    constrained_parameters,
    density,
    symmetry_index,
)
from quantdata.sdlv.tangent_levy.dynamics import (
    DynamicsFit,
    fit_dynamics,
    fit_reduced_dynamics,
)
from quantdata.sdlv.tangent_levy.experiment import validate_reconstruction_gate
from quantdata.sdlv.tangent_levy.markets import (
    MarketAudit,
    common_grid_coverage,
    grid_market_days_from_panel,
)
from quantdata.sdlv.tangent_levy.metrics import (
    energy_score,
    implied_volatility_error,
    interval_error_metrics,
    no_arbitrage_metrics,
    paired_block_bootstrap_interval,
    price_error_metrics,
    statistical_fidelity_metrics,
    variogram_score,
)
from quantdata.sdlv.tangent_levy.pricing import density_call_prices
from quantdata.sdlv.tangent_levy.simulator import simulate_codebooks
from quantdata.sdlv.tangent_levy.static_fit import (
    fit_maturity,
    reconstruct_forward_density,
    reconstruct_forward_density_positive_spline,
)


def _pricing_config() -> PricingConfig:
    return PricingConfig(integration_upper=140.0, integration_points=1_121)


def _parameters() -> DoubleExponentialParameters:
    positive_rate = 9.0
    probability = 0.55
    return DoubleExponentialParameters(
        intensity=12.0,
        positive_rate=positive_rate,
        negative_rate=probability * positive_rate / (1.0 - probability),
        positive_probability=probability,
    )


def test_configs_enforce_market_and_grid_contracts() -> None:
    market = MarketConfig()
    assert market.source == "dysanos_grid_cache"
    assert market.smoothness == 0.25
    assert market.grid_num_strikes == 100
    assert market.grid_cache_start == date(2020, 1, 1)
    assert market.grid_cache_end == date(2025, 8, 30)
    assert market.min_tau == pytest.approx(2.0 / 255.0)
    assert market.max_tau == 1.0
    with pytest.raises(ValueError, match="strictly greater than 0.01"):
        MarketConfig(smoothness=0.01)
    with pytest.raises(ValueError, match="start at zero"):
        DynamicsConfig(tau_grid=(0.01, 0.1, 0.2))
    with pytest.raises(ValueError, match="odd"):
        DynamicsConfig(jump_x_grid=(-0.2, -0.1, 0.0, 0.1))
    with pytest.raises(ValueError, match="must be boolean"):
        DynamicsConfig(require_reconstruction_pass=1)


def test_grid_panel_adapter_preserves_targets_weights_and_missing_intervals() -> None:
    config = MarketConfig(grid_num_strikes=5)
    expiries = np.asarray([[2.0 / 255.0, 1.0]])
    strikes = np.asarray(
        [
            [
                [0.4, 0.8, 0.9, 1.0, 1.1, 1.2, 1.8],
                [0.3, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0],
            ]
        ]
    )
    mids = np.asarray(
        [[[0.21, 0.13, 0.07, 0.03, 0.01], [0.36, 0.25, 0.16, 0.09, 0.04]]]
    )
    weights = np.asarray([[[2.0, 3.0, 4.0, 3.0, 0.0], [1.0] * 5]])
    panel = {
        "dates": np.asarray(["2024-03-05"], dtype="datetime64[D]"),
        "expiries": expiries,
        "xstrikes": strikes,
        "mids": mids,
        "weights": weights,
        "spot_ref": np.asarray([5_000.0]),
        "spot_ret": np.asarray([0.01]),
    }
    markets = grid_market_days_from_panel(
        panel,
        ticker="SPX",
        start=date(2024, 3, 5),
        end=date(2024, 3, 5),
        config=config,
    )
    market = markets[date(2024, 3, 5)]
    assert market.source == "dysanos_grid_cache"
    assert market.pricer is None
    np.testing.assert_array_equal(market.strikes[0], strikes[0, 0, 1:-1])
    np.testing.assert_array_equal(market.sanos_calls[1], mids[0, 1])
    np.testing.assert_array_equal(market.fit_weights[0], weights[0, 0])
    np.testing.assert_array_equal(
        market.model_masks[0],
        np.asarray([True, True, True, True, False]),
    )
    assert np.all(np.isnan(market.pure_bids[0]))
    assert not np.any(market.call_active[0])


def test_common_grid_coverage_keeps_missing_sessions_in_denominator() -> None:
    audits = [
        MarketAudit(
            market_date=date(2025, 1, 2),
            early_close=False,
            available=True,
            input_expiries=10,
            retained_expiries=8,
            retained_min_tau=0.02,
            retained_max_tau=0.80,
            input_quotes=1_000,
            active_quotes=900,
            excluded_american_quotes=0,
            option_cap_bound=True,
        ),
        MarketAudit(
            market_date=date(2025, 1, 3),
            early_close=False,
            available=True,
            input_expiries=9,
            retained_expiries=7,
            retained_min_tau=0.02,
            retained_max_tau=0.60,
            input_quotes=800,
            active_quotes=700,
            excluded_american_quotes=0,
            option_cap_bound=False,
        ),
        MarketAudit(
            market_date=date(2025, 1, 6),
            early_close=False,
            available=False,
            input_expiries=0,
            retained_expiries=0,
            retained_min_tau=np.nan,
            retained_max_tau=np.nan,
            input_quotes=0,
            active_quotes=0,
            excluded_american_quotes=0,
            option_cap_bound=False,
            reason="missing",
        ),
    ]
    report = common_grid_coverage(
        audits,
        candidate_tau=np.asarray([0.0, 0.1, 0.5, 0.75]),
        minimum_fraction=2.0 / 3.0,
    )
    np.testing.assert_allclose(
        report.coverage_fraction,
        [1.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
    )
    np.testing.assert_allclose(report.selected_tau, [0.0, 0.1, 0.5])
    assert report.available_sessions == 2
    assert report.option_cap_fraction == pytest.approx(1.0 / 3.0)


def test_double_exponential_constraints_and_martingale_identity() -> None:
    parameters = _parameters()
    left_density = (
        parameters.intensity
        * (1.0 - parameters.positive_probability)
        * parameters.negative_rate
    )
    right_density = (
        parameters.intensity
        * parameters.positive_probability
        * parameters.positive_rate
    )
    assert left_density == pytest.approx(right_density)

    target_xi = symmetry_index(parameters)
    reconstructed = constrained_parameters(
        intensity=parameters.intensity,
        positive_rate=parameters.positive_rate,
        target_symmetry_index=target_xi,
    )
    assert reconstructed.negative_rate == pytest.approx(parameters.negative_rate)
    assert reconstructed.positive_probability == pytest.approx(
        parameters.positive_probability
    )
    assert symmetry_index(reconstructed) == pytest.approx(target_xi)
    martingale = characteristic_function(
        np.asarray([-1j]),
        tau=0.4,
        parameters=parameters,
    )
    np.testing.assert_allclose(martingale, 1.0, atol=1.0e-13)


def test_double_exponential_fourier_price_matches_monte_carlo() -> None:
    parameters = _parameters()
    tau = 0.25
    log_strikes = np.asarray([-0.1, 0.0, 0.1])
    expected = call_prices(
        tau=tau,
        log_strikes=log_strikes,
        parameters=parameters,
        config=_pricing_config(),
    )
    rng = np.random.default_rng(202_508_30)
    paths = 300_000
    counts = rng.poisson(parameters.intensity * tau, size=paths)
    path_index = np.repeat(np.arange(paths), counts)
    positive = rng.random(path_index.size) < parameters.positive_probability
    jumps = np.empty(path_index.size)
    jumps[positive] = rng.exponential(
        1.0 / parameters.positive_rate,
        size=np.count_nonzero(positive),
    )
    jumps[~positive] = -rng.exponential(
        1.0 / parameters.negative_rate,
        size=np.count_nonzero(~positive),
    )
    jump_sum = np.bincount(path_index, weights=jumps, minlength=paths)
    log_return = (
        jump_sum - parameters.intensity * tau * parameters.exponential_jump_mean
    )
    terminal = np.exp(log_return)
    actual = np.mean(
        np.maximum(terminal[:, None] - np.exp(log_strikes)[None, :], 0.0),
        axis=0,
    )
    np.testing.assert_allclose(actual, expected, atol=1.5e-3, rtol=0.0)


def test_generic_density_pricer_matches_double_exponential() -> None:
    parameters = _parameters()
    jump_x = np.linspace(-2.0, 2.0, 4_001)
    eta = density(jump_x, parameters)
    log_strikes = np.asarray([-0.1, 0.0, 0.1])
    expected = call_prices(
        tau=0.25,
        log_strikes=log_strikes,
        parameters=parameters,
        config=_pricing_config(),
    )
    actual = density_call_prices(
        tau=0.25,
        log_strikes=log_strikes,
        jump_x=jump_x,
        average_density=eta,
        config=_pricing_config(),
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-4)


def test_static_fit_recovers_synthetic_parameters() -> None:
    parameters = _parameters()
    tau = 0.25
    strikes = np.exp(np.linspace(-0.2, 0.15, 15))
    calls = call_prices(
        tau=tau,
        log_strikes=np.log(strikes),
        parameters=parameters,
        config=_pricing_config(),
    )
    config = StaticFitConfig(
        pricing=_pricing_config(),
        num_starts=2,
        max_iterations=100,
    )
    fit = fit_maturity(
        tau=tau,
        strikes=strikes,
        target_calls=calls,
        bids=calls - 0.001,
        asks=calls + 0.001,
        mask=np.ones(strikes.size, dtype=bool),
        config=config,
    )
    assert fit.success
    assert fit.parameters.intensity == pytest.approx(parameters.intensity, rel=2.0e-4)
    assert fit.parameters.positive_rate == pytest.approx(
        parameters.positive_rate, rel=2.0e-4
    )
    assert fit.parameters.negative_rate == pytest.approx(
        parameters.negative_rate, rel=2.0e-4
    )
    assert fit.parameters.positive_probability == pytest.approx(
        parameters.positive_probability, rel=2.0e-4
    )

    constrained = fit_maturity(
        tau=tau,
        strikes=strikes,
        target_calls=calls,
        bids=calls - 0.001,
        asks=calls + 0.001,
        mask=np.ones(strikes.size, dtype=bool),
        config=config,
        target_symmetry_index=symmetry_index(parameters),
    )
    assert constrained.success
    assert symmetry_index(constrained.parameters) == pytest.approx(
        symmetry_index(parameters), rel=1.0e-12
    )

    cached_grid = fit_maturity(
        tau=tau,
        strikes=strikes,
        target_calls=calls,
        bids=np.full_like(calls, np.nan),
        asks=np.full_like(calls, np.nan),
        mask=np.ones(strikes.size, dtype=bool),
        config=config,
        objective_weights=np.full_like(calls, 100.0),
    )
    assert cached_grid.success
    assert cached_grid.parameters.intensity == pytest.approx(
        parameters.intensity, rel=2.0e-4
    )


def test_forward_density_reconstruction_preserves_constant_eta() -> None:
    parameters = _parameters()
    fitted_tau = np.asarray([0.1, 0.25, 0.5, 1.0])
    target_tau = np.asarray([0.0, 0.1, 0.25, 0.5, 1.0])
    jump_x = np.linspace(-0.75, 0.75, 61)
    eta = np.broadcast_to(
        density(jump_x, parameters),
        (fitted_tau.size, jump_x.size),
    ).copy()
    result = reconstruct_forward_density(
        fitted_tau=fitted_tau,
        fitted_average_density=eta,
        target_tau=target_tau,
        jump_x=jump_x,
        relative_tolerance=1.0e-12,
    )
    assert result.passed_tolerance
    assert result.maximum_relative_projection == 0.0
    assert result.relative_l1_projection == 0.0
    np.testing.assert_allclose(
        result.forward_density,
        np.broadcast_to(eta[0], result.forward_density.shape),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(result.average_density[1:], eta, rtol=1.0e-12)
    np.testing.assert_allclose(
        result.projected_fitted_average_density,
        eta,
        rtol=1.0e-12,
    )


def test_forward_density_reconstruction_honors_positive_floor_in_tails() -> None:
    fitted_tau = np.asarray([0.1, 0.25, 0.5])
    target_tau = np.asarray([0.0, 0.1, 0.25, 0.5])
    jump_x = np.asarray([-0.5, 0.0, 0.5])
    eta = np.asarray(
        [
            [1.0e-9, 2.0, 1.0e-9],
            [1.0e-9, 1.5, 1.0e-9],
            [1.0e-9, 1.0, 1.0e-9],
        ]
    )
    result = reconstruct_forward_density(
        fitted_tau=fitted_tau,
        fitted_average_density=eta,
        target_tau=target_tau,
        jump_x=jump_x,
        forward_density_floor=1.0e-6,
    )
    assert np.min(result.forward_density) >= 1.0e-6
    assert np.min(result.projected_fitted_average_density) >= 1.0e-6


def test_reduced_node_dynamics_preserve_full_grid_ra3() -> None:
    generator = np.random.default_rng(91)
    tau = np.arange(7, dtype=np.float64) / 252.0
    jump_x = np.linspace(-0.2, 0.2, 41)
    shape = 2.0 + 8.0 * np.exp(-20.0 * np.abs(jump_x))[None, None]
    history = np.broadcast_to(shape, (80, len(tau), len(jump_x))).copy()
    history += 0.01 * np.cumsum(
        generator.standard_normal(history.shape), axis=0
    )
    history -= np.min(history) - 0.1
    config = DynamicsConfig(
        tau_grid=tuple(tau),
        jump_x_grid=tuple(jump_x),
        num_factors=3,
        minimum_observations=20,
    )

    fitted = fit_reduced_dynamics(
        history,
        config=config,
        maturity_stride=2,
        jump_node_count=7,
    )

    assert fitted.num_factors == 3
    assert fitted.maturity_node_indices is not None
    assert fitted.jump_node_indices is not None
    assert len(fitted.jump_node_indices) == 7
    assert len(fitted.maturity_node_indices) == 4
    assert fitted.maximum_constraint_residual < 1.0e-12
    assert not np.any(fitted.beta[..., len(jump_x) // 2])


def test_positive_spline_reconstruction_recovers_smooth_forward_density() -> None:
    fitted_tau = np.linspace(0.04, 0.5, 12)
    target_tau = np.linspace(0.0, 0.5, 31)
    jump_x = np.asarray((-0.1, 0.1))
    forward = np.stack(
        (
            2.0 + 0.5 * target_tau + 0.2 * np.square(target_tau),
            1.0 + 0.3 * target_tau + 0.1 * np.square(target_tau),
        ),
        axis=1,
    )
    # Analytic averages of the two quadratic forward-density curves.
    eta = np.stack(
        (
            2.0 + 0.25 * fitted_tau + (0.2 / 3.0) * np.square(fitted_tau),
            1.0 + 0.15 * fitted_tau + (0.1 / 3.0) * np.square(fitted_tau),
        ),
        axis=1,
    )
    result = reconstruct_forward_density_positive_spline(
        fitted_tau=fitted_tau,
        fitted_average_density=eta,
        target_tau=target_tau,
        jump_x=jump_x,
        basis_count=8,
        curvature_penalty=1.0e-8,
        relative_tolerance=1.0e-5,
    )

    assert result.passed_tolerance
    assert np.min(result.forward_density) >= 0.0
    np.testing.assert_allclose(result.forward_density, forward, rtol=2.0e-4)
    np.testing.assert_allclose(
        result.projected_fitted_average_density,
        eta,
        rtol=2.0e-5,
    )


def test_reconstruction_gate_checks_training_and_simulation_origin() -> None:
    dates = tuple(date(2025, 1, day) for day in (2, 3, 6, 7))
    history = SimpleNamespace(
        dates=dates,
        reconstruction_passed=np.asarray([True, True, True, False]),
        reconstruction_l1_error=np.asarray([0.0, 0.0, 0.0, 0.5]),
        reconstruction_repricing_rmse=np.asarray([0.0, 0.0, 0.0, 7.0]),
    )
    validate_reconstruction_gate(
        history,
        dynamics_position=3,
        include_simulation_origin=False,
    )
    with pytest.raises(RuntimeError, match="2025-01-07"):
        validate_reconstruction_gate(
            history,
            dynamics_position=3,
            include_simulation_origin=True,
        )


def test_constraint_projection_and_dynamics_annualization() -> None:
    tau = (0.0, 0.1, 0.25)
    jump_x = tuple(np.linspace(-0.2, 0.2, 9))
    config = DynamicsConfig(
        tau_grid=tau,
        jump_x_grid=jump_x,
        num_factors=2,
        minimum_observations=2,
        covariance_shrinkage=0.0,
    )
    rng = np.random.default_rng(11)
    changes = project_factor_constraints(
        rng.normal(scale=0.01, size=(30, len(tau), len(jump_x))),
        jump_x=np.asarray(jump_x),
    )
    history = 1.0 + np.concatenate(
        [np.zeros_like(changes[:1]), np.cumsum(changes, axis=0)],
        axis=0,
    )
    fit = fit_dynamics(history, config=config)
    residual = constraint_residuals(fit.beta, jump_x=np.asarray(jump_x))
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-17)
    assert fit.maximum_constraint_residual < 2.0e-17
    reconstructed_eigenvalues = np.linalg.eigvalsh(
        fit.reconstructed_one_session_covariance()
    )[-fit.num_factors :]
    np.testing.assert_allclose(
        reconstructed_eigenvalues,
        fit.eigenvalues_one_session[::-1],
        rtol=1.0e-11,
        atol=1.0e-14,
    )


def test_hjm_zero_factors_and_simulator_replay() -> None:
    tau = np.asarray([0.0, 0.1, 0.25])
    jump_x = np.linspace(-0.2, 0.2, 9)
    beta = np.zeros((1, tau.size, jump_x.size))
    np.testing.assert_array_equal(hjm_drift(tau=tau, jump_x=jump_x, beta=beta), 0.0)
    dynamics = DynamicsFit(
        tau=tau,
        jump_x=jump_x,
        beta=beta,
        eigenvalues_one_session=np.zeros(1),
        explained_variance_ratio=np.zeros(1),
        mean_change=np.zeros((tau.size, jump_x.size)),
        observations=2,
        maximum_constraint_residual=0.0,
        covariance_shrinkage=0.0,
        dt=1.0 / 252.0,
    )
    initial = np.ones((tau.size, jump_x.size))
    config = SimulationConfig(num_paths=8, num_steps=2, record_steps=(1, 2), seed=17)
    first = simulate_codebooks(initial, dynamics, config=config)
    second = simulate_codebooks(initial, dynamics, config=config)
    np.testing.assert_array_equal(first.forward_density, second.forward_density)
    np.testing.assert_array_equal(
        first.forward_density,
        np.broadcast_to(initial, first.forward_density.shape),
    )
    assert first.failed_path_fraction == 0.0


def test_hjm_drift_preserves_full_convolution_support_and_generator() -> None:
    tau = np.asarray([0.0, 0.1, 0.25, 0.5])
    jump_x = np.linspace(-0.75, 0.75, 61)
    active = np.broadcast_to(
        (np.abs(jump_x) <= 0.2)[None],
        (tau.size, jump_x.size),
    ).copy()
    beta = project_factor_constraints(
        np.random.default_rng(7).normal(size=(3, tau.size, jump_x.size)),
        jump_x=jump_x,
        active_mask=active,
    )
    alpha = hjm_drift(
        tau=tau,
        jump_x=jump_x,
        beta=beta,
        active_mask=active,
    )
    frequencies = np.linspace(-20.0, 20.0, 81)
    residual = hjm_characteristic_residual(
        tau=tau,
        jump_x=jump_x,
        beta=beta,
        alpha=alpha,
        frequencies=frequencies,
    )

    assert np.any(np.abs(alpha[~active]) > 1.0e-10)
    np.testing.assert_allclose(
        constraint_residuals(alpha, jump_x=jump_x),
        0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-13)

    masked_alpha = np.where(active, alpha, 0.0)
    masked_residual = hjm_characteristic_residual(
        tau=tau,
        jump_x=jump_x,
        beta=beta,
        alpha=masked_alpha,
        frequencies=frequencies,
    )
    assert np.max(np.abs(masked_residual)) > 1.0e-4


def test_cell_hjm_drift_satisfies_integrated_generator_identity() -> None:
    jump_x = np.linspace(-0.75, 0.75, 61)
    support = np.abs(jump_x) <= 0.2
    active = np.broadcast_to(support[None], (8, jump_x.size)).copy()
    beta_cells = project_factor_constraints(
        np.random.default_rng(19).normal(scale=0.01, size=(2, 8, jump_x.size)),
        jump_x=jump_x,
        active_mask=active,
    )

    alpha_cells = hjm_cell_drift(beta_cells, jump_x=jump_x)
    residual = hjm_integrated_characteristic_residual(
        beta_cells=beta_cells,
        alpha_cells=alpha_cells,
        jump_x=jump_x,
        frequencies=np.linspace(-20.0, 20.0, 81),
    )

    np.testing.assert_allclose(residual, 0.0, atol=2.0e-15)
    assert np.any(np.abs(alpha_cells[:, ~support]) > 1.0e-10)


def test_static_and_probabilistic_metrics() -> None:
    target = np.asarray([0.2, 0.1, 0.04])
    model = target + np.asarray([0.0, 0.01, -0.02])
    price = price_error_metrics(
        model,
        target,
        half_spread=np.full(3, 0.01),
    )
    assert price.observations == 3
    assert price.rmse == pytest.approx(np.sqrt(5.0e-4 / 3.0))
    weighted = price_error_metrics(
        model,
        target,
        residual_multiplier=np.asarray([1.0, 2.0, 3.0]),
    )
    assert weighted.weighted_rmse == pytest.approx(
        np.sqrt((0.0**2 + 0.02**2 + 0.06**2) / 3.0)
    )
    interval = interval_error_metrics(
        model,
        target - 0.005,
        target + 0.005,
    )
    assert interval.fraction_inside == pytest.approx(1.0 / 3.0)
    assert interval.fraction_outside == pytest.approx(2.0 / 3.0)

    tau = np.asarray([0.1, 0.2])
    strikes = np.asarray([0.9, 1.0, 1.1])
    calls = bs.price(
        k=strikes[None],
        vol=0.25,
        sqrtT=np.sqrt(tau)[:, None],
    )
    model_calls = calls.copy()
    model_calls[0, 0] = 1.0 - strikes[0] - 1.0e-6
    iv_error = implied_volatility_error(
        tau=np.broadcast_to(tau[:, None], calls.shape),
        strikes=np.broadcast_to(strikes[None], calls.shape),
        model_calls=model_calls,
        target_calls=calls,
    )
    assert iv_error.observations == calls.size - 1
    assert no_arbitrage_metrics(calls, tau=tau, strikes=strikes).maximum < 1.0e-12

    ensemble = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    observation = np.asarray([0.5, 0.25])
    assert energy_score(ensemble, observation) >= 0.0
    score = variogram_score(
        ensemble,
        observation,
        pairs=np.asarray([[0, 1]]),
    )
    assert score >= 0.0
    mean, lower, upper = paired_block_bootstrap_interval(
        np.linspace(-0.2, -0.1, 40),
        block_length=10,
        samples=100,
        seed=3,
    )
    assert lower <= mean <= upper < 0.0

    changes = np.random.default_rng(19).normal(size=(100, 4))
    fidelity = statistical_fidelity_metrics(changes, changes, num_factors=3)
    assert fidelity.mean_change_error_norm == 0.0
    assert fidelity.relative_covariance_error == 0.0
    assert fidelity.correlation_error == 0.0
    assert fidelity.median_variance_ratio == 1.0
    assert fidelity.eigenvalue_spectrum_error == 0.0
    assert fidelity.maximum_principal_angle_degrees < 2.0e-6
