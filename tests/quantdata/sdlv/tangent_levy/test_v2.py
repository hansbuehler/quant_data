"""Focused checks for the Tangent Levy v2 experiment contract."""

from __future__ import annotations

from datetime import date

import numpy as np

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.tangent_levy.config import DynamicsConfig
from quantdata.sdlv.tangent_levy.dynamics import DynamicsFit, fit_dynamics
from quantdata.sdlv.tangent_levy.pricing import (
    density_call_prices,
    density_characteristic_function,
    lattice_density_call_prices_batched,
)
from quantdata.sdlv.tangent_levy.run_v2_same_date import (
    _consecutive_session_mask,
)
from quantdata.sdlv.tangent_levy.v2_audit import unconditional_coordinate_audit
from quantdata.sdlv.tangent_levy.v2_config import (
    TangentLevyV2AuditConfig,
    TangentLevyV2DynamicsConfig,
    TangentLevyV2MarketConfig,
)
from quantdata.sdlv.tangent_levy.v2_population import (
    _cell_integrals,
    _predictable_boundary_scale,
    analytic_deterministic_q_audit,
    simulate_restricted_population,
)
from quantdata.sdlv.tangent_levy.v2_spot import (
    sample_short_end_spot_step,
    short_end_characteristics,
)


def test_v2_contract_matches_exact_population_and_restricted_universe() -> None:
    market = TangentLevyV2MarketConfig()
    dynamics = TangentLevyV2DynamicsConfig()
    audit = TangentLevyV2AuditConfig()

    assert market.expected_dates == 891
    assert market.expected_quotes == 3_566_258
    assert np.allclose(np.asarray(dynamics.maturity_grid) * 252.0, np.arange(71))
    assert audit.instrument_count == 45
    assert audit.path_count == 100_000


def test_same_date_consecutive_mask_has_one_entry_per_transition() -> None:
    dates = (
        date(2025, 8, 27),
        date(2025, 8, 28),
        date(2025, 9, 2),
    )

    mask = _consecutive_session_mask(dates)

    assert mask.shape == (2,)
    assert mask.tolist() == [True, False]


def test_short_end_compensator_has_unit_expected_gross_return() -> None:
    jump_x = np.linspace(-0.3, 0.3, 121)
    density = 8.0 * np.exp(-18.0 * np.abs(jump_x))
    intensity, compensator, probabilities = short_end_characteristics(
        density, jump_x=jump_x
    )
    dt = 1.0 / 252.0
    expected_jump_exponential = probabilities @ np.exp(jump_x)
    analytical = np.exp(
        -compensator * dt + intensity * dt * (expected_jump_exponential - 1.0)
    )

    assert np.isclose(analytical, 1.0, atol=2.0e-15)


def test_short_end_antithetic_step_preserves_path_geometry() -> None:
    jump_x = np.linspace(-0.2, 0.2, 101)
    density = np.broadcast_to(
        5.0 * np.exp(-20.0 * np.abs(jump_x)), (100, jump_x.size)
    )
    step = sample_short_end_spot_step(
        density,
        jump_x=jump_x,
        dt=1.0 / 252.0,
        rng=np.random.default_rng(17),
    )

    assert step.gross_return.shape == (100,)
    assert np.all(step.gross_return > 0.0)
    assert np.all(np.isfinite(step.log_return))


def test_short_end_spot_and_option_pricer_form_one_q_martingale() -> None:
    jump_x = np.linspace(-0.3, 0.3, 301)
    density = 10.0 * np.exp(-16.0 * np.abs(jump_x))
    path_count = 200_000
    step = sample_short_end_spot_step(
        np.broadcast_to(density, (path_count, jump_x.size)),
        jump_x=jump_x,
        dt=1.0 / 252.0,
        rng=np.random.default_rng(71),
    )
    strikes = np.asarray((0.98, 1.0, 1.02))
    simulated = np.maximum(step.gross_return[:, None] - strikes[None], 0.0)
    model = density_call_prices(
        tau=1.0 / 252.0,
        log_strikes=np.log(strikes),
        jump_x=jump_x,
        average_density=density,
    )
    pair_payoffs = simulated.reshape(-1, 2, len(strikes)).mean(axis=1)
    standard_error = np.std(pair_payoffs, axis=0, ddof=1) / np.sqrt(
        len(pair_payoffs)
    )

    assert np.all(np.abs(np.mean(pair_payoffs, axis=0) - model) < 5 * standard_error)


def test_batched_lattice_prices_match_scalar_prices() -> None:
    jump_x = np.linspace(-0.3, 0.3, 301)
    densities = np.stack(
        (
            8.0 * np.exp(-14.0 * np.abs(jump_x)),
            11.0 * np.exp(-18.0 * np.abs(jump_x)),
        )
    )
    log_strikes = np.log(
        np.asarray(((0.95, 1.0, 1.05), (0.97, 1.02, 1.08)))
    )
    batched = lattice_density_call_prices_batched(
        tau=10.0 / 252.0,
        log_strikes=log_strikes,
        jump_x=jump_x,
        average_density=densities,
    )
    scalar = np.stack(
        [
            density_call_prices(
                tau=10.0 / 252.0,
                log_strikes=log_strikes[index],
                jump_x=jump_x,
                average_density=densities[index],
            )
            for index in range(len(densities))
        ]
    )

    assert np.allclose(batched, scalar, rtol=1.0e-12, atol=1.0e-12)


def test_unconditional_audit_uses_pair_averages() -> None:
    generator = np.random.default_rng(31)
    independent = generator.standard_normal((500, 3, 2))
    noise = generator.standard_normal((500, 3, 2)) * 0.02
    paired = np.stack((independent + noise, -independent + noise), axis=1).reshape(
        1_000, 3, 2
    )
    centered = unconditional_coordinate_audit(paired)
    shifted = unconditional_coordinate_audit(paired + 0.1)

    assert centered.tested_coordinates == 6
    assert shifted.maximum_absolute_z > centered.maximum_absolute_z
    assert shifted.count_above_5 == 6


def test_analytic_reduced_q_audit_proves_all_coordinate_martingales() -> None:
    config = TangentLevyV2AuditConfig()
    tau = np.arange(71, dtype=np.float64) / 252.0
    jump_x = np.linspace(-0.75, 0.75, 301)
    maturity_shape = 1.0 + 0.4 * np.sin(np.arange(71)[:, None] / 9.0)
    jump_shape = 4.0 * np.exp(-12.0 * np.abs(jump_x))[None]
    density = maturity_shape * jump_shape

    dynamics = DynamicsFit(
        tau=tau,
        jump_x=jump_x,
        beta=np.zeros((0, len(tau), len(jump_x))),
        eigenvalues_one_session=np.zeros(0),
        explained_variance_ratio=np.zeros(0),
        mean_change=np.zeros_like(density),
        observations=0,
        maximum_constraint_residual=0.0,
        covariance_shrinkage=0.0,
        dt=1.0 / 252.0,
        active_mask=np.ones_like(density, dtype=bool),
    )
    result = analytic_deterministic_q_audit(density, dynamics, config=config)

    assert result.tested_coordinates == 570
    assert result.maturity_decompositions == 50
    assert result.maximum_relative_density_error < 1.0e-15
    assert result.maximum_spot_moment_error < 1.0e-15
    assert result.passed


def test_dynamics_can_screen_admissibility_at_initial_state() -> None:
    generator = np.random.default_rng(47)
    tau = np.asarray((0.0, 1.0 / 252.0, 2.0 / 252.0))
    jump_x = np.linspace(-0.2, 0.2, 21)
    history = np.full((64, len(tau), len(jump_x)), 1.0e-6)
    history += np.abs(
        generator.normal(scale=1.0e-7, size=history.shape).cumsum(axis=0)
    )
    reference = np.full(history.shape[1:], 1.0)
    config = DynamicsConfig(
        tau_grid=tuple(tau),
        jump_x_grid=tuple(jump_x),
        num_factors=2,
        minimum_observations=10,
        inactive_density_threshold=1.0e-4,
        boundary_standard_deviations=5.0,
    )

    fitted = fit_dynamics(
        history,
        config=config,
        reference_density=reference,
    )

    assert fitted.num_factors == 2
    assert fitted.active_mask is not None
    assert np.any(fitted.active_mask)
    assert fitted.maximum_constraint_residual < 1.0e-12


def test_measured_factor_population_uses_batched_pricing_path() -> None:
    tau = np.arange(71, dtype=np.float64) / 252.0
    jump_x = np.linspace(-0.75, 0.75, 301)
    density = np.broadcast_to(
        4.0 * np.exp(-12.0 * np.abs(jump_x)),
        (len(tau), len(jump_x)),
    ).copy()
    dynamics = DynamicsFit(
        tau=tau,
        jump_x=jump_x,
        beta=np.zeros((1, len(tau), len(jump_x))),
        eigenvalues_one_session=np.zeros(1),
        explained_variance_ratio=np.zeros(1),
        mean_change=np.zeros_like(density),
        observations=63,
        maximum_constraint_residual=0.0,
        covariance_shrinkage=0.05,
        dt=1.0 / 252.0,
        active_mask=np.ones_like(density, dtype=bool),
    )
    config = TangentLevyV2AuditConfig(
        path_count=4,
        horizon_days=2,
        option_expiry_days=(1, 2),
        steps_per_day=1,
    )

    population, universe, diagnostics = simulate_restricted_population(
        density,
        dynamics,
        config=config,
        path_count=4,
        batch_size=4,
    )

    assert population.returns.shape == (4, 2, 23)
    assert population.features.shape == (4, 2, 27)
    assert len(universe.labels) == 23
    assert diagnostics.frozen_paths == 0


def test_zero_loading_factor_uses_exact_deterministic_transport() -> None:
    tau = np.arange(4, dtype=np.float64) / 252.0
    jump_x = np.linspace(-0.2, 0.2, 81)
    maturity_shape = 1.0 + 0.1 * np.arange(len(tau))[:, None]
    density = maturity_shape * np.exp(-12.0 * np.abs(jump_x))[None]
    common = {
        "tau": tau,
        "jump_x": jump_x,
        "mean_change": np.zeros_like(density),
        "observations": 0,
        "maximum_constraint_residual": 0.0,
        "covariance_shrinkage": 0.0,
        "dt": 1.0 / 252.0,
        "active_mask": np.ones_like(density, dtype=bool),
    }
    deterministic = DynamicsFit(
        beta=np.zeros((0, len(tau), len(jump_x))),
        eigenvalues_one_session=np.zeros(0),
        explained_variance_ratio=np.zeros(0),
        **common,
    )
    zero_loading = DynamicsFit(
        beta=np.zeros((1, len(tau), len(jump_x))),
        eigenvalues_one_session=np.zeros(1),
        explained_variance_ratio=np.zeros(1),
        **common,
    )
    config = TangentLevyV2AuditConfig(
        path_count=100,
        horizon_days=2,
        option_expiry_days=(1, 2),
        steps_per_day=4,
        seed=113,
    )

    exact, _, _ = simulate_restricted_population(
        density,
        deterministic,
        config=config,
        batch_size=100,
    )
    represented, _, _ = simulate_restricted_population(
        density,
        zero_loading,
        config=config,
        batch_size=100,
    )

    np.testing.assert_allclose(represented.returns, exact.returns, atol=2.0e-13)


def test_predictable_boundary_scale_reserves_density_headroom() -> None:
    shifted = np.asarray([[[0.02, 0.08]], [[0.04, 0.08]]])
    factor_scale = np.asarray([[1.0, 0.5]])
    alpha = np.asarray([[-0.1, 0.2]])
    physical = np.zeros_like(alpha)
    active = np.ones_like(alpha, dtype=bool)
    dt = 1.0 / (252.0 * 8.0)

    gamma = _predictable_boundary_scale(
        shifted,
        factor_scale=factor_scale,
        q_drift=alpha,
        physical_drift=physical,
        active_mask=active,
        dt=dt,
    )
    adverse = (
        shifted
        + np.square(gamma)[:, None, None] * alpha * dt
        - gamma[:, None, None]
        * 12.0
        * np.sqrt(dt)
        * factor_scale
    )

    assert np.all(gamma > 0.0)
    assert np.all(gamma <= 1.0)
    assert np.all(adverse >= -1.0e-15)


def test_boundary_scale_covers_hjm_drift_outside_factor_support() -> None:
    shifted = np.asarray([[[0.5, 1.0e-6]]])
    factor_scale = np.asarray([[0.1, 0.0]])
    alpha = np.asarray([[0.0, -2.0]])
    physical = np.zeros_like(alpha)
    active = np.asarray([[True, False]])
    dt = 1.0 / (252.0 * 8.0)

    gamma = _predictable_boundary_scale(
        shifted,
        factor_scale=factor_scale,
        q_drift=alpha,
        physical_drift=physical,
        active_mask=active,
        dt=dt,
    )
    candidate = shifted + np.square(gamma)[:, None, None] * alpha * dt

    expected = np.sqrt(shifted[0, 0, 1] / (-alpha[0, 1] * dt))
    assert np.isclose(gamma[0], expected)
    assert np.all(candidate >= 0.0)


def test_cell_integrals_preserve_piecewise_linear_daily_measures() -> None:
    values = np.asarray(
        (
            (1.0, 4.0),
            (3.0, 2.0),
            (2.0, 8.0),
        ),
        dtype=np.float64,
    )
    expected_daily = 0.5 * (values[:-1] + values[1:]) / DAYS_PER_YEAR

    for steps_per_day in (1, 2, 4):
        cells = _cell_integrals(values, steps_per_day=steps_per_day)
        daily = cells.reshape(2, steps_per_day, 2).sum(axis=1)

        np.testing.assert_allclose(daily, expected_daily, atol=1.0e-18)
        np.testing.assert_allclose(
            cells.sum(axis=0), expected_daily.sum(axis=0), atol=1.0e-18
        )


def test_cell_shift_conserves_each_fixed_expiry_measure() -> None:
    values = np.asarray(
        (
            (1.0, 2.0),
            (3.0, 1.0),
            (2.0, 5.0),
            (4.0, 3.0),
        ),
        dtype=np.float64,
    )
    cells = _cell_integrals(values, steps_per_day=4)
    maturity_cells = 11

    for consumed in range(maturity_cells + 1):
        consumed_measure = cells[:consumed].sum(axis=0)
        shifted_measure = cells[consumed:maturity_cells].sum(axis=0)
        np.testing.assert_allclose(
            consumed_measure + shifted_measure,
            cells[:maturity_cells].sum(axis=0),
            atol=1.0e-18,
        )


def test_zero_jump_atom_is_invisible_to_characteristic_exponent() -> None:
    jump_x = np.linspace(-0.2, 0.2, 81)
    density = 3.0 * np.exp(-10.0 * np.abs(jump_x))
    without_zero = density.copy()
    without_zero[len(jump_x) // 2] = 0.0
    frequencies = np.linspace(-30.0, 30.0, 121)
    original = density_characteristic_function(
        tau=10.0 / DAYS_PER_YEAR,
        jump_x=jump_x,
        average_density=density,
    )(frequencies)
    reduced = density_characteristic_function(
        tau=10.0 / DAYS_PER_YEAR,
        jump_x=jump_x,
        average_density=without_zero,
    )(frequencies)

    np.testing.assert_allclose(reduced, original, rtol=0.0, atol=1.0e-15)
