"""Focused checks for the Tangent Levy v2 experiment contract."""

from __future__ import annotations

import numpy as np

from quantdata.dysanos.tangent_levy.pricing import density_call_prices
from quantdata.dysanos.tangent_levy.v2_audit import unconditional_coordinate_audit
from quantdata.dysanos.tangent_levy.v2_config import (
    TangentLevyV2AuditConfig,
    TangentLevyV2DynamicsConfig,
    TangentLevyV2MarketConfig,
)
from quantdata.dysanos.tangent_levy.v2_spot import (
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
