from __future__ import annotations

from datetime import date

import numpy as np
import pytest
import torch

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.run_score_driven_student_t import (
    _historical_residuals,
    _market_tail_diagnostics,
    _stationary_log_scale_state,
)
from quantdata.dysanos.flex.score_driven_student_t import (
    ScoreDrivenStudentTConfig,
    filter_score_driven_student_t,
    score_log_mgf,
    stationary_log_mean_normalizer,
    student_t_scale_score,
)


def test_student_t_score_is_centered_and_bounded() -> None:
    generator = np.random.default_rng(1729)
    dimension = 20
    degrees_of_freedom = 7.0
    beta_draw = generator.beta(
        0.5 * dimension,
        0.5 * degrees_of_freedom,
        size=500_000,
    )
    radius = (degrees_of_freedom - 2.0) * beta_draw / (1.0 - beta_draw)
    score = student_t_scale_score(
        radius,
        dimension=dimension,
        degrees_of_freedom=degrees_of_freedom,
    )
    assert abs(float(np.mean(score))) < 0.01
    assert float(np.min(score)) >= -0.5 * dimension
    assert float(np.max(score)) <= 0.5 * degrees_of_freedom


def test_analytical_scale_normalizer_produces_unit_mean() -> None:
    generator = np.random.default_rng(9281)
    dimension = 30
    config = ScoreDrivenStudentTConfig(
        degrees_of_freedom=6.0,
        score_loading=0.04,
        persistence=0.92,
    )
    log_normalizer = stationary_log_mean_normalizer(
        dimension=dimension,
        config=config,
    )
    states = np.zeros(200_000, dtype=np.float64)
    for _ in range(250):
        beta_draw = generator.beta(
            0.5 * dimension,
            0.5 * config.degrees_of_freedom,
            size=len(states),
        )
        score = 0.5 * ((config.degrees_of_freedom + dimension) * beta_draw - dimension)
        states = config.persistence * states + config.score_loading * score
    scales = np.exp(states - log_normalizer)
    assert abs(float(np.mean(scales)) - 1.0) < 0.01


def test_zero_score_loading_reduces_to_iid_student_t() -> None:
    radii = np.asarray((2.0, 4.0, 3.0, 5.0), dtype=np.float64)
    starts = np.asarray((True, False, False, True))
    config = ScoreDrivenStudentTConfig(
        degrees_of_freedom=8.0,
        score_loading=0.0,
        persistence=0.99,
    )
    filtered = filter_score_driven_student_t(
        radii,
        starts,
        dimension=3,
        config=config,
    )
    np.testing.assert_allclose(filtered.scales, 1.0)
    assert filtered.log_mean_normalizer == 0.0
    assert np.all(np.isfinite(filtered.log_likelihood))


def test_score_log_mgf_has_zero_slope_at_origin() -> None:
    positive = score_log_mgf(
        1.0e-5,
        dimension=200,
        degrees_of_freedom=8.0,
    )
    negative = score_log_mgf(
        -1.0e-5,
        dimension=200,
        degrees_of_freedom=8.0,
    )
    assert abs((positive - negative) / 2.0e-5) < 1.0e-5
    assert positive > 0.0
    assert negative > 0.0


def test_torch_stationary_sampler_respects_analytical_mean_normalizer() -> None:
    config = ScoreDrivenStudentTConfig(
        degrees_of_freedom=7.0,
        score_loading=0.03,
        persistence=0.85,
    )
    generator = torch.Generator(device="cpu").manual_seed(771)
    state = _stationary_log_scale_state(
        50_000,
        dimension=40,
        config=config,
        dtype=torch.float64,
        device=torch.device("cpu"),
        generator=generator,
    )
    normalizer = stationary_log_mean_normalizer(dimension=40, config=config)
    assert abs(float(torch.mean(torch.exp(state - normalizer))) - 1.0) < 0.015


def test_historical_residuals_compose_the_one_session_law_once() -> None:
    transition = np.asarray(((0.8,),), dtype=np.float64)
    covariance = np.asarray(((0.25,),), dtype=np.float64)
    dates = tuple(date(2025, 1, day) for day in (2, 3, 6, 7, 8, 10))
    states = np.asarray(((1.0,), (2.0,), (3.0,), (4.0,), (5.0,), (7.0,)))

    origin, endpoint, residuals, radii = _historical_residuals(
        states,
        dates,
        mean=np.zeros(1),
        session_transition=transition,
        session_covariance=covariance,
        sessions=5,
    )

    np.testing.assert_array_equal(origin, (0,))
    np.testing.assert_array_equal(endpoint, (5,))
    expected_residual = 7.0 - 0.8**5
    expected_variance = 0.25 * sum(0.8 ** (2 * power) for power in range(5))
    np.testing.assert_allclose(residuals, ((expected_residual,),))
    np.testing.assert_allclose(radii, (expected_residual**2 / expected_variance,))


def test_market_tail_report_uses_comparable_pre_2025_gaussian_threshold() -> None:
    dates = tuple(get_business_days(date(2024, 9, 3), date(2025, 3, 31)))
    states = np.random.default_rng(19).normal(size=(len(dates), 2))

    _, report = _market_tail_diagnostics(
        states,
        dates,
        mean=np.zeros(2),
        daily_transition=0.9 * np.eye(2),
        daily_covariance=np.eye(2),
    )

    for weeks in (1, 2, 3):
        row = report[f"{weeks}w"]
        value = row["gaussian_exceedance_of_pre_2025_q95"]
        assert 0.0 <= value <= 1.0
        observed = row["2025_exceedance_of_pre_2025_q95"]
        expected_standard_error = np.sqrt(
            observed * (1.0 - observed) / row["2025_pairs"]
        )
        assert row["2025_exceedance_binomial_standard_error"] == pytest.approx(
            expected_standard_error
        )
        block = row["2025_exceedance_block_bootstrap"]
        assert block["block_length"] == min(15, row["2025_pairs"])
        assert block["replications"] == 9_999
        assert block["standard_error"] >= 0.0
        lower, upper = block["percentile_95_interval"]
        assert 0.0 <= lower <= upper <= 1.0
