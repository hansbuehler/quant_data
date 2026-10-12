from __future__ import annotations

from dataclasses import replace
from datetime import date

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_calibration import (
    RollingDlvCalibrationConfig,
    _newey_west_mean_standard_error,
    calibrate_current_dlv_drifts,
    project_compatible_physical_drift,
    project_constant_physical_drift,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    FixedContractDlvHistory,
    RollingDlvFactorFit,
)


def test_compatible_physical_projection_recovers_loading_span() -> None:
    loadings = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.2], [0.0, 0.7]],
            [[0.1, 0.4], [0.3, -0.2], [0.8, 0.1]],
        ]
    )
    risk_neutral = np.asarray(
        [[0.1, 0.2, -0.1], [0.3, -0.2, 0.05]]
    )
    risk_price = np.asarray([0.4, -0.3])
    empirical = risk_neutral + np.einsum("ijf,f->ij", loadings, risk_price)
    weights = np.asarray([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])

    projection = project_compatible_physical_drift(
        empirical_log_drift=empirical,
        risk_neutral_log_drift=risk_neutral,
        coordinate_weights=weights,
        annualized_loadings=loadings,
        ridge=0.0,
    )

    assert projection.market_price_of_risk == pytest.approx(risk_price)
    assert projection.achievable_log_drift == pytest.approx(empirical)
    assert projection.weighted_residual_rms == pytest.approx(0.0, abs=1.0e-14)
    assert projection.achievable_fraction == pytest.approx(1.0)


def test_compatible_physical_projection_holds_identified_factor_fixed() -> None:
    loadings = np.asarray(
        [
            [[1.0, 0.2], [0.5, -0.3], [0.1, 0.8]],
            [[0.4, -0.1], [0.7, 0.4], [0.2, 0.6]],
        ]
    )
    risk_neutral = np.zeros((2, 3))
    risk_price = np.asarray([-0.25, 0.7])
    empirical = np.einsum("ijf,f->ij", loadings, risk_price)
    fixed = np.asarray([-0.25, np.nan])

    projection = project_compatible_physical_drift(
        empirical_log_drift=empirical,
        risk_neutral_log_drift=risk_neutral,
        coordinate_weights=np.ones((2, 3)),
        annualized_loadings=loadings,
        ridge=0.0,
        fixed_market_price_of_risk=fixed,
    )

    assert projection.market_price_of_risk == pytest.approx(risk_price)
    assert projection.weighted_residual_rms == pytest.approx(0.0, abs=1.0e-14)


def test_current_dlv_calibration_separates_conditional_and_unconditional_p_drift() -> None:
    base_day = synthetic_one_day_transition().origin
    support = np.asarray([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    dlvsq = np.asarray(
        [
            [0.04, 0.035, 0.03, 0.04],
            [0.045, 0.04, 0.035, 0.045],
            [0.05, 0.045, 0.04, 0.05],
        ]
    )
    day = replace(
        base_day,
        dlv_model_expiries_y=np.asarray([1.0, 2.0, 3.0]) / DAYS_PER_YEAR,
        dlv_support_strikes=support,
        dlv_raw_log_excess_dlvsq=np.log(dlvsq - base_day.minimum_dlvsq),
    )
    origin_dates = (date(2024, 12, 27), date(2024, 12, 30))
    endpoint_dates = (date(2024, 12, 30), date(2024, 12, 31))
    current_log_dlvsq = np.log(day.dlvsq[1:])
    origin_log_dlvsq = np.stack(
        (current_log_dlvsq - 0.02, current_log_dlvsq - 0.01)
    )
    endpoint_log_dlvsq = origin_log_dlvsq + np.asarray([0.003, 0.005])[:, None, None]
    history = FixedContractDlvHistory(
        origin_dates=origin_dates,
        endpoint_dates=endpoint_dates,
        origin_log_dlvsq=origin_log_dlvsq,
        endpoint_log_dlvsq=endpoint_log_dlvsq,
        front_cdf_values=np.asarray([0.4, 0.6]),
        spot_gaussian_scores=np.asarray([-0.25, 0.25]),
        interpolation_clip_fractions=np.zeros(2),
        forward_tail_extrapolation_fractions=np.zeros(2),
    )
    daily_loadings = np.stack(
        (
            np.full((2, 4), 0.01),
            np.asarray(
                [
                    [0.02, 0.01, -0.01, -0.02],
                    [0.01, 0.02, -0.02, -0.01],
                ]
            ),
        ),
        axis=-1,
    )
    factors = RollingDlvFactorFit(
        as_of_date=day.trading_date,
        training_origin_dates=origin_dates,
        training_endpoint_dates=endpoint_dates,
        conditional_intercept=np.full((2, 4), 0.02),
        conditional_slope=np.full((2, 4), 0.99),
        centered_residuals=np.zeros((2, 2, 4)),
        spot_score_mean=0.0,
        spot_score_std=1.0,
        daily_loadings=daily_loadings,
        daily_factor_variances=np.sum(daily_loadings**2, axis=(0, 1)),
        factor_scores=np.zeros((2, 2)),
        explained_variance_fraction=1.0,
        interpolation_clip_fraction=0.0,
        forward_tail_extrapolation_fraction=0.0,
        front_spot_market_price_of_risk=0.42,
        front_spot_market_price_of_risk_standard_error=0.8,
    )

    calibration = calibrate_current_dlv_drifts(
        day=day,
        history=history,
        factors=factors,
        greek_factor_count=1,
        physical_ar_z_threshold=0.0,
    )

    expected_conditional = DAYS_PER_YEAR * (
        calibration.physical_conditional_intercept
        + calibration.physical_conditional_slope * current_log_dlvsq
        - current_log_dlvsq
    )
    expected_unconditional = DAYS_PER_YEAR * np.full((2, 4), 0.004)
    assert calibration.front_density.sum() == pytest.approx(1.0)
    assert calibration.future_densities.sum(axis=-1) == pytest.approx(np.ones(2))
    assert calibration.coordinate_weights.sum(axis=-1) == pytest.approx(np.ones(2))
    assert calibration.conditional_physical.empirical_log_drift == pytest.approx(
        expected_conditional
    )
    assert calibration.unconditional_physical.empirical_log_drift == pytest.approx(
        expected_unconditional
    )
    assert calibration.conditional_unconditional_weighted_rms == pytest.approx(
        float(
            np.sqrt(
                np.sum(
                    calibration.coordinate_weights
                    * (expected_conditional - expected_unconditional) ** 2
                )
                / np.sum(calibration.coordinate_weights)
            )
        )
    )
    expected_spot_risk_price = factors.front_spot_market_price_of_risk
    assert calibration.front_spot_market_price_of_risk == pytest.approx(
        expected_spot_risk_price
    )
    assert calibration.conditional_physical.market_price_of_risk[0] == pytest.approx(
        expected_spot_risk_price
    )
    assert calibration.unconditional_physical.market_price_of_risk[0] == pytest.approx(
        expected_spot_risk_price
    )


def test_constant_physical_projection_recovers_one_rolling_risk_price() -> None:
    sample_count = 4
    loadings = np.asarray(
        [
            [[0.8, 0.1], [0.4, -0.2], [0.1, 0.5]],
            [[0.2, 0.6], [-0.3, 0.2], [0.7, -0.1]],
        ]
    )
    generator = np.random.default_rng(101)
    risk_neutral = generator.normal(size=(sample_count, 2, 3))
    risk_price = np.asarray([0.3, -0.4])
    projected = np.einsum("ijf,f->ij", loadings, risk_price)
    empirical = risk_neutral + projected[None, ...]
    weights = np.full((sample_count, 2, 3), 1.0 / 3.0)
    origins = tuple(date(2024, 1, 2 + index) for index in range(sample_count))
    endpoints = tuple(date(2024, 1, 3 + index) for index in range(sample_count))

    fit = project_constant_physical_drift(
        training_origin_dates=origins,
        training_endpoint_dates=endpoints,
        empirical_log_drifts=empirical,
        risk_neutral_log_drifts=risk_neutral,
        coordinate_weights=weights,
        annualized_loadings=loadings,
        ridge=0.0,
    )

    assert fit.market_price_of_risk == pytest.approx(risk_price)
    assert fit.market_price_of_risk_standard_errors.shape == risk_price.shape
    assert np.all(fit.market_price_of_risk_standard_errors >= 0.0)
    assert fit.achievable_log_drifts == pytest.approx(empirical)
    assert fit.weighted_residual_rms == pytest.approx(0.0, abs=1.0e-14)
    assert fit.achievable_fraction == pytest.approx(1.0)


def test_constant_physical_projection_holds_identified_factor_fixed() -> None:
    sample_count = 3
    loadings = np.asarray(
        [[[0.8, 0.1], [0.4, -0.2]], [[0.2, 0.6], [-0.3, 0.2]]]
    )
    risk_price = np.asarray([-0.15, 0.45])
    risk_neutral = np.zeros((sample_count, 2, 2))
    empirical = risk_neutral + np.einsum("ijf,f->ij", loadings, risk_price)[None]
    origins = tuple(date(2024, 1, 2 + index) for index in range(sample_count))
    endpoints = tuple(date(2024, 1, 3 + index) for index in range(sample_count))

    fit = project_constant_physical_drift(
        training_origin_dates=origins,
        training_endpoint_dates=endpoints,
        empirical_log_drifts=empirical,
        risk_neutral_log_drifts=risk_neutral,
        coordinate_weights=np.ones_like(empirical),
        annualized_loadings=loadings,
        ridge=0.0,
        fixed_market_price_of_risk=np.asarray([-0.15, np.nan]),
    )

    assert fit.market_price_of_risk == pytest.approx(risk_price)
    assert fit.weighted_residual_rms == pytest.approx(0.0, abs=1.0e-14)


def test_validation_gate_requires_positive_economic_and_statistical_gain() -> None:
    config = RollingDlvCalibrationConfig()

    assert config.minimum_validation_improvement_fraction > 0.0
    assert config.minimum_validation_gain_t >= 1.96


def test_linked_newey_west_mean_error_matches_manual_lag_one() -> None:
    values = np.asarray([1.0, 2.0, 4.0, 3.0])
    linked = np.asarray([True, True, False])
    centered = values - np.mean(values)
    expected_sum = float(centered @ centered)
    expected_sum += float(centered[1:3] @ centered[:2])

    standard_error = _newey_west_mean_standard_error(
        values,
        linked=linked,
        lags=1,
    )

    assert standard_error == pytest.approx(np.sqrt(expected_sum) / len(values))
