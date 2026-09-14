"""Tests for fixed-expiry rollover raw log-DLV innovations."""

from __future__ import annotations

from datetime import date

import numpy as np
import torch

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.fixed_expiry_ar1 import (
    fit_fixed_expiry_ar1,
    torch_fixed_expiry_raw_log_loadings,
)
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs


def _weekly_dates(start: date, end: date) -> tuple[date, ...]:
    return tuple(get_business_days(start, end)[::5])


def test_fixed_expiry_ar1_fits_rollover_coordinates_and_joint_covariance() -> None:
    dates = _weekly_dates(date(2024, 1, 2), date(2024, 9, 30))
    generator = np.random.default_rng(20260912)
    num_inner = 3
    raw_dlvs = generator.normal(size=(len(dates), 4, num_inner))
    fit = fit_fixed_expiry_ar1(
        raw_dlvs,
        dates,
        rank=5,
    )

    origin, endpoint = exact_weekly_pairs(dates, weeks=1)
    predictor = raw_dlvs[origin, 1:]
    target = raw_dlvs[endpoint, :3]
    expected_slope = np.empty((3, num_inner), dtype=np.float64)
    expected_intercept = np.empty_like(expected_slope)
    for expiry in range(3):
        for strike in range(num_inner):
            design = np.column_stack((np.ones(len(origin)), predictor[:, expiry, strike]))
            coefficient = np.linalg.lstsq(
                design,
                target[:, expiry, strike],
                rcond=None,
            )[0]
            expected_intercept[expiry, strike] = coefficient[0]
            expected_slope[expiry, strike] = coefficient[1]

    np.testing.assert_allclose(fit.conditional_intercept, expected_intercept)
    np.testing.assert_allclose(fit.conditional_slope, expected_slope)
    assert fit.centered_residuals.shape == (len(origin), 3 * num_inner)
    np.testing.assert_allclose(
        np.mean(fit.centered_residuals, axis=0),
        0.0,
        atol=2.0e-16,
    )
    assert fit.raw_log_loadings.shape == (4, num_inner, 5)
    np.testing.assert_array_equal(fit.raw_log_loadings[0], 0.0)

    weekly_from_sde = fit.raw_log_loadings[1:].reshape(3 * num_inner, 5) * np.sqrt(
        fit.calendar_time_increment_y
    )
    np.testing.assert_allclose(
        weekly_from_sde,
        fit.weekly_innovation_loadings,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(fit.residual_covariance)
    order = np.argsort(eigenvalues)[::-1][:5]
    expected_rank_covariance = (eigenvectors[:, order] * eigenvalues[order][None]) @ eigenvectors[
        :, order
    ].T
    np.testing.assert_allclose(
        weekly_from_sde @ weekly_from_sde.T,
        expected_rank_covariance,
        atol=2.0e-14,
    )
    assert fit.diagnostics.sample_count == len(origin)
    assert fit.diagnostics.residual_dimension == 3 * num_inner
    assert len(fit.diagnostics.residual_covariance_eigenvalues) == 3 * num_inner
    assert 0.0 < fit.diagnostics.explained_variance_fraction <= 1.0


def test_fixed_expiry_ar1_uses_only_training_endpoints_and_native_dlv_prefix() -> None:
    dates = _weekly_dates(date(2024, 9, 3), date(2025, 3, 31))
    generator = np.random.default_rng(17)
    num_inner = 2
    dlv_tower = generator.normal(size=(len(dates), 4, num_inner))
    native_blocks = np.full((len(dates), 4, num_inner + 2), 1.0e9)
    native_blocks[:, :, :num_inner] = dlv_tower
    native_states = native_blocks.reshape(len(dates), -1)

    full = fit_fixed_expiry_ar1(
        native_states,
        dates,
        num_inner_strikes=num_inner,
        rank=2,
    )
    training_row = np.asarray([value <= date(2024, 12, 31) for value in dates])
    truncated = fit_fixed_expiry_ar1(
        dlv_tower[training_row],
        tuple(value for value in dates if value <= date(2024, 12, 31)),
        rank=2,
    )

    np.testing.assert_allclose(full.conditional_intercept, truncated.conditional_intercept)
    np.testing.assert_allclose(full.conditional_slope, truncated.conditional_slope)
    np.testing.assert_allclose(full.residual_covariance, truncated.residual_covariance)
    np.testing.assert_allclose(full.raw_log_loadings, truncated.raw_log_loadings)
    assert full.diagnostics.sample_count == truncated.diagnostics.sample_count


def test_fixed_expiry_ar1_handles_constant_predictors_and_converts_to_torch() -> None:
    dates = _weekly_dates(date(2024, 1, 2), date(2024, 5, 31))
    raw_dlvs = np.zeros((len(dates), 4, 2), dtype=np.float64)
    raw_dlvs[:, 0, 0] = np.linspace(-1.0, 1.0, len(dates)) ** 2
    fit = fit_fixed_expiry_ar1(raw_dlvs, dates, rank=1)
    loadings = torch_fixed_expiry_raw_log_loadings(
        fit,
        dtype=torch.float32,
        device="cpu",
    )

    np.testing.assert_array_equal(fit.conditional_slope, 0.0)
    assert fit.diagnostics.num_zero_variance_predictors == 6
    assert loadings.dtype == torch.float32
    assert loadings.shape == (4, 2, 1)
    torch.testing.assert_close(loadings[0], torch.zeros_like(loadings[0]))
