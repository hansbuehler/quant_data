from __future__ import annotations

import math

import numpy as np
import pytest

from quantdata.dysanos.flex.compatible_physical_drift import (
    factor_drift_audit,
    fit_compatible_constant_drift,
    holm_adjusted_p_values,
    newey_west_mean_covariance,
)


def test_holm_adjustment_preserves_original_order() -> None:
    values = np.asarray((0.04, 0.001, 0.02))

    adjusted = holm_adjusted_p_values(values)

    np.testing.assert_allclose(adjusted, (0.04, 0.003, 0.04))


def test_compatible_fit_selects_signal_and_obeys_ess_budget() -> None:
    rng = np.random.default_rng(17)
    observations = rng.normal(size=(4_000, 3))
    observations[:, 0] += 0.7
    observations[:, 2] -= 0.4

    fit = fit_compatible_constant_drift(
        observations,
        time_increment_y=0.02,
        horizon_steps=4,
        target_horizon_ess_fraction=0.25,
        hac_lag=0,
    )

    np.testing.assert_array_equal(fit.supported_modes, (True, False, True))
    assert fit.horizon_expected_ess_fraction == pytest.approx(0.25)
    assert fit.radial_scale < 1.0
    np.testing.assert_allclose(
        fit.annualized_market_price_of_risk,
        fit.weekly_factor_mean / math.sqrt(0.02),
    )


def test_factor_drift_audit_accepts_the_sample_mean() -> None:
    rng = np.random.default_rng(23)
    observations = rng.normal(size=(2_000, 2)) + np.asarray((0.2, -0.1))
    candidate = np.mean(observations, axis=0)

    audit = factor_drift_audit(observations, candidate, hac_lag=0)

    assert audit["residual_maximum_absolute_hac_t"] < 1.0e-10
    assert audit["residual_joint_p_value"] == pytest.approx(1.0)


def test_newey_west_rejects_too_short_input() -> None:
    with pytest.raises(ValueError):
        newey_west_mean_covariance(np.ones((4, 2)), lag=3)
