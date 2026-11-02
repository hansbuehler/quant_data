from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2.saved_model_audit import AUDIT_NMS, initial_ar1_loadings


def test_initial_amplitudes_match_forward_model_predictable_scales() -> None:
    loadings = np.arange(24.0).reshape(2, 3, 4)
    arrays = {
        "loadings": loadings,
        "ar1_parameters": {
            "variance_intercept": np.asarray([0.0, 1.0, -1.0, 8.0]),
            "variance_slope": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "terminal_log_squared": np.asarray([1.0, 2.0, 3.0, 4.0]),
            "fixed_factor_scales": np.asarray([0.5, 1.0, 2.0, 3.0]),
            "scale_calibration": np.asarray([1.0, 0.9, 1.1, 1.0]),
            "log_scale_bound": np.log(3.0),
        },
    }
    actual, scales = initial_ar1_loadings(arrays)
    expected = np.asarray([0.5 * np.exp(0.05), 0.9 * np.exp(0.7), 2.2 * np.exp(-0.05), 9.0])
    np.testing.assert_allclose(scales, expected)
    np.testing.assert_allclose(actual, loadings * expected)
    np.testing.assert_array_equal(arrays["loadings"], loadings)


def test_invalid_amplitudes_are_not_silently_repaired() -> None:
    arrays = {
        "loadings": np.ones((1, 2, 1)),
        "ar1_parameters": {
            "variance_intercept": np.ones(1), "variance_slope": np.ones(1),
            "terminal_log_squared": np.zeros(1), "fixed_factor_scales": np.zeros(1),
            "scale_calibration": np.ones(1), "log_scale_bound": 1.0,
        },
    }
    with pytest.raises(ValueError, match="finite and positive"):
        initial_ar1_loadings(arrays)


def test_common_audit_strikes_are_inside_both_model_ranges() -> None:
    assert len(AUDIT_NMS) == 11
    assert min(AUDIT_NMS) >= -3.0
    assert max(AUDIT_NMS) <= 2.0
    assert 0.0 in AUDIT_NMS
