"""Uniform observable variance scores must not depend on model-specific coverage labels."""

import numpy as np
import pytest
from cdxcore.bs import bs

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.explained import (
    centered_energies,
    covariance_span,
    explained_percent,
    span_reconstruction,
    straddle_risk_weights,
)
from quantdata.hedging.model_audit.run import COMMON_NMS, GRID
from quantdata.hedging.model_audit.versioned import MODELS, SPECIFICATIONS


def test_straddle_risk_weights_match_vector_black_price_derivatives() -> None:
    volatility = np.full((7, 7), .2)
    times = np.asarray(GRID.expiries_y)[:, None]
    strikes = NMS.from_nms(COMMON_NMS[None, :], times, vol=.2)
    step = 1e-5
    upper = bs.price(k=strikes, vol=volatility * np.exp(step), sqrtT=np.sqrt(times))
    lower = bs.price(k=strikes, vol=volatility * np.exp(-step), sqrtT=np.sqrt(times))
    finite = (upper - lower) / (step * np.sqrt(times))
    np.testing.assert_allclose(straddle_risk_weights(np.log(volatility)), finite.ravel(), rtol=1e-8, atol=1e-10)


def test_span_coverage_is_not_a_variance_scale_calibration_ratio() -> None:
    covariance = np.diag([4., 1., .1])
    weights = np.array([.5, 2., 1.])
    first = covariance_span(covariance, weights, rank=2)
    collapsed = covariance_span(covariance * 1e-20, weights, rank=2)
    np.testing.assert_allclose(first @ first.T, collapsed @ collapsed.T)
    assert explained_percent(100., 20.) == pytest.approx(80.)
    assert explained_percent(100., 120.) == pytest.approx(-20.)


def test_same_projection_and_causal_mean_apply_to_training_and_holdout() -> None:
    samples = np.array([[1., 10.], [3., 10.], [5., 10.]])
    mean = samples.mean(axis=0)
    weights = np.array([2., 1.])
    basis = np.array([[1.], [0.]])
    fitted = span_reconstruction(samples, mean, weights, basis)
    total, error = centered_energies(samples * weights, fitted)
    assert explained_percent(total, error) == pytest.approx(100.)
    holdout = np.array([[11., 50.]])
    prediction = span_reconstruction(holdout, mean, weights, basis)
    np.testing.assert_array_equal(prediction, [[22., 10.]])


def test_unidentified_and_non_psd_covariance_are_not_filled_with_arbitrary_pcs() -> None:
    with pytest.raises(ValueError, match="not identified"):
        covariance_span(np.zeros((3, 3)), np.ones(3), rank=2)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        covariance_span(np.diag([-1., 2., 3.]), np.ones(3), rank=2)


def test_only_the_two_requested_dysanos_versions_are_selected() -> None:
    assert [name for name in MODELS if name.startswith("DYSANOS")] == ["DYSANOS v1", "DYSANOS v2"]
    assert SPECIFICATIONS["DYSANOS v1"]["projection"] == "affine200_unweighted"
    assert SPECIFICATIONS["DYSANOS v2"]["variant"] == "orthogonal"
    assert SPECIFICATIONS["DYSANOS v1"]["rank"] == 8
    assert SPECIFICATIONS["DYSANOS v2"]["rank"] == 16
