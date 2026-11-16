"""Price-only covariance PCA, no look-ahead and variance-calibration conventions."""

from datetime import date, timedelta

import numpy as np
import pytest
from cdxcore.bs import bs

from quantdata.hedging.model_audit.straddle_pca import (
    fit_return_pca,
    forecast_summary,
    select_window,
    source_straddles,
)


def test_return_pca_recovers_covariance_eigenspace_and_diagonal() -> None:
    random = np.random.default_rng(203)
    samples = random.normal(size=(80, 6)) @ np.diag([4., 3., 2., 1., .5, .1])
    fitted = fit_return_pca(samples, rank=4)
    covariance = np.cov(samples, rowvar=False)
    values, directions = np.linalg.eigh(covariance)
    np.testing.assert_allclose(fitted.eigenvalues, values[::-1][:4], atol=1e-12)
    np.testing.assert_allclose(fitted.full_variance_diagonal, np.diag(covariance), atol=1e-12)
    np.testing.assert_allclose(fitted.directions @ fitted.directions.T, directions[:, -4:] @ directions[:, -4:].T, atol=1e-12)
    assert fitted.variance_diagonal(4).sum() < fitted.full_variance_diagonal.sum()
    np.testing.assert_allclose(fitted.reconstruct(samples, 4).mean(axis=0), samples.mean(axis=0), atol=1e-12)


def test_window_uses_only_transitions_completed_by_the_origin() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(8)]
    pairs = tuple((days[index], days[index+1], index+1) for index in range(7))
    assert select_window(pairs, days[4], count=3) == list(pairs[1:4])
    with pytest.raises(ValueError, match="Insufficient"):
        select_window(pairs, days[1], count=3)


def test_forecast_calibration_is_not_bounded_by_one() -> None:
    actual = np.array([[-1.], [1.], [-1.], [1.]])
    result = forecast_summary(np.zeros_like(actual), np.full_like(actual, 2.), actual)
    assert result["variance_over_forecast_mse"] == pytest.approx(2.)
    assert result["pooled_over_observed_variance"] == pytest.approx(1.5)


def test_market_straddles_use_prices_and_exact_zero_time_settlement(monkeypatch) -> None:
    def time_values(pricer, *, expiries_y, strikes):
        assert np.all(expiries_y > 0.)
        return bs.price(k=strikes[None], vol=.2, sqrtT=np.sqrt(expiries_y[:, None]), is_call=strikes[None] >= 1.)

    monkeypatch.setattr("quantdata.hedging.model_audit.straddle_pca.surface_time_values", time_values)
    strikes = np.array([.9, 1., 1.1])
    result = source_straddles(None, business_dtes=np.array([0., 1., 5.]), strikes=strikes)
    np.testing.assert_array_equal(result[0], np.abs(1. - strikes))
    expected = (bs.price(k=strikes[None], vol=.2, sqrtT=np.sqrt(np.array([1., 5.])[:, None] / 252.), is_call=True)
                + bs.price(k=strikes[None], vol=.2, sqrtT=np.sqrt(np.array([1., 5.])[:, None] / 252.), is_call=False))
    np.testing.assert_allclose(result[1:], expected, atol=1e-12)


def test_return_pca_alignment_preserves_current_variance_order() -> None:
    random = np.random.default_rng(89)
    samples = random.normal(size=(90, 8))
    first = fit_return_pca(samples, rank=4)
    following = fit_return_pca(samples + random.normal(scale=.001, size=samples.shape), rank=4, previous=first)
    assert np.all(np.diff(following.eigenvalues) <= 0.)
    assert np.all(np.sum(first.directions * following.directions, axis=0) > 0.)
