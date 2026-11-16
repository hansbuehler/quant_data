"""Daily embedding must reproduce the full fitted weekly covariance, not Q/5."""

import numpy as np
import pytest

from quantdata.hedging.model_audit.paper_v1 import daily_law


def test_daily_embedding_preserves_nondiagonal_weekly_law() -> None:
    generator = np.array([[.4, .05], [-.05, .25]])
    weekly_factor = np.array([[.3, 0.], [.08, .2]])
    daily, factor, diagnostics = daily_law(generator, weekly_factor)
    covariance = np.zeros((2, 2))
    propagation = np.eye(2)
    for _ in range(5):
        covariance += propagation.T @ factor @ factor.T @ propagation
        propagation = propagation @ daily
    np.testing.assert_allclose(covariance, weekly_factor @ weekly_factor.T, atol=1e-12)
    assert diagnostics["state_dimension"] == 2
    assert diagnostics["covariance_projection_applied"] is False
    assert not np.allclose(factor @ factor.T, weekly_factor @ weekly_factor.T / 5)


def test_nonembeddable_weekly_covariance_is_not_projected() -> None:
    with pytest.raises(ValueError, match="daily embedding"):
        daily_law(np.array([[.1, 4.], [-4., .1]]), np.diag([1., .01]))


def test_unstable_generator_is_rejected() -> None:
    with pytest.raises(ValueError, match="not stationary"):
        daily_law(-np.eye(2), np.eye(2))


def test_legacy_surrogate_runner_cannot_publish_another_paper_v1_row() -> None:
    from quantdata.hedging.model_audit.versioned import run

    with pytest.raises(RuntimeError, match="mislabels PCA surrogates"):
        run(None, None, None, None, None)
