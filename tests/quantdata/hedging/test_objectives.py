from __future__ import annotations

import numpy as np

from quantdata.hedging.objectives import (
    cost_lasso_minimum_variance_hedge,
    regularized_tyler_covariance,
)


def test_regularized_tyler_covariance_is_positive_definite() -> None:
    generator = np.random.default_rng(7)
    samples = generator.normal(size=(200, 4))
    samples[0] *= 100.0
    estimate = regularized_tyler_covariance(samples)
    assert estimate.converged
    assert 0.0 < estimate.shrinkage <= 1.0
    assert np.min(np.linalg.eigvalsh(estimate.covariance)) > 0.0


def test_high_dimensional_tyler_retains_cross_covariance() -> None:
    generator = np.random.default_rng(19)
    observations = 80
    dimension = 85
    factor = generator.normal(size=(observations, 1))
    loadings = np.linspace(0.3, 1.0, dimension)[None, :]
    samples = factor @ loadings + 0.2 * generator.normal(
        size=(observations, dimension)
    )
    estimate = regularized_tyler_covariance(samples)
    off_diagonal = estimate.covariance - np.diag(np.diag(estimate.covariance))
    assert estimate.converged
    assert 0.0 < estimate.shrinkage < 1.0
    assert np.max(np.abs(off_diagonal)) > 1.0e-3
    assert np.min(np.linalg.eigvalsh(estimate.covariance)) > 0.0


def test_cost_lasso_recovers_opposite_claim_exposure_without_costs() -> None:
    generator = np.random.default_rng(11)
    instruments = generator.normal(size=(2_000, 2))
    claim = 2.0 * instruments[:, 0] - 0.5 * instruments[:, 1]
    hedge = cost_lasso_minimum_variance_hedge(
        instrument_returns=instruments,
        claim_returns=claim,
        linear_transaction_costs=np.zeros(2),
    )
    assert hedge.converged
    np.testing.assert_allclose(hedge.action, [-2.0, 0.5], atol=0.08)
    assert hedge.claim_variance > 0.0
    assert hedge.objective >= 0.0


def test_large_entry_cost_kills_the_hedge() -> None:
    generator = np.random.default_rng(13)
    instruments = generator.normal(size=(500, 2))
    claim = instruments[:, 0]
    hedge = cost_lasso_minimum_variance_hedge(
        instrument_returns=instruments,
        claim_returns=claim,
        linear_transaction_costs=np.full(2, 10.0),
    )
    np.testing.assert_array_equal(hedge.action, np.zeros(2))
    assert hedge.objective == 0.5 * hedge.claim_variance
