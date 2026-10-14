from __future__ import annotations

from pathlib import Path

import numpy as np

from quantdata.gromm2.factor_law import (
    FactorLawConfig,
    FactorLawDataset,
    _bounded_spot_correlations,
    _q_point_spot_correlated_negative_log_likelihood,
    build_factor_law_dataset,
    fit_ar1_factor_law,
    fit_fixed_factor_law,
    fit_recurrent_factor_law,
)


def test_spot_correlated_score_matches_dense_gaussian_formula() -> None:
    factor_scores = np.asarray(
        [[0.4, -0.2, 0.7], [-0.8, 0.5, 0.1]],
        dtype=np.float64,
    )
    factor_scales = np.asarray(
        [[0.7, 1.1, 0.9], [1.2, 0.8, 1.4]],
        dtype=np.float64,
    )
    correlations = np.asarray(
        [[0.25, -0.15], [-0.20, 0.30]],
        dtype=np.float64,
    )

    actual = _q_point_spot_correlated_negative_log_likelihood(
        factor_scores,
        factor_scales,
        correlations,
    )
    expected = []
    for score, scale, correlation in zip(
        factor_scores,
        factor_scales,
        correlations,
        strict=True,
    ):
        correlation_matrix = np.eye(3)
        correlation_matrix[0, 1:] = correlation
        correlation_matrix[1:, 0] = correlation
        covariance = scale[:, None] * correlation_matrix * scale[None, :]
        _, log_determinant = np.linalg.slogdet(covariance)
        quadratic = score @ np.linalg.solve(covariance, score)
        expected.append(0.5 * (quadratic + log_determinant) / 3.0)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_spot_correlation_map_is_strictly_positive_definite() -> None:
    raw = np.asarray([[100.0, -50.0, 25.0], [0.2, 0.3, -0.1]])
    bound = 0.95
    correlations = _bounded_spot_correlations(raw, bound=bound)

    assert np.all(np.linalg.norm(correlations, axis=1) < bound)
    for correlation in correlations:
        matrix = np.eye(4)
        matrix[0, 1:] = correlation
        matrix[1:, 0] = correlation
        assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


def test_direct_ragged_level_scores_do_not_require_rectangular_panel(
    tmp_path: Path,
) -> None:
    observations = 128
    factor_count = 2
    origin_indices = np.arange(observations, dtype=np.int64)
    estimate = tmp_path / "estimate.npz"
    np.savez_compressed(
        estimate,
        factor_scores=np.ones((observations, factor_count)),
        physical_market_prices_of_risk=np.zeros((observations, factor_count)),
        level_mean=np.zeros(5),
        level_basis=np.ones((5, 3)),
        level_scale=np.ones(3),
        level_scores=np.ones((observations, 3)),
        pair_origin_indices=origin_indices,
        dates=np.asarray([f"d{index:03d}" for index in origin_indices]),
    )

    result = build_factor_law_dataset(
        base_estimate=estimate,
        history_path=tmp_path / "not-read.h5",
        dataset=tmp_path / "not-read",
        config=FactorLawConfig(),
    )

    assert result.inputs.shape == (observations, 3 + factor_count)
    assert result.train_count == 102


def _synthetic_factor_panel(*, observations: int = 240) -> FactorLawDataset:
    generator = np.random.default_rng(17)
    factor_count = 3
    factor_scores = np.empty((observations, factor_count))
    risk_prices = np.empty_like(factor_scores)
    log_variance = np.zeros(factor_count)
    risk_price = np.asarray([0.2, -0.4, 0.1])
    for index in range(observations):
        log_variance = 0.15 + 0.75 * log_variance + generator.normal(
            scale=0.12, size=factor_count
        )
        factor_scores[index] = np.exp(0.5 * log_variance) * generator.normal(
            size=factor_count
        )
        risk_price = 0.05 + 0.85 * risk_price + generator.normal(
            scale=0.03, size=factor_count
        )
        risk_prices[index] = risk_price
    train_count = 180
    fixed_scales = np.sqrt(np.mean(factor_scores[:train_count] ** 2, axis=0))
    risk_mean = np.mean(risk_prices[:train_count], axis=0)
    risk_scale = np.std(risk_prices[:train_count], axis=0, ddof=1)
    inputs = np.column_stack(
        (
            np.sin(np.arange(observations) / 11.0),
            np.vstack((np.zeros((1, factor_count)), factor_scores[:-1])),
        )
    )
    return FactorLawDataset(
        dates=np.asarray(
            [f"2024-01-{index + 1:02d}" for index in range(observations)]
        ),
        origin_indices=np.arange(observations, dtype=np.int64),
        inputs=inputs,
        reset=np.arange(observations) == 0,
        factor_scores=factor_scores,
        physical_risk_prices=risk_prices,
        fixed_factor_scales=fixed_scales,
        risk_price_mean=risk_mean,
        risk_price_scale=risk_scale,
        input_mean=np.zeros(inputs.shape[1]),
        input_scale=np.ones(inputs.shape[1]),
        train_count=train_count,
    )


def test_ar1_factor_law_improves_predictable_physical_drift() -> None:
    data = _synthetic_factor_panel()
    fixed = fit_fixed_factor_law(data)
    ar1 = fit_ar1_factor_law(
        data,
        config=FactorLawConfig(max_epochs=10, patience=5),
    )

    assert np.isfinite(ar1.validation_q_negative_log_likelihood)
    assert (
        ar1.validation_p_drift_standardized_rmse
        < fixed.validation_p_drift_standardized_rmse
    )
    assert ar1.accepted_p


def test_recurrent_factor_laws_have_finite_outputs() -> None:
    data = _synthetic_factor_panel(observations=96)
    data = FactorLawDataset(
        **{
            **data.__dict__,
            "train_count": 72,
        }
    )
    config = FactorLawConfig(
        hidden_size=4,
        max_epochs=20,
        patience=10,
        progress_every=20,
    )

    for architecture in ("lstm", "mamba"):
        fit = fit_recurrent_factor_law(
            data,
            architecture=architecture,
            config=config,
        )
        assert fit.architecture == architecture
        assert fit.epochs > 0
        assert np.isfinite(fit.validation_q_negative_log_likelihood)
        assert np.isfinite(fit.validation_p_drift_standardized_rmse)
        assert fit.terminal_state

    correlated = fit_recurrent_factor_law(
        data,
        architecture="mamba",
        config=config,
        learn_spot_correlation=True,
    )
    assert correlated.architecture == "mamba_corr"
    assert correlated.parameters["output_bias"].shape == (3 * data.factor_count - 1,)
    assert np.isfinite(correlated.validation_q_negative_log_likelihood)
