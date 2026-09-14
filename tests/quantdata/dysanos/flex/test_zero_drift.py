"""Tests for covariance-aware Q zero-drift inference."""

from __future__ import annotations

import torch

from quantdata.dysanos.flex.run_all_state_joint_generic_q_existence import (
    _bootstrap_means,
    _cross_gmm_top_tail_objective,
    _cyclic_state_minibatch,
    _global_objective,
    _monitoring_is_due,
    _monitoring_zero_drift_pass,
)
from quantdata.dysanos.flex.zero_drift import (
    adjusted_path_mean_covariance,
    beta_factor_block_zero_drift_statistics,
    beta_factor_covariance,
    block_zero_drift_statistics,
    correlation_conditioned_precision,
    quadratic_zero_drift,
    raw_path_mean_covariance,
    sign_flip_zero_drift_bootstrap,
    spectral_zero_drift,
)


def test_adjusted_sufficient_statistics_match_explicit_residuals() -> None:
    generator = torch.Generator().manual_seed(941)
    paths = 2_000
    instruments = 5
    spot = torch.randn(paths, generator=generator, dtype=torch.float64)
    returns = torch.randn(
        (paths, instruments), generator=generator, dtype=torch.float64
    ) + spot[:, None] * torch.linspace(0.2, 1.0, instruments)
    centered_spot = spot - torch.mean(spot)
    centered_returns = returns - torch.mean(returns, dim=0)
    beta = torch.sum(centered_spot[:, None] * centered_returns, dim=0) / torch.sum(
        centered_spot.square()
    )
    mean, covariance = adjusted_path_mean_covariance(
        path_sum=torch.sum(returns, dim=0),
        path_cross_product_sum=torch.einsum("pi,pj->ij", returns, returns),
        spot_sum=torch.sum(spot),
        spot_square_sum=torch.sum(spot.square()),
        spot_return_product_sum=torch.sum(spot[:, None] * returns, dim=0),
        beta=beta,
        num_paths=paths,
    )
    residual = returns - beta * spot[:, None]
    explicit_mean = torch.mean(residual, dim=0)
    centered_residual = residual - explicit_mean
    explicit_covariance = (
        torch.einsum("pi,pj->ij", centered_residual, centered_residual)
        / (paths - 2)
        / paths
    )
    torch.testing.assert_close(mean, explicit_mean, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        covariance,
        explicit_covariance,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_raw_sufficient_statistics_match_explicit_returns() -> None:
    generator = torch.Generator().manual_seed(945)
    returns = torch.randn((3_000, 7), generator=generator, dtype=torch.float64)
    mean, covariance = raw_path_mean_covariance(
        path_sum=torch.sum(returns, dim=0),
        path_cross_product_sum=torch.einsum("pi,pj->ij", returns, returns),
        num_paths=len(returns),
    )
    torch.testing.assert_close(mean, torch.mean(returns, dim=0))
    torch.testing.assert_close(
        covariance,
        torch.cov(returns.T) / len(returns),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_beta_factor_covariance_preserves_spot_direction_and_shrinks_residual() -> None:
    beta = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    spot_variance = torch.tensor([0.04], dtype=torch.float64)
    residual = torch.tensor(
        [[[0.01, 0.009], [0.009, 0.01]]], dtype=torch.float64
    )
    result = beta_factor_covariance(
        beta=beta,
        spot_variance_of_mean=spot_variance,
        residual_covariance_of_mean=residual,
        residual_diagonal_shrinkage=0.25,
        residual_ridge_fraction=1.0e-6,
    )
    expected_factor = spot_variance[:, None, None] * (
        beta[:, :, None] * beta[:, None, :]
    )
    torch.testing.assert_close(
        result.spot_factor_covariance_of_mean,
        expected_factor,
    )
    torch.testing.assert_close(
        result.covariance_of_mean,
        result.spot_factor_covariance_of_mean
        + result.residual_covariance_of_mean,
    )
    expected_residual_correlation = 0.75 * 0.9
    actual_residual_correlation = (
        result.residual_covariance_of_mean[0, 0, 1]
        / torch.sqrt(
            result.residual_covariance_of_mean[0, 0, 0]
            * result.residual_covariance_of_mean[0, 1, 1]
        )
    )
    torch.testing.assert_close(
        actual_residual_correlation,
        torch.tensor(expected_residual_correlation, dtype=torch.float64),
        rtol=2.0e-6,
        atol=1.0e-12,
    )


def test_spectral_zero_drift_discards_a_near_collinear_contrast() -> None:
    covariance = torch.tensor(
        [[[1.0, 0.9999], [0.9999, 1.0]]], dtype=torch.float64
    )
    common = spectral_zero_drift(
        torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        covariance,
        correlation_eigenvalue_cutoff=0.01,
    )
    contrast = spectral_zero_drift(
        torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        covariance,
        correlation_eigenvalue_cutoff=0.01,
    )
    assert common.retained_rank.tolist() == [1]
    torch.testing.assert_close(
        common.quadratic,
        torch.tensor([2.0 / 1.9999], dtype=torch.float64),
    )
    torch.testing.assert_close(contrast.quadratic, torch.zeros(1, dtype=torch.float64))


def test_beta_factor_blocks_adjust_only_the_mean_spot_expectation() -> None:
    spot = torch.tensor([-0.2, 0.1, 0.3, -0.1, 0.0], dtype=torch.float64)
    beta = torch.tensor([1.0, -0.5], dtype=torch.float64)
    residual = torch.tensor(
        [
            [0.01, -0.02],
            [0.02, -0.01],
            [0.00, 0.01],
            [0.03, -0.01],
            [0.01, 0.00],
        ],
        dtype=torch.float64,
    )
    raw = residual + spot[:, None] * beta
    statistics = beta_factor_block_zero_drift_statistics(
        raw,
        spot,
        beta,
        correlation_eigenvalue_cutoff=0.01,
    )
    torch.testing.assert_close(statistics.mean, torch.mean(residual, dim=0))
    assert 1 <= int(statistics.retained_rank) <= 2


def test_quadratic_is_invariant_to_instrument_rescaling_and_permutation() -> None:
    mean = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float64)
    covariance = torch.tensor(
        [[[2.0, 0.7, -0.1], [0.7, 1.5, 0.2], [-0.1, 0.2, 0.8]]],
        dtype=torch.float64,
    )
    baseline = quadratic_zero_drift(mean, covariance)
    scale = torch.tensor([100.0, 0.01, 3.0], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    scaled_mean = (mean * scale)[..., permutation]
    scaled_covariance = (
        covariance * scale[:, None] * scale[None, :]
    )[:, permutation][:, :, permutation]
    transformed = quadratic_zero_drift(scaled_mean, scaled_covariance)
    torch.testing.assert_close(
        transformed.quadratic,
        baseline.quadratic,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_rank_deficient_block_covariance_is_finite_for_eleven_moments() -> None:
    generator = torch.Generator().manual_seed(942)
    base = torch.randn((10, 2, 1), generator=generator, dtype=torch.float64)
    duplicated = base.expand(-1, -1, 11).clone()
    duplicated[..., 0] += 0.05
    statistics = block_zero_drift_statistics(duplicated)
    assert statistics.quadratic.shape == (2,)
    assert bool(torch.all(torch.isfinite(statistics.precision)))
    assert bool(torch.all(torch.isfinite(statistics.quadratic)))


def test_cancellation_damaged_covariance_is_projected_to_positive_definite() -> None:
    mean = torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float64)
    covariance = torch.tensor(
        [[[1.0e-12, 2.0e-6, -3.0e-6],
          [2.0e-6, -1.0e-14, 4.0e-6],
          [-3.0e-6, 4.0e-6, 2.0e-12]]],
        dtype=torch.float64,
    )
    statistics = quadratic_zero_drift(mean, covariance)
    assert bool(torch.all(torch.linalg.eigvalsh(statistics.covariance_of_mean) > 0.0))
    assert bool(torch.all(torch.isfinite(statistics.precision)))


def test_detached_precision_has_the_analytic_mean_gradient() -> None:
    mean = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    covariance = torch.tensor(
        [[[1.5, 0.3], [0.3, 0.8]]], dtype=torch.float64
    )
    statistics = quadratic_zero_drift(mean, covariance, detach_covariance=True)
    statistics.quadratic.sum().backward()
    expected = 2.0 * torch.einsum(
        "...ij,...j->...i", statistics.precision, mean.detach()
    )
    torch.testing.assert_close(mean.grad, expected, rtol=1.0e-12, atol=1.0e-12)
    assert not statistics.precision.requires_grad


def test_sign_flip_bootstrap_is_reproducible_and_preserves_shapes() -> None:
    generator = torch.Generator().manual_seed(943)
    blocks = torch.randn((64, 4, 3), generator=generator, dtype=torch.float64)
    statistics = block_zero_drift_statistics(blocks)
    first = sign_flip_zero_drift_bootstrap(
        blocks,
        precision=statistics.precision,
        covariance_of_mean=statistics.covariance_of_mean,
        samples=257,
        seed=944,
        chunk_size=31,
    )
    second = sign_flip_zero_drift_bootstrap(
        blocks,
        precision=statistics.precision,
        covariance_of_mean=statistics.covariance_of_mean,
        samples=257,
        seed=944,
        chunk_size=128,
    )
    assert set(first) == {
        "coordinate_maximum",
        "dense_sum",
        "state_maximum",
        "state_tail_mean",
    }
    for key in first:
        assert first[key].shape == (257,)
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=1.0e-12)


def test_zero_mean_has_zero_quadratic_and_gradient() -> None:
    mean = torch.zeros((3, 11), dtype=torch.float64, requires_grad=True)
    covariance = torch.eye(11, dtype=torch.float64).expand(3, -1, -1)
    statistics = quadratic_zero_drift(mean, covariance)
    statistics.quadratic.sum().backward()
    torch.testing.assert_close(statistics.quadratic, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(mean.grad, torch.zeros_like(mean))


def test_bootstrap_respects_beta_from_separate_calibration_stream() -> None:
    values = {
        "path_sums_by_replication": torch.tensor([[[3.0]], [[5.0]]]),
        "spot_sums_by_replication": torch.tensor([[1.0], [3.0]]),
        "spot_square_sums_by_replication": torch.tensor([[2.0], [10.0]]),
        "product_sums_by_replication": torch.tensor([[[7.0]], [[17.0]]]),
        "paths_per_replication": torch.tensor([2.0, 2.0]),
        "beta": torch.tensor([[2.0]]),
        "beta_is_fixed": torch.tensor(True),
    }
    raw, adjusted = _bootstrap_means(
        values,
        torch.tensor([[1.0, 1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(raw, torch.tensor([[[2.0]]], dtype=torch.float64))
    torch.testing.assert_close(
        adjusted,
        torch.tensor([[[0.0]]], dtype=torch.float64),
    )


def test_frozen_gmm_objective_has_expected_loss_and_mean_gradient() -> None:
    values = {
        "adjusted_replication_means": torch.tensor(
            [[[0.2, -0.1]], [[0.2, -0.1]]], dtype=torch.float64
        ),
        "raw_replication_means": torch.tensor(
            [[[0.1, -0.2]], [[0.3, 0.0]]], dtype=torch.float64
        ),
        "spot_replication_means": torch.tensor(
            [[-0.1], [0.1]], dtype=torch.float64
        ),
        "beta": torch.zeros((1, 2), dtype=torch.float64),
    }
    loss, gradients, metrics = _global_objective(
        [values],
        z_target=100.0,
        inner_tail_fraction=0.5,
        outer_tail_fraction=1.0,
        inner_tail_weight=0.5,
        dtype=torch.float64,
        gmm_precision_by_batch=[torch.eye(2, dtype=torch.float64)[None]],
        gmm_weight=1.0,
    )
    torch.testing.assert_close(loss, torch.tensor(0.025, dtype=torch.float64))
    torch.testing.assert_close(
        gradients[0], torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    )
    assert abs(metrics["gmm_loss"] - 0.025) < 1.0e-14


def test_conditioned_precision_retains_rank_and_is_scale_invariant() -> None:
    covariance = torch.tensor(
        [[[4.0, 2.0, 2.0], [2.0, 1.0, 1.0], [2.0, 1.0, 1.0]]],
        dtype=torch.float64,
    )
    result = correlation_conditioned_precision(
        covariance,
        diagonal_shrinkage=0.0,
        maximum_condition_number=10.0,
    )
    assert bool(torch.all(torch.isfinite(result.precision)))
    assert bool(torch.all(torch.linalg.eigvalsh(result.precision) > 0.0))
    assert float(result.condition_number) <= 10.0 + 1.0e-12

    mean = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    baseline = torch.einsum("si,sij,sj->s", mean, result.precision, mean)
    scale = torch.tensor([100.0, 0.01, 3.0], dtype=torch.float64)
    transformed = correlation_conditioned_precision(
        covariance * scale[:, None] * scale[None, :],
        diagonal_shrinkage=0.0,
        maximum_condition_number=10.0,
    )
    transformed_quadratic = torch.einsum(
        "si,sij,sj->s", mean * scale, transformed.precision, mean * scale
    )
    torch.testing.assert_close(
        transformed_quadratic,
        baseline,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_cross_gmm_top_tail_loss_and_gradients_match_algebra() -> None:
    values_a = {
        "adjusted_replication_means": torch.tensor(
            [[[0.1, -0.2]], [[0.3, 0.0]]], dtype=torch.float64
        ),
        "raw_replication_means": torch.tensor(
            [[[0.1, -0.2]], [[0.3, 0.0]]], dtype=torch.float64
        ),
    }
    values_b = {
        "adjusted_replication_means": torch.tensor(
            [[[0.3, 0.2]], [[0.5, 0.4]]], dtype=torch.float64
        ),
        "raw_replication_means": torch.tensor(
            [[[0.3, 0.2]], [[0.5, 0.4]]], dtype=torch.float64
        ),
    }
    precision = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))[None]
    loss, gradient_a, gradient_b, metrics = _cross_gmm_top_tail_objective(
        [values_a],
        [values_b],
        precision_by_batch=[precision],
        bulk_weight=2.0,
        tail_weight=3.0,
        tail_fraction=0.5,
        absolute_drift_scale_bps=1_000.0,
        dtype=torch.float64,
    )
    torch.testing.assert_close(loss, torch.tensor(27.13, dtype=torch.float64))
    torch.testing.assert_close(
        gradient_a[0], torch.tensor([[90.8, 0.3]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        gradient_b[0], torch.tensor([[90.4, -0.1]], dtype=torch.float64)
    )
    assert abs(metrics["cross_gmm_loss"] - 0.065) < 1.0e-14
    assert abs(metrics["top_tail_mean_loss"] - 9.0) < 1.0e-14

    step = 1.0e-6
    shifted = {key: value.clone() for key, value in values_a.items()}
    shifted["adjusted_replication_means"][:, 0, 0] += step
    shifted["raw_replication_means"][:, 0, 0] += step
    shifted_loss = _cross_gmm_top_tail_objective(
        [shifted],
        [values_b],
        precision_by_batch=[precision],
        bulk_weight=2.0,
        tail_weight=3.0,
        tail_fraction=0.5,
        absolute_drift_scale_bps=1_000.0,
        dtype=torch.float64,
    )[0]
    finite_difference = (shifted_loss - loss) / step
    torch.testing.assert_close(
        finite_difference,
        gradient_a[0][0, 0],
        rtol=1.0e-5,
        atol=1.0e-4,
    )


def test_cross_gmm_removes_the_one_sample_quadratic_noise_floor() -> None:
    generator = torch.Generator().manual_seed(947)
    states = 20_000
    mean_a = torch.randn((1, states, 1), generator=generator, dtype=torch.float64)
    mean_b = torch.randn((1, states, 1), generator=generator, dtype=torch.float64)
    values_a = {
        "adjusted_replication_means": mean_a,
        "raw_replication_means": mean_a,
    }
    values_b = {
        "adjusted_replication_means": mean_b,
        "raw_replication_means": mean_b,
    }
    precision = torch.ones((states, 1, 1), dtype=torch.float64)
    cross_loss = _cross_gmm_top_tail_objective(
        [values_a],
        [values_b],
        precision_by_batch=[precision],
        bulk_weight=1.0,
        tail_weight=0.0,
        tail_fraction=0.01,
        absolute_drift_scale_bps=1.0,
        dtype=torch.float64,
    )[0]
    one_sample_quadratic = torch.mean(mean_a.square())
    assert abs(float(cross_loss)) < 0.03
    assert float(one_sample_quadratic) > 0.95


def test_development_monitor_runs_only_at_the_configured_interval() -> None:
    assert not _monitoring_is_due(completed_sweeps=1, interval=10)
    assert not _monitoring_is_due(completed_sweeps=9, interval=10)
    assert _monitoring_is_due(completed_sweeps=10, interval=10)
    assert _monitoring_is_due(completed_sweeps=20, interval=10)


def test_development_monitor_requires_joint_and_coordinate_drift_targets() -> None:
    audit = {
        "zero_drift": {"mean_quadratic_per_moment": 1.25},
        "per_state": [
            {"adjusted_mean_bps_spot": [0.20, -0.45]},
            {"adjusted_mean_bps_spot": [0.10, 0.30]},
        ],
    }
    assert _monitoring_zero_drift_pass(
        audit,
        quadratic_per_moment_target=2.0,
        absolute_drift_bps_target=0.50,
    )
    assert not _monitoring_zero_drift_pass(
        audit,
        quadratic_per_moment_target=1.0,
        absolute_drift_bps_target=0.50,
    )
    assert not _monitoring_zero_drift_pass(
        audit,
        quadratic_per_moment_target=2.0,
        absolute_drift_bps_target=0.40,
    )


def test_cyclic_state_minibatches_cover_every_state_once_per_epoch() -> None:
    batches = [
        _cyclic_state_minibatch(
            num_states=12,
            states_per_sweep=4,
            sweep=sweep,
            seed=17,
        )
        for sweep in range(6)
    ]

    for epoch in range(2):
        state_epoch = torch.cat(batches[epoch * 3 : (epoch + 1) * 3])
        torch.testing.assert_close(torch.sort(state_epoch).values, torch.arange(12))
    assert not torch.equal(torch.cat(batches[:3]), torch.cat(batches[3:]))
    torch.testing.assert_close(
        batches[4],
        _cyclic_state_minibatch(
            num_states=12,
            states_per_sweep=4,
            sweep=4,
            seed=17,
        ),
    )
