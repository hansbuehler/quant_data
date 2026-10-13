from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantdata.hedging.inference import (
    EpisodeInferenceConfig,
    HacKernel,
    hac_long_run_variance,
    hac_standard_error_of_mean,
    irregular_hac_long_run_variance,
    moving_block_bootstrap_mean_interval,
    paired_episode_differences,
    summarize_episode_losses,
)


def test_episode_summary_reports_loss_distribution_and_hac_intervals() -> None:
    losses = np.asarray([-2.0, -1.0, 0.0, 1.0, 4.0])
    summary = summarize_episode_losses(
        losses,
        config=EpisodeInferenceConfig(
            hac_lags=1,
            bootstrap_block_length=2,
            bootstrap_samples=500,
            bootstrap_seed=7,
        ),
    )
    assert summary.observations == 5
    assert summary.mean == pytest.approx(0.4)
    assert summary.median == 0.0
    assert summary.sample_standard_deviation == pytest.approx(
        np.std(losses, ddof=1)
    )
    assert summary.root_mean_square == pytest.approx(
        np.sqrt(np.mean(np.square(losses)))
    )
    assert summary.mean_absolute_loss == pytest.approx(np.mean(np.abs(losses)))
    assert summary.expected_shortfall_95 == 4.0
    assert summary.expected_shortfall_99 == 4.0
    assert summary.inference.hac_95[0] < summary.mean < summary.inference.hac_95[1]
    assert summary.inference.raw_hac_long_run_variance == pytest.approx(
        hac_long_run_variance(losses, lags=1)
    )
    assert not summary.inference.hac_variance_was_floored


def test_hac_zero_lag_matches_population_autocovariance_formula() -> None:
    values = np.asarray([1.0, 2.0, 5.0, 8.0])
    centered = values - np.mean(values)
    expected = np.sqrt((centered @ centered / len(values)) / len(values))
    assert hac_standard_error_of_mean(values, lags=0) == pytest.approx(expected)


def test_positive_serial_dependence_increases_hac_uncertainty() -> None:
    values = np.repeat(np.asarray([-1.0, 1.0]), 20)
    assert hac_standard_error_of_mean(values, lags=4) > hac_standard_error_of_mean(
        values,
        lags=0,
    )


def test_hansen_hodrick_recovers_five_session_overlap_variance() -> None:
    generator = np.random.default_rng(20_261_012)
    daily_innovations = generator.standard_normal(200_004)
    overlapping_losses = np.convolve(
        daily_innovations,
        np.ones(5),
        mode="valid",
    )
    observations = len(overlapping_losses)
    hansen_hodrick = hac_standard_error_of_mean(
        overlapping_losses,
        lags=4,
        kernel=HacKernel.HANSEN_HODRICK,
    )
    bartlett = hac_standard_error_of_mean(
        overlapping_losses,
        lags=4,
        kernel=HacKernel.BARTLETT,
    )
    hansen_hodrick_long_run_variance = hansen_hodrick**2 * observations
    bartlett_long_run_variance = bartlett**2 * observations
    assert hansen_hodrick_long_run_variance == pytest.approx(25.0, rel=0.03)
    assert bartlett_long_run_variance == pytest.approx(17.0, rel=0.03)
    assert hansen_hodrick > 1.15 * bartlett


def test_weekly_inference_defaults_do_not_use_one_overlap_block() -> None:
    config = EpisodeInferenceConfig()
    assert config.hac_kernel is HacKernel.HANSEN_HODRICK
    assert config.hac_lags == 4
    assert config.bootstrap_block_length >= 10 * (config.hac_lags + 1)


def test_episode_summary_can_enforce_chronological_origin_dates() -> None:
    losses = np.asarray([1.0, 2.0, 3.0])
    dates = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    summary = summarize_episode_losses(
        losses,
        episode_dates=dates,
        config=EpisodeInferenceConfig(
            hac_lags=1,
            bootstrap_block_length=2,
            bootstrap_samples=100,
        ),
    )
    assert summary.observations == 3
    with pytest.raises(ValueError, match="unique and chronological"):
        summarize_episode_losses(
            losses,
            episode_dates=(dates[1], dates[0], dates[2]),
            config=EpisodeInferenceConfig(
                hac_lags=1,
                bootstrap_block_length=2,
                bootstrap_samples=100,
            ),
        )


def test_irregular_slice_retains_gaps_in_hac_and_bootstrap_domain() -> None:
    losses = np.asarray([1.0, -1.0])
    summary = summarize_episode_losses(
        losses,
        episode_indices=(0, 10),
        full_episode_count=11,
        config=EpisodeInferenceConfig(
            hac_lags=4,
            bootstrap_block_length=5,
            bootstrap_samples=200,
            bootstrap_seed=17,
        ),
    )

    assert summary.inference.raw_hac_long_run_variance == pytest.approx(1.0)
    assert irregular_hac_long_run_variance(
        losses,
        episode_indices=(0, 10),
        full_episode_count=11,
        lags=4,
    ) == pytest.approx(1.0)
    assert hac_long_run_variance(losses, lags=1) == pytest.approx(0.0)
    assert all(np.isfinite(summary.inference.bootstrap_95))


def test_moving_block_bootstrap_is_deterministic() -> None:
    values = np.linspace(-1.0, 2.0, 30)
    first = moving_block_bootstrap_mean_interval(
        values,
        confidence_level=0.95,
        block_length=5,
        samples=500,
        seed=11,
    )
    second = moving_block_bootstrap_mean_interval(
        values,
        confidence_level=0.95,
        block_length=5,
        samples=500,
        seed=11,
    )
    assert first == second
    assert first[0] < np.mean(values) < first[1]


def test_paired_episode_differences_align_by_identity() -> None:
    differences = paired_episode_differences(
        ("a", "b", "c"),
        np.asarray([1.0, 2.0, 3.0]),
        ("c", "a", "b"),
        np.asarray([0.5, 0.25, 1.5]),
    )
    np.testing.assert_allclose(differences, [0.75, 0.5, 2.5])
    with pytest.raises(ValueError, match="identical episode sets"):
        paired_episode_differences(
            ("a", "b"),
            np.asarray([1.0, 2.0]),
            ("a", "c"),
            np.asarray([1.0, 2.0]),
        )
