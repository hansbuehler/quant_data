from __future__ import annotations

import numpy as np
import pytest

from quantdata.hedging.mean_reduction import (
    MeanReduction,
    MEMMBoundaryError,
    MEMMInfeasibleError,
    adaptive_scenario_count,
    reduce_one_day_means,
)


def test_adaptive_scenario_count_respects_floor_dimension_and_pairing() -> None:
    assert adaptive_scenario_count(140) == 10_000
    assert adaptive_scenario_count(1_200) == 12_000
    assert adaptive_scenario_count(1_201, multiple=1_024) == 12_288


def test_demean_removes_instrument_and_target_means() -> None:
    instruments = np.array([[1.0, 4.0], [2.0, -2.0], [6.0, 1.0]])
    targets = np.array([[3.0, 2.0], [-1.0, 8.0], [7.0, 5.0]])

    result = reduce_one_day_means(
        instruments,
        targets,
        method=MeanReduction.DEMEAN,
    )

    np.testing.assert_allclose(np.mean(result.instrument_returns, axis=0), 0.0)
    np.testing.assert_allclose(np.mean(result.target_returns, axis=0), 0.0)
    np.testing.assert_allclose(result.weights, np.full(3, 1.0 / 3.0))
    np.testing.assert_allclose(result.instrument_mean_removed, [3.0, 1.0])
    assert result.diagnostics.effective_sample_size == pytest.approx(3.0)
    assert result.diagnostics.kl_divergence == 0.0
    assert result.diagnostics.converged


def test_memm_finds_exact_two_scenario_martingale_law() -> None:
    instruments = np.array([[-1.0], [2.0]])
    targets = np.array([[4.0, -2.0], [10.0, 7.0]])

    result = reduce_one_day_means(instruments, targets, method="memm")

    np.testing.assert_allclose(result.weights, [2.0 / 3.0, 1.0 / 3.0], atol=1e-11)
    np.testing.assert_array_equal(result.instrument_returns, instruments)
    np.testing.assert_allclose(result.weights @ result.instrument_returns, 0.0, atol=1e-11)
    np.testing.assert_allclose(result.weights @ result.target_returns, 0.0, atol=1e-11)
    np.testing.assert_allclose(result.target_mean_removed, [6.0, 1.0])
    assert 0.0 < result.diagnostics.effective_sample_size < 2.0
    assert result.diagnostics.kl_divergence > 0.0
    assert result.diagnostics.moment_residual <= 1e-11


def test_memm_rank_reduces_duplicate_instrument_constraints() -> None:
    first = np.array([-2.0, -1.0, 1.0, 3.0])
    instruments = np.column_stack((first, 2.0 * first, -3.0 * first))
    targets = np.arange(8.0).reshape(4, 2)

    result = reduce_one_day_means(instruments, targets, method=MeanReduction.MEMM)

    assert result.diagnostics.constraint_rank == 1
    np.testing.assert_allclose(result.weights @ instruments, 0.0, atol=1e-11)
    np.testing.assert_allclose(result.weights @ result.target_returns, 0.0, atol=1e-11)


def test_memm_zero_constraints_uses_uniform_law_and_float64() -> None:
    instruments = np.zeros((4, 2), dtype=np.float32)
    targets = np.arange(8, dtype=np.float32).reshape(4, 2)

    result = reduce_one_day_means(instruments, targets, method=MeanReduction.MEMM)

    assert result.instrument_returns.dtype == np.float64
    assert result.target_returns.dtype == np.float64
    assert result.weights.dtype == np.float64
    assert result.diagnostics.constraint_rank == 0
    assert result.diagnostics.iterations == 0
    np.testing.assert_allclose(result.weights, 0.25)
    np.testing.assert_allclose(result.weights @ result.target_returns, 0.0)


def test_memm_rejects_infeasible_positive_returns() -> None:
    instruments = np.array([[1.0], [2.0], [3.0]])
    targets = np.ones((3, 1))

    with pytest.raises(MEMMInfeasibleError, match="no scenario probabilities"):
        reduce_one_day_means(instruments, targets, method=MeanReduction.MEMM)


def test_memm_rejects_boundary_only_solution() -> None:
    instruments = np.array([[0.0], [1.0], [2.0]])
    targets = np.ones((3, 1))

    with pytest.raises(MEMMBoundaryError, match="zero probability"):
        reduce_one_day_means(instruments, targets, method=MeanReduction.MEMM)


@pytest.mark.parametrize(
    ("instruments", "targets", "message"),
    [
        (np.ones((3, 1)), np.ones((2, 1)), "same samples"),
        (np.array([[1.0], [np.nan]]), np.ones((2, 1)), "finite"),
        (np.ones(3), np.ones((3, 1)), "two-dimensional"),
    ],
)
def test_mean_reduction_validates_inputs(
    instruments: np.ndarray,
    targets: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reduce_one_day_means(instruments, targets, method=MeanReduction.DEMEAN)
