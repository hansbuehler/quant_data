"""Held-out multivariate tests for generated surface-path features."""

from __future__ import annotations

import math

import numpy as np


def paired_block_distribution_test(
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    block_length: int = 5,
    permutations: int = 1_999,
    seed: int = 30_817,
    prestandardized: bool = False,
) -> dict[str, object]:
    """Compare paired chronological path features by blockwise label swaps.

    Each observed path and simulated path must share the same conditioning
    date.  Whole adjacent blocks are swapped between the two samples, which
    retains short-range temporal dependence while testing equality of their
    unconditional held-out feature distributions.
    """
    actual = np.asarray(observed, dtype=np.float64)
    model = np.asarray(simulated, dtype=np.float64)
    if (
        actual.ndim != 2
        or model.shape != actual.shape
        or len(actual) < 10
        or np.any(~np.isfinite(actual))
        or np.any(~np.isfinite(model))
    ):
        raise ValueError("observed and simulated features must be aligned finite matrices")
    if block_length <= 0 or permutations <= 0:
        raise ValueError("block_length and permutations must be positive")
    if prestandardized:
        keep = np.ones(actual.shape[1], dtype=np.bool_)
    else:
        pooled = np.vstack((actual, model))
        center = np.mean(pooled, axis=0)
        scale = np.std(pooled, axis=0, ddof=1)
        keep = scale > 64.0 * np.finfo(np.float64).eps
        if not np.any(keep):
            raise ValueError("path features have no non-constant coordinate")
        actual = (actual[:, keep] - center[keep]) / scale[keep]
        model = (model[:, keep] - center[keep]) / scale[keep]
    combined = np.vstack((actual, model))
    squared_distances = _squared_distances(combined, combined)
    upper = squared_distances[np.triu_indices(len(combined), k=1)]
    positive = upper[upper > 0.0]
    bandwidth_squared = float(np.median(positive))
    if not math.isfinite(bandwidth_squared) or bandwidth_squared <= 0.0:
        raise ValueError("could not determine a positive MMD bandwidth")
    kernel = np.exp(-0.5 * squared_distances / bandwidth_squared)
    sample_size = len(actual)
    observed_mmd = _mmd_from_kernel(
        kernel, np.arange(sample_size), np.arange(sample_size, 2 * sample_size)
    )
    observed_energy = _energy_distance(actual, model)

    generator = np.random.default_rng(seed)
    number_blocks = math.ceil(sample_size / min(block_length, sample_size))
    block_ids = np.minimum(
        np.arange(sample_size) // min(block_length, sample_size), number_blocks - 1
    )
    permutation_values = np.empty(permutations, dtype=np.float64)
    base = np.arange(sample_size)
    for replication in range(permutations):
        swap_blocks = generator.integers(0, 2, size=number_blocks).astype(bool)
        swap = swap_blocks[block_ids]
        left = np.where(swap, base + sample_size, base)
        right = np.where(swap, base, base + sample_size)
        permutation_values[replication] = _mmd_from_kernel(kernel, left, right)
    return {
        "pairs": sample_size,
        "features": int(np.sum(keep)),
        "standardization": (
            "supplied pre-standardized features"
            if prestandardized
            else "pooled observed/simulated coordinate mean and standard deviation"
        ),
        "block_length": min(block_length, sample_size),
        "permutations": permutations,
        "standardized_energy_distance": observed_energy,
        "rbf_bandwidth_squared": bandwidth_squared,
        "mmd_squared": observed_mmd,
        "block_swap_p_value": float(
            (1 + np.sum(permutation_values >= observed_mmd)) / (permutations + 1)
        ),
    }


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(result, 0.0)


def _mmd_from_kernel(
    kernel: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    size = len(left)
    if size < 2:
        raise ValueError("MMD requires at least two observations per sample")
    left_kernel = kernel[np.ix_(left, left)]
    right_kernel = kernel[np.ix_(right, right)]
    cross_kernel = kernel[np.ix_(left, right)]
    within_left = (np.sum(left_kernel) - np.trace(left_kernel)) / (size * (size - 1))
    within_right = (np.sum(right_kernel) - np.trace(right_kernel)) / (size * (size - 1))
    return float(within_left + within_right - 2.0 * np.mean(cross_kernel))


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    cross = np.sqrt(_squared_distances(left, right))
    within_left = np.sqrt(_squared_distances(left, left))
    within_right = np.sqrt(_squared_distances(right, right))
    return float(2.0 * np.mean(cross) - np.mean(within_left) - np.mean(within_right))


__all__ = ["paired_block_distribution_test"]
