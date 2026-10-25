"""Train-only functional coordinates for a reduced VolGAN surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class VolganSurfacePca:
    """Separate PCA bases for log-IV levels and one-session log-IV changes."""

    level_mean: FloatArray
    level_components: FloatArray
    increment_mean: FloatArray
    increment_components: FloatArray
    level_explained_fraction: float
    increment_explained_fraction: float

    @classmethod
    def fit(
        cls,
        *,
        conditions: FloatArray,
        targets: FloatArray,
        train_count: int,
        rank: int,
    ) -> VolganSurfacePca:
        """Fit each basis only on the leading chronological training block."""
        if train_count < 2 or rank < 1 or rank > min(train_count - 1, conditions.shape[1] - 3):
            raise ValueError("VolGAN PCA rank exceeds the available training variation")
        level_mean, level_components, level_fraction = _fit_basis(
            conditions[:train_count, 3:], rank=rank
        )
        increment_mean, increment_components, increment_fraction = _fit_basis(
            targets[:train_count, 1:], rank=rank
        )
        return cls(
            level_mean=level_mean,
            level_components=level_components,
            increment_mean=increment_mean,
            increment_components=increment_components,
            level_explained_fraction=level_fraction,
            increment_explained_fraction=increment_fraction,
        )

    def encode_conditions(self, conditions: FloatArray) -> FloatArray:
        return np.concatenate(
            (conditions[..., :3], (conditions[..., 3:] - self.level_mean) @ self.level_components.T),
            axis=-1,
        )

    def encode_targets(self, targets: FloatArray) -> FloatArray:
        return np.concatenate(
            (targets[..., :1], (targets[..., 1:] - self.increment_mean) @ self.increment_components.T),
            axis=-1,
        )

    @classmethod
    def from_frozen_basis(
        cls,
        *,
        conditions: FloatArray,
        targets: FloatArray,
        train_count: int,
        level_mean: FloatArray,
        level_components: FloatArray,
        increment_mean: FloatArray,
        increment_components: FloatArray,
    ) -> VolganSurfacePca:
        """Reuse an earlier training-only basis without a rolling-coordinate change."""
        return cls(
            level_mean=level_mean.copy(),
            level_components=level_components.copy(),
            increment_mean=increment_mean.copy(),
            increment_components=increment_components.copy(),
            level_explained_fraction=_explained_fraction(
                conditions[:train_count, 3:], level_mean, level_components
            ),
            increment_explained_fraction=_explained_fraction(
                targets[:train_count, 1:], increment_mean, increment_components
            ),
        )


def _fit_basis(values: FloatArray, *, rank: int) -> tuple[FloatArray, FloatArray, float]:
    mean = np.mean(values, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(values - mean, full_matrices=False)
    components = right_vectors[:rank].copy()
    signs = np.sign(components[np.arange(rank), np.argmax(np.abs(components), axis=1)])
    components *= signs[:, None]
    total = float(np.sum(singular_values**2))
    explained = float(np.sum(singular_values[:rank] ** 2) / total) if total > 0 else 1.0
    return mean, components, explained


def _explained_fraction(
    values: FloatArray, mean: FloatArray, components: FloatArray
) -> float:
    centered = values - mean
    projected = (centered @ components.T) @ components
    total = float(np.sum(centered**2))
    return 1.0 - float(np.sum((centered - projected) ** 2) / total) if total > 0 else 1.0
