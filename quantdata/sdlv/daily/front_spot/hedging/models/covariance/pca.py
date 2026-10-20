"""Spot-first factor covariance for the independent historical hedge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SpotResidualPca:
    covariance: FloatArray
    spot_beta: FloatArray
    loadings: FloatArray
    explained_residual_variance: float
    idiosyncratic_variance: FloatArray

    def project_claim_cross_covariance(self, cross_covariance: FloatArray) -> FloatArray:
        """Retain claim exposure to spot and the selected residual factors."""
        cross = np.asarray(cross_covariance, dtype=np.float64)
        if cross.shape != (len(self.spot_beta) + 1,):
            raise ValueError("claim cross-covariance does not match the hedge book")
        spot_cross = cross[0]
        residual_cross = cross[1:] - self.spot_beta * spot_cross
        projected_residual_cross = self.loadings @ (self.loadings.T @ residual_cross)
        return np.concatenate((
            np.asarray([spot_cross]),
            self.spot_beta * spot_cross + projected_residual_cross,
        ))


def spot_residual_pca_covariance(
    covariance: FloatArray,
    *,
    rank: int,
) -> SpotResidualPca:
    """Keep spot exactly, then eight correlated residual-option risk axes.

    The omitted residual covariance is replaced by its diagonal, retaining
    each option's idiosyncratic risk and avoiding a singular hedge problem.
    """
    full = np.asarray(covariance, dtype=np.float64)
    if full.ndim != 2 or full.shape[0] != full.shape[1] or full.shape[0] < 2:
        raise ValueError("covariance must include spot and at least one option")
    option_count = full.shape[0] - 1
    if not 0 < rank < option_count:
        raise ValueError("PCA rank must be below the option count")
    if not np.all(np.isfinite(full)) or not np.allclose(full, full.T, rtol=1e-10, atol=1e-14):
        raise ValueError("covariance must be finite and symmetric")
    spot_variance = float(full[0, 0])
    if spot_variance <= 0.0:
        raise ValueError("spot variance must be positive")
    spot_beta = full[1:, 0] / spot_variance
    residual = full[1:, 1:] - spot_variance * np.outer(spot_beta, spot_beta)
    residual = 0.5 * (residual + residual.T)
    eigenvalues, eigenvectors = np.linalg.eigh(residual)
    if eigenvalues[0] <= 0.0:
        raise ValueError("spot-residual covariance must be positive definite")
    order = np.argsort(eigenvalues)[-rank:][::-1]
    loadings = eigenvectors[:, order]
    retained = (loadings * eigenvalues[order]) @ loadings.T
    omitted_diagonal = np.diag(residual - retained)
    idiosyncratic_variance = np.maximum(
        omitted_diagonal,
        np.maximum(1e-6 * np.diag(residual), 1e-12 * spot_variance),
    )
    projected = np.empty_like(full)
    projected[0, 0] = spot_variance
    projected[0, 1:] = full[0, 1:]
    projected[1:, 0] = full[1:, 0]
    projected[1:, 1:] = (
        spot_variance * np.outer(spot_beta, spot_beta)
        + retained
        + np.diag(idiosyncratic_variance)
    )
    return SpotResidualPca(
        covariance=projected,
        spot_beta=spot_beta,
        loadings=loadings,
        explained_residual_variance=float(np.sum(eigenvalues[order]) / np.trace(residual)),
        idiosyncratic_variance=idiosyncratic_variance,
    )
