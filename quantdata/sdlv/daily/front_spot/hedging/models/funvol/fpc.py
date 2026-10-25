"""Train-only functional Legendre projection followed by surface FPCs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import FunvolConfig

FloatArray = NDArray[np.float64]


def _inverse_softplus(values: FloatArray) -> FloatArray:
    return values + np.log(-np.expm1(-values))


def _functional_design(config: FunvolConfig) -> FloatArray:
    if config.grid.nms_nodes is None:
        strikes = np.log(np.asarray(config.grid.strikes_over_forward, dtype=np.float64))
    else:
        strikes = np.asarray(config.grid.nms_nodes, dtype=np.float64)
    expiries = np.sqrt(np.asarray(config.grid.expiries_y, dtype=np.float64))
    strike_axis = 2.0 * (strikes - strikes[0]) / (strikes[-1] - strikes[0]) - 1.0
    expiry_axis = 2.0 * (expiries - expiries[0]) / (expiries[-1] - expiries[0]) - 1.0
    expiry_mesh, strike_mesh = np.meshgrid(expiry_axis, strike_axis, indexing="ij")
    columns = []
    for total_order in range(config.functional_order + 1):
        for strike_order in range(total_order + 1):
            expiry_order = total_order - strike_order
            strike_polynomial = legval(strike_mesh, [0.0] * strike_order + [1.0])
            expiry_polynomial = legval(expiry_mesh, [0.0] * expiry_order + [1.0])
            columns.append((strike_polynomial * expiry_polynomial).ravel())
    design = np.stack(columns, axis=1)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("functional grid cannot support the selected polynomial order")
    return design


@dataclass(frozen=True, kw_only=True)
class FunctionalPca:
    """FPCs of Legendre-smoothed inverse-softplus IV surfaces."""

    mean: FloatArray
    components: FloatArray
    explained_variance_ratio: FloatArray
    design: FloatArray
    surface_shape: tuple[int, int]

    def encode(self, implied_volatilities: FloatArray) -> FloatArray:
        """Project positive IV surfaces to fitted functional PC scores."""
        surfaces = np.asarray(implied_volatilities, dtype=np.float64)
        if surfaces.shape[-2:] != self.surface_shape or np.any(~np.isfinite(surfaces)):
            raise ValueError("FuNVol IV surface has incompatible geometry or nonfinite values")
        if np.any(surfaces <= 0.0):
            raise ValueError("FuNVol IV surface must be positive")
        transformed = _inverse_softplus(surfaces).reshape(-1, self.mean.size)
        coefficient = np.linalg.lstsq(self.design, transformed.T, rcond=None)[0]
        smoothed = coefficient.T @ self.design.T
        scores = (smoothed - self.mean) @ self.components.T
        return scores.reshape(surfaces.shape[:-2] + (self.components.shape[0],))

    def decode(self, scores: FloatArray) -> FloatArray:
        """Reconstruct positive IV surfaces from functional PC scores."""
        coordinates = np.asarray(scores, dtype=np.float64)
        if coordinates.shape[-1] != self.components.shape[0]:
            raise ValueError("FuNVol score dimension is inconsistent")
        transformed = coordinates @ self.components + self.mean
        volatilities = np.logaddexp(0.0, transformed)
        return volatilities.reshape(coordinates.shape[:-1] + self.surface_shape)


def fit_functional_pca(
    implied_volatilities: FloatArray,
    *,
    config: FunvolConfig,
) -> FunctionalPca:
    """Fit smooth functional PCs using only supplied training-date surfaces."""
    surfaces = np.asarray(implied_volatilities, dtype=np.float64)
    expected_shape = (len(config.grid.expiries_y), config.grid.strike_count)
    if surfaces.ndim != 3 or surfaces.shape[1:] != expected_shape:
        raise ValueError("FuNVol training surfaces have incompatible geometry")
    if len(surfaces) < config.factor_count + 2 or np.any(~np.isfinite(surfaces)):
        raise ValueError("FuNVol needs enough finite training surfaces")
    if np.any(surfaces <= 0.0):
        raise ValueError("FuNVol training implied volatilities must be positive")
    design = _functional_design(config)
    transformed = _inverse_softplus(surfaces).reshape(len(surfaces), -1)
    coefficient = np.linalg.lstsq(design, transformed.T, rcond=None)[0]
    smoothed = coefficient.T @ design.T
    mean = np.mean(smoothed, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(smoothed - mean, full_matrices=False)
    components = right_vectors[: config.factor_count].copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0.0:
            component *= -1.0
    variance = singular_values**2
    total_variance = float(np.sum(variance))
    ratio = (
        variance[: config.factor_count] / total_variance
        if total_variance > 0.0
        else np.zeros(config.factor_count)
    )
    return FunctionalPca(
        mean=mean,
        components=components,
        explained_variance_ratio=ratio,
        design=design,
        surface_shape=expected_shape,
    )
