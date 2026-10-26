"""Expiry-dependent strike geometry for GROMM2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class Gromm2GeometryConfig:
    """Numerical geometry of the six-month GROMM2 state."""

    num_inner_strikes: int = 200
    # The DLV interior is calibrated over every observed OTM quote in this
    # range.  Boundary nodes provide the additional deconvolution buffer.
    fit_nms_range: tuple[float, float] = (-4.0, 3.0)
    grid_nms_range: tuple[float, float] = (-4.0, 3.0)
    # Quotes in the active outer band stabilize the tails without allowing
    # sparse, wide-spread observations to dominate the central surface fit.
    core_fit_nms_range: tuple[float, float] = (-3.5, 2.5)
    outer_fit_weight: float = 0.10
    lower_boundary_multiplier: float = 0.5
    upper_boundary_multiplier: float = 1.5
    forward_volatility_floor: float = 0.01
    backbone_annual_volatility: float = 0.05

    def __post_init__(self) -> None:
        verify_inp(
            self.num_inner_strikes >= 16,
            lambda: "num_inner_strikes must be at least 16",
        )
        verify_inp(
            self.grid_nms_range[0] <= self.fit_nms_range[0] < 0.0,
            lambda: "grid lower NMS must contain the fit range",
        )
        verify_inp(
            0.0 < self.fit_nms_range[1] <= self.grid_nms_range[1],
            lambda: "grid upper NMS must contain the fit range",
        )
        verify_inp(
            self.fit_nms_range[0] <= self.core_fit_nms_range[0] < 0.0
            and 0.0 < self.core_fit_nms_range[1] <= self.fit_nms_range[1],
            lambda: "core fit NMS range must be contained in the active range",
        )
        verify_inp(
            0.0 < self.outer_fit_weight <= 1.0,
            lambda: "outer fit weight must lie in (0, 1]",
        )
        verify_inp(
            0.0 < self.lower_boundary_multiplier < 1.0,
            lambda: "lower_boundary_multiplier must lie in (0, 1)",
        )
        verify_inp(
            self.upper_boundary_multiplier > 1.0,
            lambda: "upper_boundary_multiplier must exceed one",
        )
        verify_inp(
            self.forward_volatility_floor > 0.0,
            lambda: "forward_volatility_floor must be positive",
        )
        verify_inp(
            self.backbone_annual_volatility >= 0.0,
            lambda: "backbone_annual_volatility must be non-negative",
        )

    @property
    def model_business_dtes(self) -> FloatArray:
        """Daily expiries to one month, then five-day expiries to six months."""
        return np.concatenate(
            (
                np.arange(1.0, 26.0, dtype=np.float64),
                np.arange(30.0, 126.0, 5.0, dtype=np.float64),
            )
        )

    @property
    def nms_nodes(self) -> FloatArray:
        """Interior normalized-moneyness nodes including ATM exactly."""
        interval_count = self.num_inner_strikes - 1
        lower_fraction = -self.grid_nms_range[0] / (
            self.grid_nms_range[1] - self.grid_nms_range[0]
        )
        lower_intervals = int(round(lower_fraction * interval_count))
        upper_intervals = interval_count - lower_intervals
        return np.concatenate(
            (
                np.linspace(
                    self.grid_nms_range[0],
                    0.0,
                    lower_intervals + 1,
                    dtype=np.float64,
                ),
                np.linspace(
                    0.0,
                    self.grid_nms_range[1],
                    upper_intervals + 1,
                    dtype=np.float64,
                )[1:],
            )
        )


def _isotonic_nondecreasing(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted least-squares projection onto nondecreasing sequences."""
    levels: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        levels.append(float(value))
        masses.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            combined_mass = masses[-2] + masses[-1]
            combined_level = (
                masses[-2] * levels[-2] + masses[-1] * levels[-1]
            ) / combined_mass
            levels[-2:] = [combined_level]
            masses[-2:] = [combined_mass]
            ends[-2:] = [ends[-1]]
            starts.pop()
    projected = np.empty_like(values)
    for level, start, end in zip(levels, starts, ends, strict=True):
        projected[start:end] = level
    return projected


def project_total_variance(
    expiries_y: NDArray[np.floating],
    total_variances: NDArray[np.floating],
    *,
    forward_volatility_floor: float,
    weights: NDArray[np.floating] | None = None,
) -> FloatArray:
    """Project total variance onto a positive forward-variance floor.

    The constraint is equivalent to requiring ``w(t)-floor**2*t`` to be
    nonnegative and nondecreasing.
    """
    expiries = np.asarray(expiries_y, dtype=np.float64)
    variances = np.asarray(total_variances, dtype=np.float64)
    fit_weights = (
        np.ones_like(expiries)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        expiries.ndim != 1
        or variances.shape != expiries.shape
        or fit_weights.shape != expiries.shape
        or len(expiries) == 0
    ):
        raise ValueError("expiries, variances and weights must be aligned vectors")
    if (
        np.any(~np.isfinite(expiries))
        or np.any(~np.isfinite(variances))
        or np.any(~np.isfinite(fit_weights))
        or np.any(expiries <= 0.0)
        or np.any(np.diff(expiries) <= 0.0)
        or np.any(fit_weights <= 0.0)
        or forward_volatility_floor <= 0.0
    ):
        raise ValueError("invalid total-variance projection inputs")
    excess = variances - forward_volatility_floor**2 * expiries
    projected_excess = np.maximum(
        _isotonic_nondecreasing(excess, fit_weights),
        0.0,
    )
    return projected_excess + forward_volatility_floor**2 * expiries


def expanding_supports(
    projected_total_variances: NDArray[np.floating],
    *,
    config: Gromm2GeometryConfig,
) -> FloatArray:
    """Build expiry-dependent supports with two exterior boundary nodes."""
    variances = np.asarray(projected_total_variances, dtype=np.float64)
    if variances.ndim != 1 or np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("projected_total_variances must be a positive finite vector")
    scales = np.sqrt(variances)
    interior = np.exp(scales[:, None] * config.nms_nodes[None, :])
    supports = np.concatenate(
        (
            config.lower_boundary_multiplier * interior[:, :1],
            interior,
            config.upper_boundary_multiplier * interior[:, -1:],
        ),
        axis=1,
    )
    if np.any(np.diff(supports, axis=1) <= 0.0):
        raise RuntimeError("constructed supports are not strictly increasing")
    if len(supports) > 1 and (
        np.any(supports[1:, 0] > supports[:-1, 0])
        or np.any(supports[1:, -1] < supports[:-1, -1])
    ):
        raise RuntimeError("projected supports do not expand with expiry")
    return supports


def barycentric_transport(source: FloatArray, target: FloatArray) -> FloatArray:
    """Map masses to a containing support while preserving mass and mean."""
    old = np.asarray(source, dtype=np.float64)
    new = np.asarray(target, dtype=np.float64)
    if (
        old.ndim != 1
        or new.ndim != 1
        or len(old) < 2
        or len(new) < 2
        or np.any(np.diff(old) <= 0.0)
        or np.any(np.diff(new) <= 0.0)
    ):
        raise ValueError("source and target must be strictly increasing vectors")
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(new[0]), abs(new[-1]))
    if old[0] < new[0] - tolerance or old[-1] > new[-1] + tolerance:
        raise ValueError("target support does not contain source support")
    old = np.clip(old, new[0], new[-1])
    lower = np.searchsorted(new, old, side="right") - 1
    lower = np.clip(lower, 0, len(new) - 2)
    upper = lower + 1
    interval = new[upper] - new[lower]
    upper_weight = (old - new[lower]) / interval
    exact_last = old == new[-1]
    lower[exact_last] = len(new) - 2
    upper[exact_last] = len(new) - 1
    upper_weight[exact_last] = 1.0
    transport = np.zeros((len(new), len(old)), dtype=np.float64)
    columns = np.arange(len(old))
    transport[lower, columns] += 1.0 - upper_weight
    transport[upper, columns] += upper_weight
    if not np.allclose(np.sum(transport, axis=0), 1.0, rtol=0.0, atol=2.0e-14):
        raise RuntimeError("transport does not preserve mass")
    if not np.allclose(new @ transport, old, rtol=0.0, atol=2.0e-13):
        raise RuntimeError("transport does not preserve the first moment")
    return transport


def transport_tower(supports: NDArray[np.floating]) -> FloatArray:
    """Return consecutive barycentric transports for an expanding tower."""
    support_array = np.asarray(supports, dtype=np.float64)
    if support_array.ndim != 2 or len(support_array) < 1:
        raise ValueError("supports must be a non-empty matrix")
    if len(support_array) == 1:
        return np.empty((0, support_array.shape[1], support_array.shape[1]))
    return np.stack(
        [
            barycentric_transport(support_array[index - 1], support_array[index])
            for index in range(1, len(support_array))
        ]
    )
