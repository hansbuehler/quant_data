"""Configuration for the independent historical covariance hedge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.hedging.mean_reduction import MeanReduction


class CovarianceObjective(StrEnum):
    """Supported robust historical risk objectives."""

    MEAN_VARIANCE = "mean-variance"
    MEAN_VOLATILITY = "mean-volatility"
    ENTROPY = "entropy"
    POWER_EXPONENTIAL = "power-exponential"


@version("3.7")
@dataclass(frozen=True, kw_only=True)
class CovarianceHedgeConfig:
    """Logical and numerical settings for the rolling covariance hedge."""

    label: str = "covariance-daily-smoke"
    group: str = "hedging/covariance"
    user_version: str = "0.3"
    rolling_observations: int = 252
    objective: CovarianceObjective = CovarianceObjective.MEAN_VARIANCE
    risk_parameter: float = 10.0
    cost_multiplier: float = 1.0
    mean_reduction: MeanReduction = MeanReduction.DEMEAN
    mean_z_threshold: float | None = None
    mean_hac_lags: int | None = None
    recession_tolerance: float = 1.0e-8
    position_cap: float = 1_000_000.0
    solver_position_cap: float = 1_000_000.0
    maximum_solver_iterations: int = 10_000
    maximum_reweighting_iterations: int = 100
    action_tolerance: float = 1.0e-8
    minimum_standard_deviation: float = 1.0e-10
    spot_residual_pca_rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean_reduction", MeanReduction(self.mean_reduction))
        verify_inp(bool(self.label.strip()), lambda: "label must be non-empty")
        verify_inp(bool(self.group.strip()), lambda: "group must be non-empty")
        verify_inp(
            self.rolling_observations >= 2,
            lambda: "rolling_observations must be at least two",
        )
        verify_inp(
            math.isfinite(self.risk_parameter) and self.risk_parameter > 0.0,
            lambda: "risk_parameter must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.cost_multiplier) and self.cost_multiplier >= 0.0,
            lambda: "cost_multiplier must be finite and non-negative",
        )
        verify_inp(
            self.mean_z_threshold is None
            or (
                math.isfinite(self.mean_z_threshold)
                and self.mean_z_threshold >= 0.0
            ),
            lambda: "mean_z_threshold must be non-negative or None",
        )
        verify_inp(
            self.mean_hac_lags is None or self.mean_hac_lags >= 0,
            lambda: "mean_hac_lags must be non-negative or None",
        )
        verify_inp(
            math.isfinite(self.recession_tolerance)
            and self.recession_tolerance >= 0.0,
            lambda: "recession_tolerance must be finite and non-negative",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.solver_position_cap)
            and self.solver_position_cap >= self.position_cap,
            lambda: (
                "solver_position_cap must be finite and at least position_cap"
            ),
        )
        verify_inp(
            self.maximum_solver_iterations > 0,
            lambda: "maximum_solver_iterations must be positive",
        )
        verify_inp(
            self.maximum_reweighting_iterations > 0,
            lambda: "maximum_reweighting_iterations must be positive",
        )
        verify_inp(
            math.isfinite(self.action_tolerance) and self.action_tolerance > 0.0,
            lambda: "action_tolerance must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.minimum_standard_deviation)
            and self.minimum_standard_deviation > 0.0,
            lambda: "minimum_standard_deviation must be finite and positive",
        )
        verify_inp(
            self.spot_residual_pca_rank is None
            or self.spot_residual_pca_rank > 0,
            lambda: "spot_residual_pca_rank must be positive or None",
        )
