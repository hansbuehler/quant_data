"""Configuration for the historical SANOS scenario hedge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from cdxcore.err import verify_inp
from cdxcore.version import version


class HistoricalSanosObjective(StrEnum):
    """Supported empirical risk objectives."""

    MEAN_VARIANCE = "mean-variance"
    MEAN_VOLATILITY = "mean-volatility"
    ENTROPY = "entropy"
    POWER_EXPONENTIAL = "power-exponential"


@version("1.7")
@dataclass(frozen=True, kw_only=True)
class HistoricalSanosConfig:
    """Logical settings for rolling empirical SANOS hedging."""

    label: str = "historical-sanos-daily-smoke"
    group: str = "hedging/historical-sanos"
    user_version: str = "0.1"
    rolling_observations: int = 252
    objective: HistoricalSanosObjective = HistoricalSanosObjective.MEAN_VARIANCE
    risk_parameter: float = 10.0
    validation_fraction: float = 0.2
    maximum_iterations: int = 50_000
    gradient_tolerance: float = 1.0e-8
    position_cap: float = 1_000_000.0
    solver_position_cap: float = 1_000_000.0
    cost_smoothing: float = 1.0e-6
    quadratic_ridge: float = 1.0e-6

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "label must be non-empty")
        verify_inp(bool(self.group.strip()), lambda: "group must be non-empty")
        verify_inp(
            self.rolling_observations >= 20,
            lambda: "rolling_observations must be at least twenty",
        )
        verify_inp(
            math.isfinite(self.risk_parameter) and self.risk_parameter > 0.0,
            lambda: "risk_parameter must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.validation_fraction)
            and 0.1 <= self.validation_fraction <= 0.5,
            lambda: "validation_fraction must lie in [0.1,0.5]",
        )
        verify_inp(
            self.maximum_iterations > 0,
            lambda: "maximum_iterations must be positive",
        )
        verify_inp(
            math.isfinite(self.gradient_tolerance) and self.gradient_tolerance > 0.0,
            lambda: "gradient_tolerance must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.solver_position_cap)
            and self.solver_position_cap >= self.position_cap,
            lambda: (
                "solver_position_cap must be finite and at least position_cap"
            ),
        )
        verify_inp(
            math.isfinite(self.cost_smoothing) and self.cost_smoothing > 0.0,
            lambda: "cost_smoothing must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.quadratic_ridge) and self.quadratic_ridge >= 0.0,
            lambda: "quadratic_ridge must be finite and non-negative",
        )
