"""Versioned configuration for the Tangent Levy v2 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.torch.ffw import FeedForwardConfig


def _daily_maturity_grid() -> tuple[float, ...]:
    return tuple(np.arange(0, 71, dtype=np.float64) / DAYS_PER_YEAR)


def _jump_grid() -> tuple[float, ...]:
    return tuple(np.linspace(-0.75, 0.75, 301).tolist())


@version("2.0")
@dataclass(frozen=True, kw_only=True)
class TangentLevyV2MarketConfig:
    """Exact quote population documented by ``dy21.tex``."""

    ticker: str = "SPX"
    start: date = date(2022, 1, 3)
    end: date = date(2025, 8, 29)
    smoothness: float = 0.25
    minimum_business_dte: int = 1
    maximum_business_dte: int = 70
    excluded_business_dtes: tuple[float, ...] = (0.2,)
    nms_range: tuple[float, float] = (-3.0, 2.0)
    expected_dates: int = 891
    expected_quotes: int = 3_566_258

    def __post_init__(self) -> None:
        verify_inp(self.ticker == "SPX", lambda: "v2 is specified for SPX")
        verify_inp(
            self.start <= self.end,
            lambda: f"invalid market interval {self.start} through {self.end}",
        )
        verify_inp(
            math.isfinite(self.smoothness) and self.smoothness > 0.01,
            lambda: f"invalid smoothness {self.smoothness!r}",
        )
        verify_inp(
            0 < self.minimum_business_dte <= self.maximum_business_dte,
            lambda: "business-DTE bounds must be positive and ordered",
        )
        verify_inp(
            len(self.nms_range) == 2
            and all(math.isfinite(value) for value in self.nms_range)
            and self.nms_range[0] < self.nms_range[1],
            lambda: f"invalid NMS interval {self.nms_range!r}",
        )
        verify_inp(
            self.expected_dates > 0 and self.expected_quotes > 0,
            lambda: "expected population sizes must be positive",
        )


@version("2.0")
@dataclass(frozen=True, kw_only=True)
class TangentLevyV2DynamicsConfig:
    """Common diffusion and measure-change geometry for v2."""

    maturity_grid: tuple[float, ...] = field(default_factory=_daily_maturity_grid)
    jump_grid: tuple[float, ...] = field(default_factory=_jump_grid)
    factor_rank: int = 16
    covariance_shrinkage: float = 0.05
    physical_ridge: float = 1.0e-4
    training_end: date = date(2024, 12, 31)
    reconstruction_relative_tolerance: float = 0.25
    reconstruction_repricing_tolerance: float = 2.0

    def __post_init__(self) -> None:
        tau = np.asarray(self.maturity_grid, dtype=np.float64)
        jump = np.asarray(self.jump_grid, dtype=np.float64)
        verify_inp(
            tau.ndim == 1
            and tau.size == 71
            and tau[0] == 0.0
            and np.all(np.diff(tau) > 0.0),
            lambda: "v2 maturity grid must contain daily nodes 0 through 70 BD",
        )
        verify_inp(
            jump.ndim == 1
            and jump.size >= 101
            and jump.size % 2 == 1
            and np.all(np.diff(jump) > 0.0)
            and np.allclose(jump, -jump[::-1]),
            lambda: "jump grid must be odd, symmetric, and strictly increasing",
        )
        verify_inp(self.factor_rank > 0, lambda: "factor rank must be positive")
        verify_inp(
            0.0 <= self.covariance_shrinkage < 1.0,
            lambda: "covariance shrinkage must lie in [0,1)",
        )
        verify_inp(
            math.isfinite(self.physical_ridge) and self.physical_ridge >= 0.0,
            lambda: "physical ridge must be finite and non-negative",
        )
        verify_inp(
            self.reconstruction_relative_tolerance >= 0.0
            and self.reconstruction_repricing_tolerance >= 0.0,
            lambda: "reconstruction tolerances must be non-negative",
        )


@version("2.0")
@dataclass(frozen=True, kw_only=True)
class TangentLevyV2AuditConfig:
    """Restricted-universe diagnostics from ``paper/main2.tex``."""

    path_count: int = 100_000
    horizon_days: int = 20
    option_expiry_days: tuple[int, ...] = (5, 10, 15, 20)
    options_per_expiry: int = 11
    steps_per_day: int = 8
    seed: int = 20_261_009
    policy_network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=32,
    )
    policy_epochs: int = 200
    policy_batch_size: int = 256
    policy_learning_rate: float = 1.0e-3
    lambda_second_moment: float = 1.0

    def __post_init__(self) -> None:
        verify_inp(
            self.path_count >= 4 and self.path_count % 4 == 0,
            lambda: "path count must be divisible by four for pair-preserving folds",
        )
        verify_inp(self.horizon_days > 0, lambda: "horizon must be positive")
        verify_inp(
            tuple(sorted(set(self.option_expiry_days))) == self.option_expiry_days
            and self.option_expiry_days[-1] <= self.horizon_days,
            lambda: "option expiries must be unique, ordered, and within the horizon",
        )
        verify_inp(
            self.options_per_expiry > 0 and self.options_per_expiry % 2 == 1,
            lambda: "options per expiry must be a positive odd integer",
        )
        verify_inp(self.steps_per_day > 0, lambda: "steps per day must be positive")
        verify_inp(self.policy_epochs > 0, lambda: "policy epochs must be positive")
        verify_inp(self.policy_batch_size > 0, lambda: "policy batch size must be positive")
        verify_inp(
            math.isfinite(self.policy_learning_rate) and self.policy_learning_rate > 0.0,
            lambda: "policy learning rate must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.lambda_second_moment)
            and self.lambda_second_moment >= 0.0,
            lambda: "second-moment penalty must be finite and non-negative",
        )

    @property
    def instrument_count(self) -> int:
        """Spot plus the fixed-cash-strike option universe."""

        return 1 + len(self.option_expiry_days) * self.options_per_expiry


@version(
    "2.0",
    dependencies=[
        TangentLevyV2MarketConfig,
        TangentLevyV2DynamicsConfig,
        TangentLevyV2AuditConfig,
    ],
)
@dataclass(frozen=True, kw_only=True)
class TangentLevyV2Config:
    """Logical identity of a complete v2 training and audit run."""

    label: str
    group: str = "tangent-levy/v2"
    user_version: str = "2.0"
    market: TangentLevyV2MarketConfig = field(default_factory=TangentLevyV2MarketConfig)
    dynamics: TangentLevyV2DynamicsConfig = field(
        default_factory=TangentLevyV2DynamicsConfig
    )
    audit: TangentLevyV2AuditConfig = field(default_factory=TangentLevyV2AuditConfig)

    def __post_init__(self) -> None:
        verify_inp(
            bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
            lambda: f"invalid experiment label {self.label!r}",
        )
        parts = self.group.split("/")
        verify_inp(
            bool(parts)
            and all(part not in {"", ".", ".."} and is_filename(part) for part in parts),
            lambda: f"invalid experiment group {self.group!r}",
        )

    @property
    def filename(self) -> str:
        """Stable filename body for the logical v2 experiment."""

        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)
