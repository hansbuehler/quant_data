"""Versioned timing and configuration protocols for hedging experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from cdxcore.err import verify_inp
from cdxcore.version import version


class HedgeSchedule(StrEnum):
    """Supported out-of-sample re-estimation schedules."""

    DAILY = "daily"
    FROZEN_WEEKLY = "frozen-weekly"
    RESTARTED_WEEKLY = "restarted-weekly"


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
    """Logical configuration shared by all benchmark models."""

    label: str
    group: str = "hedging"
    user_version: str = "0.1"
    rolling_window_business_days: int = 252
    initialization_end: date = date(2022, 12, 31)
    evaluation_start: date = date(2023, 1, 1)
    evaluation_end: date = date(2025, 12, 31)
    maximum_option_business_dte: float = 5.0
    nms_range: tuple[float, float] = (-2.0, 2.0)
    spot_cost_bps: float = 1.0
    dlv_factor_rank: int = 16
    greek_factor_rank: int = 8
    brownian_substeps_per_day: int = 8

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "label must be non-empty")
        verify_inp(bool(self.group.strip()), lambda: "group must be non-empty")
        verify_inp(
            self.rolling_window_business_days > 0,
            lambda: "rolling_window_business_days must be positive",
        )
        verify_inp(
            self.initialization_end < self.evaluation_start <= self.evaluation_end,
            lambda: "initialization and evaluation dates are inconsistent",
        )
        verify_inp(
            math.isfinite(self.maximum_option_business_dte)
            and self.maximum_option_business_dte > 0.0,
            lambda: "maximum_option_business_dte must be finite and positive",
        )
        verify_inp(
            len(self.nms_range) == 2
            and all(math.isfinite(value) for value in self.nms_range)
            and self.nms_range[0] < self.nms_range[1],
            lambda: "nms_range must contain increasing finite bounds",
        )
        verify_inp(
            math.isfinite(self.spot_cost_bps) and self.spot_cost_bps >= 0.0,
            lambda: "spot_cost_bps must be finite and non-negative",
        )
        verify_inp(
            self.dlv_factor_rank >= self.greek_factor_rank > 0,
            lambda: "factor ranks must satisfy dlv >= greek > 0",
        )
        verify_inp(
            self.brownian_substeps_per_day > 0,
            lambda: "brownian_substeps_per_day must be positive",
        )


@dataclass(frozen=True, kw_only=True)
class EpisodeSpec:
    """Fully dated information and decision schedule for one episode."""

    schedule: HedgeSchedule
    origin_date: date
    training_dates: tuple[date, ...]
    decision_dates: tuple[date, ...]
    terminal_date: date
    frozen_contract_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.training_dates:
            raise ValueError("training_dates must be non-empty")
        if tuple(sorted(set(self.training_dates))) != self.training_dates:
            raise ValueError("training_dates must be unique and increasing")
        if self.training_dates[-1] > self.origin_date:
            raise ValueError("training data cannot extend beyond the episode origin")
        if not self.decision_dates or self.decision_dates[0] != self.origin_date:
            raise ValueError("decision_dates must begin at the episode origin")
        if tuple(sorted(set(self.decision_dates))) != self.decision_dates:
            raise ValueError("decision_dates must be unique and increasing")
        if self.decision_dates[-1] >= self.terminal_date:
            raise ValueError("every hedge decision must precede terminal settlement")
        if len(self.frozen_contract_ids) != len(set(self.frozen_contract_ids)):
            raise ValueError("frozen_contract_ids must be unique")
        if self.schedule is HedgeSchedule.DAILY:
            if len(self.decision_dates) != 1:
                raise ValueError("daily episodes have exactly one decision date")
            if self.frozen_contract_ids:
                raise ValueError("daily episodes do not carry a frozen weekly universe")
        elif not self.frozen_contract_ids:
            raise ValueError("weekly episodes require a frozen inception universe")

    @property
    def refit_dates(self) -> tuple[date, ...]:
        """Return dates on which model parameters and policy are re-estimated."""
        if self.schedule is HedgeSchedule.RESTARTED_WEEKLY:
            return self.decision_dates
        return (self.origin_date,)
