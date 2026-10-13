"""Resumable production runners for the public hedging benchmark."""

import sys

from quantdata.sdlv.daily.front_spot.hedging.runners.jax_environment import (
    force_cpu_for_prepare_cli,
)

force_cpu_for_prepare_cli(sys.argv)

from quantdata.sdlv.daily.front_spot.hedging.runners.config import (
    GeneratedDailyRunConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DailyBookFunction,
    DailyEvaluationSpec,
    DailyEvaluationSummary,
    eligible_daily_origins,
    read_daily_record,
    run_daily_evaluation,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    DailyLossStatistics,
    DailyResultsConfig,
    DailyResultSlice,
    DailyRunResults,
    initialization_volatility_thresholds,
    summarize_completed_daily_run,
    summarize_daily_results,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyBookFunction,
    WeeklyEvaluationSpec,
    WeeklyEvaluationSummary,
    eligible_weekly_origins,
    read_weekly_record,
    run_weekly_evaluation,
)

__all__ = [
    "DailyBookFunction",
    "DailyEvaluationSpec",
    "DailyEvaluationSummary",
    "DailyLossStatistics",
    "DailyResultSlice",
    "DailyResultsConfig",
    "DailyRunResults",
    "GeneratedDailyRunConfig",
    "WeeklyBookFunction",
    "WeeklyEvaluationSpec",
    "WeeklyEvaluationSummary",
    "eligible_daily_origins",
    "eligible_weekly_origins",
    "initialization_volatility_thresholds",
    "read_daily_record",
    "read_weekly_record",
    "summarize_completed_daily_run",
    "summarize_daily_results",
    "run_daily_evaluation",
    "run_weekly_evaluation",
]
