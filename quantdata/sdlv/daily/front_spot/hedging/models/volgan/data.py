"""VolGAN training examples derived only from released SANOS states."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
    SanosFixedGridHistory,
    build_sanos_fixed_grid_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class VolganTrainingData:
    """Condition/target rows and their public market dates."""

    condition_dates: tuple[date, ...]
    target_dates: tuple[date, ...]
    conditions: FloatArray
    targets: FloatArray

    def __post_init__(self) -> None:
        count = len(self.condition_dates)
        if len(self.target_dates) != count or any(
            target <= condition
            for condition, target in zip(
                self.condition_dates,
                self.target_dates,
                strict=True,
            )
        ):
            raise ValueError("VolGAN example dates are inconsistent")
        if self.conditions.ndim != 2 or self.targets.ndim != 2:
            raise ValueError("VolGAN conditions and targets must be matrices")
        if len(self.conditions) != count or len(self.targets) != count or count < 2:
            raise ValueError("VolGAN example arrays do not align")
        if not np.all(np.isfinite(self.conditions)) or not np.all(np.isfinite(self.targets)):
            raise ValueError("VolGAN examples must be finite")


@dataclass(frozen=True, kw_only=True)
class VolganLiveCondition:
    """One decision-date condition and its next-session forecast date."""

    as_of_date: date
    forecast_date: date
    values: FloatArray
    trailing_forward_log_returns: FloatArray

    def __post_init__(self) -> None:
        if self.forecast_date <= self.as_of_date:
            raise ValueError("VolGAN forecast date must follow its as-of date")
        if self.values.ndim != 1 or not np.all(np.isfinite(self.values)):
            raise ValueError("VolGAN live condition must be a finite vector")
        if self.trailing_forward_log_returns.shape != (21,) or not np.all(
            np.isfinite(self.trailing_forward_log_returns)
        ):
            raise ValueError(
                "VolGAN live condition requires 21 finite trailing returns"
            )


def build_volgan_training_data(
    panel: PublicHedgingPanel,
    *,
    end_date: date,
    observation_count: int = 252,
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid(),
) -> VolganTrainingData:
    """Build published VolGAN conditioning rows without future information."""
    require_public_hedging_panel(panel)
    history = build_sanos_fixed_grid_history(
        panel,
        end_date=end_date,
        observation_count=observation_count,
        grid=grid,
    )
    return volgan_examples_from_history(history)


def has_exact_one_day_target(panel: PublicHedgingPanel, origin_date: date) -> bool:
    """Check that the latest VolGAN training and live-condition lags are one session."""
    try:
        index = panel.dates.index(origin_date)
    except ValueError:
        return False
    return index >= 2 and all(
        panel.observations.row(offset, named=True)["business_sessions_from_previous_observation"] == 1
        for offset in range(index - 2, index + 1)
    )


def build_volgan_live_condition(
    panel: PublicHedgingPanel,
    *,
    as_of_date: date,
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid(),
) -> VolganLiveCondition:
    """Build the live condition ending exactly at ``as_of_date``."""
    require_public_hedging_panel(panel)
    history = build_sanos_fixed_grid_history(
        panel,
        end_date=as_of_date,
        observation_count=22,
        grid=grid,
    )
    return volgan_live_condition_from_history(
        history,
        as_of_date=as_of_date,
        forecast_date=panel.next_one_session_date(as_of_date),
    )


def volgan_live_condition_from_history(
    history: SanosFixedGridHistory,
    *,
    as_of_date: date,
    forecast_date: date,
) -> VolganLiveCondition:
    """Construct the published condition from a history ending at decision time."""
    if len(history.dates) < 22:
        raise ValueError("VolGAN live conditioning requires at least 22 dates")
    if history.dates[-1] != as_of_date:
        raise ValueError(
            f"VolGAN history ends on {history.dates[-1]}, not as-of date {as_of_date}"
        )
    trailing_sessions = history.sessions_from_previous_observation[-21:]
    if np.any(trailing_sessions <= 0) or np.any(trailing_sessions[-2:] != 1):
        raise ValueError("VolGAN live condition lacks exact recent sessions")
    annualization = np.sqrt(252.0)
    trailing_returns = history.forward_log_returns_into_date[-21:]
    current_surface = history.log_implied_volatilities[-1]
    values = np.concatenate(
        (
            annualization * trailing_returns[[-1, -2]],
            np.asarray(
                [
                    float(
                        np.std(
                            trailing_returns / np.sqrt(trailing_sessions),
                            ddof=0,
                        )
                        * annualization
                    )
                ]
            ),
            current_surface.ravel(),
        )
    )
    return VolganLiveCondition(
        as_of_date=as_of_date,
        forecast_date=forecast_date,
        values=values,
        trailing_forward_log_returns=np.asarray(
            trailing_returns,
            dtype=np.float64,
        ).copy(),
    )


def volgan_examples_from_history(history: SanosFixedGridHistory) -> VolganTrainingData:
    """Create one-session targets with interval-aware conditioning volatility."""
    conditions: list[FloatArray] = []
    targets: list[FloatArray] = []
    condition_dates: list[date] = []
    target_dates: list[date] = []
    annualization = np.sqrt(252.0)
    for target_index in range(22, len(history.dates)):
        condition_index = target_index - 1
        chain_start = condition_index - 20
        trailing_sessions = history.sessions_from_previous_observation[
            chain_start : condition_index + 1
        ]
        if (
            np.any(trailing_sessions <= 0)
            or np.any(trailing_sessions[-2:] != 1)
            or history.sessions_from_previous_observation[target_index] != 1
        ):
            continue
        trailing_returns = history.forward_log_returns_into_date[
            chain_start : condition_index + 1
        ]
        realized_volatility = float(
            np.std(trailing_returns / np.sqrt(trailing_sessions), ddof=0)
            * annualization
        )
        current_surface = history.log_implied_volatilities[condition_index]
        next_surface = history.log_implied_volatilities[target_index]
        conditions.append(
            np.concatenate(
                (
                    annualization
                    * history.forward_log_returns_into_date[
                        [condition_index, condition_index - 1]
                    ],
                    np.asarray([realized_volatility]),
                    current_surface.ravel(),
                )
            )
        )
        targets.append(
            np.concatenate(
                (
                    np.asarray(
                        [
                            annualization
                            * history.forward_log_returns_into_date[target_index]
                        ]
                    ),
                    (next_surface - current_surface).ravel(),
                )
            )
        )
        condition_dates.append(history.dates[condition_index])
        target_dates.append(history.dates[target_index])
    if len(conditions) < 2:
        raise ValueError("VolGAN history contains fewer than two exact-session examples")
    return VolganTrainingData(
        condition_dates=tuple(condition_dates),
        target_dates=tuple(target_dates),
        conditions=np.stack(conditions),
        targets=np.stack(targets),
    )
