"""Deterministic SANOS-DLV prices for the weekly benchmark targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.data.pricing import (
    dlv_initial_density,
    dlv_transition_matrices,
    price_dlv_digitals,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WEEKLY_HORIZON,
    WeeklyClaimDefinition,
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicMarketDay
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class WeeklySanosLvPrices:
    """Weekly target prices in discounted origin-PV units."""

    targets: tuple[WeeklyTarget, ...]
    prices: FloatArray
    background_grid_size: int
    background_stddev_width: float
    transition_mass_error: float
    transition_martingale_error: float

    def __post_init__(self) -> None:
        if self.prices.shape != (len(self.targets),):
            raise ValueError("weekly target prices do not align with targets")
        if not np.all(np.isfinite(self.prices) & (self.prices >= 0.0)):
            raise ValueError("weekly target prices must be finite and non-negative")
        if self.background_grid_size < 257 or self.background_grid_size % 2 != 1:
            raise ValueError("background_grid_size must be odd and at least 257")
        if (
            not math.isfinite(self.background_stddev_width)
            or self.background_stddev_width < 6.0
        ):
            raise ValueError("background_stddev_width must be finite and at least six")
        if not (
            math.isfinite(self.transition_mass_error)
            and self.transition_mass_error >= 0.0
            and math.isfinite(self.transition_martingale_error)
            and self.transition_martingale_error >= 0.0
        ):
            raise ValueError("weekly transition errors must be finite and non-negative")

    def by_label(self) -> dict[str, float]:
        """Return prices keyed by the stable weekly target labels."""
        return {
            target.label: float(price)
            for target, price in zip(self.targets, self.prices, strict=True)
        }


def build_common_weekly_claims(
    day: PublicMarketDay,
    *,
    config: SanosLvConfig = SanosLvConfig(),
) -> tuple[WeeklyClaimDefinition, ...]:
    """Return all weekly claims with the benchmark's common inception prices.

    European digitals retain the direct SANOS interpolation as their common
    market mark. Realized variance and barriers use the path-dependent
    SANOS-DLV reference price because a terminal-density interpolation alone
    cannot value them.
    """
    targets = weekly_targets()
    definitions = tuple(build_weekly_claim(day, target=target) for target in targets)
    sanos_lv_prices = price_weekly_sanos_lv_targets(
        day,
        targets=targets,
        config=config,
    ).by_label()
    return tuple(
        definition
        if definition.market_price is not None
        else replace(
            definition,
            market_price=sanos_lv_prices[definition.target.label],
        )
        for definition in definitions
    )


def price_weekly_sanos_lv_targets(
    day: PublicMarketDay,
    *,
    targets: tuple[WeeklyTarget, ...] | None = None,
    config: SanosLvConfig = SanosLvConfig(),
) -> WeeklySanosLvPrices:
    """Price inception-forward-normalized claims under discrete local volatility."""
    selected_targets = weekly_targets() if targets is None else targets
    if not selected_targets or len({target.label for target in selected_targets}) != len(
        selected_targets
    ):
        raise ValueError("weekly targets must be non-empty with unique labels")
    if len(day.dlv_model_expiries_y) < WEEKLY_HORIZON:
        raise ValueError("DLV model does not cover the five-session weekly horizon")
    expected_expiries = np.arange(1, WEEKLY_HORIZON + 1) / DAYS_PER_YEAR
    if not np.allclose(
        day.dlv_model_expiries_y[:WEEKLY_HORIZON],
        expected_expiries,
        rtol=0.0,
        atol=2.0e-6 / DAYS_PER_YEAR,
    ):
        raise ValueError("weekly SANOS-LV pricing requires daily DLV expiry nodes")

    support = day.dlv_support_strikes
    initial_density = dlv_initial_density(support)
    transitions = dlv_transition_matrices(
        support_strikes=support,
        raw_log_excess_dlvsq=day.dlv_raw_log_excess_dlvsq,
        model_expiries_y=day.dlv_model_expiries_y,
        minimum_dlvsq=day.minimum_dlvsq,
        maximum_dlvsq=day.maximum_dlvsq,
        step_count=WEEKLY_HORIZON,
    )
    intervals = np.diff(
        np.concatenate(([0.0], day.dlv_model_expiries_y[:WEEKLY_HORIZON]))
    )
    discount_factor = day.discount_at_business_dte(WEEKLY_HORIZON)
    mass_error = float(
        np.max(np.abs(np.sum(transitions, axis=1) - 1.0))
    )
    martingale_error = float(
        np.max(
            np.abs(
                np.einsum("i,tij->tj", support, transitions)
                - support[None, :]
            )
        )
    )

    digital_targets = tuple(
        target
        for target in selected_targets
        if target.kind is WeeklyTargetKind.DIGITAL
    )
    digital_prices: dict[str, float] = {}
    if digital_targets:
        claims = tuple(build_weekly_claim(day, target=target) for target in digital_targets)
        normalized_prices = price_dlv_digitals(
            query_expiries_y=np.full(
                len(digital_targets), WEEKLY_HORIZON / DAYS_PER_YEAR
            ),
            query_strikes=np.asarray(
                [float(claim.claim.strike) for claim in claims]
            ),
            is_call=np.asarray(
                [target.right is OptionRight.CALL for target in digital_targets],
                dtype=bool,
            ),
            support_strikes=support,
            raw_log_excess_dlvsq=day.dlv_raw_log_excess_dlvsq,
            model_expiries_y=day.dlv_model_expiries_y,
            minimum_dlvsq=day.minimum_dlvsq,
            maximum_dlvsq=day.maximum_dlvsq,
            backbone_annual_volatility=day.backbone_annual_volatility,
        )
        digital_prices = {
            target.label: float(discount_factor * price)
            for target, price in zip(
                digital_targets, normalized_prices, strict=True
            )
        }

    variance_price = _price_realized_variance(
        support=support,
        initial_density=initial_density,
        transitions=transitions,
        intervals=intervals,
        background_annual_volatility=day.backbone_annual_volatility,
    )
    down_price, up_price = _price_barriers(
        support=support,
        initial_density=initial_density,
        transitions=transitions,
        intervals=intervals,
        background_annual_volatility=day.backbone_annual_volatility,
        grid_size=config.weekly_background_grid_size,
        stddev_width=config.weekly_background_stddev_width,
    )
    path_prices = {
        WeeklyTargetKind.REALIZED_VARIANCE: discount_factor * variance_price,
        WeeklyTargetKind.DOWN_AND_OUT_PUT: discount_factor * down_price,
        WeeklyTargetKind.UP_AND_OUT_CALL: discount_factor * up_price,
    }
    prices = np.asarray(
        [
            digital_prices[target.label]
            if target.kind is WeeklyTargetKind.DIGITAL
            else path_prices[target.kind]
            for target in selected_targets
        ],
        dtype=np.float64,
    )
    return WeeklySanosLvPrices(
        targets=selected_targets,
        prices=prices,
        background_grid_size=config.weekly_background_grid_size,
        background_stddev_width=config.weekly_background_stddev_width,
        transition_mass_error=mass_error,
        transition_martingale_error=martingale_error,
    )


def _price_realized_variance(
    *,
    support: FloatArray,
    initial_density: FloatArray,
    transitions: FloatArray,
    intervals: FloatArray,
    background_annual_volatility: float,
) -> float:
    density = initial_density.copy()
    ratio = support[:, None] / support[None, :]
    expected_sum = 0.0
    for transition, interval in zip(transitions, intervals, strict=True):
        background_second_moment = math.exp(
            background_annual_volatility**2 * float(interval)
        )
        squared_return = (
            ratio**2 * background_second_moment - 2.0 * ratio + 1.0
        )
        expected_sum += float(
            np.sum(transition * squared_return * density[None, :])
        )
        density = transition @ density
    return expected_sum / len(transitions)


def _price_barriers(
    *,
    support: FloatArray,
    initial_density: FloatArray,
    transitions: FloatArray,
    intervals: FloatArray,
    background_annual_volatility: float,
    grid_size: int,
    stddev_width: float,
) -> tuple[float, float]:
    if background_annual_volatility == 0.0:
        return _price_barriers_without_background(
            support=support,
            initial_density=initial_density,
            transitions=transitions,
        )
    total_time = float(np.sum(intervals))
    half_width = (
        stddev_width * background_annual_volatility * math.sqrt(total_time)
        + 0.5 * background_annual_volatility**2 * total_time
    )
    log_background = np.linspace(-half_width, half_width, grid_size)
    grid_step = float(log_background[1] - log_background[0])
    mass = np.zeros((2, len(support), grid_size), dtype=np.float64)
    mass[:, :, grid_size // 2] = initial_density
    for transition, interval in zip(transitions, intervals, strict=True):
        advanced = np.einsum(
            "ij,bjg->big",
            transition,
            mass,
            optimize=True,
        )
        diffused = gaussian_filter1d(
            advanced,
            sigma=(
                background_annual_volatility
                * math.sqrt(float(interval))
                / grid_step
            ),
            axis=-1,
            mode="constant",
            cval=0.0,
            truncate=stddev_width,
        )
        if float(np.sum(advanced) - np.sum(diffused)) > 1.0e-12:
            raise FloatingPointError("weekly background grid lost diffusion tail mass")
        mass = _shift_uniform_mass(
            diffused,
            shift_cells=(
                -0.5
                * background_annual_volatility**2
                * float(interval)
                / grid_step
            ),
        )
        if float(np.sum(diffused) - np.sum(mass)) > 1.0e-12:
            raise FloatingPointError("weekly background grid lost drift tail mass")
        spot = support[:, None] * np.exp(log_background)[None, :]
        mass[0] *= spot > 0.95
        mass[1] *= spot < 1.05
    terminal_spot = support[:, None] * np.exp(log_background)[None, :]
    down_payoff = np.maximum(1.0 - terminal_spot, 0.0)
    up_payoff = np.maximum(terminal_spot - 1.0, 0.0)
    return (
        float(np.sum(mass[0] * down_payoff)),
        float(np.sum(mass[1] * up_payoff)),
    )


def _shift_uniform_mass(mass: FloatArray, *, shift_cells: float) -> FloatArray:
    """Translate gridded probability mass with conservative linear deposition."""
    lower_shift = math.floor(shift_cells)
    upper_weight = shift_cells - lower_shift
    shifted = np.zeros_like(mass)
    _add_integer_shift(
        shifted,
        mass,
        offset=lower_shift,
        weight=1.0 - upper_weight,
    )
    _add_integer_shift(
        shifted,
        mass,
        offset=lower_shift + 1,
        weight=upper_weight,
    )
    return shifted


def _add_integer_shift(
    target: FloatArray,
    source: FloatArray,
    *,
    offset: int,
    weight: float,
) -> None:
    if weight == 0.0 or abs(offset) >= source.shape[-1]:
        return
    if offset >= 0:
        target[..., offset:] += weight * source[..., : source.shape[-1] - offset]
    else:
        target[..., :offset] += weight * source[..., -offset:]


def _price_barriers_without_background(
    *,
    support: FloatArray,
    initial_density: FloatArray,
    transitions: FloatArray,
) -> tuple[float, float]:
    mass = np.broadcast_to(initial_density, (2, len(support))).copy()
    for transition in transitions:
        mass = mass @ transition.T
        mass[0] *= support > 0.95
        mass[1] *= support < 1.05
    return (
        float(mass[0] @ np.maximum(1.0 - support, 0.0)),
        float(mass[1] @ np.maximum(support - 1.0, 0.0)),
    )
