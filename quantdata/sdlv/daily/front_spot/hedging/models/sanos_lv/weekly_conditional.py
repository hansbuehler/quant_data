"""Conditional SANOS-DLV prices and Greeks for fixed weekly claims."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.claims import (
    BarrierDirection,
    BarrierOptionClaim,
    DigitalClaim,
    RealizedVarianceClaim,
)
from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.data.pricing import (
    dlv_initial_density,
    dlv_transition_matrices,
    price_dlv_digitals,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WEEKLY_HORIZON,
    WeeklyClaimDefinition,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicMarketDay
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorFit,
    raw_dlv_state_with_factor_bump,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import DlvGreeks
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    _shift_uniform_mass,
)

FloatArray = NDArray[np.float64]


def conditional_weekly_claim_greeks(
    day: PublicMarketDay,
    *,
    claim: WeeklyClaimDefinition,
    decision_index: int,
    realized_inception_forward_path: NDArray[np.floating],
    inception_forward_over_episode_spot: NDArray[np.floating],
    spot_over_episode_origin: float,
    factors: RollingDlvFactorFit,
    config: SanosLvConfig = SanosLvConfig(),
) -> DlvGreeks:
    """Return the remaining fixed-claim value, delta, and DLV-factor vegas."""
    realized_path, inception_curve = _validate_conditional_inputs(
        day,
        claim=claim,
        decision_index=decision_index,
        realized_inception_forward_path=realized_inception_forward_path,
        inception_forward_over_episode_spot=inception_forward_over_episode_spot,
        spot_over_episode_origin=spot_over_episode_origin,
    )

    def price(raw_state: FloatArray, *, spot_multiplier: float = 1.0) -> float:
        return _conditional_weekly_claim_price(
            day,
            claim=claim,
            decision_index=decision_index,
            realized_path=realized_path,
            inception_curve=inception_curve,
            spot_over_episode_origin=spot_over_episode_origin,
            raw_state=raw_state,
            spot_multiplier=spot_multiplier,
            config=config,
        )

    base_price = price(day.dlv_raw_log_excess_dlvsq)
    relative_step = config.relative_spot_step
    price_up = price(
        day.dlv_raw_log_excess_dlvsq,
        spot_multiplier=1.0 + relative_step,
    )
    price_down = price(
        day.dlv_raw_log_excess_dlvsq,
        spot_multiplier=1.0 - relative_step,
    )
    delta = (
        (1.0 + relative_step) * price_up
        - (1.0 - relative_step) * price_down
    ) / (2.0 * relative_step)
    factor_vegas = np.empty((1, factors.factor_rank), dtype=np.float64)
    for factor_index in range(factors.factor_rank):
        raw_up = raw_dlv_state_with_factor_bump(
            day,
            factors,
            factor_index=factor_index,
            standard_deviations=1.0,
        )
        raw_down = raw_dlv_state_with_factor_bump(
            day,
            factors,
            factor_index=factor_index,
            standard_deviations=-1.0,
        )
        factor_vegas[0, factor_index] = 0.5 * (
            price(raw_up) - price(raw_down)
        )
    return DlvGreeks(
        model_prices=np.asarray([base_price], dtype=np.float64),
        spot_deltas=np.asarray([delta], dtype=np.float64),
        factor_vegas=factor_vegas,
    )


def _conditional_weekly_claim_price(
    day: PublicMarketDay,
    *,
    claim: WeeklyClaimDefinition,
    decision_index: int,
    realized_path: FloatArray,
    inception_curve: FloatArray,
    spot_over_episode_origin: float,
    raw_state: FloatArray,
    spot_multiplier: float,
    config: SanosLvConfig,
) -> float:
    if not math.isfinite(spot_multiplier) or spot_multiplier <= 0.0:
        raise ValueError("spot multiplier must be finite and positive")
    remaining_steps = WEEKLY_HORIZON - decision_index
    current_spot = spot_over_episode_origin * spot_multiplier
    scales = _future_inception_scales(
        day,
        decision_index=decision_index,
        inception_curve=inception_curve,
        current_spot=current_spot,
        observed_current_scale=float(realized_path[-1]),
    )
    discount = day.discount_at_business_dte(remaining_steps)
    target = claim.claim
    if isinstance(target, DigitalClaim):
        expectation = _conditional_digital_value(
            day,
            target=target,
            remaining_steps=remaining_steps,
            terminal_scale=float(scales[-1]),
            raw_state=raw_state,
        )
    elif isinstance(target, RealizedVarianceClaim):
        expectation = _conditional_realized_variance_value(
            day,
            target=target,
            realized_path=realized_path,
            scales=scales,
            remaining_steps=remaining_steps,
            raw_state=raw_state,
        )
    elif isinstance(target, BarrierOptionClaim):
        expectation = _conditional_barrier_value(
            day,
            target=target,
            realized_path=realized_path,
            scales=scales,
            remaining_steps=remaining_steps,
            raw_state=raw_state,
            grid_size=config.weekly_background_grid_size,
            stddev_width=config.weekly_background_stddev_width,
        )
    else:
        raise TypeError(f"unsupported weekly claim type {type(target).__name__}")
    price = discount * expectation / current_spot
    if not math.isfinite(price) or price < 0.0:
        raise FloatingPointError("conditional weekly SANOS-LV price is invalid")
    return float(price)


def _conditional_digital_value(
    day: PublicMarketDay,
    *,
    target: DigitalClaim,
    remaining_steps: int,
    terminal_scale: float,
    raw_state: FloatArray,
) -> float:
    return float(
        target.cash_payoff
        * price_dlv_digitals(
            query_expiries_y=np.asarray([remaining_steps / DAYS_PER_YEAR]),
            query_strikes=np.asarray([target.strike / terminal_scale]),
            is_call=np.asarray([target.right is OptionRight.CALL]),
            support_strikes=day.dlv_support_strikes,
            raw_log_excess_dlvsq=raw_state,
            model_expiries_y=day.dlv_model_expiries_y,
            minimum_dlvsq=day.minimum_dlvsq,
            maximum_dlvsq=day.maximum_dlvsq,
            backbone_annual_volatility=day.backbone_annual_volatility,
        )[0]
    )


def _conditional_realized_variance_value(
    day: PublicMarketDay,
    *,
    target: RealizedVarianceClaim,
    realized_path: FloatArray,
    scales: FloatArray,
    remaining_steps: int,
    raw_state: FloatArray,
) -> float:
    transitions, intervals = _remaining_dlv_transitions(
        day,
        remaining_steps=remaining_steps,
        raw_state=raw_state,
    )
    realized_returns = np.diff(realized_path) / realized_path[:-1]
    expected_sum = float(np.sum(np.square(realized_returns)))
    support = day.dlv_support_strikes
    density = dlv_initial_density(support)
    latent_ratio = support[:, None] / support[None, :]
    for step, (transition, interval) in enumerate(
        zip(transitions, intervals, strict=True)
    ):
        deterministic_ratio = float(scales[step + 1] / scales[step])
        ratio = deterministic_ratio * latent_ratio
        background_second_moment = math.exp(
            day.backbone_annual_volatility**2 * float(interval)
        )
        squared_return = ratio**2 * background_second_moment - 2.0 * ratio + 1.0
        expected_sum += float(
            np.sum(transition * squared_return * density[None, :])
        )
        density = transition @ density
    return expected_sum / target.maturity_business_days


def _conditional_barrier_value(
    day: PublicMarketDay,
    *,
    target: BarrierOptionClaim,
    realized_path: FloatArray,
    scales: FloatArray,
    remaining_steps: int,
    raw_state: FloatArray,
    grid_size: int,
    stddev_width: float,
) -> float:
    observed_closes = realized_path[1:]
    knocked_out = (
        bool(np.any(observed_closes <= target.barrier))
        if target.direction is BarrierDirection.DOWN
        else bool(np.any(observed_closes >= target.barrier))
    )
    if knocked_out:
        return 0.0
    transitions, intervals = _remaining_dlv_transitions(
        day,
        remaining_steps=remaining_steps,
        raw_state=raw_state,
    )
    support = day.dlv_support_strikes
    initial_density = dlv_initial_density(support)
    if day.backbone_annual_volatility == 0.0:
        mass = initial_density.copy()
        for step, transition in enumerate(transitions):
            mass = transition @ mass
            future_spot = scales[step + 1] * support
            if target.direction is BarrierDirection.DOWN:
                mass *= future_spot > target.barrier
            else:
                mass *= future_spot < target.barrier
        terminal = scales[-1] * support
        payoff = _barrier_terminal_payoff(target, terminal)
        return float(mass @ payoff)

    total_time = float(np.sum(intervals))
    half_width = (
        stddev_width
        * day.backbone_annual_volatility
        * math.sqrt(total_time)
        + 0.5 * day.backbone_annual_volatility**2 * total_time
    )
    log_background = np.linspace(-half_width, half_width, grid_size)
    grid_step = float(log_background[1] - log_background[0])
    mass = np.zeros((len(support), grid_size), dtype=np.float64)
    mass[:, grid_size // 2] = initial_density
    for step, (transition, interval) in enumerate(
        zip(transitions, intervals, strict=True)
    ):
        advanced = transition @ mass
        diffused = gaussian_filter1d(
            advanced,
            sigma=(
                day.backbone_annual_volatility
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
                * day.backbone_annual_volatility**2
                * float(interval)
                / grid_step
            ),
        )
        if float(np.sum(diffused) - np.sum(mass)) > 1.0e-12:
            raise FloatingPointError("weekly background grid lost drift tail mass")
        future_spot = (
            scales[step + 1]
            * support[:, None]
            * np.exp(log_background)[None, :]
        )
        if target.direction is BarrierDirection.DOWN:
            mass *= future_spot > target.barrier
        else:
            mass *= future_spot < target.barrier
    terminal = scales[-1] * support[:, None] * np.exp(log_background)[None, :]
    return float(np.sum(mass * _barrier_terminal_payoff(target, terminal)))


def _barrier_terminal_payoff(
    target: BarrierOptionClaim,
    terminal_spot: FloatArray,
) -> FloatArray:
    if target.right is OptionRight.CALL:
        return np.maximum(terminal_spot - target.strike, 0.0)
    return np.maximum(target.strike - terminal_spot, 0.0)


def _remaining_dlv_transitions(
    day: PublicMarketDay,
    *,
    remaining_steps: int,
    raw_state: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    expected_expiries = np.arange(1, remaining_steps + 1) / DAYS_PER_YEAR
    if len(day.dlv_model_expiries_y) < remaining_steps or not np.allclose(
        day.dlv_model_expiries_y[:remaining_steps],
        expected_expiries,
        rtol=0.0,
        atol=2.0e-6 / DAYS_PER_YEAR,
    ):
        raise ValueError("conditional weekly pricing requires daily DLV expiry nodes")
    transitions = dlv_transition_matrices(
        support_strikes=day.dlv_support_strikes,
        raw_log_excess_dlvsq=raw_state,
        model_expiries_y=day.dlv_model_expiries_y,
        minimum_dlvsq=day.minimum_dlvsq,
        maximum_dlvsq=day.maximum_dlvsq,
        step_count=remaining_steps,
    )
    intervals = np.diff(
        np.concatenate(([0.0], day.dlv_model_expiries_y[:remaining_steps]))
    )
    return transitions, intervals


def _future_inception_scales(
    day: PublicMarketDay,
    *,
    decision_index: int,
    inception_curve: FloatArray,
    current_spot: float,
    observed_current_scale: float,
) -> FloatArray:
    remaining_steps = WEEKLY_HORIZON - decision_index
    current_forward_curve = np.asarray(
        [
            1.0,
            *(
                day.forward_at_business_dte(step)
                for step in range(1, remaining_steps + 1)
            ),
        ],
        dtype=np.float64,
    )
    scales = (
        current_spot
        * current_forward_curve
        / inception_curve[decision_index : WEEKLY_HORIZON + 1]
    )
    scales[0] = observed_current_scale
    return scales


def _validate_conditional_inputs(
    day: PublicMarketDay,
    *,
    claim: WeeklyClaimDefinition,
    decision_index: int,
    realized_inception_forward_path: NDArray[np.floating],
    inception_forward_over_episode_spot: NDArray[np.floating],
    spot_over_episode_origin: float,
) -> tuple[FloatArray, FloatArray]:
    if not 0 <= decision_index < WEEKLY_HORIZON:
        raise ValueError("weekly decision index must lie in [0,4]")
    if claim.claim.maturity_business_days != WEEKLY_HORIZON:
        raise ValueError("conditional weekly pricing requires a five-session claim")
    realized_path = np.asarray(
        realized_inception_forward_path,
        dtype=np.float64,
    )
    inception_curve = np.asarray(
        inception_forward_over_episode_spot,
        dtype=np.float64,
    )
    if realized_path.shape != (decision_index + 1,):
        raise ValueError("realized weekly path does not align with decision index")
    if inception_curve.shape != (WEEKLY_HORIZON + 1,):
        raise ValueError("inception forward curve must cover days zero to five")
    if (
        not math.isfinite(spot_over_episode_origin)
        or spot_over_episode_origin <= 0.0
        or np.any(~np.isfinite(realized_path) | (realized_path <= 0.0))
        or np.any(~np.isfinite(inception_curve) | (inception_curve <= 0.0))
    ):
        raise ValueError("conditional weekly state must be finite and positive")
    if day.trading_date is None:
        raise ValueError("conditional weekly pricing requires a dated market state")
    expected_current_scale = (
        spot_over_episode_origin / inception_curve[decision_index]
    )
    if not math.isclose(
        float(realized_path[-1]),
        float(expected_current_scale),
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("current spot and inception-forward path are inconsistent")
    return realized_path, inception_curve
