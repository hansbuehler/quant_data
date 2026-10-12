"""Common fixed-contract scenarios for statistical surface generators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.objectives import LinearHedge, cost_lasso_minimum_variance_hedge
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalSample,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)

FloatArray = NDArray[np.float64]


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class GeneratedHedgeConfig:
    """Common Monte Carlo and cost-lasso controls for generated hedges."""

    scenario_count: int = 4_096
    seed: int = 20_261_013
    cost_multiplier: float = 1.0
    position_cap: float = 1_000_000.0

    def __post_init__(self) -> None:
        verify_inp(
            self.scenario_count >= 2,
            lambda: "scenario_count must be at least two",
        )
        verify_inp(
            isinstance(self.seed, int) and not isinstance(self.seed, bool),
            lambda: "seed must be an integer",
        )
        verify_inp(
            math.isfinite(self.cost_multiplier) and self.cost_multiplier >= 0.0,
            lambda: "cost_multiplier must be finite and non-negative",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite-positive",
        )


@dataclass(frozen=True, kw_only=True)
class GeneratedDailyScenarios:
    """Generated one-session gains in discounted origin-spot units."""

    spot_returns: FloatArray
    option_returns: FloatArray
    claim_returns: FloatArray
    claim_terminal_values: FloatArray
    digital_probability_clip_count: int = 0

    def __post_init__(self) -> None:
        path_count = len(self.spot_returns)
        if (
            self.spot_returns.ndim != 1
            or self.option_returns.ndim != 2
            or self.option_returns.shape[0] != path_count
            or self.claim_returns.shape != (path_count,)
            or self.claim_terminal_values.shape != (path_count,)
            or path_count < 2
        ):
            raise ValueError("generated daily scenario arrays do not align")
        if any(
            not np.all(np.isfinite(values))
            for values in (
                self.spot_returns,
                self.option_returns,
                self.claim_returns,
                self.claim_terminal_values,
            )
        ):
            raise ValueError("generated daily scenarios must be finite")
        if np.any(self.claim_terminal_values < 0.0):
            raise ValueError("generated claim values must be non-negative")
        if self.digital_probability_clip_count < 0:
            raise ValueError("digital probability clip count must be non-negative")

    @property
    def instrument_returns(self) -> FloatArray:
        """Return spot followed by the fixed inception option universe."""
        return np.column_stack((self.spot_returns, self.option_returns))


@dataclass(frozen=True, kw_only=True)
class GeneratedCostLassoFit:
    """Capped generated-scenario hedge and underlying lasso diagnostics."""

    action: FloatArray
    uncapped: LinearHedge
    cap_hits: int
    converged: bool


def build_generated_daily_scenarios(
    sample: DailyDigitalSample,
    *,
    forward_log_returns: NDArray[np.floating],
    next_log_implied_volatilities: NDArray[np.floating],
    grid: GenerativeSurfaceGrid,
) -> GeneratedDailyScenarios:
    """Reprice the fixed target and hedge book in generated next-day states.

    The generated spot coordinate is ``log(S_{t+1}/F_t(t+1))``. Deterministic
    carry therefore maps an inception strike ``K/F_t(T)=k`` to the next-date
    strike ``K/F_{t+1}(T)=k/exp(r)``. All returned values are discounted back
    to the episode origin and divided by origin spot.
    """
    log_returns = np.asarray(forward_log_returns, dtype=np.float64)
    surfaces = np.asarray(next_log_implied_volatilities, dtype=np.float64)
    expected_surface_shape = (
        len(log_returns),
        len(grid.expiries_y),
        len(grid.strikes_over_forward),
    )
    if (
        log_returns.ndim != 1
        or len(log_returns) < 2
        or surfaces.shape != expected_surface_shape
        or not np.all(np.isfinite(log_returns))
        or not np.all(np.isfinite(surfaces))
    ):
        raise ValueError("generated transitions have invalid shape or values")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_terminal_spot = np.exp(log_returns)
    if not np.all(
        np.isfinite(normalized_terminal_spot) & (normalized_terminal_spot > 0.0)
    ):
        raise FloatingPointError("generated normalized spot is not finite-positive")

    origin = sample.transition.origin
    one_day_bridge = (
        origin.discount_at_business_dte(1) * origin.forward_at_business_dte(1)
    )
    spot_returns = one_day_bridge * (normalized_terminal_spot - 1.0)
    option_returns = np.empty(
        (len(log_returns), len(sample.hedge_quotes)),
        dtype=np.float64,
    )
    for option_index, quote in enumerate(sample.hedge_quotes):
        maturity = _integral_business_dte(quote.business_dte)
        if maturity == 1:
            future_value = one_day_bridge * _option_payoff(
                normalized_terminal_spot,
                strike=quote.strike_over_forward,
                right=quote.contract.right,
            )
        else:
            remaining = maturity - 1
            future_strike = quote.strike_over_forward / normalized_terminal_spot
            volatility = interpolate_generated_implied_volatility(
                surfaces,
                query_expiry_y=remaining / DAYS_PER_YEAR,
                query_strikes_over_forward=future_strike,
                grid=grid,
            )
            normalized_price = _black_option_price(
                strike=future_strike,
                annual_volatility=volatility,
                expiry_y=remaining / DAYS_PER_YEAR,
                right=quote.contract.right,
            )
            origin_bridge = (
                origin.discount_at_business_dte(maturity)
                * origin.forward_at_business_dte(maturity)
            )
            future_value = (
                normalized_terminal_spot * origin_bridge * normalized_price
            )
        option_returns[:, option_index] = future_value - quote.mark

    target_maturity = sample.target.maturity_business_days
    target_strike = sample.target_strike_over_forward
    digital_probability_clip_count = 0
    if target_maturity == 1:
        exercised = (
            normalized_terminal_spot > target_strike
            if sample.target.right is OptionRight.CALL
            else normalized_terminal_spot <= target_strike
        )
        claim_terminal_values = (
            origin.discount_at_business_dte(1) * exercised.astype(np.float64)
        )
    else:
        remaining = target_maturity - 1
        future_strike = target_strike / normalized_terminal_spot
        probability, digital_probability_clip_count = (
            _smile_consistent_digital_probability(
                surfaces,
                strike=future_strike,
                expiry_y=remaining / DAYS_PER_YEAR,
                right=sample.target.right,
                grid=grid,
            )
        )
        claim_terminal_values = (
            origin.discount_at_business_dte(target_maturity) * probability
        )
    return GeneratedDailyScenarios(
        spot_returns=np.asarray(spot_returns, dtype=np.float64),
        option_returns=option_returns,
        claim_returns=np.asarray(
            claim_terminal_values - sample.market_price,
            dtype=np.float64,
        ),
        claim_terminal_values=np.asarray(claim_terminal_values, dtype=np.float64),
        digital_probability_clip_count=digital_probability_clip_count,
    )


def fit_generated_cost_lasso(
    scenarios: GeneratedDailyScenarios,
    sample: DailyDigitalSample,
    *,
    cost_multiplier: float = 1.0,
    position_cap: float = 1_000_000.0,
    claim_quantity: float = 1.0,
) -> GeneratedCostLassoFit:
    """Fit the common spread-weighted minimum-variance generated hedge."""
    if not math.isfinite(position_cap) or position_cap <= 0.0:
        raise ValueError("position_cap must be finite-positive")
    if not math.isfinite(claim_quantity) or claim_quantity == 0.0:
        raise ValueError("claim_quantity must be finite and nonzero")
    costs = generated_linear_transaction_costs(sample)
    uncapped = cost_lasso_minimum_variance_hedge(
        instrument_returns=scenarios.instrument_returns,
        claim_returns=claim_quantity * scenarios.claim_returns,
        linear_transaction_costs=costs,
        cost_multiplier=cost_multiplier,
    )
    action = np.clip(uncapped.action, -position_cap, position_cap)
    cap_hits = int(np.count_nonzero(action != uncapped.action))
    return GeneratedCostLassoFit(
        action=np.asarray(action, dtype=np.float64),
        uncapped=uncapped,
        cap_hits=cap_hits,
        converged=(
            uncapped.converged
            and uncapped.covariance_converged
            and cap_hits == 0
        ),
    )


def generated_linear_transaction_costs(
    sample: DailyDigitalSample,
) -> FloatArray:
    """Return entry-plus-exit costs for the generated fixed-contract book."""
    return np.asarray(
        [
            2.0e-4,
            *(
                quote.half_spread
                * (
                    1.0
                    if quote.contract.expiry_date == sample.transition.next_date
                    else 2.0
                )
                for quote in sample.hedge_quotes
            ),
        ],
        dtype=np.float64,
    )


def interpolate_generated_implied_volatility(
    log_implied_volatilities: NDArray[np.floating],
    *,
    query_expiry_y: float,
    query_strikes_over_forward: NDArray[np.floating],
    grid: GenerativeSurfaceGrid,
) -> FloatArray:
    """Interpolate pathwise log strike and total variance on the fixed grid."""
    surfaces = np.asarray(log_implied_volatilities, dtype=np.float64)
    strikes = np.asarray(query_strikes_over_forward, dtype=np.float64)
    grid_strikes = np.asarray(grid.strikes_over_forward, dtype=np.float64)
    grid_expiries = np.asarray(grid.expiries_y, dtype=np.float64)
    expected_shape = (len(strikes), len(grid_expiries), len(grid_strikes))
    if (
        surfaces.shape != expected_shape
        or strikes.ndim != 1
        or not math.isfinite(query_expiry_y)
        or query_expiry_y <= 0.0
        or not np.all(np.isfinite(strikes) & (strikes > 0.0))
        or not np.all(np.isfinite(surfaces))
    ):
        raise ValueError("generated implied-volatility interpolation inputs are invalid")
    log_grid_strikes = np.log(grid_strikes)
    log_queries = np.log(strikes)
    lower = np.searchsorted(log_grid_strikes, log_queries, side="right") - 1
    lower = np.clip(lower, 0, len(grid_strikes) - 2)
    upper = lower + 1
    denominator = log_grid_strikes[upper] - log_grid_strikes[lower]
    weight = np.clip(
        (log_queries - log_grid_strikes[lower]) / denominator,
        0.0,
        1.0,
    )
    path_indices = np.arange(len(strikes))[:, None]
    expiry_indices = np.arange(len(grid_expiries))[None, :]
    lower_values = surfaces[path_indices, expiry_indices, lower[:, None]]
    upper_values = surfaces[path_indices, expiry_indices, upper[:, None]]
    log_volatility_by_expiry = (
        lower_values + weight[:, None] * (upper_values - lower_values)
    )
    volatility_by_expiry = np.exp(log_volatility_by_expiry)
    total_variance = volatility_by_expiry**2 * grid_expiries[None, :]
    upper_expiry = int(np.searchsorted(grid_expiries, query_expiry_y, side="left"))
    if upper_expiry == 0:
        query_variance = total_variance[:, 0] * query_expiry_y / grid_expiries[0]
    elif upper_expiry == len(grid_expiries):
        query_variance = (
            volatility_by_expiry[:, -1] ** 2 * query_expiry_y
        )
    else:
        lower_expiry = upper_expiry - 1
        weight_expiry = (
            (query_expiry_y - grid_expiries[lower_expiry])
            / (grid_expiries[upper_expiry] - grid_expiries[lower_expiry])
        )
        query_variance = (
            (1.0 - weight_expiry) * total_variance[:, lower_expiry]
            + weight_expiry * total_variance[:, upper_expiry]
        )
    volatility = np.sqrt(
        np.maximum(query_variance, np.finfo(np.float64).tiny) / query_expiry_y
    )
    if not np.all(np.isfinite(volatility) & (volatility > 0.0)):
        raise FloatingPointError("interpolated generated volatility is invalid")
    return np.asarray(volatility, dtype=np.float64)


def _integral_business_dte(value: float) -> int:
    rounded = int(round(value))
    if rounded < 1 or not math.isclose(
        value,
        rounded,
        rel_tol=0.0,
        abs_tol=2.0e-6,
    ):
        raise ValueError(f"option business DTE must be a positive integer, found {value}")
    return rounded


def _option_payoff(
    normalized_spot: FloatArray,
    *,
    strike: float,
    right: OptionRight,
) -> FloatArray:
    if right is OptionRight.CALL:
        return np.maximum(normalized_spot - strike, 0.0)
    return np.maximum(strike - normalized_spot, 0.0)


def _black_option_price(
    *,
    strike: FloatArray,
    annual_volatility: FloatArray,
    expiry_y: float,
    right: OptionRight,
) -> FloatArray:
    calls = np.asarray(
        bs.price(
            k=strike,
            vol=annual_volatility,
            sqrtT=math.sqrt(expiry_y),
            eps=1.0e-10,
        ),
        dtype=np.float64,
    )
    return calls if right is OptionRight.CALL else calls - 1.0 + strike


def _smile_consistent_digital_probability(
    log_implied_volatilities: FloatArray,
    *,
    strike: FloatArray,
    expiry_y: float,
    right: OptionRight,
    grid: GenerativeSurfaceGrid,
) -> tuple[FloatArray, int]:
    """Differentiate the generated smile's call curve at each pathwise strike."""
    log_step = 1.0e-4
    lower_strike = strike * math.exp(-log_step)
    upper_strike = strike * math.exp(log_step)
    lower_volatility = interpolate_generated_implied_volatility(
        log_implied_volatilities,
        query_expiry_y=expiry_y,
        query_strikes_over_forward=lower_strike,
        grid=grid,
    )
    upper_volatility = interpolate_generated_implied_volatility(
        log_implied_volatilities,
        query_expiry_y=expiry_y,
        query_strikes_over_forward=upper_strike,
        grid=grid,
    )
    lower_call = _black_option_price(
        strike=lower_strike,
        annual_volatility=lower_volatility,
        expiry_y=expiry_y,
        right=OptionRight.CALL,
    )
    upper_call = _black_option_price(
        strike=upper_strike,
        annual_volatility=upper_volatility,
        expiry_y=expiry_y,
        right=OptionRight.CALL,
    )
    raw_call_probability = -(upper_call - lower_call) / (
        upper_strike - lower_strike
    )
    raw_probability = (
        raw_call_probability
        if right is OptionRight.CALL
        else 1.0 - raw_call_probability
    )
    clip_count = int(
        np.count_nonzero((raw_probability < 0.0) | (raw_probability > 1.0))
    )
    probability = np.clip(raw_probability, 0.0, 1.0)
    if not np.all(np.isfinite(probability)):
        raise FloatingPointError("generated smile produced invalid digital values")
    return np.asarray(probability, dtype=np.float64), clip_count
