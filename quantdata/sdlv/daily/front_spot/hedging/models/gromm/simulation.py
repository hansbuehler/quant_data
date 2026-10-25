"""One-session GROMM scenarios in common benchmark accounting units."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.contracts import OptionQuote, OptionRight
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalSample,
    DailyDigitalTarget,
    build_daily_digital_sample,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_calibration import (
    RollingDlvCalibration,
    fit_rolling_dlv_calibration,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
)
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    black_mixture_call_exercise_probabilities,
    black_mixture_call_prices,
    one_day_physical_substepped,
    one_day_risk_neutral_substepped,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class GrommDailyScenarios:
    """One-day generated instrument and claim returns for one initial state."""

    origin_date: date
    measure: GrommMeasure
    target: DailyDigitalTarget
    sample: DailyDigitalSample
    calibration: RollingDlvCalibration
    instrument_returns: FloatArray
    claim_returns: FloatArray
    claim_terminal_values: FloatArray
    instrument_model_prices: FloatArray
    claim_model_price: float
    linear_transaction_costs: FloatArray
    terminal_transaction_costs: FloatArray
    settles_next: NDArray[np.bool_]
    generation_seconds: float

    def __post_init__(self) -> None:
        path_count = len(self.claim_returns)
        instrument_count = 1 + len(self.sample.hedge_quotes)
        if (
            self.sample.transition.origin.trading_date != self.origin_date
            or self.instrument_returns.shape != (path_count, instrument_count)
            or self.claim_terminal_values.shape != (path_count,)
            or self.instrument_model_prices.shape != (instrument_count,)
            or self.linear_transaction_costs.shape != (instrument_count,)
            or self.terminal_transaction_costs.shape
            != (path_count, instrument_count)
            or self.settles_next.shape != (instrument_count,)
        ):
            raise ValueError("GROMM daily scenario arrays do not align")
        arrays = (
            self.instrument_returns,
            self.claim_returns,
            self.claim_terminal_values,
            self.instrument_model_prices,
            self.linear_transaction_costs,
            self.terminal_transaction_costs,
        )
        if (
            path_count < 4
            or path_count % 2
            or any(not np.all(np.isfinite(values)) for values in arrays)
            or not np.isfinite(self.claim_model_price)
            or not np.isfinite(self.generation_seconds)
            or np.any(self.linear_transaction_costs < 0.0)
        ):
            raise ValueError("GROMM daily scenarios contain invalid values")

    @property
    def path_count(self) -> int:
        """Return the number of generated paths."""
        return len(self.claim_returns)


def generate_gromm_daily_scenarios(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    target: DailyDigitalTarget,
    config: GrommConfig = GrommConfig(),
    calibration: RollingDlvCalibration | None = None,
) -> GrommDailyScenarios:
    """Generate antithetic one-session scenarios from the rolling SDLV model.

    Instrument gains start from GROMM's own Q-consistent model marks. Actual
    SANOS market marks and realized execution costs remain in the common
    benchmark sample used to score the frozen action. This distinction makes
    the zero-claim Q policy a genuine martingale diagnostic while preserving
    common-market out-of-sample accounting.
    """
    require_public_hedging_panel(panel)
    if not jax.config.x64_enabled:
        raise RuntimeError("GROMM scenario generation requires JAX 64-bit mode")
    started = time.perf_counter()
    sample = build_daily_digital_sample(
        panel,
        origin_date=origin_date,
        target=target,
    )
    if calibration is None:
        calibration = fit_rolling_dlv_calibration(
            panel,
            as_of_date=origin_date,
            config=config.calibration,
        )
    if calibration.as_of_date != origin_date:
        raise ValueError("GROMM calibration must share the scenario origin date")

    day = sample.transition.origin
    support = jnp.asarray(day.dlv_support_strikes, dtype=jnp.float64)
    loadings = jnp.asarray(
        calibration.factors.annualized_loadings,
        dtype=jnp.float64,
    )
    initial_tower = np.concatenate(
        (calibration.front_density[None, :], calibration.future_densities),
        axis=0,
    )
    option_model_prices = _initial_option_model_prices(
        support=support,
        density_tower=initial_tower,
        quotes=sample.hedge_quotes,
        day=day,
    )
    claim_model_price = _initial_claim_model_price(
        support=support,
        density_tower=initial_tower,
        sample=sample,
    )
    instrument_model_prices = np.concatenate(([1.0], option_model_prices))
    linear_costs = np.asarray(
        [
            0.0,
            *(quote.half_spread for quote in sample.hedge_quotes),
        ],
        dtype=np.float64,
    )
    settles_next = np.asarray(
        [
            False,
            *(
                quote.contract.expiry_date == sample.transition.next_date
                for quote in sample.hedge_quotes
            ),
        ],
        dtype=bool,
    )

    instrument_returns = np.empty(
        (config.path_count, len(instrument_model_prices)),
        dtype=np.float64,
    )
    claim_values = np.empty(config.path_count, dtype=np.float64)
    terminal_costs = np.empty_like(instrument_returns)
    generator = np.random.default_rng(config.seed)
    if config.measure is GrommMeasure.Q:
        daily_step = jax.jit(
            one_day_risk_neutral_substepped,
            static_argnames=("expiry_dt", "time_power"),
        )
    else:
        if calibration.rolling_physical is None:
            raise ValueError("P scenarios require a rolling physical-drift fit")
        if not calibration.rolling_physical.accepted:
            raise ValueError(
                "P scenarios require a rolling physical-drift fit that passes "
                "its precommitted validation and daily-shift checks"
            )
        daily_step = jax.jit(
            one_day_physical_substepped,
            static_argnames=("expiry_dt", "time_power"),
        )
        market_price_of_risk = jnp.asarray(
            calibration.rolling_physical.market_price_of_risk,
            dtype=jnp.float64,
        )

    completed = 0
    while completed < config.path_count:
        pair_count = config.batch_size // 2
        factor_half = generator.standard_normal(
            (
                config.dlv_substeps,
                pair_count,
                calibration.factors.factor_rank,
            )
        )
        factor_normals = np.stack((factor_half, -factor_half), axis=2).reshape(
            config.dlv_substeps,
            config.batch_size,
            calibration.factors.factor_rank,
        )
        background_half = generator.standard_normal(pair_count)
        background_normals = np.stack(
            (background_half, -background_half), axis=1
        ).reshape(config.batch_size)
        state = RiskNeutralState(
            jnp.asarray(calibration.front_density, dtype=jnp.float64),
            jnp.asarray(day.dlvsq[1:], dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        if config.measure is GrommMeasure.Q:
            result = daily_step(
                support,
                state,
                loadings,
                jnp.asarray(factor_normals, dtype=jnp.float64),
                jnp.asarray(background_normals, dtype=jnp.float64),
                expiry_dt=1.0 / DAYS_PER_YEAR,
                time_power=config.time_power,
            )
        else:
            result = daily_step(
                support,
                state,
                loadings,
                market_price_of_risk,
                jnp.asarray(factor_normals, dtype=jnp.float64),
                jnp.asarray(background_normals, dtype=jnp.float64),
                expiry_dt=1.0 / DAYS_PER_YEAR,
                time_power=config.time_power,
            )
        terminal_spot = np.asarray(result.spot)
        terminal_option_values = _terminal_option_model_values(
            support=support,
            density_tower=result.density_tower,
            background_level=result.state.background_level,
            terminal_spot=terminal_spot,
            quotes=sample.hedge_quotes,
            day=day,
        )
        terminal_claim_values = _terminal_claim_model_values(
            support=support,
            density_tower=result.density_tower,
            background_level=result.state.background_level,
            terminal_spot=terminal_spot,
            sample=sample,
        )
        stop = completed + config.batch_size
        one_day_bridge = (
            sample.transition.discount_factor
            * sample.transition.forward_over_spot
        )
        instrument_returns[completed:stop, 0] = one_day_bridge * (
            terminal_spot - 1.0
        )
        instrument_returns[completed:stop, 1:] = (
            terminal_option_values - option_model_prices[None, :]
        )
        claim_values[completed:stop] = terminal_claim_values
        terminal_costs[completed:stop] = _terminal_transaction_costs(
            terminal_spot=terminal_spot,
            quotes=sample.hedge_quotes,
            origin_day=sample.transition.origin,
            settles_next=settles_next,
        )
        completed = stop
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                "stage": "gromm-generate-daily",
                "measure": config.measure.value,
                "completed_paths": completed,
                "path_count": config.path_count,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed * (config.path_count - completed) / completed,
                }
            ),
            flush=True,
        )

    claim_returns = claim_values - claim_model_price
    return GrommDailyScenarios(
        origin_date=origin_date,
        measure=config.measure,
        target=target,
        sample=sample,
        calibration=calibration,
        instrument_returns=instrument_returns,
        claim_returns=claim_returns,
        claim_terminal_values=claim_values,
        instrument_model_prices=instrument_model_prices,
        claim_model_price=claim_model_price,
        linear_transaction_costs=linear_costs,
        terminal_transaction_costs=terminal_costs,
        settles_next=settles_next,
        generation_seconds=time.perf_counter() - started,
    )


def _terminal_transaction_costs(
    *,
    terminal_spot: FloatArray,
    quotes: tuple[OptionQuote, ...],
    origin_day: PublicMarketDay,
    settles_next: NDArray[np.bool_],
) -> FloatArray:
    """Return one-day liquidation costs in origin-PV spot-one units."""
    path_count = len(terminal_spot)
    spot_over_origin = (
        origin_day.forward_at_business_dte(1) * np.asarray(terminal_spot)
    )
    discount = origin_day.discount_at_business_dte(1)
    costs = np.zeros((path_count, 1 + len(quotes)), dtype=np.float64)
    costs[:, 0] = 0.0
    for option_index, quote in enumerate(quotes, start=1):
        if settles_next[option_index]:
            continue
        initial_dte = int(round(quote.business_dte))
        remaining_dte = initial_dte - 1
        fixed_strike_over_origin = (
            quote.strike_over_forward
            * origin_day.forward_at_business_dte(initial_dte)
        )
        costs[:, option_index] = (
            discount
            * spot_over_origin
            * origin_day.half_spreads_for_fixed_contracts(
                business_dte=remaining_dte,
                fixed_strikes_over_current_spot=(
                    fixed_strike_over_origin / spot_over_origin
                ),
                is_call=quote.contract.right is OptionRight.CALL,
            )
        )
    return costs


def _initial_option_model_prices(
    *,
    support: jax.Array,
    density_tower: FloatArray,
    quotes: tuple[OptionQuote, ...],
    day: PublicMarketDay,
) -> FloatArray:
    values = np.empty(len(quotes), dtype=np.float64)
    for dte, indices in _quote_groups(quotes).items():
        strikes = np.asarray(
            [quotes[index].strike_over_forward for index in indices],
            dtype=np.float64,
        )
        normalized = _option_values(
            support=support,
            density_tower=jnp.asarray(density_tower[dte - 1 : dte]),
            background_level=jnp.asarray(1.0, dtype=jnp.float64),
            spot=np.asarray(1.0),
            strikes=strikes,
            rights=tuple(quotes[index].contract.right for index in indices),
            remaining_days=dte,
        )[0]
        bridge = day.discount_at_business_dte(dte) * day.forward_at_business_dte(
            dte
        )
        values[np.asarray(indices)] = bridge * normalized
    return values


def _terminal_option_model_values(
    *,
    support: jax.Array,
    density_tower: jax.Array,
    background_level: jax.Array,
    terminal_spot: FloatArray,
    quotes: tuple[OptionQuote, ...],
    day: PublicMarketDay,
) -> FloatArray:
    values = np.empty((len(terminal_spot), len(quotes)), dtype=np.float64)
    for dte, indices in _quote_groups(quotes).items():
        strikes = np.asarray(
            [quotes[index].strike_over_forward for index in indices],
            dtype=np.float64,
        )
        rights = tuple(quotes[index].contract.right for index in indices)
        bridge = day.discount_at_business_dte(dte) * day.forward_at_business_dte(
            dte
        )
        if dte == 1:
            call = np.maximum(terminal_spot[:, None] - strikes[None, :], 0.0)
            put = np.maximum(strikes[None, :] - terminal_spot[:, None], 0.0)
            normalized = np.where(
                np.asarray([right is OptionRight.CALL for right in rights])[None],
                call,
                put,
            )
        else:
            normalized = _option_values(
                support=support,
                density_tower=density_tower[:, dte - 2 : dte - 1],
                background_level=background_level,
                spot=terminal_spot,
                strikes=strikes,
                rights=rights,
                remaining_days=dte - 1,
            )[:, 0]
        values[:, np.asarray(indices)] = bridge * normalized
    return values


def _option_values(
    *,
    support: jax.Array,
    density_tower: jax.Array,
    background_level: jax.Array,
    spot: FloatArray,
    strikes: FloatArray,
    rights: tuple[OptionRight, ...],
    remaining_days: int,
) -> FloatArray:
    calls = np.asarray(
        black_mixture_call_prices(
            support,
            density_tower,
            jnp.asarray(strikes[None, :], dtype=jnp.float64),
            background_level,
            jnp.asarray([remaining_days], dtype=jnp.float64),
        )
    )
    spot_array = np.asarray(spot, dtype=np.float64)
    if spot_array.ndim:
        spot_array = spot_array[:, None, None]
    puts = calls - spot_array + strikes[None, :]
    call_mask = np.asarray([right is OptionRight.CALL for right in rights])
    values = np.where(call_mask[None, :], calls, puts)
    if np.any(~np.isfinite(values)) or np.min(values) < -2.0e-10:
        raise FloatingPointError("GROMM option values are negative or non-finite")
    return np.maximum(values, 0.0)


def _initial_claim_model_price(
    *,
    support: jax.Array,
    density_tower: FloatArray,
    sample: DailyDigitalSample,
) -> float:
    dte = sample.target.maturity_business_days
    probability = _digital_probabilities(
        support=support,
        density_tower=jnp.asarray(density_tower[dte - 1 : dte]),
        background_level=jnp.asarray(1.0, dtype=jnp.float64),
        strike=sample.target_strike_over_forward,
        right=sample.target.right,
        remaining_days=dte,
    )[0]
    return sample.target_discount_factor * float(probability)


def _terminal_claim_model_values(
    *,
    support: jax.Array,
    density_tower: jax.Array,
    background_level: jax.Array,
    terminal_spot: FloatArray,
    sample: DailyDigitalSample,
) -> FloatArray:
    dte = sample.target.maturity_business_days
    strike = sample.target_strike_over_forward
    if dte == 1:
        if sample.target.right is OptionRight.CALL:
            payoff = terminal_spot > strike
        else:
            payoff = terminal_spot < strike
        return sample.target_discount_factor * payoff.astype(np.float64)
    probability = _digital_probabilities(
        support=support,
        density_tower=density_tower[:, dte - 2 : dte - 1],
        background_level=background_level,
        strike=strike,
        right=sample.target.right,
        remaining_days=dte - 1,
    )[:, 0]
    return sample.target_discount_factor * probability


def _digital_probabilities(
    *,
    support: jax.Array,
    density_tower: jax.Array,
    background_level: jax.Array,
    strike: float,
    right: OptionRight,
    remaining_days: int,
) -> FloatArray:
    call_probability = np.asarray(
        black_mixture_call_exercise_probabilities(
            support,
            density_tower,
            jnp.asarray([[strike]], dtype=jnp.float64),
            background_level,
            jnp.asarray([remaining_days], dtype=jnp.float64),
        )
    )
    call_probability = call_probability[..., 0]
    if right is OptionRight.CALL:
        return call_probability
    return 1.0 - call_probability


def _quote_groups(quotes: tuple[OptionQuote, ...]) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {}
    for index, quote in enumerate(quotes):
        dte = int(round(quote.business_dte))
        if dte < 1 or not math.isclose(quote.business_dte, dte, abs_tol=2.0e-6):
            raise ValueError("GROMM currently requires integer business-day expiries")
        groups.setdefault(dte, []).append(index)
    return {dte: tuple(indices) for dte, indices in groups.items()}
