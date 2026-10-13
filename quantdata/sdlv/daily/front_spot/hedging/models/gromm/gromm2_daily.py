"""One-session benchmark scenarios from a dated rolling GROMM2 artifact."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from datetime import date

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from numpy.typing import NDArray

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.dynamics import EstimatedDynamics, load_estimated_dynamics
from quantdata.gromm2.pricing import price_active_calls
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step
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
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Artifacts,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.training import (
    GrommTrainingData,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class Gromm2MarketState:
    """Current state and estimated dynamics loaded from one dated artifact."""

    as_of_date: date
    supports: FloatArray
    transports: FloatArray
    expiries_y: FloatArray
    business_dtes: FloatArray
    dlvsq: FloatArray
    density_tower: FloatArray
    dynamics: EstimatedDynamics
    previous_factor_score: FloatArray
    physical_market_price_of_risk: FloatArray
    percent_inside_spread: float
    explained_variance_ratio: float
    physical_projection_fraction: float
    candidate_physical_projection_fraction: float
    physical_fallback_used: bool
    physical_selected_model: str
    physical_accepted: bool

    def __post_init__(self) -> None:
        expiry_count, support_count = self.supports.shape
        factor_count = self.dynamics.factor_count
        if (
            self.transports.shape != (expiry_count - 1, support_count, support_count)
            or self.expiries_y.shape != (expiry_count,)
            or self.business_dtes.shape != (expiry_count,)
            or self.dlvsq.shape != (expiry_count, support_count - 2)
            or self.density_tower.shape != (expiry_count, support_count)
            or self.dynamics.log_dlvsq_loadings.shape
            != (expiry_count - 1, support_count - 2, factor_count)
            or self.previous_factor_score.shape != (factor_count,)
            or self.physical_market_price_of_risk.shape != (factor_count,)
            or not isinstance(self.physical_fallback_used, bool)
            or self.physical_selected_model
            not in {"candidate", "front-only-null", "unavailable"}
            or not isinstance(self.physical_accepted, bool)
            or self.physical_fallback_used
            != (self.physical_selected_model == "front-only-null")
            or self.physical_accepted
            != (self.physical_selected_model != "unavailable")
        ):
            raise ValueError("GROMM2 market-state arrays do not align")
        arrays = (
            self.supports,
            self.transports,
            self.expiries_y,
            self.business_dtes,
            self.dlvsq,
            self.density_tower,
            self.previous_factor_score,
            self.physical_market_price_of_risk,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("GROMM2 market state must be finite")
        if (
            np.any(self.dlvsq <= 0.0)
            or np.any(self.density_tower < -1.0e-12)
            or not np.allclose(
                np.sum(self.density_tower, axis=-1), 1.0, atol=2.0e-10, rtol=0.0
            )
        ):
            raise ValueError("GROMM2 market state violates density invariants")


@dataclass(frozen=True, kw_only=True)
class Gromm2DailyScenarios:
    """One-day generated returns from the accepted GROMM2 model."""

    origin_date: date
    measure: GrommMeasure
    target: DailyDigitalTarget
    sample: DailyDigitalSample
    market_state: Gromm2MarketState
    instrument_returns: FloatArray
    pure_instrument_returns: FloatArray
    claim_returns: FloatArray
    claim_terminal_values: FloatArray
    instrument_model_prices: FloatArray
    instrument_entry_prices: FloatArray
    claim_model_price: float
    linear_transaction_costs: FloatArray
    terminal_transaction_costs: FloatArray
    settles_next: NDArray[np.bool_]
    drift_taming_fraction: float
    generation_seconds: float

    def __post_init__(self) -> None:
        path_count = len(self.claim_returns)
        instrument_count = 1 + len(self.sample.hedge_quotes)
        if (
            self.market_state.as_of_date != self.origin_date
            or self.instrument_returns.shape != (path_count, instrument_count)
            or self.pure_instrument_returns.shape
            != (path_count, instrument_count)
            or self.claim_terminal_values.shape != (path_count,)
            or self.instrument_model_prices.shape != (instrument_count,)
            or self.instrument_entry_prices.shape != (instrument_count,)
            or self.linear_transaction_costs.shape != (instrument_count,)
            or self.terminal_transaction_costs.shape
            != (path_count, instrument_count)
            or self.settles_next.shape != (instrument_count,)
        ):
            raise ValueError("GROMM2 daily scenario arrays do not align")
        arrays = (
            self.instrument_returns,
            self.pure_instrument_returns,
            self.claim_returns,
            self.claim_terminal_values,
            self.instrument_model_prices,
            self.instrument_entry_prices,
            self.linear_transaction_costs,
            self.terminal_transaction_costs,
        )
        if (
            path_count < 4
            or path_count % 2
            or any(not np.all(np.isfinite(values)) for values in arrays)
            or not np.isfinite(self.claim_model_price)
            or not np.isfinite(self.drift_taming_fraction)
            or not np.isfinite(self.generation_seconds)
        ):
            raise ValueError("GROMM2 daily scenarios contain invalid values")

    @property
    def path_count(self) -> int:
        return len(self.claim_returns)


def load_gromm2_market_state(
    artifacts: RollingGromm2Artifacts,
) -> Gromm2MarketState:
    """Load the current state and predict the compatible one-day P risk price."""
    with h5py.File(artifacts.paths.affine_history, "r") as stream:
        serialized_dtypes = tuple(
            np.dtype(stream[name].dtype)
            for name in (
                "supports",
                "transports",
                "model_expiries_y",
                "dlvsq",
            )
        )
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        dlvsq = np.asarray(stream["dlvsq"][-1], dtype=np.float64)
    dynamics = load_estimated_dynamics(artifacts.paths.dynamics)
    with np.load(artifacts.paths.dynamics, allow_pickle=False) as values:
        factor_scores = np.asarray(values["factor_scores"], dtype=np.float64)
        risk_price = np.asarray(
            values["physical_market_price_of_risk"],
            dtype=np.float64,
        )
    if factor_scores.ndim != 2 or len(factor_scores) == 0:
        raise ValueError("rolling GROMM2 dynamics contain no factor-score history")
    intervals = np.diff(expiries, prepend=0.0)
    decoded_tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(dlvsq),
            jnp.asarray(intervals),
        )
    )
    tower, mass_error, mass_tolerance = _normalize_decoded_density_tower(
        decoded_tower,
        source_epsilon=max(float(np.finfo(dtype).eps) for dtype in serialized_dtypes),
    )
    print(
        json.dumps(
            {
                "stage": "gromm2-load-density-tower",
                "serialized_dtypes": sorted({str(dtype) for dtype in serialized_dtypes}),
                "decoded_dtype": str(decoded_tower.dtype),
                "maximum_mass_error_before_normalization": mass_error,
                "mass_error_tolerance": mass_tolerance,
            }
        ),
        flush=True,
    )
    previous_score = factor_scores[-1]
    return Gromm2MarketState(
        as_of_date=artifacts.as_of_date,
        supports=supports,
        transports=transports,
        expiries_y=expiries,
        business_dtes=business_dtes,
        dlvsq=dlvsq,
        density_tower=tower,
        dynamics=dynamics,
        previous_factor_score=previous_score,
        physical_market_price_of_risk=risk_price,
        percent_inside_spread=artifacts.percent_inside_spread,
        explained_variance_ratio=artifacts.explained_variance_ratio,
        physical_projection_fraction=artifacts.physical_projection_fraction,
        candidate_physical_projection_fraction=(
            artifacts.candidate_physical_projection_fraction
        ),
        physical_fallback_used=artifacts.physical_fallback_used,
        physical_selected_model=artifacts.physical_selected_model,
        physical_accepted=artifacts.physical_accepted,
    )


def price_gromm2_daily_target(
    market_state: Gromm2MarketState,
    *,
    sample: DailyDigitalSample,
) -> float:
    """Return the exact current Q expectation of one daily digital target."""
    target = sample.target
    if market_state.as_of_date != sample.transition.origin.trading_date:
        raise ValueError("GROMM2 market state and daily sample dates differ")
    expiry_index = target.maturity_business_days - 1
    if expiry_index < 0 or expiry_index >= len(market_state.business_dtes):
        raise ValueError("daily target maturity is absent from the GROMM2 state")
    value = sample.target_discount_factor * float(
        _digital_probability(
            support=market_state.supports[expiry_index],
            density=market_state.density_tower[expiry_index][None],
            background_level=np.ones(1),
            strike=sample.target_strike_over_forward,
            remaining_days=target.maturity_business_days,
            right=target.right,
        )[0]
    )
    if not math.isfinite(value) or value < 0.0:
        raise FloatingPointError("GROMM2 daily target price is invalid")
    return value


def gromm2_market_state_with_dlvsq(
    market_state: Gromm2MarketState,
    *,
    dlvsq: NDArray[np.floating],
) -> Gromm2MarketState:
    """Replace the live DLV state and reconstruct its exact density tower."""
    values = np.asarray(dlvsq, dtype=np.float64)
    if (
        values.shape != market_state.dlvsq.shape
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("replacement GROMM2 DLV state is invalid")
    intervals = np.diff(market_state.expiries_y, prepend=0.0)
    decoded = np.asarray(
        decode_density_tower(
            jnp.asarray(market_state.supports),
            jnp.asarray(market_state.transports),
            jnp.asarray(values),
            jnp.asarray(intervals),
        )
    )
    tower, _, _ = _normalize_decoded_density_tower(decoded)
    return replace(
        market_state,
        dlvsq=values.copy(),
        density_tower=tower,
    )


def _normalize_decoded_density_tower(
    density_tower: NDArray[np.floating],
    *,
    source_epsilon: float | None = None,
) -> tuple[FloatArray, float, float]:
    """Validate bounded decode roundoff and return float64 unit-mass rows."""
    raw = np.asarray(density_tower)
    if raw.ndim != 2 or raw.dtype.kind != "f" or not np.all(np.isfinite(raw)):
        raise ValueError("decoded GROMM2 density tower must be a finite matrix")
    effective_epsilon = (
        float(np.finfo(raw.dtype).eps)
        if source_epsilon is None
        else float(source_epsilon)
    )
    if not math.isfinite(effective_epsilon) or effective_epsilon <= 0.0:
        raise ValueError("decoded GROMM2 source epsilon must be finite and positive")
    mass_tolerance = max(2.0e-10, 8.0 * effective_epsilon)
    values = np.asarray(raw, dtype=np.float64)
    minimum_density = float(np.min(values))
    if minimum_density < -1.0e-12:
        raise ValueError(
            "decoded GROMM2 density tower has negative mass: "
            f"minimum={minimum_density:.6g}"
        )
    values = np.maximum(values, 0.0)
    masses = np.sum(values, axis=-1)
    maximum_mass_error = float(np.max(np.abs(masses - 1.0)))
    if np.any(masses <= 0.0) or maximum_mass_error > mass_tolerance:
        raise ValueError(
            "decoded GROMM2 density mass error exceeds its precision bound: "
            f"maximum_error={maximum_mass_error:.6g}, "
            f"tolerance={mass_tolerance:.6g}, decoded_dtype={raw.dtype}, "
            f"source_epsilon={effective_epsilon:.6g}"
        )
    normalized = values / masses[:, None]
    if not np.allclose(
        np.sum(normalized, axis=-1),
        1.0,
        atol=2.0e-10,
        rtol=0.0,
    ):
        raise FloatingPointError("normalized GROMM2 density tower has invalid mass")
    return normalized, maximum_mass_error, mass_tolerance


def predict_physical_market_price_of_risk(
    *,
    dlvsq: FloatArray,
    dynamics: EstimatedDynamics,
    previous_factor_score: FloatArray,
) -> FloatArray:
    """Evaluate the rolling AR(1)-compatible affine P risk-price predictor."""
    levels = np.asarray(dlvsq, dtype=np.float64)
    expected_future_shape = dynamics.level_mean.shape
    if levels.shape[-2:] == (expected_future_shape[0] + 1, expected_future_shape[1]):
        levels = levels[..., 1:, :]
    if levels.shape[-2:] != expected_future_shape:
        raise ValueError("GROMM2 physical risk-price DLV geometry does not align")
    future_log_levels = np.log(levels).reshape(levels.shape[:-2] + (-1,))
    level_mean = dynamics.level_mean.reshape(-1)
    level_scores = (future_log_levels - level_mean) @ dynamics.level_basis
    level_scores /= dynamics.level_scale
    previous = np.asarray(previous_factor_score, dtype=np.float64)
    if previous.shape != level_scores.shape[:-1] + (dynamics.factor_count,):
        raise ValueError("GROMM2 previous factor scores do not align")
    intercept = np.ones(level_scores.shape[:-1] + (1,), dtype=np.float64)
    features = np.concatenate((intercept, level_scores, previous), axis=-1)
    if features.shape[-1] != dynamics.physical_factor_coefficients.shape[0]:
        raise ValueError("GROMM2 physical risk-price features do not align")
    values = features @ dynamics.physical_factor_coefficients
    if values.shape != level_scores.shape[:-1] + (
        dynamics.factor_count,
    ) or not np.all(np.isfinite(values)):
        raise FloatingPointError("GROMM2 physical risk-price prediction is invalid")
    return np.asarray(values, dtype=np.float64)


def generate_gromm2_daily_scenarios(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    target: DailyDigitalTarget,
    market_state: Gromm2MarketState,
    config: GrommConfig = GrommConfig(),
) -> Gromm2DailyScenarios:
    """Generate one-session P or Q returns using the dated GROMM2 state."""
    require_public_hedging_panel(panel)
    if market_state.as_of_date != origin_date:
        raise ValueError("GROMM2 market state must share the scenario origin date")
    if config.measure is GrommMeasure.P and not market_state.physical_accepted:
        raise RuntimeError("GROMM2 P generation requires an accepted physical fit")
    if not jax.config.x64_enabled:
        raise RuntimeError("GROMM2 scenario generation requires JAX 64-bit mode")
    started = time.perf_counter()
    sample = build_daily_digital_sample(panel, origin_date=origin_date, target=target)
    option_prices = _option_values_for_quotes(
        market_state,
        densities=market_state.density_tower[None],
        background_level=np.ones(1),
        spot=np.ones(1),
        quotes=sample.hedge_quotes,
        origin_day=sample.transition.origin,
        elapsed_days=0,
    )[0]
    claim_model_price = price_gromm2_daily_target(
        market_state,
        sample=sample,
    )
    instrument_model_prices = np.concatenate(([1.0], option_prices))
    instrument_entry_prices = np.asarray(
        [1.0, *(quote.mark for quote in sample.hedge_quotes)],
        dtype=np.float64,
    )
    linear_costs = np.asarray(
        [
            1.0e-4,
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
        (config.path_count, len(instrument_model_prices)), dtype=np.float64
    )
    pure_instrument_returns = np.empty_like(instrument_returns)
    claim_values = np.empty(config.path_count, dtype=np.float64)
    terminal_costs = np.empty_like(instrument_returns)
    generator = np.random.default_rng(config.seed)
    daily_step = jax.jit(one_day_risk_neutral_step, static_argnames=("substeps",))
    intervals = np.diff(market_state.expiries_y, prepend=0.0)[1:]
    taming_total = 0.0
    completed = 0
    while completed < config.path_count:
        pair_count = config.batch_size // 2
        factor_half = generator.standard_normal(
            (config.dlv_substeps, pair_count, market_state.dynamics.factor_count)
        )
        risk_neutral_factor_normals = np.stack(
            (factor_half, -factor_half), axis=2
        ).reshape(
            config.dlv_substeps,
            config.batch_size,
            market_state.dynamics.factor_count,
        )
        factor_normals = risk_neutral_factor_normals.copy()
        if config.measure is GrommMeasure.P:
            factor_normals += (
                market_state.physical_market_price_of_risk[None, None, :]
                * math.sqrt(1.0 / (252.0 * config.dlv_substeps))
            )
        background_half = generator.standard_normal(pair_count)
        background_normals = np.stack(
            (background_half, -background_half), axis=1
        ).reshape(config.batch_size)
        state = SimulationState(
            front_density=jnp.broadcast_to(
                jnp.asarray(market_state.density_tower[0]),
                (config.batch_size, market_state.supports.shape[1]),
            ),
            dlvsq=jnp.broadcast_to(
                jnp.asarray(market_state.dlvsq),
                (config.batch_size,) + market_state.dlvsq.shape,
            ),
            background_level=jnp.ones(config.batch_size),
        )
        q_result = daily_step(
            jnp.asarray(market_state.supports),
            jnp.asarray(market_state.transports),
            jnp.asarray(intervals),
            state,
            jnp.asarray(market_state.dynamics.log_dlvsq_loadings),
            jnp.asarray(risk_neutral_factor_normals),
            jnp.asarray(background_normals),
            remaining_front_days=jnp.asarray(1),
            substeps=config.dlv_substeps,
        )
        result = (
            q_result
            if config.measure is GrommMeasure.Q
            else daily_step(
                jnp.asarray(market_state.supports),
                jnp.asarray(market_state.transports),
                jnp.asarray(intervals),
                state,
                jnp.asarray(market_state.dynamics.log_dlvsq_loadings),
                jnp.asarray(factor_normals),
                jnp.asarray(background_normals),
                remaining_front_days=jnp.asarray(1),
                substeps=config.dlv_substeps,
            )
        )
        terminal_spot = np.asarray(result.spot, dtype=np.float64)
        terminal_options = _option_values_for_quotes(
            market_state,
            densities=np.asarray(result.density_tower),
            background_level=np.asarray(result.state.background_level),
            spot=terminal_spot,
            quotes=sample.hedge_quotes,
            origin_day=sample.transition.origin,
            elapsed_days=1,
        )
        q_terminal_spot = np.asarray(q_result.spot, dtype=np.float64)
        q_terminal_options = _option_values_for_quotes(
            market_state,
            densities=np.asarray(q_result.density_tower),
            background_level=np.asarray(q_result.state.background_level),
            spot=q_terminal_spot,
            quotes=sample.hedge_quotes,
            origin_day=sample.transition.origin,
            elapsed_days=1,
        )
        terminal_claim = sample.target_discount_factor * _digital_probability(
            support=market_state.supports[target.maturity_business_days - 1],
            density=np.asarray(result.density_tower)[
                :, target.maturity_business_days - 1
            ],
            background_level=np.asarray(result.state.background_level),
            strike=sample.target_strike_over_forward,
            remaining_days=target.maturity_business_days - 1,
            right=target.right,
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
            terminal_options - instrument_entry_prices[None, 1:]
        )
        pure_instrument_returns[completed:stop, 0] = one_day_bridge * (
            q_terminal_spot - 1.0
        )
        pure_instrument_returns[completed:stop, 1:] = (
            q_terminal_options - option_prices[None]
        )
        claim_values[completed:stop] = terminal_claim
        terminal_costs[completed:stop] = _terminal_transaction_costs(
            terminal_spot=terminal_spot,
            quotes=sample.hedge_quotes,
            origin_day=sample.transition.origin,
            settles_next=settles_next,
        )
        taming_total += float(result.drift_taming_fraction) * config.batch_size
        completed = stop
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": "gromm2-generate-daily",
                    "measure": config.measure.value,
                    "completed_paths": completed,
                    "path_count": config.path_count,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed
                    * (config.path_count - completed)
                    / completed,
                }
            ),
            flush=True,
        )
    return Gromm2DailyScenarios(
        origin_date=origin_date,
        measure=config.measure,
        target=target,
        sample=sample,
        market_state=market_state,
        instrument_returns=instrument_returns,
        pure_instrument_returns=pure_instrument_returns,
        claim_returns=claim_values - sample.market_price,
        claim_terminal_values=claim_values,
        instrument_model_prices=instrument_model_prices,
        instrument_entry_prices=instrument_entry_prices,
        claim_model_price=claim_model_price,
        linear_transaction_costs=linear_costs,
        terminal_transaction_costs=terminal_costs,
        settles_next=settles_next,
        drift_taming_fraction=taming_total / config.path_count,
        generation_seconds=time.perf_counter() - started,
    )


def gromm2_daily_training_data(
    scenarios: Gromm2DailyScenarios,
) -> GrommTrainingData:
    """Convert dated GROMM2 scenarios to the common policy-training tensors."""
    state = scenarios.market_state
    density = state.density_tower[0]
    support = state.supports[0]
    mean = float(density @ support)
    centered = support - mean
    variance = float(density @ centered**2)
    skew = float(
        density @ centered**3 / max(variance, np.finfo(np.float64).tiny) ** 1.5
    )
    log_dlvsq = np.log(state.dlvsq[1:])
    expiry_indices = np.linspace(
        0, len(log_dlvsq) - 1, num=min(8, len(log_dlvsq)), dtype=np.int64
    )
    feature = np.asarray(
        [
            float(scenarios.measure is GrommMeasure.P),
            scenarios.target.maturity_business_days / 5.0,
            scenarios.target.nms,
            1.0 if scenarios.target.right is OptionRight.CALL else -1.0,
            scenarios.sample.transition.origin.atm_vol_at_business_dte(1),
            scenarios.sample.transition.origin.atm_vol_at_business_dte(5),
            state.percent_inside_spread / 100.0,
            state.explained_variance_ratio,
            state.physical_projection_fraction,
            float(np.linalg.norm(state.physical_market_price_of_risk) / math.sqrt(252.0)),
            mean,
            variance,
            skew,
            *np.mean(log_dlvsq[expiry_indices], axis=1),
        ],
        dtype=np.float64,
    )
    features = np.broadcast_to(
        feature, (scenarios.path_count, 1, len(feature))
    ).copy()
    return GrommTrainingData(
        features=features,
        instrument_returns=scenarios.instrument_returns[:, None],
        claim_returns=scenarios.claim_returns[:, None],
        active_mask=np.ones((1, scenarios.instrument_returns.shape[1]), dtype=bool),
        linear_transaction_costs=scenarios.linear_transaction_costs[None],
        settles_next=scenarios.settles_next[None],
        terminal_transaction_costs=scenarios.terminal_transaction_costs,
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
    costs[:, 0] = 1.0e-4 * discount * spot_over_origin
    for option_index, quote in enumerate(quotes, start=1):
        if settles_next[option_index]:
            continue
        initial_dte = int(round(quote.business_dte))
        remaining_dte = initial_dte - 1
        fixed_strike_over_origin = (
            quote.strike_over_forward
            * origin_day.forward_at_business_dte(initial_dte)
        )
        costs[:, option_index] = np.asarray(
            [
                discount
                * path_spot
                * origin_day.half_spread_for_fixed_contract(
                    business_dte=remaining_dte,
                    fixed_strike_over_current_spot=(
                        fixed_strike_over_origin / path_spot
                    ),
                    is_call=quote.contract.right is OptionRight.CALL,
                )
                for path_spot in spot_over_origin
            ],
            dtype=np.float64,
        )
    return costs


def _option_values_for_quotes(
    state: Gromm2MarketState,
    *,
    densities: FloatArray,
    background_level: FloatArray,
    spot: FloatArray,
    quotes: tuple[OptionQuote, ...],
    origin_day: PublicMarketDay,
    elapsed_days: int,
) -> FloatArray:
    path_count = len(densities)
    values = np.empty((path_count, len(quotes)), dtype=np.float64)
    for dte, indices in _quote_groups(quotes).items():
        model_index = _expiry_index(state.business_dtes, dte)
        strikes = np.asarray(
            [quotes[index].strike_over_forward for index in indices], dtype=np.float64
        )
        remaining_days = dte - elapsed_days
        calls = np.asarray(
            price_active_calls(
                jnp.asarray(state.supports[model_index : model_index + 1]),
                jnp.asarray(densities[:, model_index : model_index + 1]),
                jnp.asarray(background_level),
                jnp.asarray([remaining_days / 252.0]),
                jnp.asarray(strikes[None]),
            )
        )[:, 0]
        puts = calls - np.asarray(spot)[:, None] + strikes[None]
        rights = np.asarray(
            [quotes[index].contract.right is OptionRight.CALL for index in indices]
        )
        normalized = np.where(rights[None], calls, puts)
        bridge = origin_day.discount_at_business_dte(
            dte
        ) * origin_day.forward_at_business_dte(dte)
        values[:, np.asarray(indices)] = bridge * np.maximum(normalized, 0.0)
    if np.any(~np.isfinite(values)) or np.min(values) < -2.0e-10:
        raise FloatingPointError("GROMM2 option values are invalid")
    return values


def _digital_probability(
    *,
    support: FloatArray,
    density: FloatArray,
    background_level: FloatArray,
    strike: float,
    remaining_days: int,
    right: OptionRight,
) -> FloatArray:
    latent = np.asarray(background_level)[:, None] * np.asarray(support)[None]
    if remaining_days == 0:
        call_kernel = latent > strike
    else:
        variance = 0.05**2 * remaining_days / 252.0
        score = (np.log(latent / strike) - 0.5 * variance) / math.sqrt(variance)
        call_kernel = scipy.special.ndtr(score)
    call = np.einsum("pn,pn->p", np.asarray(density), call_kernel)
    return call if right is OptionRight.CALL else 1.0 - call


def _expiry_index(business_dtes: FloatArray, dte: int) -> int:
    index = int(np.searchsorted(business_dtes, dte))
    if index == len(business_dtes) or not math.isclose(
        float(business_dtes[index]), dte, abs_tol=2.0e-6
    ):
        raise ValueError(f"GROMM2 geometry does not contain {dte} business days")
    return index


def _quote_groups(quotes: tuple[OptionQuote, ...]) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {}
    for index, quote in enumerate(quotes):
        dte = int(round(quote.business_dte))
        if dte < 1 or not math.isclose(quote.business_dte, dte, abs_tol=2.0e-6):
            raise ValueError("GROMM2 requires integer business-day hedge expiries")
        groups.setdefault(dte, []).append(index)
    return {dte: tuple(indices) for dte, indices in groups.items()}
