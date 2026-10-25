"""Five-session frozen-book scenarios from one dated rolling GROMM2 state."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from quantdata.gromm2.dynamics import (
    floating_level_surface,
    interpolate_loading_expiries,
)
from quantdata.gromm2.pricing import price_active_calls
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step
from quantdata.hedging.contracts import OptionRight
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WEEKLY_HORIZON,
    WeeklyClaimDefinition,
    WeeklyTargetKind,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyScenarioBatch,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    Gromm2MarketState,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.training import (
    GrommTrainingData,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, kw_only=True)
class Gromm2WeeklyScenarios:
    """Weekly scenarios and their path-step policy features."""

    scenarios: WeeklyScenarioBatch
    features: FloatArray
    pure_instrument_returns: FloatArray
    initial_positions: FloatArray
    drift_taming_fraction: float
    generation_seconds: float

    def __post_init__(self) -> None:
        expected = (
            self.scenarios.scenario_count,
            self.scenarios.step_count,
        )
        if self.features.ndim != 3 or self.features.shape[:2] != expected:
            raise ValueError("GROMM2 weekly features do not align with scenarios")
        if self.pure_instrument_returns.shape != (
            *expected,
            1 + self.scenarios.contract_count,
        ):
            raise ValueError("GROMM2 weekly pure returns do not align")
        if self.initial_positions.shape != (1 + self.scenarios.contract_count,):
            raise ValueError("GROMM2 weekly initial positions do not align")
        if (
            not np.all(np.isfinite(self.features))
            or not np.all(np.isfinite(self.pure_instrument_returns))
            or not np.all(np.isfinite(self.initial_positions))
            or not math.isfinite(self.drift_taming_fraction)
            or not math.isfinite(self.generation_seconds)
        ):
            raise ValueError("GROMM2 weekly output contains non-finite values")


def generate_gromm2_weekly_scenarios(
    *,
    market_state: Gromm2MarketState,
    state: WeeklyDecisionState,
    claim: WeeklyClaimDefinition,
    config: GrommConfig,
    model_fit_identity: str,
) -> Gromm2WeeklyScenarios:
    """Generate a remaining-horizon P or Q frozen-contract scenario panel."""
    if market_state.as_of_date != state.instruction.fit_as_of_date:
        raise ValueError("GROMM2 weekly state and market fit dates differ")
    if state.market_day.trading_date != state.instruction.decision_date:
        raise ValueError("GROMM2 weekly decision state has the wrong date")
    if not model_fit_identity.strip():
        raise ValueError("GROMM2 weekly model-fit identity is required")
    if not jax.config.x64_enabled:
        raise RuntimeError("GROMM2 weekly simulation requires JAX 64-bit mode")
    remaining_steps = WEEKLY_HORIZON - state.decision_index
    if not 1 <= remaining_steps <= WEEKLY_HORIZON:
        raise ValueError("GROMM2 weekly remaining horizon is invalid")

    started = time.perf_counter()
    day = state.market_day
    contract_count = len(state.contracts)
    instrument_count = contract_count + 1
    current_scale = state.discount_to_origin * state.spot_over_episode_origin
    if not math.isfinite(current_scale) or current_scale <= 0.0:
        raise ValueError("GROMM2 weekly current PV scale is invalid")
    contract_geometry = _contract_geometry(state)
    initial_option_marks = np.asarray(state.option_marks_pv, dtype=np.float64)
    initial_model_option_marks = _price_options(
        market_state=market_state,
        density_tower=market_state.density_tower[None],
        background_level=np.ones(1),
        normalized_spot=np.ones(1),
        front_index=0,
        elapsed_days=0,
        contract_geometry=contract_geometry,
        current_scale=current_scale,
    )[0]

    scenario_count = config.path_count
    features: FloatArray | None = None
    inception_paths = np.empty(
        (scenario_count, remaining_steps + 1), dtype=np.float64
    )
    spot_values = np.empty_like(inception_paths)
    spot_gains = np.empty((scenario_count, remaining_steps), dtype=np.float64)
    option_marks = np.zeros(
        (scenario_count, remaining_steps + 1, contract_count), dtype=np.float64
    )
    option_spreads = np.zeros_like(option_marks)
    option_gains = np.zeros(
        (scenario_count, remaining_steps, contract_count), dtype=np.float64
    )
    pure_spot_gains = np.empty((scenario_count, remaining_steps), dtype=np.float64)
    pure_option_gains = np.zeros_like(option_gains)
    terminal_claims = np.empty(scenario_count, dtype=np.float64)
    option_alive, option_settles_next = _option_masks(
        state=state,
        remaining_business_dtes=contract_geometry.remaining_business_dtes,
        remaining_steps=remaining_steps,
    )
    inception_paths[:, 0] = state.inception_forward_path[-1]
    spot_values[:, 0] = current_scale
    option_marks[:, 0] = initial_option_marks[None]
    option_spreads[:, 0] = state.option_half_spreads_pv[None]

    generator = np.random.default_rng(config.seed)
    daily_step = jax.jit(one_day_risk_neutral_step, static_argnames=("substeps",))
    intervals = np.diff(market_state.expiries_y, prepend=0.0)
    reference_factor_dtes = market_state.business_dtes[1:]
    taming_total = 0.0
    completed = 0
    while completed < scenario_count:
        count = min(config.batch_size, scenario_count - completed)
        if count % 2:
            raise ValueError("GROMM2 weekly batches must preserve antithetic pairs")
        factor_normals, background_normals = _weekly_antithetic_normals(
            generator=generator,
            day_count=remaining_steps,
            substeps=config.dlv_substeps,
            pair_count=count // 2,
            factor_count=market_state.dynamics.factor_count,
        )
        full_state = SimulationState(
            front_density=jnp.broadcast_to(
                jnp.asarray(market_state.density_tower[0]),
                (count, market_state.supports.shape[1]),
            ),
            dlvsq=jnp.broadcast_to(
                jnp.asarray(market_state.dlvsq),
                (count,) + market_state.dlvsq.shape,
            ),
            background_level=jnp.ones(count),
        )
        current_normalized_spot = np.ones(count, dtype=np.float64)
        current_episode_spot = np.full(
            count, state.spot_over_episode_origin, dtype=np.float64
        )
        path_prefix = np.broadcast_to(
            state.inception_forward_path,
            (count, len(state.inception_forward_path)),
        ).copy()
        batch_marks = np.broadcast_to(
            initial_option_marks,
            (count, contract_count),
        ).copy()
        batch_model_marks = np.broadcast_to(
            initial_model_option_marks,
            (count, contract_count),
        ).copy()
        batch_spreads = np.broadcast_to(
            state.option_half_spreads_pv,
            (count, contract_count),
        ).copy()
        previous_spot_pv = np.full(count, current_scale, dtype=np.float64)
        batch_features: list[FloatArray] = []
        initial_future_dlvsq = np.broadcast_to(
            state.market_day.dlvsq[1:],
            (count,) + state.market_day.dlvsq[1:].shape,
        )
        batch_features.append(
            _weekly_features(
                market_state=market_state,
                measure=config.measure,
                claim=claim,
                decision_index=state.decision_index,
                path_prefix=path_prefix,
                future_dlvsq=initial_future_dlvsq,
                spot_values_pv=np.full(count, current_scale),
                option_marks_pv=batch_marks,
                option_spreads_pv=batch_spreads,
                option_alive=option_alive[0],
                initial_positions=_initial_positions(state),
            )
        )
        front_index = 0
        for local_day in range(1, remaining_steps + 1):
            while market_state.business_dtes[front_index] < local_day:
                front_index += 1
            remaining_front_days = int(
                market_state.business_dtes[front_index] - (local_day - 1)
            )
            future_remaining_dtes = (
                market_state.business_dtes[front_index + 1 :]
                - (local_day - 1)
            )
            active_loadings = interpolate_loading_expiries(
                market_state.dynamics.log_dlvsq_loadings,
                reference_factor_dtes,
                future_remaining_dtes,
            )
            risk_neutral_factors = factor_normals[local_day - 1]
            current_factors = risk_neutral_factors.copy()
            if config.measure is GrommMeasure.P:
                if not market_state.physical_accepted:
                    raise RuntimeError(
                        "GROMM2 P generation requires an accepted physical fit"
                    )
                current_factors += market_state.physical_market_price_of_risk[
                    None
                ] * math.sqrt(
                    1.0 / (252.0 * config.dlv_substeps)
                )
            current_state = SimulationState(
                front_density=full_state.front_density,
                dlvsq=full_state.dlvsq[..., front_index:, :],
                background_level=full_state.background_level,
            )
            q_result = daily_step(
                jnp.asarray(market_state.supports[front_index:]),
                jnp.asarray(market_state.transports[front_index:]),
                jnp.asarray(intervals[front_index + 1 :]),
                current_state,
                jnp.asarray(active_loadings)[None],
                jnp.asarray(risk_neutral_factors),
                jnp.asarray(background_normals[local_day - 1]),
                remaining_front_days=jnp.asarray(remaining_front_days),
                substeps=config.dlv_substeps,
            )
            result = (
                q_result
                if config.measure is GrommMeasure.Q
                else daily_step(
                    jnp.asarray(market_state.supports[front_index:]),
                    jnp.asarray(market_state.transports[front_index:]),
                    jnp.asarray(intervals[front_index + 1 :]),
                    current_state,
                    jnp.asarray(active_loadings)[None],
                    jnp.asarray(current_factors),
                    jnp.asarray(background_normals[local_day - 1]),
                    remaining_front_days=jnp.asarray(remaining_front_days),
                    substeps=config.dlv_substeps,
                )
            )
            normalized_spot = np.asarray(result.spot, dtype=np.float64)
            q_normalized_spot = np.asarray(q_result.spot, dtype=np.float64)
            forward = day.forward_at_business_dte(local_day)
            discount = day.discount_at_business_dte(local_day)
            episode_spot = state.spot_over_episode_origin * forward * normalized_spot
            spot_pv = state.discount_to_origin * discount * episode_spot
            q_episode_spot = (
                state.spot_over_episode_origin * forward * q_normalized_spot
            )
            q_spot_pv = state.discount_to_origin * discount * q_episode_spot
            global_index = state.decision_index + local_day
            inception_spot = (
                episode_spot
                / state.inception_forward_over_episode_spot[global_index]
            )
            path_prefix = np.column_stack((path_prefix, inception_spot))
            next_marks = _price_options(
                market_state=market_state,
                density_tower=np.asarray(result.density_tower),
                background_level=np.asarray(result.state.background_level),
                normalized_spot=normalized_spot,
                front_index=front_index,
                elapsed_days=local_day,
                contract_geometry=contract_geometry,
                current_scale=current_scale,
            )
            q_next_marks = _price_options(
                market_state=market_state,
                density_tower=np.asarray(q_result.density_tower),
                background_level=np.asarray(q_result.state.background_level),
                normalized_spot=q_normalized_spot,
                front_index=front_index,
                elapsed_days=local_day,
                contract_geometry=contract_geometry,
                current_scale=current_scale,
            )
            next_spreads = _pathwise_spreads(
                state=state,
                contract_geometry=contract_geometry,
                episode_spot=episode_spot,
                local_day=local_day,
            )
            output_slice = slice(completed, completed + count)
            inception_paths[output_slice, local_day] = inception_spot
            spot_values[output_slice, local_day] = spot_pv
            spot_gains[output_slice, local_day - 1] = (
                spot_pv - previous_spot_pv
            )
            pure_spot_gains[output_slice, local_day - 1] = (
                q_spot_pv - previous_spot_pv
            )
            option_marks[output_slice, local_day] = next_marks
            option_spreads[output_slice, local_day] = next_spreads
            settling = option_settles_next[local_day - 1]
            if np.any(settling):
                settling_indices = np.flatnonzero(settling)
                settlement = np.column_stack(
                    tuple(
                        state.discount_to_origin
                        * discount
                        * np.asarray(
                            [
                                state.contracts[int(contract_index)].payoff(path_spot)
                                for path_spot in episode_spot
                            ],
                            dtype=np.float64,
                        )
                        for contract_index in settling_indices
                    )
                )
                option_gains[output_slice, local_day - 1, settling] = (
                    settlement - batch_marks[:, settling]
                )
                q_settlement = np.column_stack(
                    tuple(
                        state.discount_to_origin
                        * discount
                        * np.asarray(
                            [
                                state.contracts[int(contract_index)].payoff(path_spot)
                                for path_spot in q_episode_spot
                            ],
                            dtype=np.float64,
                        )
                        for contract_index in settling_indices
                    )
                )
                pure_option_gains[
                    output_slice, local_day - 1, settling
                ] = q_settlement - batch_model_marks[:, settling]
            surviving = option_alive[local_day]
            option_gains[output_slice, local_day - 1, surviving] = (
                next_marks[:, surviving] - batch_marks[:, surviving]
            )
            pure_option_gains[output_slice, local_day - 1, surviving] = (
                q_next_marks[:, surviving] - batch_model_marks[:, surviving]
            )
            full_state = SimulationState(
                front_density=result.state.front_density,
                dlvsq=full_state.dlvsq.at[..., front_index:, :].set(
                    result.state.dlvsq
                ),
                background_level=result.state.background_level,
            )
            current_normalized_spot = normalized_spot
            current_episode_spot = episode_spot
            previous_spot_pv = spot_pv
            batch_marks = next_marks
            batch_model_marks = next_marks
            batch_spreads = next_spreads
            taming_total += float(result.drift_taming_fraction) * count
            if local_day < remaining_steps:
                next_front_index = front_index
                while (
                    market_state.business_dtes[next_front_index] < local_day + 1
                ):
                    next_front_index += 1
                floating = _floating_future_dlvsq(
                    full_dlvsq=full_state.dlvsq,
                    front_index=next_front_index,
                    elapsed_days=local_day,
                    business_dtes=market_state.business_dtes,
                    reference_factor_dtes=reference_factor_dtes,
                )
                batch_features.append(
                    _weekly_features(
                        market_state=market_state,
                        measure=config.measure,
                        claim=claim,
                        decision_index=global_index,
                        path_prefix=path_prefix,
                        future_dlvsq=floating,
                        spot_values_pv=spot_pv,
                        option_marks_pv=next_marks,
                        option_spreads_pv=next_spreads,
                        option_alive=option_alive[local_day],
                        initial_positions=_initial_positions(state),
                    )
                )
        terminal_discount = state.discount_to_origin * day.discount_at_business_dte(
            remaining_steps
        )
        terminal_claims[completed : completed + count] = terminal_discount * np.asarray(
            [claim.claim.payoff(path) for path in path_prefix], dtype=np.float64
        )
        batch_feature_array = np.stack(batch_features, axis=1)
        if features is None:
            features = np.empty(
                (scenario_count,) + batch_feature_array.shape[1:], dtype=np.float64
            )
        features[completed : completed + count] = batch_feature_array
        completed += count
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": "gromm2-generate-weekly",
                    "measure": config.measure.value,
                    "completed_paths": completed,
                    "path_count": scenario_count,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed * (scenario_count - completed) / completed,
                }
            ),
            flush=True,
        )
    assert features is not None
    scenarios = WeeklyScenarioBatch(
        inception_forward_paths=inception_paths,
        spot_values_pv=spot_values,
        spot_step_gains=spot_gains,
        option_step_gains=option_gains,
        option_half_spreads_pv=option_spreads,
        option_alive=option_alive,
        option_settles_next=option_settles_next,
        terminal_claim_values_pv=terminal_claims,
        contract_ids=tuple(contract.contract_id for contract in state.contracts),
        target_label=claim.target.label,
        fit_as_of_date=state.instruction.fit_as_of_date,
        model_fit_identity=model_fit_identity,
        measure=config.measure.value,
        seed=config.seed,
    )
    return Gromm2WeeklyScenarios(
        scenarios=scenarios,
        features=features,
        pure_instrument_returns=np.concatenate(
            (pure_spot_gains[..., None], pure_option_gains),
            axis=-1,
        ),
        initial_positions=_initial_positions(state),
        drift_taming_fraction=taming_total / (scenario_count * remaining_steps),
        generation_seconds=time.perf_counter() - started,
    )


def gromm2_weekly_training_data(
    generated: Gromm2WeeklyScenarios,
    *,
    claim_quantity: float,
) -> GrommTrainingData:
    """Convert weekly scenarios to the recurrent deep-hedging training schema."""
    if not math.isfinite(claim_quantity) or claim_quantity <= 0.0:
        raise ValueError("weekly GROMM claim quantity must be finite-positive")
    scenarios = generated.scenarios
    path_count = scenarios.scenario_count
    step_count = scenarios.step_count
    instrument_returns = np.concatenate(
        (scenarios.spot_step_gains[..., None], scenarios.option_step_gains),
        axis=-1,
    )
    claim_returns = np.zeros((path_count, step_count), dtype=np.float64)
    claim_returns[:, -1] = claim_quantity * scenarios.terminal_claim_values_pv
    costs = np.concatenate(
        (
            np.zeros_like(scenarios.spot_values_pv[:, :-1, None]),
            scenarios.option_half_spreads_pv[:, :-1],
        ),
        axis=-1,
    )
    active = np.concatenate(
        (
            np.ones((step_count, 1), dtype=bool),
            scenarios.option_alive[:-1],
        ),
        axis=1,
    )
    settles = np.concatenate(
        (
            np.zeros((step_count, 1), dtype=bool),
            scenarios.option_settles_next,
        ),
        axis=1,
    )
    return GrommTrainingData(
        features=generated.features,
        instrument_returns=instrument_returns,
        claim_returns=claim_returns,
        active_mask=active,
        linear_transaction_costs=costs,
        initial_positions=generated.initial_positions,
        settles_next=settles,
        terminal_transaction_costs=None,
    )


def live_gromm2_weekly_feature(
    *,
    market_state: Gromm2MarketState,
    state: WeeklyDecisionState,
    claim: WeeklyClaimDefinition,
    measure: GrommMeasure,
    initial_positions: FloatArray,
) -> FloatArray:
    """Encode one observed weekly decision in the fitted policy feature space."""
    instrument_spot_pv = state.discount_to_origin * state.spot_over_episode_origin
    return _weekly_features(
        market_state=market_state,
        measure=measure,
        claim=claim,
        decision_index=state.decision_index,
        path_prefix=state.inception_forward_path[None],
        future_dlvsq=state.market_day.dlvsq[None, 1:],
        spot_values_pv=np.asarray([instrument_spot_pv]),
        option_marks_pv=state.option_marks_pv[None],
        option_spreads_pv=state.option_half_spreads_pv[None],
        option_alive=state.option_alive,
        initial_positions=initial_positions,
    )[0]


@dataclass(frozen=True, kw_only=True)
class _ContractGeometry:
    remaining_business_dtes: NDArray[np.int64]
    strike_over_fit_forward: FloatArray
    fixed_strike_over_episode_spot: FloatArray
    price_bridge_over_fit_spot: FloatArray
    is_call: BoolArray


def _contract_geometry(state: WeeklyDecisionState) -> _ContractGeometry:
    snapshot = state.market_day.reprice_contracts(
        state.contracts,
        spot_over_episode_origin=state.spot_over_episode_origin,
    )
    quotes = snapshot.quote_by_id()
    count = len(state.contracts)
    dtes = np.zeros(count, dtype=np.int64)
    strikes = np.ones(count, dtype=np.float64)
    fixed = np.asarray(
        [contract.fixed_strike for contract in state.contracts], dtype=np.float64
    )
    bridges = np.zeros(count, dtype=np.float64)
    is_call = np.asarray(
        [contract.right is OptionRight.CALL for contract in state.contracts],
        dtype=bool,
    )
    for index in np.flatnonzero(state.option_alive):
        quote = quotes[state.contracts[int(index)].contract_id]
        rounded = int(round(quote.business_dte))
        if not math.isclose(quote.business_dte, rounded, abs_tol=2.0e-6):
            raise ValueError("GROMM2 weekly options require integer DTE")
        dtes[index] = rounded
        strikes[index] = quote.strike_over_forward
        bridges[index] = (
            state.market_day.discount_at_business_dte(rounded)
            * state.market_day.forward_at_business_dte(rounded)
        )
    return _ContractGeometry(
        remaining_business_dtes=dtes,
        strike_over_fit_forward=strikes,
        fixed_strike_over_episode_spot=fixed,
        price_bridge_over_fit_spot=bridges,
        is_call=is_call,
    )


def _option_masks(
    *,
    state: WeeklyDecisionState,
    remaining_business_dtes: NDArray[np.int64],
    remaining_steps: int,
) -> tuple[BoolArray, BoolArray]:
    alive = np.zeros((remaining_steps + 1, len(state.contracts)), dtype=bool)
    settles = np.zeros((remaining_steps, len(state.contracts)), dtype=bool)
    alive[0] = state.option_alive
    for step in range(1, remaining_steps + 1):
        alive[step] = state.option_alive & (remaining_business_dtes > step)
        settles[step - 1] = state.option_alive & (remaining_business_dtes == step)
    return alive, settles


def _price_options(
    *,
    market_state: Gromm2MarketState,
    density_tower: FloatArray,
    background_level: FloatArray,
    normalized_spot: FloatArray,
    front_index: int,
    elapsed_days: int,
    contract_geometry: _ContractGeometry,
    current_scale: float,
) -> FloatArray:
    path_count = len(density_tower)
    values = np.zeros(
        (path_count, len(contract_geometry.remaining_business_dtes)),
        dtype=np.float64,
    )
    for dte in np.unique(contract_geometry.remaining_business_dtes):
        if dte <= elapsed_days:
            continue
        indices = np.flatnonzero(contract_geometry.remaining_business_dtes == dte)
        model_index = int(np.searchsorted(market_state.business_dtes, dte))
        if (
            model_index == len(market_state.business_dtes)
            or not math.isclose(market_state.business_dtes[model_index], dte)
            or model_index < front_index
        ):
            raise ValueError(f"GROMM2 weekly model has no expiry at {dte} days")
        relative_index = model_index - front_index
        strikes = contract_geometry.strike_over_fit_forward[indices]
        calls = np.asarray(
            price_active_calls(
                jnp.asarray(market_state.supports[model_index : model_index + 1]),
                jnp.asarray(density_tower[:, relative_index : relative_index + 1]),
                jnp.asarray(background_level),
                jnp.asarray([(dte - elapsed_days) / 252.0]),
                jnp.asarray(strikes[None]),
            )
        )[:, 0]
        puts = calls - normalized_spot[:, None] + strikes[None]
        normalized = np.where(
            contract_geometry.is_call[indices][None], calls, puts
        )
        values[:, indices] = (
            current_scale
            * contract_geometry.price_bridge_over_fit_spot[indices][None]
            * np.maximum(normalized, 0.0)
        )
    return values


def _pathwise_spreads(
    *,
    state: WeeklyDecisionState,
    contract_geometry: _ContractGeometry,
    episode_spot: FloatArray,
    local_day: int,
) -> FloatArray:
    values = np.zeros(
        (len(episode_spot), len(contract_geometry.remaining_business_dtes)),
        dtype=np.float64,
    )
    discount = state.discount_to_origin * state.market_day.discount_at_business_dte(
        local_day
    )
    for index, initial_dte in enumerate(contract_geometry.remaining_business_dtes):
        remaining = int(initial_dte) - local_day
        if remaining <= 0:
            continue
        values[:, index] = (
            discount
            * episode_spot
            * state.market_day.half_spreads_for_fixed_contracts(
                business_dte=remaining,
                fixed_strikes_over_current_spot=(
                    contract_geometry.fixed_strike_over_episode_spot[index]
                    / episode_spot
                ),
                is_call=bool(contract_geometry.is_call[index]),
            )
        )
    return values


def _weekly_features(
    *,
    market_state: Gromm2MarketState,
    measure: GrommMeasure,
    claim: WeeklyClaimDefinition,
    decision_index: int,
    path_prefix: FloatArray,
    future_dlvsq: FloatArray,
    spot_values_pv: FloatArray,
    option_marks_pv: FloatArray,
    option_spreads_pv: FloatArray,
    option_alive: BoolArray,
    initial_positions: FloatArray,
) -> FloatArray:
    path_count = len(path_prefix)
    if future_dlvsq.shape[0] != path_count:
        raise ValueError("GROMM2 weekly DLV features do not align with paths")
    log_levels = np.log(future_dlvsq)
    indices = np.linspace(
        0, log_levels.shape[-2] - 1, num=min(8, log_levels.shape[-2]), dtype=int
    )
    level_features = np.mean(log_levels[:, indices], axis=-1)
    returns = np.diff(path_prefix, axis=1) / path_prefix[:, :-1]
    last_return = (
        np.zeros(path_count)
        if returns.shape[1] == 0
        else returns[:, -1]
    )
    realized_variance = np.sum(returns**2, axis=1) / WEEKLY_HORIZON
    target_one_hot = np.zeros((path_count, len(WeeklyTargetKind)), dtype=np.float64)
    target_index = tuple(WeeklyTargetKind).index(claim.target.kind)
    target_one_hot[:, target_index] = 1.0
    right = (
        0.0
        if claim.target.right is None
        else (1.0 if claim.target.right is OptionRight.CALL else -1.0)
    )
    nms = 0.0 if claim.target.nms is None else claim.target.nms
    risk_price_norm = float(
        np.linalg.norm(market_state.physical_market_price_of_risk)
        / math.sqrt(252.0)
    )
    scalar = np.column_stack(
        (
            np.full(path_count, measure is GrommMeasure.P),
            np.full(path_count, decision_index / WEEKLY_HORIZON),
            np.full(path_count, (WEEKLY_HORIZON - decision_index) / WEEKLY_HORIZON),
            target_one_hot,
            np.full(path_count, right),
            np.full(path_count, nms),
            path_prefix[:, -1],
            last_return,
            realized_variance,
            np.min(path_prefix, axis=1),
            np.max(path_prefix, axis=1),
            np.full(path_count, market_state.percent_inside_spread / 100.0),
            np.full(path_count, market_state.explained_variance_ratio),
            np.full(path_count, market_state.physical_projection_fraction),
            np.full(path_count, risk_price_norm),
        )
    )
    instrument_marks = np.concatenate(
        (spot_values_pv[:, None], option_marks_pv), axis=1
    ) / spot_values_pv[:, None]
    instrument_spreads = np.concatenate(
        (np.zeros_like(spot_values_pv[:, None]), option_spreads_pv), axis=1
    ) / spot_values_pv[:, None]
    active = np.broadcast_to(
        np.concatenate(([True], option_alive)).astype(np.float64),
        instrument_marks.shape,
    )
    carried = np.broadcast_to(initial_positions, instrument_marks.shape)
    return np.concatenate(
        (
            scalar,
            level_features,
            instrument_marks,
            instrument_spreads,
            active,
            carried,
        ),
        axis=1,
    )


def _weekly_antithetic_normals(
    *,
    generator: np.random.Generator,
    day_count: int,
    substeps: int,
    pair_count: int,
    factor_count: int,
) -> tuple[FloatArray, FloatArray]:
    factor_half = generator.standard_normal(
        (day_count, substeps, pair_count, factor_count)
    )
    background_half = generator.standard_normal((day_count, pair_count))
    factors = np.stack((factor_half, -factor_half), axis=3).reshape(
        day_count, substeps, 2 * pair_count, factor_count
    )
    backgrounds = np.stack((background_half, -background_half), axis=2).reshape(
        day_count, 2 * pair_count
    )
    return factors, backgrounds


def _floating_future_dlvsq(
    *,
    full_dlvsq: jax.Array,
    front_index: int,
    elapsed_days: int,
    business_dtes: FloatArray,
    reference_factor_dtes: FloatArray,
) -> FloatArray:
    """Map an aging fixed-expiry state back to the fitted factor DTE grid."""
    start = front_index + 1
    if start >= full_dlvsq.shape[-2]:
        start = front_index
    active = full_dlvsq[..., start:, :]
    active_dtes = np.asarray(business_dtes[start:] - elapsed_days, dtype=np.float64)
    if len(active_dtes) == 0 or np.any(active_dtes <= 0.0):
        raise ValueError("GROMM2 weekly floating DLV grid is empty or expired")
    return np.exp(
        np.asarray(
            floating_level_surface(
                active,
                jnp.asarray(active_dtes),
                jnp.asarray(reference_factor_dtes),
            ),
            dtype=np.float64,
        )
    )


def _initial_positions(state: WeeklyDecisionState) -> FloatArray:
    return np.concatenate(
        (
            np.asarray([state.previous_spot_position], dtype=np.float64),
            np.asarray(state.previous_option_positions, dtype=np.float64),
        )
    )
