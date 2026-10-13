"""Historical SANOS remaining-horizon blocks for weekly policies."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS, add_business_days
from quantdata.hedging.contracts import OptionContract, OptionRight
from quantdata.hedging.data.pricing import price_dlv_surface, price_sanos_surface
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyClaimDefinition,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyScenarioBatch,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)

FloatArray = NDArray[np.float64]


class HistoricalOptionGeometry(StrEnum):
    """Released interpolation used to replay historical fixed contracts."""

    SANOS = "sanos"
    DLV = "dlv"


@dataclass(frozen=True, kw_only=True)
class HistoricalWeeklyTransitionHistory:
    """Target-independent one-session history available at a fit date."""

    as_of_date: date
    transitions: tuple[OneDayTransition, ...]
    endpoints: tuple[PublicMarketDay, ...]

    def __post_init__(self) -> None:
        if not self.transitions or len(self.endpoints) != len(self.transitions):
            raise ValueError("historical weekly transitions must be non-empty and aligned")
        if any(
            endpoint.trading_date != transition.next_date
            for transition, endpoint in zip(
                self.transitions,
                self.endpoints,
                strict=True,
            )
        ):
            raise ValueError("historical weekly transition endpoints do not align")
        if any(endpoint.trading_date > self.as_of_date for endpoint in self.endpoints):
            raise ValueError("historical weekly history extends after its as-of date")
        origins = tuple(transition.origin.trading_date for transition in self.transitions)
        if tuple(sorted(set(origins))) != origins:
            raise ValueError("historical weekly transition origins must be increasing")


@dataclass(frozen=True, kw_only=True)
class HistoricalWeeklyScenarioSet:
    """Scenario tensor plus its auditable historical block dates."""

    scenarios: WeeklyScenarioBatch
    training_origin_dates: tuple[date, ...]
    training_terminal_dates: tuple[date, ...]
    terminal_claim_pv_scales: FloatArray

    def __post_init__(self) -> None:
        if len(self.training_origin_dates) != self.scenarios.scenario_count or len(
            self.training_terminal_dates
        ) != self.scenarios.scenario_count:
            raise ValueError("historical weekly scenario dates do not align")
        if any(
            terminal <= origin or terminal > self.scenarios.fit_as_of_date
            for origin, terminal in zip(
                self.training_origin_dates,
                self.training_terminal_dates,
                strict=True,
            )
        ):
            raise ValueError("historical weekly scenario dates violate fit timing")
        if self.terminal_claim_pv_scales.shape != (self.scenarios.scenario_count,):
            raise ValueError("historical weekly claim scales do not align")
        if np.any(~np.isfinite(self.terminal_claim_pv_scales)) or np.any(
            self.terminal_claim_pv_scales <= 0.0
        ):
            raise ValueError("historical weekly claim scales must be finite-positive")


def attach_historical_weekly_claim(
    reference: HistoricalWeeklyScenarioSet,
    *,
    state: WeeklyDecisionState,
    claim: WeeklyClaimDefinition,
) -> HistoricalWeeklyScenarioSet:
    """Attach one payoff to cached historical paths without repricing options."""
    reference_batch = reference.scenarios
    current_path = np.asarray(state.inception_forward_path, dtype=np.float64)
    if current_path.ndim != 1 or len(current_path) != state.decision_index + 1:
        raise ValueError("historical weekly current target path does not align")
    if not np.allclose(
        reference_batch.inception_forward_paths[:, 0],
        current_path[-1],
        atol=1.0e-14,
        rtol=1.0e-14,
    ):
        raise ValueError("historical weekly cached paths changed decision state")
    prefix = np.broadcast_to(
        current_path[:-1],
        (reference_batch.scenario_count, len(current_path) - 1),
    )
    full_paths = np.concatenate(
        (prefix, reference_batch.inception_forward_paths),
        axis=1,
    )
    terminal_claims = reference.terminal_claim_pv_scales * np.asarray(
        [claim.claim.payoff(path) for path in full_paths],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(terminal_claims)) or np.any(terminal_claims < 0.0):
        raise FloatingPointError("historical weekly target payoffs are invalid")
    return replace(
        reference,
        scenarios=replace(
            reference.scenarios,
            terminal_claim_values_pv=terminal_claims,
            target_label=claim.target.label,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class _RelativeWeeklyOption:
    contract_index: int
    contract_id: str
    business_dte: int
    nms: float
    right: OptionRight


def build_historical_weekly_scenarios(
    history: HistoricalWeeklyTransitionHistory,
    *,
    state: WeeklyDecisionState,
    claim: WeeklyClaimDefinition,
    model_fit_identity: str,
    option_geometry: HistoricalOptionGeometry = HistoricalOptionGeometry.SANOS,
) -> HistoricalWeeklyScenarioSet:
    """Replay complete SANOS blocks at the current relative book coordinates."""
    remaining_steps = 5 - state.decision_index
    blocks = _complete_transition_blocks(history, step_count=remaining_steps)
    if not blocks:
        raise ValueError("historical weekly fit has no complete remaining-horizon blocks")
    coordinates = _relative_weekly_options(state)
    if not coordinates:
        raise ValueError("historical weekly fit has no live option coordinates")
    scenario_rows = tuple(
        _historical_scenario_row(
            block,
            state=state,
            claim=claim,
            coordinates=coordinates,
            option_geometry=option_geometry,
        )
        for block in blocks
    )
    option_alive = scenario_rows[0][5]
    option_settles_next = scenario_rows[0][6]
    if any(
        not np.array_equal(row[5], option_alive)
        or not np.array_equal(row[6], option_settles_next)
        for row in scenario_rows[1:]
    ):
        raise RuntimeError("historical weekly option masks are not deterministic")
    inception_paths = np.stack([row[0] for row in scenario_rows])
    spot_values = np.stack([row[1] for row in scenario_rows])
    spot_gains = np.stack([row[2] for row in scenario_rows])
    option_gains = np.stack([row[3] for row in scenario_rows])
    option_spreads = np.stack([row[4] for row in scenario_rows])
    terminal_claims = np.asarray([row[7] for row in scenario_rows], dtype=np.float64)
    terminal_claim_pv_scales = np.asarray(
        [row[8] for row in scenario_rows],
        dtype=np.float64,
    )
    invalid_fields = {
        name: float(np.min(values))
        for name, values in (
            ("inception_forward_paths", inception_paths),
            ("spot_values_pv", spot_values),
            ("option_half_spreads_pv", option_spreads),
            ("terminal_claim_values_pv", terminal_claims),
        )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0)
    }
    if np.any(inception_paths == 0.0) or np.any(spot_values == 0.0):
        invalid_fields["strictly_positive_level"] = 0.0
    if invalid_fields:
        raise FloatingPointError(
            f"historical weekly scenario fields are invalid: {invalid_fields}"
        )
    batch = WeeklyScenarioBatch(
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
        fit_as_of_date=history.as_of_date,
        model_fit_identity=model_fit_identity,
        measure="historical-P",
        seed=None,
    )
    return HistoricalWeeklyScenarioSet(
        scenarios=batch,
        training_origin_dates=tuple(
            block[0][0].origin.trading_date for block in blocks
        ),
        training_terminal_dates=tuple(block[-1][0].next_date for block in blocks),
        terminal_claim_pv_scales=terminal_claim_pv_scales,
    )


def _complete_transition_blocks(
    history: HistoricalWeeklyTransitionHistory,
    *,
    step_count: int,
) -> tuple[tuple[tuple[OneDayTransition, PublicMarketDay], ...], ...]:
    if not 1 <= step_count <= 5:
        raise ValueError("historical weekly block length must lie in [1,5]")
    lookup = {
        transition.origin.trading_date: (transition, endpoint)
        for transition, endpoint in zip(
            history.transitions,
            history.endpoints,
            strict=True,
        )
    }
    blocks: list[tuple[tuple[OneDayTransition, PublicMarketDay], ...]] = []
    for first in history.transitions:
        current_date = first.origin.trading_date
        block: list[tuple[OneDayTransition, PublicMarketDay]] = []
        for _ in range(step_count):
            pair = lookup.get(current_date)
            if pair is None:
                break
            block.append(pair)
            current_date = pair[0].next_date
        if len(block) == step_count:
            blocks.append(tuple(block))
    return tuple(blocks)


def _relative_weekly_options(
    state: WeeklyDecisionState,
) -> tuple[_RelativeWeeklyOption, ...]:
    snapshot = state.market_day.reprice_contracts(
        state.contracts,
        spot_over_episode_origin=state.spot_over_episode_origin,
    )
    quote_by_id = snapshot.quote_by_id()
    coordinates: list[_RelativeWeeklyOption] = []
    for contract_index in np.flatnonzero(state.option_alive):
        contract = state.contracts[int(contract_index)]
        quote = quote_by_id[contract.contract_id]
        business_dte = int(round(quote.business_dte))
        if not math.isclose(
            quote.business_dte,
            business_dte,
            rel_tol=0.0,
            abs_tol=1.0e-5,
        ):
            raise ValueError("historical weekly replay requires integer option DTE")
        coordinates.append(
            _RelativeWeeklyOption(
                contract_index=int(contract_index),
                contract_id=contract.contract_id,
                business_dte=business_dte,
                nms=quote.nms,
                right=contract.right,
            )
        )
    return tuple(coordinates)


def _historical_scenario_row(
    block: tuple[tuple[OneDayTransition, PublicMarketDay], ...],
    *,
    state: WeeklyDecisionState,
    claim: WeeklyClaimDefinition,
    coordinates: tuple[_RelativeWeeklyOption, ...],
    option_geometry: HistoricalOptionGeometry,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    NDArray[np.bool_],
    NDArray[np.bool_],
    float,
    float,
]:
    step_count = len(block)
    origin = block[0][0].origin
    days = tuple(pair[0].origin for pair in block) + (block[-1][1],)
    transitions = tuple(pair[0] for pair in block)
    spot_path = np.ones(step_count + 1, dtype=np.float64)
    discount_path = np.ones(step_count + 1, dtype=np.float64)
    spot_step_gains = np.empty(step_count, dtype=np.float64)
    for step, transition in enumerate(transitions):
        spot_path[step + 1] = spot_path[step] * (
            transition.forward_over_spot * transition.terminal_normalized_spot
        )
        discount_path[step + 1] = (
            discount_path[step] * transition.discount_factor
        )
        spot_step_gains[step] = (
            discount_path[step]
            * spot_path[step]
            * transition.discounted_spot_gain
        )
    relative_target_path = np.asarray(
        [
            1.0,
            *(
                spot_path[step] / origin.forward_at_business_dte(step)
                for step in range(1, step_count + 1)
            ),
        ],
        dtype=np.float64,
    )
    current_forward_curve = np.asarray(
        [
            state.market_day.forward_at_business_dte(step)
            for step in range(1, step_count + 1)
        ],
        dtype=np.float64,
    )
    inception_forward_curve = state.inception_forward_over_episode_spot[
        state.decision_index + 1 : state.decision_index + step_count + 1
    ]
    future_target_path = (
        state.spot_over_episode_origin
        * current_forward_curve
        * relative_target_path[1:]
        / inception_forward_curve
    )
    full_target_path = np.concatenate(
        (state.inception_forward_path, future_target_path)
    )
    scenario_target_path = np.concatenate(
        (state.inception_forward_path[-1:], future_target_path)
    )

    historical_contracts = _historical_contracts(origin, coordinates=coordinates)
    contract_count = len(state.contracts)
    option_alive = np.zeros((step_count + 1, contract_count), dtype=bool)
    option_settles_next = np.zeros((step_count, contract_count), dtype=bool)
    option_marks = np.zeros((step_count + 1, contract_count), dtype=np.float64)
    option_spreads = np.zeros_like(option_marks)
    fixed_strikes = np.asarray(
        [contract.fixed_strike for contract in historical_contracts],
        dtype=np.float64,
    )
    rights = np.asarray(
        [contract.right is OptionRight.CALL for contract in historical_contracts],
        dtype=bool,
    )
    coordinate_indices = np.asarray(
        [coordinate.contract_index for coordinate in coordinates],
        dtype=np.int64,
    )
    coordinate_dtes = np.asarray(
        [coordinate.business_dte for coordinate in coordinates],
        dtype=np.int64,
    )
    for step, day in enumerate(days):
        active = coordinate_dtes > step
        active_indices = coordinate_indices[active]
        option_alive[step, active_indices] = True
        marks, spreads = _historical_contract_values(
            day,
            business_dtes=coordinate_dtes[active] - step,
            fixed_strikes=fixed_strikes[active],
            is_call=rights[active],
            spot_over_start=float(spot_path[step]),
            option_geometry=option_geometry,
        )
        option_marks[step, active_indices] = discount_path[step] * marks
        option_spreads[step, active_indices] = discount_path[step] * spreads

    option_step_gains = np.zeros((step_count, contract_count), dtype=np.float64)
    for step in range(step_count):
        settling = coordinate_dtes == step + 1
        settling_indices = coordinate_indices[settling]
        option_settles_next[step, settling_indices] = True
        if len(settling_indices):
            payoff = np.where(
                rights[settling],
                np.maximum(spot_path[step + 1] - fixed_strikes[settling], 0.0),
                np.maximum(fixed_strikes[settling] - spot_path[step + 1], 0.0),
            )
            option_step_gains[step, settling_indices] = (
                discount_path[step + 1] * payoff
                - option_marks[step, settling_indices]
            )
        surviving = coordinate_dtes > step + 1
        surviving_indices = coordinate_indices[surviving]
        option_step_gains[step, surviving_indices] = (
            option_marks[step + 1, surviving_indices]
            - option_marks[step, surviving_indices]
        )
    terminal_claim = (
        discount_path[-1]
        * claim.claim.payoff(full_target_path)
        / state.spot_over_episode_origin
    )
    terminal_claim_pv_scale = discount_path[-1] / state.spot_over_episode_origin
    return (
        scenario_target_path,
        discount_path * spot_path,
        spot_step_gains,
        option_step_gains,
        option_spreads,
        option_alive,
        option_settles_next,
        float(terminal_claim),
        float(terminal_claim_pv_scale),
    )


def _historical_contracts(
    origin: PublicMarketDay,
    *,
    coordinates: tuple[_RelativeWeeklyOption, ...],
) -> tuple[OptionContract, ...]:
    contracts: list[OptionContract] = []
    for coordinate in coordinates:
        atm_vol = origin.atm_vol_at_business_dte(coordinate.business_dte)
        strike_over_forward = float(
            NMS.from_nms(
                coordinate.nms,
                coordinate.business_dte / DAYS_PER_YEAR,
                vol=atm_vol,
                eps_vol=origin.book_selection_epsilon_atm_vol,
            )
        )
        contracts.append(
            OptionContract(
                contract_id=coordinate.contract_id,
                expiry_date=add_business_days(
                    origin.trading_date,
                    coordinate.business_dte,
                ),
                fixed_strike=(
                    origin.forward_at_business_dte(coordinate.business_dte)
                    * strike_over_forward
                ),
                right=coordinate.right,
            )
        )
    return tuple(contracts)


def _historical_contract_values(
    day: PublicMarketDay,
    *,
    business_dtes: NDArray[np.integer],
    fixed_strikes: FloatArray,
    is_call: NDArray[np.bool_],
    spot_over_start: float,
    option_geometry: HistoricalOptionGeometry,
) -> tuple[FloatArray, FloatArray]:
    dtes = np.asarray(business_dtes, dtype=np.int64)
    if len(dtes) == 0:
        return np.empty(0), np.empty(0)
    forwards = np.asarray(
        [day.forward_at_business_dte(int(value)) for value in dtes],
        dtype=np.float64,
    )
    discounts = np.asarray(
        [day.discount_at_business_dte(int(value)) for value in dtes],
        dtype=np.float64,
    )
    strikes_over_forward = fixed_strikes / (spot_over_start * forwards)
    if option_geometry is HistoricalOptionGeometry.SANOS:
        normalized_prices = price_sanos_surface(
            query_expiries_y=dtes.astype(np.float64) / DAYS_PER_YEAR,
            query_strikes=strikes_over_forward,
            is_call=is_call,
            model_expiries_y=day.sanos_model_expiries_y,
            atm_variances=day.sanos_kernel_total_variances,
            support_strikes=day.sanos_support_strikes,
            densities=day.sanos_densities,
            smoothness=day.sanos_smoothness,
            one_minute_y=day.one_minute_y,
        )
    elif option_geometry is HistoricalOptionGeometry.DLV:
        normalized_prices = price_dlv_surface(
            query_expiries_y=dtes.astype(np.float64) / DAYS_PER_YEAR,
            query_strikes=strikes_over_forward,
            is_call=is_call,
            model_expiries_y=day.dlv_model_expiries_y,
            support_strikes=day.dlv_support_strikes,
            raw_log_excess_dlvsq=day.dlv_raw_log_excess_dlvsq,
            minimum_dlvsq=day.minimum_dlvsq,
            maximum_dlvsq=day.maximum_dlvsq,
            backbone_annual_volatility=day.backbone_annual_volatility,
        )
    else:
        raise ValueError(f"unsupported historical option geometry {option_geometry!r}")
    numerical_floor = 1.0e-9
    if np.any(~np.isfinite(normalized_prices)) or float(
        np.min(normalized_prices)
    ) < -numerical_floor:
        raise FloatingPointError("historical SANOS option prices are invalid")
    normalized_prices = np.maximum(normalized_prices, 0.0)
    marks = spot_over_start * discounts * forwards * normalized_prices
    spreads = spot_over_start * np.asarray(
        [
            day.half_spread_for_fixed_contract(
                business_dte=int(business_dte),
                fixed_strike_over_current_spot=float(
                    fixed_strike / spot_over_start
                ),
                is_call=bool(side),
            )
            for business_dte, fixed_strike, side in zip(
                dtes,
                fixed_strikes,
                is_call,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    return marks, np.minimum(spreads, marks)
