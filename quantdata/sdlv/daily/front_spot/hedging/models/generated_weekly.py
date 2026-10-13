"""Common weekly fixed-contract accounting for recursive surface generators."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.contracts import OptionRight
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WEEKLY_HORIZON,
    WeeklyClaimDefinition,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyScenarioBatch,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    _black_option_price,
    interpolate_generated_implied_volatility,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class GeneratedWeeklyPaths:
    """Recursive generator output from one dated remaining-horizon fit."""

    forward_log_returns: FloatArray
    next_log_implied_volatilities: FloatArray
    grid: GenerativeSurfaceGrid
    seed: int

    def __post_init__(self) -> None:
        if self.forward_log_returns.ndim != 2:
            raise ValueError("weekly generated returns must be a path-step matrix")
        scenario_count, step_count = self.forward_log_returns.shape
        expected = (
            scenario_count,
            step_count,
            len(self.grid.expiries_y),
            len(self.grid.strikes_over_forward),
        )
        if scenario_count < 2 or not 1 <= step_count <= WEEKLY_HORIZON:
            raise ValueError("weekly generated paths have invalid dimensions")
        if self.next_log_implied_volatilities.shape != expected:
            raise ValueError("weekly generated surfaces do not align with returns")
        if not np.all(np.isfinite(self.forward_log_returns)) or not np.all(
            np.isfinite(self.next_log_implied_volatilities)
        ):
            raise ValueError("weekly generated paths must be finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("weekly generated seed must be an integer")

    @property
    def scenario_count(self) -> int:
        return len(self.forward_log_returns)

    @property
    def step_count(self) -> int:
        return self.forward_log_returns.shape[1]


@dataclass(frozen=True, kw_only=True)
class GeneratedWeeklyScenarioGeometry:
    """Target-independent prices and gains for one generated decision state."""

    full_inception_forward_paths: FloatArray
    inception_forward_paths: FloatArray
    spot_values_pv: FloatArray
    spot_step_gains: FloatArray
    option_step_gains: FloatArray
    option_half_spreads_pv: FloatArray
    option_alive: NDArray[np.bool_]
    option_settles_next: NDArray[np.bool_]
    contract_ids: tuple[str, ...]
    fit_as_of_date: date
    model_fit_identity: str
    measure: str
    seed: int
    terminal_discount_over_episode_spot: float

    def __post_init__(self) -> None:
        arrays = (
            self.full_inception_forward_paths,
            self.inception_forward_paths,
            self.spot_values_pv,
            self.spot_step_gains,
            self.option_step_gains,
            self.option_half_spreads_pv,
            self.option_alive,
            self.option_settles_next,
        )
        if any(not isinstance(values, np.ndarray) for values in arrays):
            raise TypeError("weekly generated geometry arrays must be NumPy arrays")
        if (
            not math.isfinite(self.terminal_discount_over_episode_spot)
            or self.terminal_discount_over_episode_spot <= 0.0
        ):
            raise ValueError("weekly generated terminal discount scale is invalid")
        for values in arrays:
            values.setflags(write=False)

    @property
    def scenario_count(self) -> int:
        return len(self.inception_forward_paths)

    @property
    def step_count(self) -> int:
        return self.inception_forward_paths.shape[1] - 1


class GeneratedWeeklyScenarioCache:
    """Reuse target-independent paths and prices within one weekly episode."""

    def __init__(self) -> None:
        self._episode_origin_date: date | None = None
        self._geometries: dict[str, GeneratedWeeklyScenarioGeometry] = {}
        self._generation = 0

    @property
    def generation(self) -> int:
        """Return the generation used to invalidate associated solver state."""
        return self._generation

    @property
    def entry_count(self) -> int:
        """Return the number of decision states retained for the current origin."""
        return len(self._geometries)

    def clear(self) -> None:
        """Release every generated array retained for the completed origin."""
        self._episode_origin_date = None
        self._geometries.clear()
        self._generation += 1

    def get_or_build(
        self,
        *,
        state: WeeklyDecisionState,
        model_fit_identity: str,
        artifact_fingerprints: tuple[str, ...],
        live_state_fingerprint: str,
        policy_config_identity: str,
        scenario_count: int,
        seed: int,
        measure: str,
        path_builder: Callable[[], GeneratedWeeklyPaths],
    ) -> GeneratedWeeklyScenarioGeometry:
        """Return one exact geometry for every target sharing a decision state."""
        episode_origin_date = state.instruction.episode_origin_date
        if self._episode_origin_date != episode_origin_date:
            self.clear()
            self._episode_origin_date = episode_origin_date
        identity = generated_weekly_state_identity(
            state,
            model_fit_identity=model_fit_identity,
            artifact_fingerprints=artifact_fingerprints,
            live_state_fingerprint=live_state_fingerprint,
            policy_config_identity=policy_config_identity,
            scenario_count=scenario_count,
            seed=seed,
            measure=measure,
        )
        geometry = self._geometries.get(identity)
        if geometry is None:
            paths = path_builder()
            if paths.scenario_count != scenario_count or paths.seed != seed:
                raise ValueError("weekly generated paths do not match cache identity")
            geometry = build_generated_weekly_scenario_geometry(
                paths,
                state=state,
                model_fit_identity=model_fit_identity,
                measure=measure,
            )
            self._geometries[identity] = geometry
        return geometry


def generated_weekly_state_identity(
    state: WeeklyDecisionState,
    *,
    model_fit_identity: str,
    artifact_fingerprints: tuple[str, ...],
    live_state_fingerprint: str,
    policy_config_identity: str,
    scenario_count: int,
    seed: int,
    measure: str,
) -> str:
    """Identify exactly the target-independent inputs used by scenario pricing."""
    if (
        not artifact_fingerprints
        or any(not value.strip() for value in artifact_fingerprints)
        or not live_state_fingerprint.strip()
        or not policy_config_identity.strip()
        or scenario_count < 2
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("weekly generated cache identity is incomplete")
    digest = sha256()
    for value in (
        model_fit_identity,
        *artifact_fingerprints,
        live_state_fingerprint,
        policy_config_identity,
        str(scenario_count),
        str(seed),
        measure,
        state.instruction.schedule.value,
        state.instruction.episode_origin_date.isoformat(),
        state.instruction.decision_date.isoformat(),
        str(state.decision_index),
        state.market_day.trading_date.isoformat(),
        *(contract.contract_id for contract in state.contracts),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.float64(state.spot_over_episode_origin).tobytes())
    digest.update(np.float64(state.discount_to_origin).tobytes())
    for values in (
        state.inception_forward_over_episode_spot,
        state.inception_forward_path,
        state.option_marks_pv,
        state.option_half_spreads_pv,
        state.option_alive,
        state.market_day.business_dtes,
        state.market_day.forward_over_spot,
        state.market_day.discount_factors,
        state.market_day.expiry_indices,
        state.market_day.strikes_over_forward,
        state.market_day.is_call,
        state.market_day.half_spreads_over_spot,
        state.market_day.is_am_settlement,
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def generated_live_state_fingerprint(
    *,
    model: str,
    as_of_date: date,
    forecast_date: date,
    arrays: tuple[NDArray[np.generic], ...],
) -> str:
    """Hash every dated live conditioning input used by one path generator."""
    if not model.strip() or forecast_date <= as_of_date or not arrays:
        raise ValueError("weekly generated live state identity is incomplete")
    digest = sha256()
    for value in (model, as_of_date.isoformat(), forecast_date.isoformat()):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for values in arrays:
        array = np.ascontiguousarray(values)
        if not np.all(np.isfinite(array)):
            raise ValueError("weekly generated live state must be finite")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_generated_weekly_scenario_geometry(
    paths: GeneratedWeeklyPaths,
    *,
    state: WeeklyDecisionState,
    model_fit_identity: str,
    measure: str,
) -> GeneratedWeeklyScenarioGeometry:
    """Reprice one frozen weekly book independently of the target claim."""
    remaining_steps = WEEKLY_HORIZON - state.decision_index
    if paths.step_count != remaining_steps:
        raise ValueError("weekly generated horizon does not match decision state")
    if not model_fit_identity.strip() or not measure.strip():
        raise ValueError("weekly generated scenario identities must be non-empty")
    expected_current_target = (
        state.spot_over_episode_origin
        / state.inception_forward_over_episode_spot[state.decision_index]
    )
    if not math.isclose(
        state.inception_forward_path[-1],
        expected_current_target,
        rel_tol=2.0e-10,
        abs_tol=2.0e-12,
    ):
        raise ValueError(
            "weekly generated state is inconsistent with its inception forward"
        )

    day = state.market_day
    scenario_count = paths.scenario_count
    forward_curve = np.asarray(
        [
            1.0,
            *(day.forward_at_business_dte(step) for step in range(1, remaining_steps + 1)),
        ],
        dtype=np.float64,
    )
    discount_curve = np.asarray(
        [
            1.0,
            *(day.discount_at_business_dte(step) for step in range(1, remaining_steps + 1)),
        ],
        dtype=np.float64,
    )
    normalized_shocks = np.exp(paths.forward_log_returns)
    if not np.all(np.isfinite(normalized_shocks) & (normalized_shocks > 0.0)):
        raise FloatingPointError("weekly generated spot shocks are invalid")
    cumulative_shocks = np.column_stack(
        (
            np.ones(scenario_count, dtype=np.float64),
            np.cumprod(normalized_shocks, axis=1),
        )
    )
    spot_paths = cumulative_shocks * forward_curve[None, :]
    future_episode_spot = (
        state.spot_over_episode_origin * spot_paths[:, 1:]
    )
    future_inception_path = future_episode_spot / (
        state.inception_forward_over_episode_spot[
            state.decision_index + 1 : WEEKLY_HORIZON + 1
        ][None, :]
    )
    full_target_paths = np.concatenate(
        (
            np.broadcast_to(
                state.inception_forward_path,
                (scenario_count, len(state.inception_forward_path)),
            ),
            future_inception_path,
        ),
        axis=1,
    )
    scenario_target_paths = np.column_stack(
        (
            np.full(scenario_count, state.inception_forward_path[-1]),
            future_inception_path,
        )
    )

    snapshot = day.reprice_contracts(
        state.contracts,
        spot_over_episode_origin=state.spot_over_episode_origin,
    )
    quote_by_id = snapshot.quote_by_id()
    contract_count = len(state.contracts)
    business_dtes = np.zeros(contract_count, dtype=np.int64)
    fixed_strikes = np.zeros(contract_count, dtype=np.float64)
    is_call = np.zeros(contract_count, dtype=bool)
    for contract_index in np.flatnonzero(state.option_alive):
        contract = state.contracts[int(contract_index)]
        quote = quote_by_id[contract.contract_id]
        rounded_dte = int(round(quote.business_dte))
        if not math.isclose(
            quote.business_dte,
            rounded_dte,
            rel_tol=0.0,
            abs_tol=2.0e-6,
        ):
            raise ValueError("weekly generated options require integer DTE")
        business_dtes[contract_index] = rounded_dte
        fixed_strikes[contract_index] = (
            contract.fixed_strike / state.spot_over_episode_origin
        )
        is_call[contract_index] = contract.right is OptionRight.CALL

    option_alive = np.zeros((remaining_steps + 1, contract_count), dtype=bool)
    option_settles_next = np.zeros((remaining_steps, contract_count), dtype=bool)
    option_marks = np.zeros(
        (scenario_count, remaining_steps + 1, contract_count),
        dtype=np.float64,
    )
    option_spreads = np.zeros_like(option_marks)
    current_scale = state.discount_to_origin * state.spot_over_episode_origin
    if not math.isfinite(current_scale) or current_scale <= 0.0:
        raise ValueError("weekly generated current price scale is invalid")
    live_indices = np.flatnonzero(state.option_alive)
    option_alive[0, live_indices] = True
    option_marks[:, 0, live_indices] = (
        state.option_marks_pv[live_indices] / current_scale
    )
    option_spreads[:, 0, live_indices] = (
        state.option_half_spreads_pv[live_indices] / current_scale
    )

    for step in range(1, remaining_steps + 1):
        active = state.option_alive & (business_dtes > step)
        active_indices = np.flatnonzero(active)
        option_alive[step, active_indices] = True
        for contract_index in active_indices:
            initial_dte = int(business_dtes[contract_index])
            remaining_dte = initial_dte - step
            future_forward = (
                spot_paths[:, step]
                * day.forward_at_business_dte(initial_dte)
                / forward_curve[step]
            )
            strike_over_forward = (
                fixed_strikes[contract_index] / future_forward
            )
            volatility = interpolate_generated_implied_volatility(
                paths.next_log_implied_volatilities[:, step - 1],
                query_expiry_y=remaining_dte / DAYS_PER_YEAR,
                query_strikes_over_forward=strike_over_forward,
                grid=paths.grid,
            )
            normalized_price = _black_option_price(
                strike=strike_over_forward,
                annual_volatility=volatility,
                expiry_y=remaining_dte / DAYS_PER_YEAR,
                right=state.contracts[contract_index].right,
            )
            latent_scale = spot_paths[:, step] / forward_curve[step]
            mark = (
                day.discount_at_business_dte(initial_dte)
                * day.forward_at_business_dte(initial_dte)
                * latent_scale
                * normalized_price
            )
            if np.any(~np.isfinite(mark)) or float(np.min(mark)) < -1.0e-9:
                raise FloatingPointError("weekly generated option marks are invalid")
            mark = np.maximum(mark, 0.0)
            option_marks[:, step, contract_index] = mark
            raw_spread = np.asarray(
                [
                    discount_curve[step]
                    * path_spot
                    * day.half_spread_for_fixed_contract(
                        business_dte=remaining_dte,
                        fixed_strike_over_current_spot=(
                            fixed_strikes[contract_index] / path_spot
                        ),
                        is_call=bool(is_call[contract_index]),
                    )
                    for path_spot in spot_paths[:, step]
                ],
                dtype=np.float64,
            )
            option_spreads[:, step, contract_index] = np.minimum(
                raw_spread,
                mark,
            )

    option_gains = np.zeros(
        (scenario_count, remaining_steps, contract_count),
        dtype=np.float64,
    )
    for step in range(remaining_steps):
        settling = state.option_alive & (business_dtes == step + 1)
        settling_indices = np.flatnonzero(settling)
        option_settles_next[step, settling_indices] = True
        terminal_spot = spot_paths[:, step + 1]
        for contract_index in settling_indices:
            payoff = (
                np.maximum(
                    terminal_spot - fixed_strikes[contract_index],
                    0.0,
                )
                if is_call[contract_index]
                else np.maximum(
                    fixed_strikes[contract_index] - terminal_spot,
                    0.0,
                )
            )
            option_gains[:, step, contract_index] = (
                discount_curve[step + 1] * payoff
                - option_marks[:, step, contract_index]
            )
        surviving = state.option_alive & (business_dtes > step + 1)
        option_gains[:, step, surviving] = (
            option_marks[:, step + 1, surviving]
            - option_marks[:, step, surviving]
        )

    spot_gains = np.empty((scenario_count, remaining_steps), dtype=np.float64)
    for step in range(remaining_steps):
        bridge = (
            discount_curve[step + 1]
            / discount_curve[step]
            * forward_curve[step + 1]
            / forward_curve[step]
        )
        spot_gains[:, step] = (
            discount_curve[step]
            * spot_paths[:, step]
            * bridge
            * (normalized_shocks[:, step] - 1.0)
        )
    return GeneratedWeeklyScenarioGeometry(
        full_inception_forward_paths=full_target_paths,
        inception_forward_paths=scenario_target_paths,
        spot_values_pv=discount_curve[None, :] * spot_paths,
        spot_step_gains=spot_gains,
        option_step_gains=option_gains,
        option_half_spreads_pv=option_spreads,
        option_alive=option_alive,
        option_settles_next=option_settles_next,
        contract_ids=tuple(contract.contract_id for contract in state.contracts),
        fit_as_of_date=state.instruction.fit_as_of_date,
        model_fit_identity=model_fit_identity,
        measure=measure,
        seed=paths.seed,
        terminal_discount_over_episode_spot=(
            discount_curve[-1] / state.spot_over_episode_origin
        ),
    )


def attach_generated_weekly_claim(
    geometry: GeneratedWeeklyScenarioGeometry,
    *,
    claim: WeeklyClaimDefinition,
) -> WeeklyScenarioBatch:
    """Attach one fixed inception claim to reusable generated market geometry."""
    if claim.claim.maturity_business_days != WEEKLY_HORIZON:
        raise ValueError("weekly generated scenarios require a five-session claim")
    terminal_claims = geometry.terminal_discount_over_episode_spot * np.asarray(
        [
            claim.claim.payoff(path)
            for path in geometry.full_inception_forward_paths
        ],
        dtype=np.float64,
    )
    return WeeklyScenarioBatch(
        inception_forward_paths=geometry.inception_forward_paths,
        spot_values_pv=geometry.spot_values_pv,
        spot_step_gains=geometry.spot_step_gains,
        option_step_gains=geometry.option_step_gains,
        option_half_spreads_pv=geometry.option_half_spreads_pv,
        option_alive=geometry.option_alive,
        option_settles_next=geometry.option_settles_next,
        terminal_claim_values_pv=terminal_claims,
        contract_ids=geometry.contract_ids,
        target_label=claim.target.label,
        fit_as_of_date=geometry.fit_as_of_date,
        model_fit_identity=geometry.model_fit_identity,
        measure=geometry.measure,
        seed=geometry.seed,
    )


def build_generated_weekly_scenarios(
    paths: GeneratedWeeklyPaths,
    *,
    state: WeeklyDecisionState,
    claim: WeeklyClaimDefinition,
    model_fit_identity: str,
    measure: str,
) -> WeeklyScenarioBatch:
    """Reprice one frozen weekly book through recursive generated states."""
    geometry = build_generated_weekly_scenario_geometry(
        paths,
        state=state,
        model_fit_identity=model_fit_identity,
        measure=measure,
    )
    return attach_generated_weekly_claim(geometry, claim=claim)
