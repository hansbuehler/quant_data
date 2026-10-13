from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyFitInstruction,
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_weekly_market_day,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly import (
    GeneratedWeeklyPaths,
    GeneratedWeeklyScenarioCache,
    attach_generated_weekly_claim,
    build_generated_weekly_scenario_geometry,
    build_generated_weekly_scenarios,
)


def _state(*, decision_index: int, path: np.ndarray) -> WeeklyDecisionState:
    day = synthetic_weekly_market_day()
    snapshot = day.hedge_snapshot(
        maximum_business_dte=5.0,
        episode_origin_date=day.trading_date,
    )
    contracts = tuple(quote.contract for quote in snapshot.option_quotes)
    instruction = WeeklyFitInstruction(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        model="generated-test",
        measure="P",
        target_label="test",
        claim_quantity=1.0,
        model_config_identity="model",
        policy_config_identity="policy",
        episode_origin_date=day.trading_date,
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2025, 1, 2), date(2025, 1, 3)),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    return WeeklyDecisionState(
        instruction=instruction,
        decision_index=decision_index,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=float(path[-1]),
        inception_forward_over_episode_spot=np.ones(6),
        inception_forward_path=path,
        rolling_forward_path=path,
        discount_to_origin=1.0,
        option_marks_pv=np.asarray([quote.mark for quote in snapshot.option_quotes]),
        option_half_spreads_pv=np.asarray(
            [quote.half_spread for quote in snapshot.option_quotes]
        ),
        option_alive=np.ones(len(contracts), dtype=bool),
        previous_spot_position=0.0,
        previous_option_positions=np.zeros(len(contracts)),
    )


def _paths(*, step_count: int, path_count: int = 8) -> GeneratedWeeklyPaths:
    grid = GenerativeSurfaceGrid()
    return GeneratedWeeklyPaths(
        forward_log_returns=np.zeros((path_count, step_count)),
        next_log_implied_volatilities=np.full(
            (
                path_count,
                step_count,
                len(grid.expiries_y),
                len(grid.strikes_over_forward),
            ),
            np.log(0.2),
        ),
        grid=grid,
        seed=7,
    )


def test_generated_weekly_zero_shocks_preserve_forward_normalized_spot() -> None:
    state = _state(decision_index=0, path=np.ones(1))
    claim = build_weekly_claim(
        state.market_day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.PUT,
            nms=0.0,
        ),
    )
    scenarios = build_generated_weekly_scenarios(
        _paths(step_count=5),
        state=state,
        claim=claim,
        model_fit_identity="generated-test",
        measure="P",
    )
    assert scenarios.inception_forward_paths == pytest.approx(np.ones((8, 6)))
    assert scenarios.spot_step_gains == pytest.approx(np.zeros((8, 5)))
    assert scenarios.terminal_claim_values_pv == pytest.approx(
        np.full(8, state.market_day.discount_at_business_dte(5))
    )
    assert np.all(scenarios.option_half_spreads_pv >= 0.0)
    assert np.all(np.isfinite(scenarios.option_step_gains))


def test_generated_weekly_preserves_realized_barrier_prefix() -> None:
    state = _state(
        decision_index=2,
        path=np.asarray([1.0, 0.94, 1.01]),
    )
    claim = build_weekly_claim(
        state.market_day,
        target=WeeklyTarget(kind=WeeklyTargetKind.DOWN_AND_OUT_PUT),
    )
    scenarios = build_generated_weekly_scenarios(
        _paths(step_count=3),
        state=state,
        claim=claim,
        model_fit_identity="generated-test",
        measure="P",
    )
    assert scenarios.terminal_claim_values_pv == pytest.approx(np.zeros(8))
    assert scenarios.inception_forward_paths[:, 0] == pytest.approx(
        np.full(8, 1.01)
    )


def test_generated_weekly_geometry_is_shared_across_target_claims() -> None:
    state = _state(decision_index=0, path=np.ones(1))
    put = build_weekly_claim(
        state.market_day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.PUT,
            nms=-1.0,
        ),
    )
    call = build_weekly_claim(
        state.market_day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.CALL,
            nms=1.0,
        ),
    )
    paths = _paths(step_count=5)
    paths = replace(
        paths,
        forward_log_returns=np.full_like(paths.forward_log_returns, -0.05),
    )
    geometry = build_generated_weekly_scenario_geometry(
        paths,
        state=state,
        model_fit_identity="generated-test",
        measure="P",
    )
    put_scenarios = attach_generated_weekly_claim(geometry, claim=put)
    call_scenarios = attach_generated_weekly_claim(geometry, claim=call)

    assert put_scenarios.spot_step_gains is call_scenarios.spot_step_gains
    assert put_scenarios.option_step_gains is call_scenarios.option_step_gains
    assert (
        put_scenarios.option_half_spreads_pv
        is call_scenarios.option_half_spreads_pv
    )
    assert put_scenarios.target_label != call_scenarios.target_label
    assert np.all(put_scenarios.terminal_claim_values_pv > 0.0)
    assert call_scenarios.terminal_claim_values_pv == pytest.approx(
        np.zeros(8)
    )


def test_generated_weekly_cache_identity_covers_every_path_input() -> None:
    state = _state(decision_index=0, path=np.ones(1))
    cache = GeneratedWeeklyScenarioCache()
    build_count = 0

    def get(
        *,
        artifact: str = "artifact-a",
        live: str = "live-a",
        policy: str = "policy-a",
        scenario_count: int = 8,
        seed: int = 7,
    ):
        def build() -> GeneratedWeeklyPaths:
            nonlocal build_count
            build_count += 1
            return replace(
                _paths(step_count=5, path_count=scenario_count),
                seed=seed,
            )

        return cache.get_or_build(
            state=state,
            model_fit_identity="model-a",
            artifact_fingerprints=(artifact,),
            live_state_fingerprint=live,
            policy_config_identity=policy,
            scenario_count=scenario_count,
            seed=seed,
            measure="P",
            path_builder=build,
        )

    baseline = get()
    assert get() is baseline
    assert get(artifact="artifact-b") is not baseline
    assert get(live="live-b") is not baseline
    assert get(policy="policy-b") is not baseline
    assert get(scenario_count=9) is not baseline
    assert get(seed=8) is not baseline
    assert build_count == 6


def test_generated_weekly_cache_builds_once_per_restarted_decision() -> None:
    cache = GeneratedWeeklyScenarioCache()
    build_count = 0

    def get(state: WeeklyDecisionState) -> object:
        def build() -> GeneratedWeeklyPaths:
            nonlocal build_count
            build_count += 1
            return _paths(step_count=5 - state.decision_index)

        return cache.get_or_build(
            state=state,
            model_fit_identity="model-a",
            artifact_fingerprints=("artifact-a",),
            live_state_fingerprint=f"live-{state.decision_index}",
            policy_config_identity="policy-a",
            scenario_count=8,
            seed=7,
            measure="P",
            path_builder=build,
        )

    first = _state(decision_index=0, path=np.ones(1))
    second = _state(decision_index=1, path=np.ones(2))
    first_geometry = get(first)
    assert get(first) is first_geometry
    second_geometry = get(second)
    assert get(second) is second_geometry
    assert build_count == 2
    assert cache.entry_count == 2
