from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.weekly_adapter import (
    DiffusionWeeklyAdapter,
    DiffusionWeeklyMarketFit,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly import (
    GeneratedWeeklyPaths,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly_policy import (
    GeneratedWeeklyPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.weekly_adapter import (
    VolganWeeklyAdapter,
    VolganWeeklyMarketFit,
)


def _policy_config() -> GeneratedWeeklyPolicyConfig:
    return GeneratedWeeklyPolicyConfig(
        scenario_count=12,
        seed=7,
        hedge=HistoricalSanosConfig(
            rolling_observations=20,
            maximum_iterations=500,
            position_cap=100.0,
        ),
    )


def _state(
    *,
    adapter: VolganWeeklyAdapter | DiffusionWeeklyAdapter,
    schedule: HedgeSchedule,
    decision_index: int,
) -> tuple[WeeklyDecisionState, object]:
    day = synthetic_weekly_market_day()
    quotes = day.hedge_snapshot(
        maximum_business_dte=5.0,
        episode_origin_date=day.trading_date,
    ).option_quotes[:2]
    contracts = tuple(quote.contract for quote in quotes)
    instruction = WeeklyFitInstruction(
        schedule=schedule,
        model=adapter.model,
        measure=adapter.measure,
        target_label="realized-variance",
        claim_quantity=1.0,
        model_config_identity=adapter.model_config_identity,
        policy_config_identity=adapter.policy_config_identity,
        episode_origin_date=day.trading_date,
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2024, 1, 2), date(2024, 1, 3)),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    current_target = 1.0 + 0.002 * decision_index
    path = np.linspace(1.0, current_target, decision_index + 1)
    alive = np.ones(len(contracts), dtype=bool)
    if decision_index:
        alive[0] = False
    state = WeeklyDecisionState(
        instruction=instruction,
        decision_index=decision_index,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=current_target,
        inception_forward_over_episode_spot=np.ones(6),
        inception_forward_path=path,
        rolling_forward_path=path,
        discount_to_origin=1.0,
        option_marks_pv=np.asarray([quote.mark for quote in quotes]),
        option_half_spreads_pv=np.asarray([quote.half_spread for quote in quotes]),
        option_alive=alive,
        previous_spot_position=0.2,
        previous_option_positions=np.zeros(len(contracts)),
    )
    claim = build_weekly_claim(
        day,
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    return state, claim


def _generated_paths(*, step_count: int) -> GeneratedWeeklyPaths:
    grid = GenerativeSurfaceGrid()
    generator = np.random.default_rng(11)
    returns = generator.normal(scale=0.005, size=(12, step_count))
    return GeneratedWeeklyPaths(
        forward_log_returns=returns,
        next_log_implied_volatilities=np.full(
            (
                12,
                step_count,
                len(grid.expiries_y),
                len(grid.strikes_over_forward),
            ),
            np.log(0.2),
        ),
        grid=grid,
        seed=7,
    )


@pytest.mark.parametrize("schedule", tuple(HedgeSchedule)[1:])
def test_volgan_weekly_policy_uses_correct_schedule_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: HedgeSchedule,
) -> None:
    adapter = VolganWeeklyAdapter(
        model_config=VolganModelConfig(label="volgan-weekly-adapter-test"),
        checkpoint_directory=tmp_path,
        policy_config=_policy_config(),
    )
    decision_index = 0 if schedule is HedgeSchedule.FROZEN_WEEKLY else 1
    state, claim = _state(
        adapter=adapter,
        schedule=schedule,
        decision_index=decision_index,
    )
    monkeypatch.setattr(
        "quantdata.sdlv.daily.front_spot.hedging.models.volgan.weekly_adapter."
        "sample_volgan_weekly_paths",
        lambda *args, **kwargs: _generated_paths(
            step_count=5 - decision_index
        ),
    )
    market_fit = VolganWeeklyMarketFit(
        market_fit_identity=state.instruction.market_fit_identity,
        fit=SimpleNamespace(result=object()),
        live=SimpleNamespace(),
        checkpoint_sha256="0" * 64,
    )
    policy = adapter.fit_policy(
        market_fit,
        instruction=state.instruction,
        claim=claim,
        contracts=state.contracts,
        state=state,
    )
    action = adapter.action(policy, state=state)

    assert action.option_positions.shape == (len(state.contracts),)
    assert action.option_positions[~state.option_alive] == pytest.approx(0.0)
    assert policy.prices.q_expectation_per_fit_spot is None


def test_diffusion_weekly_policy_is_restart_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = DiffusionWeeklyAdapter(
        model_config=DiffusionModelConfig(
            label="diffusion-weekly-adapter-test",
            model_dimension=16,
            transformer_layers=1,
            attention_heads=4,
            diffusion_steps=4,
        ),
        checkpoint_directory=tmp_path,
        policy_config=_policy_config(),
    )
    state, claim = _state(
        adapter=adapter,
        schedule=HedgeSchedule.RESTARTED_WEEKLY,
        decision_index=2,
    )
    monkeypatch.setattr(
        "quantdata.sdlv.daily.front_spot.hedging.models.diffusion.weekly_adapter."
        "sample_diffusion_weekly_paths",
        lambda *args, **kwargs: _generated_paths(step_count=3),
    )
    market_fit = DiffusionWeeklyMarketFit(
        market_fit_identity=state.instruction.market_fit_identity,
        fit=SimpleNamespace(result=object()),
        live=SimpleNamespace(),
        checkpoint_sha256="0" * 64,
    )
    policy = adapter.fit_policy(
        market_fit,
        instruction=state.instruction,
        claim=claim,
        contracts=state.contracts,
        state=state,
    )
    action = adapter.action(policy, state=state)

    assert action.option_positions[~state.option_alive] == pytest.approx(0.0)
    stale_state, _ = _state(
        adapter=adapter,
        schedule=HedgeSchedule.RESTARTED_WEEKLY,
        decision_index=1,
    )
    with pytest.raises(ValueError, match="stale"):
        adapter.action(policy, state=stale_state)
