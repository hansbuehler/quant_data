from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

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


def _volgan_live() -> SimpleNamespace:
    return SimpleNamespace(
        as_of_date=date(2024, 1, 3),
        forecast_date=date(2024, 1, 4),
        values=np.zeros(8),
        trailing_forward_log_returns=np.zeros(21),
    )


def _diffusion_live() -> SimpleNamespace:
    return SimpleNamespace(
        as_of_date=date(2024, 1, 3),
        forecast_date=date(2024, 1, 4),
        sequence=np.zeros((21, 2, 1, 1)),
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
        live=_volgan_live(),
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
        live=_diffusion_live(),
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


def test_volgan_weekly_reuses_paths_geometry_and_zero_policy_across_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = VolganWeeklyAdapter(
        model_config=VolganModelConfig(label="volgan-weekly-cache-test"),
        checkpoint_directory=tmp_path,
        policy_config=_policy_config(),
    )
    state, variance_claim = _state(
        adapter=adapter,
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        decision_index=0,
    )
    target_claim = build_weekly_claim(
        state.market_day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.PUT,
            nms=0.0,
        ),
    )
    target_instruction = replace(
        state.instruction,
        target_label=target_claim.target.label,
    )
    target_state = replace(state, instruction=target_instruction)
    sample_calls = 0

    def sample(*args: object, **kwargs: object) -> GeneratedWeeklyPaths:
        nonlocal sample_calls
        sample_calls += 1
        return _generated_paths(step_count=5)

    monkeypatch.setattr(
        "quantdata.sdlv.daily.front_spot.hedging.models.volgan.weekly_adapter."
        "sample_volgan_weekly_paths",
        sample,
    )
    market_fit = VolganWeeklyMarketFit(
        market_fit_identity=state.instruction.market_fit_identity,
        fit=SimpleNamespace(result=object()),
        live=_volgan_live(),
        checkpoint_sha256="0" * 64,
    )
    variance_policy = adapter.fit_policy(
        market_fit,
        instruction=state.instruction,
        claim=variance_claim,
        contracts=state.contracts,
        state=state,
    )
    target_policy = adapter.fit_policy(
        market_fit,
        instruction=target_instruction,
        claim=target_claim,
        contracts=target_state.contracts,
        state=target_state,
    )

    assert sample_calls == 1
    assert variance_policy.scenarios.spot_step_gains is (
        target_policy.scenarios.spot_step_gains
    )
    assert variance_policy.scenarios.option_step_gains is (
        target_policy.scenarios.option_step_gains
    )
    assert variance_policy.hedge.zero_policy is target_policy.hedge.zero_policy
    assert variance_policy.hedge.long_policy is not target_policy.hedge.long_policy
    assert variance_policy.scenarios.target_label != target_policy.scenarios.target_label
    assert not np.array_equal(
        variance_policy.scenarios.terminal_claim_values_pv,
        target_policy.scenarios.terminal_claim_values_pv,
    )
    [pricing_context] = adapter._weekly_pricing_contexts.values()
    assert len(pricing_context._long_geometry._convex_problems) == 2
    assert len(pricing_context._short_geometry._convex_problems) == 2
    assert not variance_policy.scenarios.spot_step_gains.flags.writeable
    assert adapter._weekly_scenario_cache.entry_count == 1

    adapter.release_weekly_origin()

    assert adapter._weekly_scenario_cache.entry_count == 0
    assert not adapter._weekly_pricing_contexts


def test_diffusion_weekly_reuses_paths_geometry_and_zero_policy_across_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = DiffusionWeeklyAdapter(
        model_config=DiffusionModelConfig(
            label="diffusion-weekly-cache-test",
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
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        decision_index=0,
    )
    second_instruction = replace(
        state.instruction,
        claim_quantity=2.0,
        target_label="realized-variance-double",
    )
    second_state = replace(state, instruction=second_instruction)
    sample_calls = 0

    def sample(*args: object, **kwargs: object) -> GeneratedWeeklyPaths:
        nonlocal sample_calls
        sample_calls += 1
        return _generated_paths(step_count=5)

    monkeypatch.setattr(
        "quantdata.sdlv.daily.front_spot.hedging.models.diffusion.weekly_adapter."
        "sample_diffusion_weekly_paths",
        sample,
    )
    market_fit = DiffusionWeeklyMarketFit(
        market_fit_identity=state.instruction.market_fit_identity,
        fit=SimpleNamespace(result=object()),
        live=_diffusion_live(),
        checkpoint_sha256="0" * 64,
    )
    first = adapter.fit_policy(
        market_fit,
        instruction=state.instruction,
        claim=claim,
        contracts=state.contracts,
        state=state,
    )
    second = adapter.fit_policy(
        market_fit,
        instruction=second_instruction,
        claim=claim,
        contracts=second_state.contracts,
        state=second_state,
    )

    assert sample_calls == 1
    assert first.scenarios.spot_step_gains is second.scenarios.spot_step_gains
    assert first.scenarios.option_step_gains is second.scenarios.option_step_gains
    assert first.hedge.zero_policy is second.hedge.zero_policy
    assert adapter._weekly_scenario_cache.entry_count == 1


def test_volgan_weekly_cache_invalidates_checkpoint_and_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = VolganWeeklyAdapter(
        model_config=VolganModelConfig(label="volgan-weekly-invalidation-test"),
        checkpoint_directory=tmp_path,
        policy_config=_policy_config(),
    )
    initial_model_identity = adapter.model_config_identity
    initial_policy_identity = adapter.policy_config_identity
    state, claim = _state(
        adapter=adapter,
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        decision_index=0,
    )
    sample_calls = 0

    def sample(*args: object, **kwargs: object) -> GeneratedWeeklyPaths:
        nonlocal sample_calls
        sample_calls += 1
        return _generated_paths(step_count=5)

    monkeypatch.setattr(
        "quantdata.sdlv.daily.front_spot.hedging.models.volgan.weekly_adapter."
        "sample_volgan_weekly_paths",
        sample,
    )
    baseline = VolganWeeklyMarketFit(
        market_fit_identity=state.instruction.market_fit_identity,
        fit=SimpleNamespace(result=object()),
        live=_volgan_live(),
        checkpoint_sha256="0" * 64,
    )
    changed_checkpoint = replace(baseline, checkpoint_sha256="1" * 64)
    changed_live_condition = _volgan_live()
    changed_live_condition.values = np.ones_like(changed_live_condition.values)
    changed_live = replace(
        baseline,
        live=changed_live_condition,
    )
    for market_fit in (baseline, changed_checkpoint, changed_live):
        adapter.fit_policy(
            market_fit,
            instruction=state.instruction,
            claim=claim,
            contracts=state.contracts,
            state=state,
        )

    assert sample_calls == 3
    assert adapter._weekly_scenario_cache.entry_count == 3
    assert adapter.model_config_identity == initial_model_identity
    assert adapter.policy_config_identity == initial_policy_identity
