from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from quantdata.biz_calendar import add_business_days
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyFitInstruction,
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
    WeeklyPriceRecord,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyRiskObjective,
    WeeklyScenarioBatch,
    weekly_certainty_equivalent,
    weekly_scenario_accounting,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_weekly_market_day,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_adapter import (
    HistoricalSanosWeeklyPolicy,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_hedger import (
    _WeeklyOpenLoopGeometry,
    fit_historical_weekly_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_scenarios import (
    HistoricalWeeklyScenarioSet,
    HistoricalWeeklyTransitionHistory,
    build_historical_weekly_scenarios,
)


def _scenario_set(*, scenario_count: int = 80) -> HistoricalWeeklyScenarioSet:
    generator = np.random.default_rng(42)
    step_count = 3
    contract_count = 2
    spot_gains = generator.normal(scale=0.01, size=(scenario_count, step_count))
    option_gains = generator.normal(
        scale=0.008,
        size=(scenario_count, step_count, contract_count),
    )
    option_alive = np.asarray(
        [
            [True, True],
            [False, True],
            [False, True],
            [False, True],
        ]
    )
    option_settles_next = np.asarray(
        [
            [True, False],
            [False, False],
            [False, False],
        ]
    )
    option_gains[:, 1:, 0] = 0.0
    option_spreads = np.full(
        (scenario_count, step_count + 1, contract_count),
        2.0e-4,
    )
    option_spreads[:, 1:, 0] = 0.0
    target = -(
        0.4 * np.sum(spot_gains, axis=1)
        - 0.25 * np.sum(option_gains[:, :, 1], axis=1)
    )
    fit_date = date(2022, 12, 30)
    batch = WeeklyScenarioBatch(
        inception_forward_paths=np.exp(
            np.column_stack(
                (
                    np.zeros(scenario_count),
                    np.cumsum(spot_gains, axis=1),
                )
            )
        ),
        spot_values_pv=np.ones((scenario_count, step_count + 1)),
        spot_step_gains=spot_gains,
        option_step_gains=option_gains,
        option_half_spreads_pv=option_spreads,
        option_alive=option_alive,
        option_settles_next=option_settles_next,
        terminal_claim_values_pv=target - np.min(target) + 0.01,
        contract_ids=("short", "long"),
        target_label="test-claim",
        fit_as_of_date=fit_date,
        model_fit_identity="historical-weekly-test",
        measure="historical-P",
        seed=42,
    )
    origins = tuple(
        fit_date - timedelta(days=scenario_count - index + 3)
        for index in range(scenario_count)
    )
    terminals = tuple(value + timedelta(days=3) for value in origins)
    return HistoricalWeeklyScenarioSet(
        scenarios=batch,
        training_origin_dates=origins,
        training_terminal_dates=terminals,
    )


@pytest.mark.parametrize("objective", tuple(HistoricalSanosObjective))
def test_historical_weekly_gradient_matches_finite_differences(
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set()
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=objective,
        risk_parameter=1.3,
        cost_smoothing=1.0e-4,
        maximum_iterations=500,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.15,
        initial_option_positions=np.asarray([-0.2, 0.1]),
        config=config,
    )
    include_translation = objective is HistoricalSanosObjective.POWER_EXPONENTIAL
    generator = np.random.default_rng(17)
    parameters = generator.normal(scale=0.15, size=geometry.parameter_count)
    if include_translation:
        parameters = np.concatenate((parameters, [0.02]))
    indices = np.arange(60)
    loss, gradient = geometry._loss_and_gradient(
        parameters,
        indices,
        signed_claim_quantity=0.7,
        include_translation=include_translation,
    )
    finite_difference = np.empty_like(parameters)
    step = 1.0e-6
    for index in range(len(parameters)):
        up = parameters.copy()
        down = parameters.copy()
        up[index] += step
        down[index] -= step
        up_loss = geometry._loss_and_gradient(
            up,
            indices,
            signed_claim_quantity=0.7,
            include_translation=include_translation,
        )[0]
        down_loss = geometry._loss_and_gradient(
            down,
            indices,
            signed_claim_quantity=0.7,
            include_translation=include_translation,
        )[0]
        finite_difference[index] = (up_loss - down_loss) / (2.0 * step)
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, finite_difference, atol=2.0e-7, rtol=2.0e-5)


@pytest.mark.parametrize(
    "objective",
    (
        HistoricalSanosObjective.MEAN_VARIANCE,
        HistoricalSanosObjective.MEAN_VOLATILITY,
    ),
)
def test_historical_weekly_fit_respects_expiry_and_returns_pure_hedge(
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set()
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=objective,
            risk_parameter=3.0,
            cost_smoothing=1.0e-5,
            maximum_iterations=2_000,
            gradient_tolerance=1.0e-7,
        ),
    )
    assert fit.spot_positions.shape == (3,)
    assert fit.option_positions.shape == (3, 2)
    assert fit.option_positions[1:, 0] == pytest.approx(np.zeros(2))
    assert np.all(np.isfinite(fit.spot_positions))
    assert np.all(np.isfinite(fit.option_positions))
    assert fit.prices.buyer <= fit.prices.seller


def test_historical_weekly_records_virtual_recession_without_executable_cap() -> None:
    generator = np.random.default_rng(109)
    scenario_count = 80
    fit_date = date(2022, 12, 30)
    spot_gains = (0.1 + 0.001 * generator.normal(size=scenario_count))[:, None]
    scenarios = WeeklyScenarioBatch(
        inception_forward_paths=np.column_stack(
            (np.ones(scenario_count), np.exp(spot_gains[:, 0]))
        ),
        spot_values_pv=np.ones((scenario_count, 2)),
        spot_step_gains=spot_gains,
        option_step_gains=np.zeros((scenario_count, 1, 1)),
        option_half_spreads_pv=np.zeros((scenario_count, 2, 1)),
        option_alive=np.asarray([[True], [False]]),
        option_settles_next=np.asarray([[True]]),
        terminal_claim_values_pv=np.zeros(scenario_count),
        contract_ids=("zero-gain-option",),
        target_label="virtual-recession",
        fit_as_of_date=fit_date,
        model_fit_identity="virtual-recession",
        measure="historical-P",
        seed=109,
    )
    scenario_set = HistoricalWeeklyScenarioSet(
        scenarios=scenarios,
        training_origin_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 2)
            for index in range(scenario_count)
        ),
        training_terminal_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 1)
            for index in range(scenario_count)
        ),
    )
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(1),
        config=HistoricalSanosConfig(
            rolling_observations=scenario_count,
            objective=HistoricalSanosObjective.MEAN_VOLATILITY,
            risk_parameter=0.1,
            position_cap=10.0,
            solver_position_cap=100.0,
            maximum_iterations=10_000,
        ),
    )

    assert fit.long_policy.spot_positions[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.short_policy.spot_positions[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.zero_policy.spot_positions[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.internal_cap_hits >= 3
    assert fit.executable_cap_hits == 0
    assert fit.spot_positions[0] == pytest.approx(0.0, abs=1.0e-5)
    assert not fit.converged

    diagnostic_fit = replace(
        fit,
        internal_cap_hits=2,
        executable_cap_hits=1,
        cap_hits=3,
    )
    policy = HistoricalSanosWeeklyPolicy(
        market_fit_identity="market-fit",
        policy_fit_identity="policy-fit",
        prices=WeeklyPriceRecord(
            common_market_price_per_fit_spot=0.0,
            q_expectation_per_fit_spot=None,
            buyer_indifference_per_fit_spot=0.0,
            seller_indifference_per_fit_spot=0.0,
            continuation_price_per_fit_spot=0.0,
        ),
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        fit_decision_index=0,
        scenarios=scenario_set,
        hedge=diagnostic_fit,
    )
    assert policy.cap_hits == 2
    assert policy.converged is False


def test_historical_weekly_scenarios_use_complete_pre_asof_blocks() -> None:
    day = synthetic_weekly_market_day()
    snapshot = day.hedge_snapshot(
        maximum_business_dte=5.0,
        episode_origin_date=day.trading_date,
    )
    contracts = tuple(quote.contract for quote in snapshot.option_quotes)
    instruction = WeeklyFitInstruction(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        model="historical-sanos",
        measure="historical-P",
        target_label="realized-variance",
        claim_quantity=1.0,
        model_config_identity="model",
        policy_config_identity="policy",
        episode_origin_date=day.trading_date,
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2024, 1, 2), date(2024, 1, 3)),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    state = WeeklyDecisionState(
        instruction=instruction,
        decision_index=0,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=1.0,
        inception_forward_over_episode_spot=np.ones(6),
        inception_forward_path=np.ones(1),
        rolling_forward_path=np.ones(1),
        discount_to_origin=1.0,
        option_marks_pv=np.asarray([quote.mark for quote in snapshot.option_quotes]),
        option_half_spreads_pv=np.asarray(
            [quote.half_spread for quote in snapshot.option_quotes]
        ),
        option_alive=np.ones(len(contracts), dtype=bool),
        previous_spot_position=0.0,
        previous_option_positions=np.zeros(len(contracts)),
    )
    historical_dates = tuple(
        add_business_days(date(2024, 1, 2), offset)
        for offset in range(11)
    )
    historical_days = tuple(
        replace(day, trading_date=trading_date)
        for trading_date in historical_dates
    )
    transitions = tuple(
        OneDayTransition(
            origin=historical_days[index],
            next_date=historical_dates[index + 1],
            forward_adjusted_spot_return=0.002 * ((index % 3) - 1),
            discount_factor=0.999,
            forward_over_spot=1.001,
        )
        for index in range(10)
    )
    history = HistoricalWeeklyTransitionHistory(
        as_of_date=day.trading_date,
        transitions=transitions,
        endpoints=historical_days[1:],
    )
    scenario_set = build_historical_weekly_scenarios(
        history,
        state=state,
        claim=build_weekly_claim(
            day,
            target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
        ),
        model_fit_identity="historical-test",
    )
    assert scenario_set.scenarios.scenario_count == 6
    assert scenario_set.scenarios.step_count == 5
    assert scenario_set.training_origin_dates[0] == historical_dates[0]
    assert scenario_set.training_terminal_dates[-1] == historical_dates[-1]
    assert max(scenario_set.training_terminal_dates) <= day.trading_date
    assert np.all(np.isfinite(scenario_set.scenarios.option_step_gains))
    assert np.all(scenario_set.scenarios.terminal_claim_values_pv >= 0.0)


def test_restarted_scenarios_preserve_fixed_inception_forward_normalization() -> None:
    day = replace(
        synthetic_weekly_market_day(),
        forward_over_spot=np.asarray([1.01, 1.025, 1.04, 1.055, 1.07]),
    )
    spot_over_episode_origin = 1.08
    snapshot = day.hedge_snapshot(
        maximum_business_dte=5.0,
        spot_over_episode_origin=spot_over_episode_origin,
        episode_origin_date=date(2025, 1, 3),
    )
    contracts = tuple(quote.contract for quote in snapshot.option_quotes)
    inception_forward_curve = np.asarray([1.0, 1.02, 1.045, 1.075, 1.11, 1.15])
    current_normalized_spot = (
        spot_over_episode_origin / inception_forward_curve[1]
    )
    instruction = WeeklyFitInstruction(
        schedule=HedgeSchedule.RESTARTED_WEEKLY,
        model="historical-sanos",
        measure="historical-P",
        target_label="realized-variance",
        claim_quantity=1.0,
        model_config_identity="model",
        policy_config_identity="policy",
        episode_origin_date=date(2025, 1, 3),
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2024, 1, 2),),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    state = WeeklyDecisionState(
        instruction=instruction,
        decision_index=1,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=spot_over_episode_origin,
        inception_forward_over_episode_spot=inception_forward_curve,
        inception_forward_path=np.asarray([1.0, current_normalized_spot]),
        rolling_forward_path=np.asarray([1.0, 0.99]),
        discount_to_origin=1.0,
        option_marks_pv=np.asarray([quote.mark for quote in snapshot.option_quotes]),
        option_half_spreads_pv=np.asarray(
            [quote.half_spread for quote in snapshot.option_quotes]
        ),
        option_alive=np.ones(len(contracts), dtype=bool),
        previous_spot_position=0.0,
        previous_option_positions=np.zeros(len(contracts)),
    )
    historical_dates = tuple(
        add_business_days(date(2024, 1, 2), offset) for offset in range(5)
    )
    historical_forward_curve = np.asarray([1.003, 1.009, 1.018, 1.03, 1.045])
    historical_days = tuple(
        replace(
            day,
            trading_date=trading_date,
            forward_over_spot=historical_forward_curve,
        )
        for trading_date in historical_dates
    )
    returns = np.asarray([0.01, -0.004, 0.007, -0.002])
    one_day_forwards = np.asarray([1.001, 1.002, 1.003, 1.004])
    transitions = tuple(
        OneDayTransition(
            origin=historical_days[index],
            next_date=historical_dates[index + 1],
            forward_adjusted_spot_return=float(returns[index]),
            discount_factor=1.0,
            forward_over_spot=float(one_day_forwards[index]),
        )
        for index in range(4)
    )
    scenario_set = build_historical_weekly_scenarios(
        HistoricalWeeklyTransitionHistory(
            as_of_date=day.trading_date,
            transitions=transitions,
            endpoints=historical_days[1:],
        ),
        state=state,
        claim=build_weekly_claim(
            day,
            target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
        ),
        model_fit_identity="historical-forward-rescaling-test",
    )
    historical_spot_path = np.cumprod(one_day_forwards * (1.0 + returns))
    historical_forward_normalized_shocks = historical_spot_path / np.asarray(
        [
            historical_days[0].forward_at_business_dte(step)
            for step in range(1, 5)
        ]
    )
    expected_suffix = (
        spot_over_episode_origin
        * np.asarray([day.forward_at_business_dte(step) for step in range(1, 5)])
        * historical_forward_normalized_shocks
        / inception_forward_curve[2:]
    )
    np.testing.assert_allclose(
        scenario_set.scenarios.inception_forward_paths[0],
        np.concatenate(([current_normalized_spot], expected_suffix)),
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_restarted_pure_hedge_respects_the_actual_carried_book() -> None:
    scenario_count = 20
    fit_date = date(2022, 12, 30)
    scenarios = WeeklyScenarioBatch(
        inception_forward_paths=np.ones((scenario_count, 2)),
        spot_values_pv=np.ones((scenario_count, 2)),
        spot_step_gains=np.zeros((scenario_count, 1)),
        option_step_gains=np.zeros((scenario_count, 1, 1)),
        option_half_spreads_pv=np.broadcast_to(
            np.asarray([[[0.1], [0.0]]]),
            (scenario_count, 2, 1),
        ).copy(),
        option_alive=np.asarray([[True], [False]]),
        option_settles_next=np.asarray([[True]]),
        terminal_claim_values_pv=np.linspace(0.0, 0.1, scenario_count),
        contract_ids=("settling",),
        target_label="carried-book-test",
        fit_as_of_date=fit_date,
        model_fit_identity="carried-book-test",
        measure="historical-P",
        seed=None,
    )
    scenario_set = HistoricalWeeklyScenarioSet(
        scenarios=scenarios,
        training_origin_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 2)
            for index in range(scenario_count)
        ),
        training_terminal_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 1)
            for index in range(scenario_count)
        ),
    )
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.asarray([5.0]),
        config=HistoricalSanosConfig(
            rolling_observations=20,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            quadratic_ridge=0.0,
            maximum_iterations=2_000,
            gradient_tolerance=1.0e-9,
        ),
    )
    assert fit.long_policy.option_positions[0, 0] == pytest.approx(0.0, abs=1.0e-8)
    assert fit.zero_policy.option_positions[0, 0] == pytest.approx(0.0, abs=1.0e-8)
    assert fit.option_positions[0, 0] == pytest.approx(5.0, abs=1.0e-7)
    accounting = weekly_scenario_accounting(
        scenarios,
        spot_positions=fit.spot_positions,
        option_positions=fit.option_positions,
        signed_claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.asarray([5.0]),
    )
    np.testing.assert_allclose(accounting.transaction_costs, 0.0, atol=1.0e-8)


def test_mean_variance_prices_use_the_fitted_expected_cost_functional() -> None:
    base = _scenario_set()
    scenario_cost_multipliers = np.linspace(
        0.01,
        1.99,
        base.scenarios.scenario_count,
    )
    scenarios = replace(
        base.scenarios,
        spot_values_pv=(
            base.scenarios.spot_values_pv * scenario_cost_multipliers[:, None]
        ),
        option_half_spreads_pv=(
            base.scenarios.option_half_spreads_pv
            * scenario_cost_multipliers[:, None, None]
        ),
    )
    scenario_set = replace(base, scenarios=scenarios)
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
        quadratic_ridge=1.0e-6,
        maximum_iterations=2_000,
        gradient_tolerance=1.0e-8,
    )
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    assert fit.prices.buyer == pytest.approx(
        fit.long_policy.certainty_equivalent
        - fit.zero_policy.certainty_equivalent,
        abs=1.0e-12,
    )
    exact_long = weekly_scenario_accounting(
        scenarios,
        spot_positions=fit.long_policy.spot_positions,
        option_positions=fit.long_policy.option_positions,
        signed_claim_quantity=1.0,
    )
    exact_long_ce = weekly_certainty_equivalent(
        exact_long.terminal_wealth,
        objective=WeeklyRiskObjective.MEAN_VARIANCE,
        risk_parameter=config.risk_parameter,
    )
    assert abs(exact_long_ce - fit.long_policy.certainty_equivalent) > 1.0e-7
