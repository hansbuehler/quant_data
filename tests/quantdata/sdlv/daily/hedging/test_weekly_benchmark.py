from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pytest

from quantdata.biz_calendar import count_business_days_between
from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WEEKLY_HORIZON,
    WeeklyPathBasis,
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
    build_weekly_episode_path,
    evaluate_weekly_hedge,
    overlapping_weekly_origins,
    rolling_training_dates,
    weekly_fit_instructions,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    AsOfWeeklyMarketPanel,
    WeeklyDecisionState,
    WeeklyPolicyAction,
    WeeklyPriceRecord,
    run_weekly_model,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyRiskObjective,
    weekly_certainty_equivalent,
    weekly_scenario_accounting,
    weekly_scenario_batch_from_episode,
    weekly_target_values_from_suffixes,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)


class _SyntheticWeeklyPanel:
    def __init__(
        self,
        *,
        market_days: dict[date, PublicMarketDay],
        transitions: dict[date, OneDayTransition],
    ) -> None:
        self._market_days = market_days
        self._transitions = transitions
        self.dates = tuple(sorted(market_days))
        self.geometry = {"schema": HEDGING_PUBLIC_SCHEMA}

    def market_day(self, trading_date: date) -> PublicMarketDay:
        return self._market_days[trading_date]

    def one_day_transition(self, origin_date: date) -> OneDayTransition:
        return self._transitions[origin_date]

    def prior_one_day_origins(
        self,
        origin_date: date,
        *,
        count: int,
    ) -> tuple[date, ...]:
        return tuple(
            origin_date - timedelta(days=offset)
            for offset in range(count, 0, -1)
        )

    def next_one_session_date(self, origin_date: date) -> date:
        return self._transitions[origin_date].next_date


@dataclass(frozen=True, kw_only=True)
class _TestMarketFit:
    market_fit_identity: str
    artifact_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class _TestFit:
    market_fit_identity: str
    policy_fit_identity: str
    prices: WeeklyPriceRecord
    artifact_fingerprints: tuple[str, ...] = ()


class _TestWeeklyAdapter:
    model = "test-weekly"
    measure = "Q"
    model_config_identity = "model-v1"
    policy_config_identity = "policy-v1"

    def __init__(self) -> None:
        self.market_fit_calls: list[date] = []
        self.policy_fit_calls: list[date] = []
        self.fit_panel_last_dates: list[date] = []
        self.fit_path_lengths: list[int] = []
        self.action_path_lengths: list[int] = []

    def fit_market(self, panel, *, instruction) -> _TestMarketFit:
        self.market_fit_calls.append(instruction.fit_as_of_date)
        self.fit_panel_last_dates.append(panel.dates[-1])
        return _TestMarketFit(
            market_fit_identity=instruction.market_fit_identity,
        )

    def fit_policy(
        self,
        market_fit,
        *,
        instruction,
        claim,
        contracts,
        state,
    ) -> _TestFit:
        del claim, contracts
        self.policy_fit_calls.append(instruction.fit_as_of_date)
        self.fit_path_lengths.append(len(state.inception_forward_path))
        price = 0.01 + 0.001 * len(self.policy_fit_calls)
        return _TestFit(
            market_fit_identity=market_fit.market_fit_identity,
            policy_fit_identity=instruction.policy_fit_identity,
            prices=WeeklyPriceRecord(
                common_market_price_per_fit_spot=price,
                q_expectation_per_fit_spot=price,
                buyer_indifference_per_fit_spot=None,
                seller_indifference_per_fit_spot=None,
                continuation_price_per_fit_spot=price,
            ),
        )

    def action(
        self,
        fit: _TestFit,
        *,
        state: WeeklyDecisionState,
    ) -> WeeklyPolicyAction:
        del fit
        self.action_path_lengths.append(len(state.inception_forward_path))
        return WeeklyPolicyAction(
            spot_position=0.0,
            option_positions=np.zeros(len(state.contracts)),
        )


def _market_day(
    *,
    trading_date: date,
    expiry_dates: tuple[date, ...],
) -> PublicMarketDay:
    future_expiries = tuple(value for value in expiry_dates if value > trading_date)
    business_dtes = np.asarray(
        [
            count_business_days_between(trading_date, expiry_date)
            for expiry_date in future_expiries
        ],
        dtype=np.float64,
    )
    expiry_count = len(future_expiries)
    expiry_indices = np.repeat(np.arange(expiry_count), 2)
    strikes = np.tile(np.asarray([0.95, 1.05]), expiry_count)
    is_call = np.tile(np.asarray([False, True]), expiry_count)
    support = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4])
    density = np.asarray([0.0, 0.25, 0.5, 0.25, 0.0])
    model_dtes = np.arange(1, 9, dtype=np.float64)
    minimum = 0.01**2
    return PublicMarketDay(
        trading_date=trading_date,
        expiry_dates=future_expiries,
        expiries_y=business_dtes / 252.0,
        business_dtes=business_dtes,
        is_am_settlement=np.zeros(expiry_count, dtype=bool),
        forward_over_spot=1.0 + 0.001 * business_dtes,
        discount_factors=1.0 - 0.001 * business_dtes,
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.full(expiry_count, 0.2),
        expiry_indices=expiry_indices,
        strikes_over_forward=strikes,
        standardized_log_moneyness=np.tile(
            np.asarray([-0.5, 0.5]), expiry_count
        ),
        in_recommended_hedge_nms_population=np.ones(2 * expiry_count, dtype=bool),
        is_call=is_call,
        half_spreads_over_spot=np.full(2 * expiry_count, 0.001),
        sanos_prices_over_discounted_forward=np.full(2 * expiry_count, 0.03),
        dlv_prices_over_discounted_forward=np.full(2 * expiry_count, 0.03),
        sanos_model_expiries_y=model_dtes / 252.0,
        sanos_support_strikes=tuple(support.copy() for _ in model_dtes),
        sanos_kernel_total_variances=0.2**2 * model_dtes / 252.0,
        sanos_densities=tuple(density.copy() for _ in model_dtes),
        sanos_smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=model_dtes / 252.0,
        dlv_support_strikes=support,
        dlv_raw_log_excess_dlvsq=np.log(
            np.full((len(model_dtes), 3), 0.2**2 - minimum)
        ),
        minimum_dlvsq=minimum,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )


@pytest.fixture
def weekly_panel() -> tuple[_SyntheticWeeklyPanel, tuple[date, ...]]:
    dates = (
        date(2023, 2, 6),
        date(2023, 2, 7),
        date(2023, 2, 8),
        date(2023, 2, 9),
        date(2023, 2, 10),
        date(2023, 2, 13),
    )
    expiry_dates = (*dates[1:], date(2023, 2, 14), date(2023, 2, 15))
    return _build_panel(dates=dates, expiry_dates=expiry_dates), dates


def _build_panel(
    *,
    dates: tuple[date, ...],
    expiry_dates: tuple[date, ...],
) -> _SyntheticWeeklyPanel:
    market_days = {
        trading_date: _market_day(
            trading_date=trading_date,
            expiry_dates=expiry_dates,
        )
        for trading_date in dates
    }
    return_pattern = (0.01, -0.005, 0.002, -0.003, 0.004)
    returns = tuple(
        return_pattern[index % len(return_pattern)]
        for index in range(len(dates) - 1)
    )
    transitions = {
        origin_date: OneDayTransition(
            origin=market_days[origin_date],
            next_date=next_date,
            forward_adjusted_spot_return=return_value,
            discount_factor=0.999,
            forward_over_spot=1.001,
        )
        for origin_date, next_date, return_value in zip(
            dates[:-1], dates[1:], returns, strict=True
        )
    }
    return _SyntheticWeeklyPanel(
        market_days=market_days,
        transitions=transitions,
    )


def test_weekly_catalog_contains_all_agreed_targets() -> None:
    targets = weekly_targets()
    assert len(targets) == 13
    assert len({target.label for target in targets}) == 13
    digitals = tuple(
        target for target in targets if target.kind is WeeklyTargetKind.DIGITAL
    )
    assert tuple(
        target.nms for target in digitals if target.right is OptionRight.PUT
    ) == (0.0, -1.0, -0.75, -0.5, -0.25)
    assert tuple(
        target.nms for target in digitals if target.right is OptionRight.CALL
    ) == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_weekly_claims_fix_origin_geometry_and_path_basis(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    origin = panel.market_day(dates[0])
    digital = build_weekly_claim(
        origin,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    assert digital.path_basis is WeeklyPathBasis.INCEPTION_FORWARD
    assert digital.claim.strike == pytest.approx(1.0)
    assert digital.market_price is not None and digital.market_price > 0.0

    variance = build_weekly_claim(
        origin,
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    assert variance.path_basis is WeeklyPathBasis.INCEPTION_FORWARD
    assert variance.market_price is None

    down_and_out = build_weekly_claim(
        origin,
        target=WeeklyTarget(kind=WeeklyTargetKind.DOWN_AND_OUT_PUT),
    )
    assert down_and_out.claim.strike == 1.0
    assert down_and_out.claim.barrier == 0.95
    assert down_and_out.path_basis is WeeklyPathBasis.INCEPTION_FORWARD


@pytest.mark.parametrize(
    "schedule",
    (HedgeSchedule.FROZEN_WEEKLY, HedgeSchedule.RESTARTED_WEEKLY),
)
def test_weekly_episode_freezes_inception_contracts_and_refit_schedule(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
    schedule: HedgeSchedule,
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=schedule,
        rolling_window_business_days=3,
    )
    assert episode.spec.decision_dates == dates[:-1]
    assert episode.spec.terminal_date == dates[-1]
    assert len(episode.contracts) == 10
    assert episode.option_alive.sum(axis=1).tolist() == [10, 8, 6, 4, 2, 0]
    assert episode.option_settles_next.sum(axis=1).tolist() == [2, 2, 2, 2, 2]
    assert tuple(contract.contract_id for contract in episode.contracts) == (
        episode.spec.frozen_contract_ids
    )
    expected_refits = dates[:-1] if schedule is HedgeSchedule.RESTARTED_WEEKLY else dates[:1]
    assert episode.spec.refit_dates == expected_refits
    for refit_date in expected_refits:
        training_dates = rolling_training_dates(
            panel,
            episode=episode,
            decision_date=refit_date,
        )
        assert len(training_dates) == 3
        assert training_dates[-1] < refit_date


@pytest.mark.parametrize(
    "schedule, expected_identity_count",
    (
        (HedgeSchedule.FROZEN_WEEKLY, 1),
        (HedgeSchedule.RESTARTED_WEEKLY, WEEKLY_HORIZON),
    ),
)
def test_weekly_fit_identity_enforces_schedule(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
    schedule: HedgeSchedule,
    expected_identity_count: int,
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=schedule,
        rolling_window_business_days=3,
    )
    instructions = weekly_fit_instructions(
        panel,
        episode=episode,
        model="test-model",
        measure="Q",
        target_label="test-claim",
        model_config_identity="model-v1",
        policy_config_identity="policy-v1",
    )
    assert tuple(item.decision_date for item in instructions) == dates[:-1]
    assert len({item.fit_identity for item in instructions}) == expected_identity_count
    assert all(
        item.frozen_contract_ids == episode.spec.frozen_contract_ids
        for item in instructions
    )
    if schedule is HedgeSchedule.FROZEN_WEEKLY:
        assert all(item.fit_as_of_date == dates[0] for item in instructions)
        assert all(
            item.training_dates == episode.spec.training_dates
            for item in instructions
        )
    else:
        assert tuple(item.fit_as_of_date for item in instructions) == dates[:-1]
        assert all(
            item.training_dates[-1] < item.fit_as_of_date
            for item in instructions
        )


def test_weekly_policy_identity_includes_claim_quantity(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    common = {
        "panel": panel,
        "episode": episode,
        "model": "test-model",
        "measure": "Q",
        "target_label": "test-claim",
        "model_config_identity": "model-v1",
        "policy_config_identity": "policy-v1",
    }
    unit = weekly_fit_instructions(**common, claim_quantity=1.0)[0]
    double = weekly_fit_instructions(**common, claim_quantity=2.0)[0]
    assert unit.market_fit_identity == double.market_fit_identity
    assert unit.policy_fit_identity != double.policy_fit_identity


def test_weekly_as_of_panel_rejects_future_states_and_transitions(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    as_of = AsOfWeeklyMarketPanel(
        panel,
        as_of_date=dates[2],
        training_dates=dates[:2],
    )
    assert as_of.dates == dates[:3]
    require_public_hedging_panel(as_of)
    assert as_of.market_day(dates[2]).trading_date == dates[2]
    assert as_of.one_day_transition(dates[1]).next_date == dates[2]
    assert as_of.next_one_session_date(dates[1]) == dates[2]
    assert as_of.next_one_session_date(dates[2]) == dates[3]
    with pytest.raises(ValueError, match="after weekly fit cutoff"):
        as_of.market_day(dates[3])
    with pytest.raises(ValueError, match="outside the weekly fit training window"):
        as_of.one_day_transition(dates[2])


def test_weekly_as_of_panel_rejects_earlier_dates_outside_rolling_window(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    as_of = AsOfWeeklyMarketPanel(
        panel,
        as_of_date=dates[3],
        training_dates=dates[1:3],
    )
    with pytest.raises(ValueError, match="outside the weekly fit information window"):
        as_of.market_day(dates[0])
    with pytest.raises(ValueError, match="outside the weekly fit training window"):
        as_of.one_day_transition(dates[0])
    assert as_of.prior_one_day_origins(dates[3], count=2) == dates[1:3]


@pytest.mark.parametrize(
    "schedule, expected_fit_calls",
    (
        (HedgeSchedule.FROZEN_WEEKLY, 1),
        (HedgeSchedule.RESTARTED_WEEKLY, WEEKLY_HORIZON),
    ),
)
def test_weekly_runner_owns_refit_schedule_and_hides_future_path(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
    schedule: HedgeSchedule,
    expected_fit_calls: int,
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=schedule,
        rolling_window_business_days=3,
    )
    claim = build_weekly_claim(
        episode.market_days[0],
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    adapter = _TestWeeklyAdapter()
    run = run_weekly_model(
        panel,
        episode=episode,
        claim=claim,
        adapter=adapter,
    )
    assert len(adapter.market_fit_calls) == expected_fit_calls
    assert len(adapter.policy_fit_calls) == expected_fit_calls
    assert adapter.fit_panel_last_dates == adapter.market_fit_calls
    assert adapter.fit_path_lengths == (
        [1]
        if schedule is HedgeSchedule.FROZEN_WEEKLY
        else [1, 2, 3, 4, 5]
    )
    assert adapter.action_path_lengths == [1, 2, 3, 4, 5]
    assert len(set(run.market_fit_identities)) == expected_fit_calls
    assert len(set(run.policy_fit_identities)) == expected_fit_calls
    assert run.fit_as_of_dates == tuple(
        value.isoformat()
        for value in (
            dates[:1] * WEEKLY_HORIZON
            if schedule is HedgeSchedule.FROZEN_WEEKLY
            else dates[:-1]
        )
    )
    assert np.all(run.spot_positions == 0.0)
    assert np.all(run.option_positions == 0.0)
    np.testing.assert_allclose(
        run.fit_long_prices_origin_pv,
        np.asarray(
            [
                price.long_inception_price_per_fit_spot
                for price in run.fit_price_records
            ]
        )
        * np.asarray(
            [
                episode.discount_to_origin[
                    episode.spec.decision_dates.index(instruction.fit_as_of_date)
                ]
                * episode.spot_over_episode_origin[
                    episode.spec.decision_dates.index(instruction.fit_as_of_date)
                ]
                for instruction in weekly_fit_instructions(
                    panel,
                    episode=episode,
                    model=adapter.model,
                    measure=adapter.measure,
                    target_label=claim.target.label,
                    model_config_identity=adapter.model_config_identity,
                    policy_config_identity=adapter.policy_config_identity,
                )
            ]
        ),
    )


def test_weekly_market_fit_cache_is_shared_across_targets(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    claims = tuple(
        build_weekly_claim(
            episode.market_days[0],
            target=WeeklyTarget(
                kind=WeeklyTargetKind.DIGITAL,
                right=OptionRight.CALL,
                nms=nms,
            ),
        )
        for nms in (0.0, 0.5)
    )
    adapter = _TestWeeklyAdapter()
    market_cache = {}
    policy_cache = {}
    runs = tuple(
        run_weekly_model(
            panel,
            episode=episode,
            claim=claim,
            adapter=adapter,
            market_fit_cache=market_cache,
            policy_fit_cache=policy_cache,
        )
        for claim in claims
    )

    assert len(adapter.market_fit_calls) == 1
    assert len(adapter.policy_fit_calls) == 2
    assert len(market_cache) == 1
    assert len(policy_cache) == 2
    assert runs[0].market_fit_identities == runs[1].market_fit_identities
    assert runs[0].policy_fit_identities != runs[1].policy_fit_identities


def test_weekly_accounting_rebalances_and_settles_without_expiry_cost(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    claim = build_weekly_claim(
        episode.market_days[0],
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    spot_positions = np.ones(WEEKLY_HORIZON)
    option_positions = np.zeros((WEEKLY_HORIZON, len(episode.contracts)))
    first_expiry_index = next(
        index
        for index, contract in enumerate(episode.contracts)
        if contract.expiry_date == dates[1]
    )
    option_positions[0, first_expiry_index] = 1.0
    result = evaluate_weekly_hedge(
        episode,
        claim=claim,
        spot_positions=spot_positions,
        option_positions=option_positions,
        market_price=0.0,
        model_price=0.0,
    )
    expected_variance = np.mean(
        np.square(
            np.diff(episode.inception_forward_normalized_spot_path)
            / episode.inception_forward_normalized_spot_path[:-1]
        )
    )
    assert result.terminal_value == pytest.approx(
        episode.discount_to_origin[-1] * expected_variance
    )
    expected_spot_cost = 1.0e-4 * (
        1.0
        + episode.discount_to_origin[-1] * episode.spot_over_episode_origin[-1]
    )
    expected_option_cost = episode.option_half_spreads_pv[0, first_expiry_index]
    assert result.spot_transaction_cost == pytest.approx(expected_spot_cost)
    assert result.option_transaction_cost == pytest.approx(expected_option_cost)
    assert result.option_gain == pytest.approx(
        episode.option_step_gains[0, first_expiry_index]
    )
    assert result.spot_gain == pytest.approx(np.sum(episode.spot_step_gains))
    assert result.transaction_cost == pytest.approx(np.sum(result.step_transaction_costs))
    assert result.hedge_gain == pytest.approx(np.sum(result.step_hedge_gains))
    assert result.breakdown.pricing_contribution == 0.0
    assert result.breakdown.economic_pnl == pytest.approx(
        result.terminal_value + result.hedge_gain - result.transaction_cost
    )


def test_weekly_scenario_accounting_matches_realized_engine(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    claim = build_weekly_claim(
        episode.market_days[0],
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    spot_positions = np.asarray([0.5, 0.5, -0.25, -0.25, 0.0])
    option_positions = np.zeros((WEEKLY_HORIZON, len(episode.contracts)))
    for step in range(WEEKLY_HORIZON):
        option_positions[step, episode.option_alive[step]] = 0.1 * (step + 1)
    realized = evaluate_weekly_hedge(
        episode,
        claim=claim,
        spot_positions=spot_positions,
        option_positions=option_positions,
        market_price=0.0,
        model_price=0.0,
    )
    scenarios = weekly_scenario_batch_from_episode(episode, claim=claim)
    accounting = weekly_scenario_accounting(
        scenarios,
        spot_positions=spot_positions,
        option_positions=option_positions,
        signed_claim_quantity=1.0,
    )

    assert accounting.signed_claim_values[0] == pytest.approx(
        realized.terminal_value
    )
    assert accounting.hedge_gains[0] == pytest.approx(realized.hedge_gain)
    assert accounting.transaction_costs[0] == pytest.approx(
        realized.transaction_cost
    )
    np.testing.assert_allclose(
        accounting.step_hedge_gains[0],
        realized.step_hedge_gains,
    )
    np.testing.assert_allclose(
        accounting.step_transaction_costs[0],
        realized.step_transaction_costs,
    )


def test_weekly_scenario_accounting_starts_from_existing_positions(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    claim = build_weekly_claim(
        episode.market_days[0],
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    scenarios = weekly_scenario_batch_from_episode(episode, claim=claim)
    initial_options = 0.25 * scenarios.option_alive[0]
    option_positions = 0.25 * scenarios.option_alive[:-1]
    accounting = weekly_scenario_accounting(
        scenarios,
        spot_positions=np.full(WEEKLY_HORIZON, 0.5),
        option_positions=option_positions,
        signed_claim_quantity=1.0,
        initial_spot_position=0.5,
        initial_option_positions=initial_options,
    )
    assert accounting.step_transaction_costs[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("objective", tuple(WeeklyRiskObjective))
def test_weekly_certainty_equivalent_is_translation_equivariant(
    objective: WeeklyRiskObjective,
) -> None:
    wealth = np.asarray([-0.03, -0.01, 0.0, 0.02, 0.05])
    shift = 0.017
    base = weekly_certainty_equivalent(
        wealth,
        objective=objective,
        risk_parameter=2.0,
    )
    shifted = weekly_certainty_equivalent(
        wealth + shift,
        objective=objective,
        risk_parameter=2.0,
    )
    assert shifted == pytest.approx(base + shift, abs=1.0e-11)


def test_weekly_suffix_evaluation_preserves_barrier_breach_prefix(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    claim = build_weekly_claim(
        panel.market_day(dates[0]),
        target=WeeklyTarget(kind=WeeklyTargetKind.DOWN_AND_OUT_PUT),
    )
    values = weekly_target_values_from_suffixes(
        claim,
        realized_path_prefix=np.asarray([1.0, 0.94]),
        future_path_suffixes=np.asarray(
            [
                [0.98, 0.99, 1.00, 1.01],
                [1.02, 1.03, 1.04, 1.05],
            ]
        ),
        terminal_discount_to_episode_origin=0.99,
    )
    np.testing.assert_array_equal(values, np.zeros(2))


def test_weekly_claim_path_uses_the_inception_forward_curve(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    origin = episode.market_days[0]
    expected = np.asarray(
        [
            1.0,
            *(
                episode.spot_over_episode_origin[step]
                / origin.forward_at_business_dte(step)
                for step in range(1, WEEKLY_HORIZON + 1)
            ),
        ]
    )
    np.testing.assert_allclose(
        episode.inception_forward_normalized_spot_path,
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )
    rolling_terminal = np.prod(
        [transition.terminal_normalized_spot for transition in episode.transitions]
    )
    assert episode.rolling_forward_adjusted_spot_path[-1] == pytest.approx(
        rolling_terminal
    )
    assert episode.inception_forward_normalized_spot_path[-1] != pytest.approx(
        episode.rolling_forward_adjusted_spot_path[-1],
        abs=1.0e-8,
    )


def test_weekly_policy_cannot_reopen_an_expired_contract(
    weekly_panel: tuple[_SyntheticWeeklyPanel, tuple[date, ...]],
) -> None:
    panel, dates = weekly_panel
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    claim = build_weekly_claim(
        episode.market_days[0],
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    option_positions = np.zeros((WEEKLY_HORIZON, len(episode.contracts)))
    expired_index = next(
        index
        for index, contract in enumerate(episode.contracts)
        if contract.expiry_date == dates[1]
    )
    option_positions[1, expired_index] = 1.0
    with pytest.raises(ValueError, match="expired frozen contract"):
        evaluate_weekly_hedge(
            episode,
            claim=claim,
            spot_positions=np.zeros(WEEKLY_HORIZON),
            option_positions=option_positions,
            market_price=0.0,
            model_price=0.0,
        )


def test_weekly_first_later_expiry_is_closed_at_terminal() -> None:
    dates = (
        date(2023, 2, 6),
        date(2023, 2, 7),
        date(2023, 2, 8),
        date(2023, 2, 9),
        date(2023, 2, 10),
        date(2023, 2, 13),
    )
    panel = _build_panel(
        dates=dates,
        expiry_dates=(
            *dates[1:-1],
            date(2023, 2, 14),
            date(2023, 2, 15),
        ),
    )
    episode = build_weekly_episode_path(
        panel,
        origin_date=dates[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    assert episode.option_alive.sum(axis=1).tolist() == [10, 8, 6, 4, 2, 2]
    first_later_index = next(
        index
        for index, contract in enumerate(episode.contracts)
        if contract.expiry_date == date(2023, 2, 14)
    )
    claim = build_weekly_claim(
        episode.market_days[0],
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    positions = np.zeros((WEEKLY_HORIZON, len(episode.contracts)))
    positions[:, first_later_index] = 1.0
    result = evaluate_weekly_hedge(
        episode,
        claim=claim,
        spot_positions=np.zeros(WEEKLY_HORIZON),
        option_positions=positions,
        market_price=0.0,
        model_price=0.0,
    )
    assert result.option_transaction_cost == pytest.approx(
        episode.option_half_spreads_pv[0, first_later_index]
        + episode.option_half_spreads_pv[-1, first_later_index]
    )
    assert result.step_transaction_costs[-1] == pytest.approx(
        episode.option_half_spreads_pv[-1, first_later_index]
    )


def test_weekly_origins_include_every_overlapping_episode() -> None:
    dates = (
        date(2023, 2, 6),
        date(2023, 2, 7),
        date(2023, 2, 8),
        date(2023, 2, 9),
        date(2023, 2, 10),
        date(2023, 2, 13),
        date(2023, 2, 14),
    )
    panel = _build_panel(
        dates=dates,
        expiry_dates=(*dates[1:], date(2023, 2, 15)),
    )
    origins = overlapping_weekly_origins(
        panel,
        evaluation_start=dates[0],
        evaluation_end=dates[-1],
        rolling_window_business_days=3,
    )
    assert origins == dates[:2]
    first = build_weekly_episode_path(
        panel,
        origin_date=origins[0],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    second = build_weekly_episode_path(
        panel,
        origin_date=origins[1],
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        rolling_window_business_days=3,
    )
    assert first.spec.decision_dates[1:] == second.spec.decision_dates[:-1]
