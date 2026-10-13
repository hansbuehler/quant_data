"""Weekly benchmark adapter for historical SANOS scenario hedges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.uniquehash import named_unique_filename48_8

from quantdata.hedging.contracts import OptionContract
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyClaimDefinition,
    WeeklyFitInstruction,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    AsOfWeeklyMarketPanel,
    WeeklyDecisionState,
    WeeklyPolicyAction,
    WeeklyPriceRecord,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_hedger import (
    HistoricalWeeklyHedgeFit,
    fit_historical_weekly_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_scenarios import (
    HistoricalWeeklyScenarioSet,
    HistoricalWeeklyTransitionHistory,
    build_historical_weekly_scenarios,
)


@dataclass(frozen=True, kw_only=True)
class HistoricalSanosWeeklyMarketFit:
    """Target-independent rolling history available at one weekly fit date."""

    market_fit_identity: str
    history: HistoricalWeeklyTransitionHistory
    artifact_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class HistoricalSanosWeeklyPolicy:
    """One fitted remaining-horizon empirical policy."""

    market_fit_identity: str
    policy_fit_identity: str
    prices: WeeklyPriceRecord
    schedule: HedgeSchedule
    fit_decision_index: int
    scenarios: HistoricalWeeklyScenarioSet
    hedge: HistoricalWeeklyHedgeFit
    artifact_fingerprints: tuple[str, ...] = ()

    @property
    def converged(self) -> bool:
        """Return whether every fitted component passed its numerical gate."""
        return self.hedge.converged

    @property
    def cap_hits(self) -> int:
        """Return internal solver cap hits for this fit."""
        return self.hedge.internal_cap_hits


class HistoricalSanosWeeklyAdapter:
    """Fit rolling historical SANOS paths and open-loop weekly policies."""

    model = "historical-sanos"
    measure = "historical-P"

    def __init__(self, *, config: HistoricalSanosConfig = HistoricalSanosConfig()) -> None:
        self.config = config
        config_hash = named_unique_filename48_8(config.label, config)
        code_identity = config.version.unique_id48
        self.model_config_identity = (
            f"historical-sanos-market-{code_identity}-{config_hash}"
        )
        self.policy_config_identity = (
            f"historical-sanos-policy-{code_identity}-{config_hash}"
        )

    def fit_market(
        self,
        panel: AsOfWeeklyMarketPanel,
        *,
        instruction: WeeklyFitInstruction,
    ) -> HistoricalSanosWeeklyMarketFit:
        """Load only the precommitted rolling one-session history."""
        pairs = tuple(
            (
                transition,
                panel.market_day(transition.next_date),
            )
            for transition in (
                panel.one_day_transition(origin_date)
                for origin_date in instruction.training_dates
            )
        )
        return HistoricalSanosWeeklyMarketFit(
            market_fit_identity=instruction.market_fit_identity,
            history=HistoricalWeeklyTransitionHistory(
                as_of_date=instruction.fit_as_of_date,
                transitions=tuple(pair[0] for pair in pairs),
                endpoints=tuple(pair[1] for pair in pairs),
            ),
        )

    def fit_policy(
        self,
        market_fit: HistoricalSanosWeeklyMarketFit,
        *,
        instruction: WeeklyFitInstruction,
        claim: WeeklyClaimDefinition,
        contracts: tuple[OptionContract, ...],
        state: WeeklyDecisionState,
    ) -> HistoricalSanosWeeklyPolicy:
        """Fit long, short, and zero policies on complete historical blocks."""
        if market_fit.market_fit_identity != instruction.market_fit_identity:
            raise ValueError("historical weekly market fit does not match instruction")
        if tuple(contract.contract_id for contract in contracts) != (
            instruction.frozen_contract_ids
        ):
            raise ValueError("historical weekly contracts do not match instruction")
        scenarios = build_historical_weekly_scenarios(
            market_fit.history,
            state=state,
            claim=claim,
            model_fit_identity=market_fit.market_fit_identity,
        )
        hedge = fit_historical_weekly_hedge(
            scenarios,
            claim_quantity=instruction.claim_quantity,
            initial_spot_position=state.previous_spot_position,
            initial_option_positions=state.previous_option_positions,
            config=self.config,
        )
        buyer = hedge.prices.buyer / instruction.claim_quantity
        seller = hedge.prices.seller / instruction.claim_quantity
        continuation = 0.5 * (buyer + seller)
        return HistoricalSanosWeeklyPolicy(
            market_fit_identity=market_fit.market_fit_identity,
            policy_fit_identity=instruction.policy_fit_identity,
            prices=WeeklyPriceRecord(
                common_market_price_per_fit_spot=(
                    claim.market_price if state.decision_index == 0 else None
                ),
                q_expectation_per_fit_spot=None,
                buyer_indifference_per_fit_spot=float(buyer),
                seller_indifference_per_fit_spot=float(seller),
                continuation_price_per_fit_spot=float(continuation),
            ),
            schedule=instruction.schedule,
            fit_decision_index=state.decision_index,
            scenarios=scenarios,
            hedge=hedge,
        )

    def action(
        self,
        fit: HistoricalSanosWeeklyPolicy,
        *,
        state: WeeklyDecisionState,
    ) -> WeeklyPolicyAction:
        """Return the scheduled pure-hedge row at the current decision."""
        if fit.schedule is HedgeSchedule.FROZEN_WEEKLY:
            row = state.decision_index
        elif fit.schedule is HedgeSchedule.RESTARTED_WEEKLY:
            if fit.fit_decision_index != state.decision_index:
                raise ValueError("restarted historical policy is stale")
            row = 0
        else:
            raise ValueError("historical weekly policy requires a weekly schedule")
        if not 0 <= row < len(fit.hedge.spot_positions):
            raise ValueError("historical weekly policy row is unavailable")
        options = fit.hedge.option_positions[row].copy()
        options[~state.option_alive] = 0.0
        return WeeklyPolicyAction(
            spot_position=float(fit.hedge.spot_positions[row]),
            option_positions=np.asarray(options, dtype=np.float64),
        )
