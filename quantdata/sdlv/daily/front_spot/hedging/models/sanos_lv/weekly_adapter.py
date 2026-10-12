"""Weekly SANOS-LV factor-vega policy on the public hedging panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.uniquehash import named_unique_filename48_8

from quantdata.hedging.contracts import OptionContract
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
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorFit,
    fit_rolling_dlv_factors,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import (
    DlvGreeks,
    dlv_option_greeks,
    solve_delta_factor_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_conditional import (
    conditional_weekly_claim_greeks,
)


@dataclass(frozen=True, kw_only=True)
class SanosLvWeeklyMarketFit:
    """One target-independent rolling DLV factor basis."""

    market_fit_identity: str
    factors: RollingDlvFactorFit


@dataclass(frozen=True, kw_only=True)
class SanosLvWeeklyPolicy:
    """A fixed weekly target rule using one dated DLV factor basis."""

    market_fit_identity: str
    policy_fit_identity: str
    prices: WeeklyPriceRecord
    factors: RollingDlvFactorFit
    claim: WeeklyClaimDefinition


class SanosLvWeeklyAdapter:
    """Fit rolling DLV factors and hedge fixed claims by delta and factor vega."""

    model = "sanos-lv"
    measure = "Q"

    def __init__(self, *, config: SanosLvConfig = SanosLvConfig()) -> None:
        self.config = config
        config_hash = named_unique_filename48_8(config.label, config)
        code_identity = config.version.unique_id48
        self.model_config_identity = f"sanos-lv-market-{code_identity}-{config_hash}"
        self.policy_config_identity = f"sanos-lv-policy-{code_identity}-{config_hash}"

    def fit_market(
        self,
        panel: AsOfWeeklyMarketPanel,
        *,
        instruction: WeeklyFitInstruction,
    ) -> SanosLvWeeklyMarketFit:
        """Fit one rolling factor basis using only the instruction's dated panel."""
        factors = fit_rolling_dlv_factors(
            panel,
            as_of_date=instruction.fit_as_of_date,
            config=self.config.rolling_factors,
        )
        return SanosLvWeeklyMarketFit(
            market_fit_identity=instruction.market_fit_identity,
            factors=factors,
        )

    def fit_policy(
        self,
        market_fit: SanosLvWeeklyMarketFit,
        *,
        instruction: WeeklyFitInstruction,
        claim: WeeklyClaimDefinition,
        contracts: tuple[OptionContract, ...],
        state: WeeklyDecisionState,
    ) -> SanosLvWeeklyPolicy:
        """Bind one fixed claim and frozen option book to a rolling factor fit."""
        if market_fit.market_fit_identity != instruction.market_fit_identity:
            raise ValueError("SANOS-LV weekly market fit does not match instruction")
        if tuple(contract.contract_id for contract in contracts) != (
            instruction.frozen_contract_ids
        ):
            raise ValueError("SANOS-LV weekly contracts do not match instruction")
        target = self._target_greeks(
            claim=claim,
            state=state,
            factors=market_fit.factors,
        )
        common_market_price = (
            claim.market_price if state.decision_index == 0 else None
        )
        return SanosLvWeeklyPolicy(
            market_fit_identity=market_fit.market_fit_identity,
            policy_fit_identity=instruction.policy_fit_identity,
            prices=WeeklyPriceRecord(
                common_market_price_per_fit_spot=common_market_price,
                q_expectation_per_fit_spot=float(target.model_prices[0]),
                buyer_indifference_per_fit_spot=None,
                seller_indifference_per_fit_spot=None,
                continuation_price_per_fit_spot=float(target.model_prices[0]),
            ),
            factors=market_fit.factors,
            claim=claim,
        )

    def action(
        self,
        fit: SanosLvWeeklyPolicy,
        *,
        state: WeeklyDecisionState,
    ) -> WeeklyPolicyAction:
        """Recompute conditional target Greeks and return predictable holdings."""
        target = self._target_greeks(
            claim=fit.claim,
            state=state,
            factors=fit.factors,
        )
        scaled_target = DlvGreeks(
            model_prices=state.instruction.claim_quantity * target.model_prices,
            spot_deltas=state.instruction.claim_quantity * target.spot_deltas,
            factor_vegas=state.instruction.claim_quantity * target.factor_vegas,
        )
        alive_indices = np.flatnonzero(state.option_alive)
        if len(alive_indices) == 0:
            raise ValueError("SANOS-LV weekly policy has no live hedge options")
        snapshot = state.market_day.reprice_contracts(
            state.contracts,
            spot_over_episode_origin=state.spot_over_episode_origin,
        )
        quote_by_id = snapshot.quote_by_id()
        try:
            quotes = tuple(
                quote_by_id[state.contracts[index].contract_id]
                for index in alive_indices
            )
        except KeyError as error:
            raise ValueError("SANOS-LV weekly policy is missing a live contract") from error
        hedge_greeks = dlv_option_greeks(
            state.market_day,
            quotes=quotes,
            factors=fit.factors,
            relative_spot_step=self.config.relative_spot_step,
        )
        option_cost_weights = np.asarray(
            [
                quote.half_spread
                / state.spot_over_episode_origin
                * (
                    1.0
                    if quote.contract.expiry_date == state.next_session_date
                    else 2.0
                )
                for quote in quotes
            ],
            dtype=np.float64,
        )
        hedge = solve_delta_factor_hedge(
            target=scaled_target,
            hedge_options=hedge_greeks,
            option_cost_weights=option_cost_weights,
            residual_tolerance=self.config.exposure_residual_tolerance,
            position_cap=self.config.position_cap,
            previous_option_actions=state.previous_option_positions[
                alive_indices
            ],
        )
        option_positions = np.zeros(len(state.contracts), dtype=np.float64)
        option_positions[alive_indices] = hedge.option_actions
        return WeeklyPolicyAction(
            spot_position=hedge.spot_action,
            option_positions=option_positions,
        )

    def _target_greeks(
        self,
        *,
        claim: WeeklyClaimDefinition,
        state: WeeklyDecisionState,
        factors: RollingDlvFactorFit,
    ) -> DlvGreeks:
        return conditional_weekly_claim_greeks(
            state.market_day,
            claim=claim,
            decision_index=state.decision_index,
            realized_inception_forward_path=state.inception_forward_path,
            inception_forward_over_episode_spot=(
                state.inception_forward_over_episode_spot
            ),
            spot_over_episode_origin=state.spot_over_episode_origin,
            factors=factors,
            config=self.config,
        )
