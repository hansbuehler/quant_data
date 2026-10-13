"""Weekly GROMM Q expectation with conventional eight-factor DLV hedging."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

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
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    Gromm2MarketState,
    gromm2_market_state_with_dlvsq,
    load_gromm2_market_state,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_weekly import (
    generate_gromm2_weekly_scenarios,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Artifacts,
    RollingGromm2Config,
    load_rolling_gromm2,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import (
    DlvGreeks,
    dlv_option_greeks,
    solve_delta_factor_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_adapter import (
    SanosLvWeeklyAdapter,
    SanosLvWeeklyMarketFit,
)


@dataclass(frozen=True, kw_only=True)
class GrommConventionalWeeklyMarketFit:
    """Dated GROMM market state and contemporaneous eight-factor DLV basis."""

    market_fit_identity: str
    artifacts: RollingGromm2Artifacts
    market_state: Gromm2MarketState
    dlv_fit: SanosLvWeeklyMarketFit
    affine_history_sha256: str
    dynamics_sha256: str

    @property
    def artifact_fingerprints(self) -> tuple[str, ...]:
        """Return exact GROMM market-artifact fingerprints."""
        return (
            f"affine-history:{self.affine_history_sha256}",
            f"dynamics:{self.dynamics_sha256}",
        )


@dataclass(frozen=True, kw_only=True)
class GrommConventionalWeeklyPolicy:
    """One claim priced and differentiated under the same GROMM Q state."""

    market_fit_identity: str
    policy_fit_identity: str
    prices: WeeklyPriceRecord
    market_state: Gromm2MarketState
    factors: RollingDlvFactorFit
    claim: WeeklyClaimDefinition
    inception_target_greeks: DlvGreeks
    greek_decision_index: int
    artifact_fingerprints: tuple[str, ...] = ()

    @property
    def converged(self) -> bool:
        """Return the deterministic Greek-policy fit status."""
        return True

    @property
    def cap_hits(self) -> int:
        """Return target-independent fit cap hits."""
        return 0


class GrommConventionalWeeklyAdapter:
    """Differentiate the GROMM Q price along common rolling DLV bumps."""

    model = "gromm-q-dlv-vega"
    measure = "Q"

    def __init__(
        self,
        *,
        artifact_root: Path,
        gromm_config: GrommConfig,
        dlv_config: SanosLvConfig,
        rolling_config: RollingGromm2Config = RollingGromm2Config(),
    ) -> None:
        if gromm_config.measure is not GrommMeasure.Q:
            raise ValueError("conventional GROMM hedging requires the Q measure")
        if gromm_config.dlv_substeps != rolling_config.substeps:
            raise ValueError("conventional GROMM simulation substeps differ")
        self.artifact_root = Path(artifact_root)
        self.gromm_config = gromm_config
        self.dlv_config = dlv_config
        self.rolling_config = rolling_config
        self._dlv_adapter = SanosLvWeeklyAdapter(config=dlv_config)
        self.model_config_identity = "gromm-q-dlv-vega-market-" + (
            named_unique_filename48_8(
                gromm_config.label,
                gromm_config,
                rolling_config,
            )
        )
        self.policy_config_identity = "gromm-q-dlv-vega-policy-" + (
            named_unique_filename48_8(
                dlv_config.label,
                dlv_config,
                gromm_config.path_count,
            )
        )

    def fit_market(
        self,
        panel: AsOfWeeklyMarketPanel,
        *,
        instruction: WeeklyFitInstruction,
    ) -> GrommConventionalWeeklyMarketFit:
        """Load the dated Q state and fit only the contemporaneous DLV basis."""
        artifacts = load_rolling_gromm2(
            self.artifact_root,
            as_of_date=instruction.fit_as_of_date,
            config=self.rolling_config,
            dataset_manifest_sha256=panel.dataset_manifest_sha256,
        )
        if artifacts.transition_origin_dates != instruction.training_dates:
            raise ValueError(
                "rolling GROMM2 artifact does not match weekly training dates"
            )
        dlv_fit = self._dlv_adapter.fit_market(panel, instruction=instruction)
        return GrommConventionalWeeklyMarketFit(
            market_fit_identity=instruction.market_fit_identity,
            artifacts=artifacts,
            market_state=load_gromm2_market_state(artifacts),
            dlv_fit=dlv_fit,
            affine_history_sha256=file_sha256(artifacts.paths.affine_history),
            dynamics_sha256=file_sha256(artifacts.paths.dynamics),
        )

    def fit_policy(
        self,
        market_fit: GrommConventionalWeeklyMarketFit,
        *,
        instruction: WeeklyFitInstruction,
        claim: WeeklyClaimDefinition,
        contracts: tuple[OptionContract, ...],
        state: WeeklyDecisionState,
    ) -> GrommConventionalWeeklyPolicy:
        """Compute one GROMM Q price and its common factor sensitivities."""
        if market_fit.market_fit_identity != instruction.market_fit_identity:
            raise ValueError("conventional GROMM market fit does not match instruction")
        if tuple(contract.contract_id for contract in contracts) != (
            instruction.frozen_contract_ids
        ):
            raise ValueError("conventional GROMM contracts do not match instruction")
        target = self._target_greeks(
            market_state=market_fit.market_state,
            factors=market_fit.dlv_fit.factors,
            claim=claim,
            state=state,
        )
        q_expectation = float(target.model_prices[0])
        return GrommConventionalWeeklyPolicy(
            market_fit_identity=instruction.market_fit_identity,
            policy_fit_identity=instruction.policy_fit_identity,
            prices=WeeklyPriceRecord(
                common_market_price_per_fit_spot=(
                    claim.market_price if state.decision_index == 0 else None
                ),
                q_expectation_per_fit_spot=q_expectation,
                buyer_indifference_per_fit_spot=None,
                seller_indifference_per_fit_spot=None,
                continuation_price_per_fit_spot=q_expectation,
            ),
            market_state=market_fit.market_state,
            factors=market_fit.dlv_fit.factors,
            claim=claim,
            inception_target_greeks=target,
            greek_decision_index=state.decision_index,
        )

    def action(
        self,
        fit: GrommConventionalWeeklyPolicy,
        *,
        state: WeeklyDecisionState,
    ) -> WeeklyPolicyAction:
        """Return the predictable delta/eight-factor-vega hedge."""
        target = (
            fit.inception_target_greeks
            if state.decision_index == fit.greek_decision_index
            else self._target_greeks(
                market_state=fit.market_state,
                factors=fit.factors,
                claim=fit.claim,
                state=state,
            )
        )
        scaled_target = DlvGreeks(
            model_prices=state.instruction.claim_quantity * target.model_prices,
            spot_deltas=state.instruction.claim_quantity * target.spot_deltas,
            factor_vegas=state.instruction.claim_quantity * target.factor_vegas,
        )
        alive_indices = np.flatnonzero(state.option_alive)
        if len(alive_indices) == 0:
            raise ValueError("conventional GROMM policy has no live hedge options")
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
            raise ValueError(
                "conventional GROMM policy is missing a live contract"
            ) from error
        hedge_greeks = dlv_option_greeks(
            state.market_day,
            quotes=quotes,
            factors=fit.factors,
            relative_spot_step=self.dlv_config.relative_spot_step,
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
            residual_tolerance=self.dlv_config.exposure_residual_tolerance,
            position_cap=self.dlv_config.position_cap,
            previous_option_actions=state.previous_option_positions[alive_indices],
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
        market_state: Gromm2MarketState,
        factors: RollingDlvFactorFit,
        claim: WeeklyClaimDefinition,
        state: WeeklyDecisionState,
    ) -> DlvGreeks:
        live_market = _live_gromm_market_state(market_state, state=state)

        def price(
            current_market: Gromm2MarketState,
            current_state: WeeklyDecisionState,
        ) -> float:
            generated = generate_gromm2_weekly_scenarios(
                market_state=current_market,
                state=current_state,
                claim=claim,
                config=self.gromm_config,
                model_fit_identity=current_state.instruction.market_fit_identity,
            )
            fit_scale = (
                current_state.discount_to_origin
                * current_state.spot_over_episode_origin
            )
            return float(
                np.mean(generated.scenarios.terminal_claim_values_pv) / fit_scale
            )

        base_price = price(live_market, state)
        relative_step = self.dlv_config.relative_spot_step
        spot_up = _state_with_spot_multiplier(state, 1.0 + relative_step)
        spot_down = _state_with_spot_multiplier(state, 1.0 - relative_step)
        delta = (
            (1.0 + relative_step) * price(live_market, spot_up)
            - (1.0 - relative_step) * price(live_market, spot_down)
        ) / (2.0 * relative_step)
        factor_vegas = np.empty((1, factors.factor_rank), dtype=np.float64)
        for factor_index in range(factors.factor_rank):
            up = _market_state_with_factor_bump(
                live_market,
                state=state,
                factors=factors,
                factor_index=factor_index,
                standard_deviations=1.0,
            )
            down = _market_state_with_factor_bump(
                live_market,
                state=state,
                factors=factors,
                factor_index=factor_index,
                standard_deviations=-1.0,
            )
            factor_vegas[0, factor_index] = 0.5 * (
                price(up, state) - price(down, state)
            )
        return DlvGreeks(
            model_prices=np.asarray([base_price], dtype=np.float64),
            spot_deltas=np.asarray([delta], dtype=np.float64),
            factor_vegas=factor_vegas,
        )


def _live_gromm_market_state(
    market_state: Gromm2MarketState,
    *,
    state: WeeklyDecisionState,
) -> Gromm2MarketState:
    day = state.market_day
    expected_expiries = day.dlv_model_expiries_y
    if not np.allclose(
        expected_expiries,
        market_state.expiries_y,
        atol=2.0e-10,
        rtol=0.0,
    ):
        raise ValueError("live DLV expiries differ from the frozen GROMM geometry")
    return gromm2_market_state_with_dlvsq(
        market_state,
        dlvsq=day.dlvsq,
    )


def _market_state_with_factor_bump(
    market_state: Gromm2MarketState,
    *,
    state: WeeklyDecisionState,
    factors: RollingDlvFactorFit,
    factor_index: int,
    standard_deviations: float,
) -> Gromm2MarketState:
    if not 0 <= factor_index < factors.factor_rank:
        raise ValueError("conventional GROMM factor index is out of range")
    if factors.daily_loadings.shape[:2] != market_state.dlvsq[1:].shape:
        raise ValueError("conventional GROMM factor grid does not match DLV state")
    values = market_state.dlvsq.copy()
    values[1:] = np.exp(
        np.log(values[1:])
        + standard_deviations * factors.daily_loadings[..., factor_index]
    )
    values = np.clip(
        values,
        np.nextafter(state.market_day.minimum_dlvsq, np.inf),
        state.market_day.maximum_dlvsq,
    )
    return gromm2_market_state_with_dlvsq(market_state, dlvsq=values)


def _state_with_spot_multiplier(
    state: WeeklyDecisionState,
    multiplier: float,
) -> WeeklyDecisionState:
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("conventional GROMM spot multiplier is invalid")
    inception_path = state.inception_forward_path.copy()
    rolling_path = state.rolling_forward_path.copy()
    inception_path[-1] *= multiplier
    rolling_path[-1] *= multiplier
    return replace(
        state,
        spot_over_episode_origin=state.spot_over_episode_origin * multiplier,
        inception_forward_path=inception_path,
        rolling_forward_path=rolling_path,
    )
