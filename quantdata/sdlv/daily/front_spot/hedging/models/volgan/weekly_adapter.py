"""Weekly benchmark adapter for recursively sampled VolGAN paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyScenarioBatch,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly import (
    GeneratedWeeklyScenarioCache,
    attach_generated_weekly_claim,
    generated_live_state_fingerprint,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly_policy import (
    GeneratedWeeklyPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_hedger import (
    HistoricalWeeklyHedgeFit,
    WeeklyOpenLoopPricingContext,
    fit_weekly_open_loop_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.adapter import (
    VolganPublicFit,
    fit_public_volgan,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    VolganLiveCondition,
    build_volgan_live_condition,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.weekly_paths import (
    sample_volgan_weekly_paths,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class VolganWeeklyMarketFit:
    """One dated VolGAN fit and live recursive state."""

    market_fit_identity: str
    fit: VolganPublicFit
    live: VolganLiveCondition
    checkpoint_sha256: str

    @property
    def artifact_fingerprints(self) -> tuple[str, ...]:
        """Return the immutable dated-model checkpoint fingerprint."""
        return (f"checkpoint:{self.checkpoint_sha256}",)

    @property
    def live_state_fingerprint(self) -> str:
        """Return the immutable path-conditioning state fingerprint."""
        return generated_live_state_fingerprint(
            model="volgan",
            as_of_date=self.live.as_of_date,
            forecast_date=self.live.forecast_date,
            arrays=(self.live.values, self.live.trailing_forward_log_returns),
        )


@dataclass(frozen=True, kw_only=True)
class VolganWeeklyPolicy:
    """One fitted remaining-horizon VolGAN open-loop policy."""

    market_fit_identity: str
    policy_fit_identity: str
    prices: WeeklyPriceRecord
    schedule: HedgeSchedule
    fit_decision_index: int
    scenarios: WeeklyScenarioBatch
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


class VolganWeeklyAdapter:
    """Fit rolling VolGAN markets and weekly generated-scenario policies."""

    model = "volgan"
    measure = "historical-P"

    def __init__(
        self,
        *,
        model_config: VolganModelConfig,
        checkpoint_directory: Path,
        train_config: VolganTrainConfig = VolganTrainConfig(),
        policy_config: GeneratedWeeklyPolicyConfig = GeneratedWeeklyPolicyConfig(),
        observation_count: int = 252,
        warm_start_checkpoint: Path | None = None,
        expected_warm_start_sha256: str | None = None,
        environment: BaseEnvironment | None = None,
    ) -> None:
        self.model_config = model_config
        self.checkpoint_directory = Path(checkpoint_directory)
        self.train_config = train_config
        self.policy_config = policy_config
        self.observation_count = observation_count
        self.warm_start_checkpoint = warm_start_checkpoint
        self.expected_warm_start_sha256 = expected_warm_start_sha256
        self.environment = environment
        self._weekly_scenario_cache = GeneratedWeeklyScenarioCache()
        self._weekly_pricing_contexts: dict[int, WeeklyOpenLoopPricingContext] = {}
        self._weekly_cache_generation = self._weekly_scenario_cache.generation
        code_identity = GeneratedWeeklyPolicyConfig.version.unique_id48
        self.model_config_identity = (
            "volgan-weekly-market-"
            + named_unique_filename48_8(
                model_config.label,
                model_config,
                train_config,
                observation_count,
                expected_warm_start_sha256,
            )
        )
        self.policy_config_identity = (
            f"volgan-weekly-policy-{code_identity}-"
            + named_unique_filename48_8(
                model_config.label,
                policy_config,
            )
        )

    def fit_market(
        self,
        panel: AsOfWeeklyMarketPanel,
        *,
        instruction: WeeklyFitInstruction,
    ) -> VolganWeeklyMarketFit:
        """Fit on the restricted panel and retain only its dated live state."""
        public_fit = fit_public_volgan(
            panel,
            as_of_date=instruction.fit_as_of_date,
            model_config=self.model_config,
            checkpoint_directory=self.checkpoint_directory,
            observation_count=self.observation_count,
            train_config=self.train_config,
            warm_start_checkpoint=self.warm_start_checkpoint,
            expected_warm_start_sha256=self.expected_warm_start_sha256,
            environment=self.environment,
        )
        live = build_volgan_live_condition(
            panel,
            as_of_date=instruction.fit_as_of_date,
            grid=self.model_config.grid,
        )
        return VolganWeeklyMarketFit(
            market_fit_identity=instruction.market_fit_identity,
            fit=public_fit,
            live=live,
            checkpoint_sha256=file_sha256(public_fit.result.checkpoint_path),
        )

    def fit_policy(
        self,
        market_fit: VolganWeeklyMarketFit,
        *,
        instruction: WeeklyFitInstruction,
        claim: WeeklyClaimDefinition,
        contracts: tuple[OptionContract, ...],
        state: WeeklyDecisionState,
    ) -> VolganWeeklyPolicy:
        """Generate remaining paths and fit long, short, and zero schedules."""
        _validate_policy_request(
            market_fit.market_fit_identity,
            instruction=instruction,
            contracts=contracts,
        )
        geometry = self._weekly_scenario_cache.get_or_build(
            state=state,
            model_fit_identity=market_fit.market_fit_identity,
            artifact_fingerprints=market_fit.artifact_fingerprints,
            live_state_fingerprint=market_fit.live_state_fingerprint,
            policy_config_identity=self.policy_config_identity,
            scenario_count=self.policy_config.scenario_count,
            seed=self.policy_config.seed,
            measure=self.measure,
            path_builder=lambda: sample_volgan_weekly_paths(
                market_fit.fit.result,
                live=market_fit.live,
                path_count=self.policy_config.scenario_count,
                step_count=5 - state.decision_index,
                seed=self.policy_config.seed,
            ),
        )
        if self._weekly_cache_generation != self._weekly_scenario_cache.generation:
            self._weekly_pricing_contexts.clear()
            self._weekly_cache_generation = self._weekly_scenario_cache.generation
        scenarios = attach_generated_weekly_claim(geometry, claim=claim)
        pricing_context = self._weekly_pricing_contexts.get(id(geometry))
        if pricing_context is None:
            pricing_context = WeeklyOpenLoopPricingContext(
                scenarios,
                config=self.policy_config.hedge,
            )
            self._weekly_pricing_contexts[id(geometry)] = pricing_context
        hedge = fit_weekly_open_loop_hedge(
            scenarios,
            claim_quantity=instruction.claim_quantity,
            initial_spot_position=state.previous_spot_position,
            initial_option_positions=state.previous_option_positions,
            config=self.policy_config.hedge,
            pricing_context=pricing_context,
        )
        buyer = hedge.prices.buyer / instruction.claim_quantity
        seller = hedge.prices.seller / instruction.claim_quantity
        return VolganWeeklyPolicy(
            market_fit_identity=market_fit.market_fit_identity,
            policy_fit_identity=instruction.policy_fit_identity,
            prices=WeeklyPriceRecord(
                common_market_price_per_fit_spot=(
                    claim.market_price if state.decision_index == 0 else None
                ),
                q_expectation_per_fit_spot=None,
                buyer_indifference_per_fit_spot=float(buyer),
                seller_indifference_per_fit_spot=float(seller),
                continuation_price_per_fit_spot=float(0.5 * (buyer + seller)),
            ),
            schedule=instruction.schedule,
            fit_decision_index=state.decision_index,
            scenarios=scenarios,
            hedge=hedge,
        )

    def release_weekly_origin(self) -> None:
        """Release target-independent arrays after one origin is serialized."""
        self._weekly_scenario_cache.clear()
        self._weekly_pricing_contexts.clear()
        self._weekly_cache_generation = self._weekly_scenario_cache.generation

    def action(
        self,
        fit: VolganWeeklyPolicy,
        *,
        state: WeeklyDecisionState,
    ) -> WeeklyPolicyAction:
        """Return the frozen schedule row or restarted first row."""
        row = _policy_row(
            fit.schedule,
            fit.fit_decision_index,
            state.decision_index,
        )
        options = fit.hedge.option_positions[row].copy()
        options[~state.option_alive] = 0.0
        return WeeklyPolicyAction(
            spot_position=float(fit.hedge.spot_positions[row]),
            option_positions=np.asarray(options, dtype=np.float64),
        )


def _validate_policy_request(
    market_fit_identity: str,
    *,
    instruction: WeeklyFitInstruction,
    contracts: tuple[OptionContract, ...],
) -> None:
    if market_fit_identity != instruction.market_fit_identity:
        raise ValueError("VolGAN weekly market fit does not match instruction")
    if tuple(contract.contract_id for contract in contracts) != (
        instruction.frozen_contract_ids
    ):
        raise ValueError("VolGAN weekly contracts do not match instruction")


def _policy_row(
    schedule: HedgeSchedule,
    fit_decision_index: int,
    decision_index: int,
) -> int:
    if schedule is HedgeSchedule.FROZEN_WEEKLY:
        return decision_index
    if schedule is HedgeSchedule.RESTARTED_WEEKLY:
        if fit_decision_index != decision_index:
            raise ValueError("restarted VolGAN weekly policy is stale")
        return 0
    raise ValueError("VolGAN weekly policy requires a weekly schedule")
