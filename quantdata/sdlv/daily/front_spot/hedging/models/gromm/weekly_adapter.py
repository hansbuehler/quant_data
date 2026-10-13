"""Weekly benchmark adapter for dated GROMM2 P/Q deep hedges."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from cdxcore.uniquehash import named_unique_filename48_8
from numpy.typing import NDArray

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
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    Gromm2MarketState,
    load_gromm2_market_state,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_weekly import (
    Gromm2WeeklyScenarios,
    generate_gromm2_weekly_scenarios,
    gromm2_weekly_training_data,
    live_gromm2_weekly_feature,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.hedger import (
    GrommHedgeFit,
    fit_gromm_claim_minus_zero,
    gromm_pure_hedge_actions,
    gromm_signed_policy_actions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    GrommPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Artifacts,
    RollingGromm2Config,
    load_rolling_gromm2,
)
from quantdata.torch import BaseEnvironment

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class GrommWeeklyMarketFit:
    """One authenticated dated GROMM2 fit used across weekly targets."""

    market_fit_identity: str
    artifacts: RollingGromm2Artifacts
    market_state: Gromm2MarketState
    affine_history_sha256: str
    dynamics_sha256: str

    @property
    def artifact_fingerprints(self) -> tuple[str, ...]:
        """Return exact rolling-market artifact fingerprints."""
        return (
            f"affine-history:{self.affine_history_sha256}",
            f"dynamics:{self.dynamics_sha256}",
        )


@dataclass(kw_only=True)
class GrommWeeklyPolicy:
    """One signed-policy family and its realized frozen-policy feature replay."""

    market_fit_identity: str
    policy_fit_identity: str
    prices: WeeklyPriceRecord
    schedule: HedgeSchedule
    fit_decision_index: int
    claim: WeeklyClaimDefinition
    market_state: Gromm2MarketState
    measure: GrommMeasure
    generated: Gromm2WeeklyScenarios
    hedge: GrommHedgeFit
    active_mask: NDArray[np.bool_]
    initial_positions: FloatArray
    realized_features: dict[int, FloatArray]
    artifact_fingerprints: tuple[str, ...]

    @property
    def cap_hits(self) -> int:
        """Return cap hits across all four fitted policy surfaces."""
        scenarios = self.generated.scenarios
        action_count = (
            scenarios.scenario_count
            * scenarios.step_count
            * (scenarios.contract_count + 1)
        )
        return sum(
            int(round(result.cap_hit_fraction * action_count))
            for result in (
                self.hedge.claim,
                self.hedge.short,
                self.hedge.zero,
                self.hedge.pure,
            )
        )

    @property
    def converged(self) -> bool:
        """Return whether every policy converged without fitted cap hits."""
        return (
            self.hedge.claim.converged
            and self.hedge.short.converged
            and self.hedge.zero.converged
            and self.hedge.pure.converged
            and self.cap_hits == 0
        )


class GrommWeeklyAdapter:
    """Load rolling GROMM2 markets and fit recurrent weekly P/Q policies."""

    model = "gromm"

    def __init__(
        self,
        *,
        artifact_root: Path,
        checkpoint_directory: Path,
        config: GrommConfig,
        rolling_config: RollingGromm2Config = RollingGromm2Config(),
        train_config: GrommTrainConfig = GrommTrainConfig(),
        policy_template: GrommPolicyConfig | None = None,
        environment: BaseEnvironment | None = None,
    ) -> None:
        if config.dlv_substeps != rolling_config.substeps:
            raise ValueError("GROMM weekly simulation and market substeps differ")
        self.artifact_root = Path(artifact_root)
        self.checkpoint_directory = Path(checkpoint_directory)
        self.config = config
        self.rolling_config = rolling_config
        self.train_config = train_config
        self.policy_template = policy_template
        self.environment = environment
        self._component_positions: dict[
            tuple[object, str, str],
            tuple[FloatArray, FloatArray, FloatArray, FloatArray],
        ] = {}
        self.measure = config.measure.value
        self.model_config_identity = (
            "gromm2-weekly-market-"
            + named_unique_filename48_8(
                config.label,
                config.measure.value,
                rolling_config,
            )
        )
        self.policy_config_identity = (
            "gromm2-weekly-policy-"
            + named_unique_filename48_8(
                config.label,
                config.objective.value,
                config.risk_parameter,
                config.position_cap,
                train_config,
                policy_template,
            )
        )

    def fit_market(
        self,
        panel: AsOfWeeklyMarketPanel,
        *,
        instruction: WeeklyFitInstruction,
    ) -> GrommWeeklyMarketFit:
        """Load one complete fit and verify its transition origins exactly."""
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
        market_state = load_gromm2_market_state(artifacts)
        return GrommWeeklyMarketFit(
            market_fit_identity=instruction.market_fit_identity,
            artifacts=artifacts,
            market_state=market_state,
            affine_history_sha256=file_sha256(artifacts.paths.affine_history),
            dynamics_sha256=file_sha256(artifacts.paths.dynamics),
        )

    def fit_policy(
        self,
        market_fit: GrommWeeklyMarketFit,
        *,
        instruction: WeeklyFitInstruction,
        claim: WeeklyClaimDefinition,
        contracts: tuple[OptionContract, ...],
        state: WeeklyDecisionState,
    ) -> GrommWeeklyPolicy:
        """Generate the remaining path law and fit long, short, and zero policies."""
        _validate_request(
            market_fit.market_fit_identity,
            instruction=instruction,
            contracts=contracts,
        )
        generated = generate_gromm2_weekly_scenarios(
            market_state=market_fit.market_state,
            state=state,
            claim=claim,
            config=self.config,
            model_fit_identity=market_fit.market_fit_identity,
        )
        data = gromm2_weekly_training_data(
            generated,
            claim_quantity=instruction.claim_quantity,
        )
        pure_data = replace(
            data,
            instrument_returns=generated.pure_instrument_returns,
        )
        policy_config = self._policy_config(
            instruction=instruction,
            feature_count=data.feature_count,
            instrument_count=data.instrument_count,
        )
        component_key = self._component_key(instruction)
        if state.decision_index == 0:
            empty = np.zeros(data.instrument_count, dtype=np.float64)
            component_positions = (
                empty.copy(),
                empty.copy(),
                empty.copy(),
                empty.copy(),
            )
        else:
            try:
                component_positions = self._component_positions[component_key]
            except KeyError as error:
                raise ValueError(
                    "restarted GROMM policy lacks the prior signed component books"
                ) from error
        hedge = fit_gromm_claim_minus_zero(
            data,
            pure_data=pure_data,
            policy_config=policy_config,
            objective=self.config.objective,
            risk_parameter=self.config.risk_parameter,
            checkpoint_directory=self.checkpoint_directory,
            train_config=self.train_config,
            environment=self.environment,
            claim_initial_positions=component_positions[0],
            short_initial_positions=component_positions[1],
            zero_initial_positions=component_positions[2],
            pure_initial_positions=component_positions[3],
        )
        fit_scale = state.discount_to_origin * state.spot_over_episode_origin
        quantity_scale = instruction.claim_quantity * fit_scale
        buyer = hedge.buyer_indifference_price / quantity_scale
        seller = hedge.seller_indifference_price / quantity_scale
        q_expectation = (
            float(
                np.mean(generated.scenarios.terminal_claim_values_pv) / fit_scale
            )
            if self.config.measure is GrommMeasure.Q
            else None
        )
        return GrommWeeklyPolicy(
            market_fit_identity=market_fit.market_fit_identity,
            policy_fit_identity=instruction.policy_fit_identity,
            prices=WeeklyPriceRecord(
                common_market_price_per_fit_spot=(
                    claim.market_price if state.decision_index == 0 else None
                ),
                q_expectation_per_fit_spot=q_expectation,
                buyer_indifference_per_fit_spot=float(buyer),
                seller_indifference_per_fit_spot=float(seller),
                continuation_price_per_fit_spot=float(0.5 * (buyer + seller)),
            ),
            schedule=instruction.schedule,
            fit_decision_index=state.decision_index,
            claim=claim,
            market_state=market_fit.market_state,
            measure=self.config.measure,
            generated=generated,
            hedge=hedge,
            artifact_fingerprints=tuple(
                f"{name}:{file_sha256(result.checkpoint_path)}"
                for name, result in (
                    ("claim", hedge.claim),
                    ("short", hedge.short),
                    ("zero", hedge.zero),
                    ("pure", hedge.pure),
                )
            ),
            active_mask=data.active_mask,
            initial_positions=data.initial_positions.copy(),
            realized_features={},
        )

    def action(
        self,
        fit: GrommWeeklyPolicy,
        *,
        state: WeeklyDecisionState,
    ) -> WeeklyPolicyAction:
        """Replay a frozen LSTM or evaluate the first action of a restarted fit."""
        if fit.policy_fit_identity != state.instruction.policy_fit_identity:
            raise ValueError("GROMM weekly action received the wrong policy fit")
        feature = live_gromm2_weekly_feature(
            market_state=fit.market_state,
            state=state,
            claim=fit.claim,
            measure=fit.measure,
            initial_positions=fit.initial_positions,
        )
        if feature.shape != (fit.hedge.pure.policy.config.feature_count,):
            raise ValueError("GROMM weekly live feature geometry changed")
        fit.realized_features[state.decision_index] = feature
        if fit.schedule is HedgeSchedule.FROZEN_WEEKLY:
            expected = tuple(range(state.decision_index + 1))
            if tuple(sorted(fit.realized_features)) != expected:
                raise ValueError("frozen GROMM weekly feature replay is incomplete")
            history = np.stack(
                [fit.realized_features[index] for index in expected], axis=0
            )[None]
            mask = fit.active_mask[: state.decision_index + 1]
            row = state.decision_index
        elif fit.schedule is HedgeSchedule.RESTARTED_WEEKLY:
            if fit.fit_decision_index != state.decision_index:
                raise ValueError("restarted GROMM weekly policy is stale")
            history = feature[None, None]
            mask = fit.active_mask[:1]
            row = 0
        else:
            raise ValueError("GROMM weekly action requires a weekly schedule")
        claim_actions, short_actions, zero_actions = gromm_signed_policy_actions(
            fit.hedge,
            features=history,
            active_mask=mask,
        )
        signed_actions = tuple(
            values[0, row].copy()
            for values in (claim_actions, short_actions, zero_actions)
        )
        pure_actions, _ = gromm_pure_hedge_actions(
            fit.hedge,
            features=history,
            active_mask=mask,
        )
        pure_action = pure_actions[0, row].copy()
        settling = np.asarray(
            [
                contract.expiry_date == state.next_session_date
                for contract in state.contracts
            ],
            dtype=bool,
        )
        carried_components = []
        for values in signed_actions:
            carried = values.copy()
            carried[1:][settling] = 0.0
            carried_components.append(carried)
        pure_carried = pure_action.copy()
        pure_carried[1:][settling] = 0.0
        carried_components.append(pure_carried)
        self._component_positions[
            self._component_key(state.instruction)
        ] = tuple(carried_components)
        action = np.clip(
            pure_action,
            -self.config.position_cap,
            self.config.position_cap,
        )
        options = action[1:].copy()
        options[~state.option_alive] = 0.0
        return WeeklyPolicyAction(
            spot_position=float(action[0]),
            option_positions=options,
        )

    @staticmethod
    def _component_key(
        instruction: WeeklyFitInstruction,
    ) -> tuple[object, str, str]:
        return (
            instruction.episode_origin_date,
            instruction.target_label,
            instruction.measure,
        )

    def _policy_config(
        self,
        *,
        instruction: WeeklyFitInstruction,
        feature_count: int,
        instrument_count: int,
    ) -> GrommPolicyConfig:
        label = (
            f"{self.config.label}-{instruction.policy_fit_identity[:16]}-"
            f"{self.config.measure.value}"
        )
        if self.policy_template is None:
            return GrommPolicyConfig(
                label=label,
                feature_count=feature_count,
                instrument_count=instrument_count,
                position_cap=self.config.position_cap,
            )
        return replace(
            self.policy_template,
            label=label,
            feature_count=feature_count,
            instrument_count=instrument_count,
            position_cap=self.config.position_cap,
        )


def _validate_request(
    market_fit_identity: str,
    *,
    instruction: WeeklyFitInstruction,
    contracts: tuple[OptionContract, ...],
) -> None:
    if market_fit_identity != instruction.market_fit_identity:
        raise ValueError("GROMM weekly market fit does not match instruction")
    if tuple(contract.contract_id for contract in contracts) != (
        instruction.frozen_contract_ids
    ):
        raise ValueError("GROMM weekly contracts do not match instruction")
