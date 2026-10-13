"""Common daily-benchmark adapter for GROMM policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
    daily_claim_quantity_vector,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    generate_gromm2_daily_scenarios,
    gromm2_daily_training_data,
    load_gromm2_market_state,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.hedger import (
    fit_gromm_claim_minus_zero,
    gromm_pure_hedge_actions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    GrommPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Config,
    load_rolling_gromm2,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.simulation import (
    generate_gromm_daily_scenarios,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.training import (
    daily_gromm_training_data,
)
from quantdata.torch import BaseEnvironment


def run_gromm_one_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    checkpoint_directory: Path,
    config: GrommConfig,
    train_config: GrommTrainConfig = GrommTrainConfig(),
    policy_config: GrommPolicyConfig | None = None,
    environment: BaseEnvironment | None = None,
) -> HedgeSmokeResult:
    """Train and score one ATM one-day digital using only public-panel state."""
    require_public_hedging_panel(panel)
    target = DailyDigitalTarget(
        maturity_business_days=1,
        right=OptionRight.CALL,
        nms=0.0,
    )
    scenarios = generate_gromm_daily_scenarios(
        panel,
        origin_date=origin_date,
        target=target,
        config=config,
    )
    data = daily_gromm_training_data(scenarios)
    if config.measure is GrommMeasure.Q:
        pure_data = data
    else:
        q_scenarios = generate_gromm_daily_scenarios(
            panel,
            origin_date=origin_date,
            target=target,
            config=replace(config, measure=GrommMeasure.Q),
        )
        pure_data = replace(
            data,
            instrument_returns=daily_gromm_training_data(
                q_scenarios
            ).instrument_returns,
        )
    if policy_config is None:
        policy_config = GrommPolicyConfig(
            label=(
                f"{config.label}-{origin_date.isoformat()}-"
                f"{config.measure.value}-{target.label}"
            ),
            feature_count=data.feature_count,
            instrument_count=data.instrument_count,
            position_cap=config.position_cap,
        )
    fit = fit_gromm_claim_minus_zero(
        data,
        pure_data=pure_data,
        policy_config=policy_config,
        objective=config.objective,
        risk_parameter=config.risk_parameter,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        environment=environment,
    )
    pure_actions, final_cap_hits = gromm_pure_hedge_actions(
        fit,
        features=data.features[:1],
        active_mask=data.active_mask,
    )
    action = pure_actions[0, 0]
    cap_hits = (
        final_cap_hits
        + int(round(fit.claim.cap_hit_fraction * data.path_count * data.instrument_count))
        + int(round(fit.short.cap_hit_fraction * data.path_count * data.instrument_count))
        + int(round(fit.zero.cap_hit_fraction * data.path_count * data.instrument_count))
        + int(round(fit.pure.cap_hit_fraction * data.path_count * data.instrument_count))
    )
    return evaluate_daily_digital_hedge(
        scenarios.sample,
        model="gromm",
        measure=config.measure.value,
        spot_action=float(action[0]),
        option_actions=action[1:],
        model_price=scenarios.claim_model_price,
        objective_value=(fit.claim.validation_loss - fit.zero.validation_loss),
        converged=(
            fit.claim.converged
            and fit.short.converged
            and fit.zero.converged
            and fit.pure.converged
            and cap_hits == 0
        ),
        cap_hits=cap_hits,
        diagnostics={
            "scenario_paths": scenarios.path_count,
            "dlv_substeps": config.dlv_substeps,
            "factor_rank": scenarios.calibration.factors.factor_rank,
            "objective": config.objective.value,
            "risk_parameter": config.risk_parameter,
            "claim_best_epoch": fit.claim.best_epoch,
            "short_best_epoch": fit.short.best_epoch,
            "zero_best_epoch": fit.zero.best_epoch,
            "pure_best_epoch": fit.pure.best_epoch,
            "claim_validation_loss": fit.claim.validation_loss,
            "short_validation_loss": fit.short.validation_loss,
            "zero_validation_loss": fit.zero.validation_loss,
            "buyer_indifference_price": (
                scenarios.claim_model_price + fit.buyer_indifference_price
            ),
            "seller_indifference_price": (
                scenarios.claim_model_price + fit.seller_indifference_price
            ),
            "claim_validation_standard_error": (
                fit.claim.validation_standard_error
            ),
            "zero_validation_standard_error": fit.zero.validation_standard_error,
            "policy_semantics": (
                "direct executable pure hedge with exact net turnover; "
                "signed claim/short/zero policies price indifference claims"
            ),
            "price_semantics": "GROMM Q expectation from the current public DLV state",
            "market_mark_semantics": "stored SANOS reference interpolation",
            "training_data_semantics": "generated from public DLV/SANOS state only",
            "payoff_units": "discounted origin-PV per unit origin spot",
        },
    )


def run_gromm2_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    artifact_root: Path,
    checkpoint_directory: Path,
    config: GrommConfig,
    rolling_config: RollingGromm2Config = RollingGromm2Config(),
    train_config: GrommTrainConfig = GrommTrainConfig(),
    policy_config: GrommPolicyConfig | None = None,
    environment: BaseEnvironment | None = None,
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Load one dated GROMM2 state and hedge the complete daily target book."""
    require_public_hedging_panel(panel)
    targets = daily_digital_targets()
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    artifacts = load_rolling_gromm2(
        artifact_root,
        as_of_date=origin_date,
        config=rolling_config,
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
    )
    market_state = load_gromm2_market_state(artifacts)
    results: list[HedgeSmokeResult] = []
    for target_index, (target, claim_quantity) in enumerate(
        zip(targets, quantities, strict=True)
    ):
        scenario_config = replace(config, seed=config.seed + target_index)
        scenarios = generate_gromm2_daily_scenarios(
            panel,
            origin_date=origin_date,
            target=target,
            market_state=market_state,
            config=scenario_config,
        )
        data = gromm2_daily_training_data(scenarios)
        data = replace(data, claim_returns=claim_quantity * data.claim_returns)
        pure_data = replace(
            data,
            instrument_returns=scenarios.pure_instrument_returns[:, None],
        )
        current_policy = (
            GrommPolicyConfig(
                label=(
                    f"{config.label}-{origin_date.isoformat()}-"
                    f"{config.measure.value}-{target.label}"
                ),
                feature_count=data.feature_count,
                instrument_count=data.instrument_count,
                position_cap=config.position_cap,
            )
            if policy_config is None
            else replace(
                policy_config,
                label=(
                    f"{policy_config.label}-{origin_date.isoformat()}-"
                    f"{config.measure.value}-{target.label}"
                ),
                feature_count=data.feature_count,
                instrument_count=data.instrument_count,
                position_cap=config.position_cap,
            )
        )
        fit = fit_gromm_claim_minus_zero(
            data,
            pure_data=pure_data,
            policy_config=current_policy,
            objective=config.objective,
            risk_parameter=config.risk_parameter,
            checkpoint_directory=checkpoint_directory,
            train_config=train_config,
            environment=environment,
        )
        pure_actions, final_cap_hits = gromm_pure_hedge_actions(
            fit,
            features=data.features[:1],
            active_mask=data.active_mask,
        )
        action = pure_actions[0, 0]
        cap_hits = (
            final_cap_hits
            + int(
                round(
                    fit.claim.cap_hit_fraction
                    * data.path_count
                    * data.instrument_count
                )
            )
            + int(
                round(
                    fit.short.cap_hit_fraction
                    * data.path_count
                    * data.instrument_count
                )
            )
            + int(
                round(
                    fit.zero.cap_hit_fraction
                    * data.path_count
                    * data.instrument_count
                )
            )
            + int(
                round(
                    fit.pure.cap_hit_fraction
                    * data.path_count
                    * data.instrument_count
                )
            )
        )
        results.append(
            evaluate_daily_digital_hedge(
                scenarios.sample,
                model="gromm",
                measure=config.measure.value,
                spot_action=float(action[0]),
                option_actions=action[1:],
                model_price=scenarios.claim_model_price,
                objective_value=(
                    fit.claim.validation_loss - fit.zero.validation_loss
                ),
                converged=(
                    fit.claim.converged
                    and fit.short.converged
                    and fit.zero.converged
                    and fit.pure.converged
                    and cap_hits == 0
                ),
                cap_hits=cap_hits,
                claim_quantity=float(claim_quantity),
                diagnostics={
                    "target_count": len(targets),
                    "claim_quantity": float(claim_quantity),
                    "scenario_paths": scenarios.path_count,
                    "dlv_substeps": config.dlv_substeps,
                    "factor_rank": market_state.dynamics.factor_count,
                    "market_fit_inside_spread_percent": (
                        market_state.percent_inside_spread
                    ),
                    "explained_variance_ratio": (
                        market_state.explained_variance_ratio
                    ),
                    "physical_projection_fraction": (
                        market_state.physical_projection_fraction
                    ),
                    "candidate_physical_projection_fraction": (
                        market_state.candidate_physical_projection_fraction
                    ),
                    "physical_fallback_used": (
                        market_state.physical_fallback_used
                    ),
                    "physical_selected_model": (
                        market_state.physical_selected_model
                    ),
                    "physical_daily_risk_price_norm": float(
                        np.linalg.norm(
                            market_state.physical_market_price_of_risk
                        )
                        / np.sqrt(252.0)
                    ),
                    "drift_taming_fraction": scenarios.drift_taming_fraction,
                    "objective": config.objective.value,
                    "risk_parameter": config.risk_parameter,
                    "claim_best_epoch": fit.claim.best_epoch,
                    "short_best_epoch": fit.short.best_epoch,
                    "zero_best_epoch": fit.zero.best_epoch,
                    "pure_best_epoch": fit.pure.best_epoch,
                    "claim_validation_loss": fit.claim.validation_loss,
                    "short_validation_loss": fit.short.validation_loss,
                    "zero_validation_loss": fit.zero.validation_loss,
                    "buyer_indifference_price": (
                        scenarios.sample.market_price
                        + fit.buyer_indifference_price / claim_quantity
                    ),
                    "seller_indifference_price": (
                        scenarios.sample.market_price
                        + fit.seller_indifference_price / claim_quantity
                    ),
                    "policy_semantics": (
                        "direct executable pure hedge with exact net turnover; "
                        "signed claim/short/zero policies price indifference claims"
                    ),
                    "scenario_semantics": (
                        "dated rolling 200-strike GROMM2 with AR(1)-compatible "
                        f"{config.measure.value} dynamics"
                    ),
                    "price_semantics": (
                        "GROMM2 risk-neutral expectation from the current dated state"
                    ),
                    "market_mark_semantics": (
                        "stored SANOS reference interpolation"
                    ),
                    "payoff_units": "discounted origin-PV per unit origin spot",
                },
            )
        )
    return tuple(results)
