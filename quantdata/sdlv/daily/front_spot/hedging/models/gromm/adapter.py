"""Common daily-benchmark adapter for GROMM policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.mean_reduction import adaptive_scenario_count
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
    build_daily_digital_sample,
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
    price_gromm2_daily_target,
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
    reduce_gromm_one_day_training_data,
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
    data = daily_gromm_training_data(
        scenarios,
        mean_reduction=config.mean_reduction,
        mean_reduction_config=config.mean_reduction_config,
    )
    if config.measure is GrommMeasure.Q:
        pure_data = data
    else:
        q_scenarios = generate_gromm_daily_scenarios(
            panel,
            origin_date=origin_date,
            target=target,
            config=replace(config, measure=GrommMeasure.Q),
        )
        pure_data = reduce_gromm_one_day_training_data(
            replace(
                data,
                instrument_returns=q_scenarios.instrument_returns[:, None],
                scenario_weights=None,
                mean_reduction=None,
                mean_reduction_diagnostics=None,
            ),
            mean_reduction=config.mean_reduction,
            mean_reduction_config=config.mean_reduction_config,
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
            "mean_reduction": config.mean_reduction.value,
            "raw_max_standardized_instrument_mean": (
                data.mean_reduction_diagnostics.raw_max_standardized_mean
            ),
            "reduced_max_standardized_instrument_mean": (
                data.mean_reduction_diagnostics.reduced_max_standardized_mean
            ),
            "mean_reduction_effective_sample_size": (
                data.mean_reduction_diagnostics.effective_sample_size
            ),
            "raw_q_max_standardized_instrument_mean": (
                pure_data.mean_reduction_diagnostics.raw_max_standardized_mean
            ),
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
    target_maturity: int | None = None,
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
    targets = daily_digital_targets(maturity_business_days=target_maturity)
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    artifacts = load_rolling_gromm2(
        artifact_root,
        as_of_date=origin_date,
        config=rolling_config,
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
    )
    market_state = load_gromm2_market_state(artifacts)
    representative_sample = build_daily_digital_sample(
        panel,
        origin_date=origin_date,
        target=targets[0],
    )
    path_count = adaptive_scenario_count(
        1 + len(representative_sample.hedge_quotes),
        minimum=config.path_count,
        multiple=config.batch_size,
    )
    scenario_config = replace(config, path_count=path_count)
    shared_scenarios = None
    terminal_spot = None
    if all(target.maturity_business_days == 1 for target in targets):
        shared_scenarios = generate_gromm2_daily_scenarios(
            panel,
            origin_date=origin_date,
            target=targets[0],
            market_state=market_state,
            config=scenario_config,
        )
        bridge = (
            shared_scenarios.sample.transition.discount_factor
            * shared_scenarios.sample.transition.forward_over_spot
        )
        terminal_spot = 1.0 + shared_scenarios.instrument_returns[:, 0] / bridge
        first_claim = _one_day_terminal_claim(
            terminal_spot,
            strike=shared_scenarios.sample.target_strike_over_forward,
            discount_factor=shared_scenarios.sample.target_discount_factor,
            right=targets[0].right,
        )
        if not np.allclose(
            first_claim,
            shared_scenarios.claim_terminal_values,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("GROMM2 settlement is not determined by terminal spot")
    results: list[HedgeSmokeResult] = []
    for target, claim_quantity in zip(targets, quantities, strict=True):
        if shared_scenarios is None:
            scenarios = generate_gromm2_daily_scenarios(
                panel,
                origin_date=origin_date,
                target=target,
                market_state=market_state,
                config=scenario_config,
            )
        elif target == targets[0]:
            scenarios = shared_scenarios
        else:
            sample = build_daily_digital_sample(
                panel,
                origin_date=origin_date,
                target=target,
            )
            if sample.hedge_quotes != shared_scenarios.sample.hedge_quotes:
                raise RuntimeError("GROMM2 daily targets do not share a hedge book")
            terminal_claim = _one_day_terminal_claim(
                terminal_spot,
                strike=sample.target_strike_over_forward,
                discount_factor=sample.target_discount_factor,
                right=target.right,
            )
            scenarios = replace(
                shared_scenarios,
                target=target,
                sample=sample,
                claim_terminal_values=terminal_claim,
                claim_returns=terminal_claim - sample.market_price,
                claim_model_price=price_gromm2_daily_target(
                    market_state,
                    sample=sample,
                ),
            )
        data = gromm2_daily_training_data(
            scenarios,
            mean_reduction=config.mean_reduction,
            mean_reduction_config=config.mean_reduction_config,
        )
        data = replace(data, claim_returns=claim_quantity * data.claim_returns)
        pure_data = reduce_gromm_one_day_training_data(
            replace(
                data,
                instrument_returns=scenarios.pure_instrument_returns[:, None],
                scenario_weights=None,
                mean_reduction=None,
                mean_reduction_diagnostics=None,
            ),
            mean_reduction=config.mean_reduction,
            mean_reduction_config=config.mean_reduction_config,
        )
        pure_data = replace(pure_data, claim_returns=data.claim_returns)
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
                    "mean_reduction": config.mean_reduction.value,
                    "raw_max_standardized_instrument_mean": (
                        data.mean_reduction_diagnostics.raw_max_standardized_mean
                    ),
                    "reduced_max_standardized_instrument_mean": (
                        data.mean_reduction_diagnostics.reduced_max_standardized_mean
                    ),
                    "mean_reduction_effective_sample_size": (
                        data.mean_reduction_diagnostics.effective_sample_size
                    ),
                    "raw_q_max_standardized_instrument_mean": (
                        pure_data.mean_reduction_diagnostics.raw_max_standardized_mean
                    ),
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


def _one_day_terminal_claim(
    terminal_spot: np.ndarray,
    *,
    strike: float,
    discount_factor: float,
    right: OptionRight,
) -> np.ndarray:
    call = np.asarray(terminal_spot) > strike
    return discount_factor * (call if right is OptionRight.CALL else ~call)
