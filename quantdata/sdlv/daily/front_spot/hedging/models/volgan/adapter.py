"""Public-panel fitting adapter for VolGAN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalSample,
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
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
    remove_completed_checkpoint,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedHedgeConfig,
    build_generated_daily_scenarios,
    fit_generated_cost_lasso,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    VolganTrainingData,
    build_volgan_live_condition,
    build_volgan_training_data,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.hedger import (
    sample_volgan_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.training import (
    VolganTrainingResult,
    fit_volgan,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class VolganPublicFit:
    """Training data and fitted model tied to one public as-of date."""

    as_of_date: date
    data: VolganTrainingData
    result: VolganTrainingResult


def fit_public_volgan(
    panel: PublicHedgingPanel,
    *,
    as_of_date: date,
    model_config: VolganModelConfig,
    checkpoint_directory: Path,
    observation_count: int = 252,
    train_config: VolganTrainConfig = VolganTrainConfig(),
    warm_start_checkpoint: Path | None = None,
    expected_warm_start_sha256: str | None = None,
    environment: BaseEnvironment | None = None,
) -> VolganPublicFit:
    """Fit VolGAN using only released states ending at ``as_of_date``."""
    require_public_hedging_panel(panel)
    data = build_volgan_training_data(
        panel,
        end_date=as_of_date,
        observation_count=observation_count,
        grid=model_config.grid,
    )
    result = fit_volgan(
        data,
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_warm_start_sha256=expected_warm_start_sha256,
        environment=environment,
    )
    return VolganPublicFit(as_of_date=as_of_date, data=data, result=result)


def run_volgan_one_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    target: DailyDigitalTarget | None = None,
    model_config: VolganModelConfig,
    checkpoint_directory: Path,
    observation_count: int = 252,
    train_config: VolganTrainConfig = VolganTrainConfig(),
    hedge_config: GeneratedHedgeConfig = GeneratedHedgeConfig(),
    warm_start_checkpoint: Path | None = None,
    expected_warm_start_sha256: str | None = None,
    retain_fit_checkpoint: bool = True,
    environment: BaseEnvironment | None = None,
    claim_quantity: float = 1.0,
) -> HedgeSmokeResult:
    """Fit, generate, hedge, and score one daily digital target."""
    if target is None:
        target = DailyDigitalTarget(
            maturity_business_days=1,
            right=OptionRight.CALL,
            nms=0.0,
        )
    return _run_volgan_targets(
        panel,
        origin_date=origin_date,
        targets=(target,),
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=train_config,
        hedge_config=hedge_config,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_warm_start_sha256=expected_warm_start_sha256,
        retain_fit_checkpoint=retain_fit_checkpoint,
        environment=environment,
        claim_quantities={target.label: claim_quantity},
    )[0]


def run_volgan_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    model_config: VolganModelConfig,
    checkpoint_directory: Path,
    observation_count: int = 252,
    train_config: VolganTrainConfig = VolganTrainConfig(),
    hedge_config: GeneratedHedgeConfig = GeneratedHedgeConfig(),
    warm_start_checkpoint: Path | None = None,
    expected_warm_start_sha256: str | None = None,
    retain_fit_checkpoint: bool = True,
    environment: BaseEnvironment | None = None,
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Fit and generate once, then hedge all 20 daily digital targets."""
    return _run_volgan_targets(
        panel,
        origin_date=origin_date,
        targets=daily_digital_targets(),
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=train_config,
        hedge_config=hedge_config,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_warm_start_sha256=expected_warm_start_sha256,
        retain_fit_checkpoint=retain_fit_checkpoint,
        environment=environment,
        claim_quantities=claim_quantities,
    )


def _run_volgan_targets(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    targets: tuple[DailyDigitalTarget, ...],
    model_config: VolganModelConfig,
    checkpoint_directory: Path,
    observation_count: int,
    train_config: VolganTrainConfig,
    hedge_config: GeneratedHedgeConfig,
    warm_start_checkpoint: Path | None,
    expected_warm_start_sha256: str | None,
    retain_fit_checkpoint: bool,
    environment: BaseEnvironment | None,
    claim_quantities: Mapping[str, float] | None,
) -> tuple[HedgeSmokeResult, ...]:
    require_public_hedging_panel(panel)
    samples = tuple(
        build_daily_digital_sample(
            panel,
            origin_date=origin_date,
            target=target,
        )
        for target in targets
    )
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    if any(
        sample.hedge_quotes != samples[0].hedge_quotes for sample in samples[1:]
    ):
        raise RuntimeError("VolGAN daily targets do not share a hedge universe")
    public_fit = fit_public_volgan(
        panel,
        as_of_date=origin_date,
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=train_config,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_warm_start_sha256=expected_warm_start_sha256,
        environment=environment,
    )
    live = build_volgan_live_condition(
        panel,
        as_of_date=origin_date,
        grid=model_config.grid,
    )
    if live.forecast_date != samples[0].transition.next_date:
        raise RuntimeError("VolGAN forecast date does not match hedge endpoint")
    generated = sample_volgan_transitions(
        public_fit.result,
        condition=live.values,
        path_count=hedge_config.scenario_count,
        seed=hedge_config.seed,
    )
    fit_checkpoint_sha256 = file_sha256(public_fit.result.checkpoint_path)
    results = tuple(
        _evaluate_volgan_target(
            sample,
            public_fit=public_fit,
            generated_forward_log_returns=generated.forward_log_returns,
            generated_log_implied_volatilities=(
                generated.next_log_implied_volatilities
            ),
            model_config=model_config,
            hedge_config=hedge_config,
            target_count=len(samples),
            claim_quantity=float(quantities[index]),
            fit_checkpoint_sha256=fit_checkpoint_sha256,
        )
        for index, sample in enumerate(samples)
    )
    if not retain_fit_checkpoint:
        remove_completed_checkpoint(public_fit.result.checkpoint_path)
    return results


def _evaluate_volgan_target(
    sample: DailyDigitalSample,
    *,
    public_fit: VolganPublicFit,
    generated_forward_log_returns: np.ndarray,
    generated_log_implied_volatilities: np.ndarray,
    model_config: VolganModelConfig,
    hedge_config: GeneratedHedgeConfig,
    target_count: int,
    claim_quantity: float,
    fit_checkpoint_sha256: str,
) -> HedgeSmokeResult:
    scenarios = build_generated_daily_scenarios(
        sample,
        forward_log_returns=generated_forward_log_returns,
        next_log_implied_volatilities=generated_log_implied_volatilities,
        grid=model_config.grid,
    )
    hedge = fit_generated_cost_lasso(
        scenarios,
        sample,
        cost_multiplier=hedge_config.cost_multiplier,
        position_cap=hedge_config.position_cap,
        claim_quantity=claim_quantity,
    )
    return evaluate_daily_digital_hedge(
        sample,
        model="volgan",
        measure="historical-P",
        spot_action=float(hedge.action[0]),
        option_actions=hedge.action[1:],
        model_price=sample.market_price,
        objective_value=hedge.uncapped.objective,
        converged=(public_fit.result.converged and hedge.converged),
        cap_hits=hedge.cap_hits,
        claim_quantity=claim_quantity,
        diagnostics={
            "training_examples": len(public_fit.data.condition_dates),
            "target_count": target_count,
            "claim_quantity": claim_quantity,
            "completed_epochs": public_fit.result.completed_epochs,
            "validation_loss": public_fit.result.validation_loss,
            "warm_start_sha256": public_fit.result.warm_start_sha256,
            "fit_checkpoint_sha256": fit_checkpoint_sha256,
            "generated_paths": hedge_config.scenario_count,
            "generated_mean_forward_log_return": float(
                np.mean(generated_forward_log_returns)
            ),
            "generated_mean_claim_value": float(
                np.mean(scenarios.claim_terminal_values)
            ),
            "digital_probability_clip_count": (
                scenarios.digital_probability_clip_count
            ),
            "hedge_instrument_count": scenarios.instrument_returns.shape[1],
            "cost_multiplier": hedge_config.cost_multiplier,
            "covariance_shrinkage": hedge.uncapped.covariance_shrinkage,
            "covariance_iterations": hedge.uncapped.covariance_iterations,
            "policy_semantics": (
                "spread-weighted minimum-variance cost-lasso on generated "
                "fixed-contract returns"
            ),
            "price_semantics": (
                "common released SANOS market mark; VolGAN supplies P scenarios, "
                "not a risk-neutral price"
            ),
            "payoff_units": "discounted origin-PV per unit origin spot",
        },
    )
