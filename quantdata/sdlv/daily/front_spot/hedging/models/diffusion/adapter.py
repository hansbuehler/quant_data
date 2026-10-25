"""Public-panel fitting adapter for sequential diffusion baselines."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.mean_reduction import adaptive_scenario_count
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
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    DiffusionTrainingData,
    build_diffusion_live_context,
    build_diffusion_training_data,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    sample_diffusion_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.training import (
    DiffusionTrainingResult,
    fit_diffusion,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedDailyScenarios,
    GeneratedHedgeConfig,
    GeneratedObjectiveFit,
    build_generated_daily_scenarios,
    fit_generated_objective_book,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class DiffusionPublicFit:
    """Training sequences and fitted model tied to one public as-of date."""

    as_of_date: date
    data: DiffusionTrainingData
    result: DiffusionTrainingResult


def fit_public_diffusion(
    panel: PublicHedgingPanel,
    *,
    as_of_date: date,
    model_config: DiffusionModelConfig,
    checkpoint_directory: Path,
    observation_count: int = 252,
    train_config: DiffusionTrainConfig = DiffusionTrainConfig(),
    base_checkpoint: Path | None = None,
    warm_start_checkpoint: Path | None = None,
    expected_base_checkpoint_sha256: str | None = None,
    expected_warm_start_sha256: str | None = None,
    environment: BaseEnvironment | None = None,
) -> DiffusionPublicFit:
    """Fit a diffusion variant using only released states through the as-of date."""
    require_public_hedging_panel(panel)
    data = build_diffusion_training_data(
        panel,
        end_date=as_of_date,
        observation_count=observation_count,
        grid=model_config.grid,
    )
    result = fit_diffusion(
        data,
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        base_checkpoint=base_checkpoint,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_warm_start_sha256=expected_warm_start_sha256,
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
        environment=environment,
    )
    return DiffusionPublicFit(as_of_date=as_of_date, data=data, result=result)


def run_diffusion_one_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    target: DailyDigitalTarget | None = None,
    model_config: DiffusionModelConfig,
    checkpoint_directory: Path,
    observation_count: int = 252,
    train_config: DiffusionTrainConfig = DiffusionTrainConfig(),
    hedge_config: GeneratedHedgeConfig = GeneratedHedgeConfig(),
    base_checkpoint: Path | None = None,
    warm_start_checkpoint: Path | None = None,
    expected_base_checkpoint_sha256: str | None = None,
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
    return _run_diffusion_targets(
        panel,
        origin_date=origin_date,
        targets=(target,),
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=train_config,
        hedge_config=hedge_config,
        base_checkpoint=base_checkpoint,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_warm_start_sha256=expected_warm_start_sha256,
        retain_fit_checkpoint=retain_fit_checkpoint,
        environment=environment,
        claim_quantities={target.label: claim_quantity},
    )[0]


def run_diffusion_daily_book(
    panel: PublicHedgingPanel,
    *,
    target_maturity: int | None = None,
    origin_date: date,
    model_config: DiffusionModelConfig,
    checkpoint_directory: Path,
    observation_count: int = 252,
    train_config: DiffusionTrainConfig = DiffusionTrainConfig(),
    hedge_config: GeneratedHedgeConfig = GeneratedHedgeConfig(),
    base_checkpoint: Path | None = None,
    warm_start_checkpoint: Path | None = None,
    expected_base_checkpoint_sha256: str | None = None,
    expected_warm_start_sha256: str | None = None,
    retain_fit_checkpoint: bool = True,
    environment: BaseEnvironment | None = None,
    claim_quantities: Mapping[str, float] | None = None,
    scenario_artifact_directory: Path | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Fit and generate once, then hedge all 20 daily digital targets."""
    return _run_diffusion_targets(
        panel,
        origin_date=origin_date,
        targets=daily_digital_targets(maturity_business_days=target_maturity),
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=train_config,
        hedge_config=hedge_config,
        base_checkpoint=base_checkpoint,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_warm_start_sha256=expected_warm_start_sha256,
        retain_fit_checkpoint=retain_fit_checkpoint,
        environment=environment,
        claim_quantities=claim_quantities,
        scenario_artifact_directory=scenario_artifact_directory,
    )


def _run_diffusion_targets(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    targets: tuple[DailyDigitalTarget, ...],
    model_config: DiffusionModelConfig,
    checkpoint_directory: Path,
    observation_count: int,
    train_config: DiffusionTrainConfig,
    hedge_config: GeneratedHedgeConfig,
    base_checkpoint: Path | None,
    warm_start_checkpoint: Path | None,
    expected_base_checkpoint_sha256: str | None,
    expected_warm_start_sha256: str | None,
    retain_fit_checkpoint: bool,
    environment: BaseEnvironment | None,
    claim_quantities: Mapping[str, float] | None,
    scenario_artifact_directory: Path | None = None,
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
        raise RuntimeError("diffusion daily targets do not share a hedge universe")
    public_fit = fit_public_diffusion(
        panel,
        as_of_date=origin_date,
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=train_config,
        base_checkpoint=base_checkpoint,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_warm_start_sha256=expected_warm_start_sha256,
        environment=environment,
    )
    live = build_diffusion_live_context(
        panel,
        as_of_date=origin_date,
        grid=model_config.grid,
    )
    if live.forecast_date != samples[0].transition.next_date:
        raise RuntimeError("diffusion forecast date does not match hedge endpoint")
    scenario_count = adaptive_scenario_count(
        1 + len(samples[0].hedge_quotes),
        minimum=hedge_config.scenario_count,
    )
    generated = sample_diffusion_transitions(
        public_fit.result,
        observed_sequence=live.sequence,
        path_count=scenario_count,
        seed=hedge_config.seed,
    )
    fit_checkpoint_sha256 = file_sha256(public_fit.result.checkpoint_path)
    scenario_artifact_path = None
    scenario_artifact_sha256 = None
    if scenario_artifact_directory is not None:
        scenario_artifact_path = (
            scenario_artifact_directory / f"{origin_date.isoformat()}.npz"
        )
        scenario_artifact_sha256 = _save_generated_scenarios(
            scenario_artifact_path,
            forward_log_returns=generated.forward_log_returns,
            next_log_implied_volatilities=generated.next_log_implied_volatilities,
            origin_date=origin_date,
            forecast_date=live.forecast_date,
            dataset_manifest_sha256=panel.dataset_manifest_sha256,
            fit_checkpoint_sha256=fit_checkpoint_sha256,
            warm_start_checkpoint_sha256=expected_warm_start_sha256,
            base_checkpoint_sha256=expected_base_checkpoint_sha256,
            model_config=model_config,
            train_config=train_config,
            hedge_config=hedge_config,
        )
    scenarios = tuple(
        build_generated_daily_scenarios(
            sample,
            forward_log_returns=generated.forward_log_returns,
            next_log_implied_volatilities=generated.next_log_implied_volatilities,
            grid=model_config.grid,
        )
        for sample in samples
    )
    hedges = fit_generated_objective_book(
        scenarios,
        samples[0],
        config=hedge_config,
        claim_quantities=quantities,
    )
    results = tuple(
        _evaluate_diffusion_target(
            sample,
            public_fit=public_fit,
            scenarios=scenarios[index],
            hedge=hedges[index],
            generated_forward_log_returns=generated.forward_log_returns,
            return_channel_max_dispersion=(
                generated.return_channel_max_dispersion
            ),
            model_config=model_config,
            hedge_config=hedge_config,
            target_count=len(samples),
            claim_quantity=float(quantities[index]),
            fit_checkpoint_sha256=fit_checkpoint_sha256,
            scenario_artifact_path=scenario_artifact_path,
            scenario_artifact_sha256=scenario_artifact_sha256,
        )
        for index, sample in enumerate(samples)
    )
    if not retain_fit_checkpoint:
        remove_completed_checkpoint(public_fit.result.checkpoint_path)
    return results


def _save_generated_scenarios(
    path: Path,
    *,
    forward_log_returns: np.ndarray,
    next_log_implied_volatilities: np.ndarray,
    origin_date: date,
    forecast_date: date,
    dataset_manifest_sha256: str,
    fit_checkpoint_sha256: str,
    warm_start_checkpoint_sha256: str | None,
    base_checkpoint_sha256: str | None,
    model_config: DiffusionModelConfig,
    train_config: DiffusionTrainConfig,
    hedge_config: GeneratedHedgeConfig,
) -> str:
    config_json = json.dumps(
        {
            "model": asdict(model_config),
            "training": asdict(train_config),
            "simulation": {
                "seed": hedge_config.seed,
                "path_count": len(forward_log_returns),
            },
        },
        default=str,
        sort_keys=True,
    )
    metadata_json = json.dumps(
        {
            "schema": "diffusion-generated-scenarios-v1",
            "origin_date": origin_date.isoformat(),
            "forecast_date": forecast_date.isoformat(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "fit_checkpoint_sha256": fit_checkpoint_sha256,
            "warm_start_checkpoint_sha256": warm_start_checkpoint_sha256,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "configuration": json.loads(config_json),
            "seed": hedge_config.seed,
            "path_count": len(forward_log_returns),
        },
        sort_keys=True,
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (
                saved["metadata_json"].item() != metadata_json
                or not np.array_equal(
                    saved["forward_log_returns"], forward_log_returns
                )
                or not np.array_equal(
                    saved["next_log_implied_volatilities"],
                    next_log_implied_volatilities,
                )
            ):
                raise RuntimeError(f"generated scenario artifact differs: {path}")
        return file_sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(
                handle,
                metadata_json=metadata_json,
                forward_log_returns=forward_log_returns,
                next_log_implied_volatilities=next_log_implied_volatilities,
            )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    return file_sha256(path)


def _evaluate_diffusion_target(
    sample: DailyDigitalSample,
    *,
    public_fit: DiffusionPublicFit,
    scenarios: GeneratedDailyScenarios,
    hedge: GeneratedObjectiveFit,
    generated_forward_log_returns: np.ndarray,
    return_channel_max_dispersion: float,
    model_config: DiffusionModelConfig,
    hedge_config: GeneratedHedgeConfig,
    target_count: int,
    claim_quantity: float,
    fit_checkpoint_sha256: str,
    scenario_artifact_path: Path | None = None,
    scenario_artifact_sha256: str | None = None,
) -> HedgeSmokeResult:
    return evaluate_daily_digital_hedge(
        sample,
        model=f"diffusion-{model_config.variant.value}",
        measure="historical-P",
        spot_action=float(hedge.action[0]),
        option_actions=hedge.action[1:],
        model_price=sample.market_price,
        objective_value=hedge.objective_value,
        converged=(public_fit.result.converged and hedge.converged),
        cap_hits=hedge.cap_hits,
        claim_quantity=claim_quantity,
        diagnostics={
            "training_sequences": len(public_fit.data.start_dates),
            "target_count": target_count,
            "claim_quantity": claim_quantity,
            "completed_epochs": public_fit.result.completed_epochs,
            "validation_loss": public_fit.result.validation_loss,
            "warm_start_sha256": public_fit.result.warm_start_sha256,
            "fit_checkpoint_path": str(public_fit.result.checkpoint_path),
            "fit_checkpoint_sha256": fit_checkpoint_sha256,
            "scenario_artifact_path": (
                str(scenario_artifact_path) if scenario_artifact_path else None
            ),
            "scenario_artifact_sha256": scenario_artifact_sha256,
            "generated_paths": len(generated_forward_log_returns),
            "generated_mean_forward_log_return": float(
                np.mean(generated_forward_log_returns)
            ),
            "generated_std_forward_log_return": float(
                np.std(generated_forward_log_returns)
            ),
            "generated_negative_return_fraction": float(
                np.mean(generated_forward_log_returns < 0.0)
            ),
            "return_channel_max_dispersion": return_channel_max_dispersion,
            "generated_mean_claim_value": float(
                np.mean(scenarios.claim_terminal_values)
            ),
            "digital_probability_clip_count": (
                scenarios.digital_probability_clip_count
            ),
            "hedge_instrument_count": scenarios.instrument_returns.shape[1],
            "objective": hedge_config.objective.value,
            "risk_parameter": hedge_config.risk_parameter,
            "mean_reduction": hedge_config.mean_reduction.value,
            "mean_reduction_constraint_rank": (
                hedge.mean_reduction.constraint_rank
            ),
            "mean_reduction_effective_sample_size": (
                hedge.mean_reduction.effective_sample_size
            ),
            "mean_reduction_kl_divergence": (
                hedge.mean_reduction.kl_divergence
            ),
            "mean_reduction_raw_max_standardized_mean": (
                hedge.mean_reduction.raw_max_standardized_mean
            ),
            "mean_reduction_reduced_max_standardized_mean": (
                hedge.mean_reduction.reduced_max_standardized_mean
            ),
            "mean_reduction_moment_residual": (
                hedge.mean_reduction.moment_residual
            ),
            "cost_multiplier": hedge_config.cost_multiplier,
            "optimizer_iterations": hedge.optimizer.iterations,
            "optimizer_kkt_residual": hedge.optimizer.maximum_kkt_residual,
            "policy_semantics": (
                "drift-reduced weighted scenario objective with exact linear "
                "entry costs on generated fixed-contract returns"
            ),
            "price_semantics": (
                "common released SANOS market mark; diffusion supplies P "
                "scenarios, not a risk-neutral price"
            ),
            "payoff_units": "discounted origin-PV per unit origin spot",
        },
    )
