"""Public-panel daily hedge adapter for the single-asset FuNVol adaptation."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    build_sanos_fixed_grid_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import require_public_hedging_panel
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.arbitrage import (
    screen_generated_iv_surfaces,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import (
    FunvolConfig,
    FunvolTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.training import (
    fit_funvol,
    sample_funvol_one_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedHedgeConfig,
    build_generated_daily_scenarios,
    fit_generated_objective_book,
)
from quantdata.torch import BaseEnvironment


def run_funvol_one_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    checkpoint_path: Path,
    warm_start_checkpoint: Path | None = None,
    target: DailyDigitalTarget = DailyDigitalTarget(
        maturity_business_days=1,
        right=OptionRight.CALL,
        nms=0.0,
    ),
    model_config: FunvolConfig = FunvolConfig(),
    train_config: FunvolTrainConfig = FunvolTrainConfig(),
    hedge_config: GeneratedHedgeConfig = GeneratedHedgeConfig(),
    observation_count: int = 252,
    environment: BaseEnvironment | None = None,
    claim_quantity: float = 1.0,
) -> HedgeSmokeResult:
    """Train through the origin, hedge a fixed digital, and score next-day P&L.

    Model marks and hedge-book quotes come only from the public SANOS panel.
    Generated FuNVol scenarios supply a physical-law hedge, not a Q price.
    """
    return _run_funvol_targets(
        panel,
        origin_date=origin_date,
        checkpoint_path=checkpoint_path,
        warm_start_checkpoint=warm_start_checkpoint,
        targets=(target,),
        model_config=model_config,
        train_config=train_config,
        hedge_config=hedge_config,
        observation_count=observation_count,
        environment=environment,
        claim_quantities={target.label: claim_quantity},
    )[0]


def run_funvol_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    checkpoint_path: Path,
    warm_start_checkpoint: Path | None = None,
    model_config: FunvolConfig = FunvolConfig(),
    train_config: FunvolTrainConfig = FunvolTrainConfig(),
    hedge_config: GeneratedHedgeConfig = GeneratedHedgeConfig(),
    observation_count: int = 252,
    environment: BaseEnvironment | None = None,
    claim_quantities: Mapping[str, float] | None = None,
    scenario_output_path: Path | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Fit one model and scenario book for all ten one-day digital targets."""
    return _run_funvol_targets(
        panel,
        origin_date=origin_date,
        checkpoint_path=checkpoint_path,
        warm_start_checkpoint=warm_start_checkpoint,
        targets=daily_digital_targets(maturity_business_days=1),
        model_config=model_config,
        train_config=train_config,
        hedge_config=hedge_config,
        observation_count=observation_count,
        environment=environment,
        claim_quantities=claim_quantities,
        scenario_output_path=scenario_output_path,
    )


def _run_funvol_targets(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    checkpoint_path: Path,
    warm_start_checkpoint: Path | None,
    targets: tuple[DailyDigitalTarget, ...],
    model_config: FunvolConfig,
    train_config: FunvolTrainConfig,
    hedge_config: GeneratedHedgeConfig,
    observation_count: int,
    environment: BaseEnvironment | None,
    claim_quantities: Mapping[str, float] | None,
    scenario_output_path: Path | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    require_public_hedging_panel(panel)
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    if any(sample.hedge_quotes != samples[0].hedge_quotes for sample in samples[1:]):
        raise RuntimeError("FuNVol targets do not share the same fixed hedge book")
    history = build_sanos_fixed_grid_history(
        panel,
        end_date=origin_date,
        observation_count=observation_count,
        grid=model_config.grid,
    )
    historical_return_std = float(np.std(history.forward_log_returns_into_date[1:]))
    if samples[0].transition.next_date not in panel.dates:
        raise ValueError("FuNVol hedge endpoint is absent from public panel")
    fit = fit_funvol(
        history,
        config=model_config,
        train_config=train_config,
        environment=environment,
        checkpoint_path=checkpoint_path,
        warm_start_checkpoint=warm_start_checkpoint,
    )
    path_count = adaptive_scenario_count(
        1 + len(samples[0].hedge_quotes),
        minimum=hedge_config.scenario_count,
    )
    log_returns, next_volatilities = sample_funvol_one_day(
        fit,
        history,
        path_count=path_count,
        seed=hedge_config.seed,
    )
    static_screen = screen_generated_iv_surfaces(
        next_volatilities[: min(path_count, 512)],
        grid=model_config.grid,
        log_strikes=np.linspace(-0.005, 0.005, 11),
    )
    scenarios = tuple(
        build_generated_daily_scenarios(
            sample,
            forward_log_returns=log_returns,
            next_log_implied_volatilities=np.log(next_volatilities),
            grid=model_config.grid,
        )
        for sample in samples
    )
    checkpoint_sha256 = file_sha256(checkpoint_path)
    scenario_sha256 = None
    if scenario_output_path is not None:
        provenance = json.dumps(
            {
                "schema": "funvol-generated-scenarios-v1",
                "origin_date": origin_date.isoformat(),
                "dataset_manifest_sha256": panel.dataset_manifest_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "scenario_seed": hedge_config.seed,
                "target_labels": [target.label for target in targets],
            },
            sort_keys=True,
        )
        arrays = {
            "provenance_json": np.asarray(provenance),
            "forward_log_returns": log_returns,
            "next_implied_volatilities": next_volatilities,
            "instrument_returns": scenarios[0].instrument_returns,
            "target_returns": np.column_stack([scenario.claim_returns for scenario in scenarios]),
        }
        scenario_output_path.parent.mkdir(parents=True, exist_ok=True)
        if scenario_output_path.exists():
            with np.load(scenario_output_path) as saved:
                if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[name], values) for name, values in arrays.items()
                ):
                    raise RuntimeError("FuNVol saved scenario book does not match this fit")
        else:
            temporary = scenario_output_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(scenario_output_path)
        scenario_sha256 = file_sha256(scenario_output_path)
    hedges = fit_generated_objective_book(
        scenarios,
        samples[0],
        config=hedge_config,
        claim_quantities=quantities,
    )
    return tuple(
        evaluate_daily_digital_hedge(
            sample,
            model="funvol-single-asset",
            measure="historical-P",
            spot_action=float(hedge.action[0]),
            option_actions=hedge.action[1:],
            model_price=sample.market_price,
            objective_value=hedge.objective_value,
            converged=hedge.converged,
            cap_hits=hedge.cap_hits,
            claim_quantity=float(quantity),
            diagnostics={
                "completed_epochs": fit.completed_epochs,
                "training_loss": fit.training_loss,
                "validation_loss": fit.validation_loss,
                "fpc_explained_variance": fit.fpc.explained_variance_ratio.tolist(),
                "generated_paths": path_count,
                "generated_return_std": float(np.std(log_returns)),
                "historical_return_std": historical_return_std,
                "static_arbitrage_screen": vars(static_screen),
                "fit_checkpoint_path": str(checkpoint_path),
                "fit_checkpoint_sha256": checkpoint_sha256,
                "scenario_book_path": str(scenario_output_path) if scenario_output_path else None,
                "scenario_book_sha256": scenario_sha256,
                "warm_start_date": fit.warm_start_date,
                "warm_start_sha256": fit.warm_start_sha256,
                "mean_reduction": hedge_config.mean_reduction.value,
                "objective": hedge_config.objective.value,
                "risk_parameter": hedge_config.risk_parameter,
                "hedge_instrument_count": scenarios[index].instrument_returns.shape[1],
                "price_semantics": "common released SANOS market mark; FuNVol supplies P scenarios",
                "payoff_units": "discounted origin-PV per unit origin spot",
            },
        )
        for index, (sample, hedge, quantity) in enumerate(
            zip(samples, hedges, quantities, strict=True)
        )
    )
