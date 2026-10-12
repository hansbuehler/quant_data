"""Run a resumable public-data daily hedge benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from cdxcore.uniquehash import named_unique_filename48_8

from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.sizing import (
    TargetNotionalConfig,
    calibrate_daily_target_notionals,
    daily_target_notional_artifact_path,
    daily_target_notional_calibration_sha256,
    load_authenticated_daily_target_notionals,
    load_or_calibrate_daily_target_notionals,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.adapter import (
    run_covariance_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.config import (
    CovarianceHedgeConfig,
    CovarianceObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.adapter import (
    run_diffusion_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedHedgeConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.adapter import (
    run_gromm2_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Config,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.adapter import (
    run_historical_sanos_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.adapter import (
    run_sanos_lv_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.adapter import (
    run_volgan_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.config import (
    GeneratedDailyRunConfig,
    GrommDailyRunConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DailyBookFunction,
    DailyEvaluationSpec,
    dataset_manifest_sha256,
    run_daily_evaluation,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    summarize_completed_daily_run,
)
from quantdata.torch import BaseEnvironment


def main() -> None:
    """Parse one production run, execute it, and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "sanos-lv",
            "covariance",
            "historical-sanos",
            "gromm",
            "volgan",
            "diffusion-ad-seq-vol",
            "diffusion-ad-seq-vol-ft",
        ),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--objective", default="mean-variance")
    parser.add_argument("--risk-parameter", type=float, default=10.0)
    parser.add_argument("--measure", choices=("P", "Q"), default="Q")
    parser.add_argument("--position-cap", type=float)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--market-artifact-root", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--observation-count", type=int, default=252)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--scenario-count", type=int, default=4_096)
    parser.add_argument("--training-progress-seconds", type=float, default=240.0)
    parser.add_argument("--checkpoint-epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", type=int, choices=(32, 64))
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--just-load", action="store_true")
    parser.add_argument("--retain-fit-checkpoints", action="store_true")
    parser.add_argument("--target-notional-artifact-sha256")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--allow-nonconvergence", action="store_true")
    parser.add_argument("--validate-dataset", action="store_true")
    parser.add_argument("--skip-results", action="store_true")
    arguments = parser.parse_args()
    if arguments.progress_every <= 0:
        parser.error("--progress-every must be positive")

    panel = PublicHedgingPanel(
        arguments.dataset,
        validate=arguments.validate_dataset,
    )
    notional_config = TargetNotionalConfig()
    root = os.environ.get("QUANTDATA_ROOT")
    if root:
        target_notional_cache = (
            Path(root) / "processing" / "hedging" / "target-notionals"
        )
        target_notional_path = daily_target_notional_artifact_path(
            panel,
            cache_directory=target_notional_cache,
            config=notional_config,
        )
        if arguments.target_notional_artifact_sha256 is None:
            source = "loading" if target_notional_path.exists() else "calibrating"
            print(
                f"{source} fixed 2022 target notionals: {target_notional_path}",
                flush=True,
            )
            target_notionals = load_or_calibrate_daily_target_notionals(
                panel,
                cache_directory=target_notional_cache,
                config=notional_config,
            )
        else:
            print(
                "loading CPU-preflighted fixed 2022 target notionals: "
                f"{target_notional_path}",
                flush=True,
            )
            target_notionals = load_authenticated_daily_target_notionals(
                panel,
                cache_directory=target_notional_cache,
                expected_artifact_sha256=(
                    arguments.target_notional_artifact_sha256
                ),
                config=notional_config,
            )
    else:
        if arguments.target_notional_artifact_sha256 is not None:
            parser.error(
                "--target-notional-artifact-sha256 requires QUANTDATA_ROOT"
            )
        print("calibrating fixed 2022 target notionals without cache", flush=True)
        target_notionals = calibrate_daily_target_notionals(
            panel,
            config=notional_config,
        )
    claim_quantities = target_notionals.quantity_by_label
    checkpoint_directory = arguments.checkpoint_directory
    if checkpoint_directory is None:
        root = os.environ.get("QUANTDATA_ROOT")
        if not root and _is_generated_model(arguments.model):
            raise RuntimeError(
                "QUANTDATA_ROOT or --checkpoint-directory is required for generated models"
            )
        if root:
            checkpoint_directory = (
                Path(root) / "training" / "hedging" / "daily" / arguments.model
            )
    model, measure, rolling_observations, config, run_book = _build_model(
        arguments,
        claim_quantities=claim_quantities,
        default_position_cap=notional_config.numerical_position_cap,
        checkpoint_directory=checkpoint_directory,
    )
    config_identity = _config_identity(config)
    output = arguments.output
    if output is None:
        root = os.environ.get("QUANTDATA_ROOT")
        if not root:
            raise RuntimeError("QUANTDATA_ROOT is required for the default output path")
        output = (
            Path(root)
            / "experiments"
            / "hedging"
            / "daily"
            / model
            / config_identity.replace(" ", "-")
        )
    spec = DailyEvaluationSpec(
        label=config.label,
        model=model,
        measure=measure,
        model_config_identity=config_identity,
        model_config_snapshot_json=_config_snapshot_json(config),
        dataset_manifest_sha256=dataset_manifest_sha256(panel),
        evaluation_start=arguments.start,
        evaluation_end=arguments.end,
        rolling_observations=rolling_observations,
        target_labels=tuple(target.label for target in daily_digital_targets()),
        target_quantities=tuple(
            claim_quantities[target.label] for target in daily_digital_targets()
        ),
        target_notional_identity=(
            f"{type(notional_config).version.unique_id48}:"
            f"{daily_target_notional_calibration_sha256(target_notionals)}"
        ),
        require_convergence=not arguments.allow_nonconvergence,
    )
    started = time.monotonic()

    def progress(index: int, total: int, origin_date: date, reused: bool) -> None:
        if index != total and index % arguments.progress_every != 0:
            return
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed > 0.0 else 0.0
        eta = (total - index) / rate if rate > 0.0 else float("nan")
        source = "reused" if reused else "computed"
        print(
            f"daily progress {index}/{total} origin={origin_date} {source} "
            f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
            flush=True,
        )

    summary = run_daily_evaluation(
        panel,
        spec=spec,
        run_book=run_book,
        output_directory=output,
        progress=progress,
        stop_after=arguments.stop_after,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), flush=True)
    if arguments.stop_after is None and not arguments.skip_results:
        report = summarize_completed_daily_run(panel, run_directory=output)
        results_path = output / "results.json"
        report.write_json(results_path)
        print(
            json.dumps(
                {
                    "schema": "hedging-daily-results-summary-v1",
                    "run_identity": report.run_identity,
                    "result_count": report.result_count,
                    "slice_count": len(report.slices),
                    "output": str(results_path),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


def _build_model(
    arguments: argparse.Namespace,
    *,
    claim_quantities: dict[str, float],
    default_position_cap: float,
    checkpoint_directory: Path | None,
) -> tuple[str, str, int, Any, DailyBookFunction]:
    label = arguments.label or (
        f"{arguments.model}-rolling-public-settings"
        if _is_generated_model(arguments.model)
        else (
            f"{arguments.model}-{arguments.objective}-"
            f"lambda-{arguments.risk_parameter:g}"
        )
    )
    if arguments.model == "sanos-lv":
        config = SanosLvConfig(
            label=label,
            position_cap=(
                default_position_cap
                if arguments.position_cap is None
                else arguments.position_cap
            ),
        )
        return (
            "sanos-lv",
            "Q-price/historical-transition",
            config.rolling_factors.window_sessions,
            config,
            lambda panel, origin: run_sanos_lv_daily_book(
                panel,
                origin_date=origin,
                config=config,
                claim_quantities=claim_quantities,
            ),
        )
    if arguments.model == "covariance":
        try:
            objective = CovarianceObjective(arguments.objective)
        except ValueError as error:
            raise ValueError(
                "covariance supports mean-variance and mean-volatility"
            ) from error
        config = CovarianceHedgeConfig(
            label=label,
            objective=objective,
            risk_parameter=arguments.risk_parameter,
            position_cap=(
                default_position_cap
                if arguments.position_cap is None
                else arguments.position_cap
            ),
        )
        return (
            "robust-covariance",
            "historical-P",
            config.rolling_observations,
            config,
            lambda panel, origin: run_covariance_daily_book(
                panel,
                origin_date=origin,
                config=config,
                claim_quantities=claim_quantities,
            ),
        )
    if arguments.model == "gromm":
        if checkpoint_directory is None:
            raise ValueError("GROMM requires a policy checkpoint directory")
        artifact_root = arguments.market_artifact_root
        if artifact_root is None:
            root = os.environ.get("QUANTDATA_ROOT")
            if not root:
                raise ValueError(
                    "GROMM requires --market-artifact-root or QUANTDATA_ROOT"
                )
            artifact_root = Path(root) / "training" / "hedging" / "gromm2-rolling"
        objective = GrommObjective(arguments.objective)
        measure = GrommMeasure(arguments.measure)
        scenario_batch = (
            arguments.scenario_count
            if arguments.batch_size is None
            else arguments.batch_size
        )
        model_config = GrommConfig(
            label=label,
            measure=measure,
            objective=objective,
            risk_parameter=arguments.risk_parameter,
            path_count=arguments.scenario_count,
            batch_size=scenario_batch,
            position_cap=(
                default_position_cap
                if arguments.position_cap is None
                else arguments.position_cap
            ),
        )
        train_defaults = GrommTrainConfig()
        train_config = GrommTrainConfig(
            maximum_epochs=(
                train_defaults.maximum_epochs
                if arguments.epochs is None
                else arguments.epochs
            ),
            minimum_epochs=(
                train_defaults.minimum_epochs
                if arguments.epochs is None
                else min(train_defaults.minimum_epochs, arguments.epochs)
            ),
            patience_epochs=train_defaults.patience_epochs,
            batch_pairs=train_defaults.batch_pairs,
            learning_rate=train_defaults.learning_rate,
            weight_decay=train_defaults.weight_decay,
            validation_fraction=train_defaults.validation_fraction,
            minimum_relative_improvement=(
                train_defaults.minimum_relative_improvement
            ),
            seed=train_defaults.seed,
            progress_seconds=arguments.training_progress_seconds,
            checkpoint_epochs=(
                train_defaults.checkpoint_epochs
                if arguments.checkpoint_epochs is None
                else arguments.checkpoint_epochs
            ),
            compile_policy=arguments.compile_model,
            just_load=arguments.just_load,
        )
        rolling_config = RollingGromm2Config()
        run_config = GrommDailyRunConfig(
            label=label,
            model_config=model_config,
            rolling_config=rolling_config,
            train_config=train_config,
            device=arguments.device,
            precision=64,
        )
        environment = BaseEnvironment(
            precision=64,
            device=arguments.device,
            debug=False,
        )
        return (
            "gromm",
            measure.value,
            rolling_config.window_transitions,
            run_config,
            lambda panel, origin: run_gromm2_daily_book(
                panel,
                origin_date=origin,
                artifact_root=artifact_root,
                checkpoint_directory=checkpoint_directory,
                config=model_config,
                rolling_config=rolling_config,
                train_config=train_config,
                environment=environment,
                claim_quantities=claim_quantities,
            ),
        )
    if arguments.model == "volgan":
        if checkpoint_directory is None:
            raise ValueError("VolGAN requires a checkpoint directory")
        model_config = VolganModelConfig(label=label)
        train_defaults = VolganTrainConfig()
        train_config = VolganTrainConfig(
            epochs=(train_defaults.epochs if arguments.epochs is None else arguments.epochs),
            gradient_matching_epochs=(
                0
                if arguments.warm_start_checkpoint is not None
                else min(
                    train_defaults.gradient_matching_epochs,
                    (
                        train_defaults.epochs
                        if arguments.epochs is None
                        else arguments.epochs
                    ),
                )
            ),
            batch_size=(
                train_defaults.batch_size
                if arguments.batch_size is None
                else arguments.batch_size
            ),
            generator_learning_rate=train_defaults.generator_learning_rate,
            discriminator_learning_rate=train_defaults.discriminator_learning_rate,
            validation_fraction=train_defaults.validation_fraction,
            seed=train_defaults.seed,
            progress_seconds=arguments.training_progress_seconds,
            checkpoint_epochs=(
                train_defaults.checkpoint_epochs
                if arguments.checkpoint_epochs is None
                else arguments.checkpoint_epochs
            ),
            compile_models=arguments.compile_model,
            just_load=arguments.just_load,
        )
        hedge_config = GeneratedHedgeConfig(
            scenario_count=arguments.scenario_count,
            position_cap=(
                default_position_cap
                if arguments.position_cap is None
                else arguments.position_cap
            ),
        )
        precision = 64 if arguments.precision is None else arguments.precision
        warm_start_fingerprint = (
            None
            if arguments.warm_start_checkpoint is None
            else file_sha256(arguments.warm_start_checkpoint)
        )
        run_config = GeneratedDailyRunConfig(
            label=label,
            model_config=model_config,
            train_config=train_config,
            hedge_config=hedge_config,
            observation_count=arguments.observation_count,
            device=arguments.device,
            precision=precision,
            warm_start_checkpoint_sha256=warm_start_fingerprint,
        )
        environment = BaseEnvironment(
            precision=precision,
            device=arguments.device,
            debug=False,
        )
        return (
            "volgan",
            "historical-P",
            arguments.observation_count,
            run_config,
            lambda panel, origin: run_volgan_daily_book(
                panel,
                origin_date=origin,
                model_config=model_config,
                checkpoint_directory=checkpoint_directory,
                observation_count=arguments.observation_count,
                train_config=train_config,
                hedge_config=hedge_config,
                warm_start_checkpoint=arguments.warm_start_checkpoint,
                expected_warm_start_sha256=warm_start_fingerprint,
                retain_fit_checkpoint=arguments.retain_fit_checkpoints,
                environment=environment,
                claim_quantities=claim_quantities,
            ),
        )
    if arguments.model.startswith("diffusion-"):
        if checkpoint_directory is None:
            raise ValueError("diffusion requires a checkpoint directory")
        variant = DiffusionVariant(arguments.model.removeprefix("diffusion-"))
        if variant is DiffusionVariant.AD_SEQ_VOL_FT and arguments.base_checkpoint is None:
            raise ValueError("fine-tuned diffusion requires --base-checkpoint")
        if variant is DiffusionVariant.AD_SEQ_VOL and arguments.base_checkpoint is not None:
            raise ValueError("base checkpoint is only valid for fine-tuned diffusion")
        model_config = DiffusionModelConfig(label=label, variant=variant)
        train_defaults = DiffusionTrainConfig()
        train_config = DiffusionTrainConfig(
            epochs=(train_defaults.epochs if arguments.epochs is None else arguments.epochs),
            batch_size=(
                train_defaults.batch_size
                if arguments.batch_size is None
                else arguments.batch_size
            ),
            learning_rate=train_defaults.learning_rate,
            weight_decay=train_defaults.weight_decay,
            warmup_steps=(
                0
                if arguments.warm_start_checkpoint is not None
                else train_defaults.warmup_steps
            ),
            ema_decay=train_defaults.ema_decay,
            validation_fraction=train_defaults.validation_fraction,
            arbitrage_penalty=train_defaults.arbitrage_penalty,
            return_broadcast_penalty=train_defaults.return_broadcast_penalty,
            seed=train_defaults.seed,
            progress_seconds=arguments.training_progress_seconds,
            checkpoint_epochs=(
                train_defaults.checkpoint_epochs
                if arguments.checkpoint_epochs is None
                else arguments.checkpoint_epochs
            ),
            compile_model=arguments.compile_model,
            mixed_precision=(
                train_defaults.mixed_precision
                and not arguments.no_mixed_precision
                and arguments.device == "cuda"
            ),
            just_load=arguments.just_load,
        )
        hedge_config = GeneratedHedgeConfig(
            scenario_count=arguments.scenario_count,
            position_cap=(
                default_position_cap
                if arguments.position_cap is None
                else arguments.position_cap
            ),
        )
        precision = 32 if arguments.precision is None else arguments.precision
        base_fingerprint = (
            None
            if arguments.base_checkpoint is None
            else file_sha256(arguments.base_checkpoint)
        )
        warm_start_fingerprint = (
            None
            if arguments.warm_start_checkpoint is None
            else file_sha256(arguments.warm_start_checkpoint)
        )
        run_config = GeneratedDailyRunConfig(
            label=label,
            model_config=model_config,
            train_config=train_config,
            hedge_config=hedge_config,
            observation_count=arguments.observation_count,
            device=arguments.device,
            precision=precision,
            base_checkpoint_sha256=base_fingerprint,
            warm_start_checkpoint_sha256=warm_start_fingerprint,
        )
        environment = BaseEnvironment(
            precision=precision,
            device=arguments.device,
            debug=False,
        )
        return (
            arguments.model,
            "historical-P",
            arguments.observation_count,
            run_config,
            lambda panel, origin: run_diffusion_daily_book(
                panel,
                origin_date=origin,
                model_config=model_config,
                checkpoint_directory=checkpoint_directory,
                observation_count=arguments.observation_count,
                train_config=train_config,
                hedge_config=hedge_config,
                base_checkpoint=arguments.base_checkpoint,
                warm_start_checkpoint=arguments.warm_start_checkpoint,
                expected_base_checkpoint_sha256=base_fingerprint,
                expected_warm_start_sha256=warm_start_fingerprint,
                retain_fit_checkpoint=arguments.retain_fit_checkpoints,
                environment=environment,
                claim_quantities=claim_quantities,
            ),
        )
    try:
        historical_objective = HistoricalSanosObjective(arguments.objective)
    except ValueError as error:
        raise ValueError(
            "historical SANOS supports mean-variance, mean-volatility, entropy, "
            "and power-exponential"
        ) from error
    historical_config = HistoricalSanosConfig(
        label=label,
        objective=historical_objective,
        risk_parameter=arguments.risk_parameter,
        position_cap=(
            default_position_cap
            if arguments.position_cap is None
            else arguments.position_cap
        ),
    )
    return (
        "historical-sanos",
        "historical-P",
        historical_config.rolling_observations,
        historical_config,
        lambda panel, origin: run_historical_sanos_daily_book(
            panel,
            origin_date=origin,
            config=historical_config,
            claim_quantities=claim_quantities,
        ),
    )


def _is_generated_model(model: str) -> bool:
    return model == "volgan" or model.startswith("diffusion-")


def _config_identity(config: Any) -> str:
    return (
        f"{type(config).__name__}-{config.version.unique_id48}-"
        f"{named_unique_filename48_8(config.label, config)}"
    )


def _config_snapshot_json(config: Any) -> str:
    return json.dumps(
        {
            "class": f"{type(config).__module__}.{type(config).__qualname__}",
            "code_version": str(config.version),
            "fields": asdict(config),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


if __name__ == "__main__":
    main()
