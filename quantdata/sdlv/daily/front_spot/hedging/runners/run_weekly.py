"""Run a resumable public-data frozen or restarted weekly hedge benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    build_weekly_episode_path,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyMarketFit,
    WeeklyModelAdapter,
    WeeklyModelRun,
    run_weekly_model,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_sizing import (
    WeeklyTargetNotionalCalibration,
    WeeklyTargetNotionalConfig,
    calibrate_weekly_target_notionals,
    load_authenticated_weekly_target_notionals,
    load_authenticated_weekly_target_notionals_from_path,
    load_or_calibrate_weekly_target_notionals,
    weekly_target_notional_artifact_path,
    weekly_target_notional_calibration_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.weekly_adapter import (
    DiffusionWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly_policy import (
    GeneratedWeeklyPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.conventional_weekly import (
    GrommConventionalWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Config,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.selection import (
    load_gromm_architecture_selection,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.weekly_adapter import (
    GrommWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_adapter import (
    HistoricalSanosWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_ridge import (
    load_weekly_ridge_for_scope,
    repository_source_commit,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_adapter import (
    SanosLvWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    build_common_weekly_claims,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganPcaModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.weekly_adapter import (
    VolganWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2_range import (
    load_completed_rolling_gromm2_range,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.progress import (
    EvaluationProgressHeartbeat,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyBookFunction,
    WeeklyEvaluationSpec,
    eligible_weekly_origins,
    run_weekly_evaluation,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_results import (
    summarize_completed_weekly_run,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class WeeklyAdapterBuild:
    """One fully identified weekly model adapter and its runtime controls."""

    adapter: WeeklyModelAdapter
    rolling_observations: int
    position_cap: float
    configuration_snapshot_json: str


def main() -> None:
    """Parse, run, and summarize one complete weekly benchmark cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "sanos-lv",
            "historical-sanos",
            "gromm",
            "gromm-q-dlv-vega",
            "volgan",
            "diffusion-ad-seq-vol",
            "diffusion-ad-seq-vol-ft",
        ),
        required=True,
    )
    parser.add_argument(
        "--schedule",
        choices=("frozen-weekly", "restarted-weekly"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--surface-grid", choices=("published", "short-end-nms"), default="published")
    parser.add_argument("--volgan-pca-rank", type=int)
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--model-label")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--objective", default="mean-variance")
    parser.add_argument("--risk-parameter", type=float, default=1.0)
    parser.add_argument("--measure", choices=("P", "Q"), default="Q")
    parser.add_argument("--position-cap", type=float, default=1_000.0)
    parser.add_argument("--spot-cost-bps", type=float, default=0.0)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--market-artifact-root", type=Path)
    parser.add_argument("--market-artifact-range-summary", type=Path)
    parser.add_argument("--architecture-selection", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-checkpoint-sha256")
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--warm-start-checkpoint-sha256")
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
    parser.add_argument("--target-notional-artifact-sha256")
    parser.add_argument("--target-notional-artifact", type=Path)
    parser.add_argument("--historical-weekly-ridge-selection", type=Path)
    parser.add_argument("--historical-weekly-ridge-selection-sha256")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--allow-cap-hits", action="store_true")
    arguments = parser.parse_args()
    if arguments.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if arguments.target_notional_artifact_sha256 is None:
        parser.error(
            "--target-notional-artifact-sha256 is required; run the CPU weekly "
            "input preflight first"
        )
    if arguments.model == "historical-sanos" and (
        arguments.historical_weekly_ridge_selection is None
        or arguments.historical_weekly_ridge_selection_sha256 is None
    ):
        parser.error(
            "historical-sanos requires the authenticated weekly ridge selection"
        )

    panel = PublicHedgingPanel(arguments.dataset, validate=True)
    notional_config = WeeklyTargetNotionalConfig()
    target_notionals = _load_target_notionals(
        panel,
        config=notional_config,
        artifact_path=arguments.target_notional_artifact,
        expected_artifact_sha256=arguments.target_notional_artifact_sha256,
    )
    claim_quantities = target_notionals.quantity_by_label
    manifest_sha256 = dataset_manifest_sha256(panel)
    if (
        arguments.dataset_manifest_sha256 is not None
        and arguments.dataset_manifest_sha256 != manifest_sha256
    ):
        raise ValueError("public dataset manifest changed after matrix construction")
    checkpoint_directory = _checkpoint_directory(arguments)
    build = build_weekly_adapter(
        arguments,
        checkpoint_directory=checkpoint_directory,
        dataset_manifest_sha256=manifest_sha256,
    )
    schedule = HedgeSchedule(arguments.schedule)
    targets = weekly_targets()
    spec = WeeklyEvaluationSpec(
        label=arguments.label or f"{arguments.model}-{arguments.schedule}",
        model=build.adapter.model,
        measure=build.adapter.measure,
        model_config_identity=build.adapter.model_config_identity,
        policy_config_identity=build.adapter.policy_config_identity,
        configuration_snapshot_json=build.configuration_snapshot_json,
        dataset_manifest_sha256=manifest_sha256,
        schedule=schedule,
        evaluation_start=arguments.start,
        evaluation_end=arguments.end,
        rolling_observations=build.rolling_observations,
        target_labels=tuple(target.label for target in targets),
        target_quantities=tuple(
            claim_quantities[target.label] for target in targets
        ),
        target_notional_identity=(
            f"{type(notional_config).version.unique_id48}:"
            f"{weekly_target_notional_calibration_sha256(target_notionals)}:"
            f"{arguments.target_notional_artifact_sha256}"
        ),
        position_cap=build.position_cap,
        spot_cost_bps=arguments.spot_cost_bps,
        require_no_cap_hits=not arguments.allow_cap_hits,
    )
    output = arguments.output or _default_output(arguments.model, spec)

    def claim_progress(
        origin_date: date,
        index: int,
        total: int,
        target_label: str,
    ) -> None:
        print(
            f"weekly target progress origin={origin_date} "
            f"completed_work_units={index} total_work_units={total} "
            f"target={target_label}",
            flush=True,
        )

    run_book = weekly_book_function(
        adapter=build.adapter,
        schedule=schedule,
        rolling_observations=build.rolling_observations,
        claim_quantities=claim_quantities,
        spot_cost_bps=arguments.spot_cost_bps,
        claim_progress=claim_progress,
    )
    started = time.monotonic()
    origin_count = len(eligible_weekly_origins(panel, spec=spec))
    if arguments.stop_after is not None:
        origin_count = min(origin_count, arguments.stop_after)

    def progress(index: int, total: int, origin_date: date, reused: bool) -> None:
        heartbeat.update(index, last_unit=origin_date.isoformat())
        if index != total and index % arguments.progress_every != 0:
            return
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed > 0.0 else 0.0
        eta = (total - index) / rate if rate > 0.0 else float("nan")
        source = "reused" if reused else "computed"
        print(
            f"weekly progress {index}/{total} origin={origin_date} {source} "
            f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
            flush=True,
        )

    with EvaluationProgressHeartbeat(
        stage=f"{schedule.value}-{spec.model}",
        total_work_units=origin_count,
        interval_seconds=arguments.training_progress_seconds,
    ) as heartbeat:
        summary = run_weekly_evaluation(
            panel,
            spec=spec,
            run_book=run_book,
            output_directory=output,
            progress=progress,
            stop_after=arguments.stop_after,
        )
    if _can_summarize_weekly_run(summary):
        aggregation_started = time.monotonic()

        def aggregation_progress(
            index: int,
            total: int,
            origin_date: date,
        ) -> None:
            if index != total and index % arguments.progress_every != 0:
                return
            elapsed = time.monotonic() - aggregation_started
            rate = index / elapsed if elapsed > 0.0 else 0.0
            eta = (total - index) / rate if rate > 0.0 else float("nan")
            print(
                f"weekly aggregation {index}/{total} origin={origin_date} "
                f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                flush=True,
            )

        results = summarize_completed_weekly_run(
            panel,
            run_directory=output,
            progress=aggregation_progress,
        )
        results.write_json(output / "results.json")
    elif summary.completed_origins == summary.eligible_origins:
        print(
            "weekly aggregation skipped completed_origins=1 "
            "minimum_completed_origins=2",
            flush=True,
        )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), flush=True)


def _can_summarize_weekly_run(summary: WeeklyEvaluationSummary) -> bool:
    """Require at least two complete origins for statistical aggregation."""
    return (
        summary.completed_origins == summary.eligible_origins
        and len(summary.completed_origins) >= 2
    )


def weekly_book_function(
    *,
    adapter: WeeklyModelAdapter,
    schedule: HedgeSchedule,
    rolling_observations: int,
    claim_quantities: dict[str, float],
    spot_cost_bps: float,
    claim_progress: Callable[[date, int, int, str], None] | None = None,
) -> WeeklyBookFunction:
    """Create one complete-book function with bounded cross-origin fit reuse."""
    common_price_config = SanosLvConfig(label="weekly-common-market-pricing")
    market_fit_cache: dict[str, WeeklyMarketFit] = {}

    def run_book(
        panel: PublicHedgingPanel,
        origin_date: date,
    ) -> tuple[WeeklyModelRun, ...]:
        episode = build_weekly_episode_path(
            panel,
            origin_date=origin_date,
            schedule=schedule,
            rolling_window_business_days=rolling_observations,
        )
        claims = build_common_weekly_claims(
            episode.market_days[0],
            config=common_price_config,
        )
        policy_fit_cache = {}
        try:
            completed_runs: list[WeeklyModelRun] = []
            for claim_index, claim in enumerate(claims, start=1):
                completed_runs.append(
                    run_weekly_model(
                        panel,
                        episode=episode,
                        claim=claim,
                        adapter=adapter,
                        claim_quantity=claim_quantities[claim.target.label],
                        spot_cost_bps=spot_cost_bps,
                        market_fit_cache=market_fit_cache,
                        policy_fit_cache=policy_fit_cache,
                    )
                )
                if claim_progress is not None:
                    claim_progress(
                        origin_date,
                        claim_index,
                        len(claims),
                        claim.target.label,
                    )
            runs = tuple(completed_runs)
        finally:
            release_origin = getattr(adapter, "release_weekly_origin", None)
            if release_origin is not None:
                release_origin()
        while len(market_fit_cache) > 8:
            market_fit_cache.pop(next(iter(market_fit_cache)))
        return runs

    return run_book


def build_weekly_adapter(
    arguments: argparse.Namespace,
    *,
    checkpoint_directory: Path | None,
    dataset_manifest_sha256: str,
) -> WeeklyAdapterBuild:
    """Construct one weekly adapter without fitting or reading future states."""
    label = arguments.label or (
        f"{arguments.model}-{arguments.schedule}-{arguments.objective}-"
        f"lambda-{arguments.risk_parameter:g}"
    )
    if arguments.model == "sanos-lv":
        config = SanosLvConfig(label=label, position_cap=arguments.position_cap)
        adapter = SanosLvWeeklyAdapter(config=config)
        return _adapter_build(
            adapter,
            rolling_observations=config.rolling_factors.window_sessions,
            position_cap=config.position_cap,
            configurations={"model": config},
        )
    if arguments.model == "historical-sanos":
        config, ridge_evidence = _historical_config(
            arguments,
            label=label,
            dataset_manifest_sha256=dataset_manifest_sha256,
            require_weekly_ridge=True,
        )
        adapter = HistoricalSanosWeeklyAdapter(config=config)
        return _adapter_build(
            adapter,
            rolling_observations=config.rolling_observations,
            position_cap=config.position_cap,
            configurations={"model": config, **ridge_evidence},
        )
    if arguments.model == "gromm-q-dlv-vega":
        if arguments.market_artifact_range_summary is None:
            raise ValueError(
                "conventional GROMM requires --market-artifact-range-summary"
            )
        artifact_root = arguments.market_artifact_root
        if artifact_root is None:
            root = os.environ.get("QUANTDATA_ROOT")
            if not root:
                raise ValueError(
                    "conventional GROMM requires --market-artifact-root or "
                    "QUANTDATA_ROOT"
                )
            artifact_root = Path(root) / "training" / "hedging" / "gromm2-rolling"
        range_summary = load_completed_rolling_gromm2_range(
            arguments.market_artifact_range_summary,
            dataset=arguments.dataset,
            dataset_manifest_sha256=dataset_manifest_sha256,
            artifact_root=artifact_root,
        )
        if (
            range_summary.evaluation_start != arguments.start
            or range_summary.evaluation_end != arguments.end
        ):
            raise ValueError(
                "GROMM rolling artifact range does not match evaluation dates"
            )
        rolling_config = RollingGromm2Config()
        gromm_config = GrommConfig(
            label=label,
            measure=GrommMeasure.Q,
            path_count=arguments.scenario_count,
            batch_size=(
                arguments.scenario_count
                if arguments.batch_size is None
                else arguments.batch_size
            ),
            position_cap=arguments.position_cap,
        )
        dlv_config = SanosLvConfig(
            label=f"{label}-eight-factor-dlv",
            position_cap=arguments.position_cap,
        )
        adapter = GrommConventionalWeeklyAdapter(
            artifact_root=artifact_root,
            gromm_config=gromm_config,
            dlv_config=dlv_config,
            rolling_config=rolling_config,
        )
        return _adapter_build(
            adapter,
            rolling_observations=rolling_config.window_transitions,
            position_cap=dlv_config.position_cap,
            configurations={
                "gromm": gromm_config,
                "dlv_hedge": dlv_config,
                "rolling": asdict(rolling_config),
                "artifact_range_summary_sha256": file_sha256(
                    arguments.market_artifact_range_summary
                ),
                "artifact_root": artifact_root,
            },
        )
    if arguments.model == "gromm":
        if checkpoint_directory is None:
            raise ValueError("GROMM requires a policy checkpoint directory")
        if arguments.architecture_selection is None:
            raise ValueError("GROMM requires --architecture-selection")
        if arguments.market_artifact_range_summary is None:
            raise ValueError("GROMM requires --market-artifact-range-summary")
        artifact_root = arguments.market_artifact_root
        if artifact_root is None:
            root = os.environ.get("QUANTDATA_ROOT")
            if not root:
                raise ValueError(
                    "GROMM requires --market-artifact-root or QUANTDATA_ROOT"
                )
            artifact_root = Path(root) / "training" / "hedging" / "gromm2-rolling"
        range_summary = load_completed_rolling_gromm2_range(
            arguments.market_artifact_range_summary,
            dataset=arguments.dataset,
            dataset_manifest_sha256=dataset_manifest_sha256,
            artifact_root=artifact_root,
        )
        if (
            range_summary.evaluation_start != arguments.start
            or range_summary.evaluation_end != arguments.end
        ):
            raise ValueError(
                "GROMM rolling artifact range does not match evaluation dates"
            )
        if (
            arguments.measure == "P"
            and range_summary.physical_accepted_count
            != range_summary.eligible_origin_count
        ):
            raise ValueError(
                "GROMM P evaluation requires an accepted physical fit at every origin"
            )
        selection = load_gromm_architecture_selection(
            arguments.architecture_selection,
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
        )
        model_config = GrommConfig(
            label=label,
            measure=GrommMeasure(arguments.measure),
            objective=GrommObjective(arguments.objective),
            risk_parameter=arguments.risk_parameter,
            path_count=arguments.scenario_count,
            batch_size=(
                arguments.scenario_count
                if arguments.batch_size is None
                else arguments.batch_size
            ),
            position_cap=arguments.position_cap,
        )
        train_config = _gromm_train_config(arguments)
        rolling_config = RollingGromm2Config()
        policy_template = selection.architecture.policy_config(
            label=selection.selected_label,
            feature_count=selection.training_feature_count,
            instrument_count=selection.training_instrument_count,
            position_cap=model_config.position_cap,
        )
        environment = BaseEnvironment(
            precision=64,
            device=arguments.device,
            debug=False,
        )
        adapter = GrommWeeklyAdapter(
            artifact_root=artifact_root,
            checkpoint_directory=checkpoint_directory,
            config=model_config,
            rolling_config=rolling_config,
            train_config=train_config,
            policy_template=policy_template,
            environment=environment,
        )
        return _adapter_build(
            adapter,
            rolling_observations=rolling_config.window_transitions,
            position_cap=model_config.position_cap,
            configurations={
                "model": model_config,
                "rolling": rolling_config,
                "training": train_config,
                "policy_template": policy_template,
                "architecture_selection_sha256": selection.sha256,
                "artifact_range_summary_sha256": file_sha256(
                    arguments.market_artifact_range_summary
                ),
                "artifact_root": artifact_root,
                "environment": _environment_payload(environment),
            },
        )
    if checkpoint_directory is None:
        raise ValueError(f"{arguments.model} requires a checkpoint directory")
    policy_hedge, _ = _historical_config(
        arguments,
        label=f"{label}-weekly-policy",
        dataset_manifest_sha256=dataset_manifest_sha256,
        require_weekly_ridge=False,
    )
    policy_config = GeneratedWeeklyPolicyConfig(
        scenario_count=arguments.scenario_count,
        hedge=policy_hedge,
    )
    if arguments.model == "volgan":
        grid = (
                GenerativeSurfaceGrid.short_end_nms()
                if getattr(arguments, "surface_grid", "published") == "short-end-nms"
                else GenerativeSurfaceGrid()
        )
        model_config = (
            VolganModelConfig(label=getattr(arguments, "model_label", None) or label, grid=grid)
            if getattr(arguments, "volgan_pca_rank", None) is None
            else VolganPcaModelConfig(
                label=getattr(arguments, "model_label", None) or label,
                grid=grid,
                pca_rank=arguments.volgan_pca_rank,
            )
        )
        train_config = _volgan_train_config(arguments)
        warm_start_sha256 = _authenticated_file_fingerprint(
            arguments.warm_start_checkpoint,
            expected_sha256=getattr(
                arguments,
                "warm_start_checkpoint_sha256",
                None,
            ),
            label="warm-start checkpoint",
        )
        environment = BaseEnvironment(
            precision=64 if arguments.precision is None else arguments.precision,
            device=arguments.device,
            debug=False,
        )
        adapter = VolganWeeklyAdapter(
            model_config=model_config,
            checkpoint_directory=checkpoint_directory,
            train_config=train_config,
            policy_config=policy_config,
            observation_count=arguments.observation_count,
            warm_start_checkpoint=arguments.warm_start_checkpoint,
            expected_warm_start_sha256=warm_start_sha256,
            environment=environment,
        )
        return _adapter_build(
            adapter,
            rolling_observations=arguments.observation_count,
            position_cap=policy_hedge.position_cap,
            configurations={
                "model": model_config,
                "training": train_config,
                "policy": policy_config,
                "warm_start_checkpoint_sha256": warm_start_sha256,
                "environment": _environment_payload(environment),
            },
        )
    variant = DiffusionVariant(arguments.model.removeprefix("diffusion-"))
    if variant is DiffusionVariant.AD_SEQ_VOL_FT and arguments.base_checkpoint is None:
        raise ValueError("fine-tuned diffusion requires --base-checkpoint")
    if variant is DiffusionVariant.AD_SEQ_VOL and arguments.base_checkpoint is not None:
        raise ValueError("base checkpoint is only valid for fine-tuned diffusion")
    model_config = DiffusionModelConfig(
        label=getattr(arguments, "model_label", None) or label,
        variant=variant,
        grid=(
            GenerativeSurfaceGrid.short_end_nms()
            if getattr(arguments, "surface_grid", "published") == "short-end-nms"
            else GenerativeSurfaceGrid()
        ),
    )
    train_config = _diffusion_train_config(arguments)
    base_sha256 = _authenticated_file_fingerprint(
        arguments.base_checkpoint,
        expected_sha256=getattr(arguments, "base_checkpoint_sha256", None),
        label="base checkpoint",
    )
    warm_start_sha256 = _authenticated_file_fingerprint(
        arguments.warm_start_checkpoint,
        expected_sha256=getattr(
            arguments,
            "warm_start_checkpoint_sha256",
            None,
        ),
        label="warm-start checkpoint",
    )
    precision = 32 if arguments.precision is None else arguments.precision
    environment = BaseEnvironment(
        precision=precision,
        device=arguments.device,
        debug=False,
    )
    adapter = DiffusionWeeklyAdapter(
        model_config=model_config,
        checkpoint_directory=checkpoint_directory,
        train_config=train_config,
        policy_config=policy_config,
        observation_count=arguments.observation_count,
        base_checkpoint=arguments.base_checkpoint,
        warm_start_checkpoint=arguments.warm_start_checkpoint,
        expected_base_checkpoint_sha256=base_sha256,
        expected_warm_start_sha256=warm_start_sha256,
        environment=environment,
    )
    return _adapter_build(
        adapter,
        rolling_observations=arguments.observation_count,
        position_cap=policy_hedge.position_cap,
        configurations={
            "model": model_config,
            "training": train_config,
            "policy": policy_config,
            "base_checkpoint_sha256": base_sha256,
            "warm_start_checkpoint_sha256": warm_start_sha256,
            "environment": _environment_payload(environment),
        },
    )


def _historical_config(
    arguments: argparse.Namespace,
    *,
    label: str,
    dataset_manifest_sha256: str,
    require_weekly_ridge: bool,
) -> tuple[HistoricalSanosConfig, dict[str, object]]:
    ridge = HistoricalSanosConfig().quadratic_ridge
    force_null_policy = False
    evidence: dict[str, object] = {}
    selection_path = getattr(arguments, "historical_weekly_ridge_selection", None)
    selection_sha256 = getattr(
        arguments,
        "historical_weekly_ridge_selection_sha256",
        None,
    )
    if require_weekly_ridge:
        if selection_path is None or selection_sha256 is None:
            raise ValueError(
                "historical weekly policy requires an authenticated ridge selection"
            )
        target_sha256 = getattr(
            arguments,
            "target_notional_artifact_sha256",
            None,
        )
        if target_sha256 is None:
            raise ValueError("historical weekly ridge requires target-notional identity")
        selection = load_weekly_ridge_for_scope(
            selection_path,
            expected_sha256=selection_sha256,
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
            expected_target_notional_artifact_sha256=target_sha256,
            expected_source_commit=repository_source_commit(),
            expected_position_cap=arguments.position_cap,
            expected_objective=HistoricalSanosObjective(arguments.objective),
            expected_risk_parameter=arguments.risk_parameter,
        )
        force_null_policy = selection.selected_ridge is None
        if selection.selected_ridge is not None:
            ridge = selection.selected_ridge
        evidence = {
            "weekly_ridge_selection_file_sha256": selection_sha256,
            "weekly_ridge_selection_sha256": selection.sha256,
            "weekly_ridge_selected_policy": (
                "null-baseline" if force_null_policy else "ridge"
            ),
        }
    return HistoricalSanosConfig(
        label=label,
        rolling_observations=arguments.observation_count,
        objective=HistoricalSanosObjective(arguments.objective),
        risk_parameter=arguments.risk_parameter,
        position_cap=arguments.position_cap,
        quadratic_ridge=ridge,
        force_null_policy=force_null_policy,
    ), evidence


def _gromm_train_config(arguments: argparse.Namespace) -> GrommTrainConfig:
    defaults = GrommTrainConfig()
    maximum_epochs = defaults.maximum_epochs if arguments.epochs is None else arguments.epochs
    return GrommTrainConfig(
        maximum_epochs=maximum_epochs,
        minimum_epochs=min(defaults.minimum_epochs, maximum_epochs),
        patience_epochs=defaults.patience_epochs,
        batch_pairs=defaults.batch_pairs,
        learning_rate=defaults.learning_rate,
        weight_decay=defaults.weight_decay,
        validation_fraction=defaults.validation_fraction,
        minimum_relative_improvement=defaults.minimum_relative_improvement,
        seed=defaults.seed,
        progress_seconds=arguments.training_progress_seconds,
        checkpoint_epochs=(
            defaults.checkpoint_epochs
            if arguments.checkpoint_epochs is None
            else arguments.checkpoint_epochs
        ),
        compile_policy=arguments.compile_model,
        just_load=arguments.just_load,
    )


def _volgan_train_config(arguments: argparse.Namespace) -> VolganTrainConfig:
    defaults = VolganTrainConfig()
    epochs = defaults.epochs if arguments.epochs is None else arguments.epochs
    return VolganTrainConfig(
        epochs=epochs,
        gradient_matching_epochs=(
            0
            if arguments.warm_start_checkpoint is not None
            else min(defaults.gradient_matching_epochs, epochs)
        ),
        batch_size=(
            defaults.batch_size if arguments.batch_size is None else arguments.batch_size
        ),
        generator_learning_rate=defaults.generator_learning_rate,
        discriminator_learning_rate=defaults.discriminator_learning_rate,
        validation_fraction=defaults.validation_fraction,
        seed=defaults.seed,
        progress_seconds=arguments.training_progress_seconds,
        checkpoint_epochs=(
            defaults.checkpoint_epochs
            if arguments.checkpoint_epochs is None
            else arguments.checkpoint_epochs
        ),
        compile_models=arguments.compile_model,
        just_load=arguments.just_load,
    )


def _diffusion_train_config(arguments: argparse.Namespace) -> DiffusionTrainConfig:
    defaults = DiffusionTrainConfig()
    return DiffusionTrainConfig(
        epochs=defaults.epochs if arguments.epochs is None else arguments.epochs,
        batch_size=(
            defaults.batch_size if arguments.batch_size is None else arguments.batch_size
        ),
        learning_rate=defaults.learning_rate,
        weight_decay=defaults.weight_decay,
        warmup_steps=(
            0
            if arguments.warm_start_checkpoint is not None
            else defaults.warmup_steps
        ),
        ema_decay=defaults.ema_decay,
        validation_fraction=defaults.validation_fraction,
        arbitrage_penalty=defaults.arbitrage_penalty,
        return_broadcast_penalty=defaults.return_broadcast_penalty,
        seed=defaults.seed,
        progress_seconds=arguments.training_progress_seconds,
        checkpoint_epochs=(
            defaults.checkpoint_epochs
            if arguments.checkpoint_epochs is None
            else arguments.checkpoint_epochs
        ),
        compile_model=arguments.compile_model,
        mixed_precision=(
            defaults.mixed_precision
            and not arguments.no_mixed_precision
            and arguments.device == "cuda"
        ),
        just_load=arguments.just_load,
    )


def _adapter_build(
    adapter: WeeklyModelAdapter,
    *,
    rolling_observations: int,
    position_cap: float,
    configurations: dict[str, object],
) -> WeeklyAdapterBuild:
    return WeeklyAdapterBuild(
        adapter=adapter,
        rolling_observations=rolling_observations,
        position_cap=position_cap,
        configuration_snapshot_json=json.dumps(
            _configuration_payload(configurations),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ),
    )


def _configuration_payload(values: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, value in values.items():
        if hasattr(type(value), "version") and hasattr(value, "__dataclass_fields__"):
            payload[name] = {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "code_version": str(type(value).version),
                "fields": asdict(value),
            }
        else:
            payload[name] = value
    common_price = SanosLvConfig(label="weekly-common-market-pricing")
    payload["common_market_pricing"] = {
        "class": f"{type(common_price).__module__}.{type(common_price).__qualname__}",
        "code_version": str(type(common_price).version),
        "fields": asdict(common_price),
    }
    return payload


def _environment_payload(environment: BaseEnvironment) -> dict[str, object]:
    return {
        "precision": str(environment.fdtype),
        "device": str(environment.device),
        "debug": environment.debug,
    }


def _load_target_notionals(
    panel: PublicHedgingPanel,
    *,
    config: WeeklyTargetNotionalConfig,
    artifact_path: Path | None,
    expected_artifact_sha256: str | None,
) -> WeeklyTargetNotionalCalibration:
    if artifact_path is not None:
        if expected_artifact_sha256 is None:
            raise ValueError(
                "--target-notional-artifact requires "
                "--target-notional-artifact-sha256"
            )
        print(
            "loading exact CPU-preflighted weekly target notionals: "
            f"{artifact_path}",
            flush=True,
        )
        return load_authenticated_weekly_target_notionals_from_path(
            panel,
            artifact_path=artifact_path,
            expected_artifact_sha256=expected_artifact_sha256,
            config=config,
        )
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        if expected_artifact_sha256 is not None:
            raise ValueError(
                "--target-notional-artifact-sha256 requires QUANTDATA_ROOT"
            )
        return calibrate_weekly_target_notionals(panel, config=config)
    cache_directory = Path(root) / "processing" / "hedging" / "target-notionals"
    artifact = weekly_target_notional_artifact_path(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    if expected_artifact_sha256 is None:
        source = "loading" if artifact.exists() else "calibrating"
        print(f"{source} fixed 2022 weekly target notionals: {artifact}", flush=True)
        return load_or_calibrate_weekly_target_notionals(
            panel,
            cache_directory=cache_directory,
            config=config,
        )
    print(f"loading CPU-preflighted weekly target notionals: {artifact}", flush=True)
    return load_authenticated_weekly_target_notionals(
        panel,
        cache_directory=cache_directory,
        expected_artifact_sha256=expected_artifact_sha256,
        config=config,
    )


def _checkpoint_directory(arguments: argparse.Namespace) -> Path | None:
    if arguments.checkpoint_directory is not None:
        return arguments.checkpoint_directory
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        return None
    return Path(root) / "training" / "hedging" / "weekly" / arguments.model


def _default_output(model: str, spec: WeeklyEvaluationSpec) -> Path:
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        raise RuntimeError("QUANTDATA_ROOT is required for the default output path")
    return (
        Path(root)
        / "experiments"
        / "hedging"
        / "weekly"
        / spec.schedule.value
        / model
        / spec.run_identity
    )


def _authenticated_file_fingerprint(
    path: Path | None,
    *,
    expected_sha256: str | None,
    label: str,
) -> str | None:
    if path is None:
        if expected_sha256 is not None:
            raise ValueError(f"{label} SHA-256 requires its file path")
        return None
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"{label} changed after production-matrix creation")
    return actual


if __name__ == "__main__":
    main()
