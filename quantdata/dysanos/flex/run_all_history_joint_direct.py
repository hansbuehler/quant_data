"""Warm-start and refit the joint Direct-200 P model through August 2025."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    JointDirectDynamicsConfig,
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    compact_evaluation,
    evaluate_projected_market_days,
    fit_joint_direct_model,
    initialize_joint_model,
    joint_dynamics_diagnostics,
    standardize_affine_direct_model,
)
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.market_encoder import (
    checkpoint_digest,
    write_json_atomic,
)
from quantdata.dysanos.flex.model import (
    JointDirectDynamicsModel,
    LatentStateSanosModel,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _market_dates_from_completion,
)
from quantdata.dysanos.flex.training import (
    FitEvaluation,
    evaluate_model,
    materialize_day,
)
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.torch import BaseEnvironment

_DAILY_STATE_KEY = "daily_state_embedding.weight"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-completion", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-joint-checkpoint", type=Path)
    source.add_argument("--source-direct-decoder-checkpoint", type=Path)
    parser.add_argument("--source-2025-states", type=Path)
    parser.add_argument(
        "--model-business-dtes",
        type=float,
        nargs="+",
        help="Explicit model-expiry grid in business-day units.",
    )
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--day-batch-size", type=int, default=16)
    parser.add_argument("--state-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--decoder-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--dynamics-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--dynamics-weight", type=float, default=0.10)
    parser.add_argument("--gauge-weight", type=float, default=0.10)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--dynamics-ramp-epochs", type=int, default=0)
    return parser.parse_args()


def _state_sha256(states: torch.Tensor) -> str:
    values = states.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _combined_source_sha256(*digests: str) -> str:
    payload = "\n".join(digests).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _split_index(dates: tuple[date, ...]) -> int:
    indices = [index for index, value in enumerate(dates) if value.year >= 2025]
    if not indices:
        raise ValueError("The all-history Direct run requires 2025 market dates")
    split = indices[0]
    if any(value.year >= 2025 for value in dates[:split]) or any(
        value.year < 2025 for value in dates[split:]
    ):
        raise ValueError("Market dates do not have one contiguous 2025 suffix")
    return split


def _market_config_from_completion(
    completion: dict[str, object],
) -> FlexMarketConfig:
    """Restore the certified native run's exact quote-universe configuration."""
    payload = completion.get("market_config")
    if not isinstance(payload, dict):
        raise ValueError("Native completion has no market configuration")
    values = dict(payload)
    excluded = values.get("excluded_business_dtes", ())
    if not isinstance(excluded, list | tuple):
        raise ValueError("Native completion has invalid excluded business DTEs")
    values["excluded_business_dtes"] = tuple(float(value) for value in excluded)
    return FlexMarketConfig(**values)


def _geometry_from_completion(
    completion: dict[str, object],
    *,
    requested_model_business_dtes: tuple[float, ...] | None,
) -> SharedStrikeCappedUnboundedDlvFlexGeometryConfig:
    """Restore the exact completed geometry and validate any CLI assertion."""
    geometry_payload = completion.get("geometry")
    if not isinstance(geometry_payload, dict):
        raise ValueError("Native completion has no geometry payload")
    values = dict(geometry_payload)
    for name in ("nms_range", "explicit_model_business_dtes"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    geometry_type = (
        FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig
        if values.get("front_term_eta") is True
        else SharedStrikeCappedUnboundedDlvFlexGeometryConfig
    )
    geometry = geometry_type(**values)
    if requested_model_business_dtes is not None and (
        requested_model_business_dtes
        != tuple(float(value) for value in geometry.model_business_dtes)
    ):
        raise ValueError(
            "Requested model expiry grid differs from the completed native "
            f"geometry: {requested_model_business_dtes} versus "
            f"{tuple(float(value) for value in geometry.model_business_dtes)}"
        )
    return geometry


def install_all_history_warm_start(
    model: JointDirectDynamicsModel,
    *,
    source_joint_payload: dict[str, object],
    source_joint_sha256: str,
    source_2025_payload: dict[str, object],
    dates: tuple[date, ...],
) -> dict[str, object]:
    """Install accepted Direct weights and append quote-optimal 2025 states."""
    if source_joint_payload.get("schema") != "flex-joint-direct-dynamics-v1":
        raise ValueError("Source joint Direct checkpoint has an unsupported schema")
    if source_joint_payload.get("phase") != "complete":
        raise ValueError("Source joint Direct checkpoint is not complete")
    if source_2025_payload.get("schema") != "flex-direct-2025-quote-capacity-states-v1":
        raise ValueError("Source 2025 Direct states have an unsupported schema")
    if source_2025_payload.get("source_joint_checkpoint_sha256") != (source_joint_sha256):
        raise ValueError("Source 2025 states do not bind to the joint checkpoint")

    split = _split_index(dates)
    validation_dates = cast(list[str], source_2025_payload.get("validation_dates"))
    expected_validation_dates = [value.isoformat() for value in dates[split:]]
    if validation_dates != expected_validation_dates:
        raise ValueError("Source 2025 state dates do not match the market universe")

    source_state_dict = cast(dict[str, torch.Tensor], source_joint_payload.get("model_state_dict"))
    if not isinstance(source_state_dict, dict):
        raise ValueError("Source joint Direct checkpoint has no model state")
    source_training_states = source_state_dict.get(_DAILY_STATE_KEY)
    source_2025_states = source_2025_payload.get("quote_optimal_states")
    if not isinstance(source_training_states, torch.Tensor) or not isinstance(
        source_2025_states, torch.Tensor
    ):
        raise ValueError("Direct warm-start state tensors are missing")
    expected_training_shape = (split, model.config.state_size)
    expected_2025_shape = (len(dates) - split, model.config.state_size)
    if tuple(source_training_states.shape) != expected_training_shape:
        raise ValueError(
            "Source joint state shape mismatch: "
            f"{tuple(source_training_states.shape)} != {expected_training_shape}"
        )
    if tuple(source_2025_states.shape) != expected_2025_shape:
        raise ValueError(
            "Source 2025 state shape mismatch: "
            f"{tuple(source_2025_states.shape)} != {expected_2025_shape}"
        )

    target_state_dict = model.state_dict()
    copied_state_dict: dict[str, torch.Tensor] = {}
    for name, target in target_state_dict.items():
        if name == _DAILY_STATE_KEY:
            continue
        source = source_state_dict.get(name)
        if source is None or source.shape != target.shape:
            raise ValueError(f"Source joint parameter {name!r} is missing or changed")
        copied_state_dict[name] = source
    missing, unexpected = model.load_state_dict(copied_state_dict, strict=False)
    if missing != [_DAILY_STATE_KEY] or unexpected:
        raise RuntimeError(
            f"Unexpected Direct warm-start keys: missing={missing}, unexpected={unexpected}"
        )

    states = torch.cat((source_training_states.detach(), source_2025_states.detach()), dim=0)
    if not bool(torch.all(torch.isfinite(states))):
        raise ValueError("Direct warm-start states contain non-finite values")
    with torch.no_grad():
        model.daily_states.copy_(states.to(dtype=model.fdtype, device=model.device))

    installed = model.daily_states.detach().cpu().double()
    reference = states.cpu().double()
    return {
        "mode": "accepted_joint_plus_quote_optimal_2025_states",
        "num_2022_2024_states": split,
        "num_2025_states": len(dates) - split,
        "maximum_state_roundoff": float(torch.max(torch.abs(installed - reference))),
        "source_2022_2024_state_sha256": _state_sha256(source_training_states),
        "source_2025_state_sha256": _state_sha256(source_2025_states),
    }


def install_direct_decoder_initialization(
    model: JointDirectDynamicsModel,
    *,
    source_payload: dict[str, object],
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    days: tuple[FlexMarketDay, ...],
    dates: tuple[date, ...],
    env: BaseEnvironment,
) -> dict[str, object]:
    """Whiten one complete all-history affine decoder and initialize dynamics."""
    if source_payload.get("schema_version") != 2 or source_payload.get("phase") != (
        "complete"
    ):
        raise ValueError("Source all-history Direct decoder is not complete")
    audits = source_payload.get("training_audits")
    if not isinstance(audits, dict) or not all(
        isinstance(audits.get(name), dict)
        and audits[name].get("converged") is True
        for name in ("supervised", "market")
    ):
        raise ValueError("Source all-history Direct decoder has no convergence proof")
    source_fingerprints = source_payload.get("market_fingerprints")
    if not isinstance(source_fingerprints, (list, tuple)) or tuple(
        source_fingerprints
    ) != tuple(day.training_fingerprint for day in days):
        raise ValueError("Source all-history Direct markets changed")
    identity = source_payload.get("model_identity")
    if not isinstance(identity, dict):
        raise ValueError("Source all-history Direct model identity is missing")
    config_values = identity.get("model_config")
    if not isinstance(config_values, dict):
        raise ValueError("Source all-history Direct model config is missing")
    if config_values.get("geometry") != asdict(geometry):
        raise ValueError("Source all-history Direct geometry changed")
    if config_values.get("state_size") != model.config.state_size:
        raise ValueError("Source all-history Direct state size changed")
    label = config_values.get("label")
    if not isinstance(label, str):
        raise ValueError("Source all-history Direct label is missing")

    source = LatentStateSanosModel(
        env,
        num_days=len(days),
        config=PcaDecoderConfig(
            label=label,
            state_size=model.config.state_size,
            geometry=geometry,
        ),
    )
    source_state = source_payload.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("Source all-history Direct model state is missing")
    source.load_state_dict(cast(dict[str, torch.Tensor], source_state))
    standardized = standardize_affine_direct_model(source)
    origin, endpoint = exact_weekly_pairs(dates, weeks=1)
    dynamics = initialize_joint_model(
        model,
        standardized,
        origin_indices=origin,
        endpoint_indices=endpoint,
    )
    return {
        "mode": "complete_all_history_affine_direct_decoder",
        "num_states": len(days),
        "maximum_raw_reconstruction_error": (
            standardized.maximum_raw_reconstruction_error
        ),
        "minimum_factor_scale": standardized.minimum_factor_scale,
        "maximum_factor_scale": standardized.maximum_factor_scale,
        "dynamics": dynamics,
    }


def _evaluate_states(
    model: JointDirectDynamicsModel,
    states: torch.Tensor,
    days: tuple[FlexMarketDay, ...],
    *,
    objective_config: FlexObjectiveConfig,
    train_config: JointDirectTrainConfig,
    env: BaseEnvironment,
) -> FitEvaluation:
    return evaluate_projected_market_days(
        model,
        states,
        days,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        env=env,
    )


def interpolation_region_excess_statistics(
    evaluation: FitEvaluation,
    days: tuple[FlexMarketDay, ...],
    *,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
) -> dict[str, int | float]:
    """Summarize spread excess only between the first and last model expiries."""
    if len(evaluation.fits) != len(days):
        raise ValueError("Evaluation fits and market days are not aligned")
    fits = np.concatenate(evaluation.fits)
    bids = np.concatenate([day.bids for day in days])
    asks = np.concatenate([day.asks for day in days])
    business_dtes = np.concatenate([day.quote_business_dtes for day in days])
    selected = (business_dtes >= geometry.model_business_dtes[0] - 1.0e-12) & (
        business_dtes <= geometry.model_business_dtes[-1] + 1.0e-12
    )
    if not np.any(selected):
        raise RuntimeError("No market quotes lie in the model interpolation region")
    half_spread = 0.5 * (asks[selected] - bids[selected])
    excess = (
        np.maximum(fits[selected] - asks[selected], 0.0)
        + np.maximum(bids[selected] - fits[selected], 0.0)
    ) / half_spread
    positive = excess[excess > 0.0]
    return {
        "first_model_business_dte": float(geometry.model_business_dtes[0]),
        "last_model_business_dte": float(geometry.model_business_dtes[-1]),
        "num_options": int(len(excess)),
        "num_positive": int(len(positive)),
        "percent_zero": float(100.0 * np.mean(excess == 0.0)),
        "mean": float(np.mean(excess)),
        "median": float(np.median(excess)),
        "positive_mean": float(np.mean(positive)) if len(positive) else 0.0,
        "positive_median": float(np.median(positive)) if len(positive) else 0.0,
    }


def run(arguments: argparse.Namespace) -> None:
    """Refit the accepted affine Direct-200 model on every 2022--2025 date."""
    if arguments.ticker != "SPX":
        raise ValueError("The all-history Direct experiment is specified for SPX")
    if arguments.device != "cuda:0":
        raise ValueError("All-history Direct training is restricted to physical GPU 0")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    if (arguments.source_joint_checkpoint is None) != (
        arguments.source_2025_states is None
    ):
        raise ValueError(
            "The accepted joint warm start requires both the joint checkpoint "
            "and the 2025 state artifact"
        )
    source_paths = [arguments.native_completion]
    source_paths.extend(
        path
        for path in (
            arguments.source_joint_checkpoint,
            arguments.source_2025_states,
            arguments.source_direct_decoder_checkpoint,
        )
        if path is not None
    )
    for source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(source_path)

    completion = json.loads(arguments.native_completion.read_text(encoding="utf-8"))
    if completion.get("status") != "native_complete":
        raise ValueError("Native shared-strikes completion proof is incomplete")
    market_config = _market_config_from_completion(completion)

    geometry = _geometry_from_completion(
        completion,
        requested_model_business_dtes=(
            None
            if arguments.model_business_dtes is None
            else tuple(arguments.model_business_dtes)
        ),
    )
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    train_config = JointDirectTrainConfig(
        num_epochs=arguments.epochs,
        day_batch_size=arguments.day_batch_size,
        state_learning_rate=arguments.state_learning_rate,
        decoder_learning_rate=arguments.decoder_learning_rate,
        dynamics_learning_rate=arguments.dynamics_learning_rate,
        dynamics_weight=arguments.dynamics_weight,
        gauge_weight=arguments.gauge_weight,
        warmup_epochs=arguments.warmup_epochs,
        dynamics_ramp_epochs=arguments.dynamics_ramp_epochs,
        seed=9281,
    )
    model_config = JointDirectDynamicsConfig(
        label=(
            "shared-strikes-custom-through-2025-joint-direct-200"
            if arguments.model_business_dtes is not None
            else "shared-strikes-through-2025-joint-direct-200"
        ),
        state_size=200,
        geometry=geometry,
    )
    dates = _market_dates_from_completion(arguments.native_completion)
    days = _load_markets(
        ticker=arguments.ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        workers=arguments.workers,
    )
    if tuple(day.trading_date for day in days) != dates:
        raise RuntimeError("Loaded all-history markets changed order")
    split = _split_index(dates)

    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    model = JointDirectDynamicsModel(
        env,
        num_days=len(days),
        config=model_config,
    )
    source_artifacts: dict[str, object]
    if arguments.source_direct_decoder_checkpoint is not None:
        source_direct_sha256 = checkpoint_digest(
            arguments.source_direct_decoder_checkpoint
        )
        source_direct_payload = torch.load(
            arguments.source_direct_decoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        initialization = install_direct_decoder_initialization(
            model,
            source_payload=source_direct_payload,
            geometry=geometry,
            days=days,
            dates=dates,
            env=env,
        )
        source_initialization_sha256 = source_direct_sha256
        source_artifacts = {
            "direct_decoder_checkpoint": str(
                arguments.source_direct_decoder_checkpoint
            ),
            "direct_decoder_checkpoint_sha256": source_direct_sha256,
        }
    else:
        assert arguments.source_joint_checkpoint is not None
        assert arguments.source_2025_states is not None
        source_joint_sha256 = checkpoint_digest(arguments.source_joint_checkpoint)
        source_2025_sha256 = checkpoint_digest(arguments.source_2025_states)
        source_joint_payload = torch.load(
            arguments.source_joint_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        source_2025_payload = torch.load(
            arguments.source_2025_states,
            map_location="cpu",
            weights_only=False,
        )
        initialization = install_all_history_warm_start(
            model,
            source_joint_payload=source_joint_payload,
            source_joint_sha256=source_joint_sha256,
            source_2025_payload=source_2025_payload,
            dates=dates,
        )
        source_initialization_sha256 = _combined_source_sha256(
            source_joint_sha256, source_2025_sha256
        )
        source_artifacts = {
            "joint_checkpoint": str(arguments.source_joint_checkpoint),
            "joint_checkpoint_sha256": source_joint_sha256,
            "states_2025": str(arguments.source_2025_states),
            "states_2025_sha256": source_2025_sha256,
        }

    torch_days = tuple(materialize_day(day, env=env) for day in days)
    baseline = evaluate_model(
        model,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        equal_date_weighting=True,
    )
    baseline_states = model.daily_states.detach().cpu().double()
    baseline_pre_2025 = _evaluate_states(
        model,
        baseline_states[:split],
        days[:split],
        objective_config=objective_config,
        train_config=train_config,
        env=env,
    )
    baseline_2025 = _evaluate_states(
        model,
        baseline_states[split:],
        days[split:],
        objective_config=objective_config,
        train_config=train_config,
        env=env,
    )

    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.checkpoint_directory / "joint-direct-200.pt"
    report_path = arguments.report_directory / "joint-direct-200.json"
    completion_path = arguments.report_directory / "joint-direct-200-complete.json"
    lock_path = arguments.checkpoint_directory / "joint-direct-200.lock"
    with FileLock(str(lock_path)):
        history, convergence = fit_joint_direct_model(
            model,
            days,
            dates=dates,
            objective_config=objective_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            source_decoder_sha256=source_initialization_sha256,
            env=env,
        )

    evaluation = evaluate_model(
        model,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        equal_date_weighting=True,
    )
    final_states = model.daily_states.detach().cpu().double()
    evaluation_pre_2025 = _evaluate_states(
        model,
        final_states[:split],
        days[:split],
        objective_config=objective_config,
        train_config=train_config,
        env=env,
    )
    evaluation_2025 = _evaluate_states(
        model,
        final_states[split:],
        days[split:],
        objective_config=objective_config,
        train_config=train_config,
        env=env,
    )
    minimum_inside = (
        1.0 - train_config.maximum_quote_containment_ratio_drop
    ) * baseline.statistics.all.percent_zero
    if evaluation.statistics.all.percent_zero < minimum_inside:
        raise RuntimeError(
            "All-history Direct containment regressed beyond its gate: "
            f"{evaluation.statistics.all.percent_zero:.6g}% < {minimum_inside:.6g}%"
        )

    state_values = final_states.numpy()
    dynamics = joint_dynamics_diagnostics(model, state_values, state_values, dates)
    dynamics_pre_2025 = joint_dynamics_diagnostics(
        model,
        state_values,
        state_values[:split],
        dates[:split],
    )
    dynamics_2025 = joint_dynamics_diagnostics(
        model,
        state_values,
        state_values[split:],
        dates[split:],
    )

    target_path = arguments.checkpoint_directory / "joint-direct-200-targets.pt"
    temporary = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(
        {
            "schema": "flex-all-history-joint-direct-targets-v1",
            "dates": [value.isoformat() for value in dates],
            "states": final_states,
            "spot_refs": torch.as_tensor(
                [cast(float, day.spot_ref) for day in days], dtype=torch.float64
            ),
            "joint_checkpoint": str(checkpoint_path),
            "joint_checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "source_artifacts": source_artifacts,
        },
        temporary,
    )
    temporary.replace(target_path)

    report: dict[str, object] = {
        "schema": "flex-all-history-joint-direct-v1",
        "status": "complete",
        "measure": "P",
        "training_semantics": (
            f"All {len(dates)} usable dates through {dates[-1].isoformat()} enter "
            "both the equal-date quote objective and exact-week Gaussian dynamics "
            "objective. The run has no temporal validation holdout."
        ),
        "model_config": asdict(model_config),
        "market_config": asdict(market_config),
        "train_config": asdict(train_config),
        "objective_config": asdict(objective_config),
        "train_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "num_dates": len(dates),
        "num_pre_2025_dates": split,
        "num_2025_dates": len(dates) - split,
        "date_weighting": "mean quotes within each date, then mean dates",
        "initialization": initialization,
        "baseline_all_history_fit": compact_evaluation(baseline),
        "baseline_interpolation_region_fit": (
            interpolation_region_excess_statistics(
                baseline,
                days,
                geometry=geometry,
            )
        ),
        "baseline_pre_2025_fit": compact_evaluation(baseline_pre_2025),
        "baseline_2025_fit": compact_evaluation(baseline_2025),
        "all_history_fit": compact_evaluation(evaluation),
        "interpolation_region_fit": interpolation_region_excess_statistics(
            evaluation,
            days,
            geometry=geometry,
        ),
        "pre_2025_fit": compact_evaluation(evaluation_pre_2025),
        "fit_2025": compact_evaluation(evaluation_2025),
        "pre_2025_interpolation_region_fit": (
            interpolation_region_excess_statistics(
                evaluation_pre_2025,
                days[:split],
                geometry=geometry,
            )
        ),
        "fit_2025_interpolation_region_fit": (
            interpolation_region_excess_statistics(
                evaluation_2025,
                days[split:],
                geometry=geometry,
            )
        ),
        "dynamics_all_history": dynamics,
        "dynamics_pre_2025": dynamics_pre_2025,
        "dynamics_2025": dynamics_2025,
        "convergence": convergence,
        "history_tail": history[-50:],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "target_history": str(target_path),
        "target_history_sha256": checkpoint_digest(target_path),
        "state_sha256": _state_sha256(final_states),
        "source_artifacts": source_artifacts,
        "source_initialization_sha256": source_initialization_sha256,
        "ad_hoc_choices": [
            "affine Direct-200 decoder",
            "fresh optimizer with explicitly reported learning rates",
            (
                f"{train_config.warmup_epochs}-epoch learning-rate warm-up and "
                f"{train_config.dynamics_ramp_epochs}-epoch dynamics ramp"
            ),
            "dissipative K=LL'+S+k_min I parameterization",
            "full Gaussian weekly innovation covariance",
        ],
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-all-history-joint-direct-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        },
        completion_path,
    )
    print(f"all-history joint Direct experiment complete: {report_path}", flush=True)


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_all_history_joint_direct")
