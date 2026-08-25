"""High-path 2025 audit of the fitted scalar W2 conditional-moment law."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.conditional_moment import (
    direct_w2_conditional_moment_transport_loss,
    replicated_conditional_moment_statistics,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_spot_correlation import (
    fit_joint_spot_correlation_under_p,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.oce_transport import (
    SpotConditionedFirstLineTransport,
    SpotMarginalOCEAction,
    apply_first_line_transport_sequence,
    sample_conditional_paths,
    w2_roll_returns,
)
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import _load_inputs
from quantdata.dysanos.flex.run_oce_transport_q1 import (
    _build_kernels,
    _estimate_preconditioner,
    _load_oce_action_warm_start,
    _tensor_sha256,
)
from quantdata.dysanos.flex.run_sequential_oce_transport_w2 import (
    _path_log_spot,
    _source_action_config,
)
from quantdata.dysanos.flex.weekly_measure import (
    JointHierarchicalWeeklyDynamicsAdapter,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--worker-heartbeat", type=Path, default=None)
    parser.add_argument("--generation-token", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replications", type=int, default=8)
    parser.add_argument("--origins-per-batch", type=int, default=2)
    parser.add_argument("--spot-contexts-per-origin", type=int, default=32)
    parser.add_argument("--surface-paths-per-context", type=int, default=256)
    parser.add_argument("--maximum-origins", type=int, default=None)
    parser.add_argument("--heartbeat-every-origins", type=int, default=8)
    parser.add_argument("--report-every-origins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=8_241_913)
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("high-path W2 audit is restricted to GPU0")
    positive = (
        arguments.replications,
        arguments.origins_per_batch,
        arguments.spot_contexts_per_origin,
        arguments.surface_paths_per_context,
        arguments.heartbeat_every_origins,
        arguments.report_every_origins,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("high-path W2 audit controls must be positive")
    if arguments.replications < 2:
        raise ValueError("high-path W2 audit requires at least two replications")
    if (
        arguments.spot_contexts_per_origin % 2
        or arguments.surface_paths_per_context % 2
    ):
        raise ValueError("high-path W2 audit path counts must be even")
    if arguments.maximum_origins is not None and arguments.maximum_origins <= 0:
        raise ValueError("maximum origins must be positive when specified")
    if (arguments.worker_heartbeat is None) != (arguments.generation_token is None):
        raise ValueError("worker heartbeat and generation token must be specified together")


def _json_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(arguments).items()
    }


def _write_heartbeat(
    arguments: argparse.Namespace,
    *,
    completed_origins: int,
    total_origins: int,
    status: str = "RUNNING",
) -> None:
    if arguments.worker_heartbeat is None:
        return
    write_json_atomic(
        {
            "schema": "flex-direct-moment-w2-high-path-worker-heartbeat-v1",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "generation": arguments.generation_token,
            "worker_pid": os.getpid(),
            "status": status,
            "last_progress_value": {
                "phase": "complete" if status == "SUCCEEDED" else "validation-audit",
                "completed_origins": completed_origins,
                "total_origins": total_origins,
            },
        },
        arguments.worker_heartbeat,
    )


def _moment_summary(
    estimates: torch.Tensor,
    standard_errors: torch.Tensor,
) -> dict[str, float]:
    if estimates.shape != standard_errors.shape or estimates.ndim != 2:
        raise ValueError("high-path moment summary has inconsistent geometry")
    scale = torch.clamp_min(
        standard_errors,
        32.0 * torch.finfo(estimates.dtype).eps,
    )
    standardized = torch.abs(estimates) / scale
    excess = torch.clamp_min(torch.abs(estimates) - 3.0 * standard_errors, 0.0)
    return {
        "conditional_moment_signed_mean_bps_spot": float(torch.mean(estimates))
        * 1.0e4,
        "conditional_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(estimates.square()))
        )
        * 1.0e4,
        "conditional_moment_mae_bps_spot": float(torch.mean(torch.abs(estimates)))
        * 1.0e4,
        "conditional_moment_maximum_abs_bps_spot": float(
            torch.max(torch.abs(estimates))
        )
        * 1.0e4,
        "mean_mc_standard_error_bps_spot": float(torch.mean(standard_errors))
        * 1.0e4,
        "median_mc_standard_error_bps_spot": float(torch.median(standard_errors))
        * 1.0e4,
        "maximum_mc_standard_error_bps_spot": float(torch.max(standard_errors))
        * 1.0e4,
        "standardized_residual_rmse": float(
            torch.sqrt(torch.mean(standardized.square()))
        ),
        "standardized_residual_maximum": float(torch.max(standardized)),
        "percent_within_1_mc_se": float(
            100.0 * torch.mean((standardized <= 1.0).to(estimates.dtype))
        ),
        "percent_within_2_mc_se": float(
            100.0 * torch.mean((standardized <= 2.0).to(estimates.dtype))
        ),
        "percent_within_3_mc_se": float(
            100.0 * torch.mean((standardized <= 3.0).to(estimates.dtype))
        ),
        "excess_beyond_3_mc_se_rmse_bps_spot": float(
            torch.sqrt(torch.mean(excess.square()))
        )
        * 1.0e4,
    }


def _validate_candidate_checkpoint(
    path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "flex-direct-moment-transport-w2-checkpoint-v1"
        or payload.get("phase") != "complete"
        or not isinstance(payload.get("arguments"), Mapping)
        or not isinstance(payload.get("model_state_dict"), Mapping)
    ):
        raise ValueError("candidate W2 checkpoint is not a complete fitted transport")
    state = cast(Mapping[str, torch.Tensor], payload["model_state_dict"])
    if not state or any(
        not isinstance(value, torch.Tensor)
        or value.numel() == 0
        or not bool(torch.all(torch.isfinite(value)))
        for value in state.values()
    ):
        raise ValueError("candidate W2 transport state is empty or non-finite")
    return payload, cast(Mapping[str, Any], payload["arguments"])


@torch.no_grad()
def run(arguments: argparse.Namespace) -> None:
    """Audit the fixed candidate on independent replicated 2025 paths."""

    _validate_arguments(arguments)
    candidate_payload, candidate_arguments = _validate_candidate_checkpoint(
        arguments.candidate_checkpoint
    )
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    joint_completion = Path(str(candidate_arguments["joint_completion"]))
    source_checkpoint = Path(str(candidate_arguments["source_oce_checkpoint"]))
    (
        model,
        train_dates,
        train_states,
        train_log_spot,
        validation_dates,
        validation_states,
        validation_log_spot,
        _,
    ) = _load_inputs(joint_completion, env=env)
    if _tensor_sha256(train_states) != candidate_payload.get("train_state_sha256"):
        raise ValueError("candidate W2 checkpoint training-state provenance changed")
    if checkpoint_digest(joint_completion) != candidate_payload.get(
        "joint_completion_sha256"
    ):
        raise ValueError("candidate W2 joint-completion provenance changed")
    if checkpoint_digest(source_checkpoint) != candidate_payload.get(
        "source_oce_checkpoint_sha256"
    ):
        raise ValueError("candidate W2 source-law provenance changed")

    log_spot_mean = float(np.mean(train_log_spot))
    log_spot_std = float(np.std(train_log_spot, ddof=0))
    spot_fit = fit_joint_spot_correlation_under_p(
        model,
        train_states.detach().cpu().double().numpy(),
        train_log_spot,
        train_dates,
    )
    dynamics = JointHierarchicalWeeklyDynamicsAdapter(
        env,
        model=model,
        spot_front_beta=torch.as_tensor(
            spot_fit.front_beta,
            dtype=env.fdtype,
            device=env.device,
        ),
        spot_orthogonal_std=spot_fit.orthogonal_std,
    )
    simulator = WeeklySanosSimulator(env, dynamics=dynamics, calibrated_model=model)
    measure_config, p_reference, source_transport = _build_kernels(env, dynamics)
    if asdict(measure_config) != candidate_payload.get("measure_config"):
        raise ValueError("candidate W2 measure geometry changed")

    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, Mapping):
        raise ValueError("source OCE checkpoint is not a mapping")
    source_config = _source_action_config(source_payload)
    source_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    source_train_data = precompute_measure_training_data(
        simulator,
        train_states,
        source_strikes,
        num_expiries=2,
    )
    source_generator = torch.Generator(device=env.device)
    source_generator.manual_seed(
        int(cast(Mapping[str, Any], source_payload["arguments"])["seed"])
    )
    source_return_rms, source_inverse_sqrt = _estimate_preconditioner(
        p_reference,
        simulator,
        source_train_data,
        source_strikes,
        generator=source_generator,
        minimum_return_rms=float(candidate_arguments["minimum_return_rms"]),
    )
    source_action = SpotMarginalOCEAction(
        env,
        config=source_config,
        inverse_sqrt_second_moment=source_inverse_sqrt,
    )
    validated_source = _load_oce_action_warm_start(
        source_checkpoint,
        measure_config=measure_config,
        oce_config=source_config,
        return_rms=source_return_rms,
        inverse_sqrt_second_moment=source_inverse_sqrt,
        train_state_sha256=_tensor_sha256(train_states),
        joint_completion_sha256=checkpoint_digest(joint_completion),
        oce_action=source_action,
    )
    source_transport.load_state_dict(
        cast(Mapping[str, torch.Tensor], validated_source["transport_state_dict"])
    )
    source_transport.eval()

    incremental = SpotConditionedFirstLineTransport(
        env,
        dynamics=dynamics,
        config=measure_config,
        current_log_spot_mean=log_spot_mean,
        current_log_spot_std=log_spot_std,
    )
    incremental.load_state_dict(
        cast(Mapping[str, torch.Tensor], candidate_payload["model_state_dict"])
    )
    incremental.eval()
    normalized_strikes = torch.ones(1, dtype=env.fdtype, device=env.device)
    validation_data = precompute_measure_training_data(
        simulator,
        validation_states,
        normalized_strikes,
        num_expiries=2,
    )
    validation_log_spot_tensor = torch.as_tensor(
        validation_log_spot,
        dtype=env.fdtype,
        device=env.device,
    )
    inverse_sqrt = cast(
        torch.Tensor,
        candidate_payload["inverse_sqrt_second_moment"],
    ).to(dtype=env.fdtype, device=env.device)
    total_origins = min(
        len(validation_states),
        arguments.maximum_origins or len(validation_states),
    )
    _write_heartbeat(
        arguments,
        completed_origins=0,
        total_origins=total_origins,
    )

    source_estimates: list[torch.Tensor] = []
    source_errors: list[torch.Tensor] = []
    candidate_estimates: list[torch.Tensor] = []
    candidate_errors: list[torch.Tensor] = []
    difference_estimates: list[torch.Tensor] = []
    difference_errors: list[torch.Tensor] = []
    incremental_kls: list[torch.Tensor] = []
    minimum_derivative = float("inf")
    for start in range(0, total_origins, arguments.origins_per_batch):
        stop = min(start + arguments.origins_per_batch, total_origins)
        indices = torch.arange(start, stop, dtype=torch.long, device=env.device)
        source_by_replication: list[torch.Tensor] = []
        candidate_by_replication: list[torch.Tensor] = []
        kl_by_replication: list[torch.Tensor] = []
        for replication in range(arguments.replications):
            generator = torch.Generator(device=env.device)
            generator.manual_seed(
                arguments.seed + start * arguments.replications + replication
            )
            batch = sample_conditional_paths(
                validation_data,
                indices,
                spot_contexts_per_origin=arguments.spot_contexts_per_origin,
                surface_paths_per_context=arguments.surface_paths_per_context,
                generator=generator,
            )
            path_log_spot = _path_log_spot(
                validation_log_spot_tensor[indices],
                batch,
            )
            source_draw, _ = apply_first_line_transport_sequence(
                (source_transport,),
                batch.current_state,
                batch.gross_spot_return,
                batch.standard_normals,
                path_log_spot,
            )
            source_returns = w2_roll_returns(
                simulator,
                validation_data,
                batch,
                source_draw.reduced_state,
                normalized_strikes,
            ).reshape(
                len(indices),
                arguments.spot_contexts_per_origin,
                arguments.surface_paths_per_context,
                1,
            )
            _, details = direct_w2_conditional_moment_transport_loss(
                incremental_kernel=incremental,
                current_kernels=(source_transport,),
                simulator=simulator,
                data=validation_data,
                batch=batch,
                normalized_strikes=normalized_strikes,
                inverse_sqrt_second_moment=inverse_sqrt,
                current_log_spot=validation_log_spot_tensor[indices],
                kl_weight=0.0,
            )
            candidate_returns = details["candidate_returns"]
            source_by_replication.append(
                torch.mean(source_returns, dim=(1, 2)).double().cpu()
            )
            candidate_by_replication.append(
                torch.mean(candidate_returns, dim=(1, 2)).double().cpu()
            )
            kl_by_replication.append(details["incremental_kl"].reshape(1).cpu())
            derivative = incremental.minimum_first_line_coordinate_derivative(
                batch.current_state,
                batch.gross_spot_return,
                path_log_spot,
            )
            minimum_derivative = min(minimum_derivative, float(derivative.cpu()))

        source_replicates = torch.stack(source_by_replication, dim=1)
        candidate_replicates = torch.stack(candidate_by_replication, dim=1)
        source_estimate, source_error = replicated_conditional_moment_statistics(
            source_replicates
        )
        candidate_estimate, candidate_error = (
            replicated_conditional_moment_statistics(candidate_replicates)
        )
        difference_estimate, difference_error = (
            replicated_conditional_moment_statistics(
                candidate_replicates - source_replicates
            )
        )
        source_estimates.append(source_estimate)
        source_errors.append(source_error)
        candidate_estimates.append(candidate_estimate)
        candidate_errors.append(candidate_error)
        difference_estimates.append(difference_estimate)
        difference_errors.append(difference_error)
        incremental_kls.extend(kl_by_replication)
        if stop == total_origins or stop % arguments.heartbeat_every_origins == 0:
            _write_heartbeat(
                arguments,
                completed_origins=stop,
                total_origins=total_origins,
            )
        if stop == total_origins or stop % arguments.report_every_origins == 0:
            print(
                f"high-path W2 ATM validation origins={stop}/{total_origins}",
                flush=True,
            )

    source_estimate = torch.cat(source_estimates)
    source_error = torch.cat(source_errors)
    candidate_estimate = torch.cat(candidate_estimates)
    candidate_error = torch.cat(candidate_errors)
    difference_estimate = torch.cat(difference_estimates)
    difference_error = torch.cat(difference_errors)
    source_summary = _moment_summary(source_estimate, source_error)
    candidate_summary = _moment_summary(candidate_estimate, candidate_error)
    difference_summary = _moment_summary(difference_estimate, difference_error)
    source_rmse = source_summary["conditional_moment_rmse_bps_spot"]
    candidate_rmse = candidate_summary["conditional_moment_rmse_bps_spot"]
    original_audit = cast(Mapping[str, Any], candidate_payload["audit"])
    report = {
        "schema": "flex-direct-moment-w2-atm-high-path-audit-v1",
        "status": "complete",
        "generation": arguments.generation_token,
        "candidate_checkpoint": str(arguments.candidate_checkpoint),
        "candidate_checkpoint_sha256": checkpoint_digest(
            arguments.candidate_checkpoint
        ),
        "candidate_checkpoint_generation": candidate_payload.get("generation"),
        "arguments": _json_arguments(arguments),
        "validation_dates": [
            validation_dates[0].isoformat(),
            validation_dates[total_origins - 1].isoformat(),
        ],
        "num_validation_origins": total_origins,
        "option": {
            "type": "call",
            "current_expiry": "W2",
            "rolled_expiry": "W1",
            "normalized_strike": 1.0,
        },
        "sampling": {
            "measure": "direct candidate draws; no importance weights",
            "common_random_numbers": "source and candidate share each draw",
            "spot": "independently randomized stratified implied-front-week grid per replication",
            "surface": "independent antithetic normals per spot context and replication",
            "replications": arguments.replications,
            "paths_per_replication_per_origin": (
                arguments.spot_contexts_per_origin
                * arguments.surface_paths_per_context
            ),
            "total_paths_per_origin": (
                arguments.replications
                * arguments.spot_contexts_per_origin
                * arguments.surface_paths_per_context
            ),
            "standard_error": (
                "sample standard deviation of independent replication moments "
                "divided by sqrt(replications)"
            ),
        },
        "source_q1": source_summary,
        "candidate": candidate_summary,
        "candidate_minus_source_paired": difference_summary,
        "comparison": {
            "candidate_relative_rmse_change": candidate_rmse / source_rmse - 1.0,
            "candidate_rmse_improvement": 1.0 - candidate_rmse / source_rmse,
            "mean_incremental_kl_to_source": float(
                torch.mean(torch.cat(incremental_kls))
            ),
            "minimum_incremental_coordinate_derivative": minimum_derivative,
        },
        "previous_4096_path_validation_audit": original_audit["validation_2025"],
    }
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = (
        arguments.report_directory
        / "direct-moment-transport-w2-atm-high-path-validation.json"
    )
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-direct-moment-w2-atm-high-path-completion-v1",
            "status": "complete",
            "semantic_success": True,
            "generation": arguments.generation_token,
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "candidate_checkpoint": str(arguments.candidate_checkpoint),
            "candidate_checkpoint_sha256": checkpoint_digest(
                arguments.candidate_checkpoint
            ),
        },
        arguments.report_directory
        / "direct-moment-transport-w2-atm-high-path-complete.json",
    )
    _write_heartbeat(
        arguments,
        completed_origins=total_origins,
        total_origins=total_origins,
        status="SUCCEEDED",
    )


def main() -> None:
    """CLI entry point with a report-identity lock."""

    arguments = _parse_arguments()
    lock_path = arguments.report_directory / ".high-path-audit.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_direct_moment_transport_w2_high_path_audit",
        clean_sigterm=True,
    )
