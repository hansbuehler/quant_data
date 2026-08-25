"""Solve and validate one scalar ATM W2 OCE trade at one fixed state."""

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
    estimate_return_inverse_sqrt,
    sample_conditional_paths,
    solve_conditional_powerexp_oce,
    spot_marginal_oce_objective,
)
from quantdata.dysanos.flex.run_direct_moment_transport_w2_high_path_audit import (
    _validate_candidate_checkpoint,
)
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import _load_inputs
from quantdata.dysanos.flex.run_oce_transport_q1 import (
    _build_kernels,
    _estimate_preconditioner,
    _load_oce_action_warm_start,
    _tensor_sha256,
)
from quantdata.dysanos.flex.run_sequential_oce_transport_w2 import (
    _source_action_config,
)
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainingData,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-heartbeat", type=Path, default=None)
    parser.add_argument("--generation-token", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--origins-per-scan-batch", type=int, default=2)
    parser.add_argument("--scan-spot-contexts", type=int, default=32)
    parser.add_argument("--scan-surface-paths", type=int, default=128)
    parser.add_argument("--maximum-scan-origins", type=int, default=None)
    parser.add_argument("--fit-replications", type=int, default=16)
    parser.add_argument("--fit-spot-contexts", type=int, default=32)
    parser.add_argument("--fit-surface-paths", type=int, default=128)
    parser.add_argument("--test-replications", type=int, default=16)
    parser.add_argument("--test-spot-contexts", type=int, default=32)
    parser.add_argument("--test-surface-paths", type=int, default=128)
    parser.add_argument("--solver-iterations", type=int, default=64)
    parser.add_argument("--scan-seed", type=int, default=8_242_913)
    parser.add_argument("--fit-seed", type=int, default=8_243_913)
    parser.add_argument("--test-seed", type=int, default=8_244_913)
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("fixed-state scalar OCE is restricted to GPU0")
    positive = (
        arguments.origins_per_scan_batch,
        arguments.scan_spot_contexts,
        arguments.scan_surface_paths,
        arguments.fit_replications,
        arguments.fit_spot_contexts,
        arguments.fit_surface_paths,
        arguments.test_replications,
        arguments.test_spot_contexts,
        arguments.test_surface_paths,
        arguments.solver_iterations,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("fixed-state scalar OCE controls must be positive")
    if arguments.fit_replications < 2 or arguments.test_replications < 2:
        raise ValueError("fit and test require at least two independent replications")
    path_counts = (
        arguments.scan_spot_contexts,
        arguments.scan_surface_paths,
        arguments.fit_spot_contexts,
        arguments.fit_surface_paths,
        arguments.test_spot_contexts,
        arguments.test_surface_paths,
    )
    if any(value % 2 for value in path_counts):
        raise ValueError("all fixed-state scalar OCE path counts must be even")
    if (
        arguments.maximum_scan_origins is not None
        and arguments.maximum_scan_origins <= 0
    ):
        raise ValueError("maximum scan origins must be positive when specified")
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
    phase: str,
    completed: int,
    total: int,
    status: str = "RUNNING",
) -> None:
    if arguments.worker_heartbeat is None:
        return
    write_json_atomic(
        {
            "schema": "flex-fixed-state-scalar-oce-worker-heartbeat-v1",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "generation": arguments.generation_token,
            "worker_pid": os.getpid(),
            "status": status,
            "last_progress_value": {
                "phase": phase,
                "completed": completed,
                "total": total,
            },
        },
        arguments.worker_heartbeat,
    )


def _load_candidate_runtime(
    candidate_checkpoint: Path,
    *,
    env: BaseEnvironment,
    normalized_strikes: torch.Tensor | None = None,
) -> tuple[
    Mapping[str, Any],
    list[Any],
    torch.Tensor,
    WeeklySanosSimulator,
    GenericWeeklyMeasureKernel,
    SpotConditionedFirstLineTransport,
    WeeklyMeasureTrainingData,
    torch.Tensor,
    torch.Tensor,
]:
    candidate_payload, candidate_arguments = _validate_candidate_checkpoint(
        candidate_checkpoint
    )
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
    if normalized_strikes is None:
        normalized_strikes = torch.ones(1, dtype=env.fdtype, device=env.device)
    elif (
        normalized_strikes.ndim != 1
        or len(normalized_strikes) < 1
        or not bool(torch.all(torch.isfinite(normalized_strikes)))
        or not bool(torch.all(normalized_strikes > 0.0))
        or (
            len(normalized_strikes) > 1
            and not bool(torch.all(normalized_strikes[1:] > normalized_strikes[:-1]))
        )
    ):
        raise ValueError("normalized strike grid must be finite, positive, and increasing")
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
    return (
        candidate_payload,
        validation_dates,
        validation_log_spot_tensor,
        simulator,
        source_transport,
        incremental,
        validation_data,
        normalized_strikes,
        inverse_sqrt,
    )


@torch.no_grad()
def _candidate_returns(
    *,
    incremental: SpotConditionedFirstLineTransport,
    source_transport: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    inverse_sqrt: torch.Tensor,
    log_spot: torch.Tensor,
    indices: torch.Tensor,
    spot_contexts: int,
    surface_paths: int,
    seed: int,
    option_is_call: torch.Tensor | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed)
    batch = sample_conditional_paths(
        data,
        indices,
        spot_contexts_per_origin=spot_contexts,
        surface_paths_per_context=surface_paths,
        generator=generator,
    )
    _, details = direct_w2_conditional_moment_transport_loss(
        incremental_kernel=incremental,
        current_kernels=(source_transport,),
        simulator=simulator,
        data=data,
        batch=batch,
        normalized_strikes=normalized_strikes,
        inverse_sqrt_second_moment=inverse_sqrt,
        current_log_spot=log_spot[indices],
        kl_weight=0.0,
    )
    candidate_returns = details["candidate_returns"]
    if option_is_call is None:
        return candidate_returns
    if option_is_call.shape != (candidate_returns.shape[-1],):
        raise ValueError("fixed-state option-type vector has invalid shape")
    # Put-call parity changes a call roll residual into the corresponding put
    # residual by subtracting S[t+1]/S[t]-1. The spot marginal is fixed and has
    # population mean zero in this coordinate. Centering its finite quadrature
    # estimate is therefore a control variate, not a change to the martingale
    # restriction. Per-spot density normalization preserves the centering under
    # every OCE density considered below.
    gross = batch.gross_spot_return.reshape(
        batch.num_origins,
        batch.spot_contexts_per_origin,
        batch.surface_paths_per_context,
    )
    centered_spot_return = gross - 1.0
    centered_spot_return = centered_spot_return - torch.mean(
        centered_spot_return,
        dim=(1, 2),
        keepdim=True,
    )
    return torch.where(
        option_is_call[None, None, None],
        candidate_returns,
        candidate_returns - centered_spot_return[..., None],
    )


def _replicated_return_statistics(
    returns: torch.Tensor,
    *,
    replications: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if returns.shape[0] != 1 or returns.shape[1] % replications:
        raise ValueError("fixed-state return replication geometry is invalid")
    spot_contexts = returns.shape[1] // replications
    replicate_moments = torch.mean(
        returns.reshape(
            replications,
            spot_contexts,
            returns.shape[2],
            returns.shape[3],
        ),
        dim=(1, 2),
    )
    return replicated_conditional_moment_statistics(replicate_moments[None])


def _evaluate_action(
    returns: torch.Tensor,
    action: torch.Tensor,
    *,
    replications: int,
) -> dict[str, float]:
    objective, details = spot_marginal_oce_objective(returns, action)
    density = torch.exp(details["log_marginal"])
    weighted_returns = density[..., None] * returns
    raw_moment, raw_error = _replicated_return_statistics(
        returns,
        replications=replications,
    )
    weighted_moment, weighted_error = _replicated_return_statistics(
        weighted_returns,
        replications=replications,
    )
    pnl_moment = action.cpu() * raw_moment.cpu()
    pnl_error = torch.abs(action.cpu()) * raw_error.cpu()
    effective_sample_size = density.sum(dim=2).square() / density.square().sum(dim=2)
    return {
        "raw_option_moment_bps_spot": float(raw_moment[0, 0].cpu()) * 1.0e4,
        "raw_option_moment_mc_se_bps_spot": float(raw_error[0, 0].cpu()) * 1.0e4,
        "oce_weighted_option_moment_bps_spot": float(
            weighted_moment[0, 0].cpu()
        )
        * 1.0e4,
        "oce_weighted_option_moment_mc_se_bps_spot": float(
            weighted_error[0, 0].cpu()
        )
        * 1.0e4,
        "expected_trade_pnl_bps_spot": float(pnl_moment[0, 0]) * 1.0e4,
        "expected_trade_pnl_mc_se_bps_spot": float(pnl_error[0, 0]) * 1.0e4,
        "powerexp_oce_objective": float(objective.cpu()),
        "maximum_abs_density_mass_residual": float(
            torch.max(torch.abs(details["mass_residual"])).cpu()
        ),
        "density_minimum": float(torch.min(density).cpu()),
        "density_maximum": float(torch.max(density).cpu()),
        "minimum_surface_ess_fraction": float(
            torch.min(effective_sample_size / returns.shape[2]).cpu()
        ),
        "median_surface_ess_fraction": float(
            torch.median(effective_sample_size / returns.shape[2]).cpu()
        ),
        "return_minimum_bps_spot": float(torch.min(returns).cpu()) * 1.0e4,
        "return_maximum_bps_spot": float(torch.max(returns).cpu()) * 1.0e4,
        "return_rms_bps_spot": float(torch.sqrt(torch.mean(returns.square())).cpu())
        * 1.0e4,
        "fraction_positive_returns": float(torch.mean((returns > 0.0).float()).cpu()),
    }


@torch.no_grad()
def run(arguments: argparse.Namespace) -> None:
    """Select, fit, and independently test one fixed-state scalar trade."""

    _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        candidate_payload,
        validation_dates,
        validation_log_spot,
        simulator,
        source_transport,
        incremental,
        validation_data,
        normalized_strikes,
        inverse_sqrt,
    ) = _load_candidate_runtime(arguments.candidate_checkpoint, env=env)
    total_origins = min(
        len(validation_dates),
        arguments.maximum_scan_origins or len(validation_dates),
    )
    _write_heartbeat(
        arguments,
        phase="selection-scan",
        completed=0,
        total=total_origins,
    )
    scan_moments: list[torch.Tensor] = []
    for start in range(0, total_origins, arguments.origins_per_scan_batch):
        stop = min(start + arguments.origins_per_scan_batch, total_origins)
        indices = torch.arange(start, stop, dtype=torch.long, device=env.device)
        returns = _candidate_returns(
            incremental=incremental,
            source_transport=source_transport,
            simulator=simulator,
            data=validation_data,
            normalized_strikes=normalized_strikes,
            inverse_sqrt=inverse_sqrt,
            log_spot=validation_log_spot,
            indices=indices,
            spot_contexts=arguments.scan_spot_contexts,
            surface_paths=arguments.scan_surface_paths,
            seed=arguments.scan_seed + start,
        )
        scan_moments.append(torch.mean(returns, dim=(1, 2)).cpu())
        if stop == total_origins or stop % 16 == 0:
            _write_heartbeat(
                arguments,
                phase="selection-scan",
                completed=stop,
                total=total_origins,
            )
            print(f"fixed-state scan origins={stop}/{total_origins}", flush=True)
    scan = torch.cat(scan_moments)[:, 0]
    selected_index = int(torch.argmax(torch.abs(scan)))
    selected_scan_moment = float(scan[selected_index])
    index = torch.tensor((selected_index,), dtype=torch.long, device=env.device)

    def replicated_returns(
        *,
        replications: int,
        spot_contexts: int,
        surface_paths: int,
        seed: int,
        phase: str,
    ) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        _write_heartbeat(
            arguments,
            phase=phase,
            completed=0,
            total=replications,
        )
        for replication in range(replications):
            rows.append(
                _candidate_returns(
                    incremental=incremental,
                    source_transport=source_transport,
                    simulator=simulator,
                    data=validation_data,
                    normalized_strikes=normalized_strikes,
                    inverse_sqrt=inverse_sqrt,
                    log_spot=validation_log_spot,
                    indices=index,
                    spot_contexts=spot_contexts,
                    surface_paths=surface_paths,
                    seed=seed + replication,
                )
            )
            _write_heartbeat(
                arguments,
                phase=phase,
                completed=replication + 1,
                total=replications,
            )
        return torch.cat(rows, dim=1)

    fit_returns = replicated_returns(
        replications=arguments.fit_replications,
        spot_contexts=arguments.fit_spot_contexts,
        surface_paths=arguments.fit_surface_paths,
        seed=arguments.fit_seed,
        phase="fit-paths",
    )
    return_inverse_sqrt = estimate_return_inverse_sqrt(
        fit_returns.reshape(-1, fit_returns.shape[2], 1),
        minimum_standard_deviation=1.0e-8,
    )
    action, translation, fit_details = solve_conditional_powerexp_oce(
        fit_returns,
        return_inverse_sqrt,
        num_iterations=arguments.solver_iterations,
        ridge=1.0e-10,
    )
    if not bool(torch.all(torch.isfinite(action))):
        raise FloatingPointError("fixed-state scalar OCE action is non-finite")
    fit_evaluation = _evaluate_action(
        fit_returns,
        action,
        replications=arguments.fit_replications,
    )

    test_returns = replicated_returns(
        replications=arguments.test_replications,
        spot_contexts=arguments.test_spot_contexts,
        surface_paths=arguments.test_surface_paths,
        seed=arguments.test_seed,
        phase="test-paths",
    )
    test_evaluation = _evaluate_action(
        test_returns,
        action,
        replications=arguments.test_replications,
    )
    return_rms = float(torch.sqrt(torch.mean(fit_returns.square())).cpu())
    report = {
        "schema": "flex-fixed-state-scalar-oce-v1",
        "status": "complete",
        "generation": arguments.generation_token,
        "arguments": _json_arguments(arguments),
        "candidate_checkpoint": str(arguments.candidate_checkpoint),
        "candidate_checkpoint_sha256": checkpoint_digest(
            arguments.candidate_checkpoint
        ),
        "candidate_checkpoint_generation": candidate_payload.get("generation"),
        "selection": {
            "method": (
                "largest absolute candidate ATM W2 conditional moment on the "
                "independent scan sample"
            ),
            "validation_index": selected_index,
            "market_date": validation_dates[selected_index].isoformat(),
            "scan_moment_bps_spot": selected_scan_moment * 1.0e4,
            "num_scanned_origins": total_origins,
            "scan_paths_per_origin": (
                arguments.scan_spot_contexts * arguments.scan_surface_paths
            ),
        },
        "trade": {
            "instrument": "ATM call, current W2 rolled to W1 at the same cash strike",
            "powerexp_oce_action_raw_notional": float(action[0, 0].cpu()),
            "powerexp_oce_action_in_return_rms_units": float(
                action[0, 0].cpu()
            )
            * return_rms,
            "action_sign": "long" if float(action[0, 0].cpu()) > 0.0 else "short",
            "fit_return_rms_bps_spot": return_rms * 1.0e4,
            "fit_translation_minimum": float(torch.min(translation).cpu()),
            "fit_translation_maximum": float(torch.max(translation).cpu()),
            "fit_solver_action_residual_bps_spot": float(
                fit_details["action_residual"][0, 0].cpu()
            )
            * 1.0e4,
            "fit_solver_maximum_density_mass_residual": float(
                torch.max(torch.abs(fit_details["mass_residual"])).cpu()
            ),
        },
        "fit": {
            "replications": arguments.fit_replications,
            "total_paths": int(fit_returns.numel()),
            **fit_evaluation,
        },
        "independent_test": {
            "replications": arguments.test_replications,
            "total_paths": int(test_returns.numel()),
            **test_evaluation,
        },
        "interpretation": (
            "The scalar action is fixed after fitting. On the independent test, "
            "only the spotwise cash translations are re-solved because they are "
            "the conditional density normalizers, not additional option trades."
        ),
    }
    write_json_atomic(report, arguments.output)
    completion_path = arguments.output.with_name(
        f"{arguments.output.stem}-complete.json"
    )
    write_json_atomic(
        {
            "schema": "flex-fixed-state-scalar-oce-completion-v1",
            "status": "complete",
            "semantic_success": True,
            "generation": arguments.generation_token,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "candidate_checkpoint": str(arguments.candidate_checkpoint),
            "candidate_checkpoint_sha256": checkpoint_digest(
                arguments.candidate_checkpoint
            ),
        },
        completion_path,
    )
    _write_heartbeat(
        arguments,
        phase="complete",
        completed=1,
        total=1,
        status="SUCCEEDED",
    )


def main() -> None:
    """CLI entry point with an output-identity lock."""

    arguments = _parse_arguments()
    lock_path = arguments.output.with_suffix(arguments.output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_fixed_state_scalar_oce",
        clean_sigterm=True,
    )
