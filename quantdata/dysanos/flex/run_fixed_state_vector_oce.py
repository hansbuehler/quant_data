"""Solve and validate one eleven-ITM-option W2 OCE trade at one fixed state."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from filelock import FileLock

from quantdata.dysanos.flex.conditional_moment import (
    replicated_conditional_moment_statistics,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.oce_transport import (
    estimate_return_inverse_sqrt,
    solve_conditional_powerexp_oce_l1,
    spot_marginal_oce_objective,
)
from quantdata.dysanos.flex.run_fixed_state_scalar_oce import (
    _candidate_returns,
    _load_candidate_runtime,
)
from quantdata.dysanos.flex.training import TrainingConvergenceError
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--scalar-report", type=Path, required=True)
    parser.add_argument("--initial-action-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-heartbeat", type=Path, default=None)
    parser.add_argument("--generation-token", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-options", type=int, default=11)
    parser.add_argument("--minimum-log-strike", type=float, default=-0.30)
    parser.add_argument("--maximum-log-strike", type=float, default=0.30)
    parser.add_argument("--fit-replications", type=int, default=32)
    parser.add_argument("--fit-spot-contexts", type=int, default=32)
    parser.add_argument("--fit-surface-paths", type=int, default=256)
    parser.add_argument("--test-replications", type=int, default=32)
    parser.add_argument("--test-spot-contexts", type=int, default=32)
    parser.add_argument("--test-surface-paths", type=int, default=256)
    parser.add_argument("--solver-iterations", type=int, default=256)
    parser.add_argument("--minimum-return-standard-deviation", type=float, default=1.0e-4)
    parser.add_argument(
        "--l1-notional-caps",
        nargs="+",
        type=float,
        default=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0),
    )
    parser.add_argument("--l1-cost-bps-spot", type=float, default=1.0)
    parser.add_argument("--fit-seed", type=int, default=8_245_913)
    parser.add_argument("--test-seed", type=int, default=8_246_913)
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("fixed-state vector OCE is restricted to GPU0")
    if arguments.num_options != 11:
        raise ValueError("this control requires exactly eleven options")
    if (
        not math.isfinite(arguments.minimum_log_strike)
        or not math.isfinite(arguments.maximum_log_strike)
        or arguments.minimum_log_strike >= arguments.maximum_log_strike
    ):
        raise ValueError("invalid eleven-option log-strike interval")
    positive = (
        arguments.fit_replications,
        arguments.fit_spot_contexts,
        arguments.fit_surface_paths,
        arguments.test_replications,
        arguments.test_spot_contexts,
        arguments.test_surface_paths,
        arguments.solver_iterations,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("fixed-state vector OCE controls must be positive")
    if arguments.fit_replications < 12 or arguments.test_replications < 12:
        raise ValueError("eleven-option inference requires at least twelve replications")
    path_counts = (
        arguments.fit_spot_contexts,
        arguments.fit_surface_paths,
        arguments.test_spot_contexts,
        arguments.test_surface_paths,
    )
    if any(value % 2 for value in path_counts):
        raise ValueError("all fixed-state vector OCE path counts must be even")
    if (
        not math.isfinite(arguments.minimum_return_standard_deviation)
        or arguments.minimum_return_standard_deviation <= 0.0
    ):
        raise ValueError("minimum return standard deviation must be finite-positive")
    if (
        len(arguments.l1_notional_caps) < 1
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in arguments.l1_notional_caps
        )
        or any(
            right <= left
            for left, right in zip(
                arguments.l1_notional_caps[:-1],
                arguments.l1_notional_caps[1:],
                strict=True,
            )
        )
    ):
        raise ValueError("L1 notional caps must be finite-positive and increasing")
    if (
        not math.isfinite(arguments.l1_cost_bps_spot)
        or arguments.l1_cost_bps_spot < 0.0
    ):
        raise ValueError("L1 transaction cost must be finite-nonnegative")
    if arguments.fit_seed == arguments.test_seed:
        raise ValueError("fit and test random streams must be independent")
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
            "schema": "flex-fixed-state-vector-oce-worker-heartbeat-v1",
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


def _load_scalar_selection(
    path: Path,
    *,
    candidate_checkpoint: Path,
) -> tuple[int, str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("status") != "complete":
        raise ValueError("fixed-state scalar report is not complete")
    if payload.get("candidate_checkpoint_sha256") != checkpoint_digest(
        candidate_checkpoint
    ):
        raise ValueError("scalar report and candidate checkpoint provenance differ")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("scalar report has no fixed-state selection")
    validation_index = selection.get("validation_index")
    market_date = selection.get("market_date")
    if not isinstance(validation_index, int) or not isinstance(market_date, str):
        raise ValueError("scalar fixed-state selection is malformed")
    return validation_index, market_date, payload


def _load_initial_action(
    path: Path,
    *,
    arguments: argparse.Namespace,
    normalized_strikes: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Load a provenance-compatible non-converged action for continuation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "flex-fixed-state-vector-oce-v2"
    ):
        raise ValueError("initial action report has incompatible schema")
    if payload.get("candidate_checkpoint_sha256") != checkpoint_digest(
        arguments.candidate_checkpoint
    ) or payload.get("scalar_report_sha256") != checkpoint_digest(
        arguments.scalar_report
    ):
        raise ValueError("initial action report provenance differs")
    source_arguments = payload.get("arguments")
    if not isinstance(source_arguments, Mapping):
        raise ValueError("initial action report has no arguments")
    identical_controls = (
        "num_options",
        "minimum_log_strike",
        "maximum_log_strike",
        "fit_replications",
        "fit_spot_contexts",
        "fit_surface_paths",
        "test_replications",
        "test_spot_contexts",
        "test_surface_paths",
        "fit_seed",
        "test_seed",
        "minimum_return_standard_deviation",
        "l1_cost_bps_spot",
    )
    for name in identical_controls:
        if source_arguments.get(name) != getattr(arguments, name):
            raise ValueError(f"initial action report changes {name}")
    instruments = payload.get("instruments")
    if not isinstance(instruments, list) or len(instruments) != len(normalized_strikes):
        raise ValueError("initial action instrument geometry differs")
    source_strikes = torch.tensor(
        [cast(Mapping[str, Any], value)["strike_over_spot"] for value in instruments],
        dtype=normalized_strikes.dtype,
        device=normalized_strikes.device,
    )
    if not bool(torch.allclose(source_strikes, normalized_strikes, atol=1.0e-7, rtol=0.0)):
        raise ValueError("initial action strike grid differs")
    frontier = payload.get("l1_frontier")
    if not isinstance(frontier, list) or len(frontier) < 1:
        raise ValueError("initial action report has no frontier")
    maximum_cap = float(arguments.l1_notional_caps[0])
    eligible = [
        cast(Mapping[str, Any], point)
        for point in frontier
        if float(cast(Mapping[str, Any], point)["maximum_l1_raw_notional"])
        <= maximum_cap
    ]
    if not eligible:
        raise ValueError("initial action report has no action within the first cap")
    selected = max(eligible, key=lambda point: float(point["maximum_l1_raw_notional"]))
    trade = selected.get("trade")
    if not isinstance(trade, Mapping):
        raise ValueError("initial action frontier point has no trade")
    action = torch.as_tensor(
        trade["powerexp_oce_action_raw_notional"],
        dtype=normalized_strikes.dtype,
        device=normalized_strikes.device,
    ).reshape(1, -1)
    if action.shape != (1, arguments.num_options) or not bool(
        torch.all(torch.isfinite(action))
    ):
        raise ValueError("initial action vector is malformed")
    if float(torch.sum(torch.abs(action)).cpu()) > maximum_cap + 1.0e-6:
        raise ValueError("initial action exceeds the first requested cap")
    return action, float(selected["maximum_l1_raw_notional"])


def _replicate_moments(returns: torch.Tensor, *, replications: int) -> torch.Tensor:
    if returns.shape[0] != 1 or returns.shape[1] % replications:
        raise ValueError("fixed-state vector replication geometry is invalid")
    spot_contexts = returns.shape[1] // replications
    return torch.mean(
        returns.reshape(
            replications,
            spot_contexts,
            returns.shape[2],
            returns.shape[3],
        ),
        dim=(1, 2),
    )


def _moment_statistics(
    replicate_moments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return replicated_conditional_moment_statistics(replicate_moments[None])


def _evaluate_action(
    returns: torch.Tensor,
    action: torch.Tensor,
    *,
    replications: int,
    l1_cost_per_unit: float,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    objective, details = spot_marginal_oce_objective(returns, action)
    density = torch.exp(details["log_marginal"])
    raw_replicates = _replicate_moments(returns, replications=replications)
    weighted_replicates = _replicate_moments(
        density[..., None] * returns,
        replications=replications,
    )
    raw_moment, raw_error = _moment_statistics(raw_replicates)
    weighted_moment, weighted_error = _moment_statistics(weighted_replicates)
    pnl = torch.sum(returns * action[:, None, None], dim=-1, keepdim=True)
    pnl_replicates = _replicate_moments(pnl, replications=replications)
    pnl_moment, pnl_error = _moment_statistics(pnl_replicates)
    effective_sample_size = density.sum(dim=2).square() / density.square().sum(dim=2)
    l1_norm = float(torch.sum(torch.abs(action)).cpu())
    transaction_cost = l1_cost_per_unit * l1_norm
    evaluation: dict[str, object] = {
        "raw_option_moment_bps_spot": (raw_moment[0].cpu() * 1.0e4).tolist(),
        "raw_option_moment_mc_se_bps_spot": (raw_error[0].cpu() * 1.0e4).tolist(),
        "oce_weighted_option_moment_bps_spot": (
            weighted_moment[0].cpu() * 1.0e4
        ).tolist(),
        "oce_weighted_option_moment_mc_se_bps_spot": (
            weighted_error[0].cpu() * 1.0e4
        ).tolist(),
        "maximum_abs_oce_weighted_moment_bps_spot": float(
            torch.max(torch.abs(weighted_moment)).cpu()
        )
        * 1.0e4,
        "expected_trade_pnl_bps_spot": float(pnl_moment[0, 0].cpu()) * 1.0e4,
        "expected_trade_pnl_mc_se_bps_spot": float(pnl_error[0, 0].cpu()) * 1.0e4,
        "l1_transaction_cost_bps_spot": transaction_cost * 1.0e4,
        "expected_trade_pnl_after_l1_cost_bps_spot": (
            float(pnl_moment[0, 0].cpu()) - transaction_cost
        )
        * 1.0e4,
        "powerexp_oce_objective": float(objective.cpu()),
        "cost_adjusted_powerexp_oce_objective": (
            float(objective.cpu()) - transaction_cost
        ),
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
    }
    return evaluation, weighted_replicates, raw_replicates


@torch.no_grad()
def run(arguments: argparse.Namespace) -> None:
    """Fit and independently test eleven option positions at one fixed state."""

    _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    normalized_strikes = torch.exp(
        torch.linspace(
            arguments.minimum_log_strike,
            arguments.maximum_log_strike,
            arguments.num_options,
            dtype=env.fdtype,
            device=env.device,
        )
    )
    option_is_call = normalized_strikes <= 1.0
    (
        candidate_payload,
        validation_dates,
        validation_log_spot,
        simulator,
        source_transport,
        incremental,
        validation_data,
        runtime_strikes,
        _,
    ) = _load_candidate_runtime(
        arguments.candidate_checkpoint,
        env=env,
        normalized_strikes=normalized_strikes,
    )
    selected_index, selected_date, scalar_payload = _load_scalar_selection(
        arguments.scalar_report,
        candidate_checkpoint=arguments.candidate_checkpoint,
    )
    if selected_index >= len(validation_dates):
        raise ValueError("scalar fixed-state index is outside validation data")
    if validation_dates[selected_index].isoformat() != selected_date:
        raise ValueError("scalar fixed-state date no longer matches validation data")
    index = torch.tensor((selected_index,), dtype=torch.long, device=env.device)
    identity_scale = torch.eye(
        arguments.num_options,
        dtype=env.fdtype,
        device=env.device,
    )

    def replicated_returns(
        *,
        replications: int,
        spot_contexts: int,
        surface_paths: int,
        seed: int,
        phase: str,
    ) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        _write_heartbeat(arguments, phase=phase, completed=0, total=replications)
        for replication in range(replications):
            rows.append(
                _candidate_returns(
                    incremental=incremental,
                    source_transport=source_transport,
                    simulator=simulator,
                    data=validation_data,
                    normalized_strikes=runtime_strikes,
                    inverse_sqrt=identity_scale,
                    log_spot=validation_log_spot,
                    indices=index,
                    spot_contexts=spot_contexts,
                    surface_paths=surface_paths,
                    seed=seed + replication,
                    option_is_call=option_is_call,
                )
            )
            _write_heartbeat(
                arguments,
                phase=phase,
                completed=replication + 1,
                total=replications,
            )
            print(f"fixed-state vector {phase}={replication + 1}/{replications}", flush=True)
        return torch.cat(rows, dim=1)

    fit_returns = replicated_returns(
        replications=arguments.fit_replications,
        spot_contexts=arguments.fit_spot_contexts,
        surface_paths=arguments.fit_surface_paths,
        seed=arguments.fit_seed,
        phase="fit-paths",
    )
    return_inverse_sqrt = estimate_return_inverse_sqrt(
        fit_returns.reshape(-1, fit_returns.shape[2], fit_returns.shape[3]),
        minimum_standard_deviation=arguments.minimum_return_standard_deviation,
    )
    flattened = fit_returns.reshape(-1, arguments.num_options).double()
    second_moment = flattened.T @ flattened / len(flattened)
    eigenvalues = torch.linalg.eigvalsh(second_moment)
    test_returns = replicated_returns(
        replications=arguments.test_replications,
        spot_contexts=arguments.test_spot_contexts,
        surface_paths=arguments.test_surface_paths,
        seed=arguments.test_seed,
        phase="test-paths",
    )
    return_rms = torch.sqrt(torch.mean(fit_returns.square(), dim=(0, 1, 2)))
    l1_cost_per_unit = arguments.l1_cost_bps_spot * 1.0e-4
    frontier: list[dict[str, object]] = []
    initial_action: torch.Tensor | None = None
    initial_action_source_cap: float | None = None
    if arguments.initial_action_report is not None:
        initial_action, initial_action_source_cap = _load_initial_action(
            arguments.initial_action_report,
            arguments=arguments,
            normalized_strikes=runtime_strikes,
        )
    for cap_index, maximum_l1_action in enumerate(arguments.l1_notional_caps):
        _write_heartbeat(
            arguments,
            phase="l1-frontier",
            completed=cap_index,
            total=len(arguments.l1_notional_caps),
        )
        action, translation, fit_details = solve_conditional_powerexp_oce_l1(
            fit_returns,
            return_inverse_sqrt,
            maximum_l1_action=maximum_l1_action,
            l1_cost_per_unit=l1_cost_per_unit,
            initial_action=initial_action,
            num_iterations=arguments.solver_iterations,
        )
        if not bool(torch.all(torch.isfinite(action))):
            raise FloatingPointError("fixed-state vector OCE action is non-finite")
        fit_evaluation, fit_weighted_replicates, _ = _evaluate_action(
            fit_returns,
            action,
            replications=arguments.fit_replications,
            l1_cost_per_unit=l1_cost_per_unit,
        )
        test_evaluation, test_weighted_replicates, _ = _evaluate_action(
            test_returns,
            action,
            replications=arguments.test_replications,
            l1_cost_per_unit=l1_cost_per_unit,
        )
        _, fit_weighted_error = _moment_statistics(fit_weighted_replicates)
        test_weighted_moment, test_weighted_error = _moment_statistics(
            test_weighted_replicates
        )
        combined_error = torch.sqrt(
            fit_weighted_error.square() + test_weighted_error.square()
        )
        combined_z = test_weighted_moment / combined_error.clamp_min(
            torch.finfo(combined_error.dtype).tiny
        )
        frontier.append(
            {
                "maximum_l1_raw_notional": maximum_l1_action,
                "trade": {
                    "powerexp_oce_action_raw_notional": action[0].cpu().tolist(),
                    "realized_l1_raw_notional": float(
                        fit_details["l1_norm"][0].cpu()
                    ),
                    "powerexp_oce_action_times_marginal_return_rms": (
                        action[0].cpu() * return_rms.cpu()
                    ).tolist(),
                    "whitened_action": (
                        fit_details["whitened_action"][0].cpu().tolist()
                    ),
                    "whitened_action_l2_norm": float(
                        torch.linalg.vector_norm(
                            fit_details["whitened_action"][0]
                        ).cpu()
                    ),
                    "fit_solver_action_residual_bps_spot": (
                        fit_details["action_residual"][0].cpu() * 1.0e4
                    ).tolist(),
                    "fit_solver_maximum_abs_action_residual_bps_spot": float(
                        torch.max(torch.abs(fit_details["action_residual"])).cpu()
                    )
                    * 1.0e4,
                    "fit_solver_maximum_density_mass_residual": float(
                        torch.max(torch.abs(fit_details["mass_residual"])).cpu()
                    ),
                    "fit_translation_minimum": float(torch.min(translation).cpu()),
                    "fit_translation_maximum": float(torch.max(translation).cpu()),
                    "solver_accepted_iterations": int(
                        fit_details["accepted_iterations"][0].cpu()
                    ),
                    "solver_iterations_run": int(
                        fit_details["iterations_run"][0].cpu()
                    ),
                    "solver_converged": bool(fit_details["converged"][0].cpu()),
                    "solver_last_relative_action_change": float(
                        fit_details["last_relative_action_change"][0].cpu()
                    ),
                    "solver_last_objective_improvement": float(
                        fit_details["last_objective_improvement"][0].cpu()
                    ),
                },
                "fit": {
                    "replications": arguments.fit_replications,
                    "total_paths": (
                        arguments.fit_replications
                        * arguments.fit_spot_contexts
                        * arguments.fit_surface_paths
                    ),
                    **fit_evaluation,
                },
                "independent_test": {
                    "replications": arguments.test_replications,
                    "total_paths": (
                        arguments.test_replications
                        * arguments.test_spot_contexts
                        * arguments.test_surface_paths
                    ),
                    **test_evaluation,
                    "unpenalized_oce_weighted_moment_combined_fit_test_mc_se_bps_spot": (
                        combined_error[0].cpu() * 1.0e4
                    ).tolist(),
                    "unpenalized_oce_weighted_moment_combined_fit_test_z": (
                        combined_z[0].cpu().tolist()
                    ),
                    "maximum_abs_unpenalized_oce_weighted_moment_combined_fit_test_z": float(
                        torch.max(torch.abs(combined_z)).cpu()
                    ),
                },
            }
        )
        initial_action = action
        _write_heartbeat(
            arguments,
            phase="l1-frontier",
            completed=cap_index + 1,
            total=len(arguments.l1_notional_caps),
        )
        print(
            "fixed-state vector L1 cap="
            f"{maximum_l1_action:g} "
            "fit_net="
            f"{fit_evaluation['cost_adjusted_powerexp_oce_objective']:.9g} "
            "test_net="
            f"{test_evaluation['cost_adjusted_powerexp_oce_objective']:.9g}",
            flush=True,
        )
    solver_all_converged = all(
        bool(cast(Mapping[str, Any], point["trade"])["solver_converged"])
        for point in frontier
    )
    report = {
        "schema": "flex-fixed-state-vector-oce-v2",
        "status": "complete",
        "generation": arguments.generation_token,
        "arguments": _json_arguments(arguments),
        "candidate_checkpoint": str(arguments.candidate_checkpoint),
        "candidate_checkpoint_sha256": checkpoint_digest(
            arguments.candidate_checkpoint
        ),
        "candidate_checkpoint_generation": candidate_payload.get("generation"),
        "scalar_report": str(arguments.scalar_report),
        "scalar_report_sha256": checkpoint_digest(arguments.scalar_report),
        "initial_action": (
            None
            if arguments.initial_action_report is None
            else {
                "report": str(arguments.initial_action_report),
                "report_sha256": checkpoint_digest(arguments.initial_action_report),
                "source_cap": initial_action_source_cap,
            }
        ),
        "selection": {
            "method": "reuse the state selected by the scalar fixed-state control",
            "validation_index": selected_index,
            "market_date": selected_date,
            "scalar_scan_moment_bps_spot": cast(
                Mapping[str, Any], scalar_payload["selection"]
            )["scan_moment_bps_spot"],
        },
        "instruments": [
            {
                "index": strike_index,
                "type": "call" if bool(option_is_call[strike_index]) else "put",
                "log_strike_over_spot": float(torch.log(strike).cpu()),
                "strike_over_spot": float(strike.cpu()),
                "return_rms_bps_spot": float(return_rms[strike_index].cpu()) * 1.0e4,
            }
            for strike_index, strike in enumerate(runtime_strikes)
        ],
        "instrument_set": (
            "eleven ITM W2 options rolled after one week to W1 at the same "
            "cash strikes"
        ),
        "conditioning": {
            "return_second_moment_eigenvalues": eigenvalues.cpu().tolist(),
            "return_second_moment_condition_number": float(
                (eigenvalues[-1] / eigenvalues[0].clamp_min(1.0e-30)).cpu()
            ),
            "minimum_return_standard_deviation": (
                arguments.minimum_return_standard_deviation
            ),
        },
        "solver_all_converged": solver_all_converged,
        "l1_frontier": frontier,
        "interpretation": (
            "Each eleven-dimensional action is frozen after fitting. Increasing "
            "raw-notional L1 balls are solved by mechanical projection and warm "
            "starts, with one basis point of spot charged per unit raw notional. "
            "Independent test paths use a disjoint random stream; only spotwise "
            "cash translations are re-solved as density normalizers. No cap is "
            "selected using test performance."
        ),
    }
    write_json_atomic(report, arguments.output)
    if not solver_all_converged:
        raise TrainingConvergenceError(
            "one or more L1 frontier points failed projected-gradient convergence"
        )
    completion_path = arguments.output.with_name(
        f"{arguments.output.stem}-complete.json"
    )
    write_json_atomic(
        {
            "schema": "flex-fixed-state-vector-oce-completion-v2",
            "status": "complete",
            "semantic_success": True,
            "generation": arguments.generation_token,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "candidate_checkpoint": str(arguments.candidate_checkpoint),
            "candidate_checkpoint_sha256": checkpoint_digest(
                arguments.candidate_checkpoint
            ),
            "scalar_report": str(arguments.scalar_report),
            "scalar_report_sha256": checkpoint_digest(arguments.scalar_report),
            **(
                {}
                if arguments.initial_action_report is None
                else {
                    "initial_action_report": str(arguments.initial_action_report),
                    "initial_action_report_sha256": checkpoint_digest(
                        arguments.initial_action_report
                    ),
                }
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
        program="run_fixed_state_vector_oce",
        clean_sigterm=True,
    )
