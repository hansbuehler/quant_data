"""Test strict finite-sample arbitrage in generic-Q1 W2 roll scenarios."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_spot_correlation import (
    fit_joint_spot_correlation_under_p,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import (
    _load_inputs,
    _sha256_tensor,
)
from quantdata.dysanos.flex.sample_arbitrage import solve_strict_sample_arbitrage
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainingData,
    _expanded_batch,
    _make_residual_function,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--q1-completion", type=Path, required=True)
    parser.add_argument("--moment-audit-completion", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-paths", type=int, default=131_072)
    parser.add_argument("--replay-paths", type=int, default=262_144)
    parser.add_argument("--search-seed", type=int, default=7_120_211)
    parser.add_argument("--replay-seed", type=int, default=8_120_211)
    parser.add_argument("--constraints-per-iteration", type=int, default=128)
    parser.add_argument("--maximum-lp-iterations", type=int, default=128)
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _validate_arguments(arguments: argparse.Namespace) -> None:
    for label, value in (
        ("search_paths", arguments.search_paths),
        ("replay_paths", arguments.replay_paths),
    ):
        if value < 4 or value % 2:
            raise ValueError(f"{label} must be an even integer >=4, got {value}")
    if arguments.constraints_per_iteration <= 0 or arguments.maximum_lp_iterations <= 0:
        raise ValueError("LP iteration controls must be positive")


def _sample_roll_returns(
    kernel: GenericWeeklyMeasureKernel,
    residual_function: Any,
    data: WeeklyMeasureTrainingData,
    *,
    origin_index: int,
    paths: int,
    seed: int,
) -> np.ndarray:
    """Draw spot and fixed-cash-strike option value changes for one state."""
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed)
    indices = torch.as_tensor(
        [origin_index], dtype=torch.long, device=data.reduced_states.device
    )
    with torch.no_grad():
        current, gross, normals, repeated = _expanded_batch(
            data,
            indices,
            paths_per_origin=paths,
            generator=generator,
            antithetic=True,
        )
        option_returns = residual_function(
            current,
            gross,
            normals,
            data.current_calls[repeated, 1],
        )[:, 0]
        returns = torch.cat((gross[:, None] - 1.0, option_returns), dim=1)
    result = returns.detach().cpu().double().numpy()
    if result.shape != (paths, 1 + option_returns.shape[1]):
        raise RuntimeError(f"Unexpected sampled return shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Sampled roll returns contain NaN or infinity")
    return result


def _replay_action(action: np.ndarray, returns: np.ndarray) -> dict[str, object]:
    pnl = returns @ action
    numerical_band = float(
        64.0
        * np.finfo(np.float32).eps
        * np.dot(np.abs(action), 1.0 + np.max(np.abs(returns), axis=0))
    )
    return {
        "paths": len(returns),
        "minimum_pnl": float(np.min(pnl)),
        "minimum_pnl_bps_spot": float(np.min(pnl)) * 1.0e4,
        "maximum_pnl": float(np.max(pnl)),
        "maximum_pnl_bps_spot": float(np.max(pnl)) * 1.0e4,
        "mean_pnl_bps_spot": float(np.mean(pnl)) * 1.0e4,
        "numerical_band": numerical_band,
        "numerical_band_bps_spot": numerical_band * 1.0e4,
        "paths_below_negative_band": int(np.count_nonzero(pnl < -numerical_band)),
        "strict_sample_arbitrage": bool(np.min(pnl) > numerical_band),
    }


def run(arguments: argparse.Namespace) -> None:
    """Search for and independently replay strict sampled arbitrage baskets."""
    _validate_arguments(arguments)
    if arguments.device != "cuda:0":
        raise ValueError("The generic Q1 sample-arbitrage test is restricted to GPU 0")
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        train_dates,
        train_states,
        train_log_spot,
        validation_dates,
        validation_states,
        _,
        joint_sources,
    ) = _load_inputs(arguments.joint_completion, env=env)

    q1_completion = _load_json(arguments.q1_completion)
    if (
        q1_completion.get("schema")
        != "flex-joint-hierarchical-generic-q1-completion-v1"
        or q1_completion.get("status") != "complete"
    ):
        raise ValueError("Generic Q1 completion proof is not complete")
    q1_report_path = Path(str(q1_completion["report"]))
    q1_checkpoint_path = Path(str(q1_completion["checkpoint"]))
    if (
        checkpoint_digest(q1_report_path) != q1_completion.get("report_sha256")
        or checkpoint_digest(q1_checkpoint_path)
        != q1_completion.get("checkpoint_sha256")
    ):
        raise ValueError("Generic Q1 completion provenance is invalid")
    q1_report = _load_json(q1_report_path)
    q1_checkpoint = torch.load(
        q1_checkpoint_path, map_location="cpu", weights_only=False
    )

    moment_completion = _load_json(arguments.moment_audit_completion)
    if (
        moment_completion.get("schema")
        != "flex-generic-q1-high-path-arb-audit-completion-v1"
        or moment_completion.get("status") != "complete"
    ):
        raise ValueError("High-path moment-audit completion proof is not complete")
    moment_report_path = Path(str(moment_completion["report"]))
    if checkpoint_digest(moment_report_path) != moment_completion.get("report_sha256"):
        raise ValueError("High-path moment-audit provenance is invalid")
    moment_report = _load_json(moment_report_path)
    confirmation = cast(Mapping[str, object], moment_report["confirmation"])
    failed_dates = tuple(
        str(value) for value in confirmation["bonferroni_significant_dates"]
    )
    date_to_index = {value.isoformat(): index for index, value in enumerate(validation_dates)}
    if not failed_dates or any(value not in date_to_index for value in failed_dates):
        raise ValueError("High-path audit failed dates do not match 2025 states")

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
    simulator = WeeklySanosSimulator(
        env,
        dynamics=dynamics,
        calibrated_model=model,
    )
    log_moneyness = tuple(float(value) for value in np.linspace(-0.30, 0.30, 11))
    normalized_strikes = torch.exp(
        torch.tensor(log_moneyness, dtype=env.fdtype, device=env.device)
    )
    validation_data = precompute_measure_training_data(
        simulator,
        validation_states,
        normalized_strikes,
        num_expiries=2,
    )
    measure_config = WeeklyMeasureConfig(
        label="joint-hierarchical-200-generic-q1-v1",
        state_sizes_by_expiry=model.config.state_sizes_by_expiry,
        hierarchical=True,
    )
    if (
        not isinstance(q1_checkpoint, Mapping)
        or q1_checkpoint.get("schema") != "flex-weekly-measure-kernel-v1"
        or q1_checkpoint.get("phase") != "complete"
        or q1_checkpoint.get("kernel_class") != "GenericWeeklyMeasureKernel"
        or q1_checkpoint.get("kernel_config") != asdict(measure_config)
        or q1_checkpoint.get("state_sha256") != _sha256_tensor(train_states)
    ):
        raise ValueError("Generic Q1 checkpoint identity is invalid")
    if json.loads(json.dumps(asdict(measure_config))) != q1_report.get(
        "measure_config"
    ):
        raise ValueError("Generic Q1 report geometry differs from current code")
    kernel = GenericWeeklyMeasureKernel(env, dynamics=dynamics, config=measure_config)
    kernel.load_state_dict(
        cast(Mapping[str, torch.Tensor], q1_checkpoint["model_state_dict"])
    )
    kernel.eval()
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    residual_function = _make_residual_function(
        kernel,
        simulator,
        normalized_strikes,
        current_expiry_index=1,
        compile_objective=False,
    )

    instrument_labels = ["spot"] + [
        f"call_k_{float(value.cpu()):.9g}" for value in normalized_strikes
    ]
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    progress_path = arguments.report_directory / "generic-q1-sample-arbitrage-progress.json"
    run_identity = {
        "joint_completion_sha256": checkpoint_digest(arguments.joint_completion),
        "q1_completion_sha256": checkpoint_digest(arguments.q1_completion),
        "moment_audit_completion_sha256": checkpoint_digest(
            arguments.moment_audit_completion
        ),
        "search_paths": arguments.search_paths,
        "replay_paths": arguments.replay_paths,
        "search_seed": arguments.search_seed,
        "replay_seed": arguments.replay_seed,
        "constraints_per_iteration": arguments.constraints_per_iteration,
        "maximum_lp_iterations": arguments.maximum_lp_iterations,
        "failed_dates": list(failed_dates),
    }
    state_results: list[dict[str, object]] = []
    if progress_path.exists():
        progress = _load_json(progress_path)
        if (
            progress.get("schema") != "flex-generic-q1-sample-arbitrage-progress-v1"
            or progress.get("run_identity") != run_identity
            or not isinstance(progress.get("state_results"), list)
        ):
            raise ValueError("Existing sample-arbitrage progress has invalid identity")
        state_results = list(cast(list[dict[str, object]], progress["state_results"]))
        completed_dates = [str(item.get("date")) for item in state_results]
        if completed_dates != list(failed_dates[: len(completed_dates)]):
            raise ValueError("Existing sample-arbitrage progress is not a valid prefix")
        print(f"resuming after {len(state_results)} completed states", flush=True)

    for position, state_date in enumerate(
        failed_dates[len(state_results) :], start=len(state_results) + 1
    ):
        origin_index = date_to_index[state_date]
        sampled_returns = _sample_roll_returns(
            kernel,
            residual_function,
            validation_data,
            origin_index=origin_index,
            paths=arguments.search_paths,
            seed=arguments.search_seed + origin_index,
        )
        search = solve_strict_sample_arbitrage(
            sampled_returns,
            constraints_per_iteration=arguments.constraints_per_iteration,
            maximum_iterations=arguments.maximum_lp_iterations,
        )
        result: dict[str, object] = {
            "date": state_date,
            "origin_index": origin_index,
            "search": search,
            "replay": None,
        }
        if search["strict_sample_arbitrage"]:
            replay_returns = _sample_roll_returns(
                kernel,
                residual_function,
                validation_data,
                origin_index=origin_index,
                paths=arguments.replay_paths,
                seed=arguments.replay_seed + origin_index,
            )
            result["replay"] = _replay_action(
                np.asarray(search["action"], dtype=np.float64),
                replay_returns,
            )
        state_results.append(result)
        write_json_atomic(
            {
                "schema": "flex-generic-q1-sample-arbitrage-progress-v1",
                "status": "running",
                "run_identity": run_identity,
                "completed_states": len(state_results),
                "state_results": state_results,
            },
            progress_path,
        )
        print(
            f"sample-arb {position}/{len(failed_dates)} date={state_date} "
            f"margin_bps={float(search['strict_margin_bps_spot']):.7g} "
            f"strict={search['strict_sample_arbitrage']}",
            flush=True,
        )

    search_candidates = [
        item
        for item in state_results
        if cast(Mapping[str, object], item["search"])["strict_sample_arbitrage"]
    ]
    replay_confirmed = [
        item
        for item in search_candidates
        if isinstance(item["replay"], Mapping)
        and cast(Mapping[str, object], item["replay"])["strict_sample_arbitrage"]
    ]
    maximum_margin = max(
        float(cast(Mapping[str, object], item["search"])["strict_margin_bps_spot"])
        for item in state_results
    )
    report_path = arguments.report_directory / "generic-q1-sample-arbitrage.json"
    completion_path = (
        arguments.report_directory / "generic-q1-sample-arbitrage-complete.json"
    )
    report: dict[str, object] = {
        "schema": "flex-generic-q1-sample-arbitrage-v1",
        "status": "complete",
        "test": (
            "maximize the worst sampled zero-cost portfolio P&L subject to a "
            "unit L1 action budget"
        ),
        "returns": (
            "spot R-1 and fixed-cash-strike rolled calls "
            "R*C_next_1W(k/R)-C_current_2W(k)"
        ),
        "geometry": (
            "strict positive margin is a separating hyperplane proving the "
            "current price vector lies outside the sampled successor-value "
            "convex hull"
        ),
        "scope_warning": (
            "Absence on finite samples does not prove absence on continuous "
            "full support; a detected search witness is independently replayed. "
            "Transaction costs are not included."
        ),
        "instrument_labels": instrument_labels,
        "num_tested_states": len(state_results),
        "search_paths_per_state": arguments.search_paths,
        "replay_paths_per_candidate": arguments.replay_paths,
        "search_strict_arbitrage_states": len(search_candidates),
        "replay_confirmed_strict_arbitrage_states": len(replay_confirmed),
        "maximum_search_margin_bps_spot": maximum_margin,
        "conclusion": (
            "replay_confirmed_strict_sample_arbitrage"
            if replay_confirmed
            else (
                "search_sample_only_strict_arbitrage"
                if search_candidates
                else "no_strict_arbitrage_on_sampled_successor_values"
            )
        ),
        "sources": {
            **joint_sources,
            "q1_completion": str(arguments.q1_completion),
            "q1_completion_sha256": checkpoint_digest(arguments.q1_completion),
            "q1_report": str(q1_report_path),
            "q1_report_sha256": checkpoint_digest(q1_report_path),
            "q1_checkpoint": str(q1_checkpoint_path),
            "q1_checkpoint_sha256": checkpoint_digest(q1_checkpoint_path),
            "moment_audit_completion": str(arguments.moment_audit_completion),
            "moment_audit_completion_sha256": checkpoint_digest(
                arguments.moment_audit_completion
            ),
            "moment_audit_report": str(moment_report_path),
            "moment_audit_report_sha256": checkpoint_digest(moment_report_path),
        },
        "state_results": state_results,
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-generic-q1-sample-arbitrage-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
        },
        completion_path,
    )


def main() -> None:
    """CLI entry point with durable exception reporting."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_joint_hierarchical_q1_sample_arbitrage")
