"""High-path confirmation of apparent generic-Q1 martingale violations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
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
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainingData,
    audit_weekly_roll_line,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--q1-completion", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--screen-paths", type=int, default=4096)
    parser.add_argument("--confirmation-paths", type=int, default=131_072)
    parser.add_argument("--screen-origins-per-batch", type=int, default=4)
    parser.add_argument("--origins-per-batch", type=int, default=1)
    parser.add_argument("--screen-seed", type=int, default=120_211)
    parser.add_argument("--confirmation-seed", type=int, default=6_120_211)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _take_rows(
    data: WeeklyMeasureTrainingData,
    indices: torch.Tensor,
) -> WeeklyMeasureTrainingData:
    return WeeklyMeasureTrainingData(
        reduced_states=data.reduced_states[indices],
        front_support=data.front_support[indices],
        front_density=data.front_density[indices],
        front_backbone_total_vol=data.front_backbone_total_vol[indices],
        current_calls=data.current_calls[indices],
    )


def _pop_origin_details(
    audit: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    estimates = np.asarray(audit.pop("origin_estimates"), dtype=np.float64)
    standard_errors = np.asarray(
        audit.pop("origin_standard_errors"), dtype=np.float64
    )
    if estimates.shape != standard_errors.shape or estimates.ndim != 2:
        raise RuntimeError(
            "Detailed roll audit returned inconsistent estimate/SE arrays: "
            f"{estimates.shape} versus {standard_errors.shape}"
        )
    return estimates, standard_errors


def _validate_arguments(arguments: argparse.Namespace) -> None:
    for label, value in (
        ("screen_paths", arguments.screen_paths),
        ("confirmation_paths", arguments.confirmation_paths),
    ):
        if value < 4 or value % 2:
            raise ValueError(f"{label} must be an even integer >=4, got {value}")
    if arguments.confirmation_paths <= arguments.screen_paths:
        raise ValueError("confirmation_paths must exceed screen_paths")
    if arguments.screen_origins_per_batch <= 0 or arguments.origins_per_batch <= 0:
        raise ValueError("origin batch sizes must be positive")
    if not 0.0 < arguments.familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0,1)")


def run(arguments: argparse.Namespace) -> None:
    """Independently confirm screen failures using many fresh Q paths."""
    _validate_arguments(arguments)
    if arguments.device != "cuda:0":
        raise ValueError("The generic Q1 high-path audit is restricted to GPU 0")
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
    serialized_measure_config = json.loads(json.dumps(asdict(measure_config)))
    if serialized_measure_config != q1_report.get("measure_config"):
        raise ValueError("Generic Q1 report geometry differs from current code")
    if (
        not isinstance(q1_checkpoint, Mapping)
        or q1_checkpoint.get("schema") != "flex-weekly-measure-kernel-v1"
        or q1_checkpoint.get("phase") != "complete"
        or q1_checkpoint.get("kernel_class") != "GenericWeeklyMeasureKernel"
        or q1_checkpoint.get("kernel_config") != asdict(measure_config)
        or q1_checkpoint.get("state_sha256") != _sha256_tensor(train_states)
    ):
        raise ValueError("Generic Q1 checkpoint identity is invalid")
    kernel = GenericWeeklyMeasureKernel(
        env,
        dynamics=dynamics,
        config=measure_config,
    )
    kernel.load_state_dict(
        cast(Mapping[str, torch.Tensor], q1_checkpoint["model_state_dict"])
    )
    kernel.eval()
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)

    print(
        f"screening {len(validation_dates)} origins with "
        f"{arguments.screen_paths} paths each",
        flush=True,
    )
    screen = audit_weekly_roll_line(
        kernel,
        simulator,
        validation_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=arguments.screen_paths,
        origins_per_batch=arguments.screen_origins_per_batch,
        seed=arguments.screen_seed,
        include_origin_details=True,
    )
    screen_estimates, screen_standard_errors = _pop_origin_details(screen)
    # The simulator and decoded SANOS surface run in float32. Preserve the
    # production audit's float32 resolution floor after copying diagnostics to
    # float64 NumPy arrays; using float64 epsilon here creates spurious z-scores
    # for essentially zero-priced far-tail calls.
    numerical_standard_error_floor = 32.0 * np.finfo(np.float32).eps
    screen_scale = np.maximum(
        screen_standard_errors,
        numerical_standard_error_floor,
    )
    screen_z = np.abs(screen_estimates) / screen_scale
    screen_failed = screen_z > 3.0
    failed_origin_indices = np.flatnonzero(np.any(screen_failed, axis=1))
    if len(failed_origin_indices) == 0:
        raise RuntimeError(
            "The reproduced screen found no failures, contrary to the source report"
        )
    source_validation = cast(
        Mapping[str, object],
        cast(Mapping[str, object], q1_report["two_week_roll"])[
            "q1_validation_2025"
        ],
    )
    screen_reproduces_source = (
        math.isclose(
            float(screen["conditional_moment_rmse_bps_spot"]),
            float(source_validation["conditional_moment_rmse_bps_spot"]),
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            float(screen["percent_within_3_mc_se"]),
            float(source_validation["percent_within_3_mc_se"]),
            rel_tol=0.0,
            abs_tol=1.0e-4,
        )
    )
    if not screen_reproduces_source:
        raise RuntimeError(
            "The 4096-path screen does not reproduce the source audit: "
            f"RMSE={screen['conditional_moment_rmse_bps_spot']} versus "
            f"{source_validation['conditional_moment_rmse_bps_spot']}, "
            f"within3SE={screen['percent_within_3_mc_se']} versus "
            f"{source_validation['percent_within_3_mc_se']}"
        )

    failed_indices_tensor = torch.as_tensor(
        failed_origin_indices,
        dtype=torch.long,
        device=validation_states.device,
    )
    failed_data = _take_rows(validation_data, failed_indices_tensor)
    print(
        f"confirming {len(failed_origin_indices)} failed origins with "
        f"{arguments.confirmation_paths} independent paths each",
        flush=True,
    )
    confirmation = audit_weekly_roll_line(
        kernel,
        simulator,
        failed_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=arguments.confirmation_paths,
        origins_per_batch=arguments.origins_per_batch,
        seed=arguments.confirmation_seed,
        include_origin_details=True,
    )
    confirmation_estimates, confirmation_standard_errors = _pop_origin_details(
        confirmation
    )
    confirmation_scale = np.maximum(
        confirmation_standard_errors,
        numerical_standard_error_floor,
    )
    confirmation_z = np.abs(confirmation_estimates) / confirmation_scale

    failed_pairs = np.argwhere(screen_failed)
    num_screen_failed_pairs = len(failed_pairs)
    bonferroni_z = NormalDist().inv_cdf(
        1.0 - arguments.familywise_alpha / (2.0 * num_screen_failed_pairs)
    )
    origin_to_confirmation = {
        int(origin): index for index, origin in enumerate(failed_origin_indices)
    }
    pair_details: list[dict[str, object]] = []
    for origin_index_raw, strike_index_raw in failed_pairs:
        origin_index = int(origin_index_raw)
        strike_index = int(strike_index_raw)
        confirmation_index = origin_to_confirmation[origin_index]
        screen_value = screen_estimates[origin_index, strike_index]
        screen_se = screen_standard_errors[origin_index, strike_index]
        confirmation_value = confirmation_estimates[
            confirmation_index, strike_index
        ]
        confirmation_se = confirmation_standard_errors[
            confirmation_index, strike_index
        ]
        z_value = confirmation_z[confirmation_index, strike_index]
        pair_details.append(
            {
                "date": validation_dates[origin_index].isoformat(),
                "origin_index": origin_index,
                "strike_index": strike_index,
                "normalized_strike": float(normalized_strikes[strike_index].cpu()),
                "screen_estimate_bps_spot": float(screen_value * 1.0e4),
                "screen_mc_se_bps_spot": float(screen_se * 1.0e4),
                "screen_abs_z": float(screen_z[origin_index, strike_index]),
                "confirmation_estimate_bps_spot": float(
                    confirmation_value * 1.0e4
                ),
                "confirmation_mc_se_bps_spot": float(confirmation_se * 1.0e4),
                "confirmation_abs_z": float(z_value),
                "confirmation_same_sign": bool(
                    np.sign(screen_value) == np.sign(confirmation_value)
                ),
                "confirmation_exceeds_3se": bool(z_value > 3.0),
                "confirmation_exceeds_5se": bool(z_value > 5.0),
                "confirmation_bonferroni_significant": bool(
                    z_value > bonferroni_z
                ),
            }
        )
    pair_details.sort(
        key=lambda item: float(item["confirmation_abs_z"]), reverse=True
    )
    confirmed_bonferroni = [
        item for item in pair_details if item["confirmation_bonferroni_significant"]
    ]
    confirmed_origin_dates = sorted(
        {str(item["date"]) for item in confirmed_bonferroni}
    )
    per_strike: list[dict[str, object]] = []
    for strike_index in range(len(normalized_strikes)):
        strike_pairs = [
            item for item in pair_details if item["strike_index"] == strike_index
        ]
        per_strike.append(
            {
                "normalized_strike": float(normalized_strikes[strike_index].cpu()),
                "screen_failed_pairs": len(strike_pairs),
                "confirmation_exceeds_3se": sum(
                    bool(item["confirmation_exceeds_3se"])
                    for item in strike_pairs
                ),
                "confirmation_exceeds_5se": sum(
                    bool(item["confirmation_exceeds_5se"])
                    for item in strike_pairs
                ),
                "confirmation_bonferroni_significant": sum(
                    bool(item["confirmation_bonferroni_significant"])
                    for item in strike_pairs
                ),
            }
        )

    report_path = arguments.report_directory / "generic-q1-high-path-arb-audit.json"
    completion_path = (
        arguments.report_directory / "generic-q1-high-path-arb-audit-complete.json"
    )
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": "flex-generic-q1-high-path-arb-audit-v1",
        "status": "complete",
        "scope": (
            "Independent high-path confirmation of 2025 state/strike pairs "
            "outside three Monte Carlo standard errors in the original Q1 audit"
        ),
        "interpretation": (
            "A confirmed nonzero conditional roll moment rejects the candidate "
            "Q law for that historical state and traded-strike restriction. It "
            "does not by itself establish a transaction-cost-adjusted executable "
            "market arbitrage."
        ),
        "sources": {
            **joint_sources,
            "q1_completion": str(arguments.q1_completion),
            "q1_completion_sha256": checkpoint_digest(arguments.q1_completion),
            "q1_report": str(q1_report_path),
            "q1_report_sha256": checkpoint_digest(q1_report_path),
            "q1_checkpoint": str(q1_checkpoint_path),
            "q1_checkpoint_sha256": checkpoint_digest(q1_checkpoint_path),
        },
        "screen": {
            **screen,
            "seed": arguments.screen_seed,
            "reproduces_source_report": screen_reproduces_source,
            "failed_pairs": num_screen_failed_pairs,
            "failed_origins": len(failed_origin_indices),
        },
        "confirmation": {
            **confirmation,
            "seed": arguments.confirmation_seed,
            "tested_pairs": num_screen_failed_pairs,
            "tested_origins": len(failed_origin_indices),
            "independent_of_screen": True,
            "bonferroni_familywise_alpha": arguments.familywise_alpha,
            "bonferroni_two_sided_abs_z_threshold": bonferroni_z,
            "numerical_standard_error_floor_bps_spot": float(
                numerical_standard_error_floor * 1.0e4
            ),
            "pairs_exceeding_3se": sum(
                bool(item["confirmation_exceeds_3se"]) for item in pair_details
            ),
            "pairs_exceeding_5se": sum(
                bool(item["confirmation_exceeds_5se"]) for item in pair_details
            ),
            "pairs_bonferroni_significant": len(confirmed_bonferroni),
            "origins_bonferroni_significant": len(confirmed_origin_dates),
            "bonferroni_significant_dates": confirmed_origin_dates,
            "same_sign_as_screen_pairs": sum(
                bool(item["confirmation_same_sign"]) for item in pair_details
            ),
        },
        "per_strike": per_strike,
        "failed_pair_details": pair_details,
        "conclusion": (
            "candidate_Q_rejected_on_finite_grid"
            if confirmed_bonferroni
            else "no_confirmed_violation_at_familywise_threshold"
        ),
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-generic-q1-high-path-arb-audit-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
        },
        completion_path,
    )
    print(
        f"confirmed {len(confirmed_bonferroni)}/{num_screen_failed_pairs} "
        f"pairs at Bonferroni |z|>{bonferroni_z:.3f}",
        flush=True,
    )


def main() -> None:
    """CLI entry point with durable exception reporting."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_joint_hierarchical_q1_arb_audit")
