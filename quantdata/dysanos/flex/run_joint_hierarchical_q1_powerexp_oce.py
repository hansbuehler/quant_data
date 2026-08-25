"""Evaluate frozen W2 violation directions on a manual power-exp OCE ladder."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
from quantdata.dysanos.flex.powerexp_oce import evaluate_powerexp_oce_grid
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import (
    _load_inputs,
    _sha256_tensor,
)
from quantdata.dysanos.flex.run_joint_hierarchical_q1_sample_arbitrage import (
    _sample_roll_returns,
)
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    _make_residual_function,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment

_DEFAULT_NOTIONALS = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--q1-completion", type=Path, required=True)
    parser.add_argument("--moment-audit-completion", type=Path, required=True)
    parser.add_argument("--sample-arbitrage-completion", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--paths", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=9_120_211)
    parser.add_argument(
        "--notionals",
        default=",".join(str(value) for value in _DEFAULT_NOTIONALS),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _parse_notionals(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if (
        not result
        or result[0] != 0.0
        or any(not np.isfinite(item) or item < 0.0 for item in result)
        or any(right <= left for left, right in zip(result, result[1:], strict=False))
    ):
        raise ValueError(f"Invalid manual notional grid {value!r}")
    return result


def _validate_completion(
    path: Path,
    *,
    schema: str,
) -> tuple[Mapping[str, Any], Path, Mapping[str, Any]]:
    completion = _load_json(path)
    if completion.get("schema") != schema or completion.get("status") != "complete":
        raise ValueError(f"Invalid completion proof {path}")
    report_path = Path(str(completion["report"]))
    if checkpoint_digest(report_path) != completion.get("report_sha256"):
        raise ValueError(f"Completion report provenance is invalid for {path}")
    return completion, report_path, _load_json(report_path)


def _direction_records(
    details: object,
) -> dict[str, list[Mapping[str, Any]]]:
    if not isinstance(details, list):
        raise ValueError("Moment audit has no failed-pair details")
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for raw in details:
        if not isinstance(raw, Mapping) or not raw.get("confirmation_bonferroni_significant"):
            continue
        item = cast(Mapping[str, Any], raw)
        state_date = str(item["date"])
        strike_index = int(item["strike_index"])
        estimate = float(item["confirmation_estimate_bps_spot"])
        key = (state_date, strike_index)
        if key in seen or not np.isfinite(estimate) or estimate == 0.0:
            raise ValueError(f"Invalid or duplicate frozen direction {key}")
        seen.add(key)
        by_date[state_date].append(item)
    if not by_date:
        raise ValueError("Moment audit produced no significant frozen directions")
    for values in by_date.values():
        values.sort(key=lambda item: int(item["strike_index"]))
    return dict(sorted(by_date.items()))


def _summarize_direction(
    *,
    label: str,
    kind: str,
    action: np.ndarray,
    pnl: np.ndarray,
    grid: list[dict[str, Any]],
    column: int,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observations = []
    for item in grid:
        observations.append(
            {
                "notional": float(item["notional"]),
                "translation": float(item["translation"][column]),
                "oce_bps_spot": float(item["oce"][column]) * 1.0e4,
                "oce_standard_error_bps_spot": float(item["oce_standard_error"][column]) * 1.0e4,
                "notional_derivative_bps_spot": float(item["notional_derivative"][column]) * 1.0e4,
                "notional_derivative_standard_error_bps_spot": float(
                    item["notional_derivative_standard_error"][column]
                )
                * 1.0e4,
                "marginal_effective_sample_size": float(
                    item["marginal_effective_sample_size"][column]
                ),
            }
        )
    best = max(range(len(observations)), key=lambda index: observations[index]["oce_bps_spot"])
    return {
        "label": label,
        "kind": kind,
        "action": action.tolist(),
        "source_moment": None if source is None else dict(source),
        "unit_pnl_mean_bps_spot": float(np.mean(pnl)) * 1.0e4,
        "unit_pnl_standard_error_bps_spot": float(np.std(pnl, ddof=1) / np.sqrt(len(pnl))) * 1.0e4,
        "unit_pnl_minimum_bps_spot": float(np.min(pnl)) * 1.0e4,
        "unit_pnl_quantiles_bps_spot": {
            str(quantile): float(np.quantile(pnl, quantile)) * 1.0e4
            for quantile in (0.001, 0.01, 0.05, 0.5)
        },
        "unit_pnl_loss_fraction": float(np.mean(pnl < 0.0)),
        "manual_grid_best_notional": observations[best]["notional"],
        "manual_grid_best_oce_bps_spot": observations[best]["oce_bps_spot"],
        "manual_grid_best_at_upper_boundary": best == len(observations) - 1,
        "manual_grid": observations,
    }


def _aggregate(
    state_results: list[dict[str, Any]],
    notionals: tuple[float, ...],
) -> dict[str, Any]:
    directions = [
        cast(Mapping[str, Any], direction)
        for state in state_results
        for direction in cast(list[dict[str, Any]], state["directions"])
    ]
    result: dict[str, Any] = {}
    for kind in ("single_option", "equal_weight_state_basket"):
        selected = [item for item in directions if item["kind"] == kind]
        by_notional = []
        for index, notional in enumerate(notionals):
            observations = [
                cast(list[Mapping[str, Any]], item["manual_grid"])[index] for item in selected
            ]
            oce = np.array(
                [float(item["oce_bps_spot"]) for item in observations],
                dtype=np.float64,
            )
            derivative = np.array(
                [float(item["notional_derivative_bps_spot"]) for item in observations],
                dtype=np.float64,
            )
            by_notional.append(
                {
                    "notional": notional,
                    "mean_oce_bps_spot": float(np.mean(oce)),
                    "median_oce_bps_spot": float(np.median(oce)),
                    "percent_positive_oce": float(np.mean(oce > 0.0) * 100.0),
                    "percent_positive_notional_derivative": float(
                        np.mean(derivative > 0.0) * 100.0
                    ),
                    "minimum_marginal_effective_sample_size": float(
                        min(item["marginal_effective_sample_size"] for item in observations)
                    ),
                }
            )
        result[kind] = {
            "directions": len(selected),
            "fresh_sample_mean_sign_positive_percent": float(
                np.mean([float(item["unit_pnl_mean_bps_spot"]) > 0.0 for item in selected]) * 100.0
            ),
            "best_grid_notional_at_upper_boundary": sum(
                bool(item["manual_grid_best_at_upper_boundary"]) for item in selected
            ),
            "by_notional": by_notional,
        }
    return result


def run(arguments: argparse.Namespace) -> None:
    """Run the independent frozen-direction manual-notional experiment."""

    notionals = _parse_notionals(arguments.notionals)
    if arguments.device != "cuda:0":
        raise ValueError("The power-exp OCE diagnostic is restricted to GPU 0")
    if arguments.paths < 4 or arguments.paths % 2:
        raise ValueError("paths must be an even integer >=4")

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

    q1_completion, q1_report_path, q1_report = _validate_completion(
        arguments.q1_completion,
        schema="flex-joint-hierarchical-generic-q1-completion-v1",
    )
    q1_checkpoint_path = Path(str(q1_completion["checkpoint"]))
    if checkpoint_digest(q1_checkpoint_path) != q1_completion.get("checkpoint_sha256"):
        raise ValueError("Generic Q1 checkpoint provenance is invalid")
    q1_checkpoint = torch.load(q1_checkpoint_path, map_location="cpu", weights_only=False)
    _, moment_report_path, moment_report = _validate_completion(
        arguments.moment_audit_completion,
        schema="flex-generic-q1-high-path-arb-audit-completion-v1",
    )
    _, sample_report_path, sample_report = _validate_completion(
        arguments.sample_arbitrage_completion,
        schema="flex-generic-q1-sample-arbitrage-completion-v1",
    )
    if sample_report.get("conclusion") != "no_strict_arbitrage_on_sampled_successor_values":
        raise ValueError("Strict sample-arbitrage predecessor has unexpected conclusion")

    directions_by_date = _direction_records(moment_report.get("failed_pair_details"))
    date_to_index = {value.isoformat(): index for index, value in enumerate(validation_dates)}
    if any(state_date not in date_to_index for state_date in directions_by_date):
        raise ValueError("Frozen direction dates do not match the validation states")

    spot_fit = fit_joint_spot_correlation_under_p(
        model,
        train_states.detach().cpu().double().numpy(),
        train_log_spot,
        train_dates,
    )
    dynamics = JointHierarchicalWeeklyDynamicsAdapter(
        env,
        model=model,
        spot_front_beta=torch.as_tensor(spot_fit.front_beta, dtype=env.fdtype, device=env.device),
        spot_orthogonal_std=spot_fit.orthogonal_std,
    )
    simulator = WeeklySanosSimulator(env, dynamics=dynamics, calibrated_model=model)
    log_moneyness = tuple(float(value) for value in np.linspace(-0.30, 0.30, 11))
    normalized_strikes = torch.exp(torch.tensor(log_moneyness, dtype=env.fdtype, device=env.device))
    validation_data = precompute_measure_training_data(
        simulator, validation_states, normalized_strikes, num_expiries=2
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
        or json.loads(json.dumps(asdict(measure_config))) != q1_report.get("measure_config")
    ):
        raise ValueError("Generic Q1 checkpoint identity is invalid")
    kernel = GenericWeeklyMeasureKernel(env, dynamics=dynamics, config=measure_config)
    kernel.load_state_dict(cast(Mapping[str, torch.Tensor], q1_checkpoint["model_state_dict"]))
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

    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    progress_path = arguments.report_directory / "generic-q1-powerexp-oce-progress.json"
    run_identity = {
        "joint_completion_sha256": checkpoint_digest(arguments.joint_completion),
        "q1_completion_sha256": checkpoint_digest(arguments.q1_completion),
        "moment_audit_completion_sha256": checkpoint_digest(arguments.moment_audit_completion),
        "sample_arbitrage_completion_sha256": checkpoint_digest(
            arguments.sample_arbitrage_completion
        ),
        "paths": arguments.paths,
        "seed": arguments.seed,
        "notionals": list(notionals),
        "dates": list(directions_by_date),
    }
    state_results: list[dict[str, Any]] = []
    if progress_path.exists():
        progress = _load_json(progress_path)
        if (
            progress.get("schema") != "flex-generic-q1-powerexp-oce-progress-v1"
            or progress.get("run_identity") != run_identity
            or not isinstance(progress.get("state_results"), list)
        ):
            raise ValueError("Existing power-exp progress has invalid identity")
        state_results = list(cast(list[dict[str, Any]], progress["state_results"]))
        completed_dates = [str(item.get("date")) for item in state_results]
        expected_dates = list(directions_by_date)[: len(completed_dates)]
        if completed_dates != expected_dates:
            raise ValueError("Existing power-exp progress is not a valid prefix")
        print(f"resuming after {len(state_results)} completed states", flush=True)

    pending_dates = list(directions_by_date)[len(state_results) :]
    for position, state_date in enumerate(pending_dates, start=len(state_results) + 1):
        origin_index = date_to_index[state_date]
        sampled_returns = _sample_roll_returns(
            kernel,
            residual_function,
            validation_data,
            origin_index=origin_index,
            paths=arguments.paths,
            seed=arguments.seed + origin_index,
        )
        source_directions = directions_by_date[state_date]
        actions = np.zeros((sampled_returns.shape[1], len(source_directions) + 1))
        labels: list[str] = []
        for column, source in enumerate(source_directions):
            strike_index = int(source["strike_index"])
            sign = float(np.sign(float(source["confirmation_estimate_bps_spot"])))
            actions[1 + strike_index, column] = sign
            labels.append(f"call_k_{float(normalized_strikes[strike_index]):.9g}")
        actions[:, -1] = np.sum(actions[:, :-1], axis=1) / len(source_directions)
        labels.append("equal_weight_state_basket")
        pnl = sampled_returns @ actions
        grid = evaluate_powerexp_oce_grid(pnl, notionals)
        maximum_foc_residual = max(
            float(item["maximum_translation_first_order_residual"]) for item in grid
        )
        if maximum_foc_residual > 1.0e-10:
            raise RuntimeError(
                f"Power-exp translation failed at {state_date}: {maximum_foc_residual}"
            )
        direction_results = []
        for column, source in enumerate(source_directions):
            direction_results.append(
                _summarize_direction(
                    label=labels[column],
                    kind="single_option",
                    action=actions[:, column],
                    pnl=pnl[:, column],
                    grid=grid,
                    column=column,
                    source=source,
                )
            )
        direction_results.append(
            _summarize_direction(
                label=labels[-1],
                kind="equal_weight_state_basket",
                action=actions[:, -1],
                pnl=pnl[:, -1],
                grid=grid,
                column=pnl.shape[1] - 1,
                source=None,
            )
        )
        state_results.append(
            {
                "date": state_date,
                "origin_index": origin_index,
                "directions": direction_results,
                "maximum_translation_first_order_residual": maximum_foc_residual,
            }
        )
        write_json_atomic(
            {
                "schema": "flex-generic-q1-powerexp-oce-progress-v1",
                "status": "running",
                "run_identity": run_identity,
                "completed_states": len(state_results),
                "state_results": state_results,
            },
            progress_path,
        )
        print(
            f"powerexp {position}/{len(directions_by_date)} date={state_date} "
            f"directions={len(source_directions)}",
            flush=True,
        )

    aggregate = _aggregate(state_results, notionals)
    report_path = arguments.report_directory / "generic-q1-powerexp-oce.json"
    completion_path = arguments.report_directory / "generic-q1-powerexp-oce-complete.json"
    report = {
        "schema": "flex-generic-q1-powerexp-oce-v1",
        "status": "complete",
        "objective": "sup_y E[u_powerexp(y + H*pi)] - y",
        "utility": "1-exp(-z) for z>0; z-z^2/2 for z<=0",
        "direction_policy": (
            "For each independently confirmed W2 moment violation, hold one call "
            "with the frozen profitable sign. Also test the date's equal-L1 signed "
            "basket. No direction or notional is fitted on evaluation paths."
        ),
        "manual_notionals": list(notionals),
        "paths_per_state": arguments.paths,
        "sampling": "fresh origin-wise antithetic Torch normals",
        "seed": arguments.seed,
        "transaction_costs": False,
        "scope_warning": (
            "This values risky frozen directions under the rejected candidate-Q "
            "scenario law. A positive finite-notional OCE is not a strict-arbitrage "
            "certificate; strict finite-sample separation was tested separately."
        ),
        "aggregate": aggregate,
        "sources": {
            **joint_sources,
            "q1_completion": str(arguments.q1_completion),
            "q1_completion_sha256": checkpoint_digest(arguments.q1_completion),
            "moment_audit_report": str(moment_report_path),
            "moment_audit_report_sha256": checkpoint_digest(moment_report_path),
            "sample_arbitrage_report": str(sample_report_path),
            "sample_arbitrage_report_sha256": checkpoint_digest(sample_report_path),
        },
        "state_results": state_results,
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-generic-q1-powerexp-oce-completion-v1",
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
    run_guarded_cli(main, program="run_joint_hierarchical_q1_powerexp_oce")
