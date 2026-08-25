"""Reprice ideal triangular Global-state reconstructions on FLEX quotes."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.dysanos.flex.analyze_state_allocation import (
    _triangular_rrr_residual_coefficients,
)
from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    ForwardMuFlexGeometryConfig,
    NativeStateConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import FlexMarketDay, market_process_count
from quantdata.dysanos.flex.model import NativeStateSanosModel
from quantdata.dysanos.flex.run_production_gate import (
    _load_market_job,
    _market_cache_geometry,
)
from quantdata.dysanos.flex.training import evaluate_model, materialize_day
from quantdata.pool import PoolConfig
from quantdata.torch import BaseEnvironment


def _nearest_model_grid_statistics(
    fits: tuple[np.ndarray, ...],
    market_days: list[FlexMarketDay],
    *,
    geometry: ForwardMuFlexGeometryConfig,
) -> list[dict[str, int | float]]:
    fit = np.concatenate(fits)
    bids = np.concatenate([day.bids for day in market_days])
    asks = np.concatenate([day.asks for day in market_days])
    business_dtes = np.concatenate(
        [day.quote_business_dtes for day in market_days]
    )
    model_dtes = geometry.model_business_dtes
    nearest = np.argmin(
        np.abs(business_dtes[:, None] - model_dtes[None, :]),
        axis=1,
    )
    half_spread = 0.5 * (asks - bids)
    excess = (
        np.maximum(fit - asks, 0.0) + np.maximum(bids - fit, 0.0)
    ) / half_spread
    rows: list[dict[str, int | float]] = []
    for expiry_index, model_dte in enumerate(model_dtes):
        selected = excess[nearest == expiry_index]
        positive = selected[selected > 0.0]
        rows.append(
            {
                "expiry_index": expiry_index,
                "model_business_dte": float(model_dte),
                "num_options": int(len(selected)),
                "percent_zero": (
                    float(100.0 * np.mean(selected == 0.0))
                    if len(selected)
                    else 0.0
                ),
                "positive_mean": (
                    float(np.mean(positive)) if len(positive) else 0.0
                ),
                "positive_median": (
                    float(np.median(positive)) if len(positive) else 0.0
                ),
            }
        )
    return rows


def _geometry_from_report(payload: dict[str, Any]) -> ForwardMuFlexGeometryConfig:
    values = dict(payload["geometry"])
    values["nms_range"] = tuple(values["nms_range"])
    explicit_dtes = values.get("explicit_model_business_dtes")
    if explicit_dtes is not None:
        values["explicit_model_business_dtes"] = tuple(explicit_dtes)
    return ForwardMuFlexGeometryConfig(**values)


def _candidate_raw_states(
    checkpoint: dict[str, Any],
    *,
    allocation: tuple[int, ...] | None,
    block_width: int,
) -> torch.Tensor:
    state = checkpoint["model_state_dict"]
    daily = state["daily_state_embedding.weight"].double()
    weight = state["raw_decoder.I0.weight"].double()
    bias = state["raw_decoder.I0.bias"].double()
    global_raw = daily @ weight.transpose(0, 1) + bias
    if allocation is None:
        return global_raw

    mean = torch.mean(global_raw, dim=0)
    centered = global_raw - mean
    left, singular_values, right_t = torch.linalg.svd(
        centered,
        full_matrices=False,
    )
    global_rank = daily.shape[1]
    coefficients = singular_values[:global_rank].unsqueeze(1) * right_t[
        :global_rank
    ]
    residual, selected = _triangular_rrr_residual_coefficients(
        coefficients,
        allocation=allocation,
        block_width=block_width,
    )
    if selected != allocation:
        raise RuntimeError(
            f"Allocation contains numerically inactive states: {allocation} -> {selected}"
        )
    reconstruction = coefficients - residual
    return mean + left[:, :global_rank] @ reconstruction


def _decoded_checkpoint_raw_states(checkpoint: dict[str, Any]) -> torch.Tensor:
    state = checkpoint["model_state_dict"]
    daily = state["daily_state_embedding.weight"].double()
    weight = state["raw_decoder.I0.weight"].double()
    mask = state.get("decoder_weight_mask")
    if mask is not None:
        weight = weight * mask.double()
    bias = state["raw_decoder.I0.bias"].double()
    return daily @ weight.transpose(0, 1) + bias


def evaluate_allocations(
    *,
    market_report: Path,
    allocation_report: Path | None,
    explicit_allocations: tuple[tuple[int, ...], ...],
    comparison_checkpoints: tuple[Path, ...],
    global_checkpoint: Path,
    workers: int,
    device: str,
) -> dict[str, Any]:
    """Load the certified market days and exactly reprice each allocation."""
    market_payload = json.loads(market_report.read_text(encoding="utf-8"))
    allocation_payload = (
        None
        if allocation_report is None
        else json.loads(allocation_report.read_text(encoding="utf-8"))
    )
    geometry = _geometry_from_report(market_payload)
    market_config = FlexMarketConfig(**market_payload["market_config"])
    objective_config = FlexObjectiveConfig(**market_payload["objective_config"])
    dates = tuple(date.fromisoformat(value) for value in market_payload["market_dates"])
    checkpoint = torch.load(
        global_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("phase") != "complete":
        raise ValueError(f"Expected a complete Global checkpoint: {global_checkpoint}")
    global_checkpoint_state_size = int(
        checkpoint["model_state_dict"]["daily_state_embedding.weight"].shape[1]
    )

    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()
    cache_geometry = _market_cache_geometry(geometry)
    jobs = {
        trading_date: pool.delayed(_load_market_job)(
            market_payload["ticker"],
            trading_date,
            geometry=cache_geometry,
            market_config=market_config,
        )
        for trading_date in dates
    }
    market_days = []
    for index, result in enumerate(pool.parallel(jobs)):
        trading_date, outcome = result
        if outcome.error is not None or outcome.day is None:
            raise RuntimeError(
                f"Could not reload certified market day {trading_date}: "
                f"{outcome.error or 'missing'}"
            )
        market_days.append(outcome.day)
        if (index + 1) % 100 == 0 or index + 1 == len(dates):
            print(f"market {index + 1}/{len(dates)}", flush=True)
    market_days.sort(key=lambda value: value.trading_date)
    expected_fingerprints = tuple(checkpoint["market_fingerprints"])
    actual_fingerprints = tuple(
        day.training_fingerprint for day in market_days
    )
    if actual_fingerprints != expected_fingerprints:
        raise RuntimeError("Reloaded market days do not match the Global checkpoint")

    env = BaseEnvironment(precision=32, device=device, debug=False)
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    model = NativeStateSanosModel(
        env,
        num_days=len(market_days),
        config=NativeStateConfig(
            label="state-allocation-quote-evaluation",
            geometry=geometry,
        ),
    )
    model.sanos_decoder.enable_solver_compile()
    block_width = geometry.expiry_block_state_size
    candidate_allocations = list(explicit_allocations)
    if allocation_payload is not None:
        candidate_allocations.extend(
            tuple(candidate["optimized_allocation"])
            for candidate in allocation_payload[
                "ideal_global_surface_triangular_allocations"
            ]
        )
    candidate_allocations = list(dict.fromkeys(candidate_allocations))
    for allocation in candidate_allocations:
        if len(allocation) != geometry.num_model_expiries:
            raise ValueError(
                "Every allocation must have one value per expiry, found "
                f"{allocation}"
            )
        if any(value < 0 for value in allocation):
            raise ValueError(f"Allocation values cannot be negative: {allocation}")
        if sum(allocation) > global_checkpoint_state_size:
            raise ValueError(
                f"Allocation exceeds Global state size: {allocation}"
            )
    candidates: list[tuple[str, torch.Tensor, tuple[int, ...] | None]] = [
        (
            "global_exact",
            _decoded_checkpoint_raw_states(checkpoint),
            None,
        ),
        *[
            (
                "triangular_" + "_".join(str(value) for value in allocation),
                _candidate_raw_states(
                    checkpoint,
                    allocation=allocation,
                    block_width=block_width,
                ),
                allocation,
            )
            for allocation in candidate_allocations
        ],
    ]
    for comparison_path in comparison_checkpoints:
        comparison = torch.load(
            comparison_path,
            map_location="cpu",
            weights_only=False,
        )
        if comparison.get("phase") != "complete":
            raise ValueError(
                f"Expected a complete comparison checkpoint: {comparison_path}"
            )
        if tuple(comparison["market_fingerprints"]) != expected_fingerprints:
            raise RuntimeError(
                f"Comparison checkpoint market mismatch: {comparison_path}"
            )
        candidates.append(
            (
                f"checkpoint_{comparison_path.stem}",
                _decoded_checkpoint_raw_states(comparison),
                None,
            )
        )
    evaluations: list[dict[str, Any]] = []
    for label, raw, allocation in candidates:
        with torch.no_grad():
            model.daily_raw_states.copy_(
                raw.to(dtype=env.fdtype, device=env.device)
            )
        evaluation = evaluate_model(
            model,
            torch_days,
            env=env,
            objective_config=objective_config,
            day_batch_size=16,
        )
        print(f"evaluation {label} objective={evaluation.objective:.8f}", flush=True)
        evaluations.append(
            {
                "label": label,
                "allocation": None if allocation is None else list(allocation),
                **evaluation.to_dict(),
                "nearest_model_grid_statistics": _nearest_model_grid_statistics(
                    evaluation.fits,
                    market_days,
                    geometry=geometry,
                ),
            }
        )
    return {
        "schema": "flex-expiry-state-allocation-quote-evaluation-v1",
        "market_report": str(market_report),
        "allocation_report": (
            None if allocation_report is None else str(allocation_report)
        ),
        "global_checkpoint": str(global_checkpoint),
        "num_days": len(market_days),
        "num_quotes": sum(day.num_quotes for day in market_days),
        "evaluations": evaluations,
    }


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = [
        "# FLEX expiry-state allocation quote evaluation",
        "",
        f"Days: `{report['num_days']}`. Quotes: `{report['num_quotes']}`.",
        "",
        "| Candidate | Objective | Inside bid/ask | Conditional mean excess | Conditional median excess |",
        "|---|---:|---:|---:|---:|",
    ]
    for evaluation in report["evaluations"]:
        statistics = evaluation["statistics"]["all"]
        rows.append(
            f"| {evaluation['label']} | {evaluation['objective']:.8f} | "
            f"{statistics['percent_zero']:.4f}% | "
            f"{statistics['positive_mean']:.6f} | "
            f"{statistics['positive_median']:.6f} |"
        )
    output.with_suffix(".md").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-report", type=Path, required=True)
    parser.add_argument("--allocation-report", type=Path)
    parser.add_argument(
        "--allocations",
        nargs="*",
        default=[],
        help="Comma-separated state counts, one count per expiry.",
    )
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--comparison-checkpoints",
        type=Path,
        nargs="*",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    explicit_allocations = tuple(
        tuple(int(value) for value in allocation.split(","))
        for allocation in args.allocations
    )
    if (
        args.allocation_report is None
        and len(explicit_allocations) == 0
        and len(args.comparison_checkpoints) == 0
    ):
        raise ValueError(
            "Provide --allocation-report, --allocations, or a comparison checkpoint"
        )
    report = evaluate_allocations(
        market_report=args.market_report,
        allocation_report=args.allocation_report,
        explicit_allocations=explicit_allocations,
        comparison_checkpoints=tuple(args.comparison_checkpoints),
        global_checkpoint=args.global_checkpoint,
        workers=args.workers,
        device=args.device,
    )
    _write_report(report, args.output)
    print(args.output)


if __name__ == "__main__":
    run_guarded_cli(main, program="evaluate_state_allocations")
