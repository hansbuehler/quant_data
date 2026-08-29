"""Fit one generic conditional front-state transport across all FLEX history."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_all_state_vector_q_existence import (
    CommonQDraw,
    normalize_front_components,
)
from quantdata.dysanos.flex.run_atm_q_existence import (
    _atomic_torch_save,
    _load_inputs,
)
from quantdata.dysanos.flex.run_six_state_joint_generic_q_existence import (
    JointGenericFrontTransport,
    JointGenericTransportConfig,
    _evaluate_joint,
    joint_roll_returns,
)
from quantdata.dysanos.flex.run_state_generic_q_existence import (
    _summary as generic_state_summary,
)
from quantdata.torch import BaseEnvironment

OBJECTIVE_ID = "all-state-joint-generic-continuous-eleven-option-tail-excess-v4"
OPTIMIZER_UPDATE_ID = "one-global-nested-tail-excess-gradient-step-per-sweep-v2"
DRAW_POLICY_ID = "iid-joint-spot-surface-paths-v1"
DEFAULT_INNER_TAIL_FRACTION = 0.01
DEFAULT_OUTER_TAIL_FRACTION = 0.10
DEFAULT_INNER_TAIL_WEIGHT = 0.50
_SHARED_STRIKES_TRAINING_ROOT = Path(quantdata_env.locations.training.path) / "flex-shared-strikes"
_DEFAULT_JOINT_COMPLETION = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "joint-hierarchical"
    / "v1"
    / "reports"
    / "joint-hierarchical-200-complete.json"
)
_DEFAULT_TEACHER = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "generic-joint-six-v1"
    / "checkpoints"
    / "joint-generic-q.pt"
)
_DEFAULT_AFFINE_TEACHER = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "vector-all-history-exact-common-random-statewise-cv-v2"
    / "snapshots"
    / "all-state-vector-q-epoch351.pt"
)
_DEFAULT_OUTPUT_ROOT = (
    _SHARED_STRIKES_TRAINING_ROOT / "q-existence" / "generic-joint-all-history-affine-init-v1"
)


def _state_batches(num_states: int, state_batch_size: int) -> list[slice]:
    return [
        slice(start, min(start + state_batch_size, num_states))
        for start in range(0, num_states, state_batch_size)
    ]


def _make_joint_path_draw(
    *,
    front_state_size: int,
    num_paths: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CommonQDraw:
    """Draw one independent spot and one independent surface state per path."""

    if front_state_size <= 0 or num_paths < 2:
        raise ValueError("a joint draw needs positive state width and at least two paths")
    generator = torch.Generator(device=device).manual_seed(seed)
    floor = 4.0 * torch.finfo(dtype).eps
    component_uniform = torch.rand(
        num_paths,
        dtype=dtype,
        device=device,
        generator=generator,
    ).clamp(floor, 1.0 - floor)
    backbone_uniform = torch.rand(
        num_paths,
        dtype=dtype,
        device=device,
        generator=generator,
    ).clamp(floor, 1.0 - floor)
    epsilon = torch.randn(
        (num_paths, front_state_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return CommonQDraw(
        component_uniform=component_uniform,
        backbone_uniform=backbone_uniform,
        branch=torch.zeros(num_paths, dtype=dtype, device=device),
        epsilon=epsilon,
        spot_contexts=num_paths,
        surface_paths=1,
    )


def _joint_path_blocks(
    *,
    env: BaseEnvironment,
    front_state_size: int,
    total_paths: int,
    blocks: int,
    seed: int,
) -> list[CommonQDraw]:
    """Partition genuine joint paths into independently seeded equal blocks."""

    if blocks < 2 or total_paths % blocks:
        raise ValueError("joint path counts must divide evenly into at least two blocks")
    paths_per_block = total_paths // blocks
    if paths_per_block < 2:
        raise ValueError("each joint-path block needs at least two paths")
    return [
        _make_joint_path_draw(
            front_state_size=front_state_size,
            num_paths=paths_per_block,
            seed=seed + block,
            dtype=env.fdtype,
            device=env.device,
        )
        for block in range(blocks)
    ]


def _replace_draw_quarter(
    draw: CommonQDraw,
    replacement: CommonQDraw,
    *,
    quarter: int,
) -> CommonQDraw:
    """Replace one context quarter while preserving one common balanced pool."""

    if (
        quarter not in range(4)
        or draw.spot_contexts % 4
        or draw.spot_contexts != replacement.spot_contexts
        or draw.surface_paths != replacement.surface_paths
        or draw.epsilon.shape != replacement.epsilon.shape
    ):
        raise ValueError("quarter draw replacement has incompatible inputs")
    contexts = draw.spot_contexts
    paths = draw.surface_paths
    start = quarter * contexts // 4
    stop = (quarter + 1) * contexts // 4
    context_slice = slice(start, stop)
    path_slice = slice(start * paths, stop * paths)
    component_uniform = draw.component_uniform.clone()
    backbone_uniform = draw.backbone_uniform.clone()
    branch = draw.branch.clone()
    epsilon = draw.epsilon.clone()
    component_uniform[context_slice] = replacement.component_uniform[context_slice]
    backbone_uniform[context_slice] = replacement.backbone_uniform[context_slice]
    branch[path_slice] = replacement.branch[path_slice]
    epsilon[path_slice] = replacement.epsilon[path_slice]
    return CommonQDraw(
        component_uniform=component_uniform,
        backbone_uniform=backbone_uniform,
        branch=branch,
        epsilon=epsilon,
        spot_contexts=contexts,
        surface_paths=paths,
    )


def _refresh_draw_pool_quarter(
    draws: list[CommonQDraw],
    replacements: list[CommonQDraw],
    *,
    quarter: int,
) -> list[CommonQDraw]:
    """Replace exactly one quarter of every replication in a draw pool."""

    if len(draws) != len(replacements):
        raise ValueError("draw-pool replacement needs matching replication counts")
    return [
        _replace_draw_quarter(draw, replacement, quarter=quarter)
        for draw, replacement in zip(draws, replacements, strict=True)
    ]


def _bootstrap_replication_counts(
    *,
    replications: int,
    samples: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw block-bootstrap multiplicities shared by all states."""

    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randint(
        replications,
        (samples, replications),
        device=device,
        generator=generator,
    )
    counts = torch.zeros((samples, replications), dtype=torch.float64, device=device)
    counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))
    return counts


def _bootstrap_means(
    values: Mapping[str, torch.Tensor],
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap raw/adjusted means while re-estimating beta in every sample."""

    path_sums = values["path_sums_by_replication"].double()
    spot_sums = values["spot_sums_by_replication"].double()
    spot_square_sums = values["spot_square_sums_by_replication"].double()
    product_sums = values["product_sums_by_replication"].double()
    paths_per_replication = values["paths_per_replication"].double()
    if len(path_sums) != counts.shape[1]:
        raise ValueError("bootstrap counts do not match audit replications")
    path_sum = torch.einsum("sr,rbi->sbi", counts, path_sums)
    spot_sum = torch.einsum("sr,rb->sb", counts, spot_sums)
    spot_square_sum = torch.einsum("sr,rb->sb", counts, spot_square_sums)
    product_sum = torch.einsum("sr,rbi->sbi", counts, product_sums)
    num_paths = torch.einsum("sr,r->s", counts, paths_per_replication)
    raw_mean = path_sum / num_paths[:, None, None]
    spot_mean = spot_sum / num_paths[:, None]
    denominator = (spot_square_sum - spot_sum.square() / num_paths[:, None]).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    numerator = product_sum - spot_sum[..., None] * path_sum / num_paths[:, None, None]
    beta = numerator / denominator[..., None]
    adjusted_mean = raw_mean - beta * spot_mean[..., None]
    return raw_mean, adjusted_mean


def _replication_mean_statistics(values: torch.Tensor) -> dict[str, Any]:
    """Summarize non-overlapping block means instrument by instrument."""

    if values.ndim != 2 or len(values) < 2:
        raise ValueError("replication statistics need at least two blocks by instrument")
    values = values.detach().double()
    return {
        "block_means_bps_spot": (1.0e4 * values).tolist(),
        "mean_bps_spot": (1.0e4 * torch.mean(values, dim=0)).tolist(),
        "median_bps_spot": (1.0e4 * torch.quantile(values, 0.5, dim=0)).tolist(),
        "mc_standard_error_bps_spot": (
            1.0e4 * torch.std(values, dim=0, correction=1) / math.sqrt(len(values))
        ).tolist(),
    }


def _scalar_block_statistics(values: torch.Tensor) -> dict[str, Any]:
    """Return the values, mean, median, and standard error of block statistics."""

    values = values.detach().double().reshape(-1)
    if len(values) < 2:
        raise ValueError("block statistics need at least two values")
    return {
        "values": values.tolist(),
        "mean": float(torch.mean(values)),
        "median": float(torch.quantile(values, 0.5)),
        "mc_standard_error": float(
            torch.std(values, correction=1) / math.sqrt(len(values))
        ),
    }


def _tail_excess_loss_and_derivative(
    scaled: torch.Tensor,
    *,
    z_target: float,
    inner_tail_fraction: float,
    outer_tail_fraction: float,
    inner_tail_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average squared target excess over nested inner and outer tails."""

    if scaled.ndim < 1 or scaled.shape[-1] < 1:
        raise ValueError("tail objective needs at least one constraint")
    if (
        z_target <= 0.0
        or not 0.0 < inner_tail_fraction <= outer_tail_fraction <= 1.0
        or not 0.0 <= inner_tail_weight <= 1.0
    ):
        raise ValueError("invalid tail objective configuration")
    num_constraints = scaled.shape[-1]
    inner_count = max(1, math.ceil(inner_tail_fraction * num_constraints))
    outer_count = max(1, math.ceil(outer_tail_fraction * num_constraints))
    absolute = torch.abs(scaled)
    excess = torch.clamp_min(absolute - z_target, 0.0)
    coefficient = torch.zeros_like(scaled)
    inner_indices = torch.topk(absolute, inner_count, dim=-1).indices
    outer_indices = torch.topk(absolute, outer_count, dim=-1).indices
    coefficient.scatter_add_(
        -1,
        inner_indices,
        torch.full_like(
            inner_indices,
            inner_tail_weight / inner_count,
            dtype=scaled.dtype,
        ),
    )
    coefficient.scatter_add_(
        -1,
        outer_indices,
        torch.full_like(
            outer_indices,
            (1.0 - inner_tail_weight) / outer_count,
            dtype=scaled.dtype,
        ),
    )
    loss = torch.sum(coefficient * excess.square(), dim=-1)
    derivative = 2.0 * coefficient * torch.sign(scaled) * excess
    return loss, derivative


def _wandb_training_metrics(record: Mapping[str, Any]) -> dict[str, float | int]:
    """Translate one persisted sweep record into stable W&B metric names."""

    sweep = int(record["sweep"])
    names = {
        "fit_global_loss": "training/global_loss",
        "fit_global_loss_ema": "training/global_loss_ema",
        "fit_maximum_abs_raw_mean_z": "training/raw_worst",
        "fit_worst_10_percent_mean_abs_raw_mean_z": "training/raw_worst_10pct",
        "fit_maximum_abs_noise_scaled_adjusted_mean_z": "training/adjusted_worst",
        "fit_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": (
            "training/adjusted_worst_10pct"
        ),
        "fit_gradient_norm": "training/gradient_norm",
        "validation_global_loss": "validation/full_100k_global_loss",
        "validation_block_loss_mean": "validation/block_loss_mean",
        "validation_block_loss_median": "validation/block_loss_median",
        "validation_block_loss_mc_standard_error": "validation/block_loss_mc_se",
        "validation_maximum_abs_raw_mean_z": "validation/raw_worst",
        "validation_worst_10_percent_mean_abs_raw_mean_z": (
            "validation/raw_worst_10pct"
        ),
        "validation_maximum_abs_noise_scaled_adjusted_mean_z": (
            "validation/adjusted_worst"
        ),
        "validation_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": (
            "validation/adjusted_worst_10pct"
        ),
        "learning_rate": "optimizer/learning_rate",
        "sweep_seconds": "performance/sweep_seconds",
        "validation_audit_seconds": "performance/validation_seconds",
    }
    metrics: dict[str, float | int] = {"sweep": sweep}
    for source, target in names.items():
        if source in record:
            metrics[target] = float(record[source])
    block_values = tuple(record.get("validation_block_global_losses", ()))
    for index, value in enumerate(block_values):
        metrics[f"validation/block_loss_{index:02d}"] = float(value)
    if block_values:
        blocks = torch.tensor(block_values, dtype=torch.float64)
        for quantile, label in (
            (0.10, "q10"),
            (0.25, "q25"),
            (0.75, "q75"),
            (0.90, "q90"),
        ):
            metrics[f"validation/block_loss_{label}"] = float(
                torch.quantile(blocks, quantile)
            )
    return metrics


def _initialize_wandb(
    *,
    arguments: argparse.Namespace,
    law_config: JointGenericTransportConfig,
    simulation_config: Mapping[str, Any],
    history: list[dict[str, Any]],
) -> Any | None:
    """Start or resume optional hosted metric tracking and backfill checkpoints."""

    if arguments.wandb_project is None:
        return None
    import wandb

    run = wandb.init(
        project=arguments.wandb_project,
        entity=arguments.wandb_entity,
        name=arguments.wandb_run_name,
        id=arguments.wandb_run_id,
        resume="allow" if arguments.wandb_run_id is not None else None,
        mode=arguments.wandb_mode,
        tags=arguments.wandb_tags,
        config={
            "objective": OBJECTIVE_ID,
            "optimizer_update": OPTIMIZER_UPDATE_ID,
            "law": asdict(law_config),
            "simulation": dict(simulation_config),
        },
    )
    run.define_metric("sweep")
    run.define_metric("*", step_metric="sweep")
    starting_step = int(run.starting_step)
    for record in history:
        if int(record["sweep"]) >= starting_step:
            run.log(
                _wandb_training_metrics(record),
                step=int(record["sweep"]),
                commit=True,
            )
    return run


def _log_wandb_sweep(run: Any | None, record: Mapping[str, Any]) -> None:
    """Log one checkpointed sweep without making telemetry training-critical."""

    if run is None:
        return
    try:
        run.log(
            _wandb_training_metrics(record),
            step=int(record["sweep"]),
            commit=True,
        )
    except Exception as error:  # pragma: no cover - depends on remote availability
        print(f"W&B logging warning: {type(error).__name__}: {error}", flush=True)


def _block_objective_summary(
    values_by_batch: list[Mapping[str, torch.Tensor]],
    *,
    z_target: float,
    inner_tail_fraction: float = DEFAULT_INNER_TAIL_FRACTION,
    outer_tail_fraction: float = DEFAULT_OUTER_TAIL_FRACTION,
    inner_tail_weight: float = DEFAULT_INNER_TAIL_WEIGHT,
) -> dict[str, Any]:
    """Evaluate the global criterion separately on every non-overlapping block."""

    if not values_by_batch:
        raise ValueError("block objective needs at least one state batch")
    scaled_by_batch: list[torch.Tensor] = []
    raw_z_by_batch: list[torch.Tensor] = []
    expected_blocks: int | None = None
    expected_paths: torch.Tensor | None = None
    for values in values_by_batch:
        path_sum = values["path_sums_by_replication"].double()
        path_square_sum = values["path_square_sums_by_replication"].double()
        spot_sum = values["spot_sums_by_replication"].double()
        spot_square_sum = values["spot_square_sums_by_replication"].double()
        paths = values["paths_per_replication"].double()
        if (
            path_sum.ndim != 3
            or path_square_sum.shape != path_sum.shape
            or spot_sum.shape != path_sum.shape[:2]
            or spot_square_sum.shape != spot_sum.shape
            or paths.shape != path_sum.shape[:1]
            or torch.any(paths < 2)
        ):
            raise ValueError("block objective received inconsistent sufficient statistics")
        if expected_blocks is None:
            expected_blocks = len(paths)
            expected_paths = paths
        elif expected_blocks != len(paths) or not torch.equal(expected_paths, paths):
            raise ValueError("state batches use inconsistent validation blocks")
        count = paths[:, None, None]
        raw_mean = path_sum / count
        raw_variance = (
            path_square_sum - path_sum.square() / count
        ) / (count - 1.0)
        raw_se = torch.sqrt(torch.clamp_min(raw_variance, 0.0) / count)
        spot_count = paths[:, None]
        spot_mean = spot_sum / spot_count
        spot_variance = (
            spot_square_sum - spot_sum.square() / spot_count
        ) / (spot_count - 1.0)
        spot_se = torch.sqrt(torch.clamp_min(spot_variance, 0.0) / spot_count)
        beta = values["beta"].double()[None]
        adjusted_mean = raw_mean - beta * spot_mean[..., None]
        noise_scale = torch.maximum(torch.abs(beta) * spot_se[..., None], raw_se)
        scaled_by_batch.append(
            (adjusted_mean / noise_scale.clamp_min(1.0e-12)).reshape(len(paths), -1)
        )
        raw_z_by_batch.append(
            (raw_mean / raw_se.clamp_min(1.0e-12)).reshape(len(paths), -1)
        )
    assert expected_blocks is not None and expected_paths is not None
    scaled = torch.cat(scaled_by_batch, dim=1)
    raw_z = torch.cat(raw_z_by_batch, dim=1)
    loss, _ = _tail_excess_loss_and_derivative(
        scaled,
        z_target=z_target,
        inner_tail_fraction=inner_tail_fraction,
        outer_tail_fraction=outer_tail_fraction,
        inner_tail_weight=inner_tail_weight,
    )
    absolute_adjusted = torch.abs(scaled)
    absolute_raw = torch.abs(raw_z)
    worst_count = max(1, math.ceil(0.10 * scaled.shape[1]))
    return {
        "num_blocks": expected_blocks,
        "paths_per_block": expected_paths.long().tolist(),
        "global_loss": _scalar_block_statistics(loss),
        "maximum_abs_raw_mean_z": _scalar_block_statistics(
            torch.max(absolute_raw, dim=1).values
        ),
        "worst_10_percent_mean_abs_raw_mean_z": _scalar_block_statistics(
            torch.mean(torch.topk(absolute_raw, worst_count, dim=1).values, dim=1)
        ),
        "maximum_abs_noise_scaled_adjusted_mean_z": _scalar_block_statistics(
            torch.max(absolute_adjusted, dim=1).values
        ),
        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": (
            _scalar_block_statistics(
                torch.mean(torch.topk(absolute_adjusted, worst_count, dim=1).values, dim=1)
            )
        ),
    }


def _aggregate_state_summaries(
    per_state: list[dict[str, Any]],
    *,
    z_target: float,
    paths_per_state: int,
) -> dict[str, Any]:
    """Aggregate state-local summaries without pooling their control variates."""

    if not per_state:
        raise ValueError("an all-state audit requires at least one state")
    raw: list[tuple[float, int, int]] = []
    adjusted: list[tuple[float, int, int]] = []
    for state in per_state:
        state_index = int(state["state_index"])
        raw.extend(
            (abs(float(value)), state_index, instrument)
            for instrument, value in enumerate(state["raw_mean_z"])
        )
        adjusted.extend(
            (abs(float(value)), state_index, instrument)
            for instrument, value in enumerate(state["noise_scaled_adjusted_mean_z"])
        )
    raw.sort(reverse=True)
    adjusted.sort(reverse=True)
    worst_count = max(1, math.ceil(0.10 * len(adjusted)))
    return {
        "num_states": len(per_state),
        "num_moments": len(adjusted),
        "paths_per_state": paths_per_state,
        "total_state_paths": len(per_state) * paths_per_state,
        "common_random_numbers_across_states": True,
        "z_target": z_target,
        "maximum_abs_raw_mean_z": raw[0][0],
        "worst_10_percent_mean_abs_raw_mean_z": sum(value for value, _, _ in raw[:worst_count])
        / worst_count,
        "maximum_abs_noise_scaled_adjusted_mean_z": adjusted[0][0],
        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": sum(
            value for value, _, _ in adjusted[:worst_count]
        )
        / worst_count,
        "worst_raw_state_index": raw[0][1],
        "worst_raw_instrument_index": raw[0][2],
        "worst_state_index": adjusted[0][1],
        "worst_instrument_index": adjusted[0][2],
        "passes_target": adjusted[0][0] <= z_target,
        "per_state": per_state,
    }


def _audit_all_states(
    *,
    model: Any,
    law: JointGenericFrontTransport,
    history_indices: tuple[int, ...],
    dates: tuple[str, ...],
    state_kinds: tuple[str, ...],
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    state_batch_size: int,
    z_target: float,
    inner_tail_fraction: float = DEFAULT_INNER_TAIL_FRACTION,
    outer_tail_fraction: float = DEFAULT_OUTER_TAIL_FRACTION,
    inner_tail_weight: float = DEFAULT_INNER_TAIL_WEIGHT,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Audit every state in bounded-memory batches on one fixed dataset."""

    per_state: list[dict[str, Any]] = []
    values_by_batch: list[Mapping[str, torch.Tensor]] = []
    bootstrap_counts: torch.Tensor | None = None
    bootstrap_maximum: torch.Tensor | None = None
    bootstrap_worst_values: torch.Tensor | None = None
    if bootstrap_samples:
        bootstrap_counts = _bootstrap_replication_counts(
            replications=len(draws),
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            device=current_states.device,
        )
    for state_slice in _state_batches(len(current_states), state_batch_size):
        with torch.no_grad():
            values = _evaluate_joint(
                model=model,
                law=law,
                current_states=current_states[state_slice],
                current_w2_calls=current_w2_calls[state_slice],
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support[state_slice],
                density=density[state_slice],
                sqrt_variance=sqrt_variance[state_slice],
                draws=draws,
            )
        values_by_batch.append(values)
        bootstrap_raw: torch.Tensor | None = None
        bootstrap_adjusted: torch.Tensor | None = None
        bootstrap_quantiles: tuple[torch.Tensor, torch.Tensor] | None = None
        if bootstrap_counts is not None:
            bootstrap_raw, bootstrap_adjusted = _bootstrap_means(values, bootstrap_counts)
            quantile_levels = torch.tensor(
                [0.025, 0.5, 0.975],
                dtype=torch.float64,
                device=current_states.device,
            )
            bootstrap_quantiles = (
                torch.quantile(bootstrap_raw, quantile_levels, dim=0),
                torch.quantile(bootstrap_adjusted, quantile_levels, dim=0),
            )
            raw_rep = values["raw_replication_means"].double()
            spot_rep = values["spot_replication_means"].double()
            beta = values["beta"].double()
            raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(len(raw_rep))
            spot_se = torch.std(spot_rep, dim=0, correction=1) / math.sqrt(len(spot_rep))
            noise_scale = torch.maximum(torch.abs(beta) * spot_se[..., None], raw_se)
            full_adjusted_mean = torch.mean(values["adjusted_replication_means"].double(), dim=0)
            centered = torch.abs(
                (bootstrap_adjusted - full_adjusted_mean[None])
                / noise_scale.clamp_min(1.0e-12)[None]
            ).reshape(bootstrap_samples, -1)
            batch_maximum = torch.max(centered, dim=1).values
            bootstrap_maximum = (
                batch_maximum
                if bootstrap_maximum is None
                else torch.maximum(bootstrap_maximum, batch_maximum)
            )
            combined = (
                centered
                if bootstrap_worst_values is None
                else torch.cat((bootstrap_worst_values, centered), dim=1)
            )
            keep = min(
                math.ceil(0.10 * len(current_states) * len(normalized_strikes)),
                combined.shape[1],
            )
            bootstrap_worst_values = torch.topk(combined, keep, dim=1).values
        for local_index in range(state_slice.stop - state_slice.start):
            summary = generic_state_summary(
                {
                    "raw_replication_means": values["raw_replication_means"][:, local_index],
                    "spot_replication_means": values["spot_replication_means"][:, local_index],
                    "adjusted_replication_means": values["adjusted_replication_means"][
                        :, local_index
                    ],
                    "beta": values["beta"][local_index],
                    "path_sum": values["path_sum"][local_index],
                    "path_square_sum": values["path_square_sum"][local_index],
                    "cap_mean_square": values["cap_mean_square"][local_index],
                    "num_paths": values["num_paths"],
                },
                z_target=z_target,
            )
            absolute_index = state_slice.start + local_index
            summary.update(
                {
                    "state_index": history_indices[absolute_index],
                    "date": dates[absolute_index],
                    "history_partition": state_kinds[absolute_index],
                    "replication_mean_statistics": {
                        "raw": _replication_mean_statistics(
                            values["raw_replication_means"][:, local_index]
                        ),
                        "adjusted": _replication_mean_statistics(
                            values["adjusted_replication_means"][:, local_index]
                        ),
                    },
                }
            )
            if (
                bootstrap_raw is not None
                and bootstrap_adjusted is not None
                and bootstrap_quantiles is not None
            ):
                raw_quantiles, adjusted_quantiles = bootstrap_quantiles
                summary["bootstrap"] = {
                    "samples": bootstrap_samples,
                    "seed": bootstrap_seed,
                    "block_replications": len(draws),
                    "beta_reestimated_per_sample": True,
                    "raw_mean_bps_spot": (
                        1.0e4 * torch.mean(bootstrap_raw[:, local_index], dim=0)
                    ).tolist(),
                    "raw_mean_mc_standard_error_bps_spot": (
                        1.0e4 * torch.std(bootstrap_raw[:, local_index], dim=0, correction=1)
                    ).tolist(),
                    "raw_mean_quantiles_bps_spot": {
                        label: (1.0e4 * raw_quantiles[index, local_index]).tolist()
                        for index, label in enumerate(("0.025", "0.5", "0.975"))
                    },
                    "adjusted_mean_bps_spot": (
                        1.0e4 * torch.mean(bootstrap_adjusted[:, local_index], dim=0)
                    ).tolist(),
                    "adjusted_mean_mc_standard_error_bps_spot": (
                        1.0e4 * torch.std(bootstrap_adjusted[:, local_index], dim=0, correction=1)
                    ).tolist(),
                    "adjusted_mean_quantiles_bps_spot": {
                        label: (1.0e4 * adjusted_quantiles[index, local_index]).tolist()
                        for index, label in enumerate(("0.025", "0.5", "0.975"))
                    },
                }
            per_state.append(summary)
    audit = _aggregate_state_summaries(
        per_state,
        z_target=z_target,
        paths_per_state=sum(draw.num_paths for draw in draws),
    )
    global_loss, _, _ = _global_objective(
        values_by_batch,
        z_target=z_target,
        inner_tail_fraction=inner_tail_fraction,
        outer_tail_fraction=outer_tail_fraction,
        inner_tail_weight=inner_tail_weight,
        dtype=current_states.dtype,
    )
    audit["global_loss"] = float(global_loss)
    audit["block_objective"] = _block_objective_summary(
        values_by_batch,
        z_target=z_target,
        inner_tail_fraction=inner_tail_fraction,
        outer_tail_fraction=outer_tail_fraction,
        inner_tail_weight=inner_tail_weight,
    )
    if bootstrap_maximum is not None and bootstrap_worst_values is not None:
        bootstrap_levels = torch.tensor(
            [0.5, 0.9, 0.95, 0.99],
            dtype=torch.float64,
            device=current_states.device,
        )
        bootstrap_worst_10 = torch.mean(bootstrap_worst_values, dim=1)
        audit["bootstrap"] = {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "beta_reestimated_per_sample": True,
            "joint_centered_maximum_abs_noise_z_quantiles": {
                label: float(value)
                for label, value in zip(
                    ("0.5", "0.9", "0.95", "0.99"),
                    torch.quantile(bootstrap_maximum, bootstrap_levels),
                    strict=True,
                )
            },
            "joint_centered_worst_10_percent_mean_abs_noise_z_quantiles": {
                label: float(value)
                for label, value in zip(
                    ("0.5", "0.9", "0.95", "0.99"),
                    torch.quantile(bootstrap_worst_10, bootstrap_levels),
                    strict=True,
                )
            },
        }
    return audit


def _audit_consistency(
    validation: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare independent audit means using their block-bootstrap errors."""

    validation_states = cast(list[dict[str, Any]], validation["per_state"])
    confirmation_states = cast(list[dict[str, Any]], confirmation["per_state"])
    if len(validation_states) != len(confirmation_states):
        raise ValueError("audit consistency needs matching states")
    raw: list[tuple[float, int, int]] = []
    adjusted: list[tuple[float, int, int]] = []
    per_state: list[dict[str, Any]] = []
    for validation_state, confirmation_state in zip(
        validation_states, confirmation_states, strict=True
    ):
        state_index = int(validation_state["state_index"])
        if state_index != int(confirmation_state["state_index"]):
            raise ValueError("audit consistency state order differs")
        validation_bootstrap = cast(dict[str, Any], validation_state["bootstrap"])
        confirmation_bootstrap = cast(dict[str, Any], confirmation_state["bootstrap"])
        state_result: dict[str, Any] = {"state_index": state_index}
        for label, mean_key, error_key, output in (
            (
                "raw",
                "raw_mean_bps_spot",
                "raw_mean_mc_standard_error_bps_spot",
                raw,
            ),
            (
                "adjusted",
                "adjusted_mean_bps_spot",
                "adjusted_mean_mc_standard_error_bps_spot",
                adjusted,
            ),
        ):
            validation_mean = torch.tensor(validation_bootstrap[mean_key], dtype=torch.float64)
            confirmation_mean = torch.tensor(confirmation_bootstrap[mean_key], dtype=torch.float64)
            validation_error = torch.tensor(validation_bootstrap[error_key], dtype=torch.float64)
            confirmation_error = torch.tensor(
                confirmation_bootstrap[error_key], dtype=torch.float64
            )
            difference = validation_mean - confirmation_mean
            combined_error = torch.sqrt(
                validation_error.square() + confirmation_error.square()
            ).clamp_min(1.0e-12)
            difference_z = difference / combined_error
            state_result[f"{label}_mean_difference_bps_spot"] = difference.tolist()
            state_result[f"{label}_mean_difference_z"] = difference_z.tolist()
            output.extend(
                (abs(float(value)), state_index, instrument)
                for instrument, value in enumerate(difference_z)
            )
        per_state.append(state_result)
    raw.sort(reverse=True)
    adjusted.sort(reverse=True)
    worst_count = max(1, math.ceil(0.10 * len(adjusted)))
    return {
        "num_moments": len(adjusted),
        "maximum_abs_raw_mean_difference_z": raw[0][0],
        "worst_10_percent_mean_abs_raw_mean_difference_z": sum(
            value for value, _, _ in raw[:worst_count]
        )
        / worst_count,
        "maximum_abs_adjusted_mean_difference_z": adjusted[0][0],
        "worst_10_percent_mean_abs_adjusted_mean_difference_z": sum(
            value for value, _, _ in adjusted[:worst_count]
        )
        / worst_count,
        "worst_state_index": adjusted[0][1],
        "worst_instrument_index": adjusted[0][2],
        "per_state": per_state,
    }


def _global_objective(
    values_by_batch: list[Mapping[str, torch.Tensor]],
    *,
    z_target: float,
    inner_tail_fraction: float,
    outer_tail_fraction: float,
    inner_tail_weight: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    """Construct one detached nested-tail objective over all state-option moments."""

    if not values_by_batch:
        raise ValueError("the global objective needs at least one state batch")
    statistics: list[dict[str, torch.Tensor]] = []
    for values in values_by_batch:
        adjusted_rep = values["adjusted_replication_means"].double()
        raw_rep = values["raw_replication_means"].double()
        spot_rep = values["spot_replication_means"].double()
        adjusted_mean = torch.mean(adjusted_rep, dim=0)
        raw_mean = torch.mean(raw_rep, dim=0)
        raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(len(raw_rep))
        spot_se = torch.std(spot_rep, dim=0, correction=1) / math.sqrt(len(spot_rep))
        beta = values["beta"].double()
        noise_scale = torch.maximum(torch.abs(beta) * spot_se[..., None], raw_se)
        scaled = adjusted_mean / noise_scale.clamp_min(1.0e-12)
        statistics.append(
            {
                "scaled": scaled,
                "noise_scale": noise_scale,
                "absolute_raw": torch.abs(raw_mean / raw_se.clamp_min(1.0e-12)),
                "absolute_adjusted": torch.abs(scaled),
            }
        )
    scaled_flat = torch.cat([value["scaled"].reshape(-1) for value in statistics])
    loss, derivative_scaled = _tail_excess_loss_and_derivative(
        scaled_flat,
        z_target=z_target,
        inner_tail_fraction=inner_tail_fraction,
        outer_tail_fraction=outer_tail_fraction,
        inner_tail_weight=inner_tail_weight,
    )
    mean_gradients: list[torch.Tensor] = []
    offset = 0
    for value in statistics:
        scaled = value["scaled"]
        count = scaled.numel()
        derivative = derivative_scaled[offset : offset + count].reshape_as(scaled)
        mean_gradients.append(
            (derivative / value["noise_scale"].clamp_min(1.0e-12)).to(dtype=dtype)
        )
        offset += count
    assert offset == scaled_flat.numel(), (offset, scaled_flat.numel())
    absolute_raw = torch.cat([value["absolute_raw"].reshape(-1) for value in statistics])
    absolute_adjusted = torch.cat(
        [value["absolute_adjusted"].reshape(-1) for value in statistics]
    )
    worst_count = max(1, math.ceil(0.10 * absolute_adjusted.numel()))
    metrics = {
        "loss": float(loss),
        "raw_worst": float(torch.max(absolute_raw)),
        "raw_worst10": float(torch.mean(torch.topk(absolute_raw, worst_count).values)),
        "adjusted_worst": float(torch.max(absolute_adjusted)),
        "adjusted_worst10": float(
            torch.mean(torch.topk(absolute_adjusted, worst_count).values)
        ),
    }
    return loss, mean_gradients, metrics


def _train_state_batches(
    *,
    model: Any,
    law: JointGenericFrontTransport,
    optimizer: torch.optim.Optimizer,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    z_target: float,
    inner_tail_fraction: float,
    outer_tail_fraction: float,
    inner_tail_weight: float,
    order: torch.Tensor,
    state_batch_size: int,
) -> dict[str, float]:
    """Accumulate one history-wide objective and take one optimizer step."""

    batch_indices = [
        order[start : start + state_batch_size].to(device=current_states.device)
        for start in range(0, len(order), state_batch_size)
    ]
    values_by_batch: list[Mapping[str, torch.Tensor]] = []
    with torch.no_grad():
        for indices in batch_indices:
            values_by_batch.append(
                _evaluate_joint(
                    model=model,
                    law=law,
                    current_states=current_states[indices],
                    current_w2_calls=current_w2_calls[indices],
                    normalized_strikes=normalized_strikes,
                    option_is_call=option_is_call,
                    support=support[indices],
                    density=density[indices],
                    sqrt_variance=sqrt_variance[indices],
                    draws=draws,
                )
            )
        _, mean_gradients, metrics = _global_objective(
            values_by_batch,
            z_target=z_target,
            inner_tail_fraction=inner_tail_fraction,
            outer_tail_fraction=outer_tail_fraction,
            inner_tail_weight=inner_tail_weight,
            dtype=current_states.dtype,
        )
    optimizer.zero_grad(set_to_none=True)
    for indices, values, mean_gradient in zip(
        batch_indices, values_by_batch, mean_gradients, strict=True
    ):
        beta = values["beta"].to(dtype=current_states.dtype)
        for draw in draws:
            returns, _, gross = joint_roll_returns(
                model=model,
                law=law,
                current_states=current_states[indices],
                current_w2_calls=current_w2_calls[indices],
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support[indices],
                density=density[indices],
                sqrt_variance=sqrt_variance[indices],
                draw=draw,
            )
            adjusted_mean = (
                torch.mean(returns, dim=1)
                - beta * torch.mean(gross - 1.0, dim=1)[..., None]
            )
            surrogate = torch.sum(mean_gradient * adjusted_mean) / len(draws)
            surrogate.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
    optimizer.step()
    metrics["gradient_norm"] = float(gradient_norm.detach())
    return metrics


def _distill_affine_history(
    *,
    law: JointGenericFrontTransport,
    current_front_states: torch.Tensor,
    affine_payload: Mapping[str, Any],
    steps: int,
    states_per_step: int,
    paths_per_state: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Warm-start the shared map from the preserved statewise affine law."""

    teacher_state = cast(Mapping[str, torch.Tensor], affine_payload["law_state_dict"])
    device = current_front_states.device
    dtype = current_front_states.dtype
    front_state_size = current_front_states.shape[1]
    teacher_front = teacher_state["current_front_states"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    if not torch.equal(teacher_front, current_front_states):
        raise ValueError("affine teacher and requested historical states differ")
    location = teacher_state["location_shift"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    spot_loading = teacher_state["spot_loading"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    raw_noise = teacher_state["raw_noise_std"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    teacher_config = cast(Mapping[str, Any], affine_payload["law_config"])
    noise = float(teacher_config["minimum_state_noise_std"]) + torch.nn.functional.softplus(
        raw_noise
    )
    innovation_scale = torch.sqrt(
        torch.mean(
            location.square() + spot_loading.square() + noise[:, None].square(),
            dim=0,
        )
    ).clamp_min(1.0e-2)

    validation_generator = torch.Generator(device=device).manual_seed(seed + 1)
    validation_states = (
        torch.linspace(
            0,
            len(current_front_states) - 1,
            min(128, len(current_front_states)),
            dtype=torch.float64,
            device=device,
        )
        .round()
        .long()
    )
    validation_rank = torch.randn(
        (len(validation_states), paths_per_state),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    validation_epsilon = torch.randn(
        (paths_per_state, front_state_size),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    validation_target = (
        current_front_states[validation_states, None]
        + location[validation_states, None]
        + validation_rank[..., None] * spot_loading[validation_states, None]
        + noise[validation_states, None, None] * validation_epsilon[None]
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(law.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for step in range(steps):
        indices = torch.randint(
            len(current_front_states),
            (min(states_per_step, len(current_front_states)),),
            device=device,
            generator=generator,
        )
        spot_rank = torch.randn(
            (len(indices), paths_per_state),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        epsilon = torch.randn(
            (paths_per_state, front_state_size),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        target = (
            current_front_states[indices, None]
            + location[indices, None]
            + spot_rank[..., None] * spot_loading[indices, None]
            + noise[indices, None, None] * epsilon[None]
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = law(
            current_front_states=current_front_states[indices],
            spot_rank=spot_rank,
            epsilon=epsilon,
        )
        scaled_error = (prediction - target) / innovation_scale
        loss = torch.mean(scaled_error.square())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            with torch.no_grad():
                validation_prediction = law(
                    current_front_states=current_front_states[validation_states],
                    spot_rank=validation_rank,
                    epsilon=validation_epsilon,
                )
                validation_rmse = torch.sqrt(
                    torch.mean(
                        ((validation_prediction - validation_target) / innovation_scale).square()
                    )
                )
            record = {
                "step": float(step + 1),
                "fit_normalized_rmse": float(torch.sqrt(loss.detach())),
                "validation_normalized_rmse": float(validation_rmse),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(record)
            print(
                f"affine_distill step={step + 1} "
                f"fit_rmse={record['fit_normalized_rmse']:.5f} "
                f"validation_rmse={record['validation_normalized_rmse']:.5f}",
                flush=True,
            )
    return {
        "steps": steps,
        "states_per_step": states_per_step,
        "paths_per_state_per_step": paths_per_state,
        "learning_rate": learning_rate,
        "seed": seed,
        "branch_component_included": False,
        "innovation_scale": innovation_scale.detach().cpu().tolist(),
        "history": history,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument("--teacher-checkpoint", type=Path, default=_DEFAULT_TEACHER)
    parser.add_argument(
        "--affine-teacher-checkpoint",
        type=Path,
        default=_DEFAULT_AFFINE_TEACHER,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports/all-state-joint-generic-q.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints/all-state-joint-generic-q.pt",
    )
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sweeps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--resume-learning-rate", type=float)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--refresh-fit-seed-base", type=int, default=18_600_101)
    parser.add_argument("--reset-training-control", action="store_true")
    parser.add_argument("--maximum-shift", type=float, default=16.0)
    parser.add_argument("--maximum-log-scale", type=float, default=5.5)
    parser.add_argument("--maximum-lower-loading", type=float, default=6.0)
    parser.add_argument("--affine-distillation-steps", type=int, default=1_000)
    parser.add_argument("--affine-distillation-states", type=int, default=128)
    parser.add_argument("--affine-distillation-paths", type=int, default=128)
    parser.add_argument("--affine-distillation-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--affine-distillation-seed", type=int, default=12_482_101)
    parser.add_argument("--state-batch-size", type=int, default=64)
    parser.add_argument("--audit-state-batch-size", type=int, default=8)
    parser.add_argument("--fit-paths", type=int, default=10_000)
    parser.add_argument("--fit-blocks", type=int, default=10)
    parser.add_argument("--validation-paths", type=int, default=100_000)
    parser.add_argument("--validation-blocks", type=int, default=10)
    parser.add_argument("--confirmation-paths", type=int, default=100_000)
    parser.add_argument("--confirmation-blocks", type=int, default=10)
    parser.add_argument("--fit-seed", type=int, default=12_484_101)
    parser.add_argument("--validation-seed", type=int, default=12_485_101)
    parser.add_argument("--confirmation-seed", type=int, default=12_487_101)
    parser.add_argument("--validation-bootstrap-seed", type=int, default=12_490_101)
    parser.add_argument("--confirmation-bootstrap-seed", type=int, default=12_491_101)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--order-seed", type=int, default=12_483_101)
    parser.add_argument("--validation-z-target", type=float, default=2.0)
    parser.add_argument("--confirmation-z-target", type=float, default=2.0)
    parser.add_argument(
        "--inner-tail-fraction",
        type=float,
        default=DEFAULT_INNER_TAIL_FRACTION,
    )
    parser.add_argument(
        "--outer-tail-fraction",
        type=float,
        default=DEFAULT_OUTER_TAIL_FRACTION,
    )
    parser.add_argument(
        "--inner-tail-weight",
        type=float,
        default=DEFAULT_INNER_TAIL_WEIGHT,
    )
    parser.add_argument("--plateau-sweeps", type=int, default=12)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-4)
    parser.add_argument("--loss-ema-decay", type=float, default=0.9)
    parser.add_argument("--validation-pass-sweeps", type=int, default=1)
    parser.add_argument("--maximum-states", type=int)
    parser.add_argument("--audit-initial-only", action="store_true")
    parser.add_argument("--fresh-audit-seeds", type=int, nargs="+")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="+", default=())
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    counts = (
        arguments.sweeps,
        arguments.state_batch_size,
        arguments.audit_state_batch_size,
        arguments.affine_distillation_steps,
        arguments.affine_distillation_states,
        arguments.affine_distillation_paths,
        arguments.fit_paths,
        arguments.fit_blocks,
        arguments.validation_paths,
        arguments.validation_blocks,
        arguments.confirmation_paths,
        arguments.confirmation_blocks,
        arguments.plateau_sweeps,
        arguments.scheduler_patience,
        arguments.bootstrap_samples,
        arguments.validation_pass_sweeps,
    )
    if arguments.device != "cuda:0" or any(value <= 0 for value in counts):
        raise ValueError("all-state joint generic Q requires GPU0 and positive counts")
    if min(arguments.fit_blocks, arguments.validation_blocks, arguments.confirmation_blocks) < 2:
        raise ValueError("all-state joint generic MC errors need two replications")
    if (
        arguments.fit_paths % arguments.fit_blocks
        or arguments.validation_paths % arguments.validation_blocks
        or arguments.confirmation_paths % arguments.confirmation_blocks
    ):
        raise ValueError("all-state joint path counts must divide evenly into blocks")
    if arguments.cold_start and arguments.warm_start_checkpoint is not None:
        raise ValueError("cold start and warm-start checkpoint are mutually exclusive")
    if arguments.wandb_project is None and any(
        value is not None
        for value in (
            arguments.wandb_entity,
            arguments.wandb_run_name,
            arguments.wandb_run_id,
        )
    ):
        raise ValueError("W&B entity, name, or ID requires --wandb-project")
    if (
        len(
            {
                arguments.fit_seed,
                arguments.validation_seed,
                arguments.confirmation_seed,
            }
        )
        != 3
    ):
        raise ValueError("all-state fit and audit streams must be disjoint")
    if arguments.fresh_audit_seeds is not None and (
        any(seed < 0 for seed in arguments.fresh_audit_seeds)
        or len(set(arguments.fresh_audit_seeds)) != len(arguments.fresh_audit_seeds)
        or set(arguments.fresh_audit_seeds)
        & {
            arguments.fit_seed,
            arguments.validation_seed,
            arguments.confirmation_seed,
        }
    ):
        raise ValueError("fresh audit seeds must be non-negative, unique, and unused")
    if arguments.refresh_fit_seed_base is not None and (
        arguments.refresh_fit_seed_base < 0
        or arguments.refresh_fit_seed_base
        in {
            arguments.fit_seed,
            arguments.validation_seed,
            arguments.confirmation_seed,
        }
    ):
        raise ValueError("the refreshed fit seed base must be non-negative and unused")
    audit_seeds = {
        arguments.fit_seed,
        arguments.validation_seed,
        arguments.confirmation_seed,
        arguments.validation_bootstrap_seed,
        arguments.confirmation_bootstrap_seed,
    }
    if len(audit_seeds) != 5 or min(audit_seeds) < 0:
        raise ValueError("fit, audit, and bootstrap seeds must be distinct and non-negative")
    if (
        arguments.learning_rate <= 0.0
        or (arguments.resume_learning_rate is not None and arguments.resume_learning_rate <= 0.0)
        or arguments.maximum_shift <= 0.0
        or arguments.maximum_log_scale <= 0.0
        or arguments.maximum_lower_loading <= 0.0
        or arguments.affine_distillation_learning_rate <= 0.0
        or arguments.validation_z_target <= 0.0
        or arguments.confirmation_z_target <= 0.0
        or not 0.0 < arguments.inner_tail_fraction
        <= arguments.outer_tail_fraction
        <= 1.0
        or not 0.0 <= arguments.inner_tail_weight <= 1.0
        or not 0.0 <= arguments.loss_ema_decay < 1.0
        or not 0.0 < arguments.minimum_relative_improvement < 1.0
    ):
        raise ValueError("all-state joint generic rates and targets are invalid")
    if arguments.maximum_states is not None and arguments.maximum_states <= 0:
        raise ValueError("maximum states must be positive")


def _audit_log(label: str, audit: Mapping[str, Any], *, seconds: float) -> None:
    block_objective = cast(Mapping[str, Any], audit["block_objective"])
    block_loss = cast(Mapping[str, Any], block_objective["global_loss"])
    print(
        f"{label} loss={audit['global_loss']:.5f} "
        f"block_loss_mean={block_loss['mean']:.5f} "
        f"block_loss_median={block_loss['median']:.5f} "
        f"block_loss_se={block_loss['mc_standard_error']:.5f} "
        f"raw_worst={audit['maximum_abs_raw_mean_z']:.3f} "
        f"raw_worst10={audit['worst_10_percent_mean_abs_raw_mean_z']:.3f} "
        "adjusted_worst="
        f"{audit['maximum_abs_noise_scaled_adjusted_mean_z']:.3f} "
        "adjusted_worst10="
        f"{audit['worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z']:.3f} "
        f"worst_state={audit['worst_state_index']} "
        f"worst_instrument={audit['worst_instrument_index']} "
        f"seconds={seconds:.1f}",
        flush=True,
    )
    print(
        f"{label} block_global_losses="
        + ",".join(f"{float(value):.5f}" for value in block_loss["values"]),
        flush=True,
    )


def run(arguments: argparse.Namespace) -> None:
    """Fit and independently audit one state-conditioned history-wide map."""

    _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        train_dates,
        train_states,
        _,
        validation_dates,
        validation_states,
        _,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    date_values = train_dates + validation_dates
    state_kinds = ("model_fit",) * len(train_dates) + ("held_out_2025",) * len(validation_dates)
    current_states = torch.cat((train_states, validation_states), dim=0)
    if arguments.maximum_states is not None:
        current_states = current_states[: arguments.maximum_states]
        date_values = date_values[: arguments.maximum_states]
        state_kinds = state_kinds[: arguments.maximum_states]
    num_states = len(current_states)
    history_indices = tuple(range(num_states))
    dates = tuple(value.isoformat() for value in date_values)
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    front_state_size = model.config.state_sizes_by_expiry[0]
    config = JointGenericTransportConfig(
        front_state_size=front_state_size,
        maximum_shift=arguments.maximum_shift,
        maximum_log_scale=arguments.maximum_log_scale,
        maximum_lower_loading=arguments.maximum_lower_loading,
    )
    law = JointGenericFrontTransport(env, config=config)

    teacher_provenance: dict[str, Any] | None = None
    affine_provenance: dict[str, Any] | None = None
    affine_distillation: dict[str, Any] | None = None
    warm_start_provenance: dict[str, Any] | None = None
    if arguments.cold_start:
        affine_distillation = {
            "cold_start": True,
            "initial_law": "zero-context-network-base-gaussian",
        }
    else:
        teacher = torch.load(
            arguments.teacher_checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(teacher, Mapping)
            or teacher.get("schema") != "flex-six-state-joint-generic-q-training-v1"
            or teacher.get("source_completion_sha256") != sources["completion_sha256"]
        ):
            raise ValueError("six-state generic teacher is incompatible")
        teacher_config = cast(Mapping[str, Any], teacher["law_config"])
        if int(teacher_config["front_state_size"]) != front_state_size:
            raise ValueError("six-state teacher front-state width is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], teacher["law_state_dict"]))
        teacher_provenance = {
            "checkpoint": str(arguments.teacher_checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.teacher_checkpoint),
            "source_epochs": len(cast(list[Any], teacher["history"])),
            "source_law_config": dict(teacher_config),
            "weights_loaded_into_wider_bounds": True,
        }
        affine_teacher = torch.load(
            arguments.affine_teacher_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(affine_teacher, Mapping)
            or affine_teacher.get("schema") != "flex-all-state-vector-q-training-v1"
            or affine_teacher.get("source_completion_sha256") != sources["completion_sha256"]
        ):
            raise ValueError("all-state affine teacher is incompatible")
        affine_provenance = {
            "checkpoint": str(arguments.affine_teacher_checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.affine_teacher_checkpoint),
            "source_epochs": len(cast(list[Any], affine_teacher["history"])),
        }
        if not arguments.checkpoint.exists():
            if arguments.warm_start_checkpoint is not None:
                warm_start = torch.load(
                    arguments.warm_start_checkpoint,
                    map_location=env.device,
                    weights_only=False,
                )
                if (
                    not isinstance(warm_start, Mapping)
                    or warm_start.get("schema")
                    != "flex-all-state-joint-generic-q-training-v1"
                    or warm_start.get("source_completion_sha256")
                    != sources["completion_sha256"]
                    or warm_start.get("law_config") != asdict(config)
                ):
                    raise ValueError("all-state generic warm start is incompatible")
                law.load_state_dict(
                    cast(Mapping[str, torch.Tensor], warm_start["law_state_dict"])
                )
                warm_start_provenance = {
                    "checkpoint": str(arguments.warm_start_checkpoint),
                    "checkpoint_sha256": checkpoint_digest(arguments.warm_start_checkpoint),
                    "source_sweeps": len(cast(list[Any], warm_start["history"])),
                }
                affine_distillation = {
                    "skipped_for_all_state_warm_start": True,
                }
            else:
                affine_distillation = _distill_affine_history(
                    law=law,
                    current_front_states=current_states[:, :front_state_size],
                    affine_payload=affine_teacher,
                    steps=arguments.affine_distillation_steps,
                    states_per_step=arguments.affine_distillation_states,
                    paths_per_state=arguments.affine_distillation_paths,
                    learning_rate=arguments.affine_distillation_learning_rate,
                    seed=arguments.affine_distillation_seed,
                )
        elif arguments.warm_start_checkpoint is not None:
            warm_start_provenance = {
                "checkpoint": str(arguments.warm_start_checkpoint),
                "checkpoint_sha256": checkpoint_digest(arguments.warm_start_checkpoint),
                "source_sweeps": None,
            }
        del affine_teacher

    normalized_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    option_is_call = normalized_strikes <= 1.0
    with torch.no_grad():
        raw = model.raw_states_from_states(current_states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        current_surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        current_w2_calls = model.sanos_decoder.price_calls_batch(
            current_surface,
            expiries_y=model.sanos_decoder.model_expiries_y[1].expand(
                num_states, len(normalized_strikes)
            ),
            strikes=normalized_strikes[None].expand(num_states, -1),
        ).detach()
        support, density, sqrt_variance = normalize_front_components(
            current_surface.support_strikes[:, 0],
            current_surface.densities[:, 0],
            current_surface.backbone_total_vols[:, 0],
        )

    fit_draws = _joint_path_blocks(
        env=env,
        front_state_size=front_state_size,
        total_paths=arguments.fit_paths,
        blocks=arguments.fit_blocks,
        seed=arguments.fit_seed,
    )
    validation_draws = _joint_path_blocks(
        env=env,
        front_state_size=front_state_size,
        total_paths=arguments.validation_paths,
        blocks=arguments.validation_blocks,
        seed=arguments.validation_seed,
    )
    confirmation_draws = _joint_path_blocks(
        env=env,
        front_state_size=front_state_size,
        total_paths=arguments.confirmation_paths,
        blocks=arguments.confirmation_blocks,
        seed=arguments.confirmation_seed,
    )
    simulation_config = {
        "num_states": num_states,
        "date_range": [dates[0], dates[-1]],
        "draw_policy": DRAW_POLICY_ID,
        "one_spot_and_one_surface_state_per_path": True,
        "fit_paths": arguments.fit_paths,
        "fit_blocks": arguments.fit_blocks,
        "fit_paths_per_block": arguments.fit_paths // arguments.fit_blocks,
        "validation_paths": arguments.validation_paths,
        "validation_blocks": arguments.validation_blocks,
        "validation_paths_per_block": (
            arguments.validation_paths // arguments.validation_blocks
        ),
        "confirmation_paths": arguments.confirmation_paths,
        "confirmation_blocks": arguments.confirmation_blocks,
        "confirmation_paths_per_block": (
            arguments.confirmation_paths // arguments.confirmation_blocks
        ),
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
        "confirmation_seed": arguments.confirmation_seed,
        "validation_bootstrap_seed": arguments.validation_bootstrap_seed,
        "confirmation_bootstrap_seed": arguments.confirmation_bootstrap_seed,
        "bootstrap_samples": arguments.bootstrap_samples,
        "affine_distillation_steps": arguments.affine_distillation_steps,
        "affine_distillation_states": arguments.affine_distillation_states,
        "affine_distillation_paths": arguments.affine_distillation_paths,
        "affine_distillation_learning_rate": (arguments.affine_distillation_learning_rate),
        "affine_distillation_seed": arguments.affine_distillation_seed,
        "initialization": "cold_start" if arguments.cold_start else "teacher_initialized",
        "affine_teacher_sha256": (
            None if affine_provenance is None else affine_provenance["checkpoint_sha256"]
        ),
        "warm_start_sha256": (
            None if warm_start_provenance is None else warm_start_provenance["checkpoint_sha256"]
        ),
        "optimizer_update": OPTIMIZER_UPDATE_ID,
        "objective_z_target": arguments.validation_z_target,
        "objective_inner_tail_fraction": arguments.inner_tail_fraction,
        "objective_outer_tail_fraction": arguments.outer_tail_fraction,
        "objective_inner_tail_weight": arguments.inner_tail_weight,
        "common_random_numbers_across_states": True,
        "actual_expected_gross_spot_return": 1.0,
    }

    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=arguments.scheduler_patience,
        min_lr=2.5e-6,
    )
    history: list[dict[str, Any]] = []
    start_sweep = 0
    best_fit_loss = math.inf
    fit_loss_ema: float | None = None
    best_fit_loss_ema = math.inf
    best_validation_block_loss = math.inf
    best_validation_adjusted_worst = math.inf
    stale_sweeps = 0
    validation_pass_streak = 0
    checkpoint: dict[str, Any] | None = None
    if arguments.checkpoint.exists():
        payload = torch.load(
            arguments.checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "flex-all-state-joint-generic-q-training-v1"
            or payload.get("source_completion_sha256") != sources["completion_sha256"]
            or payload.get("law_config") != asdict(config)
            or payload.get("simulation_config") != simulation_config
        ):
            raise ValueError("all-state joint generic checkpoint is incompatible")
        requested_fit_draw_policy = {
            "draw_policy": DRAW_POLICY_ID,
            "refresh_seed_base": arguments.refresh_fit_seed_base,
            "refresh_fraction": 1.0,
            "seed_stride": arguments.fit_blocks,
        }
        saved_fit_draw_policy = payload.get("fit_draw_policy")
        if saved_fit_draw_policy is not None:
            saved_fit_draw_policy = dict(cast(Mapping[str, Any], saved_fit_draw_policy))
            if saved_fit_draw_policy != requested_fit_draw_policy:
                raise ValueError("all-state fit-draw policy is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], payload["law_state_dict"]))
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state_dict"]))
        scheduler.load_state_dict(cast(dict[str, Any], payload["scheduler_state_dict"]))
        scheduler.patience = arguments.scheduler_patience
        if arguments.resume_learning_rate is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = arguments.resume_learning_rate
            scheduler._last_lr = [arguments.resume_learning_rate for _ in optimizer.param_groups]
            scheduler.num_bad_epochs = 0
        history = list(cast(list[dict[str, Any]], payload["history"]))
        start_sweep = len(history)
        best_fit_loss = float(payload["best_fit_loss"])
        fit_loss_ema = (
            None if payload.get("fit_loss_ema") is None else float(payload["fit_loss_ema"])
        )
        best_fit_loss_ema = float(payload.get("best_fit_loss_ema", best_fit_loss))
        best_validation_block_loss = float(
            payload.get("best_validation_block_loss", math.inf)
        )
        best_validation_adjusted_worst = float(
            payload.get("best_validation_adjusted_worst", math.inf)
        )
        stale_sweeps = int(payload["stale_sweeps"])
        validation_pass_streak = int(payload.get("validation_pass_streak", 0))
        saved_fit_draw_pool = payload.get("fit_draw_pool")
        if saved_fit_draw_pool is not None:
            fit_draws = list(cast(list[CommonQDraw], saved_fit_draw_pool))
        if arguments.reset_training_control:
            best_fit_loss = math.inf
            fit_loss_ema = None
            best_fit_loss_ema = math.inf
            best_validation_block_loss = math.inf
            best_validation_adjusted_worst = math.inf
            stale_sweeps = 0
            validation_pass_streak = 0
            scheduler.best = math.inf
            scheduler.num_bad_epochs = 0
            scheduler.cooldown_counter = 0
        affine_distillation = dict(cast(Mapping[str, Any], payload["affine_distillation"]))
        warm_start_provenance = cast(
            dict[str, Any] | None,
            payload.get("warm_start_provenance", warm_start_provenance),
        )
        checkpoint = dict(payload)
        print(f"resuming all-state joint generic Q at sweep {start_sweep}", flush=True)

    wandb_run = _initialize_wandb(
        arguments=arguments,
        law_config=config,
        simulation_config=simulation_config,
        history=history,
    )
    wandb_provenance = (
        None
        if wandb_run is None
        else {
            "entity": wandb_run.entity,
            "project": wandb_run.project,
            "run_id": wandb_run.id,
            "run_name": wandb_run.name,
            "url": wandb_run.url,
        }
    )
    if wandb_provenance is not None:
        print(f"W&B dashboard: {wandb_provenance['url']}", flush=True)

    started = time.perf_counter()
    validation_audit = _audit_all_states(
        model=model,
        law=law,
        history_indices=history_indices,
        dates=dates,
        state_kinds=state_kinds,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=validation_draws,
        state_batch_size=arguments.audit_state_batch_size,
        z_target=arguments.validation_z_target,
        inner_tail_fraction=arguments.inner_tail_fraction,
        outer_tail_fraction=arguments.outer_tail_fraction,
        inner_tail_weight=arguments.inner_tail_weight,
    )
    _audit_log(
        "initial_validation",
        validation_audit,
        seconds=time.perf_counter() - started,
    )
    best_validation_adjusted_worst = min(
        best_validation_adjusted_worst,
        float(validation_audit["maximum_abs_noise_scaled_adjusted_mean_z"]),
    )
    initial_validation_block_loss = float(
        cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], validation_audit["block_objective"])["global_loss"],
        )["mean"]
    )
    best_validation_block_loss = min(
        best_validation_block_loss,
        initial_validation_block_loss,
    )

    if arguments.fresh_audit_seeds is not None:
        fresh_audits: list[dict[str, Any]] = []
        for seed in arguments.fresh_audit_seeds:
            audit_started = time.perf_counter()
            audit = _audit_all_states(
                model=model,
                law=law,
                history_indices=history_indices,
                dates=dates,
                state_kinds=state_kinds,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=_joint_path_blocks(
                    env=env,
                    front_state_size=front_state_size,
                    total_paths=arguments.confirmation_paths,
                    blocks=arguments.confirmation_blocks,
                    seed=seed,
                ),
                state_batch_size=arguments.audit_state_batch_size,
                z_target=arguments.confirmation_z_target,
                inner_tail_fraction=arguments.inner_tail_fraction,
                outer_tail_fraction=arguments.outer_tail_fraction,
                inner_tail_weight=arguments.inner_tail_weight,
            )
            _audit_log(
                f"fresh_audit seed={seed}",
                audit,
                seconds=time.perf_counter() - audit_started,
            )
            fresh_audits.append({"seed": seed, "audit": audit})
        write_json_atomic(
            {
                "schema": "flex-all-state-joint-generic-q-fresh-audits-v1",
                "status": "complete",
                "objective": OBJECTIVE_ID,
                "checkpoint": str(arguments.checkpoint),
                "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
                "simulation_config": simulation_config,
                "monitored_validation": validation_audit,
                "fresh_audits": fresh_audits,
                "sources": sources,
            },
            arguments.output,
        )
        return

    if not history:
        validation_pass_streak = int(bool(validation_audit["passes_target"]))
    stopped_for_validation = validation_pass_streak >= arguments.validation_pass_sweeps
    stopped_for_plateau = False
    if not arguments.audit_initial_only and not stopped_for_validation:
        order_generator = torch.Generator(device="cpu")
        for sweep in range(start_sweep, arguments.sweeps):
            sweep_started = time.perf_counter()
            fit_draw_seed = arguments.fit_seed
            sweep_fit_draws = fit_draws
            if arguments.refresh_fit_seed_base is not None:
                fit_draw_seed = arguments.refresh_fit_seed_base + sweep * arguments.fit_blocks
                fit_draws = _joint_path_blocks(
                    env=env,
                    front_state_size=front_state_size,
                    total_paths=arguments.fit_paths,
                    blocks=arguments.fit_blocks,
                    seed=fit_draw_seed,
                )
                sweep_fit_draws = fit_draws
            order_generator.manual_seed(arguments.order_seed + sweep)
            order = torch.randperm(num_states, generator=order_generator)
            fit_metrics = _train_state_batches(
                model=model,
                law=law,
                optimizer=optimizer,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=sweep_fit_draws,
                z_target=arguments.validation_z_target,
                inner_tail_fraction=arguments.inner_tail_fraction,
                outer_tail_fraction=arguments.outer_tail_fraction,
                inner_tail_weight=arguments.inner_tail_weight,
                order=order,
                state_batch_size=arguments.state_batch_size,
            )
            fit_loss = fit_metrics["loss"]
            fit_loss_ema = (
                fit_loss
                if fit_loss_ema is None
                else arguments.loss_ema_decay * fit_loss_ema
                + (1.0 - arguments.loss_ema_decay) * fit_loss
            )
            best_fit_loss = min(best_fit_loss, fit_loss)
            if fit_loss_ema < best_fit_loss_ema * (1.0 - arguments.minimum_relative_improvement):
                best_fit_loss_ema = fit_loss_ema
            audit_started = time.perf_counter()
            validation_audit = _audit_all_states(
                model=model,
                law=law,
                history_indices=history_indices,
                dates=dates,
                state_kinds=state_kinds,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=validation_draws,
                state_batch_size=arguments.audit_state_batch_size,
                z_target=arguments.validation_z_target,
                inner_tail_fraction=arguments.inner_tail_fraction,
                outer_tail_fraction=arguments.outer_tail_fraction,
                inner_tail_weight=arguments.inner_tail_weight,
            )
            audit_seconds = time.perf_counter() - audit_started
            if bool(validation_audit["passes_target"]):
                validation_pass_streak += 1
            else:
                validation_pass_streak = 0
            validation_adjusted_worst = float(
                validation_audit["maximum_abs_noise_scaled_adjusted_mean_z"]
            )
            validation_improved = validation_adjusted_worst < best_validation_adjusted_worst
            if validation_improved:
                best_validation_adjusted_worst = validation_adjusted_worst
            validation_block_loss = cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], validation_audit["block_objective"])[
                    "global_loss"
                ],
            )
            validation_block_loss_mean = float(validation_block_loss["mean"])
            scheduler.step(validation_block_loss_mean)
            if validation_block_loss_mean < best_validation_block_loss * (
                1.0 - arguments.minimum_relative_improvement
            ):
                best_validation_block_loss = validation_block_loss_mean
                stale_sweeps = 0
            else:
                stale_sweeps += 1
            record = {
                "sweep": float(sweep + 1),
                "fit_global_loss": fit_loss,
                "fit_global_loss_ema": fit_loss_ema,
                "fit_maximum_abs_raw_mean_z": fit_metrics["raw_worst"],
                "fit_worst_10_percent_mean_abs_raw_mean_z": fit_metrics["raw_worst10"],
                "fit_maximum_abs_noise_scaled_adjusted_mean_z": fit_metrics[
                    "adjusted_worst"
                ],
                "fit_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": fit_metrics[
                    "adjusted_worst10"
                ],
                "fit_gradient_norm": fit_metrics["gradient_norm"],
                "validation_maximum_abs_raw_mean_z": float(
                    validation_audit["maximum_abs_raw_mean_z"]
                ),
                "validation_worst_10_percent_mean_abs_raw_mean_z": float(
                    validation_audit["worst_10_percent_mean_abs_raw_mean_z"]
                ),
                "validation_maximum_abs_noise_scaled_adjusted_mean_z": float(
                    validation_audit["maximum_abs_noise_scaled_adjusted_mean_z"]
                ),
                "validation_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": float(
                    validation_audit["worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z"]
                ),
                "validation_global_loss": float(validation_audit["global_loss"]),
                "validation_block_loss_mean": validation_block_loss_mean,
                "validation_block_loss_median": float(validation_block_loss["median"]),
                "validation_block_loss_mc_standard_error": float(
                    validation_block_loss["mc_standard_error"]
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "fit_draw_seed": float(fit_draw_seed),
                "fit_fully_redrawn": float(arguments.refresh_fit_seed_base is not None),
                "validation_pass_streak": float(validation_pass_streak),
                "sweep_seconds": time.perf_counter() - sweep_started,
                "validation_audit_seconds": audit_seconds,
            }
            if not all(math.isfinite(value) for value in record.values()):
                raise FloatingPointError("non-finite all-state joint training record")
            record["validation_block_global_losses"] = list(
                cast(list[float], validation_block_loss["values"])
            )
            history.append(record)
            print(
                f"sweep={sweep + 1} fit "
                f"raw_worst={fit_metrics['raw_worst']:.3f} "
                f"raw_worst10={fit_metrics['raw_worst10']:.3f} "
                f"adjusted_worst={fit_metrics['adjusted_worst']:.3f} "
                f"adjusted_worst10={fit_metrics['adjusted_worst10']:.3f} "
                f"gradient_norm={fit_metrics['gradient_norm']:.3f}",
                flush=True,
            )
            checkpoint = {
                "schema": "flex-all-state-joint-generic-q-training-v1",
                "status": "training",
                "objective": OBJECTIVE_ID,
                "source_completion_sha256": sources["completion_sha256"],
                "law_config": asdict(config),
                "simulation_config": simulation_config,
                "fit_draw_policy": {
                    "draw_policy": DRAW_POLICY_ID,
                    "refresh_seed_base": arguments.refresh_fit_seed_base,
                    "refresh_fraction": 1.0,
                    "seed_stride": arguments.fit_blocks,
                },
                "fit_draw_pool": fit_draws,
                "teacher_provenance": teacher_provenance,
                "affine_teacher_provenance": affine_provenance,
                "affine_distillation": affine_distillation,
                "law_state_dict": law.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_fit_loss": best_fit_loss,
                "fit_loss_ema": fit_loss_ema,
                "best_fit_loss_ema": best_fit_loss_ema,
                "best_validation_block_loss": best_validation_block_loss,
                "best_validation_adjusted_worst": best_validation_adjusted_worst,
                "stale_sweeps": stale_sweeps,
                "validation_pass_streak": validation_pass_streak,
                "warm_start_provenance": warm_start_provenance,
                "wandb_provenance": wandb_provenance,
            }
            _atomic_torch_save(checkpoint, arguments.checkpoint)
            if validation_improved:
                best_checkpoint = arguments.checkpoint.with_name(
                    f"{arguments.checkpoint.stem}-best-validation"
                    f"{arguments.checkpoint.suffix}"
                )
                _atomic_torch_save(
                    {
                        **checkpoint,
                        "status": "best_validation",
                        "best_validation_sweep": sweep + 1,
                    },
                    best_checkpoint,
                )
            _log_wandb_sweep(wandb_run, record)
            _audit_log(
                f"sweep={sweep + 1} validation",
                validation_audit,
                seconds=record["sweep_seconds"],
            )
            if validation_pass_streak >= arguments.validation_pass_sweeps:
                stopped_for_validation = True
                break
            if stale_sweeps >= arguments.plateau_sweeps:
                stopped_for_plateau = True
                break

    if arguments.audit_initial_only:
        write_json_atomic(
            {
                "schema": "flex-all-state-joint-generic-q-initial-audit-v1",
                "status": "complete",
                "objective": OBJECTIVE_ID,
                "teacher_provenance": teacher_provenance,
                "affine_teacher_provenance": affine_provenance,
                "affine_distillation": affine_distillation,
                "simulation_config": simulation_config,
                "independent_validation": validation_audit,
                "sources": sources,
            },
            arguments.output,
        )
        return

    if checkpoint is None:
        checkpoint = {
            "schema": "flex-all-state-joint-generic-q-training-v1",
            "status": "training",
            "objective": OBJECTIVE_ID,
            "source_completion_sha256": sources["completion_sha256"],
            "law_config": asdict(config),
            "simulation_config": simulation_config,
            "fit_draw_policy": {
                "draw_policy": DRAW_POLICY_ID,
                "refresh_seed_base": arguments.refresh_fit_seed_base,
                "refresh_fraction": 1.0,
                "seed_stride": arguments.fit_blocks,
            },
            "fit_draw_pool": fit_draws,
            "teacher_provenance": teacher_provenance,
            "affine_teacher_provenance": affine_provenance,
            "affine_distillation": affine_distillation,
            "law_state_dict": law.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_fit_loss": best_fit_loss,
            "fit_loss_ema": fit_loss_ema,
            "best_fit_loss_ema": best_fit_loss_ema,
            "best_validation_block_loss": best_validation_block_loss,
            "best_validation_adjusted_worst": best_validation_adjusted_worst,
            "stale_sweeps": stale_sweeps,
            "validation_pass_streak": validation_pass_streak,
            "warm_start_provenance": warm_start_provenance,
            "wandb_provenance": wandb_provenance,
        }
        _atomic_torch_save(checkpoint, arguments.checkpoint)

    if not stopped_for_validation:
        incomplete_report = {
            "schema": "flex-all-state-joint-generic-q-report-v1",
            "status": "validation_incomplete",
            "semantic_success": False,
            "objective": OBJECTIVE_ID,
            "simulation_config": simulation_config,
            "training": {
                "sweeps_completed": len(history),
                "best_fit_loss": best_fit_loss,
                "fit_loss_ema": fit_loss_ema,
                "best_fit_loss_ema": best_fit_loss_ema,
                "best_validation_block_loss": best_validation_block_loss,
                "best_validation_adjusted_worst": best_validation_adjusted_worst,
                "validation_pass_streak": validation_pass_streak,
                "required_validation_pass_sweeps": arguments.validation_pass_sweeps,
                "stopped_for_plateau": stopped_for_plateau,
                "history": history,
            },
            "independent_validation": validation_audit,
            "independent_confirmation": None,
            "confirmation_inspected": False,
            "sources": sources,
        }
        write_json_atomic(incomplete_report, arguments.output)
        _atomic_torch_save(
            {
                **checkpoint,
                "status": "training_incomplete",
                "report": str(arguments.output),
                "report_sha256": checkpoint_digest(arguments.output),
            },
            arguments.checkpoint,
        )
        raise RuntimeError(
            "all-state joint generic Q has not passed sustained validation; "
            "confirmation remains uninspected"
        )

    validation_bootstrap_started = time.perf_counter()
    validation_audit = _audit_all_states(
        model=model,
        law=law,
        history_indices=history_indices,
        dates=dates,
        state_kinds=state_kinds,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=validation_draws,
        state_batch_size=arguments.audit_state_batch_size,
        z_target=arguments.validation_z_target,
        inner_tail_fraction=arguments.inner_tail_fraction,
        outer_tail_fraction=arguments.outer_tail_fraction,
        inner_tail_weight=arguments.inner_tail_weight,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.validation_bootstrap_seed,
    )
    _audit_log(
        "bootstrap_validation",
        validation_audit,
        seconds=time.perf_counter() - validation_bootstrap_started,
    )
    confirmation_started = time.perf_counter()
    confirmation_audit = _audit_all_states(
        model=model,
        law=law,
        history_indices=history_indices,
        dates=dates,
        state_kinds=state_kinds,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=confirmation_draws,
        state_batch_size=arguments.audit_state_batch_size,
        z_target=arguments.confirmation_z_target,
        inner_tail_fraction=arguments.inner_tail_fraction,
        outer_tail_fraction=arguments.outer_tail_fraction,
        inner_tail_weight=arguments.inner_tail_weight,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.confirmation_bootstrap_seed,
    )
    _audit_log(
        "independent_confirmation",
        confirmation_audit,
        seconds=time.perf_counter() - confirmation_started,
    )
    semantic_success = bool(
        validation_audit["passes_target"] and confirmation_audit["passes_target"]
    )
    audit_consistency = _audit_consistency(validation_audit, confirmation_audit)
    report = {
        "schema": "flex-all-state-joint-generic-q-report-v1",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "objective": OBJECTIVE_ID,
        "num_states": num_states,
        "num_model_fit_states": sum(value == "model_fit" for value in state_kinds),
        "num_held_out_2025_states": sum(value == "held_out_2025" for value in state_kinds),
        "date_range": [dates[0], dates[-1]],
        "law": {
            "form": "one triangular T_theta(h1_0,spot_rank,epsilon)",
            "full_support": True,
            "discrete_branch": False,
            "num_parameters": sum(value.numel() for value in law.parameters()),
        },
        "teacher_provenance": teacher_provenance,
        "affine_teacher_provenance": affine_provenance,
        "affine_distillation": affine_distillation,
        "warm_start_provenance": warm_start_provenance,
        "wandb_provenance": wandb_provenance,
        "simulation_config": simulation_config,
        "training": {
            "sweeps_completed": len(history),
            "best_fit_loss": best_fit_loss,
            "fit_loss_ema": fit_loss_ema,
            "best_fit_loss_ema": best_fit_loss_ema,
            "best_validation_block_loss": best_validation_block_loss,
            "best_validation_adjusted_worst": best_validation_adjusted_worst,
            "fit_draw_policy": {
                "draw_policy": DRAW_POLICY_ID,
                "refresh_seed_base": arguments.refresh_fit_seed_base,
                "refresh_fraction": 1.0,
                "seed_stride": arguments.fit_blocks,
            },
            "validation_pass_streak": validation_pass_streak,
            "required_validation_pass_sweeps": arguments.validation_pass_sweeps,
            "stopped_for_validation_target": stopped_for_validation,
            "stopped_for_plateau": stopped_for_plateau,
            "history": history,
        },
        "independent_validation": validation_audit,
        "independent_confirmation": confirmation_audit,
        "validation_confirmation_consistency": audit_consistency,
        "sources": sources,
    }
    write_json_atomic(report, arguments.output)
    final_checkpoint = {
        **checkpoint,
        "status": "complete" if semantic_success else "failed_terminal",
        "report": str(arguments.output),
        "report_sha256": checkpoint_digest(arguments.output),
    }
    _atomic_torch_save(final_checkpoint, arguments.checkpoint)
    completion = arguments.output.with_name(f"{arguments.output.stem}-complete.json")
    write_json_atomic(
        {
            "schema": "flex-all-state-joint-generic-q-completion-v1",
            "status": "complete" if semantic_success else "FAILED_TERMINAL",
            "semantic_success": semantic_success,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        },
        completion,
    )
    print(
        json.dumps(
            {
                "validation": {
                    key: validation_audit[key]
                    for key in (
                        "maximum_abs_raw_mean_z",
                        "worst_10_percent_mean_abs_raw_mean_z",
                        "maximum_abs_noise_scaled_adjusted_mean_z",
                        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z",
                        "worst_state_index",
                        "worst_instrument_index",
                    )
                },
                "confirmation": {
                    key: confirmation_audit[key]
                    for key in (
                        "maximum_abs_raw_mean_z",
                        "worst_10_percent_mean_abs_raw_mean_z",
                        "maximum_abs_noise_scaled_adjusted_mean_z",
                        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z",
                        "worst_state_index",
                        "worst_instrument_index",
                    )
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if not semantic_success:
        raise RuntimeError("all-state joint generic Q failed an audit target")


def main() -> None:
    arguments = _parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_all_state_joint_generic_q_existence",
        clean_sigterm=True,
    )
