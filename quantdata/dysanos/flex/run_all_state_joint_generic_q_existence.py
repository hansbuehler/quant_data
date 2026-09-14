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
from quantdata.dysanos.flex.full_state_q_generator import (
    FullStateConditionalGenerator,
    FullStateGeneratorConfig,
    JointFullStateConditionalGenerator,
    JointFullStateGeneratorConfig,
    JointUncompressedFrontConditionalGenerator,
    JointUncompressedFrontGeneratorConfig,
)
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
    GenericConditionalFrontLaw,
    GenericStateQLawConfig,
)
from quantdata.dysanos.flex.run_state_generic_q_existence import (
    _summary as generic_state_summary,
)
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.dysanos.flex.zero_drift import (
    adjusted_path_mean_covariance,
    beta_factor_block_zero_drift_statistics,
    beta_factor_covariance,
    block_zero_drift_statistics,
    correlation_conditioned_precision,
    quadratic_zero_drift,
    raw_path_mean_covariance,
    sign_flip_zero_drift_bootstrap,
    spectral_zero_drift,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig

OBJECTIVE_ID = "all-state-joint-generic-continuous-eleven-option-frozen-gmm-v5"
OPTIMIZER_UPDATE_ID = "one-global-nested-tail-excess-gradient-step-per-sweep-v2"
DRAW_POLICY_ID = "iid-joint-spot-surface-paths-v1"
RAW_CORRELATION_COVARIANCE = "raw-correlation"
BETA_FACTOR_RESIDUAL_SPECTRAL_COVARIANCE = "beta-factor-residual-spectral"
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
_FIXED_GRID_ROOT = (
    Path(quantdata_env.locations.training.path)
    / "flex-fixed-grid"
    / "four-week-no-am-v1"
)
_DEFAULT_STATIC_DECODER = (
    _FIXED_GRID_ROOT
    / "checkpoints"
    / "triangular-four-week-front-loaded-200-200.pt"
)
_DEFAULT_STATIC_REPORT = (
    _FIXED_GRID_ROOT
    / "reports"
    / "spx-2022-2025-fixed-grid-four-week-no-am-v1-"
    "triangular_four-week-front-loaded-200_200.json"
)


class IndividualFrontTransport(torch.nn.Module):
    """Adapt the true state-local transport to the joint simulation interface."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        current_front_state: torch.Tensor,
        config: GenericStateQLawConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.state_law = GenericConditionalFrontLaw(
            env,
            current_front_state=current_front_state,
            config=config,
        )

    def forward(
        self,
        *,
        current_front_states: torch.Tensor,
        spot_rank: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        if len(current_front_states) != 1 or spot_rank.shape[0] != 1:
            raise ValueError("an individual transport accepts exactly one state")
        return self.state_law(
            spot_rank=spot_rank[0],
            epsilon=epsilon,
        ).unsqueeze(0)


def _state_batches(num_states: int, state_batch_size: int) -> list[slice]:
    return [
        slice(start, min(start + state_batch_size, num_states))
        for start in range(0, num_states, state_batch_size)
    ]


def _cyclic_state_minibatch(
    *,
    num_states: int,
    states_per_sweep: int | None,
    sweep: int,
    seed: int,
) -> torch.Tensor:
    """Visit every initial state once per reproducibly shuffled state epoch."""

    if num_states <= 0 or sweep < 0 or seed < 0:
        raise ValueError("state minibatch inputs must be non-negative and non-empty")
    if states_per_sweep is None or states_per_sweep >= num_states:
        return torch.randperm(
            num_states,
            generator=torch.Generator(device="cpu").manual_seed(seed + sweep),
        )
    if states_per_sweep <= 0 or num_states % states_per_sweep:
        raise ValueError("states per sweep must divide the number of initial states")
    batches_per_epoch = num_states // states_per_sweep
    state_epoch, batch = divmod(sweep, batches_per_epoch)
    permutation = torch.randperm(
        num_states,
        generator=torch.Generator(device="cpu").manual_seed(seed + state_epoch),
    )
    start = batch * states_per_sweep
    return permutation[start : start + states_per_sweep]


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
    """Bootstrap raw/adjusted means, respecting a separately frozen beta."""

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
    if bool(values.get("beta_is_fixed", torch.tensor(False))):
        beta = values["beta"].double()[None].expand(len(counts), -1, -1)
    else:
        denominator = (
            spot_square_sum - spot_sum.square() / num_paths[:, None]
        ).clamp_min(torch.finfo(torch.float64).tiny)
        numerator = (
            product_sum
            - spot_sum[..., None] * path_sum / num_paths[:, None, None]
        )
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
        "fit_gradient_norm": "training/gradient_norm_pre_clip",
        "fit_clipped_gradient_norm": "training/gradient_norm_post_clip",
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
    beta_draws: list[CommonQDraw] | None = None,
    state_batch_size: int,
    z_target: float,
    inner_tail_fraction: float = DEFAULT_INNER_TAIL_FRACTION,
    outer_tail_fraction: float = DEFAULT_OUTER_TAIL_FRACTION,
    inner_tail_weight: float = DEFAULT_INNER_TAIL_WEIGHT,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
    zero_drift_diagonal_shrinkage: float = 0.10,
    zero_drift_ridge_fraction: float = 1.0e-6,
    zero_drift_absolute_tolerance_bps: float = 1.0,
    zero_drift_independent_states: bool = True,
    zero_drift_covariance_policy: str = RAW_CORRELATION_COVARIANCE,
    zero_drift_correlation_eigenvalue_cutoff: float = 0.01,
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
            fixed_beta: torch.Tensor | None = None
            if beta_draws is not None:
                calibration = _evaluate_joint(
                    model=model,
                    law=law,
                    current_states=current_states[state_slice],
                    current_w2_calls=current_w2_calls[state_slice],
                    normalized_strikes=normalized_strikes,
                    option_is_call=option_is_call,
                    support=support[state_slice],
                    density=density[state_slice],
                    sqrt_variance=sqrt_variance[state_slice],
                    draws=beta_draws,
                )
                fixed_beta = calibration["beta"]
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
                fixed_beta=fixed_beta,
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
                    "beta_reestimated_per_sample": beta_draws is None,
                    "beta_estimated_on_separate_calibration_stream": (
                        beta_draws is not None
                    ),
                    "beta_calibration_paths": (
                        None
                        if beta_draws is None
                        else sum(draw.num_paths for draw in beta_draws)
                    ),
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
    raw_blocks = torch.cat(
        [value["raw_replication_means"].double() for value in values_by_batch],
        dim=1,
    )
    spot_blocks = torch.cat(
        [value["spot_replication_means"].double() for value in values_by_batch],
        dim=1,
    )
    beta_by_state = torch.cat(
        [value["beta"].double() for value in values_by_batch], dim=0
    )
    numerical_spot_expectation = torch.mean(spot_blocks, dim=0)
    zero_drift_blocks = raw_blocks - (
        numerical_spot_expectation[None, :, None] * beta_by_state[None]
    )
    if zero_drift_covariance_policy == RAW_CORRELATION_COVARIANCE:
        zero_drift = block_zero_drift_statistics(
            zero_drift_blocks,
            diagonal_shrinkage=zero_drift_diagonal_shrinkage,
            ridge_fraction=zero_drift_ridge_fraction,
        )
    elif (
        zero_drift_covariance_policy
        == BETA_FACTOR_RESIDUAL_SPECTRAL_COVARIANCE
    ):
        zero_drift = beta_factor_block_zero_drift_statistics(
            raw_blocks,
            spot_blocks,
            beta_by_state,
            residual_diagonal_shrinkage=zero_drift_diagonal_shrinkage,
            residual_ridge_fraction=zero_drift_ridge_fraction,
            correlation_eigenvalue_cutoff=(
                zero_drift_correlation_eigenvalue_cutoff
            ),
        )
    else:
        raise ValueError(
            f"unknown zero-drift covariance policy: {zero_drift_covariance_policy}"
        )
    state_quadratic = zero_drift.quadratic
    state_count = len(state_quadratic)
    moment_count = raw_blocks.shape[-1]
    tail_count = max(1, math.ceil(0.10 * state_count))
    standard_error = torch.sqrt(
        torch.diagonal(
            zero_drift.covariance_of_mean,
            dim1=-2,
            dim2=-1,
        )
    ).clamp_min(torch.finfo(torch.float64).tiny)
    observed_zero_drift = {
        "dense_sum": float(torch.sum(state_quadratic)),
        "state_maximum": float(torch.max(state_quadratic)),
        "state_tail_mean": float(
            torch.mean(torch.topk(state_quadratic, tail_count).values)
        ),
        "coordinate_maximum": float(
            torch.max(torch.abs(zero_drift.mean) / standard_error)
        ),
    }
    for state, quadratic, normalized in zip(
        per_state,
        zero_drift.quadratic,
        zero_drift.quadratic_per_moment,
        strict=True,
    ):
        state["zero_drift"] = {
            "quadratic": float(quadratic),
            "quadratic_per_moment": float(normalized),
        }
    retained_rank = getattr(zero_drift, "retained_rank", None)
    correlation_eigenvalues = getattr(
        zero_drift,
        "correlation_eigenvalues",
        None,
    )
    if retained_rank is not None and correlation_eigenvalues is not None:
        for state, rank, eigenvalues in zip(
            per_state,
            retained_rank,
            correlation_eigenvalues,
            strict=True,
        ):
            state["zero_drift"].update(
                {
                    "retained_rank": int(rank),
                    "correlation_eigenvalues": eigenvalues.tolist(),
                }
            )
    zero_drift_report: dict[str, Any] = {
        "method": (
            "expectation-corrected-mean-raw-covariance-gmm-v3"
            if zero_drift_covariance_policy == RAW_CORRELATION_COVARIANCE
            else "expectation-corrected-mean-beta-factor-residual-spectral-gmm-v1"
        ),
        "moment_source": "option_roll_return_mean_minus_beta_times_numerical_spot_expectation",
        "covariance_source": (
            "raw_option_returns_centered_on_option_expectation"
            if zero_drift_covariance_policy == RAW_CORRELATION_COVARIANCE
            else "beta_spot_rank_one_plus_shrunk_pathwise_residual_correlation"
        ),
        "pathwise_spot_residualization": False,
        "covariance_policy": zero_drift_covariance_policy,
        "blocks": len(raw_blocks),
        "states": state_count,
        "moments_per_state": moment_count,
        "diagonal_shrinkage": zero_drift_diagonal_shrinkage,
        "ridge_fraction": zero_drift_ridge_fraction,
        "correlation_eigenvalue_cutoff": (
            None
            if zero_drift_covariance_policy == RAW_CORRELATION_COVARIANCE
            else zero_drift_correlation_eigenvalue_cutoff
        ),
        "absolute_tolerance_bps_spot": zero_drift_absolute_tolerance_bps,
        "beta_estimated_on_separate_calibration_stream": beta_draws is not None,
        "beta_calibration_paths": (
            None
            if beta_draws is None
            else sum(draw.num_paths for draw in beta_draws)
        ),
        "observed": observed_zero_drift,
        "mean_state_quadratic": float(torch.mean(state_quadratic)),
        "mean_quadratic_per_moment": float(
            torch.mean(zero_drift.quadratic_per_moment)
        ),
        "state_quadratic": state_quadratic.tolist(),
    }
    if retained_rank is not None:
        zero_drift_report.update(
            {
                "retained_rank_by_state": retained_rank.tolist(),
                "mean_retained_rank": float(torch.mean(retained_rank.double())),
            }
        )
    if bootstrap_samples:
        zero_drift_seed = bootstrap_seed + 104_729
        bootstrap_zero_drift = sign_flip_zero_drift_bootstrap(
            zero_drift_blocks,
            precision=zero_drift.precision,
            covariance_of_mean=zero_drift.covariance_of_mean,
            samples=bootstrap_samples,
            seed=zero_drift_seed,
            independent_states=zero_drift_independent_states,
        )
        bootstrap_levels = torch.tensor(
            [0.5, 0.9, 0.95, 0.975, 0.99],
            dtype=torch.float64,
            device=current_states.device,
        )
        p_values: dict[str, float] = {}
        quantiles: dict[str, dict[str, float]] = {}
        standardized_observed: list[torch.Tensor] = []
        standardized_bootstrap: list[torch.Tensor] = []
        for name, observed in observed_zero_drift.items():
            distribution = bootstrap_zero_drift[name]
            p_values[name] = float(
                (1 + torch.sum(distribution >= observed))
                / (bootstrap_samples + 1)
            )
            quantiles[name] = {
                label: float(value)
                for label, value in zip(
                    ("0.5", "0.9", "0.95", "0.975", "0.99"),
                    torch.quantile(distribution, bootstrap_levels),
                    strict=True,
                )
            }
            location = torch.mean(distribution)
            scale = torch.std(distribution, correction=1).clamp_min(
                torch.finfo(torch.float64).tiny
            )
            standardized_observed.append(
                (torch.as_tensor(observed, device=distribution.device) - location)
                / scale
            )
            standardized_bootstrap.append((distribution - location) / scale)
        observed_omnibus = torch.max(torch.stack(standardized_observed))
        bootstrap_omnibus = torch.max(torch.stack(standardized_bootstrap), dim=0).values
        omnibus_p_value = float(
            (1 + torch.sum(bootstrap_omnibus >= observed_omnibus))
            / (bootstrap_samples + 1)
        )
        coordinate_critical = float(
            torch.quantile(bootstrap_zero_drift["coordinate_maximum"], 0.975)
        )
        upper_absolute_mean = (
            torch.abs(zero_drift.mean) + coordinate_critical * standard_error
        )
        maximum_upper_bps = float(1.0e4 * torch.max(upper_absolute_mean))
        zero_drift_report["bootstrap"] = {
            "samples": bootstrap_samples,
            "seed": zero_drift_seed,
            "independent_multipliers_across_states": zero_drift_independent_states,
            "shared_multiplier_across_instruments_within_state_block": True,
            "centered_under_zero_drift_null": True,
            "quantiles": quantiles,
            "marginal_p_values": p_values,
            "omnibus_standardized_maximum": float(observed_omnibus),
            "omnibus_p_value": omnibus_p_value,
        }
        zero_drift_report["zero_drift_not_rejected"] = omnibus_p_value >= 0.025
        zero_drift_report["simultaneous_equivalence"] = {
            "coordinate_critical_value_0.975": coordinate_critical,
            "maximum_upper_absolute_mean_bps_spot": maximum_upper_bps,
            "passes_absolute_tolerance": (
                maximum_upper_bps <= zero_drift_absolute_tolerance_bps
            ),
        }
    audit["zero_drift"] = zero_drift_report
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
            "beta_reestimated_per_sample": beta_draws is None,
            "beta_estimated_on_separate_calibration_stream": beta_draws is not None,
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


def _monitoring_zero_drift_pass(
    audit: Mapping[str, Any],
    *,
    quadratic_per_moment_target: float,
    absolute_drift_bps_target: float | None = None,
) -> bool:
    """Apply predeclared joint and coordinate drift development gates."""

    zero_drift = cast(Mapping[str, Any], audit["zero_drift"])
    if float(zero_drift["mean_quadratic_per_moment"]) > quadratic_per_moment_target:
        return False
    if absolute_drift_bps_target is None:
        return True
    per_state = cast(list[Mapping[str, Any]], audit["per_state"])
    maximum_absolute_drift = max(
        abs(float(value))
        for state in per_state
        for value in cast(list[float], state["adjusted_mean_bps_spot"])
    )
    return maximum_absolute_drift <= absolute_drift_bps_target


def _monitoring_is_due(*, completed_sweeps: int, interval: int) -> bool:
    """Return whether the development monitor should run after this sweep."""

    if completed_sweeps <= 0 or interval <= 0:
        raise ValueError("completed sweeps and monitoring interval must be positive")
    return completed_sweeps % interval == 0


def _global_objective(
    values_by_batch: list[Mapping[str, torch.Tensor]],
    *,
    z_target: float,
    inner_tail_fraction: float,
    outer_tail_fraction: float,
    inner_tail_weight: float,
    dtype: torch.dtype,
    gmm_precision_by_batch: list[torch.Tensor] | None = None,
    gmm_weight: float = 0.0,
    absolute_drift_weight: float = 0.0,
    absolute_drift_scale_bps: float = 1.0,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    """Construct a detached tail plus frozen-precision GMM objective."""

    if not values_by_batch:
        raise ValueError("the global objective needs at least one state batch")
    statistics: list[dict[str, torch.Tensor]] = []
    for values in values_by_batch:
        adjusted_rep = values["adjusted_replication_means"].double()
        raw_rep = values["raw_replication_means"].double()
        adjusted_mean = torch.mean(adjusted_rep, dim=0)
        raw_mean = torch.mean(raw_rep, dim=0)
        raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(len(raw_rep))
        scaled = adjusted_mean / raw_se.clamp_min(1.0e-12)
        statistics.append(
            {
                "scaled": scaled,
                "noise_scale": raw_se,
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
    if gmm_precision_by_batch is not None and len(gmm_precision_by_batch) != len(
        values_by_batch
    ):
        raise ValueError("frozen GMM precision and state batches differ")
    if min(gmm_weight, absolute_drift_weight) < 0.0 or absolute_drift_scale_bps <= 0.0:
        raise ValueError("GMM and absolute-drift objective weights are invalid")
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
    gmm_loss = torch.zeros((), dtype=torch.float64, device=scaled_flat.device)
    absolute_drift_loss = torch.zeros_like(gmm_loss)
    total_states = sum(value["scaled"].shape[0] for value in statistics)
    total_moments = scaled_flat.numel()
    absolute_scale = absolute_drift_scale_bps * 1.0e-4
    for batch_index, (values, gradient) in enumerate(
        zip(values_by_batch, mean_gradients, strict=True)
    ):
        moment_mean = torch.mean(
            values["adjusted_replication_means"].double(), dim=0
        )
        if gmm_precision_by_batch is not None and gmm_weight:
            precision = gmm_precision_by_batch[batch_index].detach().double()
            expected = moment_mean.shape + (moment_mean.shape[-1],)
            if precision.shape != expected:
                raise ValueError("frozen GMM precision has incompatible shape")
            quadratic = torch.einsum(
                "si,sij,sj->s", moment_mean, precision, moment_mean
            )
            gmm_loss = gmm_loss + torch.sum(quadratic) / (
                total_states * moment_mean.shape[-1]
            )
            gmm_gradient = 2.0 * torch.einsum(
                "sij,sj->si", precision, moment_mean
            ) / (total_states * moment_mean.shape[-1])
            gradient.add_((gmm_weight * gmm_gradient).to(dtype=dtype))
        if absolute_drift_weight:
            absolute_drift_loss = absolute_drift_loss + torch.sum(
                (moment_mean / absolute_scale).square()
            ) / total_moments
            absolute_gradient = (
                2.0 * moment_mean / (absolute_scale**2 * total_moments)
            )
            gradient.add_((absolute_drift_weight * absolute_gradient).to(dtype=dtype))
    loss = loss + gmm_weight * gmm_loss + absolute_drift_weight * absolute_drift_loss
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
        "tail_loss": float(
            loss - gmm_weight * gmm_loss - absolute_drift_weight * absolute_drift_loss
        ),
        "gmm_loss": float(gmm_loss),
        "absolute_drift_loss": float(absolute_drift_loss),
    }
    return loss, mean_gradients, metrics


def _cross_gmm_top_tail_objective(
    values_a_by_batch: list[Mapping[str, torch.Tensor]],
    values_b_by_batch: list[Mapping[str, torch.Tensor]],
    *,
    precision_by_batch: list[torch.Tensor],
    bulk_weight: float,
    tail_weight: float,
    tail_fraction: float,
    absolute_drift_scale_bps: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    """Build unbiased cross-GMM plus a fixed-scale worst-mean tail penalty."""

    batches = len(values_a_by_batch)
    if not batches or len(values_b_by_batch) != batches or len(precision_by_batch) != batches:
        raise ValueError("cross-GMM inputs must contain matching nonempty batches")
    if (
        min(bulk_weight, tail_weight) < 0.0
        or not 0.0 < tail_fraction <= 1.0
        or absolute_drift_scale_bps <= 0.0
    ):
        raise ValueError("cross-GMM weights, tail fraction, or drift scale are invalid")

    means_a: list[torch.Tensor] = []
    means_b: list[torch.Tensor] = []
    pooled_statistics: list[dict[str, torch.Tensor]] = []
    total_states = 0
    total_moments = 0
    for values_a, values_b, precision in zip(
        values_a_by_batch, values_b_by_batch, precision_by_batch, strict=True
    ):
        adjusted_a = values_a["adjusted_replication_means"].double()
        adjusted_b = values_b["adjusted_replication_means"].double()
        raw_a = values_a["raw_replication_means"].double()
        raw_b = values_b["raw_replication_means"].double()
        mean_a = torch.mean(adjusted_a, dim=0)
        mean_b = torch.mean(adjusted_b, dim=0)
        expected_precision_shape = mean_a.shape + (mean_a.shape[-1],)
        if mean_b.shape != mean_a.shape or precision.shape != expected_precision_shape:
            raise ValueError("cross-GMM mean or precision shape is incompatible")
        pooled_adjusted = torch.cat((adjusted_a, adjusted_b), dim=0)
        pooled_raw = torch.cat((raw_a, raw_b), dim=0)
        raw_se = torch.std(pooled_raw, dim=0, correction=1) / math.sqrt(
            len(pooled_raw)
        )
        pooled_statistics.append(
            {
                "absolute_raw": torch.abs(
                    torch.mean(pooled_raw, dim=0) / raw_se.clamp_min(1.0e-12)
                ),
                "absolute_adjusted": torch.abs(
                    torch.mean(pooled_adjusted, dim=0) / raw_se.clamp_min(1.0e-12)
                ),
            }
        )
        means_a.append(mean_a)
        means_b.append(mean_b)
        total_states += mean_a.shape[0]
        total_moments += mean_a.numel()

    cross_gmm_loss = torch.zeros(
        (), dtype=torch.float64, device=means_a[0].device
    )
    for mean_a, mean_b, precision in zip(
        means_a, means_b, precision_by_batch, strict=True
    ):
        precision64 = precision.detach().double()
        divisor = total_states * mean_a.shape[-1]
        cross_gmm_loss = cross_gmm_loss + torch.einsum(
            "si,sij,sj->", mean_a, precision64, mean_b
        ) / divisor

    absolute_scale = absolute_drift_scale_bps * 1.0e-4
    pooled_means = torch.cat(
        [0.5 * (mean_a + mean_b) for mean_a, mean_b in zip(means_a, means_b, strict=True)]
    )
    squared_scaled_means = (pooled_means / absolute_scale).square().reshape(-1)
    tail_count = max(1, math.ceil(tail_fraction * total_moments))
    tail_values, tail_indices = torch.topk(squared_scaled_means, tail_count)
    top_tail_mean_loss = torch.mean(tail_values)
    pooled_tail_gradient = torch.zeros_like(pooled_means).reshape(-1)
    pooled_tail_gradient[tail_indices] = (
        pooled_means.reshape(-1)[tail_indices] / (absolute_scale**2 * tail_count)
    )
    pooled_tail_gradient = pooled_tail_gradient.reshape_as(pooled_means)
    gradients_a: list[torch.Tensor] = []
    gradients_b: list[torch.Tensor] = []
    state_offset = 0
    for mean_a, mean_b, precision in zip(
        means_a, means_b, precision_by_batch, strict=True
    ):
        state_count = mean_a.shape[0]
        precision64 = precision.detach().double()
        divisor = total_states * mean_a.shape[-1]
        tail_gradient = pooled_tail_gradient[state_offset : state_offset + state_count]
        gradients_a.append(
            bulk_weight * torch.einsum("sij,sj->si", precision64, mean_b) / divisor
            + tail_weight * tail_gradient
        )
        gradients_b.append(
            bulk_weight * torch.einsum("sij,sj->si", precision64, mean_a) / divisor
            + tail_weight * tail_gradient
        )
        state_offset += state_count
    if state_offset != total_states:
        raise AssertionError((state_offset, total_states))

    loss = bulk_weight * cross_gmm_loss + tail_weight * top_tail_mean_loss
    absolute_raw = torch.cat(
        [value["absolute_raw"].reshape(-1) for value in pooled_statistics]
    )
    absolute_adjusted = torch.cat(
        [value["absolute_adjusted"].reshape(-1) for value in pooled_statistics]
    )
    worst_count = max(1, math.ceil(0.10 * len(absolute_adjusted)))
    metrics = {
        "loss": float(loss),
        "raw_worst": float(torch.max(absolute_raw)),
        "raw_worst10": float(torch.mean(torch.topk(absolute_raw, worst_count).values)),
        "adjusted_worst": float(torch.max(absolute_adjusted)),
        "adjusted_worst10": float(
            torch.mean(torch.topk(absolute_adjusted, worst_count).values)
        ),
        "tail_loss": float(top_tail_mean_loss),
        "gmm_loss": float(cross_gmm_loss),
        "absolute_drift_loss": float(top_tail_mean_loss),
        "cross_gmm_loss": float(cross_gmm_loss),
        "top_tail_mean_loss": float(top_tail_mean_loss),
        "top_tail_count": float(tail_count),
    }
    return (
        loss,
        [gradient.to(dtype=dtype) for gradient in gradients_a],
        [gradient.to(dtype=dtype) for gradient in gradients_b],
        metrics,
    )


def _beta_factor_path_statistics(
    values: Mapping[str, torch.Tensor],
    *,
    residual_diagonal_shrinkage: float,
    residual_ridge_fraction: float,
    correlation_eigenvalue_cutoff: float,
) -> Any:
    """Construct the detached spot-factor GMM geometry from path sums."""

    num_paths = int(values["num_paths"])
    adjusted_mean, residual_covariance = adjusted_path_mean_covariance(
        path_sum=values["path_sum"],
        path_cross_product_sum=values["path_cross_product_sum"],
        spot_sum=values["spot_sum"],
        spot_square_sum=values["spot_square_sum"],
        spot_return_product_sum=values["spot_return_product_sum"],
        beta=values["beta"],
        num_paths=num_paths,
    )
    centered_spot_square_sum = (
        values["spot_square_sum"].double()
        - values["spot_sum"].double().square() / num_paths
    ).clamp_min(0.0)
    spot_variance_of_mean = centered_spot_square_sum / (
        (num_paths - 1) * num_paths
    )
    factor = beta_factor_covariance(
        beta=values["beta"],
        spot_variance_of_mean=spot_variance_of_mean,
        residual_covariance_of_mean=residual_covariance,
        residual_diagonal_shrinkage=residual_diagonal_shrinkage,
        residual_ridge_fraction=residual_ridge_fraction,
    )
    return spectral_zero_drift(
        adjusted_mean,
        factor.covariance_of_mean,
        correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
    )


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
    frozen_gmm_precision: torch.Tensor | None = None,
    fixed_training_beta: torch.Tensor | None = None,
    gmm_weight: float = 0.0,
    absolute_drift_weight: float = 0.0,
    absolute_drift_scale_bps: float = 1.0,
    cross_gmm_objective: bool = False,
    cross_gmm_bulk_weight: float = 1.0,
    cross_gmm_tail_weight: float = 1.0,
    cross_gmm_tail_fraction: float = 0.01,
    gradient_clip_norm: float = 10.0,
    gmm_covariance_policy: str = RAW_CORRELATION_COVARIANCE,
    gmm_diagonal_shrinkage: float = 0.10,
    gmm_ridge_fraction: float = 1.0e-6,
    gmm_correlation_eigenvalue_cutoff: float = 0.01,
) -> dict[str, Any]:
    """Accumulate one history-wide objective and take one optimizer step."""

    if cross_gmm_objective and (
        len(draws) % 2
        or frozen_gmm_precision is None
        or fixed_training_beta is None
    ):
        raise ValueError(
            "cross-GMM needs an even draw count and frozen precision and beta"
        )

    batch_indices = [
        order[start : start + state_batch_size].to(device=current_states.device)
        for start in range(0, len(order), state_batch_size)
    ]
    values_by_batch: list[Mapping[str, torch.Tensor]] = []
    values_b_by_batch: list[Mapping[str, torch.Tensor]] = []
    gradients_b: list[torch.Tensor] = []
    split = len(draws) // 2
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
                    draws=draws[:split] if cross_gmm_objective else draws,
                    fixed_beta=(
                        fixed_training_beta[indices]
                        if cross_gmm_objective
                        else None
                    ),
                )
            )
            if cross_gmm_objective:
                values_b_by_batch.append(
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
                        draws=draws[split:],
                        fixed_beta=fixed_training_beta[indices],
                    )
                )
        if cross_gmm_objective:
            _, mean_gradients, gradients_b, metrics = (
                _cross_gmm_top_tail_objective(
                    values_by_batch,
                    values_b_by_batch,
                    precision_by_batch=[
                        frozen_gmm_precision[indices] for indices in batch_indices
                    ],
                    bulk_weight=cross_gmm_bulk_weight,
                    tail_weight=cross_gmm_tail_weight,
                    tail_fraction=cross_gmm_tail_fraction,
                    absolute_drift_scale_bps=absolute_drift_scale_bps,
                    dtype=current_states.dtype,
                )
            )
        else:
            _, mean_gradients, metrics = _global_objective(
                values_by_batch,
                z_target=z_target,
                inner_tail_fraction=inner_tail_fraction,
                outer_tail_fraction=outer_tail_fraction,
                inner_tail_weight=inner_tail_weight,
                dtype=current_states.dtype,
                gmm_precision_by_batch=(
                    None
                    if frozen_gmm_precision is None
                    else [frozen_gmm_precision[indices] for indices in batch_indices]
                ),
                gmm_weight=gmm_weight,
                absolute_drift_weight=absolute_drift_weight,
                absolute_drift_scale_bps=absolute_drift_scale_bps,
            )
        covariance_by_batch: list[torch.Tensor] = []
        for values in ([] if cross_gmm_objective else values_by_batch):
            if gmm_covariance_policy == RAW_CORRELATION_COVARIANCE:
                _, covariance = raw_path_mean_covariance(
                    path_sum=values["path_sum"],
                    path_cross_product_sum=values["path_cross_product_sum"],
                    num_paths=int(values["num_paths"]),
                )
            elif (
                gmm_covariance_policy
                == BETA_FACTOR_RESIDUAL_SPECTRAL_COVARIANCE
            ):
                covariance = _beta_factor_path_statistics(
                    values,
                    residual_diagonal_shrinkage=gmm_diagonal_shrinkage,
                    residual_ridge_fraction=gmm_ridge_fraction,
                    correlation_eigenvalue_cutoff=(
                        gmm_correlation_eigenvalue_cutoff
                    ),
                ).covariance_of_mean
            else:
                raise ValueError(f"unknown GMM covariance policy: {gmm_covariance_policy}")
            covariance_by_batch.append(covariance)
    optimizer.zero_grad(set_to_none=True)
    backward_groups = [(draws, values_by_batch, mean_gradients)]
    if cross_gmm_objective:
        backward_groups = [
            (draws[:split], values_by_batch, mean_gradients),
            (draws[split:], values_b_by_batch, gradients_b),
        ]
    for group_draws, group_values, group_gradients in backward_groups:
        for indices, values, mean_gradient in zip(
            batch_indices, group_values, group_gradients, strict=True
        ):
            beta = values["beta"].to(dtype=current_states.dtype)
            for draw in group_draws:
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
                corrected_mean = (
                    torch.mean(returns, dim=1)
                    - beta * torch.mean(gross - 1.0, dim=1)[..., None]
                )
                surrogate = torch.sum(mean_gradient * corrected_mean) / len(
                    group_draws
                )
                surrogate.backward()
    parameters = tuple(law.parameters())
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
    clipped_gradient_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(parameter.grad.detach())
                for parameter in parameters
                if parameter.grad is not None
            ]
        )
    )
    optimizer.step()
    metrics["gradient_norm"] = float(gradient_norm.detach())
    metrics["clipped_gradient_norm"] = float(clipped_gradient_norm.detach())
    if not cross_gmm_objective:
        metrics["detached_covariance_indices"] = torch.cat(batch_indices).detach()
        metrics["detached_covariance_of_mean"] = torch.cat(
            covariance_by_batch,
            dim=0,
        ).detach()
    return metrics


def _calibrate_frozen_gmm_precision(
    *,
    model: Any,
    law: JointGenericFrontTransport,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    state_batch_size: int,
    diagonal_shrinkage: float,
    ridge_fraction: float,
    covariance_policy: str = RAW_CORRELATION_COVARIANCE,
    correlation_eigenvalue_cutoff: float = 0.01,
    maximum_precision_condition_number: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate initial detached covariance and precision for every state."""

    precision: list[torch.Tensor] = []
    covariance_values: list[torch.Tensor] = []
    beta_values: list[torch.Tensor] = []
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
            if covariance_policy == RAW_CORRELATION_COVARIANCE:
                raw_mean, covariance = raw_path_mean_covariance(
                    path_sum=values["path_sum"],
                    path_cross_product_sum=values["path_cross_product_sum"],
                    num_paths=int(values["num_paths"]),
                )
                mean = raw_mean - values["beta"].double() * (
                    values["spot_sum"].double() / int(values["num_paths"])
                )[..., None]
                statistics = quadratic_zero_drift(
                    mean,
                    covariance,
                    diagonal_shrinkage=diagonal_shrinkage,
                    ridge_fraction=ridge_fraction,
                )
            elif covariance_policy == BETA_FACTOR_RESIDUAL_SPECTRAL_COVARIANCE:
                statistics = _beta_factor_path_statistics(
                    values,
                    residual_diagonal_shrinkage=diagonal_shrinkage,
                    residual_ridge_fraction=ridge_fraction,
                    correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
                )
                covariance = statistics.covariance_of_mean
            else:
                raise ValueError(f"unknown GMM covariance policy: {covariance_policy}")
            calibrated_precision = (
                statistics.precision
                if maximum_precision_condition_number is None
                else correlation_conditioned_precision(
                    covariance,
                    diagonal_shrinkage=(
                        0.0
                        if covariance_policy
                        == BETA_FACTOR_RESIDUAL_SPECTRAL_COVARIANCE
                        else diagonal_shrinkage
                    ),
                    maximum_condition_number=maximum_precision_condition_number,
                ).precision
            )
        covariance_values.append(covariance.detach())
        precision.append(calibrated_precision.to(dtype=current_states.dtype))
        beta_values.append(values["beta"].to(dtype=current_states.dtype).detach())
    return (
        torch.cat(covariance_values, dim=0).detach(),
        torch.cat(precision, dim=0).detach(),
        torch.cat(beta_values, dim=0).detach(),
    )


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


def _distill_full_state_generator(
    *,
    law: FullStateConditionalGenerator,
    teacher_payload: Mapping[str, Any],
    current_front_state: torch.Tensor,
    steps: int,
    paths_per_step: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Initialize an unrestricted generator from one preserved state-local Q."""

    config_values = cast(Mapping[str, Any], teacher_payload["law_config"])
    network_values = cast(Mapping[str, Any], config_values["network"])
    teacher_config = GenericStateQLawConfig(
        front_state_size=int(config_values["front_state_size"]),
        initial_state_noise_std=float(config_values["initial_state_noise_std"]),
        minimum_state_noise_std=float(config_values["minimum_state_noise_std"]),
        maximum_network_shift=float(config_values["maximum_network_shift"]),
        maximum_log_scale=float(config_values["maximum_log_scale"]),
        maximum_lower_loading=float(config_values["maximum_lower_loading"]),
        generic_amplitude_fraction=float(config_values["generic_amplitude_fraction"]),
        network=FeedForwardConfig(**network_values),
    )
    if teacher_config.front_state_size != law.config.front_state_size:
        raise ValueError("full-generator teacher has an incompatible state width")
    teacher = IndividualFrontTransport(
        law.env,
        current_front_state=current_front_state,
        config=teacher_config,
    )
    teacher.load_state_dict(
        cast(Mapping[str, torch.Tensor], teacher_payload["law_state_dict"])
    )
    teacher.requires_grad_(False)
    teacher.eval()
    device = current_front_state.device
    dtype = current_front_state.dtype
    width = law.config.front_state_size
    current = current_front_state[None]

    calibration_generator = torch.Generator(device=device).manual_seed(seed)
    calibration_spot = torch.randn(
        (1, 4 * paths_per_step),
        dtype=dtype,
        device=device,
        generator=calibration_generator,
    )
    calibration_epsilon = torch.randn(
        (4 * paths_per_step, width),
        dtype=dtype,
        device=device,
        generator=calibration_generator,
    )
    with torch.no_grad():
        calibration_target = teacher(
            current_front_states=current,
            spot_rank=calibration_spot,
            epsilon=calibration_epsilon,
        )
        innovation_scale = torch.sqrt(
            torch.mean(
                (calibration_target[0] - current_front_state).square(),
                dim=0,
            )
        ).clamp_min(1.0e-3)

    validation_generator = torch.Generator(device=device).manual_seed(seed + 1)
    validation_spot = torch.randn(
        (1, 4 * paths_per_step),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    validation_epsilon = torch.randn(
        (4 * paths_per_step, width),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    with torch.no_grad():
        validation_target = teacher(
            current_front_states=current,
            spot_rank=validation_spot,
            epsilon=validation_epsilon,
        )

    generator = torch.Generator(device=device).manual_seed(seed + 2)
    optimizer = torch.optim.Adam(law.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for step in range(steps):
        spot_rank = torch.randn(
            (1, paths_per_step),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        epsilon = torch.randn(
            (paths_per_step, width),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        with torch.no_grad():
            target = teacher(
                current_front_states=current,
                spot_rank=spot_rank,
                epsilon=epsilon,
            )
        optimizer.zero_grad(set_to_none=True)
        prediction = law(
            current_front_states=current,
            spot_rank=spot_rank,
            epsilon=epsilon,
        )
        loss = torch.mean(((prediction - target) / innovation_scale).square())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            with torch.no_grad():
                validation_prediction = law(
                    current_front_states=current,
                    spot_rank=validation_spot,
                    epsilon=validation_epsilon,
                )
                validation_rmse = torch.sqrt(
                    torch.mean(
                        (
                            (validation_prediction - validation_target)
                            / innovation_scale
                        ).square()
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
                f"full_generator_distill step={step + 1} "
                f"fit_rmse={record['fit_normalized_rmse']:.5f} "
                f"validation_rmse={record['validation_normalized_rmse']:.5f}",
                flush=True,
            )
    return {
        "method": "state-local-full-generator-distilled-from-triangular-Q",
        "steps": steps,
        "paths_per_step": paths_per_step,
        "learning_rate": learning_rate,
        "seed": seed,
        "innovation_scale": innovation_scale.detach().cpu().tolist(),
        "history": history,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument("--static-decoder-checkpoint", type=Path)
    parser.add_argument("--static-report", type=Path)
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
    parser.add_argument("--initial-law-checkpoint", type=Path)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--sweeps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--resume-learning-rate", type=float)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--refresh-fit-seed-base", type=int, default=18_600_101)
    parser.add_argument("--reset-training-control", action="store_true")
    parser.add_argument("--individual-transport", action="store_true")
    parser.add_argument("--full-generic-generator", action="store_true")
    parser.add_argument("--uncompressed-front-generator", action="store_true")
    parser.add_argument("--generator-network-depth", type=int, default=4)
    parser.add_argument("--generator-network-width", type=int, default=512)
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
    parser.add_argument("--fit-states-per-sweep", type=int)
    parser.add_argument("--fit-paths", type=int, default=10_000)
    parser.add_argument("--fit-blocks", type=int, default=2)
    parser.add_argument("--validation-paths", type=int, default=102_400)
    parser.add_argument("--validation-blocks", type=int, default=64)
    parser.add_argument("--confirmation-paths", type=int, default=102_400)
    parser.add_argument("--confirmation-blocks", type=int, default=64)
    parser.add_argument("--beta-calibration-paths", type=int, default=25_600)
    parser.add_argument("--beta-calibration-blocks", type=int, default=16)
    parser.add_argument("--gmm-calibration-paths", type=int, default=10_000)
    parser.add_argument("--gmm-calibration-blocks", type=int, default=2)
    parser.add_argument("--fit-seed", type=int, default=12_484_101)
    parser.add_argument("--validation-seed", type=int, default=12_485_101)
    parser.add_argument("--terminal-validation-seed", type=int, default=12_495_101)
    parser.add_argument("--confirmation-seed", type=int, default=12_487_101)
    parser.add_argument("--validation-beta-seed", type=int, default=12_492_101)
    parser.add_argument(
        "--terminal-validation-beta-seed", type=int, default=12_496_101
    )
    parser.add_argument("--confirmation-beta-seed", type=int, default=12_493_101)
    parser.add_argument("--gmm-calibration-seed", type=int, default=12_494_101)
    parser.add_argument("--validation-bootstrap-seed", type=int, default=12_490_101)
    parser.add_argument(
        "--terminal-validation-bootstrap-seed", type=int, default=12_497_101
    )
    parser.add_argument("--confirmation-bootstrap-seed", type=int, default=12_491_101)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
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
    parser.add_argument("--gmm-weight", type=float, default=1.0)
    parser.add_argument("--absolute-drift-weight", type=float, default=0.01)
    parser.add_argument("--absolute-drift-scale-bps", type=float, default=1.0)
    parser.add_argument("--cross-gmm-objective", action="store_true")
    parser.add_argument("--cross-gmm-bulk-weight", type=float, default=1.0)
    parser.add_argument("--cross-gmm-tail-weight", type=float, default=1.0)
    parser.add_argument("--cross-gmm-tail-fraction", type=float, default=0.01)
    parser.add_argument(
        "--training-precision-maximum-condition-number",
        type=float,
        default=100.0,
    )
    parser.add_argument("--gmm-diagonal-shrinkage", type=float, default=0.10)
    parser.add_argument("--gmm-ridge-fraction", type=float, default=1.0e-6)
    parser.add_argument(
        "--gmm-covariance-policy",
        choices=(
            RAW_CORRELATION_COVARIANCE,
            BETA_FACTOR_RESIDUAL_SPECTRAL_COVARIANCE,
        ),
        default=RAW_CORRELATION_COVARIANCE,
    )
    parser.add_argument(
        "--gmm-correlation-eigenvalue-cutoff",
        type=float,
        default=0.01,
    )
    parser.add_argument("--gmm-covariance-ewa-decay", type=float, default=0.90)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument(
        "--monitoring-gmm-per-moment-target", type=float, default=1.5
    )
    parser.add_argument("--monitoring-absolute-drift-bps-target", type=float)
    parser.add_argument("--plateau-sweeps", type=int, default=12)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-4)
    parser.add_argument("--loss-ema-decay", type=float, default=0.9)
    parser.add_argument("--validation-pass-sweeps", type=int, default=1)
    parser.add_argument("--monitoring-interval", type=int, default=1)
    parser.add_argument("--maximum-states", type=int)
    parser.add_argument("--state-indices", type=int, nargs="+")
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
        arguments.generator_network_depth,
        arguments.generator_network_width,
        arguments.fit_paths,
        arguments.fit_blocks,
        arguments.validation_paths,
        arguments.validation_blocks,
        arguments.confirmation_paths,
        arguments.confirmation_blocks,
        arguments.beta_calibration_paths,
        arguments.beta_calibration_blocks,
        arguments.gmm_calibration_paths,
        arguments.gmm_calibration_blocks,
        arguments.plateau_sweeps,
        arguments.scheduler_patience,
        arguments.bootstrap_samples,
        arguments.validation_pass_sweeps,
        arguments.monitoring_interval,
    )
    cpu_audit = arguments.audit_initial_only and arguments.device == "cpu"
    if (arguments.device != "cuda:0" and not cpu_audit) or any(
        value <= 0 for value in counts
    ):
        raise ValueError(
            "all-state joint generic Q requires GPU0, except for CPU-only initial "
            "audits, and positive counts"
        )
    if min(
        arguments.fit_blocks,
        arguments.validation_blocks,
        arguments.confirmation_blocks,
        arguments.beta_calibration_blocks,
        arguments.gmm_calibration_blocks,
    ) < 2:
        raise ValueError("all-state joint generic MC errors need two replications")
    if (
        arguments.fit_paths % arguments.fit_blocks
        or arguments.validation_paths % arguments.validation_blocks
        or arguments.confirmation_paths % arguments.confirmation_blocks
        or arguments.beta_calibration_paths % arguments.beta_calibration_blocks
        or arguments.gmm_calibration_paths % arguments.gmm_calibration_blocks
    ):
        raise ValueError("all-state joint path counts must divide evenly into blocks")
    if arguments.cold_start and arguments.warm_start_checkpoint is not None:
        raise ValueError("cold start and warm-start checkpoint are mutually exclusive")
    if (
        arguments.initial_law_checkpoint is not None
        and arguments.warm_start_checkpoint is not None
    ):
        raise ValueError("only one law initialization checkpoint may be supplied")
    if arguments.individual_transport and not arguments.cold_start:
        raise ValueError("the individual-transport stage must start independently")
    if arguments.full_generic_generator and not arguments.cold_start:
        raise ValueError("a full generic generator must start independently")
    if arguments.uncompressed_front_generator and not arguments.cold_start:
        raise ValueError("an uncompressed front generator must start independently")
    if arguments.uncompressed_front_generator and (
        arguments.full_generic_generator or arguments.individual_transport
    ):
        raise ValueError(
            "the uncompressed front generator is a distinct shared transport"
        )
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
                arguments.terminal_validation_seed,
                arguments.confirmation_seed,
            }
        )
        != 4
    ):
        raise ValueError("all-state fit and audit streams must be disjoint")
    if arguments.fresh_audit_seeds is not None and (
        any(seed < 0 for seed in arguments.fresh_audit_seeds)
        or len(set(arguments.fresh_audit_seeds)) != len(arguments.fresh_audit_seeds)
        or set(arguments.fresh_audit_seeds)
        & {
            arguments.fit_seed,
            arguments.validation_seed,
            arguments.terminal_validation_seed,
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
            arguments.terminal_validation_seed,
            arguments.confirmation_seed,
        }
    ):
        raise ValueError("the refreshed fit seed base must be non-negative and unused")
    audit_seeds = {
        arguments.fit_seed,
        arguments.validation_seed,
        arguments.terminal_validation_seed,
        arguments.confirmation_seed,
        arguments.validation_bootstrap_seed,
        arguments.terminal_validation_bootstrap_seed,
        arguments.confirmation_bootstrap_seed,
        arguments.validation_beta_seed,
        arguments.terminal_validation_beta_seed,
        arguments.confirmation_beta_seed,
        arguments.gmm_calibration_seed,
    }
    if len(audit_seeds) != 11 or min(audit_seeds) < 0:
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
        or arguments.gmm_weight < 0.0
        or arguments.absolute_drift_weight < 0.0
        or arguments.absolute_drift_scale_bps <= 0.0
        or arguments.cross_gmm_bulk_weight < 0.0
        or arguments.cross_gmm_tail_weight < 0.0
        or not 0.0 < arguments.cross_gmm_tail_fraction <= 1.0
        or arguments.training_precision_maximum_condition_number <= 1.0
        or not 0.0 <= arguments.gmm_diagonal_shrinkage <= 1.0
        or arguments.gmm_ridge_fraction <= 0.0
        or not 0.0 <= arguments.gmm_correlation_eigenvalue_cutoff < 1.0
        or not 0.0 <= arguments.gmm_covariance_ewa_decay < 1.0
        or arguments.gradient_clip_norm <= 0.0
        or arguments.monitoring_gmm_per_moment_target <= 0.0
        or (
            arguments.monitoring_absolute_drift_bps_target is not None
            and arguments.monitoring_absolute_drift_bps_target <= 0.0
        )
        or not 0.0 <= arguments.loss_ema_decay < 1.0
        or not 0.0 < arguments.minimum_relative_improvement < 1.0
    ):
        raise ValueError("all-state joint generic rates and targets are invalid")
    if arguments.cross_gmm_objective and arguments.fit_blocks % 2:
        raise ValueError("cross-GMM requires an even number of fit blocks")
    if arguments.cross_gmm_objective and (
        arguments.gmm_weight or arguments.absolute_drift_weight
    ):
        raise ValueError(
            "cross-GMM replaces --gmm-weight and --absolute-drift-weight; set both to zero"
        )
    if arguments.maximum_states is not None and arguments.maximum_states <= 0:
        raise ValueError("maximum states must be positive")
    if (
        arguments.fit_states_per_sweep is not None
        and arguments.fit_states_per_sweep <= 0
    ):
        raise ValueError("fit states per sweep must be positive")
    if arguments.maximum_states is not None and arguments.state_indices is not None:
        raise ValueError("maximum states and explicit state indices are mutually exclusive")
    if arguments.state_indices is not None and (
        any(index < 0 for index in arguments.state_indices)
        or len(set(arguments.state_indices)) != len(arguments.state_indices)
    ):
        raise ValueError("state indices must be unique and non-negative")
    if (arguments.static_decoder_checkpoint is None) != (arguments.static_report is None):
        raise ValueError("static decoder checkpoint and report must be supplied together")


def _audit_log(label: str, audit: Mapping[str, Any], *, seconds: float) -> None:
    block_objective = cast(Mapping[str, Any], audit["block_objective"])
    block_loss = cast(Mapping[str, Any], block_objective["global_loss"])
    zero_drift = cast(Mapping[str, Any], audit["zero_drift"])
    per_state = cast(list[Mapping[str, Any]], audit["per_state"])
    maximum_absolute_adjusted_mean_bps = max(
        abs(float(value))
        for state in per_state
        for value in cast(list[float], state["adjusted_mean_bps_spot"])
    )
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
        f"gmm_per_moment={float(zero_drift['mean_quadratic_per_moment']):.3f} "
        f"maximum_abs_adjusted_mean_bps={maximum_absolute_adjusted_mean_bps:.3f} "
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
    torch.manual_seed(arguments.order_seed)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(
        precision=arguments.precision,
        device=arguments.device,
        debug=False,
    )
    if arguments.static_decoder_checkpoint is not None:
        model, date_values, current_states, sources = load_static_q_inputs(
            arguments.static_decoder_checkpoint,
            cast(Path, arguments.static_report),
            env=env,
        )
        state_kinds = ("static_fit",) * len(date_values)
    else:
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
        state_kinds = ("model_fit",) * len(train_dates) + ("held_out_2025",) * len(
            validation_dates
        )
        current_states = torch.cat((train_states, validation_states), dim=0)
    history_indices = tuple(range(len(current_states)))
    if arguments.state_indices is not None:
        if max(arguments.state_indices) >= len(current_states):
            raise ValueError("an explicit state index is outside the source history")
        selection = torch.tensor(
            arguments.state_indices,
            dtype=torch.long,
            device=current_states.device,
        )
        current_states = current_states[selection]
        date_values = tuple(date_values[index] for index in arguments.state_indices)
        state_kinds = tuple(state_kinds[index] for index in arguments.state_indices)
        history_indices = tuple(arguments.state_indices)
    if arguments.maximum_states is not None:
        current_states = current_states[: arguments.maximum_states]
        date_values = date_values[: arguments.maximum_states]
        state_kinds = state_kinds[: arguments.maximum_states]
        history_indices = history_indices[: arguments.maximum_states]
    num_states = len(current_states)
    if (
        arguments.fit_states_per_sweep is not None
        and arguments.fit_states_per_sweep < num_states
        and num_states % arguments.fit_states_per_sweep
    ):
        raise ValueError("fit states per sweep must divide the selected initial states")
    dates = tuple(value.isoformat() for value in date_values)
    source_identity_sha256 = sources.get(
        "completion_sha256",
        sources.get("static_report_sha256"),
    )
    if source_identity_sha256 is None:
        raise ValueError("Q source has no immutable identity digest")
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    front_state_size = model.config.state_sizes_by_expiry[0]
    config: (
        JointGenericTransportConfig
        | GenericStateQLawConfig
        | FullStateGeneratorConfig
        | JointFullStateGeneratorConfig
        | JointUncompressedFrontGeneratorConfig
    )
    law: (
        JointGenericFrontTransport
        | IndividualFrontTransport
        | FullStateConditionalGenerator
        | JointFullStateConditionalGenerator
        | JointUncompressedFrontConditionalGenerator
    )
    if arguments.uncompressed_front_generator:
        with torch.no_grad():
            current_raw_states = model.raw_states_from_states(current_states).detach()
        config = JointUncompressedFrontGeneratorConfig(
            context_state_size=model.config.geometry.native_state_size,
            front_state_size=model.config.geometry.expiry_block_state_size,
            network=FeedForwardConfig(
                activation="SiLU",
                num_depth=arguments.generator_network_depth,
                num_width=arguments.generator_network_width,
            ),
        )
        law = JointUncompressedFrontConditionalGenerator(
            env,
            current_raw_states=current_raw_states,
            config=config,
        )
    elif arguments.full_generic_generator:
        if arguments.individual_transport:
            if num_states != 1:
                raise ValueError("an individual transport requires exactly one state")
            config = FullStateGeneratorConfig(
                front_state_size=front_state_size,
                maximum_increment=arguments.maximum_shift,
                network=FeedForwardConfig(
                    activation="SiLU",
                    num_depth=arguments.generator_network_depth,
                    num_width=arguments.generator_network_width,
                ),
            )
            law = FullStateConditionalGenerator(
                env,
                current_front_state=current_states[0, :front_state_size],
                config=config,
            )
        else:
            config = JointFullStateGeneratorConfig(
                front_state_size=front_state_size,
                maximum_increment=arguments.maximum_shift,
                network=FeedForwardConfig(
                    activation="SiLU",
                    num_depth=arguments.generator_network_depth,
                    num_width=arguments.generator_network_width,
                ),
            )
            law = JointFullStateConditionalGenerator(
                env,
                current_front_states=current_states[:, :front_state_size],
                config=config,
            )
    elif arguments.individual_transport:
        if num_states != 1:
            raise ValueError("an individual transport requires exactly one state")
        config = GenericStateQLawConfig(
            front_state_size=front_state_size,
            maximum_network_shift=arguments.maximum_shift,
            maximum_log_scale=arguments.maximum_log_scale,
            maximum_lower_loading=arguments.maximum_lower_loading,
        )
        law = IndividualFrontTransport(
            env,
            current_front_state=current_states[0, :front_state_size],
            config=config,
        )
    else:
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
        if arguments.initial_law_checkpoint is not None:
            initial_payload = torch.load(
                arguments.initial_law_checkpoint,
                map_location=env.device,
                weights_only=False,
            )
            if not isinstance(initial_payload, Mapping) or initial_payload.get(
                "source_identity_sha256"
            ) != source_identity_sha256:
                raise ValueError("initial Q law checkpoint is incompatible")
            if initial_payload.get("law_config") == asdict(config):
                law.load_state_dict(
                    cast(
                        Mapping[str, torch.Tensor],
                        initial_payload["law_state_dict"],
                    )
                )
                if isinstance(
                    law,
                    JointUncompressedFrontConditionalGenerator,
                ):
                    law.rebase_context_normalization(current_raw_states)
                affine_distillation = {
                    "cold_start": True,
                    "initial_law": "transferred-Q-law-weights",
                    **(
                        {
                            "context_normalization_reestimated": True,
                            "normalization_reference_states": len(
                                current_raw_states
                            ),
                        }
                        if isinstance(
                            law,
                            JointUncompressedFrontConditionalGenerator,
                        )
                        else {}
                    ),
                }
            elif arguments.full_generic_generator and arguments.individual_transport:
                affine_distillation = _distill_full_state_generator(
                    law=cast(FullStateConditionalGenerator, law),
                    teacher_payload=initial_payload,
                    current_front_state=current_states[0, :front_state_size],
                    steps=arguments.affine_distillation_steps,
                    paths_per_step=arguments.affine_distillation_paths,
                    learning_rate=arguments.affine_distillation_learning_rate,
                    seed=arguments.affine_distillation_seed,
                )
            else:
                raise ValueError("initial Q law checkpoint is incompatible")
            warm_start_provenance = {
                "checkpoint": str(arguments.initial_law_checkpoint),
                "checkpoint_sha256": checkpoint_digest(
                    arguments.initial_law_checkpoint
                ),
                "source_sweeps": len(
                    cast(list[Any], initial_payload.get("history", []))
                ),
                "weights_only_precision_transfer": True,
            }
        if affine_distillation is None:
            affine_distillation = {
                "cold_start": True,
                "initial_law": (
                    "joint-direct-uncompressed-front-G(u0,S/S0,epsilon)"
                    if arguments.uncompressed_front_generator
                    else "state-local-full-dense-G(epsilon,spot)"
                    if arguments.full_generic_generator and arguments.individual_transport
                    else "joint-full-dense-G(h0,spot,epsilon)"
                    if arguments.full_generic_generator
                    else "state-local-affine-plus-triangular-base-gaussian"
                    if arguments.individual_transport
                    else "zero-context-network-base-gaussian"
                ),
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
            or teacher.get("source_completion_sha256") != source_identity_sha256
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
            or affine_teacher.get("source_completion_sha256") != source_identity_sha256
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
                    or warm_start.get(
                        "source_identity_sha256",
                        warm_start.get("source_completion_sha256"),
                    )
                    != source_identity_sha256
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

    transport_state_size = law.config.front_state_size
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
        front_state_size=transport_state_size,
        total_paths=arguments.fit_paths,
        blocks=arguments.fit_blocks,
        seed=arguments.fit_seed,
    )
    validation_draws = _joint_path_blocks(
        env=env,
        front_state_size=transport_state_size,
        total_paths=arguments.validation_paths,
        blocks=arguments.validation_blocks,
        seed=arguments.validation_seed,
    )
    confirmation_draws = _joint_path_blocks(
        env=env,
        front_state_size=transport_state_size,
        total_paths=arguments.confirmation_paths,
        blocks=arguments.confirmation_blocks,
        seed=arguments.confirmation_seed,
    )
    validation_beta_draws = _joint_path_blocks(
        env=env,
        front_state_size=transport_state_size,
        total_paths=arguments.beta_calibration_paths,
        blocks=arguments.beta_calibration_blocks,
        seed=arguments.validation_beta_seed,
    )
    confirmation_beta_draws = _joint_path_blocks(
        env=env,
        front_state_size=transport_state_size,
        total_paths=arguments.beta_calibration_paths,
        blocks=arguments.beta_calibration_blocks,
        seed=arguments.confirmation_beta_seed,
    )
    gmm_calibration_draws = _joint_path_blocks(
        env=env,
        front_state_size=transport_state_size,
        total_paths=arguments.gmm_calibration_paths,
        blocks=arguments.gmm_calibration_blocks,
        seed=arguments.gmm_calibration_seed,
    )
    simulation_config = {
        "num_states": num_states,
        "floating_point_precision": arguments.precision,
        "model_initialization_seed": arguments.order_seed,
        "transport_scope": (
            "shared_direct_uncompressed_front_G(u0,S/S0,epsilon)"
            if arguments.uncompressed_front_generator
            else "individual_state_full_G(epsilon,spot)"
            if arguments.full_generic_generator and arguments.individual_transport
            else "shared_full_G(h0,spot,epsilon)"
            if arguments.full_generic_generator
            else "individual_state"
            if arguments.individual_transport
            else "shared_states"
        ),
        "date_range": [dates[0], dates[-1]],
        "draw_policy": DRAW_POLICY_ID,
        "one_spot_and_one_surface_state_per_path": True,
        "fit_paths": arguments.fit_paths,
        "fit_blocks": arguments.fit_blocks,
        "fit_states_per_sweep": (
            num_states
            if arguments.fit_states_per_sweep is None
            else min(arguments.fit_states_per_sweep, num_states)
        ),
        "fit_paths_per_block": arguments.fit_paths // arguments.fit_blocks,
        "validation_paths": arguments.validation_paths,
        "validation_blocks": arguments.validation_blocks,
        "validation_paths_per_block": (
            arguments.validation_paths // arguments.validation_blocks
        ),
        "development_monitor_paths": arguments.validation_paths,
        "development_monitor_blocks": arguments.validation_blocks,
        "development_monitor_interval_sweeps": arguments.monitoring_interval,
        "terminal_validation_paths": arguments.validation_paths,
        "terminal_validation_blocks": arguments.validation_blocks,
        "confirmation_paths": arguments.confirmation_paths,
        "confirmation_blocks": arguments.confirmation_blocks,
        "confirmation_paths_per_block": (
            arguments.confirmation_paths // arguments.confirmation_blocks
        ),
        "beta_calibration_paths": arguments.beta_calibration_paths,
        "beta_calibration_blocks": arguments.beta_calibration_blocks,
        "gmm_calibration_paths": arguments.gmm_calibration_paths,
        "gmm_calibration_blocks": arguments.gmm_calibration_blocks,
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
        "development_monitor_seed": arguments.validation_seed,
        "terminal_validation_seed": arguments.terminal_validation_seed,
        "confirmation_seed": arguments.confirmation_seed,
        "validation_beta_seed": arguments.validation_beta_seed,
        "development_monitor_beta_seed": arguments.validation_beta_seed,
        "terminal_validation_beta_seed": arguments.terminal_validation_beta_seed,
        "confirmation_beta_seed": arguments.confirmation_beta_seed,
        "gmm_calibration_seed": arguments.gmm_calibration_seed,
        "validation_bootstrap_seed": arguments.validation_bootstrap_seed,
        "terminal_validation_bootstrap_seed": (
            arguments.terminal_validation_bootstrap_seed
        ),
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
        "objective_gmm_weight": arguments.gmm_weight,
        "objective_absolute_drift_weight": arguments.absolute_drift_weight,
        "objective_absolute_drift_scale_bps": arguments.absolute_drift_scale_bps,
        "gmm_diagonal_shrinkage": arguments.gmm_diagonal_shrinkage,
        "gmm_ridge_fraction": arguments.gmm_ridge_fraction,
        "gmm_precision_policy": (
            "fixed-pilot-correlation-shrinkage-condition-cap"
            if arguments.cross_gmm_objective
            else "detached-two-timescale-covariance-ewa"
        ),
        "gmm_covariance_ewa_decay": arguments.gmm_covariance_ewa_decay,
        "gradient_clip_norm": arguments.gradient_clip_norm,
        "monitoring_gmm_per_moment_target": (
            arguments.monitoring_gmm_per_moment_target
        ),
        "common_random_numbers_across_states": True,
        "actual_expected_gross_spot_return": 1.0,
        **(
            {
                "cross_gmm_objective": True,
                "cross_gmm_bulk_weight": arguments.cross_gmm_bulk_weight,
                "cross_gmm_tail_weight": arguments.cross_gmm_tail_weight,
                "cross_gmm_tail_fraction": arguments.cross_gmm_tail_fraction,
                "training_precision_maximum_condition_number": (
                    arguments.training_precision_maximum_condition_number
                ),
            }
            if arguments.cross_gmm_objective
            else {}
        ),
        **(
            {
                "gmm_covariance_policy": arguments.gmm_covariance_policy,
                "gmm_correlation_eigenvalue_cutoff": (
                    arguments.gmm_correlation_eigenvalue_cutoff
                ),
            }
            if arguments.gmm_covariance_policy
            == BETA_FACTOR_RESIDUAL_SPECTRAL_COVARIANCE
            else {}
        ),
        **(
            {
                "monitoring_absolute_drift_bps_target": (
                    arguments.monitoring_absolute_drift_bps_target
                )
            }
            if arguments.monitoring_absolute_drift_bps_target is not None
            else {}
        ),
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
    gmm_covariance_ewa: torch.Tensor | None = None
    frozen_gmm_precision: torch.Tensor | None = None
    fixed_training_beta: torch.Tensor | None = None
    if arguments.checkpoint.exists():
        payload = torch.load(
            arguments.checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "flex-all-state-joint-generic-q-training-v1"
            or payload.get(
                "source_identity_sha256",
                payload.get("source_completion_sha256"),
            )
            != source_identity_sha256
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
        saved_gmm_precision = payload.get("frozen_gmm_precision")
        if saved_gmm_precision is not None:
            frozen_gmm_precision = cast(torch.Tensor, saved_gmm_precision).to(
                device=current_states.device,
                dtype=current_states.dtype,
            )
        saved_gmm_covariance = payload.get("gmm_covariance_ewa")
        if saved_gmm_covariance is not None:
            gmm_covariance_ewa = cast(torch.Tensor, saved_gmm_covariance).to(
                device=current_states.device,
                dtype=torch.float64,
            )
        saved_training_beta = payload.get("fixed_training_beta")
        if saved_training_beta is not None:
            fixed_training_beta = cast(torch.Tensor, saved_training_beta).to(
                device=current_states.device,
                dtype=current_states.dtype,
            )
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

    if (arguments.gmm_weight or arguments.cross_gmm_objective) and (
        frozen_gmm_precision is None
        or (arguments.cross_gmm_objective and fixed_training_beta is None)
    ):
        calibration_started = time.perf_counter()
        (
            gmm_covariance_ewa,
            frozen_gmm_precision,
            fixed_training_beta,
        ) = _calibrate_frozen_gmm_precision(
            model=model,
            law=law,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draws=gmm_calibration_draws,
            state_batch_size=arguments.audit_state_batch_size,
            diagonal_shrinkage=arguments.gmm_diagonal_shrinkage,
            ridge_fraction=arguments.gmm_ridge_fraction,
            covariance_policy=arguments.gmm_covariance_policy,
            correlation_eigenvalue_cutoff=(
                arguments.gmm_correlation_eigenvalue_cutoff
            ),
            maximum_precision_condition_number=(
                arguments.training_precision_maximum_condition_number
                if arguments.cross_gmm_objective
                else None
            ),
        )
        print(
            "calibrated frozen GMM precision "
            f"states={len(frozen_gmm_precision)} "
            f"paths={arguments.gmm_calibration_paths} "
            f"seconds={time.perf_counter() - calibration_started:.1f}",
            flush=True,
        )
    if frozen_gmm_precision is not None and frozen_gmm_precision.shape != (
        num_states,
        len(normalized_strikes),
        len(normalized_strikes),
    ):
        raise ValueError("frozen GMM precision has incompatible history shape")
    if gmm_covariance_ewa is not None and gmm_covariance_ewa.shape != (
        num_states,
        len(normalized_strikes),
        len(normalized_strikes),
    ):
        raise ValueError("GMM covariance EWA has incompatible history shape")
    if fixed_training_beta is not None and fixed_training_beta.shape != (
        num_states,
        len(normalized_strikes),
    ):
        raise ValueError("fixed training beta has incompatible history shape")

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
        beta_draws=validation_beta_draws,
        state_batch_size=arguments.audit_state_batch_size,
        z_target=arguments.validation_z_target,
        inner_tail_fraction=arguments.inner_tail_fraction,
        outer_tail_fraction=arguments.outer_tail_fraction,
        inner_tail_weight=arguments.inner_tail_weight,
        bootstrap_samples=(arguments.bootstrap_samples if arguments.audit_initial_only else 0),
        bootstrap_seed=arguments.validation_bootstrap_seed,
        zero_drift_covariance_policy=arguments.gmm_covariance_policy,
        zero_drift_correlation_eigenvalue_cutoff=(
            arguments.gmm_correlation_eigenvalue_cutoff
        ),
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
    initial_validation_monitoring_loss = float(
        cast(Mapping[str, Any], validation_audit["zero_drift"])[
            "mean_quadratic_per_moment"
        ]
    )
    best_validation_block_loss = min(
        best_validation_block_loss,
        initial_validation_monitoring_loss,
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
                    front_state_size=transport_state_size,
                    total_paths=arguments.confirmation_paths,
                    blocks=arguments.confirmation_blocks,
                    seed=seed,
                ),
                state_batch_size=arguments.audit_state_batch_size,
                z_target=arguments.confirmation_z_target,
                inner_tail_fraction=arguments.inner_tail_fraction,
                outer_tail_fraction=arguments.outer_tail_fraction,
                inner_tail_weight=arguments.inner_tail_weight,
                zero_drift_covariance_policy=arguments.gmm_covariance_policy,
                zero_drift_correlation_eigenvalue_cutoff=(
                    arguments.gmm_correlation_eigenvalue_cutoff
                ),
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
        validation_pass_streak = int(
            _monitoring_zero_drift_pass(
                validation_audit,
                quadratic_per_moment_target=(
                    arguments.monitoring_gmm_per_moment_target
                ),
                absolute_drift_bps_target=(
                    arguments.monitoring_absolute_drift_bps_target
                ),
            )
        )
    stopped_for_validation = validation_pass_streak >= arguments.validation_pass_sweeps
    stopped_for_plateau = False
    if not arguments.audit_initial_only and not stopped_for_validation:
        for sweep in range(start_sweep, arguments.sweeps):
            sweep_started = time.perf_counter()
            fit_draw_seed = arguments.fit_seed
            sweep_fit_draws = fit_draws
            if arguments.refresh_fit_seed_base is not None:
                fit_draw_seed = arguments.refresh_fit_seed_base + sweep * arguments.fit_blocks
                fit_draws = _joint_path_blocks(
                    env=env,
                    front_state_size=transport_state_size,
                    total_paths=arguments.fit_paths,
                    blocks=arguments.fit_blocks,
                    seed=fit_draw_seed,
                )
                sweep_fit_draws = fit_draws
            order = _cyclic_state_minibatch(
                num_states=num_states,
                states_per_sweep=arguments.fit_states_per_sweep,
                sweep=sweep,
                seed=arguments.order_seed,
            )
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
                frozen_gmm_precision=frozen_gmm_precision,
                fixed_training_beta=fixed_training_beta,
                gmm_weight=arguments.gmm_weight,
                absolute_drift_weight=arguments.absolute_drift_weight,
                absolute_drift_scale_bps=arguments.absolute_drift_scale_bps,
                cross_gmm_objective=arguments.cross_gmm_objective,
                cross_gmm_bulk_weight=arguments.cross_gmm_bulk_weight,
                cross_gmm_tail_weight=arguments.cross_gmm_tail_weight,
                cross_gmm_tail_fraction=arguments.cross_gmm_tail_fraction,
                gradient_clip_norm=arguments.gradient_clip_norm,
                gmm_covariance_policy=arguments.gmm_covariance_policy,
                gmm_diagonal_shrinkage=arguments.gmm_diagonal_shrinkage,
                gmm_ridge_fraction=arguments.gmm_ridge_fraction,
                gmm_correlation_eigenvalue_cutoff=(
                    arguments.gmm_correlation_eigenvalue_cutoff
                ),
            )
            if not arguments.cross_gmm_objective:
                detached_sweep_covariance = cast(
                    torch.Tensor,
                    fit_metrics.pop("detached_covariance_of_mean"),
                ).double()
                detached_covariance_indices = cast(
                    torch.Tensor,
                    fit_metrics.pop("detached_covariance_indices"),
                ).long()
            if arguments.gmm_weight and not arguments.cross_gmm_objective:
                if gmm_covariance_ewa is None:
                    if len(detached_covariance_indices) != num_states:
                        raise RuntimeError(
                            "subset covariance update requires an initial all-state "
                            "GMM calibration"
                        )
                    gmm_covariance_ewa = torch.empty_like(
                        detached_sweep_covariance
                    )
                    gmm_covariance_ewa[detached_covariance_indices] = (
                        detached_sweep_covariance
                    )
                else:
                    decay = arguments.gmm_covariance_ewa_decay
                    updated_covariance = (
                        decay * gmm_covariance_ewa[detached_covariance_indices]
                        + (1.0 - decay) * detached_sweep_covariance
                    )
                    gmm_covariance_ewa = gmm_covariance_ewa.clone()
                    gmm_covariance_ewa[detached_covariance_indices] = (
                        updated_covariance
                    )
                    gmm_covariance_ewa = gmm_covariance_ewa.detach()
                zero_mean = torch.zeros(
                    gmm_covariance_ewa.shape[:-1],
                    dtype=torch.float64,
                    device=current_states.device,
                )
                if arguments.gmm_covariance_policy == RAW_CORRELATION_COVARIANCE:
                    frozen_gmm_precision = quadratic_zero_drift(
                        zero_mean,
                        gmm_covariance_ewa,
                        diagonal_shrinkage=arguments.gmm_diagonal_shrinkage,
                        ridge_fraction=arguments.gmm_ridge_fraction,
                    ).precision.to(dtype=current_states.dtype)
                else:
                    frozen_gmm_precision = spectral_zero_drift(
                        zero_mean,
                        gmm_covariance_ewa,
                        correlation_eigenvalue_cutoff=(
                            arguments.gmm_correlation_eigenvalue_cutoff
                        ),
                    ).precision.to(dtype=current_states.dtype)
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
            monitoring_evaluated = _monitoring_is_due(
                completed_sweeps=sweep + 1,
                interval=arguments.monitoring_interval,
            )
            audit_seconds = 0.0
            validation_improved = False
            if monitoring_evaluated:
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
                    beta_draws=validation_beta_draws,
                    state_batch_size=arguments.audit_state_batch_size,
                    z_target=arguments.validation_z_target,
                    inner_tail_fraction=arguments.inner_tail_fraction,
                    outer_tail_fraction=arguments.outer_tail_fraction,
                    inner_tail_weight=arguments.inner_tail_weight,
                    zero_drift_covariance_policy=arguments.gmm_covariance_policy,
                    zero_drift_correlation_eigenvalue_cutoff=(
                        arguments.gmm_correlation_eigenvalue_cutoff
                    ),
                )
                audit_seconds = time.perf_counter() - audit_started
                if _monitoring_zero_drift_pass(
                    validation_audit,
                    quadratic_per_moment_target=(
                        arguments.monitoring_gmm_per_moment_target
                    ),
                    absolute_drift_bps_target=(
                        arguments.monitoring_absolute_drift_bps_target
                    ),
                ):
                    validation_pass_streak += 1
                else:
                    validation_pass_streak = 0
            validation_adjusted_worst = float(
                validation_audit["maximum_abs_noise_scaled_adjusted_mean_z"]
            )
            if validation_adjusted_worst < best_validation_adjusted_worst:
                best_validation_adjusted_worst = validation_adjusted_worst
            validation_block_loss = cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], validation_audit["block_objective"])[
                    "global_loss"
                ],
            )
            validation_block_loss_mean = float(validation_block_loss["mean"])
            validation_monitoring_loss = float(
                cast(Mapping[str, Any], validation_audit["zero_drift"])[
                    "mean_quadratic_per_moment"
                ]
            )
            if monitoring_evaluated:
                scheduler.step(validation_monitoring_loss)
                validation_improved = validation_monitoring_loss < (
                    best_validation_block_loss
                    * (1.0 - arguments.minimum_relative_improvement)
                )
                if validation_improved:
                    best_validation_block_loss = validation_monitoring_loss
                    stale_sweeps = 0
                else:
                    stale_sweeps += 1
            if validation_block_loss_mean < 0.0:
                raise AssertionError("a squared tail loss cannot be negative")
            record = {
                "sweep": float(sweep + 1),
                "fit_global_loss": fit_loss,
                "fit_tail_loss": fit_metrics["tail_loss"],
                "fit_gmm_loss": fit_metrics["gmm_loss"],
                "fit_absolute_drift_loss": fit_metrics["absolute_drift_loss"],
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
                "fit_clipped_gradient_norm": fit_metrics["clipped_gradient_norm"],
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
                "validation_gmm_per_moment": validation_monitoring_loss,
                "validation_block_loss_mean": validation_block_loss_mean,
                "validation_block_loss_median": float(validation_block_loss["median"]),
                "validation_block_loss_mc_standard_error": float(
                    validation_block_loss["mc_standard_error"]
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "fit_draw_seed": float(fit_draw_seed),
                "fit_fully_redrawn": float(arguments.refresh_fit_seed_base is not None),
                "validation_pass_streak": float(validation_pass_streak),
                "monitoring_evaluated": float(monitoring_evaluated),
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
                f"gmm={fit_metrics['gmm_loss']:.3f} "
                f"absolute={fit_metrics['absolute_drift_loss']:.3f} "
                f"gradient_norm_pre={fit_metrics['gradient_norm']:.3f} "
                f"gradient_norm_post={fit_metrics['clipped_gradient_norm']:.3f} "
                f"gradient_clip={arguments.gradient_clip_norm:.3f}",
                flush=True,
            )
            checkpoint = {
                "schema": "flex-all-state-joint-generic-q-training-v1",
                "status": "training",
                "objective": OBJECTIVE_ID,
                "source_identity_sha256": source_identity_sha256,
                "law_config": asdict(config),
                "simulation_config": simulation_config,
                "fit_draw_policy": {
                    "draw_policy": DRAW_POLICY_ID,
                    "refresh_seed_base": arguments.refresh_fit_seed_base,
                    "refresh_fraction": 1.0,
                    "seed_stride": arguments.fit_blocks,
                },
                "fit_draw_pool": (
                    fit_draws if arguments.refresh_fit_seed_base is None else None
                ),
                "frozen_gmm_precision": frozen_gmm_precision,
                "fixed_training_beta": fixed_training_beta,
                "gmm_covariance_ewa": gmm_covariance_ewa,
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
            if monitoring_evaluated:
                _audit_log(
                    f"sweep={sweep + 1} development_monitor",
                    validation_audit,
                    seconds=audit_seconds,
                )
            else:
                print(
                    f"sweep={sweep + 1} development_monitor skipped "
                    f"interval={arguments.monitoring_interval}",
                    flush=True,
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
            "source_identity_sha256": source_identity_sha256,
            "law_config": asdict(config),
            "simulation_config": simulation_config,
            "fit_draw_policy": {
                "draw_policy": DRAW_POLICY_ID,
                "refresh_seed_base": arguments.refresh_fit_seed_base,
                "refresh_fraction": 1.0,
                "seed_stride": arguments.fit_blocks,
            },
            "fit_draw_pool": (
                fit_draws if arguments.refresh_fit_seed_base is None else None
            ),
            "frozen_gmm_precision": frozen_gmm_precision,
            "fixed_training_beta": fixed_training_beta,
            "gmm_covariance_ewa": gmm_covariance_ewa,
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
            "schema": "flex-all-state-joint-generic-q-report-v2",
            "status": "development_monitor_incomplete",
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
            "development_monitor": validation_audit,
            "independent_validation": None,
            "independent_confirmation": None,
            "terminal_validation_inspected": False,
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
            "all-state joint generic Q has not passed sustained development "
            "monitoring; terminal validation and confirmation remain uninspected"
        )

    # The fixed development monitor may select this checkpoint, so inferential
    # validation must use a newly generated stream after the law is frozen.
    terminal_validation_draws = _joint_path_blocks(
        env=env,
        front_state_size=transport_state_size,
        total_paths=arguments.validation_paths,
        blocks=arguments.validation_blocks,
        seed=arguments.terminal_validation_seed,
    )
    terminal_validation_beta_draws = _joint_path_blocks(
        env=env,
        front_state_size=transport_state_size,
        total_paths=arguments.beta_calibration_paths,
        blocks=arguments.beta_calibration_blocks,
        seed=arguments.terminal_validation_beta_seed,
    )
    validation_bootstrap_started = time.perf_counter()
    terminal_validation_audit = _audit_all_states(
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
        draws=terminal_validation_draws,
        beta_draws=terminal_validation_beta_draws,
        state_batch_size=arguments.audit_state_batch_size,
        z_target=arguments.validation_z_target,
        inner_tail_fraction=arguments.inner_tail_fraction,
        outer_tail_fraction=arguments.outer_tail_fraction,
        inner_tail_weight=arguments.inner_tail_weight,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.terminal_validation_bootstrap_seed,
        zero_drift_covariance_policy=arguments.gmm_covariance_policy,
        zero_drift_correlation_eigenvalue_cutoff=(
            arguments.gmm_correlation_eigenvalue_cutoff
        ),
    )
    _audit_log(
        "independent_validation",
        terminal_validation_audit,
        seconds=time.perf_counter() - validation_bootstrap_started,
    )
    terminal_validation_zero_drift = cast(
        Mapping[str, Any], terminal_validation_audit["zero_drift"]
    )
    terminal_validation_equivalence = cast(
        Mapping[str, Any],
        terminal_validation_zero_drift["simultaneous_equivalence"],
    )
    terminal_validation_passes = bool(
        terminal_validation_zero_drift["zero_drift_not_rejected"]
        and terminal_validation_equivalence["passes_absolute_tolerance"]
    )
    if not terminal_validation_passes:
        report = {
            "schema": "flex-all-state-joint-generic-q-report-v2",
            "status": "failed_terminal_validation",
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
                "required_validation_pass_sweeps": (
                    arguments.validation_pass_sweeps
                ),
                "stopped_for_validation_target": stopped_for_validation,
                "stopped_for_plateau": stopped_for_plateau,
                "history": history,
            },
            "development_monitor": validation_audit,
            "independent_validation": terminal_validation_audit,
            "independent_confirmation": None,
            "terminal_validation_inspected": True,
            "confirmation_inspected": False,
            "sources": sources,
        }
        write_json_atomic(report, arguments.output)
        _atomic_torch_save(
            {
                **checkpoint,
                "status": "failed_terminal_validation",
                "report": str(arguments.output),
                "report_sha256": checkpoint_digest(arguments.output),
            },
            arguments.checkpoint,
        )
        raise RuntimeError(
            "all-state joint generic Q failed fresh terminal validation; "
            "confirmation remains uninspected"
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
        beta_draws=confirmation_beta_draws,
        state_batch_size=arguments.audit_state_batch_size,
        z_target=arguments.confirmation_z_target,
        inner_tail_fraction=arguments.inner_tail_fraction,
        outer_tail_fraction=arguments.outer_tail_fraction,
        inner_tail_weight=arguments.inner_tail_weight,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.confirmation_bootstrap_seed,
        zero_drift_covariance_policy=arguments.gmm_covariance_policy,
        zero_drift_correlation_eigenvalue_cutoff=(
            arguments.gmm_correlation_eigenvalue_cutoff
        ),
    )
    _audit_log(
        "independent_confirmation",
        confirmation_audit,
        seconds=time.perf_counter() - confirmation_started,
    )
    validation_zero_drift = cast(
        Mapping[str, Any], terminal_validation_audit["zero_drift"]
    )
    confirmation_zero_drift = cast(Mapping[str, Any], confirmation_audit["zero_drift"])
    validation_equivalence = cast(
        Mapping[str, Any], validation_zero_drift["simultaneous_equivalence"]
    )
    confirmation_equivalence = cast(
        Mapping[str, Any], confirmation_zero_drift["simultaneous_equivalence"]
    )
    zero_drift_not_rejected = bool(
        validation_zero_drift["zero_drift_not_rejected"]
        and confirmation_zero_drift["zero_drift_not_rejected"]
    )
    practical_equivalence = bool(
        validation_equivalence["passes_absolute_tolerance"]
        and confirmation_equivalence["passes_absolute_tolerance"]
    )
    semantic_success = zero_drift_not_rejected and practical_equivalence
    audit_consistency = _audit_consistency(
        terminal_validation_audit,
        confirmation_audit,
    )
    report = {
        "schema": "flex-all-state-joint-generic-q-report-v2",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "zero_drift_not_rejected_on_both_audits": zero_drift_not_rejected,
        "practical_equivalence_on_both_audits": practical_equivalence,
        "objective": OBJECTIVE_ID,
        "num_states": num_states,
        "num_model_fit_states": sum(value == "model_fit" for value in state_kinds),
        "num_held_out_2025_states": sum(value == "held_out_2025" for value in state_kinds),
        "date_range": [dates[0], dates[-1]],
        "law": {
            "form": (
                "shared direct dense G_theta(u0,S/S0,epsilon) in raw native front coordinates"
                if arguments.uncompressed_front_generator
                else "individual dense G_theta(epsilon,spot_rank)"
                if arguments.full_generic_generator
                else "individual triangular T_theta(spot_rank,epsilon)"
                if arguments.individual_transport
                else "one triangular T_theta(h1_0,spot_rank,epsilon)"
            ),
            "full_support": not (
                arguments.full_generic_generator
                or arguments.uncompressed_front_generator
            ),
            "invertible": not (
                arguments.full_generic_generator
                or arguments.uncompressed_front_generator
            ),
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
        "development_monitor": validation_audit,
        "independent_validation": terminal_validation_audit,
        "independent_confirmation": confirmation_audit,
        "terminal_validation_inspected": True,
        "confirmation_inspected": True,
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
                    key: terminal_validation_audit[key]
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
