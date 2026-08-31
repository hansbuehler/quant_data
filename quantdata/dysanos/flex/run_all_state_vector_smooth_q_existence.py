"""Fit independent statewise FLEX Q laws with a smooth option-return objective."""

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
from quantdata.dysanos.flex.model import bound_native_raw, dlvsq_cap_barrier
from quantdata.dysanos.flex.run_all_state_vector_q_existence import (
    AllStateQExistenceLaw,
    CommonQDraw,
    _evaluate_chunk,
    _state_slices,
    control_variate_noise_scale_by_state,
    normalize_front_components,
    pooled_spot_control_beta_by_state,
    state_batch_roll_returns,
)
from quantdata.dysanos.flex.run_atm_q_existence import (
    AtmQExistenceLawConfig,
    _atomic_torch_save,
    _load_inputs,
)
from quantdata.dysanos.flex.smooth_q_objective import (
    SmoothPriceObjectiveConfig,
    smooth_price_objective,
    studentized_block_sign_flip_test,
    validation_confirmation_consistency,
)
from quantdata.torch import BaseEnvironment

OBJECTIVE_ID = "statewise-iid-price-normalized-smooth-block-q-v1"
DRAW_POLICY_ID = "iid-one-spot-one-surface-balanced-branch-v1"
CHECKPOINT_SCHEMA = "flex-all-state-vector-smooth-q-training-v1"
REPORT_SCHEMA = "flex-all-state-vector-smooth-q-report-v1"
_SHARED_STRIKES_TRAINING_ROOT = (
    Path(quantdata_env.locations.training.path) / "flex-shared-strikes"
)
_DEFAULT_JOINT_COMPLETION = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "joint-hierarchical"
    / "v1"
    / "reports"
    / "joint-hierarchical-200-complete.json"
)
_DEFAULT_OUTPUT_ROOT = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "vector-all-history-iid10k-smooth-price-v1"
)


def _make_iid_statewise_draw(
    *,
    front_state_size: int,
    num_paths: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CommonQDraw:
    """Draw one spot and one complete affine-law innovation per path."""

    if front_state_size <= 0 or num_paths < 2 or num_paths % 2:
        raise ValueError("an IID statewise draw needs positive width and even paths")
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
    branch = torch.cat(
        (
            -torch.ones(num_paths // 2, dtype=dtype, device=device),
            torch.ones(num_paths // 2, dtype=dtype, device=device),
        )
    )
    branch = branch[torch.randperm(num_paths, device=device, generator=generator)]
    return CommonQDraw(
        component_uniform=component_uniform,
        backbone_uniform=backbone_uniform,
        branch=branch,
        epsilon=epsilon,
        spot_contexts=num_paths,
        surface_paths=1,
    )


def _iid_path_blocks(
    *,
    env: BaseEnvironment,
    front_state_size: int,
    total_paths: int,
    blocks: int,
    seed: int,
) -> list[CommonQDraw]:
    if blocks < 2 or total_paths % blocks:
        raise ValueError("IID path counts must divide into at least two blocks")
    paths_per_block = total_paths // blocks
    if paths_per_block < 2 or paths_per_block % 2:
        raise ValueError("each IID block needs an even number of paths")
    return [
        _make_iid_statewise_draw(
            front_state_size=front_state_size,
            num_paths=paths_per_block,
            seed=seed + block,
            dtype=env.fdtype,
            device=env.device,
        )
        for block in range(blocks)
    ]


def _worst_statistics(values: torch.Tensor) -> tuple[float, float]:
    absolute = torch.abs(values.detach().double()).reshape(-1)
    count = max(1, math.ceil(0.10 * len(absolute)))
    return (
        float(torch.max(absolute)),
        float(torch.mean(torch.topk(absolute, count).values)),
    )


def _moment_diagnostics(
    *,
    raw_block_means: torch.Tensor,
    adjusted_block_means: torch.Tensor,
    spot_block_means: torch.Tensor,
    beta: torch.Tensor,
) -> dict[str, Any]:
    blocks = len(raw_block_means)
    raw_mean = torch.mean(raw_block_means.double(), dim=0)
    adjusted_mean = torch.mean(adjusted_block_means.double(), dim=0)
    raw_se = torch.std(raw_block_means.double(), dim=0, correction=1) / math.sqrt(
        blocks
    )
    adjusted_se = torch.std(
        adjusted_block_means.double(), dim=0, correction=1
    ) / math.sqrt(blocks)
    spot_se = torch.std(spot_block_means.double(), dim=0, correction=1) / math.sqrt(
        blocks
    )
    noise_scale = control_variate_noise_scale_by_state(
        beta=beta.double(),
        spot_mc_standard_error=spot_se,
        raw_mc_standard_error=raw_se,
    )
    raw_z = raw_mean / raw_se.clamp_min(1.0e-15)
    adjusted_t = adjusted_mean / adjusted_se.clamp_min(1.0e-15)
    noise_scaled_adjusted_z = adjusted_mean / noise_scale.clamp_min(1.0e-15)
    raw_worst, raw_worst10 = _worst_statistics(raw_z)
    adjusted_t_worst, adjusted_t_worst10 = _worst_statistics(adjusted_t)
    adjusted_worst, adjusted_worst10 = _worst_statistics(noise_scaled_adjusted_z)
    absolute_noise_scaled = torch.abs(noise_scaled_adjusted_z).reshape(-1)
    worst_flat = int(torch.argmax(absolute_noise_scaled))
    instruments = raw_mean.shape[1]
    worst_state = worst_flat // instruments
    worst_instrument = worst_flat % instruments
    return {
        "maximum_abs_raw_mean_z": raw_worst,
        "worst_10_percent_mean_abs_raw_mean_z": raw_worst10,
        "maximum_abs_adjusted_mean_t": adjusted_t_worst,
        "worst_10_percent_mean_abs_adjusted_mean_t": adjusted_t_worst10,
        "maximum_abs_noise_scaled_adjusted_mean_z": adjusted_worst,
        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": adjusted_worst10,
        "worst_state_index": worst_state,
        "worst_instrument_index": worst_instrument,
        "worst_adjusted_mean_bps_spot": float(
            1.0e4 * adjusted_mean[worst_state, worst_instrument]
        ),
        "worst_noise_scale_bps_spot": float(
            1.0e4 * noise_scale[worst_state, worst_instrument]
        ),
    }


def _collect_block_means(
    *,
    model: Any,
    law: AllStateQExistenceLaw,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    state_batch_size: int,
    atm_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Collect detached block means and pooled statewise control slopes."""

    num_states = len(current_states)
    instruments = len(normalized_strikes)
    blocks = len(draws)
    raw_block_means = torch.empty(
        (blocks, num_states, instruments),
        dtype=torch.float64,
        device=current_states.device,
    )
    spot_block_means = torch.empty(
        (blocks, num_states),
        dtype=torch.float64,
        device=current_states.device,
    )
    beta = torch.empty(
        (num_states, instruments),
        dtype=torch.float64,
        device=current_states.device,
    )
    barrier_sum = 0.0
    total_paths = sum(draw.num_paths for draw in draws)
    with torch.no_grad():
        for state_slice in _state_slices(num_states, state_batch_size):
            indices = torch.arange(
                state_slice.start,
                state_slice.stop,
                device=current_states.device,
            )
            values = _evaluate_chunk(
                model=model,
                law=law,
                origin_indices=indices,
                current_states=current_states[state_slice],
                current_w2_calls=current_w2_calls[state_slice],
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support[state_slice],
                density=density[state_slice],
                sqrt_variance=sqrt_variance[state_slice],
                draws=draws,
                atm_index=atm_index,
            )
            raw_block_means[:, state_slice] = values[
                "raw_replication_means"
            ].double()
            spot_block_means[:, state_slice] = values[
                "spot_replication_means"
            ].double()
            beta[state_slice] = pooled_spot_control_beta_by_state(
                path_sum=values["path_sum"],
                spot_sum=values["spot_sum"],
                spot_square_sum=values["spot_square_sum"],
                spot_return_product_sum=values["spot_return_product_sum"],
                num_paths=total_paths,
            ).double()
            barrier_sum += float(torch.sum(values["cap_values"].double()))
    adjusted_block_means = (
        raw_block_means - spot_block_means[..., None] * beta[None]
    )
    mean_barrier_loss = barrier_sum / (blocks * num_states)
    return (
        raw_block_means,
        spot_block_means,
        beta,
        adjusted_block_means,
        mean_barrier_loss,
    )


def _audit(
    *,
    model: Any,
    law: AllStateQExistenceLaw,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    current_option_prices: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    state_batch_size: int,
    atm_index: int,
    objective_config: SmoothPriceObjectiveConfig,
    include_sign_flip: bool,
) -> tuple[dict[str, Any], torch.Tensor]:
    started = time.perf_counter()
    raw, spot, beta, adjusted, barrier_loss = _collect_block_means(
        model=model,
        law=law,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=draws,
        state_batch_size=state_batch_size,
        atm_index=atm_index,
    )
    _, _, objective = smooth_price_objective(
        adjusted,
        current_option_prices,
        config=objective_config,
        derivative=False,
    )
    diagnostics = _moment_diagnostics(
        raw_block_means=raw,
        adjusted_block_means=adjusted,
        spot_block_means=spot,
        beta=beta,
    )
    audit: dict[str, Any] = {
        "paths": sum(draw.num_paths for draw in draws),
        "blocks": len(draws),
        "paths_per_block": draws[0].num_paths,
        "objective": objective,
        "mean_dlvsq_cap_barrier_loss": barrier_loss,
        **diagnostics,
        "seconds": time.perf_counter() - started,
    }
    if include_sign_flip:
        audit["studentized_block_sign_flip"] = studentized_block_sign_flip_test(
            adjusted
        )
    return audit, adjusted.detach().cpu()


def _train_sweep(
    *,
    model: Any,
    law: AllStateQExistenceLaw,
    optimizer: torch.optim.Optimizer,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    current_option_prices: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    state_batch_size: int,
    atm_index: int,
    objective_config: SmoothPriceObjectiveConfig,
) -> dict[str, Any]:
    raw, spot, beta, adjusted, barrier_loss = _collect_block_means(
        model=model,
        law=law,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=draws,
        state_batch_size=state_batch_size,
        atm_index=atm_index,
    )
    _, mean_derivative, objective = smooth_price_objective(
        adjusted,
        current_option_prices,
        config=objective_config,
        derivative=True,
    )
    assert mean_derivative is not None
    diagnostics = _moment_diagnostics(
        raw_block_means=raw,
        adjusted_block_means=adjusted,
        spot_block_means=spot,
        beta=beta,
    )

    num_states = len(current_states)
    blocks = len(draws)
    optimizer.zero_grad(set_to_none=True)
    for state_slice in _state_slices(num_states, state_batch_size):
        indices = torch.arange(
            state_slice.start,
            state_slice.stop,
            device=current_states.device,
        )
        chunk_beta = beta[state_slice].to(dtype=current_states.dtype)
        for block, draw in enumerate(draws):
            returns, raw_next, gross = state_batch_roll_returns(
                model=model,
                law=law,
                origin_indices=indices,
                current_states=current_states[state_slice],
                current_w2_calls=current_w2_calls[state_slice],
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support[state_slice],
                density=density[state_slice],
                sqrt_variance=sqrt_variance[state_slice],
                draw=draw,
            )
            adjusted_mean = torch.mean(returns, dim=1) - chunk_beta * torch.mean(
                gross - 1.0,
                dim=1,
            )[..., None]
            moment_surrogate = torch.sum(
                mean_derivative[block, state_slice].to(dtype=current_states.dtype)
                * adjusted_mean
            )
            barrier = dlvsq_cap_barrier(raw_next, geometry=model.config.geometry)
            barrier_surrogate = (
                1.0e-4
                * (len(indices) / num_states)
                * torch.mean(barrier.square())
                / blocks
            )
            (moment_surrogate + barrier_surrogate).backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
    optimizer.step()
    return {
        "objective": objective,
        "mean_dlvsq_cap_barrier_loss": barrier_loss,
        "gradient_norm": float(gradient_norm.detach()),
        "minimum_state_noise_std": float(torch.min(law.noise_std.detach())),
        "maximum_state_noise_std": float(torch.max(law.noise_std.detach())),
        **diagnostics,
    }


def _log_loss(value: float) -> float:
    """Return the base-10 loss transform used by W&B log-loss panels."""

    return math.log10(max(float(value), 1.0e-30))


def _wandb_metrics(
    *,
    sweep: int,
    training: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    confirmation: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"sweep": sweep}
    if training is not None:
        objective = cast(Mapping[str, Any], training["objective"])
        metrics.update(
            {
                "training/loss": float(objective["loss"]),
                "training/log_loss": _log_loss(float(objective["loss"])),
                "training/full_loss": float(objective["full_loss"]),
                "training/log_full_loss": _log_loss(float(objective["full_loss"])),
                "training/robust_block_loss": float(objective["robust_block_loss"]),
                "training/log_robust_block_loss": _log_loss(
                    float(objective["robust_block_loss"])
                ),
                "training/price_normalized_worst": float(
                    objective["maximum_abs_price_normalized_mean"]
                ),
                "training/price_normalized_worst_10pct": float(
                    objective["worst_10_percent_mean_abs_price_normalized_mean"]
                ),
                "training/constraint_weight_ess": float(
                    objective["constraint_weight_effective_sample_size"]
                ),
                "training/constraint_weight_maximum": float(
                    objective["constraint_weight_maximum"]
                ),
                "training/raw_worst": float(training["maximum_abs_raw_mean_z"]),
                "training/raw_worst_10pct": float(
                    training["worst_10_percent_mean_abs_raw_mean_z"]
                ),
                "training/adjusted_worst": float(
                    training["maximum_abs_noise_scaled_adjusted_mean_z"]
                ),
                "training/adjusted_worst_10pct": float(
                    training["worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z"]
                ),
                "training/gradient_norm": float(training["gradient_norm"]),
                "training/minimum_state_noise_std": float(
                    training["minimum_state_noise_std"]
                ),
                "training/maximum_state_noise_std": float(
                    training["maximum_state_noise_std"]
                ),
            }
        )
    for prefix, audit in (("validation", validation), ("confirmation", confirmation)):
        if audit is None:
            continue
        objective = cast(Mapping[str, Any], audit["objective"])
        block_loss = cast(Mapping[str, Any], objective["block_loss"])
        metrics.update(
            {
                f"{prefix}/loss": float(objective["loss"]),
                f"{prefix}/log_loss": _log_loss(float(objective["loss"])),
                f"{prefix}/full_loss": float(objective["full_loss"]),
                f"{prefix}/log_full_loss": _log_loss(float(objective["full_loss"])),
                f"{prefix}/robust_block_loss": float(
                    objective["robust_block_loss"]
                ),
                f"{prefix}/log_robust_block_loss": _log_loss(
                    float(objective["robust_block_loss"])
                ),
                f"{prefix}/block_loss_mean": float(block_loss["mean"]),
                f"{prefix}/block_loss_median": float(block_loss["median"]),
                f"{prefix}/block_loss_mc_se": float(
                    block_loss["mc_standard_error"]
                ),
                f"{prefix}/block_loss_q10": float(block_loss["q10"]),
                f"{prefix}/block_loss_q25": float(block_loss["q25"]),
                f"{prefix}/block_loss_q75": float(block_loss["q75"]),
                f"{prefix}/block_loss_q90": float(block_loss["q90"]),
                f"{prefix}/raw_worst": float(audit["maximum_abs_raw_mean_z"]),
                f"{prefix}/raw_worst_10pct": float(
                    audit["worst_10_percent_mean_abs_raw_mean_z"]
                ),
                f"{prefix}/adjusted_worst": float(
                    audit["maximum_abs_noise_scaled_adjusted_mean_z"]
                ),
                f"{prefix}/adjusted_worst_10pct": float(
                    audit["worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z"]
                ),
                f"{prefix}/constraint_weight_ess": float(
                    objective["constraint_weight_effective_sample_size"]
                ),
            }
        )
        for index, value in enumerate(cast(list[float], block_loss["values"])):
            metrics[f"{prefix}/block_loss_{index:02d}"] = float(value)
            metrics[f"{prefix}/log_block_loss_{index:02d}"] = _log_loss(float(value))
        sign_flip = audit.get("studentized_block_sign_flip")
        if isinstance(sign_flip, Mapping):
            metrics.update(
                {
                    f"{prefix}/studentized_maximum_abs_t": float(
                        sign_flip["maximum_abs_t"]
                    ),
                    f"{prefix}/studentized_worst_10pct_mean_abs_t": float(
                        sign_flip["worst_tail_mean_abs_t"]
                    ),
                    f"{prefix}/sign_flip_maximum_p_value": float(
                        sign_flip["global_maximum_p_value"]
                    ),
                    f"{prefix}/sign_flip_tail_p_value": float(
                        sign_flip["global_tail_mean_p_value"]
                    ),
                }
            )
    return metrics


def _initialize_wandb(
    arguments: argparse.Namespace,
    *,
    simulation_config: Mapping[str, Any],
) -> Any | None:
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
            "wandb_log_loss_transform": "base_10_logarithm",
            **dict(simulation_config),
        },
    )
    run.define_metric("sweep")
    run.define_metric("*", step_metric="sweep")
    return run


def _audit_log(label: str, audit: Mapping[str, Any]) -> None:
    objective = cast(Mapping[str, Any], audit["objective"])
    block_loss = cast(Mapping[str, Any], objective["block_loss"])
    print(
        f"{label} loss={objective['loss']:.7g} "
        f"full={objective['full_loss']:.7g} "
        f"block_mean={block_loss['mean']:.7g} "
        f"block_median={block_loss['median']:.7g} "
        f"block_se={block_loss['mc_standard_error']:.3g} "
        f"price_worst={objective['maximum_abs_price_normalized_mean']:.4f} "
        f"raw_worst/worst10={audit['maximum_abs_raw_mean_z']:.3f}/"
        f"{audit['worst_10_percent_mean_abs_raw_mean_z']:.3f} "
        f"adjusted_worst/worst10="
        f"{audit['maximum_abs_noise_scaled_adjusted_mean_z']:.3f}/"
        f"{audit['worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z']:.3f} "
        f"ess={objective['constraint_weight_effective_sample_size']:.1f} "
        f"seconds={audit['seconds']:.1f}",
        flush=True,
    )
    print(
        f"{label} block_losses="
        + ",".join(f"{float(value):.7g}" for value in block_loss["values"]),
        flush=True,
    )
    sign_flip = audit.get("studentized_block_sign_flip")
    if isinstance(sign_flip, Mapping):
        print(
            f"{label} sign_flip max_t={sign_flip['maximum_abs_t']:.3f} "
            f"worst10_t={sign_flip['worst_tail_mean_abs_t']:.3f} "
            f"p_max={sign_flip['global_maximum_p_value']:.4f} "
            f"p_tail={sign_flip['global_tail_mean_p_value']:.4f} "
            f"simultaneous95={sign_flip['simultaneous_95_percent_abs_t_critical_value']:.3f}",
            flush=True,
        )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports/all-state-vector-smooth-q.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints/all-state-vector-smooth-q.pt",
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sweeps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--scheduler-patience-audits", type=int, default=6)
    parser.add_argument("--plateau-audits", type=int, default=30)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-4)
    parser.add_argument("--fit-paths", type=int, default=10_000)
    parser.add_argument("--fit-blocks", type=int, default=10)
    parser.add_argument("--validation-paths", type=int, default=100_000)
    parser.add_argument("--validation-blocks", type=int, default=10)
    parser.add_argument("--confirmation-paths", type=int, default=100_000)
    parser.add_argument("--confirmation-blocks", type=int, default=10)
    parser.add_argument("--fit-seed-base", type=int, default=22_484_101)
    parser.add_argument("--validation-seed", type=int, default=22_485_101)
    parser.add_argument("--confirmation-seed", type=int, default=22_487_101)
    parser.add_argument("--initialization-seed", type=int, default=22_483_101)
    parser.add_argument("--state-batch-size", type=int, default=64)
    parser.add_argument("--audit-state-batch-size", type=int, default=8)
    parser.add_argument("--audit-every", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0e-2)
    parser.add_argument("--robust-block-weight", type=float, default=0.25)
    parser.add_argument("--block-weight-decay", type=float, default=0.5)
    parser.add_argument("--minimum-option-price", type=float, default=1.0e-6)
    parser.add_argument("--maximum-states", type=int)
    parser.add_argument("--audit-initial-only", action="store_true")
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
        arguments.scheduler_patience_audits,
        arguments.plateau_audits,
        arguments.fit_paths,
        arguments.fit_blocks,
        arguments.validation_paths,
        arguments.validation_blocks,
        arguments.confirmation_paths,
        arguments.confirmation_blocks,
        arguments.state_batch_size,
        arguments.audit_state_batch_size,
        arguments.audit_every,
    )
    if arguments.device != "cuda:0" or any(value <= 0 for value in counts):
        raise ValueError("statewise smooth Q requires GPU0 and positive counts")
    if (
        arguments.fit_paths % arguments.fit_blocks
        or arguments.validation_paths % arguments.validation_blocks
        or arguments.confirmation_paths % arguments.confirmation_blocks
    ):
        raise ValueError("path totals must divide evenly into blocks")
    if any(
        (total // blocks) % 2
        for total, blocks in (
            (arguments.fit_paths, arguments.fit_blocks),
            (arguments.validation_paths, arguments.validation_blocks),
            (arguments.confirmation_paths, arguments.confirmation_blocks),
        )
    ):
        raise ValueError("every path block must contain an even number of paths")
    if len(
        {
            arguments.fit_seed_base,
            arguments.validation_seed,
            arguments.confirmation_seed,
        }
    ) != 3:
        raise ValueError("fit, validation, and confirmation seeds must differ")
    if (
        arguments.learning_rate <= 0.0
        or arguments.minimum_learning_rate <= 0.0
        or arguments.minimum_learning_rate > arguments.learning_rate
        or not 0.0 < arguments.minimum_relative_improvement < 1.0
    ):
        raise ValueError("learning-rate or improvement configuration is invalid")
    if arguments.maximum_states is not None and arguments.maximum_states <= 0:
        raise ValueError("maximum states must be positive")
    if arguments.wandb_project is None and any(
        value is not None
        for value in (
            arguments.wandb_entity,
            arguments.wandb_run_name,
            arguments.wandb_run_id,
        )
    ):
        raise ValueError("W&B entity, name, or ID requires --wandb-project")


def run(arguments: argparse.Namespace) -> None:
    _validate_arguments(arguments)
    objective_config = SmoothPriceObjectiveConfig(
        temperature=arguments.temperature,
        robust_block_weight=arguments.robust_block_weight,
        block_weight_decay=arguments.block_weight_decay,
        minimum_option_price=arguments.minimum_option_price,
    )
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
    dates = train_dates + validation_dates
    state_kinds = ("model_fit",) * len(train_dates) + ("held_out_2025",) * len(
        validation_dates
    )
    current_states = torch.cat((train_states, validation_states), dim=0)
    if arguments.maximum_states is not None:
        current_states = current_states[: arguments.maximum_states]
        dates = dates[: arguments.maximum_states]
        state_kinds = state_kinds[: arguments.maximum_states]
    num_states = len(current_states)
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    front_state_size = model.config.state_sizes_by_expiry[0]
    law_config = AtmQExistenceLawConfig(
        label="all-state-vector-smooth-price-v1",
        front_state_size=front_state_size,
    )
    law = AllStateQExistenceLaw(
        env,
        current_front_states=current_states[:, :front_state_size],
        config=law_config,
        initialization_seed=arguments.initialization_seed,
    )

    normalized_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    option_is_call = normalized_strikes <= 1.0
    atm_index = 5
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
                num_states,
                len(normalized_strikes),
            ),
            strikes=normalized_strikes[None].expand(num_states, -1),
        ).detach()
        current_option_prices = torch.where(
            option_is_call[None],
            current_w2_calls,
            current_w2_calls - 1.0 + normalized_strikes[None],
        )
        if not bool(
            torch.all(torch.isfinite(current_option_prices) & (current_option_prices > 0.0))
        ):
            raise FloatingPointError("current ITM option prices are not positive")
        support, density, sqrt_variance = normalize_front_components(
            current_surface.support_strikes[:, 0],
            current_surface.densities[:, 0],
            current_surface.backbone_total_vols[:, 0],
        )

    validation_draws = _iid_path_blocks(
        env=env,
        front_state_size=front_state_size,
        total_paths=arguments.validation_paths,
        blocks=arguments.validation_blocks,
        seed=arguments.validation_seed,
    )
    confirmation_draws = _iid_path_blocks(
        env=env,
        front_state_size=front_state_size,
        total_paths=arguments.confirmation_paths,
        blocks=arguments.confirmation_blocks,
        seed=arguments.confirmation_seed,
    )
    simulation_config = {
        "num_states": num_states,
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "draw_policy": DRAW_POLICY_ID,
        "one_spot_and_one_surface_state_per_path": True,
        "fit_paths": arguments.fit_paths,
        "fit_blocks": arguments.fit_blocks,
        "fit_paths_per_block": arguments.fit_paths // arguments.fit_blocks,
        "fit_redrawn_every_sweep": True,
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
        "fit_seed_base": arguments.fit_seed_base,
        "validation_seed": arguments.validation_seed,
        "confirmation_seed": arguments.confirmation_seed,
        "initialization_seed": arguments.initialization_seed,
        "objective_config": asdict(objective_config),
        "beta_scope": "separate_dataset_and_historical_state",
        "beta_gradient": "detached",
        "return_normalization": "current_ITM_option_premium",
        "validation_studentization": "ten_nonoverlapping_equal_blocks",
        "actual_expected_gross_spot_return": 1.0,
    }

    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=arguments.scheduler_patience_audits,
        min_lr=arguments.minimum_learning_rate,
    )
    history: list[dict[str, Any]] = []
    start_sweep = 0
    best_validation_loss = math.inf
    stale_audits = 0
    initial_provenance: dict[str, Any] | None = None
    latest_validation: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    if arguments.checkpoint.exists():
        payload = torch.load(
            arguments.checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("objective") != OBJECTIVE_ID
            or payload.get("source_completion_sha256") != sources["completion_sha256"]
            or payload.get("law_config") != asdict(law_config)
            or payload.get("simulation_config") != simulation_config
        ):
            raise ValueError("statewise smooth-Q checkpoint identity is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], payload["law_state_dict"]))
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state_dict"]))
        scheduler.load_state_dict(cast(dict[str, Any], payload["scheduler_state_dict"]))
        history = list(cast(list[dict[str, Any]], payload["history"]))
        start_sweep = len(history)
        best_validation_loss = float(payload.get("best_validation_loss", math.inf))
        stale_audits = int(payload.get("stale_audits", 0))
        latest = payload.get("latest_validation")
        latest_validation = dict(latest) if isinstance(latest, Mapping) else None
        provenance = payload.get("initial_provenance")
        initial_provenance = (
            dict(provenance) if isinstance(provenance, Mapping) else None
        )
        checkpoint = dict(payload)
        print(
            f"resuming statewise smooth Q at sweep {start_sweep} "
            f"with best validation loss {best_validation_loss:.7g}",
            flush=True,
        )
    elif arguments.initial_checkpoint is not None:
        payload = torch.load(
            arguments.initial_checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        source_schema = payload.get("schema") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or source_schema
            not in {
                "flex-all-state-vector-q-training-v1",
                CHECKPOINT_SCHEMA,
            }
            or payload.get("source_completion_sha256") != sources["completion_sha256"]
            or payload.get("law_config") != asdict(law_config)
        ):
            raise ValueError("initial statewise affine checkpoint is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], payload["law_state_dict"]))
        initial_provenance = {
            "checkpoint": str(arguments.initial_checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.initial_checkpoint),
            "source_sweeps": len(cast(list[Any], payload["history"])),
            "source_schema": source_schema,
            "source_objective": payload.get("objective"),
            "source_status": payload.get("status"),
            "optimizer_state_loaded": False,
        }

    wandb_run = _initialize_wandb(
        arguments,
        simulation_config=simulation_config,
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

    if start_sweep > 0 and latest_validation is not None:
        initial_validation = latest_validation
        _audit_log("checkpoint_validation", initial_validation)
    else:
        initial_validation, _ = _audit(
            model=model,
            law=law,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            current_option_prices=current_option_prices,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draws=validation_draws,
            state_batch_size=arguments.audit_state_batch_size,
            atm_index=atm_index,
            objective_config=objective_config,
            include_sign_flip=True,
        )
        _audit_log("initial_validation", initial_validation)
        if wandb_run is not None:
            wandb_run.log(
                _wandb_metrics(sweep=start_sweep, validation=initial_validation),
                step=start_sweep,
            )
    if arguments.audit_initial_only:
        write_json_atomic(
            {
                "schema": REPORT_SCHEMA,
                "status": "initial_audit_complete",
                "objective": OBJECTIVE_ID,
                "simulation_config": simulation_config,
                "initial_validation": initial_validation,
                "sources": sources,
            },
            arguments.output,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    stopped_for_plateau = False
    for sweep in range(start_sweep, arguments.sweeps):
        sweep_started = time.perf_counter()
        fit_seed = arguments.fit_seed_base + sweep * arguments.fit_blocks
        fit_draws = _iid_path_blocks(
            env=env,
            front_state_size=front_state_size,
            total_paths=arguments.fit_paths,
            blocks=arguments.fit_blocks,
            seed=fit_seed,
        )
        training = _train_sweep(
            model=model,
            law=law,
            optimizer=optimizer,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            current_option_prices=current_option_prices,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draws=fit_draws,
            state_batch_size=arguments.state_batch_size,
            atm_index=atm_index,
            objective_config=objective_config,
        )
        objective = cast(Mapping[str, Any], training["objective"])
        record: dict[str, Any] = {
            "sweep": sweep + 1,
            "fit_seed": fit_seed,
            "training": training,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "sweep_seconds": time.perf_counter() - sweep_started,
        }
        print(
            f"sweep={sweep + 1} train loss={objective['loss']:.7g} "
            f"full={objective['full_loss']:.7g} "
            f"price_worst={objective['maximum_abs_price_normalized_mean']:.4f} "
            f"raw_worst/worst10={training['maximum_abs_raw_mean_z']:.3f}/"
            f"{training['worst_10_percent_mean_abs_raw_mean_z']:.3f} "
            f"adjusted_worst/worst10="
            f"{training['maximum_abs_noise_scaled_adjusted_mean_z']:.3f}/"
            f"{training['worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z']:.3f} "
            f"ess={objective['constraint_weight_effective_sample_size']:.1f} "
            f"grad={training['gradient_norm']:.3f} "
            f"lr={record['learning_rate']:.6g}",
            flush=True,
        )

        validation_improved = False
        if (sweep + 1) % arguments.audit_every == 0:
            latest_validation, _ = _audit(
                model=model,
                law=law,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                current_option_prices=current_option_prices,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=validation_draws,
                state_batch_size=arguments.audit_state_batch_size,
                atm_index=atm_index,
                objective_config=objective_config,
                include_sign_flip=True,
            )
            validation_objective = cast(
                Mapping[str, Any], latest_validation["objective"]
            )
            validation_loss = float(validation_objective["loss"])
            scheduler.step(validation_loss)
            if validation_loss < best_validation_loss * (
                1.0 - arguments.minimum_relative_improvement
            ):
                best_validation_loss = validation_loss
                stale_audits = 0
                validation_improved = True
            else:
                stale_audits += 1
            record["validation"] = latest_validation
            record["stale_audits"] = stale_audits
            _audit_log(f"sweep={sweep + 1} validation", latest_validation)

        history.append(record)
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "status": "training",
            "objective": OBJECTIVE_ID,
            "source_completion_sha256": sources["completion_sha256"],
            "law_config": asdict(law_config),
            "simulation_config": simulation_config,
            "law_state_dict": law.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_validation_loss": best_validation_loss,
            "stale_audits": stale_audits,
            "latest_validation": latest_validation,
            "initial_provenance": initial_provenance,
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
        write_json_atomic(
            {
                "schema": REPORT_SCHEMA,
                "status": "training",
                "objective": OBJECTIVE_ID,
                "sweeps_completed": len(history),
                "best_validation_loss": best_validation_loss,
                "stale_audits": stale_audits,
                "latest_training": training,
                "latest_validation": latest_validation,
                "wandb_provenance": wandb_provenance,
                "simulation_config": simulation_config,
                "sources": sources,
            },
            arguments.output,
        )
        if wandb_run is not None:
            wandb_run.log(
                _wandb_metrics(
                    sweep=sweep + 1,
                    training=training,
                    validation=latest_validation if "validation" in record else None,
                ),
                step=sweep + 1,
            )
        if (
            stale_audits >= arguments.plateau_audits
            and optimizer.param_groups[0]["lr"]
            <= arguments.minimum_learning_rate * (1.0 + 1.0e-12)
        ):
            stopped_for_plateau = True
            break

    if checkpoint is None:
        raise RuntimeError("statewise smooth-Q training produced no checkpoint")

    validation_audit, validation_blocks = _audit(
        model=model,
        law=law,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        current_option_prices=current_option_prices,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=validation_draws,
        state_batch_size=arguments.audit_state_batch_size,
        atm_index=atm_index,
        objective_config=objective_config,
        include_sign_flip=True,
    )
    confirmation_audit, confirmation_blocks = _audit(
        model=model,
        law=law,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        current_option_prices=current_option_prices,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=confirmation_draws,
        state_batch_size=arguments.audit_state_batch_size,
        atm_index=atm_index,
        objective_config=objective_config,
        include_sign_flip=True,
    )
    consistency = validation_confirmation_consistency(
        validation_blocks,
        confirmation_blocks,
    )
    _audit_log("final_validation", validation_audit)
    _audit_log("independent_confirmation", confirmation_audit)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "objective": OBJECTIVE_ID,
        "num_states": num_states,
        "num_model_fit_states": sum(value == "model_fit" for value in state_kinds),
        "num_held_out_2025_states": sum(
            value == "held_out_2025" for value in state_kinds
        ),
        "simulation_config": simulation_config,
        "training": {
            "sweeps_completed": len(history),
            "best_validation_loss": best_validation_loss,
            "stale_audits": stale_audits,
            "stopped_for_plateau": stopped_for_plateau,
            "history": history,
        },
        "validation": validation_audit,
        "confirmation": confirmation_audit,
        "validation_confirmation_consistency": consistency,
        "initial_provenance": initial_provenance,
        "wandb_provenance": wandb_provenance,
        "sources": sources,
    }
    write_json_atomic(report, arguments.output)
    _atomic_torch_save(
        {
            **checkpoint,
            "status": "complete",
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
        },
        arguments.checkpoint,
    )
    if wandb_run is not None:
        wandb_run.log(
            _wandb_metrics(
                sweep=len(history),
                validation=validation_audit,
                confirmation=confirmation_audit,
            ),
            step=len(history),
        )
        wandb_run.finish()
    print(json.dumps({"validation": validation_audit, "confirmation": confirmation_audit}, indent=2), flush=True)


def main() -> None:
    arguments = _parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_all_state_vector_smooth_q_existence",
        clean_sigterm=True,
    )
