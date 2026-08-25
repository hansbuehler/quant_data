"""Fit the eleven-option conditional-Q existence law at every historical state."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw, dlvsq_cap_barrier
from quantdata.dysanos.flex.run_atm_q_existence import (
    AtmQExistenceLawConfig,
    _atomic_torch_save,
    _load_inputs,
    _price_calls,
)
from quantdata.torch import BaseEnvironment, BaseModel

OBJECTIVE_ID = "all-state-common-random-eleven-option-moments-v1"
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
    / "vector-all-history-exact-common-random-v1"
)


@dataclass(frozen=True)
class CommonQDraw:
    """Random inputs reused without change at every historical origin state."""

    component_uniform: torch.Tensor
    backbone_uniform: torch.Tensor
    branch: torch.Tensor
    epsilon: torch.Tensor
    spot_contexts: int
    surface_paths: int

    @property
    def num_paths(self) -> int:
        return self.spot_contexts * self.surface_paths


def make_common_q_draw(
    *,
    front_state_size: int,
    spot_contexts: int,
    surface_paths: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CommonQDraw:
    """Construct one balanced replication shared by all origin states."""
    if spot_contexts < 2 or surface_paths < 4 or surface_paths % 4:
        raise ValueError("common draw needs two spots and paths divisible by four")
    generator = torch.Generator(device=device).manual_seed(seed)
    base = (torch.arange(spot_contexts, dtype=dtype, device=device) + 0.5) / (
        spot_contexts
    )
    component_shift = torch.rand((), dtype=dtype, device=device, generator=generator)
    backbone_shift = torch.rand((), dtype=dtype, device=device, generator=generator)
    component_uniform = torch.remainder(base + component_shift, 1.0)
    backbone_order = torch.randperm(spot_contexts, device=device, generator=generator)
    backbone_uniform = torch.remainder(
        base[backbone_order] + backbone_shift,
        1.0,
    )
    floor = 4.0 * torch.finfo(dtype).eps
    component_uniform = torch.clamp(component_uniform, floor, 1.0 - floor)
    backbone_uniform = torch.clamp(backbone_uniform, floor, 1.0 - floor)

    quarter = surface_paths // 4
    epsilon_quarter = torch.randn(
        (spot_contexts, quarter, front_state_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    epsilon_half = torch.cat((epsilon_quarter, -epsilon_quarter), dim=1)
    epsilon = torch.cat((epsilon_half, epsilon_half), dim=1).reshape(
        spot_contexts * surface_paths,
        front_state_size,
    )
    branch = torch.cat(
        (
            -torch.ones(surface_paths // 2, dtype=dtype, device=device),
            torch.ones(surface_paths // 2, dtype=dtype, device=device),
        )
    ).repeat(spot_contexts)
    return CommonQDraw(
        component_uniform=component_uniform,
        backbone_uniform=backbone_uniform,
        branch=branch,
        epsilon=epsilon,
        spot_contexts=spot_contexts,
        surface_paths=surface_paths,
    )


def normalize_front_components(
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize batched SANOS mixtures to unit mass and unit spot mean."""
    if (
        support.ndim != 2
        or density.shape != support.shape
        or sqrt_variance.shape != support.shape[:1]
    ):
        raise ValueError("batched front components have inconsistent shapes")
    sampling_density = torch.clamp_min(density.detach(), 0.0)
    mass = torch.sum(sampling_density, dim=1, keepdim=True)
    if not bool(torch.all(torch.isfinite(mass) & (mass > 0.0))):
        raise ValueError("a front SANOS mixture has no positive finite mass")
    sampling_density = sampling_density / mass
    mean = torch.sum(sampling_density * support.detach(), dim=1, keepdim=True)
    if not bool(torch.all(torch.isfinite(mean) & (mean > 0.0))):
        raise ValueError("a front SANOS mixture has no positive finite mean")
    if not bool(torch.all(torch.isfinite(sqrt_variance) & (sqrt_variance > 0.0))):
        raise ValueError("front backbone total volatilities must be positive")
    return support.detach() / mean, sampling_density, sqrt_variance.detach()


def sample_front_components_common(
    *,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draw: CommonQDraw,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform one common draw through every supplied front marginal."""
    if (
        support.ndim != 2
        or density.shape != support.shape
        or sqrt_variance.shape != support.shape[:1]
    ):
        raise ValueError("batched front components have inconsistent shapes")
    origins = len(support)
    cumulative = torch.cumsum(density, dim=1)
    cumulative[:, -1] = 1.0
    uniforms = draw.component_uniform.unsqueeze(0).expand(origins, -1).contiguous()
    component = torch.searchsorted(cumulative.contiguous(), uniforms).clamp_max(
        support.shape[1] - 1
    )
    component_support = torch.gather(support, 1, component)
    backbone_normal = torch.special.ndtri(draw.backbone_uniform)
    gross_context = component_support * torch.exp(
        -0.5 * sqrt_variance[:, None].square()
        + sqrt_variance[:, None] * backbone_normal[None]
    )
    standardized = (
        torch.log(gross_context)[..., None]
        - torch.log(support)[:, None]
        + 0.5 * sqrt_variance[:, None, None].square()
    ) / sqrt_variance[:, None, None]
    mixture_probability = torch.sum(
        density[:, None] * torch.special.ndtr(standardized),
        dim=-1,
    )
    floor = 4.0 * torch.finfo(support.dtype).eps
    spot_rank_context = torch.special.ndtri(
        torch.clamp(mixture_probability, floor, 1.0 - floor)
    )
    gross = gross_context[..., None].expand(-1, -1, draw.surface_paths).reshape(
        origins,
        draw.num_paths,
    )
    spot_rank = spot_rank_context[..., None].expand(
        -1, -1, draw.surface_paths
    ).reshape(origins, draw.num_paths)
    if not bool(torch.all(torch.isfinite(gross) & (gross > 0.0))):
        raise FloatingPointError("common front mixture produced invalid spot paths")
    if not bool(torch.all(torch.isfinite(spot_rank))):
        raise FloatingPointError("common front mixture produced invalid PIT ranks")
    return gross, spot_rank


class AllStateQExistenceLaw(BaseModel):
    """Independent full-support conditional-law parameters for every origin."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        current_front_states: torch.Tensor,
        config: AtmQExistenceLawConfig,
        initialization_seed: int,
    ) -> None:
        super().__init__(env)
        if (
            current_front_states.ndim != 2
            or current_front_states.shape[1] != config.front_state_size
        ):
            raise ValueError("all-state law received invalid current front states")
        self.config = config
        self.register_buffer(
            "current_front_states", current_front_states.detach().clone()
        )
        self.location_shift = torch.nn.Parameter(torch.zeros_like(current_front_states))
        self.spot_loading = torch.nn.Parameter(torch.zeros_like(current_front_states))
        generator = torch.Generator(device=env.device).manual_seed(initialization_seed)
        branch = torch.randn(
            current_front_states.shape,
            dtype=env.fdtype,
            device=env.device,
            generator=generator,
        )
        branch = 0.05 * branch / torch.linalg.vector_norm(
            branch, dim=1, keepdim=True
        )
        self.branch_direction = torch.nn.Parameter(branch)
        initial_excess_std = (
            config.initial_state_noise_std - config.minimum_state_noise_std
        )
        self.raw_noise_std = torch.nn.Parameter(
            torch.full(
                current_front_states.shape[:1],
                math.log(math.expm1(initial_excess_std)),
                dtype=env.fdtype,
                device=env.device,
            )
        )

    @property
    def noise_std(self) -> torch.Tensor:
        return self.config.minimum_state_noise_std + torch.nn.functional.softplus(
            self.raw_noise_std
        )

    def forward(
        self,
        *,
        origin_indices: torch.Tensor,
        spot_rank: torch.Tensor,
        branch: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        if (
            origin_indices.ndim != 1
            or spot_rank.ndim != 2
            or spot_rank.shape[0] != len(origin_indices)
            or branch.shape != spot_rank.shape[1:]
            or epsilon.shape
            != (spot_rank.shape[1], self.config.front_state_size)
        ):
            raise ValueError("all-state conditional-law inputs are inconsistent")
        return (
            self.current_front_states[origin_indices, None]
            + self.location_shift[origin_indices, None]
            + spot_rank[..., None] * self.spot_loading[origin_indices, None]
            + branch[None, :, None] * self.branch_direction[origin_indices, None]
            + self.noise_std[origin_indices, None, None] * epsilon[None]
        )


def spot_control_variate_by_state(
    returns: torch.Tensor,
    gross_spot_return: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the exact spot control independently by state and instrument."""
    if returns.ndim != 3 or gross_spot_return.shape != returns.shape[:2]:
        raise ValueError("statewise spot control requires aligned path tensors")
    spot_return = gross_spot_return - 1.0
    centered_spot = spot_return - torch.mean(spot_return, dim=1, keepdim=True)
    centered_returns = returns - torch.mean(returns, dim=1, keepdim=True)
    denominator = torch.sum(centered_spot.square(), dim=1).clamp_min(
        torch.finfo(returns.dtype).tiny
    )
    beta = (
        torch.sum(centered_spot[..., None] * centered_returns, dim=1)
        / denominator[:, None]
    ).detach()
    adjusted = returns - spot_return[..., None] * beta[:, None]
    return adjusted, torch.mean(adjusted, dim=1), beta


def state_batch_roll_returns(
    *,
    model: Any,
    law: AllStateQExistenceLaw,
    origin_indices: torch.Tensor,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draw: CommonQDraw,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Value all eleven rolls for one state microbatch and common draw."""
    gross, spot_rank = sample_front_components_common(
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draw=draw,
    )
    next_front = law(
        origin_indices=origin_indices,
        spot_rank=spot_rank,
        branch=draw.branch,
        epsilon=draw.epsilon,
    )
    origins, paths = gross.shape
    next_state = current_states[:, None].expand(-1, paths, -1).clone()
    next_state[..., : law.config.front_state_size] = next_front
    next_normalized_strikes = (
        normalized_strikes[None, None] / gross[..., None]
    )
    next_w1_calls, raw = _price_calls(
        model,
        next_state.reshape(origins * paths, -1),
        expiry_index=0,
        normalized_strikes=next_normalized_strikes.reshape(origins * paths, -1),
    )
    next_w1_calls = next_w1_calls.reshape(origins, paths, -1)
    call_returns = gross[..., None] * next_w1_calls - current_w2_calls[:, None]
    returns = torch.where(
        option_is_call[None, None],
        call_returns,
        call_returns - (gross - 1.0)[..., None],
    )
    return returns, raw, gross


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports" / "all-state-vector-q.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints" / "all-state-vector-q.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--fit-replications", type=int, default=24)
    parser.add_argument("--validation-replications", type=int, default=24)
    parser.add_argument("--spot-contexts", type=int, default=125)
    parser.add_argument("--surface-paths", type=int, default=8)
    parser.add_argument("--state-batch-size", type=int, default=16)
    parser.add_argument("--fit-seed", type=int, default=9_484_101)
    parser.add_argument("--validation-seed", type=int, default=9_485_101)
    parser.add_argument("--initialization-seed", type=int, default=9_483_101)
    parser.add_argument("--plateau-epochs", type=int, default=100)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-5)
    parser.add_argument("--validation-z-target", type=float, default=2.0)
    parser.add_argument("--audit-every", type=int, default=25)
    parser.add_argument("--maximum-states", type=int)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    counts = (
        arguments.epochs,
        arguments.fit_replications,
        arguments.validation_replications,
        arguments.spot_contexts,
        arguments.surface_paths,
        arguments.state_batch_size,
        arguments.plateau_epochs,
        arguments.audit_every,
    )
    if arguments.device != "cuda:0" or any(value <= 0 for value in counts):
        raise ValueError("all-state Q arguments require GPU0 and positive counts")
    if arguments.fit_replications < 2 or arguments.validation_replications < 2:
        raise ValueError("MC standard errors require at least two replications")
    if arguments.surface_paths < 4 or arguments.surface_paths % 4:
        raise ValueError("surface paths must be at least four and divisible by four")
    if arguments.fit_seed == arguments.validation_seed:
        raise ValueError("fit and validation streams must be disjoint")
    if not 0.0 < arguments.minimum_relative_improvement < 1.0:
        raise ValueError("minimum relative improvement must lie in (0, 1)")
    if arguments.validation_z_target <= 0.0:
        raise ValueError("validation z target must be positive")
    if arguments.maximum_states is not None and arguments.maximum_states <= 0:
        raise ValueError("maximum states must be positive when supplied")


def _common_draws(
    arguments: argparse.Namespace,
    *,
    seed: int,
    front_state_size: int,
    env: BaseEnvironment,
    replications: int,
) -> list[CommonQDraw]:
    return [
        make_common_q_draw(
            front_state_size=front_state_size,
            spot_contexts=arguments.spot_contexts,
            surface_paths=arguments.surface_paths,
            seed=seed + replication,
            dtype=env.fdtype,
            device=env.device,
        )
        for replication in range(replications)
    ]


def _state_slices(num_states: int, state_batch_size: int) -> list[slice]:
    return [
        slice(start, min(start + state_batch_size, num_states))
        for start in range(0, num_states, state_batch_size)
    ]


def _evaluate_chunk(
    *,
    model: Any,
    law: AllStateQExistenceLaw,
    origin_indices: torch.Tensor,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    atm_index: int,
) -> dict[str, torch.Tensor]:
    """Evaluate fixed replications for one origin-state microbatch."""
    adjusted_means: list[torch.Tensor] = []
    raw_means: list[torch.Tensor] = []
    negative_atm_means: list[torch.Tensor] = []
    positive_atm_means: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    cap_values: list[torch.Tensor] = []
    path_sums: list[torch.Tensor] = []
    path_square_sums: list[torch.Tensor] = []
    path_positive_counts: list[torch.Tensor] = []
    gross_sums: list[torch.Tensor] = []
    for draw in draws:
        returns, raw_next, gross = state_batch_roll_returns(
            model=model,
            law=law,
            origin_indices=origin_indices,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draw=draw,
        )
        _, adjusted_mean, beta = spot_control_variate_by_state(returns, gross)
        negative = draw.branch < 0.0
        positive = draw.branch > 0.0
        _, negative_atm, _ = spot_control_variate_by_state(
            returns[:, negative, atm_index : atm_index + 1],
            gross[:, negative],
        )
        _, positive_atm, _ = spot_control_variate_by_state(
            returns[:, positive, atm_index : atm_index + 1],
            gross[:, positive],
        )
        origins = len(origin_indices)
        paths = draw.num_paths
        barrier = dlvsq_cap_barrier(
            raw_next, geometry=model.config.geometry
        ).reshape(origins, paths, -1)
        adjusted_means.append(adjusted_mean)
        raw_means.append(torch.mean(returns, dim=1))
        negative_atm_means.append(negative_atm[:, 0])
        positive_atm_means.append(positive_atm[:, 0])
        betas.append(beta)
        cap_values.append(1.0e-4 * torch.mean(barrier.square(), dim=(1, 2)))
        values = returns.double()
        path_sums.append(torch.sum(values, dim=1))
        path_square_sums.append(torch.sum(values.square(), dim=1))
        path_positive_counts.append(torch.sum(values > 0.0, dim=1))
        gross_sums.append(torch.sum(gross.double(), dim=1))
    return {
        "adjusted_replication_means": torch.stack(adjusted_means),
        "raw_replication_means": torch.stack(raw_means),
        "negative_atm_replication_means": torch.stack(negative_atm_means),
        "positive_atm_replication_means": torch.stack(positive_atm_means),
        "betas": torch.stack(betas),
        "cap_values": torch.stack(cap_values),
        "path_sum": torch.sum(torch.stack(path_sums), dim=0),
        "path_square_sum": torch.sum(torch.stack(path_square_sums), dim=0),
        "path_positive_count": torch.sum(
            torch.stack(path_positive_counts), dim=0
        ),
        "gross_sum": torch.sum(torch.stack(gross_sums), dim=0),
    }


def _summary_quantiles(values: torch.Tensor) -> dict[str, float]:
    finite = values.detach().double().reshape(-1).cpu()
    return {
        "minimum": float(torch.min(finite)),
        "p50": float(torch.quantile(finite, 0.50)),
        "p90": float(torch.quantile(finite, 0.90)),
        "p95": float(torch.quantile(finite, 0.95)),
        "p99": float(torch.quantile(finite, 0.99)),
        "maximum": float(torch.max(finite)),
    }


def _audit_all_states(
    *,
    model: Any,
    law: AllStateQExistenceLaw,
    dates: tuple[Any, ...],
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
    atm_index: int,
    branch_target_bps: float,
    raw_z_target: float,
) -> dict[str, Any]:
    """Produce per-state inference using fixed common random replications."""
    num_states = len(current_states)
    instruments = len(normalized_strikes)
    adjusted_mean = torch.empty(
        (num_states, instruments), dtype=torch.float64, device="cpu"
    )
    adjusted_se = torch.empty_like(adjusted_mean)
    raw_mean = torch.empty_like(adjusted_mean)
    raw_se = torch.empty_like(adjusted_mean)
    sample_std = torch.empty_like(adjusted_mean)
    positive_percent = torch.empty_like(adjusted_mean)
    beta_mean = torch.empty_like(adjusted_mean)
    negative_atm = torch.empty(num_states, dtype=torch.float64)
    positive_atm = torch.empty(num_states, dtype=torch.float64)
    gross_mean = torch.empty(num_states, dtype=torch.float64)
    replications = len(draws)
    paths_per_replication = draws[0].num_paths
    total_paths = replications * paths_per_replication
    for state_slice in _state_slices(num_states, state_batch_size):
        indices = torch.arange(
            state_slice.start,
            state_slice.stop,
            device=current_states.device,
        )
        with torch.no_grad():
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
        adjusted_rep = values["adjusted_replication_means"].double()
        raw_rep = values["raw_replication_means"].double()
        chunk_adjusted_mean = torch.mean(adjusted_rep, dim=0)
        chunk_raw_mean = torch.mean(raw_rep, dim=0)
        chunk_adjusted_se = torch.std(adjusted_rep, dim=0, correction=1) / math.sqrt(
            replications
        )
        chunk_raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(
            replications
        )
        path_mean = values["path_sum"] / total_paths
        path_variance = (
            values["path_square_sum"] - total_paths * path_mean.square()
        ) / (total_paths - 1)
        adjusted_mean[state_slice] = chunk_adjusted_mean.cpu()
        adjusted_se[state_slice] = chunk_adjusted_se.cpu()
        raw_mean[state_slice] = chunk_raw_mean.cpu()
        raw_se[state_slice] = chunk_raw_se.cpu()
        sample_std[state_slice] = torch.sqrt(torch.clamp_min(path_variance, 0.0)).cpu()
        positive_percent[state_slice] = (
            100.0 * values["path_positive_count"] / total_paths
        ).cpu()
        beta_mean[state_slice] = torch.mean(values["betas"].double(), dim=0).cpu()
        negative_atm[state_slice] = torch.mean(
            values["negative_atm_replication_means"].double(), dim=0
        ).cpu()
        positive_atm[state_slice] = torch.mean(
            values["positive_atm_replication_means"].double(), dim=0
        ).cpu()
        gross_mean[state_slice] = (values["gross_sum"] / total_paths).cpu()
    adjusted_z = adjusted_mean / adjusted_se.clamp_min(1.0e-12)
    raw_z = raw_mean / raw_se.clamp_min(1.0e-12)
    worst_flat = int(torch.argmax(torch.abs(raw_z)))
    worst_state = worst_flat // instruments
    worst_instrument = worst_flat % instruments
    scale = 1.0e4
    per_state: list[dict[str, Any]] = []
    for index, (state_date, state_kind) in enumerate(
        zip(dates, state_kinds, strict=True)
    ):
        state_raw_z = torch.abs(raw_z[index])
        per_state.append(
            {
                "index": index,
                "date": state_date.isoformat(),
                "history_partition": state_kind,
                "raw_mean_bps_spot": (scale * raw_mean[index]).tolist(),
                "raw_mean_mc_standard_error_bps_spot": (
                    scale * raw_se[index]
                ).tolist(),
                "raw_mean_z": raw_z[index].tolist(),
                "maximum_abs_raw_mean_z": float(torch.max(state_raw_z)),
                "adjusted_mean_bps_spot": (
                    scale * adjusted_mean[index]
                ).tolist(),
                "adjusted_mean_mc_standard_error_bps_spot": (
                    scale * adjusted_se[index]
                ).tolist(),
                "adjusted_mean_z": adjusted_z[index].tolist(),
                "return_sample_std_bps_spot": (
                    scale * sample_std[index]
                ).tolist(),
                "positive_return_percent": positive_percent[index].tolist(),
                "negative_return_percent": (
                    100.0 - positive_percent[index]
                ).tolist(),
                "negative_branch_atm_adjusted_mean_bps_spot": float(
                    scale * negative_atm[index]
                ),
                "positive_branch_atm_adjusted_mean_bps_spot": float(
                    scale * positive_atm[index]
                ),
                "spot_control_beta_mean": beta_mean[index].tolist(),
                "gross_spot_return_mean": float(gross_mean[index]),
            }
        )
    maximum_abs_raw_z = float(torch.max(torch.abs(raw_z)))
    return {
        "replications": replications,
        "paths_per_replication_per_state": paths_per_replication,
        "paths_per_state": total_paths,
        "num_states": num_states,
        "total_state_paths": num_states * total_paths,
        "common_random_numbers_across_states": True,
        "maximum_abs_raw_mean_z": maximum_abs_raw_z,
        "raw_z_target": raw_z_target,
        "passes_raw_z_target": maximum_abs_raw_z <= raw_z_target,
        "worst_raw_moment": {
            "state_index": worst_state,
            "date": dates[worst_state].isoformat(),
            "history_partition": state_kinds[worst_state],
            "instrument_index": worst_instrument,
            "raw_mean_bps_spot": float(scale * raw_mean[worst_state, worst_instrument]),
            "raw_mean_mc_standard_error_bps_spot": float(
                scale * raw_se[worst_state, worst_instrument]
            ),
            "raw_mean_z": float(raw_z[worst_state, worst_instrument]),
        },
        "absolute_raw_z_quantiles": _summary_quantiles(torch.abs(raw_z)),
        "absolute_adjusted_z_quantiles": _summary_quantiles(torch.abs(adjusted_z)),
        "branch_target_bps_spot": branch_target_bps,
        "per_state": per_state,
    }

