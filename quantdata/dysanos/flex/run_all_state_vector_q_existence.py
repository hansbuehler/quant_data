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
    *,
    beta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a fitted spot control independently by state and instrument."""
    if returns.ndim != 3 or gross_spot_return.shape != returns.shape[:2]:
        raise ValueError("statewise spot control requires aligned path tensors")
    spot_return = gross_spot_return - 1.0
    if beta is None:
        centered_spot = spot_return - torch.mean(spot_return, dim=1, keepdim=True)
        centered_returns = returns - torch.mean(returns, dim=1, keepdim=True)
        denominator = torch.sum(centered_spot.square(), dim=1).clamp_min(
            torch.finfo(returns.dtype).tiny
        )
        beta = (
            torch.sum(centered_spot[..., None] * centered_returns, dim=1)
            / denominator[:, None]
        ).detach()
    elif beta.shape != (returns.shape[0], returns.shape[2]):
        raise ValueError("statewise spot-control beta has the wrong shape")
    else:
        beta = beta.detach()
    adjusted = returns - spot_return[..., None] * beta[:, None]
    return adjusted, torch.mean(adjusted, dim=1), beta


def pooled_spot_control_beta_by_state(
    *,
    path_sum: torch.Tensor,
    spot_sum: torch.Tensor,
    spot_square_sum: torch.Tensor,
    spot_return_product_sum: torch.Tensor,
    num_paths: int,
) -> torch.Tensor:
    """Fit statewise control slopes from pooled dataset sufficient statistics."""
    if (
        path_sum.ndim != 2
        or spot_sum.shape != path_sum.shape[:1]
        or spot_square_sum.shape != spot_sum.shape
        or spot_return_product_sum.shape != path_sum.shape
        or num_paths < 2
    ):
        raise ValueError("pooled spot-control statistics are inconsistent")
    denominator = spot_square_sum - spot_sum.square() / num_paths
    denominator = denominator.clamp_min(torch.finfo(path_sum.dtype).tiny)
    numerator = spot_return_product_sum - spot_sum[:, None] * path_sum / num_paths
    return (numerator / denominator[:, None]).detach()


def control_variate_noise_scale_by_state(
    *,
    beta: torch.Tensor,
    spot_mc_standard_error: torch.Tensor,
    raw_mc_standard_error: torch.Tensor,
) -> torch.Tensor:
    """Return the raw-noise scale retained for control-variate reporting."""
    if (
        beta.ndim != 2
        or spot_mc_standard_error.shape != beta.shape[:1]
        or raw_mc_standard_error.shape != beta.shape
    ):
        raise ValueError("statewise control-variate noise scales are inconsistent")
    return torch.maximum(
        torch.abs(beta) * spot_mc_standard_error[:, None],
        raw_mc_standard_error,
    )


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
    flat_next_state = next_state.reshape(origins * paths, -1)
    raw = model.raw_states_from_states(flat_next_state)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    next_w1_calls = model.sanos_decoder.price_first_expiry_calls_from_parameters(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
        strikes=next_normalized_strikes.reshape(origins * paths, -1),
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
    parser.add_argument("--state-batch-size", type=int, default=96)
    parser.add_argument("--fit-seed", type=int, default=9_484_101)
    parser.add_argument("--validation-seed", type=int, default=9_485_101)
    parser.add_argument("--initialization-seed", type=int, default=9_483_101)
    parser.add_argument("--plateau-epochs", type=int, default=100)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-5)
    parser.add_argument("--validation-z-target", type=float, default=2.0)
    parser.add_argument("--audit-every", type=int, default=25)
    parser.add_argument("--maximum-states", type=int)
    parser.add_argument("--selected-state-index", type=int)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--objective-instrument-index", type=int)
    parser.add_argument("--optimize-raw-means", action="store_true")
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
    if arguments.selected_state_index is not None and arguments.selected_state_index < 0:
        raise ValueError("selected state index must be non-negative")
    if (
        arguments.maximum_states is not None
        and arguments.selected_state_index is not None
    ):
        raise ValueError("maximum states and selected state index are mutually exclusive")
    if arguments.objective_instrument_index is not None:
        if not 0 <= arguments.objective_instrument_index < 11:
            raise ValueError("objective instrument index must be in [0, 11)")
        if arguments.selected_state_index is None:
            raise ValueError(
                "an objective instrument index requires a selected state"
            )


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
    control_beta: torch.Tensor | None = None,
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
    spot_replication_means: list[torch.Tensor] = []
    spot_sums: list[torch.Tensor] = []
    spot_square_sums: list[torch.Tensor] = []
    spot_return_product_sums: list[torch.Tensor] = []
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
        _, adjusted_mean, beta = spot_control_variate_by_state(
            returns,
            gross,
            beta=control_beta,
        )
        negative = draw.branch < 0.0
        positive = draw.branch > 0.0
        atm_control_beta = (
            None
            if control_beta is None
            else control_beta[:, atm_index : atm_index + 1]
        )
        _, negative_atm, _ = spot_control_variate_by_state(
            returns[:, negative, atm_index : atm_index + 1],
            gross[:, negative],
            beta=atm_control_beta,
        )
        _, positive_atm, _ = spot_control_variate_by_state(
            returns[:, positive, atm_index : atm_index + 1],
            gross[:, positive],
            beta=atm_control_beta,
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
        spot_return = gross.double() - 1.0
        path_sums.append(torch.sum(values, dim=1))
        path_square_sums.append(torch.sum(values.square(), dim=1))
        path_positive_counts.append(torch.sum(values > 0.0, dim=1))
        gross_sums.append(torch.sum(gross.double(), dim=1))
        spot_replication_means.append(torch.mean(spot_return, dim=1))
        spot_sums.append(torch.sum(spot_return, dim=1))
        spot_square_sums.append(torch.sum(spot_return.square(), dim=1))
        spot_return_product_sums.append(
            torch.sum(spot_return[..., None] * values, dim=1)
        )
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
        "spot_replication_means": torch.stack(spot_replication_means),
        "spot_sum": torch.sum(torch.stack(spot_sums), dim=0),
        "spot_square_sum": torch.sum(torch.stack(spot_square_sums), dim=0),
        "spot_return_product_sum": torch.sum(
            torch.stack(spot_return_product_sums), dim=0
        ),
    }


def _summary_quantiles(values: torch.Tensor) -> dict[str, float]:
    finite = values.detach().double().reshape(-1).cpu()
    worst_decile_count = max(1, math.ceil(0.10 * len(finite)))
    return {
        "minimum": float(torch.min(finite)),
        "p50": float(torch.quantile(finite, 0.50)),
        "p90": float(torch.quantile(finite, 0.90)),
        "p95": float(torch.quantile(finite, 0.95)),
        "p99": float(torch.quantile(finite, 0.99)),
        "worst_10_percent_mean": float(
            torch.mean(torch.topk(finite, worst_decile_count).values)
        ),
        "maximum": float(torch.max(finite)),
    }


def _audit_all_states(
    *,
    model: Any,
    law: AllStateQExistenceLaw,
    dates: tuple[Any, ...],
    state_kinds: tuple[str, ...],
    history_indices: tuple[int, ...],
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
    """Audit with statewise betas and means fitted on the validation paths."""
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
    validation_beta = torch.empty_like(adjusted_mean)
    spot_se = torch.empty(num_states, dtype=torch.float64)
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
        raw_rep = values["raw_replication_means"].double()
        chunk_validation_beta = pooled_spot_control_beta_by_state(
            path_sum=values["path_sum"],
            spot_sum=values["spot_sum"],
            spot_square_sum=values["spot_square_sum"],
            spot_return_product_sum=values["spot_return_product_sum"],
            num_paths=total_paths,
        )
        spot_rep = values["spot_replication_means"].double()
        adjusted_rep = (
            raw_rep - spot_rep[..., None] * chunk_validation_beta[None]
        )
        chunk_adjusted_mean = torch.mean(adjusted_rep, dim=0)
        chunk_raw_mean = torch.mean(raw_rep, dim=0)
        chunk_adjusted_se = torch.std(adjusted_rep, dim=0, correction=1) / math.sqrt(
            replications
        )
        chunk_raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(
            replications
        )
        chunk_spot_se = torch.std(spot_rep, dim=0, correction=1) / math.sqrt(
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
        validation_beta[state_slice] = chunk_validation_beta.cpu()
        spot_se[state_slice] = chunk_spot_se.cpu()
        negative_atm[state_slice] = torch.mean(
            values["negative_atm_replication_means"].double(), dim=0
        ).cpu()
        positive_atm[state_slice] = torch.mean(
            values["positive_atm_replication_means"].double(), dim=0
        ).cpu()
        gross_mean[state_slice] = (values["gross_sum"] / total_paths).cpu()
    adjusted_z = adjusted_mean / adjusted_se.clamp_min(1.0e-12)
    raw_z = raw_mean / raw_se.clamp_min(1.0e-12)
    noise_scale = control_variate_noise_scale_by_state(
        beta=validation_beta,
        spot_mc_standard_error=spot_se,
        raw_mc_standard_error=raw_se,
    )
    noise_scaled_adjusted_z = adjusted_mean / noise_scale.clamp_min(1.0e-12)
    worst_flat = int(torch.argmax(torch.abs(raw_z)))
    worst_state = worst_flat // instruments
    worst_instrument = worst_flat % instruments
    scale = 1.0e4
    per_state: list[dict[str, Any]] = []
    for index, (state_date, state_kind) in enumerate(
        zip(dates, state_kinds, strict=True)
    ):
        state_raw_z = torch.abs(raw_z[index])
        state_adjusted_z = torch.abs(adjusted_z[index])
        state_noise_scaled_z = torch.abs(noise_scaled_adjusted_z[index])
        per_state.append(
            {
                "index": history_indices[index],
                "local_index": index,
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
                "maximum_abs_adjusted_mean_z": float(
                    torch.max(state_adjusted_z)
                ),
                "control_variate_noise_scale_bps_spot": (
                    scale * noise_scale[index]
                ).tolist(),
                "noise_scaled_adjusted_mean_z": noise_scaled_adjusted_z[
                    index
                ].tolist(),
                "maximum_abs_noise_scaled_adjusted_mean_z": float(
                    torch.max(state_noise_scaled_z)
                ),
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
                "spot_control_beta_validation": validation_beta[index].tolist(),
                "validation_spot_return_mean_bps": float(
                    scale * (gross_mean[index] - 1.0)
                ),
                "validation_spot_return_mc_standard_error_bps": float(
                    scale * spot_se[index]
                ),
                "gross_spot_return_mean": float(gross_mean[index]),
            }
        )
    maximum_abs_raw_z = float(torch.max(torch.abs(raw_z)))
    maximum_abs_adjusted_z = float(torch.max(torch.abs(adjusted_z)))
    worst_adjusted_flat = int(torch.argmax(torch.abs(adjusted_z)))
    worst_adjusted_state = worst_adjusted_flat // instruments
    worst_adjusted_instrument = worst_adjusted_flat % instruments
    maximum_abs_noise_scaled_adjusted_z = float(
        torch.max(torch.abs(noise_scaled_adjusted_z))
    )
    worst_noise_scaled_flat = int(torch.argmax(torch.abs(noise_scaled_adjusted_z)))
    worst_noise_scaled_state = worst_noise_scaled_flat // instruments
    worst_noise_scaled_instrument = worst_noise_scaled_flat % instruments
    return {
        "replications": replications,
        "paths_per_replication_per_state": paths_per_replication,
        "paths_per_state": total_paths,
        "num_states": num_states,
        "total_state_paths": num_states * total_paths,
        "common_random_numbers_across_states": True,
        "control_variate": {
            "beta_source": "pooled_validation_paths_by_state",
            "beta_and_spot_mean_scope": (
                "estimated separately per dataset and historical state"
            ),
            "validation_correction": (
                "raw_validation_mean - validation_beta * validation_spot_mean"
            ),
            "validation_beta_refit": True,
            "stopping_noise_scale": (
                "max(abs(beta) * spot_mc_standard_error, "
                "raw_option_mc_standard_error)"
            ),
        },
        "maximum_abs_raw_mean_z": maximum_abs_raw_z,
        "maximum_abs_adjusted_mean_z": maximum_abs_adjusted_z,
        "maximum_abs_noise_scaled_adjusted_mean_z": (
            maximum_abs_noise_scaled_adjusted_z
        ),
        "raw_z_target": raw_z_target,
        "adjusted_z_target": raw_z_target,
        "passes_raw_z_target": maximum_abs_raw_z <= raw_z_target,
        "passes_adjusted_z_target": maximum_abs_adjusted_z <= raw_z_target,
        "passes_noise_scaled_adjusted_z_target": (
            maximum_abs_noise_scaled_adjusted_z <= raw_z_target
        ),
        "worst_raw_moment": {
            "state_index": history_indices[worst_state],
            "local_state_index": worst_state,
            "date": dates[worst_state].isoformat(),
            "history_partition": state_kinds[worst_state],
            "instrument_index": worst_instrument,
            "raw_mean_bps_spot": float(scale * raw_mean[worst_state, worst_instrument]),
            "raw_mean_mc_standard_error_bps_spot": float(
                scale * raw_se[worst_state, worst_instrument]
            ),
            "raw_mean_z": float(raw_z[worst_state, worst_instrument]),
        },
        "worst_adjusted_moment": {
            "state_index": history_indices[worst_adjusted_state],
            "local_state_index": worst_adjusted_state,
            "date": dates[worst_adjusted_state].isoformat(),
            "history_partition": state_kinds[worst_adjusted_state],
            "instrument_index": worst_adjusted_instrument,
            "adjusted_mean_bps_spot": float(
                scale
                * adjusted_mean[worst_adjusted_state, worst_adjusted_instrument]
            ),
            "adjusted_mean_mc_standard_error_bps_spot": float(
                scale * adjusted_se[worst_adjusted_state, worst_adjusted_instrument]
            ),
            "adjusted_mean_z": float(
                adjusted_z[worst_adjusted_state, worst_adjusted_instrument]
            ),
            "validation_beta": float(
                validation_beta[worst_adjusted_state, worst_adjusted_instrument]
            ),
            "validation_spot_return_mean_bps": float(
                scale * (gross_mean[worst_adjusted_state] - 1.0)
            ),
        },
        "worst_noise_scaled_adjusted_moment": {
            "state_index": history_indices[worst_noise_scaled_state],
            "local_state_index": worst_noise_scaled_state,
            "date": dates[worst_noise_scaled_state].isoformat(),
            "history_partition": state_kinds[worst_noise_scaled_state],
            "instrument_index": worst_noise_scaled_instrument,
            "adjusted_mean_bps_spot": float(
                scale
                * adjusted_mean[
                    worst_noise_scaled_state, worst_noise_scaled_instrument
                ]
            ),
            "control_variate_noise_scale_bps_spot": float(
                scale
                * noise_scale[
                    worst_noise_scaled_state, worst_noise_scaled_instrument
                ]
            ),
            "noise_scaled_adjusted_mean_z": float(
                noise_scaled_adjusted_z[
                    worst_noise_scaled_state, worst_noise_scaled_instrument
                ]
            ),
            "raw_mean_mc_standard_error_bps_spot": float(
                scale
                * raw_se[worst_noise_scaled_state, worst_noise_scaled_instrument]
            ),
            "abs_beta_times_spot_mc_standard_error_bps": float(
                scale
                * torch.abs(
                    validation_beta[
                        worst_noise_scaled_state, worst_noise_scaled_instrument
                    ]
                )
                * spot_se[worst_noise_scaled_state]
            ),
        },
        "absolute_raw_z_quantiles": _summary_quantiles(torch.abs(raw_z)),
        "absolute_adjusted_z_quantiles": _summary_quantiles(torch.abs(adjusted_z)),
        "absolute_noise_scaled_adjusted_z_quantiles": _summary_quantiles(
            torch.abs(noise_scaled_adjusted_z)
        ),
        "branch_target_bps_spot": branch_target_bps,
        "per_state": per_state,
    }


def _benchmark(
    *,
    arguments: argparse.Namespace,
    model: Any,
    law: AllStateQExistenceLaw,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draw: CommonQDraw,
) -> None:
    """Measure one representative forward/backward microbatch."""
    origins = min(arguments.state_batch_size, len(current_states))
    state_slice = slice(0, origins)
    origin_indices = torch.arange(origins, device=current_states.device)

    def iteration() -> float:
        law.zero_grad(set_to_none=True)
        returns, raw_next, _ = state_batch_roll_returns(
            model=model,
            law=law,
            origin_indices=origin_indices,
            current_states=current_states[state_slice],
            current_w2_calls=current_w2_calls[state_slice],
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support[state_slice],
            density=density[state_slice],
            sqrt_variance=sqrt_variance[state_slice],
            draw=draw,
        )
        loss = returns.square().mean() + 1.0e-4 * torch.mean(
            dlvsq_cap_barrier(raw_next, geometry=model.config.geometry).square()
        )
        loss.backward()
        return float(loss.detach())

    iteration()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = iteration()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    payload = {
        "schema": "flex-all-state-vector-q-benchmark-v1",
        "device": torch.cuda.get_device_name(0),
        "state_batch_size": origins,
        "paths_per_state": draw.num_paths,
        "state_paths": origins * draw.num_paths,
        "elapsed_seconds": elapsed,
        "state_paths_per_second": origins * draw.num_paths / elapsed,
        "peak_allocated_gib": allocated / 1024**3,
        "peak_reserved_gib": reserved / 1024**3,
        "loss": loss,
    }
    write_json_atomic(payload, arguments.output)
    print(json.dumps(payload, indent=2), flush=True)


def run(arguments: argparse.Namespace) -> None:
    """Fit and validate an independent conditional measure at every state."""
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
    dates = train_dates + validation_dates
    state_kinds = ("model_fit",) * len(train_dates) + ("held_out_2025",) * len(
        validation_dates
    )
    current_states = torch.cat((train_states, validation_states), dim=0)
    history_indices = tuple(range(len(current_states)))
    if arguments.selected_state_index is not None:
        selected = arguments.selected_state_index
        if selected >= len(current_states):
            raise ValueError("selected state index is outside the historical state set")
        current_states = current_states[selected : selected + 1]
        dates = dates[selected : selected + 1]
        state_kinds = state_kinds[selected : selected + 1]
        history_indices = (selected,)
    elif arguments.maximum_states is not None:
        current_states = current_states[: arguments.maximum_states]
        dates = dates[: arguments.maximum_states]
        state_kinds = state_kinds[: arguments.maximum_states]
        history_indices = history_indices[: arguments.maximum_states]
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    num_states = len(current_states)
    front_state_size = model.config.state_sizes_by_expiry[0]
    config = AtmQExistenceLawConfig(
        label="all-state-vector-q-existence-exact-v1",
        front_state_size=front_state_size,
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
                num_states, len(normalized_strikes)
            ),
            strikes=normalized_strikes[None].expand(num_states, -1),
        ).detach()
        support, density, sqrt_variance = normalize_front_components(
            current_surface.support_strikes[:, 0],
            current_surface.densities[:, 0],
            current_surface.backbone_total_vols[:, 0],
        )
    law = AllStateQExistenceLaw(
        env,
        current_front_states=current_states[:, :front_state_size],
        config=config,
        initialization_seed=arguments.initialization_seed,
    )
    teacher_initialization: dict[str, Any] | None = None
    if arguments.teacher_checkpoint is not None and not arguments.checkpoint.exists():
        teacher_payload = torch.load(
            arguments.teacher_checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(teacher_payload, Mapping)
            or teacher_payload.get("schema")
            != "flex-all-state-vector-q-training-v1"
            or teacher_payload.get("source_completion_sha256")
            != sources["completion_sha256"]
            or teacher_payload.get("law_config") != asdict(config)
        ):
            raise ValueError("teacher checkpoint is incompatible with the source model")
        teacher_state = cast(
            Mapping[str, torch.Tensor], teacher_payload["law_state_dict"]
        )
        teacher_indices = torch.tensor(history_indices, device=env.device)
        teacher_front = teacher_state["current_front_states"][teacher_indices]
        if not torch.equal(teacher_front, current_states[:, :front_state_size]):
            raise ValueError("teacher and requested historical front states differ")
        with torch.no_grad():
            law.location_shift.copy_(teacher_state["location_shift"][teacher_indices])
            law.spot_loading.copy_(teacher_state["spot_loading"][teacher_indices])
            law.branch_direction.copy_(
                teacher_state["branch_direction"][teacher_indices]
            )
            law.raw_noise_std.copy_(
                teacher_state["raw_noise_std"][teacher_indices]
            )
        teacher_initialization = {
            "checkpoint": str(arguments.teacher_checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.teacher_checkpoint),
            "history_state_indices": list(history_indices),
            "num_states": num_states,
            "first_history_state_date": dates[0].isoformat(),
            "last_history_state_date": dates[-1].isoformat(),
            "source_epoch": len(cast(list[Any], teacher_payload["history"])),
        }
    fit_draws = _common_draws(
        arguments,
        seed=arguments.fit_seed,
        front_state_size=front_state_size,
        env=env,
        replications=arguments.fit_replications,
    )
    validation_draws = _common_draws(
        arguments,
        seed=arguments.validation_seed,
        front_state_size=front_state_size,
        env=env,
        replications=arguments.validation_replications,
    )
    if arguments.benchmark_only:
        _benchmark(
            arguments=arguments,
            model=model,
            law=law,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draw=fit_draws[0],
        )
        return

    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=5.0e-5,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_loss = math.inf
    epochs_without_material_improvement = 0
    checkpoint: dict[str, Any] | None = None
    simulation_config = {
        "fit_replications": arguments.fit_replications,
        "validation_replications": arguments.validation_replications,
        "spot_contexts": arguments.spot_contexts,
        "surface_paths": arguments.surface_paths,
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
        "initialization_seed": arguments.initialization_seed,
        "num_states": num_states,
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "history_indices": list(history_indices),
        "moment_estimator": (
            "raw_return_mean"
            if arguments.optimize_raw_means
            else "statewise_pooled_spot_control_variate_adjusted_mean"
        ),
        "control_beta_scope": "separate_per_dataset_and_historical_state",
    }
    if arguments.objective_instrument_index is not None:
        simulation_config["objective_instrument_index"] = (
            arguments.objective_instrument_index
        )
    if arguments.checkpoint.exists():
        checkpoint_payload = torch.load(
            arguments.checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(checkpoint_payload, Mapping)
            or checkpoint_payload.get("schema")
            != "flex-all-state-vector-q-training-v1"
            or checkpoint_payload.get("source_completion_sha256")
            != sources["completion_sha256"]
            or checkpoint_payload.get("law_config") != asdict(config)
            or checkpoint_payload.get("simulation_config") != simulation_config
        ):
            raise ValueError("all-state vector-Q checkpoint identity is incompatible")
        law.load_state_dict(
            cast(Mapping[str, torch.Tensor], checkpoint_payload["law_state_dict"])
        )
        optimizer.load_state_dict(
            cast(dict[str, Any], checkpoint_payload["optimizer_state_dict"])
        )
        scheduler.load_state_dict(
            cast(dict[str, Any], checkpoint_payload["scheduler_state_dict"])
        )
        history = list(cast(list[dict[str, float]], checkpoint_payload["history"]))
        start_epoch = len(history)
        best_loss = float(checkpoint_payload.get("best_loss", math.inf))
        epochs_without_material_improvement = int(
            checkpoint_payload.get("epochs_without_material_improvement", 0)
        )
        saved_teacher_initialization = checkpoint_payload.get(
            "teacher_initialization"
        )
        if isinstance(saved_teacher_initialization, Mapping):
            teacher_initialization = dict(saved_teacher_initialization)
        checkpoint = dict(checkpoint_payload)
        print(
            f"resuming all-state vector Q at epoch {start_epoch} "
            f"with best loss {best_loss:.6g}",
            flush=True,
        )
    if teacher_initialization is None and arguments.teacher_checkpoint is not None:
        teacher_payload = torch.load(
            arguments.teacher_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(teacher_payload, Mapping):
            raise ValueError("teacher checkpoint provenance is invalid")
        teacher_initialization = {
            "checkpoint": str(arguments.teacher_checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.teacher_checkpoint),
            "history_state_indices": list(history_indices),
            "num_states": num_states,
            "first_history_state_date": dates[0].isoformat(),
            "last_history_state_date": dates[-1].isoformat(),
            "source_epoch": len(cast(list[Any], teacher_payload["history"])),
        }

    validation_audit: dict[str, Any] | None = None
    stopped_for_plateau = False
    stopped_for_validation = False
    target = config.branch_atm_target_bps
    state_slices = _state_slices(num_states, arguments.state_batch_size)
    fit_paths_per_state = sum(draw.num_paths for draw in fit_draws)
    for epoch in range(start_epoch, arguments.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        maximum_adjusted_mean = 0.0
        maximum_objective_mean = 0.0
        maximum_raw_z = 0.0
        maximum_noise_scaled_adjusted_z = 0.0
        absolute_raw_z_chunks: list[torch.Tensor] = []
        absolute_noise_scaled_adjusted_z_chunks: list[torch.Tensor] = []
        for state_slice in state_slices:
            indices = torch.arange(
                state_slice.start,
                state_slice.stop,
                device=env.device,
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
                    draws=fit_draws,
                    atm_index=atm_index,
                )
                raw_rep = values["raw_replication_means"]
                fit_beta = pooled_spot_control_beta_by_state(
                    path_sum=values["path_sum"],
                    spot_sum=values["spot_sum"],
                    spot_square_sum=values["spot_square_sum"],
                    spot_return_product_sum=values["spot_return_product_sum"],
                    num_paths=fit_paths_per_state,
                )
                adjusted_rep = (
                    raw_rep
                    - values["spot_replication_means"][..., None] * fit_beta[None]
                )
                objective_rep = raw_rep if arguments.optimize_raw_means else adjusted_rep
                pooled = 1.0e4 * torch.mean(objective_rep, dim=0)
                adjusted_pooled = 1.0e4 * torch.mean(adjusted_rep, dim=0)
                negative_atm = 1.0e4 * torch.mean(
                    values["negative_atm_replication_means"], dim=0
                )
                positive_atm = 1.0e4 * torch.mean(
                    values["positive_atm_replication_means"], dim=0
                )
                raw_mean = torch.mean(raw_rep, dim=0)
                raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(
                    len(fit_draws)
                )
                raw_z = raw_mean / raw_se.clamp_min(1.0e-12)
                spot_se = torch.std(
                    values["spot_replication_means"], dim=0, correction=1
                ) / math.sqrt(len(fit_draws))
                fit_noise_scale = control_variate_noise_scale_by_state(
                    beta=fit_beta,
                    spot_mc_standard_error=spot_se,
                    raw_mc_standard_error=raw_se,
                )
                noise_scaled_adjusted_z = (
                    torch.mean(adjusted_rep, dim=0)
                    / fit_noise_scale.clamp_min(1.0e-12)
                )
                moment_loss = (
                    torch.mean(pooled.square(), dim=1)
                    if arguments.objective_instrument_index is None
                    else pooled[:, arguments.objective_instrument_index].square()
                )
                state_loss = (
                    moment_loss
                    + 0.01
                    * ((negative_atm + target).square() + (positive_atm - target).square())
                    + torch.mean(values["cap_values"], dim=0)
                )
            loss_sum += float(torch.sum(state_loss))
            maximum_adjusted_mean = max(
                maximum_adjusted_mean,
                float(torch.max(torch.abs(adjusted_pooled))),
            )
            maximum_objective_mean = max(
                maximum_objective_mean,
                float(torch.max(torch.abs(pooled))),
            )
            maximum_raw_z = max(maximum_raw_z, float(torch.max(torch.abs(raw_z))))
            maximum_noise_scaled_adjusted_z = max(
                maximum_noise_scaled_adjusted_z,
                float(torch.max(torch.abs(noise_scaled_adjusted_z))),
            )
            absolute_raw_z_chunks.append(torch.abs(raw_z).detach().reshape(-1).cpu())
            absolute_noise_scaled_adjusted_z_chunks.append(
                torch.abs(noise_scaled_adjusted_z).detach().reshape(-1).cpu()
            )
            origins = len(indices)
            for draw in fit_draws:
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
                _, adjusted_mean, _ = spot_control_variate_by_state(
                    returns,
                    gross,
                    beta=fit_beta,
                )
                objective_mean = (
                    torch.mean(returns, dim=1)
                    if arguments.optimize_raw_means
                    else adjusted_mean
                )
                negative = draw.branch < 0.0
                positive = draw.branch > 0.0
                _, negative_mean, _ = spot_control_variate_by_state(
                    returns[:, negative, atm_index : atm_index + 1],
                    gross[:, negative],
                )
                _, positive_mean, _ = spot_control_variate_by_state(
                    returns[:, positive, atm_index : atm_index + 1],
                    gross[:, positive],
                )
                barrier = dlvsq_cap_barrier(
                    raw_next, geometry=model.config.geometry
                )
                replication_scale = 1.0 / (num_states * len(fit_draws))
                moment_surrogate = (
                    torch.sum(pooled * (1.0e4 * objective_mean))
                    / len(normalized_strikes)
                    if arguments.objective_instrument_index is None
                    else torch.sum(
                        pooled[:, arguments.objective_instrument_index]
                        * (
                            1.0e4
                            * objective_mean[
                                :, arguments.objective_instrument_index
                            ]
                        )
                    )
                )
                surrogate = (
                    2.0
                    * replication_scale
                    * moment_surrogate
                    + 0.02
                    * replication_scale
                    * torch.sum((negative_atm + target) * (1.0e4 * negative_mean[:, 0]))
                    + 0.02
                    * replication_scale
                    * torch.sum((positive_atm - target) * (1.0e4 * positive_mean[:, 0]))
                    + 1.0e-4
                    * (origins / num_states)
                    * torch.mean(barrier.square())
                    / len(fit_draws)
                )
                surrogate.backward()
        loss_value = loss_sum / num_states
        absolute_raw_z = torch.cat(absolute_raw_z_chunks)
        worst_decile_count = max(1, math.ceil(0.10 * len(absolute_raw_z)))
        worst_decile_mean_raw_z = float(
            torch.mean(torch.topk(absolute_raw_z, worst_decile_count).values)
        )
        absolute_noise_scaled_adjusted_z = torch.cat(
            absolute_noise_scaled_adjusted_z_chunks
        )
        worst_decile_mean_noise_scaled_adjusted_z = float(
            torch.mean(
                torch.topk(
                    absolute_noise_scaled_adjusted_z,
                    worst_decile_count,
                ).values
            )
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        scheduler.step(loss_value)
        if loss_value < best_loss * (1.0 - arguments.minimum_relative_improvement):
            best_loss = loss_value
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1
        record = {
            "epoch": float(epoch + 1),
            "loss": loss_value,
            "maximum_abs_adjusted_mean_bps_spot": maximum_adjusted_mean,
            "maximum_abs_objective_mean_bps_spot": maximum_objective_mean,
            "maximum_abs_raw_mean_z": maximum_raw_z,
            "worst_10_percent_mean_abs_raw_mean_z": worst_decile_mean_raw_z,
            "maximum_abs_noise_scaled_adjusted_mean_z": (
                maximum_noise_scaled_adjusted_z
            ),
            "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": (
                worst_decile_mean_noise_scaled_adjusted_z
            ),
            "minimum_state_noise_std": float(torch.min(law.noise_std.detach())),
            "maximum_state_noise_std": float(torch.max(law.noise_std.detach())),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise FloatingPointError("non-finite all-state vector-Q training record")
        history.append(record)
        checkpoint = {
            "schema": "flex-all-state-vector-q-training-v1",
            "status": "training",
            "objective": OBJECTIVE_ID,
            "source_completion_sha256": sources["completion_sha256"],
            "law_config": asdict(config),
            "simulation_config": simulation_config,
            "law_state_dict": law.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_loss": best_loss,
            "epochs_without_material_improvement": epochs_without_material_improvement,
            "teacher_initialization": teacher_initialization,
            "validation_stopping_statistic": (
                "abs(adjusted_mean) / max(abs(beta) * spot_mc_standard_error, "
                "raw_option_mc_standard_error)"
            ),
        }
        _atomic_torch_save(checkpoint, arguments.checkpoint)
        print(
            f"epoch={epoch + 1} loss={loss_value:.6g} "
            f"fit_max_objective_mean={maximum_objective_mean:.4f}bp "
            f"fit_max_adjusted_mean={maximum_adjusted_mean:.4f}bp "
            f"fit_max_raw_z={maximum_raw_z:.2f} "
            f"fit_worst10pct_mean_raw_z={worst_decile_mean_raw_z:.2f} "
            f"fit_max_noise_scaled_z={maximum_noise_scaled_adjusted_z:.2f} "
            "fit_worst10pct_mean_noise_scaled_z="
            f"{worst_decile_mean_noise_scaled_adjusted_z:.2f} "
            f"noise=({record['minimum_state_noise_std']:.5f},"
            f"{record['maximum_state_noise_std']:.5f}) "
            f"lr={record['learning_rate']:.6g}",
            flush=True,
        )
        should_audit = (epoch + 1) % arguments.audit_every == 0
        if should_audit:
            validation_audit = _audit_all_states(
                model=model,
                law=law,
                dates=dates,
                state_kinds=state_kinds,
                history_indices=history_indices,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=validation_draws,
                state_batch_size=arguments.state_batch_size,
                atm_index=atm_index,
                branch_target_bps=target,
                raw_z_target=arguments.validation_z_target,
            )
            print(
                "validation "
                f"max_adjusted_z="
                f"{validation_audit['maximum_abs_adjusted_mean_z']:.3f} "
                f"max_noise_scaled_z="
                f"{validation_audit['maximum_abs_noise_scaled_adjusted_mean_z']:.3f} "
                f"max_raw_z={validation_audit['maximum_abs_raw_mean_z']:.3f} "
                "worst10pct_mean_noise_scaled_z="
                f"{validation_audit['absolute_noise_scaled_adjusted_z_quantiles']['worst_10_percent_mean']:.3f} "
                "worst10pct_mean_raw_z="
                f"{validation_audit['absolute_raw_z_quantiles']['worst_10_percent_mean']:.3f} "
                f"target={arguments.validation_z_target:.3f}",
                flush=True,
            )
            if bool(
                validation_audit["passes_noise_scaled_adjusted_z_target"]
            ):
                stopped_for_validation = True
                break
        if epochs_without_material_improvement >= arguments.plateau_epochs:
            stopped_for_plateau = True
            break

    if checkpoint is None:
        raise RuntimeError("all-state vector-Q run produced no checkpoint state")
    if validation_audit is None or not stopped_for_validation:
        validation_audit = _audit_all_states(
            model=model,
            law=law,
            dates=dates,
            state_kinds=state_kinds,
            history_indices=history_indices,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draws=validation_draws,
            state_batch_size=arguments.state_batch_size,
            atm_index=atm_index,
            branch_target_bps=target,
            raw_z_target=arguments.validation_z_target,
        )
    semantic_success = bool(
        validation_audit["passes_noise_scaled_adjusted_z_target"]
    )
    report = {
        "schema": "flex-all-state-vector-q-report-v1",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "objective": OBJECTIVE_ID,
        "num_states": num_states,
        "num_model_fit_states": sum(kind == "model_fit" for kind in state_kinds),
        "num_held_out_2025_states": sum(
            kind == "held_out_2025" for kind in state_kinds
        ),
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "history_indices": list(history_indices),
        "teacher_initialization": teacher_initialization,
        "instruments": [
            {
                "index": index,
                "type": "call" if bool(option_is_call[index]) else "put",
                "log_strike_over_spot": float(torch.log(strike)),
                "strike_over_spot": float(strike),
            }
            for index, strike in enumerate(normalized_strikes)
        ],
        "simulation_config": simulation_config,
        "training": {
            "epochs_completed": len(history),
            "best_loss": best_loss,
            "stopped_for_validation_target": stopped_for_validation,
            "stopped_for_plateau": stopped_for_plateau,
            "plateau_epochs": arguments.plateau_epochs,
            "minimum_relative_improvement": arguments.minimum_relative_improvement,
            "state_batch_size": arguments.state_batch_size,
            "history": history,
        },
        "independent_validation": validation_audit,
        "teacher_law": {
            "form": "h1_next=h1+a+B*z_spot+b*d+sigma*epsilon",
            "dates": [value.isoformat() for value in dates],
            "history_partition": list(state_kinds),
            "location_shift": law.location_shift.detach().cpu().tolist(),
            "spot_loading": law.spot_loading.detach().cpu().tolist(),
            "branch_direction": law.branch_direction.detach().cpu().tolist(),
            "state_noise_std": law.noise_std.detach().cpu().tolist(),
        },
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
            "schema": "flex-all-state-vector-q-completion-v1",
            "status": "complete" if semantic_success else "FAILED_TERMINAL",
            "semantic_success": semantic_success,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
            "source_completion": str(arguments.joint_completion),
            "source_completion_sha256": sources["completion_sha256"],
        },
        completion,
    )
    print(
        json.dumps(
            validation_audit["worst_noise_scaled_adjusted_moment"],
            indent=2,
        ),
        flush=True,
    )
    if not semantic_success:
        raise RuntimeError(
            "all-state vector-Q law failed noise-scaled validation z target"
        )


def main() -> None:
    arguments = _parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(main, program="run_all_state_vector_q_existence", clean_sigterm=True)
