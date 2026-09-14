"""Train direct-utility policies at fixed notionals on one physical DLV tower."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.direct_utility_frontier import (
    DirectUtilityTowerPolicy,
    log_powerexp_marginal,
    normalized_powerexp_marginal,
    powerexp_utility,
)
from quantdata.dysanos.flex.fixed_expiry_ar1 import (
    fit_fixed_expiry_ar1,
    torch_fixed_expiry_raw_log_loadings,
)
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2_triangular import (
    TriangularFixedGridState,
    triangular_fixed_grid_physical_roll,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_fixed_grid_dlv_shrinking_baseline import (
    _randomized_stratified_antithetic_uniforms,
)
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True)
class PhysicalTowerDataset:
    """State features and one-step same-claim returns on common physical paths."""

    features_by_time: tuple[torch.Tensor, ...]
    returns_by_time: tuple[torch.Tensor, ...]
    spot_returns: torch.Tensor
    paths_per_independent_block: int
    diagnostics: dict[str, float | int]

    @property
    def num_paths(self) -> int:
        return len(self.features_by_time[0])


@dataclass(frozen=True)
class BetaWhiteningTransform:
    """Detached physical beta and full-rank residual whitening by trade week."""

    beta_by_time: tuple[torch.Tensor, ...]
    spot_scale_by_time: tuple[torch.Tensor, ...]
    eigenvectors_by_time: tuple[torch.Tensor, ...]
    inverse_sqrt_eigenvalues_by_time: tuple[torch.Tensor, ...]
    noise_seed: int
    diagnostics: tuple[dict[str, float | int], ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-last-notional-expansion", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=44)
    parser.add_argument("--notionals", default="2,4,8,16,32,64,128,256")
    parser.add_argument("--training-paths", type=int, default=100_000)
    parser.add_argument(
        "--validation-paths",
        type=int,
        default=0,
        help="optional separate audit paths; zero audits the fitted path set",
    )
    parser.add_argument("--path-generation-batch-size", type=int, default=10_000)
    parser.add_argument("--training-batch-size", type=int, default=10_000)
    parser.add_argument("--antithetic-pairs-per-block", type=int, default=50)
    parser.add_argument("--innovation-rank", type=int, default=25)
    parser.add_argument("--hidden-width", type=int, default=1_024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--action-penalty", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=100.0)
    parser.add_argument("--bias-natural-gradient-interval", type=int, default=0)
    parser.add_argument("--bias-natural-gradient-notionals", default="16,32,64")
    parser.add_argument("--bias-natural-gradient-ridge", type=float, default=1.0e-3)
    parser.add_argument("--monitoring-interval", type=int, default=5)
    parser.add_argument("--monitoring-bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--final-bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--residual-adversary-epochs", type=int, default=0)
    parser.add_argument("--residual-adversary-notionals", default="4096")
    parser.add_argument("--residual-base-weights-checkpoint", type=Path)
    parser.add_argument("--residual-base-uniform", action="store_true")
    parser.add_argument("--residual-checkpoint", type=Path)
    parser.add_argument("--residual-adversary-learning-rate", type=float, default=1.0e-6)
    parser.add_argument(
        "--residual-optimizer",
        choices=("normalized-adam", "direct-lbfgs"),
        default="normalized-adam",
    )
    parser.add_argument("--residual-lbfgs-iterations-per-epoch", type=int, default=5)
    parser.add_argument("--residual-adversary-monitoring-interval", type=int, default=10)
    parser.add_argument(
        "--residual-hidden-width",
        type=int,
        default=0,
        help="zero reuses --hidden-width",
    )
    parser.add_argument(
        "--residual-hidden-layers",
        type=int,
        default=0,
        help="zero reuses --hidden-layers",
    )
    parser.add_argument(
        "--residual-fixed-hidden-features",
        action="store_true",
        help="freeze residual-policy hidden layers and train only its linear output maps",
    )
    parser.add_argument("--residual-bias-natural-gradient-interval", type=int, default=0)
    parser.add_argument("--residual-bias-natural-gradient-ridge", type=float, default=1.0e-2)
    parser.add_argument("--residual-final-bias-root-steps", type=int, default=0)
    parser.add_argument("--residual-final-bias-root-ridge", type=float, default=1.0e-4)
    parser.add_argument(
        "--residual-factor-maximum-condition-number",
        type=float,
        default=1_000.0,
    )
    parser.add_argument("--residual-factor-noise-seed", type=int, default=2_026_092_5)
    parser.add_argument("--residual-adversary-seed", type=int, default=2_026_092_4)
    parser.add_argument("--training-seed", type=int, default=2_026_092_1)
    parser.add_argument("--validation-seed", type=int, default=2_026_092_2)
    parser.add_argument("--model-seed", type=int, default=2_026_092_3)
    return parser


def _notionals(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError("notionals must be comma-separated finite-positive values")
    if any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ValueError("notionals must be strictly increasing")
    return result


def _validate_arguments(arguments: argparse.Namespace) -> tuple[float, ...]:
    notionals = _notionals(arguments.notionals)
    paths_per_block = 2 * arguments.antithetic_pairs_per_block
    if arguments.training_paths < paths_per_block or arguments.training_paths % paths_per_block:
        raise ValueError("training_paths must contain complete independent blocks")
    if arguments.validation_paths and (
        arguments.validation_paths < paths_per_block or arguments.validation_paths % paths_per_block
    ):
        raise ValueError("nonzero validation_paths must contain complete independent blocks")
    if (
        arguments.path_generation_batch_size <= 0
        or arguments.path_generation_batch_size % paths_per_block
    ):
        raise ValueError("path-generation-batch-size must contain complete blocks")
    if arguments.training_paths % arguments.path_generation_batch_size or (
        arguments.validation_paths
        and arguments.validation_paths % arguments.path_generation_batch_size
    ):
        raise ValueError("path counts must be divisible by generation batch size")
    if arguments.training_batch_size <= 0:
        raise ValueError("training-batch-size must be positive")
    if (
        min(
            arguments.innovation_rank,
            arguments.hidden_width,
            arguments.hidden_layers,
            arguments.epochs,
            arguments.monitoring_interval,
            arguments.monitoring_bootstrap_samples,
            arguments.final_bootstrap_samples,
        )
        <= 0
    ):
        raise ValueError("rank, architecture, epochs, monitoring, and bootstrap must be positive")
    if (
        not math.isfinite(arguments.learning_rate)
        or arguments.learning_rate <= 0.0
        or not math.isfinite(arguments.action_penalty)
        or arguments.action_penalty < 0.0
        or not math.isfinite(arguments.gradient_clip_norm)
        or arguments.gradient_clip_norm <= 0.0
        or arguments.bias_natural_gradient_interval < 0
        or not math.isfinite(arguments.bias_natural_gradient_ridge)
        or arguments.bias_natural_gradient_ridge <= 0.0
        or arguments.residual_adversary_epochs < 0
        or not math.isfinite(arguments.residual_adversary_learning_rate)
        or arguments.residual_adversary_learning_rate <= 0.0
        or arguments.residual_adversary_monitoring_interval <= 0
        or arguments.residual_lbfgs_iterations_per_epoch <= 0
        or arguments.residual_hidden_width < 0
        or arguments.residual_hidden_layers < 0
        or arguments.residual_bias_natural_gradient_interval < 0
        or not math.isfinite(arguments.residual_bias_natural_gradient_ridge)
        or arguments.residual_bias_natural_gradient_ridge <= 0.0
        or arguments.residual_final_bias_root_steps < 0
        or not math.isfinite(arguments.residual_final_bias_root_ridge)
        or arguments.residual_final_bias_root_ridge <= 0.0
        or not math.isfinite(arguments.residual_factor_maximum_condition_number)
        or arguments.residual_factor_maximum_condition_number <= 1.0
    ):
        raise ValueError("optimizer scalars are invalid")
    return notionals


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _json_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    """Return command arguments with filesystem paths represented as strings."""

    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(arguments).items()
    }


def _physical_tower_dataset(
    *,
    model: Any,
    geometry: FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    surface: Any,
    parameters: Any,
    raw_log_loadings: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    num_paths: int,
    path_batch_size: int,
    pairs_per_block: int,
    seed: int,
) -> PhysicalTowerDataset:
    """Generate the specified zero-raw-drift physical shrinking tower."""

    device = parameters.dlvsqs.device
    dtype = parameters.dlvsqs.dtype
    rank = raw_log_loadings.shape[-1]
    generator = torch.Generator(device=device).manual_seed(seed)
    feature_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    return_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    spot_return_parts: list[torch.Tensor] = []
    maximum_mass_residual = 0.0
    maximum_forward_residual = 0.0
    minimum_density = float("inf")
    minimum_itm_price = float("inf")
    maximum_absolute_raw_innovation = 0.0
    base_support = surface.support_strikes[:, 0]
    calendar_time_increment_y = 5.0 / DAYS_PER_YEAR
    started = perf_counter()
    for start in range(0, num_paths, path_batch_size):
        paths = min(path_batch_size, num_paths - start)
        pairs = paths // 2
        normal_half = torch.randn(
            pairs,
            4,
            rank,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        normals = torch.stack((normal_half, -normal_half), dim=1).reshape(paths, 4, rank)
        uniforms = _randomized_stratified_antithetic_uniforms(
            num_paths=paths,
            num_weeks=4,
            pairs_per_block=pairs_per_block,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        state = TriangularFixedGridState(
            support=base_support.expand(paths, -1),
            densities=surface.densities.expand(paths, -1, -1),
            dlvsqs=parameters.dlvsqs.expand(paths, -1, -1),
        )
        current_spot = torch.ones(paths, dtype=dtype, device=device)
        batch_spot_returns: list[torch.Tensor] = []
        for trade_week in range(4):
            num_live_expiries = 4 - trade_week
            feature_parts[trade_week].append(
                torch.sqrt(torch.clamp_min(state.densities, 0.0)).reshape(paths, -1).float()
            )
            option_signs = torch.where(
                fixed_cash_strikes[None] <= current_spot[:, None],
                1.0,
                -1.0,
            )
            live_loadings = raw_log_loadings[:num_live_expiries]
            raw_innovation = calendar_time_increment_y**0.5 * torch.einsum(
                "tif,bf->bti", live_loadings, normals[:, trade_week]
            )
            maximum_absolute_raw_innovation = max(
                maximum_absolute_raw_innovation,
                float(torch.max(torch.abs(raw_innovation))),
            )
            roll = triangular_fixed_grid_physical_roll(
                model.sanos_decoder,
                state,
                raw_log_loadings=live_loadings,
                minimum_dlvsq=geometry.min_dlvsq,
                expiry_time_increment_y=model.sanos_decoder.model_dts_y[0],
                calendar_time_increment_y=calendar_time_increment_y,
                standard_normals=normals[:, trade_week],
                uniforms=uniforms[:, trade_week],
                fixed_cash_strikes=fixed_cash_strikes,
                option_signs=option_signs,
            )
            prices = roll.initial_call_values
            if not bool(torch.all(torch.isfinite(prices) & (prices > 0.0))):
                raise RuntimeError("physical tower produced a nonpositive ITM price")
            minimum_itm_price = min(minimum_itm_price, float(torch.min(prices)))
            next_spot = roll.revealed_spot
            batch_spot_returns.append(next_spot / current_spot - 1.0)
            one_step_returns = (roll.final_call_values / prices - 1.0).reshape(paths, -1)
            if not bool(torch.all(torch.isfinite(one_step_returns))):
                raise RuntimeError("physical tower produced a non-finite option return")
            return_parts[trade_week].append(one_step_returns.float())
            maximum_mass_residual = max(maximum_mass_residual, float(roll.maximum_mass_residual))
            maximum_forward_residual = max(
                maximum_forward_residual, float(roll.maximum_forward_residual)
            )
            minimum_density = min(minimum_density, float(roll.minimum_density))
            current_spot = next_spot
            if trade_week < 3:
                if roll.next_state is None:
                    raise RuntimeError("physical tower ended before the final expiry")
                state = roll.next_state

        spot_return_parts.append(torch.stack(batch_spot_returns, dim=1).float())
        print(f"generated {start + paths}/{num_paths} paths (seed {seed})", flush=True)
    return PhysicalTowerDataset(
        features_by_time=tuple(torch.cat(parts, dim=0) for parts in feature_parts),
        returns_by_time=tuple(torch.cat(parts, dim=0) for parts in return_parts),
        spot_returns=torch.cat(spot_return_parts, dim=0),
        paths_per_independent_block=2 * pairs_per_block,
        diagnostics={
            "num_paths": num_paths,
            "num_independent_blocks": num_paths // (2 * pairs_per_block),
            "generation_seconds": perf_counter() - started,
            "minimum_density": minimum_density,
            "minimum_itm_option_price": minimum_itm_price,
            "maximum_mass_residual": maximum_mass_residual,
            "maximum_forward_residual": maximum_forward_residual,
            "maximum_absolute_raw_log_dlv_innovation": (maximum_absolute_raw_innovation),
        },
    )


def _feature_statistics(
    dataset: PhysicalTowerDataset,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    means: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for features in dataset.features_by_time:
        mean = torch.mean(features, dim=0)
        standard_deviation = torch.std(features, dim=0, correction=0)
        scale = torch.where(
            standard_deviation > 1.0e-5,
            standard_deviation,
            torch.ones_like(standard_deviation),
        )
        means.append(mean)
        scales.append(scale)
    return tuple(means), tuple(scales)


def _dataset_batch(
    dataset: PhysicalTowerDataset,
    indices: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return (
        [values[indices] for values in dataset.features_by_time],
        [values[indices] for values in dataset.returns_by_time],
    )


def _beta_whitened_dataset(
    dataset: PhysicalTowerDataset,
    *,
    maximum_condition_number: float,
    noise_seed: int,
) -> tuple[PhysicalTowerDataset, BetaWhiteningTransform]:
    r"""Build full-rank spot/beta-residual coordinates from fixed P moments.

    No sample mean is subtracted from the returned trading instruments.  Means
    are used only to estimate beta and covariance.  Consequently zero means of
    every transformed coordinate are equivalent to zero means of spot and all
    original option returns.
    """

    if not math.isfinite(maximum_condition_number) or maximum_condition_number <= 1.0:
        raise ValueError("maximum_condition_number must be finite and above one")
    transformed_returns: list[torch.Tensor] = []
    beta_by_time: list[torch.Tensor] = []
    spot_scales: list[torch.Tensor] = []
    eigenvectors_by_time: list[torch.Tensor] = []
    inverse_sqrt_eigenvalues: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int]] = []
    generator = torch.Generator(device=dataset.spot_returns.device).manual_seed(noise_seed)
    for time_index, raw_returns in enumerate(dataset.returns_by_time):
        spot = dataset.spot_returns[:, time_index].double()
        raw = raw_returns.double()
        centered_spot = spot - torch.mean(spot)
        centered_raw = raw - torch.mean(raw, dim=0)
        spot_sum_of_squares = torch.sum(centered_spot.square()).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        beta = (
            torch.sum(centered_spot[:, None] * centered_raw, dim=0) / spot_sum_of_squares
        ).detach()
        residual = raw - spot[:, None] * beta[None]
        centered_residual = residual - torch.mean(residual, dim=0)
        covariance = centered_residual.T @ centered_residual / (dataset.num_paths - 1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(0.0)
        maximum_eigenvalue = torch.amax(eigenvalues)
        eigenvalue_floor = maximum_eigenvalue / maximum_condition_number
        conditioned_eigenvalues = torch.clamp_min(eigenvalues, eigenvalue_floor)
        inverse_sqrt = torch.rsqrt(conditioned_eigenvalues)
        added_noise_scale = torch.sqrt(torch.clamp_min(conditioned_eigenvalues - eigenvalues, 0.0))
        spot_scale = torch.std(centered_spot, correction=1).clamp_min(
            torch.finfo(torch.float64).eps
        )
        # Transform the uncentered instruments: centering here would erase the
        # very empirical drift that the root solve must remove.
        projected_residual = residual @ eigenvectors
        pair_noise = torch.randn(
            dataset.num_paths // 2,
            raw_returns.shape[1],
            dtype=torch.float64,
            device=raw_returns.device,
            generator=generator,
        )
        antithetic_noise = torch.stack((pair_noise, -pair_noise), dim=1).reshape(
            dataset.num_paths,
            raw_returns.shape[1],
        )
        augmented_residual = projected_residual + antithetic_noise * added_noise_scale[None]
        residual_coordinates = augmented_residual * inverse_sqrt[None]
        factor_returns = torch.cat((spot[:, None] / spot_scale, residual_coordinates), dim=1)
        transformed_returns.append(factor_returns.to(dtype=raw_returns.dtype))
        beta_by_time.append(beta)
        spot_scales.append(spot_scale)
        eigenvectors_by_time.append(eigenvectors)
        inverse_sqrt_eigenvalues.append(inverse_sqrt)
        diagnostics.append(
            {
                "trade_week": time_index,
                "coordinates": raw_returns.shape[1],
                "raw_minimum_covariance_eigenvalue": float(torch.amin(eigenvalues)),
                "raw_maximum_covariance_eigenvalue": float(maximum_eigenvalue),
                "eigenvalue_floor": float(eigenvalue_floor),
                "floored_eigenvalues": int(torch.sum(eigenvalues < eigenvalue_floor)),
                "maximum_added_noise_standard_deviation": float(torch.amax(added_noise_scale)),
                "maximum_condition_number": maximum_condition_number,
            }
        )
    transformed_diagnostics = dict(dataset.diagnostics)
    transformed_diagnostics.update(
        {
            "factor_coordinates": (
                "spot/std(spot), followed by full-rank covariance-whitened "
                "(raw option return - fixed physical beta*spot)"
            ),
            "beta_and_covariance_scope": (
                "estimated once on the original equally weighted fitted path set; detached"
            ),
            "whitening_maximum_condition_number": maximum_condition_number,
        }
    )
    return (
        PhysicalTowerDataset(
            features_by_time=dataset.features_by_time,
            returns_by_time=tuple(transformed_returns),
            spot_returns=dataset.spot_returns,
            paths_per_independent_block=dataset.paths_per_independent_block,
            diagnostics=transformed_diagnostics,
        ),
        BetaWhiteningTransform(
            beta_by_time=tuple(beta_by_time),
            spot_scale_by_time=tuple(spot_scales),
            eigenvectors_by_time=tuple(eigenvectors_by_time),
            inverse_sqrt_eigenvalues_by_time=tuple(inverse_sqrt_eigenvalues),
            noise_seed=noise_seed,
            diagnostics=tuple(diagnostics),
        ),
    )


def _full_path_policy_values(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Evaluate the empirical utility density on the entire fixed path set."""

    leveraged_parts: list[torch.Tensor] = []
    normalized_parts: list[torch.Tensor] = []
    action_norm_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    with torch.no_grad():
        for start in range(0, dataset.num_paths, batch_size):
            stop = min(start + batch_size, dataset.num_paths)
            indices = torch.arange(start, stop, device=policy.notionals.device)
            features, returns = _dataset_batch(dataset, indices)
            forward = policy(features, returns)
            leveraged_parts.append(forward.leveraged_pnl)
            normalized_parts.append(forward.normalized_pnl)
            for time_index, action_norm in enumerate(forward.action_l1_by_time):
                action_norm_parts[time_index].append(action_norm)
        leveraged = torch.cat(leveraged_parts, dim=1)
        normalized = torch.cat(normalized_parts, dim=1)
        weights = normalized_powerexp_marginal(leveraged)
        utility = torch.mean(powerexp_utility(leveraged), dim=1)
        effective_sample_size = dataset.num_paths**2 / torch.sum(weights.square(), dim=1)
    return {
        "leveraged_pnl": leveraged,
        "normalized_pnl": normalized,
        "weights": weights,
        "direct_utility": utility,
        "effective_sample_size": effective_sample_size,
        "maximum_weight": torch.amax(weights, dim=1),
        "action_l1_by_time": tuple(torch.cat(parts, dim=1) for parts in action_norm_parts),
    }


def _objective_gradient_audit(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    *,
    batch_size: int,
) -> dict[str, object]:
    r"""Audit the full-path normalized derivative of direct utility.

    The reported derivative is the gradient of direct utility divided by the
    positive scalar ``N * mean(u'(pi_N))`` separately for each head. It has the
    same zeros as the unscaled objective derivative but remains numerically
    comparable when the marginal utility spans many orders of magnitude.
    """

    values = _full_path_policy_values(policy, dataset, batch_size=batch_size)
    weights = values["weights"]
    policy.zero_grad(set_to_none=True)
    for start in range(0, dataset.num_paths, batch_size):
        stop = min(start + batch_size, dataset.num_paths)
        indices = torch.arange(start, stop, device=policy.notionals.device)
        features, returns = _dataset_batch(dataset, indices)
        forward = policy(features, returns)
        normalized_loss_by_head = (
            -torch.sum(
                weights[:, start:stop].detach() * forward.normalized_pnl,
                dim=1,
            )
            / dataset.num_paths
        )
        torch.sum(normalized_loss_by_head).backward()

    squared_norm = torch.zeros_like(policy.notionals)
    maximum_absolute = torch.zeros_like(policy.notionals)
    parameters_per_head = 0
    for parameter in policy.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or parameter.shape[0] != len(policy.notionals):
            raise RuntimeError("utility policy parameters must retain an independent head axis")
        flat_gradient = parameter.grad.reshape(len(policy.notionals), -1)
        squared_norm += torch.sum(flat_gradient.square(), dim=1)
        maximum_absolute = torch.maximum(
            maximum_absolute,
            torch.amax(torch.abs(flat_gradient), dim=1),
        )
        parameters_per_head += flat_gradient.shape[1]
    l2_norm = torch.sqrt(squared_norm)
    rms = torch.sqrt(squared_norm / parameters_per_head)
    policy.zero_grad(set_to_none=True)
    return {
        "normalization": "direct utility gradient divided by N*mean marginal utility",
        "parameters_per_head": parameters_per_head,
        "heads": [
            {
                "notional": float(notional),
                "l2_norm": float(l2_norm[head]),
                "rms_per_parameter": float(rms[head]),
                "maximum_absolute_coordinate": float(maximum_absolute[head]),
            }
            for head, notional in enumerate(policy.notionals)
        ],
    }


def _residual_adversary_values(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    *,
    base_q_weights: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Evaluate a fresh utility adversary under one detached empirical measure."""

    if base_q_weights.shape != (dataset.num_paths,) or not bool(
        torch.all(torch.isfinite(base_q_weights) & (base_q_weights > 0.0))
    ):
        raise ValueError("base Q weights must be one finite-positive value per path")
    values = _full_path_policy_values(policy, dataset, batch_size=batch_size)
    pnl = values["leveraged_pnl"]
    log_combined_weight = torch.log(base_q_weights)[None] + log_powerexp_marginal(pnl)
    log_combined_weight = log_combined_weight - torch.logsumexp(
        log_combined_weight,
        dim=1,
        keepdim=True,
    )
    log_combined_weight = log_combined_weight + math.log(dataset.num_paths)
    combined_weights = torch.exp(log_combined_weight).clamp_min(torch.finfo(pnl.dtype).tiny)
    combined_weights = combined_weights / torch.mean(
        combined_weights,
        dim=1,
        keepdim=True,
    )
    return {
        **values,
        "combined_weights": combined_weights,
        "utility_under_q": torch.mean(
            base_q_weights[None] * powerexp_utility(pnl),
            dim=1,
        ),
        "combined_effective_sample_size": (
            dataset.num_paths**2 / torch.sum(combined_weights.square(), dim=1)
        ),
    }


def _gradient_statistics_from_current_gradients(
    policy: DirectUtilityTowerPolicy,
) -> list[dict[str, float]]:
    """Summarize the independent head slices of accumulated gradients."""

    squared_norm = torch.zeros_like(policy.notionals)
    maximum_absolute = torch.zeros_like(policy.notionals)
    parameters_per_head = 0
    for parameter in policy.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or parameter.shape[0] != len(policy.notionals):
            raise RuntimeError("utility policy parameters must retain an independent head axis")
        flat_gradient = parameter.grad.reshape(len(policy.notionals), -1)
        squared_norm += torch.sum(flat_gradient.square(), dim=1)
        maximum_absolute = torch.maximum(
            maximum_absolute,
            torch.amax(torch.abs(flat_gradient), dim=1),
        )
        parameters_per_head += flat_gradient.shape[1]
    return [
        {
            "l2_norm": float(torch.sqrt(squared_norm[head])),
            "rms_per_parameter": float(torch.sqrt(squared_norm[head] / parameters_per_head)),
            "maximum_absolute_coordinate": float(maximum_absolute[head]),
        }
        for head in range(len(policy.notionals))
    ]


def _residual_adversary_gradient(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    *,
    combined_weights: torch.Tensor,
    batch_size: int,
) -> list[dict[str, float]]:
    """Accumulate the detached-Q residual-adversary gradient on every path."""

    policy.zero_grad(set_to_none=True)
    for start in range(0, dataset.num_paths, batch_size):
        stop = min(start + batch_size, dataset.num_paths)
        indices = torch.arange(start, stop, device=policy.notionals.device)
        features, returns = _dataset_batch(dataset, indices)
        forward = policy(features, returns)
        loss = -torch.sum(combined_weights[:, start:stop].detach() * forward.normalized_pnl) / (
            dataset.num_paths * len(policy.notionals)
        )
        loss.backward()
    statistics = _gradient_statistics_from_current_gradients(policy)
    policy.zero_grad(set_to_none=True)
    return statistics


def _direct_residual_objective_gradient(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    *,
    base_q_weights: torch.Tensor,
    batch_size: int,
) -> list[dict[str, float]]:
    """Return the unscaled gradient of negative utility under detached base Q."""

    policy.zero_grad(set_to_none=True)
    for start in range(0, dataset.num_paths, batch_size):
        stop = min(start + batch_size, dataset.num_paths)
        indices = torch.arange(start, stop, device=policy.notionals.device)
        features, returns = _dataset_batch(dataset, indices)
        forward = policy(features, returns)
        loss = -torch.mean(
            torch.sum(
                base_q_weights[start:stop][None]
                * powerexp_utility(forward.leveraged_pnl),
                dim=1,
            )
            / dataset.num_paths
        )
        loss.backward()
    statistics = _gradient_statistics_from_current_gradients(policy)
    policy.zero_grad(set_to_none=True)
    return statistics


def _strategy_significance_audit(
    values: dict[str, torch.Tensor],
    *,
    base_q_weights: torch.Tensor,
    paths_per_independent_block: int,
) -> list[dict[str, float]]:
    """Audit frozen-strategy P&L and utility over independent path blocks."""

    pnl = values["leveraged_pnl"]
    if pnl.shape[1] % paths_per_independent_block:
        raise ValueError("strategy audit requires complete independent blocks")
    blocks = pnl.shape[1] // paths_per_independent_block
    weighted_pnl = base_q_weights[None] * pnl
    weighted_utility = base_q_weights[None] * powerexp_utility(pnl)

    def statistics(contributions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_means = contributions.reshape(
            len(pnl),
            blocks,
            paths_per_independent_block,
        ).mean(dim=2)
        mean = torch.mean(block_means, dim=1)
        standard_error = torch.std(block_means, dim=1, unbiased=True) / math.sqrt(blocks)
        t_score = mean / standard_error
        return mean, standard_error, t_score

    mean_pnl, pnl_standard_error, pnl_t_score = statistics(weighted_pnl)
    mean_utility, utility_standard_error, utility_t_score = statistics(weighted_utility)
    q_variance = torch.mean(
        base_q_weights[None] * (pnl - mean_pnl[:, None]).square(),
        dim=1,
    )
    return [
        {
            "mean_pnl": float(mean_pnl[head]),
            "pnl_standard_deviation_under_q": float(torch.sqrt(q_variance[head])),
            "mean_pnl_over_standard_deviation": float(
                mean_pnl[head] / torch.sqrt(q_variance[head])
            ),
            "block_standard_error_of_mean_pnl": float(pnl_standard_error[head]),
            "mean_pnl_t_score": float(pnl_t_score[head]),
            "mean_utility": float(mean_utility[head]),
            "block_standard_error_of_mean_utility": float(utility_standard_error[head]),
            "mean_utility_t_score": float(utility_t_score[head]),
            "num_independent_blocks": blocks,
        }
        for head in range(len(pnl))
    ]


def _run_residual_adversary(
    *,
    dataset: PhysicalTowerDataset,
    base_q_weights: torch.Tensor,
    base_q_label: str,
    feature_means: Sequence[torch.Tensor],
    feature_scales: Sequence[torch.Tensor],
    output_sizes: Sequence[int],
    notionals: Sequence[float],
    hidden_width: int,
    hidden_layers: int,
    train_hidden_layers: bool,
    optimizer_kind: str,
    lbfgs_iterations_per_epoch: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    action_penalty: float,
    gradient_clip_norm: float,
    monitoring_interval: int,
    bias_natural_gradient_interval: int,
    bias_natural_gradient_ridge: float,
    final_bias_root_steps: int,
    final_bias_root_ridge: float,
    batch_size: int,
) -> tuple[
    dict[str, object],
    DirectUtilityTowerPolicy,
    torch.optim.Optimizer,
    torch.Tensor,
]:
    """Fit a fresh adversary under detached base-Q weights on identical paths."""

    base_q_weights = base_q_weights.detach()
    policy = DirectUtilityTowerPolicy(
        notionals=notionals,
        feature_means=feature_means,
        feature_scales=feature_scales,
        output_sizes=output_sizes,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        seed=seed,
        train_hidden_layers=train_hidden_layers,
    )
    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if optimizer_kind == "normalized-adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
        )
    elif optimizer_kind == "direct-lbfgs":
        optimizer = torch.optim.LBFGS(
            trainable_parameters,
            lr=learning_rate,
            max_iter=lbfgs_iterations_per_epoch,
            max_eval=2 * lbfgs_iterations_per_epoch,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"unsupported residual optimizer: {optimizer_kind}")
    initial = _residual_adversary_values(
        policy,
        dataset,
        base_q_weights=base_q_weights,
        batch_size=batch_size,
    )
    initial_gradient = _residual_adversary_gradient(
        policy,
        dataset,
        combined_weights=initial["combined_weights"],
        batch_size=batch_size,
    )
    initial_direct_gradient = _direct_residual_objective_gradient(
        policy,
        dataset,
        base_q_weights=base_q_weights,
        batch_size=batch_size,
    )
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "utility_under_q": initial["utility_under_q"].cpu().tolist(),
            "gradient": initial_gradient,
        }
    ]
    for epoch in range(1, epochs + 1):
        values = _residual_adversary_values(
            policy,
            dataset,
            base_q_weights=base_q_weights,
            batch_size=batch_size,
        )
        if optimizer_kind == "normalized-adam":
            combined_weights = values["combined_weights"]
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for start in range(0, dataset.num_paths, batch_size):
                stop = min(start + batch_size, dataset.num_paths)
                indices = torch.arange(start, stop, device=policy.notionals.device)
                features, returns = _dataset_batch(dataset, indices)
                forward = policy(features, returns)
                surrogate = (
                    -torch.sum(
                        combined_weights[:, start:stop].detach()
                        * forward.normalized_pnl,
                        dim=1,
                    )
                    / dataset.num_paths
                )
                batch_fraction = (stop - start) / dataset.num_paths
                loss_by_head = surrogate + (
                    action_penalty * batch_fraction * forward.action_penalty_by_head
                )
                batch_loss = torch.mean(loss_by_head)
                batch_loss.backward()
                epoch_loss += float(batch_loss.detach())
            raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                gradient_clip_norm,
            )
            optimizer.step()
        else:

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss_value = torch.zeros((), device=policy.notionals.device)
                for start in range(0, dataset.num_paths, batch_size):
                    stop = min(start + batch_size, dataset.num_paths)
                    indices = torch.arange(start, stop, device=policy.notionals.device)
                    features, returns = _dataset_batch(dataset, indices)
                    forward = policy(features, returns)
                    batch_fraction = (stop - start) / dataset.num_paths
                    loss = -torch.mean(
                        torch.sum(
                            base_q_weights[start:stop][None]
                            * powerexp_utility(forward.leveraged_pnl),
                            dim=1,
                        )
                        / dataset.num_paths
                    ) + action_penalty * batch_fraction * torch.mean(
                        forward.action_penalty_by_head
                    )
                    loss.backward()
                    loss_value = loss_value + loss.detach()
                return loss_value

            epoch_loss = float(optimizer.step(closure))
            squared_gradient = sum(
                torch.sum(parameter.grad.square())
                for parameter in trainable_parameters
                if parameter.grad is not None
            )
            raw_gradient_norm = torch.sqrt(squared_gradient)
        natural_gradient_diagnostics = None
        if bias_natural_gradient_interval and epoch % bias_natural_gradient_interval == 0:
            natural_gradient_diagnostics = _bias_natural_gradient_step(
                policy,
                optimizer,
                dataset,
                batch_size=batch_size,
                head_indices=range(len(notionals)),
                ridge=bias_natural_gradient_ridge,
                base_q_weights=base_q_weights,
            )
        record: dict[str, object] = {
            "epoch": epoch,
            "surrogate_loss": epoch_loss,
            "pre_update_utility_under_q": values["utility_under_q"].cpu().tolist(),
            "pre_clip_gradient_norm": float(raw_gradient_norm),
            "bias_natural_gradient": natural_gradient_diagnostics,
        }
        if epoch % monitoring_interval == 0 or epoch == epochs:
            monitored = _residual_adversary_values(
                policy,
                dataset,
                base_q_weights=base_q_weights,
                batch_size=batch_size,
            )
            record["utility_under_q"] = monitored["utility_under_q"].cpu().tolist()
            record["combined_effective_sample_size"] = (
                monitored["combined_effective_sample_size"].cpu().tolist()
            )
            print(
                f"residual epoch {epoch}: "
                + ", ".join(
                    f"N={notional:g}:uQ={utility:.6g}"
                    for notional, utility in zip(
                        notionals,
                        record["utility_under_q"],
                        strict=True,
                    )
                ),
                flush=True,
            )
        history.append(record)
    final_bias_root_history: list[dict[str, object]] = []
    for step in range(1, final_bias_root_steps + 1):
        diagnostics = _bias_natural_gradient_step(
            policy,
            optimizer,
            dataset,
            batch_size=batch_size,
            head_indices=range(len(notionals)),
            ridge=final_bias_root_ridge,
            base_q_weights=base_q_weights,
        )
        final_bias_root_history.append({"step": step, "diagnostics": diagnostics})
        print(
            f"residual final bias root {step}: "
            f"max moment={max(value['maximum_absolute_unweighted_moment_before'] for value in diagnostics):.6g}",
            flush=True,
        )
    final = _residual_adversary_values(
        policy,
        dataset,
        base_q_weights=base_q_weights,
        batch_size=batch_size,
    )
    final_gradient = _residual_adversary_gradient(
        policy,
        dataset,
        combined_weights=final["combined_weights"],
        batch_size=batch_size,
    )
    final_direct_gradient = _direct_residual_objective_gradient(
        policy,
        dataset,
        base_q_weights=base_q_weights,
        batch_size=batch_size,
    )
    final_strategy_significance = _strategy_significance_audit(
        final,
        base_q_weights=base_q_weights,
        paths_per_independent_block=dataset.paths_per_independent_block,
    )
    report = {
        "base_q_label": base_q_label,
        "base_q_weights_detached": True,
        "path_scope": "identical fitted path set",
        "notionals": list(notionals),
        "policy_class": {
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "hidden_seed": seed,
            "train_hidden_layers": train_hidden_layers,
            "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in policy.parameters()
                if parameter.requires_grad
            ),
        },
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer_kind,
        "lbfgs_iterations_per_epoch": lbfgs_iterations_per_epoch,
        "initial_utility_under_q": initial["utility_under_q"].cpu().tolist(),
        "final_utility_under_q": final["utility_under_q"].cpu().tolist(),
        "final_combined_effective_sample_size": final["combined_effective_sample_size"]
        .cpu()
        .tolist(),
        "initial_gradient": initial_gradient,
        "final_gradient": final_gradient,
        "initial_direct_objective_gradient": initial_direct_gradient,
        "final_direct_objective_gradient": final_direct_gradient,
        "final_strategy_significance": final_strategy_significance,
        "final_bias_root_history": final_bias_root_history,
        "final_action_l1_by_trade_week": [
            {
                "trade_week": time_index,
                "maximum": float(torch.amax(action_norms[0])),
                "median": float(torch.quantile(action_norms[0], 0.5)),
                "fraction_at_notional_cap": float(
                    torch.mean((action_norms[0] >= notionals[0] * (1.0 - 1.0e-5)).float())
                ),
            }
            for time_index, action_norms in enumerate(final["action_l1_by_time"])
        ],
        "history": history,
    }
    return report, policy, optimizer, final["combined_weights"].detach()


def _residual_adversary_checkpoint(
    *,
    policy: DirectUtilityTowerPolicy,
    optimizer: torch.optim.Optimizer,
    report: dict[str, object],
    base_checkpoint: Path | None,
    empirical_q_weights: torch.Tensor,
    beta_whitening: BetaWhiteningTransform,
) -> dict[str, object]:
    """Return a self-contained certificate checkpoint without copying base Q."""

    return {
        "schema_version": 1,
        "experiment_kind": "same_path_detached_q_residual_utility_adversary",
        "base_q_checkpoint": (None if base_checkpoint is None else str(base_checkpoint)),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "empirical_q_weights": empirical_q_weights.cpu(),
        "beta_whitening": {
            "beta_by_time": tuple(value.cpu() for value in beta_whitening.beta_by_time),
            "spot_scale_by_time": tuple(value.cpu() for value in beta_whitening.spot_scale_by_time),
            "eigenvectors_by_time": tuple(
                value.cpu() for value in beta_whitening.eigenvectors_by_time
            ),
            "inverse_sqrt_eigenvalues_by_time": tuple(
                value.cpu() for value in beta_whitening.inverse_sqrt_eigenvalues_by_time
            ),
            "noise_seed": beta_whitening.noise_seed,
            "diagnostics": beta_whitening.diagnostics,
        },
        "report": report,
    }


def _bias_natural_gradient_step(
    policy: DirectUtilityTowerPolicy,
    optimizer: torch.optim.Optimizer,
    dataset: PhysicalTowerDataset,
    *,
    batch_size: int,
    head_indices: Sequence[int],
    ridge: float,
    base_q_weights: torch.Tensor | None = None,
) -> list[dict[str, object]]:
    """Apply damped utility-curvature root steps in constant-action directions."""

    returns = torch.cat(dataset.returns_by_time, dim=1)
    if base_q_weights is None:
        values = _full_path_policy_values(policy, dataset, batch_size=batch_size)
        weights_by_head = values["weights"]
        utility_by_head = values["direct_utility"]
    else:
        values = _residual_adversary_values(
            policy,
            dataset,
            base_q_weights=base_q_weights,
            batch_size=batch_size,
        )
        weights_by_head = values["combined_weights"]
        utility_by_head = values["utility_under_q"]
    diagnostics: list[dict[str, object]] = []
    with torch.no_grad():
        for head in head_indices:
            pnl = values["leveraged_pnl"][head]
            weights = weights_by_head[head]
            kappa = torch.where(pnl > 0.0, 1.0, torch.reciprocal(1.0 - pnl))
            mean = torch.mean(weights[:, None] * returns, dim=0)
            weighted_returns = returns * torch.sqrt(weights * kappa / dataset.num_paths)[:, None]
            curvature = weighted_returns.T @ weighted_returns
            scale = torch.sqrt(torch.clamp_min(torch.diagonal(curvature), 1.0e-12))
            correlation = curvature / (scale[:, None] * scale[None])
            correlation = 0.5 * (correlation + correlation.T)
            diagonal = torch.arange(len(correlation), device=correlation.device)
            applied_ridge = ridge
            for _ in range(6):
                regularized = correlation.clone()
                regularized[diagonal, diagonal] += applied_ridge
                cholesky, info = torch.linalg.cholesky_ex(regularized)
                if int(torch.max(info)) == 0:
                    break
                applied_ridge *= 10.0
            else:
                raise RuntimeError("could not regularize the utility curvature")
            correction = (
                torch.cholesky_solve(
                    (mean / scale)[:, None],
                    cholesky,
                ).squeeze(1)
                / scale
            )
            final_biases = [network.biases[-1] for network in policy.policies]
            original_biases = [bias[head].clone() for bias in final_biases]
            before_utility = float(utility_by_head[head])
            accepted_step = 0.0
            after_utility = before_utility
            for line_search_power in range(11):
                step = 0.5**line_search_power
                offset = 0
                for bias, original, block_returns in zip(
                    final_biases,
                    original_biases,
                    dataset.returns_by_time,
                    strict=True,
                ):
                    width = block_returns.shape[1]
                    bias[head].copy_(original + step * correction[offset : offset + width])
                    offset += width
                if base_q_weights is None:
                    trial = _full_path_policy_values(
                        policy,
                        dataset,
                        batch_size=batch_size,
                    )
                    trial_utility = float(trial["direct_utility"][head])
                else:
                    trial = _residual_adversary_values(
                        policy,
                        dataset,
                        base_q_weights=base_q_weights,
                        batch_size=batch_size,
                    )
                    trial_utility = float(trial["utility_under_q"][head])
                if trial_utility >= before_utility:
                    accepted_step = step
                    after_utility = trial_utility
                    break
            if accepted_step == 0.0:
                for bias, original in zip(final_biases, original_biases, strict=True):
                    bias[head].copy_(original)
            else:
                for bias in final_biases:
                    state = optimizer.state.get(bias)
                    if state:
                        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                            if name in state:
                                state[name][head].zero_()
            diagnostics.append(
                {
                    "notional": float(policy.notionals[head]),
                    "ridge": applied_ridge,
                    "accepted_step": accepted_step,
                    "utility_before": before_utility,
                    "utility_after": after_utility,
                    "maximum_absolute_unweighted_moment_before": float(torch.max(torch.abs(mean))),
                    "correction_l2_norm": float(torch.linalg.vector_norm(correction)),
                }
            )
    return diagnostics


def _correlation_aware_multiplier_test(
    observations: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | bool]:
    """Joint max/RMS zero-mean test retaining empirical cross-option dependence."""

    blocks, coordinates = observations.shape
    mean = np.mean(observations, axis=0)
    standard_deviation = np.std(observations, axis=0, ddof=1)
    active = standard_deviation > max(1.0e-14, 1.0e-10 * float(np.max(standard_deviation)))
    observed_t = np.sqrt(blocks) * mean[active] / standard_deviation[active]
    maximum = float(np.max(np.abs(observed_t)))
    rms = float(np.sqrt(np.mean(np.square(observed_t))))
    centered = observations[:, active] - mean[active]
    generator = np.random.default_rng(seed)
    bootstrap_maximum: list[np.ndarray] = []
    bootstrap_rms: list[np.ndarray] = []
    remaining = samples
    while remaining:
        count = min(128, remaining)
        signs = generator.choice((-1.0, 1.0), size=(count, blocks))
        bootstrap_t = signs @ centered / (np.sqrt(blocks) * standard_deviation[active])
        bootstrap_maximum.append(np.max(np.abs(bootstrap_t), axis=1))
        bootstrap_rms.append(np.sqrt(np.mean(np.square(bootstrap_t), axis=1)))
        remaining -= count
    maximum_values = np.concatenate(bootstrap_maximum)
    rms_values = np.concatenate(bootstrap_rms)
    maximum_p = float((1 + np.sum(maximum_values >= maximum)) / (samples + 1))
    rms_p = float((1 + np.sum(rms_values >= rms)) / (samples + 1))
    combined_p = min(1.0, 2.0 * min(maximum_p, rms_p))
    return {
        "num_independent_blocks": blocks,
        "num_coordinates": coordinates,
        "num_active_coordinates": int(np.sum(active)),
        "maximum_absolute_t": maximum,
        "root_mean_square_t": rms,
        "maximum_absolute_t_p_value": maximum_p,
        "root_mean_square_t_p_value": rms_p,
        "combined_p_value": combined_p,
        "zero_mean_not_rejected_at_five_percent": combined_p >= 0.05,
    }


def _weighted_martingale_audit(
    dataset: PhysicalTowerDataset,
    weights: torch.Tensor,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Test original return means under externally supplied empirical densities."""

    if weights.ndim != 2 or weights.shape[1] != dataset.num_paths:
        raise ValueError("weights must have shape (heads, paths)")
    if not bool(torch.all(torch.isfinite(weights) & (weights >= 0.0))):
        raise ValueError("empirical density weights must be finite and nonnegative")
    mass_residual = torch.abs(torch.mean(weights, dim=1) - 1.0)
    if bool(torch.any(mass_residual > 1.0e-4)):
        raise ValueError("empirical density weights must have sample mean one")
    paths_per_block = dataset.paths_per_independent_block
    num_blocks = dataset.num_paths // paths_per_block
    weight_blocks = weights.reshape(len(weights), num_blocks, paths_per_block)
    option_blocks_by_head: list[list[np.ndarray]] = [[] for _ in range(len(weights))]
    coordinate_summaries: list[list[dict[str, float | int]]] = [[] for _ in range(len(weights))]
    with torch.no_grad():
        for time_index, returns in enumerate(dataset.returns_by_time):
            return_blocks = returns.reshape(num_blocks, paths_per_block, -1)
            block_means = (
                torch.einsum("mbp,bpi->mbi", weight_blocks, return_blocks) / paths_per_block
            )
            mean = torch.mean(block_means, dim=1)
            standard_error = torch.std(block_means, dim=1, correction=1) / math.sqrt(num_blocks)
            active = standard_error > 1.0e-14
            coordinate_t = torch.where(
                active,
                mean / torch.where(active, standard_error, 1.0),
                torch.nan,
            )
            for head in range(len(weights)):
                option_blocks_by_head[head].append(block_means[head].cpu().double().numpy())
                coordinate_summaries[head].append(
                    {
                        "trade_week": time_index,
                        "num_coordinates": returns.shape[1],
                        "maximum_absolute_weighted_mean": float(torch.max(torch.abs(mean[head]))),
                        "maximum_absolute_t": float(
                            torch.max(torch.abs(coordinate_t[head][active[head]]))
                        ),
                        "root_mean_square_t": float(
                            torch.sqrt(torch.mean(coordinate_t[head][active[head]].square()))
                        ),
                        "num_absolute_t_above_2": int(
                            torch.sum(torch.abs(coordinate_t[head]) > 2.0)
                        ),
                    }
                )
        spot_blocks = (
            torch.einsum(
                "mbp,bpi->mbi",
                weight_blocks,
                dataset.spot_returns.reshape(num_blocks, paths_per_block, 4),
            )
            / paths_per_block
        )
    heads: list[dict[str, object]] = []
    for head in range(len(weights)):
        option_test = _correlation_aware_multiplier_test(
            np.concatenate(option_blocks_by_head[head], axis=1),
            samples=bootstrap_samples,
            seed=bootstrap_seed + head,
        )
        spot_test = _correlation_aware_multiplier_test(
            spot_blocks[head].cpu().double().numpy(),
            samples=bootstrap_samples,
            seed=bootstrap_seed + 100 + head,
        )
        effective_sample_size = dataset.num_paths**2 / torch.sum(weights[head].square())
        heads.append(
            {
                "density_mass_residual": float(mass_residual[head]),
                "density_effective_sample_size": float(effective_sample_size),
                "density_effective_sample_fraction": float(
                    effective_sample_size / dataset.num_paths
                ),
                "maximum_density_weight": float(torch.max(weights[head])),
                "weighted_option_zero_drift": option_test,
                "weighted_option_by_trade_week": coordinate_summaries[head],
                "weighted_spot_zero_drift": spot_test,
            }
        )
    return {
        "num_paths": dataset.num_paths,
        "bootstrap_samples": bootstrap_samples,
        "heads": heads,
    }


def _evaluate(
    policy: DirectUtilityTowerPolicy,
    dataset: PhysicalTowerDataset,
    *,
    batch_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Evaluate utilities, density weights, action norms, and martingale moments."""

    leveraged_parts: list[torch.Tensor] = []
    normalized_parts: list[torch.Tensor] = []
    norm_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    with torch.no_grad():
        for start in range(0, dataset.num_paths, batch_size):
            stop = min(start + batch_size, dataset.num_paths)
            indices = torch.arange(start, stop, device=policy.notionals.device)
            features, returns = _dataset_batch(dataset, indices)
            forward = policy(features, returns)
            leveraged_parts.append(forward.leveraged_pnl)
            normalized_parts.append(forward.normalized_pnl)
            for time_index, values in enumerate(forward.action_l1_by_time):
                norm_parts[time_index].append(values)
        leveraged = torch.cat(leveraged_parts, dim=1)
        normalized = torch.cat(normalized_parts, dim=1)
        weights = normalized_powerexp_marginal(leveraged)
        utility = torch.mean(powerexp_utility(leveraged), dim=1)
        density_mass_residual = torch.abs(torch.mean(weights, dim=1) - 1.0)
        effective_sample_size = dataset.num_paths**2 / torch.sum(weights.square(), dim=1)
        maximum_weight = torch.amax(weights, dim=1)
        action_norms = [torch.cat(parts, dim=1) for parts in norm_parts]

        paths_per_block = dataset.paths_per_independent_block
        num_blocks = dataset.num_paths // paths_per_block
        weight_blocks = weights.reshape(len(weights), num_blocks, paths_per_block)
        option_blocks_by_head: list[list[np.ndarray]] = [[] for _ in range(len(weights))]
        coordinate_summaries: list[list[dict[str, float | int]]] = [[] for _ in range(len(weights))]
        for time_index, returns in enumerate(dataset.returns_by_time):
            return_blocks = returns.reshape(num_blocks, paths_per_block, -1)
            block_means = (
                torch.einsum("mbp,bpi->mbi", weight_blocks, return_blocks) / paths_per_block
            )
            mean = torch.mean(block_means, dim=1)
            standard_error = torch.std(block_means, dim=1, correction=1) / math.sqrt(num_blocks)
            active = standard_error > 1.0e-14
            coordinate_t = torch.where(
                active,
                mean / torch.where(active, standard_error, 1.0),
                torch.nan,
            )
            for head in range(len(weights)):
                option_blocks_by_head[head].append(block_means[head].cpu().double().numpy())
                coordinate_summaries[head].append(
                    {
                        "trade_week": time_index,
                        "num_coordinates": returns.shape[1],
                        "maximum_absolute_weighted_mean": float(torch.max(torch.abs(mean[head]))),
                        "maximum_absolute_t": float(
                            torch.max(torch.abs(coordinate_t[head][active[head]]))
                        ),
                        "root_mean_square_t": float(
                            torch.sqrt(torch.mean(coordinate_t[head][active[head]].square()))
                        ),
                        "num_absolute_t_above_2": int(
                            torch.sum(torch.abs(coordinate_t[head]) > 2.0)
                        ),
                    }
                )
        spot_blocks = (
            torch.einsum(
                "mbp,bpi->mbi",
                weight_blocks,
                dataset.spot_returns.reshape(num_blocks, paths_per_block, 4),
            )
            / paths_per_block
        )

    heads: list[dict[str, object]] = []
    for head, notional in enumerate(policy.notionals.cpu().tolist()):
        option_blocks = np.concatenate(option_blocks_by_head[head], axis=1)
        option_test = _correlation_aware_multiplier_test(
            option_blocks,
            samples=bootstrap_samples,
            seed=bootstrap_seed + head,
        )
        spot_test = _correlation_aware_multiplier_test(
            spot_blocks[head].cpu().double().numpy(),
            samples=bootstrap_samples,
            seed=bootstrap_seed + 100 + head,
        )
        norm_summary: list[dict[str, float | int]] = []
        for time_index, values in enumerate(action_norms):
            head_values = values[head]
            quantiles = torch.quantile(
                head_values,
                torch.tensor(
                    [0.5, 0.9, 0.99],
                    dtype=head_values.dtype,
                    device=head_values.device,
                ),
            )
            norm_summary.append(
                {
                    "trade_week": time_index,
                    "median": float(quantiles[0]),
                    "q90": float(quantiles[1]),
                    "q99": float(quantiles[2]),
                    "maximum": float(torch.max(head_values)),
                    "fraction_at_notional_cap": float(
                        torch.mean(
                            (
                                head_values
                                >= policy.notionals[head] - 1.0e-5 * policy.notionals[head]
                            ).float()
                        )
                    ),
                }
            )
        head_pnl = leveraged[head]
        heads.append(
            {
                "notional": notional,
                "direct_utility": float(utility[head]),
                "mean_pnl": float(torch.mean(head_pnl)),
                "standard_deviation_pnl": float(torch.std(head_pnl, correction=1)),
                "loss_probability": float(torch.mean((head_pnl < 0.0).float())),
                "density_mass_residual": float(density_mass_residual[head]),
                "density_effective_sample_size": float(effective_sample_size[head]),
                "density_effective_sample_fraction": float(
                    effective_sample_size[head] / dataset.num_paths
                ),
                "maximum_density_weight": float(maximum_weight[head]),
                "weighted_option_zero_drift": option_test,
                "weighted_option_by_trade_week": coordinate_summaries[head],
                "weighted_spot_zero_drift": spot_test,
                "action_l1_by_trade_week": norm_summary,
                "normalized_pnl_mean": float(torch.mean(normalized[head])),
            }
        )
    return {
        "num_paths": dataset.num_paths,
        "bootstrap_samples": bootstrap_samples,
        "heads": heads,
    }


def _checkpoint_payload(
    *,
    policy: DirectUtilityTowerPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    arguments: argparse.Namespace,
    notionals: Sequence[float],
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "epoch": epoch,
        "arguments": _json_arguments(arguments),
        "notionals": list(notionals),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }


def run(arguments: argparse.Namespace) -> None:
    """Generate common paths and train every fixed-notional policy head."""

    notionals = _validate_arguments(arguments)
    natural_gradient_notionals = _notionals(arguments.bias_natural_gradient_notionals)
    unknown_notionals = set(natural_gradient_notionals) - set(notionals)
    if unknown_notionals:
        raise ValueError(
            "bias-natural-gradient-notionals are absent from the frontier: "
            f"{sorted(unknown_notionals)}"
        )
    natural_gradient_heads = [notionals.index(value) for value in natural_gradient_notionals]
    env = BaseEnvironment(precision=64, device=arguments.device, debug=False)
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    geometry = model.config.geometry
    if not isinstance(geometry, FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("direct utility requires the fitted fixed cash grid")
    if not 0 <= arguments.state_index < len(latent_states):
        raise ValueError("state-index lies outside the static state table")
    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        fixed_expiry_fit = fit_fixed_expiry_ar1(
            all_raw.cpu().double().numpy(),
            dates,
            num_inner_strikes=geometry.num_inner_strikes,
            rank=arguments.innovation_rank,
            training_end=date(2024, 12, 31),
        )
        raw_log_loadings = torch_fixed_expiry_raw_log_loadings(
            fixed_expiry_fit,
            dtype=env.fdtype,
            device=env.device,
        )
        selected_raw = all_raw[arguments.state_index : arguments.state_index + 1]
        parameters = bound_native_raw(selected_raw, geometry=geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        fixed_cash_strikes = torch.as_tensor(
            geometry.inner_relative_strikes,
            dtype=env.fdtype,
            device=env.device,
        )
        training_data = _physical_tower_dataset(
            model=model,
            geometry=geometry,
            surface=surface,
            parameters=parameters,
            raw_log_loadings=raw_log_loadings,
            fixed_cash_strikes=fixed_cash_strikes,
            num_paths=arguments.training_paths,
            path_batch_size=arguments.path_generation_batch_size,
            pairs_per_block=arguments.antithetic_pairs_per_block,
            seed=arguments.training_seed,
        )
        validation_data = None
        if arguments.validation_paths:
            validation_data = _physical_tower_dataset(
                model=model,
                geometry=geometry,
                surface=surface,
                parameters=parameters,
                raw_log_loadings=raw_log_loadings,
                fixed_cash_strikes=fixed_cash_strikes,
                num_paths=arguments.validation_paths,
                path_batch_size=arguments.path_generation_batch_size,
                pairs_per_block=arguments.antithetic_pairs_per_block,
                seed=arguments.validation_seed,
            )
    audit_data = training_data if validation_data is None else validation_data
    audit_seed = arguments.training_seed if validation_data is None else arguments.validation_seed
    audit_scope = "fitted_path_set" if validation_data is None else "separate_path_set"
    feature_means, feature_scales = _feature_statistics(training_data)
    policy = DirectUtilityTowerPolicy(
        notionals=notionals,
        feature_means=feature_means,
        feature_scales=feature_scales,
        output_sizes=[values.shape[1] for values in training_data.returns_by_time],
        hidden_width=arguments.hidden_width,
        hidden_layers=arguments.hidden_layers,
        seed=arguments.model_seed,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    history: list[dict[str, object]]
    if arguments.resume:
        payload = torch.load(arguments.checkpoint, map_location=env.device, weights_only=False)
        if payload.get("schema_version") != 2:
            raise ValueError("resume checkpoint is not an empirical-density version-2 checkpoint")
        checkpoint_arguments = payload.get("arguments", {})
        immutable_arguments = (
            "static_decoder_checkpoint",
            "static_report",
            "state_index",
            "training_paths",
            "validation_paths",
            "antithetic_pairs_per_block",
            "innovation_rank",
            "hidden_width",
            "hidden_layers",
            "action_penalty",
            "training_seed",
            "validation_seed",
            "model_seed",
        )
        current_arguments = _json_arguments(arguments)
        mismatched = [
            name
            for name in immutable_arguments
            if checkpoint_arguments.get(name) != current_arguments.get(name)
        ]
        checkpoint_notionals = payload.get("notionals")
        if arguments.allow_last_notional_expansion:
            valid_notionals = (
                isinstance(checkpoint_notionals, list)
                and len(checkpoint_notionals) == len(notionals)
                and checkpoint_notionals[:-1] == list(notionals[:-1])
                and checkpoint_notionals[-1] < notionals[-1]
            )
        else:
            valid_notionals = checkpoint_notionals == list(notionals)
        if mismatched or not valid_notionals:
            raise ValueError(
                f"resume checkpoint does not describe the same path set/model: {mismatched}"
            )
        policy.load_state_dict(payload["model_state_dict"])
        if arguments.allow_last_notional_expansion:
            with torch.no_grad():
                policy.notionals[-1] = notionals[-1]
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = arguments.learning_rate
        history = payload["history"]
        start_epoch = int(payload["epoch"]) + 1
        initial_gradient = history[0]["objective_gradient"]
        gradient_baseline = initial_gradient
        if arguments.allow_last_notional_expansion:
            gradient_baseline = _objective_gradient_audit(
                policy,
                training_data,
                batch_size=arguments.training_batch_size,
            )
        print(
            f"resumed epoch {start_epoch - 1} on the identical {audit_scope}",
            flush=True,
        )
    else:
        history = []
        initial_audit = _evaluate(
            policy,
            audit_data,
            batch_size=arguments.training_batch_size,
            bootstrap_samples=arguments.monitoring_bootstrap_samples,
            bootstrap_seed=audit_seed + 10_000,
        )
        initial_gradient = _objective_gradient_audit(
            policy,
            training_data,
            batch_size=arguments.training_batch_size,
        )
        history.append(
            {
                "epoch": 0,
                "path_set_audit": initial_audit,
                "objective_gradient": initial_gradient,
            }
        )
        start_epoch = 1
        gradient_baseline = initial_gradient
        print(f"initial {audit_scope} audit complete", flush=True)
    training_started = perf_counter()
    for epoch in range(start_epoch, arguments.epochs + 1):
        full_values = _full_path_policy_values(
            policy,
            training_data,
            batch_size=arguments.training_batch_size,
        )
        weights = full_values["weights"]
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for start in range(0, training_data.num_paths, arguments.training_batch_size):
            stop = min(start + arguments.training_batch_size, training_data.num_paths)
            indices = torch.arange(start, stop, device=env.device)
            features, returns = _dataset_batch(training_data, indices)
            forward = policy(features, returns)
            surrogate_by_head = (
                -torch.sum(
                    weights[:, start:stop].detach() * forward.normalized_pnl,
                    dim=1,
                )
                / training_data.num_paths
            )
            batch_fraction = (stop - start) / training_data.num_paths
            loss_by_head = surrogate_by_head + (
                arguments.action_penalty * batch_fraction * forward.action_penalty_by_head
            )
            batch_loss = torch.mean(loss_by_head)
            batch_loss.backward()
            epoch_loss += float(batch_loss.detach())
        raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), arguments.gradient_clip_norm
        )
        optimizer.step()
        natural_gradient_diagnostics = None
        if (
            arguments.bias_natural_gradient_interval
            and epoch % arguments.bias_natural_gradient_interval == 0
        ):
            natural_gradient_diagnostics = _bias_natural_gradient_step(
                policy,
                optimizer,
                training_data,
                batch_size=arguments.training_batch_size,
                head_indices=natural_gradient_heads,
                ridge=arguments.bias_natural_gradient_ridge,
            )
        training_summary: dict[str, object] = {
            "full_path_surrogate_loss": epoch_loss,
            "pre_update_direct_utility_by_head": full_values["direct_utility"].cpu().tolist(),
            "pre_update_effective_sample_size_by_head": full_values["effective_sample_size"]
            .cpu()
            .tolist(),
            "pre_update_maximum_density_weight_by_head": full_values["maximum_weight"]
            .cpu()
            .tolist(),
            "pre_clip_accumulated_gradient_norm": float(raw_gradient_norm),
            "gradient_clip_norm": arguments.gradient_clip_norm,
            "optimizer_steps_this_epoch": 1,
            "bias_natural_gradient": natural_gradient_diagnostics,
        }
        record: dict[str, object] = {"epoch": epoch, "training": training_summary}
        if epoch % arguments.monitoring_interval == 0 or epoch == arguments.epochs:
            path_set_audit = _evaluate(
                policy,
                audit_data,
                batch_size=arguments.training_batch_size,
                bootstrap_samples=arguments.monitoring_bootstrap_samples,
                bootstrap_seed=audit_seed + 10_000 + epoch * 100,
            )
            objective_gradient = _objective_gradient_audit(
                policy,
                training_data,
                batch_size=arguments.training_batch_size,
            )
            record["path_set_audit"] = path_set_audit
            record["objective_gradient"] = objective_gradient
            summary = ", ".join(
                (
                    f"N={head['notional']:g}:u={head['direct_utility']:.4g},"
                    f"p={head['weighted_option_zero_drift']['combined_p_value']:.3g},"
                    f"ESS={head['density_effective_sample_fraction']:.3g}"
                )
                for head in path_set_audit["heads"]
            )
            print(f"epoch {epoch}: {summary}", flush=True)
        else:
            print(
                f"epoch {epoch}: loss={epoch_loss:.6g}, grad={float(raw_gradient_norm):.6g}",
                flush=True,
            )
        history.append(record)
        if epoch % arguments.monitoring_interval == 0 or epoch == arguments.epochs:
            checkpoint = _checkpoint_payload(
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                arguments=arguments,
                notionals=notionals,
                history=history,
            )
            _atomic_torch_save(checkpoint, arguments.checkpoint)
        write_json_atomic(
            {
                "schema_version": 2,
                "experiment_kind": "empirical_direct_powerexp_utility_density_frontier",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "state_index": arguments.state_index,
                "state_date": dates[arguments.state_index].isoformat(),
                "audit_scope": audit_scope,
                "notionals": list(notionals),
                "history": history,
            },
            arguments.output,
        )

    final_path_set_audit = _evaluate(
        policy,
        audit_data,
        batch_size=arguments.training_batch_size,
        bootstrap_samples=arguments.final_bootstrap_samples,
        bootstrap_seed=audit_seed + 1_000_000,
    )
    final_objective_gradient = _objective_gradient_audit(
        policy,
        training_data,
        batch_size=arguments.training_batch_size,
    )
    initial_gradient_by_head = gradient_baseline["heads"]
    final_gradient_by_head = final_objective_gradient["heads"]
    candidate_diagnostics: list[dict[str, object]] = []
    for head, drift in enumerate(final_path_set_audit["heads"]):
        final_gradient = final_gradient_by_head[head]
        initial_l2 = initial_gradient_by_head[head]["l2_norm"]
        candidate_diagnostics.append(
            {
                "notional": drift["notional"],
                "objective_gradient": final_gradient,
                "objective_gradient_l2_fraction_of_initial": (
                    final_gradient["l2_norm"] / initial_l2
                ),
                "option_zero_drift": drift["weighted_option_zero_drift"],
            }
        )
    selected_candidate = min(
        candidate_diagnostics,
        key=lambda candidate: (
            -candidate["option_zero_drift"]["combined_p_value"],
            candidate["option_zero_drift"]["maximum_absolute_t"],
            candidate["objective_gradient"]["l2_norm"],
        ),
    )
    residual_adversary = None
    residual_checkpoint = None
    if arguments.residual_adversary_epochs:
        residual_notionals = _notionals(arguments.residual_adversary_notionals)
        if len(residual_notionals) != 1:
            raise ValueError("the residual root solve currently requires one notional")
        base_head = notionals.index(float(selected_candidate["notional"]))
        if (
            arguments.residual_base_uniform
            and arguments.residual_base_weights_checkpoint is not None
        ):
            raise ValueError(
                "residual-base-uniform and residual-base-weights-checkpoint are exclusive"
            )
        if arguments.residual_base_uniform:
            base_q_weights = torch.ones(
                training_data.num_paths,
                dtype=training_data.returns_by_time[0].dtype,
                device=env.device,
            )
            base_q_label = "uniform physical path weights"
            residual_parent_checkpoint = None
        elif arguments.residual_base_weights_checkpoint is None:
            base_values = _full_path_policy_values(
                policy,
                training_data,
                batch_size=arguments.training_batch_size,
            )
            base_q_weights = base_values["weights"][base_head].detach()
            base_q_label = f"direct-utility N={notionals[base_head]:g}"
            residual_parent_checkpoint = arguments.checkpoint
        else:
            residual_parent = torch.load(
                arguments.residual_base_weights_checkpoint,
                map_location=env.device,
                weights_only=False,
            )
            if residual_parent.get("schema_version") != 1:
                raise ValueError("residual base checkpoint has the wrong schema")
            base_q_weights = residual_parent["empirical_q_weights"].to(
                dtype=training_data.returns_by_time[0].dtype,
                device=env.device,
            )
            if base_q_weights.shape == (1, training_data.num_paths):
                base_q_weights = base_q_weights[0]
            base_q_weights = base_q_weights.clamp_min(torch.finfo(base_q_weights.dtype).tiny)
            base_q_weights = base_q_weights / torch.mean(base_q_weights)
            base_q_label = str(arguments.residual_base_weights_checkpoint)
            residual_parent_checkpoint = arguments.residual_base_weights_checkpoint
        factor_dataset, beta_whitening = _beta_whitened_dataset(
            training_data,
            maximum_condition_number=(arguments.residual_factor_maximum_condition_number),
            noise_seed=arguments.residual_factor_noise_seed,
        )
        residual_started = perf_counter()
        (
            residual_adversary,
            residual_policy,
            residual_optimizer,
            residual_q_weights,
        ) = _run_residual_adversary(
            dataset=factor_dataset,
            base_q_weights=base_q_weights,
            base_q_label=base_q_label,
            feature_means=feature_means,
            feature_scales=feature_scales,
            output_sizes=[values.shape[1] for values in factor_dataset.returns_by_time],
            notionals=residual_notionals,
            hidden_width=(arguments.residual_hidden_width or arguments.hidden_width),
            hidden_layers=(arguments.residual_hidden_layers or arguments.hidden_layers),
            train_hidden_layers=not arguments.residual_fixed_hidden_features,
            optimizer_kind=arguments.residual_optimizer,
            lbfgs_iterations_per_epoch=arguments.residual_lbfgs_iterations_per_epoch,
            seed=arguments.residual_adversary_seed,
            epochs=arguments.residual_adversary_epochs,
            learning_rate=arguments.residual_adversary_learning_rate,
            action_penalty=arguments.action_penalty,
            gradient_clip_norm=arguments.gradient_clip_norm,
            monitoring_interval=arguments.residual_adversary_monitoring_interval,
            bias_natural_gradient_interval=(arguments.residual_bias_natural_gradient_interval),
            bias_natural_gradient_ridge=(arguments.residual_bias_natural_gradient_ridge),
            final_bias_root_steps=arguments.residual_final_bias_root_steps,
            final_bias_root_ridge=arguments.residual_final_bias_root_ridge,
            batch_size=arguments.training_batch_size,
        )
        residual_adversary["seconds"] = perf_counter() - residual_started
        residual_adversary["beta_whitening"] = {
            "definition": (
                "spot/std(spot) plus full-rank PCA whitening of "
                "raw option return - fixed physical beta*spot, with fixed-seed "
                "antithetic Gaussian noise only below the eigenvalue floor; "
                "no mean subtraction"
            ),
            "beta_scope": (
                "computed once under the original equally weighted fixed paths; detached"
            ),
            "diagnostics": beta_whitening.diagnostics,
            "noise_seed": beta_whitening.noise_seed,
        }
        residual_adversary["candidate_raw_martingale_audit"] = _weighted_martingale_audit(
            training_data,
            residual_q_weights,
            bootstrap_samples=arguments.final_bootstrap_samples,
            bootstrap_seed=audit_seed + 2_000_000,
        )
        residual_adversary["candidate_factor_martingale_audit"] = _weighted_martingale_audit(
            factor_dataset,
            residual_q_weights,
            bootstrap_samples=arguments.final_bootstrap_samples,
            bootstrap_seed=audit_seed + 3_000_000,
        )
        residual_checkpoint = arguments.residual_checkpoint
        if residual_checkpoint is None:
            residual_checkpoint = arguments.checkpoint.with_name(
                f"{arguments.checkpoint.stem}.residual-adversary.pt"
            )
        _atomic_torch_save(
            _residual_adversary_checkpoint(
                policy=residual_policy,
                optimizer=residual_optimizer,
                report=residual_adversary,
                base_checkpoint=residual_parent_checkpoint,
                empirical_q_weights=residual_q_weights,
                beta_whitening=beta_whitening,
            ),
            residual_checkpoint,
        )
    report = {
        "schema_version": 2,
        "experiment_kind": "empirical_direct_powerexp_utility_density_frontier",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_state_index": arguments.state_index,
        "source_state_date": dates[arguments.state_index].isoformat(),
        "sources": sources,
        "notionals": list(notionals),
        "utility": ("direct power-exponential: 1-exp(-z) for gains and z-z^2/2 for losses"),
        "gradient": (
            "per-head direct-utility gradient normalized by N*mean marginal; "
            "log-normalized marginal weights detached only within each update"
        ),
        "action": (
            "network emits actual positions; per-date action is raw_action*N/"
            "max(N,L1(raw_action)); no instrument-count or date normalization"
        ),
        "return_definition": (
            "one-step price return of the same fixed-cash-strike, fixed-expiry claim; "
            "the front-expiry endpoint price is its payoff"
        ),
        "audit_scope": audit_scope,
        "gradient_fraction_baseline": (
            "post-expansion checkpoint"
            if arguments.allow_last_notional_expansion
            else "initialization"
        ),
        "physical_simulation": (
            "X-only fixed cash grid; four-week shrinking tower; raw log-DLV "
            "innovations from rank-truncated fixed-expiry AR residual covariance; "
            "zero raw DLV drift; no re-gridding"
        ),
        "fixed_expiry_ar1": fixed_expiry_fit.diagnostics.to_dict(),
        "training_dataset": training_data.diagnostics,
        "audit_dataset": audit_data.diagnostics,
        "arguments": _json_arguments(arguments),
        "history": history,
        "final_path_set_audit": final_path_set_audit,
        "final_objective_gradient": final_objective_gradient,
        "candidate_diagnostics": candidate_diagnostics,
        "selected_candidate": selected_candidate,
        "residual_adversary": residual_adversary,
        "residual_adversary_checkpoint": (
            None if residual_checkpoint is None else str(residual_checkpoint)
        ),
        "speed": {"training_seconds": perf_counter() - training_started},
    }
    write_json_atomic(report, arguments.output)
    if start_epoch <= arguments.epochs:
        _atomic_torch_save(
            _checkpoint_payload(
                policy=policy,
                optimizer=optimizer,
                epoch=arguments.epochs,
                arguments=arguments,
                notionals=notionals,
                history=history,
            ),
            arguments.checkpoint,
        )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
