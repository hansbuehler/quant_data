"""Audit the PDF's fixed-contract LVsq diffusion on fitted FLEX states.

This is a theorem diagnostic, not a rolling weekly Q transition.  It evolves
the original 808-coordinate raw native state while holding the front block,
all forward volatilities, and all backbone multipliers fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.lvsq_drift import (
    fixed_contract_lvsq_drift,
    fixed_contract_lvsq_euler_step,
    smooth_lvsq_log_loadings,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class FixedContractLvsqDiagnosticConfig:
    """Numerical specification for the one-week fixed-contract audit."""

    calendar_business_days: int = 5
    annualized_log_volatility: float = 0.10
    sensitivity_annualized_log_volatilities: tuple[float, ...] = (0.05, 0.20)
    substeps: tuple[int, ...] = (1, 2, 5, 10)
    brownian_base_steps: int = 10
    num_monte_carlo_paths: int = 2048
    num_monte_carlo_states: int = 21
    path_batch_size: int = 512
    analytic_state_batch_size: int = 128
    z_resolution_standard_error_floor_bps_spot: float = 1.0e-8
    seed: int = 20260907

    def __post_init__(self) -> None:
        verify_inp(
            self.calendar_business_days > 0,
            lambda: "calendar_business_days must be positive",
        )
        verify_inp(
            np.isfinite(self.annualized_log_volatility)
            and self.annualized_log_volatility > 0.0,
            lambda: "annualized_log_volatility must be finite and positive",
        )
        verify_inp(
            all(
                np.isfinite(value) and value > 0.0
                for value in self.sensitivity_annualized_log_volatilities
            ),
            lambda: "sensitivity volatilities must be finite and positive",
        )
        verify_inp(
            self.substeps
            and all(value > 0 for value in self.substeps)
            and all(self.brownian_base_steps % value == 0 for value in self.substeps),
            lambda: "every substep count must divide brownian_base_steps",
        )
        verify_inp(
            self.num_monte_carlo_paths >= 2
            and self.num_monte_carlo_paths % 2 == 0,
            lambda: "num_monte_carlo_paths must be an even integer of at least two",
        )
        verify_inp(
            0 < self.path_batch_size <= self.num_monte_carlo_paths
            and self.num_monte_carlo_paths % self.path_batch_size == 0
            and self.path_batch_size % 2 == 0,
            lambda: (
                "path_batch_size must be even and divide num_monte_carlo_paths"
            ),
        )
        verify_inp(
            self.num_monte_carlo_states > 0 and self.analytic_state_batch_size > 0,
            lambda: "state counts and analytic_state_batch_size must be positive",
        )
        verify_inp(
            np.isfinite(self.z_resolution_standard_error_floor_bps_spot)
            and self.z_resolution_standard_error_floor_bps_spot > 0.0,
            lambda: "z-resolution standard-error floor must be finite and positive",
        )


class PricingCapHitError(RuntimeError):
    """A simulated raw DLV crossed the pricing-only cap."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--paths", type=int, default=2048)
    parser.add_argument("--states", type=int, default=21)
    parser.add_argument("--path-batch-size", type=int, default=512)
    parser.add_argument("--analytic-state-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser


def _maximum(current: float, value: torch.Tensor) -> float:
    return max(current, float(value.detach().cpu()))


def _minimum(current: float, value: torch.Tensor) -> float:
    return min(current, float(value.detach().cpu()))


def _analytic_audit(
    *,
    decoder: Any,
    raw_states: torch.Tensor,
    loadings: torch.Tensor,
    state_batch_size: int,
) -> dict[str, float]:
    diagnostics = {
        "maximum_volatility_equation_residual": 0.0,
        "maximum_drift_equation_residual": 0.0,
        "maximum_density_diffusion_mass_residual": 0.0,
        "maximum_density_diffusion_forward_residual": 0.0,
        "minimum_density": float("inf"),
        "maximum_absolute_relative_density_diffusion": 0.0,
        "minimum_raw_cap_margin": float("inf"),
    }
    for start in range(0, len(raw_states), state_batch_size):
        stop = min(start + state_batch_size, len(raw_states))
        result = fixed_contract_lvsq_drift(
            decoder,
            raw_states[start:stop],
            raw_log_loadings=loadings,
        )
        diagnostics["maximum_volatility_equation_residual"] = _maximum(
            diagnostics["maximum_volatility_equation_residual"],
            result.maximum_volatility_equation_residual,
        )
        diagnostics["maximum_drift_equation_residual"] = _maximum(
            diagnostics["maximum_drift_equation_residual"],
            result.maximum_drift_equation_residual,
        )
        diagnostics["maximum_density_diffusion_mass_residual"] = _maximum(
            diagnostics["maximum_density_diffusion_mass_residual"],
            result.maximum_density_diffusion_mass_residual,
        )
        diagnostics["maximum_density_diffusion_forward_residual"] = _maximum(
            diagnostics["maximum_density_diffusion_forward_residual"],
            result.maximum_density_diffusion_forward_residual,
        )
        diagnostics["minimum_density"] = _minimum(
            diagnostics["minimum_density"], result.minimum_density
        )
        diagnostics["maximum_absolute_relative_density_diffusion"] = _maximum(
            diagnostics["maximum_absolute_relative_density_diffusion"],
            result.maximum_absolute_relative_density_diffusion,
        )
        diagnostics["minimum_raw_cap_margin"] = _minimum(
            diagnostics["minimum_raw_cap_margin"], result.minimum_raw_cap_margin
        )
    return diagnostics


def _decode_surface_components(
    decoder: Any,
    raw_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = bound_native_raw(raw_states, geometry=decoder.geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    return surface.densities, surface.support_strikes, surface.backbone_total_vols


def _black_call_kernel(
    support: torch.Tensor,
    strikes: torch.Tensor,
    backbone_total_vols: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen conditional Black-call kernel at model expiries."""

    spot = support[..., None]
    strike = strikes[None, None, None, :]
    sqrt_variance = backbone_total_vols[..., None, None]
    safe_sqrt_variance = torch.clamp(sqrt_variance, min=1.0e-10)
    d1 = (
        torch.log(spot / strike) + 0.5 * safe_sqrt_variance.square()
    ) / safe_sqrt_variance
    d2 = d1 - safe_sqrt_variance
    black = spot * torch.special.ndtr(d1) - strike * torch.special.ndtr(d2)
    return torch.where(sqrt_variance > 0.0, black, torch.relu(spot - strike))


def _streaming_statistics(
    total: torch.Tensor,
    total_square: torch.Tensor,
    *,
    num_paths: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = total / num_paths
    centered_sum_square = torch.clamp(
        total_square - total.square() / num_paths,
        min=0.0,
    )
    sample_variance = centered_sum_square / (num_paths - 1)
    standard_error = torch.sqrt(sample_variance / num_paths)
    z = torch.where(
        standard_error > torch.finfo(mean.dtype).tiny,
        mean / standard_error,
        torch.where(mean == 0.0, torch.zeros_like(mean), torch.full_like(mean, float("inf"))),
    )
    return mean, standard_error, z


def _z_distribution_summary(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.double().reshape(-1)
    absolute = torch.abs(flat)
    quantiles = torch.quantile(
        absolute,
        torch.tensor((0.50, 0.90, 0.95, 0.99), dtype=flat.dtype, device=flat.device),
    )
    return {
        "num_coordinates": len(flat),
        "mean_z": float(torch.mean(flat).cpu()),
        "root_mean_square_z": float(torch.sqrt(torch.mean(flat.square())).cpu()),
        "mean_absolute_z": float(torch.mean(absolute).cpu()),
        "median_absolute_z": float(quantiles[0].cpu()),
        "p90_absolute_z": float(quantiles[1].cpu()),
        "p95_absolute_z": float(quantiles[2].cpu()),
        "p99_absolute_z": float(quantiles[3].cpu()),
        "maximum_absolute_z": float(torch.max(absolute).cpu()),
        "num_absolute_z_above_2": int(torch.sum(absolute > 2.0).cpu()),
        "num_absolute_z_above_3": int(torch.sum(absolute > 3.0).cpu()),
        "num_absolute_z_above_5": int(torch.sum(absolute > 5.0).cpu()),
    }


def _option_coordinate_details(
    values: torch.Tensor,
    *,
    option_mean: torch.Tensor,
    option_standard_error: torch.Tensor,
    option_z: torch.Tensor,
    strikes: torch.Tensor,
    model_business_dtes: tuple[float, ...],
) -> dict[str, float | int | str]:
    flat_index = int(torch.argmax(torch.abs(values)).cpu())
    state_position, dynamic_expiry_index, strike_index = np.unravel_index(
        flat_index,
        tuple(values.shape),
    )
    strike = float(strikes[strike_index].cpu())
    return {
        "selected_state_position": int(state_position),
        "model_expiry_index": int(dynamic_expiry_index + 1),
        "model_business_dte": float(model_business_dtes[dynamic_expiry_index + 1]),
        "strike_index": int(strike_index),
        "relative_strike": strike,
        "option_kind": "call" if strike <= 1.0 else "put",
        "mean_bps_spot": float(
            (10_000.0 * option_mean[state_position, dynamic_expiry_index, strike_index]).cpu()
        ),
        "sample_standard_error_bps_spot": float(
            (
                10_000.0
                * option_standard_error[state_position, dynamic_expiry_index, strike_index]
            ).cpu()
        ),
        "mean_z": float(option_z[state_position, dynamic_expiry_index, strike_index].cpu()),
    }


def _monte_carlo_audit(
    *,
    decoder: Any,
    raw_states: torch.Tensor,
    loadings: torch.Tensor,
    base_normals: torch.Tensor,
    num_substeps: int,
    path_batch_size: int,
    calendar_horizon_y: float,
    z_resolution_standard_error_floor_bps_spot: float,
) -> dict[str, object]:
    geometry = decoder.geometry
    num_states = len(raw_states)
    num_paths = len(base_normals)
    num_independent_pairs = num_paths // 2
    num_factors = loadings.shape[-1]
    group_size = base_normals.shape[1] // num_substeps
    grouped_normals = (
        base_normals.reshape(num_paths, num_substeps, group_size, num_factors).sum(dim=2)
        / sqrt(group_size)
    )
    with torch.no_grad():
        initial_densities, support, backbone_total_vols = _decode_surface_components(
            decoder, raw_states
        )
    strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=raw_states.dtype, device=raw_states.device)
    )
    call_payoff = _black_call_kernel(support, strikes, backbone_total_vols)
    put_payoff = call_payoff - support[..., None] + strikes
    payoff = torch.where((strikes <= 1.0)[None, None, None, :], call_payoff, put_payoff)
    initial_options = torch.sum(initial_densities[..., None] * payoff, dim=2)

    density_shape = initial_densities[:, 1:].shape
    option_shape = initial_options[:, 1:].shape
    density_sum = torch.zeros(density_shape, dtype=raw_states.dtype, device=raw_states.device)
    density_square_sum = torch.zeros_like(density_sum)
    option_sum = torch.zeros(option_shape, dtype=raw_states.dtype, device=raw_states.device)
    option_square_sum = torch.zeros_like(option_sum)
    maximum_volatility_residual = 0.0
    maximum_drift_residual = 0.0
    maximum_mass_residual = 0.0
    maximum_forward_residual = 0.0
    minimum_density = float("inf")
    maximum_relative_density_diffusion = 0.0
    minimum_cap_margin = float("inf")
    maximum_frozen_coordinate_change = 0.0
    step_y = calendar_horizon_y / num_substeps

    for path_start in range(0, num_paths, path_batch_size):
        path_stop = path_start + path_batch_size
        current_batch_size = path_stop - path_start
        current = (
            raw_states[:, None, :]
            .expand(num_states, current_batch_size, -1)
            .reshape(num_states * current_batch_size, -1)
            .clone()
        )
        normals = (
            grouped_normals[path_start:path_stop][None]
            .expand(num_states, -1, -1, -1)
            .reshape(num_states * current_batch_size, num_substeps, num_factors)
        )
        for step_index in range(num_substeps):
            restriction = fixed_contract_lvsq_drift(
                decoder,
                current,
                raw_log_loadings=loadings,
            )
            maximum_volatility_residual = _maximum(
                maximum_volatility_residual,
                restriction.maximum_volatility_equation_residual,
            )
            maximum_drift_residual = _maximum(
                maximum_drift_residual,
                restriction.maximum_drift_equation_residual,
            )
            maximum_mass_residual = _maximum(
                maximum_mass_residual,
                restriction.maximum_density_diffusion_mass_residual,
            )
            maximum_forward_residual = _maximum(
                maximum_forward_residual,
                restriction.maximum_density_diffusion_forward_residual,
            )
            minimum_density = _minimum(minimum_density, restriction.minimum_density)
            maximum_relative_density_diffusion = _maximum(
                maximum_relative_density_diffusion,
                restriction.maximum_absolute_relative_density_diffusion,
            )
            minimum_cap_margin = _minimum(
                minimum_cap_margin, restriction.minimum_raw_cap_margin
            )
            current = fixed_contract_lvsq_euler_step(
                restriction,
                current,
                calendar_time_increment_y=step_y,
                standard_normals=normals[:, step_index],
            )
            stepped_blocks = current.reshape(
                num_states,
                current_batch_size,
                geometry.num_model_expiries,
                geometry.expiry_block_state_size,
            )
            stepped_cap_margin = torch.as_tensor(
                geometry.maximum_log_dlvsq,
                dtype=current.dtype,
                device=current.device,
            ) - stepped_blocks[:, :, 1:, : geometry.num_inner_strikes]
            minimum_cap_margin = _minimum(
                minimum_cap_margin, torch.min(stepped_cap_margin)
            )
            if not bool(torch.all(stepped_cap_margin > 0.0)):
                raise PricingCapHitError(
                    "a simulated DLV reached the pricing cap; the finite-step "
                    "diffusion audit is invalid"
                )

        current_blocks = current.reshape(
            num_states,
            current_batch_size,
            geometry.num_model_expiries,
            geometry.expiry_block_state_size,
        )
        final_cap_margin = torch.as_tensor(
            geometry.maximum_log_dlvsq,
            dtype=current.dtype,
            device=current.device,
        ) - current_blocks[:, :, 1:, : geometry.num_inner_strikes]
        minimum_cap_margin = _minimum(minimum_cap_margin, torch.min(final_cap_margin))
        assert bool(torch.all(final_cap_margin > 0.0))
        initial_blocks = raw_states.reshape(
            num_states,
            1,
            geometry.num_model_expiries,
            geometry.expiry_block_state_size,
        )
        frozen_changes = torch.cat(
            (
                (current_blocks[:, :, :1] - initial_blocks[:, :, :1]).reshape(-1),
                (
                    current_blocks[..., geometry.num_inner_strikes :]
                    - initial_blocks[..., geometry.num_inner_strikes :]
                ).reshape(-1),
            )
        )
        maximum_frozen_coordinate_change = _maximum(
            maximum_frozen_coordinate_change, torch.max(torch.abs(frozen_changes))
        )
        final_densities, final_support, final_backbone_total_vols = (
            _decode_surface_components(decoder, current)
        )
        if not bool(torch.all(final_support == support[:, None].expand(-1, current_batch_size, -1, -1).reshape_as(final_support))):
            raise RuntimeError("fixed forward volatilities did not preserve cash support")
        if not bool(
            torch.all(
                final_backbone_total_vols
                == backbone_total_vols[:, None]
                .expand(-1, current_batch_size, -1)
                .reshape_as(final_backbone_total_vols)
            )
        ):
            raise RuntimeError("frozen raw coordinates did not preserve backbone volatility")
        final_densities = final_densities.reshape(
            num_states,
            current_batch_size,
            geometry.num_model_expiries,
            -1,
        )
        density_change = final_densities[:, :, 1:] - initial_densities[:, None, 1:]
        final_options = torch.sum(
            final_densities[..., None] * payoff[:, None],
            dim=3,
        )
        option_change = final_options[:, :, 1:] - initial_options[:, None, 1:]
        density_pair_means = density_change.reshape(
            num_states,
            current_batch_size // 2,
            2,
            *density_change.shape[2:],
        ).mean(dim=2)
        option_pair_means = option_change.reshape(
            num_states,
            current_batch_size // 2,
            2,
            *option_change.shape[2:],
        ).mean(dim=2)
        density_sum += torch.sum(density_pair_means, dim=1)
        density_square_sum += torch.sum(density_pair_means.square(), dim=1)
        option_sum += torch.sum(option_pair_means, dim=1)
        option_square_sum += torch.sum(option_pair_means.square(), dim=1)

    density_mean, density_standard_error, density_z = _streaming_statistics(
        density_sum,
        density_square_sum,
        num_paths=num_independent_pairs,
    )
    option_mean, option_standard_error, option_z = _streaming_statistics(
        option_sum,
        option_square_sum,
        num_paths=num_independent_pairs,
    )
    worst_absolute_mean = _option_coordinate_details(
        option_mean,
        option_mean=option_mean,
        option_standard_error=option_standard_error,
        option_z=option_z,
        strikes=strikes,
        model_business_dtes=geometry.model_business_dtes,
    )
    worst_absolute_z = _option_coordinate_details(
        option_z,
        option_mean=option_mean,
        option_standard_error=option_standard_error,
        option_z=option_z,
        strikes=strikes,
        model_business_dtes=geometry.model_business_dtes,
    )
    option_z_by_expiry = [
        {
            "model_expiry_index": expiry_index,
            "model_business_dte": float(geometry.model_business_dtes[expiry_index]),
            "z_distribution": _z_distribution_summary(option_z[:, expiry_index - 1]),
            "resolved_z_distribution": _z_distribution_summary(
                option_z[:, expiry_index - 1][
                    10_000.0 * option_standard_error[:, expiry_index - 1]
                    >= z_resolution_standard_error_floor_bps_spot
                ]
            ),
        }
        for expiry_index in range(1, geometry.num_model_expiries)
    ]
    resolved_z_mask = (
        10_000.0 * option_standard_error
        >= z_resolution_standard_error_floor_bps_spot
    )
    return {
        "num_substeps": num_substeps,
        "num_paths": num_paths,
        "num_independent_antithetic_pairs": num_independent_pairs,
        "calendar_step_years": step_y,
        "maximum_absolute_density_mean": float(torch.max(torch.abs(density_mean)).cpu()),
        "root_mean_square_density_mean": float(torch.sqrt(torch.mean(density_mean.square())).cpu()),
        "maximum_absolute_density_z": float(torch.max(torch.abs(density_z)).cpu()),
        "maximum_density_standard_error": float(torch.max(density_standard_error).cpu()),
        "maximum_absolute_itm_option_mean_bps_spot": float(
            (10_000.0 * torch.max(torch.abs(option_mean))).cpu()
        ),
        "root_mean_square_itm_option_mean_bps_spot": float(
            (10_000.0 * torch.sqrt(torch.mean(option_mean.square()))).cpu()
        ),
        "maximum_absolute_itm_option_z": float(torch.max(torch.abs(option_z)).cpu()),
        "maximum_itm_option_standard_error_bps_spot": float(
            (10_000.0 * torch.max(option_standard_error)).cpu()
        ),
        "worst_absolute_itm_option_mean": worst_absolute_mean,
        "worst_absolute_itm_option_z": worst_absolute_z,
        "itm_option_z_distribution": _z_distribution_summary(option_z),
        "itm_option_resolved_z_distribution": _z_distribution_summary(
            option_z[resolved_z_mask]
        ),
        "z_resolution_standard_error_floor_bps_spot": (
            z_resolution_standard_error_floor_bps_spot
        ),
        "num_numerically_unresolved_z_coordinates": int(
            torch.sum(~resolved_z_mask).cpu()
        ),
        "itm_option_z_distribution_by_expiry": option_z_by_expiry,
        "maximum_volatility_equation_residual": maximum_volatility_residual,
        "maximum_drift_equation_residual": maximum_drift_residual,
        "maximum_density_diffusion_mass_residual": maximum_mass_residual,
        "maximum_density_diffusion_forward_residual": maximum_forward_residual,
        "minimum_density": minimum_density,
        "maximum_absolute_relative_density_diffusion": maximum_relative_density_diffusion,
        "minimum_raw_cap_margin": minimum_cap_margin,
        "maximum_frozen_coordinate_change": maximum_frozen_coordinate_change,
        "all_values_finite": bool(
            torch.all(torch.isfinite(density_mean))
            and torch.all(torch.isfinite(density_z))
            and torch.all(torch.isfinite(option_mean))
            and torch.all(torch.isfinite(option_z))
        ),
    }


def run(arguments: argparse.Namespace) -> None:
    """Run the all-state identity audit and nested finite-step experiment."""

    config = FixedContractLvsqDiagnosticConfig(
        num_monte_carlo_paths=arguments.paths,
        num_monte_carlo_states=arguments.states,
        path_batch_size=arguments.path_batch_size,
        analytic_state_batch_size=arguments.analytic_state_batch_size,
        seed=arguments.seed,
    )
    torch.manual_seed(config.seed)
    env = BaseEnvironment(precision=arguments.precision, device=arguments.device, debug=False)
    model, date_values, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    if config.num_monte_carlo_states > len(latent_states):
        raise ValueError("requested more Monte Carlo states than the fitted history contains")
    with torch.no_grad():
        raw_states = model.raw_states_from_states(latent_states)
        _, support, _ = _decode_surface_components(model.sanos_decoder, raw_states[:1])
    loadings = smooth_lvsq_log_loadings(
        support[0, 0],
        num_model_expiries=model.config.geometry.num_model_expiries,
        annualized_log_volatility=config.annualized_log_volatility,
    )

    analytic_by_volatility: dict[str, dict[str, float]] = {}
    volatility_values = (
        config.annualized_log_volatility,
        *config.sensitivity_annualized_log_volatilities,
    )
    for volatility in volatility_values:
        current_loadings = loadings * (volatility / config.annualized_log_volatility)
        audit = _analytic_audit(
            decoder=model.sanos_decoder,
            raw_states=raw_states,
            loadings=current_loadings,
            state_batch_size=config.analytic_state_batch_size,
        )
        analytic_by_volatility[f"{volatility:.6g}"] = audit
        print(
            f"analytic nu={volatility:.3f} eq_vol={audit['maximum_volatility_equation_residual']:.3e} "
            f"eq_drift={audit['maximum_drift_equation_residual']:.3e} "
            f"max_eta={audit['maximum_absolute_relative_density_diffusion']:.3e}",
            flush=True,
        )

    state_indices = np.linspace(
        0,
        len(raw_states) - 1,
        config.num_monte_carlo_states,
        dtype=np.int64,
    )
    raw_blocks = raw_states.reshape(
        len(raw_states),
        model.config.geometry.num_model_expiries,
        model.config.geometry.expiry_block_state_size,
    )
    state_cap_margins = torch.as_tensor(
        model.config.geometry.maximum_log_dlvsq,
        dtype=raw_states.dtype,
        device=raw_states.device,
    ) - torch.amax(
        raw_blocks[:, 1:, : model.config.geometry.num_inner_strikes],
        dim=(1, 2),
    )
    worst_cap_state_index = int(torch.argmin(state_cap_margins).cpu())
    if worst_cap_state_index not in state_indices:
        state_indices[len(state_indices) // 2] = worst_cap_state_index
        state_indices.sort()
    index_tensor = torch.as_tensor(state_indices, dtype=torch.long, device=env.device)
    monte_carlo_states = raw_states[index_tensor]
    half_normals = torch.randn(
        (
            config.num_monte_carlo_paths // 2,
            config.brownian_base_steps,
            loadings.shape[-1],
        ),
        dtype=env.fdtype,
        device=env.device,
    )
    base_normals = torch.stack((half_normals, -half_normals), dim=1).reshape(
        config.num_monte_carlo_paths,
        config.brownian_base_steps,
        loadings.shape[-1],
    )
    monte_carlo: list[dict[str, object]] = []
    for num_substeps in config.substeps:
        try:
            audit = _monte_carlo_audit(
                decoder=model.sanos_decoder,
                raw_states=monte_carlo_states,
                loadings=loadings,
                base_normals=base_normals,
                num_substeps=num_substeps,
                path_batch_size=config.path_batch_size,
                calendar_horizon_y=config.calendar_business_days / DAYS_PER_YEAR,
                z_resolution_standard_error_floor_bps_spot=(
                    config.z_resolution_standard_error_floor_bps_spot
                ),
            )
            audit["valid"] = True
            for coordinate_key in (
                "worst_absolute_itm_option_mean",
                "worst_absolute_itm_option_z",
            ):
                coordinate = audit[coordinate_key]
                assert isinstance(coordinate, dict)
                selected_position = int(coordinate["selected_state_position"])
                source_index = int(state_indices[selected_position])
                coordinate["source_state_index"] = source_index
                coordinate["source_state_date"] = date_values[source_index].isoformat()
        except PricingCapHitError as error:
            audit = {
                "num_substeps": num_substeps,
                "num_paths": config.num_monte_carlo_paths,
                "valid": False,
                "failure": str(error),
            }
        monte_carlo.append(audit)
        if audit["valid"]:
            print(
                f"MC steps={num_substeps} option_mean_max="
                f"{audit['maximum_absolute_itm_option_mean_bps_spot']:.4f}bp "
                f"option_z_max={audit['maximum_absolute_itm_option_z']:.3f} "
                f"cap_margin={audit['minimum_raw_cap_margin']:.3e}",
                flush=True,
            )
        else:
            print(f"MC steps={num_substeps} invalid: {audit['failure']}", flush=True)

    primary_analytic = analytic_by_volatility[f"{config.annualized_log_volatility:.6g}"]
    local_identity_passed = (
        primary_analytic["maximum_volatility_equation_residual"] < 1.0e-9
        and primary_analytic["maximum_drift_equation_residual"] < 1.0e-9
        and primary_analytic["maximum_density_diffusion_mass_residual"] < 1.0e-9
        and primary_analytic["maximum_density_diffusion_forward_residual"] < 1.0e-9
        and primary_analytic["minimum_density"] > -1.0e-12
        and primary_analytic["minimum_raw_cap_margin"] > 0.0
    )
    finest = monte_carlo[-1]
    monte_carlo_completed = all(bool(value["valid"]) for value in monte_carlo)
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "fixed_contract_lvsq_diagnostic",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "local_identity_passed" if local_identity_passed else "local_identity_failed"
        ),
        "interpretation": {
            "proved": (
                "The PDF drift and volatility identities hold on the fitted fixed-cash "
                "four-expiry surfaces for the specified loading field."
            ),
            "not_proved": (
                "This is not a rolling one-week Q law: the front density is frozen, no "
                "spot is drawn, absolute expiries do not roll, and no new far expiry is added."
            ),
            "forward_volatilities": "frozen bit-for-bit in the original raw state",
            "backbone_multipliers": "frozen bit-for-bit in the original raw state",
        },
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "num_source_states": len(raw_states),
        "native_state_size": model.config.geometry.native_state_size,
        "num_inner_strikes": model.config.geometry.num_inner_strikes,
        "num_model_expiries": model.config.geometry.num_model_expiries,
        "model_business_dtes": list(model.config.geometry.model_business_dtes),
        "sources": sources,
        "analytic_all_states_by_annualized_log_volatility": analytic_by_volatility,
        "monte_carlo_state_indices": state_indices.tolist(),
        "monte_carlo_state_dates": [date_values[index].isoformat() for index in state_indices],
        "worst_initial_cap_margin_state_index": worst_cap_state_index,
        "worst_initial_cap_margin_state_date": date_values[
            worst_cap_state_index
        ].isoformat(),
        "nested_antithetic_monte_carlo": monte_carlo,
        "finest_step_summary": finest,
        "local_identity_passed": local_identity_passed,
        "monte_carlo_completed_without_pricing_cap": monte_carlo_completed,
    }
    write_json_atomic(report, arguments.output)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    """Command-line entry point."""

    run(_parser().parse_args())


if __name__ == "__main__":
    main()
