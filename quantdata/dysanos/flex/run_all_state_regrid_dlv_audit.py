"""Audit repeated exact-density DLV rolls across every certified initial state."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.fixed_grid_dlv import (
    build_fixed_grid_dlv_kernel,
    exact_fixed_grid_dlv_diagnostics,
    make_fixed_grid_dlv_randoms,
    sample_fixed_grid_dlv_paths,
    set_constant_backbone_volatility,
    strip_fixed_grid_black_backbone,
)
from quantdata.dysanos.flex.floating_grid_dlv import (
    FloatingGridDlvSurface,
    build_floating_grid_dlv_kernel,
    floating_surface_from_regrid,
    price_floating_surface_options,
    replace_floating_far_expiry,
    roll_floating_grid_dlv_kernel,
    smooth_floating_log_loadings,
)
from quantdata.dysanos.flex.last_expiry_ar1 import (
    TorchLastExpiryAr1,
    fit_last_expiry_ar1,
    interpolate_ar1_raw_curves,
    make_antithetic_ar1_normals,
    physical_dlvsq_from_raw,
    torch_last_expiry_ar1,
)
from quantdata.dysanos.flex.lvsq_drift import smooth_lvsq_log_loadings
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_multistep_regrid_dlv_simulator import (
    _factor_covariance_audit,
    _factor_scores,
    _synchronize,
    _wald_test,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class AllStateRegridDlvAuditConfig:
    """Balanced all-state path allocation and repeated weekly-roll audit."""

    num_paths: int = 10_000
    num_steps: int = 16
    state_batch_size: int = 96
    calendar_business_days: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    bs_backbone_annualized_volatility: float = 0.10
    num_option_strikes: int = 11
    correlation_eigenvalue_cutoff: float = 0.01
    last_expiry_ar1_rank: int = 25
    num_trajectories: int = 4
    seed: int = 2_026_091_2

    def __post_init__(self) -> None:
        verify_inp(
            self.num_paths >= 4 and self.num_paths % 2 == 0,
            lambda: "num_paths must be even and at least four",
        )
        verify_inp(self.num_steps > 0, lambda: "num_steps must be positive")
        verify_inp(
            self.state_batch_size > 0,
            lambda: "state_batch_size must be positive",
        )
        verify_inp(
            self.num_option_strikes >= 3,
            lambda: "num_option_strikes must be at least three",
        )
        verify_inp(
            self.num_trajectories >= 0,
            lambda: "num_trajectories must be nonnegative",
        )
        verify_inp(
            self.last_expiry_ar1_rank > 0,
            lambda: "last_expiry_ar1_rank must be positive",
        )
        verify_inp(
            self.dlvsq_annualized_log_volatility > 0.0
            and self.bs_backbone_annualized_volatility >= 0.0
            and self.correlation_eigenvalue_cutoff > 0.0,
            lambda: (
                "DLV volatility and correlation cutoff must be positive, and "
                "Black backbone volatility must be nonnegative"
            ),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--num-paths", type=int, default=10_000)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--state-batch-size", type=int, default=96)
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2_026_091_2)
    parser.add_argument(
        "--bs-backbone-annualized-volatility",
        type=float,
        default=0.10,
        help="Set to zero for a pure discrete-DLV-factor (X-only) audit.",
    )
    return parser


def _balanced_state_groups(
    *,
    num_states: int,
    num_pairs: int,
    generator: np.random.Generator,
) -> list[tuple[np.ndarray, int]]:
    """Allocate pair counts as evenly as possible while covering every state."""

    if num_pairs < num_states:
        raise ValueError("num_paths must provide one antithetic pair per state")
    pairs_per_state, extra = divmod(num_pairs, num_states)
    permutation = generator.permutation(num_states)
    groups: list[tuple[np.ndarray, int]] = []
    if extra:
        groups.append((permutation[:extra], pairs_per_state + 1))
    if extra < num_states:
        groups.append((permutation[extra:], pairs_per_state))
    return groups


def _coordinate_z(observations: np.ndarray) -> np.ndarray:
    standard_deviation = np.std(observations, axis=0, ddof=1)
    maximum = float(np.max(standard_deviation))
    active = standard_deviation > max(1.0e-14, 1.0e-10 * maximum)
    result = np.full(observations.shape[1], np.nan)
    result[active] = (
        np.mean(observations, axis=0)[active]
        / standard_deviation[active]
        * np.sqrt(len(observations))
    )
    return result


def _wald_test_with_reference_covariance(
    covariance_observations: np.ndarray,
    mean_observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, float | int]:
    """Test one sample's mean in modes selected from an independent sample."""

    standard_deviation = np.std(covariance_observations, axis=0, ddof=1)
    active = standard_deviation > max(
        1.0e-14,
        1.0e-10 * float(np.max(standard_deviation)),
    )
    covariance_values = (
        covariance_observations[:, active] / standard_deviation[active]
    )
    mean_values = mean_observations[:, active] / standard_deviation[active]
    correlation = np.atleast_2d(
        np.cov(covariance_values, rowvar=False, ddof=1)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    retained = eigenvalues > correlation_eigenvalue_cutoff
    if not bool(np.any(retained)):
        raise RuntimeError("independent-covariance audit retained no directions")
    projected_mean = eigenvectors[:, retained].T @ np.mean(mean_values, axis=0)
    statistic = float(
        len(mean_values)
        * np.sum(np.square(projected_mean) / eigenvalues[retained])
    )
    degrees_of_freedom = int(np.sum(retained))
    return {
        "num_covariance_observations": len(covariance_observations),
        "num_mean_observations": len(mean_observations),
        "degrees_of_freedom": degrees_of_freedom,
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
        "minimum_retained_correlation_eigenvalue": float(
            np.min(eigenvalues[retained])
        ),
    }


def _split_sample_wald_tests(
    observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, object]:
    """Cross-check a mean using covariance modes estimated on the other half."""

    even = observations[0::2]
    odd = observations[1::2]
    if min(len(even), len(odd)) < 2:
        raise ValueError("split-sample inference needs at least four observations")
    return {
        "even_covariance_odd_mean": _wald_test_with_reference_covariance(
            even,
            odd,
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        ),
        "odd_covariance_even_mean": _wald_test_with_reference_covariance(
            odd,
            even,
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        ),
    }


def _pair_matrix_audit(
    observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
    first_coordinate_is_spot: bool,
) -> dict[str, object]:
    """Report zero-mean coordinate and covariance-aware tests over pair means."""

    z = _coordinate_z(observations)
    mean = np.mean(observations, axis=0)
    option_start = 1 if first_coordinate_is_spot else 0
    result: dict[str, object] = {
        "num_independent_antithetic_pairs": len(observations),
        "num_moments": observations.shape[1],
        "maximum_absolute_coordinate_z": float(np.nanmax(np.abs(z))),
        "root_mean_square_coordinate_z": float(np.sqrt(np.nanmean(np.square(z)))),
        "num_absolute_coordinate_z_above_2": int(np.sum(np.abs(z) > 2.0)),
        "coordinate_z": z.tolist(),
        "covariance_aware_omnibus": _wald_test(
            observations,
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        ),
        "correlation_cutoff_sensitivity": {
            f"{cutoff:.3f}": _wald_test(
                observations,
                correlation_eigenvalue_cutoff=cutoff,
            )
            for cutoff in (0.005, 0.01, 0.02, 0.05)
        },
        "independent_covariance_split": _split_sample_wald_tests(
            observations,
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        ),
    }
    if first_coordinate_is_spot:
        result.update(
            {
                "spot_mean_bps": float(10_000.0 * mean[0]),
                "spot_z": float(z[0]),
            }
        )
    option_mean = mean[option_start:]
    option_z = z[option_start:]
    if len(option_mean):
        result.update(
            {
                "maximum_absolute_option_mean_bps_spot": float(
                    10_000.0 * np.max(np.abs(option_mean))
                ),
                "maximum_absolute_option_z": float(np.nanmax(np.abs(option_z))),
            }
        )
    return result


def _per_state_z_audit(
    observations: np.ndarray,
    origin_state_indices: np.ndarray,
    dates: tuple[object, ...],
) -> dict[str, object]:
    """Summarize coordinate z-scores separately for every initial state."""

    by_state: list[dict[str, object]] = []
    maxima: list[float] = []
    root_mean_squares: list[float] = []
    for state_index in range(len(dates)):
        state_observations = observations[origin_state_indices == state_index]
        if len(state_observations) < 2:
            by_state.append(
                {
                    "source_state_index": state_index,
                    "source_state_date": dates[state_index].isoformat(),
                    "num_independent_antithetic_pairs": len(state_observations),
                    "maximum_absolute_coordinate_z": None,
                    "root_mean_square_coordinate_z": None,
                    "num_absolute_coordinate_z_above_2": None,
                    "reason": "at least two independent pairs are required",
                }
            )
            continue
        z = _coordinate_z(state_observations)
        maximum = float(np.nanmax(np.abs(z)))
        root_mean_square = float(np.sqrt(np.nanmean(np.square(z))))
        maxima.append(maximum)
        root_mean_squares.append(root_mean_square)
        by_state.append(
            {
                "source_state_index": state_index,
                "source_state_date": dates[state_index].isoformat(),
                "num_independent_antithetic_pairs": len(state_observations),
                "maximum_absolute_coordinate_z": maximum,
                "root_mean_square_coordinate_z": root_mean_square,
                "num_absolute_coordinate_z_above_2": int(np.sum(np.abs(z) > 2.0)),
            }
        )
    return {
        "interpretation": (
            "Each state uses only its own antithetic pairs. The maximum is a "
            "multiple-comparison diagnostic; exact density towers provide the "
            "state-by-state structural martingale evidence."
        ),
        "num_testable_states": len(maxima),
        "maximum_over_states": float(np.max(maxima)) if maxima else None,
        "p95_state_maximum": float(np.quantile(maxima, 0.95)) if maxima else None,
        "median_state_maximum": float(np.median(maxima)) if maxima else None,
        "root_mean_square_over_states": float(
            np.sqrt(np.mean(np.square(root_mean_squares)))
        ) if root_mean_squares else None,
        "num_states_with_any_absolute_z_above_2": (
            int(np.sum(np.asarray(maxima) > 2.0)) if maxima else None
        ),
        "by_state": by_state,
    }


def _pair_mean_by_state(value: torch.Tensor, *, num_states: int, num_pairs: int) -> np.ndarray:
    trailing = value.shape[2:]
    paired = value.reshape(num_states, num_pairs, 2, *trailing).mean(dim=2)
    return paired.reshape(num_states * num_pairs, *trailing).cpu().double().numpy()


def _first_member_by_state(
    value: torch.Tensor,
    *,
    num_states: int,
    num_pairs: int,
) -> np.ndarray:
    """Return one independent, correctly distributed member from every pair."""

    trailing = value.shape[2:]
    paired = value.reshape(num_states, num_pairs, 2, *trailing)
    return paired[:, :, 0].reshape(
        num_states * num_pairs,
        *trailing,
    ).cpu().double().numpy()


def _first_member_matrix_audit(
    observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, object]:
    """Audit one independent member per antithetic pair without pair averaging."""

    result = _pair_matrix_audit(
        observations,
        correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        first_coordinate_is_spot=False,
    )
    result["num_independent_first_members"] = result.pop(
        "num_independent_antithetic_pairs"
    )
    result["sampling"] = (
        "first member of every antithetic pair; members are independent across pairs"
    )
    return result


def _selected_branch_surface(
    surface: FloatingGridDlvSurface,
    *,
    densities_by_branch: torch.Tensor,
    dlvsqs_by_branch: torch.Tensor,
    branch_indices: torch.Tensor,
) -> FloatingGridDlvSurface:
    """Gather selected fixed-support surface branches and flatten path dimensions."""

    batch, expiries, support_size = surface.densities.shape
    if (
        branch_indices.ndim not in (1, 2)
        or branch_indices.shape[0] != batch
        or densities_by_branch.shape[:1] != (batch,)
        or densities_by_branch.shape[2:] != (expiries, support_size)
        or dlvsqs_by_branch.shape != densities_by_branch.shape[:-1] + (
            support_size - 2,
        )
    ):
        raise ValueError("selected-branch surface tensors have incompatible shapes")
    batch_index = torch.arange(batch, device=surface.densities.device)
    if branch_indices.ndim == 2:
        batch_index = batch_index[:, None].expand_as(branch_indices)
    selected_densities = densities_by_branch[batch_index, branch_indices]
    selected_dlvsqs = dlvsqs_by_branch[batch_index, branch_indices]
    path_shape = branch_indices.shape[1:]
    support = surface.support_strikes.reshape(
        batch,
        *(1 for _ in path_shape),
        expiries,
        support_size,
    ).expand(batch, *path_shape, expiries, support_size)
    backbone = surface.backbone_total_vols.reshape(
        batch,
        *(1 for _ in path_shape),
        expiries,
    ).expand(batch, *path_shape, expiries)
    flat = branch_indices.numel()
    return FloatingGridDlvSurface(
        support_strikes=support.reshape(flat, expiries, support_size),
        densities=selected_densities.reshape(flat, expiries, support_size),
        dlvsqs=selected_dlvsqs.reshape(flat, expiries, support_size - 2),
        backbone_total_vols=backbone.reshape(flat, expiries),
    )


def _direct_survivor_option_values(
    current_surface: FloatingGridDlvSurface,
    rolled_surface: FloatingGridDlvSurface,
    *,
    current_spot: torch.Tensor,
    gross_spot_return: torch.Tensor,
    revealed_front_nodes: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Price survivors on inherited coordinates before the relative re-grid."""

    batch, expiries, support_size = current_surface.densities.shape
    if (
        rolled_surface.densities.shape != (batch, expiries, support_size)
        or current_spot.shape != (batch,)
        or gross_spot_return.shape != (batch,)
        or revealed_front_nodes.shape != (batch,)
    ):
        raise ValueError("direct survivor valuation tensors have incompatible shapes")
    front_support = current_surface.support_strikes[:, 0]
    latent_front = torch.gather(
        front_support,
        dim=-1,
        index=revealed_front_nodes[:, None],
    ).squeeze(-1)
    # S_{t+h}=S_t*x_front*Y.  Before re-gridding, the inherited support is x,
    # so the component-forward scale is S_t*Y=S_{t+h}/x_front.
    direct_component_spot = current_spot * gross_spot_return / latent_front
    survivors = expiries - 1
    direct_surface = FloatingGridDlvSurface(
        support_strikes=current_surface.support_strikes[:, 1:],
        densities=rolled_surface.densities[:, :survivors],
        dlvsqs=rolled_surface.dlvsqs[:, :survivors],
        backbone_total_vols=rolled_surface.backbone_total_vols[:, :survivors],
    )
    all_calls = torch.ones_like(is_call, dtype=torch.bool)
    calls = price_floating_surface_options(
        direct_surface,
        current_spot=direct_component_spot,
        fixed_cash_strikes=fixed_cash_strikes,
        is_call=all_calls,
    )
    next_spot = current_spot * gross_spot_return
    if fixed_cash_strikes.ndim == 1:
        parity_strike = fixed_cash_strikes.reshape(1, 1, -1)
        option_is_call = is_call.reshape(1, 1, -1)
    else:
        parity_strike = fixed_cash_strikes[:, None]
        option_is_call = is_call[:, None]
    # The inherited conditional density has mean x_front, not one.  Therefore
    # its cash forward is S_t*Y*x_front=S_{t+h}, which is the parity term.
    puts = calls - next_spot[:, None, None] + parity_strike
    return torch.where(option_is_call, calls, puts)


def _intrinsic_values(
    spot: torch.Tensor,
    strikes: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    if strikes.ndim == 1:
        strikes = strikes[None].expand(len(spot), -1)
    if is_call.ndim == 1:
        is_call = is_call[None].expand(len(spot), -1)
    if strikes.shape != is_call.shape or strikes.shape[0] != len(spot):
        raise ValueError("intrinsic-value inputs have incompatible shapes")
    calls = torch.relu(spot[:, None] - strikes)
    puts = torch.relu(strikes - spot[:, None])
    return torch.where(is_call, calls, puts)


def _surface_metrics(
    surface: FloatingGridDlvSurface,
    *,
    current_spot: torch.Tensor,
    selected_front_nodes: torch.Tensor,
    minimum_dlvsq: float,
    maximum_dlvsq: float,
) -> dict[str, np.ndarray]:
    density = surface.densities
    support = surface.support_strikes
    variance = torch.sum(density * (support - 1.0).square(), dim=-1)
    effective_nodes = 1.0 / torch.sum(density.square(), dim=-1)
    maximum_atom = torch.amax(density, dim=(-2, -1))
    log_width = torch.log(support[:, 0, -1] / support[:, 0, 0])
    dlvsq_range = maximum_dlvsq - minimum_dlvsq
    near_bound = (
        (surface.dlvsqs - minimum_dlvsq <= 1.0e-8 * dlvsq_range)
        | (maximum_dlvsq - surface.dlvsqs <= 1.0e-8 * dlvsq_range)
    ).to(surface.dlvsqs.dtype).mean(dim=(-2, -1))
    boundary = (
        (selected_front_nodes == 0)
        | (selected_front_nodes == density.shape[-1] - 1)
    )
    sqrt_dlvsq = torch.sqrt(surface.dlvsqs)
    expiry_mean = torch.mean(sqrt_dlvsq, dim=1, keepdim=True)
    expiry_dispersion = torch.sqrt(
        torch.mean(torch.square(sqrt_dlvsq - expiry_mean), dim=(1, 2))
    )
    values = {
        "spot": current_spot,
        "front_variance": variance[:, 0],
        "minimum_expiry_variance": torch.amin(variance, dim=1),
        "front_effective_nodes": effective_nodes[:, 0],
        "minimum_expiry_effective_nodes": torch.amin(effective_nodes, dim=1),
        "maximum_atom": maximum_atom,
        "support_log_width": log_width,
        "near_dlvsq_bound_fraction": near_bound,
        "selected_front_boundary": boundary.to(density.dtype),
        "expiry_sqrt_dlvsq_dispersion": expiry_dispersion,
    }
    return {
        name: value.detach().cpu().double().numpy()
        for name, value in values.items()
    }


def _quantiles(value: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(value)),
        "p01": float(np.quantile(value, 0.01)),
        "median": float(np.median(value)),
        "p99": float(np.quantile(value, 0.99)),
        "maximum": float(np.max(value)),
    }


def _summarize_degeneracy(chunks: dict[str, list[np.ndarray]]) -> dict[str, object]:
    values = {name: np.concatenate(parts) for name, parts in chunks.items()}
    return {
        "num_paths": len(values["spot"]),
        "spot": _quantiles(values["spot"]),
        "front_variance": _quantiles(values["front_variance"]),
        "minimum_expiry_variance": _quantiles(values["minimum_expiry_variance"]),
        "front_effective_nodes": _quantiles(values["front_effective_nodes"]),
        "minimum_expiry_effective_nodes": _quantiles(
            values["minimum_expiry_effective_nodes"]
        ),
        "maximum_density_atom": _quantiles(values["maximum_atom"]),
        "support_log_width": _quantiles(values["support_log_width"]),
        "expiry_sqrt_dlvsq_dispersion": _quantiles(
            values["expiry_sqrt_dlvsq_dispersion"]
        ),
        "near_dlvsq_bound_fraction": _quantiles(
            values["near_dlvsq_bound_fraction"]
        ),
        "fraction_selected_front_boundary": float(
            np.mean(values["selected_front_boundary"])
        ),
        "fraction_front_variance_below_1e_12": float(
            np.mean(values["front_variance"] < 1.0e-12)
        ),
        "fraction_any_expiry_variance_below_1e_12": float(
            np.mean(values["minimum_expiry_variance"] < 1.0e-12)
        ),
        "fraction_front_effective_nodes_below_1_01": float(
            np.mean(values["front_effective_nodes"] < 1.01)
        ),
        "fraction_any_expiry_effective_nodes_below_1_01": float(
            np.mean(values["minimum_expiry_effective_nodes"] < 1.01)
        ),
    }


def _advance_ar1_boundary(
    surface: FloatingGridDlvSurface,
    scores: torch.Tensor,
    *,
    model: TorchLastExpiryAr1,
    standard_normals: torch.Tensor,
    decoder: FlexSanosDecoder,
    minimum_dlvsq: float,
    maximum_dlvsq: float,
) -> tuple[FloatingGridDlvSurface, torch.Tensor, dict[str, np.ndarray | float]]:
    """Generate and install one newly issued floating four-week DLV slice."""

    next_scores, canonical_raw = model.step(scores, standard_normals)
    canonical_raw, envelope_clipped = model.clamp_to_training_envelope(
        canonical_raw
    )
    path_raw, extrapolated = interpolate_ar1_raw_curves(
        model,
        canonical_raw,
        surface.support_strikes[:, 0, 1:-1],
    )
    far_dlvsqs, clipped = physical_dlvsq_from_raw(
        path_raw,
        minimum_dlvsq=minimum_dlvsq,
        maximum_dlvsq=maximum_dlvsq,
    )
    replacement = replace_floating_far_expiry(
        decoder,
        surface,
        far_dlvsqs=far_dlvsqs,
    )
    metrics: dict[str, np.ndarray | float] = {
        "extrapolated_coordinate_fraction": (
            extrapolated.to(surface.dlvsqs.dtype)
            .mean(dim=1)
            .detach()
            .cpu()
            .double()
            .numpy()
        ),
        "clipped_coordinate_fraction": (
            clipped.to(surface.dlvsqs.dtype)
            .mean(dim=1)
            .detach()
            .cpu()
            .double()
            .numpy()
        ),
        "training_envelope_clipped_coordinate_fraction": (
            envelope_clipped.to(surface.dlvsqs.dtype)
            .mean(dim=1)
            .detach()
            .cpu()
            .double()
            .numpy()
        ),
        "score_norm": (
            torch.linalg.vector_norm(next_scores, dim=1)
            .detach()
            .cpu()
            .double()
            .numpy()
        ),
        "maximum_mass_residual": float(replacement.maximum_mass_residual.cpu()),
        "maximum_mean_residual": float(replacement.maximum_mean_residual.cpu()),
    }
    return replacement.surface, next_scores, metrics


def _empty_numerical_tracker() -> dict[str, float]:
    return {
        "maximum_density_tower_residual": 0.0,
        "maximum_decoder_density_residual": 0.0,
        "maximum_mass_residual": 0.0,
        "maximum_mean_residual": 0.0,
        "maximum_conditional_mass_residual": 0.0,
        "maximum_conditional_mean_residual": 0.0,
        "maximum_fallback_coordinate_fraction_per_batch": 0.0,
        "minimum_density": float("inf"),
        "minimum_dlvsq": float("inf"),
        "maximum_dlvsq": 0.0,
    }


def _update_numerical_tracker(
    tracker: dict[str, float],
    *,
    density_tower_residual: float,
    decoder_density_residual: float,
    mass_residual: float,
    mean_residual: float,
    conditional_mass_residual: float,
    conditional_mean_residual: float,
    fallback_coordinate_fraction: float,
    minimum_density: float,
    minimum_dlvsq: float,
    maximum_dlvsq: float,
) -> None:
    maxima = {
        "maximum_density_tower_residual": density_tower_residual,
        "maximum_decoder_density_residual": decoder_density_residual,
        "maximum_mass_residual": mass_residual,
        "maximum_mean_residual": mean_residual,
        "maximum_conditional_mass_residual": conditional_mass_residual,
        "maximum_conditional_mean_residual": conditional_mean_residual,
        "maximum_fallback_coordinate_fraction_per_batch": fallback_coordinate_fraction,
        "maximum_dlvsq": maximum_dlvsq,
    }
    for name, value in maxima.items():
        tracker[name] = max(tracker[name], value)
    tracker["minimum_density"] = min(tracker["minimum_density"], minimum_density)
    tracker["minimum_dlvsq"] = min(tracker["minimum_dlvsq"], minimum_dlvsq)


def _trajectory_frame(
    *,
    week: int,
    spot: torch.Tensor,
    support: torch.Tensor,
    densities: torch.Tensor,
    dlvsqs: torch.Tensor,
    backbone_total_vols: torch.Tensor,
) -> dict[str, object]:
    return {
        "week": week,
        "spot": float(spot.detach().cpu()),
        "relative_support": support.detach().cpu().double().tolist(),
        "densities": densities.detach().cpu().double().tolist(),
        "dlvsqs": dlvsqs.detach().cpu().double().tolist(),
        "backbone_total_vols": backbone_total_vols.detach().cpu().double().tolist(),
    }


def run(arguments: argparse.Namespace) -> None:
    """Run the requested all-state audit and write numerical evidence."""

    config = AllStateRegridDlvAuditConfig(
        num_paths=arguments.num_paths,
        num_steps=arguments.num_steps,
        state_batch_size=arguments.state_batch_size,
        num_trajectories=arguments.num_trajectories,
        seed=arguments.seed,
        bs_backbone_annualized_volatility=(
            arguments.bs_backbone_annualized_volatility
        ),
    )
    env = BaseEnvironment(
        precision=arguments.precision,
        device=arguments.device,
        debug=False,
    )
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    geometry = model.config.geometry
    if not isinstance(
        geometry,
        FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    ):
        raise ValueError("all-state DLV audit requires the fixed grid")
    num_states = len(latent_states)
    num_pairs = config.num_paths // 2
    allocation_generator = np.random.default_rng(config.seed)
    groups = _balanced_state_groups(
        num_states=num_states,
        num_pairs=num_pairs,
        generator=allocation_generator,
    )
    selected_trajectory_states = np.unique(
        np.linspace(
            0,
            num_states - 1,
            min(config.num_trajectories, num_states),
            dtype=np.int64,
        )
    )
    trajectories: dict[int, dict[str, object]] = {
        int(index): {
            "source_state_index": int(index),
            "source_state_date": dates[int(index)].isoformat(),
            "antithetic_member": 0,
            "frames": [],
        }
        for index in selected_trajectory_states
    }
    calendar_dt = config.calendar_business_days / DAYS_PER_YEAR
    relative_strikes = torch.exp(
        torch.linspace(
            -0.30,
            0.30,
            config.num_option_strikes,
            dtype=env.fdtype,
            device=env.device,
        )
    )
    relative_is_call = relative_strikes <= 1.0
    fixed_loadings: torch.Tensor | None = None
    origin_state_chunks: list[np.ndarray] = []
    spot_pair_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    rolling_pair_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    fixed_branch_first_member_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    regrid_conditioning_first_member_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    full_transition_first_member_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    direct_transition_first_member_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    maximum_absolute_fixed_branch_antithetic_pair_mean = [
        0.0 for _ in range(config.num_steps)
    ]
    maximum_transition_decomposition_residual = [
        0.0 for _ in range(config.num_steps)
    ]
    maximum_direct_regrid_cash_price_residual = [
        0.0 for _ in range(config.num_steps)
    ]
    original_pair_chunks: list[list[np.ndarray]] = [
        [] for _ in range(min(config.num_steps, geometry.num_model_expiries))
    ]
    factor_score_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    intended_factor_chunks: list[list[np.ndarray]] = [
        [] for _ in range(config.num_steps)
    ]
    degeneracy_chunks: list[dict[str, list[np.ndarray]]] = [
        {
            name: []
            for name in (
                "spot",
                "front_variance",
                "minimum_expiry_variance",
                "front_effective_nodes",
                "minimum_expiry_effective_nodes",
                "maximum_atom",
                "support_log_width",
                "near_dlvsq_bound_fraction",
                "selected_front_boundary",
                "expiry_sqrt_dlvsq_dispersion",
            )
        }
        for _ in range(config.num_steps)
    ]
    boundary_chunks: list[dict[str, list[np.ndarray]]] = [
        {
            "extrapolated_coordinate_fraction": [],
            "clipped_coordinate_fraction": [],
            "training_envelope_clipped_coordinate_fraction": [],
            "score_norm": [],
        }
        for _ in range(config.num_steps)
    ]
    boundary_maximum_mass_residual = [0.0 for _ in range(config.num_steps)]
    boundary_maximum_mean_residual = [0.0 for _ in range(config.num_steps)]
    numerical_by_step = [
        _empty_numerical_tracker() for _ in range(config.num_steps)
    ]
    exact_fixed_grid = {
        "maximum_density_tower_residual": 0.0,
        "maximum_option_tower_residual": 0.0,
        "maximum_spot_mean_residual": 0.0,
    }
    maximum_backbone_residual = 0.0
    minimum_forward_parameter = float("inf")
    maximum_forward_parameter = 0.0
    maximum_first_regrid_cash_price_residual = 0.0
    processed_pairs = 0
    total_started = perf_counter()
    torch_generator = torch.Generator(device=torch.device(env.device)).manual_seed(
        config.seed
    )

    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        block_width = geometry.expiry_block_state_size
        historical_last_raw = (
            all_raw.reshape(num_states, geometry.num_model_expiries, block_width)[
                :, -1, : geometry.num_inner_strikes
            ]
            .cpu()
            .double()
            .numpy()
        )
        boundary_fit = fit_last_expiry_ar1(
            historical_last_raw,
            dates,
            np.asarray(geometry.inner_relative_strikes, dtype=np.float64),
            rank=config.last_expiry_ar1_rank,
        )
        boundary_model = torch_last_expiry_ar1(
            boundary_fit,
            dtype=env.fdtype,
            device=env.device,
        )
        for state_group, pairs_per_state in groups:
            for state_start in range(0, len(state_group), config.state_batch_size):
                state_indices_np = state_group[
                    state_start : state_start + config.state_batch_size
                ]
                state_indices = torch.as_tensor(
                    state_indices_np,
                    dtype=torch.long,
                    device=env.device,
                )
                num_batch_states = len(state_indices_np)
                paths_per_state = 2 * pairs_per_state
                batch_paths = num_batch_states * paths_per_state
                initial_boundary_scores = boundary_model.project(
                    all_raw[state_indices]
                    .reshape(
                        num_batch_states,
                        geometry.num_model_expiries,
                        block_width,
                    )[:, -1, : geometry.num_inner_strikes]
                )
                boundary_scores = (
                    initial_boundary_scores[:, None]
                    .expand(-1, paths_per_state, -1)
                    .reshape(batch_paths, boundary_model.rank)
                )
                if config.bs_backbone_annualized_volatility == 0.0:
                    initial_raw = all_raw[state_indices]
                    forward_parameters = torch.zeros(
                        geometry.num_model_expiries,
                        dtype=env.fdtype,
                        device=env.device,
                    )
                else:
                    initial_raw, forward_parameters = (
                        set_constant_backbone_volatility(
                            all_raw[state_indices],
                            decoder=model.sanos_decoder,
                            annualized_volatility=(
                                config.bs_backbone_annualized_volatility
                            ),
                        )
                    )
                initial_parameters = bound_native_raw(initial_raw, geometry=geometry)
                initial_surface = model.sanos_decoder.forward_batch(
                    dlvsqs=initial_parameters.dlvsqs,
                    forward_vols=initial_parameters.forward_vols,
                    eta=initial_parameters.eta,
                )
                minimum_forward_parameter = min(
                    minimum_forward_parameter,
                    float(forward_parameters.min().cpu()),
                )
                maximum_forward_parameter = max(
                    maximum_forward_parameter,
                    float(forward_parameters.max().cpu()),
                )
                if fixed_loadings is None:
                    fixed_loadings = smooth_lvsq_log_loadings(
                        initial_surface.support_strikes[0, 0],
                        num_model_expiries=geometry.num_model_expiries,
                        annualized_log_volatility=(
                            config.dlvsq_annualized_log_volatility
                        ),
                    )
                fixed_kernel = build_fixed_grid_dlv_kernel(
                    model.sanos_decoder,
                    initial_raw,
                    raw_log_loadings=fixed_loadings,
                    calendar_time_increment_y=calendar_dt,
                )
                if config.bs_backbone_annualized_volatility == 0.0:
                    fixed_kernel = strip_fixed_grid_black_backbone(fixed_kernel)
                initial_surface = fixed_kernel.initial_surface
                expected_backbone = (
                    config.bs_backbone_annualized_volatility
                    * torch.sqrt(model.sanos_decoder.model_expiries_y)
                )
                maximum_backbone_residual = max(
                    maximum_backbone_residual,
                    float(
                        torch.max(
                            torch.abs(
                                initial_surface.backbone_total_vols
                                - expected_backbone
                            )
                        ).cpu()
                    ),
                )
                exact = exact_fixed_grid_dlv_diagnostics(
                    fixed_kernel,
                    fixed_cash_strikes=relative_strikes,
                )
                exact_fixed_grid["maximum_density_tower_residual"] = max(
                    exact_fixed_grid["maximum_density_tower_residual"],
                    float(exact.maximum_density_tower_residual.cpu()),
                )
                exact_fixed_grid["maximum_option_tower_residual"] = max(
                    exact_fixed_grid["maximum_option_tower_residual"],
                    float(exact.maximum_option_tower_residual.cpu()),
                )
                exact_fixed_grid["maximum_spot_mean_residual"] = max(
                    exact_fixed_grid["maximum_spot_mean_residual"],
                    float(exact.maximum_spot_mean_residual.cpu()),
                )
                weekly_randoms = [
                    make_fixed_grid_dlv_randoms(
                        batch_size=num_batch_states,
                        num_paths=paths_per_state,
                        dtype=env.fdtype,
                        device=env.device,
                        generator=torch_generator,
                        antithetic=True,
                    )
                    for _ in range(config.num_steps)
                ]
                sampled = sample_fixed_grid_dlv_paths(
                    model.sanos_decoder,
                    fixed_kernel,
                    fixed_cash_strikes=relative_strikes,
                    randoms=weekly_randoms[0],
                )
                initial_options = sampled.initial_fixed_cash_option_values
                initial_floating_surface = FloatingGridDlvSurface(
                    support_strikes=initial_surface.support_strikes,
                    densities=initial_surface.densities,
                    dlvsqs=initial_surface.dlvsqs,
                    backbone_total_vols=initial_surface.backbone_total_vols,
                )
                selected_branch_surface = _selected_branch_surface(
                    initial_floating_surface,
                    densities_by_branch=(
                        fixed_kernel.finite_step.densities_by_branch
                    ),
                    dlvsqs_by_branch=fixed_kernel.finite_step.dlvsqs_by_branch,
                    branch_indices=sampled.surface_branch_indices,
                )
                selected_branch_options = price_floating_surface_options(
                    selected_branch_surface,
                    current_spot=torch.ones(
                        batch_paths,
                        dtype=env.fdtype,
                        device=env.device,
                    ),
                    fixed_cash_strikes=relative_strikes,
                ).reshape(
                    num_batch_states,
                    paths_per_state,
                    geometry.num_model_expiries,
                    config.num_option_strikes,
                )
                current_spot = sampled.normalized_spot.reshape(-1)
                surface = floating_surface_from_regrid(
                    model.sanos_decoder,
                    sampled.regridded_state,
                )
                ar_normals = make_antithetic_ar1_normals(
                    num_states=num_batch_states,
                    pairs_per_state=pairs_per_state,
                    rank=boundary_model.rank,
                    dtype=env.fdtype,
                    device=env.device,
                    generator=torch_generator,
                )
                surface, boundary_scores, boundary_metrics = _advance_ar1_boundary(
                    surface,
                    boundary_scores,
                    model=boundary_model,
                    standard_normals=ar_normals,
                    decoder=model.sanos_decoder,
                    minimum_dlvsq=geometry.min_dlvsq,
                    maximum_dlvsq=geometry.max_dlvsq,
                )
                for name in boundary_chunks[0]:
                    value = boundary_metrics[name]
                    assert isinstance(value, np.ndarray)
                    boundary_chunks[0][name].append(value)
                boundary_maximum_mass_residual[0] = max(
                    boundary_maximum_mass_residual[0],
                    float(boundary_metrics["maximum_mass_residual"]),
                )
                boundary_maximum_mean_residual[0] = max(
                    boundary_maximum_mean_residual[0],
                    float(boundary_metrics["maximum_mean_residual"]),
                )
                origin_state_chunks.append(
                    np.repeat(state_indices_np, pairs_per_state)
                )
                spot_pair_chunks[0].append(
                    _pair_mean_by_state(
                        sampled.normalized_spot - 1.0,
                        num_states=num_batch_states,
                        num_pairs=pairs_per_state,
                    ).reshape(-1)
                )
                first_regrid_future = price_floating_surface_options(
                    surface,
                    current_spot=current_spot,
                    fixed_cash_strikes=relative_strikes,
                )[:, : geometry.num_model_expiries - 1]
                first_regrid_final = torch.cat(
                    (
                        sampled.fixed_cash_option_values[:, :, :1],
                        first_regrid_future.reshape(
                            num_batch_states,
                            paths_per_state,
                            geometry.num_model_expiries - 1,
                            config.num_option_strikes,
                        ),
                    ),
                    dim=2,
                )
                direct_first_final = sampled.fixed_cash_option_values
                maximum_direct_regrid_cash_price_residual[0] = max(
                    maximum_direct_regrid_cash_price_residual[0],
                    float(
                        torch.max(
                            torch.abs(first_regrid_final - direct_first_final)
                        ).cpu()
                    ),
                )
                first_change = first_regrid_final - initial_options[:, None]
                direct_first_change = (
                    direct_first_final - initial_options[:, None]
                )
                fixed_branch_change = (
                    selected_branch_options - initial_options[:, None]
                )
                regrid_conditioning_change = (
                    first_regrid_final - selected_branch_options
                )
                decomposition_residual = (
                    first_change
                    - fixed_branch_change
                    - regrid_conditioning_change
                )
                maximum_transition_decomposition_residual[0] = max(
                    maximum_transition_decomposition_residual[0],
                    float(torch.max(torch.abs(decomposition_residual)).cpu()),
                )
                maximum_absolute_fixed_branch_antithetic_pair_mean[0] = max(
                    maximum_absolute_fixed_branch_antithetic_pair_mean[0],
                    float(
                        torch.max(
                            torch.abs(
                                fixed_branch_change.reshape(
                                    num_batch_states,
                                    pairs_per_state,
                                    2,
                                    geometry.num_model_expiries,
                                    config.num_option_strikes,
                                ).mean(dim=2)
                            )
                        ).cpu()
                    ),
                )
                fixed_branch_first_member_chunks[0].append(
                    _first_member_by_state(
                        fixed_branch_change,
                        num_states=num_batch_states,
                        num_pairs=pairs_per_state,
                    ).reshape(num_batch_states * pairs_per_state, -1)
                )
                regrid_conditioning_first_member_chunks[0].append(
                    _first_member_by_state(
                        regrid_conditioning_change,
                        num_states=num_batch_states,
                        num_pairs=pairs_per_state,
                    ).reshape(num_batch_states * pairs_per_state, -1)
                )
                full_transition_first_member_chunks[0].append(
                    _first_member_by_state(
                        first_change,
                        num_states=num_batch_states,
                        num_pairs=pairs_per_state,
                    ).reshape(num_batch_states * pairs_per_state, -1)
                )
                direct_transition_first_member_chunks[0].append(
                    _first_member_by_state(
                        direct_first_change,
                        num_states=num_batch_states,
                        num_pairs=pairs_per_state,
                    ).reshape(num_batch_states * pairs_per_state, -1)
                )
                rolling_pair_chunks[0].append(
                    _pair_mean_by_state(
                        first_change,
                        num_states=num_batch_states,
                        num_pairs=pairs_per_state,
                    ).reshape(num_batch_states * pairs_per_state, -1)
                )
                original_pair_chunks[0].append(
                    np.concatenate(
                        (
                            spot_pair_chunks[0][-1][:, None],
                            rolling_pair_chunks[0][-1],
                        ),
                        axis=1,
                    )
                )
                state_position = torch.arange(
                    num_batch_states,
                    device=env.device,
                )[:, None]
                selected_dlvsqs = fixed_kernel.finite_step.dlvsqs_by_branch[
                    state_position,
                    sampled.surface_branch_indices,
                ]
                fixed_design = fixed_loadings.reshape(
                    1,
                    1,
                    geometry.num_model_expiries,
                    geometry.num_inner_strikes,
                    -1,
                ).expand(num_batch_states, paths_per_state, -1, -1, -1)
                first_factor_scores = _factor_scores(
                    initial_dlvsqs=initial_surface.dlvsqs[:, None]
                    .expand(-1, paths_per_state, -1, -1)
                    .reshape(batch_paths, geometry.num_model_expiries, -1),
                    final_dlvsqs=selected_dlvsqs.reshape(
                        batch_paths,
                        geometry.num_model_expiries,
                        -1,
                    ),
                    raw_log_loadings=fixed_design.reshape(
                        batch_paths,
                        geometry.num_model_expiries,
                        geometry.num_inner_strikes,
                        -1,
                    ),
                    minimum_dlvsq=geometry.min_dlvsq,
                    calendar_time_increment_y=calendar_dt,
                    coordinate_weights=initial_surface.densities[:, None, :, 1:-1]
                    .expand(-1, paths_per_state, -1, -1)
                    .reshape(batch_paths, geometry.num_model_expiries, -1),
                )
                factor_score_chunks[0].append(
                    first_factor_scores.cpu().double().numpy()
                )
                intended_factor_chunks[0].append(
                    fixed_kernel.finite_step.standardized_innovations[
                        sampled.surface_branch_indices
                    ].reshape(batch_paths, -1).cpu().double().numpy()
                )
                maximum_first_regrid_cash_price_residual = max(
                    maximum_first_regrid_cash_price_residual,
                    float(sampled.regridded_state.maximum_cash_price_residual.cpu()),
                )
                _update_numerical_tracker(
                    numerical_by_step[0],
                    density_tower_residual=float(
                        fixed_kernel.finite_step.maximum_density_mean_residual.cpu()
                    ),
                    decoder_density_residual=float(
                        fixed_kernel.finite_step.maximum_decoder_density_residual.cpu()
                    ),
                    mass_residual=float(
                        max(
                            float(
                                fixed_kernel.finite_step.maximum_mass_residual.cpu()
                            ),
                            float(boundary_metrics["maximum_mass_residual"]),
                        )
                    ),
                    mean_residual=float(
                        max(
                            float(
                                fixed_kernel.finite_step.maximum_forward_residual.cpu()
                            ),
                            float(boundary_metrics["maximum_mean_residual"]),
                        )
                    ),
                    conditional_mass_residual=float(
                        sampled.maximum_conditional_mass_residual.cpu()
                    ),
                    conditional_mean_residual=float(
                        sampled.maximum_conditional_mean_residual.cpu()
                    ),
                    fallback_coordinate_fraction=float(
                        fixed_kernel.finite_step.fallback_coordinate_fraction.cpu()
                    ),
                    minimum_density=float(
                        min(
                            float(fixed_kernel.finite_step.minimum_density.cpu()),
                            float(torch.min(surface.densities).cpu()),
                        )
                    ),
                    minimum_dlvsq=float(
                        min(
                            float(fixed_kernel.finite_step.minimum_dlvsq.cpu()),
                            float(torch.min(surface.dlvsqs).cpu()),
                        )
                    ),
                    maximum_dlvsq=float(
                        max(
                            float(fixed_kernel.finite_step.maximum_dlvsq.cpu()),
                            float(torch.max(surface.dlvsqs).cpu()),
                        )
                    ),
                )
                first_metrics = _surface_metrics(
                    surface,
                    current_spot=current_spot,
                    selected_front_nodes=sampled.revealed_front_nodes.reshape(-1),
                    minimum_dlvsq=geometry.min_dlvsq,
                    maximum_dlvsq=geometry.max_dlvsq,
                )
                for name, value in first_metrics.items():
                    degeneracy_chunks[0][name].append(value)

                for local_state_position, source_state_index in enumerate(
                    state_indices_np
                ):
                    trajectory = trajectories.get(int(source_state_index))
                    if trajectory is None:
                        continue
                    frames = trajectory["frames"]
                    assert isinstance(frames, list)
                    frames.append(
                        _trajectory_frame(
                            week=0,
                            spot=torch.ones((), dtype=env.fdtype, device=env.device),
                            support=initial_surface.support_strikes[
                                local_state_position
                            ],
                            densities=initial_surface.densities[local_state_position],
                            dlvsqs=initial_surface.dlvsqs[local_state_position],
                            backbone_total_vols=(
                                initial_surface.backbone_total_vols[
                                    local_state_position
                                ]
                            ),
                        )
                    )
                    flat_path = local_state_position * paths_per_state
                    frames.append(
                        _trajectory_frame(
                            week=1,
                            spot=current_spot[flat_path],
                            support=surface.support_strikes[flat_path],
                            densities=surface.densities[flat_path],
                            dlvsqs=surface.dlvsqs[flat_path],
                            backbone_total_vols=surface.backbone_total_vols[
                                flat_path
                            ],
                        )
                    )

                for step_index in range(1, config.num_steps):
                    rolling_cash_strikes = (
                        current_spot[:, None] * relative_strikes[None]
                    )
                    rolling_is_call = relative_is_call[None].expand(
                        batch_paths,
                        -1,
                    )
                    rolling_initial = price_floating_surface_options(
                        surface,
                        current_spot=current_spot,
                        fixed_cash_strikes=rolling_cash_strikes,
                        is_call=rolling_is_call,
                    )
                    loadings = smooth_floating_log_loadings(
                        surface.support_strikes[:, 0],
                        num_model_expiries=geometry.num_model_expiries,
                        annualized_log_volatility=(
                            config.dlvsq_annualized_log_volatility
                        ),
                    )
                    kernel = build_floating_grid_dlv_kernel(
                        model.sanos_decoder,
                        surface,
                        raw_log_loadings=loadings,
                        calendar_time_increment_y=calendar_dt,
                    )
                    randoms = weekly_randoms[step_index]
                    rolled = roll_floating_grid_dlv_kernel(
                        model.sanos_decoder,
                        kernel,
                        surface_branch_uniforms=(
                            randoms.surface_branch_uniforms.reshape(-1)
                        ),
                        front_node_uniforms=(
                            randoms.front_node_uniforms.reshape(-1)
                        ),
                        front_backbone_normals=(
                            randoms.front_backbone_normals.reshape(-1)
                        ),
                    )
                    selected_branch_surface = _selected_branch_surface(
                        surface,
                        densities_by_branch=kernel.densities_by_branch,
                        dlvsqs_by_branch=kernel.dlvsqs_by_branch,
                        branch_indices=rolled.surface_branch_indices,
                    )
                    selected_branch_options = price_floating_surface_options(
                        selected_branch_surface,
                        current_spot=current_spot,
                        fixed_cash_strikes=rolling_cash_strikes,
                        is_call=rolling_is_call,
                    )
                    ar_normals = make_antithetic_ar1_normals(
                        num_states=num_batch_states,
                        pairs_per_state=pairs_per_state,
                        rank=boundary_model.rank,
                        dtype=env.fdtype,
                        device=env.device,
                        generator=torch_generator,
                    )
                    next_surface, boundary_scores, boundary_metrics = (
                        _advance_ar1_boundary(
                            rolled.surface,
                            boundary_scores,
                            model=boundary_model,
                            standard_normals=ar_normals,
                            decoder=model.sanos_decoder,
                            minimum_dlvsq=geometry.min_dlvsq,
                            maximum_dlvsq=geometry.max_dlvsq,
                        )
                    )
                    for name in boundary_chunks[step_index]:
                        value = boundary_metrics[name]
                        assert isinstance(value, np.ndarray)
                        boundary_chunks[step_index][name].append(value)
                    boundary_maximum_mass_residual[step_index] = max(
                        boundary_maximum_mass_residual[step_index],
                        float(boundary_metrics["maximum_mass_residual"]),
                    )
                    boundary_maximum_mean_residual[step_index] = max(
                        boundary_maximum_mean_residual[step_index],
                        float(boundary_metrics["maximum_mean_residual"]),
                    )
                    next_spot = current_spot * rolled.gross_spot_return
                    rolling_payoff = _intrinsic_values(
                        next_spot,
                        rolling_cash_strikes,
                        rolling_is_call,
                    )
                    rolling_future = price_floating_surface_options(
                        next_surface,
                        current_spot=next_spot,
                        fixed_cash_strikes=rolling_cash_strikes,
                        is_call=rolling_is_call,
                    )[:, : geometry.num_model_expiries - 1]
                    direct_future = _direct_survivor_option_values(
                        surface,
                        rolled.surface,
                        current_spot=current_spot,
                        gross_spot_return=rolled.gross_spot_return,
                        revealed_front_nodes=rolled.revealed_front_nodes,
                        fixed_cash_strikes=rolling_cash_strikes,
                        is_call=rolling_is_call,
                    )
                    rolling_final = torch.cat(
                        (rolling_payoff[:, None], rolling_future),
                        dim=1,
                    )
                    direct_final = torch.cat(
                        (rolling_payoff[:, None], direct_future),
                        dim=1,
                    )
                    maximum_direct_regrid_cash_price_residual[step_index] = max(
                        maximum_direct_regrid_cash_price_residual[step_index],
                        float(
                            torch.max(
                                torch.abs(rolling_final - direct_final)
                            ).cpu()
                        ),
                    )
                    rolling_change = (
                        rolling_final - rolling_initial
                    ).reshape(
                        num_batch_states,
                        paths_per_state,
                        geometry.num_model_expiries,
                        config.num_option_strikes,
                    )
                    direct_change = (
                        direct_final - rolling_initial
                    ).reshape_as(rolling_change)
                    fixed_branch_change = (
                        selected_branch_options - rolling_initial
                    ).reshape_as(rolling_change)
                    regrid_conditioning_change = (
                        rolling_final - selected_branch_options
                    ).reshape_as(rolling_change)
                    decomposition_residual = (
                        rolling_change
                        - fixed_branch_change
                        - regrid_conditioning_change
                    )
                    maximum_transition_decomposition_residual[step_index] = max(
                        maximum_transition_decomposition_residual[step_index],
                        float(torch.max(torch.abs(decomposition_residual)).cpu()),
                    )
                    maximum_absolute_fixed_branch_antithetic_pair_mean[step_index] = max(
                        maximum_absolute_fixed_branch_antithetic_pair_mean[step_index],
                        float(
                            torch.max(
                                torch.abs(
                                    fixed_branch_change.reshape(
                                        num_batch_states,
                                        pairs_per_state,
                                        2,
                                        geometry.num_model_expiries,
                                        config.num_option_strikes,
                                    ).mean(dim=2)
                                )
                            ).cpu()
                        ),
                    )
                    fixed_branch_first_member_chunks[step_index].append(
                        _first_member_by_state(
                            fixed_branch_change,
                            num_states=num_batch_states,
                            num_pairs=pairs_per_state,
                        ).reshape(num_batch_states * pairs_per_state, -1)
                    )
                    regrid_conditioning_first_member_chunks[step_index].append(
                        _first_member_by_state(
                            regrid_conditioning_change,
                            num_states=num_batch_states,
                            num_pairs=pairs_per_state,
                        ).reshape(num_batch_states * pairs_per_state, -1)
                    )
                    full_transition_first_member_chunks[step_index].append(
                        _first_member_by_state(
                            rolling_change,
                            num_states=num_batch_states,
                            num_pairs=pairs_per_state,
                        ).reshape(num_batch_states * pairs_per_state, -1)
                    )
                    direct_transition_first_member_chunks[step_index].append(
                        _first_member_by_state(
                            direct_change,
                            num_states=num_batch_states,
                            num_pairs=pairs_per_state,
                        ).reshape(num_batch_states * pairs_per_state, -1)
                    )
                    rolling_pair_chunks[step_index].append(
                        _pair_mean_by_state(
                            rolling_change,
                            num_states=num_batch_states,
                            num_pairs=pairs_per_state,
                        ).reshape(num_batch_states * pairs_per_state, -1)
                    )
                    cumulative_spot_change = next_spot.reshape(
                        num_batch_states,
                        paths_per_state,
                    ) - 1.0
                    spot_pair_chunks[step_index].append(
                        _pair_mean_by_state(
                            cumulative_spot_change,
                            num_states=num_batch_states,
                            num_pairs=pairs_per_state,
                        ).reshape(-1)
                    )
                    if step_index < geometry.num_model_expiries:
                        original_payoff = _intrinsic_values(
                            next_spot,
                            relative_strikes,
                            relative_is_call,
                        )
                        future_count = (
                            geometry.num_model_expiries - step_index - 1
                        )
                        if future_count:
                            original_future = price_floating_surface_options(
                                next_surface,
                                current_spot=next_spot,
                                fixed_cash_strikes=relative_strikes,
                            )[:, :future_count]
                            original_final = torch.cat(
                                (original_payoff[:, None], original_future),
                                dim=1,
                            )
                        else:
                            original_final = original_payoff[:, None]
                        original_change = (
                            original_final.reshape(
                                num_batch_states,
                                paths_per_state,
                                geometry.num_model_expiries - step_index,
                                config.num_option_strikes,
                            )
                            - initial_options[:, None, step_index:]
                        )
                        original_pair_option = _pair_mean_by_state(
                            original_change,
                            num_states=num_batch_states,
                            num_pairs=pairs_per_state,
                        ).reshape(num_batch_states * pairs_per_state, -1)
                        original_pair_chunks[step_index].append(
                            np.concatenate(
                                (
                                    spot_pair_chunks[step_index][-1][:, None],
                                    original_pair_option,
                                ),
                                axis=1,
                            )
                        )
                    path_index = torch.arange(batch_paths, device=env.device)
                    selected_dlvsqs = kernel.dlvsqs_by_branch[
                        path_index,
                        rolled.surface_branch_indices,
                    ]
                    factor_score_chunks[step_index].append(
                        _factor_scores(
                            initial_dlvsqs=surface.dlvsqs,
                            final_dlvsqs=selected_dlvsqs,
                            raw_log_loadings=loadings,
                            minimum_dlvsq=geometry.min_dlvsq,
                            calendar_time_increment_y=calendar_dt,
                            coordinate_weights=surface.densities[..., 1:-1],
                        ).cpu().double().numpy()
                    )
                    intended_factor_chunks[step_index].append(
                        kernel.standardized_innovations[
                            rolled.surface_branch_indices
                        ].cpu().double().numpy()
                    )
                    _update_numerical_tracker(
                        numerical_by_step[step_index],
                        density_tower_residual=float(
                            kernel.maximum_density_mean_residual.cpu()
                        ),
                        decoder_density_residual=float(
                            kernel.maximum_decoder_density_residual.cpu()
                        ),
                        mass_residual=max(
                            float(kernel.maximum_mass_residual.cpu()),
                            float(boundary_metrics["maximum_mass_residual"]),
                        ),
                        mean_residual=max(
                            float(kernel.maximum_mean_residual.cpu()),
                            float(boundary_metrics["maximum_mean_residual"]),
                        ),
                        conditional_mass_residual=float(
                            rolled.maximum_conditional_mass_residual.cpu()
                        ),
                        conditional_mean_residual=float(
                            rolled.maximum_conditional_mean_residual.cpu()
                        ),
                        fallback_coordinate_fraction=float(
                            kernel.fallback_coordinate_fraction.cpu()
                        ),
                        minimum_density=min(
                            float(kernel.minimum_density.cpu()),
                            float(torch.min(next_surface.densities).cpu()),
                        ),
                        minimum_dlvsq=min(
                            float(kernel.minimum_dlvsq.cpu()),
                            float(torch.min(next_surface.dlvsqs).cpu()),
                        ),
                        maximum_dlvsq=max(
                            float(kernel.maximum_dlvsq.cpu()),
                            float(torch.max(next_surface.dlvsqs).cpu()),
                        ),
                    )
                    step_metrics = _surface_metrics(
                        next_surface,
                        current_spot=next_spot,
                        selected_front_nodes=rolled.revealed_front_nodes,
                        minimum_dlvsq=geometry.min_dlvsq,
                        maximum_dlvsq=geometry.max_dlvsq,
                    )
                    for name, value in step_metrics.items():
                        degeneracy_chunks[step_index][name].append(value)
                    for local_state_position, source_state_index in enumerate(
                        state_indices_np
                    ):
                        trajectory = trajectories.get(int(source_state_index))
                        if trajectory is None:
                            continue
                        frames = trajectory["frames"]
                        assert isinstance(frames, list)
                        flat_path = local_state_position * paths_per_state
                        frames.append(
                            _trajectory_frame(
                                week=step_index + 1,
                                spot=next_spot[flat_path],
                                support=next_surface.support_strikes[flat_path],
                                densities=next_surface.densities[flat_path],
                                dlvsqs=next_surface.dlvsqs[flat_path],
                                backbone_total_vols=(
                                    next_surface.backbone_total_vols[flat_path]
                                ),
                            )
                        )
                    current_spot = next_spot
                    surface = next_surface
                    del kernel, rolled, selected_dlvsqs, loadings

                processed_pairs += num_batch_states * pairs_per_state
                elapsed = perf_counter() - total_started
                print(
                    f"processed {processed_pairs:,}/{num_pairs:,} pairs "
                    f"({2 * processed_pairs:,}/{config.num_paths:,} paths) "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )
                del fixed_kernel, sampled, surface, weekly_randoms
        _synchronize(env.device)
    total_seconds = perf_counter() - total_started

    origin_state_indices = np.concatenate(origin_state_chunks)
    if len(origin_state_indices) != num_pairs:
        raise RuntimeError("balanced state allocation produced the wrong pair count")
    spot_pairs = [np.concatenate(parts) for parts in spot_pair_chunks]
    rolling_pairs = [np.concatenate(parts) for parts in rolling_pair_chunks]
    original_pairs = [np.concatenate(parts) for parts in original_pair_chunks]
    fixed_branch_first_members = [
        np.concatenate(parts) for parts in fixed_branch_first_member_chunks
    ]
    regrid_conditioning_first_members = [
        np.concatenate(parts) for parts in regrid_conditioning_first_member_chunks
    ]
    full_transition_first_members = [
        np.concatenate(parts) for parts in full_transition_first_member_chunks
    ]
    direct_transition_first_members = [
        np.concatenate(parts) for parts in direct_transition_first_member_chunks
    ]
    martingale_by_horizon: list[dict[str, object]] = []
    transition_decomposition_by_horizon: list[dict[str, object]] = []
    for step_index in range(config.num_steps):
        spot_observations = spot_pairs[step_index][:, None]
        rolling_observations = rolling_pairs[step_index]
        result: dict[str, object] = {
            "horizon_weeks": step_index + 1,
            "spot_from_origin": _pair_matrix_audit(
                spot_observations,
                correlation_eigenvalue_cutoff=config.correlation_eigenvalue_cutoff,
                first_coordinate_is_spot=True,
            ),
            "rolling_one_week_options": _pair_matrix_audit(
                rolling_observations,
                correlation_eigenvalue_cutoff=config.correlation_eigenvalue_cutoff,
                first_coordinate_is_spot=False,
            ),
        }
        if step_index < len(original_pairs):
            result["original_contracts_with_spot"] = _pair_matrix_audit(
                original_pairs[step_index],
                correlation_eigenvalue_cutoff=config.correlation_eigenvalue_cutoff,
                first_coordinate_is_spot=True,
            )
            result["original_contracts_per_state_z"] = _per_state_z_audit(
                original_pairs[step_index],
                origin_state_indices,
                dates,
            )
        martingale_by_horizon.append(result)
        transition_decomposition_by_horizon.append(
            {
                "horizon_weeks": step_index + 1,
                "fixed_strike_expiry_surface_branch": (
                    _first_member_matrix_audit(
                        fixed_branch_first_members[step_index],
                        correlation_eigenvalue_cutoff=(
                            config.correlation_eigenvalue_cutoff
                        ),
                    )
                ),
                "spot_conditioning_and_regrid": _first_member_matrix_audit(
                    regrid_conditioning_first_members[step_index],
                    correlation_eigenvalue_cutoff=(
                        config.correlation_eigenvalue_cutoff
                    ),
                ),
                "full_transition": _first_member_matrix_audit(
                    full_transition_first_members[step_index],
                    correlation_eigenvalue_cutoff=(
                        config.correlation_eigenvalue_cutoff
                    ),
                ),
                "direct_fixed_cash_transition": _first_member_matrix_audit(
                    direct_transition_first_members[step_index],
                    correlation_eigenvalue_cutoff=(
                        config.correlation_eigenvalue_cutoff
                    ),
                ),
                "full_transition_antithetic_pairs": (
                    result["rolling_one_week_options"]
                ),
                "maximum_absolute_fixed_branch_antithetic_pair_mean": (
                    maximum_absolute_fixed_branch_antithetic_pair_mean[step_index]
                ),
                "maximum_additive_decomposition_residual": (
                    maximum_transition_decomposition_residual[step_index]
                ),
                "maximum_direct_regrid_cash_price_residual": (
                    maximum_direct_regrid_cash_price_residual[step_index]
                ),
            }
        )

    factor_scores = np.stack(
        [np.concatenate(parts) for parts in factor_score_chunks]
    )
    intended_factors = np.stack(
        [np.concatenate(parts) for parts in intended_factor_chunks]
    )
    covariance_audit = _factor_covariance_audit(
        torch.from_numpy(factor_scores)[:, None],
        torch.from_numpy(intended_factors)[:, None],
    )
    degeneracy_by_horizon = [
        {
            "horizon_weeks": step_index + 1,
            **_summarize_degeneracy(degeneracy_chunks[step_index]),
        }
        for step_index in range(config.num_steps)
    ]
    boundary_by_horizon: list[dict[str, object]] = []
    for step_index, chunks in enumerate(boundary_chunks):
        values = {name: np.concatenate(parts) for name, parts in chunks.items()}
        boundary_by_horizon.append(
            {
                "horizon_weeks": step_index + 1,
                "mean_extrapolated_coordinate_fraction": float(
                    np.mean(values["extrapolated_coordinate_fraction"])
                ),
                "mean_clipped_coordinate_fraction": float(
                    np.mean(values["clipped_coordinate_fraction"])
                ),
                "fraction_paths_with_any_clipped_coordinate": float(
                    np.mean(values["clipped_coordinate_fraction"] > 0.0)
                ),
                "mean_training_envelope_clipped_coordinate_fraction": float(
                    np.mean(
                        values[
                            "training_envelope_clipped_coordinate_fraction"
                        ]
                    )
                ),
                "fraction_paths_with_any_training_envelope_clip": float(
                    np.mean(
                        values[
                            "training_envelope_clipped_coordinate_fraction"
                        ]
                        > 0.0
                    )
                ),
                "score_norm": _quantiles(values["score_norm"]),
                "maximum_density_mass_residual": (
                    boundary_maximum_mass_residual[step_index]
                ),
                "maximum_density_mean_residual": (
                    boundary_maximum_mean_residual[step_index]
                ),
            }
        )
    numerical_stability_passed = all(
        value["maximum_density_tower_residual"] <= 1.0e-10
        and value["maximum_decoder_density_residual"] <= 1.0e-10
        and value["maximum_mass_residual"] <= 1.0e-10
        and value["maximum_mean_residual"] <= 1.0e-10
        and value["maximum_conditional_mass_residual"] <= 1.0e-10
        and value["maximum_conditional_mean_residual"] <= 1.0e-10
        and value["minimum_density"] >= -1.0e-12
        and value["minimum_dlvsq"] > geometry.min_dlvsq
        and value["maximum_dlvsq"] < geometry.max_dlvsq
        for value in numerical_by_step
    )
    pair_counts = np.bincount(origin_state_indices, minlength=num_states)
    report: dict[str, object] = {
        "schema_version": 3,
        "experiment_kind": (
            "all_state_repeated_price_preserving_regrid_dlv_ar1_boundary_audit"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "sources": sources,
        "initial_state_sampling": {
            "kind": "balanced allocation over every certified historical state",
            "requested_state_count": 981,
            "certified_state_count": num_states,
            "requested_state_count_note": (
                "The user-supplied 981 is a transposition; the certified report "
                "and checkpoint contain exactly 891 unique states."
            ),
            "num_unique_source_states": int(np.sum(pair_counts > 0)),
            "num_paths": config.num_paths,
            "num_independent_antithetic_pairs": num_pairs,
            "minimum_pairs_per_state": int(np.min(pair_counts)),
            "maximum_pairs_per_state": int(np.max(pair_counts)),
        },
        "backbone": {
            "annualized_volatility": config.bs_backbone_annualized_volatility,
            "role": (
                "disabled exactly at the pricing and path layer; native DLV "
                "density branches are unchanged"
                if config.bs_backbone_annualized_volatility == 0.0
                else "independent mean-one Black factor"
            ),
            "maximum_total_volatility_residual": maximum_backbone_residual,
            "forward_parameters_minimum": minimum_forward_parameter,
            "forward_parameters_maximum": maximum_forward_parameter,
            "eta": 0.40,
        },
        "floating_four_week_boundary": {
            "kind": (
                "training-only 25-factor PCA curve plus unrestricted weekly VAR(1)"
            ),
            "coordinate": "raw log(DLVsq - minimum DLVsq) on canonical strikes",
            "support_guard": (
                "coordinatewise training-period raw-curve envelope applied to "
                "unbounded Gaussian forecasts before physical DLV bounds"
            ),
            "dynamic_role": (
                "newly issued floating 4W boundary; old fixed-expiry 2W--4W "
                "contracts still determine new 1W--3W"
            ),
            "fit": boundary_fit.diagnostics.to_dict(),
            "by_horizon": boundary_by_horizon,
        },
        "exact_fixed_grid_kernel_all_states": exact_fixed_grid,
        "numerical_stability": {
            "passed": numerical_stability_passed,
            "tolerances": {
                "tower_mass_mean_decoder": 1.0e-10,
                "minimum_density": -1.0e-12,
                "dlvsq_bounds": "strict configured bounds",
            },
            "maximum_first_regrid_cash_price_residual": (
                maximum_first_regrid_cash_price_residual
            ),
            "by_horizon": [
                {"horizon_weeks": step_index + 1, **value}
                for step_index, value in enumerate(numerical_by_step)
            ],
        },
        "degeneracy_by_horizon": degeneracy_by_horizon,
        "martingale_audits_by_horizon": martingale_by_horizon,
        "fixed_expiry_vs_regrid_transition_decomposition": {
            "identity": (
                "full option change = fixed-strike/fixed-expiry surface-branch "
                "change + spot-conditioning/re-grid change"
            ),
            "sampling": (
                "Component Wald tests use the first member of each antithetic "
                "pair so the observations are independent and non-degenerate. "
                "The primary full-transition audit continues to use pair means."
            ),
            "by_horizon": transition_decomposition_by_horizon,
        },
        "realized_density_weighted_dlvsq_covariance_audit": covariance_audit,
        "speed": {
            "total_seconds": total_seconds,
            "path_week_steps_per_second": (
                config.num_paths * config.num_steps / total_seconds
            ),
        },
        "antithetic_sources": {
            "origin": "both paths in each pair share one initial state",
            "surface_branch": "u and 1-u",
            "discrete_spot_node": "inverse-CDF u and 1-u",
            "black_spot_factor": (
                "disabled; generated normals are ignored"
                if config.bs_backbone_annualized_volatility == 0.0
                else "Z and -Z"
            ),
            "floating_four_week_ar1_innovation": "Z and -Z",
            "pair_layout": "adjacent through every weekly roll",
        },
        "interpretation": (
            "Weeks one through four audit the original fixed-cash contracts. "
            "Every week also audits spot from origin and a fresh four-expiry "
            "fixed-cash book struck relative to that path's pre-step spot. "
            "The newly issued floating 4W DLV state follows the isolated AR(1) "
            "boundary law and is not copied from the surviving 3W state."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, arguments.output)
    if arguments.trajectory_output is not None:
        trajectory_report = {
            "schema_version": 1,
            "experiment_kind": "regrid_dlv_selected_path_trajectories",
            "created_at_utc": report["created_at_utc"],
            "num_steps": config.num_steps,
            "model_expiry_business_days": list(
                geometry.explicit_model_business_dtes
            ),
            "relative_option_strikes": relative_strikes.cpu().double().tolist(),
            "paths": list(trajectories.values()),
        }
        arguments.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(trajectory_report, arguments.trajectory_output)
    summary = " ".join(
        f"w{value['horizon_weeks']}:"
        f"spot_z={value['spot_from_origin']['spot_z']:.3f},"
        f"rolling_z={value['rolling_one_week_options']['maximum_absolute_coordinate_z']:.3f}"
        for value in martingale_by_horizon
    )
    print(summary, flush=True)
    print(
        f"numerical_stability_passed={numerical_stability_passed} "
        f"wrote {arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    run(_parser().parse_args())
