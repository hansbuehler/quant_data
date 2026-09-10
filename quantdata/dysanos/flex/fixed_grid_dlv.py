"""Finite-step fixed-cash DLV simulation with an explicit spot filtration.

The calendar-time DLV shock is represented by a finite cubature kernel whose
branch-weighted densities equal the starting surface.  Conditional on a
surface branch, the front SANOS node and independent Black backbone are then
sampled.  Later option values are conditional on the revealed node, the
realized spot, and the surface branch.  Consequently both spot and every
fixed-cash option satisfy their one-step tower identities before Monte Carlo
sampling is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal

import torch

from quantdata.dysanos.flex.config import CappedUnboundedDlvFlexGeometryConfig
from quantdata.dysanos.flex.lvsq_drift import (
    DensityMartingaleLvsqStep,
    FixedHorizonLvsqStep,
    density_martingale_lvsq_step,
    fixed_horizon_lvsq_exponential_step,
)
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface

FixedGridDlvKernelMethod = Literal["density_martingale", "exponential_newton"]


@dataclass(frozen=True)
class FixedGridDlvKernel:
    """One-week finite DLV kernel and its decoded initial surface."""

    initial_raw_states: torch.Tensor
    initial_surface: SanosSurface
    finite_step: DensityMartingaleLvsqStep | FixedHorizonLvsqStep
    calendar_time_increment_y: float
    method: FixedGridDlvKernelMethod


@dataclass(frozen=True)
class FixedGridDlvRandoms:
    """Independent uniforms and normals used by the fixed-grid simulator."""

    surface_branch_uniforms: torch.Tensor
    front_node_uniforms: torch.Tensor
    front_backbone_normals: torch.Tensor


@dataclass(frozen=True)
class FixedGridDlvPathBatch:
    """One-week spot and conditional fixed-cash option values."""

    normalized_spot: torch.Tensor
    surface_branch_indices: torch.Tensor
    revealed_front_nodes: torch.Tensor
    fixed_cash_option_values: torch.Tensor
    initial_fixed_cash_option_values: torch.Tensor
    maximum_conditional_mass_residual: torch.Tensor
    maximum_conditional_mean_residual: torch.Tensor
    regridded_state: RegriddedDlvPathState


@dataclass(frozen=True)
class RegriddedDlvPathState:
    """Price-preserving DLV state on the revealed path's relative grid."""

    relative_support: torch.Tensor
    density0: torch.Tensor
    densities: torch.Tensor
    dlvsqs: torch.Tensor
    backbone_total_vols: torch.Tensor
    normalized_option_values: torch.Tensor
    fixed_cash_option_values: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_mean_residual: torch.Tensor
    maximum_cash_price_residual: torch.Tensor


@dataclass(frozen=True)
class FixedGridDlvExactDiagnostics:
    """Deterministic tower residuals before path sampling."""

    initial_spot_mean: torch.Tensor
    maximum_spot_mean_residual: torch.Tensor
    maximum_option_tower_residual: torch.Tensor
    maximum_density_tower_residual: torch.Tensor


def _decode_surface(
    decoder: FlexSanosDecoder,
    raw_states: torch.Tensor,
) -> SanosSurface:
    parameters = bound_native_raw(raw_states, geometry=decoder.geometry)
    return decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )


def set_constant_backbone_volatility(
    raw_states: torch.Tensor,
    *,
    decoder: FlexSanosDecoder,
    annualized_volatility: float,
    backbone_eta: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set the Black backbone itself to one constant annualized volatility.

    FLEX defines the Black forward increment as ``forward_vol * eta``.  Eta is
    held at one deterministic level and the forward-vol coordinates are chosen
    to make cumulative Black total volatility equal
    ``annualized_volatility * sqrt(T)`` at every model expiry.  This keeps spot
    and DLVs as the only random quantities and remains feasible even for source
    states whose fitted eta is very small.
    """

    if not isfinite(annualized_volatility) or annualized_volatility <= 0.0:
        raise ValueError("annualized_volatility must be finite and positive")
    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("constant backbone replacement requires native FLEX blocks")
    if raw_states.ndim != 2 or raw_states.shape[-1] != geometry.native_state_size:
        raise ValueError("raw_states have an invalid native-state shape")
    if not isfinite(backbone_eta) or not (
        geometry.min_eta < backbone_eta < geometry.max_eta
    ):
        raise ValueError("backbone_eta must lie strictly inside the FLEX bounds")
    target_total_vols = annualized_volatility * torch.sqrt(decoder.model_expiries_y)
    target_increments = torch.diff(
        target_total_vols,
        prepend=torch.zeros(
            1,
            dtype=raw_states.dtype,
            device=raw_states.device,
        ),
    )
    target_backbone_forward = target_increments / torch.sqrt(
        decoder.model_dts_y
    )
    forward_parameters = target_backbone_forward / backbone_eta
    if not bool(
        torch.all(forward_parameters > geometry.min_forward_vol)
        and torch.all(forward_parameters < geometry.max_forward_vol)
    ):
        raise ValueError(
            "constant backbone volatility requires forward parameters outside "
            "the FLEX bounds"
        )
    probability = (
        (forward_parameters - geometry.min_forward_vol)
        / (geometry.max_forward_vol - geometry.min_forward_vol)
    )
    raw_forward = torch.logit(probability)
    eta_probability = (
        (backbone_eta - geometry.min_eta)
        / (geometry.max_eta - geometry.min_eta)
    )
    raw_eta = torch.logit(
        torch.as_tensor(
            eta_probability,
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
    )
    blocks = raw_states.reshape(
        len(raw_states),
        geometry.num_model_expiries,
        geometry.expiry_block_state_size,
    ).clone()
    blocks[..., geometry.num_inner_strikes] = raw_forward
    blocks[..., geometry.num_inner_strikes + 1] = raw_eta
    return blocks.reshape_as(raw_states), forward_parameters


def strip_fixed_grid_black_backbone(
    kernel: FixedGridDlvKernel,
) -> FixedGridDlvKernel:
    """Return the same DLV kernel with its independent Black factor disabled.

    SANOS represents the discrete DLV factor and the independent Black factor
    separately.  A strictly zero Black volatility is outside the bounded native
    parameterization, so an X-only experiment removes it at the pricing and path
    layer after constructing the unchanged DLV density branches.
    """

    surface = replace(
        kernel.initial_surface,
        backbone_total_vols=torch.zeros_like(
            kernel.initial_surface.backbone_total_vols
        ),
    )
    return replace(kernel, initial_surface=surface)


def build_fixed_grid_dlv_kernel(
    decoder: FlexSanosDecoder,
    raw_states: torch.Tensor,
    *,
    raw_log_loadings: torch.Tensor,
    calendar_time_increment_y: float,
    moment_tolerance: float = 1.0e-12,
    method: FixedGridDlvKernelMethod = "density_martingale",
) -> FixedGridDlvKernel:
    """Construct an exactly martingalized finite DLV branch kernel."""

    initial_surface = _decode_surface(decoder, raw_states)
    if not bool(
        torch.all(
            initial_surface.support_strikes
            == initial_surface.support_strikes[:, :1]
        )
    ):
        raise ValueError("fixed-grid simulation requires one cash support per state")
    if method == "density_martingale":
        finite_step = density_martingale_lvsq_step(
            decoder,
            raw_states,
            raw_log_loadings=raw_log_loadings,
            calendar_time_increment_y=calendar_time_increment_y,
        )
    elif method == "exponential_newton":
        finite_step = fixed_horizon_lvsq_exponential_step(
            decoder,
            raw_states,
            raw_log_loadings=raw_log_loadings,
            calendar_time_increment_y=calendar_time_increment_y,
            moment_tolerance=moment_tolerance,
        )
    else:
        raise ValueError(f"unknown fixed-grid DLV kernel method {method!r}")
    return FixedGridDlvKernel(
        initial_raw_states=raw_states,
        initial_surface=initial_surface,
        finite_step=finite_step,
        calendar_time_increment_y=calendar_time_increment_y,
        method=method,
    )


def make_fixed_grid_dlv_randoms(
    *,
    batch_size: int,
    num_paths: int,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    antithetic: bool = True,
) -> FixedGridDlvRandoms:
    """Generate surface, discrete-spot, and backbone innovations.

    Antithetic draws are stored as adjacent pairs.  The surface cubature is
    ordered so ``u`` and ``1-u`` select opposite Rademacher branches; the spot
    node uses antithetic categorical quantiles and the Black factor uses
    ``Z`` and ``-Z``.
    """

    if batch_size <= 0 or num_paths <= 0:
        raise ValueError("batch_size and num_paths must be positive")
    if antithetic and num_paths % 2 != 0:
        raise ValueError("antithetic simulation requires an even num_paths")
    draws = num_paths // 2 if antithetic else num_paths
    shape = (batch_size, draws)
    surface_uniform = torch.rand(
        shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    front_uniform = torch.rand(
        shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    backbone_normal = torch.randn(
        shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if antithetic:
        maximum_uniform = torch.nextafter(
            torch.ones((), dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        )

        def paired_uniform(value: torch.Tensor) -> torch.Tensor:
            complement = torch.clamp(1.0 - value, max=maximum_uniform)
            return torch.stack((value, complement), dim=-1).reshape(
                batch_size,
                num_paths,
            )

        surface_uniform = paired_uniform(surface_uniform)
        front_uniform = paired_uniform(front_uniform)
        backbone_normal = torch.stack(
            (backbone_normal, -backbone_normal),
            dim=-1,
        ).reshape(batch_size, num_paths)
    return FixedGridDlvRandoms(
        surface_branch_uniforms=surface_uniform,
        front_node_uniforms=front_uniform,
        front_backbone_normals=backbone_normal,
    )


def _sample_categorical(
    probabilities: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Sample the last axis of a batched probability tensor."""

    if probabilities.shape[:-1] != uniforms.shape:
        raise ValueError("probabilities and uniforms have incompatible shapes")
    normalized = torch.clamp_min(probabilities, 0.0)
    normalized = normalized / torch.clamp_min(
        torch.sum(normalized, dim=-1, keepdim=True),
        torch.finfo(normalized.dtype).tiny,
    )
    cdf = torch.cumsum(normalized, dim=-1)
    cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), dim=-1)
    return torch.sum(uniforms.unsqueeze(-1) > cdf, dim=-1).to(torch.int64)


def _black_calls(
    forwards: torch.Tensor,
    strikes: torch.Tensor,
    total_volatility: torch.Tensor,
) -> torch.Tensor:
    """Black calls broadcasting component forwards against fixed cash strikes."""

    safe_volatility = torch.clamp_min(
        total_volatility,
        torch.finfo(forwards.dtype).tiny,
    )
    d1 = (
        torch.log(forwards / strikes) + 0.5 * safe_volatility.square()
    ) / safe_volatility
    black = forwards * torch.special.ndtr(d1) - strikes * torch.special.ndtr(
        d1 - safe_volatility
    )
    return torch.where(
        total_volatility > 0.0,
        black,
        torch.relu(forwards - strikes),
    )


def _surface_fixed_cash_options(
    surface: SanosSurface,
    fixed_cash_strikes: torch.Tensor,
) -> torch.Tensor:
    """Price the ITM call/put set directly from an unconditional surface."""

    component = surface.support_strikes[..., None]
    strike = fixed_cash_strikes.reshape(1, 1, 1, -1)
    total_volatility = surface.backbone_total_vols[..., None, None]
    calls = torch.sum(
        surface.densities[..., None]
        * _black_calls(component, strike, total_volatility),
        dim=2,
    )
    puts = calls - 1.0 + fixed_cash_strikes.reshape(1, 1, -1)
    return torch.where(
        (fixed_cash_strikes <= 1.0).reshape(1, 1, -1),
        calls,
        puts,
    )


def exact_fixed_grid_dlv_diagnostics(
    kernel: FixedGridDlvKernel,
    *,
    fixed_cash_strikes: torch.Tensor,
) -> FixedGridDlvExactDiagnostics:
    """Evaluate exact spot and option towers over every DLV branch."""

    surface = kernel.initial_surface
    step = kernel.finite_step
    batch, branches, expiries, support_size = step.densities_by_branch.shape
    branch_surface = SanosSurface(
        expiries_y=surface.expiries_y,
        support_strikes=(
            surface.support_strikes[:, None]
            .expand(batch, branches, expiries, support_size)
            .reshape(batch * branches, expiries, support_size)
        ),
        density0=(
            surface.density0[:, None]
            .expand(batch, branches, support_size)
            .reshape(batch * branches, support_size)
        ),
        densities=step.densities_by_branch.reshape(
            batch * branches,
            expiries,
            support_size,
        ),
        dlvsqs=step.dlvsqs_by_branch.reshape(
            batch * branches,
            expiries,
            support_size - 2,
        ),
        forward_vols=(
            surface.forward_vols[:, None]
            .expand(batch, branches, expiries)
            .reshape(batch * branches, expiries)
        ),
        total_vols=(
            surface.total_vols[:, None]
            .expand(batch, branches, expiries)
            .reshape(batch * branches, expiries)
        ),
        eta=(
            surface.eta[:, None]
            .expand(batch, branches, *surface.eta.shape[1:])
            .reshape(batch * branches, *surface.eta.shape[1:])
        ),
        backbone_total_vols=(
            surface.backbone_total_vols[:, None]
            .expand(batch, branches, expiries)
            .reshape(batch * branches, expiries)
        ),
    )
    initial_options = _surface_fixed_cash_options(surface, fixed_cash_strikes)
    branch_options = _surface_fixed_cash_options(
        branch_surface,
        fixed_cash_strikes,
    ).reshape(batch, branches, expiries, -1)
    expected_options = torch.sum(
        step.branch_probabilities.reshape(1, branches, 1, 1) * branch_options,
        dim=1,
    )
    spot_mean = torch.sum(
        surface.densities[:, 0] * surface.support_strikes[:, 0],
        dim=-1,
    )
    return FixedGridDlvExactDiagnostics(
        initial_spot_mean=spot_mean,
        maximum_spot_mean_residual=torch.max(torch.abs(spot_mean - 1.0)),
        maximum_option_tower_residual=torch.max(
            torch.abs(expected_options - initial_options)
        ),
        maximum_density_tower_residual=step.maximum_density_mean_residual,
    )


def sample_fixed_grid_dlv_paths(
    decoder: FlexSanosDecoder,
    kernel: FixedGridDlvKernel,
    *,
    fixed_cash_strikes: torch.Tensor,
    randoms: FixedGridDlvRandoms,
) -> FixedGridDlvPathBatch:
    """Sample spot and conditional cash option values from a fixed-grid kernel."""

    surface = kernel.initial_surface
    step = kernel.finite_step
    batch = len(kernel.initial_raw_states)
    if (
        randoms.surface_branch_uniforms.shape
        != randoms.front_node_uniforms.shape
        or randoms.surface_branch_uniforms.shape
        != randoms.front_backbone_normals.shape
        or randoms.surface_branch_uniforms.ndim != 2
        or randoms.surface_branch_uniforms.shape[0] != batch
    ):
        raise ValueError("fixed-grid random tensors must share shape (batch, paths)")
    for value in (
        randoms.surface_branch_uniforms,
        randoms.front_node_uniforms,
        randoms.front_backbone_normals,
        fixed_cash_strikes,
    ):
        if value.dtype != surface.densities.dtype or value.device != surface.densities.device:
            raise ValueError("randoms and strikes must preserve surface dtype and device")
    if not bool(
        torch.all(
            torch.isfinite(randoms.surface_branch_uniforms)
            & (randoms.surface_branch_uniforms >= 0.0)
            & (randoms.surface_branch_uniforms < 1.0)
            & torch.isfinite(randoms.front_node_uniforms)
            & (randoms.front_node_uniforms >= 0.0)
            & (randoms.front_node_uniforms < 1.0)
            & torch.isfinite(randoms.front_backbone_normals)
        )
    ):
        raise ValueError("fixed-grid random tensors contain invalid values")
    if fixed_cash_strikes.ndim != 1 or not bool(
        torch.all(torch.isfinite(fixed_cash_strikes) & (fixed_cash_strikes > 0.0))
    ):
        raise ValueError("fixed_cash_strikes must be a positive finite vector")

    paths = randoms.surface_branch_uniforms.shape[1]
    branches = len(step.branch_probabilities)
    branch_probabilities = step.branch_probabilities.reshape(1, 1, branches).expand(
        batch,
        paths,
        branches,
    )
    branch_indices = _sample_categorical(
        branch_probabilities,
        randoms.surface_branch_uniforms,
    )
    front_probabilities = surface.densities[:, None, 0].expand(
        batch,
        paths,
        surface.densities.shape[-1],
    )
    front_nodes = _sample_categorical(
        front_probabilities,
        randoms.front_node_uniforms,
    )
    front_support = surface.support_strikes[:, None, 0].expand(
        batch,
        paths,
        surface.support_strikes.shape[-1],
    )
    latent_front = torch.gather(
        front_support,
        dim=-1,
        index=front_nodes.unsqueeze(-1),
    ).squeeze(-1)
    front_volatility = surface.backbone_total_vols[:, :1]
    normalized_spot = latent_front * torch.exp(
        -0.5 * front_volatility.square()
        + front_volatility * randoms.front_backbone_normals
    )

    regridded = regrid_fixed_grid_dlv_paths(
        decoder,
        kernel,
        normalized_spot=normalized_spot,
        surface_branch_indices=branch_indices,
        revealed_front_nodes=front_nodes,
        fixed_cash_strikes=fixed_cash_strikes,
    )
    is_call = (fixed_cash_strikes <= 1.0).reshape(1, 1, -1)
    front_calls = torch.relu(
        normalized_spot[..., None] - fixed_cash_strikes.reshape(1, 1, -1)
    )
    front_puts = torch.relu(
        fixed_cash_strikes.reshape(1, 1, -1) - normalized_spot[..., None]
    )
    front_values = torch.where(is_call, front_calls, front_puts)

    return FixedGridDlvPathBatch(
        normalized_spot=normalized_spot,
        surface_branch_indices=branch_indices,
        revealed_front_nodes=front_nodes,
        fixed_cash_option_values=torch.cat(
            (front_values.unsqueeze(2), regridded.fixed_cash_option_values),
            dim=2,
        ),
        initial_fixed_cash_option_values=_surface_fixed_cash_options(
            surface,
            fixed_cash_strikes,
        ),
        maximum_conditional_mass_residual=regridded.maximum_mass_residual,
        maximum_conditional_mean_residual=regridded.maximum_mean_residual,
        regridded_state=regridded,
    )


def regrid_fixed_grid_dlv_paths(
    decoder: FlexSanosDecoder,
    kernel: FixedGridDlvKernel,
    *,
    normalized_spot: torch.Tensor,
    surface_branch_indices: torch.Tensor,
    revealed_front_nodes: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
) -> RegriddedDlvPathState:
    """Condition and re-express a fixed-cash surface in current-spot units.

    The relative discrete grid is the old cash grid divided by the revealed
    front SANOS node.  The independent front Black multiplier is already in
    ``normalized_spot`` and therefore cancels from future component ratios.
    The returned normalized option values are multiplied back by spot and
    checked against direct fixed-cash valuation path by path.
    """

    surface = kernel.initial_surface
    step = kernel.finite_step
    if (
        normalized_spot.shape != surface_branch_indices.shape
        or normalized_spot.shape != revealed_front_nodes.shape
        or normalized_spot.ndim != 2
        or normalized_spot.shape[0] != len(kernel.initial_raw_states)
    ):
        raise ValueError("re-grid path tensors must share shape (batch, paths)")
    batch, paths = normalized_spot.shape
    support_size = surface.support_strikes.shape[-1]
    if not bool(
        torch.all(torch.isfinite(normalized_spot) & (normalized_spot > 0.0))
        and torch.all(
            (surface_branch_indices >= 0)
            & (surface_branch_indices < len(step.branch_probabilities))
        )
        and torch.all(
            (revealed_front_nodes >= 0)
            & (revealed_front_nodes < support_size)
        )
    ):
        raise ValueError("re-grid spot, branch, or node lies outside its domain")

    batch_index = torch.arange(batch, device=surface.densities.device)[:, None]
    selected_dlvsqs = step.dlvsqs_by_branch[
        batch_index,
        surface_branch_indices,
    ]
    front_support = surface.support_strikes[:, None, 0].expand(
        batch,
        paths,
        support_size,
    )
    latent_front = torch.gather(
        front_support,
        dim=-1,
        index=revealed_front_nodes.unsqueeze(-1),
    ).squeeze(-1)
    relative_support = front_support / latent_front[..., None]
    density0 = torch.nn.functional.one_hot(
        revealed_front_nodes,
        num_classes=support_size,
    ).to(dtype=surface.densities.dtype)
    conditional_density = density0
    conditional_densities: list[torch.Tensor] = []
    normalized_values: list[torch.Tensor] = []
    cash_values: list[torch.Tensor] = []
    maximum_mass_residual = torch.zeros(
        (),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    maximum_mean_residual = torch.zeros_like(maximum_mass_residual)
    for expiry_index in range(1, surface.densities.shape[1]):
        support = surface.support_strikes[:, None, expiry_index].expand(
            batch,
            paths,
            support_size,
        )
        conditional_density = decoder.solve_density_step_batch(
            support=support.reshape(batch * paths, support_size),
            dlvsqs=selected_dlvsqs[:, :, expiry_index].reshape(
                batch * paths,
                support_size - 2,
            ),
            time_increment_y=decoder.model_dts_y[expiry_index],
            rhs_density=conditional_density.reshape(batch * paths, support_size),
        ).reshape(batch, paths, support_size)
        expiry_relative_support = support / latent_front[..., None]
        conditional_densities.append(conditional_density)
        maximum_mass_residual = torch.maximum(
            maximum_mass_residual,
            torch.max(torch.abs(torch.sum(conditional_density, dim=-1) - 1.0)),
        )
        maximum_mean_residual = torch.maximum(
            maximum_mean_residual,
            torch.max(
                torch.abs(
                    torch.sum(
                        conditional_density * expiry_relative_support,
                        dim=-1,
                    )
                    - 1.0
                )
            ),
        )
        component_forward = normalized_spot[..., None] * expiry_relative_support
        cash_strike = fixed_cash_strikes.reshape(1, 1, 1, -1)
        relative_strike = cash_strike / normalized_spot[..., None, None]
        increment_variance = (
            surface.backbone_total_vols[:, expiry_index].square()
            - surface.backbone_total_vols[:, 0].square()
        ).reshape(batch, 1, 1, 1)
        increment_volatility = torch.sqrt(torch.clamp_min(increment_variance, 0.0))
        component_cash_calls = _black_calls(
            component_forward[..., None],
            cash_strike,
            increment_volatility,
        )
        cash_calls = torch.sum(
            conditional_density[..., None] * component_cash_calls,
            dim=2,
        )
        cash_puts = cash_calls - normalized_spot[..., None] + fixed_cash_strikes.reshape(
            1,
            1,
            -1,
        )
        is_call = (fixed_cash_strikes <= 1.0).reshape(1, 1, -1)
        cash_values.append(torch.where(is_call, cash_calls, cash_puts))

        component_normalized_calls = _black_calls(
            expiry_relative_support[..., None],
            relative_strike,
            increment_volatility,
        )
        normalized_calls = torch.sum(
            conditional_density[..., None] * component_normalized_calls,
            dim=2,
        )
        normalized_puts = (
            normalized_calls
            - 1.0
            + fixed_cash_strikes.reshape(1, 1, -1)
            / normalized_spot[..., None]
        )
        normalized_values.append(
            torch.where(is_call, normalized_calls, normalized_puts)
        )

    normalized_option_values = torch.stack(normalized_values, dim=2)
    fixed_cash_option_values = torch.stack(cash_values, dim=2)
    cash_from_regrid = normalized_spot[..., None, None] * normalized_option_values
    old_dts = decoder.model_dts_y
    rolled_dts = torch.cat((old_dts[:1], old_dts[2:]))
    dlvsq_time_scale = old_dts[1:] / rolled_dts
    rolled_dlvsqs = selected_dlvsqs[:, :, 1:] * dlvsq_time_scale.reshape(
        1,
        1,
        -1,
        1,
    )
    rolled_backbone = torch.sqrt(
        torch.clamp_min(
            surface.backbone_total_vols[:, 1:].square()
            - surface.backbone_total_vols[:, :1].square(),
            0.0,
        )
    )
    return RegriddedDlvPathState(
        relative_support=relative_support,
        density0=density0,
        densities=torch.stack(conditional_densities, dim=2),
        dlvsqs=rolled_dlvsqs,
        backbone_total_vols=rolled_backbone[:, None].expand(batch, paths, -1),
        normalized_option_values=normalized_option_values,
        fixed_cash_option_values=fixed_cash_option_values,
        maximum_mass_residual=maximum_mass_residual,
        maximum_mean_residual=maximum_mean_residual,
        maximum_cash_price_residual=torch.max(
            torch.abs(cash_from_regrid - fixed_cash_option_values)
        ),
    )


__all__ = [
    "FixedGridDlvExactDiagnostics",
    "FixedGridDlvKernel",
    "FixedGridDlvKernelMethod",
    "FixedGridDlvPathBatch",
    "FixedGridDlvRandoms",
    "RegriddedDlvPathState",
    "build_fixed_grid_dlv_kernel",
    "exact_fixed_grid_dlv_diagnostics",
    "make_fixed_grid_dlv_randoms",
    "regrid_fixed_grid_dlv_paths",
    "sample_fixed_grid_dlv_paths",
    "set_constant_backbone_volatility",
    "strip_fixed_grid_black_backbone",
]
