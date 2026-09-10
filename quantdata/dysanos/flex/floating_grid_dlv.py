"""Finite density-martingale DLV simulation on path-dependent floating grids."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

import torch

from quantdata.dysanos.flex.config import CappedUnboundedDlvFlexGeometryConfig
from quantdata.dysanos.flex.fixed_grid_dlv import RegriddedDlvPathState
from quantdata.dysanos.flex.lvsq_drift import (
    discrete_lvsq_grid_operator_product,
    symmetric_rademacher_cubature,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder


@dataclass(frozen=True)
class FloatingGridDlvSurface:
    """A normalized conditional DLV surface on an arbitrary shared support."""

    support_strikes: torch.Tensor
    densities: torch.Tensor
    dlvsqs: torch.Tensor
    backbone_total_vols: torch.Tensor


@dataclass(frozen=True)
class FloatingGridDlvKernel:
    """Exact-mean density branches for a batch of floating-grid surfaces."""

    initial_surface: FloatingGridDlvSurface
    densities_by_branch: torch.Tensor
    dlvsqs_by_branch: torch.Tensor
    branch_probabilities: torch.Tensor
    standardized_innovations: torch.Tensor
    maximum_density_mean_residual: torch.Tensor
    maximum_decoder_density_residual: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_mean_residual: torch.Tensor
    fallback_coordinate_fraction: torch.Tensor
    minimum_density: torch.Tensor
    minimum_dlvsq: torch.Tensor
    maximum_dlvsq: torch.Tensor


@dataclass(frozen=True)
class FloatingGridDlvRoll:
    """One sampled spot step and its re-gridded four-expiry DLV surface."""

    gross_spot_return: torch.Tensor
    surface_branch_indices: torch.Tensor
    revealed_front_nodes: torch.Tensor
    surface: FloatingGridDlvSurface
    maximum_conditional_mass_residual: torch.Tensor
    maximum_conditional_mean_residual: torch.Tensor


@dataclass(frozen=True)
class FloatingGridFarExpiryReplacement:
    """A floating surface with a separately generated far-expiry boundary."""

    surface: FloatingGridDlvSurface
    maximum_mass_residual: torch.Tensor
    maximum_mean_residual: torch.Tensor


def _validate_surface(
    decoder: FlexSanosDecoder,
    surface: FloatingGridDlvSurface,
) -> tuple[int, int, int]:
    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("floating-grid DLV simulation requires native DLV bounds")
    if surface.support_strikes.ndim != 3:
        raise ValueError("support_strikes must have shape (batch, expiries, support)")
    batch, expiries, support_size = surface.support_strikes.shape
    if (
        expiries != geometry.num_model_expiries
        or support_size != geometry.num_inner_strikes + 2
        or surface.densities.shape != (batch, expiries, support_size)
        or surface.dlvsqs.shape
        != (batch, expiries, geometry.num_inner_strikes)
        or surface.backbone_total_vols.shape != (batch, expiries)
    ):
        raise ValueError("floating-grid surface tensors have incompatible shapes")
    valid_domain = bool(
        torch.all(torch.isfinite(surface.support_strikes))
        and torch.all(torch.diff(surface.support_strikes, dim=-1) > 0.0)
        and torch.all(
            surface.support_strikes == surface.support_strikes[:, :1]
        )
        and torch.all(torch.isfinite(surface.densities))
        and torch.all(surface.densities >= 0.0)
        and torch.all(torch.isfinite(surface.dlvsqs))
        and torch.all(surface.dlvsqs > geometry.min_dlvsq)
        and torch.all(surface.dlvsqs < geometry.max_dlvsq)
        and torch.all(torch.isfinite(surface.backbone_total_vols))
        and torch.all(surface.backbone_total_vols >= 0.0)
    )
    if not valid_domain:
        raise ValueError(
            "floating-grid surface lies outside its numerical domain: "
            f"support_finite={bool(torch.all(torch.isfinite(surface.support_strikes)))}, "
            f"minimum_support_step={float(torch.min(torch.diff(surface.support_strikes, dim=-1)).cpu()):.6g}, "
            f"density_finite={bool(torch.all(torch.isfinite(surface.densities)))}, "
            f"minimum_density={float(torch.min(surface.densities).cpu()):.6g}, "
            f"dlvsq_finite={bool(torch.all(torch.isfinite(surface.dlvsqs)))}, "
            f"minimum_dlvsq={float(torch.min(surface.dlvsqs).cpu()):.6g}, "
            f"maximum_dlvsq={float(torch.max(surface.dlvsqs).cpu()):.6g}, "
            "backbone_finite="
            f"{bool(torch.all(torch.isfinite(surface.backbone_total_vols)))}"
        )
    tolerance = 2.0e-10
    mass = torch.sum(surface.densities, dim=-1)
    mean = torch.sum(surface.densities * surface.support_strikes, dim=-1)
    if not bool(
        torch.max(torch.abs(mass - 1.0)) <= tolerance
        and torch.max(torch.abs(mean - 1.0)) <= tolerance
    ):
        raise ValueError("floating-grid densities must have unit mass and mean")
    return batch, expiries, support_size


def _clamp_dlvsq_to_open_bounds(
    values: torch.Tensor,
    *,
    geometry: CappedUnboundedDlvFlexGeometryConfig,
) -> torch.Tensor:
    """Apply the configured pricing bounds without returning either endpoint."""

    minimum = torch.as_tensor(
        geometry.min_dlvsq,
        dtype=values.dtype,
        device=values.device,
    )
    maximum = torch.as_tensor(
        geometry.max_dlvsq,
        dtype=values.dtype,
        device=values.device,
    )
    return torch.clamp(
        values,
        min=torch.nextafter(minimum, maximum),
        max=torch.nextafter(maximum, minimum),
    )


def smooth_floating_log_loadings(
    support: torch.Tensor,
    *,
    num_model_expiries: int,
    annualized_log_volatility: float,
) -> torch.Tensor:
    """Evaluate level/skew/curvature/expiry loadings on each floating grid."""

    if support.ndim != 2 or support.shape[-1] < 7:
        raise ValueError("support must have shape (batch, at least seven nodes)")
    if num_model_expiries < 2:
        raise ValueError("num_model_expiries must be at least two")
    if not isfinite(annualized_log_volatility) or annualized_log_volatility <= 0.0:
        raise ValueError("annualized_log_volatility must be finite and positive")
    inner = support[:, 1:-1]
    log_inner = torch.log(inner)
    log_scale = torch.amax(torch.abs(log_inner), dim=-1, keepdim=True)
    if not bool(torch.all(log_scale > 0.0)):
        raise ValueError("floating support needs at least one non-unit inner node")
    strike_coordinate = log_inner / log_scale
    strike_coordinate = strike_coordinate[:, None].expand(
        len(support),
        num_model_expiries - 1,
        inner.shape[-1],
    )
    expiry_coordinate = torch.linspace(
        -1.0,
        1.0,
        num_model_expiries - 1,
        dtype=support.dtype,
        device=support.device,
    ).reshape(1, -1, 1).expand_as(strike_coordinate)
    factors = torch.stack(
        (
            torch.ones_like(strike_coordinate),
            strike_coordinate,
            2.0 * strike_coordinate.square() - 1.0,
            expiry_coordinate,
        ),
        dim=-1,
    )
    factors = factors / torch.linalg.vector_norm(factors, dim=-1, keepdim=True)
    taper = torch.sin(
        torch.arange(
            1,
            inner.shape[-1] + 1,
            dtype=support.dtype,
            device=support.device,
        )
        * pi
        / (inner.shape[-1] + 1)
    )
    result = torch.zeros(
        (
            len(support),
            num_model_expiries,
            inner.shape[-1],
            4,
        ),
        dtype=support.dtype,
        device=support.device,
    )
    result[:, 1:] = (
        annualized_log_volatility * taper.reshape(1, 1, -1, 1) * factors
    )
    return result


def _sample_categorical(
    probabilities: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.clamp_min(probabilities, 0.0)
    probabilities = probabilities / torch.clamp_min(
        torch.sum(probabilities, dim=-1, keepdim=True),
        torch.finfo(probabilities.dtype).tiny,
    )
    cdf = torch.cumsum(probabilities, dim=-1)
    cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), dim=-1)
    return torch.sum(uniforms[..., None] > cdf, dim=-1).to(torch.int64)


def build_floating_grid_dlv_kernel(
    decoder: FlexSanosDecoder,
    surface: FloatingGridDlvSurface,
    *,
    raw_log_loadings: torch.Tensor,
    calendar_time_increment_y: float,
    positivity_tolerance: float = 1.0e-12,
    density_mass_floor: float = 1.0e-14,
) -> FloatingGridDlvKernel:
    """Build symmetric exact-mean density branches on arbitrary supports."""

    batch, expiries, support_size = _validate_surface(decoder, surface)
    geometry = decoder.geometry
    num_inner = support_size - 2
    if not isfinite(calendar_time_increment_y) or calendar_time_increment_y <= 0.0:
        raise ValueError("calendar_time_increment_y must be finite and positive")
    if raw_log_loadings.shape != (batch, expiries, num_inner, 4):
        raise ValueError("raw_log_loadings have an incompatible floating-grid shape")
    if not bool(
        torch.all(torch.isfinite(raw_log_loadings))
        and torch.all(raw_log_loadings[:, 0] == 0.0)
    ):
        raise ValueError("floating-grid loadings must be finite with a frozen front")

    factors = raw_log_loadings.shape[-1]
    density_diffusion = torch.zeros(
        (batch, expiries, support_size, factors),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    raw_drift = torch.zeros(
        (batch, expiries, num_inner),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    previous_density_diffusion = density_diffusion[:, 0]
    for expiry_index in range(1, expiries):
        support = surface.support_strikes[:, expiry_index]
        density = surface.densities[:, expiry_index]
        dlvsq = surface.dlvsqs[:, expiry_index]
        excess = dlvsq - geometry.min_dlvsq
        physical_diffusion = excess[..., None] * raw_log_loadings[:, expiry_index]
        source = discrete_lvsq_grid_operator_product(
            support,
            density[:, 1:-1, None] * physical_diffusion,
            expiry_time_increment_y=decoder.model_dts_y[expiry_index],
        )
        rhs = previous_density_diffusion - source
        flat_support = support[:, None].expand(-1, factors, -1).reshape(
            batch * factors,
            support_size,
        )
        flat_dlvsq = dlvsq[:, None].expand(-1, factors, -1).reshape(
            batch * factors,
            num_inner,
        )
        current_density_diffusion = decoder.solve_density_step_batch(
            support=flat_support,
            dlvsqs=flat_dlvsq,
            time_increment_y=decoder.model_dts_y[expiry_index],
            rhs_density=rhs.permute(0, 2, 1).reshape(batch * factors, support_size),
        ).reshape(batch, factors, support_size).permute(0, 2, 1)
        relative = current_density_diffusion / torch.clamp_min(
            density[..., None],
            density_mass_floor,
        )
        physical_drift = -torch.sum(
            relative[:, 1:-1] * physical_diffusion,
            dim=-1,
        )
        raw_drift[:, expiry_index] = (
            physical_drift / excess
            - 0.5 * torch.sum(raw_log_loadings[:, expiry_index].square(), dim=-1)
        )
        density_diffusion[:, expiry_index] = current_density_diffusion
        previous_density_diffusion = current_density_diffusion

    innovations, probabilities = symmetric_rademacher_cubature(
        factors,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    branches = len(innovations)
    density_shock = torch.einsum(
        "btnf,rf->brtn",
        density_diffusion,
        innovations,
    )
    branch_densities = (
        surface.densities[:, None]
        + calendar_time_increment_y**0.5 * density_shock
    )
    minimum_density = torch.min(branch_densities)
    if not bool(minimum_density >= -positivity_tolerance):
        raise RuntimeError(
            "floating density branch became materially negative: "
            f"{float(minimum_density.detach().cpu()):.6g}"
        )
    branch_densities = torch.clamp_min(branch_densities, 0.0)
    shock = torch.einsum("btnf,rf->brtn", raw_log_loadings, innovations)
    fallback_dlvsqs = geometry.min_dlvsq + (
        surface.dlvsqs[:, None] - geometry.min_dlvsq
    ) * torch.exp(
        calendar_time_increment_y * raw_drift[:, None]
        + calendar_time_increment_y**0.5 * shock
    )
    fallback_dlvsqs = _clamp_dlvsq_to_open_bounds(
        fallback_dlvsqs,
        geometry=geometry,
    )
    branch_dlvsqs = fallback_dlvsqs.clone()
    branch_dlvsqs[:, :, 0] = surface.dlvsqs[:, None, 0]
    fallback_count = torch.zeros(
        (),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    for expiry_index in range(1, expiries):
        support = surface.support_strikes[:, expiry_index]
        strikes = support[:, 1:-1]
        payoff = torch.relu(support[:, None] - strikes[..., None])
        current_density = branch_densities[:, :, expiry_index]
        previous_density = branch_densities[:, :, expiry_index - 1]
        current_calls = torch.einsum("bij,brj->bri", payoff, current_density)
        previous_calls = torch.einsum("bij,brj->bri", payoff, previous_density)
        box = (
            strikes.square()
            * decoder.model_dts_y[expiry_index]
            / (support[:, 2:] - support[:, :-2])
        )
        denominator = box[:, None] * current_density[..., 1:-1]
        recovered = (current_calls - previous_calls) / torch.clamp_min(
            denominator,
            torch.finfo(surface.densities.dtype).tiny,
        )
        admissible = (
            (current_density[..., 1:-1] > density_mass_floor)
            & torch.isfinite(recovered)
            & (recovered > geometry.min_dlvsq)
            & (recovered < geometry.max_dlvsq)
        )
        fallback_count += torch.sum(~admissible)
        branch_dlvsqs[:, :, expiry_index] = torch.where(
            admissible,
            recovered,
            branch_dlvsqs[:, :, expiry_index],
        )

    decoded = branch_densities[:, :, :1].clone()
    decoded_parts = [decoded[:, :, 0]]
    previous = decoded[:, :, 0]
    for expiry_index in range(1, expiries):
        support = surface.support_strikes[:, None, expiry_index].expand(
            batch,
            branches,
            support_size,
        )
        previous = decoder.solve_density_step_batch(
            support=support.reshape(batch * branches, support_size),
            dlvsqs=branch_dlvsqs[:, :, expiry_index].reshape(
                batch * branches,
                num_inner,
            ),
            time_increment_y=decoder.model_dts_y[expiry_index],
            rhs_density=previous.reshape(batch * branches, support_size),
        ).reshape(batch, branches, support_size)
        decoded_parts.append(previous)
    decoded_densities = torch.stack(decoded_parts, dim=2)
    density_mean = torch.sum(
        probabilities.reshape(1, branches, 1, 1) * branch_densities,
        dim=1,
    )
    mass = torch.sum(branch_densities, dim=-1)
    mean = torch.sum(
        branch_densities * surface.support_strikes[:, None],
        dim=-1,
    )
    dynamic_coordinates = batch * branches * (expiries - 1) * num_inner
    return FloatingGridDlvKernel(
        initial_surface=surface,
        densities_by_branch=branch_densities,
        dlvsqs_by_branch=branch_dlvsqs,
        branch_probabilities=probabilities,
        standardized_innovations=innovations,
        maximum_density_mean_residual=torch.max(
            torch.abs(density_mean - surface.densities)
        ),
        maximum_decoder_density_residual=torch.max(
            torch.abs(decoded_densities - branch_densities)
        ),
        maximum_mass_residual=torch.max(torch.abs(mass - 1.0)),
        maximum_mean_residual=torch.max(torch.abs(mean - 1.0)),
        fallback_coordinate_fraction=fallback_count / dynamic_coordinates,
        minimum_density=minimum_density,
        minimum_dlvsq=torch.min(branch_dlvsqs),
        maximum_dlvsq=torch.max(branch_dlvsqs),
    )


def floating_surface_from_regrid(
    decoder: FlexSanosDecoder,
    regridded: RegriddedDlvPathState,
) -> FloatingGridDlvSurface:
    """Append an unconstrained far expiry to an exact rolled path surface."""

    batch, paths, remaining, support_size = regridded.densities.shape
    expiries = decoder.geometry.num_model_expiries
    if remaining != expiries - 1:
        raise ValueError("re-gridded state does not contain every surviving expiry")
    flat = batch * paths
    support = regridded.relative_support.reshape(flat, support_size)
    densities = regridded.densities.reshape(flat, remaining, support_size)
    dlvsqs = regridded.dlvsqs.reshape(flat, remaining, support_size - 2)
    backbone = regridded.backbone_total_vols.reshape(flat, remaining)
    tail_dlvsq = dlvsqs[:, -1]
    tail_density = decoder.solve_density_step_batch(
        support=support,
        dlvsqs=tail_dlvsq,
        time_increment_y=decoder.model_dts_y[-1],
        rhs_density=densities[:, -1],
    )
    last_increment_variance = (
        backbone[:, -1].square() - backbone[:, -2].square()
        if remaining > 1
        else backbone[:, -1].square()
    )
    tail_backbone = torch.sqrt(
        torch.clamp_min(backbone[:, -1].square() + last_increment_variance, 0.0)
    )
    return FloatingGridDlvSurface(
        support_strikes=support[:, None].expand(flat, expiries, support_size),
        densities=torch.cat((densities, tail_density[:, None]), dim=1),
        dlvsqs=torch.cat((dlvsqs, tail_dlvsq[:, None]), dim=1),
        backbone_total_vols=torch.cat((backbone, tail_backbone[:, None]), dim=1),
    )


def replace_floating_far_expiry(
    decoder: FlexSanosDecoder,
    surface: FloatingGridDlvSurface,
    *,
    far_dlvsqs: torch.Tensor,
) -> FloatingGridFarExpiryReplacement:
    """Replace the copied tail by a valid newly issued floating expiry.

    The first three expiry densities and DLV states are fixed-contract
    survivors.  ``far_dlvsqs`` is a separately modeled floating-tenor boundary;
    the SANOS step constructs its density as a martingale extension of the
    surviving penultimate density.
    """

    batch, _, support_size = _validate_surface(decoder, surface)
    geometry = decoder.geometry
    if far_dlvsqs.shape != (batch, support_size - 2):
        raise ValueError("far DLVs must have shape (batch, inner strikes)")
    if not bool(
        torch.all(torch.isfinite(far_dlvsqs))
        and torch.all(far_dlvsqs > geometry.min_dlvsq)
        and torch.all(far_dlvsqs < geometry.max_dlvsq)
    ):
        raise ValueError("far DLVs lie outside strict configured bounds")
    far_density = decoder.solve_density_step_batch(
        support=surface.support_strikes[:, -1],
        dlvsqs=far_dlvsqs,
        time_increment_y=decoder.model_dts_y[-1],
        rhs_density=surface.densities[:, -2],
    )
    mass_residual = torch.max(torch.abs(torch.sum(far_density, dim=-1) - 1.0))
    mean_residual = torch.max(
        torch.abs(
            torch.sum(far_density * surface.support_strikes[:, -1], dim=-1)
            - 1.0
        )
    )
    replaced = FloatingGridDlvSurface(
        support_strikes=surface.support_strikes,
        densities=torch.cat((surface.densities[:, :-1], far_density[:, None]), dim=1),
        dlvsqs=torch.cat((surface.dlvsqs[:, :-1], far_dlvsqs[:, None]), dim=1),
        backbone_total_vols=surface.backbone_total_vols,
    )
    _validate_surface(decoder, replaced)
    return FloatingGridFarExpiryReplacement(
        surface=replaced,
        maximum_mass_residual=mass_residual,
        maximum_mean_residual=mean_residual,
    )


def roll_floating_grid_dlv_kernel(
    decoder: FlexSanosDecoder,
    kernel: FloatingGridDlvKernel,
    *,
    surface_branch_uniforms: torch.Tensor,
    front_node_uniforms: torch.Tensor,
    front_backbone_normals: torch.Tensor,
) -> FloatingGridDlvRoll:
    """Sample one kernel branch and spot, then roll back to four expiries."""

    surface = kernel.initial_surface
    batch, expiries, support_size = _validate_surface(decoder, surface)
    expected_shape = (batch,)
    if not (
        surface_branch_uniforms.shape
        == front_node_uniforms.shape
        == front_backbone_normals.shape
        == expected_shape
    ):
        raise ValueError("floating-grid randoms must have shape (batch,)")
    branch_probabilities = kernel.branch_probabilities[None].expand(batch, -1)
    branch_index = _sample_categorical(
        branch_probabilities,
        surface_branch_uniforms,
    )
    front_node = _sample_categorical(surface.densities[:, 0], front_node_uniforms)
    batch_index = torch.arange(batch, device=surface.densities.device)
    selected_dlvsqs = kernel.dlvsqs_by_branch[batch_index, branch_index]
    front_support = surface.support_strikes[:, 0]
    latent_front = torch.gather(
        front_support,
        -1,
        front_node[:, None],
    ).squeeze(-1)
    gross_spot = latent_front * torch.exp(
        -0.5 * surface.backbone_total_vols[:, 0].square()
        + surface.backbone_total_vols[:, 0] * front_backbone_normals
    )
    relative_support = front_support / latent_front[:, None]
    conditional = torch.nn.functional.one_hot(
        front_node,
        num_classes=support_size,
    ).to(dtype=surface.densities.dtype)
    densities: list[torch.Tensor] = []
    maximum_mass_residual = torch.zeros_like(gross_spot[:1]).squeeze(0)
    maximum_mean_residual = torch.zeros_like(maximum_mass_residual)
    for expiry_index in range(1, expiries):
        conditional = decoder.solve_density_step_batch(
            support=surface.support_strikes[:, expiry_index],
            dlvsqs=selected_dlvsqs[:, expiry_index],
            time_increment_y=decoder.model_dts_y[expiry_index],
            rhs_density=conditional,
        )
        densities.append(conditional)
        maximum_mass_residual = torch.maximum(
            maximum_mass_residual,
            torch.max(torch.abs(torch.sum(conditional, dim=-1) - 1.0)),
        )
        maximum_mean_residual = torch.maximum(
            maximum_mean_residual,
            torch.max(
                torch.abs(
                    torch.sum(conditional * relative_support, dim=-1) - 1.0
                )
            ),
        )
    surviving_densities = torch.stack(densities, dim=1)
    old_dts = decoder.model_dts_y
    rolled_dlvsqs = selected_dlvsqs[:, 1:] * (
        old_dts[1:] / old_dts[:-1]
    ).reshape(1, -1, 1)
    rolled_dlvsqs = _clamp_dlvsq_to_open_bounds(
        rolled_dlvsqs,
        geometry=decoder.geometry,
    )
    rolled_backbone = torch.sqrt(
        torch.clamp_min(
            surface.backbone_total_vols[:, 1:].square()
            - surface.backbone_total_vols[:, :1].square(),
            0.0,
        )
    )
    temporary = RegriddedDlvPathState(
        relative_support=relative_support[:, None],
        density0=torch.nn.functional.one_hot(
            front_node,
            num_classes=support_size,
        ).to(dtype=surface.densities.dtype)[:, None],
        densities=surviving_densities[:, None],
        dlvsqs=rolled_dlvsqs[:, None],
        backbone_total_vols=rolled_backbone[:, None],
        normalized_option_values=torch.empty(
            (batch, 1, expiries - 1, 0),
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        ),
        fixed_cash_option_values=torch.empty(
            (batch, 1, expiries - 1, 0),
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        ),
        maximum_mass_residual=maximum_mass_residual,
        maximum_mean_residual=maximum_mean_residual,
        maximum_cash_price_residual=torch.zeros_like(maximum_mass_residual),
    )
    rolled_surface = floating_surface_from_regrid(decoder, temporary)
    return FloatingGridDlvRoll(
        gross_spot_return=gross_spot,
        surface_branch_indices=branch_index,
        revealed_front_nodes=front_node,
        surface=rolled_surface,
        maximum_conditional_mass_residual=maximum_mass_residual,
        maximum_conditional_mean_residual=maximum_mean_residual,
    )


def price_floating_surface_options(
    surface: FloatingGridDlvSurface,
    *,
    current_spot: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    is_call: torch.Tensor | None = None,
) -> torch.Tensor:
    """Price calls or puts in the cash numeraire on a floating DLV surface.

    ``fixed_cash_strikes`` may be one shared strike vector or one vector per
    path.  The latter is useful for auditing freshly issued relative-strike
    contracts after paths have diverged.  In that case callers should pass the
    option type fixed at contract inception through ``is_call``; inferring it
    from the terminal spot would change the contract along the path.
    """

    batch, expiries, _ = surface.support_strikes.shape
    if current_spot.shape != (batch,):
        raise ValueError("current_spot must have shape (batch,)")
    if fixed_cash_strikes.ndim == 1:
        num_strikes = len(fixed_cash_strikes)
        strike = fixed_cash_strikes.reshape(1, 1, 1, num_strikes)
        parity_strike = fixed_cash_strikes.reshape(1, 1, num_strikes)
        default_is_call = (fixed_cash_strikes <= 1.0).reshape(1, 1, num_strikes)
        expected_is_call_shapes = {(num_strikes,), (1, num_strikes)}
    elif fixed_cash_strikes.ndim == 2 and fixed_cash_strikes.shape[0] == batch:
        num_strikes = fixed_cash_strikes.shape[1]
        strike = fixed_cash_strikes[:, None, None]
        parity_strike = fixed_cash_strikes[:, None]
        default_is_call = (
            fixed_cash_strikes <= current_spot[:, None]
        )[:, None]
        expected_is_call_shapes = {
            (batch, num_strikes),
            (batch, 1, num_strikes),
        }
    else:
        raise ValueError(
            "fixed_cash_strikes must have shape (strikes,) or (batch, strikes)"
        )
    if not bool(
        torch.all(torch.isfinite(fixed_cash_strikes))
        and torch.all(fixed_cash_strikes > 0.0)
    ):
        raise ValueError("fixed_cash_strikes must be finite and positive")
    if is_call is None:
        option_is_call = default_is_call
    else:
        if is_call.dtype != torch.bool or tuple(is_call.shape) not in expected_is_call_shapes:
            raise ValueError("is_call has an incompatible shape or dtype")
        option_is_call = is_call.reshape(
            (1 if fixed_cash_strikes.ndim == 1 else batch),
            1,
            num_strikes,
        )
    component_forward = current_spot[:, None, None] * surface.support_strikes
    volatility = surface.backbone_total_vols[..., None, None]
    safe_volatility = torch.clamp_min(
        volatility,
        torch.finfo(component_forward.dtype).tiny,
    )
    forwards = component_forward[..., None]
    d1 = (torch.log(forwards / strike) + 0.5 * safe_volatility.square()) / (
        safe_volatility
    )
    component_calls = forwards * torch.special.ndtr(d1) - strike * torch.special.ndtr(
        d1 - safe_volatility
    )
    component_calls = torch.where(
        volatility > 0.0,
        component_calls,
        torch.relu(forwards - strike),
    )
    calls = torch.sum(surface.densities[..., None] * component_calls, dim=2)
    puts = calls - current_spot[:, None, None] + parity_strike
    return torch.where(option_is_call, calls, puts)


__all__ = [
    "FloatingGridFarExpiryReplacement",
    "FloatingGridDlvKernel",
    "FloatingGridDlvRoll",
    "FloatingGridDlvSurface",
    "build_floating_grid_dlv_kernel",
    "floating_surface_from_regrid",
    "price_floating_surface_options",
    "replace_floating_far_expiry",
    "roll_floating_grid_dlv_kernel",
    "smooth_floating_log_loadings",
]
