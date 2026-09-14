"""Attempt 2.0: rolling fixed-grid simulation of the DLVsq SDE.

The calendar-time SDE holds the current front density fixed while the DLVsq
slabs ending at the later fixed expiries evolve under equation (15).  At the
end of one week, the front density is revealed as the new spot and the three
surviving DLVsq slabs are relabelled ``2 -> 1``, ``3 -> 2``, ``4 -> 3``.
All densities remain on the original fixed cash grid in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.lvsq_drift import (
    fixed_contract_lvsq_drift,
    fixed_contract_lvsq_euler_step,
)
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.sanos import FlexSanosDecoder


@dataclass(frozen=True)
class FixedGridAttempt2Randoms:
    """Brownian increments and front-density uniforms for one weekly roll."""

    surface_standard_normals: torch.Tensor
    front_node_uniforms: torch.Tensor


@dataclass(frozen=True)
class FixedGridAttempt2Roll:
    """One fixed-grid SDE roll aligned with the four old contract maturities."""

    normalized_spot: torch.Tensor
    front_node_indices: torch.Tensor
    endpoint_raw_states: torch.Tensor
    endpoint_unconditional_densities: torch.Tensor
    aligned_conditional_densities: torch.Tensor
    initial_option_values: torch.Tensor
    endpoint_unconditional_survivor_option_values: torch.Tensor
    aligned_conditional_option_values: torch.Tensor
    maximum_conditional_mass_residual: torch.Tensor
    maximum_conditional_forward_residual: torch.Tensor
    maximum_fixed_grid_change: torch.Tensor
    maximum_frozen_coordinate_change: torch.Tensor
    maximum_volatility_equation_residual: torch.Tensor
    maximum_drift_equation_residual: torch.Tensor
    maximum_density_diffusion_mass_residual: torch.Tensor
    maximum_density_diffusion_forward_residual: torch.Tensor
    minimum_density_during_sde: torch.Tensor
    minimum_raw_cap_margin: torch.Tensor


def make_fixed_grid_attempt2_randoms(
    *,
    num_states: int,
    num_paths: int,
    num_substeps: int,
    num_factors: int,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    antithetic: bool = True,
) -> FixedGridAttempt2Randoms:
    """Generate independent state/path shocks with adjacent antithetic pairs."""

    if min(num_states, num_paths, num_substeps, num_factors) <= 0:
        raise ValueError("random tensor dimensions must be positive")
    if antithetic and num_paths % 2:
        raise ValueError("antithetic simulation requires an even path count")
    num_draws = num_paths // 2 if antithetic else num_paths
    normals = torch.randn(
        (num_states, num_draws, num_substeps, num_factors),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    uniforms = torch.rand(
        (num_states, num_draws),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if antithetic:
        normals = torch.stack((normals, -normals), dim=2).reshape(
            num_states,
            num_paths,
            num_substeps,
            num_factors,
        )
        maximum_uniform = torch.nextafter(
            torch.ones((), dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        )
        uniforms = torch.stack(
            (uniforms, torch.clamp(1.0 - uniforms, max=maximum_uniform)),
            dim=2,
        ).reshape(num_states, num_paths)
    return FixedGridAttempt2Randoms(
        surface_standard_normals=normals,
        front_node_uniforms=uniforms,
    )


def _sample_categorical(
    probabilities: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Sample the last probability axis using explicitly supplied uniforms."""

    if probabilities.shape[:-1] != uniforms.shape:
        raise ValueError("probabilities and uniforms have incompatible shapes")
    if not bool(
        torch.all(torch.isfinite(probabilities))
        and torch.all(probabilities >= 0.0)
        and torch.all(torch.isfinite(uniforms))
        and torch.all((uniforms >= 0.0) & (uniforms < 1.0))
    ):
        raise ValueError("categorical inputs must be finite probabilities and uniforms")
    probabilities = probabilities / torch.clamp_min(
        torch.sum(probabilities, dim=-1, keepdim=True),
        torch.finfo(probabilities.dtype).tiny,
    )
    cdf = torch.cumsum(probabilities, dim=-1)
    cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), dim=-1)
    return torch.sum(uniforms.unsqueeze(-1) > cdf, dim=-1).to(torch.int64)


def fixed_grid_attempt2_conditional_densities(
    decoder: FlexSanosDecoder,
    *,
    support: torch.Tensor,
    endpoint_dlvsqs: torch.Tensor,
    front_node_indices: torch.Tensor,
) -> torch.Tensor:
    r"""Reveal spot and roll the old DLVsq slabs on one unchanged cash grid.

    ``endpoint_dlvsqs[..., j]`` still carries the old fixed-expiry index.  The
    result's density index zero is the realized old first expiry, and result
    index ``j`` is the conditional value of the old expiry ``j+1``.  Hence its
    four entries align directly with the four option prices observed before
    the weekly transition.
    """

    geometry = decoder.geometry
    if not isinstance(
        geometry,
        FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    ):
        raise ValueError("Attempt 2.0 requires a fixed relative-strike geometry")
    if endpoint_dlvsqs.ndim < 3 or endpoint_dlvsqs.shape[-2:] != (
        geometry.num_model_expiries,
        geometry.num_inner_strikes,
    ):
        raise ValueError("endpoint_dlvsqs have an invalid shape")
    leading_shape = endpoint_dlvsqs.shape[:-2]
    support_size = geometry.num_inner_strikes + 2
    if support.shape != leading_shape + (support_size,):
        raise ValueError("support must align with every endpoint DLV state")
    if front_node_indices.shape != leading_shape:
        raise ValueError("front_node_indices must align with endpoint DLV states")
    if not bool(
        torch.all(torch.isfinite(support))
        and torch.all(torch.diff(support, dim=-1) > 0.0)
        and torch.all(
            (front_node_indices >= 0) & (front_node_indices < support_size)
        )
    ):
        raise ValueError("support or front node lies outside the fixed grid")

    flat_size = front_node_indices.numel()
    flat_support = support.reshape(flat_size, support_size)
    flat_dlvsqs = endpoint_dlvsqs.reshape(
        flat_size,
        geometry.num_model_expiries,
        geometry.num_inner_strikes,
    )
    density = torch.nn.functional.one_hot(
        front_node_indices.reshape(-1),
        num_classes=support_size,
    ).to(dtype=endpoint_dlvsqs.dtype)
    aligned_densities = [density]
    for old_expiry_index in range(1, geometry.num_model_expiries):
        density = decoder.solve_density_step_batch(
            support=flat_support,
            dlvsqs=flat_dlvsqs[:, old_expiry_index],
            time_increment_y=decoder.model_dts_y[old_expiry_index],
            rhs_density=density,
        )
        aligned_densities.append(density)
    return torch.stack(aligned_densities, dim=1).reshape(
        leading_shape
        + (
            geometry.num_model_expiries,
            support_size,
        )
    )


def _call_option_payoffs(
    support: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
) -> torch.Tensor:
    """Return call payoffs at every requested fixed cash strike."""

    return torch.relu(support[..., None] - fixed_cash_strikes)


def fixed_grid_attempt2_sde_roll(
    decoder: FlexSanosDecoder,
    raw_states: torch.Tensor,
    *,
    raw_log_loadings: torch.Tensor,
    calendar_time_increment_y: float,
    randoms: FixedGridAttempt2Randoms,
    fixed_cash_strikes: torch.Tensor,
) -> FixedGridAttempt2Roll:
    """Simulate equation (15), reveal the front spot, and roll three slabs."""

    geometry = decoder.geometry
    if not isinstance(
        geometry,
        FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    ):
        raise ValueError("Attempt 2.0 requires a fixed relative-strike geometry")
    if not isfinite(calendar_time_increment_y) or calendar_time_increment_y <= 0.0:
        raise ValueError("calendar_time_increment_y must be finite and positive")
    if raw_states.ndim != 2 or raw_states.shape[-1] != geometry.native_state_size:
        raise ValueError("raw_states must be a batch of complete native states")
    if fixed_cash_strikes.ndim != 1 or not bool(
        torch.all(torch.isfinite(fixed_cash_strikes) & (fixed_cash_strikes > 0.0))
    ):
        raise ValueError("fixed_cash_strikes must be a positive finite vector")
    normals = randoms.surface_standard_normals
    uniforms = randoms.front_node_uniforms
    if normals.ndim != 4 or normals.shape[:2] != uniforms.shape:
        raise ValueError("Attempt 2.0 randoms must have shapes (state,path,step,factor)")
    if normals.shape[0] != len(raw_states):
        raise ValueError("random state count differs from raw_states")
    if raw_log_loadings.ndim != 3:
        raise ValueError("Attempt 2.0 currently requires shared raw log-loadings")
    if normals.shape[-1] != raw_log_loadings.shape[-1]:
        raise ValueError("normal and loading factor counts differ")
    for value in (normals, uniforms, fixed_cash_strikes):
        if value.dtype != raw_states.dtype or value.device != raw_states.device:
            raise ValueError("states, randoms, and strikes must share dtype and device")
    if not bool(torch.all(torch.isfinite(normals))):
        raise ValueError("surface normals must be finite")

    num_states, num_paths, num_substeps, _ = normals.shape
    num_expiries = geometry.num_model_expiries
    num_inner = geometry.num_inner_strikes
    support_size = num_inner + 2
    block_width = geometry.expiry_block_state_size
    parameters = bound_native_raw(raw_states, geometry=geometry)
    initial_surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    if not bool(
        torch.all(
            initial_surface.support_strikes
            == initial_surface.support_strikes[:, :1]
        )
    ):
        raise ValueError("Attempt 2.0 requires one fixed cash support")

    current = (
        raw_states[:, None]
        .expand(num_states, num_paths, geometry.native_state_size)
        .reshape(num_states * num_paths, geometry.native_state_size)
        .clone()
    )
    step_y = calendar_time_increment_y / num_substeps
    zero = torch.zeros((), dtype=raw_states.dtype, device=raw_states.device)
    infinity = torch.full_like(zero, torch.inf)
    maximum_volatility_residual = zero
    maximum_drift_residual = zero
    maximum_mass_diffusion_residual = zero
    maximum_forward_diffusion_residual = zero
    minimum_density = infinity
    minimum_cap_margin = infinity
    for step_index in range(num_substeps):
        restriction = fixed_contract_lvsq_drift(
            decoder,
            current,
            raw_log_loadings=raw_log_loadings,
        )
        maximum_volatility_residual = torch.maximum(
            maximum_volatility_residual,
            restriction.maximum_volatility_equation_residual,
        )
        maximum_drift_residual = torch.maximum(
            maximum_drift_residual,
            restriction.maximum_drift_equation_residual,
        )
        maximum_mass_diffusion_residual = torch.maximum(
            maximum_mass_diffusion_residual,
            restriction.maximum_density_diffusion_mass_residual,
        )
        maximum_forward_diffusion_residual = torch.maximum(
            maximum_forward_diffusion_residual,
            restriction.maximum_density_diffusion_forward_residual,
        )
        minimum_density = torch.minimum(minimum_density, restriction.minimum_density)
        minimum_cap_margin = torch.minimum(
            minimum_cap_margin,
            restriction.minimum_raw_cap_margin,
        )
        current = fixed_contract_lvsq_euler_step(
            restriction,
            current,
            calendar_time_increment_y=step_y,
            standard_normals=normals[:, :, step_index].reshape(
                num_states * num_paths,
                normals.shape[-1],
            ),
        )
        current_blocks = current.reshape(
            num_states,
            num_paths,
            num_expiries,
            block_width,
        )
        cap_margin = (
            torch.as_tensor(
                geometry.maximum_log_dlvsq,
                dtype=raw_states.dtype,
                device=raw_states.device,
            )
            - current_blocks[:, :, 1:, :num_inner]
        )
        minimum_cap_margin = torch.minimum(minimum_cap_margin, torch.min(cap_margin))
        if not bool(torch.all(cap_margin > 0.0)):
            raise RuntimeError("an Attempt 2.0 SDE path reached the pricing DLV cap")

    endpoint_parameters = bound_native_raw(current, geometry=geometry)
    endpoint_surface = decoder.forward_batch(
        dlvsqs=endpoint_parameters.dlvsqs,
        forward_vols=endpoint_parameters.forward_vols,
        eta=endpoint_parameters.eta,
    )
    endpoint_support = endpoint_surface.support_strikes.reshape(
        num_states,
        num_paths,
        num_expiries,
        support_size,
    )
    initial_support = initial_surface.support_strikes[:, 0]
    expected_support = initial_support[:, None, None].expand_as(endpoint_support)
    maximum_fixed_grid_change = torch.max(torch.abs(endpoint_support - expected_support))
    if not bool(maximum_fixed_grid_change == 0.0):
        raise RuntimeError("frozen forward volatilities did not preserve the cash grid")

    front_probabilities = initial_surface.densities[:, None, 0].expand(
        num_states,
        num_paths,
        support_size,
    )
    front_nodes = _sample_categorical(front_probabilities, uniforms)
    path_support = initial_support[:, None].expand(num_states, num_paths, support_size)
    normalized_spot = torch.gather(
        path_support,
        dim=-1,
        index=front_nodes.unsqueeze(-1),
    ).squeeze(-1)
    endpoint_dlvsqs = endpoint_parameters.dlvsqs.reshape(
        num_states,
        num_paths,
        num_expiries,
        num_inner,
    )
    aligned_densities = fixed_grid_attempt2_conditional_densities(
        decoder,
        support=path_support,
        endpoint_dlvsqs=endpoint_dlvsqs,
        front_node_indices=front_nodes,
    )
    conditional_mass_residual = torch.sum(aligned_densities, dim=-1) - 1.0
    conditional_forward_residual = torch.sum(
        aligned_densities * path_support[:, :, None],
        dim=-1,
    ) - normalized_spot[:, :, None]

    payoffs = _call_option_payoffs(initial_support, fixed_cash_strikes)
    initial_options = torch.einsum(
        "bts,bsk->btk",
        initial_surface.densities,
        payoffs,
    )
    aligned_options = torch.einsum(
        "bpts,bsk->bptk",
        aligned_densities,
        payoffs,
    )
    endpoint_densities = endpoint_surface.densities.reshape(
        num_states,
        num_paths,
        num_expiries,
        support_size,
    )
    endpoint_survivor_options = torch.einsum(
        "bpts,bsk->bptk",
        endpoint_densities[:, :, 1:],
        payoffs,
    )

    initial_blocks = raw_states.reshape(
        num_states,
        1,
        num_expiries,
        block_width,
    )
    endpoint_blocks = current.reshape(
        num_states,
        num_paths,
        num_expiries,
        block_width,
    )
    frozen_change = torch.cat(
        (
            (endpoint_blocks[:, :, :1] - initial_blocks[:, :, :1]).reshape(-1),
            (
                endpoint_blocks[..., num_inner:]
                - initial_blocks[..., num_inner:]
            ).reshape(-1),
        )
    )
    return FixedGridAttempt2Roll(
        normalized_spot=normalized_spot,
        front_node_indices=front_nodes,
        endpoint_raw_states=current.reshape(
            num_states,
            num_paths,
            geometry.native_state_size,
        ),
        endpoint_unconditional_densities=endpoint_densities,
        aligned_conditional_densities=aligned_densities,
        initial_option_values=initial_options,
        endpoint_unconditional_survivor_option_values=endpoint_survivor_options,
        aligned_conditional_option_values=aligned_options,
        maximum_conditional_mass_residual=torch.max(
            torch.abs(conditional_mass_residual)
        ),
        maximum_conditional_forward_residual=torch.max(
            torch.abs(conditional_forward_residual)
        ),
        maximum_fixed_grid_change=maximum_fixed_grid_change,
        maximum_frozen_coordinate_change=torch.max(torch.abs(frozen_change)),
        maximum_volatility_equation_residual=maximum_volatility_residual,
        maximum_drift_equation_residual=maximum_drift_residual,
        maximum_density_diffusion_mass_residual=maximum_mass_diffusion_residual,
        maximum_density_diffusion_forward_residual=(
            maximum_forward_diffusion_residual
        ),
        minimum_density_during_sde=minimum_density,
        minimum_raw_cap_margin=minimum_cap_margin,
    )


__all__ = [
    "FixedGridAttempt2Randoms",
    "FixedGridAttempt2Roll",
    "fixed_grid_attempt2_conditional_densities",
    "fixed_grid_attempt2_sde_roll",
    "make_fixed_grid_attempt2_randoms",
]
