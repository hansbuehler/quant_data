"""Calendar-time drift restriction for the discrete local-variance state.

This module implements the fixed-cash-strike construction in ``dlv_spine.pdf``.
It is deliberately separate from the native SANOS expiry transition: here the
densities and local variances diffuse in calendar time while the expiry dates
and cash support remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

import torch

from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.sanos import FlexSanosDecoder


@dataclass(frozen=True)
class FixedContractLvsqDrift:
    """Direct raw-state drift, diffusion, and equation diagnostics."""

    raw_drift: torch.Tensor
    raw_diffusion: torch.Tensor
    dlvsq_drift: torch.Tensor
    dlvsq_diffusion: torch.Tensor
    densities: torch.Tensor
    density_diffusion: torch.Tensor
    relative_density_diffusion: torch.Tensor
    maximum_volatility_equation_residual: torch.Tensor
    maximum_drift_equation_residual: torch.Tensor
    maximum_density_diffusion_mass_residual: torch.Tensor
    maximum_density_diffusion_forward_residual: torch.Tensor
    minimum_density: torch.Tensor
    maximum_absolute_relative_density_diffusion: torch.Tensor
    minimum_raw_cap_margin: torch.Tensor


@dataclass(frozen=True)
class FixedHorizonLvsqStep:
    """Finite-cubature exponential step with exact mean-density matching."""

    raw_states_by_branch: torch.Tensor
    branch_probabilities: torch.Tensor
    standardized_innovations: torch.Tensor
    densities_by_branch: torch.Tensor
    dlvsqs_by_branch: torch.Tensor
    effective_raw_drift: torch.Tensor
    local_raw_drift: torch.Tensor
    newton_iterations_by_expiry: tuple[int, ...]
    maximum_density_mean_residual: torch.Tensor
    maximum_density_dlvsq_mean_residual: torch.Tensor
    maximum_decoder_density_residual: torch.Tensor
    maximum_frozen_coordinate_change: torch.Tensor
    maximum_effective_minus_local_raw_drift: torch.Tensor
    minimum_raw_cap_margin: torch.Tensor


@dataclass(frozen=True)
class DensityMartingaleLvsqStep:
    """Symmetric density branches with analytically recovered DLV states."""

    raw_states_by_branch: torch.Tensor
    branch_probabilities: torch.Tensor
    standardized_innovations: torch.Tensor
    densities_by_branch: torch.Tensor
    dlvsqs_by_branch: torch.Tensor
    local_raw_drift: torch.Tensor
    maximum_density_mean_residual: torch.Tensor
    maximum_decoder_density_residual: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_forward_residual: torch.Tensor
    maximum_first_order_dlvsq_diffusion_residual: torch.Tensor
    root_mean_square_first_order_dlvsq_diffusion_residual: torch.Tensor
    relative_root_mean_square_dlvsq_diffusion_residual: torch.Tensor
    minimum_density: torch.Tensor
    minimum_dlvsq: torch.Tensor
    maximum_dlvsq: torch.Tensor
    fallback_coordinate_fraction: torch.Tensor
    minimum_raw_cap_margin: torch.Tensor


@dataclass(frozen=True)
class RelativeStrikeDlvRegrid:
    """Raw DLV state recentered on the canonical relative-strike grid."""

    raw_states: torch.Tensor
    extrapolated_coordinate_fraction: torch.Tensor
    maximum_absolute_raw_change: torch.Tensor


def regrid_raw_dlvsq_to_relative_spot(
    raw_states: torch.Tensor,
    gross_spot_return: torch.Tensor,
    *,
    geometry: FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
) -> RelativeStrikeDlvRegrid:
    r"""Recenter raw log-DLV curves after a realized spot move.

    A canonical new relative strike ``k`` is the old relative strike
    ``gross_spot_return * k``.  The raw log excess-DLV curve is linearly
    interpolated in log strike.  Queries outside the calibrated inner grid use
    flat endpoint extrapolation; the reported fraction makes that approximation
    visible to simulation audits.

    Forward-volatility and eta coordinates are copied bit-for-bit.  This is a
    coordinate regrid, not an expiry roll and not a refit of the SANOS density.
    """

    if raw_states.ndim != 2 or raw_states.shape[-1] != geometry.native_state_size:
        raise ValueError("raw_states must be a batch of complete native states")
    if gross_spot_return.shape != raw_states.shape[:1]:
        raise ValueError("gross_spot_return must contain one value per state")
    if not bool(
        torch.all(torch.isfinite(gross_spot_return))
        and torch.all(gross_spot_return > 0.0)
    ):
        raise ValueError("gross_spot_return must be finite and positive")

    num_states = len(raw_states)
    num_expiries = geometry.num_model_expiries
    num_inner = geometry.num_inner_strikes
    block_width = geometry.expiry_block_state_size
    log_grid = torch.log(
        torch.as_tensor(
            geometry.inner_relative_strikes,
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
    )
    query = log_grid[None] + torch.log(gross_spot_return)[:, None]
    extrapolated = (query < log_grid[0]) | (query > log_grid[-1])
    upper = torch.searchsorted(log_grid, query).clamp(1, num_inner - 1)
    lower = upper - 1
    lower_grid = log_grid[lower]
    upper_grid = log_grid[upper]
    weight = ((query - lower_grid) / (upper_grid - lower_grid)).clamp(0.0, 1.0)

    blocks = raw_states.reshape(num_states, num_expiries, block_width)
    raw_dlvsq = blocks[..., :num_inner]
    lower_values = torch.gather(
        raw_dlvsq,
        dim=2,
        index=lower[:, None].expand(-1, num_expiries, -1),
    )
    upper_values = torch.gather(
        raw_dlvsq,
        dim=2,
        index=upper[:, None].expand(-1, num_expiries, -1),
    )
    regridded_dlvsq = torch.lerp(
        lower_values,
        upper_values,
        weight[:, None],
    )
    regridded_blocks = blocks.clone()
    regridded_blocks[..., :num_inner] = regridded_dlvsq
    result = regridded_blocks.reshape_as(raw_states)
    return RelativeStrikeDlvRegrid(
        raw_states=result,
        extrapolated_coordinate_fraction=torch.mean(extrapolated.to(raw_states.dtype)),
        maximum_absolute_raw_change=torch.max(torch.abs(result - raw_states)),
    )


def symmetric_rademacher_cubature(
    num_factors: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate equally weighted independent signed unit innovations."""

    if num_factors <= 0 or num_factors > 12:
        raise ValueError("num_factors must lie between one and twelve")
    num_branches = 2**num_factors
    branch_indices = torch.arange(num_branches, device=device)
    bit_indices = torch.arange(num_factors, device=device)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(branch_indices[:, None], bit_indices[None]),
        1,
    )
    innovations = torch.where(bits == 0, -1.0, 1.0).to(dtype=dtype)
    probabilities = torch.full(
        (num_branches,),
        1.0 / num_branches,
        dtype=dtype,
        device=device,
    )
    return innovations, probabilities


def discrete_lvsq_grid_operator_product(
    support: torch.Tensor,
    interior_values: torch.Tensor,
    *,
    expiry_time_increment_y: torch.Tensor,
) -> torch.Tensor:
    r"""Apply the state-independent matrix ``G-hat`` on an irregular grid.

    ``support`` has shape ``(..., N+2)``. ``interior_values`` may have shape
    ``(..., N)`` or ``(..., N, F)``; the returned tensor has the corresponding
    ``(..., N+2)`` or ``(..., N+2, F)`` shape. The expiry increment belongs to
    the SANOS recursion and is distinct from a calendar-time SDE step.
    """

    if support.ndim < 1 or support.shape[-1] < 3:
        raise ValueError("support must end in at least three nodes")
    if expiry_time_increment_y.ndim != 0 or not bool(expiry_time_increment_y > 0.0):
        raise ValueError("expiry_time_increment_y must be a positive scalar")
    num_inner = support.shape[-1] - 2
    scalar_values = interior_values.shape == support.shape[:-1] + (num_inner,)
    factor_values = (
        interior_values.ndim == support.ndim + 1
        and interior_values.shape[:-2] == support.shape[:-1]
        and interior_values.shape[-2] == num_inner
    )
    if not scalar_values and not factor_values:
        raise ValueError(
            "interior values must align with support and contain N interior nodes"
        )
    values = interior_values.unsqueeze(-1) if scalar_values else interior_values
    if not bool(torch.all(torch.diff(support, dim=-1) > 0.0)):
        raise ValueError("support must be strictly increasing")

    inner_support = support[..., 1:-1]
    omega = (
        inner_support.square()
        * expiry_time_increment_y
        / (support[..., 2:] - support[..., :-2])
    )
    iota_minus = omega / (inner_support - support[..., :-2])
    iota_plus = omega / (support[..., 2:] - inner_support)
    result = torch.zeros(
        support.shape + (values.shape[-1],),
        dtype=values.dtype,
        device=values.device,
    )
    result[..., 0, :] = -iota_minus[..., :1] * values[..., 0, :]
    result[..., 1:-1, :] = (
        (iota_minus + iota_plus).unsqueeze(-1) * values
    )
    if num_inner > 1:
        result[..., 1:-2, :] -= (
            iota_minus[..., 1:].unsqueeze(-1) * values[..., 1:, :]
        )
        result[..., 2:-1, :] -= (
            iota_plus[..., :-1].unsqueeze(-1) * values[..., :-1, :]
        )
    result[..., -1, :] = -iota_plus[..., -1:] * values[..., -1, :]
    return result.squeeze(-1) if scalar_values else result


def smooth_lvsq_log_loadings(
    support: torch.Tensor,
    *,
    num_model_expiries: int,
    annualized_log_volatility: float,
) -> torch.Tensor:
    r"""Return common level/skew/curvature/expiry-slope raw-DLV loadings.

    The first expiry block is exactly zero because the PDF treats its density
    as the modeled initial condition and has no first-slab local variance.
    Remaining rows have four common Brownian factors and a sine taper in strike.
    """

    if support.ndim != 1 or len(support) < 7:
        raise ValueError("support must be a one-dimensional grid with at least five interiors")
    if num_model_expiries < 2:
        raise ValueError("num_model_expiries must be at least two")
    if not torch.isfinite(torch.as_tensor(annualized_log_volatility)) or (
        annualized_log_volatility <= 0.0
    ):
        raise ValueError("annualized_log_volatility must be finite and positive")
    if not bool(torch.all(support > 0.0)) or not bool(
        torch.all(torch.diff(support) > 0.0)
    ):
        raise ValueError("support must be positive and strictly increasing")

    inner = support[1:-1]
    maximum_absolute_log_strike = torch.max(torch.abs(torch.log(inner)))
    if not bool(maximum_absolute_log_strike > 0.0):
        raise ValueError("support must contain at least one non-ATM interior strike")
    relative_log_strike = torch.log(inner) / maximum_absolute_log_strike
    expiry_coordinate = torch.linspace(
        -1.0,
        1.0,
        num_model_expiries - 1,
        dtype=support.dtype,
        device=support.device,
    )
    strike_coordinate = relative_log_strike.view(1, -1).expand(
        num_model_expiries - 1,
        -1,
    )
    expiry_coordinate = expiry_coordinate[:, None].expand_as(strike_coordinate)
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
            len(inner) + 1,
            dtype=support.dtype,
            device=support.device,
        )
        * pi
        / (len(inner) + 1)
    )
    loadings = torch.zeros(
        (num_model_expiries, len(inner), 4),
        dtype=support.dtype,
        device=support.device,
    )
    loadings[1:] = annualized_log_volatility * taper[None, :, None] * factors
    return loadings


def fixed_contract_lvsq_drift(
    decoder: FlexSanosDecoder,
    raw_states: torch.Tensor,
    *,
    raw_log_loadings: torch.Tensor,
) -> FixedContractLvsqDrift:
    r"""Compute the PDF drift directly in the original native raw state.

    Only DLV coordinates in expiry blocks 2,...,M receive a drift or diffusion.
    The complete first block and both non-DLV coordinates in every block remain
    fixed. The current decoder must use one fixed support at every expiry.
    """

    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("direct raw LVsq drift requires capped log-DLV coordinates")
    num_expiries = geometry.num_model_expiries
    num_inner = geometry.num_inner_strikes
    block_width = geometry.expiry_block_state_size
    if raw_states.ndim != 2 or raw_states.shape[-1] != geometry.native_state_size:
        raise ValueError("raw_states must be a batch of complete native states")
    if raw_log_loadings.ndim == 3:
        if raw_log_loadings.shape[:2] != (num_expiries, num_inner):
            raise ValueError("shared raw_log_loadings have an invalid shape")
        loadings = raw_log_loadings.unsqueeze(0).expand(len(raw_states), -1, -1, -1)
    elif raw_log_loadings.ndim == 4:
        if raw_log_loadings.shape[:3] != (len(raw_states), num_expiries, num_inner):
            raise ValueError("batched raw_log_loadings have an invalid shape")
        loadings = raw_log_loadings
    else:
        raise ValueError("raw_log_loadings must have three or four dimensions")
    if loadings.shape[-1] <= 0 or not bool(torch.all(torch.isfinite(loadings))):
        raise ValueError("raw_log_loadings must contain finite Brownian factors")
    if not bool(torch.all(loadings[:, 0] == 0.0)):
        raise ValueError("the first-expiry raw DLV loading must be exactly zero")

    raw_blocks = raw_states.reshape(len(raw_states), num_expiries, block_width)
    dynamic_raw_dlvsq = raw_blocks[:, 1:, :num_inner]
    minimum_cap_margin = torch.min(
        torch.as_tensor(
            geometry.maximum_log_dlvsq,
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
        - dynamic_raw_dlvsq
    )
    if not bool(minimum_cap_margin > 0.0):
        raise RuntimeError(
            "a pricing DLV cap is active; the raw-coordinate drift identity is invalid"
        )

    parameters = bound_native_raw(raw_states, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    support = surface.support_strikes
    reference_support = support[:, :1]
    if not bool(torch.all(support == reference_support)):
        raise ValueError("the PDF drift requires one fixed cash support at every expiry")

    num_factors = loadings.shape[-1]
    density_diffusion = torch.zeros(
        surface.densities.shape + (num_factors,),
        dtype=raw_states.dtype,
        device=raw_states.device,
    )
    relative_density_diffusion = torch.zeros_like(density_diffusion)
    raw_drift_blocks = torch.zeros_like(raw_blocks)
    raw_diffusion_blocks = torch.zeros(
        raw_blocks.shape + (num_factors,),
        dtype=raw_states.dtype,
        device=raw_states.device,
    )
    dlvsq_drift = torch.zeros_like(parameters.dlvsqs)
    dlvsq_diffusion = torch.zeros(
        parameters.dlvsqs.shape + (num_factors,),
        dtype=raw_states.dtype,
        device=raw_states.device,
    )

    previous_density_diffusion = density_diffusion[:, 0]
    volatility_residuals: list[torch.Tensor] = []
    drift_residuals: list[torch.Tensor] = []
    for expiry_index in range(1, num_expiries):
        expiry_dt = decoder.model_dts_y[expiry_index]
        expiry_support = support[:, expiry_index]
        density = surface.densities[:, expiry_index]
        dlvsq = parameters.dlvsqs[:, expiry_index]
        raw_loading = loadings[:, expiry_index]
        excess_dlvsq = dlvsq - geometry.min_dlvsq
        physical_diffusion = excess_dlvsq.unsqueeze(-1) * raw_loading
        source = discrete_lvsq_grid_operator_product(
            expiry_support,
            density[:, 1:-1].unsqueeze(-1) * physical_diffusion,
            expiry_time_increment_y=expiry_dt,
        )
        rhs = previous_density_diffusion - source

        batch_size = len(raw_states)
        support_size = expiry_support.shape[-1]
        flat_support = (
            expiry_support[:, None]
            .expand(-1, num_factors, -1)
            .reshape(batch_size * num_factors, support_size)
        )
        flat_dlvsq = (
            dlvsq[:, None]
            .expand(-1, num_factors, -1)
            .reshape(batch_size * num_factors, num_inner)
        )
        flat_rhs = rhs.permute(0, 2, 1).reshape(
            batch_size * num_factors,
            support_size,
        )
        current_density_diffusion = decoder.solve_density_step_batch(
            support=flat_support,
            dlvsqs=flat_dlvsq,
            time_increment_y=expiry_dt,
            rhs_density=flat_rhs,
        ).reshape(batch_size, num_factors, support_size).permute(0, 2, 1)
        relative_diffusion = current_density_diffusion / density.unsqueeze(-1)

        physical_drift = -torch.sum(
            relative_diffusion[:, 1:-1] * physical_diffusion,
            dim=-1,
        )
        raw_drift = (
            physical_drift / excess_dlvsq
            - 0.5 * torch.sum(raw_loading.square(), dim=-1)
        )
        raw_drift_blocks[:, expiry_index, :num_inner] = raw_drift
        raw_diffusion_blocks[:, expiry_index, :num_inner] = raw_loading
        dlvsq_drift[:, expiry_index] = physical_drift
        dlvsq_diffusion[:, expiry_index] = physical_diffusion
        density_diffusion[:, expiry_index] = current_density_diffusion
        relative_density_diffusion[:, expiry_index] = relative_diffusion

        lhs = current_density_diffusion + discrete_lvsq_grid_operator_product(
            expiry_support,
            current_density_diffusion[:, 1:-1] * dlvsq.unsqueeze(-1),
            expiry_time_increment_y=expiry_dt,
        )
        volatility_residuals.append(torch.max(torch.abs(lhs - rhs)))
        drift_residuals.append(
            torch.max(
                torch.abs(
                    density[:, 1:-1] * physical_drift
                    + torch.sum(
                        current_density_diffusion[:, 1:-1] * physical_diffusion,
                        dim=-1,
                    )
                )
            )
        )
        previous_density_diffusion = current_density_diffusion

    mass_diffusion_residual = torch.max(torch.abs(torch.sum(density_diffusion, dim=2)))
    forward_diffusion_residual = torch.max(
        torch.abs(torch.sum(density_diffusion * support.unsqueeze(-1), dim=2))
    )
    return FixedContractLvsqDrift(
        raw_drift=raw_drift_blocks.reshape_as(raw_states),
        raw_diffusion=raw_diffusion_blocks.reshape(
            len(raw_states),
            geometry.native_state_size,
            num_factors,
        ),
        dlvsq_drift=dlvsq_drift,
        dlvsq_diffusion=dlvsq_diffusion,
        densities=surface.densities,
        density_diffusion=density_diffusion,
        relative_density_diffusion=relative_density_diffusion,
        maximum_volatility_equation_residual=torch.max(
            torch.stack(volatility_residuals)
        ),
        maximum_drift_equation_residual=torch.max(torch.stack(drift_residuals)),
        maximum_density_diffusion_mass_residual=mass_diffusion_residual,
        maximum_density_diffusion_forward_residual=forward_diffusion_residual,
        minimum_density=torch.min(surface.densities[:, 1:]),
        maximum_absolute_relative_density_diffusion=torch.max(
            torch.abs(relative_density_diffusion[:, 1:])
        ),
        minimum_raw_cap_margin=minimum_cap_margin,
    )


def fixed_contract_lvsq_euler_step(
    drift: FixedContractLvsqDrift,
    raw_states: torch.Tensor,
    *,
    calendar_time_increment_y: float,
    standard_normals: torch.Tensor,
) -> torch.Tensor:
    """Apply one Euler--Maruyama step with common Brownian factors."""

    if calendar_time_increment_y <= 0.0:
        raise ValueError("calendar_time_increment_y must be positive")
    if standard_normals.ndim != 2 or standard_normals.shape != (
        len(raw_states),
        drift.raw_diffusion.shape[-1],
    ):
        raise ValueError("standard_normals must provide one common factor draw per state")
    return (
        raw_states
        + calendar_time_increment_y * drift.raw_drift
        + calendar_time_increment_y**0.5
        * torch.einsum("bsf,bf->bs", drift.raw_diffusion, standard_normals)
    )


def fixed_horizon_lvsq_exponential_step(
    decoder: FlexSanosDecoder,
    raw_states: torch.Tensor,
    *,
    raw_log_loadings: torch.Tensor,
    calendar_time_increment_y: float,
    standardized_innovations: torch.Tensor | None = None,
    branch_probabilities: torch.Tensor | None = None,
    maximum_newton_iterations: int = 12,
    maximum_line_search_steps: int = 12,
    moment_tolerance: float = 1.0e-12,
) -> FixedHorizonLvsqStep:
    r"""Take one exponential DLV step that martingalizes the whole density.

    For each dynamic expiry this solves for a deterministic effective log-DLV
    drift ``a_h`` in

    ``z_next = z + h*a_h + sqrt(h)*C*epsilon``

    such that the cubature expectation of ``q_next * dlvsq_next`` equals its
    current value.  The pathwise SANOS equation and the already-matched mean of
    the preceding density then imply ``E[q_next] = q_current`` exactly.  The
    solve is triangular in expiry and uses equation (15)'s local drift as its
    Newton initial value.
    """

    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("finite-h LVsq stepping requires capped log-DLV coordinates")
    if not isfinite(calendar_time_increment_y) or calendar_time_increment_y <= 0.0:
        raise ValueError("calendar_time_increment_y must be finite and positive")
    if maximum_newton_iterations <= 0 or maximum_line_search_steps <= 0:
        raise ValueError("Newton and line-search iteration counts must be positive")
    if not isfinite(moment_tolerance) or moment_tolerance <= 0.0:
        raise ValueError("moment_tolerance must be finite and positive")

    num_states = len(raw_states)
    num_expiries = geometry.num_model_expiries
    num_inner = geometry.num_inner_strikes
    block_width = geometry.expiry_block_state_size
    if raw_states.ndim != 2 or raw_states.shape != (
        num_states,
        geometry.native_state_size,
    ):
        raise ValueError("raw_states must be a batch of complete native states")
    if raw_log_loadings.ndim == 3:
        if raw_log_loadings.shape[:2] != (num_expiries, num_inner):
            raise ValueError("shared raw_log_loadings have an invalid shape")
        loadings = raw_log_loadings.unsqueeze(0).expand(num_states, -1, -1, -1)
    elif raw_log_loadings.ndim == 4:
        if raw_log_loadings.shape[:3] != (num_states, num_expiries, num_inner):
            raise ValueError("batched raw_log_loadings have an invalid shape")
        loadings = raw_log_loadings
    else:
        raise ValueError("raw_log_loadings must have three or four dimensions")
    num_factors = loadings.shape[-1]
    if num_factors <= 0 or not bool(torch.all(torch.isfinite(loadings))):
        raise ValueError("raw_log_loadings must contain finite Brownian factors")
    if not bool(torch.all(loadings[:, 0] == 0.0)):
        raise ValueError("the first-expiry raw DLV loading must be exactly zero")

    if standardized_innovations is None:
        if branch_probabilities is not None:
            raise ValueError("branch probabilities require explicit innovations")
        innovations, probabilities = symmetric_rademacher_cubature(
            num_factors,
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
    else:
        innovations = standardized_innovations.to(
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
        if innovations.ndim != 2 or innovations.shape[-1] != num_factors:
            raise ValueError("standardized innovations have an invalid shape")
        if branch_probabilities is None:
            raise ValueError("explicit innovations require branch probabilities")
        probabilities = branch_probabilities.to(
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
    if probabilities.shape != (len(innovations),):
        raise ValueError("branch probabilities must align with innovations")
    if not bool(
        torch.all(torch.isfinite(innovations))
        and torch.all(torch.isfinite(probabilities))
        and torch.all(probabilities > 0.0)
    ):
        raise ValueError("cubature innovations and probabilities must be finite")
    probability_tolerance = 128.0 * torch.finfo(raw_states.dtype).eps
    if not bool(torch.abs(torch.sum(probabilities) - 1.0) <= probability_tolerance):
        raise ValueError("branch probabilities must sum to one")
    innovation_mean = torch.sum(probabilities[:, None] * innovations, dim=0)
    innovation_covariance = torch.einsum(
        "r,rf,rg->fg",
        probabilities,
        innovations,
        innovations,
    )
    if not bool(
        torch.max(torch.abs(innovation_mean)) <= probability_tolerance
        and torch.max(
            torch.abs(
                innovation_covariance
                - torch.eye(
                    num_factors,
                    dtype=raw_states.dtype,
                    device=raw_states.device,
                )
            )
        )
        <= probability_tolerance
    ):
        raise ValueError("cubature innovations must have zero mean and unit covariance")

    local = fixed_contract_lvsq_drift(
        decoder,
        raw_states,
        raw_log_loadings=loadings,
    )
    raw_blocks = raw_states.reshape(num_states, num_expiries, block_width)
    local_drift_blocks = local.raw_drift.reshape_as(raw_blocks)
    parameters = bound_native_raw(raw_states, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    support = surface.support_strikes
    if not bool(torch.all(support == support[:, :1])):
        raise ValueError("finite-h LVsq stepping requires one fixed cash support")

    num_branches = len(innovations)
    sqrt_h = calendar_time_increment_y**0.5
    maximum_log_dlvsq = torch.as_tensor(
        geometry.maximum_log_dlvsq,
        dtype=raw_states.dtype,
        device=raw_states.device,
    )
    branch_raw_blocks = (
        raw_blocks[:, None]
        .expand(num_states, num_branches, num_expiries, block_width)
        .clone()
    )
    effective_drift_blocks = torch.zeros_like(raw_blocks)
    density_branches: list[torch.Tensor] = [
        surface.densities[:, None, 0].expand(num_states, num_branches, -1)
    ]
    dlvsq_branches: list[torch.Tensor] = [
        parameters.dlvsqs[:, None, 0].expand(num_states, num_branches, -1)
    ]
    iterations_by_expiry: list[int] = []
    minimum_cap_margin = torch.as_tensor(
        float("inf"), dtype=raw_states.dtype, device=raw_states.device
    )

    def evaluate_expiry(
        *,
        expiry_index: int,
        effective_drift: torch.Tensor,
        previous_density: torch.Tensor,
        with_jacobian: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        base_raw = raw_blocks[:, expiry_index, :num_inner]
        base_excess = parameters.dlvsqs[:, expiry_index] - geometry.min_dlvsq
        shock = torch.einsum(
            "bnf,rf->brn",
            loadings[:, expiry_index],
            innovations,
        )
        branch_raw = (
            base_raw[:, None]
            + calendar_time_increment_y * effective_drift[:, None]
            + sqrt_h * shock
        )
        cap_margin_by_state = torch.amin(
            maximum_log_dlvsq - branch_raw,
            dim=(1, 2),
        )
        branch_excess = base_excess[:, None] * torch.exp(
            calendar_time_increment_y * effective_drift[:, None]
            + sqrt_h * shock
        )
        branch_dlvsq = geometry.min_dlvsq + branch_excess
        expiry_support = support[:, expiry_index]
        flat_support = (
            expiry_support[:, None]
            .expand(-1, num_branches, -1)
            .reshape(num_states * num_branches, num_inner + 2)
        )
        flat_dlvsq = branch_dlvsq.reshape(num_states * num_branches, num_inner)
        flat_previous = previous_density.reshape(
            num_states * num_branches,
            num_inner + 2,
        )
        branch_density = decoder.solve_density_step_batch(
            support=flat_support,
            dlvsqs=flat_dlvsq,
            time_increment_y=decoder.model_dts_y[expiry_index],
            rhs_density=flat_previous,
        ).reshape(num_states, num_branches, num_inner + 2)
        density_dlvsq = branch_density[..., 1:-1] * branch_dlvsq
        target_density_dlvsq = (
            surface.densities[:, expiry_index, 1:-1]
            * parameters.dlvsqs[:, expiry_index]
        )
        residual = torch.sum(
            probabilities[None, :, None] * density_dlvsq,
            dim=1,
        ) - target_density_dlvsq
        if not with_jacobian:
            return (
                branch_raw,
                branch_dlvsq,
                branch_density,
                None,
                residual,
                cap_margin_by_state,
            )

        dlvsq_derivative_diagonal = torch.diag_embed(
            branch_density[..., 1:-1]
            * calendar_time_increment_y
            * branch_excess
        )
        density_rhs_derivative = -discrete_lvsq_grid_operator_product(
            expiry_support[:, None].expand(-1, num_branches, -1),
            dlvsq_derivative_diagonal,
            expiry_time_increment_y=decoder.model_dts_y[expiry_index],
        )
        jacobian_support = (
            expiry_support[:, None, None]
            .expand(-1, num_branches, num_inner, -1)
            .reshape(num_states * num_branches * num_inner, num_inner + 2)
        )
        jacobian_dlvsq = (
            branch_dlvsq[:, :, None]
            .expand(-1, -1, num_inner, -1)
            .reshape(num_states * num_branches * num_inner, num_inner)
        )
        jacobian_rhs = density_rhs_derivative.permute(0, 1, 3, 2).reshape(
            num_states * num_branches * num_inner,
            num_inner + 2,
        )
        density_derivative = decoder.solve_density_step_batch(
            support=jacobian_support,
            dlvsqs=jacobian_dlvsq,
            time_increment_y=decoder.model_dts_y[expiry_index],
            rhs_density=jacobian_rhs,
        ).reshape(
            num_states,
            num_branches,
            num_inner,
            num_inner + 2,
        ).permute(0, 1, 3, 2)
        branch_jacobian = (
            branch_dlvsq[..., :, None] * density_derivative[..., 1:-1, :]
            + dlvsq_derivative_diagonal
        )
        jacobian = torch.sum(
            probabilities[None, :, None, None] * branch_jacobian,
            dim=1,
        )
        return (
            branch_raw,
            branch_dlvsq,
            branch_density,
            jacobian,
            residual,
            cap_margin_by_state,
        )

    previous_density = density_branches[0]
    for expiry_index in range(1, num_expiries):
        effective_drift = local_drift_blocks[:, expiry_index, :num_inner].clone()
        converged = False
        final_values: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        for iteration in range(1, maximum_newton_iterations + 1):
            values = evaluate_expiry(
                expiry_index=expiry_index,
                effective_drift=effective_drift,
                previous_density=previous_density,
                with_jacobian=True,
            )
            _, _, _, jacobian, residual, cap_margin_by_state = values
            assert jacobian is not None
            residual_norm = torch.amax(torch.abs(residual), dim=1)
            if bool(torch.all(residual_norm <= moment_tolerance)):
                converged = True
                final_values = values
                iterations_by_expiry.append(iteration - 1)
                break
            solution = torch.linalg.solve_ex(
                jacobian,
                residual.unsqueeze(-1),
            )
            if not bool(torch.all(solution.info == 0)):
                raise RuntimeError(
                    f"finite-h LVsq Newton matrix is singular at expiry {expiry_index}"
                )
            newton_step = solution.result.squeeze(-1)
            accepted = torch.zeros(
                num_states, dtype=torch.bool, device=raw_states.device
            )
            accepted_drift = effective_drift.clone()
            damping = torch.ones(
                num_states, dtype=raw_states.dtype, device=raw_states.device
            )
            for _ in range(maximum_line_search_steps):
                trial_drift = effective_drift - damping[:, None] * newton_step
                trial_values = evaluate_expiry(
                    expiry_index=expiry_index,
                    effective_drift=trial_drift,
                    previous_density=previous_density,
                    with_jacobian=False,
                )
                trial_residual = trial_values[4]
                trial_cap_margin = trial_values[5]
                trial_norm = torch.amax(torch.abs(trial_residual), dim=1)
                newly_accepted = (
                    ~accepted
                    & (trial_cap_margin > 0.0)
                    & (trial_norm < residual_norm)
                )
                accepted_drift[newly_accepted] = trial_drift[newly_accepted]
                accepted |= newly_accepted
                if bool(torch.all(accepted)):
                    break
                damping[~accepted] *= 0.5
            if not bool(torch.all(accepted)):
                worst = float(torch.max(residual_norm[~accepted]).detach().cpu())
                raise RuntimeError(
                    "finite-h LVsq Newton line search failed at expiry "
                    f"{expiry_index}; residual={worst:.6g}"
                )
            effective_drift = accepted_drift
            minimum_cap_margin = torch.minimum(
                minimum_cap_margin,
                torch.min(cap_margin_by_state),
            )
        if not converged or final_values is None:
            final_values = evaluate_expiry(
                expiry_index=expiry_index,
                effective_drift=effective_drift,
                previous_density=previous_density,
                with_jacobian=False,
            )
            worst = float(torch.max(torch.abs(final_values[4])).detach().cpu())
            raise RuntimeError(
                f"finite-h LVsq Newton did not converge at expiry {expiry_index}; "
                f"residual={worst:.6g}"
            )
        branch_raw, branch_dlvsq, branch_density, _, _, cap_margin_by_state = (
            final_values
        )
        minimum_cap_margin = torch.minimum(
            minimum_cap_margin,
            torch.min(cap_margin_by_state),
        )
        effective_drift_blocks[:, expiry_index, :num_inner] = effective_drift
        branch_raw_blocks[:, :, expiry_index, :num_inner] = branch_raw
        density_branches.append(branch_density)
        dlvsq_branches.append(branch_dlvsq)
        previous_density = branch_density

    raw_states_by_branch = branch_raw_blocks.reshape(
        num_states,
        num_branches,
        geometry.native_state_size,
    )
    densities_by_branch = torch.stack(density_branches, dim=2)
    dlvsqs_by_branch = torch.stack(dlvsq_branches, dim=2)
    flat_raw = raw_states_by_branch.reshape(
        num_states * num_branches,
        geometry.native_state_size,
    )
    decoded_parameters = bound_native_raw(flat_raw, geometry=geometry)
    decoded_surface = decoder.forward_batch(
        dlvsqs=decoded_parameters.dlvsqs,
        forward_vols=decoded_parameters.forward_vols,
        eta=decoded_parameters.eta,
    )
    decoded_density = decoded_surface.densities.reshape_as(densities_by_branch)
    density_mean = torch.sum(
        probabilities[None, :, None, None] * densities_by_branch,
        dim=1,
    )
    density_dlvsq_mean = torch.sum(
        probabilities[None, :, None, None]
        * densities_by_branch[..., 1:-1]
        * dlvsqs_by_branch,
        dim=1,
    )
    target_density_dlvsq = surface.densities[..., 1:-1] * parameters.dlvsqs
    front_change = branch_raw_blocks[:, :, :1] - raw_blocks[:, None, :1]
    non_dlv_change = (
        branch_raw_blocks[..., num_inner:] - raw_blocks[:, None, :, num_inner:]
    )
    return FixedHorizonLvsqStep(
        raw_states_by_branch=raw_states_by_branch,
        branch_probabilities=probabilities,
        standardized_innovations=innovations,
        densities_by_branch=densities_by_branch,
        dlvsqs_by_branch=dlvsqs_by_branch,
        effective_raw_drift=effective_drift_blocks.reshape_as(raw_states),
        local_raw_drift=local.raw_drift,
        newton_iterations_by_expiry=tuple(iterations_by_expiry),
        maximum_density_mean_residual=torch.max(
            torch.abs(density_mean[:, 1:] - surface.densities[:, 1:])
        ),
        maximum_density_dlvsq_mean_residual=torch.max(
            torch.abs(
                density_dlvsq_mean[:, 1:] - target_density_dlvsq[:, 1:]
            )
        ),
        maximum_decoder_density_residual=torch.max(
            torch.abs(decoded_density - densities_by_branch)
        ),
        maximum_frozen_coordinate_change=torch.maximum(
            torch.max(torch.abs(front_change)),
            torch.max(torch.abs(non_dlv_change)),
        ),
        maximum_effective_minus_local_raw_drift=torch.max(
            torch.abs(effective_drift_blocks - local_drift_blocks)
        ),
        minimum_raw_cap_margin=minimum_cap_margin,
    )


def density_martingale_lvsq_step(
    decoder: FlexSanosDecoder,
    raw_states: torch.Tensor,
    *,
    raw_log_loadings: torch.Tensor,
    calendar_time_increment_y: float,
    standardized_innovations: torch.Tensor | None = None,
    branch_probabilities: torch.Tensor | None = None,
    positivity_tolerance: float = 1.0e-12,
    density_mass_floor: float = 1.0e-14,
) -> DensityMartingaleLvsqStep:
    r"""Take a symmetric finite step in density space and recover DLVs.

    The risk-neutral density itself has zero calendar drift.  Therefore

    ``q_next(epsilon) = q + sqrt(h) * b_q * epsilon``

    has exactly the correct branch-weighted mean for any symmetric cubature.
    Each branch's DLVs are then recovered algebraically from the discrete SANOS
    call recursion.  This avoids both Euler weak drift and a finite-step Newton
    solve.  It is valid only while all branch densities and recovered DLVs stay
    inside their admissible domains.
    """

    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("density-martingale stepping requires native log-DLVs")
    if not isfinite(calendar_time_increment_y) or calendar_time_increment_y <= 0.0:
        raise ValueError("calendar_time_increment_y must be finite and positive")
    if not isfinite(positivity_tolerance) or positivity_tolerance <= 0.0:
        raise ValueError("positivity_tolerance must be finite and positive")
    if not isfinite(density_mass_floor) or density_mass_floor <= 0.0:
        raise ValueError("density_mass_floor must be finite and positive")
    local = fixed_contract_lvsq_drift(
        decoder,
        raw_states,
        raw_log_loadings=raw_log_loadings,
    )
    num_states = len(raw_states)
    num_expiries = geometry.num_model_expiries
    num_inner = geometry.num_inner_strikes
    block_width = geometry.expiry_block_state_size
    num_factors = local.raw_diffusion.shape[-1]
    if standardized_innovations is None:
        if branch_probabilities is not None:
            raise ValueError("branch probabilities require explicit innovations")
        innovations, probabilities = symmetric_rademacher_cubature(
            num_factors,
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
    else:
        innovations = standardized_innovations.to(
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
        if branch_probabilities is None:
            raise ValueError("explicit innovations require branch probabilities")
        probabilities = branch_probabilities.to(
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
    if innovations.ndim != 2 or innovations.shape[-1] != num_factors:
        raise ValueError("standardized innovations have an invalid shape")
    if probabilities.shape != (len(innovations),):
        raise ValueError("branch probabilities do not align with innovations")
    probability_tolerance = 128.0 * torch.finfo(raw_states.dtype).eps
    innovation_mean = torch.sum(probabilities[:, None] * innovations, dim=0)
    innovation_covariance = torch.einsum(
        "r,rf,rg->fg",
        probabilities,
        innovations,
        innovations,
    )
    if not bool(
        torch.all(torch.isfinite(innovations))
        and torch.all(torch.isfinite(probabilities) & (probabilities > 0.0))
        and torch.abs(torch.sum(probabilities) - 1.0) <= probability_tolerance
        and torch.max(torch.abs(innovation_mean)) <= probability_tolerance
        and torch.max(
            torch.abs(
                innovation_covariance
                - torch.eye(
                    num_factors,
                    dtype=raw_states.dtype,
                    device=raw_states.device,
                )
            )
        )
        <= probability_tolerance
    ):
        raise ValueError("cubature must have zero mean and unit covariance")

    parameters = bound_native_raw(raw_states, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    if not bool(torch.all(surface.support_strikes == surface.support_strikes[:, :1])):
        raise ValueError("density-martingale stepping requires fixed cash support")
    density_shock = torch.einsum(
        "btnf,rf->brtn",
        local.density_diffusion,
        innovations,
    )
    branch_densities = (
        surface.densities[:, None]
        + calendar_time_increment_y**0.5 * density_shock
    )
    minimum_density = torch.min(branch_densities)
    if not bool(minimum_density >= -positivity_tolerance):
        raise RuntimeError(
            "symmetric density branch became negative; reduce the calendar step "
            f"or loading scale (minimum={float(minimum_density.detach().cpu()):.6g})"
        )
    branch_densities = torch.clamp_min(branch_densities, 0.0)
    num_branches = len(innovations)
    support_size = num_inner + 2
    raw_blocks = raw_states.reshape(num_states, num_expiries, block_width)
    local_raw_drift_blocks = local.raw_drift.reshape_as(raw_blocks)
    local_raw_diffusion_blocks = local.raw_diffusion.reshape(
        num_states,
        num_expiries,
        block_width,
        num_factors,
    )
    local_branch_raw = (
        raw_blocks[:, None]
        + calendar_time_increment_y * local_raw_drift_blocks[:, None]
        + calendar_time_increment_y**0.5
        * torch.einsum(
            "btnf,rf->brtn",
            local_raw_diffusion_blocks,
            innovations,
        )
    )
    maximum_log_dlvsq = torch.as_tensor(
        geometry.maximum_log_dlvsq,
        dtype=raw_states.dtype,
        device=raw_states.device,
    )
    if not bool(
        torch.all(local_branch_raw[:, :, 1:, :num_inner] < maximum_log_dlvsq)
    ):
        raise RuntimeError("local fallback DLV branch reaches the pricing cap")
    branch_dlvsqs = parameters.dlvsqs[:, None].expand(
        num_states,
        num_branches,
        num_expiries,
        num_inner,
    ).clone()
    branch_dlvsqs[:, :, 1:] = geometry.min_dlvsq + torch.exp(
        local_branch_raw[:, :, 1:, :num_inner]
    )
    fallback_coordinates = torch.zeros(
        (),
        dtype=raw_states.dtype,
        device=raw_states.device,
    )
    dynamic_coordinates = num_states * num_branches * (num_expiries - 1) * num_inner
    for expiry_index in range(1, num_expiries):
        support = surface.support_strikes[:, expiry_index]
        strikes = support[:, 1:-1]
        payoff = torch.relu(support[:, None, :] - strikes[:, :, None])
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
        recovered = (
            current_calls - previous_calls
        ) / torch.clamp_min(denominator, torch.finfo(raw_states.dtype).tiny)
        defined = (
            current_density[..., 1:-1] > density_mass_floor
        ) & torch.isfinite(recovered)
        admissible = (
            defined
            & (recovered > geometry.min_dlvsq)
            & (recovered < geometry.max_dlvsq)
        )
        fallback_coordinates += torch.sum(~admissible)
        branch_dlvsqs[:, :, expiry_index] = torch.where(
            admissible,
            recovered,
            branch_dlvsqs[:, :, expiry_index],
        )

    dynamic_dlvsqs = branch_dlvsqs[:, :, 1:]
    minimum_dlvsq = torch.min(dynamic_dlvsqs)
    maximum_dlvsq = torch.max(dynamic_dlvsqs)
    if not bool(
        minimum_dlvsq > geometry.min_dlvsq
        and maximum_dlvsq < geometry.max_dlvsq
    ):
        raise RuntimeError(
            "recovered density-branch DLV lies outside its open bounds: "
            f"[{float(minimum_dlvsq.detach().cpu()):.6g}, "
            f"{float(maximum_dlvsq.detach().cpu()):.6g}]"
        )
    branch_raw_blocks = raw_blocks[:, None].expand(
        num_states,
        num_branches,
        num_expiries,
        block_width,
    ).clone()
    branch_raw_blocks[:, :, 1:, :num_inner] = torch.log(
        dynamic_dlvsqs - geometry.min_dlvsq
    )
    raw_states_by_branch = branch_raw_blocks.reshape(
        num_states,
        num_branches,
        geometry.native_state_size,
    )
    decoded = _decode_lvsq_branch_densities(
        decoder,
        raw_states_by_branch,
    )
    density_mean = torch.sum(
        probabilities.reshape(1, num_branches, 1, 1) * branch_densities,
        dim=1,
    )
    mass_residual = torch.sum(branch_densities, dim=-1) - 1.0
    forward_residual = torch.sum(
        branch_densities * surface.support_strikes[:, None],
        dim=-1,
    ) - 1.0
    centered_dlvsq = dynamic_dlvsqs - parameters.dlvsqs[:, None, 1:]
    recovered_diffusion = torch.einsum(
        "r,brtn,rf->btnf",
        probabilities,
        centered_dlvsq,
        innovations,
    ) / calendar_time_increment_y**0.5
    target_diffusion = local.dlvsq_diffusion[:, 1:]
    diffusion_error = recovered_diffusion - target_diffusion
    root_mean_square_diffusion_error = torch.sqrt(torch.mean(diffusion_error.square()))
    root_mean_square_target_diffusion = torch.sqrt(torch.mean(target_diffusion.square()))
    return DensityMartingaleLvsqStep(
        raw_states_by_branch=raw_states_by_branch,
        branch_probabilities=probabilities,
        standardized_innovations=innovations,
        densities_by_branch=branch_densities,
        dlvsqs_by_branch=branch_dlvsqs,
        local_raw_drift=local.raw_drift,
        maximum_density_mean_residual=torch.max(
            torch.abs(density_mean - surface.densities)
        ),
        maximum_decoder_density_residual=torch.max(
            torch.abs(decoded - branch_densities)
        ),
        maximum_mass_residual=torch.max(torch.abs(mass_residual)),
        maximum_forward_residual=torch.max(torch.abs(forward_residual)),
        maximum_first_order_dlvsq_diffusion_residual=torch.max(
            torch.abs(diffusion_error)
        ),
        root_mean_square_first_order_dlvsq_diffusion_residual=(
            root_mean_square_diffusion_error
        ),
        relative_root_mean_square_dlvsq_diffusion_residual=(
            root_mean_square_diffusion_error
            / torch.clamp_min(
                root_mean_square_target_diffusion,
                torch.finfo(raw_states.dtype).tiny,
            )
        ),
        minimum_density=minimum_density,
        minimum_dlvsq=minimum_dlvsq,
        maximum_dlvsq=maximum_dlvsq,
        fallback_coordinate_fraction=fallback_coordinates / dynamic_coordinates,
        minimum_raw_cap_margin=torch.min(
            maximum_log_dlvsq
            - branch_raw_blocks[:, :, 1:, :num_inner]
        ),
    )


def _decode_lvsq_branch_densities(
    decoder: FlexSanosDecoder,
    raw_states_by_branch: torch.Tensor,
) -> torch.Tensor:
    """Decode branch states without exposing their flattened batch layout."""

    geometry = decoder.geometry
    states, branches, native_size = raw_states_by_branch.shape
    flat = raw_states_by_branch.reshape(states * branches, native_size)
    parameters = bound_native_raw(flat, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    return surface.densities.reshape(
        states,
        branches,
        geometry.num_model_expiries,
        geometry.num_inner_strikes + 2,
    )


__all__ = [
    "FixedContractLvsqDrift",
    "DensityMartingaleLvsqStep",
    "FixedHorizonLvsqStep",
    "RelativeStrikeDlvRegrid",
    "discrete_lvsq_grid_operator_product",
    "density_martingale_lvsq_step",
    "fixed_contract_lvsq_drift",
    "fixed_contract_lvsq_euler_step",
    "fixed_horizon_lvsq_exponential_step",
    "regrid_raw_dlvsq_to_relative_spot",
    "smooth_lvsq_log_loadings",
    "symmetric_rademacher_cubature",
]
