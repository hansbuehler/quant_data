"""Shrinking fixed-expiry implementation of the Attempt 2.0 DLVsq SDE."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quantdata.dysanos.flex.lvsq_drift import discrete_lvsq_grid_operator_product
from quantdata.dysanos.flex.sanos import FlexSanosDecoder


@dataclass(frozen=True)
class TriangularFixedGridState:
    """Density/DLV tower for the fixed expiries which have not yet expired."""

    support: torch.Tensor
    densities: torch.Tensor
    dlvsqs: torch.Tensor


@dataclass(frozen=True)
class TriangularFixedGridRoll:
    """One weekly transition, aligned to every claim alive at its start."""

    next_state: TriangularFixedGridState | None
    revealed_spot: torch.Tensor
    initial_call_values: torch.Tensor
    endpoint_unconditional_call_values: torch.Tensor
    final_call_values: torch.Tensor
    minimum_density: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_forward_residual: torch.Tensor


def regrid_triangular_state_to_revealed_spot(
    state: TriangularFixedGridState,
    revealed_spot: torch.Tensor,
) -> TriangularFixedGridState:
    """Express a conditional cash-grid tower in new-spot-relative units."""

    if revealed_spot.shape != state.support.shape[:1] or not bool(
        torch.all(torch.isfinite(revealed_spot) & (revealed_spot > 0.0))
    ):
        raise ValueError("revealed_spot must be one positive finite value per path")
    return TriangularFixedGridState(
        support=state.support / revealed_spot[:, None],
        densities=state.densities,
        dlvsqs=state.dlvsqs,
    )


def _densities_from_front(
    decoder: FlexSanosDecoder,
    *,
    support: torch.Tensor,
    front_density: torch.Tensor,
    dlvsqs: torch.Tensor,
    expiry_time_increment_y: torch.Tensor,
) -> torch.Tensor:
    densities = [front_density]
    for expiry_index in range(1, dlvsqs.shape[1]):
        densities.append(
            decoder.solve_density_step_batch(
                support=support,
                dlvsqs=dlvsqs[:, expiry_index],
                time_increment_y=expiry_time_increment_y,
                rhs_density=densities[-1],
            )
        )
    return torch.stack(densities, dim=1)


def _drift(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return drift and diffusion of log(DLVsq-minimum) for the live tower."""

    batch, num_expiries, num_inner = state.dlvsqs.shape
    num_factors = raw_log_loadings.shape[-1]
    drift = torch.zeros_like(state.dlvsqs)
    diffusion = torch.zeros(
        (batch, num_expiries, num_inner, num_factors),
        dtype=state.dlvsqs.dtype,
        device=state.dlvsqs.device,
    )
    previous_density_diffusion = torch.zeros(
        (batch, num_inner + 2, num_factors),
        dtype=state.dlvsqs.dtype,
        device=state.dlvsqs.device,
    )
    for expiry_index in range(1, num_expiries):
        density = state.densities[:, expiry_index]
        dlvsq = state.dlvsqs[:, expiry_index]
        loading = raw_log_loadings[expiry_index]
        excess = dlvsq - minimum_dlvsq
        physical_diffusion = excess.unsqueeze(-1) * loading
        source = discrete_lvsq_grid_operator_product(
            state.support,
            density[:, 1:-1].unsqueeze(-1) * physical_diffusion,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        rhs = previous_density_diffusion - source
        flat_support = (
            state.support[:, None]
            .expand(-1, num_factors, -1)
            .reshape(batch * num_factors, num_inner + 2)
        )
        flat_dlvsq = (
            dlvsq[:, None].expand(-1, num_factors, -1).reshape(batch * num_factors, num_inner)
        )
        density_diffusion = (
            decoder.solve_density_step_batch(
                support=flat_support,
                dlvsqs=flat_dlvsq,
                time_increment_y=expiry_time_increment_y,
                rhs_density=rhs.permute(0, 2, 1).reshape(batch * num_factors, num_inner + 2),
            )
            .reshape(batch, num_factors, num_inner + 2)
            .permute(0, 2, 1)
        )
        interior_density = density[:, 1:-1]
        identified = interior_density > 0.0
        relative_density_diffusion = torch.where(
            identified[..., None],
            density_diffusion[:, 1:-1] / torch.where(identified, interior_density, 1.0)[..., None],
            0.0,
        )
        # At zero-probability nodes DLV is price-unidentified; the absorbing
        # convention avoids the undefined 0/0 without altering any payoff.
        physical_drift = -torch.sum(relative_density_diffusion * physical_diffusion, dim=-1)
        drift[:, expiry_index] = physical_drift / excess - 0.5 * torch.sum(loading.square(), dim=-1)
        diffusion[:, expiry_index] = loading
        previous_density_diffusion = density_diffusion
    return drift, diffusion


def _reveal_and_shrink_fixed_grid_roll(
    decoder: FlexSanosDecoder,
    initial_state: TriangularFixedGridState,
    endpoint_state: TriangularFixedGridState,
    *,
    expiry_time_increment_y: torch.Tensor,
    uniforms: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None,
    minimum_density: torch.Tensor,
) -> TriangularFixedGridRoll:
    """Reveal the front node and price every claim before and after the roll."""

    probabilities = initial_state.densities[:, 0]
    cdf = torch.cumsum(probabilities, dim=-1)
    cdf[:, -1] = 1.0
    nodes = torch.sum(uniforms[:, None] > cdf, dim=-1)
    spot = torch.gather(initial_state.support, 1, nodes[:, None]).squeeze(1)
    delta = torch.nn.functional.one_hot(nodes, initial_state.support.shape[-1]).to(
        initial_state.support.dtype
    )
    conditional = [delta]
    for expiry_index in range(1, endpoint_state.dlvsqs.shape[1]):
        conditional.append(
            decoder.solve_density_step_batch(
                support=endpoint_state.support,
                dlvsqs=endpoint_state.dlvsqs[:, expiry_index],
                time_increment_y=expiry_time_increment_y,
                rhs_density=conditional[-1],
            )
        )
    conditional_densities = torch.stack(conditional, dim=1)
    if fixed_cash_strikes.ndim == 1:
        strikes = fixed_cash_strikes[None].expand(len(initial_state.support), -1)
    elif fixed_cash_strikes.ndim == 2 and fixed_cash_strikes.shape[0] == len(initial_state.support):
        strikes = fixed_cash_strikes
    else:
        raise ValueError("fixed_cash_strikes must be one vector or one row per path")
    if option_signs is None:
        signs = torch.ones_like(strikes)
    elif option_signs.ndim == 1:
        signs = option_signs[None].expand_as(strikes)
    elif option_signs.shape == strikes.shape:
        signs = option_signs
    else:
        raise ValueError("option_signs must align with fixed_cash_strikes")
    payoffs = torch.relu(signs[:, None] * (initial_state.support[:, :, None] - strikes[:, None]))
    initial_calls = torch.einsum("bts,bsk->btk", initial_state.densities, payoffs)
    endpoint_unconditional_calls = torch.einsum("bts,bsk->btk", endpoint_state.densities, payoffs)
    final_calls = torch.einsum("bts,bsk->btk", conditional_densities, payoffs)
    mass_residual = torch.max(torch.abs(torch.sum(conditional_densities, -1) - 1.0))
    forward_residual = torch.max(
        torch.abs(
            torch.sum(conditional_densities * initial_state.support[:, None], dim=-1)
            - spot[:, None]
        )
    )
    next_state = None
    if endpoint_state.dlvsqs.shape[1] > 1:
        next_state = TriangularFixedGridState(
            support=endpoint_state.support,
            densities=conditional_densities[:, 1:],
            dlvsqs=endpoint_state.dlvsqs[:, 1:],
        )
    return TriangularFixedGridRoll(
        next_state=next_state,
        revealed_spot=spot,
        initial_call_values=initial_calls,
        endpoint_unconditional_call_values=endpoint_unconditional_calls,
        final_call_values=final_calls,
        minimum_density=minimum_density,
        maximum_mass_residual=mass_residual,
        maximum_forward_residual=forward_residual,
    )


def triangular_fixed_grid_roll(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    calendar_time_increment_y: float,
    standard_normals: torch.Tensor,
    uniforms: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
) -> TriangularFixedGridRoll:
    """Evolve one week, reveal the front node, and remove the expired slab."""

    num_substeps = standard_normals.shape[1]
    current = state
    step_y = calendar_time_increment_y / num_substeps
    minimum_density = torch.min(current.densities)
    for step_index in range(num_substeps):
        drift, diffusion = _drift(
            decoder,
            current,
            raw_log_loadings=raw_log_loadings,
            minimum_dlvsq=minimum_dlvsq,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        raw = torch.log(current.dlvsqs - minimum_dlvsq)
        noise = step_y**0.5 * torch.einsum(
            "btif,bf->bti", diffusion, standard_normals[:, step_index]
        )
        proposal_raw = raw + step_y * drift + noise
        proposal_dlvsqs = minimum_dlvsq + torch.exp(proposal_raw)
        proposal_densities = _densities_from_front(
            decoder,
            support=current.support,
            front_density=current.densities[:, 0],
            dlvsqs=proposal_dlvsqs,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        proposal_drift, _ = _drift(
            decoder,
            TriangularFixedGridState(current.support, proposal_densities, proposal_dlvsqs),
            raw_log_loadings=raw_log_loadings,
            minimum_dlvsq=minimum_dlvsq,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        # Diffusion is additive in log excess-DLV coordinates, so stochastic
        # Heun removes the leading Euler weak-drift error without a Milstein term.
        raw = raw + 0.5 * step_y * (drift + proposal_drift) + noise
        if not bool(torch.all(torch.isfinite(raw))):
            raise RuntimeError(
                "non-finite log-DLV in triangular Attempt 2.0 at Euler substep "
                f"{step_index}; max finite |raw|="
                f"{float(torch.max(torch.abs(raw[torch.isfinite(raw)]))):.6g}"
            )
        dlvsqs = minimum_dlvsq + torch.exp(raw)
        if not bool(torch.all(torch.isfinite(dlvsqs))):
            raise RuntimeError(
                "non-finite DLVsq in triangular Attempt 2.0 at Euler substep "
                f"{step_index}; max log-DLV={float(torch.max(raw)):.6g}"
            )
        densities = _densities_from_front(
            decoder,
            support=current.support,
            front_density=current.densities[:, 0],
            dlvsqs=dlvsqs,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        if not bool(torch.all(torch.isfinite(densities))):
            raise RuntimeError(
                "non-finite density in triangular Attempt 2.0 at Euler substep "
                f"{step_index}; max log-DLV={float(torch.max(raw)):.6g}"
            )
        current = TriangularFixedGridState(current.support, densities, dlvsqs)
        minimum_density = torch.minimum(minimum_density, torch.min(densities))

    return _reveal_and_shrink_fixed_grid_roll(
        decoder,
        state,
        current,
        expiry_time_increment_y=expiry_time_increment_y,
        uniforms=uniforms,
        fixed_cash_strikes=fixed_cash_strikes,
        option_signs=option_signs,
        minimum_density=minimum_density,
    )


def triangular_fixed_grid_physical_roll(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    calendar_time_increment_y: float,
    standard_normals: torch.Tensor,
    uniforms: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
) -> TriangularFixedGridRoll:
    r"""Apply the physical zero-drift raw-DLV roll, reveal spot, and shrink.

    This implements the test proposal

    ``log DLVsq[t+1,j-1] - log DLVsq[t,j] = xi[j] dW[t]``

    in log excess-DLV coordinates. The front slab has zero loading because it
    expires during the step. Unlike :func:`triangular_fixed_grid_roll`, this
    function deliberately does not apply the analytic risk-neutral drift.
    """

    batch, num_expiries, num_inner = state.dlvsqs.shape
    if raw_log_loadings.ndim != 3 or raw_log_loadings.shape[:2] != (
        num_expiries,
        num_inner,
    ):
        raise ValueError("raw_log_loadings must align with the live DLV tower")
    if standard_normals.shape != (batch, raw_log_loadings.shape[-1]):
        raise ValueError("standard_normals must provide one factor vector per path")
    if not bool(torch.all(raw_log_loadings[0] == 0.0)):
        raise ValueError("the expiring front DLV slab must have zero loading")
    raw = torch.log(state.dlvsqs - minimum_dlvsq)
    raw_change = calendar_time_increment_y**0.5 * torch.einsum(
        "tif,bf->bti", raw_log_loadings, standard_normals
    )
    endpoint_dlvsqs = minimum_dlvsq + torch.exp(raw + raw_change)
    endpoint_densities = _densities_from_front(
        decoder,
        support=state.support,
        front_density=state.densities[:, 0],
        dlvsqs=endpoint_dlvsqs,
        expiry_time_increment_y=expiry_time_increment_y,
    )
    if not bool(torch.all(torch.isfinite(endpoint_densities))):
        raise RuntimeError("non-finite density in physical fixed-grid DLV roll")
    endpoint = TriangularFixedGridState(
        support=state.support,
        densities=endpoint_densities,
        dlvsqs=endpoint_dlvsqs,
    )
    return _reveal_and_shrink_fixed_grid_roll(
        decoder,
        state,
        endpoint,
        expiry_time_increment_y=expiry_time_increment_y,
        uniforms=uniforms,
        fixed_cash_strikes=fixed_cash_strikes,
        option_signs=option_signs,
        minimum_density=torch.minimum(torch.min(state.densities), torch.min(endpoint_densities)),
    )


__all__ = [
    "TriangularFixedGridRoll",
    "TriangularFixedGridState",
    "regrid_triangular_state_to_revealed_spot",
    "triangular_fixed_grid_physical_roll",
    "triangular_fixed_grid_roll",
]
