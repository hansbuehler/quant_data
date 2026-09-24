"""Shrinking fixed-expiry implementation of the Attempt 2.0 DLVsq SDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from quantdata.dysanos.flex.lvsq_drift import discrete_lvsq_grid_operator_product
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.dysanos.flex.theoretical_traded_density import (
    ViableExponentialAction,
    exponential_log_density_increment,
    theoretical_exponential_action,
    viable_exponential_action,
)


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


@dataclass(frozen=True)
class TriangularFixedGridTradedDensityRoll:
    """One physical roll and its exact traded exponential-density increment."""

    roll: TriangularFixedGridRoll
    log_marginal_utility: torch.Tensor
    maximum_factor_replication_residual: torch.Tensor
    maximum_action_l1: torch.Tensor
    minimum_traded_factor_rank: torch.Tensor
    maximum_requested_factor_norm: torch.Tensor
    maximum_viable_factor_norm: torch.Tensor
    minimum_factor_retention_fraction: torch.Tensor
    maximum_unscaled_action_l1: torch.Tensor
    integrated_squared_factor_shift: torch.Tensor


@dataclass(frozen=True)
class IndependentDlvNoiseCalibration:
    """Diagonal log-DLV noise calibrated to marginal option variance."""

    augmented_raw_log_loadings: torch.Tensor
    independent_raw_log_loadings: torch.Tensor
    diagnostics: dict[str, Any]


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


def _drift_components(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    collect_density_diffusion: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return raw-DLV coefficients and optionally the induced density diffusion."""

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
    density_diffusions = [previous_density_diffusion] if collect_density_diffusion else None
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
        if density_diffusions is not None:
            density_diffusions.append(density_diffusion)
    return (
        drift,
        diffusion,
        None if density_diffusions is None else torch.stack(density_diffusions, dim=1),
    )


def _drift(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return drift and diffusion of log(DLVsq-minimum) for the live tower."""

    drift, diffusion, density_diffusion = _drift_components(
        decoder,
        state,
        raw_log_loadings=raw_log_loadings,
        minimum_dlvsq=minimum_dlvsq,
        expiry_time_increment_y=expiry_time_increment_y,
        collect_density_diffusion=False,
    )
    assert density_diffusion is None
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


def triangular_fixed_grid_ar1_roll(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    conditional_intercept: torch.Tensor,
    conditional_slope: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    calendar_time_increment_y: float,
    standard_normals: torch.Tensor,
    uniforms: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
) -> TriangularFixedGridRoll:
    r"""Apply the fitted fixed-expiry AR(1) roll on the unchanged cash grid.

    Each post-roll slab is predicted from the matching non-front slab before
    the expiry shrink: ``z' = intercept + slope*z + L eps``.  After the
    front slab is removed, this becomes the next shorter-expiry tower.  The
    innovation loading ``L`` is shared exactly with the risk-neutral SDE; only
    the conditional drift differs between these two path laws.
    """

    batch, num_expiries, num_inner = state.dlvsqs.shape
    if raw_log_loadings.ndim != 3 or raw_log_loadings.shape[:2] != (
        num_expiries,
        num_inner,
    ):
        raise ValueError("raw_log_loadings must align with the live DLV tower")
    if conditional_intercept.shape != (num_expiries - 1, num_inner) or (
        conditional_slope.shape != conditional_intercept.shape
    ):
        raise ValueError("AR(1) coefficients must align with live non-front slabs")
    if standard_normals.shape != (batch, raw_log_loadings.shape[-1]):
        raise ValueError("standard_normals must provide one factor vector per path")
    if not bool(torch.all(raw_log_loadings[0] == 0.0)):
        raise ValueError("the expiring front DLV slab must have zero loading")
    raw = torch.log(state.dlvsqs - minimum_dlvsq)
    innovation = calendar_time_increment_y**0.5 * torch.einsum(
        "tif,bf->bti", raw_log_loadings, standard_normals
    )
    endpoint_raw = raw.clone()
    endpoint_raw[:, 1:] = (
        conditional_intercept[None]
        + conditional_slope[None] * raw[:, 1:]
        + innovation[:, 1:]
    )
    endpoint_dlvsqs = minimum_dlvsq + torch.exp(endpoint_raw)
    endpoint_densities = _densities_from_front(
        decoder,
        support=state.support,
        front_density=state.densities[:, 0],
        dlvsqs=endpoint_dlvsqs,
        expiry_time_increment_y=expiry_time_increment_y,
    )
    if not bool(torch.all(torch.isfinite(endpoint_densities))):
        raise RuntimeError("non-finite density in AR(1) fixed-grid DLV roll")
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


def triangular_fixed_grid_ar1_heun_roll(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    conditional_intercept: torch.Tensor,
    conditional_slope: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    calendar_time_increment_y: float,
    standard_normals: torch.Tensor,
    uniforms: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
) -> TriangularFixedGridRoll:
    r"""Use the AR(1) affine drift in the same stochastic-Heun scheme as Q.

    The coordinatewise weekly AR(1) mean is ``alpha + beta*z``.  Its affine
    drift extension is ``(alpha + (beta - 1)*z) / h``, whose one-week Euler
    mean equals the fitted AR(1) mean.  The unconstrained empirical slopes can
    be negative or above one, so this is intentionally an affine extension,
    not a claim that every coordinate is a continuous mean-reverting OU.
    """

    batch, num_expiries, num_inner = state.dlvsqs.shape
    if raw_log_loadings.ndim != 3 or raw_log_loadings.shape[:2] != (
        num_expiries,
        num_inner,
    ):
        raise ValueError("raw_log_loadings must align with the live DLV tower")
    if conditional_intercept.shape != (num_expiries - 1, num_inner) or (
        conditional_slope.shape != conditional_intercept.shape
    ):
        raise ValueError("AR(1) coefficients must align with live non-front slabs")
    if (
        standard_normals.ndim != 3
        or standard_normals.shape[0] != batch
        or standard_normals.shape[-1] != raw_log_loadings.shape[-1]
    ):
        raise ValueError("standard_normals must provide one factor vector per path and substep")
    if not bool(torch.all(raw_log_loadings[0] == 0.0)):
        raise ValueError("the expiring front DLV slab must have zero loading")

    num_substeps = standard_normals.shape[1]
    step_y = calendar_time_increment_y / num_substeps
    current = state
    minimum_density = torch.min(current.densities)

    def ar1_drift(dlvsqs: torch.Tensor) -> torch.Tensor:
        raw = torch.log(dlvsqs - minimum_dlvsq)
        drift = torch.zeros_like(raw)
        drift[:, 1:] = (
            conditional_intercept[None]
            + (conditional_slope[None] - 1.0) * raw[:, 1:]
        ) / calendar_time_increment_y
        return drift

    for step_index in range(num_substeps):
        raw = torch.log(current.dlvsqs - minimum_dlvsq)
        drift = ar1_drift(current.dlvsqs)
        noise = step_y**0.5 * torch.einsum(
            "tif,bf->bti", raw_log_loadings, standard_normals[:, step_index]
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
        proposal_drift = ar1_drift(proposal_dlvsqs)
        raw = raw + 0.5 * step_y * (drift + proposal_drift) + noise
        if not bool(torch.all(torch.isfinite(raw))):
            raise RuntimeError(f"non-finite AR(1)-Heun log-DLV at substep {step_index}")
        dlvsqs = minimum_dlvsq + torch.exp(raw)
        densities = _densities_from_front(
            decoder,
            support=current.support,
            front_density=current.densities[:, 0],
            dlvsqs=dlvsqs,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        if not bool(torch.all(torch.isfinite(densities))):
            raise RuntimeError(f"non-finite AR(1)-Heun density at substep {step_index}")
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


def _project_onto_loading_span(
    values: torch.Tensor,
    loadings: torch.Tensor,
    *,
    relative_tolerance: float = 1.0e-12,
) -> torch.Tensor:
    """Project row-wise coordinate values onto the Brownian loading span."""

    if values.ndim != 2 or loadings.ndim != 2 or values.shape[1] != loadings.shape[0]:
        raise ValueError("values and loadings must share one coordinate axis")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie strictly between zero and one")
    loading_pinv = torch.linalg.pinv(loadings, rtol=relative_tolerance, atol=0.0)
    return values @ loading_pinv.T @ loadings.T


def triangular_fixed_grid_achievable_heun_roll(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    conditional_intercept: torch.Tensor,
    conditional_slope: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    calendar_time_increment_y: float,
    standard_normals: torch.Tensor,
    uniforms: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
) -> TriangularFixedGridRoll:
    r"""Use the closest AR(1) drift admitting a change to the analytical Q law.

    At every Heun evaluation, the physical drift is

    ``mu_P_star = mu_Q + Proj_col(L)(mu_AR1 - mu_Q)``.

    Hence ``mu_P_star - mu_Q`` lies exactly in the span of the common Brownian
    loading matrix ``L``. This removes the historically fitted drift component
    which cannot be changed by a Girsanov density while preserving the part
    supported by the fitted DLV innovations.
    """

    batch, num_expiries, num_inner = state.dlvsqs.shape
    if raw_log_loadings.ndim != 3 or raw_log_loadings.shape[:2] != (
        num_expiries,
        num_inner,
    ):
        raise ValueError("raw_log_loadings must align with the live DLV tower")
    if conditional_intercept.shape != (num_expiries - 1, num_inner) or (
        conditional_slope.shape != conditional_intercept.shape
    ):
        raise ValueError("AR(1) coefficients must align with live non-front slabs")
    if (
        standard_normals.ndim != 3
        or standard_normals.shape[0] != batch
        or standard_normals.shape[-1] != raw_log_loadings.shape[-1]
    ):
        raise ValueError("standard_normals must provide one factor vector per path and substep")
    if not bool(torch.all(raw_log_loadings[0] == 0.0)):
        raise ValueError("the expiring front DLV slab must have zero loading")

    num_substeps = standard_normals.shape[1]
    step_y = calendar_time_increment_y / num_substeps
    current = state
    minimum_density = torch.min(current.densities)
    loading_matrix = raw_log_loadings[1:].reshape(-1, raw_log_loadings.shape[-1])
    loading_pinv = torch.linalg.pinv(loading_matrix, rtol=1.0e-12, atol=0.0)

    def achievable_drift(current_state: TriangularFixedGridState) -> torch.Tensor:
        q_drift, _ = _drift(
            decoder,
            current_state,
            raw_log_loadings=raw_log_loadings,
            minimum_dlvsq=minimum_dlvsq,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        raw = torch.log(current_state.dlvsqs - minimum_dlvsq)
        ar1_drift = torch.zeros_like(raw)
        ar1_drift[:, 1:] = (
            conditional_intercept[None]
            + (conditional_slope[None] - 1.0) * raw[:, 1:]
        ) / calendar_time_increment_y
        difference = (ar1_drift[:, 1:] - q_drift[:, 1:]).reshape(batch, -1)
        projected = difference @ loading_pinv.T @ loading_matrix.T
        drift = q_drift.clone()
        drift[:, 1:] += projected.reshape(batch, num_expiries - 1, num_inner)
        return drift

    for step_index in range(num_substeps):
        raw = torch.log(current.dlvsqs - minimum_dlvsq)
        drift = achievable_drift(current)
        noise = step_y**0.5 * torch.einsum(
            "tif,bf->bti", raw_log_loadings, standard_normals[:, step_index]
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
        proposal = TriangularFixedGridState(
            current.support,
            proposal_densities,
            proposal_dlvsqs,
        )
        proposal_drift = achievable_drift(proposal)
        raw = raw + 0.5 * step_y * (drift + proposal_drift) + noise
        if not bool(torch.all(torch.isfinite(raw))):
            raise RuntimeError(f"non-finite achievable-Heun log-DLV at substep {step_index}")
        dlvsqs = minimum_dlvsq + torch.exp(raw)
        densities = _densities_from_front(
            decoder,
            support=current.support,
            front_density=current.densities[:, 0],
            dlvsqs=dlvsqs,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        if not bool(torch.all(torch.isfinite(densities))):
            raise RuntimeError(f"non-finite achievable-Heun density at substep {step_index}")
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


def _option_return_loadings(
    state: TriangularFixedGridState,
    density_diffusion: torch.Tensor,
    *,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return option values and their relative diffusion loadings."""

    batch, num_expiries, num_nodes = state.densities.shape
    if density_diffusion.ndim != 4 or density_diffusion.shape[:3] != (
        batch,
        num_expiries,
        num_nodes,
    ):
        raise ValueError("density_diffusion must align with the state density tower")
    if fixed_cash_strikes.ndim == 1:
        strikes = fixed_cash_strikes[None].expand(batch, -1)
    elif fixed_cash_strikes.ndim == 2 and fixed_cash_strikes.shape[0] == batch:
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
    payoff = torch.relu(signs[:, None] * (state.support[:, :, None] - strikes[:, None]))
    values = torch.einsum("bts,bsk->btk", state.densities, payoff)
    if not bool(torch.all(torch.isfinite(values) & (values > 0.0))):
        raise RuntimeError("the theoretical traded density requires positive option values")
    value_diffusion = torch.einsum("btsf,bsk->btkf", density_diffusion, payoff)
    return values, (value_diffusion / values[..., None]).reshape(batch, -1, value_diffusion.shape[-1])


def dlv_option_return_loadings(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return values and relative diffusions of options induced by DLV shocks."""

    _, _, density_diffusion = _drift_components(
        decoder,
        state,
        raw_log_loadings=raw_log_loadings,
        minimum_dlvsq=minimum_dlvsq,
        expiry_time_increment_y=expiry_time_increment_y,
        collect_density_diffusion=True,
    )
    assert density_diffusion is not None
    return _option_return_loadings(
        state,
        density_diffusion,
        fixed_cash_strikes=fixed_cash_strikes,
        option_signs=option_signs,
    )


def calibrate_independent_dlv_noise(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    target_option_variance_fraction: float,
    option_signs: torch.Tensor | None = None,
) -> IndependentDlvNoiseCalibration:
    r"""Add one independent factor per non-front DLV coordinate.

    At the supplied calibration state, let ``B`` be the existing relative
    option-return loading and ``J`` the loading generated by unit independent
    noise in each live non-front log-DLV coordinate. Coordinate ``k`` receives
    an independent loading proportional to its fitted marginal loading norm
    ``||L[k]||``. A single positive multiplier is chosen so the median
    propagated option-variance fraction is ``f``.

    This construction gives *every* non-front DLV coordinate a genuine
    independent shock and hence makes its diffusion full row rank. The coupled
    DLV-to-option map prevents every option marginal from equalling ``f``
    exactly, so the complete achieved range is returned rather than hidden.
    The expiring front slab remains unshocked: its option returns are generated
    by the already fixed front density and revealed spot.
    """

    if state.dlvsqs.ndim != 3 or state.dlvsqs.shape[0] != 1:
        raise ValueError("independent DLV noise calibration requires one state")
    if raw_log_loadings.shape[:2] != state.dlvsqs.shape[1:]:
        raise ValueError("raw log loadings must align with the calibration state")
    if not 0.0 < target_option_variance_fraction < 1.0:
        raise ValueError("target option variance fraction must lie in (0, 1)")

    num_expiries, num_inner, base_rank = raw_log_loadings.shape
    num_independent = (num_expiries - 1) * num_inner
    unit = torch.zeros(
        (num_expiries, num_inner, num_independent),
        dtype=raw_log_loadings.dtype,
        device=raw_log_loadings.device,
    )
    unit[1:] = torch.eye(
        num_independent,
        dtype=raw_log_loadings.dtype,
        device=raw_log_loadings.device,
    ).reshape(num_expiries - 1, num_inner, num_independent)
    combined_unit = torch.cat((raw_log_loadings, unit), dim=-1)
    _, _, density_diffusion = _drift_components(
        decoder,
        state,
        raw_log_loadings=combined_unit,
        minimum_dlvsq=minimum_dlvsq,
        expiry_time_increment_y=expiry_time_increment_y,
        collect_density_diffusion=True,
    )
    assert density_diffusion is not None
    _, return_loadings = _option_return_loadings(
        state,
        density_diffusion,
        fixed_cash_strikes=fixed_cash_strikes,
        option_signs=option_signs,
    )
    base = return_loadings[0, :, :base_rank].double()
    coordinate = return_loadings[0, :, base_rank:].double()
    base_variance = torch.sum(base.square(), dim=1)
    unit_variance = torch.sum(coordinate.square(), dim=1)
    numerical_floor = torch.finfo(torch.float64).eps * torch.max(base_variance).clamp_min(1.0)
    active = (base_variance > numerical_floor) & (unit_variance > numerical_floor)
    if not bool(torch.any(active)):
        raise RuntimeError("no surface-sensitive options were available for noise calibration")

    natural_coordinate_scale = torch.linalg.vector_norm(
        raw_log_loadings[1:],
        dim=-1,
    ).reshape(-1)
    if not bool(torch.all(torch.isfinite(natural_coordinate_scale))):
        raise RuntimeError("fitted DLV loading norms are non-finite")
    if not bool(torch.all(natural_coordinate_scale > 0.0)):
        raise RuntimeError("every independently shocked DLV coordinate must have positive variance")
    unit_added_variance = coordinate.square() @ natural_coordinate_scale.double().square()
    unit_ratio = unit_added_variance[active] / base_variance[active]
    positive_ratio = unit_ratio[unit_ratio > 0.0]
    if len(positive_ratio) != int(torch.sum(active)):
        raise RuntimeError("independent DLV noise does not reach every surface-sensitive option")
    target_added_ratio = target_option_variance_fraction / (
        1.0 - target_option_variance_fraction
    )
    common_variance_multiplier = target_added_ratio / torch.median(positive_ratio)
    coordinate_scale = (
        torch.sqrt(common_variance_multiplier).to(raw_log_loadings.dtype)
        * natural_coordinate_scale
    )
    coordinate_variance = coordinate_scale.square()
    independent = unit * coordinate_scale[None, None]
    augmented = torch.cat((raw_log_loadings, independent), dim=-1)

    added_variance = coordinate.square() @ coordinate_variance.double()
    achieved_fraction = added_variance[active] / (
        base_variance[active] + added_variance[active]
    )
    relative_error = achieved_fraction / target_option_variance_fraction - 1.0
    quantiles = torch.quantile(
        achieved_fraction,
        torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), dtype=torch.float64, device=active.device),
    )
    positive_scales = coordinate_scale[coordinate_scale > 0.0]
    return IndependentDlvNoiseCalibration(
        augmented_raw_log_loadings=augmented,
        independent_raw_log_loadings=independent,
        diagnostics={
            "definition": (
                "one independent Gaussian factor per non-front log-DLV coordinate, "
                "scaled in common relative to each coordinate's fitted marginal volatility"
            ),
            "calibration_scope": "initial state instantaneous relative option-return loadings",
            "target_total_option_variance_fraction": target_option_variance_fraction,
            "num_base_factors": base_rank,
            "num_independent_dlv_factors": num_independent,
            "num_option_coordinates": int(len(base_variance)),
            "num_surface_sensitive_option_coordinates": int(torch.sum(active)),
            "num_expiring_front_option_coordinates": int(torch.sum(~active)),
            "achieved_fraction_minimum": float(quantiles[0]),
            "achieved_fraction_q10": float(quantiles[1]),
            "achieved_fraction_median": float(quantiles[2]),
            "achieved_fraction_q90": float(quantiles[3]),
            "achieved_fraction_maximum": float(quantiles[4]),
            "achieved_fraction_rms_relative_error": float(
                torch.sqrt(torch.mean(relative_error.square()))
            ),
            "maximum_independent_raw_log_loading": float(torch.max(coordinate_scale)),
            "minimum_positive_independent_raw_log_loading": float(torch.min(positive_scales)),
            "common_coordinate_variance_multiplier": float(common_variance_multiplier),
            "independent_fraction_of_each_dlv_coordinate_variance": float(
                common_variance_multiplier / (1.0 + common_variance_multiplier)
            ),
        },
    )


def initial_dlv_measure_switch_diagnostics(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    conditional_intercept: torch.Tensor,
    conditional_slope: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    calendar_time_increment_y: float,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Diagnose the exact local AR(1)-to-Q switch at one initial state."""

    if state.dlvsqs.ndim != 3 or state.dlvsqs.shape[0] != 1:
        raise ValueError("measure-switch diagnostics require one initial state")
    _, num_expiries, num_inner = state.dlvsqs.shape
    if raw_log_loadings.shape[:2] != (num_expiries, num_inner):
        raise ValueError("raw log loadings must align with the diagnostic state")
    if conditional_intercept.shape != (num_expiries - 1, num_inner) or (
        conditional_slope.shape != conditional_intercept.shape
    ):
        raise ValueError("AR(1) coefficients must align with live non-front slabs")
    if calendar_time_increment_y <= 0.0:
        raise ValueError("calendar time increment must be positive")

    q_drift, _, density_diffusion = _drift_components(
        decoder,
        state,
        raw_log_loadings=raw_log_loadings,
        minimum_dlvsq=minimum_dlvsq,
        expiry_time_increment_y=expiry_time_increment_y,
        collect_density_diffusion=True,
    )
    assert density_diffusion is not None
    raw = torch.log(state.dlvsqs - minimum_dlvsq)
    ar1_drift = torch.zeros_like(raw)
    ar1_drift[:, 1:] = (
        conditional_intercept[None]
        + (conditional_slope[None] - 1.0) * raw[:, 1:]
    ) / calendar_time_increment_y
    drift_difference = (ar1_drift[:, 1:] - q_drift[:, 1:]).reshape(1, -1)
    loading_matrix = raw_log_loadings[1:].reshape(-1, raw_log_loadings.shape[-1])
    loading_pinv = torch.linalg.pinv(loading_matrix, rtol=1.0e-12, atol=0.0)
    factor_shift = drift_difference @ loading_pinv.T
    drift_residual = factor_shift @ loading_matrix.T - drift_difference

    _, return_loadings = _option_return_loadings(
        state,
        density_diffusion,
        fixed_cash_strikes=fixed_cash_strikes,
        option_signs=option_signs,
    )
    theoretical_action = theoretical_exponential_action(
        return_loadings,
        factor_shift,
        relative_tolerance=1.0e-12,
    )
    loading_singular_values = torch.linalg.svdvals(loading_matrix)
    return_norms = torch.linalg.vector_norm(return_loadings[0], dim=1)
    active_options = return_norms > torch.finfo(return_loadings.dtype).eps * torch.max(
        return_norms
    )
    active_return_singular_values = torch.linalg.svdvals(return_loadings[0, active_options])
    integrated_shift = calendar_time_increment_y * torch.sum(factor_shift.square())
    return {
        "num_dlv_coordinates": int(loading_matrix.shape[0]),
        "num_factors": int(loading_matrix.shape[1]),
        "dlv_loading_numerical_rank": int(
            torch.linalg.matrix_rank(loading_matrix, rtol=1.0e-12, atol=0.0)
        ),
        "minimum_dlv_loading_singular_value": float(torch.min(loading_singular_values)),
        "maximum_dlv_loading_singular_value": float(torch.max(loading_singular_values)),
        "dlv_loading_condition_number": float(
            torch.max(loading_singular_values) / torch.min(loading_singular_values)
        ),
        "num_surface_sensitive_option_coordinates": int(torch.sum(active_options)),
        "option_loading_numerical_rank": int(theoretical_action.numerical_rank[0]),
        "minimum_active_option_loading_singular_value": float(
            torch.min(active_return_singular_values)
        ),
        "maximum_active_option_loading_singular_value": float(
            torch.max(active_return_singular_values)
        ),
        "maximum_absolute_dlv_drift_replication_residual": float(
            torch.max(torch.abs(drift_residual))
        ),
        "maximum_absolute_traded_factor_replication_residual": float(
            torch.max(torch.abs(theoretical_action.factor_replication_residual))
        ),
        "factor_shift_l2": float(torch.linalg.vector_norm(factor_shift)),
        "one_week_integrated_squared_factor_shift": float(integrated_shift),
        "one_week_lognormal_ess_fraction": float(torch.exp(-integrated_shift)),
        "minimum_norm_traded_action_l1": float(
            torch.sum(torch.abs(theoretical_action.action))
        ),
    }


def triangular_fixed_grid_achievable_traded_density_roll(
    decoder: FlexSanosDecoder,
    state: TriangularFixedGridState,
    *,
    raw_log_loadings: torch.Tensor,
    conditional_intercept: torch.Tensor,
    conditional_slope: torch.Tensor,
    minimum_dlvsq: float,
    expiry_time_increment_y: torch.Tensor,
    calendar_time_increment_y: float,
    standard_normals: torch.Tensor,
    uniforms: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    option_signs: torch.Tensor | None = None,
    action_relative_tolerance: float = 1.0e-10,
    traded_ridge_ratio: float = 0.0,
    maximum_factor_shift_norm: float = float("inf"),
    maximum_action_l1: float = float("inf"),
) -> TriangularFixedGridTradedDensityRoll:
    r"""Roll under achievable P and construct its theoretical Q density.

    At every substep, the known factor market price ``lambda`` is converted
    into an option portfolio ``a`` satisfying ``B.T @ a = lambda_viable``.
    A traded-direction ridge and hard factor/action limits may shrink the
    requested drift before both P and its density are defined. The density
    increment is the normalized exponential marginal utility of this *traded*
    portfolio return. The Gaussian driver is used to simulate P but is
    deliberately absent from the density formula.
    """

    batch, num_expiries, num_inner = state.dlvsqs.shape
    if raw_log_loadings.ndim != 3 or raw_log_loadings.shape[:2] != (
        num_expiries,
        num_inner,
    ):
        raise ValueError("raw_log_loadings must align with the live DLV tower")
    if conditional_intercept.shape != (num_expiries - 1, num_inner) or (
        conditional_slope.shape != conditional_intercept.shape
    ):
        raise ValueError("AR(1) coefficients must align with live non-front slabs")
    if (
        standard_normals.ndim != 3
        or standard_normals.shape[0] != batch
        or standard_normals.shape[-1] != raw_log_loadings.shape[-1]
    ):
        raise ValueError("standard_normals must provide one factor vector per path and substep")
    if not bool(torch.all(raw_log_loadings[0] == 0.0)):
        raise ValueError("the expiring front DLV slab must have zero loading")

    num_substeps = standard_normals.shape[1]
    step_y = calendar_time_increment_y / num_substeps
    current = state
    minimum_density = torch.min(current.densities)
    log_marginal_utility = torch.zeros(batch, dtype=state.dlvsqs.dtype, device=state.dlvsqs.device)
    maximum_factor_residual = torch.zeros((), dtype=state.dlvsqs.dtype, device=state.dlvsqs.device)
    observed_maximum_action_l1 = torch.zeros(
        (), dtype=state.dlvsqs.dtype, device=state.dlvsqs.device
    )
    maximum_requested_factor_norm = torch.zeros_like(observed_maximum_action_l1)
    maximum_viable_factor_norm = torch.zeros_like(observed_maximum_action_l1)
    minimum_factor_retention = torch.ones_like(observed_maximum_action_l1)
    maximum_unscaled_action_l1 = torch.zeros_like(observed_maximum_action_l1)
    integrated_squared_factor_shift = torch.zeros(
        batch,
        dtype=state.dlvsqs.dtype,
        device=state.dlvsqs.device,
    )
    minimum_traded_rank = torch.full(
        (),
        raw_log_loadings.shape[-1],
        dtype=torch.int64,
        device=state.dlvsqs.device,
    )
    loading_matrix = raw_log_loadings[1:].reshape(-1, raw_log_loadings.shape[-1])
    loading_pinv = torch.linalg.pinv(loading_matrix, rtol=1.0e-12, atol=0.0)

    def coefficients(
        current_state: TriangularFixedGridState,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        ViableExponentialAction,
    ]:
        q_drift, _, density_diffusion = _drift_components(
            decoder,
            current_state,
            raw_log_loadings=raw_log_loadings,
            minimum_dlvsq=minimum_dlvsq,
            expiry_time_increment_y=expiry_time_increment_y,
            collect_density_diffusion=True,
        )
        assert density_diffusion is not None
        raw = torch.log(current_state.dlvsqs - minimum_dlvsq)
        ar1_drift = torch.zeros_like(raw)
        ar1_drift[:, 1:] = (
            conditional_intercept[None]
            + (conditional_slope[None] - 1.0) * raw[:, 1:]
        ) / calendar_time_increment_y
        difference = (ar1_drift[:, 1:] - q_drift[:, 1:]).reshape(batch, -1)
        requested_factor_shift = difference @ loading_pinv.T
        values, return_loadings = _option_return_loadings(
            current_state,
            density_diffusion,
            fixed_cash_strikes=fixed_cash_strikes,
            option_signs=option_signs,
        )
        viable = viable_exponential_action(
            return_loadings,
            requested_factor_shift,
            traded_ridge_ratio=traded_ridge_ratio,
            maximum_factor_shift_norm=maximum_factor_shift_norm,
            maximum_action_l1=maximum_action_l1,
            relative_tolerance=action_relative_tolerance,
        )
        physical_drift = q_drift.clone()
        physical_drift[:, 1:] += (viable.viable_factor_shift @ loading_matrix.T).reshape(
            batch,
            num_expiries - 1,
            num_inner,
        )
        return (
            physical_drift,
            viable.viable_factor_shift,
            density_diffusion,
            values,
            return_loadings,
            viable,
        )

    for step_index in range(num_substeps):
        raw = torch.log(current.dlvsqs - minimum_dlvsq)
        (
            drift,
            factor_shift,
            density_diffusion,
            current_values,
            return_loadings,
            viable,
        ) = coefficients(current)
        theoretical_action = viable.theoretical_action
        return_drift_difference = torch.einsum(
            "bif,bf->bi",
            return_loadings,
            factor_shift,
        )

        noise = step_y**0.5 * torch.einsum(
            "tif,bf->bti", raw_log_loadings, standard_normals[:, step_index]
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
        proposal = TriangularFixedGridState(current.support, proposal_densities, proposal_dlvsqs)
        proposal_drift, _, _, _, _, _ = coefficients(proposal)
        raw = raw + 0.5 * step_y * (drift + proposal_drift) + noise
        if not bool(torch.all(torch.isfinite(raw))):
            raise RuntimeError(f"non-finite theoretical-density log-DLV at substep {step_index}")
        dlvsqs = minimum_dlvsq + torch.exp(raw)
        densities = _densities_from_front(
            decoder,
            support=current.support,
            front_density=current.densities[:, 0],
            dlvsqs=dlvsqs,
            expiry_time_increment_y=expiry_time_increment_y,
        )
        if not bool(torch.all(torch.isfinite(densities))):
            raise RuntimeError(f"non-finite theoretical-density density at substep {step_index}")
        next_state = TriangularFixedGridState(current.support, densities, dlvsqs)
        next_values, _ = _option_return_loadings(
            next_state,
            torch.zeros_like(density_diffusion),
            fixed_cash_strikes=fixed_cash_strikes,
            option_signs=option_signs,
        )
        realized_returns = (next_values / current_values - 1.0).reshape(batch, -1)
        log_marginal_utility += exponential_log_density_increment(
            theoretical_action.action,
            realized_returns,
            return_drift_difference,
            time_increment=step_y,
        )
        maximum_factor_residual = torch.maximum(
            maximum_factor_residual,
            torch.max(torch.abs(theoretical_action.factor_replication_residual)),
        )
        observed_maximum_action_l1 = torch.maximum(
            observed_maximum_action_l1,
            torch.max(torch.sum(torch.abs(theoretical_action.action), dim=1)),
        )
        minimum_traded_rank = torch.minimum(
            minimum_traded_rank,
            torch.min(theoretical_action.numerical_rank),
        )
        maximum_requested_factor_norm = torch.maximum(
            maximum_requested_factor_norm,
            torch.max(torch.linalg.vector_norm(viable.requested_factor_shift, dim=1)),
        )
        maximum_viable_factor_norm = torch.maximum(
            maximum_viable_factor_norm,
            torch.max(torch.linalg.vector_norm(viable.viable_factor_shift, dim=1)),
        )
        minimum_factor_retention = torch.minimum(
            minimum_factor_retention,
            torch.min(viable.factor_retention_fraction),
        )
        maximum_unscaled_action_l1 = torch.maximum(
            maximum_unscaled_action_l1,
            torch.max(viable.unscaled_action_l1),
        )
        integrated_squared_factor_shift += step_y * torch.sum(factor_shift.square(), dim=1)
        current = next_state
        minimum_density = torch.minimum(minimum_density, torch.min(densities))

    roll = _reveal_and_shrink_fixed_grid_roll(
        decoder,
        state,
        current,
        expiry_time_increment_y=expiry_time_increment_y,
        uniforms=uniforms,
        fixed_cash_strikes=fixed_cash_strikes,
        option_signs=option_signs,
        minimum_density=minimum_density,
    )
    return TriangularFixedGridTradedDensityRoll(
        roll=roll,
        log_marginal_utility=log_marginal_utility,
        maximum_factor_replication_residual=maximum_factor_residual,
        maximum_action_l1=observed_maximum_action_l1,
        minimum_traded_factor_rank=minimum_traded_rank,
        maximum_requested_factor_norm=maximum_requested_factor_norm,
        maximum_viable_factor_norm=maximum_viable_factor_norm,
        minimum_factor_retention_fraction=minimum_factor_retention,
        maximum_unscaled_action_l1=maximum_unscaled_action_l1,
        integrated_squared_factor_shift=integrated_squared_factor_shift,
    )


__all__ = [
    "IndependentDlvNoiseCalibration",
    "TriangularFixedGridRoll",
    "TriangularFixedGridState",
    "TriangularFixedGridTradedDensityRoll",
    "calibrate_independent_dlv_noise",
    "dlv_option_return_loadings",
    "initial_dlv_measure_switch_diagnostics",
    "regrid_triangular_state_to_revealed_spot",
    "triangular_fixed_grid_achievable_heun_roll",
    "triangular_fixed_grid_achievable_traded_density_roll",
    "triangular_fixed_grid_ar1_roll",
    "triangular_fixed_grid_ar1_heun_roll",
    "triangular_fixed_grid_physical_roll",
    "triangular_fixed_grid_roll",
]
