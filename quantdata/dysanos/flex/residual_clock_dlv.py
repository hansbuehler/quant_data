"""Exact functional DLV transitions on an explicit residual-expiry clock.

The fitted SANOS decoder has a fixed model clock.  That clock is appropriate
for the initial decode, but it is not the clock of a surface after one or more
calendar rolls.  This module therefore carries the residual expiry clock as
part of the state and uses its consecutive increments in every density solve.

The local DLV drift and the finite transition are deliberately separate.  The
local characteristics expose the continuous-time ``psi``, ``beta``, ``alpha``
and log-excess-DLV drift ``mu_z``.  The finite transition evolves both density
and variance flux linearly in a supplied innovation and only then divides the
two to recover the DLV.  Consequently, its density martingale identity is
exact for any admissible zero-mean cubature; it is not an Euler step of
``mu_z`` and it does not subtract nearly equal option prices in low-mass tails.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

import torch

from quantdata.dysanos.flex.config import CappedUnboundedDlvFlexGeometryConfig
from quantdata.dysanos.flex.lvsq_drift import (
    discrete_lvsq_grid_operator_product,
    symmetric_rademacher_cubature,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface

DEFAULT_FRONT_SPLIT_SCALE_CANDIDATES = (
    1.0,
    *tuple(
        value
        for step in range(1, 81)
        for value in (2.0 ** (step / 20.0), 2.0 ** (-step / 20.0))
    ),
)


class FunctionalDlvInadmissibleError(RuntimeError):
    """An exact density branch cannot be represented by admissible DLVs."""

    def __init__(
        self,
        message: str,
        *,
        expiry_index: int | None = None,
        batch_positions: tuple[int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.expiry_index = expiry_index
        self.batch_positions = batch_positions


@dataclass(frozen=True)
class ResidualClockDlvSurface:
    """A normalized DLV surface with its current residual-expiry clock.

    ``temporary_frozen_front`` identifies the exact short slab inserted by
    density interpolation.  Its DLV is a price-authoritative variance rate,
    rather than a fitted model coordinate, and can therefore lie outside the
    fitted open bounds.  The row must be frozen and is discarded at the next
    chronological roll; every survivor row remains strictly fitted-bound.
    """

    residual_expiries_y: torch.Tensor
    support_strikes: torch.Tensor
    density0: torch.Tensor
    densities: torch.Tensor
    dlvsqs: torch.Tensor
    backbone_total_vols: torch.Tensor
    temporary_frozen_front: bool = False


@dataclass(frozen=True)
class ResidualClockDlvIngress:
    """A one-time canonical density rebuild and its numerical certificates."""

    surface: ResidualClockDlvSurface
    maximum_input_density_perturbation: torch.Tensor
    maximum_input_density0_perturbation: torch.Tensor
    maximum_input_density_tower_perturbation: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_forward_residual: torch.Tensor
    maximum_recursion_residual: torch.Tensor
    minimum_density: torch.Tensor
    minimum_dlvsq_bound_margin: torch.Tensor
    maximum_fixed_option_perturbation: torch.Tensor | None


@dataclass(frozen=True)
class FunctionalDlvCharacteristics:
    r"""Local fixed-contract DLV characteristics for a functional surface.

    ``psi`` is the density diffusion, ``beta`` the physical DLV diffusion,
    ``alpha`` the physical DLV drift and ``mu_z`` the drift of
    ``z = log(dlvsq - min_dlvsq)``.  The first expiry is frozen, so all four
    tensors are zero there.
    """

    surface: ResidualClockDlvSurface
    expiry_time_increments_y: torch.Tensor
    raw_log_loadings: torch.Tensor
    psi: torch.Tensor
    relative_psi: torch.Tensor
    beta: torch.Tensor
    alpha: torch.Tensor
    mu_z: torch.Tensor
    maximum_volatility_equation_residual: torch.Tensor
    maximum_drift_equation_residual: torch.Tensor
    maximum_density_diffusion_mass_residual: torch.Tensor
    maximum_density_diffusion_forward_residual: torch.Tensor
    minimum_density: torch.Tensor
    maximum_absolute_relative_density_diffusion: torch.Tensor


@dataclass(frozen=True)
class FunctionalDlvTransition:
    """One deterministic admissible density-space transition."""

    initial_surface: ResidualClockDlvSurface
    surface: ResidualClockDlvSurface
    standardized_innovation: torch.Tensor
    maximum_variance_flux_recursion_residual: torch.Tensor
    maximum_decoder_residual: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_mean_residual: torch.Tensor
    fallback_coordinate_fraction: torch.Tensor
    minimum_density: torch.Tensor
    minimum_dlvsq: torch.Tensor
    maximum_dlvsq: torch.Tensor


@dataclass(frozen=True)
class FunctionalDlvKernel:
    """A finite cubature of exact-mean functional DLV transitions."""

    initial_surface: ResidualClockDlvSurface
    characteristics: FunctionalDlvCharacteristics
    calendar_time_increment_y: torch.Tensor
    densities_by_branch: torch.Tensor
    dlvsqs_by_branch: torch.Tensor
    branch_probabilities: torch.Tensor
    standardized_innovations: torch.Tensor
    effective_mu_z: torch.Tensor
    effective_log_excess_dlvsq_diffusion: torch.Tensor
    centered_log_increment_variance: torch.Tensor
    maximum_density_tower_residual: torch.Tensor
    maximum_variance_flux_tower_residual: torch.Tensor
    maximum_variance_flux_recursion_residual: torch.Tensor
    maximum_decoder_residual: torch.Tensor
    maximum_mass_residual: torch.Tensor
    maximum_mean_residual: torch.Tensor
    maximum_effective_minus_local_mu_z: torch.Tensor
    maximum_effective_diffusion_minus_loading: torch.Tensor
    fallback_coordinate_fraction: torch.Tensor
    minimum_density: torch.Tensor
    minimum_dlvsq: torch.Tensor
    maximum_dlvsq: torch.Tensor


@dataclass(frozen=True)
class ResidualFrontInsertion:
    """An exact front-slab split and its preservation diagnostics."""

    surface: ResidualClockDlvSurface
    short_dlvsq_scale: torch.Tensor
    maximum_old_density_change: torch.Tensor
    maximum_old_option_change: torch.Tensor
    maximum_split_decoder_residual: torch.Tensor


@dataclass(frozen=True)
class ResidualClockDlvDensification:
    """An exact refinement of a residual-expiry clock and its diagnostics."""

    surface: ResidualClockDlvSurface
    original_expiry_indices: torch.Tensor
    maximum_old_density_change: torch.Tensor
    maximum_old_dlvsq_change: torch.Tensor
    maximum_old_backbone_variance_change: torch.Tensor
    maximum_old_option_change: torch.Tensor
    maximum_inserted_variance_flux_residual: torch.Tensor
    maximum_split_decoder_residual: torch.Tensor


@dataclass(frozen=True)
class ResidualClockDlvRoll:
    """A sampled spot step and its conditional survivor surface."""

    gross_spot_return: torch.Tensor
    branch_indices: torch.Tensor
    front_node_indices: torch.Tensor
    surface: ResidualClockDlvSurface
    maximum_conditional_mass_residual: torch.Tensor
    maximum_conditional_mean_residual: torch.Tensor
    maximum_selected_branch_reconstruction_residual: torch.Tensor


def _geometry(decoder: FlexSanosDecoder) -> CappedUnboundedDlvFlexGeometryConfig:
    geometry = decoder.geometry
    if not isinstance(geometry, CappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("functional DLV stepping requires open bounded DLVs")
    return geometry


def _scalar_time(
    value: float | torch.Tensor,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim != 0 or not bool(torch.isfinite(result) & (result > 0.0)):
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _expiry_time_increments_y(surface: ResidualClockDlvSurface) -> torch.Tensor:
    return torch.diff(
        surface.residual_expiries_y,
        prepend=torch.zeros_like(surface.residual_expiries_y[:1]),
    )


def _calls(support: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    strikes = support[..., 1:-1]
    payoff = torch.relu(support[..., None, :] - strikes[..., :, None])
    return torch.sum(payoff * density[..., None, :], dim=-1)


def _open_bound(
    values: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    lower = torch.as_tensor(minimum, dtype=values.dtype, device=values.device)
    upper = torch.as_tensor(maximum, dtype=values.dtype, device=values.device)
    return torch.clamp(
        values,
        min=torch.nextafter(lower, upper),
        max=torch.nextafter(upper, lower),
    )


def _canonical_initial_density(support: torch.Tensor) -> torch.Tensor:
    """Return the two-node unit-mass martingale law bracketing one."""

    batch, support_size = support.shape
    exact = support == 1.0
    exact_count = torch.sum(exact, dim=-1)
    right_index = torch.sum(support < 1.0, dim=-1)
    bracketed = (right_index > 0) & (right_index < support_size)
    if not bool(torch.all((exact_count == 1) | bracketed)):
        failed = torch.nonzero(~((exact_count == 1) | bracketed)).flatten().tolist()
        raise ValueError(
            "support must contain or strictly bracket X=1 for batch positions "
            f"{failed}"
        )
    if not bool(torch.all(exact_count <= 1)):
        raise ValueError("strictly increasing support cannot contain X=1 twice")

    result = torch.zeros_like(support)
    exact_rows = exact_count == 1
    if bool(torch.any(exact_rows)):
        exact_index = torch.argmax(exact.to(torch.int64), dim=-1)
        result[exact_rows, exact_index[exact_rows]] = 1.0
    interpolated_rows = ~exact_rows
    if bool(torch.any(interpolated_rows)):
        right = right_index[interpolated_rows]
        left = right - 1
        rows = torch.nonzero(interpolated_rows).flatten()
        left_support = support[rows, left]
        right_support = support[rows, right]
        width = right_support - left_support
        left_weight = (right_support - 1.0) / width
        right_weight = (1.0 - left_support) / width
        result[rows, left] = left_weight
        result[rows, right] = right_weight
    return result


def canonicalize_residual_clock_dlv_ingress(
    decoder: FlexSanosDecoder,
    surface: ResidualClockDlvSurface,
    *,
    input_density_tolerance: float = 2.0e-10,
    structural_tolerance: float = 2.0e-10,
    current_spot: torch.Tensor | None = None,
    fixed_cash_strikes: torch.Tensor | None = None,
    is_call: torch.Tensor | None = None,
) -> ResidualClockDlvIngress:
    """Canonicalize a decoded surface once, before any stochastic transition.

    The initial density is replaced by the unique law supported on the two
    adjacent grid nodes bracketing ``X=1`` (or by a point mass when one is a
    node).  Every later density is then rebuilt in float64 from the supplied
    DLVs and the explicit residual clock.  This is an ingress certificate, not
    a projection available to recurrent transitions: a material discrepancy
    from the supplied density tower raises rather than being repaired.

    Fixed-cash option perturbations are reported when ``current_spot``,
    ``fixed_cash_strikes`` and ``is_call`` are supplied together.
    """

    if surface.temporary_frozen_front:
        raise ValueError("roll the temporary frozen front before canonical ingress")
    geometry = _geometry(decoder)
    if not isfinite(input_density_tolerance) or input_density_tolerance <= 0.0:
        raise ValueError("input_density_tolerance must be finite and positive")
    if not isfinite(structural_tolerance) or structural_tolerance <= 0.0:
        raise ValueError("structural_tolerance must be finite and positive")
    if surface.densities.dtype != torch.float64:
        raise ValueError(
            "the exact residual-clock DLV API requires torch.float64 tensors"
        )
    if surface.support_strikes.ndim != 3:
        raise ValueError("support_strikes must have shape (batch, expiries, support)")
    batch, expiries, support_size = surface.support_strikes.shape
    num_inner = support_size - 2
    if expiries <= 0 or support_size != geometry.num_inner_strikes + 2:
        raise ValueError("surface dimensions do not match the decoder geometry")
    expected_shapes = {
        "residual_expiries_y": (expiries,),
        "density0": (batch, support_size),
        "densities": (batch, expiries, support_size),
        "dlvsqs": (batch, expiries, num_inner),
        "backbone_total_vols": (batch, expiries),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(getattr(surface, name).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_shape}"
            )
    tensors = (
        surface.residual_expiries_y,
        surface.support_strikes,
        surface.density0,
        surface.densities,
        surface.dlvsqs,
        surface.backbone_total_vols,
    )
    if any(
        value.dtype != surface.densities.dtype
        or value.device != surface.densities.device
        for value in tensors
    ):
        raise ValueError("surface tensors must share dtype and device")
    if not bool(
        torch.all(torch.isfinite(surface.residual_expiries_y))
        and torch.all(surface.residual_expiries_y > 0.0)
        and torch.all(torch.diff(surface.residual_expiries_y) > 0.0)
    ):
        raise ValueError("residual_expiries_y must be finite, positive and increasing")
    if not bool(
        torch.all(torch.isfinite(surface.support_strikes))
        and torch.all(torch.diff(surface.support_strikes, dim=-1) > 0.0)
        and torch.all(surface.support_strikes == surface.support_strikes[:, :1])
        and torch.all(torch.isfinite(surface.density0))
        and torch.all(torch.isfinite(surface.densities))
        and torch.all(torch.isfinite(surface.dlvsqs))
        and torch.all(surface.dlvsqs > geometry.min_dlvsq)
        and torch.all(surface.dlvsqs < geometry.max_dlvsq)
        and torch.all(torch.isfinite(surface.backbone_total_vols))
        and torch.all(surface.backbone_total_vols >= 0.0)
        and torch.all(
            torch.diff(surface.backbone_total_vols.square(), dim=-1)
            >= -structural_tolerance
        )
    ):
        raise ValueError("surface values outside the ingress structural domain")

    density0 = _canonical_initial_density(surface.support_strikes[:, 0])
    rebuilt: list[torch.Tensor] = []
    previous_density = density0
    for expiry_index, expiry_dt_y in enumerate(
        _expiry_time_increments_y(surface)
    ):
        previous_density = decoder.solve_density_step_batch(
            support=surface.support_strikes[:, expiry_index],
            dlvsqs=surface.dlvsqs[:, expiry_index],
            time_increment_y=expiry_dt_y,
            rhs_density=previous_density,
        )
        rebuilt.append(previous_density)
    densities = torch.stack(rebuilt, dim=1)
    if not bool(torch.all(torch.isfinite(densities)) and torch.all(densities >= 0.0)):
        minimum = torch.min(densities)
        raise FunctionalDlvInadmissibleError(
            "the DLV tower rebuilt from the canonical initial law is not a "
            "finite nonnegative density; minimum="
            f"{float(minimum.detach().cpu()):.6g}"
        )

    maximum_input_density0_perturbation = torch.max(
        torch.abs(density0 - surface.density0)
    )
    maximum_input_density_tower_perturbation = torch.max(
        torch.abs(densities - surface.densities)
    )
    maximum_input_density_perturbation = torch.maximum(
        maximum_input_density0_perturbation,
        maximum_input_density_tower_perturbation,
    )
    if not bool(maximum_input_density_perturbation <= input_density_tolerance):
        raise ValueError(
            "decoded density tower differs materially from its canonical float64 "
            "rebuild: maximum perturbation="
            f"{float(maximum_input_density_perturbation.detach().cpu()):.6g}, "
            f"tolerance={input_density_tolerance:.6g}"
        )

    canonical = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=surface.support_strikes,
        density0=density0,
        densities=densities,
        dlvsqs=surface.dlvsqs,
        backbone_total_vols=surface.backbone_total_vols,
    )
    diagnostics = validate_residual_clock_dlv_surface(
        decoder,
        canonical,
        structural_tolerance=structural_tolerance,
    )
    minimum_dlvsq_bound_margin = torch.minimum(
        torch.min(canonical.dlvsqs - geometry.min_dlvsq),
        torch.min(geometry.max_dlvsq - canonical.dlvsqs),
    )

    option_inputs = (current_spot, fixed_cash_strikes, is_call)
    if any(value is None for value in option_inputs) and not all(
        value is None for value in option_inputs
    ):
        raise ValueError(
            "current_spot, fixed_cash_strikes and is_call must be supplied together"
        )
    maximum_fixed_option_perturbation: torch.Tensor | None = None
    if current_spot is not None:
        assert fixed_cash_strikes is not None and is_call is not None
        input_values = price_residual_clock_options(
            surface,
            current_spot=current_spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )
        canonical_values = price_residual_clock_options(
            canonical,
            current_spot=current_spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )
        maximum_fixed_option_perturbation = torch.max(
            torch.abs(canonical_values - input_values)
        )

    return ResidualClockDlvIngress(
        surface=canonical,
        maximum_input_density_perturbation=maximum_input_density_perturbation,
        maximum_input_density0_perturbation=(
            maximum_input_density0_perturbation
        ),
        maximum_input_density_tower_perturbation=(
            maximum_input_density_tower_perturbation
        ),
        maximum_mass_residual=diagnostics["maximum_mass_residual"],
        maximum_forward_residual=diagnostics["maximum_mean_residual"],
        maximum_recursion_residual=diagnostics["maximum_recursion_residual"],
        minimum_density=torch.min(densities),
        minimum_dlvsq_bound_margin=minimum_dlvsq_bound_margin,
        maximum_fixed_option_perturbation=maximum_fixed_option_perturbation,
    )


def validate_residual_clock_dlv_surface(
    decoder: FlexSanosDecoder,
    surface: ResidualClockDlvSurface,
    *,
    structural_tolerance: float = 2.0e-10,
    check_recursion: bool = True,
) -> dict[str, torch.Tensor]:
    """Validate clock, probability, DLV and optional SANOS invariants.

    A surface marked ``temporary_frozen_front`` may have a finite positive
    first DLV row outside the learned coordinate bounds.  That narrow
    exception never applies to survivor rows.
    """

    geometry = _geometry(decoder)
    if not isfinite(structural_tolerance) or structural_tolerance <= 0.0:
        raise ValueError("structural_tolerance must be finite and positive")
    if surface.densities.dtype != torch.float64:
        raise ValueError(
            "the exact residual-clock DLV API requires torch.float64 tensors"
        )
    if surface.support_strikes.ndim != 3:
        raise ValueError("support_strikes must have shape (batch, expiries, support)")
    batch, expiries, support_size = surface.support_strikes.shape
    num_inner = support_size - 2
    if expiries <= 0 or support_size != geometry.num_inner_strikes + 2:
        raise ValueError("surface dimensions do not match the decoder geometry")
    if not isinstance(surface.temporary_frozen_front, bool):
        raise ValueError("temporary_frozen_front must be boolean")
    if surface.temporary_frozen_front and expiries < 2:
        raise ValueError("a temporary frozen front requires at least one survivor")
    expected_shapes = {
        "residual_expiries_y": (expiries,),
        "density0": (batch, support_size),
        "densities": (batch, expiries, support_size),
        "dlvsqs": (batch, expiries, num_inner),
        "backbone_total_vols": (batch, expiries),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(getattr(surface, name).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_shape}"
            )
    tensors = (
        surface.residual_expiries_y,
        surface.support_strikes,
        surface.density0,
        surface.densities,
        surface.dlvsqs,
        surface.backbone_total_vols,
    )
    if any(
        value.dtype != surface.densities.dtype
        or value.device != surface.densities.device
        for value in tensors
    ):
        raise ValueError("surface tensors must share dtype and device")
    valid_clock = bool(
        torch.all(torch.isfinite(surface.residual_expiries_y))
        and torch.all(surface.residual_expiries_y > 0.0)
        and torch.all(torch.diff(surface.residual_expiries_y) > 0.0)
    )
    if not valid_clock:
        raise ValueError("residual_expiries_y must be finite, positive and increasing")
    dlvsq_domain_valid = torch.isfinite(surface.dlvsqs)
    if surface.temporary_frozen_front:
        dlvsq_domain_valid[:, 0] &= surface.dlvsqs[:, 0] > 0.0
        dlvsq_domain_valid[:, 1:] &= (
            (surface.dlvsqs[:, 1:] > geometry.min_dlvsq)
            & (surface.dlvsqs[:, 1:] < geometry.max_dlvsq)
        )
    else:
        dlvsq_domain_valid &= (
            (surface.dlvsqs > geometry.min_dlvsq)
            & (surface.dlvsqs < geometry.max_dlvsq)
        )
    valid_domain = bool(
        torch.all(torch.isfinite(surface.support_strikes))
        and torch.all(torch.diff(surface.support_strikes, dim=-1) > 0.0)
        and torch.all(
            surface.support_strikes == surface.support_strikes[:, :1]
        )
        and torch.all(torch.isfinite(surface.density0))
        and torch.all(surface.density0 >= 0.0)
        and torch.all(torch.isfinite(surface.densities))
        and torch.all(surface.densities >= 0.0)
        and torch.all(dlvsq_domain_valid)
        and torch.all(torch.isfinite(surface.backbone_total_vols))
        and torch.all(surface.backbone_total_vols >= 0.0)
        and torch.all(
            torch.diff(surface.backbone_total_vols.square(), dim=-1)
            >= -structural_tolerance
        )
    )
    if not valid_domain:
        raise ValueError("surface values lie outside their numerical domain")

    all_densities = torch.cat((surface.density0[:, None], surface.densities), dim=1)
    all_supports = torch.cat(
        (surface.support_strikes[:, :1], surface.support_strikes),
        dim=1,
    )
    mass_residual = torch.max(torch.abs(torch.sum(all_densities, dim=-1) - 1.0))
    mean_residual = torch.max(
        torch.abs(torch.sum(all_densities * all_supports, dim=-1) - 1.0)
    )
    if not bool(
        mass_residual <= structural_tolerance
        and mean_residual <= structural_tolerance
    ):
        raise ValueError("surface densities must have unit mass and forward")

    recursion_residual = torch.zeros_like(mass_residual)
    if check_recursion:
        previous_density = surface.density0
        for expiry_index, expiry_dt_y in enumerate(
            _expiry_time_increments_y(surface)
        ):
            decoded = decoder.solve_density_step_batch(
                support=surface.support_strikes[:, expiry_index],
                dlvsqs=surface.dlvsqs[:, expiry_index],
                time_increment_y=expiry_dt_y,
                rhs_density=previous_density,
            )
            recursion_residual = torch.maximum(
                recursion_residual,
                torch.max(
                    torch.abs(decoded - surface.densities[:, expiry_index])
                ),
            )
            previous_density = surface.densities[:, expiry_index]
        if not bool(recursion_residual <= structural_tolerance):
            raise ValueError(
                "surface DLV recursion is inconsistent with its residual clock: "
                f"{float(recursion_residual.detach().cpu()):.6g}"
            )
    return {
        "maximum_mass_residual": mass_residual,
        "maximum_mean_residual": mean_residual,
        "maximum_recursion_residual": recursion_residual,
    }


def residual_clock_surface_from_decoded(
    decoder: FlexSanosDecoder,
    decoded: SanosSurface,
    *,
    residual_expiries_y: torch.Tensor | None = None,
) -> ResidualClockDlvSurface:
    """Convert a decoded SANOS surface, optionally rebuilding on an exact clock.

    When ``residual_expiries_y`` is omitted, both the decoder clock and decoded
    densities are preserved.  Supplying a clock deliberately rebuilds the
    complete density tower from the fitted DLVs.  This explicit rebuild avoids
    silently relabelling historical float32 expiry buffers as exact daily
    times.
    """

    if decoded.expiries_y.ndim != 2 or not bool(
        torch.all(decoded.expiries_y == decoded.expiries_y[:1])
    ):
        raise ValueError("decoded surfaces must share one expiry clock")
    if residual_expiries_y is None:
        clock = decoded.expiries_y[0]
        densities = decoded.densities
    else:
        clock = residual_expiries_y.to(
            dtype=decoded.densities.dtype,
            device=decoded.densities.device,
        )
        if clock.shape != decoded.expiries_y.shape[1:]:
            raise ValueError("residual_expiries_y does not align with decoded expiries")
        previous_density = decoded.density0
        rebuilt: list[torch.Tensor] = []
        expiry_dts_y = torch.diff(clock, prepend=torch.zeros_like(clock[:1]))
        for expiry_index, expiry_dt_y in enumerate(expiry_dts_y):
            previous_density = decoder.solve_density_step_batch(
                support=decoded.support_strikes[:, expiry_index],
                dlvsqs=decoded.dlvsqs[:, expiry_index],
                time_increment_y=expiry_dt_y,
                rhs_density=previous_density,
            )
            rebuilt.append(previous_density)
        densities = torch.stack(rebuilt, dim=1)
    result = ResidualClockDlvSurface(
        residual_expiries_y=clock,
        support_strikes=decoded.support_strikes,
        density0=decoded.density0,
        densities=densities,
        dlvsqs=decoded.dlvsqs,
        backbone_total_vols=decoded.backbone_total_vols,
    )
    validate_residual_clock_dlv_surface(decoder, result)
    return result


def expand_residual_clock_dlv_surface(
    surface: ResidualClockDlvSurface,
    copies: int,
) -> ResidualClockDlvSurface:
    """Copy one conditional surface into independent batch positions."""

    if surface.support_strikes.shape[0] != 1 or copies <= 0:
        raise ValueError("expansion requires one source surface and positive copies")

    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value.expand(copies, *value.shape[1:]).clone()

    return ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=repeat(surface.support_strikes),
        density0=repeat(surface.density0),
        densities=repeat(surface.densities),
        dlvsqs=repeat(surface.dlvsqs),
        backbone_total_vols=repeat(surface.backbone_total_vols),
        temporary_frozen_front=surface.temporary_frozen_front,
    )


def smooth_residual_clock_log_loadings(
    surface: ResidualClockDlvSurface,
    *,
    annualized_log_volatility: float,
) -> torch.Tensor:
    """Return level, skew, curvature and residual-expiry loading factors."""

    if not isfinite(annualized_log_volatility) or annualized_log_volatility < 0.0:
        raise ValueError("annualized_log_volatility must be finite and nonnegative")
    batch, expiries, support_size = surface.support_strikes.shape
    inner = surface.support_strikes[:, 0, 1:-1]
    log_inner = torch.log(inner)
    strike_scale = torch.amax(torch.abs(log_inner), dim=-1, keepdim=True)
    if not bool(torch.all(strike_scale > 0.0)):
        raise ValueError("support requires a non-unit interior node")
    strike = log_inner / strike_scale
    strike = strike[:, None].expand(batch, max(expiries - 1, 0), support_size - 2)
    if expiries > 1:
        dynamic_expiries = surface.residual_expiries_y[1:]
        if expiries == 2:
            expiry = torch.zeros_like(dynamic_expiries)
        else:
            expiry = 2.0 * (dynamic_expiries - dynamic_expiries[0]) / (
                dynamic_expiries[-1] - dynamic_expiries[0]
            ) - 1.0
        expiry = expiry.reshape(1, -1, 1).expand_as(strike)
        factors = torch.stack(
            (
                torch.ones_like(strike),
                strike,
                2.0 * strike.square() - 1.0,
                expiry,
            ),
            dim=-1,
        )
        factors = factors / torch.linalg.vector_norm(factors, dim=-1, keepdim=True)
    else:
        factors = torch.empty(
            (batch, 0, support_size - 2, 4),
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        )
    taper = torch.sin(
        torch.arange(
            1,
            support_size - 1,
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        )
        * pi
        / (support_size - 1)
    )
    result = torch.zeros(
        (batch, expiries, support_size - 2, 4),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    if expiries > 1:
        result[:, 1:] = (
            annualized_log_volatility
            * taper[None, None, :, None]
            * factors
        )
    return result


def functional_dlv_characteristics(
    decoder: FlexSanosDecoder,
    surface: ResidualClockDlvSurface,
    *,
    raw_log_loadings: torch.Tensor,
    density_mass_floor: float = 0.0,
    structural_tolerance: float = 2.0e-10,
) -> FunctionalDlvCharacteristics:
    """Compute the local analytical DLV drift on the current functional state.

    Every identified dynamic interior density must exceed
    ``density_mass_floor``.  The default zero threshold accepts every
    literally positive float64 mass, including identified tails far below
    machine epsilon relative to total mass.  The theorem uses the literal
    ratio ``psi / q``; density is never floored before that division.  Exact
    zero mass is admitted only when the whole row is boundary-absorbed, in
    which case its loading and local characteristics are completed at zero.
    """

    validate_residual_clock_dlv_surface(
        decoder,
        surface,
        structural_tolerance=structural_tolerance,
    )
    geometry = _geometry(decoder)
    batch, expiries, support_size = surface.support_strikes.shape
    num_inner = support_size - 2
    if raw_log_loadings.ndim == 3:
        if raw_log_loadings.shape[:2] != (expiries, num_inner):
            raise ValueError("shared raw_log_loadings have an invalid shape")
        loadings = raw_log_loadings[None].expand(batch, -1, -1, -1)
    elif raw_log_loadings.ndim == 4:
        if raw_log_loadings.shape[:3] != (batch, expiries, num_inner):
            raise ValueError("batched raw_log_loadings have an invalid shape")
        loadings = raw_log_loadings
    else:
        raise ValueError("raw_log_loadings must have three or four dimensions")
    if loadings.shape[-1] <= 0 or not bool(
        torch.all(torch.isfinite(loadings))
        and torch.all(loadings[:, 0] == 0.0)
    ):
        raise ValueError("loadings must be finite and freeze the first expiry")
    if not isfinite(density_mass_floor) or density_mass_floor < 0.0:
        raise ValueError("density_mass_floor must be finite and nonnegative")
    boundary_absorbed_dynamic_rows = torch.zeros(
        (batch, max(expiries - 1, 0)),
        dtype=torch.bool,
        device=surface.densities.device,
    )
    if expiries > 1:
        dynamic_density = surface.densities[:, 1:]
        dynamic_interior_density = dynamic_density[..., 1:-1]
        identified = dynamic_interior_density > density_mass_floor
        zero_mass = dynamic_interior_density == 0.0
        if not bool(torch.all(identified | zero_mass)):
            failed = torch.nonzero(
                torch.any(~(identified | zero_mass), dim=(-2, -1))
            ).flatten().tolist()
            raise FunctionalDlvInadmissibleError(
                "local psi/q drift is unidentified at a dynamic interior density "
                f"for batch positions {failed}",
                batch_positions=tuple(failed),
            )
        row_has_zero_mass = torch.any(zero_mass, dim=-1)
        boundary_absorbed_dynamic_rows = (
            torch.all(zero_mass, dim=-1)
            & torch.all(
                dynamic_density == surface.densities[:, :-1],
                dim=-1,
            )
        )
        if not bool(
            torch.all(~row_has_zero_mass | boundary_absorbed_dynamic_rows)
        ):
            failed = torch.nonzero(
                row_has_zero_mass & ~boundary_absorbed_dynamic_rows
            ).tolist()
            raise FunctionalDlvInadmissibleError(
                "zero dynamic mass is permitted only for an exactly "
                f"boundary-absorbed row; failed (batch, dynamic row) {failed}"
            )
        if bool(torch.any(boundary_absorbed_dynamic_rows)):
            # Once a row is absorbed, its interior DLV is law-neutral.  Mask
            # the requested loading so the deterministic completion remains
            # fixed and subsequent diagnostics describe the effective law.
            loadings = loadings.clone()
            loadings[:, 1:] = torch.where(
                boundary_absorbed_dynamic_rows[..., None, None],
                torch.zeros_like(loadings[:, 1:]),
                loadings[:, 1:],
            )

    factors = loadings.shape[-1]
    psi = torch.zeros(
        (batch, expiries, support_size, factors),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    relative_psi = torch.zeros_like(psi)
    beta = torch.zeros(
        (batch, expiries, num_inner, factors),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    alpha = torch.zeros_like(surface.dlvsqs)
    mu_z = torch.zeros_like(surface.dlvsqs)
    expiry_dts_y = _expiry_time_increments_y(surface)
    previous_psi = psi[:, 0]
    volatility_residuals: list[torch.Tensor] = []
    drift_residuals: list[torch.Tensor] = []
    for expiry_index in range(1, expiries):
        support = surface.support_strikes[:, expiry_index]
        density = surface.densities[:, expiry_index]
        dlvsq = surface.dlvsqs[:, expiry_index]
        excess = dlvsq - geometry.min_dlvsq
        current_beta = excess[..., None] * loadings[:, expiry_index]
        source = discrete_lvsq_grid_operator_product(
            support,
            density[:, 1:-1, None] * current_beta,
            expiry_time_increment_y=expiry_dts_y[expiry_index],
        )
        rhs = previous_psi - source
        flat_support = support[:, None].expand(-1, factors, -1).reshape(
            batch * factors,
            support_size,
        )
        flat_dlvsq = dlvsq[:, None].expand(-1, factors, -1).reshape(
            batch * factors,
            num_inner,
        )
        current_psi = decoder.solve_density_step_batch(
            support=flat_support,
            dlvsqs=flat_dlvsq,
            time_increment_y=expiry_dts_y[expiry_index],
            rhs_density=rhs.permute(0, 2, 1).reshape(
                batch * factors,
                support_size,
            ),
        ).reshape(batch, factors, support_size).permute(0, 2, 1)
        boundary_absorbed = boundary_absorbed_dynamic_rows[
            :, expiry_index - 1
        ]
        if bool(torch.any(boundary_absorbed)):
            absorbed_diffusion_residual = torch.max(
                torch.abs(current_psi[boundary_absorbed])
            )
            if not bool(absorbed_diffusion_residual <= structural_tolerance):
                raise RuntimeError(
                    "a boundary-absorbed row acquired material density "
                    "diffusion: maximum="
                    f"{float(absorbed_diffusion_residual.detach().cpu()):.6g}"
                )
            current_psi = torch.where(
                boundary_absorbed[:, None, None],
                torch.zeros_like(current_psi),
                current_psi,
            )
        current_relative_psi = torch.zeros_like(current_psi)
        current_identified = density[:, 1:-1] > density_mass_floor
        safe_density = torch.where(
            current_identified,
            density[:, 1:-1],
            torch.ones_like(density[:, 1:-1]),
        )
        current_relative_psi[:, 1:-1] = torch.where(
            current_identified[..., None],
            current_psi[:, 1:-1] / safe_density[..., None],
            torch.zeros_like(current_psi[:, 1:-1]),
        )
        current_alpha = -torch.sum(
            current_relative_psi[:, 1:-1] * current_beta,
            dim=-1,
        )
        current_mu_z = (
            current_alpha / excess
            - 0.5 * torch.sum(loadings[:, expiry_index].square(), dim=-1)
        )
        lhs = current_psi + discrete_lvsq_grid_operator_product(
            support,
            current_psi[:, 1:-1] * dlvsq[..., None],
            expiry_time_increment_y=expiry_dts_y[expiry_index],
        )
        volatility_residuals.append(torch.max(torch.abs(lhs - rhs)))
        drift_residuals.append(
            torch.max(
                torch.abs(
                    density[:, 1:-1] * current_alpha
                    + torch.sum(current_psi[:, 1:-1] * current_beta, dim=-1)
                )
            )
        )
        psi[:, expiry_index] = current_psi
        relative_psi[:, expiry_index] = current_relative_psi
        beta[:, expiry_index] = current_beta
        alpha[:, expiry_index] = current_alpha
        mu_z[:, expiry_index] = current_mu_z
        previous_psi = current_psi

    zero = torch.zeros((), dtype=surface.densities.dtype, device=surface.densities.device)
    return FunctionalDlvCharacteristics(
        surface=surface,
        expiry_time_increments_y=expiry_dts_y,
        raw_log_loadings=loadings,
        psi=psi,
        relative_psi=relative_psi,
        beta=beta,
        alpha=alpha,
        mu_z=mu_z,
        maximum_volatility_equation_residual=(
            torch.max(torch.stack(volatility_residuals))
            if volatility_residuals
            else zero
        ),
        maximum_drift_equation_residual=(
            torch.max(torch.stack(drift_residuals)) if drift_residuals else zero
        ),
        maximum_density_diffusion_mass_residual=torch.max(
            torch.abs(torch.sum(psi, dim=2))
        ),
        maximum_density_diffusion_forward_residual=torch.max(
            torch.abs(
                torch.sum(psi * surface.support_strikes[..., None], dim=2)
            )
        ),
        minimum_density=torch.min(surface.densities),
        maximum_absolute_relative_density_diffusion=torch.max(
            torch.abs(relative_psi)
        ),
    )


def _recover_dlvsq(
    decoder: FlexSanosDecoder,
    *,
    support: torch.Tensor,
    previous_density: torch.Tensor,
    current_density: torch.Tensor,
    expiry_time_increment_y: torch.Tensor,
    fallback: torch.Tensor,
    density_mass_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    geometry = _geometry(decoder)
    calls = _calls(support, current_density)
    previous_calls = _calls(support, previous_density)
    denominator = (
        support[..., 1:-1].square()
        * expiry_time_increment_y
        / (support[..., 2:] - support[..., :-2])
        * current_density[..., 1:-1]
    )
    recovered = (calls - previous_calls) / torch.clamp_min(
        denominator,
        torch.finfo(current_density.dtype).tiny,
    )
    active = current_density[..., 1:-1] > density_mass_floor
    finite = torch.isfinite(recovered)
    in_bounds = (
        (recovered > geometry.min_dlvsq)
        & (recovered < geometry.max_dlvsq)
    )
    active_invalid = active & (~finite | ~in_bounds)
    use_recovered = active & finite & in_bounds
    result = torch.where(use_recovered, recovered, fallback)
    result = _open_bound(
        result,
        minimum=geometry.min_dlvsq,
        maximum=geometry.max_dlvsq,
    )
    decoded = decoder.solve_density_step_batch(
        support=support,
        dlvsqs=result,
        time_increment_y=expiry_time_increment_y,
        rhs_density=previous_density,
    )
    residual = torch.amax(torch.abs(decoded - current_density), dim=-1)
    return result, ~active, active_invalid, residual


def apply_functional_dlv_innovation(
    decoder: FlexSanosDecoder,
    characteristics: FunctionalDlvCharacteristics,
    *,
    standardized_innovation: torch.Tensor,
    calendar_time_increment_y: float | torch.Tensor,
    negativity_materiality_threshold: float = 1.0e-12,
    density_mass_floor: float = 0.0,
    decoder_tolerance: float = 2.0e-9,
) -> FunctionalDlvTransition:
    r"""Apply one supplied innovation in density and variance-flux coordinates.

    For density ``q``, DLV ``ell`` and physical DLV diffusion ``beta``, define

    ``phi = q * ell`` and ``gamma = psi * ell + q * beta``.

    A branch with ``delta = sqrt(h) * xi`` is constructed as

    ``q_branch = q + psi @ delta`` and
    ``phi_branch = phi + gamma @ delta``.

    The SANOS recursion is linear in ``(q, phi)``, so this preserves the full
    density tower algebraically.  At positive mass,
    ``ell_branch = phi_branch / q_branch`` is formed only after both quantities
    have been certified.  At exact ``q_branch = phi_branch = 0`` on a fully
    boundary-absorbed row, the previous DLV is retained as a law-neutral
    deterministic completion.  This avoids the ill-conditioned alternative of
    subtracting two order-one call prices to recover a variance flux of order
    ``q``.
    """

    surface = characteristics.surface
    geometry = _geometry(decoder)
    batch, expiries, _ = surface.support_strikes.shape
    factors = characteristics.raw_log_loadings.shape[-1]
    if surface.temporary_frozen_front and not bool(
        torch.all(characteristics.raw_log_loadings[:, 0] == 0.0)
        and torch.all(characteristics.psi[:, 0] == 0.0)
        and torch.all(characteristics.relative_psi[:, 0] == 0.0)
        and torch.all(characteristics.beta[:, 0] == 0.0)
        and torch.all(characteristics.alpha[:, 0] == 0.0)
        and torch.all(characteristics.mu_z[:, 0] == 0.0)
    ):
        raise ValueError("temporary front characteristics must be exactly frozen")
    innovation = standardized_innovation.to(
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    if innovation.shape != (batch, factors) or not bool(
        torch.all(torch.isfinite(innovation))
    ):
        raise ValueError("standardized_innovation must have shape (batch, factors)")
    step_y = _scalar_time(
        calendar_time_increment_y,
        reference=surface.densities,
        name="calendar_time_increment_y",
    )
    if (
        not isfinite(negativity_materiality_threshold)
        or negativity_materiality_threshold <= 0.0
    ):
        raise ValueError(
            "negativity_materiality_threshold must be finite and positive"
        )
    if not isfinite(density_mass_floor) or density_mass_floor < 0.0:
        raise ValueError("density_mass_floor must be finite and nonnegative")

    density_shock = torch.einsum("btnf,bf->btn", characteristics.psi, innovation)
    next_densities = surface.densities + torch.sqrt(step_y) * density_shock
    minimum_density = torch.min(next_densities)
    boundary_density_valid = torch.isfinite(
        next_densities[..., (0, -1)]
    ) & (
        next_densities[..., (0, -1)] >= 0.0
    )
    next_interior_density = next_densities[..., 1:-1]
    interior_density_identified = next_interior_density > density_mass_floor
    zero_mass_completion = next_interior_density == 0.0
    interior_density_valid = torch.isfinite(next_interior_density) & (
        interior_density_identified | zero_mass_completion
    )
    if not bool(
        torch.all(boundary_density_valid)
        and torch.all(interior_density_valid)
    ):
        qualifier = (
            "materially"
            if bool(
                torch.isfinite(minimum_density)
                and minimum_density < -negativity_materiality_threshold
            )
            else "numerically"
        )
        failed_paths = torch.nonzero(
            torch.any(~boundary_density_valid, dim=(-2, -1))
            | torch.any(~interior_density_valid, dim=(-2, -1))
        ).flatten().tolist()
        raise FunctionalDlvInadmissibleError(
            "functional density transition is nonfinite, negative or has a "
            "positive interior mass at or below the requested identification "
            "threshold; reduce the loading scale "
            f"({qualifier}, minimum={float(minimum_density.detach().cpu()):.6g}, "
            f"batch positions={failed_paths})",
            batch_positions=tuple(failed_paths),
        )

    interior_density = surface.densities[..., 1:-1]
    variance_flux = interior_density * surface.dlvsqs
    variance_flux_diffusion = (
        characteristics.psi[..., 1:-1, :] * surface.dlvsqs[..., None]
        + interior_density[..., None] * characteristics.beta
    )
    variance_flux_shock = torch.einsum(
        "btnf,bf->btn",
        variance_flux_diffusion,
        innovation,
    )
    next_variance_flux = (
        variance_flux + torch.sqrt(step_y) * variance_flux_shock
    )
    previous_branch_densities = torch.cat(
        (surface.density0[:, None], next_densities[:, :-1]),
        dim=1,
    )
    row_has_zero_mass = torch.any(zero_mass_completion, dim=-1)
    # A single zero in an otherwise active row is not identifiable.  The sole
    # admissible zero case is an unchanged boundary law with zero variance
    # flux on every interior coordinate.
    row_is_boundary_absorbed = (
        torch.all(zero_mass_completion, dim=-1)
        & torch.all(next_variance_flux == 0.0, dim=-1)
        & torch.all(next_densities == previous_branch_densities, dim=-1)
    )
    if not bool(torch.all(~row_has_zero_mass | row_is_boundary_absorbed)):
        failed = torch.nonzero(
            row_has_zero_mass & ~row_is_boundary_absorbed
        ).tolist()
        raise FunctionalDlvInadmissibleError(
            "zero-mass DLV completion is permitted only for an exactly "
            f"boundary-absorbed branch row; failed (batch, row) {failed}"
        )
    flux_valid = torch.isfinite(next_variance_flux) & (
        (interior_density_identified & (next_variance_flux > 0.0))
        | (zero_mass_completion & (next_variance_flux == 0.0))
    )
    if not bool(torch.all(flux_valid)):
        failed_paths = torch.nonzero(
            torch.any(~flux_valid, dim=(-2, -1))
        ).flatten().tolist()
        failed_expiry = int(torch.nonzero(~flux_valid)[0, 1])
        raise FunctionalDlvInadmissibleError(
            "functional variance-flux transition became nonfinite or nonpositive, "
            "or is incompatible with its density, for batch positions "
            f"{failed_paths} at expiry index "
            f"{failed_expiry}; reduce the loading scale",
            expiry_index=failed_expiry,
            batch_positions=tuple(failed_paths),
        )

    safe_next_density = torch.where(
        interior_density_identified,
        next_interior_density,
        torch.ones_like(next_interior_density),
    )
    identified_next_dlvsqs = next_variance_flux / safe_next_density
    # At q=phi=0 the DLV has no effect on the transition.  Retaining the prior
    # fitted value is a deterministic completion rather than a fallback on an
    # economically active coordinate.
    next_dlvsqs = torch.where(
        zero_mass_completion,
        surface.dlvsqs,
        identified_next_dlvsqs,
    )
    if surface.temporary_frozen_front:
        next_dlvsqs = torch.cat(
            (surface.dlvsqs[:, :1], next_dlvsqs[:, 1:]),
            dim=1,
        )
    dlvsq_valid = torch.isfinite(next_dlvsqs)
    if surface.temporary_frozen_front:
        dlvsq_valid[:, 0] &= next_dlvsqs[:, 0] > 0.0
        dlvsq_valid[:, 1:] &= (
            (next_dlvsqs[:, 1:] > geometry.min_dlvsq)
            & (next_dlvsqs[:, 1:] < geometry.max_dlvsq)
        )
    else:
        dlvsq_valid &= (
            (next_dlvsqs > geometry.min_dlvsq)
            & (next_dlvsqs < geometry.max_dlvsq)
        )
    if not bool(torch.all(dlvsq_valid)):
        failed_paths = torch.nonzero(
            torch.any(~dlvsq_valid, dim=(-2, -1))
        ).flatten().tolist()
        failed_expiry = int(torch.nonzero(~dlvsq_valid)[0, 1])
        raise FunctionalDlvInadmissibleError(
            "variance flux divided by branch density produced a nonfinite or "
            "out-of-bound DLV for batch positions "
            f"{failed_paths} at expiry index {failed_expiry}; reduce the loading "
            "scale",
            expiry_index=failed_expiry,
            batch_positions=tuple(failed_paths),
        )

    zero = torch.zeros(
        (), dtype=surface.densities.dtype, device=surface.densities.device
    )
    maximum_variance_flux_recursion_residual = zero.clone()
    maximum_decoder_residual = zero.clone()
    previous_density = surface.density0
    for expiry_index in range(expiries):
        flux_lhs = next_densities[:, expiry_index] + (
            discrete_lvsq_grid_operator_product(
                surface.support_strikes[:, expiry_index],
                next_variance_flux[:, expiry_index],
                expiry_time_increment_y=(
                    characteristics.expiry_time_increments_y[expiry_index]
                ),
            )
        )
        maximum_variance_flux_recursion_residual = torch.maximum(
            maximum_variance_flux_recursion_residual,
            torch.max(torch.abs(flux_lhs - previous_density)),
        )
        decoded = decoder.solve_density_step_batch(
            support=surface.support_strikes[:, expiry_index],
            dlvsqs=next_dlvsqs[:, expiry_index],
            time_increment_y=characteristics.expiry_time_increments_y[
                expiry_index
            ],
            rhs_density=previous_density,
        )
        maximum_decoder_residual = torch.maximum(
            maximum_decoder_residual,
            torch.max(torch.abs(decoded - next_densities[:, expiry_index])),
        )
        previous_density = next_densities[:, expiry_index]
    if not bool(
        maximum_variance_flux_recursion_residual <= decoder_tolerance
        and maximum_decoder_residual <= decoder_tolerance
    ):
        raise RuntimeError(
            "variance-flux branch does not reproduce its density transition: "
            "direct recursion residual="
            f"{float(maximum_variance_flux_recursion_residual.detach().cpu()):.6g}, "
            "decoder residual="
            f"{float(maximum_decoder_residual.detach().cpu()):.6g}"
        )
    result_surface = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=surface.support_strikes,
        density0=surface.density0,
        densities=next_densities,
        dlvsqs=next_dlvsqs,
        backbone_total_vols=surface.backbone_total_vols,
        temporary_frozen_front=surface.temporary_frozen_front,
    )
    diagnostics = validate_residual_clock_dlv_surface(
        decoder,
        result_surface,
        structural_tolerance=decoder_tolerance,
    )
    return FunctionalDlvTransition(
        initial_surface=surface,
        surface=result_surface,
        standardized_innovation=innovation,
        maximum_variance_flux_recursion_residual=(
            maximum_variance_flux_recursion_residual
        ),
        maximum_decoder_residual=torch.maximum(
            maximum_decoder_residual,
            diagnostics["maximum_recursion_residual"],
        ),
        maximum_mass_residual=diagnostics["maximum_mass_residual"],
        maximum_mean_residual=diagnostics["maximum_mean_residual"],
        fallback_coordinate_fraction=zero,
        minimum_density=minimum_density,
        minimum_dlvsq=torch.min(next_dlvsqs),
        maximum_dlvsq=torch.max(next_dlvsqs),
    )


def _expand_surface_by_branches(
    surface: ResidualClockDlvSurface,
    branches: int,
) -> ResidualClockDlvSurface:
    batch, expiries, support_size = surface.support_strikes.shape

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(batch, branches, *value.shape[1:]).reshape(
            batch * branches,
            *value.shape[1:],
        )

    return ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=expand(surface.support_strikes),
        density0=expand(surface.density0),
        densities=expand(surface.densities),
        dlvsqs=expand(surface.dlvsqs),
        backbone_total_vols=expand(surface.backbone_total_vols),
        temporary_frozen_front=surface.temporary_frozen_front,
    )


def _expand_characteristics_by_branches(
    characteristics: FunctionalDlvCharacteristics,
    branches: int,
) -> FunctionalDlvCharacteristics:
    surface = characteristics.surface
    batch = len(surface.support_strikes)

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(batch, branches, *value.shape[1:]).reshape(
            batch * branches,
            *value.shape[1:],
        )

    return FunctionalDlvCharacteristics(
        surface=_expand_surface_by_branches(surface, branches),
        expiry_time_increments_y=characteristics.expiry_time_increments_y,
        raw_log_loadings=expand(characteristics.raw_log_loadings),
        psi=expand(characteristics.psi),
        relative_psi=expand(characteristics.relative_psi),
        beta=expand(characteristics.beta),
        alpha=expand(characteristics.alpha),
        mu_z=expand(characteristics.mu_z),
        maximum_volatility_equation_residual=(
            characteristics.maximum_volatility_equation_residual
        ),
        maximum_drift_equation_residual=(
            characteristics.maximum_drift_equation_residual
        ),
        maximum_density_diffusion_mass_residual=(
            characteristics.maximum_density_diffusion_mass_residual
        ),
        maximum_density_diffusion_forward_residual=(
            characteristics.maximum_density_diffusion_forward_residual
        ),
        minimum_density=characteristics.minimum_density,
        maximum_absolute_relative_density_diffusion=(
            characteristics.maximum_absolute_relative_density_diffusion
        ),
    )


def build_functional_dlv_cubature(
    decoder: FlexSanosDecoder,
    surface: ResidualClockDlvSurface,
    *,
    raw_log_loadings: torch.Tensor,
    calendar_time_increment_y: float | torch.Tensor,
    standardized_innovations: torch.Tensor | None = None,
    branch_probabilities: torch.Tensor | None = None,
    negativity_materiality_threshold: float = 1.0e-12,
    density_mass_floor: float = 0.0,
    decoder_tolerance: float = 2.0e-9,
) -> FunctionalDlvKernel:
    """Build an exact-mean bounded cubature from the current functional state."""

    characteristics = functional_dlv_characteristics(
        decoder,
        surface,
        raw_log_loadings=raw_log_loadings,
        density_mass_floor=density_mass_floor,
        structural_tolerance=decoder_tolerance,
    )
    factors = characteristics.raw_log_loadings.shape[-1]
    if standardized_innovations is None:
        if branch_probabilities is not None:
            raise ValueError("branch probabilities require explicit innovations")
        innovations, probabilities = symmetric_rademacher_cubature(
            factors,
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        )
    else:
        if branch_probabilities is None:
            raise ValueError("explicit innovations require branch probabilities")
        innovations = standardized_innovations.to(
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        )
        probabilities = branch_probabilities.to(
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        )
    if innovations.ndim != 2 or innovations.shape[-1] != factors:
        raise ValueError("standardized innovations have an invalid shape")
    if probabilities.shape != (len(innovations),):
        raise ValueError("branch probabilities do not align with innovations")
    probability_tolerance = 128.0 * torch.finfo(surface.densities.dtype).eps
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
                    factors,
                    dtype=surface.densities.dtype,
                    device=surface.densities.device,
                )
            )
        )
        <= probability_tolerance
    ):
        raise ValueError("cubature must have positive weights, zero mean and unit covariance")

    batch, expiries, support_size = surface.support_strikes.shape
    branches = len(innovations)
    expanded_characteristics = _expand_characteristics_by_branches(
        characteristics,
        branches,
    )
    expanded_innovations = innovations[None].expand(batch, -1, -1).reshape(
        batch * branches,
        factors,
    )
    transition = apply_functional_dlv_innovation(
        decoder,
        expanded_characteristics,
        standardized_innovation=expanded_innovations,
        calendar_time_increment_y=calendar_time_increment_y,
        negativity_materiality_threshold=negativity_materiality_threshold,
        density_mass_floor=density_mass_floor,
        decoder_tolerance=decoder_tolerance,
    )
    branch_densities = transition.surface.densities.reshape(
        batch,
        branches,
        expiries,
        support_size,
    )
    branch_dlvsqs = transition.surface.dlvsqs.reshape(
        batch,
        branches,
        expiries,
        support_size - 2,
    )
    density_mean = torch.sum(
        probabilities.reshape(1, branches, 1, 1) * branch_densities,
        dim=1,
    )
    branch_variance_fluxes = (
        branch_densities[..., 1:-1] * branch_dlvsqs
    )
    variance_flux_mean = torch.sum(
        probabilities.reshape(1, branches, 1, 1) * branch_variance_fluxes,
        dim=1,
    )
    initial_variance_flux = (
        surface.densities[..., 1:-1] * surface.dlvsqs
    )
    step_y = _scalar_time(
        calendar_time_increment_y,
        reference=surface.densities,
        name="calendar_time_increment_y",
    )
    log_increment = torch.zeros_like(branch_dlvsqs)
    first_fitted_expiry = 1 if surface.temporary_frozen_front else 0
    if first_fitted_expiry < expiries:
        initial_z = torch.log(
            surface.dlvsqs[:, first_fitted_expiry:]
            - _geometry(decoder).min_dlvsq
        )
        branch_z = torch.log(
            branch_dlvsqs[:, :, first_fitted_expiry:]
            - _geometry(decoder).min_dlvsq
        )
        log_increment[:, :, first_fitted_expiry:] = (
            branch_z - initial_z[:, None]
        )
    effective_mu_z = torch.sum(
        probabilities.reshape(1, branches, 1, 1) * log_increment,
        dim=1,
    ) / step_y
    centered_log_increment = log_increment - step_y * effective_mu_z[:, None]
    effective_log_diffusion = torch.einsum(
        "r,brtn,rf->btnf",
        probabilities,
        centered_log_increment,
        innovations,
    ) / torch.sqrt(step_y)
    centered_log_variance = torch.sum(
        probabilities.reshape(1, branches, 1, 1)
        * centered_log_increment.square(),
        dim=1,
    ) / step_y
    return FunctionalDlvKernel(
        initial_surface=surface,
        characteristics=characteristics,
        calendar_time_increment_y=step_y,
        densities_by_branch=branch_densities,
        dlvsqs_by_branch=branch_dlvsqs,
        branch_probabilities=probabilities,
        standardized_innovations=innovations,
        effective_mu_z=effective_mu_z,
        effective_log_excess_dlvsq_diffusion=effective_log_diffusion,
        centered_log_increment_variance=centered_log_variance,
        maximum_density_tower_residual=torch.max(
            torch.abs(density_mean - surface.densities)
        ),
        maximum_variance_flux_tower_residual=torch.max(
            torch.abs(variance_flux_mean - initial_variance_flux)
        ),
        maximum_variance_flux_recursion_residual=(
            transition.maximum_variance_flux_recursion_residual
        ),
        maximum_decoder_residual=transition.maximum_decoder_residual,
        maximum_mass_residual=transition.maximum_mass_residual,
        maximum_mean_residual=transition.maximum_mean_residual,
        maximum_effective_minus_local_mu_z=torch.max(
            torch.abs(effective_mu_z - characteristics.mu_z)
        ),
        maximum_effective_diffusion_minus_loading=torch.max(
            torch.abs(
                effective_log_diffusion - characteristics.raw_log_loadings
            )
        ),
        fallback_coordinate_fraction=transition.fallback_coordinate_fraction,
        minimum_density=transition.minimum_density,
        minimum_dlvsq=transition.minimum_dlvsq,
        maximum_dlvsq=transition.maximum_dlvsq,
    )


def price_residual_clock_options(
    surface: ResidualClockDlvSurface,
    *,
    current_spot: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Price fixed-cash calls or puts at every remaining surface expiry."""

    batch = len(surface.support_strikes)
    if current_spot.shape != (batch,) or not bool(
        torch.all(torch.isfinite(current_spot) & (current_spot > 0.0))
    ):
        raise ValueError("current_spot must contain one finite positive value per surface")
    if fixed_cash_strikes.ndim == 1:
        strikes = fixed_cash_strikes[None].expand(batch, -1)
    elif fixed_cash_strikes.ndim == 2 and fixed_cash_strikes.shape[0] == batch:
        strikes = fixed_cash_strikes
    else:
        raise ValueError("fixed_cash_strikes must have shape (K,) or (batch, K)")
    if not bool(torch.all(torch.isfinite(strikes) & (strikes > 0.0))):
        raise ValueError("fixed_cash_strikes must be finite and positive")
    if is_call.ndim == 1 and is_call.shape == strikes.shape[1:]:
        calls_flag = is_call[None].expand(batch, -1)
    elif is_call.shape == strikes.shape:
        calls_flag = is_call
    else:
        raise ValueError("is_call must align with fixed_cash_strikes")
    component = current_spot[:, None, None] * surface.support_strikes
    strike = strikes[:, None, None]
    volatility = surface.backbone_total_vols[..., None]
    safe_volatility = torch.clamp_min(
        volatility,
        torch.finfo(component.dtype).tiny,
    )
    d1 = (
        torch.log(component[..., None] / strike)
        + 0.5 * safe_volatility[..., None].square()
    ) / safe_volatility[..., None]
    component_calls = (
        component[..., None] * torch.special.ndtr(d1)
        - strike * torch.special.ndtr(d1 - safe_volatility[..., None])
    )
    component_calls = torch.where(
        volatility[..., None] > 0.0,
        component_calls,
        torch.relu(component[..., None] - strike),
    )
    calls = torch.sum(surface.densities[..., None] * component_calls, dim=2)
    puts = calls - current_spot[:, None, None] + strikes[:, None]
    return torch.where(calls_flag[:, None], calls, puts)


def densify_residual_clock_dlv_surface(
    decoder: FlexSanosDecoder,
    surface: ResidualClockDlvSurface,
    *,
    requested_residual_expiries_y: torch.Tensor,
    current_spot: torch.Tensor | None = None,
    fixed_cash_strikes: torch.Tensor | None = None,
    is_call: torch.Tensor | None = None,
    density_mass_floor: float = 0.0,
    decoder_tolerance: float = 2.0e-9,
) -> ResidualClockDlvDensification:
    r"""Insert exact interior nodes into the residual-expiry clock.

    Within an original slab ending at ``T`` with terminal density ``q_T``
    and DLV ``ell_T``, the density at an inserted time ``u`` is the convex
    interpolation of the slab's endpoint densities. Its variance flux is
    held at the original terminal value, ``q_u ell_u = q_T ell_T``.

    This makes every refined SANOS substep exact and leaves every original
    density and option price unchanged. Black-backbone *total variance* is
    interpolated linearly. Requested nodes may coincide with existing nodes,
    but must lie between the current front and final residual expiries.
    """

    validate_residual_clock_dlv_surface(
        decoder,
        surface,
        structural_tolerance=decoder_tolerance,
    )
    if not isfinite(density_mass_floor) or density_mass_floor < 0.0:
        raise ValueError("density_mass_floor must be finite and nonnegative")
    requested = torch.as_tensor(
        requested_residual_expiries_y,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError("requested_residual_expiries_y must be a nonempty vector")
    if not bool(
        torch.all(torch.isfinite(requested))
        and torch.all(torch.diff(requested) > 0.0)
        and requested[0] >= surface.residual_expiries_y[0]
        and requested[-1] <= surface.residual_expiries_y[-1]
    ):
        raise ValueError(
            "requested residual expiries must be finite, increasing and lie "
            "between the surface front and final expiries"
        )

    original_clock = surface.residual_expiries_y
    requested_is_old = torch.any(
        requested[:, None] == original_clock[None],
        dim=1,
    )
    refined_clock = torch.sort(
        torch.cat((original_clock, requested[~requested_is_old]))
    ).values
    original_expiry_indices = torch.searchsorted(refined_clock, original_clock)
    if not bool(
        torch.all(refined_clock[original_expiry_indices] == original_clock)
    ):
        raise RuntimeError("refined clock lost an original expiry")

    geometry = _geometry(decoder)
    refined_supports: list[torch.Tensor] = []
    refined_densities: list[torch.Tensor] = []
    refined_dlvsqs: list[torch.Tensor] = []
    refined_backbone_total_vols: list[torch.Tensor] = []
    inserted_flux_residuals: list[torch.Tensor] = []
    for refined_y in refined_clock:
        right_index = int(torch.searchsorted(original_clock, refined_y).item())
        if bool(refined_y == original_clock[right_index]):
            refined_supports.append(surface.support_strikes[:, right_index])
            refined_densities.append(surface.densities[:, right_index])
            refined_dlvsqs.append(surface.dlvsqs[:, right_index])
            refined_backbone_total_vols.append(
                surface.backbone_total_vols[:, right_index]
            )
            continue

        left_y = original_clock[right_index - 1]
        right_y = original_clock[right_index]
        weight = (refined_y - left_y) / (right_y - left_y)
        left_density = surface.densities[:, right_index - 1]
        right_density = surface.densities[:, right_index]
        density = (1.0 - weight) * left_density + weight * right_density
        interior_density = density[:, 1:-1]
        terminal_flux = (
            right_density[:, 1:-1] * surface.dlvsqs[:, right_index]
        )
        identified = interior_density > density_mass_floor
        zero_mass = interior_density == 0.0
        if not bool(torch.all(identified | zero_mass)):
            failed = torch.nonzero(
                torch.any(~(identified | zero_mass), dim=-1)
            ).flatten().tolist()
            raise FunctionalDlvInadmissibleError(
                "an interpolated density lies between zero and the requested "
                f"identification floor for batch positions {failed}",
                expiry_index=right_index,
                batch_positions=tuple(failed),
            )
        if not bool(torch.all(terminal_flux[zero_mass] == 0.0)):
            raise FunctionalDlvInadmissibleError(
                "an interpolated zero-mass coordinate has nonzero terminal "
                "variance flux",
                expiry_index=right_index,
            )
        safe_density = torch.where(
            identified,
            interior_density,
            torch.ones_like(interior_density),
        )
        dlvsq = torch.where(
            zero_mass,
            surface.dlvsqs[:, right_index],
            terminal_flux / safe_density,
        )
        valid_dlvsq = (
            torch.isfinite(dlvsq)
            & (dlvsq > geometry.min_dlvsq)
            & (dlvsq < geometry.max_dlvsq)
        )
        if not bool(torch.all(valid_dlvsq)):
            failed = torch.nonzero(
                torch.any(~valid_dlvsq, dim=-1)
            ).flatten().tolist()
            raise FunctionalDlvInadmissibleError(
                "an exact interior slab split leaves the fitted DLV domain for "
                f"batch positions {failed}",
                expiry_index=right_index,
                batch_positions=tuple(failed),
            )
        left_variance = surface.backbone_total_vols[:, right_index - 1].square()
        right_variance = surface.backbone_total_vols[:, right_index].square()
        total_variance = (1.0 - weight) * left_variance + weight * right_variance
        refined_supports.append(surface.support_strikes[:, right_index])
        refined_densities.append(density)
        refined_dlvsqs.append(dlvsq)
        refined_backbone_total_vols.append(torch.sqrt(total_variance))
        inserted_flux_residuals.append(
            torch.max(torch.abs(interior_density * dlvsq - terminal_flux))
        )

    refined = ResidualClockDlvSurface(
        residual_expiries_y=refined_clock,
        support_strikes=torch.stack(refined_supports, dim=1),
        density0=surface.density0,
        densities=torch.stack(refined_densities, dim=1),
        dlvsqs=torch.stack(refined_dlvsqs, dim=1),
        backbone_total_vols=torch.stack(refined_backbone_total_vols, dim=1),
        temporary_frozen_front=surface.temporary_frozen_front,
    )
    diagnostics = validate_residual_clock_dlv_surface(
        decoder,
        refined,
        structural_tolerance=decoder_tolerance,
    )
    old_density_change = torch.max(
        torch.abs(refined.densities[:, original_expiry_indices] - surface.densities)
    )
    old_dlvsq_change = torch.max(
        torch.abs(refined.dlvsqs[:, original_expiry_indices] - surface.dlvsqs)
    )
    old_backbone_variance_change = torch.max(
        torch.abs(
            refined.backbone_total_vols[:, original_expiry_indices].square()
            - surface.backbone_total_vols.square()
        )
    )
    zero = torch.zeros(
        (), dtype=surface.densities.dtype, device=surface.densities.device
    )
    old_option_change = zero
    supplied_option_argument = (
        current_spot is not None
        or fixed_cash_strikes is not None
        or is_call is not None
    )
    if supplied_option_argument:
        if fixed_cash_strikes is None or is_call is None:
            raise ValueError("fixed_cash_strikes and is_call must be supplied together")
        spot = (
            torch.ones(
                len(surface.support_strikes),
                dtype=surface.densities.dtype,
                device=surface.densities.device,
            )
            if current_spot is None
            else current_spot
        )
        before = price_residual_clock_options(
            surface,
            current_spot=spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )
        after = price_residual_clock_options(
            refined,
            current_spot=spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )[:, original_expiry_indices]
        old_option_change = torch.max(torch.abs(after - before))
    return ResidualClockDlvDensification(
        surface=refined,
        original_expiry_indices=original_expiry_indices,
        maximum_old_density_change=old_density_change,
        maximum_old_dlvsq_change=old_dlvsq_change,
        maximum_old_backbone_variance_change=old_backbone_variance_change,
        maximum_old_option_change=old_option_change,
        maximum_inserted_variance_flux_residual=(
            torch.max(torch.stack(inserted_flux_residuals))
            if inserted_flux_residuals
            else zero
        ),
        maximum_split_decoder_residual=diagnostics[
            "maximum_recursion_residual"
        ],
    )


def exact_functional_dlv_option_tower_residual(
    kernel: FunctionalDlvKernel,
    *,
    current_spot: torch.Tensor,
    fixed_cash_strikes: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Return the calendar-cubature option tower residual before a chronological roll."""

    surface = kernel.initial_surface
    batch, branches, expiries, support_size = kernel.densities_by_branch.shape
    repeated = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=surface.support_strikes[:, None].expand(
            batch, branches, expiries, support_size
        ).reshape(batch * branches, expiries, support_size),
        density0=surface.density0[:, None].expand(
            batch, branches, support_size
        ).reshape(batch * branches, support_size),
        densities=kernel.densities_by_branch.reshape(
            batch * branches,
            expiries,
            support_size,
        ),
        dlvsqs=kernel.dlvsqs_by_branch.reshape(
            batch * branches,
            expiries,
            support_size - 2,
        ),
        backbone_total_vols=surface.backbone_total_vols[:, None].expand(
            batch, branches, expiries
        ).reshape(batch * branches, expiries),
        temporary_frozen_front=surface.temporary_frozen_front,
    )
    initial = price_residual_clock_options(
        surface,
        current_spot=current_spot,
        fixed_cash_strikes=fixed_cash_strikes,
        is_call=is_call,
    )
    branch_strikes = (
        fixed_cash_strikes[:, None].expand(batch, branches, -1).reshape(
            batch * branches,
            -1,
        )
        if fixed_cash_strikes.ndim == 2
        else fixed_cash_strikes
    )
    branch_is_call = (
        is_call[:, None].expand(batch, branches, -1).reshape(
            batch * branches,
            -1,
        )
        if is_call.ndim == 2
        else is_call
    )
    branch_values = price_residual_clock_options(
        repeated,
        current_spot=current_spot[:, None].expand(batch, branches).reshape(-1),
        fixed_cash_strikes=branch_strikes,
        is_call=branch_is_call,
    ).reshape(batch, branches, expiries, -1)
    expected = torch.sum(
        kernel.branch_probabilities.reshape(1, branches, 1, 1) * branch_values,
        dim=1,
    )
    return torch.max(torch.abs(expected - initial))


def insert_residual_front_by_density_interpolation(
    decoder: FlexSanosDecoder,
    surface: ResidualClockDlvSurface,
    *,
    elapsed_y: float | torch.Tensor,
    current_spot: torch.Tensor | None = None,
    fixed_cash_strikes: torch.Tensor | None = None,
    is_call: torch.Tensor | None = None,
    density_mass_floor: float = 0.0,
    decoder_tolerance: float = 2.0e-9,
) -> ResidualFrontInsertion:
    r"""Insert a front slab by convex interpolation of its terminal density.

    Write the old front horizon as ``g``, the inserted horizon as ``h`` and
    ``lambda = h / g``.  The interpolated density and call curve are

    ``q_h = (1-lambda) q_0 + lambda q_g`` and
    ``C_h = (1-lambda) C_0 + lambda C_g``.

    The discrete SANOS recursion uses an annualized DLV ``ell`` through

    ``C_g - C_0 = g * box(q_g * ell_g)``,

    where ``box`` excludes the time increment.  It follows that

    ``ell_h = ell_g * q_g / q_h``

    on the inserted slab: the factor ``lambda * g / h`` is exactly one.  On
    the remaining slab, ``C_g-C_h=(g-h)*box(q_g*ell_g)``, so its DLV remains
    ``ell_g``.  The temporary short DLV is not a fitted coordinate: it may lie
    outside the configured open bounds, but must be finite and strictly
    positive.  It is frozen by the functional transition and removed by the
    next roll.  The implementation independently decodes both legs and raises
    :class:`FunctionalDlvInadmissibleError` if the exact split fails.

    The Black backbone follows the explicit constant-annualized-variance
    policy inside the old front gap.  All old densities, DLVs after the split,
    total variances and option prices are retained exactly.
    """

    if surface.temporary_frozen_front:
        raise ValueError("roll the temporary frozen front before inserting another")
    validate_residual_clock_dlv_surface(
        decoder,
        surface,
        structural_tolerance=decoder_tolerance,
    )
    if not isfinite(density_mass_floor) or density_mass_floor < 0.0:
        raise ValueError("density_mass_floor must be finite and nonnegative")
    step_y = _scalar_time(
        elapsed_y,
        reference=surface.densities,
        name="elapsed_y",
    )
    front_y = surface.residual_expiries_y[0]
    zero = torch.zeros(
        (),
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    clock_tolerance = 128.0 * torch.finfo(surface.densities.dtype).eps * torch.maximum(
        torch.ones_like(front_y),
        torch.abs(front_y),
    )
    if bool(torch.abs(front_y - step_y) <= clock_tolerance):
        return ResidualFrontInsertion(
            surface=surface,
            short_dlvsq_scale=torch.ones(
                len(surface.support_strikes),
                dtype=surface.densities.dtype,
                device=surface.densities.device,
            ),
            maximum_old_density_change=zero,
            maximum_old_option_change=zero,
            maximum_split_decoder_residual=zero,
        )
    if bool(front_y < step_y):
        raise ValueError("the front residual expiry is shorter than elapsed_y")

    support = surface.support_strikes[:, 0]
    old_front_density = surface.densities[:, 0]
    old_front_dlvsq = surface.dlvsqs[:, 0]
    interpolation_weight = step_y / front_y
    front_density = (
        (1.0 - interpolation_weight) * surface.density0
        + interpolation_weight * old_front_density
    )
    front_interior_density = front_density[:, 1:-1]
    identified = front_interior_density > density_mass_floor
    zero_mass = front_interior_density == 0.0
    if not bool(torch.all(identified | zero_mass)):
        failed = torch.nonzero(
            torch.any(~(identified | zero_mass), dim=-1)
        ).flatten().tolist()
        raise FunctionalDlvInadmissibleError(
            "the interpolated front density has an interior coordinate between "
            "zero and the requested identification floor for batch positions "
            f"{failed}",
            expiry_index=0,
            batch_positions=tuple(failed),
        )
    old_front_flux = old_front_dlvsq * old_front_density[:, 1:-1]
    if not bool(torch.all(old_front_flux[zero_mass] == 0.0)):
        raise FunctionalDlvInadmissibleError(
            "an interpolated zero-mass front coordinate has nonzero old "
            "variance flux",
            expiry_index=0,
        )
    path_has_zero_mass = torch.any(zero_mass, dim=-1)
    path_is_boundary_absorbed = (
        torch.all(zero_mass, dim=-1)
        & torch.all(old_front_flux == 0.0, dim=-1)
        & torch.all(front_density == surface.density0, dim=-1)
        & torch.all(old_front_density == surface.density0, dim=-1)
    )
    if not bool(torch.all(~path_has_zero_mass | path_is_boundary_absorbed)):
        failed = torch.nonzero(
            path_has_zero_mass & ~path_is_boundary_absorbed
        ).flatten().tolist()
        raise FunctionalDlvInadmissibleError(
            "zero-mass front completion is permitted only for an exactly "
            f"boundary-absorbed path; failed batch positions {failed}",
            expiry_index=0,
            batch_positions=tuple(failed),
        )
    safe_front_density = torch.where(
        identified,
        front_interior_density,
        torch.ones_like(front_interior_density),
    )
    identified_short = old_front_flux / safe_front_density
    closed_form_short = torch.where(
        zero_mass,
        old_front_dlvsq,
        identified_short,
    )
    short_is_valid = torch.isfinite(closed_form_short) & (
        closed_form_short > 0.0
    )
    if not bool(torch.all(short_is_valid)):
        failed = torch.nonzero(
            torch.any(~short_is_valid, dim=-1)
        ).flatten().tolist()
        raise FunctionalDlvInadmissibleError(
            "the interpolated temporary front DLV is nonfinite or nonpositive "
            f"for batch positions {failed}",
            expiry_index=0,
            batch_positions=tuple(failed),
        )
    flux_formula_residual = torch.amax(
        torch.abs(front_interior_density * closed_form_short - old_front_flux),
        dim=-1,
    )
    closed_form_density = decoder.solve_density_step_batch(
        support=support,
        dlvsqs=closed_form_short,
        time_increment_y=step_y,
        rhs_density=surface.density0,
    )
    short_decoder_error = torch.amax(
        torch.abs(closed_form_density - front_density),
        dim=-1,
    )
    residual_density = decoder.solve_density_step_batch(
        support=support,
        dlvsqs=old_front_dlvsq,
        time_increment_y=front_y - step_y,
        rhs_density=front_density,
    )
    residual_decoder_error = torch.amax(
        torch.abs(residual_density - old_front_density),
        dim=-1,
    )
    maximum_split_residual = torch.max(
        torch.stack(
            (
                torch.max(flux_formula_residual),
                torch.max(short_decoder_error),
                torch.max(residual_decoder_error),
            )
        )
    )
    if not bool(maximum_split_residual <= decoder_tolerance):
        failed = torch.nonzero(
            (flux_formula_residual > decoder_tolerance)
            | (short_decoder_error > decoder_tolerance)
            | (residual_decoder_error > decoder_tolerance)
        ).flatten().tolist()
        raise FunctionalDlvInadmissibleError(
            "the convex-density front split does not satisfy both SANOS recursions "
            f"for batch positions {failed}: "
            f"{float(maximum_split_residual.detach().cpu()):.6g}",
            batch_positions=tuple(failed),
        )

    new_surface = ResidualClockDlvSurface(
        residual_expiries_y=torch.cat((step_y[None], surface.residual_expiries_y)),
        support_strikes=torch.cat(
            (surface.support_strikes[:, :1], surface.support_strikes),
            dim=1,
        ),
        density0=surface.density0,
        densities=torch.cat((front_density[:, None], surface.densities), dim=1),
        dlvsqs=torch.cat(
            (
                closed_form_short[:, None],
                old_front_dlvsq[:, None],
                surface.dlvsqs[:, 1:],
            ),
            dim=1,
        ),
        backbone_total_vols=torch.cat(
            (
                surface.backbone_total_vols[:, :1]
                * torch.sqrt(interpolation_weight),
                surface.backbone_total_vols,
            ),
            dim=1,
        ),
        temporary_frozen_front=True,
    )
    diagnostics = validate_residual_clock_dlv_surface(
        decoder,
        new_surface,
        structural_tolerance=decoder_tolerance,
    )
    old_density_change = torch.max(
        torch.abs(new_surface.densities[:, 1:] - surface.densities)
    )
    old_option_change = zero
    supplied_option_argument = (
        current_spot is not None
        or fixed_cash_strikes is not None
        or is_call is not None
    )
    if supplied_option_argument:
        if fixed_cash_strikes is None or is_call is None:
            raise ValueError("fixed_cash_strikes and is_call must be supplied together")
        spot = (
            torch.ones(
                len(surface.support_strikes),
                dtype=surface.densities.dtype,
                device=surface.densities.device,
            )
            if current_spot is None
            else current_spot
        )
        before = price_residual_clock_options(
            surface,
            current_spot=spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )
        after = price_residual_clock_options(
            new_surface,
            current_spot=spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )[:, 1:]
        old_option_change = torch.max(torch.abs(after - before))
    return ResidualFrontInsertion(
        surface=new_surface,
        short_dlvsq_scale=torch.mean(
            closed_form_short / old_front_dlvsq,
            dim=-1,
        ),
        maximum_old_density_change=old_density_change,
        maximum_old_option_change=old_option_change,
        maximum_split_decoder_residual=torch.max(
            torch.stack(
                (
                    maximum_split_residual,
                    diagnostics["maximum_recursion_residual"],
                )
            )
        ),
    )


def insert_residual_front(
    decoder: FlexSanosDecoder,
    surface: ResidualClockDlvSurface,
    *,
    elapsed_y: float | torch.Tensor,
    current_spot: torch.Tensor | None = None,
    fixed_cash_strikes: torch.Tensor | None = None,
    is_call: torch.Tensor | None = None,
    scale_candidates: tuple[float, ...] = DEFAULT_FRONT_SPLIT_SCALE_CANDIDATES,
    density_mass_floor: float = 1.0e-14,
    decoder_tolerance: float = 2.0e-9,
) -> ResidualFrontInsertion:
    """Insert an exact elapsed-time front by splitting the first residual slab."""

    if surface.temporary_frozen_front:
        raise ValueError("roll the temporary frozen front before inserting another")
    validate_residual_clock_dlv_surface(
        decoder,
        surface,
        structural_tolerance=decoder_tolerance,
    )
    step_y = _scalar_time(
        elapsed_y,
        reference=surface.densities,
        name="elapsed_y",
    )
    front_y = surface.residual_expiries_y[0]
    zero = torch.zeros((), dtype=surface.densities.dtype, device=surface.densities.device)
    clock_tolerance = 128.0 * torch.finfo(surface.densities.dtype).eps * torch.maximum(
        torch.ones_like(front_y),
        torch.abs(front_y),
    )
    if bool(torch.abs(front_y - step_y) <= clock_tolerance):
        return ResidualFrontInsertion(
            surface=surface,
            short_dlvsq_scale=torch.ones(
                len(surface.support_strikes),
                dtype=surface.densities.dtype,
                device=surface.densities.device,
            ),
            maximum_old_density_change=zero,
            maximum_old_option_change=zero,
            maximum_split_decoder_residual=zero,
        )
    if bool(front_y < step_y):
        raise ValueError("the front residual expiry is shorter than elapsed_y")

    geometry = _geometry(decoder)
    batch = len(surface.support_strikes)
    support = surface.support_strikes[:, 0]
    old_front_density = surface.densities[:, 0]
    old_front_dlvsq = surface.dlvsqs[:, 0]
    chosen_short = torch.empty_like(old_front_dlvsq)
    chosen_residual = torch.empty_like(old_front_dlvsq)
    chosen_density = torch.empty_like(old_front_density)
    chosen_scale = torch.full(
        (batch,),
        torch.nan,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    chosen_decoder_residual = torch.full_like(chosen_scale, torch.inf)
    unresolved = torch.ones(batch, dtype=torch.bool, device=surface.densities.device)
    for scale_value in scale_candidates:
        if not bool(torch.any(unresolved)):
            break
        unresolved_indices = torch.nonzero(unresolved).flatten()
        scale = torch.as_tensor(
            scale_value,
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        )
        candidate_short = geometry.min_dlvsq + scale * (
            old_front_dlvsq[unresolved_indices] - geometry.min_dlvsq
        )
        candidate_short = _open_bound(
            candidate_short,
            minimum=geometry.min_dlvsq,
            maximum=geometry.max_dlvsq,
        )
        front_density = decoder.solve_density_step_batch(
            support=support[unresolved_indices],
            dlvsqs=candidate_short,
            time_increment_y=step_y,
            rhs_density=surface.density0[unresolved_indices],
        )
        recovered, tail_fallback, active_invalid, residual = _recover_dlvsq(
            decoder,
            support=support[unresolved_indices],
            previous_density=front_density,
            current_density=old_front_density[unresolved_indices],
            expiry_time_increment_y=front_y - step_y,
            fallback=old_front_dlvsq[unresolved_indices],
            density_mass_floor=density_mass_floor,
        )
        front_mass = torch.abs(torch.sum(front_density, dim=-1) - 1.0)
        front_mean = torch.abs(
            torch.sum(front_density * support[unresolved_indices], dim=-1) - 1.0
        )
        admissible = (
            ~torch.any(active_invalid, dim=-1)
            & ~torch.any(tail_fallback, dim=-1)
            & (torch.amin(front_density, dim=-1) >= 0.0)
            & (front_mass <= decoder_tolerance)
            & (front_mean <= decoder_tolerance)
            & (residual <= decoder_tolerance)
        )
        accepted_indices = unresolved_indices[admissible]
        chosen_short[accepted_indices] = candidate_short[admissible]
        chosen_residual[accepted_indices] = recovered[admissible]
        chosen_density[accepted_indices] = front_density[admissible]
        chosen_scale[accepted_indices] = scale
        chosen_decoder_residual[accepted_indices] = residual[admissible]
        unresolved[accepted_indices] = False
    if bool(torch.any(unresolved)):
        failed = torch.nonzero(unresolved).flatten().tolist()
        raise FunctionalDlvInadmissibleError(
            "the supplied scale_candidates found no admissible exact front split "
            f"for batch positions {failed}; expand the candidates or use the "
            "convex-density interpolation splitter",
            batch_positions=tuple(failed),
        )

    new_surface = ResidualClockDlvSurface(
        residual_expiries_y=torch.cat((step_y[None], surface.residual_expiries_y)),
        support_strikes=torch.cat(
            (surface.support_strikes[:, :1], surface.support_strikes),
            dim=1,
        ),
        density0=surface.density0,
        densities=torch.cat((chosen_density[:, None], surface.densities), dim=1),
        dlvsqs=torch.cat(
            (
                chosen_short[:, None],
                chosen_residual[:, None],
                surface.dlvsqs[:, 1:],
            ),
            dim=1,
        ),
        backbone_total_vols=torch.cat(
            (
                surface.backbone_total_vols[:, :1]
                * torch.sqrt(step_y / front_y),
                surface.backbone_total_vols,
            ),
            dim=1,
        ),
    )
    diagnostics = validate_residual_clock_dlv_surface(
        decoder,
        new_surface,
        structural_tolerance=decoder_tolerance,
    )
    old_density_change = torch.max(
        torch.abs(new_surface.densities[:, 1:] - surface.densities)
    )
    old_option_change = zero
    supplied_option_argument = (
        current_spot is not None
        or fixed_cash_strikes is not None
        or is_call is not None
    )
    if supplied_option_argument:
        if fixed_cash_strikes is None or is_call is None:
            raise ValueError("fixed_cash_strikes and is_call must be supplied together")
        spot = (
            torch.ones(
                batch,
                dtype=surface.densities.dtype,
                device=surface.densities.device,
            )
            if current_spot is None
            else current_spot
        )
        before = price_residual_clock_options(
            surface,
            current_spot=spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )
        after = price_residual_clock_options(
            new_surface,
            current_spot=spot,
            fixed_cash_strikes=fixed_cash_strikes,
            is_call=is_call,
        )[:, 1:]
        old_option_change = torch.max(torch.abs(after - before))
    return ResidualFrontInsertion(
        surface=new_surface,
        short_dlvsq_scale=chosen_scale,
        maximum_old_density_change=old_density_change,
        maximum_old_option_change=old_option_change,
        maximum_split_decoder_residual=torch.maximum(
            torch.max(chosen_decoder_residual),
            diagnostics["maximum_recursion_residual"],
        ),
    )


def _sample_categorical(
    probabilities: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    if probabilities.shape[:-1] != uniforms.shape:
        raise ValueError("probabilities and uniforms have incompatible shapes")
    if not bool(
        torch.all(torch.isfinite(uniforms) & (uniforms >= 0.0) & (uniforms < 1.0))
    ):
        raise ValueError("categorical uniforms must lie in [0, 1)")
    normalized = torch.clamp_min(probabilities, 0.0)
    normalized = normalized / torch.clamp_min(
        torch.sum(normalized, dim=-1, keepdim=True),
        torch.finfo(normalized.dtype).tiny,
    )
    cdf = torch.cumsum(normalized, dim=-1)
    cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), dim=-1)
    # Right-continuous inversion skips leading or interior zero-mass plateaus,
    # including when the supplied uniform is exactly zero or a CDF boundary.
    indices = torch.sum(uniforms[..., None] >= cdf, dim=-1).to(torch.int64)
    selected_probability = torch.gather(
        normalized, dim=-1, index=indices[..., None]
    ).squeeze(-1)
    if not bool(torch.all(selected_probability > 0.0)):
        raise RuntimeError("inverse-CDF sampling selected a zero-probability outcome")
    return indices


def roll_residual_clock_dlv(
    decoder: FlexSanosDecoder,
    kernel: FunctionalDlvKernel,
    *,
    surface_branch_uniforms: torch.Tensor,
    front_node_uniforms: torch.Tensor,
    front_backbone_normals: torch.Tensor,
    elapsed_y: float | torch.Tensor,
    structural_tolerance: float = 2.0e-9,
) -> ResidualClockDlvRoll:
    """Select a cubature branch, draw front spot and condition all survivors."""

    surface = kernel.initial_surface
    batch, expiries, support_size = surface.support_strikes.shape
    if expiries < 2:
        raise ValueError("at least two expiries are required to roll")
    expected_shape = (batch,)
    if not (
        surface_branch_uniforms.shape
        == front_node_uniforms.shape
        == front_backbone_normals.shape
        == expected_shape
    ):
        raise ValueError("all roll innovations must have shape (batch,)")
    if not bool(torch.all(torch.isfinite(front_backbone_normals))):
        raise ValueError("front_backbone_normals must be finite")
    step_y = _scalar_time(
        elapsed_y,
        reference=surface.densities,
        name="elapsed_y",
    )
    clock_tolerance = 128.0 * torch.finfo(surface.densities.dtype).eps
    if not bool(
        torch.abs(surface.residual_expiries_y[0] - step_y)
        <= clock_tolerance
    ):
        raise ValueError("insert an elapsed-time front before rolling")
    if not bool(
        torch.abs(kernel.calendar_time_increment_y - step_y)
        <= clock_tolerance
    ):
        raise ValueError(
            "elapsed_y must match the functional cubature calendar interval"
        )
    branches = len(kernel.branch_probabilities)
    branch_indices = _sample_categorical(
        kernel.branch_probabilities[None].expand(batch, branches),
        surface_branch_uniforms,
    )
    batch_index = torch.arange(batch, device=surface.densities.device)
    selected_densities = kernel.densities_by_branch[
        batch_index,
        branch_indices,
    ]
    selected_dlvsqs = kernel.dlvsqs_by_branch[
        batch_index,
        branch_indices,
    ]
    front_node_indices = _sample_categorical(
        selected_densities[:, 0],
        front_node_uniforms,
    )
    front_support = surface.support_strikes[:, 0]
    latent_front = torch.gather(
        front_support,
        -1,
        front_node_indices[:, None],
    ).squeeze(-1)
    front_vol = surface.backbone_total_vols[:, 0]
    gross_spot = latent_front * torch.exp(
        -0.5 * front_vol.square()
        + front_vol * front_backbone_normals
    )
    relative_support = front_support / latent_front[:, None]
    density0 = torch.nn.functional.one_hot(
        front_node_indices,
        num_classes=support_size,
    ).to(dtype=surface.densities.dtype)
    conditional = density0
    densities: list[torch.Tensor] = []
    expiry_dts_y = _expiry_time_increments_y(surface)
    mass_residual = torch.zeros(
        (), dtype=surface.densities.dtype, device=surface.densities.device
    )
    mean_residual = torch.zeros_like(mass_residual)
    for expiry_index in range(1, expiries):
        conditional = decoder.solve_density_step_batch(
            support=surface.support_strikes[:, expiry_index],
            dlvsqs=selected_dlvsqs[:, expiry_index],
            time_increment_y=expiry_dts_y[expiry_index],
            rhs_density=conditional,
        )
        densities.append(conditional)
        mass_residual = torch.maximum(
            mass_residual,
            torch.max(torch.abs(torch.sum(conditional, dim=-1) - 1.0)),
        )
        mean_residual = torch.maximum(
            mean_residual,
            torch.max(
                torch.abs(
                    torch.sum(conditional * relative_support, dim=-1) - 1.0
                )
            ),
        )
    survivor_count = expiries - 1
    rolled = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y[1:] - step_y,
        support_strikes=relative_support[:, None].expand(
            batch,
            survivor_count,
            support_size,
        ),
        density0=density0,
        densities=torch.stack(densities, dim=1),
        dlvsqs=selected_dlvsqs[:, 1:],
        backbone_total_vols=torch.sqrt(
            torch.clamp_min(
                surface.backbone_total_vols[:, 1:].square()
                - front_vol[:, None].square(),
                0.0,
            )
        ),
        temporary_frozen_front=False,
    )
    validate_residual_clock_dlv_surface(
        decoder,
        rolled,
        structural_tolerance=structural_tolerance,
    )
    return ResidualClockDlvRoll(
        gross_spot_return=gross_spot,
        branch_indices=branch_indices,
        front_node_indices=front_node_indices,
        surface=rolled,
        maximum_conditional_mass_residual=mass_residual,
        maximum_conditional_mean_residual=mean_residual,
        maximum_selected_branch_reconstruction_residual=(
            kernel.maximum_decoder_residual
        ),
    )


__all__ = [
    "DEFAULT_FRONT_SPLIT_SCALE_CANDIDATES",
    "FunctionalDlvCharacteristics",
    "FunctionalDlvInadmissibleError",
    "FunctionalDlvKernel",
    "FunctionalDlvTransition",
    "ResidualClockDlvIngress",
    "ResidualClockDlvDensification",
    "ResidualClockDlvRoll",
    "ResidualClockDlvSurface",
    "ResidualFrontInsertion",
    "apply_functional_dlv_innovation",
    "build_functional_dlv_cubature",
    "canonicalize_residual_clock_dlv_ingress",
    "densify_residual_clock_dlv_surface",
    "exact_functional_dlv_option_tower_residual",
    "expand_residual_clock_dlv_surface",
    "functional_dlv_characteristics",
    "insert_residual_front",
    "insert_residual_front_by_density_interpolation",
    "price_residual_clock_options",
    "residual_clock_surface_from_decoded",
    "roll_residual_clock_dlv",
    "smooth_residual_clock_log_loadings",
    "validate_residual_clock_dlv_surface",
]
