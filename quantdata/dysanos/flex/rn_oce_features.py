"""Fixed-size policy features for residual-clock risk-neutral DLV surfaces.

The functional risk-neutral transition generally leaves the Direct-200
decoder manifold.  An OCE policy therefore needs a deterministic state that
is defined directly on the transitioned surface.  This module encodes the
actual front twenty one-business-day slabs and derives floating-book NMS
strikes from the same risk-neutral option law.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.residual_clock_dlv import (
    ResidualClockDlvSurface,
    price_residual_clock_options,
)

RN_OCE_DAILY_SLABS = 20
RN_OCE_EXPIRY_DAYS = (5, 10, 15, 20)
RN_OCE_PRODUCTION_POLICY_STATE_SIZE = 4222
RN_OCE_POLICY_STATE_SCHEMA = "dysanos-rn-oce-policy-state-v1"


@dataclass(frozen=True)
class RnOcePolicyStateMetadata:
    """Exact layout and numerical conventions of an encoded policy state."""

    schema: str
    shape: tuple[int, int]
    daily_slab_days: tuple[int, ...]
    support_size: int
    num_inner_strikes: int
    feature_size: int
    minimum_dlvsq: float
    minimum_excess_dlvsq: float
    log_normalized_support_slice: tuple[int, int]
    log_excess_dlvsq_slice: tuple[int, int]
    backbone_total_vol_slice: tuple[int, int]
    density0_reconstruction: str
    dlvsq_flattening: str
    dtype: str
    device: str

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata for feature-cache manifests."""

        return {
            "schema": self.schema,
            "shape": list(self.shape),
            "daily_slab_days": list(self.daily_slab_days),
            "support_size": self.support_size,
            "num_inner_strikes": self.num_inner_strikes,
            "feature_size": self.feature_size,
            "minimum_dlvsq": self.minimum_dlvsq,
            "minimum_excess_dlvsq": self.minimum_excess_dlvsq,
            "log_normalized_support_slice": list(
                self.log_normalized_support_slice
            ),
            "log_excess_dlvsq_slice": list(self.log_excess_dlvsq_slice),
            "backbone_total_vol_slice": list(self.backbone_total_vol_slice),
            "density0_reconstruction": self.density0_reconstruction,
            "dlvsq_flattening": self.dlvsq_flattening,
            "dtype": self.dtype,
            "device": self.device,
        }


@dataclass(frozen=True)
class RnOcePolicyState:
    """Encoded policy-state tensor and its self-describing layout."""

    values: torch.Tensor
    metadata: RnOcePolicyStateMetadata


@dataclass(frozen=True)
class DecodedRnOcePolicyState:
    """The complete front-20 state reconstructed from policy features."""

    normalized_support: torch.Tensor
    density0: torch.Tensor
    dlvsqs: torch.Tensor
    backbone_total_vols: torch.Tensor


@dataclass(frozen=True)
class RnOceFloatingBookSnapshot:
    """Policy state and aligned floating-book issue-time observables.

    The option tensors use expiry-major order ``(5D, 10D, 15D, 20D)`` and
    preserve the supplied increasing NMS order within every expiry.
    """

    policy_state: torch.Tensor
    policy_state_metadata: RnOcePolicyStateMetadata
    relative_strikes: torch.Tensor
    normalized_option_entry_prices: torch.Tensor
    is_call: torch.Tensor
    is_put: torch.Tensor
    normalized_moneyness: torch.Tensor
    expiry_days: tuple[int, ...]


def canonical_density0_from_normalized_support(
    normalized_support: torch.Tensor,
) -> torch.Tensor:
    """Return the unique two-node martingale law representing current spot.

    The normalized current spot is one.  If one is a support node, its law is
    a point mass there.  Otherwise, the two bracketing nodes receive the
    unique nonnegative weights with unit mass and unit forward mean.
    """

    if normalized_support.ndim != 2 or normalized_support.shape[1] < 2:
        raise ValueError("normalized_support must have shape (batch, support)")
    if not normalized_support.dtype.is_floating_point or not bool(
        torch.all(torch.isfinite(normalized_support))
        and torch.all(normalized_support > 0.0)
        and torch.all(torch.diff(normalized_support, dim=-1) > 0.0)
    ):
        raise ValueError(
            "normalized_support must be finite, positive and strictly increasing"
        )
    batch, support_size = normalized_support.shape
    exact = normalized_support == 1.0
    exact_count = torch.sum(exact, dim=-1)
    right_index = torch.sum(normalized_support < 1.0, dim=-1)
    bracketed = (right_index > 0) & (right_index < support_size)
    if not bool(torch.all((exact_count == 1) | bracketed)):
        raise ValueError("normalized support must contain or strictly bracket one")
    if not bool(torch.all(exact_count <= 1)):
        raise ValueError("strictly increasing support cannot contain one twice")

    density0 = torch.zeros_like(normalized_support)
    exact_rows = exact_count == 1
    if bool(torch.any(exact_rows)):
        exact_index = torch.argmax(exact.to(torch.int64), dim=-1)
        density0[exact_rows, exact_index[exact_rows]] = 1.0
    interpolated_rows = ~exact_rows
    if bool(torch.any(interpolated_rows)):
        rows = torch.nonzero(interpolated_rows).flatten()
        right = right_index[rows]
        left = right - 1
        left_support = normalized_support[rows, left]
        right_support = normalized_support[rows, right]
        width = right_support - left_support
        density0[rows, left] = (right_support - 1.0) / width
        density0[rows, right] = (1.0 - left_support) / width
    assert density0.shape == (batch, support_size)
    return density0


def _front_daily_indices(surface: ResidualClockDlvSurface) -> torch.Tensor:
    """Validate and return the actual residual 1D--20D expiry positions."""

    if surface.residual_expiries_y.ndim != 1:
        raise ValueError("residual_expiries_y must be a vector")
    if len(surface.residual_expiries_y) < RN_OCE_DAILY_SLABS:
        raise ValueError(
            "RN OCE features require at least twenty residual-expiry rows"
        )
    expected_days = torch.arange(
        1,
        RN_OCE_DAILY_SLABS + 1,
        dtype=surface.residual_expiries_y.dtype,
        device=surface.residual_expiries_y.device,
    )
    actual_days = (
        surface.residual_expiries_y[:RN_OCE_DAILY_SLABS] * DAYS_PER_YEAR
    )
    if not bool(
        torch.all(torch.isfinite(actual_days))
        and torch.allclose(actual_days, expected_days, rtol=0.0, atol=5.0e-10)
    ):
        raise ValueError(
            "RN OCE features require actual front residual expiries 1D, 2D, "
            "..., 20D"
        )
    return torch.arange(
        RN_OCE_DAILY_SLABS,
        dtype=torch.long,
        device=surface.residual_expiries_y.device,
    )


def _validate_surface_feature_domain(
    surface: ResidualClockDlvSurface,
) -> tuple[int, int, int]:
    """Validate only the structural domain needed by the feature map."""

    if surface.support_strikes.ndim != 3:
        raise ValueError(
            "support_strikes must have shape (batch, expiries, support)"
        )
    batch, expiries, support_size = surface.support_strikes.shape
    if batch <= 0 or support_size < 3:
        raise ValueError("the residual-clock surface has an invalid shape")
    num_inner = support_size - 2
    expected_shapes = {
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
    if not surface.support_strikes.dtype.is_floating_point:
        raise ValueError("surface tensors must have a floating-point dtype")
    if any(
        value.dtype != surface.support_strikes.dtype
        or value.device != surface.support_strikes.device
        for value in tensors
    ):
        raise ValueError("surface tensors must share dtype and device")
    _front_daily_indices(surface)
    support = surface.support_strikes[:, :RN_OCE_DAILY_SLABS]
    reference_support = support[:, :1].expand_as(support)
    if not bool(
        torch.all(torch.isfinite(support))
        and torch.all(support > 0.0)
        and torch.all(torch.diff(support, dim=-1) > 0.0)
        and torch.equal(support, reference_support)
    ):
        raise ValueError(
            "the first twenty rows must use one common finite positive "
            "normalized support"
        )
    density0 = surface.density0
    canonical_density0 = canonical_density0_from_normalized_support(
        surface.support_strikes[:, 0]
    )
    if not bool(
        torch.all(torch.isfinite(density0))
        and torch.all(density0 >= 0.0)
        and torch.allclose(
            torch.sum(density0, dim=-1),
            torch.ones(batch, dtype=density0.dtype, device=density0.device),
            rtol=0.0,
            atol=2.0e-10,
        )
        and torch.allclose(
            density0,
            canonical_density0,
            rtol=0.0,
            atol=2.0e-10,
        )
    ):
        raise ValueError(
            "density0 must be the canonical normalized-spot martingale law"
        )
    densities = surface.densities[:, :RN_OCE_DAILY_SLABS]
    if not bool(
        torch.all(torch.isfinite(densities))
        and torch.all(densities >= 0.0)
        and torch.allclose(
            torch.sum(densities, dim=-1),
            torch.ones(
                (batch, RN_OCE_DAILY_SLABS),
                dtype=densities.dtype,
                device=densities.device,
            ),
            rtol=0.0,
            atol=2.0e-10,
        )
        and torch.allclose(
            torch.sum(densities * support, dim=-1),
            torch.ones(
                (batch, RN_OCE_DAILY_SLABS),
                dtype=densities.dtype,
                device=densities.device,
            ),
            rtol=0.0,
            atol=2.0e-10,
        )
    ):
        raise ValueError(
            "front densities must be finite nonnegative unit-mass martingale laws"
        )
    dlvsqs = surface.dlvsqs[:, :RN_OCE_DAILY_SLABS]
    backbone = surface.backbone_total_vols[:, :RN_OCE_DAILY_SLABS]
    if not bool(
        torch.all(torch.isfinite(dlvsqs))
        and torch.all(dlvsqs > 0.0)
        and torch.all(torch.isfinite(backbone))
        and torch.all(backbone >= 0.0)
        and torch.all(torch.diff(backbone.square(), dim=-1) >= -2.0e-10)
    ):
        raise ValueError(
            "front DLV rows and backbone total volatilities are outside their "
            "finite numerical domain"
        )
    return batch, support_size, num_inner


def encode_rn_oce_policy_state(
    surface: ResidualClockDlvSurface,
    *,
    minimum_dlvsq: float,
    minimum_excess_dlvsq: float = 0.0,
) -> RnOcePolicyState:
    r"""Encode a densified risk-neutral surface into one fixed-width state.

    The feature order is

    ``[log(relative support), log(DLVsq-minimum), backbone total volatility]``.

    DLV rows are flattened in residual-expiry-major, then strike-major order.
    ``density0`` is not stored: it is the unique two-node unit-mass,
    unit-forward law determined by the common normalized support and is
    reconstructed exactly.  The feature map is therefore invertible on its
    accepted domain.  Dataset-level centering and scaling are deliberately
    left to the training fold.

    Parameters
    ----------
    surface:
        A surface already densified at the actual residual expiries 1D--20D.
    minimum_dlvsq:
        Fitted open lower bound used to define excess DLV squared.
    minimum_excess_dlvsq:
        Nonnegative admissibility threshold.  Values at or below it are
        rejected rather than clipped, preserving invertibility.  The default
        accepts every strictly positive floating-point excess.

    Returns
    -------
    RnOcePolicyState
        ``values`` has shape ``(batch, S + 20*(S-2) + 20)``, where ``S``
        is the complete support size.
    """

    if not isfinite(minimum_dlvsq) or minimum_dlvsq < 0.0:
        raise ValueError("minimum_dlvsq must be finite and nonnegative")
    if not isfinite(minimum_excess_dlvsq) or minimum_excess_dlvsq < 0.0:
        raise ValueError("minimum_excess_dlvsq must be finite and nonnegative")
    batch, support_size, num_inner = _validate_surface_feature_domain(surface)
    support = surface.support_strikes[:, 0]
    log_support = torch.log(support)
    dlvsqs = surface.dlvsqs[:, :RN_OCE_DAILY_SLABS]
    minimum = torch.as_tensor(
        minimum_dlvsq,
        dtype=dlvsqs.dtype,
        device=dlvsqs.device,
    )
    minimum_excess = torch.as_tensor(
        minimum_excess_dlvsq,
        dtype=dlvsqs.dtype,
        device=dlvsqs.device,
    )
    excess = dlvsqs - minimum
    if not bool(torch.all(torch.isfinite(excess)) and torch.all(excess > minimum_excess)):
        raise ValueError(
            "front DLV squared rows must exceed minimum_dlvsq by more than "
            "minimum_excess_dlvsq"
        )
    log_excess = torch.log(excess).flatten(start_dim=1)
    backbone = surface.backbone_total_vols[:, :RN_OCE_DAILY_SLABS]

    support_stop = support_size
    dlvsq_stop = support_stop + RN_OCE_DAILY_SLABS * num_inner
    backbone_stop = dlvsq_stop + RN_OCE_DAILY_SLABS
    values = torch.cat((log_support, log_excess, backbone), dim=1)
    if tuple(values.shape) != (batch, backbone_stop) or not bool(
        torch.all(torch.isfinite(values))
    ):
        raise FloatingPointError("RN OCE policy-state encoding is not finite")

    metadata = RnOcePolicyStateMetadata(
        schema=RN_OCE_POLICY_STATE_SCHEMA,
        shape=(batch, backbone_stop),
        daily_slab_days=tuple(range(1, RN_OCE_DAILY_SLABS + 1)),
        support_size=support_size,
        num_inner_strikes=num_inner,
        feature_size=backbone_stop,
        minimum_dlvsq=float(minimum_dlvsq),
        minimum_excess_dlvsq=float(minimum_excess_dlvsq),
        log_normalized_support_slice=(0, support_stop),
        log_excess_dlvsq_slice=(support_stop, dlvsq_stop),
        backbone_total_vol_slice=(dlvsq_stop, backbone_stop),
        density0_reconstruction=(
            "canonical two-node unit-mass/unit-forward law from normalized support"
        ),
        dlvsq_flattening="residual-expiry-major, then strike-major",
        dtype=str(values.dtype),
        device=str(values.device),
    )
    return RnOcePolicyState(values=values, metadata=metadata)


def decode_rn_oce_policy_state(
    state: RnOcePolicyState,
) -> DecodedRnOcePolicyState:
    """Invert an encoded front-20 policy state into its numerical components."""

    values = state.values
    metadata = state.metadata
    if metadata.schema != RN_OCE_POLICY_STATE_SCHEMA:
        raise ValueError(f"unsupported RN OCE state schema {metadata.schema!r}")
    if tuple(values.shape) != metadata.shape or values.ndim != 2:
        raise ValueError(
            f"encoded values must have shape {metadata.shape}, got {tuple(values.shape)}"
        )
    if values.shape[1] != metadata.feature_size or not bool(
        torch.all(torch.isfinite(values))
    ):
        raise ValueError("encoded RN OCE values are inconsistent or nonfinite")
    support_start, support_stop = metadata.log_normalized_support_slice
    dlvsq_start, dlvsq_stop = metadata.log_excess_dlvsq_slice
    backbone_start, backbone_stop = metadata.backbone_total_vol_slice
    expected_slices = (
        (support_start, support_stop),
        (dlvsq_start, dlvsq_stop),
        (backbone_start, backbone_stop),
    )
    if expected_slices != (
        (0, metadata.support_size),
        (
            metadata.support_size,
            metadata.support_size
            + RN_OCE_DAILY_SLABS * metadata.num_inner_strikes,
        ),
        (
            metadata.support_size
            + RN_OCE_DAILY_SLABS * metadata.num_inner_strikes,
            metadata.feature_size,
        ),
    ):
        raise ValueError("RN OCE metadata slices are inconsistent")

    support = torch.exp(values[:, support_start:support_stop])
    density0 = canonical_density0_from_normalized_support(support)
    minimum = torch.as_tensor(
        metadata.minimum_dlvsq,
        dtype=values.dtype,
        device=values.device,
    )
    dlvsqs = (
        minimum + torch.exp(values[:, dlvsq_start:dlvsq_stop])
    ).reshape(
        len(values),
        RN_OCE_DAILY_SLABS,
        metadata.num_inner_strikes,
    )
    backbone = values[:, backbone_start:backbone_stop]
    if not bool(
        torch.all(torch.isfinite(support))
        and torch.all(torch.isfinite(dlvsqs))
        and torch.all(torch.isfinite(backbone))
    ):
        raise FloatingPointError("decoded RN OCE policy state is not finite")
    return DecodedRnOcePolicyState(
        normalized_support=support,
        density0=density0,
        dlvsqs=dlvsqs,
        backbone_total_vols=backbone,
    )


def rn_atm_implied_total_vols(
    surface: ResidualClockDlvSurface,
) -> torch.Tensor:
    """Return ATM Black implied total vols for the actual front 20 daily rows.

    The ATM call is priced from the complete risk-neutral DLV mixture.  For a
    normalized zero-carry ATM call, ``C = 2*Phi(w/2)-1``, so the inversion is
    analytic and introduces no root-finder tolerance.
    """

    batch, _, _ = _validate_surface_feature_domain(surface)
    one = torch.ones(
        batch,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    atm_calls = price_residual_clock_options(
        surface,
        current_spot=one,
        fixed_cash_strikes=one[:, None],
        is_call=torch.ones(
            (batch, 1), dtype=torch.bool, device=surface.densities.device
        ),
    )[:, :RN_OCE_DAILY_SLABS, 0]
    tolerance = 256.0 * torch.finfo(atm_calls.dtype).eps
    if not bool(
        torch.all(torch.isfinite(atm_calls))
        and torch.all(atm_calls >= -tolerance)
        and torch.all(atm_calls <= 1.0 + tolerance)
    ):
        raise FloatingPointError("RN ATM call prices lie outside [0, 1]")
    probability = 0.5 * (torch.clamp(atm_calls, min=0.0, max=1.0) + 1.0)
    probability = torch.clamp(
        probability,
        max=torch.nextafter(torch.ones_like(probability), torch.zeros_like(probability)),
    )
    total_vols = 2.0 * torch.special.ndtri(probability)
    if not bool(
        torch.all(torch.isfinite(total_vols)) and torch.all(total_vols >= 0.0)
    ):
        raise FloatingPointError("RN ATM implied total volatilities are invalid")
    return total_vols


def rn_nms_relative_strikes(
    surface: ResidualClockDlvSurface,
    *,
    normalized_moneyness: torch.Tensor,
) -> torch.Tensor:
    r"""Return 5D/10D/15D/20D relative strikes under the RN surface.

    This applies the historical DYSANOS NMS convention

    ``K/F = exp(z * sqrt((sigma_eps**2*eps + w_T**2)/(1+eps)))``

    with ``w_T`` inferred from the current risk-neutral ATM call and
    ``sigma_eps = w_1/sqrt(1/252)``.  The result has shape
    ``(batch, 4, num_normalized_moneyness)`` in the expiry order
    ``(5, 10, 15, 20)``.
    """

    if normalized_moneyness.ndim != 1 or len(normalized_moneyness) == 0:
        raise ValueError("normalized_moneyness must be a nonempty vector")
    if (
        normalized_moneyness.dtype != surface.densities.dtype
        or normalized_moneyness.device != surface.densities.device
    ):
        raise ValueError(
            "normalized_moneyness must share the surface dtype and device"
        )
    if not bool(torch.all(torch.isfinite(normalized_moneyness))):
        raise ValueError("normalized_moneyness must be finite")
    total_vols = rn_atm_implied_total_vols(surface)
    expiry_index = torch.as_tensor(
        tuple(day - 1 for day in RN_OCE_EXPIRY_DAYS),
        dtype=torch.long,
        device=total_vols.device,
    )
    expiry_total_vols = total_vols[:, expiry_index]
    front_time_y = surface.residual_expiries_y[0]
    epsilon_vol = total_vols[:, :1] / torch.sqrt(front_time_y)
    scale = torch.sqrt(
        (
            epsilon_vol.square() * float(NMS.EPS)
            + expiry_total_vols.square()
        )
        / (1.0 + float(NMS.EPS))
    )
    strikes = torch.exp(scale[..., None] * normalized_moneyness[None, None])
    expected_shape = (
        len(surface.densities),
        len(RN_OCE_EXPIRY_DAYS),
        len(normalized_moneyness),
    )
    if tuple(strikes.shape) != expected_shape or not bool(
        torch.all(torch.isfinite(strikes) & (strikes > 0.0))
    ):
        raise FloatingPointError("RN NMS conversion produced invalid strikes")
    return strikes


def make_rn_oce_floating_book_snapshot(
    surface: ResidualClockDlvSurface,
    *,
    minimum_dlvsq: float,
    normalized_moneyness: torch.Tensor,
    minimum_excess_dlvsq: float = 0.0,
) -> RnOceFloatingBookSnapshot:
    """Build the complete issue-time input for the 49-instrument OCE book.

    The returned prices are normalized by current spot, which is one in the
    residual-clock coordinates.  Calls are used at nonpositive NMS, including
    the ATM coordinate, and puts at positive NMS.  This reproduces the old
    floating-book convention while deriving strikes and prices entirely from
    the current risk-neutral surface.

    Parameters
    ----------
    surface:
        A surface densified at every residual expiry from 1D through 20D.
    minimum_dlvsq:
        Fitted lower DLV-squared bound used by the policy-state encoding.
    normalized_moneyness:
        The strictly increasing twelve-coordinate floating-book NMS vector,
        including exactly one zero.
    minimum_excess_dlvsq:
        Optional positive-excess threshold passed to the invertible state
        encoder.

    Returns
    -------
    RnOceFloatingBookSnapshot
        Policy states have shape ``(B, 4222)`` for the production 202-point
        support.  Strike and price tensors have shape ``(B, 4, 12)``; option
        flags have shape ``(4, 12)``.
    """

    if normalized_moneyness.ndim != 1 or len(normalized_moneyness) != 12:
        raise ValueError("the RN OCE floating book requires exactly 12 NMS values")
    if not bool(
        torch.all(torch.isfinite(normalized_moneyness))
        and torch.all(torch.diff(normalized_moneyness) > 0.0)
        and torch.sum(normalized_moneyness == 0.0) == 1
    ):
        raise ValueError(
            "the 12 NMS values must be finite, strictly increasing and contain "
            "exactly one zero"
        )

    encoded = encode_rn_oce_policy_state(
        surface,
        minimum_dlvsq=minimum_dlvsq,
        minimum_excess_dlvsq=minimum_excess_dlvsq,
    )
    if encoded.values.shape[1] != RN_OCE_PRODUCTION_POLICY_STATE_SIZE:
        raise ValueError(
            "the RN OCE floating book requires the production 202-point "
            f"support and {RN_OCE_PRODUCTION_POLICY_STATE_SIZE} policy features"
        )
    relative_strikes = rn_nms_relative_strikes(
        surface,
        normalized_moneyness=normalized_moneyness,
    )
    batch = len(surface.densities)
    is_call_by_strike = normalized_moneyness <= 0.0
    is_call = is_call_by_strike[None].expand(
        len(RN_OCE_EXPIRY_DAYS), -1
    )
    is_put = ~is_call
    if not bool(
        torch.all(torch.diff(relative_strikes, dim=-1) > 0.0)
        and torch.all(relative_strikes[..., is_call_by_strike & (normalized_moneyness < 0.0)] < 1.0)
        and torch.all(relative_strikes[..., normalized_moneyness == 0.0] == 1.0)
        and torch.all(relative_strikes[..., normalized_moneyness > 0.0] > 1.0)
    ):
        raise FloatingPointError("RN OCE relative strikes do not preserve NMS order")

    flattened_strikes = relative_strikes.flatten(start_dim=1)
    flattened_calls = is_call.flatten()[None].expand(batch, -1)
    values_at_all_expiries = price_residual_clock_options(
        surface,
        current_spot=torch.ones(
            batch,
            dtype=surface.densities.dtype,
            device=surface.densities.device,
        ),
        fixed_cash_strikes=flattened_strikes,
        is_call=flattened_calls,
    )
    expiry_positions = torch.as_tensor(
        tuple(day - 1 for day in RN_OCE_EXPIRY_DAYS),
        dtype=torch.long,
        device=surface.densities.device,
    )
    option_positions = torch.arange(
        len(RN_OCE_EXPIRY_DAYS) * len(normalized_moneyness),
        dtype=torch.long,
        device=surface.densities.device,
    ).reshape(len(RN_OCE_EXPIRY_DAYS), len(normalized_moneyness))
    selected_expiries = values_at_all_expiries[:, expiry_positions]
    entry_prices = torch.gather(
        selected_expiries,
        dim=2,
        index=option_positions[None].expand(batch, -1, -1),
    )
    expected_option_shape = (
        batch,
        len(RN_OCE_EXPIRY_DAYS),
        len(normalized_moneyness),
    )
    if tuple(entry_prices.shape) != expected_option_shape or not bool(
        torch.all(torch.isfinite(entry_prices) & (entry_prices > 0.0))
    ):
        raise FloatingPointError(
            "RN OCE normalized option entry prices must be finite and positive"
        )
    if tuple(relative_strikes.shape) != expected_option_shape:
        raise RuntimeError("RN OCE strike and option-price ordering is inconsistent")
    return RnOceFloatingBookSnapshot(
        policy_state=encoded.values,
        policy_state_metadata=encoded.metadata,
        relative_strikes=relative_strikes,
        normalized_option_entry_prices=entry_prices,
        is_call=is_call,
        is_put=is_put,
        normalized_moneyness=normalized_moneyness.detach().clone(),
        expiry_days=RN_OCE_EXPIRY_DAYS,
    )


__all__ = [
    "RN_OCE_DAILY_SLABS",
    "RN_OCE_EXPIRY_DAYS",
    "RN_OCE_POLICY_STATE_SCHEMA",
    "RN_OCE_PRODUCTION_POLICY_STATE_SIZE",
    "DecodedRnOcePolicyState",
    "RnOceFloatingBookSnapshot",
    "RnOcePolicyState",
    "RnOcePolicyStateMetadata",
    "canonical_density0_from_normalized_support",
    "decode_rn_oce_policy_state",
    "encode_rn_oce_policy_state",
    "make_rn_oce_floating_book_snapshot",
    "rn_atm_implied_total_vols",
    "rn_nms_relative_strikes",
]
