from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.residual_clock_dlv import (
    ResidualClockDlvSurface,
    price_residual_clock_options,
)
from quantdata.dysanos.flex.rn_oce_features import (
    RN_OCE_POLICY_STATE_SCHEMA,
    canonical_density0_from_normalized_support,
    decode_rn_oce_policy_state,
    encode_rn_oce_policy_state,
    make_rn_oce_floating_book_snapshot,
    rn_atm_implied_total_vols,
    rn_nms_relative_strikes,
)


def _surface(*, batch: int = 2) -> ResidualClockDlvSurface:
    dtype = torch.float64
    days = torch.arange(1, 21, dtype=dtype)
    log_support = torch.cat(
        (
            torch.linspace(-0.30, 0.0, 101, dtype=dtype),
            torch.linspace(0.003, 0.30, 101, dtype=dtype),
        )
    )
    support = torch.stack(
        tuple(torch.exp(log_support * (1.0 + 0.05 * row)) for row in range(batch))
    )
    density0 = canonical_density0_from_normalized_support(support)
    support_by_expiry = support[:, None].expand(-1, 20, -1).clone()
    densities = density0[:, None].expand(-1, 20, -1).clone()
    strike_position = torch.arange(200, dtype=dtype)
    expiry_position = torch.arange(20, dtype=dtype)
    dlvsqs = (
        0.011
        + 1.0e-5 * strike_position[None, None]
        + 2.0e-5 * expiry_position[None, :, None]
        + 1.0e-4 * torch.arange(batch, dtype=dtype)[:, None, None]
    )
    annual_vol = torch.linspace(0.20, 0.30, batch, dtype=dtype)
    backbone = annual_vol[:, None] * torch.sqrt(days[None] / DAYS_PER_YEAR)
    return ResidualClockDlvSurface(
        residual_expiries_y=days / DAYS_PER_YEAR,
        support_strikes=support_by_expiry,
        density0=density0,
        densities=densities,
        dlvsqs=dlvsqs,
        backbone_total_vols=backbone,
    )


def test_policy_state_has_fixed_complete_layout_and_round_trips() -> None:
    surface = _surface()

    encoded = encode_rn_oce_policy_state(surface, minimum_dlvsq=0.001)
    repeated = encode_rn_oce_policy_state(surface, minimum_dlvsq=0.001)
    decoded = decode_rn_oce_policy_state(encoded)

    assert encoded.values.shape == (2, 4222)
    assert torch.equal(encoded.values, repeated.values)
    assert encoded.metadata.schema == RN_OCE_POLICY_STATE_SCHEMA
    assert encoded.metadata.feature_size == 4222
    assert encoded.metadata.log_normalized_support_slice == (0, 202)
    assert encoded.metadata.log_excess_dlvsq_slice == (202, 4202)
    assert encoded.metadata.backbone_total_vol_slice == (4202, 4222)
    assert encoded.metadata.daily_slab_days == tuple(range(1, 21))
    assert "unit-forward" in encoded.metadata.density0_reconstruction
    assert encoded.metadata.as_dict()["shape"] == [2, 4222]
    assert torch.allclose(decoded.normalized_support, surface.support_strikes[:, 0])
    assert torch.allclose(decoded.density0, surface.density0)
    assert torch.allclose(decoded.dlvsqs, surface.dlvsqs[:, :20])
    assert torch.allclose(
        decoded.backbone_total_vols,
        surface.backbone_total_vols[:, :20],
    )


def test_canonical_density_reconstructs_bracketed_normalized_spot() -> None:
    support = torch.tensor(
        ((0.8, 0.9, 1.2), (0.7, 1.0, 1.3)), dtype=torch.float64
    )

    density0 = canonical_density0_from_normalized_support(support)

    assert torch.allclose(torch.sum(density0, dim=-1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(torch.sum(density0 * support, dim=-1), torch.ones(2, dtype=torch.float64))
    assert torch.equal(density0[1], torch.tensor((0.0, 1.0, 0.0), dtype=torch.float64))


def test_policy_state_rejects_non_daily_or_noninvertible_input() -> None:
    surface = _surface(batch=1)
    stale_clock = surface.residual_expiries_y.clone()
    stale_clock[10] += 0.25 / DAYS_PER_YEAR
    with pytest.raises(ValueError, match="actual front residual expiries"):
        encode_rn_oce_policy_state(
            replace(surface, residual_expiries_y=stale_clock),
            minimum_dlvsq=0.001,
        )

    noncanonical_density0 = surface.density0.roll(1, dims=-1)
    with pytest.raises(ValueError, match="canonical normalized-spot"):
        encode_rn_oce_policy_state(
            replace(surface, density0=noncanonical_density0),
            minimum_dlvsq=0.001,
        )

    boundary_dlvsqs = surface.dlvsqs.clone()
    boundary_dlvsqs[0, 0, 0] = 0.001 + 1.0e-13
    with pytest.raises(ValueError, match="minimum_excess_dlvsq"):
        encode_rn_oce_policy_state(
            replace(surface, dlvsqs=boundary_dlvsqs),
            minimum_dlvsq=0.001,
            minimum_excess_dlvsq=1.0e-12,
        )


def test_rn_atm_total_vol_and_nms_strikes_match_old_convention() -> None:
    surface = _surface()
    scores = torch.tensor((-1.5, 0.0, 0.5), dtype=torch.float64)

    total_vols = rn_atm_implied_total_vols(surface)
    strikes = rn_nms_relative_strikes(
        surface,
        normalized_moneyness=scores,
    )

    assert torch.allclose(
        total_vols,
        surface.backbone_total_vols[:, :20],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    expiry_index = torch.tensor((4, 9, 14, 19))
    annual_front_vol = total_vols[:, :1] / torch.sqrt(
        surface.residual_expiries_y[0]
    )
    expected_scale = torch.sqrt(
        (
            annual_front_vol.square() * float(NMS.EPS)
            + total_vols[:, expiry_index].square()
        )
        / (1.0 + float(NMS.EPS))
    )
    expected = torch.exp(expected_scale[..., None] * scores[None, None])
    assert strikes.shape == (2, 4, 3)
    assert torch.allclose(strikes, expected, rtol=2.0e-13, atol=2.0e-13)
    assert torch.equal(strikes[..., 1], torch.ones((2, 4), dtype=torch.float64))


def test_rn_nms_uses_the_transitioned_density_not_only_backbone() -> None:
    surface = _surface(batch=1)
    density = torch.zeros_like(surface.densities)
    density[..., 80] = 0.5
    density[..., 120] = 0.5
    left = surface.support_strikes[:, :, 80]
    right = surface.support_strikes[:, :, 120]
    right_weight = (1.0 - left) / (right - left)
    density.zero_()
    density[..., 80] = 1.0 - right_weight
    density[..., 120] = right_weight
    zero_backbone = torch.zeros_like(surface.backbone_total_vols)
    dispersed = replace(
        surface,
        densities=density,
        backbone_total_vols=zero_backbone,
    )
    scores = torch.tensor((-1.0, 1.0), dtype=torch.float64)

    point_mass_strikes = rn_nms_relative_strikes(
        replace(surface, backbone_total_vols=zero_backbone),
        normalized_moneyness=scores,
    )
    dispersed_strikes = rn_nms_relative_strikes(
        dispersed,
        normalized_moneyness=scores,
    )

    assert torch.equal(point_mass_strikes, torch.ones_like(point_mass_strikes))
    assert torch.all(dispersed_strikes[..., 0] < 1.0)
    assert torch.all(dispersed_strikes[..., 1] > 1.0)


def test_nms_rejects_mismatched_dtype() -> None:
    surface = _surface(batch=1)

    with pytest.raises(ValueError, match="share the surface dtype"):
        rn_nms_relative_strikes(
            surface,
            normalized_moneyness=torch.tensor((-1.0, 0.0), dtype=torch.float32),
        )


def test_floating_book_snapshot_is_expiry_major_and_exactly_priced() -> None:
    surface = _surface()
    scores = torch.tensor(
        (-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5),
        dtype=torch.float64,
    )

    snapshot = make_rn_oce_floating_book_snapshot(
        surface,
        minimum_dlvsq=0.001,
        normalized_moneyness=scores,
    )

    assert snapshot.policy_state.shape == (2, 4222)
    assert snapshot.relative_strikes.shape == (2, 4, 12)
    assert snapshot.normalized_option_entry_prices.shape == (2, 4, 12)
    assert snapshot.is_call.shape == (4, 12)
    assert snapshot.is_put.shape == (4, 12)
    assert snapshot.expiry_days == (5, 10, 15, 20)
    assert torch.equal(snapshot.normalized_moneyness, scores)
    assert torch.all(snapshot.is_call[:, :9])
    assert torch.all(snapshot.is_put[:, 9:])
    assert not bool(torch.any(snapshot.is_put[:, :9]))
    assert not bool(torch.any(snapshot.is_call[:, 9:]))
    assert torch.all(snapshot.normalized_option_entry_prices > 0.0)

    for expiry_position, expiry_day in enumerate(snapshot.expiry_days):
        directly_priced = price_residual_clock_options(
            surface,
            current_spot=torch.ones(2, dtype=torch.float64),
            fixed_cash_strikes=snapshot.relative_strikes[:, expiry_position],
            is_call=snapshot.is_call[expiry_position],
        )[:, expiry_day - 1]
        assert torch.allclose(
            snapshot.normalized_option_entry_prices[:, expiry_position],
            directly_priced,
        )


def test_floating_book_snapshot_validates_grid_and_positive_prices() -> None:
    surface = _surface(batch=1)
    scores = torch.tensor(
        (-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        make_rn_oce_floating_book_snapshot(
            surface,
            minimum_dlvsq=0.001,
            normalized_moneyness=scores.roll(1),
        )
    with pytest.raises(ValueError, match="exactly 12"):
        make_rn_oce_floating_book_snapshot(
            surface,
            minimum_dlvsq=0.001,
            normalized_moneyness=scores[:-1],
        )

    zero_vol_surface = replace(
        surface,
        backbone_total_vols=torch.zeros_like(surface.backbone_total_vols),
    )
    with pytest.raises(FloatingPointError):
        make_rn_oce_floating_book_snapshot(
            zero_vol_surface,
            minimum_dlvsq=0.001,
            normalized_moneyness=scores,
        )
