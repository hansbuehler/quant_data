"""Tests for exact functional DLV transitions on residual-expiry clocks."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quantdata.dysanos.flex import residual_clock_dlv as residual_clock_dlv_module
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.lvsq_drift import (
    density_martingale_lvsq_step,
    fixed_contract_lvsq_drift,
    smooth_lvsq_log_loadings,
)
from quantdata.dysanos.flex.model import (
    NativeParameters,
    raw_from_native_parameters,
    split_forward_mu_native_raw,
)
from quantdata.dysanos.flex.residual_clock_dlv import (
    FunctionalDlvInadmissibleError,
    ResidualClockDlvSurface,
    apply_functional_dlv_innovation,
    build_functional_dlv_cubature,
    canonicalize_residual_clock_dlv_ingress,
    densify_residual_clock_dlv_surface,
    exact_functional_dlv_option_tower_residual,
    expand_residual_clock_dlv_surface,
    functional_dlv_characteristics,
    insert_residual_front,
    insert_residual_front_by_density_interpolation,
    price_residual_clock_options,
    residual_clock_surface_from_decoded,
    roll_residual_clock_dlv,
    smooth_residual_clock_log_loadings,
    validate_residual_clock_dlv_surface,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.torch import BaseEnvironment


def _problem(
    business_dtes: tuple[float, ...],
) -> tuple[FlexSanosDecoder, torch.Tensor, torch.Tensor, ResidualClockDlvSurface]:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=len(business_dtes),
        explicit_model_business_dtes=business_dtes,
        inner_relative_strikes=(0.72, 0.84, 0.94, 1.0, 1.08, 1.25),
    )
    decoder = FlexSanosDecoder(env, geometry=geometry)
    base_dlvsq = torch.tensor(
        [0.11, 0.09, 0.07, 0.08, 0.10, 0.13],
        dtype=env.fdtype,
    )
    dlvsqs = torch.stack(
        [base_dlvsq + 0.01 * index for index in range(len(business_dtes))]
    )[None]
    parameters = NativeParameters(
        dlvsqs=dlvsqs,
        forward_vols=torch.tensor(
            [[0.20 + 0.01 * index for index in range(len(business_dtes))]],
            dtype=env.fdtype,
        ),
        eta=torch.tensor(
            [[0.30 + 0.01 * index for index in range(len(business_dtes))]],
            dtype=env.fdtype,
        ),
    )
    raw = raw_from_native_parameters(parameters, geometry=geometry)
    decoded = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    exact_clock = torch.tensor(business_dtes, dtype=env.fdtype) / 252.0
    surface = residual_clock_surface_from_decoded(
        decoder,
        decoded,
        residual_expiries_y=exact_clock,
    )
    loadings = smooth_lvsq_log_loadings(
        decoded.support_strikes[0, 0],
        num_model_expiries=len(business_dtes),
        annualized_log_volatility=0.01,
    )
    return decoder, raw, loadings, surface


def _native_clock_problem() -> tuple[
    FlexSanosDecoder,
    torch.Tensor,
    torch.Tensor,
    ResidualClockDlvSurface,
]:
    return _problem((5.0, 10.0, 15.0, 20.0))


def _daily_problem() -> tuple[
    FlexSanosDecoder,
    torch.Tensor,
    torch.Tensor,
    ResidualClockDlvSurface,
]:
    return _problem((1.0, 2.0, 3.0, 4.0, 5.0, 10.0))


def _black_calls(
    forwards: torch.Tensor,
    strikes: torch.Tensor,
    total_vols: torch.Tensor,
) -> torch.Tensor:
    safe = torch.clamp_min(total_vols, torch.finfo(forwards.dtype).tiny)
    d1 = (torch.log(forwards / strikes) + 0.5 * safe.square()) / safe
    value = forwards * torch.special.ndtr(d1) - strikes * torch.special.ndtr(
        d1 - safe
    )
    return torch.where(total_vols > 0.0, value, torch.relu(forwards - strikes))


def test_characteristics_and_cubature_match_native_fixed_clock() -> None:
    decoder, raw, loadings, surface = _native_clock_problem()
    local = fixed_contract_lvsq_drift(
        decoder,
        raw,
        raw_log_loadings=loadings,
    )
    characteristics = functional_dlv_characteristics(
        decoder,
        surface,
        raw_log_loadings=loadings,
    )
    local_mu_z, _, _ = split_forward_mu_native_raw(
        local.raw_drift,
        geometry=decoder.geometry,
    )

    torch.testing.assert_close(
        characteristics.expiry_time_increments_y,
        decoder.model_dts_y,
    )
    torch.testing.assert_close(characteristics.psi, local.density_diffusion)
    torch.testing.assert_close(characteristics.beta, local.dlvsq_diffusion)
    torch.testing.assert_close(characteristics.alpha, local.dlvsq_drift)
    torch.testing.assert_close(characteristics.mu_z, local_mu_z)

    old_step = density_martingale_lvsq_step(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
    )
    new_step = build_functional_dlv_cubature(
        decoder,
        surface,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
        standardized_innovations=old_step.standardized_innovations,
        branch_probabilities=old_step.branch_probabilities,
    )
    torch.testing.assert_close(
        new_step.densities_by_branch,
        old_step.densities_by_branch,
        atol=3.0e-14,
        rtol=2.0e-12,
    )
    torch.testing.assert_close(
        new_step.dlvsqs_by_branch,
        old_step.dlvsqs_by_branch,
        atol=3.0e-13,
        rtol=2.0e-11,
    )


def test_effective_log_drift_and_diffusion_are_the_cubature_moments() -> None:
    decoder, _, loadings, surface = _native_clock_problem()
    step_y = torch.tensor(2.0 / 252.0, dtype=torch.float64)
    kernel = build_functional_dlv_cubature(
        decoder,
        surface,
        raw_log_loadings=loadings,
        calendar_time_increment_y=step_y,
    )
    initial_z = torch.log(surface.dlvsqs - decoder.geometry.min_dlvsq)
    branch_z = torch.log(kernel.dlvsqs_by_branch - decoder.geometry.min_dlvsq)
    increment = branch_z - initial_z[:, None]
    expected_mu = torch.sum(
        kernel.branch_probabilities[None, :, None, None] * increment,
        dim=1,
    ) / step_y
    centered = increment - step_y * expected_mu[:, None]
    expected_diffusion = torch.einsum(
        "r,brtn,rf->btnf",
        kernel.branch_probabilities,
        centered,
        kernel.standardized_innovations,
    ) / torch.sqrt(step_y)
    expected_variance = torch.sum(
        kernel.branch_probabilities[None, :, None, None] * centered.square(),
        dim=1,
    ) / step_y

    torch.testing.assert_close(kernel.effective_mu_z, expected_mu)
    torch.testing.assert_close(
        kernel.effective_log_excess_dlvsq_diffusion,
        expected_diffusion,
    )
    torch.testing.assert_close(
        kernel.centered_log_increment_variance,
        expected_variance,
    )


def test_effective_characteristics_converge_under_step_refinement() -> None:
    decoder, _, loadings, surface = _native_clock_problem()
    # Keep the truncation term well above the algebraic inversion's floating
    # point floor, so the first-order refinement rate is observable.
    loadings = 10.0 * loadings
    characteristics = functional_dlv_characteristics(
        decoder,
        surface,
        raw_log_loadings=loadings,
    )
    target_variance = torch.sum(
        characteristics.raw_log_loadings.square(),
        dim=-1,
    )
    errors: list[tuple[float, float, float, float]] = []
    for step_y in (4.0e-3, 2.0e-3, 1.0e-3):
        kernel = build_functional_dlv_cubature(
            decoder,
            surface,
            raw_log_loadings=loadings,
            calendar_time_increment_y=step_y,
        )
        effective_loading = kernel.effective_log_excess_dlvsq_diffusion.reshape(
            len(surface.support_strikes),
            -1,
            kernel.effective_log_excess_dlvsq_diffusion.shape[-1],
        )
        local_loading = characteristics.raw_log_loadings.reshape(
            len(surface.support_strikes),
            -1,
            characteristics.raw_log_loadings.shape[-1],
        )
        errors.append(
            (
                float(torch.max(torch.abs(kernel.effective_mu_z - characteristics.mu_z))),
                float(
                    torch.max(
                        torch.abs(
                            kernel.effective_log_excess_dlvsq_diffusion
                            - characteristics.raw_log_loadings
                        )
                    )
                ),
                float(
                    torch.max(
                        torch.abs(
                            kernel.centered_log_increment_variance - target_variance
                        )
                    )
                ),
                float(
                    torch.max(
                        torch.abs(
                            effective_loading @ effective_loading.mT
                            - local_loading @ local_loading.mT
                        )
                    )
                ),
            )
        )
    for coordinate in range(4):
        assert errors[2][coordinate] < 0.7 * errors[1][coordinate]
        assert errors[1][coordinate] < 0.7 * errors[0][coordinate]


def test_apply_matches_cubature_branch_and_is_batch_invariant() -> None:
    decoder, _, loadings, one = _native_clock_problem()
    kernel = build_functional_dlv_cubature(
        decoder,
        one,
        raw_log_loadings=loadings,
        calendar_time_increment_y=1.0 / 252.0,
    )
    one_characteristics = functional_dlv_characteristics(
        decoder,
        one,
        raw_log_loadings=loadings,
    )
    one_transition = apply_functional_dlv_innovation(
        decoder,
        one_characteristics,
        standardized_innovation=kernel.standardized_innovations[:1],
        calendar_time_increment_y=1.0 / 252.0,
    )
    torch.testing.assert_close(
        one_transition.surface.densities,
        kernel.densities_by_branch[:, 0],
    )
    torch.testing.assert_close(
        one_transition.surface.dlvsqs,
        kernel.dlvsqs_by_branch[:, 0],
    )

    batch = expand_residual_clock_dlv_surface(one, 2)
    batch_characteristics = functional_dlv_characteristics(
        decoder,
        batch,
        raw_log_loadings=loadings,
    )
    batch_transition = apply_functional_dlv_innovation(
        decoder,
        batch_characteristics,
        standardized_innovation=kernel.standardized_innovations[:1].expand(2, -1),
        calendar_time_increment_y=1.0 / 252.0,
    )
    torch.testing.assert_close(
        batch_transition.surface.densities,
        one_transition.surface.densities.expand(2, -1, -1),
    )
    torch.testing.assert_close(
        batch_transition.surface.dlvsqs,
        one_transition.surface.dlvsqs.expand(2, -1, -1),
    )


def test_nonuniform_roll_uses_the_updated_residual_clock() -> None:
    decoder, _, _, surface = _daily_problem()
    zero_loadings = smooth_residual_clock_log_loadings(
        surface,
        annualized_log_volatility=0.0,
    )
    kernel = build_functional_dlv_cubature(
        decoder,
        surface,
        raw_log_loadings=zero_loadings,
        calendar_time_increment_y=1.0 / 252.0,
    )
    rolled = roll_residual_clock_dlv(
        decoder,
        kernel,
        surface_branch_uniforms=torch.tensor([0.37], dtype=torch.float64),
        front_node_uniforms=torch.tensor([0.41], dtype=torch.float64),
        front_backbone_normals=torch.tensor([0.2], dtype=torch.float64),
        elapsed_y=1.0 / 252.0,
    )
    expected_clock = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 9.0],
        dtype=torch.float64,
    ) / 252.0
    torch.testing.assert_close(rolled.surface.residual_expiries_y, expected_clock)
    diagnostics = validate_residual_clock_dlv_surface(decoder, rolled.surface)
    assert float(diagnostics["maximum_recursion_residual"]) < 2.0e-12
    assert float(rolled.maximum_conditional_mass_residual) < 2.0e-12
    assert float(rolled.maximum_conditional_mean_residual) < 2.0e-12
    expiry_dts = torch.diff(
        rolled.surface.residual_expiries_y,
        prepend=torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        expiry_dts[-1],
        torch.tensor(5.0 / 252.0, dtype=torch.float64),
    )

    stale_clock = ResidualClockDlvSurface(
        residual_expiries_y=torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )
        / 252.0,
        support_strikes=rolled.surface.support_strikes,
        density0=rolled.surface.density0,
        densities=rolled.surface.densities,
        dlvsqs=rolled.surface.dlvsqs,
        backbone_total_vols=rolled.surface.backbone_total_vols,
    )
    with pytest.raises(ValueError, match="residual clock"):
        validate_residual_clock_dlv_surface(decoder, stale_clock)

    half_day_kernel = build_functional_dlv_cubature(
        decoder,
        surface,
        raw_log_loadings=zero_loadings,
        calendar_time_increment_y=0.5 / 252.0,
    )
    with pytest.raises(ValueError, match="cubature calendar interval"):
        roll_residual_clock_dlv(
            decoder,
            half_day_kernel,
            surface_branch_uniforms=torch.tensor([0.37], dtype=torch.float64),
            front_node_uniforms=torch.tensor([0.41], dtype=torch.float64),
            front_backbone_normals=torch.tensor([0.2], dtype=torch.float64),
            elapsed_y=1.0 / 252.0,
        )


def test_front_split_preserves_old_density_options_and_black_variance() -> None:
    decoder, _, _, full = _daily_problem()
    clock = torch.tensor([5.0, 10.0], dtype=torch.float64) / 252.0
    dlvsqs = full.dlvsqs[:, -2:]
    support = full.support_strikes[:, -2:]
    previous = full.density0
    densities: list[torch.Tensor] = []
    for expiry_index, dt_y in enumerate(torch.diff(clock, prepend=clock[:1] * 0.0)):
        previous = decoder.solve_density_step_batch(
            support=support[:, expiry_index],
            dlvsqs=dlvsqs[:, expiry_index],
            time_increment_y=dt_y,
            rhs_density=previous,
        )
        densities.append(previous)
    gap = ResidualClockDlvSurface(
        residual_expiries_y=clock,
        support_strikes=support,
        density0=full.density0,
        densities=torch.stack(densities, dim=1),
        dlvsqs=dlvsqs,
        backbone_total_vols=full.backbone_total_vols[:, -2:],
    )
    strikes = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    option_types = torch.tensor([True, True, False])
    before = price_residual_clock_options(
        gap,
        current_spot=torch.ones(1, dtype=torch.float64),
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )
    inserted = insert_residual_front(
        decoder,
        gap,
        elapsed_y=1.0 / 252.0,
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )
    after = price_residual_clock_options(
        inserted.surface,
        current_spot=torch.ones(1, dtype=torch.float64),
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )

    torch.testing.assert_close(
        inserted.surface.residual_expiries_y,
        torch.tensor([1.0, 5.0, 10.0], dtype=torch.float64) / 252.0,
    )
    torch.testing.assert_close(inserted.surface.densities[:, 1:], gap.densities)
    torch.testing.assert_close(after[:, 1:], before)
    assert float(inserted.maximum_split_decoder_residual) < 2.0e-12
    front_variance = inserted.surface.backbone_total_vols[:, 0].square()
    residual_variance = (
        inserted.surface.backbone_total_vols[:, 1].square() - front_variance
    )
    torch.testing.assert_close(
        front_variance + residual_variance,
        gap.backbone_total_vols[:, 0].square(),
    )


def test_density_interpolation_front_split_is_exact_for_nonuniform_gap() -> None:
    decoder, _, _, full = _daily_problem()
    old_clock = torch.tensor([5.0, 10.0], dtype=torch.float64) / 252.0
    old_dlvsqs = full.dlvsqs[:, -2:]
    old_support = full.support_strikes[:, -2:]
    previous = full.density0
    old_densities: list[torch.Tensor] = []
    for expiry_index, dt_y in enumerate(
        torch.diff(old_clock, prepend=torch.zeros_like(old_clock[:1]))
    ):
        previous = decoder.solve_density_step_batch(
            support=old_support[:, expiry_index],
            dlvsqs=old_dlvsqs[:, expiry_index],
            time_increment_y=dt_y,
            rhs_density=previous,
        )
        old_densities.append(previous)
    gap = ResidualClockDlvSurface(
        residual_expiries_y=old_clock,
        support_strikes=old_support,
        density0=full.density0,
        densities=torch.stack(old_densities, dim=1),
        dlvsqs=old_dlvsqs,
        backbone_total_vols=full.backbone_total_vols[:, -2:],
    )
    strikes = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    option_types = torch.tensor([True, True, False])
    elapsed_y = torch.tensor(2.0 / 252.0, dtype=torch.float64)
    interpolation_weight = elapsed_y / old_clock[0]
    expected_front_density = (
        (1.0 - interpolation_weight) * gap.density0
        + interpolation_weight * gap.densities[:, 0]
    )
    expected_short_dlvsq = (
        gap.dlvsqs[:, 0]
        * gap.densities[:, 0, 1:-1]
        / expected_front_density[:, 1:-1]
    )
    result = insert_residual_front_by_density_interpolation(
        decoder,
        gap,
        elapsed_y=elapsed_y,
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )

    torch.testing.assert_close(
        result.surface.residual_expiries_y,
        torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64) / 252.0,
    )
    torch.testing.assert_close(result.surface.densities[:, 0], expected_front_density)
    torch.testing.assert_close(
        result.surface.dlvsqs[:, 0],
        expected_short_dlvsq,
        atol=3.0e-13,
        rtol=2.0e-12,
    )
    torch.testing.assert_close(
        result.surface.dlvsqs[:, 1],
        gap.dlvsqs[:, 0],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(result.surface.densities[:, 1:], gap.densities)
    before = price_residual_clock_options(
        gap,
        current_spot=torch.ones(1, dtype=torch.float64),
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )
    after = price_residual_clock_options(
        result.surface,
        current_spot=torch.ones(1, dtype=torch.float64),
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )
    torch.testing.assert_close(after[:, 1:], before, atol=0.0, rtol=0.0)
    assert float(result.maximum_old_density_change) == 0.0
    assert float(result.maximum_old_option_change) == 0.0
    assert float(result.maximum_split_decoder_residual) < 2.0e-12
    front_variance = result.surface.backbone_total_vols[:, 0].square()
    torch.testing.assert_close(
        front_variance,
        interpolation_weight * gap.backbone_total_vols[:, 0].square(),
    )
    residual_variance = (
        result.surface.backbone_total_vols[:, 1].square() - front_variance
    )
    torch.testing.assert_close(
        front_variance + residual_variance,
        gap.backbone_total_vols[:, 0].square(),
    )
    diagnostics = validate_residual_clock_dlv_surface(decoder, result.surface)
    assert float(diagnostics["maximum_recursion_residual"]) < 2.0e-12


def test_densification_splits_each_slab_in_density_and_variance_flux() -> None:
    decoder, _, _, surface = _problem((1.0, 5.0, 10.0))
    requested_days = torch.arange(1.0, 11.0, dtype=torch.float64)
    strikes = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    result = densify_residual_clock_dlv_surface(
        decoder,
        surface,
        requested_residual_expiries_y=requested_days / 252.0,
        current_spot=torch.ones(1, dtype=torch.float64),
        fixed_cash_strikes=strikes,
        is_call=torch.tensor([True, True, False]),
    )

    torch.testing.assert_close(
        result.surface.residual_expiries_y,
        requested_days / 252.0,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.surface.densities[:, result.original_expiry_indices],
        surface.densities,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.surface.dlvsqs[:, result.original_expiry_indices],
        surface.dlvsqs,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.surface.backbone_total_vols[:, result.original_expiry_indices],
        surface.backbone_total_vols,
        atol=0.0,
        rtol=0.0,
    )

    inserted_day = 3
    weight = torch.tensor((inserted_day - 1.0) / (5.0 - 1.0))
    expected_density = (
        (1.0 - weight) * surface.densities[:, 0]
        + weight * surface.densities[:, 1]
    )
    torch.testing.assert_close(
        result.surface.densities[:, inserted_day - 1],
        expected_density,
    )
    expected_flux = surface.densities[:, 1, 1:-1] * surface.dlvsqs[:, 1]
    torch.testing.assert_close(
        result.surface.densities[:, inserted_day - 1, 1:-1]
        * result.surface.dlvsqs[:, inserted_day - 1],
        expected_flux,
        atol=3.0e-18,
        rtol=2.0e-13,
    )
    expected_variance = (
        (1.0 - weight) * surface.backbone_total_vols[:, 0].square()
        + weight * surface.backbone_total_vols[:, 1].square()
    )
    torch.testing.assert_close(
        result.surface.backbone_total_vols[:, inserted_day - 1].square(),
        expected_variance,
    )
    assert float(result.maximum_old_density_change) == 0.0
    assert float(result.maximum_old_dlvsq_change) == 0.0
    assert float(result.maximum_old_backbone_variance_change) == 0.0
    assert float(result.maximum_old_option_change) == 0.0
    assert float(result.maximum_inserted_variance_flux_residual) < 1.0e-17
    assert float(result.maximum_split_decoder_residual) < 2.0e-12


def test_densification_supports_the_dy21_daily_clock_through_day_44() -> None:
    original_days = (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
    )
    decoder, _, _, surface = _problem(original_days)
    result = densify_residual_clock_dlv_surface(
        decoder,
        surface,
        requested_residual_expiries_y=(
            torch.arange(1.0, 45.0, dtype=torch.float64) / 252.0
        ),
    )
    expected_days = torch.tensor(
        [*range(1, 45), 45, 50, 55, 60, 65, 70],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        result.surface.residual_expiries_y,
        expected_days / 252.0,
        atol=0.0,
        rtol=0.0,
    )
    diagnostics = validate_residual_clock_dlv_surface(decoder, result.surface)
    assert float(diagnostics["maximum_recursion_residual"]) < 2.0e-12
    assert float(result.maximum_inserted_variance_flux_residual) < 1.0e-17


def test_densification_rejects_requested_nodes_outside_the_surface_clock() -> None:
    decoder, _, _, surface = _problem((1.0, 5.0, 10.0))

    with pytest.raises(ValueError, match="between the surface front and final"):
        densify_residual_clock_dlv_surface(
            decoder,
            surface,
            requested_residual_expiries_y=(
                torch.tensor([0.5, 2.0], dtype=torch.float64) / 252.0
            ),
        )


def test_temporary_interpolated_front_may_be_below_fitted_floor_until_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    inner = torch.linspace(0.998, 1.002, 20, dtype=torch.float64)
    inner[torch.argmin(torch.abs(inner - 1.0))] = 1.0
    inner, _ = torch.sort(inner)
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=len(inner),
        num_model_expiries=2,
        explicit_model_business_dtes=(5.0, 10.0),
        inner_relative_strikes=tuple(float(value) for value in inner),
    )
    decoder = FlexSanosDecoder(env, geometry=geometry)
    support = torch.cat(
        (
            torch.tensor([0.996], dtype=torch.float64),
            inner,
            torch.tensor([1.004], dtype=torch.float64),
        )
    )[None]
    density0 = torch.zeros_like(support)
    density0[0, torch.nonzero(support[0] == 1.0).item()] = 1.0
    first_dlvsq = torch.full((1, len(inner)), 0.0024, dtype=torch.float64)
    second_dlvsq = torch.full((1, len(inner)), 0.0030, dtype=torch.float64)
    first_density = decoder.solve_density_step_batch(
        support=support,
        dlvsqs=first_dlvsq,
        time_increment_y=torch.tensor(5.0 / 252.0, dtype=torch.float64),
        rhs_density=density0,
    )
    second_density = decoder.solve_density_step_batch(
        support=support,
        dlvsqs=second_dlvsq,
        time_increment_y=torch.tensor(5.0 / 252.0, dtype=torch.float64),
        rhs_density=first_density,
    )
    surface = ResidualClockDlvSurface(
        residual_expiries_y=torch.tensor([5.0, 10.0], dtype=torch.float64)
        / 252.0,
        support_strikes=support[:, None].expand(-1, 2, -1).clone(),
        density0=density0,
        densities=torch.stack((first_density, second_density), dim=1),
        dlvsqs=torch.stack((first_dlvsq, second_dlvsq), dim=1),
        backbone_total_vols=torch.tensor([[0.02, 0.03]], dtype=torch.float64),
    )

    def fail_recovery(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("exact density interpolation must not recover from calls")

    monkeypatch.setattr(
        residual_clock_dlv_module,
        "_recover_dlvsq",
        fail_recovery,
    )
    strikes = torch.tensor([0.999, 1.0, 1.001], dtype=torch.float64)
    option_types = torch.tensor([True, True, False])
    inserted = insert_residual_front_by_density_interpolation(
        decoder,
        surface,
        elapsed_y=1.0 / 252.0,
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )

    assert inserted.surface.temporary_frozen_front
    assert float(torch.min(inserted.surface.dlvsqs[:, 0])) < geometry.min_dlvsq
    assert float(torch.min(inserted.surface.dlvsqs[:, 0])) > 0.0
    assert bool(torch.all(inserted.surface.dlvsqs[:, 1:] > geometry.min_dlvsq))
    assert bool(torch.all(inserted.surface.dlvsqs[:, 1:] < geometry.max_dlvsq))
    assert float(inserted.maximum_old_density_change) == 0.0
    assert float(inserted.maximum_old_option_change) == 0.0
    assert float(inserted.maximum_split_decoder_residual) < 2.0e-12
    torch.testing.assert_close(
        inserted.surface.densities[:, 0, 1:-1]
        * inserted.surface.dlvsqs[:, 0],
        surface.densities[:, 0, 1:-1] * surface.dlvsqs[:, 0],
        atol=2.0e-18,
        rtol=2.0e-12,
    )
    validate_residual_clock_dlv_surface(decoder, inserted.surface)
    with pytest.raises(ValueError):
        validate_residual_clock_dlv_surface(
            decoder,
            replace(inserted.surface, temporary_frozen_front=False),
        )

    invalid_front = inserted.surface.dlvsqs.clone()
    invalid_front[:, 0, 0] = 0.0
    with pytest.raises(ValueError):
        validate_residual_clock_dlv_surface(
            decoder,
            replace(inserted.surface, dlvsqs=invalid_front),
        )
    invalid_survivor = inserted.surface.dlvsqs.clone()
    invalid_survivor[:, 1, 0] = 0.0
    with pytest.raises(ValueError):
        validate_residual_clock_dlv_surface(
            decoder,
            replace(inserted.surface, dlvsqs=invalid_survivor),
        )

    loadings = torch.zeros((*inserted.surface.dlvsqs.shape, 1), dtype=torch.float64)
    characteristics = functional_dlv_characteristics(
        decoder,
        inserted.surface,
        raw_log_loadings=loadings,
    )
    for value in (
        characteristics.psi[:, 0],
        characteristics.relative_psi[:, 0],
        characteristics.beta[:, 0],
        characteristics.alpha[:, 0],
        characteristics.mu_z[:, 0],
    ):
        assert bool(torch.all(value == 0.0))
    forged_psi = characteristics.psi.clone()
    forged_psi[:, 0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="exactly frozen"):
        apply_functional_dlv_innovation(
            decoder,
            replace(characteristics, psi=forged_psi),
            standardized_innovation=torch.zeros((1, 1), dtype=torch.float64),
            calendar_time_increment_y=1.0 / 252.0,
        )
    assert expand_residual_clock_dlv_surface(
        inserted.surface,
        copies=2,
    ).temporary_frozen_front

    kernel = build_functional_dlv_cubature(
        decoder,
        inserted.surface,
        raw_log_loadings=loadings,
        calendar_time_increment_y=1.0 / 252.0,
    )
    expected_front = inserted.surface.dlvsqs[:, None, 0].expand_as(
        kernel.dlvsqs_by_branch[:, :, 0]
    )
    torch.testing.assert_close(
        kernel.dlvsqs_by_branch[:, :, 0],
        expected_front,
        atol=0.0,
        rtol=0.0,
    )
    assert bool(torch.all(kernel.effective_mu_z[:, 0] == 0.0))
    assert bool(
        torch.all(kernel.effective_log_excess_dlvsq_diffusion[:, 0] == 0.0)
    )
    assert bool(torch.all(kernel.centered_log_increment_variance[:, 0] == 0.0))
    assert bool(torch.all(torch.isfinite(kernel.effective_mu_z)))
    assert float(kernel.maximum_density_tower_residual) < 2.0e-16
    assert float(kernel.maximum_variance_flux_tower_residual) < 2.0e-16
    assert float(kernel.maximum_decoder_residual) < 2.0e-12
    assert float(
        exact_functional_dlv_option_tower_residual(
            kernel,
            current_spot=torch.ones(1, dtype=torch.float64),
            fixed_cash_strikes=strikes,
            is_call=option_types,
        )
    ) < 2.0e-15

    rolled = roll_residual_clock_dlv(
        decoder,
        kernel,
        surface_branch_uniforms=torch.full((1,), 0.5, dtype=torch.float64),
        front_node_uniforms=torch.full((1,), 0.5, dtype=torch.float64),
        front_backbone_normals=torch.zeros(1, dtype=torch.float64),
        elapsed_y=1.0 / 252.0,
    )
    assert not rolled.surface.temporary_frozen_front
    assert bool(torch.all(rolled.surface.dlvsqs > geometry.min_dlvsq))
    assert bool(torch.all(rolled.surface.dlvsqs < geometry.max_dlvsq))
    validate_residual_clock_dlv_surface(decoder, rolled.surface)


def test_temporary_interpolated_front_may_exceed_fitted_cap_until_roll() -> None:
    decoder, _, _, base = _problem((5.0, 10.0))
    scaled_dlvsqs = 45.0 * base.dlvsqs
    assert bool(torch.all(scaled_dlvsqs < decoder.geometry.max_dlvsq))
    previous = base.density0
    densities: list[torch.Tensor] = []
    for expiry_index, expiry_dt_y in enumerate(
        torch.diff(
            base.residual_expiries_y,
            prepend=torch.zeros_like(base.residual_expiries_y[:1]),
        )
    ):
        previous = decoder.solve_density_step_batch(
            support=base.support_strikes[:, expiry_index],
            dlvsqs=scaled_dlvsqs[:, expiry_index],
            time_increment_y=expiry_dt_y,
            rhs_density=previous,
        )
        densities.append(previous)
    surface = replace(
        base,
        densities=torch.stack(densities, dim=1),
        dlvsqs=scaled_dlvsqs,
    )
    inserted = insert_residual_front_by_density_interpolation(
        decoder,
        surface,
        elapsed_y=1.0 / 252.0,
    )

    assert inserted.surface.temporary_frozen_front
    assert float(torch.max(inserted.surface.dlvsqs[:, 0])) > (
        decoder.geometry.max_dlvsq
    )
    assert bool(
        torch.all(
            inserted.surface.dlvsqs[:, 1:] < decoder.geometry.max_dlvsq
        )
    )
    validate_residual_clock_dlv_surface(decoder, inserted.surface)
    with pytest.raises(ValueError):
        validate_residual_clock_dlv_surface(
            decoder,
            replace(inserted.surface, temporary_frozen_front=False),
        )


def test_boundary_absorbed_path_remains_admissible_on_later_days() -> None:
    decoder, _, loadings, surface = _daily_problem()
    first_kernel = build_functional_dlv_cubature(
        decoder,
        surface,
        raw_log_loadings=torch.zeros_like(loadings),
        calendar_time_increment_y=1.0 / 252.0,
    )
    first_roll = roll_residual_clock_dlv(
        decoder,
        first_kernel,
        surface_branch_uniforms=torch.full((1,), 0.5, dtype=torch.float64),
        front_node_uniforms=torch.zeros(1, dtype=torch.float64),
        front_backbone_normals=torch.zeros(1, dtype=torch.float64),
        elapsed_y=1.0 / 252.0,
    )
    absorbed = first_roll.surface
    assert bool(torch.all(absorbed.densities[..., 1:-1] == 0.0))
    assert bool(torch.all(absorbed.densities == absorbed.density0[:, None]))

    requested_loadings = smooth_residual_clock_log_loadings(
        absorbed,
        annualized_log_volatility=0.05,
    )
    assert bool(torch.any(requested_loadings[:, 1:] != 0.0))
    characteristics = functional_dlv_characteristics(
        decoder,
        absorbed,
        raw_log_loadings=requested_loadings,
    )
    assert bool(torch.all(characteristics.raw_log_loadings == 0.0))
    assert bool(torch.all(characteristics.psi == 0.0))
    assert bool(torch.all(characteristics.beta == 0.0))
    assert bool(torch.all(characteristics.alpha == 0.0))
    assert bool(torch.all(characteristics.mu_z == 0.0))

    second_kernel = build_functional_dlv_cubature(
        decoder,
        absorbed,
        raw_log_loadings=requested_loadings,
        calendar_time_increment_y=1.0 / 252.0,
    )
    torch.testing.assert_close(
        second_kernel.densities_by_branch,
        absorbed.densities[:, None].expand_as(
            second_kernel.densities_by_branch
        ),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        second_kernel.dlvsqs_by_branch,
        absorbed.dlvsqs[:, None].expand_as(second_kernel.dlvsqs_by_branch),
        atol=0.0,
        rtol=0.0,
    )
    assert float(second_kernel.maximum_density_tower_residual) == 0.0
    assert float(second_kernel.maximum_variance_flux_tower_residual) == 0.0
    assert float(second_kernel.maximum_decoder_residual) < 2.0e-12
    assert float(second_kernel.fallback_coordinate_fraction) == 0.0

    second_roll = roll_residual_clock_dlv(
        decoder,
        second_kernel,
        surface_branch_uniforms=torch.full((1,), 0.5, dtype=torch.float64),
        front_node_uniforms=torch.zeros(1, dtype=torch.float64),
        front_backbone_normals=torch.zeros(1, dtype=torch.float64),
        elapsed_y=1.0 / 252.0,
    )
    assert bool(torch.all(second_roll.surface.densities[..., 1:-1] == 0.0))
    validate_residual_clock_dlv_surface(decoder, second_roll.surface)


def test_scale_search_only_recomputes_unresolved_batch_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder, _, _, full = _daily_problem()
    old_clock = torch.tensor([5.0, 10.0], dtype=torch.float64) / 252.0
    old_support = full.support_strikes[:, -2:]

    def scaled_gap(multiplier: float) -> ResidualClockDlvSurface:
        dlvsqs = multiplier * full.dlvsqs[:, -2:]
        previous = full.density0
        densities: list[torch.Tensor] = []
        for expiry_index, dt_y in enumerate(
            torch.diff(old_clock, prepend=torch.zeros_like(old_clock[:1]))
        ):
            previous = decoder.solve_density_step_batch(
                support=old_support[:, expiry_index],
                dlvsqs=dlvsqs[:, expiry_index],
                time_increment_y=dt_y,
                rhs_density=previous,
            )
            densities.append(previous)
        return ResidualClockDlvSurface(
            residual_expiries_y=old_clock,
            support_strikes=old_support,
            density0=full.density0,
            densities=torch.stack(densities, dim=1),
            dlvsqs=dlvsqs,
            backbone_total_vols=full.backbone_total_vols[:, -2:],
        )

    first = scaled_gap(1.0)
    second = scaled_gap(45.0)
    mixed = ResidualClockDlvSurface(
        residual_expiries_y=old_clock,
        support_strikes=torch.cat((first.support_strikes, second.support_strikes)),
        density0=torch.cat((first.density0, second.density0)),
        densities=torch.cat((first.densities, second.densities)),
        dlvsqs=torch.cat((first.dlvsqs, second.dlvsqs)),
        backbone_total_vols=torch.cat(
            (first.backbone_total_vols, second.backbone_total_vols)
        ),
    )
    candidates = (0.5, 5.0)
    original_solve = decoder.solve_density_step_batch
    solve_batch_sizes: list[int] = []

    def counted_solve(*args: object, **kwargs: object) -> torch.Tensor:
        support = kwargs["support"]
        assert isinstance(support, torch.Tensor)
        solve_batch_sizes.append(len(support))
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(decoder, "solve_density_step_batch", counted_solve)
    mixed_result = insert_residual_front(
        decoder,
        mixed,
        elapsed_y=1.0 / 252.0,
        scale_candidates=candidates,
    )
    monkeypatch.setattr(decoder, "solve_density_step_batch", original_solve)

    # Validation uses two full-batch solves before the search.  Candidate one
    # uses both paths, candidate two only the unresolved path, and final
    # validation uses three full-batch solves on the inserted tower.
    assert solve_batch_sizes == [2, 2, 2, 2, 1, 1, 2, 2, 2]
    first_result = insert_residual_front(
        decoder,
        first,
        elapsed_y=1.0 / 252.0,
        scale_candidates=candidates,
    )
    second_result = insert_residual_front(
        decoder,
        second,
        elapsed_y=1.0 / 252.0,
        scale_candidates=candidates,
    )
    torch.testing.assert_close(
        mixed_result.short_dlvsq_scale,
        torch.tensor([0.5, 5.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    for name in (
        "support_strikes",
        "density0",
        "densities",
        "dlvsqs",
        "backbone_total_vols",
    ):
        expected = torch.cat(
            (
                getattr(first_result.surface, name),
                getattr(second_result.surface, name),
            )
        )
        torch.testing.assert_close(
            getattr(mixed_result.surface, name),
            expected,
            atol=0.0,
            rtol=0.0,
        )
    torch.testing.assert_close(
        mixed_result.surface.residual_expiries_y,
        first_result.surface.residual_expiries_y,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        mixed_result.maximum_split_decoder_residual,
        torch.maximum(
            first_result.maximum_split_decoder_residual,
            second_result.maximum_split_decoder_residual,
        ),
    )


def test_cubature_preserves_density_and_fixed_option_towers() -> None:
    decoder, _, _, one = _daily_problem()
    surface = expand_residual_clock_dlv_surface(one, 2)
    loadings = smooth_residual_clock_log_loadings(
        surface,
        annualized_log_volatility=0.005,
    )
    kernel = build_functional_dlv_cubature(
        decoder,
        surface,
        raw_log_loadings=loadings,
        calendar_time_increment_y=1.0 / 252.0,
    )
    residual = exact_functional_dlv_option_tower_residual(
        kernel,
        current_spot=torch.ones(2, dtype=torch.float64),
        fixed_cash_strikes=torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64),
        is_call=torch.tensor([True, True, False]),
    )
    assert float(kernel.maximum_density_tower_residual) < 2.0e-14
    assert float(kernel.maximum_variance_flux_tower_residual) < 2.0e-16
    assert float(kernel.maximum_variance_flux_recursion_residual) < 2.0e-12
    assert float(kernel.maximum_decoder_residual) < 2.0e-12
    assert float(kernel.maximum_mass_residual) < 2.0e-12
    assert float(kernel.maximum_mean_residual) < 2.0e-12
    assert float(kernel.fallback_coordinate_fraction) == 0.0
    assert float(residual) < 2.0e-14


def test_enumerated_branch_and_front_node_option_tower() -> None:
    decoder, _, _, surface = _daily_problem()
    loadings = smooth_residual_clock_log_loadings(
        surface,
        annualized_log_volatility=0.005,
    )
    kernel = build_functional_dlv_cubature(
        decoder,
        surface,
        raw_log_loadings=loadings,
        calendar_time_increment_y=1.0 / 252.0,
    )
    strikes = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    option_types = torch.tensor([True, True, False])
    initial = price_residual_clock_options(
        surface,
        current_spot=torch.ones(1, dtype=torch.float64),
        fixed_cash_strikes=strikes,
        is_call=option_types,
    )[0]
    support = surface.support_strikes[0, 0]
    front_density = surface.densities[0, 0]
    front_vol = surface.backbone_total_vols[0, 0]
    front_calls = torch.sum(
        front_density[:, None]
        * _black_calls(
            support[:, None],
            strikes[None],
            front_vol.expand(len(support), len(strikes)),
        ),
        dim=0,
    )
    front_values = torch.where(
        option_types,
        front_calls,
        front_calls - 1.0 + strikes,
    )
    torch.testing.assert_close(front_values, initial[0], atol=2.0e-14, rtol=2.0e-13)
    torch.testing.assert_close(
        torch.sum(front_density * support),
        torch.ones((), dtype=torch.float64),
    )

    expiry_dts = kernel.characteristics.expiry_time_increments_y
    expected_survivors = torch.zeros_like(initial[1:])
    support_size = len(support)
    for branch_index, branch_probability in enumerate(kernel.branch_probabilities):
        conditional = torch.eye(support_size, dtype=torch.float64)
        for expiry_index in range(1, len(surface.residual_expiries_y)):
            conditional = decoder.solve_density_step_batch(
                support=surface.support_strikes[:, expiry_index].expand(
                    support_size,
                    -1,
                ),
                dlvsqs=kernel.dlvsqs_by_branch[
                    0,
                    branch_index,
                    expiry_index,
                ][None].expand(support_size, -1),
                time_increment_y=expiry_dts[expiry_index],
                rhs_density=conditional,
            )
            enumerated_density = torch.sum(
                front_density[:, None] * conditional,
                dim=0,
            )
            torch.testing.assert_close(
                enumerated_density,
                kernel.densities_by_branch[
                    0,
                    branch_index,
                    expiry_index,
                ],
                atol=3.0e-13,
                rtol=2.0e-12,
            )
            total_vol = surface.backbone_total_vols[0, expiry_index]
            calls = torch.sum(
                enumerated_density[:, None]
                * _black_calls(
                    support[:, None],
                    strikes[None],
                    total_vol.expand(support_size, len(strikes)),
                ),
                dim=0,
            )
            values = torch.where(option_types, calls, calls - 1.0 + strikes)
            expected_survivors[expiry_index - 1] += branch_probability * values
    torch.testing.assert_close(
        expected_survivors,
        initial[1:],
        atol=3.0e-13,
        rtol=2.0e-12,
    )


def test_exact_apply_updates_variance_flux_without_call_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder, _, loadings, surface = _native_clock_problem()
    characteristics = functional_dlv_characteristics(
        decoder,
        surface,
        raw_log_loadings=loadings,
    )
    innovation = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
    step_y = torch.tensor(1.0 / 252.0, dtype=torch.float64)
    expected_density = surface.densities + torch.sqrt(step_y) * torch.einsum(
        "btnf,bf->btn",
        characteristics.psi,
        innovation,
    )
    variance_flux = surface.densities[..., 1:-1] * surface.dlvsqs
    variance_flux_diffusion = (
        characteristics.psi[..., 1:-1, :] * surface.dlvsqs[..., None]
        + surface.densities[..., 1:-1, None] * characteristics.beta
    )
    expected_variance_flux = variance_flux + torch.sqrt(step_y) * torch.einsum(
        "btnf,bf->btn",
        variance_flux_diffusion,
        innovation,
    )

    def fail_call_recovery(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("the strict transition must not recover DLVs from calls")

    monkeypatch.setattr(
        "quantdata.dysanos.flex.residual_clock_dlv._calls",
        fail_call_recovery,
    )
    transition = apply_functional_dlv_innovation(
        decoder,
        characteristics,
        standardized_innovation=innovation,
        calendar_time_increment_y=step_y,
    )

    torch.testing.assert_close(transition.surface.densities, expected_density)
    torch.testing.assert_close(
        transition.surface.densities[..., 1:-1] * transition.surface.dlvsqs,
        expected_variance_flux,
        atol=2.0e-18,
        rtol=2.0e-13,
    )
    assert float(transition.maximum_variance_flux_recursion_residual) < 2.0e-12
    assert float(transition.maximum_decoder_residual) < 2.0e-12
    assert float(transition.fallback_coordinate_fraction) == 0.0


def test_positive_tiny_density_is_identified_without_flooring() -> None:
    decoder, _, _, surface = _native_clock_problem()
    tiny_dlvsqs = torch.full_like(surface.dlvsqs, 1.01e-4)
    previous_density = surface.density0
    densities: list[torch.Tensor] = []
    expiry_dts_y = torch.diff(
        surface.residual_expiries_y,
        prepend=torch.zeros_like(surface.residual_expiries_y[:1]),
    )
    for expiry_index, expiry_dt_y in enumerate(expiry_dts_y):
        previous_density = decoder.solve_density_step_batch(
            support=surface.support_strikes[:, expiry_index],
            dlvsqs=tiny_dlvsqs[:, expiry_index],
            time_increment_y=expiry_dt_y,
            rhs_density=previous_density,
        )
        densities.append(previous_density)
    tiny_densities = torch.stack(densities, dim=1)
    tiny_surface = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=surface.support_strikes,
        density0=surface.density0,
        densities=tiny_densities,
        dlvsqs=tiny_dlvsqs,
        backbone_total_vols=surface.backbone_total_vols,
    )
    minimum_density = torch.min(tiny_densities)
    assert 0.0 < float(minimum_density) < 1.0e-14

    kernel = build_functional_dlv_cubature(
        decoder,
        tiny_surface,
        raw_log_loadings=smooth_residual_clock_log_loadings(
            tiny_surface,
            annualized_log_volatility=0.005,
        ),
        calendar_time_increment_y=1.0 / 252.0,
    )

    assert float(kernel.minimum_density) > 0.0
    assert float(kernel.minimum_dlvsq) > decoder.geometry.min_dlvsq
    assert float(kernel.maximum_dlvsq) < decoder.geometry.max_dlvsq
    assert float(kernel.maximum_variance_flux_tower_residual) < 2.0e-16
    assert float(kernel.maximum_variance_flux_recursion_residual) < 2.0e-12
    assert float(kernel.maximum_decoder_residual) < 2.0e-12
    assert float(kernel.fallback_coordinate_fraction) == 0.0


def test_exact_apply_rejects_nonpositive_variance_flux() -> None:
    decoder, _, loadings, surface = _native_clock_problem()
    characteristics = functional_dlv_characteristics(
        decoder,
        surface,
        raw_log_loadings=loadings,
    )
    invalid_beta = characteristics.beta.clone()
    invalid_beta[0, 1, 0, 0] = -1.0e3
    invalid_characteristics = replace(characteristics, beta=invalid_beta)

    with pytest.raises(
        FunctionalDlvInadmissibleError,
        match="variance-flux transition became nonfinite or nonpositive",
    ):
        apply_functional_dlv_innovation(
            decoder,
            invalid_characteristics,
            standardized_innovation=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
            calendar_time_increment_y=1.0 / 252.0,
        )


def test_exact_api_requires_float64_and_identified_dynamic_density() -> None:
    decoder, _, loadings, surface = _native_clock_problem()
    float_surface = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y.float(),
        support_strikes=surface.support_strikes.float(),
        density0=surface.density0.float(),
        densities=surface.densities.float(),
        dlvsqs=surface.dlvsqs.float(),
        backbone_total_vols=surface.backbone_total_vols.float(),
    )
    with pytest.raises(ValueError, match="requires torch.float64"):
        validate_residual_clock_dlv_surface(decoder, float_surface)
    with pytest.raises(FunctionalDlvInadmissibleError, match="psi/q drift"):
        functional_dlv_characteristics(
            decoder,
            surface,
            raw_log_loadings=loadings,
            density_mass_floor=1.0,
        )


def test_ingress_canonicalizer_repairs_only_numerical_decode_noise() -> None:
    decoder, _, _, surface = _native_clock_problem()
    noisy_density0 = surface.density0.clone()
    noisy_density0[0, 0] = -5.0e-14
    noisy_densities = surface.densities.clone()
    noisy_densities[0, 2, 3] += 7.0e-14
    noisy = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=surface.support_strikes,
        density0=noisy_density0,
        densities=noisy_densities,
        dlvsqs=surface.dlvsqs,
        backbone_total_vols=surface.backbone_total_vols,
    )
    result = canonicalize_residual_clock_dlv_ingress(
        decoder,
        noisy,
        current_spot=torch.tensor([5000.0], dtype=torch.float64),
        fixed_cash_strikes=torch.tensor([4500.0, 5000.0], dtype=torch.float64),
        is_call=torch.tensor([True, False]),
    )

    torch.testing.assert_close(result.surface.density0, surface.density0)
    torch.testing.assert_close(result.surface.densities, surface.densities)
    torch.testing.assert_close(result.surface.dlvsqs, surface.dlvsqs)
    assert float(result.maximum_input_density0_perturbation) == pytest.approx(
        5.0e-14
    )
    assert float(result.maximum_input_density_tower_perturbation) == pytest.approx(
        7.0e-14,
        rel=2.0e-3,
    )
    assert float(result.maximum_input_density_perturbation) == pytest.approx(
        7.0e-14,
        rel=2.0e-3,
    )
    assert float(result.maximum_mass_residual) < 2.0e-14
    assert float(result.maximum_forward_residual) < 2.0e-14
    assert float(result.maximum_recursion_residual) < 2.0e-14
    assert float(result.minimum_density) >= 0.0
    assert float(result.minimum_dlvsq_bound_margin) > 0.0
    assert result.maximum_fixed_option_perturbation is not None
    assert float(result.maximum_fixed_option_perturbation) > 0.0


def test_ingress_canonicalizer_builds_adjacent_two_node_initial_law() -> None:
    decoder, _, _, surface = _native_clock_problem()
    support = surface.support_strikes.clone()
    exact_index = int(torch.nonzero(support[0, 0] == 1.0).item())
    support[:, :, exact_index] = 1.01
    left_index = exact_index - 1
    left_support = support[:, 0, left_index]
    right_support = support[:, 0, exact_index]
    right_weight = (1.0 - left_support) / (right_support - left_support)
    density0 = torch.zeros_like(surface.density0)
    density0[:, left_index] = 1.0 - right_weight
    density0[:, exact_index] = right_weight
    densities: list[torch.Tensor] = []
    previous = density0
    expiry_dts_y = torch.diff(
        surface.residual_expiries_y,
        prepend=torch.zeros_like(surface.residual_expiries_y[:1]),
    )
    for expiry_index, expiry_dt_y in enumerate(expiry_dts_y):
        previous = decoder.solve_density_step_batch(
            support=support[:, expiry_index],
            dlvsqs=surface.dlvsqs[:, expiry_index],
            time_increment_y=expiry_dt_y,
            rhs_density=previous,
        )
        densities.append(previous)
    off_node = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=support,
        density0=density0,
        densities=torch.stack(densities, dim=1),
        dlvsqs=surface.dlvsqs,
        backbone_total_vols=surface.backbone_total_vols,
    )

    result = canonicalize_residual_clock_dlv_ingress(decoder, off_node)

    assert torch.count_nonzero(result.surface.density0) == 2
    torch.testing.assert_close(result.surface.density0, density0)
    torch.testing.assert_close(
        torch.sum(result.surface.density0 * support[:, 0], dim=-1),
        torch.ones((1,), dtype=torch.float64),
    )


def test_ingress_canonicalizer_rejects_material_density_discrepancy() -> None:
    decoder, _, _, surface = _native_clock_problem()
    inconsistent_densities = surface.densities.clone()
    inconsistent_densities[0, 1, 3] += 1.0e-5
    inconsistent = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=surface.support_strikes,
        density0=surface.density0,
        densities=inconsistent_densities,
        dlvsqs=surface.dlvsqs,
        backbone_total_vols=surface.backbone_total_vols,
    )

    with pytest.raises(ValueError, match="differs materially"):
        canonicalize_residual_clock_dlv_ingress(decoder, inconsistent)


def test_ingress_canonicalizer_requires_support_to_bracket_one() -> None:
    decoder, _, _, surface = _native_clock_problem()
    support = torch.linspace(
        0.1,
        0.9,
        surface.support_strikes.shape[-1],
        dtype=torch.float64,
    )[None, None].expand_as(surface.support_strikes)
    unbracketed = ResidualClockDlvSurface(
        residual_expiries_y=surface.residual_expiries_y,
        support_strikes=support,
        density0=surface.density0,
        densities=surface.densities,
        dlvsqs=surface.dlvsqs,
        backbone_total_vols=surface.backbone_total_vols,
    )

    with pytest.raises(ValueError, match="bracket X=1"):
        canonicalize_residual_clock_dlv_ingress(decoder, unbracketed)


def test_categorical_sampler_skips_zero_mass_at_uniform_boundaries() -> None:
    indices = residual_clock_dlv_module._sample_categorical(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.3, 0.0, 0.7],
                [0.3, 0.0, 0.7],
            ],
            dtype=torch.float64,
        ),
        torch.tensor([0.0, 0.3, 0.0], dtype=torch.float64),
    )

    assert indices.tolist() == [2, 2, 0]
