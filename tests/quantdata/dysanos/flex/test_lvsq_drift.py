"""Mathematical tests for the fixed-contract discrete-LVsq drift."""

from __future__ import annotations

from datetime import date

import numpy as np
import torch

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.fixed_grid_dlv import (
    build_fixed_grid_dlv_kernel,
    exact_fixed_grid_dlv_diagnostics,
    make_fixed_grid_dlv_randoms,
    sample_fixed_grid_dlv_paths,
    set_constant_backbone_volatility,
    strip_fixed_grid_black_backbone,
)
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2 import (
    fixed_grid_attempt2_conditional_densities,
    fixed_grid_attempt2_sde_roll,
    make_fixed_grid_attempt2_randoms,
)
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2_triangular import (
    TriangularFixedGridState,
    regrid_triangular_state_to_revealed_spot,
    triangular_fixed_grid_physical_roll,
    triangular_fixed_grid_roll,
)
from quantdata.dysanos.flex.floating_grid_dlv import (
    build_floating_grid_dlv_kernel,
    floating_surface_from_regrid,
    price_floating_surface_options,
    replace_floating_far_expiry,
    roll_floating_grid_dlv_kernel,
    smooth_floating_log_loadings,
)
from quantdata.dysanos.flex.last_expiry_ar1 import (
    fit_last_expiry_ar1,
    interpolate_ar1_raw_curves,
    make_antithetic_ar1_normals,
    torch_last_expiry_ar1,
)
from quantdata.dysanos.flex.lvsq_drift import (
    density_martingale_lvsq_step,
    discrete_lvsq_grid_operator_product,
    fixed_contract_lvsq_drift,
    fixed_contract_lvsq_euler_step,
    fixed_horizon_lvsq_exponential_step,
    regrid_raw_dlvsq_to_relative_spot,
    smooth_lvsq_log_loadings,
    symmetric_rademacher_cubature,
)
from quantdata.dysanos.flex.model import (
    NativeParameters,
    bound_native_raw,
    raw_from_native_parameters,
    split_forward_mu_native_raw,
)
from quantdata.dysanos.flex.run_all_state_regrid_dlv_audit import (
    _direct_survivor_option_values,
)
from quantdata.dysanos.flex.run_fixed_grid_dlv_attempt2_triangular import (
    _stratified_antithetic_uniforms,
    _stratified_block_mean_observations,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.torch import BaseEnvironment


def _problem() -> tuple[
    BaseEnvironment,
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    FlexSanosDecoder,
    torch.Tensor,
    torch.Tensor,
]:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=4,
        explicit_model_business_dtes=(5.0, 10.0, 15.0, 20.0),
        inner_relative_strikes=(0.72, 0.84, 0.94, 1.0, 1.08, 1.25),
    )
    decoder = FlexSanosDecoder(env, geometry=geometry)
    parameters = NativeParameters(
        dlvsqs=torch.tensor(
            [[
                [0.11, 0.09, 0.07, 0.08, 0.10, 0.13],
                [0.12, 0.10, 0.08, 0.09, 0.11, 0.14],
                [0.13, 0.11, 0.09, 0.10, 0.12, 0.15],
                [0.14, 0.12, 0.10, 0.11, 0.13, 0.16],
            ]],
            dtype=env.fdtype,
        ),
        forward_vols=torch.tensor([[0.20, 0.21, 0.22, 0.23]], dtype=env.fdtype),
        eta=torch.tensor([[0.30, 0.31, 0.32, 0.33]], dtype=env.fdtype),
    )
    raw = raw_from_native_parameters(parameters, geometry=geometry)
    with torch.no_grad():
        bounded = decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
    support = bounded.support_strikes[0, 0]
    loadings = smooth_lvsq_log_loadings(
        support,
        num_model_expiries=geometry.num_model_expiries,
        annualized_log_volatility=0.10,
    )
    return env, geometry, decoder, raw, loadings


def _front_term_eta_problem() -> tuple[
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    FlexSanosDecoder,
    torch.Tensor,
    torch.Tensor,
]:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=3,
        explicit_model_business_dtes=(5.0, 10.0, 15.0),
    )
    decoder = FlexSanosDecoder(env, geometry=geometry)
    parameters = NativeParameters(
        dlvsqs=torch.tensor(
            [[
                [0.11, 0.09, 0.07, 0.08, 0.10, 0.13],
                [0.12, 0.10, 0.08, 0.09, 0.11, 0.14],
                [0.13, 0.11, 0.09, 0.10, 0.12, 0.15],
            ]],
            dtype=env.fdtype,
        ),
        forward_vols=torch.tensor([[0.20, 0.21, 0.22]], dtype=env.fdtype),
        eta=torch.tensor([[0.30, 0.32, 0.32]], dtype=env.fdtype),
    )
    raw = raw_from_native_parameters(parameters, geometry=geometry)
    bounded = bound_native_raw(raw, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=bounded.dlvsqs,
        forward_vols=bounded.forward_vols,
        eta=bounded.eta,
    )
    loadings = smooth_lvsq_log_loadings(
        surface.support_strikes[0, 0],
        num_model_expiries=geometry.num_model_expiries,
        annualized_log_volatility=0.10,
    )[..., :2]
    return geometry, decoder, raw, loadings


def test_grid_operator_has_mass_and_forward_null_vectors_and_full_rank() -> None:
    env, geometry, decoder, raw, _ = _problem()
    parameters = decoder.forward_batch(
        dlvsqs=torch.full(
            (1, geometry.num_model_expiries, geometry.num_inner_strikes),
            0.12,
            dtype=env.fdtype,
        ),
        forward_vols=torch.full((1, geometry.num_model_expiries), 0.2, dtype=env.fdtype),
        eta=torch.full((1, geometry.num_model_expiries), 0.3, dtype=env.fdtype),
    )
    support = parameters.support_strikes[0, 0]
    identity = torch.eye(geometry.num_inner_strikes, dtype=env.fdtype)
    matrix = discrete_lvsq_grid_operator_product(
        support,
        identity,
        expiry_time_increment_y=decoder.model_dts_y[1],
    )

    torch.testing.assert_close(torch.sum(matrix, dim=0), torch.zeros(6, dtype=env.fdtype))
    torch.testing.assert_close(support @ matrix, torch.zeros(6, dtype=env.fdtype))
    assert int(torch.linalg.matrix_rank(matrix)) == geometry.num_inner_strikes
    assert raw.shape == (1, geometry.native_state_size)


def test_pdf_volatility_and_drift_equations_hold_in_direct_raw_state() -> None:
    _, geometry, decoder, raw, loadings = _problem()
    result = fixed_contract_lvsq_drift(
        decoder,
        raw,
        raw_log_loadings=loadings,
    )

    assert float(result.maximum_volatility_equation_residual) < 2.0e-14
    assert float(result.maximum_drift_equation_residual) < 2.0e-18
    assert float(result.maximum_density_diffusion_mass_residual) < 2.0e-14
    assert float(result.maximum_density_diffusion_forward_residual) < 2.0e-14
    assert float(result.minimum_density) > 0.0
    raw_drift_blocks = result.raw_drift.reshape(
        1,
        geometry.num_model_expiries,
        geometry.expiry_block_state_size,
    )
    raw_diffusion_blocks = result.raw_diffusion.reshape(
        1,
        geometry.num_model_expiries,
        geometry.expiry_block_state_size,
        -1,
    )
    assert bool(torch.all(raw_drift_blocks[:, 0] == 0.0))
    assert bool(torch.all(raw_diffusion_blocks[:, 0] == 0.0))
    assert bool(torch.all(raw_drift_blocks[..., geometry.num_inner_strikes :] == 0.0))
    assert bool(torch.all(raw_diffusion_blocks[..., geometry.num_inner_strikes :, :] == 0.0))


def test_front_term_eta_fixed_contract_drift_uses_native_layout() -> None:
    geometry, decoder, raw, loadings = _front_term_eta_problem()
    result = fixed_contract_lvsq_drift(
        decoder,
        raw,
        raw_log_loadings=loadings,
    )
    raw_dlvsq_drift, raw_forward_drift, raw_eta_drift = (
        split_forward_mu_native_raw(result.raw_drift, geometry=geometry)
    )
    raw_dlvsq_diffusion, raw_forward_diffusion, raw_eta_diffusion = (
        split_forward_mu_native_raw(
            torch.movedim(result.raw_diffusion, -1, -2),
            geometry=geometry,
        )
    )

    assert result.raw_drift.shape == raw.shape
    assert result.raw_diffusion.shape == raw.shape + (loadings.shape[-1],)
    assert bool(torch.all(raw_dlvsq_drift[:, 0] == 0.0))
    assert bool(torch.all(raw_dlvsq_diffusion[:, :, 0] == 0.0))
    assert bool(torch.all(raw_forward_drift == 0.0))
    assert bool(torch.all(raw_forward_diffusion == 0.0))
    assert bool(torch.all(raw_eta_drift == 0.0))
    assert bool(torch.all(raw_eta_diffusion == 0.0))
    assert float(result.maximum_volatility_equation_residual) < 2.0e-14
    assert float(result.maximum_drift_equation_residual) < 2.0e-18


def test_attempt2_spot_reveal_and_expiry_roll_have_exact_density_towers() -> None:
    _, geometry, decoder, raw, _ = _problem()
    parameters = bound_native_raw(raw, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    support = surface.support_strikes[0, 0]
    support_size = len(support)
    conditional = fixed_grid_attempt2_conditional_densities(
        decoder,
        support=support.expand(support_size, -1),
        endpoint_dlvsqs=parameters.dlvsqs.expand(support_size, -1, -1),
        front_node_indices=torch.arange(support_size),
    )

    # Averaging the revealed q0 and rolled q1,...,q3 over S1 ~ q_t^1
    # recovers the four old fixed-expiry densities q_t^1,...,q_t^4.
    tower_mean = torch.einsum("r,rts->ts", surface.densities[0, 0], conditional)
    torch.testing.assert_close(tower_mean, surface.densities[0], atol=2.0e-14, rtol=0.0)


def test_attempt2_sde_roll_preserves_the_fixed_grid_and_pathwise_forwards() -> None:
    env, _, decoder, raw, loadings = _problem()
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    randoms = make_fixed_grid_attempt2_randoms(
        num_states=1,
        num_paths=8,
        num_substeps=2,
        num_factors=loadings.shape[-1],
        dtype=env.fdtype,
        device=env.device,
        generator=generator,
        antithetic=True,
    )
    result = fixed_grid_attempt2_sde_roll(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
        randoms=randoms,
        fixed_cash_strikes=torch.exp(
            torch.linspace(-0.3, 0.3, 11, dtype=env.fdtype)
        ),
    )

    assert result.aligned_conditional_densities.shape == (1, 8, 4, 8)
    assert result.aligned_conditional_option_values.shape == (1, 8, 4, 11)
    assert result.endpoint_unconditional_survivor_option_values.shape == (
        1,
        8,
        3,
        11,
    )
    assert result.maximum_fixed_grid_change == 0.0
    assert result.maximum_frozen_coordinate_change == 0.0
    assert result.maximum_conditional_mass_residual < 2.0e-14
    assert result.maximum_conditional_forward_residual < 2.0e-14
    assert result.minimum_density_during_sde > 0.0
    assert result.minimum_raw_cap_margin > 0.0


def test_attempt2_triangular_roll_removes_only_the_expired_fixed_expiry() -> None:
    env, geometry, decoder, raw, loadings = _problem()
    parameters = bound_native_raw(raw, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    state = TriangularFixedGridState(
        support=surface.support_strikes[:, 0],
        densities=surface.densities,
        dlvsqs=parameters.dlvsqs,
    )
    roll = triangular_fixed_grid_roll(
        decoder,
        state,
        raw_log_loadings=loadings,
        minimum_dlvsq=geometry.min_dlvsq,
        expiry_time_increment_y=decoder.model_dts_y[0],
        calendar_time_increment_y=5.0 / 252.0,
        standard_normals=torch.zeros((1, 2, 4), dtype=env.fdtype),
        uniforms=torch.full((1,), 0.5, dtype=env.fdtype),
        fixed_cash_strikes=surface.support_strikes[0, 0, 1:-1],
    )

    assert roll.next_state is not None
    assert roll.initial_call_values.shape == (1, 4, 6)
    assert roll.final_call_values.shape == (1, 4, 6)
    assert roll.endpoint_unconditional_call_values.shape == (1, 4, 6)
    assert roll.next_state.densities.shape == (1, 3, 8)
    assert roll.next_state.dlvsqs.shape == (1, 3, 6)
    assert roll.maximum_mass_residual < 2.0e-14
    assert roll.maximum_forward_residual < 2.0e-14

    regridded = regrid_triangular_state_to_revealed_spot(
        roll.next_state, roll.revealed_spot
    )
    torch.testing.assert_close(
        torch.sum(regridded.densities * regridded.support[:, None], dim=-1),
        torch.ones((1, 3), dtype=env.fdtype),
        atol=2.0e-14,
        rtol=0.0,
    )
    cash_strikes = surface.support_strikes[0, 0, 1:-1]
    old_payoffs = torch.relu(roll.next_state.support[:, :, None] - cash_strikes)
    new_payoffs = torch.relu(
        regridded.support[:, :, None]
        - cash_strikes[None, None] / roll.revealed_spot[:, None, None]
    )
    old_prices = torch.einsum("bts,bsk->btk", roll.next_state.densities, old_payoffs)
    new_prices = torch.einsum("bts,bsk->btk", regridded.densities, new_payoffs)
    torch.testing.assert_close(
        roll.revealed_spot[:, None, None] * new_prices,
        old_prices,
        atol=2.0e-14,
        rtol=0.0,
    )


def test_physical_triangular_roll_applies_raw_innovation_without_analytic_drift() -> None:
    env, geometry, decoder, raw, loadings = _problem()
    parameters = bound_native_raw(raw, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    state = TriangularFixedGridState(
        support=surface.support_strikes[:, 0],
        densities=surface.densities,
        dlvsqs=parameters.dlvsqs,
    )
    normals = torch.tensor([[0.2, -0.3, 0.1, 0.4]], dtype=env.fdtype)
    result = triangular_fixed_grid_physical_roll(
        decoder,
        state,
        raw_log_loadings=loadings,
        minimum_dlvsq=geometry.min_dlvsq,
        expiry_time_increment_y=decoder.model_dts_y[0],
        calendar_time_increment_y=5.0 / 252.0,
        standard_normals=normals,
        uniforms=torch.full((1,), 0.5, dtype=env.fdtype),
        fixed_cash_strikes=surface.support_strikes[0, 0, 1:-1],
    )

    expected_raw = torch.log(parameters.dlvsqs - geometry.min_dlvsq)
    expected_raw = expected_raw + (5.0 / 252.0) ** 0.5 * torch.einsum(
        "tif,bf->bti", loadings, normals
    )
    expected_dlvsqs = geometry.min_dlvsq + torch.exp(expected_raw)
    assert result.next_state is not None
    torch.testing.assert_close(
        result.next_state.dlvsqs,
        expected_dlvsqs[:, 1:],
        atol=2.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.endpoint_unconditional_call_values[:, 0],
        result.initial_call_values[:, 0],
        atol=2.0e-14,
        rtol=0.0,
    )
    assert result.maximum_mass_residual < 2.0e-14
    assert result.maximum_forward_residual < 2.0e-14


def test_stratified_antithetic_uniforms_cover_each_block_and_pair() -> None:
    pairs_per_block = 5
    uniforms = _stratified_antithetic_uniforms(
        num_states=2,
        num_pairs=10,
        num_weeks=3,
        pairs_per_block=pairs_per_block,
        dtype=torch.float64,
        device="cpu",
        generator=torch.Generator().manual_seed(20260912),
    )
    paired = uniforms.reshape(2, 10, 2, 3)
    torch.testing.assert_close(
        paired[:, :, 0] + paired[:, :, 1],
        torch.ones((2, 10, 3), dtype=torch.float64),
    )
    base = paired[:, :, 0].reshape(2, 2, pairs_per_block, 3)
    strata = torch.floor(base * pairs_per_block).to(torch.int64)
    expected = torch.arange(pairs_per_block).reshape(1, 1, -1, 1).expand_as(strata)
    torch.testing.assert_close(torch.sort(strata, dim=2).values, expected)

    independent_spot = _stratified_antithetic_uniforms(
        num_states=2,
        num_pairs=10,
        num_weeks=3,
        pairs_per_block=pairs_per_block,
        dtype=torch.float64,
        device="cpu",
        generator=torch.Generator().manual_seed(20260912),
        complement_pairs=False,
    ).reshape(2, 2, 2 * pairs_per_block, 3)
    independent_strata = torch.floor(
        independent_spot * (2 * pairs_per_block)
    ).to(torch.int64)
    independent_expected = torch.arange(2 * pairs_per_block).reshape(
        1, 1, -1, 1
    ).expand_as(independent_strata)
    torch.testing.assert_close(
        torch.sort(independent_strata, dim=2).values,
        independent_expected,
    )


def test_stratified_inference_uses_independent_block_means() -> None:
    values = np.arange(2 * 12 * 3, dtype=np.float64).reshape(2, 12, 3)
    blocks = _stratified_block_mean_observations(values, pairs_per_block=3)

    assert blocks.shape == (2, 2, 3)
    np.testing.assert_allclose(blocks[:, 0], np.mean(values[:, :6], axis=1))
    np.testing.assert_allclose(blocks[:, 1], np.mean(values[:, 6:], axis=1))


def test_density_diffusion_matches_decoder_jvp() -> None:
    _, _, decoder, raw, loadings = _problem()
    result = fixed_contract_lvsq_drift(decoder, raw, raw_log_loadings=loadings)

    def densities(value: torch.Tensor) -> torch.Tensor:
        geometry = decoder.geometry
        blocks = value.reshape(
            len(value),
            geometry.num_model_expiries,
            geometry.expiry_block_state_size,
        )
        dlvsqs = geometry.min_dlvsq + torch.exp(
            blocks[..., : geometry.num_inner_strikes]
        )
        forward = geometry.min_forward_vol + (
            geometry.max_forward_vol - geometry.min_forward_vol
        ) * torch.sigmoid(blocks[..., geometry.num_inner_strikes])
        eta = geometry.min_eta + (
            geometry.max_eta - geometry.min_eta
        ) * torch.sigmoid(blocks[..., geometry.num_inner_strikes + 1])
        return decoder.forward_batch(
            dlvsqs=dlvsqs,
            forward_vols=forward,
            eta=eta,
        ).densities

    for factor_index in range(loadings.shape[-1]):
        _, jvp = torch.autograd.functional.jvp(
            densities,
            raw,
            result.raw_diffusion[..., factor_index],
        )
        torch.testing.assert_close(
            jvp,
            result.density_diffusion[..., factor_index],
            atol=2.0e-13,
            rtol=2.0e-11,
        )


def test_euler_step_copies_front_and_all_non_dlv_coordinates_exactly() -> None:
    _, geometry, decoder, raw, loadings = _problem()
    result = fixed_contract_lvsq_drift(decoder, raw, raw_log_loadings=loadings)
    stepped = fixed_contract_lvsq_euler_step(
        result,
        raw,
        calendar_time_increment_y=5.0 / 252.0,
        standard_normals=torch.tensor([[0.4, -0.3, 0.2, -0.1]], dtype=raw.dtype),
    )
    before = raw.reshape(1, geometry.num_model_expiries, geometry.expiry_block_state_size)
    after = stepped.reshape_as(before)

    torch.testing.assert_close(after[:, 0], before[:, 0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        after[..., geometry.num_inner_strikes :],
        before[..., geometry.num_inner_strikes :],
        atol=0.0,
        rtol=0.0,
    )
    assert bool(torch.any(after[:, 1:, : geometry.num_inner_strikes] != before[:, 1:, : geometry.num_inner_strikes]))


def test_smooth_loading_norm_has_requested_sine_taper() -> None:
    _, geometry, decoder, raw, loadings = _problem()
    parameters = decoder.forward_batch(
        dlvsqs=torch.full(
            (1, geometry.num_model_expiries, geometry.num_inner_strikes),
            0.12,
            dtype=raw.dtype,
        ),
        forward_vols=torch.full((1, geometry.num_model_expiries), 0.2, dtype=raw.dtype),
        eta=torch.full((1, geometry.num_model_expiries), 0.3, dtype=raw.dtype),
    )
    expected = 0.10 * torch.sin(
        torch.arange(1, geometry.num_inner_strikes + 1, dtype=raw.dtype)
        * np.pi
        / (geometry.num_inner_strikes + 1)
    )

    torch.testing.assert_close(
        torch.linalg.vector_norm(loadings[1:], dim=-1),
        expected[None].expand(geometry.num_model_expiries - 1, -1),
    )
    assert bool(torch.all(loadings[0] == 0.0))
    assert parameters.densities.shape[-1] == geometry.num_inner_strikes + 2


def test_rademacher_cubature_has_exact_first_two_moments() -> None:
    innovations, probabilities = symmetric_rademacher_cubature(
        4,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert innovations.shape == (16, 4)
    torch.testing.assert_close(
        torch.sum(probabilities), torch.tensor(1.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        torch.sum(probabilities[:, None] * innovations, dim=0),
        torch.zeros(4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.einsum("r,rf,rg->fg", probabilities, innovations, innovations),
        torch.eye(4, dtype=torch.float64),
    )


def test_relative_spot_regrid_is_identity_for_unit_return() -> None:
    _, geometry, _, raw, _ = _problem()
    result = regrid_raw_dlvsq_to_relative_spot(
        raw,
        torch.ones(1, dtype=raw.dtype),
        geometry=geometry,
    )

    torch.testing.assert_close(result.raw_states, raw, atol=0.0, rtol=0.0)
    assert float(result.extrapolated_coordinate_fraction) == 0.0
    assert float(result.maximum_absolute_raw_change) == 0.0


def test_relative_spot_regrid_shifts_log_linear_curve() -> None:
    _, geometry, _, raw, _ = _problem()
    gross = torch.tensor([1.03], dtype=raw.dtype)
    blocks = raw.reshape(
        1,
        geometry.num_model_expiries,
        geometry.expiry_block_state_size,
    ).clone()
    log_strikes = torch.log(
        torch.as_tensor(geometry.inner_relative_strikes, dtype=raw.dtype)
    )
    blocks[..., : geometry.num_inner_strikes] = 0.4 + 0.7 * log_strikes
    result = regrid_raw_dlvsq_to_relative_spot(
        blocks.reshape_as(raw),
        gross,
        geometry=geometry,
    )
    regridded = result.raw_states.reshape_as(blocks)
    interior = slice(0, geometry.num_inner_strikes - 1)

    torch.testing.assert_close(
        regridded[..., interior],
        blocks[..., interior] + 0.7 * torch.log(gross)[:, None, None],
    )
    torch.testing.assert_close(
        regridded[..., geometry.num_inner_strikes :],
        blocks[..., geometry.num_inner_strikes :],
        atol=0.0,
        rtol=0.0,
    )
    assert float(result.extrapolated_coordinate_fraction) > 0.0


def test_finite_horizon_exponential_step_matches_density_means_exactly() -> None:
    _, geometry, decoder, raw, loadings = _problem()
    result = fixed_horizon_lvsq_exponential_step(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
        moment_tolerance=2.0e-13,
    )

    assert result.raw_states_by_branch.shape == (
        1,
        16,
        geometry.native_state_size,
    )
    assert result.newton_iterations_by_expiry
    assert max(result.newton_iterations_by_expiry) <= 12
    assert float(result.maximum_density_dlvsq_mean_residual) < 2.0e-13
    assert float(result.maximum_density_mean_residual) < 2.0e-12
    assert float(result.maximum_decoder_density_residual) < 2.0e-14
    assert float(result.maximum_frozen_coordinate_change) == 0.0
    assert float(result.minimum_raw_cap_margin) > 0.0
    assert bool(
        torch.any(result.effective_raw_drift != result.local_raw_drift)
    )


def test_fixed_grid_kernel_has_exact_spot_and_option_towers() -> None:
    env, _, decoder, raw, loadings = _problem()
    strikes = torch.exp(torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype))
    kernel = build_fixed_grid_dlv_kernel(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
        moment_tolerance=2.0e-13,
    )

    diagnostics = exact_fixed_grid_dlv_diagnostics(
        kernel,
        fixed_cash_strikes=strikes,
    )

    assert float(diagnostics.maximum_spot_mean_residual) < 2.0e-14
    assert float(diagnostics.maximum_density_tower_residual) < 2.0e-12
    assert float(diagnostics.maximum_option_tower_residual) < 2.0e-12


def test_constant_backbone_helper_targets_the_actual_black_factor() -> None:
    _, _, decoder, raw, _ = _problem()
    adjusted, forward_parameters = set_constant_backbone_volatility(
        raw,
        decoder=decoder,
        annualized_volatility=0.10,
    )
    parameters = bound_native_raw(adjusted, geometry=decoder.geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )

    torch.testing.assert_close(
        parameters.forward_vols,
        forward_parameters[None].expand_as(parameters.forward_vols),
    )
    torch.testing.assert_close(
        parameters.eta,
        torch.full_like(parameters.eta, 0.40),
    )
    torch.testing.assert_close(
        surface.backbone_total_vols,
        0.10 * torch.sqrt(decoder.model_expiries_y)[None],
        atol=2.0e-15,
        rtol=2.0e-14,
    )


def test_stripped_backbone_gives_an_exact_x_only_path_law() -> None:
    env, _, decoder, raw, loadings = _problem()
    strikes = torch.exp(torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype))
    kernel = strip_fixed_grid_black_backbone(
        build_fixed_grid_dlv_kernel(
            decoder,
            raw,
            raw_log_loadings=loadings,
            calendar_time_increment_y=5.0 / 252.0,
        )
    )
    randoms = make_fixed_grid_dlv_randoms(
        batch_size=1,
        num_paths=64,
        dtype=env.fdtype,
        device=env.device,
        generator=torch.Generator(device=env.device).manual_seed(20260910),
        antithetic=True,
    )
    changed_normals = type(randoms)(
        surface_branch_uniforms=randoms.surface_branch_uniforms,
        front_node_uniforms=randoms.front_node_uniforms,
        front_backbone_normals=randoms.front_backbone_normals + 17.0,
    )
    first = sample_fixed_grid_dlv_paths(
        decoder,
        kernel,
        fixed_cash_strikes=strikes,
        randoms=randoms,
    )
    second = sample_fixed_grid_dlv_paths(
        decoder,
        kernel,
        fixed_cash_strikes=strikes,
        randoms=changed_normals,
    )

    assert bool(torch.all(kernel.initial_surface.backbone_total_vols == 0.0))
    torch.testing.assert_close(first.normalized_spot, second.normalized_spot)
    torch.testing.assert_close(
        first.fixed_cash_option_values,
        second.fixed_cash_option_values,
    )
    torch.testing.assert_close(
        first.normalized_spot,
        torch.gather(
            kernel.initial_surface.support_strikes[:, None, 0].expand(
                1,
                64,
                -1,
            ),
            -1,
            first.revealed_front_nodes.unsqueeze(-1),
        ).squeeze(-1),
    )


def test_density_martingale_step_recovers_valid_exact_mean_dlv_branches() -> None:
    _, geometry, decoder, raw, loadings = _problem()

    result = density_martingale_lvsq_step(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
    )

    assert result.raw_states_by_branch.shape == (
        1,
        16,
        geometry.native_state_size,
    )
    assert float(result.maximum_density_mean_residual) < 2.0e-14
    assert float(result.maximum_decoder_density_residual) < 2.0e-13
    assert float(result.maximum_mass_residual) < 2.0e-14
    assert float(result.maximum_forward_residual) < 2.0e-14
    assert float(result.minimum_density) > 0.0
    assert float(result.minimum_dlvsq) > geometry.min_dlvsq
    assert float(result.maximum_dlvsq) < geometry.max_dlvsq
    assert float(result.fallback_coordinate_fraction) == 0.0


def test_front_term_eta_finite_steps_preserve_non_dlv_coordinates() -> None:
    geometry, decoder, raw, loadings = _front_term_eta_problem()
    density_step = density_martingale_lvsq_step(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
    )
    exponential_step = fixed_horizon_lvsq_exponential_step(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
        moment_tolerance=2.0e-13,
    )
    initial_dlvsq, initial_forward, initial_eta = split_forward_mu_native_raw(
        raw,
        geometry=geometry,
    )

    for raw_states_by_branch in (
        density_step.raw_states_by_branch,
        exponential_step.raw_states_by_branch,
    ):
        branch_dlvsq, branch_forward, branch_eta = split_forward_mu_native_raw(
            raw_states_by_branch,
            geometry=geometry,
        )
        torch.testing.assert_close(
            branch_dlvsq[:, :, 0],
            initial_dlvsq[:, None, 0].expand_as(branch_dlvsq[:, :, 0]),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            branch_forward,
            initial_forward[:, None].expand_as(branch_forward),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            branch_eta,
            initial_eta[:, None].expand_as(branch_eta),
            atol=0.0,
            rtol=0.0,
        )

    assert float(density_step.maximum_density_mean_residual) < 2.0e-14
    assert float(density_step.maximum_decoder_density_residual) < 2.0e-13
    assert float(exponential_step.maximum_density_mean_residual) < 2.0e-12
    assert float(exponential_step.maximum_decoder_density_residual) < 2.0e-14
    assert float(exponential_step.maximum_frozen_coordinate_change) == 0.0


def test_fixed_grid_paths_condition_on_spot_and_surface_branch() -> None:
    env, geometry, decoder, raw, loadings = _problem()
    strikes = torch.exp(torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype))
    kernel = build_fixed_grid_dlv_kernel(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
        moment_tolerance=2.0e-13,
    )
    generator = torch.Generator(device=env.device).manual_seed(20260908)
    randoms = make_fixed_grid_dlv_randoms(
        batch_size=1,
        num_paths=256,
        dtype=env.fdtype,
        device=env.device,
        generator=generator,
        antithetic=True,
    )

    torch.testing.assert_close(
        randoms.surface_branch_uniforms[:, 0::2]
        + randoms.surface_branch_uniforms[:, 1::2],
        torch.ones((1, 128), dtype=env.fdtype),
    )
    torch.testing.assert_close(
        randoms.front_node_uniforms[:, 0::2]
        + randoms.front_node_uniforms[:, 1::2],
        torch.ones((1, 128), dtype=env.fdtype),
    )
    torch.testing.assert_close(
        randoms.front_backbone_normals[:, 0::2]
        + randoms.front_backbone_normals[:, 1::2],
        torch.zeros((1, 128), dtype=env.fdtype),
    )

    paths = sample_fixed_grid_dlv_paths(
        decoder,
        kernel,
        fixed_cash_strikes=strikes,
        randoms=randoms,
    )

    assert paths.normalized_spot.shape == (1, 256)
    assert paths.fixed_cash_option_values.shape == (
        1,
        256,
        geometry.num_model_expiries,
        len(strikes),
    )
    assert bool(torch.all(paths.normalized_spot > 0.0))
    assert bool(
        torch.all(
            paths.surface_branch_indices[:, 0::2]
            + paths.surface_branch_indices[:, 1::2]
            == len(kernel.finite_step.branch_probabilities) - 1
        )
    )
    assert float(paths.maximum_conditional_mass_residual) < 2.0e-12
    assert float(paths.maximum_conditional_mean_residual) < 2.0e-12
    regridded = paths.regridded_state
    assert regridded.densities.shape == (
        1,
        256,
        geometry.num_model_expiries - 1,
        geometry.num_inner_strikes + 2,
    )
    assert regridded.dlvsqs.shape == (
        1,
        256,
        geometry.num_model_expiries - 1,
        geometry.num_inner_strikes,
    )
    torch.testing.assert_close(
        torch.gather(
            regridded.relative_support,
            -1,
            paths.revealed_front_nodes.unsqueeze(-1),
        ).squeeze(-1),
        torch.ones((1, 256), dtype=env.fdtype),
    )
    assert float(regridded.maximum_cash_price_residual) < 2.0e-12
    torch.testing.assert_close(
        paths.fixed_cash_option_values[:, :, 1:],
        regridded.fixed_cash_option_values,
        atol=0.0,
        rtol=0.0,
    )
    intrinsic = torch.where(
        (strikes <= 1.0).reshape(1, 1, -1),
        torch.relu(paths.normalized_spot[..., None] - strikes),
        torch.relu(strikes - paths.normalized_spot[..., None]),
    )
    torch.testing.assert_close(
        paths.fixed_cash_option_values[:, :, 0],
        intrinsic,
        atol=0.0,
        rtol=0.0,
    )


def test_floating_grid_roll_preserves_prices_and_exact_density_tower() -> None:
    env, geometry, decoder, raw, loadings = _problem()
    strikes = torch.exp(torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype))
    fixed_kernel = build_fixed_grid_dlv_kernel(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
    )
    first_randoms = make_fixed_grid_dlv_randoms(
        batch_size=1,
        num_paths=32,
        dtype=env.fdtype,
        device=env.device,
        generator=torch.Generator(device=env.device).manual_seed(20260909),
        antithetic=True,
    )
    first_paths = sample_fixed_grid_dlv_paths(
        decoder,
        fixed_kernel,
        fixed_cash_strikes=strikes,
        randoms=first_randoms,
    )
    surface = floating_surface_from_regrid(decoder, first_paths.regridded_state)

    assert surface.densities.shape == (
        32,
        geometry.num_model_expiries,
        geometry.num_inner_strikes + 2,
    )
    torch.testing.assert_close(
        torch.sum(surface.densities, dim=-1),
        torch.ones((32, geometry.num_model_expiries), dtype=env.fdtype),
        atol=2.0e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(surface.densities * surface.support_strikes, dim=-1),
        torch.ones((32, geometry.num_model_expiries), dtype=env.fdtype),
        atol=2.0e-13,
        rtol=0.0,
    )
    first_regrid_prices = price_floating_surface_options(
        surface,
        current_spot=first_paths.normalized_spot.reshape(-1),
        fixed_cash_strikes=strikes,
    )
    torch.testing.assert_close(
        first_regrid_prices[:, : geometry.num_model_expiries - 1],
        first_paths.regridded_state.fixed_cash_option_values.reshape(
            32,
            geometry.num_model_expiries - 1,
            len(strikes),
        ),
        atol=3.0e-13,
        rtol=2.0e-12,
    )
    path_spot = first_paths.normalized_spot.reshape(-1)
    relative_prices = price_floating_surface_options(
        surface,
        current_spot=torch.ones_like(path_spot),
        fixed_cash_strikes=strikes,
    )
    path_cash_strikes = path_spot[:, None] * strikes[None]
    path_option_types = (strikes <= 1.0)[None].expand(32, -1)
    path_cash_prices = price_floating_surface_options(
        surface,
        current_spot=path_spot,
        fixed_cash_strikes=path_cash_strikes,
        is_call=path_option_types,
    )
    torch.testing.assert_close(
        path_cash_prices,
        path_spot[:, None, None] * relative_prices,
        atol=3.0e-13,
        rtol=2.0e-12,
    )

    floating_loadings = smooth_floating_log_loadings(
        surface.support_strikes[:, 0],
        num_model_expiries=geometry.num_model_expiries,
        annualized_log_volatility=0.10,
    )
    floating_kernel = build_floating_grid_dlv_kernel(
        decoder,
        surface,
        raw_log_loadings=floating_loadings,
        calendar_time_increment_y=5.0 / 252.0,
    )

    assert float(floating_kernel.maximum_density_mean_residual) < 3.0e-14
    assert float(floating_kernel.maximum_decoder_density_residual) < 3.0e-12
    assert float(floating_kernel.maximum_mass_residual) < 3.0e-13
    assert float(floating_kernel.maximum_mean_residual) < 3.0e-13
    assert float(floating_kernel.minimum_dlvsq) > geometry.min_dlvsq
    assert float(floating_kernel.maximum_dlvsq) < geometry.max_dlvsq

    second_randoms = make_fixed_grid_dlv_randoms(
        batch_size=1,
        num_paths=32,
        dtype=env.fdtype,
        device=env.device,
        generator=torch.Generator(device=env.device).manual_seed(20260910),
        antithetic=True,
    )
    rolled = roll_floating_grid_dlv_kernel(
        decoder,
        floating_kernel,
        surface_branch_uniforms=second_randoms.surface_branch_uniforms[0],
        front_node_uniforms=second_randoms.front_node_uniforms[0],
        front_backbone_normals=second_randoms.front_backbone_normals[0],
    )

    assert bool(torch.all(rolled.gross_spot_return > 0.0))
    assert bool(
        torch.all(
            rolled.surface_branch_indices[0::2]
            + rolled.surface_branch_indices[1::2]
            == len(floating_kernel.branch_probabilities) - 1
        )
    )
    assert float(rolled.maximum_conditional_mass_residual) < 3.0e-12
    assert float(rolled.maximum_conditional_mean_residual) < 3.0e-12
    torch.testing.assert_close(
        torch.sum(rolled.surface.densities, dim=-1),
        torch.ones((32, geometry.num_model_expiries), dtype=env.fdtype),
        atol=3.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(
            rolled.surface.densities * rolled.surface.support_strikes,
            dim=-1,
        ),
        torch.ones((32, geometry.num_model_expiries), dtype=env.fdtype),
        atol=3.0e-12,
        rtol=0.0,
    )
    next_spot = path_spot * rolled.gross_spot_return
    direct_survivors = _direct_survivor_option_values(
        surface,
        rolled.surface,
        current_spot=path_spot,
        gross_spot_return=rolled.gross_spot_return,
        revealed_front_nodes=rolled.revealed_front_nodes,
        fixed_cash_strikes=path_cash_strikes,
        is_call=path_option_types,
    )
    regridded_survivors = price_floating_surface_options(
        rolled.surface,
        current_spot=next_spot,
        fixed_cash_strikes=path_cash_strikes,
        is_call=path_option_types,
    )[:, : geometry.num_model_expiries - 1]
    torch.testing.assert_close(
        direct_survivors,
        regridded_survivors,
        atol=4.0e-13,
        rtol=2.0e-12,
    )


def test_floating_grid_can_repeat_sixteen_weekly_rolls() -> None:
    env, geometry, decoder, raw, loadings = _problem()
    strikes = torch.exp(torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype))
    generator = torch.Generator(device=env.device).manual_seed(20260912)
    fixed_kernel = build_fixed_grid_dlv_kernel(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
    )
    randoms = make_fixed_grid_dlv_randoms(
        batch_size=1,
        num_paths=16,
        dtype=env.fdtype,
        device=env.device,
        generator=generator,
        antithetic=True,
    )
    paths = sample_fixed_grid_dlv_paths(
        decoder,
        fixed_kernel,
        fixed_cash_strikes=strikes,
        randoms=randoms,
    )
    surface = floating_surface_from_regrid(decoder, paths.regridded_state)

    for _ in range(15):
        floating_loadings = smooth_floating_log_loadings(
            surface.support_strikes[:, 0],
            num_model_expiries=geometry.num_model_expiries,
            annualized_log_volatility=0.10,
        )
        kernel = build_floating_grid_dlv_kernel(
            decoder,
            surface,
            raw_log_loadings=floating_loadings,
            calendar_time_increment_y=5.0 / 252.0,
        )
        randoms = make_fixed_grid_dlv_randoms(
            batch_size=1,
            num_paths=16,
            dtype=env.fdtype,
            device=env.device,
            generator=generator,
            antithetic=True,
        )
        rolled = roll_floating_grid_dlv_kernel(
            decoder,
            kernel,
            surface_branch_uniforms=randoms.surface_branch_uniforms[0],
            front_node_uniforms=randoms.front_node_uniforms[0],
            front_backbone_normals=randoms.front_backbone_normals[0],
        )
        surface = rolled.surface
        assert float(kernel.maximum_density_mean_residual) < 3.0e-14
        assert float(kernel.maximum_decoder_density_residual) < 3.0e-12
        assert float(rolled.maximum_conditional_mass_residual) < 3.0e-12
        assert float(rolled.maximum_conditional_mean_residual) < 3.0e-12

    torch.testing.assert_close(
        torch.sum(surface.densities, dim=-1),
        torch.ones((16, geometry.num_model_expiries), dtype=env.fdtype),
        atol=3.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(surface.densities * surface.support_strikes, dim=-1),
        torch.ones((16, geometry.num_model_expiries), dtype=env.fdtype),
        atol=3.0e-12,
        rtol=0.0,
    )


def test_floating_far_expiry_can_be_replaced_without_changing_survivors() -> None:
    env, geometry, decoder, raw, loadings = _problem()
    kernel = build_fixed_grid_dlv_kernel(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=5.0 / 252.0,
    )
    randoms = make_fixed_grid_dlv_randoms(
        batch_size=1,
        num_paths=8,
        dtype=env.fdtype,
        device=env.device,
        generator=torch.Generator(device=env.device).manual_seed(20260913),
        antithetic=True,
    )
    paths = sample_fixed_grid_dlv_paths(
        decoder,
        kernel,
        fixed_cash_strikes=torch.tensor([1.0], dtype=env.fdtype),
        randoms=randoms,
    )
    surface = floating_surface_from_regrid(decoder, paths.regridded_state)
    far_dlvsqs = 1.15 * surface.dlvsqs[:, -1]
    replacement = replace_floating_far_expiry(
        decoder,
        surface,
        far_dlvsqs=far_dlvsqs,
    )

    torch.testing.assert_close(
        replacement.surface.densities[:, :-1],
        surface.densities[:, :-1],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        replacement.surface.dlvsqs[:, :-1],
        surface.dlvsqs[:, :-1],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        replacement.surface.dlvsqs[:, -1],
        far_dlvsqs,
        atol=0.0,
        rtol=0.0,
    )
    assert bool(
        torch.any(replacement.surface.densities[:, -1] != surface.densities[:, -1])
    )
    assert float(replacement.maximum_mass_residual) < 2.0e-14
    assert float(replacement.maximum_mean_residual) < 2.0e-14


def test_plain_last_expiry_ar1_is_stationary_and_antithetic() -> None:
    dates = tuple(
        get_business_days(
            date(2024, 1, 2),
            date(2025, 5, 30),
        )
    )
    generator = np.random.default_rng(20260914)
    transition = np.diag([0.80, 0.65, 0.50])
    scores = np.zeros((len(dates), 3), dtype=np.float64)
    for index in range(1, len(dates)):
        scores[index] = (
            scores[index - 1] @ transition
            + 0.05 * generator.standard_normal(3)
        )
    strikes = np.exp(np.linspace(-0.3, 0.3, 6))
    loadings = generator.standard_normal((3, len(strikes)))
    curves = -2.0 + scores @ loadings
    fit = fit_last_expiry_ar1(
        curves,
        dates,
        strikes,
        rank=3,
    )
    model = torch_last_expiry_ar1(
        fit,
        dtype=torch.float64,
        device="cpu",
    )
    normals = make_antithetic_ar1_normals(
        num_states=2,
        pairs_per_state=4,
        rank=model.rank,
        dtype=torch.float64,
        device="cpu",
        generator=torch.Generator().manual_seed(20260914),
    )
    interpolated, extrapolated = interpolate_ar1_raw_curves(
        model,
        torch.as_tensor(curves[:16], dtype=torch.float64),
        torch.as_tensor(strikes, dtype=torch.float64)[None].expand(16, -1),
    )

    assert fit.diagnostics.weekly_spectral_radius < 1.0
    assert fit.diagnostics.num_validation_pairs > 0
    assert fit.diagnostics.explained_variance_fraction > 0.999999
    torch.testing.assert_close(
        normals[0::2] + normals[1::2],
        torch.zeros_like(normals[0::2]),
    )
    torch.testing.assert_close(
        interpolated,
        torch.as_tensor(curves[:16], dtype=torch.float64),
    )
    assert not bool(torch.any(extrapolated))
