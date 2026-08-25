"""Mathematical tests for the P-free finite martingale transport core."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.config import (
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.finite_martingale_transport import (
    MartingaleCouplingInfeasibleError,
    call_prices_from_distribution,
    condition_sanos_step_on_spot,
    reconstruct_sanos_step_calls,
    solve_finite_martingale_coupling,
    solve_shared_sanos_density_step_from_dlvsqs,
    solve_shared_sanos_step_transition,
    solve_shared_sanos_step_transition_from_dlvsqs,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.torch import BaseEnvironment


def test_known_binomial_martingale_coupling() -> None:
    source_support = np.asarray((0.75, 1.25))
    source_probability = np.asarray((0.5, 0.5))
    target_support = np.asarray((0.5, 1.0, 1.5))
    target_probability = np.asarray((0.25, 0.5, 0.25))

    result = solve_finite_martingale_coupling(
        source_support,
        source_probability,
        target_support,
        target_probability,
    )

    expected_joint = np.asarray(((0.25, 0.25, 0.0), (0.0, 0.25, 0.25)))
    np.testing.assert_allclose(result.joint_mass, expected_joint, atol=1.0e-12)
    np.testing.assert_allclose(
        result.transition_probabilities,
        2.0 * expected_joint,
        atol=1.0e-12,
    )


def test_convex_order_infeasibility_is_rejected_before_transport() -> None:
    with pytest.raises(MartingaleCouplingInfeasibleError) as caught:
        solve_finite_martingale_coupling(
            np.asarray((0.0, 2.0)),
            np.asarray((0.5, 0.5)),
            np.asarray((0.5, 1.5)),
            np.asarray((0.5, 0.5)),
        )

    certificate = caught.value.convex_order
    assert certificate.mean_difference == pytest.approx(0.0)
    assert certificate.minimum_call_slack < 0.0
    assert not certificate.is_feasible


def test_coupling_marginal_and_martingale_residuals_are_small() -> None:
    result = solve_finite_martingale_coupling(
        np.asarray((0.8, 1.0, 1.2)),
        np.asarray((0.25, 0.5, 0.25)),
        np.asarray((0.6, 0.9, 1.1, 1.4)),
        np.asarray((0.2, 0.3, 0.3, 0.2)),
    )

    np.testing.assert_allclose(
        result.joint_mass.sum(axis=1),
        result.source_probabilities,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        result.joint_mass.sum(axis=0),
        result.target_probabilities,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        result.joint_mass @ result.target_support,
        result.source_probabilities * result.source_support,
        atol=1.0e-11,
    )
    assert result.diagnostics.maximum_source_marginal_residual < 1.0e-11
    assert result.diagnostics.maximum_target_marginal_residual < 1.0e-11
    assert result.diagnostics.maximum_martingale_residual < 1.0e-11


def test_conditional_calls_reconstruct_target_two_week_curve() -> None:
    result = solve_finite_martingale_coupling(
        np.asarray((0.75, 1.25)),
        np.asarray((0.5, 0.5)),
        np.asarray((0.5, 1.0, 1.5)),
        np.asarray((0.25, 0.5, 0.25)),
    )
    strikes = np.linspace(0.25, 1.75, 31)

    conditional = result.conditional_call_prices(strikes)
    reconstructed = result.reconstructed_target_call_prices(strikes)
    direct = call_prices_from_distribution(
        result.target_support,
        result.target_probabilities,
        strikes,
    )

    assert conditional.shape == (2, len(strikes))
    np.testing.assert_allclose(reconstructed, direct, atol=1.0e-12)
    np.testing.assert_allclose(result.target_call_prices(strikes), direct, atol=1.0e-12)


def test_normalized_conditional_rows_are_mean_one() -> None:
    result = solve_finite_martingale_coupling(
        np.asarray((0.75, 1.25)),
        np.asarray((0.5, 0.5)),
        np.asarray((0.5, 1.0, 1.5)),
        np.asarray((0.25, 0.5, 0.25)),
    )

    relative_support, probabilities = result.normalized_conditional_distributions()

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.sum(relative_support * probabilities, axis=1),
        1.0,
        atol=1.0e-12,
    )


def _native_sanos_step():
    support = np.asarray((0.5, 0.9, 1.1, 1.5))
    intrinsic_calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(intrinsic_calls, prepend=1.0) / np.diff(support, prepend=0.0)
    previous_density = np.diff(slopes, append=0.0)
    base_weights = np.asarray((0.02, 0.025))
    w_minus = base_weights / (support[1:-1] - support[:-2])
    w_plus = base_weights / (support[2:] - support[1:-1])

    # Reproduce the exact tridiagonal matrix used by the native SANOS solve.
    matrix = np.asarray(
        (
            (1.0, -w_minus[0], 0.0, 0.0),
            (0.0, 1.0 + w_minus[0] + w_plus[0], -w_minus[1], 0.0),
            (0.0, -w_plus[0], 1.0 + w_minus[1] + w_plus[1], 0.0),
            (0.0, 0.0, -w_plus[1], 1.0),
        )
    )
    next_density = np.linalg.solve(matrix, previous_density)
    return solve_shared_sanos_step_transition(
        support,
        previous_density,
        next_density,
        w_minus=w_minus,
        w_plus=w_plus,
    )


def test_native_shared_sanos_inverse_is_a_martingale_transition() -> None:
    step = _native_sanos_step()
    transition = step.transition_probabilities

    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(transition @ step.support, step.support, atol=1.0e-12)
    np.testing.assert_allclose(
        step.previous_density @ transition,
        step.next_density,
        atol=1.0e-12,
    )
    assert np.min(transition) >= 0.0


def test_banded_native_transition_matches_dense_inverse_orientation() -> None:
    step = _native_sanos_step()
    support = step.support
    base_weights = np.asarray((0.02, 0.025))
    w_minus = base_weights / (support[1:-1] - support[:-2])
    w_plus = base_weights / (support[2:] - support[1:-1])
    matrix = np.asarray(
        (
            (1.0, -w_minus[0], 0.0, 0.0),
            (0.0, 1.0 + w_minus[0] + w_plus[0], -w_minus[1], 0.0),
            (0.0, -w_plus[0], 1.0 + w_minus[1] + w_plus[1], 0.0),
            (0.0, 0.0, -w_plus[1], 1.0),
        )
    )

    dense_transition = np.linalg.solve(matrix.T, np.eye(len(support)))

    np.testing.assert_allclose(
        step.transition_probabilities,
        dense_transition,
        atol=2.0e-15,
        rtol=2.0e-15,
    )


def test_native_sanos_transition_reconstructs_calls_with_backbone_increment() -> None:
    step = _native_sanos_step()
    strikes = np.linspace(0.65, 1.35, 15)

    reconstructed, direct = reconstruct_sanos_step_calls(
        step,
        strikes,
        previous_backbone_total_vol=0.08,
        next_backbone_total_vol=0.14,
        quadrature_points=80,
    )

    np.testing.assert_allclose(reconstructed, direct, atol=2.0e-12, rtol=2.0e-12)


def test_actual_flex_shared_surface_exposes_native_w2_martingale_transition() -> None:
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=2,
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlvsqs = torch.asarray(
        [[[0.12, 0.09, 0.07, 0.08, 0.11, 0.14], [0.10, 0.08, 0.06, 0.07, 0.10, 0.13]]],
        dtype=env.fdtype,
    )
    with torch.no_grad():
        surface = decoder.forward_batch(
            dlvsqs=dlvsqs,
            forward_vols=torch.asarray([[0.22, 0.24]], dtype=env.fdtype),
            eta=torch.asarray([[0.35, 0.40]], dtype=env.fdtype),
        )
    support0 = surface.support_strikes[0, 0].numpy()
    support1 = surface.support_strikes[0, 1].numpy()
    np.testing.assert_allclose(support0, support1, atol=0.0, rtol=0.0)
    support = support0
    step = solve_shared_sanos_step_transition_from_dlvsqs(
        support,
        surface.densities[0, 0].numpy(),
        surface.densities[0, 1].numpy(),
        dlvsqs=dlvsqs[0, 1].numpy(),
        time_increment_y=float(decoder.model_dts_y[1]),
        tolerance=2.0e-10,
    )

    np.testing.assert_allclose(
        step.transition_probabilities @ support,
        support,
        atol=2.0e-11,
    )
    reconstructed, direct = reconstruct_sanos_step_calls(
        step,
        np.linspace(0.7, 1.3, 17),
        previous_backbone_total_vol=float(surface.backbone_total_vols[0, 0]),
        next_backbone_total_vol=float(surface.backbone_total_vols[0, 1]),
        quadrature_points=80,
    )
    np.testing.assert_allclose(reconstructed, direct, atol=2.0e-11, rtol=2.0e-11)

    propagated_density = solve_shared_sanos_density_step_from_dlvsqs(
        support,
        surface.densities[0, 0].numpy(),
        dlvsqs=dlvsqs[0, 1].numpy(),
        time_increment_y=float(decoder.model_dts_y[1]),
        tolerance=2.0e-10,
    )
    np.testing.assert_allclose(
        propagated_density,
        surface.densities[0, 1].numpy(),
        atol=2.0e-11,
    )


def test_posterior_transition_conditions_only_on_observed_total_spot() -> None:
    step = _native_sanos_step()
    observed_spot = 0.97
    law = condition_sanos_step_on_spot(
        step,
        observed_spot,
        previous_backbone_total_vol=0.08,
        next_backbone_total_vol=0.14,
    )

    np.testing.assert_allclose(law.latent_posterior_probabilities.sum(), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(law.component_probabilities.sum(), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.sum(law.component_probabilities * law.component_forward_spots),
        observed_spot,
        atol=1.0e-13,
    )
    strikes = np.linspace(0.7, 1.3, 13)
    calls = law.call_prices(strikes)
    assert calls.shape == strikes.shape
    assert np.all(calls >= np.maximum(observed_spot - strikes, 0.0) - 1.0e-13)
    assert np.all(calls <= observed_spot + 1.0e-13)


def test_posterior_conditional_calls_reconstruct_unconditional_w2_curve() -> None:
    step = _native_sanos_step()
    previous_vol = 0.08
    next_vol = 0.14
    strikes = np.linspace(0.7, 1.3, 13)
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    backbone = np.exp(-0.5 * previous_vol**2 + np.sqrt(2.0) * previous_vol * nodes)
    weights = weights / np.sqrt(np.pi)
    reconstructed = np.zeros_like(strikes)
    for latent_index, latent_spot in enumerate(step.support):
        for multiplier, weight in zip(backbone, weights, strict=True):
            law = condition_sanos_step_on_spot(
                step,
                float(latent_spot * multiplier),
                previous_backbone_total_vol=previous_vol,
                next_backbone_total_vol=next_vol,
            )
            reconstructed += step.previous_density[latent_index] * weight * law.call_prices(strikes)
    _, direct = reconstruct_sanos_step_calls(
        step,
        strikes,
        previous_backbone_total_vol=previous_vol,
        next_backbone_total_vol=next_vol,
        quadrature_points=80,
    )

    np.testing.assert_allclose(reconstructed, direct, atol=1.0e-11, rtol=1.0e-11)
