"""Focused tests for the durable native shared-strike Q audit."""

from __future__ import annotations

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.finite_martingale_transport import (
    build_revealed_node_rolled_sanos_chain,
    solve_shared_sanos_step_transition_from_dlvsqs,
)
from quantdata.dysanos.flex.run_native_sanos_q_audit import (
    NativeSanosQAuditConfig,
    _even_indices,
    audit_native_sanos_surface,
    summarize_date_audits,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.torch import BaseEnvironment


def _synthetic_surface() -> tuple[np.ndarray, ...]:
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=3,
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlvsqs = torch.asarray(
        [
            [
                [0.12, 0.09, 0.07, 0.08, 0.11, 0.14],
                [0.10, 0.08, 0.06, 0.07, 0.10, 0.13],
                [0.09, 0.07, 0.05, 0.06, 0.09, 0.12],
            ]
        ],
        dtype=env.fdtype,
    )
    with torch.no_grad():
        surface = decoder.forward_batch(
            dlvsqs=dlvsqs,
            forward_vols=torch.asarray([[0.22, 0.24, 0.25]], dtype=env.fdtype),
            eta=torch.asarray([[0.35, 0.40, 0.42]], dtype=env.fdtype),
        )
    return (
        surface.support_strikes[0].numpy(),
        surface.densities[0].numpy(),
        surface.dlvsqs[0].numpy(),
        decoder.model_dts_y.numpy(),
        surface.backbone_total_vols[0].numpy(),
    )


def test_native_surface_audit_passes_w2_and_sequential_w3() -> None:
    support, density, dlv, dts, backbone = _synthetic_surface()
    config = NativeSanosQAuditConfig(
        maximum_expiry=3,
        mc_paths_per_replication=2_000,
        mc_replications=2,
        posterior_grid_points=65,
        seed=17,
        minimum_atm_split_percent=0.0,
        maximum_mc_standard_errors=8.0,
    )

    audit = audit_native_sanos_surface(
        support_strikes=support,
        densities=density,
        dlvsqs=dlv,
        model_time_increments_y=dts,
        backbone_total_vols=backbone,
        config=config,
    )

    assert audit["w2_passed"]
    assert audit["all_requested_expiries_passed"]
    reports = audit["expiry_reports"]
    assert len(reports) == 2
    for report in reports:
        assert report["exact_passed"]
        assert report["noncollapse_passed"]
        assert report["all_strike_call_tower_maximum_absolute_residual"] < 1.0e-12
        assert report["transition_diagnostics"]["maximum_row_martingale_residual"] < 1.0e-12
        assert report["monte_carlo_atm"]["mark_to_market"]["pooled"]["paths"] == 4_000
    revealed = reports[0]["revealed_node_rolled_chain"]
    assert revealed["passed"]
    assert revealed["strict_ordinary_cap_projection"]["status"] == "exact"
    assert (
        revealed["strict_ordinary_cap_projection"]["maximum_absolute_call_tower_error_bps_spot"]
        < 1.0e-9
    )
    for node in revealed["representative_nodes"]:
        assert node["density0_atom_mass"] == 1.0
        assert node["first_density_transition_row_residual"] < 1.0e-13
        np.testing.assert_allclose(node["dlvsq_time_scale_factors"][0], 5.0 / 4.8, rtol=1.0e-14)


def test_summary_uses_equal_date_votes_and_w2_cannot_be_hidden() -> None:
    template = {
        "expiry_reports": [
            {
                "noncollapse": {"weighted_conditional_std": 0.1},
                "monte_carlo_atm": {"mark_to_market": {"pooled": {"mean": 0.01}}},
            }
        ],
        "w2_passed": True,
        "all_requested_expiries_passed": True,
    }
    failed = {
        "expiry_reports": [
            {
                "noncollapse": {"weighted_conditional_std": 0.3},
                "monte_carlo_atm": {"mark_to_market": {"pooled": {"mean": -0.03}}},
            },
            {"passed": True},
        ],
        "w2_passed": False,
        "all_requested_expiries_passed": True,
    }

    summary = summarize_date_audits([template, failed])

    assert summary["date_weighting"] == "equal"
    assert summary["w2_pass_percent"] == 50.0
    assert summary["w2_weighted_conditional_std_mean"] == 0.2
    np.testing.assert_allclose(summary["w2_atm_mark_mean_equal_date_average"], -0.01, atol=1.0e-16)
    assert not summary["w2_semantic_gate_passed"]
    assert summary["all_requested_semantic_gates_passed"]


def test_revealed_node_custom_grid_chain_is_exact_and_rescales_front_dlv() -> None:
    support, density, dlv, dts, backbone = _synthetic_surface()
    steps = tuple(
        solve_shared_sanos_step_transition_from_dlvsqs(
            support[target],
            density[target - 1],
            density[target],
            dlvsqs=dlv[target],
            time_increment_y=float(dts[target]),
        )
        for target in (1, 2)
    )
    node = 4

    chain = build_revealed_node_rolled_sanos_chain(
        support[0],
        steps,
        source_node_index=node,
        old_dlvsqs=dlv,
        old_time_increments_y=dts,
        old_backbone_total_vols=backbone,
    )

    expected_atom = np.zeros(support.shape[1])
    expected_atom[node] = 1.0
    np.testing.assert_array_equal(chain.density0, expected_atom)
    np.testing.assert_allclose(
        chain.normalized_support, support[0] / support[0, node], atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        chain.densities[0], steps[0].transition_probabilities[node], atol=1.0e-14
    )
    np.testing.assert_allclose(
        chain.densities[1],
        steps[0].transition_probabilities[node] @ steps[1].transition_probabilities,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        chain.densities @ chain.normalized_support,
        1.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        chain.dlvsq_time_scale_factors,
        (5.0 / 4.8, 1.0),
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(chain.dlvsqs[0], dlv[1] * (5.0 / 4.8), rtol=1.0e-14)
    np.testing.assert_allclose(chain.dlvsqs[1], dlv[2], rtol=1.0e-14)
    np.testing.assert_allclose(
        chain.backbone_total_vols,
        np.sqrt(backbone[1:] ** 2 - backbone[0] ** 2),
        rtol=1.0e-14,
    )
    # Re-run the native SANOS step on the shifted natural grid.  This verifies
    # that the timing-scaled DLV coordinates actually decode the propagated
    # row, rather than merely recording the expected scale factor.
    previous_density = chain.density0
    for expiry_index in range(len(chain.densities)):
        decoded = solve_shared_sanos_step_transition_from_dlvsqs(
            chain.normalized_support,
            previous_density,
            chain.densities[expiry_index],
            dlvsqs=chain.dlvsqs[expiry_index],
            time_increment_y=float(dts[expiry_index]),
        )
        np.testing.assert_allclose(
            previous_density @ decoded.transition_probabilities,
            chain.densities[expiry_index],
            atol=1.0e-13,
        )
        previous_density = chain.densities[expiry_index]


def test_even_date_selection_keeps_endpoints() -> None:
    np.testing.assert_array_equal(_even_indices(10, 4), np.asarray((0, 3, 6, 9)))
    np.testing.assert_array_equal(_even_indices(4, 0), np.arange(4))
