"""Focused tests for the checkpointed residual-clock RN OCE path runner."""

from __future__ import annotations

import gc
from dataclasses import replace

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.model import NativeParameters
from quantdata.dysanos.flex.residual_clock_dlv import (
    build_functional_dlv_cubature,
    residual_clock_surface_from_decoded,
    roll_residual_clock_dlv,
    smooth_residual_clock_log_loadings,
)
from quantdata.dysanos.flex.run_residual_clock_rn_oce_paths import (
    NUM_INSTRUMENTS,
    OutputShapes,
    _densify_front_twenty,
    _option_returns,
    _randoms,
    _write_issue_features,
    _write_option_return,
    _write_spot_return,
    make_work_plan,
    maturity_entry_day,
    open_output_maps,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.torch import BaseEnvironment


def test_work_plan_preserves_adjacent_pairs_and_global_order() -> None:
    plan = make_work_plan(num_paths=14, path_batch_size=6)

    assert [
        (item.pair_offset, item.num_pairs, item.path_first, item.path_stop)
        for item in plan
    ] == [
        (0, 3, 0, 6),
        (3, 3, 6, 12),
        (6, 1, 12, 14),
    ]
    assert all(item.path_first % 2 == item.path_stop % 2 == 0 for item in plan)

    with pytest.raises(ValueError, match="positive and even"):
        make_work_plan(num_paths=13, path_batch_size=6)
    with pytest.raises(ValueError, match="positive and even"):
        make_work_plan(num_paths=14, path_batch_size=5)


@pytest.mark.parametrize(
    ("transition_index", "horizon_days", "expected"),
    (
        (3, 5, None),
        (4, 5, 0),
        (19, 20, 0),
        (28, 5, 24),
        (43, 20, 24),
    ),
)
def test_maturity_transition_maps_to_the_correct_issue_day(
    transition_index: int,
    horizon_days: int,
    expected: int | None,
) -> None:
    assert (
        maturity_entry_day(
            transition_index=transition_index,
            horizon_days=horizon_days,
            issue_days=25,
        )
        == expected
    )


def test_all_return_coordinates_are_written_once_by_transition_44() -> None:
    counts = np.zeros((25, NUM_INSTRUMENTS), dtype=np.int64)
    for transition_index in range(44):
        if transition_index < 25:
            counts[transition_index, 0] += 1
        for expiry_position, horizon_days in enumerate((5, 10, 15, 20)):
            entry_day = maturity_entry_day(
                transition_index=transition_index,
                horizon_days=horizon_days,
                issue_days=25,
            )
            if entry_day is not None:
                first = 1 + 12 * expiry_position
                counts[entry_day, first : first + 12] += 1

    assert np.all(counts == 1)


def test_random_stream_is_batch_invariant_and_pair_antithetic() -> None:
    complete = make_work_plan(num_paths=14, path_batch_size=14)[0]
    expected = _randoms(
        complete,
        total_paths=14,
        transition_index=7,
        seed=12_345,
        device=torch.device("cpu"),
    )
    pieces = [
        _randoms(
            spec,
            total_paths=14,
            transition_index=7,
            seed=12_345,
            device=torch.device("cpu"),
        )
        for spec in make_work_plan(num_paths=14, path_batch_size=6)
    ]
    actual = tuple(
        torch.cat([piece[position] for piece in pieces])
        for position in range(3)
    )

    for expected_value, actual_value in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_value, expected_value)
    branch_uniform, node_uniform, normal = actual
    torch.testing.assert_close(
        branch_uniform.reshape(-1, 2).sum(dim=1),
        torch.ones(7, dtype=torch.float64),
    )
    torch.testing.assert_close(
        node_uniform.reshape(-1, 2).sum(dim=1),
        torch.ones(7, dtype=torch.float64),
    )
    torch.testing.assert_close(
        normal.reshape(-1, 2).sum(dim=1),
        torch.zeros(7, dtype=torch.float64),
    )


def test_terminal_option_returns_use_issue_spot_relative_payoffs() -> None:
    result = _option_returns(
        gross_horizon=torch.tensor((1.2, 0.8), dtype=torch.float64),
        issue_prices=torch.full((2, 2), 0.1, dtype=torch.float64),
        relative_strikes=torch.ones((2, 2), dtype=torch.float64),
        is_put=torch.tensor((False, True)),
    )

    torch.testing.assert_close(
        result,
        torch.tensor(((1.0, -1.0), (-1.0, 1.0)), dtype=torch.float64),
    )


def test_front_twenty_is_replenished_at_each_issue_date() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    business_dtes = (1.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=6,
        num_model_expiries=len(business_dtes),
        explicit_model_business_dtes=business_dtes,
        inner_relative_strikes=(0.72, 0.84, 0.94, 1.0, 1.08, 1.25),
    )
    decoder = FlexSanosDecoder(env, geometry=geometry)
    base_dlvsq = torch.tensor(
        (0.11, 0.09, 0.07, 0.08, 0.10, 0.13),
        dtype=env.fdtype,
    )
    parameters = NativeParameters(
        dlvsqs=torch.stack(
            [base_dlvsq + 0.01 * index for index in range(len(business_dtes))]
        )[None],
        forward_vols=torch.tensor(
            [[0.20 + 0.01 * index for index in range(len(business_dtes))]],
            dtype=env.fdtype,
        ),
        eta=torch.tensor(
            [[0.30 + 0.01 * index for index in range(len(business_dtes))]],
            dtype=env.fdtype,
        ),
    )
    decoded = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    surface = residual_clock_surface_from_decoded(
        decoder,
        decoded,
        residual_expiries_y=torch.tensor(business_dtes, dtype=env.fdtype) / 252.0,
    )

    first_issue = _densify_front_twenty(decoder, surface).surface
    expected = torch.arange(1.0, 21.0, dtype=env.fdtype) / 252.0
    torch.testing.assert_close(
        first_issue.residual_expiries_y[:20],
        expected,
        atol=0.0,
        rtol=0.0,
    )
    loading = smooth_residual_clock_log_loadings(
        first_issue,
        annualized_log_volatility=0.0,
    )
    kernel = build_functional_dlv_cubature(
        decoder,
        first_issue,
        raw_log_loadings=loading,
        calendar_time_increment_y=1.0 / 252.0,
    )
    rolled = roll_residual_clock_dlv(
        decoder,
        kernel,
        surface_branch_uniforms=torch.tensor((0.25,), dtype=env.fdtype),
        front_node_uniforms=torch.tensor((0.25,), dtype=env.fdtype),
        front_backbone_normals=torch.zeros(1, dtype=env.fdtype),
        elapsed_y=1.0 / 252.0,
    ).surface
    torch.testing.assert_close(
        rolled.residual_expiries_y[:19],
        expected[:19],
        atol=0.0,
        rtol=0.0,
    )

    second_issue = _densify_front_twenty(decoder, rolled).surface
    torch.testing.assert_close(
        second_issue.residual_expiries_y[:20],
        expected,
        atol=0.0,
        rtol=0.0,
    )
    assert len(second_issue.residual_expiries_y) == len(first_issue.residual_expiries_y)

    perturbed_clock = second_issue.residual_expiries_y.clone()
    perturbed_clock[14] = torch.nextafter(
        perturbed_clock[14],
        torch.full_like(perturbed_clock[14], float("inf")),
    )
    perturbed = replace(
        second_issue,
        residual_expiries_y=perturbed_clock,
    )
    redensified = _densify_front_twenty(decoder, perturbed).surface
    assert len(redensified.residual_expiries_y) == len(perturbed_clock)
    torch.testing.assert_close(
        redensified.residual_expiries_y[:20],
        expected,
        atol=5.0e-10 / 252.0,
        rtol=0.0,
    )


def test_memmaps_preserve_path_order_and_resume_contents(tmp_path) -> None:
    shapes = OutputShapes(num_paths=8, issue_days=2, policy_state_size=3)
    plan = make_work_plan(num_paths=8, path_batch_size=4)
    outputs = open_output_maps(
        tmp_path,
        shapes=shapes,
        state_dtype="float32",
        resume=False,
    )
    later = plan[1]
    _write_issue_features(
        outputs,
        spec=later,
        day=1,
        policy_state=torch.full((4, 3), 21.0),
        prices=torch.full((4, 4, 12), 22.0),
        strikes=torch.full((4, 4, 12), 23.0),
    )
    _write_spot_return(
        outputs,
        spec=later,
        day=1,
        value=torch.full((4,), 24.0),
    )
    _write_option_return(
        outputs,
        spec=later,
        entry_day=1,
        expiry_position=2,
        value=torch.full((4, 12), 25.0),
    )
    outputs.flush()
    del outputs
    gc.collect()

    resumed = open_output_maps(
        tmp_path,
        shapes=shapes,
        state_dtype="float32",
        resume=True,
    )
    assert np.all(resumed.feature_valid[:4] == 0)
    assert np.all(resumed.feature_valid[4:, 1] == 1)
    assert np.all(resumed.policy_states[4:, 1] == 21.0)
    assert np.all(resumed.option_prices[4:, 1] == 22.0)
    assert np.all(resumed.relative_strikes[4:, 1] == 23.0)
    assert np.all(np.isnan(resumed.returns[:4]))
    assert np.all(resumed.returns[4:, 1, 0] == 24.0)
    assert np.all(resumed.returns[4:, 1, 25:37] == 25.0)
    assert np.all(resumed.return_valid[4:, 1, 0] == 1)
    assert np.all(resumed.return_valid[4:, 1, 25:37] == 1)
    assert resumed.returns.shape == (8, 2, NUM_INSTRUMENTS)

    first = plan[0]
    _write_spot_return(
        resumed,
        spec=first,
        day=0,
        value=torch.arange(4, dtype=torch.float32),
    )
    resumed.flush()
    assert np.array_equal(resumed.returns[:4, 0, 0], np.arange(4))
    assert np.all(resumed.returns[4:, 1, 0] == 24.0)
    del resumed
    gc.collect()

    incompatible = OutputShapes(
        num_paths=8,
        issue_days=2,
        policy_state_size=4,
    )
    with pytest.raises(ValueError, match="resume array"):
        open_output_maps(
            tmp_path,
            shapes=incompatible,
            state_dtype="float32",
            resume=True,
        )
