from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.dysanos.grid.rndysanos.local_tail import (
    LocalTailConfig,
    LocalTailProgress,
    _append_prefix_sample,
    local_neighbor_statistics,
    normalize_state,
    prepare_delta_hedged_local_data,
    reduced_option_state,
    shared_instrument_metric_diagnostics,
    state_normalization,
    valuate_local_tail,
)


def _universe(*, samples: int = 6, steps: int = 2) -> dict[str, np.ndarray]:
    expiries, strikes, state_variables = 2, 2, 3
    shape = (samples, steps, expiries, strikes)
    rng = np.random.default_rng(83)
    xstate = rng.normal(size=(samples, steps, state_variables)).astype(np.float32)
    strike_shift = np.arange(expiries * strikes, dtype=np.float32).reshape(
        1, 1, expiries, strikes
    )
    log_strikes = xstate[..., :1, None] + strike_shift
    return {
        "C": np.full(shape, 8.0, dtype=np.float32),
        "dC": rng.normal(0.0, 0.2, shape).astype(np.float32),
        "dS": rng.normal(0.0, 0.1, (samples, steps)).astype(np.float32),
        "delta": np.full(shape, 0.2, dtype=np.float32),
        "expiries": np.array([[[0.25, 0.75]]], dtype=np.float32),
        "log_strikes": log_strikes.astype(np.float32),
        "spot": np.full((samples, steps), 10.0, dtype=np.float32),
        "xstate": xstate,
    }


def test_local_tail_config_uses_local_effective_sample_size() -> None:
    config = LocalTailConfig(
        state_mode="raw", neighbors=100, tail_ranks=(1, 2, 10, 20, 50)
    )

    assert config.local_tail_probabilities == pytest.approx(
        (0.01, 0.02, 0.10, 0.20, 0.50)
    )
    assert config.zero_event_upper_probability_95 == pytest.approx(
        1.0 - 0.05 ** 0.01
    )
    with pytest.raises(ValueError, match="half"):
        LocalTailConfig(state_mode="raw", neighbors=50, tail_ranks=(30,))
    with pytest.raises(ValueError, match="signal thresholds"):
        LocalTailConfig(
            state_mode="raw",
            neighbors=100,
            tail_ranks=(1,),
            signal_thresholds=(3.0, 2.0),
        )


def test_radius_diagnostic_reservoir_stops_cleanly_at_capacity() -> None:
    sample = np.full(3, -1.0, dtype=np.float32)

    count = _append_prefix_sample(sample, 0, np.array([1.0, 2.0], dtype=np.float32))
    count = _append_prefix_sample(
        sample, count, np.array([3.0, 4.0, 5.0], dtype=np.float32)
    )
    count = _append_prefix_sample(sample, count, np.array([6.0], dtype=np.float32))

    assert count == 3
    assert np.array_equal(sample, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_reduced_state_uses_only_own_instrument_fields() -> None:
    data, geometry = prepare_delta_hedged_local_data(_universe())
    raw = reduced_option_state(data, geometry=geometry, mode="raw")
    instrument = reduced_option_state(
        data, geometry=geometry, mode="instrument", instrument=3
    )

    assert raw.shape == (6, 2, 3)
    assert instrument.shape == (6, 2, 5)
    assert np.array_equal(instrument[..., :3], raw)
    assert np.array_equal(instrument[..., 3], data["log_strikes"][..., 3])
    assert np.all(instrument[..., 4] == data["expiries"][1])
    assert not np.any(instrument[..., 3] == data["log_strikes"][..., 0])


def test_instrument_metric_detects_shift_equivalent_strikes_and_constant_expiry() -> None:
    data, geometry = prepare_delta_hedged_local_data(_universe(samples=8))

    diagnostics = shared_instrument_metric_diagnostics(
        data,
        training_paths=4,
        geometry=geometry,
        scale_floor=1.0e-6,
    )

    assert diagnostics["shared_metric"]
    assert diagnostics["max_abs_standardized_log_strike_difference"] < 1.0e-5
    instrument = reduced_option_state(
        data, geometry=geometry, mode="instrument", instrument=0
    )
    mean, scale, constant = state_normalization(
        instrument[:4], scale_floor=1.0e-6
    )
    normalized = normalize_state(instrument, mean=mean, scale=scale)
    assert constant[-1]
    assert np.all(normalized[..., -1] == 0.0)


def test_local_neighbor_statistics_has_correct_signed_quantiles_and_cvar() -> None:
    returns = torch.tensor([[[-3.0], [-1.0], [2.0], [4.0]]])
    roundoff = torch.full_like(returns, 0.5)

    output = local_neighbor_statistics(
        returns, roundoff, tail_ranks=(1, 2)
    )

    assert output["positive_mean"].item() == pytest.approx(1.25)
    assert output["negative_mean"].item() == pytest.approx(0.75)
    assert output["positive_probability"].item() == pytest.approx(0.5)
    assert output["negative_probability"].item() == pytest.approx(0.5)
    assert output["mean"].item() == pytest.approx(0.5)
    assert output["second_moment"].item() == pytest.approx(7.5)
    assert output["variance"].item() == pytest.approx(7.25)
    assert torch.equal(output["positive_count"], torch.tensor([[2]]))
    assert torch.equal(output["negative_count"], torch.tensor([[2]]))
    assert torch.equal(
        output["quantile"],
        torch.tensor([[[[-3.0], [-1.0]], [[-4.0], [-2.0]]]]),
    )
    assert torch.equal(
        output["lower_cvar"],
        torch.tensor([[[[-3.0], [-2.0]], [[-4.0], [-3.0]]]]),
    )


def test_local_neighbor_statistics_uses_likelihood_weights_for_moments_and_tails():
    returns = torch.tensor([[[-10.0], [1.0], [2.0]]])
    roundoff = torch.zeros_like(returns)
    log_weights = torch.log(torch.tensor([[0.01, 0.49, 0.50]]))

    output = local_neighbor_statistics(
        returns,
        roundoff,
        tail_ranks=(1,),
        neighbor_log_weights=log_weights,
    )

    assert output["mean"].item() == pytest.approx(1.39)
    assert output["positive_probability"].item() == pytest.approx(0.99)
    assert output["negative_probability"].item() == pytest.approx(0.01)
    assert output["negative_count"].item() == 1
    assert output["quantile"][0, 0, 0, 0].item() == pytest.approx(1.0)
    assert output["lower_cvar"][0, 0, 0, 0].item() == pytest.approx(0.67)
    assert output["effective_sample_size"].item() == pytest.approx(
        1.0 / (0.01**2 + 0.49**2 + 0.50**2)
    )


def test_state_normalization_uses_independent_path_likelihood_weights() -> None:
    state = np.array([[[0.0]], [[10.0]]], dtype=np.float32)

    mean, scale, constant = state_normalization(
        state,
        scale_floor=1.0e-6,
        path_weights=np.array([9.0, 1.0]),
    )

    assert mean == pytest.approx([1.0])
    assert scale == pytest.approx([3.0])
    assert not constant[0]


def test_exact_local_valuation_finds_opposite_one_sided_instruments() -> None:
    config = LocalTailConfig(
        state_mode="raw",
        neighbors=2,
        tail_ranks=(1,),
        query_batch_size=1,
        instrument_batch_size=1,
    )
    reference_state = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    reference_returns = torch.tensor(
        [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]
    )
    reference_roundoff = torch.full_like(reference_returns, 0.01)
    query_state = torch.tensor([[1.1]])
    query_returns = torch.tensor([[1.5, -1.5]])
    query_roundoff = torch.full_like(query_returns, 0.01)

    output = valuate_local_tail(
        reference_state=reference_state,
        reference_returns=reference_returns,
        reference_roundoff=reference_roundoff,
        query_state=query_state,
        query_returns=query_returns,
        query_roundoff=query_roundoff,
        config=config,
        split="validation",
        leave_one_out=False,
    )

    assert output["zero_adverse"]["count_by_direction"] == [1, 1]
    assert output["zero_adverse"]["states_with_any_by_direction"] == [1, 1]
    assert output["zero_adverse"]["realized_counter_sign_count_by_direction"] == [0, 0]
    assert output["quantile_positive"]["count_by_direction_level"] == [[1], [1]]
    assert output["lower_cvar_positive"]["count_by_direction_level"] == [[1], [1]]
    assert output["conditional_moment_signal"]["thresholds"] == [2.0, 3.0]
    assert output["conditional_moment_signal"]["count_by_direction_level"] == [
        [1, 1],
        [1, 1],
    ]
    assert output["conditional_moment_signal"][
        "realized_counter_sign_count_by_direction_level"
    ] == [[0, 0], [0, 0]]
    assert output["conditional_moment_signal"][
        "maximum_abs_mean_over_sigma"
    ] == pytest.approx(5.0)
    policy = output["zero_adverse_single_instrument_policy"]
    assert policy["trading_states"] == 1
    assert policy["long_trades"] == 1
    assert policy["short_trades"] == 0
    assert policy["realized_gains_beyond_roundoff"] == 1
    assert policy["realized_losses_beyond_roundoff"] == 0
    assert policy["pnl_mean_when_trading"] == pytest.approx(1.5)
    assert output["quantile_positive"]["both_directions_positive_count_by_level"] == [0]
    assert output["neighbor_radius"]["maximum"] == pytest.approx(0.9)


def test_local_valuation_progress_reports_disjoint_candidate_state_counts() -> None:
    config = LocalTailConfig(
        state_mode="raw",
        neighbors=2,
        tail_ranks=(1,),
        query_batch_size=1,
        instrument_batch_size=2,
    )
    reference_state = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
    reference_returns = torch.tensor(
        [[1.0, 1.0], [2.0, -1.0], [1.0, -2.0], [-1.0, -3.0]]
    )
    reference_roundoff = torch.full_like(reference_returns, 0.01)
    progress: list[LocalTailProgress] = []

    output = valuate_local_tail(
        reference_state=reference_state,
        reference_returns=reference_returns,
        reference_roundoff=reference_roundoff,
        query_state=torch.tensor([[0.1], [10.1]]),
        query_returns=torch.tensor([[1.5, 0.5], [0.5, -2.5]]),
        query_roundoff=torch.full((2, 2), 0.01),
        config=config,
        split="validation",
        leave_one_out=False,
        progress=progress.append,
        progress_interval_seconds=1.0e9,
    )

    assert len(progress) == 2
    assert progress[0].processed_states == 1
    assert progress[0].long_only_candidate_states == 1
    assert progress[0].short_only_candidate_states == 0
    assert progress[-1].processed_states == 2
    assert progress[-1].states_with_any_long_candidate == 1
    assert progress[-1].states_with_any_short_candidate == 1
    assert progress[-1].long_only_candidate_states == 1
    assert progress[-1].short_only_candidate_states == 1
    assert progress[-1].both_direction_candidate_states == 0
    assert output["zero_adverse"]["states_by_direction_exclusivity"] == {
        "long_only": 1,
        "short_only": 1,
        "both_directions": 0,
        "neither_direction": 0,
    }


def test_training_local_valuation_is_leave_one_out() -> None:
    config = LocalTailConfig(
        state_mode="raw",
        neighbors=2,
        tail_ranks=(1,),
        query_batch_size=2,
        instrument_batch_size=1,
    )
    state = torch.tensor([[0.0], [1.0], [2.0]])
    returns = torch.tensor([[-100.0], [1.0], [2.0]])
    roundoff = torch.full_like(returns, 0.01)

    output = valuate_local_tail(
        reference_state=state,
        reference_returns=returns,
        reference_roundoff=roundoff,
        query_state=state,
        query_returns=returns,
        query_roundoff=roundoff,
        config=config,
        split="training",
        leave_one_out=True,
    )

    assert output["leave_one_out"]
    assert output["queries"] == 3
    assert output["zero_adverse"]["count_by_direction"] == [1, 0]
    assert output["zero_adverse"]["realized_counter_sign_count_by_direction"] == [1, 0]


def test_sampled_training_queries_use_explicit_leave_one_out_indices() -> None:
    config = LocalTailConfig(
        state_mode="raw",
        neighbors=2,
        tail_ranks=(1,),
        query_batch_size=1,
        instrument_batch_size=1,
    )
    reference_state = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    reference_returns = torch.tensor([[-4.0], [100.0], [2.0], [3.0]])
    reference_roundoff = torch.full_like(reference_returns, 0.01)

    output = valuate_local_tail(
        reference_state=reference_state,
        reference_returns=reference_returns,
        reference_roundoff=reference_roundoff,
        query_state=reference_state[1:2],
        query_returns=reference_returns[1:2],
        query_roundoff=reference_roundoff[1:2],
        config=config,
        split="training",
        leave_one_out=True,
        leave_one_out_reference_indices=torch.tensor([1]),
    )

    assert output["queries"] == 1
    assert output["zero_adverse"]["count_by_direction"] == [0, 0]
    assert output["conditional_moment_signal"]["zero_sigma_count"] == 0
    assert output["zero_adverse_single_instrument_policy"]["trading_states"] == 0


def test_zero_adverse_policy_records_held_out_counter_sign() -> None:
    config = LocalTailConfig(
        state_mode="raw",
        neighbors=2,
        tail_ranks=(1,),
        query_batch_size=1,
        instrument_batch_size=1,
    )
    reference_state = torch.tensor([[0.0], [1.0], [2.0]])
    reference_returns = torch.tensor([[1.0], [2.0], [3.0]])
    reference_roundoff = torch.full_like(reference_returns, 0.01)

    output = valuate_local_tail(
        reference_state=reference_state,
        reference_returns=reference_returns,
        reference_roundoff=reference_roundoff,
        query_state=torch.tensor([[0.5]]),
        query_returns=torch.tensor([[-0.25]]),
        query_roundoff=torch.tensor([[0.01]]),
        config=config,
        split="validation",
        leave_one_out=False,
    )

    policy = output["zero_adverse_single_instrument_policy"]
    assert policy["trading_states"] == 1
    assert policy["realized_losses_beyond_roundoff"] == 1
    assert not policy["empirical_nonnegative_within_roundoff"]
    assert policy["pnl_mean_when_trading"] == pytest.approx(-0.25)


def test_instrument_search_reranks_raw_candidates_and_certifies_exactness() -> None:
    config = LocalTailConfig(
        state_mode="instrument",
        neighbors=2,
        instrument_candidate_neighbors=4,
        tail_ranks=(1,),
        query_batch_size=1,
        instrument_batch_size=1,
    )
    reference_state = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    reference_feature = torch.tensor(
        [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    )
    reference_returns = torch.tensor(
        [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]
    )
    reference_roundoff = torch.full_like(reference_returns, 0.01)

    output = valuate_local_tail(
        reference_state=reference_state,
        reference_returns=reference_returns,
        reference_roundoff=reference_roundoff,
        query_state=torch.tensor([[1.5]]),
        query_returns=torch.tensor([[2.5, -2.5]]),
        query_roundoff=torch.full((1, 2), 0.01),
        reference_instrument_feature=reference_feature,
        query_instrument_feature=torch.tensor([[1.5, 1.5]]),
        config=config,
        split="validation",
        leave_one_out=False,
    )

    assert output["neighbor_search"]["candidate_neighbors"] == 4
    assert output["neighbor_search"]["certified_exact_fraction"] == pytest.approx(1.0)
    assert output["zero_adverse"]["count_by_direction"] == [1, 1]
