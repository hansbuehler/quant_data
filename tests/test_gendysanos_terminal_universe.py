from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.grid.gendysanos.run_terminal_arbitrage_tests import (
    _returns_and_roundoff,
    _sample_indices,
)
from quantdata.dysanos.grid.gendysanos.terminal_universe import (
    build_one_time_terminal_universe,
    build_terminal_path_universe,
    expiry_step_offsets,
    sample_sanos_terminal_ratios,
)
from quantdata.dysanos.grid.gendysanos.torch_prep import (
    DysanosUniverseStreamDiagnostics,
    GenDysanosExperiment,
    _stabilize_incremental_calls,
    _terminal_call_delta,
)
from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    DEFAULT_ROUNDOFF_MULTIPLIER,
    FLOAT32_UNIT_ROUNDOFF,
)
from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig


def test_discrete_sanos_terminal_sampler_uses_one_draw_per_expiry() -> None:
    component_spots = np.array([[[0.5, 1.5], [0.8, 1.2]]], dtype=np.float64)
    weights = np.array([[[0.25, 0.75], [0.75, 0.25]]], dtype=np.float64)
    terminal, diagnostics = sample_sanos_terminal_ratios(
        component_spots=component_spots,
        component_weights=weights,
        sqrt_variances=np.zeros((1, 2), dtype=np.float64),
        smoothness=0.0,
        rng=np.random.default_rng(123),
    )

    assert terminal.shape == (1, 2)
    assert terminal[0] == pytest.approx([1.5, 0.8])
    assert diagnostics.minimum_raw_weight == 0.25
    assert diagnostics.maximum_mass_error == 0.0
    assert diagnostics.clipped_negative_weight_count == 0


def test_smoothed_sanos_samples_reproduce_call_price_within_mc_noise() -> None:
    samples = 200_000
    components = np.broadcast_to(
        np.array([0.8, 1.2], dtype=np.float64), (samples, 1, 2)
    )
    weights = np.broadcast_to(
        np.array([0.4, 0.6], dtype=np.float64), (samples, 1, 2)
    )
    sqrt_variance = np.full((samples, 1), 0.3, dtype=np.float64)
    terminal, _ = sample_sanos_terminal_ratios(
        component_spots=components,
        component_weights=weights,
        sqrt_variances=sqrt_variance,
        smoothness=0.25,
        rng=np.random.default_rng(7),
    )

    strike = 1.0
    payoff = np.maximum(terminal[:, 0] - strike, 0.0)
    standard_error = payoff.std(ddof=0) / np.sqrt(samples)

    from scipy.stats import norm

    smoothed_std = np.sqrt(0.25) * 0.3
    component_price = []
    for spot in (0.8, 1.2):
        d1 = -np.log(strike / spot) / smoothed_std + 0.5 * smoothed_std
        d2 = d1 - smoothed_std
        component_price.append(spot * norm.cdf(d1) - strike * norm.cdf(d2))
    expected = 0.4 * component_price[0] + 0.6 * component_price[1]
    assert abs(payoff.mean() - expected) <= 4.0 * standard_error


def test_terminal_universe_uses_shared_payoff_and_cash_prices() -> None:
    xstate = np.array([[np.log(2.0), 0.1]], dtype=np.float64)
    expiries = np.array([0.25], dtype=np.float64)
    strikes = np.array([[[0.75, 1.25]]], dtype=np.float64)
    calls = np.array([[[0.3, 0.1]]], dtype=np.float64)
    components = np.array([[[1.5]]], dtype=np.float64)
    weights = np.ones_like(components)
    universe, _ = build_one_time_terminal_universe(
        xstate=xstate,
        expiries=expiries,
        normalized_strikes=strikes,
        normalized_calls=calls,
        component_spots=components,
        component_weights=weights,
        sqrt_variances=np.zeros((1, 1), dtype=np.float64),
        smoothness=0.0,
        rng=np.random.default_rng(1),
    )

    assert universe["C"][0, 0, 0] == pytest.approx([0.6, 0.2])
    assert universe["dC"][0, 0, 0] == pytest.approx([0.9, 0.3])
    assert universe["dS"][0, 0] == pytest.approx(1.0)
    assert universe["spot"][0, 0] == pytest.approx(2.0)
    assert universe["log_strikes"][0, 0, 0] == pytest.approx(np.log([0.75, 1.25]))


def test_expiries_are_mapped_to_daily_path_offsets() -> None:
    expiries = np.array([2.0 / 255.0, 5.0 / 255.0, 0.5, 1.0])
    offsets, maximum_error = expiry_step_offsets(expiries, dt=1.0 / 252.0)

    assert np.array_equal(offsets, np.array([2, 5, 126, 252]))
    assert maximum_error == pytest.approx(
        np.max(np.abs(offsets / 252.0 - expiries))
    )


def test_terminal_path_universe_uses_one_spot_path_for_all_trade_expiries() -> None:
    # Two samples, two action dates, and two expiries at offsets two and four.
    spots = np.array(
        [
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
        ],
        dtype=np.float64,
    )
    xstate_path = np.stack(
        [np.log(spots), np.broadcast_to(np.arange(6), spots.shape)],
        axis=-1,
    )
    strikes = np.full((2, 2, 2, 2), 1.0, dtype=np.float64)
    strikes[..., 1] = 1.1
    calls = np.full_like(strikes, 0.1)
    universe = build_terminal_path_universe(
        xstate_path=xstate_path,
        trading_times=2,
        expiries=np.array([2.0 / 252.0, 4.0 / 252.0]),
        expiry_offsets=np.array([2, 4], dtype=np.int64),
        normalized_strikes=strikes,
        normalized_calls=calls,
        normalized_deltas=np.full_like(calls, 0.25),
    )

    # At t=0 the two expiries use S_2 and S_4 on the same sample path.
    assert universe["dC"][0, 0, 0] == pytest.approx([0.1, 0.0], abs=1.0e-6)
    assert universe["dC"][0, 0, 1] == pytest.approx([0.3, 0.2], abs=1.0e-6)
    # At t=1 the same contracts use S_3 and S_5, respectively.
    assert universe["dC"][0, 1, 0] == pytest.approx(
        [0.09, -0.02], abs=1.0e-6
    )
    assert universe["dC"][0, 1, 1] == pytest.approx(
        [0.29, 0.18], abs=1.0e-6
    )
    assert universe["dS"][0] == pytest.approx([0.1, 0.1], abs=1.0e-6)
    assert np.allclose(
        universe["expiry_dS"][0], [[0.2, 0.4], [0.2, 0.4]], atol=1.0e-6
    )
    assert np.allclose(universe["delta"], 0.25)
    assert universe["xstate"].shape == (2, 2, 2)
    assert universe["dC"].shape == (2, 2, 2, 2)


def test_one_step_mode_is_first_trade_slice_of_same_full_path() -> None:
    spots = np.array([[1.0, 1.1, 1.2, 1.3, 1.4]], dtype=np.float64)
    xstate_path = np.stack([np.log(spots), np.zeros_like(spots)], axis=-1)
    strikes = np.ones((1, 2, 1, 1), dtype=np.float64)
    calls = np.full_like(strikes, 0.1)
    path = build_terminal_path_universe(
        xstate_path=xstate_path,
        trading_times=2,
        expiries=np.array([2.0 / 252.0]),
        expiry_offsets=np.array([2], dtype=np.int64),
        normalized_strikes=strikes,
        normalized_calls=calls,
        normalized_deltas=np.full_like(calls, 0.25),
    )
    one_step = build_terminal_path_universe(
        xstate_path=xstate_path,
        trading_times=1,
        expiries=np.array([2.0 / 252.0]),
        expiry_offsets=np.array([2], dtype=np.int64),
        normalized_strikes=strikes[:, :1],
        normalized_calls=calls[:, :1],
        normalized_deltas=np.full_like(calls[:, :1], 0.25),
    )

    for name in (
        "dC",
        "dS",
        "expiry_dS",
        "C",
        "spot",
        "delta",
        "log_strikes",
        "xstate",
    ):
        assert np.array_equal(one_step[name], path[name][:, :1])


def test_weekly_mode_uses_daily_intermediate_spot_path() -> None:
    spots = np.arange(1.0, 2.1, 0.1, dtype=np.float64)[None, :]
    xstate_path = np.stack(
        [np.log(spots), np.arange(spots.shape[1], dtype=np.float64)[None, :]],
        axis=-1,
    )
    strikes = np.ones((1, 2, 1, 1), dtype=np.float64)
    calls = np.full_like(strikes, 0.1)
    weekly = build_terminal_path_universe(
        xstate_path=xstate_path,
        trading_times=2,
        trading_step_stride=5,
        expiries=np.array([2.0 / 252.0]),
        expiry_offsets=np.array([2], dtype=np.int64),
        normalized_strikes=strikes,
        normalized_calls=calls,
        normalized_deltas=np.full_like(calls, 0.25),
    )

    # Actions occur at daily indices 0 and 5; expiries use indices 2 and 7.
    assert weekly["spot"][0] == pytest.approx([1.0, 1.5], abs=1.0e-6)
    assert weekly["dS"][0] == pytest.approx([0.5, 0.5], abs=1.0e-6)
    assert weekly["expiry_dS"][0, :, 0] == pytest.approx(
        [0.2, 0.2], abs=1.0e-6
    )
    assert weekly["dC"][0, :, 0, 0] == pytest.approx(
        [0.1, 0.05], abs=1.0e-6
    )
    assert weekly["xstate"][0, :, 1] == pytest.approx([0.0, 5.0])


def test_one_day_option_return_reprices_fixed_cash_strike() -> None:
    spots = np.array([[2.0, 3.0, 6.0]], dtype=np.float64)
    xstate_path = np.stack(
        [np.log(spots), np.arange(3, dtype=np.float64)[None, :]],
        axis=-1,
    )
    calls = np.array([[[[0.2]], [[0.2]]]], dtype=np.float64)
    calls_tp1 = np.array([[[[0.1]], [[0.15]]]], dtype=np.float64)
    universe = build_terminal_path_universe(
        xstate_path=xstate_path,
        trading_times=2,
        expiries=np.array([1.0], dtype=np.float64),
        expiry_offsets=np.array([252], dtype=np.int64),
        normalized_strikes=np.ones_like(calls),
        normalized_calls=calls,
        normalized_deltas=np.full_like(calls, 0.25),
        normalized_calls_tp1=calls_tp1,
        option_return_step_offset=1,
    )

    # Cash prices are [0.4, 0.6] at t and [0.3, 0.9] one day later.
    assert universe["C"][0, :, 0, 0] == pytest.approx([0.4, 0.6])
    assert universe["dC"][0, :, 0, 0] == pytest.approx([-0.1, 0.3])
    assert universe["dS"][0] == pytest.approx([1.0, 3.0])
    assert universe["option_dS"][0, :, 0] == pytest.approx([1.0, 3.0])
    assert "expiry_dS" not in universe


def test_weekly_incremental_return_settles_early_expiry_on_exact_day() -> None:
    spot_path = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    xstate_path = np.zeros((1, spot_path.size, 2), dtype=np.float64)
    xstate_path[0, :, 0] = np.log(spot_path)
    calls = np.array([[[[0.1], [0.2]]]], dtype=np.float64)
    calls_at_week_end = np.array([[[[0.9], [0.3]]]], dtype=np.float64)

    universe = build_terminal_path_universe(
        xstate_path=xstate_path,
        trading_times=1,
        trading_step_stride=5,
        expiries=np.array([2.0, 7.0], dtype=np.float64),
        expiry_offsets=np.array([2, 7], dtype=np.int64),
        normalized_strikes=np.ones_like(calls),
        normalized_calls=calls,
        normalized_deltas=np.full_like(calls, 0.25),
        normalized_calls_tp1=calls_at_week_end,
        option_return_step_offset=5,
    )

    # The two-day contract settles at S_2=4. The seven-day contract remains
    # alive and is marked at 0.3*S_5=9.6 at the weekly endpoint.
    assert universe["dC"][0, 0, :, 0] == pytest.approx([2.9, 9.4])
    assert universe["dS"][0, 0] == pytest.approx(31.0)
    # The first stock hedge closes with its option at the two-day expiry. The
    # second option and hedge remain open until the weekly endpoint.
    assert universe["option_dS"][0, 0] == pytest.approx([3.0, 31.0])
    assert universe["option_settlement_dS"][0, 0] == pytest.approx([3.0, 31.0])


def test_weekly_incremental_call_stabilization_ignores_expired_pricer_branch() -> None:
    import torch

    calls = torch.tensor(
        [[[[float("nan")], [-1.0e-6], [0.2]]]], dtype=torch.float32
    )
    diagnostics = DysanosUniverseStreamDiagnostics()

    stable = _stabilize_incremental_calls(
        calls,
        expiry_offsets=np.array([2, 7, 30]),
        stride=5,
        diagnostics=diagnostics,
    )

    assert stable[0, 0, :, 0].tolist() == pytest.approx([0.0, 0.0, 0.2])
    assert diagnostics.incremental_call_roundoff_clip_count == 1
    with pytest.raises(FloatingPointError, match="negative beyond"):
        _stabilize_incremental_calls(
            torch.tensor([[[[0.0], [-1.0e-3], [0.2]]]], dtype=torch.float32),
            expiry_offsets=np.array([2, 7, 30]),
            stride=5,
            diagnostics=DysanosUniverseStreamDiagnostics(),
        )


def test_four_terminal_market_time_grids() -> None:
    expected = {
        "one-step": (1, 1),
        "one-step-weekly": (1, 5),
        "path": (9, 1),
        "weekly": (9, 5),
    }
    for mode, geometry in expected.items():
        experiment = GenDysanosExperiment(mode=mode, num_steps=10)
        assert (
            experiment.trading_times,
            experiment.trading_step_stride,
        ) == geometry


def test_canonical_market_flags_use_ten_trading_timestamps() -> None:
    daily_expiry = GenDysanosExperiment(
        trading_frequency="daily", return_horizon="expiry"
    )
    weekly_incremental = GenDysanosExperiment(
        trading_frequency="weekly", return_horizon="incremental"
    )

    assert daily_expiry.num_trading_times == 10
    assert daily_expiry.trading_times == 10
    assert daily_expiry.trading_step_stride == 1
    assert weekly_incremental.num_trading_times == 10
    assert weekly_incremental.trading_times == 9
    assert weekly_incremental.trading_step_stride == 5


def test_importance_sampling_accepts_five_surface_factors() -> None:
    importance = GaussianSpotImportanceConfig(
        spot_std_multiplier=8.0,
        base_probability=0.1,
        target_action_count=3,
    )

    experiment = GenDysanosExperiment(
        num_alpha=5,
        use_matrix_exp=True,
        bandwidth=1.0,
        importance_sampling=importance,
    )

    assert experiment.num_alpha == 5
    assert experiment.importance_sampling == importance


@pytest.mark.parametrize(
    ("use_matrix_exp", "bandwidth"),
    ((False, 1.0), (True, 0.5)),
)
def test_importance_sampling_requires_exact_gaussian_mode(
    use_matrix_exp: bool,
    bandwidth: float,
) -> None:
    importance = GaussianSpotImportanceConfig(
        spot_std_multiplier=8.0,
        base_probability=0.1,
        target_action_count=3,
    )

    with pytest.raises(ValueError, match="importance sampling requires"):
        GenDysanosExperiment(
            num_alpha=5,
            use_matrix_exp=use_matrix_exp,
            bandwidth=bandwidth,
            importance_sampling=importance,
        )


def test_option_return_horizon_is_validated() -> None:
    experiment = GenDysanosExperiment(option_return_horizon="one-day")
    assert experiment.option_return_horizon == "one-day"
    with pytest.raises(ValueError, match="option_return_horizon"):
        GenDysanosExperiment(option_return_horizon="invalid")


def test_terminal_arbitrage_returns_include_spot_and_delta_hedged_options() -> None:
    shape = (2, 1, 1, 2)
    universe = {
        "C": np.full(shape, 2.0, dtype=np.float32),
        "dC": np.array([[[[0.5, -0.25]]], [[[1.0, 0.5]]]], dtype=np.float32),
        "dS": np.array([[0.2], [-0.1]], dtype=np.float32),
        "expiry_dS": np.array([[[0.6]], [[-0.4]]], dtype=np.float32),
        "delta": np.full(shape, 0.5, dtype=np.float32),
        "spot": np.full((2, 1), 10.0, dtype=np.float32),
    }

    returns, roundoff = _returns_and_roundoff(universe)

    assert returns.shape == roundoff.shape == (2, 1, 3)
    assert returns[..., 0] == pytest.approx(universe["dS"])
    assert returns[0, 0, 1:] == pytest.approx([0.2, -0.55])
    assert returns[1, 0, 1:] == pytest.approx([1.2, 0.7])
    assert np.all(np.isfinite(roundoff))
    assert np.all(roundoff > 0.0)


def test_terminal_roundoff_uses_unreduced_operands_without_cash_floor() -> None:
    shape = (1, 1, 1, 1)
    universe = {
        "C": np.full(shape, 2.0e-9, dtype=np.float32),
        "dC": np.full(shape, -2.0e-9, dtype=np.float32),
        "dS": np.array([[1.0e-9]], dtype=np.float32),
        "expiry_dS": np.array([[[2.0e-9]]], dtype=np.float32),
        "delta": np.full(shape, 0.4, dtype=np.float32),
        "spot": np.array([[1.0e-8]], dtype=np.float32),
        "log_strikes": np.full(shape, np.log(1.5), dtype=np.float32),
        "option_expired": np.ones((1, 1, 1), dtype=bool),
    }

    _, roundoff = _returns_and_roundoff(universe)

    multiplier = DEFAULT_ROUNDOFF_MULTIPLIER * FLOAT32_UNIT_ROUNDOFF
    assert roundoff[0, 0, 0] == pytest.approx(
        multiplier * (1.1e-8 + 1.0e-8), rel=1.0e-6
    )
    assert roundoff[0, 0, 1] == pytest.approx(
        multiplier * (1.2e-8 + 1.5e-8 + 2.0e-9 + 0.8e-9),
        rel=1.0e-6,
    )
    assert float(roundoff.max()) < multiplier


def test_terminal_arbitrage_query_indices_use_at_most_one_state_per_path() -> None:
    indices = _sample_indices(paths=100, steps=9, maximum=40, seed=17)

    assert indices.shape == (40,)
    assert np.unique(indices // 9).size == 40
    assert np.all((indices % 9 >= 0) & (indices % 9 < 9))
    assert np.array_equal(
        indices,
        _sample_indices(paths=100, steps=9, maximum=40, seed=17),
    )


def test_terminal_arbitrage_query_indices_can_select_direct_P_component() -> None:
    eligible = np.array([1, 4, 7, 13], dtype=np.int64)
    indices = _sample_indices(
        paths=20,
        steps=3,
        maximum=10,
        seed=19,
        eligible_paths=eligible,
    )

    assert indices.shape == (eligible.size,)
    assert np.array_equal(np.sort(indices // 3), eligible)
    assert np.all((indices % 3 >= 0) & (indices % 3 < 3))


def test_terminal_arbitrage_query_indices_reject_invalid_eligible_paths() -> None:
    with pytest.raises(ValueError, match="eligible query paths"):
        _sample_indices(
            paths=20,
            steps=3,
            maximum=10,
            seed=19,
            eligible_paths=np.array([1, 1], dtype=np.int64),
        )


def test_terminal_call_delta_uses_scale_family_spot_derivative() -> None:
    import torch

    class Pricer:
        smoothness = 0.0
        xstrikes = torch.tensor([[[0.8, 1.2]]])
        xdensities = torch.tensor([[[0.4, 0.6]]])
        sqrtVars = torch.tensor([[0.2]])

    strikes = torch.tensor([[[1.0, 1.3]]])
    delta = _terminal_call_delta(Pricer(), strikes)

    # At K=1 only the 1.2 component finishes in the money. At K=1.3 neither
    # does. The derivative with respect to the current scale S is x*1{x>K}.
    assert torch.allclose(delta, torch.tensor([[[0.72, 0.0]]]))


def test_terminal_call_delta_uses_smoothed_black_scholes_derivative() -> None:
    import torch

    class Pricer:
        smoothness = 0.25
        xstrikes = torch.tensor([[[1.0]]], dtype=torch.float64)
        xdensities = torch.tensor([[[1.0]]], dtype=torch.float64)
        sqrtVars = torch.tensor([[0.2]], dtype=torch.float64)

    delta = _terminal_call_delta(
        Pricer(), torch.tensor([[[1.0]]], dtype=torch.float64)
    )

    # sqrt(smoothness)*sqrtVar=0.1, so ATM d1=0.05 and delta=N(0.05).
    expected = torch.special.ndtr(torch.tensor(0.05, dtype=torch.float64))
    assert torch.allclose(delta, expected.reshape(1, 1, 1))
