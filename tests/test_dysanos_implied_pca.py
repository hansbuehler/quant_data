from __future__ import annotations

import numpy as np
import pytest
from cdxcore.bs import bs

from quantdata.dysanos.grid.gendysanos.terminal_universe import (
    build_terminal_path_universe,
)
from quantdata.dysanos.implied.cache import PCAUniverseBlockCache
from quantdata.dysanos.implied.comparison import write_aggregate_comparison
from quantdata.dysanos.implied.decoder import PCATotalVarianceDecoder
from quantdata.dysanos.implied.experiment import ImpliedPCAExperiment
from quantdata.dysanos.implied.pca import (
    LogIVPCA,
    fit_log_iv_pca,
    validate_pca_market,
)
from quantdata.dysanos.implied.return_diagnostics import (
    ReturnDiagnosticsAccumulator,
)
from quantdata.dysanos.implied.run_experiment import _slice_return_universe
from quantdata.dysanos.implied.static_arbitrage import (
    StaticArbitrageAccumulator,
    TradedStrikeStaticArbitrageAccumulator,
    check_static_arbitrage,
    check_traded_strike_static_arbitrage,
)
from quantdata.dysanos.implied.universe import PCAUniverseBlock


def _flat_pca(volatility: float = 0.2) -> LogIVPCA:
    expiries = np.array([0.1, 0.4], dtype=np.float64)
    nms = np.broadcast_to(
        np.array([-1.5, 0.0, 1.0], dtype=np.float64), (2, 3)
    ).copy()
    component = np.zeros((1, nms.size), dtype=np.float64)
    component[0, 0] = 1.0
    return LogIVPCA(
        mean_log_iv=np.full(nms.size, np.log(volatility)),
        components=component,
        explained_variance=np.array([1.0]),
        explained_variance_ratio=np.array([1.0]),
        cumulative_explained_variance_ratio=np.array([1.0]),
        expiries=expiries,
        nms=nms,
    )


def _market(vols: np.ndarray) -> dict[str, np.ndarray]:
    D, T, N = vols.shape
    expiries = np.broadcast_to(
        np.linspace(0.1, 0.5, T), (D, T)
    ).copy()
    nms = np.broadcast_to(
        np.linspace(-1.5, 1.0, N), (D, T, N)
    ).copy()
    strikes = np.empty((D, T, N + 2), dtype=np.float64)
    strikes[..., 1:-1] = np.exp(0.1 * nms)
    strikes[..., 0] = strikes[..., 1] * 0.5
    strikes[..., -1] = strikes[..., -2] * 1.5
    return {
        "dates": np.arange(D),
        "time_y": np.arange(D, dtype=np.float64) / 252.0,
        "mid_vols": vols,
        "nms": nms,
        "expiries": expiries,
        "xstrikes": strikes,
        "spot_ref": np.linspace(4_000.0, 4_100.0, D),
        "spot_ret": np.zeros(D, dtype=np.float64),
    }


def test_market_validation_uses_positive_mid_vols_and_common_grid() -> None:
    market = _market(np.full((5, 2, 4), 0.2, dtype=np.float64))
    market["fit_vols"] = np.full((5, 2, 4), np.nan)

    validated = validate_pca_market(market)

    assert validated.mid_vols.shape == (5, 2, 4)
    changed = dict(market)
    changed["nms"] = market["nms"].copy()
    changed["nms"][2, 0, 0] += 0.01
    with pytest.raises(ValueError, match="common NMS grid"):
        validate_pca_market(changed)


def test_log_iv_level_pca_reconstructs_low_rank_surfaces_and_fixes_signs() -> None:
    rng = np.random.default_rng(17)
    D, T, N = 80, 2, 4
    factors = rng.normal(size=(D, 2))
    loadings = rng.normal(size=(2, T * N))
    log_vols = np.log(0.22) + factors @ loadings * 0.02
    vols = np.exp(log_vols.reshape(D, T, N))
    expiries = np.array([0.1, 0.5])
    nms = np.broadcast_to(np.linspace(-1.5, 1.0, N), (T, N)).copy()

    model, diagnostics = fit_log_iv_pca(
        vols, expiries=expiries, nms=nms, num_components=2
    )

    assert model.decode(diagnostics.scores) == pytest.approx(vols, abs=1.0e-12)
    pivots = np.argmax(np.abs(model.components), axis=1)
    assert np.all(model.components[np.arange(2), pivots] > 0.0)
    assert model.components @ model.components.T == pytest.approx(np.eye(2))
    assert diagnostics.reconstruction["log_iv_rmse"] < 1.0e-12
    assert np.all(model.decode(diagnostics.scores) > 0.0)


def test_total_variance_interpolation_and_nms_round_trip() -> None:
    decoder = PCATotalVarianceDecoder(_flat_pca())
    scores = np.array([0.0])
    maturities = np.array([0.05, 0.25])
    nms = np.array([-0.5, 0.75])

    total_variance, hits = decoder.total_variance(scores, maturities, nms)
    strikes = decoder.relative_strikes_from_nms(scores, maturities, nms)
    reconstructed_nms = decoder.nms_from_relative_strikes(
        scores, maturities, strikes
    )

    assert total_variance == pytest.approx(0.2**2 * maturities)
    assert reconstructed_nms == pytest.approx(nms, abs=1.0e-12)
    assert hits.total_hits == 0
    _, wing_hits = decoder.total_variance(
        scores, np.array([0.1, 0.1]), np.array([-2.0, 2.0])
    )
    assert wing_hits.left_hits == 1
    assert wing_hits.right_hits == 1


def test_fixed_surface_cash_delta_matches_black_scholes_on_flat_surface() -> None:
    decoder = PCATotalVarianceDecoder(_flat_pca())
    scores = np.array([0.0])
    spot = np.array([100.0, 100.0, 100.0])
    strike = np.array([90.0, 100.0, 110.0])
    maturity = np.full(3, 0.25)

    delta, _ = decoder.cash_deltas(scores, spot, maturity, strike)
    expected = bs(
        k=strike / spot,
        vol=0.2,
        sqrtT=np.sqrt(maturity),
        what=bs.DELTA,
    )

    assert delta == pytest.approx(expected, abs=2.0e-8)


def test_zero_maturity_prices_same_fixed_contract_at_intrinsic() -> None:
    decoder = PCATotalVarianceDecoder(_flat_pca())
    calls, hits = decoder.normalized_call_prices(
        np.array([0.0]),
        np.zeros(3),
        np.array([0.8, 1.0, 1.2]),
    )

    assert calls == pytest.approx([0.2, 0.0, 0.0])
    assert hits.queries == 0


def test_static_checker_passes_bs_and_detects_irregular_butterfly() -> None:
    strikes = np.array([0.7, 0.86, 1.0, 1.27, 1.6])
    expiries = np.array([0.1, 0.5])
    calls = np.stack(
        [
            bs(k=strikes, vol=0.2, sqrtT=np.sqrt(expiry), what=bs.PRICE)
            for expiry in expiries
        ]
    )
    clean = check_static_arbitrage(
        calls[None], strikes=strikes, expiries=expiries
    )
    assert clean["primary"]["aggregate"]["violating_grid_portfolios"] == 0

    bad = calls[None].copy()
    bad[:, 0, 2] += 0.08
    report = check_static_arbitrage(
        bad, strikes=strikes, expiries=expiries
    )
    assert report["primary"]["strike_convexity"]["violating_grid_portfolios"] > 0
    assert report["primary"]["calendar_monotonicity"]["violating_grid_portfolios"] > 0


def test_traded_strike_checker_uses_each_surface_irregular_grid() -> None:
    samples, actions, expiry_count, strike_count = 4, 2, 2, 5
    expiries = np.array([0.1, 0.5])
    base_strikes = np.array([0.7, 0.86, 1.0, 1.27, 1.6])
    scales = np.linspace(0.97, 1.03, samples * actions * expiry_count).reshape(
        samples, actions, expiry_count, 1
    )
    strikes = scales * base_strikes
    calls = np.empty_like(strikes)
    for expiry in range(expiry_count):
        calls[:, :, expiry] = bs(
            k=strikes[:, :, expiry],
            vol=0.2,
            sqrtT=np.sqrt(expiries[expiry]),
            what=bs.PRICE,
        )

    clean = check_traded_strike_static_arbitrage(
        calls, strikes=strikes
    )
    assert clean["geometry"] == {
        "trading_times": actions,
        "expiries": expiry_count,
        "traded_strikes_per_expiry": strike_count,
    }
    assert clean["primary"]["aggregate"]["violating_grid_portfolios"] == 0

    bad = calls.copy()
    bad[0, 0, 0, 3] = bad[0, 0, 0, 2] + 0.02
    width = strikes[1, 0, 0, 1] - strikes[1, 0, 0, 0]
    bad[1, 0, 0, 0] = bad[1, 0, 0, 1] + width + 0.02
    bad[2, 0, 0, 2] += 0.08
    report = check_traded_strike_static_arbitrage(bad, strikes=strikes)

    for constraint in (
        "call_monotonicity",
        "vertical_spread_slope",
        "strike_convexity",
    ):
        assert report["primary"][constraint]["violating_grid_portfolios"] > 0


def test_traded_strike_streaming_matches_one_block_with_weights() -> None:
    strikes = np.broadcast_to(
        np.array([0.7, 0.86, 1.0, 1.27, 1.6]), (4, 2, 1, 5)
    ).copy()
    calls = bs(k=strikes, vol=0.2, sqrtT=np.sqrt(0.25), what=bs.PRICE)
    calls[1, 0, 0, 2] += 0.08
    calls[3, 1, 0, 3] = calls[3, 1, 0, 2] + 0.02
    log_weight = np.log(np.array([0.5, 1.0, 1.5, 2.0]))
    full = TradedStrikeStaticArbitrageAccumulator(seed=9)
    streamed = TradedStrikeStaticArbitrageAccumulator(seed=9)
    full.update(calls, strikes=strikes, log_path_weights=log_weight)
    streamed.update(
        calls[:2],
        strikes=strikes[:2],
        log_path_weights=log_weight[:2],
    )
    streamed.update(
        calls[2:],
        strikes=strikes[2:],
        log_path_weights=log_weight[2:],
        path_offset=2,
    )

    full_report = full.report()["primary"]
    streamed_report = streamed.report()["primary"]
    for constraint in (*(
        "call_monotonicity",
        "vertical_spread_slope",
        "strike_convexity",
    ), "aggregate"):
        for name in (
            "violating_surfaces",
            "violating_surface_fraction",
            "paths_with_violation",
            "path_fraction",
            "violating_grid_portfolios",
            "physical_weighted_violating_portfolio_fraction",
        ):
            assert streamed_report[constraint][name] == pytest.approx(
                full_report[constraint][name]
            )


def test_shared_terminal_builder_can_preserve_unbounded_pca_delta() -> None:
    xstate = np.zeros((1, 3, 2), dtype=np.float64)
    strikes = np.ones((1, 1, 1, 1), dtype=np.float64)
    calls = np.full_like(strikes, 0.1)
    delta = np.full_like(strikes, 1.2)
    kwargs = dict(
        xstate_path=xstate,
        trading_times=1,
        expiries=np.array([2.0 / 252.0]),
        expiry_offsets=np.array([2], dtype=np.int64),
        normalized_strikes=strikes,
        normalized_calls=calls,
        normalized_deltas=delta,
    )

    with pytest.raises(ValueError, match="deltas must be in"):
        build_terminal_path_universe(**kwargs)
    universe = build_terminal_path_universe(
        **kwargs, require_bounded_deltas=False
    )
    assert universe["delta"].item() == pytest.approx(1.2)


def test_experiment_cache_identity_uses_more_than_path_count() -> None:
    first = ImpliedPCAExperiment(num_samples=100_000, seed=1)
    second = ImpliedPCAExperiment(num_samples=100_000, seed=2)
    third = ImpliedPCAExperiment(
        num_samples=100_000, seed=1, mode="weekly"
    )

    assert len({first.filename, second.filename, third.filename}) == 3


def test_pca_experiment_uses_canonical_ten_timestamp_geometry() -> None:
    daily_expiry = ImpliedPCAExperiment(num_samples=100_000)
    weekly_incremental = ImpliedPCAExperiment(
        num_samples=100_000,
        trading_frequency="weekly",
        return_horizon="incremental",
    )

    assert daily_expiry.trading_times == 10
    assert daily_expiry.trading_step_stride == 1
    assert daily_expiry.reference_paths == 80_000
    assert daily_expiry.audit_paths == 20_000
    assert weekly_incremental.trading_times == 9
    assert weekly_incremental.trading_step_stride == 5


def test_chunked_universe_cache_round_trips_versioned_blocks(tmp_path) -> None:
    experiment = ImpliedPCAExperiment(num_samples=2, importance_sampling=None)
    cache = PCAUniverseBlockCache(tmp_path, experiment=experiment)
    cache.begin()
    cache.write(
        PCAUniverseBlock(
            start=0,
            stop=2,
            universe={
                "xstate": np.arange(6, dtype=np.float32).reshape(2, 1, 3),
                "expiries": np.array([[[0.1]]], dtype=np.float32),
            },
        )
    )
    cache.complete(diagnostics={"test": "complete"})

    assert cache.is_complete
    block = next(cache.iter_blocks())
    assert block.start == 0 and block.stop == 2
    assert block.universe["xstate"] == pytest.approx(
        np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    )
    assert cache.manifest()["configuration"]["num_samples"] == 2


def test_return_diagnostics_apply_path_likelihood_once() -> None:
    S, A, T, N = 2, 1, 1, 1
    universe = {
        "dC": np.array([0.1, 0.3], dtype=np.float32).reshape(S, A, T, N),
        "dS": np.zeros((S, A), dtype=np.float32),
        "expiry_dS": np.zeros((S, A, T), dtype=np.float32),
        "C": np.full((S, A, T, N), 0.1, dtype=np.float32),
        "spot": np.ones((S, A), dtype=np.float32),
        "delta": np.zeros((S, A, T, N), dtype=np.float32),
        "log_strikes": np.zeros((S, A, T, N), dtype=np.float32),
        "log_path_weight": np.log(np.array([1.0, 2.0])),
    }
    accumulator = ReturnDiagnosticsAccumulator(expiries=T, strikes=N)
    accumulator.update(universe)
    report = accumulator.report()

    # Instrument zero is spot; option instrument one has weighted mean 7/30.
    assert report["per_instrument"]["weighted_mean"][1] == pytest.approx(
        (0.1 + 2.0 * 0.3) / 3.0
    )
    assert report["path_weights"]["mean_p_over_q"] == pytest.approx(1.5)
    assert report["transaction_costs"]["spot_long"]["negative"] == [2]
    assert report["transaction_costs"]["spot_short"]["negative"] == [2]


def test_full_smile_delta_matches_finer_fixed_surface_difference() -> None:
    pca = _flat_pca()
    component = np.linspace(-1.0, 1.0, pca.nms.size)
    component /= np.linalg.norm(component)
    smile_pca = LogIVPCA(
        mean_log_iv=pca.mean_log_iv,
        components=component[None],
        explained_variance=np.array([0.04]),
        explained_variance_ratio=np.array([1.0]),
        cumulative_explained_variance_ratio=np.array([1.0]),
        expiries=pca.expiries,
        nms=pca.nms,
    )
    decoder = PCATotalVarianceDecoder(smile_pca)
    scores = np.array([0.7])
    spot = np.array([100.0])
    strike = np.array([110.0])
    maturity = np.array([0.25])

    delta, _ = decoder.cash_deltas(scores, spot, maturity, strike)
    reference, _ = decoder.cash_deltas(
        scores, spot, maturity, strike, relative_step=1.0e-6
    )
    volatility, _ = decoder.implied_volatility(
        scores, maturity, strike / spot
    )
    frozen_vol_delta = bs(
        k=strike / spot,
        vol=volatility,
        sqrtT=np.sqrt(maturity),
        what=bs.DELTA,
    )

    assert delta == pytest.approx(reference, abs=2.0e-7)
    assert abs(float(delta.item() - frozen_vol_delta.item())) > 1.0e-4


@pytest.mark.parametrize(
    "constraint",
    ("call_monotonicity", "vertical_spread_slope", "calendar_monotonicity"),
)
def test_static_checker_detects_deliberate_constraint_violations(
    constraint: str,
) -> None:
    strikes = np.array([0.7, 0.85, 1.0, 1.2, 1.5])
    expiries = np.array([0.1, 0.5])
    calls = np.stack(
        [
            bs(k=strikes, vol=0.2, sqrtT=np.sqrt(expiry), what=bs.PRICE)
            for expiry in expiries
        ]
    )[None]
    if constraint == "call_monotonicity":
        calls[:, 0, 3] = calls[:, 0, 2] + 0.02
    elif constraint == "vertical_spread_slope":
        calls[:, 0, 1] = calls[:, 0, 2] + (strikes[2] - strikes[1]) + 0.02
    else:
        calls[:, 1, 2] = calls[:, 0, 2] - 0.02

    report = check_static_arbitrage(
        calls, strikes=strikes, expiries=expiries
    )

    assert report["primary"][constraint]["violating_grid_portfolios"] > 0


def test_streaming_static_diagnostics_match_single_block_with_weights() -> None:
    strikes = np.array([0.7, 0.9, 1.0, 1.2, 1.5])
    expiries = np.array([0.1, 0.5])
    base = np.stack(
        [
            bs(k=strikes, vol=0.2, sqrtT=np.sqrt(expiry), what=bs.PRICE)
            for expiry in expiries
        ]
    )
    calls = np.broadcast_to(base, (4, *base.shape)).copy()
    calls[1, 0, 2] += 0.05
    calls[3, 1, 2] -= 0.03
    log_weight = np.log(np.array([0.5, 1.0, 1.5, 2.0]))
    full = StaticArbitrageAccumulator(
        strikes=strikes, expiries=expiries, reservoir_size=10_000, seed=7
    )
    streamed = StaticArbitrageAccumulator(
        strikes=strikes, expiries=expiries, reservoir_size=10_000, seed=7
    )
    full.update(calls, log_path_weights=log_weight)
    streamed.update(calls[:2], log_path_weights=log_weight[:2])
    streamed.update(calls[2:], log_path_weights=log_weight[2:])

    full_report = full.report()["primary"]["aggregate"]
    streamed_report = streamed.report()["primary"]["aggregate"]
    for name in (
        "violating_surfaces",
        "violating_surface_fraction",
        "paths_with_violation",
        "path_fraction",
        "violating_grid_portfolios",
    ):
        assert streamed_report[name] == pytest.approx(full_report[name])
    assert streamed_report["conditional_magnitude"] == pytest.approx(
        full_report["conditional_magnitude"]
    )

    extreme = StaticArbitrageAccumulator(
        strikes=strikes, expiries=expiries, reservoir_size=10_000, seed=7
    )
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        extreme.update(
            calls,
            log_path_weights=np.array([-1_000.0, -500.0, -100.0, 0.0]),
        )
    assert extreme.report()["primary"]["aggregate"][
        "violating_grid_portfolios"
    ] == full_report["violating_grid_portfolios"]


def test_streaming_return_diagnostics_and_path_slice_match_single_block() -> None:
    samples, actions, expiries, strikes = 4, 2, 1, 1
    universe = {
        "dC": np.arange(8, dtype=np.float32).reshape(4, 2, 1, 1) / 10.0,
        "dS": np.zeros((samples, actions), dtype=np.float32),
        "expiry_dS": np.zeros((samples, actions, expiries), dtype=np.float32),
        "C": np.full((samples, actions, expiries, strikes), 0.1, dtype=np.float32),
        "spot": np.ones((samples, actions), dtype=np.float32),
        "delta": np.zeros((samples, actions, expiries, strikes), dtype=np.float32),
        "log_strikes": np.zeros(
            (samples, actions, expiries, strikes), dtype=np.float32
        ),
        "log_path_weight": np.log(np.array([0.5, 1.0, 1.5, 2.0])),
        "xstate": np.zeros((samples, actions, 3), dtype=np.float32),
    }
    full = ReturnDiagnosticsAccumulator(expiries=1, strikes=1)
    streamed = ReturnDiagnosticsAccumulator(expiries=1, strikes=1)
    full.update(universe)
    first = _slice_return_universe(universe, 0, 2)
    second = _slice_return_universe(universe, 2, 4)
    assert first["dC"].shape[0] == second["dC"].shape[0] == 2
    assert "xstate" not in first
    streamed.update(first)
    streamed.update(second)

    full_report = full.report()
    streamed_report = streamed.report()
    for name in (
        "weighted_mean",
        "weighted_positive_probability",
        "weighted_negative_probability",
        "weighted_path_clustered_mean_error",
    ):
        assert streamed_report["per_instrument"][name] == pytest.approx(
            full_report["per_instrument"][name]
        )


def test_one_day_return_report_does_not_label_repriced_calls_as_payoffs() -> None:
    universe = {
        "dC": np.array([[[[0.02]]]], dtype=np.float32),
        "dS": np.array([[0.1]], dtype=np.float32),
        "option_dS": np.array([[[0.1]]], dtype=np.float32),
        "C": np.array([[[[0.1]]]], dtype=np.float32),
        "spot": np.array([[1.0]], dtype=np.float32),
        "delta": np.array([[[[0.5]]]], dtype=np.float32),
        "log_strikes": np.log(np.array([[[[1.05]]]], dtype=np.float32)),
    }
    accumulator = ReturnDiagnosticsAccumulator(expiries=1, strikes=1)
    accumulator.update(universe)

    coverage = accumulator.report()["payoff_and_break_even_coverage"]

    assert coverage["option_return_horizon"] == "one-day"
    assert coverage["expiry_payoff_observed"] is False
    assert coverage["proposal_positive_payoff_count"] is None
    assert coverage["proposal_spot_crossing_count"] == [1]


def _static_report(fraction: float) -> dict[str, object]:
    return {
        "primary": {
            "aggregate": {
                "violating_surface_fraction": fraction,
                "physical_weighted_violating_surface_fraction_path_clustered_error": 0.01,
                "path_fraction": fraction,
                "conditional_magnitude": {"q99": 0.02, "maximum": 0.03},
            }
        }
    }


def _local_split() -> dict[str, object]:
    return {
        "zero_adverse": {
            "states_by_direction_exclusivity": {
                "long_only": 1,
                "short_only": 2,
                "both_directions": 3,
            }
        },
        "zero_adverse_single_instrument_policy": {
            "realized_losses_beyond_roundoff": 1
        },
        "local_effective_sample_size": {"mean": 100.0},
        "neighbor_radius": {"mean": 0.5},
    }


def test_aggregate_comparison_requires_and_writes_matched_reports(tmp_path) -> None:
    per_instrument = {
        "weighted_positive_probability": [0.5, 0.4, 0.6],
        "weighted_negative_probability": [0.5, 0.6, 0.4],
        "weighted_minority_probability": [0.5, 0.4, 0.4],
        "weighted_path_clustered_positive_probability_error": [0.01] * 3,
        "weighted_path_clustered_negative_probability_error": [0.01] * 3,
    }
    return_split = {
        "per_instrument": per_instrument,
        "zero_observed_option_sides": {
            "without_positive_path": [],
            "without_negative_path": [],
        },
    }
    common_sampling = {
        "mode": "one-step",
        "option_return_horizon": "expiry",
        "expiry_step_offsets": [2],
        "importance_sampling": None,
    }
    pca_report = {
        "status": "complete",
        "configuration": {
            "num_samples": 100_000,
            "num_alpha": 5,
            "mode": "one-step",
            "option_return_horizon": "expiry",
            "num_traded_nms": 2,
            "num_surface_state": 20,
            "neighbors": 1_024,
            "candidate_neighbors": 8_192,
            "cost_proportion": 1.0e-4,
            "seed": 7,
        },
        "ar1": {"surface_first": True},
        "sampling_diagnostics": common_sampling,
        "generated_surface_static_arbitrage": _static_report(0.2),
        "return_diagnostics": {
            "training": return_split,
            "validation": return_split,
        },
        "conditional_local_tail": {
            "configuration": {
                "neighbors": 1_024,
                "instrument_candidate_neighbors": 8_192,
            },
            "training": _local_split(),
            "validation": _local_split(),
        },
    }
    sanos_unconditional = {
        "strict_one_sided_instruments_with_roundoff_tolerance": 0,
        "per_instrument": per_instrument,
    }
    sanos_report = {
        "status": "complete",
        "paths": 100_000,
        "num_surface_alpha": 5,
        "surface_first": True,
        "market": "one-step",
        "instruments": 3,
        "configuration": {
            "seed": 7,
            "cost_proportion": 1.0e-4,
            "state_features": 21,
        },
        "sampling_diagnostics": common_sampling,
        "config": {
            "state_mode": "raw",
            "neighbors": 1_024,
            "instrument_candidate_neighbors": 8_192,
            "roundoff_multiplier": 32,
        },
        "generated_surface_static_arbitrage": _static_report(0.0),
        "unconditional_training": sanos_unconditional,
        "unconditional_validation": sanos_unconditional,
        "training": _local_split(),
        "validation": _local_split(),
    }
    output = tmp_path / "comparison.json"

    report = write_aggregate_comparison(
        pca_reports=[pca_report], sanos_reports=[sanos_report], output=output
    )

    assert report["status"] == "complete"
    assert report["matched_runs"][0]["configuration_match"] is True
    assert output.exists()
    assert output.with_suffix(".md").exists()
    assert output.with_name("comparison-summary.png").exists()
