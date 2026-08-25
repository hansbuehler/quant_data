from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest
from cdxcore.bs import bs

from quantdata.dysanos.localvolbysanos.real_trading.atm_loss_classifier import (
    AtmLossClassifierConfig,
    build_atm_loss_panel,
    build_classifier_trading_results,
    expanding_loss_probabilities,
)
from quantdata.dysanos.localvolbysanos.real_trading.atm_pnl_predictor import (
    expanding_pnl_predictions,
)
from quantdata.dysanos.localvolbysanos.real_trading.attribution import _regression
from quantdata.dysanos.localvolbysanos.real_trading.delta_buckets import (
    build_delta_bucket_panel,
    build_eight_category_attribution,
    build_source_characteristic_attribution,
)
from quantdata.dysanos.localvolbysanos.real_trading.experiment import (
    Instrument,
    JointDynamics,
    OptionPair,
    RealTradingConfig,
    ScenarioPaths,
    UtilityName,
    _entropy_value_gradient,
    _powexp_utility_marginal,
    build_instruments,
    deterministic_normal_nodes,
    instrument_action_forward_deltas,
    instrument_return_matrix,
    optimize_oce,
)
from quantdata.dysanos.localvolbysanos.real_trading.loss_diagnostics import (
    _ols_return_diagnostics,
)
from quantdata.dysanos.localvolbysanos.real_trading.reversion_oce_counterfactual import (
    build_entry_horizon_returns,
    freeze_surface_generator,
    realized_delta_returns,
)
from quantdata.dysanos.localvolbysanos.real_trading.sequential import (
    SequentialAnalysisConfig,
    build_sign_only_ledger,
)
from quantdata.dysanos.localvolbysanos.real_trading.spot_only import _horizon_dates
from quantdata.dysanos.localvolbysanos.real_trading.surface_reversion_attribution import (
    ExpandingSurfaceFit,
    SpotStateFit,
    _newey_west_mean_t,
    _surface_training_shocks,
    forecast_spot_log_return,
)
from quantdata.dysanos.localvolbysanos.real_trading.tail_risk_diagnostics import (
    expanding_inverse_median_exposure,
)
from quantdata.dysanos.localvolbysanos.surface_dynamics import (
    MatrixExponentialAR1,
    PCAResult,
    SurfacePanel,
)
from quantdata.biz_calendar import DAYS_PER_YEAR


def _pairs() -> list[OptionPair]:
    return [
        OptionPair(
            expiry_index=0,
            expiry_days=10,
            expiry_date=date(2025, 1, 17),
            strike_over_forward=0.95,
            effective_nms=-1.0,
            call_bid=0.060,
            call_ask=0.062,
            put_bid=0.009,
            put_ask=0.011,
        ),
        OptionPair(
            expiry_index=0,
            expiry_days=10,
            expiry_date=date(2025, 1, 17),
            strike_over_forward=1.05,
            effective_nms=1.0,
            call_bid=0.008,
            call_ask=0.010,
            put_bid=0.058,
            put_ask=0.061,
        ),
    ]


def test_requested_universe_dividers() -> None:
    pairs = _pairs()
    itm = build_instruments(pairs, "ITM")
    actual = build_instruments(pairs, "actual")
    normalised = build_instruments(pairs, "normalised")

    assert [instrument.is_call for instrument in itm] == [True, False]
    assert [instrument.is_call for instrument in actual] == [False, True]
    assert [instrument.is_call for instrument in normalised] == [False, True]

    assert actual[0].long_divider == pytest.approx(pairs[0].put_ask)
    assert actual[0].short_divider == pytest.approx(pairs[0].put_bid)
    assert normalised[0].long_divider == pytest.approx(pairs[0].call_ask)
    assert normalised[0].short_divider == pytest.approx(pairs[0].call_bid)
    assert normalised[1].long_divider == pytest.approx(pairs[1].put_ask)
    assert normalised[1].short_divider == pytest.approx(pairs[1].put_bid)


def test_entropy_value_and_gradient_at_zero() -> None:
    returns = np.asarray(
        [
            [0.10, -0.20],
            [0.00, 0.10],
            [-0.10, 0.40],
            [0.20, -0.30],
        ],
        dtype=np.float64,
    )
    value, gradient = _entropy_value_gradient(
        np.zeros(returns.shape[1]), returns, 10.0
    )
    assert value == pytest.approx(0.0)
    np.testing.assert_allclose(gradient, np.mean(returns, axis=0), atol=1.0e-15)


def test_power_exponential_marginal_matches_finite_difference() -> None:
    values = np.asarray([-0.20, -0.01, 0.01, 0.20], dtype=np.float64)
    risk_aversion = 10.0
    _, marginal = _powexp_utility_marginal(values, risk_aversion)
    epsilon = 1.0e-7
    upper, _ = _powexp_utility_marginal(values + epsilon, risk_aversion)
    lower, _ = _powexp_utility_marginal(values - epsilon, risk_aversion)
    np.testing.assert_allclose(
        marginal,
        (upper - lower) / (2.0 * epsilon),
        rtol=1.0e-8,
        atol=1.0e-9,
    )


@pytest.mark.parametrize("utility", ["entropy", "powexp"])
def test_oce_optimizer_chooses_no_trade_for_negative_certain_returns(
    utility: str,
) -> None:
    config = RealTradingConfig(scenario_paths=32)
    returns = -np.ones((config.scenario_paths, 3), dtype=np.float64) * np.asarray(
        [0.01, 0.02, 0.03]
    )
    result = optimize_oce(
        returns,
        np.zeros(returns.shape[1], dtype=np.float64),
        utility=utility,
        risk_aversion=10.0,
        config=config,
    )
    assert result.success
    assert result.oce == pytest.approx(0.0, abs=1.0e-10)
    np.testing.assert_allclose(result.weights, 0.0, atol=1.0e-9)


def test_oce_optimizer_caps_signed_net_forward_delta() -> None:
    config = RealTradingConfig(
        scenario_paths=32,
        max_gross_fraction=1.0,
        max_position_fraction=1.0,
        max_abs_net_delta_fraction=0.4,
    )
    returns = np.full((config.scenario_paths, 1), 0.10, dtype=np.float64)
    action_forward_deltas = np.asarray([2.0], dtype=np.float64)

    result = optimize_oce(
        returns,
        action_forward_deltas,
        utility="entropy",
        risk_aversion=10.0,
        config=config,
    )

    assert result.success
    assert result.weights[0] == pytest.approx(0.2, abs=1.0e-8)
    assert result.net_forward_delta_fraction == pytest.approx(0.4, abs=1.0e-8)


def test_oce_optimizer_adjusts_bounds_for_carried_spot() -> None:
    config = RealTradingConfig(
        scenario_paths=32,
        max_gross_fraction=1.0,
        max_position_fraction=1.0,
        max_abs_net_delta_fraction=0.4,
    )
    returns = np.full((config.scenario_paths, 1), 0.10, dtype=np.float64)
    action_forward_deltas = np.asarray([2.0], dtype=np.float64)

    result = optimize_oce(
        returns,
        action_forward_deltas,
        utility="entropy",
        risk_aversion=10.0,
        config=config,
        fixed_scenario_returns=np.zeros(config.scenario_paths),
        initial_forward_delta_fraction=0.3,
    )

    assert result.success
    assert result.weights[0] == pytest.approx(0.05, abs=1.0e-8)
    assert result.delta_adjustment_fraction == pytest.approx(0.1, abs=1.0e-8)
    assert result.net_forward_delta_fraction == pytest.approx(0.4, abs=1.0e-8)


def test_entropy_gradient_includes_fixed_spot_endowment() -> None:
    returns = np.asarray(
        [[0.10], [0.00], [-0.10], [0.20]],
        dtype=np.float64,
    )
    fixed = np.asarray([-0.04, 0.01, 0.03, -0.02], dtype=np.float64)
    weights = np.asarray([0.2], dtype=np.float64)
    risk_aversion = 10.0

    value, gradient = _entropy_value_gradient(
        weights,
        returns,
        risk_aversion,
        fixed,
    )
    portfolio = fixed + returns[:, 0] * weights[0]
    probabilities = np.exp(-risk_aversion * portfolio)
    probabilities /= probabilities.sum()
    expected_value = -math.log(
        np.mean(np.exp(-risk_aversion * portfolio))
    ) / risk_aversion

    assert value == pytest.approx(expected_value)
    assert gradient[0] == pytest.approx(returns[:, 0] @ probabilities)


def test_sobol_nodes_are_reproducible_and_balanced() -> None:
    config = RealTradingConfig(scenario_paths=64, seed=1234)
    first = deterministic_normal_nodes(config)
    second = deterministic_normal_nodes(config)
    assert first.shape == (64, 5, 6)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert float(np.max(np.abs(np.mean(first, axis=0)))) < 0.08


def test_surviving_option_scenarios_exit_at_model_mid() -> None:
    num_scenarios = 3
    paths = ScenarioPaths(
        terminal_factor_state=np.zeros((num_scenarios, 1), dtype=np.float64),
        cumulative_log_forward_returns=np.zeros((num_scenarios, 5), dtype=np.float64),
    )
    panel = SurfacePanel(
        dates=[date(2025, 1, 2)],
        expiry_days=np.asarray([5, 10]),
        sqrt_expiry_years=np.sqrt(np.asarray([5, 10]) / DAYS_PER_YEAR),
        effective_nms=np.asarray([-1.0, 1.0]),
        states={},
    )
    volatility = 0.20
    future_log_volatility = np.full(
        (num_scenarios, 2, 2), math.log(volatility), dtype=np.float64
    )
    instrument = Instrument(
        pair_index=0,
        expiry_days=10,
        expiry_date=date(2025, 1, 17),
        strike_over_forward=1.0,
        effective_nms=0.0,
        is_call=True,
        bid=0.04,
        ask=0.05,
        long_divider=0.05,
        short_divider=0.04,
    )
    returns, actions = instrument_return_matrix(
        [instrument],
        paths,
        future_log_volatility,
        panel,
        config=RealTradingConfig(scenario_paths=32),
    )
    model_mid = float(
        bs.price(
            k=1.0,
            vol=volatility,
            sqrtT=math.sqrt(5 / DAYS_PER_YEAR),
            is_call=True,
        )
    )
    assert actions == [(0, "long"), (0, "short")]
    np.testing.assert_allclose(returns[:, 0], (model_mid - 0.05) / 0.05)
    np.testing.assert_allclose(returns[:, 1], (0.04 - model_mid) / 0.04)


def test_action_forward_deltas_use_side_specific_dividers() -> None:
    instrument = Instrument(
        pair_index=0,
        expiry_days=10,
        expiry_date=date(2025, 1, 17),
        strike_over_forward=1.0,
        effective_nms=0.0,
        is_call=True,
        bid=0.04,
        ask=0.05,
        long_divider=0.05,
        short_divider=0.04,
    )
    actions = [(0, "long"), (0, "short")]
    action_deltas = instrument_action_forward_deltas([instrument], actions)
    implied_volatility = float(
        bs.implied(
            k=np.asarray([1.0]),
            prices=np.asarray([0.045]),
            is_call=np.asarray([True]),
            sqrtT=np.asarray([math.sqrt(10 / DAYS_PER_YEAR)]),
        )[0]
    )
    pure_delta = float(
        bs.delta(
            k=1.0,
            vol=implied_volatility,
            is_call=True,
            sqrtT=math.sqrt(10 / DAYS_PER_YEAR),
        )
    )

    np.testing.assert_allclose(
        action_deltas,
        np.asarray([pure_delta / 0.05, -pure_delta / 0.04]),
    )


def test_quadratic_spot_attribution_recovers_components() -> None:
    spot_return = np.linspace(-0.10, 0.10, 64)
    realized_return = 0.02 + 1.5 * spot_return + 4.0 * np.square(spot_return)
    result = _regression(spot_return, realized_return, hac_lags=4)

    assert result["quadratic_linear_spot_coefficient"] == pytest.approx(1.5)
    assert result["quadratic_spot_coefficient"] == pytest.approx(4.0)
    assert result["quadratic_spot_r_squared"] == pytest.approx(1.0)
    reconstructed = (
        np.asarray(result["quadratic_linear_spot_component"])
        + np.asarray(result["quadratic_convexity_component"])
        + np.asarray(result["quadratic_spot_neutral_component"])
    )
    np.testing.assert_allclose(reconstructed, realized_return, atol=1.0e-15)


def test_spot_only_horizon_dates_follow_panel_sessions() -> None:
    dates = [date(2025, 1, day) for day in range(2, 9)]
    horizons = _horizon_dates(dates, 2)
    assert horizons["date"].to_list() == dates[:-2]
    assert horizons["horizon_date"].to_list() == dates[2:]


def test_sign_only_sequential_ledger_retains_daily_pnl() -> None:
    dates = [date(2025, 1, day) for day in range(2, 6)]
    comparison = pl.DataFrame(
        {
            "universe": ["normalised"] * 3,
            "utility": ["entropy"] * 3,
            "risk_aversion": [10.0] * 3,
            "date": dates[:3],
            "horizon_date": dates[1:],
            "spot_complete": [True] * 3,
            "net_entry_delta": [2.0, -1.0, 1.0],
            "horizon_spot_return": [0.10, -0.05, 0.02],
        }
    )

    ledger, summaries = build_sign_only_ledger(
        comparison,
        config=SequentialAnalysisConfig(capital_dollars=1_000_000.0),
    )

    np.testing.assert_allclose(
        ledger["daily_spot_pnl_dollars"].to_numpy(),
        np.asarray([100_000.0, 50_000.0, 20_000.0]),
    )
    assert ledger["cap_clip_cash_fraction"].to_list() == pytest.approx(
        [0.0, 0.1, 0.0]
    )
    assert summaries[0]["final_retained_pnl_dollars"] == pytest.approx(
        170_000.0
    )


def test_sign_only_sequential_ledger_retains_prior_sign_on_zero() -> None:
    dates = [date(2025, 1, day) for day in range(2, 6)]
    comparison = pl.DataFrame(
        {
            "universe": ["ITM"] * 3,
            "utility": ["cvar50"] * 3,
            "risk_aversion": [1.0] * 3,
            "date": dates[:3],
            "horizon_date": dates[1:],
            "spot_complete": [True] * 3,
            "net_entry_delta": [0.5, 0.0, -0.5],
            "horizon_spot_return": [0.01, 0.02, 0.03],
        }
    )

    ledger, summaries = build_sign_only_ledger(
        comparison,
        config=SequentialAnalysisConfig(),
    )

    assert ledger["target_spot_fraction"].to_list() == [1.0, 1.0, -1.0]
    assert ledger["retained_previous_sign"].to_list() == [False, True, False]
    assert summaries[0]["num_zero_signals_retained"] == 1
    assert summaries[0]["final_retained_pnl_dollars"] == pytest.approx(0.0)


def test_loss_diagnostics_standardizes_ols_predictors() -> None:
    features = np.column_stack(
        (
            np.arange(12, dtype=float),
            np.square(np.arange(12, dtype=float)),
        )
    )
    returns = 0.01 + 0.002 * features[:, 0] - 0.0001 * features[:, 1]

    coefficients, _, _, metrics = _ols_return_diagnostics(returns, features)

    scales = np.std(features, axis=0, ddof=0)
    assert coefficients[0] == pytest.approx(float(np.mean(returns)))
    assert coefficients[1] == pytest.approx(0.002 * scales[0])
    assert coefficients[2] == pytest.approx(-0.0001 * scales[1])
    assert metrics["r_squared"] == pytest.approx(1.0)


def test_atm_loss_panel_uses_five_ATMs_strictly_before_today() -> None:
    dates = [date(2025, 1, day) for day in range(2, 10)]
    comparison = pl.DataFrame(
        {
            "date": dates,
            "universe": ["normalised"] * len(dates),
            "utility": ["entropy"] * len(dates),
            "risk_aversion": [10.0] * len(dates),
            "spot_complete": [True] * len(dates),
            "spot_only_return": np.linspace(-0.01, 0.01, len(dates)),
            "unit_gross_spot_return": np.linspace(0.01, -0.01, len(dates)),
        }
    )
    empty_cvar = comparison.clear()
    market = pl.DataFrame(
        {
            "date": dates,
            "front_week_atm_volatility": np.arange(1.0, 9.0),
        }
    )

    panel = build_atm_loss_panel(
        comparison,
        empty_cvar,
        market,
        config=AtmLossClassifierConfig(minimum_training_days=30),
    )

    continuous = panel.filter(pl.col("variant") == "continuous").sort("date")
    assert continuous["date"].to_list() == dates[5:]
    assert continuous["past_five_atm_median"].to_list() == [3.0, 4.0, 5.0]
    assert continuous["front_week_atm_volatility"].to_list() == [6.0, 7.0, 8.0]


def test_expanding_atm_classifier_does_not_use_current_label() -> None:
    features = np.column_stack(
        (
            np.linspace(0.1, 0.3, 40),
            np.linspace(0.11, 0.29, 40),
        )
    )
    loss = np.tile(np.asarray([0, 1], dtype=np.int8), 20)
    config = AtmLossClassifierConfig(minimum_training_days=30)

    probabilities, _ = expanding_loss_probabilities(
        loss,
        features,
        config=config,
    )
    changed = loss.copy()
    changed[-1] = 1 - changed[-1]
    changed_probabilities, _ = expanding_loss_probabilities(
        changed,
        features,
        config=config,
    )

    assert changed_probabilities == pytest.approx(probabilities)


def test_atm_classifier_trading_filter_holds_cash_on_predicted_losses() -> None:
    predictions = pl.DataFrame(
        {
            "objective": ["test"] * 3,
            "variant": ["digital"] * 3,
            "date": [date(2025, 1, day) for day in range(2, 5)],
            "strategy_return": [0.01, -0.02, 0.03],
            "loss_probability": [0.4, 0.6, 0.2],
            "predicted_loss": [False, True, False],
        }
    )

    paths, summary = build_classifier_trading_results(predictions)

    filtered_paths = paths.filter(pl.col("approach") == "classifier_filter")
    filtered_summary = summary.filter(
        pl.col("approach") == "classifier_filter"
    ).row(0, named=True)
    assert filtered_paths["spot_notional_dollars"].to_list() == [
        1_000_000.0,
        0.0,
        1_000_000.0,
    ]
    assert filtered_paths["pnl_dollars"].to_list() == [10_000.0, 0.0, 30_000.0]
    assert filtered_summary["terminal_pnl_dollars"] == pytest.approx(40_000.0)
    assert filtered_summary["mean_abs_notional_dollars"] == pytest.approx(
        2_000_000.0 / 3.0
    )


def test_expanding_atm_pnl_model_does_not_use_current_pnl() -> None:
    features = np.column_stack(
        (
            np.linspace(0.1, 0.3, 40),
            np.linspace(0.11, 0.29, 40),
        )
    )
    pnl_dollars = np.linspace(-10_000.0, 20_000.0, 40)

    predictions, _, _ = expanding_pnl_predictions(
        pnl_dollars,
        features,
        minimum_training_days=30,
    )
    changed_pnl = pnl_dollars.copy()
    changed_pnl[-1] = -1_000_000.0
    changed_predictions, _, _ = expanding_pnl_predictions(
        changed_pnl,
        features,
        minimum_training_days=30,
    )

    assert changed_predictions == pytest.approx(predictions)


def test_delta_bucket_panel_is_exhaustive_and_reconciles() -> None:
    market_date = date(2025, 1, 2)
    positions = pl.DataFrame(
        {
            "date": [market_date] * 4,
            "universe": ["combined"] * 4,
            "utility": ["entropy"] * 4,
            "risk_aversion": [10.0] * 4,
            "side": ["long", "long", "short", "short"],
            "option_type": ["call", "put", "call", "put"],
            "strike_over_forward": [0.9, 0.9, 1.1, 1.1],
            "expiry_date": [date(2025, 1, 10)] * 4,
            "expiry_days": [6] * 4,
            "effective_nms": [-1.0, -1.0, 1.0, 1.0],
            "allocation_fraction": [0.1] * 4,
            "allocation_dollars": [100_000.0] * 4,
            "entry_pure_forward_delta": [0.7, -0.3, 0.3, -0.7],
            "net_forward_delta_contribution": [0.7, -0.3, -0.3, 0.7],
            "spot_only_return_contribution": [0.007, -0.003, -0.003, 0.007],
        }
    )
    comparison = pl.DataFrame(
        {
            "date": [market_date],
            "universe": ["combined"],
            "utility": ["entropy"],
            "risk_aversion": [10.0],
            "date_index": [0],
            "spot_complete": [True],
            "horizon_spot_return": [0.01],
            "net_entry_delta": [0.8],
            "spot_only_return": [0.008],
        }
    )

    annotated, panel, reconciliation = build_delta_bucket_panel(
        positions,
        comparison,
    )

    assert len(panel) == 8
    assert set(annotated["delta_bucket"]) == {
        "long_below_atm_call",
        "long_below_atm_put",
        "short_above_atm_call",
        "short_above_atm_put",
    }
    assert panel["bucket_net_forward_delta"].sum() == pytest.approx(0.8)
    assert reconciliation["net_delta_error"].item() == pytest.approx(0.0)
    assert reconciliation["spot_only_return_error"].item() == pytest.approx(0.0)


def test_eight_category_attribution_keeps_source_universes_separate() -> None:
    market_date = date(2025, 1, 2)
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        for strike_region in ("below_atm", "above_atm"):
            for option_type in ("call", "put"):
                is_itm_option = (
                    strike_region == "below_atm" and option_type == "call"
                ) or (
                    strike_region == "above_atm" and option_type == "put"
                )
                universe = "ITM" if is_itm_option else "normalised"
                rows.append(
                    {
                        "date": market_date,
                        "universe": universe,
                        "utility": "entropy",
                        "risk_aversion": 10.0,
                        "strike_region": strike_region,
                        "side": side,
                        "option_type": option_type,
                        "delta_bucket": (
                            f"{side}_{strike_region}_{option_type}"
                        ),
                        "num_option_positions": 1,
                        "bucket_net_forward_delta": 0.1,
                        "bucket_spot_only_return": 0.001,
                        "spot_only_return": 0.004,
                    }
                )
    panel = pl.DataFrame(rows)

    attribution, summary = build_eight_category_attribution(panel)

    assert len(attribution) == 8
    assert len(summary) == 8
    assert set(attribution["source_universe"]) == {"ITM", "normalised"}
    assert summary["cumulative_bucket_pnl_dollars"].to_list() == [1_000.0] * 8


def test_characteristic_attribution_has_separate_source_margins() -> None:
    market_date = date(2025, 1, 2)
    rows: list[dict[str, object]] = []
    for universe, scale in (("ITM", 1.0), ("normalised", 2.0)):
        for side_sign, side in ((1.0, "long"), (-1.0, "short")):
            for region_sign, strike_region in (
                (1.0, "below_atm"),
                (-1.0, "above_atm"),
            ):
                rows.append(
                    {
                        "date": market_date,
                        "universe": universe,
                        "utility": "entropy",
                        "risk_aversion": 10.0,
                        "side": side,
                        "strike_region": strike_region,
                        "bucket_net_forward_delta": scale * side_sign * region_sign,
                        "bucket_spot_only_return": 0.001 * scale,
                    }
                )
    panel = pl.DataFrame(rows)

    daily, summary = build_source_characteristic_attribution(panel)

    assert len(daily) == 8
    assert set(summary["source_universe"]) == {"ITM", "normalised"}
    itm_side = summary.filter(
        (pl.col("source_universe") == "ITM")
        & (pl.col("grouping") == "side")
    )
    normalised_side = summary.filter(
        (pl.col("source_universe") == "normalised")
        & (pl.col("grouping") == "side")
    )
    assert itm_side["cumulative_attributed_pnl_dollars"].to_list() == [2_000.0] * 2
    assert normalised_side["cumulative_attributed_pnl_dollars"].to_list() == [
        4_000.0
    ] * 2


@pytest.mark.parametrize(
    ("utility", "expected_weights", "expected_oce"),
    (
        ("cvar10", np.asarray([0.0, 1.0]), 1.0 / 30.0),
        ("cvar50", np.asarray([1.0, 0.0]), 0.01),
    ),
)
def test_cvar_optimizes_requested_lower_tail(
    utility: UtilityName,
    expected_weights: np.ndarray,
    expected_oce: float,
) -> None:
    returns = np.column_stack(
        (
            np.full(32, 0.01),
            np.concatenate((np.full(16, -0.10), np.full(16, 0.20))),
        )
    )
    config = RealTradingConfig(
        scenario_paths=32,
        max_gross_fraction=1.0,
        max_position_fraction=1.0,
        max_worst_loss_fraction=1.0,
    )

    result = optimize_oce(
        returns,
        np.zeros(2),
        utility=utility,
        risk_aversion=1.0,
        config=config,
    )

    assert result.success
    np.testing.assert_allclose(result.weights, expected_weights, atol=1.0e-10)
    assert result.oce == pytest.approx(expected_oce)


def test_tail_risk_exposure_uses_only_expanding_median() -> None:
    values = np.concatenate((np.arange(1.0, 31.0), np.asarray([60.0, 10.0])))

    exposure = expanding_inverse_median_exposure(values, 30)

    assert exposure[0] == pytest.approx(15.5 / 60.0)
    assert exposure[1] == pytest.approx(1.0)


def test_spot_forecast_reverts_state_between_sessions() -> None:
    generator = np.asarray([[-365.25 * math.log(2.0)]])
    ar = MatrixExponentialAR1(
        generator=generator,
        factor_scales=np.ones(1),
        one_calendar_day_transition=np.asarray([[0.5]]),
        eigenvalues=np.asarray([generator[0, 0]]),
        half_life_calendar_days=np.asarray([1.0]),
        spectral_radius_one_day=0.5,
        r_squared=0.5,
        optimizer_success=True,
        optimizer_message="test",
        num_function_evaluations=1,
    )
    spot_fit = SpotStateFit(
        intercept=0.01,
        state_coefficients=np.asarray([0.02]),
        shock_coefficients=np.asarray([-0.03]),
        expanding_mean_log_return=0.0,
    )

    reverting = forecast_spot_log_return(
        np.asarray([1.0]),
        np.asarray([1, 1]),
        ar,
        spot_fit,
        revert_state=True,
    )
    frozen = forecast_spot_log_return(
        np.asarray([1.0]),
        np.asarray([1, 1]),
        ar,
        spot_fit,
        revert_state=False,
    )

    assert reverting == pytest.approx(0.01 + 0.02 + 0.01 + 0.01)
    assert frozen == pytest.approx(0.01 + 0.02 + 0.01 + 0.02)


def test_newey_west_mean_t_has_expected_sign() -> None:
    assert _newey_west_mean_t(np.asarray([1.0, 2.0, 1.0]), max_lag=1) > 0.0
    assert _newey_west_mean_t(np.asarray([-1.0, -2.0, -1.0]), max_lag=1) < 0.0


def test_surface_training_shocks_align_adjacent_dates() -> None:
    ar = MatrixExponentialAR1(
        generator=np.zeros((1, 1)),
        factor_scales=np.ones(1),
        one_calendar_day_transition=np.ones((1, 1)),
        eigenvalues=np.zeros(1),
        half_life_calendar_days=np.full(1, np.inf),
        spectral_radius_one_day=1.0,
        r_squared=0.0,
        optimizer_success=True,
        optimizer_message="test",
        num_function_evaluations=1,
    )
    fit = ExpandingSurfaceFit(
        pca=PCAResult(
            mean_surface=np.zeros((1, 1)),
            loadings=np.ones((1, 1, 1)),
            scores=np.asarray([[0.0], [1.0], [3.0]]),
            explained_variance_ratio=np.ones(1),
        ),
        ar=ar,
        standardized_states=np.asarray([[0.0], [1.0], [3.0]]),
        current_residual_surface=np.zeros((1, 1)),
    )

    shocks = _surface_training_shocks(
        fit,
        [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)],
    )

    np.testing.assert_allclose(shocks, np.asarray([[1.0], [2.0]]))


def test_freeze_surface_generator_changes_only_ar_transition() -> None:
    pca = PCAResult(
        mean_surface=np.zeros((1, 1)),
        loadings=np.ones((1, 1, 1)),
        scores=np.zeros((4, 1)),
        explained_variance_ratio=np.ones(1),
    )
    ar = MatrixExponentialAR1(
        generator=np.asarray([[-1.0]]),
        factor_scales=np.ones(1),
        one_calendar_day_transition=np.asarray([[math.exp(-1.0 / 365.25)]]),
        eigenvalues=np.asarray([-1.0]),
        half_life_calendar_days=np.asarray([math.log(2.0) * 365.25]),
        spectral_radius_one_day=math.exp(-1.0 / 365.25),
        r_squared=0.5,
        optimizer_success=True,
        optimizer_message="test",
        num_function_evaluations=1,
    )
    dynamics = JointDynamics(
        pca=pca,
        ar=ar,
        spot_intercept=0.0,
        spot_state_coefficients=np.ones(1),
        spot_surface_shock_coefficients=np.ones(1),
        joint_covariance=np.eye(2),
        joint_cholesky=np.eye(2),
        spot_r_squared=0.2,
    )

    frozen = freeze_surface_generator(dynamics)

    np.testing.assert_allclose(frozen.ar.generator, 0.0)
    np.testing.assert_allclose(frozen.ar.one_calendar_day_transition, 1.0)
    assert frozen.pca is dynamics.pca
    assert frozen.joint_cholesky is dynamics.joint_cholesky
    assert frozen.spot_state_coefficients is dynamics.spot_state_coefficients


def test_realized_delta_returns_close_expiring_slices_separately() -> None:
    instruments = [
        Instrument(
            pair_index=0,
            expiry_days=1,
            expiry_date=date(2025, 1, 3),
            strike_over_forward=1.0,
            effective_nms=0.0,
            is_call=True,
            bid=0.01,
            ask=0.02,
            long_divider=0.02,
            short_divider=0.01,
        ),
        Instrument(
            pair_index=1,
            expiry_days=2,
            expiry_date=date(2025, 1, 6),
            strike_over_forward=1.0,
            effective_nms=0.0,
            is_call=False,
            bid=0.01,
            ask=0.02,
            long_divider=0.02,
            short_divider=0.01,
        ),
    ]
    result = realized_delta_returns(
        np.ones(2),
        np.asarray([1.0, -0.5]),
        instruments,
        [(0, "long"), (1, "long")],
        entry_date=date(2025, 1, 2),
        entry_spot=100.0,
        target_dates={2: date(2025, 1, 6)},
        spot_by_date={
            date(2025, 1, 2): 100.0,
            date(2025, 1, 3): 110.0,
            date(2025, 1, 6): 120.0,
        },
    )[2]

    assert result["continuous_return"] == pytest.approx(0.0)
    assert result["gross_scheduled_delta"] == pytest.approx(1.5)
    assert result["digital_return"] == pytest.approx(0.0)

    missing = realized_delta_returns(
        np.ones(2),
        np.asarray([1.0, -0.5]),
        instruments,
        [(0, "long"), (1, "long")],
        entry_date=date(2025, 1, 2),
        entry_spot=100.0,
        target_dates={2: date(2025, 1, 6)},
        spot_by_date={
            date(2025, 1, 2): 100.0,
            date(2025, 1, 6): 120.0,
        },
    )
    assert missing == {}


def test_entry_horizon_returns_use_net_delta_and_its_sign() -> None:
    comparisons = pl.DataFrame(
        {
            "date": [date(2025, 1, 2)],
            "date_index": [0],
            "universe": ["normalised"],
            "utility": ["entropy"],
            "risk_aversion": [10.0],
            "objective": ["Entropy 10"],
            "fitted_ar_net_delta": [0.5],
            "frozen_state_net_delta": [-0.25],
        }
    )

    returns = build_entry_horizon_returns(
        comparisons,
        [date(2025, 1, 2), date(2025, 1, 3)],
        np.asarray([100.0, 110.0]),
        horizons_sessions=(1,),
    )

    assert returns.height == 4
    fitted = returns.filter(pl.col("model_mode") == "fitted_ar").sort("variant")
    frozen = returns.filter(pl.col("model_mode") == "frozen_state").sort("variant")
    assert fitted["realized_return"].to_list() == pytest.approx([0.05, 0.10])
    assert frozen["realized_return"].to_list() == pytest.approx([-0.025, -0.10])
