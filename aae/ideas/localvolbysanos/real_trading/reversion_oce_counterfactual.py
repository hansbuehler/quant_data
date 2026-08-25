"""Chronological OCE ablation of fitted surface reversion versus ``G = 0``."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp

from ..surface_dynamics import MatrixExponentialAR1, SurfacePanel, load_surface_panel
from .attribution import _hac_mean_standard_error
from .experiment import (
    CVAR_LOWER_TAIL_FRACTIONS,
    UNIVERSES,
    Instrument,
    JointDynamics,
    RealTradingConfig,
    UniverseName,
    UtilityName,
    _calendar_gaps,
    _empirical_lower_tail_mean,
    _entropy_value_gradient,
    _load_markets,
    _powexp_oce_value,
    build_instruments,
    build_option_pairs,
    deterministic_normal_nodes,
    fit_joint_dynamics,
    instrument_action_forward_deltas,
    instrument_return_matrix,
    optimize_oce,
    scenario_log_volatility_surfaces,
    simulate_paths,
)

MODEL_MODES = ("fitted_ar", "frozen_state")
OBJECTIVE_SPECS: tuple[tuple[UtilityName, float, str], ...] = (
    ("entropy", 10.0, "Entropy 10"),
    ("powexp", 10.0, "Power-exp 10"),
    ("cvar10", 1.0, "Worst 90% CVaR"),
)


@dataclass(frozen=True, kw_only=True)
class ReversionOCECounterfactualConfig:
    """Chronological fit, optimizer, and realized-return conventions."""

    minimum_training_dates: int = 60
    realized_horizons_sessions: tuple[int, ...] = (1, 5)
    universes: tuple[UniverseName, ...] = UNIVERSES
    objective_specs: tuple[tuple[UtilityName, float, str], ...] = OBJECTIVE_SPECS
    capital_dollars: float = 1_000_000.0
    annual_sessions: int = 252
    trading: RealTradingConfig = RealTradingConfig(
        utilities=("entropy", "powexp", "cvar10"),
        risk_aversions=(10.0,),
        max_abs_net_delta_fraction=1.0,
    )

    def __post_init__(self) -> None:
        verify_inp(
            self.minimum_training_dates >= self.trading.num_factors * 4,
            lambda: (
                "minimum_training_dates must be at least four times the "
                "number of surface factors"
            ),
        )
        verify_inp(
            self.realized_horizons_sessions
            and all(value >= 1 for value in self.realized_horizons_sessions)
            and len(set(self.realized_horizons_sessions))
            == len(self.realized_horizons_sessions),
            lambda: "realized horizons must contain unique positive integers",
        )
        verify_inp(
            self.universes
            and all(universe in UNIVERSES for universe in self.universes),
            lambda: f"universes must be selected from {UNIVERSES}",
        )
        verify_inp(
            self.objective_specs
            and all(
                utility in self.trading.utilities
                and np.isfinite(risk_aversion)
                and risk_aversion > 0.0
                for utility, risk_aversion, _ in self.objective_specs
            ),
            lambda: "objective_specs must agree with the trading configuration",
        )
        verify_inp(
            np.isfinite(self.capital_dollars) and self.capital_dollars > 0.0,
            lambda: "capital_dollars must be finite and positive",
        )
        verify_inp(
            self.annual_sessions >= 1,
            lambda: "annual_sessions must be positive",
        )


def _slice_panel(panel: SurfacePanel, end_index: int) -> SurfacePanel:
    """Return panel observations through and including ``end_index``."""

    return SurfacePanel(
        dates=panel.dates[: end_index + 1],
        expiry_days=panel.expiry_days,
        sqrt_expiry_years=panel.sqrt_expiry_years,
        effective_nms=panel.effective_nms,
        states={
            state: values[: end_index + 1]
            for state, values in panel.states.items()
        },
    )


def freeze_surface_generator(dynamics: JointDynamics) -> JointDynamics:
    """Set only the deterministic surface generator to zero."""

    num_factors = dynamics.ar.generator.shape[0]
    frozen_ar = replace(
        dynamics.ar,
        generator=np.zeros_like(dynamics.ar.generator),
        one_calendar_day_transition=np.eye(num_factors, dtype=np.float64),
        eigenvalues=np.zeros(num_factors, dtype=np.float64),
        half_life_calendar_days=np.full(num_factors, np.inf, dtype=np.float64),
        spectral_radius_one_day=1.0,
        r_squared=float("nan"),
    )
    return replace(dynamics, ar=frozen_ar)


def _portfolio_oce(
    portfolio: np.ndarray,
    *,
    utility: UtilityName,
    risk_aversion: float,
) -> float:
    if utility == "entropy":
        empty_returns = np.empty((len(portfolio), 0), dtype=np.float64)
        return _entropy_value_gradient(
            np.empty(0, dtype=np.float64),
            empty_returns,
            risk_aversion,
            portfolio,
        )[0]
    if utility == "powexp":
        return _powexp_oce_value(portfolio, risk_aversion)
    return _empirical_lower_tail_mean(
        portfolio,
        CVAR_LOWER_TAIL_FRACTIONS[utility],
    )


def realized_delta_returns(
    weights: np.ndarray,
    action_forward_deltas: np.ndarray,
    instruments: list[Instrument],
    actions: list[tuple[int, str]],
    *,
    entry_date: date,
    entry_spot: float,
    target_dates: dict[int, date],
    spot_by_date: dict[date, float],
) -> dict[int, dict[str, float]]:
    """Mark entry delta slices at option expiry or the requested horizon."""

    verify_inp(
        weights.shape == action_forward_deltas.shape == (len(actions),),
        lambda: "weights, action deltas, and actions must align",
    )
    verify_inp(
        np.isfinite(entry_spot) and entry_spot > 0.0,
        lambda: "entry_spot must be positive and finite",
    )
    contributions = weights * action_forward_deltas
    results: dict[int, dict[str, float]] = {}
    for horizon, target_date in target_dates.items():
        close_slices: dict[date, float] = {}
        spot_available = True
        for contribution, (instrument_index, _) in zip(
            contributions, actions, strict=True
        ):
            if abs(contribution) <= 1.0e-14:
                continue
            instrument = instruments[instrument_index]
            expiry_date = instrument.expiry_date
            close_date = min(expiry_date, target_date)
            verify(
                close_date > entry_date,
                lambda: f"close date {close_date} is not after entry {entry_date}",
            )
            if close_date not in spot_by_date:
                spot_available = False
                break
            close_slices[close_date] = close_slices.get(close_date, 0.0) + float(
                contribution
            )
        if not spot_available:
            continue
        continuous_return = float(
            sum(
                delta * (spot_by_date[close_date] / entry_spot - 1.0)
                for close_date, delta in close_slices.items()
            )
        )
        gross_scheduled_delta = float(
            sum(abs(delta) for delta in close_slices.values())
        )
        digital_return = (
            continuous_return / gross_scheduled_delta
            if gross_scheduled_delta > 0.0
            else 0.0
        )
        results[horizon] = {
            "continuous_return": continuous_return,
            "digital_return": digital_return,
            "gross_scheduled_delta": gross_scheduled_delta,
        }
    return results


def _cross_evaluation_rows(
    returns_by_mode: dict[str, np.ndarray],
    weights_by_mode: dict[str, np.ndarray],
    *,
    common: dict[str, object],
    utility: UtilityName,
    risk_aversion: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for portfolio_mode, weights in weights_by_mode.items():
        for evaluation_mode, returns in returns_by_mode.items():
            portfolio = returns @ weights
            quantiles = np.quantile(portfolio, [0.01, 0.05, 0.5, 0.95])
            rows.append(
                {
                    **common,
                    "portfolio_mode": portfolio_mode,
                    "evaluation_mode": evaluation_mode,
                    "oce": _portfolio_oce(
                        portfolio,
                        utility=utility,
                        risk_aversion=risk_aversion,
                    ),
                    "expected_return": float(np.mean(portfolio)),
                    "return_std": float(np.std(portfolio, ddof=0)),
                    "return_min": float(np.min(portfolio)),
                    "return_p01": float(quantiles[0]),
                    "return_p05": float(quantiles[1]),
                    "return_median": float(quantiles[2]),
                    "return_p95": float(quantiles[3]),
                }
            )
    return rows


def _strategy_metrics(
    returns: np.ndarray,
    *,
    horizon_sessions: int,
    config: ReversionOCECounterfactualConfig,
) -> dict[str, float | int]:
    verify(
        returns.ndim == 1 and len(returns) >= 2 and np.all(np.isfinite(returns)),
        lambda: "strategy return vector must be finite and non-empty",
    )
    mean = float(np.mean(returns))
    standard_deviation = float(np.std(returns, ddof=0))
    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    wealth = 1.0 + cumulative
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    hac_standard_error = _hac_mean_standard_error(
        returns,
        lags=horizon_sessions - 1,
    )
    return {
        "num_returns": len(returns),
        "terminal_pnl_dollars": float(np.sum(returns)) * config.capital_dollars,
        "mean_pnl_dollars": mean * config.capital_dollars,
        "pnl_std_dollars": standard_deviation * config.capital_dollars,
        "annualized_sharpe": (
            mean
            / standard_deviation
            * math.sqrt(config.annual_sessions / horizon_sessions)
            if standard_deviation > 0.0
            else float("nan")
        ),
        "mean_hac_t": (
            mean / hac_standard_error
            if hac_standard_error > 0.0
            else float("nan")
        ),
        "expected_shortfall_10_dollars": _empirical_lower_tail_mean(
            returns, 0.10
        )
        * config.capital_dollars,
        "maximum_drawdown": float(np.min(drawdown)),
        "positive_fraction": float(np.mean(returns > 0.0)),
    }


def summarize_realized_returns(
    realized: pl.DataFrame,
    *,
    config: ReversionOCECounterfactualConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return per-mode strategy statistics and paired AR-minus-frozen tests."""

    summary_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    keys = ("universe", "utility", "risk_aversion", "horizon_sessions", "variant")
    for key, group in realized.partition_by(*keys, as_dict=True).items():
        horizon = int(key[3])
        for mode in MODEL_MODES:
            mode_group = group.filter(pl.col("model_mode") == mode).sort("date")
            returns = mode_group["realized_return"].to_numpy()
            summary_rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "model_mode": mode,
                    "mean_abs_spot_notional_dollars": float(
                        mode_group["spot_notional_fraction"].abs().mean()
                    )
                    * config.capital_dollars,
                    **_strategy_metrics(
                        returns,
                        horizon_sessions=horizon,
                        config=config,
                    ),
                }
            )
        fitted = (
            group.filter(pl.col("model_mode") == "fitted_ar")
            .select("date", pl.col("realized_return").alias("fitted_return"))
            .sort("date")
        )
        frozen = (
            group.filter(pl.col("model_mode") == "frozen_state")
            .select("date", pl.col("realized_return").alias("frozen_return"))
            .sort("date")
        )
        paired = fitted.join(frozen, on="date", how="inner", validate="1:1")
        difference = (
            paired["fitted_return"].to_numpy()
            - paired["frozen_return"].to_numpy()
        )
        difference_standard_error = _hac_mean_standard_error(
            difference,
            lags=horizon - 1,
        )
        difference_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "num_paired_returns": len(paired),
                "fitted_minus_frozen_terminal_pnl_dollars": float(
                    np.sum(difference) * config.capital_dollars
                ),
                "fitted_minus_frozen_mean_pnl_dollars": float(
                    np.mean(difference) * config.capital_dollars
                ),
                "fitted_minus_frozen_hac_t": (
                    float(np.mean(difference) / difference_standard_error)
                    if difference_standard_error > 0.0
                    else float("nan")
                ),
                "return_correlation": float(
                    np.corrcoef(
                        paired["fitted_return"].to_numpy(),
                        paired["frozen_return"].to_numpy(),
                    )[0, 1]
                ),
            }
        )
    return (
        pl.DataFrame(summary_rows).sort(*keys, "model_mode"),
        pl.DataFrame(difference_rows).sort(*keys),
    )


def summarize_portfolio_changes(comparisons: pl.DataFrame) -> pl.DataFrame:
    """Summarize how much fitted reversion changes allocations and delta."""

    return (
        comparisons.group_by("universe", "utility", "risk_aversion", "objective")
        .agg(
            pl.len().alias("num_dates"),
            pl.col("fitted_ar_net_delta").mean().alias("mean_fitted_ar_net_delta"),
            pl.col("frozen_state_net_delta")
            .mean()
            .alias("mean_frozen_state_net_delta"),
            (pl.col("fitted_ar_net_delta") - pl.col("frozen_state_net_delta"))
            .abs()
            .mean()
            .alias("mean_abs_net_delta_difference"),
            pl.corr("fitted_ar_net_delta", "frozen_state_net_delta").alias(
                "net_delta_correlation"
            ),
            (
                pl.col("fitted_ar_net_delta").sign()
                == pl.col("frozen_state_net_delta").sign()
            )
            .mean()
            .alias("net_delta_sign_agreement"),
            pl.col("allocation_l1_distance")
            .mean()
            .alias("mean_allocation_l1_distance"),
            pl.col("allocation_jaccard_overlap")
            .mean()
            .alias("mean_allocation_jaccard_overlap"),
            pl.col("fitted_ar_gross_fraction")
            .mean()
            .alias("mean_fitted_ar_gross_fraction"),
            pl.col("frozen_state_gross_fraction")
            .mean()
            .alias("mean_frozen_state_gross_fraction"),
            pl.col("fitted_ar_oce").mean().alias("mean_fitted_ar_own_oce"),
            pl.col("frozen_state_oce").mean().alias("mean_frozen_state_own_oce"),
        )
        .sort("universe", "utility", "risk_aversion")
    )


def summarize_delta_difference_timing(
    comparisons: pl.DataFrame,
    spots: np.ndarray,
    *,
    horizons_sessions: tuple[int, ...],
) -> pl.DataFrame:
    """Test whether the reversion-induced delta change times future spot."""

    rows: list[dict[str, object]] = []
    for key, group in comparisons.partition_by(
        "universe", "utility", "risk_aversion", "objective", as_dict=True
    ).items():
        indices = group["date_index"].to_numpy()
        delta_difference = (
            group["fitted_ar_net_delta"].to_numpy()
            - group["frozen_state_net_delta"].to_numpy()
        )
        sign_difference = np.sign(group["fitted_ar_net_delta"].to_numpy()) - np.sign(
            group["frozen_state_net_delta"].to_numpy()
        )
        for horizon in horizons_sessions:
            verify(
                int(np.max(indices)) + horizon < len(spots),
                lambda: "delta timing horizon extends past the spot panel",
            )
            spot_return = spots[indices + horizon] / spots[indices] - 1.0
            rows.append(
                {
                    "universe": key[0],
                    "utility": key[1],
                    "risk_aversion": float(key[2]),
                    "objective": key[3],
                    "horizon_sessions": horizon,
                    "num_dates": len(group),
                    "mean_delta_difference": float(np.mean(delta_difference)),
                    "mean_abs_delta_difference": float(
                        np.mean(np.abs(delta_difference))
                    ),
                    "delta_difference_future_spot_correlation": float(
                        np.corrcoef(delta_difference, spot_return)[0, 1]
                    ),
                    "mean_sign_difference": float(np.mean(sign_difference)),
                    "sign_difference_future_spot_correlation": float(
                        np.corrcoef(sign_difference, spot_return)[0, 1]
                    ),
                }
            )
    return pl.DataFrame(rows).sort(
        "universe", "utility", "risk_aversion", "horizon_sessions"
    )


def summarize_scenario_cross_evaluation(cross_evaluation: pl.DataFrame) -> pl.DataFrame:
    """Average each optimized portfolio under both scenario environments."""

    return (
        cross_evaluation.group_by(
            "universe",
            "utility",
            "risk_aversion",
            "objective",
            "portfolio_mode",
            "evaluation_mode",
        )
        .agg(
            pl.len().alias("num_dates"),
            pl.col("oce").mean().alias("mean_oce"),
            pl.col("expected_return").mean().alias("mean_expected_return"),
            pl.col("return_std").mean().alias("mean_return_std"),
            pl.col("return_p05").mean().alias("mean_return_p05"),
            pl.col("return_min").mean().alias("mean_return_min"),
        )
        .sort(
            "universe",
            "utility",
            "risk_aversion",
            "evaluation_mode",
            "portfolio_mode",
        )
    )


def build_entry_horizon_returns(
    comparisons: pl.DataFrame,
    panel_dates: list[date],
    spots: np.ndarray,
    *,
    horizons_sessions: tuple[int, ...],
) -> pl.DataFrame:
    """Hold the entry net delta, or only its sign, directly to each horizon."""

    verify_inp(
        spots.shape == (len(panel_dates),)
        and np.all(np.isfinite(spots))
        and np.all(spots > 0.0),
        lambda: "panel dates and positive spot observations must align",
    )
    rows: list[dict[str, object]] = []
    for row in comparisons.iter_rows(named=True):
        date_index = int(row["date_index"])
        for horizon in horizons_sessions:
            target_index = date_index + horizon
            verify(
                target_index < len(panel_dates),
                lambda: "entry-horizon return extends past the spot panel",
            )
            spot_return = float(spots[target_index] / spots[date_index] - 1.0)
            common = {
                key: row[key]
                for key in (
                    "date",
                    "date_index",
                    "universe",
                    "utility",
                    "risk_aversion",
                    "objective",
                )
            }
            for model_mode in MODEL_MODES:
                delta = float(row[f"{model_mode}_net_delta"])
                for variant, notional in (
                    ("continuous_entry_delta", delta),
                    ("digital_entry_sign", float(np.sign(delta))),
                ):
                    rows.append(
                        {
                            **common,
                            "model_mode": model_mode,
                            "horizon_sessions": horizon,
                            "variant": variant,
                            "target_date": panel_dates[target_index],
                            "realized_return": notional * spot_return,
                            "spot_notional_fraction": notional,
                            "gross_scheduled_delta": abs(notional),
                        }
                    )
    return pl.DataFrame(rows)


def _plot_delta_comparison(comparisons: pl.DataFrame, output_file: Path) -> None:
    import matplotlib.pyplot as plt

    normalised = comparisons.filter(pl.col("universe") == "normalised")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (_, _, objective) in zip(axes, OBJECTIVE_SPECS, strict=True):
        group = normalised.filter(pl.col("objective") == objective)
        axis.scatter(
            group["frozen_state_net_delta"].to_numpy(),
            group["fitted_ar_net_delta"].to_numpy(),
            alpha=0.65,
        )
        limits = (
            min(
                float(group["frozen_state_net_delta"].min()),
                float(group["fitted_ar_net_delta"].min()),
            ),
            max(
                float(group["frozen_state_net_delta"].max()),
                float(group["fitted_ar_net_delta"].max()),
            ),
        )
        axis.plot(limits, limits, color="black", linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        axis.axvline(0.0, color="black", linewidth=0.5, alpha=0.5)
        axis.set_title(objective)
        axis.set_xlabel("Frozen-state net delta")
        axis.set_ylabel("Fitted-AR net delta")
        axis.grid(alpha=0.2)
    figure.suptitle("Chronological OCE surface-reversion ablation: normalised universe")
    figure.savefig(output_file, dpi=170)
    plt.close(figure)


def run_reversion_oce_counterfactual(
    ticker: str,
    panel_file: Path,
    daily_summary_file: Path,
    output_dir: Path,
    *,
    config: ReversionOCECounterfactualConfig = ReversionOCECounterfactualConfig(),
) -> dict[str, object]:
    """Run the chronological fitted-AR versus frozen-state OCE ablation."""

    panel = load_surface_panel(panel_file)
    markets, market_spot_returns = _load_markets(ticker, panel.dates)
    daily = pl.read_parquet(daily_summary_file).select("date", "spot_ref").sort("date")
    verify(
        daily["date"].to_list() == panel.dates,
        lambda: "daily spot dates do not align with the surface panel",
    )
    spots = daily["spot_ref"].to_numpy()
    spot_by_date = dict(zip(panel.dates, spots, strict=True))
    nodes = deterministic_normal_nodes(config.trading)
    maximum_realized_horizon = max(config.realized_horizons_sessions)
    selected_indices = range(
        config.minimum_training_dates - 1,
        len(panel.dates) - maximum_realized_horizon,
    )

    result_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    cross_evaluation_rows: list[dict[str, object]] = []
    realized_rows: list[dict[str, object]] = []
    missing_realized_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for run_index, date_index in enumerate(selected_indices):
        market_date = panel.dates[date_index]
        day = markets[date_index]
        if day is None:
            failures.append({"date": str(market_date), "error": "market unavailable"})
            continue
        try:
            training_panel = _slice_panel(panel, date_index)
            fitted_dynamics = fit_joint_dynamics(
                training_panel,
                market_spot_returns[: date_index + 1],
                config=config.trading,
            )
            dynamics_by_mode = {
                "fitted_ar": fitted_dynamics,
                "frozen_state": freeze_surface_generator(fitted_dynamics),
            }
            standardized_state = (
                fitted_dynamics.pca.scores[-1] / fitted_dynamics.ar.factor_scales
            )
            current_surface = panel.states["log_forward_volatility"][date_index]
            calendar_gaps = _calendar_gaps(
                panel.dates,
                date_index,
                config.trading.horizon_sessions,
            )
            scenario_by_mode: dict[str, tuple[object, np.ndarray]] = {}
            for model_mode, dynamics in dynamics_by_mode.items():
                paths = simulate_paths(
                    standardized_state,
                    calendar_gaps,
                    dynamics,
                    nodes,
                )
                surfaces = scenario_log_volatility_surfaces(
                    current_surface,
                    standardized_state,
                    paths,
                    dynamics,
                    config=config.trading,
                )
                scenario_by_mode[model_mode] = (paths, surfaces)

            pairs = build_option_pairs(
                day,
                panel,
                current_surface,
                config=config.trading,
            )
            target_dates = {
                horizon: panel.dates[date_index + horizon]
                for horizon in config.realized_horizons_sessions
            }
            for universe in config.universes:
                instruments = build_instruments(pairs, universe)
                returns_by_mode: dict[str, np.ndarray] = {}
                common_actions: list[tuple[int, str]] | None = None
                for model_mode, (paths, surfaces) in scenario_by_mode.items():
                    returns, actions = instrument_return_matrix(
                        instruments,
                        paths,
                        surfaces,
                        panel,
                        config=config.trading,
                    )
                    if common_actions is None:
                        common_actions = actions
                    else:
                        verify(
                            common_actions == actions,
                            lambda: "counterfactual action sets do not align",
                        )
                    returns_by_mode[model_mode] = returns
                assert common_actions is not None
                action_deltas = instrument_action_forward_deltas(
                    instruments,
                    common_actions,
                )
                for utility, risk_aversion, objective in config.objective_specs:
                    common = {
                        "date": market_date,
                        "date_index": date_index,
                        "universe": universe,
                        "utility": utility,
                        "risk_aversion": risk_aversion,
                        "objective": objective,
                    }
                    weights_by_mode: dict[str, np.ndarray] = {}
                    results_by_mode = {}
                    realized_by_mode: dict[str, dict[int, dict[str, float]]] = {}
                    for model_mode in MODEL_MODES:
                        result = optimize_oce(
                            returns_by_mode[model_mode],
                            action_deltas,
                            utility=utility,
                            risk_aversion=risk_aversion,
                            config=config.trading,
                        )
                        weights_by_mode[model_mode] = result.weights
                        results_by_mode[model_mode] = result
                        realized_by_mode[model_mode] = realized_delta_returns(
                            result.weights,
                            action_deltas,
                            instruments,
                            common_actions,
                            entry_date=market_date,
                            entry_spot=float(spots[date_index]),
                            target_dates=target_dates,
                            spot_by_date=spot_by_date,
                        )
                        missing_horizons = set(target_dates) - set(
                            realized_by_mode[model_mode]
                        )
                        missing_realized_rows.extend(
                            {
                                **common,
                                "model_mode": model_mode,
                                "horizon_sessions": horizon,
                                "reason": "one or more expiry close dates has no saved spot",
                            }
                            for horizon in sorted(missing_horizons)
                        )
                        result_rows.append(
                            {
                                **common,
                                "model_mode": model_mode,
                                "oce": result.oce,
                                "expected_return": result.expected_return,
                                "return_std": result.return_std,
                                "return_p05": result.return_p05,
                                "return_min": result.return_min,
                                "gross_fraction": result.gross_fraction,
                                "net_forward_delta_fraction": (
                                    result.net_forward_delta_fraction
                                ),
                                "num_positions": result.num_positions,
                                "success": result.success,
                                "iterations": result.iterations,
                                "ar_training_r_squared": fitted_dynamics.ar.r_squared,
                                "ar_mean_reverting": fitted_dynamics.ar.is_mean_reverting,
                                "spot_training_r_squared": (
                                    fitted_dynamics.spot_r_squared
                                ),
                            }
                        )
                        for action_index in np.flatnonzero(
                            result.weights > 1.0e-7
                        ):
                            instrument_index, side = common_actions[
                                int(action_index)
                            ]
                            instrument = instruments[instrument_index]
                            position_rows.append(
                                {
                                    **common,
                                    "model_mode": model_mode,
                                    "action_index": int(action_index),
                                    "side": side,
                                    "option_type": (
                                        "call" if instrument.is_call else "put"
                                    ),
                                    "expiry_date": instrument.expiry_date,
                                    "expiry_days": instrument.expiry_days,
                                    "strike_over_forward": (
                                        instrument.strike_over_forward
                                    ),
                                    "allocation_fraction": float(
                                        result.weights[action_index]
                                    ),
                                    "action_forward_delta": float(
                                        action_deltas[action_index]
                                    ),
                                    "net_forward_delta_contribution": float(
                                        result.weights[action_index]
                                        * action_deltas[action_index]
                                    ),
                                }
                            )
                        for horizon, horizon_result in realized_by_mode[
                            model_mode
                        ].items():
                            for variant, return_column in (
                                (
                                    "continuous_expiry_slices",
                                    "continuous_return",
                                ),
                                (
                                    "digital_expiry_normalised",
                                    "digital_return",
                                ),
                            ):
                                realized_rows.append(
                                    {
                                        **common,
                                        "model_mode": model_mode,
                                        "horizon_sessions": horizon,
                                        "variant": variant,
                                        "target_date": target_dates[horizon],
                                        "realized_return": horizon_result[
                                            return_column
                                        ],
                                        "spot_notional_fraction": (
                                            result.net_forward_delta_fraction
                                            if variant == "continuous"
                                            else float(
                                                np.sign(
                                                    result.net_forward_delta_fraction
                                                )
                                            )
                                        ),
                                        "gross_scheduled_delta": horizon_result[
                                            "gross_scheduled_delta"
                                        ],
                                    }
                                )
                    fitted_weights = weights_by_mode["fitted_ar"]
                    frozen_weights = weights_by_mode["frozen_state"]
                    allocation_union = float(
                        np.sum(np.maximum(fitted_weights, frozen_weights))
                    )
                    comparison_rows.append(
                        {
                            **common,
                            "fitted_ar_net_delta": results_by_mode[
                                "fitted_ar"
                            ].net_forward_delta_fraction,
                            "frozen_state_net_delta": results_by_mode[
                                "frozen_state"
                            ].net_forward_delta_fraction,
                            "fitted_ar_gross_fraction": results_by_mode[
                                "fitted_ar"
                            ].gross_fraction,
                            "frozen_state_gross_fraction": results_by_mode[
                                "frozen_state"
                            ].gross_fraction,
                            "fitted_ar_oce": results_by_mode["fitted_ar"].oce,
                            "frozen_state_oce": results_by_mode["frozen_state"].oce,
                            "allocation_l1_distance": float(
                                np.sum(np.abs(fitted_weights - frozen_weights))
                            ),
                            "allocation_jaccard_overlap": (
                                float(
                                    np.sum(
                                        np.minimum(fitted_weights, frozen_weights)
                                    )
                                    / allocation_union
                                )
                                if allocation_union > 0.0
                                else 1.0
                            ),
                        }
                    )
                    cross_evaluation_rows.extend(
                        _cross_evaluation_rows(
                            returns_by_mode,
                            weights_by_mode,
                            common=common,
                            utility=utility,
                            risk_aversion=risk_aversion,
                        )
                    )
        except Exception as exception:
            failures.append(
                {
                    "date": str(market_date),
                    "error": f"{type(exception).__name__}: {exception}",
                }
            )
        if (run_index + 1) % 5 == 0:
            print(
                f"counterfactual dates {run_index + 1}/"
                f"{len(selected_indices)}; failures={len(failures)}",
                flush=True,
            )

    results = pl.DataFrame(result_rows).sort(
        "universe", "utility", "risk_aversion", "date", "model_mode"
    )
    positions = pl.DataFrame(position_rows).sort(
        "universe", "utility", "risk_aversion", "date", "model_mode", "action_index"
    )
    comparisons = pl.DataFrame(comparison_rows).sort(
        "universe", "utility", "risk_aversion", "date"
    )
    cross_evaluation = pl.DataFrame(cross_evaluation_rows).sort(
        "universe",
        "utility",
        "risk_aversion",
        "date",
        "portfolio_mode",
        "evaluation_mode",
    )
    realized = pl.concat(
        (
            pl.DataFrame(realized_rows),
            build_entry_horizon_returns(
                comparisons,
                panel.dates,
                spots,
                horizons_sessions=config.realized_horizons_sessions,
            ),
        ),
        how="vertical_relaxed",
    ).sort(
        "universe",
        "utility",
        "risk_aversion",
        "horizon_sessions",
        "variant",
        "date",
        "model_mode",
    )
    missing_realized = (
        pl.DataFrame(missing_realized_rows).sort(
            "universe",
            "utility",
            "risk_aversion",
            "horizon_sessions",
            "date",
            "model_mode",
        )
        if missing_realized_rows
        else pl.DataFrame()
    )
    portfolio_summary = summarize_portfolio_changes(comparisons)
    delta_timing_summary = summarize_delta_difference_timing(
        comparisons,
        spots,
        horizons_sessions=config.realized_horizons_sessions,
    )
    cross_evaluation_summary = summarize_scenario_cross_evaluation(
        cross_evaluation
    )
    realized_summary, realized_differences = summarize_realized_returns(
        realized,
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results.write_parquet(output_dir / "oce_results.parquet")
    positions.write_parquet(output_dir / "positions.parquet")
    comparisons.write_parquet(output_dir / "portfolio_comparisons.parquet")
    portfolio_summary.write_parquet(output_dir / "portfolio_comparison_summary.parquet")
    delta_timing_summary.write_parquet(output_dir / "delta_difference_timing.parquet")
    cross_evaluation.write_parquet(output_dir / "scenario_cross_evaluation.parquet")
    cross_evaluation_summary.write_parquet(
        output_dir / "scenario_cross_evaluation_summary.parquet"
    )
    realized.write_parquet(output_dir / "realized_delta_returns.parquet")
    if not missing_realized.is_empty():
        missing_realized.write_parquet(output_dir / "missing_realized_delta_returns.parquet")
    realized_summary.write_parquet(output_dir / "realized_delta_summary.parquet")
    realized_differences.write_parquet(
        output_dir / "realized_delta_difference_summary.parquet"
    )
    figure_file = output_dir / "net_delta_counterfactual.png"
    _plot_delta_comparison(comparisons, figure_file)

    report: dict[str, object] = {
        "ticker": ticker,
        "panel_file": str(panel_file),
        "daily_summary_file": str(daily_summary_file),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "methodology": (
            "At every date, fit PCA, matrix-exponential AR, joint spot equation, "
            "and innovation covariance using only observations through that date."
        ),
        "controlled_ablation": (
            "frozen_state sets only G=0; PCA, factor scales, current residual, "
            "spot coefficients, innovation covariance, Sobol nodes, instruments, "
            "entry bid/ask, and optimization constraints are identical"
        ),
        "scenario_horizon_sessions": config.trading.horizon_sessions,
        "realized_strategy": (
            "replace optimized options by signed entry forward deltas; close each "
            "slice at option expiry inside the evaluation horizon, otherwise at "
            "the horizon spot midpoint"
        ),
        "num_requested_dates": len(selected_indices),
        "num_completed_dates": len({row["date"] for row in result_rows}),
        "num_failures": len(failures),
        "failures": failures,
        "num_missing_realized_portfolio_horizons": missing_realized.height,
        "portfolio_comparison_summary": portfolio_summary.to_dicts(),
        "delta_difference_timing_summary": delta_timing_summary.to_dicts(),
        "scenario_cross_evaluation_summary": cross_evaluation_summary.to_dicts(),
        "realized_delta_summary": realized_summary.to_dicts(),
        "realized_delta_difference_summary": realized_differences.to_dicts(),
    }
    (output_dir / "reversion_oce_counterfactual_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    project = Path("aae/ideas/localvolbysanos")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument(
        "--panel-file",
        type=Path,
        default=project / "results/2025_full_reference_grid/balanced_panel.parquet",
    )
    parser.add_argument(
        "--daily-summary-file",
        type=Path,
        default=project / "results/2025_full_reference_grid/daily_summary.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project
        / "real_trading/results/2025/reversion_oce_counterfactual",
    )
    parser.add_argument("--minimum-training-dates", type=int, default=60)
    arguments = parser.parse_args()
    run_reversion_oce_counterfactual(
        arguments.ticker,
        arguments.panel_file,
        arguments.daily_summary_file,
        arguments.output_dir,
        config=ReversionOCECounterfactualConfig(
            minimum_training_dates=arguments.minimum_training_dates
        ),
    )


if __name__ == "__main__":
    main()
