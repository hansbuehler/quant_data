"""Attribute realized OCE portfolio returns to the five-session SPX return."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.bs import bs
from cdxcore.err import verify, verify_inp

from quantdata.biz_calendar import DAYS_PER_YEAR

SERIES_KEYS = ("universe", "utility", "risk_aversion")
PORTFOLIO_KEYS = ("date", *SERIES_KEYS)


@dataclass(frozen=True, kw_only=True)
class SpotAttributionConfig:
    """Regression and tail-diagnostic conventions."""

    hac_lags: int = 4
    top_fraction: float = 0.10

    def __post_init__(self) -> None:
        verify_inp(self.hac_lags >= 0, lambda: "hac_lags must be non-negative")
        verify_inp(
            0.0 < self.top_fraction < 0.5,
            lambda: "top_fraction must lie between zero and one half",
        )


def _hac_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    *,
    lags: int,
) -> np.ndarray:
    """Return Bartlett/Newey-West OLS covariance with population moment scaling."""

    num_observations, num_coefficients = design.shape
    inverse_information = np.linalg.inv(design.T @ design)
    score = design * residual[:, None]
    meat = score.T @ score
    for lag in range(1, min(lags, num_observations - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)
    correction = num_observations / (num_observations - num_coefficients)
    return correction * inverse_information @ meat @ inverse_information


def _regression(
    spot_return: np.ndarray,
    realized_return: np.ndarray,
    *,
    hac_lags: int,
) -> dict[str, object]:
    """Fit linear and quadratic spot regressions and return observation components."""

    design = np.column_stack((np.ones(len(spot_return)), spot_return))
    coefficients = np.linalg.lstsq(design, realized_return, rcond=None)[0]
    fitted = design @ coefficients
    residual = realized_return - fitted
    covariance = _hac_covariance(design, residual, lags=hac_lags)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    total_variation = float(
        np.sum(np.square(realized_return - np.mean(realized_return)))
    )
    residual_variation = float(np.sum(np.square(residual)))
    r_squared = 1.0 - residual_variation / total_variation

    spot_squared = np.square(spot_return)
    quadratic_design = np.column_stack(
        (np.ones(len(spot_return)), spot_return, spot_squared)
    )
    quadratic_coefficients = np.linalg.lstsq(
        quadratic_design, realized_return, rcond=None
    )[0]
    quadratic_residual = realized_return - quadratic_design @ quadratic_coefficients
    quadratic_covariance = _hac_covariance(
        quadratic_design,
        quadratic_residual,
        lags=hac_lags,
    )
    quadratic_standard_errors = np.sqrt(
        np.maximum(np.diag(quadratic_covariance), 0.0)
    )
    quadratic_r_squared = 1.0 - float(
        np.sum(np.square(quadratic_residual))
    ) / total_variation

    beta = float(coefficients[1])
    spot_component = beta * spot_return
    spot_neutral_component = realized_return - spot_component
    quadratic_linear_component = float(quadratic_coefficients[1]) * spot_return
    quadratic_convexity_component = float(quadratic_coefficients[2]) * spot_squared
    quadratic_spot_neutral_component = (
        realized_return - quadratic_linear_component - quadratic_convexity_component
    )
    return {
        "intercept": float(coefficients[0]),
        "spot_beta": beta,
        "intercept_hac_se": float(standard_errors[0]),
        "spot_beta_hac_se": float(standard_errors[1]),
        "spot_beta_hac_t": (
            beta / float(standard_errors[1])
            if standard_errors[1] > 0.0
            else float("nan")
        ),
        "spot_r_squared": r_squared,
        "quadratic_spot_r_squared": quadratic_r_squared,
        "quadratic_linear_spot_coefficient": float(quadratic_coefficients[1]),
        "quadratic_spot_coefficient": float(quadratic_coefficients[2]),
        "quadratic_spot_coefficient_hac_se": float(quadratic_standard_errors[2]),
        "quadratic_spot_coefficient_hac_t": (
            float(quadratic_coefficients[2] / quadratic_standard_errors[2])
            if quadratic_standard_errors[2] > 0.0
            else float("nan")
        ),
        "linear_regression_fitted": fitted,
        "linear_regression_residual": residual,
        "linear_spot_component": spot_component,
        "spot_neutral_component": spot_neutral_component,
        "quadratic_linear_spot_component": quadratic_linear_component,
        "quadratic_convexity_component": quadratic_convexity_component,
        "quadratic_spot_neutral_component": quadratic_spot_neutral_component,
    }


def _linear_attribution_regression(
    explanatory: np.ndarray,
    response: np.ndarray,
    *,
    hac_lags: int,
) -> dict[str, float]:
    design = np.column_stack((np.ones(len(explanatory)), explanatory))
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ coefficients
    covariance = _hac_covariance(design, residual, lags=hac_lags)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    total_variation = float(np.sum(np.square(response - np.mean(response))))
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "slope_hac_se": float(standard_errors[1]),
        "r_squared": 1.0
        - float(np.sum(np.square(residual))) / total_variation,
    }


def _hac_mean_standard_error(values: np.ndarray, *, lags: int) -> float:
    design = np.ones((len(values), 1), dtype=np.float64)
    residual = values - np.mean(values)
    covariance = _hac_covariance(design, residual, lags=lags)
    return float(np.sqrt(max(float(covariance[0, 0]), 0.0)))


def _tail_statistics(
    realized_return: np.ndarray,
    *,
    top_fraction: float,
) -> dict[str, float]:
    positive = realized_return[realized_return > 0.0]
    negative = realized_return[realized_return < 0.0]
    positive_pnl = float(np.sum(positive))
    negative_pnl = float(np.sum(negative))
    top_count = max(1, int(math.ceil(top_fraction * len(realized_return))))
    ordered = np.sort(realized_return)
    top_pnl = float(np.sum(ordered[-top_count:]))
    standard_deviation = float(np.std(realized_return, ddof=0))
    centered = realized_return - np.mean(realized_return)
    skewness = (
        float(np.mean(np.power(centered / standard_deviation, 3.0)))
        if standard_deviation > 0.0
        else float("nan")
    )
    return {
        "positive_fraction": float(np.mean(realized_return > 0.0)),
        "mean_winner_return": (
            float(np.mean(positive)) if len(positive) > 0 else float("nan")
        ),
        "mean_loser_return": (
            float(np.mean(negative)) if len(negative) > 0 else float("nan")
        ),
        "profit_factor": (
            positive_pnl / -negative_pnl if negative_pnl < 0.0 else float("inf")
        ),
        "top_decile_share_of_positive_pnl": (
            top_pnl / positive_pnl if positive_pnl > 0.0 else float("nan")
        ),
        "mean_without_top_decile": float(np.mean(ordered[:-top_count])),
        "realized_skewness": skewness,
        "realized_p05": float(np.quantile(realized_return, 0.05)),
        "realized_p95": float(np.quantile(realized_return, 0.95)),
        "realized_min": float(ordered[0]),
        "realized_max": float(ordered[-1]),
    }


def _entry_position_exposures(positions: pl.DataFrame) -> pl.DataFrame:
    """Add signed forward delta and executable premium cost to every position."""

    strikes = positions["strike_over_forward"].to_numpy()
    mid_prices = 0.5 * (positions["bid"].to_numpy() + positions["ask"].to_numpy())
    is_call = positions["option_type"].to_numpy() == "call"
    sqrt_expiry = np.sqrt(positions["expiry_days"].to_numpy() / DAYS_PER_YEAR)
    implied_volatility = bs.implied(
        k=strikes,
        prices=mid_prices,
        is_call=is_call,
        sqrtT=sqrt_expiry,
        on_exceed_bounds="quiet",
    )
    pure_delta = np.asarray(
        bs.delta(
            k=strikes,
            vol=implied_volatility,
            is_call=is_call,
            sqrtT=sqrt_expiry,
        ),
        dtype=np.float64,
    )
    verify(
        np.all(np.isfinite(implied_volatility)) and np.all(np.isfinite(pure_delta)),
        lambda: "one or more entry options has no finite implied volatility or delta",
    )
    side_sign = np.where(positions["side"].to_numpy() == "long", 1.0, -1.0)
    entry_price = np.where(
        side_sign > 0.0,
        positions["ask"].to_numpy(),
        positions["bid"].to_numpy(),
    )
    allocation = positions["allocation_fraction"].to_numpy()
    divider = positions["divider"].to_numpy()
    return positions.with_columns(
        pl.Series("entry_implied_volatility", implied_volatility),
        pl.Series("entry_pure_forward_delta", pure_delta),
        pl.Series(
            "net_forward_delta_contribution",
            side_sign * allocation * pure_delta / divider,
        ),
        pl.Series(
            "net_entry_cost_contribution",
            side_sign * allocation * entry_price / divider,
        ),
        pl.Series(
            "gross_entry_cost_contribution",
            allocation * entry_price / divider,
        ),
    )


def _entry_exposures(positions: pl.DataFrame) -> pl.DataFrame:
    """Return signed forward delta and executable premium cost by portfolio."""

    enriched = _entry_position_exposures(positions)
    exposures = enriched.group_by(*PORTFOLIO_KEYS).agg(
        pl.col("net_forward_delta_contribution").sum().alias("net_forward_delta"),
        pl.col("net_entry_cost_contribution").sum().alias(
            "net_entry_cost_fraction"
        ),
        pl.col("gross_entry_cost_contribution").sum().alias(
            "gross_entry_cost_fraction"
        ),
    )
    return exposures.with_columns(
        (pl.col("net_forward_delta") / pl.col("gross_entry_cost_fraction")).alias(
            "net_forward_delta_per_gross_cost"
        ),
        pl.when(pl.col("net_entry_cost_fraction").abs() > 1.0e-8)
        .then(pl.col("net_forward_delta") / pl.col("net_entry_cost_fraction").abs())
        .otherwise(None)
        .alias("net_forward_delta_per_abs_net_cost"),
    )


def analyze_spot_attribution(
    comparison_file: Path,
    positions_file: Path,
    spot_file: Path,
    output_dir: Path,
    *,
    config: SpotAttributionConfig = SpotAttributionConfig(),
) -> dict[str, object]:
    """Regress each complete realized portfolio series on its horizon SPX return."""

    comparison = pl.read_parquet(comparison_file).filter(
        pl.col("complete") & pl.col("realized_return").is_not_null()
    )
    exposures = _entry_exposures(pl.read_parquet(positions_file))
    spot = (
        pl.read_parquet(spot_file)
        .select("date", "spot_ref")
        .unique("date")
        .sort("date")
    )
    current_spot = spot.rename({"spot_ref": "current_spot_ref"})
    horizon_spot = spot.rename(
        {"date": "horizon_date", "spot_ref": "horizon_spot_ref"}
    )
    observations = (
        comparison.join(
            exposures,
            on=list(PORTFOLIO_KEYS),
            how="left",
            validate="1:1",
        )
        .join(current_spot, on="date", how="left", validate="m:1")
        .join(horizon_spot, on="horizon_date", how="left", validate="m:1")
        .with_columns(
            (pl.col("horizon_spot_ref") / pl.col("current_spot_ref") - 1.0).alias(
                "horizon_spot_return"
            )
        )
    )
    verify(
        observations.select(
            pl.all_horizontal(
                pl.col("current_spot_ref").is_not_null(),
                pl.col("horizon_spot_ref").is_not_null(),
                pl.col("horizon_spot_return").is_finite(),
            ).all()
        ).item(),
        lambda: "one or more comparison dates have no finite SPX spot return",
    )

    summary_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    for key, group in observations.partition_by(*SERIES_KEYS, as_dict=True).items():
        group = group.sort("date")
        spot_return = group["horizon_spot_return"].to_numpy()
        realized_return = group["realized_return"].to_numpy()
        verify(
            len(realized_return) >= 20
            and np.std(spot_return, ddof=0) > 0.0
            and np.std(realized_return, ddof=0) > 0.0,
            lambda: f"series {key} is too small or constant for attribution",
        )
        regression = _regression(
            spot_return,
            realized_return,
            hac_lags=config.hac_lags,
        )
        spot_component = np.asarray(regression["linear_spot_component"])
        neutral_component = np.asarray(regression["spot_neutral_component"])
        quadratic_linear_component = np.asarray(
            regression["quadratic_linear_spot_component"]
        )
        quadratic_convexity_component = np.asarray(
            regression["quadratic_convexity_component"]
        )
        quadratic_neutral_component = np.asarray(
            regression["quadratic_spot_neutral_component"]
        )
        entry_delta_spot_component = (
            group["net_forward_delta"].to_numpy() * spot_return
        )
        entry_delta_spot_residual = realized_return - entry_delta_spot_component
        entry_delta_regression = _linear_attribution_regression(
            entry_delta_spot_component,
            realized_return,
            hac_lags=config.hac_lags,
        )
        mean_entry_delta = float(group["net_forward_delta"].mean())
        fixed_delta_spot_component = mean_entry_delta * spot_return
        delta_timing_component = (
            entry_delta_spot_component - fixed_delta_spot_component
        )
        timing_hac_se = _hac_mean_standard_error(
            delta_timing_component,
            lags=config.hac_lags,
        )
        dynamic_delta_std = float(np.std(entry_delta_spot_component, ddof=0))
        spot_std = float(np.std(spot_return, ddof=0))
        risk_matched_fixed_delta = dynamic_delta_std / spot_std
        risk_matched_fixed_spot_component = risk_matched_fixed_delta * spot_return
        full_strategy_std = float(np.std(realized_return, ddof=0))
        full_strategy_risk_matched_delta = full_strategy_std / spot_std
        full_strategy_risk_matched_spot_component = (
            full_strategy_risk_matched_delta * spot_return
        )
        summary_rows.append(
            {
                "universe": key[0],
                "utility": key[1],
                "risk_aversion": key[2],
                "num_observations": len(group),
                "mean_spot_return": float(np.mean(spot_return)),
                "mean_realized_return": float(np.mean(realized_return)),
                "median_realized_return": float(np.median(realized_return)),
                "realized_return_std": float(np.std(realized_return, ddof=0)),
                "mean_net_forward_delta": mean_entry_delta,
                "median_net_forward_delta": float(
                    group["net_forward_delta"].median()
                ),
                "mean_net_entry_cost_fraction": float(
                    group["net_entry_cost_fraction"].mean()
                ),
                "median_net_entry_cost_fraction": float(
                    group["net_entry_cost_fraction"].median()
                ),
                "mean_gross_entry_cost_fraction": float(
                    group["gross_entry_cost_fraction"].mean()
                ),
                "mean_net_forward_delta_per_gross_cost": float(
                    group["net_forward_delta_per_gross_cost"].mean()
                ),
                "median_net_forward_delta_per_abs_net_cost": float(
                    group["net_forward_delta_per_abs_net_cost"].median()
                ),
                "mean_entry_delta_spot_component": float(
                    np.mean(entry_delta_spot_component)
                ),
                "mean_entry_delta_spot_residual": float(
                    np.mean(entry_delta_spot_residual)
                ),
                "entry_delta_spot_correlation": float(
                    np.corrcoef(entry_delta_spot_component, realized_return)[0, 1]
                ),
                "entry_delta_spot_r_squared": entry_delta_regression["r_squared"],
                "entry_delta_calibration_intercept": entry_delta_regression[
                    "intercept"
                ],
                "entry_delta_calibration_slope": entry_delta_regression["slope"],
                "entry_delta_calibration_slope_hac_se": entry_delta_regression[
                    "slope_hac_se"
                ],
                "mean_fixed_delta_spot_component": float(
                    np.mean(fixed_delta_spot_component)
                ),
                "mean_delta_timing_component": float(
                    np.mean(delta_timing_component)
                ),
                "delta_timing_hac_se": timing_hac_se,
                "delta_timing_hac_t": (
                    float(np.mean(delta_timing_component)) / timing_hac_se
                    if timing_hac_se > 0.0
                    else float("nan")
                ),
                "entry_delta_future_spot_correlation": float(
                    np.corrcoef(
                        group["net_forward_delta"].to_numpy(), spot_return
                    )[0, 1]
                ),
                "entry_delta_spot_component_std": dynamic_delta_std,
                "fixed_delta_spot_component_std": float(
                    np.std(fixed_delta_spot_component, ddof=0)
                ),
                "risk_matched_fixed_delta": risk_matched_fixed_delta,
                "mean_risk_matched_fixed_spot_component": float(
                    np.mean(risk_matched_fixed_spot_component)
                ),
                "dynamic_delta_return_to_risk": float(
                    np.mean(entry_delta_spot_component) / dynamic_delta_std
                ),
                "fixed_spot_return_to_risk": float(
                    np.mean(spot_return) / spot_std
                ),
                "full_strategy_risk_matched_fixed_delta": (
                    full_strategy_risk_matched_delta
                ),
                "mean_full_strategy_risk_matched_spot_component": float(
                    np.mean(full_strategy_risk_matched_spot_component)
                ),
                "full_strategy_return_to_risk": float(
                    np.mean(realized_return) / full_strategy_std
                ),
                "spot_beta": regression["spot_beta"],
                "spot_beta_hac_se": regression["spot_beta_hac_se"],
                "spot_beta_hac_t": regression["spot_beta_hac_t"],
                "spot_r_squared": regression["spot_r_squared"],
                "quadratic_spot_r_squared": regression[
                    "quadratic_spot_r_squared"
                ],
                "quadratic_linear_spot_coefficient": regression[
                    "quadratic_linear_spot_coefficient"
                ],
                "quadratic_spot_coefficient": regression[
                    "quadratic_spot_coefficient"
                ],
                "quadratic_spot_coefficient_hac_se": regression[
                    "quadratic_spot_coefficient_hac_se"
                ],
                "quadratic_spot_coefficient_hac_t": regression[
                    "quadratic_spot_coefficient_hac_t"
                ],
                "mean_linear_spot_component": float(np.mean(spot_component)),
                "mean_spot_neutral_component": float(np.mean(neutral_component)),
                "mean_abs_linear_spot_component": float(
                    np.mean(np.abs(spot_component))
                ),
                "mean_abs_spot_neutral_component": float(
                    np.mean(np.abs(neutral_component))
                ),
                "mean_quadratic_linear_spot_component": float(
                    np.mean(quadratic_linear_component)
                ),
                "mean_quadratic_convexity_component": float(
                    np.mean(quadratic_convexity_component)
                ),
                "mean_quadratic_spot_neutral_component": float(
                    np.mean(quadratic_neutral_component)
                ),
                **_tail_statistics(
                    realized_return,
                    top_fraction=config.top_fraction,
                ),
            }
        )
        derived_columns = {
            name: np.asarray(regression[name])
            for name in (
                "linear_regression_fitted",
                "linear_regression_residual",
                "linear_spot_component",
                "spot_neutral_component",
                "quadratic_linear_spot_component",
                "quadratic_convexity_component",
                "quadratic_spot_neutral_component",
            )
        }
        derived_columns["entry_delta_spot_component"] = entry_delta_spot_component
        derived_columns["entry_delta_spot_residual"] = entry_delta_spot_residual
        derived_columns["fixed_delta_spot_component"] = fixed_delta_spot_component
        derived_columns["delta_timing_component"] = delta_timing_component
        derived_columns["risk_matched_fixed_spot_component"] = (
            risk_matched_fixed_spot_component
        )
        derived_columns["full_strategy_risk_matched_spot_component"] = (
            full_strategy_risk_matched_spot_component
        )
        for index, row in enumerate(group.to_dicts()):
            observation_rows.append(
                {
                    **row,
                    **{
                        name: float(values[index])
                        for name, values in derived_columns.items()
                    },
                }
            )

    summary = pl.DataFrame(summary_rows).sort(*SERIES_KEYS)
    attributed = pl.DataFrame(observation_rows, infer_schema_length=None).sort(
        *SERIES_KEYS, "date"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(output_dir / "spot_attribution_summary.parquet")
    attributed.write_parquet(output_dir / "spot_attribution_observations.parquet")
    report = {
        "comparison_file": str(comparison_file),
        "positions_file": str(positions_file),
        "spot_file": str(spot_file),
        "output_dir": str(output_dir),
        "spot_return": "SPX spot_ref simple return from entry to five-session horizon",
        "regression": "realized_return = intercept + beta * spot_return + residual",
        "standard_errors": (
            f"Newey-West/Bartlett HAC with {config.hac_lags} lags for overlapping returns"
        ),
        "mean_attribution": (
            "linear: mean return = beta * mean spot return + mean spot-neutral; "
            "quadratic: mean return = mean linear spot + mean convexity + mean "
            "quadratic spot-neutral"
        ),
        "residual_warning": (
            "spot-neutral component is not pure volatility; it includes nonlinear spot, "
            "volatility, theta, entry spread, and interaction effects"
        ),
        "entry_exposure": (
            "net forward delta is P&L fraction per unit relative forward move; "
            "net cost is positive for executable premium debit and negative for credit"
        ),
        "entry_delta_attribution": (
            "entry delta spot component = net forward delta * horizon SPX return; "
            "the residual contains gamma, volatility, theta, spread, and interactions"
        ),
        "fixed_spot_benchmark": (
            "dynamic entry-delta P&L = mean entry delta * horizon SPX return + "
            "(entry delta - mean entry delta) * horizon SPX return; the second "
            "term is the in-sample delta-timing component"
        ),
        "full_strategy_spot_benchmark": (
            "a second fixed long-SPX benchmark is scaled to the population standard "
            "deviation of each complete realized option-return series; funding and "
            "spot trading costs are ignored"
        ),
        "summary": summary.to_dicts(),
    }
    (output_dir / "spot_attribution_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-file",
        type=Path,
        default=Path(
            "quantdata/dysanos/localvolbysanos/real_trading/results/2025/"
            "realized_comparison.parquet"
        ),
    )
    parser.add_argument(
        "--positions-file",
        type=Path,
        default=Path(
            "quantdata/dysanos/localvolbysanos/real_trading/results/2025/positions.parquet"
        ),
    )
    parser.add_argument(
        "--spot-file",
        type=Path,
        default=Path(
            "quantdata/dysanos/localvolbysanos/results/2025_full_reference_grid/"
            "daily_summary.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quantdata/dysanos/localvolbysanos/real_trading/results/2025"),
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = analyze_spot_attribution(
        arguments.comparison_file,
        arguments.positions_file,
        arguments.spot_file,
        arguments.output_dir,
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
