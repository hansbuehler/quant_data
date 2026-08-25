"""Compare conditional OCE forecasts with five-session realized option returns."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context

from aae.dysanos.markets import DysanosMarketDay, ListedOptionBook, load_market_day
from quantdata.biz_calendar import DAYS_PER_YEAR

PORTFOLIO_KEYS = ("date", "universe", "utility", "risk_aversion")


@dataclass(frozen=True, kw_only=True)
class RealizedComparisonConfig:
    """Timing and matching controls for the realized-return comparison."""

    horizon_sessions: int = 5
    strike_absolute_tolerance: float = 1.0e-3
    strike_relative_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        verify_inp(
            self.horizon_sessions >= 1,
            lambda: "horizon_sessions must be positive",
        )
        verify_inp(
            self.strike_absolute_tolerance > 0.0,
            lambda: "strike_absolute_tolerance must be positive",
        )
        verify_inp(
            self.strike_relative_tolerance > 0.0,
            lambda: "strike_relative_tolerance must be positive",
        )


def _expiry_indices(book: ListedOptionBook, expiry_date: date) -> np.ndarray:
    return np.flatnonzero(
        np.fromiter(
            (candidate == expiry_date for candidate in book.expiry_dates),
            dtype=bool,
            count=len(book.expiry_dates),
        )
    )


def _nearest_index(values: np.ndarray, target: float) -> tuple[int, float]:
    index = int(np.argmin(np.abs(values - target)))
    return index, float(abs(values[index] - target))


def _absolute_contract(
    row: dict[str, object],
    current: DysanosMarketDay,
    *,
    config: RealizedComparisonConfig,
) -> tuple[float, float, float, int, bool]:
    """Return absolute strike, scale, strike error, expiry ID, and AM flag."""

    book = current.listed_options
    verify(book is not None, lambda: "current listed-option details are unavailable")
    expiry_date = row["expiry_date"]
    assert isinstance(expiry_date, date)
    expiry_indices = _expiry_indices(book, expiry_date)
    verify(len(expiry_indices) > 0, lambda: f"current expiry {expiry_date} is missing")
    strike_over_forward = float(row["strike_over_forward"])
    target_expiry_days = int(row["expiry_days"])
    expiry_index = min(
        expiry_indices,
        key=lambda index: (
            abs(book.expiries[index] * DAYS_PER_YEAR - target_expiry_days),
            _nearest_index(
                book.strikes_over_forward[index], strike_over_forward
            )[1],
        ),
    )
    strike_index, error = _nearest_index(
        book.strikes_over_forward[expiry_index], strike_over_forward
    )
    tolerance = max(
        config.strike_absolute_tolerance / book.forwards[expiry_index],
        config.strike_relative_tolerance * abs(strike_over_forward),
    )
    verify(
        error <= tolerance,
        lambda: (
            f"current normalized strike {strike_over_forward} is missing; "
            f"nearest error is {error}"
        ),
    )
    absolute_strike = float(book.absolute_strikes[expiry_index][strike_index])
    discounted_forward = float(
        book.discount_factors[expiry_index] * book.forwards[expiry_index]
    )
    return (
        absolute_strike,
        discounted_forward,
        error,
        int(book.expiry_ids[expiry_index]),
        bool(book.is_am[expiry_index]),
    )


def _future_mid(
    row: dict[str, object],
    absolute_strike: float,
    expiry_id: int,
    future: DysanosMarketDay,
    *,
    config: RealizedComparisonConfig,
) -> tuple[float | None, bool, float | None, str | None]:
    """Return the exact contract's observed future mid in index-point dollars."""

    book = future.listed_options
    if book is None:
        return None, False, None, "future listed-option details unavailable"
    expiry_indices = np.flatnonzero(book.expiry_ids == expiry_id)
    if len(expiry_indices) == 0:
        return None, False, None, f"future expiry ID {expiry_id} missing"
    expiry_index = int(expiry_indices.item())
    strike_index, error = _nearest_index(
        book.absolute_strikes[expiry_index], absolute_strike
    )
    tolerance = max(
        config.strike_absolute_tolerance,
        config.strike_relative_tolerance * abs(absolute_strike),
    )
    if error > tolerance:
        return None, False, error, f"future absolute strike mismatch {error:.6g}"

    is_call = str(row["option_type"]) == "call"
    bids = book.call_bids[expiry_index] if is_call else book.put_bids[expiry_index]
    asks = book.call_asks[expiry_index] if is_call else book.put_asks[expiry_index]
    active = book.call_active[expiry_index] if is_call else book.put_active[expiry_index]
    bid = float(bids[strike_index])
    ask = float(asks[strike_index])
    if not np.isfinite(bid) or not np.isfinite(ask) or bid < 0.0 or ask <= bid:
        return None, bool(active[strike_index]), error, "invalid future bid/ask"
    discounted_forward = float(
        book.discount_factors[expiry_index] * book.forwards[expiry_index]
    )
    return 0.5 * (bid + ask) * discounted_forward, bool(active[strike_index]), error, None


def _settlement_proxy(
    row: dict[str, object],
    absolute_strike: float,
    expiry_market: DysanosMarketDay | None,
) -> tuple[float | None, str | None]:
    if expiry_market is None:
        return None, "expiry-date SPX spot is unavailable"
    if str(row["option_type"]) == "call":
        return max(expiry_market.spot_ref - absolute_strike, 0.0), None
    return max(absolute_strike - expiry_market.spot_ref, 0.0), None


def _position_realized(
    row: dict[str, object],
    current: DysanosMarketDay,
    future: DysanosMarketDay,
    expiry_market: DysanosMarketDay | None,
    *,
    config: RealizedComparisonConfig,
) -> dict[str, object]:
    try:
        (
            absolute_strike,
            current_scale,
            current_match_error,
            expiry_id,
            is_am,
        ) = _absolute_contract(row, current, config=config)
        expiry_date = row["expiry_date"]
        assert isinstance(expiry_date, date)
        if expiry_date <= future.market_date:
            exit_price, error = _settlement_proxy(row, absolute_strike, expiry_market)
            future_active = None
            future_match_error = None
            exit_kind = "spot_settlement_proxy"
        else:
            exit_price, future_active, future_match_error, error = _future_mid(
                row, absolute_strike, expiry_id, future, config=config
            )
            exit_kind = "observed_mid"
        if error is not None or exit_price is None:
            return {
                "available": False,
                "error": error,
                "absolute_strike": absolute_strike,
                "expiry_id": expiry_id,
                "is_am": is_am,
                "current_match_error": current_match_error,
                "future_match_error": future_match_error,
                "future_active": future_active,
                "exit_kind": exit_kind,
                "entry_price_absolute": None,
                "divider_absolute": None,
                "exit_price_absolute": None,
                "position_return": None,
                "realized_contribution": None,
            }

        bid_absolute = float(row["bid"]) * current_scale
        ask_absolute = float(row["ask"]) * current_scale
        divider_absolute = float(row["divider"]) * current_scale
        side = str(row["side"])
        if side == "long":
            entry_price = ask_absolute
            position_return = (exit_price - entry_price) / divider_absolute
        else:
            entry_price = bid_absolute
            position_return = (entry_price - exit_price) / divider_absolute
        contribution = float(row["allocation_fraction"]) * position_return
        return {
            "available": True,
            "error": None,
            "absolute_strike": absolute_strike,
            "expiry_id": expiry_id,
            "is_am": is_am,
            "current_match_error": current_match_error,
            "future_match_error": future_match_error,
            "future_active": future_active,
            "exit_kind": exit_kind,
            "entry_price_absolute": entry_price,
            "divider_absolute": divider_absolute,
            "exit_price_absolute": exit_price,
            "position_return": position_return,
            "realized_contribution": contribution,
        }
    except Exception as exception:
        return {
            "available": False,
            "error": f"{type(exception).__name__}: {exception}",
            "absolute_strike": None,
            "expiry_id": None,
            "is_am": None,
            "current_match_error": None,
            "future_match_error": None,
            "future_active": None,
            "exit_kind": None,
            "entry_price_absolute": None,
            "divider_absolute": None,
            "exit_price_absolute": None,
            "position_return": None,
            "realized_contribution": None,
        }


def _load_markets(
    ticker: str,
    dates: list[date],
    *,
    detailed_dates: set[date],
) -> list[DysanosMarketDay | None]:
    markets: list[DysanosMarketDay | None] = []
    for index, market_date in enumerate(dates):
        markets.append(
            load_market_day(
                ticker,
                market_date,
                include_contract_details=market_date in detailed_dates,
                verbose=Context.quiet,
            )
        )
        if (index + 1) % 25 == 0 or index + 1 == len(dates):
            print(f"loaded markets {index + 1}/{len(dates)}", flush=True)
    return markets


def _summary(comparison: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in comparison.partition_by(
        "universe", "utility", "risk_aversion", as_dict=True
    ).items():
        complete = group.filter(pl.col("complete"))
        predicted = complete["expected_return"].to_numpy()
        realized = complete["realized_return"].to_numpy()
        error = realized - predicted
        correlation = (
            float(np.corrcoef(predicted, realized)[0, 1])
            if len(complete) >= 2
            and np.std(predicted) > 0.0
            and np.std(realized) > 0.0
            else float("nan")
        )
        slope = (
            float(np.cov(predicted, realized, ddof=0)[0, 1] / np.var(predicted))
            if len(complete) >= 2 and np.var(predicted) > 0.0
            else float("nan")
        )
        rows.append(
            {
                "universe": key[0],
                "utility": key[1],
                "risk_aversion": key[2],
                "num_portfolios": group.height,
                "num_complete": complete.height,
                "coverage": complete.height / group.height,
                "mean_predicted_return": float(np.mean(predicted)),
                "median_predicted_return": float(np.median(predicted)),
                "mean_realized_return": float(np.mean(realized)),
                "median_realized_return": float(np.median(realized)),
                "mean_error_realized_minus_predicted": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "correlation": correlation,
                "calibration_slope": slope,
                "realized_positive_fraction": float(np.mean(realized > 0.0)),
                "direction_accuracy": float(
                    np.mean(np.sign(realized) == np.sign(predicted))
                ),
                "mean_settlement_proxy_positions": float(
                    np.mean(complete["num_settlement_proxy_positions"].to_numpy())
                ),
            }
        )
    return pl.DataFrame(rows).sort("universe", "utility", "risk_aversion")


def compare_realized_returns(
    ticker: str,
    results_dir: Path,
    output_dir: Path,
    *,
    config: RealizedComparisonConfig = RealizedComparisonConfig(),
) -> dict[str, object]:
    """Compare predicted scenario means with exact-contract observed mid returns."""

    results = pl.read_parquet(results_dir / "oce_results.parquet")
    positions = pl.read_parquet(results_dir / "positions.parquet")
    dates = sorted(results["date"].unique().to_list())
    date_index = {market_date: index for index, market_date in enumerate(dates)}
    load_dates = dates
    loaded_markets = _load_markets(
        ticker,
        load_dates,
        detailed_dates=set(dates),
    )
    market_by_date = dict(zip(load_dates, loaded_markets, strict=True))
    markets = [market_by_date[market_date] for market_date in dates]
    date_index = {market_date: index for index, market_date in enumerate(dates)}

    position_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(positions.to_dicts()):
        current_date = row["date"]
        assert isinstance(current_date, date)
        current_index = date_index[current_date]
        if current_index + config.horizon_sessions >= len(dates):
            continue
        horizon_date = dates[current_index + config.horizon_sessions]
        current = markets[current_index]
        future = markets[current_index + config.horizon_sessions]
        if current is None or future is None:
            realized = {
                "available": False,
                "error": "current or future market unavailable",
            }
        else:
            expiry_date = row["expiry_date"]
            assert isinstance(expiry_date, date)
            realized = _position_realized(
                row,
                current,
                future,
                market_by_date.get(expiry_date),
                config=config,
            )
        position_rows.append(
            {
                **row,
                "horizon_date": horizon_date,
                **realized,
            }
        )
        if (row_index + 1) % 2_500 == 0 or row_index + 1 == positions.height:
            print(
                f"valued positions {row_index + 1}/{positions.height}", flush=True
            )

    position_comparison = pl.DataFrame(position_rows, infer_schema_length=None)
    portfolio = position_comparison.group_by(*PORTFOLIO_KEYS).agg(
        pl.col("horizon_date").first(),
        pl.col("available").all().alias("complete"),
        pl.len().alias("num_positions_evaluated"),
        (~pl.col("available")).sum().alias("num_missing_positions"),
        (pl.col("exit_kind") == "spot_settlement_proxy")
        .sum()
        .alias("num_settlement_proxy_positions"),
        pl.col("realized_contribution").sum().alias("realized_return_partial"),
    )
    portfolio = portfolio.with_columns(
        pl.when(pl.col("complete"))
        .then(pl.col("realized_return_partial"))
        .otherwise(None)
        .alias("realized_return")
    )
    comparison = (
        results.join(portfolio, on=list(PORTFOLIO_KEYS), how="left")
        .filter(pl.col("date_index") + config.horizon_sessions < len(dates))
        .with_columns(
            (pl.col("realized_return") - pl.col("expected_return")).alias(
                "error_realized_minus_predicted"
            ),
            (pl.col("realized_return") * pl.lit(1_000_000.0)).alias(
                "realized_pnl_per_million"
            ),
        )
        .sort(*PORTFOLIO_KEYS)
    )
    summary = _summary(comparison)

    output_dir.mkdir(parents=True, exist_ok=True)
    position_comparison.write_parquet(output_dir / "realized_positions.parquet")
    comparison.write_parquet(output_dir / "realized_comparison.parquet")
    summary.write_parquet(output_dir / "realized_summary.parquet")

    missing_errors = (
        position_comparison.filter(~pl.col("available"))
        .group_by("error")
        .len()
        .sort("len", descending=True)
        .to_dicts()
    )
    report = {
        "ticker": ticker,
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "entry": "observed ask for longs and observed bid for shorts",
        "exit": "observed future mid for survivors",
        "expiry": "intrinsic payoff using expiry-date SPX spot_ref when available",
        "funding": "ignored, matching the prediction run",
        "horizon_sessions": config.horizon_sessions,
        "num_comparisons": comparison.height,
        "num_complete": int(comparison["complete"].fill_null(False).sum()),
        "num_missing_positions": int((~position_comparison["available"]).sum()),
        "missing_errors": missing_errors,
        "summary": summary.to_dicts(),
    }
    (output_dir / "realized_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("aae/ideas/localvolbysanos/real_trading/results/2025"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("aae/ideas/localvolbysanos/real_trading/results/2025"),
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = compare_realized_returns(
        arguments.ticker,
        arguments.results_dir,
        arguments.output_dir,
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
