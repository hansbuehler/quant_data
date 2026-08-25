"""Market bid/ask fit diagnostics for annual local-volatility samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context

from quantdata.dysanos.localvolbysanos.markets import MarketConfig, load_market_day
from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.pool import PoolConfig


@dataclass(frozen=True)
class MarketFitDay:
    """Quote-level spread errors and daily summaries for one market."""

    daily: dict[str, object]
    expiry_rows: list[dict[str, object]]
    quote_expiry_days: np.ndarray
    quote_strikes: np.ndarray
    quote_errors: np.ndarray


def _error_summary(errors: np.ndarray) -> dict[str, float | int]:
    outside = errors > 0.0
    outside_errors = errors[outside]
    return {
        "num_quotes": len(errors),
        "num_inside_bid_ask": int(np.count_nonzero(~outside)),
        "percent_inside_bid_ask": float(100.0 * np.mean(~outside)),
        "mean_error_all": float(np.mean(errors)),
        "median_error_all": float(np.median(errors)),
        "mean_error_outside": (
            float(np.mean(outside_errors)) if len(outside_errors) else 0.0
        ),
        "median_error_outside": (
            float(np.median(outside_errors)) if len(outside_errors) else 0.0
        ),
    }


def _fit_date(
    ticker: str,
    market_date: date,
    *,
    market_config: MarketConfig,
) -> tuple[MarketFitDay | None, str | None]:
    try:
        day = load_market_day(
            ticker,
            market_date,
            config=market_config,
            verbose=Context.quiet,
        )
        verify(day is not None, lambda: f"no filtered market for {ticker} {market_date}")
        fit = day.af_pricer.compute_market_fit(
            expiries=day.expiries,
            strikes=list(day.strikes),
            bids=list(day.bids),
            asks=list(day.asks),
            masks=list(day.model_masks),
            and_vols=False,
        )
        errors = np.concatenate(fit.fit_errs).astype(np.float64, copy=False)
        expiry_days = np.rint(day.expiries * DAYS_PER_YEAR).astype(np.int16)
        quote_expiry_days = np.concatenate(
            [
                np.full(len(values), expiry_days[index], dtype=np.int16)
                for index, values in enumerate(fit.fit_errs)
            ]
        )
        quote_strikes = np.concatenate(day.strikes).astype(np.float64, copy=False)
        expiry_rows: list[dict[str, object]] = []
        for expiry, day_count, values in zip(
            day.expiries,
            expiry_days,
            fit.fit_errs,
            strict=True,
        ):
            expiry_rows.append(
                {
                    "date": market_date,
                    "expiry_years": float(expiry),
                    "expiry_days": int(day_count),
                    **_error_summary(np.asarray(values, dtype=np.float64)),
                }
            )
        return (
            MarketFitDay(
                daily={"date": market_date, **_error_summary(errors)},
                expiry_rows=expiry_rows,
                quote_expiry_days=quote_expiry_days,
                quote_strikes=quote_strikes,
                quote_errors=errors,
            ),
            None,
        )
    except Exception as exception:
        return None, f"{type(exception).__name__}: {exception}"


def _plot_time_series(daily: pl.DataFrame, *, output_file: Path) -> None:
    import matplotlib.pyplot as plt

    dates = daily["date"].to_numpy()
    inside = daily["percent_inside_bid_ask"].to_numpy()
    mean_error = daily["mean_error_outside"].to_numpy()
    median_error = daily["median_error_outside"].to_numpy()
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(dates, inside, linewidth=1.2)
    axes[0].axhline(float(np.mean(inside)), linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("quotes inside bid/ask (%)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(dates, mean_error, label="mean", linewidth=1.2)
    axes[1].plot(dates, median_error, label="median", linewidth=1.2)
    axes[1].set_ylabel("error outside spread\n(half-spread units)")
    axes[1].set_xlabel("market date")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("SPX arbitrage-free surface fit to retained quotes, 2025")
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def run_market_fit_diagnostics(
    *,
    ticker: str = "SPX",
    sample_file: Path = Path(
        "quantdata/dysanos/localvolbysanos/results/2025_full_reference_grid/"
        "balanced_panel.parquet"
    ),
    output_directory: Path = Path(
        "quantdata/dysanos/localvolbysanos/results/2025_full_reference_grid"
    ),
    market_config: MarketConfig = MarketConfig(smoothness=0.25),
    workers: int = 4,
) -> dict[str, object]:
    """Measure the AF surface fit on every retained quote in the annual sample."""

    verify_inp(workers >= 1, lambda: "workers must be positive")
    sample = pl.scan_parquet(sample_file).select("date").unique().collect()
    dates = sorted(sample["date"].to_list())
    verify(len(dates) > 0, lambda: f"no dates found in {sample_file}")
    pool = PoolConfig(num_workers=workers, threading=False).pool()
    jobs = {
        market_date: pool.delayed(_fit_date)(
            ticker,
            market_date,
            market_config=market_config,
        )
        for market_date in dates
    }
    completed: list[MarketFitDay] = []
    failures: list[dict[str, str]] = []
    for index, (market_date, result, error) in enumerate(pool.parallel(jobs), start=1):
        if result is None:
            assert error is not None
            failures.append({"date": market_date.isoformat(), "error": error})
        else:
            assert error is None
            completed.append(result)
        if index % 25 == 0 or index == len(dates):
            print(
                f"market fit {index}/{len(dates)} "
                f"completed={len(completed)} failed={len(failures)}",
                flush=True,
            )
    completed.sort(key=lambda item: item.daily["date"])
    verify(len(completed) > 0, lambda: "no market-fit days completed")
    daily = pl.DataFrame([item.daily for item in completed]).sort("date")
    by_expiry = pl.DataFrame(
        [row for item in completed for row in item.expiry_rows]
    ).sort("date", "expiry_days")
    quote_frames: list[pl.DataFrame] = []
    for item in completed:
        num_quotes = len(item.quote_errors)
        quote_frames.append(
            pl.DataFrame(
                {
                    "date": [item.daily["date"]] * num_quotes,
                    "expiry_days": item.quote_expiry_days,
                    "forward_relative_strike": item.quote_strikes,
                    "spread_error": item.quote_errors,
                    "inside_bid_ask": item.quote_errors == 0.0,
                }
            )
        )
    quotes = pl.concat(quote_frames)
    monthly = (
        quotes.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(
            pl.len().alias("num_quotes"),
            (100.0 * pl.col("inside_bid_ask").mean()).alias(
                "percent_inside_bid_ask"
            ),
            pl.col("spread_error").mean().alias("mean_error_all"),
            pl.col("spread_error").median().alias("median_error_all"),
            pl.col("spread_error")
            .filter(~pl.col("inside_bid_ask"))
            .mean()
            .alias("mean_error_outside"),
            pl.col("spread_error")
            .filter(~pl.col("inside_bid_ask"))
            .median()
            .alias("median_error_outside"),
        )
        .sort("month")
    )
    errors = quotes["spread_error"].to_numpy()
    aggregate = _error_summary(errors)
    inside = daily["percent_inside_bid_ask"].to_numpy()
    worst_index = int(np.argmin(inside))
    report: dict[str, object] = {
        "ticker": ticker,
        "dates_requested": len(dates),
        "dates_completed": len(completed),
        "failures": failures,
        "error_definition": (
            "distance outside bid/ask divided by max(half-spread, 1e-4); "
            "zero for a fit inside bid/ask"
        ),
        "aggregate_quote_weighted": aggregate,
        "daily_percent_inside_bid_ask": {
            "mean": float(np.mean(inside)),
            "median": float(np.median(inside)),
            "minimum": float(np.min(inside)),
            "maximum": float(np.max(inside)),
            "worst_date": daily["date"][worst_index].isoformat(),
        },
        "monthly": monthly.to_dicts(),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(output_directory / "market_fit_daily.parquet", compression="zstd")
    by_expiry.write_parquet(
        output_directory / "market_fit_by_expiry.parquet", compression="zstd"
    )
    quotes.write_parquet(
        output_directory / "market_fit_quote_errors.parquet", compression="zstd"
    )
    monthly.write_parquet(
        output_directory / "market_fit_monthly.parquet", compression="zstd"
    )
    with (output_directory / "market_fit_report.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    _plot_time_series(daily, output_file=output_directory / "market_fit_time.png")
    return report


if __name__ == "__main__":
    completed_report = run_market_fit_diagnostics()
    print(json.dumps(completed_report, indent=2))
