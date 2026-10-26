"""Compare historical GROMM2 30-calendar-day variance with official VIX."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import ndtr

from quantdata import SubDir, env
from quantdata.biz_calendar import AM_EXIRY_SUBTRACT, DAYS_PER_YEAR, get_business_days
from quantdata.gromm2.market import solve_sanos_step_numpy
from quantdata.vixdata import load_cboe_vix
from quantdata.vixdata.formula import (
    constant_maturity_vix,
    discrete_total_variance,
    select_vix_strip,
)


def _density_at_expiry(
    *,
    supports: np.ndarray,
    transports: np.ndarray,
    expiries_y: np.ndarray,
    dlvsq: np.ndarray,
    expiry_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one DLV density at an arbitrary maturity on its upper support."""
    if not 0.0 < expiry_y <= float(expiries_y[-1]):
        raise ValueError("expiry must lie inside the fixed-expiry model horizon")
    upper = int(np.searchsorted(expiries_y, expiry_y, side="left"))
    previous = _initial_density(supports[0])
    previous_expiry = 0.0
    for index in range(upper + 1):
        if index > 0:
            previous = transports[index - 1] @ previous
        current_expiry = min(float(expiries_y[index]), expiry_y)
        interval = current_expiry - previous_expiry
        if interval <= 0.0:
            raise RuntimeError("non-positive DLV interval")
        previous = solve_sanos_step_numpy(
            supports[index], previous, dlvsq[index] * interval
        )
        previous_expiry = current_expiry
        if current_expiry >= expiry_y:
            return supports[index], previous
    raise RuntimeError("failed to reach requested expiry")


def _initial_density(support: np.ndarray) -> np.ndarray:
    """Return the atomic unit-forward distribution on one support."""
    calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(calls, prepend=1.0) / np.diff(support, prepend=0.0)
    return np.diff(slopes, append=0.0)


def marginal_variance_from_density(
    *,
    support: np.ndarray,
    density: np.ndarray,
    expiry_y: float,
    background_annual_volatility: float,
) -> float:
    """Return annualized terminal simple-return variance without simulation."""
    if expiry_y <= 0.0:
        raise ValueError("expiry must be positive")
    if background_annual_volatility < 0.0:
        raise ValueError("background volatility must be non-negative")
    latent_mean = float(density @ support)
    latent_second_moment = float(density @ support**2)
    total_second_moment = latent_second_moment * np.exp(
        background_annual_volatility**2 * expiry_y
    )
    return (total_second_moment - latent_mean**2) / expiry_y


def model_free_variance_from_density(
    *,
    support: np.ndarray,
    density: np.ndarray,
    expiry_y: float,
    background_annual_volatility: float,
    num_strikes_per_side: int = 801,
    minimum_strike: float = 0.0025,
    maximum_strike: float = 6.0,
) -> float:
    """Return the normalized model-free variance of one DLV density state."""
    if expiry_y <= 0.0 or num_strikes_per_side < 3:
        raise ValueError("expiry and strike count are invalid")
    if not 0.0 < minimum_strike < 1.0 < maximum_strike:
        raise ValueError("strike bounds must bracket the forward")
    if background_annual_volatility < 0.0:
        raise ValueError("background volatility must be non-negative")
    left = np.exp(np.linspace(np.log(minimum_strike), 0.0, num_strikes_per_side))
    right = np.exp(np.linspace(0.0, np.log(maximum_strike), num_strikes_per_side))
    strikes = np.concatenate((left, right[1:]))
    if background_annual_volatility == 0.0:
        kernels = np.maximum(support[None, :] - strikes[:, None], 0.0)
    else:
        standard_deviation = background_annual_volatility * np.sqrt(expiry_y)
        log_ratio = np.log(support[None, :] / strikes[:, None])
        d1 = log_ratio / standard_deviation + 0.5 * standard_deviation
        d2 = d1 - standard_deviation
        kernels = support[None, :] * ndtr(d1) - strikes[:, None] * ndtr(d2)
    calls = kernels @ density
    left_count = len(left)
    puts = np.maximum(calls[:left_count] + left - 1.0, 0.0)
    left_integral = np.trapezoid(puts / left**2, left)
    right_integral = np.trapezoid(calls[left_count - 1 :] / right**2, right)
    return float(2.0 * (left_integral + right_integral) / expiry_y)


def _eligible_expiries(
    *,
    trading_date: date,
    raw: pd.DataFrame,
    sessions: np.ndarray,
) -> list[tuple[date, bool, float, float]]:
    """Return Cboe-eligible Friday expiries in the 23--37 calendar-day window.

    The raw OptionMetrics book is EOD rather than Cboe's intraday constituent
    snapshot.  The settlement-minute adjustment is nevertheless retained so
    that PM and standard AM contracts receive their published relative clocks.
    """
    entries: list[tuple[date, bool, float, float]] = []
    available = set(zip(raw["exdate"].astype(str), raw["am_settlement"].astype(float)))
    origin = int(np.searchsorted(sessions, np.datetime64(trading_date), side="right"))
    for days in range(23, 39):
        nominal = trading_date + np.timedelta64(days, "D").astype(object)
        if nominal.weekday() != 4:
            continue
        settlement = sessions[
            np.searchsorted(sessions, np.datetime64(nominal), side="right") - 1
        ].astype(object)
        is_am = 15 <= nominal.day <= 21
        if (settlement.isoformat(), float(is_am)) not in available:
            continue
        calendar_days = (settlement - trading_date).days - (405 / 1440 if is_am else 15 / 1440)
        if not 23 < calendar_days < 37:
            continue
        business_days = float(
            np.searchsorted(sessions, np.datetime64(settlement), side="right") - origin
        )
        business_days -= AM_EXIRY_SUBTRACT if is_am else 0.0
        entries.append((settlement, is_am, calendar_days, business_days))
    return sorted(entries, key=lambda value: value[2])


def _model_option_prices(
    *,
    support: np.ndarray,
    density: np.ndarray,
    normalized_strikes: np.ndarray,
    expiry_y: float,
    background_annual_volatility: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized call and put prices under one decoded GROMM2 state."""
    if background_annual_volatility == 0.0:
        calls = np.maximum(support[None, :] - normalized_strikes[:, None], 0.0) @ density
        puts = np.maximum(normalized_strikes[:, None] - support[None, :], 0.0) @ density
        return calls, puts
    standard_deviation = background_annual_volatility * np.sqrt(expiry_y)
    log_ratio = np.log(support[None, :] / normalized_strikes[:, None])
    d1 = log_ratio / standard_deviation + 0.5 * standard_deviation
    d2 = d1 - standard_deviation
    calls = (support[None, :] * ndtr(d1) - normalized_strikes[:, None] * ndtr(d2)) @ density
    puts = (normalized_strikes[:, None] * ndtr(-d2) - support[None, :] * ndtr(-d1)) @ density
    return np.maximum(calls, 0.0), np.maximum(puts, 0.0)


def _term_from_raw_book(
    *,
    trading_date: date,
    expiry: tuple[date, bool, float, float],
    raw: pd.DataFrame,
    zero_curve: pd.DataFrame,
    supports: np.ndarray,
    transports: np.ndarray,
    expiries_y: np.ndarray,
    dlvsq: np.ndarray,
    background_annual_volatility: float,
) -> dict[str, float | int | str | bool | None]:
    """Price one published VIX constituent strip from raw bids and asks."""
    settlement, is_am, calendar_days, business_days = expiry
    rows = raw[
        (raw["exdate"].astype(str) == settlement.isoformat())
        & (raw["am_settlement"].astype(float) == float(is_am))
        & (raw["exercise_style"] == "E")
        & (raw["ss_flag"].astype(str) == "0")
    ].copy()
    if rows.duplicated(["strike_price", "cp_flag"]).any():
        raise ValueError(f"duplicate strike/side contracts for {settlement}")
    books = rows.pivot(index="strike_price", columns="cp_flag", values=["best_bid", "best_offer"]).sort_index()
    strikes = books.index.to_numpy(dtype=np.float64)

    def book(field: str, side: str) -> np.ndarray:
        return books[(field, side)].to_numpy(dtype=np.float64, na_value=np.nan)

    curve = zero_curve.sort_values("days")
    rate = float(np.interp(calendar_days, curve["days"].astype(float), curve["rate"].astype(float))) / 100.0
    discount = float(np.exp(-rate * calendar_days / 365.0))
    call_bids, call_asks, put_bids, put_asks = [
        book(field, side)
        for field, side in (("best_bid", "C"), ("best_offer", "C"), ("best_bid", "P"), ("best_offer", "P"))
    ]
    strip = select_vix_strip(
        strikes=strikes,
        call_bids=call_bids,
        call_asks=call_asks,
        put_bids=put_bids,
        put_asks=put_asks,
        discount_factor=discount,
    )
    expiry_y = business_days / DAYS_PER_YEAR
    support, density = _density_at_expiry(
        supports=supports,
        transports=transports,
        expiries_y=expiries_y,
        dlvsq=dlvsq,
        expiry_y=expiry_y,
    )
    calls, puts = _model_option_prices(
        support=support,
        density=density,
        normalized_strikes=strip.strikes / strip.forward,
        expiry_y=expiry_y,
        background_annual_volatility=background_annual_volatility,
    )
    model_prices = np.where(strip.strikes < strip.k0, puts, calls)
    at_k0 = strip.strikes == strip.k0
    model_prices[at_k0] = 0.5 * (calls[at_k0] + puts[at_k0])
    model_prices *= strip.forward * discount
    return {
        "expiry": settlement.isoformat(),
        "is_am": is_am,
        "calendar_days": calendar_days,
        "business_days": business_days,
        "market_total_variance": discrete_total_variance(strip, discount_factor=discount),
        "gromm2_total_variance": discrete_total_variance(
            strip, discount_factor=discount, prices=model_prices
        ),
        "selected_strikes": len(strip.strikes),
        "put_zero_bid_stop": strip.put_zero_bid_stop,
        "call_zero_bid_stop": strip.call_zero_bid_stop,
    }


def run_discrete_cboe_comparison(
    *,
    history_path: Path,
    output_directory: Path,
    start: int = 0,
    limit: int | None = None,
) -> dict[str, object]:
    """Compare GROMM2 with Cboe's discrete-strike, two-term VIX calculation.

    Raw market bids decide Cboe eligibility and two-zero-bid wing cutoffs;
    GROMM2 reprices precisely those selected contracts.  The optional terminal
    marginal variance at 30 days is deliberately retained as a distinct proxy,
    not a model-free VIX value.
    """
    with h5py.File(history_path, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        dlvsq = np.asarray(stream["dlvsq"], dtype=np.float64)
        calibration = json.loads(stream.attrs["calibration_config"])
    if start < 0 or start >= len(dates):
        raise ValueError("start must identify a retained state")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
    stop = len(dates) if limit is None else min(start + limit, len(dates))
    dates, dlvsq = dates[start:stop], dlvsq[start:stop]
    background_annual_volatility = float(calibration["backbone_annual_volatility"])
    state_dates = [date.fromisoformat(value) for value in dates]
    sessions = np.asarray(
        get_business_days(
            state_dates[0] - timedelta(days=10),
            state_dates[-1] + timedelta(days=65),
        ),
        dtype="datetime64[D]",
    )
    raw_cache = env.locations("widb", default_fmt=SubDir.PANDAS_PARQUET).raw
    official = load_cboe_vix(state_dates[0], state_dates[-1]).select(
        "date", pl.col("close").alias("cboe_vix")
    )
    records: list[dict[str, object]] = []
    for index, (trading_date, state) in enumerate(zip(state_dates, dlvsq, strict=True)):
        record: dict[str, object] = {"date": trading_date, "status": "unavailable"}
        raw = raw_cache.read(f"options/108105/{trading_date.isoformat()}")
        zero_curve = raw_cache.read(f"zero_curve/{trading_date.isoformat()}")
        if raw is None or zero_curve is None:
            record["reason"] = "missing cached raw option book or zero curve"
        else:
            try:
                expiries = _eligible_expiries(trading_date=trading_date, raw=raw, sessions=sessions)
                near = [item for item in expiries if item[2] <= 30.0]
                far = [item for item in expiries if item[2] > 30.0]
                if not near or not far:
                    raise ValueError("no eligible Friday expiries bracket 30 calendar days")
                terms = [
                    _term_from_raw_book(
                        trading_date=trading_date,
                        expiry=item,
                        raw=raw,
                        zero_curve=zero_curve,
                        supports=supports,
                        transports=transports,
                        expiries_y=expiries_y,
                        dlvsq=state,
                        background_annual_volatility=background_annual_volatility,
                    )
                    for item in (near[-1], far[0])
                ]
                first, second = terms
                record.update(
                    status="ok",
                    near_expiry=first["expiry"],
                    next_expiry=second["expiry"],
                    gromm2_discrete=constant_maturity_vix(
                        near_days=float(first["calendar_days"]),
                        next_days=float(second["calendar_days"]),
                        near_total_variance=float(first["gromm2_total_variance"]),
                        next_total_variance=float(second["gromm2_total_variance"]),
                    ),
                    market_discrete=constant_maturity_vix(
                        near_days=float(first["calendar_days"]),
                        next_days=float(second["calendar_days"]),
                        near_total_variance=float(first["market_total_variance"]),
                        next_total_variance=float(second["market_total_variance"]),
                    ),
                    near_selected_strikes=first["selected_strikes"],
                    next_selected_strikes=second["selected_strikes"],
                )
                support, density = _density_at_expiry(
                    supports=supports,
                    transports=transports,
                    expiries_y=expiries_y,
                    dlvsq=state,
                    expiry_y=30.0 / 365.0,
                )
                marginal = marginal_variance_from_density(
                    support=support,
                    density=density,
                    expiry_y=30.0 / 365.0,
                    background_annual_volatility=background_annual_volatility,
                )
                record["gromm2_marginal_variance_proxy"] = 100.0 * np.sqrt(max(marginal, 0.0))
            except (KeyError, ValueError) as error:
                record["reason"] = str(error)
        records.append(record)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(state_dates):
            print(f"discrete-vix completed={index + 1}/{len(state_dates)}", flush=True)
    frame = pl.from_dicts(records, infer_schema_length=None).join(official, on="date", how="left")
    complete = frame.filter(pl.col("status") == "ok").with_columns(
        (pl.col("gromm2_discrete") - pl.col("cboe_vix")).alias("difference_vol_points"),
        (pl.col("gromm2_marginal_variance_proxy") - pl.col("cboe_vix")).alias("marginal_proxy_difference_vol_points"),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_directory / "vix_discrete_comparison.parquet")
    report: dict[str, object] = {
        "schema": "gromm2-cboe-discrete-vix-v1",
        "history_path": str(history_path.resolve()),
        "state_index_range": [start, stop],
        "observations": frame.height,
        "complete_observations": complete.height,
        "method": "Cboe discrete OTM strips with raw-book two-zero-bid cutoffs and total-variance interpolation",
        "marginal_proxy": "terminal simple-return variance at exactly 30 calendar days; not a VIX calculation",
        "gromm2_discrete": _error_report(complete["difference_vol_points"].to_numpy()) if complete.height else {},
        "market_discrete": _error_report((complete["market_discrete"] - complete["cboe_vix"]).to_numpy()) if complete.height else {},
        "marginal_variance_proxy": _error_report(complete["marginal_proxy_difference_vol_points"].to_numpy()) if complete.height else {},
    }
    (output_directory / "vix_discrete_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def merge_discrete_cboe_comparisons(
    *,
    chunk_directories: list[Path],
    output_directory: Path,
) -> dict[str, object]:
    """Merge non-overlapping resumable Cboe-VIX blocks into one retained result."""
    if not chunk_directories:
        raise ValueError("at least one chunk directory is required")
    frames = [
        pl.read_parquet(directory / "vix_discrete_comparison.parquet")
        for directory in chunk_directories
    ]
    frame = pl.concat(frames, how="diagonal_relaxed").sort("date")
    if frame["date"].n_unique() != frame.height:
        raise ValueError("discrete VIX chunks overlap")
    complete = frame.filter(pl.col("status") == "ok")
    complete = complete.with_columns(
        (pl.col("gromm2_discrete") - pl.col("cboe_vix")).alias("difference_vol_points"),
        (pl.col("gromm2_marginal_variance_proxy") - pl.col("cboe_vix")).alias(
            "marginal_proxy_difference_vol_points"
        ),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_directory / "vix_discrete_comparison.parquet")
    report: dict[str, object] = {
        "schema": "gromm2-cboe-discrete-vix-v1",
        "source_chunks": [str(directory.resolve()) for directory in chunk_directories],
        "observations": frame.height,
        "complete_observations": complete.height,
        "method": "Cboe discrete OTM strips with raw-book two-zero-bid cutoffs and total-variance interpolation",
        "marginal_proxy": "terminal simple-return variance at exactly 30 calendar days; not a VIX calculation",
        "gromm2_discrete": _error_report(complete["difference_vol_points"].to_numpy()),
        "market_discrete": _error_report(
            (complete["market_discrete"] - complete["cboe_vix"]).to_numpy()
        ),
        "marginal_variance_proxy": _error_report(
            complete["marginal_proxy_difference_vol_points"].to_numpy()
        ),
    }
    (output_directory / "vix_discrete_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    axis.plot(complete["date"], complete["cboe_vix"], label="Cboe VIX close", lw=1.0)
    axis.plot(
        complete["date"],
        complete["market_discrete"],
        label="SPX data VIX (Cboe formula)",
        lw=1.0,
        linestyle=":",
    )
    axis.plot(
        complete["date"],
        complete["gromm2_discrete"],
        label="GROMM2 VIX (Cboe strips)",
        lw=1.0,
    )
    axis.plot(
        complete["date"],
        complete["gromm2_marginal_variance_proxy"],
        label="Analytic GROMM2 terminal-marginal proxy",
        lw=1.0,
        linestyle="--",
    )
    axis.set_ylabel("annualized volatility (%)")
    axis.set_title("GROMM2 and discrete Cboe VIX methodology")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_directory / "vix_discrete_comparison.png", dpi=180)
    plt.close(figure)
    return report


def run_comparison(
    *,
    history_path: Path,
    output_directory: Path,
    horizon_calendar_days: int = 30,
    num_strikes_per_side: int = 801,
) -> dict[str, object]:
    """Write aligned GROMM2 and Cboe VIX levels for every retained state."""
    if horizon_calendar_days <= 0:
        raise ValueError("horizon_calendar_days must be positive")
    expiry_y = horizon_calendar_days / 365.0
    with h5py.File(history_path, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        dlvsq = np.asarray(stream["dlvsq"], dtype=np.float64)
        calibration = json.loads(stream.attrs["calibration_config"])
    background_annual_volatility = float(calibration["backbone_annual_volatility"])
    option_levels = np.empty(len(dates), dtype=np.float64)
    marginal_variance_levels = np.empty(len(dates), dtype=np.float64)
    for index, state in enumerate(dlvsq):
        support, density = _density_at_expiry(
            supports=supports,
            transports=transports,
            expiries_y=expiries_y,
            dlvsq=state,
            expiry_y=expiry_y,
        )
        variance = model_free_variance_from_density(
            support=support,
            density=density,
            expiry_y=expiry_y,
            background_annual_volatility=background_annual_volatility,
            num_strikes_per_side=num_strikes_per_side,
        )
        option_levels[index] = 100.0 * np.sqrt(max(variance, 0.0))
        marginal_variance = marginal_variance_from_density(
            support=support,
            density=density,
            expiry_y=expiry_y,
            background_annual_volatility=background_annual_volatility,
        )
        marginal_variance_levels[index] = 100.0 * np.sqrt(
            max(marginal_variance, 0.0)
        )
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(dates):
            print(f"vix completed={index + 1}/{len(dates)}", flush=True)
    model = pl.DataFrame(
        {
            "date": [date.fromisoformat(value) for value in dates],
            "gromm2_vix": option_levels,
            "gromm2_marginal_variance_vix": marginal_variance_levels,
        }
    )
    official = load_cboe_vix(start=model["date"].min(), end=model["date"].max()).select(
        "date", pl.col("close").alias("cboe_vix")
    )
    comparison = model.join(official, on="date", how="inner").with_columns(
        (pl.col("gromm2_vix") - pl.col("cboe_vix")).alias("difference_vol_points"),
        (pl.col("gromm2_marginal_variance_vix") - pl.col("cboe_vix")).alias(
            "marginal_variance_difference_vol_points"
        ),
        (pl.col("gromm2_marginal_variance_vix") - pl.col("gromm2_vix")).alias(
            "marginal_variance_minus_option_vol_points"
        ),
    )
    if comparison.height != model.height:
        raise RuntimeError("official VIX history is missing GROMM2 state dates")
    difference = comparison["difference_vol_points"].to_numpy()
    marginal_variance_difference = comparison[
        "marginal_variance_difference_vol_points"
    ].to_numpy()
    estimator_difference = comparison[
        "marginal_variance_minus_option_vol_points"
    ].to_numpy()
    report: dict[str, object] = {
        "schema": "gromm2-vix-comparison-v1",
        "history_path": str(history_path.resolve()),
        "state_dates": [dates[0], dates[-1]],
        "observations": comparison.height,
        "horizon_calendar_days": horizon_calendar_days,
        "strike_nodes_per_side": num_strikes_per_side,
        "background_annual_volatility": background_annual_volatility,
        "marginal_variance_method": "terminal_second_moment",
        "option_integral": _error_report(difference),
        "marginal_variance": _error_report(marginal_variance_difference),
        "marginal_variance_minus_option_vol_points": {
            "mean": float(np.mean(estimator_difference)),
            "median_absolute": float(np.median(np.abs(estimator_difference))),
            "maximum_absolute": float(np.max(np.abs(estimator_difference))),
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison.write_parquet(output_directory / "vix_comparison.parquet")
    (output_directory / "vix_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    axis.plot(comparison["date"], comparison["cboe_vix"], label="Cboe VIX close")
    axis.plot(comparison["date"], comparison["gromm2_vix"], label="GROMM2 option integral")
    axis.plot(
        comparison["date"],
        comparison["gromm2_marginal_variance_vix"],
        label="GROMM2 terminal variance",
        linestyle="--",
    )
    axis.set_ylabel("annualized volatility (%)")
    axis.set_title("30-calendar-day model-free implied volatility")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_directory / "vix_comparison.png", dpi=180)
    plt.close(figure)
    return report


def _error_report(difference: np.ndarray) -> dict[str, float]:
    """Summarize model-minus-Cboe errors in annualized volatility points."""
    return {
        "mean_difference_vol_points": float(np.mean(difference)),
        "median_absolute_difference_vol_points": float(np.median(np.abs(difference))),
        "rmse_vol_points": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_difference_vol_points": float(np.max(np.abs(difference))),
        "fraction_absolute_difference_above_one_vol_point": float(
            np.mean(np.abs(difference) > 1.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--horizon-calendar-days", type=int, default=30)
    parser.add_argument("--strike-nodes-per-side", type=int, default=801)
    parser.add_argument(
        "--cboe-discrete",
        action="store_true",
        help="Use raw-book constituent selection and Cboe's two-term discrete formula.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.cboe_discrete:
        report = run_discrete_cboe_comparison(
            history_path=arguments.history,
            output_directory=arguments.output_directory,
            start=arguments.start,
            limit=arguments.limit,
        )
        print(json.dumps(report, indent=2), flush=True)
        return
    print(
        json.dumps(
            run_comparison(
                history_path=arguments.history,
                output_directory=arguments.output_directory,
                horizon_calendar_days=arguments.horizon_calendar_days,
                num_strikes_per_side=arguments.strike_nodes_per_side,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
