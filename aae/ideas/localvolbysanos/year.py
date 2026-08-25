"""Build a balanced annual panel from full-grid SANOS DLV inversions."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS, get_business_days
from quantdata.pool import PoolConfig
from quantdata.widb.bad import BAD

from .experiment import DayGridResult, ExperimentConfig, compute_day_grid


def _compute_date(
    ticker: str,
    market_date: date,
    *,
    config: ExperimentConfig,
) -> tuple[DayGridResult | None, str | None]:
    try:
        return (
            compute_day_grid(
                ticker,
                market_date,
                config=config,
                verbose=Context.quiet,
            ),
            None,
        )
    except Exception as exception:
        return None, f"{type(exception).__name__}: {exception}"


def _effective_nms(
    strikes: np.ndarray,
    *,
    expiry: float,
    front_week_vol: float,
) -> np.ndarray:
    """Return front-week-vol-scaled NMS coordinates for forward strikes K/F."""

    return np.asarray(
        NMS.to_nms(
            strikes,
            float(expiry),
            vol=front_week_vol,
            eps_vol=front_week_vol,
        ),
        dtype=np.float64,
    )


def _valid_atm_support(
    nms: np.ndarray,
    dlv: np.ndarray,
    forward_volatility: np.ndarray,
) -> tuple[float, float]:
    valid = (
        np.isfinite(dlv)
        & (dlv > 0.0)
        & np.isfinite(forward_volatility)
        & (forward_volatility > 0.0)
    )
    atm_index = int(np.argmin(np.abs(nms)))
    verify(valid[atm_index], lambda: "ATM DLV or forward volatility is invalid")
    lower = atm_index
    upper = atm_index
    while lower > 0 and valid[lower - 1]:
        lower -= 1
    while upper + 1 < len(valid) and valid[upper + 1]:
        upper += 1
    return float(nms[lower]), float(nms[upper])


def _fixed_nms_grid(lower: float, upper: float, num_strikes: int) -> np.ndarray:
    verify(
        lower < 0.0 < upper,
        lambda: f"balanced NMS support must straddle zero, found [{lower}, {upper}]",
    )
    distance = upper - lower
    num_lower = int(num_strikes * abs(lower) // distance)
    num_lower = min(max(2, num_lower), num_strikes - 3)
    num_upper = num_strikes - num_lower - 1
    grid = np.concatenate(
        (
            np.linspace(lower, 0.0, num_lower, endpoint=False),
            np.linspace(0.0, upper, num_upper + 1),
        )
    )
    assert grid.shape == (num_strikes,)
    assert np.count_nonzero(grid == 0.0) == 1
    return grid


def _raw_frame(
    market_date: date,
    result: DayGridResult,
    *,
    config: ExperimentConfig,
) -> pl.DataFrame:
    front_week_index = config.model_expiry_days.index(config.short_anchor_days)
    front_week_vol = float(result.atm_vols[front_week_index])
    frames: list[pl.DataFrame] = []
    for expiry_index, (
        expiry_days,
        expiry,
        atm_vol,
        strikes,
        calls,
        forward_volatility,
        density,
        increment,
        dlv,
    ) in enumerate(
        zip(
            config.model_expiry_days,
            config.model_expiries,
            result.atm_vols,
            result.inner_grids,
            result.call_prices,
            result.forward_volatilities,
            result.densities,
            result.call_increments,
            result.dlvs,
            strict=True,
        )
    ):
        nms = _effective_nms(
            strikes,
            expiry=float(expiry),
            front_week_vol=front_week_vol,
        )
        dlvsq = np.square(dlv)
        log_dlvsq = np.full_like(dlvsq, np.nan)
        valid_dlv = np.isfinite(dlvsq) & (dlvsq > 0.0)
        log_dlvsq[valid_dlv] = np.log(dlvsq[valid_dlv])
        log_forward_volatility = np.full_like(forward_volatility, np.nan)
        valid_volatility = np.isfinite(forward_volatility) & (forward_volatility > 0.0)
        log_forward_volatility[valid_volatility] = np.log(
            forward_volatility[valid_volatility]
        )
        frames.append(
            pl.DataFrame(
                {
                    "date": [market_date] * len(strikes),
                    "expiry_index": np.full(len(strikes), expiry_index, dtype=np.int16),
                    "expiry_days": np.full(len(strikes), expiry_days, dtype=np.int16),
                    "expiry_years": np.full(len(strikes), expiry, dtype=np.float64),
                    "reference_strike_index": np.arange(len(strikes), dtype=np.int16),
                    "forward_relative_strike": strikes,
                    "log_forward_relative_strike": np.log(strikes),
                    "effective_nms": nms,
                    "front_week_atm_volatility": np.full(
                        len(strikes), front_week_vol, dtype=np.float64
                    ),
                    "expiry_atm_volatility": np.full(
                        len(strikes), atm_vol, dtype=np.float64
                    ),
                    "call_price": calls,
                    "forward_volatility": forward_volatility,
                    "log_forward_volatility": log_forward_volatility,
                    "density": density[1:-1],
                    "call_increment": increment,
                    "dlv": dlv,
                    "dlvsq": dlvsq,
                    "log_dlvsq": log_dlvsq,
                    "dlv_valid": valid_dlv,
                }
            )
        )
    return pl.concat(frames)


def _balanced_panel(
    dates: list[date],
    results: list[DayGridResult],
    *,
    config: ExperimentConfig,
) -> tuple[pl.DataFrame, dict[str, np.ndarray], tuple[float, float]]:
    lower_requested, upper_requested = config.nms_range
    lower = float(lower_requested)
    upper = float(upper_requested)
    front_week_index = config.model_expiry_days.index(config.short_anchor_days)
    for result in results:
        front_week_vol = float(result.atm_vols[front_week_index])
        for expiry, strikes, dlv, forward_volatility in zip(
            config.model_expiries,
            result.inner_grids,
            result.dlvs,
            result.forward_volatilities,
            strict=True,
        ):
            nms = _effective_nms(
                strikes,
                expiry=float(expiry),
                front_week_vol=front_week_vol,
            )
            local_lower, local_upper = _valid_atm_support(
                nms, dlv, forward_volatility
            )
            lower = max(lower, local_lower)
            upper = min(upper, local_upper)

    model_nms = _fixed_nms_grid(lower, upper, config.base_nms_strikes)
    num_dates = len(dates)
    num_expiries = len(config.model_expiry_days)
    num_strikes = len(model_nms)
    log_dlvsq = np.empty((num_dates, num_expiries, num_strikes), dtype=np.float64)
    log_forward_volatility = np.empty_like(log_dlvsq)
    forward_relative_strike = np.empty_like(log_dlvsq)

    frames: list[pl.DataFrame] = []
    for date_index, (market_date, result) in enumerate(zip(dates, results, strict=True)):
        front_week_vol = float(result.atm_vols[front_week_index])
        for expiry_index, (
            expiry_days,
            expiry,
            raw_strikes,
            raw_dlv,
            raw_forward_volatility,
        ) in enumerate(
            zip(
                config.model_expiry_days,
                config.model_expiries,
                result.inner_grids,
                result.dlvs,
                result.forward_volatilities,
                strict=True,
            )
        ):
            raw_nms = _effective_nms(
                raw_strikes,
                expiry=float(expiry),
                front_week_vol=front_week_vol,
            )
            valid = (
                np.isfinite(raw_dlv)
                & (raw_dlv > 0.0)
                & np.isfinite(raw_forward_volatility)
                & (raw_forward_volatility > 0.0)
            )
            log_dlvsq_values = np.interp(
                model_nms, raw_nms[valid], np.log(np.square(raw_dlv[valid]))
            )
            log_forward_volatility_values = np.interp(
                model_nms,
                raw_nms[valid],
                np.log(raw_forward_volatility[valid]),
            )
            strikes = np.asarray(
                NMS.from_nms(
                    model_nms,
                    float(expiry),
                    vol=front_week_vol,
                    eps_vol=front_week_vol,
                ),
                dtype=np.float64,
            )
            log_dlvsq[date_index, expiry_index] = log_dlvsq_values
            log_forward_volatility[date_index, expiry_index] = (
                log_forward_volatility_values
            )
            forward_relative_strike[date_index, expiry_index] = strikes
            frames.append(
                pl.DataFrame(
                    {
                        "date": [market_date] * num_strikes,
                        "date_index": np.full(
                            num_strikes, date_index, dtype=np.int16
                        ),
                        "expiry_index": np.full(
                            num_strikes, expiry_index, dtype=np.int16
                        ),
                        "expiry_days": np.full(
                            num_strikes, expiry_days, dtype=np.int16
                        ),
                        "nms_index": np.arange(num_strikes, dtype=np.int16),
                        "effective_nms": model_nms,
                        "forward_relative_strike": strikes,
                        "front_week_atm_volatility": np.full(
                            num_strikes, front_week_vol, dtype=np.float64
                        ),
                        "log_dlvsq": log_dlvsq_values,
                        "dlvsq": np.exp(log_dlvsq_values),
                        "dlv": np.exp(0.5 * log_dlvsq_values),
                        "log_forward_volatility": log_forward_volatility_values,
                        "forward_volatility": np.exp(
                            log_forward_volatility_values
                        ),
                    }
                )
            )
    arrays = {
        "dates": np.asarray([value.isoformat() for value in dates]),
        "expiry_days": np.asarray(config.model_expiry_days, dtype=np.int16),
        "effective_nms": model_nms,
        "forward_relative_strike": forward_relative_strike,
        "log_dlvsq": log_dlvsq,
        "log_forward_volatility": log_forward_volatility,
    }
    return pl.concat(frames), arrays, (lower, upper)


def _daily_summary_frame(
    dates: list[date],
    results: list[DayGridResult],
    *,
    config: ExperimentConfig,
) -> pl.DataFrame:
    front_week_index = config.model_expiry_days.index(config.short_anchor_days)
    rows: list[dict[str, object]] = []
    for market_date, result in zip(dates, results, strict=True):
        dlv = result.summary["dlv"]
        recovery = result.summary["input_price_recovery"]
        rows.append(
            {
                "date": market_date,
                "spot_ref": result.summary["spot_ref"],
                "front_week_atm_volatility": float(
                    result.atm_vols[front_week_index]
                ),
                "num_reference_strikes": result.summary["master_grid"][
                    "num_strikes"
                ],
                "num_dlv_values": dlv["num_values"],
                "num_dlv_nonfinite": dlv["num_nonfinite"],
                "dlv_median": dlv["median"],
                "dlv_p99": dlv["p99"],
                "dlv_max": dlv["max"],
                "price_recovery_passed": recovery["passed"],
                "price_recovery_abs_error_max": recovery["abs_error_max"],
            }
        )
    return pl.DataFrame(rows)


def _plot_year(
    arrays: dict[str, np.ndarray],
    *,
    output_file: Path,
) -> None:
    import matplotlib.pyplot as plt

    dates = arrays["dates"]
    expiry_days = arrays["expiry_days"]
    nms = arrays["effective_nms"]
    atm_index = int(np.flatnonzero(nms == 0.0).item())
    atm_dlv = np.exp(0.5 * arrays["log_dlvsq"][:, :, atm_index]).T
    atm_forward_volatility = np.exp(
        arrays["log_forward_volatility"][:, :, atm_index]
    ).T
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, values, title in (
        (axes[0], atm_dlv, "ATM discrete local volatility"),
        (axes[1], atm_forward_volatility, "ATM forward implied volatility"),
    ):
        image = axis.imshow(values, aspect="auto", interpolation="nearest")
        axis.set_yticks(np.arange(len(expiry_days)), labels=[f"{x}d" for x in expiry_days])
        axis.set_ylabel("model expiry")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.02, pad=0.02)
    tick_indices = np.linspace(0, len(dates) - 1, 6, dtype=int)
    axes[-1].set_xticks(tick_indices, labels=dates[tick_indices])
    axes[-1].set_xlabel("market date")
    figure.suptitle("SPX front-week-scaled forward-relative state, 2025")
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def run_year(
    year: int,
    *,
    ticker: str = "SPX",
    config: ExperimentConfig = ExperimentConfig(
        full_reference_grid=True,
        density_negative_tolerance=1.0e-6,
    ),
    output_directory: Path | None = None,
    workers: int = 4,
) -> dict[str, object]:
    """Run a year and write raw and balanced panels for statistical modeling."""

    verify_inp(config.full_reference_grid, lambda: "year panel requires full grid")
    verify_inp(workers >= 1, lambda: "workers must be positive")
    output_directory = output_directory or Path(
        f"aae/ideas/localvolbysanos/results/{year}_full_reference_grid"
    )
    requested_dates = get_business_days(date(year, 1, 1), date(year, 12, 31))
    last_eligible_date = min(
        date(year, 12, 31), BAD.LAST_LIKELY_VALID_DATE - timedelta(days=1)
    )
    eligible_dates = [value for value in requested_dates if value <= last_eligible_date]
    pool = PoolConfig(num_workers=workers, threading=False).pool()
    jobs = {
        market_date: pool.delayed(_compute_date)(
            ticker,
            market_date,
            config=config,
        )
        for market_date in eligible_dates
    }
    completed: list[tuple[date, DayGridResult]] = []
    failures: list[dict[str, str]] = []
    for index, (market_date, result, error) in enumerate(pool.parallel(jobs), start=1):
        if result is None:
            assert error is not None
            failures.append({"date": market_date.isoformat(), "error": error})
        else:
            assert error is None
            completed.append((market_date, result))
        if index % 25 == 0 or index == len(eligible_dates):
            print(
                f"{year} DLV {index}/{len(eligible_dates)} "
                f"completed={len(completed)} failed={len(failures)}",
                flush=True,
            )
    completed.sort(key=lambda item: item[0])
    verify(len(completed) > 0, lambda: f"no {ticker} markets completed for {year}")
    dates = [item[0] for item in completed]
    results = [item[1] for item in completed]
    raw = pl.concat(
        [_raw_frame(value, result, config=config) for value, result in completed]
    )
    panel, arrays, common_support = _balanced_panel(dates, results, config=config)
    daily = _daily_summary_frame(dates, results, config=config)

    output_directory.mkdir(parents=True, exist_ok=True)
    raw.write_parquet(output_directory / "raw_full_grid.parquet", compression="zstd")
    panel.write_parquet(output_directory / "balanced_panel.parquet", compression="zstd")
    daily.write_parquet(output_directory / "daily_summary.parquet", compression="zstd")
    _plot_year(arrays, output_file=output_directory / "year_state.png")

    all_log_dlvsq = arrays["log_dlvsq"]
    all_log_forward_volatility = arrays["log_forward_volatility"]
    maximum_recovery_error = max(
        float(result.summary["input_price_recovery"]["abs_error_max"])
        for result in results
    )
    report: dict[str, object] = {
        "ticker": ticker,
        "year": year,
        "strike_coordinate": "K_over_F",
        "effective_coordinate": "front_week_vol_scaled_nms",
        "requested_dates": len(requested_dates),
        "eligible_dates": len(eligible_dates),
        "last_eligible_date": last_eligible_date.isoformat(),
        "unavailable_dates": len(requested_dates) - len(eligible_dates),
        "completed_dates": len(completed),
        "failures": failures,
        "config": asdict(config),
        "panel_shape": list(all_log_dlvsq.shape),
        "common_effective_nms_support": list(common_support),
        "effective_nms_strikes": arrays["effective_nms"].tolist(),
        "log_dlvsq": {
            "minimum": float(np.min(all_log_dlvsq)),
            "median": float(np.median(all_log_dlvsq)),
            "maximum": float(np.max(all_log_dlvsq)),
        },
        "log_forward_volatility": {
            "minimum": float(np.min(all_log_forward_volatility)),
            "median": float(np.median(all_log_forward_volatility)),
            "maximum": float(np.max(all_log_forward_volatility)),
        },
        "maximum_input_price_recovery_abs_error": maximum_recovery_error,
        "num_input_price_recovery_failures": int(
            np.count_nonzero(~daily["price_recovery_passed"].to_numpy())
        ),
    }
    with (output_directory / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    completed_report = run_year(2025)
    print(json.dumps(completed_report, indent=2))
