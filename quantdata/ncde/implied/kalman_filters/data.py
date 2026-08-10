"""
Data loading, ranking, and validation helpers for option-quote Kalman runs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from cdxcore.verbose import Context

from quantdata.ncde.implied.day import OptionQuotesDayConfig, generate_day
from quantdata.pool import PoolConfig

from .constants import PRIMARY_TRADE_FIELDS
from .iv_math import direct_mid_vol, to_sqrt_years


def load_day_expiries(
    ticker: str,
    d: date,
    *,
    option_quotes_day_config: OptionQuotesDayConfig | None = None,
    pool_config: PoolConfig | None = None,
    verbose: Context | None = Context.all,
) -> dict[date, pl.DataFrame]:
    return generate_day(
        ticker,
        d,
        option_quotes_day_config=option_quotes_day_config or OptionQuotesDayConfig(),
        pool_config=pool_config or PoolConfig(num_workers=4, threading=False),
        verbose=verbose,
    )


def expiry_summary(results: dict[date, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for expiry, df in results.items():
        rows.append(
            {
                "expiry": expiry,
                "n_rows": len(df),
                "n_opt_tickers": df["opt_ticker"].n_unique(),
                "mean_abs_nms5": float(df["nms5"].abs().mean()),
                "median_dte": float(df["dte"].median()),
            }
        )
    return pl.DataFrame(rows,
                        schema={
                            "expiry": pl.Date,
                            "n_rows": pl.Int64,
                            "n_opt_tickers": pl.Int64,
                            "mean_abs_nms5": pl.Float64,
                            "median_dte": pl.Float64, 
                        }                      
                        ).sort("expiry")

def option_rank(df_expiry: pl.DataFrame, is_call: bool | None = None) -> pl.DataFrame:
    df = df_expiry if is_call is None else df_expiry.filter(pl.col("is_call") == is_call)
    return (
        df.group_by("opt_ticker")
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("nms5").abs().median().alias("median_abs_nms5"),
                pl.col("is_intrinsic").mean().alias("intrinsic_share"),
            ]
        )
        .sort(["median_abs_nms5", "n_rows"], descending=[False, True])
    )


def single_option_frame(df_expiry: pl.DataFrame, opt_ticker: str) -> dict[str, object]:
    df_opt = df_expiry.filter(pl.col("opt_ticker") == opt_ticker).sort("t_d")
    if len(df_opt) == 0:
        raise ValueError(f"No rows found for opt_ticker={opt_ticker!r}")

    strike = float(df_opt["strike"][0])
    is_call = bool(df_opt["is_call"][0])
    spot = df_opt["spot_mid_price"].to_numpy()
    sqrt_t = to_sqrt_years(df_opt["dte"].to_numpy())
    direct_vol = direct_mid_vol(
        mid_price=df_opt["mid_price"].to_numpy(),
        spot=spot,
        strike=np.full(len(df_opt), strike, dtype=float),
        sqrt_t=sqrt_t,
        is_call=np.full(len(df_opt), is_call, dtype=bool),
    )
    return {
        "df": df_opt,
        "strike": strike,
        "is_call": is_call,
        "spot": spot,
        "sqrt_t": sqrt_t,
        "direct_mid_vol": direct_vol,
        "stored_mid_vol": df_opt["mid_vol"].to_numpy(),
    }


def validate_expiry_frame(df_expiry: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    issues = []
    summary_rows = []
    required_quote_fields = [
        "t_d",
        "mid_price",
        "half_spread",
        "dte",
        "spot_mid_price",
        "spot_trade_price",
        "strike",
    ]

    for key, df_opt in df_expiry.group_by("opt_ticker", maintain_order=True):
        opt_ticker = key[0] if isinstance(key, tuple) else key
        arrays = {col: df_opt[col].to_numpy() for col in required_quote_fields + PRIMARY_TRADE_FIELDS}
        quote_ok = all(np.all(np.isfinite(arrays[col])) for col in required_quote_fields)
        trade_present = np.isfinite(arrays["trade_price"])
        trade_consistent = (
            np.array_equal(np.isfinite(arrays["trade_t_d"]), trade_present)
            and np.array_equal(np.isfinite(arrays["trade_size"]), trade_present)
        )

        if not quote_ok:
            bad = [col for col in required_quote_fields if not np.all(np.isfinite(arrays[col]))]
            issues.append((opt_ticker, f"non-finite required quote fields: {bad}"))
        if not trade_consistent:
            issues.append((opt_ticker, "trade_t_d / trade_price / trade_size are not jointly present"))

        summary_rows.append(
            {
                "opt_ticker": opt_ticker,
                "n_rows": len(df_opt),
                "n_trades": int(trade_present.sum()),
                "n_missing_trades": int((~trade_present).sum()),
                "quote_fields_ok": quote_ok,
                "trade_fields_ok": trade_consistent,
                "n_missing_dt_d": int(np.sum(~np.isfinite(df_opt["dt_d"].to_numpy()))),
                "n_missing_spot_quote_dt_d": int(np.sum(~np.isfinite(df_opt["spot_quote_dt_d"].to_numpy()))),
                "median_abs_nms5": float(np.nanmedian(np.abs(df_opt["nms5"].to_numpy()))),
            }
        )

    summary = pl.DataFrame(summary_rows).sort(["median_abs_nms5", "n_rows"])
    issue_frame = pl.DataFrame({"opt_ticker": [a for a, _ in issues], "issue": [b for _, b in issues]})
    return summary, issue_frame
