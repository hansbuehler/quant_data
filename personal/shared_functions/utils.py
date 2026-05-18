import datetime
import re

import polars as pl
import pytz


def r_squared(observed: pl.Series, residuals: pl.Series) -> float:
    ss_res = (residuals**2).sum()
    ss_tot = ((observed - observed.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def parse_option_ticker(ticker: str) -> tuple:
    """
    Parse a Polygon option ticker into (strike_dollars, expiry_datetime, option_type).
    'O:SPY240103C00402000' → (402.0, datetime(2024, 1, 3, 16, 0, 0, tzinfo=ET), 'C')
    """
    m = re.match(r"O:[A-Z]+(\d{6})([CP])(\d{8})", ticker)
    if not m:
        raise ValueError(f"Cannot parse ticker: {ticker}")
    date_str, option_type, strike_str = m.group(1), m.group(2), m.group(3)
    year = 2000 + int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    et = pytz.timezone("America/New_York")
    expiry = datetime.datetime(year, month, day, 16, 0, 0, tzinfo=et)
    strike = int(strike_str) / 1000.0
    return strike, expiry, option_type


def synchronize_iv_surface(df_long: pl.DataFrame) -> pl.DataFrame:
    """
    Convert long IV DataFrame to wide format, forward-filling each series via join_asof.

    When df_long has an 'is_call' column, column names are suffixed with _C / _P
    (e.g. bid_iv_479_C, kalman_iv_479_P). Without it, the old _<strike> naming is used.
    """
    has_type = "is_call" in df_long.columns
    spine = df_long.select("ts_ns").unique().sort("ts_ns")
    result = spine

    if has_type:
        keys = df_long.select(["strike", "is_call"]).unique().sort(["strike", "is_call"]).rows()
        for strike, is_call in keys:
            K_int = int(strike)
            suffix = f"_{K_int}_{'C' if is_call else 'P'}"
            df_k = (
                df_long.filter((pl.col("strike") == strike) & (pl.col("is_call") == is_call))
                .sort("ts_ns")
                .select(
                    [
                        "ts_ns",
                        pl.col("bid_iv").alias(f"bid_iv{suffix}"),
                        pl.col("ask_iv").alias(f"ask_iv{suffix}"),
                        pl.col("kalman_iv").alias(f"kalman_iv{suffix}"),
                    ]
                )
            )
            result = result.join_asof(df_k, on="ts_ns", strategy="backward")
    else:
        for strike in sorted(df_long["strike"].unique().to_list()):
            K_int = int(strike)
            df_k = (
                df_long.filter(pl.col("strike") == strike)
                .sort("ts_ns")
                .select(
                    [
                        "ts_ns",
                        pl.col("bid_iv").alias(f"bid_iv_{K_int}"),
                        pl.col("ask_iv").alias(f"ask_iv_{K_int}"),
                        pl.col("kalman_iv").alias(f"kalman_iv_{K_int}"),
                    ]
                )
            )
            result = result.join_asof(df_k, on="ts_ns", strategy="backward")

    if "forward" in df_long.columns:
        fwd_df = df_long.select(["ts_ns", "forward"]).group_by("ts_ns").first().sort("ts_ns")
        result = result.join_asof(fwd_df, on="ts_ns", strategy="backward")

    return result


def synchronize_price_surface(df_long: pl.DataFrame) -> pl.DataFrame:
    """
    Convert long price DataFrame to wide format, forward-filling each series via join_asof.

    Expects columns: ts_ns, strike, is_call, bid_price, ask_price, observed_price.
    Column names are suffixed with _C / _P (e.g. bid_price_479_C, observed_price_479_P).
    """
    spine = df_long.select("ts_ns").unique().sort("ts_ns")
    result = spine

    keys = df_long.select(["strike", "is_call"]).unique().sort(["strike", "is_call"]).rows()
    for strike, is_call in keys:
        K_int = int(strike)
        suffix = f"_{K_int}_{'C' if is_call else 'P'}"
        df_k = (
            df_long.filter((pl.col("strike") == strike) & (pl.col("is_call") == is_call))
            .sort("ts_ns")
            .select(
                [
                    "ts_ns",
                    pl.col("bid_price").alias(f"bid_price{suffix}"),
                    pl.col("ask_price").alias(f"ask_price{suffix}"),
                    pl.col("observed_price").alias(f"observed_price{suffix}"),
                ]
            )
        )
        result = result.join_asof(df_k, on="ts_ns", strategy="backward")

    return result
