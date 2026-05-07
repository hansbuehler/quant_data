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
    Convert long IV DataFrame (ts_ns, strike, bid_iv, ask_iv, kalman_iv) to wide format,
    forward-filling each strike's IV across the merged timestamp spine via join_asof.
    """
    strikes = sorted(df_long["strike"].unique().to_list())
    spine = df_long.select("ts_ns").unique().sort("ts_ns")

    result = spine
    for strike in strikes:
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

    return result
