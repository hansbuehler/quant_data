import datetime

import black_scholes
import ekf
import events
import jax
import jax.numpy as jnp
import polars as pl
import pytz
import utils


def kalman_filter_forward_window_last_all(
    df_quotes,
    df_trades,
    df_spy,
    strike: float,
    expiry: datetime.datetime,
    option_type: str = "C",
):
    df_trades = events.filter_df(df_trades)
    df_spy = events.filter_df(df_spy)

    df_events = events.build_event_stream(df_quotes, df_trades)

    df_trades_result, df_quotes_result = ekf.kalman_filters_price_from_events(
        df_events, trade_on_quotes=True
    )

    df_events = (
        df_events.join_asof(df_spy.select(["ts_ns", "price"]), on="ts_ns", strategy="backward")
        .rename({"price": "spot"})
        .with_columns(pl.col("spot").fill_null(strategy="forward"))
    )

    df_events = df_events.with_columns(
        (
            (pl.lit(expiry) - pl.col("ts_ns")).dt.total_nanoseconds() / (365.25 * 24 * 3600 * 1e9)
        ).alias("T")
    )

    price = jnp.array(df_events["observed_price"].to_numpy())
    S = jnp.array(df_events["spot"].to_numpy())
    K = jnp.full_like(price, strike)
    T = jnp.array(df_events["T"].to_numpy())
    r = jnp.full_like(price, 0.0)

    iv_fn = black_scholes.newton_iv if option_type == "C" else black_scholes.newton_iv_put

    iv = iv_fn(price, S, K, T, r)

    df_events = df_events.with_columns(pl.Series("iv", iv.tolist()))
    bid_price = jnp.array(df_events["bid_price"].to_numpy())
    ask_price = jnp.array(df_events["ask_price"].to_numpy())
    bid_iv = iv_fn(bid_price, S, K, T, r)
    ask_iv = iv_fn(ask_price, S, K, T, r)
    bid_iv, ask_iv = jnp.minimum(bid_iv, ask_iv), jnp.maximum(bid_iv, ask_iv)
    df_events = df_events.with_columns(
        [
            pl.Series("bid_iv", bid_iv.tolist()),
            pl.Series("ask_iv", ask_iv.tolist()),
        ]
    )

    df_events = events.events_to_iv(df_events)

    df_iv_trades_result, df_iv_quotes_result = ekf.kalman_filters_iv_from_events(
        df_events, trade_on_quotes=True
    )

    return df_trades_result, df_iv_trades_result, df_quotes_result, df_iv_quotes_result


def build_iv_surface(
    df_tickers: pl.DataFrame,
    trade_date,
    client,
    df_spy=None,
    min_trades: int = 1,
    option_type: str = "C",
    max_iv: float = 5.0,
) -> pl.DataFrame:
    """
    Run IV Kalman filter for all options in df_tickers.
    Returns a long DataFrame: ts_ns, strike, bid_iv, ask_iv, kalman_iv.
    option_type: 'C' for calls, 'P' for puts (must match the BS formula used).
    max_iv: rows with any IV above this are set to NaN (illiquid/degenerate quotes).
    """
    if df_spy is None:
        df_spy = client.fetch_trades("SPY", trade_date)

    candidates = df_tickers.filter(pl.col("trade_count") >= min_trades)
    frames = []

    for row in candidates.iter_rows(named=True):
        ticker = row["ticker"]
        try:
            strike, expiry, otype = utils.parse_option_ticker(ticker)
            if otype != option_type:
                continue
            df_quotes = client.fetch_quotes(ticker, trade_date)
            df_trades = client.fetch_trades(ticker, trade_date)

            if df_quotes.is_empty():
                print(f"Skipping {ticker}: no quotes")
                continue

            _, _, _, df_iv_q = kalman_filter_forward_window_last_all(
                df_quotes=df_quotes,
                df_trades=df_trades,
                df_spy=df_spy,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
            )

            iv_cols = ["bid_iv", "ask_iv", "kalman_iv"]
            frames.append(
                df_iv_q.select(["ts_ns"] + iv_cols)
                .with_columns(
                    [
                        pl.when((pl.col(c) > max_iv) | (pl.col(c) < 0))
                        .then(None)
                        .otherwise(pl.col(c))
                        .alias(c)
                        for c in iv_cols
                    ]
                )
                .with_columns(pl.lit(strike).alias("strike"))
            )
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if not frames:
        raise RuntimeError("No options produced valid IV data")

    return pl.concat(frames).sort("ts_ns")
