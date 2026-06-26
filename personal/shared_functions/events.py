import numpy as np
import polars as pl

from quantdata.massive.trade_conds import NORMAL_TRADE_EXCLUDE_FLAGS

# ── CONFIG ──────────────────────────────────────────────────────────────────
ONE_SIDED_R_MULTIPLIER = 10.0  # inflate variance for one-sided quotes
TRADE_R_DIVISOR = 4.0  # trades are more trusted than quotes
STALE_QUOTE_THRESHOLD = 60_000  # ms — mark quote as stale beyond this
TRADE_LAG_NS = 250_000_000  # 250ms in nanoseconds


def filter_df(df, col="trade_code"):
    return df.filter((pl.col(col) & int(NORMAL_TRADE_EXCLUDE_FLAGS)) == 0)


def match_trade_to_closest_quote(
    df_trades: pl.DataFrame, df_quotes: pl.DataFrame, trade_lag_ns=TRADE_LAG_NS, take_last=True
) -> pl.DataFrame:
    """
    For each trade at time t, use the last quote in [t, t+250ms].
    Falls back to the last quote before t if no forward quote exists in the window.
    Optional:
    Trade_lag_ns is the maximum allowed lag for forward matches (in ns); if a forward match is found but its lag exceeds
      this threshold, it will be rejected and the backward match will be used instead. Default is 250ms (250_000_000 ns).
    """
    quotes_sorted = df_quotes.sort("ts_ns").select(
        ["ts_ns", "bid_price", "ask_price", "bid_size", "ask_size"]
    )

    # --- forward-window match: last quote in [t, t+250ms] ---
    # Backward join on (t + 250ms) gives the last quote <= t+250ms;
    # reject it if it landed before t (i.e. outside the window).
    if take_last:
        forward = (
            df_trades.with_columns(
                (pl.col("ts_ns") + pl.duration(nanoseconds=trade_lag_ns)).alias("ts_ns_window_end")
            )
            .join_asof(
                quotes_sorted.rename({"ts_ns": "quote_ts_fwd"}),
                left_on="ts_ns_window_end",
                right_on="quote_ts_fwd",
                strategy="backward",
            )
            .drop("ts_ns_window_end")
            .with_columns(
                pl.when(pl.col("quote_ts_fwd") >= pl.col("ts_ns"))
                .then((pl.col("quote_ts_fwd") - pl.col("ts_ns")).dt.total_nanoseconds())
                .otherwise(None)
                .alias("fwd_lag_ns")
            )
        )
    else:
        forward = (
            df_trades.join_asof(
                df_quotes.sort("ts_ns")
                .select(["ts_ns", "bid_price", "ask_price", "bid_size", "ask_size"])
                .rename({"ts_ns": "quote_ts_fwd"}),
                left_on="ts_ns",
                right_on="quote_ts_fwd",
                strategy="forward",
            )
            .with_columns(
                (pl.col("quote_ts_fwd") - pl.col("ts_ns"))
                .dt.total_nanoseconds()
                .alias("fwd_lag_ns")
            )
            .with_columns(
                # null out if outside the 250ms window
                pl.when(pl.col("fwd_lag_ns") <= TRADE_LAG_NS)
                .then(pl.col("fwd_lag_ns"))
                .otherwise(None)
                .alias("fwd_lag_ns")
            )
        )

    # --- backward match: last quote at or before t (fallback only) ---
    backward = df_trades.join_asof(
        quotes_sorted.rename({"ts_ns": "quote_ts_bwd"}),
        left_on="ts_ns",
        right_on="quote_ts_bwd",
        strategy="backward",
    ).with_columns(
        (pl.col("ts_ns") - pl.col("quote_ts_bwd")).dt.total_nanoseconds().alias("bwd_lag_ns")
    )

    # --- prefer last quote in window; fall back to backward if window is empty ---
    combined = forward.with_columns(
        [
            backward["bwd_lag_ns"].alias("bwd_lag_ns"),
            backward["bid_price"].alias("bid_price_bwd"),
            backward["ask_price"].alias("ask_price_bwd"),
            backward["bid_size"].alias("bid_size_bwd"),
            backward["ask_size"].alias("ask_size_bwd"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("fwd_lag_ns").is_not_null())
            .then(pl.col("bid_price"))
            .otherwise(pl.col("bid_price_bwd"))
            .alias("best_bid"),
            pl.when(pl.col("fwd_lag_ns").is_not_null())
            .then(pl.col("ask_price"))
            .otherwise(pl.col("ask_price_bwd"))
            .alias("best_ask"),
            pl.when(pl.col("fwd_lag_ns").is_not_null())
            .then(pl.col("bid_size"))
            .otherwise(pl.col("bid_size_bwd"))
            .alias("best_bid_size"),
            pl.when(pl.col("fwd_lag_ns").is_not_null())
            .then(pl.col("ask_size"))
            .otherwise(pl.col("ask_size_bwd"))
            .alias("best_ask_size"),
            pl.when(pl.col("fwd_lag_ns").is_not_null())
            .then(pl.col("fwd_lag_ns"))
            .otherwise(-pl.col("bwd_lag_ns"))
            .alias("matched_lag_ns"),
        ]
    )

    return combined


# ── 1. PROCESS QUOTES ───────────────────────────────────────────────────────
def process_quotes(
    df_quotes: pl.DataFrame,
    one_sided_R_multiplier=ONE_SIDED_R_MULTIPLIER,
    price_type: str = "mid",
) -> pl.DataFrame:
    """
    price_type : "mid"   -> observed_price = (bid + ask) / 2  (default)
                 "micro" -> observed_price = size-weighted microprice
                            (ask_size * bid + bid_size * ask) / (ask_size + bid_size)
    One-sided quotes always fall back to the single available side.
    """
    if price_type not in ("mid", "micro"):
        raise ValueError(f"price_type must be 'mid' or 'micro', got {price_type!r}")

    if price_type == "micro":
        two_sided_price = (
            pl.col("ask_size") * pl.col("bid_price") + pl.col("bid_size") * pl.col("ask_price")
        ) / (pl.col("ask_size") + pl.col("bid_size"))
    else:  # "mid"
        two_sided_price = (pl.col("bid_price") + pl.col("ask_price")) / 2

    df = (
        df_quotes.with_columns(
            [
                pl.when(pl.col("bid_price") <= 0)
                .then(None)
                .otherwise(pl.col("bid_price"))
                .alias("bid_price"),
                pl.when(pl.col("ask_price") <= 0)
                .then(None)
                .otherwise(pl.col("ask_price"))
                .alias("ask_price"),
                pl.when(pl.col("bid_size") <= 0)
                .then(None)
                .otherwise(pl.col("bid_size"))
                .alias("bid_size"),
                pl.when(pl.col("ask_size") <= 0)
                .then(None)
                .otherwise(pl.col("ask_size"))
                .alias("ask_size"),
            ]
        )
        .with_columns(
            pl.when(pl.col("bid_price").is_not_null() & pl.col("ask_price").is_not_null())
            .then(pl.lit("two_sided"))
            .when(pl.col("bid_price").is_not_null())
            .then(pl.lit("bid_only"))
            .when(pl.col("ask_price").is_not_null())
            .then(pl.lit("ask_only"))
            .otherwise(pl.lit("empty"))
            .alias("quote_side")
        )
        .filter(pl.col("quote_side") != "empty")
        .with_columns(
            [
                pl.when(pl.col("quote_side") == "two_sided")
                .then(two_sided_price)
                .when(pl.col("quote_side") == "bid_only")
                .then(pl.col("bid_price"))
                .otherwise(pl.col("ask_price"))
                .alias("observed_price"),
                pl.when(pl.col("quote_side") == "two_sided")
                .then(pl.col("ask_price") - pl.col("bid_price"))
                .otherwise(None)
                .alias("spread"),
            ]
        )
        .with_columns(pl.col("spread").fill_null(strategy="forward").alias("spread_filled"))
        .with_columns(
            [
                pl.when(pl.col("quote_side") == "two_sided")
                .then(
                    (pl.col("spread") / 2) ** 2
                    / (1 + (pl.col("bid_size") + pl.col("ask_size")).log1p())
                )
                .otherwise((pl.col("spread_filled") / 2) ** 2 * one_sided_R_multiplier)
                .alias("R"),
                pl.lit("quote").alias("event_type"),
                pl.lit(0).cast(pl.Int64).alias("matched_lag_ns"),  # no lag for quotes
            ]
        )
        .select(
            [
                "ts_ns",
                "event_type",
                "observed_price",
                "spread",
                "bid_price",
                "ask_price",
                "bid_size",
                "ask_size",
                "R",
                "matched_lag_ns",
            ]
        )
    )
    return df


# ── 2. PROCESS TRADES ───────────────────────────────────────────────────────
def process_trades(
    df_trades: pl.DataFrame, df_quotes: pl.DataFrame, trade_R_divisor=TRADE_R_DIVISOR
) -> pl.DataFrame:
    """
    Matches each trade to the closest forward quote within 250ms.
    Falls back to the last prior quote if no forward quote exists in the window.
    Lag is carried as matched_lag_ns for later diagnostics — not used in R.
    """
    df_matched = match_trade_to_closest_quote(df_trades, df_quotes)

    df = (
        df_matched.with_columns(
            [
                pl.lit("trade").alias("event_type"),
                pl.col("price").alias("observed_price"),
                (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
                pl.col("best_bid").alias("bid_price"),
                pl.col("best_ask").alias("ask_price"),
                pl.col("best_bid_size").alias("bid_size"),
                pl.col("best_ask_size").alias("ask_size"),
            ]
        )
        .with_columns(
            pl.when(pl.col("spread").is_not_null())
            .then((pl.col("spread") / 2) ** 2 / trade_R_divisor)
            .otherwise((pl.col("observed_price") * 0.01) ** 2)
            .alias("R")
        )
        .select(
            [
                "ts_ns",
                "event_type",
                "observed_price",
                "spread",
                "bid_price",
                "ask_price",
                "bid_size",
                "ask_size",
                "R",
                "matched_lag_ns",
            ]
        )
    )
    return df


# ── 3. MERGE INTO EVENT STREAM ───────────────────────────────────────────────
def build_event_stream(
    df_quotes: pl.DataFrame, df_trades: pl.DataFrame, price_type: str = "mid"
) -> pl.DataFrame:
    df_q = process_quotes(df_quotes, price_type=price_type)
    df_t = process_trades(df_trades, df_quotes)  # pass raw quotes, not processed

    df_events = (
        pl.concat([df_q, df_t], how="diagonal")
        .sort("ts_ns")
        .with_columns(pl.col("spread").fill_null(strategy="forward"))
        .with_columns(
            pl.col("bid_price").fill_null(strategy="forward"),
            pl.col("ask_price").fill_null(strategy="forward"),
        )
        .with_columns(
            pl.min_horizontal("bid_price", "ask_price").alias("bid_price"),
            pl.max_horizontal("bid_price", "ask_price").alias("ask_price"),
        )
    )
    return df_events


# ── 4. KALMAN FILTER ─────────────────────────────────────────────────────────


def kalman_filter(df_events: pl.DataFrame, Q: float = 0.01) -> pl.DataFrame:
    obs_prices = df_events["observed_price"].to_numpy()
    Rs = df_events["R"].to_numpy()
    n = len(obs_prices)

    x_est = np.empty(n)
    P_est = np.empty(n)

    x_est[0] = obs_prices[0]
    P_est[0] = Rs[0]

    for t in range(1, n):
        x_pred = x_est[t - 1]
        P_pred = P_est[t - 1] + Q
        R_t = Rs[t]
        K = P_pred / (P_pred + R_t)
        x_est[t] = x_pred + K * (obs_prices[t] - x_pred)
        P_est[t] = (1 - K) * P_pred

    return df_events.with_columns(
        [
            pl.Series("kalman_price", x_est),
            pl.Series("kalman_var", P_est),
        ]
    )


def events_to_iv(df_events: pl.DataFrame, trade_R_divisor=TRADE_R_DIVISOR) -> pl.DataFrame:

    df_events = df_events.with_columns(
        (
            pl.col("R")
            * (((pl.col("ask_iv") - pl.col("bid_iv")) / 2) ** 2)
            / ((pl.col("spread") / 2) ** 2)
        ).alias("R_iv")
    )
    df_events = df_events.rename({"iv": "observed_iv"})
    df_events = df_events.with_columns(
        (((pl.col("ask_iv") - pl.col("bid_iv")) / 2) ** 2).alias("R_iv")
    ).with_columns(
        pl.when(pl.col("event_type") == "trade")
        .then(pl.col("R_iv").fill_null(strategy="forward") / trade_R_divisor)
        .otherwise(pl.col("R_iv"))
        .alias("R_iv")
    )
    return df_events


# ── USAGE ────────────────────────────────────────────────────────────────────
# df_events  = build_event_stream(df_quotes, df_trades)
# df_results = kalman_filter(df_events, Q=0.01)
