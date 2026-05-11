import datetime

import black_scholes
import ekf
import events
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytz
import utils
from cdxcore.bs import bs as _cdx_bs

from quantdata.massive.taq import TAQMatchConfig, match_taq


def _add_spot_T(
    df: pl.DataFrame,
    spy_ts: pl.DataFrame,
    expiry: datetime.datetime,
    forward_ts: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join SPY spot, time-to-expiry (years), and optionally implied forward onto df."""
    result = (
        df.sort("ts_ns")
        .join_asof(spy_ts, on="ts_ns", strategy="backward")
        .with_columns(
            pl.col("spot").fill_null(strategy="forward"),
            (
                (pl.lit(expiry) - pl.col("ts_ns")).dt.total_nanoseconds()
                / (365.25 * 24 * 3600 * 1e9)
            ).alias("T"),
        )
    )
    if forward_ts is not None:
        result = result.join_asof(
            forward_ts.sort("ts_ns"), on="ts_ns", strategy="backward"
        ).with_columns(pl.col("forward").fill_null(strategy="forward"))
    return result


def _compute_iv_batch(
    df: pl.DataFrame,
    price_col: str,
) -> pl.DataFrame:
    """
    Compute BS implied vol for every row in one cdxcore.bs.implied call.

    Expects columns: spot, T, _strike, is_call, {price_col}, bid_price, ask_price.
    Adds columns:    observed_iv, bid_iv, ask_iv.

    is_call is read per-row from df["is_call"], so calls and puts can be mixed.
    """
    F_col = "forward" if "forward" in df.columns else "spot"
    S = df[F_col].to_numpy()
    K = df["_strike"].to_numpy()
    T = df["T"].to_numpy()
    is_call = df["is_call"].to_numpy()  # bool array, one per row

    prices = np.stack(
        [  # (n, 3)
            df[price_col].to_numpy(),
            df["bid_price"].to_numpy(),
            df["ask_price"].to_numpy(),
        ],
        axis=1,
    )

    norm_prices = prices / S.reshape(-1, 1)
    k = np.tile((K / S).reshape(-1, 1), (1, 3))
    sqrtT = np.tile(np.sqrt(np.maximum(T, 0.0)).reshape(-1, 1), (1, 3))
    is_call_tiled = np.tile(is_call.reshape(-1, 1), (1, 3))  # (n, 3) to match k/prices

    valid = (
        np.isfinite(norm_prices).all(axis=1) & np.isfinite(k[:, 0]) & (S > 0) & (sqrtT[:, 0] > 0)
    )
    ivs = np.full((len(df), 3), np.nan)
    if valid.any():
        ivs[valid] = _cdx_bs.implied(
            k=k[valid],
            prices=norm_prices[valid],
            is_call=is_call_tiled[valid],  # (n_valid, 3) bool array
            sqrtT=sqrtT[valid],
        )
    obs_iv, bid_iv_, ask_iv_ = ivs[:, 0], ivs[:, 1], ivs[:, 2]
    bid_iv_, ask_iv_ = np.minimum(bid_iv_, ask_iv_), np.maximum(bid_iv_, ask_iv_)
    return df.with_columns(
        [
            pl.Series("observed_iv", obs_iv.tolist()),
            pl.Series("bid_iv", bid_iv_.tolist()),
            pl.Series("ask_iv", ask_iv_.tolist()),
        ]
    )


def kalman_filter_window(df_quotes, df_matched):
    pass


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

    df_trades_result, df_quotes_result = ekf.kalman_filters_price_from_events(df_events)

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

    df_iv_trades_result, df_iv_quotes_result = ekf.kalman_filters_iv_from_events(df_events)

    return df_trades_result, df_iv_trades_result, df_quotes_result, df_iv_quotes_result


def kalman_filter_from_matched(
    df_quotes: pl.DataFrame,
    df_matched: pl.DataFrame,
    df_spy: pl.DataFrame,
    strike: float,
    expiry: datetime.datetime,
    option_type: str = "C",
):
    """
    Train Kalman filters (price and IV) on quotes; test on matched trades.

    Parameters
    ----------
    df_quotes : raw quotes (MassiveClient.SCHEMA_QUOTES)
    df_matched : output of match_taq (SCHEMA_TAQ)
    df_spy : SPY trades (MassiveClient.SCHEMA_TRADES), used for spot
    strike, expiry, option_type : option contract parameters for BS IV
    """
    spy_ts = events.filter_df(df_spy).select(["ts_ns", "price"]).rename({"price": "spot"})
    is_call = option_type == "C"

    # ── Price Kalman ──────────────────────────────────────────────────────────
    df_trades_result, df_quotes_result = ekf.kalman_filters_price(df_quotes, df_matched)

    # ── Add spot, T, strike, and option type for _compute_iv_batch ───────────
    tags = [pl.lit(strike).alias("_strike"), pl.lit(is_call).alias("is_call")]
    df_quotes_result = _add_spot_T(df_quotes_result, spy_ts, expiry).with_columns(tags)
    df_trades_result = _add_spot_T(df_trades_result, spy_ts, expiry).with_columns(tags)

    # ── BS IV (single cdxcore.bs pass per df) ────────────────────────────────
    df_q_with_iv = _compute_iv_batch(df_quotes_result, price_col="observed_price").with_columns(
        (((pl.col("ask_iv") - pl.col("bid_iv")) * 100 / 2) ** 2).alias("R_iv")
    )
    df_t_with_iv = _compute_iv_batch(df_trades_result, price_col="price")

    # ── IV Kalman ─────────────────────────────────────────────────────────────
    df_iv_trades_result, df_iv_quotes_result = ekf.kalman_filters_iv(df_q_with_iv, df_t_with_iv)

    return df_trades_result, df_iv_trades_result, df_quotes_result, df_iv_quotes_result


def build_implied_forward(
    raw_calls: dict[float, pl.DataFrame],
    raw_puts: dict[float, pl.DataFrame],
    spy_ts: pl.DataFrame,
    max_leg_staleness_s: float = 60.0,
) -> pl.DataFrame:
    """
    Estimate the true implied forward F from put-call parity, with dynamic ATM selection.

    For each overlapping strike K, builds a Δ(K, t) = C_mid(t) - P_mid(t) series
    (staleness-filtered). All series plus the SPY spot are joined onto a common spine.
    At each spine tick, the two strikes nearest to the current spot that have valid Δ
    are selected, and the 2×2 system is solved:

        Δi = C(Ki) - P(Ki) = e^{-rT} * (F - Ki)
        e^{-rT} = (Δ1 - Δ2) / (K2 - K1)
        F       = K1 + Δ1 / e^{-rT}

    This recovers the true (undiscounted) forward F = S*e^{(r-q)T} without any rate
    or dividend data, and re-selects the ATM pair as spot moves intraday.

    Falls back to the single-strike undiscounted formula when only one valid Δ is
    available at a given tick.

    `spy_ts` must have columns [ts_ns, spot].
    """
    max_stale_ns = int(max_leg_staleness_s * 1e9)
    overlapping = sorted(set(raw_calls) & set(raw_puts))

    if not overlapping:
        raise RuntimeError("No overlapping call/put strikes to estimate implied forward")

    def _build_delta(k: float) -> pl.DataFrame:
        c = (
            raw_calls[k]
            .sort("ts_ns")
            .select(
                [
                    pl.col("ts_ns").alias("c_ts"),
                    ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("c_mid"),
                ]
            )
        )
        p = (
            raw_puts[k]
            .sort("ts_ns")
            .select(
                [
                    pl.col("ts_ns").alias("p_ts"),
                    ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("p_mid"),
                ]
            )
        )
        spine = (
            pl.concat(
                [
                    c.select(pl.col("c_ts").alias("ts_ns")),
                    p.select(pl.col("p_ts").alias("ts_ns")),
                ]
            )
            .unique()
            .sort("ts_ns")
        )
        col = f"delta_{int(k)}"
        return (
            spine.join_asof(c, left_on="ts_ns", right_on="c_ts", strategy="backward")
            .join_asof(p, left_on="ts_ns", right_on="p_ts", strategy="backward")
            .filter(
                pl.col("c_mid").is_not_null()
                & pl.col("p_mid").is_not_null()
                & ((pl.col("ts_ns") - pl.col("c_ts")).dt.total_nanoseconds() <= max_stale_ns)
                & ((pl.col("ts_ns") - pl.col("p_ts")).dt.total_nanoseconds() <= max_stale_ns)
            )
            .with_columns((pl.col("c_mid") - pl.col("p_mid")).alias(col))
            .select(["ts_ns", col])
        )

    # Build per-strike Δ series and join onto a common spine with SPY spot
    delta_dfs = {k: _build_delta(k) for k in overlapping}
    delta_cols = [f"delta_{int(k)}" for k in overlapping]

    spine = (
        pl.concat([df.select("ts_ns") for df in delta_dfs.values()] + [spy_ts.select("ts_ns")])
        .unique()
        .sort("ts_ns")
    )

    wide = spine.join_asof(spy_ts.sort("ts_ns"), on="ts_ns", strategy="backward")
    for k, df in delta_dfs.items():
        wide = wide.join_asof(df, on="ts_ns", strategy="backward")

    # At each tick: pick the two nearest-ATM valid strikes and solve for F
    def _pick_forward(row: dict) -> float | None:
        spot = row["spot"]
        if spot is None:
            return None
        valid = [
            (k, row[f"delta_{int(k)}"]) for k in overlapping if row[f"delta_{int(k)}"] is not None
        ]
        if not valid:
            return None
        if len(valid) == 1:
            k, d = valid[0]
            f = k + d  # undiscounted single-strike fallback
            return f if f > 0 else None
        valid.sort(key=lambda x: abs(x[0] - spot))
        (ka, da), (kb, db) = valid[0], valid[1]
        k1, d1, k2, d2 = (ka, da, kb, db) if ka < kb else (kb, db, ka, da)
        e_rT = (d1 - d2) / (k2 - k1)
        if e_rT <= 0:
            return None
        f = k1 + d1 / e_rT
        return f if f > 0 else None

    return (
        wide.filter(pl.col("spot").is_not_null())
        .with_columns(
            pl.struct(["spot"] + delta_cols)
            .map_elements(_pick_forward, return_dtype=pl.Float64)
            .alias("forward")
        )
        .filter(pl.col("forward").is_not_null())
        .select(["ts_ns", "forward"])
    )


def build_iv_surface(
    df_tickers: pl.DataFrame,
    trade_date,
    taq_config: TAQMatchConfig = TAQMatchConfig(),
    df_spy=None,
    min_trades: int = 1,
    max_iv: float = 5.0,
) -> pl.DataFrame:
    """
    Run IV Kalman filter for all options in df_tickers using matched trades.
    Processes calls and puts together in the same batched IV computation.

    Structured in three phases:
      1. Per-ticker: fetch data + price Kalman → collect processed frames.
      2. Batch: concat all frames, compute all implied vols in one cdxcore.bs call.
      3. Per-ticker: IV Kalman on each option's IV-enriched frame.

    Returns a long DataFrame: ts_ns, strike, is_call, bid_iv, ask_iv, kalman_iv.
    """
    from quantdata.massive.client import MassiveClient

    client = MassiveClient()

    if df_spy is None:
        df_spy = client.fetch_trades("SPY", trade_date)

    spy_ts = events.filter_df(df_spy).select(["ts_ns", "price"]).rename({"price": "spot"})
    candidates = df_tickers.filter(pl.col("trade_count") >= min_trades)

    # ── Phase 1: fetch + price Kalman ────────────────────────────────────────
    # spot/T/forward tagging happens after the forward is estimated (see below)
    ticker_meta = []  # (strike, is_call, expiry) in insertion order
    quote_frames = []  # price-Kalman results, not yet tagged with spot/T/forward
    trade_frames = []
    raw_calls_q = {}  # (expiry, strike) → raw df_quotes, for forward estimation
    raw_puts_q = {}

    _bad_checks = [
        ("crossed", pl.col("bid_price") >= pl.col("ask_price")),
        ("sub-tick bid", pl.col("bid_price") < 0.011),
        ("zero bid size", (pl.col("bid_size") <= 0) | pl.col("bid_size").is_null()),
        ("zero ask size", (pl.col("ask_size") <= 0) | pl.col("ask_size").is_null()),
    ]
    _bad_expr = (
        (pl.col("bid_price") >= pl.col("ask_price"))
        | (pl.col("bid_price") < 0.011)
        | (pl.col("bid_size") <= 0)
        | pl.col("bid_size").is_null()
        | (pl.col("ask_size") <= 0)
        | pl.col("ask_size").is_null()
    )

    for row in candidates.iter_rows(named=True):
        ticker = row["ticker"]
        try:
            strike, expiry, otype = utils.parse_option_ticker(ticker)
            is_call = otype == "C"

            df_quotes = client.fetch_quotes(ticker, trade_date)
            df_matched = match_taq(ticker, trade_date, config=taq_config)

            if df_quotes.is_empty():
                print(f"Skipping {ticker}: no quotes")
                continue

            _bad_quotes = df_quotes.filter(_bad_expr)
            if _bad_quotes.height >= 10:
                _labels = [
                    f"{lbl}×{_bad_quotes.filter(expr).height}"
                    for lbl, expr in _bad_checks
                    if _bad_quotes.filter(expr).height > 0
                ]
                print(f"Skipping {ticker}: {_bad_quotes.height} bad quotes ({', '.join(_labels)})")
                continue
            elif _bad_quotes.height > 0:
                _labels = [
                    f"{lbl}×{len(ts)} (first={ts[0]})"
                    for lbl, expr in _bad_checks
                    if (ts := _bad_quotes.filter(expr)["ts_ns"].to_list())
                ]
                print(f"{ticker}: dropping {_bad_quotes.height} bad quotes ({', '.join(_labels)})")
                df_quotes = df_quotes.filter(~_bad_expr)

            if df_matched.filter(pl.col("quote_ts_ns").is_not_null()).is_empty():
                print(f"Skipping {ticker}: no matched trades")
                continue

            df_trades_r, df_quotes_r = ekf.kalman_filters_price(df_quotes, df_matched)

            quote_frames.append(df_quotes_r)
            trade_frames.append(df_trades_r)
            ticker_meta.append((strike, is_call, expiry))

            if is_call:
                raw_calls_q[(expiry, strike)] = df_quotes
            else:
                raw_puts_q[(expiry, strike)] = df_quotes

        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if not ticker_meta:
        raise RuntimeError("No options produced valid data")

    # ── Implied forward per expiry ────────────────────────────────────────────
    expiries_seen = sorted({e for _, _, e in ticker_meta})
    forward_by_expiry: dict = {}
    for exp in expiries_seen:
        calls_at = {k: df for (e, k), df in raw_calls_q.items() if e == exp}
        puts_at = {k: df for (e, k), df in raw_puts_q.items() if e == exp}
        overlapping = set(calls_at) & set(puts_at)
        if overlapping:
            try:
                df_fwd = build_implied_forward(calls_at, puts_at, spy_ts=spy_ts)
                print(
                    f"Expiry {exp.date()}: implied forward from {len(overlapping)} overlapping strike(s), dynamic ATM selection"
                )
                forward_by_expiry[exp] = df_fwd
            except Exception as e:
                print(
                    f"Expiry {exp.date()}: forward estimation failed ({e}), falling back to SPY spot"
                )
                forward_by_expiry[exp] = spy_ts.rename({"spot": "forward"})
        else:
            print(f"Expiry {exp.date()}: no overlapping strikes, using SPY spot as forward")
            forward_by_expiry[exp] = spy_ts.rename({"spot": "forward"})

    # ── Tag frames with spot, T, and forward ─────────────────────────────────
    tagged_quote_frames = []
    tagged_trade_frames = []
    for i, (strike, is_call, expiry) in enumerate(ticker_meta):
        tags = [pl.lit(strike).alias("_strike"), pl.lit(is_call).alias("is_call")]
        fwd_ts = forward_by_expiry[expiry]
        tagged_quote_frames.append(
            _add_spot_T(quote_frames[i], spy_ts, expiry, forward_ts=fwd_ts).with_columns(tags)
        )
        tagged_trade_frames.append(
            _add_spot_T(trade_frames[i], spy_ts, expiry, forward_ts=fwd_ts).with_columns(tags)
        )

    # ── Phase 2: batch IV computation across all options at once ──────────────
    all_quotes_iv = _compute_iv_batch(
        pl.concat(tagged_quote_frames), price_col="observed_price"
    ).with_columns((((pl.col("ask_iv") - pl.col("bid_iv")) * 100 / 2) ** 2).alias("R_iv"))
    all_trades_iv = _compute_iv_batch(pl.concat(tagged_trade_frames), price_col="price")

    # ── Phase 3: per-ticker IV Kalman ─────────────────────────────────────────
    iv_cols = ["bid_iv", "ask_iv", "kalman_iv"]
    frames = []

    for strike, is_call, expiry in ticker_meta:
        try:
            mask = (pl.col("_strike") == strike) & (pl.col("is_call") == is_call)
            df_q_iv = all_quotes_iv.filter(mask)
            df_t_iv = all_trades_iv.filter(mask)

            _, df_iv_q = ekf.kalman_filters_iv(df_q_iv, df_t_iv)

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
                .with_columns(
                    [
                        pl.lit(strike).alias("strike"),
                        pl.lit(is_call).alias("is_call"),
                    ]
                )
            )
        except Exception as e:
            print(f"Skipping strike {strike} ({'C' if is_call else 'P'}): {e}")

    if not frames:
        raise RuntimeError("No options produced valid IV data")

    return pl.concat(frames).sort("ts_ns")
