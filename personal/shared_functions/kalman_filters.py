import datetime

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytz
from cdxcore.bs import bs as _cdx_bs
from scipy.stats import norm as _norm

import personal.shared_functions.black_scholes as black_scholes
import personal.shared_functions.ekf as ekf
import personal.shared_functions.events as events
import personal.shared_functions.utils as utils
from quantdata.massive.taq import TAQMatchConfig, match_taq


def _bs_price_from_iv(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """Undiscounted Black-Scholes price: C = F·N(d1) - K·N(d2)."""
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    valid = (
        np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(F)
        & (F > 0)
        & np.isfinite(K)
        & (K > 0)
        & (sqrtT > 0)
    )
    price = np.full(len(F), np.nan)
    if valid.any():
        sv = sigma[valid]
        sv_sqrtT = sv * sqrtT[valid]
        d1 = (np.log(F[valid] / K[valid]) + 0.5 * sv**2 * T[valid]) / sv_sqrtT
        d2 = d1 - sv_sqrtT
        call_p = F[valid] * _norm.cdf(d1) - K[valid] * _norm.cdf(d2)
        put_p = K[valid] * _norm.cdf(-d2) - F[valid] * _norm.cdf(-d1)
        price[valid] = np.where(is_call[valid], call_p, put_p)
    return price


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
        result = (
            result.join_asof(forward_ts.sort("ts_ns"), on="ts_ns", strategy="backward")
            .with_columns(pl.col("forward").fill_null(strategy="forward"))
            .with_columns(pl.col("forward").fill_null(pl.col("spot")))
        )
        if "discount_factor" in forward_ts.columns:
            result = result.with_columns(pl.col("discount_factor").fill_null(strategy="forward"))
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
    simple: bool = True,
    discount_factor_clip: tuple[float, float] | None = (0.998, 1.002),
) -> pl.DataFrame:
    """
    Estimate the true implied forward F from put-call parity, with dynamic ATM selection.

    When simple=True, skips the put-call parity estimation entirely and returns
    forward = spot with discount_factor = 1.0.  This is more robust when the
    estimation noise dominates the true carry signal (e.g. same-day SPY options).

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
    if simple:
        return (
            spy_ts.select(["ts_ns", "spot"])
            .rename({"spot": "forward"})
            .with_columns(pl.lit(1.0).alias("discount_factor"))
        )

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
        col = f"delta_{round(k * 100)}"
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
    delta_cols = [f"delta_{round(k * 100)}" for k in overlapping]

    spine = (
        pl.concat([df.select("ts_ns") for df in delta_dfs.values()] + [spy_ts.select("ts_ns")])
        .unique()
        .sort("ts_ns")
    )

    wide = spine.join_asof(spy_ts.sort("ts_ns"), on="ts_ns", strategy="backward")
    for k, df in delta_dfs.items():
        wide = wide.join_asof(df, on="ts_ns", strategy="backward")

    # Vectorized 2-strike solver:
    # 1. find the 2 valid strikes nearest to spot
    # 2. solve  e^{-rT} = (Δ1 - Δ2) / (K2 - K1),  F = K1 + Δ1 / e^{-rT}
    ks = list(overlapping)
    col_k = lambda k: f"delta_{round(k * 100)}"
    is_valid = lambda k: pl.col(col_k(k)).is_not_null()
    abs_dist = lambda k: (pl.lit(float(k)) - pl.col("spot")).abs()

    return (
        wide.filter(pl.col("spot").is_not_null())
        # nearest valid strike to spot
        .with_columns(
            pl.min_horizontal(
                [pl.when(is_valid(k)).then(abs_dist(k)).otherwise(None) for k in ks]
            ).alias("_dist1"),
        )
        .with_columns(
            pl.max_horizontal(
                [
                    pl.when(is_valid(k) & (abs_dist(k) == pl.col("_dist1")))
                    .then(pl.lit(float(k)))
                    .otherwise(None)
                    for k in ks
                ]
            ).alias("_ka"),
        )
        # second nearest valid strike (excluding _ka)
        .with_columns(
            pl.min_horizontal(
                [
                    pl.when(is_valid(k) & (pl.lit(float(k)) != pl.col("_ka")))
                    .then(abs_dist(k))
                    .otherwise(None)
                    for k in ks
                ]
            ).alias("_dist2"),
        )
        .with_columns(
            pl.max_horizontal(
                [
                    pl.when(
                        is_valid(k)
                        & (pl.lit(float(k)) != pl.col("_ka"))
                        & (abs_dist(k) == pl.col("_dist2"))
                    )
                    .then(pl.lit(float(k)))
                    .otherwise(None)
                    for k in ks
                ]
            ).alias("_kb"),
        )
        # order as k1 < k2
        .with_columns(
            pl.min_horizontal(["_ka", "_kb"]).alias("_k1"),
            pl.max_horizontal(["_ka", "_kb"]).alias("_k2"),
        )
        # look up deltas for k1 and k2
        .with_columns(
            pl.coalesce(
                [pl.when(pl.col("_k1") == float(k)).then(pl.col(col_k(k))) for k in ks]
            ).alias("_d1"),
            pl.coalesce(
                [pl.when(pl.col("_k2") == float(k)).then(pl.col(col_k(k))) for k in ks]
            ).alias("_d2"),
        )
        # solve 2×2 system
        .with_columns(
            pl.when(pl.col("_k2") > pl.col("_k1"))
            .then((pl.col("_d1") - pl.col("_d2")) / (pl.col("_k2") - pl.col("_k1")))
            .otherwise(None)
            .alias("_e_rT"),
        )
        .with_columns(
            pl.when(pl.col("_e_rT") > 0)
            .then(
                pl.col("_e_rT").clip(*discount_factor_clip)
                if discount_factor_clip is not None
                else pl.col("_e_rT")
            )
            .otherwise(None)
            .alias("discount_factor"),
        )
        .with_columns(
            pl.when(pl.col("discount_factor").is_not_null())
            .then(pl.col("_k1") + pl.col("_d1") / pl.col("discount_factor"))
            .otherwise(None)
            .alias("_fwd"),
        )
        .with_columns(
            pl.when(pl.col("_fwd") > 0).then(pl.col("_fwd")).otherwise(None).alias("forward"),
        )
        .filter(pl.col("forward").is_not_null())
        .select(["ts_ns", "forward", "discount_factor"])
    )


def build_iv_surface(
    df_tickers: pl.DataFrame,
    trade_date,
    taq_config: TAQMatchConfig = TAQMatchConfig(),
    df_spy=None,
    min_trades: int = 1,
    max_iv: float = 5.0,
    simple=True,
    discount_factor_clip: tuple[float, float] | None = (0.998, 1.002),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run IV Kalman filter for all options in df_tickers, operating directly on
    implied vols (no price Kalman step).

    Structured in three phases:
      1. Per-ticker: fetch quotes + matched trades, process quotes.
      2. Batch: concat all frames, compute all implied vols in one cdxcore.bs call.
      3. Per-ticker: IV Kalman trained on quotes (in-sample), tested on matched trades (out-of-sample).

    Returns
    -------
    (df_quotes_iv, df_trades_iv)
        Both are long DataFrames keyed by (ts_ns, strike, is_call).
        df_quotes_iv  — train: observed_iv, bid_iv, ask_iv, kalman_iv
        df_trades_iv  — test:  observed_iv, kalman_matched, trade_residual
    """
    from quantdata.massive.client import MassiveClient

    client = MassiveClient()

    if df_spy is None:
        df_spy = client.fetch_trades("SPY", trade_date)

    spy_ts = events.filter_df(df_spy).select(["ts_ns", "price"]).rename({"price": "spot"})
    candidates = df_tickers.filter(pl.col("trade_count") >= min_trades)

    # ── Phase 1: fetch + process quotes ──────────────────────────────────────
    ticker_meta = []  # (strike, is_call, expiry) in insertion order
    quote_frames = []  # processed quotes, not yet tagged with spot/T/forward
    trade_frames = []  # matched trades (quote_ts_ns not null)
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
            # if _bad_quotes.height >= 10:
            #     _labels = [
            #         f"{lbl}×{_bad_quotes.filter(expr).height}"
            #         for lbl, expr in _bad_checks
            #         if _bad_quotes.filter(expr).height > 0
            #     ]
            #     print(f"Skipping {ticker}: {_bad_quotes.height} bad quotes ({', '.join(_labels)})")
            #     continue
            # elif _bad_quotes.height > 0:
            if _bad_quotes.height > 0:
                _labels = [
                    f"{lbl}×{len(ts)} (first={ts[0]})"
                    for lbl, expr in _bad_checks
                    if (ts := _bad_quotes.filter(expr)["ts_ns"].to_list())
                ]
                print(f"{ticker}: dropping {_bad_quotes.height} bad quotes ({', '.join(_labels)})")
                df_quotes = df_quotes.filter(~_bad_expr)

            df_matched_valid = df_matched.filter(pl.col("quote_ts_ns").is_not_null())
            if df_matched_valid.is_empty():
                print(f"Skipping {ticker}: no matched trades")
                continue

            quote_frames.append(events.process_quotes(df_quotes))
            trade_frames.append(df_matched_valid)
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
                df_fwd = build_implied_forward(
                    calls_at,
                    puts_at,
                    spy_ts=spy_ts,
                    simple=simple,
                    discount_factor_clip=discount_factor_clip,
                )
                nn_df = df_fwd["discount_factor"].is_not_null().sum()
                print(
                    f"Expiry {exp.date()}: implied forward from {len(overlapping)} overlapping "
                    f"strike(s), {nn_df}/{df_fwd.shape[0]} rows with non-null discount_factor"
                )
                forward_by_expiry[exp] = df_fwd
            except Exception as e:
                print(
                    f"Expiry {exp.date()}: forward estimation failed ({e}), falling back to SPY spot"
                )
                forward_by_expiry[exp] = spy_ts.rename({"spot": "forward"}).with_columns(
                    pl.lit(None, dtype=pl.Float64).alias("discount_factor")
                )
        else:
            print(f"Expiry {exp.date()}: no overlapping strikes, using SPY spot as forward")
            forward_by_expiry[exp] = spy_ts.rename({"spot": "forward"}).with_columns(
                pl.lit(None, dtype=pl.Float64).alias("discount_factor")
            )

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
    ).with_columns((((pl.col("ask_iv") - pl.col("bid_iv")) / 2) ** 2).alias("R_iv"))
    all_trades_iv = _compute_iv_batch(pl.concat(tagged_trade_frames), price_col="price")

    # ── Phase 3: per-ticker IV Kalman ─────────────────────────────────────────
    iv_cols = ["observed_iv", "bid_iv", "ask_iv", "kalman_iv"]
    quote_result_frames = []
    trade_result_frames = []

    for strike, is_call, expiry in ticker_meta:
        try:
            mask = (pl.col("_strike") == strike) & (pl.col("is_call") == is_call)
            df_q_iv = all_quotes_iv.filter(mask)
            df_t_iv = all_trades_iv.filter(mask)

            df_iv_trades_r, df_iv_quotes_r = ekf.kalman_filters_iv(df_q_iv, df_t_iv)

            meta_cols = [pl.lit(strike).alias("strike"), pl.lit(is_call).alias("is_call")]

            fwd_col = "forward" if "forward" in df_iv_quotes_r.columns else "spot"
            K_sc = np.full(len(df_iv_quotes_r), strike)
            ic_sc = np.full(len(df_iv_quotes_r), is_call, dtype=bool)
            kalman_prices_q = _bs_price_from_iv(
                df_iv_quotes_r[fwd_col].to_numpy().astype(float),
                K_sc,
                df_iv_quotes_r["T"].to_numpy().astype(float),
                df_iv_quotes_r["kalman_iv"].to_numpy().astype(float),
                ic_sc,
            )

            extra_cols = [c for c in ["forward", "discount_factor"] if c in df_iv_quotes_r.columns]
            quote_result_frames.append(
                df_iv_quotes_r.select(["ts_ns"] + iv_cols + ["bid_price", "ask_price"] + extra_cols)
                .with_columns(
                    [
                        pl.when((pl.col(c) > max_iv) | (pl.col(c) < 0))
                        .then(None)
                        .otherwise(pl.col(c))
                        .alias(c)
                        for c in iv_cols
                    ]
                )
                .with_columns(pl.Series("kalman_price", kalman_prices_q.tolist()))
                .with_columns(meta_cols)
            )

            fwd_col_t = "forward" if "forward" in df_iv_trades_r.columns else "spot"
            K_t = np.full(len(df_iv_trades_r), strike)
            ic_t = np.full(len(df_iv_trades_r), is_call, dtype=bool)
            kalman_prices_t = _bs_price_from_iv(
                df_iv_trades_r[fwd_col_t].to_numpy().astype(float),
                K_t,
                df_iv_trades_r["T"].to_numpy().astype(float),
                df_iv_trades_r["kalman_matched"].to_numpy().astype(float),
                ic_t,
            )

            trade_result_frames.append(
                df_iv_trades_r.select(
                    [
                        "ts_ns",
                        "observed_iv",
                        "bid_iv",
                        "ask_iv",
                        "bid_price",
                        "ask_price",
                        "kalman_matched",
                        "trade_residual",
                    ]
                )
                .with_columns(
                    pl.when((pl.col("observed_iv") > max_iv) | (pl.col("observed_iv") < 0))
                    .then(None)
                    .otherwise(pl.col("observed_iv"))
                    .alias("observed_iv")
                )
                .with_columns(pl.Series("kalman_price", kalman_prices_t.tolist()))
                .with_columns(meta_cols)
            )

        except Exception as e:
            print(f"Skipping strike {strike} ({'C' if is_call else 'P'}): {e}")

    if not quote_result_frames:
        raise RuntimeError("No options produced valid IV data")

    df_quotes_iv = pl.concat(quote_result_frames).sort("ts_ns")
    df_trades_iv = (
        pl.concat(trade_result_frames).sort("ts_ns") if trade_result_frames else pl.DataFrame()
    )
    return df_quotes_iv, df_trades_iv
