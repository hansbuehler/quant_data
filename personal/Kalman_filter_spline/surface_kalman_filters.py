"""
surface_kalman_filters.py

Two public functions:

build_iv_observations(df_tickers, trade_date, df_spy, min_trades)
    Phases 1+2 of the IV pipeline: fetch, clean, implied forward, compute IV.
    No per-option Kalman step.
    Returns (df_quotes_iv, df_trades_iv).
    Columns (quotes):  ts_ns, strike, is_call, log_moneyness, T,
                       observed_iv, bid_iv, ask_iv, R_iv
    Columns (trades):  ts_ns, quote_ts_ns, strike, is_call,
                       log_moneyness, T, observed_iv

run_spline_surface_kalman(df_quotes_iv, df_trades_iv, cfg)
    Joint Kalman filter: state θ ∈ ℝ^n_params updated by every quote.
    Returns (df_quotes_result, df_trades_result).
    Extra columns (quotes): spline_iv
    Extra columns (trades): spline_matched, trade_residual
"""

import numpy as np
import polars as pl
import surface_model as sm
from scipy.stats import norm as _norm

import personal.shared_functions.utils as utils


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


# ---------------------------------------------------------------------------
# Phase 1+2 pipeline — build IV observations (no Kalman)
# ---------------------------------------------------------------------------


def build_iv_observations(
    df_tickers: pl.DataFrame,
    trade_date,
    df_spy: pl.DataFrame,
    min_trades: int = 1,
    taq_config=None,
    simple: bool = False,
    discount_factor_clip: tuple[float, float] | None = (0.998, 1.002),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run Phases 1+2 of the IV pipeline (no per-option Kalman step).

    Identical to kalman_filters.build_iv_surface up to Phase 2, then stops.
    Adds log_moneyness = log(K/F) before returning.
    """
    import personal.shared_functions.events as events
    from personal.shared_functions.kalman_filters import (
        _add_spot_T,
        _compute_iv_batch,
        build_implied_forward,
    )
    from quantdata.massive.client import MassiveClient
    from quantdata.massive.taq import TAQMatchConfig, match_taq

    if taq_config is None:
        taq_config = TAQMatchConfig()

    client = MassiveClient()
    spy_ts = events.filter_df(df_spy).select(["ts_ns", "price"]).rename({"price": "spot"})
    candidates = df_tickers.filter(pl.col("trade_count") >= min_trades)

    _bad_expr = (
        (pl.col("bid_price") >= pl.col("ask_price"))
        | (pl.col("bid_price") < 0.011)
        | (pl.col("bid_size") <= 0)
        | pl.col("bid_size").is_null()
        | (pl.col("ask_size") <= 0)
        | pl.col("ask_size").is_null()
    )

    # ── Phase 1: fetch + process quotes per ticker ────────────────────────────
    ticker_meta = []
    quote_frames = []
    trade_frames = []
    raw_calls_q: dict = {}
    raw_puts_q: dict = {}

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

            bad = df_quotes.filter(_bad_expr)
            if bad.height >= 10:
                print(f"Skipping {ticker}: {bad.height} bad quotes")
                continue
            if bad.height > 0:
                df_quotes = df_quotes.filter(~_bad_expr)

            df_matched_valid = df_matched.filter(pl.col("quote_ts_ns").is_not_null())
            if df_matched_valid.is_empty():
                print(f"Skipping {ticker}: no matched trades")
                continue

            quote_frames.append(events.process_quotes(df_quotes))
            trade_frames.append(df_matched_valid)
            ticker_meta.append((strike, is_call, expiry))

            key = (expiry, strike)
            if is_call:
                raw_calls_q[key] = df_quotes
            else:
                raw_puts_q[key] = df_quotes

        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")

    if not ticker_meta:
        raise RuntimeError("No options produced valid data")

    # ── Implied forward per expiry ────────────────────────────────────────────
    expiries_seen = sorted({exp for _, _, exp in ticker_meta})
    forward_by_expiry: dict = {}
    for exp in expiries_seen:
        calls_at = {k: df for (e, k), df in raw_calls_q.items() if e == exp}
        puts_at = {k: df for (e, k), df in raw_puts_q.items() if e == exp}
        if set(calls_at) & set(puts_at):
            try:
                forward_by_expiry[exp] = build_implied_forward(
                    calls_at,
                    puts_at,
                    spy_ts=spy_ts,
                    simple=simple,
                    discount_factor_clip=discount_factor_clip,
                )
                print(
                    f"Expiry {exp.date()}: implied forward from {len(set(calls_at) & set(puts_at))} strike(s)"
                )
            except Exception as exc:
                print(f"Expiry {exp.date()}: forward failed ({exc}), using spot")
                forward_by_expiry[exp] = spy_ts.rename({"spot": "forward"})
        else:
            print(f"Expiry {exp.date()}: no overlapping strikes, using spot as forward")
            forward_by_expiry[exp] = spy_ts.rename({"spot": "forward"})

    # ── Tag frames with spot, T, forward ─────────────────────────────────────
    tagged_quotes = []
    tagged_trades = []
    for i, (strike, is_call, expiry) in enumerate(ticker_meta):
        tags = [pl.lit(strike).alias("_strike"), pl.lit(is_call).alias("is_call")]
        fwd_ts = forward_by_expiry[expiry]
        tagged_quotes.append(
            _add_spot_T(quote_frames[i], spy_ts, expiry, fwd_ts).with_columns(tags)
        )
        tagged_trades.append(
            _add_spot_T(trade_frames[i], spy_ts, expiry, fwd_ts).with_columns(tags)
        )

    # ── Phase 2: batch IV computation ─────────────────────────────────────────
    all_q = pl.concat(tagged_quotes)
    all_t = pl.concat(tagged_trades)

    quotes_iv = _compute_iv_batch(all_q, price_col="observed_price").with_columns(
        (((pl.col("ask_iv") - pl.col("bid_iv")) / 2) ** 2).alias("R_iv")
    )
    trades_iv = _compute_iv_batch(all_t, price_col="price")

    # ── Add log_moneyness = log(K / F) ────────────────────────────────────────
    def _add_log_moneyness(df: pl.DataFrame) -> pl.DataFrame:
        fwd_col = "forward" if "forward" in df.columns else "spot"
        fwd = df[fwd_col].to_numpy().astype(float)
        k = df["_strike"].to_numpy().astype(float)
        log_m = np.where((fwd > 0) & np.isfinite(fwd) & (k > 0), np.log(k / fwd), np.nan)
        return df.with_columns(pl.Series("log_moneyness", log_m.tolist()))

    quotes_iv = _add_log_moneyness(quotes_iv).rename({"_strike": "strike"})
    trades_iv = _add_log_moneyness(trades_iv).rename({"_strike": "strike"})

    # ── Select and sort final columns ─────────────────────────────────────────
    q_cols = [
        "ts_ns",
        "strike",
        "is_call",
        "log_moneyness",
        "T",
        "observed_iv",
        "bid_iv",
        "ask_iv",
        "R_iv",
        "bid_price",
        "ask_price",
        "forward",
        "discount_factor",
    ]
    t_cols = [
        "ts_ns",
        "quote_ts_ns",
        "strike",
        "is_call",
        "log_moneyness",
        "T",
        "observed_iv",
        "bid_iv",
        "ask_iv",
        "bid_price",
        "ask_price",
        "forward",
        "discount_factor",
    ]

    quotes_iv = quotes_iv.select([c for c in q_cols if c in quotes_iv.columns]).sort("ts_ns")
    trades_iv = trades_iv.select([c for c in t_cols if c in trades_iv.columns]).sort("ts_ns")

    return quotes_iv, trades_iv


# ---------------------------------------------------------------------------
# Kalman filter step
# ---------------------------------------------------------------------------


def _kf_step(
    theta: np.ndarray,
    P: np.ndarray,
    H: np.ndarray,
    z_hat: float,
    z: float,
    R: float,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One EKF predict+update step with scalar observation.

    z_hat is the nonlinear prediction h(theta); H is the Jacobian ∂h/∂theta.
    The innovation z - z_hat uses the nonlinear prediction, not H @ theta.
    """
    P_pred = P + Q
    innov = z - z_hat
    S = float(H @ P_pred @ H) + R
    K = (P_pred @ H) / S  # shape (n,)
    theta_new = theta + K * innov
    P_new = P_pred - np.outer(K, H) @ P_pred
    P_new = (P_new + P_new.T) * 0.5  # enforce symmetry
    return theta_new, P_new


# ---------------------------------------------------------------------------
# Joint spline surface Kalman filter
# ---------------------------------------------------------------------------


def run_spline_surface_kalman(
    df_quotes_iv: pl.DataFrame,
    df_trades_iv: pl.DataFrame,
    cfg,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Run the spline-state Kalman filter on the merged IV observation stream.

    At each quote tick (in time order) the shared state θ ∈ ℝ^n_params is
    updated using H = basis_row(log_moneyness, T).  Trades are evaluated
    out-of-sample against the state recorded at their matched quote timestamp.

    Returns
    -------
    df_quotes_result : quotes with spline_iv column added
    df_trades_result : trades with spline_matched and trade_residual columns added
    df_state         : state history — columns ts_ns + theta_0 … theta_{n-1}
                       Use join_asof backward on ts_ns to look up θ at any time.
    """
    Q = sm.build_Q(cfg)
    P = sm.build_P0(cfg)
    theta = cfg.model.initial_theta(cfg)
    n_p = cfg.n_params

    df_q = df_quotes_iv.sort("ts_ns")
    n_q = len(df_q)

    ks = df_q["log_moneyness"].to_numpy(allow_copy=True).astype(float)
    Ts = df_q["T"].to_numpy(allow_copy=True).astype(float)
    obs_ivs = df_q["observed_iv"].to_numpy(allow_copy=True).astype(float)
    Rs = df_q["R_iv"].to_numpy(allow_copy=True).astype(float)
    ts_arr = df_q["ts_ns"].to_numpy(allow_copy=True)

    spline_ivs = np.full(n_q, np.nan)
    thetas_hist = np.full((n_q, n_p), np.nan)  # state AFTER each step

    for i in range(n_q):
        z_hat, H = cfg.model.observe(theta, ks[i], Ts[i])

        if (
            np.isfinite(obs_ivs[i])
            and np.isfinite(Rs[i])
            and Rs[i] > 0
            and np.isfinite(H).all()
            and np.isfinite(z_hat)
        ):
            theta, P = _kf_step(theta, P, H, z_hat, obs_ivs[i], Rs[i], Q)
        else:
            P = P + Q  # predict-only when observation is invalid

        spline_ivs[i] = cfg.model.observe(theta, ks[i], Ts[i])[0]  # post-update prediction
        thetas_hist[i] = theta

    F_q = df_q["forward"].to_numpy(allow_copy=True).astype(float)
    K_q = df_q["strike"].to_numpy(allow_copy=True).astype(float)
    is_call_q = df_q["is_call"].to_numpy(allow_copy=True)
    kalman_prices_q = _bs_price_from_iv(F_q, K_q, Ts, spline_ivs, is_call_q)

    df_quotes_result = df_q.with_columns(
        [
            pl.Series("spline_iv", spline_ivs.tolist()),
            pl.Series("kalman_price", kalman_prices_q.tolist()),
        ]
    )

    # ── Trade evaluation ──────────────────────────────────────────────────────
    if df_trades_iv.is_empty() or "quote_ts_ns" not in df_trades_iv.columns:
        return (
            df_quotes_result,
            df_trades_iv.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias("spline_matched"),
                    pl.lit(None, dtype=pl.Float64).alias("trade_residual"),
                ]
            ),
            df_state_out,
        )

    # Build state-history DataFrame and join trades backward on quote_ts_ns.
    # Re-use the original Polars series so the dtype (incl. timezone) is preserved.
    # Public columns use "theta_j" names; internal join columns keep "_th{j}" prefix.
    th_cols = {f"_th{j}": thetas_hist[:, j].tolist() for j in range(n_p)}
    df_state = pl.DataFrame({"ts_ns": df_q["ts_ns"], **th_cols})

    # Public state history (clean names for notebook use)
    df_state_out = df_state.rename({f"_th{j}": f"theta_{j}" for j in range(n_p)})

    df_t = df_trades_iv.sort("quote_ts_ns").join_asof(
        df_state, left_on="quote_ts_ns", right_on="ts_ns", strategy="backward"
    )

    # Surface evaluation: apply nonlinear h(theta_at_trade, k, T) for each trade row
    t_ks = df_t["log_moneyness"].to_numpy(allow_copy=True).astype(float)
    t_Ts = df_t["T"].to_numpy(allow_copy=True).astype(float)
    th_t = np.column_stack([df_t[f"_th{j}"].to_numpy(allow_copy=True) for j in range(n_p)])

    def _eval_trade(i: int) -> float:
        th = th_t[i]
        k, T = t_ks[i], t_Ts[i]
        if np.isfinite(th).all() and np.isfinite(k) and np.isfinite(T):
            return cfg.model.observe(th, k, T)[0]
        return np.nan

    spline_matched = np.array([_eval_trade(i) for i in range(len(df_t))])

    F_t = df_t["forward"].to_numpy(allow_copy=True).astype(float)
    K_t = df_t["strike"].to_numpy(allow_copy=True).astype(float)
    T_t = df_t["T"].to_numpy(allow_copy=True).astype(float)
    is_call_t = df_t["is_call"].to_numpy(allow_copy=True)
    kalman_prices_t = _bs_price_from_iv(F_t, K_t, T_t, spline_matched, is_call_t)

    drop = [f"_th{j}" for j in range(n_p)]
    df_trades_result = (
        df_t.with_columns(
            [
                pl.Series("spline_matched", spline_matched.tolist()),
                pl.Series("kalman_price", kalman_prices_t.tolist()),
            ]
        )
        .with_columns((pl.col("observed_iv") - pl.col("spline_matched")).alias("trade_residual"))
        .drop(drop)
    )

    return df_quotes_result, df_trades_result, df_state_out
