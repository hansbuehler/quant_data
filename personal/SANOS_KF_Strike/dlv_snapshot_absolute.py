"""
Snapshot batching and intraday fitting loop.
Absolute-strike variant: prices are C/df; strikes are absolute dollars.

The fit_fn signature is extended with a fwd keyword argument so that the
per-snapshot forward can be passed in (the martingale target changes with F).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from dlv_pricing_absolute import v_atm_from_price


def make_batch_snapshots(df: pl.DataFrame, batch_freq: str) -> pl.DataFrame:
    """
    Aggregate per-option quote data into time-batched snapshots.

    Each row in the output is the last quote for a (batch_ts, strike) pair.

    Parameters
    ----------
    df : DataFrame with columns ts_ns, strike, pure_strike, bid_norm,
         ask_norm, mid_norm, T, forward, log_moneyness
         In the absolute variant: pure_strike = strike (dollars),
         bid_norm/ask_norm = price/df (not price/(F*df)).
    batch_freq : "tick" | "1s" | "1m" | "5m"

    Returns
    -------
    DataFrame sorted by (batch_ts, strike)
    """
    if batch_freq == "tick":
        df = df.with_columns(pl.col("ts_ns").alias("batch_ts"))
    elif batch_freq in ("1s", "1m", "5m"):
        df = df.with_columns(pl.col("ts_ns").dt.truncate(batch_freq).alias("batch_ts"))
    else:
        raise ValueError(f"Unknown batch_freq: {batch_freq!r}")

    return (
        df.sort("ts_ns")
        .group_by(["batch_ts", "strike"])
        .agg(
            [
                pl.col("log_moneyness").last(),
                pl.col("pure_strike").last(),
                pl.col("bid_norm").last(),
                pl.col("ask_norm").last(),
                pl.col("mid_norm").last(),
                pl.col("T").last(),
                pl.col("forward").last(),
            ]
        )
        .sort(["batch_ts", "strike"])
    )


def run_intraday_loop(
    *,
    batch_ts_list: list,
    snaps_calls: pl.DataFrame,
    snaps_puts: pl.DataFrame,
    fit_fn,
    tte_y_med: float,
    calls_only: bool = True,
    puts_only: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """
    Iterate over time snapshots and fit a SANOS density at each one.

    Compared to the normalized variant, this version:
      - extracts fwd_snap per snapshot and passes it to fit_fn as fwd=fwd_snap
      - passes fwd to v_atm_from_price for IV inversion

    fit_fn signature: fit_fn(bid_norm, ask_norm, pure_K, is_calls, prev_density, fwd)
    where bid_norm/ask_norm are C/df (or P/df) and pure_K are absolute dollar strikes.

    Results format:
      r["dlvsqs"]   : np.ndarray of shape (1, N-2)
      r["densities"]: np.ndarray of shape (1, N)
      r["fwd"]      : float forward for this snapshot
    """
    results = []
    prev_density = None
    first_error_shown = False

    for batch_ts in batch_ts_list:
        snap_c = snaps_calls.filter(pl.col("batch_ts") == batch_ts).sort("pure_strike")
        snap_p = snaps_puts.filter(pl.col("batch_ts") == batch_ts).sort("pure_strike")

        T_snap = (
            float(snap_c["T"].median())
            if not snap_c.is_empty()
            else float(snap_p["T"].median())
            if not snap_p.is_empty()
            else tte_y_med
        )
        fwd_snap = (
            float(snap_p["forward"].median())
            if puts_only and not snap_p.is_empty()
            else float(snap_c["forward"].median())
            if not snap_c.is_empty()
            else float("nan")
        )

        # ATM IV: single BS inversion at the near-ATM mid price (C/df or P/df)
        atm_iv = 0.20
        snap_atm = snap_p if puts_only else snap_c
        if not snap_atm.is_empty() and np.isfinite(fwd_snap):
            lk_atm = np.log(snap_atm["pure_strike"].to_numpy().astype(float) / fwd_snap)
            atm_idx = int(np.argmin(np.abs(lk_atm)))
            V_atm = v_atm_from_price(
                float(snap_atm["mid_norm"][atm_idx]),
                float(lk_atm[atm_idx]),
                T_snap,
                is_call=not puts_only,
                fwd=fwd_snap,
            )
            if np.isfinite(V_atm) and V_atm > 0:
                atm_iv = float(np.sqrt(V_atm / max(T_snap, 1e-8)))

        parts = []
        if not snap_c.is_empty() and not puts_only:
            parts.append(snap_c.with_columns(pl.lit(True).alias("is_call")))
        if not snap_p.is_empty() and not calls_only:
            parts.append(snap_p.with_columns(pl.lit(False).alias("is_call")))

        dlvsqs_out = densities_out = None
        failed = True
        K_lo_obs = K_hi_obs = float("nan")

        if parts:
            snap_obs = pl.concat(parts).sort("pure_strike")
            pure_K = snap_obs["pure_strike"].to_numpy().astype(float)
            K_lo_obs = float(np.min(pure_K))
            K_hi_obs = float(np.max(pure_K))
            bid_n = snap_obs["bid_norm"].to_numpy().astype(float)
            ask_n = snap_obs["ask_norm"].to_numpy().astype(float)
            is_calls = snap_obs["is_call"].to_numpy().astype(bool)

            try:
                density, dlvsqs, _ = fit_fn(
                    bid_norm=bid_n,
                    ask_norm=ask_n,
                    pure_K=pure_K,
                    is_calls=is_calls,
                    prev_density=prev_density,
                    fwd=fwd_snap,
                )
                dlvsqs_out = dlvsqs[np.newaxis, :]
                densities_out = density[np.newaxis, :]
                failed = False
                prev_density = density
            except Exception as exc:
                if verbose and not first_error_shown:
                    print(f"[fit] first error at {batch_ts}: {type(exc).__name__}: {exc}")
                    first_error_shown = True

        results.append(
            dict(
                ts=batch_ts,
                T=T_snap,
                fwd=fwd_snap,
                atm_iv=atm_iv,
                dlvsqs=dlvsqs_out,
                densities=densities_out,
                n_calls=len(snap_c),
                n_puts=len(snap_p),
                K_lo_obs=K_lo_obs,
                K_hi_obs=K_hi_obs,
                failed=failed,
            )
        )

    return results
