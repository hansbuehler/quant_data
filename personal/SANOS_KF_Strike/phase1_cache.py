"""
Run Phase 1 (data load + DLV batch fitting) and cache to disk.
Invoke once per (TICKER, TRADE_DATE, BATCH_FREQ, DLV config);
sweep_kf.py loads the cache for all KF parameter combinations.

Usage:
    python phase1_cache.py [--out phase1_cache.pkl]
"""

import argparse
import os
import pickle
import sys
from datetime import date
from functools import partial

import numpy as np
import polars as pl

project_root = "/Users/elie/Documents/GitHub/quant_data/"
kf_dir = os.path.join(project_root, "personal", "Kalman_filter_spline")
sanos_kf_dir = os.path.join(project_root, "personal", "SANOS_KF_Strike")
for _p in [project_root, kf_dir, sanos_kf_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QUANTDATA_RAW_DATA", "/quantdata/raw")

from dlv_fit_absolute import fit_density
from dlv_grid_absolute import make_density0, make_strike_grid
from dlv_snapshot_absolute import make_batch_snapshots, run_intraday_loop

from personal.Kalman_filter_spline.surface_kalman_filters import build_iv_observations
from quantdata.massive.client import MassiveClient

# ── Phase 1 config ────────────────────────────────────────────────────────
TICKER = "SPY"
TRADE_DATE = date(2024, 1, 3)
BATCH_FREQ = "1s"
MIN_TRADES = 100

LOWER_SCALE = 0.9
UPPER_SCALE = 1.1
N_PAD = 1
DLV_MIN = 0.001
DLV_MAX = 2.0
PUTS_ONLY = True
CALLS_ONLY = False

REGULARIZER_WEIGHT = 0.0
LOG_DLV_SPACE = False
TEMPORAL_REGULARIZER_WEIGHT = 0.0
TEMPORAL_WEIGHT_CLIP = float("inf")
BID_ASK_PENALTY_WEIGHT = 0.0


def main(out_path: str) -> None:
    df_tickers = pl.read_csv(
        os.path.join(project_root, "personal", f"spy_options_tickers_{TRADE_DATE}.csv")
    )
    client = MassiveClient()
    df_spy = client.fetch_trades(TICKER, TRADE_DATE)

    df_quotes_all, _ = build_iv_observations(df_tickers, TRADE_DATE, df_spy, min_trades=MIN_TRADES)
    df_quotes_all = df_quotes_all.drop_nulls()

    has_df = "discount_factor" in df_quotes_all.columns
    df_scale = pl.col("discount_factor") if has_df else pl.lit(1.0)
    df_q = df_quotes_all.with_columns(
        (pl.col("bid_price") / df_scale).alias("bid_norm"),
        (pl.col("ask_price") / df_scale).alias("ask_norm"),
        ((pl.col("bid_price") + pl.col("ask_price")) / 2.0 / df_scale).alias("mid_norm"),
        pl.col("strike").alias("pure_strike"),
    )
    df_calls = df_q.filter(pl.col("is_call"))
    df_puts = df_q.filter(~pl.col("is_call"))

    tte_y_max = float(df_q["T"].max())
    tte_y_med = float(df_q["T"].median())

    df_anchor = df_puts if PUTS_ONLY else df_calls
    ref_fwd = float(df_anchor["forward"].median())
    obs_ks = np.sort(df_anchor["strike"].unique().to_numpy().astype(float))

    grid = make_strike_grid(
        obs_ks, fwd=ref_fwd, lower_scale=LOWER_SCALE, upper_scale=UPPER_SCALE, n_pad=N_PAD
    )
    xstrikes_row = grid["xstrikes_row"]
    anchor_pure = grid["anchor_pure"]
    interior_pure = grid["interior_pure"]

    tte_geom = tte_y_max * 1.1
    grid_K, density0 = make_density0(xstrikes_row, fwd=ref_fwd)

    _fit_fn = partial(
        fit_density,
        xstrikes_row=xstrikes_row,
        grid_K=grid_K,
        density0=density0,
        tte_geom=tte_geom,
        dlv_min=DLV_MIN,
        dlv_max=DLV_MAX,
        regularizer_weight=REGULARIZER_WEIGHT,
        log_dlv_space=LOG_DLV_SPACE,
        temporal_regularizer_weight=TEMPORAL_REGULARIZER_WEIGHT,
        temporal_weight_clip=TEMPORAL_WEIGHT_CLIP,
        bid_ask_penalty_weight=BID_ASK_PENALTY_WEIGHT,
    )

    snaps_calls = make_batch_snapshots(df_calls, BATCH_FREQ)
    snaps_puts = make_batch_snapshots(df_puts, BATCH_FREQ)
    snap_ref = snaps_puts if PUTS_ONLY else snaps_calls
    batch_ts_list = sorted(snap_ref["batch_ts"].unique().to_list())
    print(f"{len(batch_ts_list)} snapshots at {BATCH_FREQ}")

    try:
        from tqdm.auto import tqdm
    except ImportError:

        def tqdm(x, **kw):
            return x

    results = run_intraday_loop(
        batch_ts_list=tqdm(batch_ts_list, desc="Fitting DLV snapshots"),
        snaps_calls=snaps_calls,
        snaps_puts=snaps_puts,
        fit_fn=_fit_fn,
        tte_y_med=tte_y_med,
        calls_only=CALLS_ONLY,
        puts_only=PUTS_ONLY,
        verbose=False,
        anchor_pure=anchor_pure,
    )

    n_failed = sum(r["failed"] for r in results)
    dlv_anchor_pure = anchor_pure[1:-1]
    N_dlv = len(dlv_anchor_pure)
    print(f"Done. Failed: {n_failed}/{len(results)}  N_dlv: {N_dlv}")

    snap_src = snaps_puts if PUTS_ONLY else snaps_calls
    snap_lookup = {}
    for ts in batch_ts_list:
        grp = snap_src.filter(pl.col("batch_ts") == ts).sort("pure_strike")
        if not grp.is_empty():
            snap_lookup[ts] = (
                grp["pure_strike"].to_numpy().astype(float),
                grp["bid_norm"].to_numpy().astype(float),
                grp["ask_norm"].to_numpy().astype(float),
            )

    cache = dict(
        results=results,
        grid_K=grid_K,
        xstrikes_row=xstrikes_row,
        tte_geom=tte_geom,
        anchor_pure=anchor_pure,
        interior_pure=interior_pure,
        dlv_anchor_pure=dlv_anchor_pure,
        N_dlv=N_dlv,
        ref_fwd=ref_fwd,
        dlv_max_sq_net=DLV_MAX**2 - DLV_MIN**2,
        trading_day_s=23400.0,
        snap_lookup=snap_lookup,
        is_call=not PUTS_ONLY,
        phase1_config=dict(
            ticker=TICKER,
            trade_date=str(TRADE_DATE),
            batch_freq=BATCH_FREQ,
            dlv_min=DLV_MIN,
            dlv_max=DLV_MAX,
            puts_only=PUTS_ONLY,
        ),
    )

    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=f"phase1_cache_{TRADE_DATE}.pkl")
    args = parser.parse_args()
    main(args.out)
