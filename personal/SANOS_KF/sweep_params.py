"""
sweep_params.py

Parallel hyperparameter sweep for the fixed SANOS-KF (sanos_kf.run_sanos_kf).

Loads one trade-date of quotes/trades ONCE in the parent process, then fans a
large parameter grid out to a process pool. Each worker builds a SanosKFConfig
from its grid point, runs the EKF, and scores it with the same unified-metrics
pipeline used in sanos_kf_fixed_eval.ipynb. Results stream to a CSV.

Usage
-----
    python personal/SANOS_KF/sweep_params.py                       # full grid
    python personal/SANOS_KF/sweep_params.py --workers 8
    python personal/SANOS_KF/sweep_params.py --dry-run             # just count
    python personal/SANOS_KF/sweep_params.py --limit 50            # first 50 pts
    python personal/SANOS_KF/sweep_params.py --ticker SPY --date 2024-01-03
    python personal/SANOS_KF/sweep_params.py --sort ho_nse_mean    # final ranking

Edit GRID / BASE below to change what is swept. The grid is the Cartesian
product of every list in GRID, merged on top of BASE.
"""

from __future__ import annotations

# Keep each worker single-threaded so N processes don't oversubscribe the CPU
# (JAX/XLA + BLAS each spawn thread pools). MUST be set before jax is imported.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ.setdefault("QUANTDATA_RAW_DATA", "/quantdata/raw")

import argparse
import csv
import gc
import itertools
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = "/Users/elie/Documents/GitHub/quant_data/"
for _p in (
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "personal", "Kalman_filter_spline"),
    os.path.join(PROJECT_ROOT, "personal", "shared_functions"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# The sweep grid.
#
# BASE  = params fixed across every run.
# GRID  = params swept in EVERY mode (full Cartesian product).
# MODES = the SANOS-KF observation variants. There is now ONE density (the exact
#         tridiagonal SANOS solve); a mode only chooses whether the OBSERVATION is
#         intrinsic (η=0) or smoothed (kernel variance η·V_ATM, paper §3.1). η is the
#         given smoothness, never a fitted state.
#
# Enable/disable a mode by editing MODES. Anything in SanosKFConfig is fair game.
# ---------------------------------------------------------------------------
BASE: dict = dict(
    n_anchors=300,  # caps strike-aligned interior nodes; ↑ = finer, ~N³ solve
    k_min=-0.60,
    k_max=0.10,  # cover the data's log-moneyness (≈ -0.56..+0.07)
    correlated=True,
    batch_seconds=1.0,  # Gap 3: 1-second batched updates
)

# Shared dims swept across ALL modes (absolute-strike space): ell in DOLLARS, Sigma/Q/P0
# on the net-DLV² scale.
GRID: dict = dict(
    Sigma_init=[0.01, 0.04, 0.1],
    Q_scale_dlv=[0.1, 0.3, 1.0, 3.0],
    P0_scale=[1e-2, 1e-1, 1.0],
    ell=[1.0, 3.0, 8.0],  # DOLLARS (absolute strike distance)
)

# mode name -> (fixed overrides for the mode, mode-specific swept dims).
# "intrinsic" : η=0 — exact tridiagonal density, discrete-call observation (linear interp).
# "smoothed"  : η>0 — same density, each node priced as a lognormal of variance η·V_ATM
#               (paper §3.1). Sweep η; paper values 0.06 (=0.25²), 0.25 (=0.5², default), 0.5.
MODES: dict = {
    "intrinsic": (
        dict(smoothing=False),
        dict(),
    ),
    "smoothed": (
        dict(smoothing=True),
        dict(eta=[0.06, 0.25, 0.5]),
    ),
}

# Market side fed to every run: "both" | "calls" | "puts".
SIDE = "puts"
_SIDE_KW = {"calls": dict(calls_only=True), "puts": dict(puts_only=True), "both": {}}


# ---------------------------------------------------------------------------
# Worker globals (set once per worker process by _init_worker)
# ---------------------------------------------------------------------------
_DFQ: pl.DataFrame | None = None
_DFT: pl.DataFrame | None = None


def _init_worker(quotes_path: str, trades_path: str) -> None:
    """Pool initializer: load the data from parquet ONCE per worker into module
    globals. We pass file PATHS (not the frames) so the parent can free its copy
    and we don't pickle ~GB DataFrames into every worker (that 8× copy was the
    OOM). Each worker still holds one copy, so keep --workers modest."""
    global _DFQ, _DFT
    _DFQ = pl.read_parquet(quotes_path)
    _DFT = pl.read_parquet(trades_path)


def _run_one(params: dict) -> dict:
    """Build a config from `params`, run the EKF, return {params..., metrics...}.

    Imported lazily so jax is initialized inside the worker (after the
    single-thread env vars above are in effect)."""
    import personal.shared_functions.unified_metrics as um
    from personal.SANOS_KF.sanos_kf import (
        SanosKFConfig,
        build_exact_grid,
        exact_solve_density,
        run_sanos_kf,
    )

    df_quotes, df_trades = _DFQ, _DFT
    # `params` carries a "mode" tag for reporting; strip it before building cfg.
    params = dict(params)
    mode = params.pop("mode", "")
    cfg_kw = {**BASE, **params, **_SIDE_KW[SIDE]}
    t0 = time.perf_counter()
    try:
        cfg = SanosKFConfig(**cfg_kw)
        # The density is always the exact tridiagonal solve, so negative-mass scoring
        # applies to every run (intrinsic and smoothed): request the state history.
        q, _, st = run_sanos_kf(df_quotes, df_trades, cfg, return_state=True)
        qb = q.to_pandas()
        del q

        bs_ns = int((cfg.batch_seconds or 1.0) * 1e9)
        qb["window"] = qb["ts_ns"].astype("int64") // bs_ns
        qb = (
            qb.sort_values("ts_ns")
            .drop_duplicates(["window", "strike", "is_call"], keep="last")
            .dropna(subset=["sanos_kf_price_prior"])
        )

        F = qb["forward"].to_numpy(float)
        K = qb["strike"].to_numpy(float)
        ic = qb["is_call"].to_numpy(bool)
        dfac = (
            qb["discount_factor"].to_numpy(float) if "discount_factor" in qb else np.ones(len(qb))
        )

        # normalized call (price/(F·df)) -> native option price via put-call parity
        def to_native(c):
            return np.where(ic, c, c - (F - K))

        prior_native = to_native(qb["sanos_kf_price_prior"].to_numpy(float) * F)
        post_native = to_native(qb["sanos_kf_price"].to_numpy(float) * F)

        df_eval = um.from_option_prices(
            ts=qb["window"].to_numpy(),
            strike=K,
            forward=F,
            is_call=ic,
            bid=qb["bid_price"].to_numpy(float),
            ask=qb["ask_price"].to_numpy(float),
            prior_price=prior_native,
            post_price=post_native,
            discount_factor=dfac,
        )

        # score negative mass on the model's own (exact tridiagonal) density
        grid_K, tte_ref = build_exact_grid(df_quotes, cfg)
        F0 = float(np.nanmedian(df_quotes["forward"].to_numpy().astype(float)))
        stp = st.to_pandas()
        stp["window"] = stp["ts_ns"].astype("int64") // bs_ns
        stp = stp.sort_values("ts_ns").drop_duplicates("window", keep="last")
        xcols = [c for c in stp.columns if c.startswith("x_")]
        negs = []
        for xrow in stp[xcols].to_numpy(float):
            d = exact_solve_density(xrow, grid_K, tte_ref, cfg.dlv_min, fwd=F0)
            negs.append(np.nan if d is None else float(np.minimum(d, 0.0).sum()))
        neg_vals = np.array(negs)

        m = um.evaluate(df_eval, model_name="sweep", neg_mass_values=neg_vals)
        m.pop("model", None)
        # free this run's big transients before the worker takes its next task
        del qb, df_eval, st
        gc.collect()
        return {
            "mode": mode,
            **params,
            **m,
            "ok": True,
            "err": "",
            "secs": time.perf_counter() - t0,
        }
    except Exception as exc:  # keep the sweep going; record the failure
        return {
            "mode": mode,
            **params,
            "ok": False,
            "err": f"{type(exc).__name__}: {exc}",
            "secs": time.perf_counter() - t0,
        }


def _build_points() -> list[dict]:
    """Cartesian product of GRID × (per-mode fixed + mode-specific dims)."""
    points: list[dict] = []
    for mode, (fixed, mode_grid) in MODES.items():
        dims = {**GRID, **mode_grid}
        keys = list(dims.keys())
        for combo in itertools.product(*dims.values()):
            points.append({"mode": mode, **fixed, **dict(zip(keys, combo))})
    return points


# Param keys that uniquely identify a grid point (for --resume de-duplication).
SIG_KEYS = (
    "mode",
    "smoothing",
    "Sigma_init",
    "Q_scale_dlv",
    "P0_scale",
    "ell",
    "eta",
)


def _sig(d) -> tuple:
    """Hashable identity for a grid point / CSV row. Missing or NaN values map to
    None so a point and its persisted row compare equal across modes; floats are
    rounded to absorb CSV round-trip noise."""
    out = []
    for k in SIG_KEYS:
        v = d.get(k) if isinstance(d, dict) else d[k]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        elif isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
        elif isinstance(v, (int, float)):
            out.append(round(float(v), 12))
        else:
            out.append(str(v))
    return tuple(out)


def _load_data(ticker: str, trade_date: date, min_trades: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    from personal.Kalman_filter_spline.surface_kalman_filters import build_iv_observations
    from quantdata.massive.client import MassiveClient

    df_tickers = pl.read_csv(os.path.join(PROJECT_ROOT, "personal", "spy_options_tickers.csv"))
    client = MassiveClient()
    df_spy = client.fetch_trades(ticker, trade_date)
    df_quotes, df_trades = build_iv_observations(
        df_tickers, trade_date, df_spy, min_trades=min_trades
    )
    return df_quotes, df_trades


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--date", default="2024-01-03", help="trade date YYYY-MM-DD")
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="parallel workers. Each holds a full data copy (~GB) — keep modest "
        "to avoid OOM (default capped at 4).",
    )
    ap.add_argument("--limit", type=int, default=None, help="only run the first N grid points")
    ap.add_argument(
        "--mode", choices=list(MODES), default=None, help="restrict to one density mode"
    )
    ap.add_argument(
        "--out", default=None, help="output CSV path (default: sweep_<ticker>_<date>.csv)"
    )
    ap.add_argument(
        "--sort", default="ho_nse_mean", help="metric to sort the final table by (ascending)"
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip grid points already present in --out and append the rest",
    )
    ap.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=50,
        help="recycle each worker after this many configs to cap memory creep (0 = never recycle).",
    )
    ap.add_argument("--dry-run", action="store_true", help="print grid size and exit")
    args = ap.parse_args()

    points = _build_points()
    if args.mode is not None:
        points = [p for p in points if p["mode"] == args.mode]
    if args.limit is not None:
        points = points[: args.limit]

    trade_date = date.fromisoformat(args.date)
    out = args.out or f"sweep_{args.ticker}_{args.date}.csv"

    # --resume: drop points already recorded in the existing CSV.
    resuming = args.resume and os.path.exists(out)
    done_sigs: set = set()
    if resuming:
        prev = pd.read_csv(out)
        done_sigs = {_sig(row) for row in prev.to_dict("records")}
        before = len(points)
        points = [p for p in points if _sig(p) not in done_sigs]
        print(f"Resume: {len(done_sigs)} already done, {before - len(points)} skipped")

    by_mode = {m: sum(p["mode"] == m for p in points) for m in MODES}
    print(
        f"Grid: {len(points)} points to run  " + "  ".join(f"{m}={n}" for m, n in by_mode.items())
    )
    print(f"  shared dims: {list(GRID.keys())}")
    if args.dry_run:
        return
    if not points:
        print("Nothing to run.")
        return

    print(f"Loading {args.ticker} {args.date} ...")
    df_quotes, df_trades = _load_data(args.ticker, trade_date, args.min_trades)
    print(f"quotes: {len(df_quotes):,}   trades: {len(df_trades):,}")

    # Spill the data to parquet so workers load it from disk (one copy each) and
    # the parent can free its own copy — avoids pickling ~GB into every worker.
    tmpdir = tempfile.mkdtemp(prefix="sweep_data_")
    q_path = os.path.join(tmpdir, "quotes.parquet")
    t_path = os.path.join(tmpdir, "trades.parquet")
    df_quotes.write_parquet(q_path)
    df_trades.write_parquet(t_path)
    del df_quotes, df_trades
    gc.collect()
    print(f"Running {len(points)} configs on {args.workers} workers -> {out}")

    # Stable CSV schema so we can stream-append heterogeneous rows (modes differ
    # in which params they carry; failed rows lack metrics). Params first (mode
    # leading), then run status, then the unified-metrics fields.
    all_points = _build_points()
    param_fields = ["mode"] + sorted({k for p in all_points for k in p} - {"mode"})
    status_fields = ["ok", "err", "secs"]
    metric_fields = [
        "ho_nse_mean",
        "ho_nse_med",
        "ho_nse_std",
        "is_nse_mean",
        "is_nse_med",
        "is_nse_std",
        "ho_rmse_mean",
        "ho_rmse_med",
        "ho_rmse_std",
        "is_rmse_mean",
        "is_rmse_med",
        "is_rmse_std",
        "neg_mass_mean",
        "neg_mass_med",
        "arb_frac",
        "n_snapshots",
    ]
    fieldnames = param_fields + status_fields + metric_fields
    mtpc = args.max_tasks_per_child or None

    t0 = time.perf_counter()
    # Append when resuming (keep prior rows); otherwise truncate + write header.
    # Flush after every result so the file is updated throughout the run.
    mode_flag = "a" if resuming else "w"
    with open(out, mode_flag, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not resuming:
            writer.writeheader()
            fh.flush()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(q_path, t_path),
            max_tasks_per_child=mtpc,
        ) as ex:
            futs = {ex.submit(_run_one, p): p for p in points}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                writer.writerow(r)
                fh.flush()  # land this row on disk before moving on
                tag = "ok " if r.get("ok") else "ERR"
                score = r.get("ho_nse_mean", float("nan"))
                print(
                    f"[{i}/{len(points)}] {tag} ho_nse={score:.4f} "
                    f"({r['secs']:.1f}s) {futs[fut]}"
                    + (f"  {r['err']}" if not r.get("ok") else ""),
                    flush=True,
                )

    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)  # drop the spilled parquet copies

    # Re-read the streamed CSV and rewrite it sorted (the run is done; the
    # streamed file already holds every row should this step be interrupted).
    df = pd.read_csv(out)
    if args.sort in df.columns:
        df = df.sort_values(args.sort, na_position="last")
        df.to_csv(out, index=False)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Wrote {len(df)} rows to {out}")

    swept = ["mode", *GRID.keys(), "eta"]
    keep = [
        c
        for c in (*swept, "ho_nse_mean", "ho_nse_med", "ho_rmse_mean", "is_nse_mean", "secs")
        if c in df.columns
    ]
    print("\nTop 10 by", args.sort)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[keep].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
