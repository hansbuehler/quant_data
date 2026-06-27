"""
sweep_params.py

Parallel hyperparameter sweep for the fixed SANOS-KF (sanos_kf.run_sanos_kf).

One invocation sweeps a grid over MANY trade dates and BOTH market sides, with a
SEPARATE filter run per side (like the single-strike IV-KF sweep):

  * multi-date  — pass --dates / --start..--end (or a single --date). Each date's
    quotes/trades are loaded ONCE in the parent, spilled to parquet, and fanned out
    to the worker pool; rows from every date stream into ONE CSV with a leading
    `date` column.
  * per-side    — puts and calls are evaluated INDEPENDENTLY. For each grid point we
    run the EKF twice: once fed puts-only (scored on puts), once fed calls-only
    (scored on calls). One CSV row per (date, side, grid point). Metrics for the two
    sides never mix (unlike a single joint fit scored on both).
  * sigmoid A/B — the DLV² parametrization is a sweep AXIS with its OWN grid per mode:
      "sigmoid" (use_sigmoid=True)  — state is the unconstrained logit; Q_scale_dlv /
        P0_scale live on the LOGIT scale and max_dlvsq (the sigmoid ceiling) is swept.
      "legacy"  (use_sigmoid=False) — state is the net DLV² (clipped ≥0); Q_scale_dlv /
        P0_scale are in DLV² units and max_dlvsq is ignored (not swept).
    The two parametrizations are NOT comparable on the same Q/P0 numbers, hence
    separate grids (GRID_SIGMOID vs GRID_LEGACY in SIGMOID_MODES).

Each worker builds a SanosKFConfig from its grid point, runs the EKF, and scores it
with the same unified-metrics pipeline used in sanos_kf_fixed_eval.ipynb. Results
stream to a CSV and are re-sorted at the end.

Usage
-----
    python personal/SANOS_KF/sweep_params.py --date 2024-01-03            # one date
    python personal/SANOS_KF/sweep_params.py --start 2024-01-03 --end 2024-01-17
    python personal/SANOS_KF/sweep_params.py --dates 2024-01-03 2024-01-04
    python personal/SANOS_KF/sweep_params.py --workers 8
    python personal/SANOS_KF/sweep_params.py --dry-run                    # just count
    python personal/SANOS_KF/sweep_params.py --sigmoid-mode legacy        # one param'n
    python personal/SANOS_KF/sweep_params.py --obs-mode smoothed --side puts
    python personal/SANOS_KF/sweep_params.py --resume                     # append rest

Edit GRID_* / BASE / MODES below to change what is swept. The grid is the Cartesian
product of every list in the chosen grids, merged on top of BASE.
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
# BASE          = params fixed across every run.
# SIGMOID_MODES = the DLV² parametrization axis. Each mode pins use_sigmoid AND
#                 carries its OWN shared grid (the two parametrizations measure Q/P0 on
#                 different scales, so they cannot share one grid — see module docstring).
# OBS_MODES     = the observation variants. There is ONE density (the exact tridiagonal
#                 SANOS solve); a mode only chooses whether the OBSERVATION is intrinsic
#                 (η=0) or smoothed (kernel variance η·V_ATM, paper §3.1). η is the given
#                 smoothness, never a fitted state.
# SIDES         = market sides, each fed to its OWN filter run and scored independently.
#
# A grid point is the Cartesian product:
#   sigmoid_mode × obs_mode × (shared grid ∪ obs-mode dims) × side.
# Anything in SanosKFConfig is fair game in any grid.
# ---------------------------------------------------------------------------
BASE: dict = dict(
    n_anchors=300,  # caps strike-aligned interior nodes; ↑ = finer, ~N³ solve
    k_min=-0.60,
    k_max=0.10,  # cover the data's log-moneyness (≈ -0.56..+0.07)
    correlated=True,
    batch_seconds=1.0,  # Gap 3: 1-second batched updates
)

# Sigmoid parametrization (use_sigmoid=True). State x is the UNCONSTRAINED sigmoid logit
# (DLV² = σ(x)·(max_dlvsq−min_dlvsq)+min_dlvsq), so:
#   - Sigma_init : a TARGET DLV² (logit-inverted internally for the flat init);
#   - Q_scale_dlv, P0_scale : process noise / initial covariance on the LOGIT scale
#     (NOT DLV² units) — a unit logit step moves DLV² by up to 0.25·span;
#   - max_dlvsq : the sigmoid ceiling (DLV² upper bound); min_dlvsq stays at its tiny floor;
#   - ell : RBF correlation length in DOLLARS (absolute strike distance).
GRID_SIGMOID: dict = dict(
    Sigma_init=[0.01, 0.1],
    Q_scale_dlv=[0.1, 1.0],
    P0_scale=[1e-2],  # pinned tight to keep the overnight grid small (was [1e-2, 1.0])
    max_dlvsq=[1.0, 10.0],  # sigmoid ceiling: DLV² ceilings for vol≈1, ~3
    ell=[1.0, 5.0],  # DOLLARS (absolute strike distance)
)  # 2×2×1×2×2 = 16

# Legacy parametrization (use_sigmoid=False). State x is the net DLV² (clipped ≥0), so
# Sigma_init / Q_scale_dlv / P0_scale are all in DLV² units and max_dlvsq is IGNORED
# (so it is not swept). These are the pre-sigmoid grid ranges.
GRID_LEGACY: dict = dict(
    Sigma_init=[0.01, 0.1],  # dropped 0.04 for the overnight grid
    Q_scale_dlv=[0.1, 1.0, 3.0],  # dropped 0.3
    P0_scale=[1e-2, 1.0],  # dropped 1e-1
    ell=[1.0, 3.0],  # DOLLARS (absolute strike distance); dropped 8.0
)  # 2×3×2×2 = 24

# sigmoid_mode -> (fixed overrides, shared grid for that parametrization).
SIGMOID_MODES: dict = {
    "sigmoid": (dict(use_sigmoid=True), GRID_SIGMOID),
    "legacy": (dict(use_sigmoid=False), GRID_LEGACY),
}

# obs_mode -> (fixed overrides, mode-specific swept dims).
# "intrinsic" : η=0 — exact tridiagonal density, discrete-call observation (linear interp).
# "smoothed"  : η>0 — same density, each node priced as a lognormal of variance η·V_ATM
#               (paper §3.1). Sweep η; paper values 0.06 (=0.25²), 0.25 (=0.5², default), 0.5.
OBS_MODES: dict = {
    "intrinsic": (
        dict(smoothing=False),
        dict(),
    ),
    "smoothed": (
        dict(smoothing=True),
        dict(eta=[0.06, 0.5]),
    ),
}

# Market sides scored INDEPENDENTLY. Each side gets its own filter run (fed that side
# only) and its own metric row; puts and calls metrics never mix.
SIDES = ["puts", "calls"]
_SIDE_KW = {"calls": dict(calls_only=True), "puts": dict(puts_only=True)}

TICKER_DEFAULT = "SPY"


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
    """Build a config from `params`, run the EKF fed ONE side, score THAT side, and
    return {params..., metrics...}.

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
    # `params` carries report-only tags; strip them before building cfg.
    params = dict(params)
    sigmoid_mode = params.pop("sigmoid_mode", "")
    obs_mode = params.pop("obs_mode", "")
    side = params.pop("side", "puts")
    side_is_call = side == "calls"
    cfg_kw = {**BASE, **params, **_SIDE_KW[side]}
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
        # Score ONLY the side this run was fed (per-side independent evaluation).
        qb = qb[qb["is_call"] == side_is_call]
        if qb.empty:
            raise ValueError(f"no {side} quotes priced")

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
            d = exact_solve_density(
                xrow,
                grid_K,
                tte_ref,
                cfg.min_dlvsq,
                cfg.max_dlvsq,
                fwd=F0,
                use_sigmoid=cfg.use_sigmoid,
            )
            negs.append(np.nan if d is None else float(np.minimum(d, 0.0).sum()))
        neg_vals = np.array(negs)

        m = um.evaluate(df_eval, model_name="sweep", neg_mass_values=neg_vals)
        m.pop("model", None)
        # free this run's big transients before the worker takes its next task
        del qb, df_eval, st
        gc.collect()
        return {
            "sigmoid_mode": sigmoid_mode,
            "obs_mode": obs_mode,
            "side": side,
            **params,
            **m,
            "ok": True,
            "err": "",
            "secs": time.perf_counter() - t0,
        }
    except Exception as exc:  # keep the sweep going; record the failure
        return {
            "sigmoid_mode": sigmoid_mode,
            "obs_mode": obs_mode,
            "side": side,
            **params,
            "ok": False,
            "err": f"{type(exc).__name__}: {exc}",
            "secs": time.perf_counter() - t0,
        }


def _build_points() -> list[dict]:
    """Cartesian product of sigmoid_mode × obs_mode × (shared grid ∪ obs dims) × side."""
    points: list[dict] = []
    for sig_name, (sig_fixed, sig_grid) in SIGMOID_MODES.items():
        for obs_name, (obs_fixed, obs_grid) in OBS_MODES.items():
            dims = {**sig_grid, **obs_grid}
            keys = list(dims.keys())
            for combo in itertools.product(*dims.values()):
                pt = {
                    "sigmoid_mode": sig_name,
                    "obs_mode": obs_name,
                    **sig_fixed,
                    **obs_fixed,
                    **dict(zip(keys, combo)),
                }
                for side in SIDES:
                    points.append({**pt, "side": side})
    return points


# Param keys that uniquely identify a grid point (for --resume de-duplication).
# `date` is prepended by the driver so a point is unique across dates too.
SIG_KEYS = (
    "date",
    "sigmoid_mode",
    "obs_mode",
    "side",
    "use_sigmoid",
    "smoothing",
    "Sigma_init",
    "Q_scale_dlv",
    "P0_scale",
    "max_dlvsq",
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


def _resolve_dates(args: argparse.Namespace) -> list[date]:
    """Trading dates to sweep: a --start..--end range, an explicit --dates list, or
    the single --date fallback (weekends/holidays dropped)."""
    from quantdata.biz_calendar import get_business_days, to_date

    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("--start and --end must be given together")
        return list(get_business_days(to_date(args.start), to_date(args.end)))
    if args.dates:
        out: list[date] = []
        for s in args.dates:
            d = to_date(s)
            if not get_business_days(d, d):
                print(f"skip {d}: not a trading day")
                continue
            out.append(d)
        return out
    d = to_date(args.date)
    if not get_business_days(d, d):
        raise SystemExit(f"{d} is not a trading day")
    return [d]


# Per-date data cache. build_iv_observations walks the WHOLE contract universe doing
# one small per-contract read, so a cold load is thousands of tiny I/O ops (~minutes,
# ~0% CPU). We pickle its output to parquet keyed by (ticker, date, min_trades); a warm
# load then collapses to a single parquet read — and every resume/re-run is instant.
DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, "personal", "SANOS_KF", "data_cache")


def _tickers_path(trade_date: date) -> str:
    """Per-date option universe. A date-specific file is REQUIRED for correctness:
    the generic spy_options_tickers.csv is the 2024-01-03 universe, whose near-term
    contracts expire within days, so later trade dates would see no valid quotes
    ("No options produced valid data"). Prefer ticker_universes/, then personal/, and
    only fall back to the generic file (valid for ~2024-01-03..05) as a last resort."""
    candidates = [
        os.path.join(
            PROJECT_ROOT, "personal", "ticker_universes", f"spy_options_tickers_{trade_date}.csv"
        ),
        os.path.join(PROJECT_ROOT, "personal", f"spy_options_tickers_{trade_date}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    generic = os.path.join(PROJECT_ROOT, "personal", "spy_options_tickers.csv")
    print(f"  WARNING: no per-date universe for {trade_date}; using generic {generic}")
    return generic


def _data_cache_paths(ticker: str, trade_date: date, min_trades: int) -> tuple[str, str]:
    base = f"{ticker}_{trade_date}_mt{min_trades}"
    return (
        os.path.join(DATA_CACHE_DIR, f"quotes_{base}.parquet"),
        os.path.join(DATA_CACHE_DIR, f"trades_{base}.parquet"),
    )


def _ensure_data_cache(
    ticker: str, trade_date: date, min_trades: int, rebuild: bool = False
) -> tuple[str, str]:
    """Build the date's (df_quotes, df_trades) via build_iv_observations and pickle to
    parquet, unless a fresh cache already exists. Returns the two parquet PATHS — the
    workers read them directly (one copy each), so the parent never holds the frames."""
    q_path, t_path = _data_cache_paths(ticker, trade_date, min_trades)
    if not rebuild and os.path.exists(q_path) and os.path.exists(t_path):
        print(f"data cache HIT → {os.path.basename(q_path)} (skipping the slow rebuild)")
        return q_path, t_path

    from personal.Kalman_filter_spline.surface_kalman_filters import build_iv_observations
    from quantdata.massive.client import MassiveClient

    tickers_path = _tickers_path(trade_date)
    print(
        f"data cache MISS → building {ticker} {trade_date} (slow, ~minutes) "
        f"[universe: {os.path.basename(tickers_path)}] ..."
    )
    df_tickers = pl.read_csv(tickers_path)
    client = MassiveClient()
    df_spy = client.fetch_trades(ticker, trade_date)
    df_quotes, df_trades = build_iv_observations(
        df_tickers, trade_date, df_spy, min_trades=min_trades
    )
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # write to a temp name then rename, so an interrupted build never leaves a half cache.
    df_quotes.write_parquet(q_path + ".tmp")
    df_trades.write_parquet(t_path + ".tmp")
    os.replace(q_path + ".tmp", q_path)
    os.replace(t_path + ".tmp", t_path)
    print(
        f"data cache WRITTEN → {os.path.basename(q_path)} "
        f"({len(df_quotes):,} quotes, {len(df_trades):,} trades)"
    )
    del df_quotes, df_trades, df_spy
    gc.collect()
    return q_path, t_path


def _run_date(
    ticker: str,
    trade_date: date,
    min_trades: int,
    points: list[dict],
    workers: int,
    mtpc: int | None,
    writer: csv.DictWriter,
    fh,
    rebuild_cache: bool = False,
) -> int:
    """Load one date (via the per-date parquet cache), fan its grid out to a pool,
    stream rows (tagged with `date`) into the shared CSV writer. Returns rows written."""
    print(f"\n===== {trade_date} =====")
    # Workers read the cached parquet directly (one copy each), so the parent never
    # holds the frames — the cache IS the spill, and it persists across runs.
    q_path, t_path = _ensure_data_cache(ticker, trade_date, min_trades, rebuild_cache)

    print(f"Running {len(points)} configs on {workers} workers ...")
    n_written = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(q_path, t_path),
        max_tasks_per_child=mtpc,
    ) as ex:
        futs = {ex.submit(_run_one, p): p for p in points}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            r["date"] = str(trade_date)
            writer.writerow(r)
            fh.flush()  # land this row on disk before moving on
            n_written += 1
            tag = "ok " if r.get("ok") else "ERR"
            score = r.get("ho_nse_mean", float("nan"))
            p = futs[fut]
            lbl = f"{p['sigmoid_mode']}/{p['obs_mode']}/{p['side']}"
            print(
                f"[{trade_date}][{i}/{len(points)}] {tag} {lbl} ho_nse={score:.4f} "
                f"({r['secs']:.1f}s)" + (f"  {r['err']}" if not r.get("ok") else ""),
                flush=True,
            )
    gc.collect()
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ticker", default=TICKER_DEFAULT)
    ap.add_argument("--date", default="2024-01-03", help="single trade date YYYY-MM-DD")
    ap.add_argument("--start", help="range start YYYY-MM-DD (inclusive); use with --end")
    ap.add_argument("--end", help="range end YYYY-MM-DD (inclusive); use with --start")
    ap.add_argument("--dates", nargs="*", help="explicit trade dates YYYY-MM-DD")
    ap.add_argument(
        "--min-trades",
        type=int,
        default=0,
        help="0 = pre-committed universe; >0 selects on same-day trades (look-ahead)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="parallel workers. Each holds a full data copy (~GB) — keep modest "
        "to avoid OOM (default capped at 4).",
    )
    ap.add_argument("--limit", type=int, default=None, help="only run the first N grid points")
    ap.add_argument(
        "--sigmoid-mode",
        choices=list(SIGMOID_MODES),
        default=None,
        help="restrict to one DLV² parametrization",
    )
    ap.add_argument(
        "--obs-mode", choices=list(OBS_MODES), default=None, help="restrict to one observation mode"
    )
    ap.add_argument("--side", choices=SIDES, default=None, help="restrict to one market side")
    ap.add_argument(
        "--out", default=None, help="output CSV path (default: sweep_<ticker>_<span>.csv)"
    )
    ap.add_argument(
        "--sort", default="ho_nse_mean", help="metric to sort the final table by (ascending)"
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip (date, grid point) rows already present in --out and append the rest",
    )
    ap.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=50,
        help="recycle each worker after this many configs to cap memory creep (0 = never recycle).",
    )
    ap.add_argument(
        "--rebuild-data-cache",
        action="store_true",
        help="rebuild the per-date data cache (data_cache/*.parquet) even if present",
    )
    ap.add_argument("--dry-run", action="store_true", help="print grid size and exit")
    args = ap.parse_args()

    dates = _resolve_dates(args)

    points = _build_points()
    if args.sigmoid_mode is not None:
        points = [p for p in points if p["sigmoid_mode"] == args.sigmoid_mode]
    if args.obs_mode is not None:
        points = [p for p in points if p["obs_mode"] == args.obs_mode]
    if args.side is not None:
        points = [p for p in points if p["side"] == args.side]
    if args.limit is not None:
        points = points[: args.limit]

    span = str(dates[0]) if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}"
    out = args.out or os.path.join(
        PROJECT_ROOT, "personal", "SANOS_KF", f"sweep_{args.ticker}_{span}.csv"
    )

    by_sig = {m: sum(p["sigmoid_mode"] == m for p in points) for m in SIGMOID_MODES}
    by_side = {s: sum(p["side"] == s for p in points) for s in SIDES}
    print(
        f"Grid: {len(points)} points/date × {len(dates)} date(s) = {len(points) * len(dates)} runs"
    )
    print("  by sigmoid_mode: " + "  ".join(f"{m}={n}" for m, n in by_sig.items()))
    print("  by side: " + "  ".join(f"{s}={n}" for s, n in by_side.items()))
    print(f"  dates: {', '.join(str(d) for d in dates)}")
    if args.dry_run:
        return
    if not points:
        print("Nothing to run.")
        return

    # Stable CSV schema so we can stream-append heterogeneous rows (modes differ in
    # which params they carry; failed rows lack metrics). date + tags first, then
    # params, then run status, then the unified-metrics fields.
    all_points = _build_points()
    tag_fields = ["date", "sigmoid_mode", "obs_mode", "side"]
    param_fields = sorted({k for p in all_points for k in p} - set(tag_fields))
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
        "ho_out_pct_mean",
        "ho_out_pct_med",
        "is_out_pct_mean",
        "is_out_pct_med",
        "neg_mass_mean",
        "neg_mass_med",
        "arb_frac",
        "n_snapshots",
    ]
    fieldnames = tag_fields + param_fields + status_fields + metric_fields
    mtpc = args.max_tasks_per_child or None

    # --resume: drop (date, point) pairs already recorded in the existing CSV.
    resuming = args.resume and os.path.exists(out)
    done_sigs: set = set()
    if resuming:
        prev = pd.read_csv(out)
        done_sigs = {_sig(row) for row in prev.to_dict("records")}
        print(f"Resume: {len(done_sigs)} rows already done in {out}")

    t0 = time.perf_counter()
    total_written = 0
    # Append when resuming (keep prior rows); otherwise truncate + write header.
    mode_flag = "a" if resuming else "w"
    with open(out, mode_flag, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not resuming:
            writer.writeheader()
            fh.flush()
        for d in dates:
            day_points = points
            if resuming:
                day_points = [p for p in points if _sig({**p, "date": str(d)}) not in done_sigs]
                skipped = len(points) - len(day_points)
                if skipped:
                    print(f"[{d}] resume: {skipped} already done, {len(day_points)} to run")
            if not day_points:
                print(f"[{d}] nothing to run, skipping")
                continue
            try:
                total_written += _run_date(
                    args.ticker,
                    d,
                    args.min_trades,
                    day_points,
                    args.workers,
                    mtpc,
                    writer,
                    fh,
                    rebuild_cache=args.rebuild_data_cache,
                )
            except Exception as exc:
                print(f"[{d}] FAILED: {type(exc).__name__}: {exc}")

    # Re-read the streamed CSV and rewrite it sorted (the run is done; the streamed
    # file already holds every row should this step be interrupted).
    df = pd.read_csv(out)
    if args.sort in df.columns:
        df = df.sort_values(["side", args.sort], na_position="last")
        df.to_csv(out, index=False)
    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s. Wrote {total_written} rows this run; {len(df)} total in {out}"
    )

    swept = ["date", "sigmoid_mode", "obs_mode", "side", *GRID_LEGACY.keys(), "max_dlvsq", "eta"]
    keep = [
        c
        for c in (
            *dict.fromkeys(swept),  # de-dup while preserving order
            "ho_nse_mean",
            "ho_nse_med",
            "ho_out_pct_mean",
            "is_out_pct_mean",
            "ho_rmse_mean",
            "is_nse_mean",
            "secs",
        )
        if c in df.columns
    ]
    for side in SIDES:
        sub = df[df["side"] == side] if "side" in df.columns else df
        if sub.empty:
            continue
        print(f"\n[{side}] Top 10 by {args.sort}")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(sub[keep].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
