"""
SSVI surface-Kalman sweep — 1 s batch, unified metrics, puts/calls scored apart.

Drives the spline-state surface Kalman (surface_kalman_filters.run_spline_surface_kalman)
with the 3-parameter SSVI slice model (surface_model.SSVI_SLICE_MODEL, θ=[ϑ, ρ, ζ])
over a sweep of the KF hyper-parameters:

    Q_scale       — process-noise scale
    P0_scale      — initial state-covariance scale
    Q_structure   — "scaled_identity" (Q = Q_scale·I) vs
                    "diagonal"        (Q = diag(Q_scale · model.q_diag_weights))
    fit_mode      — "joint"    (legacy: θ fit on the union of puts+calls) vs
                    "per_side" (θ fit on puts-only, then calls-only, separately)

REUSES the single-strike Phase-1 cache (Kalman_filter_individual/phase1_cache_{date}.pkl):
both pipelines share the identical Phase 1+2 (fetch + IV batch), so there is no need
to re-fetch.  Each worker loads that cache once and adapts the frames for the surface
KF (rename _strike→strike, add log_moneyness=log(K/F)); every (Q_scale, P0_scale,
Q_structure) point then runs only the cheap Phase-3 filter.

Each combo is scored with the project's unified metric set (held-out / in-sample NSE
& RMSE, out-of-band %, Breeden-Litzenberger negative mass + arb fraction).  The
surface produces multi-strike snapshots, so negative mass comes from evaluate()'s
built-in per-snapshot BL (no reconstruction needed).  One result row per
(Q_scale, P0_scale, Q_structure, side); puts and calls are ranked independently.

Usage
-----
    # build the (shared) Phase-1 cache once per date, if not already present:
    python ../Kalman_filter_individual/phase1_cache.py --date 2024-01-03
    python ssvi_sweep.py [--date 2024-01-03] [--out results/ssvi_sweep_2024-01-03.csv] [--workers 4]

Edit Q_SCALES / P0_SCALES / Q_STRUCTURES below to define the search space.
"""

# Keep each worker single-threaded so N processes don't oversubscribe the CPU
# (BLAS spawns thread pools). Set before numpy/scipy import.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import itertools
import multiprocessing as mp
import pickle
import sys

import numpy as np
from scipy.special import ndtr

project_root = "/Users/elie/Documents/GitHub/quant_data/"
_HERE = os.path.dirname(os.path.abspath(__file__))
# surface_kalman_filters / surface_model use bare imports → their dir must be on path.
_SPLINE = os.path.join(project_root, "personal", "Kalman_filter_spline")
_INDIV = os.path.join(project_root, "personal", "Kalman_filter_individual")
for _p in (project_root, _SPLINE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QUANTDATA_RAW_DATA", "!/quantdata/raw")
os.environ.setdefault("QUANTDATA_TRAINING_DATA", "!/quantdata/training")
os.environ.setdefault("QUANTDATA_PROCESSING_DATA", "!/quantdata/processing")
os.environ.setdefault("QUANTDATA_CACHE_MODE", "on")
os.environ.setdefault("QUANTDATA_TEST_MODE", "0")

# ── Search space — edit this ──────────────────────────────────────────────────
Q_SCALES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
P0_SCALES = [1e-2, 1e-1, 1.0, 10.0]
Q_STRUCTURES = ["scaled_identity", "diagonal"]
BATCH_S = 1.0  # batch window width (seconds) — fixed, not swept

# Market sides scored INDEPENDENTLY. How each side relates to the SSVI fit depends
# on FIT_MODE (below).
SIDES = ["puts", "calls"]

# Fit modes — swept like the other axes (edit to restrict). Each produces its own
# block of result rows (ranked independently), so the legacy behaviour is preserved
# bit-for-bit whenever "joint" is present.
#   "joint"    — legacy: θ is fit on the UNION of quotes (puts + calls), then the
#                eval table is split so each side's NSE/RMSE/neg-mass never mix.
#   "per_side" — θ is fit on ONE side only (puts-only, then calls-only, as two
#                independent filter runs); each side is scored against its own fit.
FIT_MODES = ["per_side"]

TRADE_DATE = "2024-01-03"

# Metric columns emitted by unified_metrics.evaluate (minus "model"), full set.
_METRIC_KEYS = [
    "ho_nse_mean",
    "ho_nse_med",
    "ho_nse_std",
    "is_nse_mean",
    "is_nse_med",
    "is_nse_std",
    "ho_ese_mean",
    "ho_ese_med",
    "ho_ese_std",
    "is_ese_mean",
    "is_ese_med",
    "is_ese_std",
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
_PARAM_KEYS = ["Q_scale", "P0_scale", "Q_structure", "fit_mode"]

# ── Worker globals (set once per process by _init_worker) ─────────────────────
_sm = None  # surface_model
_skf = None  # surface_kalman_filters
_um = None  # unified_metrics
_DFQ = None  # adapted quote IV frame (surface-ready)
_DFT = None  # adapted trade IV frame (surface-ready)


def _cache_path(trade_date: str) -> str:
    """The SHARED single-strike Phase-1 cache (reused, not rebuilt here)."""
    return os.path.join(_INDIV, f"phase1_cache_{trade_date}.pkl")


def _adapt(df):
    """Make a build_iv_frames frame surface-KF-ready: rename _strike→strike and
    add log_moneyness = log(K/F). (build_iv_observations does this internally; we
    do it here so we can reuse the individual cache instead of re-fetching.)"""
    import polars as pl

    if "_strike" in df.columns:
        df = df.rename({"_strike": "strike"})
    return df.with_columns(
        pl.when(pl.col("forward") > 0)
        .then((pl.col("strike") / pl.col("forward")).log())
        .otherwise(None)
        .alias("log_moneyness")
    )


def _init_worker(trade_date: str) -> None:
    global _sm, _skf, _um, _DFQ, _DFT
    import surface_kalman_filters as surface_kalman_filters
    import surface_model as surface_model

    import personal.shared_functions.unified_metrics as unified_metrics

    _sm = surface_model
    _skf = surface_kalman_filters
    _um = unified_metrics
    with open(_cache_path(trade_date), "rb") as f:
        cache = pickle.load(f)
    _DFQ = _adapt(cache["all_quotes_iv"])
    _DFT = _adapt(cache["all_trades_iv"])


def _nan_row() -> dict:
    return {k: (0 if k == "n_snapshots" else np.nan) for k in _METRIC_KEYS}


# ── Forward-consistent negative mass (replaces evaluate()'s inline per-row BL) ──
# evaluate()'s default BL second-differences the stored per-quote kalman_price — each
# rendered at its OWN forward — against a single window forward, which fabricates
# negative mass even though the SSVI slice is butterfly-arb-free (project() enforces
# ϑζ(1+|ρ|)<4 & ϑζ²(1+|ρ|)≤4). Since one posterior θ defines the whole window's curve,
# we rebuild that single curve at ONE forward F* and BL it, then feed the result to
# evaluate(neg_mass_values=...) so the inline path is bypassed.
def _ssvi_iv(theta, k, T):
    """Vectorised SSVI implied vol (mirrors surface_model._observe_ssvi_slice)."""
    vth, rho, zeta = float(theta[0]), float(theta[1]), float(theta[2])
    vth = max(vth, 1e-12)
    Ts = max(float(T), 1e-6)
    u = zeta * k
    s = np.sqrt((u + rho) ** 2 + (1.0 - rho**2))
    w = np.maximum(0.5 * vth * (1.0 + rho * u + s), 1e-16)  # total variance
    return np.sqrt(w / Ts)


def _black76_call(F, K, T, sig):
    """Undiscounted (forward) Black-76 call — matches negative_mass_bl's C/df space."""
    sig = np.maximum(sig, 1e-9)
    Ts = max(float(T), 1e-9)
    rt = sig * np.sqrt(Ts)
    d1 = (np.log(F / K) + 0.5 * sig**2 * Ts) / rt
    return F * ndtr(d1) - K * ndtr(d1 - rt)


def _curve_neg_mass(theta, strikes, T, F):
    """Negative BL mass of the single posterior-θ SSVI curve priced at ONE forward F."""
    Ks = np.unique(np.asarray(strikes, float))
    if Ks.size < 3 or not np.isfinite(F) or F <= 0:
        return np.nan
    k = np.log(Ks / F)
    C = _black76_call(F, Ks, T, _ssvi_iv(theta, k, T))
    nm, _ = _um.negative_mass_bl(Ks, C, F)
    return nm


def _score_sides(df_quotes_res, df_state, sides: list[str]) -> dict[str, dict]:
    """Window-dedup a quote result frame and score each requested side from it.

    For "joint" fits the frame holds both sides (each `side` masks its half); for
    "per_side" fits it already holds a single side (then `sides` is that one side).
    ``df_state`` carries the per-quote posterior θ (row-aligned with ``df_quotes_res``),
    used to rebuild each window's curve for the forward-consistent negative mass.
    """
    th_cols = [f"theta_{j}" for j in range(_sm.SSVI_SLICE_MODEL.n_params)]
    qb = df_quotes_res.hstack(df_state.select(th_cols)).to_pandas()
    # 1 s window id; posterior θ (hence kalman_price) is shared within a window.
    qb["window"] = qb["ts_ns"].astype("int64") // int(BATCH_S * 1e9)
    qb = (
        qb.sort_values("ts_ns")
        .drop_duplicates(["window", "strike", "is_call"], keep="last")
        .dropna(subset=["kalman_price_prior"])
    )

    out: dict[str, dict] = {}
    for side in sides:
        side_mask = qb["is_call"] if side == "calls" else ~qb["is_call"]
        qs = qb[side_mask]
        if qs.empty:
            out[side] = _nan_row()
            continue
        # One arb-free θ-curve per window, BL'd at a single forward (see note above).
        negs = []
        for _, gw in qs.groupby("window"):
            theta = gw[th_cols].iloc[-1].to_numpy(float)  # shared within the window
            nm = _curve_neg_mass(
                theta,
                gw["strike"].to_numpy(float),
                float(gw["T"].median()),
                float(gw["forward"].median()),
            )
            if np.isfinite(nm):
                negs.append(nm)
        df_eval = _um.from_option_prices(
            ts=qs["window"].to_numpy(),
            strike=qs["strike"].to_numpy(float),
            forward=qs["forward"].to_numpy(float),
            is_call=qs["is_call"].to_numpy(bool),
            bid=qs["bid_price"].to_numpy(float),
            ask=qs["ask_price"].to_numpy(float),
            prior_price=qs["kalman_price_prior"].to_numpy(float),
            post_price=qs["kalman_price"].to_numpy(float),
            discount_factor=qs["discount_factor"].to_numpy(float)
            if "discount_factor" in qs
            else None,
        )
        out[side] = _um.evaluate(
            df_eval, model_name=f"ssvi|{side}", neg_mass_values=np.asarray(negs, float)
        )
    return out


def _eval_params(params: dict) -> dict:
    """Run the 1 s batched SSVI surface KF for one config, then score each side.

    fit_mode="joint"    — one filter run on the union of quotes; both sides scored
                          from that single shared-θ fit (legacy).
    fit_mode="per_side" — one filter run PER side on that side's quotes only; each
                          side scored against its own independently-fit θ.
    """
    import polars as pl

    cfg = _sm.SurfaceModelConfig(
        model=_sm.SSVI_SLICE_MODEL,
        Q_scale=params["Q_scale"],
        P0_scale=params["P0_scale"],
        Q_structure=params["Q_structure"],
    )

    if params["fit_mode"] == "joint":
        df_quotes_res, _, df_state = _skf.run_spline_surface_kalman(
            _DFQ, _DFT, cfg, batch_seconds=BATCH_S
        )
        return _score_sides(df_quotes_res, df_state, SIDES)

    # per_side: restrict BOTH the fit and the eval to one side's quotes/trades.
    out: dict[str, dict] = {}
    for side in SIDES:
        want_call = side == "calls"
        dfq = _DFQ.filter(pl.col("is_call") == want_call)
        dft = _DFT.filter(pl.col("is_call") == want_call)
        if dfq.is_empty():
            out[side] = _nan_row()
            continue
        df_quotes_res, _, df_state = _skf.run_spline_surface_kalman(
            dfq, dft, cfg, batch_seconds=BATCH_S
        )
        out.update(_score_sides(df_quotes_res, df_state, [side]))
    return out


def _run_one(params: dict) -> list[dict]:
    try:
        per_side = _eval_params(params)
    except Exception as e:  # keep the sweep alive; failed combos sink in the sort
        print(f"{params} failed: {e}")
        per_side = {s: _nan_row() for s in SIDES}
    base = {k: params[k] for k in _PARAM_KEYS}
    return [
        {**base, "side": side, **{k: m.get(k, np.nan) for k in _METRIC_KEYS}}
        for side, m in per_side.items()
    ]


def _make_tasks() -> list[dict]:
    return [
        {"Q_scale": q, "P0_scale": p0, "Q_structure": qs, "fit_mode": fm}
        for q, p0, qs, fm in itertools.product(Q_SCALES, P0_SCALES, Q_STRUCTURES, FIT_MODES)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main(trade_date: str, out_path: str, n_workers: int) -> None:
    if not os.path.exists(_cache_path(trade_date)):
        raise SystemExit(
            f"Missing shared cache {_cache_path(trade_date)}. Build it with:\n"
            f"  python personal/Kalman_filter_individual/phase1_cache.py --date {trade_date}"
        )

    tasks = _make_tasks()
    print(
        f"Sweeping {len(tasks)} SSVI configs (batch={BATCH_S}s, fit_modes={FIT_MODES}) "
        f"× {len(SIDES)} sides on {trade_date} across {n_workers} workers ..."
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(trade_date,)) as pool:
        records = [r for sub in pool.imap_unordered(_run_one, tasks, chunksize=1) for r in sub]

    # Rank within each (fit_mode, side) block independently so legacy "joint" and the
    # new "per_side" fits never share a leaderboard (nan-safe; failures sink to bottom).
    records.sort(
        key=lambda r: (r["fit_mode"], r["side"], np.isnan(r["ho_nse_mean"]), r["ho_nse_mean"])
    )
    rank_counter: dict[tuple, int] = {}
    for r in records:
        key = (r["fit_mode"], r["side"])
        rank_counter[key] = rank_counter.get(key, 0) + 1
        r["rank"] = rank_counter[key]

    fieldnames = ["rank", "side"] + _PARAM_KEYS + _METRIC_KEYS
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} rows ({len(SIDES)} sides × {len(tasks)} configs) → {out_path}")

    header = (
        f"{'rank':>4}  {'Q_scale':>8}  {'P0_scale':>8}  {'Q_struct':>15}  {'ho_nse':>7}"
        f"  {'is_nse':>7}  {'ho_ese':>7}  {'ho_rmse':>8}  {'ho_out%':>7}  {'neg_mass':>9}  {'arb':>6}  {'n_snap':>7}"
    )
    for fit_mode in FIT_MODES:
        for side in SIDES:
            side_rows = [r for r in records if r["fit_mode"] == fit_mode and r["side"] == side]
            print(f"\n[{fit_mode} | {side}] Top 10 by held-out mean NSE:")
            print(header)
            for r in side_rows[:10]:
                print(
                    f"{r['rank']:>4}  {r['Q_scale']:>8.0e}  {r['P0_scale']:>8.0e}"
                    f"  {r['Q_structure']:>15}  {r['ho_nse_mean']:>7.3f}  {r['is_nse_mean']:>7.3f}"
                    f"  {r['ho_ese_mean']:>7.3f}  {r['ho_rmse_mean']:>8.5f}  {r['ho_out_pct_mean']:>7.1f}"
                    f"  {r['neg_mass_mean']:>9.2e}  {r['arb_frac']:>6.1%}  {r['n_snapshots']:>7}"
                )
            if side_rows:
                b = side_rows[0]
                print(
                    f"  Best [{fit_mode} | {side}]: Q_scale={b['Q_scale']:.0e} "
                    f"P0_scale={b['P0_scale']:.0e} {b['Q_structure']} → "
                    f"ho_nse_mean={b['ho_nse_mean']:.3f} ho_rmse_mean={b['ho_rmse_mean']:.5f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", default=TRADE_DATE, help="trade date (selects the shared phase1_cache_{date}.pkl)"
    )
    parser.add_argument(
        "--out", default=None, help="output CSV (default: results/ssvi_sweep_{date}.csv)"
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()
    out = args.out or os.path.join(_HERE, "results", f"ssvi_sweep_{args.date}.csv")
    main(args.date, out, args.workers)
