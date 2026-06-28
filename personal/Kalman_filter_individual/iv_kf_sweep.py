"""
Single-strike IV Kalman-filter q_scale sweep — 1 s batch, unified metrics.

Drives the batched single-strike IV Kalman (kalman_filters.iv_surface_from_frames,
the model behind build_iv_surface) over a phase1 cache produced by phase1_cache.py.
Each worker loads the cache once; every q_scale then runs only the cheap Phase-3
filter, so the slow data load + IV computation is paid exactly once.

The batch window is fixed at 1 s (BATCH_S) and only q_scale is swept.  Each combo
is scored with the project's unified metric set (personal.shared_functions.
unified_metrics), reconstructed exactly as the "KF single-strike (1s batch)" row of
unified_metrics_comparison.ipynb:
  * held-out / in-sample NSE  (|prior/post call - mid| / half-spread, price space)
  * held-out / in-sample RMSE (price space)
  * implied negative mass + arbitrage fraction (Breeden-Litzenberger)

Results are sorted by held-out mean NSE (ho_nse_mean).

Usage
-----
    python phase1_cache.py --date 2024-01-03                      # run once per date
    python iv_kf_sweep.py [--date 2024-01-03] [--out results/iv_kf_sweep_2024-01-03.csv] [--workers 4]

Edit Q_SCALES below to define the search space.
"""

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

project_root = "/Users/elie/Documents/GitHub/quant_data/"
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (project_root, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QUANTDATA_RAW_DATA", "!/quantdata/raw")
os.environ.setdefault("QUANTDATA_TRAINING_DATA", "!/quantdata/training")
os.environ.setdefault("QUANTDATA_PROCESSING_DATA", "!/quantdata/processing")
os.environ.setdefault("QUANTDATA_CACHE_MODE", "on")
os.environ.setdefault("QUANTDATA_TEST_MODE", "0")

# ── Search space — edit this ──────────────────────────────────────────────────
Q_SCALES = [1e-9, 1e-6, 1e-3, 1e-2, 0.1, 30.0]
BATCH_S = 1.0  # batch window width (seconds) — fixed, not swept

# Market sides scored INDEPENDENTLY. The single-strike KF is per-contract, so each
# side is just a post-hoc filter of the same KF run (no re-run). One result row per
# (q_scale, side); metrics for "puts" / "calls" never mix.
SIDES = ["puts", "calls"]

TRADE_DATE = "2024-01-03"
N_NEG_MASS_SNAPS = 500  # snapshots sampled for the negative-mass reconstruction

# Metric columns emitted by unified_metrics.evaluate (minus "model").
# Order mirrors evaluate()'s return dict so the CSV captures the full metric set,
# including the out-of-band fractions (|model - mid| > half_spread, in %).
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

# ── Worker globals (set once per process by _init_worker) ─────────────────────
_kf = None  # personal.shared_functions.kalman_filters
_um = None  # personal.shared_functions.unified_metrics
_cache = None


def _cache_path(trade_date: str) -> str:
    return os.path.join(_HERE, f"phase1_cache_{trade_date}.pkl")


def _init_worker(trade_date: str) -> None:
    global _kf, _um, _cache
    import personal.shared_functions.kalman_filters as kalman_filters
    import personal.shared_functions.unified_metrics as unified_metrics

    _kf = kalman_filters
    _um = unified_metrics
    with open(_cache_path(trade_date), "rb") as f:
        _cache = pickle.load(f)


def _negative_mass(df_single):
    """Per-snapshot negative mass for a SINGLE option side, reconstructing that
    side's smile over its strike grid.

    `df_single` is already restricted to one side (one quote-stream per strike).
    Build a (n_snap, n_strike) posterior call-price matrix at evenly-sampled times
    and apply negative_mass_surface. post_call is in undiscounted call space (put
    quotes are mapped via parity in from_option_prices), so this works either side.
    """
    ds = df_single[["ts", "strike", "forward", "post_call"]].copy()
    if ds.empty:
        return np.array([])

    unique_K = np.sort(ds["strike"].unique())
    if len(unique_K) < 3:
        return np.array([])

    n_snap = min(N_NEG_MASS_SNAPS, ds["ts"].nunique())
    ts_int = ds["ts"].astype(np.int64).values
    snap_ts_int = np.linspace(ts_int.min(), ts_int.max(), n_snap, dtype=np.int64)

    groups = {K: g.sort_values("ts").reset_index(drop=True) for K, g in ds.groupby("strike")}

    ref_K = unique_K[len(unique_K) // 2]
    ref_grp = groups[ref_K]
    ref_ts = ref_grp["ts"].astype(np.int64).values
    ref_asof = np.clip(np.searchsorted(ref_ts, snap_ts_int, side="right") - 1, 0, len(ref_grp) - 1)
    F_arr = ref_grp["forward"].values[ref_asof]

    C_mat = np.full((n_snap, len(unique_K)), np.nan)
    for j, K_j in enumerate(unique_K):
        grp = groups[K_j]
        grp_ts = grp["ts"].astype(np.int64).values
        idx = np.clip(np.searchsorted(grp_ts, snap_ts_int, side="right") - 1, 0, len(grp) - 1)
        C_mat[:, j] = grp["post_call"].values[idx]

    valid = np.isfinite(C_mat).all(axis=1) & np.isfinite(F_arr)
    if not valid.any():
        return np.array([])
    return _um.negative_mass_surface(unique_K, C_mat[valid], F_arr[valid])


def _eval_q(q_scale: float) -> dict:
    """Run the 1 s batched KF ONCE for one q_scale, then score each side in SIDES
    independently. Returns {side: unified-metric dict}."""
    df_quotes_iv, _ = _kf.iv_surface_from_frames(
        _cache["all_quotes_iv"],
        _cache["all_trades_iv"],
        _cache["ticker_meta"],
        batch_seconds=BATCH_S,
        q_scale=q_scale,
    )

    qb = df_quotes_iv.sort(["strike", "is_call", "ts_ns"]).to_pandas()
    # 1 s window id; posterior price is constant within (strike, is_call, window).
    qb["window"] = qb["ts_ns"].astype("int64") // int(BATCH_S * 1e9)
    # held-out prior = previous window's posterior price (representative = last quote).
    post_w = qb.groupby(["strike", "is_call", "window"], as_index=False).agg(
        _post_win=("kalman_price", "last")
    )
    post_w["kalman_price_prior"] = post_w.groupby(["strike", "is_call"])["_post_win"].shift(1)
    qb = qb.merge(
        post_w[["strike", "is_call", "window", "kalman_price_prior"]],
        on=["strike", "is_call", "window"],
        how="left",
    ).dropna(subset=["kalman_price_prior"])

    out: dict[str, dict] = {}
    for side in SIDES:
        side_mask = qb["is_call"] if side == "calls" else ~qb["is_call"]
        qs = qb[side_mask]
        if qs.empty:
            out[side] = {k: (0 if k == "n_snapshots" else np.nan) for k in _METRIC_KEYS}
            continue
        df_single = _um.from_option_prices(
            ts=qs["ts_ns"],
            strike=qs["strike"],
            forward=qs["forward"],
            is_call=qs["is_call"],
            bid=qs["bid_price"],
            ask=qs["ask_price"],
            prior_price=qs["kalman_price_prior"],
            post_price=qs["kalman_price"],
            discount_factor=qs["discount_factor"] if "discount_factor" in qs else None,
        )
        nm = _negative_mass(df_single)
        out[side] = _um.evaluate(
            df_single,
            model_name=f"q={q_scale:g}|{side}",
            neg_mass_values=nm if len(nm) else None,
        )
    return out


def _run_one(q_scale: float) -> list[dict]:
    try:
        per_side = _eval_q(q_scale)
    except Exception as e:  # keep the sweep alive; failed combos sink in the sort
        print(f"q_scale={q_scale:g} failed: {e}")
        nan_row = {k: (0 if k == "n_snapshots" else np.nan) for k in _METRIC_KEYS}
        per_side = {s: nan_row for s in SIDES}
    return [
        {"q_scale": q_scale, "side": side, **{k: m.get(k, np.nan) for k in _METRIC_KEYS}}
        for side, m in per_side.items()
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main(trade_date: str, out_path: str, n_workers: int) -> None:
    if not os.path.exists(_cache_path(trade_date)):
        raise SystemExit(
            f"Missing cache {_cache_path(trade_date)}. Run: python phase1_cache.py --date {trade_date}"
        )

    tasks = list(Q_SCALES)
    print(
        f"Sweeping {len(tasks)} q_scale values (batch={BATCH_S}s) on {trade_date} "
        f"across {n_workers} workers ..."
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(trade_date,)) as pool:
        # each task yields one record PER side; flatten.
        records = [r for sub in pool.imap_unordered(_run_one, tasks, chunksize=1) for r in sub]

    # Rank within each side independently (nan-safe; failures sink to the bottom).
    records.sort(key=lambda r: (r["side"], np.isnan(r["ho_nse_mean"]), r["ho_nse_mean"]))
    side_counter: dict[str, int] = {}
    for r in records:
        side_counter[r["side"]] = side_counter.get(r["side"], 0) + 1
        r["rank"] = side_counter[r["side"]]

    fieldnames = ["rank", "side", "q_scale"] + _METRIC_KEYS
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} rows ({len(SIDES)} sides × {len(tasks)} q) → {out_path}")

    header = (
        f"{'rank':>4}  {'q_scale':>8}  {'ho_nse':>7}  {'ho_nse_med':>10}  {'is_nse':>7}"
        f"  {'ho_ese':>7}  {'ho_rmse':>8}  {'ho_out%':>7}  {'neg_mass':>9}  {'arb':>6}  {'n_snap':>7}"
    )
    for side in SIDES:
        side_rows = [r for r in records if r["side"] == side]
        print(f"\n[{side}] Top 10 by held-out mean NSE:")
        print(header)
        for r in side_rows[:10]:
            print(
                f"{r['rank']:>4}  {r['q_scale']:>8.0e}  {r['ho_nse_mean']:>7.3f}"
                f"  {r['ho_nse_med']:>10.3f}  {r['is_nse_mean']:>7.3f}"
                f"  {r['ho_ese_mean']:>7.3f}  {r['ho_rmse_mean']:>8.5f}"
                f"  {r['ho_out_pct_mean']:>7.1f}  {r['neg_mass_mean']:>9.2e}"
                f"  {r['arb_frac']:>6.1%}  {r['n_snapshots']:>7}"
            )
        if side_rows:
            best = side_rows[0]
            print(
                f"  Best [{side}]: q_scale={best['q_scale']:.0e}  → "
                f"ho_nse_mean={best['ho_nse_mean']:.3f}  ho_rmse_mean={best['ho_rmse_mean']:.5f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", default=TRADE_DATE, help="trade date (selects phase1_cache_{date}.pkl)"
    )
    parser.add_argument(
        "--out", default=None, help="output CSV (default: results/iv_kf_sweep_{date}.csv)"
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()
    out = args.out or os.path.join(_HERE, "results", f"iv_kf_sweep_{args.date}.csv")
    main(args.date, out, args.workers)
