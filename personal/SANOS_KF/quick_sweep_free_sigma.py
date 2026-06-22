"""
quick_sweep_free_sigma.py

Fast, focused sweep over the ONE model: the exact tridiagonal SANOS density in
absolute-strike space (see sanos_kf.py). There is no BS-mixture and no "free σ*"
state — σ* (cfg.sigma_star_init) is a GIVEN regularizer that, when cfg.smoothing
is on, blurs each density node into a lognormal of vol σ*. It smooths only the
OBSERVATION (how market quotes are priced from the density), never the density
itself, which is always the exact solve. The question this sweep answers:

    does the σ*-smoothed observation (cfg.smoothing=True) beat the intrinsic
    discrete-call observation (σ*=0, cfg.smoothing=False)?

Both arms are the exact-solve density; they differ only in the observation
payoff. Reuses sweep_params' worker (_run_one), data loader, and BASE/SIDE so
the score is identical to the full sweep — we only swap in a small, targeted
grid and run it across a process pool.

    python personal/SANOS_KF/quick_sweep_free_sigma.py
    python personal/SANOS_KF/quick_sweep_free_sigma.py --workers 4 --mode exact_smoothed
"""

from __future__ import annotations

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
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import pandas as pd
import sweep_params as sp  # noqa: E402  (same dir; sys.path set by sweep_params import)

# --- Focused grids -----------------------------------------------------------
# Both grids are the exact-solve density; they differ only in the observation
# (σ*-smoothed vs intrinsic). σ* (sigma_star_init) is a given regularizer, not a
# fitted state. `ell` is the RBF correlation length in DOLLARS (strike space).

# exact_smoothed: smoothing on (σ* > 0). Sweep the two knobs that decide fit
# quality (RBF length `ell` and DLV process noise) with a light σ* exploration.
EXACT_SMOOTHED = dict(
    smoothing=True,
    Sigma_init=[0.04],
    P0_scale=[0.1],
    ell=[1.0, 5.0],  # DOLLARS — the key dim
    Q_scale_dlv=[0.1, 1.0],
    sigma_star_init=[1e-4, 0.001, 0.01, 0.05, 0.1],
)

# exact_intrinsic baseline: smoothing off (σ* = 0, intrinsic discrete calls).
# A few points for the comparison line on the same date/side.
EXACT_INTRINSIC = dict(
    smoothing=False,
    sigma_star_init=[0.0],
    Sigma_init=[0.04],
    P0_scale=[0.1],
    Q_scale_dlv=[0.3, 1.0],
    ell=[1.0, 3.0],  # DOLLARS
)


def _points(mode: str, grid: dict) -> list[dict]:
    keys = list(grid)
    vals = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [{"mode": mode, **dict(zip(keys, c))} for c in itertools.product(*vals)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--date", default="2024-01-03")
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument(
        "--mode",
        choices=["exact_smoothed", "exact_intrinsic"],
        default=None,
        help="restrict the sweep to a single mode (default: run both)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    points = _points("exact_smoothed", EXACT_SMOOTHED) + _points("exact_intrinsic", EXACT_INTRINSIC)
    if args.mode is not None:
        points = [p for p in points if p["mode"] == args.mode]
    out = args.out or f"quick_sweep_free_sigma_{args.ticker}_{args.date}.csv"
    trade_date = date.fromisoformat(args.date)

    ns = sum(p["mode"] == "exact_smoothed" for p in points)
    ni = sum(p["mode"] == "exact_intrinsic" for p in points)
    print(f"Grid: {len(points)} points (exact_smoothed={ns}, exact_intrinsic={ni})")

    print(f"Loading {args.ticker} {args.date} ...")
    df_quotes, df_trades = sp._load_data(args.ticker, trade_date, args.min_trades)
    print(f"quotes: {len(df_quotes):,}   trades: {len(df_trades):,}")

    tmpdir = tempfile.mkdtemp(prefix="quick_sweep_")
    q_path, t_path = os.path.join(tmpdir, "q.parquet"), os.path.join(tmpdir, "t.parquet")
    df_quotes.write_parquet(q_path)
    df_trades.write_parquet(t_path)
    del df_quotes, df_trades
    gc.collect()

    fields = [
        "mode",
        "smoothing",
        "Sigma_init",
        "Q_scale_dlv",
        "P0_scale",
        "ell",
        "sigma_star_init",
        "ok",
        "err",
        "secs",
        "ho_nse_mean",
        "ho_nse_med",
        "ho_nse_std",
        "is_nse_mean",
        "is_nse_med",
        "ho_rmse_mean",
        "is_rmse_mean",
        "arb_frac",
        "n_snapshots",
    ]

    rows = []
    t0 = time.perf_counter()
    print(f"Running on {args.workers} workers -> {out}")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=sp._init_worker,
            initargs=(q_path, t_path),
            max_tasks_per_child=20,
        ) as ex:
            futs = {ex.submit(sp._run_one, p): p for p in points}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                w.writerow(r)
                fh.flush()
                rows.append(r)
                tag = "ok " if r.get("ok") else "ERR"
                print(
                    f"[{i}/{len(points)}] {tag} {r['mode']:11s} "
                    f"ho_nse_med={r.get('ho_nse_med', float('nan')):.4f} "
                    f"({r['secs']:.1f}s)" + (f"  {r['err']}" if not r.get("ok") else ""),
                    flush=True,
                )

    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    df = pd.read_csv(out)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Wrote {len(df)} rows to {out}\n")
    show = [
        c
        for c in [
            "mode",
            "ell",
            "Q_scale_dlv",
            "sigma_star_init",
            "ho_nse_med",
            "ho_nse_mean",
            "is_nse_med",
            "arb_frac",
            "secs",
        ]
        if c in df.columns
    ]
    for mode in ("exact_smoothed", "exact_intrinsic"):
        sub = df[df["mode"] == mode].sort_values("ho_nse_med", na_position="last")
        print(f"=== {mode}  (best ho_nse_med first) ===")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(sub[show].head(8).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
