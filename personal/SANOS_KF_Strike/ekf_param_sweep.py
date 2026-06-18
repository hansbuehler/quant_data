"""
One-phase SANOS EKF hyperparameter sweep — q_scale / ell / p0_scale / warm / correlated.

Drives the direct-observation EKF in sanos_ekf_eval.run() (the latest model),
which assimilates raw market quotes in price space instead of fitting SANOS per
snapshot and filtering the fit. Requires a phase1 cache produced by
phase1_cache.py (data loading + DLV fitting done once and serialised to disk).

Each worker process loads the cache once via the Pool initializer (which
populates sanos_ekf_eval's module globals through load_cache), then evaluates
many parameter combinations without re-loading.

Usage
-----
    python phase1_cache.py --out phase1_cache_2024-01-03.pkl   # run once per date
    python ekf_param_sweep.py [--date 2024-01-03] [--out results/ekf_sweep_results_2024-01-03.csv] [--workers 4]

Edit PARAM_GRID below to define the search space.
Results are sorted by held-out mean NSE (one-step-ahead prior metric).
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sanos_ekf_eval as ekf

# ── Parameter grid — edit this ────────────────────────────────────────────────
# q_scale     process-noise scale in DLV²² per trading day (state moves little
#             intraday, so this is small — O(0.1–10), unlike the fit-then-filter
#             pipeline where the observation was DLV² with R≈0).
# ell         RBF correlation length over DLV nodes (tight wins; ties wings to
#             quoted neighbours).
# p0_scale    initial prior covariance scale.
# warm        True = seed x0 from first snapshot's SANOS fit (warm-start hurts —
#             it caps the wings); False = flat init.
# correlated  True = correlated Q/P0 (helps); False = diagonal.
PARAM_GRID = {
    "q_scale": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1, 3e1, 1e2],
    "ell": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
    "p0_scale": [1e-3, 1e-2, 1e-1, 1.0],
    "warm": [False, True],
    "correlated": [True, False],
}
# ≈ 11×9×4×2×2 = 1584 combos. At ~2.3 s/run on 11 workers that's ~6 min.
# Trim any axis (or pin warm=[False], correlated=[True]) for a quick pass.

TRADE_DATE = "2024-01-03"


# ── Worker globals (set once per process by _init_worker) ─────────────────────
def _init_worker(trade_date: str) -> None:
    # Repopulate sanos_ekf_eval's module globals for this date in each process.
    ekf.load_cache(trade_date)


def _run_one(task: dict) -> dict:
    o = ekf.run(
        q_scale=task["q_scale"],
        ell=task["ell"],
        p0_scale=task["p0_scale"],
        warm=task["warm"],
        correlated=task["correlated"],
    )
    return {
        "q_scale": task["q_scale"],
        "ell": task["ell"],
        "p0_scale": task["p0_scale"],
        "warm": int(task["warm"]),
        "correlated": int(task["correlated"]),
        "ho_mean": o["ho_mean"],
        "ho_med": o["ho_med"],
        "is_mean": o["is_mean"],
        "is_med": o["is_med"],
        "n_eval": o["n_eval"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(trade_date: str, out_path: str, n_workers: int) -> None:
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    tasks = [dict(zip(keys, c)) for c in combos]
    total = len(tasks)
    sizes = "  ".join(f"{k}×{len(PARAM_GRID[k])}" for k in keys)
    eta_s = total * 2.3 / max(n_workers, 1)  # ~2.3 s/run measured
    print(
        f"Sweeping {total} combinations ({sizes}) on {trade_date} across "
        f"{n_workers} workers ... (~{eta_s / 60:.1f} min)"
    )

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(trade_date,),
    ) as pool:
        records = list(pool.imap_unordered(_run_one, tasks, chunksize=2))

    # nan-safe sort: failed runs (nan ho_mean) sink to the bottom.
    records.sort(key=lambda r: (np.isnan(r["ho_mean"]), r["ho_mean"]))
    for i, r in enumerate(records, 1):
        r["rank"] = i

    fieldnames = [
        "rank",
        "q_scale",
        "ell",
        "p0_scale",
        "warm",
        "correlated",
        "ho_mean",
        "ho_med",
        "is_mean",
        "is_med",
        "n_eval",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {total} rows → {out_path}")

    # Baseline for context (computed once in the main process).
    ekf.load_cache(trade_date)
    base = ekf.persist_baseline()
    print(
        f"\nbaseline (persist-all, same metric): "
        f"HO mean {base['ho_mean']:.3f} / med {base['ho_med']:.3f}   "
        f"(gate ref: mean 0.67 / med 0.57)"
    )

    print("\nTop 10 by held-out mean NSE:")
    header = (
        f"{'rank':>4}  {'q_scale':>8}  {'ell':>5}  {'p0':>6}  {'warm':>4}  {'corr':>4}"
        f"  {'ho_mean':>8}  {'ho_med':>7}  {'is_mean':>8}  {'is_med':>7}  {'n_eval':>7}"
    )
    print(header)
    for r in records[:10]:
        print(
            f"{r['rank']:>4}  {r['q_scale']:>8.0e}  {r['ell']:>5.1f}  {r['p0_scale']:>6.0e}"
            f"  {r['warm']:>4}  {r['correlated']:>4}"
            f"  {r['ho_mean']:>8.3f}  {r['ho_med']:>7.3f}  {r['is_mean']:>8.3f}"
            f"  {r['is_med']:>7.3f}  {r['n_eval']:>7}"
        )
    best = records[0]
    print(
        f"\nBest: q_scale={best['q_scale']:.0e}  ell={best['ell']}"
        f"  p0={best['p0_scale']:.0e}  warm={best['warm']}  corr={best['correlated']}"
        f"  → HO mean={best['ho_mean']:.3f} med={best['ho_med']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", default=TRADE_DATE, help="trade date (selects phase1_cache_{date}.pkl)"
    )
    parser.add_argument(
        "--out", default=None, help="output CSV (default: results/ekf_sweep_results_{date}.csv)"
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()
    out = args.out or f"results/ekf_sweep_results_{args.date}.csv"
    main(args.date, out, args.workers)
