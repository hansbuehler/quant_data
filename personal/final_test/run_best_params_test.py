"""
run_best_params_test.py — held-out TEST of the three models at their best params.

This is NOT a sweep. Each model runs ONE configuration (its best, plugged in below)
on the 20 held-out test days (default 2024-01-18 .. 2024-02-14) and is scored with
the project's unified metric set. Output is one tidy CSV:

    model, date, side, ok, err, params(json), <unified-metrics...>

with 3 models × 20 days × 2 sides = 120 rows.

Why this layout (the parallelization choice)
--------------------------------------------
The parallel axis is the DATE, not the model. Three-workers-one-per-model gives only
3-way parallelism and load-balances terribly (SANOS dwarfs indiv+SSVI, so two cores
idle every day) — and indiv+SSVI SHARE the phase-1 cache, so splitting them wastes a
double load. Across dates we get 20-way parallelism, each date an independent unit,
and indiv+SSVI naturally share their one cache load inside the worker.

Two phases, because the cost is lopsided:
  * Phase A (build) — for these 20 NEW days no caches exist yet; building them is the
    long pole (network fetch + IV batch, I/O-bound, ~0% CPU). Each build runs in its own
    subprocess (full env inherited — the shell exports QUANTDATA_* → …/quant_data/DATA)
    for clean memory reclamation between the 40 big builds, and they run in parallel
    across dates (I/O-bound ⇒ concurrency helps; memory ⇒ keep --build-workers modest).
  * Phase B (eval) — reads the on-disk caches only (never fetches), so it is env-free.
    Parallel across dates in an in-process pool; each worker loads both caches and runs
    all three best-param models × both sides. SANOS holds ~GB + JAX per worker, so keep
    --workers modest and recycle the worker after each date (max_tasks_per_child=1).

Usage
-----
    python personal/final_test/run_best_params_test.py            # 20 default test days
    python personal/final_test/run_best_params_test.py --start 2024-01-18 --end 2024-02-14
    python personal/final_test/run_best_params_test.py --dates 2024-01-18 2024-01-19
    python personal/final_test/run_best_params_test.py --skip-build      # caches present
    python personal/final_test/run_best_params_test.py --workers 4 --build-workers 3

Edit BEST_INDIV / BEST_SSVI / BEST_SANOS below before running.
"""

from __future__ import annotations

# Keep every worker / build subprocess single-threaded so N processes don't
# oversubscribe the CPU (BLAS + JAX/XLA each spawn thread pools). Set before any
# numpy / jax import. QUANTDATA_* is intentionally left to the inherited shell env
# (which exports them all → …/quant_data/DATA), so builds work like the standalone
# scripts.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

PROJECT_ROOT = "/Users/elie/Documents/GitHub/quant_data/"
_HERE = os.path.dirname(os.path.abspath(__file__))
_INDIV = os.path.join(PROJECT_ROOT, "personal", "Kalman_filter_individual")
_SSVI = os.path.join(PROJECT_ROOT, "personal", "Kalman_filter_SSVI")
_SPLINE = os.path.join(PROJECT_ROOT, "personal", "Kalman_filter_spline")
_PHASE1_BUILDER = os.path.join(_INDIV, "phase1_cache.py")

TICKER = "SPY"
SIDES = ["puts", "calls"]

DEFAULT_START = "2024-01-18"
DEFAULT_END = "2024-02-14"  # 20 held-out business days

# ─────────────────────────────────────────────────────────────────────────────
#  BEST PARAMETERS  — plug in the winning config per model, per side.
#
#  Per-side dicts so puts and calls can carry DIFFERENT best params (the sweeps
#  rank each side independently). If a model's best is the same on both sides,
#  just make the two entries identical.
#
#  • BEST_INDIV[side]  -> {"q_scale": ...}                 (iv_kf_sweep._eval_q)
#  • BEST_SSVI[side]   -> {Q_scale, P0_scale, Q_structure, fit_mode}
#                                                          (ssvi_sweep._eval_params)
#  • BEST_SANOS[side]  -> the full SanosKFConfig grid point + the two report tags
#       sigmoid_mode / obs_mode. Keys must match what sweep_params._run_one expects:
#       sigmoid_mode, obs_mode, use_sigmoid, smoothing, Sigma_init, Q_scale_dlv,
#       P0_scale, max_dlvsq, ell  (+ eta only when obs_mode="smoothed").
#
#  The values below are PLACEHOLDERS taken from the sweep grids' "best-pin" hints —
#  REPLACE them with the actual winners before the real run.
# ─────────────────────────────────────────────────────────────────────────────
BEST_INDIV: dict[str, dict] = {
    "puts": {"q_scale": 0.01},  # TODO: plug in best q_scale (puts)
    "calls": {"q_scale": 0.01},  # TODO: plug in best q_scale (calls)
}

_SSVI_COMMON = {"P0_scale": 1.0, "Q_structure": "diagonal", "fit_mode": "per_side"}
BEST_SSVI: dict[str, dict] = {
    "puts": {"Q_scale": 1e-4, **_SSVI_COMMON},  # TODO: plug in best Q_scale (puts)
    "calls": {"Q_scale": 1e-4, **_SSVI_COMMON},  # TODO: plug in best Q_scale (calls)
}

_SANOS_COMMON = {
    "sigmoid_mode": "sigmoid",
    "obs_mode": "smoothed",
    "use_sigmoid": True,
    "smoothing": True,
    "Sigma_init": 0.1,
    "P0_scale": 1e-2,
    "max_dlvsq": 10.0,
    "ell": 1.0,
    "eta": 0.06,
    "Q_scale_dlv": 0.1,
}
BEST_SANOS: dict[str, dict] = {
    "puts": {**_SANOS_COMMON},  # TODO: plug in best (puts)
    "calls": {**_SANOS_COMMON},  # TODO: plug in best (calls)
}

# Unified-metrics columns (identical across the three sweeps' evaluate()).
METRIC_KEYS = [
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


def _setup_paths() -> None:
    """Make the three (non-package) sweep modules importable, plus the project root."""
    for p in (PROJECT_ROOT, _INDIV, _SSVI, _SPLINE):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_dates(args) -> list[str]:
    _setup_paths()
    from quantdata.biz_calendar import get_business_days, to_date

    if args.dates:
        out = []
        for s in args.dates:
            d = to_date(s)
            if not get_business_days(d, d):
                print(f"skip {d}: not a trading day")
                continue
            out.append(str(d))
        return out
    start = args.start or DEFAULT_START
    end = args.end or DEFAULT_END
    return [str(d) for d in get_business_days(to_date(start), to_date(end))]


# ── cache paths ───────────────────────────────────────────────────────────────
def _phase1_path(date_str: str) -> str:
    """Shared single-strike / SSVI cache (indiv pipeline owns the location)."""
    return os.path.join(_INDIV, f"phase1_cache_{date_str}.pkl")


def _sanos_paths(date_str: str) -> tuple[str, str]:
    _setup_paths()
    from personal.SANOS_KF import sweep_params
    from quantdata.biz_calendar import to_date

    return sweep_params._data_cache_paths(TICKER, to_date(date_str), 0)


# ════════════════════════════════════════════════════════════════════════════
#  PHASE A — cache building (one subprocess per build, full env inherited)
# ════════════════════════════════════════════════════════════════════════════
# Each build runs in its own subprocess purely for clean memory reclamation between
# the 40 big builds — NOT for env isolation. The shell already exports every
# QUANTDATA_* (RAW/PROCESSING/TRAINING/ROOT → …/quant_data/DATA), so the builders'
# setdefault()s are no-ops and both point at the same real data roots: just inherit
# the full environment (the standalone scripts run exactly this way).
def _build_phase1(date_str: str, rebuild: bool) -> tuple[str, bool, str]:
    out = _phase1_path(date_str)
    if os.path.exists(out) and not rebuild:
        return (date_str, True, "phase1 cache exists")
    cmd = [sys.executable, _PHASE1_BUILDER, "--date", date_str, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(out)
    return (date_str, ok, "" if ok else (r.stderr.strip()[-500:] or "phase1 build failed"))


def _build_sanos(date_str: str, rebuild: bool) -> tuple[str, bool, str]:
    q_path, t_path = _sanos_paths(date_str)
    if os.path.exists(q_path) and os.path.exists(t_path) and not rebuild:
        return (date_str, True, "sanos cache exists")
    # Build the SANOS parquet in a fresh subprocess (this same script, --build-sanos)
    # so its ~GB of build transients are reclaimed when the process exits.
    cmd = [sys.executable, os.path.abspath(__file__), "--build-sanos", date_str]
    if rebuild:
        cmd.append("--rebuild-cache")
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(q_path) and os.path.exists(t_path)
    return (date_str, ok, "" if ok else (r.stderr.strip()[-500:] or "sanos build failed"))


def _build_sanos_inproc(date_str: str, rebuild: bool) -> None:
    """The --build-sanos subprocess body: build the per-date SANOS parquet cache."""
    _setup_paths()
    from personal.SANOS_KF import sweep_params
    from quantdata.biz_calendar import to_date

    sweep_params._ensure_data_cache(TICKER, to_date(date_str), 0, rebuild=rebuild)


def _phase_a_build(dates: list[str], build_workers: int, rebuild: bool) -> None:
    """Build both caches for every date, parallel across dates. Threads here just
    wait on the build subprocesses (the real isolation/parallelism is the OS
    processes), so the GIL is irrelevant."""
    jobs: list[tuple[str, str]] = []  # (kind, date)
    for d in dates:
        jobs.append(("phase1", d))
        jobs.append(("sanos", d))
    print(
        f"[Phase A] building {len(jobs)} caches ({len(dates)} dates × 2) "
        f"on {build_workers} parallel builders ..."
    )
    t0 = time.perf_counter()
    fails: list[str] = []
    with ThreadPoolExecutor(max_workers=build_workers) as ex:
        futs = {}
        for kind, d in jobs:
            fn = _build_phase1 if kind == "phase1" else _build_sanos
            futs[ex.submit(fn, d, rebuild)] = (kind, d)
        for i, fut in enumerate(as_completed(futs), 1):
            kind, d = futs[fut]
            _, ok, msg = fut.result()
            tag = "ok " if ok else "ERR"
            print(
                f"[Phase A][{i}/{len(jobs)}] {tag} {kind} {d}"
                + (f"  {msg}" if (not ok or "exists" in msg) else ""),
                flush=True,
            )
            if not ok:
                fails.append(f"{kind} {d}: {msg}")
    print(
        f"[Phase A] done in {time.perf_counter() - t0:.0f}s"
        + (f", {len(fails)} FAILED" if fails else "")
    )
    for f in fails:
        print(f"  FAIL {f}")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE B — evaluation (in-process pool across dates)
# ════════════════════════════════════════════════════════════════════════════
def _row(
    model: str, date_str: str, side: str, params: dict, metrics: dict, ok: bool, err: str
) -> dict:
    r = {
        "model": model,
        "date": date_str,
        "side": side,
        "ok": ok,
        "err": err,
        "params": json.dumps(params, sort_keys=True),
    }
    for k in METRIC_KEYS:
        r[k] = metrics.get(k) if metrics else None
    return r


ALL_MODELS = ["indiv_kf", "ssvi", "sanos_kf"]


def _eval_date(date_str: str, models: list[str]) -> list[dict]:
    """Run the requested best-param models × both sides for ONE date.
    Each (model, side) is wrapped so a single failure never sinks the date."""
    _setup_paths()
    rows: list[dict] = []

    # ── indiv KF — one filter run scores both sides; call per-side so each side can
    #    use its own best q_scale, keeping only that side's metrics.
    if "indiv_kf" in models:
        try:
            import iv_kf_sweep

            iv_kf_sweep._init_worker(date_str)
            for side in SIDES:
                p = BEST_INDIV[side]
                try:
                    m = iv_kf_sweep._eval_q(p["q_scale"])[side]
                    rows.append(_row("indiv_kf", date_str, side, p, m, True, ""))
                except Exception as e:
                    rows.append(
                        _row("indiv_kf", date_str, side, p, {}, False, f"{type(e).__name__}: {e}")
                    )
        except Exception as e:
            for side in SIDES:
                rows.append(
                    _row("indiv_kf", date_str, side, BEST_INDIV[side], {}, False, f"init: {e}")
                )

    # ── SSVI — per_side fit isolates the side internally; keep that side's metrics.
    if "ssvi" in models:
        try:
            import ssvi_sweep

            ssvi_sweep._init_worker(date_str)
            for side in SIDES:
                p = BEST_SSVI[side]
                try:
                    m = ssvi_sweep._eval_params(dict(p))[side]
                    rows.append(_row("ssvi", date_str, side, p, m, True, ""))
                except Exception as e:
                    rows.append(
                        _row("ssvi", date_str, side, p, {}, False, f"{type(e).__name__}: {e}")
                    )
        except Exception as e:
            for side in SIDES:
                rows.append(_row("ssvi", date_str, side, BEST_SSVI[side], {}, False, f"init: {e}"))

    # ── SANOS KF — _run_one runs one filter fed one side and returns its own row.
    if "sanos_kf" in models:
        try:
            from personal.SANOS_KF import sweep_params

            q_path, t_path = _sanos_paths(date_str)
            sweep_params._init_worker(q_path, t_path)
            for side in SIDES:
                p = BEST_SANOS[side]
                try:
                    r = sweep_params._run_one({**p, "side": side})
                    rows.append(
                        _row(
                            "sanos_kf",
                            date_str,
                            side,
                            p,
                            r,
                            bool(r.get("ok")),
                            str(r.get("err", "")),
                        )
                    )
                except Exception as e:
                    rows.append(
                        _row("sanos_kf", date_str, side, p, {}, False, f"{type(e).__name__}: {e}")
                    )
        except Exception as e:
            for side in SIDES:
                rows.append(
                    _row("sanos_kf", date_str, side, BEST_SANOS[side], {}, False, f"init: {e}")
                )

    return rows


def _phase_b_eval(dates: list[str], workers: int, out_path: str, models: list[str]) -> None:
    n_sides = len(SIDES)
    print(
        f"[Phase B] scoring {len(dates)} dates × {len(models)} model(s) ({','.join(models)}) "
        f"× {n_sides} sides on {workers} workers → {out_path}"
    )
    fieldnames = ["model", "date", "side", "ok", "err", "params"] + METRIC_KEYS
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.perf_counter()
    n = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # Recycle each worker after one date so SANOS's ~GB + JAX is fully reclaimed.
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
            futs = {ex.submit(_eval_date, d, models): d for d in dates}
            for i, fut in enumerate(as_completed(futs), 1):
                d = futs[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    print(f"[Phase B][{i}/{len(dates)}] {d} CRASHED: {e}", flush=True)
                    continue
                for r in rows:
                    writer.writerow(r)
                fh.flush()
                n += len(rows)
                ok = sum(bool(r["ok"]) for r in rows)
                print(f"[Phase B][{i}/{len(dates)}] {d}: {ok}/{len(rows)} ok", flush=True)
    print(f"[Phase B] done in {time.perf_counter() - t0:.0f}s — wrote {n} rows → {out_path}")


def _merge_into(target: str, fresh: str, models: list[str], dates: list[str]) -> None:
    """Splice the freshly-computed rows (file `fresh`) into `target`, REPLACING only
    the (model ∈ models, date ∈ dates) rows there and keeping every other row intact."""
    import pandas as pd

    new = pd.read_csv(fresh)
    if not os.path.exists(target):
        new.to_csv(target, index=False)
        print(f"[merge] {target} did not exist → wrote {len(new)} fresh rows")
        return
    old = pd.read_csv(target)
    dset = set(dates)
    keep = ~(old["model"].isin(models) & old["date"].astype(str).isin(dset))
    merged = pd.concat([old[keep], new], ignore_index=True).sort_values(["model", "side", "date"])
    merged.to_csv(target, index=False)
    print(
        f"[merge] {target}: replaced {int((~keep).sum())} stale rows with "
        f"{len(new)} fresh ({','.join(models)}) → {len(merged)} total"
    )


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", help=f"range start (default {DEFAULT_START})")
    ap.add_argument("--end", help=f"range end (default {DEFAULT_END})")
    ap.add_argument("--dates", nargs="*", help="explicit trade dates YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=4, help="Phase-B eval workers (SANOS ~GB each)")
    ap.add_argument("--build-workers", type=int, default=3, help="Phase-A parallel cache builders")
    ap.add_argument("--skip-build", action="store_true", help="assume all caches present")
    ap.add_argument("--rebuild-cache", action="store_true", help="rebuild caches even if present")
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "best_params_test.csv"))
    ap.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"comma-separated subset to run (default all: {','.join(ALL_MODELS)}). "
        "e.g. --models indiv_kf to re-score only the single-strike KF.",
    )
    ap.add_argument(
        "--merge-into",
        default=None,
        help="splice the fresh rows into this existing CSV, replacing only the rows "
        "for the run's (models × dates) and keeping everything else (e.g. the other "
        "models' rows from a prior full run).",
    )
    # hidden: build ONE date's SANOS parquet in an isolated subprocess
    ap.add_argument("--build-sanos", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.build_sanos:  # subprocess entry point — do nothing else
        _build_sanos_inproc(args.build_sanos, args.rebuild_cache)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; choose from {ALL_MODELS}")

    dates = _resolve_dates(args)
    if not dates:
        raise SystemExit("no trading days resolved")
    print(f"Test days ({len(dates)}): {dates[0]} .. {dates[-1]}  | models: {','.join(models)}")

    # SANOS is the only model needing its parquet cache; phase-1 covers indiv+ssvi.
    if not args.skip_build:
        _phase_a_build(dates, args.build_workers, args.rebuild_cache)
    else:
        print("[Phase A] skipped (--skip-build)")

    _phase_b_eval(dates, args.workers, args.out, models)

    if args.merge_into:
        _merge_into(args.merge_into, args.out, models, dates)


if __name__ == "__main__":
    main()
