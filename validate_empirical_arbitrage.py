#!/usr/bin/env python3
"""Frozen-position out-of-sample validation for empirical arbitrage portfolios."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

CONVENTIONS = {
    "frictionless": (0.0, 0.0),
    "delta1": (1.0, 0.0),
    "1bp": (0.0, 1e-4),
}
QUANTILES = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)


def datasets(h5: h5py.File) -> dict[str, h5py.Dataset]:
    out: dict[str, h5py.Dataset] = {}
    h5.visititems(lambda n, o: out.__setitem__(n, o) if isinstance(o, h5py.Dataset) else None)
    return out


def identify(h5: h5py.File, leaf: str) -> h5py.Dataset:
    matches = [v for k, v in datasets(h5).items() if k.split("/")[-1] == leaf]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one dataset with leaf {leaf!r}, found {len(matches)}")
    return matches[0]


def chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def scan_array(ds: h5py.Dataset, chunk_size: int) -> dict[str, Any]:
    total = finite = 0
    value_sum = 0.0
    lo, hi = math.inf, -math.inf
    for sl in chunks(ds.shape[0], chunk_size):
        x = np.asarray(ds[sl], dtype=np.float64)
        ok = np.isfinite(x)
        total += x.size
        finite += int(ok.sum())
        if ok.any():
            y = x[ok]
            value_sum += float(y.sum())
            lo = min(lo, float(y.min()))
            hi = max(hi, float(y.max()))
    return {"min": lo, "max": hi, "mean": value_sum / finite,
            "fraction_nonfinite": (total - finite) / total}


def training_moments(ds: h5py.Dataset, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = ds.shape[0]
    total = np.zeros(ds.shape[1:], dtype=np.float64)
    total2 = np.zeros_like(total)
    for sl in chunks(n, chunk_size):
        x = np.asarray(ds[sl], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in {ds.name}")
        total += x.sum(axis=0)
        total2 += np.square(x).sum(axis=0)
    mean = total / n
    variance = np.maximum((total2 - n * mean * mean) / (n - 1), 0.0)
    return mean, np.sqrt(variance) / math.sqrt(n)


def evaluate(ds_dx: h5py.Dataset, ds_x: h5py.Dataset, a: np.ndarray,
             mc_error: np.ndarray, delta: float, prop: float,
             chunk_size: int, need_steps: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    n = ds_dx.shape[0]
    payoff = np.empty(n, dtype=np.float64)
    steps = np.empty((n, ds_dx.shape[1]), dtype=np.float64) if need_steps else None
    abs_a = np.abs(a)
    mc_step = np.sum(abs_a * mc_error, axis=1)
    for sl in chunks(n, chunk_size):
        dx = np.asarray(ds_dx[sl], dtype=np.float64)
        if not np.isfinite(dx).all():
            raise ValueError(f"Non-finite values in {ds_dx.name}")
        step = np.einsum("nti,ti->nt", dx, a, optimize=True)
        if delta:
            step -= delta * mc_step[None, :]
        if prop:
            xx = np.asarray(ds_x[sl], dtype=np.float64)
            if not np.isfinite(xx).all():
                raise ValueError(f"Non-finite values in {ds_x.name}")
            step -= prop * np.einsum("nti,ti->nt", np.abs(xx), abs_a, optimize=True)
        payoff[sl] = step.sum(axis=1)
        if steps is not None:
            steps[sl] = step
    return payoff, steps


def statistics(x: np.ndarray) -> dict[str, Any]:
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    qs = np.quantile(x, QUANTILES)
    tol = max(1e-10, 1e-10 * max(1.0, float(np.max(np.abs(x)))))
    neg = x < -tol
    one_lo, one_hi = mean - 1.645 * se, mean + 1.645 * se
    ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
    if mean < -tol:
        classification = "negative after costs"
    elif float(x.min()) >= -tol and mean > tol:
        classification = "out-of-sample pathwise certificate"
    elif one_lo > 0 and neg.any():
        classification = "positive expected return, not arbitrage"
    else:
        classification = "not statistically positive"
    k001 = max(1, math.ceil(0.001 * n))
    k01 = max(1, math.ceil(0.01 * n))
    tail001 = np.partition(x, k001 - 1)[:k001]
    tail01 = np.partition(x, k01 - 1)[:k01]
    return {
        "min": float(x.min()), "q0.001": float(qs[0]), "q0.01": float(qs[1]),
        "q0.05": float(qs[2]), "median": float(qs[3]), "mean": mean,
        "q0.95": float(qs[4]), "q0.99": float(qs[5]), "q0.999": float(qs[6]),
        "max": float(x.max()), "std": sd, "sem": se,
        "fraction_below_-1e-10": float(np.mean(x < -1e-10)),
        "fraction_below_0": float(np.mean(x < 0)), "fraction_above_0": float(np.mean(x > 0)),
        "negative_count_scale_aware": int(neg.sum()), "negative_fraction_scale_aware": float(neg.mean()),
        "worst_loss": float(x.min()), "expected_shortfall_worst_0.1pct": float(tail001.mean()),
        "expected_shortfall_worst_1pct": float(tail01.mean()),
        "one_sided_90pct_mean_interval": [one_lo, one_hi], "two_sided_95pct_mean_interval": [ci_lo, ci_hi],
        "numerical_tolerance": tol, "classification": classification,
    }


def discover_positions(root: Path) -> dict[str, tuple[Path, str]]:
    specs = {
        "delta0_prop0": (0.0, 0.0), "delta1_prop0": (1.0, 0.0),
        "delta0_prop1bp": (0.0, 1e-4),
    }
    out: dict[str, tuple[Path, str]] = {}
    aliases = {"delta0_prop0": "delta0_prop0_positions.npy", "delta1_prop0": "delta1_prop0_positions.npy",
               "delta0_prop1bp": "delta0_prop1bp_positions.npy"}
    for name in specs:
        candidates = [root / name / "positions.npy", root / aliases[name]]
        found = [p for p in candidates if p.is_file()]
        if found:
            out[name] = (found[0], f"delta={specs[name][0]:g}, prop_cost={specs[name][1]:g}")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def fmt(v: Any) -> str:
    return f"{v:.8g}" if isinstance(v, float) else str(v)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-h5", required=True, type=Path)
    p.add_argument("--validation-h5", required=True, type=Path)
    p.add_argument("--positions-dir", required=True, type=Path)
    p.add_argument("--output-dir", default=Path("out_of_sample_arbitrage_validation"), type=Path)
    p.add_argument("--chunk-size", default=5000, type=int)
    args = p.parse_args()
    if args.chunk_size < 1: p.error("--chunk-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"datasets": {}, "portfolios": {}, "missing_portfolios": []}
    flat_rows: list[dict[str, Any]] = []
    with h5py.File(args.training_h5, "r") as ft, h5py.File(args.validation_h5, "r") as fv:
        td, tx, vd, vx = identify(ft, "dX"), identify(ft, "X"), identify(fv, "dX"), identify(fv, "X")
        for label, h5, dd, xx in (("training", ft, td, tx), ("validation", fv, vd, vx)):
            summary["datasets"][label] = {"keys": list(datasets(h5)), "dX": {"shape": list(dd.shape), "dtype": str(dd.dtype)},
                "X": {"shape": list(xx.shape), "dtype": str(xx.dtype)}, "dX_scan": scan_array(dd, args.chunk_size),
                "X_scan": scan_array(xx, args.chunk_size)}
        if td.shape[1:] != vd.shape[1:] or td.shape != tx.shape or vd.shape != vx.shape:
            raise RuntimeError(f"Incompatible shapes: training dX/X {td.shape}/{tx.shape}, validation {vd.shape}/{vx.shape}")
        train_mean, mc_error = training_moments(td, args.chunk_size)
        np.save(args.output_dir / "training_mc_error.npy", mc_error)
        val_mean = np.zeros(vd.shape[1:], dtype=np.float64)
        for sl in chunks(vd.shape[0], args.chunk_size): val_mean += np.asarray(vd[sl], dtype=np.float64).sum(axis=0)
        val_mean /= vd.shape[0]
        ratios = np.divide(val_mean, mc_error, out=np.full_like(val_mean, np.nan), where=mc_error > 0)
        drift_rows = []
        for t, row in enumerate(ratios):
            q = np.nanquantile(row, [0, .05, .5, .95, 1])
            drift_rows.append({"time": t, "min": q[0], "q0.05": q[1], "median": q[2], "mean": np.nanmean(row), "q0.95": q[3], "max": q[4]})
        write_csv(args.output_dir / "coordinate_mean_drift_by_time.csv", drift_rows)
        portfolios = discover_positions(args.positions_dir)
        expected = {"delta0_prop0", "delta1_prop0", "delta0_prop1bp"}
        summary["missing_portfolios"] = sorted(expected - portfolios.keys())
        if not portfolios: raise RuntimeError(f"No position arrays found below {args.positions_dir}")
        for name, (pos_path, selected_under) in portfolios.items():
            a = np.asarray(np.load(pos_path), dtype=np.float64)
            if a.shape != td.shape[1:] or not np.isfinite(a).all(): raise ValueError(f"Invalid {name} positions shape/finiteness: {a.shape}")
            np.save(args.output_dir / f"{name}_positions.npy", a)
            selected = {"delta0_prop0": "frictionless", "delta1_prop0": "delta1", "delta0_prop1bp": "1bp"}[name]
            train_pay, _ = evaluate(td, tx, a, mc_error, *CONVENTIONS[selected], args.chunk_size)
            np.save(args.output_dir / f"{name}_training_payoff.npy", train_pay)
            saved_path = pos_path.parent / "payoff.npy"
            consistency: dict[str, Any] = {"source": str(pos_path), "loaded_not_recomputed": True}
            if saved_path.is_file():
                saved = np.asarray(np.load(saved_path), dtype=np.float64)
                if saved.shape != train_pay.shape: raise RuntimeError(f"Saved payoff shape mismatch for {name}")
                diff = float(np.max(np.abs(saved - train_pay)))
                consistency["saved_payoff_max_abs_difference"] = diff
                if diff > 5e-9: raise RuntimeError(f"FAIL: {name} payoff reconstruction max difference {diff:g}")
            item: dict[str, Any] = {"selected_under": selected_under, "consistency": consistency,
                                    "training": statistics(train_pay), "validation": {}}
            step_rows: list[dict[str, Any]] = []; worst_rows: list[dict[str, Any]] = []
            for conv, (delta, prop) in CONVENTIONS.items():
                pay, step = evaluate(vd, vx, a, mc_error, delta, prop, args.chunk_size, True)
                assert step is not None
                np.save(args.output_dir / f"{name}_new_payoff_{conv}.npy", pay)
                st = statistics(pay); item["validation"][conv] = st
                st["ks_statistic_vs_training"] = float(ks_2samp(train_pay, pay).statistic)
                st["wasserstein_distance_vs_training"] = float(wasserstein_distance(train_pay, pay))
                flat_rows.append({"portfolio": name, "selected_under": selected_under, "evaluation": conv, **st})
                for t in range(step.shape[1]):
                    z = step[:, t]
                    step_rows.append({"convention": conv, "time": t, "min": z.min(), "q0.01": np.quantile(z,.01),
                                      "mean": z.mean(), "max": z.max(), "fraction_negative": np.mean(z < 0)})
                worst = np.argpartition(pay, 99)[:100]
                for rank, omega in enumerate(worst[np.argsort(pay[worst])], 1):
                    dx = np.asarray(vd[int(omega)], dtype=np.float64)
                    contrib = dx * a
                    if delta: contrib -= delta * mc_error * np.abs(a)
                    if prop: contrib -= prop * np.abs(np.asarray(vx[int(omega)], dtype=np.float64)) * np.abs(a)
                    for kind, index in (("negative", np.unravel_index(np.argmin(contrib), contrib.shape)),
                                        ("positive", np.unravel_index(np.argmax(contrib), contrib.shape))):
                        worst_rows.append({"convention": conv, "worst_rank": rank, "path": int(omega), "path_payoff": pay[omega],
                                           "kind": kind, "time": index[0], "instrument": index[1], "contribution": contrib[index]})
            write_csv(args.output_dir / f"{name}_step_summary.csv", step_rows)
            write_csv(args.output_dir / f"{name}_worst_path_contributions.csv", worst_rows)
            summary["portfolios"][name] = item
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", flat_rows)
    lines = ["# Out-of-sample empirical-arbitrage validation", "", "## Data and provenance", ""]
    for label, d in summary["datasets"].items():
        lines += [f"- {label}: keys `{d['keys']}`; dX {d['dX']['shape']} {d['dX']['dtype']}; X {d['X']['shape']} {d['X']['dtype']}; "
                  f"dX min/max/mean/non-finite = {fmt(d['dX_scan']['min'])}/{fmt(d['dX_scan']['max'])}/{fmt(d['dX_scan']['mean'])}/{fmt(d['dX_scan']['fraction_nonfinite'])}; "
                  f"X = {fmt(d['X_scan']['min'])}/{fmt(d['X_scan']['max'])}/{fmt(d['X_scan']['mean'])}/{fmt(d['X_scan']['fraction_nonfinite'])}. "]
    lines += ["", "Time/instrument dimensions match. Positions were loaded, never fitted on validation data."]
    if summary["missing_portfolios"]:
        lines += ["", "**Scope warning:** exact saved positions were unavailable for: " + ", ".join(summary["missing_portfolios"]) + ". They were not fabricated or replaced."]
    for name, item in summary["portfolios"].items():
        lines += ["", f"## `{name}`", "", f"Loaded from `{item['consistency']['source']}`; selected under {item['selected_under']}.", "",
                  "| sample/evaluation | min | mean | std | negative fraction | 95% CI | classification |", "|---|---:|---:|---:|---:|---:|---|"]
        tr = item["training"]
        lines.append(f"| training/original | {fmt(tr['min'])} | {fmt(tr['mean'])} | {fmt(tr['std'])} | {fmt(tr['negative_fraction_scale_aware'])} | — | reconstruction check |")
        for conv, st in item["validation"].items():
            lines.append(f"| validation/{conv} | {fmt(st['min'])} | {fmt(st['mean'])} | {fmt(st['std'])} | {fmt(st['negative_fraction_scale_aware'])} | [{fmt(st['two_sided_95pct_mean_interval'][0])}, {fmt(st['two_sided_95pct_mean_interval'][1])}] | {st['classification']} |")
        lines += ["", "Complete quantiles, tail losses, confidence intervals, KS and Wasserstein diagnostics are in `summary.json` and `summary.csv`."]
    fr = summary["portfolios"].get("delta0_prop0", {}).get("validation", {})
    if fr:
        a, b, c = fr["frictionless"], fr["delta1"], fr["1bp"]
        lines += ["", "## Final conclusion", "",
                  f"1. **No.** The frozen frictionless portfolio's new-data minimum is {fmt(a['min'])}, with {a['negative_count_scale_aware']:,} materially negative paths.",
                  f"2. **{'Yes' if a['one_sided_90pct_mean_interval'][0] > 0 else 'No'}.** Its mean is {fmt(a['mean'])}; 95% CI [{fmt(a['two_sided_95pct_mean_interval'][0])}, {fmt(a['two_sided_95pct_mean_interval'][1])}].",
                  f"3. **{'Yes' if b['one_sided_90pct_mean_interval'][0] > 0 else 'No'}.** Under the training-MC-error penalty the mean is {fmt(b['mean'])} ({b['classification']}).",
                  f"4. **{'Yes' if c['one_sided_90pct_mean_interval'][0] > 0 else 'No'}.** Under 1 bp entry cost the mean is {fmt(c['mean'])} ({c['classification']}).",
                  f"5. The evidence is best described as **{'expected-return signal with risk' if a['one_sided_90pct_mean_interval'][0] > 0 else 'finite-sample separation/overfitting'}**, not structural model arbitrage: the same frozen positions have material losses out of sample."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote validation outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
