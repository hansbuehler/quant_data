"""Plot epoch OCEs and convergence diagnostics without touching a trainer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("long_minus_empty", "short_minus_empty", "empty")
TITLES = ("Long − Empty", "Short − Empty", "Empty (unadjusted)")
COLORS = ("#864a9e", "#d88424", "#c33b43")


def transform_history(rows: list[dict]) -> dict[str, np.ndarray]:
    """Convert negative-OCE epoch scores, subtracting the SAME-epoch Empty.

    These curves are not independently selected best-policy valuations. Nor
    are variations across epochs a Monte Carlo standard-error estimate.
    """
    epoch = np.array([r["epoch"] for r in rows], dtype=int)
    if not len(epoch) or not np.array_equal(epoch, np.arange(len(epoch))):
        raise ValueError("expected contiguous epoch history beginning at zero")
    loss = np.array([[r[f"{k}_loss"] for k in ("empty", "long", "short")] for r in rows])
    total = np.array([r["sum_loss"] for r in rows])
    if not np.isfinite(loss).all() or not np.isfinite(total).all() or not np.allclose(
            loss.sum(1), total, atol=1e-12, rtol=1e-10):
        raise ValueError("nonfinite or inconsistent summed epoch objective")
    return dict(epoch=epoch, empty=-loss[:, 0], long_minus_empty=loss[:, 0]-loss[:, 1],
        short_minus_empty=loss[:, 0]-loss[:, 2], sum_oce=-total,
        raw_long=-loss[:, 1], raw_short=-loss[:, 2])


def tail_progress(curves: dict[str, np.ndarray], *, window: int = 100) -> dict:
    """Descriptive running-best progress, not a convergence test or MC interval."""
    if window < 1:
        raise ValueError("window must be positive")
    epoch = curves["epoch"]
    cut = max(0, int(epoch[-1])-window)
    report = {}
    for key in ("empty", "raw_long", "raw_short", "sum_oce"):
        value = curves[key]
        before, best = float(value[epoch <= cut].max()), float(value.max())
        report[key] = dict(best_epoch=int(epoch[value.argmax()]), best_before_tail=before,
            best=best, improvement_in_tail=best-before)
    return dict(through_epoch=int(epoch[-1]), tail_start_epoch=cut, metrics=report,
        interpretation="epoch cap is not convergence; inspect continued progress and plateau stability")


def read_case(record: dict) -> dict:
    run = Path(record["run"])
    identity = json.loads((run/"identity.json").read_text())
    if identity.get("objective") != "oce" or identity.get("cost_treatment") != "inside_utility":
        raise ValueError("this figure requires the corrected OCE objective, costs inside utility")
    if "history" in record:
        rows = record["history"]
    else:
        # One immutable read of an atomically replaced live CSV.
        with (run/"triple-losses.csv").open(newline="") as stream:
            rows = [{k:int(v) if k=="epoch" else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    rows = [r for r in rows if r["epoch"] <= record["checkpoint_epoch"]]
    curves = transform_history(rows)
    if curves["epoch"][-1] != record["checkpoint_epoch"]:
        raise ValueError("history and uncertainty snapshot have different epoch endpoints")
    for metric in METRICS:
        point = record["statistics"][metric]
        if not np.isfinite([point["oce"], point["three_standard_errors"]]).all() or point["three_standard_errors"] < 0:
            raise ValueError("invalid selected-checkpoint MC interval")
    return dict(record=record, identity=identity, curves=curves, progress=tail_progress(curves))


def rolling_mean(values: np.ndarray, width: int = 20) -> tuple[np.ndarray, int]:
    width = min(width, len(values))
    return np.convolve(values, np.ones(width)/width, mode="valid"), width


def plot_cases(cases: list[dict], output: Path, *, tail: int | None = None, pending: tuple[str, ...] = ()) -> Path:
    """Plot means, a labelled rolling mean, and selected-checkpoint +/-3SE bars."""
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 3.45*len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        record, identity, curves = case["record"], case["identity"], case["curves"]
        epoch = curves["epoch"]
        first = max(0, len(epoch)-tail) if tail else 0
        for col, (metric, title, color) in enumerate(zip(METRICS, TITLES, COLORS, strict=True)):
            axis = axes[row, col]
            value = curves[metric]
            axis.plot(epoch[first:], value[first:], color=color, alpha=.35, linewidth=.8, label="Epoch mean (no historical MC band)")
            smooth, width = rolling_mean(value)
            x = epoch[width-1:]
            mask = x >= epoch[first]
            axis.plot(x[mask], smooth[mask], color=color, linewidth=1.7, label="20-epoch rolling mean")
            point = record["statistics"][metric]
            marker_x = epoch[-1] + max(1., .025*(epoch[-1]-epoch[first]))
            axis.errorbar([marker_x], [point["oce"]], yerr=[point["three_standard_errors"]], fmt="D",
                color="black", markersize=4, capsize=4, elinewidth=1.2, label="Selected-policy snapshot ±3 MC SE")
            axis.axhline(0, color="#505050", linewidth=.6, linestyle="--")
            if tail is None:
                axis.set_yscale("symlog", linthresh=1e-4)
            clock = "Zero-cost phase epoch (0 = cost switch)" if identity.get("continuation") else "Epoch"
            axis.set(title=f"{record['label']} · {title}", xlabel=clock,
                ylabel="OCE — higher is better" + (" (symlog)" if tail is None else ""))
            axis.grid(alpha=.18)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    fig.suptitle("GROMM v1 (−4,3), one-step/day: training OCE" + (f" — last {tail} epochs" if tail else " — full recorded history"), fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .025, "Curves: same-epoch means. Black markers: independently selected best-policy snapshots, not points on the same-epoch curve.\n"
        "Bars include estimated Q-centering uncertainty, exclude fitting/selection uncertainty. " + " ".join(pending),
        ha="center", fontsize=8)
    fig.subplots_adjust(top=.925, bottom=.15, left=.07, right=.985, hspace=.55, wspace=.34)
    target = output/("oce-tail.png" if tail else "oce-history.png")
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_objective(cases: list[dict], output: Path) -> Path:
    """Show the actual optimized sum of raw OCEs, distinct from price differences."""
    fig, axes = plt.subplots(1, len(cases), figsize=(15, 4.2), squeeze=False)
    for i, (axis, case) in enumerate(zip(axes[0], cases, strict=True)):
        color = COLORS[i % len(COLORS)]
        epoch, value = case["curves"]["epoch"], case["curves"]["sum_oce"]
        first = max(0, len(epoch)-100)
        axis.plot(epoch[first:], value[first:], color=color, alpha=.35, linewidth=.8, label="Epoch sum OCE")
        axis.plot(epoch[first:], np.maximum.accumulate(value)[first:], color=color, linewidth=1.6, label="Best epoch sum so far")
        axis.set(title=case["record"]["label"], xlabel="Zero-cost phase epoch" if case["identity"].get("continuation") else "Epoch",
            ylabel="Sum of three raw OCEs — higher is better")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Actual optimized objective: last 100 epochs (not baseline-subtracted prices)", fontsize=12)
    fig.text(.5, .095, "Historical epoch MC errors were not recorded; curves are training means only. No convergence is inferred from reaching the epoch cap.", ha="center", fontsize=8)
    fig.subplots_adjust(top=.82, bottom=.27, left=.065, right=.985, wspace=.38)
    target = output/"objective-tail.png"
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not args.output.resolve().is_relative_to(root) or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("use a new external project output directory")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(args.snapshot.read_text())
    cases = [read_case(r) for r in snapshot["records"]]
    pending = tuple(snapshot.get("pending", ()))
    files = [plot_cases(cases, args.output, pending=pending), plot_cases(cases, args.output, tail=100, pending=pending), plot_objective(cases, args.output)]
    summary = dict(snapshot=str(args.snapshot), snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        convergence={c["record"]["label"]:c["progress"] for c in cases}, figures=[str(p) for p in files],
        uncertainty="Historical mean curves have no logged MC bands; checkpoint markers show ±3 MC standard errors")
    (args.output/"plot-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
