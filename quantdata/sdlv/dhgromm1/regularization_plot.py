"""Read-only comparison of original and warm-started action-penalty phases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

UNPENALIZED, PENALIZED = "#864a9e", "#d88424"
METRICS = ("long_minus_empty", "short_minus_empty", "empty")
TITLES = ("Long − Empty", "Short − Empty", "Empty")


def history_curves(rows: list[dict], *, penalized: bool) -> dict[str, np.ndarray]:
    """Raw OCE and the optimized objective must not be conflated.

    Differences use the SAME epoch's Empty policy, not an independently
    selected best Empty from another date. Historical means have no MC bands.
    """
    epoch = np.array([int(row["epoch"]) for row in rows])
    if not len(epoch) or not np.array_equal(epoch, np.arange(len(epoch))):
        raise ValueError("expected contiguous history from phase epoch zero")
    labels = ("empty", "long", "short")
    loss = np.array([[float(r[f"{k}_loss"]) for k in labels] for r in rows])
    total = np.array([float(r["sum_loss"]) for r in rows])
    raw = (np.array([[float(r[f"{k}_raw_oce"]) for k in labels] for r in rows])
           if penalized else -loss)
    if not np.isfinite(raw).all() or not np.isfinite(loss).all() or not np.allclose(loss.sum(1), total):
        raise ValueError("nonfinite or inconsistent history objectives")
    if penalized:
        penalty = np.array([[float(r[f"{k}_regularizer"]) for k in labels] for r in rows])
        if not np.isfinite(penalty).all() or np.any(penalty < 0) or not np.allclose(-raw+penalty, loss, atol=1e-8):
            raise ValueError("raw OCE minus the outside-utility penalty does not match the objective")
    return dict(epoch=epoch, empty=raw[:,0], long_minus_empty=raw[:,1]-raw[:,0],
        short_minus_empty=raw[:,2]-raw[:,0], raw_sum=raw.sum(1), objective_sum=-total)


def snapshot_history(run: Path) -> dict:
    """One immutable read of an atomically replaced live CSV; never lock a trainer."""
    raw = (run/"triple-losses.csv").read_bytes()
    rows = [{k:int(v) if k=="epoch" else float(v) for k,v in r.items()}
            for r in csv.DictReader(io.StringIO(raw.decode()))]
    return dict(run=str(run), csv_sha256=hashlib.sha256(raw).hexdigest(), rows=rows,
                identity=json.loads((run/"identity.json").read_text()))


def sequence_snapshot(sequence: Path) -> dict:
    plan = json.loads((sequence/"plan.json").read_text())
    cases = []
    for index, name in enumerate(plan["sources"]):
        source = Path(name)
        baseline = snapshot_history(source)
        identity = baseline["identity"]
        policy = identity["model_config"]["policy"]
        if identity["objective"] != "oce" or policy["trade_l1"] != 0 or policy["trade_l2"] != 0:
            raise ValueError("comparison requires the corrected zero-cost OCE baseline")
        history_curves(baseline["rows"], penalized=False)
        folder = sequence/f"{index+1:02d}-{source.name}"
        penalized = None
        if (folder/"triple-losses.csv").exists():
            penalized = snapshot_history(folder)
            reg_identity = penalized["identity"]
            if (reg_identity["objective"] != "oce_minus_action_regularizer"
                    or reg_identity["transaction_costs"] != 0
                    or Path(reg_identity["source_run"]) != source
                    or reg_identity["action_regularizer"]["treatment"] != "outside_utility"):
                raise ValueError("incompatible source or regularizer semantics")
            history_curves(penalized["rows"], penalized=True)
        reports = {}
        for key, filename in (("initial", "initial-evaluation.json"), ("final", "report.json")):
            path = folder/filename
            if path.exists():
                reports[key] = json.loads(path.read_text())
        measure = identity.get("measure", "Q")
        has_a10 = not ("NO a10" in identity["policy_kind"])
        cases.append(dict(label=f"{measure} {'with' if has_a10 else 'without'} a₁₀",
            measure=measure, baseline=baseline, penalized=penalized, reports=reports))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), sequence=str(sequence), plan=plan, cases=cases,
        interpretation="Successive phases, NOT same-initialization parallel controls. Phase zero loads independently selected "
            "source-best weights with fresh Adam; its values can differ from the last baseline epoch.",
        uncertainty="Bars: available independently selected policies +/-3 MC SE, including Q-centering uncertainty, "
            "excluding fitting/selection uncertainty. No per-epoch MC errors were recorded.")


def report_point(report: dict, metric: str) -> dict:
    point = report["statistics"]["empty"]["raw_oce"] if metric=="empty" else report[f"raw_{metric}"]
    if not np.isfinite([point["oce"], point["three_standard_errors"]]).all() or point["three_standard_errors"] < 0:
        raise ValueError("invalid selected-policy uncertainty")
    return point


def draw_curve(axis, x: np.ndarray, y: np.ndarray, color: str) -> None:
    axis.plot(x, y, color=color, alpha=.28, linewidth=.65)
    width = min(20, len(y))
    axis.plot(x[width-1:], np.convolve(y, np.ones(width)/width, mode="valid"), color=color, linewidth=1.7)


def plot_raw(cases: list[dict], *, output: Path, measure: str) -> Path:
    """Long/short adjusted by Empty; keep the actual trading utility unpenalized."""
    fig, axes = plt.subplots(len(cases), 3, figsize=(14.5, 3.35*len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        original = history_curves(case["baseline"]["rows"], penalized=False)
        regularized = history_curves(case["penalized"]["rows"], penalized=True) if case["penalized"] else None
        for col, (metric, title) in enumerate(zip(METRICS, TITLES, strict=True)):
            axis = axes[row,col]
            draw_curve(axis, original["epoch"]-original["epoch"][-1], original[metric], UNPENALIZED)
            if regularized is not None:
                draw_curve(axis, regularized["epoch"], regularized[metric], PENALIZED)
            else:
                axis.text(.72, .53, "Penalized run\nqueued", transform=axis.transAxes, ha="center", color="#777777")
            for key, x, color in (("initial", 0, UNPENALIZED),
                                  ("final", regularized["epoch"][-1]+10 if regularized is not None else 0, PENALIZED)):
                if key in case["reports"]:
                    point = report_point(case["reports"][key], metric)
                    axis.errorbar(x, point["oce"], yerr=point["three_standard_errors"], fmt="o", color=color,
                        markerfacecolor="white", markersize=4, capsize=3, zorder=5)
            axis.axvline(0, color="#666666", linewidth=.8, linestyle="--")
            axis.axhline(0, color="#666666", linewidth=.5)
            axis.set(xlim=(-510, 530), yscale="symlog", title=f"{case['label']} · {title}",
                     xlabel="Phase epoch: baseline ← 0 → penalized", ylabel="Raw OCE — higher is better")
            axis.set_yscale("symlog", linthresh=1e-4 if measure=="Q" else 1e-3)
            axis.grid(alpha=.18)
        phase = (f"penalized {regularized['epoch'][-1]}/500" if regularized is not None else "penalized queued")
        axes[row,0].text(.02, .95, f"Baseline {original['epoch'][-1]} recorded epochs; {phase}",
                         transform=axes[row,0].transAxes, fontsize=8, va="top",
                         bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    handles = [Line2D([0], [0], color=c, linewidth=2, label=label) for c,label in (
        (UNPENALIZED, "Unpenalized training phase"), (PENALIZED, "Penalized warm-start phase"))]
    handles.append(Line2D([0], [0], color="#555555", marker="o", markerfacecolor="white", linestyle="none",
                          label="Selected policy ±3 MC SE (where evaluated)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.075), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"GROMM v1 (−4,3), {measure}, one-step/day · training OCE with and without action regularization", fontsize=12)
    fig.text(.5,.025,"Faint: epoch means; thick: 20-epoch mean; symlog y scale. No historical MC bands. Zero trading costs in both phases.\n"
        "At 0: load selected baseline weights, reset Adam, add L1=10⁻⁴ and L2=10⁻⁸ outside utility. Not an unpenalized continuation control.",
        ha="center",fontsize=8)
    fig.subplots_adjust(top=.9,bottom=.19,left=.075,right=.985,hspace=.55,wspace=.32)
    target = output/f"{measure.lower()}-raw-oce-comparison.png"
    fig.savefig(target,dpi=150)
    plt.close(fig)
    return target


def plot_objectives(cases: list[dict], output: Path) -> Path:
    """Actual optimized sums, with raw sum OCE during penalized training alongside."""
    fig, axes = plt.subplots(2,2,figsize=(13,8))
    for case, axis in zip(cases, axes.flat, strict=True):
        original = history_curves(case["baseline"]["rows"],penalized=False)
        draw_curve(axis,original["epoch"]-original["epoch"][-1],original["objective_sum"],UNPENALIZED)
        if case["penalized"]:
            penalized = history_curves(case["penalized"]["rows"],penalized=True)
            draw_curve(axis,penalized["epoch"],penalized["objective_sum"],PENALIZED)
            axis.plot(penalized["epoch"],penalized["raw_sum"],color="#c33b43",linewidth=1,
                      linestyle=":",label="Raw sum OCE during penalized phase")
            axis.legend(fontsize=8)
        else:
            axis.text(.7,.5,"Penalized run queued",transform=axis.transAxes,ha="center",color="#777777")
        axis.axvline(0,color="#666666",linewidth=.8,linestyle="--")
        axis.set(xlim=(-510,530),title=case["label"],xlabel="Phase epoch: baseline ← 0 → penalized",
                 ylabel="Sum of three objectives — higher is better")
        axis.set_yscale("symlog",linthresh=1e-3)
        axis.grid(alpha=.2)
    fig.suptitle("Optimized objective: purple = ΣU; orange = Σ(U − L1/L2 penalty)",fontsize=12)
    fig.text(.5,.02,"Training means, no historical MC bands. Different objective phases, same paths and zero trading costs.\n"
        "Penalties are outside utility; dotted red deliberately omits them. Phase zero uses selected weights and fresh Adam.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    target = output/"optimized-objective-comparison.png"
    fig.savefig(target,dpi=150)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=Path)
    group.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("use a new external project output directory")
    snapshot = sequence_snapshot(args.sequence) if args.sequence else json.loads(args.snapshot.read_text())
    output.mkdir(parents=True,exist_ok=False)
    (output/"snapshot.json").write_text(json.dumps(snapshot,indent=2,allow_nan=False)+"\n")
    files = [plot_raw([c for c in snapshot["cases"] if c["measure"]==measure],output=output,measure=measure)
             for measure in ("Q","P")]
    files.append(plot_objectives(snapshot["cases"],output))
    print(json.dumps(dict(figures=[str(p) for p in files],snapshot=str(output/"snapshot.json")),indent=2),flush=True)


if __name__ == "__main__":
    main()
