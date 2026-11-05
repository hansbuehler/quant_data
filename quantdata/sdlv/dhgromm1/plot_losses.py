"""Plot completed or live training histories without changing the hedge run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("QUANTDATA_ROOT", "/tmp")) / "tmp/quant_data/matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def joint_history(run: Path) -> list[dict]:
    """Read one atomically published joint history, including its warm start."""
    with (run / "joint-losses.csv").open(newline="") as stream:
        rows = [{key: int(value) if key == "epoch" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows or [row["epoch"] for row in rows] != list(range(len(rows))):
        raise ValueError("joint history must contain contiguous epochs starting at zero")
    for row in rows:
        values = [row[key] for key in ("long_loss", "short_loss", "sum_loss", "online_sum_loss")]
        if not np.isfinite(values).all() or not np.isclose(row["sum_loss"], sum(values[:2]), atol=1e-12, rtol=1e-10):
            raise ValueError(f"invalid joint loss history at epoch {row['epoch']}")
    return rows


def plot_joint_losses(run: Path, output: Path, *, partial: bool, log_y: bool = False) -> dict:
    """Plot epoch-end training losses without reading or changing checkpoints."""
    report_path = run / "report.json"
    if not partial and not report_path.exists():
        raise ValueError("completion report unavailable; use --partial for an active run")
    rows = joint_history(run)
    identity = json.loads((run / "identity.json").read_text())
    selected = min(rows, key=lambda row: row["sum_loss"])
    if log_y and any(row[key] <= 0 for row in rows for key in ("long_loss", "short_loss", "sum_loss")):
        raise ValueError("logarithmic loss axes require strictly positive losses")
    policy_inputs = ("observable-only" if "simulator latent state" in identity.get("excluded_inputs", [])
                     else "full-state")
    summary = dict(run=str(run), epochs_run=rows[-1]["epoch"], target_epochs=identity["joint_epochs"],
        selected_epoch=selected["epoch"], initial=rows[0], latest=rows[-1], selected=selected,
        statistic="full training sample, frozen epoch-end policy; negative OCE plus trade regularizer",
        selection="minimum training SUM, same epoch for both independent networks", validation=False,
        policy_inputs=policy_inputs, y_scale="log" if log_y else "linear")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("losses.png", "epoch_losses.csv", "summary.json"):
        if (output / name).exists():
            raise FileExistsError(f"output already exists: {output / name}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, title, color) in zip(axes, (
        ("long_loss", "Bid / long claim", "#864a9e"),
        ("short_loss", "Ask / short claim", "#d88424"),
        ("sum_loss", "Joint objective: sum", "#c33b43"),
    )):
        ax.plot([row["epoch"] for row in rows], [row[key] for row in rows],
                color=color, marker=".", markersize=4, linewidth=1.4, label="Full-set epoch-end training loss")
        ax.axhline(rows[0][key], color="#303030", linestyle="--", linewidth=1, label="Migrated warm start")
        ax.scatter([selected["epoch"]], [selected[key]], marker="*", s=110, color=color,
                   edgecolors="black", linewidths=.6, zorder=5, label="Selected common checkpoint")
        ax.set(title=title, xlabel="Epoch (0 = warm start)", ylabel="Penalized negative OCE — lower is better")
        if log_y:
            ax.set_yscale("log")
            ax.set_ylabel("Training loss (log axis) — lower is better")
        ax.grid(alpha=.18)
    fig.suptitle(f"GROMM v1 (−4,3): {policy_inputs}, one-step/day deep hedge — training losses\n"
                 f"{run.name}; through epoch {rows[-1]['epoch']}/{identity['joint_epochs']}; "
                 f"selected epoch {selected['epoch']}", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.subplots_adjust(top=.79, bottom=.23, left=.065, right=.985, wspace=.30)
    fig.savefig(output / "losses.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    with (output / "epoch_losses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def training_histories(run: Path, *, ancestors: set[Path] | None = None) -> dict[str, list[dict]]:
    """Join an explicitly recorded continuation to its previous epoch history."""
    seen = set(ancestors or ())
    run = run.resolve()
    if run in seen:
        raise ValueError("cyclic continuation history")
    seen.add(run)
    pipeline_path = run / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
    if pipeline.get("resume_run"):
        records = training_histories(Path(pipeline["resume_run"]), ancestors=seen)
        for label, epoch in pipeline.get("resume_epochs", {}).items():
            # An interrupted process may have flushed its final epoch log just
            # before saving. The recovered atomic checkpoint sets the boundary.
            records[label] = [record for record in records[label] if record["epoch"] <= epoch]
    else:
        records = {name: [] for name in ("empty", "long", "short")}
    in_full_training = False
    for line in (run / "job.log").read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("stage") == "full-training":
            in_full_training = True
        if in_full_training and record.get("stage") == "train":
            history = records[record["label"]]
            if history and record["epoch"] <= history[-1]["epoch"]:
                raise ValueError("continuation repeats or rewinds recorded training epochs")
            history.append(record)
    return records


def initial_history_losses(run: Path) -> dict[str, float]:
    """Recover epoch-zero warm-start losses, not a later resume checkpoint loss."""
    pipeline = json.loads((run / "pipeline.json").read_text())
    initial = initial_history_losses(Path(pipeline["resume_run"])) if pipeline.get("resume_run") else {}
    in_full_training = False
    for line in (run / "job.log").read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("stage") == "full-training":
            in_full_training = True
        if in_full_training and record.get("stage") == "warm-start":
            initial.setdefault(record["label"], record["initial_training_loss"])
    return initial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true", help="wait for the active run to finish before plotting")
    parser.add_argument("--partial", action="store_true", help="plot the available epochs of an active run")
    parser.add_argument("--log-y", action="store_true", help="logarithmic axes for positive joint training losses")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not output.is_relative_to(Path(os.environ["QUANTDATA_ROOT"]).resolve()):
        raise ValueError("plot outputs must be inside the external data root")
    if (run / "joint-losses.csv").exists():
        while args.wait and not (run / "report.json").exists():
            print(json.dumps(dict(stage="waiting_for_joint_training_report")), flush=True)
            time.sleep(30)
        summary = plot_joint_losses(run, output, partial=args.partial, log_y=args.log_y)
        print(json.dumps(dict(output=str(output), summary=summary), indent=2))
        return
    # Historical independent-policy plots also compute the no-trade baseline.
    from quantdata.sdlv.dhgromm1.training import _economic_utility, optimize_translation

    while args.wait:
        pipeline = json.loads((run / "pipeline.json").read_text())
        if pipeline.get("status") == "failed":
            raise RuntimeError(f"training pipeline failed: {pipeline.get('error')}")
        if pipeline.get("status") == "complete":
            break
        print(json.dumps(dict(stage="waiting_for_training_report", pipeline_stage=pipeline.get("stage"))), flush=True)
        time.sleep(30)
    pipeline_path = run / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
    records = training_histories(run)
    report_path = run / "training/report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    if not any(records.values()):
        raise ValueError("no full-sample training history is available")
    if not args.partial and any(not history and not (report or {}).get("fits", {}).get(label, {}).get("frozen")
                                for label, history in records.items()):
        raise ValueError("each fitted policy needs a full-sample training history")
    if report is None and not args.partial:
        raise ValueError("evaluation report unavailable; use --partial for an active run")
    scenarios = Path(report["scenarios"] if report else pipeline.get("scenarios", run / "scenarios"))
    payoffs = []
    for path in sorted((scenarios / "train").glob("[0-9]*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            payoffs.append(saved["payoff"])
    payoff = np.concatenate(payoffs)
    risk = report["model_config"]["risk_aversion"] if report else 100.0
    baseline = {}
    initial_losses = initial_history_losses(run)
    warm_start = bool(initial_losses)
    for label, sign in (("empty", 0), ("long", 1), ("short", -1)):
        pnl = sign * payoff
        m = optimize_translation(pnl, risk)
        baseline[label] = -float(_economic_utility(pnl, m, risk).mean())
    output.mkdir(parents=True, exist_ok=True)
    for name in ("losses.png", "epoch_losses.csv", "summary.json"):
        if (output / name).exists():
            raise FileExistsError(f"output already exists: {output / name}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=False)
    summary = {}
    rows = []
    colors = {"empty": "#c33b43", "long": "#864a9e", "short": "#d88424"}
    for ax, (label, history) in zip(axes, records.items()):
        if not history:
            ax.axhline(baseline[label], color="#303030", linestyle=":", linewidth=1.2)
            frozen = (report or {}).get("fits", {}).get(label, {}).get("frozen") or label not in pipeline.get("trained_policies", records)
            ax.text(.5, .5, "Frozen — no training" if frozen else "Training has not started",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label.capitalize()} claim")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Penalized training loss — lower is better")
            ax.grid(alpha=.18)
            summary[label] = dict(epochs_run=0, frozen=bool(frozen), selected_epoch=0 if frozen else None,
                                  initial_no_trade_loss=baseline[label])
            continue
        epochs = [r["epoch"] for r in history]
        if epochs != list(range(1, len(history) + 1)):
            raise ValueError(f"noncontiguous or duplicated {label} history")
        best = history[-1]["best_training_loss"]
        chosen = next((r["epoch"] for r in history if r["training_loss"] == best), 0)
        x = [0, *epochs]
        initial = initial_losses.get(label, baseline[label]) if warm_start else baseline[label]
        y = [initial, *[r["training_loss"] for r in history]]
        ax.plot(x, y, color=colors[label], marker=".", markersize=4, linewidth=1.6,
                label="Full training set, frozen epoch-end policy")
        ax.plot(epochs, [r["online_minibatch_loss"] for r in history], color="#8c8c8c", alpha=.8,
                linewidth=1, linestyle="--", label="Minibatch average while weights change")
        ax.axhline(baseline[label], color="#303030", linestyle=":", linewidth=1.2,
                   label="No-trade baseline")
        ax.scatter([chosen], [best], marker="*", s=110, color=colors[label], edgecolors="black",
                   linewidths=.6, zorder=5, label="Best checkpoint so far" if report is None else "Selected checkpoint")
        progress = f"through epoch {epochs[-1]}" if report is None else f"stopped at {epochs[-1]}"
        ax.set_title(f"{label.capitalize()} claim\nBest epoch {chosen}; {progress}")
        ax.set_xlabel("Epoch (0 = warm start)" if warm_start else "Epoch (0 = no trading)")
        ax.set_ylabel("Penalized training loss — lower is better")
        ax.grid(alpha=.18)
        summary[label] = dict(epochs_run=epochs[-1], selected_epoch=chosen, best_training_loss=best,
            initial_no_trade_loss=baseline[label], initial_policy_loss=initial, final_training_loss=history[-1]["training_loss"],
            selected_test_economic_oce=(report["policies"][label]["economic_oce_frozen_translation"]["mean"]
                                       if report else None))
        rows.append(dict(label=label, epoch=0, training_loss=initial, online_minibatch_loss="",
                         best_training_loss=initial))
        rows.extend({key: r[key] for key in ("label", "epoch", "training_loss", "online_minibatch_loss",
                                            "best_training_loss")} for r in history)
    handles, labels = next(ax.get_legend_handles_labels() for ax, history in zip(axes, records.values()) if history)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    caption = " — live snapshot" if report is None else ""
    fig.suptitle(f"GROMM v1 (−4,3): expiry-return deep hedge; risk aversion {risk:g}{caption}\n{run.name}", fontsize=12)
    fig.subplots_adjust(top=.77, bottom=.25, left=.065, right=.985, wspace=.30)
    fig.savefig(output / "losses.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    with (output / "epoch_losses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), summary=summary), indent=2))


if __name__ == "__main__":
    main()
