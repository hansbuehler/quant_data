"""Plot saved hedge losses without reopening training or validation scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(policy_directory: Path, *, output: Path) -> dict[str, object]:
    """Plot saved full-population losses and retain their exact source snapshot."""
    history_path = policy_directory / "training_history.json"
    report = json.loads((policy_directory / "training_report.json").read_text())
    rows = json.loads(history_path.read_text())["epochs"]
    if not rows:
        raise ValueError("the checkpoint has no completed training epochs")
    epochs = np.asarray([row["epoch"] for row in rows])
    losses = np.asarray([row["objective"] for row in rows])
    penalties = np.asarray([row["penalty"] for row in rows])
    validation = np.asarray([row.get("validation_objective", np.nan) for row in rows])
    has_validation = bool(np.any(np.isfinite(validation)))
    weight = float(report["policy_config"]["penalty_weight"])
    if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(penalties)):
        raise ValueError("saved training losses and penalties must be finite")
    best = int(np.argmin(losses))
    settings = {"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
                "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12}
    with plt.rc_context(settings):
        figure, axis = plt.subplots(figsize=(11, 5.5), layout="constrained")
        axis.plot(epochs, losses, color="#295f98", linewidth=1.7, label="Training loss: −OCE + numerical penalty")
        axis.plot(epochs, losses - weight * penalties, color="#db8b28", linewidth=1.2,
                  linestyle="--", label="−OCE, excluding numerical penalty")
        if has_validation:
            axis.plot(epochs, validation, color="#278a53", linewidth=1.8, label="Validation loss: same joint objective")
            matches = np.flatnonzero(epochs == report.get("best_epoch", -1))
            selected = int(matches[0]) if len(matches) and np.isfinite(validation[matches[0]]) else int(np.nanargmin(validation))
            axis.scatter([epochs[selected]], [validation[selected]], color="#7e3b92", s=45, zorder=4,
                         label=f"Best validation epoch: {epochs[selected]}")
        axis.scatter([epochs[best]], [losses[best]], color="#b12a40", s=42, zorder=4,
                     label=f"Best completed epoch: {epochs[best]}")
        axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":")
        portfolio = "joint zero/exotic hedge" if "shared_parameters" in report else (
            "payoff hedge" if report["policy_config"].get("liability_units", 0.0) else "empty-portfolio hedge")
        subtitle = "Training sample only; no validation curve recorded"
        if has_validation:
            counts = f"{report['training_paths']:,} training / {report['validation_paths']:,} validation paths" if (
                "training_paths" in report and "validation_paths" in report) else "Training and validation loss"
            subtitle = counts + "; validation selects stopping"
        axis.set(xlabel="Epoch", ylabel="Loss (lower is better)",
                 title=f"GROMM2 {portfolio} · λ = {report['policy_config']['risk_aversion']:g}\n{subtitle}")
        axis.grid(alpha=0.18)
        axis.legend(loc="upper right")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)
    snapshot = {"source": str(history_path), "training_only": not has_validation,
                "policy_report": report, "epochs": rows, "plot": str(output)}
    output.with_suffix(".json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    result = {"epochs": int(epochs[-1]), "best_epoch": int(epochs[best]),
              "best_loss": float(losses[best]), "latest_loss": float(losses[-1]), "plot": str(output)}
    if "shared_parameters" in report:
        result["subtasks"] = plot_joint_losses(policy_directory, output=output.with_name(output.stem + "_tasks" + output.suffix))
    return result


def plot_joint_losses(policy_directory: Path, *, output: Path) -> dict[str, object]:
    """Plot joint and individual net-of-cost risks from one saved epoch snapshot."""
    report = json.loads((policy_directory / "training_report.json").read_text())
    rows = json.loads((policy_directory / "training_history.json").read_text())["epochs"]
    if not rows or any("train" not in row or "validation" not in row for row in rows):
        raise ValueError("joint subtask plots require saved training and validation metrics")
    epochs = np.asarray([row["epoch"] for row in rows])
    series = {}
    for split in ("train", "validation"):
        for task in ("empty", "claim"):
            series[f"{split}_{task}"] = np.asarray([
                row[split][task]["loss_at_training_translation"] for row in rows])
        series[f"{split}_joint"] = np.asarray([row[split]["economic_objective"] for row in rows])
        if not np.allclose(series[f"{split}_joint"], series[f"{split}_empty"] + series[f"{split}_claim"], rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("recorded joint loss does not equal the two economic subtask losses")
    if not all(np.all(np.isfinite(values)) for values in series.values()):
        raise ValueError("joint subtask losses must be finite")
    titles = {"joint": "Joint: zero + exotic", "empty": "Zero portfolio", "claim": "Exotic hedge"}
    selected_epoch = int(report["best_epoch"])
    settings = {"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
                "legend.fontsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11}

    def draw(axis, task: str) -> None:
        train_label = f"Training — {report['training_paths']:,} paths" if "training_paths" in report else "Training"
        validation_label = f"Validation — {report['validation_paths']:,} paths" if "validation_paths" in report else "Validation"
        axis.plot(epochs, series[f"train_{task}"], color="#bc3f58", linewidth=1.8, label=train_label)
        axis.plot(epochs, series[f"validation_{task}"], color="#278a53", linewidth=1.8, label=validation_label)
        if task == "empty":
            axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":")
        axis.axvline(selected_epoch, color="#7e3b92", linewidth=1.0, linestyle="--",
                     label=f"Selected joint checkpoint: epoch {selected_epoch}")
        axis.set(title=titles[task], xlabel="Epoch", ylabel="Risk loss (lower is better)")
        axis.grid(alpha=0.18)
        axis.legend(loc="upper left")

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(settings):
        figure, axes = plt.subplots(3, 1, figsize=(11, 12), layout="constrained")
        preconditioned = report.get("preconditioned", str(report.get("preconditioning", "")).startswith("Separate"))
        measure = "Entropy-preconditioned path measures" if preconditioned else "Original Q samples · equal path weights"
        figure.suptitle(f"GROMM2 joint options-only hedge · λ = {report['policy_config']['risk_aversion']:g}\n"
                       f"{report['policy_config'].get('transaction_cost_bps', 0.0):g} bp turnover cost · training vs validation\n{measure}", fontsize=17)
        for axis, task in zip(axes, titles, strict=True):
            draw(axis, task)
        figure.supxlabel("Net-of-cost OCE risk before numerical action penalties; joint loss is the sum of both subtasks.", fontsize=11)
        figure.savefig(output, dpi=170)
        plt.close(figure)
        individual = {}
        for task in titles:
            figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
            draw(axis, task)
            file = output.with_name(output.stem + "_" + task + output.suffix)
            figure.savefig(file, dpi=170)
            plt.close(figure)
            individual[task] = str(file)
    snapshot = {"source": str(policy_directory / "training_history.json"), "epochs": epochs.tolist(),
        "series": {name: values.tolist() for name, values in series.items()}, "selected_joint_epoch": selected_epoch,
        "loss_convention": "Negative OCE at training-fitted translations; transaction costs included, numerical action penalties excluded.",
        "individual_plots": individual}
    output.with_suffix(".json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    return {"epochs": int(epochs[-1]), "plot": str(output), "individual_plots": individual}


def plot_independent_losses(policy_directory: Path, *, output: Path) -> dict[str, object]:
    """Show the independently selected zero, short-claim and long-claim loss curves."""
    report = json.loads((policy_directory / "training_report.json").read_text())
    rows = json.loads((policy_directory / "training_history.json").read_text())["epochs"]
    epochs = np.asarray([entry["epoch"] for entry in rows])
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    for axis, name, title in zip(axes, ("zero", "short", "long"),
                                 ("Zero portfolio", "Short payoff: seller hedge", "Long payoff: buyer hedge"), strict=True):
        axis.plot(epochs, [entry["train"][name]["risk_at_training_translation"] for entry in rows],
            color="#c93c55", label=f"Training: {report['training_paths']:,} paths")
        axis.plot(epochs, [entry["validation"][name]["risk_at_training_translation"] for entry in rows],
            color="#258b57", label=f"Validation: {report['validation_paths']:,} paths")
        best = report["policies"][name]["best_epoch"]
        axis.axvline(best, color="#8b43b0", linestyle="--", label=f"Selected checkpoint: epoch {best}")
        axis.set(title=title, xlabel="Epoch", ylabel="Net-of-cost OCE risk; lower is better")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=10)
    background = report["scenario_model"].get("background_volatility", 0.0)
    figure.suptitle(f"GROMM2 independent portfolio policies · λ={report['config']['risk_aversion']:g}\n"
        f"{report['config']['label']} · Black smoothing {100 * background:g}% · original Q, equal weights", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    result = {"plot": str(output), "completed_epochs": int(epochs[-1]),
        "best_epochs": {name: record["best_epoch"] for name, record in report["policies"].items()}}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subtasks", action="store_true", help="Plot joint, zero-portfolio and exotic losses together and separately")
    arguments = parser.parse_args()
    plot = plot_joint_losses if arguments.subtasks else plot_training_loss
    print(json.dumps(plot(arguments.policy_directory, output=arguments.output)), flush=True)


if __name__ == "__main__":
    main()
