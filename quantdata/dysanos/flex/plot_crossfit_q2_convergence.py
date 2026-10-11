"""Plot both directions of the cross-fitted DYSANOS Q2 optimization."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True, kw_only=True)
class DirectionHistory:
    label: str
    epochs: np.ndarray
    training_oce: np.ndarray
    gradient_rms: np.ndarray
    validation_epochs: np.ndarray
    validation_oce: np.ndarray
    adam_epochs: int
    selected_epoch: int


def _history(path: Path) -> list[Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "dysanos-convex-q-conditional-oce-v1":
        raise ValueError(f"invalid conditional OCE checkpoint: {path}")
    return list(payload.get("history", []))


def _stage_arrays(
    path: Path,
    *,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = [
        record
        for record in _history(path)
        if "epoch" in record
        and "full_population_oce" in record
        and "gradient_rms" in record
    ]
    if not records:
        raise ValueError(f"checkpoint has no exact-gradient history: {path}")
    epochs = np.asarray([offset + int(record["epoch"]) for record in records])
    training_oce = np.asarray(
        [float(record["full_population_oce"]) for record in records]
    )
    gradient_rms = np.asarray([float(record["gradient_rms"]) for record in records])
    validation_records = [record for record in records if "validation_oce" in record]
    validation_epochs = np.asarray(
        [offset + int(record["epoch"]) for record in validation_records]
    )
    validation_oce = np.asarray(
        [float(record["validation_oce"]) for record in validation_records]
    )
    return epochs, training_oce, gradient_rms, validation_epochs, validation_oce


def load_direction(directory: Path, *, label: str) -> DirectionHistory:
    adam_path = directory / "adam" / "conditional-oce.pt"
    lbfgs_path = directory / "lbfgs" / "conditional-oce.pt"
    adam = _stage_arrays(adam_path, offset=0)
    adam_epochs = int(adam[0][-1])
    lbfgs = _stage_arrays(lbfgs_path, offset=adam_epochs)
    selection = json.loads((directory / "selected" / "selection.json").read_text())
    selected = selection["selected"]
    selected_epoch = int(selected["epoch"])
    if "/lbfgs/" in str(selected["path"]).replace("\\", "/"):
        selected_epoch += adam_epochs

    return DirectionHistory(
        label=label,
        epochs=np.concatenate((adam[0], lbfgs[0])),
        training_oce=np.concatenate((adam[1], lbfgs[1])),
        gradient_rms=np.concatenate((adam[2], lbfgs[2])),
        validation_epochs=np.concatenate((adam[3], lbfgs[3])),
        validation_oce=np.concatenate((adam[4], lbfgs[4])),
        adam_epochs=adam_epochs,
        selected_epoch=selected_epoch,
    )


def plot_histories(histories: tuple[DirectionHistory, ...], output: Path) -> None:
    if len(histories) != 2:
        raise ValueError("the cross-fitted convergence plot requires two directions")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    colors = (("#c62828", "#1565c0"), ("#7b1fa2", "#2e7d32"))
    for history, (training_color, validation_color) in zip(
        histories,
        colors,
        strict=True,
    ):
        axes[0].plot(
            history.epochs,
            history.training_oce,
            color=training_color,
            linewidth=1.7,
            label=f"{history.label} training",
        )
        axes[0].plot(
            history.validation_epochs,
            history.validation_oce,
            color=validation_color,
            linewidth=1.5,
            marker="o",
            markersize=2.6,
            label=f"{history.label} opposite fold",
        )
        axes[1].semilogy(
            history.epochs,
            history.gradient_rms,
            color=training_color,
            linewidth=1.7,
            label=history.label,
        )
        for axis in axes:
            axis.axvline(
                history.adam_epochs,
                color=training_color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
            )
            axis.axvline(
                history.selected_epoch,
                color=validation_color,
                linestyle=":",
                linewidth=1.3,
                alpha=0.85,
            )

    axes[0].set_ylabel("OCE Value")
    axes[1].set_ylabel("Exact Gradient RMS")
    axes[1].set_xlabel("Exact Fold-Gradient Epoch")
    axes[0].legend(ncol=2, frameon=False)
    axes[1].legend(ncol=2, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.margins(x=0.01)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--crossfit-directory", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    arguments = parser().parse_args()
    histories = (
        load_direction(arguments.crossfit_directory / "A-to-B", label="$A\\to B$"),
        load_direction(arguments.crossfit_directory / "B-to-A", label="$B\\to A$"),
    )
    plot_histories(histories, arguments.output)
    print(arguments.output, flush=True)


if __name__ == "__main__":
    main()
