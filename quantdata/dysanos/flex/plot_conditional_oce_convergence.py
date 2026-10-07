"""Plot objective and exact-gradient convergence for conditional OCE fits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True, kw_only=True)
class ConvergenceStage:
    """One optimizer stage read from a serialized training history."""

    label: str
    local_epoch: np.ndarray
    oce_objective: np.ndarray
    gradient_rms: np.ndarray


def _training_history(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("training_history")
    elif path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        history = payload.get("history")
    else:
        raise ValueError(f"unsupported convergence input suffix {path.suffix!r}")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} has no nonempty training history")
    if any(not isinstance(record, Mapping) for record in history):
        raise ValueError(f"{path} contains an invalid training-history record")
    return history


def load_stage(label: str, path: Path) -> ConvergenceStage:
    """Load one optimizer stage, preserving every recorded exact evaluation."""

    history = _training_history(path)
    epoch: list[float] = []
    objective: list[float] = []
    gradient_rms: list[float] = []
    for record in history:
        if "full_population_oce" in record:
            oce = float(record["full_population_oce"])
        elif "full_population_log_loss" in record:
            oce = -float(record["full_population_log_loss"])
        else:
            continue
        if "epoch" not in record or "gradient_rms" not in record:
            continue
        epoch.append(float(record["epoch"]))
        objective.append(oce)
        gradient_rms.append(float(record["gradient_rms"]))
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (epoch, objective, gradient_rms)
    )
    if not len(arrays[0]):
        raise ValueError(f"{path} has no complete full-population convergence records")
    if any(np.any(~np.isfinite(values)) for values in arrays):
        raise ValueError(f"{path} contains non-finite convergence values")
    if np.any(arrays[2] <= 0.0):
        raise ValueError(f"{path} contains a nonpositive gradient RMS")
    return ConvergenceStage(
        label=label,
        local_epoch=arrays[0],
        oce_objective=arrays[1],
        gradient_rms=arrays[2],
    )


def plot_stages(stages: list[ConvergenceStage], output: Path) -> None:
    """Render cumulative optimizer steps with explicit stage boundaries."""

    if not stages:
        raise ValueError("at least one convergence stage is required")
    colors = ("#b2182b", "#6a3d9a", "#e66101", "#2166ac")
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    offset = 0.0
    for index, stage in enumerate(stages):
        step = offset + stage.local_epoch - stage.local_epoch[0] + 1.0
        color = colors[index % len(colors)]
        axes[0].plot(step, stage.oce_objective, color=color, linewidth=1.6, label=stage.label)
        axes[1].plot(step, stage.gradient_rms, color=color, linewidth=1.6, label=stage.label)
        if index:
            boundary = float(step[0]) - 0.5
            for axis in axes:
                axis.axvline(boundary, color="0.55", linewidth=0.8, linestyle="--")
        offset = float(step[-1])

    axes[0].set_ylabel("OCE objective")
    axes[1].set_ylabel("Exact gradient RMS")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, ncols=min(len(stages), 3))
    figure.suptitle("Q2 optimizer convergence", fontsize=12)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=2,
        metavar=("LABEL", "REPORT_OR_CHECKPOINT"),
        required=True,
        help="Optimizer-stage label and JSON report or Torch checkpoint.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    stages = [load_stage(label, Path(path)) for label, path in arguments.stage]
    plot_stages(stages, arguments.output)
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
