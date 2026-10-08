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
    cumulative_evaluations: np.ndarray
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
    cumulative_evaluations: list[float] = []
    objective: list[float] = []
    gradient_rms: list[float] = []
    evaluations = 0.0
    for record in history:
        if "full_population_oce" in record:
            oce = float(record["full_population_oce"])
        elif "full_population_log_loss" in record:
            oce = -float(record["full_population_log_loss"])
        else:
            continue
        if "epoch" not in record or "gradient_rms" not in record:
            continue
        evaluations += float(record.get("lbfgs_closure_evaluations", 1.0))
        cumulative_evaluations.append(evaluations)
        objective.append(oce)
        gradient_rms.append(float(record["gradient_rms"]))
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (cumulative_evaluations, objective, gradient_rms)
    )
    if not len(arrays[0]):
        raise ValueError(f"{path} has no complete full-population convergence records")
    if any(np.any(~np.isfinite(values)) for values in arrays):
        raise ValueError(f"{path} contains non-finite convergence values")
    if np.any(arrays[2] <= 0.0):
        raise ValueError(f"{path} contains a nonpositive gradient RMS")
    return ConvergenceStage(
        label=label,
        cumulative_evaluations=arrays[0],
        oce_objective=arrays[1],
        gradient_rms=arrays[2],
    )


def plot_stages(
    stages: list[ConvergenceStage],
    output: Path,
    *,
    gradient_rms_target: float | None = None,
) -> None:
    """Render cumulative optimizer steps with explicit stage boundaries."""

    if not stages:
        raise ValueError("at least one convergence stage is required")
    if gradient_rms_target is not None and gradient_rms_target <= 0.0:
        raise ValueError("gradient RMS target must be positive")
    colors = ("#b2182b", "#6a3d9a", "#e66101", "#2166ac")
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    offset = 0.0
    for index, stage in enumerate(stages):
        step = offset + stage.cumulative_evaluations
        color = colors[index % len(colors)]
        axes[0].plot(step, stage.oce_objective, color=color, linewidth=1.6, label=stage.label)
        axes[1].plot(step, stage.gradient_rms, color=color, linewidth=1.6, label=stage.label)
        if index:
            boundary = offset
            for axis in axes:
                axis.axvline(boundary, color="0.55", linewidth=0.8, linestyle="--")
        offset = float(step[-1])

    axes[0].set_ylabel("OCE objective")
    axes[1].set_ylabel("Exact gradient RMS")
    axes[1].set_xlabel("Cumulative full-population evaluations")
    axes[1].set_yscale("log")
    if gradient_rms_target is not None:
        axes[1].axhline(
            gradient_rms_target,
            color="0.35",
            linewidth=0.9,
            linestyle=":",
        )
        axes[1].annotate(
            "stopping target",
            xy=(1.0, gradient_rms_target),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    for axis in axes:
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, ncols=min(len(stages), 3))
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
    parser.add_argument(
        "--gradient-rms-target",
        type=float,
        help="optional exact-gradient RMS stopping target shown on the plot",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    stages = [load_stage(label, Path(path)) for label, path in arguments.stage]
    plot_stages(
        stages,
        arguments.output,
        gradient_rms_target=arguments.gradient_rms_target,
    )
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
