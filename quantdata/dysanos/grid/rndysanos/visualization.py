"""Checkpoint discovery and plotting for the OCE marimo dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

TRAINING_COLOR = "tab:red"
TRAINING_ALT_COLOR = "tab:purple"
TRAINING_LIGHT_COLOR = "tab:orange"
VALIDATION_COLOR = "tab:blue"
VALIDATION_ALT_COLOR = "tab:green"
VALIDATION_LIGHT_COLOR = "lightskyblue"


def discover_checkpoints(checkpoint_root: str, group: str = "") -> list[Path]:
    """Find model checkpoints for one group without loading model tensors."""

    root = Path(checkpoint_root).expanduser()
    directory = root / group if group else root
    return sorted(directory.glob("*.pt")) if directory.exists() else []


def load_checkpoint_history(path: str | Path) -> dict[str, Any]:
    """Load a trusted local OCE checkpoint for visualization."""

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"config", "tracker", "main_epoch", "moment_epoch"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"checkpoint {path} is missing fields {sorted(missing)}")
    return state


def _values(tracker: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    return list(tracker.get(name, []))


def _last(values: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return values[-1] if values else None


def plot_checkpoint(state: Mapping[str, Any]) -> plt.Figure:
    """Build the diagnostic dashboard for one checkpoint."""

    config = state["config"]
    tracker = state["tracker"]
    label = config["label"]
    group = config.get("group", "") or "default"
    training_values = _values(tracker, "training_values")
    validation_values = _values(tracker, "validation_values")
    valuation_epochs = np.asarray(tracker.get("valuation_epochs", []))
    latest_training = _last(training_values)
    latest_validation = _last(validation_values)

    figure, axes = plt.subplots(4, 3, figsize=(19, 18), constrained_layout=True)
    figure.suptitle(
        f"OCE measure change — {label} — group/session {group}", fontsize=15
    )

    loss_axis = axes[0, 0]
    epochs = np.asarray(tracker.get("epochs", []))
    batch_loss = np.asarray(tracker.get("batch_loss", []))
    if len(epochs):
        loss_axis.plot(epochs, batch_loss, color=TRAINING_COLOR, alpha=0.55, label="training batch loss")
    if len(valuation_epochs):
        loss_axis.plot(
            valuation_epochs,
            [value["loss"] for value in training_values],
            color=TRAINING_ALT_COLOR,
            marker=".",
            label="full-training loss",
        )
        if validation_values:
            loss_axis.plot(
                valuation_epochs[-len(validation_values) :],
                [value["loss"] for value in validation_values],
                color=VALIDATION_COLOR,
                marker=".",
                label="validation loss",
            )
    for restart_epoch in tracker.get("restart_epochs", []):
        loss_axis.axvline(restart_epoch, color="black", linestyle=":", alpha=0.5)
    loss_axis.set(title="Loss and effective learning rate", xlabel="epoch", ylabel="negative utility")
    learning_axis = loss_axis.twinx()
    if len(epochs):
        learning_axis.plot(epochs, tracker["lr"], color=TRAINING_LIGHT_COLOR, label="learning rate")
        learning_axis.set_yscale("log")
    learning_axis.set_ylabel("learning rate")
    handles, labels = loss_axis.get_legend_handles_labels()
    handles_lr, labels_lr = learning_axis.get_legend_handles_labels()
    if handles or handles_lr:
        loss_axis.legend(handles + handles_lr, labels + labels_lr, loc="best")

    moment_axis = axes[0, 1]
    moment_epochs = np.asarray(tracker.get("moment_epochs", []))
    if len(moment_epochs):
        moment_axis.plot(
            moment_epochs,
            tracker["moment_drift_batch_loss"],
            color=TRAINING_COLOR,
            alpha=0.6,
            label="DRIFT batch loss",
        )
        moment_axis.plot(
            moment_epochs,
            tracker["moment_sqr_batch_loss"],
            color=TRAINING_ALT_COLOR,
            alpha=0.6,
            label="SQR batch loss",
        )
    moment_training = _values(tracker, "moment_training_values")
    moment_validation = _values(tracker, "moment_validation_values")
    moment_value_epochs = np.asarray(tracker.get("moment_valuation_epochs", []))
    if moment_training:
        if len(moment_value_epochs) != len(moment_training):
            moment_value_epochs = np.linspace(
                1, max(moment_epochs[-1], 1), len(moment_training)
            )
        moment_axis.plot(
            moment_value_epochs,
            [value["drift_loss_mean"] for value in moment_training],
            color=TRAINING_LIGHT_COLOR,
            marker=".",
            label="full-training DRIFT loss",
        )
        moment_axis.plot(
            moment_value_epochs,
            [value["sqr_loss_mean"] for value in moment_training],
            color=TRAINING_ALT_COLOR,
            marker="x",
            label="full-training SQR loss",
        )
    if moment_validation:
        validation_moment_epochs = moment_value_epochs[-len(moment_validation) :]
        moment_axis.plot(
            validation_moment_epochs,
            [value["drift_loss_mean"] for value in moment_validation],
            color=VALIDATION_COLOR,
            marker=".",
            label="validation DRIFT loss",
        )
        moment_axis.plot(
            validation_moment_epochs,
            [value["sqr_loss_mean"] for value in moment_validation],
            color=VALIDATION_ALT_COLOR,
            marker="x",
            label="validation SQR loss",
        )
    moment_axis.set(title="Conditional moments", xlabel="moment epoch", ylabel="population MSE")
    if moment_axis.lines:
        moment_axis.set_yscale("log")
        moment_axis.legend(loc="best")

    density_axis = axes[0, 2]
    if training_values:
        density_axis.plot(
            valuation_epochs,
            [value["density_ess"] / value["n_error_observations"] for value in training_values],
            color=TRAINING_COLOR,
            marker=".",
            label="training observation ESS fraction",
        )
        density_axis.plot(
            valuation_epochs,
            [value["path_density_ess"] for value in training_values],
            color=TRAINING_ALT_COLOR,
            marker=".",
            label="training path ESS",
        )
    if validation_values:
        density_axis.plot(
            valuation_epochs[-len(validation_values) :],
            [value["density_ess"] / value["n_error_observations"] for value in validation_values],
            color=VALIDATION_COLOR,
            marker=".",
            label="validation observation ESS fraction",
        )
    density_axis.set(title="Density concentration", xlabel="epoch", ylabel="ESS / observations (path ESS unscaled)")
    if density_axis.lines:
        density_axis.legend(loc="best")

    def drift_panel(axis: plt.Axes, value: Mapping[str, Any], color: str, name: str) -> None:
        x = np.arange(len(value["mean_dH"]))
        mean = np.asarray(value["mean_dH"])
        err = np.asarray(value["err_dH"])
        axis.plot(x, mean, ".", color=color, label=name)
        axis.fill_between(x, mean - 2.0 * err, mean + 2.0 * err, color=color, alpha=0.18)

    drift_axis = axes[1, 0]
    if latest_training is not None:
        drift_panel(drift_axis, latest_training, TRAINING_COLOR, "full training ±2 MC error")
    if latest_validation is not None:
        drift_panel(drift_axis, latest_validation, VALIDATION_COLOR, "validation ±2 MC error")
    drift_axis.axhline(0.0, color="black", linewidth=0.8)
    drift_axis.set(title="Density-weighted drift", xlabel="instrument", ylabel="mean D_new dH")
    if drift_axis.lines:
        drift_axis.legend(loc="best")

    z_axis = axes[1, 1]
    if latest_training is not None:
        z_axis.plot(latest_training["mean_dH_over_err"], ".", color=TRAINING_COLOR, label="full training")
    if latest_validation is not None:
        z_axis.plot(latest_validation["mean_dH_over_err"], ".", color=VALIDATION_COLOR, label="validation")
    z_axis.axhline(2.0, color="black", linestyle="--")
    z_axis.axhline(-2.0, color="black", linestyle="--")
    z_axis.set(title="Drift / MC error", xlabel="instrument", ylabel="mean / err (S*TAU convention)")
    if z_axis.lines:
        z_axis.legend(loc="best")

    roc_axis = axes[1, 2]
    if latest_training is not None:
        roc_axis.plot(latest_training["return_on_cost"], ".", color=TRAINING_COLOR, label="full training")
    if latest_validation is not None:
        roc_axis.plot(latest_validation["return_on_cost"], ".", color=VALIDATION_COLOR, label="validation")
    roc_axis.axhline(1.0, color="black", linestyle="--")
    roc_axis.axhline(-1.0, color="black", linestyle="--")
    roc_axis.set(title="Near-martingale return on cost", xlabel="instrument", ylabel="E[D_new dH] / E[D_new c]")
    if roc_axis.lines:
        roc_axis.legend(loc="best")

    cost_axis = axes[2, 0]
    if latest_training is not None:
        x = np.arange(len(latest_training["c"]))
        cost_axis.plot(x, latest_training["c_cond"], color=TRAINING_COLOR, label="mean c_cond")
        cost_axis.plot(x, latest_training["bar_c"], color=TRAINING_ALT_COLOR, label="global bar_c")
        cost_axis.plot(x, latest_training["c_prop"], color=TRAINING_LIGHT_COLOR, label="mean c_prop")
        cost_axis.plot(x, latest_training["c"], color="black", label="effective c")
    cost_axis.set(title="Training cost components", xlabel="instrument", ylabel="cost")
    if cost_axis.lines:
        cost_axis.legend(loc="best")

    binding_axis = axes[2, 1]
    if latest_training is not None:
        binding_axis.plot(latest_training["binding_cond_fraction"], color=TRAINING_COLOR, label="c_cond binding")
        binding_axis.plot(latest_training["binding_uncond_fraction"], color=TRAINING_ALT_COLOR, label="bar_c binding")
        binding_axis.plot(latest_training["binding_prop_fraction"], color=TRAINING_LIGHT_COLOR, label="c_prop binding")
    binding_axis.set(title="Cost binding frequencies (ties counted)", xlabel="instrument", ylabel="fraction")
    if binding_axis.lines:
        binding_axis.legend(loc="best")

    calibration_axis = axes[2, 2]
    if latest_training is not None:
        calibration_axis.plot(
            latest_training["variance_calibration_ratio"],
            ".",
            color=TRAINING_COLOR,
            label="full training",
        )
    if latest_validation is not None:
        calibration_axis.plot(
            latest_validation["variance_calibration_ratio"],
            ".",
            color=VALIDATION_COLOR,
            label="validation",
        )
    calibration_axis.axhline(1.0, color="black", linestyle="--")
    calibration_axis.set(title="Conditional-variance calibration", xlabel="instrument", ylabel="residual variance / predicted v")
    if calibration_axis.lines:
        calibration_axis.legend(loc="best")

    excess_axis = axes[3, 0]
    if training_values and "unsigned_excess_mean_across_instruments" in training_values[-1]:
        multiples = np.asarray(training_values[-1]["excess_cost_multiples"])
        histories = np.asarray(
            [value["unsigned_excess_mean_across_instruments"] for value in training_values]
        )
        for index, multiple in enumerate(multiples):
            excess_axis.plot(
                valuation_epochs,
                histories[:, index],
                marker=".",
                label=f"training N={multiple:g}",
            )
        reference = np.asarray(
            training_values[0]["reference_unsigned_excess_mean_across_instruments"]
        )
        for index, multiple in enumerate(multiples):
            excess_axis.axhline(
                reference[index],
                linestyle=":",
                alpha=0.45,
                label=f"reference N={multiple:g}",
            )
    excess_axis.set(
        title="Density-weighted unsigned excess",
        xlabel="epoch",
        ylabel="mean across instruments",
    )
    if excess_axis.lines:
        excess_axis.set_yscale("log")
        excess_axis.legend(loc="best", ncol=2)

    cap_axis = axes[3, 1]
    if latest_training is not None and "action_cap_objective_derivative" in latest_training:
        cap_axis.plot(
            latest_training["action_cap_objective_derivative"],
            ".",
            color=TRAINING_COLOR,
            label="full-training objective derivative",
        )
    if latest_validation is not None and "action_cap_objective_derivative" in latest_validation:
        cap_axis.plot(
            latest_validation["action_cap_objective_derivative"],
            ".",
            color=VALIDATION_COLOR,
            label="validation objective derivative",
        )
    cap_axis.axhline(0.0, color="black", linewidth=0.8)
    cap_axis.set(
        title="Derivative by instrument cap",
        xlabel="instrument",
        ylabel="d OCE objective / d cap",
    )
    if cap_axis.lines:
        cap_axis.legend(loc="best")

    normalization_axis = axes[3, 2]
    if training_values and "density_mean_minus_one_over_err" in training_values[-1]:
        normalization_axis.plot(
            valuation_epochs,
            [value["density_mean_minus_one_over_err"] for value in training_values],
            color=TRAINING_COLOR,
            marker=".",
            label="full training",
        )
    if validation_values and "density_mean_minus_one_over_err" in validation_values[-1]:
        normalization_axis.plot(
            valuation_epochs[-len(validation_values) :],
            [value["density_mean_minus_one_over_err"] for value in validation_values],
            color=VALIDATION_COLOR,
            marker=".",
            label="validation",
        )
    normalization_axis.axhline(2.0, color="black", linestyle="--")
    normalization_axis.axhline(-2.0, color="black", linestyle="--")
    normalization_axis.set(
        title="Density normalization",
        xlabel="epoch",
        ylabel="(E[D_new] - 1) / MC error",
    )
    if normalization_axis.lines:
        normalization_axis.legend(loc="best")
    return figure


__all__ = ["discover_checkpoints", "load_checkpoint_history", "plot_checkpoint"]
