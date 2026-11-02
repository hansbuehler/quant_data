"""Rebuild plots and a tenor table from a completed variance-index run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

from quantdata.dysanos.v11.variance_indices import BUSINESS_DTES

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LABELS = {"gaussian": "Gaussian AR", "lstm": "LSTM", "selective_ssm": "Mamba-style SSM"}
COLORS = {"gaussian": "#777777", "lstm": "#d95f02", "selective_ssm": "#7a5195"}


def plot_run(*, output: Path, historic_targets: Path) -> None:
    history = pl.read_parquet(historic_targets)
    values = np.column_stack([history[f"vix_{tenor}d"].to_numpy() for tenor in BUSINESS_DTES])
    samples = {"gaussian": np.load(output / "gaussian_audit.npz")["indices"],
               **{name: np.load(output / name / "audit_indices.npz")["indices"]
                  for name in ("lstm", "selective_ssm")}}
    probabilities = np.linspace(.005, .995, 101)
    target_quantiles = np.nanquantile(np.log(values), probabilities, axis=0)
    scale = np.nanstd(np.log(values), axis=0)
    rows = []
    for name, simulated in samples.items():
        discrepancy = np.mean(((np.quantile(np.log(simulated), probabilities, axis=0) - target_quantiles) / scale)**2, axis=0)
        for j, tenor in enumerate(BUSINESS_DTES):
            rows.append({"model": LABELS[name], "business_days": tenor,
                         "historic_n": int(np.isfinite(values[:, j]).sum()), "sampled_paths": len(simulated),
                         "quantile_discrepancy": float(discrepancy[j]),
                         **{f"historic_p{p}": float(np.nanpercentile(values[:, j], p)) for p in (5, 50, 95, 99)},
                         **{f"model_p{p}": float(np.percentile(simulated[:, j], p)) for p in (5, 50, 95, 99)}})
    pl.DataFrame(rows).write_csv(output / "tenor_comparison.csv")

    fig, axes = plt.subplots(4, 4, figsize=(14, 11), layout="constrained")
    for j, (tenor, ax) in enumerate(zip(BUSINESS_DTES, axes.flat, strict=False)):
        observed = values[np.isfinite(values[:, j]), j]
        all_values = np.concatenate([observed, *[v[:, j] for v in samples.values()]])
        bins = np.linspace(all_values.min(), all_values.max(), 31)
        ax.hist(observed, bins=bins, density=True, alpha=.3, color="#348abd", label="Historical SANOS")
        for name, simulated in samples.items():
            ax.hist(simulated[:, j], bins=bins, density=True, histtype="step", color=COLORS[name],
                    linestyle="--" if name == "gaussian" else "-", label=LABELS[name])
        ax.set_title(f"{tenor} business days; N={len(observed)}", fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(BUSINESS_DTES):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9)
    fig.suptitle("DYSANOS [−4,3]: analytic variance-index distributions", fontsize=14)
    fig.savefig(output / "variance_index_histograms.png", dpi=160)
    fig.savefig(output / "variance_index_histograms.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, percentile in zip(axes, (50, 95, 99), strict=True):
        ax.plot(BUSINESS_DTES, np.nanpercentile(values, percentile, axis=0), color="#348abd",
                marker="o", markersize=3, linewidth=2, label="Historical SANOS")
        for name, simulated in samples.items():
            ax.plot(BUSINESS_DTES, np.percentile(simulated, percentile, axis=0), color=COLORS[name],
                    linestyle="--" if name == "gaussian" else "-", label=LABELS[name])
        ax.set(title="Median" if percentile == 50 else f"{percentile}th percentile", xlabel="Business-day expiry",
               ylabel="Variance-index vol points")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Full-history distribution fit: 891 dates; only 83 at 70D", fontsize=12)
    fig.savefig(output / "variance_index_quantiles.png", dpi=170)
    fig.savefig(output / "variance_index_quantiles.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    for name in ("lstm", "selective_ssm"):
        progress = json.loads((output / name / "progress.json").read_text())["history"]
        epochs = np.asarray([r["epoch"] for r in progress])
        losses = np.asarray([r["quantile_loss"] for r in progress])
        ax.plot(epochs, losses, color=COLORS[name], alpha=.15, linewidth=.7)
        window = min(20, len(losses))
        ax.plot(epochs[window - 1:], np.convolve(losses, np.ones(window) / window, mode="valid"),
                color=COLORS[name], label=LABELS[name])
    ax.set(xlabel="Epoch", ylabel="Standardized log-quantile loss", yscale="log",
           title="Variance-index training loss (20-epoch moving average)")
    ax.legend(frameon=False)
    fig.savefig(output / "training_loss.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historic-targets", type=Path, required=True)
    plot_run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
