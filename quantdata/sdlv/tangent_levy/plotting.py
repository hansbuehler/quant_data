"""Reproducible diagnostic figures for cached Tangent Lévy artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dynamics import DynamicsFit
from .experiment import HistoricalCalibration


def write_diagnostic_figure(
    history: HistoricalCalibration,
    dynamics: DynamicsFit,
    *,
    output: Path,
    reconstruction_tolerance: float = 0.25,
) -> Path:
    """Write the core fit, reconstruction, spectrum, and support diagnostics."""

    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    daily_rmse = np.asarray(
        [
            np.median([fit.weighted_rmse for fit in day_fit.fits])
            for day_fit in history.day_fits
        ],
        dtype=np.float64,
    )
    dates = np.asarray(history.dates, dtype="datetime64[D]")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(dates, daily_rmse, linewidth=0.8)
    axes[0, 0].set_title("Median maturity fit error")
    axes[0, 0].set_ylabel("weighted residual units")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        dates,
        history.reconstruction_l1_error,
        linewidth=0.8,
    )
    axes[0, 1].axhline(
        reconstruction_tolerance,
        color="tab:red",
        linestyle="--",
        linewidth=0.8,
    )
    axes[0, 1].set_yscale(
        "symlog",
        linthresh=max(reconstruction_tolerance, 1.0e-12),
    )
    axes[0, 1].set_title("Cumulative-density projection")
    axes[0, 1].set_ylabel("relative L1 change")
    axes[0, 1].grid(alpha=0.25)

    factor_index = np.arange(1, dynamics.num_factors + 1)
    axes[1, 0].bar(factor_index, dynamics.explained_variance_ratio)
    axes[1, 0].set_title("Retained constrained factors")
    axes[1, 0].set_xlabel("factor")
    axes[1, 0].set_ylabel("explained variance ratio")
    axes[1, 0].set_xticks(factor_index)

    active_mask = (
        np.ones(dynamics.beta.shape[1:], dtype=bool)
        if dynamics.active_mask is None
        else dynamics.active_mask
    )
    image = axes[1, 1].imshow(
        active_mask,
        origin="lower",
        aspect="auto",
        extent=(
            dynamics.jump_x[0],
            dynamics.jump_x[-1],
            dynamics.tau[0] * 252.0,
            dynamics.tau[-1] * 252.0,
        ),
        interpolation="nearest",
    )
    axes[1, 1].set_title("Dynamics active support")
    axes[1, 1].set_xlabel("jump size")
    axes[1, 1].set_ylabel("business days to maturity")
    figure.colorbar(image, ax=axes[1, 1], ticks=[0, 1])

    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
