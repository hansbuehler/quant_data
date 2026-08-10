"""Publication figures for direct PCA-IV benchmark diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

from .pca import LogIVPCA, PCAFitDiagnostics


def _save(figure: Any, output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def write_iv_eigensurface_figure(
    model: LogIVPCA,
    diagnostics: PCAFitDiagnostics,
    output: str | Path,
) -> Path:
    """Plot the first five one-standard-deviation IV eigensurfaces."""

    import matplotlib.pyplot as plt

    surfaces = diagnostics.first_five_iv_eigensurfaces
    verify_inp(
        surfaces.ndim == 3 and surfaces.shape[1:] == model.surface_shape,
        lambda: f"invalid IV eigensurface geometry {surfaces.shape}",
    )
    components = surfaces.shape[0]
    figure, axes = plt.subplots(
        components,
        1,
        figsize=(8.0, 2.6 * components),
        constrained_layout=True,
        squeeze=False,
    )
    magnitude = max(float(np.max(np.abs(surfaces))), np.finfo(float).eps)
    for component, axis in enumerate(axes[:, 0]):
        image = axis.pcolormesh(
            model.nms,
            np.broadcast_to(model.expiries[:, None], model.nms.shape),
            surfaces[component],
            shading="auto",
            cmap="coolwarm",
            vmin=-magnitude,
            vmax=magnitude,
        )
        axis.axvline(0.0, color="black", linewidth=0.5, alpha=0.6)
        axis.set_title(f"log-IV level PC {component + 1}: one-score-std IV change")
        axis.set_ylabel("expiry (years)")
        figure.colorbar(image, ax=axis, label="implied-volatility change")
    axes[-1, 0].set_xlabel("normalized moneyness z")
    return _save(figure, output)


def write_pca_reconstruction_figure(
    model: LogIVPCA,
    diagnostics: PCAFitDiagnostics,
    output: str | Path,
) -> Path:
    """Plot explained variance and IV reconstruction error by expiry."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    components = np.arange(1, model.num_components + 1)
    axes[0].bar(components, model.explained_variance_ratio, label="individual")
    axes[0].plot(
        components,
        model.cumulative_explained_variance_ratio,
        color="black",
        marker=".",
        label="cumulative",
    )
    axes[0].set(xlabel="level component", ylabel="variance fraction")
    axes[0].legend()
    expiry = [item["expiry"] for item in diagnostics.errors_by_expiry]
    rmse = [item["iv_rmse"] for item in diagnostics.errors_by_expiry]
    maximum = [
        item["iv_maximum_absolute_error"]
        for item in diagnostics.errors_by_expiry
    ]
    axes[1].plot(expiry, rmse, marker="o", label="RMSE")
    axes[1].plot(expiry, maximum, marker="o", label="maximum absolute")
    axes[1].set(xlabel="expiry (years)", ylabel="IV error")
    axes[1].legend()
    return _save(figure, output)


def write_distribution_fidelity_figure(
    invariant: Mapping[str, Any],
    weekly: Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Plot matched invariant-level and weekly-increment distance metrics."""

    import matplotlib.pyplot as plt

    names = (
        "mean_rms_in_historical_std",
        "relative_covariance_frobenius",
        "sliced_wasserstein_in_historical_std",
        "coordinate_ks_mean",
    )
    invariant_metrics = invariant["historical_comparison"]
    weekly_metrics = weekly[
        "weekly_increment_vs_historical_weekly_increment"
    ]
    x = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    axis.bar(
        x - width / 2,
        [invariant_metrics[name] for name in names],
        width,
        label="invariant levels",
    )
    axis.bar(
        x + width / 2,
        [weekly_metrics[name] for name in names],
        width,
        label="weekly increments",
    )
    axis.set_xticks(x, [name.replace("_", "\n") for name in names])
    axis.set_ylabel("distance / discrepancy")
    axis.legend()
    return _save(figure, output)


def write_static_violation_figure(
    historical: Mapping[str, Any],
    generated: Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Plot physical surface-violation rates across roundoff conventions."""

    import matplotlib.pyplot as plt

    multipliers = [0, 1, 8, 32, 128]

    def fractions(report: Mapping[str, Any]) -> list[float]:
        values = report["by_roundoff_multiplier"]
        return [
            values[str(multiplier)]["aggregate"][
                "violating_surface_fraction"
            ]
            for multiplier in multipliers
        ]

    figure, axis = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    axis.plot(multipliers, fractions(historical), marker="o", label="historical PCA")
    axis.plot(multipliers, fractions(generated), marker="o", label="generated PCA")
    axis.set(
        xlabel="float32 roundoff multiplier",
        ylabel="surfaces with a violation",
        yscale="symlog",
    )
    axis.legend()
    return _save(figure, output)


def write_minority_tail_figure(
    reports: Mapping[str, Mapping[str, Any]], output: str | Path
) -> Path:
    """Plot the twenty smallest physical minority-sign probabilities."""

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    for split, report in reports.items():
        values = np.asarray(
            report["per_instrument"]["weighted_minority_probability"],
            dtype=np.float64,
        )[1:]
        axis.plot(
            np.arange(1, min(20, values.size) + 1),
            np.sort(values)[:20],
            marker="o",
            label=split,
        )
    axis.set(
        xlabel="rank among option instruments",
        ylabel="physical minority-sign probability",
    )
    axis.legend()
    return _save(figure, output)


def write_local_candidate_figure(
    report: Mapping[str, Any], output: str | Path
) -> Path:
    """Plot ordinary local one-sided candidate states by split/direction."""

    import matplotlib.pyplot as plt

    splits = ("training", "validation")
    directions = ("long_only", "short_only", "both_directions")
    values = np.asarray(
        [
            [
                report[split]["zero_adverse"][
                    "states_by_direction_exclusivity"
                ][direction]
                for direction in directions
            ]
            for split in splits
        ],
        dtype=np.float64,
    )
    x = np.arange(len(splits))
    width = 0.25
    figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for index, direction in enumerate(directions):
        axis.bar(
            x + (index - 1) * width,
            values[:, index],
            width,
            label=direction.replace("_", " "),
        )
    axis.set_xticks(x, splits)
    axis.set_ylabel("query states with candidates")
    axis.legend()
    return _save(figure, output)


__all__ = [
    "write_distribution_fidelity_figure",
    "write_iv_eigensurface_figure",
    "write_local_candidate_figure",
    "write_minority_tail_figure",
    "write_pca_reconstruction_figure",
    "write_static_violation_figure",
]
