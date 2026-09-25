"""Plot nominal-expiry market-fit diagnostics from certified reprices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-native", type=Path, required=True)
    parser.add_argument("--weekly-affine", type=Path, required=True)
    parser.add_argument("--daily-native", type=Path, required=True)
    parser.add_argument("--daily-affine", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "flex-market-expiry-fit-v1":
        raise ValueError(f"unexpected market-expiry report schema: {path}")
    if payload.get("status") not in {None, "complete"}:
        raise ValueError(f"market-expiry report is not complete: {path}")
    rows = payload.get("by_nominal_market_dte")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"market-expiry report has no rows: {path}")
    return payload


def _series(
    payload: dict[str, Any],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows = payload["by_nominal_market_dte"]
    expiries = np.asarray(
        [row["nominal_market_dte"] for row in rows],
        dtype=np.float64,
    )
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if (
        np.any(~np.isfinite(expiries))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(expiries) <= 0.0)
    ):
        raise ValueError("market-expiry plot series must be finite and increasing")
    return expiries, values


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    temporary.replace(path)
    plt.close(figure)


def _plot(
    reports: dict[str, dict[str, Any]],
    *,
    field: str,
    title: str,
    ylabel: str,
    log_scale: bool,
    output: Path,
) -> None:
    styles = {
        "weekly_native": ("Weekly native", "#3b6fbd", "-", "o"),
        "weekly_affine": ("Weekly affine-200", "#3b6fbd", "--", "s"),
        "daily_native": ("Daily + weekly native", "#ed7626", "-", "o"),
        "daily_affine": ("Daily + weekly affine-200", "#ed7626", "--", "s"),
    }
    maximum_expiry = max(float(_series(report, field)[0][-1]) for report in reports.values())
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5), sharey=False)
    for axis, maximum, subtitle in (
        (axes[0], 14.0, "Short end: 1–14 business days"),
        (axes[1], maximum_expiry, f"Full fitted range: 1–{int(maximum_expiry)} days"),
    ):
        for key, report in reports.items():
            expiries, values = _series(report, field)
            selected = expiries <= maximum
            label, color, line_style, marker = styles[key]
            axis.plot(
                expiries[selected],
                values[selected],
                label=label,
                color=color,
                linestyle=line_style,
                marker=marker,
                linewidth=2.2,
                markersize=3.8,
            )
        axis.set_title(subtitle, fontsize=16)
        axis.set_xlabel("Nominal market expiry (business days)", fontsize=13)
        axis.set_ylabel(ylabel, fontsize=13)
        axis.grid(True, alpha=0.25, which="both")
        axis.tick_params(labelsize=11)
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0.0, 100.0)
    figure.suptitle(title, fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    _save_figure(figure, output)


def main() -> None:
    arguments = _arguments()
    reports = {
        "weekly_native": _read(arguments.weekly_native),
        "weekly_affine": _read(arguments.weekly_affine),
        "daily_native": _read(arguments.daily_native),
        "daily_affine": _read(arguments.daily_affine),
    }
    _plot(
        reports,
        field="percent_inside_bid_ask",
        title=(
            "Adding 1–4D nodes fixes weekly-model short-end extrapolation; "
            "affine-200 preserves most fit"
        ),
        ylabel="Quotes inside bid–ask (%)",
        log_scale=False,
        output=arguments.output_directory / "presentation_market_fit_by_expiry.png",
    )
    _plot(
        reports,
        field="conditional_mean_excess",
        title="Excess outside bid–ask: compression is secondary to expiry geometry",
        ylabel="Conditional mean excess (half-spreads)",
        log_scale=True,
        output=arguments.output_directory / "presentation_excess_by_expiry.png",
    )


if __name__ == "__main__":
    main()
