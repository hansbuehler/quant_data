"""
Plotting helpers for single-option Kalman diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .single_option import SingleOptionKalmanResult


def _finite_values(frame: pl.DataFrame, column: str):
    values = frame[column].to_numpy()
    return values[np.isfinite(values)]


def plot_single_option_result(
    result: SingleOptionKalmanResult,
    *,
    title: str | None = None,
    secondary: str | None = "mid_vega",
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    frame = result.frame
    summary = result.summary
    opt_ticker = summary["opt_ticker"]
    title = title or f"{opt_ticker} rho={summary['rho']:.2f}"

    secondary_specs = {
        "mid_vega": ("mid vega", "tab:green"),
        "moneyness": ("moneyness K/S - 1", "tab:purple"),
        "abs_moneyness": ("|K/S - 1|", "tab:brown"),
    }
    if secondary is not None and secondary not in secondary_specs:
        raise ValueError(f"Unknown secondary series: {secondary}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title)
    ax_price, ax_vol, ax_hist = axes[0, 0], axes[0, 1], axes[1, 0]
    axes[1, 1].axis("off")

    t_d = frame["t_d"].to_numpy()
    ax_price.set_title("Prices")
    ax_price.plot(t_d, frame["mid_price"].to_numpy(), ".", color="black", label="mid")
    trade_price = frame["trade_price"].to_numpy()
    trade_mask = np.isfinite(trade_price)
    if trade_mask.any():
        ax_price.plot(t_d[trade_mask], trade_price[trade_mask], ".", color="tab:blue", label="trade")
    ax_price.plot(t_d, frame["filtered_price"].to_numpy(), "-", color="tab:red", label="filtered")
    ax_price.set_xlabel("t_d")
    ax_price.set_ylabel("price")
    ax_price.legend(loc="best")

    ax_vol.set_title("Implied vols")
    ax_vol.plot(t_d, frame["direct_mid_vol"].to_numpy(), ".", color="black", label="direct")
    ax_vol.plot(t_d, frame["stored_mid_vol"].to_numpy(), ":", color="grey", label="stored mid_vol")
    ax_vol.plot(t_d, frame["kalman_prior_vol"].to_numpy(), "--", color="tab:orange", alpha=0.85, label="prior")
    ax_vol.plot(t_d, frame["filtered_vol"].to_numpy(), "-", color="tab:red", label="filtered")
    intrinsic_mask = frame["is_intrinsic"].to_numpy().astype(bool)
    if intrinsic_mask.any():
        ax_vol.plot(t_d[intrinsic_mask], frame["filtered_vol"].to_numpy()[intrinsic_mask], ",", color="tab:orange", label="intrinsic rows")
    ax_vol.set_xlabel("t_d")
    ax_vol.set_ylabel("vol")
    ax_vol.legend(loc="upper left")

    if secondary is not None:
        secondary_label, secondary_color = secondary_specs[secondary]
        secondary_values = frame[secondary].to_numpy()
        secondary_mask = np.isfinite(secondary_values)
        if secondary_mask.any():
            ax2 = ax_vol.twinx()
            ax2.plot(t_d[secondary_mask], secondary_values[secondary_mask], "-", color=secondary_color, alpha=0.9, linewidth=1.2, label=secondary_label)
            ax2.set_ylabel(secondary_label)
            ax2.legend(loc="upper right")

    ax_hist.set_title("Direct IV differences")
    hist_specs = [
        ("direct_minus_previous_stored_mid_vol", "direct - stored[t-1]", "tab:gray"),
        ("direct_minus_kalman_prior_vol", "direct - prior[t|t-1]", "tab:orange"),
        ("direct_minus_filtered_vol", "direct - filtered[t]", "tab:red"),
    ]
    hist_values = [(label, _finite_values(frame, column), color) for column, label, color in hist_specs]
    combined = np.concatenate([values for _, values, _ in hist_values if values.size]) if any(values.size for _, values, _ in hist_values) else np.array([], dtype=float)
    if combined.size:
        lo, hi = np.nanpercentile(combined, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
        if lo == hi:
            pad = max(abs(lo) * 0.1, 1e-4)
            lo, hi = lo - pad, hi + pad
        bins = np.linspace(lo, hi, 60)
        for label, values, color in hist_values:
            if values.size:
                ax_hist.hist(values, bins=bins, alpha=0.45, color=color, label=f"{label} n={values.size}")
        ax_hist.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax_hist.set_xlim(lo, hi)
        ax_hist.set_xlabel("vol difference")
        ax_hist.set_ylabel("quote count")
        ax_hist.legend(loc="best")
    else:
        ax_hist.text(0.5, 0.5, "No finite differences", ha="center", va="center", transform=ax_hist.transAxes)
        ax_hist.set_axis_off()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140)
    return fig
