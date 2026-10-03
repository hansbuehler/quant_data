r"""Section D plots: market quotes vs. dy2 vs. DLV x Bates surface plots,
and the fit-vs-recalibration-freedom plot for the reserve sweep. Reads
`report/data/full_study_results.json` (written by run_full_study.py) plus
re-derives a couple of live price curves for the surface plot (JSON stores
summary metrics, not full price curves, to keep the file small).
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "report" / "data"
FIGURES = HERE / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _load_results() -> list:
    with open(DATA / "full_study_results.json") as f:
        return json.load(f)


def plot_surface_comparison(ticker: str, d_str: str) -> None:
    """Market bid/ask band vs. dy2 vs. DLV x Bates, live-repriced (not from
    the JSON summary) so we get full price CURVES across a strike mesh, not
    just the sparse evaluation-contract points."""
    warnings.filterwarnings("ignore")
    from quantdata.dysanos.bates_crc.real_data_study.day_loader import build_evaluation_set, load_day
    from quantdata.dysanos.bates_crc.real_data_study.dy2_baseline import price_dy2
    from quantdata.dysanos.bates_crc.real_data_study.chain_bates_calibration import calibrate_chain_bates
    from quantdata.dysanos.bates_crc.real_data_study import config as cfg

    y, m, dd = (int(x) for x in d_str.split("-"))
    d = date(y, m, dd)
    book = load_day(ticker, d)
    ev = build_evaluation_set(book)
    calib = calibrate_chain_bates(ev, n_quad=cfg.FOURIER_N_QUAD_OPTIMIZATION, u_max=cfg.FOURIER_U_MAX_OPTIMIZATION)

    tenors = sorted(set(c.expiry_years for c in ev.contracts))[:4]
    fig, axes = plt.subplots(1, len(tenors), figsize=(4.5 * len(tenors), 4.2), sharey=False)
    if len(tenors) == 1:
        axes = [axes]
    for ax, tau in zip(axes, tenors):
        contracts = [c for c in ev.contracts if c.expiry_years == tau]
        strikes_mesh = np.linspace(min(c.strike for c in contracts) * 0.97, max(c.strike for c in contracts) * 1.03, 40)
        tau_arr = np.full_like(strikes_mesh, tau)
        p_dy2 = price_dy2(book, strikes_mesh, tau_arr)
        p_bates = calib.price(strikes_mesh, tau_arr)

        mkt_k = np.array([c.strike for c in contracts])
        mkt_bid = np.array([c.bid for c in contracts])
        mkt_ask = np.array([c.ask for c in contracts])
        ax.fill_between(mkt_k, mkt_bid, mkt_ask, color="#a0aec0", alpha=0.5, label="market bid/ask")
        ax.plot(strikes_mesh, p_dy2, color="#2b6cb0", lw=2, label="dy2")
        ax.plot(strikes_mesh, p_bates, color="#c05621", lw=1.8, ls="--", label="DLV x Bates")
        ax.set_title(f"$T={tau:.4f}$")
        ax.set_xlabel("pure strike $K$")
        ax.set_ylabel("pure call price")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{ticker} {d_str}: market vs. dy2 vs. DLV x Bates")
    fig.tight_layout()
    fig.savefig(FIGURES / f"surface_comparison_{ticker}_{d_str}.png", dpi=150)
    plt.close(fig)


def plot_fit_vs_recalibration_freedom(results: list, ticker: str, d_str: str) -> None:
    entry = next((r for r in results if r.get("ticker") == ticker and r.get("date") == d_str), None)
    if entry is None or "reserve_sweep" not in entry:
        return
    sweep = entry["reserve_sweep"]["budgets"]
    budgets = sorted(float(k) for k in sweep.keys())

    baseline_err = entry["models"]["dlv_bates"]["mean_fit_err"]
    before = [baseline_err if b == 0.0 else sweep[str(b)]["fit_before_recalibration"]["mean_fit_err"] for b in budgets]
    after = [baseline_err if b == 0.0 else sweep[str(b)].get("fit_after_recalibration", {}).get("mean_fit_err", np.nan)
             for b in budgets]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(budgets, before, "o-", color="#c05621", label="before recalibration (reserve turned on)")
    ax.plot(budgets, after, "o-", color="#2b6cb0", label="after recalibration (whole model refit)")
    ax.axhline(baseline_err, color="black", ls=":", lw=1, label="ordinary-Bates baseline")
    ax.set_xlabel("reserve budget $r_1$")
    ax.set_ylabel("mean spread-unit fit error")
    ax.set_title(f"{ticker} {d_str}: fit cost of the reserve, and how much recalibration recovers")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fit_vs_recalibration_freedom_{ticker}_{d_str}.png", dpi=150)
    plt.close(fig)


def plot_model_comparison_bars(results: list) -> None:
    spx = [r for r in results if r.get("ticker") == "SPX" and "models" in r]
    labels = [r["date"] for r in spx]
    model_names = ["dy2", "sanos_snapshot", "dlv_black", "dlv_bates"]
    colors = {"dy2": "#2b6cb0", "sanos_snapshot": "#38a169", "dlv_black": "#d69e2e", "dlv_bates": "#c05621"}

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = np.arange(len(labels))
    for i, name in enumerate(model_names):
        vals = [r["models"].get(name, {}).get("pcnt_inside_bid_ask", np.nan) * 100 for r in spx]
        ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=colors[name])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("% of evaluation contracts inside bid/ask")
    ax.set_title("SPX: bid-ask coverage by model and date")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison_bars.png", dpi=150)
    plt.close(fig)


def main() -> None:
    results = _load_results()
    spx_dates_with_data = [r["date"] for r in results if r.get("ticker") == "SPX" and "models" in r]
    preferred_rep_date = "2011-07-07"  # held-out, matches sec_main_comparison.tex's figure reference
    rep_date = preferred_rep_date if preferred_rep_date in spx_dates_with_data else (
        spx_dates_with_data[len(spx_dates_with_data) // 2] if spx_dates_with_data else None)
    if rep_date is not None:
        plot_surface_comparison("SPX", rep_date)
    for r in results:
        if r.get("ticker") == "SPX" and "reserve_sweep" in r:
            plot_fit_vs_recalibration_freedom(results, "SPX", r["date"])
    plot_model_comparison_bars(results)
    print("Wrote figures to", FIGURES)


if __name__ == "__main__":
    main()
