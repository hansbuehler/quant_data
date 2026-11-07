"""Reproducible preliminary one-day hedge comparisons, with coverage and caveats."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def build_report(*, ordinary_directory: Path, orthogonal_directory: Path,
                 output_directory: Path) -> dict[str, object]:
    frames = []
    coverage = None
    inputs = {}
    for mode, directory in (("ordinary", ordinary_directory), ("spot-orthogonal", orthogonal_directory)):
        path = directory / "hedges" / "results.parquet"
        frame = pd.read_parquet(path)
        frame["variant"] = mode
        if mode != "ordinary":
            frame = frame.loc[~frame["method"].isin(["unhedged", "BS delta+vega"])]
        frames.append(frame)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        current = json.loads((directory / "observed_coverage.json").read_text())
        if coverage is not None and current != coverage:
            raise ValueError("Ordinary and orthogonal comparisons use different complete-case dates")
        coverage = current
    data = pd.concat(frames, ignore_index=True)
    data["label"] = np.where(data["method"].str.startswith("DLV"), data["variant"] + ": " + data["method"], data["method"])
    per_claim = data.groupby(["label", "side", "claim"], sort=False).agg(
        dates=("date", "nunique"), mean_pnl=("costed_pnl", "mean"), median_pnl=("costed_pnl", "median"),
        std_pnl=("costed_pnl", "std"), mean_absolute_pnl=("costed_pnl", lambda values: np.mean(np.abs(values))),
        mean_entry_cost=("entry_cost", "mean"), mean_hypothetical_exit_cost=("hypothetical_exit_cost", "mean"),
        cap_fraction=("cap_hits", lambda values: np.mean(values > 0)),
        mean_dlv_target_price=("dlv_target_price", "mean"))
    model_basis = data.assign(price_difference=data["dlv_target_price"] - data["market_target_price"])
    basis = model_basis.groupby(["variant", "claim"])["price_difference"].agg(["mean", "std", "min", "max"])
    summary = per_claim.groupby(["label", "side"], sort=False).mean(numeric_only=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    per_claim.to_parquet(output_directory / "per_claim.parquet")
    summary.to_parquet(output_directory / "summary_table.parquet")
    basis.to_parquet(output_directory / "digital_price_basis.parquet")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
                         "legend.fontsize": 10, "figure.titlesize": 17})
    pdf_path = output_directory / "HEDGE2_PRELIMINARY.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(14, 9))
        fig.suptitle("Hedge2 — preliminary one-day digital comparison", y=.96)
        paragraphs = [
            "Requested period: 2022–2025. Initialize in 2022; fit eight-PC AR(1) dynamics from the latest 252 exact daily transitions.",
            "Data: stored smoothness-0.1 SANOS interpolators; direct smoothness-zero DLV conversion on 9 maturities × 200 floating strikes.",
            f"Comparable sample: {coverage['eligible_dates']} complete-book origins, {coverage['eligible_start']} to {coverage['eligible_end']}.",
            f"Of {coverage['candidate_dates']} fitted origins, {len(coverage['excluded_dates'])} lack a complete observed book or a next session. Source history ends August 2025.",
            "Each origin: ten fixed-cash-strike 1D digital targets; ATM put and call, plus OTM NMS ±0.25, ±0.50, ±0.75, ±1.00.",
            "Hedge book: daily forward-financed spot plus PM OTM Europeans inside (−2,2) expiry-ATM NMS, through 5D or the first later expiry.",
            "Observed European mids; full option half-spread on entry; spot cost zero. No terminal exit charge; hypothetical exit cost is separate.",
            "Common inception digital marks: smooth SANOS curve. DLV hedge exposures: separate 1% spot and ±1 daily-PC standard-deviation shocks.",
            "Full Greek hedge matches exposures at minimum entry cost. Lower budgets re-solve a diagonal-risk-weighted Greek residual objective.",
            "Important: 66.5% of latent change energy involves guarded tail nodes. Atomic DLV digital prices need not closely match smooth source digitals.",
            "This is a complete-case cross-model hedge experiment, not an unconditional historical sample or a test of exact smooth Greeks.",
            "Parametric, full statistical rolling, and multi-day comparisons are NOT included yet. Selected-date statistical fits are diagnostics only.",
            "Tables use averages of ten product-specific daily sample standard deviations—not a pooled portfolio standard deviation.",
            "No confidence intervals are claimed here: targets on the same date, and opposite sides, are dependent. Final inference will group by date.",
        ]
        for index, paragraph in enumerate(paragraphs):
            fig.text(.055, .88 - .054 * index, paragraph, fontsize=11, va="top", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
        for side, side_name in ((-1, "Short target"), (1, "Long target")):
            table = summary.xs(side, level="side")
            fig, axis = plt.subplots(figsize=(14, 9))
            axis.axis("off")
            axis.set_title(f"{side_name} — same {coverage['eligible_dates']} dates; USD per $1m digital payout", pad=24)
            rows = []
            labels = []
            for label, values in table.iterrows():
                labels.append(label.replace("DLV Greek cost ", "Greek budget "))
                rows.append([f"{values['std_pnl']*1e6:,.0f}", f"{values['mean_pnl']*1e6:,.0f}",
                             f"{values['median_pnl']*1e6:,.0f}", f"{values['mean_entry_cost']*1e6:,.0f}",
                             f"{100*values['cap_fraction']:.1f}%"])
            graphic = axis.table(cellText=rows, rowLabels=labels,
                                  colLabels=["Std P&L", "Mean net P&L", "Median net P&L", "Entry cost", "Cap-hit rate"],
                                  loc="center", cellLoc="right", colWidths=[.15]*5)
            graphic.auto_set_font_size(False)
            graphic.set_fontsize(11)
            graphic.scale(1., 1.8)
            fig.subplots_adjust(left=.36, right=.99, top=.90, bottom=.12)
            fig.text(.05, .055, "Product metrics are computed across dates, then averaged over the ten products. Cap-hit rate counts target-date hedges with any position at ±1,000.", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
        fig, axis = plt.subplots(figsize=(12, 8))
        for mode, color in (("ordinary", "tab:blue"), ("spot-orthogonal", "tab:orange")):
            chosen = summary.xs(-1, level="side")
            chosen = chosen.loc[[f"{mode}: DLV Greek cost {fraction:.1f}" for fraction in (0., .2, .4, .6, .8, 1.)]]
            axis.plot(chosen["std_pnl"]*1e6, chosen["mean_entry_cost"]*1e6, "o-", color=color, label=f"DLV Greek, {mode}")
            for fraction, (_, values) in zip((0., .2, .4, .6, .8, 1.), chosen.iterrows(), strict=True):
                axis.annotate(f"{100*fraction:.0f}%", (values["std_pnl"]*1e6, values["mean_entry_cost"]*1e6),
                               xytext=(4, 7), textcoords="offset points", fontsize=10)
        for label, color in (("unhedged", "tab:gray"), ("BS delta+vega", "tab:green")):
            values = summary.loc[(label, -1)]
            axis.scatter(values["std_pnl"]*1e6, values["mean_entry_cost"]*1e6, s=100, color=color, label=label)
        axis.set(xlabel="Average target-specific daily P&L standard deviation (USD per $1m)",
                  ylabel="Average entry cost (USD per $1m)", title="Short digitals — realized risk versus hedge entry cost")
        axis.grid(alpha=.25)
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    metadata = {"pdf": str(pdf_path), "source_results_sha256": inputs,
                "coverage": coverage, "statistic": "mean across products of per-product sample statistics across market dates",
                "units": "initial-notional units in parquet; USD per1m on PDF tables",
                "preliminary_not_complete_benchmark": True}
    (output_directory / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ordinary-directory", "orthogonal-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(**vars(args))
    print(result["pdf"], flush=True)


if __name__ == "__main__":
    main()
