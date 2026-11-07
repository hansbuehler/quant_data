"""Matched-date one-day Greek and objective-specific parametric hedge evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.metrics import paired_risk_interval, summarize_products


def _configuration_label(row: pd.Series) -> str:
    if row["method"] in ("unhedged", "BS delta+vega"):
        return str(row["method"])
    if row["method"].startswith("DLV Greek"):
        return f"Greek budget {100 * row['greek_budget']:.0f}%"
    risk = "infinity" if np.isinf(row["risk_aversion"]) else f"{row['risk_aversion']:g}"
    objective = "MV" if row["objective"] == "mean_variance" else "Power-exp"
    return f"Parametric {objective}, lambda={risk}"


def build_objective_report(*, greek_ordinary_directory: Path, greek_orthogonal_directory: Path,
                           parametric_ordinary_directory: Path, parametric_orthogonal_directory: Path,
                           output_directory: Path) -> dict[str, object]:
    """Include only fully completed, identically dated cohorts; retain every configuration."""
    frames, source_hashes, audit_summaries = [], {}, {}
    dates = None
    coverage = None
    directories = {"ordinary": (greek_ordinary_directory, parametric_ordinary_directory),
                   "spot-orthogonal": (greek_orthogonal_directory, parametric_orthogonal_directory)}
    for variant, (greek_directory, parametric_directory) in directories.items():
        greek_path = greek_directory / "hedges" / "results.parquet"
        parametric_path = parametric_directory / "results.parquet"
        greek = pd.read_parquet(greek_path)
        parametric = pd.read_parquet(parametric_path)
        completed = json.loads((parametric_directory / "summary.json").read_text())
        if completed["status"] != "complete" or not bool(parametric["optimization_certified"].all()):
            raise ValueError("A partial or uncertified parametric run cannot enter the completed-cohort report")
        current_dates = set(greek["date"])
        if current_dates != set(parametric["date"]) or dates is not None and current_dates != dates:
            raise ValueError("Every strategy must use the same observed-market origin dates")
        dates = current_dates
        current_coverage = json.loads((greek_directory / "observed_coverage.json").read_text())
        if coverage is not None and current_coverage != coverage:
            raise ValueError("PCA variants use different coverage restrictions")
        coverage = current_coverage
        greek["objective"], greek["risk_aversion"] = "exposure", np.nan
        greek["greek_budget"] = greek["method"].str.extract(r"cost ([0-9.]+)", expand=False).astype(float)
        parametric["greek_budget"] = np.nan
        for frame in (greek, parametric):
            frame["variant"] = variant
            frame["label"] = frame.apply(_configuration_label, axis=1)
            frames.append(frame)
        for path in (greek_path, parametric_path):
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit_summaries[variant] = completed
    data = pd.concat(frames, ignore_index=True)
    grouping = ("variant", "label", "method", "objective", "risk_aversion", "greek_budget", "side", "claim")
    products = summarize_products(data, group_columns=grouping)
    summary = products.groupby(list(grouping[:-1]), sort=False, dropna=False).mean(numeric_only=True).reset_index()
    summary["products"] = 10
    sessions = get_business_days(date.fromisoformat(min(dates)), date.fromisoformat(max(dates)))
    lookup = {day.isoformat(): index for index, day in enumerate(sessions)}
    ordered_dates = sorted(dates)
    indices = np.array([lookup[value] for value in ordered_dates])
    intervals = []
    for (variant, side), group in data.groupby(["variant", "side"], sort=False):
        reference = group.loc[group["method"] == "unhedged"].pivot(index="date", columns="claim", values="costed_pnl")
        reference = reference.loc[ordered_dates].sort_index(axis=1)
        for label, chosen in group.groupby("label", sort=False):
            candidate = chosen.pivot(index="date", columns="claim", values="costed_pnl").reindex(index=reference.index, columns=reference.columns)
            interval = paired_risk_interval(candidate=candidate.to_numpy(), reference=reference.to_numpy(),
                session_indices=indices, count=2000, block_length=20, seed=20261106)
            intervals.append({"variant": variant, "side": side, "label": label, **interval})
    interval_frame = pd.DataFrame(intervals)
    summary = summary.merge(interval_frame, on=["variant", "side", "label"], validate="one_to_one")
    output_directory.mkdir(parents=True, exist_ok=True)
    products.to_parquet(output_directory / "per_product.parquet", index=False)
    summary.to_parquet(output_directory / "summary.parquet", index=False)
    interval_frame.to_parquet(output_directory / "paired_uncertainty.parquet", index=False)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10})
    pdf_path = output_directory / "HEDGE2.pdf"
    with PdfPages(pdf_path) as pdf:
        figure = plt.figure(figsize=(14, 9))
        figure.suptitle("Hedge2 — rolling one-day digital hedges", fontsize=19, y=.96)
        paragraphs = [
            "Scope: 2022 initialization; fresh causal 252-transition, eight-factor PCA/mean-reverting AR(1) on each eligible 2023–2025 date.",
            "Source: saved smoothness-0.1 SANOS market interpolation, converted algebraically to zero-smoother 9 × 200 floating-grid DLV states.",
            f"Matched sample: {len(dates)} complete-book origins, {min(dates)} to {max(dates)}; ten 1D digital products, long and short separately.",
            "Source ends 2025-08-29. September–December 2025 is unavailable. Incomplete observed quote books remain excluded, never model-filled.",
            "Book: forward-financed daily spot plus observed PM OTM Europeans in (−2,2) expiry-ATM NMS, through 5D or first later expiry.",
            "Spot cost zero; option entry half-spread multiplier one; no charged terminal exit. P&L uses one fixed initial notional, not premium returns.",
            "Greek: 1% fixed-state spot shock and ±1 daily-PC standard deviations; actual cost-constrained re-solves at 0,20,…,100% of full-match cost.",
            "Parametric: same finite-shock exposures, full spot/factor covariance, separately optimized MV and Gaussian power-exponential OCE.",
            "Lambda is 1,10,100,1000 in initial-notional wealth units. Infinite MV is pure variance; costs do not affect its optimizer but are paid in realized P&L.",
            "All parametric objective certificates passed; independent cash replay is reported in separate linked audit artifacts. Exact solver warnings remain saved.",
            "Major modeling caveat: 66.5% of latent change energy involves guarded tail nodes; atomic DLV digitals differ from smooth SANOS market marks.",
            "Statistics: average of ten product-specific sample statistics across dates, never a pooled portfolio or independent-strike test.",
            "Uncertainty: 2,000 paired circular 20-business-session block draws, sharing date weights across products and models and retaining quote gaps.",
            "Intervals are descriptive 95%, without multiple-comparison adjustment. No configuration is selected using these same out-of-sample outcomes.",
            "Full statistical runs and multi-day results are not complete and do not enter these tables. Diagnostic selected-date fits are not OOS evidence.",
        ]
        for index, paragraph in enumerate(paragraphs):
            figure.text(.04, .88 - .052 * index, paragraph, fontsize=10.5, va="top")
        pdf.savefig(figure)
        plt.close(figure)
        for variant in directories:
            for side in (-1, 1):
                chosen = summary.loc[(summary["variant"] == variant) & (summary["side"] == side)]
                figure, axis = plt.subplots(figsize=(15, 9))
                axis.axis("off")
                axis.set_title(f"{variant}; {'short' if side == -1 else 'long'} target — {len(dates)} matched dates; USD per $1m payout", pad=26)
                body, labels = [], []
                for row in chosen.itertuples():
                    labels.append(row.label)
                    body.append([f"{row.std_pnl*1e6:,.0f}", f"{row.mean_pnl*1e6:,.0f}", f"{row.median_pnl*1e6:,.0f}",
                        f"{row.mean_entry_cost*1e6:,.0f}", f"{-row.lower_5pct_mean_pnl*1e6:,.0f}", f"{100*row.cap_fraction:.1f}%",
                        f"[{row.std_difference_ci95_lower*1e6:,.0f}, {row.std_difference_ci95_upper*1e6:,.0f}]"])
                graphic = axis.table(cellText=body, rowLabels=labels,
                    colLabels=["Std net P&L", "Mean net", "Median net", "Entry cost", "5% tail loss", "Cap hits", "Std − unhedged\n95% interval"],
                    loc="center", cellLoc="right", colWidths=[.13, .11, .11, .11, .12, .1, .22])
                graphic.auto_set_font_size(False)
                graphic.set_fontsize(10)
                graphic.scale(1., 1.7)
                figure.subplots_adjust(left=.29, right=.98, top=.91, bottom=.12)
                figure.text(.035, .06, "Sample std uses ddof=1. Tail loss = negative empirical lower-5% mean P&L, averaged across products. Cap = any optimized position at ±1,000.", fontsize=10)
                figure.text(.035, .035, "Black is an uncapped analytic benchmark; its reported threshold exceedances are not binding optimizer-cap hits. Infinite MV solutions can be nonunique.", fontsize=10)
                pdf.savefig(figure)
                plt.close(figure)
        for side in (-1, 1):
            for objective in ("mean_variance", "power_exponential"):
                figure, axes = plt.subplots(1, 2, figsize=(15, 7))
                figure.suptitle(f"{'Short' if side == -1 else 'Long'} digital targets — {objective.replace('_', ' ')}", fontsize=17)
                for axis, metric, xlabel in zip(axes, ("std_pnl", "lower_5pct_mean_pnl"), ("Net P&L standard deviation", "Lower-5% mean loss"), strict=True):
                    for variant, color in (("ordinary", "tab:blue"), ("spot-orthogonal", "tab:orange")):
                        subset = summary.loc[(summary["variant"] == variant) & (summary["side"] == side)]
                        parametric = subset.loc[subset["objective"] == objective].sort_values("risk_aversion")
                        greek = subset.loc[subset["method"].str.startswith("DLV Greek")].sort_values("greek_budget")
                        sign = -1. if metric == "lower_5pct_mean_pnl" else 1.
                        for family, values, style in (("Parametric", parametric, "o-"), ("Greek", greek, "s--")):
                            axis.plot(sign*values[metric]*1e6, values["mean_entry_cost"]*1e6, style, color=color, label=f"{variant} {family}")
                            for row in values.itertuples():
                                annotation = ("inf" if np.isinf(row.risk_aversion) else f"{row.risk_aversion:g}") if family == "Parametric" else f"{100*row.greek_budget:.0f}%"
                                axis.annotate(annotation, (sign*getattr(row, metric)*1e6, row.mean_entry_cost*1e6), xytext=(3, 5), textcoords="offset points", fontsize=8)
                    subset = summary.loc[(summary["variant"] == "ordinary") & (summary["side"] == side)]
                    for method, color in (("unhedged", "tab:gray"), ("BS delta+vega", "tab:green")):
                        values = subset.loc[subset["method"] == method].iloc[0]
                        axis.scatter(sign*values[metric]*1e6, values["mean_entry_cost"]*1e6, color=color, s=80, label=method)
                    axis.set(xlabel=xlabel + " (USD per $1m)", ylabel="Mean entry cost (USD per $1m)")
                    axis.grid(alpha=.25)
                    axis.legend(fontsize=9)
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)
    metadata = {"pdf": str(pdf_path), "source_results_sha256": source_hashes, "coverage": coverage,
        "parametric_completion": audit_summaries, "bootstrap": {"draws": 2000, "block_sessions": 20, "seed": 20261106},
        "statistical_and_multiday_not_complete": True,
        "statistic": "average across products of product-specific across-date statistics",
        "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("report_objectives.py", "metrics.py")}}
    (output_directory / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("greek-ordinary-directory", "greek-orthogonal-directory", "parametric-ordinary-directory",
                 "parametric-orthogonal-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = build_objective_report(**vars(parser.parse_args()))
    print(result["pdf"], flush=True)


if __name__ == "__main__":
    main()
