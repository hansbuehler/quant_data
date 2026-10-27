from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

DATA_ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data")) / "experiments/hedging"
OUTPUT = DATA_ROOT / "frontier-plots-20261027-v1"
REPORT = DATA_ROOT / "generated-objective-grid-20261026-v1/ALL_GENERATORS.md"
GREEK_CAP_REPORT = DATA_ROOT / "sanos-greek-cost-cap-20261027-v1/RESULTS.json"
OBJECTIVES = ("mean-variance", "entropy", "power-exponential", "cvar")
MODEL_COLORS = {
    "Historical SANOS": "#0072B2",
    "FuNVol": "#D55E00",
    "VolGAN": "#009E73",
    "GROMM P": "#CC79A7",
    "GROMM Q": "#56B4E9",
    "SANOS parametric": "#A6761D",
    "Historical-PCA": "#7B3294",
}
BASELINE_STYLES = {
    "Unhedged": ("#333333", "X"),
    "BS delta": ("#9A9A9A", "s"),
    "BS delta + vega": ("#8C564B", "P"),
    "SANOS Greek": ("#2E7D32", "*")
}
LAMBDA_MARKERS = {"1": "o", "10": "s", "100": "D"}


def parse_report_table(section: str) -> dict[tuple[str, str, str], float]:
    body = REPORT.read_text().split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    rows = [line.strip("| ").split(" | ") for line in body.splitlines() if line.startswith("|")]
    models = rows[0][2:]
    values = {}
    for row in rows[2:]:
        objective, risk_parameter = row[:2]
        for model, raw in zip(models, row[2:], strict=True):
            try:
                values[(objective, risk_parameter, model)] = float(raw.replace("†", ""))
            except ValueError:
                pass
    return values


def record_statistics(pattern: str, *, model: str | None = None,
                      pnl_key: str = "hedged_pnl") -> tuple[float, float]:
    pnl_by_target = defaultdict(list)
    cost_by_target = defaultdict(list)
    files = sorted(glob.glob(pattern))
    for filename in files:
        document = json.loads(Path(filename).read_text())
        rows = document if isinstance(document, list) else document["results"]
        for row in rows:
            if model is not None and row["model"] != model:
                continue
            target = row["claim"]
            scale = 1000.0 / row["claim_quantity"]
            pnl_by_target[target].append(row[pnl_key] * scale)
            cost_by_target[target].append(
                0.0 if pnl_key == "unhedged_pnl" else row["transaction_cost"] * scale
            )
    lengths = {len(values) for values in pnl_by_target.values()}
    if len(files) != 44 or len(pnl_by_target) != 10 or lengths != {44}:
        raise ValueError(f"Incomplete 44-date, ten-target cohort: {pattern}, {lengths}")
    standard_deviation = float(np.mean([
        np.std(values, ddof=1) for values in pnl_by_target.values()
    ]))
    mean_entry_cost = float(np.mean([
        np.mean(values) for values in cost_by_target.values()
    ]))
    return standard_deviation, mean_entry_cost


def load_points() -> tuple[dict[str, dict[str, dict[str, tuple[float, float]]]],
                           dict[str, tuple[float, float]]]:
    standard_deviations = parse_report_table("Out-of-sample P&L standard deviation")
    entry_costs = parse_report_table("Mean hedge entry cost")
    curves = {objective: defaultdict(dict) for objective in OBJECTIVES}
    for (objective, risk_parameter, model), standard_deviation in standard_deviations.items():
        key = objective, risk_parameter, model
        if objective in curves and risk_parameter in LAMBDA_MARKERS and key in entry_costs:
            curves[objective][model][risk_parameter] = (
                standard_deviation, entry_costs[key]
            )

    full_records = str(DATA_ROOT / "sanos-full-aprmay-20261026-v1/results/2024-*.json")
    black_records = str(DATA_ROOT / "entry-only-replay-20261025-v1/bs-greeks-mean-variance-lambda10/records/2024-*.json")
    baselines = {
        "Unhedged": record_statistics(black_records, pnl_key="unhedged_pnl"),
        "BS delta": record_statistics(full_records, model="bs-delta-only"),
        "BS delta + vega": record_statistics(black_records, model="bs-delta-vega"),
        "SANOS Greek": record_statistics(full_records, model="sanos-greek-ordinary"),
    }

    parametric_patterns = {
        "1": DATA_ROOT / "lambda-1-10-aprmay-20261026-v1/sanos-greek-parametric-lambda1/run/2024-*.json",
        "10": DATA_ROOT / "sanos-full-aprmay-20261026-v1/results/2024-*.json",
        "100": DATA_ROOT / "sanos-lambda100-aprmay-20261026-v1/sanos-greek-parametric/run/2024-*.json",
    }
    for risk_parameter, pattern in parametric_patterns.items():
        curves["mean-variance"]["SANOS parametric"][risk_parameter] = record_statistics(
            str(pattern), model="sanos-parametric-ordinary"
        )
    for risk_parameter in LAMBDA_MARKERS:
        directory = f"lambda{risk_parameter}"
        pattern = DATA_ROOT / f"sanos-normal-pca-aprmay-20261026-v1/{directory}/run/records/2024-*.json"
        curves["mean-variance"]["Historical-PCA"][risk_parameter] = record_statistics(
            str(pattern), model="sanos-normal-statistical-pca"
        )
    return curves, baselines


def add_frontier(ax: plt.Axes, objective: str,
                 curves: dict[str, dict[str, dict[str, tuple[float, float]]]],
                 baselines: dict[str, tuple[float, float]]) -> None:
    for model, risk_points in curves[objective].items():
        visible = [
            (risk_parameter, risk_points[risk_parameter])
            for risk_parameter in ("1", "10", "100")
            if risk_parameter in risk_points
            and 180 <= risk_points[risk_parameter][0] <= 500
            and 0 <= risk_points[risk_parameter][1] <= 6.3
        ]
        if not visible:
            continue
        color = MODEL_COLORS[model]
        ax.plot([point[0] for _, point in visible],
                [point[1] for _, point in visible],
                color=color, linewidth=2.1, alpha=0.91, zorder=2)
        for risk_parameter, (standard_deviation, entry_cost) in visible:
            ax.scatter(standard_deviation, entry_cost,
                       marker=LAMBDA_MARKERS[risk_parameter], s=85,
                       color=color, edgecolor="white", linewidth=0.85, zorder=3)
    greek_cap = json.loads(GREEK_CAP_REPORT.read_text())["curves"]["global_cap"]
    ax.plot([point["pnl_sd_kusd"] for point in greek_cap],
            [point["mean_entry_cost_kusd"] for point in greek_cap],
            color=BASELINE_STYLES["SANOS Greek"][0], linestyle="--",
            marker="o", markersize=3.5, linewidth=1.9, zorder=2)
    for name, (standard_deviation, entry_cost) in baselines.items():
        color, marker = BASELINE_STYLES[name]
        ax.scatter(standard_deviation, entry_cost, color=color, marker=marker,
                   s=175 if name == "SANOS Greek" else 130,
                   edgecolor="white", linewidth=1.0, zorder=5)
    ax.set_xlim(180, 500)
    ax.set_ylim(-0.17, 6.3)
    ax.set_xticks([200, 250, 300, 350, 400, 450, 500])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.grid(color="#D9DEE3", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_title({"mean-variance": "Mean–variance", "entropy": "Entropy",
                  "power-exponential": "Power-exponential", "cvar": "CVaR OCE"}[objective],
                 fontsize=15, fontweight="semibold")
    ax.set_xlabel("Out-of-sample P&L SD  ·  USD thousands", fontsize=11)
    ax.set_ylabel("Mean entry cost  ·  USD thousands", fontsize=11)


def legend_handles() -> list[Line2D]:
    handles = [Line2D([0], [0], color=color, linewidth=2.2, label=model)
               for model, color in MODEL_COLORS.items()]
    handles += [Line2D([0], [0], color=BASELINE_STYLES["SANOS Greek"][0],
                       linestyle="--", marker="o", markersize=4,
                       label="SANOS Greek cost cap")]
    handles += [Line2D([0], [0], color=color, marker=marker, linestyle="none",
                       markersize=10, label=name)
                for name, (color, marker) in BASELINE_STYLES.items()]
    handles += [Line2D([0], [0], color="#555555", marker=marker, linestyle="none",
                       markersize=8, markerfacecolor="#555555",
                       label=f"λ = {risk}")
                for risk, marker in LAMBDA_MARKERS.items()]
    return handles


def main() -> None:
    curves, baselines = load_points()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    with PdfPages(OUTPUT / "COST_VERSUS_RISK.pdf") as pdf:
        for objective in OBJECTIVES:
            fig, ax = plt.subplots(figsize=(12.5, 7.2))
            add_frontier(ax, objective, curves, baselines)
            fig.legend(handles=legend_handles(), loc="center left",
                       bbox_to_anchor=(0.77, 0.52), frameon=False, fontsize=9.5)
            note = ("Finite risk aversions λ = 1, 10, 100; Greek cap scales existing "
                    "positions rather than reoptimizing them. Diffusion runs are pending.")
            fig.text(0.07, 0.015, note, fontsize=9.5, color="#555555")
            fig.suptitle("One-day digital hedges · April–May 2024 · 44 dates",
                         fontsize=16, y=0.98)
            fig.subplots_adjust(left=0.09, right=0.75, bottom=0.14, top=0.87)
            fig.savefig(OUTPUT / f"{objective}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for ax, objective in zip(axes.flat, OBJECTIVES, strict=True):
        add_frontier(ax, objective, curves, baselines)
    fig.suptitle("Entry cost versus out-of-sample risk · one-day digital hedges",
                 fontsize=18, y=0.985)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.06), frameon=False, fontsize=10)
    fig.text(0.5, 0.025,
             "44 April–May 2024 dates; ten targets. Risk aversions λ = 1, 10, 100. "
             "Dashed Greek curve scales existing positions under a common dollar cap.",
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.92,
                        wspace=0.22, hspace=0.33)
    fig.savefig(OUTPUT / "COST_VERSUS_RISK_OVERVIEW.png", dpi=180)
    plt.close(fig)

    provenance = {
        "cohort": "2024-04-01 to 2024-05-31; 44 dates; ten 1D digitals per date",
        "units": "USD thousands per USD 1 million digital payout",
        "aggregation": "mean of ten target-specific out-of-sample P&L sample SDs; mean entry cost",
        "source_report": str(REPORT),
        "greek_cost_cap_report": str(GREEK_CAP_REPORT),
        "baselines": baselines,
        "points": curves,
        "risk_parameters_plotted": list(LAMBDA_MARKERS),
    }
    (OUTPUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
