from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from quantdata.sdlv.daily.front_spot.hedging.reports.all_generator_objective_summary import (
    summarize_all_objectives,
)

DATA_ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data")) / "experiments/hedging"
OUTPUT = DATA_ROOT / "frontier-plots-20261027-v2"
REPORT = DATA_ROOT / "generated-objective-grid-20261026-v1/ALL_GENERATORS.md"
GREEK_CAP_REPORT = DATA_ROOT / "sanos-greek-cost-cap-reoptimized-20261027-v1/SUMMARY.json"
PARAMETRIC_LABEL = "SANOS parametric"
HISTORICAL_LABEL = "SANOS historical"
HISTORICAL_PCA_LABEL = "SANOS historical PCA"
PARAMETRIC_OBJECTIVES = DATA_ROOT / "sanos-parametric-objectives-20261027-v1"
HISTORICAL_PCA_OBJECTIVES = DATA_ROOT / "sanos-normal-pca-objective-grid-20261028-v1/one-day"
OBJECTIVES = ("mean-variance", "entropy", "power-exponential", "cvar")
MODEL_COLORS = {
    PARAMETRIC_LABEL: "#A6761D",
    HISTORICAL_LABEL: "#0072B2",
    HISTORICAL_PCA_LABEL: "#7B3294",
    "FuNVol": "#D55E00",
    "VolGAN": "#009E73",
    "GROMM P": "#CC79A7",
    "GROMM Q": "#56B4E9",
    "AD-Seq-Vol": "#E69F00",
    "AD-Seq-Vol-FT": "#5E3C99",
}
BASELINE_STYLES = {
    "SANOS Greek": ("#2E7D32", "*"),
    "Unhedged": ("#333333", "X"),
    "BS delta": ("#9A9A9A", "s"),
    "BS delta + vega": ("#8C564B", "P"),
}
LAMBDA_MARKERS = {"1": "o", "10": "s", "100": "D", "1000": "v"}


def parse_report_table(section: str) -> dict[tuple[str, str, str], float]:
    body = REPORT.read_text().split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    rows = [line.strip("| ").split(" | ") for line in body.splitlines() if line.startswith("|")]
    models = rows[0][2:]
    values = {}
    for objective, risk_parameter, *cells in rows[2:]:
        for model, raw in zip(models, cells, strict=True):
            if "†" in raw:
                continue
            try:
                values[(objective, risk_parameter, model)] = float(raw)
            except ValueError:
                pass
    return values


def parse_coverage() -> dict[tuple[str, str, str], str]:
    body = REPORT.read_text().split("## Out-of-sample P&L standard deviation\n", 1)[1].split("\n## ", 1)[0]
    rows = [line.strip("| ").split(" | ") for line in body.splitlines() if line.startswith("|")]
    models = rows[0][2:]
    coverage = {}
    generator_names = {
        "FuNVol": "funvol", "VolGAN": "volgan", "GROMM P": "gromm-P",
        "GROMM Q": "gromm-Q", "AD-Seq-Vol": "diffusion-base",
        "AD-Seq-Vol-FT": "diffusion-ft",
    }
    for objective, risk_parameter, *values in rows[2:]:
        for model, value in zip(models, values, strict=True):
            if value == "failed":
                status = "Failed"
            elif value == "pending":
                if model == "Historical SANOS":
                    name = "cvar-oce" if objective == "cvar" else objective
                    folder = DATA_ROOT / "sanos-objective-grid-20261027-v1/historical-sanos" / f"{name}-lambda{risk_parameter}.0/records"
                else:
                    name = "cvar" if objective == "cvar" else objective
                    folder = DATA_ROOT / "generated-objective-grid-20261026-v1" / generator_names[model] / f"{name}-lambda{risk_parameter}/records"
                completed = len(tuple(folder.glob("2024-*.json")))
                status = f"{completed}/44" if completed else "Not run"
            elif "†" in value:
                status = "Cap-limited"
            else:
                status = "Complete"
            coverage[(objective, risk_parameter, model)] = status
    return coverage


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


def pca_record_pattern(objective: str, risk_parameter: str) -> Path:
    if objective == "mean-variance" and risk_parameter in {"1", "10", "100"}:
        return (DATA_ROOT / f"sanos-normal-pca-aprmay-20261026-v1/lambda{risk_parameter}"
                / "run/records/2024-*.json")
    return (HISTORICAL_PCA_OBJECTIVES / f"{objective}-lambda{risk_parameter}"
            / "run/records/2024-*.json")


def pca_cell_status(objective: str, risk_parameter: str) -> str:
    pattern = pca_record_pattern(objective, risk_parameter)
    files = sorted(glob.glob(str(pattern)))
    if len(files) == 44:
        for filename in files:
            rows = json.loads(Path(filename).read_text())["results"]
            if len(rows) != 10 or any(not row["converged"] or row["cap_hits"] for row in rows):
                return "Failed"
        return "Complete"
    if objective == "mean-variance" and risk_parameter in {"1", "10", "100"}:
        return f"{len(files)}/44" if files else "Not run"
    failure = pattern.parent.parent / "numerical-gate-failure.json"
    if failure.exists():
        failed_date = json.loads(failure.read_text())["origin_date"]
        if not (pattern.parent / f"{failed_date}.json").exists():
            return "Failed"
    return f"{len(files)}/44" if files else "Not run"


def load_points() -> tuple[dict[str, dict[str, dict[str, tuple[float, float]]]],
                           dict[str, tuple[float, float]]]:
    standard_deviations = parse_report_table("Out-of-sample P&L standard deviation")
    entry_costs = parse_report_table("Mean hedge entry cost")
    curves = {objective: defaultdict(dict) for objective in OBJECTIVES}
    for (objective, risk_parameter, model), standard_deviation in standard_deviations.items():
        key = objective, risk_parameter, model
        if objective in curves and risk_parameter in LAMBDA_MARKERS and key in entry_costs:
            display_model = HISTORICAL_LABEL if model == "Historical SANOS" else model
            curves[objective][display_model][risk_parameter] = (
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

    for objective in OBJECTIVES:
        for risk_parameter in LAMBDA_MARKERS:
            pattern = (PARAMETRIC_OBJECTIVES / objective / f"lambda{risk_parameter}"
                       / "records/2024-*.json")
            if len(glob.glob(str(pattern))) == 44:
                curves[objective][PARAMETRIC_LABEL][risk_parameter] = record_statistics(
                    str(pattern), model="sanos-parametric-ordinary"
                )
    for objective in OBJECTIVES:
        for risk_parameter in LAMBDA_MARKERS:
            if pca_cell_status(objective, risk_parameter) == "Complete":
                curves[objective][HISTORICAL_PCA_LABEL][risk_parameter] = record_statistics(
                    str(pca_record_pattern(objective, risk_parameter)),
                    model="sanos-normal-statistical-pca",
                )
    missing = []
    for objective, risk_parameter, model in standard_deviations.keys() & entry_costs.keys():
        if objective not in OBJECTIVES or risk_parameter not in LAMBDA_MARKERS:
            continue
        label = HISTORICAL_LABEL if model == "Historical SANOS" else model
        if risk_parameter not in curves[objective].get(label, {}):
            missing.append((objective, risk_parameter, model))
    if missing:
        raise RuntimeError(f"certified objective cells are absent from PDF: {sorted(missing)}")
    return curves, baselines


def add_frontier(ax: plt.Axes, objective: str,
                 curves: dict[str, dict[str, dict[str, tuple[float, float]]]],
                 baselines: dict[str, tuple[float, float]]) -> None:
    plotted = list(baselines.values())
    for model, risk_points in curves[objective].items():
        visible = [
            (risk_parameter, risk_points[risk_parameter])
            for risk_parameter in LAMBDA_MARKERS
            if risk_parameter in risk_points
            and all(np.isfinite(risk_points[risk_parameter]))
        ]
        if not visible:
            continue
        plotted.extend(point for _, point in visible)
        color = MODEL_COLORS[model]
        ax.plot([point[0] for _, point in visible],
                [point[1] for _, point in visible],
                color=color, linewidth=2.1, alpha=0.91, zorder=2)
        for risk_parameter, (standard_deviation, entry_cost) in visible:
            ax.scatter(standard_deviation, entry_cost,
                       marker=LAMBDA_MARKERS[risk_parameter], s=85,
                       color=color, edgecolor="white", linewidth=0.85, zorder=3)
    greek_cap = [point for point in json.loads(GREEK_CAP_REPORT.read_text())["curves"]
                 if point["method"] == "global_cap"]
    plotted.extend((point["pnl_sd_kusd"], point["mean_entry_cost_kusd"])
                   for point in greek_cap)
    ax.plot([point["pnl_sd_kusd"] for point in greek_cap],
            [point["mean_entry_cost_kusd"] for point in greek_cap],
            color=BASELINE_STYLES["SANOS Greek"][0],
            marker="o", markersize=3.5, linewidth=2.0, zorder=2)
    for name, (standard_deviation, entry_cost) in baselines.items():
        color, marker = BASELINE_STYLES[name]
        ax.scatter(standard_deviation, entry_cost, color=color, marker=marker,
                   s=175 if name == "SANOS Greek" else 130,
                   edgecolor="white", linewidth=1.0, zorder=5)
    ax.set_xlim(min(180.0, min(point[0] for point in plotted) * 0.95),
                max(500.0, max(point[0] for point in plotted) * 1.05))
    ax.set_ylim(-0.17, max(6.3, max(point[1] for point in plotted) * 1.08))
    ax.grid(color="#D9DEE3", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_title({"mean-variance": "Mean–variance", "entropy": "Entropy",
                  "power-exponential": "Power-exponential", "cvar": "CVaR OCE"}[objective],
                 fontsize=15, fontweight="semibold")
    ax.set_xlabel("Out-of-sample P&L SD  ·  USD thousands", fontsize=11)
    ax.set_ylabel("Mean entry cost  ·  USD thousands", fontsize=11)


def legend_handles(curves: dict[str, dict[str, dict[str, tuple[float, float]]]]) -> list[Line2D]:
    present_models = {model for objective in curves.values() for model, points in objective.items()
                      if points}
    handles = [Line2D([0], [0], color=BASELINE_STYLES["SANOS Greek"][0],
                      marker="o", markersize=4, label="SANOS Greek")]
    handles += [Line2D([0], [0], color=color, linewidth=2.2, label=model)
               for model, color in MODEL_COLORS.items() if model in present_models]
    handles += [Line2D([0], [0], color=color, marker=marker, linestyle="none",
                       markersize=10, label=name)
                for name, (color, marker) in BASELINE_STYLES.items() if name != "SANOS Greek"]
    handles += [Line2D([0], [0], color="#555555", marker=marker, linestyle="none",
                       markersize=8, markerfacecolor="#555555",
                       label=f"λ = {risk}")
                for risk, marker in LAMBDA_MARKERS.items()]
    return handles


def add_coverage_page(
    pdf: PdfPages,
    curves: dict[str, dict[str, dict[str, tuple[float, float]]]],
) -> None:
    coverage = parse_coverage()
    models = (
        "SANOS parametric", "SANOS historical", "SANOS historical PCA",
        "FuNVol", "VolGAN", "GROMM P", "GROMM Q", "AD-Seq-Vol", "AD-Seq-Vol-FT",
    )
    rows = []
    colors = []
    for objective in OBJECTIVES:
        for risk_parameter in LAMBDA_MARKERS:
            status = []
            for model in models:
                if risk_parameter in curves[objective].get(model, {}):
                    status.append("Complete")
                elif model == HISTORICAL_PCA_LABEL:
                    status.append(pca_cell_status(objective, risk_parameter))
                elif model == PARAMETRIC_LABEL:
                    status.append("Not run")
                else:
                    original_name = "Historical SANOS" if model == HISTORICAL_LABEL else model
                    status.append(coverage[(objective, risk_parameter, original_name)])
            rows.append([objective, risk_parameter, *status])
            colors.append([
                "#FFFFFF", "#FFFFFF",
                *(
                    "#DCF1E5" if value == "Complete" else
                    "#F9DFD8" if value == "Failed" else
                    "#F4D8AA" if value == "Cap-limited" else
                    "#FFF0CC" if "/44" in value else "#F3EBD7"
                    for value in status
                ),
            ])
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        cellColours=colors,
        colLabels=(
            "Objective", "λ", "SANOS param.", "SANOS hist.", "SANOS hist. PCA",
            "FuNVol", "VolGAN", "GROMM P", "GROMM Q", "AD-Seq-Vol", "AD-Seq-Vol-FT",
        ),
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=(0.02, 0.08, 0.96, 0.84),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D6DCE2")
        if row_index == 0:
            cell.set_facecolor("#DCE7EE")
            cell.set_text_props(weight="bold")
    snapshot = datetime.now(ZoneInfo("Europe/London")).strftime("%d %b %Y, %H:%M UK")
    fig.suptitle(f"Coverage of the 44-date objective grid · snapshot {snapshot}",
                 fontsize=17, y=0.98)
    fig.text(
        0.5, 0.045,
        "Complete = certified and plotted; n/44 = partial, not plotted; Failed = certification failed; "
        "Cap-limited = position limit reached; Not run = no saved dates.",
        ha="center", fontsize=10, color="#555555",
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    temporary_report = REPORT.with_suffix(".md.tmp")
    temporary_report.write_text(
        summarize_all_objectives(
            generated_root=DATA_ROOT / "generated-objective-grid-20261026-v1",
            historical_root=DATA_ROOT / "sanos-objective-grid-20261027-v1",
        ),
        encoding="utf-8",
    )
    temporary_report.replace(REPORT)
    curves, baselines = load_points()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    with PdfPages(OUTPUT / "COST_VERSUS_RISK.pdf") as pdf:
        for objective in OBJECTIVES:
            fig, ax = plt.subplots(figsize=(12.5, 7.2))
            add_frontier(ax, objective, curves, baselines)
            fig.legend(handles=legend_handles(curves), loc="center left",
                       bbox_to_anchor=(0.77, 0.52), frameon=False, fontsize=9.5)
            note = ("Only certified 44-date cells are plotted; λ = 1, 10, 100, 1000.\n"
                    "SANOS parametric is fitted per objective; Greek caps minimize factor risk separately.")
            fig.text(0.07, 0.025, note, fontsize=9.2, color="#555555")
            fig.suptitle("One-day digital hedges · April–May 2024 · 44 dates",
                         fontsize=16, y=0.98)
            fig.subplots_adjust(left=0.09, right=0.75, bottom=0.14, top=0.87)
            fig.savefig(OUTPUT / f"{objective}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
        add_coverage_page(pdf, curves)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for ax, objective in zip(axes.flat, OBJECTIVES, strict=True):
        add_frontier(ax, objective, curves, baselines)
    fig.suptitle("Entry cost versus out-of-sample risk · one-day digital hedges",
                 fontsize=18, y=0.985)
    fig.legend(handles=legend_handles(curves), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.06), frameon=False, fontsize=10)
    fig.text(0.5, 0.025,
             "44 April–May 2024 dates; ten targets. Risk aversions λ = 1, 10, 100, 1000 where complete. "
             "Greek cap optimizes factor risk separately; SANOS parametric appears only for independently fitted objectives.",
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
