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
from plot_frontiers import (
    BASELINE_STYLES,
    HISTORICAL_LABEL,
    HISTORICAL_PCA_LABEL,
    LAMBDA_MARKERS,
    MODEL_COLORS,
    PARAMETRIC_LABEL,
    PARAMETRIC_OBJECTIVES,
    pca_record_pattern,
)

DATA_ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data")) / "experiments/hedging"
OUTPUT = DATA_ROOT / "payoff-loss-plots-20261027-v2"
GREEK_CAP_REPORT = DATA_ROOT / "sanos-greek-cost-cap-reoptimized-20261027-v1/SUMMARY.json"


OBJECTIVES = ("mean-variance", "entropy", "power-exponential", "cvar")
GENERATORS = (HISTORICAL_LABEL, "FuNVol", "VolGAN", "GROMM P", "GROMM Q", "AD-Seq-Vol", "AD-Seq-Vol-FT")
RISK_PARAMETERS = ("1", "10", "100", "1000")
RISK_COLORS = {"1": "#0072B2", "10": "#E69F00", "100": "#009E73", "1000": "#CC79A7"}
GENERATED_ROOT = DATA_ROOT / "generated-objective-grid-20261026-v1"
HISTORICAL_ROOT = DATA_ROOT / "sanos-objective-grid-20261027-v1/historical-sanos"


def read_records(pattern: str) -> list[dict[str, object]] | None:
    paths = sorted(glob.glob(pattern))
    if len(paths) != 44:
        return None
    records = []
    for path in paths:
        document = json.loads(Path(path).read_text())
        rows = document if isinstance(document, list) else document["results"]
        if len(rows) != 10 or any(
            not row["converged"] or row["cap_hits"] for row in rows
        ):
            return None
        records.extend(rows)
    if len(records) != 440:
        raise RuntimeError(f"Incomplete objective cell: {pattern}")
    return records


def objective_records(model: str, objective: str,
                      risk_parameter: str) -> list[dict[str, object]] | None:
    risk_label = risk_parameter
    if model == HISTORICAL_LABEL:
        name = "cvar-oce" if objective == "cvar" else objective
        risk_label += ".0"
        folder = HISTORICAL_ROOT / f"{name}-lambda{risk_label}/records"
    else:
        generator = {"FuNVol": "funvol", "VolGAN": "volgan",
                     "GROMM P": "gromm-P", "GROMM Q": "gromm-Q",
                     "AD-Seq-Vol": "diffusion-base", "AD-Seq-Vol-FT": "diffusion-ft"}[model]
        name = "cvar" if objective == "cvar" else objective
        folder = GENERATED_ROOT / generator / f"{name}-lambda{risk_label}/records"
    return read_records(str(folder / "2024-*.json"))


def baseline_records() -> dict[str, list[dict[str, object]]]:
    full = DATA_ROOT / "sanos-full-aprmay-20261026-v1/results/2024-*.json"
    black = DATA_ROOT / "entry-only-replay-20261025-v1/bs-greeks-mean-variance-lambda10/records/2024-*.json"
    full_rows = read_records_with_multiple_models(str(full))
    black_rows = read_records(str(black))
    if black_rows is None:
        raise RuntimeError("The BS baseline cohort is incomplete")
    unhedged = [
        {**row, "hedged_pnl": row["unhedged_pnl"], "transaction_cost": 0.0,
         "actions": {}}
        for row in black_rows
    ]
    return {
        "Unhedged": unhedged,
        "BS delta": [row for row in full_rows if row["model"] == "bs-delta-only"],
        "BS delta + vega": black_rows,
        "SANOS Greek": [row for row in full_rows if row["model"] == "sanos-greek-ordinary"],
    }


def read_records_with_multiple_models(pattern: str) -> list[dict[str, object]]:
    paths = sorted(glob.glob(pattern))
    if len(paths) != 44:
        raise RuntimeError(f"Incomplete benchmark cohort: {pattern}")
    records = []
    for path in paths:
        records.extend(json.loads(Path(path).read_text()))
    return records


def supplemental_records() -> dict[str, dict[str, list[dict[str, object]]]]:
    extra = {objective: {} for objective in OBJECTIVES}
    for objective in OBJECTIVES:
        for risk_parameter in RISK_PARAMETERS:
            rows = read_records(str(pca_record_pattern(objective, risk_parameter)))
            if rows is not None:
                extra[objective][risk_parameter] = rows
    return extra


def metrics(records: list[dict[str, object]]) -> dict[str, float]:
    groups = defaultdict(list)
    for row in records:
        groups[row["claim"]].append(row)
    if len(groups) != 10 or {len(rows) for rows in groups.values()} != {44}:
        raise RuntimeError("All metrics require ten targets and 44 dates")
    by_target = []
    for rows in groups.values():
        pnl = np.asarray([1_000.0 * row["hedged_pnl"] / row["claim_quantity"]
                          for row in rows], dtype=float)
        cost = np.asarray([1_000.0 * row["transaction_cost"] / row["claim_quantity"]
                           for row in rows], dtype=float)
        by_target.append({
            "standard_deviation": float(np.std(pnl, ddof=1)),
            "mean_downside_loss": float(np.mean(np.maximum(-pnl, 0.0))),
            "mean_absolute_error": float(np.mean(np.abs(pnl))),
            "mean_entry_cost": float(np.mean(cost)),
        })
    return {key: float(np.mean([row[key] for row in by_target])) for key in by_target[0]}


def load_results() -> tuple[dict[str, dict[str, dict[str, list[dict[str, object]]]]],
                            dict[str, list[dict[str, object]]]]:
    results = {objective: {} for objective in OBJECTIVES}
    for objective in OBJECTIVES:
        for model in GENERATORS:
            cases = {}
            for risk_parameter in RISK_PARAMETERS:
                rows = objective_records(model, objective, risk_parameter)
                if rows is not None:
                    cases[risk_parameter] = rows
            if cases:
                results[objective][model] = cases
    extra = supplemental_records()
    for objective in OBJECTIVES:
        for risk_parameter in RISK_PARAMETERS:
            pattern = (PARAMETRIC_OBJECTIVES / objective / f"lambda{risk_parameter}"
                       / "records/2024-*.json")
            rows = read_records(str(pattern))
            if rows is not None:
                results[objective].setdefault(PARAMETRIC_LABEL, {})[risk_parameter] = rows
    for objective, cases in extra.items():
        if cases:
            results[objective][HISTORICAL_PCA_LABEL] = cases
    return results, baseline_records()


def add_loss_frontier(ax: plt.Axes, *, objective: str, loss_key: str,
                      results: dict[str, dict[str, dict[str, list[dict[str, object]]]]],
                      baselines: dict[str, list[dict[str, object]]]) -> None:
    plotted = []
    for model, cases in results[objective].items():
        color = MODEL_COLORS[model]
        finite = [(risk, metrics(cases[risk])) for risk in RISK_PARAMETERS
                  if risk in cases]
        if finite:
            plotted.extend(value for _, value in finite)
            ax.plot([value[loss_key] for _, value in finite],
                    [value["mean_entry_cost"] for _, value in finite],
                    color=color, linewidth=2.0, alpha=0.9, zorder=2)
            for risk, value in finite:
                ax.scatter(value[loss_key], value["mean_entry_cost"],
                           color=color, marker=LAMBDA_MARKERS[risk], s=80,
                           edgecolor="white", linewidth=0.8, zorder=3)
    greek_cap = [point for point in json.loads(GREEK_CAP_REPORT.read_text())["curves"]
                 if point["method"] == "global_cap"]
    cap_metric = ("mean_downside_loss_kusd" if loss_key == "mean_downside_loss"
                  else "mean_absolute_error_kusd")
    ax.plot([point[cap_metric] for point in greek_cap],
            [point["mean_entry_cost_kusd"] for point in greek_cap],
            color=BASELINE_STYLES["SANOS Greek"][0],
            marker="o", markersize=3.5, linewidth=2.0, zorder=2)
    for name, rows in baselines.items():
        color, marker = BASELINE_STYLES[name]
        value = metrics(rows)
        plotted.append(value)
        ax.scatter(value[loss_key], value["mean_entry_cost"],
                   color=color, marker=marker, s=165 if name == "SANOS Greek" else 125,
                   edgecolor="white", linewidth=1.0, zorder=5)
    ax.set_title({"mean-variance": "Mean–variance", "entropy": "Entropy",
                  "power-exponential": "Power-exponential", "cvar": "CVaR OCE"}[objective],
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Mean downside loss  ·  USD thousands" if loss_key == "mean_downside_loss"
                  else "Mean absolute hedging error  ·  USD thousands", fontsize=11)
    ax.set_ylabel("Mean entry cost  ·  USD thousands", fontsize=11)
    ax.set_ylim(-0.15, max(6.3, max(value["mean_entry_cost"] for value in plotted) * 1.08))
    ax.set_xlim(0.0, max(280.0 if loss_key == "mean_downside_loss" else 500.0,
                         max(value[loss_key] for value in plotted) * 1.08))
    ax.grid(color="#D9DEE3", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def plot_loss_frontier(results: dict[str, dict[str, dict[str, list[dict[str, object]]]]],
                       baselines: dict[str, list[dict[str, object]]],
                       *, loss_key: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for ax, objective in zip(axes.flat, OBJECTIVES, strict=True):
        add_loss_frontier(ax, objective=objective, loss_key=loss_key,
                          results=results, baselines=baselines)
    label = ("mean downside loss E[max(−PnL, 0)]" if loss_key == "mean_downside_loss"
             else "mean absolute hedging error E[|PnL|]")
    fig.suptitle(f"Entry cost versus {label} · one-day digital hedges", fontsize=16, y=0.985)
    handles = [Line2D([0], [0], color=BASELINE_STYLES["SANOS Greek"][0],
                      marker="o", markersize=4, label="SANOS Greek")]
    handles += [Line2D([0], [0], color=color, linewidth=2.2, label=model)
               for model, color in MODEL_COLORS.items()
               if any(model in models and models[model] for models in results.values())]
    handles += [Line2D([0], [0], color=color, marker=marker, linestyle="none",
                       markersize=10, label=name)
                for name, (color, marker) in BASELINE_STYLES.items() if name != "SANOS Greek"]
    handles += [Line2D([0], [0], color="#555555", marker=marker, linestyle="none",
                       markersize=8, markerfacecolor="#555555",
                       label=f"λ = {risk}")
                for risk, marker in LAMBDA_MARKERS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.06), frameon=False, fontsize=10)
    fig.text(0.5, 0.025,
             "44 April–May 2024 dates; ten targets. Loss uses costed out-of-sample long-claim P&L. "
             "Risk aversions λ = 1, 10, 100, 1000 where complete. Greek caps optimize factor risk separately; "
             "SANOS parametric appears only where independently fitted for the plotted objective.",
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.92,
                        wspace=0.22, hspace=0.33)
    fig.savefig(OUTPUT / f"{filename}.png", dpi=180)
    fig.savefig(OUTPUT / f"{filename}.pdf")
    plt.close(fig)


def payoff_values(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    payoff = np.asarray([row["payoff"] for row in rows], dtype=float)
    terminal = np.asarray([
        row["market_price"]
        - (row["hedged_pnl"] - row["unhedged_pnl"] + 2.0 * row["transaction_cost"])
        / row["claim_quantity"]
        for row in rows
    ], dtype=float)
    return payoff, terminal


def plot_payoff_panels(results: dict[str, dict[str, dict[str, list[dict[str, object]]]]],
                      baselines: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    audit = {}
    with PdfPages(OUTPUT / "HEDGE_VERSUS_PAYOFF.pdf") as pdf:
        for model in (PARAMETRIC_LABEL, HISTORICAL_PCA_LABEL) + GENERATORS:
            objectives = OBJECTIVES
            fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
            for ax, objective in zip(axes.flat, OBJECTIVES, strict=True):
                ax.plot([-0.15, 1.15], [-0.15, 1.15], color="#222222",
                        linestyle="--", linewidth=1.5, zorder=1)
                ax.set_xlim(-0.17, 1.17)
                ax.set_ylim(-1.35, 2.1)
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.grid(color="#DFE3E6", linewidth=0.7, alpha=0.8)
                ax.set_axisbelow(True)
                ax.set_title({"mean-variance": "Mean–variance", "entropy": "Entropy",
                              "power-exponential": "Power-exponential", "cvar": "CVaR OCE"}[objective],
                             fontsize=14, fontweight="bold")
                if objective not in objectives or model not in results[objective]:
                    ax.text(0.5, 0.5, "Not applicable", transform=ax.transAxes,
                            ha="center", va="center", fontsize=15, color="#666666")
                    continue
                cells = results[objective][model]
                status = []
                for risk_parameter in RISK_PARAMETERS:
                    if risk_parameter not in cells:
                        continue
                    payoff, terminal = payoff_values(cells[risk_parameter])
                    random = np.random.default_rng(20261027)
                    jitter = random.uniform(-0.018, 0.018, size=len(payoff))
                    shift = {"1": -0.09, "10": -0.03, "100": 0.03, "1000": 0.09}[risk_parameter]
                    ax.scatter(payoff + shift + jitter, terminal, s=12,
                               color=RISK_COLORS[risk_parameter], alpha=0.22,
                               linewidth=0.0, rasterized=True)
                    violations = float(np.mean(terminal < payoff - 1.0e-10))
                    required_top_up = float(np.max(np.maximum(payoff - terminal, 0.0)))
                    off_chart = int(np.count_nonzero((terminal < -1.35) | (terminal > 2.1)))
                    status.append(f"λ={risk_parameter}: {violations:.0%} below" +
                                  (f", {off_chart} off-chart" if off_chart else ""))
                    audit[f"{model}|{objective}|{risk_parameter}"] = {
                        "observations": len(payoff),
                        "observed_superhedge_violation_fraction": violations,
                        "maximum_observed_top_up_per_unit_payout": required_top_up,
                        "off_chart_observations": off_chart,
                    }
                ax.text(0.025, 0.97, "\n".join(status), transform=ax.transAxes,
                        ha="left", va="top", fontsize=9.5,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84})
                ax.set_xlabel("Digital payout per unit claim", fontsize=10.5)
                ax.set_ylabel("Opposite hedge terminal value", fontsize=10.5)
            fig.suptitle(f"{model}: opposite long-claim hedge versus digital payout",
                         fontsize=17, y=0.985)
            legend = [Line2D([0], [0], color=RISK_COLORS[risk], marker="o",
                             linestyle="none", label=f"λ = {risk}")
                      for risk in RISK_PARAMETERS]
            fig.legend(handles=legend, loc="lower center", ncol=3,
                       bbox_to_anchor=(0.5, 0.082), frameon=False, fontsize=10)
            fig.text(0.5, 0.025,
                     "Out-of-sample market transitions; 44 dates × 10 digitals per complete cell. "
                     "Below diagonal violates short-side coverage at the market premium. "
                     "These are reversed long-side actions, not separately optimized superhedges.",
                     ha="center", fontsize=9.5, color="#555555")
            fig.subplots_adjust(left=0.07, right=0.97, bottom=0.17, top=0.93,
                                wspace=0.18, hspace=0.3)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
        for ax, (name, rows) in zip(axes.flat, baselines.items(), strict=True):
            payoff, terminal = payoff_values(rows)
            random = np.random.default_rng(20261027)
            ax.scatter(payoff + random.uniform(-0.025, 0.025, len(payoff)), terminal,
                       color=BASELINE_STYLES[name][0], s=16, alpha=0.3, linewidth=0.0)
            ax.plot([-0.15, 1.15], [-0.15, 1.15], color="#222222", linestyle="--")
            ax.set_xlim(-0.17, 1.17)
            ax.set_ylim(-1.35, 2.1)
            ax.grid(color="#DFE3E6", linewidth=0.7, alpha=0.8)
            ax.set_title(name, fontsize=14, fontweight="bold")
            ax.set_xlabel("Digital payout per unit claim")
            ax.set_ylabel("Opposite hedge terminal value")
            audit[f"{name}|baseline"] = {
                "observations": len(payoff),
                "observed_superhedge_violation_fraction": float(np.mean(terminal < payoff - 1e-10)),
                "maximum_observed_top_up_per_unit_payout": float(np.max(np.maximum(payoff - terminal, 0))),
            }
        fig.suptitle("Fixed baselines: opposite hedge versus digital payout", fontsize=17, y=0.985)
        fig.text(0.5, 0.025,
                 "Below the dashed diagonal: the opposite hedge does not cover the digital payout. "
                 "No short-side superhedge was optimized.",
                 ha="center", fontsize=10, color="#555555")
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.11, top=0.93,
                            wspace=0.18, hspace=0.3)
        pdf.savefig(fig)
        plt.close(fig)
    return audit


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    results, baselines = load_results()
    plot_loss_frontier(results, baselines, loss_key="mean_downside_loss",
                       filename="COST_VERSUS_MEAN_DOWNSIDE_LOSS")
    plot_loss_frontier(results, baselines, loss_key="mean_absolute_error",
                       filename="COST_VERSUS_MEAN_ABSOLUTE_ERROR")
    audit = plot_payoff_panels(results, baselines)
    metrics_by_cell = {
        f"{model}|{objective}|{risk}": metrics(rows)
        for objective, models in results.items()
        for model, cases in models.items()
        for risk, rows in cases.items()
    }
    metrics_by_cell.update({f"{name}|baseline": metrics(rows)
                            for name, rows in baselines.items()})
    payload = {
        "cohort": "April–May 2024, 44 dates and ten one-day digitals",
        "units": "USD thousands per USD 1 million digital payout",
        "mean_loss_definition": "E[max(-costed_long_claim_pnl, 0)]",
        "mean_absolute_error_definition": "E[abs(costed_long_claim_pnl)]",
        "short_candidate": "reverse the fitted long-claim action, fund with market digital premium, pay entry costs",
        "short_terminal_formula": "market_price - (hedged_pnl - unhedged_pnl + 2 * transaction_cost) / claim_quantity",
        "greek_cost_cap_report": str(GREEK_CAP_REPORT),
        "risk_parameters_plotted": list(RISK_PARAMETERS),
        "metrics": metrics_by_cell,
        "observed_payoff_gaps": audit,
    }
    (OUTPUT / "PAYOFF_AND_LOSS_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
