"""Plot the matched reoptimized SANOS Greek entry-cost frontiers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

METRICS = (
    ("pnl_sd_kusd", "Out-of-sample P&L standard deviation"),
    ("mean_downside_loss_kusd", "Out-of-sample mean downside loss"),
)
METHODS = (
    ("global_cap", "Global dollar cap", "#2E7D32"),
    ("per_hedge_fraction", "Per-hedge cost fraction", "#7B3294"),
)


def run(*, optimized: Path, baselines: Path, output: Path) -> None:
    current = json.loads((optimized / "SUMMARY.json").read_text())
    baseline_report = json.loads(baselines.read_text())
    current_curves = {
        method: {row["cap_percentage"]: row for row in current["curves"] if row["method"] == method}
        for method, _, _ in METHODS
    }
    if any(set(curve) != set(range(0, 101, 10)) for curve in current_curves.values()):
        raise ValueError("Reoptimized curves do not contain every ten-percent cost cap")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    baseline_styles = {
        "Unhedged": ("#333333", "X"),
        "BS delta": ("#9A9A9A", "s"),
        "BS delta + vega": ("#8C564B", "P"),
    }
    for row_index, (metric, title) in enumerate(METRICS):
        for column_index, (method, label, color) in enumerate(METHODS):
            axis = axes[row_index, column_index]
            reoptimized = [current_curves[method][percentage] for percentage in range(100, -1, -10)]
            axis.plot(
                [point[metric] for point in reoptimized],
                [point["mean_entry_cost_kusd"] for point in reoptimized],
                color=color, linewidth=2.3, marker="o", markersize=5,
                label="SANOS Greek cost cap",
            )
            for percentage in (100, 50, 0):
                point = current_curves[method][percentage]
                offset = (6, -15) if percentage == 50 else (6, 8)
                axis.annotate(
                    f"{percentage}%", (point[metric], point["mean_entry_cost_kusd"]),
                    xytext=offset, textcoords="offset points", fontsize=8.5, color=color,
                )
            for name, (baseline_color, marker) in baseline_styles.items():
                point = baseline_report["baselines"][name]
                axis.scatter(
                    point[metric], point["mean_entry_cost_kusd"],
                    color=baseline_color, marker=marker, s=95, edgecolor="white",
                    linewidth=0.8, label=name, zorder=4,
                )
            axis.set_title(f"{label}: {title}", fontsize=12.5)
            axis.set_xlabel("Risk · USD thousands per USD 1m payout")
            axis.set_ylabel("Mean entry cost · USD thousands per USD 1m payout")
            axis.set_ylim(-0.15, 4.15)
            axis.grid(color="#DEE2E6", alpha=0.8)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
                  bbox_to_anchor=(0.5, 0.057), fontsize=9.5)
    figure.suptitle("SANOS Greek: optimize factor risk under an option entry-cost cap", fontsize=17, y=0.98)
    figure.text(
        0.5, 0.018,
        "44 April–May 2024 dates × 10 digitals; 10% cap steps. Spot is rebalanced freely at mid, "
        "including at zero option cost. Each constrained point is optimized independently.",
        ha="center", fontsize=9.3, color="#555555",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.19, top=0.90,
                           wspace=0.19, hspace=0.36)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "REOPTIMIZED_GREEK_COST_CAP_FRONTIER.png", dpi=180)
    figure.savefig(output / "REOPTIMIZED_GREEK_COST_CAP_FRONTIER.pdf")
    plt.close(figure)

    with (output / "RESULTS.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(current["curves"][0]))
        writer.writeheader()
        writer.writerows(current["curves"])

    statuses: Counter[str] = Counter()
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    budget_violations = []
    for trading_date in current["dates"]:
        rows = json.loads((optimized / "records" / f"{trading_date}.json").read_text())
        for row in rows:
            statuses[row["solver_status"]] += 1
            grouped[(row["origin_date"], row["claim"], row["method"])].append(row)
            budget_violations.append(row["entry_cost_kusd"] - row["budget_kusd"])
    monotonic_violations = 0
    for rows in grouped.values():
        rows.sort(key=lambda row: row["budget_kusd"])
        for lower, higher in zip(rows, rows[1:], strict=False):
            if higher["predicted_factor_variance_kusd2"] > (
                lower["predicted_factor_variance_kusd2"]
                + 1.0e-4 * max(1.0, lower["predicted_factor_variance_kusd2"])
            ):
                monotonic_violations += 1
    audit = {
        "cohort": current["dates"],
        "dataset_manifest_sha256": current["dataset_manifest_sha256"],
        "source_original_results": current["source_original_results"],
        "method": current["method"],
        "record_count": sum(statuses.values()),
        "solver_statuses": dict(statuses),
        "maximum_budget_violation_kusd": max(budget_violations),
        "predicted_factor_risk_monotonicity_violations": monotonic_violations,
        "baseline_report": str(baselines),
        "baselines": baseline_report["baselines"],
    }
    if monotonic_violations:
        raise RuntimeError("Predicted factor risk is not monotone in the cost cap")
    (output / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(optimized=arguments.optimized, baselines=arguments.baselines, output=arguments.output)


if __name__ == "__main__":
    main()
