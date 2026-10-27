"""Replay the matched SANOS Greek hedge under smaller entry-cost caps."""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data")) / "experiments/hedging"
OUTPUT = ROOT / "sanos-greek-cost-cap-20261027-v1"
GREEK_SOURCE = ROOT / "sanos-full-aprmay-20261026-v1/results/2024-*.json"
BLACK_SOURCE = (
    ROOT / "entry-only-replay-20261025-v1"
    / "bs-greeks-mean-variance-lambda10/records/2024-*.json"
)
RISK_UNITS = "USD thousands per USD 1 million digital payout"


def read_records(pattern: Path, *, model: str | None = None) -> list[dict]:
    files = sorted(glob.glob(str(pattern)))
    if len(files) != 44:
        raise ValueError(f"Expected 44 daily result files, got {len(files)}: {pattern}")
    rows = []
    for filename in files:
        payload = json.loads(Path(filename).read_text())
        daily_rows = payload if isinstance(payload, list) else payload["results"]
        selected = [row for row in daily_rows if model is None or row["model"] == model]
        if len(selected) != 10:
            raise ValueError(f"Expected ten {model} targets in {filename}")
        rows.extend(selected)
    if len({row["claim"] for row in rows}) != 10:
        raise ValueError("The 44-day cohort must have ten common digital targets")
    if any(not row["converged"] or row["cap_hits"] for row in rows):
        raise ValueError("The original hedge contains failed or capped results")
    return rows


def normalized_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([row["claim"] for row in rows])
    quantities = np.asarray([row["claim_quantity"] for row in rows], dtype=float)
    if np.any(quantities <= 0.0):
        raise ValueError("Claim quantities must be positive")
    scale = 1_000.0 / quantities
    hedged = np.asarray([row["hedged_pnl"] for row in rows], dtype=float) * scale
    unhedged = np.asarray([row["unhedged_pnl"] for row in rows], dtype=float) * scale
    costs = np.asarray([row["transaction_cost"] for row in rows], dtype=float) * scale
    if np.any(~np.isfinite(hedged + unhedged + costs)) or np.any(costs < 0.0):
        raise ValueError("Normalized results must be finite with non-negative costs")
    return labels, hedged, unhedged, costs


def metrics(labels: np.ndarray, pnl: np.ndarray, costs: np.ndarray) -> dict[str, float]:
    by_target = defaultdict(list)
    for label, index in zip(labels, range(len(labels)), strict=True):
        by_target[label].append(index)
    if len(by_target) != 10 or {len(indices) for indices in by_target.values()} != {44}:
        raise ValueError("All statistics require 44 observations for each target")
    return {
        "pnl_sd_kusd": float(np.mean([
            np.std(pnl[indices], ddof=1) for indices in by_target.values()
        ])),
        "mean_downside_loss_kusd": float(np.mean([
            np.mean(np.maximum(-pnl[indices], 0.0)) for indices in by_target.values()
        ])),
        "mean_entry_cost_kusd": float(np.mean(costs)),
    }


def replay_caps() -> dict[str, object]:
    greek_rows = read_records(GREEK_SOURCE, model="sanos-greek-ordinary")
    labels, original_pnl, unhedged_pnl, original_cost = normalized_arrays(greek_rows)
    maximum_cost = float(np.max(original_cost))
    if maximum_cost <= 0.0:
        raise ValueError("The uncapped Greek hedge must have positive entry cost")

    curves: dict[str, list[dict[str, float]]] = {"global_cap": [], "per_hedge_fraction": []}
    for percentage in range(100, -1, -10):
        fraction = percentage / 100.0
        cap = fraction * maximum_cost
        global_scale = np.minimum(
            1.0,
            np.divide(cap, original_cost, out=np.ones_like(original_cost), where=original_cost > 0.0),
        )
        for method, scale in (
            ("global_cap", global_scale),
            ("per_hedge_fraction", np.full_like(original_cost, fraction)),
        ):
            pnl = unhedged_pnl + scale * (original_pnl - unhedged_pnl)
            costs = scale * original_cost
            if method == "global_cap" and np.any(costs > cap + 1.0e-10):
                raise RuntimeError("A replayed trade exceeds its entry-cost cap")
            curves[method].append({
                "cap_percentage": percentage,
                "global_cost_cap_kusd": cap,
                "binding_fraction": float(np.mean(scale < 1.0 - 1.0e-12)),
                **metrics(labels, pnl, costs),
            })

    if not np.isclose(curves["global_cap"][0]["pnl_sd_kusd"], 215.7200, atol=0.01):
        raise RuntimeError("The full-cost point does not reproduce the published Greek result")
    if not np.isclose(curves["global_cap"][-1]["pnl_sd_kusd"], 435.8845, atol=0.01):
        raise RuntimeError("The zero-cost point does not reproduce the unhedged result")

    delta_rows = read_records(GREEK_SOURCE, model="bs-delta-only")
    black_rows = read_records(BLACK_SOURCE)
    delta_labels, delta_pnl, _, delta_costs = normalized_arrays(delta_rows)
    black_labels, black_pnl, _, black_costs = normalized_arrays(black_rows)
    baseline_rows = {
        "Unhedged": (labels, unhedged_pnl, np.zeros_like(original_cost)),
        "BS delta": (delta_labels, delta_pnl, delta_costs),
        "BS delta + vega": (black_labels, black_pnl, black_costs),
        "SANOS Greek": (labels, original_pnl, original_cost),
    }
    baselines = {
        name: metrics(baseline_labels, pnl, costs)
        for name, (baseline_labels, pnl, costs) in baseline_rows.items()
    }
    return {
        "cohort": "2024-04-01 through 2024-05-31; 44 dates; ten one-day digitals",
        "units": RISK_UNITS,
        "source": str(GREEK_SOURCE),
        "maximum_original_entry_cost_kusd": maximum_cost,
        "method": (
            "Scale each existing Greek spot and option action by min(1, budget/original cost) "
            "for the global cap, or by the stated fraction for each-hedge caps. "
            "Entry costs and hedge gains scale linearly; no Greek exposures are reoptimized."
        ),
        "curves": curves,
        "baselines": baselines,
    }


def plot(payload: dict[str, object]) -> None:
    curves = payload["curves"]
    baselines = payload["baselines"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    styles = {
        "global_cap": ("#0072B2", "Global dollar cap"),
        "per_hedge_fraction": ("#CC79A7", "Same fraction of each hedge"),
    }
    baseline_styles = {
        "Unhedged": ("#333333", "X"),
        "BS delta": ("#9A9A9A", "s"),
        "BS delta + vega": ("#8C564B", "P"),
    }
    for axis, x_key, title in (
        (axes[0], "pnl_sd_kusd", "Out-of-sample P&L standard deviation"),
        (axes[1], "mean_downside_loss_kusd", "Out-of-sample mean downside loss"),
    ):
        for method, (color, label) in styles.items():
            points = curves[method]
            axis.plot(
                [point[x_key] for point in points],
                [point["mean_entry_cost_kusd"] for point in points],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=5,
                label=label,
                zorder=3,
            )
            for point in points:
                if point["cap_percentage"] in (100, 50, 0) and method == "per_hedge_fraction":
                    offset = (-24, 8) if point["cap_percentage"] == 0 else (6, 8)
                    axis.annotate(
                        f"{point['cap_percentage']}%",
                        (point[x_key], point["mean_entry_cost_kusd"]),
                        xytext=offset,
                        textcoords="offset points",
                        fontsize=9,
                        color=color,
                    )
        for name, (color, marker) in baseline_styles.items():
            point = baselines[name]
            axis.scatter(
                point[x_key], point["mean_entry_cost_kusd"],
                color=color, marker=marker, s=110,
                edgecolor="white", linewidth=0.9,
                label=name, zorder=4,
            )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("Risk · USD thousands per USD 1m payout")
        axis.set_ylabel("Mean entry cost · USD thousands per USD 1m payout")
        axis.set_ylim(-0.12, 4.1)
        axis.grid(color="#DEE2E6", alpha=0.8)
        axis.set_axisbelow(True)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=5, frameon=False,
                  bbox_to_anchor=(0.5, 0.07), fontsize=10)
    figure.suptitle("SANOS Greek: entry-cost caps versus realized hedge risk", fontsize=18)
    figure.text(
        0.5, 0.025,
        "44 April–May 2024 dates · ten 1D digital targets · global cap starts at USD 9.01k per USD 1m payout "
        "and falls in 10% steps · actions are scaled, not reoptimized",
        ha="center", fontsize=9.4, color="#555555",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.87, wspace=0.18)
    figure.savefig(OUTPUT / "SANOS_GREEK_COST_CAP_FRONTIER.png", dpi=180)
    figure.savefig(OUTPUT / "SANOS_GREEK_COST_CAP_FRONTIER.pdf")
    plt.close(figure)


def main() -> None:
    payload = replay_caps()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot(payload)
    (OUTPUT / "RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUTPUT / "RESULTS.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", *payload["curves"]["global_cap"][0]])
        writer.writeheader()
        for method, points in payload["curves"].items():
            for point in points:
                writer.writerow({"method": method, **point})
    print(json.dumps({
        "maximum_original_entry_cost_kusd": payload["maximum_original_entry_cost_kusd"],
        "global_cap_curve": payload["curves"]["global_cap"],
        "per_hedge_fraction_curve": payload["curves"]["per_hedge_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
