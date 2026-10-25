"""Summarize matched one-day entry-only hedges and formal exit costs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _records(directory: Path) -> list[dict[str, object]]:
    paths = sorted((directory / "records").glob("*.json"))
    if len(paths) != 44:
        raise RuntimeError(f"{directory}: expected 44 completed origin dates, got {len(paths)}")
    records: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload["results"]
        if len(rows) != 10 or any(
            not row["converged"] or row["cap_hits"] for row in rows
        ):
            raise RuntimeError(f"{path}: incomplete or uncertified target book")
        if any("formal_exit_cost" not in row["diagnostics"] for row in rows):
            raise RuntimeError(f"{path}: formal exit diagnostic is missing")
        records.extend(rows)
    return records


def _summary(records: list[dict[str, object]]) -> dict[str, float]:
    by_claim: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_claim[str(record["claim"])].append(record)
    if len(by_claim) != 10 or any(len(rows) != 44 for rows in by_claim.values()):
        raise RuntimeError("matched book must have ten targets on all 44 dates")
    metrics: dict[str, list[float]] = defaultdict(list)
    for rows in by_claim.values():
        pnl = np.asarray([row["hedged_pnl"] for row in rows], dtype=np.float64)
        costs = np.asarray([row["transaction_cost"] for row in rows], dtype=np.float64)
        exits = np.asarray(
            [row["diagnostics"]["formal_exit_cost"] for row in rows], dtype=np.float64
        )
        spot_actions = np.asarray(
            [row["actions"].get("spot", 0.0) for row in rows], dtype=np.float64
        )
        metrics["mean_pnl"].append(float(np.mean(pnl)))
        metrics["median_pnl"].append(float(np.median(pnl)))
        metrics["sd_pnl"].append(float(np.std(pnl, ddof=1)))
        metrics["scaled_mad_pnl"].append(
            float(1.4826022185 * np.median(np.abs(pnl - np.median(pnl))))
        )
        metrics["mean_absolute_pnl"].append(float(np.mean(np.abs(pnl))))
        metrics["mean_entry_cost"].append(float(np.mean(costs)))
        metrics["mean_formal_exit_cost"].append(float(np.mean(exits)))
        metrics["mean_abs_spot_action"].append(float(np.mean(np.abs(spot_actions))))
        metrics["maximum_abs_spot_action"].append(float(np.max(np.abs(spot_actions))))
    return {
        name: float(np.max(values) if name.startswith("maximum_") else np.mean(values))
        for name, values in metrics.items()
    }


def summarize_entry_only(root: Path, output_path: Path, pdf_path: Path | None = None) -> None:
    """Require a complete common cohort and write a reproducible JSON table."""
    candidates = {
        "Black delta-vega": root / "bs-greeks-mean-variance-lambda10",
        "SANOS Greek": root / "sanos-lv-mean-variance-lambda10",
        **{
            f"SANOS parametric MV {risk}": root / f"sanos-parametric-mean-variance-lambda{risk}"
            for risk in (10, 100)
        },
        **{
            f"SANOS normal-statistical MV {risk}": root
            / f"sanos-normal-statistical-mean-variance-lambda{risk}"
            for risk in (10, 100)
        },
        **{
            f"{model} {objective} {risk}": root / f"{slug}-{objective}-lambda{risk}"
            for model, slug in (
                ("SANOS statistical", "sanos-statistical"),
                ("VolGAN", "volgan"),
                ("FuNVol", "funvol"),
            )
            for objective in ("mean-variance", "entropy")
            for risk in (10, 100)
        },
    }
    records_by_name = {name: _records(directory) for name, directory in candidates.items()}
    reference = records_by_name["SANOS Greek"]
    reference_unhedged = {
        (row["origin_date"], row["claim"]): row["unhedged_pnl"] for row in reference
    }
    for name, records in records_by_name.items():
        unhedged = {
            (row["origin_date"], row["claim"]): row["unhedged_pnl"] for row in records
        }
        if unhedged.keys() != reference_unhedged.keys() or any(
            not np.isclose(unhedged[key], value, rtol=0.0, atol=1.0e-12)
            for key, value in reference_unhedged.items()
        ):
            raise RuntimeError(f"{name}: market claims or target quantities differ")
    table = {name: _summary(records) for name, records in records_by_name.items()}
    unhedged_records = [
        {
            **record,
            "hedged_pnl": record["unhedged_pnl"],
            "transaction_cost": 0.0,
            "actions": {"spot": 0.0},
            "diagnostics": {"formal_exit_cost": 0.0},
        }
        for record in reference
    ]
    table = {"Unhedged": _summary(unhedged_records), **table}
    output_path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if pdf_path is not None:
        _write_pdf(table, pdf_path)
    print("model | mean P&L | median P&L | SD | scaled MAD | mean |P&L| | entry cost | formal exit | mean |spot| | max |spot|")
    for name, row in table.items():
        values = " | ".join(
            f"{row[key]:.6f}"
            for key in (
                "mean_pnl",
                "median_pnl",
                "sd_pnl",
                "scaled_mad_pnl",
                "mean_absolute_pnl",
                "mean_entry_cost",
                "mean_formal_exit_cost",
                "mean_abs_spot_action",
                "maximum_abs_spot_action",
            )
        )
        print(f"{name} | {values}")


def _write_pdf(table: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (
        "Model / objective / λ",
        "Mean P&L",
        "Median P&L",
        "SD",
        "Scaled MAD",
        "Mean |P&L|",
        "Entry cost",
        "Formal exit",
        "Max |spot|",
    )
    keys = (
        "mean_pnl",
        "median_pnl",
        "sd_pnl",
        "scaled_mad_pnl",
        "mean_absolute_pnl",
        "mean_entry_cost",
        "mean_formal_exit_cost",
    )
    rows = [
        [name, *(f"{metrics[key] * 1.0e4:.1f}" for key in keys),
         f"{metrics['maximum_abs_spot_action']:.1f}"]
        for name, metrics in table.items()
    ]
    figure, axes = plt.subplots(figsize=(16.5, 10.5))
    axes.axis("off")
    figure.text(0.04, 0.965, "One-day digital hedges: April–May 2024", fontsize=18, weight="bold")
    figure.text(
        0.04,
        0.93,
        "44 matched dates × 10 digitals. P&L and costs: basis points of origin spot; "
        "statistics computed per target, then averaged.",
        fontsize=10,
    )
    figure.text(
        0.04,
        0.905,
        "Spot trades at mid. Options pay only entry half-spread. Formal exit is a separate "
        "hypothetical half-spread metric, excluded from optimization and P&L.",
        fontsize=9,
    )
    rendered = axes.table(
        cellText=rows,
        colLabels=labels,
        colWidths=[0.32, 0.082, 0.082, 0.065, 0.085, 0.08, 0.08, 0.08, 0.07],
        loc="center",
        cellLoc="right",
        bbox=[0.01, 0.02, 0.98, 0.85],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.2)
    for column in range(len(labels)):
        rendered[(0, column)].set_facecolor("#dae5f1")
        rendered[(0, column)].set_text_props(weight="bold")
    for row_index, (name, metrics) in enumerate(table.items(), start=1):
        rendered[(row_index, 0)].set_text_props(ha="left")
        if metrics["mean_absolute_pnl"] > table["Unhedged"]["mean_absolute_pnl"]:
            rendered[(row_index, 5)].set_facecolor("#f5c7c3")
        if metrics["maximum_abs_spot_action"] > 10.0:
            rendered[(row_index, 8)].set_facecolor("#ffe0b3")
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    arguments = parser.parse_args()
    summarize_entry_only(arguments.root, arguments.output, arguments.pdf)


if __name__ == "__main__":
    main()
