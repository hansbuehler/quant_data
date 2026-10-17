"""Plot realized one-day digital hedging P&L per $1 million payout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = sorted((arguments.run / "records").glob("*.json"))
    if not records:
        raise RuntimeError("no completed daily records")

    rows: list[dict[str, object]] = []
    for record in records:
        payload = json.loads(record.read_text())
        results = payload["results"]
        if len(results) != 10 or any(not result["converged"] for result in results):
            raise RuntimeError(f"incomplete or uncertified 1D book: {record.name}")
        for result in results:
            if result["claim_quantity"] <= 0 or "-1d-" not in result["claim"]:
                raise RuntimeError(f"invalid 1D target or quantity: {record.name}")
            scale = 1_000_000 / result["claim_quantity"]
            rows.append(
                {
                    "date": result["origin_date"],
                    "target": result["claim"],
                    "hedged_usd": result["hedged_pnl"] * scale,
                    "unhedged_usd": result["unhedged_pnl"] * scale,
                    "cost_usd": result["transaction_cost"] * scale,
                    "claim_quantity": result["claim_quantity"],
                }
            )

    arguments.output.mkdir(parents=True, exist_ok=True)
    csv_path = arguments.output / "rolling_1d_pnl_usd.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dates = sorted({row["date"] for row in rows})
    targets = sorted({row["target"] for row in rows})
    if len(rows) != len(dates) * len(targets):
        raise RuntimeError("the rolling P&L panel is incomplete")
    lookup = {(row["date"], row["target"]): row for row in rows}
    daily_hedged = np.array(
        [np.mean([lookup[date, target]["hedged_usd"] for target in targets]) for date in dates],
        dtype=float,
    )
    daily_unhedged = np.array(
        [np.mean([lookup[date, target]["unhedged_usd"] for target in targets]) for date in dates],
        dtype=float,
    )
    daily_cost = np.array(
        [np.mean([lookup[date, target]["cost_usd"] for target in targets]) for date in dates],
        dtype=float,
    )
    x_dates = np.array(dates, dtype="datetime64[D]")
    summary = {
        "schema": "rolling-1d-pnl-usd-v1",
        "source_run": str(arguments.run),
        "first_date": dates[0],
        "last_date": dates[-1],
        "completed_dates": len(dates),
        "targets": len(targets),
        "scale": "Each target is rescaled ex post to $1,000,000 digital cash payout; the equal-weight book allocates $100,000 payout to each of ten targets per date.",
        "hedged_mean_daily_usd": float(daily_hedged.mean()),
        "unhedged_mean_daily_usd": float(daily_unhedged.mean()),
        "hedged_mean_absolute_daily_usd": float(np.abs(daily_hedged).mean()),
        "unhedged_mean_absolute_daily_usd": float(np.abs(daily_unhedged).mean()),
        "mean_daily_cost_usd": float(daily_cost.mean()),
        "hedged_cumulative_usd": float(daily_hedged.sum()),
        "unhedged_cumulative_usd": float(daily_unhedged.sum()),
    }
    summary["per_target"] = {
        target: {
            "hedged_mean_absolute_daily_usd": float(np.mean([
                abs(lookup[date, target]["hedged_usd"]) for date in dates
            ])),
            "unhedged_mean_absolute_daily_usd": float(np.mean([
                abs(lookup[date, target]["unhedged_usd"]) for date in dates
            ])),
            "hedged_cumulative_usd": float(sum(
                lookup[date, target]["hedged_usd"] for date in dates
            )),
            "unhedged_cumulative_usd": float(sum(
                lookup[date, target]["unhedged_usd"] for date in dates
            )),
        }
        for target in targets
    }
    summary["aggregation_warning"] = (
        "The equal-weight long put/call book has internal payoff cancellation; "
        "judge hedge quality from individual targets, not only its pooled P&L."
    )
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    with PdfPages(arguments.output / "rolling_1d_pnl_usd.pdf") as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        for row_index, target in enumerate(("digital-put-1d-atm", "digital-put-1d-nms-0.25")):
            hedged = np.array([lookup[date, target]["hedged_usd"] for date in dates], dtype=float)
            unhedged = np.array([lookup[date, target]["unhedged_usd"] for date in dates], dtype=float)
            for column_index, (hedged_values, unhedged_values, title) in enumerate((
                (hedged, unhedged, "Daily P&L"),
                (hedged.cumsum(), unhedged.cumsum(), "Cumulative P&L"),
            )):
                axis = axes[row_index, column_index]
                axis.plot(x_dates, hedged_values, color="#1764ab", linewidth=1.1, label="Hedged, net costs")
                axis.plot(x_dates, unhedged_values, color="#c8513d", linewidth=1.0, alpha=0.8, label="Unhedged")
                axis.axhline(0, color="black", linewidth=0.6)
                axis.set_title(f"{target}: {title}")
                axis.set_ylabel("USD per $1m digital payout")
                axis.xaxis.set_major_locator(mdates.AutoDateLocator())
                axis.grid(alpha=0.2)
                axis.legend(loc="upper left", fontsize=8)
        figure.suptitle(f"{arguments.run.name}: {len(dates)} completed dates, {dates[0]} to {dates[-1]}")
        pdf.savefig(figure)
        figure.savefig(arguments.output / "rolling_1d_pnl_usd.png", dpi=180)
        plt.close(figure)

        figure, axes = plt.subplots(5, 2, figsize=(12, 15), layout="constrained", sharex=True)
        for axis, target in zip(axes.flat, targets, strict=True):
            hedged = np.array([lookup[date, target]["hedged_usd"] for date in dates], dtype=float)
            unhedged = np.array([lookup[date, target]["unhedged_usd"] for date in dates], dtype=float)
            axis.plot(x_dates, hedged.cumsum(), color="#1764ab", linewidth=1.1, label="Hedged, net costs")
            axis.plot(x_dates, unhedged.cumsum(), color="#c8513d", linewidth=1.0, label="Unhedged")
            axis.axhline(0, color="black", linewidth=0.5)
            axis.set_title(target.replace("digital-", "").replace("-1d-", " · "))
            axis.grid(alpha=0.2)
        axes.flat[0].legend(loc="upper left", fontsize=8)
        figure.suptitle("Cumulative P&L by target — $1m cash payout for each digital")
        pdf.savefig(figure)
        plt.close(figure)

        for target in targets:
            hedged = np.array([lookup[date, target]["hedged_usd"] for date in dates], dtype=float)
            unhedged = np.array([lookup[date, target]["unhedged_usd"] for date in dates], dtype=float)
            figure, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
            for axis, hedged_values, unhedged_values, title in (
                (axes[0], hedged, unhedged, "Daily P&L"),
                (axes[1], hedged.cumsum(), unhedged.cumsum(), "Cumulative P&L"),
            ):
                axis.plot(x_dates, hedged_values, color="#1764ab", linewidth=1.2, label="Hedged, net costs")
                axis.plot(x_dates, unhedged_values, color="#c8513d", linewidth=1.0, label="Unhedged")
                axis.axhline(0, color="black", linewidth=0.6)
                axis.set_title(title)
                axis.set_ylabel("USD per $1m digital payout")
                axis.grid(alpha=0.2)
                axis.legend(loc="upper left")
            figure.suptitle(f"{target}: {len(dates)} completed dates, {dates[0]} to {dates[-1]}")
            pdf.savefig(figure)
            figure.savefig(arguments.output / f"{target}_pnl_usd.png", dpi=150)
            plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
