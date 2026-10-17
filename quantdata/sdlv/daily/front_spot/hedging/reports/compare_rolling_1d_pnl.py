"""Compare available one-day digital hedges on identical completed dates."""

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


def _load_run(directory: Path) -> dict[str, dict[str, dict[str, object]]]:
    books: dict[str, dict[str, dict[str, object]]] = {}
    for file in sorted((directory / "records").glob("*.json")):
        book = {
            result["claim"]: result
            for result in json.loads(file.read_text())["results"]
            if "-1d-" in result["claim"]
        }
        if len(book) == 10 and all(result["converged"] for result in book.values()):
            books[file.stem] = book
    if not books:
        raise RuntimeError(f"no certified 1D dates in {directory}")
    return books


def _target_order(label: str) -> tuple[float, int]:
    nms = 0.0 if label.endswith("-atm") else float(label.rsplit("nms", 1)[1])
    return nms, 0 if "-put-" in label else 1


def _strategy_label(name: str, record: dict[str, object]) -> str:
    if name not in {"Covariance", "Historical-SANOS"}:
        return name
    diagnostics = record["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{name} record lacks objective diagnostics")
    objective = diagnostics.get("objective")
    risk_parameter = diagnostics.get("risk_parameter")
    if not isinstance(objective, str) or not isinstance(risk_parameter, (float, int)):
        raise ValueError(f"{name} objective or risk parameter is missing")
    if name == "Covariance" and objective == "entropy":
        objective_label = "Gaussian entropy (=MV)"
    elif name == "Historical-SANOS" and objective == "entropy":
        objective_label = "empirical entropy"
    else:
        objective_label = objective.replace("-", " ")
    return f"{name}: {objective_label} λ={risk_parameter:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariance", type=Path, required=True)
    parser.add_argument("--historical-sanos", type=Path, required=True)
    parser.add_argument("--sanos-lv", type=Path, required=True)
    parser.add_argument("--bs-greeks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile-low", type=float, default=20.0)
    parser.add_argument("--percentile-high", type=float, default=80.0)
    arguments = parser.parse_args()
    if not (0.0 <= arguments.percentile_low < 50.0 < arguments.percentile_high <= 100.0):
        parser.error("percentiles must satisfy 0 <= low < 50 < high <= 100")
    sources = {
        "Covariance": arguments.covariance,
        "Historical-SANOS": arguments.historical_sanos,
        "SANOS-LV": arguments.sanos_lv,
    }
    if arguments.bs_greeks is not None:
        sources["BS delta + vega"] = arguments.bs_greeks
    books = {name: _load_run(path) for name, path in sources.items()}
    dates = sorted(set.intersection(*(set(model_books) for model_books in books.values())))
    if not dates:
        raise RuntimeError("no common completed 1D dates")
    targets = sorted(books["Covariance"][dates[0]], key=_target_order)
    if len(targets) != 10:
        raise RuntimeError("no common complete 1D target panel")

    rows: list[dict[str, object]] = []
    for date in dates:
        for target in targets:
            reference = books["Covariance"][date][target]
            quantity = reference["claim_quantity"]
            if quantity <= 0:
                raise RuntimeError(f"invalid claim quantity for {date} {target}")
            unhedged = 1_000_000 * reference["unhedged_pnl"] / quantity
            row: dict[str, object] = {"date": date, "target": target, "unhedged_usd": unhedged}
            for name, model_books in books.items():
                result = model_books[date][target]
                model_unhedged = 1_000_000 * result["unhedged_pnl"] / result["claim_quantity"]
                if not np.isclose(model_unhedged, unhedged, rtol=1e-10, atol=1e-5):
                    raise RuntimeError(f"market cashflow mismatch for {date} {target}: {name}")
                row[f"{name}_hedged_usd"] = 1_000_000 * result["hedged_pnl"] / result["claim_quantity"]
                row[f"{name}_cost_usd"] = 1_000_000 * result["transaction_cost"] / result["claim_quantity"]
            rows.append(row)

    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "common_1d_pnl_usd.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(row["date"], row["target"]): row for row in rows}
    colors = {
        "Unhedged": "#c8513d",
        "Covariance": "#7357a6",
        "Historical-SANOS": "#1764ab",
        "SANOS-LV": "#24866b",
    }
    if arguments.bs_greeks is not None:
        colors["BS delta + vega"] = "#d19a25"
    display_labels = {
        name: (
            "Unhedged"
            if name == "Unhedged"
            else _strategy_label(name, books[name][dates[0]][targets[0]])
        )
        for name in colors
    }

    def series(target: str, name: str) -> np.ndarray:
        key = "unhedged_usd" if name == "Unhedged" else f"{name}_hedged_usd"
        return np.array([lookup[date, target][key] for date in dates], dtype=float)

    summary = {
        "schema": "common-1d-model-pnl-usd-v3",
        "source_runs": {name: str(path) for name, path in sources.items()},
        "strategy_labels": display_labels,
        "common_completed_dates": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "normalization": "$1,000,000 cash payout per digital; each fitted hedge is scaled ex post by its original claim quantity",
        "per_target_mean_absolute_daily_usd": {
            target: {name: float(np.mean(np.abs(series(target, name)))) for name in colors}
            for target in targets
        },
        "per_target_median_absolute_daily_usd": {
            target: {name: float(np.median(np.abs(series(target, name)))) for name in colors}
            for target in targets
        },
        "absolute_daily_pnl_percentiles": [
            arguments.percentile_low,
            arguments.percentile_high,
        ],
        "per_target_percentile_low_absolute_daily_usd": {
            target: {
                name: float(np.percentile(np.abs(series(target, name)), arguments.percentile_low))
                for name in colors
            }
            for target in targets
        },
        "per_target_percentile_high_absolute_daily_usd": {
            target: {
                name: float(np.percentile(np.abs(series(target, name)), arguments.percentile_high))
                for name in colors
            }
            for target in targets
        },
    }
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2))

    x_dates = np.array(dates, dtype="datetime64[D]")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    with PdfPages(arguments.output / "common_1d_pnl_usd.pdf") as pdf:
        figure, axes = plt.subplots(3, 1, figsize=(14, 15), layout="constrained")
        positions = np.arange(len(targets), dtype=float)
        names = tuple(colors)
        bar_width = 0.78 / len(names)
        for axis, statistic, field in zip(
            axes[:2],
            ("Mean", "Median"),
            ("per_target_mean_absolute_daily_usd", "per_target_median_absolute_daily_usd"),
            strict=True,
        ):
            for index, name in enumerate(names):
                values = [summary[field][target][name] / 1_000 for target in targets]
                axis.bar(
                    positions + (index - (len(names) - 1) / 2) * bar_width,
                    values,
                    width=0.94 * bar_width,
                    color=colors[name],
                    label=display_labels[name],
                )
            axis.set_xticks(positions, [
                target.replace("digital-call-1d-", "Digital call\n")
                .replace("digital-put-1d-", "Digital put\n")
                .replace("nms", "NMS ")
                .replace("atm", "ATM")
                for target in targets
            ])
            axis.set_ylabel(f"{statistic} absolute daily P&L\nUSD thousands per $1m digital payout")
            axis.set_title(f"1D digital targets — {statistic.lower()}, same {len(dates)} completed dates")
            axis.legend(ncol=len(names), fontsize=8)
            axis.grid(axis="y", alpha=0.2)
        interval_axis = axes[2]
        for index, name in enumerate(names):
            lower = np.array([
                summary["per_target_percentile_low_absolute_daily_usd"][target][name] / 1_000
                for target in targets
            ])
            median = np.array([
                summary["per_target_median_absolute_daily_usd"][target][name] / 1_000
                for target in targets
            ])
            upper = np.array([
                summary["per_target_percentile_high_absolute_daily_usd"][target][name] / 1_000
                for target in targets
            ])
            interval_axis.errorbar(
                positions + (index - (len(names) - 1) / 2) * bar_width,
                median,
                yerr=np.vstack((median - lower, upper - median)),
                fmt="o",
                markersize=4,
                capsize=3,
                color=colors[name],
                label=display_labels[name],
            )
        interval_axis.set_xticks(positions, [
            target.replace("digital-call-1d-", "Digital call\n")
            .replace("digital-put-1d-", "Digital put\n")
            .replace("nms", "NMS ")
            .replace("atm", "ATM")
            for target in targets
        ])
        interval_axis.set_ylabel("Absolute daily P&L\nUSD thousands per $1m digital payout")
        interval_axis.set_title(
            f"Empirical {arguments.percentile_low:g}th–{arguments.percentile_high:g}th "
            "percentiles; dot = median, whiskers = percentiles"
        )
        interval_axis.legend(ncol=len(names), fontsize=8)
        interval_axis.grid(axis="y", alpha=0.2)
        pdf.savefig(figure)
        figure.savefig(arguments.output / "common_1d_mae_usd.png", dpi=180)
        figure.savefig(arguments.output / "common_1d_pnl_usd.png", dpi=180)
        plt.close(figure)

    with PdfPages(arguments.output / "DETAILS.pdf") as details:
        for target in targets:
            unhedged = series(target, "Unhedged")
            for name in sources:
                hedged = series(target, name)
                figure, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
                for axis, cumulative in zip(axes, (False, True), strict=True):
                    axis.plot(
                        x_dates,
                        hedged.cumsum() if cumulative else hedged,
                        color=colors[name],
                        linewidth=1.2,
                        label=f"{display_labels[name]}, net costs",
                    )
                    axis.plot(
                        x_dates,
                        unhedged.cumsum() if cumulative else unhedged,
                        color=colors["Unhedged"],
                        linewidth=1.0,
                        label="Unhedged",
                    )
                    axis.axhline(0, color="black", linewidth=0.6)
                    axis.set_title("Cumulative P&L" if cumulative else "Daily P&L")
                    axis.set_ylabel("USD per $1m digital payout")
                    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
                    axis.grid(alpha=0.2)
                    axis.legend(loc="upper left", fontsize=8)
                figure.suptitle(f"{target} · {name} · {len(dates)} common dates")
                details.savefig(figure)
                plt.close(figure)
    print(json.dumps({"output": str(arguments.output), "dates": len(dates), "first": dates[0], "last": dates[-1]}, indent=2))


if __name__ == "__main__":
    main()
