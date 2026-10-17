"""Landscape one-day digital report for the drift-corrected statistical tensor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from quantdata.sdlv.daily.front_spot.hedging.reports.compare_rolling_1d_pnl import (
    _load_run,
    _target_order,
)

RISK_GRID = {
    "mean-variance": (0.1, 1.0, 10.0),
    "mean-volatility": (0.1, 1.0, 2.0, 3.0),
    "entropy": (0.1, 1.0, 10.0),
    "power-exponential": (0.1, 1.0, 10.0),
}
MODELS = ("Covariance", "Historical-SANOS")
COLORS = {
    "Unhedged": "#c8513d",
    "BS delta + vega": "#d19a25",
    "SANOS-LV": "#24866b",
    "Covariance": "#7357a6",
    "Historical-SANOS": "#1764ab",
}


def _case_key(model: str, objective: str, risk: float) -> str:
    return f"{model}|{objective}|{risk:g}"


def _metadata(directory: Path) -> dict[str, object]:
    return json.loads((directory / "run.json").read_text())


def _verify_case(
    directory: Path, *, model: str, objective: str, risk: float, reference: dict[str, object]
) -> None:
    metadata = _metadata(directory)
    fields = metadata["model_config_snapshot"]["fields"]
    expected_model = "robust-covariance" if model == "Covariance" else "historical-sanos"
    if (
        metadata["model"] != expected_model
        or fields.get("mean_reduction") != "demean"
        or fields.get("objective") != objective
        or not math.isclose(float(fields.get("risk_parameter", -1)), risk)
        or metadata.get("dataset_manifest_sha256") != reference.get("dataset_manifest_sha256")
    ):
        raise ValueError(f"incompatible statistical run: {directory}")


def _pnl(
    books: dict[str, dict[str, dict[str, object]]], dates: list[str], targets: list[str]
) -> np.ndarray:
    return np.array(
        [
            [
                1_000_000 * books[date][target]["hedged_pnl"]
                / books[date][target]["claim_quantity"]
                for target in targets
            ]
            for date in dates
        ],
        dtype=float,
    )


def _unhedged(
    books: dict[str, dict[str, dict[str, object]]], dates: list[str], targets: list[str]
) -> np.ndarray:
    return np.array(
        [
            [
                1_000_000 * books[date][target]["unhedged_pnl"]
                / books[date][target]["claim_quantity"]
                for target in targets
            ]
            for date in dates
        ],
        dtype=float,
    )


def _target_label(target: str) -> str:
    return (
        target.replace("digital-call-1d-", "Call\n")
        .replace("digital-put-1d-", "Put\n")
        .replace("nms", "NMS ")
        .replace("atm", "ATM")
    )


def _bars(axis: plt.Axes, values: dict[str, np.ndarray], targets: list[str], title: str) -> None:
    positions = np.arange(len(targets))
    width = 0.8 / len(values)
    for index, (name, statistic) in enumerate(values.items()):
        axis.bar(
            positions + (index - (len(values) - 1) / 2) * width,
            statistic / 1_000,
            width=0.94 * width,
            color=COLORS[name],
            label=name,
        )
    axis.set_xticks(positions, [_target_label(target) for target in targets])
    axis.set_ylabel("USD thousands per $1m payout")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)


def _case_page(
    pdf: PdfPages,
    *,
    model: str,
    objective: str,
    risk: float,
    case_books: dict[str, dict[str, dict[str, object]]] | None,
    references: dict[str, dict[str, dict[str, dict[str, object]]]],
    targets: list[str],
) -> dict[str, object]:
    figure, axes = plt.subplots(2, 1, figsize=(16, 9), layout="constrained")
    title = f"{model} · {objective.replace('-', ' ')} · risk aversion {risk:g}"
    if case_books is None:
        for axis in axes:
            axis.axis("off")
        axes[0].text(0.5, 0.5, "Corrected zero-mean run pending", ha="center", va="center", fontsize=18)
        figure.suptitle(title)
        pdf.savefig(figure)
        plt.close(figure)
        return {"status": "pending"}

    dates = sorted(set(case_books).intersection(*(set(books) for books in references.values())))
    if not dates:
        raise RuntimeError(f"no common dates for {title}")
    values = {"Unhedged": _unhedged(references["SANOS-LV"], dates, targets)}
    for name, books in references.items():
        values[name] = _pnl(books, dates, targets)
    values[model] = _pnl(case_books, dates, targets)
    for name, books in {**references, model: case_books}.items():
        candidate = _unhedged(books, dates, targets)
        if not np.allclose(candidate, values["Unhedged"], rtol=1e-10, atol=1e-5):
            raise RuntimeError(f"market cashflow mismatch: {title}, {name}")
    absolute = {name: np.abs(array) for name, array in values.items()}
    _bars(axes[0], {name: array.mean(axis=0) for name, array in absolute.items()}, targets, "Mean absolute daily P&L")
    _bars(axes[1], {name: np.median(array, axis=0) for name, array in absolute.items()}, targets, "Median absolute daily P&L")
    axes[0].legend(ncol=4, fontsize=10)
    figure.suptitle(f"{title} · {len(dates)} common completed dates, {dates[0]}–{dates[-1]}")
    pdf.savefig(figure)
    plt.close(figure)
    return {"status": "complete", "common_dates": len(dates), "first_date": dates[0], "last_date": dates[-1]}


def _best_half(
    case_pnl: dict[str, np.ndarray], *, statistic: str
) -> tuple[list[str], np.ndarray]:
    if statistic == "mean":
        score = lambda values: np.abs(values).mean(axis=0)
    elif statistic == "median":
        score = lambda values: np.median(np.abs(values), axis=0)
    else:
        raise ValueError(f"unknown statistic: {statistic}")
    ranked = sorted(case_pnl, key=lambda key: (float(score(case_pnl[key]).mean()), key))
    selected = ranked[: math.ceil(len(ranked) / 2)]
    return selected, np.mean([score(case_pnl[key]) for key in selected], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=4, action="append", metavar=("MODEL", "OBJECTIVE", "RISK", "RUN"), default=[])
    parser.add_argument("--sanos-lv", required=True, type=Path)
    parser.add_argument("--bs-greeks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    reference_metadata = _metadata(arguments.sanos_lv)
    references = {
        "BS delta + vega": _load_run(arguments.bs_greeks),
        "SANOS-LV": _load_run(arguments.sanos_lv),
    }
    targets = sorted(next(iter(references["SANOS-LV"].values())), key=_target_order)
    cases: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    source_paths: dict[str, str] = {}
    for model, objective, risk_text, path_text in arguments.case:
        risk = float(risk_text)
        if model not in MODELS or risk not in RISK_GRID.get(objective, ()):
            parser.error(f"unsupported statistical case: {model}, {objective}, {risk:g}")
        key = _case_key(model, objective, risk)
        if key in cases:
            parser.error(f"duplicate case: {key}")
        directory = Path(path_text)
        _verify_case(directory, model=model, objective=objective, risk=risk, reference=reference_metadata)
        cases[key] = _load_run(directory)
        source_paths[key] = str(directory)

    arguments.output.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, object] = {}
    with PdfPages(arguments.output / "STATISTICAL_DETAILS.pdf") as pdf:
        for model in MODELS:
            for objective, risks in RISK_GRID.items():
                for risk in risks:
                    key = _case_key(model, objective, risk)
                    coverage[key] = _case_page(
                        pdf, model=model, objective=objective, risk=risk,
                        case_books=cases.get(key), references=references, targets=targets,
                    )

    complete = len(cases) == len(MODELS) * sum(map(len, RISK_GRID.values()))
    summary: dict[str, object] = {
        "schema": "statistical-1d-matrix-v1",
        "source_runs": source_paths,
        "reference_runs": {"SANOS-LV": str(arguments.sanos_lv), "BS delta + vega": str(arguments.bs_greeks)},
        "coverage": coverage,
        "best_half_status": "pending complete corrected tensor",
        "selection_note": "Best half is selected ex post on the reported dates; it is descriptive, not a deployable out-of-sample strategy.",
    }
    if complete:
        dates = sorted(set.intersection(*(set(books) for books in (*cases.values(), *references.values()))))
        if len(dates) < 100:
            raise RuntimeError(f"only {len(dates)} dates common to the corrected tensor")
        benchmark = _unhedged(references["SANOS-LV"], dates, targets)
        for books in (*cases.values(), *references.values()):
            if not np.allclose(_unhedged(books, dates, targets), benchmark, rtol=1e-10, atol=1e-5):
                raise RuntimeError("market cashflows differ across the corrected tensor")
        figure, axes = plt.subplots(2, 1, figsize=(16, 9), layout="constrained")
        selections: dict[str, dict[str, list[str]]] = {model: {} for model in MODELS}
        for axis, statistic in zip(axes, ("mean", "median"), strict=True):
            reduction = np.mean if statistic == "mean" else np.median
            values = {"Unhedged": reduction(np.abs(benchmark), axis=0)}
            for name, books in references.items():
                values[name] = reduction(np.abs(_pnl(books, dates, targets)), axis=0)
            for model in MODELS:
                model_cases = {
                    key: _pnl(books, dates, targets)
                    for key, books in cases.items() if key.startswith(f"{model}|")
                }
                selected, values[model] = _best_half(model_cases, statistic=statistic)
                selections[model][statistic] = selected
            _bars(axis, values, targets, f"{statistic.title()} absolute daily P&L; best 7 of 13 cases per statistical model")
            axis.legend(ncol=5, fontsize=9)
        figure.suptitle(f"Corrected statistical tensor · {len(dates)} identical dates · ex-post best-half selection")
        with PdfPages(arguments.output / "STATISTICAL_SUMMARY.pdf") as pdf:
            pdf.savefig(figure)
        figure.savefig(arguments.output / "statistical_summary.png", dpi=180)
        plt.close(figure)
        summary.update(best_half_status="complete", common_dates=len(dates), selected_cases=selections)
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(arguments.output), "cases": len(cases), "best_half_status": summary["best_half_status"]}, indent=2))


if __name__ == "__main__":
    main()
