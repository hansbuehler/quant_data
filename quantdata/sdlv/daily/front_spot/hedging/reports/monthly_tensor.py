"""Compare a one-month 1D digital hedge tensor on identical market dates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def _family(case: dict[str, object]) -> str:
    model = str(case["model"])
    if model == "robust-covariance":
        return "Covariance"
    if model == "historical-sanos":
        return "Historical-SANOS"
    if model == "gromm":
        return f"GROMM-{case['measure']}"
    return model


def _configuration(case: dict[str, object]) -> str:
    objective = case.get("objective")
    if objective is None:
        return str(case["variant"])
    return (
        f"{str(objective).replace('-', ' ')} · λ={case['risk_parameter']:g}"
        f" · {case.get('mean_reduction', 'demean')}"
    )


def _pnl(books: dict[str, dict[str, dict[str, object]]], dates: list[str], targets: list[str], *, unhedged: bool = False) -> np.ndarray:
    key = "unhedged_pnl" if unhedged else "hedged_pnl"
    return np.asarray(
        [
            [
                1_000_000 * float(books[day][target][key])
                / float(books[day][target]["claim_quantity"])
                for target in targets
            ]
            for day in dates
        ],
        dtype=float,
    )


def _score(values: np.ndarray, statistic: str) -> float:
    absolute = np.abs(values)
    if statistic == "mean":
        return float(absolute.mean())
    if statistic == "median":
        return float(np.median(absolute))
    raise ValueError(f"unknown statistic: {statistic}")


def _profile(values: np.ndarray, statistic: str) -> np.ndarray:
    absolute = np.abs(values)
    if statistic == "mean":
        return absolute.mean(axis=0) / 1_000
    if statistic == "median":
        return np.median(absolute, axis=0) / 1_000
    raise ValueError(f"unknown statistic: {statistic}")


def _best_cases(cases: dict[str, np.ndarray], statistic: str) -> dict[str, str]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for key in cases:
        by_family[key.split("|", 1)[0]].append(key)
    return {
        family: min(keys, key=lambda key: (_score(cases[key], statistic), key))
        for family, keys in by_family.items()
    }


def _draw_headline(
    pdf: PdfPages,
    *,
    targets: list[str],
    dates: list[str],
    references: dict[str, np.ndarray],
    cases: dict[str, np.ndarray],
    selections: dict[str, dict[str, str]],
    title: str | None = None,
    interval_title: str | None = None,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(18, 14), layout="constrained")
    positions = np.arange(len(targets))
    for axis, statistic in zip(axes[:2], ("mean", "median"), strict=True):
        selected = {
            family: cases[key]
            for family, key in selections[statistic].items()
        }
        series = {**references, **selected}
        width = 0.8 / len(series)
        for index, (name, values) in enumerate(series.items()):
            axis.bar(
                positions + (index - (len(series) - 1) / 2) * width,
                _profile(values, statistic),
                width=0.92 * width,
                label=name,
            )
        axis.set_title(f"{statistic.title()} absolute daily P&L")
        axis.set_ylabel("USD thousands / $1m digital payout")
        axis.legend(ncol=min(len(series), 5), fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    for name, values in {**references, **{
        family: cases[key] for family, key in selections["median"].items()
    }}.items():
        median = np.median(np.abs(values), axis=0) / 1_000
        low, high = np.percentile(np.abs(values), (10, 90), axis=0) / 1_000
        axes[2].errorbar(
            positions, median, yerr=np.vstack((median - low, high - median)),
            fmt="o", capsize=2, label=name, alpha=0.75,
        )
    axes[2].set_title(
        interval_title or "10–90% interval of absolute daily P&L (median-selected cases)"
    )
    axes[2].set_ylabel("USD thousands / $1m digital payout")
    axes[2].legend(ncol=min(len(references) + len(selections["median"]), 5), fontsize=8)
    axes[2].grid(axis="y", alpha=0.2)
    labels = [_target_label(target) for target in targets]
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=35, ha="right")
    figure.suptitle(
        title or (
            f"1D digital hedges · {len(dates)} common April 2024 dates · "
            "best case per model selected ex post"
        )
    )
    pdf.savefig(figure)
    plt.close(figure)


def _draw_family(
    pdf: PdfPages,
    *,
    family: str,
    targets: list[str],
    dates: list[str],
    references: dict[str, np.ndarray],
    cases: dict[str, np.ndarray],
    descriptions: dict[str, str],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(18, 11), layout="constrained")
    positions = np.arange(len(targets))
    for axis, statistic in zip(axes, ("mean", "median"), strict=True):
        for name in ("Unhedged", "BS delta + vega"):
            axis.plot(
                positions,
                _profile(references[name], statistic),
                linewidth=2.2,
                label=name,
            )
        for key, values in sorted(cases.items()):
            axis.plot(positions, _profile(values, statistic), marker="o", markersize=3, label=descriptions[key])
        axis.set_ylabel("USD thousands / $1m digital payout")
        axis.set_title(f"{statistic.title()} absolute daily P&L")
        axis.grid(axis="y", alpha=0.2)
        axis.set_xticks(positions, [_target_label(target) for target in targets], rotation=35, ha="right")
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    figure.suptitle(f"{family} · all completed configurations · {len(dates)} common dates")
    pdf.savefig(figure)
    plt.close(figure)


def _target_label(label: str) -> str:
    return (
        label.replace("digital-call-1d-", "Call ")
        .replace("digital-put-1d-", "Put ")
        .replace("nms", "NMS ")
        .replace("atm", "ATM")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--sanos-lv", type=Path, required=True)
    parser.add_argument("--bs-greeks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-case",
        nargs=3,
        action="append",
        metavar=("FAMILY", "CONFIGURATION", "RUN_DIRECTORY"),
        default=[],
    )
    arguments = parser.parse_args()
    matrix = json.loads(arguments.matrix.read_text())
    if matrix["config"]["evaluation_start"] != "2024-04-01" or matrix["config"]["evaluation_end"] != "2024-04-30":
        raise ValueError("this report requires the April 2024 matrix")
    baseline_books = _load_run(arguments.sanos_lv)
    bs_books = _load_run(arguments.bs_greeks)
    dates = sorted(set(baseline_books) & set(bs_books))
    if len(dates) != 22:
        raise RuntimeError(f"reference books cover only {len(dates)}/22 April dates")
    targets = sorted(baseline_books[dates[0]], key=_target_order)
    references = {
        "Unhedged": _pnl(baseline_books, dates, targets, unhedged=True),
        "BS delta + vega": _pnl(bs_books, dates, targets),
        "SANOS-LV": _pnl(baseline_books, dates, targets),
    }
    if not np.allclose(
        references["Unhedged"], _pnl(bs_books, dates, targets, unhedged=True),
        rtol=1e-10, atol=1e-5,
    ):
        raise RuntimeError("reference market cashflows disagree")
    cases: dict[str, np.ndarray] = {}
    descriptions: dict[str, str] = {}
    coverage: dict[str, object] = {}
    for case in matrix["cases"]:
        if case["model"] in {"sanos-lv", "gromm-q-dlv-vega"}:
            continue
        command = case.get("command")
        if not command or "--output" not in command:
            coverage[case["case_id"]] = case.get("readiness", "unavailable")
            continue
        directory = Path(command[command.index("--output") + 1])
        if not (directory / "records").is_dir():
            coverage[case["case_id"]] = "not-run"
            continue
        try:
            books = _load_run(directory)
        except RuntimeError:
            coverage[case["case_id"]] = "no-certified-dates"
            continue
        if not set(dates).issubset(books):
            coverage[case["case_id"]] = f"partial:{len(set(dates) & set(books))}/22"
            continue
        if not np.allclose(
            references["Unhedged"], _pnl(books, dates, targets, unhedged=True),
            rtol=1e-10, atol=1e-5,
        ):
            raise RuntimeError(f"market cashflow mismatch: {case['case_id']}")
        key = f"{_family(case)}|{case['case_id']}"
        cases[key] = _pnl(books, dates, targets)
        descriptions[key] = _configuration(case)
        coverage[case["case_id"]] = "complete"
    for family, configuration, directory_name in arguments.extra_case:
        directory = Path(directory_name)
        books = _load_run(directory)
        if not set(dates).issubset(books):
            raise RuntimeError(f"extra case covers only {len(set(dates) & set(books))}/22 dates: {directory}")
        if not np.allclose(
            references["Unhedged"], _pnl(books, dates, targets, unhedged=True),
            rtol=1e-10, atol=1e-5,
        ):
            raise RuntimeError(f"market cashflow mismatch: {directory}")
        key = f"{family}|{directory.resolve()}"
        if key in cases:
            raise ValueError(f"duplicate extra case: {directory}")
        cases[key] = _pnl(books, dates, targets)
        descriptions[key] = configuration
        coverage[key] = "complete"
    selections = {
        statistic: _best_cases(cases, statistic)
        for statistic in ("mean", "median")
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    with PdfPages(arguments.output / "APRIL_2024_TENSOR.pdf") as pdf:
        _draw_headline(pdf, targets=targets, dates=dates, references=references, cases=cases, selections=selections)
        for family in sorted({key.split("|", 1)[0] for key in cases}):
            selected_cases = {key: values for key, values in cases.items() if key.startswith(f"{family}|")}
            _draw_family(
                pdf,
                family=family,
                targets=targets,
                dates=dates,
                references=references,
                cases=selected_cases,
                descriptions=descriptions,
            )
    summary = {
        "schema": "hedging-monthly-tensor-v1",
        "evaluation_dates": dates,
        "target_labels": targets,
        "coverage": coverage,
        "selected_cases_ex_post": selections,
        "selection_warning": "Best cases are selected on April outcomes and are descriptive, not out-of-sample policy selection.",
        "units": "USD per $1m digital cash payout",
    }
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed_cases": len(cases), "families": len(selections["mean"]), "pdf": str(arguments.output / "APRIL_2024_TENSOR.pdf")}, indent=2))


if __name__ == "__main__":
    main()
