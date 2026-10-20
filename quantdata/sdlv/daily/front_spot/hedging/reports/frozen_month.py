"""Compare a fixed set of daily hedges on a later, untouched month."""

from __future__ import annotations

import argparse
import json
from datetime import date
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
from quantdata.sdlv.daily.front_spot.hedging.reports.monthly_tensor import (
    _draw_headline,
    _pnl,
)


def _paired_hac_se(daily_difference: np.ndarray, *, maximum_lag: int = 5) -> float:
    centered = daily_difference - np.mean(daily_difference)
    count = len(centered)
    lag_count = min(maximum_lag, count - 1)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_count + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2 * (1 - lag / (lag_count + 1)) * covariance
    return float(np.sqrt(max(long_run_variance, 0.0) / count))


def _draw_summary_table(pdf: PdfPages, summary: dict[str, object]) -> None:
    models = summary["models"]
    assert isinstance(models, dict)
    rows = sorted(models.items(), key=lambda item: item[1]["mean_absolute_pnl_usd"])
    figure, axes = plt.subplots(figsize=(18, 9))
    axes.axis("off")
    cells = [
        [
            name,
            details["configuration"],
            f"{details['mean_absolute_pnl_usd'] / 1_000:,.1f}",
            f"{details['median_absolute_pnl_usd'] / 1_000:,.1f}",
            f"{details['mean_absolute_improvement_vs_bs_usd'] / 1_000:+,.1f}",
            f"{details['paired_hac_se_vs_bs_usd'] / 1_000:,.1f}",
        ]
        for name, details in rows
    ]
    table = axes.table(
        cellText=cells,
        colLabels=["Model", "Pre-May configuration", "Mean |P&L|", "Median |P&L|", "Gain vs BS", "HAC SE"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.17, 0.30, 0.13, 0.13, 0.14, 0.10],
        bbox=[0.02, 0.18, 0.96, 0.69],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for column in range(6):
        table[0, column].set_facecolor("#dce8ef")
    axes.set_title(
        f"May 2024 · {summary['dates']} common daily origins · 10 one-day digitals",
        fontsize=20,
        pad=22,
    )
    axes.text(
        0.02,
        0.08,
        "USD thousands per $1m digital payout. Positive gain means lower mean absolute P&L than BS. "
        "HAC SE uses date-level paired losses and up to five daily lags; 22 dates give limited inference.",
        transform=axes.transAxes,
        fontsize=12,
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def compare_fixed_month(
    *,
    sanos_lv: Path,
    bs_greeks: Path,
    cases: list[tuple[str, str, Path]],
    start: date,
    end: date,
    expected_dates: int,
    output: Path,
) -> dict[str, object]:
    """Require complete common dates and compare preselected configurations."""
    if start > end or expected_dates < 1:
        raise ValueError("invalid evaluation range")
    names = [name for name, _, _ in cases]
    if len(names) != len(set(names)) or set(names) & {
        "Unhedged", "BS delta + vega", "SANOS-LV"
    }:
        raise ValueError("case names must be unique and distinct from references")
    sources = {
        "SANOS-LV": sanos_lv,
        "BS delta + vega": bs_greeks,
        **{name: path for name, _, path in cases},
    }
    books = {name: _load_run(path) for name, path in sources.items()}
    dates = sorted(
        day for day in books["SANOS-LV"]
        if start.isoformat() <= day <= end.isoformat()
    )
    if len(dates) != expected_dates:
        raise RuntimeError(f"reference has {len(dates)}/{expected_dates} dates")
    targets = sorted(books["SANOS-LV"][dates[0]], key=_target_order)
    if len(targets) != 10:
        raise RuntimeError("reference does not contain ten one-day digital targets")
    reference = _pnl(books["SANOS-LV"], dates, targets, unhedged=True)
    for name, model_books in books.items():
        if not set(dates).issubset(model_books):
            raise RuntimeError(f"{name} does not cover every reference date")
        for day in dates:
            if set(model_books[day]) != set(targets):
                raise RuntimeError(f"{name} has a different target book on {day}")
        if not np.allclose(
            reference,
            _pnl(model_books, dates, targets, unhedged=True),
            rtol=1e-10,
            atol=1e-5,
        ):
            raise RuntimeError(f"{name} has different market cashflows")
    pnl = {name: _pnl(model_books, dates, targets) for name, model_books in books.items()}
    references = {
        "Unhedged": reference,
        "BS delta + vega": pnl["BS delta + vega"],
        "SANOS-LV": pnl["SANOS-LV"],
    }
    selected = {name: pnl[name] for name in names}
    selection = {name: name for name in names}
    bs_daily_loss = np.mean(np.abs(pnl["BS delta + vega"]), axis=1)
    summary = {
        "schema": "hedging-frozen-month-v1",
        "selection": "configurations fixed before the evaluation month",
        "start": dates[0],
        "end": dates[-1],
        "dates": len(dates),
        "targets": targets,
        "units": "USD per $1m digital cash payout",
        "models": {
            name: {
                "configuration": next(
                    (configuration for case_name, configuration, _ in cases if case_name == name),
                    "fixed reference",
                ),
                "source": str((sanos_lv if name == "Unhedged" else sources[name]).resolve()),
                "mean_absolute_pnl_usd": float(np.mean(np.abs(values))),
                "median_absolute_pnl_usd": float(np.median(np.abs(values))),
                "mean_absolute_improvement_vs_bs_usd": float(np.mean(
                    bs_daily_loss - np.mean(np.abs(values), axis=1)
                )),
                "paired_hac_se_vs_bs_usd": _paired_hac_se(
                    bs_daily_loss - np.mean(np.abs(values), axis=1)
                ),
                "targets_better_than_bs": int(np.sum(
                    np.mean(np.abs(values), axis=0)
                    < np.mean(np.abs(pnl["BS delta + vega"]), axis=0)
                )),
            }
            for name, values in {**references, **selected}.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    with PdfPages(output / "FROZEN_MONTH.pdf") as pdf:
        _draw_summary_table(pdf, summary)
        _draw_headline(
            pdf,
            targets=targets,
            dates=dates,
            references=references,
            cases=selected,
            selections={"mean": selection, "median": selection},
            title=(
                f"1D digital hedges · {len(dates)} common "
                f"{start:%B %Y} dates · configurations frozen before the month"
            ),
            interval_title="10–90% interval of absolute daily P&L (fixed cases)",
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanos-lv", type=Path, required=True)
    parser.add_argument("--bs-greeks", type=Path, required=True)
    parser.add_argument("--case", nargs=3, action="append", default=[])
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--expected-dates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare_fixed_month(
        sanos_lv=arguments.sanos_lv,
        bs_greeks=arguments.bs_greeks,
        cases=[(name, config, Path(path)) for name, config, path in arguments.case],
        start=arguments.start,
        end=arguments.end,
        expected_dates=arguments.expected_dates,
        output=arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
