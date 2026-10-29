"""Summarize the rolling historical-SANOS hedge objective grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.reports.objective_grid_summary import (
    _load_records,
    _statistics,
)

OBJECTIVES = ("mean-variance", "entropy", "power-exponential", "cvar-oce")
RISK_PARAMETERS = ("1", "10", "100", "1000", "inf")


def _configuration_directory(root: Path, objective: str, risk: str) -> Path:
    if risk == "inf" and objective != "mean-variance":
        objective = "cvar-oce"
    risk_label = risk if risk == "inf" else f"{risk}.0"
    return root / "historical-sanos" / f"{objective}-lambda{risk_label}"


def summarize_sanos_objective_grid(root: Path) -> dict[str, object]:
    """Report only fully matched, certified April–May SANOS cells."""
    identity = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if identity["schema"] != "historical-sanos-objective-grid-v1":
        raise RuntimeError("unexpected historical SANOS run identity")
    source = Path(identity["source_run"])
    source_run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if source_run["dataset_manifest_sha256"] != identity["dataset_manifest_sha256"]:
        raise RuntimeError("historical SANOS and comparison datasets differ")
    reference = _load_records(source)
    dates = tuple(identity["cohort"])
    targets = tuple(identity["target_labels"])
    if len(dates) != 44 or tuple(sorted(reference)) != dates:
        raise RuntimeError("historical SANOS comparison cohort is incomplete")
    rows: list[dict[str, object]] = []
    for objective in OBJECTIVES:
        for risk in RISK_PARAMETERS:
            directory = _configuration_directory(root, objective, risk)
            books = _load_records(directory)
            failures = sorted((directory / "failures").glob("2024-*.json"))
            capped_books: dict[str, dict[str, dict[str, object]]] = {}
            for path in failures:
                if path.stem in books:
                    continue
                results = json.loads(path.read_text(encoding="utf-8"))["results"]
                by_target = {str(result["claim"]): result for result in results}
                if (
                    len(results) != len(targets)
                    or tuple(by_target) != targets
                    or any(
                        result["origin_date"] != path.stem
                        or (not result["converged"] and not result["cap_hits"])
                        for result in results
                    )
                ):
                    continue
                capped_books[path.stem] = by_target
            reported_books = {**books, **capped_books}
            cap_hit_count = sum(
                int(result["cap_hits"] > 0)
                for by_target in reported_books.values()
                for result in by_target.values()
            )
            cap_hit_dates = sum(
                any(result["cap_hits"] for result in by_target.values())
                for by_target in reported_books.values()
            )
            all_dates_reported = tuple(sorted(reported_books)) == dates
            row: dict[str, object] = {
                "objective": objective,
                "risk_parameter": risk,
                "origin_count": len(books),
                "expected_origin_count": len(dates),
                "status": (
                    "complete" if tuple(sorted(books)) == dates
                    else "cap-limited" if all_dates_reported and capped_books
                    else "incomplete"
                ),
                "failure_dates": [path.stem for path in failures if path.stem not in books],
                "cap_hit_count": cap_hit_count,
                "cap_hit_percentage": 100.0 * cap_hit_count / (len(dates) * len(targets)),
                "cap_hit_dates": cap_hit_dates,
                "shared_worst_case_limit": risk == "inf" and objective != "mean-variance",
            }
            if row["status"] in ("complete", "cap-limited"):
                for origin in dates:
                    if tuple(reported_books[origin]) != targets:
                        raise RuntimeError(f"{directory}: target catalog differs")
                    for target in targets:
                        actual = reported_books[origin][target]
                        expected = reference[origin][target]
                        if not np.isclose(
                            float(actual["unhedged_pnl"]),
                            float(expected["unhedged_pnl"]),
                            rtol=0.0,
                            atol=1.0e-10,
                        ) or not np.isclose(
                            float(actual["claim_quantity"]),
                            float(expected["claim_quantity"]),
                            rtol=0.0,
                            atol=1.0e-12,
                        ):
                            raise RuntimeError(f"{directory}: market claim differs on {origin} {target}")
                row.update(_statistics(reported_books, dates=dates, targets=targets))
                if row["mean_entry_cost_kusd"] > 1_000.0:
                    row["economic_warning"] = "mean entry cost exceeds claim payout"
            rows.append(row)
    return {
        "schema": "historical-sanos-objective-grid-summary-v1",
        "cohort": "2024-04-01 through 2024-05-31; 44 dates; ten 1D digitals per date",
        "payout_scale": "USD thousands per USD 1 million digital payout",
        "convention": "252 causal historical SANOS-repriced returns; demeaned; spot at mid; options entry half-spread only",
        "risk_limit": "mean-variance infinity minimizes variance; other infinity cases share the finite-scenario maximin hedge",
        "unhedged": _statistics(reference, dates=dates, targets=targets, unhedged=True),
        "rows": rows,
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Rolling historical-SANOS hedge objective grid",
        "",
        str(summary["cohort"]),
        "",
        str(summary["convention"]),
        "",
        str(summary["risk_limit"]),
        "",
        "All financial values are USD thousands per USD 1 million digital payout; "
        "target-level statistics are averaged across ten digitals. "
        "Cap hits count target-date hedges with at least one instrument at its position limit; "
        "cap-limited outcomes are descriptive, not certified hedge results.",
        "",
        "| Objective | λ | Status | Cap hits | Mean P&L | Median P&L | SD | Mean absolute P&L | Entry cost | Mean gross | Max action |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        status = str(row["status"])
        if row["failure_dates"]:
            status += (
                f"; {len(row['failure_dates'])} failures"
                f" (first {row['failure_dates'][0]})"
            )
        if "economic_warning" in row:
            status += f"; {row['economic_warning']}"
        values = (
            " | ".join(
                f"{float(row[key]):.2f}"
                for key in (
                    "mean_pnl_kusd",
                    "median_pnl_kusd",
                    "sd_pnl_kusd",
                    "mean_absolute_pnl_kusd",
                    "mean_entry_cost_kusd",
                    "mean_gross_action",
                    "maximum_absolute_action",
                )
            )
            if row["status"] in ("complete", "cap-limited")
            else "— | — | — | — | — | — | —"
        )
        cap_hits = f"{row['cap_hit_count']}/440 ({row['cap_hit_percentage']:.1f}%)"
        lines.append(
            f"| {row['objective']} | {row['risk_parameter']} | {status} | "
            f"{cap_hits} | {values} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = summarize_sanos_objective_grid(arguments.root)
    (arguments.root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arguments.root / "SUMMARY.md").write_text(_markdown(summary), encoding="utf-8")
    print(_markdown(summary), flush=True)


if __name__ == "__main__":
    main()
