"""Summarize the matched generated-scenario one-day hedge objective grid."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OBJECTIVES = ("mean-variance", "entropy", "power-exponential", "cvar")
RISK_PARAMETERS = ("1", "10", "100", "1000", "inf")
GENERATORS = (
    "funvol",
    "volgan",
    "gromm-P",
    "gromm-Q",
    "diffusion-base",
    "diffusion-ft",
)


def _load_records(directory: Path) -> dict[str, dict[str, dict[str, object]]]:
    books: dict[str, dict[str, dict[str, object]]] = {}
    for path in sorted((directory / "records").glob("2024-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload["results"]
        if len(records) != 10:
            raise RuntimeError(f"{path}: expected ten 1D digital targets")
        by_target = {str(record["claim"]): record for record in records}
        if len(by_target) != 10 or any(
            record["origin_date"] != path.stem
            or not record["converged"]
            or record["cap_hits"]
            for record in records
        ):
            raise RuntimeError(f"{path}: target book is incomplete or uncertified")
        books[path.stem] = by_target
    return books


def _load_cap_limited_records(directory: Path) -> dict[str, dict[str, dict[str, object]]]:
    books: dict[str, dict[str, dict[str, object]]] = {}
    for path in sorted((directory / "failures").glob("2024-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload if isinstance(payload, list) else payload.get("results")
        if not isinstance(results, list):
            continue
        by_target = {str(result["claim"]): result for result in results}
        if (
            len(results) != 10
            or len(by_target) != 10
            or any(
                result["origin_date"] != path.stem
                or (not result["converged"] and not result["cap_hits"])
                for result in results
            )
            or not any(result["cap_hits"] for result in results)
        ):
            continue
        books[path.stem] = by_target
    return books


def _configuration_directory(root: Path, generator: str, objective: str, risk: str) -> Path:
    if risk == "inf" and objective != "mean-variance":
        objective = "cvar"
    return root / generator / f"{objective}-lambda{risk}"


def _statistics(
    books: dict[str, dict[str, dict[str, object]]],
    *,
    dates: tuple[str, ...],
    targets: tuple[str, ...],
    unhedged: bool = False,
) -> dict[str, float]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for target in targets:
        rows = [books[origin][target] for origin in dates]
        quantities = np.asarray([row["claim_quantity"] for row in rows], dtype=float)
        if np.any(~np.isfinite(quantities) | (quantities <= 0.0)):
            raise RuntimeError("claim payout quantity is invalid")
        field = "unhedged_pnl" if unhedged else "hedged_pnl"
        pnl = 1.0e3 * np.asarray([row[field] for row in rows], dtype=float) / quantities
        entry_cost = (
            np.zeros_like(pnl)
            if unhedged
            else 1.0e3
            * np.asarray([row["transaction_cost"] for row in rows], dtype=float)
            / quantities
        )
        gross_actions = np.asarray(
            [sum(abs(value) for value in row.get("actions", {}).values()) for row in rows],
            dtype=float,
        )
        maximum_actions = np.asarray(
            [max((abs(value) for value in row.get("actions", {}).values()), default=0.0) for row in rows],
            dtype=float,
        )
        metrics["mean_pnl_kusd"].append(float(np.mean(pnl)))
        metrics["median_pnl_kusd"].append(float(np.median(pnl)))
        metrics["sd_pnl_kusd"].append(float(np.std(pnl, ddof=1)))
        metrics["mean_absolute_pnl_kusd"].append(float(np.mean(np.abs(pnl))))
        metrics["mean_entry_cost_kusd"].append(float(np.mean(entry_cost)))
        metrics["mean_gross_action"].append(float(np.mean(gross_actions)))
        metrics["maximum_absolute_action"].append(float(np.max(maximum_actions)))
    return {
        key: float(np.max(values) if key.startswith("maximum_") else np.mean(values))
        for key, values in metrics.items()
    }


def summarize_objective_grid(root: Path) -> dict[str, object]:
    """Require matched certified records before reporting any numeric comparison."""
    reference = _load_records(root / "volgan" / "source-export-run")
    dates = tuple(sorted(reference))
    if len(dates) != 44:
        raise RuntimeError("VolGAN source export has not completed all 44 dates")
    targets = tuple(reference[dates[0]])
    if any(tuple(reference[origin]) != targets for origin in dates):
        raise RuntimeError("source target catalog changed within the cohort")
    rows: list[dict[str, object]] = []
    for generator in GENERATORS:
        for objective in OBJECTIVES:
            for risk in RISK_PARAMETERS:
                directory = _configuration_directory(root, generator, objective, risk)
                books = _load_records(directory)
                capped_books = {
                    origin: by_target
                    for origin, by_target in _load_cap_limited_records(directory).items()
                    if origin not in books
                }
                reported_books = {**books, **capped_books}
                failures = sorted((directory / "failures").glob("2024-*.json"))
                unresolved_failures = [path.stem for path in failures if path.stem not in books]
                cap_hit_count = sum(
                    int(result["cap_hits"] > 0)
                    for by_target in reported_books.values()
                    for result in by_target.values()
                )
                row: dict[str, object] = {
                    "generator": generator,
                    "objective": objective,
                    "risk_parameter": risk,
                    "origin_count": len(reported_books),
                    "expected_origin_count": len(dates),
                    "status": (
                        "complete" if tuple(sorted(books)) == dates
                        else "cap-limited" if capped_books and tuple(sorted(reported_books)) == dates
                        else "incomplete"
                    ),
                    "failure_dates": unresolved_failures,
                    "cap_hit_count": cap_hit_count,
                    "cap_hit_percentage": 100.0 * cap_hit_count / (len(dates) * len(targets)),
                    "resolved_failure_dates": [
                        path.stem for path in failures if path.stem in books
                    ],
                    "shared_worst_case_limit": risk == "inf" and objective != "mean-variance",
                }
                if row["status"] in ("complete", "cap-limited"):
                    if any(tuple(reported_books[origin]) != targets for origin in dates):
                        raise RuntimeError(f"{directory}: target catalog differs")
                    for origin in dates:
                        for target in targets:
                            expected = reference[origin][target]
                            actual = reported_books[origin][target]
                            if not np.isclose(
                                float(actual["unhedged_pnl"])
                                / float(actual["claim_quantity"]),
                                float(expected["unhedged_pnl"])
                                / float(expected["claim_quantity"]),
                                rtol=0.0,
                                atol=1.0e-10,
                            ):
                                raise RuntimeError(
                                    f"{directory}: market claim differs on {origin} {target}"
                                )
                    row.update(_statistics(reported_books, dates=dates, targets=targets))
                    if row["mean_entry_cost_kusd"] > 1_000.0:
                        row["economic_warning"] = "mean entry cost exceeds claim payout"
                rows.append(row)
    return {
        "schema": "generated-objective-grid-summary-v1",
        "cohort": "2024-04-01 through 2024-05-31; 44 dates; ten 1D digitals per date",
        "payout_scale": "USD thousands per USD 1 million digital payout",
        "convention": "spot at mid; options entry half-spread only; zero sample means",
        "risk_limit": "mean-variance infinity minimizes variance, with cost/ridge used only to break exact ties; all other infinity cases share the finite-scenario maximin hedge",
        "unhedged": _statistics(reference, dates=dates, targets=targets, unhedged=True),
        "rows": rows,
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Generated-scenario hedge objective grid",
        "",
        str(summary["cohort"]),
        "",
        str(summary["convention"]),
        "",
        str(summary["risk_limit"]),
        "",
        "All financial numbers are USD thousands per USD 1 million digital payout; "
        "target-level statistics are averaged across ten digitals. "
        "Cap hits count target-date hedges with at least one instrument at its position limit; "
        "cap-limited outcomes are descriptive, not certified hedge results.",
        "",
        "| Generator | Objective | λ | Dates | Status | Cap hits | Mean P&L | Median P&L | SD | Mean absolute P&L | Entry cost | Mean gross | Max action |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        statistics = (
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
        status = str(row["status"])
        if row["failure_dates"]:
            status += (
                f"; {len(row['failure_dates'])} flagged dates"
                f" (first {row['failure_dates'][0]})"
            )
        if "economic_warning" in row:
            status += f"; {row['economic_warning']}"
        lines.append(
            f"| {row['generator']} | {row['objective']} | {row['risk_parameter']} | "
            f"{row['origin_count']}/44 | {status} | "
            f"{row['cap_hit_count']}/440 ({row['cap_hit_percentage']:.1f}%) | {statistics} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = summarize_objective_grid(arguments.root)
    (arguments.root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arguments.root / "SUMMARY.md").write_text(_markdown(summary), encoding="utf-8")
    print(_markdown(summary), flush=True)


if __name__ == "__main__":
    main()
