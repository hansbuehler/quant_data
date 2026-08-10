"""Normalize and summarize one-expiry static arbitrage reports."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .required_runs import clopper_pearson_one_sided

_REPORT_ROOT = Path("quantdata/dysanos/grid/gendysanos/reports/expiry-arbitrage")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _repair_cellwise_inference(report: dict[str, Any]) -> None:
    """Use paths—not pooled state/expiry pairs—as independent observations."""

    counts = np.asarray(
        report["direct_P_positive_margin_count_by_time_expiry"], dtype=np.int64
    )
    if counts.shape != (10, 7):
        raise ValueError(f"unexpected direct-P count shape {counts.shape}")
    direct_paths = int(report["sample_direct_P_paths"])
    alpha = 0.05 / counts.size
    lower = np.empty(counts.shape, dtype=np.float64)
    upper = np.empty(counts.shape, dtype=np.float64)
    for index in np.ndindex(counts.shape):
        lower[index], upper[index] = clopper_pearson_one_sided(
            int(counts[index]), direct_paths, alpha=alpha
        )
    report.pop("direct_P_positive_probability_lower_95", None)
    report.pop("direct_P_positive_probability_upper_95", None)
    report["direct_P_paths_are_independent_observations"] = True
    report["cellwise_familywise_alpha"] = 0.05
    report["cellwise_alpha"] = alpha
    report["direct_P_confirmed_time_expiry_cells"] = int(np.sum(lower > 0.0))
    report["maximum_direct_P_cell_probability_lower"] = float(np.max(lower))
    report["direct_P_probability_lower_by_time_expiry"] = lower.tolist()
    report["direct_P_probability_upper_by_time_expiry"] = upper.tolist()
    report["inference_note"] = (
        "binomial bounds are computed separately for each fixed trading-time/expiry "
        "cell across independent direct-P paths, with Bonferroni familywise level 5%; "
        "pooled state-expiry counts are descriptive and are not treated as independent"
    )


def run(report_root: Path = _REPORT_ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(report_root.glob("lp-control-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if "direct_P_positive_margin_count_by_time_expiry" not in report:
            continue
        report["experiment"] = "static arbitrage test (one expiry)"
        report["test_name"] = "static arbitrage test"
        _repair_cellwise_inference(report)
        _atomic_json(path, report)
        configuration = report["configuration"]
        top = report.get("top_opportunities", [])
        counts = np.asarray(
            report["direct_P_positive_margin_count_by_time_expiry"], dtype=np.int64
        )
        positive_cells = np.argwhere(counts > 0)
        rows.append(
            {
                "generator": configuration["generator"],
                "frequency": configuration["trading_frequency"],
                "paths": configuration["path_count"],
                "hedge": configuration["hedge"],
                "cost_rate": configuration["cost_rate"],
                "sample_paths": report["sample_paths"],
                "evaluated_state_expiry_pairs": report[
                    "evaluated_state_expiry_pairs"
                ],
                "positive_margin_count": report["positive_margin_count"],
                "direct_P_positive_margin_count": report[
                    "direct_P_positive_margin_count"
                ],
                "importance_positive_margin_count": report[
                    "importance_positive_margin_count"
                ],
                "direct_P_confirmed_time_expiry_cells": report[
                    "direct_P_confirmed_time_expiry_cells"
                ],
                "direct_P_positive_cells": positive_cells.tolist(),
                "maximum_direct_P_cell_count": int(np.max(counts)),
                "maximum_margin": (
                    max(float(item["margin"]) for item in top) if top else None
                ),
                "minimum_top_margin_over_tolerance": (
                    min(
                        float(item["margin"]) / float(item["numerical_tolerance"])
                        for item in top
                    )
                    if top
                    else None
                ),
                "report": str(path),
            }
        )
    rows.sort(
        key=lambda row: (
            row["paths"],
            row["generator"],
            row["frequency"],
            row["hedge"],
            row["cost_rate"],
        )
    )
    hierarchy: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["paths"], row["generator"], row["frequency"], row["hedge"])
        grouped.setdefault(key, []).append(row)
    for key, group in grouped.items():
        group.sort(key=lambda row: row["cost_rate"])
        passed = all(
            left["positive_margin_count"] >= right["positive_margin_count"]
            for left, right in zip(group, group[1:], strict=False)
        )
        hierarchy.append(
            {
                "paths": key[0],
                "generator": key[1],
                "frequency": key[2],
                "hedge": key[3],
                "costs": [row["cost_rate"] for row in group],
                "positive_margin_counts": [
                    row["positive_margin_count"] for row in group
                ],
                "passed": passed,
            }
        )
    summary = {
        "status": "complete",
        "experiment": "static arbitrage test summary (one expiry)",
        "test_name": "static arbitrage test",
        "rows": rows,
        "cost_hierarchy": hierarchy,
        "all_cost_hierarchy_checks_pass": all(item["passed"] for item in hierarchy),
        "numerical_note": (
            "a positive margin exceeds an action-specific 32*eps32 operand band; "
            "inference is cellwise over independent direct-P paths"
        ),
    }
    if not all(math.isfinite(row["cost_rate"]) for row in rows):
        raise FloatingPointError("summary contains a non-finite cost rate")
    output = report_root / "payoff-cone-control-summary.json"
    _atomic_json(output, summary)
    print(f"Wrote {len(rows)} static-arbitrage rows to {output}", flush=True)
    return summary


if __name__ == "__main__":
    run()
