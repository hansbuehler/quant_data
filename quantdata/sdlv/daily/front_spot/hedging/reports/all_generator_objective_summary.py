"""Compare all generator objective cells on the same 44 digital-hedging dates."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.reports.objective_grid_summary import (
    summarize_objective_grid,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.sanos_objective_grid_summary import (
    summarize_sanos_objective_grid,
)

MODEL_LABELS = (
    ("historical-sanos", "Historical SANOS"),
    ("funvol", "FuNVol"),
    ("volgan", "VolGAN"),
    ("gromm-P", "GROMM P"),
    ("gromm-Q", "GROMM Q"),
    ("diffusion-base", "AD-Seq-Vol"),
    ("diffusion-ft", "AD-Seq-Vol-FT"),
)
OBJECTIVES = ("mean-variance", "entropy", "power-exponential", "cvar")
RISKS = ("1", "10", "100", "1000")


def _cell(rows: list[dict[str, object]], objective: str, risk: str) -> dict[str, object]:
    matches = [
        row for row in rows
        if row["objective"] == objective and row["risk_parameter"] == risk
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing or duplicate objective cell: {objective} {risk}")
    return matches[0]


def _format_cell(row: dict[str, object], metric: str) -> str:
    if row["status"] == "complete":
        return f"{float(row[metric]):.2f}"
    if row["status"] == "cap-limited":
        return f"{float(row[metric]):.2f}†"
    return "failed" if row["failure_dates"] else "pending"


def _format_cap_hits(row: dict[str, object]) -> str:
    if row["status"] in ("complete", "cap-limited"):
        return f"{row['cap_hit_count']}/440 ({float(row['cap_hit_percentage']):.1f}%)"
    return "failed" if row["failure_dates"] else "pending"


def summarize_all_objectives(
    *,
    generated_root: Path,
    historical_root: Path,
) -> str:
    """Show matched SD and entry cost without implying partial-period comparability."""
    generated = summarize_objective_grid(generated_root)
    historical = summarize_sanos_objective_grid(historical_root)
    if (
        generated["cohort"] != historical["cohort"]
        or abs(
            float(generated["unhedged"]["sd_pnl_kusd"])
            - float(historical["unhedged"]["sd_pnl_kusd"])
        ) > 1.0e-8
    ):
        raise RuntimeError("generated and historical objective cohorts differ")
    rows_by_model: dict[str, list[dict[str, object]]] = {
        model: [row for row in generated["rows"] if row["generator"] == model]
        for model, _ in MODEL_LABELS
        if model != "historical-sanos"
    }
    rows_by_model["historical-sanos"] = [
        {**row, "objective": "cvar" if row["objective"] == "cvar-oce" else row["objective"]}
        for row in historical["rows"]
    ]
    labels = [label for _, label in MODEL_LABELS]
    lines = [
        "# Matched hedge-objective comparison",
        "",
        str(generated["cohort"]),
        "",
        "All entries are USD thousands per USD 1 million digital payout. "
        "Each cell averages the ten target-specific statistics over all 44 dates; "
        "pending and failed cells "
        "are never replaced with partial-period statistics.",
        "",
        f"Unhedged standard deviation: {float(generated['unhedged']['sd_pnl_kusd']):.2f}.",
        "",
    ]
    for title, metric in (
        ("Out-of-sample P&L standard deviation", "sd_pnl_kusd"),
        ("Mean hedge entry cost", "mean_entry_cost_kusd"),
    ):
        lines.extend(
            (
                f"## {title}",
                "",
                "| Objective | λ | " + " | ".join(labels) + " |",
                "|---|---:|" + "---:|" * len(labels),
            )
        )
        for objective in OBJECTIVES:
            for risk in RISKS:
                values = [
                    _format_cell(_cell(rows_by_model[model], objective, risk), metric)
                    for model, _ in MODEL_LABELS
                ]
                lines.append(
                    f"| {objective} | {risk} | " + " | ".join(values) + " |"
                )
        lines.append("")
    lines.extend(
        (
            "## Position-cap hit rate",
            "",
            "A hit means at least one hedge instrument reached its position limit "
            "for that target on that date; the denominator is 44 dates × 10 targets.",
            "",
            "| Objective | λ | " + " | ".join(labels) + " |",
            "|---|---:|" + "---:|" * len(labels),
        )
    )
    for objective in OBJECTIVES:
        for risk in RISKS:
            values = [
                _format_cap_hits(_cell(rows_by_model[model], objective, risk))
                for model, _ in MODEL_LABELS
            ]
            lines.append(f"| {objective} | {risk} | " + " | ".join(values) + " |")
    lines.append("")
    lines.extend(
        (
            "A failed cell did not pass its numerical or position-cap gate. "
            "The historical SANOS column "
            "uses 252 causal return scenarios; the generators use 10,000 paths per date. "
            "The current FuNVol replay retains its source-run position cap of 1,000,000, "
            "whereas the other columns use a cap of 1,000. "
            "A mean entry cost above 1,000 exceeds the "
            "target's USD 1 million payout.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = summarize_all_objectives(
        generated_root=arguments.generated_root,
        historical_root=arguments.historical_root,
    )
    arguments.output.write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
