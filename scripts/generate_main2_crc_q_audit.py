"""Materialize the retained CRC reset-control audit for ``main2.tex``.

The 100,000-path population and neural checkpoint deliberately remain below
``QUANTDATA_ROOT``.  This script retains only their small, provenance-checked
summary in the source tree and writes the TeX macros consumed by the paper.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("QUANTDATA_ROOT", "/home/hans/quantdata_data"))
DEFAULT_RUN = (
    DATA_ROOT
    / "experiments/dysanos/bates-crc/"
    "spx-2025-08-29-v7-q-diagnostic-invalid-reset-gate"
)
DEFAULT_GATE = (
    DATA_ROOT
    / "experiments/dysanos/bates-crc/"
    "spx-2025-08-29-v7-exact-reset-gate/exact-reset-validation.json"
)
DEFAULT_OUTPUT_JSON = REPOSITORY / "quantdata/sdlv/tex/generated/crc_q_audit_diagnostic.json"
DEFAULT_OUTPUT_TEX = REPOSITORY / "quantdata/sdlv/tex/generated/crc_q_audit_diagnostic.tex"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected an object in {path}")
    return payload


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _format_float(value: object, decimals: int = 3) -> str:
    return f"{float(value):.{decimals}f}"


def _format_tex_scientific(value: object, decimals: int = 2) -> str:
    """Format a small number as TeX scientific notation rather than ``e`` text."""

    mantissa, exponent = f"{float(value):.{decimals}e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def build_summary(
    *,
    q_audit: dict[str, Any],
    exact_gate: dict[str, Any],
    population: dict[str, Any],
) -> dict[str, Any]:
    """Validate the retained reports and return their paper-level summary."""

    _require(q_audit.get("schema") == "gambara-teichmann-crc-q-audit-v1", "unexpected Q-audit schema")
    _require(
        exact_gate.get("schema") == "gambara-teichmann-crc-exact-reset-validation-v1",
        "unexpected exact-reset gate schema",
    )
    _require(
        population.get("schema") == "gambara-teichmann-crc-restricted-population-v1",
        "unexpected CRC population schema",
    )
    _require(exact_gate.get("status") == "rejected", "this artifact is reserved for the failed CRC control")

    config = q_audit["config"]
    unconditional = q_audit["unconditional"]
    conditional = q_audit["conditional"]
    squared = conditional["squared"]
    entropy = conditional["entropy"]
    path_count = int(config["path_count"])
    _require(path_count == 100_000, "main2 CRC audit requires 100,000 paths")
    _require(int(population["config"]["path_count"]) == path_count, "population/audit path counts differ")
    _require(int(squared["pair_count"]) * 2 == path_count, "squared audit pairing is inconsistent")
    _require(int(entropy["pair_count"]) * 2 == path_count, "entropy audit pairing is inconsistent")

    return {
        "schema": "main2-crc-q-audit-diagnostic-v1",
        "provenance": {
            "q_audit_schema": q_audit["schema"],
            "exact_gate_schema": exact_gate["schema"],
            "population_schema": population["schema"],
            "target_fit_report": population["target_fit_report"],
            "checkpoint": population["checkpoint"],
        },
        "exact_reset_gate": {
            "status": exact_gate["status"],
            "sample_count": int(exact_gate["sample_count"]),
            "accepted_count": int(exact_gate["accepted_count"]),
            "tolerance": float(exact_gate["tolerance"]),
            "median_abs_price_error": float(exact_gate["median_abs_price_error"]),
            "maximum_abs_price_error": float(exact_gate["maximum_abs_price_error"]),
        },
        "population": {
            "path_count": path_count,
            "pair_count": int(squared["pair_count"]),
            "horizon_days": int(population["config"]["horizon_days"]),
            "steps_per_day": int(population["config"]["steps_per_day"]),
            "maximum_reset_price_error": float(population["maximum_reset_price_error"]),
        },
        "unconditional": {
            "tested_coordinates": int(unconditional["tested_coordinates"]),
            "maximum_absolute_z": float(unconditional["maximum_absolute_z"]),
            "p99_absolute_z": float(unconditional["p99_absolute_z"]),
            "count_above_2_576": int(unconditional["count_above_2_576"]),
            "count_above_5": int(unconditional["count_above_5"]),
        },
        "conditional": {
            "squared_t_statistic": float(squared["t_statistic"]),
            "entropy_t_statistic": float(entropy["t_statistic"]),
        },
    }


def _tex(summary: dict[str, Any]) -> str:
    gate = summary["exact_reset_gate"]
    population = summary["population"]
    unconditional = summary["unconditional"]
    conditional = summary["conditional"]
    macros = {
        "CrcQPathCount": str(population["path_count"]),
        "CrcQPairCount": str(population["pair_count"]),
        "CrcQHorizonDays": str(population["horizon_days"]),
        "CrcQStepsPerDay": str(population["steps_per_day"]),
        "CrcResetGateSamples": str(gate["sample_count"]),
        "CrcResetGateAccepted": str(gate["accepted_count"]),
        "CrcResetGateTolerance": _format_tex_scientific(gate["tolerance"], decimals=0),
        "CrcResetGateMedianError": _format_tex_scientific(gate["median_abs_price_error"]),
        "CrcResetGateMaximumError": _format_tex_scientific(gate["maximum_abs_price_error"]),
        "CrcPopulationMaximumResetError": _format_tex_scientific(population["maximum_reset_price_error"]),
        "CrcQCoordinates": str(unconditional["tested_coordinates"]),
        "CrcQMaximumAbsZ": _format_float(unconditional["maximum_absolute_z"]),
        "CrcQPercentileNineNineAbsZ": _format_float(unconditional["p99_absolute_z"]),
        "CrcQAboveTwoFiveSevenSix": str(unconditional["count_above_2_576"]),
        "CrcQAboveFive": str(unconditional["count_above_5"]),
        "CrcQSquaredT": _format_float(conditional["squared_t_statistic"]),
        "CrcQEntropyT": _format_float(conditional["entropy_t_statistic"]),
    }
    return "% Generated by scripts/generate_main2_crc_q_audit.py; do not edit.\n" + "".join(
        f"\\providecommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-audit", type=Path, default=DEFAULT_RUN / "q-audit.json")
    parser.add_argument("--exact-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--population-report", type=Path, default=DEFAULT_RUN / "population-report.json")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    arguments = parser.parse_args()

    summary = build_summary(
        q_audit=_read_json(arguments.q_audit),
        exact_gate=_read_json(arguments.exact_gate),
        population=_read_json(arguments.population_report),
    )
    _write_atomic(arguments.output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_atomic(arguments.output_tex, _tex(summary))


if __name__ == "__main__":
    main()
