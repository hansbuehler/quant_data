"""Run every requested hedge objective on one complete saved generator cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from quantdata.hedging.scenario_optimization import ScenarioObjective
from quantdata.sdlv.daily.front_spot.hedging.runners.replay_objective_return_books import (
    replay_return_books,
)


def run_grid(
    *,
    model: str,
    book_directory: Path,
    reference_run: Path,
    output_root: Path,
    raw_action_ridge: float = 0.001,
) -> None:
    """Preserve a separate explicit success or failure for every objective cell."""
    reference_root = reference_run.parent
    expected_dates = tuple(sorted(path.stem for path in (reference_root / "records").glob("2024-*.json")))
    actual_dates = tuple(sorted(path.stem for path in book_directory.glob("2024-*.npz")))
    if len(expected_dates) != 44 or actual_dates != expected_dates:
        raise RuntimeError(f"{model}: need the same 44 retained scenario dates as the reference")
    cells = tuple(
        (objective, risk)
        for objective in (
            ScenarioObjective.MEAN_VARIANCE,
            ScenarioObjective.ENTROPY,
            ScenarioObjective.POWER_EXPONENTIAL,
        )
        for risk in (1.0, 10.0, 100.0)
    ) + (
        (ScenarioObjective.MEAN_VARIANCE, float("inf")),
        (ScenarioObjective.CVAR_OCE, float("inf")),
    ) + tuple(
        (ScenarioObjective.CVAR_OCE, risk)
        for risk in (1.0, 10.0, 100.0)
    )
    model_root = output_root / model
    model_root.mkdir(parents=True, exist_ok=True)
    status_path = model_root / "GRID_STATUS.json"
    status = {
        "schema": "hedging-generator-objective-grid-status-v1",
        "model": model,
        "expected_dates": list(expected_dates),
        "cells": {},
    }
    if status_path.exists():
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        if saved["model"] != model or saved["expected_dates"] != list(expected_dates):
            raise RuntimeError(f"{model}: objective-grid status has another cohort")
        status = saved
    started = monotonic()
    for index, (objective, risk) in enumerate(cells, start=1):
        objective_label = "cvar" if objective is ScenarioObjective.CVAR_OCE else objective.value
        risk_label = "inf" if risk == float("inf") else f"{risk:g}"
        label = f"{objective_label}-lambda{risk_label}"
        destination = model_root / label
        if status["cells"].get(label, {}).get("status") in ("complete", "cap-limited"):
            print(f"objective cell {index}/{len(cells)} {model} {label} reused", flush=True)
            continue
        try:
            replay_return_books(
                model=model,
                book_directory=book_directory,
                reference_run=reference_run,
                output_directory=destination,
                objective=objective,
                risk_parameter=risk,
                raw_action_ridge=raw_action_ridge,
                continue_cap_limited=True,
            )
        except Exception as error:
            status["cells"][label] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_dates": len(tuple((destination / "records").glob("2024-*.json"))),
            }
            print(f"objective cell {index}/{len(cells)} {model} {label} FAILED: {error}", flush=True)
        else:
            records = {path.stem for path in (destination / "records").glob("2024-*.json")}
            flagged = {path.stem for path in (destination / "failures").glob("2024-*.json")}
            if records | flagged != set(expected_dates):
                raise RuntimeError(f"{model} {label}: scenario dates were not all evaluated")
            capped_count = len(flagged - records)
            cell_status = "cap-limited" if capped_count else "complete"
            status["cells"][label] = {
                "status": cell_status,
                "completed_dates": 44,
                "cap_limited_dates": capped_count,
            }
            print(
                f"objective cell {index}/{len(cells)} {model} {label} {cell_status}",
                flush=True,
            )
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(status_path)
        elapsed = monotonic() - started
        eta = elapsed * (len(cells) - index) / index
        print(
            f"objective grid {index}/{len(cells)} {model} elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--book-directory", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-action-ridge", type=float, default=0.001)
    run_grid(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
