"""Audited monthly FuNVol one-day hedge summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.torch import torch


def summarize_rolling_funvol(output_directory: Path) -> dict[str, object]:
    """Require complete records and chain hashes before reporting monthly MAE."""
    run = json.loads((output_directory / "run.json").read_text(encoding="utf-8"))
    panel = PublicHedgingPanel(Path(run["dataset"]))
    if panel.dataset_manifest_sha256 != run["dataset_manifest_sha256"]:
        raise RuntimeError("FuNVol summary dataset manifest changed")
    if file_sha256(Path(run["initializer_checkpoint"])) != run["initializer_sha256"]:
        raise RuntimeError("FuNVol summary initializer checkpoint changed")
    start_date = date.fromisoformat(run["start_date"])
    end_date = date.fromisoformat(run["end_date"])
    dates = tuple(day for day in panel.dates if start_date <= day <= end_date)
    if not dates:
        raise RuntimeError("FuNVol summary range is empty")
    records_by_month: dict[str, list[dict[str, object]]] = defaultdict(list)
    per_target: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    previous_date = date.fromisoformat(run["start_date"])
    previous_sha256 = run["initializer_sha256"]
    for index, origin_date in enumerate(dates):
        if index == 0:
            previous_date = date.fromisoformat(
                torch.load(run["initializer_checkpoint"], map_location="cpu", weights_only=False)[
                    "as_of_date"
                ]
            )
        path = output_directory / "records" / f"{origin_date.isoformat()}.json"
        if not path.exists():
            raise RuntimeError(f"FuNVol summary is incomplete: {origin_date}")
        records = json.loads(path.read_text(encoding="utf-8"))
        if len(records) != len(run["target_labels"]):
            raise RuntimeError(f"FuNVol summary target count mismatch: {origin_date}")
        checkpoint = output_directory / "checkpoints" / f"{origin_date.isoformat()}.pt"
        checkpoint_sha256 = file_sha256(checkpoint)
        if {record["claim"] for record in records} != set(run["target_labels"]):
            raise RuntimeError(f"FuNVol summary target labels mismatch: {origin_date}")
        for record in records:
            diagnostics = record["diagnostics"]
            if (
                record["origin_date"] != origin_date.isoformat()
                or not record["converged"]
                or record["cap_hits"] != 0
                or diagnostics["fit_checkpoint_sha256"] != checkpoint_sha256
                or diagnostics["warm_start_sha256"] != previous_sha256
                or diagnostics["warm_start_date"] != previous_date.isoformat()
                or diagnostics["generated_paths"] < run["hedge_config_scenario_count"]
                or not np.isclose(
                    record["claim_quantity"], run["target_quantities"][record["claim"]]
                )
            ):
                raise RuntimeError(
                    f"FuNVol summary validation failed: {origin_date} {record['claim']}"
                )
            month = origin_date.strftime("%Y-%m")
            records_by_month[month].append(record)
            per_target[month][record["claim"]].append(record)
        previous_date = origin_date
        previous_sha256 = checkpoint_sha256

    def aggregate(records: list[dict[str, object]]) -> dict[str, float | int]:
        hedged = np.asarray([row["hedged_pnl"] for row in records], dtype=np.float64)
        unhedged = np.asarray([row["unhedged_pnl"] for row in records], dtype=np.float64)
        costs = np.asarray([row["transaction_cost"] for row in records], dtype=np.float64)
        return {
            "observations": len(records),
            "hedged_mae_spot_units": float(np.mean(np.abs(hedged))),
            "unhedged_mae_spot_units": float(np.mean(np.abs(unhedged))),
            "hedged_mae_usd_per_million_origin_spot": float(1_000_000 * np.mean(np.abs(hedged))),
            "unhedged_mae_usd_per_million_origin_spot": float(
                1_000_000 * np.mean(np.abs(unhedged))
            ),
            "mean_transaction_cost_bps_spot": float(10_000 * np.mean(costs)),
        }

    return {
        "schema": "funvol-rolling-monthly-summary-v1",
        "dataset_manifest_sha256": run["dataset_manifest_sha256"],
        "target_notional_artifact_sha256": run["target_notional_artifact_sha256"],
        "initializer_sha256": run["initializer_sha256"],
        "dates": len(dates),
        "months": {
            month: {
                "all_targets": aggregate(rows),
                "by_target": {
                    target: aggregate(target_rows)
                    for target, target_rows in sorted(per_target[month].items())
                },
            }
            for month, rows in sorted(records_by_month.items())
        },
    }


def main() -> None:
    """Print a complete monthly report and save it beside the run manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = summarize_rolling_funvol(arguments.output_directory)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (arguments.output_directory / "monthly_summary.json").write_text(serialized, encoding="utf-8")
    print(serialized, end="", flush=True)


if __name__ == "__main__":
    main()
