"""Causal CPU rolling FuNVol hedge evaluation on the public panel."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date
from pathlib import Path
from time import monotonic

from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import daily_digital_targets
from quantdata.sdlv.daily.front_spot.hedging.benchmark.sizing import (
    daily_target_notional_artifact_path,
    load_authenticated_daily_target_notionals,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.adapter import run_funvol_daily_book
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import (
    FunvolConfig,
    FunvolTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import GeneratedHedgeConfig
from quantdata.torch import BaseEnvironment, torch


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_records(
    records: tuple[dict[str, object], ...],
    *,
    previous_date: str,
    allow_spot_dispersion_warning: bool = False,
) -> bool:
    if len(records) != 10:
        raise RuntimeError(f"FuNVol expected ten one-day digitals, got {len(records)}")
    dispersion_passed = True
    for record in records:
        diagnostics = record["diagnostics"]
        if not record["converged"] or record["cap_hits"]:
            raise RuntimeError(f"FuNVol hedge failed numerical gate: {record['claim']}")
        if diagnostics["warm_start_date"] != previous_date:
            raise RuntimeError(f"FuNVol wrong weight inheritance for {record['claim']}")
        generated_std = diagnostics["generated_return_std"]
        historical_std = diagnostics["historical_return_std"]
        if not (math.isfinite(generated_std) and math.isfinite(historical_std) and historical_std > 0):
            raise RuntimeError("FuNVol spot dispersion diagnostics are invalid")
        current_dispersion_passed = 0.5 * historical_std <= generated_std <= 2.0 * historical_std
        dispersion_passed = dispersion_passed and current_dispersion_passed
        if not current_dispersion_passed and not allow_spot_dispersion_warning:
            raise RuntimeError(
                f"FuNVol spot dispersion gate failed: {generated_std} vs {historical_std}"
            )
        maximum_action = max(abs(value) for value in record["actions"].values())
        if maximum_action > 100.0:
            raise RuntimeError(f"FuNVol excessive action {maximum_action:.4g}: {record['claim']}")
    return dispersion_passed


def run_rolling_funvol(
    *,
    dataset: Path,
    output_directory: Path,
    initializer_checkpoint: Path,
    start_date: date,
    end_date: date,
    epochs: int,
    scenario_count: int,
    allow_spot_dispersion_warning: bool = False,
) -> None:
    """Run one selected objective over consecutive dates, preserving parent hashes."""
    if start_date > end_date:
        raise ValueError("FuNVol evaluation date range is reversed")
    panel = PublicHedgingPanel(dataset)
    root = os.environ.get("QUANTDATA_ROOT")
    if root is None:
        raise RuntimeError("QUANTDATA_ROOT is required for authenticated 2022 target notionals")
    cache_directory = Path(root) / "processing" / "hedging" / "target-notionals"
    artifact = daily_target_notional_artifact_path(panel, cache_directory=cache_directory)
    artifact_sha256 = file_sha256(artifact)
    calibration = load_authenticated_daily_target_notionals(
        panel,
        cache_directory=cache_directory,
        expected_artifact_sha256=artifact_sha256,
    )
    targets = daily_digital_targets(maturity_business_days=1)
    claim_quantities = {
        target.label: calibration.quantity_by_label[target.label] for target in targets
    }
    dates = tuple(day for day in panel.dates if start_date <= day <= end_date)
    if not dates:
        raise ValueError("FuNVol evaluation range contains no public dates")
    if dates[-1] == panel.dates[-1]:
        raise ValueError("FuNVol last origin has no next-day market mark")
    initializer_sha256 = file_sha256(initializer_checkpoint)
    config = FunvolConfig()
    training = FunvolTrainConfig(epochs=epochs, validation_fraction=0.0)
    hedge = GeneratedHedgeConfig(scenario_count=scenario_count)
    identity = {
        "schema": "funvol-rolling-daily-v1",
        "dataset": str(dataset),
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "target_notional_artifact_sha256": artifact_sha256,
        "initializer_checkpoint": str(initializer_checkpoint),
        "initializer_sha256": initializer_sha256,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "model_config": repr(config),
        "train_config": repr(training),
        "hedge_config": repr(hedge),
        "hedge_config_scenario_count": scenario_count,
        "allow_spot_dispersion_warning": allow_spot_dispersion_warning,
        "target_labels": [target.label for target in targets],
        "target_quantities": claim_quantities,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("FuNVol rolling run identity changed")
    else:
        _write_json(manifest, identity)
    record_directory = output_directory / "records"
    checkpoint_directory = output_directory / "checkpoints"
    scenario_directory = output_directory / "scenarios"
    diagnostic_directory = output_directory / "diagnostics"
    record_directory.mkdir(exist_ok=True)
    checkpoint_directory.mkdir(exist_ok=True)
    scenario_directory.mkdir(exist_ok=True)
    diagnostic_directory.mkdir(exist_ok=True)
    previous_checkpoint = initializer_checkpoint
    previous_date = str(
        torch.load(initializer_checkpoint, map_location="cpu", weights_only=False)["as_of_date"]
    )
    started = monotonic()
    for index, origin_date in enumerate(dates):
        checkpoint = checkpoint_directory / f"{origin_date.isoformat()}.pt"
        record_path = record_directory / f"{origin_date.isoformat()}.json"
        if record_path.exists():
            records = tuple(json.loads(record_path.read_text(encoding="utf-8")))
            dispersion_passed = _validate_records(
                records,
                previous_date=previous_date,
                allow_spot_dispersion_warning=allow_spot_dispersion_warning,
            )
            if not checkpoint.exists() or any(
                record["diagnostics"]["fit_checkpoint_sha256"] != file_sha256(checkpoint)
                for record in records
            ):
                raise RuntimeError(f"FuNVol completed record/checkpoint mismatch: {origin_date}")
        else:
            results = run_funvol_daily_book(
                panel,
                origin_date=origin_date,
                checkpoint_path=checkpoint,
                warm_start_checkpoint=previous_checkpoint,
                model_config=config,
                train_config=training,
                hedge_config=hedge,
                claim_quantities=claim_quantities,
                scenario_output_path=scenario_directory / f"{origin_date.isoformat()}.npz",
                environment=BaseEnvironment(precision=64, device="cpu"),
            )
            records = tuple(result.to_dict() for result in results)
            dispersion_passed = _validate_records(
                records,
                previous_date=previous_date,
                allow_spot_dispersion_warning=allow_spot_dispersion_warning,
            )
            _write_json(
                diagnostic_directory / f"{origin_date.isoformat()}.json",
                {
                    "origin_date": origin_date.isoformat(),
                    "dataset_manifest_sha256": panel.dataset_manifest_sha256,
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "scenario_book_sha256": records[0]["diagnostics"]["scenario_book_sha256"],
                    "generated_return_std": records[0]["diagnostics"]["generated_return_std"],
                    "historical_return_std": records[0]["diagnostics"]["historical_return_std"],
                    "spot_dispersion_gate_passed": dispersion_passed,
                    "static_arbitrage_screen": records[0]["diagnostics"]["static_arbitrage_screen"],
                    "all_converged": all(record["converged"] for record in records),
                    "total_cap_hits": sum(record["cap_hits"] for record in records),
                    "maximum_absolute_action": max(
                        abs(value) for record in records for value in record["actions"].values()
                    ),
                },
            )
            _write_json(record_path, records)
        previous_checkpoint = checkpoint
        previous_date = origin_date.isoformat()
        elapsed = monotonic() - started
        eta = elapsed * (len(dates) - index - 1) / (index + 1)
        print(
            f"FuNVol rolling {index + 1}/{len(dates)} {origin_date}: "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s "
            f"spot_dispersion={'pass' if dispersion_passed else 'warning'}",
            flush=True,
        )


def main() -> None:
    """Parse an isolated April–May benchmark invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initializer-checkpoint", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 4, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 5, 31))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--scenario-count", type=int, default=10_000)
    parser.add_argument("--allow-spot-dispersion-warning", action="store_true")
    arguments = parser.parse_args()
    run_rolling_funvol(**vars(arguments))


if __name__ == "__main__":
    main()
