"""Replay retained FuNVol one-day scenarios under a new hedge objective."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np

from quantdata.hedging.scenario_optimization import ScenarioObjective
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedDailyScenarios,
    GeneratedHedgeConfig,
    fit_generated_objective_book,
)


def replay_funvol_scenarios(
    *,
    source_directory: Path,
    output_directory: Path,
    objective: ScenarioObjective,
    risk_parameter: float,
    stop_after: int | None = None,
) -> None:
    """Fit a new action on each authenticated, unchanged scenario book."""
    source = json.loads((source_directory / "run.json").read_text(encoding="utf-8"))
    panel = PublicHedgingPanel(Path(source["dataset"]), validate=False)
    if panel.dataset_manifest_sha256 != source["dataset_manifest_sha256"]:
        raise RuntimeError("FuNVol replay dataset identity changed")
    targets = daily_digital_targets(maturity_business_days=1)
    labels = [target.label for target in targets]
    if labels != source["target_labels"]:
        raise RuntimeError("FuNVol replay target catalog changed")
    quantities = np.asarray([source["target_quantities"][label] for label in labels])
    config = replace(
        GeneratedHedgeConfig(scenario_count=source["hedge_config_scenario_count"]),
        objective=objective,
        risk_parameter=risk_parameter,
    )
    identity = {
        "schema": "funvol-objective-replay-v1",
        "source_directory": str(source_directory),
        "source_run_sha256": file_sha256(source_directory / "run.json"),
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "objective": objective.value,
        "risk_parameter": risk_parameter,
        "mean_reduction": config.mean_reduction.value,
        "cost_convention": "spot at mid; option entry only; formal exit reported",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("FuNVol replay identity changed")
    else:
        manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records_directory = output_directory / "records"
    records_directory.mkdir(exist_ok=True)
    scenario_paths = sorted((source_directory / "scenarios").glob("*.npz"))
    if stop_after is not None:
        scenario_paths = scenario_paths[:stop_after]
    started = monotonic()
    for index, scenario_path in enumerate(scenario_paths, start=1):
        origin_date = date.fromisoformat(scenario_path.stem)
        destination = records_directory / f"{origin_date.isoformat()}.json"
        if destination.exists():
            continue
        source_records = json.loads(
            (source_directory / "records" / f"{origin_date.isoformat()}.json").read_text(
                encoding="utf-8"
            )
        )
        scenario_sha256 = file_sha256(scenario_path)
        if [record["claim"] for record in source_records] != labels or any(
            record["diagnostics"]["scenario_book_sha256"] != scenario_sha256
            for record in source_records
        ):
            raise RuntimeError(f"FuNVol replay scenario provenance mismatch: {origin_date}")
        samples = tuple(
            build_daily_digital_sample(panel, origin_date=origin_date, target=target)
            for target in targets
        )
        with np.load(scenario_path) as saved:
            provenance = json.loads(str(saved["provenance_json"]))
            if (
                provenance["origin_date"] != origin_date.isoformat()
                or provenance["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
                or provenance["target_labels"] != labels
            ):
                raise RuntimeError(f"FuNVol replay saved provenance changed: {origin_date}")
            instrument_returns = np.asarray(saved["instrument_returns"], dtype=np.float64)
            target_returns = np.asarray(saved["target_returns"], dtype=np.float64)
        if instrument_returns.shape[1] != 1 + len(samples[0].hedge_quotes):
            raise RuntimeError(f"FuNVol replay hedge book changed: {origin_date}")
        if target_returns.shape != (len(instrument_returns), len(targets)):
            raise RuntimeError(f"FuNVol replay target scenarios changed: {origin_date}")
        scenarios = tuple(
            GeneratedDailyScenarios(
                spot_returns=instrument_returns[:, 0],
                option_returns=instrument_returns[:, 1:],
                claim_returns=target_returns[:, target_index],
                claim_terminal_values=target_returns[:, target_index] + sample.market_price,
            )
            for target_index, sample in enumerate(samples)
        )
        fits = fit_generated_objective_book(
            scenarios,
            samples[0],
            config=config,
            claim_quantities=quantities,
        )
        results = [
            evaluate_daily_digital_hedge(
                sample,
                model="funvol-single-asset",
                measure="historical-P",
                spot_action=float(fit.action[0]),
                option_actions=fit.action[1:],
                model_price=sample.market_price,
                objective_value=fit.objective_value,
                converged=fit.converged,
                cap_hits=fit.cap_hits,
                claim_quantity=float(quantity),
                diagnostics={
                    "source_scenario_sha256": scenario_sha256,
                    "objective": objective.value,
                    "risk_parameter": risk_parameter,
                    "mean_reduction": config.mean_reduction.value,
                },
            ).to_dict()
            for sample, fit, quantity in zip(samples, fits, quantities, strict=True)
        ]
        if any(not result["converged"] or result["cap_hits"] for result in results):
            raise RuntimeError(f"FuNVol replay did not converge: {origin_date}")
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        elapsed = monotonic() - started
        remaining = elapsed * (len(scenario_paths) - index) / index
        print(
            f"FuNVol replay {index}/{len(scenario_paths)} {origin_date}: "
            f"elapsed={elapsed:.0f}s eta={remaining:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--objective", type=ScenarioObjective, required=True)
    parser.add_argument("--risk-parameter", type=float, required=True)
    parser.add_argument("--stop-after", type=int)
    replay_funvol_scenarios(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
