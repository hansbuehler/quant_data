"""Replay retained one-day generator scenarios across hedge objectives."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np

from quantdata.hedging.mean_reduction import MeanReduction, reduce_one_day_means
from quantdata.hedging.scenario_optimization import ScenarioObjective
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedDailyScenarios,
    GeneratedHedgeConfig,
    fit_generated_objective_book,
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def replay_scenario_books(
    *,
    source_run: Path,
    scenario_directory: Path,
    output_directory: Path,
    objective: ScenarioObjective,
    risk_parameter: float,
    raw_action_ridge: float,
    stop_after: int | None = None,
    continue_cap_limited: bool = False,
) -> None:
    """Refit static actions without retraining or resampling the generator."""
    source = json.loads((source_run / "run.json").read_text(encoding="utf-8"))
    if source["model"] != "volgan":
        raise ValueError("this replay currently authenticates VolGAN source runs")
    panel = PublicHedgingPanel(default_public_candidate_path(), validate=False)
    if source["dataset_manifest_sha256"] != panel.dataset_manifest_sha256:
        raise RuntimeError("source run uses a different public dataset")
    targets = daily_digital_targets(maturity_business_days=1)
    labels = [target.label for target in targets]
    if labels != source["target_labels"]:
        raise RuntimeError("source target catalog differs from the 1D cohort")
    quantities = np.asarray(source["target_quantities"], dtype=np.float64)
    if quantities.shape != (len(targets),) or np.any(quantities <= 0.0):
        raise ValueError("source target quantities are invalid")
    source_hedge = source["model_config_snapshot"]["fields"]["hedge_config"]
    if source_hedge["mean_reduction"] != "demean":
        raise ValueError("source run did not use the agreed demeaning convention")
    config = GeneratedHedgeConfig(
        scenario_count=int(source_hedge["scenario_count"]),
        objective=objective,
        risk_parameter=risk_parameter,
        mean_reduction=MeanReduction.DEMEAN,
        cost_multiplier=float(source_hedge["cost_multiplier"]),
        position_cap=float(source_hedge["position_cap"]),
        raw_action_ridge=raw_action_ridge,
    )
    identity = {
        "schema": "generated-objective-replay-v1",
        "source_run": str(source_run),
        "source_run_sha256": file_sha256(source_run / "run.json"),
        "scenario_directory": str(scenario_directory),
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "model": source["model"],
        "measure": source["measure"],
        "objective": objective.value,
        "risk_parameter": risk_parameter,
        "raw_action_ridge": raw_action_ridge,
        "optimizer_certificate_tolerance": config.certificate_tolerance,
        "mean_reduction": config.mean_reduction.value,
        "position_cap": config.position_cap,
        "target_labels": labels,
        "target_quantities": quantities.tolist(),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / "run.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        older_identity = {
            key: value
            for key, value in identity.items()
            if key != "optimizer_certificate_tolerance"
        }
        if existing == older_identity:
            if any(
                not result["converged"] or result["cap_hits"]
                for path in (output_directory / "records").glob("*.json")
                for result in json.loads(path.read_text(encoding="utf-8"))["results"]
            ):
                raise RuntimeError("legacy VolGAN record failed its stricter certificate")
            _write_json(manifest, identity)
        elif existing != identity:
            raise RuntimeError("generated objective replay identity changed")
    else:
        _write_json(manifest, identity)
    records_directory = output_directory / "records"
    records_directory.mkdir(exist_ok=True)
    scenario_paths = sorted(scenario_directory.glob("2024-*.npz"))
    if stop_after is not None:
        scenario_paths = scenario_paths[:stop_after]
    started = monotonic()
    for index, scenario_path in enumerate(scenario_paths, start=1):
        origin_date = date.fromisoformat(scenario_path.stem)
        destination = records_directory / f"{origin_date.isoformat()}.json"
        failure_path = output_directory / "failures" / f"{origin_date.isoformat()}.json"
        if destination.exists():
            print(f"replay {index}/{len(scenario_paths)} {origin_date} reused", flush=True)
            continue
        source_record = json.loads(
            (source_run / "records" / f"{origin_date.isoformat()}.json").read_text(
                encoding="utf-8"
            )
        )
        source_results = source_record["results"]
        scenario_sha256 = file_sha256(scenario_path)
        if continue_cap_limited and failure_path.exists():
            existing = json.loads(failure_path.read_text(encoding="utf-8"))["results"]
            if (
                [result["claim"] for result in existing] == labels
                and all(
                    result["diagnostics"].get("source_scenario_sha256") == scenario_sha256
                    and (result["converged"] or result["cap_hits"])
                    for result in existing
                )
                and any(result["cap_hits"] for result in existing)
            ):
                print(f"replay {index}/{len(scenario_paths)} {origin_date} cap-limited reused", flush=True)
                continue
        if [result["claim"] for result in source_results] != labels or any(
            result["diagnostics"]["scenario_book_sha256"] != scenario_sha256
            for result in source_results
        ):
            raise RuntimeError(f"source record and scenario book disagree: {origin_date}")
        samples = tuple(
            build_daily_digital_sample(panel, origin_date=origin_date, target=target)
            for target in targets
        )
        with np.load(scenario_path, allow_pickle=False) as saved:
            provenance = json.loads(str(saved["provenance_json"]))
            if (
                provenance["origin_date"] != origin_date.isoformat()
                or provenance["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
                or provenance["target_labels"] != labels
                or provenance["target_quantities"] != quantities.tolist()
                or provenance["fit_checkpoint_sha256"]
                != source_results[0]["diagnostics"]["fit_checkpoint_sha256"]
            ):
                raise RuntimeError(f"scenario provenance mismatch: {origin_date}")
            instrument_returns = np.asarray(saved["instrument_returns"], dtype=np.float64)
            target_returns = np.asarray(saved["target_returns"], dtype=np.float64)
        if instrument_returns.shape[1] != 1 + len(samples[0].hedge_quotes):
            raise RuntimeError(f"scenario hedge book changed: {origin_date}")
        if target_returns.shape != (len(instrument_returns), len(targets)):
            raise RuntimeError(f"scenario target book changed: {origin_date}")
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
        reduced = reduce_one_day_means(
            instrument_returns,
            target_returns,
            method=config.mean_reduction,
        )
        results = []
        for target_index, (sample, fit, quantity) in enumerate(
            zip(samples, fits, quantities, strict=True)
        ):
            scenario_wealth = (
                quantity * reduced.target_returns[:, target_index]
                + reduced.instrument_returns @ fit.action
            )
            weighted_mean = float(reduced.weights @ scenario_wealth)
            training_standard_deviation = float(
                np.sqrt(reduced.weights @ (scenario_wealth - weighted_mean) ** 2)
            )
            result = evaluate_daily_digital_hedge(
                sample,
                model="volgan",
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
                    "raw_action_ridge": raw_action_ridge,
                    "optimizer_certificate_tolerance": config.certificate_tolerance,
                    "training_standard_deviation": training_standard_deviation,
                    "training_standard_deviation_per_unit_payout": (
                        training_standard_deviation / float(quantity)
                    ),
                    "training_objective_value": fit.objective_value,
                    "optimizer_kkt_residual": fit.optimizer.maximum_kkt_residual,
                    "optimizer_solver_message": fit.optimizer.solver_message,
                    "optimizer_iterations": fit.optimizer.iterations,
                },
            ).to_dict()
            results.append(result)
        if any(not result["converged"] or result["cap_hits"] for result in results):
            failures_directory = output_directory / "failures"
            failures_directory.mkdir(exist_ok=True)
            _write_json(
                failure_path,
                {"origin_date": origin_date.isoformat(), "results": results},
            )
            if not continue_cap_limited or any(
                not result["converged"] and not result["cap_hits"] for result in results
            ):
                raise RuntimeError(f"generated objective replay failed: {origin_date}")
            elapsed = monotonic() - started
            remaining = elapsed * (len(scenario_paths) - index) / index
            cap_hits = sum(result["cap_hits"] > 0 for result in results)
            print(
                f"replay {index}/{len(scenario_paths)} {origin_date} cap-limited "
                f"targets={cap_hits}/{len(results)} elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                flush=True,
            )
            continue
        _write_json(destination, {"origin_date": origin_date.isoformat(), "results": results})
        elapsed = monotonic() - started
        remaining = elapsed * (len(scenario_paths) - index) / index
        print(
            f"replay {index}/{len(scenario_paths)} {origin_date} "
            f"elapsed={elapsed:.0f}s eta={remaining:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--scenario-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--objective", type=ScenarioObjective, required=True)
    parser.add_argument("--risk-parameter", type=float, required=True)
    parser.add_argument("--raw-action-ridge", type=float, default=0.0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--continue-cap-limited", action="store_true")
    replay_scenario_books(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
