"""Fit one hedge objective against authenticated generator return books."""

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


def replay_return_books(
    *,
    model: str,
    book_directory: Path,
    reference_run: Path,
    output_directory: Path,
    objective: ScenarioObjective,
    risk_parameter: float,
    raw_action_ridge: float,
    stop_after: int | None = None,
    continue_cap_limited: bool = False,
) -> None:
    """Refit all digital targets without retraining or resampling the generator."""
    panel = PublicHedgingPanel(default_public_candidate_path(), validate=False)
    reference = json.loads(reference_run.read_text(encoding="utf-8"))
    targets = daily_digital_targets(maturity_business_days=1)
    labels = [target.label for target in targets]
    quantities = np.asarray(reference["target_quantities"], dtype=np.float64)
    if (
        reference["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
        or reference["target_labels"] != labels
        or quantities.shape != (len(labels),)
    ):
        raise ValueError("reference targets do not match the current public dataset")
    config = GeneratedHedgeConfig(
        scenario_count=10_000,
        objective=objective,
        risk_parameter=risk_parameter,
        mean_reduction=MeanReduction.DEMEAN,
        cost_multiplier=1.0,
        position_cap=1_000.0,
        raw_action_ridge=raw_action_ridge,
    )
    identity = {
        "schema": "generated-objective-return-book-replay-v1",
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "model": model,
        "book_directory": str(book_directory.resolve()),
        "reference_run_sha256": file_sha256(reference_run),
        "objective": objective.value,
        "risk_parameter": risk_parameter,
        "raw_action_ridge": raw_action_ridge,
        "position_cap": config.position_cap,
        "optimizer_certificate_tolerance": config.certificate_tolerance,
        "mean_reduction": config.mean_reduction.value,
        "target_labels": labels,
        "target_quantities": quantities.tolist(),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("return-book objective run identity changed")
    else:
        _write_json(manifest, identity)
    records_directory = output_directory / "records"
    records_directory.mkdir(exist_ok=True)
    books = sorted(book_directory.glob("2024-*.npz"))
    if stop_after is not None:
        books = books[:stop_after]
    started = monotonic()
    for index, book_path in enumerate(books, start=1):
        origin_date = date.fromisoformat(book_path.stem)
        destination = records_directory / f"{origin_date.isoformat()}.json"
        failure_path = output_directory / "failures" / f"{origin_date.isoformat()}.json"
        book_sha256 = file_sha256(book_path)
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))["results"]
            if (
                [result["claim"] for result in existing] != labels
                or any(
                    result["diagnostics"].get("source_scenario_sha256") != book_sha256
                    or not result["converged"]
                    or result["cap_hits"]
                    for result in existing
                )
            ):
                raise RuntimeError(f"replayed return book changed: {origin_date}")
            print(f"replay {index}/{len(books)} {origin_date} reused", flush=True)
            continue
        if continue_cap_limited and failure_path.exists():
            existing = json.loads(failure_path.read_text(encoding="utf-8"))["results"]
            if (
                [result["claim"] for result in existing] == labels
                and all(
                    result["diagnostics"].get("source_scenario_sha256") == book_sha256
                    and (result["converged"] or result["cap_hits"])
                    for result in existing
                )
                and any(result["cap_hits"] for result in existing)
            ):
                print(f"replay {index}/{len(books)} {origin_date} cap-limited reused", flush=True)
                continue
        samples = tuple(
            build_daily_digital_sample(panel, origin_date=origin_date, target=target)
            for target in targets
        )
        with np.load(book_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            if (
                metadata["schema"] != "hedging-generated-return-book-v1"
                or metadata["model"] != model
                or metadata["origin_date"] != origin_date.isoformat()
                or metadata["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
                or metadata["reference_run_sha256"] != identity["reference_run_sha256"]
                or metadata["target_labels"] != labels
                or metadata["target_quantities"] != quantities.tolist()
            ):
                raise RuntimeError(f"return-book provenance changed: {origin_date}")
            instruments = np.asarray(saved["instrument_returns"], dtype=np.float64)
            target_returns = np.asarray(saved["target_returns"], dtype=np.float64)
        if (
            instruments.shape != (10_000, 1 + len(samples[0].hedge_quotes))
            or target_returns.shape != (10_000, len(targets))
            or any(sample.hedge_quotes != samples[0].hedge_quotes for sample in samples)
        ):
            raise RuntimeError(f"return-book geometry changed: {origin_date}")
        scenarios = tuple(
            GeneratedDailyScenarios(
                spot_returns=instruments[:, 0],
                option_returns=instruments[:, 1:],
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
            instruments,
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
            training_mean = float(reduced.weights @ scenario_wealth)
            training_sd = float(
                np.sqrt(reduced.weights @ (scenario_wealth - training_mean) ** 2)
            )
            result = evaluate_daily_digital_hedge(
                sample,
                model=model,
                measure=(model.removeprefix("gromm-") if model.startswith("gromm-") else "historical-P"),
                spot_action=float(fit.action[0]),
                option_actions=fit.action[1:],
                model_price=sample.market_price,
                objective_value=fit.objective_value,
                converged=fit.converged,
                cap_hits=fit.cap_hits,
                claim_quantity=float(quantity),
                diagnostics={
                    "source_scenario_sha256": book_sha256,
                    "objective": objective.value,
                    "risk_parameter": risk_parameter,
                    "raw_action_ridge": raw_action_ridge,
                    "optimizer_certificate_tolerance": config.certificate_tolerance,
                    "training_standard_deviation": training_sd,
                    "training_standard_deviation_per_unit_payout": training_sd / float(quantity),
                    "optimizer_kkt_residual": fit.optimizer.maximum_kkt_residual,
                    "optimizer_solver_message": fit.optimizer.solver_message,
                    "optimizer_iterations": fit.optimizer.iterations,
                },
            ).to_dict()
            results.append(result)
        if any(not result["converged"] or result["cap_hits"] for result in results):
            failures = output_directory / "failures"
            failures.mkdir(exist_ok=True)
            _write_json(failure_path, {
                "origin_date": origin_date.isoformat(),
                "results": results,
            })
            if not continue_cap_limited or any(
                not result["converged"] and not result["cap_hits"] for result in results
            ):
                raise RuntimeError(f"generated objective replay failed: {origin_date}")
            elapsed = monotonic() - started
            eta = elapsed * (len(books) - index) / index
            cap_hits = sum(result["cap_hits"] > 0 for result in results)
            print(
                f"replay {index}/{len(books)} {origin_date} cap-limited "
                f"targets={cap_hits}/{len(results)} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
            continue
        _write_json(destination, {"origin_date": origin_date.isoformat(), "results": results})
        elapsed = monotonic() - started
        eta = elapsed * (len(books) - index) / index
        print(
            f"replay {index}/{len(books)} {origin_date} elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--book-directory", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--objective", type=ScenarioObjective, required=True)
    parser.add_argument("--risk-parameter", type=float, required=True)
    parser.add_argument("--raw-action-ridge", type=float, default=0.0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--continue-cap-limited", action="store_true")
    replay_return_books(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
