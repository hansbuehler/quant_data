"""Replay the common hedge-objective grid on rolling historical SANOS returns."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np

from quantdata.hedging.mean_reduction import MeanReduction, reduce_one_day_means
from quantdata.hedging.scenario_optimization import (
    ScenarioObjective,
    ScenarioOptimizationConfig,
    fit_static_scenario_hedge,
)
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
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    build_historical_return_panel,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_identity(source_run: Path, panel: PublicHedgingPanel) -> tuple[dict, tuple[date, ...]]:
    source = json.loads((source_run / "run.json").read_text(encoding="utf-8"))
    if source["dataset_manifest_sha256"] != panel.dataset_manifest_sha256:
        raise RuntimeError("SANOS and comparison runs use different public datasets")
    labels = [target.label for target in daily_digital_targets(maturity_business_days=1)]
    quantities = np.asarray(source["target_quantities"], dtype=np.float64)
    if source["target_labels"] != labels or quantities.shape != (len(labels),):
        raise RuntimeError("comparison target catalog or quantities differ")
    if np.any(~np.isfinite(quantities) | (quantities <= 0.0)):
        raise RuntimeError("comparison claim quantities are invalid")
    dates = tuple(date.fromisoformat(path.stem) for path in sorted((source_run / "records").glob("2024-*.json")))
    if len(dates) != 44 or len(set(dates)) != 44:
        raise RuntimeError("comparison cohort does not contain 44 distinct dates")
    return source, dates


def _samples(panel: PublicHedgingPanel, origin_date: date) -> tuple:
    return tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in daily_digital_targets(maturity_business_days=1)
    )


def _prepare_one(panel: PublicHedgingPanel, origin_date: date, destination: Path) -> None:
    samples = _samples(panel, origin_date)
    returns = build_historical_return_panel(
        panel,
        origin_date=origin_date,
        option_quotes=samples[0].hedge_quotes,
        targets=tuple(sample.target for sample in samples),
        count=252,
        price_source="sanos",
    )
    if returns.training_terminal_dates[-1] > origin_date:
        raise RuntimeError("historical SANOS replay leaks a future transition")
    identity = {
        "origin_date": origin_date.isoformat(),
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "training_start": returns.training_origin_dates[0].isoformat(),
        "training_end": returns.training_terminal_dates[-1].isoformat(),
        "instrument_ids": returns.instrument_ids,
        "target_labels": returns.target_labels,
        "price_source": "sanos",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            instrument_returns=returns.instrument_returns,
            target_returns=returns.target_returns,
            identity_json=json.dumps(identity, sort_keys=True),
        )
    temporary.replace(destination)


def _fit_one(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    scenario_path: Path,
    quantities: np.ndarray,
    config: ScenarioOptimizationConfig,
) -> list[dict]:
    samples = _samples(panel, origin_date)
    instrument_ids = ("spot", *(quote.contract.contract_id for quote in samples[0].hedge_quotes))
    labels = tuple(sample.target.label for sample in samples)
    with np.load(scenario_path, allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity_json"]))
        instruments = np.asarray(saved["instrument_returns"], dtype=np.float64)
        targets = np.asarray(saved["target_returns"], dtype=np.float64)
    if (
        identity["origin_date"] != origin_date.isoformat()
        or identity["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
        or tuple(identity["instrument_ids"]) != instrument_ids
        or tuple(identity["target_labels"]) != labels
        or identity["training_end"] > origin_date.isoformat()
        or instruments.shape != (252, len(instrument_ids))
        or targets.shape != (252, len(labels))
    ):
        raise RuntimeError(f"SANOS scenario geometry or provenance changed: {origin_date}")
    reduction = reduce_one_day_means(instruments, targets, method=MeanReduction.DEMEAN)
    costs = np.asarray([0.0, *(quote.half_spread for quote in samples[0].hedge_quotes)])
    results = []
    for target_index, (sample, quantity) in enumerate(zip(samples, quantities, strict=True)):
        fit = fit_static_scenario_hedge(
            instrument_returns=reduction.instrument_returns,
            claim_returns=float(quantity) * reduction.target_returns[:, target_index],
            scenario_weights=reduction.weights,
            linear_transaction_costs=costs,
            config=config,
        )
        results.append(
            evaluate_daily_digital_hedge(
                sample,
                model="historical-sanos-objective-grid",
                measure="historical-P",
                spot_action=float(fit.action[0]),
                option_actions=fit.action[1:],
                model_price=sample.market_price,
                objective_value=fit.objective_value,
                converged=fit.converged,
                cap_hits=fit.cap_hits,
                claim_quantity=float(quantity),
                diagnostics={
                    "objective": config.objective.value,
                    "risk_parameter": config.risk_parameter,
                    "scenario_count": len(instruments),
                    "scenario_book_sha256": file_sha256(scenario_path),
                    "price_source": "sanos",
                    "mean_reduction": "demean",
                    "optimizer_kkt_residual": fit.maximum_kkt_residual,
                    "optimizer_solver_message": fit.solver_message,
                    "optimizer_iterations": fit.iterations,
                    "raw_action_ridge": config.raw_action_ridge,
                },
            ).to_dict()
        )
    return results


def run_grid(
    *,
    root: Path,
    source_run: Path,
    phase: str,
    objective: ScenarioObjective | None = None,
    risk_parameter: float | None = None,
    stop_after: int | None = None,
) -> None:
    """Prepare immutable scenario books or fit one objective/risk cell."""
    panel = PublicHedgingPanel(default_public_candidate_path(), validate=False)
    source, dates = _source_identity(source_run, panel)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "run.json"
    identity = {
        "schema": "historical-sanos-objective-grid-v1",
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "source_run_sha256": file_sha256(source_run / "run.json"),
        "source_run": str(source_run),
        "cohort": [origin_date.isoformat() for origin_date in dates],
        "target_labels": source["target_labels"],
        "target_quantities": source["target_quantities"],
        "observations": 252,
        "scenario_price_source": "sanos",
        "mean_reduction": "demean",
        "cost_convention": "spot at mid; option entry half-spread only",
        "position_cap": 1000.0,
        "raw_action_ridge": 0.001,
    }
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("SANOS objective-grid run identity changed")
    else:
        _write_json(manifest, identity)
    if phase == "fit" and (objective is None or risk_parameter is None):
        raise ValueError("fitting requires an objective and a risk parameter")
    config = (
        ScenarioOptimizationConfig(
            objective=objective,
            risk_parameter=risk_parameter,
            position_cap=1000.0,
            raw_action_ridge=0.001,
            certificate_tolerance=2.0e-6,
        )
        if phase == "fit"
        else None
    )
    risk_label = "inf" if risk_parameter is not None and math.isinf(risk_parameter) else str(risk_parameter)
    output_directory = root / "historical-sanos" / f"{objective.value}-lambda{risk_label}" if config else root / "scenarios"
    selected_dates = dates[:stop_after] if stop_after is not None else dates
    started = monotonic()
    for index, origin_date in enumerate(selected_dates, start=1):
        scenario_path = root / "scenarios" / f"{origin_date.isoformat()}.npz"
        destination = (
            output_directory / "records" / f"{origin_date.isoformat()}.json"
            if config
            else scenario_path
        )
        if not destination.exists():
            if config is None:
                _prepare_one(panel, origin_date, scenario_path)
            else:
                try:
                    results = _fit_one(
                        panel,
                        origin_date=origin_date,
                        scenario_path=scenario_path,
                        quantities=np.asarray(source["target_quantities"], dtype=np.float64),
                        config=config,
                    )
                    if any(not result["converged"] or result["cap_hits"] for result in results):
                        _write_json(
                            output_directory / "failures" / f"{origin_date.isoformat()}.json",
                            {"results": results, "reason": "uncertified or position cap hit"},
                        )
                    else:
                        _write_json(destination, {"results": results})
                except Exception as error:
                    _write_json(
                        output_directory / "failures" / f"{origin_date.isoformat()}.json",
                        {"error": repr(error), "traceback": traceback.format_exc()},
                    )
        elapsed = monotonic() - started
        eta = elapsed / index * (len(selected_dates) - index)
        print(
            f"{phase} {objective.value if objective else ''} {risk_label if config else ''} "
            f"{index}/{len(selected_dates)} {origin_date} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "fit", "all"), required=True)
    parser.add_argument("--objective", type=ScenarioObjective)
    parser.add_argument("--risk-parameter", type=float)
    parser.add_argument("--stop-after", type=int)
    arguments = parser.parse_args()
    if arguments.phase == "all":
        run_grid(
            root=arguments.root,
            source_run=arguments.source_run,
            phase="prepare",
            stop_after=arguments.stop_after,
        )
        for objective, risks in (
            (ScenarioObjective.MEAN_VARIANCE, (1.0, 10.0, 100.0, math.inf)),
            (ScenarioObjective.ENTROPY, (1.0, 10.0, 100.0)),
            (ScenarioObjective.POWER_EXPONENTIAL, (1.0, 10.0, 100.0)),
            (ScenarioObjective.CVAR_OCE, (1.0, 10.0, 100.0, math.inf)),
        ):
            for risk_parameter in risks:
                run_grid(
                    root=arguments.root,
                    source_run=arguments.source_run,
                    phase="fit",
                    objective=objective,
                    risk_parameter=risk_parameter,
                    stop_after=arguments.stop_after,
                )
        return
    run_grid(
        root=arguments.root,
        source_run=arguments.source_run,
        phase=arguments.phase,
        objective=arguments.objective,
        risk_parameter=arguments.risk_parameter,
        stop_after=arguments.stop_after,
    )


if __name__ == "__main__":
    main()
