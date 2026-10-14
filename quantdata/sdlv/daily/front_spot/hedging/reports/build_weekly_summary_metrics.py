"""Build authenticated weekly benchmark report inputs from production matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TypeAlias

import numpy as np

from quantdata.hedging.inference import (
    paired_episode_differences,
    summarize_episode_losses,
)
from quantdata.hedging.metrics import maximum_drawdown_from_losses
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import weekly_targets
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.policy_labels import (
    canonical_policy_variant,
    policy_specification_label,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_results import (
    WeeklyLossStatistics,
    WeeklyResultsConfig,
    WeeklyResultSlice,
    WeeklyRunResults,
    authenticate_completed_weekly_results,
)

JsonObject: TypeAlias = dict[str, object]

SUMMARY_SCHEMA = "hedging-benchmark-weekly-summary-metrics-v4"
WEEKLY_MATRIX_SCHEMA = "hedging-weekly-production-matrix-v1"
DISPLAY_SCALE = 10_000.0
EXPECTED_ORIGINS = 528
EXPECTED_TARGETS = len(weekly_targets())
EXPECTED_RESULT_COUNT = EXPECTED_ORIGINS * EXPECTED_TARGETS
OBJECTIVE_GRIDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mean_variance", ("0.1", "1", "10")),
    ("mean_volatility", ("0.1", "1", "2", "3")),
    ("entropy", ("0.1", "1", "10")),
    ("power_exponential", ("0.1", "1", "10")),
)

_MODEL_DISPLAY_NAMES = {
    "sanos-lv": "SANOS--LV",
    "historical-sanos": "Historical SANOS",
    "gromm": "GROMM",
    "volgan": "VolGAN",
    "ad-seq-vol": "AD--Seq--Vol",
    "ad-seq-vol-ft": "AD--Seq--Vol--FT",
    "diffusion-ad-seq-vol": "AD--Seq--Vol",
    "diffusion-ad-seq-vol-ft": "AD--Seq--Vol--FT",
}


@dataclass(frozen=True, kw_only=True)
class WeeklyPolicyInput:
    """One completed weekly policy selected for authenticated reporting."""

    key: str
    results_path: Path
    expected_model: str
    expected_schedule: str
    display_name: str
    fitted_objective: str
    fitted_parameter: float | None
    expected_measure: str
    variant: str


def weekly_policy_inputs_from_matrices(
    matrix_paths: tuple[Path, ...],
    *,
    require_ready_complete: bool = False,
) -> tuple[WeeklyPolicyInput, ...]:
    """Discover completed weekly policies from immutable production matrices."""
    if not matrix_paths:
        raise ValueError("at least one weekly production matrix is required")
    inputs: dict[str, WeeklyPolicyInput] = {}
    commands: dict[str, tuple[str, ...]] = {}
    for matrix_path in matrix_paths:
        matrix = _read_json_object(matrix_path)
        if matrix.get("schema") != WEEKLY_MATRIX_SCHEMA:
            raise ValueError(f"unexpected weekly production matrix: {matrix_path}")
        cases = matrix.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"weekly matrix cases must be a list: {matrix_path}")
        for raw_case in cases:
            case = _as_object(raw_case, "weekly production case")
            case_id = _require_string(case, "case_id")
            command_value = case.get("command")
            if not isinstance(command_value, list) or not all(
                isinstance(value, str) for value in command_value
            ):
                continue
            command = tuple(command_value)
            try:
                output = Path(command[command.index("--output") + 1])
                schedule = command[command.index("--schedule") + 1]
            except (ValueError, IndexError) as error:
                raise ValueError(
                    f"weekly matrix case {case_id} lacks output or schedule"
                ) from error
            results_path = output / "results.json"
            readiness = _require_string(case, "readiness")
            if not results_path.is_file():
                if require_ready_complete and readiness == "ready":
                    raise ValueError(
                        f"ready weekly production case is incomplete: {case_id}"
                    )
                continue
            model = _require_string(case, "model")
            measure = _require_string(case, "measure")
            variant = _require_string(case, "variant")
            objective_value = case.get("objective")
            fitted_objective = (
                "reference"
                if objective_value is None
                else _require_string(case, "objective")
            )
            parameter_value = case.get("risk_parameter")
            fitted_parameter = (
                None
                if parameter_value is None
                else _require_number(case, "risk_parameter")
            )
            policy = WeeklyPolicyInput(
                key=case_id,
                results_path=results_path,
                expected_model=model,
                expected_schedule=schedule,
                display_name=_MODEL_DISPLAY_NAMES.get(
                    model,
                    model.replace("-", " ").title(),
                ),
                fitted_objective=fitted_objective,
                fitted_parameter=fitted_parameter,
                expected_measure=measure,
                variant=variant,
            )
            if case_id in inputs:
                if inputs[case_id] != policy or commands[case_id] != command:
                    raise ValueError(
                        f"weekly production matrices disagree on case {case_id}"
                    )
                continue
            inputs[case_id] = policy
            commands[case_id] = command
    if not inputs:
        raise ValueError("weekly production matrices contain no completed cases")
    return tuple(inputs.values())


def build_weekly_policy_summary_metrics(
    *,
    policies: tuple[WeeklyPolicyInput, ...],
    dataset: Path | None = None,
) -> JsonObject:
    """Authenticate and aggregate an explicitly identified weekly policy set."""
    if not policies:
        raise ValueError("at least one completed weekly policy is required")
    keys = tuple(policy.key for policy in policies)
    if len(set(keys)) != len(keys) or any(not key.strip() for key in keys):
        raise ValueError("weekly policy keys must be unique and non-empty")
    panel = PublicHedgingPanel(
        default_public_candidate_path() if dataset is None else dataset,
        validate=True,
    )
    loaded = tuple(_read_policy(panel, policy) for policy in policies)
    manifests = {
        _require_string(policy, "dataset_manifest_sha256") for policy in loaded
    }
    if len(manifests) != 1:
        raise ValueError("completed weekly runs use different dataset manifests")
    target_notionals = {
        _require_string(policy, "target_notional_identity") for policy in loaded
    }
    if len(target_notionals) != 1:
        raise ValueError("completed weekly runs use different target notionals")

    models: JsonObject = {}
    for policy in loaded:
        key = _require_string(policy, "key")
        models[key] = {
            name: value
            for name, value in policy.items()
            if name not in {"key", "record_losses", "result"}
        }
    _validate_common_unhedged(models)
    return {
        "schema": SUMMARY_SCHEMA,
        "dataset_manifest_sha256": manifests.pop(),
        "target_notional_identity": target_notionals.pop(),
        "display_scale": DISPLAY_SCALE,
        "display_unit": "basis points of inception normalized spot",
        "models": models,
        "comparisons": _reference_comparisons(loaded),
        "objective_tables": _objective_tables(models),
    }


def write_weekly_summary_metrics(
    payload: JsonObject,
    *,
    json_output: Path,
    objective_tables_output: Path | None = None,
    target_appendix_output: Path | None = None,
) -> None:
    """Write authenticated weekly JSON and generated LaTeX tables."""
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if objective_tables_output is not None:
        objective_tables_output.parent.mkdir(parents=True, exist_ok=True)
        objective_tables_output.write_text(
            _objective_tables_tex(payload),
            encoding="utf-8",
        )
    if target_appendix_output is not None:
        target_appendix_output.parent.mkdir(parents=True, exist_ok=True)
        target_appendix_output.write_text(
            _target_appendix_tex(payload),
            encoding="utf-8",
        )


def _read_policy(
    panel: PublicHedgingPanel,
    policy: WeeklyPolicyInput,
) -> JsonObject:
    run_payload = _read_json_object(policy.results_path.parent / "run.json")
    spec = WeeklyEvaluationSpec.from_dict(run_payload)
    result = authenticate_completed_weekly_results(
        panel,
        results_path=policy.results_path,
    )
    if result.run_identity != spec.run_identity:
        raise ValueError(f"weekly run and results identities differ: {policy.results_path}")
    if spec.model != policy.expected_model or result.model != policy.expected_model:
        raise ValueError(f"unexpected weekly model: {policy.results_path}")
    if (
        spec.schedule.value != policy.expected_schedule
        or result.schedule != policy.expected_schedule
    ):
        raise ValueError(f"unexpected weekly schedule: {policy.results_path}")
    if result.measure != policy.expected_measure:
        raise ValueError(
            "weekly production matrix measure does not match authenticated "
            f"run: {policy.results_path}"
        )
    canonical_variant = canonical_policy_variant(
        case_id=policy.key,
        model=result.model,
        schedule=result.schedule,
    )
    if policy.variant != canonical_variant:
        raise ValueError(
            "weekly production matrix variant does not match the canonical "
            f"policy: {policy.results_path}"
        )
    if result.dataset_manifest_sha256 != spec.dataset_manifest_sha256:
        raise ValueError(f"weekly result manifest differs from run: {policy.results_path}")
    if (
        len(result.record_sha256_by_origin) != EXPECTED_ORIGINS
        or result.result_count != EXPECTED_RESULT_COUNT
    ):
        raise ValueError(
            f"weekly results are not the complete production population: "
            f"{policy.results_path}"
        )
    _validate_fitted_objective(
        spec=spec,
        fitted_objective=policy.fitted_objective,
        fitted_parameter=policy.fitted_parameter,
        results_path=policy.results_path,
    )
    record_losses = _read_record_losses(
        policy.results_path.parent / "records",
        expected_result=result,
    )
    metrics = _with_drawdown_metrics(
        metrics=_aggregate_result(result),
        record_losses=record_losses,
    )
    metrics = _with_price_metrics(
        metrics=metrics,
        record_losses=record_losses,
    )
    return {
        "key": policy.key,
        "model": result.model,
        "display_name": policy.display_name,
        "measure": result.measure,
        "variant": canonical_variant,
        "specification": policy_specification_label(
            measure=result.measure,
            variant=canonical_variant,
        ),
        "schedule": result.schedule,
        "run_identity": result.run_identity,
        "dataset_manifest_sha256": result.dataset_manifest_sha256,
        "target_notional_identity": spec.target_notional_identity,
        "result_count": result.result_count,
        "origin_count": len(result.record_sha256_by_origin),
        "fitted_objective": policy.fitted_objective,
        "fitted_parameter": policy.fitted_parameter,
        "fit_cap_hits": result.fit_cap_hits,
        "executed_cap_hits": result.executed_cap_hits,
        "nonconverged_fits": result.nonconverged_fits,
        "cap_hits": result.cap_hits,
        "aggregation_identity": result.aggregation_identity,
        "metrics": metrics,
        "record_losses": record_losses,
        "result": result,
    }


def _validate_fitted_objective(
    *,
    spec: WeeklyEvaluationSpec,
    fitted_objective: str,
    fitted_parameter: float | None,
    results_path: Path,
) -> None:
    if fitted_objective == "reference":
        if fitted_parameter is not None:
            raise ValueError(
                f"weekly reference cannot have a fitted parameter: {results_path}"
            )
        return
    snapshot = json.loads(spec.configuration_snapshot_json)
    pairs = _objective_parameter_pairs(snapshot)
    expected = (fitted_objective, fitted_parameter)
    if fitted_parameter is None or not any(
        objective == expected[0]
        and math.isclose(parameter, fitted_parameter, rel_tol=0.0, abs_tol=1e-14)
        for objective, parameter in pairs
    ):
        raise ValueError(
            "weekly production matrix objective does not match authenticated "
            f"run: {results_path}"
        )


def _objective_parameter_pairs(value: object) -> set[tuple[str, float]]:
    pairs: set[tuple[str, float]] = set()
    if isinstance(value, dict):
        objective = value.get("objective")
        parameter = value.get("risk_parameter")
        if (
            isinstance(objective, str)
            and not isinstance(parameter, bool)
            and isinstance(parameter, int | float)
        ):
            pairs.add((objective, float(parameter)))
        for child in value.values():
            pairs.update(_objective_parameter_pairs(child))
    elif isinstance(value, list):
        for child in value:
            pairs.update(_objective_parameter_pairs(child))
    return pairs


def _read_record_losses(
    records_directory: Path,
    *,
    expected_result: WeeklyRunResults,
) -> dict[str, tuple[JsonObject, ...]]:
    expected_origins = tuple(expected_result.record_sha256_by_origin)
    paths = tuple(sorted(Path(records_directory).glob("*.json")))
    if tuple(path.stem for path in paths) != expected_origins:
        raise ValueError("weekly record files do not match authenticated origins")
    rows: dict[str, list[JsonObject]] = {}
    seen: set[tuple[str, str]] = set()
    for path in paths:
        origin = path.stem
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != (
            expected_result.record_sha256_by_origin[origin]
        ):
            raise ValueError(f"weekly record changed after authentication: {path}")
        payload = json.loads(content)
        record = _as_object(payload, f"weekly record {path}")
        if (
            _require_string(record, "run_identity") != expected_result.run_identity
            or _require_string(record, "origin_date") != origin
        ):
            raise ValueError(f"weekly record identity mismatch: {path}")
        runs = record.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"weekly record runs must be a list: {path}")
        for raw_run in runs:
            run = _as_object(raw_run, "weekly record run")
            result = _require_object(run, "result")
            if _require_string(result, "origin_date") != origin:
                raise ValueError(f"weekly result origin mismatch: {path}")
            claim = _require_string(result, "target_label")
            breakdown = _require_object(result, "breakdown")
            key = (origin, claim)
            if key in seen:
                raise ValueError(f"duplicate weekly origin/claim record: {key}")
            seen.add(key)
            quantity = _require_number(breakdown, "claim_quantity")
            market_price = _require_number(breakdown, "market_price")
            model_price = _require_number(breakdown, "model_price")
            payoff = _require_number(breakdown, "payoff")
            price_records = run.get("fit_price_records")
            if not isinstance(price_records, list) or len(price_records) != 5:
                raise ValueError(
                    f"weekly run must contain five fit-price records: {path}"
                )
            inception_prices = _as_object(
                price_records[0],
                "weekly inception price record",
            )
            inception_market_price = _require_number(
                inception_prices,
                "common_market_price_per_fit_spot",
            )
            q_expectation = _optional_number(
                inception_prices,
                "q_expectation_per_fit_spot",
            )
            buyer_indifference = _optional_number(
                inception_prices,
                "buyer_indifference_per_fit_spot",
            )
            seller_indifference = _optional_number(
                inception_prices,
                "seller_indifference_per_fit_spot",
            )
            accounting_price = (
                buyer_indifference
                if buyer_indifference is not None
                else q_expectation
            )
            if accounting_price is None:
                raise ValueError(
                    f"weekly inception prices lack Q or buyer price: {path}"
                )
            if not math.isclose(
                inception_market_price,
                market_price,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"weekly inception market price disagrees with accounting: {path}"
                )
            if not math.isclose(
                accounting_price,
                model_price,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"weekly inception model price disagrees with accounting: {path}"
                )
            rows.setdefault(claim, []).append(
                {
                    "origin_date": origin,
                    "unhedged_loss": quantity * (market_price - payoff),
                    "economic_loss": -_require_number(breakdown, "economic_pnl"),
                    "pure_hedge_loss": -_require_number(
                        breakdown,
                        "pure_hedge_error",
                    ),
                    "pricing_loss": -_require_number(
                        breakdown,
                        "pricing_contribution",
                    ),
                    "market_price": market_price,
                    "accounting_model_price": model_price,
                    "q_expectation": q_expectation,
                    "buyer_indifference": buyer_indifference,
                    "seller_indifference": seller_indifference,
                }
            )
    if len(seen) != expected_result.result_count:
        raise ValueError("weekly record population does not match results.json")
    expected_claims = {target.label for target in weekly_targets()}
    if set(rows) != expected_claims:
        raise ValueError("weekly record population does not contain 13 targets")
    for claim, values in rows.items():
        values.sort(key=lambda value: str(value["origin_date"]))
        if len(values) != EXPECTED_ORIGINS:
            raise ValueError(
                f"weekly claim {claim!r} does not contain 528 observations"
            )
    return {claim: tuple(values) for claim, values in rows.items()}


def _aggregate_result(result: WeeklyRunResults) -> JsonObject:
    overall = tuple(
        value
        for value in result.slices
        if value.slice_label == "overall"
        and value.year is None
        and value.volatility_regime is None
    )
    expected = {target.label for target in weekly_targets()}
    if {value.claim for value in overall} != expected:
        raise ValueError("weekly result does not contain the complete target book")
    return {
        "all": _aggregate_slices(overall),
        "targets": {
            value.claim: _slice_payload(value)
            for value in sorted(overall, key=lambda item: item.claim)
        },
    }


def _with_drawdown_metrics(
    *,
    metrics: JsonObject,
    record_losses: dict[str, tuple[JsonObject, ...]],
) -> JsonObject:
    all_metrics = dict(_require_object(metrics, "all"))
    targets = {
        claim: dict(_as_object(value, f"weekly target {claim}"))
        for claim, value in _require_object(metrics, "targets").items()
    }
    for loss_name, prefix in (
        ("economic_loss", "economic"),
        ("pure_hedge_loss", "pure"),
    ):
        _, matrix = _loss_matrix(record_losses, loss_name=loss_name)
        target_drawdowns = np.asarray(
            [maximum_drawdown_from_losses(values) for values in matrix],
            dtype=np.float64,
        )
        all_metrics[f"{prefix}_book_maximum_drawdown"] = (
            maximum_drawdown_from_losses(np.mean(matrix, axis=0))
        )
        all_metrics[f"mean_target_{prefix}_maximum_drawdown"] = float(
            np.mean(target_drawdowns)
        )
        all_metrics[f"worst_target_{prefix}_maximum_drawdown"] = float(
            np.max(target_drawdowns)
        )
        for claim, drawdown in zip(
            sorted(record_losses),
            target_drawdowns,
            strict=True,
        ):
            targets[claim][f"{prefix}_maximum_drawdown"] = float(drawdown)
    return {"all": all_metrics, "targets": targets}


def _with_price_metrics(
    *,
    metrics: JsonObject,
    record_losses: dict[str, tuple[JsonObject, ...]],
) -> JsonObject:
    """Attach inception unit-price summaries without mixing claim quantities."""
    all_metrics = dict(_require_object(metrics, "all"))
    targets = {
        claim: dict(_as_object(value, f"weekly target {claim}"))
        for claim, value in _require_object(metrics, "targets").items()
    }
    all_rows = tuple(
        row
        for claim in sorted(record_losses)
        for row in record_losses[claim]
    )
    all_metrics["prices"] = _price_summary(all_rows)
    for claim, rows in record_losses.items():
        targets[claim]["prices"] = _price_summary(rows)
    return {"all": all_metrics, "targets": targets}


def _price_summary(rows: tuple[JsonObject, ...]) -> JsonObject:
    if not rows:
        raise ValueError("weekly price summary requires observations")
    market = np.asarray(
        [_require_number(row, "market_price") for row in rows],
        dtype=np.float64,
    )
    accounting = np.asarray(
        [_require_number(row, "accounting_model_price") for row in rows],
        dtype=np.float64,
    )
    return {
        "observations": len(rows),
        "mean_market_price": float(np.mean(market)),
        "mean_accounting_model_price": float(np.mean(accounting)),
        "mean_accounting_minus_market": float(np.mean(accounting - market)),
        **{
            f"mean_{name}": _mean_optional_record_number(rows, name)
            for name in (
                "q_expectation",
                "buyer_indifference",
                "seller_indifference",
            )
        },
    }


def _mean_optional_record_number(
    rows: tuple[JsonObject, ...],
    name: str,
) -> float | None:
    values = tuple(row.get(name) for row in rows)
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError(f"weekly price field {name} is missing for some origins")
    numbers = tuple(_number(value) for value in present)
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError(f"weekly price field {name} must be finite")
    return float(np.mean(numbers))


def _aggregate_slices(slices: tuple[WeeklyResultSlice, ...]) -> JsonObject:
    losses = {
        name: _aggregate_losses(tuple(getattr(value, name) for value in slices))
        for name in (
            "unhedged_loss",
            "economic_loss",
            "pure_hedge_loss",
            "pricing_loss",
        )
    }
    unhedged_rmse = _require_number(
        _require_object(losses, "unhedged_loss"),
        "root_mean_square",
    )
    pure_rmse = _require_number(
        _require_object(losses, "pure_hedge_loss"),
        "root_mean_square",
    )
    return {
        **losses,
        "pure_rmse_reduction": 1.0 - pure_rmse / unhedged_rmse,
        "mean_transaction_cost": _mean(
            value.mean_transaction_cost for value in slices
        ),
        "mean_spot_transaction_cost": _mean(
            value.mean_spot_transaction_cost for value in slices
        ),
        "mean_option_transaction_cost": _mean(
            value.mean_option_transaction_cost for value in slices
        ),
        "mean_total_turnover_notional": _mean(
            value.mean_total_turnover_notional for value in slices
        ),
        "mean_entry_turnover_notional": _mean(
            value.mean_entry_turnover_notional for value in slices
        ),
        "mean_gross_exposure": _mean(value.mean_gross_exposure for value in slices),
        "mean_net_exposure": _mean(value.mean_net_exposure for value in slices),
        "mean_absolute_net_exposure": _mean(
            value.mean_absolute_net_exposure for value in slices
        ),
        "mean_nonzero_actions_per_decision": _mean(
            value.mean_nonzero_actions_per_decision for value in slices
        ),
        "cap_hits": sum(value.cap_hits for value in slices),
        "numerical_cap_hit_result_rate": _mean(
            value.numerical_cap_hit_result_rate for value in slices
        ),
    }


def _aggregate_losses(losses: tuple[WeeklyLossStatistics, ...]) -> JsonObject:
    observations = sum(value.observations for value in losses)
    mean = sum(value.observations * value.mean for value in losses) / observations
    sample_variance_numerator = sum(
        (value.observations - 1) * value.sample_standard_deviation**2
        + value.observations * (value.mean - mean) ** 2
        for value in losses
    )
    return {
        "observations": observations,
        "mean": mean,
        "sample_standard_deviation": math.sqrt(
            sample_variance_numerator / (observations - 1)
        ),
        "root_mean_square": math.sqrt(
            sum(
                value.observations * value.root_mean_square**2 for value in losses
            )
            / observations
        ),
        "mean_absolute_loss": sum(
            value.observations * value.mean_absolute_loss for value in losses
        )
        / observations,
        "mean_target_expected_shortfall_95": _mean(
            value.expected_shortfall_95 for value in losses
        ),
        "mean_target_expected_shortfall_99": _mean(
            value.expected_shortfall_99 for value in losses
        ),
        "objectives": {
            objective: {
                parameter: _mean(
                    getattr(value, objective)[parameter] for value in losses
                )
                for parameter in parameters
            }
            for objective, parameters in OBJECTIVE_GRIDS
        },
    }


def _slice_payload(value: WeeklyResultSlice) -> JsonObject:
    payload = asdict(value)
    return {
        name: item
        for name, item in payload.items()
        if name not in {"slice_label", "year", "volatility_regime"}
    }


def _validate_common_unhedged(models: JsonObject) -> None:
    by_schedule: dict[str, list[tuple[float, str]]] = {}
    for key, raw_model in models.items():
        model = _as_object(raw_model, key)
        all_metrics = _require_object(_require_object(model, "metrics"), "all")
        unhedged = _require_object(all_metrics, "unhedged_loss")
        by_schedule.setdefault(_require_string(model, "schedule"), []).append(
            (_require_number(unhedged, "root_mean_square"), key)
        )
    for schedule, values in by_schedule.items():
        reference = values[0][0]
        if any(
            not math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-14)
            for value, _ in values[1:]
        ):
            raise ValueError(
                f"completed weekly {schedule} runs do not share unhedged claims"
            )


def _reference_comparisons(loaded: tuple[JsonObject, ...]) -> JsonObject:
    comparisons: JsonObject = {}
    schedules = sorted(
        {_require_string(value, "schedule") for value in loaded}
    )
    for schedule in schedules:
        candidates = tuple(
            value
            for value in loaded
            if _require_string(value, "schedule") == schedule
        )
        references = tuple(
            value
            for value in candidates
            if _require_string(value, "model") == "sanos-lv"
            and _require_string(value, "fitted_objective") == "reference"
        )
        if not references:
            continue
        if len(references) != 1:
            raise ValueError(f"weekly {schedule} has multiple reference policies")
        reference = references[0]
        reference_key = _require_string(reference, "key")
        reference = _require_number(
            _require_object(
                _require_object(
                    _require_object(references[0], "metrics"),
                    "all",
                ),
                "pure_hedge_loss",
            ),
            "root_mean_square",
        )
        comparisons[schedule] = {
            "reference_case": reference_key,
            "pure_rmse_reduction_vs_reference": {
                _require_string(policy, "key"): 1.0
                - _require_number(
                    _require_object(
                        _require_object(
                            _require_object(policy, "metrics"),
                            "all",
                        ),
                        "pure_hedge_loss",
                    ),
                    "root_mean_square",
                )
                / reference
                for policy in candidates
            },
            "paired_book_loss_differences": {
                _require_string(policy, "key"): _paired_book_comparison(
                    candidate=policy,
                    reference=references[0],
                )
                for policy in candidates
            },
        }
    return comparisons


def _paired_book_comparison(
    *,
    candidate: JsonObject,
    reference: JsonObject,
) -> JsonObject:
    summaries: JsonObject = {}
    inference_config = WeeklyResultsConfig().inference
    for loss_name in ("pure_hedge_loss", "economic_loss"):
        candidate_ids, candidate_losses = _book_loss_series(
            candidate,
            loss_name=loss_name,
        )
        reference_ids, reference_losses = _book_loss_series(
            reference,
            loss_name=loss_name,
        )
        differences = paired_episode_differences(
            candidate_ids,
            candidate_losses,
            reference_ids,
            reference_losses,
        )
        summary = summarize_episode_losses(
            differences,
            episode_dates=tuple(date.fromisoformat(value) for value in candidate_ids),
            config=inference_config,
        )
        inference = summary.inference
        summaries[loss_name] = {
            "convention": "candidate minus SANOS-LV reference loss",
            "observations": summary.observations,
            "mean": summary.mean,
            "median": summary.median,
            "sample_standard_deviation": summary.sample_standard_deviation,
            "root_mean_square": summary.root_mean_square,
            "mean_absolute_difference": summary.mean_absolute_loss,
            "hac_standard_error": inference.hac_standard_error,
            "raw_hac_long_run_variance": inference.raw_hac_long_run_variance,
            "hac_long_run_variance": inference.hac_long_run_variance,
            "hac_variance_was_floored": inference.hac_variance_was_floored,
            "hac_kernel": inference.hac_kernel.value,
            "hac_lags": inference.hac_lags,
            "hac_confidence_interval_95": list(inference.hac_95),
            "hac_confidence_interval_99": list(inference.hac_99),
            "bootstrap_confidence_interval_95": list(inference.bootstrap_95),
            "bootstrap_confidence_interval_99": list(inference.bootstrap_99),
            "bootstrap_block_length": min(
                inference_config.bootstrap_block_length,
                summary.observations,
            ),
            "bootstrap_samples": inference_config.bootstrap_samples,
            "bootstrap_seed": inference_config.bootstrap_seed,
        }
    return summaries


def _book_loss_series(
    policy: JsonObject,
    *,
    loss_name: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    record_losses = _require_record_losses(policy)
    origins, matrix = _loss_matrix(record_losses, loss_name=loss_name)
    return origins, np.mean(matrix, axis=0)


def _loss_matrix(
    record_losses: dict[str, tuple[JsonObject, ...]],
    *,
    loss_name: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    claims = tuple(sorted(record_losses))
    if len(claims) != EXPECTED_TARGETS:
        raise ValueError("weekly loss matrix requires 13 targets")
    origins = tuple(
        str(value["origin_date"]) for value in record_losses[claims[0]]
    )
    if any(
        tuple(str(value["origin_date"]) for value in record_losses[claim])
        != origins
        for claim in claims[1:]
    ):
        raise ValueError("weekly targets do not share one chronological origin grid")
    matrix = np.asarray(
        [
            [_require_number(value, loss_name) for value in record_losses[claim]]
            for claim in claims
        ],
        dtype=np.float64,
    )
    return origins, matrix


def _objective_tables(models: JsonObject) -> JsonObject:
    tables: JsonObject = {}
    for objective, parameters in OBJECTIVE_GRIDS:
        rows: list[JsonObject] = []
        for key, raw_model in models.items():
            model = _as_object(raw_model, key)
            all_metrics = _require_object(_require_object(model, "metrics"), "all")
            economic = _require_object(all_metrics, "economic_loss")
            objectives = _require_object(economic, "objectives")
            scores = _require_object(objectives, objective)
            rows.append(
                {
                    "model": key,
                    "display_name": _require_string(model, "display_name"),
                    "measure": _require_string(model, "measure"),
                    "variant": _require_string(model, "variant"),
                    "specification": _require_string(model, "specification"),
                    "schedule": _require_string(model, "schedule"),
                    "fitted_objective": _require_string(model, "fitted_objective"),
                    "fitted_parameter": model.get("fitted_parameter"),
                    "scores": {
                        parameter: _require_number(scores, parameter)
                        for parameter in parameters
                    },
                    "pure_rmse": _require_number(
                        _require_object(all_metrics, "pure_hedge_loss"),
                        "root_mean_square",
                    ),
                    "mean_transaction_cost": _require_number(
                        all_metrics,
                        "mean_transaction_cost",
                    ),
                    "economic_book_maximum_drawdown": _require_number(
                        all_metrics,
                        "economic_book_maximum_drawdown",
                    ),
                    "worst_target_economic_maximum_drawdown": _require_number(
                        all_metrics,
                        "worst_target_economic_maximum_drawdown",
                    ),
                }
            )
        tables[objective] = {"parameters": list(parameters), "rows": rows}
    return tables


def _objective_tables_tex(payload: JsonObject) -> str:
    tables = _require_object(payload, "objective_tables")
    lines = ["% Generated weekly objective tables; do not edit by hand."]
    for objective, parameters in OBJECTIVE_GRIDS:
        table = _require_object(tables, objective)
        rows = table.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"weekly objective rows must be a list: {objective}")
        lines.extend(
            (
                "\\begin{landscape}",
                f"\\subsection*{{Weekly {_tex_objective_name(objective)} scores}}",
                "\\begin{longtable}{@{}llllr"
                + "r" * len(parameters)
                + "rrrr@{}}",
                "\\toprule",
                "Schedule & Policy & Specification & Fit objective & Fit parameter & "
                + " & ".join(
                    _parameter_heading(objective, parameter)
                    for parameter in parameters
                )
                + " & Book MDD & Worst target MDD & Pure RMSE & Cost \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for raw_row in rows:
            row = _as_object(raw_row, f"weekly {objective} row")
            scores = _require_object(row, "scores")
            fitted_parameter = row.get("fitted_parameter")
            fit_parameter = (
                "--"
                if fitted_parameter is None
                else f"{_number(fitted_parameter):g}"
            )
            lines.append(
                " & ".join(
                    (
                        _tex_escape(_require_string(row, "schedule")),
                        _require_string(row, "display_name"),
                        _tex_escape(_require_string(row, "specification")),
                        _tex_escape(_require_string(row, "fitted_objective")),
                        fit_parameter,
                        *(
                            _basis_points(_require_number(scores, parameter))
                            for parameter in parameters
                        ),
                        _basis_points(
                            _require_number(row, "economic_book_maximum_drawdown")
                        ),
                        _basis_points(
                            _require_number(
                                row,
                                "worst_target_economic_maximum_drawdown",
                            )
                        ),
                        _basis_points(_require_number(row, "pure_rmse")),
                        _basis_points(_require_number(row, "mean_transaction_cost")),
                    )
                )
                + " \\\\"
            )
        lines.extend(("\\bottomrule", "\\end{longtable}", "\\end{landscape}", ""))
    return "\n".join(lines) + "\n"


def _target_appendix_tex(payload: JsonObject) -> str:
    models = _require_object(payload, "models")
    lines = ["% Generated weekly target appendix; do not edit by hand."]
    for key, raw_model in models.items():
        model = _as_object(raw_model, key)
        targets = _require_object(_require_object(model, "metrics"), "targets")
        model_heading = (
            _require_string(model, "display_name")
            + ": "
            + _tex_escape(_require_string(model, "schedule"))
            + "; "
            + _tex_escape(_require_string(model, "specification"))
            + "; "
            + _policy_tex(model)
        )
        lines.extend(
            (
                "\\begin{landscape}",
                "\\subsection*{" + model_heading + "}",
                "\\begin{longtable}{@{}lrrrrrr@{}}",
                "\\caption{Mean inception unit prices across weekly origins, in basis points of inception normalized spot.}\\\\",
                "\\toprule",
                "Target & Market & Accounting & $Q$ expectation & Buyer & Seller & Accounting--market \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for claim in sorted(targets):
            prices = _require_object(_require_object(targets, claim), "prices")
            lines.append(
                " & ".join(
                    (
                        _tex_escape(claim),
                        _basis_points(_require_number(prices, "mean_market_price")),
                        _basis_points(
                            _require_number(prices, "mean_accounting_model_price")
                        ),
                        _optional_basis_points(prices.get("mean_q_expectation")),
                        _optional_basis_points(
                            prices.get("mean_buyer_indifference")
                        ),
                        _optional_basis_points(
                            prices.get("mean_seller_indifference")
                        ),
                        _basis_points(
                            _require_number(
                                prices,
                                "mean_accounting_minus_market",
                            )
                        ),
                    )
                )
                + " \\\\"
            )
        lines.extend(
            (
                "\\bottomrule",
                "\\end{longtable}",
                "\\medskip",
                "\\begin{longtable}{@{}lrrrrrrr@{}}",
                "\\toprule",
                "Target & Unhedged RMSE & Pure RMSE & Economic RMSE & Economic MDD & ES$_{95}$ & Cost & Gross \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for claim in sorted(targets):
            target = _require_object(targets, claim)
            lines.append(
                " & ".join(
                    (
                        _tex_escape(claim),
                        _basis_points(
                            _require_number(
                                _require_object(target, "unhedged_loss"),
                                "root_mean_square",
                            )
                        ),
                        _basis_points(
                            _require_number(
                                _require_object(target, "pure_hedge_loss"),
                                "root_mean_square",
                            )
                        ),
                        _basis_points(
                            _require_number(
                                _require_object(target, "economic_loss"),
                                "root_mean_square",
                            )
                        ),
                        _basis_points(
                            _require_number(target, "economic_maximum_drawdown")
                        ),
                        _basis_points(
                            _require_number(
                                _require_object(target, "economic_loss"),
                                "expected_shortfall_95",
                            )
                        ),
                        _basis_points(_require_number(target, "mean_transaction_cost")),
                        f"{_require_number(target, 'mean_gross_exposure'):.4f}",
                    )
                )
                + " \\\\"
            )
        lines.extend(("\\bottomrule", "\\end{longtable}", "\\end{landscape}", ""))
        for objective, parameters in OBJECTIVE_GRIDS:
            lines.extend(
                (
                    "\\begin{landscape}",
                    "\\subsection*{"
                    + model_heading
                    + "; target "
                    + _objective_title(objective)
                    + " scores}",
                    "\\begin{longtable}{@{}l"
                    + "r" * len(parameters)
                    + "rrr@{}}",
                    "\\toprule",
                    "Target & "
                    + " & ".join(
                        _parameter_heading(objective, parameter)
                        for parameter in parameters
                    )
                    + " & Pure RMSE & Economic RMSE & Economic MDD \\\\",
                    "\\midrule",
                    "\\endhead",
                )
            )
            for claim in sorted(targets):
                target = _require_object(targets, claim)
                economic = _require_object(target, "economic_loss")
                scores = _require_object(economic, objective)
                lines.append(
                    " & ".join(
                        (
                            _tex_escape(claim),
                            *(
                                _basis_points(
                                    _require_number(scores, parameter)
                                )
                                for parameter in parameters
                            ),
                            _basis_points(
                                _require_number(
                                    _require_object(target, "pure_hedge_loss"),
                                    "root_mean_square",
                                )
                            ),
                            _basis_points(
                                _require_number(
                                    economic,
                                    "root_mean_square",
                                )
                            ),
                            _basis_points(
                                _require_number(
                                    target,
                                    "economic_maximum_drawdown",
                                )
                            ),
                        )
                    )
                    + " \\\\"
                )
            lines.extend(
                ("\\bottomrule", "\\end{longtable}", "\\end{landscape}", "")
            )
    return "\n".join(lines) + "\n"


def _mean(values: Iterable[float]) -> float:
    items = tuple(float(value) for value in values)
    if not items:
        raise ValueError("cannot average an empty weekly metric")
    return sum(items) / len(items)


def _basis_points(value: float) -> str:
    return f"{DISPLAY_SCALE * value:,.2f}"


def _optional_basis_points(value: object) -> str:
    if value is None:
        return "--"
    return _basis_points(_number(value))


def _tex_objective_name(value: str) -> str:
    return value.replace("_", "--")


def _objective_title(objective: str) -> str:
    return {
        "mean_variance": "Mean--variance",
        "mean_volatility": "Mean--volatility",
        "entropy": "Entropy OCE",
        "power_exponential": "Power-exponential OCE",
    }[objective]


def _parameter_heading(objective: str, parameter: str) -> str:
    symbol = "\\kappa" if objective == "mean_volatility" else "\\lambda"
    return f"${symbol}={parameter}$"


def _policy_tex(payload: JsonObject) -> str:
    objective = _require_string(payload, "fitted_objective")
    if objective == "reference":
        return "reference"
    parameter = _number(payload.get("fitted_parameter"))
    symbol = "\\kappa" if objective == "mean-volatility" else "\\lambda"
    title = {
        "mean-variance": "MV",
        "mean-volatility": "MVol",
        "entropy": "Ent",
        "power-exponential": "Power",
    }.get(objective, objective)
    return f"{title}, ${symbol}={parameter:g}$"


def _tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("weekly report value must be numeric")
    return float(value)


def _read_json_object(path: Path) -> JsonObject:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"weekly report JSON must be an object: {path}")
    return payload


def _as_object(value: object, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_object(payload: JsonObject, name: str) -> JsonObject:
    return _as_object(payload.get(name), name)


def _require_string(payload: JsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_number(payload: JsonObject, name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _optional_number(payload: JsonObject, name: str) -> float | None:
    value = payload.get(name)
    if value is None:
        return None
    return _require_number(payload, name)


def _require_record_losses(
    payload: JsonObject,
) -> dict[str, tuple[JsonObject, ...]]:
    value = payload.get("record_losses")
    if not isinstance(value, dict) or not all(
        isinstance(claim, str)
        and isinstance(rows, tuple)
        and all(isinstance(row, dict) for row in rows)
        for claim, rows in value.items()
    ):
        raise ValueError("record_losses must be a target-to-row mapping")
    return value


def _default_dataset_path() -> Path | None:
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        return None
    return default_public_candidate_path()


def main() -> None:
    """Parse completed weekly runs and write reproducible report inputs."""
    report_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_default_dataset_path())
    parser.add_argument(
        "--weekly-matrix",
        type=Path,
        action="append",
        required=True,
        help="discover every completed case in this weekly production matrix",
    )
    parser.add_argument(
        "--require-ready-complete",
        action="store_true",
        help="reject any ready weekly matrix case without completed results",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=report_directory / "hedging_benchmark_weekly_metrics.json",
    )
    parser.add_argument(
        "--objective-tables-output",
        type=Path,
        default=report_directory / "hedging_benchmark_weekly_objective_tables.tex",
    )
    parser.add_argument(
        "--target-appendix-output",
        type=Path,
        default=report_directory / "hedging_benchmark_weekly_target_appendix.tex",
    )
    arguments = parser.parse_args()
    if arguments.dataset is None:
        parser.error("--dataset is required without QUANTDATA_ROOT")
    policies = weekly_policy_inputs_from_matrices(
        tuple(arguments.weekly_matrix),
        require_ready_complete=arguments.require_ready_complete,
    )
    payload = build_weekly_policy_summary_metrics(
        policies=policies,
        dataset=arguments.dataset,
    )
    write_weekly_summary_metrics(
        payload,
        json_output=arguments.json_output,
        objective_tables_output=arguments.objective_tables_output,
        target_appendix_output=arguments.target_appendix_output,
    )
    print(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "json_output": str(arguments.json_output),
                "objective_tables_output": str(arguments.objective_tables_output),
                "target_appendix_output": str(arguments.target_appendix_output),
                "policy_count": len(_require_object(payload, "models")),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
