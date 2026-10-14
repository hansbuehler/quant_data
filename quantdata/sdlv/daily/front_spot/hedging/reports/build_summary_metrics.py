"""Build reproducible headline and target-level hedging benchmark metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TypeAlias

import numpy as np

from quantdata.hedging.inference import (
    EpisodeInferenceConfig,
    HacKernel,
    paired_episode_differences,
    summarize_episode_losses,
)
from quantdata.hedging.metrics import maximum_drawdown_from_losses
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.policy_labels import (
    canonical_policy_variant,
    policy_specification_label,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DailyEvaluationSpec,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    authenticate_completed_daily_results,
)

JsonObject: TypeAlias = dict[str, object]

SUMMARY_SCHEMA = "hedging-benchmark-summary-metrics-v5"
RESULTS_SCHEMA = "hedging-daily-results-v2"
EXPECTED_CLAIMS = 20
EXPECTED_OBSERVATIONS = 611
DISPLAY_SCALE = 10_000.0
DAILY_MATRIX_SCHEMA = "hedging-daily-production-matrix-v1"
OBJECTIVE_GRIDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mean_variance", ("0.1", "1", "10")),
    ("mean_volatility", ("0.1", "1", "2", "3")),
    ("entropy", ("0.1", "1", "10")),
    ("power_exponential", ("0.1", "1", "10")),
)
DAILY_COMPARISON_INFERENCE = EpisodeInferenceConfig(
    hac_lags=5,
    hac_kernel=HacKernel.BARTLETT,
    bootstrap_block_length=50,
    bootstrap_samples=2_000,
    bootstrap_seed=20_261_014,
)

_MODEL_DISPLAY_NAMES = {
    "sanos-lv": "SANOS--LV",
    "covariance": "Robust covariance",
    "robust-covariance": "Robust covariance",
    "historical-sanos": "Historical SANOS",
    "gromm": "GROMM",
    "gromm-q-dlv-vega": "GROMM Q + DLV vegas",
    "volgan": "VolGAN",
    "diffusion-ad-seq-vol": "AD--Seq--Vol",
    "diffusion-ad-seq-vol-ft": "AD--Seq--Vol--FT",
}


@dataclass(frozen=True, kw_only=True)
class DailyPolicyInput:
    """One completed daily policy selected for authenticated reporting."""

    key: str
    results_path: Path
    expected_model: str
    display_name: str
    fitted_objective: str
    fitted_parameter: float | None
    expected_measure: str | None = None
    variant: str = "unspecified"


def build_summary_metrics(
    *,
    sanos_results: Path,
    historical_results: Path,
    dataset: Path | None = None,
) -> JsonObject:
    """Read, authenticate, and aggregate the completed daily policies."""
    return build_daily_policy_summary_metrics(
        policies=(
            DailyPolicyInput(
                key="sanos-lv",
                results_path=Path(sanos_results),
                expected_model="sanos-lv",
                display_name="SANOS--LV",
                fitted_objective="reference",
                fitted_parameter=None,
                variant="reference-delta-eight-factor-vega",
            ),
            DailyPolicyInput(
                key="historical-sanos",
                results_path=Path(historical_results),
                expected_model="historical-sanos",
                display_name="Historical SANOS",
                fitted_objective="from-run",
                fitted_parameter=None,
                variant="rolling-fixed-contract-scenarios",
            ),
        ),
        dataset=dataset,
        legacy_comparison=True,
    )


def build_daily_policy_summary_metrics(
    *,
    policies: tuple[DailyPolicyInput, ...],
    dataset: Path | None = None,
    legacy_comparison: bool = False,
) -> JsonObject:
    """Authenticate and aggregate an explicitly identified daily policy set."""
    if not policies:
        raise ValueError("at least one completed daily policy is required")
    keys = tuple(policy.key for policy in policies)
    if len(set(keys)) != len(keys) or any(not key.strip() for key in keys):
        raise ValueError("daily policy keys must be unique and non-empty")
    panel = PublicHedgingPanel(
        default_public_candidate_path() if dataset is None else dataset,
        validate=True,
    )
    loaded_policies = tuple(
        _read_policy(
            panel,
            policy.results_path,
            key=policy.key,
            expected_model=policy.expected_model,
            display_name=policy.display_name,
            fitted_objective=(
                None
                if policy.fitted_objective == "from-run"
                else policy.fitted_objective
            ),
            fitted_parameter=policy.fitted_parameter,
            expected_measure=policy.expected_measure,
            variant=policy.variant,
        )
        for policy in policies
    )
    manifests = {
        _require_string(policy, "dataset_manifest_sha256")
        for policy in loaded_policies
    }
    if len(manifests) != 1:
        raise ValueError("completed summary runs use different dataset manifests")
    target_notionals = {
        _require_string(policy, "target_notional_identity")
        for policy in loaded_policies
    }
    if len(target_notionals) != 1:
        raise ValueError("completed summary runs use different target notionals")

    models: JsonObject = {}
    for policy in loaded_policies:
        key = _require_string(policy, "key")
        model = {
            name: value
            for name, value in policy.items()
            if name not in {"key", "results", "record_losses"}
        }
        model["metrics"] = _aggregate_run(
            _require_object(policy, "results"),
            record_losses=_require_record_losses(policy),
        )
        models[key] = model

    unhedged = [
        _require_number(
            _require_object(_require_object(model, "metrics"), "all"),
            "unhedged_rmse",
        )
        for model in models.values()
        if isinstance(model, dict)
    ]
    if any(
        not math.isclose(value, unhedged[0], rel_tol=0.0, abs_tol=1e-14)
        for value in unhedged[1:]
    ):
        raise ValueError("completed runs do not share the same unhedged claims")

    comparisons = _reference_comparisons(loaded_policies, models=models)
    if legacy_comparison:
        sanos_all = _require_object(
            _require_object(_require_object(models, "sanos-lv"), "metrics"),
            "all",
        )
        historical_all = _require_object(
            _require_object(
                _require_object(models, "historical-sanos"), "metrics"
            ),
            "all",
        )
        comparisons["sanos_pure_rmse_reduction_vs_historical"] = 1.0 - (
            _require_number(sanos_all, "pure_rmse")
            / _require_number(historical_all, "pure_rmse")
        )

    return {
        "schema": SUMMARY_SCHEMA,
        "dataset_manifest_sha256": manifests.pop(),
        "target_notional_identity": target_notionals.pop(),
        "display_scale": DISPLAY_SCALE,
        "display_units": "basis points of inception normalized spot",
        "objective_scaling": (
            "risk functionals are evaluated on unscaled inception-spot-normalized "
            "economic losses; only the final scalar is multiplied by 10,000"
        ),
        "models": models,
        "comparisons": comparisons,
        "objective_tables": _objective_tables(models),
    }


def daily_policy_inputs_from_matrices(
    matrix_paths: tuple[Path, ...],
    *,
    require_ready_complete: bool = False,
    failed_case_ids: frozenset[str] = frozenset(),
) -> tuple[DailyPolicyInput, ...]:
    """Discover completed daily policies from immutable production matrices."""
    if not matrix_paths:
        raise ValueError("at least one daily production matrix is required")
    inputs: dict[str, DailyPolicyInput] = {}
    commands: dict[str, tuple[str, ...]] = {}
    seen_case_ids: set[str] = set()
    for matrix_path in matrix_paths:
        matrix = _read_json_object(matrix_path)
        if matrix.get("schema") != DAILY_MATRIX_SCHEMA:
            raise ValueError(f"unexpected daily production matrix: {matrix_path}")
        cases = matrix.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"daily matrix cases must be a list: {matrix_path}")
        for raw_case in cases:
            case = _as_object(raw_case, "daily production case")
            case_id = _require_string(case, "case_id")
            seen_case_ids.add(case_id)
            command_value = case.get("command")
            if not isinstance(command_value, list) or not all(
                isinstance(value, str) for value in command_value
            ):
                continue
            command = tuple(command_value)
            try:
                output_index = command.index("--output") + 1
                output = Path(command[output_index])
            except (ValueError, IndexError) as error:
                raise ValueError(
                    f"daily matrix case {case_id} has no output directory"
                ) from error
            results_path = output / "results.json"
            readiness = _require_string(case, "readiness")
            if case_id in failed_case_ids:
                continue
            if not results_path.is_file():
                if require_ready_complete and readiness == "ready":
                    raise ValueError(
                        f"ready daily production case is incomplete: {case_id}"
                    )
                continue
            expected_model = _require_string(case, "model")
            expected_measure = _require_string(case, "measure")
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
            policy = DailyPolicyInput(
                key=case_id,
                results_path=results_path,
                expected_model=expected_model,
                display_name=_MODEL_DISPLAY_NAMES.get(
                    expected_model,
                    expected_model.replace("-", " ").title(),
                ),
                fitted_objective=fitted_objective,
                fitted_parameter=fitted_parameter,
                expected_measure=expected_measure,
                variant=variant,
            )
            if case_id in inputs:
                if inputs[case_id] != policy or commands[case_id] != command:
                    raise ValueError(
                        f"daily production matrices disagree on case {case_id}"
                    )
                continue
            inputs[case_id] = policy
            commands[case_id] = command
    unknown_failures = failed_case_ids - seen_case_ids
    if unknown_failures:
        raise ValueError(
            "daily production failures are absent from their matrices: "
            f"{sorted(unknown_failures)}"
        )
    if not inputs:
        raise ValueError("daily production matrices contain no completed cases")
    return tuple(inputs.values())


def write_summary_metrics(
    payload: JsonObject,
    *,
    json_output: Path,
    tex_output: Path,
    objective_tables_output: Path | None = None,
    objective_appendix_output: Path | None = None,
) -> None:
    """Write deterministic JSON evidence and matching TeX fragments."""
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    tex_output.write_text(_tex_fragment(payload), encoding="utf-8")
    if objective_tables_output is not None:
        objective_tables_output.parent.mkdir(parents=True, exist_ok=True)
        objective_tables_output.write_text(
            _objective_tables_tex(payload),
            encoding="utf-8",
        )
    if objective_appendix_output is not None:
        objective_appendix_output.parent.mkdir(parents=True, exist_ok=True)
        objective_appendix_output.write_text(
            _objective_appendix_tex(payload),
            encoding="utf-8",
        )


def _read_policy(
    panel: PublicHedgingPanel,
    path: Path,
    *,
    key: str,
    expected_model: str,
    display_name: str,
    fitted_objective: str | None = None,
    fitted_parameter: float | None = None,
    expected_measure: str | None = None,
    variant: str = "unspecified",
) -> JsonObject:
    results_path = Path(path)
    run = _read_json_object(results_path.parent / "run.json")
    spec = DailyEvaluationSpec.from_dict(run)
    results = _authenticate_policy_results(
        panel,
        results_path,
        expected_model=expected_model,
    )
    if spec.run_identity != _require_string(results, "run_identity"):
        raise ValueError(f"run and results identities do not match: {results_path}")
    if spec.model != expected_model:
        raise ValueError(f"expected {expected_model} run: {results_path}")
    snapshot = json.loads(spec.model_config_snapshot_json)
    fields = _require_object(snapshot, "fields")
    if fitted_objective is None:
        fitted_objective = _require_string(fields, "objective")
        fitted_parameter = _require_number(fields, "risk_parameter")
    elif fitted_objective == "reference":
        if fitted_parameter is not None:
            raise ValueError(
                f"reference policy cannot have a fitted parameter: {results_path}"
            )
    else:
        run_objective = _require_string(fields, "objective")
        run_parameter = _require_number(fields, "risk_parameter")
        if run_objective != fitted_objective or fitted_parameter is None or not math.isclose(
            run_parameter,
            fitted_parameter,
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError(
                "daily production matrix objective does not match authenticated "
                f"run: {results_path}"
            )
        fitted_objective = run_objective
        fitted_parameter = run_parameter
    measure = _require_string(results, "measure")
    if expected_measure is not None and measure != expected_measure:
        raise ValueError(
            "daily production matrix measure does not match authenticated "
            f"run: {results_path}"
        )
    canonical_variant = canonical_policy_variant(
        case_id=key,
        model=expected_model,
    )
    if variant != canonical_variant:
        raise ValueError(
            "daily production matrix variant does not match the canonical "
            f"policy: {results_path}"
        )
    return {
        "key": key,
        "model": expected_model,
        "display_name": display_name,
        "measure": measure,
        "variant": canonical_variant,
        "specification": policy_specification_label(
            measure=measure,
            variant=canonical_variant,
        ),
        "run_identity": _require_string(results, "run_identity"),
        "aggregation_identity": _require_string(results, "aggregation_identity"),
        "dataset_manifest_sha256": _require_string(results, "dataset_manifest_sha256"),
        "target_notional_identity": spec.target_notional_identity,
        "result_count": _require_integer(results, "result_count"),
        "fitted_objective": fitted_objective,
        "fitted_parameter": fitted_parameter,
        "results": results,
        "record_losses": _read_record_losses(
            results_path.parent / "records",
            expected_results=results,
        ),
    }


def _authenticate_policy_results(
    panel: PublicHedgingPanel,
    path: Path,
    *,
    expected_model: str,
) -> JsonObject:
    report = authenticate_completed_daily_results(
        panel,
        results_path=path,
    )
    payload = report.to_dict()
    if payload.get("schema") != RESULTS_SCHEMA:
        raise ValueError(f"unexpected daily-results schema: {path}")
    if payload.get("model") != expected_model:
        raise ValueError(f"expected {expected_model} results: {path}")
    if (
        _require_integer(payload, "result_count")
        != EXPECTED_CLAIMS * EXPECTED_OBSERVATIONS
    ):
        raise ValueError(
            f"daily results are not the complete production population: {path}"
        )
    return payload


def _read_results(path: Path, *, expected_model: str) -> JsonObject:
    payload = _read_json_object(path)
    if payload.get("schema") != RESULTS_SCHEMA:
        raise ValueError(f"unexpected daily-results schema: {path}")
    if payload.get("model") != expected_model:
        raise ValueError(f"expected {expected_model} results: {path}")
    if _require_integer(payload, "result_count") != EXPECTED_CLAIMS * EXPECTED_OBSERVATIONS:
        raise ValueError(f"daily results are not the complete production population: {path}")
    return payload


def _read_record_losses(
    records_directory: Path,
    *,
    expected_results: JsonObject,
) -> dict[str, tuple[dict[str, object], ...]]:
    model = _require_string(expected_results, "model")
    supplies_q_expectation = model in {
        "sanos-lv",
        "gromm",
        "gromm-q-dlv-vega",
    }
    rows: dict[str, list[dict[str, object]]] = {}
    seen: set[tuple[str, str]] = set()
    for path in sorted(Path(records_directory).glob("*.json")):
        payload = _read_json_object(path)
        origin = _require_string(payload, "origin_date")
        values = payload.get("results")
        if not isinstance(values, list):
            raise ValueError(f"daily record results must be a list: {path}")
        for raw in values:
            result = _as_object(raw, "daily record result")
            if _require_string(result, "origin_date") != origin:
                raise ValueError(f"daily record origin mismatch: {path}")
            claim = _require_string(result, "claim")
            key = (origin, claim)
            if key in seen:
                raise ValueError(f"duplicate daily origin/claim record: {key}")
            seen.add(key)
            diagnostics_value = result.get("diagnostics", {})
            diagnostics = _as_object(diagnostics_value, "daily diagnostics")
            market_price = _require_number(result, "market_price")
            model_price = _require_number(result, "model_price")
            rows.setdefault(claim, []).append(
                {
                    "origin_date": origin,
                    "unhedged_loss": -_require_number(result, "unhedged_pnl"),
                    "economic_loss": -_require_number(result, "hedged_pnl"),
                    "pure_hedge_loss": -_require_number(result, "pure_hedge_error"),
                    "pricing_loss": -_require_number(result, "pricing_difference"),
                    "market_price": market_price,
                    "accounting_model_price": model_price,
                    "q_expectation": model_price if supplies_q_expectation else None,
                    "buyer_indifference": _optional_number(
                        diagnostics,
                        "buyer_indifference_price",
                    ),
                    "seller_indifference": _optional_number(
                        diagnostics,
                        "seller_indifference_price",
                    ),
                }
            )
    if len(seen) != _require_integer(expected_results, "result_count"):
        raise ValueError("daily record population does not match results.json")
    if len(rows) != EXPECTED_CLAIMS:
        raise ValueError("daily record population must contain 20 claims")
    for claim, values in rows.items():
        values.sort(key=lambda value: str(value["origin_date"]))
        if len(values) != EXPECTED_OBSERVATIONS:
            raise ValueError(f"daily claim {claim!r} does not contain 611 observations")
    return {claim: tuple(values) for claim, values in rows.items()}


def _aggregate_run(
    payload: JsonObject,
    *,
    record_losses: dict[str, tuple[dict[str, object], ...]],
) -> JsonObject:
    overall = _overall_slices(payload)
    groups = {
        "all": overall,
        "one_day": [value for value in overall if "-1d-" in value["claim"]],
        "three_day": [value for value in overall if "-3d-" in value["claim"]],
    }
    if len(groups["one_day"]) != 10 or len(groups["three_day"]) != 10:
        raise ValueError("daily results must contain ten 1D and ten 3D claims")
    aggregates: JsonObject = {
        label: _aggregate_rows(rows, record_losses=record_losses)
        for label, rows in groups.items()
    }
    aggregates["targets"] = {
        _require_string(row, "claim"): _target_metrics(
            row,
            record_losses=record_losses[_require_string(row, "claim")],
        )
        for row in overall
    }
    aggregates["strata"] = _stratified_slice_metrics(payload)
    return aggregates


def _overall_slices(payload: JsonObject) -> list[JsonObject]:
    slices_value = payload.get("slices")
    if not isinstance(slices_value, list):
        raise ValueError("daily results slices must be a list")
    overall = [
        _as_object(value, "daily result slice")
        for value in slices_value
        if isinstance(value, dict)
        and value.get("slice_label") == "overall"
        and value.get("year") is None
        and value.get("volatility_regime") is None
    ]
    if len(overall) != EXPECTED_CLAIMS:
        raise ValueError("daily results must contain 20 overall claim slices")
    labels = {_require_string(value, "claim") for value in overall}
    if len(labels) != EXPECTED_CLAIMS:
        raise ValueError("daily overall claim labels must be unique")
    for value in overall:
        pure = _require_object(value, "pure_hedge_loss")
        if _require_integer(pure, "observations") != EXPECTED_OBSERVATIONS:
            raise ValueError("every overall claim slice must contain 611 observations")
    return overall


def _aggregate_rows(
    rows: list[JsonObject],
    *,
    record_losses: dict[str, tuple[dict[str, object], ...]],
) -> JsonObject:
    claims = [_require_string(value, "claim") for value in rows]
    origins = [
        tuple(str(value["origin_date"]) for value in record_losses[claim])
        for claim in claims
    ]
    if any(values != origins[0] for values in origins[1:]):
        raise ValueError("daily claims do not share one chronological origin grid")
    economic_matrix = np.asarray(
        [
            [_record_number(value, "economic_loss") for value in record_losses[claim]]
            for claim in claims
        ],
        dtype=np.float64,
    )
    pure_matrix = np.asarray(
        [
            [_record_number(value, "pure_hedge_loss") for value in record_losses[claim]]
            for claim in claims
        ],
        dtype=np.float64,
    )
    target_economic_drawdowns = np.asarray(
        [maximum_drawdown_from_losses(values) for values in economic_matrix],
        dtype=np.float64,
    )
    target_pure_drawdowns = np.asarray(
        [maximum_drawdown_from_losses(values) for values in pure_matrix],
        dtype=np.float64,
    )
    unhedged_rmse = _pooled_rmse(rows, "unhedged_loss")
    pure_rmse = _pooled_rmse(rows, "pure_hedge_loss")
    economic_rmse = _pooled_rmse(rows, "economic_loss")
    return {
        "claims": len(rows),
        "observations_per_claim": EXPECTED_OBSERVATIONS,
        "unhedged_rmse": unhedged_rmse,
        "pure_rmse": pure_rmse,
        "pure_rmse_reduction": 1.0 - pure_rmse / unhedged_rmse,
        "economic_rmse": economic_rmse,
        "expected_shortfall_95": _mean_loss_statistic(
            rows, "pure_hedge_loss", "expected_shortfall_95"
        ),
        "expected_shortfall_99": _mean_loss_statistic(
            rows, "pure_hedge_loss", "expected_shortfall_99"
        ),
        "mean_absolute_loss": _mean_loss_statistic(
            rows, "pure_hedge_loss", "mean_absolute_loss"
        ),
        "economic_book_maximum_drawdown": maximum_drawdown_from_losses(
            np.mean(economic_matrix, axis=0)
        ),
        "pure_book_maximum_drawdown": maximum_drawdown_from_losses(
            np.mean(pure_matrix, axis=0)
        ),
        "mean_target_economic_maximum_drawdown": float(
            np.mean(target_economic_drawdowns)
        ),
        "worst_target_economic_maximum_drawdown": float(
            np.max(target_economic_drawdowns)
        ),
        "mean_target_pure_maximum_drawdown": float(np.mean(target_pure_drawdowns)),
        "worst_target_pure_maximum_drawdown": float(np.max(target_pure_drawdowns)),
        "economic_objectives": _aggregate_objectives(rows, "economic_loss"),
        "pure_objectives": _aggregate_objectives(rows, "pure_hedge_loss"),
        "mean_transaction_cost": _mean_row_number(rows, "mean_transaction_cost"),
        "mean_action_l1": _mean_row_number(rows, "mean_action_l1"),
        "mean_entry_turnover_notional": _mean_row_number(
            rows, "mean_entry_turnover_notional"
        ),
        "mean_gross_exposure": _mean_row_number(rows, "mean_gross_exposure"),
        "mean_net_exposure": _mean_row_number(rows, "mean_net_exposure"),
        "mean_absolute_net_exposure": _mean_row_number(
            rows,
            "mean_absolute_net_exposure",
        ),
        "mean_nonzero_actions": _mean_row_number(rows, "mean_nonzero_actions"),
        "cap_hits": sum(_require_integer(value, "cap_hits") for value in rows),
        "numerical_cap_hit_result_rate": _mean_row_number(
            rows,
            "numerical_cap_hit_result_rate",
        ),
        "nonconverged_results": sum(
            _require_integer(value, "nonconverged_results") for value in rows
        ),
        "nonconvergence_rate": _mean_row_number(rows, "nonconvergence_rate"),
        "prices": _price_summary(
            tuple(
                record
                for claim in claims
                for record in record_losses[claim]
            )
        ),
    }


def _target_metrics(
    row: JsonObject,
    *,
    record_losses: tuple[dict[str, object], ...],
) -> JsonObject:
    economic = np.asarray(
        [_record_number(value, "economic_loss") for value in record_losses],
        dtype=np.float64,
    )
    pure = np.asarray(
        [_record_number(value, "pure_hedge_loss") for value in record_losses],
        dtype=np.float64,
    )
    payload = {
        name: value
        for name, value in row.items()
        if name not in {"claim", "slice_label", "year", "volatility_regime"}
    }
    payload.update({
        "economic_objectives": _loss_objectives(_require_object(row, "economic_loss")),
        "pure_objectives": _loss_objectives(_require_object(row, "pure_hedge_loss")),
        "economic_maximum_drawdown": maximum_drawdown_from_losses(economic),
        "pure_maximum_drawdown": maximum_drawdown_from_losses(pure),
        "pure_rmse": _require_number(
            _require_object(row, "pure_hedge_loss"), "root_mean_square"
        ),
        "economic_rmse": _require_number(
            _require_object(row, "economic_loss"), "root_mean_square"
        ),
        "expected_shortfall_95": _require_number(
            _require_object(row, "pure_hedge_loss"), "expected_shortfall_95"
        ),
        "prices": _price_summary(record_losses),
    })
    return payload


def _stratified_slice_metrics(payload: JsonObject) -> JsonObject:
    slices_value = payload.get("slices")
    if not isinstance(slices_value, list):
        raise ValueError("daily results slices must be a list")
    groups: dict[str, JsonObject] = {}
    for raw_slice in slices_value:
        row = _as_object(raw_slice, "daily result slice")
        label = _require_string(row, "slice_label")
        if label == "overall":
            continue
        claim = _require_string(row, "claim")
        group = groups.setdefault(
            label,
            {
                "year": row.get("year"),
                "volatility_regime": row.get("volatility_regime"),
                "targets": {},
            },
        )
        if (
            group["year"] != row.get("year")
            or group["volatility_regime"] != row.get("volatility_regime")
        ):
            raise ValueError(f"daily stratum metadata is inconsistent: {label}")
        targets = _require_object(group, "targets")
        if claim in targets:
            raise ValueError(f"duplicate daily stratum target: {label}/{claim}")
        targets[claim] = {
            name: value
            for name, value in row.items()
            if name not in {"claim", "slice_label", "year", "volatility_regime"}
        }
    return {label: groups[label] for label in sorted(groups)}


def _price_summary(rows: tuple[dict[str, object], ...]) -> JsonObject:
    if not rows:
        raise ValueError("daily price summary requires observations")
    market = np.asarray(
        [_record_number(row, "market_price") for row in rows],
        dtype=np.float64,
    )
    accounting = np.asarray(
        [_record_number(row, "accounting_model_price") for row in rows],
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
    rows: tuple[dict[str, object], ...],
    name: str,
) -> float | None:
    values = tuple(row.get(name) for row in rows)
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError(f"daily price field {name} is missing for some origins")
    numbers = tuple(_number(value) for value in present)
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError(f"daily price field {name} must be finite")
    return float(np.mean(numbers))


def _reference_comparisons(
    loaded: tuple[JsonObject, ...],
    *,
    models: JsonObject,
) -> JsonObject:
    references = tuple(
        policy
        for policy in loaded
        if _require_string(policy, "model") == "sanos-lv"
        and _require_string(policy, "fitted_objective") == "reference"
    )
    if not references:
        return {}
    if len(references) != 1:
        raise ValueError("daily results contain multiple SANOS-LV references")
    reference_policy = references[0]
    reference_key = _require_string(reference_policy, "key")
    reference_all = _require_object(
        _require_object(_require_object(models, reference_key), "metrics"),
        "all",
    )
    reference_rmse = _require_number(reference_all, "pure_rmse")
    return {
        "reference_case": reference_key,
        "pure_rmse_reduction_vs_reference": {
            key: 1.0
            - _require_number(
                _require_object(
                    _require_object(_require_object(models, key), "metrics"),
                    "all",
                ),
                "pure_rmse",
            )
            / reference_rmse
            for key in models
        },
        "paired_book_loss_differences": {
            _require_string(policy, "key"): _paired_book_comparison(
                candidate=policy,
                reference=reference_policy,
            )
            for policy in loaded
        },
    }


def _paired_book_comparison(
    *,
    candidate: JsonObject,
    reference: JsonObject,
) -> JsonObject:
    summaries: JsonObject = {}
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
            config=DAILY_COMPARISON_INFERENCE,
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
            "quantile_95": summary.quantile_95,
            "quantile_99": summary.quantile_99,
            "expected_shortfall_95": summary.expected_shortfall_95,
            "expected_shortfall_99": summary.expected_shortfall_99,
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
                DAILY_COMPARISON_INFERENCE.bootstrap_block_length,
                summary.observations,
            ),
            "bootstrap_samples": DAILY_COMPARISON_INFERENCE.bootstrap_samples,
            "bootstrap_seed": DAILY_COMPARISON_INFERENCE.bootstrap_seed,
        }
    return summaries


def _book_loss_series(
    policy: JsonObject,
    *,
    loss_name: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    record_losses = _require_record_losses(policy)
    claims = tuple(sorted(record_losses))
    if len(claims) != EXPECTED_CLAIMS:
        raise ValueError("daily paired comparison requires the complete target book")
    origins = tuple(
        str(row["origin_date"])
        for row in record_losses[claims[0]]
    )
    matrix = np.asarray(
        [
            [_record_number(row, loss_name) for row in record_losses[claim]]
            for claim in claims
        ],
        dtype=np.float64,
    )
    for claim in claims[1:]:
        claim_origins = tuple(
            str(row["origin_date"])
            for row in record_losses[claim]
        )
        if claim_origins != origins:
            raise ValueError("daily paired comparison origin grids differ")
    return origins, np.mean(matrix, axis=0)


def _aggregate_objectives(rows: list[JsonObject], loss_name: str) -> JsonObject:
    return {
        objective: {
            parameter: float(
                np.mean(
                    [
                        _require_number(
                            _require_object(
                                _require_object(row, loss_name), objective
                            ),
                            parameter,
                        )
                        for row in rows
                    ]
                )
            )
            for parameter in parameters
        }
        for objective, parameters in OBJECTIVE_GRIDS
    }


def _loss_objectives(loss: JsonObject) -> JsonObject:
    return {
        objective: {
            parameter: _require_number(_require_object(loss, objective), parameter)
            for parameter in parameters
        }
        for objective, parameters in OBJECTIVE_GRIDS
    }


def _objective_tables(models: JsonObject) -> JsonObject:
    tables: JsonObject = {}
    for objective, parameters in OBJECTIVE_GRIDS:
        rows: list[JsonObject] = []
        for key, raw_model in models.items():
            model = _as_object(raw_model, key)
            metrics = _require_object(model, "metrics")
            all_metrics = _require_object(metrics, "all")
            economic = _require_object(all_metrics, "economic_objectives")
            scores = _require_object(economic, objective)
            rows.append(
                {
                    "model": key,
                    "display_name": _require_string(model, "display_name"),
                    "measure": _require_string(model, "measure"),
                    "variant": _require_string(model, "variant"),
                    "specification": _require_string(model, "specification"),
                    "fitted_objective": _require_string(model, "fitted_objective"),
                    "fitted_parameter": model.get("fitted_parameter"),
                    "scores": {
                        parameter: _require_number(scores, parameter)
                        for parameter in parameters
                    },
                    "economic_book_maximum_drawdown": _require_number(
                        all_metrics, "economic_book_maximum_drawdown"
                    ),
                    "worst_target_economic_maximum_drawdown": _require_number(
                        all_metrics, "worst_target_economic_maximum_drawdown"
                    ),
                }
            )
        tables[objective] = {"parameters": list(parameters), "rows": rows}
    return tables


def _pooled_rmse(rows: list[JsonObject], loss_name: str) -> float:
    squared = [
        _require_number(_require_object(value, loss_name), "root_mean_square") ** 2
        for value in rows
    ]
    return math.sqrt(sum(squared) / len(squared))


def _mean_loss_statistic(
    rows: list[JsonObject], loss_name: str, statistic: str
) -> float:
    values = [
        _require_number(_require_object(value, loss_name), statistic) for value in rows
    ]
    return sum(values) / len(values)


def _mean_row_number(rows: list[JsonObject], name: str) -> float:
    values = [_require_number(value, name) for value in rows]
    return sum(values) / len(values)


def _tex_fragment(payload: JsonObject) -> str:
    models = _require_object(payload, "models")
    sanos_key = _first_model_key(models, "sanos-lv")
    historical_key = _first_model_key(models, "historical-sanos")
    sanos = _require_object(_require_object(models, sanos_key), "metrics")
    historical = _require_object(
        _require_object(models, historical_key), "metrics"
    )
    comparison = _require_object(payload, "comparisons")
    comparison_value = comparison.get(
        "sanos_pure_rmse_reduction_vs_historical"
    )
    if isinstance(comparison_value, bool) or not isinstance(
        comparison_value, int | float
    ):
        comparison_value = 1.0 - (
            _metric(sanos, "all", "pure_rmse")
            / _metric(historical, "all", "pure_rmse")
        )
    lines = [
        "% Generated by build_summary_metrics.py; do not edit by hand.",
        _macro(
            "SanosVsUnhedgedReduction",
            _percent(_metric(sanos, "all", "pure_rmse_reduction")),
        ),
        _macro(
            "HistoricalVsUnhedgedReduction",
            _percent(_metric(historical, "all", "pure_rmse_reduction")),
        ),
        _macro(
            "SanosVsHistoricalReduction",
            _percent(float(comparison_value)),
        ),
    ]
    for prefix, metrics in (("Sanos", sanos), ("Historical", historical)):
        for group_name, group_prefix in (
            ("all", "All"),
            ("one_day", "OneDay"),
            ("three_day", "ThreeDay"),
        ):
            lines.extend(
                (
                    _macro(
                        f"{prefix}{group_prefix}UnhedgedRmse",
                        _basis_points(_metric(metrics, group_name, "unhedged_rmse")),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}PureRmse",
                        _basis_points(_metric(metrics, group_name, "pure_rmse")),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}Reduction",
                        _percent(_metric(metrics, group_name, "pure_rmse_reduction")),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}EconomicRmse",
                        _basis_points(_metric(metrics, group_name, "economic_rmse")),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}EsNinetyFive",
                        _basis_points(
                            _metric(metrics, group_name, "expected_shortfall_95")
                        ),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}BookDrawdown",
                        _basis_points(
                            _metric(
                                metrics,
                                group_name,
                                "economic_book_maximum_drawdown",
                            )
                        ),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}WorstDrawdown",
                        _basis_points(
                            _metric(
                                metrics,
                                group_name,
                                "worst_target_economic_maximum_drawdown",
                            )
                        ),
                    ),
                )
            )
        lines.extend(
            (
                _macro(
                    f"{prefix}AllMeanCost",
                    _basis_points(_metric(metrics, "all", "mean_transaction_cost")),
                ),
                _macro(
                    f"{prefix}AllGrossExposure",
                    _percent(_metric(metrics, "all", "mean_gross_exposure")),
                ),
                _macro(
                    f"{prefix}AllCapHits",
                    str(int(_metric(metrics, "all", "cap_hits"))),
                ),
                _macro(
                    f"{prefix}AllNonconverged",
                    str(int(_metric(metrics, "all", "nonconverged_results"))),
                ),
            )
        )
    sanos_all = _require_object(sanos, "all")
    sanos_model = _require_object(models, sanos_key)
    historical_model = _require_object(models, historical_key)
    lines.extend(
        (
            _macro(
                "SanosAllPureRmseSix",
                _basis_points(_require_number(sanos_all, "pure_rmse")),
            ),
            _macro(
                "SanosAllEconomicRmseSix",
                _basis_points(_require_number(sanos_all, "economic_rmse")),
            ),
            _macro(
                "SanosRecordCount", f"{_require_integer(sanos_model, 'result_count'):,}"
            ),
            _macro(
                "HistoricalRecordCount",
                f"{_require_integer(historical_model, 'result_count'):,}",
            ),
            _macro(
                "DatasetManifestPrefix",
                _require_string(payload, "dataset_manifest_sha256")[:12],
            ),
            _macro(
                "SanosRunPrefix", _require_string(sanos_model, "run_identity")[:12]
            ),
            _macro(
                "HistoricalRunPrefix",
                _require_string(historical_model, "run_identity")[:12],
            ),
        )
    )
    return "\n".join(lines) + "\n"


def _objective_tables_tex(payload: JsonObject) -> str:
    tables = _require_object(payload, "objective_tables")
    lines = ["% Generated by build_summary_metrics.py; do not edit by hand."]
    comparisons = _require_object(payload, "comparisons")
    paired_value = comparisons.get("paired_book_loss_differences")
    if isinstance(paired_value, dict):
        models = _require_object(payload, "models")
        lines.extend(
            (
                "\\begin{landscape}",
                "\\subsection*{Paired daily target-book comparisons}",
                "\\scriptsize",
                "\\begin{longtable}{@{}p{25mm}p{65mm}lrrr@{}}",
                "\\caption{Paired daily equal-weighted target-book loss differences relative to SANOS--LV, in basis points. HAC uses a five-lag Bartlett kernel; MBB uses chronological moving blocks.}\\label{tab:daily-paired-book}\\\\",
                "\\toprule",
                "Model & Specification & Loss & Mean & HAC 95\\% CI & MBB 95\\% CI \\\\",
                "\\midrule",
                "\\endfirsthead",
                "\\toprule",
                "Model & Specification & Loss & Mean & HAC 95\\% CI & MBB 95\\% CI \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for model_key, raw_model in models.items():
            model = _as_object(raw_model, model_key)
            comparison = _require_object(paired_value, model_key)
            for loss_name, loss_label in (
                ("pure_hedge_loss", "Pure"),
                ("economic_loss", "Economic"),
            ):
                statistics = _require_object(comparison, loss_name)
                lines.append(
                    " & ".join(
                        (
                            _require_string(model, "display_name"),
                            _tex_escape(_require_string(model, "specification")),
                            loss_label,
                            _basis_points(_require_number(statistics, "mean")),
                            _basis_point_interval(
                                statistics,
                                "hac_confidence_interval_95",
                            ),
                            _basis_point_interval(
                                statistics,
                                "bootstrap_confidence_interval_95",
                            ),
                        )
                    )
                    + " \\\\"
                )
        lines.extend(
            ("\\bottomrule", "\\end{longtable}", "\\end{landscape}", "")
        )
    for index, (objective, _) in enumerate(OBJECTIVE_GRIDS, start=1):
        table = _require_object(tables, objective)
        parameters = _require_string_list(table, "parameters")
        rows = _require_object_list(table, "rows")
        lines.extend(
            (
                "\\begin{landscape}",
                f"\\subsection*{{Daily {_objective_title(objective)} scores}}",
                "\\scriptsize",
                f"\\begin{{longtable}}{{@{{}}p{{25mm}}p{{65mm}}p{{20mm}}{'r' * len(parameters)}rr@{{}}}}",
                f"\\caption{{{_objective_title(objective)}: full out-of-sample economic-risk score.}}\\label{{tab:objective-{index}}}\\\\",
                "\\toprule",
                "Model & Specification & Fitted policy & "
                + " & ".join(
                    _parameter_heading(objective, value) for value in parameters
                )
                + " & Book MDD & Worst target MDD \\\\",
                "\\midrule",
                "\\endfirsthead",
                "\\toprule",
                "Model & Specification & Fitted policy & "
                + " & ".join(
                    _parameter_heading(objective, value) for value in parameters
                )
                + " & Book MDD & Worst target MDD \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for row in rows:
            scores = _require_object(row, "scores")
            lines.append(
                " & ".join(
                    (
                        _require_string(row, "display_name"),
                        _tex_escape(_require_string(row, "specification")),
                        _policy_tex(row),
                        *(
                            _basis_points(_require_number(scores, parameter))
                            for parameter in parameters
                        ),
                        _basis_points(
                            _require_number(row, "economic_book_maximum_drawdown")
                        ),
                        _basis_points(
                            _require_number(
                                row, "worst_target_economic_maximum_drawdown"
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


def _objective_appendix_tex(payload: JsonObject) -> str:
    models = _require_object(payload, "models")
    lines = ["% Generated by build_summary_metrics.py; do not edit by hand."]
    for model_key, raw_model in models.items():
        model = _as_object(raw_model, model_key)
        targets = _require_object(_require_object(model, "metrics"), "targets")
        heading = (
            _require_string(model, "display_name")
            + ": "
            + _tex_escape(_require_string(model, "specification"))
            + "; "
            + _policy_tex(model)
        )
        lines.extend(
            (
                "\\begin{landscape}",
                "\\subsection*{" + heading + "}",
                "\\scriptsize",
                "\\begin{longtable}{@{}lrrrrrr@{}}",
                "\\caption{Mean inception unit prices across daily origins, in basis points of inception normalized spot.}\\\\",
                "\\toprule",
                "Target & Market & Accounting & $Q$ expectation & Buyer & Seller & Accounting--market \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for claim in sorted(targets, key=_target_sort_key):
            prices = _require_object(_require_object(targets, claim), "prices")
            lines.append(
                " & ".join(
                    (
                        _target_tex(claim),
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
                            _require_number(prices, "mean_accounting_minus_market")
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
                "\\tiny",
                "\\begin{longtable}{@{}lrrrrrrrrrr@{}}",
                "\\caption{Complete target-level economic-loss distribution.}\\\\",
                "\\toprule",
                "Target & Mean & 95\\% CI & Median & SD & RMSE & MAE & Q$_{95}$ & Q$_{99}$ & ES$_{95}$ & ES$_{99}$ \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for claim in sorted(targets, key=_target_sort_key):
            loss = _require_object(_require_object(targets, claim), "economic_loss")
            lines.append(
                " & ".join(
                    (
                        _target_tex(claim),
                        _basis_points(_require_number(loss, "mean")),
                        _basis_point_interval(loss, "confidence_interval_95"),
                        _basis_points(_require_number(loss, "median")),
                        _basis_points(
                            _require_number(loss, "sample_standard_deviation")
                        ),
                        _basis_points(_require_number(loss, "root_mean_square")),
                        _basis_points(_require_number(loss, "mean_absolute_loss")),
                        _basis_points(_require_number(loss, "quantile_95")),
                        _basis_points(_require_number(loss, "quantile_99")),
                        _basis_points(_require_number(loss, "expected_shortfall_95")),
                        _basis_points(_require_number(loss, "expected_shortfall_99")),
                    )
                )
                + " \\\\"
            )
        lines.extend(
            (
                "\\bottomrule",
                "\\end{longtable}",
                "\\medskip",
                "\\scriptsize",
                "\\begin{longtable}{@{}lrrrrrrrr@{}}",
                "\\caption{Target-level cost, turnover, exposure, sparsity, and numerical-gate incidence.}\\\\",
                "\\toprule",
                "Target & Cost & Entry turnover & Gross & Net & $|$Net$|$ & Nonzero & Cap rate & Nonconv. rate \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        for claim in sorted(targets, key=_target_sort_key):
            target = _require_object(targets, claim)
            lines.append(
                " & ".join(
                    (
                        _target_tex(claim),
                        _basis_points(_require_number(target, "mean_transaction_cost")),
                        f"{_require_number(target, 'mean_entry_turnover_notional'):.4f}",
                        f"{_require_number(target, 'mean_gross_exposure'):.4f}",
                        f"{_require_number(target, 'mean_net_exposure'):.4f}",
                        f"{_require_number(target, 'mean_absolute_net_exposure'):.4f}",
                        f"{_require_number(target, 'mean_nonzero_actions'):.2f}",
                        _percent(
                            _require_number(
                                target,
                                "numerical_cap_hit_result_rate",
                            )
                        ),
                        _percent(_require_number(target, "nonconvergence_rate")),
                    )
                )
                + " \\\\"
            )
        lines.extend(("\\bottomrule", "\\end{longtable}", "\\end{landscape}", ""))
    for index, (objective, parameters) in enumerate(OBJECTIVE_GRIDS, start=1):
        lines.extend(
            (
                "\\begin{landscape}",
                f"\\subsection{{{_objective_title(objective)}}}",
                f"\\label{{app:objective-{index}}}",
                "\\scriptsize",
                f"\\begin{{longtable}}{{ll{'r' * len(parameters)}rr}}",
                f"\\caption{{Target-level {_objective_title(objective).lower()} scores and drawdowns.}}\\\\",
                "\\toprule",
                "Model/policy & Target & "
                + " & ".join(
                    _parameter_heading(objective, value) for value in parameters
                )
                + " & Pure RMSE & Economic MDD \\\\",
                "\\midrule",
                "\\endfirsthead",
                "\\toprule",
                "Model/policy & Target & "
                + " & ".join(
                    _parameter_heading(objective, value) for value in parameters
                )
                + " & Pure RMSE & Economic MDD \\\\",
                "\\midrule",
                "\\endhead",
            )
        )
        model_items = tuple(models.items())
        for model_index, (model_key, raw_model) in enumerate(model_items):
            model = _as_object(raw_model, model_key)
            metrics = _require_object(model, "metrics")
            targets = _require_object(metrics, "targets")
            for claim in sorted(targets, key=_target_sort_key):
                target = _require_object(targets, claim)
                objectives = _require_object(target, "economic_objectives")
                scores = _require_object(objectives, objective)
                lines.append(
                    " & ".join(
                        (
                            f"{_require_string(model, 'display_name')} "
                            f"[{_tex_escape(_require_string(model, 'specification'))}] "
                            f"({_policy_tex(model)})",
                            _target_tex(claim),
                            *(
                                _basis_points(_require_number(scores, parameter))
                                for parameter in parameters
                            ),
                            _basis_points(_require_number(target, "pure_rmse")),
                            _basis_points(
                                _require_number(target, "economic_maximum_drawdown")
                            ),
                        )
                    )
                    + " \\\\"
                )
            if model_index + 1 < len(model_items):
                lines.append("\\addlinespace")
        lines.extend(
            ("\\bottomrule", "\\end{longtable}", "\\end{landscape}", "")
        )
    return "\n".join(lines) + "\n"


def _metric(metrics: JsonObject, group: str, name: str) -> float:
    return _require_number(_require_object(metrics, group), name)


def _first_model_key(models: JsonObject, expected_model: str) -> str:
    for key, raw_model in models.items():
        model = _as_object(raw_model, key)
        if _require_string(model, "model") == expected_model:
            return key
    raise ValueError(f"summary has no {expected_model} policy")


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _basis_points(value: float) -> str:
    scaled = DISPLAY_SCALE * value
    if abs(scaled) < 0.005:
        return "0.00"
    return f"{scaled:.2f}"


def _optional_basis_points(value: object) -> str:
    return "--" if value is None else _basis_points(_number(value))


def _basis_point_interval(payload: JsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a numeric pair")
    return f"[{_basis_points(_number(value[0]))}, {_basis_points(_number(value[1]))}]"


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


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
    parameter = payload.get("fitted_parameter")
    if isinstance(parameter, bool) or not isinstance(parameter, int | float):
        raise ValueError("fitted_parameter must be numeric for an optimized policy")
    symbol = "\\kappa" if objective == "mean-volatility" else "\\lambda"
    title = {
        "mean-variance": "MV",
        "mean-volatility": "MVol",
        "entropy": "Ent",
        "power-exponential": "Power",
    }.get(objective, objective)
    return f"{title}, ${symbol}={float(parameter):g}$"


def _target_tex(claim: str) -> str:
    parts = claim.split("-")
    if len(parts) < 4 or parts[0] != "digital":
        return claim.replace("_", "\\_")
    right = parts[1]
    maturity = parts[2].upper()
    strike = "ATM" if claim.endswith("-atm") else f"${claim.rsplit('nms', maxsplit=1)[1]}$ NMS"
    return f"{maturity} {right} {strike}"


def _target_sort_key(claim: str) -> tuple[int, int, float]:
    maturity = 1 if "-1d-" in claim else 3
    right = 0 if "-put-" in claim else 1
    nms = 0.0 if claim.endswith("-atm") else float(claim.rsplit("nms", maxsplit=1)[1])
    return maturity, right, nms


def _record_number(payload: dict[str, object], name: str) -> float:
    return _require_number(payload, name)


def _optional_number(payload: JsonObject, name: str) -> float | None:
    value = payload.get(name)
    return None if value is None else _number(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("value must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return number


def _require_record_losses(
    payload: JsonObject,
) -> dict[str, tuple[dict[str, object], ...]]:
    value = payload.get("record_losses")
    if not isinstance(value, dict):
        raise ValueError("record_losses must be an object")
    return value


def _read_json_object(path: Path) -> JsonObject:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _as_object(payload, str(path))


def _as_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_object(payload: JsonObject, name: str) -> JsonObject:
    return _as_object(payload.get(name), name)


def _require_object_list(payload: JsonObject, name: str) -> list[JsonObject]:
    values = payload.get(name)
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    return [_as_object(value, name) for value in values]


def _require_string_list(payload: JsonObject, name: str) -> list[str]:
    values = payload.get(name)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{name} must be a string list")
    return values


def _require_string(payload: JsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_number(payload: JsonObject, name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_integer(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _default_result_path(*parts: str) -> Path | None:
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        return None
    return Path(root).joinpath("experiments", "hedging", "daily", *parts)


def _default_dataset_path() -> Path | None:
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        return None
    return default_public_candidate_path()


def main() -> None:
    """Parse completed runs and write reproducible report inputs."""
    report_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_default_dataset_path())
    parser.add_argument(
        "--daily-matrix",
        type=Path,
        action="append",
        default=[],
        help=(
            "discover and authenticate every completed case in this production "
            "matrix; may be repeated"
        ),
    )
    parser.add_argument(
        "--require-ready-complete",
        action="store_true",
        help="reject any ready matrix case without completed results",
    )
    parser.add_argument(
        "--sanos-results",
        type=Path,
        default=_default_result_path(
            "production-matrix-v3", "sanos-lv-reference", "results.json"
        ),
    )
    parser.add_argument(
        "--historical-results",
        type=Path,
        default=_default_result_path(
            "production-matrix-v4",
            "historical-sanos-mean-variance-0p1",
            "results.json",
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=report_directory / "hedging_benchmark_summary_metrics.json",
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=report_directory / "hedging_benchmark_summary_metrics.tex",
    )
    parser.add_argument(
        "--objective-tables-output",
        type=Path,
        default=report_directory / "hedging_benchmark_summary_objective_tables.tex",
    )
    parser.add_argument(
        "--objective-appendix-output",
        type=Path,
        default=report_directory / "hedging_benchmark_summary_objective_appendix.tex",
    )
    arguments = parser.parse_args()
    if arguments.dataset is None:
        parser.error("--dataset is required without QUANTDATA_ROOT")
    if not arguments.daily_matrix and (
        arguments.sanos_results is None or arguments.historical_results is None
    ):
        parser.error(
            "--sanos-results and --historical-results are required unless "
            "--daily-matrix is supplied"
        )
    if arguments.daily_matrix:
        policies = daily_policy_inputs_from_matrices(
            tuple(arguments.daily_matrix),
            require_ready_complete=arguments.require_ready_complete,
        )
        payload = build_daily_policy_summary_metrics(
            policies=policies,
            dataset=arguments.dataset,
        )
    else:
        payload = build_summary_metrics(
            sanos_results=arguments.sanos_results,
            historical_results=arguments.historical_results,
            dataset=arguments.dataset,
        )
    write_summary_metrics(
        payload,
        json_output=arguments.json_output,
        tex_output=arguments.tex_output,
        objective_tables_output=arguments.objective_tables_output,
        objective_appendix_output=arguments.objective_appendix_output,
    )
    print(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "json_output": str(arguments.json_output),
                "tex_output": str(arguments.tex_output),
                "objective_tables_output": str(arguments.objective_tables_output),
                "objective_appendix_output": str(arguments.objective_appendix_output),
                "policy_count": len(_require_object(payload, "models")),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
