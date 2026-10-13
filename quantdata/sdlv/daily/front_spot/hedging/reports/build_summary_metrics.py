"""Build reproducible headline and target-level hedging benchmark metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import TypeAlias

import numpy as np

from quantdata.hedging.metrics import maximum_drawdown_from_losses

JsonObject: TypeAlias = dict[str, object]

SUMMARY_SCHEMA = "hedging-benchmark-summary-metrics-v2"
RESULTS_SCHEMA = "hedging-daily-results-v2"
EXPECTED_CLAIMS = 20
EXPECTED_OBSERVATIONS = 611
DISPLAY_SCALE = 10_000.0
OBJECTIVE_GRIDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mean_variance", ("0.1", "1", "10")),
    ("mean_volatility", ("0.1", "1", "2", "3")),
    ("entropy", ("0.1", "1", "10")),
    ("power_exponential", ("0.1", "1", "10")),
)


def build_summary_metrics(
    *,
    sanos_results: Path,
    historical_results: Path,
) -> JsonObject:
    """Read, authenticate, and aggregate the completed daily policies."""
    policies = (
        _read_policy(
            sanos_results,
            expected_model="sanos-lv",
            display_name="SANOS--LV",
            fitted_objective="reference",
            fitted_parameter=None,
        ),
        _read_policy(
            historical_results,
            expected_model="historical-sanos",
            display_name="Historical SANOS",
        ),
    )
    manifests = {
        _require_string(policy, "dataset_manifest_sha256") for policy in policies
    }
    if len(manifests) != 1:
        raise ValueError("completed summary runs use different dataset manifests")

    models: JsonObject = {}
    for policy in policies:
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

    sanos_metrics = _require_object(_require_object(models, "sanos-lv"), "metrics")
    historical_metrics = _require_object(
        _require_object(models, "historical-sanos"),
        "metrics",
    )
    sanos_all = _require_object(sanos_metrics, "all")
    historical_all = _require_object(historical_metrics, "all")
    sanos_unhedged = _require_number(sanos_all, "unhedged_rmse")
    historical_unhedged = _require_number(historical_all, "unhedged_rmse")
    if not math.isclose(sanos_unhedged, historical_unhedged, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("completed runs do not share the same unhedged claims")

    return {
        "schema": SUMMARY_SCHEMA,
        "dataset_manifest_sha256": manifests.pop(),
        "display_scale": DISPLAY_SCALE,
        "display_units": "basis points of inception normalized spot",
        "objective_scaling": (
            "risk functionals are evaluated on unscaled inception-spot-normalized "
            "economic losses; only the final scalar is multiplied by 10,000"
        ),
        "models": models,
        "comparisons": {
            "sanos_pure_rmse_reduction_vs_historical": 1.0
            - _require_number(sanos_all, "pure_rmse")
            / _require_number(historical_all, "pure_rmse"),
        },
        "objective_tables": _objective_tables(models),
    }


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
    path: Path,
    *,
    expected_model: str,
    display_name: str,
    fitted_objective: str | None = None,
    fitted_parameter: float | None = None,
) -> JsonObject:
    results_path = Path(path)
    results = _read_results(results_path, expected_model=expected_model)
    run = _read_json_object(results_path.parent / "run.json")
    if _require_string(run, "run_identity") != _require_string(results, "run_identity"):
        raise ValueError(f"run and results identities do not match: {results_path}")
    if fitted_objective is None:
        snapshot = _require_object(run, "model_config_snapshot")
        fields = _require_object(snapshot, "fields")
        fitted_objective = _require_string(fields, "objective")
        fitted_parameter = _require_number(fields, "risk_parameter")
    return {
        "key": expected_model,
        "display_name": display_name,
        "measure": _require_string(results, "measure"),
        "run_identity": _require_string(results, "run_identity"),
        "dataset_manifest_sha256": _require_string(results, "dataset_manifest_sha256"),
        "result_count": _require_integer(results, "result_count"),
        "fitted_objective": fitted_objective,
        "fitted_parameter": fitted_parameter,
        "results": results,
        "record_losses": _read_record_losses(
            results_path.parent / "records",
            expected_results=results,
        ),
    }


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
            rows.setdefault(claim, []).append(
                {
                    "origin_date": origin,
                    "unhedged_loss": -_require_number(result, "unhedged_pnl"),
                    "economic_loss": -_require_number(result, "hedged_pnl"),
                    "pure_hedge_loss": -_require_number(result, "pure_hedge_error"),
                    "pricing_loss": -_require_number(result, "pricing_difference"),
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
        "mean_entry_turnover_notional": _mean_row_number(
            rows, "mean_entry_turnover_notional"
        ),
        "mean_gross_exposure": _mean_row_number(rows, "mean_gross_exposure"),
        "cap_hits": sum(_require_integer(value, "cap_hits") for value in rows),
        "nonconverged_results": sum(
            _require_integer(value, "nonconverged_results") for value in rows
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
    return {
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
    }


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
        for key in ("sanos-lv", "historical-sanos"):
            model = _require_object(models, key)
            metrics = _require_object(model, "metrics")
            all_metrics = _require_object(metrics, "all")
            economic = _require_object(all_metrics, "economic_objectives")
            scores = _require_object(economic, objective)
            rows.append(
                {
                    "model": key,
                    "display_name": _require_string(model, "display_name"),
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
    sanos = _require_object(_require_object(models, "sanos-lv"), "metrics")
    historical = _require_object(
        _require_object(models, "historical-sanos"), "metrics"
    )
    comparison = _require_object(payload, "comparisons")
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
            _percent(
                _require_number(
                    comparison, "sanos_pure_rmse_reduction_vs_historical"
                )
            ),
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
    sanos_model = _require_object(models, "sanos-lv")
    historical_model = _require_object(models, "historical-sanos")
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
    for index, (objective, _) in enumerate(OBJECTIVE_GRIDS, start=1):
        table = _require_object(tables, objective)
        parameters = _require_string_list(table, "parameters")
        rows = _require_object_list(table, "rows")
        lines.extend(
            (
                "\\begin{table}[H]",
                "\\centering",
                "\\small",
                f"\\caption{{{_objective_title(objective)}: full out-of-sample economic-risk score.}}",
                f"\\label{{tab:objective-{index}}}",
                f"\\begin{{tabular}}{{ll{'r' * len(parameters)}rr}}",
                "\\toprule",
                "Model & Fitted policy & "
                + " & ".join(
                    _parameter_heading(objective, value) for value in parameters
                )
                + " & Book MDD & Worst target MDD \\\\",
                "\\midrule",
            )
        )
        for row in rows:
            scores = _require_object(row, "scores")
            lines.append(
                " & ".join(
                    (
                        _require_string(row, "display_name"),
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
        lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}", ""))
    return "\n".join(lines) + "\n"


def _objective_appendix_tex(payload: JsonObject) -> str:
    models = _require_object(payload, "models")
    lines = ["% Generated by build_summary_metrics.py; do not edit by hand."]
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
        for model_key in ("sanos-lv", "historical-sanos"):
            model = _require_object(models, model_key)
            metrics = _require_object(model, "metrics")
            targets = _require_object(metrics, "targets")
            for claim in sorted(targets, key=_target_sort_key):
                target = _require_object(targets, claim)
                objectives = _require_object(target, "economic_objectives")
                scores = _require_object(objectives, objective)
                lines.append(
                    " & ".join(
                        (
                            f"{_require_string(model, 'display_name')} ({_policy_tex(model)})",
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
            if model_key == "sanos-lv":
                lines.append("\\addlinespace")
        lines.extend(
            ("\\bottomrule", "\\end{longtable}", "\\end{landscape}", "")
        )
    return "\n".join(lines) + "\n"


def _metric(metrics: JsonObject, group: str, name: str) -> float:
    return _require_number(_require_object(metrics, group), name)


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _basis_points(value: float) -> str:
    scaled = DISPLAY_SCALE * value
    if abs(scaled) < 0.005:
        return "0.00"
    return f"{scaled:.2f}"


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


def _default_result_path(*parts: str) -> Path | None:
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        return None
    return Path(root).joinpath("experiments", "hedging", "daily", *parts)


def main() -> None:
    """Parse completed runs and write reproducible report inputs."""
    report_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sanos-results",
        type=Path,
        default=_default_result_path(
            "sanos-lv", "production-v2", "results", "results.json"
        ),
    )
    parser.add_argument(
        "--historical-results",
        type=Path,
        default=_default_result_path(
            "historical-sanos",
            "mean-volatility-kappa1-v2",
            "results",
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
    if arguments.sanos_results is None or arguments.historical_results is None:
        parser.error(
            "--sanos-results and --historical-results are required without "
            "QUANTDATA_ROOT"
        )
    payload = build_summary_metrics(
        sanos_results=arguments.sanos_results,
        historical_results=arguments.historical_results,
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
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
