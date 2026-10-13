"""Build reproducible headline metrics for the hedging benchmark summary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]

SUMMARY_SCHEMA = "hedging-benchmark-summary-metrics-v1"
RESULTS_SCHEMA = "hedging-daily-results-v2"
EXPECTED_CLAIMS = 20
EXPECTED_OBSERVATIONS = 611


def build_summary_metrics(
    *,
    sanos_results: Path,
    historical_results: Path,
) -> JsonObject:
    """Read and validate the two completed runs, then aggregate headline risk."""
    sanos = _read_results(sanos_results, expected_model="sanos-lv")
    historical = _read_results(
        historical_results,
        expected_model="historical-sanos",
    )
    sanos_manifest = _require_string(sanos, "dataset_manifest_sha256")
    historical_manifest = _require_string(historical, "dataset_manifest_sha256")
    if sanos_manifest != historical_manifest:
        raise ValueError("completed summary runs use different dataset manifests")

    sanos_metrics = _aggregate_run(sanos)
    historical_metrics = _aggregate_run(historical)
    sanos_all = _require_object(sanos_metrics, "all")
    historical_all = _require_object(historical_metrics, "all")
    sanos_unhedged = _require_number(sanos_all, "unhedged_rmse")
    historical_unhedged = _require_number(historical_all, "unhedged_rmse")
    if not math.isclose(sanos_unhedged, historical_unhedged, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("completed runs do not share the same unhedged claims")
    sanos_pure = _require_number(sanos_all, "pure_rmse")
    historical_pure = _require_number(historical_all, "pure_rmse")

    return {
        "schema": SUMMARY_SCHEMA,
        "dataset_manifest_sha256": sanos_manifest,
        "models": {
            "sanos-lv": {
                "run_identity": _require_string(sanos, "run_identity"),
                "measure": _require_string(sanos, "measure"),
                "result_count": _require_integer(sanos, "result_count"),
                "metrics": sanos_metrics,
            },
            "historical-sanos": {
                "run_identity": _require_string(historical, "run_identity"),
                "measure": _require_string(historical, "measure"),
                "result_count": _require_integer(historical, "result_count"),
                "metrics": historical_metrics,
            },
        },
        "comparisons": {
            "sanos_pure_rmse_reduction_vs_historical": 1.0
            - sanos_pure / historical_pure,
        },
    }


def write_summary_metrics(
    payload: JsonObject,
    *,
    json_output: Path,
    tex_output: Path,
) -> None:
    """Write deterministic JSON evidence and the matching TeX macro fragment."""
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    tex_output.write_text(_tex_fragment(payload), encoding="utf-8")


def _read_results(path: Path, *, expected_model: str) -> JsonObject:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"daily results must be a JSON object: {path}")
    if payload.get("schema") != RESULTS_SCHEMA:
        raise ValueError(f"unexpected daily-results schema: {path}")
    if payload.get("model") != expected_model:
        raise ValueError(f"expected {expected_model} results: {path}")
    if _require_integer(payload, "result_count") != (
        EXPECTED_CLAIMS * EXPECTED_OBSERVATIONS
    ):
        raise ValueError(f"daily results are not the complete production population: {path}")
    return payload


def _aggregate_run(payload: JsonObject) -> JsonObject:
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
    groups = {
        "all": overall,
        "one_day": [value for value in overall if "-1d-" in value["claim"]],
        "three_day": [value for value in overall if "-3d-" in value["claim"]],
    }
    if len(groups["one_day"]) != 10 or len(groups["three_day"]) != 10:
        raise ValueError("daily results must contain ten 1D and ten 3D claims")
    return {label: _aggregate_rows(rows) for label, rows in groups.items()}


def _aggregate_rows(rows: list[JsonObject]) -> JsonObject:
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
            rows,
            "pure_hedge_loss",
            "expected_shortfall_95",
        ),
        "mean_absolute_loss": _mean_loss_statistic(
            rows,
            "pure_hedge_loss",
            "mean_absolute_loss",
        ),
        "mean_transaction_cost": _mean_row_number(rows, "mean_transaction_cost"),
        "mean_entry_turnover_notional": _mean_row_number(
            rows,
            "mean_entry_turnover_notional",
        ),
        "mean_nonzero_actions": _mean_row_number(rows, "mean_nonzero_actions"),
        "cap_hits": sum(_require_integer(value, "cap_hits") for value in rows),
        "nonconverged_results": sum(
            _require_integer(value, "nonconverged_results") for value in rows
        ),
    }


def _pooled_rmse(rows: list[JsonObject], loss_name: str) -> float:
    squared = [
        _require_number(_require_object(value, loss_name), "root_mean_square") ** 2
        for value in rows
    ]
    return math.sqrt(sum(squared) / len(squared))


def _mean_loss_statistic(
    rows: list[JsonObject],
    loss_name: str,
    statistic: str,
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
        _require_object(models, "historical-sanos"),
        "metrics",
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
                    comparison,
                    "sanos_pure_rmse_reduction_vs_historical",
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
                        _decimal(_metric(metrics, group_name, "unhedged_rmse"), 5),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}PureRmse",
                        _decimal(_metric(metrics, group_name, "pure_rmse"), 5),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}Reduction",
                        _percent(
                            _metric(metrics, group_name, "pure_rmse_reduction")
                        ),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}EconomicRmse",
                        _decimal(_metric(metrics, group_name, "economic_rmse"), 5),
                    ),
                    _macro(
                        f"{prefix}{group_prefix}EsNinetyFive",
                        _decimal(
                            _metric(metrics, group_name, "expected_shortfall_95"),
                            5,
                        ),
                    ),
                )
            )
        lines.extend(
            (
                _macro(
                    f"{prefix}AllMeanCost",
                    _decimal(_metric(metrics, "all", "mean_transaction_cost"), 6),
                ),
                _macro(
                    f"{prefix}AllTurnover",
                    _significant(
                        _metric(metrics, "all", "mean_entry_turnover_notional"),
                        4,
                    ),
                ),
                _macro(
                    f"{prefix}AllNonzero",
                    _decimal(_metric(metrics, "all", "mean_nonzero_actions"), 1),
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
    historical_all = _require_object(historical, "all")
    sanos_model = _require_object(models, "sanos-lv")
    historical_model = _require_object(models, "historical-sanos")
    lines.extend(
        (
            _macro(
                "SanosAllPureRmseSix",
                _decimal(_require_number(sanos_all, "pure_rmse"), 6),
            ),
            _macro(
                "SanosAllEconomicRmseSix",
                _decimal(_require_number(sanos_all, "economic_rmse"), 6),
            ),
            _macro(
                "SanosRecordCount",
                f"{_require_integer(sanos_model, 'result_count'):,}",
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
                "SanosRunPrefix",
                _require_string(sanos_model, "run_identity")[:12],
            ),
            _macro(
                "HistoricalRunPrefix",
                _require_string(historical_model, "run_identity")[:12],
            ),
        )
    )
    return "\n".join(lines) + "\n"


def _metric(metrics: JsonObject, group: str, name: str) -> float:
    return _require_number(_require_object(metrics, group), name)


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _decimal(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def _significant(value: float, digits: int) -> str:
    return f"{value:.{digits}g}"


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _as_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_object(payload: JsonObject, name: str) -> JsonObject:
    return _as_object(payload.get(name), name)


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
            "sanos-lv",
            "production-v2",
            "results",
            "results.json",
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
    )
    print(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "json_output": str(arguments.json_output),
                "tex_output": str(arguments.tex_output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
