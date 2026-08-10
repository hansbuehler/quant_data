"""Configuration-matched PCA versus SANOS benchmark aggregation."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

from .diagnostics import (
    DATA_QUALIFICATION,
    DYNAMIC_ARBITRAGE_QUALIFICATION,
)

_MATCH_FIELDS = (
    "paths",
    "surface_factors",
    "surface_first",
    "mode",
    "option_return_horizon",
    "traded_nms",
    "expiry_step_offsets",
    "importance_proposal",
    "state_definition",
    "neighbors",
    "candidate_neighbors",
    "cost_proportion",
    "roundoff_multiplier",
    "seed",
)


def _load_report(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(value: Mapping[str, Any], *names: str) -> Any:
    result: Any = value
    for name in names:
        if not isinstance(result, Mapping) or name not in result:
            return None
        result = result[name]
    return result


def _importance_signature(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping) and "config" in value:
        value = value["config"]
    if not isinstance(value, Mapping):
        return value
    return {
        name: value.get(name)
        for name in (
            "spot_std_multiplier",
            "base_probability",
            "target_action_count",
        )
    }


def _pca_match_configuration(report: Mapping[str, Any]) -> dict[str, Any]:
    configuration = report.get("configuration", {})
    sampling = report.get("sampling_diagnostics", {})
    local_configuration = _nested(
        report, "conditional_local_tail", "configuration"
    ) or {}
    surface_states = configuration.get("num_surface_state")
    return {
        "paths": configuration.get("num_samples"),
        "surface_factors": configuration.get("num_alpha"),
        "surface_first": _nested(report, "ar1", "surface_first"),
        "mode": configuration.get("mode"),
        "option_return_horizon": configuration.get("option_return_horizon"),
        "traded_nms": configuration.get("num_traded_nms"),
        "expiry_step_offsets": sampling.get("expiry_step_offsets"),
        "importance_proposal": _importance_signature(
            sampling.get("importance_sampling")
        ),
        "state_definition": {
            "mode": "raw",
            "features": 1 + surface_states if surface_states is not None else None,
        },
        "neighbors": local_configuration.get(
            "neighbors", configuration.get("neighbors")
        ),
        "candidate_neighbors": local_configuration.get(
            "instrument_candidate_neighbors",
            configuration.get("candidate_neighbors"),
        ),
        "cost_proportion": configuration.get("cost_proportion"),
        "roundoff_multiplier": 32,
        "seed": configuration.get("seed"),
    }


def _sanos_match_configuration(report: Mapping[str, Any]) -> dict[str, Any]:
    configuration = report.get("configuration", {})
    sampling = report.get("sampling_diagnostics", {})
    local_config = report.get("config", {})
    offsets = sampling.get("expiry_step_offsets")
    expiries = len(offsets) if isinstance(offsets, Sequence) else None
    instruments = report.get("instruments")
    state_mean = _nested(report, "state", "training_mean")
    state_features = (
        len(state_mean)
        if isinstance(state_mean, Sequence)
        else configuration.get("state_features")
    )
    traded_nms = (
        (instruments - 1) // expiries
        if isinstance(instruments, int) and expiries
        else configuration.get("num_traded_nms")
    )
    return {
        "paths": report.get("paths", configuration.get("num_samples")),
        "surface_factors": report.get(
            "num_surface_alpha", configuration.get("num_alpha")
        ),
        "surface_first": report.get(
            "surface_first", configuration.get("surface_first")
        ),
        "mode": sampling.get(
            "mode", report.get("market", configuration.get("mode"))
        ),
        "option_return_horizon": sampling.get(
            "option_return_horizon",
            configuration.get("option_return_horizon"),
        ),
        "traded_nms": traded_nms,
        "expiry_step_offsets": offsets,
        "importance_proposal": _importance_signature(
            sampling.get("importance_sampling")
        ),
        "state_definition": {
            "mode": local_config.get("state_mode", "raw"),
            "features": state_features,
        },
        "neighbors": local_config.get(
            "neighbors", configuration.get("neighbors")
        ),
        "candidate_neighbors": local_config.get(
            "instrument_candidate_neighbors",
            configuration.get("candidate_neighbors"),
        ),
        "cost_proportion": configuration.get("cost_proportion"),
        "roundoff_multiplier": local_config.get(
            "roundoff_multiplier", configuration.get("roundoff_multiplier")
        ),
        "seed": configuration.get("seed", report.get("seed")),
    }


def _nominal_key(report: Mapping[str, Any], *, pca: bool) -> tuple[Any, ...]:
    if pca:
        configuration = report.get("configuration", {})
        return (
            configuration.get("num_samples"),
            configuration.get("num_alpha"),
            configuration.get("mode"),
            configuration.get("option_return_horizon"),
        )
    configuration = report.get("configuration", {})
    return (
        report.get("paths", configuration.get("num_samples")),
        report.get("num_surface_alpha", configuration.get("num_alpha")),
        report.get("market", configuration.get("mode")),
        _nested(report, "sampling_diagnostics", "option_return_horizon")
        or configuration.get("option_return_horizon"),
    )


def _minimum_minority_probability(
    per_instrument: Mapping[str, Any],
) -> dict[str, Any] | None:
    if "weighted_minority_probability" in per_instrument:
        minority = np.asarray(
            per_instrument["weighted_minority_probability"], dtype=np.float64
        )
    else:
        positive = np.asarray(
            per_instrument.get("weighted_positive_probability", []),
            dtype=np.float64,
        )
        negative = np.asarray(
            per_instrument.get("weighted_negative_probability", []),
            dtype=np.float64,
        )
        if positive.shape != negative.shape or positive.size == 0:
            return None
        minority = np.minimum(positive, negative)
    if minority.size <= 1:
        return None
    option = int(np.argmin(minority[1:])) + 1
    result: dict[str, Any] = {
        "value": float(minority[option]),
        "instrument": option,
    }
    positive = np.asarray(
        per_instrument.get("weighted_positive_probability", []),
        dtype=np.float64,
    )
    negative = np.asarray(
        per_instrument.get("weighted_negative_probability", []),
        dtype=np.float64,
    )
    positive_error = np.asarray(
        per_instrument.get(
            "weighted_path_clustered_positive_probability_error", []
        ),
        dtype=np.float64,
    )
    negative_error = np.asarray(
        per_instrument.get(
            "weighted_path_clustered_negative_probability_error", []
        ),
        dtype=np.float64,
    )
    if (
        positive.shape == negative.shape == positive_error.shape == negative_error.shape
        and option < positive.size
    ):
        error = (
            positive_error[option]
            if positive[option] <= negative[option]
            else negative_error[option]
        )
        result["path_clustered_standard_error"] = float(error)
        result["normal_95_interval"] = [
            max(0.0, float(minority[option] - 1.96 * error)),
            min(1.0, float(minority[option] + 1.96 * error)),
        ]
    return result


def _local_metrics(report: Mapping[str, Any], split: str) -> dict[str, Any]:
    local = report.get("conditional_local_tail", report)
    value = local.get(split, {}) if isinstance(local, Mapping) else {}
    exclusivity = _nested(
        value, "zero_adverse", "states_by_direction_exclusivity"
    ) or {}
    candidates = sum(
        int(exclusivity.get(name, 0))
        for name in ("long_only", "short_only", "both_directions", "both")
    )
    policy = value.get("zero_adverse_single_instrument_policy", {})
    return {
        "one_sided_candidate_states": candidates,
        "realized_adverse_policy_outcomes": policy.get(
            "realized_losses_beyond_roundoff"
        ),
        "local_effective_sample_size_mean": _nested(
            value, "local_effective_sample_size", "mean"
        ),
        "neighbor_radius_mean": _nested(value, "neighbor_radius", "mean"),
    }


def _static_metrics(report: Mapping[str, Any]) -> dict[str, Any] | None:
    static = report.get("generated_surface_static_arbitrage")
    if not isinstance(static, Mapping):
        return None
    aggregate = _nested(static, "primary", "aggregate")
    if not isinstance(aggregate, Mapping):
        return None
    magnitude = aggregate.get("conditional_magnitude", {})
    return {
        "violating_surface_fraction": aggregate.get(
            "violating_surface_fraction"
        ),
        "violating_surface_fraction_path_clustered_error": aggregate.get(
            "physical_weighted_violating_surface_fraction_path_clustered_error"
        ),
        "path_fraction": aggregate.get("path_fraction"),
        "q99_violation_value": magnitude.get("q99"),
        "maximum_violation_value": magnitude.get("maximum"),
    }


def _importance_metrics(report: Mapping[str, Any]) -> dict[str, Any] | None:
    importance = _nested(report, "sampling_diagnostics", "importance_sampling")
    if importance is None:
        return {
            "enabled": False,
            "mean_p_over_q": 1.0,
            "effective_sample_fraction": 1.0,
        }
    weights = importance.get("path_weights", {})
    spot = importance.get("log_spot_control_at_target_steps", {})
    variance = importance.get("realized_log_variance_control", {})
    return {
        "enabled": True,
        "mean_p_over_q": weights.get("mean_density"),
        "effective_sample_fraction": weights.get("effective_sample_fraction"),
        "maximum_spot_mean_error_in_direct_P_std": spot.get(
            "maximum_mean_difference_in_direct_P_std"
        ),
        "maximum_relative_spot_variance_error": spot.get(
            "maximum_relative_variance_difference"
        ),
        "relative_realized_variance_mean_error": variance.get(
            "relative_mean_difference"
        ),
        "relative_realized_variance_variance_error": variance.get(
            "relative_variance_difference"
        ),
    }


def _run_metrics(report: Mapping[str, Any], *, pca: bool) -> dict[str, Any]:
    if pca:
        returns = report.get("return_diagnostics", {})
        unconditional = {
            split: returns.get(split, {}) for split in ("training", "validation")
        }
    else:
        unconditional = {
            split: report.get(f"unconditional_{split}", {})
            for split in ("training", "validation")
        }
    unconditional_metrics: dict[str, Any] = {}
    for split, value in unconditional.items():
        per_instrument = value.get("per_instrument", {})
        if pca:
            zero = value.get("zero_observed_option_sides", {})
            one_sided = len(
                set(zero.get("without_positive_path", ()))
                | set(zero.get("without_negative_path", ()))
            )
        else:
            one_sided = value.get(
                "strict_one_sided_instruments_with_roundoff_tolerance"
            )
        unconditional_metrics[split] = {
            "one_sided_instruments": one_sided,
            "minimum_minority_probability": (
                _minimum_minority_probability(per_instrument)
            ),
        }
    return {
        "static_arbitrage": _static_metrics(report),
        "unconditional": unconditional_metrics,
        "conditional_local_tail": {
            split: _local_metrics(report, split)
            for split in ("training", "validation")
        },
        "importance_sampling": _importance_metrics(report),
        "surface_fidelity": report.get("invariant_surface_distribution"),
        "weekly_path_fidelity": report.get("weekly_path_fidelity"),
    }


def _difference(left: Any, right: Any) -> dict[str, Any] | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return None
    return {
        "PCA_minus_SANOS": float(left - right),
        "PCA_over_SANOS": float(left / right) if abs(right) > 1.0e-12 else None,
    }


def _primary_differences(
    pca: Mapping[str, Any], sanos: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pca_static = pca.get("static_arbitrage")
    sanos_static = sanos.get("static_arbitrage")
    if isinstance(pca_static, Mapping) and isinstance(sanos_static, Mapping):
        for name in (
            "violating_surface_fraction",
            "path_fraction",
            "q99_violation_value",
            "maximum_violation_value",
        ):
            result[f"static_{name}"] = _difference(
                pca_static.get(name), sanos_static.get(name)
            )
        pca_error = pca_static.get(
            "violating_surface_fraction_path_clustered_error"
        )
        sanos_error = sanos_static.get(
            "violating_surface_fraction_path_clustered_error"
        )
        difference = result.get("static_violating_surface_fraction")
        if (
            difference is not None
            and isinstance(pca_error, (int, float))
            and isinstance(sanos_error, (int, float))
        ):
            error = math.hypot(float(pca_error), float(sanos_error))
            center = difference["PCA_minus_SANOS"]
            difference["independent_path_clustered_standard_error"] = error
            difference["normal_95_interval"] = [
                center - 1.96 * error,
                center + 1.96 * error,
            ]
    for split in ("training", "validation"):
        left = pca["unconditional"][split]
        right = sanos["unconditional"][split]
        result[f"{split}_one_sided_instruments"] = _difference(
            left.get("one_sided_instruments"), right.get("one_sided_instruments")
        )
        left_minority = left.get("minimum_minority_probability") or {}
        right_minority = right.get("minimum_minority_probability") or {}
        result[f"{split}_minimum_minority_probability"] = _difference(
            left_minority.get("value"), right_minority.get("value")
        )
        left_local = pca["conditional_local_tail"][split]
        right_local = sanos["conditional_local_tail"][split]
        for name in (
            "one_sided_candidate_states",
            "realized_adverse_policy_outcomes",
            "local_effective_sample_size_mean",
            "neighbor_radius_mean",
        ):
            result[f"{split}_{name}"] = _difference(
                left_local.get(name), right_local.get(name)
            )
    return result


def _write_summary_figure(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    matched = [row for row in rows if row.get("configuration_match")]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    labels = [row["label"] for row in matched]
    metric_paths = (
        ("static_arbitrage", "violating_surface_fraction"),
        ("unconditional", "validation", "one_sided_instruments"),
        (
            "conditional_local_tail",
            "validation",
            "one_sided_candidate_states",
        ),
    )
    titles = (
        "static violating surfaces",
        "validation one-sided instruments",
        "validation local candidates",
    )
    for axis, path, title in zip(axes, metric_paths, titles, strict=True):
        pca_values = [_nested(row["PCA"], *path) for row in matched]
        sanos_values = [_nested(row["SANOS"], *path) for row in matched]
        x = np.arange(len(labels))
        if labels and all(value is not None for value in pca_values + sanos_values):
            axis.bar(x - 0.2, pca_values, 0.4, label="PCA")
            axis.bar(x + 0.2, sanos_values, 0.4, label="SANOS")
            axis.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
        else:
            axis.text(0.5, 0.5, "no fully matched reports", ha="center", va="center")
            axis.set_xticks([])
        axis.set_title(title)
    if matched:
        axes[0].legend()
    figure.subplots_adjust(bottom=0.34, wspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _write_markdown(report: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# PCA-IV-AR(1) versus SANOS",
        "",
        f"Status: **{report['status']}**.",
        "",
        DATA_QUALIFICATION,
        "",
        DYNAMIC_ARBITRAGE_QUALIFICATION,
        "",
        "| Run | Matched | PCA static fraction | SANOS static fraction | "
        "PCA/SANOS validation local candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["matched_runs"]:
        pca_static = _nested(
            row, "PCA", "static_arbitrage", "violating_surface_fraction"
        )
        sanos_static = _nested(
            row, "SANOS", "static_arbitrage", "violating_surface_fraction"
        )
        pca_local = _nested(
            row,
            "PCA",
            "conditional_local_tail",
            "validation",
            "one_sided_candidate_states",
        )
        sanos_local = _nested(
            row,
            "SANOS",
            "conditional_local_tail",
            "validation",
            "one_sided_candidate_states",
        )

        def shown(value: Any) -> str:
            return "N/A" if value is None else f"{value:.6g}"

        lines.append(
            f"| {row['label']} | {'yes' if row['configuration_match'] else 'no'} "
            f"| {shown(pca_static)} | {shown(sanos_static)} "
            f"| {shown(pca_local)} / {shown(sanos_local)} |"
        )
    if report["unmatched_or_mismatched"]:
        lines.extend(("", "## Incomplete matches", ""))
        for item in report["unmatched_or_mismatched"]:
            lines.append(f"- {item}")
    lines.extend(("", "## Interpretation", "", report["interpretation"], ""))
    output.write_text("\n".join(lines), encoding="utf-8")


def write_aggregate_comparison(
    *,
    pca_reports: Iterable[str | Path | Mapping[str, Any]],
    sanos_reports: Iterable[str | Path | Mapping[str, Any]],
    output: str | Path,
) -> dict[str, Any]:
    """Write a configuration-verified side-by-side JSON/Markdown comparison."""

    pca = [_load_report(value) for value in pca_reports]
    sanos = [_load_report(value) for value in sanos_reports]
    verify_inp(bool(pca), "at least one PCA report is required")
    sanos_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in sanos:
        sanos_by_key.setdefault(_nominal_key(item, pca=False), []).append(item)

    rows = []
    problems = []
    for pca_report in pca:
        key = _nominal_key(pca_report, pca=True)
        label = f"paths={key[0]},alpha={key[1]},mode={key[2]},return={key[3]}"
        candidates = sanos_by_key.get(key, [])
        if len(candidates) != 1:
            problems.append(
                f"{label}: expected one nominal SANOS report, found {len(candidates)}"
            )
            rows.append(
                {
                    "label": label,
                    "configuration_match": False,
                    "configuration_differences": {"SANOS_report_count": len(candidates)},
                    "PCA": _run_metrics(pca_report, pca=True),
                    "SANOS": {},
                    "differences": {},
                }
            )
            continue
        sanos_report = candidates[0]
        pca_configuration = _pca_match_configuration(pca_report)
        sanos_configuration = _sanos_match_configuration(sanos_report)
        differences = {
            name: {
                "PCA": pca_configuration.get(name),
                "SANOS": sanos_configuration.get(name),
            }
            for name in _MATCH_FIELDS
            if pca_configuration.get(name) != sanos_configuration.get(name)
        }
        pca_metrics = _run_metrics(pca_report, pca=True)
        sanos_metrics = _run_metrics(sanos_report, pca=False)
        matched = not differences
        if not matched:
            problems.append(
                f"{label}: material configuration differences in "
                f"{', '.join(differences)}"
            )
        rows.append(
            {
                "label": label,
                "configuration_match": matched,
                "configuration_differences": differences,
                "PCA_configuration": pca_configuration,
                "SANOS_configuration": sanos_configuration,
                "PCA": pca_metrics,
                "SANOS": sanos_metrics,
                "differences": (
                    _primary_differences(pca_metrics, sanos_metrics)
                    if matched
                    else {}
                ),
            }
        )

    unused_sanos = sum(len(values) for values in sanos_by_key.values()) - sum(
        1 for row in rows if row.get("SANOS_configuration") is not None
    )
    if unused_sanos:
        problems.append(f"{unused_sanos} SANOS report(s) were not paired")
    incomplete_pca = [
        item.get("status") for item in pca if item.get("status") != "complete"
    ]
    incomplete_sanos = [
        item.get("status") for item in sanos if item.get("status") != "complete"
    ]
    if incomplete_pca:
        problems.append(f"incomplete PCA report statuses: {incomplete_pca}")
    if incomplete_sanos:
        problems.append(f"incomplete SANOS report statuses: {incomplete_sanos}")
    complete = bool(sanos) and not problems and all(
        row["configuration_match"] for row in rows
    )
    interpretation = (
        "The reports are configuration matched. Differences are descriptive; "
        "probabilistic intervals use independent paths as clusters where both "
        "reports expose the required standard errors. Static and sampled "
        "fixed-contract conclusions remain separate."
        if complete
        else "No material-improvement conclusion is issued until every PCA run "
        "has one complete SANOS report with all material configuration fields "
        "verified and the required metrics present."
    )
    report: dict[str, Any] = {
        "status": "complete" if complete else "incomplete",
        "comparison": (
            "PCA-IV-AR(1) versus statically arbitrage-free SANOS decoder"
        ),
        "data_qualification": DATA_QUALIFICATION,
        "dynamic_arbitrage_qualification": DYNAMIC_ARBITRAGE_QUALIFICATION,
        "methodological_exclusion": (
            "No PCA path is penalized, rejected, projected, or reweighted by a "
            "static-arbitrage score. This is not Cont--Vuletic scenario reweighting."
        ),
        "matching_fields": list(_MATCH_FIELDS),
        "matched_runs": rows,
        "unmatched_or_mismatched": problems,
        "uncertainty": (
            "Independent Monte Carlo paths are the clusters. No standard error "
            "uses sqrt(paths * action_dates)."
        ),
        "interpretation": interpretation,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = path.with_suffix(".md")
    figure_path = path.with_name(f"{path.stem}-summary.png")
    report["markdown_summary"] = str(markdown_path)
    report["comparison_figure"] = str(figure_path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _write_markdown(report, markdown_path)
    _write_summary_figure(rows, figure_path)
    return report


__all__ = ["write_aggregate_comparison"]
