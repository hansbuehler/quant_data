"""Run direct-state batch descent and FLEX state-compression benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from pathlib import Path

from cdxcore.util import is_filename
from cdxcore.verbose import Context
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.config import (
    FixedEtaFlexGeometryConfig,
    FlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    ForwardMuFlexGeometryConfig,
    LatentDecoderConfig,
    NativeStateConfig,
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import (
    FlexMarketDay,
    load_market_day,
    market_process_count,
)
from quantdata.dysanos.flex.training import (
    TrainingConvergenceError,
    fit_latent_states,
    fit_native_states,
    latent_checkpoint_proof,
    model_checkpoint_phase,
    native_checkpoint_proof,
    native_checkpoint_state_audit,
    native_target_identity,
)
from quantdata.pool import PoolConfig
from quantdata.torch.ffw import FeedForwardConfig

_DEFAULT_STATE_SIZES = (20, 100, 200, 1_000)
_CONTROLLED_DECODER_WIDTH = 1_000
_EXPIRY_TRIANGULAR_ALLOCATIONS = {
    "triangular-uniform": (10,) * 14,
    "triangular-conservative": (14, 15, 14, 13, 12, 10, 10, 10, 7, 7, 7, 7, 7, 7),
    "triangular-stronger": (16, 19, 17, 15, 13, 10, 10, 10, 5, 5, 5, 5, 5, 5),
    "triangular-strong-200": (
        23,
        27,
        25,
        22,
        19,
        14,
        14,
        14,
        7,
        7,
        7,
        7,
        7,
        7,
    ),
    "triangular-four-week-uniform-200": (50, 50, 50, 50),
    "triangular-four-week-front-loaded-200": (100, 50, 25, 25),
}
_EXPIRY_BENCHMARK_VARIANTS = (
    "global",
    *_EXPIRY_TRIANGULAR_ALLOCATIONS,
)
_DEFAULT_EXPIRY_BENCHMARK_VARIANTS = ("global", "triangular-uniform")


def _market_cache_geometry(
    geometry: FlexGeometryConfig,
) -> FlexGeometryConfig:
    """Use only market-eligibility fields in the cached market identity."""
    if type(geometry) is FlexGeometryConfig:
        return geometry
    return FlexGeometryConfig(
        **{
            field.name: getattr(geometry, field.name)
            for field in fields(FlexGeometryConfig)
        }
    )


def _latent_decoder_config(
    *,
    label: str,
    state_size: int,
    geometry: FlexGeometryConfig,
) -> LatentDecoderConfig:
    """Keep the standard comparison controlled and remove the 2,815 control bottleneck."""
    decoder_width = max(_CONTROLLED_DECODER_WIDTH, state_size)
    return LatentDecoderConfig(
        label=label,
        state_size=state_size,
        decoder_input_size=decoder_width,
        geometry=geometry,
        decoder_network=FeedForwardConfig(
            activation="SiLU",
            num_depth=2,
            num_width=decoder_width,
        ),
    )


def _pca_decoder_config(
    *,
    label: str,
    state_size: int,
    geometry: FlexGeometryConfig,
) -> PcaDecoderConfig:
    """Build the exact affine decoder whose bias and basis come from PCA."""
    return PcaDecoderConfig(
        label=label,
        state_size=state_size,
        geometry=geometry,
    )


def _triangular_pca_decoder_config(
    *,
    label: str,
    state_sizes_by_expiry: tuple[int, ...],
    geometry: ForwardMuFlexGeometryConfig,
) -> TriangularPcaDecoderConfig:
    """Build one expiry-causal affine decoder with the requested allocation."""
    return TriangularPcaDecoderConfig(
        label=label,
        state_size=sum(state_sizes_by_expiry),
        state_sizes_by_expiry=state_sizes_by_expiry,
        geometry=geometry,
    )


@dataclass(frozen=True)
class _MarketLoadResult:
    """One market outcome without allowing an isolated bad day to abort a run."""

    day: FlexMarketDay | None
    error: str | None


def _statistics_rows(
    statistics: dict[str, dict[str, int | float]],
) -> list[str]:
    rows: list[str] = []
    for bucket in ("all", "0DTE", "1DTE", "2DTE"):
        item = statistics[bucket]
        rows.append(
            f"| {bucket} | {item['num_options']} | {item['percent_zero']:.4f}% | "
            f"{item['positive_mean']:.6g} | {item['positive_median']:.6g} | "
            f"{item['surface_mean']:.6g} | {item['surface_median']:.6g} |"
        )
    return rows


def _surface_fit_rows(statistics: dict[str, int | float]) -> list[str]:
    return [
        f"| Mean absolute error | {statistics['half_spread_mae']:.6g} | "
        f"{statistics['normalized_price_mae']:.6g} |",
        f"| Median absolute error | "
        f"{statistics['half_spread_median_absolute_error']:.6g} | "
        f"{statistics['normalized_price_median_absolute_error']:.6g} |",
        f"| Root mean square error | — | "
        f"{statistics['normalized_price_rmse']:.6g} |",
        f"| Signed bias | — | {statistics['normalized_price_bias']:.6g} |",
    ]


def _fit_section(
    title: str,
    result: dict[str, object],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Bucket | Options | Zero excess | Conditional positive mean | "
        "Conditional positive median | Surface mean | Surface median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_statistics_rows(result["statistics"]),
        "",
        "Whole-surface error against `market.py` `fit`:",
        "",
        "| Measure | Half-spreads | Normalized price |",
        "| --- | ---: | ---: |",
        *_surface_fit_rows(result["surface_fit"]),
        "",
        f"Mean training objective: `{result['objective']:.8g}`.",
        "",
    ]
    proof = result.get("checkpoint_proof")
    if isinstance(proof, dict):
        lines.extend(
            (
                f"Checkpoint SHA-256: `{proof['checkpoint_sha256']}`.",
                f"Latent-state SHA-256: `{proof['latent_state_sha256']}` "
                f"with shape `{proof['latent_state_shape']}`.",
                f"Model identity: `{proof['model_identity']['model_class']}` "
                f"version `{proof['model_identity']['config_version']}`.",
                "",
            )
        )
    convergence = result.get("market_convergence", result.get("convergence"))
    if isinstance(convergence, dict):
        lines.extend(
            (
                "Market-training convergence: "
                f"**{convergence['converged']}**; trailing-window relative "
                f"improvement `{convergence['window_relative_improvement']}`; "
                f"final gap to best `{convergence['final_relative_gap_to_best']}`.",
                "",
            )
        )
    return lines


def _markdown_report(report: dict[str, object]) -> str:
    geometry = report["geometry"]
    native_state_size = int(report.get("native_state_size", 2_815))
    model_dtes = geometry.get("explicit_model_business_dtes")
    if model_dtes is None:
        first_model_dte = geometry["first_model_business_dte"]
        last_model_dte = first_model_dte + geometry[
            "model_expiry_step_business_days"
        ] * (geometry["num_model_expiries"] - 1)
    else:
        first_model_dte = model_dtes[0]
        last_model_dte = model_dtes[-1]
    strike_description = (
        "one fixed relative-strike support shared across dates and expiries"
        if "inner_relative_strikes" in geometry
        else "one daily floating support shared across expiries"
        if "minimum_log_spacing_fraction" in geometry
        else "expiry-specific canonical-NMS supports"
    )
    backbone_description = (
        "one forward-mu coordinate per expiry interval"
        if geometry.get("forward_mu_backbone")
        else "one surface-wide eta coordinate"
    )
    lines = [
        "# FLEX batch-descent capacity benchmark",
        "",
        f"Status: **{report['status']}**.",
        "",
        "Geometry: canonical NMS `(-3,2)` market range, "
        f"{geometry['num_inner_strikes']} inner strikes, "
        f"{geometry['num_model_expiries']} model expiries from "
        f"{first_model_dte} to {last_model_dte} business days, "
        f"{strike_description}, "
        f"{native_state_size:,} native parameters per market date, and "
        f"{backbone_description}.",
        "",
        "Every stage uses every eligible real call and put book, with no "
        "validation holdout. The native state is learned directly by batched "
        "gradient descent; no LP solution is used as an input or benchmark.",
        "",
    ]
    failure = report.get("failure")
    if isinstance(failure, dict):
        lines.extend(
            (
                "## Training failure",
                "",
                f"Stage: `{failure['stage']}`.",
                "",
                f"{failure['type']}: {failure['message']}",
                "",
                "No earlier model state was substituted for the failed "
                "candidate.",
                "",
            )
        )
    if report.get("mode") in {
        "compression",
        "pca_compression",
        "expiry_triangular_compression",
    }:
        market_train_config = report["train_config"]
        supervised_train_config = report["supervised_train_config"]
        lines.extend(
            (
                "Compression optimizer configuration:",
                "",
                f"- supervised decoder learning rate: "
                f"`{supervised_train_config['learning_rate']}`;",
                f"- quote-objective learning rate: "
                f"`{market_train_config['learning_rate']}`;",
                f"- compiled SANOS optimizer steps: "
                f"`{market_train_config['compile_sanos_solver']}`;",
                "- final candidate and reported metrics: eager SANOS pricing;",
                "- convergence and no-regression gates: mandatory, with no "
                "fallback model selection.",
                "",
            )
        )
    native = report.get("native")
    if isinstance(native, dict):
        lines.extend(
            _fit_section(
                f"1. Direct {native_state_size:,}-state batch descent",
                native,
            )
        )
    latent = report.get("latent")
    if isinstance(latent, list):
        for result in latent:
            variant = result.get("variant")
            state_title = (
                str(variant).replace("_", " ").title()
                if variant is not None
                else f"state size {result['state_size']}"
            )
            initial = result.get("initial_evaluation")
            if isinstance(initial, dict):
                lines.extend(
                    _fit_section(
                        f"2a. Initial {state_title}",
                        initial,
                    )
                )
            lines.extend(
                _fit_section(
                    f"2b. Fine-tuned {state_title}",
                    result,
                )
            )
    return "\n".join(lines)


def _write_report(
    report: dict[str, object],
    *,
    output_directory: Path,
    label: str,
) -> None:
    """Write restart-visible JSON and Markdown after each completed stage."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{label}.json"
    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    json_temporary.replace(json_path)
    markdown_path = output_directory / f"{label}.md"
    markdown_temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown_temporary.write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    markdown_temporary.replace(markdown_path)


def _native_completion_path(*, output_directory: Path, label: str) -> Path:
    """Return the durable proof file for a completed native-stage report."""
    return output_directory / f"{label}-native-complete.json"


def _native_review_path(*, output_directory: Path, label: str) -> Path:
    """Return the standalone native-stage review path."""
    return output_directory / f"{label}-native-review.md"


def _native_review_markdown(completion: dict[str, object]) -> str:
    """Render fit quality and immutable checkpoint diagnostics for review."""
    proof = completion["native_checkpoint_proof"]
    audit = completion["native_completion_audit"]
    history = audit["history"]
    state = audit["state"]

    def optional(value: object) -> str:
        return "—" if value is None else f"{value}"

    lines = [
        "# FLEX native-state completion review",
        "",
        f"Status: **{completion['status']}**.",
        "",
        f"Checkpoint schema: `{proof['schema_version']}`; phase: "
        f"`{proof['phase']}`.",
        "",
        f"Checkpoint SHA-256: `{proof['checkpoint_sha256']}`.",
        "",
        f"Raw-state SHA-256: `{proof['native_state_sha256']}`.",
        "",
        "## Training convergence",
        "",
        "| Epochs | First objective | Last objective | Best objective | "
        "Best epoch | Last-25 relative improvement |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {history['num_epochs']} | {optional(history['first_objective'])} | "
        f"{optional(history['last_objective'])} | "
        f"{optional(history['best_objective'])} | "
        f"{optional(history['best_epoch'])} | "
        f"{optional(history['last_25_epoch_relative_improvement'])} |",
        "",
        "## Native-state bounds",
        "",
        f"All `{state['num_market_days']} x {state['native_state_size']}` raw "
        f"states finite: **{state['all_raw_states_finite']}**.",
        "",
        "| Coordinate | Minimum | Median | Maximum | Near lower | Near upper |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    coordinate_names = tuple(state["coordinates"])
    for name in coordinate_names:
        values = state["coordinates"][name]
        lines.append(
            f"| {name} | {values['minimum']:.8g} | {values['median']:.8g} | "
            f"{values['maximum']:.8g} | "
            f"{100.0 * values['fraction_near_lower_bound']:.6g}% | "
            f"{100.0 * values['fraction_near_upper_bound']:.6g}% |"
        )
    lines.extend(
        (
            "",
            "Near-bound fractions use sigmoid probabilities within "
            f"`{state['boundary_probability']}` of zero or one.",
            "",
            *_fit_section(
                "Market reconstruction",
                completion["native"],
            ),
        )
    )
    return "\n".join(lines)


def _validate_native_fit_report(native: dict[str, object]) -> None:
    """Require the complete finite metric payload before certification."""
    objective = native.get("objective")
    if (
        not isinstance(objective, int | float)
        or not math.isfinite(objective)
        or objective < 0.0
    ):
        raise ValueError("Native completion objective must be finite and non-negative")
    statistics = native.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("Native completion report has no excess statistics")
    required_excess_fields = {
        "num_options",
        "num_zero",
        "percent_zero",
        "num_positive",
        "positive_mean",
        "positive_median",
        "surface_mean",
        "surface_median",
    }
    for bucket in ("all", "0DTE", "1DTE", "2DTE"):
        values = statistics.get(bucket)
        if not isinstance(values, dict) or not required_excess_fields.issubset(
            values
        ):
            raise ValueError(
                f"Native completion report has incomplete {bucket} statistics"
            )
        if not all(
            isinstance(value, int | float) and math.isfinite(value)
            for value in values.values()
        ):
            raise ValueError(
                f"Native completion report has non-finite {bucket} statistics"
            )
        if (
            values["num_options"] < 0
            or values["num_zero"] < 0
            or values["num_positive"] < 0
            or values["num_zero"] + values["num_positive"]
            != values["num_options"]
            or not 0.0 <= values["percent_zero"] <= 100.0
        ):
            raise ValueError(
                f"Native completion report has inconsistent {bucket} counts"
            )
        expected_percent_zero = (
            100.0 * values["num_zero"] / values["num_options"]
            if values["num_options"] > 0
            else 0.0
        )
        if not math.isclose(
            values["percent_zero"],
            expected_percent_zero,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ) or any(
            values[field] < 0.0
            for field in (
                "positive_mean",
                "positive_median",
                "surface_mean",
                "surface_median",
            )
        ):
            raise ValueError(
                f"Native completion report has inconsistent {bucket} metrics"
            )
    surface_fit = native.get("surface_fit")
    required_surface_fields = {
        "num_options",
        "half_spread_mae",
        "half_spread_median_absolute_error",
        "normalized_price_mae",
        "normalized_price_median_absolute_error",
        "normalized_price_rmse",
        "normalized_price_bias",
    }
    if not isinstance(surface_fit, dict) or not required_surface_fields.issubset(
        surface_fit
    ):
        raise ValueError("Native completion report has incomplete surface metrics")
    if not all(
        isinstance(value, int | float) and math.isfinite(value)
        for value in surface_fit.values()
    ):
        raise ValueError("Native completion report has non-finite surface metrics")
    if any(
        surface_fit[field] < 0.0
        for field in required_surface_fields - {
            "num_options",
            "normalized_price_bias",
        }
    ):
        raise ValueError("Native completion report has negative surface errors")
    if surface_fit["num_options"] != statistics["all"]["num_options"]:
        raise ValueError(
            "Native completion surface and excess option counts do not match"
        )
    history = native.get("history", [])
    if not isinstance(history, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("objective"), int | float)
        and math.isfinite(item["objective"])
        and item["objective"] >= 0.0
        for item in history
    ):
        raise ValueError(
            "Native completion history must contain finite non-negative objectives"
        )
    _native_history_audit(history)


def _certified_native_payload(
    repriced_native: dict[str, object],
    completed_report: dict[str, object],
) -> dict[str, object]:
    """Retain certified training evidence after verifying the loaded reprice."""
    certified_native = completed_report.get("native")
    if not isinstance(certified_native, dict):
        raise ValueError("Completed native report has no certified native payload")
    _validate_native_fit_report(repriced_native)
    _validate_native_fit_report(certified_native)
    comparison_fields = (
        "statistics",
        "surface_fit",
        "objective",
        "target_identity",
    )
    mismatches = [
        field
        for field in comparison_fields
        if json.dumps(repriced_native.get(field), sort_keys=True)
        != json.dumps(certified_native.get(field), sort_keys=True)
    ]
    if mismatches:
        raise ValueError(
            "Loaded native reprice does not match its certified payload: "
            + ", ".join(mismatches)
        )
    return dict(certified_native)


def _certified_native_reprice_train_config(
    train_config: FlexTrainConfig,
    completed_report: dict[str, object],
) -> FlexTrainConfig:
    """Reprice the immutable native target with its certified reduction order."""
    certified_train_config = completed_report.get("train_config")
    if not isinstance(certified_train_config, dict):
        raise ValueError("Completed native report has no certified train config")
    day_batch_size = certified_train_config.get("day_batch_size")
    if not isinstance(day_batch_size, int) or day_batch_size <= 0:
        raise ValueError(
            "Completed native report has an invalid certified day batch size"
        )
    return replace(
        train_config,
        day_batch_size=day_batch_size,
        just_load=True,
        warm_start=False,
    )


def _native_history_audit(
    history: list[dict[str, object]],
) -> dict[str, int | float | None]:
    """Audit complete or sparse objective history using explicit epoch labels."""
    if not history:
        return {
            "num_epochs": 0,
            "num_recorded_epochs": 0,
            "first_objective": None,
            "last_objective": None,
            "best_objective": None,
            "best_epoch": None,
            "last_25_epoch_relative_improvement": None,
        }

    epochs: list[int] = []
    objectives: list[float] = []
    for index, item in enumerate(history):
        raw_epoch = item.get("epoch", index + 1)
        if (
            not isinstance(raw_epoch, int | float)
            or not math.isfinite(raw_epoch)
            or raw_epoch <= 0
            or not float(raw_epoch).is_integer()
        ):
            raise ValueError(
                "Native completion history epochs must be finite positive integers"
            )
        epochs.append(int(raw_epoch))
        objectives.append(float(item["objective"]))
    if any(right <= left for left, right in zip(epochs, epochs[1:])):
        raise ValueError(
            "Native completion history epochs must be strictly increasing"
        )

    best_index = min(range(len(objectives)), key=objectives.__getitem__)
    comparison_epoch = epochs[-1] - 25
    comparison_objective = next(
        (
            objective
            for epoch, objective in zip(epochs, objectives, strict=True)
            if epoch == comparison_epoch
        ),
        None,
    )
    return {
        "num_epochs": epochs[-1],
        "num_recorded_epochs": len(epochs),
        "first_objective": objectives[0],
        "last_objective": objectives[-1],
        "best_objective": objectives[best_index],
        "best_epoch": epochs[best_index],
        "last_25_epoch_relative_improvement": (
            (comparison_objective - objectives[-1])
            / max(abs(comparison_objective), 1.0e-16)
            if comparison_objective is not None
            else None
        ),
    }


def _completed_native_market_dates(
    report: dict[str, object],
    *,
    requested_dates: tuple[date, ...],
    expected_num_days: int,
) -> tuple[date, ...]:
    """Recover the exact ordered dates represented by the native checkpoint."""
    raw_dates = report.get("market_dates")
    if not isinstance(raw_dates, list) or not raw_dates or not all(
        isinstance(value, str) for value in raw_dates
    ):
        raise ValueError("Native completion report has no valid market-date list")
    try:
        market_dates = tuple(date.fromisoformat(value) for value in raw_dates)
    except ValueError as error:
        raise ValueError(
            "Native completion report contains an invalid market date"
        ) from error
    if tuple(sorted(set(market_dates))) != market_dates:
        raise ValueError(
            "Native completion market dates must be unique and sorted"
        )
    requested_set = set(requested_dates)
    if any(value not in requested_set for value in market_dates):
        raise ValueError(
            "Native completion contains a date outside the requested universe"
        )
    if report.get("num_days") != len(market_dates):
        raise ValueError(
            "Native completion report day count does not match its market dates"
        )
    if len(market_dates) != expected_num_days:
        raise ValueError(
            "Native completion market-date count does not match its checkpoint"
        )
    return market_dates


def _record_native_completion(
    report: dict[str, object],
    *,
    dates: tuple[date, ...],
    geometry: FlexGeometryConfig,
    checkpoint_path: Path,
    output_directory: Path,
    label: str,
) -> None:
    """Preserve native completion separately from later compression progress."""
    native = report.get("native")
    if not isinstance(native, dict):
        raise ValueError("Native completion report has no native fit")
    _validate_native_fit_report(native)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = _native_completion_path(
        output_directory=output_directory,
        label=label,
    )
    completed_report = dict(report)
    checkpoint_proof = native_checkpoint_proof(checkpoint_path)
    completed_report["native_checkpoint_proof"] = checkpoint_proof
    _completed_native_market_dates(
        completed_report,
        requested_dates=dates,
        expected_num_days=int(checkpoint_proof["num_market_days"]),
    )
    history = native.get("history", [])
    history_audit = _native_history_audit(history)
    completed_report["native_completion_audit"] = {
        "history": history_audit,
        "state": native_checkpoint_state_audit(
            checkpoint_path,
            geometry=geometry,
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(completed_report, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    review_path = _native_review_path(
        output_directory=output_directory,
        label=label,
    )
    review_temporary = review_path.with_suffix(review_path.suffix + ".tmp")
    review_temporary.write_text(
        _native_review_markdown(completed_report),
        encoding="utf-8",
    )
    review_temporary.replace(review_path)


def _require_completed_native_report(
    *,
    ticker: str,
    dates: tuple[date, ...],
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
    objective_config: FlexObjectiveConfig,
    checkpoint_path: Path,
    output_directory: Path,
    label: str,
) -> dict[str, object]:
    """Reject compression unless the matching native stage fully completed."""
    completion_path = _native_completion_path(
        output_directory=output_directory,
        label=label,
    )
    report_path = (
        completion_path
        if completion_path.exists()
        else output_directory / f"{label}.json"
    )
    if not report_path.exists():
        raise FileNotFoundError(
            "Compression requires a completed native report at "
            f"{report_path}. Run and review the native stage first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    json_geometry = json.loads(json.dumps(asdict(geometry)))
    json_market_config = json.loads(json.dumps(asdict(market_config)))
    json_objective_config = json.loads(json.dumps(asdict(objective_config)))
    expected = {
        "schema": "flex-batch-descent-capacity-v1",
        "label": label,
        "ticker": ticker,
        "requested_dates": [value.isoformat() for value in dates],
        "geometry": json_geometry,
        "market_config": json_market_config,
        "objective_config": json_objective_config,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if report.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Native completion report does not match this compression run: "
            + ", ".join(mismatches)
        )
    if report.get("status") != "native_complete" or not isinstance(
        report.get("native"),
        dict,
    ):
        raise RuntimeError(
            "Compression requires status 'native_complete' and native fit "
            f"statistics; found status {report.get('status')!r}."
        )
    _validate_native_fit_report(report["native"])
    current_proof = native_checkpoint_proof(checkpoint_path)
    saved_proof = report.get("native_checkpoint_proof")
    if report_path == completion_path:
        if saved_proof != current_proof:
            raise ValueError(
                "Native completion report does not match the current checkpoint"
            )
    elif saved_proof is not None and saved_proof != current_proof:
        raise ValueError(
            "Native report checkpoint proof does not match the current checkpoint"
        )
    if current_proof["schema_version"] == 2 and current_proof[
        "phase"
    ] != "complete":
        raise RuntimeError(
            "Compression requires a completed native checkpoint; found phase "
            f"{current_proof['phase']!r}."
        )
    if current_proof["schema_version"] not in {1, 2}:
        raise ValueError(
            "Unsupported native checkpoint schema "
            f"{current_proof['schema_version']!r}"
        )
    _completed_native_market_dates(
        report,
        requested_dates=dates,
        expected_num_days=int(current_proof["num_market_days"]),
    )
    if report_path != completion_path:
        _record_native_completion(
            report,
            dates=dates,
            geometry=geometry,
            checkpoint_path=checkpoint_path,
            output_directory=output_directory,
            label=label,
        )
        return json.loads(completion_path.read_text(encoding="utf-8"))
    return report


def certify_native_completion(
    *,
    ticker: str,
    dates: tuple[date, ...],
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
    objective_config: FlexObjectiveConfig,
    checkpoint_directory: Path,
    output_directory: Path,
    label: str,
) -> dict[str, object]:
    """Bind a finished native report to its checkpoint without loading markets."""
    checkpoint_path = checkpoint_directory / "native.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No native-state checkpoint exists at {checkpoint_path}"
        )
    lock_path = checkpoint_directory / f"{label}.lock"
    with FileLock(str(lock_path)):
        _require_completed_native_report(
            ticker=ticker,
            dates=dates,
            geometry=geometry,
            market_config=market_config,
            objective_config=objective_config,
            checkpoint_path=checkpoint_path,
            output_directory=output_directory,
            label=label,
        )
        completion_path = _native_completion_path(
            output_directory=output_directory,
            label=label,
        )
        return json.loads(completion_path.read_text(encoding="utf-8"))


def _latent_restart_config(
    train_config: FlexTrainConfig,
    checkpoint_path: Path,
) -> FlexTrainConfig:
    """Load completed sizes read-only while resuming an unfinished phase."""
    if (
        checkpoint_path.exists()
        and model_checkpoint_phase(checkpoint_path) == "complete"
        and not train_config.warm_start
    ):
        return replace(
            train_config,
            just_load=True,
            warm_start=False,
        )
    return train_config


def _load_market_job(
    ticker: str,
    trading_date: date,
    *,
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
) -> _MarketLoadResult:
    try:
        return _MarketLoadResult(
            day=load_market_day(
                ticker,
                trading_date,
                geometry=geometry,
                config=market_config,
                verbose=Context.quiet,
            ),
            error=None,
        )
    except Exception as error:  # noqa: BLE001
        return _MarketLoadResult(
            day=None,
            error=f"{type(error).__name__}: {error}",
        )


def run_production_gates(
    *,
    ticker: str,
    dates: tuple[date, ...],
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
    objective_config: FlexObjectiveConfig,
    train_config: FlexTrainConfig,
    supervised_train_config: FlexTrainConfig,
    workers: int,
    output_directory: Path,
    checkpoint_directory: Path,
    label: str,
    state_sizes: tuple[int, ...] = _DEFAULT_STATE_SIZES,
    run_compression: bool = False,
    pca_decoder: bool = False,
    expiry_benchmark: bool = False,
    expiry_benchmark_variants: tuple[
        str, ...
    ] = _DEFAULT_EXPIRY_BENCHMARK_VARIANTS,
    native_weights_warm_start_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Serialize all report and checkpoint mutations for one logical run."""
    if not is_filename(label):
        raise ValueError(f"'label' must be a valid filename, found {label!r}")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    training_lock = FileLock(str(checkpoint_directory / f"{label}.lock"))
    with training_lock:
        return _run_production_gates_locked(
            ticker=ticker,
            dates=dates,
            geometry=geometry,
            market_config=market_config,
            objective_config=objective_config,
            train_config=train_config,
            supervised_train_config=supervised_train_config,
            workers=workers,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            label=label,
            state_sizes=state_sizes,
            run_compression=run_compression,
            pca_decoder=pca_decoder,
            expiry_benchmark=expiry_benchmark,
            expiry_benchmark_variants=expiry_benchmark_variants,
            native_weights_warm_start_checkpoint=(
                native_weights_warm_start_checkpoint
            ),
            training_lock=training_lock,
        )


def _run_production_gates_locked(
    *,
    ticker: str,
    dates: tuple[date, ...],
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
    objective_config: FlexObjectiveConfig,
    train_config: FlexTrainConfig,
    supervised_train_config: FlexTrainConfig,
    workers: int,
    output_directory: Path,
    checkpoint_directory: Path,
    label: str,
    state_sizes: tuple[int, ...] = _DEFAULT_STATE_SIZES,
    run_compression: bool = False,
    pca_decoder: bool = False,
    expiry_benchmark: bool = False,
    expiry_benchmark_variants: tuple[
        str, ...
    ] = _DEFAULT_EXPIRY_BENCHMARK_VARIANTS,
    native_weights_warm_start_checkpoint: Path | None,
    training_lock: FileLock,
) -> dict[str, object]:
    """Run fitting while the caller holds the logical checkpoint lock."""
    if len(dates) == 0:
        raise ValueError("At least one requested market date is required")
    if expiry_benchmark and not isinstance(
        geometry,
        ForwardMuFlexGeometryConfig,
    ):
        raise ValueError("Expiry benchmark requires forward-mu geometry")
    if expiry_benchmark and (not run_compression or pca_decoder):
        raise ValueError(
            "Expiry benchmark requires compression and selects its own affine "
            "decoder variants"
        )
    if run_compression and native_weights_warm_start_checkpoint is not None:
        raise ValueError(
            "Native weights-only warm starts apply only to the native stage"
        )
    if (
        len(expiry_benchmark_variants) == 0
        or len(set(expiry_benchmark_variants)) != len(expiry_benchmark_variants)
        or any(
            variant not in _EXPIRY_BENCHMARK_VARIANTS
            for variant in expiry_benchmark_variants
        )
    ):
        raise ValueError(
            "Expiry benchmark variants must be a non-empty unique subset of "
            f"{_EXPIRY_BENCHMARK_VARIANTS}, found {expiry_benchmark_variants}"
        )
    if not is_filename(label):
        raise ValueError(f"'label' must be a valid filename, found {label!r}")
    if len(state_sizes) == 0 or any(
        state_size <= 0 or state_size > geometry.native_state_size
        for state_size in state_sizes
    ):
        raise ValueError(
            "Every compression state size must lie in "
            f"[1,{geometry.native_state_size}], found {state_sizes}"
        )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    native_checkpoint_path = checkpoint_directory / "native.pt"
    if run_compression and not native_checkpoint_path.exists():
        raise FileNotFoundError(
            "Compression requires a completed native-state checkpoint at "
            f"{native_checkpoint_path}. Run the native stage first."
        )
    completion_path = _native_completion_path(
        output_directory=output_directory,
        label=label,
    )
    if not run_compression:
        if completion_path.exists() and not train_config.warm_start:
            raise RuntimeError(
                "The native FLEX result is already certified. Use the "
                "compression or certification command, or request an explicit "
                "weights-only warm start to replace it."
            )
        if native_checkpoint_path.exists():
            checkpoint_phase = model_checkpoint_phase(native_checkpoint_path)
            if checkpoint_phase == "legacy_model" and not train_config.warm_start:
                raise RuntimeError(
                    "Native-mode loading cannot establish that a schema-1 "
                    "checkpoint is complete. Use '--certify-native-only' after "
                    "the legacy run writes its native_complete report, or use "
                    "an explicit '--warm-start' after an interrupted run."
                )
            if checkpoint_phase == "complete" and not (
                train_config.just_load or train_config.warm_start
            ):
                raise RuntimeError(
                    "The native FLEX checkpoint is complete; use 'just_load', "
                    "certification, or an explicit weights-only warm start."
                )
        if train_config.warm_start:
            completion_path.unlink(missing_ok=True)
    completed_native_report: dict[str, object] | None = None
    load_dates = dates
    if run_compression:
        completed_native_report = _require_completed_native_report(
            ticker=ticker,
            dates=dates,
            geometry=geometry,
            market_config=market_config,
            objective_config=objective_config,
            checkpoint_path=native_checkpoint_path,
            output_directory=output_directory,
            label=label,
        )
        load_dates = _completed_native_market_dates(
            completed_native_report,
            requested_dates=dates,
            expected_num_days=int(
                completed_native_report["native_checkpoint_proof"][
                    "num_market_days"
                ]
            ),
        )

    report: dict[str, object] = {
        "schema": "flex-batch-descent-capacity-v1",
        "label": label,
        "status": "market_loading",
        "ticker": ticker,
        "requested_dates": [value.isoformat() for value in dates],
        "geometry": asdict(geometry),
        "market_config": asdict(market_config),
        "objective_config": asdict(objective_config),
        "train_config": asdict(train_config),
        "supervised_train_config": asdict(supervised_train_config),
        "state_sizes": list(state_sizes),
        "native_state_size": geometry.native_state_size,
        "validation_holdout": False,
        "lp_warm_start": False,
        "native_weights_warm_start": (
            None
            if native_weights_warm_start_checkpoint is None
            else native_checkpoint_proof(native_weights_warm_start_checkpoint)
        ),
        "mode": (
            "expiry_triangular_compression"
            if run_compression and expiry_benchmark
            else "pca_compression"
            if run_compression and pca_decoder
            else "compression" if run_compression else "native"
        ),
        "expiry_benchmark_variants": (
            list(expiry_benchmark_variants) if expiry_benchmark else None
        ),
    }
    if completed_native_report is not None:
        report["native_failed_market_dates"] = completed_native_report.get(
            "failed_market_dates",
            [],
        )
        report["native_missing_market_dates"] = completed_native_report.get(
            "missing_market_dates",
            [],
        )
        for key in (
            "native_checkpoint_proof",
            "native_completion_audit",
            "native_recovery",
        ):
            if key in completed_native_report:
                report[key] = completed_native_report[key]
    _write_report(report, output_directory=output_directory, label=label)

    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()
    market_cache_geometry = _market_cache_geometry(geometry)
    jobs = {
        trading_date: pool.delayed(_load_market_job)(
            ticker,
            trading_date,
            geometry=market_cache_geometry,
            market_config=market_config,
        )
        for trading_date in load_dates
    }
    days: list[FlexMarketDay] = []
    failed_market_dates: list[dict[str, str]] = []
    missing_market_dates: list[str] = []
    for index, result in enumerate(pool.parallel(jobs)):
        trading_date, outcome = result
        if outcome.error is not None:
            failed_market_dates.append(
                {
                    "date": trading_date.isoformat(),
                    "error": outcome.error,
                }
            )
            print(
                f"market {index + 1}/{len(load_dates)} date={trading_date} "
                f"failed={outcome.error}",
                flush=True,
            )
        elif outcome.day is not None:
            days.append(outcome.day)
            print(
                f"market {index + 1}/{len(load_dates)} date={trading_date} "
                f"quotes={outcome.day.num_quotes}",
                flush=True,
            )
        else:
            missing_market_dates.append(trading_date.isoformat())
            print(
                f"market {index + 1}/{len(load_dates)} date={trading_date} "
                "no-market",
                flush=True,
            )
    days.sort(key=lambda day: day.trading_date)
    if len(days) == 0:
        raise RuntimeError("No eligible FLEX market days were loaded")
    market_days = tuple(days)
    report["market_dates"] = [day.trading_date.isoformat() for day in market_days]
    report["failed_market_dates"] = failed_market_dates
    report["missing_market_dates"] = missing_market_dates
    report["num_days"] = len(market_days)
    report["num_quotes"] = sum(day.num_quotes for day in market_days)
    report["status"] = (
        "compression_native_loading" if run_compression else "native_running"
    )
    _write_report(report, output_directory=output_directory, label=label)

    with training_lock:
        native_reprice_train_config = train_config
        if run_compression:
            assert completed_native_report is not None
            _require_completed_native_report(
                ticker=ticker,
                dates=dates,
                geometry=geometry,
                market_config=market_config,
                objective_config=objective_config,
                checkpoint_path=native_checkpoint_path,
                output_directory=output_directory,
                label=label,
            )
            native_reprice_train_config = _certified_native_reprice_train_config(
                train_config,
                completed_report=completed_native_report,
            )
        try:
            native_fit = fit_native_states(
                market_days,
                config=NativeStateConfig(
                    label=f"{label}-native",
                    geometry=geometry,
                ),
                objective_config=objective_config,
                train_config=native_reprice_train_config,
                checkpoint_path=native_checkpoint_path,
                weights_only_warm_start_path=(
                    native_weights_warm_start_checkpoint
                ),
            )
        except TrainingConvergenceError as error:
            report["status"] = "native_failed"
            report["failure"] = {
                "stage": "native",
                "type": type(error).__name__,
                "message": str(error),
            }
            _write_report(report, output_directory=output_directory, label=label)
            raise
        repriced_native = {
            "statistics": native_fit.evaluation.statistics.to_dict(),
            "surface_fit": native_fit.evaluation.surface_fit.to_dict(),
            "objective": native_fit.evaluation.objective,
            "elapsed_seconds": native_fit.elapsed_seconds,
            "history": list(native_fit.history),
            "convergence": native_fit.convergence,
            "target_identity": native_target_identity(
                native_fit.model,
                market_fingerprints=native_fit.market_fingerprints,
            ),
        }
        native_payload = (
            _certified_native_payload(
                repriced_native,
                completed_native_report,
            )
            if completed_native_report is not None
            else repriced_native
        )
        native_payload["convergence"] = native_fit.convergence
        report["native"] = native_payload
        report["status"] = "native_complete"
        _write_report(report, output_directory=output_directory, label=label)
        if not run_compression:
            _record_native_completion(
                report,
                dates=dates,
                geometry=geometry,
                checkpoint_path=native_checkpoint_path,
                output_directory=output_directory,
                label=label,
            )
            return report

        compression_specs: list[
            tuple[
                str,
                LatentDecoderConfig
                | PcaDecoderConfig
                | TriangularPcaDecoderConfig,
                Path,
            ]
        ] = []
        if expiry_benchmark:
            assert isinstance(geometry, ForwardMuFlexGeometryConfig)
            triangular_allocations = {
                variant: _EXPIRY_TRIANGULAR_ALLOCATIONS[variant]
                for variant in expiry_benchmark_variants
                if variant in _EXPIRY_TRIANGULAR_ALLOCATIONS
            }
            for variant, allocation in triangular_allocations.items():
                if len(allocation) != geometry.num_model_expiries:
                    raise ValueError(
                        f"{variant!r} requires {len(allocation)} model expiries, "
                        f"found {geometry.num_model_expiries}"
                    )
            report["expiry_state_allocations"] = {
                variant.removeprefix("triangular-"): list(allocation)
                for variant, allocation in triangular_allocations.items()
            }
            if "global" in expiry_benchmark_variants:
                allocation_sizes = {
                    sum(allocation)
                    for allocation in triangular_allocations.values()
                }
                surface_state_size = (
                    next(iter(allocation_sizes))
                    if len(allocation_sizes) == 1
                    else 140
                )
                compression_specs.append(
                    (
                        f"global_{surface_state_size}",
                        _pca_decoder_config(
                            label=f"{label}-global-{surface_state_size}",
                            state_size=surface_state_size,
                            geometry=geometry,
                        ),
                        checkpoint_directory / f"global-{surface_state_size}.pt",
                    )
                )
            for variant, allocation in triangular_allocations.items():
                allocation_label = variant.removeprefix("triangular-")
                surface_state_size = sum(allocation)
                compression_specs.append(
                    (
                        f"triangular_{allocation_label}_{surface_state_size}",
                        _triangular_pca_decoder_config(
                            label=f"{label}-{variant}-{surface_state_size}",
                            state_sizes_by_expiry=allocation,
                            geometry=geometry,
                        ),
                        checkpoint_directory
                        / f"{variant}-{surface_state_size}.pt",
                    )
                )
        else:
            for state_size in state_sizes:
                config = (
                    _pca_decoder_config(
                        label=f"{label}-pca-{state_size}",
                        state_size=state_size,
                        geometry=geometry,
                    )
                    if pca_decoder
                    else _latent_decoder_config(
                        label=f"{label}-latent-{state_size}",
                        state_size=state_size,
                        geometry=geometry,
                    )
                )
                compression_specs.append(
                    (
                        f"{'pca' if pca_decoder else 'latent'}_{state_size}",
                        config,
                        checkpoint_directory
                        / f"{'pca' if pca_decoder else 'latent'}-{state_size}.pt",
                    )
                )

        latent_reports: list[dict[str, object]] = []
        for variant, latent_config, latent_checkpoint_path in compression_specs:
            state_size = latent_config.state_size
            report["status"] = f"{variant}_running"
            _write_report(report, output_directory=output_directory, label=label)
            try:
                latent_fit = fit_latent_states(
                    market_days,
                    native_fit,
                    config=latent_config,
                    objective_config=objective_config,
                    supervised_train_config=supervised_train_config,
                    market_train_config=_latent_restart_config(
                        train_config,
                        latent_checkpoint_path,
                    ),
                    checkpoint_path=latent_checkpoint_path,
                )
            except TrainingConvergenceError as error:
                report["latent"] = latent_reports
                report["status"] = f"{variant}_failed"
                report["failure"] = {
                    "stage": variant,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                _write_report(
                    report,
                    output_directory=output_directory,
                    label=label,
                )
                raise
            latent_reports.append(
                {
                    "variant": variant,
                    "state_size": state_size,
                    "state_sizes_by_expiry": (
                        list(latent_config.state_sizes_by_expiry)
                        if isinstance(
                            latent_config,
                            TriangularPcaDecoderConfig,
                        )
                        else None
                    ),
                    "decoder_input_size": latent_fit.model.config.decoder_input_size,
                    "decoder_width": latent_fit.model.config.decoder_network.num_width,
                    "statistics": latent_fit.evaluation.statistics.to_dict(),
                    "surface_fit": latent_fit.evaluation.surface_fit.to_dict(),
                    "objective": latent_fit.evaluation.objective,
                    "initial_evaluation": (
                        None
                        if latent_fit.initial_evaluation is None
                        else latent_fit.initial_evaluation.to_dict()
                    ),
                    "elapsed_seconds": latent_fit.elapsed_seconds,
                    "supervised_history": list(latent_fit.supervised_history),
                    "market_history": list(latent_fit.market_history),
                    "supervised_convergence": latent_fit.supervised_convergence,
                    "market_convergence": latent_fit.market_convergence,
                    "checkpoint_proof": latent_checkpoint_proof(
                        latent_checkpoint_path
                    ),
                }
            )
            report["latent"] = latent_reports
            report["status"] = f"{variant}_complete"
            _write_report(report, output_directory=output_directory, label=label)
            _write_report(
                report,
                output_directory=output_directory,
                label=f"{label}-{variant}",
            )

    report["status"] = "complete"
    _write_report(report, output_directory=output_directory, label=label)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end", default="2025-08-29")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument(
        "--market-dates-report",
        type=Path,
        help=(
            "Use the exact ordered market_dates from a completed JSON report "
            "instead of rebuilding the calendar-date request."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fit-weight", type=float, default=0.05)
    parser.add_argument(
        "--exclude-market-business-dtes",
        type=float,
        nargs="+",
        default=(),
        help=(
            "Exclude market expiries at these exact business-day maturities "
            "before training and evaluation."
        ),
    )
    parser.add_argument(
        "--fixed-eta",
        type=float,
        help=(
            "Fix the backbone volatility multiplier at this value while "
            "retaining the standard raw-state layout."
        ),
    )
    parser.add_argument(
        "--forward-mu",
        action="store_true",
        help=(
            "Use expiry-major native blocks with one learned multiplier on "
            "each forward-volatility contribution to the backbone."
        ),
    )
    parser.add_argument(
        "--shared-strikes",
        action="store_true",
        help=(
            "Use the forward-mu capped-unbounded-DLV geometry with one "
            "daily floating strike support shared by all model expiries."
        ),
    )
    parser.add_argument(
        "--model-business-dtes",
        type=float,
        nargs="+",
        help=(
            "Use these explicit, strictly increasing business-day model "
            "expiries instead of the default equally spaced grid."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--supervised-epochs", type=int, default=300)
    parser.add_argument("--day-batch-size", type=int, default=16)
    parser.add_argument("--supervised-day-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-2)
    parser.add_argument("--supervised-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--compile-sanos-solver", action="store_true")
    parser.add_argument("--just-load", action="store_true")
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "Load matching model weights but deliberately reset optimizer, "
            "scheduler, RNG, epoch, and history."
        ),
    )
    parser.add_argument(
        "--native-weights-warm-start-checkpoint",
        type=Path,
        help=(
            "Initialize compatible learned native parameters from this "
            "checkpoint while retaining the current geometry buffers and "
            "starting a fresh optimizer and history."
        ),
    )
    parser.add_argument(
        "--run-compression",
        action="store_true",
        help=(
            "Load an already completed native checkpoint and run the four "
            "compression experiments. Without this flag only native fitting runs."
        ),
    )
    parser.add_argument(
        "--pca-decoder",
        action="store_true",
        help=(
            "Use an affine decoder initialized analytically by PCA instead "
            "of the hidden-layer decoder. Requires '--run-compression'."
        ),
    )
    parser.add_argument(
        "--expiry-benchmark",
        action="store_true",
        help=(
            "With a forward-mu geometry and '--run-compression', compare "
            "selected global or expiry-causal affine decoders."
        ),
    )
    parser.add_argument(
        "--expiry-benchmark-variants",
        choices=_EXPIRY_BENCHMARK_VARIANTS,
        nargs="+",
        default=list(_DEFAULT_EXPIRY_BENCHMARK_VARIANTS),
        help=(
            "Select expiry benchmark variants. This permits an explicit "
            "run or weights-only continuation of one or more allocations "
            "without retraining the others."
        ),
    )
    parser.add_argument(
        "--certify-native-only",
        action="store_true",
        help=(
            "Verify and bind a finished native report to its checkpoint, then "
            "exit without loading markets or starting compression."
        ),
    )
    parser.add_argument(
        "--state-sizes",
        type=int,
        nargs="+",
        default=list(_DEFAULT_STATE_SIZES),
    )
    parser.add_argument("--label", default="fixed-weekly-batch-descent")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "batch-descent",
    )
    parser.add_argument("--checkpoint-directory", type=Path)
    return parser.parse_args()


def _market_dates_from_report(path: Path) -> tuple[date, ...]:
    """Load and validate one completed report's exact calibration dates."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("market_dates")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Report has no non-empty 'market_dates' list: {path}")
    try:
        dates = tuple(date.fromisoformat(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Report contains invalid market dates: {path}") from error
    if dates != tuple(sorted(set(dates))):
        raise ValueError(
            f"Report market dates must be unique and strictly increasing: {path}"
        )
    return dates


def main() -> None:
    """CLI entry point restricted by invocation to one selected CUDA device."""
    args = _parse_args()
    if args.sample_every <= 0:
        raise ValueError("'sample_every' must be positive")
    if args.workers <= 0:
        raise ValueError("'workers' must be positive")
    dates = tuple(
        get_business_days(
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        )[:: args.sample_every]
    )
    if args.market_dates_report is not None:
        dates = _market_dates_from_report(args.market_dates_report)
    if args.max_dates is not None:
        dates = dates[: args.max_dates]
    training_root = Path(quantdata_env.locations.training.path) / "flex-fixed-weekly"
    checkpoint_directory = args.checkpoint_directory or training_root / args.label
    selected_geometries = sum(
        (
            args.fixed_eta is not None,
            args.forward_mu,
            args.shared_strikes,
        )
    )
    if selected_geometries > 1:
        raise ValueError(
            "'--fixed-eta', '--forward-mu', and '--shared-strikes' are "
            "mutually exclusive"
        )
    geometry_kwargs: dict[str, object] = {}
    if args.model_business_dtes is not None:
        model_business_dtes = tuple(args.model_business_dtes)
        geometry_kwargs = {
            "num_model_expiries": len(model_business_dtes),
            "explicit_model_business_dtes": model_business_dtes,
        }
    geometry = (
        FixedEtaFlexGeometryConfig(fixed_eta=args.fixed_eta, **geometry_kwargs)
        if args.fixed_eta is not None
        else SharedStrikeCappedUnboundedDlvFlexGeometryConfig(**geometry_kwargs)
        if args.shared_strikes
        else ForwardMuFlexGeometryConfig(**geometry_kwargs)
        if args.forward_mu
        else FlexGeometryConfig(**geometry_kwargs)
    )
    market_config = FlexMarketConfig(
        excluded_business_dtes=tuple(args.exclude_market_business_dtes),
    )
    objective_config = FlexObjectiveConfig(fit_weight=args.fit_weight)
    if args.certify_native_only:
        if args.run_compression:
            raise ValueError(
                "'--certify-native-only' and '--run-compression' are mutually "
                "exclusive"
            )
        completion = certify_native_completion(
            ticker=args.ticker,
            dates=dates,
            geometry=geometry,
            market_config=market_config,
            objective_config=objective_config,
            checkpoint_directory=checkpoint_directory,
            output_directory=args.output_directory,
            label=args.label,
        )
        print(
            json.dumps(
                {
                    "status": completion["status"],
                    "native_checkpoint_proof": completion[
                        "native_checkpoint_proof"
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if args.pca_decoder and not args.run_compression:
        raise ValueError("'--pca-decoder' requires '--run-compression'")
    if args.expiry_benchmark and not (
        (args.forward_mu or args.shared_strikes)
        and args.run_compression
        and not args.pca_decoder
    ):
        raise ValueError(
            "'--expiry-benchmark' requires a forward-mu geometry and "
            "'--run-compression' without '--pca-decoder'"
        )
    run_production_gates(
        ticker=args.ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        objective_config=objective_config,
        train_config=FlexTrainConfig(
            num_epochs=args.epochs,
            day_batch_size=args.day_batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            precision=args.precision,
            compile_sanos_solver=args.compile_sanos_solver,
            just_load=args.just_load,
            warm_start=args.warm_start,
        ),
        supervised_train_config=FlexTrainConfig(
            num_epochs=args.supervised_epochs,
            day_batch_size=args.supervised_day_batch_size,
            learning_rate=args.supervised_learning_rate,
            device=args.device,
            precision=args.precision,
            warm_start=args.warm_start,
        ),
        workers=args.workers,
        output_directory=args.output_directory,
        checkpoint_directory=checkpoint_directory,
        label=args.label,
        state_sizes=tuple(args.state_sizes),
        run_compression=args.run_compression,
        pca_decoder=args.pca_decoder,
        expiry_benchmark=args.expiry_benchmark,
        expiry_benchmark_variants=tuple(args.expiry_benchmark_variants),
        native_weights_warm_start_checkpoint=(
            args.native_weights_warm_start_checkpoint
        ),
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_production_gate")
