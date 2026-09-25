"""Compare two-eta native and affine-200 fits with the reported baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.training import native_checkpoint_state_audit


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for geometry in ("weekly", "daily"):
        parser.add_argument(f"--{geometry}-native", type=Path, required=True)
        parser.add_argument(f"--{geometry}-affine", type=Path, required=True)
        parser.add_argument(f"--{geometry}-joint", type=Path, required=True)
        parser.add_argument(f"--old-{geometry}-native", type=Path, required=True)
        parser.add_argument(f"--old-{geometry}-affine", type=Path, required=True)
        parser.add_argument(f"--old-{geometry}-joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    status = payload.get("status")
    if not isinstance(status, str) or not (
        status == "complete" or status.endswith("_complete")
    ):
        raise ValueError(f"report is not complete: {path}")
    return payload


def _read_expiry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "flex-market-expiry-fit-v1":
        raise ValueError(f"unexpected market-expiry schema: {path}")
    if payload.get("status") not in {None, "complete"}:
        raise ValueError(f"market-expiry report is not complete: {path}")
    rows = payload.get("by_nominal_market_dte")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"market-expiry report has no DTE rows: {path}")
    if sum(int(row["num_quotes"]) for row in rows) != int(
        payload["all"]["num_quotes"]
    ):
        raise ValueError(f"market-expiry rows do not partition all quotes: {path}")
    return payload


def _joint_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = _read(path)
    report_value = marker.get("report")
    if not isinstance(report_value, str):
        raise ValueError(f"joint completion has no report path: {path}")
    report_path = Path(report_value)
    if not report_path.is_absolute():
        report_path = path.parent / report_path
    return marker, _read(report_path)


def _front_term_geometry(
    payload: dict[str, Any],
) -> FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig:
    values = payload.get("geometry")
    if not isinstance(values, dict) or values.get("front_term_eta") is not True:
        raise ValueError("native report is not the front/term-eta geometry")
    arguments = dict(values)
    for key in ("nms_range", "explicit_model_business_dtes"):
        if arguments.get(key) is not None:
            arguments[key] = tuple(arguments[key])
    return FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(**arguments)


def _two_eta_geometry_identity(
    native_payload: dict[str, Any],
    affine_payload: dict[str, Any],
    joint_payload: dict[str, Any],
) -> FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig:
    """Prove that native, affine, and joint artifacts use one two-eta geometry."""
    native_geometry = _front_term_geometry(native_payload)
    affine_geometry = _front_term_geometry(affine_payload)
    model_config = joint_payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("joint report has no model_config payload")
    joint_geometry_values = model_config.get("geometry")
    if not isinstance(joint_geometry_values, dict):
        raise ValueError("joint report has no geometry payload")
    joint_geometry = _front_term_geometry({"geometry": joint_geometry_values})
    if affine_geometry != native_geometry:
        raise ValueError("native and affine reports use different geometries")
    if joint_geometry != native_geometry:
        raise ValueError("native and joint reports use different geometries")
    if int(model_config.get("state_size", -1)) != 200:
        raise ValueError("joint report is not a 200-state model")
    return native_geometry


def _fit(statistics: dict[str, Any], objective: float) -> dict[str, float | int]:
    all_quotes = statistics["all"]
    return {
        "num_options": int(all_quotes["num_options"]),
        "inside_spread_percent": float(all_quotes["percent_zero"]),
        "surface_mean_half_spreads": float(all_quotes["surface_mean"]),
        "positive_mean_half_spreads": float(all_quotes["positive_mean"]),
        "positive_median_half_spreads": float(all_quotes["positive_median"]),
        "objective": float(objective),
    }


def _native(payload: dict[str, Any]) -> dict[str, float | int]:
    result = payload["native"]
    return _fit(result["statistics"], result["objective"])


def _affine(payload: dict[str, Any]) -> dict[str, float | int]:
    candidates = payload["latent"]
    selected = [item for item in candidates if int(item["state_size"]) == 200]
    if len(selected) != 1:
        raise ValueError("affine report does not contain exactly one 200-state result")
    result = selected[0]
    return _fit(result["statistics"], result["objective"])


def _joint_fit(
    payload: dict[str, Any],
    *,
    period: str,
) -> dict[str, float | int]:
    key = {"all_history": "all_history_fit", "2025": "fit_2025"}.get(period)
    if key is None:
        raise ValueError(f"unsupported joint-fit period {period!r}")
    result = payload.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"joint report has no {key}")
    return _fit(result["statistics"], result["objective"])


def _comparison(
    current: dict[str, float | int],
    baseline: dict[str, float | int],
    *,
    inside_tolerance_points: float,
) -> dict[str, object]:
    inside_change = float(current["inside_spread_percent"]) - float(
        baseline["inside_spread_percent"]
    )
    surface_ratio = float(current["surface_mean_half_spreads"]) / float(
        baseline["surface_mean_half_spreads"]
    )
    positive_ratio = float(current["positive_mean_half_spreads"]) / float(
        baseline["positive_mean_half_spreads"]
    )
    checks = {
        "inside_spread_within_tolerance": abs(inside_change)
        <= inside_tolerance_points,
        "surface_mean_within_20_percent": 0.8 <= surface_ratio <= 1.2,
        "positive_mean_within_20_percent": 0.8 <= positive_ratio <= 1.2,
    }
    return {
        "new": current,
        "reported_baseline": baseline,
        "inside_spread_change_percentage_points": inside_change,
        "surface_mean_ratio": surface_ratio,
        "positive_mean_ratio": positive_ratio,
        "checks": checks,
        "consistent": all(checks.values()),
    }


def _expiry_comparison(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, object]:
    current_rows = {
        int(row["nominal_market_dte"]): row
        for row in current["by_nominal_market_dte"]
    }
    baseline_rows = {
        int(row["nominal_market_dte"]): row
        for row in baseline["by_nominal_market_dte"]
    }
    if current_rows.keys() != baseline_rows.keys():
        raise ValueError("new and baseline market-expiry reports have different DTEs")
    rows: list[dict[str, int | float]] = []
    for nominal_dte in current_rows:
        new = current_rows[nominal_dte]
        old = baseline_rows[nominal_dte]
        if int(new["num_quotes"]) != int(old["num_quotes"]):
            raise ValueError(
                f"quote count changed at nominal market DTE {nominal_dte}"
            )
        rows.append(
            {
                "nominal_market_dte": nominal_dte,
                "num_quotes": int(new["num_quotes"]),
                "old_percent_inside_bid_ask": float(
                    old["percent_inside_bid_ask"]
                ),
                "new_percent_inside_bid_ask": float(
                    new["percent_inside_bid_ask"]
                ),
                "inside_spread_change_percentage_points": float(
                    new["percent_inside_bid_ask"]
                )
                - float(old["percent_inside_bid_ask"]),
                "old_mean_excess": float(old["mean_excess"]),
                "new_mean_excess": float(new["mean_excess"]),
            }
        )
    weights = [int(row["num_quotes"]) for row in rows]
    changes = [
        float(row["inside_spread_change_percentage_points"]) for row in rows
    ]
    weighted_absolute_change = sum(
        weight * abs(change) for weight, change in zip(weights, changes, strict=True)
    ) / sum(weights)
    return {
        "new_source": current.get("source_checkpoint"),
        "reported_baseline_source": baseline.get("source_checkpoint"),
        "same_quote_universe": True,
        "quote_weighted_mean_absolute_inside_change_percentage_points": (
            weighted_absolute_change
        ),
        "maximum_absolute_inside_change_percentage_points": max(
            abs(change) for change in changes
        ),
        "by_nominal_market_dte": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Front/term-eta fit comparison",
        "",
        "| Geometry | Mode | Old inside | New inside | Change (pp) | Old surface mean (half-spreads) | New surface mean (half-spreads) | Consistent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for geometry in ("weekly", "daily"):
        for mode in ("native", "affine_200", "joint_affine_200"):
            item = report["comparisons"][geometry][mode]
            old = item["reported_baseline"]
            new = item["new"]
            lines.append(
                f"| {geometry} | {mode} | {old['inside_spread_percent']:.4f}% "
                f"| {new['inside_spread_percent']:.4f}% "
                f"| {item['inside_spread_change_percentage_points']:+.4f} "
                f"| {old['surface_mean_half_spreads']:.4f} "
                f"| {new['surface_mean_half_spreads']:.4f} "
                f"| {'yes' if item['consistent'] else 'no'} |"
            )
    lines.extend(
        (
            "",
            "Consistency means the inside-spread rate differs by at most one "
            "percentage point for native fits or two points for affine-200, "
            "and both mean fit-error measures remain within 20% of the reported baseline.",
            "",
        )
    )
    lines.extend(
        (
            "## Final joint checkpoint: 2025 slice",
            "",
            "| Geometry | Old inside | New inside | Change (pp) | Old surface mean (half-spreads) | New surface mean (half-spreads) | Consistent |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for geometry in ("weekly", "daily"):
        item = report["joint_2025_comparisons"][geometry]
        old = item["reported_baseline"]
        new = item["new"]
        lines.append(
            f"| {geometry} | {old['inside_spread_percent']:.4f}% "
            f"| {new['inside_spread_percent']:.4f}% "
            f"| {item['inside_spread_change_percentage_points']:+.4f} "
            f"| {old['surface_mean_half_spreads']:.4f} "
            f"| {new['surface_mean_half_spreads']:.4f} "
            f"| {'yes' if item['consistent'] else 'no'} |"
        )
    lines.append("")
    lines.extend(
        (
            "## Short-expiry comparison",
            "",
            "| Geometry | Mode | DTE | Quotes | Old inside | New inside | Change (pp) | Old mean excess | New mean excess |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for geometry in ("weekly", "daily"):
        for mode in ("native", "affine_200"):
            rows = report["expiry_comparisons"][geometry][mode][
                "by_nominal_market_dte"
            ]
            for item in rows:
                if int(item["nominal_market_dte"]) > 5:
                    continue
                lines.append(
                    f"| {geometry} | {mode} | {item['nominal_market_dte']} "
                    f"| {item['num_quotes']} "
                    f"| {item['old_percent_inside_bid_ask']:.4f}% "
                    f"| {item['new_percent_inside_bid_ask']:.4f}% "
                    f"| {item['inside_spread_change_percentage_points']:+.4f} "
                    f"| {item['old_mean_excess']:.4f} "
                    f"| {item['new_mean_excess']:.4f} |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    comparisons: dict[str, object] = {}
    joint_2025_comparisons: dict[str, object] = {}
    expiry_comparisons: dict[str, object] = {}
    identity: dict[str, object] = {}
    for geometry in ("weekly", "daily"):
        current_native_path = getattr(arguments, f"{geometry}_native")
        current_native_payload = _read(current_native_path)
        current_affine_payload = _read(getattr(arguments, f"{geometry}_affine"))
        current_joint_marker, current_joint_payload = _joint_report(
            getattr(arguments, f"{geometry}_joint")
        )
        old_joint_marker, old_joint_payload = _joint_report(
            getattr(arguments, f"old_{geometry}_joint")
        )
        old_native_payload = _read(getattr(arguments, f"old_{geometry}_native"))
        old_affine_payload = _read(getattr(arguments, f"old_{geometry}_affine"))
        current_report_directory = current_native_path.parent
        old_report_directory = getattr(
            arguments, f"old_{geometry}_native"
        ).parent
        expiry_comparisons[geometry] = {
            "native": _expiry_comparison(
                _read_expiry(current_report_directory / "market-expiry-fit.json"),
                _read_expiry(old_report_directory / "market-expiry-fit.json"),
            ),
            "affine_200": _expiry_comparison(
                _read_expiry(
                    current_report_directory / "market-expiry-fit-affine-200.json"
                ),
                _read_expiry(
                    old_report_directory / "market-expiry-fit-affine-200.json"
                ),
            ),
        }
        geometry_config = _two_eta_geometry_identity(
            current_native_payload,
            current_affine_payload,
            current_joint_payload,
        )
        expected_native_size = (
            geometry_config.num_model_expiries
            * (geometry_config.num_inner_strikes + 1)
            + 2
        )
        if int(current_native_payload["native_state_size"]) != expected_native_size:
            raise ValueError(f"{geometry} native state size is inconsistent")
        comparisons[geometry] = {
            "native": _comparison(
                _native(current_native_payload),
                _native(old_native_payload),
                inside_tolerance_points=1.0,
            ),
            "affine_200": _comparison(
                _affine(current_affine_payload),
                _affine(old_affine_payload),
                inside_tolerance_points=2.0,
            ),
            "joint_affine_200": _comparison(
                _joint_fit(current_joint_payload, period="all_history"),
                _joint_fit(old_joint_payload, period="all_history"),
                inside_tolerance_points=2.0,
            ),
        }
        joint_2025_comparisons[geometry] = _comparison(
            _joint_fit(current_joint_payload, period="2025"),
            _joint_fit(old_joint_payload, period="2025"),
            inside_tolerance_points=2.0,
        )
        identity[geometry] = {
            "native_state_size": expected_native_size,
            "eta_coordinates": 2,
            "native_affine_joint_geometry_match": True,
            "affine_state_size": 200,
            "joint_checkpoint": current_joint_marker.get("checkpoint"),
            "joint_checkpoint_sha256": current_joint_marker.get("checkpoint_sha256"),
            "joint_report": current_joint_marker.get("report"),
            "joint_report_sha256": current_joint_marker.get("report_sha256"),
            "reported_baseline_joint_checkpoint": old_joint_marker.get(
                "checkpoint"
            ),
            "reported_baseline_joint_checkpoint_sha256": old_joint_marker.get(
                "checkpoint_sha256"
            ),
            "recomputed_native_state_audit": native_checkpoint_state_audit(
                current_native_path.parent.parent / "checkpoints" / "native.pt",
                geometry=geometry_config,
            ),
            "joint_all_history_fit": current_joint_payload["all_history_fit"],
            "joint_2025_fit": current_joint_payload["fit_2025"],
            "joint_dynamics": current_joint_payload["dynamics_all_history"],
            "affine_training_config": current_affine_payload.get("train_config"),
        }
    report = {
        "schema": "dysanos-front-term-eta-comparison-v1",
        "status": "complete",
        "comparisons": comparisons,
        "joint_2025_comparisons": joint_2025_comparisons,
        "expiry_comparisons": expiry_comparisons,
        "identity_and_joint_fit": identity,
        "all_reported_fits_consistent": all(
            item[mode]["consistent"]
            for item in comparisons.values()
            for mode in ("native", "affine_200", "joint_affine_200")
        )
        and all(item["consistent"] for item in joint_2025_comparisons.values()),
    }
    write_json_atomic(report, arguments.output)
    markdown = arguments.output.with_suffix(".md")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
