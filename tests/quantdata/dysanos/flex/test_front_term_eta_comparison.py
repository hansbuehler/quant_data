"""Tests for the front/term-eta fit comparison report."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.run_front_term_eta_comparison import (
    _comparison,
    _expiry_comparison,
    _joint_fit,
    _joint_report,
    _read,
    _two_eta_geometry_identity,
)


@pytest.mark.parametrize("status", ["complete", "native_complete", "pca_200_complete"])
def test_read_accepts_completion_statuses(tmp_path, status: str) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": status}), encoding="utf-8")

    assert _read(path)["status"] == status


def test_read_rejects_incomplete_report(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": "running"}), encoding="utf-8")

    with pytest.raises(ValueError, match="not complete"):
        _read(path)


def test_fit_comparison_applies_declared_tolerances() -> None:
    baseline = {
        "inside_spread_percent": 80.0,
        "surface_mean_half_spreads": 0.5,
        "positive_mean_half_spreads": 2.0,
    }
    current = {
        "inside_spread_percent": 79.2,
        "surface_mean_half_spreads": 0.55,
        "positive_mean_half_spreads": 1.8,
    }

    result = _comparison(current, baseline, inside_tolerance_points=1.0)

    assert result["consistent"] is True
    assert result["inside_spread_change_percentage_points"] == pytest.approx(-0.8)


def test_joint_completion_dereferences_detailed_report(tmp_path) -> None:
    report = tmp_path / "joint.json"
    marker = tmp_path / "joint-complete.json"
    report.write_text(
        json.dumps({"status": "complete", "all_history_fit": {"quotes": 12}}),
        encoding="utf-8",
    )
    marker.write_text(
        json.dumps({"status": "complete", "report": str(report)}),
        encoding="utf-8",
    )

    marker_payload, report_payload = _joint_report(marker)

    assert marker_payload["report"] == str(report)
    assert report_payload["all_history_fit"]["quotes"] == 12


def test_joint_fit_extracts_all_history_and_2025_statistics() -> None:
    statistics = {
        "all": {
            "num_options": 100,
            "percent_zero": 80.0,
            "surface_mean": 0.4,
            "positive_mean": 2.0,
            "positive_median": 0.7,
        }
    }
    payload = {
        "all_history_fit": {"objective": 0.5, "statistics": statistics},
        "fit_2025": {"objective": 0.6, "statistics": statistics},
    }

    assert _joint_fit(payload, period="all_history")["objective"] == 0.5
    assert _joint_fit(payload, period="2025")["objective"] == 0.6
    with pytest.raises(ValueError, match="unsupported"):
        _joint_fit(payload, period="pre_2025")


def test_expiry_comparison_requires_and_summarizes_same_quote_universe() -> None:
    baseline = {
        "by_nominal_market_dte": [
            {
                "nominal_market_dte": 1,
                "num_quotes": 100,
                "percent_inside_bid_ask": 80.0,
                "mean_excess": 0.5,
            },
            {
                "nominal_market_dte": 2,
                "num_quotes": 300,
                "percent_inside_bid_ask": 90.0,
                "mean_excess": 0.2,
            },
        ]
    }
    current = {
        "by_nominal_market_dte": [
            {
                "nominal_market_dte": 1,
                "num_quotes": 100,
                "percent_inside_bid_ask": 82.0,
                "mean_excess": 0.4,
            },
            {
                "nominal_market_dte": 2,
                "num_quotes": 300,
                "percent_inside_bid_ask": 89.0,
                "mean_excess": 0.22,
            },
        ]
    }

    result = _expiry_comparison(current, baseline)

    assert result["same_quote_universe"] is True
    assert result[
        "quote_weighted_mean_absolute_inside_change_percentage_points"
    ] == pytest.approx(1.25)
    assert result["maximum_absolute_inside_change_percentage_points"] == 2.0


def test_expiry_comparison_rejects_changed_quote_count() -> None:
    baseline = {
        "by_nominal_market_dte": [
            {
                "nominal_market_dte": 1,
                "num_quotes": 100,
                "percent_inside_bid_ask": 80.0,
                "mean_excess": 0.5,
            }
        ]
    }
    current = {
        "by_nominal_market_dte": [
            {
                "nominal_market_dte": 1,
                "num_quotes": 99,
                "percent_inside_bid_ask": 80.0,
                "mean_excess": 0.5,
            }
        ]
    }

    with pytest.raises(ValueError, match="quote count changed"):
        _expiry_comparison(current, baseline)


def test_two_eta_identity_requires_matching_native_affine_and_joint_geometry() -> None:
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    payload = {"geometry": asdict(geometry)}
    joint = {
        "model_config": {
            "geometry": asdict(geometry),
            "state_size": 200,
        }
    }

    restored = _two_eta_geometry_identity(payload, payload, joint)

    assert restored == geometry
    assert restored.num_eta_coordinates == 2


def test_two_eta_identity_rejects_a_changed_joint_geometry() -> None:
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    changed = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        min_eta=0.02
    )
    payload = {"geometry": asdict(geometry)}
    joint = {
        "model_config": {
            "geometry": asdict(changed),
            "state_size": 200,
        }
    }

    with pytest.raises(ValueError, match="native and joint"):
        _two_eta_geometry_identity(payload, payload, joint)
