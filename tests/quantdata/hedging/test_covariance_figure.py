"""The Figure 1 extension preserves baselines and requires complete matched dates."""

import json
from pathlib import Path

import pytest

from quantdata.hedging.model_audit.report import _covariance_plot_data


def test_density_extension_is_adjacent_to_its_unchanged_baseline(tmp_path: Path) -> None:
    dates = ["2024-04-01", "2024-04-02"]
    (tmp_path / "observable_reference.json").write_text(json.dumps({"dates": dates}))
    extra = {"origin_dates": dates, "models": [{"model": "ordinary density-weighted",
        "baseline_model": "DYSANOS ordinary", "summary": {"surface_days": 2}}]}
    (tmp_path / "covariance_extension.json").write_text(json.dumps(extra))
    baseline = {"DYSANOS ordinary": {"surface_days": 2}, "FuNVol": {"surface_days": 2}}
    labels, values = _covariance_plot_data(tmp_path, baseline)
    assert labels == ["DYSANOS ordinary", "ordinary density-weighted", "FuNVol"]
    assert values["DYSANOS ordinary"] is baseline["DYSANOS ordinary"]
    assert "ordinary density-weighted" not in baseline


def test_density_extension_rejects_mismatched_origins(tmp_path: Path) -> None:
    (tmp_path / "observable_reference.json").write_text(json.dumps({"dates": ["2024-04-01"]}))
    (tmp_path / "covariance_extension.json").write_text(json.dumps({"origin_dates": ["2024-04-02"], "models": []}))
    with pytest.raises(ValueError, match="same observed dates"):
        _covariance_plot_data(tmp_path, {"DYSANOS ordinary": {"surface_days": 2}})
