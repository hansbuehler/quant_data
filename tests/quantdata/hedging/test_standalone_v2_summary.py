"""Validate date matching and the aggregate explained-energy convention."""

import json

import numpy as np
import pytest

from quantdata.hedging.model_audit import standalone_v2_summary


def test_matched_statistics_use_weighted_energy_and_quote_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(standalone_v2_summary, "origins", lambda _: ("2024-04-01", "2024-04-02"))
    run = tmp_path / "run"
    run.mkdir()
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "calibration.json").write_text("{}")
    data = np.array([[2., 1., 1., 1., 1., 1., 1.], [8., 2., 2., 2., 2., 2., 2.]])
    np.savez(run / "straddle-density-per-day.npz", dates=np.array(["2024-04-02", "2024-04-01"]), per_day=data)
    np.savez(run / "quotefit.npz", dates=np.array(["2024-04-01", "2024-04-02"]),
        n=[90, 10], inside=[81, 2], excess_sum=[0., 0.])
    result = standalone_v2_summary.summarize(reference=tmp_path, run=run, calibration=calibration, output=tmp_path / "out")
    assert result["projection"]["shares"][4] == pytest.approx(.7)
    assert result["same_day_quotes"]["inside_rate"] == pytest.approx(.83)
    assert json.loads((tmp_path / "out/STANDALONE_V2_METRICS.json").read_text())["dates"] == ["2024-04-01", "2024-04-02"]


def test_date_alignment_rejects_missing_or_duplicate_dates():
    with pytest.raises(ValueError, match="duplicate"):
        standalone_v2_summary.aligned_indices(np.array(["a", "a"]), ("a",))
    with pytest.raises(ValueError, match="lacks matched"):
        standalone_v2_summary.aligned_indices(np.array(["a"]), ("b",))
