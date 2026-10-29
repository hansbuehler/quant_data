"""Regression tests for the SANOS historical-PCA grid controller."""

from __future__ import annotations

import json
from datetime import date, timedelta

from quantdata.hedging import normal_statistical_pca_objective_grid as grid


def test_reconcile_certified_cell_after_previous_runner_failure(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output_root = tmp_path / "output"
    cell_root = output_root / "mean-variance-lambda1000"
    records = cell_root / "run/records"
    records.mkdir(parents=True)
    origins = [(date(2024, 4, 1) + timedelta(days=day)).isoformat() for day in range(44)]
    for origin in origins:
        (records / f"{origin}.json").write_text("{}")
    (cell_root / "run/summary.json").write_text(
        json.dumps(
            {
                "completed_origins": origins,
                "result_count": 440,
                "nonconverged_results": 0,
                "fit_cap_hits": 0,
                "executed_cap_hits": 0,
            }
        )
    )
    (output_root / "GRID_STATUS.json").write_text(
        json.dumps({"mean-variance-lambda1000": {"status": "failed", "exit_code": 1}})
    )
    monkeypatch.setattr(grid, "OBJECTIVE_CELLS", (("mean-variance", 1000),))
    monkeypatch.setattr(grid, "default_public_candidate_path", lambda: dataset)

    def unexpected_runner(*args, **kwargs):
        raise AssertionError("certified results must not be recomputed")

    monkeypatch.setattr(grid.subprocess, "run", unexpected_runner)
    grid.run_grid(dataset=dataset, manifest_sha256="a" * 64, output_root=output_root)

    status = json.loads((output_root / "GRID_STATUS.json").read_text())
    assert status["mean-variance-lambda1000"]["status"] == "complete"
    assert status["mean-variance-lambda1000"]["exit_code"] is None
    assert "previous_failed_exit_code" not in status["mean-variance-lambda1000"]
    assert status["mean-variance-lambda1000"]["result_source"] == "existing_certified_results"


def test_missing_record_prevents_reconciliation(tmp_path) -> None:
    cell_root = tmp_path / "cell"
    records = cell_root / "run/records"
    records.mkdir(parents=True)
    origins = [(date(2024, 4, 1) + timedelta(days=day)).isoformat() for day in range(44)]
    for origin in origins[:-1]:
        (records / f"{origin}.json").write_text("{}")
    (cell_root / "run/summary.json").write_text(
        json.dumps(
            {
                "completed_origins": origins,
                "result_count": 440,
                "nonconverged_results": 0,
                "fit_cap_hits": 0,
                "executed_cap_hits": 0,
            }
        )
    )
    assert not grid.certified_cell(cell_root)
