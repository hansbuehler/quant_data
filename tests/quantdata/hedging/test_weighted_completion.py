"""Publication distinguishes completed weighted model failures from missing work."""

from pathlib import Path

import pytest

from quantdata.hedging.model_audit.run import _json
from quantdata.hedging.model_audit.weighted_report import _records


def test_weighted_completion_accepts_evaluated_failure_not_runner_error(tmp_path: Path) -> None:
    path = tmp_path / "Weighted/2024-04-01.json"
    row = {"origin_date": "2024-04-01", "status": "model_invalid", "path_count": 32, "steps": 115, "seed": 20_261_111}
    _json(path, row)
    assert _records(tmp_path, "Weighted", ["2024-04-01"], recursive=True)[0]["status"] == "model_invalid"
    row["status"] = "error"
    _json(path, row)
    with pytest.raises(ValueError, match="Incomplete"):
        _records(tmp_path, "Weighted", ["2024-04-01"], recursive=True)


def test_weighted_completion_rejects_different_path_horizon(tmp_path: Path) -> None:
    _json(tmp_path / "Weighted/2024-04-01.json", {"origin_date": "2024-04-01", "status": "scored",
        "path_count": 32, "steps": 128, "seed": 20_261_111})
    with pytest.raises(ValueError, match="horizon"):
        _records(tmp_path, "Weighted", ["2024-04-01"], recursive=True)
