"""Completion gates for the selected-model report and deliberate retained artifacts."""

from pathlib import Path

import pandas as pd
import pytest

from quantdata.hedging.model_audit.finalize import FINAL_MODELS, coverage_state, retain_deck
from quantdata.hedging.model_audit.run import _json


def test_completion_counts_model_invalid_not_runner_errors(tmp_path: Path) -> None:
    _json(tmp_path / "observable_reference.json", {"dates": ["2024-04-01", "2024-04-02"]})
    _json(tmp_path / "persistence/Example/2024-04-01.json", {"status": "model_invalid"})
    _json(tmp_path / "persistence/Example/2024-04-02.json", {"status": "error"})
    _json(tmp_path / "persistence/Example/2025-01-01.json", {"status": "scored"})
    assert coverage_state(tmp_path, ("Example",)) == {
        "Example": {"daily": 0, "quote_forecasts": 0, "persistence": 1}}


def test_final_deck_requires_complete_matching_horizons(tmp_path: Path) -> None:
    rows = [{"model": model, "surface_origins": 44, "quote_forecast_origins": 44, "recursive_origins": 44,
             "recursive_min_sessions": 115, "recursive_max_sessions": 128} for model in FINAL_MODELS]
    pd.DataFrame(rows).to_csv(tmp_path / "coverage.csv", index=False)
    with pytest.raises(ValueError, match="identical path horizons"):
        retain_deck(tmp_path, tmp_path / "retained")
    rows[0]["surface_origins"] = 43
    pd.DataFrame(rows).to_csv(tmp_path / "coverage.csv", index=False)
    with pytest.raises(ValueError, match="missing origin"):
        retain_deck(tmp_path, tmp_path / "retained")


def test_executive_summary_uses_selected_sixteen_factor_models(monkeypatch) -> None:
    from quantdata.hedging.model_audit import report

    monkeypatch.setattr(report, "MODEL_ORDER", FINAL_MODELS)
    summaries = {FINAL_MODELS[0]: {"pooled_covariance": {"trace_ratio": .166}},
        FINAL_MODELS[1]: {"pooled_covariance": {"trace_ratio": .214}},
        "VolGAN PCA8": {"pooled_covariance": {"trace_ratio": .97}, "conditional_covariance": {"trace_ratio": .00025}}}
    coverage = pd.DataFrame([{"requested_origins": 44, "surface_origins": 44,
                             "quote_forecast_origins": 44, "recursive_origins": 44}])
    stability = pd.DataFrame({"model": list(FINAL_MODELS[:2]), "stationary_backbone": [True, True],
                              "spectral_radius": [.735, .73]})
    rendered = report._executive_summary(summaries, coverage, stability)
    assert "16.6" in rendered and "21.4" in rendered
    assert "All requested origin-level diagnostics are complete" in rendered


@pytest.mark.parametrize(("branch", "actual_branch", "actual_head", "message"), [
    ("main", "main", "expected", "never main"),
    ("hedging-benchmark", "hedging-benchmark", "unexpected", "checkout changed"),
])
def test_publisher_rejects_main_or_concurrent_history_changes(tmp_path: Path, monkeypatch,
        branch: str, actual_branch: str, actual_head: str, message: str) -> None:
    from quantdata.hedging.model_audit import publish as publisher

    _json(tmp_path / "READY_TO_PUBLISH.json", {"complete": True, "retained_directory": str(tmp_path / "retained")})

    def git_answer(repository, *arguments):
        return actual_branch if "--show-current" in arguments else actual_head

    monkeypatch.setattr(publisher, "_command", git_answer)
    with pytest.raises(ValueError, match=message):
        publisher.publish(tmp_path, tmp_path, branch=branch, expected_head="expected")
