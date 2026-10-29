from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.reports.all_generator_objective_summary import (
    _format_cap_hits,
    _format_cell,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.objective_grid_summary import (
    GENERATORS,
    _configuration_directory,
    _load_cap_limited_records,
    _statistics,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.sanos_objective_grid_summary import (
    _configuration_directory as _sanos_configuration_directory,
)


def test_objective_grid_statistics_use_one_million_dollar_payout() -> None:
    books = {
        "2024-04-01": {
            "digital-call-1d-atm": {
                "claim_quantity": 0.1,
                "hedged_pnl": 0.01,
                "unhedged_pnl": 0.02,
                "transaction_cost": 0.001,
            }
        },
        "2024-04-02": {
            "digital-call-1d-atm": {
                "claim_quantity": 0.1,
                "hedged_pnl": -0.01,
                "unhedged_pnl": -0.02,
                "transaction_cost": 0.002,
            }
        },
    }

    statistics = _statistics(
        books,
        dates=("2024-04-01", "2024-04-02"),
        targets=("digital-call-1d-atm",),
    )

    assert statistics["mean_pnl_kusd"] == pytest.approx(0.0)
    assert statistics["sd_pnl_kusd"] == pytest.approx(141.421356237)
    assert statistics["mean_absolute_pnl_kusd"] == pytest.approx(100.0)
    assert statistics["mean_entry_cost_kusd"] == pytest.approx(15.0)


def test_infinite_tail_objectives_share_one_scenario_run(tmp_path: Path) -> None:
    expected = tmp_path / "volgan" / "cvar-lambdainf"

    for objective in ("cvar", "entropy", "power-exponential"):
        assert _configuration_directory(tmp_path, "volgan", objective, "inf") == expected

    assert _configuration_directory(
        tmp_path, "volgan", "mean-variance", "inf"
    ) == tmp_path / "volgan" / "mean-variance-lambdainf"


def test_generated_grid_includes_every_available_model_variant() -> None:
    assert GENERATORS == (
        "funvol",
        "volgan",
        "gromm-P",
        "gromm-Q",
        "diffusion-base",
        "diffusion-ft",
    )


def test_combined_grid_never_reports_partial_period_statistics() -> None:
    assert _format_cell({"status": "incomplete", "failure_dates": []}, "sd_pnl_kusd") == "pending"
    assert _format_cell({"status": "incomplete", "failure_dates": ["2024-04-01"]}, "sd_pnl_kusd") == "failed"
    assert _format_cell({"status": "complete", "failure_dates": [], "sd_pnl_kusd": 12.345}, "sd_pnl_kusd") == "12.35"
    assert _format_cell({"status": "cap-limited", "sd_pnl_kusd": 12.345}, "sd_pnl_kusd") == "12.35†"
    assert _format_cap_hits(
        {"status": "cap-limited", "cap_hit_count": 439, "cap_hit_percentage": 99.7727}
    ) == "439/440 (99.8%)"


def test_historical_sanos_uses_shared_worst_case_limit(tmp_path: Path) -> None:
    for objective in ("cvar-oce", "entropy", "power-exponential"):
        assert _sanos_configuration_directory(tmp_path, objective, "inf") == (
            tmp_path / "historical-sanos" / "cvar-oce-lambdainf"
        )
    assert _sanos_configuration_directory(tmp_path, "mean-variance", "100") == (
        tmp_path / "historical-sanos" / "mean-variance-lambda100.0"
    )


def test_cap_limited_records_require_cap_as_the_only_convergence_failure(tmp_path: Path) -> None:
    failures = tmp_path / "failures"
    failures.mkdir()
    results = [
        {
            "claim": f"digital-{index}",
            "origin_date": "2024-04-01",
            "converged": index != 0,
            "cap_hits": int(index == 0),
        }
        for index in range(10)
    ]
    path = failures / "2024-04-01.json"
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    assert len(_load_cap_limited_records(tmp_path)) == 1

    results[0]["cap_hits"] = 0
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    assert not _load_cap_limited_records(tmp_path)
