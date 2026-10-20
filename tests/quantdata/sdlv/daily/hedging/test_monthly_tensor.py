from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot.hedging.reports.frozen_month import (
    compare_fixed_month,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.monthly_tensor import (
    _best_cases,
    _family,
    _profile,
)


def test_best_case_is_selected_per_model_and_statistic() -> None:
    cases = {
        "Covariance|low-mean": np.asarray([[0.0], [0.0], [9.0]]),
        "Covariance|low-median": np.asarray([[2.0], [2.0], [2.0]]),
        "Historical-SANOS|only": np.asarray([[1.0], [1.0], [1.0]]),
    }
    assert _best_cases(cases, "mean")["Covariance"] == "Covariance|low-median"
    assert _best_cases(cases, "median")["Covariance"] == "Covariance|low-mean"
    assert _best_cases(cases, "mean")["Historical-SANOS"] == "Historical-SANOS|only"


def test_family_and_profile_preserve_configuration_axes() -> None:
    assert _family({"model": "gromm", "measure": "P"}) == "GROMM-P"
    assert _family({"model": "gromm", "measure": "Q"}) == "GROMM-Q"
    assert _profile(np.asarray([[1_000.0], [-3_000.0]]), "mean").tolist() == [2.0]


def test_frozen_month_requires_common_market_cashflows(tmp_path: Path) -> None:
    targets = (
        "digital-put-1d-atm",
        "digital-put-1d-nms-1.00",
        "digital-put-1d-nms-0.75",
        "digital-put-1d-nms-0.50",
        "digital-put-1d-nms-0.25",
        "digital-call-1d-atm",
        "digital-call-1d-nms+0.25",
        "digital-call-1d-nms+0.50",
        "digital-call-1d-nms+0.75",
        "digital-call-1d-nms+1.00",
    )
    sources = {
        "sanos": (0.5, 1.0),
        "bs": (0.25, 1.0),
        "fixed": (0.1, 1.0),
    }
    for name, (hedged_pnl, unhedged_pnl) in sources.items():
        directory = tmp_path / name / "records"
        directory.mkdir(parents=True)
        for day in ("2024-05-01", "2024-05-02"):
            (directory / f"{day}.json").write_text(
                json.dumps({
                    "results": [
                        {
                            "claim": target,
                            "converged": True,
                            "claim_quantity": 1.0,
                            "hedged_pnl": hedged_pnl,
                            "unhedged_pnl": unhedged_pnl,
                        }
                        for target in targets
                    ]
                }),
                encoding="utf-8",
            )
    common = dict(
        sanos_lv=tmp_path / "sanos",
        bs_greeks=tmp_path / "bs",
        cases=[("Fixed", "April-selected", tmp_path / "fixed")],
        start=date(2024, 5, 1),
        end=date(2024, 5, 2),
        expected_dates=2,
        output=tmp_path / "report",
    )
    result = compare_fixed_month(**common)
    assert result["models"]["Fixed"]["mean_absolute_pnl_usd"] == 100_000.0
    assert result["models"]["Fixed"]["mean_absolute_improvement_vs_bs_usd"] == 150_000.0
    assert result["models"]["Fixed"]["targets_better_than_bs"] == 10
    assert (tmp_path / "report" / "FROZEN_MONTH.pdf").is_file()

    record = tmp_path / "fixed" / "records" / "2024-05-02.json"
    changed = json.loads(record.read_text(encoding="utf-8"))
    changed["results"][0]["unhedged_pnl"] = 2.0
    record.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different market cashflows"):
        compare_fixed_month(**common)
