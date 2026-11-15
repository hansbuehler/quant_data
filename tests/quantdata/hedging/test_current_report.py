import math

import pytest

from quantdata.hedging.model_audit.current_report import legacy_blue, literature_static_summary
from quantdata.hedging.model_audit.report import Rated, _format, _table


def test_legacy_body_and_prior_ratings_are_blue_without_changing_values():
    original = (r"\documentclass{article}\begin{document}\maketitle"
        r"44 dates: \textcolor{red}{0.8145}; $259$ days."
        r"\includegraphics[width=\linewidth]{covariance.pdf}\end{document}")
    updated = legacy_blue(original)
    assert r"\begingroup\color{blue}" in updated
    assert r"\textcolor{blue}{0.8145}" in updated
    assert r"44 dates" in updated and r"$259$" in updated
    assert "legacy/covariance.pdf" in updated
    assert "maketitle" not in updated and r"\textcolor{red}" not in updated


def test_historical_numeric_cells_override_quality_color_in_current_table():
    assert _format(Rated(.8145, "old")) == r"\textcolor{blue}{0.815}"
    assert _format(Rated(44, "old")) == r"\textcolor{blue}{44}"


def test_tail_and_historical_captions_match_their_color_rules():
    tail = _table(["PC1"], [[Rated(4, "fail")]], caption="Tail diagnostics.", label="tab:standalone-v2-tails")
    assert "red for exact-binomial p below 0.01" in tail
    assert "No orange or red grades" not in tail
    assert "<" not in tail
    historical = _table(["Count"], [[Rated(0, "old")]], caption="Earlier results.", label="tab:current-static")
    assert "Blue: retained historical counts; no current grading." in historical


def test_literature_rates_use_eligible_tests_and_finite_prices():
    checks = []
    for day in (1, 5, 10, 125, 259):
        for tenor in (1., 2., 5., 10.):
            checks.append({"simulation_day": day, "business_dte": tenor, "paths": 10, "strikes": 21,
                "invalid_price_nodes": 21, "convexity_violations": 2, "convexity_tests": 100,
                "calendar_violations": None, "calendar_tests": None,
                "inversion": [{"count": 189, "library_min_vol_cases": 9,
                    "library_max_vol_cases": 18, "failed": 19}]})
    result = literature_static_summary([{"origin_date": "2024-04-01", "status": "scored", "results": checks}], ["2024-04-01"])
    first = result.iloc[0]
    assert first.convexity_pct == 2.
    assert first.unavailable_pct == 10.
    assert first.iv_library_max_vol_cases_pct == pytest.approx(100. * 18 / 189)
    assert first.iv_failed_pct == pytest.approx(100. * 19 / 189)
    assert math.isnan(first.calendar_pct)
    with pytest.raises(ValueError, match="approved origins"):
        literature_static_summary([], ["2024-04-01"])
