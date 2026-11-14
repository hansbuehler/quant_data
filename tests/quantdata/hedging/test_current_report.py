from quantdata.hedging.model_audit.current_report import legacy_blue
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
