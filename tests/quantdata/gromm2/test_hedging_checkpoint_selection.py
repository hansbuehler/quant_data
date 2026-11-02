from __future__ import annotations

import copy

import pytest

from quantdata.gromm2.plot_hedging_training import checkpoint_selection_comparison


def _inputs():
    report = {"epoch": 2, "policies": {name: {"best_epoch": 0} for name in ("zero", "short", "long")}}
    objectives = {"zero": [0.0, 0.1, 0.2], "short": [0.1, 0.02, 0.03], "long": [-0.01, -0.02, -0.03]}
    rows = [{"epoch": epoch, "train": {name: {"objective": values[epoch], "DH": values[epoch]}
        for name, values in objectives.items()}, "validation": {name: {"DH": values[epoch] + 0.01 * epoch}
        for name, values in objectives.items()}} for epoch in range(3)]
    return report, rows


def test_training_selection_is_independent_per_policy_and_ignores_validation():
    report, rows = _inputs()
    result = checkpoint_selection_comparison(report, rows)
    assert result["training"]["best_epochs"] == {"zero": 0, "short": 1, "long": 2}
    assert result["training"]["train"]["bid"] == pytest.approx(0.03)
    assert result["training"]["train"]["ask"] == pytest.approx(0.02)
    assert result["training"]["validation"]["bid"] == pytest.approx(0.01)
    assert result["training"]["validation"]["ask"] == pytest.approx(0.03)
    changed = copy.deepcopy(rows)
    for row in changed:
        for value in row["validation"].values():
            value["DH"] = 123.0
    assert checkpoint_selection_comparison(report, changed)["training"]["best_epochs"] == result["training"]["best_epochs"]


def test_selection_uses_training_objective_not_reoptimized_dh():
    report, rows = _inputs()
    rows[2]["train"]["short"]["DH"] = -10.0
    assert checkpoint_selection_comparison(report, rows)["training"]["best_epochs"]["short"] == 1


def test_existing_validation_selection_is_preserved():
    report, rows = _inputs()
    result = checkpoint_selection_comparison(report, rows)
    assert result["validation"]["best_epochs"] == {"zero": 0, "short": 0, "long": 0}
    assert result["validation"]["validation"]["bid"] == pytest.approx(0.01)
    assert result["validation"]["validation"]["ask"] == pytest.approx(0.1)
    assert result["mc_standard_errors"] is None


def test_exact_training_ties_choose_earliest_epoch_even_if_rows_are_reversed():
    report, rows = _inputs()
    rows[2]["train"]["short"]["objective"] = rows[1]["train"]["short"]["objective"]
    assert checkpoint_selection_comparison(report, list(reversed(rows)))["training"]["best_epochs"]["short"] == 1


def test_future_epochs_are_not_used_before_their_report_is_saved():
    report, rows = _inputs()
    future = copy.deepcopy(rows[-1])
    future["epoch"] = 3
    future["train"]["short"]["objective"] = -99.0
    assert checkpoint_selection_comparison(report, rows + [future])["training"]["best_epochs"]["short"] == 1


@pytest.mark.parametrize("failure", ["empty", "duplicate", "incomplete", "missing_selected", "nonfinite"])
def test_invalid_saved_histories_are_rejected(failure):
    report, rows = _inputs()
    if failure == "empty":
        rows = []
    elif failure == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif failure == "incomplete":
        rows = rows[:-1]
    elif failure == "missing_selected":
        report["policies"]["zero"]["best_epoch"] = 99
    else:
        rows[1]["train"]["zero"]["objective"] = float("nan")
    with pytest.raises(ValueError):
        checkpoint_selection_comparison(report, rows)
