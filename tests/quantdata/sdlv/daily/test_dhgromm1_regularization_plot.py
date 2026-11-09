"""Plot the same raw utilities in both phases, never a penalty disguised as OCE."""

import numpy as np
import pytest

from quantdata.sdlv.dhgromm1.regularization_plot import history_curves, plot_raw, report_point


def rows(*, penalized=False):
    result = []
    for epoch in range(3):
        raw = np.array([.01,.03,-.02])+epoch*.001
        penalty = np.array([.002,.004,.008]) if penalized else np.zeros(3)
        loss = -raw+penalty
        row = dict(epoch=epoch,sum_loss=loss.sum())
        for i,label in enumerate(("empty","long","short")):
            row[f"{label}_loss"] = loss[i]
            if penalized:
                row[f"{label}_raw_oce"] = raw[i]
                row[f"{label}_regularizer"] = penalty[i]
        result.append(row)
    return result


def test_raw_oce_is_identical_despite_different_optimized_penalties():
    baseline = history_curves(rows(),penalized=False)
    penalized = history_curves(rows(penalized=True),penalized=True)
    for name in ("empty","long_minus_empty","short_minus_empty","raw_sum"):
        np.testing.assert_allclose(baseline[name],penalized[name])
    np.testing.assert_allclose(penalized["objective_sum"],baseline["objective_sum"]-.014)
    np.testing.assert_allclose(penalized["long_minus_empty"],.02)
    np.testing.assert_allclose(penalized["short_minus_empty"],-.03)


@pytest.mark.parametrize("bad", ["gap","nan","wrong_sum","wrong_penalty"])
def test_inconsistent_histories_are_rejected(bad):
    records = rows(penalized=True)
    if bad=="gap":
        records[1]["epoch"] = 8
    elif bad=="nan":
        records[1]["long_raw_oce"] = float("nan")
    elif bad=="wrong_sum":
        records[1]["sum_loss"] += 1.
    else:
        records[1]["long_regularizer"] += 1.
    with pytest.raises(ValueError):
        history_curves(records,penalized=True)


def test_mc_marker_uses_paired_report_difference_and_three_se():
    report = dict(statistics=dict(empty=dict(raw_oce=dict(oce=.01,three_standard_errors=.001))),
        raw_long_minus_empty=dict(oce=-.02,three_standard_errors=.004),
        raw_short_minus_empty=dict(oce=-.03,three_standard_errors=.005))
    assert report_point(report,"short_minus_empty") == dict(oce=-.03,three_standard_errors=.005)
    assert report_point(report,"empty")["three_standard_errors"] == .001
    report["raw_long_minus_empty"]["three_standard_errors"] = -.1
    with pytest.raises(ValueError):
        report_point(report,"long_minus_empty")


def test_pending_penalized_run_can_be_rendered_without_inventing_history(tmp_path):
    case = dict(label="P without a₁₀",baseline=dict(rows=rows()),penalized=None,reports={})
    target = plot_raw([case],output=tmp_path,measure="P")
    assert target.exists() and target.stat().st_size>1000
