"""OCE signs, same-epoch Empty subtraction, and descriptive convergence trends."""

import numpy as np
import pytest

from quantdata.sdlv.dhgromm1.oce_history_plot import rolling_mean, tail_progress, transform_history


def history():
    return [dict(epoch=i, empty_loss=e, long_loss=l, short_loss=s, sum_loss=e+l+s)
            for i, (e,l,s) in enumerate([(-.01,.02,-.03),(-.02,.01,-.04),(-.025,.005,-.045)])]


def test_oce_signs_and_same_epoch_empty_subtraction():
    curves = transform_history(history())
    np.testing.assert_allclose(curves["empty"], [.01,.02,.025])
    np.testing.assert_allclose(curves["long_minus_empty"], [-.03,-.03,-.03])
    np.testing.assert_allclose(curves["short_minus_empty"], [.02,.02,.02])
    np.testing.assert_allclose(curves["sum_oce"], [.02,.05,.065])


def test_running_best_progress_does_not_treat_epoch_cap_as_convergence():
    report = tail_progress(transform_history(history()), window=1)
    assert report["through_epoch"]==2 and report["tail_start_epoch"]==1
    assert report["metrics"]["sum_oce"]["best_epoch"]==2
    assert report["metrics"]["sum_oce"]["improvement_in_tail"]==pytest.approx(.015)
    assert "converged" not in report


def test_rolling_mean_is_not_a_monte_carlo_error_estimate():
    values,width = rolling_mean(np.array([1.,3.,5.]), width=2)
    np.testing.assert_array_equal(values, [2.,4.])
    assert width==2


@pytest.mark.parametrize("kind", ["gap", "nan", "sum"])
def test_invalid_epoch_history_is_rejected(kind):
    rows = history()
    if kind=="gap":
        rows[1]["epoch"] = 4
    elif kind=="nan":
        rows[1]["long_loss"] = float("nan")
    else:
        rows[1]["sum_loss"] += 1
    with pytest.raises(ValueError):
        transform_history(rows)
