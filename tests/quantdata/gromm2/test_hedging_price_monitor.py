from __future__ import annotations

import pytest

from quantdata.gromm2.hedging_price_monitor import selected_price_snapshot


def _report():
    return dict(epoch=3, stop_reason="running", config=dict(turnover_cost_basis="unit_l1"),
        policies={name: dict(best_epoch=epoch) for name, epoch in dict(zero=0, short=2, long=1).items()})


def _history():
    return [dict(epoch=epoch, validation={name: dict(DH=value) for name, value in values.items()})
        for epoch, values in [(0, dict(zero=0.0, short=0.1, long=-0.01)),
            (1, dict(zero=0.01, short=0.08, long=-0.015)),
            (2, dict(zero=0.02, short=0.02, long=-0.01)),
            (3, dict(zero=0.1, short=0.1, long=-0.1))]]


def test_monitor_uses_individually_selected_checkpoints_not_latest_weights():
    snapshot = selected_price_snapshot(_report(), _history(), dict(mean=0.0175))
    assert snapshot["bid"] == pytest.approx(0.015)
    assert snapshot["ask"] == pytest.approx(0.02)
    assert snapshot["q_between_point_quotes"]
    assert not snapshot["crossed_point_quotes"]
    assert snapshot["bid_mc_standard_error"] is None


def test_point_crossing_is_flagged_without_claiming_mc_significance():
    snapshot = selected_price_snapshot(_report(), _history(), dict(mean=0.03))
    assert not snapshot["q_between_point_quotes"]
    assert "not a significance test" in snapshot["interpretation"]


def test_missing_history_for_an_atomically_newer_report_is_retried():
    assert selected_price_snapshot(_report(), _history()[:2], dict(mean=0.0175)) is None
