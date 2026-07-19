from __future__ import annotations

import pytest

from openqai.core.nn.metric_tracker import MetricTracker

def test_metric_tracker_state_roundtrip() -> None:
    tracker = MetricTracker()
    tracker.record(1, {"loss": 0.5, "count": 2.0, "mae": 0.1})
    tracker.record(2, {"loss": 0.25, "count": 3.0, "mae": 0.05})

    restored = MetricTracker().load_state_dict(tracker.state_dict())

    assert restored["loss"]["epochs"] == [1, 2]
    assert restored["loss"]["values"] == pytest.approx([0.5, 0.25])
    assert restored["mae"]["values"] == pytest.approx([0.1, 0.05])
