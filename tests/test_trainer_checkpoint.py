from __future__ import annotations

import pytest
import torch

from quantdata.torch import MetricTacker


def test_metric_tracker_state_roundtrip() -> None:
    tracker = MetricTacker()
    tracker.record(1, 0.5, {"count": 2.0, "mae": 0.1})
    tracker.record(2, 0.25, {"count": 3.0, "mae": 0.05})

    restored = MetricTacker().load_state_dict(tracker.state_dict())

    assert restored["loss"]["epochs"] == [1, 2]
    assert restored["loss"]["values"] == pytest.approx([0.5, 0.25])
    assert restored["mae"]["values"] == pytest.approx([0.1, 0.05])

