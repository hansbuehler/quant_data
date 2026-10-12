from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    remove_completed_checkpoint,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.select_generated_budget import (
    _evenly_spaced_dates,
    _parse_epoch_candidates,
    select_epoch_budget,
)


def test_one_standard_error_rule_selects_cheapest_eligible_budget() -> None:
    selected, scores = select_epoch_budget(
        {
            10: (1.08, 1.10, 1.12, 1.10),
            20: (0.99, 1.01, 1.00, 1.00),
            40: (0.92, 1.08, 0.94, 1.06),
        }
    )

    assert selected == 20
    assert tuple(score.epochs for score in scores) == (10, 20, 40)
    assert scores[-1].standard_error > 0.0


def test_epoch_candidates_are_sorted_unique_and_positive() -> None:
    assert _parse_epoch_candidates("40, 10,20,20") == (10, 20, 40)
    with pytest.raises(ValueError, match="positive"):
        _parse_epoch_candidates("0,10")


def test_validation_origins_are_distinct_and_span_range() -> None:
    start = date(2022, 7, 1)
    values = tuple(start + timedelta(days=index) for index in range(20))

    selected = _evenly_spaced_dates(values, 5)

    assert selected[0] == values[0]
    assert selected[-1] == values[-1]
    assert len(set(selected)) == 5


def test_completed_checkpoint_cleanup_preserves_lock(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fit.pt"
    checkpoint.write_bytes(b"completed")
    checkpoint.with_suffix(".pt.tmp").write_bytes(b"stale")

    remove_completed_checkpoint(checkpoint)

    assert not checkpoint.exists()
    assert not checkpoint.with_suffix(".pt.tmp").exists()
    assert checkpoint.with_suffix(".pt.lock").exists()
