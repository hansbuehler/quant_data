"""Regression checks for strict DYSANOS 1.1 temporal split helpers."""

from datetime import date

import pytest

from quantdata.dysanos.v11.run_strict_pre2025_direct import _train_count


def test_train_count_accepts_contiguous_2025_suffix() -> None:
    assert _train_count(
        (date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 2))
    ) == 2


def test_train_count_rejects_noncontiguous_suffix() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _train_count((date(2024, 12, 31), date(2025, 1, 2), date(2024, 12, 30)))
