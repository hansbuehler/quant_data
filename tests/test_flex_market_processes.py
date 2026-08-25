"""Tests for FLEX market-surface process guardrails."""

from __future__ import annotations

import pytest

from quantdata.dysanos.flex.market import market_process_count


def test_market_process_count_accepts_limit() -> None:
    assert market_process_count(4) == 4


def test_market_process_count_caps_legacy_configuration() -> None:
    with pytest.warns(RuntimeWarning, match="Capping FLEX market processes"):
        assert market_process_count(16) == 4


@pytest.mark.parametrize("requested", [0, -1, True])
def test_market_process_count_rejects_invalid_values(requested: int) -> None:
    with pytest.raises(ValueError, match="positive market process count"):
        market_process_count(requested)
