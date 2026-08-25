from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.flex.sample_arbitrage import solve_strict_sample_arbitrage


def test_strict_sample_arbitrage_detects_separating_portfolio() -> None:
    returns = np.asarray(
        [
            [0.01, -0.02],
            [0.02, 0.01],
            [0.03, -0.01],
        ]
    )
    result = solve_strict_sample_arbitrage(returns)
    assert result["strict_sample_arbitrage"]
    action = np.asarray(result["action"])
    assert np.abs(action).sum() == pytest.approx(1.0)
    assert np.min(returns @ action) == pytest.approx(result["strict_margin"])
    assert result["strict_margin"] > 0.01


def test_strict_sample_arbitrage_rejects_two_sided_support() -> None:
    returns = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )
    result = solve_strict_sample_arbitrage(returns)
    assert not result["strict_sample_arbitrage"]
    assert result["strict_margin"] == pytest.approx(0.0, abs=1.0e-12)
