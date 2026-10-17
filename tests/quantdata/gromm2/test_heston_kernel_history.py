from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2.heston_kernel_history import initial_variance_grid


def test_initial_variance_grid_spans_historical_states() -> None:
    variances = np.asarray([0.04, 0.01, 0.09, 0.02])
    grid = initial_variance_grid(variances, point_count=5)
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.09)
    assert np.all(np.diff(grid) > 0.0)
