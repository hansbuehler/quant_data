from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.v11.rolling_daily_joint_dynamics import (
    RollingOriginSimulation,
    rolling_origin_indices,
)


def test_rolling_origin_indices_leave_one_oos_step() -> None:
    origins = rolling_origin_indices(10, lookback=4, stride=2)
    np.testing.assert_array_equal(origins, np.asarray((4, 6, 8)))


@pytest.mark.parametrize(
    ("observation_count", "lookback", "stride"),
    ((0, 4, 1), (10, 0, 1), (10, 4, 0), (10, 10, 1)),
)
def test_rolling_origin_indices_reject_invalid_windows(
    observation_count: int,
    lookback: int,
    stride: int,
) -> None:
    with pytest.raises(ValueError):
        rolling_origin_indices(
            observation_count,
            lookback=lookback,
            stride=stride,
        )


def test_rolling_simulation_requires_aligned_antithetic_samples() -> None:
    with pytest.raises(ValueError):
        RollingOriginSimulation(
            next_states=np.ones((3, 2)),
            log_spot_returns=np.ones(3),
            origin_rows=np.arange(3),
            retained_variance_fractions=np.ones(1),
            condition_numbers=np.ones(1),
        )
