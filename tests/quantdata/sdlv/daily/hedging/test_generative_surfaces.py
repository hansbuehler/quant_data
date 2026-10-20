from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
    build_sanos_fixed_grid_history,
    fixed_grid_sanos_day,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)

PUBLIC_CANDIDATE = Path(
    "/home/hans/quantdata_data/datasets/hedging/"
    "spx-2022-2025-hedging-public-candidate-v1"
)


def test_fixed_grid_sanos_day_repairs_only_noninterior_tail_nodes() -> None:
    panel = PublicHedgingPanel(PUBLIC_CANDIDATE)
    grid = GenerativeSurfaceGrid()
    state = fixed_grid_sanos_day(panel.market_day(date(2023, 1, 17)), grid=grid)

    assert state.log_implied_volatilities.shape == (
        len(grid.expiries_y),
        len(grid.strikes_over_forward),
    )
    assert np.all(np.isfinite(state.log_implied_volatilities))
    assert np.all(state.implied_volatilities > 0.0)
    assert 0 < np.count_nonzero(state.implied_vol_is_interior) < state.implied_vol_is_interior.size


def test_short_expiry_recovers_interior_nodes_from_sanos_geometry() -> None:
    panel = PublicHedgingPanel(PUBLIC_CANDIDATE)
    state = fixed_grid_sanos_day(panel.market_day(date(2023, 6, 20)))

    assert np.count_nonzero(state.implied_vol_is_interior[0]) == 1
    assert np.all(np.isfinite(state.log_implied_volatilities[0]))
    assert state.implied_volatilities[0, 4] > state.implied_volatilities[0, 5]


def test_fixed_grid_history_uses_only_trailing_public_dates() -> None:
    panel = PublicHedgingPanel(PUBLIC_CANDIDATE)
    history = build_sanos_fixed_grid_history(
        panel,
        end_date=date(2023, 1, 17),
        observation_count=8,
    )

    assert history.dates[-1] == date(2023, 1, 17)
    assert history.log_implied_volatilities.shape == (8, 9, 11)
    assert np.isnan(history.forward_log_returns_into_date[0])
    assert np.all(np.isfinite(history.forward_log_returns_into_date[1:]))
    assert np.array_equal(history.linked_into_date, np.asarray([False, True, True, True, True, True, True, True]))
