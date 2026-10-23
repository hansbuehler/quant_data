"""Daily-radius compounding uses five state transitions per week."""

from datetime import date, timedelta

import numpy as np

from quantdata.dysanos.v11.daily_radius_cap import daily_radius_panels


def test_daily_radius_panels_use_5_10_15_20_sessions() -> None:
    states = np.arange(40, dtype=np.float64)[:, None] * 0.01
    dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(40))
    panels = daily_radius_panels(
        states,
        dates,
        equilibrium_mean=np.zeros(1),
        transition=np.array([[0.9]]),
        innovation_cholesky=np.array([[0.1]]),
        spot_loading=np.array([0.2]),
        training_spot_score_mean=0.0,
    )
    assert [panels[week]["sessions"] for week in (1, 2, 3, 4)] == [5, 10, 15, 20]
    assert panels[1]["pre_evaluation"]["observations"] == 35
    assert panels[4]["pre_evaluation"]["observations"] == 20
    assert panels[4]["forward_evaluation"]["observations"] == 0
