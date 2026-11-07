"""A 2022 history cutoff must apply to training inputs as well as hedge origins."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantdata.hedge2.dynamics import DynamicsConfig
from quantdata.hedge2.rolling import load_dynamics, run_rolling


def test_rolling_training_never_uses_pre_2022_states(tmp_path) -> None:
    dates = tuple(date(2021, 12, 29) + timedelta(days=index) for index in range(13))
    states = np.random.default_rng(7).normal(size=(len(dates), 2))
    source = tmp_path / "source" / "days"
    source.mkdir(parents=True)
    for day, state in zip(dates, states, strict=True):
        np.savez(source / f"{day.isoformat()}.npz", raw_u=state, reconstruction_passed=True)
    observations = tmp_path / "observations.parquet"
    pd.DataFrame({"date": dates, "session_index": np.arange(len(dates)),
                  "normalized_spot_return_into_date": np.linspace(-.01, .01, len(dates))}).to_parquet(observations)
    output = tmp_path / "fit"
    result = run_rolling(daily_directory=source.parent, observations_path=observations,
                         output_directory=output, start=date(2022, 1, 1), end=dates[-1],
                         config=DynamicsConfig(window_transitions=4, rank=1))
    assert result["models"] == 6
    for path in (output / "days").glob("*.npz"):
        assert load_dynamics(path).training_start >= date(2022, 1, 1)
    assert not (output / "days" / "2022-01-04.npz").exists()
    assert (output / "days" / "2022-01-05.npz").exists()
