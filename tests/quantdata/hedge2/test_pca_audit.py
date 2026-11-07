"""Guard-attributed PCA score energies must retain their cross term."""

import json
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics
from quantdata.hedge2.pca_audit import audit_pca
from quantdata.hedge2.rolling import save_dynamics


def test_guard_audit_keeps_pca_and_reconciles_factor_energy(tmp_path) -> None:
    config = DlvConfig(num_inner_strikes=10)
    daily = tmp_path / "daily"
    (daily / "days").mkdir(parents=True)
    (daily / "config.json").write_text(json.dumps(asdict(config)))
    dates = tuple(date(2022, 1, 3) + timedelta(days=index) for index in range(20))
    rng = np.random.default_rng(67)
    states = rng.normal(size=(len(dates), 99))
    spot = rng.normal(scale=.01, size=len(dates))
    for index, day in enumerate(dates):
        identified = np.ones((9, 10), dtype=bool)
        identified[:, 0] = index % 2 == 0
        np.savez(daily / "days" / f"{day.isoformat()}.npz", reconstruction_passed=True,
                 raw_dlvsqs=np.full((9, 10), .04), dlvsqs=np.full((9, 10), .04),
                 densities=np.full((9, 12), 1. / 12.), identified=identified,
                 source_first_business_dte=1., expiries_y=config.expiries_y, raw_u=states[index])
    observations = tmp_path / "observations.parquet"
    pd.DataFrame({"date": dates, "session_index": np.arange(len(dates)),
                  "normalized_spot_return_into_date": spot}).to_parquet(observations)
    dynamics = tmp_path / "dynamics"
    dynamics.mkdir()
    dynamics_config = DynamicsConfig(window_transitions=15, rank=2)
    fitted = fit_dynamics(dates=dates, states=states, spot_returns_into_date=spot,
                          session_indices=np.arange(len(dates)), as_of_date=dates[-1], config=dynamics_config)
    save_dynamics(fitted, dynamics / "days" / f"{dates[-1].isoformat()}.npz")
    (dynamics / "config.json").write_text(json.dumps({**asdict(dynamics_config), "history_start": "2022-01-01",
                                                    "requested_end": "2025-12-31", "observations": str(observations)}))
    summary = audit_pca(daily_directory=daily, dynamics_directory=dynamics, output_directory=tmp_path / "audit")
    assert summary["windows"] == 1
    assert summary["pca_is_unchanged"]
    factors = pd.read_parquet(tmp_path / "audit" / "factors.parquet")
    np.testing.assert_allclose(factors.guarded_score_energy_ratio + factors.other_score_energy_ratio
                               + factors.cross_score_energy_ratio, 1., atol=1e-14)
    assert summary["median_guarded_transition_fraction"] == pytest.approx(9. / 99.)
