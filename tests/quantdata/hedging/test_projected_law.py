"""Projected native recursion keeps the full spot/surface dependence."""

from datetime import date, timedelta

import numpy as np

from quantdata.hedge2.dynamics import DynamicsConfig
from quantdata.hedging.model_audit.projected_law import ProjectedLaw
from quantdata.hedging.model_audit.projection_metrics import affine_projection, fit_projected


def test_recursive_first_step_matches_exact_joint_replay_and_nested_snapshots() -> None:
    rng = np.random.default_rng(17)
    levels = rng.normal(size=(61, 10))
    spot = rng.normal(scale=.01, size=61)
    encoder, decoder, _ = affine_projection(levels, np.ones(10), rank=5)
    dates = tuple(date(2023, 1, 1) + timedelta(days=index) for index in range(61))
    fit = fit_projected(levels=levels, dates=dates, spots=spot, sessions=np.arange(61),
        encoder=encoder, decoder=decoder, config=DynamicsConfig(window_transitions=60, rank=3, spot_orthogonal=True))
    centered_spot = spot[1:] - spot[1:].mean()
    beta = centered_spot @ np.diff(levels, axis=0) / (centered_spot @ centered_spot)
    law = ProjectedLaw(dynamics=fit, encoder=encoder, decoder=decoder, spot_loading=beta)
    all_paths = law.recursive_paths(levels[-1], paths=6, steps=5, seed=81)
    selected = np.random.default_rng(81).integers(0, 60, 6)
    np.testing.assert_allclose(all_paths[:, 0], law.next_native(levels[-1])[selected], atol=1e-12)
    snapshots = law.recursive_paths(levels[-1], paths=6, steps=5, seed=81, snapshot_steps=(1, 5))
    np.testing.assert_array_equal(snapshots, all_paths[:, [0, 4]])
    assert np.linalg.norm(law.perpendicular_spot_loading) > 0.
