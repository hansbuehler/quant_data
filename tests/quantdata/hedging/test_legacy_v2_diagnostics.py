"""The diagnostic must preserve the completed tail law and full spot loading."""

from types import SimpleNamespace

import numpy as np

from quantdata.hedging.model_audit.legacy_v2_diagnostics import native_snapshots
from quantdata.hedging.model_audit.projected_law import ProjectedLaw
from quantdata.hedging.model_audit.tail import TailSampler


def fixture():
    decoder = np.array([[1., 0.], [0., 1.], [0., 0.]])
    beta = np.array([.1, .2, .4])
    fit = SimpleNamespace(state_mean=np.zeros(2), directions=np.eye(2), ar_matrix=np.diag([.9, .8]),
        ar_intercept=np.array([.01, -.02]), spot_state_loading=beta @ decoder,
        joint_innovation_covariance=np.diag([.01, .02, .03]))
    law = ProjectedLaw(dynamics=fit, encoder=decoder, decoder=decoder, spot_loading=beta)
    state = SimpleNamespace(native=np.array([.3, -.2, .1]))
    tail = {"fit": {"config": {"persistence": .5, "stationary_log_variance": .1, "unconditional_scale": 1.}},
        "conditional_spot_loading": [.01, .02], "conditional_factor": np.diag([.2, .3]).tolist(),
        "initial_next_log_variance_mean": 0., "initial_next_log_variance_variance": .1}
    return state, law, tail


def test_first_snapshot_matches_completed_one_day_law():
    state, law, tail = fixture()
    snapshots, shocks = native_snapshots(state, law, tail, paths=4, horizons=(1,), seed=71)
    expected_shocks = TailSampler(law.dynamics, tail, paths=4, seed=71).draw()
    np.testing.assert_array_equal(shocks, expected_shocks)
    fit = law.dynamics
    projected = state.native @ law.encoder
    drift = fit.ar_matrix @ projected + fit.ar_intercept - projected
    expected = state.native + (drift + shocks[:, 1:]) @ law.decoder.T + shocks[:, :1] * law.spot_loading
    np.testing.assert_allclose(snapshots[1], expected, atol=1e-15)
    np.testing.assert_allclose(snapshots[1][:, 2], state.native[2] + shocks[:, 0] * .4)


def test_snapshot_horizons_share_one_recursive_population():
    state, law, tail = fixture()
    short, short_shocks = native_snapshots(state, law, tail, paths=4, horizons=(1, 3), seed=71)
    long, long_shocks = native_snapshots(state, law, tail, paths=4, horizons=(1, 3, 5), seed=71)
    np.testing.assert_array_equal(short_shocks, long_shocks)
    for horizon in short:
        np.testing.assert_array_equal(short[horizon], long[horizon])


def test_retaining_fewer_paths_does_not_change_the_random_stream():
    state, law, tail = fixture()
    full, _ = native_snapshots(state, law, tail, paths=4, horizons=(1, 3), seed=71)
    subset, _ = native_snapshots(state, law, tail, paths=4, retain_paths=2, horizons=(1, 3), seed=71)
    for horizon in full:
        np.testing.assert_array_equal(subset[horizon], full[horizon][:2])
