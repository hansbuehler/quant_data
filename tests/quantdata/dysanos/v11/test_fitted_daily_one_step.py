"""Shape and antithetic checks for the fitted daily one-step sampler."""

import numpy as np

from quantdata.dysanos.v11.run_fitted_daily_one_step import _simulate_one_step


def test_normal_sampler_uses_one_antithetic_pair_per_two_paths() -> None:
    parameters = {
        "equilibrium_mean": np.zeros(2),
        "transition": np.eye(2),
        "innovation_cholesky": np.eye(2),
        "spot_loading": np.array([0.3, -0.2]),
        "training_spot_score_mean": np.array(0.0),
    }
    state, score = _simulate_one_step(
        np.array([1.0, 2.0]), parameters,
        path_count=8, seed=17, tail_config=None,
    )
    np.testing.assert_allclose(state[:4] + state[4:], np.array([[2.0, 4.0]] * 4))
    np.testing.assert_allclose(score[:4] + score[4:], 0.0)
