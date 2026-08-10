import numpy as np

from quantdata.dysanos.grid.gendysanos.compare_surface_spot_drift import (
    SurfaceSpotDriftComparisonConfig,
    _hac_mean_test,
    compare_surface_spot_drifts,
)


def _synthetic_states(*, spot_effect: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations = 1_400
    states = np.zeros((observations, 3), dtype=np.float64)
    transition = np.array([[0.91, 0.03], [-0.02, 0.86]])
    loading = np.array([0.12, -0.09]) if spot_effect else np.zeros(2)
    for step in range(observations - 1):
        states[step + 1, 0] = states[step, 0] + 0.01 * rng.normal()
        states[step + 1, 1:] = (
            states[step, 1:] @ transition
            + states[step, 0] * loading
            + 0.015 * rng.normal(size=2)
        )
    time_y = np.arange(observations, dtype=np.float64) / 252.0
    return states, time_y


def test_spot_driven_surface_drift_improves_forecasts_when_present():
    states, time_y = _synthetic_states(spot_effect=True, seed=101)
    report = compare_surface_spot_drifts(
        states,
        time_y,
        config=SurfaceSpotDriftComparisonConfig(
            training_fraction=0.7,
            minimum_rolling_training=500,
            rolling_validation_size=300,
            hac_lag=10,
        ),
    )

    validation = report["validation"]
    log_score = validation["log_spot_vs_autonomous"][
        "gaussian_log_score_gain_alternative_minus_autonomous"
    ]
    squared_error = validation["log_spot_vs_autonomous"][
        "standardized_squared_error_reduction_autonomous_minus_alternative"
    ]
    assert log_score["mean"] > 0.0
    assert log_score["hac_t_statistic"] > 3.0
    assert squared_error["mean"] > 0.0
    assert report["log_spot_loading"]["effect_rms"] > 0.2


def test_spot_driven_surface_drift_does_not_help_when_absent():
    states, time_y = _synthetic_states(spot_effect=False, seed=102)
    report = compare_surface_spot_drifts(
        states,
        time_y,
        config=SurfaceSpotDriftComparisonConfig(
            training_fraction=0.7,
            minimum_rolling_training=500,
            rolling_validation_size=300,
            hac_lag=10,
        ),
    )

    validation = report["validation"]
    log_score = validation["log_spot_vs_autonomous"][
        "gaussian_log_score_gain_alternative_minus_autonomous"
    ]
    assert log_score["mean"] < 0.02
    assert report["log_spot_loading"]["effect_rms"] < 0.1


def test_hac_mean_test_reports_effective_positive_mean():
    rng = np.random.default_rng(103)
    values = np.empty(1_000)
    values[0] = 0.2 + rng.normal(scale=0.1)
    for index in range(1, len(values)):
        values[index] = 0.2 + 0.7 * (values[index - 1] - 0.2) + rng.normal(
            scale=0.1
        )

    result = _hac_mean_test(values, maximum_lag=20)

    assert result["mean"] > 0.15
    assert result["hac_t_statistic"] > 5.0
    assert result["two_sided_normal_p_value"] < 1.0e-5
