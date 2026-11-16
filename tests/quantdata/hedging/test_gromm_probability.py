"""Probability weighting changes the PCA metric, never the native loading units."""

import numpy as np
import pytest

from quantdata.hedging.model_audit.gromm_probability import probability_metric, to_native_loading


def test_metric_uses_causal_date_weighted_probability_masses() -> None:
    densities = np.array([[[.1, .5, .3, .1], [.05, .6, .3, .05]],
                          [[.1, .3, .5, .1], [.05, .2, .7, .05]]])
    metric = probability_metric(densities, np.array([1., 3.]))
    np.testing.assert_allclose(metric**2, [[.3, .6]])


def test_loading_mapping_exactly_undoes_metric() -> None:
    metric = np.array([[.2, .8], [.4, .1]])
    native = np.arange(12.).reshape(2, 2, 3)
    np.testing.assert_allclose(to_native_loading(native * metric[..., None], metric), native)


def test_zero_mass_is_not_replaced_with_an_arbitrary_floor() -> None:
    densities = np.broadcast_to([0., 0., 1., 0.], (3, 2, 4))
    with pytest.raises(ValueError, match="Zero-mass"):
        probability_metric(densities, np.ones(3))


def test_unnormalized_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="normalized"):
        probability_metric(np.ones((3, 2, 4)), np.ones(3))
