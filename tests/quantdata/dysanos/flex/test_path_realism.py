"""Tests for held-out surface-path distribution diagnostics."""

from __future__ import annotations

import numpy as np

from quantdata.dysanos.flex.path_realism import paired_block_distribution_test
from quantdata.dysanos.flex.run_surface_path_realism import _student_t_shock


def test_identical_paired_paths_are_not_rejected() -> None:
    generator = np.random.default_rng(3)
    values = generator.normal(size=(80, 6))
    result = paired_block_distribution_test(values, values.copy(), permutations=199)
    assert result["mmd_squared"] <= 0.0
    assert result["block_swap_p_value"] == 1.0


def test_shifted_path_distribution_is_rejected() -> None:
    generator = np.random.default_rng(5)
    observed = generator.normal(size=(120, 8))
    simulated = generator.normal(loc=1.2, size=(120, 8))
    result = paired_block_distribution_test(
        observed,
        simulated,
        block_length=4,
        permutations=499,
    )
    assert result["standardized_energy_distance"] > 0.5
    assert result["block_swap_p_value"] <= 0.01


def test_student_t_shock_uses_one_radial_scale_per_path() -> None:
    seed = 812
    shape = (7, 5)
    degrees_of_freedom = 8.0
    expected_generator = np.random.default_rng(seed)
    gaussian = expected_generator.normal(size=shape)
    radial_scale = expected_generator.chisquare(
        degrees_of_freedom, size=(shape[0], 1)
    )
    expected = gaussian * np.sqrt(
        (degrees_of_freedom - 2.0) / radial_scale
    )

    shock, radius = _student_t_shock(
        np.random.default_rng(seed), shape, degrees_of_freedom
    )

    np.testing.assert_allclose(shock, expected)
    np.testing.assert_allclose(radius, np.sum(np.square(expected), axis=1))
