import math

import numpy as np
import pytest

from quantdata.dysanos.flex.run_consistent_tact_audit import (
    unconditional_coordinate_diagnostics,
    weighted_antithetic_test,
)


def test_weighted_antithetic_test_uses_pairs_as_sampling_units() -> None:
    result = weighted_antithetic_test(
        np.array([1.0, 3.0, 2.0, 4.0]),
        np.ones(4),
    )

    assert result["paths"] == 4
    assert result["antithetic_pairs"] == 2
    assert result["mean_daily_captured_return"] == pytest.approx(2.5)
    assert result["pair_based_standard_error"] == pytest.approx(0.5)
    assert result["z_score"] == pytest.approx(5.0)
    assert result["two_sided_normal_p_value"] == pytest.approx(
        math.erfc(5.0 / math.sqrt(2.0))
    )


def test_unconditional_diagnostics_match_direct_ratio_calculation() -> None:
    random = np.random.default_rng(781)
    returns = random.normal(size=(8, 2, 3))
    weights = np.array([0.7, 1.3, 0.9, 1.1, 1.2, 0.8, 0.6, 1.4])
    materiality_floor = 0.1

    result = unconditional_coordinate_diagnostics(
        returns,
        weights,
        batch_size=4,
        materiality_floor=materiality_floor,
    )

    weight_sum = np.sum(weights)
    mean = np.einsum("p,pti->ti", weights, returns) / weight_sum
    influence = weights[:, None, None] * (returns - mean) / np.mean(weights)
    pair_influence = influence.reshape(4, 2, 2, 3).mean(axis=1)
    standard_error = np.sqrt(np.sum(pair_influence**2, axis=0) / 3 / 4)
    absolute_z = np.abs(mean / standard_error)
    absolute_materiality = np.abs(
        mean / np.sqrt(standard_error**2 + materiality_floor**2)
    )

    assert result["paths"] == 8
    assert result["antithetic_pairs"] == 4
    assert result["coordinates"] == 6
    assert result["maximum_absolute_z"] == pytest.approx(np.max(absolute_z))
    assert result["percentile_99_absolute_z"] == pytest.approx(
        np.percentile(absolute_z, 99.0)
    )
    assert result["maximum_absolute_materiality_score"] == pytest.approx(
        np.max(absolute_materiality)
    )
    assert result["percentile_99_absolute_materiality_score"] == pytest.approx(
        np.percentile(absolute_materiality, 99.0)
    )


def test_unconditional_diagnostics_reject_incomplete_antithetic_pair() -> None:
    with pytest.raises(ValueError, match="antithetic pairs"):
        unconditional_coordinate_diagnostics(
            np.ones((5, 2, 3)),
            np.ones(5),
            batch_size=2,
        )
