from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.flex.run_oce_density_tilt import fit_oce_density_tilt


def test_oce_density_tilt_matches_direct_formula() -> None:
    base_weights = np.asarray((0.5, 1.0, 1.5, 1.0))
    pnl = np.asarray((0.2, -0.1, 0.4, 0.0))

    fit = fit_oce_density_tilt(base_weights, pnl)

    expected = base_weights * np.exp(-pnl)
    expected /= np.mean(expected)
    assert fit.weights == pytest.approx(expected)
    assert np.exp(fit.log_weights) == pytest.approx(expected)
    assert np.mean(fit.weights) == pytest.approx(1.0, abs=1.0e-15)
    assert fit.effective_sample_size == pytest.approx(
        np.sum(expected) ** 2 / np.sum(np.square(expected))
    )


def test_oce_density_tilt_is_invariant_to_irrelevant_constants() -> None:
    base_weights = np.asarray((0.25, 0.75, 1.25, 1.75))
    pnl = np.asarray((-0.4, 0.3, 0.1, 0.2))

    reference = fit_oce_density_tilt(base_weights, pnl)
    shifted = fit_oce_density_tilt(7.0 * base_weights, pnl + 3.0)

    assert shifted.weights == pytest.approx(reference.weights)


def test_oce_density_tilt_floors_float64_underflow_without_losing_mass() -> None:
    base_weights = np.ones(3)
    pnl = np.asarray((1_000.0, 0.0, 0.0))

    fit = fit_oce_density_tilt(base_weights, pnl)

    assert fit.float64_floor_count == 1
    assert fit.minimum_exact_log_weight < np.log(np.finfo(np.float64).tiny)
    assert np.all(np.isfinite(fit.weights))
    assert np.all(fit.weights > 0.0)
    assert np.mean(fit.weights) == pytest.approx(1.0, abs=1.0e-15)
    assert np.exp(fit.log_weights) == pytest.approx(fit.weights)
