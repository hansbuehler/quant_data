from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.run_oce_smc import (
    _population_audit,
    _systematic_resample,
)


def test_systematic_resampling_preserves_equal_particle_population() -> None:
    weights = torch.full((8,), 1.0 / 8.0, dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)

    indices = _systematic_resample(weights, generator=generator)

    assert torch.equal(indices, torch.arange(8))


def test_population_audit_uses_between_population_uncertainty() -> None:
    means = np.asarray(((0.01, 0.02), (0.03, 0.00), (0.02, 0.01)))
    audit = _population_audit(means, cost_band=np.asarray((0.015, 0.015)))

    assert audit["mean"] == pytest.approx((0.02, 0.01))
    assert audit["maximum_cost_band_violation_bps"] == pytest.approx(50.0)
    assert audit["fraction_outside_cost_band"] == pytest.approx(0.5)
