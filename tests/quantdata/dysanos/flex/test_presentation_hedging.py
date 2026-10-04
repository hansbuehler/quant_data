"""Numerical tests for the Direct-200 presentation hedging transition."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)

CHECKPOINT = Path(
    "/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/"
    "dte1-5-weekly-no-am-all-history-v1/checkpoints/joint-direct-200.pt"
)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="presentation checkpoint unavailable")
def test_exact_daily_transition_composes_to_fitted_weekly_law() -> None:
    _, model = load_presentation_model(CHECKPOINT, device=torch.device("cpu"))
    daily_transition, daily_factor, diagnostics = exact_daily_transition(model)
    assert daily_transition.shape == (200, 200)
    assert daily_factor.shape == (200, 200)
    # The stored checkpoint and model matrix exponential are float32.  The
    # fractional transition itself is solved in float64, so this is the
    # expected comparison tolerance against the stored weekly computation.
    assert diagnostics["transition_max_abs_error"] < 3.0e-6
    assert diagnostics["covariance_max_abs_error"] < 3.0e-6
    assert torch.all(torch.isfinite(daily_factor))
