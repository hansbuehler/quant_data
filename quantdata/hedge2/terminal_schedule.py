"""Loss-scale-aware learning-rate scheduling for signed terminal risk values."""

from __future__ import annotations

import math

import torch


def step_training_scheduler(scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                            current_loss: float, *, tolerance: float) -> None:
    """Compare against the running best with an absolute, unit-floored scale.

    An improvement must exceed tolerance * max(1, abs(best)). This matches
    the stability check's loss units, not its adjacent-check comparison.
    Unlike a relative threshold, it is well defined for negative risks.
    """
    if (not math.isfinite(current_loss) or not math.isfinite(tolerance)
            or tolerance < 0 or scheduler.mode != "min"):
        raise ValueError("Terminal scheduling requires finite loss/tolerance and minimum-risk mode")
    best = float(scheduler.best)
    if not math.isfinite(best) and best != math.inf:
        raise ValueError("The scheduler's prior best must be finite or positive infinity")
    scheduler.threshold_mode = "abs"
    scheduler.threshold = tolerance * max(1., abs(best)) if math.isfinite(best) else tolerance
    scheduler.step(current_loss)
