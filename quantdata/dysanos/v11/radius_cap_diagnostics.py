"""Weekly innovation-radius and decoder-cap audits for frozen DYSANOS models."""

from __future__ import annotations

from datetime import date

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs


def innovation_radius_panels(
    states: np.ndarray,
    dates: tuple[date, ...],
    *,
    mean: np.ndarray,
    weekly_transition: np.ndarray,
    weekly_innovation_factor: np.ndarray,
    forward_start: date = date(2025, 1, 1),
) -> dict[int, dict[str, object]]:
    """Whiten exact 1--4-week residuals using the frozen Gaussian AR law.

    The variance is the actual compounded AR covariance, not a one-week
    covariance reused at longer horizons.  Radii are squared lengths divided
    by state dimension; the reference threshold is chi-square at the same d.
    """
    values = np.asarray(states, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64)
    transition = np.asarray(weekly_transition, dtype=np.float64)
    factor = np.asarray(weekly_innovation_factor, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) != len(dates)
        or center.shape != (values.shape[1],)
        or transition.shape != (values.shape[1], values.shape[1])
        or factor.shape != transition.shape
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(transition))
        or np.any(~np.isfinite(factor))
    ):
        raise ValueError("states, dates and weekly AR parameters do not align")
    dimension = values.shape[1]
    weekly_covariance = factor @ factor.T
    threshold_99 = float(chi2.ppf(0.99, dimension) / dimension)
    report: dict[int, dict[str, object]] = {}
    for weeks in (1, 2, 3, 4):
        origin, endpoint = exact_weekly_pairs(dates, weeks=weeks)
        if len(origin) == 0:
            raise ValueError(f"no exact {weeks}-week pairs")
        transition_power = np.linalg.matrix_power(transition, weeks)
        covariance = np.zeros_like(weekly_covariance)
        power = np.eye(dimension)
        for _ in range(weeks):
            covariance += power.T @ weekly_covariance @ power
            power = power @ transition
        cholesky = np.linalg.cholesky(0.5 * (covariance + covariance.T))
        residual = values[endpoint] - center - (values[origin] - center) @ transition_power
        white = solve_triangular(cholesky, residual.T, lower=True).T
        radii = np.sum(white * white, axis=1) / dimension
        forward = np.asarray([dates[index] >= forward_start for index in endpoint])

        def summarize(selected: np.ndarray) -> dict[str, float | int]:
            subset = radii[selected]
            if len(subset) == 0:
                return {"observations": 0}
            return {
                "observations": len(subset),
                "mean": float(np.mean(subset)),
                "median": float(np.median(subset)),
                "p95": float(np.quantile(subset, 0.95)),
                "p99": float(np.quantile(subset, 0.99)),
                "maximum": float(np.max(subset)),
                "fraction_above_gaussian_p99": float(np.mean(subset > threshold_99)),
            }

        report[weeks] = {
            "gaussian_p99": threshold_99,
            "pre_evaluation": summarize(~forward),
            "forward_evaluation": summarize(forward),
        }
    return report


def decoder_cap_diagnostics(
    model: JointDirectDynamicsModel,
    states: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 256,
) -> dict[str, float | int]:
    """Count DLV coordinates at or near the model's hard pricing cap."""
    geometry = model.config.geometry
    cap = geometry.maximum_log_dlvsq
    count_dlvs = geometry.num_model_expiries * geometry.num_inner_strikes
    values = torch.as_tensor(states, dtype=model.fdtype, device="cpu")
    if values.ndim != 2 or values.shape[1] != model.config.state_size or batch_size <= 0:
        raise ValueError("states do not match the model or batch size is invalid")
    total = 0
    at_cap = 0
    near_cap = 0
    maximum_overshoot = float("-inf")
    with torch.inference_mode():
        for first in range(0, len(values), batch_size):
            selected = values[first : first + batch_size].to(model.device)
            raw = model.raw_states_from_states(selected)[..., :count_dlvs]
            total += raw.numel()
            at_cap += int(torch.sum(raw >= cap).item())
            near_cap += int(torch.sum(raw >= cap - 0.1).item())
            maximum_overshoot = max(maximum_overshoot, float(torch.max(raw - cap).item()))
    return {
        "states": len(values),
        "dlv_coordinates": total,
        "fraction_at_hard_cap": at_cap / total,
        "fraction_within_0_1_log_unit_of_cap": near_cap / total,
        "maximum_log_coordinate_overshoot": maximum_overshoot,
    }
