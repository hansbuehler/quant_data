"""Training-only numerical adequacy against an explicitly feasible zero hedge."""

from __future__ import annotations

import numpy as np

from quantdata.hedge2.terminal import TerminalPaths, evaluate_terminal_policy, terminal_risk


def assess_zero_action_reference(*, paths: TerminalPaths, initial_inventory: np.ndarray,
                                 fitted_risk: float, objective: str, risk_aversion: float,
                                 target_side: int, precision: int) -> dict[str, float | bool | str]:
    """Liquidate live initial holdings once, then stay flat through termination.

    Liquidation pays today's entry turnover cost; it is not a free removal
    of carried positions. Expiry and knockout remain free. This is a
    necessary training adequacy check, not an optimality or OOS certificate.
    It never selects a checkpoint or replaces a policy with this reference.
    """
    if precision not in (32, 64) or not np.isfinite(fitted_risk):
        raise ValueError("Reference comparison needs finite fitted risk and declared training precision")
    result = evaluate_terminal_policy(paths=paths, initial_inventory=initial_inventory,
        policy=lambda features, inventory, step: np.zeros_like(inventory),
        target_side=target_side, position_cap=None)
    risk = terminal_risk(result.wealth, paths.weights, objective=objective, risk_aversion=risk_aversion)
    tolerance = 256. * np.finfo(np.float32 if precision == 32 else np.float64).eps * max(1., abs(fitted_risk), abs(risk.value))
    weights = paths.weights / paths.weights.sum()
    return {"reference": "liquidate live carried inventory at current costs, then hold zero",
        "zero_action_risk": float(risk.value), "zero_action_mean_entry_cost": float(weights @ result.entry_cost),
        "fitted_policy_risk": float(fitted_risk), "risk_excess_over_zero": float(fitted_risk - risk.value),
        "risk_comparison_tolerance": float(tolerance),
        "no_worse_than_zero_reference": bool(fitted_risk <= risk.value + tolerance)}
