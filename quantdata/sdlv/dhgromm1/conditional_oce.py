"""Ordinary OCE on probability-weighted branches, with OUTER-pair MC errors."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from quantdata.sdlv.dhgromm1.training import _economic_utility


def weighted_cash(net: np.ndarray, weights: np.ndarray) -> float:
    """One cash scalar for the entire distribution, never one cash per branch."""
    net, weights = np.asarray(net, float), np.asarray(weights, float)
    if net.shape != weights.shape or net.ndim != 1 or not np.isfinite(net).all() or not np.isfinite(weights).all():
        raise ValueError("expected finite matching one-dimensional wealth and weights")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive total mass")
    keep = weights > 0
    net, weights = net[keep], weights[keep]
    if np.ptp(net) == 0:
        return float(net[0])
    def objective(cash):
        z = net-cash
        derivative = np.exp(-100*np.maximum(z, 0))+100*np.maximum(-z, 0)
        return float(np.dot(weights, derivative-1))
    return float(brentq(objective, float(net.min()), float(net.max()), xtol=1e-14))


def pair_statistic(values: np.ndarray) -> dict:
    """Inner branches are NOT independent paths; cluster at their outer pair."""
    if values.ndim != 1 or len(values) < 4 or len(values) % 2 or not np.isfinite(values).all():
        raise ValueError("expected finite complete outer antithetic pairs")
    pairs = values.reshape(-1, 2).mean(1)
    return dict(oce=float(values.mean()), three_standard_errors=float(3*pairs.std(ddof=1)/np.sqrt(len(pairs))),
                outer_paths=len(values), independent_outer_pairs=len(pairs))


def summarize(net: np.ndarray, penalty: np.ndarray, weights: np.ndarray, outer: np.ndarray,
              *, outer_paths: int, training_cash: np.ndarray, payoff_mean: float) -> tuple[dict, dict]:
    """Both raw utility and OCE-minus-L1/L2, with original centering frozen.

    The pilot's MC errors hold all original training estimates fixed, exactly
    matching three_validation_only_errors in the previous validation reports.
    Cash profiling is evaluation of a fixed hedge, not training or selection.
    """
    mass = np.bincount(outer, weights=weights, minlength=outer_paths)
    np.testing.assert_allclose(mass, 1, atol=2e-12, rtol=0)
    reports, retained = {}, {}
    for treatment in ("fixed_training_cash", "profiled_cash"):
        scores, objectives = np.zeros((outer_paths, 6)), np.zeros((outer_paths, 6))
        cash = training_cash.copy()
        for i in range(6):
            if treatment == "profiled_cash":
                cash[i] = weighted_cash(net[:, i], weights)
            utility = _economic_utility(net[:, i], cash[i], 100.)
            np.add.at(scores[:, i], outer, weights*utility)
            np.add.at(objectives[:, i], outer, weights*(utility-penalty[:, i]))
        result = {}
        for model, start in (("dh", 0), ("dh_l1_l2", 3)):
            s = scores[:, start:start+3]
            result[model] = dict(statistics={label:dict(**pair_statistic(s[:, i]),cash=float(cash[start+i]))
                for i, label in enumerate(("empty", "long", "short"))},
                long_minus_empty=pair_statistic(s[:, 1]-s[:, 0]),short_minus_empty=pair_statistic(s[:, 2]-s[:, 0]),
                bid=pair_statistic(payoff_mean+s[:, 1]-s[:, 0]),ask=pair_statistic(payoff_mean+s[:, 0]-s[:, 2]),
                objective_minus_regularizer={label:pair_statistic(objectives[:, start+i])
                    for i, label in enumerate(("empty", "long", "short"))})
        reports[treatment] = result
        retained[treatment+"_outer_utilities"] = scores
    reports["uncertainty"] = "±3 outer-antithetic-pair MC SE; includes nested randomness and branch covariance; original training centering/scaling/weights fixed; excludes fitting/selection and training-centering uncertainty"
    reports["objective"] = "Ordinary full-distribution power-exponential OCE only; no bottom90/hinge. L1/L2 outside utility, not cash costs."
    return reports, retained
