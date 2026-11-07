"""Full-covariance Greek hedges with Gaussian power-OCE risk, without surface simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, lsq_linear
from scipy.special import erfcx, log_ndtr, ndtr, roots_hermitenorm

from quantdata.hedge2.objectives import power_oce
from quantdata.hedge2.optimization import OptimizedHedge, ScenarioOptimizer


@lru_cache(maxsize=8)
def gaussian_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 32:
        raise ValueError("Gaussian OCE quadrature needs at least 32 nodes")
    nodes, weights = roots_hermitenorm(order)
    weights /= np.sqrt(2. * np.pi)
    selected = weights > 1e-20
    nodes, weights = nodes[selected], weights[selected]
    weights /= weights.sum()
    return nodes, weights


def gaussian_power_risk_quadrature(std: float, *, risk_aversion: float, order: int = 256) -> tuple[float, float]:
    """Independent quadrature check, not the production Gaussian objective."""
    if not np.isfinite(std) or std < 0 or not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Gaussian risk requires finite positive risk aversion and nonnegative std")
    if std == 0:
        return 0., 0.
    nodes, weights = gaussian_quadrature(order)
    oce = power_oce(std * nodes, weights, risk_aversion=risk_aversion)
    derivative = -float(weights @ ((oce.marginal_utility - 1.) * nodes))
    return -oce.certainty_equivalent, max(derivative, 0.)


def _normal_power_values(std: float, *, risk_aversion: float) -> tuple[float, float, float]:
    """Normal-law power OCE and std derivative from truncated-normal integrals.

    At the optimal dimensionless cash shift a=m/std, the cash condition
    simplifies the risk to std*(-a + t*M2(a)/2), where t=lambda*std.
    An expm1 tail difference and scaled erfc keep the cash root stable.
    Only t<0.01 uses the fourth-order small-risk expansion.
    """
    if not np.isfinite(std) or std < 0 or not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Gaussian risk requires finite positive risk aversion and nonnegative std")
    if std == 0:
        return 0., 0., 0.
    scaled_std = risk_aversion * std
    if not np.isfinite(scaled_std):
        raise ValueError("Gaussian lambda*std exceeds floating-point range")
    normal_density_at_zero = 1. / np.sqrt(2. * np.pi)
    def terms(shift: float) -> tuple[float, float, float, float]:
        probability = float(ndtr(shift))
        density = float(np.exp(-.5 * shift**2) * normal_density_at_zero)
        tail_argument = shift + scaled_std
        if tail_argument >= 0:
            log_tail = -.5 * shift**2 + np.log(.5 * erfcx(tail_argument / np.sqrt(2.)))
        else:
            log_tail = scaled_std * (shift + .5 * scaled_std) + log_ndtr(-tail_argument)
        log_survival = float(log_ndtr(-shift))
        tail_gap = -np.exp(log_survival) * np.expm1(min(float(log_tail - log_survival), 0.))
        first_moment = density + shift * probability
        second_moment = (1. + shift**2) * probability + shift * density
        return float(np.exp(log_tail)), first_moment, second_moment, float(tail_gap)

    if scaled_std < .01:
        risk = std * (.5 * scaled_std - normal_density_at_zero * scaled_std**2 / 3.
                      + scaled_std**3 / 32. - 7. * normal_density_at_zero * scaled_std**4 / 480.)
        derivative = scaled_std * (1. - normal_density_at_zero * scaled_std
                                  + scaled_std**2 / 8. - 7. * normal_density_at_zero * scaled_std**3 / 96.)
        shift = (-scaled_std / 4. + normal_density_at_zero * scaled_std**2 / 12.
                 + (normal_density_at_zero**2 / 12. - 1. / 64.) * scaled_std**3)
        _, first_moment, _, tail_gap = terms(shift)
        return float(risk), float(derivative), float(scaled_std * first_moment - tail_gap)

    def cash_condition(shift: float) -> float:
        _, first_moment, _, tail_gap = terms(shift)
        return first_moment - tail_gap / scaled_std

    lower = -1.
    while cash_condition(lower) > 0:
        lower *= 2.
    shift = brentq(cash_condition, lower, 0., xtol=1e-12, rtol=1e-12)
    tail, first_moment, second_moment, tail_gap = terms(shift)
    risk = std * (-shift + .5 * scaled_std * second_moment)
    derivative = scaled_std * (tail + float(ndtr(shift)))
    if not np.isfinite(risk) or not np.isfinite(derivative) or risk < 0 or derivative < 0:
        raise ValueError("Normal power-OCE evaluation left its finite nonnegative range")
    return float(risk), float(derivative), float(scaled_std * first_moment - tail_gap)


def gaussian_power_risk(std: float, *, risk_aversion: float) -> tuple[float, float]:
    """Return Gaussian power-OCE risk and its std derivative."""
    risk, derivative, _ = _normal_power_values(std, risk_aversion=risk_aversion)
    return risk, derivative


@dataclass(frozen=True, kw_only=True)
class ParametricHedge:
    hedge: OptimizedHedge
    effective_mv_aversion: float
    aversion_equation_residual: float
    covariance_rank: int
    gaussian_risk_method: str = "analytic_truncated_normal"


class ParametricOptimizer:
    """Use exact joint covariance for MV, and normal-law OCE as a scalar risk of std.

    Symmetric sigma points reproduce covariance exactly for the internal
    quadratic programs; they are NOT the distribution used for power OCE.
    Gaussian power OCE uses analytic truncated-normal integrals, then its optimum is
    found on the mean-variance/cost frontier via k=rho'(std)/std.
    """

    def __init__(self, *, exposures: np.ndarray, covariance: np.ndarray,
                 costs: np.ndarray, position_cap: float = 1000.) -> None:
        if (exposures.ndim != 2 or covariance.shape != (exposures.shape[1], exposures.shape[1])
                or costs.shape != (exposures.shape[0],)
                or not np.isfinite(exposures).all() or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14)):
            raise ValueError("Parametric exposures and symmetric covariance must have matching finite geometry")
        eigenvalues, vectors = np.linalg.eigh(covariance)
        scale = max(float(eigenvalues.max()), 1e-16)
        if eigenvalues.min() < -1e-10 * scale:
            raise ValueError("Parametric factor covariance is not positive semidefinite")
        self.root = np.sqrt(np.maximum(eigenvalues, 0.))[:, None] * vectors.T
        self.exposures = exposures
        self.costs = costs
        self.position_cap = position_cap
        self.rank = int(np.sum(eigenvalues > scale * 1e-12))
        dimensions = covariance.shape[0]
        self.points = np.r_[self.root, -self.root] * np.sqrt(dimensions)
        self.optimizer = ScenarioOptimizer(instrument_gains=self.points @ exposures.T,
            weights=np.full(2 * dimensions, .5 / dimensions), costs=costs, position_cap=position_cap)

    def solve(self, *, target_exposures: np.ndarray, objective: str, risk_aversion: float,
              target_side: int = -1, previous_positions: np.ndarray | None = None) -> ParametricHedge:
        if target_exposures.shape != (self.exposures.shape[1],) or not np.isfinite(target_exposures).all():
            raise ValueError("Parametric target exposure does not match the factor covariance")
        target = self.points @ target_exposures
        arguments = dict(claim_gains=target, target_side=target_side, previous_positions=previous_positions)
        if objective == "mean_variance":
            hedge = self.optimizer.solve(objective=objective, risk_aversion=risk_aversion, **arguments)
            return ParametricHedge(hedge=hedge, effective_mv_aversion=risk_aversion,
                                    aversion_equation_residual=0., covariance_rank=self.rank)
        if objective != "power_exponential" or not np.isfinite(risk_aversion) or risk_aversion <= 0:
            raise ValueError("Parametric power risk aversion must be positive and finite")
        candidates = {}

        def evaluate(effective: float) -> tuple[OptimizedHedge, float, float, float]:
            if effective in candidates:
                return candidates[effective]
            if effective == 0:
                previous = np.zeros(len(self.costs)) if previous_positions is None else np.asarray(previous_positions)
                positions = previous.copy()
                free = self.costs == 0.
                fixed = self.exposures[~free].T @ previous[~free] + target_side * target_exposures
                matrix = self.root @ self.exposures[free].T
                if free.any() and np.any(matrix):
                    limit = lsq_linear(matrix, -self.root @ fixed,
                        bounds=(-self.position_cap, self.position_cap), tol=1e-12, max_iter=1000)
                    if not limit.success or not np.isfinite(limit.x).all():
                        raise RuntimeError(f"Zero-cost variance-limit solve failed: {limit.message}")
                    positions[free] = limit.x
                std = float(np.linalg.norm(self.root @ (self.exposures.T @ positions + target_side * target_exposures)))
                hedge = OptimizedHedge(positions=positions, entry_cost=0., training_objective=0.,
                    training_mean=0., training_std=std, cash_first_order_residual=0.,
                    convex_first_order_gap_bound=0., cap_hits=int(np.sum(np.abs(positions) >= self.position_cap * (1.-1e-6))),
                    solver="zero-cost bounded variance limit", status="analytic root endpoint", warnings=(), iterations=0)
            else:
                hedge = self.optimizer.solve(objective="mean_variance", risk_aversion=effective, **arguments)
            residual = self.exposures.T @ hedge.positions + target_side * target_exposures
            std = float(np.linalg.norm(self.root @ residual))
            risk, derivative, cash_residual = _normal_power_values(std, risk_aversion=risk_aversion)
            needed = risk_aversion if std == 0 else derivative / std
            candidates[effective] = hedge, risk, needed, cash_residual
            return hedge, risk, needed

        lower = 0.
        lower_needed = evaluate(lower)[2]
        if lower_needed <= lower:
            effective = lower
        else:
            effective = brentq(lambda value: value - evaluate(value)[2], lower, risk_aversion,
                               xtol=1e-9, rtol=1e-8, maxiter=100)
        hedge, risk, needed, cash_residual = evaluate(effective)
        previous = np.zeros_like(hedge.positions) if previous_positions is None else previous_positions
        residual = self.exposures.T @ hedge.positions + target_side * target_exposures
        gradient = needed * self.exposures @ (self.root.T @ self.root @ residual)
        choices = np.array([-np.full_like(previous, self.position_cap), previous, np.full_like(previous, self.position_cap)])
        gap = float(np.sum(hedge.positions * gradient + self.costs * np.abs(hedge.positions - previous)
                          - np.min(choices * gradient + self.costs * np.abs(choices - previous), axis=0)))
        corrected = replace(hedge, training_objective=risk + hedge.entry_cost,
                             training_mean=-hedge.entry_cost, cash_first_order_residual=cash_residual,
                             convex_first_order_gap_bound=max(gap, 0.))
        return ParametricHedge(hedge=corrected, effective_mv_aversion=effective,
                                aversion_equation_residual=abs(effective - needed),
                                covariance_rank=self.rank)
