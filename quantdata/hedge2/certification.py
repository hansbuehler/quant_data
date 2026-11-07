"""Independent objective-unit acceptance of one-day statistical hedge candidates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from quantdata.hedge2.dual import certify_mean_variance, certify_power
from quantdata.hedge2.objectives import power_oce
from quantdata.hedge2.optimization import OptimizedHedge, ScenarioOptimizer
from quantdata.hedge2.polish import polish_power


def certify_candidate(*, optimizer: ScenarioOptimizer, candidate: OptimizedHedge,
                      claim_gains: np.ndarray, objective: str, risk_aversion: float,
                      target_side: int, gap_tolerance: float = 1e-4,
                      previous_positions: np.ndarray | None = None,
                      certificate_directory: Path | None = None) -> tuple[OptimizedHedge, dict[str, object]]:
    """Refine only when necessary; retain warnings and the original candidate gap.

    Multi-day turnover uses the same fixed prior portfolio in the primal,
    direct refinement and independent lower bound. No warning alone rejects
    a portfolio; acceptance uses a valid objective-unit gap.
    """
    if not np.isfinite(gap_tolerance) or gap_tolerance <= 0:
        raise ValueError("Certificate tolerance must be positive in objective units")
    metadata = {"original_gap": candidate.convex_first_order_gap_bound,
                "original_solver": candidate.solver, "original_status": candidate.status,
                "original_warnings": list(candidate.warnings),
                "polished": False, "objective_gap_tolerance": gap_tolerance}
    previous = np.zeros_like(candidate.positions) if previous_positions is None else np.asarray(previous_positions, dtype=float)
    if previous.shape != candidate.positions.shape or not np.isfinite(previous).all() or np.any(np.abs(previous) > optimizer.position_cap):
        raise ValueError("Certification needs feasible, aligned prior holdings")
    if certificate_directory is not None and candidate.convex_first_order_gap_bound > gap_tolerance:
        certificate_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(certificate_directory / "original_candidate.npz", positions=candidate.positions,
                            previous_positions=previous, claim_gains=claim_gains)
    hedge = candidate
    if objective == "power_exponential" and candidate.convex_first_order_gap_bound > gap_tolerance:
        refined = polish_power(instrument_gains=optimizer.gains, claim_gains=claim_gains,
            weights=optimizer.weights, costs=optimizer.costs, initial_positions=candidate.positions,
            risk_aversion=risk_aversion, target_side=target_side, position_cap=optimizer.position_cap,
            previous_positions=previous)
        positions = np.clip(refined.positions, -optimizer.position_cap, optimizer.position_cap)
        wealth = optimizer.gains @ positions + target_side * claim_gains
        mean = float(optimizer.weights @ wealth)
        cost = float(optimizer.costs @ np.abs(positions - previous))
        oce = power_oce(wealth, optimizer.weights, risk_aversion=risk_aversion)
        gradient = -optimizer.gains.T @ (optimizer.weights * oce.marginal_utility)
        choices = np.array([-np.full_like(positions, optimizer.position_cap), previous,
                            np.full_like(positions, optimizer.position_cap)])
        gap = float(np.sum(positions * gradient + optimizer.costs * np.abs(positions - previous)
            - np.min(choices * gradient + optimizer.costs * np.abs(choices - previous), axis=0)))
        hedge = replace(candidate, positions=positions, entry_cost=cost,
            training_objective=-oce.certainty_equivalent + cost, training_mean=mean - cost,
            training_std=float(np.sqrt(optimizer.weights @ (wealth - mean)**2)),
            cash_first_order_residual=oce.cash_first_order_residual,
            convex_first_order_gap_bound=max(gap, 0.),
            cap_hits=int(np.sum(np.abs(positions) >= optimizer.position_cap * (1. - 1e-6))),
            solver="SLSQP direct power refinement", status=refined.solver_message,
            iterations=refined.iterations, warnings=(*candidate.warnings, *refined.warnings))
        metadata.update(polished=True, objective_improvement=refined.before_objective - refined.after_objective,
                        polish_solver_success=refined.solver_success)
    gap = hedge.convex_first_order_gap_bound
    if gap > gap_tolerance:
        certifier = certify_power if objective == "power_exponential" else certify_mean_variance
        dual = certifier(instrument_gains=optimizer.gains, claim_gains=claim_gains,
            weights=optimizer.weights, costs=optimizer.costs, positions=hedge.positions,
            risk_aversion=risk_aversion, target_side=target_side, position_cap=optimizer.position_cap,
            previous_positions=previous, gap_tolerance=gap_tolerance)
        gap = min(gap, dual.objective_gap)
        metadata.update(dual_lower_bound=dual.dual_lower_bound, dual_objective_gap=dual.objective_gap,
                        primal_upper_bound=dual.primal_upper_bound,
                        dual_solver=dual.solver, dual_status=dual.status, dual_warnings=list(dual.warnings))
        if certificate_directory is not None:
            variable = dual.density if objective == "power_exponential" else dual.dual_variable
            np.savez_compressed(certificate_directory / "lower_bound.npz", dual_variable=variable,
                positions=hedge.positions, primal_upper_bound=dual.primal_upper_bound,
                dual_lower_bound=dual.dual_lower_bound)
    metadata.update(certified_objective_gap=gap, optimization_certified=bool(gap <= gap_tolerance))
    return hedge, metadata
