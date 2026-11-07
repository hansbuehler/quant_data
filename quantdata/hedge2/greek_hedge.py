"""Cost-efficient Greek matching and genuinely reoptimized cost frontiers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from quantdata.hedge2.greek_budget import solve_greek_budget_projection
from quantdata.hedge2.greek_feasibility import GreekBoxCertificate, certify_greek_box


class GreekBoxInfeasible(RuntimeError):
    """An independent separator proves the exact reference cannot fit the box."""

    def __init__(self, *, certificate: GreekBoxCertificate, solver_status: int,
                 solver_message: str) -> None:
        self.certificate = certificate
        self.solver_status = solver_status
        self.solver_message = solver_message
        self.portfolio_context: dict[str, object] | None = None
        super().__init__(f"Exact Greek match failed: HiGHS status={solver_status}; {solver_message}; "
            f"independent minimum-cap lower bound={certificate.dual_lower_bound:.17g}, "
            f"declared cap={certificate.declared_cap:.17g}")


@dataclass(frozen=True, kw_only=True)
class GreekHedge:
    positions: np.ndarray
    entry_cost: float
    risk_residual_norm: float
    full_entry_cost: float
    cap_hits: int
    solver: str
    status: str
    warnings: tuple[str, ...]


def _minimum_cost_certificate(*, matrix: np.ndarray, target: np.ndarray,
                              costs: np.ndarray, previous: np.ndarray, positions: np.ndarray,
                              normalized_dual: np.ndarray, row_scale: np.ndarray,
                              cost_scale: float, position_cap: float) -> tuple[float, float, float]:
    """Bound exact-match turnover cost using the original, unscaled equality rows.

    Any finite equality multiplier supplies a Lagrangian lower bound. Each
    box-constrained turnover coordinate attains its minimum at either end
    or at unchanged inventory. Wider arithmetic starts BEFORE row division,
    inventory subtraction and matrix products; casting rounded solver RHSs
    would not recover that precision. The chosen represented multiplier is
    arbitrary, so its solver/mapping accuracy is not a certificate premise.
    The returned gap and pre-cancellation roundoff allowance are both in
    cost-scaled units. Platforms without wider longdouble retain their
    larger allowance rather than silently relax the acceptance criterion.
    """
    precision = np.longdouble
    original = np.asarray(matrix, dtype=precision)
    original_target = np.asarray(target, dtype=precision)
    old = np.asarray(previous, dtype=precision)
    candidate = np.asarray(positions, dtype=precision)
    fee = np.asarray(costs, dtype=precision) / precision(cost_scale)
    dual = np.asarray(normalized_dual, dtype=precision) / np.asarray(row_scale, dtype=precision)
    cap = precision(position_cap)
    projection = original.T @ dual
    change = np.array([-cap - old, np.zeros_like(old), cap - old])
    terms = fee[None, :] * np.abs(change) - change * projection[None, :]
    lower = dual @ (original_target - original @ old) + terms.min(axis=0).sum()
    upper = (fee * np.abs(candidate - old)).sum()
    projection_magnitude = np.abs(original).T @ np.abs(dual)
    magnitude = (np.abs(dual) @ (np.abs(original_target) + np.abs(original) @ np.abs(old))
        + (fee[None, :] * np.abs(change) + np.abs(change) * projection_magnitude[None, :]).max(axis=0).sum()
        + upper)
    allowance = precision(128 + 16 * sum(original.shape)) * np.finfo(precision).eps * max(magnitude, precision(1.))
    gap = upper - lower
    if not np.isfinite([lower, upper, magnitude, allowance, gap]).all():
        raise RuntimeError("Exact Greek minimum-cost certificate arithmetic is non-finite")
    return float(lower), float(gap), float(allowance)


def solve_greek_hedge(*, exposures: np.ndarray, target_exposures: np.ndarray,
                      costs: np.ndarray, previous_positions: np.ndarray | None = None,
                      risk_scales: np.ndarray | None = None, target_side: int = -1,
                      budget_fraction: float = 1., position_cap: float = 1000.,
                      whiten_equalities: bool = False) -> GreekHedge:
    """Hedge a long (+1) or short (-1) claim, paying option entry costs only.

    First find a minimum-entry-cost exact exposure match. Lower budgets
    minimize the remaining scaled Greek exposure, rather than scale that
    solution. Spot is included explicitly and may have zero cost.
    """
    instruments, risks = exposures.shape
    previous = np.zeros(instruments) if previous_positions is None else np.asarray(previous_positions, dtype=float)
    scales = np.ones(risks) if risk_scales is None else np.asarray(risk_scales, dtype=float)
    if (target_exposures.shape != (risks,) or costs.shape != (instruments,)
            or previous.shape != (instruments,) or scales.shape != (risks,)
            or not all(np.isfinite(values).all() for values in (exposures, target_exposures, costs, previous, scales))
            or np.any(costs < 0) or np.any(scales <= 0) or position_cap <= 0
            or np.any(np.abs(previous) > position_cap + 1e-8)
            or target_side not in (-1, 1) or not 0 <= budget_fraction <= 1
            or not isinstance(whiten_equalities, bool)):
        raise ValueError("Invalid Greek hedge inputs, side, budget or position limit")
    target = -target_side * target_exposures
    matrix = exposures.T
    row_scale = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(target))
    active = row_scale > 1e-14
    scaled_matrix = matrix[active] / row_scale[active, None]
    rhs = (target - matrix @ previous)[active] / row_scale[active]
    equality, equality_rhs = scaled_matrix, rhs
    if whiten_equalities:
        left, singular, _ = np.linalg.svd(scaled_matrix, full_matrices=False)
        rank_tolerance = np.finfo(float).eps * max(scaled_matrix.shape, default=1) * float(singular.max(initial=0.))
        if not len(rhs) or len(singular) != len(rhs) or np.any(singular <= rank_tolerance):
            raise RuntimeError("Exact Greek numerical recovery requires an invertible equality-row transform")
        transform = left.T / singular[:, None]
        equality, equality_rhs = transform @ scaled_matrix, transform @ rhs
    signs = np.c_[np.eye(instruments), -np.eye(instruments)]
    cost_scale = max(float(costs.max()), 1e-12)
    result = linprog(c=np.r_[costs, costs] / cost_scale,
                     A_ub=np.r_[signs, -signs],
                     b_ub=np.r_[position_cap - previous, position_cap + previous],
                     A_eq=equality @ signs, b_eq=equality_rhs, bounds=(0., 2. * position_cap),
                     method="highs-ipm" if whiten_equalities else "highs",
                     options={"dual_feasibility_tolerance": 1e-10 if whiten_equalities else 1e-8,
                              "primal_feasibility_tolerance": 1e-10 if whiten_equalities else 1e-8,
                              **({"ipm_optimality_tolerance": 1e-12} if whiten_equalities else {})})
    if not result.success:
        if result.status == 2:
            certificate = certify_greek_box(exposures=exposures, target_exposures=target_exposures,
                position_cap=position_cap, target_side=target_side)
            if certificate.certified_infeasible:
                raise GreekBoxInfeasible(certificate=certificate, solver_status=int(result.status),
                    solver_message=str(result.message))
        raise RuntimeError(f"Exact Greek match failed: HiGHS status={result.status}; {result.message}")
    full = previous + signs @ result.x
    residual = matrix @ full - target
    if np.max(np.abs(residual[active] / row_scale[active]), initial=0.) > 1e-6:
        raise RuntimeError("The exact Greek match failed its independent exposure residual check")
    full_cost = float(costs @ np.abs(full - previous))
    collected_warnings = []
    if whiten_equalities:
        equality_dual = transform.T @ np.asarray(result.eqlin.marginals)
        lower_bound, cost_gap, allowance = _minimum_cost_certificate(matrix=matrix[active], target=target[active],
            costs=costs, previous=previous, positions=full, normalized_dual=equality_dual,
            row_scale=row_scale[active], cost_scale=cost_scale, position_cap=position_cap)
        if (not np.isfinite([cost_gap, lower_bound, allowance]).all()
                or cost_gap < -allowance or cost_gap + allowance > 1e-8 * max(1., full_cost / cost_scale)):
            raise RuntimeError("Whitened exact Greek match failed its original-row minimum-cost dual check: "
                f"gap={cost_gap:.17g}, roundoff_allowance={allowance:.17g}")
        collected_warnings.append("Equivalent invertible SVD equality-row recovery; original-row "
            f"minimum-cost gap bound <= {(max(cost_gap, 0.) + allowance) * cost_scale:.17g}")
    positions, solver, status = full, "HiGHS", str(result.message)
    if budget_fraction < 1. and full_cost > 0.:
        budget = budget_fraction * full_cost
        variable = cp.Variable(instruments)
        target_scale = max(float(np.linalg.norm(scales * target)), 1e-10)
        residual_expression = cp.multiply(scales / target_scale, matrix @ variable - target)
        normalized_costs = costs / full_cost
        constraints = [variable <= position_cap, variable >= -position_cap,
                       normalized_costs @ cp.abs(variable - previous) <= budget_fraction]
        problem = cp.Problem(cp.Minimize(cp.sum_squares(residual_expression)), constraints)
        failures = []
        for candidate in ("CLARABEL", "OSQP"):
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if candidate == "CLARABEL":
                        problem.solve(solver=candidate, tol_gap_abs=1e-9, tol_gap_rel=1e-9,
                                       tol_feas=1e-9, max_iter=500)
                    else:
                        problem.solve(solver=candidate, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
                if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or variable.value is None:
                    failures.append(f"{candidate}: {problem.status}")
                    continue
                proposal = np.asarray(variable.value)
                proposal_cost = float(costs @ np.abs(proposal - previous))
                if (not np.isfinite(proposal).all() or np.any(np.abs(proposal) > position_cap + 1e-5)
                        or proposal_cost > budget + max(1e-10, full_cost * 1e-6)):
                    failures.append(f"{candidate}: independent position/cost residual failed")
                    continue
                positions, solver, status = proposal, candidate, problem.status
                break
            except cp.error.SolverError as error:
                failures.append(f"{candidate}: {error}")
            finally:
                collected_warnings.extend(str(item.message) for item in caught)
        else:
            original_error = "Cost-constrained Greek match failed: " + "; ".join(failures)
            try:
                recovered = solve_greek_budget_projection(matrix=scales[:, None] * matrix,
                    target=scales * target, costs=costs, previous=previous,
                    budget=budget, position_cap=position_cap)
            except RuntimeError as error:
                raise RuntimeError(original_error + "; " + str(error)) from error
            positions, solver, status = recovered.positions, recovered.solver, recovered.status
            collected_warnings.extend((original_error, *recovered.warnings,
                f"Independent Greek squared-risk objective gap <= {recovered.objective_gap_bound:.17g}; "
                f"normalization scale={recovered.normalization_scale:.17g}"))
    projected = np.clip(positions, -position_cap, position_cap)
    adjustment = float(np.max(np.abs(projected - positions), initial=0.))
    if adjustment > 0:
        collected_warnings.append(f"Projected Greek solver cap roundoff; maximum adjustment={adjustment:.17g}")
        positions = projected
    entry_cost = float(costs @ np.abs(positions - previous))
    return GreekHedge(positions=positions, entry_cost=entry_cost,
                      risk_residual_norm=float(np.linalg.norm(scales * (matrix @ positions - target))),
                      full_entry_cost=full_cost, cap_hits=int(np.sum(np.abs(positions) >= position_cap * (1. - 1e-6))),
                      solver=solver, status=status, warnings=tuple(collected_warnings))
