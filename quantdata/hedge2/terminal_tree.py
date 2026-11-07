"""Small complete-tree power-OCE references, not sampled-path production policies."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack
from scipy.special import xlogy

from quantdata.hedge2.objectives import power_oce
from quantdata.hedge2.terminal import (
    TerminalEvaluation,
    TerminalPaths,
    evaluate_terminal_policy,
)


@dataclass(frozen=True, kw_only=True)
class TreeGeometry:
    node_map: np.ndarray
    active: np.ndarray
    turnover: csr_matrix
    initial_offset: np.ndarray
    leaf_gains: csr_matrix


@dataclass(frozen=True, kw_only=True)
class TerminalPowerCertificate:
    primal_upper_bound: float
    dual_lower_bound: float
    objective_gap: float
    density: np.ndarray
    flows: np.ndarray
    solver: str
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class TerminalTreeHedge:
    node_positions: np.ndarray
    node_map: np.ndarray
    evaluation: TerminalEvaluation
    certificate: TerminalPowerCertificate
    solver: str
    status: str
    warnings: tuple[str, ...]
    certified: bool


def tree_geometry(paths: TerminalPaths, initial_inventory: np.ndarray) -> TreeGeometry:
    """Represent charged rebalancing only; settlement and KO have no turnover row."""
    count, steps, instruments = paths.gains.shape
    initial = np.asarray(initial_inventory, dtype=float)
    if initial.shape != (instruments,) or not np.isfinite(initial).all() or np.any(initial[~paths.live_contracts[0]]):
        raise ValueError("Tree root inventory must be finite, aligned and alive")
    nodes, offset = np.empty((count, steps), dtype=int), 0
    for step in range(steps):
        identifiers, inverse = np.unique(paths.history_ids[:, step], return_inverse=True)
        nodes[:, step] = offset + inverse
        offset += len(identifiers)
    if offset * instruments > 4096:
        raise ValueError("The explicit-tree reference is limited to 4096 node-instrument decisions")
    active = np.zeros((offset, instruments), dtype=bool)
    initial_offset = np.zeros((offset, instruments))
    alive = np.full(count, paths.initially_active, dtype=bool)
    rows, columns, values = [], [], []
    for step in range(steps):
        identifiers, first = np.unique(nodes[:, step], return_index=True)
        active[identifiers] = alive[first, None] & paths.live_contracts[step]
        for node, representative in zip(identifiers, first, strict=True):
            assets = np.flatnonzero(active[node])
            indices = node * instruments + assets
            rows.extend(indices)
            columns.extend(indices)
            values.extend(np.ones(len(assets)))
            if step == 0:
                initial_offset[node, assets] = initial[assets]
            else:
                parent = nodes[representative, step - 1]
                rows.extend(indices)
                columns.extend(parent * instruments + assets)
                values.extend(-np.ones(len(assets)))
        alive &= ~paths.terminate_after_step[:, step]
    dimension = offset * instruments
    turnover = coo_matrix((values, (rows, columns)), shape=(dimension, dimension)).tocsr()
    leaf_rows = np.broadcast_to(np.arange(count)[:, None, None], paths.gains.shape).ravel()
    leaf_columns = (nodes[:, :, None] * instruments + np.arange(instruments)).ravel()
    leaf_gains = coo_matrix((paths.gains.ravel(), (leaf_rows, leaf_columns)), shape=(count, dimension)).tocsr()
    return TreeGeometry(node_map=nodes, active=active, turnover=turnover,
                        initial_offset=initial_offset, leaf_gains=leaf_gains)


def evaluate_node_positions(*, paths: TerminalPaths, geometry: TreeGeometry,
                            node_positions: np.ndarray, initial_inventory: np.ndarray,
                            target_side: int, position_cap: float) -> TerminalEvaluation:
    """Independently replay a nonanticipative explicit-tree position table."""
    if node_positions.shape != geometry.active.shape:
        raise ValueError("Node positions do not match the information tree")
    return evaluate_terminal_policy(paths=paths, initial_inventory=initial_inventory,
        policy=lambda features, inventory, step: node_positions[geometry.node_map[:, step]],
        target_side=target_side, position_cap=position_cap)


def certify_terminal_power(*, paths: TerminalPaths, initial_inventory: np.ndarray,
                           node_positions: np.ndarray, target_side: int,
                           risk_aversion: float, position_cap: float) -> TerminalPowerCertificate:
    """A feasible density and bounded turnover flows give an independent lower bound.

    With turnover B*a-d, tilted expected gains G and tilted costs C, any
    |flow|<=C gives -E[q*claim]-E[v0(q)]/lambda-flow*d
    -cap*sum(abs(-G+B.T*flow)). Inactive decisions are fixed to zero.
    The LP proposes flows; their clipped values are evaluated independently.
    """
    if (risk_aversion <= 0 or not np.isfinite(risk_aversion)
            or position_cap <= 0 or not np.isfinite(position_cap)):
        raise ValueError("The power certificate needs finite positive risk and position limits")
    geometry = tree_geometry(paths, initial_inventory)
    evaluation = evaluate_node_positions(paths=paths, geometry=geometry, node_positions=node_positions,
        initial_inventory=initial_inventory, target_side=target_side, position_cap=position_cap)
    probabilities = paths.weights / paths.weights.sum()
    oce = power_oce(evaluation.wealth, probabilities, risk_aversion=risk_aversion)
    primal = -oce.certainty_equivalent
    best_bound, best_density, best_flows = -np.inf, None, None
    collected, selected_status = [], "zero-flow feasible lower bound"
    selected_solver = "analytical"
    dimension = geometry.active.size
    mask = geometry.active.ravel()
    offset = geometry.initial_offset.ravel()
    for candidate in (oce.marginal_utility, np.ones(len(probabilities))):
        density = np.maximum(np.asarray(candidate, dtype=float), 0.)
        density /= probabilities @ density
        tilted = probabilities * density
        gain = np.asarray(geometry.leaf_gains.T @ tilted).ravel() * mask
        costs = np.zeros(geometry.active.shape)
        for step in range(paths.gains.shape[1]):
            np.add.at(costs, geometry.node_map[:, step], tilted[:, None] * paths.costs[:, step])
        costs = costs.ravel() * mask
        below = density <= 1.
        conjugate = np.where(below, xlogy(density, density) - density + 1., .5 * (density - 1.)**2)
        constant = -float(tilted @ (target_side * paths.claim_gain)) - float(probabilities @ conjugate) / risk_aversion

        def bound(flows: np.ndarray) -> tuple[float, np.ndarray]:
            repaired = np.clip(np.asarray(flows, dtype=float), -costs, costs)
            coefficient = -gain + geometry.turnover.T @ repaired
            value = constant - offset @ repaired - position_cap * np.abs(coefficient[mask]).sum()
            return float(value), repaired

        flow_candidates = [(np.zeros(dimension), "analytical", "zero-flow feasible lower bound")]
        constraints = vstack([hstack([geometry.turnover.T, -eye(dimension)]),
                              hstack([-geometry.turnover.T, -eye(dimension)])], format="csr")
        result = linprog(c=np.r_[offset, position_cap * mask], A_ub=constraints,
            b_ub=np.r_[gain, -gain], bounds=[(-value, value) for value in costs] +
            [(0., None) if enabled else (0., 0.) for enabled in mask],
            method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                      "dual_feasibility_tolerance": 1e-9})
        if result.x is not None and np.isfinite(result.x).all():
            flow_candidates.append((result.x[:dimension], "HiGHS", str(result.message)))
        if not result.success:
            collected.append(f"HiGHS flow proposal: {result.message}")
        for flows, solver, status in flow_candidates:
            value, repaired = bound(flows)
            if value > best_bound:
                best_bound, best_density, best_flows = value, density.copy(), repaired
                selected_solver, selected_status = solver, status
    if best_bound > primal + 1e-8 * max(1., abs(primal)):
        raise RuntimeError("Terminal power lower bound violates weak duality")
    return TerminalPowerCertificate(primal_upper_bound=primal, dual_lower_bound=best_bound,
        objective_gap=max(primal - best_bound, 0.), density=best_density,
        flows=best_flows.reshape(geometry.active.shape), solver=selected_solver,
        status=selected_status, warnings=tuple(collected))


def solve_complete_tree_power(*, paths: TerminalPaths, initial_inventory: np.ndarray,
                              target_side: int, risk_aversion: float,
                              position_cap: float = 1000., gap_tolerance: float = 1e-6) -> TerminalTreeHedge:
    """CPU reference for a specified COMPLETE tree, never a unique-history MC fit.

    Each decision node represents its whole observable history. Full net
    terminal wealth includes the same future turnover fees as the cash replay.
    There is no separate outside-the-OCE cost or terminal liquidation fee.
    """
    if (target_side not in (-1, 1) or risk_aversion <= 0 or not np.isfinite(risk_aversion)
            or position_cap <= 0 or not np.isfinite(position_cap) or gap_tolerance <= 0
            or np.any(np.abs(initial_inventory) > position_cap)):
        raise ValueError("Invalid complete-tree side, risk, inventory or numerical tolerance")
    geometry = tree_geometry(paths, initial_inventory)
    variable = cp.Variable(geometry.active.size)
    turnover = cp.Variable(geometry.active.size, nonneg=True)
    count, steps, instruments = paths.gains.shape
    leaf_rows = np.broadcast_to(np.arange(count)[:, None, None], paths.gains.shape).ravel()
    leaf_columns = (geometry.node_map[:, :, None] * instruments + np.arange(instruments)).ravel()
    charged = coo_matrix((paths.costs.ravel(), (leaf_rows, leaf_columns)),
                         shape=geometry.leaf_gains.shape).tocsr()
    wealth = target_side * paths.claim_gain + geometry.leaf_gains @ variable - charged @ turnover
    shifted = cp.Variable()
    positive, negative = cp.Variable(count, nonneg=True), cp.Variable(count, nonneg=True)
    mask = geometry.active.ravel()
    changes = geometry.turnover @ variable - geometry.initial_offset.ravel()
    constraints = [variable <= position_cap * mask, variable >= -position_cap * mask,
                   turnover >= changes, turnover >= -changes,
                   positive - negative == risk_aversion * wealth - shifted]
    probabilities = paths.weights / paths.weights.sum()
    objective = (-shifted + probabilities @ (cp.exp(-positive) + negative + .5 * cp.square(negative)) - 1.) / risk_aversion
    problem = cp.Problem(cp.Minimize(objective), constraints)
    collected, best, best_dual = [], None, None
    for solver in ("CLARABEL", "SCS"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if solver == "CLARABEL":
                    problem.solve(solver=solver, max_iter=1000, tol_gap_abs=1e-10,
                                   tol_gap_rel=1e-10, tol_feas=1e-10)
                else:
                    problem.solve(solver=solver, max_iters=100000, eps=1e-8)
                collected.extend(str(item.message) for item in caught)
            if variable.value is None or not np.isfinite(variable.value).all():
                collected.append(f"{solver}: no finite portfolio; status={problem.status}")
                continue
            candidate = np.asarray(variable.value).reshape(geometry.active.shape)
            if np.max(np.abs(candidate), initial=0.) > position_cap + 1e-6:
                collected.append(f"{solver}: candidate exceeds position cap")
                continue
            candidate = np.clip(candidate, -position_cap, position_cap) * geometry.active
            evaluation = evaluate_node_positions(paths=paths, geometry=geometry, node_positions=candidate,
                initial_inventory=initial_inventory, target_side=target_side, position_cap=position_cap)
            certificate = certify_terminal_power(paths=paths, initial_inventory=initial_inventory,
                node_positions=candidate, target_side=target_side, risk_aversion=risk_aversion, position_cap=position_cap)
            if best_dual is None or certificate.dual_lower_bound > best_dual.dual_lower_bound:
                best_dual = certificate
            if best is None or certificate.primal_upper_bound < best.certificate.primal_upper_bound:
                best = TerminalTreeHedge(node_positions=candidate, node_map=geometry.node_map,
                    evaluation=evaluation, certificate=certificate, solver=solver,
                    status=str(problem.status), warnings=tuple(collected), certified=False)
            if best.certificate.primal_upper_bound - best_dual.dual_lower_bound <= gap_tolerance:
                break
        except (cp.error.SolverError, ValueError) as error:
            collected.append(f"{solver}: {error}")
    if best is None:
        raise RuntimeError("No independently feasible terminal-tree portfolio: " + "; ".join(collected))
    certificate = replace(best_dual, primal_upper_bound=best.certificate.primal_upper_bound,
        objective_gap=max(best.certificate.primal_upper_bound - best_dual.dual_lower_bound, 0.))
    return TerminalTreeHedge(node_positions=best.node_positions, node_map=best.node_map,
        evaluation=best.evaluation, certificate=certificate, solver=best.solver,
        status=best.status, warnings=tuple(collected), certified=certificate.objective_gap <= gap_tolerance)
