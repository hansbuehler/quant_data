"""Continue existing GROMM2 calibrations without the additive-floor dead zone."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import highspy
import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog, minimize

from quantdata.gromm2.calibration import _atomic_npz
from quantdata.gromm2.dlv import decode_density_tower, density0_from_support, solve_sanos_step
from quantdata.gromm2.observed_calibration import (
    ObservedFitConfig,
    fit_statistics,
    prepare_observed_fit,
    price_observed_quotes,
    reference_grid_nms,
)
from quantdata.gromm2.rolling_calibration import (
    estimate_window,
    fit_summary,
    joint_innovation_loadings,
    write_json,
)


@version("0.2", dependencies=[ObservedFitConfig, reference_grid_nms])
@dataclass(frozen=True, kw_only=True)
class RepairConfig:
    """Numerical controls for the unchanged observed-quote objective."""

    maximum_iterations: int = 3_000
    maximum_line_search_steps: int = 50
    function_tolerance: float = 1.0e-12
    gradient_tolerance: float = 1.0e-6
    checkpoint_every: int = 25
    minimum_inside_percent: float = 90.0
    reference_date_quote_range: bool = False
    linear_tower_initialization: bool = False
    linear_program_seconds: float = 600.0
    linear_program_method: str = "highspy-ipm"
    linear_program_scaled_auxiliary: bool = True
    linear_program_sequential: bool = False
    linear_tower_fallback: bool = False

    def __post_init__(self) -> None:
        verify_inp(self.maximum_iterations > 0, lambda: "maximum_iterations must be positive")
        verify_inp(self.maximum_line_search_steps > 0, lambda: "maximum_line_search_steps must be positive")
        verify_inp(self.function_tolerance > 0.0 and self.gradient_tolerance > 0.0, lambda: "optimizer tolerances must be positive")
        verify_inp(self.checkpoint_every > 0, lambda: "checkpoint_every must be positive")
        verify_inp(0.0 < self.minimum_inside_percent <= 100.0, lambda: "invalid quality threshold")
        verify_inp(self.linear_program_seconds > 0.0, lambda: "linear_program_seconds must be positive")
        verify_inp(self.linear_program_method in {"highs", "highs-ipm", "highs-ds", "highspy-ipm"}, lambda: "invalid linear-program optimizer")


def total_variance_functions(data: dict[str, np.ndarray], *, config: ObservedFitConfig) -> tuple[Callable, Callable]:
    """Use log(total DLV squared), preserving prices, bounds, and upper penalty."""
    arrays = {name: jnp.asarray(value) for name, value in data.items() if value.dtype.kind not in "US"}
    intervals = jnp.diff(arrays["model_expiries_y"], prepend=0.0)
    maximum_raw = math.log(config.maximum_dlvsq - config.minimum_dlvsq)

    def decode(log_total: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        dlvsq = jnp.exp(log_total)
        densities = decode_density_tower(arrays["supports"], arrays["transports"], dlvsq, intervals)
        return dlvsq, densities, price_observed_quotes(densities, arrays)

    def loss(log_total: jax.Array) -> jax.Array:
        dlvsq, _, prices = decode(log_total)
        errors = jnp.abs(prices - arrays["fair_values"]) / arrays["half_spreads"]
        quote_loss = jnp.average(jax.nn.relu(errors - 1.0) + config.fair_value_weight * errors, weights=arrays["quote_weights"])
        raw = jnp.log(jnp.maximum(dlvsq - config.minimum_dlvsq, jnp.finfo(dlvsq.dtype).tiny))
        barrier = 0.10 * jax.nn.softplus((raw - maximum_raw) / 0.10) / 0.35
        return quote_loss + 0.5 * jnp.sum(barrier**2)

    return loss, decode


def observed_config(report: dict[str, object]) -> ObservedFitConfig:
    """Recover exactly the saved model specification, not a new source fit."""
    from quantdata.gromm2.geometry import Gromm2GeometryConfig

    fields = dict(report["config"])
    fields["geometry"] = Gromm2GeometryConfig(**fields["geometry"])
    return ObservedFitConfig(**fields)


def relaxed_expiry_bound(data: dict[str, np.ndarray], *, config: ObservedFitConfig, maximum_business_dte: float | None = None) -> dict[str, object]:
    """Lower-bound fit loss by relaxing all DLV and cross-expiry constraints."""
    reports = []
    loss_sum = weight_sum = 0.0
    for expiry in np.unique(data["expiries_y"]):
        if maximum_business_dte is not None and expiry * 252.0 > maximum_business_dte + 1.0e-5:
            continue
        selected = data["expiries_y"] == expiry
        count = int(np.count_nonzero(selected))
        quote = {name: value[selected] for name, value in data.items()
                 if name in {"lower", "upper", "alpha", "before_first", "front_prices", "lower_kernels", "upper_kernels", "fair_values", "half_spreads", "quote_weights", "strikes"}}
        support_indices = sorted(set(quote["upper"].tolist() + ([] if np.all(quote["before_first"]) else quote["lower"].tolist())))
        node_count = data["supports"].shape[1]
        variable_count = len(support_indices) * node_count
        basis = np.zeros((count, variable_count))
        for index, support_index in enumerate(support_indices):
            lower = (quote["lower"] == support_index) & ~quote["before_first"]
            upper = quote["upper"] == support_index
            basis[:, index * node_count:(index + 1) * node_count] = (
                (1.0 - quote["alpha"])[:, None] * lower[:, None] * quote["lower_kernels"]
                + quote["alpha"][:, None] * upper[:, None] * quote["upper_kernels"])
        constants = np.where(quote["before_first"], (1.0 - quote["alpha"]) * quote["front_prices"], 0.0)
        normalized = sparse.csr_matrix(basis / quote["half_spreads"][:, None])
        target = (quote["fair_values"] - constants) / quote["half_spreads"]
        identity, zeros = sparse.eye(count), sparse.csr_matrix((count, count))
        constraints = sparse.vstack((
            sparse.hstack((normalized, -identity, zeros)),
            sparse.hstack((-normalized, -identity, zeros)),
            sparse.hstack((normalized, zeros, -identity)),
            sparse.hstack((-normalized, zeros, -identity)),
        ), format="csr")
        equality = np.zeros((2 * len(support_indices), variable_count + 2 * count))
        for index, support_index in enumerate(support_indices):
            equality[2 * index, index * node_count:(index + 1) * node_count] = 1.0
            equality[2 * index + 1, index * node_count:(index + 1) * node_count] = data["supports"][support_index]
        weights = quote["quote_weights"]
        result = linprog(np.r_[np.zeros(variable_count), weights, config.fair_value_weight * weights],
                         A_ub=constraints, b_ub=np.r_[target + 1.0, -target + 1.0, target, -target],
                         A_eq=sparse.csr_matrix(equality), b_eq=np.ones(len(equality)), bounds=(0.0, None), method="highs")
        if not result.success:
            result = linprog(np.r_[np.zeros(variable_count), weights, config.fair_value_weight * weights],
                             A_ub=constraints, b_ub=np.r_[target + 1.0, -target + 1.0, target, -target],
                             A_eq=sparse.csr_matrix(equality), b_eq=np.ones(len(equality)), bounds=(0.0, None), method="highs-ipm", options={"presolve": False})
            if not result.success:
                raise RuntimeError(f"relaxed density LP failed at {expiry * 252.0}D: {result.message}")
        errors = np.abs(basis @ result.x[:variable_count] + constants - quote["fair_values"]) / quote["half_spreads"]
        loss_sum += float(result.fun)
        weight_sum += float(np.sum(weights))
        interior_low = min(data["supports"][index, 1] for index in support_indices)
        interior_high = max(data["supports"][index, -2] for index in support_indices)
        reports.append({"business_dte": float(expiry * 252.0), "quotes": count,
                        "optimistic_objective": float(result.fun / np.sum(weights)),
                        "inside_percent": float(np.mean(errors <= 1.0 + 1.0e-6) * 100.0),
                        "strikes_outside_interior_fraction": float(np.mean((quote["strikes"] < interior_low) | (quote["strikes"] > interior_high))),
                        "interior_strike_range": [float(interior_low), float(interior_high)],
                        "observed_strike_range": [float(np.min(quote["strikes"])), float(np.max(quote["strikes"]))]})
        print(json.dumps(dict(event="relaxed_expiry", **reports[-1])), flush=True)
    return {"optimistic_objective": loss_sum / weight_sum, "expiries": reports,
            "interpretation": "independent probability laws at every quote expiry; same grid and Black kernels, but no DLV or cross-expiry constraints"}


def highspy_interior_point(objective: np.ndarray, *, equality, equality_target: np.ndarray,
                          inequality, inequality_target: np.ndarray, time_limit_seconds: float,
                          stop_event: threading.Event | None = None) -> OptimizeResult:
    """Retain a feasible interior solution without numerically fragile crossover."""
    matrix = sparse.vstack((equality, inequality), format="csc")
    model = highspy.HighsLp()
    model.num_col_, model.num_row_ = matrix.shape[1], matrix.shape[0]
    model.col_cost_ = objective
    model.col_lower_ = np.zeros(len(objective))
    model.col_upper_ = np.full(len(objective), highspy.kHighsInf)
    model.row_lower_ = np.r_[equality_target, np.full(len(inequality_target), -highspy.kHighsInf)]
    model.row_upper_ = np.r_[equality_target, inequality_target]
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
    solver = highspy.Highs()
    if stop_event is not None:
        def request_interrupt(event):
            if stop_event.is_set():
                event.interrupt()
        solver.cbIpmInterrupt += request_interrupt
        solver.cbSimplexInterrupt += request_interrupt
    for name, value in {"solver": "ipm", "run_crossover": "off", "presolve": "off", "threads": 1,
                        "time_limit": time_limit_seconds, "primal_feasibility_tolerance": 1.0e-8,
                        "dual_feasibility_tolerance": 1.0e-8, "ipm_optimality_tolerance": 1.0e-9}.items():
        if solver.setOptionValue(name, value) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected numerical option {name}={value}")
    solver.passModel(model)
    solver.run()
    status = solver.getModelStatus()
    solution = solver.getSolution()
    point = np.asarray(solution.col_value) if solution.value_valid else None
    return OptimizeResult(success=status == highspy.HighsModelStatus.kOptimal, x=point,
                          fun=float(objective @ point) if point is not None else None,
                          message=solver.modelStatusToString(status), status=int(status))


def linear_tower_initializer(data: dict[str, np.ndarray], *, config: ObservedFitConfig, time_limit_seconds: float = 600.0, method: str = "highspy-ipm", scaled_auxiliary: bool = True, stop_event: threading.Event | None = None) -> dict[str, np.ndarray]:
    """Solve the exact bounded tower in densities and variance-times-mass flux."""
    supports = data["supports"]
    expiry_count, node_count = supports.shape
    inner_count = node_count - 2
    density_count = expiry_count * node_count
    flux_count = expiry_count * inner_count
    quote_count = len(data["strikes"])
    intervals = np.diff(data["model_expiries_y"], prepend=0.0)
    transport_rows = [np.arange(density_count)]
    transport_columns = [np.arange(density_count)]
    transport_values = [np.ones(density_count)]
    for index, transport in enumerate(data["transports"], start=1):
        block = sparse.coo_matrix(transport)
        transport_rows.append(index * node_count + block.row)
        transport_columns.append((index - 1) * node_count + block.col)
        transport_values.append(-block.data)
    density_matrix = sparse.coo_matrix((np.concatenate(transport_values),
                                       (np.concatenate(transport_rows), np.concatenate(transport_columns))),
                                      shape=(density_count, density_count)).tocsr()
    flux_blocks = []
    flux_scales = []
    positions = np.arange(inner_count)
    for support, interval in zip(supports, intervals, strict=True):
        box = support[1:-1]**2 / (support[2:] - support[:-2])
        plus = interval * box / (support[2:] - support[1:-1])
        minus = interval * box / (support[1:-1] - support[:-2])
        flux_scales.append(plus + minus if scaled_auxiliary else np.ones_like(plus))
        flux_blocks.append(sparse.coo_matrix((np.r_[-minus, plus + minus, -plus],
                           (np.r_[positions, positions + 1, positions + 2], np.tile(positions, 3))), shape=(node_count, inner_count)) @ sparse.diags(1.0 / flux_scales[-1]))
    flux_matrix = sparse.block_diag(flux_blocks, format="csr")
    extra_variables = (3 if scaled_auxiliary else 2) * quote_count
    equality = sparse.hstack((density_matrix, flux_matrix, sparse.csr_matrix((density_count, extra_variables))), format="csr")
    equality_target = np.zeros(density_count)
    equality_target[:node_count] = np.asarray(density0_from_support(jnp.asarray(supports[0])))
    row_indices = np.broadcast_to(np.arange(quote_count)[:, None], (quote_count, node_count))
    nodes = np.arange(node_count)[None, :]
    lower_weights = np.where(data["before_first"], 0.0, 1.0 - data["alpha"])
    lower_values = lower_weights[:, None] * data["lower_kernels"]
    upper_values = data["alpha"][:, None] * data["upper_kernels"]
    price_matrix = sparse.coo_matrix((np.r_[lower_values.ravel(), upper_values.ravel()],
                        (np.tile(row_indices.ravel(), 2), np.r_[(data["lower"][:, None] * node_count + nodes).ravel(),
                                                              (data["upper"][:, None] * node_count + nodes).ravel()])),
                        shape=(quote_count, density_count)).tocsr()
    price_matrix.eliminate_zeros()
    constants = np.where(data["before_first"], (1.0 - data["alpha"]) * data["front_prices"], 0.0)
    identity, zeros = sparse.eye(quote_count), sparse.csr_matrix((quote_count, quote_count))
    no_flux = sparse.csr_matrix((quote_count, flux_count))
    if scaled_auxiliary:
        price_equations = sparse.hstack((-price_matrix, no_flux, identity, sparse.csr_matrix((quote_count, 2 * quote_count))), format="csr")
        equality = sparse.vstack((equality, price_equations), format="csr")
        equality_target = np.r_[equality_target, constants]
        no_state = sparse.csr_matrix((quote_count, density_count + flux_count))
        quote_constraints = sparse.vstack((
            sparse.hstack((no_state, identity, -identity, zeros)),
            sparse.hstack((no_state, -identity, -identity, zeros)),
            sparse.hstack((no_state, identity, zeros, -identity)),
            sparse.hstack((no_state, -identity, zeros, -identity)),
        ), format="csr")
        target = data["fair_values"]
        quote_target = np.r_[target + data["half_spreads"], -target + data["half_spreads"], target, -target]
        objective_scale = float(np.median(data["half_spreads"]))
        cost_weights = objective_scale * data["quote_weights"] / data["half_spreads"]
    else:
        price_matrix = sparse.diags(1.0 / data["half_spreads"]) @ price_matrix
        quote_constraints = sparse.vstack((
            sparse.hstack((price_matrix, no_flux, -identity, zeros)),
            sparse.hstack((-price_matrix, no_flux, -identity, zeros)),
            sparse.hstack((price_matrix, no_flux, zeros, -identity)),
            sparse.hstack((-price_matrix, no_flux, zeros, -identity)),
        ), format="csr")
        target = (data["fair_values"] - constants) / data["half_spreads"]
        quote_target = np.r_[target + 1.0, -target + 1.0, target, -target]
        objective_scale = 1.0
        cost_weights = data["quote_weights"]
    interior_columns = (np.arange(expiry_count)[:, None] * node_count + np.arange(1, node_count - 1)).ravel()
    interior_selector = sparse.coo_matrix((np.ones(flux_count), (np.arange(flux_count), interior_columns)), shape=(flux_count, density_count)).tocsr()
    no_slack = sparse.csr_matrix((flux_count, extra_variables))
    scaled_selector = sparse.diags(np.concatenate(flux_scales)) @ interior_selector
    bounds = sparse.vstack((
        sparse.hstack((config.minimum_dlvsq * scaled_selector, -sparse.eye(flux_count), no_slack)),
        sparse.hstack((-config.maximum_dlvsq * scaled_selector, sparse.eye(flux_count), no_slack)),
    ), format="csr")
    inequality = sparse.vstack((quote_constraints, bounds), format="csr")
    inequality_target = np.r_[quote_target, np.zeros(2 * flux_count)]
    weights = data["quote_weights"]
    objective = np.r_[np.zeros(density_count + flux_count + (quote_count if scaled_auxiliary else 0)), cost_weights, config.fair_value_weight * cost_weights]
    print(json.dumps(dict(event="tower_lp_start", variables=len(objective), equality_rows=equality.shape[0],
                          inequality_rows=inequality.shape[0], nonzeros=equality.nnz + inequality.nnz,
                          time_limit_seconds=time_limit_seconds, method=method, scaled_auxiliary=scaled_auxiliary)), flush=True)
    result = (highspy_interior_point(objective, equality=equality, equality_target=equality_target,
                                    inequality=inequality, inequality_target=inequality_target, time_limit_seconds=time_limit_seconds,
                                    stop_event=stop_event)
              if method == "highspy-ipm" else
              linprog(objective, A_ub=inequality, b_ub=inequality_target,
                     A_eq=equality, b_eq=equality_target, bounds=(0.0, None), method=method,
                     options={"time_limit": time_limit_seconds, "primal_feasibility_tolerance": 1.0e-9,
                              "dual_feasibility_tolerance": 1.0e-8, "ipm_optimality_tolerance": 1.0e-10,
                              "presolve": not scaled_auxiliary,
                              "disp": True}))
    if not result.success:
        if result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"bounded tower LP did not converge or retain a feasible initializer: {result.message}")
        equality_error = float(np.max(np.abs(equality @ result.x - equality_target)))
        inequality_error = float(np.max(inequality @ result.x - inequality_target))
        if equality_error > 1.0e-7 or inequality_error > 1.0e-7 or float(np.min(result.x)) < -1.0e-9:
            raise RuntimeError(f"non-converged tower LP initializer is not feasible: {result.message}")
        print(json.dumps(dict(event="tower_lp_nonconverged_feasible_initializer", message=str(result.message),
                              maximum_equality_error=equality_error, maximum_inequality_error=inequality_error)), flush=True)
    densities = result.x[:density_count].reshape(expiry_count, node_count)
    flux = result.x[density_count:density_count + flux_count].reshape(expiry_count, inner_count) / np.asarray(flux_scales)
    dlvsq = np.divide(flux, densities[:, 1:-1], out=np.full_like(flux, config.minimum_dlvsq), where=densities[:, 1:-1] > 0.0)
    dlvsq = np.clip(dlvsq, config.minimum_dlvsq, config.maximum_dlvsq)
    decoded = np.asarray(decode_density_tower(jnp.asarray(supports), jnp.asarray(data["transports"]), jnp.asarray(dlvsq), jnp.asarray(intervals)))
    prices = np.asarray(price_observed_quotes(jnp.asarray(decoded), {name: jnp.asarray(value) for name, value in data.items() if value.dtype.kind not in "US"}))
    errors = np.abs(prices - data["fair_values"]) / data["half_spreads"]
    normalized_objective = result.fun / (objective_scale * np.sum(weights))
    print(json.dumps(dict(event="tower_lp_finished", objective=float(normalized_objective),
                          decoded_inside_percent=float(np.mean(errors <= 1.0) * 100.0),
                          maximum_decode_probability_error=float(np.max(np.abs(decoded - densities))))), flush=True)
    return {"dlvsq": dlvsq, "lp_densities": densities, "decoded_densities": decoded, "flux": flux,
            "prices": prices, "lp_objective": np.asarray(normalized_objective),
            "lp_maximum_density_decode_error": np.asarray(np.max(np.abs(decoded - densities))),
            "lp_optimizer_success": np.asarray(result.success), "lp_optimizer_message": np.asarray(str(result.message))}


def sequential_tower_initializer(data: dict[str, np.ndarray], *, config: ObservedFitConfig, time_limit_seconds: float = 600.0) -> dict[str, np.ndarray]:
    """Construct a feasible bounded tower with all quotes in each expiry bracket."""
    supports = data["supports"]
    expiry_count, node_count = supports.shape
    inner_count = node_count - 2
    intervals = np.diff(data["model_expiries_y"], prepend=0.0)
    initial_dlvsq = config.minimum_dlvsq + np.exp(data["initial_raw"])
    previous = np.asarray(density0_from_support(jnp.asarray(supports[0])))
    densities, variances = [], []
    objectives = []
    started = time.monotonic()
    for index, (support, interval) in enumerate(zip(supports, intervals, strict=True)):
        if index:
            previous = data["transports"][index - 1] @ densities[-1]
        selected = data["upper"] == index
        count = int(np.count_nonzero(selected))
        reference_density = np.asarray(solve_sanos_step(jnp.asarray(support), jnp.asarray(previous), jnp.asarray(initial_dlvsq[index] * interval)))
        if not count:
            densities.append(reference_density)
            variances.append(initial_dlvsq[index])
            continue
        box = support[1:-1]**2 / (support[2:] - support[:-2])
        plus = interval * box / (support[2:] - support[1:-1])
        minus = interval * box / (support[1:-1] - support[:-2])
        positions = np.arange(inner_count)
        operator = sparse.coo_matrix((np.r_[-minus, plus + minus, -plus],
                        (np.r_[positions, positions + 1, positions + 2], np.tile(positions, 3))), shape=(node_count, inner_count)).tocsr()
        constants = np.where(data["before_first"][selected], data["front_prices"][selected],
                             data["lower_kernels"][selected] @ (densities[index - 1] if index else previous))
        constants *= 1.0 - data["alpha"][selected]
        kernels = data["alpha"][selected, None] * data["upper_kernels"][selected]
        basis = sparse.csr_matrix(kernels / data["half_spreads"][selected, None])
        target = (data["fair_values"][selected] - constants) / data["half_spreads"][selected]
        identity, zeros = sparse.eye(count), sparse.csr_matrix((count, count))
        no_flux = sparse.csr_matrix((count, inner_count))
        constraints = sparse.vstack((
            sparse.hstack((basis, no_flux, -identity, zeros)),
            sparse.hstack((-basis, no_flux, -identity, zeros)),
            sparse.hstack((basis, no_flux, zeros, -identity)),
            sparse.hstack((-basis, no_flux, zeros, -identity)),
        ), format="csr")
        selector = sparse.coo_matrix((np.ones(inner_count), (positions, positions + 1)), shape=(inner_count, node_count)).tocsr()
        no_slack = sparse.csr_matrix((inner_count, 2 * count))
        bounds = sparse.vstack((
            sparse.hstack((config.minimum_dlvsq * selector, -sparse.eye(inner_count), no_slack)),
            sparse.hstack((-config.maximum_dlvsq * selector, sparse.eye(inner_count), no_slack)),
        ), format="csr")
        inequality = sparse.vstack((constraints, bounds), format="csr")
        inequality_target = np.r_[target + 1.0, -target + 1.0, target, -target, np.zeros(2 * inner_count)]
        equality = sparse.hstack((sparse.eye(node_count), operator, sparse.csr_matrix((node_count, 2 * count))), format="csr")
        weights = data["quote_weights"][selected]
        cost = np.r_[np.zeros(node_count + inner_count), weights, config.fair_value_weight * weights]
        remaining = time_limit_seconds - (time.monotonic() - started)
        if remaining <= 0.0:
            raise TimeoutError("sequential tower LP initializer reached its time limit")
        result = linprog(cost, A_ub=inequality, b_ub=inequality_target, A_eq=equality, b_eq=previous,
                         bounds=(0.0, None), method="highs",
                         options={"time_limit": remaining, "primal_feasibility_tolerance": 1.0e-9, "dual_feasibility_tolerance": 1.0e-8})
        if not result.success:
            raise RuntimeError(f"sequential bounded tower LP failed at expiry {index}: {result.message}")
        primary_value = float(result.fun)
        original_variables = len(cost)
        zero_density = sparse.csr_matrix((inequality.shape[0], node_count))
        density_identity = sparse.hstack((sparse.eye(node_count), sparse.csr_matrix((node_count, original_variables - node_count))), format="csr")
        tie_inequality = sparse.vstack((
            sparse.hstack((inequality, zero_density)),
            sparse.hstack((sparse.csr_matrix(cost[None]), sparse.csr_matrix((1, node_count)))),
            sparse.hstack((density_identity, -sparse.eye(node_count))),
            sparse.hstack((-density_identity, -sparse.eye(node_count))),
        ), format="csr")
        tie_result = linprog(np.r_[np.zeros(original_variables), np.ones(node_count)],
                            A_ub=tie_inequality,
                            b_ub=np.r_[inequality_target, primary_value + max(1.0e-7, abs(primary_value) * 1.0e-9), reference_density, -reference_density],
                            A_eq=sparse.hstack((equality, sparse.csr_matrix((node_count, node_count)))), b_eq=previous,
                            bounds=(0.0, None), method="highs",
                            options={"time_limit": max(1.0, time_limit_seconds - (time.monotonic() - started)),
                                     "primal_feasibility_tolerance": 1.0e-9, "dual_feasibility_tolerance": 1.0e-8})
        if tie_result.success:
            result = tie_result
        else:
            print(json.dumps(dict(event="sequential_lp_tie_break_warning", expiry=index, message=str(tie_result.message))), flush=True)
        density = result.x[:node_count]
        flux = result.x[node_count:node_count + inner_count]
        variance = np.divide(flux, density[1:-1], out=np.full(inner_count, config.minimum_dlvsq), where=density[1:-1] > 0.0)
        variance = np.clip(variance, config.minimum_dlvsq, config.maximum_dlvsq)
        densities.append(np.asarray(solve_sanos_step(jnp.asarray(support), jnp.asarray(previous), jnp.asarray(variance * interval))))
        variances.append(variance)
        objectives.append(primary_value)
        prices = kernels @ densities[-1] + constants
        coverage = float(np.mean(np.abs(prices - data["fair_values"][selected]) <= data["half_spreads"][selected] + 1.0e-12) * 100.0)
        print(json.dumps(dict(event="sequential_lp_expiry", completed=index + 1, total=expiry_count,
                              business_dte=float(data["model_expiries_y"][index] * 252.0), quotes=count,
                              objective=float(primary_value / np.sum(weights)), inside_percent=coverage,
                              elapsed_seconds=time.monotonic() - started)), flush=True)
    decoded = np.asarray(densities)
    prices = np.asarray(price_observed_quotes(jnp.asarray(decoded), {name: jnp.asarray(value) for name, value in data.items() if value.dtype.kind not in "US"}))
    return {"dlvsq": np.asarray(variances), "lp_densities": decoded, "decoded_densities": decoded, "prices": prices,
            "lp_objective": np.asarray(np.sum(objectives) / np.sum(data["quote_weights"])),
            "lp_maximum_density_decode_error": np.asarray(0.0)}


def repair_fit(*, original_directory: Path, run_directory: Path, fixed_grid: Path, config: RepairConfig, initializer: Path | None = None) -> dict[str, object]:
    """Continue a saved tower in a separately checkpointed CPU-only directory."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("historical GROMM2 repair is CPU-only")
    run_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_directory / "fit.lock"), timeout=0):
        return _repair_fit(original_directory=original_directory, run_directory=run_directory, fixed_grid=fixed_grid, config=config, initializer=initializer)


def _repair_fit(*, original_directory: Path, run_directory: Path, fixed_grid: Path, config: RepairConfig, initializer: Path | None) -> dict[str, object]:
    original = json.loads((original_directory / "report.json").read_text())
    fitting = observed_config(original)
    source = Path(original["source_market"])
    reference_curve = None
    if config.reference_date_quote_range:
        with np.load(fixed_grid, allow_pickle=False) as saved:
            reference_curve = np.asarray(saved["model_expiries_y"]), np.asarray(saved["projected_total_variances"])
    data = prepare_observed_fit(source, config=fitting, fixed_grid=fixed_grid,
                                initializer=initializer or original_directory, reference_nms_curve=reference_curve)
    with np.load(original_directory / "fit.npz", allow_pickle=False) as saved:
        original_dlvsq = np.asarray(saved["dlvsq"])
    input_digest = hashlib.sha256()
    for name, value in sorted(data.items()):
        input_digest.update(name.encode())
        input_digest.update(np.ascontiguousarray(value).tobytes())
    provenance = {
        "original_directory": str(original_directory.resolve()),
        "original_fit_sha256": hashlib.sha256((original_directory / "fit.npz").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inputs_sha256": input_digest.hexdigest(),
        "model_config": asdict(fitting),
        "repair_version": RepairConfig.version.unique_id48,
    }
    identity = json.dumps(provenance, sort_keys=True)
    checkpoint = run_directory / "checkpoint.npz"
    initial = np.log(fitting.minimum_dlvsq + np.exp(data["initial_raw"]))
    previous_iterations = 0
    best_value = math.inf
    best = initial.copy()
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["identity"].item()) != identity:
                raise ValueError("repair checkpoint identity differs from the requested fit")
            initial = np.asarray(saved["current_log_total"])
            best = np.asarray(saved["best_log_total"])
            best_value = float(saved["best_objective"])
            previous_iterations = int(saved["completed_iterations"])
    write_json(run_directory / "provenance.json", provenance)
    loss, decode = total_variance_functions(data, config=fitting)
    value_gradient = jax.jit(jax.value_and_grad(loss))
    decode_only = jax.jit(decode)
    stop = threading.Event()
    done = threading.Event()
    started = time.monotonic()
    progress = {"stage": "compiling", "date": original["date"], "iterations": previous_iterations, "evaluations": 0}
    current = initial.copy()
    completed = previous_iterations

    def save() -> None:
        _atomic_npz(checkpoint, {
            "identity": np.asarray(identity), "current_log_total": current,
            "best_log_total": best, "best_objective": np.asarray(best_value),
            "completed_iterations": np.asarray(completed),
            "repair_config": np.asarray(json.dumps(asdict(config), sort_keys=True)),
            "optimizer_memory_restarted_on_resume": np.asarray(True),
        })

    def heartbeat() -> None:
        while not done.wait(60):
            print(json.dumps(dict(event="repair_heartbeat", **progress, elapsed_seconds=time.monotonic() - started, best_objective=best_value)), flush=True)
            try:
                result = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True, text=True, timeout=15, check=True)
                unhealthy = [item for item in json.loads(result.stdout)["checks"] if item["health"] != "OK"]
                if unhealthy:
                    print(json.dumps(dict(event="resource_alert", checks=unhealthy)), flush=True)
                if any(item["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for item in unhealthy):
                    stop.set()
                    if not done.wait(60):
                        os.kill(os.getpid(), signal.SIGKILL)
                    return
            except Exception as error:
                print(json.dumps(dict(event="resource_warning", message=str(error))), flush=True)

    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    threading.Thread(target=heartbeat, daemon=True).start()

    def objective(flat):
        nonlocal best, best_value
        point = flat.reshape(initial.shape)
        value, gradient = value_gradient(jnp.asarray(point))
        value, gradient = float(value), np.asarray(gradient)
        if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("non-finite repair objective or gradient")
        progress["evaluations"] += 1
        if value < best_value:
            best_value, best = value, point.copy()
        return value, gradient.ravel()

    def callback(flat):
        nonlocal current, completed
        current = flat.reshape(initial.shape).copy()
        completed += 1
        progress.update(stage="fitting", iterations=completed)
        if completed % config.checkpoint_every == 0 or stop.is_set():
            save()
            best_prices = np.asarray(decode_only(jnp.asarray(best))[-1])
            coverage = float(np.mean(np.abs(best_prices - data["fair_values"]) <= data["half_spreads"]) * 100.0)
            print(json.dumps(dict(event="repair_progress", **progress, best_objective=best_value, inside_percent=coverage, elapsed_seconds=time.monotonic() - started)), flush=True)
        if stop.is_set():
            raise InterruptedError("repair checkpointed; stopped at the latest accepted L-BFGS iteration")

    quote_policy = "reference-date-relative-strike-range" if config.reference_date_quote_range else "history-date-NMS-range"
    print(json.dumps(dict(event="repair_start", date=original["date"], original_inside_percent=original["fit"]["all"]["inside_percent"], original_objective=original["best_objective"], backend=jax.default_backend(), dead_floor_fraction=float(np.mean(original_dlvsq < fitting.minimum_dlvsq + 1.0e-8)), quote_range_policy=quote_policy, old_quote_count=original["fit"]["all"]["quotes"], new_quote_count=len(data["strikes"]), config=asdict(config))), flush=True)
    try:
        if config.linear_tower_initialization and not checkpoint.exists():
            objective(initial.ravel())
            progress["stage"] = "bounded_tower_linear_program"
            linear_path = run_directory / "linear_tower_initialization.npz"
            if linear_path.exists():
                with np.load(linear_path, allow_pickle=False) as saved:
                    if str(saved["identity"].item()) != identity:
                        raise ValueError("linear tower initializer identity differs")
                    initial = np.log(np.asarray(saved["dlvsq"]))
            else:
                linear = (sequential_tower_initializer(data, config=fitting, time_limit_seconds=config.linear_program_seconds)
                          if config.linear_program_sequential else
                          linear_tower_initializer(data, config=fitting, time_limit_seconds=config.linear_program_seconds,
                                                  method=config.linear_program_method, scaled_auxiliary=config.linear_program_scaled_auxiliary,
                                                  stop_event=stop))
                _atomic_npz(linear_path, {**linear, "identity": np.asarray(identity)})
                initial = np.log(linear["dlvsq"])
            current = initial.copy()
            save()
            if stop.is_set():
                raise InterruptedError("tower LP initializer saved; stop requested before nonlinear continuation")
        result = minimize(
            objective, initial.ravel(), method="L-BFGS-B", jac=True,
            bounds=[(math.log(fitting.minimum_dlvsq), math.log(fitting.maximum_dlvsq))] * initial.size,
            callback=callback,
            options={"maxiter": max(config.maximum_iterations - previous_iterations, 1), "maxfun": 100_000,
                     "maxls": config.maximum_line_search_steps, "ftol": config.function_tolerance,
                     "gtol": config.gradient_tolerance, "maxcor": 20},
        )
        current = result.x.reshape(initial.shape)
        save()
        dlvsq, densities, prices = (np.asarray(value) for value in decode_only(jnp.asarray(best)))
        errors = np.abs(prices - data["fair_values"]) / data["half_spreads"]
        core = data["quote_weights"] == 1.0
        groups = {"all": np.ones(len(errors), dtype=bool), "core": core, "outer": ~core,
                  "first_25_days": data["expiries_y"] <= 25.0 / 252.0 + 1.0e-7,
                  "beyond_25_days": data["expiries_y"] > 25.0 / 252.0 + 1.0e-7}
        raw = np.log(np.maximum(dlvsq - fitting.minimum_dlvsq, np.finfo(float).tiny))
        report = {
            **original, "repair": {**provenance, "config": asdict(config), "completed_iterations": completed,
                                      "evaluations": progress["evaluations"], "optimizer_success": bool(result.success),
                                      "optimizer_message": str(result.message), "elapsed_seconds": time.monotonic() - started,
                                      "coordinate": "log(total DLV squared); same model and observed-quote objective",
                                      "quasi_newton_memory_restarted_on_resume": previous_iterations > 0},
            "quote_range_policy": quote_policy,
            "quote_range_definition": "log(K/F) / sqrt(linearly interpolated origin projected total variance); w(0)=0; same scale as saved computational support" if config.reference_date_quote_range else "existing history-date NMS with one-hour epsilon",
            "prior_quote_population": {"policy": "history-date-NMS-range", "fit": original["fit"]},
            "initialization_date": str(data["initialization_date"]),
            "inputs_sha256": provenance["inputs_sha256"],
            "source_native_terminal_pricing_quote_fraction": float(np.mean(data["expiries_y"] * 252.0 > data["source_terminal_business_dte"] + 1.0e-5)),
            "stop_reason": "bounded_lbfgs_" + str(result.status), "best_objective": best_value,
            "fit": {name: fit_statistics(errors[selected], data["quote_weights"][selected]) for name, selected in groups.items()},
            "per_expiry_fit": [{"business_dte": float(expiry * 252.0),
                                **fit_statistics(errors[data["expiries_y"] == expiry], data["quote_weights"][data["expiries_y"] == expiry])}
                               for expiry in np.unique(data["expiries_y"])],
            "density": {"minimum_probability": float(densities.min()), "maximum_mass_error": float(np.max(np.abs(densities.sum(axis=-1) - 1.0))), "maximum_forward_error": float(np.max(np.abs(np.sum(densities * data["supports"], axis=-1) - 1.0))), "maximum_boundary_mass": float(np.max(densities[:, 0] + densities[:, -1]))},
            "dlvsq": {"minimum": float(dlvsq.min()), "maximum": float(dlvsq.max()),
                      "median": float(np.median(dlvsq)),
                      "density_weighted_mean": float(np.sum(dlvsq * densities[:, 1:-1]) / np.sum(densities[:, 1:-1])),
                      "mean_density_mass_at_variance_above_one": float(np.mean(np.sum(densities[:, 1:-1] * (dlvsq > 1.0), axis=-1))),
                      "fraction_at_upper_bound": float(np.mean(dlvsq >= fitting.maximum_dlvsq * (1.0 - 1.0e-10)))},
            "original_market_inside_percent": float(100.0 * np.mean((prices >= data["market_bids"]) & (prices <= data["market_asks"]))),
        }
        _atomic_npz(run_directory / "fit.npz", {"raw": raw, "dlvsq": dlvsq, "densities": densities, "prices": prices,
                    "errors_half_spreads": errors, "spot_ref": data["spot_ref"], "forward_over_spot": data["forward_over_spot"],
                    "supports": data["supports"], "date": data["date"]})
        write_json(run_directory / "report.json", report)
        print(json.dumps(dict(event="repair_finished", date=original["date"], inside_percent=report["fit"]["all"]["inside_percent"], best_objective=best_value, optimizer_success=bool(result.success), message=str(result.message))), flush=True)
        return report
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def run_reference_repair(*, original_calibration: Path, run_directory: Path, origin: str, config: RepairConfig) -> dict[str, object]:
    """Refit the complete fixed-grid history causally, then apply every gate."""
    if not config.reference_date_quote_range:
        raise ValueError("the complete reference repair requires explicit reference-date quote eligibility")
    run_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_directory / "pipeline.lock"), timeout=0):
        return _run_reference_repair(original_calibration=original_calibration, run_directory=run_directory, origin=origin, config=config)


def _run_reference_repair(*, original_calibration: Path, run_directory: Path, origin: str, config: RepairConfig) -> dict[str, object]:
    original_history = original_calibration / "rolling" / origin / "history"
    history_dates = sorted(path.name for path in original_history.iterdir() if (path / "report.json").is_file())
    if not history_dates or history_dates[-1] >= origin:
        raise ValueError("historical dates must be nonempty and strictly precede the origin")
    reference = run_directory / "own_dates" / origin
    reference.parent.mkdir(parents=True, exist_ok=True)
    if not reference.exists():
        reference.symlink_to((original_calibration / "own_dates" / origin).resolve(), target_is_directory=True)
    fixed_grid = reference / "inputs.npz"
    directory = run_directory / "rolling" / origin
    directory.mkdir(parents=True, exist_ok=True)
    write_json(run_directory / "repair_configuration.json", {
        "original_calibration": str(original_calibration.resolve()), "origin": origin,
        "config": asdict(config), "source_refitting_permitted": False, "backend": jax.default_backend(),
        "quote_range_policy": "reference-date-relative-strike-range",
        "origin_fit_reused_unchanged": str(reference.resolve()),
        "historical_initialization_policy": "previous repaired historical date; first date uses its existing same-date causal fit",
    })
    reports = []
    initializer = None
    started = time.monotonic()
    for number, day in enumerate(history_dates, start=1):
        output = directory / "history" / day
        report_path = output / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("quote_range_policy") != "reference-date-relative-strike-range":
                raise ValueError(f"historical quote-range policy differs on {day}")
        else:
            first_stage = output / "nonlinear_continuation"
            report = repair_fit(original_directory=original_history / day, run_directory=first_stage,
                                fixed_grid=fixed_grid, config=config, initializer=initializer)
            selected_stage = first_stage
            if report["fit"]["all"]["inside_percent"] < config.minimum_inside_percent and config.linear_tower_fallback:
                fallback = output / "linear_restart"
                candidate = repair_fit(original_directory=original_history / day, run_directory=fallback,
                                       fixed_grid=fixed_grid, initializer=initializer,
                                       config=replace(config, linear_tower_initialization=True, linear_tower_fallback=False))
                if candidate["fit"]["all"]["inside_percent"] >= config.minimum_inside_percent or candidate["best_objective"] < report["best_objective"]:
                    report, selected_stage = candidate, fallback
            report = {**report, "selected_repair_stage": str(selected_stage),
                      "selection_rule": "calibration quality gate first, then the unchanged observed-book objective"}
            output.mkdir(parents=True, exist_ok=True)
            with np.load(selected_stage / "fit.npz", allow_pickle=False) as saved:
                _atomic_npz(output / "fit.npz", {name: np.asarray(saved[name]) for name in saved.files})
            write_json(report_path, report)
        reports.append(report)
        initializer = output
        summary = fit_summary(reports, threshold=config.minimum_inside_percent)
        summary = {**summary, "expected_dates": len(history_dates), "complete_history": number == len(history_dates),
                   "ready": number == len(history_dates) and summary["ready"]}
        write_json(directory / "fit_summary.json", summary)
        write_json(run_directory / "repair_progress.json", {
            "stage": "historical_reference_quote_refits", "completed": number, "total": len(history_dates),
            "last_date": day, "elapsed_seconds": time.monotonic() - started,
            "dates_below_threshold": summary["dates_below_threshold"],
        })
        print(json.dumps(dict(event="reference_history_fitted", date=day, completed=number, total=len(history_dates),
                              inside_percent=report["fit"]["all"]["inside_percent"], elapsed_seconds=time.monotonic() - started)), flush=True)
        jax.clear_caches()
    summary = {**fit_summary(reports, threshold=config.minimum_inside_percent),
               "expected_dates": len(history_dates), "complete_history": True}
    if not summary["ready"]:
        raise RuntimeError("reference quote-range history still misses calibration quality gates; no Q model was emitted")
    with np.load(fixed_grid, allow_pickle=False) as saved:
        supports = np.asarray(saved["supports"])
        expiries = np.asarray(saved["model_expiries_y"])
    states, fronts, spots, forwards = [], [], [], []
    for day in history_dates:
        with np.load(directory / "history" / day / "fit.npz", allow_pickle=False) as saved:
            states.append(np.asarray(saved["dlvsq"]))
            fronts.append(np.asarray(saved["densities"])[0])
            spots.append(float(saved["spot_ref"]))
            forwards.append(np.asarray(saved["forward_over_spot"]))
    origin_report = json.loads((reference / "report.json").read_text())
    fitting = observed_config(origin_report)
    estimate = estimate_window(dates=history_dates, log_dlvsq=np.log(np.asarray(states)), supports=supports,
                               model_expiries_y=expiries, forward_over_spot=np.asarray(forwards),
                               spot_ref=np.asarray(spots), front_densities=np.asarray(fronts),
                               backbone_vol=fitting.geometry.backbone_annual_volatility)
    estimate.update(joint_innovation_loadings(estimate["innovations"], estimate["spot_gaussian_scores"], factor_count=16))
    _atomic_npz(directory / "ar1.npz", {**estimate, "history_dates": np.asarray(history_dates),
                                       "reference_date": np.asarray(origin), "supports": supports, "model_expiries_y": expiries})
    report = {
        "model_label": fitting.model_label, "origin": origin, "history_start": history_dates[0], "history_end": history_dates[-1],
        "history_dates": len(history_dates), "one_session_pairs": len(estimate["pair_end_dates"]),
        "fixed_reference_grid": str(fixed_grid), "origin_excluded_from_ar1": True, "historical_calibration": summary,
        "quote_range_policy": "reference-date-relative-strike-range", "source_refitted": False,
        "fixed_contract_clipped_query_fraction": float(estimate["fixed_contract_clipped_query_fraction"]),
        "absolute_ar1_slope_above_one_fraction": float(np.mean(np.abs(estimate["slope"]) > 1.0)),
        "innovation_sd_median": float(np.median(np.sqrt(estimate["innovation_variance"]))),
    }
    write_json(directory / "report.json", report)
    write_json(run_directory / "rolling_complete.json", {"model_label": fitting.model_label, "completed_origins": 1})
    print(json.dumps(dict(event="reference_repair_model_ready", **report)), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    originals = parser.add_mutually_exclusive_group(required=True)
    originals.add_argument("--original-directory", type=Path)
    originals.add_argument("--original-calibration", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--fixed-grid", type=Path)
    parser.add_argument("--maximum-iterations", type=int, default=3_000)
    parser.add_argument("--reference-date-quote-range", action="store_true")
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--origin", default="2025-08-29")
    parser.add_argument("--linear-tower-initialization", action="store_true")
    parser.add_argument("--linear-program-seconds", type=float, default=600.0)
    parser.add_argument("--linear-program-method", choices=("highs", "highs-ipm", "highs-ds", "highspy-ipm"), default="highspy-ipm")
    parser.add_argument("--sequential-linear-program", action="store_true")
    parser.add_argument("--linear-tower-fallback", action="store_true")
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    config = RepairConfig(maximum_iterations=arguments.maximum_iterations, reference_date_quote_range=arguments.reference_date_quote_range,
                          linear_tower_initialization=arguments.linear_tower_initialization, linear_program_seconds=arguments.linear_program_seconds,
                          linear_program_method=arguments.linear_program_method, linear_program_sequential=arguments.sequential_linear_program,
                          linear_tower_fallback=arguments.linear_tower_fallback)
    if arguments.original_calibration is not None:
        run_reference_repair(original_calibration=arguments.original_calibration, run_directory=arguments.run_directory,
                             origin=arguments.origin, config=config)
    else:
        if arguments.fixed_grid is None:
            parser.error("--fixed-grid is required for a single-date repair")
        repair_fit(original_directory=arguments.original_directory, run_directory=arguments.run_directory,
                   fixed_grid=arguments.fixed_grid, initializer=arguments.initializer, config=config)


if __name__ == "__main__":
    main()
