"""Time-resolved conditional support diagnostics for generated markets."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from scipy.spatial import cKDTree
from scipy.stats import norm

from quantdata.dysanos.grid.rndysanos.local_tail import (
    LocalTailConfig,
    normalize_state,
    state_normalization,
)
from quantdata.dysanos.implied.conditional_diagnostics import (
    _cost_adjusted_local_directions,
)

from .required_runs import (
    DerivedCaseArrays,
    PathSplit,
    clopper_pearson_one_sided,
    positive_relative_weights,
    propagated_basket_band,
    solve_constant_basket,
)


def _sample_paths(
    candidates: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Select deterministic query paths without changing their split."""

    candidates = np.asarray(candidates, dtype=np.int64)
    verify_inp(candidates.ndim == 1, "query candidates must be one-dimensional")
    if candidates.size <= maximum:
        return np.sort(candidates)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=maximum, replace=False))


def _audit_frozen_balls(
    *,
    candidates: list[dict[str, Any]],
    centers: np.ndarray,
    audit_state: np.ndarray,
    audit_returns: np.ndarray,
    audit_costs: np.ndarray,
    audit_bands: np.ndarray,
    audit_log_weights: np.ndarray,
    audit_components: np.ndarray,
    time_index: int,
    family_size: int,
    prerequisite_checks_passed: bool,
    minimum_direct_paths: int = 20,
    minimum_effective_size: float = 20.0,
) -> dict[str, Any]:
    """Evaluate frozen reference balls using actual audit outcomes."""

    adjusted_alpha = 0.05 / family_size
    z = float(norm.ppf(1.0 - adjusted_alpha))
    audit_log_weights = np.asarray(audit_log_weights, dtype=np.float64)
    components = np.asarray(audit_components)
    surviving: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for candidate in candidates:
        center = centers[int(candidate["query_index"])]
        radius = float(candidate["neighbor_radius"])
        squared_distance = np.sum(
            np.square(audit_state.astype(np.float64, copy=False) - center),
            axis=1,
        )
        region = squared_distance <= radius * radius
        instrument = int(candidate["instrument"])
        sign = 1.0 if candidate["direction"] == "long" else -1.0
        value = (
            sign * audit_returns[:, instrument] - audit_costs[:, instrument]
        ).astype(np.float64, copy=False)
        band = audit_bands[:, instrument].astype(np.float64, copy=False)
        regional_paths = int(np.count_nonzero(region))
        region_weight = (
            positive_relative_weights(audit_log_weights[region])
            if regional_paths
            else np.empty(0, dtype=np.float64)
        )
        direct_region = region & (components == 0)
        direct_paths = int(np.count_nonzero(direct_region))
        mass = float(region_weight.sum())
        square_mass = float(np.square(region_weight).sum())
        effective_size = mass * mass / square_mass if square_mass > 0.0 else 0.0
        base = {
            **candidate,
            "trading_time_index": time_index,
            "state_region": {
                "kind": "closed normalized Euclidean ball",
                "center": center.astype(float).tolist(),
                "radius": radius,
            },
            "audit_paths_in_region": regional_paths,
            "direct_P_audit_paths_in_region": direct_paths,
            "regional_effective_sample_size": effective_size,
        }
        if (
            direct_paths < minimum_direct_paths
            or effective_size < minimum_effective_size
        ):
            unresolved.append(
                {
                    **base,
                    "status": "unresolved",
                    "reason": "insufficient direct-P regional paths or weighted ESS",
                    "certified": False,
                }
            )
            continue
        loss = value < -band
        gain = value > band
        direct_value = value[direct_region]
        direct_loss = int(np.count_nonzero(loss[direct_region]))
        direct_gain = int(np.count_nonzero(gain[direct_region]))
        gain_lower, _ = clopper_pearson_one_sided(
            direct_gain, direct_paths, alpha=adjusted_alpha
        )
        _, loss_upper = clopper_pearson_one_sided(
            direct_loss, direct_paths, alpha=adjusted_alpha
        )
        direct_mean = float(np.mean(direct_value))
        direct_error = float(
            np.std(direct_value, ddof=0) / math.sqrt(direct_paths)
        )
        direct_mean_lower = direct_mean - z * direct_error
        region_value = value[region]
        region_gain = gain[region].astype(np.float64)
        weighted_mean = float(np.dot(region_weight, region_value) / mass)
        weighted_gain = float(np.dot(region_weight, region_gain) / mass)
        weighted_mean_error = float(
            np.sqrt(
                np.dot(
                    np.square(region_weight),
                    np.square(region_value - weighted_mean),
                )
            )
            / mass
        )
        weighted_gain_error = float(
            np.sqrt(
                np.dot(
                    np.square(region_weight),
                    np.square(region_gain - weighted_gain),
                )
            )
            / mass
        )
        weighted_mean_lower = weighted_mean - z * weighted_mean_error
        weighted_gain_lower = weighted_gain - z * weighted_gain_error
        conditions = {
            "no_audit_loss_beyond_band": int(np.count_nonzero(loss[region])) == 0,
            "adjusted_direct_P_gain_lower_positive": gain_lower > 0.0,
            "adjusted_direct_P_mean_lower_positive": direct_mean_lower > 0.0,
            "adjusted_weighted_P_gain_lower_positive": weighted_gain_lower > 0.0,
            "adjusted_weighted_P_mean_lower_positive": weighted_mean_lower > 0.0,
            "importance_coverage_and_replay_pass": prerequisite_checks_passed,
        }
        item = {
            **base,
            "status": "evaluated",
            "audit_loss_count": int(np.count_nonzero(loss[region])),
            "audit_gain_count": int(np.count_nonzero(gain[region])),
            "direct_P_loss_count": direct_loss,
            "direct_P_gain_count": direct_gain,
            "adjusted_direct_P_loss_probability_upper": loss_upper,
            "adjusted_direct_P_gain_probability_lower": gain_lower,
            "direct_P_mean": direct_mean,
            "adjusted_direct_P_mean_lower": direct_mean_lower,
            "weighted_P_gain_probability": weighted_gain,
            "adjusted_weighted_P_gain_probability_lower": weighted_gain_lower,
            "weighted_P_mean": weighted_mean,
            "adjusted_weighted_P_mean_lower": weighted_mean_lower,
            "conditions": conditions,
            "certified": bool(all(conditions.values())),
        }
        (surviving if item["certified"] else rejected).append(item)
    return {
        "reference_screened_candidates": len(candidates),
        "multiplicity": {
            "method": "Bonferroni familywise correction",
            "family_size": family_size,
            "adjusted_alpha": adjusted_alpha,
        },
        "minimum_direct_P_paths": minimum_direct_paths,
        "minimum_regional_effective_sample_size": minimum_effective_size,
        "surviving": surviving,
        "rejected": rejected,
        "unresolved": unresolved,
    }


def _fit_and_audit_local_baskets(
    *,
    centers: np.ndarray,
    reference_neighbor_indices: np.ndarray,
    neighbor_radius: np.ndarray,
    audit_state: np.ndarray,
    case: DerivedCaseArrays,
    split: PathSplit,
    log_path_weights: np.ndarray,
    importance_components: np.ndarray,
    time_index: int,
    neighbors: int,
    family_size: int,
    prerequisite_checks_passed: bool,
    progress_interval_seconds: float,
    solver_workers: int = 1,
) -> dict[str, Any]:
    """Fit an all-instrument L1 basket at every query and audit its state ball."""

    centers = np.asarray(centers, dtype=np.float64)
    reference_neighbor_indices = np.asarray(
        reference_neighbor_indices, dtype=np.int64
    )
    neighbor_radius = np.asarray(neighbor_radius, dtype=np.float64)
    queries = centers.shape[0]
    verify_inp(
        centers.ndim == 2
        and reference_neighbor_indices.ndim == 2
        and reference_neighbor_indices.shape[0] == queries
        and neighbor_radius.shape == (queries,),
        "local basket query geometry is inconsistent",
    )
    verify_inp(solver_workers >= 1, "basket solver workers must be positive")
    if queries == 0:
        return {
            "searched_query_states": 0,
            "reference_candidates": 0,
            "zero_reference_solutions": 0,
            "surviving": [],
            "rejected": [],
            "unresolved": [],
        }
    adjusted_alpha = 0.05 / family_size
    z = float(norm.ppf(1.0 - adjusted_alpha))
    audit_log_weights = np.asarray(log_path_weights, dtype=np.float64)[split.audit]
    audit_components = np.asarray(importance_components)[split.audit]
    audit_tree = cKDTree(np.asarray(audit_state, dtype=np.float64))
    audit_region_rows = audit_tree.query_ball_point(
        centers,
        neighbor_radius,
        return_sorted=True,
        workers=-1,
    )
    surviving: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    zero_reference_solutions = 0
    reference_candidates = 0
    started = time.monotonic()
    last_progress = -math.inf

    def report_progress(query_index: int) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if (
            now - last_progress >= progress_interval_seconds
            or query_index + 1 == queries
        ):
            print(
                f"All-instrument local baskets t={time_index}: "
                f"{query_index + 1:,}/{queries:,}; "
                f"reference candidates={reference_candidates:,}, "
                f"surviving={len(surviving):,}",
                flush=True,
            )
            last_progress = now

    def solve_query(query_index: int) -> dict[str, Any]:
        local_rows = reference_neighbor_indices[query_index]
        return solve_constant_basket(
            case,
            time_index=time_index,
            reference_paths=split.reference[local_rows],
            log_path_weights=log_path_weights,
        )

    if solver_workers == 1:
        solutions = [solve_query(query_index) for query_index in range(queries)]
    else:
        print(
            f"All-instrument local baskets t={time_index}: solving "
            f"{queries:,} independent LPs on {solver_workers} CPU workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=solver_workers) as executor:
            solutions = list(executor.map(solve_query, range(queries)))

    for query_index, solution in enumerate(solutions):
        center = centers[query_index]
        action = np.asarray(solution["action"], dtype=np.float64)
        base = {
            "query_index": query_index,
            "trading_time_index": time_index,
            "state_region": {
                "kind": "closed normalized Euclidean ball",
                "center": center.astype(float).tolist(),
                "radius": float(neighbor_radius[query_index]),
            },
            "strategy_class": "local all-instrument state-independent L1 basket",
            "reference_solution": solution,
        }
        if np.abs(action).sum() <= 1.0e-14 or solution["gain_count"] == 0:
            zero_reference_solutions += 1
            report_progress(query_index)
            continue
        reference_candidates += 1
        region_rows = np.asarray(audit_region_rows[query_index], dtype=np.int64)
        regional_paths = int(region_rows.size)
        direct = audit_components[region_rows] == 0
        direct_paths = int(np.count_nonzero(direct))
        region_weights = (
            positive_relative_weights(audit_log_weights[region_rows])
            if regional_paths
            else np.empty(0, dtype=np.float64)
        )
        mass = float(region_weights.sum())
        square_mass = float(np.square(region_weights).sum())
        effective_size = mass * mass / square_mass if square_mass > 0.0 else 0.0
        if direct_paths < 20 or effective_size < 20.0:
            unresolved.append(
                {
                    **base,
                    "status": "unresolved",
                    "audit_paths_in_region": regional_paths,
                    "direct_P_audit_paths_in_region": direct_paths,
                    "regional_effective_sample_size": effective_size,
                    "certified": False,
                }
            )
            report_progress(query_index)
            continue
        regional_case_rows = split.audit[region_rows]
        returns = case.returns[regional_case_rows, time_index].astype(
            np.float64, copy=False
        )
        costs = case.costs[regional_case_rows, time_index].astype(
            np.float64, copy=False
        )
        bands = case.numerical_band[regional_case_rows, time_index].astype(
            np.float64, copy=False
        )
        pnl = returns @ action - costs @ np.abs(action)
        band = propagated_basket_band(action, returns, costs, bands)
        loss = pnl < -band
        gain = pnl > band
        direct_loss = int(np.count_nonzero(loss[direct]))
        direct_gain = int(np.count_nonzero(gain[direct]))
        gain_lower, _ = clopper_pearson_one_sided(
            direct_gain, direct_paths, alpha=adjusted_alpha
        )
        _, loss_upper = clopper_pearson_one_sided(
            direct_loss, direct_paths, alpha=adjusted_alpha
        )
        direct_mean = float(np.mean(pnl[direct]))
        direct_mean_lower = direct_mean - z * float(
            np.std(pnl[direct], ddof=0) / math.sqrt(direct_paths)
        )
        weighted_mean = float(np.dot(region_weights, pnl) / mass)
        weighted_gain = float(
            np.dot(region_weights, gain.astype(np.float64)) / mass
        )
        weighted_error = float(
            np.sqrt(
                np.dot(
                    np.square(region_weights),
                    np.square(pnl - weighted_mean),
                )
            )
            / mass
        )
        weighted_gain_error = float(
            np.sqrt(
                np.dot(
                    np.square(region_weights),
                    np.square(gain.astype(np.float64) - weighted_gain),
                )
            )
            / mass
        )
        weighted_mean_lower = weighted_mean - z * weighted_error
        weighted_gain_lower = weighted_gain - z * weighted_gain_error
        conditions = {
            "no_audit_loss_beyond_band": int(np.count_nonzero(loss)) == 0,
            "adjusted_direct_P_gain_lower_positive": gain_lower > 0.0,
            "adjusted_direct_P_mean_lower_positive": direct_mean_lower > 0.0,
            "adjusted_weighted_P_gain_lower_positive": weighted_gain_lower > 0.0,
            "adjusted_weighted_P_mean_lower_positive": weighted_mean_lower > 0.0,
            "importance_coverage_and_replay_pass": prerequisite_checks_passed,
        }
        item = {
            **base,
            "status": "evaluated",
            "audit_paths_in_region": regional_paths,
            "direct_P_audit_paths_in_region": direct_paths,
            "regional_effective_sample_size": effective_size,
            "audit_loss_count": int(np.count_nonzero(loss)),
            "audit_neutral_count": int(np.count_nonzero(~loss & ~gain)),
            "audit_gain_count": int(np.count_nonzero(gain)),
            "minimum_audit_economic_pnl": float(np.min(pnl)),
            "maximum_audit_economic_pnl": float(np.max(pnl)),
            "minimum_audit_pnl_plus_numerical_band": float(np.min(pnl + band)),
            "direct_P_loss_count": direct_loss,
            "direct_P_gain_count": direct_gain,
            "adjusted_direct_P_loss_probability_upper": loss_upper,
            "adjusted_direct_P_gain_probability_lower": gain_lower,
            "direct_P_mean": direct_mean,
            "adjusted_direct_P_mean_lower": direct_mean_lower,
            "weighted_P_gain_probability": weighted_gain,
            "adjusted_weighted_P_gain_probability_lower": weighted_gain_lower,
            "weighted_P_mean": weighted_mean,
            "adjusted_weighted_P_mean_lower": weighted_mean_lower,
            "conditions": conditions,
            "certified": bool(all(conditions.values())),
        }
        (surviving if item["certified"] else rejected).append(item)
        report_progress(query_index)
    return {
        "definition": (
            "maximize the local physical mean over every L1-bounded basket of "
            "all spot and option instruments, subject to nonnegative economic "
            "P&L on every reference neighbor"
        ),
        "searched_query_states": queries,
        "searched_all_query_states_independent_of_individual_screens": True,
        "instruments": int(case.returns.shape[-1]),
        "neighbors": int(neighbors),
        "reference_candidates": reference_candidates,
        "zero_reference_solutions": zero_reference_solutions,
        "multiplicity": {
            "method": "Bonferroni familywise correction",
            "family_size": family_size,
            "adjusted_alpha": adjusted_alpha,
        },
        "surviving": surviving,
        "rejected": rejected,
        "unresolved": unresolved,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_time_separated_local_tail(
    *,
    state: np.ndarray,
    case: DerivedCaseArrays,
    log_path_weights: np.ndarray,
    importance_components: np.ndarray,
    split: PathSplit,
    neighbors: int,
    query_paths_per_split: int,
    instrument_block_size: int,
    seed: int,
    device: torch.device,
    tail_ranks: tuple[int, ...] = (1, 2, 10, 20, 50),
    signal_thresholds: tuple[float, ...] = (2.0, 3.0),
    progress_interval_seconds: float = 30.0,
    certification_prerequisites_passed: bool = False,
    basket_solver_workers: int = 1,
) -> dict[str, Any]:
    """Fit and value a separate conditional law at every trading time.

    Only reference paths enter normalization and nearest-neighbour fitting.
    Queries are drawn from the direct-P component. A reference query excludes
    its complete path; because time is never pooled, this is exactly one row.
    """

    state = np.asarray(state, dtype=np.float32)
    log_path_weights = np.asarray(log_path_weights, dtype=np.float64)
    importance_components = np.asarray(importance_components)
    S, times, features = state.shape
    verify_inp(
        case.returns.shape[:2] == (S, times),
        "conditional state and return geometry differ",
    )
    verify_inp(
        log_path_weights.shape == importance_components.shape == (S,),
        "conditional likelihood arrays have invalid shape",
    )
    verify_inp(
        np.intersect1d(split.reference, split.audit).size == 0
        and split.reference.size + split.audit.size == S,
        "conditional reference/audit split is not disjoint and exhaustive",
    )
    verify_inp(
        basket_solver_workers >= 1,
        "basket solver workers must be positive",
    )
    reference_components = importance_components[split.reference]
    audit_components = importance_components[split.audit]
    reference_direct_rows = np.flatnonzero(reference_components == 0)
    audit_direct_rows = np.flatnonzero(audit_components == 0)
    verify_inp(
        reference_direct_rows.size > 0 and audit_direct_rows.size > 0,
        "conditional valuation requires direct-P queries in both splits",
    )
    training_rows = _sample_paths(
        reference_direct_rows,
        maximum=query_paths_per_split,
        seed=seed + 11,
    )
    audit_rows = _sample_paths(
        audit_direct_rows,
        maximum=query_paths_per_split,
        seed=seed + 29,
    )
    effective_neighbors = min(neighbors, split.reference.size - 1)
    verify_inp(
        effective_neighbors >= 2,
        "conditional valuation requires at least three reference paths",
    )
    effective_tail_ranks = tuple(
        rank for rank in tail_ranks if 2 * rank <= effective_neighbors
    )
    if not effective_tail_ranks:
        effective_tail_ranks = (1,)
    config = LocalTailConfig(
        state_mode="raw",
        neighbors=effective_neighbors,
        instrument_candidate_neighbors=effective_neighbors,
        tail_ranks=effective_tail_ranks,
        signal_thresholds=signal_thresholds,
        query_batch_size=min(query_paths_per_split, 3_584),
        instrument_batch_size=instrument_block_size,
        state_scale_floor=1.0e-6,
    )

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.array(value, dtype=np.float32, order="C", copy=True),
            dtype=torch.float32,
            device=device,
        )

    reports: list[dict[str, Any]] = []
    for time_index in range(times):
        reference_state_raw = state[split.reference, time_index]
        reference_weights = positive_relative_weights(
            log_path_weights[split.reference]
        )
        mean, scale, constant = state_normalization(
            reference_state_raw[:, None, :],
            scale_floor=config.state_scale_floor,
            path_weights=reference_weights,
        )
        reference_state = normalize_state(
            reference_state_raw[:, None, :], mean=mean, scale=scale
        )[:, 0]
        audit_state = normalize_state(
            state[split.audit, time_index][:, None, :],
            mean=mean,
            scale=scale,
        )[:, 0]
        reference_return = case.returns[split.reference, time_index]
        audit_return = case.returns[split.audit, time_index]
        reference_cost = case.costs[split.reference, time_index]
        audit_cost = case.costs[split.audit, time_index]
        reference_band = case.numerical_band[split.reference, time_index]
        audit_band = case.numerical_band[split.audit, time_index]
        reference_log_weight = log_path_weights[split.reference]

        training = _cost_adjusted_local_directions(
            reference_state=tensor(reference_state),
            reference_returns=tensor(reference_return),
            reference_cost=tensor(reference_cost),
            reference_roundoff=tensor(reference_band),
            reference_log_weights=tensor(reference_log_weight),
            query_state=tensor(reference_state[training_rows]),
            query_returns=tensor(reference_return[training_rows]),
            query_cost=tensor(reference_cost[training_rows]),
            query_roundoff=tensor(reference_band[training_rows]),
            config=config,
            split=f"reference-t{time_index}",
            leave_one_out_indices=torch.as_tensor(
                training_rows, dtype=torch.int64, device=device
            ),
            progress_interval_seconds=progress_interval_seconds,
            return_neighbor_indices=True,
        )
        reference_neighbor_indices = np.asarray(
            training.pop("_neighbor_indices"), dtype=np.int64
        )
        reference_neighbor_radius = np.asarray(
            training.pop("_neighbor_radius"), dtype=np.float64
        )
        audit = _cost_adjusted_local_directions(
            reference_state=tensor(reference_state),
            reference_returns=tensor(reference_return),
            reference_cost=tensor(reference_cost),
            reference_roundoff=tensor(reference_band),
            reference_log_weights=tensor(reference_log_weight),
            query_state=tensor(audit_state[audit_rows]),
            query_returns=tensor(audit_return[audit_rows]),
            query_cost=tensor(audit_cost[audit_rows]),
            query_roundoff=tensor(audit_band[audit_rows]),
            config=config,
            split=f"audit-t{time_index}",
            leave_one_out_indices=None,
            progress_interval_seconds=progress_interval_seconds,
        )
        candidate_audit = _audit_frozen_balls(
            candidates=training["screened_candidates"],
            centers=reference_state[training_rows],
            audit_state=audit_state,
            audit_returns=audit_return,
            audit_costs=audit_cost,
            audit_bands=audit_band,
            audit_log_weights=log_path_weights[split.audit],
            audit_components=importance_components[split.audit],
            time_index=time_index,
            family_size=max(1, 48 * times * query_paths_per_split),
            prerequisite_checks_passed=certification_prerequisites_passed,
        )
        local_baskets = _fit_and_audit_local_baskets(
            centers=reference_state[training_rows],
            reference_neighbor_indices=reference_neighbor_indices,
            neighbor_radius=reference_neighbor_radius,
            audit_state=audit_state,
            case=case,
            split=split,
            log_path_weights=log_path_weights,
            importance_components=importance_components,
            time_index=time_index,
            neighbors=config.neighbors,
            family_size=max(1, 48 * times * query_paths_per_split),
            prerequisite_checks_passed=certification_prerequisites_passed,
            progress_interval_seconds=progress_interval_seconds,
            solver_workers=basket_solver_workers,
        )
        reports.append(
            {
                "trading_time_index": time_index,
                "state_normalization": {
                    "reference_only_mean": mean.tolist(),
                    "reference_only_population_scale": scale.tolist(),
                    "constant_feature_indices": np.flatnonzero(constant).tolist(),
                },
                "reference": training,
                "audit": audit,
                "frozen_candidate_region_audit": candidate_audit,
                "local_conditional_baskets": local_baskets,
            }
        )
    return {
        "status": "complete",
        "device": str(device),
        "configuration": asdict(config),
        "state_definition": "raw z_t=(log-spot, surface state)",
        "time_pooling": False,
        "split": {
            "reference_paths": int(split.reference.size),
            "audit_paths": int(split.audit.size),
            "mixture_stratified": True,
            "reference_queries": int(training_rows.size),
            "audit_queries": int(audit_rows.size),
            "queries_from_direct_P_component": True,
            "reference_queries_leave_one_complete_path_out": True,
            "component_counts": split.component_counts,
        },
        "times": reports,
    }


__all__ = ["run_time_separated_local_tail"]
