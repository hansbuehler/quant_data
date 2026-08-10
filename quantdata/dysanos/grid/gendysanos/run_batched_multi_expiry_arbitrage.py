"""Run batched multi-expiry static and local conditional-portfolio audits."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from cdxcore.err import verify_inp
from scipy.spatial import cKDTree

from .conditional_static_arbitrage import weighted_mean_standard_error
from .multi_expiry_arbitrage import (
    multi_expiry_lower_bound,
    multi_expiry_numerical_band,
    multi_expiry_scenario_returns,
    multi_expiry_terminal_pnl,
    observation_numerical_band,
    solve_local_multi_expiry_portfolio,
    solve_multi_expiry_static_arbitrage_lp,
    weighted_quantile,
)
from .required_runs import RequiredCase, positive_relative_weights
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, LoadedExpiryCase, _load_case

AuditName = Literal["statewise", "local", "both"]


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False, default=_json_default),
            encoding="utf-8",
        )
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                # Network shares may briefly retain a lock after a large
                # checkpoint has been read.
                time.sleep(0.25)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_batches(
    component: np.ndarray,
    *,
    batches: int,
    batch_paths: int,
    seed: int,
) -> np.ndarray:
    """Select reproducible disjoint path batches without breaking path clusters."""

    component = np.asarray(component)
    requested = batches * batch_paths
    verify_inp(requested <= component.size, "audit batches exceed the simulated universe")
    rng = np.random.default_rng(seed)
    selected = rng.permutation(component.size)[:requested]
    result = selected.reshape(batches, batch_paths)
    assert np.unique(result).size == requested
    return result


def _entry_primitives(
    loaded: LoadedExpiryCase,
    *,
    path: int,
    trading_time: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert loaded.spot is not None
    assert loaded.option_price is not None
    assert loaded.delta is not None
    assert loaded.log_strike is not None
    spot = float(loaded.spot[path, trading_time])
    option_price = loaded.option_price[path, trading_time].astype(np.float64)
    delta = loaded.delta[path, trading_time].astype(np.float64)
    cash_strike = (
        np.exp(loaded.log_strike[path, trading_time].astype(np.float64)) * spot
    )
    costs = loaded.costs[path, trading_time].astype(np.float64)
    return spot, option_price, delta, cash_strike, costs


def _terminal_spots(
    loaded: LoadedExpiryCase,
    paths: np.ndarray,
    *,
    trading_time: int,
) -> np.ndarray:
    assert loaded.spot is not None
    paths = np.asarray(paths, dtype=np.int64)
    return (
        loaded.spot[paths, trading_time, None].astype(np.float64)
        + loaded.returns[paths, trading_time, :, 0].astype(np.float64)
    )


def _state_geometry(
    loaded: LoadedExpiryCase,
    paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[cKDTree], np.ndarray]:
    """Normalize by the supplied paths and build one same-time tree per timestamp."""

    paths = np.asarray(paths, dtype=np.int64)
    state_mean = loaded.state[paths].mean(axis=0, dtype=np.float64)
    state_scale = loaded.state[paths].std(axis=0, dtype=np.float64)
    state_scale = np.maximum(state_scale, 1.0e-6)
    normalized = (loaded.state.astype(np.float64) - state_mean) / state_scale
    trees = [cKDTree(normalized[paths, trading_time]) for trading_time in range(10)]
    return state_mean, state_scale, trees, normalized


def _weighted_stats_with_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    result = weighted_mean_standard_error(values, weights)
    # A deterministic positive payoff has zero estimated standard error and
    # hence an infinite mean/SE ratio.  Significance is evaluated directly
    # from the finite mean and standard error below, so do not serialize this
    # redundant non-finite ratio in the strict-JSON audit report.
    result.pop("mean_over_standard_error")
    result.update(
        {
            "p10": weighted_quantile(values, weights, 0.10),
            "median": weighted_quantile(values, weights, 0.50),
            "p90": weighted_quantile(values, weights, 0.90),
        }
    )
    return result


def _conditional_diagnostic(
    loaded: LoadedExpiryCase,
    *,
    query_path: int,
    trading_time: int,
    action: np.ndarray,
    state_tree: cKDTree,
    normalized_state: np.ndarray,
    neighbor_paths: np.ndarray,
    neighbors: int,
    hedge: Literal["raw", "delta"],
) -> dict[str, Any]:
    """Estimate conditional P&L in a ball formed from the complete universe."""

    k = min(neighbors + 1, neighbor_paths.size)
    distance, local = state_tree.query(
        normalized_state[query_path, trading_time],
        k=k,
    )
    local = np.asarray(local, dtype=np.int64).reshape(-1)
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    selected = neighbor_paths[local]
    keep = selected != query_path
    selected = selected[keep][:neighbors]
    kept_distance = distance[keep][:neighbors]
    verify_inp(selected.size > 1, "conditional state ball is empty")
    spot, option_price, delta, cash_strike, costs = _entry_primitives(
        loaded,
        path=query_path,
        trading_time=trading_time,
    )
    terminal_spot = _terminal_spots(loaded, selected, trading_time=trading_time)
    pnl = multi_expiry_terminal_pnl(
        action,
        terminal_spot,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    scenario_return = multi_expiry_scenario_returns(
        terminal_spot,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        hedge=hedge,
    ).reshape(selected.size, -1)
    band = observation_numerical_band(action, scenario_return, costs.reshape(-1))
    weight = positive_relative_weights(loaded.log_path_weight[selected])
    weighted = _weighted_stats_with_quantiles(pnl, weight)
    direct = loaded.component[selected] == 0
    direct_stats = (
        _weighted_stats_with_quantiles(
            pnl[direct],
            np.ones(int(np.sum(direct)), dtype=np.float64),
        )
        if np.sum(direct) > 1
        else None
    )
    weighted_significant = float(weighted["mean"]) > 3.0 * float(
        weighted["standard_error"]
    )
    direct_significant = (
        direct_stats is not None
        and float(direct_stats["mean"])
        > 3.0 * float(direct_stats["standard_error"])
    )
    return {
        "neighbors": int(selected.size),
        "radius": float(np.max(kept_distance)),
        "direct_P_neighbors": int(np.sum(direct)),
        "importance_weighted_P": weighted,
        "direct_P": direct_stats,
        "loss_count_beyond_numerical_band": int(np.sum(pnl < -band)),
        "gain_count_beyond_numerical_band": int(np.sum(pnl > band)),
        "importance_mean_above_three_standard_errors": weighted_significant,
        "direct_P_mean_above_three_standard_errors": direct_significant,
        "both_means_above_three_standard_errors": (
            weighted_significant and direct_significant
        ),
    }


def _optional_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _batch_distribution(
    batches: list[dict[str, Any]],
    key: str,
) -> dict[str, float | int] | None:
    values = np.asarray(
        [batch[key] for batch in batches if batch[key] is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return None
    return {
        "batches": int(values.size),
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def run_statewise(
    loaded: LoadedExpiryCase,
    *,
    case: RequiredCase,
    batches: np.ndarray,
    diagnostic_neighbors: int,
    output_root: Path,
) -> dict[str, Any]:
    """Run the Statewise multi-expiry static-arbitrage LP test."""

    direct_paths = np.flatnonzero(loaded.component == 0)
    _, _, full_trees, normalized = _state_geometry(loaded, direct_paths)
    all_paths = np.arange(case.path_count, dtype=np.int64)
    # Trees use every simulated state; direct-P scaling prevents the widened
    # proposal component from defining the economic distance metric.
    full_trees = [cKDTree(normalized[:, trading_time]) for trading_time in range(10)]
    started = time.monotonic()
    batch_reports: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(batches):
        strict_count = 0
        candidate_count = 0
        significant_count = 0
        direct_candidate_count = 0
        lower_bounds: list[float] = []
        actual_medians: list[float] = []
        candidate_weight = 0.0
        batch_weight = positive_relative_weights(loaded.log_path_weight[batch])
        for path_number, path in enumerate(batch, start=1):
            for trading_time in range(10):
                spot, option_price, delta, cash_strike, costs = _entry_primitives(
                    loaded,
                    path=int(path),
                    trading_time=trading_time,
                )
                result = solve_multi_expiry_static_arbitrage_lp(
                    spot=spot,
                    option_price=option_price,
                    delta=delta,
                    cash_strike=cash_strike,
                    costs=costs,
                    hedge=case.hedge,
                )
                strict_count += int(result.strict_certificate)
                if not result.static_arbitrage_candidate:
                    continue
                candidate_count += 1
                direct_candidate_count += int(loaded.component[path] == 0)
                candidate_weight += float(batch_weight[path_number - 1])
                lower_bounds.append(result.lower_bound)
                diagnostic = _conditional_diagnostic(
                    loaded,
                    query_path=int(path),
                    trading_time=trading_time,
                    action=result.action,
                    state_tree=full_trees[trading_time],
                    normalized_state=normalized,
                    neighbor_paths=all_paths,
                    neighbors=diagnostic_neighbors,
                    hedge=case.hedge,
                )
                significant = bool(
                    diagnostic["both_means_above_three_standard_errors"]
                )
                significant_count += int(significant)
                actual_medians.append(
                    float(diagnostic["importance_weighted_P"]["median"])
                )
                opportunities.append(
                    {
                        "batch": batch_number,
                        "path": int(path),
                        "proposal_component": int(loaded.component[path]),
                        "trading_time_index": trading_time,
                        "strict_certificate": result.strict_certificate,
                        "complete_support_lower_bound": result.lower_bound,
                        "numerical_band": result.numerical_band,
                        "deterministic_positive_payoff_witness": result.witness_value,
                        "action_l1": float(np.abs(result.action).sum()),
                        "action": result.action.tolist(),
                        "conditional_expected_return_diagnostic": diagnostic,
                    }
                )
            if path_number % 100 == 0 or path_number == batch.size:
                print(
                    f"[{case.generator}] statewise batch {batch_number + 1}/"
                    f"{batches.shape[0]} paths {path_number:,}/{batch.size:,}: "
                    f"candidates={candidate_count:,}, significant={significant_count:,}",
                    flush=True,
                )
        tested = int(batch.size * 10)
        batch_reports.append(
            {
                "batch": batch_number,
                "paths": int(batch.size),
                "direct_P_paths": int(np.sum(loaded.component[batch] == 0)),
                "tested_state_time_pairs": tested,
                "strict_certificate_count": strict_count,
                "static_arbitrage_candidate_count": candidate_count,
                "static_arbitrage_candidate_fraction": candidate_count / tested,
                "direct_P_candidate_count": direct_candidate_count,
                "self_normalized_P_candidate_fraction": (
                    candidate_weight / float(batch_weight.sum()) / 10.0
                ),
                "conditional_mean_significant_count": significant_count,
                "median_complete_support_lower_bound": _optional_median(lower_bounds),
                "median_conditional_actual_PnL": _optional_median(actual_medians),
            }
        )
        partial = {
            "status": "running",
            "test_name": "Statewise multi-expiry static-arbitrage LP test",
            "configuration": asdict(case),
            "completed_batches": batch_number + 1,
            "batch_results": batch_reports,
            "opportunities": opportunities,
        }
        _atomic_json(
            output_root
            / (
                f"statewise-multi-expiry-{case.generator}-{case.trading_frequency}-"
                f"{case.path_count}-{case.hedge}-{case.cost_rate:.0e}.json"
            ),
            partial,
        )

    distribution_keys = (
        "static_arbitrage_candidate_count",
        "static_arbitrage_candidate_fraction",
        "self_normalized_P_candidate_fraction",
        "conditional_mean_significant_count",
        "median_complete_support_lower_bound",
        "median_conditional_actual_PnL",
    )
    report = {
        "status": "complete",
        "test_name": "Statewise multi-expiry static-arbitrage LP test",
        "configuration": asdict(case),
        "method": {
            "portfolio": (
                "one global-L1 portfolio of seven matched spots and 140 calls, "
                "entered at one state and carried without rebalancing to the seven expiries"
            ),
            "complete_support": (
                "non-loss is imposed at zero, every cash strike, and the upper tail "
                "for each expiry; all expiry cashflows use a common zero-rate value date"
            ),
            "strict_positivity": (
                "after exact non-loss, a fixed positive functional over every kink "
                "and an upper-tail probe must exceed the propagated numerical band"
            ),
            "conditional_diagnostic": (
                f"the {diagnostic_neighbors} nearest full-universe states in direct-P "
                "normalized (s,h); direct-P and p/q-weighted physical statistics"
            ),
        },
        "audit_paths": int(batches.size),
        "batches": int(batches.shape[0]),
        "paths_per_batch": int(batches.shape[1]),
        "batch_results": batch_reports,
        "across_batch_distributions": {
            key: _batch_distribution(batch_reports, key) for key in distribution_keys
        },
        "total_candidates": int(
            sum(batch["static_arbitrage_candidate_count"] for batch in batch_reports)
        ),
        "total_conditionally_significant": int(
            sum(batch["conditional_mean_significant_count"] for batch in batch_reports)
        ),
        "elapsed_seconds": time.monotonic() - started,
        "opportunities": opportunities,
    }
    output = output_root / (
        f"statewise-multi-expiry-{case.generator}-{case.trading_frequency}-"
        f"{case.path_count}-{case.hedge}-{case.cost_rate:.0e}.json"
    )
    _atomic_json(output, report)
    print(f"Wrote {output}", flush=True)
    return report


def _local_validation(
    loaded: LoadedExpiryCase,
    *,
    query_path: int,
    trading_time: int,
    action: np.ndarray,
    validation_paths: np.ndarray,
    hedge: Literal["raw", "delta"],
) -> dict[str, Any]:
    spot, option_price, delta, cash_strike, costs = _entry_primitives(
        loaded,
        path=query_path,
        trading_time=trading_time,
    )
    terminal_spot = _terminal_spots(
        loaded,
        validation_paths,
        trading_time=trading_time,
    )
    scenario_return = multi_expiry_scenario_returns(
        terminal_spot,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        hedge=hedge,
    ).reshape(validation_paths.size, -1)
    flat_cost = costs.reshape(-1)
    flat_action = action.reshape(-1)
    pnl = scenario_return @ flat_action - flat_cost @ np.abs(flat_action)
    band = observation_numerical_band(flat_action, scenario_return, flat_cost)
    weight = positive_relative_weights(loaded.log_path_weight[validation_paths])
    weighted = _weighted_stats_with_quantiles(pnl, weight)
    direct = loaded.component[validation_paths] == 0
    direct_stats = (
        _weighted_stats_with_quantiles(
            pnl[direct],
            np.ones(int(np.sum(direct)), dtype=np.float64),
        )
        if np.sum(direct) > 1
        else None
    )
    no_loss = not np.any(pnl < -band) and np.any(pnl > band)
    weighted_significant = float(weighted["mean"]) > 3.0 * float(
        weighted["standard_error"]
    )
    direct_significant = (
        direct_stats is not None
        and float(direct_stats["mean"])
        > 3.0 * float(direct_stats["standard_error"])
    )
    return {
        "paths": int(validation_paths.size),
        "direct_P_paths": int(np.sum(direct)),
        "loss_count_beyond_numerical_band": int(np.sum(pnl < -band)),
        "gain_count_beyond_numerical_band": int(np.sum(pnl > band)),
        "no_observed_loss_and_some_gain": no_loss,
        "importance_weighted_P": weighted,
        "direct_P": direct_stats,
        "importance_mean_above_three_standard_errors": weighted_significant,
        "direct_P_mean_above_three_standard_errors": direct_significant,
        "confirmed_chain": no_loss and weighted_significant and direct_significant,
    }


def run_local(
    loaded: LoadedExpiryCase,
    *,
    case: RequiredCase,
    batches: np.ndarray,
    training_neighbors: int,
    minimum_validation_neighbors: int,
    output_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the ten-fold Local conditional-portfolio test."""

    started = time.monotonic()
    audit_paths = batches.reshape(-1)
    output = output_root / (
        f"local-conditional-portfolio-{case.generator}-{case.trading_frequency}-"
        f"{case.path_count}-{case.hedge}-{case.cost_rate:.0e}.json"
    )
    batch_reports: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    completed_folds = 0
    previous_elapsed = 0.0
    if resume and output.exists():
        with output.open("r", encoding="utf-8") as stream:
            previous = json.load(stream)
        verify_inp(previous.get("status") == "running", "resume report is not partial")
        verify_inp(
            previous.get("configuration") == asdict(case),
            "resume report configuration does not match",
        )
        completed_folds = int(previous["completed_folds"])
        verify_inp(
            0 <= completed_folds <= batches.shape[0],
            "resume report has an invalid completed-fold count",
        )
        batch_reports = list(previous["batch_results"])
        candidates = list(previous["candidates"])
        previous_elapsed = float(previous.get("elapsed_seconds", 0.0))
        print(
            f"[{case.generator}] resuming local audit after fold "
            f"{completed_folds}/{batches.shape[0]}",
            flush=True,
        )
    for held_out in range(completed_folds, batches.shape[0]):
        validation = batches[held_out]
        training = np.concatenate(
            [batches[index] for index in range(batches.shape[0]) if index != held_out]
        )
        state_mean = loaded.state[training].mean(axis=0, dtype=np.float64)
        state_scale = np.maximum(
            loaded.state[training].std(axis=0, dtype=np.float64),
            1.0e-6,
        )
        normalized_training = (
            loaded.state[training].astype(np.float64) - state_mean
        ) / state_scale
        normalized_validation = (
            loaded.state[validation].astype(np.float64) - state_mean
        ) / state_scale
        training_trees = [
            cKDTree(normalized_training[:, trading_time]) for trading_time in range(10)
        ]
        validation_trees = [
            cKDTree(normalized_validation[:, trading_time]) for trading_time in range(10)
        ]
        training_candidate_count = 0
        validation_populated_count = 0
        validation_no_loss_count = 0
        validation_confirmed_count = 0
        exact_support_count = 0
        training_margins: list[float] = []
        validation_medians: list[float] = []
        for query_number, query_path in enumerate(validation, start=1):
            for trading_time in range(10):
                query_state = normalized_validation[query_number - 1, trading_time]
                distance, local_training = training_trees[trading_time].query(
                    query_state,
                    k=min(training_neighbors, training.size),
                )
                local_training = np.asarray(local_training, dtype=np.int64).reshape(-1)
                radius = float(np.max(np.asarray(distance, dtype=np.float64)))
                training_paths = training[local_training]
                local_validation = validation_trees[trading_time].query_ball_point(
                    query_state,
                    radius,
                )
                validation_paths = validation[
                    np.asarray(local_validation, dtype=np.int64)
                ]
                validation_paths = validation_paths[validation_paths != query_path]

                spot, option_price, delta, cash_strike, costs = _entry_primitives(
                    loaded,
                    path=int(query_path),
                    trading_time=trading_time,
                )
                training_terminal = _terminal_spots(
                    loaded,
                    training_paths,
                    trading_time=trading_time,
                )
                training_return = multi_expiry_scenario_returns(
                    training_terminal,
                    spot=spot,
                    option_price=option_price,
                    delta=delta,
                    cash_strike=cash_strike,
                    hedge=case.hedge,
                ).reshape(training_paths.size, -1)
                fitted = solve_local_multi_expiry_portfolio(
                    training_return,
                    costs.reshape(-1),
                    weights=positive_relative_weights(
                        loaded.log_path_weight[training_paths]
                    ),
                )
                if not fitted.training_candidate:
                    continue
                training_candidate_count += 1
                training_margins.append(fitted.training_minimum_pnl)
                validation_report = None
                if validation_paths.size >= minimum_validation_neighbors:
                    validation_populated_count += 1
                    validation_report = _local_validation(
                        loaded,
                        query_path=int(query_path),
                        trading_time=trading_time,
                        action=fitted.action.reshape(7, 21),
                        validation_paths=validation_paths,
                        hedge=case.hedge,
                    )
                    validation_no_loss_count += int(
                        validation_report["no_observed_loss_and_some_gain"]
                    )
                    validation_confirmed_count += int(
                        validation_report["confirmed_chain"]
                    )
                    validation_medians.append(
                        float(validation_report["importance_weighted_P"]["median"])
                    )

                action = fitted.action.reshape(7, 21)
                exact_lower, _, tail_slope = multi_expiry_lower_bound(
                    action,
                    spot=spot,
                    option_price=option_price,
                    delta=delta,
                    cash_strike=cash_strike,
                    costs=costs,
                    hedge=case.hedge,
                )
                exact_band = multi_expiry_numerical_band(
                    action,
                    spot=spot,
                    option_price=option_price,
                    delta=delta,
                    cash_strike=cash_strike,
                    costs=costs,
                    hedge=case.hedge,
                )
                exact_support = exact_lower >= -exact_band
                exact_support_count += int(exact_support)
                candidates.append(
                    {
                        "held_out_batch": held_out,
                        "query_path": int(query_path),
                        "proposal_component": int(loaded.component[query_path]),
                        "trading_time_index": trading_time,
                        "state_ball_radius": radius,
                        "training_neighbors": int(training_paths.size),
                        "validation_neighbors": int(validation_paths.size),
                        "training_minimum_PnL": fitted.training_minimum_pnl,
                        "training": {
                            "candidate": True,
                            "importance_weighted_P": fitted.training_statistics,
                            "active_constraints": fitted.active_constraints,
                            "constraint_generation_iterations": (
                                fitted.constraint_generation_iterations
                            ),
                            "loss_count_beyond_numerical_band": int(
                                np.sum(fitted.training_pnl < -fitted.training_band)
                            ),
                            "gain_count_beyond_numerical_band": int(
                                np.sum(fitted.training_pnl > fitted.training_band)
                            ),
                        },
                        "validation": validation_report,
                        "complete_support_lower_bound": (
                            exact_lower if math.isfinite(exact_lower) else None
                        ),
                        "complete_support_numerical_band": exact_band,
                        "complete_support_nonloss": exact_support,
                        "complete_support_tail_slope": tail_slope.tolist(),
                        "action_l1": float(np.abs(action).sum()),
                        "action": action.tolist(),
                    }
                )
            if query_number % 100 == 0 or query_number == validation.size:
                print(
                    f"[{case.generator}] local fold {held_out + 1}/"
                    f"{batches.shape[0]} queries {query_number:,}/{validation.size:,}: "
                    f"training={training_candidate_count:,}, "
                    f"validation-confirmed={validation_confirmed_count:,}",
                    flush=True,
                )
        tested = int(validation.size * 10)
        batch_reports.append(
            {
                "held_out_batch": held_out,
                "training_paths": int(training.size),
                "validation_paths": int(validation.size),
                "tested_state_time_pairs": tested,
                "training_candidate_count": training_candidate_count,
                "training_candidate_fraction": training_candidate_count / tested,
                "validation_populated_count": validation_populated_count,
                "validation_no_loss_count": validation_no_loss_count,
                "validation_confirmed_chain_count": validation_confirmed_count,
                "complete_support_nonloss_count": exact_support_count,
                "median_training_minimum_PnL": _optional_median(
                    training_margins
                ),
                "median_validation_actual_PnL": _optional_median(validation_medians),
            }
        )
        _atomic_json(
            output,
            {
                "status": "running",
                "test_name": "Local conditional-portfolio test",
                "configuration": asdict(case),
                "completed_folds": held_out + 1,
                "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                "batch_results": batch_reports,
                "candidates": candidates,
            },
        )

    distribution_keys = (
        "training_candidate_count",
        "training_candidate_fraction",
        "validation_populated_count",
        "validation_no_loss_count",
        "validation_confirmed_chain_count",
        "complete_support_nonloss_count",
        "median_training_minimum_PnL",
        "median_validation_actual_PnL",
    )
    report = {
        "status": "complete",
        "test_name": "Local conditional-portfolio test",
        "configuration": asdict(case),
        "method": {
            "cross_fitting": (
                "ten path-level folds; each held-out batch is queried once and "
                "the other nine batches supply its training neighbours"
            ),
            "state_ball": (
                f"radius of the {training_neighbors}th training neighbour in "
                "training-normalized (s,h); validation uses every held-out path "
                "inside the frozen radius except the query path"
            ),
            "portfolio": (
                "one global-L1 multi-expiry portfolio maximizes the training-ball "
                "importance-weighted mean subject to non-loss on every training neighbour"
            ),
            "reporting": (
                "the complete training and validation chain is retained; a validation "
                "failure does not delete or relabel the training candidate"
            ),
        },
        "audit_paths": int(audit_paths.size),
        "batches": int(batches.shape[0]),
        "paths_per_batch": int(batches.shape[1]),
        "batch_results": batch_reports,
        "across_batch_distributions": {
            key: _batch_distribution(batch_reports, key) for key in distribution_keys
        },
        "total_training_candidates": int(
            sum(batch["training_candidate_count"] for batch in batch_reports)
        ),
        "total_validation_confirmed": int(
            sum(batch["validation_confirmed_chain_count"] for batch in batch_reports)
        ),
        "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        "candidates": candidates,
    }
    _atomic_json(output, report)
    print(f"Wrote {output}", flush=True)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_inp(args.audit in {"statewise", "local", "both"}, "invalid audit name")
    verify_inp(args.batches > 1, "at least two batches are required")
    verify_inp(args.batch_paths > 1, "batch-paths must exceed one")
    verify_inp(args.neighbors > 1, "neighbors must exceed one")
    verify_inp(
        args.minimum_validation_neighbors > 1,
        "minimum-validation-neighbors must exceed one",
    )
    case = RequiredCase(
        generator=args.generator,
        trading_frequency=args.frequency,
        return_horizon="expiry",
        hedge=args.hedge,
        cost_rate=args.cost_rate,
        path_count=args.paths,
    )
    loaded = _load_case(
        case,
        cache_root=Path(args.cache_root),
        need_primitives=True,
    )
    batches = _audit_batches(
        loaded.component,
        batches=args.batches,
        batch_paths=args.batch_paths,
        seed=args.seed,
    )
    output_root = Path(args.output_root)
    result: dict[str, Any] = {}
    if args.audit in {"statewise", "both"}:
        result["statewise"] = run_statewise(
            loaded,
            case=case,
            batches=batches,
            diagnostic_neighbors=args.neighbors,
            output_root=output_root,
        )
    if args.audit in {"local", "both"}:
        result["local"] = run_local(
            loaded,
            case=case,
            batches=batches,
            training_neighbors=args.neighbors,
            minimum_validation_neighbors=args.minimum_validation_neighbors,
            output_root=output_root,
            resume=args.resume,
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"), required=True)
    parser.add_argument("--frequency", choices=("daily", "weekly"), default="weekly")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--hedge", choices=("raw", "delta"), default="delta")
    parser.add_argument(
        "--cost-rate",
        type=float,
        choices=(0.0, 1.0e-4, 1.0e-3),
        default=1.0e-4,
    )
    parser.add_argument("--audit", choices=("statewise", "local", "both"), default="both")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-paths", type=int, default=1_000)
    parser.add_argument("--neighbors", type=int, default=1_024)
    parser.add_argument("--minimum-validation-neighbors", type=int, default=20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching partial local-audit report after its last completed fold",
    )
    parser.add_argument("--seed", type=int, default=82_061)
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--output-root",
        default=(
            "quantdata/dysanos/grid/gendysanos/reports/"
            "batched-multi-expiry-arbitrage"
        ),
    )
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
