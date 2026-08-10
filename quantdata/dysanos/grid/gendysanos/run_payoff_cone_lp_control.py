"""Run the exact one-expiry static arbitrage test at sampled current states."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp
from scipy.spatial import cKDTree

from .conditional_static_arbitrage import (
    terminal_portfolio_pnl,
    weighted_mean_standard_error,
)
from .expiry_arbitrage import payoff_cone_numerical_band, solve_payoff_cone_lp
from .required_runs import (
    RequiredCase,
    clopper_pearson_one_sided,
    stratified_path_split,
)
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, _load_case


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_inp(args.sample_paths > 1, "sample-paths must exceed one")
    verify_inp(
        args.kernel_bandwidth_neighbors > 1,
        "kernel-bandwidth-neighbors must exceed one",
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
    assert loaded.option_price is not None
    assert loaded.delta is not None
    assert loaded.log_strike is not None
    assert loaded.spot is not None
    split = stratified_path_split(
        loaded.component, audit_fraction=0.2, seed=case.path_count + 97
    )
    validation = split.audit
    direct = validation[loaded.component[validation] == 0]
    importance = validation[loaded.component[validation] != 0]
    half = max(1, args.sample_paths // 2)
    selected = np.concatenate((direct[:half], importance[: args.sample_paths - half]))
    cash_strike = np.exp(loaded.log_strike[selected].astype(np.float64)) * loaded.spot[
        selected, :, None, None
    ]
    direct_paths_all = np.flatnonzero(loaded.component == 0)
    direct_spot_trees = [
        cKDTree(np.log(loaded.spot[direct_paths_all, trading_time, None]))
        for trading_time in range(10)
    ]
    direct_log_spot = np.log(loaded.spot[direct_paths_all].astype(np.float64))
    proposal_log_spot = np.log(loaded.spot.astype(np.float64))
    proposal_paths = np.arange(case.path_count)
    counts = np.zeros((10, 7), dtype=np.int64)
    direct_counts = np.zeros((10, 7), dtype=np.int64)
    importance_counts = np.zeros((10, 7), dtype=np.int64)
    maximum = np.full((10, 7), -np.inf)
    evaluated = np.zeros((10, 7), dtype=np.int64)
    top: list[dict[str, Any]] = []
    for local_path, path in enumerate(selected):
        for trading_time in range(10):
            for expiry in range(7):
                action, margin = solve_payoff_cone_lp(
                    spot=float(loaded.spot[path, trading_time]),
                    option_price=loaded.option_price[path, trading_time, expiry],
                    delta=loaded.delta[path, trading_time, expiry],
                    cash_strike=cash_strike[local_path, trading_time, expiry],
                    costs=loaded.costs[path, trading_time, expiry],
                    hedge=case.hedge,
                )
                tolerance = payoff_cone_numerical_band(
                    action,
                    spot=float(loaded.spot[path, trading_time]),
                    option_price=loaded.option_price[path, trading_time, expiry],
                    delta=loaded.delta[path, trading_time, expiry],
                    cash_strike=cash_strike[local_path, trading_time, expiry],
                    costs=loaded.costs[path, trading_time, expiry],
                    hedge=case.hedge,
                )
                evaluated[trading_time, expiry] += 1
                maximum[trading_time, expiry] = max(
                    maximum[trading_time, expiry], margin
                )
                if margin > tolerance:
                    counts[trading_time, expiry] += 1
                    if loaded.component[path] == 0:
                        direct_counts[trading_time, expiry] += 1
                    else:
                        importance_counts[trading_time, expiry] += 1
                    query_spot = float(loaded.spot[path, trading_time])
                    direct_k = min(args.kernel_bandwidth_neighbors + 1, direct_paths_all.size)
                    direct_distance, _ = direct_spot_trees[trading_time].query(
                        [[math.log(query_spot)]], k=direct_k
                    )
                    direct_distance = np.asarray(direct_distance).reshape(-1)
                    positive_distance = direct_distance[direct_distance > 0.0]
                    bandwidth = float(
                        positive_distance[-1]
                        if positive_distance.size
                        else np.finfo(np.float64).eps
                    )
                    bandwidth = max(bandwidth, np.finfo(np.float64).eps)

                    def conditional_pnl(neighbors: np.ndarray) -> np.ndarray:
                        terminal_spot = (
                            loaded.spot[neighbors, trading_time]
                            + loaded.returns[neighbors, trading_time, expiry, 0]
                        )
                        return terminal_portfolio_pnl(
                            action,
                            terminal_spot,
                            spot=query_spot,
                            option_price=loaded.option_price[path, trading_time, expiry],
                            delta=loaded.delta[path, trading_time, expiry],
                            cash_strike=np.exp(
                                loaded.log_strike[path, trading_time, expiry].astype(
                                    np.float64
                                )
                            )
                            * query_spot,
                            costs=loaded.costs[path, trading_time, expiry],
                            hedge=case.hedge,
                        )

                    direct_pnl = conditional_pnl(direct_paths_all)
                    proposal_pnl = conditional_pnl(proposal_paths)
                    direct_kernel = np.exp(
                        -0.5
                        * np.square(
                            (
                                direct_log_spot[:, trading_time]
                                - math.log(query_spot)
                            )
                            / bandwidth
                        )
                    )
                    proposal_log_weight = (
                        loaded.log_path_weight
                        - 0.5
                        * np.square(
                            (
                                proposal_log_spot[:, trading_time]
                                - math.log(query_spot)
                            )
                            / bandwidth
                        )
                    )
                    proposal_weight = np.exp(
                        proposal_log_weight - float(np.max(proposal_log_weight))
                    )
                    opportunity = {
                            "path": int(path),
                            "proposal_component": int(loaded.component[path]),
                            "trading_time_index": trading_time,
                            "expiry_index": expiry,
                            "margin": margin,
                            "numerical_tolerance": float(tolerance),
                            "action_l1": float(np.abs(action).sum()),
                            "spot_action": float(action[0]),
                            "option_action": action[1:].tolist(),
                            "conditional_on_entry_spot": {
                                "entry_spot": query_spot,
                                "log_spot_gaussian_kernel_bandwidth": bandwidth,
                                "bandwidth_defined_by_direct_P_neighbor": int(
                                    args.kernel_bandwidth_neighbors
                                ),
                                "direct_P": {
                                    **weighted_mean_standard_error(
                                        direct_pnl,
                                        direct_kernel,
                                    ),
                                },
                                "importance_weighted_P": {
                                    **weighted_mean_standard_error(
                                        proposal_pnl,
                                        proposal_weight,
                                    ),
                                },
                                "estimator": (
                                    "Nadaraya-Watson Gaussian-kernel conditional mean over "
                                    "all terminal-spot observations; the direct-P sample uses "
                                    "kernel weights and the full proposal sample uses kernel "
                                    "times p/q weights"
                                ),
                            },
                        }
                    top.append(opportunity)
        if (local_path + 1) % 100 == 0 or local_path + 1 == selected.size:
            print(
                f"[{case.generator}/{case.path_count}] LP-evaluated "
                f"{local_path + 1:,}/{selected.size:,} paths; positive "
                f"state-expiry pairs={int(counts.sum()):,} "
                f"(direct-P={int(direct_counts.sum()):,})",
                flush=True,
            )
    top.sort(key=lambda item: item["margin"], reverse=True)
    direct_paths = int(np.sum(loaded.component[selected] == 0))
    direct_trials = direct_paths * 10 * 7
    direct_positive = int(direct_counts.sum())
    cell_alpha = 0.05 / (10 * 7)
    cell_probability_lower = np.zeros((10, 7), dtype=np.float64)
    cell_probability_upper = np.ones((10, 7), dtype=np.float64)
    for trading_time in range(10):
        for expiry in range(7):
            lower, upper = clopper_pearson_one_sided(
                int(direct_counts[trading_time, expiry]),
                direct_paths,
                alpha=cell_alpha,
            )
            cell_probability_lower[trading_time, expiry] = lower
            cell_probability_upper[trading_time, expiry] = upper
    report = {
        "status": "complete",
        "experiment": "static arbitrage test (one expiry)",
        "test_name": "static arbitrage test",
        "configuration": asdict(case),
        "sample_paths": int(selected.size),
        "sample_direct_P_paths": int(np.sum(loaded.component[selected] == 0)),
        "sample_importance_paths": int(np.sum(loaded.component[selected] != 0)),
        "evaluated_state_expiry_pairs": int(evaluated.sum()),
        "positive_margin_count": int(counts.sum()),
        "direct_P_positive_margin_count": direct_positive,
        "importance_positive_margin_count": int(importance_counts.sum()),
        "direct_P_state_expiry_pairs": direct_trials,
        "direct_P_paths_are_independent_observations": True,
        "cellwise_familywise_alpha": 0.05,
        "cellwise_alpha": cell_alpha,
        "direct_P_confirmed_time_expiry_cells": int(
            np.sum(cell_probability_lower > 0.0)
        ),
        "maximum_direct_P_cell_probability_lower": float(
            np.max(cell_probability_lower)
        ),
        "positive_margin_count_by_time_expiry": counts.tolist(),
        "direct_P_positive_margin_count_by_time_expiry": direct_counts.tolist(),
        "importance_positive_margin_count_by_time_expiry": importance_counts.tolist(),
        "direct_P_probability_lower_by_time_expiry": cell_probability_lower.tolist(),
        "direct_P_probability_upper_by_time_expiry": cell_probability_upper.tolist(),
        "maximum_margin_by_time_expiry": maximum.tolist(),
        "top_opportunities": top[:20],
        "interpretation": (
            "at each sampled current state the LP checks every terminal spot; "
            "a positive margin certifies a one-expiry static arbitrage and does "
            "not use the path's realized future return"
        ),
    }
    label = (
        f"lp-control-{case.generator}-{case.trading_frequency}-{case.path_count}-"
        f"expiry-{case.hedge}-{case.cost_rate:.0e}-{selected.size}paths.json"
    )
    output = Path(args.output_root) / label
    _atomic_json(output, report)
    print(
        f"{case.generator}: {report['positive_margin_count']:,}/"
        f"{report['evaluated_state_expiry_pairs']:,} state-expiry pairs have a positive "
        f"static-arbitrage margin; report={output}",
        flush=True,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"), required=True)
    parser.add_argument("--frequency", choices=("daily", "weekly"), default="weekly")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--hedge", choices=("raw", "delta"), default="delta")
    parser.add_argument("--cost-rate", type=float, choices=(0.0, 1.0e-4, 1.0e-3), default=1.0e-4)
    parser.add_argument("--sample-paths", type=int, default=100)
    parser.add_argument("--kernel-bandwidth-neighbors", type=int, default=1024)
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--output-root",
        default="quantdata/dysanos/grid/gendysanos/reports/expiry-arbitrage",
    )
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
