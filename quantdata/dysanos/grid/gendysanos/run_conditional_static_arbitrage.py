"""Run a held-out local conditional-support arbitrage test by expiry."""

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
    frozen_payoff_cone_lower_bound,
    solve_conditional_support_lp,
    terminal_portfolio_pnl,
    weighted_mean_standard_error,
)
from .expiry_arbitrage import payoff_cone_numerical_band
from .required_runs import RequiredCase, stratified_path_split
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, _load_case


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                allow_nan=False,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_inp(args.training_neighbors > 1, "training-neighbors must exceed one")
    verify_inp(args.minimum_validation_neighbors > 1, "minimum validation count must exceed one")
    verify_inp(args.query_paths > 0, "query-paths must be positive")
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
        loaded.component,
        audit_fraction=0.2,
        seed=case.path_count + 97,
    )
    reference = split.reference
    validation = split.audit
    direct_queries = validation[loaded.component[validation] == 0][: args.query_paths]
    verify_inp(
        direct_queries.size == args.query_paths,
        lambda: (
            f"requested {args.query_paths} direct-P queries, found "
            f"{direct_queries.size}"
        ),
    )

    state_mean = loaded.state[reference].mean(axis=0, dtype=np.float64)
    state_scale = loaded.state[reference].std(axis=0, dtype=np.float64)
    state_scale = np.maximum(state_scale, 1.0e-6)
    reference_state = (
        loaded.state[reference].astype(np.float64) - state_mean
    ) / state_scale
    validation_state = (
        loaded.state[validation].astype(np.float64) - state_mean
    ) / state_scale
    path_weight = np.exp(
        np.maximum(
            loaded.log_path_weight,
            math.log(np.nextafter(np.float64(0.0), np.float64(1.0))),
        )
    )
    reference_trees = [
        cKDTree(reference_state[:, trading_time]) for trading_time in range(10)
    ]
    validation_trees = [
        cKDTree(validation_state[:, trading_time]) for trading_time in range(10)
    ]

    evaluated = 0
    training_candidates = 0
    validation_no_loss = 0
    validated_positive_mean = 0
    exact_payoff_cone_certificates = 0
    unresolved_small_ball = 0
    results: list[dict[str, Any]] = []
    for query_number, query_path in enumerate(direct_queries, start=1):
        for trading_time in range(10):
            query_state = (
                loaded.state[query_path, trading_time].astype(np.float64)
                - state_mean[trading_time]
            ) / state_scale[trading_time]
            distance, local_reference = reference_trees[trading_time].query(
                query_state,
                k=args.training_neighbors,
            )
            reference_neighbors = reference[np.asarray(local_reference).reshape(-1)]
            radius = float(np.max(np.asarray(distance)))
            local_validation = validation_trees[trading_time].query_ball_point(
                query_state,
                radius,
            )
            validation_neighbors = validation[np.asarray(local_validation, dtype=np.int64)]
            validation_neighbors = validation_neighbors[
                validation_neighbors != query_path
            ]
            if validation_neighbors.size < args.minimum_validation_neighbors:
                unresolved_small_ball += 7

            query_spot = float(loaded.spot[query_path, trading_time])
            query_cash_strike = (
                np.exp(
                    loaded.log_strike[
                        query_path, trading_time
                    ].astype(np.float64)
                )
                * query_spot
            )
            for expiry in range(7):
                evaluated += 1
                reference_terminal_spot = (
                    loaded.spot[reference_neighbors, trading_time]
                    + loaded.returns[
                        reference_neighbors, trading_time, expiry, 0
                    ]
                )
                action, margin = solve_conditional_support_lp(
                    reference_terminal_spot,
                    spot=query_spot,
                    option_price=loaded.option_price[
                        query_path, trading_time, expiry
                    ],
                    delta=loaded.delta[query_path, trading_time, expiry],
                    cash_strike=query_cash_strike[expiry],
                    costs=loaded.costs[query_path, trading_time, expiry],
                    hedge=case.hedge,
                )
                tolerance = payoff_cone_numerical_band(
                    action,
                    spot=query_spot,
                    option_price=loaded.option_price[
                        query_path, trading_time, expiry
                    ],
                    delta=loaded.delta[query_path, trading_time, expiry],
                    cash_strike=query_cash_strike[expiry],
                    costs=loaded.costs[query_path, trading_time, expiry],
                    hedge=case.hedge,
                )
                candidate = margin > tolerance
                if candidate:
                    training_candidates += 1
                exact_margin, exact_tail_slope = frozen_payoff_cone_lower_bound(
                    action,
                    spot=query_spot,
                    option_price=loaded.option_price[
                        query_path, trading_time, expiry
                    ],
                    delta=loaded.delta[query_path, trading_time, expiry],
                    cash_strike=query_cash_strike[expiry],
                    costs=loaded.costs[query_path, trading_time, expiry],
                    hedge=case.hedge,
                )
                exact_certificate = candidate and exact_margin > tolerance
                exact_payoff_cone_certificates += int(exact_certificate)

                item: dict[str, Any] = {
                    "query_path": int(query_path),
                    "trading_time_index": trading_time,
                    "expiry_index": expiry,
                    "state_ball_radius": radius,
                    "training_neighbors": int(reference_neighbors.size),
                    "validation_neighbors": int(validation_neighbors.size),
                    "training_margin": margin,
                    "numerical_tolerance": tolerance,
                    "training_candidate": candidate,
                    "action_l1": float(np.abs(action).sum()),
                    "all_terminal_spot_lower_bound": (
                        exact_margin if math.isfinite(exact_margin) else None
                    ),
                    "all_terminal_spot_tail_slope": exact_tail_slope,
                    "all_terminal_spot_tail_slope_tolerance": (
                        32.0
                        * np.finfo(np.float32).eps
                        * max(1.0, float(np.abs(action).sum()))
                    ),
                    "exact_payoff_cone_certificate": exact_certificate,
                }
                if validation_neighbors.size >= args.minimum_validation_neighbors:
                    validation_terminal_spot = (
                        loaded.spot[validation_neighbors, trading_time]
                        + loaded.returns[
                            validation_neighbors, trading_time, expiry, 0
                        ]
                    )
                    validation_pnl = terminal_portfolio_pnl(
                        action,
                        validation_terminal_spot,
                        spot=query_spot,
                        option_price=loaded.option_price[
                            query_path, trading_time, expiry
                        ],
                        delta=loaded.delta[query_path, trading_time, expiry],
                        cash_strike=query_cash_strike[expiry],
                        costs=loaded.costs[query_path, trading_time, expiry],
                        hedge=case.hedge,
                    )
                    proposal_stats = weighted_mean_standard_error(
                        validation_pnl,
                        path_weight[validation_neighbors],
                    )
                    direct_mask = loaded.component[validation_neighbors] == 0
                    direct_stats = (
                        weighted_mean_standard_error(validation_pnl[direct_mask])
                        if np.sum(direct_mask) > 1
                        else None
                    )
                    loss_count = int(np.sum(validation_pnl < -tolerance))
                    gain_count = int(np.sum(validation_pnl > tolerance))
                    no_loss = candidate and loss_count == 0 and gain_count > 0
                    positive_mean = (
                        no_loss
                        and float(proposal_stats["mean"])
                        - 3.0 * float(proposal_stats["standard_error"])
                        > 0.0
                    )
                    validation_no_loss += int(no_loss)
                    validated_positive_mean += int(positive_mean)
                    item["validation"] = {
                        "loss_count": loss_count,
                        "gain_count": gain_count,
                        "proposal_weighted_P": proposal_stats,
                        "direct_P": direct_stats,
                        "direct_P_observations": int(np.sum(direct_mask)),
                        "no_observed_loss": no_loss,
                        "mean_above_three_standard_errors": positive_mean,
                    }
                if candidate:
                    item["action"] = action.tolist()
                    results.append(item)
        if query_number % 10 == 0 or query_number == direct_queries.size:
            print(
                f"[{case.generator}] conditional-support queries "
                f"{query_number}/{direct_queries.size}: training candidates="
                f"{training_candidates:,}, validation no-loss={validation_no_loss:,}, "
                f"mean>3se={validated_positive_mean:,}",
                flush=True,
            )

    report = {
        "status": "complete",
        "experiment": "conditional static arbitrage support test (one expiry)",
        "configuration": asdict(case),
        "method": {
            "query_states": "direct-P validation paths",
            "state_metric": (
                "Euclidean distance after per-trading-time, training-only "
                "coordinate normalization"
            ),
            "ball": (
                f"radius of the {args.training_neighbors}th training neighbor; "
                "all disjoint validation states inside the frozen radius are audited"
            ),
            "neighbor_choice": (
                "1,024 proposal neighbours target about 100 direct-P training "
                "observations under the 10% direct-P mixture while retaining an "
                "independent validation ball; this is more stable than choosing "
                "the 100 nearest proposal observations"
            ),
            "portfolio": (
                "one matched spot and 20 calls of one expiry; L1 budget one; "
                "minimum P&L maximized over training-neighbor terminal spots"
            ),
            "warning": (
                "finite conditional-support evidence is not a proof over every "
                "terminal spot and is distinct from the exact static arbitrage test"
            ),
        },
        "query_paths": int(direct_queries.size),
        "evaluated_query_time_expiry_cells": evaluated,
        "training_candidate_count": training_candidates,
        "validation_no_observed_loss_count": validation_no_loss,
        "validation_positive_mean_count": validated_positive_mean,
        "exact_payoff_cone_certificate_count": exact_payoff_cone_certificates,
        "unresolved_small_validation_ball_count": unresolved_small_ball,
        "candidates": results,
    }
    output = Path(args.output_root) / (
        f"conditional-static-{case.generator}-{case.trading_frequency}-"
        f"{case.path_count}-{case.hedge}-{case.cost_rate:.0e}-"
        f"k{args.training_neighbors}.json"
    )
    _atomic_json(output, report)
    print(f"Wrote {output}", flush=True)
    return report


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
    parser.add_argument("--query-paths", type=int, default=100)
    parser.add_argument("--training-neighbors", type=int, default=1_024)
    parser.add_argument("--minimum-validation-neighbors", type=int, default=150)
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--output-root",
        default="quantdata/dysanos/grid/gendysanos/reports/conditional-static-arbitrage",
    )
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
