"""Replay a learned arb-tracing portfolio from one exact starting state."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.verbose import Context
from scipy.spatial import cKDTree

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig

from .arb_tracing import ArbTracingConfig, OrthogonalArbDirection
from .required_runs import (
    RequiredCase,
    clopper_pearson_one_sided,
    positive_relative_weights,
    stratified_path_split,
)
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, _load_case
from .terminal_universe import expiry_step_offsets

_DEFAULT_TRACE_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/arb-tracing"
)
_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/conditional-arb-replay"
)


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


def _model_config(value: dict[str, Any]) -> ArbTracingConfig:
    fields = dict(value)
    fields["network_config"] = FeedForwardConfig(**fields["network_config"])
    return ArbTracingConfig(**fields)


@torch.no_grad()
def _frozen_actions(
    report: dict[str, Any],
    state: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the final sum of frozen stage directions."""

    verify_inp(
        state.ndim == 3 and state.shape[1] == 10,
        lambda: f"candidate states must have shape (S,10,F), got {state.shape}",
    )
    env = BaseEnvironment(precision=32, device=device, debug=False)
    tensor_state = torch.as_tensor(state, dtype=env.fdtype, device=env.device)
    total = None
    first_mean = None
    first_whitener = None
    for stage in report["stages"]:
        config = _model_config(stage["model_config"])
        model = OrthogonalArbDirection(
            env,
            config=config,
            state_mean=torch.zeros(
                config.trading_times,
                config.state_features,
                dtype=env.fdtype,
                device=env.device,
            ),
            state_whitener=torch.eye(
                config.state_features,
                dtype=env.fdtype,
                device=env.device,
            )[None].repeat(config.trading_times, 1, 1),
        )
        checkpoint = torch.load(
            stage["checkpoint"], map_location=env.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        direction, _ = model(tensor_state)
        scale = torch.as_tensor(
            stage["base_update_notional_by_time"],
            dtype=env.fdtype,
            device=env.device,
        )[None, :, None, None]
        contribution = scale * direction
        total = contribution if total is None else total + contribution
        if first_mean is None:
            first_mean = model.state_mean.detach().cpu().numpy()
            first_whitener = model.state_whitener.detach().cpu().numpy()
        del model, checkpoint, direction, contribution
    assert total is not None and first_mean is not None and first_whitener is not None
    result = total.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("reconstructed frozen action is non-finite")
    del tensor_state, total
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result, first_mean, first_whitener


def _portfolio_pnl_band(
    action: np.ndarray,
    returns: np.ndarray,
    costs: np.ndarray,
    instrument_band: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a fixed portfolio and its propagated float32 band."""

    action = np.asarray(action, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    instrument_band = np.asarray(instrument_band, dtype=np.float64)
    verify_inp(
        returns.shape == costs.shape == instrument_band.shape
        and returns.shape[1:] == action.shape,
        lambda: (
            "conditional portfolio geometry differs: "
            f"{action.shape}/{returns.shape}/{costs.shape}/{instrument_band.shape}"
        ),
    )
    absolute = np.abs(action)
    pnl = np.sum(action[None] * returns - absolute[None] * costs, axis=(1, 2))
    band = np.sum(absolute[None] * instrument_band, axis=(1, 2))
    band += 32.0 * np.finfo(np.float32).eps * np.sum(
        absolute[None] * (np.abs(returns) + costs), axis=(1, 2)
    )
    if not np.all(np.isfinite(pnl)) or not np.all(np.isfinite(band)):
        raise FloatingPointError("conditional portfolio valuation is non-finite")
    return pnl, band


def _weighted_statistics(
    pnl: np.ndarray,
    band: np.ndarray,
    log_weights: np.ndarray,
    component: np.ndarray,
) -> dict[str, Any]:
    weights = positive_relative_weights(log_weights)
    normalized = weights / weights.sum()
    loss = pnl < -band
    mean = float(np.dot(normalized, pnl))
    mean_error = float(
        np.sqrt(np.sum(np.square(normalized * (pnl - mean))))
    )
    probability = float(np.dot(normalized, loss))
    probability_error = float(
        np.sqrt(np.sum(np.square(normalized * (loss - probability))))
    )
    direct = component == 0
    return {
        "paths": int(pnl.size),
        "weighted_P_mean": mean,
        "weighted_P_mean_standard_error": mean_error,
        "weighted_P_loss_probability": probability,
        "weighted_P_loss_probability_standard_error": probability_error,
        "proposal_loss_count": int(np.count_nonzero(loss)),
        "direct_P_paths": int(np.count_nonzero(direct)),
        "direct_P_loss_count": int(np.count_nonzero(loss & direct)),
    }


def _select_candidate_state(
    *,
    loaded,
    report: dict[str, Any],
    query_paths: int,
    neighbors: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Choose the strongest locally screened training state deterministically."""

    split = stratified_path_split(
        loaded.component,
        audit_fraction=0.2,
        seed=loaded.state.shape[0] + 97,
    )
    direct_reference = split.reference[loaded.component[split.reference] == 0]
    verify_inp(direct_reference.size > 0, "no direct-P training states are available")
    rng = np.random.default_rng(seed)
    query = np.sort(
        rng.choice(
            direct_reference,
            size=min(query_paths, direct_reference.size),
            replace=False,
        )
    )
    query_action, state_mean, state_whitener = _frozen_actions(
        report, loaded.state[query], device=device
    )
    candidate_times = sorted(
        {
            int(item["trading_time_index"])
            for stage in report["stages"]
            for item in stage["selection"]
            if item.get("training_near_arbitrage_candidate", False)
        }
    )
    verify_inp(
        bool(candidate_times),
        "the source report contains no training near-arbitrage trading time",
    )
    candidates: list[dict[str, Any]] = []
    for time_index in candidate_times:
        reference_state = np.einsum(
            "sf,fg->sg",
            loaded.state[split.reference, time_index].astype(np.float64)
            - state_mean[time_index],
            state_whitener[time_index].astype(np.float64),
        )
        query_state = np.einsum(
            "sf,fg->sg",
            loaded.state[query, time_index].astype(np.float64)
            - state_mean[time_index],
            state_whitener[time_index].astype(np.float64),
        )
        tree = cKDTree(reference_state)
        distance, local = tree.query(
            query_state,
            k=min(neighbors + 1, split.reference.size),
            workers=-1,
        )
        for query_index, path_index in enumerate(query):
            paths = split.reference[np.asarray(local[query_index]).reshape(-1)]
            distances = np.asarray(distance[query_index]).reshape(-1)
            keep = paths != path_index
            paths = paths[keep][:neighbors]
            distances = distances[keep][:neighbors]
            action = query_action[query_index, time_index]
            pnl, band = _portfolio_pnl_band(
                action,
                loaded.returns[paths, time_index],
                loaded.costs[paths, time_index],
                loaded.numerical_band[paths, time_index],
            )
            stats = _weighted_statistics(
                pnl,
                band,
                loaded.log_path_weight[paths],
                loaded.component[paths],
            )
            loss_upper_score = (
                stats["weighted_P_loss_probability"]
                + 3.0 * stats["weighted_P_loss_probability_standard_error"]
            )
            mean_lower = (
                stats["weighted_P_mean"]
                - 3.0 * stats["weighted_P_mean_standard_error"]
            )
            candidates.append(
                {
                    "query_path": int(path_index),
                    "trading_time_index": time_index,
                    "neighbor_radius": float(distances[-1]),
                    "action": action,
                    "action_l1": float(np.abs(action).sum()),
                    "local_statistics": stats,
                    "selection_score": (loss_upper_score, -mean_lower),
                }
            )
    selected = min(candidates, key=lambda item: item["selection_score"])
    selected["states_screened"] = len(candidates)
    selected["candidate_trading_times"] = candidate_times
    selected.pop("selection_score")
    return selected


def _load_physical_sampler(generator: str, *, seed: int):
    """Load the fitted generator and return its direct-P sampler and expiries."""

    if generator == "dysanos":
        from .torch_prep import GenDysanosExperiment, _load_generator_components

        experiment = GenDysanosExperiment(
            num_samples=2,
            num_alpha=None,
            use_matrix_exp=True,
            bandwidth=1.0,
            trading_frequency="weekly",
            return_horizon="expiry",
            num_trading_times=1,
            seed=seed,
            importance_sampling=None,
        )
        grid_config, _, _, sampler = _load_generator_components(
            experiment, verbose=Context.quiet
        )
        expiries = np.asarray(grid_config.grid_expiries, dtype=np.float64)
    else:
        from quantdata.dysanos.implied.decoder import PCATotalVarianceDecoder
        from quantdata.dysanos.implied.experiment import ImpliedPCAExperiment
        from quantdata.dysanos.implied.market import load_fitted_pca
        from quantdata.dysanos.implied.simulator import ImpliedPCASampler

        experiment = ImpliedPCAExperiment(
            num_samples=2,
            num_alpha=None,
            use_matrix_exp=True,
            bandwidth=1.0,
            trading_frequency="weekly",
            return_horizon="expiry",
            num_trading_times=1,
            seed=seed,
            importance_sampling=None,
        )
        _, market, pca, diagnostics = load_fitted_pca(
            experiment.fit_config, verbose=Context.quiet
        )
        PCATotalVarianceDecoder(pca)
        sampler = ImpliedPCASampler(
            pca=pca,
            scores=diagnostics.scores,
            time_y=market.time_y,
            spot_ref=market.spot_ref,
            num_alpha=experiment.num_alpha,
            seed=seed,
            verbose=Context.quiet,
        )
        expiries = np.asarray(pca.expiries, dtype=np.float64)
    sampler.rng = np.random.default_rng(seed)
    return sampler, expiries


def _conditional_delta_pnl(
    terminal_spot: np.ndarray,
    *,
    action: np.ndarray,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    cost_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Value one fixed multi-expiry delta-hedged portfolio."""

    terminal_spot = np.asarray(terminal_spot, dtype=np.float64)
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    expiries, strikes = option_price.shape
    verify_inp(
        terminal_spot.ndim == 2
        and terminal_spot.shape[1] == expiries
        and delta.shape == cash_strike.shape == option_price.shape
        and action.shape == (expiries, strikes + 1),
        "conditional expiry portfolio has inconsistent geometry",
    )
    spot_return = terminal_spot - spot
    option_return = (
        np.maximum(terminal_spot[:, :, None] - cash_strike[None], 0.0)
        - option_price[None]
        - delta[None] * spot_return[:, :, None]
    )
    returns = np.concatenate((spot_return[:, :, None], option_return), axis=2)
    option_cost = cost_rate * (
        np.abs(option_price) + np.abs(delta * spot)
    )
    costs = np.concatenate(
        (
            np.full((expiries, 1), cost_rate * abs(spot)),
            option_cost,
        ),
        axis=1,
    )
    spot_scale = np.abs(terminal_spot) + abs(spot)
    option_scale = (
        np.abs(terminal_spot[:, :, None])
        + np.abs(cash_strike[None])
        + np.abs(option_price[None])
        + np.abs(delta[None])
        * (np.abs(terminal_spot[:, :, None]) + abs(spot))
    )
    scale = np.concatenate((spot_scale[:, :, None], option_scale), axis=2)
    instrument_band = 32.0 * np.finfo(np.float32).eps * (
        scale + costs[None]
    )
    return _portfolio_pnl_band(
        action, returns, np.broadcast_to(costs, returns.shape), instrument_band
    )


def _deterministic_terminal_spot_lower_bound(
    *,
    action: np.ndarray,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    cost_rate: float,
) -> dict[str, Any]:
    """Minimize the frozen P&L over every positive terminal-spot vector."""

    action = np.asarray(action, dtype=np.float64)
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    expiries, strikes = option_price.shape
    verify_inp(
        action.shape == (expiries, strikes + 1)
        and delta.shape == cash_strike.shape == option_price.shape,
        "deterministic terminal-spot test has inconsistent geometry",
    )
    reports: list[dict[str, Any]] = []
    total_lower_bound = 0.0
    joint_minimizer = np.empty(expiries, dtype=np.float64)
    unbounded = False
    for expiry in range(expiries):
        spot_action = action[expiry, 0]
        option_action = action[expiry, 1:]
        option_cost = cost_rate * (
            np.abs(option_price[expiry])
            + np.abs(delta[expiry] * spot)
        )
        fixed_cost = (
            abs(spot_action) * cost_rate * abs(spot)
            + np.dot(np.abs(option_action), option_cost)
        )
        candidates = np.unique(
            np.concatenate((np.asarray([0.0]), cash_strike[expiry]))
        )
        option_return = (
            np.maximum(
                candidates[:, None] - cash_strike[expiry, None], 0.0
            )
            - option_price[expiry, None]
            - delta[expiry, None] * (candidates[:, None] - spot)
        )
        value = (
            spot_action * (candidates - spot)
            + option_return @ option_action
            - fixed_cost
        )
        minimum_index = int(np.argmin(value))
        finite_lower_bound = float(value[minimum_index])
        tail_slope = float(
            spot_action + np.dot(option_action, 1.0 - delta[expiry])
        )
        tail_tolerance = (
            32.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.abs(action[expiry]).sum()))
        )
        tail_unbounded = bool(tail_slope < -tail_tolerance)
        lower_bound = -math.inf if tail_unbounded else finite_lower_bound
        unbounded = unbounded or tail_unbounded
        total_lower_bound = (
            -math.inf
            if unbounded
            else total_lower_bound + finite_lower_bound
        )
        joint_minimizer[expiry] = candidates[minimum_index]
        reports.append(
            {
                "expiry_index": expiry,
                "lower_bound": None if tail_unbounded else finite_lower_bound,
                "minimizing_terminal_spot": float(candidates[minimum_index]),
                "tail_slope": tail_slope,
                "tail_slope_tolerance": tail_tolerance,
                "unbounded_below_in_upper_tail": tail_unbounded,
            }
        )
    joint_pnl, joint_band = _conditional_delta_pnl(
        joint_minimizer[None],
        action=action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        cost_rate=cost_rate,
    )
    return {
        "domain": "all vectors of strictly positive terminal spots; zero is used as the continuous lower-tail limit",
        "method": (
            "separable piecewise-linear minimization at zero and every cash "
            "strike, with a nonnegative upper-tail slope required per expiry"
        ),
        "lower_bound": None if unbounded else total_lower_bound,
        "unbounded_below": unbounded,
        "finite_joint_test_spot": joint_minimizer.tolist(),
        "finite_joint_test_pnl": float(joint_pnl[0]),
        "finite_joint_test_numerical_band": float(joint_band[0]),
        "nonnegative_for_all_terminal_spots": bool(
            not unbounded and total_lower_bound >= -float(joint_band[0])
        ),
        "by_expiry": reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = RequiredCase(
        generator=args.generator,
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="delta",
        cost_rate=1.0e-4,
        path_count=args.source_paths,
    )
    trace_path = Path(args.trace_root) / (
        f"arb-tracing-powerexp-g1e+02-{case.generator}-{case.path_count}.json"
    )
    report = json.loads(trace_path.read_text(encoding="utf-8"))
    verify_inp(report.get("status") == "complete", "arb-tracing report is incomplete")
    loaded = _load_case(
        case, cache_root=Path(args.cache_root), need_primitives=True
    )
    selected = _select_candidate_state(
        loaded=loaded,
        report=report,
        query_paths=args.query_paths,
        neighbors=args.neighbors,
        seed=args.seed + 17,
        device=args.device,
    )
    path_index = selected["query_path"]
    time_index = selected["trading_time_index"]
    action = np.asarray(selected.pop("action"), dtype=np.float64)
    initial_state = loaded.state[path_index, time_index].astype(np.float64)
    spot = float(loaded.spot[path_index, time_index])
    option_price = loaded.option_price[path_index, time_index].astype(np.float64)
    delta = loaded.delta[path_index, time_index].astype(np.float64)
    cash_strike = (
        np.exp(loaded.log_strike[path_index, time_index].astype(np.float64))
        * spot
    )
    cached_cost = loaded.costs[path_index, time_index].astype(np.float64)
    del loaded, report
    gc.collect()

    sampler, expiries = _load_physical_sampler(case.generator, seed=args.seed)
    offsets, expiry_error = expiry_step_offsets(expiries, dt=1.0 / DAYS_PER_YEAR)
    verify_inp(
        abs(math.exp(initial_state[0]) - spot)
        <= 1.0e-5 * max(1.0, abs(spot)),
        lambda: (
            "cached spot and selected log-spot differ: "
            f"{math.exp(initial_state[0])}/{spot}"
        ),
    )
    expected_cost = np.concatenate(
        (
            np.full((expiries.size, 1), case.cost_rate * abs(spot)),
            case.cost_rate
            * (np.abs(option_price) + np.abs(delta * spot)),
        ),
        axis=1,
    )
    verify_inp(
        np.allclose(cached_cost, expected_cost, rtol=2.0e-6, atol=1.0e-8),
        "conditional replay costs differ from cached delta-hedged costs",
    )
    deterministic_test = _deterministic_terminal_spot_lower_bound(
        action=action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        cost_rate=case.cost_rate,
    )

    pnl_values: list[np.ndarray] = []
    band_values: list[np.ndarray] = []
    step_offsets = np.concatenate((np.asarray([0]), offsets))
    for start in range(0, args.conditional_paths, args.block_size):
        stop = min(start + args.block_size, args.conditional_paths)
        state = sampler.sample_conditional_xstates_at_offsets(
            initial_state=initial_state,
            step_offsets=step_offsets,
            num_samples=stop - start,
            dt=1.0 / DAYS_PER_YEAR,
        )
        terminal_spot = np.exp(state[:, 1:, 0])
        pnl, band = _conditional_delta_pnl(
            terminal_spot,
            action=action,
            spot=spot,
            option_price=option_price,
            delta=delta,
            cash_strike=cash_strike,
            cost_rate=case.cost_rate,
        )
        pnl_values.append(pnl)
        band_values.append(band)
        print(
            f"[{case.generator}] conditional continuations "
            f"{stop:,}/{args.conditional_paths:,}; losses="
            f"{sum(np.count_nonzero(x < -y) for x, y in zip(pnl_values, band_values)):,}",
            flush=True,
        )
    pnl = np.concatenate(pnl_values)
    band = np.concatenate(band_values)
    loss = pnl < -band
    gain = pnl > band
    loss_count = int(np.count_nonzero(loss))
    gain_count = int(np.count_nonzero(gain))
    _, loss_upper = clopper_pearson_one_sided(
        loss_count, args.conditional_paths, alpha=0.05
    )
    mean = float(np.mean(pnl))
    mean_error = float(np.std(pnl, ddof=0) / math.sqrt(pnl.size))
    result = {
        "status": "complete",
        "experiment": "exact-state conditional independent-continuation replay",
        "configuration": {
            **asdict(case),
            "conditional_paths": args.conditional_paths,
            "candidate_query_paths": args.query_paths,
            "candidate_neighbors": args.neighbors,
            "seed": args.seed,
            "block_size": args.block_size,
        },
        "method": {
            "candidate_source": (
                "direct-P training states at trading times passing the 100,000-path "
                "power-exp training near-arbitrage screen"
            ),
            "candidate_selection": (
                "minimum local weighted loss upper score, then maximum local "
                "three-standard-error mean lower bound"
            ),
            "portfolio": (
                "sum of all frozen arb-tracing stage directions and selected "
                "notionals; no parameter is refitted"
            ),
            "continuation_law": (
                "100,000 independent exact Gaussian matrix-exponential "
                "continuations under P from the identical raw state x_t"
            ),
            "expiry_sampling": (
                "the exact Gaussian transition is composed directly between "
                "the seven expiry offsets; omitted daily states are integrated out"
            ),
        },
        "source_trace": str(trace_path),
        "selected_state": {
            **selected,
            "raw_x": initial_state.tolist(),
            "spot": spot,
            "action": action.tolist(),
        },
        "expiry_offsets": offsets.tolist(),
        "maximum_expiry_time_error": expiry_error,
        "deterministic_terminal_spot_test": deterministic_test,
        "conditional_test": {
            "paths": args.conditional_paths,
            "loss_count_beyond_float32_band": loss_count,
            "gain_count_beyond_float32_band": gain_count,
            "loss_probability": loss_count / args.conditional_paths,
            "loss_probability_one_sided_95_upper": loss_upper,
            "mean_pnl": mean,
            "mean_pnl_standard_error": mean_error,
            "mean_pnl_lower_three_standard_errors": mean - 3.0 * mean_error,
            "minimum_pnl_minus_band": float(np.min(pnl - band)),
            "pnl_quantiles": {
                str(probability): float(np.quantile(pnl, probability))
                for probability in (0.0, 0.001, 0.01, 0.05, 0.5, 0.95)
            },
            "no_observed_loss": loss_count == 0,
            "positive_mean_above_three_standard_errors": (
                mean - 3.0 * mean_error > 0.0
            ),
            "conditional_near_arbitrage_candidate": (
                loss_count == 0 and mean - 3.0 * mean_error > 0.0
            ),
        },
    }
    output = Path(args.output_root) / (
        f"conditional-arb-replay-{case.generator}-{case.path_count}-"
        f"n{args.conditional_paths}.json"
    )
    _atomic_json(output, result)
    print(f"Wrote {output}", flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"), required=True)
    parser.add_argument("--source-paths", type=int, default=100_000)
    parser.add_argument("--conditional-paths", type=int, default=100_000)
    parser.add_argument("--query-paths", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=1_024)
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2_026_081)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--trace-root", default=str(_DEFAULT_TRACE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
