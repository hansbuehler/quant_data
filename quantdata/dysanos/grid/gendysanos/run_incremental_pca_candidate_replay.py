"""Replay top PCA-IV period candidates from their exact starting states."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.implied.decoder import PCATotalVarianceDecoder
from quantdata.dysanos.implied.experiment import ImpliedPCAExperiment
from quantdata.dysanos.implied.market import load_fitted_pca
from quantdata.dysanos.implied.simulator import ImpliedPCASampler
from quantdata.dysanos.implied.universe import traded_nms_indices

from .required_runs import derive_case_arrays, propagated_basket_band
from .run_incremental_candidate_replay import (
    _conservative_mean_lower,
    _sparse_portfolio,
    recover_candidate_state,
    summarize_conditional_pnl,
)
from .terminal_universe import build_terminal_path_universe, expiry_step_offsets

_DEFAULT_REPORT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/required/"
    "pca-iv-weekly-1000000-incremental-raw-1e-03.json"
)
_DEFAULT_OUTPUT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/incremental-candidate-replay/"
    "pca-iv-weekly-1000000-incremental-raw-1e-03-top10-n1000000.json"
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


def select_available_candidates(
    report: dict[str, Any], *, maximum_count: int
) -> list[dict[str, Any]]:
    """Return locally validated candidates in score order."""

    verify_inp(
        maximum_count > 0,
        lambda: f"'maximum_count' must be positive, got {maximum_count}",
    )
    candidates = report["candidate_certification"]["local_conditional_baskets"][
        "surviving"
    ]
    verify_inp(
        bool(candidates),
        "the source report has no locally validated candidates",
    )
    return sorted(candidates, key=_conservative_mean_lower, reverse=True)[
        :maximum_count
    ]


def _initial_surface(
    decoder: PCATotalVarianceDecoder,
    *,
    scores: np.ndarray,
    traded_nms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expiries = decoder.pca.expiries
    relative_strikes = decoder.relative_strikes_from_nms(
        scores[None], expiries[:, None], traded_nms
    )[0]
    normalized_calls, _ = decoder.normalized_call_prices(
        scores[None], expiries[:, None], relative_strikes[None]
    )
    normalized_calls = normalized_calls[0]
    verify_inp(
        relative_strikes.shape == normalized_calls.shape
        and np.all(np.isfinite(relative_strikes))
        and np.all(np.isfinite(normalized_calls)),
        "initial PCA-IV surface is non-finite or has inconsistent geometry",
    )
    return relative_strikes, normalized_calls


def _endpoint_calls(
    decoder: PCATotalVarianceDecoder,
    *,
    endpoint_scores: np.ndarray,
    endpoint_spot: np.ndarray,
    cash_strikes: np.ndarray,
    expiries: np.ndarray,
    expiry_offsets: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Reprice alive fixed-cash calls after one weekly period."""

    paths = endpoint_scores.shape[0]
    calls = np.zeros((paths, *cash_strikes.shape), dtype=np.float64)
    alive = expiry_offsets > stride
    if np.any(alive):
        relative_strikes = (
            cash_strikes[None, alive] / endpoint_spot[:, None, None]
        )
        alive_calls, _ = decoder.normalized_call_prices(
            endpoint_scores,
            (expiries[alive] - stride / DAYS_PER_YEAR)[None, :, None],
            relative_strikes,
        )
        calls[:, alive] = alive_calls
    if not np.all(np.isfinite(calls)) or np.any(calls < 0.0):
        raise FloatingPointError("endpoint PCA-IV calls are invalid")
    return calls


def _replay_candidate(
    *,
    rank: int,
    report: dict[str, Any],
    candidate: dict[str, Any],
    decoder: PCATotalVarianceDecoder,
    sampler: ImpliedPCASampler,
    expiries: np.ndarray,
    expiry_offsets: np.ndarray,
    traded_nms: np.ndarray,
    paths: int,
    block_size: int,
    seed: int,
    cost_rate: float,
) -> dict[str, Any]:
    initial_state = recover_candidate_state(report, candidate)
    initial_spot = math.exp(float(initial_state[0]))
    action = np.asarray(candidate["reference_solution"]["action"], dtype=np.float64)
    relative_strikes, normalized_calls = _initial_surface(
        decoder, scores=initial_state[1:], traded_nms=traded_nms
    )
    expected_instruments = expiries.size + normalized_calls.size
    verify_inp(
        action.shape == (expected_instruments,),
        lambda: (
            f"candidate action has shape {action.shape}, expected "
            f"{(expected_instruments,)}"
        ),
    )
    cash_strikes = relative_strikes * initial_spot
    normalized_deltas = np.zeros_like(normalized_calls)
    stride = 5
    sampler.rng = np.random.default_rng(seed)
    pnl_parts: list[np.ndarray] = []
    band_parts: list[np.ndarray] = []
    for start in range(0, paths, block_size):
        stop = min(start + block_size, paths)
        block_paths = stop - start
        xstate_path = sampler.sample_conditional_xstates_at_offsets(
            initial_state=initial_state,
            step_offsets=np.arange(stride + 1, dtype=np.int64),
            num_samples=block_paths,
            dt=1.0 / DAYS_PER_YEAR,
        )
        endpoint_spot = np.exp(xstate_path[:, -1, 0])
        endpoint_calls = _endpoint_calls(
            decoder,
            endpoint_scores=xstate_path[:, -1, 1:],
            endpoint_spot=endpoint_spot,
            cash_strikes=cash_strikes,
            expiries=expiries,
            expiry_offsets=expiry_offsets,
            stride=stride,
        )
        universe = build_terminal_path_universe(
            xstate_path=xstate_path,
            trading_times=1,
            trading_step_stride=stride,
            expiries=expiries.astype(np.float32),
            expiry_offsets=expiry_offsets,
            normalized_strikes=np.broadcast_to(
                relative_strikes[None, None],
                (block_paths, 1, *relative_strikes.shape),
            ),
            normalized_calls=np.broadcast_to(
                normalized_calls[None, None],
                (block_paths, 1, *normalized_calls.shape),
            ),
            normalized_deltas=np.broadcast_to(
                normalized_deltas[None, None],
                (block_paths, 1, *normalized_deltas.shape),
            ),
            normalized_calls_tp1=endpoint_calls[:, None],
            option_return_step_offset=stride,
            require_bounded_deltas=False,
        )
        case = derive_case_arrays(universe, hedge="raw", cost_rate=cost_rate)
        returns = case.returns[:, 0].astype(np.float64)
        costs = case.costs[:, 0].astype(np.float64)
        instrument_band = case.numerical_band[:, 0].astype(np.float64)
        pnl_parts.append(returns @ action - costs @ np.abs(action))
        band_parts.append(
            propagated_basket_band(action, returns, costs, instrument_band)
        )
        print(
            f"candidate {rank}: {stop:,}/{paths:,} conditional paths",
            flush=True,
        )

    summary = summarize_conditional_pnl(
        np.concatenate(pnl_parts), np.concatenate(band_parts)
    )
    print(
        f"candidate {rank}: t={candidate['trading_time_index']}, "
        f"query={candidate['query_index']}, losses={summary['economic_loss_count']:,}/"
        f"{paths:,}, min={summary['minimum_economic_pnl']:.8g}",
        flush=True,
    )
    return {
        "rank": rank,
        "ranking_score": _conservative_mean_lower(candidate),
        "source_trading_time_index": int(candidate["trading_time_index"]),
        "source_query_index": int(candidate["query_index"]),
        "source_validation_statistics": {
            name: candidate[name]
            for name in (
                "audit_paths_in_region",
                "direct_P_audit_paths_in_region",
                "minimum_audit_economic_pnl",
                "direct_P_mean",
                "adjusted_direct_P_mean_lower",
                "weighted_P_mean",
                "adjusted_weighted_P_mean_lower",
            )
        },
        "starting_state": {
            "log_spot": float(initial_state[0]),
            "spot": initial_spot,
            "surface_state": initial_state[1:].tolist(),
            "full_storage_state": initial_state.tolist(),
        },
        "portfolio": {
            "l1_norm": float(np.abs(action).sum()),
            "full_action": action.tolist(),
            "nonzero_positions": _sparse_portfolio(
                action=action,
                normalized_strikes=relative_strikes,
                normalized_calls=normalized_calls,
                expiries=expiries,
                spot=initial_spot,
            ),
        },
        "conditional_replay": summary,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    configuration = report["configuration"]
    cost_rate = float(configuration["cost_rate"])
    expected = {
        "generator": "pca-iv",
        "trading_frequency": "weekly",
        "return_horizon": "incremental",
        "hedge": "raw",
        "path_count": 1_000_000,
    }
    verify_inp(
        report.get("status") == "complete"
        and all(configuration[name] == value for name, value in expected.items())
        and math.isfinite(cost_rate)
        and cost_rate >= 0.0,
        lambda: f"source report is incomplete or not the requested case: {configuration}",
    )
    candidates = select_available_candidates(
        report, maximum_count=args.candidates
    )
    experiment = ImpliedPCAExperiment(
        num_samples=2,
        num_alpha=None,
        use_matrix_exp=True,
        bandwidth=1.0,
        trading_frequency="weekly",
        return_horizon="incremental",
        num_trading_times=10,
        seed=args.seed,
        importance_sampling=None,
    )
    _, market, pca, diagnostics = load_fitted_pca(
        experiment.fit_config, verbose=Context.quiet
    )
    decoder = PCATotalVarianceDecoder(pca)
    sampler = ImpliedPCASampler(
        pca=pca,
        scores=diagnostics.scores,
        time_y=market.time_y,
        spot_ref=market.spot_ref,
        num_alpha=experiment.num_alpha,
        seed=args.seed,
        verbose=Context.quiet,
    )
    expiries = np.asarray(pca.expiries, dtype=np.float64)
    expiry_offsets, expiry_error = expiry_step_offsets(
        expiries, dt=1.0 / DAYS_PER_YEAR
    )
    indices = traded_nms_indices(
        pca.surface_shape[1], experiment.num_traded_nms
    )
    traded_nms = pca.nms[:, indices]
    results = [
        _replay_candidate(
            rank=rank,
            report=report,
            candidate=candidate,
            decoder=decoder,
            sampler=sampler,
            expiries=expiries,
            expiry_offsets=expiry_offsets,
            traded_nms=traded_nms,
            paths=args.paths,
            block_size=args.block_size,
            seed=args.seed + 1_000_003 * rank,
            cost_rate=cost_rate,
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]
    output = {
        "status": "complete",
        "experiment": "exact-state weekly incremental PCA-IV candidate replay",
        "source_report": str(args.source_report),
        "selection": (
            "up to ten locally validated finite-neighbourhood candidates with "
            "the largest minimum of the "
            "multiplicity-adjusted direct-P and importance-weighted validation "
            "mean lower bounds"
        ),
        "continuation_law": (
            "independent five-daily-step matrix-exponential continuations under "
            "the fitted physical PCA-IV law from each identical state"
        ),
        "contract_timing": (
            "fixed cash strikes; options expiring within the week settle at "
            "their exact expiry, while all other options are repriced after "
            "five daily steps; each expiry-matched spot coordinate closes at "
            "the same endpoint as its option group"
        ),
        "configuration": {
            "requested_candidates": args.candidates,
            "available_candidates": len(candidates),
            "conditional_paths_per_candidate": args.paths,
            "seed": args.seed,
            "block_size": args.block_size,
            "cost_rate": cost_rate,
            "hedge": "raw",
            "maximum_expiry_time_error": expiry_error,
        },
        "candidates": results,
        "candidates_with_any_economic_loss": sum(
            item["conditional_replay"]["any_path_loses_money"]
            for item in results
        ),
        "candidates_with_any_loss_beyond_float32_band": sum(
            item["conditional_replay"]["any_path_loses_beyond_float32_band"]
            for item in results
        ),
    }
    _atomic_json(args.output, output)
    print(f"Wrote {args.output}", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--block-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2_026_807)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
