"""Replay raw or delta-hedged period portfolios from exact starting states."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR

from .required_runs import (
    Hedge,
    clopper_pearson_one_sided,
    derive_case_arrays,
    propagated_basket_band,
)
from .terminal_universe import build_terminal_path_universe, expiry_step_offsets
from .torch_prep import (
    DysanosUniverseStreamDiagnostics,
    GenDysanosExperiment,
    _load_generator_components,
    _stabilize_incremental_calls,
    _terminal_call_delta,
)

_DEFAULT_REPORT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/required/"
    "dysanos-weekly-1000000-incremental-raw-1e-03.json"
)
_DEFAULT_OUTPUT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/incremental-candidate-replay/"
    "dysanos-weekly-1000000-incremental-raw-1e-03-top10-n10000.json"
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


def _conservative_mean_lower(candidate: dict[str, Any]) -> float:
    """Return the weaker of the direct-P and likelihood-weighted mean bounds."""

    return min(
        float(candidate["adjusted_direct_P_mean_lower"]),
        float(candidate["adjusted_weighted_P_mean_lower"]),
    )


def select_strongest_candidates(
    report: dict[str, Any], *, count: int
) -> list[dict[str, Any]]:
    """Rank locally validated candidates by their conservative mean bound."""

    verify_inp(count > 0, lambda: f"'count' must be positive, got {count}")
    candidates = report["candidate_certification"]["local_conditional_baskets"][
        "surviving"
    ]
    verify_inp(
        len(candidates) >= count,
        lambda: f"requested {count} candidates but only {len(candidates)} survive",
    )
    return sorted(candidates, key=_conservative_mean_lower, reverse=True)[:count]


def recover_candidate_state(
    report: dict[str, Any], candidate: dict[str, Any]
) -> np.ndarray:
    """Invert the report's training-only state normalization at one query."""

    time_index = int(candidate["trading_time_index"])
    normalization = report["conditional"]["times"][time_index][
        "state_normalization"
    ]
    mean = np.asarray(normalization["reference_only_mean"], dtype=np.float64)
    scale = np.asarray(
        normalization["reference_only_population_scale"], dtype=np.float64
    )
    center = np.asarray(candidate["state_region"]["center"], dtype=np.float64)
    verify_inp(
        mean.shape == scale.shape == center.shape,
        "candidate state normalization has inconsistent geometry",
    )
    state = mean + scale * center
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("recovered candidate state is non-finite")
    return state


def summarize_conditional_pnl(
    pnl: np.ndarray, band: np.ndarray
) -> dict[str, Any]:
    """Summarize economic and float32-band losses over conditional paths."""

    pnl = np.asarray(pnl, dtype=np.float64)
    band = np.asarray(band, dtype=np.float64)
    verify_inp(
        pnl.ndim == 1 and pnl.shape == band.shape and pnl.size > 0,
        "conditional P&L arrays have inconsistent geometry",
    )
    if not np.all(np.isfinite(pnl)) or not np.all(np.isfinite(band)):
        raise FloatingPointError("conditional P&L replay contains NaN or infinity")
    economic_loss = pnl < 0.0
    band_loss = pnl < -band
    economic_loss_count = int(np.count_nonzero(economic_loss))
    band_loss_count = int(np.count_nonzero(band_loss))
    _, economic_loss_upper = clopper_pearson_one_sided(
        economic_loss_count, pnl.size, alpha=0.05
    )
    _, band_loss_upper = clopper_pearson_one_sided(
        band_loss_count, pnl.size, alpha=0.05
    )
    mean = float(np.mean(pnl))
    error = float(np.std(pnl, ddof=0) / math.sqrt(pnl.size))
    return {
        "paths": int(pnl.size),
        "economic_loss_count": economic_loss_count,
        "economic_loss_frequency": float(np.mean(economic_loss)),
        "economic_loss_probability_one_sided_95_upper": economic_loss_upper,
        "loss_count_beyond_float32_band": band_loss_count,
        "loss_frequency_beyond_float32_band": float(np.mean(band_loss)),
        "loss_probability_beyond_band_one_sided_95_upper": band_loss_upper,
        "minimum_economic_pnl": float(np.min(pnl)),
        "minimum_pnl_plus_float32_band": float(np.min(pnl + band)),
        "mean_pnl": mean,
        "mean_pnl_standard_error": error,
        "mean_pnl_lower_three_standard_errors": mean - 3.0 * error,
        "pnl_quantiles": {
            str(probability): float(np.quantile(pnl, probability))
            for probability in (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "any_path_loses_money": bool(np.any(economic_loss)),
        "any_path_loses_beyond_float32_band": bool(np.any(band_loss)),
    }


def _decode_initial_surface(model, *, state: np.ndarray, expiries: np.ndarray):
    """Decode the traded grid and entry prices from one DYSANOS state."""

    surface_state = torch.as_tensor(
        state[None, 1:], dtype=model.env.fdtype, device=model.env.device
    )
    torch_expiries = torch.as_tensor(
        expiries[None], dtype=model.env.fdtype, device=model.env.device
    )
    with torch.no_grad():
        pricer = model.decoder(
            state=surface_state, expiries=torch_expiries, training=False
        )
        calls = pricer(pricer.strikes, training=False)
        deltas = _terminal_call_delta(pricer, pricer.strikes)
    return (
        pricer.strikes.detach().cpu().numpy()[0],
        calls.detach().cpu().numpy()[0],
        deltas.detach().cpu().numpy()[0],
    )


def _sparse_portfolio(
    *,
    action: np.ndarray,
    normalized_strikes: np.ndarray,
    normalized_calls: np.ndarray,
    expiries: np.ndarray,
    spot: float,
) -> list[dict[str, Any]]:
    """Describe non-zero positions in the flattened period portfolio."""

    strikes = normalized_strikes.shape[1]
    positions: list[dict[str, Any]] = []
    spot_instruments = int(expiries.size)
    for instrument, notional in enumerate(action):
        if abs(notional) <= 1.0e-14:
            continue
        if instrument < spot_instruments:
            positions.append(
                {
                    "instrument_index": instrument,
                    "kind": "spot",
                    "expiry_index": instrument,
                    "expiry_years": float(expiries[instrument]),
                    "notional": float(notional),
                    "entry_price": spot,
                }
            )
            continue
        flattened = instrument - spot_instruments
        expiry_index, strike_index = divmod(flattened, strikes)
        relative_strike = float(normalized_strikes[expiry_index, strike_index])
        positions.append(
            {
                "instrument_index": instrument,
                "kind": "call",
                "expiry_index": expiry_index,
                "expiry_years": float(expiries[expiry_index]),
                "strike_index": strike_index,
                "relative_strike": relative_strike,
                "cash_strike": relative_strike * spot,
                "notional": float(notional),
                "entry_price": (
                    float(normalized_calls[expiry_index, strike_index]) * spot
                ),
            }
        )
    return positions


def _replay_candidate(
    *,
    rank: int,
    report: dict[str, Any],
    candidate: dict[str, Any],
    model,
    sampler,
    expiries: np.ndarray,
    expiry_offsets: np.ndarray,
    paths: int,
    block_size: int,
    seed: int,
    cost_rate: float,
    hedge: Hedge,
) -> dict[str, Any]:
    """Simulate independent weekly continuations for one frozen candidate."""

    initial_state = recover_candidate_state(report, candidate)
    action = np.asarray(candidate["reference_solution"]["action"], dtype=np.float64)
    normalized_strikes, normalized_calls, normalized_deltas = (
        _decode_initial_surface(model, state=initial_state, expiries=expiries)
    )
    spot = math.exp(float(initial_state[0]))
    expected_instruments = expiries.size + normalized_calls.size
    verify_inp(
        action.shape == (expected_instruments,),
        lambda: (
            f"candidate action has shape {action.shape}, expected "
            f"{(expected_instruments,)}"
        ),
    )

    stride = 5
    sampler.rng = np.random.default_rng(seed)
    pnl_parts: list[np.ndarray] = []
    band_parts: list[np.ndarray] = []
    diagnostics = DysanosUniverseStreamDiagnostics()
    for start in range(0, paths, block_size):
        stop = min(start + block_size, paths)
        block_paths = stop - start
        xstate_path = sampler.sample_conditional_xstates_at_offsets(
            initial_state=initial_state,
            step_offsets=np.arange(stride + 1, dtype=np.int64),
            num_samples=block_paths,
            dt=1.0 / DAYS_PER_YEAR,
        )
        endpoint_state = torch.as_tensor(
            xstate_path[:, -1, 1:],
            dtype=model.env.fdtype,
            device=model.env.device,
        )
        torch_expiries = torch.as_tensor(
            expiries[None], dtype=model.env.fdtype, device=model.env.device
        )
        endpoint_spot = np.exp(xstate_path[:, -1, 0].astype(np.float64))
        endpoint_strikes = (
            normalized_strikes[None] * spot / endpoint_spot[:, None, None]
        )
        with torch.no_grad():
            endpoint_pricer = model.decoder(
                state=endpoint_state,
                expiries=torch_expiries,
                training=False,
            )
            endpoint_calls = endpoint_pricer(
                torch.as_tensor(
                    endpoint_strikes,
                    dtype=model.env.fdtype,
                    device=model.env.device,
                ),
                dt=stride / DAYS_PER_YEAR,
                training=False,
            )
            endpoint_calls = _stabilize_incremental_calls(
                endpoint_calls[:, None],
                expiry_offsets=expiry_offsets,
                stride=stride,
                diagnostics=diagnostics,
            )[:, 0]
        endpoint_calls_np = endpoint_calls.detach().cpu().numpy()
        universe = build_terminal_path_universe(
            xstate_path=xstate_path,
            trading_times=1,
            trading_step_stride=stride,
            expiries=expiries.astype(np.float32),
            expiry_offsets=expiry_offsets,
            normalized_strikes=np.broadcast_to(
                normalized_strikes[None, None],
                (block_paths, 1, *normalized_strikes.shape),
            ),
            normalized_calls=np.broadcast_to(
                normalized_calls[None, None],
                (block_paths, 1, *normalized_calls.shape),
            ),
            normalized_deltas=np.broadcast_to(
                normalized_deltas[None, None],
                (block_paths, 1, *normalized_deltas.shape),
            ),
            normalized_calls_tp1=endpoint_calls_np[:, None],
            option_return_step_offset=stride,
        )
        case = derive_case_arrays(universe, hedge=hedge, cost_rate=cost_rate)
        returns = case.returns[:, 0].astype(np.float64)
        costs = case.costs[:, 0].astype(np.float64)
        instrument_band = case.numerical_band[:, 0].astype(np.float64)
        pnl = returns @ action - costs @ np.abs(action)
        band = propagated_basket_band(action, returns, costs, instrument_band)
        pnl_parts.append(pnl)
        band_parts.append(band)

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
            "spot": spot,
            "surface_state": initial_state[1:].tolist(),
            "full_storage_state": initial_state.tolist(),
        },
        "portfolio": {
            "l1_norm": float(np.abs(action).sum()),
            "full_action": action.tolist(),
            "nonzero_positions": _sparse_portfolio(
                action=action,
                normalized_strikes=normalized_strikes,
                normalized_calls=normalized_calls,
                expiries=expiries,
                spot=spot,
            ),
        },
        "conditional_replay": summary,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the ten exact-state conditional continuation replays."""

    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    configuration = report["configuration"]
    cost_rate = float(configuration["cost_rate"])
    hedge = cast(Hedge, str(configuration["hedge"]))
    expected = {
        "generator": "dysanos",
        "trading_frequency": "weekly",
        "return_horizon": "incremental",
        "path_count": 1_000_000,
    }
    verify_inp(
        report.get("status") == "complete"
        and all(configuration[name] == value for name, value in expected.items())
        and hedge in {"raw", "delta"}
        and math.isfinite(cost_rate)
        and cost_rate >= 0.0,
        lambda: f"source report is not the requested experiment: {configuration}",
    )
    candidates = select_strongest_candidates(report, count=args.candidates)
    experiment = GenDysanosExperiment(
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
    grid_config, _, model, sampler = _load_generator_components(
        experiment, verbose=Context.quiet
    )
    compute_device = torch.device(args.device)
    verify_inp(
        compute_device.type != "cuda" or torch.cuda.is_available(),
        lambda: f"CUDA device {args.device!r} was requested but is unavailable",
    )
    # The fitted generator is loaded on CPU. Conditional repricing is the
    # expensive replay operation, so move only the trained decoder after the
    # NumPy sampler has captured the fitted transition.
    model.env.device = compute_device
    model.to(compute_device)
    model.decoder.xnms = model.decoder.xnms.to(compute_device)
    model.eval()
    expiries = np.asarray(grid_config.grid_expiries, dtype=np.float64)
    expiry_offsets, expiry_error = expiry_step_offsets(
        expiries, dt=1.0 / DAYS_PER_YEAR
    )
    results = [
        _replay_candidate(
            rank=rank,
            report=report,
            candidate=candidate,
            model=model,
            sampler=sampler,
            expiries=expiries,
            expiry_offsets=expiry_offsets,
            paths=args.paths,
            block_size=args.block_size,
            seed=args.seed + 1_000_003 * rank,
            cost_rate=cost_rate,
            hedge=hedge,
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]
    output = {
        "status": "complete",
        "experiment": (
            f"exact-state weekly incremental {hedge}-return candidate replay"
        ),
        "source_report": str(args.source_report),
        "selection": (
            "locally validated finite-neighbourhood candidates with the "
            "largest minimum of the "
            "multiplicity-adjusted direct-P and importance-weighted validation "
            "mean lower bounds"
        ),
        "continuation_law": (
            "independent five-daily-step matrix-exponential continuations under "
            "the fitted physical DYSANOS law from each identical state"
        ),
        "contract_timing": (
            "fixed cash strikes; options expiring within the week settle at "
            "their exact expiry, while all other options are repriced after "
            "five daily steps; each expiry-matched spot coordinate closes at "
            "the same endpoint as its option group"
        ),
        "configuration": {
            "candidates": args.candidates,
            "conditional_paths_per_candidate": args.paths,
            "seed": args.seed,
            "block_size": args.block_size,
            "device": str(compute_device),
            "cost_rate": cost_rate,
            "hedge": hedge,
            "maximum_expiry_time_error": expiry_error,
        },
        "candidates": results,
        "candidates_with_any_economic_loss": sum(
            item["conditional_replay"]["any_path_loses_money"] for item in results
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
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--block-size", type=int, default=2_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2_026_807)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
