"""Legacy diagnostic helpers retained for stored-report compatibility.

The former command-line experiment pooled trading times and did not implement
the matched 48-case design. New experiments must use
``run_required_experiments``. A few private array helpers remain imported by
older PCA report readers until those stored formats are retired.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.dysanos.grid.rndysanos.local_tail import (
    LocalTailConfig,
    LocalTailProgress,
    normalize_state,
    state_normalization,
    valuate_local_tail,
)
from quantdata.dysanos.grid.rndysanos.run_experiment import memory_snapshot
from quantdata.dysanos.grid.rndysanos.run_v2_mask_comparison import (
    _attach_universe,
    _cleanup_shared_memory,
    _share_universe,
)
from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    DEFAULT_ROUNDOFF_MULTIPLIER,
    FLOAT32_UNIT_ROUNDOFF,
    SignAccumulator,
)
from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig
from quantdata.torch import DeviceLock

from .run_terminal_mode_comparison import _mode_view
from .terminal_universe import TerminalPathDiagnostics
from .torch_prep import GenDysanosExperiment, prepare_terminal_path_experiment

_MODES = ("one-step", "one-step-weekly", "path", "weekly")
_TARGET_PROPOSAL_TAIL_COUNT = 1_000


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {name: _json_value(item) for name, item in value.items()}
    return value


def _returns_and_roundoff(
    universe: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return spot plus delta-hedged option P&Ls and float32 error bands.

    Expiry universes use each contract's ``expiry_dS``. Repricing universes
    use the matched ``option_dS`` supplied by the same universe builder. The
    arithmetic and roundoff definitions are otherwise identical.
    """

    dC = np.asarray(universe["dC"], dtype=np.float32)
    delta = np.asarray(universe["delta"], dtype=np.float32)
    dS = np.asarray(universe["dS"], dtype=np.float32)
    option_dS_name = "expiry_dS" if "expiry_dS" in universe else "option_dS"
    option_dS = np.asarray(universe[option_dS_name], dtype=np.float32)
    C = np.asarray(universe["C"], dtype=np.float32)
    spot = np.asarray(universe["spot"], dtype=np.float32)
    samples, steps, expiries, strikes = dC.shape
    instruments = 1 + expiries * strikes
    output = np.empty((samples, steps, instruments), dtype=np.float32)
    output[..., 0] = dS
    option_output = output[..., 1:].reshape(dC.shape)
    np.multiply(delta, option_dS[..., None], out=option_output)
    np.subtract(dC, option_output, out=option_output)

    roundoff = np.empty_like(output)
    endpoint_spot = spot + dS
    roundoff[..., 0] = (
        DEFAULT_ROUNDOFF_MULTIPLIER
        * FLOAT32_UNIT_ROUNDOFF
        * np.maximum(
            np.abs(endpoint_spot) + np.abs(spot),
            np.finfo(np.float32).tiny,
        )
    )
    option_roundoff = roundoff[..., 1:].reshape(dC.shape)
    hedge = delta * option_dS[..., None]
    if "log_strikes" in universe and "option_expired" in universe:
        cash_strike = (
            np.exp(np.asarray(universe["log_strikes"], dtype=np.float32))
            * spot[:, :, None, None]
        )
        settlement_dS = np.asarray(
            universe.get("option_settlement_dS", option_dS), dtype=np.float32
        )
        settlement_spot = spot[:, :, None] + settlement_dS
        expiry_scale = (
            np.abs(settlement_spot[..., None])
            + np.abs(cash_strike)
            + np.abs(C)
        )
        marked_scale = np.abs(C + dC) + np.abs(C)
        expired = np.asarray(universe["option_expired"], dtype=bool)[..., None]
        option_roundoff[...] = np.where(expired, expiry_scale, marked_scale)
    else:
        # Compatibility with older stored universes which did not retain the
        # fixed strike/expiry marker needed to reconstruct the exact operands.
        np.add(np.abs(C + dC), np.abs(C), out=option_roundoff)
    option_roundoff += np.abs(hedge)
    np.maximum(
        option_roundoff,
        np.finfo(np.float32).tiny,
        out=option_roundoff,
    )
    option_roundoff *= DEFAULT_ROUNDOFF_MULTIPLIER * FLOAT32_UNIT_ROUNDOFF
    del endpoint_spot, hedge
    if not np.isfinite(output).all() or not np.isfinite(roundoff).all():
        raise ValueError("conditional-arbitrage returns contain non-finite values")
    if np.any(roundoff <= 0.0):
        raise ValueError("conditional-arbitrage roundoff bands must be positive")
    return output, roundoff


def _unconditional_signs(
    returns: np.ndarray,
    roundoff: np.ndarray,
    *,
    strikes: int,
    log_path_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    accumulator = SignAccumulator(instruments=returns.shape[-1])
    accumulator.update(returns, roundoff)
    arrays = accumulator.arrays()
    path_weights = (
        np.ones(returns.shape[0], dtype=np.float64)
        if log_path_weights is None
        else _stable_path_weights(log_path_weights)
    )
    mass = float(path_weights.sum()) * returns.shape[1]
    mean = np.einsum(
        "s,sti->i", path_weights, returns.astype(np.float64, copy=False)
    ) / mass
    second_moment = np.einsum(
        "s,sti->i",
        path_weights,
        np.square(returns.astype(np.float64, copy=False)),
    ) / mass
    sigma = np.sqrt(np.maximum(second_moment - np.square(mean), 0.0))
    positive_probability = np.einsum(
        "s,sti->i", path_weights, returns > roundoff
    ) / mass
    negative_probability = np.einsum(
        "s,sti->i", path_weights, returns < -roundoff
    ) / mass
    ratio = np.divide(
        np.abs(mean),
        sigma,
        out=np.full_like(mean, np.inf),
        where=sigma > 0.0,
    )
    ratio[(sigma == 0.0) & (mean == 0.0)] = 0.0
    minority = np.minimum(
        np.asarray(arrays["positive"]), np.asarray(arrays["negative"])
    )
    order = np.argsort(-ratio)

    def instrument(index: int) -> dict[str, Any]:
        if index == 0:
            label = {"kind": "spot", "expiry_index": None, "strike_index": None}
        else:
            option = index - 1
            label = {
                "kind": "delta_hedged_option",
                "expiry_index": option // strikes,
                "strike_index": option % strikes,
            }
        return {
            "instrument": int(index),
            **label,
            "abs_mean_over_std": float(ratio[index]),
            "positive": int(np.asarray(arrays["positive"])[index]),
            "negative": int(np.asarray(arrays["negative"])[index]),
            "neutral": int(np.asarray(arrays["neutral"])[index]),
            "mean": float(mean[index]),
            "std_population": float(sigma[index]),
            "weighted_positive_probability": float(positive_probability[index]),
            "weighted_negative_probability": float(negative_probability[index]),
        }

    return {
        "observations": int(arrays["observations"]),
        "independent_paths": int(arrays["paths"]),
        "steps": int(arrays["steps"]),
        "measure": (
            "likelihood-weighted original P"
            if log_path_weights is not None
            else "uniform empirical"
        ),
        "independent_path_effective_sample_size": float(
            np.square(path_weights.sum()) / np.square(path_weights).sum()
        ),
        "strict_one_sided_instruments_with_roundoff_tolerance": int(
            np.count_nonzero(minority == 0)
        ),
        "maximum_abs_mean_over_std": float(np.max(ratio)),
        "instruments_abs_mean_over_std_gt_2": int(np.count_nonzero(ratio > 2.0)),
        "instruments_abs_mean_over_std_gt_3": int(np.count_nonzero(ratio > 3.0)),
        "top_instruments": [instrument(int(index)) for index in order[:20]],
        "per_instrument": _json_value(
            arrays
            | {
                "weighted_mean": mean,
                "weighted_std_population": sigma,
                "weighted_positive_probability": positive_probability,
                "weighted_negative_probability": negative_probability,
            }
        ),
    }


def _option_tail_coverage(
    universe: Mapping[str, np.ndarray],
    returns: np.ndarray,
    roundoff: np.ndarray,
    *,
    log_path_weights: np.ndarray,
    instrument_batch_size: int,
) -> dict[str, Any]:
    """Measure proposal and reweighted-P payoff/P&L support per option."""

    samples, steps, instruments_with_spot = returns.shape
    instruments = instruments_with_spot - 1
    payoff = (
        np.asarray(universe["C"], dtype=np.float32)
        + np.asarray(universe["dC"], dtype=np.float32)
    ).reshape(samples, steps, instruments)
    option_returns = returns[..., 1:]
    option_roundoff = roundoff[..., 1:]
    weights = _stable_path_weights(log_path_weights)
    mass = float(weights.sum()) * steps
    payoff_count = np.zeros(instruments, dtype=np.int64)
    zero_payoff_count = np.zeros(instruments, dtype=np.int64)
    positive_count = np.zeros(instruments, dtype=np.int64)
    negative_count = np.zeros(instruments, dtype=np.int64)
    payoff_probability = np.zeros(instruments, dtype=np.float64)
    positive_probability = np.zeros(instruments, dtype=np.float64)
    negative_probability = np.zeros(instruments, dtype=np.float64)
    for left in range(0, instruments, instrument_batch_size):
        right = min(left + instrument_batch_size, instruments)
        pays = payoff[..., left:right] > 0.0
        zero_payoff = ~pays
        positive = (
            option_returns[..., left:right]
            > option_roundoff[..., left:right]
        )
        negative = (
            option_returns[..., left:right]
            < -option_roundoff[..., left:right]
        )
        payoff_count[left:right] = pays.sum(axis=(0, 1))
        zero_payoff_count[left:right] = zero_payoff.sum(axis=(0, 1))
        positive_count[left:right] = positive.sum(axis=(0, 1))
        negative_count[left:right] = negative.sum(axis=(0, 1))
        payoff_probability[left:right] = np.einsum(
            "s,sti->i", weights, pays
        ) / mass
        positive_probability[left:right] = np.einsum(
            "s,sti->i", weights, positive
        ) / mass
        negative_probability[left:right] = np.einsum(
            "s,sti->i", weights, negative
        ) / mass
    both_return_signs = (positive_count > 0) & (negative_count > 0)
    both_payoff_sides = (payoff_count > 0) & (zero_payoff_count > 0)
    return {
        "definition": (
            "proposal counts and likelihood-weighted original-P probabilities; "
            "positive/negative refer to the long expiry-matched delta-hedged "
            "return beyond its float32 roundoff band"
        ),
        "observations_per_instrument": samples * steps,
        "options": instruments,
        "options_with_both_payoff_sides": int(np.count_nonzero(both_payoff_sides)),
        "options_with_both_delta_hedged_return_signs": int(
            np.count_nonzero(both_return_signs)
        ),
        "all_options_have_both_payoff_sides": bool(np.all(both_payoff_sides)),
        "all_options_have_both_delta_hedged_return_signs": bool(
            np.all(both_return_signs)
        ),
        "target_proposal_count_per_side": _TARGET_PROPOSAL_TAIL_COUNT,
        "all_options_meet_target_payoff_count_per_side": bool(
            np.all(payoff_count >= _TARGET_PROPOSAL_TAIL_COUNT)
            and np.all(zero_payoff_count >= _TARGET_PROPOSAL_TAIL_COUNT)
        ),
        "all_options_meet_target_delta_hedged_count_per_sign": bool(
            np.all(positive_count >= _TARGET_PROPOSAL_TAIL_COUNT)
            and np.all(negative_count >= _TARGET_PROPOSAL_TAIL_COUNT)
        ),
        "minimum_proposal_payoff_count": int(np.min(payoff_count)),
        "minimum_proposal_zero_payoff_count": int(np.min(zero_payoff_count)),
        "minimum_proposal_positive_delta_hedged_count": int(
            np.min(positive_count)
        ),
        "minimum_proposal_negative_delta_hedged_count": int(
            np.min(negative_count)
        ),
        "minimum_weighted_P_payoff_probability": float(
            np.min(payoff_probability)
        ),
        "minimum_weighted_P_positive_delta_hedged_probability": float(
            np.min(positive_probability)
        ),
        "minimum_weighted_P_negative_delta_hedged_probability": float(
            np.min(negative_probability)
        ),
        "per_instrument": {
            "proposal_payoff_count": payoff_count.tolist(),
            "proposal_zero_payoff_count": zero_payoff_count.tolist(),
            "proposal_positive_delta_hedged_count": positive_count.tolist(),
            "proposal_negative_delta_hedged_count": negative_count.tolist(),
            "weighted_P_payoff_probability": payoff_probability.tolist(),
            "weighted_P_positive_delta_hedged_probability": (
                positive_probability.tolist()
            ),
            "weighted_P_negative_delta_hedged_probability": (
                negative_probability.tolist()
            ),
        },
    }


def _sample_indices(
    *,
    paths: int,
    steps: int,
    maximum: int,
    seed: int,
    eligible_paths: np.ndarray | None = None,
) -> np.ndarray:
    if eligible_paths is None:
        candidates = np.arange(paths, dtype=np.int64)
    else:
        candidates = np.asarray(eligible_paths, dtype=np.int64)
        if (
            candidates.ndim != 1
            or candidates.size == 0
            or np.any(candidates < 0)
            or np.any(candidates >= paths)
            or np.unique(candidates).size != candidates.size
        ):
            raise ValueError("eligible query paths must be unique valid path indices")
    count = min(candidates.size, maximum)
    generator = np.random.default_rng(seed)
    selected_paths = np.sort(
        generator.choice(candidates, size=count, replace=False)
    )
    selected_times = generator.integers(0, steps, size=count, dtype=np.int64)
    return selected_paths.astype(np.int64) * steps + selected_times


def _device_tensor(value: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.ascontiguousarray(value, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )


def _stable_path_weights(log_weights: np.ndarray) -> np.ndarray:
    """Exponentiate likelihoods up to an irrelevant common scale."""

    value = np.asarray(log_weights, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("path log weights must be a finite vector")
    shifted = np.maximum(
        value - np.max(value), math.log(np.finfo(np.float64).tiny)
    )
    weight = np.exp(shifted)
    if not np.all(np.isfinite(weight)) or not np.any(weight > 0.0):
        raise ValueError("path likelihood weights are numerically degenerate")
    return weight


def _path_weight_diagnostics(log_weights: np.ndarray) -> dict[str, float]:
    weight = _stable_path_weights(log_weights)
    mass = float(weight.sum())
    return {
        "minimum_log_weight": float(np.min(log_weights)),
        "maximum_log_weight": float(np.max(log_weights)),
        "independent_path_effective_sample_size": float(
            mass**2 / np.square(weight).sum()
        ),
        "effective_sample_fraction": float(
            mass**2 / np.square(weight).sum() / weight.size
        ),
    }


def _run_job(
    descriptors: dict[str, dict[str, Any]],
    diagnostics: TerminalPathDiagnostics,
    args: argparse.Namespace,
) -> None:
    report_path = Path(args.report)
    universe, handles = _attach_universe(descriptors)
    try:
        universe = _mode_view(universe, mode=args.mode)
        started = time.monotonic()
        returns, roundoff = _returns_and_roundoff(universe)
        state = np.asarray(universe["xstate"], dtype=np.float32)
        samples, steps, instruments = returns.shape
        log_path_weights = np.asarray(
            universe.get(
                "log_path_weight", np.zeros(samples, dtype=np.float64)
            ),
            dtype=np.float64,
        )
        if log_path_weights.shape != (samples,) or not np.all(
            np.isfinite(log_path_weights)
        ):
            raise ValueError("universe path log weights are invalid")
        importance_components = (
            np.asarray(universe["importance_component"], dtype=np.int64)
            if "importance_component" in universe
            else None
        )
        if importance_components is not None and (
            importance_components.shape != (samples,)
            or np.any(importance_components < 0)
        ):
            raise ValueError("universe importance components are invalid")
        training_paths = min(samples - 1, 4 * samples // 5)
        validation_paths = samples - training_paths
        strikes = np.asarray(universe["dC"]).shape[-1]
        option_delta = np.asarray(universe["delta"], dtype=np.float32)
        mean, scale, constant = state_normalization(
            state[:training_paths],
            scale_floor=args.state_scale_floor,
            path_weights=_stable_path_weights(
                log_path_weights[:training_paths]
            ),
        )
        normalized_training = normalize_state(
            state[:training_paths], mean=mean, scale=scale
        ).reshape(training_paths * steps, -1)
        normalized_validation = normalize_state(
            state[training_paths:], mean=mean, scale=scale
        ).reshape(validation_paths * steps, -1)
        training_returns = returns[:training_paths].reshape(
            training_paths * steps, instruments
        )
        validation_returns = returns[training_paths:].reshape(
            validation_paths * steps, instruments
        )
        training_roundoff = roundoff[:training_paths].reshape(
            training_paths * steps, instruments
        )
        validation_roundoff = roundoff[training_paths:].reshape(
            validation_paths * steps, instruments
        )
        training_log_weights = np.repeat(
            log_path_weights[:training_paths], steps
        )
        validation_log_weights = np.repeat(
            log_path_weights[training_paths:], steps
        )
        training_query_paths = (
            None
            if importance_components is None
            else np.flatnonzero(importance_components[:training_paths] == 0)
        )
        validation_query_paths = (
            None
            if importance_components is None
            else np.flatnonzero(
                importance_components[training_paths:] == 0
            )
        )
        training_indices = _sample_indices(
            paths=training_paths,
            steps=steps,
            maximum=args.maximum_query_paths,
            seed=args.seed + 11,
            eligible_paths=training_query_paths,
        )
        validation_indices = _sample_indices(
            paths=validation_paths,
            steps=steps,
            maximum=args.maximum_query_paths,
            seed=args.seed + 29,
            eligible_paths=validation_query_paths,
        )
        report: dict[str, Any] = {
            "status": "prepared",
            "market": args.mode,
            "paths": samples,
            "training_paths": training_paths,
            "validation_paths": validation_paths,
            "num_surface_alpha": args.num_alpha,
            "surface_first": True,
            "steps": steps,
            "instruments": instruments,
            "return_semantics": (
                "spot next-action dS and expiry-matched delta-hedged option "
                "dH=dC-delta*(S_expiry-S_trade), without transaction costs"
            ),
            "roundoff_semantics": (
                f"{DEFAULT_ROUNDOFF_MULTIPLIER:g} times float32 unit "
                "roundoff applied to the unreduced arithmetic scale"
            ),
            "sampling_diagnostics": asdict(diagnostics),
            "path_weight_diagnostics": {
                "training": _path_weight_diagnostics(
                    log_path_weights[:training_paths]
                ),
                "validation": _path_weight_diagnostics(
                    log_path_weights[training_paths:]
                ),
            },
            "option_delta_diagnostics": {
                "minimum": float(np.min(option_delta)),
                "maximum": float(np.max(option_delta)),
                "mean": float(np.mean(option_delta, dtype=np.float64)),
                "strictly_positive_fraction": float(
                    np.count_nonzero(option_delta > 0.0) / option_delta.size
                ),
            },
            "state": {
                "mode": "raw xstate",
                "training_mean": mean.astype(float).tolist(),
                "training_population_scale": scale.astype(float).tolist(),
                "constant_feature_indices": np.flatnonzero(constant).astype(int).tolist(),
            },
            "training_query_states": int(training_indices.size),
            "validation_query_states": int(validation_indices.size),
            "query_sampling": (
                "at most one uniformly selected action time from each "
                + (
                    "direct-P base-mixture path; query states are therefore "
                    "unweighted P draws"
                    if importance_components is not None
                    else "deterministically sampled independent path"
                )
            ),
            "preparation_seconds": time.monotonic() - started,
            "option_tail_coverage_training": _option_tail_coverage(
                {
                    "C": universe["C"][:training_paths],
                    "dC": universe["dC"][:training_paths],
                },
                returns[:training_paths],
                roundoff[:training_paths],
                log_path_weights=log_path_weights[:training_paths],
                instrument_batch_size=args.instrument_batch_size,
            ),
            "option_tail_coverage_validation": _option_tail_coverage(
                {
                    "C": universe["C"][training_paths:],
                    "dC": universe["dC"][training_paths:],
                },
                returns[training_paths:],
                roundoff[training_paths:],
                log_path_weights=log_path_weights[training_paths:],
                instrument_batch_size=args.instrument_batch_size,
            ),
            "unconditional_training": _unconditional_signs(
                returns[:training_paths],
                roundoff[:training_paths],
                strikes=strikes,
                log_path_weights=log_path_weights[:training_paths],
            ),
            "unconditional_validation": _unconditional_signs(
                returns[training_paths:],
                roundoff[training_paths:],
                strikes=strikes,
                log_path_weights=log_path_weights[training_paths:],
            ),
        }
        _atomic_json(report_path, report)

        config = LocalTailConfig(
            state_mode="raw",
            neighbors=args.neighbors,
            tail_ranks=tuple(args.tail_ranks),
            signal_thresholds=tuple(args.signal_thresholds),
            query_batch_size=args.query_batch_size,
            instrument_batch_size=args.instrument_batch_size,
            roundoff_multiplier=DEFAULT_ROUNDOFF_MULTIPLIER,
            state_scale_floor=args.state_scale_floor,
        )
        with DeviceLock() as device_lock:
            device = device_lock.device
            if device.type != "cuda":
                raise RuntimeError("full coherent-market arbitrage tests require CUDA")
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            report.update(
                status="transferring",
                device=str(device),
                config=asdict(config),
                memory_before=memory_snapshot(device),
            )
            _atomic_json(report_path, report)
            reference_state = _device_tensor(normalized_training, device=device)
            reference_returns = _device_tensor(training_returns, device=device)
            reference_roundoff = _device_tensor(training_roundoff, device=device)
            reference_log_weights = _device_tensor(
                training_log_weights, device=device
            )
            query_training_state = _device_tensor(
                normalized_training[training_indices], device=device
            )
            query_training_returns = _device_tensor(
                training_returns[training_indices], device=device
            )
            query_training_roundoff = _device_tensor(
                training_roundoff[training_indices], device=device
            )
            query_training_log_weights = (
                None
                if importance_components is not None
                else _device_tensor(
                    training_log_weights[training_indices], device=device
                )
            )
            query_validation_state = _device_tensor(
                normalized_validation[validation_indices], device=device
            )
            query_validation_returns = _device_tensor(
                validation_returns[validation_indices], device=device
            )
            query_validation_roundoff = _device_tensor(
                validation_roundoff[validation_indices], device=device
            )
            query_validation_log_weights = (
                None
                if importance_components is not None
                else _device_tensor(
                    validation_log_weights[validation_indices], device=device
                )
            )
            leave_one_out_indices = torch.as_tensor(
                training_indices, dtype=torch.int64, device=device
            )
            del (
                universe,
                returns,
                roundoff,
                state,
                normalized_training,
                normalized_validation,
                training_returns,
                validation_returns,
                training_roundoff,
                validation_roundoff,
                training_log_weights,
                validation_log_weights,
                log_path_weights,
            )
            gc.collect()

            def progress(value: LocalTailProgress) -> None:
                report["live_progress"] = asdict(value)
                _atomic_json(report_path, report)
                print(
                    f"{args.paths}-{args.mode} {value.split}: "
                    f"{value.processed_states:,}/{value.total_states:,}; "
                    f"one-sided long={value.long_only_candidate_states:,}, "
                    f"short={value.short_only_candidate_states:,}, "
                    f"both={value.both_direction_candidate_states:,}",
                    flush=True,
                )

            report["status"] = "valuating_training"
            _atomic_json(report_path, report)
            training = valuate_local_tail(
                reference_state=reference_state,
                reference_returns=reference_returns,
                reference_roundoff=reference_roundoff,
                reference_log_weights=reference_log_weights,
                query_state=query_training_state,
                query_returns=query_training_returns,
                query_roundoff=query_training_roundoff,
                query_log_weights=query_training_log_weights,
                config=config,
                split="training",
                leave_one_out=True,
                leave_one_out_reference_indices=leave_one_out_indices,
                progress=progress,
                progress_interval_seconds=args.progress_seconds,
            )
            report.update(status="valuating_validation", training=training)
            _atomic_json(report_path, report)
            validation = valuate_local_tail(
                reference_state=reference_state,
                reference_returns=reference_returns,
                reference_roundoff=reference_roundoff,
                reference_log_weights=reference_log_weights,
                query_state=query_validation_state,
                query_returns=query_validation_returns,
                query_roundoff=query_validation_roundoff,
                query_log_weights=query_validation_log_weights,
                config=config,
                split="validation",
                leave_one_out=False,
                progress=progress,
                progress_interval_seconds=args.progress_seconds,
            )
            torch.cuda.synchronize(device)
            report.update(
                status="complete",
                training=training,
                validation=validation,
                memory_after=memory_snapshot(device),
                elapsed_seconds=time.monotonic() - started,
            )
            report.pop("live_progress", None)
            _atomic_json(report_path, report)
    except BaseException as error:
        try:
            prior = (
                json.loads(report_path.read_text(encoding="utf-8"))
                if report_path.exists()
                else {}
            )
            prior.update(status="failed", error=repr(error))
            _atomic_json(report_path, prior)
        finally:
            traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    comparison_path = output_root / "terminal-arbitrage-comparison.json"
    jobs = [
        argparse.Namespace(
            **(
                vars(args)
                | {
                    "paths": paths,
                    "mode": mode,
                    "seed": args.seed + paths + 1_009 * _MODES.index(mode),
                    "report": str(output_root / f"{paths}-{mode}.json"),
                }
            )
        )
        for paths in sorted(args.path_counts, reverse=True)
        for mode in args.modes
    ]
    comparison: dict[str, Any] = {
        "status": "preparing",
        "path_counts": list(args.path_counts),
        "modes": list(args.modes),
        "num_surface_alpha": args.num_alpha,
        "surface_first": True,
        "jobs": [job.report for job in jobs],
        "estimator": (
            "exact uniform k-nearest-neighbour conditional moments and tails; "
            "full training reference, sampled independent-path queries"
        ),
    }
    importance_config = (
        None
        if args.importance_spot_std_multiplier is None
        else GaussianSpotImportanceConfig(
            spot_std_multiplier=args.importance_spot_std_multiplier,
            base_probability=args.importance_base_probability,
            target_action_count=args.importance_target_action_count,
        )
    )
    comparison["importance_sampling"] = (
        None if importance_config is None else asdict(importance_config)
    )
    _atomic_json(comparison_path, comparison)
    shared: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    diagnostics: dict[tuple[int, str], TerminalPathDiagnostics] = {}
    all_handles: list[Any] = []
    try:
        source_modes = {
            "weekly" if mode in {"weekly", "one-step-weekly"} else "path"
            for mode in args.modes
        }
        for paths in args.path_counts:
            for source_mode in sorted(source_modes):
                key = (paths, source_mode)
                started = time.monotonic()
                universe, path_diagnostics = prepare_terminal_path_experiment(
                    GenDysanosExperiment(
                        num_samples=paths,
                        mode=source_mode,
                        num_steps=args.trading_state_steps,
                        num_alpha=args.num_alpha,
                        use_matrix_exp=True,
                        bandwidth=1.0,
                        seed=args.seed + paths,
                        block_size=args.generator_block_size,
                        importance_sampling=importance_config,
                    )
                )
                descriptors, handles, total_bytes = _share_universe(universe)
                del universe
                gc.collect()
                shared[key] = descriptors
                diagnostics[key] = path_diagnostics
                all_handles.extend(handles)
                comparison.setdefault("preparation", {})[
                    f"{paths}-{source_mode}"
                ] = {
                    "seconds": time.monotonic() - started,
                    "shared_bytes": total_bytes,
                    "diagnostics": asdict(path_diagnostics),
                }
                comparison["status"] = f"prepared-{paths}-{source_mode}"
                _atomic_json(comparison_path, comparison)
        atexit.register(_cleanup_shared_memory, all_handles)

        context = mp.get_context("spawn")
        pending: list[mp.Process] = []
        for job in jobs:
            source_mode = (
                "weekly"
                if job.mode in {"weekly", "one-step-weekly"}
                else "path"
            )
            key = (job.paths, source_mode)
            pending.append(
                context.Process(
                    target=_run_job,
                    args=(shared[key], diagnostics[key], job),
                    name=f"arb-{job.paths}-{job.mode}",
                )
            )
        running: list[mp.Process] = []
        exitcodes: dict[str, int | None] = {}
        while pending or running:
            while pending and len(running) < args.workers:
                process = pending.pop(0)
                process.start()
                running.append(process)
            time.sleep(1.0)
            for process in list(running):
                if process.is_alive():
                    continue
                process.join()
                exitcodes[process.name] = process.exitcode
                running.remove(process)
            comparison.update(
                status="running" if pending or running else "collecting",
                active_jobs=[process.name for process in running],
                pending_jobs=[process.name for process in pending],
                exitcodes=exitcodes,
            )
            _atomic_json(comparison_path, comparison)

        controls: dict[str, Any] = {}
        for job in jobs:
            path = Path(job.report)
            controls[f"{job.paths}-{job.mode}"] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {"status": "missing"}
            )
        comparison.update(
            status=(
                "complete"
                if all(code == 0 for code in exitcodes.values())
                and len(exitcodes) == len(jobs)
                else "failed"
            ),
            controls=controls,
        )
        _atomic_json(comparison_path, comparison)
        return comparison
    finally:
        _cleanup_shared_memory(all_handles)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-counts", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--modes", choices=_MODES, nargs="+", default=list(_MODES))
    parser.add_argument("--trading-state-steps", type=int, default=10)
    parser.add_argument(
        "--num-alpha",
        type=int,
        default=None,
        help="Retained surface innovation factors; omit to retain all 20",
    )
    parser.add_argument("--generator-block-size", type=int, default=10_000)
    parser.add_argument("--neighbors", type=int, default=1_024)
    parser.add_argument("--tail-ranks", type=int, nargs="+", default=[1, 2, 10, 20, 50])
    parser.add_argument("--signal-thresholds", type=float, nargs="+", default=[2.0, 3.0])
    parser.add_argument("--maximum-query-paths", type=int, default=20_000)
    parser.add_argument("--query-batch-size", type=int, default=1_024)
    parser.add_argument("--instrument-batch-size", type=int, default=20)
    parser.add_argument("--state-scale-floor", type=float, default=1.0e-6)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=600_000)
    parser.add_argument(
        "--importance-spot-std-multiplier",
        type=float,
        default=None,
        help=(
            "Enable exact Gaussian path importance sampling with this future "
            "log-spot standard-deviation multiplier"
        ),
    )
    parser.add_argument(
        "--importance-base-probability", type=float, default=0.1
    )
    parser.add_argument(
        "--importance-target-action-count", type=int, default=3
    )
    parser.add_argument(
        "--output-root",
        default=(
            "quantdata/dysanos/grid/gendysanos/reports/"
            "terminal-delta-arbitrage-tests"
        ),
    )
    return parser


def main() -> None:
    raise RuntimeError(
        "this legacy pooled-time CLI is retired; run "
        "python -m quantdata.dysanos.grid.gendysanos.run_required_experiments"
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
