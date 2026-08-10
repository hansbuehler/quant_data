"""Run exact adaptive-neighbour conditional-tail diagnostics on one universe."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.torch import DeviceLock

from .conditional_tail_experiment import process_memory
from .local_tail import (
    LocalTailConfig,
    LocalTailProgress,
    normalize_state,
    prepare_delta_hedged_local_data,
    reduced_option_state,
    shared_instrument_metric_diagnostics,
    state_normalization,
    valuate_local_tail,
)
from .torch_prep import RNDysanosExperiment, prep_experiment


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device_tensor(value: np.ndarray, *, device: torch.device) -> torch.Tensor:
    array = np.ascontiguousarray(value, dtype=np.float32)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare the universe and value training/validation local tails."""

    config = LocalTailConfig(
        state_mode=args.state_mode,
        neighbors=args.neighbors,
        instrument_candidate_neighbors=args.candidate_neighbors,
        tail_ranks=tuple(args.tail_ranks),
        query_batch_size=args.query_batch_size,
        instrument_batch_size=args.instrument_batch_size,
        roundoff_multiplier=args.roundoff_multiplier,
    )
    report_path = Path(args.report)
    training_paths = args.paths // 2
    report: dict[str, Any] = {
        "status": "preparing",
        "paths": args.paths,
        "training_paths": training_paths,
        "validation_paths": args.paths - training_paths,
        "validation_fraction": 0.5,
        "universe_seed_offset": args.universe_seed_offset,
        "return": "delta-hedged option dH=dC-delta*dS, before transaction costs",
        "config": asdict(config),
        "estimator": (
            "exact uniform adaptive k-nearest-neighbour kernel; training is "
            "leave-one-out and validation references training only"
        ),
    }
    _atomic_json(report_path, report)

    preparation_started = time.monotonic()
    experiment = RNDysanosExperiment(
        num_samples=args.paths,
        pcnt_validation=0.5,
        cnt_universe=1,
        universe_seed_offset=args.universe_seed_offset,
    )
    universe = prep_experiment(experiment, plot_sampler=False)[0]
    data, geometry = prepare_delta_hedged_local_data(
        universe,
        roundoff_multiplier=config.roundoff_multiplier,
    )
    del universe
    gc.collect()

    metric_diagnostics: dict[str, Any] | None = None
    normalized_instrument_feature: np.ndarray | None = None
    if config.state_mode == "instrument":
        metric_diagnostics = shared_instrument_metric_diagnostics(
            data,
            training_paths=training_paths,
            geometry=geometry,
            scale_floor=config.state_scale_floor,
        )
        log_strikes = data["log_strikes"]
        log_strike_mean = log_strikes[:training_paths].mean(
            axis=(0, 1), dtype=np.float64
        )
        log_strike_scale = log_strikes[:training_paths].std(
            axis=(0, 1), dtype=np.float64
        )
        constant_log_strike = log_strike_scale < config.state_scale_floor
        log_strike_scale = np.maximum(log_strike_scale, config.state_scale_floor)
        normalized_instrument_feature = (
            (log_strikes - log_strike_mean) / log_strike_scale
        ).astype(np.float32, copy=False)
    else:
        log_strike_mean = None
        log_strike_scale = None
        constant_log_strike = None
    state = reduced_option_state(
        data,
        geometry=geometry,
        mode="raw",
        instrument=0,
    )
    mean, scale, constant = state_normalization(
        state[:training_paths], scale_floor=config.state_scale_floor
    )
    normalized_state = normalize_state(state, mean=mean, scale=scale)
    preparation_seconds = time.monotonic() - preparation_started
    report.update(
        status="prepared",
        preparation_seconds=preparation_seconds,
        geometry=asdict(geometry),
        state={
            "mode": config.state_mode,
            "features": state.shape[-1] + (2 if config.state_mode == "instrument" else 0),
            "shared_xstate_features": state.shape[-1],
            "training_mean": mean.astype(float).tolist(),
            "training_population_scale": scale.astype(float).tolist(),
            "constant_feature_indices": np.flatnonzero(constant).astype(int).tolist(),
            "instrument_metric_diagnostics": metric_diagnostics,
            "instrument_log_strike_training_mean": (
                None if log_strike_mean is None else log_strike_mean.astype(float).tolist()
            ),
            "instrument_log_strike_training_population_scale": (
                None if log_strike_scale is None else log_strike_scale.astype(float).tolist()
            ),
            "constant_log_strike_instrument_indices": (
                None
                if constant_log_strike is None
                else np.flatnonzero(constant_log_strike).astype(int).tolist()
            ),
            "expiry_coordinate": (
                "omitted from distance because it is constant for each separately estimated instrument"
                if config.state_mode == "instrument"
                else "not used"
            ),
        },
        local_tail_probabilities=list(config.local_tail_probabilities),
        zero_event_upper_probability_95=config.zero_event_upper_probability_95,
    )
    _atomic_json(report_path, report)

    returns = data["returns"]
    roundoff = data["roundoff"]
    steps = geometry.steps
    instruments = geometry.expiries * geometry.strikes
    train_stop = training_paths * steps
    state_flat = normalized_state.reshape(args.paths * steps, -1)
    returns_flat = returns.reshape(args.paths * steps, instruments)
    roundoff_flat = roundoff.reshape(args.paths * steps, instruments)
    instrument_feature_flat = (
        None
        if normalized_instrument_feature is None
        else normalized_instrument_feature.reshape(args.paths * steps, instruments)
    )
    del data, state, normalized_state, normalized_instrument_feature, returns, roundoff
    gc.collect()

    with DeviceLock() as device_lock:
        device = device_lock.device
        if device.type != "cuda":
            raise RuntimeError("the exact full-universe local-tail experiment requires CUDA")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report.update(
            status="transferring",
            device=str(device),
            memory_before_valuation=process_memory(device),
        )
        _atomic_json(report_path, report)

        reference_state = _device_tensor(state_flat[:train_stop], device=device)
        validation_state = _device_tensor(state_flat[train_stop:], device=device)
        reference_returns = _device_tensor(returns_flat[:train_stop], device=device)
        validation_returns = _device_tensor(returns_flat[train_stop:], device=device)
        reference_roundoff = _device_tensor(roundoff_flat[:train_stop], device=device)
        validation_roundoff = _device_tensor(roundoff_flat[train_stop:], device=device)
        reference_instrument_feature = (
            None
            if instrument_feature_flat is None
            else _device_tensor(instrument_feature_flat[:train_stop], device=device)
        )
        validation_instrument_feature = (
            None
            if instrument_feature_flat is None
            else _device_tensor(instrument_feature_flat[train_stop:], device=device)
        )
        del state_flat, returns_flat, roundoff_flat, instrument_feature_flat
        gc.collect()

        valuation_started = time.monotonic()

        def progress(value: LocalTailProgress) -> None:
            processed = value.processed_states
            candidate_counts = {
                "long_only": value.long_only_candidate_states,
                "short_only": value.short_only_candidate_states,
                "both_directions": value.both_direction_candidate_states,
            }
            report["live_progress"] = {
                "split": value.split,
                "processed_states": processed,
                "total_states": value.total_states,
                "processed_fraction": processed / value.total_states,
                "zero_adverse_candidate_states": candidate_counts,
                "zero_adverse_candidate_state_rates": {
                    name: count / processed for name, count in candidate_counts.items()
                },
            }
            _atomic_json(report_path, report)
            print(
                f"{value.split}: {processed:,}/{value.total_states:,} states "
                f"({100.0 * processed / value.total_states:.2f}%); "
                "one-sided-support candidates "
                f"long-only={value.long_only_candidate_states:,} "
                f"({100.0 * value.long_only_candidate_states / processed:.3f}%), "
                f"short-only={value.short_only_candidate_states:,} "
                f"({100.0 * value.short_only_candidate_states / processed:.3f}%), "
                f"both={value.both_direction_candidate_states:,} "
                f"({100.0 * value.both_direction_candidate_states / processed:.3f}%)",
                flush=True,
            )

        report["status"] = "valuating_training"
        _atomic_json(report_path, report)
        training = valuate_local_tail(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_roundoff=reference_roundoff,
            query_state=reference_state,
            query_returns=reference_returns,
            query_roundoff=reference_roundoff,
            reference_instrument_feature=reference_instrument_feature,
            query_instrument_feature=reference_instrument_feature,
            config=config,
            split="training",
            leave_one_out=True,
            progress=progress,
            progress_interval_seconds=args.progress_seconds,
        )
        report.update(status="valuating_validation", training=training)
        _atomic_json(report_path, report)
        validation = valuate_local_tail(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_roundoff=reference_roundoff,
            query_state=validation_state,
            query_returns=validation_returns,
            query_roundoff=validation_roundoff,
            reference_instrument_feature=reference_instrument_feature,
            query_instrument_feature=validation_instrument_feature,
            config=config,
            split="validation",
            leave_one_out=False,
            progress=progress,
            progress_interval_seconds=args.progress_seconds,
        )
        torch.cuda.synchronize(device)
        report.update(
            status="complete",
            valuation_seconds=time.monotonic() - valuation_started,
            training=training,
            validation=validation,
            memory_after_valuation=process_memory(device),
        )
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, required=True)
    parser.add_argument("--state-mode", choices=("raw", "instrument"), required=True)
    parser.add_argument("--universe-seed-offset", type=int, required=True)
    parser.add_argument("--neighbors", type=int, default=1_024)
    parser.add_argument("--candidate-neighbors", type=int, default=8_192)
    parser.add_argument("--tail-ranks", type=int, nargs="+", default=[1, 2, 10, 20, 50])
    parser.add_argument("--query-batch-size", type=int, default=3_584)
    parser.add_argument("--instrument-batch-size", type=int, default=20)
    parser.add_argument("--roundoff-multiplier", type=float, default=32.0)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.paths <= 0 or args.paths % 2:
        raise ValueError("--paths must be a positive even number")
    if args.universe_seed_offset < 0:
        raise ValueError("--universe-seed-offset must be nonnegative")
    if not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0.0:
        raise ValueError("--progress-seconds must be finite and positive")
    run(args)
    print(f"completed local-tail report {args.report}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
