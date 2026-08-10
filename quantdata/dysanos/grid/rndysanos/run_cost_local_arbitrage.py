"""Fit a cost-adjusted exact local ``arb(z; omega)`` option-direction mask."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from quantdata.torch import BaseEnvironment, DeviceLock

from .conditional_tail_experiment import process_memory
from .direct_arbitrage import effective_cost, state_normalization, unconditional_cost
from .local_arbitrage import (
    LocalArbitrageConfig,
    LocalArbitrageProgress,
    pack_direction_mask,
    valuate_local_arbitrage_mask,
)
from .run_direct_arbitrage import _conditional_cost_model
from .torch_drift_change import (
    DysanosConfig,
    prepare_universe,
    split_training_validation,
)
from .torch_prep import RNDysanosExperiment, prep_experiment
from .unconditional_option_test import (
    FLOAT32_UNIT_ROUNDOFF,
    delta_hedged_roundoff_band,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _PackedMaskWriter:
    """Write flattened candidate batches directly to a packed ``.npy`` file."""

    def __init__(self, path: Path, *, states: int, instruments: int) -> None:
        self.path = path
        self.instruments = instruments
        path.parent.mkdir(parents=True, exist_ok=True)
        self._output = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.uint8,
            shape=(states, 2, math.ceil(instruments / 8)),
        )

    def __call__(self, start: int, stop: int, value: np.ndarray) -> None:
        if value.shape != (stop - start, 2, self.instruments):
            raise ValueError(
                f"candidate batch shape {value.shape} does not match "
                f"{(stop - start, 2, self.instruments)}"
            )
        self._output[start:stop] = pack_direction_mask(value)

    def close(self) -> None:
        self._output.flush()
        del self._output


def _cost_summary(cost: torch.Tensor) -> dict[str, Any]:
    return {
        "minimum": float(cost.min().item()),
        "mean": float(cost.mean(dtype=torch.float64).item()),
        "maximum": float(cost.max().item()),
        "mean_by_instrument": cost.mean(dim=(0, 1), dtype=torch.float64)
        .cpu()
        .tolist(),
    }


def run(
    args: argparse.Namespace,
    *,
    prepared_universe: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fit masks on one preloaded universe or load it through ``prep_experiment``."""

    if args.paths <= 0 or args.paths % 2:
        raise ValueError("--paths must be a positive even number")
    config = LocalArbitrageConfig(
        neighbors=args.neighbors,
        query_batch_size=args.query_batch_size,
        instrument_batch_size=args.instrument_batch_size,
        roundoff_multiplier=args.roundoff_multiplier,
    )
    report_path = Path(args.report)
    report: dict[str, Any] = {
        "status": "preparing",
        "paths": args.paths,
        "training_paths": args.paths // 2,
        "validation_paths": args.paths // 2,
        "universe_seed_offset": args.universe_seed_offset,
        "state": "raw xstate, standardized using the training half only",
        "return": "delta-hedged option dH=dC-delta*dS",
        "cost_definition": "max(conditional standard error, unconditional standard error, proportional cost)",
        "cost_prop": args.cost_prop,
        "config": asdict(config),
        "arb_function": (
            "arb_d(z,omega)=1 iff every kNN training gain d*dH-c is >= "
            "-roundoff and at least one is > roundoff"
        ),
    }
    _atomic_json(report_path, report)

    preparation_started = time.monotonic()
    if prepared_universe is None:
        experiment = RNDysanosExperiment(
            num_samples=args.paths,
            pcnt_validation=0.5,
            cnt_universe=1,
            universe_seed_offset=args.universe_seed_offset,
        )
        universe = prep_experiment(experiment, plot_sampler=False)[0]
    else:
        universe = prepared_universe
    preparation_seconds = time.monotonic() - preparation_started

    with DeviceLock() as device_lock:
        env = BaseEnvironment(device=device_lock.device, debug=False)
        torch.cuda.set_device(env.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(env.device)
        data_config = DysanosConfig(
            label="cost-local-arbitrage-data",
            delta_hedged=True,
            cash_returns=False,
            cost_cond=0.0,
            cost_uncond=1.0,
            cost_prop=args.cost_prop,
        )
        all_data, geometry = prepare_universe(
            universe, config=data_config, env=env
        )
        if geometry.samples != args.paths:
            raise RuntimeError(
                f"prepared universe has {geometry.samples} paths, expected {args.paths}"
            )
        dH_roundoff = delta_hedged_roundoff_band(
            C=np.asarray(universe["C"]),
            dC=np.asarray(universe["dC"]),
            delta=np.asarray(universe["delta"]),
            dS=np.asarray(universe["dS"]),
            multiplier=config.roundoff_multiplier,
        )
        training, validation = split_training_validation(all_data, 0.5)
        assert validation is not None
        training_paths = training["z"].shape[0]
        n_cost = training_paths * geometry.steps
        bar_c = unconditional_cost(
            training,
            cost_uncond=1.0,
            batch_size=args.valuation_batch_size,
        )
        conditional_model, conditional_checkpoint = _conditional_cost_model(
            report_path=Path(args.conditional_report),
            env=env,
            geometry=geometry,
            allow_different_sample_count=True,
        )
        training_cost = effective_cost(
            training,
            bar_c=bar_c,
            cost_prop=args.cost_prop,
            conditional_variance_model=conditional_model,
            n_cost=n_cost,
            batch_size=args.valuation_batch_size,
        )
        validation_cost = effective_cost(
            validation,
            bar_c=bar_c,
            cost_prop=args.cost_prop,
            conditional_variance_model=conditional_model,
            n_cost=n_cost,
            batch_size=args.valuation_batch_size,
        )
        del conditional_model

        raw_state = all_data["z"][..., : geometry.state_variables]
        training_state = raw_state[:training_paths]
        state_mean, state_scale = state_normalization(
            {"z": training_state}, batch_size=args.valuation_batch_size
        )
        normalized_state = (raw_state - state_mean) / state_scale
        option_returns = all_data["dH"][..., 1:]
        instruments = option_returns.shape[-1]
        roundoff = torch.as_tensor(
            np.ascontiguousarray(dH_roundoff),
            dtype=env.fdtype,
            device=env.device,
        )
        del dH_roundoff, all_data, raw_state, training, validation
        gc.collect()
        net_roundoff = config.roundoff_multiplier * FLOAT32_UNIT_ROUNDOFF * torch.maximum(
            torch.ones((), dtype=env.fdtype, device=env.device),
            option_returns.abs()
            + torch.cat((training_cost, validation_cost), dim=0)[..., 1:],
        )
        roundoff.add_(net_roundoff)
        del net_roundoff
        option_cost = torch.cat((training_cost, validation_cost), dim=0)[..., 1:]
        del training_cost, validation_cost

        states_per_split = training_paths * geometry.steps
        state_flat = normalized_state.reshape(args.paths * geometry.steps, -1)
        returns_flat = option_returns.reshape(args.paths * geometry.steps, instruments)
        cost_flat = option_cost.reshape(args.paths * geometry.steps, instruments)
        roundoff_flat = roundoff.reshape(args.paths * geometry.steps, instruments)
        reference_state = state_flat[:states_per_split]
        reference_returns = returns_flat[:states_per_split]
        reference_roundoff = roundoff_flat[:states_per_split]
        report.update(
            status="valuating_training",
            preparation_seconds=preparation_seconds,
            device=str(env.device),
            geometry=asdict(geometry),
            n_cost=n_cost,
            conditional_checkpoint=conditional_checkpoint,
            conditional_checkpoint_note=(
                "frozen conditional-variance model trained on an independent "
                "corrected-simulator universe; its variance is rescaled using "
                "this run's n_cost"
            ),
            state_training_mean=state_mean.cpu().tolist(),
            state_training_population_scale=state_scale.cpu().tolist(),
            bar_c=bar_c.cpu().tolist(),
            cost={
                "training": _cost_summary(option_cost[:training_paths]),
                "validation": _cost_summary(option_cost[training_paths:]),
            },
            memory_before_valuation=process_memory(env.device),
        )
        _atomic_json(report_path, report)

        def progress(value: LocalArbitrageProgress) -> None:
            processed = value.processed_states
            counts = {
                "long_only": value.long_only_candidate_states,
                "short_only": value.short_only_candidate_states,
                "both_directions": value.both_direction_candidate_states,
            }
            report["live_progress"] = {
                "split": value.split,
                "processed_states": processed,
                "total_states": value.total_states,
                "processed_fraction": processed / value.total_states,
                "candidate_states": counts,
                "candidate_state_rates": {
                    name: count / processed for name, count in counts.items()
                },
            }
            _atomic_json(report_path, report)
            print(
                f"{value.split}: {processed:,}/{value.total_states:,} states "
                f"({100.0 * processed / value.total_states:.2f}%); "
                f"cost-adjusted candidates long-only={counts['long_only']:,} "
                f"({100.0 * counts['long_only'] / processed:.3f}%), "
                f"short-only={counts['short_only']:,} "
                f"({100.0 * counts['short_only'] / processed:.3f}%), "
                f"both={counts['both_directions']:,} "
                f"({100.0 * counts['both_directions'] / processed:.3f}%)",
                flush=True,
            )

        mask_storage_root = (
            report_path.parent
            if args.mask_storage_root is None
            else Path(args.mask_storage_root)
        )
        mask_paths = {
            split: mask_storage_root / f"{report_path.stem}-{split}-mask.npy"
            for split in ("training", "validation")
        }
        training_writer = _PackedMaskWriter(
            mask_paths["training"], states=states_per_split, instruments=instruments
        )
        training_report = valuate_local_arbitrage_mask(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_roundoff=reference_roundoff,
            query_state=reference_state,
            query_returns=reference_returns,
            query_cost=cost_flat[:states_per_split],
            query_roundoff=roundoff_flat[:states_per_split],
            config=config,
            split="training",
            leave_one_out=True,
            candidate_sink=training_writer,
            progress=progress,
            progress_interval_seconds=args.progress_seconds,
        )
        training_writer.close()
        report.update(status="valuating_validation", training=training_report)
        _atomic_json(report_path, report)

        validation_writer = _PackedMaskWriter(
            mask_paths["validation"],
            states=states_per_split,
            instruments=instruments,
        )
        validation_report = valuate_local_arbitrage_mask(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_roundoff=reference_roundoff,
            query_state=state_flat[states_per_split:],
            query_returns=returns_flat[states_per_split:],
            query_cost=cost_flat[states_per_split:],
            query_roundoff=roundoff_flat[states_per_split:],
            config=config,
            split="validation",
            leave_one_out=False,
            candidate_sink=validation_writer,
            progress=progress,
            progress_interval_seconds=args.progress_seconds,
        )
        validation_writer.close()
        torch.cuda.synchronize(env.device)
        report.update(
            status="complete",
            training=training_report,
            validation=validation_report,
            mask_files={name: str(path) for name, path in mask_paths.items()},
            mask_storage={
                "format": "NumPy .npy uint8, little-endian packed option axis",
                "flattening": "path-major then time-major",
                "packed_shape": [states_per_split, 2, math.ceil(instruments / 8)],
                "unpacked_shape": [training_paths, geometry.steps, 2, instruments],
            },
            memory_after_valuation=process_memory(env.device),
        )
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--universe-seed-offset", type=int, default=100_000)
    parser.add_argument("--neighbors", type=int, default=10_240)
    parser.add_argument("--query-batch-size", type=int, default=2_048)
    parser.add_argument("--instrument-batch-size", type=int, default=10)
    parser.add_argument("--valuation-batch-size", type=int, default=4_096)
    parser.add_argument("--roundoff-multiplier", type=float, default=32.0)
    parser.add_argument("--cost-prop", type=float, default=0.0001)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--mask-storage-root")
    parser.add_argument(
        "--conditional-report",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "full-200000-e1000000-up-to10-per-cap-conditional-same.json"
        ),
    )
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.universe_seed_offset < 0:
        raise ValueError("--universe-seed-offset must be nonnegative")
    if not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0.0:
        raise ValueError("--progress-seconds must be finite and positive")
    report = run(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
