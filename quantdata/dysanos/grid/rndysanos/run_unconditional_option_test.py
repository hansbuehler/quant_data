"""Test individual options for unconditional one-sided simulated returns.

The trading rule is deliberately independent of ``z``.  Raw sign imbalance is
measured on the full universe.  Cost-adjusted long and short directions use a
50/50 path split, with every cost calibrated from the training half.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.torch import BaseEnvironment, DeviceLock

from .run_direct_arbitrage import _conditional_cost_model
from .run_experiment import memory_snapshot
from .torch_drift_change import UniverseGeometry, validate_universe
from .torch_prep import RNDysanosExperiment, prep_experiment
from .unconditional_option_test import (
    DEFAULT_ROUNDOFF_MULTIPLIER,
    FLOAT32_UNIT_ROUNDOFF,
    SignAccumulator,
    delta_hedged_option_cost_notionals,
    delta_hedged_option_returns,
    delta_hedged_roundoff_band,
    net_return_roundoff_band,
    population_standard_error,
    rank_imbalance,
    rank_no_arbitrage_candidates,
)

_SENSITIVITY_MULTIPLIERS = (0.0, 1.0, 8.0, 32.0, 128.0)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_conditional_cost_model(
    *,
    report_path: Path,
    env: BaseEnvironment,
    data_geometry: UniverseGeometry,
) -> tuple[torch.nn.Module, str]:
    """Load a fixed moment model while allowing a new Monte Carlo path count."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_geometry = UniverseGeometry(
        samples=int(report["paths"]),
        steps=data_geometry.steps,
        expiries=data_geometry.expiries,
        strikes=data_geometry.strikes,
        state_variables=data_geometry.state_variables,
    )
    return _conditional_cost_model(
        report_path=report_path,
        env=env,
        geometry=checkpoint_geometry,
    )


def _path_slices(start: int, stop: int, batch_size: int):
    for left in range(start, stop, batch_size):
        yield slice(left, min(left + batch_size, stop))


def _option_batch(
    universe: dict[str, np.ndarray], paths: slice
) -> tuple[np.ndarray, np.ndarray]:
    dH = delta_hedged_option_returns(
        dC=universe["dC"][paths],
        delta=universe["delta"][paths],
        dS=universe["dS"][paths],
    )
    H = delta_hedged_option_cost_notionals(
        C=universe["C"][paths],
        delta=universe["delta"][paths],
        spot=universe["spot"][paths],
    )
    return dH, H


def _raw_signs(
    universe: dict[str, np.ndarray],
    *,
    geometry: UniverseGeometry,
    batch_size: int,
) -> dict[str, Any]:
    accumulators = {
        multiplier: SignAccumulator(geometry.expiries * geometry.strikes)
        for multiplier in _SENSITIVITY_MULTIPLIERS
    }
    positive_by_step = np.zeros(geometry.steps, dtype=np.int64)
    negative_by_step = np.zeros(geometry.steps, dtype=np.int64)
    neutral_by_step = np.zeros(geometry.steps, dtype=np.int64)
    for paths in _path_slices(0, geometry.samples, batch_size):
        dH = delta_hedged_option_returns(
            dC=universe["dC"][paths],
            delta=universe["delta"][paths],
            dS=universe["dS"][paths],
        )
        for multiplier, accumulator in accumulators.items():
            tolerance = delta_hedged_roundoff_band(
                C=universe["C"][paths],
                dC=universe["dC"][paths],
                delta=universe["delta"][paths],
                dS=universe["dS"][paths],
                multiplier=multiplier,
            )
            accumulator.update(dH, tolerance)
            if multiplier == DEFAULT_ROUNDOFF_MULTIPLIER:
                positive_by_step += np.sum(dH > tolerance, axis=(0, 2))
                negative_by_step += np.sum(dH < -tolerance, axis=(0, 2))
                neutral_by_step += np.sum(
                    np.abs(dH) <= tolerance, axis=(0, 2)
                )
    observations_by_step = geometry.samples * geometry.expiries * geometry.strikes
    return {
        "definition": "imbalance = abs(count(dH < -e) - count(dH > e))",
        "recommended_multiplier": DEFAULT_ROUNDOFF_MULTIPLIER,
        "recommended_top": rank_imbalance(
            accumulators[DEFAULT_ROUNDOFF_MULTIPLIER],
            strikes=geometry.strikes,
            count=geometry.expiries * geometry.strikes,
        ),
        "by_transition": [
            {
                "transition": step,
                "positive": int(positive_by_step[step]),
                "negative": int(negative_by_step[step]),
                "neutral": int(neutral_by_step[step]),
                "positive_fraction": (
                    float(positive_by_step[step]) / observations_by_step
                ),
                "negative_fraction": (
                    float(negative_by_step[step]) / observations_by_step
                ),
            }
            for step in range(geometry.steps)
        ],
        "sensitivity": {
            f"{multiplier:g}": rank_imbalance(
                accumulator,
                strikes=geometry.strikes,
                count=20,
            )
            for multiplier, accumulator in accumulators.items()
        },
    }


def _unconditional_standard_error(
    universe: dict[str, np.ndarray],
    *,
    stop: int,
    geometry: UniverseGeometry,
    batch_size: int,
) -> np.ndarray:
    J = geometry.expiries * geometry.strikes
    total = np.zeros(J, dtype=np.float64)
    total_sqr = np.zeros(J, dtype=np.float64)
    observations = 0
    for paths in _path_slices(0, stop, batch_size):
        dH, _ = _option_batch(universe, paths)
        dH64 = dH.astype(np.float64, copy=False)
        total += dH64.sum(axis=(0, 1), dtype=np.float64)
        total_sqr += np.square(dH64).sum(axis=(0, 1), dtype=np.float64)
        observations += dH.shape[0] * dH.shape[1]
    expected = stop * geometry.steps
    if observations != expected:
        raise RuntimeError(f"cost observations {observations} != {expected}")
    return population_standard_error(total, total_sqr, observations)


def _conditional_variance(
    model: torch.nn.Module,
    universe: dict[str, np.ndarray],
    paths: slice,
    *,
    geometry: UniverseGeometry,
    env: BaseEnvironment,
    option_dH: np.ndarray,
) -> np.ndarray:
    S = option_dH.shape[0]
    TAU = geometry.steps
    xstate = torch.as_tensor(
        universe["xstate"][paths, :TAU], dtype=env.fdtype, device=env.device
    )
    log_strikes = torch.as_tensor(
        universe["log_strikes"][paths].reshape(S, TAU, -1),
        dtype=env.fdtype,
        device=env.device,
    )
    expiries = torch.as_tensor(
        universe["expiries"].reshape(1, 1, geometry.expiries),
        dtype=env.fdtype,
        device=env.device,
    ).expand(S, TAU, geometry.expiries)
    z = torch.cat((xstate, log_strikes, expiries), dim=-1)
    spot_dH = torch.as_tensor(
        universe["dS"][paths, :, None], dtype=env.fdtype, device=env.device
    )
    options = torch.as_tensor(option_dH, dtype=env.fdtype, device=env.device)
    dH = torch.cat((spot_dH, options), dim=-1)
    with torch.inference_mode():
        _, _, variance = model.moment_outputs(z, dH)
    output = variance[..., 1:].cpu().numpy()
    if not np.isfinite(output).all() or np.any(output < 0.0):
        raise RuntimeError("conditional variance is non-finite or negative")
    return output


def _direction_report(
    training_long: SignAccumulator,
    training_short: SignAccumulator,
    validation_long: SignAccumulator,
    validation_short: SignAccumulator,
    *,
    strikes: int,
) -> dict[str, Any]:
    count = 2 * training_long.instruments
    training = rank_no_arbitrage_candidates(
        training_long, training_short, strikes=strikes, count=count
    )
    validation = rank_no_arbitrage_candidates(
        validation_long, validation_short, strikes=strikes, count=count
    )
    validation_by_key = {
        (item["option"], item["direction"]): item for item in validation
    }
    paired = [
        {
            "training": item,
            "validation": validation_by_key[(item["option"], item["direction"])],
        }
        for item in training[:20]
    ]
    return {
        "selection_rule": (
            "training fewest negative observations, then fewest losing paths, "
            "then largest path-clustered 95% lower mean"
        ),
        "training_selected_with_validation": paired,
        "training_empirical_no_loss_count": sum(
            item["negative"] == 0 and item["mean_lower_95"] > 0.0
            for item in training
        ),
        "validation_empirical_no_loss_count": sum(
            item["negative"] == 0 and item["mean_lower_95"] > 0.0
            for item in validation
        ),
        "training_selected_validated_count": sum(
            pair["training"]["negative"] == 0
            and pair["training"]["mean_lower_95"] > 0.0
            and pair["validation"]["negative"] == 0
            and pair["validation"]["mean_lower_95"] > 0.0
            for pair in paired
        ),
    }


def _costed_signs(
    universe: dict[str, np.ndarray],
    *,
    geometry: UniverseGeometry,
    batch_size: int,
    conditional_model: torch.nn.Module,
    env: BaseEnvironment,
    cost_prop: float,
) -> dict[str, Any]:
    split = geometry.samples // 2
    J = geometry.expiries * geometry.strikes
    n_cost = split * geometry.steps
    bar_c = _unconditional_standard_error(
        universe,
        stop=split,
        geometry=geometry,
        batch_size=batch_size,
    )
    accumulators = {
        mode: {
            name: SignAccumulator(J)
            for name in ("training_long", "training_short", "validation_long", "validation_short")
        }
        for mode in ("unconditional", "conditional")
    }
    binding_cond = np.zeros(J, dtype=np.int64)
    binding_prop = np.zeros(J, dtype=np.int64)
    binding_uncond = np.zeros(J, dtype=np.int64)
    binding_prop_uncond = np.zeros(J, dtype=np.int64)
    conditional_observations = 0

    for split_name, start, stop in (
        ("training", 0, split),
        ("validation", split, geometry.samples),
    ):
        for paths in _path_slices(start, stop, batch_size):
            dH, H = _option_batch(universe, paths)
            proportional = cost_prop * np.abs(H)
            unconditional = np.maximum(
                proportional, bar_c.reshape(1, 1, -1)
            )
            variance = _conditional_variance(
                conditional_model,
                universe,
                paths,
                geometry=geometry,
                env=env,
                option_dH=dH,
            )
            conditional_component = np.sqrt(variance / float(n_cost))
            conditional = np.maximum(proportional, conditional_component)
            binding_uncond += (
                bar_c.reshape(1, 1, -1) >= proportional
            ).sum(axis=(0, 1), dtype=np.int64)
            binding_prop_uncond += (
                proportional >= bar_c.reshape(1, 1, -1)
            ).sum(axis=(0, 1), dtype=np.int64)
            binding_cond += (conditional_component >= proportional).sum(
                axis=(0, 1), dtype=np.int64
            )
            binding_prop += (proportional >= conditional_component).sum(
                axis=(0, 1), dtype=np.int64
            )
            conditional_observations += dH.shape[0] * dH.shape[1]

            for mode, cost in (
                ("unconditional", unconditional),
                ("conditional", conditional),
            ):
                tolerance = net_return_roundoff_band(dH, cost)
                accumulators[mode][f"{split_name}_long"].update(dH - cost, tolerance)
                accumulators[mode][f"{split_name}_short"].update(-dH - cost, tolerance)

    report: dict[str, Any] = {
        "n_cost": n_cost,
        "training_paths": split,
        "validation_paths": geometry.samples - split,
        "cost_prop": cost_prop,
        "unconditional_bar_c": bar_c.tolist(),
        "conditional_binding_fraction": (
            binding_cond / float(conditional_observations)
        ).tolist(),
        "proportional_binding_fraction_in_conditional_mode": (
            binding_prop / float(conditional_observations)
        ).tolist(),
        "unconditional_binding_fraction": (
            binding_uncond / float(conditional_observations)
        ).tolist(),
        "proportional_binding_fraction_in_unconditional_mode": (
            binding_prop_uncond / float(conditional_observations)
        ).tolist(),
    }
    for mode, values in accumulators.items():
        report[mode] = _direction_report(
            values["training_long"],
            values["training_short"],
            values["validation_long"],
            values["validation_short"],
            strikes=geometry.strikes,
        )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.paths <= 0 or args.paths % 2:
        raise ValueError("--paths must be a positive even number")
    experiment = RNDysanosExperiment(
        num_samples=args.paths,
        pcnt_validation=0.5,
        cnt_universe=1,
        universe_seed_offset=args.universe_seed_offset,
    )
    started = time.monotonic()
    universe = prep_experiment(experiment, plot_sampler=False)[0]
    geometry = validate_universe(universe)
    if geometry.samples != args.paths:
        raise RuntimeError(f"prepared {geometry.samples} paths, expected {args.paths}")
    prepared_seconds = time.monotonic() - started

    raw_started = time.monotonic()
    raw = _raw_signs(universe, geometry=geometry, batch_size=args.batch_size)
    raw_seconds = time.monotonic() - raw_started

    raw_only = getattr(args, "raw_only", False)
    if raw_only:
        checkpoint = None
        costed = None
        cost_seconds = 0.0
        memory_before = None
        memory_after = None
        device = None
    else:
        with DeviceLock() as device_lock:
            env = BaseEnvironment(device=device_lock.device, debug=False)
            conditional_model, checkpoint = _load_conditional_cost_model(
                report_path=Path(args.conditional_report),
                env=env,
                data_geometry=geometry,
            )
            memory_before = memory_snapshot(env.device)
            cost_started = time.monotonic()
            costed = _costed_signs(
                universe,
                geometry=geometry,
                batch_size=args.batch_size,
                conditional_model=conditional_model,
                env=env,
                cost_prop=args.cost_prop,
            )
            if env.device.type == "cuda":
                torch.cuda.synchronize(env.device)
            cost_seconds = time.monotonic() - cost_started
            memory_after = memory_snapshot(env.device)
            device = str(env.device)
        del conditional_model
    del universe
    gc.collect()

    report = {
        "method": "state-independent single-option long/short sign test",
        "return_semantics": "delta-hedged, non-cash-normalized option dH",
        "paths": args.paths,
        "observations": args.paths * geometry.steps,
        "universe_seed_offset": args.universe_seed_offset,
        "geometry": {
            "samples": geometry.samples,
            "steps": geometry.steps,
            "expiries": geometry.expiries,
            "strikes": geometry.strikes,
            "state_variables": geometry.state_variables,
        },
        "float_dtype": "float32",
        "unit_roundoff": FLOAT32_UNIT_ROUNDOFF,
        "roundoff_multiplier": DEFAULT_ROUNDOFF_MULTIPLIER,
        "roundoff_definition": (
            "raw: 32*u*max(1, abs(C_tp1)+abs(C_t)+abs(delta*dS)); "
            "net: 32*u*max(1, abs(dH)+cost)"
        ),
        "conditional_checkpoint": checkpoint,
        "device": device,
        "preparation_seconds": prepared_seconds,
        "raw_seconds": raw_seconds,
        "cost_seconds": cost_seconds,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "raw": raw,
        "costed": costed,
        "limitations": (
            "Raw-only pre-network sign gate; no conditional-moment estimator "
            "was loaded."
            if raw_only
            else (
                "The frozen conditional-moment estimator was trained previously "
                "on an independent universe and is reused at both sample sizes "
                "to isolate the effect of the new Monte Carlo universe and "
                "n_cost scaling."
            )
        ),
    }
    _atomic_json(Path(args.report), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, required=True)
    parser.add_argument("--universe-seed-offset", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--cost-prop", type=float, default=0.0001)
    parser.add_argument(
        "--conditional-report",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "full-200000-e1000000-up-to10-per-cap-conditional-same.json"
        ),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="stop after the pre-network raw sign-imbalance gate",
    )
    return parser


def main() -> None:
    report = run(_parser().parse_args())
    summary = {
        "report": report["paths"],
        "worst_raw": report["raw"]["recommended_top"][:5],
        "by_transition": report["raw"]["by_transition"],
    }
    if report["costed"] is not None:
        summary.update(
            unconditional=report["costed"]["unconditional"],
            conditional=report["costed"]["conditional"],
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
