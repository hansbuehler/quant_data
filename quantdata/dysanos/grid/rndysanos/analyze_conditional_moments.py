"""Evaluate an existing conditional-moment checkpoint on its sampled states.

This diagnostic intentionally loads legacy checkpoints without applying the
current training-checkpoint compatibility gate.  It never resumes or mutates
training; it only reconstructs the matching model geometry and reads weights.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from quantdata.torch import BaseEnvironment, DeviceLock
from quantdata.torch.ffw import FeedForwardConfig

from .conditional_tail import moment_signal_statistics
from .torch_drift_change import (
    DysanosConfig,
    DysanosModel,
    UniverseGeometry,
    prepare_universe,
    split_training_validation,
)
from .torch_prep import RNDysanosExperiment, prep_experiment


def _model_config(value: dict[str, Any]) -> DysanosConfig:
    allowed = {item.name for item in fields(DysanosConfig)}
    config = {name: item for name, item in value.items() if name in allowed}
    for name in ("pf_config", "moment_config"):
        if isinstance(config.get(name), dict):
            config[name] = FeedForwardConfig(**config[name])
    return DysanosConfig(**config)


def _geometry(value: dict[str, Any]) -> UniverseGeometry:
    allowed = {item.name for item in fields(UniverseGeometry)}
    return UniverseGeometry(**{name: item for name, item in value.items() if name in allowed})


def _load_moment_state(model: DysanosModel, state: dict[str, torch.Tensor]) -> None:
    """Load only legacy moment subnetworks, ignoring unrelated OCE changes."""

    if model.drift is None or model.sqr is None:
        raise RuntimeError("checkpoint configuration has no conditional-moment model")
    for name, module in (("drift", model.drift), ("sqr", model.sqr)):
        prefix = f"{name}."
        module_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not module_state:
            raise RuntimeError(f"checkpoint contains no {name} subnetwork weights")
        module.load_state_dict(module_state, strict=True)


@torch.no_grad()
def analyze_split(
    model: DysanosModel,
    data: dict[str, torch.Tensor],
    *,
    batch_size: int,
    g: float,
    split: str,
) -> dict[str, Any]:
    """Aggregate the signal test without materializing all model outputs."""

    paths = data["z"].shape[0]
    output: dict[str, Any] | None = None
    per_instrument_count: torch.Tensor | None = None
    per_instrument_positive_sigma_count: torch.Tensor | None = None
    comparisons = 0
    states = 0
    observations_per_instrument = 0
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        drift, sqr, _ = model.moment_outputs(
            data["z"][start:stop], data["dH"][start:stop]
        )
        value = moment_signal_statistics(drift, sqr, g=g)
        batch_counts = torch.as_tensor(
            value.pop("per_instrument_hit_count"), dtype=torch.int64
        )
        batch_positive_sigma_counts = torch.as_tensor(
            value.pop("per_instrument_positive_sigma_hit_count"), dtype=torch.int64
        )
        value.pop("per_instrument_hit_fraction")
        if output is None:
            output = value
            per_instrument_count = batch_counts
            per_instrument_positive_sigma_count = batch_positive_sigma_counts
        else:
            for name in (
                "comparisons",
                "states",
                "hit_comparisons",
                "states_with_any_hit",
                "positive_hit_comparisons",
                "negative_hit_comparisons",
                "positive_sigma_hit_comparisons",
                "raw_negative_variance_comparisons",
                "zero_sigma_comparisons",
                "zero_sigma_nonzero_mean_comparisons",
            ):
                output[name] += value[name]
            output["instruments_with_any_hit"] = 0
            current_max = value["max_finite_abs_mean_over_sigma"]
            if current_max is not None:
                prior_max = output["max_finite_abs_mean_over_sigma"]
                output["max_finite_abs_mean_over_sigma"] = (
                    current_max if prior_max is None else max(prior_max, current_max)
                )
            assert per_instrument_count is not None
            per_instrument_count += batch_counts
            assert per_instrument_positive_sigma_count is not None
            per_instrument_positive_sigma_count += batch_positive_sigma_counts
        comparisons += value["comparisons"]
        states += value["states"]
        observations_per_instrument += drift[..., 0].numel()
    assert (
        output is not None
        and per_instrument_count is not None
        and per_instrument_positive_sigma_count is not None
    )
    output["split"] = split
    output["paths"] = paths
    output["steps"] = data["z"].shape[1]
    output["comparisons"] = comparisons
    output["states"] = states
    output["hit_comparison_fraction"] = output["hit_comparisons"] / comparisons
    output["positive_sigma_hit_comparison_fraction"] = (
        output["positive_sigma_hit_comparisons"] / comparisons
    )
    output["states_with_any_hit_fraction"] = output["states_with_any_hit"] / states
    output["instruments_with_any_hit"] = int((per_instrument_count > 0).sum().item())
    output["per_instrument_hit_count"] = per_instrument_count.tolist()
    output["per_instrument_positive_sigma_hit_count"] = (
        per_instrument_positive_sigma_count.tolist()
    )
    output["per_instrument_hit_fraction"] = (
        per_instrument_count.to(torch.float64) / observations_per_instrument
    ).tolist()
    order = torch.argsort(per_instrument_count, descending=True)
    output["top_instruments"] = [
        {
            "instrument": int(index),
            "hit_count": int(per_instrument_count[index]),
            "hit_fraction": float(per_instrument_count[index] / observations_per_instrument),
            "positive_sigma_hit_count": int(per_instrument_positive_sigma_count[index]),
            "positive_sigma_hit_fraction": float(
                per_instrument_positive_sigma_count[index] / observations_per_instrument
            ),
        }
        for index in order[: min(20, len(order))]
    ]
    return output


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _model_config(checkpoint["config"])
    stored_geometry = _geometry(checkpoint["geometry"])
    experiment = RNDysanosExperiment(
        num_samples=args.paths,
        pcnt_validation=args.validation_fraction,
        cnt_universe=1,
        universe_seed_offset=args.universe_seed_offset,
    )
    universe = prep_experiment(experiment, plot_sampler=False)[0]
    with DeviceLock() as device_lock:
        env = BaseEnvironment(device=device_lock.device, debug=False)
        data, geometry = prepare_universe(universe, config=config, env=env)
        if geometry != stored_geometry:
            raise RuntimeError(
                f"checkpoint geometry {stored_geometry} does not match sampled data {geometry}"
            )
        model = DysanosModel(env, geometry=geometry, config=config)
        _load_moment_state(model, checkpoint["model_state"])
        model.freeze_moments()
        model.eval()
        training, validation = split_training_validation(data, args.validation_fraction)
        report = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_schema": checkpoint.get("schema"),
            "checkpoint_moment_epoch": checkpoint.get("moment_epoch"),
            "paths": args.paths,
            "validation_fraction": args.validation_fraction,
            "universe_seed_offset": args.universe_seed_offset,
            "device": str(env.device),
            "g": args.g,
            "condition": "abs(E[Y|z]) > g*sqrt(max(E[Y^2|z]-E[Y|z]^2, 0))",
            "target": (
                "Y=D_prev*dH; this stage-1 checkpoint has D_prev=1 and dH "
                "contains spot plus delta-hedged option returns"
            ),
            "training": analyze_split(
                model,
                training,
                batch_size=args.batch_size,
                g=args.g,
                split="training",
            ),
            "validation": (
                None
                if validation is None
                else analyze_split(
                    model,
                    validation,
                    batch_size=args.batch_size,
                    g=args.g,
                    split="validation",
                )
            ),
        }
    _atomic_json(Path(args.report), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paths", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--universe-seed-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--g", type=float, default=3.0)
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.paths <= 0 or args.batch_size <= 0:
        raise ValueError("--paths and --batch-size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be strictly between zero and one")
    report = run(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["analyze_split", "main", "run"]
