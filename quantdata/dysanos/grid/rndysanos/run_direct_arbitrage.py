"""Search independently simulated Dysanos paths for direct conditional arbitrage.

Example
-------
::

    python -m quantdata.dysanos.grid.rndysanos.run_direct_arbitrage \
        --paths 200000 --epochs 200000 --cost-mode combined --seed 70771 \
        --report quantdata/dysanos/grid/rndysanos/reports/direct-arbitrage.json
"""

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
from cdxcore.err import verify_inp

from quantdata.torch import BaseEnvironment, DeviceLock

from .direct_arbitrage import (
    DirectArbitrageConfig,
    DirectArbitrageTrainConfig,
    effective_cost,
    train_direct_policy,
    unconditional_cost,
)
from .local_arbitrage import unpack_direction_mask
from .run_experiment import (
    _counter_difference,
    _dynamo_counters,
    memory_snapshot,
    pair_training_validation_universes,
)
from .torch_drift_change import (
    DysanosConfig,
    DysanosModel,
    prepare_universe,
    split_training_validation,
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


def _conditional_cost_model(
    *,
    report_path: Path,
    env: BaseEnvironment,
    geometry: Any,
    allow_different_sample_count: bool = False,
) -> tuple[DysanosModel, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    regime = report.get("regimes", {}).get("conditional")
    if regime is None or not regime.get("stages"):
        raise ValueError(f"{report_path} contains no conditional model stage")
    checkpoint_path = Path(regime["stages"][0]["checkpoint"]).expanduser()
    state = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
    stored_config = state.get("config")
    if not isinstance(stored_config, dict):
        raise RuntimeError(f"checkpoint {checkpoint_path} has no model configuration")
    config = DysanosConfig(
        label=stored_config["label"],
        group=stored_config["group"],
        user_version=stored_config["user_version"],
        utility=stored_config["utility"],
        lmbda=stored_config["lmbda"],
        delta_hedged=stored_config["delta_hedged"],
        cash_returns=stored_config["cash_returns"],
        learn_cost=stored_config["learn_cost"],
        cost_cond=stored_config["cost_cond"],
        cost_uncond=stored_config["cost_uncond"],
        cost_prop=stored_config["cost_prop"],
        seed=stored_config["seed"],
        action_scale=stored_config["action_scale"],
        drift_scale=stored_config["drift_scale"],
        sqr_scale=stored_config["sqr_scale"],
        cash_floor=stored_config["cash_floor"],
        density_floor=stored_config["density_floor"],
        denominator_floor=stored_config["denominator_floor"],
    )
    verify_inp(
        config.delta_hedged and not config.cash_returns,
        "conditional cost checkpoint has incompatible return semantics",
    )
    verify_inp(
        float(config.cost_cond) == 1.0,
        lambda: f"conditional cost multiplier must be one, got {config.cost_cond}",
    )
    expected_geometry = asdict(geometry)
    stored_geometry = state.get("geometry")
    if allow_different_sample_count and isinstance(stored_geometry, dict):
        stored_geometry = {
            name: value
            for name, value in stored_geometry.items()
            if name != "samples"
        }
        expected_geometry = {
            name: value
            for name, value in expected_geometry.items()
            if name != "samples"
        }
    if stored_geometry != expected_geometry:
        raise RuntimeError(
            f"checkpoint geometry {stored_geometry} != {expected_geometry}"
        )
    if state.get("phase") != "main" or int(state.get("moment_epoch", 0)) <= 0:
        raise RuntimeError("conditional moment checkpoint is unfinished")
    model = DysanosModel(env, geometry=geometry, config=config)
    incompatible = model.load_state_dict(state["model_state"], strict=False)
    unexpected = set(incompatible.unexpected_keys)
    if not unexpected.issubset({"offset.value"}) or incompatible.missing_keys:
        raise RuntimeError(
            "conditional cost checkpoint has incompatible model tensors: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.freeze_moments()
    model.requires_grad_(False)
    model.eval()
    return model, str(checkpoint_path)


def _label_top_instruments(
    valuation: dict[str, Any], *, strikes: int
) -> None:
    for item in valuation["top_instruments_by_mean_gross_exposure"]:
        instrument = item["instrument"]
        if instrument == 0:
            item.update(kind="spot", expiry_index=None, strike_index=None)
        else:
            option = instrument - 1
            item.update(
                kind="delta_hedged_option",
                expiry_index=option // strikes,
                strike_index=option % strikes,
            )


def _excluded_direction_mask(
    *,
    mask_path: Path,
    paths: int,
    steps: int,
    option_instruments: int,
    total_instruments: int,
    device: torch.device,
) -> torch.Tensor:
    packed_array = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    expected = (paths * steps, 2, math.ceil(option_instruments / 8))
    if packed_array.shape != expected or packed_array.dtype != np.uint8:
        raise ValueError(
            f"packed mask {mask_path} must have shape/dtype {expected}/uint8, "
            f"got {packed_array.shape}/{packed_array.dtype}"
        )
    packed = torch.tensor(packed_array, dtype=torch.uint8, device=device)
    options = unpack_direction_mask(
        packed, instruments=option_instruments
    ).reshape(paths, steps, 2, option_instruments)
    del packed
    excluded = torch.zeros(
        paths,
        steps,
        2 * total_instruments,
        dtype=torch.bool,
        device=device,
    )
    excluded[..., 1:total_instruments] = options[..., 0, :]
    excluded[..., total_instruments + 1 :] = options[..., 1, :]
    return excluded


def run(
    args: argparse.Namespace,
    *,
    prepared_universe: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if args.paths <= 0 or args.paths % 2:
        raise ValueError("--paths must be a positive even number")
    preparation_started = time.monotonic()
    if prepared_universe is None:
        experiment = RNDysanosExperiment(
            num_samples=args.paths // 2,
            pcnt_validation=0.5,
            cnt_universe=2,
            universe_seed_offset=args.universe_seed_offset,
        )
        source = prep_experiment(experiment, plot_sampler=False)
        paired = pair_training_validation_universes(source, consume=True)
        del source
        universe = paired.pop()
        del paired
    else:
        universe = prepared_universe
    gc.collect()
    preparation_seconds = time.monotonic() - preparation_started

    with DeviceLock() as device_lock:
        env = BaseEnvironment(device=device_lock.device, debug=False)
        data_config = DysanosConfig(
            label="direct-arbitrage-data",
            delta_hedged=True,
            cash_returns=False,
            cost_cond=0.0,
            cost_uncond=1.0,
        )
        all_data, geometry = prepare_universe(
            universe, config=data_config, env=env
        )
        if geometry.samples != args.paths:
            raise RuntimeError(
                f"prepared universe has {geometry.samples} paths, expected {args.paths}"
            )
        del universe
        training, validation = split_training_validation(all_data, 0.5)
        del all_data
        assert validation is not None
        training_paths = training["z"].shape[0]
        validation_paths = validation["z"].shape[0]
        if training_paths != args.paths // 2 or validation_paths != args.paths // 2:
            raise RuntimeError(
                f"expected equal {args.paths // 2}-path splits, found "
                f"{training_paths} and {validation_paths}"
            )
        n_cost = training_paths * geometry.steps
        bar_c = unconditional_cost(
            training,
            cost_uncond=1.0,
            batch_size=args.valuation_batch_size,
        )
        conditional_model = None
        conditional_checkpoint = None
        if args.cost_mode == "combined":
            conditional_model, conditional_checkpoint = _conditional_cost_model(
                report_path=Path(args.conditional_report),
                env=env,
                geometry=geometry,
                allow_different_sample_count=prepared_universe is not None,
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
        if args.state_mode == "raw":
            training["z"] = training["z"][
                ..., : geometry.state_variables
            ].clone()
            validation["z"] = validation["z"][
                ..., : geometry.state_variables
            ].clone()
        training_policy = {
            "z": training["z"],
            "dH": training["dH"],
            "cost": training_cost,
        }
        validation_policy = {
            "z": validation["z"],
            "dH": validation["dH"],
            "cost": validation_cost,
        }
        mask_report: dict[str, Any] | None = None
        if args.arbitrage_mask_report is not None:
            mask_report_path = Path(args.arbitrage_mask_report)
            mask_report = json.loads(mask_report_path.read_text(encoding="utf-8"))
            if mask_report.get("status") != "complete":
                raise RuntimeError(f"arbitrage mask report {mask_report_path} is incomplete")
            if (
                mask_report.get("paths") != args.paths
                or mask_report.get("universe_seed_offset") != args.universe_seed_offset
            ):
                raise RuntimeError(
                    f"arbitrage mask report {mask_report_path} uses a different universe"
                )
            mask_files = mask_report.get("mask_files", {})
            training_policy["excluded_directions"] = _excluded_direction_mask(
                mask_path=Path(mask_files["training"]),
                paths=training_paths,
                steps=geometry.steps,
                option_instruments=geometry.expiries * geometry.strikes,
                total_instruments=geometry.instruments,
                device=env.device,
            )
            validation_policy["excluded_directions"] = _excluded_direction_mask(
                mask_path=Path(mask_files["validation"]),
                paths=validation_paths,
                steps=geometry.steps,
                option_instruments=geometry.expiries * geometry.strikes,
                total_instruments=geometry.instruments,
                device=env.device,
            )
        del training, validation
        gc.collect()

        config = DirectArbitrageConfig(
            label=(
                f"{args.objective}-{args.state_mode}-"
                f"{'masked' if mask_report is not None else 'unmasked'}"
                f"-seed-{args.seed}"
            ),
            group=(
                f"validation/{args.paths}-paths/{args.objective}-{args.state_mode}/"
                f"{'masked' if mask_report is not None else 'unmasked'}"
            ),
            user_version="1.0",
            state_features=training_policy["z"].shape[-1],
            instruments=geometry.instruments,
            epsilon=args.epsilon,
            delta=args.delta,
            numerical_tolerance=args.numerical_tolerance,
            smooth_temperature=args.smooth_temperature,
            tail_fraction=args.tail_fraction,
            downside_weight=args.downside_weight,
            tail_weight=args.tail_weight,
            overlap_weight=args.overlap_weight,
            mean_reward_weight=args.mean_reward_weight,
            discovery_fraction=args.discovery_fraction,
            constraint_ramp_fraction=args.constraint_ramp_fraction,
            initial_trade_fraction=args.initial_trade_fraction,
            mask_arbitrage_candidates=mask_report is not None,
            objective=args.objective,
            cap_weight=args.cap_weight,
            seed=args.seed,
        )
        train_config = DirectArbitrageTrainConfig(
            batch_size=args.batch_size,
            valuation_batch_size=args.valuation_batch_size,
            lr=args.lr,
            final_lr=args.final_lr,
            weight_decay=args.weight_decay,
            compile=not args.no_compile,
            value_update=args.value_update,
            checkpoint_update=args.checkpoint_update,
            checkpoint_root=args.checkpoint_root,
        )
        memory_before = memory_snapshot(env.device)
        compilation_before = _dynamo_counters()
        started = time.monotonic()
        result = train_direct_policy(
            training_policy,
            validation_policy,
            config=config,
            train_config=train_config,
            epochs=args.epochs,
            env=env,
            restart=args.restart,
        )
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        elapsed_seconds = time.monotonic() - started
        compilation = _counter_difference(_dynamo_counters(), compilation_before)
        _label_top_instruments(result.training, strikes=geometry.strikes)
        assert result.validation is not None
        _label_top_instruments(result.validation, strikes=geometry.strikes)
        report = {
            "method": (
                "state-only unbounded signed output with radial L1 clipping"
                if args.objective == "radial-l1-v2"
                else "state-only long/short softmax direction with sigmoid exposure gate"
            ),
            "objective_name": (
                "V2.0"
                if args.objective == "radial-l1-v2"
                else "legacy simplex soft-hit"
            ),
            "objective_formula": (
                "-E[X] + lambda_minus E[(-X)_+] + "
                "lambda_tail CVaR_tail((-X)_+) + "
                "lambda_cap E[((||v||_1-1)_+)^2]"
                if args.objective == "radial-l1-v2"
                else None
            ),
            "state_mode": args.state_mode,
            "state_features": training_policy["z"].shape[-1],
            "arbitrage_mask_report": args.arbitrage_mask_report,
            "arbitrage_mask_definition": (
                None if mask_report is None else mask_report.get("arb_function")
            ),
            "return_semantics": "delta-hedged, non-cash-normalized dH",
            "gross_constraint": (
                "a=v/max(1,||v||_1), hence ||a||_1<=1 at every state"
                if args.objective == "radial-l1-v2"
                else "sum(long + short) <= 1 at every state"
            ),
            "cost_mode": args.cost_mode,
            "cost_definition": (
                "max(conditional standard error, unconditional standard error, "
                "one-basis-point proportional cost)"
                if args.cost_mode == "combined"
                else "max(unconditional standard error, one-basis-point proportional cost)"
            ),
            "conditional_checkpoint": conditional_checkpoint,
            "paths": args.paths,
            "training_paths": training_paths,
            "validation_paths": validation_paths,
            "n_cost": n_cost,
            "epochs": args.epochs,
            "universe_seed_offset": args.universe_seed_offset,
            "preparation_seconds": preparation_seconds,
            "elapsed_seconds": elapsed_seconds,
            "device": str(env.device),
            "geometry": asdict(geometry),
            "config": asdict(config),
            "train_config": asdict(train_config),
            "checkpoint": str(result.checkpoint_path),
            "restarted": result.restarted,
            "compiled_graphs": compilation,
            "memory_before": memory_before,
            "memory_after": memory_snapshot(env.device),
            "bar_c": bar_c.cpu().numpy().tolist(),
            "training": result.training,
            "validation": result.validation,
            "tracker": result.tracker,
            "conclusion": {
                "training_qualifies": result.training["qualifies_empirically"],
                "validation_qualifies": result.validation["qualifies_empirically"],
                "population_almost_sure_claim": False,
                "reason": (
                    "Monte Carlo validation can reject candidates and bound loss "
                    "probability, but cannot prove a population almost-sure statement."
                ),
            },
        }
        _atomic_json(Path(args.report), report)
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=200_000)
    parser.add_argument(
        "--cost-mode", choices=["unconditional", "combined"], required=True
    )
    parser.add_argument("--state-mode", choices=["full", "raw"], default="full")
    parser.add_argument("--arbitrage-mask-report")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--universe-seed-offset", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--numerical-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--smooth-temperature", type=float, default=0.0025)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--downside-weight", type=float, default=10.0)
    parser.add_argument("--tail-weight", type=float, default=100.0)
    parser.add_argument("--overlap-weight", type=float, default=1.0)
    parser.add_argument("--mean-reward-weight", type=float, default=0.1)
    parser.add_argument("--discovery-fraction", type=float, default=0.05)
    parser.add_argument("--constraint-ramp-fraction", type=float, default=0.45)
    parser.add_argument("--initial-trade-fraction", type=float, default=0.01)
    parser.add_argument(
        "--objective",
        choices=["radial-l1-v2", "simplex-soft-hit"],
        default="radial-l1-v2",
    )
    parser.add_argument("--cap-weight", type=float, default=1.0)
    parser.add_argument("--cost-prop", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--valuation-batch-size", type=int, default=4_096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--final-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--value-update", type=int, default=10_000)
    parser.add_argument("--checkpoint-update", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint-root", default="~/.cache/rndysanos/direct-arbitrage"
    )
    parser.add_argument(
        "--conditional-report",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "full-200000-e1000000-up-to10-per-cap-conditional-same.json"
        ),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
