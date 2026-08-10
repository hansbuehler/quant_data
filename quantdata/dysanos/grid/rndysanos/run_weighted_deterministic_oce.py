"""Solve the deterministic complete-path powerexp benchmark under exact IS.

The action is one unrestricted constant matrix ``a[t, instrument]``.  The
tail-complete Gaussian proposal is weighted back to the fitted physical
measure in the objective, validation statistics, and martingale diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from quantdata.dysanos.grid.gendysanos.torch_prep import (
    GenDysanosExperiment,
    prepare_terminal_path_experiment,
)
from quantdata.dysanos.grid.torch_simulator import GaussianSpotImportanceConfig
from quantdata.torch import BaseEnvironment, DeviceLock

from .run_experiment import (
    _atomic_json,
    _baseline_split,
    _counter_difference,
    _dynamo_counters,
    _stage_report,
)
from .torch_drift_change import DysanosConfig, TrainConfig, train


def run(args: argparse.Namespace) -> dict[str, object]:
    """Generate one universe and solve the full-training L-BFGS problem."""

    if args.paths <= 0 or args.paths % 2:
        raise ValueError("'paths' must be positive and even")
    output = Path(args.output)
    started = time.monotonic()
    universe, diagnostics = prepare_terminal_path_experiment(
        GenDysanosExperiment(
            num_samples=args.paths,
            mode="path",
            option_return_horizon=args.option_return_horizon,
            num_steps=args.trading_state_steps,
            num_alpha=None,
            use_matrix_exp=True,
            bandwidth=1.0,
            seed=args.seed,
            block_size=args.generator_block_size,
            importance_sampling=(
                GaussianSpotImportanceConfig(
                    spot_std_multiplier=args.importance_spot_std_multiplier,
                    base_probability=args.importance_base_probability,
                    target_action_count=args.importance_target_action_count,
                )
                if args.importance_sampling
                else None
            ),
        )
    )
    generation_seconds = time.monotonic() - started
    with DeviceLock() as device_lock:
        env = BaseEnvironment(
            precision=args.precision,
            device=device_lock.device,
            debug=False,
        )
        hedge_label = "dh" if args.delta_hedged else "unhedged"
        scale_label = "cash" if args.cash_returns else "raw"
        horizon_label = args.option_return_horizon.replace("-", "")
        sampling_label = "is" if args.importance_sampling else "directp"
        config = DysanosConfig(
            label=(
                f"weighted-deterministic-powexp-p{args.precision}-"
                f"{hedge_label}-{scale_label}-{horizon_label}-{sampling_label}-"
                f"{args.paths}-{args.seed}"
            ),
            group=(
                f"gendysanos/{sampling_label}-path/{args.paths}/seed-{args.seed}/"
                f"powexp-deterministic-p{args.precision}-{hedge_label}-"
                f"{scale_label}-{horizon_label}"
            ),
            user_version="3.1",
            utility="powexp",
            lmbda=1.0,
            oce_variant="simplified",
            action_type="deterministic",
            deterministic_precondition=args.precondition,
            deterministic_preprocess=args.preprocess,
            state_mode="raw",
            delta_hedged=args.delta_hedged,
            cash_returns=args.cash_returns,
            learn_cost=False,
            cost_cond=0.0,
            cost_uncond=0.0,
            cost_prop=args.cost_prop,
        )
        train_config = TrainConfig(
            trn_batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            validation_fraction=0.5,
            optimizer="SciPy",
            lr=args.lr,
            final_lr=args.lr,
            weight_decay=0.0,
            compile=not args.no_compile,
            value_seconds=args.value_seconds,
            value_update=args.value_update,
            checkpoint_seconds=args.checkpoint_seconds,
            checkpoint_update=args.checkpoint_update,
            heavy_output_update=args.epochs,
            early_stopping=True,
            minimum_epochs=args.minimum_epochs,
            convergence_points=args.convergence_points,
            convergence_relative_tolerance=(
                args.convergence_relative_tolerance
            ),
            require_cost_band=False,
            scipy_gradient_tolerance=args.scipy_gradient_tolerance,
            scipy_function_tolerance=args.scipy_function_tolerance,
            scipy_max_line_search_steps=args.scipy_max_line_search_steps,
            checkpoint_root=args.checkpoint_root,
        )
        before_compile = _dynamo_counters()
        training_started = time.monotonic()
        result = train(
            universe,
            config=config,
            train_config=train_config,
            epochs=args.epochs,
            moment_epochs=1,
            restart=args.restart,
            env=env,
        )
        torch.cuda.synchronize(env.device)
        stage = _stage_report(
            result=result,
            elapsed_seconds=time.monotonic() - training_started,
            device=env.device,
            compilation=_counter_difference(
                _dynamo_counters(), before_compile
            ),
        )
        report: dict[str, object] = {
            "status": "complete",
            "experiment": "likelihood-weighted deterministic path powerexp",
            "objective": (
                "sup_a E_P[u(sum (a+b)*dH)] after separable b preprocessing"
                if args.preprocess
                else "sup_{a[t,omega]} E_P[u(sum a*dH)]"
            ),
            "paths": args.paths,
            "training_paths": args.paths // 2,
            "validation_paths": args.paths // 2,
            "return_semantics": (
                (
                    f"{args.option_return_horizon} delta-hedged P&L"
                    if args.delta_hedged
                    else f"{args.option_return_horizon} unhedged option P&L"
                )
                + (
                    " divided by initial gross notional"
                    if args.cash_returns
                    else " in raw normalized-dollar units"
                )
            ),
            "cost_prop": args.cost_prop,
            "importance_semantics": (
                "all objectives, normalizers, means, and MC errors include p/q"
                if args.importance_sampling
                else "direct sampling under P; all path weights equal one"
            ),
            "generation_seconds": generation_seconds,
            "sampling_diagnostics": asdict(diagnostics),
            "model_config": asdict(config),
            "train_config": asdict(train_config),
            "device": str(env.device),
            "baseline": {
                "training": _baseline_split(result.training),
                "validation": _baseline_split(result.validation),
            },
            "stage": stage,
        }
        _atomic_json(output, report)
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=700_000)
    parser.add_argument("--trading-state-steps", type=int, default=10)
    parser.add_argument(
        "--option-return-horizon",
        choices=["expiry", "one-day"],
        default="one-day",
    )
    parser.add_argument("--generator-block-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--val-batch-size", type=int, default=8_192)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--precision", type=int, choices=[32, 64], default=64)
    parser.add_argument(
        "--delta-hedged",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cash-returns",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--precondition",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cost-prop", type=float, default=0.0)
    parser.add_argument("--value-update", type=int, default=5)
    parser.add_argument("--value-seconds", type=float, default=1_800.0)
    parser.add_argument("--checkpoint-update", type=int, default=5)
    parser.add_argument("--checkpoint-seconds", type=float, default=1_800.0)
    parser.add_argument("--convergence-points", type=int, default=7)
    parser.add_argument(
        "--convergence-relative-tolerance", type=float, default=1.0e-4
    )
    parser.add_argument("--scipy-gradient-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--scipy-function-tolerance", type=float, default=0.0)
    parser.add_argument("--scipy-max-line-search-steps", type=int, default=50)
    parser.add_argument("--importance-spot-std-multiplier", type=float, default=8.0)
    parser.add_argument("--importance-base-probability", type=float, default=0.1)
    parser.add_argument("--importance-target-action-count", type=int, default=3)
    parser.add_argument(
        "--importance-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--checkpoint-root",
        default="/home/hans/.cache/rndysanos/weighted-deterministic-oce",
    )
    parser.add_argument(
        "--output",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/"
            "weighted-deterministic-oce-100000.json"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    stage = report["stage"]
    assert isinstance(stage, dict)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output,
                "convergence": stage["convergence"],
                "training": stage["splits"]["training"]["objective_gradient"],
                "validation": stage["splits"]["validation"]["objective_gradient"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
