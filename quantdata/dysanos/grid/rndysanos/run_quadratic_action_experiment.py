"""Train a regularized conditional-action diagnostic on a cached market."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.torch import BaseEnvironment, DeviceLock

from .quadratic_action import (
    QuadraticActionConfig,
    QuadraticActionModel,
    objective_gradient,
    parameter_norm,
    valuate_quadratic_action,
)
from .run_cached_expiry_oce import _load_cached_universe
from .run_experiment import _counter_difference, _dynamo_counters, memory_snapshot
from .torch_drift_change import DysanosConfig, prepare_universe

_DEFAULT_MARKET_REPORT = (
    "quantdata/dysanos/grid/gendysanos/reports/required/"
    "dysanos-weekly-1000000-expiry-delta-1e-04.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {name: _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(value), indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_weight(weight: torch.Tensor) -> torch.Tensor:
    mean = weight.mean(dtype=torch.float64)
    if not torch.isfinite(mean).item() or mean <= 0.0:
        raise ValueError("sampling weights must have positive finite mean")
    result = weight / mean.to(weight.dtype)
    if not torch.isfinite(result).all().item():
        raise ValueError("normalized sampling weights are non-finite")
    return result


def prepare_quadratic_data(
    universe: dict[str, np.ndarray],
    *,
    training_paths: int,
    cash_returns: bool,
    trade_spot: bool,
    standardize_returns: bool,
    env: BaseEnvironment,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Prepare and normalize training/validation tensors without leakage."""

    preparation = DysanosConfig(
        label="quadratic-action-preparation",
        utility="entropy",
        state_mode="raw",
        delta_hedged=True,
        cash_returns=cash_returns,
        trade_spot=trade_spot,
        learn_cost=False,
        cost_cond=0.0,
        cost_uncond=0.0,
        cost_prop=0.0,
    )
    data, geometry = prepare_universe(universe, config=preparation, env=env)
    paths = data["z"].shape[0]
    if not 0 < training_paths < paths:
        raise ValueError(
            f"training paths must lie in (0,{paths}), got {training_paths}"
        )
    training = {
        name: value[:training_paths].contiguous() for name, value in data.items()
    }
    validation = {
        name: value[training_paths:].contiguous() for name, value in data.items()
    }
    for split in (training, validation):
        split["sampling_weight"] = _normalise_weight(split["sampling_weight"])
        if not trade_spot:
            split["dH"] = split["dH"][..., 1:].contiguous()

    # State and return scales are fitted once on training data.  Time is
    # explicit because the requested action is a(t,x_t), not merely a(x_t).
    trn_weight = training["sampling_weight"][:, None, None]
    weight_mass = trn_weight.sum(dtype=torch.float64) * training["z"].shape[1]
    state_mean = (trn_weight * training["z"]).sum(
        dim=(0, 1), dtype=torch.float64
    ) / weight_mass
    state_second = (trn_weight * training["z"].square()).sum(
        dim=(0, 1), dtype=torch.float64
    ) / weight_mass
    state_std = (state_second - state_mean.square()).clamp_min(0.0).sqrt()
    state_std = state_std.clamp_min(1.0e-6)

    if standardize_returns:
        # Positive instrument/time rescaling preserves the existence of an
        # arbitrage while putting the quadratic action charge on a common scale.
        return_second = (trn_weight * training["dH"].square()).sum(
            dim=0, dtype=torch.float64
        ) / trn_weight.sum(dtype=torch.float64)
        return_scale = return_second.sqrt().clamp_min(1.0e-8)
    else:
        return_scale = torch.ones_like(training["dH"][0], dtype=torch.float64)

    TAU = training["z"].shape[1]
    time_coordinate = torch.linspace(
        -1.0,
        1.0,
        TAU,
        dtype=env.fdtype,
        device=env.device,
    ).reshape(1, TAU, 1)
    for split in (training, validation):
        normalized_state = (
            split["z"] - state_mean.to(env.fdtype)
        ) / state_std.to(env.fdtype)
        split["z"] = torch.cat(
            (
                normalized_state,
                time_coordinate.expand(normalized_state.shape[0], -1, -1),
            ),
            dim=-1,
        ).contiguous()
        split["dH"] = (
            split["dH"] / return_scale.to(env.fdtype)
        ).contiguous()
        split.pop("H", None)

    diagnostics = {
        "paths": paths,
        "training_paths": training_paths,
        "validation_paths": paths - training_paths,
        "steps": TAU,
        "instruments": training["dH"].shape[-1],
        "state_features": training["z"].shape[-1],
        "cash_returns": cash_returns,
        "trade_spot": trade_spot,
        "standardize_returns": standardize_returns,
        "state_mean": state_mean.cpu().numpy(),
        "state_std": state_std.cpu().numpy(),
        "return_scale": return_scale.cpu().numpy(),
        "geometry": {
            "samples": geometry.samples,
            "steps": geometry.steps,
            "expiries": geometry.expiries,
            "strikes": geometry.strikes,
            "state_variables": geometry.state_variables,
        },
    }
    return training, validation, diagnostics


def _checkpoint_path(args: argparse.Namespace) -> Path:
    utility_label = (
        f"downside-p{args.downside_power}"
        if args.utility == "downside"
        else args.utility
    )
    return Path(args.checkpoint_root).expanduser() / (
        f"{utility_label}-{'bounded' if args.bounded_action else 'direct'}-"
        f"kappa-{args.quadratic_coefficient:g}-paths-{args.paths}-seed-{args.seed}.pt"
    )


def _save_checkpoint(
    path: Path,
    *,
    model: QuadraticActionModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "schema": 1,
                "model_version": QuadraticActionModel.version.unique_id48,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "history": history,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_quadratic_action(
    model: QuadraticActionModel,
    training: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train with constant-rate AdamW and periodic exact split valuations."""

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0
    )
    checkpoint = _checkpoint_path(args)
    history: list[dict[str, Any]] = []
    initial_step = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=model.device, weights_only=False)
        if (
            state.get("schema") != 1
            or state.get("model_version")
            != QuadraticActionModel.version.unique_id48
        ):
            raise ValueError("quadratic-action checkpoint is incompatible")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        initial_step = int(state["step"])
        history = list(state["history"])

    compiled = torch.compile(model) if args.compile else model
    generator = torch.Generator(device=model.device)
    generator.manual_seed(args.seed)
    paths = training["z"].shape[0]
    started = time.monotonic()
    stop_reason = "maximum_steps_reached"
    converged = False

    for step in range(initial_step + 1, args.steps + 1):
        indices = torch.randint(
            paths,
            (min(args.batch_size, paths),),
            generator=generator,
            device=model.device,
        )
        batch = {name: value[indices] for name, value in training.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(batch)
        loss_cpu = float(loss.detach().cpu().item())
        if not math.isfinite(loss_cpu):
            raise FloatingPointError(
                f"quadratic-action objective became non-finite at step {step}"
            )
        loss.backward()
        nonfinite = sum(
            int(torch.count_nonzero(~torch.isfinite(parameter.grad)).item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if nonfinite:
            raise FloatingPointError(
                f"found {nonfinite} non-finite gradient elements at step {step}"
            )
        optimizer.step()

        should_value = step == 1 or step % args.valuation_interval == 0
        if step == args.steps:
            should_value = True
        if not should_value:
            continue
        training_gradient = objective_gradient(
            model,
            training,
            batch_size=args.valuation_batch_size,
        )
        validation_gradient = objective_gradient(
            model,
            validation,
            batch_size=args.valuation_batch_size,
        )
        training_value = valuate_quadratic_action(
            model,
            training,
            batch_size=args.valuation_batch_size,
        )
        validation_value = valuate_quadratic_action(
            model,
            validation,
            batch_size=args.valuation_batch_size,
        )
        point = {
            "step": step,
            "elapsed_seconds": time.monotonic() - started,
            "minibatch_loss": loss_cpu,
            "parameter_norm": parameter_norm(list(model.parameters())),
            "training": {
                "loss": training_gradient["loss"],
                "gradient_norm": training_gradient["gradient_norm"],
                "first_order_max_abs_z": training_value[
                    "first_order_residual_max_abs_z"
                ],
                "saturation_rate_max": training_value.get(
                    "action_cap_saturation_rate_max", 0.0
                ),
            },
            "validation": {
                "loss": validation_gradient["loss"],
                "gradient_norm": validation_gradient["gradient_norm"],
                "first_order_max_abs_z": validation_value[
                    "first_order_residual_max_abs_z"
                ],
                "saturation_rate_max": validation_value.get(
                    "action_cap_saturation_rate_max", 0.0
                ),
            },
        }
        history.append(point)
        print(json.dumps(point), flush=True)
        _save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=step,
            history=history,
        )

        if step >= args.minimum_steps and len(history) >= args.convergence_points:
            recent = np.asarray(
                [item["training"]["loss"] for item in history[-args.convergence_points :]],
                dtype=np.float64,
            )
            scale = max(1.0, abs(float(np.mean(recent))))
            relative_range = float(np.ptp(recent) / scale)
            gradient_norm = float(training_gradient["gradient_norm"])
            if (
                relative_range <= args.convergence_tolerance
                and gradient_norm <= args.gradient_tolerance
            ):
                converged = True
                stop_reason = "objective_and_gradient_converged"
                break

    return {
        "converged": converged,
        "stop_reason": stop_reason,
        "completed_steps": step,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "training": valuate_quadratic_action(
            model, training, batch_size=args.valuation_batch_size
        ),
        "validation": valuate_quadratic_action(
            model, validation, batch_size=args.valuation_batch_size
        ),
        "training_objective_gradient": objective_gradient(
            model, training, batch_size=args.valuation_batch_size
        ),
        "validation_objective_gradient": objective_gradient(
            model, validation, batch_size=args.valuation_batch_size
        ),
        "checkpoint": str(checkpoint),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load cached paths, train one variant, and write its report."""

    source_universe, source = _load_cached_universe(
        args.market_report,
        paths=args.paths,
        cost_prop=args.market_cost_prop,
    )
    lock = DeviceLock(fall_back=True) if args.device == "lock" else None
    context = lock if lock is not None else nullcontext()
    with context:
        device = (
            lock.device
            if lock is not None
            else torch.device(
                "cuda:0"
                if args.device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )
        )
        env = BaseEnvironment(
            precision=args.precision,
            device=device,
            debug=args.debug,
        )
        if env.device.type == "cuda":
            torch.empty((), dtype=env.fdtype, device=env.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(env.device)
        training, validation, preparation = prepare_quadratic_data(
            source_universe,
            training_paths=source["training_paths"],
            cash_returns=args.cash_returns,
            trade_spot=args.trade_spot,
            standardize_returns=args.standardize_returns,
            env=env,
        )
        del source_universe
        config = QuadraticActionConfig(
            utility=args.utility,
            lmbda=args.lmbda,
            downside_power=args.downside_power,
            quadratic_coefficient=args.quadratic_coefficient,
            bounded_action=args.bounded_action,
            initial_action_scale=args.initial_action_scale,
        )
        model = QuadraticActionModel(
            env,
            num_input=training["z"].shape[-1],
            num_instruments=training["dH"].shape[-1],
            config=config,
        )
        before_compile = _dynamo_counters()
        result = train_quadratic_action(
            model,
            training,
            validation,
            args=args,
        )
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        report = {
            "status": "complete",
            "experiment": "quadratic regularized conditional-action diagnostic",
            "objective": (
                "sup_{a,y} E[u(y + a(t,x_t)' dH_t - "
                "kappa ||a(t,x_t)||_2^2) - y]"
            ),
            "density_warning": (
                "The marginal-utility density satisfies the penalized first-order "
                "identity E_D[dH|x]=2*kappa*a(x), not the martingale condition."
            ),
            "configuration": {
                "utility": args.utility,
                "lambda": args.lmbda,
                "downside_power": (
                    args.downside_power if args.utility == "downside" else None
                ),
                "quadratic_coefficient": args.quadratic_coefficient,
                "architecture": (
                    "scaled-tanh" if args.bounded_action else "direct-unbounded"
                ),
                "initial_action_scale": args.initial_action_scale,
                "paths": args.paths,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "learning_rate_schedule": "constant",
                "gradient_clipping": False,
                "seed": args.seed,
                "precision": args.precision,
                "compile": args.compile,
            },
            "source": source,
            "preparation": preparation,
            "device": str(env.device),
            "compilation": _counter_difference(_dynamo_counters(), before_compile),
            "memory": memory_snapshot(env.device),
            "result": result,
        }
    _atomic_json(Path(args.report), report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--market-report", default=_DEFAULT_MARKET_REPORT)
    result.add_argument("--market-cost-prop", type=float, default=1.0e-4)
    result.add_argument("--paths", type=int, default=1_000)
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--minimum-steps", type=int, default=100)
    result.add_argument("--valuation-interval", type=int, default=100)
    result.add_argument("--convergence-points", type=int, default=5)
    result.add_argument("--convergence-tolerance", type=float, default=1.0e-4)
    result.add_argument("--gradient-tolerance", type=float, default=1.0e-4)
    result.add_argument("--batch-size", type=int, default=1_024)
    result.add_argument("--valuation-batch-size", type=int, default=8_192)
    result.add_argument("--lr", type=float, default=1.0e-3)
    result.add_argument("--seed", type=int, default=830_000)
    result.add_argument(
        "--utility",
        choices=("entropy", "powexp", "downside"),
        default="entropy",
    )
    result.add_argument("--lmbda", type=float, default=1.0)
    result.add_argument("--downside-power", type=int, choices=(1, 2, 3), default=2)
    result.add_argument("--quadratic-coefficient", type=float, default=0.5)
    result.add_argument(
        "--bounded-action", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument("--initial-action-scale", type=float, default=1.0)
    result.add_argument(
        "--cash-returns", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument(
        "--trade-spot", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument(
        "--standardize-returns", action=argparse.BooleanOptionalAction, default=True
    )
    result.add_argument("--precision", type=int, choices=(32, 64), default=32)
    result.add_argument(
        "--device", choices=("lock", "cuda", "cpu"), default="lock"
    )
    result.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--debug", action="store_true")
    result.add_argument(
        "--checkpoint-root",
        default="~/.cache/rndysanos/quadratic-action",
    )
    result.add_argument(
        "--report",
        default=(
            "quantdata/dysanos/grid/rndysanos/reports/quadratic-action-smoke.json"
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    compact = {
        "report": args.report,
        "device": report["device"],
        "configuration": report["configuration"],
        "converged": report["result"]["converged"],
        "stop_reason": report["result"]["stop_reason"],
        "training": {
            "gradient": report["result"]["training_objective_gradient"],
            "first_order_max_abs_z": report["result"]["training"][
                "first_order_residual_max_abs_z"
            ],
            "saturation_rate_max": report["result"]["training"].get(
                "action_cap_saturation_rate_max", 0.0
            ),
        },
        "validation": {
            "gradient": report["result"]["validation_objective_gradient"],
            "first_order_max_abs_z": report["result"]["validation"][
                "first_order_residual_max_abs_z"
            ],
            "saturation_rate_max": report["result"]["validation"].get(
                "action_cap_saturation_rate_max", 0.0
            ),
        },
        "memory": report["memory"],
        "compilation": report["compilation"],
    }
    print(json.dumps(_jsonable(compact), indent=2), flush=True)


if __name__ == "__main__":
    main()
