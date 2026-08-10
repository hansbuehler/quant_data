"""Run a small compiled-GPU mechanics test for conditional tail quantiles."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from quantdata.torch import BaseEnvironment, DeviceLock
from quantdata.torch.ffw import FeedForwardConfig

from .conditional_tail import (
    ConditionalTailConfig,
    ConditionalTailQuantileModel,
    signed_returns,
)
from .direct_arbitrage import state_normalization


def _memory(device: torch.device) -> dict[str, float]:
    output: dict[str, float] = {}
    with Path("/proc/self/status").open(encoding="utf-8") as stream:
        for line in stream:
            name, _, value = line.partition(":")
            if name in {"VmRSS", "VmHWM"}:
                output[{"VmRSS": "rss_gib", "VmHWM": "peak_rss_gib"}[name]] = (
                    float(value.split()[0]) / 1024.0**2
                )
    if device.type == "cuda":
        output.update(
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024.0**3,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024.0**3,
        )
    return output


def _dynamo_counters() -> dict[str, int]:
    try:
        from torch._dynamo.utils import counters

        return {
            "frames_total": int(counters["frames"]["total"]),
            "frames_ok": int(counters["frames"]["ok"]),
            "unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": int(sum(counters["graph_break"].values())),
        }
    except (ImportError, KeyError, TypeError):
        return {}


def _synthetic_data(
    env: BaseEnvironment,
    *,
    paths: int,
    steps: int,
    state_features: int,
    instruments: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)
    z = torch.randn(
        paths,
        steps,
        state_features,
        dtype=env.fdtype,
        device=env.device,
        generator=generator,
    )
    loadings = torch.linspace(
        -1.0, 1.0, instruments, dtype=env.fdtype, device=env.device
    )
    location = 0.002 * z[..., :1] * loadings
    scale = 0.01 + 0.002 * torch.sigmoid(z[..., 1:2])
    noise = torch.randn(
        paths,
        steps,
        instruments,
        dtype=env.fdtype,
        device=env.device,
        generator=generator,
    )
    return {"z": z, "returns": location + scale * noise}


@torch.no_grad()
def _valuate(
    model: ConditionalTailQuantileModel,
    data: dict[str, torch.Tensor],
) -> dict[str, Any]:
    loss, by_output = model.loss(data["z"], data["returns"])
    quantiles = model(data["z"])
    targets = signed_returns(data["returns"]).unsqueeze(-2)
    coverage = (targets <= quantiles).to(torch.float64).mean(dim=(0, 1, 4))
    cvar = model.cvar_scores(data["z"], data["returns"]).mean(dim=(0, 1, 4))
    return {
        "loss": float(loss.item()),
        "pinball_by_direction_level_instrument_mean": by_output.mean(dim=-1).cpu().tolist(),
        "coverage_by_direction_level": coverage.cpu().tolist(),
        "cvar_score_by_direction_level": cvar.cpu().tolist(),
        "monotone": bool(torch.all(quantiles[..., 1:, :] >= quantiles[..., :-1, :]).item()),
        "sub_one_path_tied_to_one_path": bool(
            torch.equal(quantiles[..., 0, :], quantiles[..., 1, :])
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    with DeviceLock() as device_lock:
        env = BaseEnvironment(device=device_lock.device, debug=False)
        if env.device.type == "cuda":
            torch.cuda.set_device(env.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        data = _synthetic_data(
            env,
            paths=args.paths,
            steps=args.steps,
            state_features=args.state_features,
            instruments=args.instruments,
            seed=args.seed,
        )
        training_paths = args.paths // 2
        training = {name: value[:training_paths] for name, value in data.items()}
        validation = {name: value[training_paths:] for name, value in data.items()}
        state_mean, state_scale = state_normalization(training, batch_size=args.batch_size)
        config = ConditionalTailConfig(
            label="smoke-conditional-tail",
            state_features=args.state_features,
            instruments=args.instruments,
            training_paths=training_paths,
            network_config=FeedForwardConfig(
                num_depth=2,
                num_width=100,
                activation="LeakyReLU",
                use_bias=True,
            ),
        )
        model = ConditionalTailQuantileModel(
            env,
            config=config,
            state_mean=state_mean,
            state_scale=state_scale,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
        gamma = 1.0 if args.epochs == 1 else (args.final_lr / args.lr) ** (1.0 / (args.epochs - 1))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        generator = torch.Generator(device=env.device)
        generator.manual_seed(args.seed + 1)
        initial_training = _valuate(model, training)
        initial_validation = _valuate(model, validation)
        before_compile = _dynamo_counters()
        loss_function = (
            torch.compile(model.training_loss, dynamic=True)
            if args.compile
            else model.training_loss
        )
        batch_losses: list[float] = []
        maximum_gradient_norm = 0.0
        for epoch in range(args.epochs):
            indices = torch.randint(
                training_paths,
                (min(args.batch_size, training_paths),),
                generator=generator,
                device=env.device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_function(training["z"][indices], training["returns"][indices])
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite CPU objective at epoch {epoch + 1}")
            loss.backward()
            gradient_sqr = torch.zeros((), dtype=torch.float64, device=env.device)
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError(f"non-finite gradient at epoch {epoch + 1}")
                gradient_sqr += parameter.grad.to(torch.float64).square().sum()
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_sqr.sqrt().item()))
            optimizer.step()
            if epoch + 1 < args.epochs:
                scheduler.step()
            batch_losses.append(loss_value)
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        after_compile = _dynamo_counters()
        report = {
            "device": str(env.device),
            "paths": args.paths,
            "training_paths": training_paths,
            "validation_paths": args.paths - training_paths,
            "steps": args.steps,
            "state_features": args.state_features,
            "instruments": args.instruments,
            "epochs": args.epochs,
            "compile_requested": args.compile,
            "compiled_graphs": {
                name: value - before_compile.get(name, 0)
                for name, value in after_compile.items()
            },
            "tail_levels": config.level_metadata(),
            "initial_training": initial_training,
            "initial_validation": initial_validation,
            "final_training": _valuate(model, training),
            "final_validation": _valuate(model, validation),
            "first_batch_loss": batch_losses[0],
            "final_batch_loss": batch_losses[-1],
            "maximum_gradient_norm": maximum_gradient_norm,
            "memory": _memory(env.device),
        }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--state-features", type=int, default=21)
    parser.add_argument("--instruments", type=int, default=141)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--final-lr", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=83_771)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in ("paths", "epochs", "steps", "state_features", "instruments", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.paths < 2:
        raise ValueError("--paths must be at least two")
    report = run(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
