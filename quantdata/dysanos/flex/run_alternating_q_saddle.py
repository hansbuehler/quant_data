"""Alternate a conditional-action best response with an empirical Q density.

This is a mechanics experiment for the saddle problem

    sup_D inf_a log E_P[D exp(-a(X)'R + 0.5 a(X)'a(X))].

The density is represented empirically relative to a supplied positive base
measure.  Mirror ascent preserves strict positivity, while exponential
tempering projects every update onto an explicit P-relative ESS floor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from scipy.special import logsumexp

from quantdata.dysanos.flex.run_consistent_tact_audit import (
    atomic_json,
    atomic_torch,
    check_resources,
    fit_stage,
    make_policy,
    path_scores,
    transition_gain,
    unconditional_coordinate_diagnostics,
    weighted_antithetic_test,
    weighted_normalizers,
)


@dataclass(frozen=True, kw_only=True)
class DensityProjection:
    relative_density: np.ndarray
    combined_weights: np.ndarray
    temperature: float
    effective_sample_size: float


@dataclass(frozen=True, kw_only=True)
class DensityStep:
    projection: DensityProjection
    objective_before: float
    objective_after: float
    relative_entropy: float
    weighted_l1_change: float


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("weights must be a nonempty vector")
    if np.any(~np.isfinite(values) | (values <= 0.0)):
        raise ValueError("weights must be finite-positive")
    return float(np.sum(values, dtype=np.float64) ** 2 / np.sum(values**2))


def normalized_base_weights(base_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(base_weights, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("base weights must be a nonempty vector")
    if np.any(~np.isfinite(values) | (values <= 0.0)):
        raise ValueError("base weights must be finite-positive")
    return values * (len(values) / np.sum(values, dtype=np.float64))


def game_objective(log_path_loss: np.ndarray, combined_weights: np.ndarray) -> float:
    losses = np.asarray(log_path_loss, dtype=np.float64)
    weights = np.asarray(combined_weights, dtype=np.float64)
    if losses.shape != weights.shape:
        raise ValueError("path losses and weights must have matching shapes")
    if np.any(~np.isfinite(losses)):
        raise ValueError("path losses must be finite")
    if np.any(~np.isfinite(weights) | (weights <= 0.0)):
        raise ValueError("combined weights must be finite-positive")
    return float(logsumexp(np.log(weights) + losses) - math.log(np.sum(weights)))


def _density_at_temperature(
    *,
    candidate_log_relative_density: np.ndarray,
    base_weights: np.ndarray,
    temperature: float,
    minimum_relative_density: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    base = normalized_base_weights(base_weights)
    base_probability = base / np.sum(base, dtype=np.float64)
    log_relative = temperature * np.asarray(
        candidate_log_relative_density, dtype=np.float64
    )
    log_normalizer = float(logsumexp(np.log(base_probability) + log_relative))
    relative = np.exp(log_relative - log_normalizer)
    relative = minimum_relative_density + (1.0 - minimum_relative_density) * relative
    combined = base * relative
    combined *= len(combined) / np.sum(combined, dtype=np.float64)
    relative = combined / base
    return relative, combined, effective_sample_size(combined)


def project_relative_density(
    *,
    candidate_log_relative_density: np.ndarray,
    base_weights: np.ndarray,
    minimum_ess_fraction: float,
    minimum_relative_density: float,
    bisection_iterations: int = 60,
) -> DensityProjection:
    """Normalize a positive relative density and temper it to the ESS floor."""

    candidate = np.asarray(candidate_log_relative_density, dtype=np.float64)
    base = normalized_base_weights(base_weights)
    if candidate.shape != base.shape or np.any(~np.isfinite(candidate)):
        raise ValueError("candidate log density must be finite and match base weights")
    if not 0.0 < minimum_ess_fraction <= 1.0:
        raise ValueError("minimum ESS fraction must lie in (0, 1]")
    if not 0.0 < minimum_relative_density < 1.0:
        raise ValueError("minimum relative density must lie in (0, 1)")
    target_ess = minimum_ess_fraction * len(base)
    base_ess = effective_sample_size(base)
    if base_ess + 1.0e-9 < target_ess:
        raise ValueError(
            f"base ESS {base_ess:.6g} is below requested floor {target_ess:.6g}"
        )

    relative, combined, ess = _density_at_temperature(
        candidate_log_relative_density=candidate,
        base_weights=base,
        temperature=1.0,
        minimum_relative_density=minimum_relative_density,
    )
    temperature = 1.0
    if ess + 1.0e-9 < target_ess:
        feasible = 0.0
        infeasible = 1.0
        for _ in range(bisection_iterations):
            middle = 0.5 * (feasible + infeasible)
            trial = _density_at_temperature(
                candidate_log_relative_density=candidate,
                base_weights=base,
                temperature=middle,
                minimum_relative_density=minimum_relative_density,
            )
            if trial[2] + 1.0e-9 >= target_ess:
                feasible = middle
                relative, combined, ess = trial
            else:
                infeasible = middle
        temperature = feasible

    if np.any(~np.isfinite(relative) | (relative <= 0.0)):
        raise FloatingPointError("projected relative density is not finite-positive")
    if float(np.min(relative)) + 1.0e-12 < minimum_relative_density:
        raise FloatingPointError("projected density violated its positive floor")
    if not np.isclose(np.mean(combined), 1.0, atol=1.0e-12, rtol=0.0):
        raise FloatingPointError("projected combined density does not have mean one")
    if ess + 1.0e-7 < target_ess:
        raise FloatingPointError("projected density violated its ESS floor")
    return DensityProjection(
        relative_density=relative,
        combined_weights=combined,
        temperature=temperature,
        effective_sample_size=ess,
    )


def mirror_ascent_density_step(
    *,
    relative_density: np.ndarray,
    base_weights: np.ndarray,
    log_path_loss: np.ndarray,
    learning_rate: float,
    entropy_strength: float,
    reward_clip: float,
    minimum_ess_fraction: float,
    minimum_relative_density: float,
) -> DensityStep:
    """Take one entropy-regularized mirror-ascent step in the empirical density."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("density learning rate must be finite-positive")
    if not math.isfinite(entropy_strength) or entropy_strength < 0.0:
        raise ValueError("entropy strength must be finite-nonnegative")
    if learning_rate * entropy_strength >= 1.0:
        raise ValueError("learning_rate * entropy_strength must be below one")
    if not math.isfinite(reward_clip) or reward_clip <= 1.0:
        raise ValueError("reward clip must exceed one")

    base = normalized_base_weights(base_weights)
    relative = np.asarray(relative_density, dtype=np.float64)
    losses = np.asarray(log_path_loss, dtype=np.float64)
    if relative.shape != base.shape or losses.shape != base.shape:
        raise ValueError("density, base weights, and losses must have matching shapes")
    if np.any(~np.isfinite(relative) | (relative <= 0.0)):
        raise ValueError("relative density must be finite-positive")
    combined = base * relative
    combined *= len(combined) / np.sum(combined, dtype=np.float64)
    objective_before = game_objective(losses, combined)
    reward = np.exp(
        np.clip(losses - objective_before, -math.log(reward_clip), math.log(reward_clip))
    )
    candidate_log_relative = (
        (1.0 - learning_rate * entropy_strength) * np.log(relative)
        + learning_rate * (reward - 1.0)
    )
    projection = project_relative_density(
        candidate_log_relative_density=candidate_log_relative,
        base_weights=base,
        minimum_ess_fraction=minimum_ess_fraction,
        minimum_relative_density=minimum_relative_density,
    )
    objective_after = game_objective(losses, projection.combined_weights)
    base_probability = base / np.sum(base, dtype=np.float64)
    relative_entropy = float(
        np.sum(
            base_probability
            * projection.relative_density
            * np.log(projection.relative_density),
            dtype=np.float64,
        )
    )
    weighted_l1_change = float(
        np.sum(
            base_probability
            * np.abs(projection.relative_density - relative),
            dtype=np.float64,
        )
    )
    return DensityStep(
        projection=projection,
        objective_before=objective_before,
        objective_after=objective_after,
        relative_entropy=relative_entropy,
        weighted_l1_change=weighted_l1_change,
    )


def _save_npy_atomic(values: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def _path_log_loss(gain: np.ndarray) -> np.ndarray:
    if gain.ndim != 2 or np.any(~np.isfinite(gain)):
        raise ValueError("gain must be a finite path-by-time matrix")
    return logsumexp(-np.asarray(gain, dtype=np.float64), axis=1) - math.log(
        gain.shape[1]
    )


def run(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / "run.lock", timeout=0):
        print(json.dumps({"stage": "resources", **check_resources(arguments.output)}), flush=True)
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.set_num_threads(arguments.cpu_threads)
        torch.set_float32_matmul_precision("high")
        if arguments.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        states = np.load(arguments.cache / "train-states.npy", mmap_mode="r", allow_pickle=False)
        prices = np.load(arguments.cache / "train-prices.npy", mmap_mode="r", allow_pickle=False)
        returns = np.load(arguments.cache / "train-returns.npy", mmap_mode="r", allow_pickle=False)
        base_weights = normalized_base_weights(
            np.load(arguments.cache / "train-q1-weights.npy", mmap_mode="r", allow_pickle=False)
        )
        paths = len(base_weights)
        if states.shape != (paths, 25, 200):
            raise ValueError(f"unexpected state shape {states.shape}")
        if prices.shape != (paths, 25, 4, 12):
            raise ValueError(f"unexpected price shape {prices.shape}")
        if returns.shape != (paths, 25, 49):
            raise ValueError(f"unexpected return shape {returns.shape}")
        if paths % arguments.antithetic_block_size:
            raise ValueError("paths must contain complete antithetic generation blocks")

        normalizer_path = arguments.output / "normalizers.npz"
        if normalizer_path.exists():
            with np.load(normalizer_path, allow_pickle=False) as payload:
                normalizers = {name: payload[name] for name in payload.files}
        else:
            normalizers = weighted_normalizers(
                states, prices, base_weights, batch_size=arguments.batch_size
            )
            temporary = normalizer_path.with_name(f".{normalizer_path.name}.tmp.npz")
            np.savez(temporary, **normalizers)
            os.replace(temporary, normalizer_path)

        policy = make_policy(
            device=torch.device(arguments.device),
            normalizers=normalizers,
            seed=arguments.seed,
            label="alternating-q-saddle",
        )
        state_path = arguments.output / "saddle-state.pt"
        history: list[dict[str, Any]] = []
        relative_density = np.ones(paths, dtype=np.float64)
        start_outer = 0
        if state_path.exists():
            payload = torch.load(state_path, map_location=policy.env.device, weights_only=False)
            if payload.get("schema") != "dysanos-alternating-q-saddle-v1":
                raise RuntimeError(f"unexpected checkpoint schema in {state_path}")
            policy.load_state_dict(payload["policy_state_dict"])
            relative_density = np.asarray(payload["relative_density"], dtype=np.float64)
            history = list(payload["history"])
            start_outer = int(payload["next_outer_iteration"])
            print(json.dumps({"stage": "resume", "outer_iteration": start_outer}), flush=True)
        else:
            policy, _ = fit_stage(
                policy=policy,
                states=states,
                prices=prices,
                returns=returns,
                weights=base_weights,
                checkpoint=arguments.output / "initial-affine.pt",
                stage="affine",
                epochs=arguments.initial_affine_epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.action_learning_rate,
                maximum_gradient_norm=arguments.maximum_gradient_norm,
                maximum_line_search_steps=arguments.maximum_line_search_steps,
                backtracking_factor=arguments.backtracking_factor,
            )
            policy, _ = fit_stage(
                policy=policy,
                states=states,
                prices=prices,
                returns=returns,
                weights=base_weights,
                checkpoint=arguments.output / "initial-full.pt",
                stage="full",
                epochs=arguments.initial_full_epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.action_learning_rate,
                maximum_gradient_norm=arguments.maximum_gradient_norm,
                maximum_line_search_steps=arguments.maximum_line_search_steps,
                backtracking_factor=arguments.backtracking_factor,
            )

        started = time.perf_counter()
        for outer in range(start_outer, arguments.outer_iterations):
            combined_weights = normalized_base_weights(base_weights * relative_density)
            policy, action_history = fit_stage(
                policy=policy,
                states=states,
                prices=prices,
                returns=returns,
                weights=combined_weights,
                checkpoint=arguments.output / f"outer-{outer + 1:03d}-action.pt",
                stage="full",
                epochs=arguments.action_epochs_per_iteration,
                batch_size=arguments.batch_size,
                learning_rate=arguments.action_learning_rate,
                maximum_gradient_norm=arguments.maximum_gradient_norm,
                maximum_line_search_steps=arguments.maximum_line_search_steps,
                backtracking_factor=arguments.backtracking_factor,
            )
            gain = transition_gain(
                policy, states, prices, returns, batch_size=arguments.batch_size
            )
            log_path_loss = _path_log_loss(gain)
            best_response_value = game_objective(log_path_loss, combined_weights)
            density_records: list[dict[str, float]] = []
            for density_step_index in range(arguments.density_steps_per_iteration):
                if best_response_value >= -arguments.fixed_action_target_tolerance:
                    break
                step = mirror_ascent_density_step(
                    relative_density=relative_density,
                    base_weights=base_weights,
                    log_path_loss=log_path_loss,
                    learning_rate=arguments.density_learning_rate,
                    entropy_strength=arguments.density_entropy_strength,
                    reward_clip=arguments.density_reward_clip,
                    minimum_ess_fraction=arguments.minimum_ess_fraction,
                    minimum_relative_density=arguments.minimum_relative_density,
                )
                relative_density = step.projection.relative_density
                best_response_value = step.objective_after
                density_records.append(
                    {
                        "step": float(density_step_index + 1),
                        "objective_before": step.objective_before,
                        "objective_after": step.objective_after,
                        "projection_temperature": step.projection.temperature,
                        "effective_sample_size": step.projection.effective_sample_size,
                        "relative_entropy": step.relative_entropy,
                        "weighted_l1_change": step.weighted_l1_change,
                    }
                )
            combined_weights = normalized_base_weights(base_weights * relative_density)
            record: dict[str, Any] = {
                "outer_iteration": outer + 1,
                "action_best_response_value_before_density": (
                    density_records[0]["objective_before"]
                    if density_records
                    else best_response_value
                ),
                "fixed_action_value_after_density": best_response_value,
                "combined_density_effective_sample_size": effective_sample_size(
                    combined_weights
                ),
                "combined_density_minimum": float(np.min(combined_weights)),
                "combined_density_maximum": float(np.max(combined_weights)),
                "relative_density_minimum": float(np.min(relative_density)),
                "relative_density_maximum": float(np.max(relative_density)),
                "density_steps": density_records,
                "action_final": action_history[-1],
                "elapsed_seconds": time.perf_counter() - started,
                "resources": check_resources(arguments.output),
            }
            history.append(record)
            _save_npy_atomic(combined_weights, arguments.output / "weights.npy")
            _save_npy_atomic(relative_density, arguments.output / "relative-density.npy")
            atomic_torch(
                {
                    "schema": "dysanos-alternating-q-saddle-v1",
                    "policy_state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in policy.state_dict().items()
                    },
                    "relative_density": relative_density,
                    "history": history,
                    "next_outer_iteration": outer + 1,
                },
                state_path,
            )
            atomic_json(
                {
                    "schema": "dysanos-alternating-q-saddle-report-v1",
                    "complete": False,
                    "arguments": {
                        name: str(value) if isinstance(value, Path) else value
                        for name, value in vars(arguments).items()
                    },
                    "base_density_effective_sample_size": effective_sample_size(
                        base_weights
                    ),
                    "history": history,
                },
                arguments.output / "report.partial.json",
            )
            print(json.dumps({"stage": "outer-complete", **record}), flush=True)

        combined_weights = normalized_base_weights(base_weights * relative_density)
        raw_score, quadratic_score = path_scores(
            policy, states, prices, returns, batch_size=arguments.batch_size
        )
        report = {
            "schema": "dysanos-alternating-q-saddle-report-v1",
            "complete": True,
            "definition": (
                "alternate the full affine-plus-deep conditional-action best response "
                "with a positive empirical mirror-ascent density relative to Q1"
            ),
            "arguments": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(arguments).items()
            },
            "base_density_effective_sample_size": effective_sample_size(base_weights),
            "final_density_effective_sample_size": effective_sample_size(combined_weights),
            "history": history,
            "in_sample_action_score": {
                "raw": weighted_antithetic_test(
                    raw_score,
                    combined_weights,
                    antithetic_block_size=arguments.antithetic_block_size,
                ),
                "quadratic": weighted_antithetic_test(
                    quadratic_score,
                    combined_weights,
                    antithetic_block_size=arguments.antithetic_block_size,
                ),
            },
            "unconditional_coordinate_diagnostics": unconditional_coordinate_diagnostics(
                returns,
                combined_weights,
                batch_size=arguments.batch_size,
                antithetic_block_size=arguments.antithetic_block_size,
            ),
            "resources_final": check_resources(arguments.output),
        }
        atomic_json(report, arguments.output / "report.json")
        print(json.dumps({"stage": "complete", "report": str(arguments.output / "report.json")}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=10_000)
    result.add_argument("--antithetic-block-size", type=int, default=200)
    result.add_argument("--initial-affine-epochs", type=int, default=50)
    result.add_argument("--initial-full-epochs", type=int, default=100)
    result.add_argument("--outer-iterations", type=int, default=10)
    result.add_argument("--action-epochs-per-iteration", type=int, default=50)
    result.add_argument("--action-learning-rate", type=float, default=1.0e-3)
    result.add_argument("--maximum-gradient-norm", type=float, default=10.0)
    result.add_argument("--maximum-line-search-steps", type=int, default=20)
    result.add_argument("--backtracking-factor", type=float, default=0.2)
    result.add_argument("--density-steps-per-iteration", type=int, default=20)
    result.add_argument("--density-learning-rate", type=float, default=0.05)
    result.add_argument("--density-entropy-strength", type=float, default=0.01)
    result.add_argument("--density-reward-clip", type=float, default=10.0)
    result.add_argument("--minimum-ess-fraction", type=float, default=0.5)
    result.add_argument("--minimum-relative-density", type=float, default=0.02)
    result.add_argument("--fixed-action-target-tolerance", type=float, default=1.0e-4)
    result.add_argument("--seed", type=int, default=20261009)
    result.add_argument("--cpu-threads", type=int, default=4)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
