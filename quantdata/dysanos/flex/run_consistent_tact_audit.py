"""Run the paper's two-fold conditional and unconditional DYSANOS audits."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from scipy.special import logsumexp

from quantdata.dysanos.flex.run_convex_q_conditional_oce import (
    SharedExpiryOceConfig,
    SharedExpiryOcePolicy,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def resource_snapshot(output: Path) -> dict[str, float]:
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value, *_ = line.split()
        if name in {"MemTotal:", "MemAvailable:"}:
            memory[name[:-1]] = int(value) * 1024
    data_fs = os.statvfs(output)
    tmp_fs = os.statvfs("/tmp")
    return {
        "memory_used_fraction": 1.0 - memory["MemAvailable"] / memory["MemTotal"],
        "memory_available_gib": memory["MemAvailable"] / 2**30,
        "data_disk_free_fraction": data_fs.f_bavail / data_fs.f_blocks,
        "data_disk_free_gib": data_fs.f_bavail * data_fs.f_frsize / 2**30,
        "temporary_disk_used_fraction": 1.0 - tmp_fs.f_bavail / tmp_fs.f_blocks,
    }


def check_resources(output: Path) -> dict[str, float]:
    result = resource_snapshot(output)
    if result["memory_used_fraction"] >= 0.9:
        raise RuntimeError("RAM usage reached the 90% project stop threshold")
    if result["data_disk_free_fraction"] <= 0.1:
        raise RuntimeError("data filesystem reached the 10% free-space stop threshold")
    if result["temporary_disk_used_fraction"] >= 0.9:
        raise RuntimeError("temporary filesystem reached the 90% usage stop threshold")
    return result


def weighted_normalizers(
    states: np.ndarray,
    prices: np.ndarray,
    weights: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if states.shape[:2] != prices.shape[:2] or len(states) != len(weights):
        raise ValueError("normalizer arrays have inconsistent path/time axes")
    if prices.shape[2:] != (4, 12):
        raise ValueError(f"unexpected price shape {prices.shape}")
    weight_sum = float(np.sum(weights, dtype=np.float64))
    state_sum = np.zeros(states.shape[-1], dtype=np.float64)
    state_square_sum = np.zeros(states.shape[-1], dtype=np.float64)
    log_price_sum = np.zeros(prices.shape[-1], dtype=np.float64)
    log_price_square_sum = np.zeros(prices.shape[-1], dtype=np.float64)
    tiny = np.finfo(np.float32).tiny
    for first in range(0, len(states), batch_size):
        last = min(first + batch_size, len(states))
        weight = np.asarray(weights[first:last], dtype=np.float64)
        state = np.asarray(states[first:last], dtype=np.float64)
        log_price = np.log(np.maximum(np.asarray(prices[first:last], dtype=np.float64), tiny))
        state_sum += np.einsum("p,ptj->j", weight, state, optimize=True)
        state_square_sum += np.einsum(
            "p,ptj->j", weight, np.square(state), optimize=True
        )
        log_price_sum += np.einsum("p,ptek->k", weight, log_price, optimize=True)
        log_price_square_sum += np.einsum(
            "p,ptek->k", weight, np.square(log_price), optimize=True
        )
    state_denominator = states.shape[1] * weight_sum
    price_denominator = prices.shape[1] * prices.shape[2] * weight_sum
    state_mean = state_sum / state_denominator
    state_variance = np.maximum(state_square_sum / state_denominator - state_mean**2, 0.0)
    price_mean = log_price_sum / price_denominator
    price_variance = np.maximum(
        log_price_square_sum / price_denominator - price_mean**2,
        0.0,
    )
    state_scale = np.sqrt(state_variance)
    price_scale = np.sqrt(price_variance)
    state_scale = np.where(state_scale > 1.0e-6, state_scale, 1.0)
    price_scale = np.where(price_scale > 1.0e-6, price_scale, 1.0)
    return {
        "state_mean": state_mean.astype(np.float32),
        "state_scale": state_scale.astype(np.float32),
        "log_price_mean": price_mean.astype(np.float32),
        "log_price_scale": price_scale.astype(np.float32),
    }


def make_policy(
    *,
    device: torch.device,
    normalizers: dict[str, np.ndarray],
    seed: int,
    label: str,
) -> SharedExpiryOcePolicy:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    env = BaseEnvironment(precision=32, device=device, debug=False)
    config = SharedExpiryOceConfig(
        label=label,
        state_size=200,
        num_expiries=4,
        num_strikes=12,
        # The statistic is explicitly uncapped; raw expiry heads are used below.
        maximum_action_l1=1.0,
        expiry_specific_linear_skip=True,
        include_calendar_time=True,
        action_parameterization="signed",
        network=FeedForwardConfig(
            activation="SiLU",
            num_depth=2,
            num_width=128,
        ),
    )
    return SharedExpiryOcePolicy(
        env,
        config=config,
        state_mean=torch.as_tensor(normalizers["state_mean"], device=device),
        state_scale=torch.as_tensor(normalizers["state_scale"], device=device),
        log_price_mean=torch.as_tensor(normalizers["log_price_mean"], device=device),
        log_price_scale=torch.as_tensor(normalizers["log_price_scale"], device=device),
    )


def unbounded_action(
    policy: SharedExpiryOcePolicy,
    states: torch.Tensor,
    prices: torch.Tensor,
) -> torch.Tensor:
    if states.ndim != 3 or prices.ndim != 4:
        raise ValueError("action inputs have invalid ranks")
    paths, steps, state_size = states.shape
    if prices.shape != (paths, steps, 4, 12):
        raise ValueError(f"unexpected price batch shape {tuple(prices.shape)}")
    calendar = torch.arange(
        steps, dtype=states.dtype, device=states.device
    ) / max(steps - 1, 1)
    calendar = calendar[None].expand(paths, -1).reshape(-1)
    _, raw_heads = policy(
        states.reshape(paths * steps, state_size),
        prices.reshape(paths * steps, 4, 12),
        calendar_time=calendar,
    )
    spot = raw_heads[..., :1].sum(dim=1)
    options = raw_heads[..., 1:].reshape(paths * steps, -1)
    return torch.cat((spot, options), dim=1).reshape(paths, steps, 49)


def load_batch(
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    *,
    first: int,
    last: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.array(states[first:last], copy=True), device=device),
        torch.as_tensor(np.array(prices[first:last], copy=True), device=device),
        torch.as_tensor(np.array(returns[first:last], copy=True), device=device),
    )


@torch.no_grad()
def transition_gain(
    policy: SharedExpiryOcePolicy,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    gain = np.empty(returns.shape[:2], dtype=np.float64)
    for first in range(0, len(states), batch_size):
        last = min(first + batch_size, len(states))
        state, price, result = load_batch(
            states, prices, returns, first=first, last=last, device=policy.env.device
        )
        action = unbounded_action(policy, state, price)
        gain[first:last] = torch.sum(
            action * result - 0.5 * action.square(), dim=-1
        ).double().cpu().numpy()
    return gain


def log_objective_and_probability(
    gain: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    log_terms = np.log(np.asarray(weights, dtype=np.float64))[:, None] - gain
    denominator = float(logsumexp(log_terms))
    probability = np.exp(log_terms - denominator)
    probability /= np.sum(probability, dtype=np.float64)
    normalizer = math.log(gain.shape[1] * float(np.sum(weights, dtype=np.float64)))
    objective = denominator - normalizer
    ess = float(1.0 / np.sum(np.square(probability), dtype=np.float64))
    return objective, probability, ess


def trainable_parameters(
    policy: SharedExpiryOcePolicy,
    *,
    affine_only: bool,
) -> tuple[torch.nn.Parameter, ...]:
    for parameter in policy.parameters():
        parameter.requires_grad_(not affine_only)
    if affine_only:
        if policy.linear_skip is None:
            raise RuntimeError("affine stage requires expiry-specific linear skips")
        for parameter in policy.linear_skip.parameters():
            parameter.requires_grad_(True)
        return tuple(policy.linear_skip.parameters())
    return tuple(policy.parameters())


def gradient_statistics(
    parameters: tuple[torch.nn.Parameter, ...],
) -> dict[str, float | int]:
    square = 0.0
    maximum = 0.0
    count = 0
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        nonfinite += int(torch.count_nonzero(~torch.isfinite(gradient)).item())
        finite = torch.nan_to_num(gradient).double()
        square += float(torch.sum(finite.square()).cpu())
        maximum = max(maximum, float(torch.amax(torch.abs(finite)).cpu()))
        count += gradient.numel()
    return {
        "gradient_l2_norm": math.sqrt(square),
        "gradient_rms": math.sqrt(square / max(count, 1)),
        "gradient_maximum_absolute": maximum,
        "parameter_count": count,
        "nonfinite_gradient_count": nonfinite,
    }


def fit_stage(
    *,
    policy: SharedExpiryOcePolicy,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    checkpoint: Path,
    stage: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    maximum_gradient_norm: float,
    maximum_line_search_steps: int,
    backtracking_factor: float,
) -> tuple[SharedExpiryOcePolicy, list[dict[str, Any]]]:
    affine_only = stage == "affine"
    parameters = trainable_parameters(policy, affine_only=affine_only)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 0
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=policy.env.device, weights_only=False)
        if payload.get("schema") != "dysanos-main2-tact-policy-v1":
            raise RuntimeError(f"unexpected checkpoint schema in {checkpoint}")
        if payload["stage"] != stage or bool(payload["affine_only"]) != affine_only:
            raise RuntimeError(f"checkpoint stage mismatch in {checkpoint}")
        policy.load_state_dict(payload["current_state_dict"])
        history = list(payload["history"])
        best_loss = float(payload["best_loss"])
        best_state = payload["best_state_dict"]
        start_epoch = int(payload["epoch"])
        if payload.get("complete", False):
            policy.load_state_dict(best_state)
            return policy, history
        print(
            json.dumps(
                {"stage": "resume", "checkpoint": str(checkpoint), "epoch": start_epoch}
            ),
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        gain = transition_gain(
            policy, states, prices, returns, batch_size=batch_size
        )
        loss, probability, tilted_ess = log_objective_and_probability(gain, weights)
        if loss < best_loss:
            best_loss = loss
            best_state = cpu_state_dict(policy)

        policy.zero_grad(set_to_none=True)
        for first in range(0, len(states), batch_size):
            last = min(first + batch_size, len(states))
            state, price, result = load_batch(
                states,
                prices,
                returns,
                first=first,
                last=last,
                device=policy.env.device,
            )
            action = unbounded_action(policy, state, price)
            batch_gain = torch.sum(
                action * result - 0.5 * action.square(), dim=-1
            )
            batch_probability = torch.as_tensor(
                probability[first:last],
                dtype=batch_gain.dtype,
                device=batch_gain.device,
            )
            torch.sum(-batch_probability * batch_gain).backward()

        preclip_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, maximum_gradient_norm)
        )
        stats = gradient_statistics(parameters)
        if stats["nonfinite_gradient_count"]:
            raise FloatingPointError(f"non-finite gradient in {checkpoint}")
        original_parameters = tuple(
            parameter.detach().clone() for parameter in parameters
        )
        accepted_step = 0.0
        candidate_loss = loss
        line_search_evaluations = 0
        step_size = learning_rate
        for line_search_index in range(maximum_line_search_steps):
            line_search_evaluations = line_search_index + 1
            with torch.no_grad():
                for parameter, original in zip(
                    parameters, original_parameters, strict=True
                ):
                    if parameter.grad is None:
                        parameter.copy_(original)
                    else:
                        parameter.copy_(original - step_size * parameter.grad)
            candidate_gain = transition_gain(
                policy, states, prices, returns, batch_size=batch_size
            )
            candidate_loss, _, _ = log_objective_and_probability(
                candidate_gain, weights
            )
            if candidate_loss < loss - 1.0e-12:
                accepted_step = step_size
                break
            step_size *= backtracking_factor
        if accepted_step == 0.0:
            with torch.no_grad():
                for parameter, original in zip(
                    parameters, original_parameters, strict=True
                ):
                    parameter.copy_(original)
        elif candidate_loss < best_loss:
            best_loss = candidate_loss
            best_state = cpu_state_dict(policy)

        record = {
            "epoch": epoch + 1,
            "log_expectation": loss,
            "candidate_log_expectation": candidate_loss,
            "objective_improvement_from_zero": -candidate_loss,
            "gradient_l2_norm_before_clipping": preclip_norm,
            **stats,
            "accepted_step_size": accepted_step,
            "line_search_evaluations": line_search_evaluations,
            "tilted_transition_effective_sample_size": tilted_ess,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        complete = accepted_step == 0.0 or epoch + 1 >= epochs
        if best_state is None:
            raise RuntimeError("training failed to create a finite best state")
        atomic_torch(
            {
                "schema": "dysanos-main2-tact-policy-v1",
                "stage": stage,
                "affine_only": affine_only,
                "epoch": epoch + 1,
                "complete": complete,
                "current_state_dict": cpu_state_dict(policy),
                "best_state_dict": best_state,
                "best_loss": best_loss,
                "history": history,
                "policy_config": asdict(policy.config),
            },
            checkpoint,
        )
        print(
            json.dumps(
                {
                    "stage": "train",
                    "checkpoint": checkpoint.name,
                    **record,
                }
            ),
            flush=True,
        )
        if accepted_step == 0.0:
            break
    if best_state is None:
        raise RuntimeError("training produced no best state")
    policy.load_state_dict(best_state)
    return policy, history


def fit_fold(
    *,
    measure: str,
    fold: str,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    output: Path,
    arguments: argparse.Namespace,
) -> tuple[SharedExpiryOcePolicy, dict[str, Any]]:
    fold_directory = output / measure.lower() / fold.lower()
    fold_directory.mkdir(parents=True, exist_ok=True)
    normalizer_path = fold_directory / "normalizers.npz"
    if normalizer_path.exists():
        with np.load(normalizer_path, allow_pickle=False) as payload:
            normalizers = {name: payload[name] for name in payload.files}
    else:
        normalizers = weighted_normalizers(
            states, prices, weights, batch_size=arguments.batch_size
        )
        temporary = normalizer_path.with_name(f".{normalizer_path.name}.tmp.npz")
        np.savez(temporary, **normalizers)
        os.replace(temporary, normalizer_path)
    policy = make_policy(
        device=torch.device(arguments.device),
        normalizers=normalizers,
        seed=arguments.seed,
        label=f"tact-{measure.lower()}-{fold.lower()}",
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if parameter_count != 56_709:
        raise RuntimeError(
            f"unexpected action-policy parameter count {parameter_count:,}"
        )
    policy, affine_history = fit_stage(
        policy=policy,
        states=states,
        prices=prices,
        returns=returns,
        weights=weights,
        checkpoint=fold_directory / "affine.pt",
        stage="affine",
        epochs=arguments.affine_epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        maximum_gradient_norm=arguments.maximum_gradient_norm,
        maximum_line_search_steps=arguments.maximum_line_search_steps,
        backtracking_factor=arguments.backtracking_factor,
    )
    policy, full_history = fit_stage(
        policy=policy,
        states=states,
        prices=prices,
        returns=returns,
        weights=weights,
        checkpoint=fold_directory / "full.pt",
        stage="full",
        epochs=arguments.full_epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        maximum_gradient_norm=arguments.maximum_gradient_norm,
        maximum_line_search_steps=arguments.maximum_line_search_steps,
        backtracking_factor=arguments.backtracking_factor,
    )
    return policy, {
        "training_paths": len(states),
        "training_antithetic_pairs": len(states) // 2,
        "training_weight_effective_sample_size": float(
            np.sum(weights, dtype=np.float64) ** 2
            / np.sum(np.square(weights), dtype=np.float64)
        ),
        "affine_epochs": len(affine_history),
        "full_epochs": len(full_history),
        "affine_final": affine_history[-1],
        "full_final": full_history[-1],
    }


@torch.no_grad()
def path_scores(
    policy: SharedExpiryOcePolicy,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_score = np.empty(len(states), dtype=np.float64)
    quadratic_score = np.empty(len(states), dtype=np.float64)
    for first in range(0, len(states), batch_size):
        last = min(first + batch_size, len(states))
        state, price, result = load_batch(
            states, prices, returns, first=first, last=last, device=policy.env.device
        )
        action = unbounded_action(policy, state, price)
        daily_raw = torch.sum(action * result, dim=-1)
        daily_quadratic = daily_raw - 0.5 * torch.sum(action.square(), dim=-1)
        raw_score[first:last] = torch.mean(daily_raw, dim=1).double().cpu().numpy()
        quadratic_score[first:last] = (
            torch.mean(daily_quadratic, dim=1).double().cpu().numpy()
        )
    return raw_score, quadratic_score


def weighted_antithetic_test(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int | list[float]]:
    if values.shape != weights.shape or len(values) % 2:
        raise ValueError("weighted antithetic test needs matching even vectors")
    weight_sum = float(np.sum(weights, dtype=np.float64))
    mean_weight = weight_sum / len(weights)
    mean = float(np.sum(weights * values, dtype=np.float64) / weight_sum)
    influence = weights * (values - mean) / mean_weight
    pair_influence = influence.reshape(-1, 2).mean(axis=1)
    standard_error = float(
        np.std(pair_influence, ddof=1) / math.sqrt(len(pair_influence))
    )
    z = mean / standard_error
    return {
        "paths": len(values),
        "antithetic_pairs": len(values) // 2,
        "mean_daily_captured_return": mean,
        "pair_based_standard_error": standard_error,
        "z_score": z,
        "two_sided_normal_p_value": math.erfc(abs(z) / math.sqrt(2.0)),
        "confidence_interval_95": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "effective_sample_size": float(
            weight_sum**2 / np.sum(np.square(weights), dtype=np.float64)
        ),
    }


def unconditional_coordinate_diagnostics(
    returns: np.ndarray,
    weights: np.ndarray,
    *,
    batch_size: int,
    materiality_floor: float = 1.0e-8,
) -> dict[str, Any]:
    """Compute ratio-estimator scores using adjacent antithetic pairs as units."""
    if returns.ndim != 3 or weights.shape != (len(returns),):
        raise ValueError(
            f"unexpected unconditional audit shapes {returns.shape}, {weights.shape}"
        )
    if len(returns) < 4 or len(returns) % 2:
        raise ValueError("unconditional audit needs at least two antithetic pairs")
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("unconditional audit batch size must be positive and even")
    weight_sum = float(np.sum(weights, dtype=np.float64))
    mean_weight = weight_sum / len(weights)
    weighted_return_sum = np.zeros(returns.shape[1:], dtype=np.float64)
    for first in range(0, len(returns), batch_size):
        last = min(first + batch_size, len(returns))
        weighted_return_sum += np.einsum(
            "p,pti->ti",
            np.asarray(weights[first:last], dtype=np.float64),
            np.asarray(returns[first:last], dtype=np.float64),
            optimize=True,
        )
    mean = weighted_return_sum / weight_sum

    pair_square_sum = np.zeros_like(mean)
    pair_count = len(returns) // 2
    for first in range(0, len(returns), batch_size):
        last = min(first + batch_size, len(returns))
        weight = np.asarray(weights[first:last], dtype=np.float64)
        value = np.asarray(returns[first:last], dtype=np.float64)
        influence = weight[:, None, None] * (value - mean) / mean_weight
        pair_influence = influence.reshape(-1, 2, *mean.shape).mean(axis=1)
        pair_square_sum += np.sum(np.square(pair_influence), axis=0)
    standard_error = np.sqrt(pair_square_sum / (pair_count - 1) / pair_count)
    z_score = np.divide(
        mean,
        standard_error,
        out=np.full_like(mean, np.nan),
        where=standard_error > 0.0,
    )
    materiality_score = mean / np.sqrt(
        np.square(standard_error) + materiality_floor**2
    )
    absolute_z = np.abs(z_score)
    absolute_materiality = np.abs(materiality_score)
    worst_flat = int(np.nanargmax(absolute_z))
    worst_time, worst_instrument = np.unravel_index(worst_flat, mean.shape)
    return {
        "paths": len(returns),
        "antithetic_pairs": pair_count,
        "coordinates": int(mean.size),
        "effective_sample_size": float(
            weight_sum**2 / np.sum(np.square(weights), dtype=np.float64)
        ),
        "maximum_absolute_z": float(np.nanmax(absolute_z)),
        "percentile_99_absolute_z": float(np.nanpercentile(absolute_z, 99.0)),
        "coordinates_absolute_z_above_1_96": int(np.sum(absolute_z > 1.96)),
        "coordinates_absolute_z_above_2_576": int(np.sum(absolute_z > 2.576)),
        "coordinates_absolute_z_above_5": int(np.sum(absolute_z > 5.0)),
        "materiality_floor": materiality_floor,
        "maximum_absolute_materiality_score": float(
            np.nanmax(absolute_materiality)
        ),
        "percentile_99_absolute_materiality_score": float(
            np.nanpercentile(absolute_materiality, 99.0)
        ),
        "coordinates_absolute_materiality_score_above_5": int(
            np.sum(absolute_materiality > 5.0)
        ),
        "worst_coordinate": {
            "time_index_zero_based": int(worst_time),
            "time_index_one_based": int(worst_time + 1),
            "instrument_index_zero_based": int(worst_instrument),
            "instrument_index_one_based": int(worst_instrument + 1),
            "mean": float(mean[worst_time, worst_instrument]),
            "pair_based_standard_error": float(
                standard_error[worst_time, worst_instrument]
            ),
            "z_score": float(z_score[worst_time, worst_instrument]),
            "materiality_score": float(
                materiality_score[worst_time, worst_instrument]
            ),
        },
    }


def run_measure(
    *,
    measure: str,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    output: Path,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    source_half = len(states) // 2
    fold_paths = arguments.paths_per_fold
    a_slice = slice(0, fold_paths)
    b_slice = slice(source_half, source_half + fold_paths)
    policy_a, fit_a = fit_fold(
        measure=measure,
        fold="A",
        states=states[a_slice],
        prices=prices[a_slice],
        returns=returns[a_slice],
        weights=weights[a_slice],
        output=output,
        arguments=arguments,
    )
    score_b, quadratic_b = path_scores(
        policy_a,
        states[b_slice],
        prices[b_slice],
        returns[b_slice],
        batch_size=arguments.batch_size,
    )
    del policy_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    policy_b, fit_b = fit_fold(
        measure=measure,
        fold="B",
        states=states[b_slice],
        prices=prices[b_slice],
        returns=returns[b_slice],
        weights=weights[b_slice],
        output=output,
        arguments=arguments,
    )
    score_a, quadratic_a = path_scores(
        policy_b,
        states[a_slice],
        prices[a_slice],
        returns[a_slice],
        batch_size=arguments.batch_size,
    )
    del policy_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_of_fold_score = np.concatenate((score_a, score_b))
    out_of_fold_quadratic = np.concatenate((quadratic_a, quadratic_b))
    out_of_fold_weights = np.concatenate((weights[a_slice], weights[b_slice]))
    measure_directory = output / measure.lower()
    np.save(measure_directory / "out-of-fold-raw-scores.npy", out_of_fold_score)
    np.save(
        measure_directory / "out-of-fold-quadratic-scores.npy",
        out_of_fold_quadratic,
    )
    return {
        "measure": measure,
        "objective": (
            "min log E_Q[exp(-a(X)'R + 0.5*a(X)'a(X))] over pooled "
            "path-time observations"
        ),
        "action_class": "uncapped 56,709-parameter shared-expiry affine-plus-deep",
        "normalization": "Q-weighted training fold only, frozen on the held-out fold",
        "unconditional_coordinate_diagnostics": unconditional_coordinate_diagnostics(
            returns,
            weights,
            batch_size=arguments.batch_size,
        ),
        "fold_A_fit": fit_a,
        "fold_B_fit": fit_b,
        "A_to_B": {
            "raw_return": weighted_antithetic_test(score_b, weights[b_slice]),
            "quadratic_score": weighted_antithetic_test(
                quadratic_b, weights[b_slice]
            ),
        },
        "B_to_A": {
            "raw_return": weighted_antithetic_test(score_a, weights[a_slice]),
            "quadratic_score": weighted_antithetic_test(
                quadratic_a, weights[a_slice]
            ),
        },
        "combined_out_of_fold": {
            "T_act": weighted_antithetic_test(
                out_of_fold_score, out_of_fold_weights
            ),
            "quadratic_score": weighted_antithetic_test(
                out_of_fold_quadratic, out_of_fold_weights
            ),
        },
    }


def run(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / "run.lock", timeout=0):
        print(json.dumps({"stage": "resources", **check_resources(arguments.output)}), flush=True)
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if arguments.paths_per_fold <= 0 or arguments.paths_per_fold > 50_000:
            raise ValueError("paths per fold must lie in [1, 50000]")
        if arguments.paths_per_fold % 2:
            raise ValueError("paths per fold must preserve complete antithetic pairs")
        torch.set_num_threads(arguments.cpu_threads)
        torch.set_float32_matmul_precision("high")
        if arguments.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        states = np.load(
            arguments.cache / "train-states.npy", mmap_mode="r", allow_pickle=False
        )
        prices = np.load(
            arguments.cache / "train-prices.npy", mmap_mode="r", allow_pickle=False
        )
        returns = np.load(
            arguments.cache / "train-returns.npy", mmap_mode="r", allow_pickle=False
        )
        measures = {
            "P": np.ones(100_000, dtype=np.float64),
            "Q1": np.load(
                arguments.cache / "train-q1-weights.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            "Q2": np.load(
                arguments.q2_audit / "old-q2-weights.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
        }
        if states.shape != (100_000, 25, 200):
            raise ValueError(f"unexpected state shape {states.shape}")
        if prices.shape != (100_000, 25, 4, 12):
            raise ValueError(f"unexpected price shape {prices.shape}")
        if returns.shape != (100_000, 25, 49):
            raise ValueError(f"unexpected return shape {returns.shape}")

        requested = [item.strip().upper() for item in arguments.measures.split(",")]
        report: dict[str, Any] = {
            "schema": "dysanos-main2-consistent-tact-crossfit-v1",
            "definition": {
                "fit": (
                    "two-fold cross-fit with pooled-transition exponential objective "
                    "and epsilon=1 action regularization"
                ),
                "score": "mean daily frozen out-of-fold a(X)'R",
                "statistic": (
                    "self-normalized Q mean divided by antithetic-pair ratio "
                    "standard error"
                ),
                "fold_assignment": (
                    f"first and second {arguments.paths_per_fold // 2:,} source "
                    "antithetic pairs form A and B"
                ),
            },
            "arguments": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(arguments).items()
            },
            "resources_at_start": check_resources(arguments.output),
            "measures": {},
        }
        for measure in requested:
            if measure not in measures:
                raise ValueError(f"unknown measure {measure!r}")
            result = run_measure(
                measure=measure,
                states=states,
                prices=prices,
                returns=returns,
                weights=measures[measure],
                output=arguments.output,
                arguments=arguments,
            )
            report["measures"][measure] = result
            report["resources_latest"] = check_resources(arguments.output)
            atomic_json(report, arguments.output / "report.partial.json")
            print(
                json.dumps(
                    {
                        "stage": "measure-complete",
                        "measure": measure,
                        "T_act": result["combined_out_of_fold"]["T_act"],
                    }
                ),
                flush=True,
            )
        report["complete"] = True
        atomic_json(report, arguments.output / "report.json")
        print(json.dumps({"stage": "complete", "report": str(arguments.output / "report.json")}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="directory containing the 100,000-path states, prices, returns, and Q1 weights",
    )
    result.add_argument(
        "--q2-audit",
        type=Path,
        required=True,
        help="directory containing old-q2-weights.npy",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--measures", default="Q2,Q1")
    result.add_argument("--paths-per-fold", type=int, default=50_000)
    result.add_argument("--batch-size", type=int, default=5_000)
    result.add_argument("--affine-epochs", type=int, default=30)
    result.add_argument("--full-epochs", type=int, default=30)
    result.add_argument("--learning-rate", type=float, default=1.0e-3)
    result.add_argument("--maximum-gradient-norm", type=float, default=10.0)
    result.add_argument("--maximum-line-search-steps", type=int, default=20)
    result.add_argument("--backtracking-factor", type=float, default=0.2)
    result.add_argument("--seed", type=int, default=20261006)
    result.add_argument("--cpu-threads", type=int, default=8)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
