"""Jointly fit the Q2-generating policy and its full-cost best response."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.run_convex_q_conditional_oce import (
    SharedExpiryOcePolicy,
    _alignment_check,
    _atomic_torch_save,
    _cached_batch,
    _feature_cache_paths,
    _fixed_feature_pnl,
    _full_population_oce_probabilities,
    _gradient_statistics,
    _prepare_feature_cache,
    _serialized_arguments,
    _setup,
    _validate_arguments,
)
from quantdata.dysanos.flex.run_convex_q_conditional_oce import (
    _parser as _single_policy_parser,
)

_EXPIRY_DAYS = (5, 10, 15, 20)
_CHECKPOINT_SCHEMA = "dysanos-joint-q2-full-cost-oce-v1"


def _parser() -> argparse.ArgumentParser:
    parser = _single_policy_parser()
    parser.description = __doc__
    parser.add_argument("--full-spot-cost-bps", type=float, required=True)
    parser.add_argument("--full-option-cost-bps", type=float, required=True)
    parser.add_argument(
        "--a3-start-update",
        type=int,
        default=50,
        help="Minimum number of full-population a2 updates before a3 can start.",
    )
    parser.add_argument("--a3-start-density-rms-tolerance", type=float, default=0.0)
    parser.add_argument("--a3-start-density-patience", type=int, default=1)
    parser.add_argument(
        "--freeze-a3",
        action="store_true",
        help=(
            "Resume and optimize a2 only, leaving the saved a3 policy and "
            "optimizer state unchanged until a later unfrozen resume."
        ),
    )
    parser.add_argument("--a3-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--a3-minimum-learning-rate", type=float, default=1.0e-7)
    parser.add_argument("--a3-warmup-updates", type=int, default=0)
    parser.add_argument("--a3-gradient-rms-tolerance", type=float, default=0.0)
    parser.add_argument("--a3-gradient-maximum-tolerance", type=float, default=0.0)
    parser.add_argument("--a3-scale-foc-tolerance", type=float, default=0.0)
    parser.add_argument("--q2-log-density-rms-tolerance", type=float, default=0.0)
    parser.add_argument("--joint-convergence-patience", type=int, default=5)
    parser.add_argument("--status-every-updates", type=int, default=1)
    return parser


def _validate_joint_arguments(arguments: argparse.Namespace) -> None:
    _validate_arguments(arguments)
    if not arguments.full_population_gradient:
        raise ValueError("joint Q2/a3 training requires --full-population-gradient")
    if arguments.fixed_feature_cache is None:
        raise ValueError("joint Q2/a3 training requires --fixed-feature-cache")
    if arguments.optimizer != "adam":
        raise ValueError("joint Q2/a3 training currently supports Adam only")
    if arguments.prepare_features_only:
        raise ValueError("prepare the shared feature cache with the single-policy runner")
    if arguments.initialize_policy_checkpoint is not None:
        raise ValueError("joint Q2/a3 training starts both policies from zero output")
    if arguments.valuation_only:
        raise ValueError("joint Q2/a3 training cannot run in valuation-only mode")
    if arguments.frozen_policy_checkpoints:
        raise ValueError("joint Q2/a3 training does not use frozen additive policies")
    if arguments.baseline_pnl is not None or arguments.claim_pnl is not None:
        raise ValueError("joint Q2/a3 training requires zero claim and no baseline PnL")
    if arguments.long_short_actions:
        raise ValueError("joint Q2/a3 training currently uses signed actions")
    if arguments.a3_start_update < 0 or arguments.a3_warmup_updates < 0:
        raise ValueError("a3 update controls must be nonnegative")
    if arguments.a3_start_update >= arguments.epochs:
        raise ValueError("a3 must start before the final configured epoch")
    integer_values = (
        arguments.a3_start_density_patience,
        arguments.joint_convergence_patience,
        arguments.status_every_updates,
    )
    if any(value <= 0 for value in integer_values):
        raise ValueError("joint convergence and status intervals must be positive")
    positive_values = (
        arguments.a3_learning_rate,
        arguments.a3_minimum_learning_rate,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
        raise ValueError("a3 learning rates must be finite-positive")
    if arguments.a3_minimum_learning_rate > arguments.a3_learning_rate:
        raise ValueError("a3 minimum learning rate exceeds its peak learning rate")
    nonnegative_values = (
        arguments.full_spot_cost_bps,
        arguments.full_option_cost_bps,
        arguments.a3_gradient_rms_tolerance,
        arguments.a3_gradient_maximum_tolerance,
        arguments.a3_scale_foc_tolerance,
        arguments.a3_start_density_rms_tolerance,
        arguments.q2_log_density_rms_tolerance,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in nonnegative_values):
        raise ValueError("joint costs and tolerances must be finite-nonnegative")
    if arguments.full_spot_cost_bps < arguments.spot_cost_bps:
        raise ValueError("full spot costs cannot be below the Q2-training costs")
    if arguments.full_option_cost_bps < arguments.option_cost_bps:
        raise ValueError("full option costs cannot be below the Q2-training costs")


def _cosine_learning_rate(
    *,
    peak: float,
    minimum: float,
    warmup_updates: int,
    update: int,
    total_updates: int,
) -> float:
    if update < warmup_updates:
        return peak * (update + 1) / max(warmup_updates, 1)
    progress = (update - warmup_updates) / max(
        total_updates - warmup_updates - 1,
        1,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum + (peak - minimum) * cosine


def _joint_oce_probabilities(
    *,
    q1_density: np.ndarray,
    low_cost_pnl: np.ndarray,
    full_cost_pnl: np.ndarray | None,
) -> dict[str, float | np.ndarray]:
    """Build detached Q2 and, when requested, the a3 exponential tilt.

    ``q2_probability`` sums to one on the fixed empirical population. The
    relative density is with respect to the supplied Q1 probabilities, so its
    logarithm is invariant to the arbitrary scale of ``q1_density``.
    """

    low_log_loss, q2_probability = _full_population_oce_probabilities(
        low_cost_pnl,
        q1_density,
    )
    result: dict[str, float | np.ndarray] = {
        "a2_log_loss": low_log_loss,
        "a2_oce": -low_log_loss,
        "q2_probability": q2_probability,
        "q2_log_relative_density": -low_cost_pnl - low_log_loss,
        "q2_effective_sample_size": float(
            1.0 / np.sum(np.square(q2_probability), dtype=np.float64)
        ),
        "a2_tilted_expected_pnl": float(
            np.sum(q2_probability * low_cost_pnl, dtype=np.float64)
        ),
    }
    if full_cost_pnl is None:
        return result
    full_log_loss, a3_tilted_probability = _full_population_oce_probabilities(
        full_cost_pnl,
        q2_probability,
    )
    result.update(
        {
            "a3_log_loss": full_log_loss,
            "a3_oce_under_q2": -full_log_loss,
            "a3_tilted_probability": a3_tilted_probability,
            "a3_tilted_effective_sample_size": float(
                1.0
                / np.sum(np.square(a3_tilted_probability), dtype=np.float64)
            ),
            "a3_tilted_expected_pnl": float(
                np.sum(
                    a3_tilted_probability * full_cost_pnl,
                    dtype=np.float64,
                )
            ),
        }
    )
    return result


def _paired_oce_standard_error(
    *,
    q2_probability: np.ndarray,
    full_cost_pnl: np.ndarray,
    antithetic_block_size: int,
) -> float:
    """Estimate fixed-policy OCE error from independent antithetic pairs.

    The influence function includes the self-normalization of the empirical Q2
    density. Paths are paired within each generation block because every block
    stores its base normals followed by their negatives.
    """

    if (
        q2_probability.ndim != 1
        or full_cost_pnl.shape != q2_probability.shape
        or len(q2_probability) == 0
    ):
        raise ValueError("Q2 probabilities and full-cost PnL must be matching vectors")
    if antithetic_block_size <= 0 or antithetic_block_size % 2:
        raise ValueError("antithetic block size must be positive and even")
    if np.any(~np.isfinite(q2_probability) | (q2_probability <= 0.0)):
        raise ValueError("Q2 probabilities must be finite-positive")
    if np.any(~np.isfinite(full_cost_pnl)):
        raise ValueError("full-cost PnL must be finite")

    log_q2_probability = np.log(q2_probability)
    log_weighted_exponential_mean = float(
        np.logaddexp.reduce(log_q2_probability - full_cost_pnl)
    )
    population_size = len(q2_probability)
    influence = (
        -np.exp(
            math.log(population_size)
            + log_q2_probability
            - full_cost_pnl
            - log_weighted_exponential_mean
        )
        + population_size * q2_probability
    )
    paired_blocks: list[np.ndarray] = []
    for first in range(0, population_size, antithetic_block_size):
        last = min(first + antithetic_block_size, population_size)
        count = last - first
        if count % 2:
            raise ValueError(f"final antithetic block has odd size {count}")
        half = count // 2
        paired_blocks.append(
            0.5
            * (
                influence[first : first + half]
                + influence[first + half : last]
            )
        )
    paired_influence = np.concatenate(paired_blocks)
    return float(
        np.std(paired_influence, ddof=1) / math.sqrt(len(paired_influence))
    )


def _gradient_norm(model: torch.nn.Module) -> float:
    gradients = [
        torch.linalg.vector_norm(parameter.grad.detach())
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(gradients)).cpu())


def _full_cost_tensor(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
) -> torch.Tensor:
    instruments = 1 + len(_EXPIRY_DAYS) * len(arguments.normalized_moneyness)
    return torch.tensor(
        (
            arguments.full_spot_cost_bps,
            *(arguments.full_option_cost_bps for _ in range(instruments - 1)),
        ),
        dtype=setup["env"].fdtype,
        device=setup["env"].device,
    ) / 1.0e4


def _new_policy(setup: Mapping[str, Any]) -> SharedExpiryOcePolicy:
    source: SharedExpiryOcePolicy = setup["policy"]
    return SharedExpiryOcePolicy(
        setup["env"],
        config=setup["policy_config"],
        state_mean=source.state_mean,
        state_scale=source.state_scale,
        log_price_mean=source.log_price_mean,
        log_price_scale=source.log_price_scale,
    )


def _population_values(
    *,
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    a2_policy: SharedExpiryOcePolicy,
    a3_policy: SharedExpiryOcePolicy,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    full_transaction_cost: torch.Tensor,
    a3_enabled: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
    num_paths = arguments.num_paths
    batch_size = arguments.optimization_batch_size
    a2_values = np.empty(num_paths, dtype=np.float64)
    a3_values = np.empty(num_paths, dtype=np.float64) if a3_enabled else None
    a2_action_sum = 0.0
    a3_action_sum = 0.0
    a2_cap_sum = 0.0
    a3_cap_sum = 0.0
    a2_maximum_action = 0.0
    a3_maximum_action = 0.0
    with torch.no_grad():
        for first in range(0, num_paths, batch_size):
            last = min(first + batch_size, num_paths)
            batch = _cached_batch(
                states=states,
                prices=prices,
                returns=returns,
                first=first,
                last=last,
                dtype=setup["env"].fdtype,
                device=setup["env"].device,
            )
            a2_pnl, a2_diagnostics = _fixed_feature_pnl(
                a2_policy,
                states=batch[0],
                option_prices=batch[1],
                returns=batch[2],
                transaction_cost=setup["transaction_cost"],
            )
            a2_values[first:last] = a2_pnl.cpu().double().numpy()
            count = last - first
            a2_action_sum += float(a2_diagnostics.mean_action_l1) * count
            a2_cap_sum += float(a2_diagnostics.cap_active_fraction) * count
            a2_maximum_action = max(
                a2_maximum_action,
                float(a2_diagnostics.maximum_absolute_action),
            )
            if a3_values is not None:
                a3_pnl, a3_diagnostics = _fixed_feature_pnl(
                    a3_policy,
                    states=batch[0],
                    option_prices=batch[1],
                    returns=batch[2],
                    transaction_cost=full_transaction_cost,
                )
                a3_values[first:last] = a3_pnl.cpu().double().numpy()
                a3_action_sum += float(a3_diagnostics.mean_action_l1) * count
                a3_cap_sum += float(a3_diagnostics.cap_active_fraction) * count
                a3_maximum_action = max(
                    a3_maximum_action,
                    float(a3_diagnostics.maximum_absolute_action),
                )
    diagnostics = {
        "a2_mean_action_l1": a2_action_sum / num_paths,
        "a2_maximum_absolute_action": a2_maximum_action,
        "a2_cap_active_fraction": a2_cap_sum / num_paths,
        "a3_mean_action_l1": a3_action_sum / num_paths if a3_enabled else 0.0,
        "a3_maximum_absolute_action": a3_maximum_action,
        "a3_cap_active_fraction": a3_cap_sum / num_paths if a3_enabled else 0.0,
    }
    return a2_values, a3_values, diagnostics


def _population_gradients(
    *,
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    a2_policy: SharedExpiryOcePolicy,
    a3_policy: SharedExpiryOcePolicy,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    full_transaction_cost: torch.Tensor,
    q2_probability: np.ndarray,
    a3_tilted_probability: np.ndarray | None,
) -> dict[str, float | int]:
    a2_policy.zero_grad(set_to_none=True)
    a3_policy.zero_grad(set_to_none=True)
    batch_size = arguments.optimization_batch_size
    for first in range(0, arguments.num_paths, batch_size):
        last = min(first + batch_size, arguments.num_paths)
        batch = _cached_batch(
            states=states,
            prices=prices,
            returns=returns,
            first=first,
            last=last,
            dtype=setup["env"].fdtype,
            device=setup["env"].device,
        )
        a2_pnl, _ = _fixed_feature_pnl(
            a2_policy,
            states=batch[0],
            option_prices=batch[1],
            returns=batch[2],
            transaction_cost=setup["transaction_cost"],
        )
        a2_probability = torch.as_tensor(
            q2_probability[first:last],
            dtype=torch.float64,
            device=setup["env"].device,
        )
        (-torch.sum(a2_probability * a2_pnl.double())).backward()
        if a3_tilted_probability is not None:
            a3_pnl, _ = _fixed_feature_pnl(
                a3_policy,
                states=batch[0],
                option_prices=batch[1],
                returns=batch[2],
                transaction_cost=full_transaction_cost,
            )
            a3_probability = torch.as_tensor(
                a3_tilted_probability[first:last],
                dtype=torch.float64,
                device=setup["env"].device,
            )
            (-torch.sum(a3_probability * a3_pnl.double())).backward()

    a2_rms, a2_maximum, a2_parameters = _gradient_statistics(a2_policy)
    result: dict[str, float | int] = {
        "a2_gradient_rms": a2_rms,
        "a2_gradient_maximum_absolute": a2_maximum,
        "a2_gradient_norm_before_clip": _gradient_norm(a2_policy),
        "a2_parameter_count": a2_parameters,
    }
    if a3_tilted_probability is not None:
        a3_rms, a3_maximum, a3_parameters = _gradient_statistics(a3_policy)
        result.update(
            {
                "a3_gradient_rms": a3_rms,
                "a3_gradient_maximum_absolute": a3_maximum,
                "a3_gradient_norm_before_clip": _gradient_norm(a3_policy),
                "a3_parameter_count": a3_parameters,
            }
        )
    return result


def _convergence_checks(
    record: Mapping[str, float | int | bool],
    arguments: argparse.Namespace,
) -> dict[str, bool]:
    return {
        "a2_gradient_rms": (
            arguments.full_gradient_rms_tolerance <= 0.0
            or float(record["a2_gradient_rms"])
            <= arguments.full_gradient_rms_tolerance
        ),
        "a2_gradient_maximum": (
            arguments.full_gradient_maximum_tolerance <= 0.0
            or float(record["a2_gradient_maximum_absolute"])
            <= arguments.full_gradient_maximum_tolerance
        ),
        "a2_scale_foc": (
            arguments.full_gradient_scale_foc_tolerance <= 0.0
            or abs(float(record["a2_tilted_expected_pnl"]))
            <= arguments.full_gradient_scale_foc_tolerance
        ),
        "q2_density_stability": (
            arguments.q2_log_density_rms_tolerance <= 0.0
            or float(record["q2_log_density_rms_change"])
            <= arguments.q2_log_density_rms_tolerance
        ),
        "a3_enabled": bool(record["a3_enabled"]),
        "a3_gradient_rms": (
            bool(record["a3_enabled"])
            and (
                arguments.a3_gradient_rms_tolerance <= 0.0
                or float(record["a3_gradient_rms"])
                <= arguments.a3_gradient_rms_tolerance
            )
        ),
        "a3_gradient_maximum": (
            bool(record["a3_enabled"])
            and (
                arguments.a3_gradient_maximum_tolerance <= 0.0
                or float(record["a3_gradient_maximum_absolute"])
                <= arguments.a3_gradient_maximum_tolerance
            )
        ),
        "a3_scale_foc": (
            bool(record["a3_enabled"])
            and (
                arguments.a3_scale_foc_tolerance <= 0.0
                or abs(float(record["a3_tilted_expected_pnl"]))
                <= arguments.a3_scale_foc_tolerance
            )
        ),
    }


def _checkpoint_payload(
    *,
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    a2_policy: SharedExpiryOcePolicy,
    a3_policy: SharedExpiryOcePolicy,
    a2_optimizer: torch.optim.Optimizer,
    a3_optimizer: torch.optim.Optimizer,
    epoch: int,
    a3_updates: int,
    history: list[dict[str, Any]],
    convergence_count: int,
    a3_started: bool,
    a3_start_stability_count: int,
    previous_log_q2_density: np.ndarray | None,
    phase: str,
) -> dict[str, Any]:
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "phase": phase,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arguments": _serialized_arguments(arguments),
        "policy_config": asdict(setup["policy_config"]),
        "a2_policy_state_dict": a2_policy.state_dict(),
        "a3_policy_state_dict": a3_policy.state_dict(),
        "a2_optimizer_state_dict": a2_optimizer.state_dict(),
        "a3_optimizer_state_dict": a3_optimizer.state_dict(),
        "epoch": epoch,
        "a3_updates": a3_updates,
        "history": history,
        "convergence_count": convergence_count,
        "a3_started": a3_started,
        "a3_start_stability_count": a3_start_stability_count,
        "previous_log_q2_density": (
            None
            if previous_log_q2_density is None
            else torch.from_numpy(previous_log_q2_density.copy())
        ),
    }


def _load_checkpoint(
    path: Path,
    *,
    arguments: argparse.Namespace,
    setup: Mapping[str, Any],
    a2_policy: SharedExpiryOcePolicy,
    a3_policy: SharedExpiryOcePolicy,
    a2_optimizer: torch.optim.Optimizer,
    a3_optimizer: torch.optim.Optimizer,
) -> tuple[int, int, list[dict[str, Any]], int, bool, int, np.ndarray | None, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("joint Q2/a3 checkpoint identity is invalid")
    if payload.get("policy_config") != asdict(setup["policy_config"]):
        raise ValueError("joint Q2/a3 checkpoint policy geometry has changed")
    saved_arguments = payload.get("arguments")
    current_arguments = _serialized_arguments(arguments)
    identity_keys = (
        "checkpoint",
        "target_history",
        "returns",
        "weights",
        "fixed_feature_cache",
        "num_paths",
        "num_steps",
        "linear_action_skip",
        "include_calendar_time",
        "normalized_moneyness",
        "hidden_width",
        "hidden_layers",
        "action_l1_cap",
        "spot_cost_bps",
        "option_cost_bps",
        "full_spot_cost_bps",
        "full_option_cost_bps",
        "seed",
    )
    if not isinstance(saved_arguments, Mapping) or any(
        saved_arguments.get(key) != current_arguments.get(key) for key in identity_keys
    ):
        raise ValueError("joint Q2/a3 checkpoint experiment identity has changed")
    a2_policy.load_state_dict(payload["a2_policy_state_dict"])
    a3_policy.load_state_dict(payload["a3_policy_state_dict"])
    a2_optimizer.load_state_dict(payload["a2_optimizer_state_dict"])
    a3_optimizer.load_state_dict(payload["a3_optimizer_state_dict"])
    previous = payload.get("previous_log_q2_density")
    previous_array = (
        None
        if previous is None
        else np.asarray(previous.detach().cpu().double().numpy())
    )
    return (
        int(payload["epoch"]),
        int(payload["a3_updates"]),
        list(payload.get("history", [])),
        int(payload.get("convergence_count", 0)),
        bool(payload.get("a3_started", int(payload["a3_updates"]) > 0)),
        int(payload.get("a3_start_stability_count", 0)),
        previous_array,
        str(payload["phase"]),
    )


def _write_status(
    *,
    path: Path,
    phase: str,
    epoch: int,
    arguments: argparse.Namespace,
    latest: Mapping[str, Any] | None,
) -> None:
    write_json_atomic(
        {
            "schema": _CHECKPOINT_SCHEMA,
            "phase": phase,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "epoch": epoch,
            "maximum_epochs": arguments.epochs,
            "latest": latest,
        },
        path,
    )


def _save_final_arrays(
    *,
    output_directory: Path,
    a2_pnl: np.ndarray,
    a3_pnl: np.ndarray,
    q2_probability: np.ndarray,
) -> None:
    for name, values in (
        ("a2-low-cost-pnl.npy", a2_pnl),
        ("a3-full-cost-pnl.npy", a3_pnl),
        ("q2-probability.npy", q2_probability),
    ):
        path = output_directory / name
        temporary = path.with_name(f".{path.name}.tmp")
        with open(temporary, "wb") as stream:
            np.save(stream, values)
        os.replace(temporary, path)


def run(arguments: argparse.Namespace) -> None:
    """Run restartable joint Q2 construction and full-cost best-response fitting."""

    _validate_joint_arguments(arguments)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output_directory / "joint-q2-full-cost.lock"):
        _run_locked(arguments)


def _run_locked(arguments: argparse.Namespace) -> None:
    torch.manual_seed(arguments.seed)
    setup = _setup(arguments)
    setup["frozen_policies"] = ()
    a2_policy: SharedExpiryOcePolicy = setup["policy"]
    torch.manual_seed(arguments.seed + 1)
    a3_policy = _new_policy(setup)
    full_transaction_cost = _full_cost_tensor(setup, arguments)

    returns = np.load(arguments.returns, mmap_mode="r", allow_pickle=False)
    weights = np.load(arguments.weights, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        1_000_000,
        25,
        1 + len(_EXPIRY_DAYS) * len(arguments.normalized_moneyness),
    )
    if returns.shape != expected_shape:
        raise ValueError(f"expected returns shape {expected_shape}, found {returns.shape}")
    if weights.shape != (1_000_000,) or np.any(~np.isfinite(weights) | (weights <= 0.0)):
        raise ValueError("Q1 weights must be one million finite-positive values")
    selected_weights = np.array(
        weights[: arguments.num_paths],
        dtype=np.float64,
        copy=True,
    )
    selected_weights *= arguments.num_paths / np.sum(selected_weights)

    alignment_error = _alignment_check(setup, arguments, returns)
    print(f"fixed-path alignment maximum error={alignment_error:.6g}", flush=True)
    _prepare_feature_cache(setup, arguments, directory=arguments.fixed_feature_cache)
    state_path, price_path, _ = _feature_cache_paths(arguments.fixed_feature_cache)
    states = np.load(state_path, mmap_mode="r", allow_pickle=False)
    prices = np.load(price_path, mmap_mode="r", allow_pickle=False)

    a2_optimizer = torch.optim.Adam(a2_policy.parameters(), lr=arguments.learning_rate)
    a3_optimizer = torch.optim.Adam(a3_policy.parameters(), lr=arguments.a3_learning_rate)
    checkpoint_path = arguments.output_directory / "joint-q2-full-cost.pt"
    status_path = arguments.output_directory / "joint-q2-full-cost-status.json"
    report_path = arguments.output_directory / "joint-q2-full-cost-report.json"
    epoch = 0
    a3_updates = 0
    history: list[dict[str, Any]] = []
    convergence_count = 0
    a3_started = False
    a3_start_stability_count = 0
    previous_log_q2_density: np.ndarray | None = None
    phase = "training"
    if arguments.resume and checkpoint_path.exists():
        (
            epoch,
            a3_updates,
            history,
            convergence_count,
            a3_started,
            a3_start_stability_count,
            previous_log_q2_density,
            phase,
        ) = _load_checkpoint(
            checkpoint_path,
            arguments=arguments,
            setup=setup,
            a2_policy=a2_policy,
            a3_policy=a3_policy,
            a2_optimizer=a2_optimizer,
            a3_optimizer=a3_optimizer,
        )
        if phase == "budget_exhausted" and epoch < arguments.epochs:
            phase = "training"
        print(f"resumed joint Q2/a3 phase={phase} epoch={epoch}", flush=True)

    while phase == "training" and epoch < arguments.epochs:
        started = time.monotonic()
        a3_enabled = a3_started and not arguments.freeze_a3
        a2_learning_rate = _cosine_learning_rate(
            peak=arguments.learning_rate,
            minimum=arguments.minimum_learning_rate,
            warmup_updates=arguments.warmup_updates,
            update=epoch,
            total_updates=arguments.epochs,
        )
        a3_learning_rate = _cosine_learning_rate(
            peak=arguments.a3_learning_rate,
            minimum=arguments.a3_minimum_learning_rate,
            warmup_updates=arguments.a3_warmup_updates,
            update=a3_updates,
            total_updates=max(arguments.epochs - arguments.a3_start_update, 1),
        )
        for group in a2_optimizer.param_groups:
            group["lr"] = a2_learning_rate
        for group in a3_optimizer.param_groups:
            group["lr"] = a3_learning_rate

        a2_pnl, a3_pnl, diagnostics = _population_values(
            setup=setup,
            arguments=arguments,
            a2_policy=a2_policy,
            a3_policy=a3_policy,
            states=states,
            prices=prices,
            returns=returns,
            full_transaction_cost=full_transaction_cost,
            a3_enabled=a3_enabled,
        )
        probabilities = _joint_oce_probabilities(
            q1_density=selected_weights,
            low_cost_pnl=a2_pnl,
            full_cost_pnl=a3_pnl,
        )
        log_q2_density = np.asarray(
            probabilities["q2_log_relative_density"],
            dtype=np.float64,
        )
        density_rms_change = (
            math.inf
            if previous_log_q2_density is None
            else float(
                np.sqrt(
                    np.mean(
                        np.square(log_q2_density - previous_log_q2_density),
                        dtype=np.float64,
                    )
                )
            )
        )
        eligible_to_start_a3 = epoch >= arguments.a3_start_update
        density_ready_to_start_a3 = (
            arguments.a3_start_density_rms_tolerance <= 0.0
            or density_rms_change <= arguments.a3_start_density_rms_tolerance
        )
        if not a3_started:
            a3_start_stability_count = (
                a3_start_stability_count + 1
                if eligible_to_start_a3 and density_ready_to_start_a3
                else 0
            )
            if a3_start_stability_count >= arguments.a3_start_density_patience:
                a3_started = True
        gradients = _population_gradients(
            setup=setup,
            arguments=arguments,
            a2_policy=a2_policy,
            a3_policy=a3_policy,
            states=states,
            prices=prices,
            returns=returns,
            full_transaction_cost=full_transaction_cost,
            q2_probability=np.asarray(probabilities["q2_probability"]),
            a3_tilted_probability=(
                np.asarray(probabilities["a3_tilted_probability"])
                if a3_enabled
                else None
            ),
        )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "a3_enabled": a3_enabled,
            "a3_frozen": arguments.freeze_a3,
            "a3_starts_next_update": (
                a3_started and not a3_enabled and not arguments.freeze_a3
            ),
            "a3_start_stability_count": a3_start_stability_count,
            "a3_updates": a3_updates,
            "a2_learning_rate": a2_learning_rate,
            "a3_learning_rate": a3_learning_rate if a3_enabled else 0.0,
            "a2_oce_under_q1": probabilities["a2_oce"],
            "a2_tilted_expected_pnl": probabilities["a2_tilted_expected_pnl"],
            "q2_effective_sample_size": probabilities["q2_effective_sample_size"],
            "q2_log_density_rms_change": density_rms_change,
            **diagnostics,
            **gradients,
        }
        if a3_enabled:
            a3_mc_standard_error = _paired_oce_standard_error(
                q2_probability=np.asarray(probabilities["q2_probability"]),
                full_cost_pnl=np.asarray(a3_pnl),
                antithetic_block_size=arguments.batch_size,
            )
            record.update(
                {
                    "a3_oce_under_q2": probabilities["a3_oce_under_q2"],
                    "a3_paired_mc_standard_error": a3_mc_standard_error,
                    "a3_z_vs_zero": (
                        float(probabilities["a3_oce_under_q2"])
                        / a3_mc_standard_error
                        if a3_mc_standard_error > 0.0
                        else math.inf
                    ),
                    "a3_tilted_expected_pnl": probabilities[
                        "a3_tilted_expected_pnl"
                    ],
                    "a3_tilted_effective_sample_size": probabilities[
                        "a3_tilted_effective_sample_size"
                    ],
                }
            )
        checks = _convergence_checks(record, arguments)
        convergence_count = (
            convergence_count + 1 if all(checks.values()) else 0
        )
        converged = convergence_count >= arguments.joint_convergence_patience
        record["convergence_checks"] = checks
        record["convergence_patience_count"] = convergence_count
        record["seconds"] = time.monotonic() - started

        if not converged:
            torch.nn.utils.clip_grad_norm_(
                a2_policy.parameters(),
                arguments.gradient_clip_norm,
            )
            a2_optimizer.step()
            if a3_enabled:
                torch.nn.utils.clip_grad_norm_(
                    a3_policy.parameters(),
                    arguments.gradient_clip_norm,
                )
                a3_optimizer.step()
                a3_updates += 1
        previous_log_q2_density = log_q2_density.copy()
        epoch += 1
        history.append(record)
        if epoch % arguments.status_every_updates == 0 or converged:
            print(json.dumps(record), flush=True)
            _write_status(
                path=status_path,
                phase="converged" if converged else "training",
                epoch=epoch,
                arguments=arguments,
                latest=record,
            )
        if (
            epoch % arguments.checkpoint_every_updates == 0
            or epoch == arguments.epochs
            or converged
        ):
            _atomic_torch_save(
                _checkpoint_payload(
                    setup=setup,
                    arguments=arguments,
                    a2_policy=a2_policy,
                    a3_policy=a3_policy,
                    a2_optimizer=a2_optimizer,
                    a3_optimizer=a3_optimizer,
                    epoch=epoch,
                    a3_updates=a3_updates,
                    history=history,
                    convergence_count=convergence_count,
                    a3_started=a3_started,
                    a3_start_stability_count=a3_start_stability_count,
                    previous_log_q2_density=previous_log_q2_density,
                    phase="converged" if converged else "training",
                ),
                checkpoint_path,
            )
        if converged:
            phase = "converged"

    if phase == "training":
        phase = "budget_exhausted"
        _atomic_torch_save(
            _checkpoint_payload(
                setup=setup,
                arguments=arguments,
                a2_policy=a2_policy,
                a3_policy=a3_policy,
                a2_optimizer=a2_optimizer,
                a3_optimizer=a3_optimizer,
                epoch=epoch,
                a3_updates=a3_updates,
                history=history,
                convergence_count=convergence_count,
                a3_started=a3_started,
                a3_start_stability_count=a3_start_stability_count,
                previous_log_q2_density=previous_log_q2_density,
                phase=phase,
            ),
            checkpoint_path,
        )

    a2_pnl, a3_pnl, diagnostics = _population_values(
        setup=setup,
        arguments=arguments,
        a2_policy=a2_policy,
        a3_policy=a3_policy,
        states=states,
        prices=prices,
        returns=returns,
        full_transaction_cost=full_transaction_cost,
        a3_enabled=True,
    )
    assert a3_pnl is not None
    probabilities = _joint_oce_probabilities(
        q1_density=selected_weights,
        low_cost_pnl=a2_pnl,
        full_cost_pnl=a3_pnl,
    )
    _save_final_arrays(
        output_directory=arguments.output_directory,
        a2_pnl=a2_pnl,
        a3_pnl=a3_pnl,
        q2_probability=np.asarray(probabilities["q2_probability"]),
    )
    report = {
        "schema": _CHECKPOINT_SCHEMA,
        "phase": phase,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "q1": "supplied convex entropy measure",
            "q2_density": "dQ2/dQ1 = exp(-Pi_low(a2)) / E_Q1[exp(-Pi_low(a2))]",
            "a2_objective": "sup_a2 -log E_Q1[exp(-Pi_low(a2))]",
            "a2_costs_bps": {
                "spot": arguments.spot_cost_bps,
                "options": arguments.option_cost_bps,
            },
            "a3_objective": "sup_a3 -log E_Q2[exp(-Pi_c(a3))]",
            "a3_costs_bps": {
                "spot": arguments.full_spot_cost_bps,
                "options": arguments.full_option_cost_bps,
            },
            "coupling": "Q2 density is detached from the a3 gradient; no Q3 is formed",
        },
        "arguments": _serialized_arguments(arguments),
        "policy_config": asdict(setup["policy_config"]),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history_sha256": checkpoint_digest(arguments.target_history),
        "fixed_path_alignment_maximum_error": alignment_error,
        "epoch": epoch,
        "a3_updates": a3_updates,
        "convergence_patience_count": convergence_count,
        "latest_training_record": history[-1] if history else None,
        "final": {
            "a2_oce_under_q1": probabilities["a2_oce"],
            "a2_tilted_expected_pnl": probabilities["a2_tilted_expected_pnl"],
            "q2_effective_sample_size": probabilities["q2_effective_sample_size"],
            "a3_oce_under_q2": probabilities["a3_oce_under_q2"],
            "a3_paired_mc_standard_error": _paired_oce_standard_error(
                q2_probability=np.asarray(probabilities["q2_probability"]),
                full_cost_pnl=a3_pnl,
                antithetic_block_size=arguments.batch_size,
            ),
            "a3_tilted_expected_pnl": probabilities["a3_tilted_expected_pnl"],
            "a3_tilted_effective_sample_size": probabilities[
                "a3_tilted_effective_sample_size"
            ],
            **diagnostics,
        },
    }
    write_json_atomic(report, report_path)
    _write_status(
        path=status_path,
        phase=phase,
        epoch=epoch,
        arguments=arguments,
        latest=report["final"],
    )
    print(json.dumps({"status": phase, "final": report["final"]}, indent=2), flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
