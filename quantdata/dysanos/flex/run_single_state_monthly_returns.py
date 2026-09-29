"""Generate a fixed 25-step return population for an empirical measure fit."""

from __future__ import annotations

import argparse
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.dysanos.flex.front_spot_copula import (
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _Book,
    _book,
    _feature_statistics,
    _nms_book,
    _simulate_expiry_path_batch,
    _simulate_path_batch,
)


class _ZeroPolicy:
    """Return zero actions while reusing the canonical path simulator."""

    def __init__(self, num_outputs: int) -> None:
        self.num_outputs = num_outputs

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (len(features), self.num_outputs),
            dtype=features.dtype,
            device=features.device,
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=1_000_000)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument(
        "--book-expiry-days",
        type=int,
        nargs="+",
        default=[5, 10],
    )
    strike_group = parser.add_mutually_exclusive_group()
    strike_group.add_argument(
        "--relative-strikes",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional increasing relative-strike book. The ITM call is used at "
            "strikes no greater than one and the ITM put above one."
        ),
    )
    strike_group.add_argument(
        "--normalized-moneyness",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional increasing NMS book. Each NMS coordinate is converted "
            "to a pathwise relative strike when the option is issued."
        ),
    )
    parser.add_argument(
        "--option-return-horizon",
        choices=("one-day", "to-expiry"),
        default="one-day",
    )
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2_026_093_0)
    parser.add_argument("--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1))
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.normalized_moneyness is not None:
        _nms_book(
            tuple(arguments.book_expiry_days),
            normalized_moneyness=tuple(arguments.normalized_moneyness),
        )
    else:
        relative_strikes = (
            tuple(arguments.relative_strikes)
            if arguments.relative_strikes is not None
            else (0.99, 1.0, 1.01)
        )
        _book(tuple(arguments.book_expiry_days), relative_strikes=relative_strikes)
    integers = (
        arguments.num_paths,
        arguments.num_steps,
        arguments.batch_size,
        arguments.num_threads,
    )
    if min(integers) <= 0:
        raise ValueError("path generation controls must be positive")
    if arguments.num_paths % 2 or arguments.batch_size % 2:
        raise ValueError("path and batch counts must be even for antithetic simulation")
    if arguments.output.suffix.lower() != ".npy":
        raise ValueError("fixed returns output must use the .npy suffix")


def _history(payload: Any) -> tuple[list[Any], torch.Tensor, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("target-history payload is not a mapping")
    dates = payload.get("dates")
    states = payload.get("states")
    spots = payload.get("spot_refs")
    if (
        not isinstance(dates, list)
        or not isinstance(states, torch.Tensor)
        or not isinstance(spots, torch.Tensor)
    ):
        raise ValueError("target history is missing dates, states, or spot references")
    return dates, states, spots


def run(arguments: argparse.Namespace) -> None:
    """Generate and atomically publish the fixed gross-return matrix."""

    _validate_arguments(arguments)
    if arguments.normalized_moneyness is not None:
        book = _nms_book(
            tuple(arguments.book_expiry_days),
            normalized_moneyness=tuple(arguments.normalized_moneyness),
        )
    else:
        relative_strikes = (
            tuple(arguments.relative_strikes)
            if arguments.relative_strikes is not None
            else (0.99, 1.0, 1.01)
        )
        book = _book(
            tuple(arguments.book_expiry_days), relative_strikes=relative_strikes
        )
    if arguments.device == "cpu":
        torch.set_num_threads(arguments.num_threads)
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    payload = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates, states_value, spots_value = _history(payload)
    state_index = len(dates) // 2 if arguments.state_index < 0 else arguments.state_index
    if not 0 <= state_index < len(dates):
        raise ValueError("state index is outside the historical population")
    historical_states = states_value.to(dtype=env.fdtype, device=env.device)
    initial_state = historical_states[state_index]
    daily_transition, daily_factor, _ = exact_daily_transition(model)
    copula_fit, _ = fit_front_spot_copula_from_history(
        model=model,
        states=states_value.double().numpy(),
        spots=spots_value.double().numpy(),
        dates=dates,
        daily_transition=daily_transition.cpu().double().numpy(),
        daily_factor=daily_factor.cpu().double().numpy(),
        fit_before=arguments.copula_fit_before,
    )
    copula_loading = torch.as_tensor(
        copula_fit.state_loading, dtype=env.fdtype, device=env.device
    )
    feature_mean, feature_scale, _ = _feature_statistics(model, historical_states)
    transaction_cost = torch.zeros(len(book.labels), dtype=env.fdtype, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(arguments.seed)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(arguments.num_paths, arguments.num_steps, len(book.labels)),
    )
    started = time.monotonic()
    policy = _ZeroPolicy(len(book.labels))
    simulate_path_batch = (
        _simulate_expiry_path_batch
        if arguments.option_return_horizon == "to-expiry"
        else _simulate_path_batch
    )
    with torch.inference_mode():
        for first in range(0, arguments.num_paths, arguments.batch_size):
            batch_paths = min(arguments.batch_size, arguments.num_paths - first)
            _, returns, _ = simulate_path_batch(
                model=model,
                network=policy,
                book=book,
                initial_state=initial_state,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                daily_transition=daily_transition,
                daily_factor=daily_factor,
                copula_loading=copula_loading,
                copula_orthogonal_std=copula_fit.orthogonal_std,
                copula_total_std=copula_fit.total_std,
                transaction_cost=transaction_cost,
                action_l1_cap=1.0,
                action_ridge=0.0,
                num_paths=batch_paths,
                num_steps=arguments.num_steps,
                generator=generator,
                retain_returns=True,
            )
            assert returns is not None
            matrix[first : first + batch_paths] = returns.cpu().numpy()
            if first == 0 or first + batch_paths == arguments.num_paths or first % (
                25 * arguments.batch_size
            ) == 0:
                elapsed = time.monotonic() - started
                rate = (first + batch_paths) / max(elapsed, 1.0e-12)
                print(
                    f"generated {first + batch_paths}/{arguments.num_paths} paths "
                    f"at {rate:.1f} paths/s",
                    flush=True,
                )
    matrix.flush()
    del matrix
    os.replace(temporary, arguments.output)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
