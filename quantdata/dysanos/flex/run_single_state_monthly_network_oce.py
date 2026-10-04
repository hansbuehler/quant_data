"""Fit and validate a 25-step OCE measure from one initial DYSANOS state."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.front_spot_copula import fit_front_spot_copula_from_history
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.network_oce import l1_capped_action
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_inverse_cdf
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@dataclass(frozen=True)
class _Book:
    """Rolling spot plus the requested ITM/ATM options at each expiry."""

    expiry_days: tuple[int, ...]
    relative_strikes: tuple[float, ...] | None
    normalized_moneyness: tuple[float, ...] | None
    initial_days: tuple[float, ...]
    final_days: tuple[float, ...]
    is_put: tuple[bool, ...]
    labels: tuple[str, ...]


def _book(
    expiry_days: tuple[int, ...],
    *,
    relative_strikes: tuple[float, ...] = (0.99, 1.0, 1.01),
) -> _Book:
    if (
        not expiry_days
        or any(day <= 1 for day in expiry_days)
        or tuple(sorted(set(expiry_days))) != expiry_days
    ):
        raise ValueError("book expiry days must be unique, increasing integers above one")
    if (
        not relative_strikes
        or any(not math.isfinite(strike) or strike <= 0.0 for strike in relative_strikes)
        or tuple(sorted(set(relative_strikes))) != relative_strikes
    ):
        raise ValueError("book relative strikes must be finite-positive and increasing")
    book_relative_strikes: list[float] = []
    initial_days: list[float] = []
    final_days: list[float] = []
    is_put: list[bool] = []
    labels = ["spot"]
    for day in expiry_days:
        for strike in relative_strikes:
            put = strike > 1.0
            book_relative_strikes.append(strike)
            initial_days.append(float(day))
            final_days.append(float(day - 1))
            is_put.append(put)
            if strike == 1.0:
                labels.append(f"{day}D ATM call")
            else:
                labels.append(
                    f"{day}D {100.0 * strike:.6g}% ITM {'put' if put else 'call'}"
                )
    return _Book(
        expiry_days=expiry_days,
        relative_strikes=tuple(book_relative_strikes),
        normalized_moneyness=None,
        initial_days=tuple(initial_days),
        final_days=tuple(final_days),
        is_put=tuple(is_put),
        labels=tuple(labels),
    )


def _nms_book(
    expiry_days: tuple[int, ...],
    *,
    normalized_moneyness: tuple[float, ...],
) -> _Book:
    """Construct an option book whose relative strikes are set at issue time."""

    if (
        not expiry_days
        or any(day <= 1 for day in expiry_days)
        or tuple(sorted(set(expiry_days))) != expiry_days
    ):
        raise ValueError("book expiry days must be unique, increasing integers above one")
    if (
        not normalized_moneyness
        or any(not math.isfinite(score) for score in normalized_moneyness)
        or tuple(sorted(set(normalized_moneyness))) != normalized_moneyness
    ):
        raise ValueError("book NMS coordinates must be finite, unique, and increasing")
    book_nms: list[float] = []
    initial_days: list[float] = []
    final_days: list[float] = []
    is_put: list[bool] = []
    labels = ["spot"]
    for day in expiry_days:
        for score in normalized_moneyness:
            put = score > 0.0
            book_nms.append(score)
            initial_days.append(float(day))
            final_days.append(float(day - 1))
            is_put.append(put)
            if score == 0.0:
                labels.append(f"{day}D NMS 0 ATM call")
            else:
                labels.append(
                    f"{day}D NMS {score:.6g} ITM {'put' if put else 'call'}"
                )
    return _Book(
        expiry_days=expiry_days,
        relative_strikes=None,
        normalized_moneyness=tuple(book_nms),
        initial_days=tuple(initial_days),
        final_days=tuple(final_days),
        is_put=tuple(is_put),
        labels=tuple(labels),
    )


_DEFAULT_BOOK = _book((5, 10))
_LABELS = _DEFAULT_BOOK.labels


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fixed-path-returns-output",
        type=Path,
        default=None,
        help=(
            "Optional .npy file for the fixed validation-path gross-return matrix. "
            "When fitted by a downstream convex solver, this becomes that solver's "
            "training sample."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=-1)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument(
        "--book-expiry-days",
        type=int,
        nargs="+",
        default=list(_DEFAULT_BOOK.expiry_days),
    )
    parser.add_argument(
        "--option-return-horizon",
        choices=("one-day", "to-expiry"),
        default="one-day",
    )
    parser.add_argument("--training-paths", type=int, default=1_000_000)
    parser.add_argument("--validation-paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--warmup-updates", type=int, default=50)
    parser.add_argument("--action-l1-cap", type=float, default=16.0)
    parser.add_argument("--spot-cost-bps", type=float, default=1.0)
    parser.add_argument("--option-cost-bps", type=float, default=10.0)
    parser.add_argument("--action-ridge", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=100.0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2_026_092_9)
    parser.add_argument("--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1))
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    _book(tuple(arguments.book_expiry_days))
    integers = (
        arguments.num_steps,
        arguments.training_paths,
        arguments.validation_paths,
        arguments.batch_size,
        arguments.hidden_width,
        arguments.hidden_layers,
        arguments.checkpoint_every_updates,
    )
    if min(integers) <= 0:
        raise ValueError("monthly OCE integer controls must be positive")
    if arguments.training_paths % 2 or arguments.validation_paths % 2:
        raise ValueError("training and validation path counts must be even")
    if arguments.batch_size % 2:
        raise ValueError("the antithetic path batch size must be even")
    positive = (
        arguments.learning_rate,
        arguments.minimum_learning_rate,
        arguments.action_l1_cap,
        arguments.spot_cost_bps,
        arguments.option_cost_bps,
        arguments.gradient_clip_norm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("monthly OCE scalar controls must be finite-positive")
    if arguments.minimum_learning_rate > arguments.learning_rate:
        raise ValueError("minimum learning rate exceeds the initial rate")
    if arguments.warmup_updates < 0:
        raise ValueError("warmup updates cannot be negative")
    if not math.isfinite(arguments.action_ridge) or arguments.action_ridge < 0.0:
        raise ValueError("action ridge must be finite and nonnegative")


def _decode(model: Any, states: torch.Tensor) -> tuple[Any, Any]:
    with torch.no_grad():
        raw = model.raw_states_from_states(states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
    return parameters, surface


def _features(
    model: Any,
    states: torch.Tensor,
    parameters: Any,
    surface: Any,
) -> torch.Tensor:
    density = torch.sqrt(torch.clamp_min(surface.densities, 0.0)).reshape(len(states), -1)
    excess_dlv = torch.clamp_min(
        parameters.dlvsqs - model.config.geometry.min_dlvsq,
        torch.finfo(states.dtype).tiny,
    )
    log_dlv = torch.log(excess_dlv).reshape(len(states), -1)
    values = torch.cat((states, density, log_dlv), dim=1)
    if not bool(torch.all(torch.isfinite(values))):
        raise FloatingPointError("monthly OCE features are non-finite")
    return values


def _feature_statistics(
    model: Any,
    historical_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    parameters, surface = _decode(model, historical_states)
    values = _features(model, historical_states, parameters, surface)
    mean = torch.mean(values, dim=0)
    standard_deviation = torch.std(values, dim=0, correction=0)
    scale = torch.where(
        standard_deviation > 1.0e-5,
        standard_deviation,
        torch.ones_like(standard_deviation),
    )
    blocks = {
        "h": historical_states.shape[1],
        "sqrt_density": surface.densities[0].numel(),
        "log_dlvsq_excess": parameters.dlvsqs[0].numel(),
        "total": values.shape[1],
    }
    return mean, scale, blocks


def _price_calls(
    model: Any,
    surface: Any,
    *,
    days: torch.Tensor,
    strikes: torch.Tensor,
) -> torch.Tensor:
    return model.sanos_decoder.price_calls_batch(
        surface,
        expiries_y=days[None].expand(len(surface.densities), -1) / DAYS_PER_YEAR,
        strikes=strikes,
    )


def _expiry_option_returns(
    *,
    gross_horizon: torch.Tensor,
    issue_values: torch.Tensor,
    strikes: torch.Tensor,
    is_put: torch.Tensor,
) -> torch.Tensor:
    """Return fixed-cash options from entry price to terminal intrinsic value."""

    if strikes.ndim == 1 and strikes.shape[0] == issue_values.shape[1]:
        strike_values = strikes[None].expand_as(issue_values)
    else:
        strike_values = strikes
    if strike_values.shape != issue_values.shape:
        raise ValueError(
            "expiry strikes and issue values must have the same path/instrument shape"
        )
    intrinsic = torch.where(
        is_put[None],
        torch.clamp_min(strike_values - gross_horizon[:, None], 0.0),
        torch.clamp_min(gross_horizon[:, None] - strike_values, 0.0),
    )
    return intrinsic / issue_values - 1.0


def _nms_relative_strikes(
    model: Any,
    surface: Any,
    *,
    days: torch.Tensor,
    normalized_moneyness: torch.Tensor,
) -> torch.Tensor:
    """Convert NMS coordinates to pathwise relative strikes at issue time."""

    if days.ndim != 1 or normalized_moneyness.shape != days.shape:
        raise ValueError("NMS days and coordinates must be matching vectors")
    nodes = model.sanos_decoder.model_expiries_y
    query = days / DAYS_PER_YEAR
    if bool(torch.any(query < 0.0)) or bool(torch.any(query > nodes[-1] + 1.0e-12)):
        raise ValueError("NMS option expiry lies outside the decoded surface")
    raw_upper = torch.searchsorted(nodes, query, right=False)
    upper = torch.clamp(raw_upper, max=len(nodes) - 1)
    before_first = raw_upper == 0
    lower = torch.clamp(raw_upper - 1, min=0)
    previous_time = torch.where(before_first, torch.zeros_like(query), nodes[lower])
    upper_weight = torch.clamp(
        (query - previous_time) / (nodes[upper] - previous_time),
        min=0.0,
        max=1.0,
    )
    upper_variance = surface.total_vols[:, upper].square()
    lower_variance = surface.total_vols[:, lower].square()
    lower_variance = torch.where(
        before_first[None], torch.zeros_like(lower_variance), lower_variance
    )
    total_variance = (
        (1.0 - upper_weight[None]) * lower_variance
        + upper_weight[None] * upper_variance
    )
    epsilon_vol = surface.total_vols[:, :1] / torch.sqrt(nodes[:1])
    nms_scale = torch.sqrt(
        (
            epsilon_vol.square() * float(NMS.EPS)
            + total_variance
        )
        / (1.0 + float(NMS.EPS))
    )
    strikes = torch.exp(normalized_moneyness[None] * nms_scale)
    if not bool(torch.all(torch.isfinite(strikes) & (strikes > 0.0))):
        raise FloatingPointError("NMS conversion produced invalid relative strikes")
    return strikes


def _simulate_path_batch(
    *,
    model: Any,
    network: FeedForward,
    book: _Book,
    initial_state: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    transaction_cost: torch.Tensor,
    action_l1_cap: float,
    action_ridge: float,
    num_paths: int,
    num_steps: int,
    generator: torch.Generator,
    retain_returns: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Simulate one antithetic path batch and evaluate the shared policy."""

    dtype = initial_state.dtype
    device = initial_state.device
    dimension = initial_state.shape[0]
    states = initial_state[None].expand(num_paths, -1).clone()
    parameters, surface = _decode(model, states)
    pnl = torch.zeros(num_paths, dtype=dtype, device=device)
    return_values: list[torch.Tensor] = []
    action_l1_sum = torch.zeros((), dtype=torch.float64, device=device)
    cap_active_sum = torch.zeros((), dtype=torch.float64, device=device)
    action_count = 0
    if book.relative_strikes is None:
        raise ValueError("one-day simulation currently requires fixed relative strikes")
    strikes = torch.tensor(book.relative_strikes, dtype=dtype, device=device)
    initial_days = torch.tensor(book.initial_days, dtype=dtype, device=device)
    final_days = torch.tensor(book.final_days, dtype=dtype, device=device)
    is_put = torch.tensor(book.is_put, dtype=torch.bool, device=device)
    equilibrium = model.equilibrium_mean.detach()
    for _ in range(num_steps):
        state_features = _features(model, states, parameters, surface).detach()
        raw_action = network((state_features - feature_mean) / feature_scale)
        action = l1_capped_action(raw_action, action_l1_cap)
        raw_l1 = torch.sum(torch.abs(raw_action), dim=1)
        action_l1_sum += torch.sum(torch.sum(torch.abs(action), dim=1).double())
        cap_active_sum += torch.sum((raw_l1 >= action_l1_cap).double())
        action_count += num_paths

        with torch.no_grad():
            half_state = torch.randn(
                (num_paths // 2, dimension),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            half_spot = torch.randn(
                num_paths // 2, dtype=dtype, device=device, generator=generator
            )
            state_normals = torch.cat((half_state, -half_state), dim=0)
            independent_spot = torch.cat((half_spot, -half_spot), dim=0)
            expected = equilibrium + (states - equilibrium) @ daily_transition
            next_states = expected + state_normals @ daily_factor.T
            gaussian_score = (
                state_normals @ copula_loading
                + copula_orthogonal_std * independent_spot
            ) / copula_total_std
            gross_spot = front_week_sanos_inverse_cdf(surface, gaussian_score)

            issue_calls = _price_calls(
                model,
                surface,
                days=initial_days,
                strikes=strikes[None].expand(num_paths, -1),
            )
            issue_values = torch.where(
                is_put[None], issue_calls - 1.0 + strikes[None], issue_calls
            )
            next_parameters, next_surface = _decode(model, next_states)
            final_normalized_calls = _price_calls(
                model,
                next_surface,
                days=final_days,
                strikes=strikes[None] / gross_spot[:, None],
            )
            final_calls = gross_spot[:, None] * final_normalized_calls
            final_values = torch.where(
                is_put[None],
                final_calls - gross_spot[:, None] + strikes[None],
                final_calls,
            )
            if not bool(torch.all(torch.isfinite(issue_values) & (issue_values > 0.0))):
                raise FloatingPointError("monthly OCE book has a nonpositive issue value")
            option_returns = final_values / issue_values - 1.0
            returns = torch.cat(((gross_spot - 1.0)[:, None], option_returns), dim=1)
            if not bool(torch.all(torch.isfinite(returns))):
                raise FloatingPointError("monthly OCE generated non-finite returns")
        trading_pnl = torch.sum(action * returns, dim=1)
        trading_cost = torch.sum(transaction_cost[None] * torch.abs(action), dim=1)
        ridge_cost = 0.5 * action_ridge * torch.sum(action.square(), dim=1)
        pnl = pnl + trading_pnl - trading_cost - ridge_cost
        if retain_returns:
            return_values.append(returns)
        states = next_states
        parameters = next_parameters
        surface = next_surface
    stacked_returns = torch.stack(return_values, dim=1) if retain_returns else None
    return pnl, stacked_returns, {
        "mean_action_l1": (action_l1_sum / action_count).detach(),
        "cap_active_fraction": (cap_active_sum / action_count).detach(),
    }


def _simulate_expiry_path_batch(
    *,
    model: Any,
    network: FeedForward,
    book: _Book,
    initial_state: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    transaction_cost: torch.Tensor,
    action_l1_cap: float,
    action_ridge: float,
    num_paths: int,
    num_steps: int,
    generator: torch.Generator,
    retain_returns: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Trade daily but hold every 5D/10D fixed-cash option to its payoff."""

    dtype = initial_state.dtype
    device = initial_state.device
    dimension = initial_state.shape[0]
    states = initial_state[None].expand(num_paths, -1).clone()
    parameters, surface = _decode(model, states)
    pnl = torch.zeros(num_paths, dtype=dtype, device=device)
    path_spot = torch.ones(num_paths, dtype=dtype, device=device)
    strikes = (
        torch.tensor(book.relative_strikes, dtype=dtype, device=device)
        if book.relative_strikes is not None
        else None
    )
    normalized_moneyness = (
        torch.tensor(book.normalized_moneyness, dtype=dtype, device=device)
        if book.normalized_moneyness is not None
        else None
    )
    initial_days = torch.tensor(book.initial_days, dtype=dtype, device=device)
    is_put = torch.tensor(book.is_put, dtype=torch.bool, device=device)
    horizons = torch.tensor(book.initial_days, dtype=torch.int64, device=device)
    equilibrium = model.equilibrium_mean.detach()

    entry_spots: list[torch.Tensor] = []
    entry_prices: list[torch.Tensor] = []
    entry_strikes: list[torch.Tensor] = []
    entry_option_actions: list[torch.Tensor] = []
    action_l1_sum = torch.zeros((), dtype=torch.float64, device=device)
    cap_active_sum = torch.zeros((), dtype=torch.float64, device=device)
    action_count = 0
    return_values = (
        torch.full(
            (num_paths, num_steps, len(book.labels)),
            torch.nan,
            dtype=dtype,
            device=device,
        )
        if retain_returns
        else None
    )

    # The final longest-expiry trade is opened at day num_steps-1 and settles
    # after its stated horizon, hence num_steps + max_horizon - 1 transitions.
    total_transitions = num_steps + int(torch.amax(horizons)) - 1
    for day in range(total_transitions):
        action: torch.Tensor | None = None
        if day < num_steps:
            state_features = _features(model, states, parameters, surface).detach()
            raw_action = network((state_features - feature_mean) / feature_scale)
            action = l1_capped_action(raw_action, action_l1_cap)
            raw_l1 = torch.sum(torch.abs(raw_action), dim=1)
            action_l1_sum += torch.sum(torch.sum(torch.abs(action), dim=1).double())
            cap_active_sum += torch.sum((raw_l1 >= action_l1_cap).double())
            action_count += num_paths

            with torch.no_grad():
                issue_strikes = (
                    strikes[None].expand(num_paths, -1)
                    if strikes is not None
                    else _nms_relative_strikes(
                        model,
                        surface,
                        days=initial_days,
                        normalized_moneyness=normalized_moneyness,
                    )
                )
                issue_calls = _price_calls(
                    model,
                    surface,
                    days=initial_days,
                    strikes=issue_strikes,
                )
                issue_values = torch.where(
                    is_put[None], issue_calls - 1.0 + issue_strikes, issue_calls
                )
                if not bool(
                    torch.all(torch.isfinite(issue_values) & (issue_values > 0.0))
                ):
                    raise FloatingPointError(
                        "monthly expiry OCE book has a nonpositive issue value"
                    )
            entry_spots.append(path_spot.detach())
            entry_prices.append(issue_values.detach())
            entry_strikes.append(issue_strikes.detach())
            entry_option_actions.append(action[:, 1:])
            trading_cost = torch.sum(
                transaction_cost[None] * torch.abs(action), dim=1
            )
            ridge_cost = 0.5 * action_ridge * torch.sum(action.square(), dim=1)
            pnl = pnl - trading_cost - ridge_cost

        with torch.no_grad():
            half_state = torch.randn(
                (num_paths // 2, dimension),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            half_spot = torch.randn(
                num_paths // 2, dtype=dtype, device=device, generator=generator
            )
            state_normals = torch.cat((half_state, -half_state), dim=0)
            independent_spot = torch.cat((half_spot, -half_spot), dim=0)
            expected = equilibrium + (states - equilibrium) @ daily_transition
            next_states = expected + state_normals @ daily_factor.T
            gaussian_score = (
                state_normals @ copula_loading
                + copula_orthogonal_std * independent_spot
            ) / copula_total_std
            gross_spot = front_week_sanos_inverse_cdf(surface, gaussian_score)
            next_path_spot = path_spot * gross_spot
            next_parameters, next_surface = _decode(model, next_states)

        if action is not None:
            spot_return = gross_spot - 1.0
            pnl = pnl + action[:, 0] * spot_return
            if return_values is not None:
                return_values[:, day, 0] = spot_return

        maturity_day = day + 1
        for horizon in book.expiry_days:
            entry_day = maturity_day - horizon
            if not 0 <= entry_day < num_steps:
                continue
            option_indices = torch.nonzero(horizons == horizon).squeeze(1)
            gross_horizon = next_path_spot / entry_spots[entry_day]
            selected_strikes = entry_strikes[entry_day][:, option_indices]
            selected_puts = is_put[option_indices]
            option_returns = _expiry_option_returns(
                gross_horizon=gross_horizon,
                issue_values=entry_prices[entry_day][:, option_indices],
                strikes=selected_strikes,
                is_put=selected_puts,
            )
            pnl = pnl + torch.sum(
                entry_option_actions[entry_day][:, option_indices] * option_returns,
                dim=1,
            )
            if return_values is not None:
                return_values[:, entry_day, option_indices + 1] = option_returns

        states = next_states
        parameters = next_parameters
        surface = next_surface
        path_spot = next_path_spot

    if return_values is not None and not bool(torch.all(torch.isfinite(return_values))):
        raise FloatingPointError("monthly expiry OCE left an invalid return coordinate")
    return pnl, return_values, {
        "mean_action_l1": (action_l1_sum / action_count).detach(),
        "cap_active_fraction": (cap_active_sum / action_count).detach(),
    }


def _learning_rate(arguments: argparse.Namespace, update: int, total_updates: int) -> float:
    if update < arguments.warmup_updates:
        return arguments.learning_rate * (update + 1) / max(arguments.warmup_updates, 1)
    progress = (update - arguments.warmup_updates) / max(
        total_updates - arguments.warmup_updates - 1, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return arguments.minimum_learning_rate + (
        arguments.learning_rate - arguments.minimum_learning_rate
    ) * cosine


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_numpy_save(value: torch.Tensor, path: Path) -> None:
    """Write one CPU tensor as an atomic, memory-mappable NumPy array."""

    if value.device.type != "cpu":
        raise ValueError("the fixed-path return matrix must be on CPU before saving")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file:
        np.save(file, value.numpy(), allow_pickle=False)
    os.replace(temporary, path)


def _network_payload(
    *,
    network: FeedForward,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    processed_paths: int,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    feature_blocks: Mapping[str, int],
    history: list[dict[str, float | int]],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": "dysanos-single-state-monthly-network-oce-v1",
        "phase": "training" if processed_paths < arguments.training_paths else "complete",
        "processed_training_paths": processed_paths,
        "network_state_dict": copy.deepcopy(network.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "generator_state": generator.get_state().cpu(),
        "feature_mean": feature_mean.detach().cpu(),
        "feature_scale": feature_scale.detach().cpu(),
        "feature_blocks": dict(feature_blocks),
        "history": history,
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
    }


def _audit_flat_returns(
    returns: torch.Tensor,
    weights: torch.Tensor,
    *,
    device: torch.device,
    chunk_pairs: int = 10_000,
) -> dict[str, Any]:
    """Compute a covariance-aware antithetic audit without a dense path copy."""

    if returns.ndim != 2 or len(returns) % 2 or weights.shape != (len(returns),):
        raise ValueError("monthly audit arrays are not aligned")
    paths, coordinates = returns.shape
    pairs = paths // 2
    mass = torch.mean(weights.double())
    weighted_sum = torch.zeros(coordinates, dtype=torch.float64, device=device)
    for first in range(0, paths, 2 * chunk_pairs):
        last = min(first + 2 * chunk_pairs, paths)
        values = returns[first:last].to(device=device, dtype=torch.float64)
        density = weights[first:last].to(device=device, dtype=torch.float64)
        weighted_sum += torch.sum(density[:, None] * values, dim=0)
    mean = weighted_sum / (paths * mass.to(device))

    pair_sum = torch.zeros(coordinates, dtype=torch.float64, device=device)
    pair_cross = torch.zeros((coordinates, coordinates), dtype=torch.float64, device=device)
    for first in range(0, pairs, chunk_pairs):
        last = min(first + chunk_pairs, pairs)
        left = returns[first:last].to(device=device, dtype=torch.float64)
        right = returns[pairs + first : pairs + last].to(device=device, dtype=torch.float64)
        left_weight = weights[first:last].to(device=device, dtype=torch.float64)
        right_weight = weights[pairs + first : pairs + last].to(
            device=device, dtype=torch.float64
        )
        pair_influence = 0.5 * (
            left_weight[:, None] * (left - mean[None])
            + right_weight[:, None] * (right - mean[None])
        ) / mass.to(device)
        pair_sum += torch.sum(pair_influence, dim=0)
        pair_cross += pair_influence.T @ pair_influence
    pair_mean = pair_sum / pairs
    covariance_of_mean = (
        pair_cross - pairs * torch.outer(pair_mean, pair_mean)
    ) / (pairs * (pairs - 1))
    covariance_of_mean = 0.5 * (covariance_of_mean + covariance_of_mean.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_of_mean)
    tolerance = (
        torch.finfo(torch.float64).eps
        * torch.amax(torch.abs(eigenvalues)).clamp_min(torch.finfo(torch.float64).tiny)
        * coordinates
    )
    retained = eigenvalues > tolerance
    projected = eigenvectors[:, retained].T @ mean
    statistic = torch.sum(projected.square() / eigenvalues[retained])
    standard_error = torch.sqrt(torch.clamp_min(torch.diag(covariance_of_mean), 0.0))
    z = mean / standard_error.clamp_min(torch.finfo(torch.float64).tiny)
    degrees = int(torch.sum(retained))
    return {
        "weighted_mean": mean.cpu().tolist(),
        "standard_error": standard_error.cpu().tolist(),
        "z": z.cpu().tolist(),
        "maximum_absolute_mean_bps": float(torch.amax(torch.abs(mean))) * 1.0e4,
        "maximum_absolute_z": float(torch.amax(torch.abs(z))),
        "root_mean_square_z": float(torch.sqrt(torch.mean(z.square()))),
        "joint_statistic": float(statistic),
        "joint_degrees_of_freedom": degrees,
        "joint_p_value": float(chi2.sf(float(statistic), degrees)),
        "density_mass": float(mass),
        "effective_sample_size": float(paths**2 / torch.sum(weights.double().square())),
        "effective_sample_fraction": float(paths / torch.sum(weights.double().square())),
        "minimum_weight": float(torch.amin(weights)),
        "maximum_weight": float(torch.amax(weights)),
        "zero_weight_count": int(torch.sum(weights == 0.0)),
    }


def run(arguments: argparse.Namespace) -> None:
    """Train on one million monthly paths and audit on a disjoint million."""

    _validate_arguments(arguments)
    book = _book(tuple(arguments.book_expiry_days))
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    payload = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = payload.get("dates")
    states_value = payload.get("states")
    spots_value = payload.get("spot_refs")
    if (
        not isinstance(dates, list)
        or not isinstance(states_value, torch.Tensor)
        or not isinstance(spots_value, torch.Tensor)
    ):
        raise ValueError("target history is missing dates, states, or spot references")
    state_index = len(dates) // 2 if arguments.state_index < 0 else arguments.state_index
    if not 0 <= state_index < len(dates):
        raise ValueError("state index is outside the historical population")
    historical_states = states_value.to(dtype=env.fdtype, device=env.device)
    initial_state = historical_states[state_index]
    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
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
    feature_mean, feature_scale, feature_blocks = _feature_statistics(
        model, historical_states
    )
    network_config = FeedForwardConfig(
        activation="SiLU",
        num_depth=arguments.hidden_layers,
        num_width=arguments.hidden_width,
    )
    network = FeedForward(env, feature_blocks["total"], len(book.labels), network_config)
    final_layer = network.I0 if len(network.In) == 0 else network.In[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        if final_layer.bias is not None:
            final_layer.bias.zero_()
    optimizer = torch.optim.Adam(network.parameters(), lr=arguments.learning_rate)
    transaction_cost = torch.tensor(
        (
            arguments.spot_cost_bps,
            *(arguments.option_cost_bps for _ in book.relative_strikes),
        ),
        dtype=env.fdtype,
        device=env.device,
    ) / 1.0e4
    training_generator = torch.Generator(device=env.device).manual_seed(arguments.seed)
    total_updates = math.ceil(arguments.training_paths / arguments.batch_size)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    history: list[dict[str, float | int]] = []
    processed = 0
    if arguments.resume:
        saved = torch.load(
            arguments.network_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(saved, Mapping)
            or saved.get("schema") != "dysanos-single-state-monthly-network-oce-v1"
            or saved.get("phase") not in {"training", "complete"}
        ):
            raise ValueError("monthly OCE resume checkpoint is invalid")
        saved_arguments = saved.get("arguments")
        if not isinstance(saved_arguments, Mapping):
            raise ValueError("monthly OCE resume checkpoint has no arguments")
        invariant_arguments = (
            "checkpoint",
            "target_history",
            "state_index",
            "num_steps",
            "book_expiry_days",
            "option_return_horizon",
            "training_paths",
            "validation_paths",
            "batch_size",
            "hidden_width",
            "hidden_layers",
            "learning_rate",
            "minimum_learning_rate",
            "warmup_updates",
            "action_l1_cap",
            "spot_cost_bps",
            "option_cost_bps",
            "action_ridge",
            "gradient_clip_norm",
            "seed",
            "copula_fit_before",
        )
        current_arguments = {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        }
        if any(
            saved_arguments.get(key) != current_arguments[key]
            for key in invariant_arguments
        ):
            raise ValueError("monthly OCE resume arguments differ from the checkpoint")
        network.load_state_dict(saved["network_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        training_generator.set_state(saved["generator_state"].cpu())
        processed = int(saved["processed_training_paths"])
        history = list(saved.get("history", []))
        if (
            processed < 0
            or processed > arguments.training_paths
            or (
                processed < arguments.training_paths
                and processed % arguments.batch_size != 0
            )
        ):
            raise ValueError("monthly OCE resume path count is incompatible")
        print(
            f"resumed monthly OCE at {processed}/{arguments.training_paths} paths",
            flush=True,
        )
    simulate_path_batch = (
        _simulate_expiry_path_batch
        if arguments.option_return_horizon == "to-expiry"
        else _simulate_path_batch
    )
    start_update = processed // arguments.batch_size
    for update in range(start_update, total_updates):
        batch_paths = min(arguments.batch_size, arguments.training_paths - processed)
        if batch_paths % 2:
            raise RuntimeError("the final monthly training batch is not antithetic")
        learning_rate = _learning_rate(arguments, update, total_updates)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        pnl, _, path_diagnostics = simulate_path_batch(
            model=model,
            network=network,
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
            action_l1_cap=arguments.action_l1_cap,
            action_ridge=arguments.action_ridge,
            num_paths=batch_paths,
            num_steps=arguments.num_steps,
            generator=training_generator,
            retain_returns=False,
        )
        loss = torch.logsumexp(-pnl, dim=0) - math.log(batch_paths)
        loss.backward()
        gradient_square = torch.zeros((), dtype=torch.float64, device=env.device)
        for parameter in network.parameters():
            if parameter.grad is not None:
                gradient_square += torch.sum(parameter.grad.double().square())
        gradient_rms = torch.sqrt(gradient_square / parameter_count)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(), arguments.gradient_clip_norm
        )
        optimizer.step()
        with torch.no_grad():
            log_weight = -pnl - torch.logsumexp(-pnl, dim=0) + math.log(batch_paths)
            weights = torch.exp(log_weight)
            ess_fraction = batch_paths / torch.sum(weights.square())
        processed += batch_paths
        if update % 10 == 0 or processed == arguments.training_paths:
            record = {
                "update": update + 1,
                "processed_paths": processed,
                "loss": float(loss.detach()),
                "learning_rate": learning_rate,
                "gradient_rms": float(gradient_rms),
                "gradient_norm_before_clip": float(gradient_norm),
                "batch_ess_fraction": float(ess_fraction),
                "mean_action_l1": float(path_diagnostics["mean_action_l1"]),
                "cap_active_fraction": float(path_diagnostics["cap_active_fraction"]),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        if (
            (update + 1) % arguments.checkpoint_every_updates == 0
            or processed == arguments.training_paths
        ):
            _atomic_torch_save(
                _network_payload(
                    network=network,
                    optimizer=optimizer,
                    generator=training_generator,
                    processed_paths=processed,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    feature_blocks=feature_blocks,
                    history=history,
                    arguments=arguments,
                ),
                arguments.network_checkpoint,
            )

    network.eval()
    validation_pnl = torch.empty(arguments.validation_paths, dtype=torch.float64)
    validation_generator = torch.Generator(device=env.device).manual_seed(arguments.seed + 1)
    pair_offset = 0
    with torch.inference_mode():
        for first in range(0, arguments.validation_paths, arguments.batch_size):
            batch_paths = min(arguments.batch_size, arguments.validation_paths - first)
            pnl, _, _ = simulate_path_batch(
                model=model,
                network=network,
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
                action_l1_cap=arguments.action_l1_cap,
                action_ridge=arguments.action_ridge,
                num_paths=batch_paths,
                num_steps=arguments.num_steps,
                generator=validation_generator,
                retain_returns=False,
            )
            half = batch_paths // 2
            validation_pnl[pair_offset : pair_offset + half] = pnl[:half].cpu().double()
            validation_pnl[
                arguments.validation_paths // 2
                + pair_offset : arguments.validation_paths // 2
                + pair_offset
                + half
            ] = pnl[half:].cpu().double()
            pair_offset += half
            if first % (100 * arguments.batch_size) == 0:
                print(f"validation pass 1: {first + batch_paths}/{arguments.validation_paths}", flush=True)
    log_normalizer = torch.logsumexp(-validation_pnl, dim=0) - math.log(
        arguments.validation_paths
    )
    validation_weights = torch.exp(-validation_pnl - log_normalizer)
    validation_ess = arguments.validation_paths**2 / torch.sum(validation_weights.square())

    flat_coordinates = arguments.num_steps * len(book.labels)
    validation_returns = torch.empty(
        (arguments.validation_paths, flat_coordinates), dtype=torch.float32
    )
    validation_generator.manual_seed(arguments.seed + 1)
    network.zero_grad(set_to_none=True)
    pair_offset = 0
    action_l1_weighted = 0.0
    cap_active_weighted = 0.0
    for first in range(0, arguments.validation_paths, arguments.batch_size):
        batch_paths = min(arguments.batch_size, arguments.validation_paths - first)
        pnl, returns, path_diagnostics = simulate_path_batch(
            model=model,
            network=network,
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
            action_l1_cap=arguments.action_l1_cap,
            action_ridge=arguments.action_ridge,
            num_paths=batch_paths,
            num_steps=arguments.num_steps,
            generator=validation_generator,
            retain_returns=True,
        )
        assert returns is not None
        half = batch_paths // 2
        first_weights = validation_weights[pair_offset : pair_offset + half]
        second_weights = validation_weights[
            arguments.validation_paths // 2
            + pair_offset : arguments.validation_paths // 2
            + pair_offset
            + half
        ]
        sequential_weights = torch.cat((first_weights, second_weights)).to(env.device)
        validation_gradient_piece = -torch.sum(sequential_weights.detach() * pnl.double())
        validation_gradient_piece = validation_gradient_piece / arguments.validation_paths
        validation_gradient_piece.backward()
        flat_returns = returns.detach().reshape(batch_paths, -1).cpu()
        validation_returns[pair_offset : pair_offset + half] = flat_returns[:half]
        validation_returns[
            arguments.validation_paths // 2
            + pair_offset : arguments.validation_paths // 2
            + pair_offset
            + half
        ] = flat_returns[half:]
        pair_offset += half
        action_l1_weighted += float(path_diagnostics["mean_action_l1"]) * batch_paths
        cap_active_weighted += float(path_diagnostics["cap_active_fraction"]) * batch_paths
        if first % (100 * arguments.batch_size) == 0:
            print(f"validation pass 2: {first + batch_paths}/{arguments.validation_paths}", flush=True)
    if arguments.fixed_path_returns_output is not None:
        _atomic_numpy_save(validation_returns, arguments.fixed_path_returns_output)
        print(
            f"wrote fixed-path gross returns to {arguments.fixed_path_returns_output}",
            flush=True,
        )
    validation_gradient_square = torch.zeros((), dtype=torch.float64, device=env.device)
    for parameter in network.parameters():
        if parameter.grad is not None:
            validation_gradient_square += torch.sum(parameter.grad.double().square())
    validation_gradient_rms = torch.sqrt(validation_gradient_square / parameter_count)

    p_audit = _audit_flat_returns(
        validation_returns,
        torch.ones(arguments.validation_paths, dtype=torch.float64),
        device=env.device,
    )
    q_audit = _audit_flat_returns(
        validation_returns,
        validation_weights,
        device=env.device,
    )
    cost = np.tile(
        np.asarray(
            (
                arguments.spot_cost_bps,
                *(arguments.option_cost_bps for _ in book.relative_strikes),
            )
        )
        / 1.0e4,
        arguments.num_steps,
    )
    q_mean = np.asarray(q_audit["weighted_mean"])
    q_standard_error = np.asarray(q_audit["standard_error"])
    cost_violation = np.maximum(np.abs(q_mean) - cost, 0.0)
    cost_violation_z = cost_violation / np.maximum(
        q_standard_error, np.finfo(np.float64).tiny
    )

    final_checkpoint = torch.load(
        arguments.network_checkpoint, map_location="cpu", weights_only=False
    )
    final_checkpoint["phase"] = "complete"
    final_checkpoint["validation_log_oce"] = float(log_normalizer)
    _atomic_torch_save(final_checkpoint, arguments.network_checkpoint)
    result = {
        "schema": "dysanos-single-state-monthly-network-oce-report-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "initial_state": "one historical Direct-200 snapshot",
            "path_horizon": (
                f"{arguments.num_steps} daily entry dates plus a "
                f"{max(book.expiry_days)}-day settlement tail"
                if arguments.option_return_horizon == "to-expiry"
                else f"{arguments.num_steps} daily steps"
            ),
            "policy_input": "(h_t, vec(sqrt(q_t)), vec(log(DLVsq_t-DLVsq_min)))",
            "feature_blocks": feature_blocks,
            "policy_output": list(book.labels),
            "rolling_contracts": (
                f"fresh fixed-cash {list(book.expiry_days)}D ITM/ATM options are "
                "opened at every "
                "entry date and held to their terminal payoff"
                if arguments.option_return_horizon == "to-expiry"
                else (
                    f"fresh fixed-cash {list(book.expiry_days)}D ITM/ATM options are "
                    "opened at every "
                    "step and revalued one day later at 4D and 9D"
                )
            ),
            "objective": (
                "log E_P exp(-sum_t[a_theta(x_t)'R_{t+1}-cost'|a_theta(x_t)|"
                "-action_ridge*||a_theta(x_t)||^2/2])"
            ),
            "density": "terminal dQ/dP=exp(-net path PnL)/E_P[exp(-net path PnL)]",
            "fixed_path_returns": (
                "optional gross-return matrix ordered as (path, time, instrument); "
                "it is an in-sample calibration set for any downstream convex fit"
            ),
            "action_geometry": (
                "seven signed outputs; equivalent to fourteen nonnegative long/short "
                "outputs under symmetric transaction costs"
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
        "state_index": state_index,
        "state_date": str(dates[state_index]),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history": str(arguments.target_history),
        "target_history_sha256": checkpoint_digest(arguments.target_history),
        "network_checkpoint": str(arguments.network_checkpoint),
        "network_checkpoint_sha256": checkpoint_digest(arguments.network_checkpoint),
        "network_parameter_count": parameter_count,
        "transition_diagnostics": transition_diagnostics,
        "spot_copula_fit": copula_fit.to_dict(),
        "spot_copula_diagnostics": copula_diagnostics,
        "training_history": history,
        "validation": {
            "log_oce": float(log_normalizer),
            "effective_sample_size": float(validation_ess),
            "effective_sample_fraction": float(validation_ess / arguments.validation_paths),
            "minimum_weight": float(torch.amin(validation_weights)),
            "maximum_weight": float(torch.amax(validation_weights)),
            "zero_weight_count": int(torch.sum(validation_weights == 0.0)),
            "gradient_rms": float(validation_gradient_rms),
            "mean_action_l1": action_l1_weighted / arguments.validation_paths,
            "cap_active_fraction": cap_active_weighted / arguments.validation_paths,
            "under_p": p_audit,
            "under_q": q_audit,
            "unconditional_cost_band": {
                "maximum_violation_bps": float(np.max(cost_violation) * 1.0e4),
                "root_mean_square_violation_bps": float(
                    np.sqrt(np.mean(cost_violation**2)) * 1.0e4
                ),
                "maximum_violation_z": float(np.max(cost_violation_z)),
                "fraction_outside_band": float(np.mean(cost_violation > 0.0)),
                "fraction_outside_band_above_1p96_se": float(
                    np.mean(cost_violation_z > 1.96)
                ),
            },
        },
    }
    write_json_atomic(result, arguments.output)
    print(
        json.dumps(
            {
                "state_date": result["state_date"],
                "training_last": history[-1],
                "validation": result["validation"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
