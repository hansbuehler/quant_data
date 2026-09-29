"""Audit a convex empirical Q with a shared expiry-slice OCE policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.util import is_filename
from cdxcore.version import version
from filelock import FileLock
from scipy.special import logsumexp

from quantdata.dysanos.flex.front_spot_copula import (
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.network_oce import l1_capped_action
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _Book,
    _expiry_option_returns,
    _nms_book,
    _nms_relative_strikes,
    _price_calls,
)
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_inverse_cdf
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

_EXPIRY_DAYS = (5, 10, 15, 20)
_NORMALIZED_MONEYNESS = (
    -1.5,
    -1.3,
    -1.1,
    -0.9,
    -0.7,
    -0.5,
    -0.3,
    -0.1,
    0.1,
    0.3,
    0.5,
)


@version("1.0", dependencies=[FeedForwardConfig])
@dataclass(frozen=True, kw_only=True)
class SharedExpiryOceConfig:
    """Architecture and action geometry of the conditional OCE policy."""

    label: str
    group: str = "flex/convex-q-conditional-oce"
    user_version: str = "1.0"
    state_size: int = 200
    num_expiries: int = 4
    num_strikes: int = 11
    maximum_action_l1: float = 16.0
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=128,
    )

    def __post_init__(self) -> None:
        verify_inp(
            is_filename(self.label),
            lambda: f"policy label must be a filename, found {self.label!r}",
        )
        verify_inp(self.state_size > 0, lambda: "policy state size must be positive")
        verify_inp(
            self.num_expiries > 0 and self.num_strikes > 0,
            lambda: "policy expiry and strike counts must be positive",
        )
        verify_inp(
            math.isfinite(self.maximum_action_l1) and self.maximum_action_l1 > 0.0,
            lambda: "policy L1 cap must be finite-positive",
        )


def assemble_expiry_slice_actions(
    raw_heads: torch.Tensor,
    *,
    maximum_l1: float,
) -> torch.Tensor:
    """Aggregate four spot heads and retain every expiry-specific option head."""

    if raw_heads.ndim != 3 or raw_heads.shape[-1] < 2:
        raise ValueError("raw expiry heads must have shape (paths,expiries,1+strikes)")
    spot = torch.sum(raw_heads[..., :1], dim=1)
    options = raw_heads[..., 1:].reshape(len(raw_heads), -1)
    return l1_capped_action(torch.cat((spot, options), dim=1), maximum_l1)


class SharedExpiryOcePolicy(BaseModel):
    """One shared 11+1 network applied separately to each expiry slice."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: SharedExpiryOceConfig,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
        log_price_mean: torch.Tensor,
        log_price_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        if state_mean.shape != (config.state_size,) or state_scale.shape != (
            config.state_size,
        ):
            raise ValueError("policy state normalization has the wrong shape")
        if log_price_mean.shape != (config.num_strikes,) or log_price_scale.shape != (
            config.num_strikes,
        ):
            raise ValueError("policy price normalization has the wrong shape")
        self.config = config
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_scale", state_scale.detach().clone())
        self.register_buffer("log_price_mean", log_price_mean.detach().clone())
        self.register_buffer("log_price_scale", log_price_scale.detach().clone())
        self.network = FeedForward(
            env,
            config.state_size + config.num_strikes,
            config.num_strikes + 1,
            config.network,
        )
        final_layer = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            if final_layer.bias is not None:
                final_layer.bias.zero_()

    def forward(
        self,
        states: torch.Tensor,
        option_prices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_prices = (
            len(states),
            self.config.num_expiries,
            self.config.num_strikes,
        )
        if states.shape != (len(states), self.config.state_size):
            raise ValueError("policy hidden state has the wrong shape")
        if option_prices.shape != expected_prices:
            raise ValueError(
                f"policy option prices must have shape {expected_prices}, "
                f"found {tuple(option_prices.shape)}"
            )
        safe_prices = torch.clamp_min(option_prices, torch.finfo(option_prices.dtype).tiny)
        normalized_state = (states - self.state_mean) / self.state_scale
        normalized_prices = (torch.log(safe_prices) - self.log_price_mean) / (
            self.log_price_scale
        )
        state_block = normalized_state[:, None, :].expand(
            -1, self.config.num_expiries, -1
        )
        features = torch.cat((state_block, normalized_prices), dim=-1)
        raw_heads = self.network(features)
        action = assemble_expiry_slice_actions(
            raw_heads,
            maximum_l1=self.config.maximum_action_l1,
        )
        return action, raw_heads


@dataclass(frozen=True)
class _SimulationDiagnostics:
    mean_action_l1: torch.Tensor
    maximum_absolute_action: torch.Tensor
    cap_active_fraction: torch.Tensor


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


def _simulate_expiry_policy_batch(
    *,
    model: Any,
    policy: SharedExpiryOcePolicy,
    book: _Book,
    initial_state: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    transaction_cost: torch.Tensor,
    num_paths: int,
    num_steps: int,
    generator: torch.Generator,
    retain_returns: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, _SimulationDiagnostics]:
    """Reproduce the fixed paths and evaluate the conditional expiry-slice policy."""

    dtype = initial_state.dtype
    device = initial_state.device
    dimension = initial_state.shape[0]
    num_expiries = len(book.expiry_days)
    if len(book.labels) != 1 + num_expiries * policy.config.num_strikes:
        raise ValueError("book and policy action geometries differ")
    states = initial_state[None].expand(num_paths, -1).clone()
    parameters, surface = _decode(model, states)
    pnl = torch.zeros(num_paths, dtype=dtype, device=device)
    path_spot = torch.ones(num_paths, dtype=dtype, device=device)
    normalized_moneyness = torch.tensor(
        book.normalized_moneyness, dtype=dtype, device=device
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
    maximum_absolute_action = torch.zeros((), dtype=torch.float64, device=device)
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

    total_transitions = num_steps + int(torch.amax(horizons)) - 1
    for day in range(total_transitions):
        action: torch.Tensor | None = None
        if day < num_steps:
            with torch.no_grad():
                issue_strikes = _nms_relative_strikes(
                    model,
                    surface,
                    days=initial_days,
                    normalized_moneyness=normalized_moneyness,
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
                if not bool(torch.all(torch.isfinite(issue_values) & (issue_values > 0.0))):
                    raise FloatingPointError("conditional OCE has a nonpositive option price")
            prices_by_expiry = issue_values.reshape(
                num_paths, num_expiries, policy.config.num_strikes
            )
            action, raw_heads = policy(states.detach(), prices_by_expiry.detach())
            raw_action = torch.cat(
                (
                    torch.sum(raw_heads[..., :1], dim=1),
                    raw_heads[..., 1:].reshape(num_paths, -1),
                ),
                dim=1,
            )
            raw_l1 = torch.sum(torch.abs(raw_action), dim=1)
            action_l1_sum += torch.sum(torch.sum(torch.abs(action), dim=1).double())
            maximum_absolute_action = torch.maximum(
                maximum_absolute_action,
                torch.amax(torch.abs(action)).double(),
            )
            cap_active_sum += torch.sum(
                (raw_l1 >= policy.config.maximum_action_l1).double()
            )
            action_count += num_paths

            entry_spots.append(path_spot.detach())
            entry_prices.append(issue_values.detach())
            entry_strikes.append(issue_strikes.detach())
            entry_option_actions.append(action[:, 1:])
            pnl = pnl - torch.sum(transaction_cost[None] * torch.abs(action), dim=1)

        with torch.no_grad():
            half_state = torch.randn(
                (num_paths // 2, dimension),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            half_spot = torch.randn(
                num_paths // 2,
                dtype=dtype,
                device=device,
                generator=generator,
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
            option_returns = _expiry_option_returns(
                gross_horizon=gross_horizon,
                issue_values=entry_prices[entry_day][:, option_indices],
                strikes=selected_strikes,
                is_put=is_put[option_indices],
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
        raise FloatingPointError("conditional OCE left an invalid return coordinate")
    return pnl, return_values, _SimulationDiagnostics(
        mean_action_l1=(action_l1_sum / action_count).detach(),
        maximum_absolute_action=maximum_absolute_action.detach(),
        cap_active_fraction=(cap_active_sum / action_count).detach(),
    )


def weighted_log_oce(pnl: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    """Return ``log E_Q exp(-PnL)`` for positive empirical density weights."""

    if pnl.ndim != 1 or density.shape != pnl.shape or len(pnl) == 0:
        raise ValueError("PnL and density must be matching nonempty vectors")
    if not bool(torch.all(torch.isfinite(pnl))):
        raise ValueError("PnL must be finite")
    if not bool(torch.all(torch.isfinite(density) & (density > 0.0))):
        raise ValueError("density weights must be finite-positive")
    log_probability = torch.log(density) - torch.logsumexp(torch.log(density), dim=0)
    return torch.logsumexp(log_probability - pnl, dim=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--returns", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--convex-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=1_000_000)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument(
        "--fixed-feature-cache",
        type=Path,
        default=None,
        help="Optional cache of exact hidden states and current option prices.",
    )
    parser.add_argument("--optimization-batch-size", type=int, default=32_768)
    parser.add_argument("--prepare-features-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--action-l1-cap", type=float, default=16.0)
    parser.add_argument("--spot-cost-bps", type=float, default=1.0)
    parser.add_argument("--option-cost-bps", type=float, default=10.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=100.0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2_026_093_0)
    parser.add_argument("--alignment-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--copula-fit-before",
        type=date.fromisoformat,
        default=date(2025, 1, 1),
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    integer_values = (
        arguments.num_paths,
        arguments.num_steps,
        arguments.batch_size,
        arguments.optimization_batch_size,
        arguments.epochs,
        arguments.hidden_width,
        arguments.hidden_layers,
        arguments.checkpoint_every_updates,
    )
    if any(value <= 0 for value in integer_values):
        raise ValueError("conditional OCE integer controls must be positive")
    if arguments.num_paths % 2 or arguments.batch_size % 2:
        raise ValueError("conditional OCE paths and batches must be even")
    if arguments.optimization_batch_size % 2:
        raise ValueError("conditional OCE optimization batches must be even")
    if arguments.prepare_features_only and arguments.fixed_feature_cache is None:
        raise ValueError("feature-only mode requires --fixed-feature-cache")
    if arguments.num_steps != 25:
        raise ValueError("the fitted convex density has exactly 25 entry dates")
    positive_values = (
        arguments.learning_rate,
        arguments.minimum_learning_rate,
        arguments.action_l1_cap,
        arguments.spot_cost_bps,
        arguments.option_cost_bps,
        arguments.gradient_clip_norm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
        raise ValueError("conditional OCE scalar controls must be finite-positive")
    if arguments.minimum_learning_rate > arguments.learning_rate:
        raise ValueError("minimum learning rate exceeds the peak learning rate")
    if arguments.warmup_updates < 0:
        raise ValueError("warmup updates cannot be negative")


def _learning_rate(arguments: argparse.Namespace, update: int, total_updates: int) -> float:
    if update < arguments.warmup_updates:
        return arguments.learning_rate * (update + 1) / max(arguments.warmup_updates, 1)
    progress = (update - arguments.warmup_updates) / max(
        total_updates - arguments.warmup_updates - 1,
        1,
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


def _normalization(
    model: Any,
    historical_states: torch.Tensor,
    book: _Book,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, surface = _decode(model, historical_states)
    dtype = historical_states.dtype
    device = historical_states.device
    initial_days = torch.tensor(book.initial_days, dtype=dtype, device=device)
    normalized_moneyness = torch.tensor(
        book.normalized_moneyness, dtype=dtype, device=device
    )
    is_put = torch.tensor(book.is_put, dtype=torch.bool, device=device)
    strikes = _nms_relative_strikes(
        model,
        surface,
        days=initial_days,
        normalized_moneyness=normalized_moneyness,
    )
    calls = _price_calls(model, surface, days=initial_days, strikes=strikes)
    prices = torch.where(is_put[None], calls - 1.0 + strikes, calls)
    log_prices = torch.log(torch.clamp_min(prices, torch.finfo(dtype).tiny)).reshape(
        len(historical_states), len(book.expiry_days), len(_NORMALIZED_MONEYNESS)
    )
    state_mean = torch.mean(historical_states, dim=0)
    state_std = torch.std(historical_states, dim=0, correction=0)
    state_scale = torch.where(state_std > 1.0e-5, state_std, torch.ones_like(state_std))
    log_price_mean = torch.mean(log_prices, dim=(0, 1))
    log_price_std = torch.std(log_prices.reshape(-1, len(_NORMALIZED_MONEYNESS)), dim=0, correction=0)
    log_price_scale = torch.where(
        log_price_std > 1.0e-5,
        log_price_std,
        torch.ones_like(log_price_std),
    )
    return state_mean, state_scale, log_price_mean, log_price_scale


def _gradient_statistics(model: BaseModel) -> tuple[float, float, int]:
    square = torch.zeros((), dtype=torch.float64, device=model.device)
    parameters = 0
    maximum = torch.zeros((), dtype=torch.float64, device=model.device)
    for parameter in model.parameters():
        parameters += parameter.numel()
        if parameter.grad is not None:
            gradient = parameter.grad.detach().double()
            square += torch.sum(gradient.square())
            maximum = torch.maximum(maximum, torch.amax(torch.abs(gradient)))
    rms = torch.sqrt(square / max(parameters, 1))
    return float(rms.cpu()), float(maximum.cpu()), parameters


def _setup(arguments: argparse.Namespace) -> dict[str, Any]:
    book = _nms_book(
        _EXPIRY_DAYS,
        normalized_moneyness=_NORMALIZED_MONEYNESS,
    )
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
    normalizers = _normalization(model, historical_states, book)
    policy_config = SharedExpiryOceConfig(
        label="convex-q-shared-expiry-11plus1-v1",
        state_size=initial_state.shape[0],
        maximum_action_l1=arguments.action_l1_cap,
        network=FeedForwardConfig(
            activation="SiLU",
            num_depth=arguments.hidden_layers,
            num_width=arguments.hidden_width,
        ),
    )
    policy = SharedExpiryOcePolicy(
        env,
        config=policy_config,
        state_mean=normalizers[0],
        state_scale=normalizers[1],
        log_price_mean=normalizers[2],
        log_price_scale=normalizers[3],
    )
    transaction_cost = torch.tensor(
        (
            arguments.spot_cost_bps,
            *(arguments.option_cost_bps for _ in range(len(book.labels) - 1)),
        ),
        dtype=env.fdtype,
        device=env.device,
    ) / 1.0e4
    return {
        "book": book,
        "env": env,
        "model": model,
        "policy": policy,
        "policy_config": policy_config,
        "initial_state": initial_state,
        "dates": dates,
        "state_index": state_index,
        "daily_transition": daily_transition,
        "daily_factor": daily_factor,
        "transition_diagnostics": transition_diagnostics,
        "copula_fit": copula_fit,
        "copula_diagnostics": copula_diagnostics,
        "copula_loading": copula_loading,
        "transaction_cost": transaction_cost,
    }


def _simulate(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    *,
    num_paths: int,
    generator: torch.Generator,
    retain_returns: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, _SimulationDiagnostics]:
    return _simulate_expiry_policy_batch(
        model=setup["model"],
        policy=setup["policy"],
        book=setup["book"],
        initial_state=setup["initial_state"],
        daily_transition=setup["daily_transition"],
        daily_factor=setup["daily_factor"],
        copula_loading=setup["copula_loading"],
        copula_orthogonal_std=setup["copula_fit"].orthogonal_std,
        copula_total_std=setup["copula_fit"].total_std,
        transaction_cost=setup["transaction_cost"],
        num_paths=num_paths,
        num_steps=arguments.num_steps,
        generator=generator,
        retain_returns=retain_returns,
    )


def _feature_cache_paths(directory: Path) -> tuple[Path, Path, Path]:
    return (
        directory / "policy-states.npy",
        directory / "policy-option-prices.npy",
        directory / "feature-cache.json",
    )


def _prepare_feature_cache(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    *,
    directory: Path,
) -> None:
    """Cache exact fixed-path policy inputs once, preserving CUDA RNG ordering."""

    state_path, price_path, metadata_path = _feature_cache_paths(directory)
    expected_metadata = {
        "schema": "dysanos-convex-q-conditional-oce-feature-cache-v1",
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history_sha256": checkpoint_digest(arguments.target_history),
        "state_index": setup["state_index"],
        "num_paths": arguments.num_paths,
        "num_steps": arguments.num_steps,
        "generation_batch_size": arguments.batch_size,
        "seed": arguments.seed,
        "state_size": setup["policy_config"].state_size,
        "num_expiries": setup["policy_config"].num_expiries,
        "num_strikes": setup["policy_config"].num_strikes,
    }
    if metadata_path.exists() and state_path.exists() and price_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in expected_metadata.items()):
            states = np.load(state_path, mmap_mode="r", allow_pickle=False)
            prices = np.load(price_path, mmap_mode="r", allow_pickle=False)
            if states.shape == (
                arguments.num_paths,
                arguments.num_steps,
                setup["policy_config"].state_size,
            ) and prices.shape == (
                arguments.num_paths,
                arguments.num_steps,
                setup["policy_config"].num_expiries,
                setup["policy_config"].num_strikes,
            ):
                print(f"reusing fixed feature cache {directory}", flush=True)
                return
        raise ValueError(f"fixed feature cache identity mismatch in {directory}")

    directory.mkdir(parents=True, exist_ok=True)
    temporary_state = state_path.with_name(f".{state_path.name}.tmp")
    temporary_price = price_path.with_name(f".{price_path.name}.tmp")
    state_map = np.lib.format.open_memmap(
        temporary_state,
        mode="w+",
        dtype=np.float32,
        shape=(
            arguments.num_paths,
            arguments.num_steps,
            setup["policy_config"].state_size,
        ),
    )
    price_map = np.lib.format.open_memmap(
        temporary_price,
        mode="w+",
        dtype=np.float32,
        shape=(
            arguments.num_paths,
            arguments.num_steps,
            setup["policy_config"].num_expiries,
            setup["policy_config"].num_strikes,
        ),
    )
    model = setup["model"]
    book = setup["book"]
    dtype = setup["env"].fdtype
    device = setup["env"].device
    dimension = setup["initial_state"].shape[0]
    equilibrium = model.equilibrium_mean.detach()
    initial_days = torch.tensor(book.initial_days, dtype=dtype, device=device)
    normalized_moneyness = torch.tensor(
        book.normalized_moneyness, dtype=dtype, device=device
    )
    is_put = torch.tensor(book.is_put, dtype=torch.bool, device=device)
    total_transitions = arguments.num_steps + max(book.expiry_days) - 1
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    started = time.monotonic()
    with torch.inference_mode():
        for first in range(0, arguments.num_paths, arguments.batch_size):
            last = min(first + arguments.batch_size, arguments.num_paths)
            count = last - first
            if count % 2:
                raise RuntimeError("final fixed feature batch is not antithetic")
            states = setup["initial_state"][None].expand(count, -1).clone()
            for day in range(total_transitions):
                if day < arguments.num_steps:
                    _, surface = _decode(model, states)
                    strikes = _nms_relative_strikes(
                        model,
                        surface,
                        days=initial_days,
                        normalized_moneyness=normalized_moneyness,
                    )
                    calls = _price_calls(
                        model,
                        surface,
                        days=initial_days,
                        strikes=strikes,
                    )
                    values = torch.where(is_put[None], calls - 1.0 + strikes, calls)
                    if not bool(torch.all(torch.isfinite(values) & (values > 0.0))):
                        raise FloatingPointError("feature cache has a nonpositive option price")
                    state_map[first:last, day] = states.cpu().numpy()
                    price_map[first:last, day] = values.reshape(
                        count,
                        setup["policy_config"].num_expiries,
                        setup["policy_config"].num_strikes,
                    ).cpu().numpy()
                half_state = torch.randn(
                    (count // 2, dimension),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                # Preserve the exact original state/spot random-stream interleaving.
                torch.randn(count // 2, dtype=dtype, device=device, generator=generator)
                state_normals = torch.cat((half_state, -half_state), dim=0)
                states = equilibrium + (states - equilibrium) @ setup["daily_transition"]
                states = states + state_normals @ setup["daily_factor"].T
            if first == 0 or last == arguments.num_paths or first % (
                25 * arguments.batch_size
            ) == 0:
                rate = last / max(time.monotonic() - started, 1.0e-12)
                print(
                    f"cached policy inputs {last}/{arguments.num_paths} at {rate:.1f} paths/s",
                    flush=True,
                )
    state_map.flush()
    price_map.flush()
    del state_map, price_map
    os.replace(temporary_state, state_path)
    os.replace(temporary_price, price_path)
    write_json_atomic(
        {
            **expected_metadata,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "state_shape": [
                arguments.num_paths,
                arguments.num_steps,
                setup["policy_config"].state_size,
            ],
            "price_shape": [
                arguments.num_paths,
                arguments.num_steps,
                setup["policy_config"].num_expiries,
                setup["policy_config"].num_strikes,
            ],
        },
        metadata_path,
    )


def _fixed_feature_pnl(
    policy: SharedExpiryOcePolicy,
    *,
    states: torch.Tensor,
    option_prices: torch.Tensor,
    returns: torch.Tensor,
    transaction_cost: torch.Tensor,
) -> tuple[torch.Tensor, _SimulationDiagnostics]:
    """Evaluate a policy on cached states, prices, and fixed returns."""

    if states.ndim != 3 or option_prices.ndim != 4 or returns.ndim != 3:
        raise ValueError("cached policy tensors have invalid ranks")
    if states.shape[:2] != option_prices.shape[:2] or states.shape[:2] != returns.shape[:2]:
        raise ValueError("cached policy tensors have incompatible path/time axes")
    if returns.shape[2] != 1 + policy.config.num_expiries * policy.config.num_strikes:
        raise ValueError("cached returns have the wrong instrument dimension")
    pnl = torch.zeros(len(states), dtype=states.dtype, device=states.device)
    action_l1_sum = torch.zeros((), dtype=torch.float64, device=states.device)
    maximum_action = torch.zeros((), dtype=torch.float64, device=states.device)
    cap_active_sum = torch.zeros((), dtype=torch.float64, device=states.device)
    for day in range(states.shape[1]):
        action, raw_heads = policy(states[:, day], option_prices[:, day])
        raw_action = torch.cat(
            (
                torch.sum(raw_heads[..., :1], dim=1),
                raw_heads[..., 1:].reshape(len(states), -1),
            ),
            dim=1,
        )
        pnl = pnl + torch.sum(action * returns[:, day], dim=1)
        pnl = pnl - torch.sum(transaction_cost[None] * torch.abs(action), dim=1)
        action_l1_sum += torch.sum(torch.sum(torch.abs(action), dim=1).double())
        maximum_action = torch.maximum(
            maximum_action,
            torch.amax(torch.abs(action)).double(),
        )
        cap_active_sum += torch.sum(
            (
                torch.sum(torch.abs(raw_action), dim=1)
                >= policy.config.maximum_action_l1
            ).double()
        )
    observations = len(states) * states.shape[1]
    return pnl, _SimulationDiagnostics(
        mean_action_l1=(action_l1_sum / observations).detach(),
        maximum_absolute_action=maximum_action.detach(),
        cap_active_fraction=(cap_active_sum / observations).detach(),
    )


def _cached_batch(
    *,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    first: int,
    last: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.array(states[first:last], copy=True), dtype=dtype, device=device),
        torch.as_tensor(np.array(prices[first:last], copy=True), dtype=dtype, device=device),
        torch.as_tensor(np.array(returns[first:last], copy=True), dtype=dtype, device=device),
    )


def _alignment_check(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    returns: np.ndarray,
) -> float:
    paths = min(arguments.batch_size, arguments.num_paths)
    generator = torch.Generator(device=setup["env"].device).manual_seed(arguments.seed)
    with torch.no_grad():
        _, generated, _ = _simulate(
            setup,
            arguments,
            num_paths=paths,
            generator=generator,
            retain_returns=True,
        )
    assert generated is not None
    reference = torch.as_tensor(
        np.array(returns[:paths], copy=True),
        dtype=generated.dtype,
        device=generated.device,
    )
    maximum = float(torch.amax(torch.abs(generated - reference)).cpu())
    if maximum > 2.0e-6:
        raise RuntimeError(
            "regenerated paths do not match the convex calibration population: "
            f"maximum return difference {maximum:.6g}"
        )
    return maximum


def _checkpoint_payload(
    *,
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    epoch: int,
    path_offset: int,
    updates: int,
    history: list[dict[str, float | int]],
    phase: str,
) -> dict[str, Any]:
    return {
        "schema": "dysanos-convex-q-conditional-oce-v1",
        "phase": phase,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
        "policy_config": asdict(setup["policy_config"]),
        "policy_state_dict": setup["policy"].state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "generator_state": generator.get_state().cpu(),
        "epoch": epoch,
        "path_offset": path_offset,
        "updates": updates,
        "history": history,
    }


def _load_checkpoint(
    path: Path,
    *,
    setup: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> tuple[int, int, int, list[dict[str, float | int]], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "dysanos-convex-q-conditional-oce-v1"
    ):
        raise ValueError("conditional OCE checkpoint identity is invalid")
    setup["policy"].load_state_dict(payload["policy_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    generator.set_state(payload["generator_state"])
    return (
        int(payload["epoch"]),
        int(payload["path_offset"]),
        int(payload["updates"]),
        list(payload.get("history", [])),
        str(payload["phase"]),
    )


def _audit_cached(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    feature_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Compute the exact fixed-population OCE and gradient from cached inputs."""

    state_path, price_path, _ = _feature_cache_paths(feature_directory)
    states = np.load(state_path, mmap_mode="r", allow_pickle=False)
    prices = np.load(price_path, mmap_mode="r", allow_pickle=False)
    policy: SharedExpiryOcePolicy = setup["policy"]
    policy.eval()
    pnl_path = output_directory / "network-pnl.npy"
    temporary_pnl = pnl_path.with_name(f".{pnl_path.name}.tmp")
    pnl_memmap = np.lib.format.open_memmap(
        temporary_pnl,
        mode="w+",
        dtype=np.float64,
        shape=(arguments.num_paths,),
    )
    action_l1_sum = 0.0
    cap_sum = 0.0
    maximum_action = 0.0
    started = time.monotonic()
    batch_size = arguments.optimization_batch_size
    with torch.no_grad():
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
            pnl, diagnostics = _fixed_feature_pnl(
                policy,
                states=batch[0],
                option_prices=batch[1],
                returns=batch[2],
                transaction_cost=setup["transaction_cost"],
            )
            pnl_memmap[first:last] = pnl.cpu().double().numpy()
            action_l1_sum += float(diagnostics.mean_action_l1) * (last - first)
            cap_sum += float(diagnostics.cap_active_fraction) * (last - first)
            maximum_action = max(maximum_action, float(diagnostics.maximum_absolute_action))
            if first == 0 or last == arguments.num_paths:
                print(
                    f"cached audit pass 1: {last}/{arguments.num_paths} "
                    f"at {last / max(time.monotonic() - started, 1.0e-12):.1f} paths/s",
                    flush=True,
                )
    pnl_memmap.flush()
    del pnl_memmap
    os.replace(temporary_pnl, pnl_path)
    pnl = np.load(pnl_path, mmap_mode="r", allow_pickle=False)
    selected_weights = np.array(
        weights[: arguments.num_paths], dtype=np.float64, copy=True
    )
    log_terms = np.log(selected_weights) - np.asarray(pnl)
    denominator_log = logsumexp(log_terms)
    log_normalizer = float(denominator_log - math.log(np.sum(selected_weights)))
    combined_density = np.exp(
        log_terms - denominator_log + math.log(arguments.num_paths)
    )
    expected_net_pnl = float(
        np.sum(selected_weights * np.asarray(pnl), dtype=np.float64)
        / np.sum(selected_weights)
    )
    pnl_second_moment = float(
        np.sum(selected_weights * np.square(np.asarray(pnl)), dtype=np.float64)
        / np.sum(selected_weights)
    )

    policy.zero_grad(set_to_none=True)
    started = time.monotonic()
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
        pnl_batch, _ = _fixed_feature_pnl(
            policy,
            states=batch[0],
            option_prices=batch[1],
            returns=batch[2],
            transaction_cost=setup["transaction_cost"],
        )
        probability = torch.as_tensor(
            np.exp(
                np.log(selected_weights[first:last])
                - np.asarray(pnl[first:last])
                - denominator_log
            ),
            dtype=pnl_batch.dtype,
            device=pnl_batch.device,
        )
        (-torch.sum(probability.detach() * pnl_batch)).backward()
        if first == 0 or last == arguments.num_paths:
            print(
                f"cached audit pass 2: {last}/{arguments.num_paths} "
                f"at {last / max(time.monotonic() - started, 1.0e-12):.1f} paths/s",
                flush=True,
            )
    gradient_rms, gradient_maximum, parameter_count = _gradient_statistics(policy)
    parameter_square = sum(
        float(torch.sum(parameter.detach().double().square()).cpu())
        for parameter in policy.parameters()
    )
    return {
        "log_loss": log_normalizer,
        "oce_value": -log_normalizer,
        "zero_policy_oce_value": 0.0,
        "effective_sample_size_after_adversary": float(
            arguments.num_paths**2
            / np.sum(np.square(combined_density), dtype=np.float64)
        ),
        "expected_net_pnl_under_q": expected_net_pnl,
        "net_pnl_standard_deviation_under_q": math.sqrt(
            max(pnl_second_moment - expected_net_pnl**2, 0.0)
        ),
        "mean_action_l1_per_state": action_l1_sum / arguments.num_paths,
        "maximum_absolute_action": maximum_action,
        "cap_active_fraction": cap_sum / arguments.num_paths,
        "gradient_rms": gradient_rms,
        "gradient_maximum_absolute": gradient_maximum,
        "network_parameter_l2": math.sqrt(parameter_square),
        "network_parameter_count": parameter_count,
    }


def _audit(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    weights: np.ndarray,
    *,
    output_directory: Path,
) -> dict[str, Any]:
    policy: SharedExpiryOcePolicy = setup["policy"]
    policy.eval()
    pnl_path = output_directory / "network-pnl.npy"
    pnl_memmap = np.lib.format.open_memmap(
        pnl_path.with_name(f".{pnl_path.name}.tmp"),
        mode="w+",
        dtype=np.float64,
        shape=(arguments.num_paths,),
    )
    generator = torch.Generator(device=setup["env"].device).manual_seed(arguments.seed)
    action_l1_sum = 0.0
    cap_sum = 0.0
    maximum_action = 0.0
    started = time.monotonic()
    with torch.no_grad():
        for first in range(0, arguments.num_paths, arguments.batch_size):
            last = min(first + arguments.batch_size, arguments.num_paths)
            pnl, _, diagnostics = _simulate(
                setup,
                arguments,
                num_paths=last - first,
                generator=generator,
                retain_returns=False,
            )
            pnl_memmap[first:last] = pnl.cpu().double().numpy()
            action_l1_sum += float(diagnostics.mean_action_l1) * (last - first)
            cap_sum += float(diagnostics.cap_active_fraction) * (last - first)
            maximum_action = max(
                maximum_action,
                float(diagnostics.maximum_absolute_action),
            )
            if first == 0 or last == arguments.num_paths or first % (
                25 * arguments.batch_size
            ) == 0:
                print(
                    f"audit pass 1: {last}/{arguments.num_paths} "
                    f"at {last / max(time.monotonic() - started, 1.0e-12):.1f} paths/s",
                    flush=True,
                )
    pnl_memmap.flush()
    temporary_pnl = pnl_path.with_name(f".{pnl_path.name}.tmp")
    del pnl_memmap
    os.replace(temporary_pnl, pnl_path)
    pnl = np.load(pnl_path, mmap_mode="r")
    selected_weights = np.array(
        weights[: arguments.num_paths], dtype=np.float64, copy=True
    )
    log_terms = np.log(selected_weights) - np.asarray(pnl)
    log_normalizer = float(logsumexp(log_terms) - math.log(np.sum(selected_weights)))
    combined_log_density = log_terms - logsumexp(log_terms) + math.log(arguments.num_paths)
    combined_density = np.exp(combined_log_density)
    expected_net_pnl = float(
        np.sum(selected_weights * np.asarray(pnl), dtype=np.float64)
        / np.sum(selected_weights)
    )
    pnl_second_moment = float(
        np.sum(selected_weights * np.square(np.asarray(pnl)), dtype=np.float64)
        / np.sum(selected_weights)
    )

    policy.zero_grad(set_to_none=True)
    generator.manual_seed(arguments.seed)
    denominator_log = logsumexp(log_terms)
    started = time.monotonic()
    for first in range(0, arguments.num_paths, arguments.batch_size):
        last = min(first + arguments.batch_size, arguments.num_paths)
        pnl_batch, _, _ = _simulate(
            setup,
            arguments,
            num_paths=last - first,
            generator=generator,
            retain_returns=False,
        )
        probability = torch.as_tensor(
            np.exp(
                np.log(selected_weights[first:last])
                - np.asarray(pnl[first:last])
                - denominator_log
            ),
            dtype=pnl_batch.dtype,
            device=pnl_batch.device,
        )
        (-torch.sum(probability.detach() * pnl_batch)).backward()
        if first == 0 or last == arguments.num_paths or first % (
            25 * arguments.batch_size
        ) == 0:
            print(
                f"audit pass 2: {last}/{arguments.num_paths} "
                f"at {last / max(time.monotonic() - started, 1.0e-12):.1f} paths/s",
                flush=True,
            )
    gradient_rms, gradient_maximum, parameter_count = _gradient_statistics(policy)
    parameter_square = sum(
        float(torch.sum(parameter.detach().double().square()).cpu())
        for parameter in policy.parameters()
    )
    return {
        "log_loss": log_normalizer,
        "oce_value": -log_normalizer,
        "zero_policy_oce_value": 0.0,
        "effective_sample_size_after_adversary": float(
            arguments.num_paths**2
            / np.sum(np.square(combined_density), dtype=np.float64)
        ),
        "expected_net_pnl_under_q": expected_net_pnl,
        "net_pnl_standard_deviation_under_q": math.sqrt(
            max(pnl_second_moment - expected_net_pnl**2, 0.0)
        ),
        "mean_action_l1_per_state": action_l1_sum / arguments.num_paths,
        "maximum_absolute_action": maximum_action,
        "cap_active_fraction": cap_sum / arguments.num_paths,
        "gradient_rms": gradient_rms,
        "gradient_maximum_absolute": gradient_maximum,
        "network_parameter_l2": math.sqrt(parameter_square),
        "network_parameter_count": parameter_count,
    }


def run(arguments: argparse.Namespace) -> None:
    """Train and audit the conditional policy on the fixed convex-Q paths."""

    _validate_arguments(arguments)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = arguments.output_directory / "conditional-oce.lock"
    with FileLock(lock_path):
        _run_locked(arguments)


def _run_locked(arguments: argparse.Namespace) -> None:
    """Run while holding the logical model's training lock."""

    returns = np.load(arguments.returns, mmap_mode="r", allow_pickle=False)
    weights = np.load(arguments.weights, mmap_mode="r", allow_pickle=False)
    convex_report = json.loads(arguments.convex_report.read_text())
    if returns.shape != (1_000_000, 25, 45):
        raise ValueError(f"expected the fitted 1000000x25x45 return matrix, found {returns.shape}")
    if weights.shape != (1_000_000,) or np.any(~np.isfinite(weights) | (weights <= 0.0)):
        raise ValueError("convex density weights must be one million finite-positive values")
    if not 0 < arguments.num_paths <= len(returns):
        raise ValueError("requested path count exceeds the convex calibration population")

    setup = _setup(arguments)
    checkpoint_path = arguments.output_directory / "conditional-oce.pt"
    report_path = arguments.output_directory / "conditional-oce-report.json"
    alignment_error = _alignment_check(setup, arguments, returns)
    print(f"fixed-path alignment maximum error={alignment_error:.6g}", flush=True)
    if arguments.alignment_only:
        write_json_atomic(
            {
                "status": "alignment_complete",
                "paths_checked": min(arguments.batch_size, arguments.num_paths),
                "maximum_return_error": alignment_error,
            },
            report_path,
        )
        return

    cached_states: np.ndarray | None = None
    cached_prices: np.ndarray | None = None
    if arguments.fixed_feature_cache is not None:
        _prepare_feature_cache(
            setup,
            arguments,
            directory=arguments.fixed_feature_cache,
        )
        if arguments.prepare_features_only:
            write_json_atomic(
                {
                    "status": "feature_cache_complete",
                    "paths": arguments.num_paths,
                    "steps": arguments.num_steps,
                    "feature_cache": str(arguments.fixed_feature_cache),
                    "maximum_return_error": alignment_error,
                },
                report_path,
            )
            return
        state_path, price_path, _ = _feature_cache_paths(arguments.fixed_feature_cache)
        cached_states = np.load(state_path, mmap_mode="r", allow_pickle=False)
        cached_prices = np.load(price_path, mmap_mode="r", allow_pickle=False)

    policy: SharedExpiryOcePolicy = setup["policy"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    generator = torch.Generator(device=setup["env"].device).manual_seed(arguments.seed)
    epoch = 0
    path_offset = 0
    updates = 0
    history: list[dict[str, float | int]] = []
    phase = "training"
    if arguments.resume and checkpoint_path.exists():
        epoch, path_offset, updates, history, phase = _load_checkpoint(
            checkpoint_path,
            setup=setup,
            optimizer=optimizer,
            generator=generator,
        )
        print(
            f"resumed conditional OCE phase={phase} epoch={epoch} path={path_offset}",
            flush=True,
        )
    training_batch_size = (
        arguments.optimization_batch_size
        if cached_states is not None
        else arguments.batch_size
    )
    batches_per_epoch = math.ceil(arguments.num_paths / training_batch_size)
    total_updates = arguments.epochs * batches_per_epoch
    selected_weights = np.array(
        weights[: arguments.num_paths], dtype=np.float64, copy=True
    )
    selected_weights *= arguments.num_paths / np.sum(selected_weights)

    if phase == "training":
        policy.train()
        while epoch < arguments.epochs:
            while path_offset < arguments.num_paths:
                last = min(path_offset + training_batch_size, arguments.num_paths)
                batch_paths = last - path_offset
                if batch_paths % 2:
                    raise RuntimeError("final conditional OCE batch is not antithetic")
                learning_rate = _learning_rate(arguments, updates, total_updates)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                if cached_states is None or cached_prices is None:
                    pnl, _, diagnostics = _simulate(
                        setup,
                        arguments,
                        num_paths=batch_paths,
                        generator=generator,
                        retain_returns=False,
                    )
                else:
                    batch = _cached_batch(
                        states=cached_states,
                        prices=cached_prices,
                        returns=returns,
                        first=path_offset,
                        last=last,
                        dtype=setup["env"].fdtype,
                        device=setup["env"].device,
                    )
                    pnl, diagnostics = _fixed_feature_pnl(
                        policy,
                        states=batch[0],
                        option_prices=batch[1],
                        returns=batch[2],
                        transaction_cost=setup["transaction_cost"],
                    )
                density = torch.as_tensor(
                    selected_weights[path_offset:last],
                    dtype=torch.float64,
                    device=pnl.device,
                )
                loss = weighted_log_oce(pnl.double(), density)
                loss.backward()
                gradient_rms, gradient_maximum, parameter_count = _gradient_statistics(policy)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), arguments.gradient_clip_norm
                )
                optimizer.step()
                path_offset = last
                updates += 1
                if updates % 10 == 0 or path_offset == arguments.num_paths:
                    record: dict[str, float | int] = {
                        "epoch": epoch + 1,
                        "update": updates,
                        "processed_paths_in_epoch": path_offset,
                        "batch_oce": -float(loss.detach()),
                        "learning_rate": learning_rate,
                        "gradient_rms": gradient_rms,
                        "gradient_maximum_absolute": gradient_maximum,
                        "gradient_norm_before_clip": float(gradient_norm),
                        "network_parameter_count": parameter_count,
                        "mean_action_l1": float(diagnostics.mean_action_l1),
                        "maximum_absolute_action": float(
                            diagnostics.maximum_absolute_action
                        ),
                        "cap_active_fraction": float(diagnostics.cap_active_fraction),
                    }
                    history.append(record)
                    print(json.dumps(record), flush=True)
                if updates % arguments.checkpoint_every_updates == 0:
                    _atomic_torch_save(
                        _checkpoint_payload(
                            setup=setup,
                            arguments=arguments,
                            optimizer=optimizer,
                            generator=generator,
                            epoch=epoch,
                            path_offset=path_offset,
                            updates=updates,
                            history=history,
                            phase="training",
                        ),
                        checkpoint_path,
                    )
            epoch += 1
            path_offset = 0
            if epoch < arguments.epochs and cached_states is None:
                generator.manual_seed(arguments.seed)
        phase = "trained"
        _atomic_torch_save(
            _checkpoint_payload(
                setup=setup,
                arguments=arguments,
                optimizer=optimizer,
                generator=generator,
                epoch=epoch,
                path_offset=path_offset,
                updates=updates,
                history=history,
                phase=phase,
            ),
            checkpoint_path,
        )

    if arguments.fixed_feature_cache is None:
        audit = _audit(
            setup,
            arguments,
            selected_weights,
            output_directory=arguments.output_directory,
        )
    else:
        audit = _audit_cached(
            setup,
            arguments,
            selected_weights,
            returns,
            feature_directory=arguments.fixed_feature_cache,
            output_directory=arguments.output_directory,
        )
    phase = "complete"
    _atomic_torch_save(
        _checkpoint_payload(
            setup=setup,
            arguments=arguments,
            optimizer=optimizer,
            generator=generator,
            epoch=epoch,
            path_offset=path_offset,
            updates=updates,
            history=history,
            phase=phase,
        ),
        checkpoint_path,
    )
    result = {
        "schema": "dysanos-convex-q-conditional-oce-report-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "reference_measure": "the fitted convex empirical Q on the same one-million paths",
            "policy_input": "current Direct-200 hidden state h and eleven current option prices for one expiry",
            "policy_output": "shared 11 option actions plus one spot action per expiry; four spot heads are summed",
            "objective": "sup_theta -log E_Q exp(-sum_t[a_theta(h_t,prices_t)'R_t-cost'abs(a_theta)])",
            "scope": "in-sample conditional-strategy stress test of the fixed empirical Q",
        },
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
        "policy_config": asdict(setup["policy_config"]),
        "fixed_path_alignment_maximum_error": alignment_error,
        "state_index": setup["state_index"],
        "state_date": str(setup["dates"][setup["state_index"]]),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history_sha256": checkpoint_digest(arguments.target_history),
        "convex_density_ess": convex_report["fit"]["effective_sample_size"],
        "training_history": history,
        "audit": audit,
        "transition_diagnostics": setup["transition_diagnostics"],
        "spot_copula_fit": setup["copula_fit"].to_dict(),
        "spot_copula_diagnostics": setup["copula_diagnostics"],
    }
    write_json_atomic(result, report_path)
    print(json.dumps({"status": "complete", "audit": audit}, indent=2), flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
