"""Audit a convex empirical Q with a shared expiry-slice OCE policy."""

from __future__ import annotations

import argparse
import copy
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
from quantdata.dysanos.flex.oce_curvature_optimizers import Kfac, Shampoo
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


@version("1.3", dependencies=[FeedForwardConfig])
@dataclass(frozen=True, kw_only=True)
class SharedExpiryOceConfig:
    """Architecture and action geometry of the conditional OCE policy."""

    label: str
    group: str = "flex/convex-q-conditional-oce"
    user_version: str = "1.2"
    state_size: int = 200
    num_expiries: int = 4
    num_strikes: int = 11
    maximum_action_l1: float = 16.0
    expiry_specific_linear_skip: bool = False
    include_calendar_time: bool = False
    action_parameterization: str = "signed"
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
        verify_inp(
            self.action_parameterization in ("signed", "long-short"),
            lambda: (
                "policy action parameterization must be 'signed' or 'long-short', "
                f"found {self.action_parameterization!r}"
            ),
        )


def _assemble_expiry_slice_action_details(
    raw_heads: torch.Tensor,
    *,
    maximum_l1: float,
    action_parameterization: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return net action, transaction-cost basis, and uncapped gross exposure."""

    if raw_heads.ndim != 3 or raw_heads.shape[-1] < 2:
        raise ValueError("raw expiry heads must have shape (paths,expiries,heads)")
    if action_parameterization == "signed":
        spot = torch.sum(raw_heads[..., :1], dim=1)
        options = raw_heads[..., 1:].reshape(len(raw_heads), -1)
        raw_action = torch.cat((spot, options), dim=1)
        action = l1_capped_action(raw_action, maximum_l1)
        return action, torch.abs(action), torch.sum(torch.abs(raw_action), dim=1)
    if action_parameterization != "long-short" or raw_heads.shape[-1] % 2:
        raise ValueError("long-short expiry heads must contain equal long and short logits")

    head_size = raw_heads.shape[-1] // 2
    if head_size < 2:
        raise ValueError("long-short expiry heads require spot plus at least one option")
    long_logits = raw_heads[..., :head_size]
    short_logits = raw_heads[..., head_size:]
    long_action = torch.cat(
        (
            torch.relu(torch.sum(long_logits[..., :1], dim=1)),
            torch.relu(long_logits[..., 1:]).reshape(len(raw_heads), -1),
        ),
        dim=1,
    )
    short_action = torch.cat(
        (
            torch.relu(torch.sum(short_logits[..., :1], dim=1)),
            torch.relu(short_logits[..., 1:]).reshape(len(raw_heads), -1),
        ),
        dim=1,
    )
    uncapped_gross = torch.sum(long_action + short_action, dim=1)
    scale = maximum_l1 / torch.maximum(
        uncapped_gross,
        torch.as_tensor(
            maximum_l1,
            dtype=raw_heads.dtype,
            device=raw_heads.device,
        ),
    )
    scaled_long = long_action * scale[:, None]
    scaled_short = short_action * scale[:, None]
    return scaled_long - scaled_short, scaled_long + scaled_short, uncapped_gross


def assemble_expiry_slice_actions(
    raw_heads: torch.Tensor,
    *,
    maximum_l1: float,
    action_parameterization: str = "signed",
) -> torch.Tensor:
    """Aggregate four spot heads and retain every expiry-specific option head."""

    return _assemble_expiry_slice_action_details(
        raw_heads,
        maximum_l1=maximum_l1,
        action_parameterization=action_parameterization,
    )[0]


class SharedExpiryOcePolicy(BaseModel):
    """One shared spot-plus-strikes network applied to each expiry slice."""

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
        head_size = config.num_strikes + 1
        output_size = (
            head_size if config.action_parameterization == "signed" else 2 * head_size
        )
        feature_size = (
            config.state_size
            + config.num_strikes
            + int(config.include_calendar_time)
        )
        self.network = FeedForward(
            env,
            feature_size,
            output_size,
            config.network,
        )
        self.linear_skip = torch.nn.ModuleList(
            torch.nn.Linear(
                feature_size,
                output_size,
                bias=True,
                device=env.device,
                dtype=env.fdtype,
            )
            for _ in range(config.num_expiries)
        ) if config.expiry_specific_linear_skip else None
        final_layer = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            if final_layer.bias is not None:
                final_layer.bias.zero_()
            if self.linear_skip is not None:
                for layer in self.linear_skip:
                    layer.weight.zero_()
                    layer.bias.zero_()

    def forward(
        self,
        states: torch.Tensor,
        option_prices: torch.Tensor,
        *,
        calendar_time: torch.Tensor | float | None = None,
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
        feature_blocks = [state_block, normalized_prices]
        if self.config.include_calendar_time:
            if calendar_time is None:
                raise ValueError("calendar time is required by this policy")
            time_value = torch.as_tensor(
                calendar_time,
                dtype=states.dtype,
                device=states.device,
            )
            if time_value.ndim == 0:
                time_value = time_value.expand(len(states))
            if time_value.shape == (len(states),):
                time_value = time_value[:, None]
            if time_value.shape != (len(states), 1):
                raise ValueError(
                    "calendar time must be scalar or have one value per path"
                )
            feature_blocks.append(
                time_value[:, None, :].expand(-1, self.config.num_expiries, -1)
            )
        elif calendar_time is not None:
            raise ValueError("calendar time was supplied to a time-homogeneous policy")
        features = torch.cat(feature_blocks, dim=-1)
        raw_heads = self.network(features)
        if self.linear_skip is not None:
            raw_heads = raw_heads + torch.stack(
                tuple(
                    layer(features[:, expiry])
                    for expiry, layer in enumerate(self.linear_skip)
                ),
                dim=1,
            )
        action = assemble_expiry_slice_actions(
            raw_heads,
            maximum_l1=self.config.maximum_action_l1,
            action_parameterization=self.config.action_parameterization,
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
            action, raw_heads = policy(
                states.detach(),
                prices_by_expiry.detach(),
                calendar_time=(
                    day / max(num_steps - 1, 1)
                    if policy.config.include_calendar_time
                    else None
                ),
            )
            action, cost_basis, uncapped_gross = _assemble_expiry_slice_action_details(
                raw_heads,
                maximum_l1=policy.config.maximum_action_l1,
                action_parameterization=policy.config.action_parameterization,
            )
            action_l1_sum += torch.sum(torch.sum(cost_basis, dim=1).double())
            maximum_absolute_action = torch.maximum(
                maximum_absolute_action,
                torch.amax(torch.abs(action)).double(),
            )
            cap_active_sum += torch.sum(
                (uncapped_gross >= policy.config.maximum_action_l1).double()
            )
            action_count += num_paths

            entry_spots.append(path_spot.detach())
            entry_prices.append(issue_values.detach())
            entry_strikes.append(issue_strikes.detach())
            entry_option_actions.append(action[:, 1:])
            pnl = pnl - torch.sum(transaction_cost[None] * cost_basis, dim=1)

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


def _full_population_oce_probabilities(
    objective_pnl: np.ndarray,
    density: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return the exact empirical OCE loss and normalized tilted probabilities."""

    if objective_pnl.ndim != 1 or density.shape != objective_pnl.shape or len(density) == 0:
        raise ValueError("objective PnL and density must be matching nonempty vectors")
    if np.any(~np.isfinite(objective_pnl)):
        raise ValueError("objective PnL must be finite")
    if np.any(~np.isfinite(density) | (density <= 0.0)):
        raise ValueError("density weights must be finite-positive")
    log_terms = np.log(density) - objective_pnl
    denominator_log = float(logsumexp(log_terms))
    probabilities = np.exp(log_terms - denominator_log)
    probabilities /= np.sum(probabilities, dtype=np.float64)
    return denominator_log - math.log(np.sum(density, dtype=np.float64)), probabilities


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
    parser.add_argument(
        "--full-population-gradient",
        action="store_true",
        help=(
            "Use every fixed path to form one exact empirical OCE gradient before "
            "each optimizer update."
        ),
    )
    parser.add_argument("--full-gradient-rms-tolerance", type=float, default=0.0)
    parser.add_argument("--full-gradient-maximum-tolerance", type=float, default=0.0)
    parser.add_argument("--full-gradient-scale-foc-tolerance", type=float, default=0.0)
    parser.add_argument("--full-gradient-patience", type=int, default=1)
    parser.add_argument("--prepare-features-only", action="store_true")
    parser.add_argument(
        "--initialize-policy-checkpoint",
        type=Path,
        default=None,
        help="Warm-start only the policy weights; optimizer and schedule restart.",
    )
    parser.add_argument(
        "--valuation-only",
        action="store_true",
        help="Audit the initialized policy without changing its weights.",
    )
    parser.add_argument(
        "--frozen-policy-checkpoints",
        type=Path,
        nargs="*",
        default=(),
        help="Frozen additive policy components evaluated before the trainable residual.",
    )
    parser.add_argument(
        "--baseline-pnl",
        type=Path,
        default=None,
        help="Prior cumulative PnL subtracted from the new cumulative PnL objective.",
    )
    parser.add_argument(
        "--claim-pnl",
        type=Path,
        default=None,
        help="Optional fixed pathwise claim added to the entropy-hedging objective.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--linear-action-skip", action="store_true")
    parser.add_argument(
        "--include-calendar-time",
        action="store_true",
        help="Append the normalized entry day t/(num_steps-1) to every action input.",
    )
    parser.add_argument(
        "--normalized-moneyness",
        type=float,
        nargs="+",
        default=list(_NORMALIZED_MONEYNESS),
        help="Increasing normalized-moneyness coordinates for every expiry slice.",
    )
    parser.add_argument(
        "--long-short-actions",
        action="store_true",
        help=(
            "Learn separate nonnegative long and short legs; transaction costs and "
            "the action cap apply to their gross sum."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "lbfgs", "newton-cg", "kfac", "shampoo"),
        default="adam",
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument("--lbfgs-max-iterations", type=int, default=4)
    parser.add_argument("--lbfgs-max-evaluations", type=int, default=10)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1.0e-7)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1.0e-10)
    parser.add_argument("--newton-hvp-relative-step", type=float, default=1.0e-3)
    parser.add_argument("--newton-cg-max-iterations", type=int, default=8)
    parser.add_argument("--newton-cg-relative-tolerance", type=float, default=0.1)
    parser.add_argument("--newton-initial-trust-radius", type=float, default=0.05)
    parser.add_argument("--newton-maximum-trust-radius", type=float, default=1.0)
    parser.add_argument("--newton-acceptance-eta", type=float, default=0.15)
    parser.add_argument("--newton-damping", type=float, default=1.0e-4)
    parser.add_argument("--kfac-factor-decay", type=float, default=0.9)
    parser.add_argument("--kfac-damping", type=float, default=1.0e-3)
    parser.add_argument("--kfac-momentum", type=float, default=0.0)
    parser.add_argument("--kfac-kl-clip", type=float, default=1.0e-2)
    parser.add_argument("--kfac-factor-update-frequency", type=int, default=1)
    parser.add_argument("--shampoo-beta2", type=float, default=0.9)
    parser.add_argument("--shampoo-momentum", type=float, default=0.0)
    parser.add_argument("--shampoo-damping", type=float, default=1.0e-4)
    parser.add_argument("--shampoo-precondition-frequency", type=int, default=1)
    parser.add_argument("--shampoo-maximum-parameter-step", type=float, default=0.05)
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
    _nms_book(
        _EXPIRY_DAYS,
        normalized_moneyness=tuple(arguments.normalized_moneyness),
    )
    integer_values = (
        arguments.num_paths,
        arguments.num_steps,
        arguments.batch_size,
        arguments.optimization_batch_size,
        arguments.epochs,
        arguments.hidden_width,
        arguments.hidden_layers,
        arguments.checkpoint_every_updates,
        arguments.full_gradient_patience,
        arguments.lbfgs_history_size,
        arguments.lbfgs_max_iterations,
        arguments.lbfgs_max_evaluations,
        arguments.kfac_factor_update_frequency,
        arguments.shampoo_precondition_frequency,
        arguments.newton_cg_max_iterations,
    )
    if any(value <= 0 for value in integer_values):
        raise ValueError("conditional OCE integer controls must be positive")
    if arguments.num_paths % 2 or arguments.batch_size % 2:
        raise ValueError("conditional OCE paths and batches must be even")
    if arguments.optimization_batch_size % 2:
        raise ValueError("conditional OCE optimization batches must be even")
    if arguments.prepare_features_only and arguments.fixed_feature_cache is None:
        raise ValueError("feature-only mode requires --fixed-feature-cache")
    if arguments.full_population_gradient and arguments.fixed_feature_cache is None:
        raise ValueError("full-population gradients require --fixed-feature-cache")
    if arguments.optimizer != "adam" and not arguments.full_population_gradient:
        raise ValueError(
            f"{arguments.optimizer} requires exact full-population gradients"
        )
    if arguments.long_short_actions and arguments.frozen_policy_checkpoints:
        raise ValueError(
            "long-short actions require one cumulative trainable policy, not frozen additives"
        )
    if arguments.resume and arguments.initialize_policy_checkpoint is not None:
        raise ValueError("resume and policy-only warm start are mutually exclusive")
    if arguments.valuation_only and arguments.initialize_policy_checkpoint is None:
        raise ValueError("valuation-only mode requires an initialized policy")
    if arguments.valuation_only and arguments.resume:
        raise ValueError("valuation-only mode cannot resume optimizer state")
    if (
        arguments.frozen_policy_checkpoints
        or arguments.baseline_pnl is not None
        or arguments.claim_pnl is not None
    ) and arguments.fixed_feature_cache is None:
        raise ValueError("additive policies and fixed claims require the feature cache")
    # A frozen policy can either be a baseline for an incremental density tilt
    # or a fixed component of the cumulative hedge. In the latter case the
    # trainable residual is optimized around zero without subtracting its PnL.
    if arguments.num_steps != 25:
        raise ValueError("the fitted convex density has exactly 25 entry dates")
    positive_values = (
        arguments.learning_rate,
        arguments.minimum_learning_rate,
        arguments.action_l1_cap,
        arguments.gradient_clip_norm,
        arguments.lbfgs_tolerance_grad,
        arguments.lbfgs_tolerance_change,
        arguments.newton_hvp_relative_step,
        arguments.newton_cg_relative_tolerance,
        arguments.newton_initial_trust_radius,
        arguments.newton_maximum_trust_radius,
        arguments.newton_damping,
        arguments.kfac_damping,
        arguments.kfac_kl_clip,
        arguments.shampoo_damping,
        arguments.shampoo_maximum_parameter_step,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
        raise ValueError("conditional OCE scalar controls must be finite-positive")
    if arguments.minimum_learning_rate > arguments.learning_rate:
        raise ValueError("minimum learning rate exceeds the peak learning rate")
    if arguments.newton_initial_trust_radius > arguments.newton_maximum_trust_radius:
        raise ValueError("initial Newton trust radius exceeds its maximum")
    if not 0.0 < arguments.newton_cg_relative_tolerance < 1.0:
        raise ValueError("Newton CG relative tolerance must be in (0,1)")
    if not 0.0 <= arguments.newton_acceptance_eta < 0.25:
        raise ValueError("Newton trust-region acceptance eta must be in [0,0.25)")
    unit_interval_values = (
        arguments.kfac_factor_decay,
        arguments.kfac_momentum,
        arguments.shampoo_beta2,
        arguments.shampoo_momentum,
    )
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in unit_interval_values):
        raise ValueError("curvature optimizer decay, momentum, and eta must be in [0,1)")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (arguments.spot_cost_bps, arguments.option_cost_bps)
    ):
        raise ValueError("conditional OCE transaction costs must be finite-nonnegative")
    if arguments.warmup_updates < 0:
        raise ValueError("warmup updates cannot be negative")
    full_gradient_tolerances = (
        arguments.full_gradient_rms_tolerance,
        arguments.full_gradient_maximum_tolerance,
        arguments.full_gradient_scale_foc_tolerance,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in full_gradient_tolerances):
        raise ValueError("full-population convergence tolerances must be finite-nonnegative")


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


def _serialized_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    def serialize(value: Any) -> Any:
        if isinstance(value, (Path, date)):
            return str(value)
        if isinstance(value, (tuple, list)):
            return [serialize(item) for item in value]
        return value

    return {key: serialize(value) for key, value in vars(arguments).items()}


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
    num_strikes = len(book.labels[1:]) // len(book.expiry_days)
    log_prices = torch.log(torch.clamp_min(prices, torch.finfo(dtype).tiny)).reshape(
        len(historical_states), len(book.expiry_days), num_strikes
    )
    state_mean = torch.mean(historical_states, dim=0)
    state_std = torch.std(historical_states, dim=0, correction=0)
    state_scale = torch.where(state_std > 1.0e-5, state_std, torch.ones_like(state_std))
    log_price_mean = torch.mean(log_prices, dim=(0, 1))
    log_price_std = torch.std(
        log_prices.reshape(-1, num_strikes), dim=0, correction=0
    )
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
    normalized_moneyness = tuple(arguments.normalized_moneyness)
    book = _nms_book(
        _EXPIRY_DAYS,
        normalized_moneyness=normalized_moneyness,
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
    num_strikes = len(normalized_moneyness)
    label_parts = ["convex-q-shared-expiry", f"{num_strikes}plus1"]
    if arguments.long_short_actions:
        label_parts.append("long-short")
    if arguments.linear_action_skip:
        label_parts.append("linear-skip")
    if arguments.include_calendar_time:
        label_parts.append("time")
    label_parts.append("v1")
    policy_config = SharedExpiryOceConfig(
        label="-".join(label_parts),
        state_size=initial_state.shape[0],
        num_strikes=num_strikes,
        maximum_action_l1=arguments.action_l1_cap,
        expiry_specific_linear_skip=arguments.linear_action_skip,
        include_calendar_time=arguments.include_calendar_time,
        action_parameterization=(
            "long-short" if arguments.long_short_actions else "signed"
        ),
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
        "normalized_moneyness": list(arguments.normalized_moneyness),
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
    frozen_policies: tuple[SharedExpiryOcePolicy, ...] = (),
) -> tuple[torch.Tensor, _SimulationDiagnostics]:
    """Evaluate the globally capped policy on cached states and returns."""

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
        calendar_time = (
            day / max(states.shape[1] - 1, 1)
            if policy.config.include_calendar_time
            else None
        )
        _, raw_heads = policy(
            states[:, day],
            option_prices[:, day],
            calendar_time=calendar_time,
        )
        if frozen_policies:
            if policy.config.action_parameterization != "signed":
                raise ValueError(
                    "frozen additive policies are unsupported for long-short actions"
                )
            frozen_raw_sum = torch.zeros_like(raw_heads)
            with torch.no_grad():
                for frozen_policy in frozen_policies:
                    _, frozen_raw_heads = frozen_policy(
                        states[:, day],
                        option_prices[:, day],
                        calendar_time=calendar_time,
                    )
                    frozen_raw_sum = frozen_raw_sum + frozen_raw_heads
            raw_heads = raw_heads + frozen_raw_sum
        action, cost_basis, uncapped_gross = _assemble_expiry_slice_action_details(
            raw_heads,
            maximum_l1=policy.config.maximum_action_l1,
            action_parameterization=policy.config.action_parameterization,
        )
        pnl = pnl + torch.sum(action * returns[:, day], dim=1)
        pnl = pnl - torch.sum(transaction_cost[None] * cost_basis, dim=1)
        action_l1_sum += torch.sum(torch.sum(cost_basis, dim=1).double())
        maximum_action = torch.maximum(
            maximum_action,
            torch.amax(torch.abs(action)).double(),
        )
        cap_active_sum += torch.sum(
            (
                uncapped_gross >= policy.config.maximum_action_l1
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


def _full_population_training_update(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    *,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    density: np.ndarray,
    baseline_pnl: np.ndarray | None,
    claim_pnl: np.ndarray | None,
    perform_step: bool = True,
    clip_gradient: bool = True,
) -> dict[str, float | int]:
    """Take one optimizer step using the exact fixed-population OCE gradient.

    The first pass evaluates the population normalizer without retaining an
    autograd graph. The second pass recomputes each chunk and accumulates its
    contribution to the same normalized population gradient. Consequently the
    chunk size affects memory and speed, but not the objective being optimized.
    """

    policy: SharedExpiryOcePolicy = setup["policy"]
    num_paths = arguments.num_paths
    batch_size = arguments.optimization_batch_size
    objective_values = np.empty(num_paths, dtype=np.float64)
    network_pnl_values = np.empty(num_paths, dtype=np.float64)
    action_l1_sum = 0.0
    cap_sum = 0.0
    maximum_action = 0.0
    started = time.monotonic()
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
            pnl, diagnostics = _fixed_feature_pnl(
                policy,
                states=batch[0],
                option_prices=batch[1],
                returns=batch[2],
                transaction_cost=setup["transaction_cost"],
                frozen_policies=setup["frozen_policies"],
            )
            network_pnl_values[first:last] = pnl.detach().cpu().double().numpy()
            objective_values[first:last] = network_pnl_values[first:last]
            if claim_pnl is not None:
                objective_values[first:last] += np.asarray(claim_pnl[first:last])
            if baseline_pnl is not None:
                objective_values[first:last] -= np.asarray(baseline_pnl[first:last])
            count = last - first
            action_l1_sum += float(diagnostics.mean_action_l1) * count
            cap_sum += float(diagnostics.cap_active_fraction) * count
            maximum_action = max(
                maximum_action,
                float(diagnostics.maximum_absolute_action),
            )

    log_loss, tilted_probabilities = _full_population_oce_probabilities(
        objective_values,
        density,
    )
    tilted_expected_objective = float(
        np.sum(tilted_probabilities * objective_values, dtype=np.float64)
    )
    tilted_expected_network_pnl = float(
        np.sum(tilted_probabilities * network_pnl_values, dtype=np.float64)
    )
    effective_sample_size = float(
        1.0 / np.sum(np.square(tilted_probabilities), dtype=np.float64)
    )

    optimizer.zero_grad(set_to_none=True)
    if hasattr(optimizer, "begin_full_batch"):
        optimizer.begin_full_batch()
    for first in range(0, num_paths, batch_size):
        last = min(first + batch_size, num_paths)
        probability = torch.as_tensor(
            tilted_probabilities[first:last],
            dtype=torch.float64,
            device=setup["env"].device,
        )
        if hasattr(optimizer, "set_batch_probabilities"):
            optimizer.set_batch_probabilities(probability)
        batch = _cached_batch(
            states=states,
            prices=prices,
            returns=returns,
            first=first,
            last=last,
            dtype=setup["env"].fdtype,
            device=setup["env"].device,
        )
        pnl, _ = _fixed_feature_pnl(
            policy,
            states=batch[0],
            option_prices=batch[1],
            returns=batch[2],
            transaction_cost=setup["transaction_cost"],
            frozen_policies=setup["frozen_policies"],
        )
        (-torch.sum(probability.detach() * pnl.double())).backward()

    gradient_rms, gradient_maximum, parameter_count = _gradient_statistics(policy)
    gradient_norm = torch.linalg.vector_norm(
        torch.stack(
            tuple(
                torch.linalg.vector_norm(parameter.grad.detach())
                for parameter in policy.parameters()
                if parameter.grad is not None
            )
        )
    )
    if clip_gradient:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), arguments.gradient_clip_norm)
    if perform_step:
        optimizer.step()
    result: dict[str, float | int] = {
        "full_population_oce": -log_loss,
        "full_population_log_loss": log_loss,
        "tilted_expected_objective_pnl": tilted_expected_objective,
        "tilted_expected_network_pnl": tilted_expected_network_pnl,
        "effective_sample_size_after_adversary": effective_sample_size,
        "gradient_rms": gradient_rms,
        "gradient_maximum_absolute": gradient_maximum,
        "gradient_norm_before_clip": float(gradient_norm),
        "network_parameter_count": parameter_count,
        "mean_action_l1": action_l1_sum / num_paths,
        "maximum_absolute_action": maximum_action,
        "cap_active_fraction": cap_sum / num_paths,
        "seconds": time.monotonic() - started,
    }
    for group in optimizer.param_groups:
        for key in (
            "last_fisher_inner_product",
            "last_kl_scale",
            "last_preconditioned_direction_norm",
            "last_parameter_step_scale",
        ):
            if key in group:
                result[key] = float(group[key])
    return result


def _trainable_parameter_vector(policy: torch.nn.Module) -> np.ndarray:
    """Copy all trainable policy parameters into one float64 CPU vector."""

    values = [
        parameter.detach().reshape(-1).cpu().double().numpy()
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    if not values:
        raise RuntimeError("policy has no trainable parameters")
    return np.concatenate(values)


def _set_trainable_parameter_vector(
    policy: torch.nn.Module,
    values: np.ndarray,
) -> None:
    """Load one CPU vector into the trainable policy parameters."""

    offset = 0
    with torch.no_grad():
        for parameter in policy.parameters():
            if not parameter.requires_grad:
                continue
            count = parameter.numel()
            if offset + count > len(values):
                raise ValueError("parameter vector is shorter than the policy")
            parameter.copy_(
                torch.as_tensor(
                    values[offset : offset + count],
                    dtype=parameter.dtype,
                    device=parameter.device,
                ).reshape_as(parameter)
            )
            offset += count
    if offset != len(values):
        raise ValueError("parameter vector is longer than the policy")


def _trainable_gradient_vector(policy: torch.nn.Module) -> np.ndarray:
    """Copy the current exact policy gradient into one float64 CPU vector."""

    values: list[np.ndarray] = []
    for parameter in policy.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError("trainable policy parameter is missing its gradient")
        values.append(parameter.grad.detach().reshape(-1).cpu().double().numpy())
    return np.concatenate(values)


def _trust_region_boundary_distance(
    step: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> float:
    """Return positive tau such that ``||step + tau direction|| == radius``."""

    direction_square = float(np.dot(direction, direction))
    step_direction = float(np.dot(step, direction))
    discriminant = step_direction**2 + direction_square * (
        radius**2 - float(np.dot(step, step))
    )
    return (-step_direction + math.sqrt(max(discriminant, 0.0))) / max(
        direction_square,
        np.finfo(np.float64).tiny,
    )


def _run_trust_region_newton_cg(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    states: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    density: np.ndarray,
    baseline_pnl: np.ndarray | None,
    claim_pnl: np.ndarray | None,
    checkpoint_path: Path,
    epoch: int,
    updates: int,
    history: list[dict[str, float | int]],
) -> tuple[int, int, list[dict[str, float | int]], str]:
    """Minimize exact empirical log-OCE with trust-region Newton-CG.

    Hessian-vector products use centered differences of exact full-population
    gradients. This includes the derivative of the normalized OCE density,
    unlike differentiating the detached first-gradient path weights. The
    finite difference also provides a generalized local curvature estimate at
    the positive-part transaction-cost kinks.
    """

    policy: SharedExpiryOcePolicy = setup["policy"]
    radius = arguments.newton_initial_trust_radius
    function_evaluations = 0
    hvp_evaluations = 0

    def evaluate(values: np.ndarray) -> tuple[float, np.ndarray, dict[str, float | int]]:
        nonlocal function_evaluations
        _set_trainable_parameter_vector(policy, values)
        record = _full_population_training_update(
            setup,
            arguments,
            optimizer,
            states=states,
            prices=prices,
            returns=returns,
            density=density,
            baseline_pnl=baseline_pnl,
            claim_pnl=claim_pnl,
            perform_step=False,
            clip_gradient=False,
        )
        function_evaluations += 1
        return (
            float(record["full_population_log_loss"]),
            _trainable_gradient_vector(policy),
            record,
        )

    values = _trainable_parameter_vector(policy)
    loss, gradient, base_record = evaluate(values)
    while epoch < arguments.epochs:
        base_values = values.copy()
        base_loss = loss
        base_gradient = gradient.copy()
        gradient_norm = float(np.linalg.norm(base_gradient))
        if gradient_norm == 0.0:
            break

        def hessian_vector_product(direction: np.ndarray) -> np.ndarray:
            nonlocal hvp_evaluations
            direction_norm = float(np.linalg.norm(direction))
            difference = arguments.newton_hvp_relative_step * max(
                1.0, float(np.linalg.norm(base_values))
            ) / max(direction_norm, np.finfo(np.float64).tiny)
            _, plus_gradient, _ = evaluate(base_values + difference * direction)
            _, minus_gradient, _ = evaluate(base_values - difference * direction)
            _set_trainable_parameter_vector(policy, base_values)
            hvp_evaluations += 1
            return (
                (plus_gradient - minus_gradient) / (2.0 * difference)
                + arguments.newton_damping * direction
            )

        step = np.zeros_like(base_gradient)
        hessian_step = np.zeros_like(base_gradient)
        residual = base_gradient.copy()
        direction = -residual
        residual_square = float(np.dot(residual, residual))
        hit_boundary = False
        negative_curvature = False
        cg_iterations = 0
        for cg_iterations in range(1, arguments.newton_cg_max_iterations + 1):
            hessian_direction = hessian_vector_product(direction)
            curvature = float(np.dot(direction, hessian_direction))
            if curvature <= 0.0:
                tau = _trust_region_boundary_distance(step, direction, radius)
                step += tau * direction
                hessian_step += tau * hessian_direction
                hit_boundary = True
                negative_curvature = True
                break
            alpha = residual_square / curvature
            candidate_step = step + alpha * direction
            if float(np.linalg.norm(candidate_step)) >= radius:
                tau = _trust_region_boundary_distance(step, direction, radius)
                step += tau * direction
                hessian_step += tau * hessian_direction
                hit_boundary = True
                break
            step = candidate_step
            hessian_step += alpha * hessian_direction
            next_residual = residual + alpha * hessian_direction
            next_square = float(np.dot(next_residual, next_residual))
            if math.sqrt(next_square) <= (
                arguments.newton_cg_relative_tolerance * gradient_norm
            ):
                residual = next_residual
                residual_square = next_square
                break
            beta = next_square / max(residual_square, np.finfo(np.float64).tiny)
            direction = -next_residual + beta * direction
            residual = next_residual
            residual_square = next_square

        predicted_reduction = -float(
            np.dot(base_gradient, step) + 0.5 * np.dot(step, hessian_step)
        )
        candidate_values = base_values + step
        candidate_loss, candidate_gradient, candidate_record = evaluate(candidate_values)
        actual_reduction = base_loss - candidate_loss
        ratio = actual_reduction / max(
            predicted_reduction,
            np.finfo(np.float64).tiny,
        )
        if ratio < 0.25:
            radius *= 0.25
        elif ratio > 0.75 and hit_boundary:
            radius = min(2.0 * radius, arguments.newton_maximum_trust_radius)
        accepted = bool(
            actual_reduction > 0.0 and ratio > arguments.newton_acceptance_eta
        )
        if accepted:
            values = candidate_values
            loss = candidate_loss
            gradient = candidate_gradient
            record = dict(candidate_record)
        else:
            values = base_values
            loss = base_loss
            gradient = base_gradient
            record = dict(base_record)
            _set_trainable_parameter_vector(policy, values)
        epoch += 1
        updates += 1
        record.update(
            {
                "epoch": epoch,
                "update": updates,
                "processed_paths_in_epoch": arguments.num_paths,
                "optimizer": "trust-region-newton-cg",
                "newton_accepted": int(accepted),
                "newton_actual_reduction": actual_reduction,
                "newton_predicted_reduction": predicted_reduction,
                "newton_reduction_ratio": ratio,
                "newton_trust_radius": radius,
                "newton_step_norm": float(np.linalg.norm(step)),
                "newton_cg_iterations": cg_iterations,
                "newton_negative_curvature": int(negative_curvature),
                "newton_hvp_evaluations": hvp_evaluations,
                "newton_function_gradient_evaluations": function_evaluations,
            }
        )
        # Keep a private copy because the history entry is subsequently retained
        # as an immutable audit of this outer iteration.
        base_record = dict(record)
        history.append(record)
        print(json.dumps(record), flush=True)
        _atomic_torch_save(
            _checkpoint_payload(
                setup=setup,
                arguments=arguments,
                optimizer=optimizer,
                generator=generator,
                epoch=epoch,
                path_offset=0,
                updates=updates,
                history=history,
                phase="trained" if epoch == arguments.epochs else "training",
            ),
            checkpoint_path,
        )
        if (
            float(record["gradient_rms"])
            <= arguments.full_gradient_rms_tolerance
            and float(record["gradient_maximum_absolute"])
            <= arguments.full_gradient_maximum_tolerance
            and abs(float(record["tilted_expected_network_pnl"]))
            <= arguments.full_gradient_scale_foc_tolerance
        ):
            break
    _set_trainable_parameter_vector(policy, values)
    return epoch, updates, history, "trained"


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
        "arguments": _serialized_arguments(arguments),
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


def _load_policy_state(
    path: Path,
    *,
    target_policy: SharedExpiryOcePolicy | None = None,
) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "dysanos-convex-q-conditional-oce-v1"
    ):
        raise ValueError(f"conditional OCE policy checkpoint is invalid: {path}")
    source = payload["policy_state_dict"]
    if target_policy is None:
        return source

    target = target_policy.state_dict()
    if source.keys() != target.keys():
        raise ValueError(f"policy checkpoint keys do not match the target policy: {path}")
    converted: dict[str, torch.Tensor] = {}
    doubled_outputs: list[str] = []
    for key, target_value in target.items():
        source_value = source[key]
        if source_value.shape == target_value.shape:
            converted[key] = source_value
            continue
        if (
            source_value.ndim >= 1
            and target_value.shape[0] == 2 * source_value.shape[0]
            and target_value.shape[1:] == source_value.shape[1:]
        ):
            # A signed logit x is represented exactly by the two nonnegative
            # legs relu(x)-relu(-x), with gross exposure |x|.
            converted[key] = torch.cat((source_value, -source_value), dim=0)
            doubled_outputs.append(key)
            continue
        raise ValueError(
            f"policy checkpoint tensor {key!r} has shape {tuple(source_value.shape)}; "
            f"target shape is {tuple(target_value.shape)}"
        )
    if doubled_outputs:
        print(
            "converted signed policy outputs to exact long-short legs: "
            + ", ".join(doubled_outputs),
            flush=True,
        )
    return converted


def _load_frozen_policies(
    setup: Mapping[str, Any],
    checkpoints: tuple[Path, ...] | list[Path],
) -> tuple[SharedExpiryOcePolicy, ...]:
    policies: list[SharedExpiryOcePolicy] = []
    for checkpoint in checkpoints:
        frozen_policy = copy.deepcopy(setup["policy"])
        frozen_policy.load_state_dict(
            _load_policy_state(checkpoint, target_policy=frozen_policy)
        )
        frozen_policy.eval()
        frozen_policy.requires_grad_(False)
        policies.append(frozen_policy)
        print(f"loaded frozen additive policy from {checkpoint}", flush=True)
    return tuple(policies)


def _audit_cached(
    setup: Mapping[str, Any],
    arguments: argparse.Namespace,
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    feature_directory: Path,
    output_directory: Path,
    baseline_pnl: np.ndarray | None = None,
    claim_pnl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit a cumulative policy and its incremental density tilt."""

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
                frozen_policies=setup["frozen_policies"],
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
    objective_pnl: np.ndarray = pnl
    if baseline_pnl is not None or claim_pnl is not None:
        objective_path = output_directory / (
            "incremental-pnl.npy" if claim_pnl is None else "objective-pnl.npy"
        )
        temporary_objective = objective_path.with_name(
            f".{objective_path.name}.tmp"
        )
        objective_memmap = np.lib.format.open_memmap(
            temporary_objective,
            mode="w+",
            dtype=np.float64,
            shape=(arguments.num_paths,),
        )
        objective_memmap[:] = np.asarray(pnl)
        if claim_pnl is not None:
            objective_memmap[:] += np.asarray(claim_pnl[: arguments.num_paths])
        if baseline_pnl is not None:
            objective_memmap[:] -= np.asarray(baseline_pnl[: arguments.num_paths])
        objective_memmap.flush()
        del objective_memmap
        os.replace(temporary_objective, objective_path)
        objective_pnl = np.load(objective_path, mmap_mode="r", allow_pickle=False)
    selected_weights = np.array(
        weights[: arguments.num_paths], dtype=np.float64, copy=True
    )
    log_terms = np.log(selected_weights) - np.asarray(objective_pnl)
    denominator_log = logsumexp(log_terms)
    log_normalizer = float(denominator_log - math.log(np.sum(selected_weights)))
    combined_density = np.exp(
        log_terms - denominator_log + math.log(arguments.num_paths)
    )
    expected_incremental_pnl = float(
        np.sum(selected_weights * np.asarray(objective_pnl), dtype=np.float64)
        / np.sum(selected_weights)
    )
    incremental_second_moment = float(
        np.sum(
            selected_weights * np.square(np.asarray(objective_pnl)),
            dtype=np.float64,
        )
        / np.sum(selected_weights)
    )
    expected_total_pnl = float(
        np.sum(selected_weights * np.asarray(pnl), dtype=np.float64)
        / np.sum(selected_weights)
    )
    expected_claim_pnl = (
        float(
            np.sum(
                selected_weights * np.asarray(claim_pnl[: arguments.num_paths]),
                dtype=np.float64,
            )
            / np.sum(selected_weights)
        )
        if claim_pnl is not None
        else 0.0
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
            frozen_policies=setup["frozen_policies"],
        )
        probability = torch.as_tensor(
            np.exp(
                np.log(selected_weights[first:last])
                - np.asarray(objective_pnl[first:last])
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
        "expected_net_pnl_under_q": expected_incremental_pnl,
        "expected_incremental_net_pnl_under_q": expected_incremental_pnl,
        "expected_total_net_pnl_under_q": expected_total_pnl,
        "expected_claim_pnl_under_q": expected_claim_pnl,
        "expected_hedged_claim_pnl_under_q": expected_incremental_pnl,
        "net_pnl_standard_deviation_under_q": math.sqrt(
            max(
                incremental_second_moment - expected_incremental_pnl**2,
                0.0,
            )
        ),
        "mean_action_l1_per_state": action_l1_sum / arguments.num_paths,
        "mean_action_gross_l1_per_state": action_l1_sum / arguments.num_paths,
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
    expected_instruments = 1 + len(_EXPIRY_DAYS) * len(arguments.normalized_moneyness)
    expected_shape = (1_000_000, 25, expected_instruments)
    if returns.shape != expected_shape:
        raise ValueError(
            f"expected the fitted return matrix with shape {expected_shape}, "
            f"found {returns.shape}"
        )
    if weights.shape != (1_000_000,) or np.any(~np.isfinite(weights) | (weights <= 0.0)):
        raise ValueError("convex density weights must be one million finite-positive values")
    if not 0 < arguments.num_paths <= len(returns):
        raise ValueError("requested path count exceeds the convex calibration population")

    setup = _setup(arguments)
    setup["frozen_policies"] = _load_frozen_policies(
        setup,
        arguments.frozen_policy_checkpoints,
    )
    if arguments.initialize_policy_checkpoint is not None:
        setup["policy"].load_state_dict(
            _load_policy_state(
                arguments.initialize_policy_checkpoint,
                target_policy=setup["policy"],
            )
        )
        print(
            f"warm-started policy from {arguments.initialize_policy_checkpoint}",
            flush=True,
        )
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
    if arguments.optimizer == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            policy.parameters(), lr=arguments.learning_rate
        )
    elif arguments.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            policy.parameters(),
            lr=arguments.learning_rate,
            max_iter=arguments.lbfgs_max_iterations,
            max_eval=arguments.lbfgs_max_evaluations,
            tolerance_grad=arguments.lbfgs_tolerance_grad,
            tolerance_change=arguments.lbfgs_tolerance_change,
            history_size=arguments.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
    elif arguments.optimizer == "kfac":
        optimizer = Kfac(
            policy,
            lr=arguments.learning_rate,
            factor_decay=arguments.kfac_factor_decay,
            damping=arguments.kfac_damping,
            momentum=arguments.kfac_momentum,
            kl_clip=arguments.kfac_kl_clip,
            factor_update_frequency=arguments.kfac_factor_update_frequency,
        )
    elif arguments.optimizer == "shampoo":
        optimizer = Shampoo(
            policy.parameters(),
            lr=arguments.learning_rate,
            beta2=arguments.shampoo_beta2,
            momentum=arguments.shampoo_momentum,
            damping=arguments.shampoo_damping,
            precondition_frequency=arguments.shampoo_precondition_frequency,
            maximum_parameter_step=arguments.shampoo_maximum_parameter_step,
        )
    else:
        # Newton-CG uses the same full-gradient evaluator but controls parameter
        # updates itself. This optimizer only supplies zero_grad/checkpoint state.
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
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
    if arguments.valuation_only:
        phase = "trained"
    training_batch_size = (
        arguments.optimization_batch_size
        if cached_states is not None
        else arguments.batch_size
    )
    batches_per_epoch = math.ceil(arguments.num_paths / training_batch_size)
    total_updates = (
        arguments.epochs
        if arguments.full_population_gradient
        else arguments.epochs * batches_per_epoch
    )
    selected_weights = np.array(
        weights[: arguments.num_paths], dtype=np.float64, copy=True
    )
    selected_weights *= arguments.num_paths / np.sum(selected_weights)
    baseline_pnl: np.ndarray | None = None
    if arguments.baseline_pnl is not None:
        baseline_pnl = np.load(arguments.baseline_pnl, mmap_mode="r", allow_pickle=False)
        if baseline_pnl.shape != (1_000_000,) or np.any(
            ~np.isfinite(baseline_pnl[: arguments.num_paths])
        ):
            raise ValueError("baseline PnL must contain one million finite values")
    claim_pnl: np.ndarray | None = None
    if arguments.claim_pnl is not None:
        claim_pnl = np.load(arguments.claim_pnl, mmap_mode="r", allow_pickle=False)
        if claim_pnl.shape != (1_000_000,) or np.any(
            ~np.isfinite(claim_pnl[: arguments.num_paths])
        ):
            raise ValueError("claim PnL must contain one million finite values")

    if (
        phase == "training"
        and arguments.full_population_gradient
        and arguments.optimizer == "newton-cg"
    ):
        if cached_states is None or cached_prices is None:
            raise RuntimeError("Newton-CG requires cached full-population features")
        policy.train()
        epoch, updates, history, phase = _run_trust_region_newton_cg(
            setup,
            arguments,
            optimizer,
            generator,
            states=cached_states,
            prices=cached_prices,
            returns=returns,
            density=selected_weights,
            baseline_pnl=baseline_pnl,
            claim_pnl=claim_pnl,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            updates=updates,
            history=history,
        )

    if phase == "training" and arguments.full_population_gradient:
        if cached_states is None or cached_prices is None:
            raise RuntimeError("full-population training requires cached policy features")
        if path_offset != 0:
            raise RuntimeError(
                "a full-population checkpoint cannot resume inside a population pass"
            )
        policy.train()
        convergence_count = 0
        while epoch < arguments.epochs:
            learning_rate = _learning_rate(arguments, updates, total_updates)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if arguments.optimizer in ("adam", "kfac", "shampoo"):
                record = _full_population_training_update(
                    setup,
                    arguments,
                    optimizer,
                    states=cached_states,
                    prices=cached_prices,
                    returns=returns,
                    density=selected_weights,
                    baseline_pnl=baseline_pnl,
                    claim_pnl=claim_pnl,
                )
            elif arguments.optimizer == "lbfgs":
                closure_evaluations = 0

                def closure() -> torch.Tensor:
                    nonlocal closure_evaluations
                    closure_evaluations += 1
                    closure_record = _full_population_training_update(
                        setup,
                        arguments,
                        optimizer,
                        states=cached_states,
                        prices=cached_prices,
                        returns=returns,
                        density=selected_weights,
                        baseline_pnl=baseline_pnl,
                        claim_pnl=claim_pnl,
                        perform_step=False,
                        clip_gradient=False,
                    )
                    return torch.tensor(
                        closure_record["full_population_log_loss"],
                        dtype=torch.float64,
                        device=setup["env"].device,
                    )

                optimizer.step(closure)
                record = _full_population_training_update(
                    setup,
                    arguments,
                    optimizer,
                    states=cached_states,
                    prices=cached_prices,
                    returns=returns,
                    density=selected_weights,
                    baseline_pnl=baseline_pnl,
                    claim_pnl=claim_pnl,
                    perform_step=False,
                    clip_gradient=False,
                )
                record["lbfgs_closure_evaluations"] = closure_evaluations + 1
            else:
                raise RuntimeError(
                    "Newton-CG must be dispatched before the standard optimizer loop"
                )
            epoch += 1
            updates += 1
            record.update(
                {
                    "epoch": epoch,
                    "update": updates,
                    "processed_paths_in_epoch": arguments.num_paths,
                    "learning_rate": learning_rate,
                }
            )
            convergence_checks: list[bool] = []
            if arguments.full_gradient_rms_tolerance > 0.0:
                convergence_checks.append(
                    float(record["gradient_rms"])
                    <= arguments.full_gradient_rms_tolerance
                )
            if arguments.full_gradient_maximum_tolerance > 0.0:
                convergence_checks.append(
                    float(record["gradient_maximum_absolute"])
                    <= arguments.full_gradient_maximum_tolerance
                )
            if arguments.full_gradient_scale_foc_tolerance > 0.0:
                convergence_checks.append(
                    abs(float(record["tilted_expected_network_pnl"]))
                    <= arguments.full_gradient_scale_foc_tolerance
                )
            convergence_count = (
                convergence_count + 1
                if convergence_checks and all(convergence_checks)
                else 0
            )
            record["convergence_patience_count"] = convergence_count
            history.append(record)
            print(json.dumps(record), flush=True)
            converged = convergence_count >= arguments.full_gradient_patience
            if (
                updates % arguments.checkpoint_every_updates == 0
                or epoch == arguments.epochs
                or converged
            ):
                _atomic_torch_save(
                    _checkpoint_payload(
                        setup=setup,
                        arguments=arguments,
                        optimizer=optimizer,
                        generator=generator,
                        epoch=epoch,
                        path_offset=0,
                        updates=updates,
                        history=history,
                        phase=(
                            "trained"
                            if epoch == arguments.epochs or converged
                            else "training"
                        ),
                    ),
                    checkpoint_path,
                )
            if converged:
                print(
                    f"full-population gradient converged for {convergence_count} "
                    "consecutive updates",
                    flush=True,
                )
                break
        phase = "trained"

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
                        frozen_policies=setup["frozen_policies"],
                    )
                objective_pnl = pnl
                if baseline_pnl is not None:
                    objective_pnl = pnl - torch.as_tensor(
                        np.array(baseline_pnl[path_offset:last], copy=True),
                        dtype=pnl.dtype,
                        device=pnl.device,
                    )
                if claim_pnl is not None:
                    objective_pnl = objective_pnl + torch.as_tensor(
                        np.array(claim_pnl[path_offset:last], copy=True),
                        dtype=pnl.dtype,
                        device=pnl.device,
                    )
                density = torch.as_tensor(
                    selected_weights[path_offset:last],
                    dtype=torch.float64,
                    device=pnl.device,
                )
                loss = weighted_log_oce(objective_pnl.double(), density)
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
            baseline_pnl=baseline_pnl,
            claim_pnl=claim_pnl,
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
            "reference_measure": "the supplied normalized empirical weights on the fixed one-million paths",
            "policy_input": "current Direct-200 hidden state h and eleven current option prices for one expiry",
            "policy_output": (
                "separate nonnegative long and short legs for 11 options plus spot "
                "per expiry; spot logits are aggregated before the positive-part "
                "map, net action is long minus short, and costs/cap use gross legs"
                if arguments.long_short_actions
                else "shared 11 option actions plus one spot action per expiry; four spot heads are summed"
            ),
            "objective": (
                "fixed-policy -log E_Q exp(-net PnL); policy weights unchanged"
                if arguments.valuation_only
                else (
                    "sup_theta -log E_Q exp(-(claim PnL + sum_t["
                    "a_theta(h_t,prices_t)'R_t-cost'abs(a_theta)]))"
                    if arguments.claim_pnl is not None
                    else "sup_theta -log E_Q exp(-sum_t[a_theta(h_t,prices_t)'R_t-cost'abs(a_theta)])"
                )
            ),
            "scope": (
                "in-sample valuation of a supplied fixed policy"
                if arguments.valuation_only
                else "in-sample conditional-strategy stress test of the fixed empirical Q"
            ),
        },
        "arguments": _serialized_arguments(arguments),
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
