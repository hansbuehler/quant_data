"""Expiry-separated conditional-arbitrage models and diagnostics.

Each expiry is an independent market containing its matched spot instrument
and the 20 option strikes.  No action or objective combines instruments from
different expiries.  Within one expiry the path P&L is

``X_e = sum_t a_e(t, x_t)' R_e(t) - |a_e(t, x_t)|' c_e(t)``.

The helpers in this module implement three searches for the action ``a_e``:

* a generic lower bound over the complete piecewise-linear terminal-payoff
  cone;
* maximum mean subject to a zero negative-part constraint; and
* a power-exponential optimized certainty equivalent (OCE).

All searches use the same radial L1 map.  An exterior quadratic penalty acts
on the unconstrained output, so an output outside the unit ball retains a
radial gradient back towards feasibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from scipy.optimize import linprog
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility

ExpiryArbitrageMethod = Literal["payoff-cone", "negative-part", "powerexp-oce"]


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class ExpiryArbitrageConfig:
    """Logical configuration of an expiry-separated conditional policy."""

    label: str
    state_features: int
    trading_times: int = 10
    expiries: int = 7
    instruments_per_expiry: int = 21
    group: str = ""
    user_version: str = "0.1"
    method: ExpiryArbitrageMethod = "negative-part"
    hedge: Literal["raw", "delta"] = "delta"
    cost_rate: float = 1.0e-4
    lmbda: float = 1.0
    raw_norm_penalty: float = 1.0
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=128,
        activation="LeakyReLU",
        use_bias=True,
    )

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.label, str) and is_filename(self.label),
            lambda: f"'label' must be a filename, got {self.label!r}",
        )
        verify_inp(
            isinstance(self.group, str) and not self.group.startswith("/"),
            lambda: f"'group' must be relative, got {self.group!r}",
        )
        verify_inp(
            self.method in {"payoff-cone", "negative-part", "powerexp-oce"},
            lambda: f"unknown method {self.method!r}",
        )
        verify_inp(self.hedge in {"raw", "delta"}, lambda: f"unknown hedge {self.hedge!r}")
        for name in ("state_features", "trading_times", "expiries", "instruments_per_expiry"):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: f"'{name}' must be positive, got {value!r}",
            )
        verify_inp(
            self.instruments_per_expiry >= 2,
            "each expiry requires one spot and at least one option instrument",
        )
        for name in ("lmbda", "raw_norm_penalty"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite-positive, got {value}",
            )
        verify_inp(
            math.isfinite(self.cost_rate) and self.cost_rate >= 0.0,
            lambda: f"'cost_rate' must be finite and nonnegative, got {self.cost_rate}",
        )

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class ExpiryArbitrageTrainConfig:
    """Operational settings which may warm-start the same logical model."""

    batch_size: int = 4_096
    valuation_batch_size: int = 16_384
    epochs: int = 30_000
    minimum_epochs: int = 5_000
    value_update: int = 500
    checkpoint_update: int = 2_000
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    compile: bool = True
    augmented_rho: float = 100.0
    feasibility_tolerance: float = 1.0e-10
    convergence_points: int = 8
    convergence_relative_tolerance: float = 2.0e-4
    seed: int = 82_061

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "valuation_batch_size",
            "epochs",
            "minimum_epochs",
            "value_update",
            "checkpoint_update",
            "convergence_points",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: f"'{name}' must be positive, got {value!r}",
            )
        verify_inp(
            self.minimum_epochs <= self.epochs,
            "minimum epochs cannot exceed maximum epochs",
        )
        for name in ("lr", "augmented_rho", "convergence_relative_tolerance"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite-positive, got {value}",
            )
        verify_inp(
            math.isfinite(self.weight_decay) and self.weight_decay >= 0.0,
            lambda: f"invalid weight decay {self.weight_decay}",
        )
        verify_inp(
            math.isfinite(self.feasibility_tolerance) and self.feasibility_tolerance >= 0.0,
            lambda: f"invalid feasibility tolerance {self.feasibility_tolerance}",
        )


@version("1.0")
class ExpiryActionNetwork(BaseModel):
    """One independent action network per expiry with a radial L1 budget."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ExpiryArbitrageConfig,
        state_mean: Tensor,
        state_scale: Tensor,
    ) -> None:
        super().__init__(env)
        expected = (config.state_features,)
        verify_inp(
            state_mean.shape == state_scale.shape == expected,
            lambda: (
                f"state normalization must have shape {expected}, found "
                f"{state_mean.shape}/{state_scale.shape}"
            ),
        )
        if not torch.isfinite(state_mean).all().item():
            raise ValueError("state mean contains NaN or infinity")
        if not torch.isfinite(state_scale).all().item() or not torch.all(state_scale > 0.0).item():
            raise ValueError("state scale must be finite-positive")
        self.config = config
        num_input = config.state_features + config.trading_times
        self.networks = torch.nn.ModuleList(
            FeedForward(
                env,
                num_input=num_input,
                num_output=config.instruments_per_expiry,
                config=config.network_config,
            )
            for _ in range(config.expiries)
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_scale", state_scale.detach().clone())
        self.register_buffer(
            "time_basis",
            torch.eye(config.trading_times, dtype=env.fdtype, device=env.device),
        )
        self.zero_output_layers()

    @torch.no_grad()
    def zero_output_layers(self) -> None:
        """Start at no trade while retaining gradients in the final maps."""

        for network in self.networks:
            output_layer = network.In[-1] if network.In else network.I0
            output_layer.weight.zero_()
            if output_layer.bias is not None:
                output_layer.bias.zero_()

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        expected = (
            state.ndim == 3
            and state.shape[1] == self.config.trading_times
            and state.shape[2] == self.config.state_features
        )
        verify_inp(expected, lambda: f"state has invalid shape {state.shape}")
        normalized = (state - self.state_mean) / self.state_scale
        time = self.time_basis.unsqueeze(0).expand(state.shape[0], -1, -1)
        inputs = torch.cat((normalized, time), dim=-1)
        raw = torch.stack([network(inputs) for network in self.networks], dim=2)
        raw_l1 = raw.abs().sum(dim=-1)
        action = raw / raw_l1.clamp_min(1.0).unsqueeze(-1)
        self.verify(lambda: torch.isfinite(action), "expiry action contains NaN or infinity")
        self.verify(lambda: torch.isfinite(raw_l1), "raw action norm contains NaN or infinity")
        return action, raw_l1


@version("1.0", dependencies=[ExpiryActionNetwork, ExpiryArbitrageConfig, torch_utility])
class ExpiryArbitrageModel(BaseModel):
    """Expiry-separated action model with optional OCE cash offsets."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ExpiryArbitrageConfig,
        state_mean: Tensor,
        state_scale: Tensor,
    ) -> None:
        super().__init__(env)
        self.config = config
        self.action = ExpiryActionNetwork(
            env,
            config=config,
            state_mean=state_mean,
            state_scale=state_scale,
        )
        self.cash_offset = torch.nn.Parameter(
            torch.zeros(config.expiries, dtype=env.fdtype, device=env.device),
            requires_grad=config.method == "powerexp-oce",
        )
        self.utility, self.utility_derivative, self.utility_label = torch_utility(
            "powexp", config.lmbda
        )

    def actions(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return feasible actions, raw norms, and the exterior cap penalty."""

        action, raw_l1 = self.action(state)
        cap_penalty = torch.relu(raw_l1 - 1.0).square()
        return action, raw_l1, cap_penalty

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Delegate module calls to :meth:`actions` for ``torch.compile``."""

        return self.actions(state)


def expiry_portfolio_pnl(action: Tensor, returns: Tensor, costs: Tensor) -> tuple[Tensor, Tensor]:
    """Return trading-time and path P&L for expiry-separated actions."""

    verify_inp(
        action.shape == returns.shape == costs.shape,
        lambda: f"action/return/cost shapes differ: {action.shape}/{returns.shape}/{costs.shape}",
    )
    time_pnl = (action * returns - action.abs() * costs).sum(dim=-1)
    path_pnl = time_pnl.sum(dim=1)
    return time_pnl, path_pnl


def propagated_expiry_band(
    action: Tensor,
    returns: Tensor,
    costs: Tensor,
    instrument_band: Tensor,
    *,
    roundoff_multiplier: float = 32.0,
) -> tuple[Tensor, Tensor]:
    """Propagate float32 instrument bands through time and each expiry."""

    verify_inp(
        action.shape == returns.shape == costs.shape == instrument_band.shape,
        "basket-band inputs have incompatible shapes",
    )
    absolute_action = action.abs()
    eps32 = torch.finfo(torch.float32).eps
    time_band = (absolute_action * instrument_band).sum(dim=-1) + (
        roundoff_multiplier
        * eps32
        * (absolute_action * (returns.abs() + costs)).sum(dim=-1)
    )
    return time_band, time_band.sum(dim=1)


def payoff_cone_values(
    action: Tensor,
    *,
    spot: Tensor,
    option_price: Tensor,
    delta: Tensor,
    cash_strike: Tensor,
    costs: Tensor,
    hedge: Literal["raw", "delta"],
) -> tuple[Tensor, Tensor]:
    r"""Return payoff at every kink and the upper-tail portfolio slope.

    Terminal spot is restricted to ``[0, infinity)``.  The P&L is piecewise
    linear, so its finite minimum is attained at zero or at a cash strike,
    provided the upper-tail slope is nonnegative.  The returned scenario axis
    contains exactly those points.  The first instrument is the matched spot;
    the remaining instruments are calls ordered by strike.
    """

    verify_inp(hedge in {"raw", "delta"}, lambda: f"unknown hedge {hedge!r}")
    B, T, E, K = option_price.shape
    expected_action = (B, T, E, K + 1)
    verify_inp(
        action.shape == costs.shape == expected_action,
        lambda: f"expected action/cost shape {expected_action}, found {action.shape}/{costs.shape}",
    )
    verify_inp(
        delta.shape == cash_strike.shape == option_price.shape,
        "option primitive shapes differ",
    )
    verify_inp(spot.shape == (B, T), lambda: f"spot has invalid shape {spot.shape}")
    zero = torch.zeros((B, T, E, 1), dtype=spot.dtype, device=spot.device)
    terminal_spot = torch.cat((zero, cash_strike), dim=-1)
    y = terminal_spot.unsqueeze(-1)
    strike = cash_strike.unsqueeze(-2)
    option_return = torch.relu(y - strike) - option_price.unsqueeze(-2)
    if hedge == "delta":
        option_return = option_return - delta.unsqueeze(-2) * (
            y - spot[:, :, None, None, None]
        )
        option_slope = 1.0 - delta
    else:
        option_slope = torch.ones_like(delta)
    spot_return = terminal_spot - spot[:, :, None, None]
    scenario_return = torch.cat((spot_return.unsqueeze(-1), option_return), dim=-1)
    scenario_pnl = (action.unsqueeze(-2) * scenario_return).sum(dim=-1) - (
        action.abs() * costs
    ).sum(dim=-1, keepdim=True)
    tail_slope = action[..., 0] + (action[..., 1:] * option_slope).sum(dim=-1)
    return scenario_pnl, tail_slope


def solve_payoff_cone_lp(
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Literal["raw", "delta"],
) -> tuple[np.ndarray, float]:
    r"""Solve one observable-state payoff-cone separation problem exactly.

    The linear program maximizes the worst terminal P&L over all spot values
    in ``[0,infinity)`` subject to an L1 action budget.  Long and short
    notionals are separate nonnegative variables.  No realized future return
    is used.
    """

    verify_inp(hedge in {"raw", "delta"}, lambda: f"unknown hedge {hedge!r}")
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    K = option_price.size
    verify_inp(
        option_price.shape == delta.shape == cash_strike.shape == (K,)
        and costs.shape == (K + 1,),
        "payoff-cone LP arrays have incompatible shapes",
    )
    values = (spot, *option_price, *delta, *cash_strike, *costs)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("payoff-cone LP input contains NaN or infinity")
    verify_inp(spot > 0.0, lambda: f"spot must be positive, got {spot}")
    verify_inp(np.all(costs >= 0.0), "payoff-cone LP costs must be nonnegative")
    terminal_spot = np.concatenate(([0.0], cash_strike))
    option_return = np.maximum(
        terminal_spot[:, None] - cash_strike[None, :], 0.0
    ) - option_price[None, :]
    if hedge == "delta":
        option_return -= delta[None, :] * (terminal_spot[:, None] - spot)
        option_slope = 1.0 - delta
    else:
        option_slope = np.ones_like(delta)
    scenario_return = np.concatenate(
        ((terminal_spot - spot)[:, None], option_return), axis=1
    )
    instrument_slope = np.concatenate(([1.0], option_slope))
    J = K + 1

    # Variables are (long[J], short[J], worst_margin).  For every terminal
    # kink, worst_margin <= (long-short)'return-(long+short)'cost.
    A_scenario = np.concatenate(
        (
            -(scenario_return - costs[None, :]),
            -(-scenario_return - costs[None, :]),
            np.ones((terminal_spot.size, 1)),
        ),
        axis=1,
    )
    # A negative upper-tail slope makes the terminal loss unbounded.
    A_tail = np.concatenate(
        (-instrument_slope, instrument_slope, np.asarray([0.0]))
    )[None, :]
    A_budget = np.concatenate((np.ones(2 * J), np.asarray([0.0])))[None, :]
    A_ub = np.concatenate((A_scenario, A_tail, A_budget), axis=0)
    b_ub = np.concatenate(
        (np.zeros(terminal_spot.size + 1), np.asarray([1.0]))
    )
    objective = np.zeros(2 * J + 1)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * (2 * J) + [(None, None)],
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"payoff-cone LP failed with status {result.status}: {result.message}"
        )
    action = result.x[:J] - result.x[J : 2 * J]
    margin = float(result.x[-1])
    if not np.all(np.isfinite(action)) or not math.isfinite(margin):
        raise FloatingPointError("payoff-cone LP solution contains NaN or infinity")
    verify_inp(
        float(np.abs(action).sum()) <= 1.0 + 1.0e-8,
        "payoff-cone LP violated its L1 budget",
    )
    return action, margin


def payoff_cone_numerical_band(
    action: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Literal["raw", "delta"],
    roundoff_multiplier: float = 32.0,
) -> float:
    """Propagate float32 operand error through one payoff-cone action."""

    action = np.asarray(action, dtype=np.float64)
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    K = option_price.size
    verify_inp(
        action.shape == costs.shape == (K + 1,)
        and delta.shape == cash_strike.shape == (K,),
        "payoff-cone numerical-band arrays have incompatible shapes",
    )
    terminal_spot = np.concatenate(([0.0], cash_strike))
    option_return = np.maximum(
        terminal_spot[:, None] - cash_strike[None, :], 0.0
    ) - option_price[None, :]
    if hedge == "delta":
        option_return -= delta[None, :] * (terminal_spot[:, None] - spot)
    elif hedge != "raw":
        raise ValueError(f"unknown hedge {hedge!r}")
    scenario_return = np.concatenate(
        ((terminal_spot - spot)[:, None], option_return), axis=1
    )
    unreduced_scale = np.max(np.abs(scenario_return), axis=0) + costs
    band = (
        roundoff_multiplier
        * np.finfo(np.float32).eps
        * max(1.0, float(np.sum(np.abs(action) * unreduced_scale)))
    )
    if not math.isfinite(band) or band <= 0.0:
        raise FloatingPointError("payoff-cone numerical band is not finite-positive")
    return band


def exact_payoff_cone_lower_bound(
    action: Tensor,
    *,
    spot: Tensor,
    option_price: Tensor,
    delta: Tensor,
    cash_strike: Tensor,
    costs: Tensor,
    hedge: Literal["raw", "delta"],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return time/path worst-case bounds and upper-tail slopes."""

    values, tail_slope = payoff_cone_values(
        action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    time_worst = values.min(dim=-1).values
    # A negative upper-tail slope means the true lower bound is minus infinity.
    slope_tolerance = 32.0 * torch.finfo(action.dtype).eps * action.abs().sum(dim=-1)
    time_worst = torch.where(
        tail_slope >= -slope_tolerance,
        time_worst,
        torch.full_like(time_worst, -torch.inf),
    )
    return time_worst, time_worst.sum(dim=1), tail_slope


def smooth_payoff_cone_lower_bound(
    action: Tensor,
    *,
    spot: Tensor,
    option_price: Tensor,
    delta: Tensor,
    cash_strike: Tensor,
    costs: Tensor,
    hedge: Literal["raw", "delta"],
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiable, zero-preserving soft minimum used only during fitting."""

    verify_inp(
        math.isfinite(temperature) and temperature > 0.0,
        lambda: f"temperature must be finite-positive, got {temperature}",
    )
    values, tail_slope = payoff_cone_values(
        action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    smooth = -temperature * torch.logsumexp(-values / temperature, dim=-1) + (
        temperature * math.log(values.shape[-1])
    )
    # The spot level converts a negative dimensionless slope into P&L units.
    tail_penalty = spot[:, :, None].clamp_min(torch.finfo(spot.dtype).tiny) * torch.relu(
        -tail_slope
    )
    time_worst = smooth - tail_penalty
    return time_worst, time_worst.sum(dim=1), tail_slope


def weighted_expiry_mean(value: Tensor, path_weight: Tensor) -> Tensor:
    """Return ``N**-1 sum_r (p/q)_r value_r`` for every expiry."""

    verify_inp(
        value.ndim >= 2 and value.shape[0] == path_weight.shape[0],
        lambda: f"value/weight shapes are incompatible: {value.shape}/{path_weight.shape}",
    )
    verify_inp(path_weight.ndim == 1, lambda: f"weights have invalid shape {path_weight.shape}")
    reduce_dims = tuple(range(1, value.ndim - 1))
    if reduce_dims:
        value = value.mean(dim=reduce_dims)
    return (path_weight[:, None] * value).mean(dim=0)


def powerexp_oce_by_expiry(
    model: ExpiryArbitrageModel,
    path_pnl: Tensor,
    path_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return weighted OCE values and both first-order residuals by expiry."""

    verify_inp(
        path_pnl.shape == (path_weight.shape[0], model.config.expiries),
        lambda: f"P&L has invalid shape {path_pnl.shape}",
    )
    utility = model.utility(path_pnl, model.cash_offset)
    marginal = model.utility_derivative(path_pnl, model.cash_offset)
    # ``torch_utility`` includes ``-y`` inside each observation.  Under
    # importance sampling the empirical OCE is N^-1 sum(w*u0(X+y))-y, so the
    # finite-sample mass correction below keeps the deterministic ``-y``
    # outside the weighted expectation.
    mass = path_weight.mean()
    value = weighted_expiry_mean(utility, path_weight) + (
        mass - 1.0
    ) * model.cash_offset
    y_residual = weighted_expiry_mean(marginal, path_weight) - 1.0
    return value, y_residual, marginal


__all__ = [
    "ExpiryActionNetwork",
    "ExpiryArbitrageConfig",
    "ExpiryArbitrageMethod",
    "ExpiryArbitrageModel",
    "ExpiryArbitrageTrainConfig",
    "exact_payoff_cone_lower_bound",
    "expiry_portfolio_pnl",
    "payoff_cone_values",
    "payoff_cone_numerical_band",
    "powerexp_oce_by_expiry",
    "propagated_expiry_band",
    "smooth_payoff_cone_lower_bound",
    "solve_payoff_cone_lp",
    "weighted_expiry_mean",
]
