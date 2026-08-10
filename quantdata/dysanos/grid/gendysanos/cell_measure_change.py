"""Independent trading-time/expiry policies for utility measure changes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForwardConfig
from quantdata.torch.utility import torch_utility

CellUtility = Literal["powerexp", "exponential", "mean-variance"]


@version("2.0")
@dataclass(frozen=True, kw_only=True)
class CellMeasureChangeConfig:
    """Logical geometry of 70 mutually independent conditional policies."""

    label: str
    source_cache_uid: str
    state_features: int
    risk_aversion: float
    utility: CellUtility
    trading_times: int = 10
    expiries: int = 7
    instruments_per_expiry: int = 21
    group: str = ""
    user_version: str = "0.1"
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=64,
        activation="SiLU",
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
            isinstance(self.source_cache_uid, str) and bool(self.source_cache_uid),
            "'source_cache_uid' must be a non-empty string",
        )
        verify_inp(
            self.utility in {"powerexp", "exponential", "mean-variance"},
            lambda: f"unknown utility {self.utility!r}",
        )
        for name in (
            "state_features",
            "trading_times",
            "expiries",
            "instruments_per_expiry",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        verify_inp(
            math.isfinite(self.risk_aversion) and self.risk_aversion > 0.0,
            lambda: (
                "'risk_aversion' must be finite-positive, got "
                f"{self.risk_aversion}"
            ),
        )

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class CellMeasureChangeTrainConfig:
    """Operational training settings which permit checkpoint warm starts."""

    batch_size: int = 4_096
    valuation_batch_size: int = 16_384
    epochs: int = 20_000
    minimum_epochs: int = 5_000
    value_update: int = 500
    checkpoint_update: int = 2_000
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    compile: bool = True
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
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        verify_inp(
            self.minimum_epochs <= self.epochs,
            "minimum epochs cannot exceed maximum epochs",
        )
        for name in ("lr", "convergence_relative_tolerance"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: (
                    f"'{name}' must be finite-positive, got {value}"
                ),
            )
        verify_inp(
            math.isfinite(self.weight_decay) and self.weight_decay >= 0.0,
            lambda: f"invalid weight decay {self.weight_decay}",
        )


@version("2.0")
class IndependentFeedForward(BaseModel):
    """Vectorized MLPs with a disjoint parameter slice for every cell."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        cells: int,
        num_input: int,
        num_output: int,
        config: FeedForwardConfig,
    ) -> None:
        super().__init__(env)
        verify_inp(cells > 0 and num_input > 0 and num_output > 0, "invalid MLP geometry")
        width = (
            config.num_width
            if config.num_width > 0
            else abs(config.num_width) * num_output
        )
        dimensions = (
            [num_input, num_output]
            if config.num_depth == 0
            else [num_input, *([width] * config.num_depth), num_output]
        )
        self.cells = cells
        self.num_input = num_input
        self.num_output = num_output
        self.activation = config.get_activation()
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for input_size, output_size in zip(dimensions[:-1], dimensions[1:], strict=True):
            weight = torch.nn.Parameter(
                torch.empty(
                    (cells, input_size, output_size),
                    dtype=env.fdtype,
                    device=env.device,
                )
            )
            bias = torch.nn.Parameter(
                torch.empty((cells, output_size), dtype=env.fdtype, device=env.device)
            )
            bound = 1.0 / math.sqrt(input_size)
            torch.nn.init.uniform_(weight, -bound, bound)
            torch.nn.init.uniform_(bias, -bound, bound)
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, value: Tensor) -> Tensor:
        assert value.ndim == 3 and value.shape[1:] == (
            self.cells,
            self.num_input,
        ), (value.shape, self.cells, self.num_input)
        # ``bmm`` executes all independent cell matrices in one GPU launch.
        value = value.transpose(0, 1)
        for layer, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            value = torch.bmm(value, weight) + bias[:, None, :]
            if layer + 1 < len(self.weights):
                value = self.activation(value)
        value = value.transpose(0, 1)
        assert value.shape[1:] == (self.cells, self.num_output)
        self.verify(lambda: torch.isfinite(value), "independent MLP output is non-finite")
        return value


@version("2.0")
class IndependentCellAction(BaseModel):
    """One unbounded, preconditioned network for every ``(t, expiry)`` cell."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: CellMeasureChangeConfig,
        state_mean: Tensor,
        state_scale: Tensor,
        return_rms: Tensor,
    ) -> None:
        super().__init__(env)
        state_shape = (config.trading_times, config.state_features)
        return_shape = (
            config.trading_times,
            config.expiries,
            config.instruments_per_expiry,
        )
        verify_inp(
            state_mean.shape == state_scale.shape == state_shape,
            lambda: (
                f"state normalization must have shape {state_shape}, found "
                f"{state_mean.shape}/{state_scale.shape}"
            ),
        )
        verify_inp(
            return_rms.shape == return_shape,
            lambda: f"return RMS must have shape {return_shape}, found {return_rms.shape}",
        )
        for name, value in (
            ("state_mean", state_mean),
            ("state_scale", state_scale),
            ("return_rms", return_rms),
        ):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains NaN or infinity")
        verify_inp(torch.all(state_scale > 0.0).item(), "state scale must be positive")
        verify_inp(torch.all(return_rms > 0.0).item(), "return RMS must be positive")

        self.config = config
        self.network = IndependentFeedForward(
            env,
            cells=config.trading_times * config.expiries,
            num_input=config.state_features,
            num_output=config.instruments_per_expiry,
            config=config.network_config,
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_scale", state_scale.detach().clone())
        # Multiplication by risk aversion makes the network's raw output scale
        # approximately invariant under the theoretical a -> a/gamma scaling.
        self.register_buffer(
            "action_denominator",
            config.risk_aversion * return_rms.detach().clone(),
        )
        self.zero_output_layers()

    @torch.no_grad()
    def zero_output_layers(self) -> None:
        """Start from no trade without suppressing its first derivative."""

        self.network.weights[-1].zero_()
        self.network.biases[-1].zero_()

    def forward(self, state: Tensor) -> Tensor:
        expected = (
            state.ndim == 3
            and state.shape[1] == self.config.trading_times
            and state.shape[2] == self.config.state_features
        )
        assert expected, (
            "state must have shape (S, trading_times, state_features), found "
            f"{tuple(state.shape)}"
        )
        normalized = (state - self.state_mean[None, :, :]) / self.state_scale[
            None, :, :
        ]
        expanded = normalized[:, :, None, :].expand(
            -1,
            -1,
            self.config.expiries,
            -1,
        )
        raw = self.network(
            expanded.reshape(
                state.shape[0],
                self.config.trading_times * self.config.expiries,
                self.config.state_features,
            )
        ).reshape(
            state.shape[0],
            self.config.trading_times,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )
        action = raw / self.action_denominator[None, :, :, :]
        expected_shape = (
            state.shape[0],
            self.config.trading_times,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )
        assert action.shape == expected_shape, (action.shape, expected_shape)
        self.verify(
            lambda: torch.isfinite(action),
            "independent-cell action contains NaN or infinity",
        )
        return action


def cell_pnl(action: Tensor, returns: Tensor, costs: Tensor) -> Tensor:
    """Return net P&L separately for every path, trading time, and expiry."""

    assert action.shape == returns.shape == costs.shape
    pnl = (action * returns - action.abs() * costs).sum(dim=-1)
    if not torch.isfinite(pnl).all().item():
        raise FloatingPointError("cell P&L contains NaN or infinity")
    return pnl


def cell_utility_and_marginal(
    pnl: Tensor,
    *,
    utility: CellUtility,
    risk_aversion: float,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the weighted objective by cell and the pathwise marginal utility."""

    assert pnl.ndim == 3 and weight.shape == pnl.shape[:1]
    expanded_weight = weight[:, None, None]
    if utility == "mean-variance":
        mean = (expanded_weight * pnl).mean(dim=0)
        centered = pnl - mean[None, :, :]
        variance = (expanded_weight * centered.square()).mean(dim=0)
        value = mean - 0.5 * risk_aversion * variance
        marginal = 1.0 - risk_aversion * centered
    else:
        utility_name = "entropy" if utility == "exponential" else "powexp"
        utility_function, derivative, _ = torch_utility(
            utility_name,
            risk_aversion,
        )
        zero = torch.zeros((), dtype=pnl.dtype, device=pnl.device)
        path_utility = utility_function(pnl, zero)
        marginal = derivative(pnl, zero)
        value = (expanded_weight * path_utility).mean(dim=0)
    if not torch.isfinite(value).all().item() or not torch.isfinite(marginal).all().item():
        raise FloatingPointError(f"{utility} objective or marginal became NaN/infinity")
    return value, marginal


__all__ = [
    "CellMeasureChangeConfig",
    "CellMeasureChangeTrainConfig",
    "CellUtility",
    "IndependentCellAction",
    "IndependentFeedForward",
    "cell_pnl",
    "cell_utility_and_marginal",
]
