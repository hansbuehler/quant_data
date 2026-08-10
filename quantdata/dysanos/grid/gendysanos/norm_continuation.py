"""Fixed-radius continuation models for conditional-arbitrage searches.

For every trading time, one network jointly trades all matched expiries and
instruments.  Its unconstrained output ``v_t(x_t)`` is projected radially onto
an L1 ball of radius ``n``.  The utility sees only the feasible action, while
an exterior quadratic penalty supplies a radial recovery gradient whenever
the unconstrained output leaves the ball.
"""

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
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

from .arb_tracing import tracing_utility_objective

NormContinuationUtility = Literal["powerexp", "entropy"]


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class NormContinuationConfig:
    """Logical geometry and objective of one radius-continuation search."""

    label: str
    source_cache_uid: str
    state_features: int
    risk_aversion: float
    utility: NormContinuationUtility
    radii: tuple[float, ...]
    exterior_penalty: float
    trading_times: int = 10
    expiries: int = 7
    instruments_per_expiry: int = 21
    covariance_relative_floor: float = 1.0e-10
    group: str = ""
    user_version: str = "0.1"
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=96,
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
            self.utility in {"powerexp", "entropy"},
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
            bool(self.radii)
            and all(math.isfinite(radius) and radius > 0.0 for radius in self.radii)
            and all(left < right for left, right in zip(self.radii, self.radii[1:])),
            lambda: f"radii must be finite-positive and strictly increasing: {self.radii!r}",
        )
        for name in ("risk_aversion", "exterior_penalty", "covariance_relative_floor"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: (
                    f"'{name}' must be finite-positive, got {value!r}"
                ),
            )

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class NormContinuationTrainConfig:
    """Operational settings shared by all continuation radii."""

    batch_size: int = 4_096
    valuation_batch_size: int = 16_384
    epochs_per_radius: int = 10_000
    minimum_epochs_per_radius: int = 1_500
    value_update: int = 250
    checkpoint_update: int = 2_000
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    compile: bool = True
    convergence_points: int = 10
    convergence_relative_tolerance: float = 1.0e-4
    seed: int = 94_103

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "valuation_batch_size",
            "epochs_per_radius",
            "minimum_epochs_per_radius",
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
            self.minimum_epochs_per_radius <= self.epochs_per_radius,
            "minimum epochs per radius cannot exceed the maximum",
        )
        for name in ("lr", "convergence_relative_tolerance"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: (
                    f"'{name}' must be finite-positive, got {value!r}"
                ),
            )
        verify_inp(
            math.isfinite(self.weight_decay) and self.weight_decay >= 0.0,
            lambda: f"invalid weight decay {self.weight_decay!r}",
        )


def _radius_tensor(reference: Tensor, radius: float | Tensor) -> Tensor:
    """Return a scalar radius on ``reference``'s device without graph guards."""

    if not isinstance(radius, Tensor):
        verify_inp(
            math.isfinite(radius) and radius > 0.0,
            lambda: f"radius must be finite-positive, got {radius!r}",
        )
    result = torch.as_tensor(radius, dtype=reference.dtype, device=reference.device)
    verify_inp(result.ndim == 0, lambda: f"radius must be scalar, got {result.shape}")
    return result


def radial_l1_clip(
    raw: Tensor, radius: float | Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Project each path/time action radially onto its L1 ball.

    The final two axes are the expiry and within-expiry instrument axes.  The
    returned Boolean tensor identifies outputs in the exterior region where
    the radial utility derivative is zero but tangential derivatives remain.
    """

    verify_inp(raw.ndim >= 3, lambda: f"raw action has invalid shape {raw.shape}")
    radius_tensor = _radius_tensor(raw, radius)
    raw_l1 = raw.abs().sum(dim=(-1, -2))
    scale = radius_tensor / raw_l1.clamp_min(radius_tensor)
    action = raw * scale[..., None, None]
    clipped = raw_l1 > radius_tensor
    return action, raw_l1, clipped


def exterior_l1_penalty(raw_l1: Tensor, radius: float | Tensor) -> Tensor:
    r"""Return :math:`E[(\|v_t\|_1-n)_+^2]` separately by trading time."""

    verify_inp(raw_l1.ndim == 2, lambda: f"raw L1 norm has invalid shape {raw_l1.shape}")
    radius_tensor = _radius_tensor(raw_l1, radius)
    result = torch.relu(raw_l1 - radius_tensor).square().mean(dim=0)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError("exterior L1 penalty produced NaN or infinity")
    return result


def clipped_radius_pnl_derivative(
    pnl: Tensor, clipped: Tensor, radius: float | Tensor
) -> Tensor:
    """Differentiate feasible P&L with respect to a frozen L1 radius.

    Transaction costs are homogeneous in the action.  Thus a clipped action
    and its P&L scale linearly with the radius, while an interior action is
    locally independent of the radius.
    """

    verify_inp(
        pnl.shape == clipped.shape,
        lambda: f"P&L/clipping shapes differ: {pnl.shape}/{clipped.shape}",
    )
    radius_tensor = _radius_tensor(pnl, radius)
    result = torch.where(clipped, pnl / radius_tensor, torch.zeros_like(pnl))
    if not torch.isfinite(result).all().item():
        raise FloatingPointError("radius P&L derivative produced NaN or infinity")
    return result


@version("1.0")
class NormContinuationAction(BaseModel):
    """Independent per-time action networks in physical orthogonal factors."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: NormContinuationConfig,
        state_mean: Tensor,
        state_whitener: Tensor,
    ) -> None:
        super().__init__(env)
        expected_mean = (config.trading_times, config.state_features)
        expected_whitener = (
            config.trading_times,
            config.state_features,
            config.state_features,
        )
        verify_inp(
            state_mean.shape == expected_mean,
            lambda: f"state mean must have shape {expected_mean}, got {state_mean.shape}",
        )
        verify_inp(
            state_whitener.shape == expected_whitener,
            lambda: (
                f"state whitener must have shape {expected_whitener}, "
                f"got {state_whitener.shape}"
            ),
        )
        for name, value in (("state mean", state_mean), ("state whitener", state_whitener)):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains NaN or infinity")
        self.config = config
        instruments = config.expiries * config.instruments_per_expiry
        self.networks = torch.nn.ModuleList(
            FeedForward(
                env,
                num_input=config.state_features,
                num_output=instruments,
                config=config.network_config,
            )
            for _ in range(config.trading_times)
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_whitener", state_whitener.detach().clone())
        self.zero_output_layers()

    @torch.no_grad()
    def zero_output_layers(self) -> None:
        """Start from no trade while retaining the local output derivative."""

        for network in self.networks:
            output_layer = network.In[-1] if network.In else network.I0
            output_layer.weight.zero_()
            if output_layer.bias is not None:
                output_layer.bias.zero_()

    def raw_action(self, state: Tensor) -> Tensor:
        """Return the unconstrained action in whitened state coordinates."""

        expected = (
            state.ndim == 3
            and state.shape[1] == self.config.trading_times
            and state.shape[2] == self.config.state_features
        )
        assert expected, (
            "state must have shape (S,trading_times,state_features), found "
            f"{tuple(state.shape)}"
        )
        centered = state - self.state_mean[None, :, :]
        factor = torch.einsum("btf,tfg->btg", centered, self.state_whitener)
        raw = torch.stack(
            [network(factor[:, time]) for time, network in enumerate(self.networks)],
            dim=1,
        ).reshape(
            state.shape[0],
            self.config.trading_times,
            self.config.expiries,
            self.config.instruments_per_expiry,
        )
        self.verify(lambda: torch.isfinite(factor), "orthogonal state factor is non-finite")
        self.verify(lambda: torch.isfinite(raw), "raw continuation action is non-finite")
        return raw

    def forward(
        self, state: Tensor, radius: float | Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the feasible action, raw L1 norm, and clipping indicator."""

        action, raw_l1, clipped = radial_l1_clip(self.raw_action(state), radius)
        self.verify(lambda: torch.isfinite(action), "projected continuation action is non-finite")
        self.verify(lambda: torch.isfinite(raw_l1), "raw continuation norm is non-finite")
        return action, raw_l1, clipped


def continuation_pnl(action: Tensor, returns: Tensor, costs: Tensor) -> Tensor:
    """Return exact net P&L for each path and trading time."""

    assert action.shape == returns.shape == costs.shape
    pnl = (action * returns - action.abs() * costs).sum(dim=(-1, -2))
    if not torch.isfinite(pnl).all().item():
        raise FloatingPointError("continuation P&L contains NaN or infinity")
    return pnl


def continuation_loss(
    model: NormContinuationAction,
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    *,
    radius: float | Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the penalized loss and its unpenalized components by time."""

    action, raw_l1, clipped = model(state, radius)
    pnl = continuation_pnl(action, returns, costs)
    objective = tracing_utility_objective(
        pnl,
        path_weight,
        utility=model.config.utility,
        risk_aversion=model.config.risk_aversion,
    )
    penalty = exterior_l1_penalty(raw_l1, radius)
    loss = (-objective + model.config.exterior_penalty * penalty).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("norm-continuation objective became NaN or infinity")
    return loss, {
        "objective": objective,
        "penalty": penalty,
        "action_l1_mean": action.abs().sum(dim=(-1, -2)).mean(dim=0),
        "action_l1_maximum": action.abs().sum(dim=(-1, -2)).amax(dim=0),
        "raw_l1_mean": raw_l1.mean(dim=0),
        "raw_l1_maximum": raw_l1.amax(dim=0),
        "clipped_fraction": clipped.to(dtype=state.dtype).mean(dim=0),
    }


__all__ = [
    "NormContinuationAction",
    "NormContinuationConfig",
    "NormContinuationTrainConfig",
    "NormContinuationUtility",
    "clipped_radius_pnl_derivative",
    "continuation_loss",
    "continuation_pnl",
    "exterior_l1_penalty",
    "radial_l1_clip",
]
