"""Parallel fixed-notional direct-utility policies for a shrinking DLV tower."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


def powerexp_utility(value: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the power-exponential utility without inactive-branch overflow.

    .. math::

        u(z)=1-e^{-z}\quad(z>0),\qquad u(z)=z-z^2/2\quad(z\leq0).
    """

    positive = torch.clamp_min(value, 0.0)
    negative = torch.clamp_max(value, 0.0)
    return torch.where(
        value > 0.0,
        -torch.expm1(-positive),
        negative - 0.5 * negative.square(),
    )


def log_powerexp_marginal(value: torch.Tensor) -> torch.Tensor:
    """Return log marginal utility without evaluating an unstable branch."""

    nonpositive = torch.clamp_max(value, 0.0)
    return torch.where(value > 0.0, -value, torch.log1p(-nonpositive))


def normalized_powerexp_marginal(value: torch.Tensor) -> torch.Tensor:
    r"""Return positive path weights with sample mean one for every head."""

    if value.ndim != 2 or value.shape[1] < 2:
        raise ValueError("direct-utility P&L must have shape (heads, paths)")
    log_weight = log_powerexp_marginal(value)
    log_weight = log_weight - torch.logsumexp(log_weight, dim=1, keepdim=True)
    log_weight = log_weight + math.log(value.shape[1])
    return torch.exp(log_weight)


class BatchedIndependentMlp(nn.Module):
    """An independent MLP per notional, evaluated as one tensor program."""

    def __init__(
        self,
        *,
        num_heads: int,
        input_size: int,
        output_size: int,
        hidden_width: int,
        hidden_layers: int,
        generator: torch.Generator,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        if min(num_heads, input_size, output_size, hidden_width, hidden_layers) <= 0:
            raise ValueError("all batched MLP dimensions must be positive")
        sizes = (input_size,) + hidden_layers * (hidden_width,) + (output_size,)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer, (left, right) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            weight = torch.empty(
                num_heads,
                left,
                right,
                dtype=dtype,
                device=device,
            )
            bias = torch.zeros(num_heads, right, dtype=dtype, device=device)
            if layer == len(sizes) - 2:
                weight.normal_(mean=0.0, std=1.0e-6, generator=generator)
            else:
                bound = math.sqrt(6.0 / (left + right))
                weight.uniform_(-bound, bound, generator=generator)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return values with shape ``(heads, paths, outputs)``."""

        if inputs.ndim != 2 or inputs.shape[1] != self.weights[0].shape[1]:
            raise ValueError("batched MLP input has the wrong shape")
        hidden = torch.einsum("bi,mio->mbo", inputs, self.weights[0])
        hidden = torch.nn.functional.silu(hidden + self.biases[0][:, None])
        for weight, bias in zip(self.weights[1:-1], self.biases[1:-1], strict=True):
            hidden = torch.einsum("mbi,mio->mbo", hidden, weight)
            hidden = torch.nn.functional.silu(hidden + bias[:, None])
        return torch.einsum("mbi,mio->mbo", hidden, self.weights[-1]) + self.biases[-1][:, None]


@dataclass(frozen=True)
class DirectUtilityForward:
    """Policy P&L and action diagnostics for one common path batch."""

    normalized_pnl: torch.Tensor
    leveraged_pnl: torch.Tensor
    action_penalty_by_head: torch.Tensor
    action_l1_by_time: tuple[torch.Tensor, ...]


class DirectUtilityTowerPolicy(nn.Module):
    """Independent direct-utility policies for several fixed notionals."""

    def __init__(
        self,
        *,
        notionals: Sequence[float],
        feature_means: Sequence[torch.Tensor],
        feature_scales: Sequence[torch.Tensor],
        output_sizes: Sequence[int],
        hidden_width: int,
        hidden_layers: int,
        seed: int,
        train_hidden_layers: bool = True,
    ) -> None:
        super().__init__()
        if not notionals or any(not math.isfinite(value) or value <= 0.0 for value in notionals):
            raise ValueError("notionals must be non-empty and finite-positive")
        if not (len(feature_means) == len(feature_scales) == len(output_sizes) == 4):
            raise ValueError("the shrinking tower needs four policy dates")
        dtype = feature_means[0].dtype
        device = feature_means[0].device
        if any(
            mean.ndim != 1
            or scale.shape != mean.shape
            or mean.dtype != dtype
            or mean.device != device
            for mean, scale in zip(feature_means, feature_scales, strict=True)
        ):
            raise ValueError("feature statistics must be aligned vectors")
        self.register_buffer("notionals", torch.as_tensor(notionals, dtype=dtype, device=device))
        generator = torch.Generator(device=device).manual_seed(seed)
        self.policies = nn.ModuleList()
        for time_index, (mean, scale, output_size) in enumerate(
            zip(feature_means, feature_scales, output_sizes, strict=True)
        ):
            self.register_buffer(f"feature_mean_{time_index}", mean)
            self.register_buffer(f"feature_scale_{time_index}", scale)
            self.policies.append(
                BatchedIndependentMlp(
                    num_heads=len(notionals),
                    input_size=len(mean),
                    output_size=output_size,
                    hidden_width=hidden_width,
                    hidden_layers=hidden_layers,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
            )
        if not train_hidden_layers:
            for policy in self.policies:
                for parameter in (*policy.weights[:-1], *policy.biases[:-1]):
                    parameter.requires_grad_(False)

    def forward(
        self,
        features_by_time: Sequence[torch.Tensor],
        returns_by_time: Sequence[torch.Tensor],
    ) -> DirectUtilityForward:
        """Evaluate every notional head on the same path batch."""

        if len(features_by_time) != 4 or len(returns_by_time) != 4:
            raise ValueError("direct utility requires four feature/return slabs")
        paths = len(features_by_time[0])
        actual_pnl = torch.zeros(
            len(self.notionals),
            paths,
            dtype=self.notionals.dtype,
            device=self.notionals.device,
        )
        penalty = torch.zeros_like(actual_pnl[:, 0])
        action_norms: list[torch.Tensor] = []
        for time_index, (policy, features, returns) in enumerate(
            zip(self.policies, features_by_time, returns_by_time, strict=True)
        ):
            if len(features) != paths or returns.shape != (
                paths,
                policy.biases[-1].shape[-1],
            ):
                raise ValueError("policy features and returns are not path-aligned")
            mean = getattr(self, f"feature_mean_{time_index}")
            scale = getattr(self, f"feature_scale_{time_index}")
            standardized = (features - mean) / scale
            raw_action = policy(standardized)
            raw_l1 = torch.sum(torch.abs(raw_action), dim=-1)
            radius = self.notionals[:, None]
            action = (
                raw_action
                * radius[..., None]
                / torch.maximum(
                    raw_l1,
                    radius,
                )[..., None]
            )
            actual_pnl = actual_pnl + torch.einsum("mbi,bi->mb", action, returns)
            penalty = penalty + torch.mean(torch.relu(raw_l1 - radius).square(), dim=1)
            action_norms.append(torch.sum(torch.abs(action), dim=-1))
        return DirectUtilityForward(
            normalized_pnl=actual_pnl / self.notionals[:, None],
            leveraged_pnl=actual_pnl,
            action_penalty_by_head=penalty,
            action_l1_by_time=tuple(action_norms),
        )


@dataclass(frozen=True)
class DirectUtilityLoss:
    """Stable loss and population diagnostics for one path batch."""

    loss: torch.Tensor
    surrogate_by_head: torch.Tensor
    direct_utility_by_head: torch.Tensor
    effective_sample_size_by_head: torch.Tensor
    maximum_weight_by_head: torch.Tensor
    density_mass_residual: torch.Tensor
    forward: DirectUtilityForward


def direct_utility_loss(
    forward: DirectUtilityForward,
    *,
    action_penalty: float,
    head_gradient_scales: torch.Tensor | None = None,
) -> DirectUtilityLoss:
    """Return the normalized-marginal surrogate for direct utility.

    A finite-positive scale may precondition each independent head. Since the
    heads share no trainable parameters, this changes only their numerical
    step sizes and leaves every head's stationary condition unchanged.
    """

    if not math.isfinite(action_penalty) or action_penalty < 0.0:
        raise ValueError("action_penalty must be finite and nonnegative")
    weights = normalized_powerexp_marginal(forward.leveraged_pnl)
    # This has the direct-utility gradient divided by the positive scalar
    # N E[u'(N Z)] for each independent head. Recompute weights every update,
    # but detach them inside this first-order envelope surrogate.
    surrogate = -torch.mean(
        weights.detach() * forward.normalized_pnl,
        dim=1,
    )
    loss_by_head = surrogate + action_penalty * forward.action_penalty_by_head
    if head_gradient_scales is not None:
        if head_gradient_scales.shape != loss_by_head.shape or not bool(
            torch.all(torch.isfinite(head_gradient_scales) & (head_gradient_scales > 0.0))
        ):
            raise ValueError("head gradient scales must be one finite-positive value per head")
        loss_by_head = loss_by_head * head_gradient_scales
    mass_residual = torch.max(torch.abs(torch.mean(weights, dim=1) - 1.0))
    effective_sample_size = weights.shape[1] ** 2 / torch.sum(weights.square(), dim=1)
    return DirectUtilityLoss(
        loss=torch.mean(loss_by_head),
        surrogate_by_head=surrogate.detach(),
        direct_utility_by_head=torch.mean(powerexp_utility(forward.leveraged_pnl), dim=1).detach(),
        effective_sample_size_by_head=effective_sample_size.detach(),
        maximum_weight_by_head=torch.amax(weights, dim=1).detach(),
        density_mass_residual=mass_residual.detach(),
        forward=forward,
    )


__all__ = [
    "BatchedIndependentMlp",
    "DirectUtilityForward",
    "DirectUtilityLoss",
    "DirectUtilityTowerPolicy",
    "direct_utility_loss",
    "log_powerexp_marginal",
    "normalized_powerexp_marginal",
    "powerexp_utility",
]
