"""Regularized conditional-action objectives for drift diagnostics.

This module deliberately does not construct a risk-neutral density.  For

``Y = a(z)' dH - kappa * ||a(z)||**2``

is evaluated through the optimized-certainty-equivalent objective

``sup_{a,y} E[u(Y+y)-y]``.

the first-order condition is

``E[u'(Y) * (dH - 2*kappa*a(z)) | z] = 0``.

Here the OCE first-order condition in ``y`` normalizes marginal utility to one.
The action condition consequently leaves conditional drift
``2*kappa*a(z)`` under that marginal-utility density rather than zero.  The
quadratic term makes the optimization coercive and is useful for learning
conditional drift without relying on an output cap.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.utility import torch_utility

from .torch_drift_change import ScaledTanhNetwork


@version("1.0")
def downside_oce_utility(
    x: Tensor,
    y: Tensor,
    *,
    scale: float,
    power: int,
) -> tuple[Tensor, Tensor]:
    r"""Return ``-scale*(-(x+y))_+**power-y`` and its derivative in ``x``."""

    verify_inp(
        math.isfinite(scale) and scale > 1.0,
        lambda: f"'scale' must be finite and greater than one, got {scale}",
    )
    verify_inp(power >= 1, lambda: f"'power' must be positive, got {power}")
    xbar = x + y
    downside = torch.clamp(-xbar, min=0.0)
    utility = -scale * downside.pow(power) - y
    if power == 1:
        # Select the right derivative zero at the CVaR cutoff.
        marginal = torch.where(
            xbar < 0.0,
            torch.full_like(xbar, scale),
            torch.zeros_like(xbar),
        )
    else:
        marginal = scale * float(power) * downside.pow(power - 1)
    return utility, marginal


@version("1.3", dependencies=[downside_oce_utility, torch_utility])
@dataclass(frozen=True, kw_only=True)
class QuadraticActionConfig:
    """Logical configuration of the regularized action experiment."""

    utility: str = "entropy"
    lmbda: float = 1.0
    downside_power: int = 2
    quadratic_coefficient: float = 0.5
    bounded_action: bool = False
    initial_action_scale: float = 1.0
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=100,
        activation="LeakyReLU",
        use_bias=True,
    )

    def __post_init__(self) -> None:
        verify_inp(
            self.utility in {"entropy", "powexp", "downside"},
            lambda: (
                "'utility' must be 'entropy', 'powexp', or 'downside', got "
                f"{self.utility!r}"
            ),
        )
        verify_inp(
            isinstance(self.downside_power, int) and self.downside_power >= 1,
            lambda: (
                "'downside_power' must be a positive integer, got "
                f"{self.downside_power!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.lmbda) and self.lmbda > 0.0,
            lambda: f"'lmbda' must be finite and positive, got {self.lmbda}",
        )
        verify_inp(
            math.isfinite(self.quadratic_coefficient)
            and self.quadratic_coefficient > 0.0,
            lambda: (
                "'quadratic_coefficient' must be finite and positive, got "
                f"{self.quadratic_coefficient}"
            ),
        )
        verify_inp(
            math.isfinite(self.initial_action_scale)
            and self.initial_action_scale > 0.0,
            lambda: (
                "'initial_action_scale' must be finite and positive, got "
                f"{self.initial_action_scale}"
            ),
        )


@version("1.1")
class DirectActionNetwork(BaseModel):
    """Unbounded feed-forward action with a neutral zero initialization."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        num_input: int,
        num_output: int,
        config: FeedForwardConfig,
    ) -> None:
        super().__init__(env)
        self.network = FeedForward(
            env,
            num_input=num_input,
            num_output=num_output,
            config=config,
        )
        self.zero_output_layer()

    @torch.no_grad()
    def zero_output_layer(self) -> None:
        """Set the last affine map to zero without changing hidden features."""

        output_layer = (
            self.network.In[-1]
            if len(self.network.In) > 0
            else self.network.I0
        )
        output_layer.weight.zero_()
        if output_layer.bias is not None:
            output_layer.bias.zero_()

    def forward(self, z: Tensor) -> Tensor:
        action = self.network(z)
        self.verify(lambda: torch.isfinite(action), "direct action is non-finite")
        return action


@version(
    "1.3",
    dependencies=[
        DirectActionNetwork,
        QuadraticActionConfig,
        ScaledTanhNetwork,
    ],
)
class QuadraticActionModel(BaseModel):
    """Conditional portfolio with a quadratic position charge inside utility."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        num_input: int,
        num_instruments: int,
        config: QuadraticActionConfig,
    ) -> None:
        super().__init__(env)
        verify_inp(num_input > 0, "'num_input' must be positive")
        verify_inp(num_instruments > 0, "'num_instruments' must be positive")
        self.config = config
        if config.bounded_action:
            self.action = ScaledTanhNetwork(
                env,
                num_input=num_input,
                num_output=num_instruments,
                config=config.network_config,
                initial_scale=config.initial_action_scale,
                scale_per_output=True,
            )
            self.action.zero_output_layer()
        else:
            self.action = DirectActionNetwork(
                env,
                num_input=num_input,
                num_output=num_instruments,
                config=config.network_config,
            )
        self.cash_offset = torch.nn.Parameter(
            torch.zeros((), dtype=env.fdtype, device=env.device)
        )
        if config.utility == "downside":
            self.utility = None
            self.utility_derivative = None
            self.utility_label = (
                f"downside-oce-p{config.downside_power}@{config.lmbda:.2g}"
            )
        else:
            self.utility, self.utility_derivative, self.utility_label = torch_utility(
                config.utility,
                config.lmbda,
            )

    def outputs(self, data: Mapping[str, Tensor]) -> dict[str, Tensor]:
        z, dH = data["z"], data["dH"]
        verify_inp(
            z.shape[:-1] == dH.shape[:-1],
            lambda: f"state/return leading shapes differ: {z.shape} and {dH.shape}",
        )
        action = self.action(z)
        verify_inp(
            action.shape == dH.shape,
            lambda: f"action shape {action.shape} != return shape {dH.shape}",
        )
        penalty = self.config.quadratic_coefficient * action.square().sum(dim=-1)
        gross_pnl = (action * dH).sum(dim=-1)
        pnl = gross_pnl - penalty
        if self.config.utility == "downside":
            utility, marginal = downside_oce_utility(
                pnl,
                self.cash_offset,
                scale=1.0 + self.config.lmbda,
                power=self.config.downside_power,
            )
        else:
            assert self.utility is not None
            assert self.utility_derivative is not None
            utility = self.utility(pnl, self.cash_offset)
            marginal = self.utility_derivative(pnl, self.cash_offset)
        residual = marginal[..., None] * (
            dH - 2.0 * self.config.quadratic_coefficient * action
        )
        self.verify(lambda: torch.isfinite(pnl), "quadratic-action P&L is non-finite")
        self.verify(
            lambda: torch.isfinite(utility),
            "quadratic-action utility is non-finite",
        )
        self.verify(
            lambda: torch.isfinite(marginal) & (marginal >= 0.0),
            "quadratic-action marginal utility is invalid",
        )
        return {
            "action": action,
            "gross_pnl": gross_pnl,
            "penalty": penalty,
            "pnl": pnl,
            "utility": utility,
            "marginal_utility": marginal,
            "first_order_residual": residual,
            "cash_offset": self.cash_offset,
        }

    def forward(self, data: Mapping[str, Tensor]) -> Tensor:
        output = self.outputs(data)
        weight = data.get("sampling_weight")
        if weight is None:
            weight = torch.ones(
                output["pnl"].shape[0],
                dtype=output["pnl"].dtype,
                device=output["pnl"].device,
            )
        verify_inp(
            weight.shape == (output["pnl"].shape[0],),
            lambda: f"sampling-weight shape {weight.shape} is invalid",
        )
        loss = -(weight[:, None] * output["utility"]).mean()
        self.verify(lambda: torch.isfinite(loss), "quadratic-action loss is non-finite")
        return loss


def iter_path_batches(
    data: Mapping[str, Tensor], batch_size: int | None
) -> Iterator[dict[str, Tensor]]:
    """Yield complete-path batches without changing any within-path reduction."""

    paths = data["z"].shape[0]
    size = paths if batch_size is None else batch_size
    verify_inp(size > 0, lambda: f"'batch_size' must be positive, got {size}")
    for start in range(0, paths, size):
        stop = min(start + size, paths)
        yield {name: value[start:stop] for name, value in data.items()}


def objective_gradient(
    model: QuadraticActionModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
) -> dict[str, float | int]:
    """Differentiate the exact full-split empirical objective in path batches."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("the model has no trainable parameters")
    for parameter in parameters:
        parameter.grad = None
    paths = data["z"].shape[0]
    total_loss = 0.0
    for batch in iter_path_batches(data, batch_size):
        fraction = batch["z"].shape[0] / float(paths)
        loss = model(batch) * fraction
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError("quadratic-action objective became non-finite")
        loss.backward()
        total_loss += loss_value
    squared_norm = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        nonfinite += int(torch.count_nonzero(~torch.isfinite(gradient)).item())
        squared_norm += torch.nan_to_num(gradient.to(torch.float64)).square().sum()
    return {
        "loss": total_loss,
        "gradient_norm": float(torch.sqrt(squared_norm).cpu().item()),
        "nonfinite_gradient_count": nonfinite,
    }


@torch.no_grad()
def valuate_quadratic_action(
    model: QuadraticActionModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int | None,
) -> dict[str, Any]:
    """Return saturation, first-order, and induced-density diagnostics."""

    S, TAU, TNX = data["dH"].shape
    device = data["dH"].device
    zero_ti = lambda: torch.zeros((TAU, TNX), dtype=torch.float64, device=device)
    sum_weight = torch.zeros(TAU, dtype=torch.float64, device=device)
    sum_marginal = torch.zeros(TAU, dtype=torch.float64, device=device)
    sum_density_dH = zero_ti()
    sum_density_action = zero_ti()
    sum_residual, sum_residual_sqr = zero_ti(), zero_ti()
    sum_action, sum_action_sqr, max_action = zero_ti(), zero_ti(), zero_ti()
    saturation = zero_ti()
    total_utility = torch.zeros((), dtype=torch.float64, device=device)
    total_pnl = torch.zeros_like(total_utility)
    total_penalty = torch.zeros_like(total_utility)
    bounded = isinstance(model.action, ScaledTanhNetwork)

    for batch in iter_path_batches(data, batch_size):
        output = model.outputs(batch)
        weight = batch.get("sampling_weight")
        if weight is None:
            weight = torch.ones(
                batch["z"].shape[0],
                dtype=batch["z"].dtype,
                device=device,
            )
        w = weight[:, None]
        wi = weight[:, None, None]
        marginal_weight = w * output["marginal_utility"]
        sum_weight += w.sum(dim=0, dtype=torch.float64).expand(TAU)
        sum_marginal += marginal_weight.sum(dim=0, dtype=torch.float64)
        sum_density_dH += (
            marginal_weight[..., None] * batch["dH"]
        ).sum(dim=0, dtype=torch.float64)
        sum_density_action += (
            marginal_weight[..., None] * output["action"]
        ).sum(dim=0, dtype=torch.float64)
        weighted_residual = wi * output["first_order_residual"]
        sum_residual += weighted_residual.sum(dim=0, dtype=torch.float64)
        # The Monte Carlo estimator is formed under the proposal measure.
        # Its sampling variance is therefore the variance of ``weight*residual``.
        sum_residual_sqr += weighted_residual.square().sum(
            dim=0, dtype=torch.float64
        )
        sum_action += (wi * output["action"]).sum(dim=0, dtype=torch.float64)
        sum_action_sqr += (
            wi * output["action"].square()
        ).sum(dim=0, dtype=torch.float64)
        max_action = torch.maximum(
            max_action,
            output["action"].abs().amax(dim=0).to(torch.float64),
        )
        if bounded:
            cap = model.action.scale.detach()[None, :]
            saturation += (
                wi * (output["action"].abs() >= 0.99 * cap)
            ).sum(dim=0, dtype=torch.float64)
        total_utility += (w * output["utility"]).sum(dtype=torch.float64)
        total_pnl += (w * output["pnl"]).sum(dtype=torch.float64)
        total_penalty += (w * output["penalty"]).sum(dtype=torch.float64)

    denominator = sum_weight[:, None].clamp_min(torch.finfo(torch.float64).tiny)
    marginal_denominator = sum_marginal[:, None].clamp_min(
        torch.finfo(torch.float64).tiny
    )
    residual_mean = sum_residual / denominator
    residual_variance = (
        sum_residual_sqr / denominator - residual_mean.square()
    ).clamp_min(0.0)
    residual_error = torch.sqrt(residual_variance / float(S))
    normalized_residual = torch.where(
        residual_error > 0.0,
        residual_mean / residual_error,
        torch.zeros_like(residual_mean),
    )
    action_mean = sum_action / denominator
    action_std = (
        sum_action_sqr / denominator - action_mean.square()
    ).clamp_min(0.0).sqrt()
    density_mean_dH = sum_density_dH / marginal_denominator
    density_mean_action = sum_density_action / marginal_denominator
    density_identity_residual = density_mean_dH - (
        2.0 * model.config.quadratic_coefficient * density_mean_action
    )
    observations = sum_weight.sum().clamp_min(torch.finfo(torch.float64).tiny)

    def cpu(value: Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    result: dict[str, Any] = {
        "utility": float((total_utility / observations).cpu().item()),
        "pnl_mean": float((total_pnl / observations).cpu().item()),
        "quadratic_penalty_mean": float(
            (total_penalty / observations).cpu().item()
        ),
        "action_mean": cpu(action_mean),
        "action_std": cpu(action_std),
        "action_max_abs": cpu(max_action),
        "first_order_residual_mean": cpu(residual_mean),
        "first_order_residual_error": cpu(residual_error),
        "first_order_residual_z": cpu(normalized_residual),
        "first_order_residual_max_abs_z": float(
            normalized_residual.abs().amax().cpu().item()
        ),
        "marginal_density_mean_dH": cpu(density_mean_dH),
        "marginal_density_mean_action": cpu(density_mean_action),
        "marginal_density_identity_residual": cpu(density_identity_residual),
        "marginal_density_identity_max_abs": float(
            density_identity_residual.abs().amax().cpu().item()
        ),
        "marginal_density_martingale_max_abs": float(
            density_mean_dH.abs().amax().cpu().item()
        ),
        "cash_offset": float(model.cash_offset.detach().cpu().item()),
    }
    if bounded:
        result["action_cap"] = cpu(model.action.scale)
        result["action_cap_saturation_rate"] = cpu(saturation / denominator)
        result["action_cap_saturation_rate_max"] = float(
            (saturation / denominator).amax().cpu().item()
        )
    result["cash_offset_first_order_residual"] = float(
        (sum_marginal.sum() / sum_weight.sum() - 1.0).cpu().item()
    )
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"valuation output {name!r} is non-finite")
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise FloatingPointError(f"valuation output {name!r} is non-finite")
    return result


def conditional_gaussian_entropy_action(
    mean: Tensor,
    covariance: Tensor,
    *,
    lmbda: float,
    quadratic_coefficient: float,
) -> Tensor:
    r"""Return ``(2*kappa*I + lambda*Sigma)^{-1} mean``."""

    verify_inp(mean.ndim >= 1, lambda: f"mean shape {mean.shape} is invalid")
    verify_inp(
        covariance.shape == mean.shape + (mean.shape[-1],),
        lambda: (
            f"covariance shape {covariance.shape} is incompatible with "
            f"mean shape {mean.shape}"
        ),
    )
    verify_inp(
        math.isfinite(lmbda) and lmbda > 0.0,
        lambda: f"'lmbda' must be finite and positive, got {lmbda}",
    )
    verify_inp(
        math.isfinite(quadratic_coefficient)
        and quadratic_coefficient > 0.0,
        lambda: (
            "'quadratic_coefficient' must be finite and positive, got "
            f"{quadratic_coefficient}"
        ),
    )
    identity = torch.eye(
        mean.shape[-1], dtype=mean.dtype, device=mean.device
    )
    system = (
        2.0 * quadratic_coefficient * identity
        + lmbda * covariance
    )
    action = torch.linalg.solve(system, mean.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(action).all().item():
        raise FloatingPointError("conditional Gaussian action is non-finite")
    return action


def parameter_norm(parameters: Sequence[Tensor]) -> float:
    """Return the finite Euclidean norm of a parameter sequence."""

    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        total += parameter.detach().cpu().to(torch.float64).square().sum()
    result = float(torch.sqrt(total).item())
    if not math.isfinite(result):
        raise FloatingPointError("parameter norm is non-finite")
    return result


__all__ = [
    "DirectActionNetwork",
    "QuadraticActionConfig",
    "QuadraticActionModel",
    "conditional_gaussian_entropy_action",
    "downside_oce_utility",
    "iter_path_batches",
    "objective_gradient",
    "parameter_norm",
    "valuate_quadratic_action",
]
