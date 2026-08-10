"""Independent conditional expected-utility problems by time and instrument."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel, FeedForwardConfig
from quantdata.torch.utility import torch_utility


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


@version("1.0", dependencies=[FeedForwardConfig])
class PerOptionAction(BaseModel):
    """One independent scalar MLP for every ``(time, instrument)`` pair.

    Parameters are stored in grouped tensors, but no parameter is shared
    between two time/instrument conditions.  This is exactly equivalent to a
    module list of scalar-output networks and is substantially faster to
    compile and execute on a GPU.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        steps: int,
        instruments: int,
        num_input: int,
        config: FeedForwardConfig,
        initial_scale: float = 1.0,
    ) -> None:
        super().__init__(env)
        verify_inp(steps > 0, "'steps' must be positive")
        verify_inp(instruments > 0, "'instruments' must be positive")
        verify_inp(num_input > 0, "'num_input' must be positive")
        verify_inp(
            config.num_width > 0,
            "per-option networks require a positive explicit width",
        )
        verify_inp(
            math.isfinite(initial_scale) and initial_scale > 0.0,
            "'initial_scale' must be finite and positive",
        )
        self.steps = steps
        self.instruments = instruments
        self.num_input = num_input
        self.num_depth = config.num_depth
        self.activation = config.get_activation()
        width = config.num_width

        if self.num_depth == 0:
            self.input_weight = torch.nn.Parameter(
                torch.empty(
                    (steps, instruments, num_input),
                    dtype=env.fdtype,
                    device=env.device,
                )
            )
            self.register_parameter("input_bias", None)
            self.hidden_weights = torch.nn.ParameterList()
            self.hidden_biases = torch.nn.ParameterList()
            self.register_parameter("output_weight", None)
        else:
            self.input_weight = torch.nn.Parameter(
                torch.empty(
                    (steps, instruments, width, num_input),
                    dtype=env.fdtype,
                    device=env.device,
                )
            )
            self.input_bias = (
                torch.nn.Parameter(
                    torch.empty(
                        (steps, instruments, width),
                        dtype=env.fdtype,
                        device=env.device,
                    )
                )
                if config.use_bias
                else None
            )
            self.hidden_weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.empty(
                            (steps, instruments, width, width),
                            dtype=env.fdtype,
                            device=env.device,
                        )
                    )
                    for _ in range(self.num_depth - 1)
                ]
            )
            self.hidden_biases = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.empty(
                            (steps, instruments, width),
                            dtype=env.fdtype,
                            device=env.device,
                        )
                    )
                    for _ in range(self.num_depth - 1)
                ]
                if config.use_bias
                else []
            )
            self.output_weight = torch.nn.Parameter(
                torch.empty(
                    (steps, instruments, width),
                    dtype=env.fdtype,
                    device=env.device,
                )
            )
        self.output_bias = torch.nn.Parameter(
            torch.zeros(
                (steps, instruments),
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self.raw_scale = torch.nn.Parameter(
            torch.full(
                (steps, instruments),
                _inverse_softplus(initial_scale),
                dtype=env.fdtype,
                device=env.device,
            )
        )
        self._reset_parameters(config.use_bias)
        self.zero_output_layer()

    @torch.no_grad()
    def _reset_parameters(self, use_bias: bool) -> None:
        def uniform(parameter: Tensor, fan_in: int) -> None:
            bound = 1.0 / math.sqrt(float(fan_in))
            parameter.uniform_(-bound, bound)

        uniform(self.input_weight, self.num_input)
        if self.input_bias is not None:
            uniform(self.input_bias, self.num_input)
        width = self.output_bias.shape[-1]
        if self.num_depth > 0:
            width = self.input_weight.shape[-2]
            for weight in self.hidden_weights:
                uniform(weight, width)
            if use_bias:
                for bias in self.hidden_biases:
                    uniform(bias, width)
            assert self.output_weight is not None
            uniform(self.output_weight, width)

    @torch.no_grad()
    def zero_output_layer(self) -> None:
        """Start every independent policy at the neutral zero action."""

        if self.output_weight is None:
            self.input_weight.zero_()
        else:
            self.output_weight.zero_()
        self.output_bias.zero_()

    @property
    def scale(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_scale)

    def logits(self, z: Tensor) -> Tensor:
        expected = (self.steps, self.num_input)
        if z.shape[-2:] != expected:
            raise ValueError(f"state shape {z.shape[-2:]} != expected {expected}")
        if self.num_depth == 0:
            x = torch.einsum("stf,tif->sti", z, self.input_weight)
        else:
            x = torch.einsum("stf,tiwf->stiw", z, self.input_weight)
            if self.input_bias is not None:
                x = x + self.input_bias
            for index, weight in enumerate(self.hidden_weights):
                x = self.activation(x)
                x = torch.einsum("stiw,tiow->stio", x, weight)
                if self.hidden_biases:
                    x = x + self.hidden_biases[index]
            x = self.activation(x)
            assert self.output_weight is not None
            x = torch.einsum("stiw,tiw->sti", x, self.output_weight)
        output = x + self.output_bias
        self.verify(lambda: torch.isfinite(output), "per-option logits are non-finite")
        return output

    def normalized_action(self, z: Tensor) -> Tensor:
        return torch.tanh(self.logits(z))

    def forward(self, z: Tensor) -> Tensor:
        output = self.scale * self.normalized_action(z)
        self.verify(lambda: torch.isfinite(output), "per-option action is non-finite")
        return output


@version("1.0", dependencies=[PerOptionAction, torch_utility])
class IndividualOCEModel(BaseModel):
    """Sum of independent one-trade powerexp objectives."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        steps: int,
        instruments: int,
        num_input: int,
        network_config: FeedForwardConfig,
        utility: str = "powexp",
        lmbda: float = 1.0,
        cost_prop: float = 0.0,
    ) -> None:
        super().__init__(env)
        verify_inp(
            math.isfinite(cost_prop) and cost_prop >= 0.0,
            "'cost_prop' must be finite and nonnegative",
        )
        self.action = PerOptionAction(
            env,
            steps=steps,
            instruments=instruments,
            num_input=num_input,
            config=network_config,
        )
        self.utility, self.utility_derivative, self.utility_label = torch_utility(
            utility, lmbda
        )
        self.cost_prop = cost_prop

    def outputs(self, data: Mapping[str, Tensor]) -> dict[str, Tensor]:
        z, dH, H = data["z"], data["dH"], data["H"]
        action = self.action(z)
        cost = self.cost_prop * H.abs()
        pnl = action * dH - action.abs() * cost
        zero = torch.zeros((), dtype=pnl.dtype, device=pnl.device)
        utility = self.utility(pnl, zero)
        marginal = self.utility_derivative(pnl, zero)
        return {
            "action": action,
            "cost": cost,
            "pnl": pnl,
            "utility": utility,
            "marginal_utility": marginal,
        }

    def forward(self, data: Mapping[str, Tensor]) -> Tensor:
        output = self.outputs(data)
        sampling_weight = data["sampling_weight"][:, None, None]
        loss = -(sampling_weight * output["utility"]).mean()
        self.verify(lambda: torch.isfinite(loss), "individual OCE loss is non-finite")
        return loss


def iter_path_batches(
    data: Mapping[str, Tensor], batch_size: int
) -> Iterator[dict[str, Tensor]]:
    paths = data["z"].shape[0]
    for start in range(0, paths, batch_size):
        stop = min(start + batch_size, paths)
        yield {name: value[start:stop] for name, value in data.items()}


def individual_cap_derivative(
    model: IndividualOCEModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int,
) -> np.ndarray:
    """Return ``d loss[t,omega] / d cap[t,omega]`` on a full split."""

    total = torch.zeros_like(model.action.raw_scale, dtype=torch.float64)
    paths = data["z"].shape[0]
    conditions = model.action.steps * model.action.instruments
    for batch in iter_path_batches(data, batch_size):
        loss = model(batch)
        derivative = torch.autograd.grad(loss, model.action.raw_scale)[0]
        weight = batch["z"].shape[0] / float(paths)
        total.add_(derivative.detach().to(torch.float64), alpha=weight)
    jacobian = torch.sigmoid(model.action.raw_scale.detach()).to(torch.float64)
    derivative = total * float(conditions) / jacobian.clamp_min(1.0e-12)
    result = derivative.cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("individual cap derivative is non-finite")
    return result


@torch.no_grad()
def valuate_individual(
    model: IndividualOCEModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Compute per-condition utility, FOC, saturation, and action statistics."""

    shape = (model.action.steps, model.action.instruments)
    utility_sum = torch.zeros(shape, dtype=torch.float64, device=model.device)
    marginal_sum = torch.zeros_like(utility_sum)
    foc_sum = torch.zeros_like(utility_sum)
    action_sum = torch.zeros_like(utility_sum)
    action_sqr = torch.zeros_like(utility_sum)
    saturation_sum = torch.zeros_like(utility_sum)
    sampling_mass = torch.zeros((), dtype=torch.float64, device=model.device)
    for batch in iter_path_batches(data, batch_size):
        output = model.outputs(batch)
        weight = batch["sampling_weight"][:, None, None]
        action = output["action"]
        direction_return = batch["dH"] - torch.sign(action) * output["cost"]
        utility_sum += (weight * output["utility"]).sum(dim=0, dtype=torch.float64)
        marginal_sum += (weight * output["marginal_utility"]).sum(
            dim=0, dtype=torch.float64
        )
        foc_sum += (
            weight * output["marginal_utility"] * direction_return
        ).sum(dim=0, dtype=torch.float64)
        action_sum += (weight * action).sum(dim=0, dtype=torch.float64)
        action_sqr += (weight * action.square()).sum(dim=0, dtype=torch.float64)
        saturation_sum += (
            weight * (model.action.normalized_action(batch["z"]).abs() >= 0.99)
        ).sum(dim=0, dtype=torch.float64)
        sampling_mass += batch["sampling_weight"].sum(dtype=torch.float64)
    mean_action = action_sum / sampling_mass
    action_std = (
        action_sqr / sampling_mass - mean_action.square()
    ).clamp_min(0.0).sqrt()
    return {
        "loss": float(-(utility_sum / sampling_mass).mean().item()),
        "utility": (utility_sum / sampling_mass).cpu().numpy(),
        "marginal_utility_mean": (marginal_sum / sampling_mass).cpu().numpy(),
        "constant_action_foc": (foc_sum / sampling_mass).cpu().numpy(),
        "action_mean": mean_action.cpu().numpy(),
        "action_std": action_std.cpu().numpy(),
        "action_cap": model.action.scale.detach().cpu().numpy(),
        "action_cap_saturation_rate": (
            saturation_sum / sampling_mass
        ).cpu().numpy(),
    }


__all__ = [
    "IndividualOCEModel",
    "PerOptionAction",
    "individual_cap_derivative",
    "iter_path_batches",
    "valuate_individual",
]
