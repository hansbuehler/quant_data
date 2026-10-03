"""Curvature-aware optimizers for the fixed-population OCE experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from cdxcore.err import verify_inp


def _symmetric_inverse_power(
    matrix: torch.Tensor,
    *,
    exponent: float,
    damping: float,
) -> torch.Tensor:
    """Return a damped inverse power of a finite symmetric matrix."""

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    powered = torch.clamp_min(eigenvalues, 0.0).add(damping).pow(exponent)
    return (eigenvectors * powered[None]) @ eigenvectors.T


class Shampoo(torch.optim.Optimizer):
    """Matrix-preconditioned Shampoo with diagonal AdaGrad grafting.

    The OCE runner supplies one exact full-population gradient per call. Matrix
    statistics are therefore updated once per population, not once per memory
    chunk. Two-dimensional weight tensors use left/right Kronecker factors;
    vectors and higher-rank fallbacks use diagonal AdaGrad scaling.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        beta2: float = 0.9,
        momentum: float = 0.0,
        damping: float = 1.0e-4,
        precondition_frequency: int = 1,
        maximum_parameter_step: float = 0.05,
    ) -> None:
        verify_inp(math.isfinite(lr) and lr > 0.0, lambda: "Shampoo lr must be positive")
        verify_inp(0.0 <= beta2 < 1.0, lambda: "Shampoo beta2 must be in [0,1)")
        verify_inp(
            0.0 <= momentum < 1.0,
            lambda: "Shampoo momentum must be in [0,1)",
        )
        verify_inp(
            math.isfinite(damping) and damping > 0.0,
            lambda: "Shampoo damping must be positive",
        )
        verify_inp(
            precondition_frequency > 0,
            lambda: "Shampoo precondition frequency must be positive",
        )
        verify_inp(
            math.isfinite(maximum_parameter_step) and maximum_parameter_step > 0.0,
            lambda: "Shampoo maximum parameter step must be positive",
        )
        super().__init__(
            parameters,
            defaults={
                "lr": lr,
                "beta2": beta2,
                "momentum": momentum,
                "damping": damping,
                "precondition_frequency": precondition_frequency,
                "maximum_parameter_step": maximum_parameter_step,
            },
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Apply one exact-gradient Shampoo update."""

        loss = closure() if closure is not None else None
        for group in self.param_groups:
            directions: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
            direction_square = torch.zeros((), dtype=torch.float64)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                state = self.state[parameter]
                step = int(state.get("step", 0)) + 1
                state["step"] = step
                diagonal = state.setdefault(
                    "diagonal",
                    torch.zeros_like(parameter, memory_format=torch.preserve_format),
                )
                diagonal.mul_(group["beta2"]).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - group["beta2"],
                )
                graft = gradient / torch.sqrt(diagonal + group["damping"])

                if gradient.ndim == 2:
                    rows, columns = gradient.shape
                    left = state.setdefault(
                        "left",
                        torch.zeros(
                            (rows, rows),
                            dtype=gradient.dtype,
                            device=gradient.device,
                        ),
                    )
                    right = state.setdefault(
                        "right",
                        torch.zeros(
                            (columns, columns),
                            dtype=gradient.dtype,
                            device=gradient.device,
                        ),
                    )
                    left.mul_(group["beta2"]).addmm_(
                        gradient,
                        gradient.T,
                        beta=1.0,
                        alpha=(1.0 - group["beta2"]) / max(columns, 1),
                    )
                    right.mul_(group["beta2"]).addmm_(
                        gradient.T,
                        gradient,
                        beta=1.0,
                        alpha=(1.0 - group["beta2"]) / max(rows, 1),
                    )
                    if (
                        step == 1
                        or step % group["precondition_frequency"] == 0
                        or "left_inverse_root" not in state
                    ):
                        state["left_inverse_root"] = _symmetric_inverse_power(
                            left,
                            exponent=-0.25,
                            damping=group["damping"],
                        )
                        state["right_inverse_root"] = _symmetric_inverse_power(
                            right,
                            exponent=-0.25,
                            damping=group["damping"],
                        )
                    direction = (
                        state["left_inverse_root"]
                        @ gradient
                        @ state["right_inverse_root"]
                    )
                    direction_norm = torch.linalg.vector_norm(direction)
                    graft_norm = torch.linalg.vector_norm(graft)
                    if float(direction_norm) > 0.0:
                        direction = direction * (graft_norm / direction_norm)
                else:
                    direction = graft

                if group["momentum"] > 0.0:
                    buffer = state.setdefault("momentum_buffer", torch.zeros_like(direction))
                    buffer.mul_(group["momentum"]).add_(
                        direction,
                        alpha=1.0 - group["momentum"],
                    )
                    direction = buffer
                directions.append((parameter, direction))
                direction_square += torch.sum(direction.double().square()).cpu()

            direction_norm = math.sqrt(float(direction_square))
            learning_rate = float(group["lr"])
            scale = min(
                1.0,
                group["maximum_parameter_step"]
                / max(learning_rate * direction_norm, torch.finfo(torch.float64).tiny),
            )
            for parameter, direction in directions:
                parameter.add_(direction, alpha=-learning_rate * scale)
            group["last_preconditioned_direction_norm"] = direction_norm
            group["last_parameter_step_scale"] = scale
        return loss


class Kfac(torch.optim.Optimizer):
    """Damped K-FAC for the linear layers of a shared OCE policy.

    Activations and output sensitivities are accumulated under the current
    tilted empirical path probabilities. The runner calls ``begin_full_batch``
    once per exact gradient and ``set_batch_probabilities`` before every memory
    chunk. Biases are folded into the activation factor.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float,
        factor_decay: float = 0.9,
        damping: float = 1.0e-3,
        momentum: float = 0.0,
        kl_clip: float = 1.0e-2,
        factor_update_frequency: int = 1,
    ) -> None:
        modules = tuple(
            module
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
            and any(parameter.requires_grad for parameter in module.parameters())
        )
        verify_inp(bool(modules), lambda: "K-FAC requires trainable linear layers")
        verify_inp(math.isfinite(lr) and lr > 0.0, lambda: "K-FAC lr must be positive")
        verify_inp(
            0.0 <= factor_decay < 1.0,
            lambda: "K-FAC factor decay must be in [0,1)",
        )
        verify_inp(
            math.isfinite(damping) and damping > 0.0,
            lambda: "K-FAC damping must be positive",
        )
        verify_inp(
            0.0 <= momentum < 1.0,
            lambda: "K-FAC momentum must be in [0,1)",
        )
        verify_inp(
            math.isfinite(kl_clip) and kl_clip > 0.0,
            lambda: "K-FAC KL clip must be positive",
        )
        verify_inp(
            factor_update_frequency > 0,
            lambda: "K-FAC factor update frequency must be positive",
        )
        parameters = tuple(parameter for module in modules for parameter in module.parameters())
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("K-FAC linear modules share parameter tensors")
        super().__init__(
            parameters,
            defaults={
                "lr": lr,
                "factor_decay": factor_decay,
                "damping": damping,
                "momentum": momentum,
                "kl_clip": kl_clip,
                "factor_update_frequency": factor_update_frequency,
            },
        )
        self.modules = modules
        self._batch_probabilities: torch.Tensor | None = None
        self._factor_sums: dict[torch.nn.Module, dict[str, torch.Tensor | float]] = {}
        self._handles = tuple(
            handle
            for module in modules
            for handle in (
                module.register_forward_pre_hook(self._capture_activation),
                module.register_full_backward_hook(self._capture_sensitivity),
            )
        )

    def begin_full_batch(self) -> None:
        """Reset the factor accumulators for one exact population gradient."""

        self._factor_sums = {}
        self._batch_probabilities = None

    def set_batch_probabilities(self, probabilities: torch.Tensor | None) -> None:
        """Set normalized path probabilities for the current memory chunk."""

        self._batch_probabilities = (
            probabilities.detach() if probabilities is not None else None
        )

    def _expanded_probabilities(self, tensor: torch.Tensor) -> torch.Tensor:
        probabilities = self._batch_probabilities
        if probabilities is None or tensor.shape[0] != len(probabilities):
            raise RuntimeError("K-FAC batch probabilities are missing or misaligned")
        repeats = tensor.numel() // (tensor.shape[0] * tensor.shape[-1])
        return probabilities.to(dtype=tensor.dtype, device=tensor.device)[:, None].expand(
            -1, repeats
        ).reshape(-1)

    def _module_sums(
        self,
        module: torch.nn.Module,
        *,
        activation_size: int,
        output_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor | float]:
        values = self._factor_sums.get(module)
        if values is None:
            values = {
                "activation": torch.zeros(
                    (activation_size, activation_size), dtype=dtype, device=device
                ),
                "sensitivity": torch.zeros(
                    (output_size, output_size), dtype=dtype, device=device
                ),
                "activation_mass": 0.0,
                "sensitivity_mass": 0.0,
            }
            self._factor_sums[module] = values
        return values

    @torch.no_grad()
    def _capture_activation(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        if self._batch_probabilities is None:
            return
        activation = inputs[0].detach()
        probabilities = self._expanded_probabilities(activation)
        flattened = activation.reshape(-1, activation.shape[-1])
        if getattr(module, "bias", None) is not None:
            flattened = torch.cat(
                (flattened, torch.ones_like(flattened[:, :1])), dim=1
            )
        values = self._module_sums(
            module,
            activation_size=flattened.shape[1],
            output_size=module.out_features,
            dtype=flattened.dtype,
            device=flattened.device,
        )
        weighted = flattened * torch.sqrt(probabilities)[:, None]
        values["activation"].addmm_(weighted.T, weighted)
        values["activation_mass"] += float(torch.sum(probabilities).cpu())

    @torch.no_grad()
    def _capture_sensitivity(
        self,
        module: torch.nn.Module,
        _gradient_inputs: tuple[torch.Tensor | None, ...],
        gradient_outputs: tuple[torch.Tensor | None, ...],
    ) -> None:
        if self._batch_probabilities is None or gradient_outputs[0] is None:
            return
        sensitivity = gradient_outputs[0].detach()
        probabilities = self._expanded_probabilities(sensitivity)
        flattened = sensitivity.reshape(-1, sensitivity.shape[-1])
        # Backprop supplied q_i * local_sensitivity. Dividing by sqrt(q_i)
        # produces vectors whose outer products sum to E_q[delta delta'].
        weighted = flattened / torch.sqrt(
            torch.clamp_min(probabilities, torch.finfo(probabilities.dtype).tiny)
        )[:, None]
        values = self._factor_sums[module]
        values["sensitivity"].addmm_(weighted.T, weighted)
        values["sensitivity_mass"] += float(torch.sum(probabilities).cpu())

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Apply one damped natural-gradient K-FAC update."""

        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        preconditioned: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        fisher_inner_product = torch.zeros((), dtype=torch.float64)
        for module in self.modules:
            if module.weight.grad is None:
                continue
            sums = self._factor_sums.get(module)
            if sums is None:
                raise RuntimeError("K-FAC did not observe a trainable linear module")
            activation = sums["activation"] / max(
                float(sums["activation_mass"]), torch.finfo(torch.float64).tiny
            )
            sensitivity = sums["sensitivity"] / max(
                float(sums["sensitivity_mass"]), torch.finfo(torch.float64).tiny
            )
            state = self.state[module.weight]
            step = int(state.get("step", 0)) + 1
            state["step"] = step
            if "activation_factor" not in state:
                state["activation_factor"] = activation.clone()
                state["sensitivity_factor"] = sensitivity.clone()
            else:
                state["activation_factor"].mul_(group["factor_decay"]).add_(
                    activation, alpha=1.0 - group["factor_decay"]
                )
                state["sensitivity_factor"].mul_(group["factor_decay"]).add_(
                    sensitivity, alpha=1.0 - group["factor_decay"]
                )
            if (
                step == 1
                or step % group["factor_update_frequency"] == 0
                or "activation_inverse" not in state
            ):
                state["activation_inverse"] = _symmetric_inverse_power(
                    state["activation_factor"],
                    exponent=-1.0,
                    damping=group["damping"],
                )
                state["sensitivity_inverse"] = _symmetric_inverse_power(
                    state["sensitivity_factor"],
                    exponent=-1.0,
                    damping=group["damping"],
                )
            gradient = module.weight.grad.detach()
            if module.bias is not None:
                if module.bias.grad is None:
                    raise RuntimeError("K-FAC bias is missing its exact gradient")
                gradient = torch.cat((gradient, module.bias.grad.detach()[:, None]), dim=1)
            direction = (
                state["sensitivity_inverse"]
                @ gradient
                @ state["activation_inverse"]
            )
            weight_direction = direction[:, : module.in_features]
            preconditioned.append((module.weight, weight_direction))
            fisher_inner_product += torch.sum(
                module.weight.grad.detach().double() * weight_direction.double()
            ).cpu()
            if module.bias is not None:
                bias_direction = direction[:, -1]
                preconditioned.append((module.bias, bias_direction))
                fisher_inner_product += torch.sum(
                    module.bias.grad.detach().double() * bias_direction.double()
                ).cpu()

        fisher_value = max(float(fisher_inner_product), 0.0)
        learning_rate = float(group["lr"])
        scale = min(
            1.0,
            math.sqrt(
                group["kl_clip"]
                / max(
                    learning_rate * learning_rate * fisher_value,
                    torch.finfo(torch.float64).tiny,
                )
            ),
        )
        for parameter, direction in preconditioned:
            if group["momentum"] > 0.0:
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(direction))
                buffer.mul_(group["momentum"]).add_(
                    direction,
                    alpha=1.0 - group["momentum"],
                )
                direction = buffer
            parameter.add_(direction, alpha=-learning_rate * scale)
        group["last_fisher_inner_product"] = fisher_value
        group["last_kl_scale"] = scale
        self._batch_probabilities = None
        return loss

