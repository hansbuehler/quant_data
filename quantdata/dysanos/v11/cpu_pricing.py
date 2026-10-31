"""Exact first-order CPU adjoints for the frozen SANOS decoder.

The batched Thomas solves run in NumPy rather than building thousands of
elementary Torch graph nodes. No market fit or model parameter is changed.
"""

from __future__ import annotations

import numpy as np
import torch

from quantdata.dysanos.flex.sanos import uses_shared_strikes


def _solve(lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    d, y = diagonal.copy(), rhs.copy()
    for j in range(1, d.shape[-1]):
        multiplier = lower[..., j - 1] / d[..., j - 1]
        d[..., j] -= multiplier * upper[..., j - 1]
        y[..., j] -= multiplier * y[..., j - 1]
    x = y
    x[..., -1] /= d[..., -1]
    for j in range(d.shape[-1] - 2, -1, -1):
        x[..., j] = (x[..., j] - upper[..., j] * x[..., j + 1]) / d[..., j]
    return x


def _bands(w_minus: torch.Tensor, w_plus: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minus = w_minus.detach().double().numpy()
    plus = w_plus.detach().double().numpy()
    diagonal = np.ones((*minus.shape[:-1], minus.shape[-1] + 2))
    diagonal[..., 1:-1] += minus + plus
    lower = np.zeros((*minus.shape[:-1], minus.shape[-1] + 1))
    upper = lower.copy()
    lower[..., 1:] = -plus
    upper[..., :-1] = -minus
    return lower, diagonal, upper


class _CpuSanosSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_minus: torch.Tensor, w_plus: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if any(value.device.type != "cpu" for value in (w_minus, w_plus, rhs)):
            raise ValueError("CPU SANOS adjoints cannot price GPU tensors")
        lower, diagonal, upper = _bands(w_minus, w_plus)
        solution = torch.from_numpy(_solve(lower, diagonal, upper, rhs.detach().double().numpy())).to(rhs.dtype)
        ctx.save_for_backward(solution, w_minus, w_plus)
        return solution

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        solution, w_minus, w_plus = ctx.saved_tensors
        lower, diagonal, upper = _bands(w_minus, w_plus)
        adjoint = _solve(upper, diagonal, lower, output_gradient.detach().double().numpy())
        x = solution.detach().double().numpy()[..., 1:-1]
        minus_gradient = -(adjoint[..., 1:-1] - adjoint[..., :-2]) * x
        plus_gradient = -(adjoint[..., 1:-1] - adjoint[..., 2:]) * x
        return (
            torch.from_numpy(minus_gradient).to(w_minus.dtype),
            torch.from_numpy(plus_gradient).to(w_plus.dtype),
            torch.from_numpy(adjoint).to(solution.dtype),
        )


def _solve_step(*, w_minus: torch.Tensor, w_plus: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return _CpuSanosSolve.apply(w_minus, w_plus, rhs)


def _identity_transfer(previous_support: torch.Tensor, previous_density: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    # Shared-strike geometry uses the identical support at every expiry. The
    # call-price transfer/re-differentiation is therefore exactly the identity.
    return previous_density


def enable_fast_cpu_decoder(model: object) -> None:
    """Install mathematically identical CPU solves on this model instance only."""
    if model.device.type != "cpu":
        raise ValueError("This pricing accelerator is CPU-only")
    model.sanos_decoder._sanos_step = _solve_step
    if uses_shared_strikes(model.config.geometry):
        model.sanos_decoder._transfer_density = _identity_transfer
