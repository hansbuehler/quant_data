"""Focused tests for batched OCE optimizers."""

from __future__ import annotations

import torch

from quantdata.oce.batched_optimizer_oce import Shampoo, Yogi


def _fit_quadratic(optimizer_name: str) -> tuple[float, float]:
    parameter = torch.nn.Parameter(
        torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float64)
    )
    target = torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=torch.float64)
    if optimizer_name == "yogi":
        optimizer = Yogi((parameter,), lr=0.05)
    else:
        optimizer = Shampoo((parameter,), lr=0.05)
    initial = float(torch.sum((parameter.detach() - target).square()))
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * torch.sum((parameter - target).square())
        loss.backward()
        optimizer.step()
    final = float(torch.sum((parameter.detach() - target).square()))
    return initial, final


def test_yogi_reduces_quadratic() -> None:
    initial, final = _fit_quadratic("yogi")
    assert final < 0.05 * initial


def test_shampoo_reduces_quadratic() -> None:
    initial, final = _fit_quadratic("shampoo")
    assert final < 0.05 * initial
