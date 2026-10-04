from __future__ import annotations

import torch

from quantdata.dysanos.flex.oce_curvature_optimizers import Kfac, Shampoo


def test_shampoo_reduces_deterministic_quadratic() -> None:
    parameter = torch.nn.Parameter(
        torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float64)
    )
    target = torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=torch.float64)
    optimizer = Shampoo(
        (parameter,),
        lr=0.05,
        beta2=0.0,
        damping=1.0e-5,
        maximum_parameter_step=0.1,
    )
    initial = float(torch.sum((parameter.detach() - target).square()))

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * torch.sum((parameter - target).square())
        loss.backward()
        optimizer.step()

    final = float(torch.sum((parameter.detach() - target).square()))
    assert final < 0.1 * initial


def test_kfac_reduces_full_population_weighted_linear_loss() -> None:
    model = torch.nn.Linear(2, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.5, -0.75]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.5], dtype=torch.float64))
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
        dtype=torch.float64,
    )
    target = torch.tensor([0.5, -0.25, 0.25, -0.625], dtype=torch.float64)
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    optimizer = Kfac(
        model,
        lr=0.1,
        factor_decay=0.0,
        damping=1.0e-2,
        kl_clip=0.1,
    )

    def objective() -> torch.Tensor:
        residual = model(features).squeeze(1) - target
        return 0.5 * torch.sum(probabilities * residual.square())

    initial = float(objective().detach())
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        optimizer.begin_full_batch()
        optimizer.set_batch_probabilities(probabilities)
        loss = objective()
        loss.backward()
        optimizer.step()
    final = float(objective().detach())

    assert final < initial
    assert optimizer.param_groups[0]["last_fisher_inner_product"] > 0.0
    assert 0.0 < optimizer.param_groups[0]["last_kl_scale"] <= 1.0
