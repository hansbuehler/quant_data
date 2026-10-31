"""Analytic log contracts and their fast CPU SANOS adjoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.dysanos.v11.cpu_pricing import _solve_step, enable_fast_cpu_decoder
from quantdata.dysanos.v11.variance_indices import (
    historic_variance_indices,
    log_contract_total_variance,
)
from quantdata.torch import BaseEnvironment


def test_black_log_contract() -> None:
    total = log_contract_total_variance(torch.ones(2, 1), torch.ones(2, 1), torch.tensor([.04, .09]))
    torch.testing.assert_close(total, torch.tensor([.04, .09]))


def test_discrete_black_log_contract() -> None:
    support = torch.tensor([[.5, 1.5]], dtype=torch.float64, requires_grad=True)
    density = torch.tensor([[.5, .5]], dtype=torch.float64)
    total = log_contract_total_variance(support, density, torch.tensor([.01], dtype=torch.float64))
    assert total.item() == pytest.approx(.01 - np.log(.75))
    total.sum().backward()
    torch.testing.assert_close(support.grad, -2 * density / support)


def test_historic_interpolation_and_no_extrapolation() -> None:
    pricer = {"expiries": np.array([5., 10.]) / 252, "variances": np.array([.01, .02]),
              "smoothness": .25, "mstrikes": [np.ones(1), np.ones(1)],
              "mdensities": [np.ones(1), np.ones(1)]}
    values = historic_variance_indices(pricer, business_dtes=(5, 7, 10, 15))
    np.testing.assert_allclose(values[:3], 100 * np.sqrt(.25 * .01 / (5 / 252)))
    assert np.isnan(values[-1])


def test_cpu_solve_matches_original_values_and_gradients() -> None:
    generator = torch.Generator().manual_seed(15)
    minus = (torch.rand(3, 7, dtype=torch.float64, generator=generator) * 8).requires_grad_()
    plus = (torch.rand(3, 7, dtype=torch.float64, generator=generator) * 8).requires_grad_()
    rhs = torch.rand(3, 9, dtype=torch.float64, generator=generator).requires_grad_()
    weights = torch.randn(3, 9, dtype=torch.float64, generator=generator)
    reference = FlexSanosDecoder._solve_sanos_step(w_minus=minus, w_plus=plus, rhs=rhs)
    actual = _solve_step(w_minus=minus, w_plus=plus, rhs=rhs)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    ref_grad = torch.autograd.grad((reference * weights).sum(), (minus, plus, rhs))
    actual_grad = torch.autograd.grad((actual * weights).sum(), (minus, plus, rhs))
    for a, b in zip(actual_grad, ref_grad, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_cpu_solve_gradcheck() -> None:
    minus = torch.full((1, 3), .3, dtype=torch.float64, requires_grad=True)
    plus = torch.full((1, 3), .4, dtype=torch.float64, requires_grad=True)
    rhs = torch.ones(1, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda m, p, r: _solve_step(w_minus=m, w_plus=p, rhs=r), (minus, plus, rhs))


def test_fast_shared_decoder_preserves_log_contract_values_and_gradients() -> None:
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=10, num_model_expiries=3, explicit_model_business_dtes=(1., 5., 10.),
    )
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    dlv = torch.full((2, 3, 10), .04, dtype=torch.float64, requires_grad=True)
    vol = torch.full((2, 3), .15, dtype=torch.float64, requires_grad=True)
    eta = torch.full((2, 3), .2, dtype=torch.float64, requires_grad=True)

    def evaluate() -> torch.Tensor:
        surface = decoder.forward_batch(dlvsqs=dlv, forward_vols=vol, eta=eta)
        return log_contract_total_variance(surface.support_strikes, surface.densities, surface.backbone_total_vols.square())

    reference = evaluate()
    ref_grad = torch.autograd.grad(reference.sum(), (dlv, vol, eta))
    enable_fast_cpu_decoder(SimpleNamespace(device=torch.device("cpu"), config=SimpleNamespace(geometry=geometry), sanos_decoder=decoder))
    actual = evaluate()
    act_grad = torch.autograd.grad(actual.sum(), (dlv, vol, eta))
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    for a, b in zip(act_grad, ref_grad, strict=True):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
