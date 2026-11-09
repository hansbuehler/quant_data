"""Audit OCE signs, cash translation and gradients against the shared utility."""

import pytest
import torch

from quantdata.sdlv.dhgromm1.hedge import oce_value
from quantdata.torch.utility import torch_utility


def test_oce_matches_shared_powexp_values_and_gradients() -> None:
    pnl = torch.tensor([-.1, -.02, 0., .02, .2], dtype=torch.float64, requires_grad=True)
    m = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    utility, _, _ = torch_utility("powexp", 100.)
    actual = oce_value(pnl, m, 100.)
    expected = utility(pnl, -m).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    gradients = torch.autograd.grad(actual, (pnl, m), retain_graph=True)
    reference = torch.autograd.grad(expected, (pnl, m))
    for left, right in zip(gradients, reference):
        torch.testing.assert_close(left, right, rtol=1e-13, atol=1e-13)


def test_oce_cash_translation_sign() -> None:
    pnl = torch.tensor([-.02, .01, .03], dtype=torch.float64)
    m = torch.tensor(.005, dtype=torch.float64)
    torch.testing.assert_close(oce_value(pnl + .4, m + .4, 100.), oce_value(pnl, m, 100.) + .4)


@pytest.mark.parametrize("sign", [0, 1, -1])
def test_penalized_loss_trade_gradient_by_finite_difference(sign: int) -> None:
    action = torch.tensor([.003, -.002], dtype=torch.float64, requires_grad=True)
    returns = torch.tensor([[.4, -1.], [-.8, 2.], [.1, .4]], dtype=torch.float64)
    payoff = torch.tensor([.02, .01, .04], dtype=torch.float64)
    m = torch.tensor(.005, dtype=torch.float64)

    def loss(a: torch.Tensor) -> torch.Tensor:
        gain = returns @ a
        return -oce_value(sign * payoff + gain, m, 100.) + 1e-4 * a.abs().sum() + 1e-8 * a.square().sum()

    gradient, = torch.autograd.grad(loss(action), action)
    for index in range(2):
        step = torch.zeros_like(action)
        step[index] = 1e-7
        finite_difference = (loss(action.detach() + step) - loss(action.detach() - step)) / (2e-7)
        assert gradient[index].item() == pytest.approx(finite_difference.item(), rel=1e-7, abs=1e-9)
