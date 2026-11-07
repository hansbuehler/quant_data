"""Finite large gradients must not be mistaken for non-finite derivatives."""

import copy

import numpy as np
import pytest
import torch

from quantdata.hedge2.terminal_train import clip_terminal_gradients


def parameter_with_gradient(values: list[float], *, dtype=torch.float32) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.zeros(len(values), dtype=dtype))
    parameter.grad = torch.tensor(values, dtype=dtype)
    return parameter


def test_float32_norm_overflow_is_clipped_without_discarding_finite_gradients() -> None:
    parameter = parameter_with_gradient([3e25, -4e25])
    assert not torch.isfinite(torch.linalg.vector_norm(parameter.grad))
    norm = clip_terminal_gradients([parameter], maximum_norm=5.)
    assert np.isfinite(norm)
    assert norm == pytest.approx(5e25, rel=1e-6)
    torch.testing.assert_close(parameter.grad, torch.tensor([3., -4.]))
    assert parameter.grad.dtype == torch.float32


def test_clipping_agrees_with_standard_operation_on_ordinary_gradients() -> None:
    actual = [parameter_with_gradient([3., -4.]), parameter_with_gradient([12.])]
    expected = copy.deepcopy(actual)
    for destination, source in zip(expected, actual, strict=True):
        destination.grad = source.grad.clone()
    reference = float(torch.nn.utils.clip_grad_norm_(expected, 2.))
    assert clip_terminal_gradients(actual, maximum_norm=2.) == pytest.approx(reference)
    for candidate, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate.grad, target.grad)


def test_clipping_product_does_not_underflow_when_scale_alone_would() -> None:
    parameter = parameter_with_gradient([3e300, -4e300], dtype=torch.float64)
    norm = clip_terminal_gradients([parameter], maximum_norm=5e-100)
    assert norm == pytest.approx(5e300)
    torch.testing.assert_close(parameter.grad, torch.tensor([3e-100, -4e-100], dtype=torch.float64), rtol=1e-14, atol=0.)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_true_nonfinite_gradient_rejects_before_mutating_any_gradient_or_optimizer(bad: float) -> None:
    parameters = [parameter_with_gradient([3., 4.]), parameter_with_gradient([bad])]
    optimizer = torch.optim.Adam(parameters)
    state = copy.deepcopy(optimizer.state_dict())
    before = [parameter.grad.clone() for parameter in parameters]
    with pytest.raises(RuntimeError, match="Non-finite terminal-policy gradient"):
        clip_terminal_gradients(parameters, maximum_norm=1.)
        optimizer.step()
    assert optimizer.state_dict() == state
    for parameter, expected in zip(parameters, before, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=0., atol=0., equal_nan=True)
        torch.testing.assert_close(parameter, torch.zeros_like(parameter))


def test_zero_absent_and_mixed_precision_gradients() -> None:
    absent = torch.nn.Parameter(torch.zeros(2))
    assert clip_terminal_gradients([absent], maximum_norm=5.) == 0.
    zero = parameter_with_gradient([0.])
    assert clip_terminal_gradients([zero, absent], maximum_norm=5.) == 0.
    first = parameter_with_gradient([3.], dtype=torch.float32)
    second = parameter_with_gradient([4.], dtype=torch.float64)
    assert clip_terminal_gradients([first, second], maximum_norm=1.) == pytest.approx(5.)
    torch.testing.assert_close(first.grad, torch.tensor([.6]), rtol=1e-6, atol=0.)
    torch.testing.assert_close(second.grad, torch.tensor([.8], dtype=torch.float64), rtol=1e-6, atol=0.)


@pytest.mark.parametrize("maximum_norm", [0., -1., np.nan, np.inf])
def test_invalid_clipping_threshold_is_rejected(maximum_norm: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        clip_terminal_gradients([], maximum_norm=maximum_norm)
