"""Mathematical tests for direct conditional-moment measure fitting."""

from __future__ import annotations

import pytest
import torch

from quantdata.dysanos.flex.conditional_moment import (
    replicated_conditional_moment_statistics,
    squared_conditional_moment_loss,
    standard_normal_transport_kl,
)


def test_squared_conditional_moment_loss_covers_every_linear_strategy() -> None:
    returns = torch.tensor(
        (
            (((1.0, 2.0), (3.0, 0.0)), ((-1.0, 4.0), (1.0, 2.0))),
            (((-2.0, 1.0), (2.0, -1.0)), ((4.0, 3.0), (0.0, 1.0))),
        ),
        dtype=torch.float64,
    )
    inverse_sqrt = torch.tensor(((2.0, 0.0), (0.0, 0.5)), dtype=torch.float64)

    loss, moments, whitened = squared_conditional_moment_loss(
        returns,
        inverse_sqrt,
    )

    expected_moments = torch.tensor(((1.0, 2.0), (1.0, 1.0)), dtype=torch.float64)
    expected_whitened = torch.tensor(((2.0, 1.0), (2.0, 0.5)), dtype=torch.float64)
    torch.testing.assert_close(moments, expected_moments)
    torch.testing.assert_close(whitened, expected_whitened)
    torch.testing.assert_close(loss, torch.mean(expected_whitened.square()))
    strategy = torch.tensor((0.7, -1.3), dtype=torch.float64)
    torch.testing.assert_close(
        moments @ strategy,
        torch.mean(returns @ strategy, dim=(1, 2)),
    )


def test_zero_option_moments_zero_every_linear_strategy_and_loss() -> None:
    returns = torch.tensor(
        ((((-1.0, 2.0), (1.0, -2.0)),),),
        dtype=torch.float64,
    )
    identity = torch.eye(2, dtype=torch.float64)

    loss, moments, _ = squared_conditional_moment_loss(returns, identity)

    torch.testing.assert_close(moments, torch.zeros_like(moments))
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    strategies = torch.tensor(
        ((1.0, 0.0), (0.0, 1.0), (2.0, -3.0)),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        moments @ strategies.T,
        torch.zeros((1, 3), dtype=torch.float64),
    )


def test_replicated_conditional_moments_report_sample_standard_error() -> None:
    replicate_moments = torch.tensor(
        [
            [[1.0, 4.0], [3.0, 6.0], [5.0, 8.0], [7.0, 10.0]],
            [[2.0, -1.0], [4.0, 1.0], [6.0, 3.0], [8.0, 5.0]],
        ],
        dtype=torch.float64,
    )

    estimate, standard_error = replicated_conditional_moment_statistics(
        replicate_moments
    )

    torch.testing.assert_close(
        estimate,
        torch.tensor([[4.0, 7.0], [5.0, 2.0]], dtype=torch.float64),
    )
    expected = torch.std(replicate_moments, dim=1, correction=1) / 2.0
    torch.testing.assert_close(standard_error, expected)


def test_replicated_conditional_moments_require_independent_replications() -> None:
    with pytest.raises(ValueError, match="at least two replications"):
        replicated_conditional_moment_statistics(torch.zeros((2, 1, 3)))


def test_standard_normal_transport_kl_identity_is_zero() -> None:
    base = torch.tensor(((1.0, -2.0), (0.5, 0.25)), dtype=torch.float64)
    log_abs_det = torch.zeros(2, dtype=torch.float64)

    kl, path_kl = standard_normal_transport_kl(base, base, log_abs_det)

    torch.testing.assert_close(path_kl, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(kl, torch.zeros((), dtype=torch.float64))


def test_standard_normal_transport_kl_has_transport_gradients() -> None:
    base = torch.tensor(
        ((-1.0, 0.5), (0.25, -0.75), (1.5, 0.2)),
        dtype=torch.float64,
    )
    shift = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    scale = torch.exp(log_scale)
    transformed = shift + scale * base
    log_abs_det = torch.full(
        (len(base),),
        base.shape[-1],
        dtype=torch.float64,
    ) * log_scale

    kl, path_kl = standard_normal_transport_kl(
        base,
        transformed,
        log_abs_det,
    )
    kl.backward()

    expected = torch.mean(
        0.5 * torch.sum(transformed.detach().square() - base.square(), dim=-1)
        - log_abs_det.detach()
    )
    torch.testing.assert_close(kl.detach(), expected)
    assert path_kl.shape == (3,)
    assert shift.grad is not None and bool(torch.isfinite(shift.grad))
    assert log_scale.grad is not None and bool(torch.isfinite(log_scale.grad))
