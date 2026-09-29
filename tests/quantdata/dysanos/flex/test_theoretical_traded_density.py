from __future__ import annotations

import torch

from quantdata.dysanos.flex.theoretical_traded_density import (
    exponential_log_density_increment,
    theoretical_exponential_action,
    viable_exponential_action,
)


def test_theoretical_action_replicates_factor_shift_with_minimum_norm() -> None:
    loadings = torch.tensor(
        (
            ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            ((2.0, 0.0), (0.0, 0.5), (1.0, -1.0)),
        ),
        dtype=torch.float64,
    )
    factor_shift = torch.tensor(((0.2, -0.3), (-0.4, 0.1)), dtype=torch.float64)

    result = theoretical_exponential_action(loadings, factor_shift)

    torch.testing.assert_close(result.replicated_factor_shift, factor_shift)
    torch.testing.assert_close(
        result.factor_replication_residual,
        torch.zeros_like(factor_shift),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        result.action,
        torch.linalg.lstsq(loadings.transpose(1, 2), factor_shift[..., None]).solution[..., 0],
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_traded_density_equals_exact_gaussian_girsanov_density() -> None:
    dtype = torch.float64
    loadings = torch.tensor(
        (((1.0, 0.2), (0.1, 0.7), (-0.4, 0.3)),),
        dtype=dtype,
    ).expand(5, -1, -1)
    factor_shift = torch.tensor(((0.3, -0.2),), dtype=dtype).expand(5, -1)
    innovation = torch.tensor(
        ((0.5, -1.2), (-0.3, 0.8), (1.1, 0.2), (-0.7, -0.4), (0.0, 1.4)),
        dtype=dtype,
    )
    dt = 0.025
    result = theoretical_exponential_action(loadings, factor_shift)
    return_drift = torch.einsum("bif,bf->bi", loadings, factor_shift)
    realized_return = dt * return_drift + dt**0.5 * torch.einsum(
        "bif,bf->bi", loadings, innovation
    )

    actual = exponential_log_density_increment(
        result.action,
        realized_return,
        return_drift,
        time_increment=dt,
    )
    expected = -dt**0.5 * torch.sum(factor_shift * innovation, dim=1) - 0.5 * dt * torch.sum(
        factor_shift.square(), dim=1
    )

    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_theoretical_action_reports_an_unspanned_factor_shift() -> None:
    loadings = torch.tensor((((1.0, 0.0), (2.0, 0.0)),), dtype=torch.float64)
    factor_shift = torch.tensor(((0.2, 0.4),), dtype=torch.float64)

    result = theoretical_exponential_action(loadings, factor_shift)

    torch.testing.assert_close(
        result.replicated_factor_shift,
        torch.tensor(((0.2, 0.0),), dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.factor_replication_residual,
        torch.tensor(((0.0, -0.4),), dtype=torch.float64),
    )
    assert int(result.numerical_rank[0]) == 1


def test_viable_action_shrinks_weak_modes_and_obeys_both_limits() -> None:
    loadings = torch.tensor(
        (((1.0, 0.0), (0.0, 1.0e-4), (0.5, 0.0)),),
        dtype=torch.float64,
    )
    requested = torch.tensor(((3.0, 4.0),), dtype=torch.float64)

    result = viable_exponential_action(
        loadings,
        requested,
        traded_ridge_ratio=1.0e-6,
        maximum_factor_shift_norm=2.0,
        maximum_action_l1=5.0,
    )

    torch.testing.assert_close(
        result.theoretical_action.replicated_factor_shift,
        result.viable_factor_shift,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert float(torch.linalg.vector_norm(result.viable_factor_shift, dim=1)[0]) <= 2.0 + 1.0e-12
    assert float(torch.sum(torch.abs(result.theoretical_action.action), dim=1)[0]) <= 5.0 + 1.0e-12
    assert float(torch.abs(result.viable_factor_shift[0, 1])) < 0.01
    assert 0.0 < float(result.factor_retention_fraction[0]) < 1.0
