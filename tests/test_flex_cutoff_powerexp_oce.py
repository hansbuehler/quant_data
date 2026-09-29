from __future__ import annotations

import numpy as np
import torch

from quantdata.dysanos.flex.cutoff_powerexp_oce import (
    cutoff_powerexp_marginal,
    cutoff_powerexp_utility,
    solve_cutoff_powerexp_density,
)
from quantdata.dysanos.flex.direct_utility_frontier import powerexp_utility
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2_triangular import (
    _project_onto_loading_span,
)
from quantdata.dysanos.flex.run_direct_utility_frontier import (
    _correlation_aware_multiplier_test,
)
from quantdata.dysanos.flex.spectral_trading import covariance_supported_coordinates


def test_cutoff_utility_is_flat_above_its_head_limit() -> None:
    values = torch.tensor(
        ((-1.0, 0.0, 0.4, 2.0), (-1.0, 0.0, 0.8, 2.0)),
        dtype=torch.float64,
    )
    cutoffs = torch.tensor((0.5, 1.0), dtype=torch.float64)

    utility = cutoff_powerexp_utility(values, cutoffs)
    marginal = cutoff_powerexp_marginal(values, cutoffs)

    torch.testing.assert_close(
        utility,
        powerexp_utility(torch.minimum(values, cutoffs[:, None])),
    )
    torch.testing.assert_close(
        utility[:, -1],
        powerexp_utility(cutoffs[:, None])[:, 0],
    )
    torch.testing.assert_close(marginal[:, -1], torch.zeros(2, dtype=torch.float64))


def test_cutoff_translation_produces_normalized_nonnegative_density() -> None:
    generator = torch.Generator().manual_seed(81)
    pnl = 0.2 + 0.7 * torch.randn((3, 4096), generator=generator, dtype=torch.float64)
    cutoffs = torch.tensor((0.25, 0.5, 1.0), dtype=torch.float64)

    result = solve_cutoff_powerexp_density(pnl, cutoffs)

    torch.testing.assert_close(
        torch.mean(result.weights, dim=1),
        torch.ones(3, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert bool(torch.all(torch.isfinite(result.weights) & (result.weights >= 0.0)))
    assert bool(torch.all(result.boundary_subgradient_weight >= 0.0))
    assert bool(
        torch.all(
            result.boundary_subgradient_weight
            <= torch.exp(-cutoffs) + 1.0e-10
        )
    )


def test_zero_pnl_has_zero_translation_and_unit_density() -> None:
    pnl = torch.zeros((2, 1024), dtype=torch.float64)
    cutoffs = torch.tensor((0.25, 2.0), dtype=torch.float64)

    result = solve_cutoff_powerexp_density(pnl, cutoffs)

    torch.testing.assert_close(
        result.translation,
        torch.zeros(2, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(result.weights, torch.ones_like(result.weights))


def test_cutoff_density_distributes_subgradient_across_a_tied_boundary_atom() -> None:
    pnl = torch.tensor([[0.0] * 500 + [1.0] * 500], dtype=torch.float64)
    cutoffs = torch.tensor((0.5,), dtype=torch.float64)

    result = solve_cutoff_powerexp_density(pnl, cutoffs)

    torch.testing.assert_close(
        torch.mean(result.weights, dim=1),
        torch.ones(1, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.boundary_subgradient_weight,
        torch.tensor((250.0,), dtype=torch.float64),
        atol=1.0e-10,
        rtol=0.0,
    )
    assert int(torch.sum(result.weights[0, 500:] > 0.0)) == 500


def test_loading_projection_removes_only_unspanned_drift() -> None:
    loadings = torch.tensor(
        ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)),
        dtype=torch.float64,
    )
    values = torch.tensor(
        ((2.0, -1.0, 0.5, 3.0), (-1.0, 4.0, 2.0, -0.5)),
        dtype=torch.float64,
    )

    projected = _project_onto_loading_span(values, loadings)
    residual = values - projected

    torch.testing.assert_close(
        residual @ loadings,
        torch.zeros((2, 2), dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_spectral_coordinates_discard_near_null_modes_without_centering_drift() -> None:
    generator = torch.Generator().manual_seed(142)
    strong = 0.2 + torch.randn(4_000, generator=generator)
    weak = 1.0e-3 * torch.randn(4_000, generator=generator)
    returns = torch.stack((strong, 2.0 * strong, weak), dim=1)

    transformed, spectral = covariance_supported_coordinates(
        (returns,),
        num_independent_blocks=400,
        standard_error_multiple=2.0,
        computation_device="cpu",
        batch_size=500,
    )

    assert transformed[0].shape == (4_000, 1)
    assert spectral.diagnostics[0]["num_retained_modes"] == 1
    assert spectral.diagnostics[0]["num_discarded_modes"] == 2
    torch.testing.assert_close(
        torch.var(transformed[0], dim=0, correction=1),
        torch.ones(1),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert float(torch.abs(torch.mean(transformed[0]))) > 0.1


def test_zero_variance_multiplier_test_distinguishes_zero_and_nonzero_mean() -> None:
    zero = _correlation_aware_multiplier_test(
        np.zeros((20, 2)),
        samples=99,
        seed=7,
    )
    nonzero = _correlation_aware_multiplier_test(
        np.ones((20, 2)),
        samples=99,
        seed=7,
    )

    assert zero["combined_p_value"] == 1.0
    assert zero["zero_mean_not_rejected_at_five_percent"]
    assert nonzero["combined_p_value"] == 0.02
    assert not nonzero["zero_mean_not_rejected_at_five_percent"]
