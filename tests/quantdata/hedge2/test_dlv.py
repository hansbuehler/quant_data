"""Backward implicit Dupire must reproduce a discrete martingale exactly."""

from __future__ import annotations

import numpy as np
import pytest

from quantdata.hedge2.dlv import (
    DlvConfig,
    decode_densities,
    positive_reference_density,
    recover_dlvs,
    shared_inner_strikes,
)


def _example(config: DlvConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward_vols = np.full(len(config.business_dtes), 0.2)
    inner = shared_inner_strikes(forward_vols, config=config)
    support = np.r_[0.5 * inner[0], inner, 1.5 * inner[-1]]
    local_variance = np.full((len(config.business_dtes), config.num_inner_strikes), 0.04)
    densities = decode_densities(support, local_variance, config.expiries_y)
    payoff = np.maximum(support[None, :] - support[:, None], 0.)
    return support, densities @ payoff.T, forward_vols


def test_backward_implicit_dupire_recovers_nonuniform_grid() -> None:
    config = DlvConfig(num_inner_strikes=20, business_dtes=(1., 3., 5.))
    support, calls, forward_vols = _example(config)
    recovered = recover_dlvs(support, calls, forward_vols=forward_vols, config=config)
    mask = recovered["identified"]
    np.testing.assert_allclose(recovered["raw_dlvsqs"][mask], 0.04, rtol=2e-5, atol=2e-9)
    assert recovered["mass_error"] < 1e-12
    assert recovered["mean_error"] < 1e-12
    assert recovered["max_repricing_error"] < 1e-11


def test_guarded_extreme_values_report_price_change() -> None:
    original = DlvConfig(num_inner_strikes=20, business_dtes=(1., 3., 5.))
    support, calls, forward_vols = _example(original)
    guarded = DlvConfig(num_inner_strikes=20, business_dtes=(1., 3., 5.), max_local_vol=0.1)
    result = recover_dlvs(support, calls, forward_vols=forward_vols, config=guarded)
    assert result["capped_nodes"] > 0
    assert np.max(result["dlvsqs"]) == pytest.approx(0.01)
    assert result["max_repricing_error"] > 1e-4
    assert not result["reconstruction_passed"]


def test_calendar_arbitrage_is_reported() -> None:
    config = DlvConfig(num_inner_strikes=20, business_dtes=(1., 3., 5.))
    support, calls, forward_vols = _example(config)
    calls[1] = calls[0] - 1e-3
    with pytest.raises(ValueError, match="no-arbitrage"):
        recover_dlvs(support, calls, forward_vols=forward_vols, config=config)


def test_shared_grid_is_fixed_across_expiries_and_floats_with_vol() -> None:
    config = DlvConfig()
    base = shared_inner_strikes(np.full(9, 0.2), config=config)
    shocked = shared_inner_strikes(np.full(9, 0.3), config=config)
    assert base.shape == (200,)
    assert np.sum(base == 1.) == 1
    assert np.all(np.diff(base) > 0)
    assert shocked[0] < base[0] < 1. < base[-1] < shocked[-1]


def test_positive_reference_law_keeps_all_nodes_and_moments() -> None:
    support = np.r_[0.2, np.linspace(0.8, 1.1, 18), 1.8]
    density = positive_reference_density(support)
    assert density.min() > 0
    assert density.sum() == pytest.approx(1., abs=1e-15)
    assert density @ support == pytest.approx(1., abs=1e-15)


def test_zero_variance_is_preserved_separately_from_log_training_floor() -> None:
    config = DlvConfig(num_inner_strikes=20, business_dtes=(1., 3., 5.))
    support, _, forward_vols = _example(config)
    calls = np.broadcast_to(np.maximum(1. - support, 0.), (3, len(support))).copy()
    result = recover_dlvs(support, calls, forward_vols=forward_vols, config=config)
    assert np.max(result["dlvsqs"]) == 0.
    assert np.min(result["training_dlvsqs"]) > 0.
    assert result["zero_zero_nodes"] > 0
    assert result["max_repricing_error"] <= result["cumulative_equation_defect_bound"] + 1e-12
