"""Checks for the zero-cost minimum-variance hedge limit."""

from __future__ import annotations

import json
from datetime import date

import numpy as np

from quantdata.hedging.minimum_variance_daily import (
    _greek_limit,
    minimum_normal_actions,
    minimum_variance_actions,
)


def test_empirical_minimum_variance_is_uncosted_least_squares() -> None:
    rng = np.random.default_rng(7)
    instruments = rng.normal(size=(100, 3))
    target = instruments @ np.array([0.5, -0.2, 0.3])
    actions, diagnostics = minimum_variance_actions(
        instruments, target[:, None], position_cap=1_000.0
    )
    np.testing.assert_allclose(actions[:, 0], [-0.5, 0.2, -0.3], atol=1.0e-12)
    assert diagnostics[0]["training_variance"] < 1.0e-24
    assert diagnostics[0]["cap_hits"] == 0


def test_empirical_minimum_variance_respects_executable_cap() -> None:
    rng = np.random.default_rng(8)
    instruments = rng.normal(size=(100, 2))
    target = 10.0 * instruments[:, 0]
    actions, diagnostics = minimum_variance_actions(
        instruments, target[:, None], position_cap=1.0
    )
    assert np.max(np.abs(actions)) <= 1.0 + 1.0e-10
    assert diagnostics[0]["solver"] == "bounded-lsq"
    assert diagnostics[0]["cap_hits"] >= 1


def test_degenerate_minimum_variance_prefers_cheaper_instrument() -> None:
    rng = np.random.default_rng(10)
    shock = rng.normal(size=100)
    instruments = np.column_stack((shock, shock))
    actions, diagnostics = minimum_variance_actions(
        instruments,
        shock[:, None],
        position_cap=10.0,
        linear_transaction_costs=np.array([0.01, 1.0]),
    )
    assert abs(actions[0, 0] + 1.0) < 1.0e-5
    assert abs(actions[1, 0]) < 1.0e-5
    assert diagnostics[0]["matrix_rank"] == 1
    assert "tie-break" in diagnostics[0]["solver"]


def test_robust_normal_limit_uses_direct_covariance_solve() -> None:
    rng = np.random.default_rng(9)
    instruments = rng.normal(size=(252, 3))
    target = instruments @ np.array([0.5, -0.2, 0.3]) + rng.normal(
        scale=0.01, size=252
    )
    actions, diagnostics = minimum_normal_actions(
        instruments, target[:, None], position_cap=1_000.0
    )
    assert actions.shape == (3, 1)
    assert np.all(np.isfinite(actions))
    assert diagnostics[0]["solver"] == "direct-covariance"
    assert diagnostics[0]["tyler_shrinkage"] > 0.0


def test_robust_normal_spot_residual_pca_limit() -> None:
    rng = np.random.default_rng(11)
    spot = rng.normal(size=(252, 1))
    options = 0.5 * spot + rng.normal(size=(252, 12))
    instruments = np.column_stack((spot, options))
    target = 0.8 * spot[:, 0] + options[:, 0]
    actions, diagnostics = minimum_normal_actions(
        instruments,
        target[:, None],
        position_cap=1_000.0,
        spot_residual_pca_rank=8,
    )
    assert actions.shape == (13, 1)
    assert diagnostics[0]["spot_residual_pca_rank"] == 8
    assert 0.0 < diagnostics[0]["explained_residual_variance"] < 1.0


def test_factor_limit_replaces_finite_parametric_record(tmp_path) -> None:
    origin_date = date(2024, 4, 1)
    greek = {
        "model": "sanos-greek-ordinary",
        "diagnostics": {"factor_residual_norm": 0.0},
        "cap_hits": 0,
        "objective_value": 1.0,
    }
    finite = {**greek, "model": "sanos-parametric-ordinary"}
    (tmp_path / f"{origin_date}.json").write_text(json.dumps([greek, finite]))
    records = _greek_limit(tmp_path, origin_date)
    assert len(records) == 2
    assert records[1]["model"] == "sanos-parametric-ordinary"
    assert records[1]["objective_value"] == 0.0
