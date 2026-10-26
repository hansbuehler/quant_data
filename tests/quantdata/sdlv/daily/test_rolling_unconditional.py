from datetime import date
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.sdlv.daily import rolling_unconditional
from quantdata.sdlv.daily.front_spot import estimation, run_oce
from quantdata.sdlv.daily.front_spot.estimation import JointInnovationFit, rolling_history_indices
from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower
from quantdata.sdlv.daily.front_spot.simulation import InitialModelState
from quantdata.sdlv.daily.rolling_unconditional import (
    audit_from_moments,
    merge_moments,
    pair_moments,
    reuse_completed_chunks,
    validate_resume_identity,
)


def test_backend_continuation_preserves_numerical_identity():
    driver = str(Path(rolling_unconditional.__file__).resolve())
    previous = dict(seed=123, paths=100000, code_sha256={driver: rolling_unconditional._GPU_ONLY_DRIVER_SHA256,
                                                       "jax_model.py": "unchanged-kernel"})
    current = dict(seed=123, paths=100000, compute_backend="cpu",
                   code_sha256={driver: "reviewed-new-driver", "jax_model.py": "unchanged-kernel"})
    validate_resume_identity(previous, current)
    with pytest.raises(ValueError, match="Resume identity changed"):
        validate_resume_identity(previous, {**current, "seed": 124})
    with pytest.raises(ValueError, match="Resume identity changed"):
        validate_resume_identity(previous, {**current, "code_sha256": {driver: "reviewed-new-driver",
                                                                       "jax_model.py": "changed-kernel"}})
    with pytest.raises(ValueError, match="Unreviewed resume driver"):
        validate_resume_identity({**previous, "code_sha256": {driver: "unknown-driver"}}, current)


def test_backend_continuation_copies_only_complete_contiguous_chunks(tmp_path):
    source, output = tmp_path / "gpu", tmp_path / "cpu"
    (source / "pair-moments").mkdir(parents=True)
    arrays = dict(pair_count=np.array(4), mean=np.ones((20, 45)), m2=np.ones((20, 45)),
                  active_mask=np.ones((20, 45), dtype=bool))
    original = source / "pair-moments/000000.npz"
    np.savez(original, **arrays)
    # A later file must not make a missing middle chunk look completed.
    np.savez(source / "pair-moments/000016.npz", **arrays)
    imported = reuse_completed_chunks(source, output, paths=24, chunk_size=8)
    assert imported["reused_paths"] == 8
    assert imported["remaining_paths"] == 16
    assert (output / "pair-moments/000000.npz").read_bytes() == original.read_bytes()
    assert not (output / "pair-moments/000016.npz").exists()
    assert reuse_completed_chunks(source, output, paths=24, chunk_size=8) == imported
    np.savez(source / "pair-moments/000008.npz", **{**arrays, "pair_count": np.array(3)})
    with pytest.raises(ValueError, match="incomplete pair count"):
        reuse_completed_chunks(source, output, paths=24, chunk_size=8)
    with pytest.raises(ValueError, match="distinct output"):
        reuse_completed_chunks(source, source, paths=24, chunk_size=8)


def test_window_is_252_dates_strictly_before_reference():
    dates = tuple(np.busday_offset("2024-01-02", np.arange(280)).astype(str))
    selected = rolling_history_indices(dates, reference_date=date.fromisoformat(dates[270]))
    np.testing.assert_array_equal(selected, np.arange(18, 270))
    with pytest.raises(ValueError, match="need 252"):
        rolling_history_indices(dates, reference_date=date.fromisoformat(dates[251]))
    with pytest.raises(ValueError, match="unique"):
        rolling_history_indices((dates[0], dates[0]), reference_date=date(2025, 1, 1))


def test_window_bounds_both_ends_of_exact_session_pairs(tmp_path, monkeypatch):
    count = 280
    dates = np.busday_offset("2024-01-02", np.arange(count)).astype(str)
    support = np.array([0.5, 0.85, 1, 1.15, 1.7])
    raw = np.full((count, 25, 3), -3, dtype=np.float32)
    dataset = tmp_path / "market.npz"
    np.savez(dataset, dates=dates, supports=np.tile(support, (count, 1)), spot_ref=np.full(count, 100.0))
    # Missing one true-session transition is not bridged as a one-day return.
    origins = np.delete(np.arange(count - 1), 120)
    monkeypatch.setattr(estimation, "one_session_pairs", lambda _: (origins, origins + 1))
    history = estimation.align_fixed_contract_history(dataset_path=dataset, raw_grids=raw,
                                                       training_start=date.fromisoformat(dates[18]),
                                                       training_end=date.fromisoformat(dates[269]))
    assert history.origin_indices.min() == 18
    assert history.endpoint_indices.max() == 269
    assert len(history.origin_indices) == 250
    assert 120 not in history.origin_indices


def test_streaming_moments_match_paper_unconditional_tests():
    rng = np.random.default_rng(142)
    returns = rng.normal(0.001, 0.1, (120, 3, 4)).astype(np.float32)
    active = np.ones((3, 4), dtype=bool)
    active[2, 2:] = False
    returns[:, ~active] = 0
    labels = ("spot", "option1", "option2", "option3")
    combined = (0, np.zeros((3, 4)), np.zeros((3, 4)))
    for block in np.split(returns, 3):
        combined = merge_moments(combined, pair_moments(block))
    actual, arrays = audit_from_moments(combined, active, labels)
    expected = run_oce.unconditional_spot_scaled_martingale_audit(returns, active, labels)
    pairs = (returns[::2].astype(float) + returns[1::2].astype(float)) / 2
    np.testing.assert_allclose(arrays["mean"], pairs.mean(axis=0), atol=1e-16)
    np.testing.assert_allclose(arrays["pair_based_standard_error"], pairs.std(axis=0, ddof=1) / np.sqrt(60))
    for group in ("all_instruments", "spot", "options"):
        for key, value in expected[group].items():
            if key != "top_absolute_z_coordinates":
                assert actual[group][key] == pytest.approx(value)


def test_omitting_policy_features_keeps_identical_Q_returns():
    support = np.array([0.5, 0.85, 1, 1.15, 1.7])
    dlvsq = np.full((25, 3), 0.04)
    tower = np.asarray(decode_density_tower(jnp.asarray(support), jnp.asarray(dlvsq), dt=1 / 252))
    state = InitialModelState("2025-08-29", 0, support, dlvsq, tower)
    innovations = JointInnovationFit(
        conditional_intercept=np.zeros((24, 3)), conditional_slope=np.zeros((24, 3)),
        centered_residuals=np.zeros((3, 72)), spot_score_mean=0, spot_score_std=1,
        annualized_loadings=np.full((24, 3, 2), 0.05), daily_eigenvalues=np.ones(2),
        explained_variance_fraction=1, factor_scores=np.zeros((3, 2)))
    args = dict(innovations=innovations, initial_state=state, path_count=4, batch_size=4, dlv_substeps=1, seed=271)
    with_features = run_oce.generate_front_spot_oce_population(None, **args)
    without = run_oce.generate_front_spot_oce_population(None, record_features=False, **args)
    np.testing.assert_array_equal(with_features.returns, without.returns)
    np.testing.assert_array_equal(with_features.strikes, without.strikes)
    assert without.features.shape == (4, 20, 0)
