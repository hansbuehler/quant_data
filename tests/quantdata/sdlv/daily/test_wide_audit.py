from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily import wide_audit
from quantdata.sdlv.daily.front_spot import calibration, estimation, simulation
from quantdata.sdlv.daily.front_spot.estimation import decode_dlvsq
from quantdata.sdlv.daily.front_spot.run_oce import FrontSpotOcePopulation


def test_explicit_initial_state_uses_wider_inputs_without_paper_loader(tmp_path, monkeypatch):
    support = np.array([0.5, 0.85, 1.0, 1.15, 1.7])
    dataset = tmp_path / "market.npz"
    np.savez(dataset, dates=np.array(["2025-08-29"]), supports=support[None])
    raw = np.full((1, 25, 3), np.log(0.04 - 0.0001), dtype=np.float32)

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not silently use retained narrow-grid weights")

    monkeypatch.setattr(simulation, "load_raw_grid", forbidden)
    state = simulation.load_initial_model_state(dataset_path=dataset, raw_grids=raw)
    np.testing.assert_array_equal(state.support, support)
    np.testing.assert_allclose(state.dlvsq, decode_dlvsq(raw[0]))
    np.testing.assert_allclose(state.density_tower.sum(axis=-1), 1, atol=1e-12)
    np.testing.assert_allclose(state.density_tower @ support, 1, atol=1e-12)
    with pytest.raises(ValueError, match="geometry"):
        simulation.load_initial_model_state(dataset_path=dataset, raw_grids=raw[:, :, :2])


def _source_fixture(tmp_path):
    run = tmp_path / "training"
    run.mkdir()
    dates = np.array(["2022-01-03"] * 890 + ["2025-08-29"])
    np.savez(run / "market.npz", dates=dates, supports=np.tile([0.5, 0.9, 1, 1.1, 1.5], (891, 1)),
             model_times_y=np.arange(1, 26) / DAYS_PER_YEAR, backbone_annual_vol=0.05)
    identity = dict(dataset_sha256=wide_audit.sha256(run / "market.npz"), model_label="Gromm-v1-4,3")
    snapshot = tmp_path / "snapshot.npz"
    np.savez(snapshot, raw=np.full((891, 25, 3), -3, dtype=np.float32),
             identity=np.array(json.dumps(identity)), epoch=np.array(620))
    output = tmp_path / "audit"
    output.mkdir()
    return run, snapshot, output


def test_snapshot_keeps_interim_status_and_checks_dataset_hash(tmp_path):
    run, snapshot, output = _source_fixture(tmp_path)
    raw, source = wide_audit.load_source(run, output, snapshot)
    assert raw.shape == (891, 25, 3)
    assert source["fit_status"] == "interim"
    assert source["fit_epoch"] == 620
    assert source["checkpoint_sha256"] == wide_audit.sha256(snapshot)
    with (run / "market.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="does not belong"):
        wide_audit.load_source(run, output, snapshot)


def test_final_audit_requires_completed_training_report(tmp_path):
    run, _, output = _source_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        wide_audit.load_source(run, output, None)


def test_full_diffusion_calibration_never_loads_old_fits(tmp_path, monkeypatch):
    count = 122
    dates = np.busday_offset("2024-01-02", np.arange(count)).astype(str)
    generator = np.random.default_rng(109)
    support = np.array([0.5, 0.85, 1, 1.15, 1.7])
    raw = (-3 + generator.normal(0, 0.1, (count, 25, 3))).astype(np.float32)
    dataset = tmp_path / "wide-market.npz"
    np.savez(dataset, dates=dates, supports=np.tile(support, (count, 1)),
             spot_ref=100 * np.exp(np.cumsum(generator.normal(0, 0.01, count))))

    def forbidden(*args, **kwargs):
        raise AssertionError("New diffusion calibration must not read old fitted weights")

    monkeypatch.setattr(estimation, "load_raw_grids", forbidden)
    monkeypatch.setattr(calibration, "load_raw_grids", forbidden)
    monkeypatch.setattr(estimation, "one_session_pairs", lambda _: (np.arange(count - 1), np.arange(1, count)))
    fitted = calibration.calibrate_front_spot_model(dataset_path=dataset, raw_grids=raw)
    assert fitted.innovations.annualized_loadings.shape == (24, 3, 8)
    assert np.all(np.isfinite(fitted.innovations.annualized_loadings))
    np.testing.assert_allclose(fitted.history.origin_log_dlvsq, np.log(decode_dlvsq(raw[:-1, 1:])))
    assert fitted.history.dates == tuple(dates)


@pytest.mark.parametrize("objective", ["squared", "entropy"])
def test_conditional_driver_combines_complete_pairs(tmp_path, objective):
    generator = np.random.default_rng(571)
    features = generator.normal(size=(16, 2, 3)).astype(np.float32)
    returns = generator.normal(0.002, 0.01, size=(16, 2, 2)).astype(np.float32)
    active = np.ones((2, 2), dtype=bool)
    population = FrontSpotOcePopulation(
        features=features, returns=returns, active_mask=active, strikes=np.array([[1.0]]),
        instrument_labels=("spot", "call"), instrument_universe="restricted",
        initial_date="2025-08-29", generation_seconds=0.0,
    )
    output = tmp_path / objective
    monitor = wide_audit.AuditMonitor(tmp_path)
    result = wide_audit.conditional_audit(population, output=output, objective=objective,
                                        epochs=1, device="cpu", monitor=monitor)
    scores = np.load(output / "out-of-fold-pair-scores.npy")
    assert scores.shape == (8,)
    combined = result["combined_out_of_fold"]
    assert combined["antithetic_pair_count"] == 8
    assert combined["mean_daily_conditional_drift"] == pytest.approx(scores.mean())
    assert combined["pair_based_standard_error"] == pytest.approx(scores.std(ddof=1) / np.sqrt(8))
    assert result["heldout"]["A"]["antithetic_pair_count"] == 4
    assert result["heldout"]["B"]["antithetic_pair_count"] == 4


def _rolling_fixture(tmp_path, monkeypatch):
    dates = np.busday_offset("2025-08-29", np.arange(-252, 1)).astype(str)
    dataset = tmp_path / "market.npz"
    np.savez(dataset, dates=dates)
    source = dict(model_label="Gromm-v1-4,3", dataset=str(dataset), dataset_sha256="dataset",
                  checkpoint_sha256="checkpoint", training_report_sha256="report", initial_date=dates[-1])
    initial = simulation.InitialModelState(date=dates[-1], date_index=252,
                                          support=np.array([0.5, 1.0, 1.5]), dlvsq=np.ones((25, 1)),
                                          density_tower=np.tile([0.25, 0.5, 0.25], (25, 1)))
    innovations = estimation.JointInnovationFit(
        conditional_intercept=np.zeros((24, 1)), conditional_slope=np.zeros((24, 1)),
        centered_residuals=np.zeros((251, 24, 1)), spot_score_mean=0.0, spot_score_std=1.0,
        annualized_loadings=np.ones((24, 1, 8)), daily_eigenvalues=np.ones(8),
        explained_variance_fraction=0.75, factor_scores=np.zeros((251, 8)),
    )
    history = SimpleNamespace(origin_indices=np.arange(251), endpoint_indices=np.arange(1, 252))
    monkeypatch.setattr(wide_audit, "align_fixed_contract_history", lambda **kwargs: history)
    monkeypatch.setattr(wide_audit, "fit_joint_innovations", lambda *args, **kwargs: innovations)
    monkeypatch.setattr(wide_audit, "load_initial_model_state", lambda **kwargs: initial)
    identity = dict(schema="gromm-wide-rolling252-unconditional-v1", window_date_count=252,
                    horizon_days=20, factor_rank=8, dlv_substeps=8, source=source,
                    reference_date=dates[-1], window_dates=list(dates[:-1]),
                    window_start=dates[0], window_end=dates[-2],
                    code_sha256={str(path.resolve()): wide_audit.sha256(path)
                                 for path in (wide_audit.Path(wide_audit.__file__).parent / "front_spot").glob("*.py")})
    summary = dict(reference_date=dates[-1], window_start=dates[0], window_end=dates[-2])
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    (tmp_path / "calibration.json").write_text(json.dumps(summary))
    np.savez(tmp_path / "diffusion.npz", annualized_loadings=innovations.annualized_loadings + 1e-13,
             eigenvalues=innovations.daily_eigenvalues, support=initial.support, dlvsq=initial.dlvsq,
             density_tower=initial.density_tower,
             origin_indices=history.origin_indices, endpoint_indices=history.endpoint_indices)
    return source, identity, innovations


def test_rolling_full_audit_retains_exact_saved_diffusion(tmp_path, monkeypatch):
    source, _, recomputed = _rolling_fixture(tmp_path, monkeypatch)
    initial, fitted, summary = wide_audit.load_rolling_calibration(tmp_path, np.zeros(0), source)
    assert initial.date == "2025-08-29"
    assert summary["window_end"] == "2025-08-28"
    np.testing.assert_array_equal(fitted.annualized_loadings, recomputed.annualized_loadings + 1e-13)
    assert np.any(fitted.annualized_loadings != recomputed.annualized_loadings)


@pytest.mark.parametrize("change,match", [("source", "source mismatch"),
                                          ("window", "strictly past"),
                                          ("code", "numerical code changed"),
                                          ("geometry", "geometry")])
def test_rolling_full_audit_rejects_incompatible_inputs(tmp_path, monkeypatch, change, match):
    source, identity, _ = _rolling_fixture(tmp_path, monkeypatch)
    if change == "source":
        source = dict(source, checkpoint_sha256="other fit")
    elif change == "window":
        identity["window_dates"][-1] = identity["reference_date"]
    elif change == "code":
        identity["code_sha256"] = {}
    else:
        identity["factor_rank"] = 16
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match=match):
        wide_audit.load_rolling_calibration(tmp_path, np.zeros(0), source)
