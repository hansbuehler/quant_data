"""Causal score enrichment uses preceding surfaces and preserves empirical PIT laws."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from quantdata.hedge2 import front_scores
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.front_spot import gaussian_scores
from tests.quantdata.hedge2.test_pricing import _pricer


def test_pair_uses_previous_model_and_distinguishes_parity_from_financing(tmp_path, monkeypatch) -> None:
    before, after = date(2024, 4, 1), date(2024, 4, 2)
    source_file = tmp_path / "saved-model.h5"
    source_file.write_bytes(b"previous certified source")
    (tmp_path / "SPX").mkdir()
    (tmp_path / "SPX" / f"{before}.pdq").write_bytes(b"previous licensed curve")
    (tmp_path / "days").mkdir()
    (tmp_path / "days" / f"{before}.npz").write_bytes(b"previous DLV state")
    curve = CashCurve(spot=100., business_dtes=np.array([2., 5.]), forwards=np.array([102., 104.]),
        reference_forwards=np.array([101., 102.]), discounts=np.array([.99, .98]))
    source = SimpleNamespace(expiries=np.array([2. / 252., 5. / 252.]))
    calls = []

    def load(day):
        calls.append(("model", day))
        return source, {"source_model": str(source_file), "source_sha256": front_scores._sha256(source_file)}

    def licensed(directory, day):
        assert directory == tmp_path
        calls.append(("curve", day))
        return curve

    def historical(**kwargs):
        assert kwargs["previous_source"] is source
        assert kwargs["previous_parity_forward"] == pytest.approx(curve.at(1.)[0])
        assert kwargs["realized_spot"] == 100.4
        return gaussian_scores(np.array([.7]))

    def state(directory, day):
        assert directory == tmp_path
        calls.append(("state", day))
        return SimpleNamespace(decode=lambda: SimpleNamespace(pricer=_pricer(), lower_clips=0, upper_clips=0))

    monkeypatch.setattr(front_scores, "licensed_curve", licensed)
    monkeypatch.setattr(front_scores, "historical_gaussian_score", historical)
    monkeypatch.setattr(front_scores, "normalize_surface_moments", lambda value: {
        "source_mass_error": 1e-8, "source_mean_error": 2e-8})
    monkeypatch.setattr(front_scores, "DailyState", SimpleNamespace(load=state))
    row = front_scores.front_score_record(archive=SimpleNamespace(load=load), licensed_fit_directory=tmp_path,
        daily_directory=tmp_path, before=before, after=after, realized_spot=100.4, previous_spot=100.)
    assert calls == [("curve", before), ("model", before), ("state", before)]
    assert row["date"] == str(after)
    assert row["sanos_gaussian_score"] == pytest.approx(ndtri(.7))
    assert row["previous_parity_forward_1d"] == pytest.approx(np.sqrt(100. * 102.))
    assert row["previous_reference_forward_1d"] == pytest.approx(np.sqrt(100. * 101.))
    assert row["dlv_front_mean"] == pytest.approx(row["previous_parity_forward_1d"], abs=1e-10)
    assert abs(row["dlv_front_relative_mean_residual"]) < 1e-12
    assert row["normalized_spot_return_into_date"] == pytest.approx(100.4 / np.sqrt(100. * 101.) - 1.)
    assert row["source_short_expiry_extrapolation"] is True
    assert row["source_mass_error"] == 1e-8
    assert row["dlv_front_variance"] > 0.


@pytest.mark.parametrize("before,after", [
    (date(2024, 4, 1), date(2024, 4, 3)),
    (date(2024, 4, 6), date(2024, 4, 9)),
    (date(2024, 4, 2), date(2024, 4, 1)),
])
def test_pair_rejects_nonconsecutive_sessions_before_loading(before, after) -> None:
    with pytest.raises(ValueError, match="exact business-session"):
        front_scores.front_score_record(archive=None, licensed_fit_directory=None, daily_directory=None,
            before=before, after=after, realized_spot=100., previous_spot=100.)


def _inputs(tmp_path, monkeypatch):
    observations = tmp_path / "observations.parquet"
    pd.DataFrame({"date": [date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 4), date(2024, 4, 5)],
        "spot": [100., 101., 102., 103.]}).to_parquet(observations, index=False)
    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "config.json").write_text("{}")
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable previous input")
    calls = []

    def record(**kwargs):
        calls.append((kwargs["before"], kwargs["after"]))
        return {"date": str(kwargs["after"]), "pair_start": str(kwargs["before"]),
            "spot": kwargs["realized_spot"], "previous_spot": kwargs["previous_spot"],
            "source_short_expiry_extrapolation": False, "dlv_realization_outside_support": False,
            "dlv_front_relative_mean_residual": 1e-15,
            "sanos_gaussian_score": 2., "dlv_gaussian_score": -3.,
            "sanos_lower_clipped": False, "sanos_upper_clipped": False,
            "dlv_lower_clipped": True, "dlv_upper_clipped": False,
            "inputs": {"previous": {"path": str(source), "sha256": front_scores._sha256(source)}}}

    monkeypatch.setattr(front_scores, "front_score_record", record)
    arguments = dict(archive=SimpleNamespace(root=tmp_path), licensed_fit_directory=tmp_path,
        daily_directory=daily, observations_path=observations, output_directory=tmp_path / "output",
        start=date(2024, 4, 1), end=date(2024, 4, 5))
    return arguments, calls, source


def test_panel_excludes_missing_sessions_resumes_and_preserves_score_mean(tmp_path, monkeypatch) -> None:
    arguments, calls, _ = _inputs(tmp_path, monkeypatch)
    result = front_scores.prepare_front_scores(**arguments)
    assert result["pairs"] == 2
    assert calls == [(date(2024, 4, 1), date(2024, 4, 2)), (date(2024, 4, 4), date(2024, 4, 5))]
    assert result["sanos"]["mean"] == 2.
    assert result["dlv"]["mean"] == -3.
    assert result["dlv"]["lower_clipped"] == 2
    assert result["dlv"]["std_population"] == 0.
    front_scores.prepare_front_scores(**arguments)
    assert len(calls) == 2
    saved = pd.read_parquet(arguments["output_directory"] / "scores.parquet")
    assert saved["pair_start"].tolist() == ["2024-04-01", "2024-04-04"]


def test_resume_rejects_changed_causal_input(tmp_path, monkeypatch) -> None:
    arguments, _, source = _inputs(tmp_path, monkeypatch)
    front_scores.prepare_front_scores(**arguments)
    source.write_bytes(b"a different previous input")
    with pytest.raises(ValueError, match="Cached front-score input changed"):
        front_scores.prepare_front_scores(**arguments)


def test_resume_rejects_different_score_semantics(tmp_path, monkeypatch) -> None:
    arguments, _, _ = _inputs(tmp_path, monkeypatch)
    front_scores.prepare_front_scores(**arguments)
    with pytest.raises(ValueError, match="different inputs or implementation"):
        front_scores.prepare_front_scores(**arguments, probability_floor=1e-5)


def _alignment_inputs():
    frame = pd.DataFrame({"date": ["2024-04-02", "2024-04-03", "2024-04-04"],
        "pair_start": ["2024-04-01", "2024-04-02", "2024-04-03"],
        "sanos_gaussian_score": [.1, .8, np.nan], "dlv_gaussian_score": [99., 99., 99.]})
    model = SimpleNamespace(training_start=date(2024, 4, 1), training_end=date(2024, 4, 3),
        as_of_date=date(2024, 4, 3), residuals=np.zeros((2, 3)))
    return model, frame


def test_alignment_uses_smooth_scores_in_residual_order_and_excludes_future() -> None:
    model, frame = _alignment_inputs()
    values = front_scores.aligned_front_scores(model=model, scores=frame)
    np.testing.assert_array_equal(values, [.1, .8])
    assert not values.flags.writeable
    frame.loc[2, "sanos_gaussian_score"] = 1000.
    np.testing.assert_array_equal(front_scores.aligned_front_scores(model=model, scores=frame), values)


def test_alignment_rejects_missing_innovation_and_non_session_pair() -> None:
    model, frame = _alignment_inputs()
    with pytest.raises(ValueError, match="every fitted innovation"):
        front_scores.aligned_front_scores(model=model, scores=frame.iloc[1:])
    frame.loc[1, "pair_start"] = "2024-04-01"
    with pytest.raises(ValueError, match="exact business-session"):
        front_scores.aligned_front_scores(model=model, scores=frame)


def test_alignment_rejects_nonfinite_training_score_and_future_fit() -> None:
    model, frame = _alignment_inputs()
    frame.loc[0, "sanos_gaussian_score"] = np.nan
    with pytest.raises(ValueError, match="finite smooth-SANOS"):
        front_scores.aligned_front_scores(model=model, scores=frame)
    model.as_of_date = date(2024, 4, 2)
    with pytest.raises(ValueError, match="causal training range"):
        front_scores.aligned_front_scores(model=model, scores=frame)
