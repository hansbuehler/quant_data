from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from quantdata.gromm2.geometry import transport_tower
from quantdata.gromm2.legacy_forward_model import load_legacy_ar1_model


def legacy_files(directory: Path) -> dict[str, Path | str]:
    directory.mkdir(parents=True, exist_ok=True)
    history, estimate, factor_law = (directory / name for name in ("history.h5", "estimate.npz", "factor-laws.npz"))
    dates = np.asarray(["2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29"], dtype="S10")
    supports = np.asarray([[0.5, 0.91, 1.0, 1.09, 1.5], [0.49, 0.89, 1.0, 1.11, 1.51], [0.48, 0.87, 1.0, 1.13, 1.52]])
    with h5py.File(history, "w") as stream:
        stream.attrs.update(schema="gromm2-affine-dlv-history-v1", geometry_config=json.dumps({"grid_nms_range": [-3, 2], "backbone_annual_volatility": 0.05}),
                            calibration_config=json.dumps({"backbone_annual_volatility": 0.05}), percent_inside_spread=83.5,
                            mean_excess_half_spreads=0.17, midpoint_mae_half_spreads=0.63)
        for name, values in {"dates": dates, "supports": supports.astype("f4"), "transports": transport_tower(supports).astype("f4"),
                             "model_expiries_y": (np.arange(1, 4) / 252.0).astype("f4"), "model_business_dtes": np.arange(1, 4),
                             "nms_nodes": np.asarray([-3.0, 0.0, 2.0]), "dlvsq": np.full((4, 3, 3), 0.04, dtype="f4")}.items():
            stream.create_dataset(name, data=values)
    scores = np.asarray([[0.2, 0.4], [-0.3, 0.2], [0.1, 0.5]], dtype="f4")
    np.savez(estimate, log_dlvsq_loadings=np.full((2, 3, 2), 0.1, dtype="f4"), factor_scores=scores, pair_origin_indices=np.arange(3))
    estimate.with_suffix(".json").write_text(json.dumps({"schema": "gromm2-estimate-v1", "factor_count": 2, "pair_count": 3}))
    config = {"log_scale_bound": float(np.log(3)), "squared_score_floor": 0.0025}
    np.savez(factor_law, schema=np.asarray("gromm2-factor-laws-v1"), config=np.asarray(json.dumps(config)), dates=dates[:-1], origin_indices=np.arange(3),
             fixed_factor_scales=np.ones(2), ar1_parameter_variance_intercept=np.full(2, -1.0), ar1_parameter_variance_slope=np.full(2, 0.1),
             ar1_parameter_scale_calibration=np.ones(2), ar1_terminal_1=np.log(scores[-1].astype("f8") ** 2 + config["squared_score_floor"]))
    report = {"schema": "gromm2-factor-laws-v1", "config": config, "factor_count": 2, "pair_count": 3,
              "validation_end_date": "2025-08-28", "base_estimate_sha256": hashlib.sha256(estimate.read_bytes()).hexdigest(),
              "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest()}
    factor_law.with_suffix(".json").write_text(json.dumps(report))
    return {"history": history, "estimate": estimate, "factor_law": factor_law, "origin": "2025-08-29"}


def refresh_history_hash(arguments: dict[str, Path | str]) -> None:
    path = Path(arguments["factor_law"]).with_suffix(".json")
    report = json.loads(path.read_text())
    report["history_sha256"] = hashlib.sha256(Path(arguments["history"]).read_bytes()).hexdigest()
    path.write_text(json.dumps(report))


def test_adapter_retains_state_and_reconstructs_exact_geometry(tmp_path: Path) -> None:
    arguments = legacy_files(tmp_path)
    with h5py.File(arguments["history"], "r") as stream:
        saved_supports = np.asarray(stream["supports"], dtype="f8")
        saved_dlvsq = np.asarray(stream["dlvsq"][-1], dtype="f8")
    arrays, metadata = load_legacy_ar1_model(**arguments)
    np.testing.assert_array_equal(arrays["supports"], saved_supports)
    np.testing.assert_array_equal(arrays["dlvsq"], saved_dlvsq)
    np.testing.assert_allclose(arrays["transports"].sum(axis=1), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.einsum("jn,jnm->jm", arrays["supports"][1:], arrays["transports"]), arrays["supports"][:-1], atol=1.0e-14)
    np.testing.assert_allclose(arrays["densities"].sum(axis=-1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose((arrays["densities"] * arrays["supports"]).sum(axis=-1), 1.0, atol=1.0e-12)
    assert metadata["measure"] == "Q"
    assert "(-3,2)" in metadata["model_label"]
    assert "whole-history" in metadata["calibration_protocol"]
    assert metadata["historical_fit_coverage"]["percent_inside_spread"] == 83.5
    assert set(arrays["ar1_parameters"]) == {"variance_intercept", "variance_slope", "scale_calibration", "fixed_factor_scales", "terminal_log_squared", "log_scale_bound", "squared_score_floor"}


def test_source_hash_mismatch_is_not_silently_accepted(tmp_path: Path) -> None:
    arguments = legacy_files(tmp_path)
    with h5py.File(arguments["history"], "r+") as stream:
        stream["dlvsq"][-1, 0, 0] = 0.1
    with pytest.raises(ValueError, match="provenance"):
        load_legacy_ar1_model(**arguments)


@pytest.mark.parametrize("value", [0.0, -0.01, np.nan])
def test_bad_saved_dlv_is_rejected_without_flooring(tmp_path: Path, value: float) -> None:
    arguments = legacy_files(tmp_path)
    with h5py.File(arguments["history"], "r+") as stream:
        stream["dlvsq"][-1, 0, 0] = value
    refresh_history_hash(arguments)
    with pytest.raises(ValueError, match="strictly positive"):
        load_legacy_ar1_model(**arguments)


def test_adapter_rejects_future_history_and_wrong_origin(tmp_path: Path) -> None:
    arguments = legacy_files(tmp_path)
    arguments["origin"] = "2025-08-28"
    with pytest.raises(ValueError, match="end exactly"):
        load_legacy_ar1_model(**arguments)


def test_transport_corruption_is_not_called_precision_recovery(tmp_path: Path) -> None:
    arguments = legacy_files(tmp_path)
    with h5py.File(arguments["history"], "r+") as stream:
        stream["transports"][0, 0, 0] += 0.01
    refresh_history_hash(arguments)
    with pytest.raises(ValueError, match="transports differ materially"):
        load_legacy_ar1_model(**arguments)


@pytest.mark.parametrize("name,value,message", [
    ("ar1_terminal_1", np.zeros(2), "last observed factor-score"),
    ("ar1_parameter_variance_slope", np.ones(2), "not stable"),
    ("fixed_factor_scales", np.zeros(2), "strictly positive"),
    ("dates", np.asarray(["2025-08-26", "2025-08-27", "2025-08-29"], dtype="S10"), "historical states"),
])
def test_incompatible_factor_state_is_rejected(tmp_path: Path, name: str, value: np.ndarray, message: str) -> None:
    arguments = legacy_files(tmp_path)
    path = Path(arguments["factor_law"])
    with np.load(path, allow_pickle=False) as stream:
        arrays = {key: stream[key] for key in stream.files}
    arrays[name] = value
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        load_legacy_ar1_model(**arguments)
