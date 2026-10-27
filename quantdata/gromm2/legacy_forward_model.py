"""Read-only adapter for the saved (-3, 2) GROMM2 AR(1) experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.geometry import transport_tower
from quantdata.gromm2.runtime import configure_backend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(name: str, values: np.ndarray, *, positive: bool = False) -> None:
    if not np.all(np.isfinite(values)) or (positive and np.any(values <= 0.0)):
        condition = "finite and strictly positive" if positive else "finite"
        raise ValueError(f"saved {name} must be {condition}; no values are repaired")


def load_legacy_ar1_model(
    *, history: Path, estimate: Path, factor_law: Path, origin: str, backend: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the archived fixed-grid affine-200 state and its joint factor law.

    The saved geometry and whole-history calibration are retained.  Only the
    archived float32 barycentric coefficients are reconstructed in float64 from
    those same supports; no probabilities, DLV levels or fitted parameters are
    projected, recentered, floored or renormalized.  This is the earlier
    whole-history protocol, not the new 252-day rolling-date calibration.

    The returned AR(1) state is the last observed log-squared factor score.
    Its forecast supplies the next day's loading amplitudes, while the Gaussian
    front-bridge innovations remain standard normal under Q.  Physical drift
    coefficients are deliberately not returned.
    """
    configure_backend(backend)
    date.fromisoformat(origin)
    history, estimate, factor_law = (Path(path).resolve() for path in (history, estimate, factor_law))
    estimate_report_path, factor_report_path = estimate.with_suffix(".json"), factor_law.with_suffix(".json")
    sources = (history, estimate, factor_law, estimate_report_path, factor_report_path)
    hashes = {str(path): _sha256(path) for path in sources}
    estimate_report = json.loads(estimate_report_path.read_text())
    factor_report = json.loads(factor_report_path.read_text())
    if estimate_report.get("schema") != "gromm2-estimate-v1" or factor_report.get("schema") != "gromm2-factor-laws-v1":
        raise ValueError("unsupported saved estimate or factor-law schema")
    if factor_report.get("base_estimate_sha256") != hashes[str(estimate)] or factor_report.get("history_sha256") != hashes[str(history)]:
        raise ValueError("saved factor-law provenance does not match the supplied history and estimate")

    with h5py.File(history, "r") as stream:
        if stream.attrs.get("schema") != "gromm2-affine-dlv-history-v1":
            raise ValueError("unsupported saved affine history schema")
        dates = np.asarray(stream["dates"]).astype("U10")
        geometry = json.loads(str(stream.attrs["geometry_config"]))
        calibration = json.loads(str(stream.attrs["calibration_config"]))
        support_precision = np.finfo(stream["supports"].dtype).eps
        transport_precision = np.finfo(stream["transports"].dtype).eps
        supports = np.asarray(stream["supports"], dtype=np.float64)
        archived_transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        business_days = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        nms_nodes = np.asarray(stream["nms_nodes"], dtype=np.float64)
        if dates.ndim != 1 or not len(dates) or np.any(dates[1:] <= dates[:-1]) or dates[-1] != origin:
            raise ValueError("saved history must be strictly chronological and end exactly at the requested origin")
        if stream["dlvsq"].shape[:1] != dates.shape:
            raise ValueError("saved DLV dates and state counts differ")
        dlvsq = np.asarray(stream["dlvsq"][-1], dtype=np.float64)
        coverage = {name: float(stream.attrs[name]) for name in (
            "percent_inside_spread", "mean_excess_half_spreads", "midpoint_mae_half_spreads",
        )}
    if tuple(geometry.get("grid_nms_range", ())) != (-3.0, 2.0):
        raise ValueError("the requested legacy model must use the saved (-3, 2) grid")
    background_volatility = float(geometry["backbone_annual_volatility"])
    if not np.isfinite(background_volatility) or background_volatility < 0.0 or calibration.get("backbone_annual_volatility") != background_volatility:
        raise ValueError("saved calibration and geometry background volatilities disagree")
    if supports.ndim != 2 or dlvsq.shape != (len(supports), supports.shape[1] - 2):
        raise ValueError("saved supports and DLV geometry disagree")
    expiry_count, node_count = supports.shape
    if expiries_y.shape != (expiry_count,) or business_days.shape != (expiry_count,) or nms_nodes.shape != (node_count - 2,):
        raise ValueError("saved expiry or strike metadata has incompatible geometry")
    for name, values in (("supports", supports), ("DLV squared", dlvsq), ("expiry years", expiries_y), ("business days", business_days)):
        _finite(name, values, positive=True)
    if np.any(np.diff(supports, axis=-1) <= 0.0) or np.any(np.diff(expiries_y) <= 0.0) or np.any(np.diff(business_days) <= 0.0):
        raise ValueError("saved supports and expiry coordinates must be strictly increasing")
    if not np.allclose(252.0 * expiries_y, business_days, rtol=1.0e-7, atol=1.0e-6):
        raise ValueError("saved business-day and year coordinates disagree")
    _finite("archived transports", archived_transports)
    if np.any(archived_transports < 0.0):
        raise ValueError("saved barycentric transports contain negative coefficients")
    transports = transport_tower(supports)
    if archived_transports.shape != transports.shape:
        raise ValueError("saved transports have incompatible geometry")
    transport_difference = float(np.max(np.abs(archived_transports - transports)))
    spacing = np.min(np.diff(supports[1:], axis=-1), axis=-1)
    magnitude = np.maximum(np.max(np.abs(supports[1:]), axis=-1), np.max(np.abs(supports[:-1]), axis=-1))
    coefficient_tolerance = 8.0 * support_precision * magnitude / spacing + 8.0 * transport_precision
    if np.any(np.abs(archived_transports - transports) > coefficient_tolerance[:, None, None]):
        raise ValueError("saved transports differ materially from the stated barycentric geometry")
    archived_mass_error = float(np.max(np.abs(np.sum(archived_transports, axis=1) - 1.0)))
    archived_mean_error = np.max(np.abs(np.einsum("jn,jnm->jm", supports[1:], archived_transports) - supports[:-1]), axis=-1)
    if archived_mass_error > 8.0 * transport_precision or np.any(archived_mean_error > 8.0 * (support_precision + transport_precision) * magnitude):
        raise ValueError("saved transports violate their source-precision mass or first-moment bounds")

    with np.load(estimate, allow_pickle=False) as values:
        loadings = np.asarray(values["log_dlvsq_loadings"], dtype=np.float64)
        scores = np.asarray(values["factor_scores"], dtype=np.float64)
        pair_origins = np.asarray(values["pair_origin_indices"], dtype=np.int64)
    if loadings.ndim != 3 or loadings.shape[:2] != dlvsq[1:].shape:
        raise ValueError("saved loading field and historical DLV geometry disagree")
    factor_count = loadings.shape[-1]
    if factor_count < 1 or scores.ndim != 2 or scores.shape[1] != factor_count or pair_origins.shape != (len(scores),):
        raise ValueError("saved factor scores and pair origins have incompatible geometry")
    if not len(pair_origins) or np.any(np.diff(pair_origins) <= 0) or pair_origins[0] < 0 or pair_origins[-1] + 1 >= len(dates):
        raise ValueError("saved factor transitions extend beyond the available causal history")
    _finite("loading field", loadings)
    _finite("factor scores", scores)
    if estimate_report.get("factor_count") != factor_count or estimate_report.get("pair_count") != len(scores):
        raise ValueError("saved estimate report and factor arrays disagree")

    with np.load(factor_law, allow_pickle=False) as values:
        if str(values["schema"]) != "gromm2-factor-laws-v1":
            raise ValueError("unsupported factor-law array schema")
        config = json.loads(str(values["config"]))
        law_dates = np.asarray(values["dates"]).astype("U10")
        law_origins = np.asarray(values["origin_indices"], dtype=np.int64)
        parameters = {name: np.asarray(values[f"ar1_parameter_{name}"], dtype=np.float64) for name in (
            "variance_intercept", "variance_slope", "scale_calibration",
        )}
        parameters["fixed_factor_scales"] = np.asarray(values["fixed_factor_scales"], dtype=np.float64)
        parameters["terminal_log_squared"] = np.asarray(values["ar1_terminal_1"], dtype=np.float64)
    if config != factor_report.get("config") or factor_report.get("factor_count") != factor_count or factor_report.get("pair_count") != len(scores):
        raise ValueError("saved factor-law configuration or report disagrees with its arrays")
    if not np.array_equal(law_origins, pair_origins) or not np.array_equal(law_dates, dates[pair_origins]):
        raise ValueError("saved factor-law transitions do not refer to the supplied historical states")
    if factor_report.get("validation_end_date") != str(law_dates[-1]) or law_dates[-1] >= origin:
        raise ValueError("saved factor-law date range is inconsistent or looks ahead of the origin")
    for name, values in parameters.items():
        if values.shape != (factor_count,):
            raise ValueError(f"saved AR(1) {name} has incompatible factor geometry")
        _finite(f"AR(1) {name}", values, positive=name in {"scale_calibration", "fixed_factor_scales"})
    if np.any(np.abs(parameters["variance_slope"]) >= 1.0):
        raise ValueError("saved AR(1) variance recursion is not stable")
    bound, floor = float(config["log_scale_bound"]), float(config["squared_score_floor"])
    if not np.isfinite(bound) or not np.isfinite(floor) or bound <= 0.0 or floor <= 0.0:
        raise ValueError("saved AR(1) volatility bounds are invalid")
    expected_terminal = np.log((scores[-1] / parameters["fixed_factor_scales"]) ** 2 + floor)
    if not np.allclose(parameters["terminal_log_squared"], expected_terminal, rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("saved AR(1) terminal state is not the last observed factor-score state")
    parameters.update(log_scale_bound=bound, squared_score_floor=floor)

    densities = np.asarray(decode_density_tower(
        jnp.asarray(supports), jnp.asarray(transports), jnp.asarray(dlvsq),
        jnp.asarray(np.diff(expiries_y, prepend=0.0)),
    ))
    _finite("decoded densities", densities)
    mass_error = float(np.max(np.abs(np.sum(densities, axis=-1) - 1.0)))
    mean_error = float(np.max(np.abs(np.sum(densities * supports, axis=-1) - 1.0)))
    if np.min(densities) < -1.0e-12 or mass_error > 1.0e-8 or mean_error > 1.0e-8:
        raise ValueError("saved model fails decoded density positivity, mass or forward checks")
    arrays = {
        "supports": supports, "transports": transports, "expiries_y": expiries_y,
        "dlvsq": dlvsq, "densities": densities, "loadings": loadings,
        "ar1_parameters": parameters, "loading_reference_days": business_days[1:],
    }
    metadata = {
        "origin": origin, "measure": "Q", "model_label": "GROMM2 legacy (-3,2) affine-200 AR(1)",
        "background_volatility": background_volatility, "historical_fit_coverage": coverage,
        "history_start": str(dates[0]), "history_end": str(dates[-1]), "history_dates": len(dates),
        "factor_transition_count": len(scores), "last_factor_origin_date": str(law_dates[-1]),
        "initialization": "Saved last observed log-squared factor scores; no physical drift under Q",
        "calibration_protocol": "Earlier joint whole-history affine calibration through the origin, not the new 252-day rolling protocol",
        "transport_precision": "Barycentric transport reconstructed in float64 from unchanged saved supports",
        "maximum_archived_transport_difference": transport_difference,
        "maximum_transport_precision_tolerance": float(np.max(coefficient_tolerance)),
        "initial_density_mass_error": mass_error, "initial_density_forward_error": mean_error,
        "source_sha256": hashes,
    }
    return arrays, metadata
