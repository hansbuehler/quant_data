"""Audit Figure 1's common denominator and separate front conditioning from DLV shocks."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.gromm import observable_surface
from quantdata.hedging.model_audit.metrics import covariance
from quantdata.hedging.model_audit.run import (
    _dlv_surface,
    _hash,
    _json,
    retained_dynamics_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.gromm2_daily import (
    load_gromm2_market_state,
)


def same_observed_changes(values: dict[str, np.ndarray]) -> np.ndarray:
    """Reject even a small change of reference coordinates, dates or values across models."""
    reference = next(iter(values.values()))
    for model, actual in values.items():
        if not np.array_equal(actual, reference):
            raise ValueError(f"Observed variance reference differs for {model}")
    return reference


def figure_identity(audit: Path) -> dict:
    selection = json.loads((audit / "MODEL_SELECTION.json").read_text())["models"]
    observations, dates = {}, None
    for model in selection:
        records = [json.loads(path.read_text()) for path in sorted((audit / "daily" / model.replace(" ", "_")).glob("*.json"))]
        current_dates = tuple((record["origin_date"], record["endpoint"]) for record in records)
        if dates is not None and dates != current_dates:
            raise ValueError("Forecast origins or endpoints differ between models")
        dates = current_dates
        observations[model] = np.array([record["actual_change"] for record in records])
    actual = same_observed_changes(observations)
    matrix = covariance(actual)
    windows = []
    for origin, _ in dates:
        dysanos = json.loads((audit / "daily/DYSANOS_ordinary_(16_factors)" / f"{origin}.json").read_text())["diagnostics"]
        gromm = json.loads((audit / "daily/GROMM-Q" / f"{origin}.json").read_text())["diagnostics"]
        manifest = json.loads(Path(gromm["fit_manifest"]).read_text())
        windows.append({"origin_date": origin, "dysanos_training_start": dysanos["training_start"],
            "gromm_training_start": manifest["training_start_date"], "dysanos_training_end": dysanos["training_end"],
            "gromm_training_end": manifest["training_end_date"]})
    return {"models": selection, "all_observed_changes_bitwise_identical": True,
        "origins": len(actual), "observable_cells": actual.shape[1], "observed_covariance_ddof": 1,
        "observed_covariance_trace": float(np.trace(matrix)),
        "business_dtes": [1, 2, 3, 4, 5, 7, 10], "nms": [-1.5, -1., -.5, 0., .5, 1., 1.5],
        "quantity": "daily changes of interpolated floating-NMS log implied volatility, not native parameters",
        "reference": "common SANOS-interpolated released market panel, smoothness 0.25; not raw quote IVs",
        "training_windows": windows,
        "identical_training_starts": sum(row["dysanos_training_start"] == row["gromm_training_start"] for row in windows)}


def _surface_summary(values: np.ndarray, *, ddof: int) -> dict:
    rows = values.reshape(len(values), -1)
    valid = np.isfinite(rows).all(axis=1)
    return {"valid_scenarios": int(valid.sum()), "scenario_count": len(values),
        "log_iv_variance_trace": float(np.trace(covariance(rows[valid], ddof=ddof))) if valid.sum() >= 2 else None}


def dysanos_channels(root: Path, audit: Path, model: str, day: date) -> dict:
    fit = load_dynamics(retained_dynamics_path(root, model, day, output=audit))
    source = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    values = json.loads((source / "config.json").read_text())
    config = DlvConfig(**{field.name: values[field.name] for field in fields(DlvConfig) if field.name in values})
    with np.load(source / "days" / f"{day}.npz", allow_pickle=False) as saved:
        state = DailyState(day=day, native=saved["raw_u"], reference_support=saved["support_strikes"], config=config)
    forecast = state.native + fit.forecast_state_change(state.native)
    spot = fit.residuals[:, :1] * fit.spot_state_loading
    factors = fit.residuals[:, 1:] @ fit.directions.T
    groups = {"full": forecast + spot + factors, "spot_projection_only": forecast + spot,
              "pca_innovations_only": forecast + factors}
    result = {}
    for name, natives in groups.items():
        surface, guards = _dlv_surface(state, natives)
        result[name] = {**_surface_summary(surface, ddof=0), "guards": guards}
    reference = json.loads((audit / "daily" / model.replace(" ", "_") / f"{day}.json").read_text())
    expected = float(np.trace(reference["surface_covariance"]))
    if not np.isclose(result["full"]["log_iv_variance_trace"], expected, rtol=1e-10, atol=1e-12):
        raise ValueError("The full DYSANOS channel replay differs from the reported forecast")
    return {"model": model, "origin_date": str(day), "channels": result, "full_forecast_reproduced": True,
        "scope": "same saved drift, decoder, guard and 252 paired empirical innovations; channels are not additive"}


def gromm_channels(root: Path, audit: Path, day: date, step) -> dict:
    directory = root / f"training/hedging/gromm-current-public-20261024-v1/artifacts/{day}"
    manifest = json.loads((directory / "manifest.json").read_text())
    paths = SimpleNamespace(affine_history=directory / "affine-200-history.h5", dynamics=directory / "rank16-ar1-dynamics.npz")
    for field, path in (("affine_history_sha256", paths.affine_history), ("dynamics_sha256", paths.dynamics)):
        if _hash(path) != manifest[field]:
            raise ValueError(f"GROMM source checksum differs: {field}")
    names = ("percent_inside_spread", "explained_variance_ratio", "physical_projection_fraction",
             "candidate_physical_projection_fraction", "physical_fallback_used", "physical_selected_model", "physical_accepted")
    market = load_gromm2_market_state(SimpleNamespace(paths=paths, as_of_date=day, **{name: manifest[name] for name in names}))
    random = np.random.default_rng(20_261_111)
    half = random.standard_normal((8, 256, market.dynamics.factor_count))
    normals = np.stack((half, -half), axis=2).reshape(8, 512, -1)
    background_half = random.standard_normal(256)
    background = np.stack((background_half, -background_half), axis=1).reshape(512)
    state = SimulationState(front_density=jnp.broadcast_to(jnp.asarray(market.density_tower[0]), (512, market.supports.shape[1])),
        dlvsq=jnp.broadcast_to(jnp.asarray(market.dlvsq), (512, *market.dlvsq.shape)), background_level=jnp.ones(512))
    loadings = market.dynamics.log_dlvsq_loadings
    first = np.zeros_like(loadings)
    first[..., 0] = loadings[..., 0]
    residual = loadings.copy()
    residual[..., 0] = 0.
    groups = {"full": loadings, "front_bridge_only": np.zeros_like(loadings),
              "front_plus_shared_spot_dlv_loading": first, "front_plus_residual_dlv_loadings": residual}
    channels = {}
    cancellation = None
    for name, selected in groups.items():
        result = step(jnp.asarray(market.supports), jnp.asarray(market.transports),
            jnp.asarray(np.diff(market.expiries_y, prepend=0.)[1:]), state,
            jnp.asarray(selected), jnp.asarray(normals), jnp.asarray(background), remaining_front_days=jnp.asarray(1), substeps=8)
        spot, densities = np.asarray(result.spot), np.asarray(result.density_tower)
        surface, _ = observable_surface(market, densities, np.asarray(result.state.background_level), spot, elapsed=1)
        channels[name] = {**_surface_summary(surface, ddof=1), "drift_taming_fraction": float(result.drift_taming_fraction)}
        if name == "full":
            without_background, _ = observable_surface(market, densities, np.ones(len(spot)), np.asarray(result.latent_spot), elapsed=1)
            cancellation = float(np.max(np.abs(surface - without_background)))
        print(f"GROMM variance channel {day}: {name} trace={channels[name]['log_iv_variance_trace']:.6g}", flush=True)
    reference = json.loads((audit / "daily/GROMM-Q" / f"{day}.json").read_text())
    expected = float(np.trace(reference["surface_covariance"]))
    if not np.isclose(channels["full"]["log_iv_variance_trace"], expected, rtol=1e-7, atol=1e-10):
        raise ValueError("The full GROMM replay differs from the reported forecast")
    return {"model": "GROMM-Q", "origin_date": str(day), "channels": channels,
        "background_brownian_cancellation_max_log_iv_error": cancellation,
        "full_forecast_reproduced": True, "scope": "same initial surface, 512 antithetic draws and eight steps; each loading ablation recomputes Q drift, channels are not additive"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    _json(arguments.output / "figure_identity.json", figure_identity(arguments.audit))
    jax.config.update("jax_enable_x64", True)
    step = jax.jit(one_day_risk_neutral_step, static_argnames=("substeps",))
    started = time.monotonic()
    for day in (date(2024, 4, 1), date(2024, 4, 30), date(2024, 5, 31)):
        models = ("GROMM-Q", "DYSANOS ordinary (16 factors)", "DYSANOS spot-orthogonal (16 factors)")
        for model in models:
            path = arguments.output / model.replace(" ", "_") / f"{day}.json"
            if path.exists():
                continue
            result = (gromm_channels(arguments.root, arguments.audit, day, step) if model == "GROMM-Q"
                      else dysanos_channels(arguments.root, arguments.audit, model, day))
            _json(path, result)
            print(f"Variance channels complete {model} {day} elapsed={time.monotonic()-started:.1f}s", flush=True)
    identity = json.loads((arguments.output / "figure_identity.json").read_text())
    summaries = json.loads((arguments.audit / "summary.json").read_text())
    lines = ["# GROMM versus DYSANOS: observable variance audit", "",
        "## Figure 1", "",
        f"All {len(identity['models'])} models have bitwise-identical observed changes on the same {identity['origins']} origins "
        f"and {identity['observable_cells']} floating-NMS log-IV coordinates. The common observed covariance trace is "
        f"{identity['observed_covariance_trace']:.12g}, using denominator n−1. These are changes in SANOS-interpolated market IVs "
        "(reference smoothness 0.25), not DLV-parameter changes or raw quote IVs.", "",
        "The conditional bar is mean_t trace(Cov_model[ΔlogIV | t]) / trace(Cov_data[ΔlogIV]). "
        "The pooled bar additionally includes the across-origin covariance of forecast means. "
        "The conditional denominator is unconditional observed variance; the ratio need not equal one.", "",
        "| Model | Conditional variance / observed | Pooled variance / observed |", "|---|---:|---:|"]
    for model in ("DYSANOS ordinary (16 factors)", "DYSANOS spot-orthogonal (16 factors)", "GROMM-P", "GROMM-Q"):
        result = summaries[model]
        lines.append(f"| {model} | {result['conditional_covariance']['trace_ratio']:.2%} | {result['pooled_covariance']['trace_ratio']:.2%} |")
    lines.extend(["", "## Controlled front-conditioning check", "",
        "Same saved GROMM-Q initial surface, 512 antithetic draws, seed and eight substeps. "
        "Switching off DLV diffusion recomputes the consistent Q drift, leaving the front information bridge active. "
        "Counterfactual channel variances are not additive contributions.", "",
        "| Origin | Full GROMM log-IV trace | Front bridge only | Front-only / full |", "|---|---:|---:|---:|"])
    cancellation = []
    for path in sorted((arguments.output / "GROMM-Q").glob("*.json")):
        result = json.loads(path.read_text())
        channels = result["channels"]
        full, front = channels["full"]["log_iv_variance_trace"], channels["front_bridge_only"]["log_iv_variance_trace"]
        lines.append(f"| {result['origin_date']} | {full:.6f} | {front:.6f} | {front/full:.2%} |")
        cancellation.append(result["background_brownian_cancellation_max_log_iv_error"])
    lines.extend(["", "## Interpretation and limits", "",
        "GROMM conditions a fixed-expiry density tower on the evolving/settling front density. "
        "That creates floating-IV variability even with future DLV diffusion set to zero. "
        "The retained DYSANOS adapter instead moves a floating native state using estimated PCA/AR innovations and, "
        "for its orthogonal variant, a spot-state loading. At fixed native state its normalized IV surface does not move with spot. "
        "The orthogonal variant does restore learned spot-related changes; it is not a spot-independent model.", "",
        "The remaining gap cannot be assigned solely to the front bridge. The two estimators do not share a calibrated diffusion: "
        "GROMM aligns fixed-contract innovations and smooths them before factor estimation, while the retained DYSANOS fit "
        "projects the floating native state with substantial floor/undefined-coordinate handling. Their native coverage percentages "
        "have different denominators. Prior same-shock diagnostics show large decoded-IV variance loss in DYSANOS projection/AR fitting.", "",
        f"Training calendars are not matched across models: only {identity['identical_training_starts']}/{identity['origins']} start dates agree, "
        "although both use 252 eligible transitions. The 0.1-SANOS DYSANOS inputs also differ from the 0.25 common market reference. "
        "Consequently this is a common out-of-sample observable diagnostic, not an architecture-only experiment.", "",
        f"Removing the common multiplicative Black-background level from both spot and pricing changes log IV by at most "
        f"{max(cancellation):.3g} across these checks. Its independent Brownian shock therefore does not explain the floating-IV variance gap. "
        "The static 5% smoothing kernel remains present and is a separate geometry difference.", "",
        "All full-forecast channel replays reproduce the published conditional variance. This diagnoses the retained implementations, "
        "not an inherent limitation of the DYSANOS model class. The three-origin ablations do not establish a full-window variance attribution."])
    (arguments.output / "FINDINGS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
