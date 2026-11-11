"""Audit retained April--May fits without retraining or modifying checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import fields, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quantdata.biz_calendar import NMS
from quantdata.hedge2.batch_pricing import decode_batch
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.implied import (
    implied_volatilities_with_validity,
    summarize_inversions,
)
from quantdata.hedging.model_audit.inference import checkpoint_payload
from quantdata.hedging.model_audit.metrics import (
    autocorrelation,
    correlation_from_covariance,
    covariance,
    covariance_comparison,
    factor_innovation_covariance,
    frozen_pca,
    linear_stability,
    mixture_covariance,
    quote_scores,
    tail_scores,
    var_es_score,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
    SanosFixedGridHistory,
    fixed_grid_sanos_day,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import VolganPcaModelConfig
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    volgan_live_condition_from_history,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.hedger import sample_volgan_transitions
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.model import VolganGenerator
from quantdata.torch import BaseEnvironment, torch

COMMON_NMS = np.array([-1.5, -1., -.5, 0., .5, 1., 1.5])
GRID = GenerativeSurfaceGrid.short_end_nms()
COMMON_INDICES = np.array([GRID.nms_nodes.index(value) for value in COMMON_NMS])
QUANTILES = (.01, .05, .95, .99)
LAGS = (1, 2, 5, 10, 21, 63)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return (payload if isinstance(payload, list) else payload["results"])[0]


def build_history(panel: PublicHedgingPanel, output: Path) -> SanosFixedGridHistory:
    """Cache the fixed observable geometry, not any model's native PCA basis."""
    cache = output / "reference_history.npz"
    stop = panel.dates.index(panel.next_one_session_date(date(2024, 5, 31))) + 1
    start = max(0, panel.dates.index(date(2024, 4, 1)) - 300)
    selected_dates = panel.dates[start:stop]
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["manifest_sha256"]) != panel.dataset_manifest_sha256:
                raise ValueError("Reference cache belongs to a different data manifest")
            surfaces, validity = data["surfaces"], data["validity"]
    else:
        surfaces, validity = [], []
        started = time.monotonic()
        for index, day in enumerate(selected_dates):
            surface = fixed_grid_sanos_day(panel.market_day(day), grid=GRID)
            surfaces.append(surface.log_implied_volatilities)
            validity.append(surface.implied_vol_is_interior)
            if index % 25 == 0:
                print(f"reference surfaces {index + 1}/{len(selected_dates)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        surfaces, validity = np.asarray(surfaces), np.asarray(validity)
        np.savez_compressed(cache, surfaces=surfaces, validity=validity,
                            dates=np.array([str(day) for day in selected_dates]),
                            manifest_sha256=panel.dataset_manifest_sha256)
    observations = panel.observations.to_pandas().iloc[start:stop]
    returns = np.log1p(observations["forward_adjusted_spot_return_into_date"].to_numpy(dtype=float))
    sessions = observations["business_sessions_from_previous_observation"].to_numpy(dtype=int)
    returns[0], sessions[0] = np.nan, 0
    return SanosFixedGridHistory(dates=selected_dates, log_implied_volatilities=surfaces,
        implied_vol_is_interior=validity, forward_log_returns_into_date=returns,
        sessions_from_previous_observation=sessions)


def _volgan(root: Path, panel: PublicHedgingPanel, history: SanosFixedGridHistory,
            index: int, inputs: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    day = history.dates[index]
    path = root / f"experiments/hedging/volgan-pca8-rolling-aprmay-20261024-v1/run/records/{day}.json"
    diagnostics = _record(path)["diagnostics"]
    checkpoint_path = Path(diagnostics["fit_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = VolganPcaModelConfig(label="volgan-short-end-nms35-pca8-2022", grid=GRID)
    if checkpoint["model_code_version"] != config.version.unique_id48 or checkpoint["model_user_version"] != config.user_version:
        raise ValueError("VolGAN checkpoint code/user version mismatch; audit will not retrain")
    if checkpoint["training_end_date"] != str(day) or _hash(checkpoint_path) != diagnostics["fit_checkpoint_sha256"]:
        raise ValueError("VolGAN dated checkpoint identity mismatch")
    state = checkpoint["generator"]
    names = ("condition_mean", "condition_scale", "target_mean", "target_scale", "level_mean",
             "level_components", "increment_mean", "increment_components")
    generator = VolganGenerator(BaseEnvironment(precision=64, device="cpu"), config=config,
                               **{name: state[name] for name in names})
    generator.load_state_dict(state)
    generator.eval()
    window = SanosFixedGridHistory(dates=history.dates[:index + 1],
        log_implied_volatilities=history.log_implied_volatilities[:index + 1],
        implied_vol_is_interior=history.implied_vol_is_interior[:index + 1],
        forward_log_returns_into_date=history.forward_log_returns_into_date[:index + 1],
        sessions_from_previous_observation=history.sessions_from_previous_observation[:index + 1])
    condition = volgan_live_condition_from_history(window, as_of_date=day, forecast_date=panel.next_one_session_date(day))
    sample = sample_volgan_transitions(SimpleNamespace(generator=generator), condition=condition.values,
                                      path_count=10_000, seed=20_261_111)
    inputs.append({"path": str(checkpoint_path), "sha256": diagnostics["fit_checkpoint_sha256"]})
    return sample.forward_log_returns, sample.next_log_implied_volatilities, {
        "checkpoint": str(checkpoint_path), "epoch": int(checkpoint["epoch"]),
        "training_start": checkpoint["training_start_date"], "training_end": checkpoint["training_end_date"],
        "sampling": "new CPU draws from saved checkpoint; no retraining", "seed": 20_261_111}


def _saved(root: Path, model: str, day: date, inputs: list[dict], *, audit_directory: Path | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    regenerated = audit_directory / f"regenerated/{model}/{day}.npz" if audit_directory is not None else None
    if regenerated is not None and regenerated.exists():
        with np.load(regenerated, allow_pickle=False) as data:
            returns, surfaces = data["forward_log_returns"], data["next_log_implied_volatilities"]
            metadata = json.loads(str(data["metadata_json"]))
        checkpoint_payload(Path(metadata["checkpoint"]), expected_sha256=metadata["checkpoint_sha256"], audit_directory=audit_directory)
        inputs.append({"path": str(regenerated), "sha256": _hash(regenerated)})
        return returns, surfaces, {**metadata, "scenario_path": str(regenerated),
            "sampling": metadata.get("sampling", "new CPU draws from saved EMA; 512 paths")}
    if model == "FuNVol":
        path = root / f"training/hedging/funvol-causal-chain-20261025-warning-v1/aprmay/scenarios/{day}.npz"
    else:
        variant = "base" if model == "AD-Seq-Vol" else "ft"
        if day == date(2024, 4, 1):
            record_path = root / f"experiments/hedging/short-end-regeneration-20261024/diffusion-{variant}-rolling-fixed-v2/{day}/records/{day}.json"
        else:
            record_path = root / f"experiments/hedging/diffusion-{variant}-gpu-20261027-v1/{day}/records/{day}.json"
        diagnostics = _record(record_path)["diagnostics"]
        if "scenario_artifact_path" not in diagnostics:
            if audit_directory is None:
                raise FileNotFoundError(f"No retained next-IV surface artifact recorded in {record_path}; checkpoint alone is not a forecast")
            path = audit_directory / f"regenerated/{model}/{day}.npz"
        else:
            path = Path(diagnostics["scenario_artifact_path"])
            if _hash(path) != diagnostics["scenario_artifact_sha256"]:
                raise ValueError("Saved diffusion scenario hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        returns = data["forward_log_returns"]
        surfaces = np.log(data["next_implied_volatilities"]) if model == "FuNVol" else data["next_log_implied_volatilities"]
    inputs.append({"path": str(path), "sha256": _hash(path)})
    regenerated = audit_directory is not None and path.is_relative_to(audit_directory)
    return returns, surfaces, {"scenario_path": str(path), "sampling": "new CPU draws from saved EMA; 512 paths" if regenerated else "retained, unmodified one-day scenarios"}


def _dlv_surface(state: DailyState, natives: np.ndarray) -> tuple[np.ndarray, dict]:
    """Price decoded surfaces on floating observable NMS coordinates."""
    decoded = decode_batch(state, natives, max_business_dte=10.)
    inversion_records = []
    times = np.asarray(GRID.expiries_y)
    days = times * 252.
    atm_prices = decoded.europeans(business_dtes=days, strikes=np.ones(len(days)))
    atm_vol, atm_valid = implied_volatilities_with_validity(expiries_y=np.broadcast_to(times, atm_prices.shape).ravel(),
        normalized_strikes=np.ones(atm_prices.size), normalized_prices=atm_prices.ravel(), is_call=np.ones(atm_prices.size, dtype=bool), diagnostics=inversion_records)
    atm_vol = atm_vol.reshape(atm_prices.shape)
    eligible = atm_valid.reshape(atm_prices.shape).all(axis=1)
    surfaces = np.full((len(natives), len(times), len(COMMON_NMS)), np.nan)
    valid = np.zeros(surfaces.shape, dtype=bool)
    if eligible.any():
        selected = (decoded if eligible.all() else replace(decoded, support=decoded.support[eligible],
            densities=decoded.densities[eligible], lower_clips=decoded.lower_clips[eligible], upper_clips=decoded.upper_clips[eligible]))
        strikes = NMS.from_nms(COMMON_NMS[None, None, :], times[None, :, None], vol=atm_vol[eligible, :, None])
        prices = selected.europeans(business_dtes=np.repeat(days, len(COMMON_NMS)),
            strikes=strikes.reshape(eligible.sum(), -1), is_call=(strikes >= 1.).reshape(eligible.sum(), -1))
        vol, identified = implied_volatilities_with_validity(expiries_y=np.broadcast_to(times[None, :, None], strikes.shape).ravel(),
            normalized_strikes=strikes.ravel(), normalized_prices=prices.ravel(), is_call=(strikes >= 1.).ravel(), diagnostics=inversion_records)
        vol, identified = vol.reshape(strikes.shape), identified.reshape(strikes.shape)
        values = np.full_like(vol, np.nan)
        np.log(vol, out=values, where=identified & (vol > 0))
        surfaces[eligible], valid[eligible] = values, identified
    return surfaces, {"lower_clips": int(decoded.lower_clips.sum()), "upper_clips": int(decoded.upper_clips.sum()),
                      "invalid_node_rate": float(np.mean(~valid)), "invalid_path_rate": float(np.mean(~valid.all(axis=(1, 2)))),
                      "atm_invalid_scenario_count": int(np.sum(~eligible)),
                      "iv_inversion": summarize_inversions(inversion_records)}


def _backbone_stability(dynamics: FittedDynamics, *, joint_covariance: np.ndarray | None = None) -> dict:
    """Assess only the floating-volatility PCA AR backbone, not the spot level."""
    projected_spot = dynamics.directions.T @ dynamics.spot_state_loading
    innovations = factor_innovation_covariance(
        dynamics.joint_innovation_covariance if joint_covariance is None else joint_covariance, projected_spot)
    stability = linear_stability(dynamics.ar_matrix, innovations)
    stability["scope"] = "floating-volatility PCA AR(1) backbone"
    stability["innovation_covariance"] = innovations.tolist()
    stability["stationary_mean"] = (np.linalg.solve(np.eye(len(dynamics.ar_intercept))
        - dynamics.ar_matrix, dynamics.ar_intercept).tolist() if stability["stable_linear_core"] else None)
    return stability


def retained_dynamics_path(root: Path, model: str, day: date, *, output: Path | None = None) -> Path:
    """Resolve a dated fit without substituting eight factors for a requested sixteen."""
    variant = "orthogonal" if "spot-orthogonal" in model else "ordinary"
    if "(16 factors)" in model:
        if output is None:
            raise ValueError("Sixteen-factor fits require their dedicated comparison directory")
        return output / "fits" / variant / "days" / f"{day}.npz"
    return root / f"quant_data/hedge2/runs/rolling-2022-2025-{variant}-v2-20261106/days/{day}.npz"


def _dysanos(root: Path, model: str, day: date, inputs: list[dict],
             *, output: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    base = root / "quant_data/hedge2/runs"
    source = base / "dlv-full-v4-20261106T1130"
    dynamics_path = retained_dynamics_path(root, model, day, output=output)
    dynamics = load_dynamics(dynamics_path)
    if "(16 factors)" in model and dynamics.directions.shape[1] != 16:
        raise ValueError("The requested sixteen-factor checkpoint has a different rank")
    config_values = json.loads((source / "config.json").read_text())
    config = DlvConfig(**{field.name: config_values[field.name] for field in fields(DlvConfig) if field.name in config_values})
    state_path = source / f"days/{day}.npz"
    with np.load(state_path, allow_pickle=False) as data:
        state = DailyState(day=day, native=data["raw_u"], reference_support=data["support_strikes"], config=config)
        provenance = {name: data[name].item() for name in ("source_smoothness", "representation_smoothness", "floored_nodes", "undefined_nodes", "capped_nodes", "max_repricing_error")}
    current, _ = _dlv_surface(state, state.native[None])
    forecast = state.native + dynamics.forecast_state_change(state.native)
    residuals = dynamics.residuals
    if model.endswith("(tail)"):
        from quantdata.hedging.model_audit.tail import TailSampler, fit_tail

        if output is None:
            raise ValueError("Tail Diffusion requires a dedicated output directory")
        tail_record = fit_tail(dynamics, root=root, output=output, model=model)
        residuals = TailSampler(dynamics, tail_record, paths=10_000, seed=20_261_111).draw()
        provenance["tail_diffusion"] = tail_record
    natives = forecast + residuals[:, :1] * dynamics.spot_state_loading + residuals[:, 1:] @ dynamics.directions.T
    surfaces, guards = _dlv_surface(state, natives)
    stability = _backbone_stability(dynamics)
    if model.endswith("(tail)"):
        joint_covariance = dynamics.joint_innovation_covariance.copy()
        conditional = np.asarray(tail_record["conditional_factor"])
        scale = tail_record["fit"]["config"]["unconditional_scale"]
        joint_covariance[1:, 1:] += (scale - 1.) * (conditional @ conditional.T)
        stability = _backbone_stability(dynamics, joint_covariance=joint_covariance)
    provenance.update(guards)
    provenance.update({"stability": stability, "raw_spectral_radius": dynamics.unconstrained_spectral_radius,
        "native_pca_coverage": float(dynamics.explained_variance_fraction.sum()),
        "training_start": str(dynamics.training_start), "training_end": str(dynamics.training_end),
        "sampling": "exact enumeration of 252 paired empirical innovations; original historical spot law",
        "note": "legacy 9x200 DLV fit to 0.1 SANOS; not the unimplemented corrected 13-expiry model"})
    if model.endswith("(tail)"):
        provenance["sampling"] = "10000 causal conditional Tail Diffusion draws; normal spot, OU-scaled conditional surface innovations"
    for path in (dynamics_path, state_path):
        inputs.append({"path": str(path), "sha256": _hash(path)})
    return np.log1p(residuals[:, 0]), surfaces, current[0], provenance


def _quote_reconstruction(root: Path, model: str, day: date) -> dict:
    if not model.startswith("DYSANOS"):
        return {"status": "not_applicable", "reason": "Conditional generator has no separately fitted same-day observable surface decoder in this audit"}
    directory = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    market = root / f"experiments/hedge2/market-full-20261106T0001/market/days/{day}.parquet"
    quotes = pd.read_parquet(market)
    quotes = quotes.loc[(quotes.bdte >= 1.) & (quotes.bdte <= 10.) & (quotes.nms.abs() <= 1.5)].copy()
    from quantdata.hedge2.pricing import DlvPricer

    with np.load(directory / f"days/{day}.npz", allow_pickle=False) as data:
        pricer = DlvPricer(support=data["support_strikes"], expiries_y=data["expiries_y"], dlvsqs=data["training_dlvsqs"])
    prices = np.empty(len(quotes))
    for tenor in quotes.bdte.unique():
        selected = (quotes.bdte == tenor).to_numpy()
        block = quotes.loc[selected]
        prices[selected] = block.df * block.fwd * pricer.european_at(expiry_y=tenor / 252.,
            strikes=(block.strike / block.fwd).to_numpy(), is_call=(block.option_type == "call").to_numpy())
    result = quote_scores(prices, quotes.observed_bid, quotes.observed_ask)
    result["status"] = "scored"
    result["mode"] = "same-day reconstruction against observed licensed quotes"
    result["source_fit_scores"] = quote_scores(quotes.fit_price, quotes.observed_bid, quotes.observed_ask)
    return result


def _summarize(records: list[dict], basis: np.ndarray) -> dict:
    surface_records = [item for item in records if item.get("status") == "scored"]
    if len(surface_records) < 2:
        return {"surface_days": len(surface_records)}
    means = np.asarray([item["surface_mean"] for item in surface_records])
    covariances = np.asarray([item["surface_covariance"] for item in surface_records])
    actual = np.asarray([item["actual_change"] for item in surface_records])
    average = covariances.mean(axis=0)
    mixture = mixture_covariance(means, covariances)
    data_covariance = covariance(actual)
    summary = {"surface_days": len(surface_records), "conditional_covariance": covariance_comparison(average, data_covariance),
        "pooled_covariance": covariance_comparison(mixture, data_covariance),
        "projected_covariance": covariance_comparison(basis.T @ mixture @ basis, basis.T @ data_covariance @ basis),
        "forecast_rmse": float(np.sqrt(np.mean((actual - means)**2))),
        "persistence_rmse": float(np.sqrt(np.mean(actual**2))),
        "invalid_surface_rate": float(np.mean([item["invalid_surface_rate"] for item in surface_records])),
        "extreme_surface_rate": float(np.mean([item["extreme_surface_rate"] for item in surface_records]))}
    joint_cov = mixture_covariance(np.array([item["joint_mean"] for item in surface_records]),
                                   np.array([item["joint_covariance"] for item in surface_records]))
    model_corr = correlation_from_covariance(joint_cov)[0, 1:]
    data_joint = np.column_stack(([item["actual_spot"] for item in surface_records], actual @ basis))
    data_corr = correlation_from_covariance(covariance(data_joint))[0, 1:]
    summary["spot_pc_correlations_model"] = model_corr.tolist()
    summary["spot_pc_correlations_data"] = data_corr.tolist()
    summary["spot_pc_corr_err_j"] = np.abs(model_corr - data_corr).tolist()
    average_joint = np.mean([item["joint_covariance"] for item in surface_records], axis=0)
    conditional_corr = correlation_from_covariance(average_joint)[0, 1:]
    residuals = data_joint - np.array([item["joint_mean"] for item in surface_records])
    residual_corr = correlation_from_covariance(covariance(residuals))[0, 1:]
    summary["spot_pc_correlations_conditional_model"] = conditional_corr.tolist()
    summary["spot_pc_correlations_forecast_residuals"] = residual_corr.tolist()
    summary["spot_pc_conditional_corr_err_j"] = np.abs(conditional_corr - residual_corr).tolist()
    summary["tails"] = {}
    for column, label in enumerate(("PC1", "PC2", "PC3", "spot_log_return")):
        values = np.array([item["actual_targets"][column] for item in surface_records])
        summary["tails"][label] = {}
        for quantile_index, quantile in enumerate(QUANTILES):
            forecasts = [item["quantiles"][quantile_index][column] for item in surface_records]
            summary["tails"][label][str(quantile)] = tail_scores(values, forecasts, probability=quantile)
        summary["tails"][label]["ES"] = var_es_score(values,
            [item["quantiles"][1][column] for item in surface_records],
            [item["es_lower"][column] for item in surface_records])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=["DYSANOS ordinary", "DYSANOS spot-orthogonal", "FuNVol", "VolGAN PCA8", "AD-Seq-Vol", "AD-Seq-Vol-FT"])
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    root, output = arguments.root, arguments.output
    panel = PublicHedgingPanel(root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, output)
    common = history.log_implied_volatilities[:, :, COMMON_INDICES].reshape(len(history.dates), -1)
    eligible = [index for index in range(1, len(history.dates)) if history.dates[index] < date(2024, 4, 1)
                and history.sessions_from_previous_observation[index] == 1][-252:]
    if len(eligible) != 252:
        raise ValueError("The common pre-April PCA needs 252 exact training transitions")
    pca = frozen_pca(common[eligible] - common[np.array(eligible) - 1])
    basis = pca["basis"]
    np.savez_compressed(output / "frozen_observable_pca.npz", **pca, nms_nodes=COMMON_NMS,
        expiries_y=np.array(GRID.expiries_y), training_endpoints=np.array([str(history.dates[index]) for index in eligible]))
    origins = [index for index, day in enumerate(history.dates) if date(2024, 4, 1) <= day <= date(2024, 5, 31)]
    sessions = np.cumsum(history.sessions_from_previous_observation)
    levels = common[origins] @ basis
    observable = {"dates": [str(history.dates[index]) for index in origins],
        "training_first_endpoint": str(history.dates[eligible[0]]), "training_last_endpoint": str(history.dates[eligible[-1]]),
        "pca_training_shares": pca["shares"].tolist(), "pca_basis_policy": "frozen before April; model-native factors never concatenated",
        "nms_nodes": COMMON_NMS.tolist(), "business_dtes": (np.array(GRID.expiries_y) * 252.).tolist(),
        "reference_manifest_sha256": panel.dataset_manifest_sha256,
        "pc_level_acf": autocorrelation(levels, sessions[origins], lags=LAGS),
        "pc_change_acf": autocorrelation((common[np.array(origins) + 1] - common[origins]) @ basis, sessions[origins], lags=LAGS),
        "atm_log_iv_level_acf": autocorrelation(history.log_implied_volatilities[origins, :, GRID.nms_nodes.index(0.)], sessions[origins], lags=LAGS)}
    _json(output / "observable_reference.json", observable)
    all_summaries = json.loads((output / "summary.json").read_text()) if (output / "summary.json").exists() else {}
    inputs = []
    started = time.monotonic()
    for model in arguments.models:
        records = []
        for position, index in enumerate(origins):
            day = history.dates[index]
            path = output / "daily" / model.replace(" ", "_") / f"{day}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if (model.startswith("DYSANOS") and previous.get("status") == "scored"
                        and previous["diagnostics"]["stability"].get("scope") != "floating-volatility PCA AR(1) backbone"):
                    variant = "orthogonal" if "spot-orthogonal" in model else "ordinary"
                    fit_path = root / f"quant_data/hedge2/runs/rolling-2022-2025-{variant}-v2-20261106/days/{day}.npz"
                    previous["diagnostics"]["stability"] = _backbone_stability(load_dynamics(fit_path))
                    _json(path, previous)
                if model.startswith("DYSANOS") and previous.get("status") == "scored" and "scenario_covariance_ddof" not in previous:
                    correction = (previous["valid_scenario_count"] - 1) / previous["valid_scenario_count"]
                    for name in ("surface_covariance", "joint_covariance"):
                        previous[name] = (np.asarray(previous[name]) * correction).tolist()
                    previous["scenario_covariance_ddof"] = 0
                    _json(path, previous)
                recovered = output / f"regenerated/{model}/{day}.npz"
                if previous.get("status") == "scored" or not recovered.exists():
                    records.append(previous)
                    continue
            try:
                if history.sessions_from_previous_observation[index + 1] != 1:
                    raise ValueError("The endpoint is not exactly one session later")
                if model.startswith("DYSANOS"):
                    returns, next_common, current, diagnostics = _dysanos(root, model, day, inputs, output=output)
                else:
                    if model == "VolGAN PCA8":
                        returns, surfaces, diagnostics = _volgan(root, panel, history, index, inputs)
                    else:
                        returns, surfaces, diagnostics = _saved(root, model, day, inputs, audit_directory=output)
                    next_common = surfaces[:, :, COMMON_INDICES]
                    current = history.log_implied_volatilities[index][:, COMMON_INDICES]
                next_common = next_common.reshape(len(returns), -1)
                current = current.reshape(-1)
                valid = np.isfinite(next_common).all(axis=1) & np.isfinite(returns)
                invalid_rate = float(np.mean(~valid))
                if valid.sum() < 4:
                    raise ValueError("Fewer than four valid observable forecast scenarios")
                log_returns, next_common = returns[valid], next_common[valid]
                changes = next_common - current
                targets = np.column_stack((changes @ basis, log_returns))
                actual = common[index + 1] - common[index]
                actual_spot = float(history.forward_log_returns_into_date[index + 1])
                joint = np.column_stack((log_returns, changes @ basis))
                covariance_ddof = 0 if model.startswith("DYSANOS") and not model.endswith("(tail)") else 1
                thresholds = np.quantile(targets, QUANTILES, axis=0)
                shortfalls = [float(targets[targets[:, column] <= thresholds[1, column], column].mean()) for column in range(4)]
                record = {"model": model, "origin_date": str(day), "endpoint": str(history.dates[index + 1]),
                    "status": "scored", "scenario_count": int(len(returns)), "valid_scenario_count": int(valid.sum()),
                    "surface_mean": changes.mean(axis=0).tolist(), "surface_covariance": covariance(changes, ddof=covariance_ddof).tolist(),
                    "joint_mean": joint.mean(axis=0).tolist(), "joint_covariance": covariance(joint, ddof=covariance_ddof).tolist(),
                    "scenario_covariance_ddof": covariance_ddof,
                    "actual_change": actual.tolist(), "actual_spot": actual_spot,
                    "actual_targets": np.r_[actual @ basis, actual_spot].tolist(),
                    "quantiles": thresholds.tolist(), "es_lower": shortfalls,
                    "invalid_surface_rate": invalid_rate, "extreme_surface_rate": float(np.mean(np.any(next_common > np.log(3.), axis=1))),
                    "forecast_rmse": float(np.sqrt(np.mean((changes.mean(axis=0) - actual)**2))),
                    "diagnostics": diagnostics, "bidask": _quote_reconstruction(root, model, day)}
            except FileNotFoundError as error:
                record = {"model": model, "origin_date": str(day), "status": "unavailable", "reason": str(error)}
            except (ValueError, RuntimeError, FloatingPointError) as error:
                record = {"model": model, "origin_date": str(day), "status": "error", "reason": f"{type(error).__name__}: {error}"}
            _json(path, record)
            records.append(record)
            elapsed = time.monotonic() - started
            print(f"{model}: {position+1}/{len(origins)} {day} {record['status']} elapsed={elapsed:.1f}s", flush=True)
        all_summaries[model] = _summarize(records, basis)
        _json(output / "summary.json", all_summaries)
    _json(output / "inputs.json", inputs)
    print(f"audit complete elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
