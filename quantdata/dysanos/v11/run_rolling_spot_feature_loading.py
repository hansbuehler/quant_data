"""Rolling direct DYSANOS spot-loading calibration for decoded DLV/backbone moves.

The adjustment is made in the 200-dimensional affine DYSANOS state, before
the usual bounded native-coordinate decoder.  It therefore cannot alter
prices directly or create an inadmissible surface.  For each 2025 origin a
252-session window identifies the state direction associated with the
standardised front-SANOS spot score and then regularises that direction to
match selected decoded central-DLV and backbone-total-volatility covariances.

This is an experimental forecasting overlay, not a replacement for the
published dynamics.  The decoder is frozen at its pre-2025 fit; only rolling
transition statistics use observations available at each origin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from quantdata.dysanos.flex.model import JointDirectDynamicsModel, bound_native_raw
from quantdata.dysanos.flex.presentation_hedging import (
    _checkpoint_config,
    load_presentation_model,
)
from quantdata.dysanos.v11.atm_spot_correlation import _decode_atm_volatilities
from quantdata.dysanos.v11.daily_joint_ar_spot import (
    DailyJointArSpotFit,
    fit_daily_joint_ar_spot,
    sample_daily_joint_ar_spot,
)
from quantdata.dysanos.v11.daily_joint_dynamics import (
    _historical_spot_scores,
    _terminal_front_gross_returns,
)
from quantdata.torch import BaseEnvironment


def _selected_log_features(model: object, state: torch.Tensor) -> torch.Tensor:
    """Return central log-DLV-squared and log-backbone-total-vol features."""
    raw = model.raw_states_from_states(state)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    central = parameters.dlvsqs[..., model.config.geometry.num_inner_strikes // 2]
    increments = parameters.forward_vols * parameters.eta
    total = torch.cumsum(
        increments * torch.sqrt(model.sanos_decoder.model_dts_y), dim=-1
    )
    return torch.cat((torch.log(central), torch.log(total)), dim=-1)


def selected_log_features(model: object, states: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Decode selected features in bounded batches without retaining graphs."""
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for first in range(0, len(states), batch_size):
            batch = torch.as_tensor(states[first : first + batch_size], dtype=model.fdtype)
            values.append(_selected_log_features(model, batch).cpu().double().numpy())
    return np.concatenate(values)


def feature_jacobian(model: object, state: np.ndarray) -> np.ndarray:
    """Jacobian of decoded selected features with respect to affine state."""
    point = torch.as_tensor(state, dtype=model.fdtype).detach().requires_grad_(True)

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        return _selected_log_features(model, value.unsqueeze(0)).squeeze(0)

    return (
        torch.autograd.functional.jacobian(evaluate, point, vectorize=True)
        .detach()
        .cpu()
        .double()
        .numpy()
    )


def calibrated_spot_loading(
    *,
    fit: DailyJointArSpotFit,
    jacobian: np.ndarray,
    feature_changes: np.ndarray,
    standardized_scores: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float, float]:
    """Regularise the state/spot loading to the decoded feature covariance.

    The target is ``Cov(delta y, z)`` in the trailing window; feature changes
    are standardised in the least-squares equation so DLV and backbone rows
    contribute comparably.  ``ridge`` controls the departure from the jointly
    fitted state/spot loading.
    """
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    beta_target = feature_changes.T @ standardized_scores / (len(standardized_scores) - 1)
    base = fit.factor_basis @ fit.innovation_factor.T @ fit.gamma
    beta_base = jacobian @ base
    scale = np.maximum(feature_changes.std(axis=0, ddof=1), 1.0e-7)
    design = jacobian / scale[:, None]
    response = (beta_target - beta_base) / scale
    correction = np.linalg.solve(
        design.T @ design + ridge * np.eye(design.shape[1]), design.T @ response
    )
    calibrated = base + correction
    residual_before = float(np.sqrt(np.mean(((beta_base - beta_target) / scale) ** 2)))
    residual_after = float(
        np.sqrt(np.mean(((jacobian @ calibrated - beta_target) / scale) ** 2))
    )
    return calibrated, residual_before, residual_after


def _load_history(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    dates = saved.get("dates")
    states = saved.get("states")
    spots = saved.get("spot_refs")
    if not isinstance(dates, list) or not isinstance(states, torch.Tensor) or not isinstance(spots, torch.Tensor):
        raise ValueError(f"invalid state history: {path}")
    return dates, states.double().numpy(), spots.double().numpy()


def _load_model(checkpoint: Path) -> object:
    """Load a complete or epoch-capped Direct-200 checkpoint reproducibly."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("phase") == "complete":
        return load_presentation_model(checkpoint, device="cpu", precision=32)[1]
    if payload.get("schema") != "flex-joint-direct-dynamics-v1":
        raise ValueError(f"not a Direct-200 checkpoint: {checkpoint}")
    config = _checkpoint_config(payload)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no model state")
    daily_state = state_dict.get("daily_state_embedding.weight")
    if not isinstance(daily_state, torch.Tensor) or daily_state.ndim != 2:
        raise ValueError("checkpoint has no daily state history")
    environment = BaseEnvironment(precision=32, device="cpu", debug=False)
    model = JointDirectDynamicsModel(environment, num_days=len(daily_state), config=config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_experiment(
    *,
    checkpoint: Path,
    pre2025_history: Path,
    forward_history: Path,
    output: Path,
    lookback: int,
    rank: int,
    paths_per_origin: int,
    stride: int,
    ridge: float,
    batch_size: int,
    seed: int,
    exclude_forecast_start: str | None = None,
    exclude_forecast_end: str | None = None,
) -> dict[str, object]:
    """Run rolling 2025 next-session forecasts and return leverage diagnostics."""
    if paths_per_origin <= 0 or paths_per_origin % 2 or lookback < 30 or rank <= 0:
        raise ValueError("invalid rolling experiment dimensions")
    model = _load_model(checkpoint)
    pre_dates, pre_states, pre_spots = _load_history(pre2025_history)
    forward_dates, forward_states, forward_spots = _load_history(forward_history)
    if pre_dates[-1] >= forward_dates[0]:
        raise ValueError("history dates overlap or are out of order")
    dates = pre_dates + forward_dates
    states = np.concatenate((pre_states, forward_states))
    spots = np.concatenate((pre_spots, forward_spots))
    gross_returns = spots[1:] / spots[:-1]
    spot_scores = _historical_spot_scores(model, states[:-1], gross_returns, batch_size=batch_size)
    features = selected_log_features(model, states, batch_size=batch_size)
    feature_changes = np.diff(features, axis=0)
    start = len(pre_states) - 1
    origins = np.arange(start, len(states) - 1, stride, dtype=np.int64)
    forecast_dates = np.asarray(dates, dtype="U10")[origins + 1]
    if (exclude_forecast_start is None) != (exclude_forecast_end is None):
        raise ValueError("both forecast-exclusion endpoints must be supplied together")
    if exclude_forecast_start is not None and exclude_forecast_end is not None:
        excluded = (forecast_dates >= exclude_forecast_start) & (
            forecast_dates <= exclude_forecast_end
        )
        origins = origins[~excluded]
        forecast_dates = forecast_dates[~excluded]
    if len(origins) == 0:
        raise ValueError("forecast-date exclusion removed every origin")
    next_state_rows: list[np.ndarray] = []
    spot_rows: list[np.ndarray] = []
    origin_rows: list[np.ndarray] = []
    fit_rmse_before: list[float] = []
    fit_rmse_after: list[float] = []
    loading_norms: list[float] = []
    started = perf_counter()
    last_report = started
    for index, origin in enumerate(origins):
        window_states = states[origin - lookback : origin + 1]
        window_scores = spot_scores[origin - lookback : origin]
        fit = fit_daily_joint_ar_spot(window_states, window_scores, rank=rank)
        standardized = (window_scores - fit.spot_score_mean) / fit.spot_score_std
        jacobian = feature_jacobian(model, states[origin])
        loading, before, after = calibrated_spot_loading(
            fit=fit,
            jacobian=jacobian,
            feature_changes=feature_changes[origin - lookback : origin],
            standardized_scores=standardized,
            ridge=ridge,
        )
        next_states, next_scores = sample_daily_joint_ar_spot(
            fit,
            states[origin],
            count=paths_per_origin,
            seed=int(
                np.random.SeedSequence((seed, int(origin))).generate_state(
                    1, dtype=np.uint32
                )[0]
            ),
        )
        standardized_next = (next_scores[0] - fit.spot_score_mean) / fit.spot_score_std
        baseline = fit.factor_basis @ fit.innovation_factor.T @ fit.gamma
        next_states[0] += (standardized_next[:, None] * (loading - baseline)[None, :])
        next_state_rows.append(next_states[0])
        gross = _terminal_front_gross_returns(model, states[origin], next_scores[0])
        spot_rows.append(np.log(gross))
        origin_rows.append(np.full(paths_per_origin, origin, dtype=np.int64))
        fit_rmse_before.append(before)
        fit_rmse_after.append(after)
        loading_norms.append(float(np.linalg.norm(loading)))
        now = perf_counter()
        if index + 1 == len(origins) or now - last_report >= 300.0 or (index + 1) % 10 == 0:
            done = index + 1
            elapsed = now - started
            print(
                f"rolling_feature_loading={done}/{len(origins)} elapsed_s={elapsed:.1f} "
                f"eta_s={elapsed * (len(origins) - done) / done:.1f}",
                flush=True,
            )
            last_report = now
    simulated_states = np.concatenate(next_state_rows)
    simulated_returns = np.concatenate(spot_rows)
    simulated_features = selected_log_features(model, simulated_states, batch_size=batch_size)
    simulated_atm = _decode_atm_volatilities(model, simulated_states, batch_size=batch_size)
    origin_feature = np.repeat(features[origins], paths_per_origin, axis=0)
    origin_atm = _decode_atm_volatilities(model, states[origins], batch_size=batch_size)
    origin_atm = np.repeat(origin_atm, paths_per_origin, axis=0)
    feature_correlation = [
        float(np.corrcoef(simulated_returns, simulated_features[:, column] - origin_feature[:, column])[0, 1])
        for column in range(simulated_features.shape[1])
    ]
    atm_correlation = [
        float(np.corrcoef(simulated_returns, np.log(simulated_atm[:, column]) - np.log(origin_atm[:, column]))[0, 1])
        for column in range(simulated_atm.shape[1])
    ]
    historical_return = np.log(spots[origins + 1] / spots[origins])
    historical_feature_correlation = [
        float(np.corrcoef(historical_return, feature_changes[origins, column])[0, 1])
        for column in range(features.shape[1])
    ]
    historical_atm = _decode_atm_volatilities(model, states[origins], batch_size=batch_size)
    historical_atm_next = _decode_atm_volatilities(model, states[origins + 1], batch_size=batch_size)
    historical_atm_correlation = [
        float(np.corrcoef(historical_return, np.log(historical_atm_next[:, column]) - np.log(historical_atm[:, column]))[0, 1])
        for column in range(historical_atm.shape[1])
    ]
    result: dict[str, object] = {
        "schema": "dysanos-v11-rolling-feature-loading-v1",
        "decoder_training_end": pre_dates[-1],
        "forecast_dates": forecast_dates.tolist(),
        "lookback": lookback,
        "rank": rank,
        "paths_per_origin": paths_per_origin,
        "stride": stride,
        "ridge": ridge,
        "exclude_forecast_start": exclude_forecast_start,
        "exclude_forecast_end": exclude_forecast_end,
        "feature_fit_rmse_before_mean": float(np.mean(fit_rmse_before)),
        "feature_fit_rmse_after_mean": float(np.mean(fit_rmse_after)),
        "loading_norm_mean": float(np.mean(loading_norms)),
        "business_dtes": list(model.config.geometry.model_business_dtes),
        "historical_central_dlv_spot_correlation": historical_feature_correlation[: len(model.config.geometry.model_business_dtes)],
        "simulated_central_dlv_spot_correlation": feature_correlation[: len(model.config.geometry.model_business_dtes)],
        "historical_backbone_spot_correlation": historical_feature_correlation[len(model.config.geometry.model_business_dtes) :],
        "simulated_backbone_spot_correlation": feature_correlation[len(model.config.geometry.model_business_dtes) :],
        "historical_atm_spot_correlation": historical_atm_correlation,
        "simulated_atm_spot_correlation": atm_correlation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pre2025-history", type=Path, required=True)
    parser.add_argument("--forward-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--paths-per-origin", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--exclude-forecast-start")
    parser.add_argument("--exclude-forecast-end")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    result = run_experiment(**vars(arguments))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
