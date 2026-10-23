"""Measure ATM leverage after removing the decoded Black backbone.

This is a diagnostic, not a refit: the fitted DLV lattice and its state law are
unchanged, while the independent Black convolution is removed at valuation and
front-spot sampling.  A zero backbone sits outside the bounded native FLEX
coordinates, so it must be imposed on the decoded :class:`SanosSurface`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from quantdata.dysanos.flex.front_spot_copula import (
    _decode_surface,
    antithetic_normals,
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import exact_daily_transition
from quantdata.dysanos.v11.atm_spot_correlation import (
    _atm_iv_from_normalized_calls,
    _decode_atm_volatilities,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.factorial_statistics import (
    AUDIT_BUSINESS_DTES,
    _leverage_values,
)
from quantdata.dysanos.v11.native_atm_history import load_atm_history


def _without_backbone(surface: object) -> object:
    """Return a SANOS surface whose option values contain its DLV mixture only."""
    return replace(
        surface,
        backbone_total_vols=torch.zeros_like(surface.backbone_total_vols),
    )


def _decode_dlv_only_atm(
    model: object,
    states: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Decode DLV-only forward-ATM Black IV on the native expiry grid."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != model.config.state_size:
        raise ValueError("states do not match the DYSANOS state geometry")
    expiries_y = np.asarray(model.config.geometry.model_expiries_y, dtype=np.float64)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for first in range(0, len(values), batch_size):
            state = torch.as_tensor(
                values[first : first + batch_size],
                dtype=model.fdtype,
                device=model.device,
            )
            surface = _without_backbone(_decode_surface(model, state))
            expiries = torch.as_tensor(
                expiries_y, dtype=model.fdtype, device=model.device
            )[None].expand(len(state), -1)
            calls = model.sanos_decoder.price_calls_batch(
                surface,
                expiries_y=expiries,
                strikes=torch.ones_like(expiries),
            )
            outputs.append(calls.cpu().double().numpy())
    return _atm_iv_from_normalized_calls(np.concatenate(outputs), expiries_y)


def _dlv_only_front_gross(
    surface: object,
    gaussian_score: np.ndarray,
) -> np.ndarray:
    """Map Gaussian copula scores into the front discrete DLV marginal.

    With the continuous Black factor removed, the front law is exactly the
    finite DLV mixture.  Its inverse CDF is intentionally discrete rather than
    being smoothed with an artificial epsilon backbone.
    """
    score = np.asarray(gaussian_score, dtype=np.float64)
    support = surface.support_strikes[0, 0].detach().cpu().double().numpy()
    mass = np.maximum(
        surface.densities[0, 0].detach().cpu().double().numpy(), 0.0
    )
    if (
        score.ndim != 1
        or support.ndim != 1
        or mass.shape != support.shape
        or np.any(~np.isfinite(score))
        or np.any(support <= 0.0)
        or np.sum(mass) <= 0.0
    ):
        raise ValueError("invalid DLV-only front mixture")
    mass /= np.sum(mass)
    support /= float(mass @ support)
    cdf = np.cumsum(mass)
    indices = np.searchsorted(cdf, ndtr(score), side="left")
    return support[np.minimum(indices, len(support) - 1)]


def _cross_tenor_correlation(log_iv_changes: np.ndarray) -> np.ndarray:
    """Return the contemporaneous correlation matrix of rolling-tenor IV moves."""
    changes = np.asarray(log_iv_changes, dtype=np.float64)
    if (
        changes.ndim != 2
        or len(changes) < 3
        or np.any(~np.isfinite(changes))
        or np.any(np.std(changes, axis=0) <= np.finfo(np.float64).tiny)
    ):
        raise ValueError("ATM-IV changes must be finite with nonzero variance")
    return np.corrcoef(changes, rowvar=False)


def run(
    *,
    joint_checkpoint: Path,
    target_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    observation_report: Path,
    output: Path,
    path_count: int = 10_000,
    batch_size: int = 256,
    seed: int = 20261019,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Audit the DLV-only leverage term structure at the terminal state."""
    if path_count <= 0 or path_count % 2 or batch_size <= 0:
        raise ValueError("path count must be positive/even and batch size positive")
    model, checkpoint_audit = load_capped_joint_model(joint_checkpoint, device=device)
    payload = torch.load(target_history, map_location="cpu", weights_only=False)
    dates = payload.get("dates")
    state_value = payload.get("states")
    if not isinstance(dates, list) or not isinstance(state_value, torch.Tensor):
        raise ValueError("invalid fitted-state history")
    states = state_value.double().numpy()
    observed_dates, dtes, native_atm, simple_spot = load_atm_history(
        history_path=historical_sanos_history, dataset_path=historical_dataset
    )
    if list(observed_dates[: len(dates)]) != dates:
        raise ValueError("native historical observations and states do not align")
    gross_train = 1.0 + simple_spot[: len(dates) - 1]
    normalized_spot = np.concatenate(([1.0], np.cumprod(gross_train)))
    transition_t, factor_t, transition_diagnostics = exact_daily_transition(model)
    transition = transition_t.cpu().double().numpy()
    factor = factor_t.cpu().double().numpy()
    copula, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states,
        spots=normalized_spot,
        dates=dates,
        daily_transition=transition,
        daily_factor=factor,
        fit_before=date(2025, 1, 1),
    )
    model_dtes = np.asarray(model.config.geometry.model_expiries_y) * 252.0
    indices: list[int] = []
    for tenor in AUDIT_BUSINESS_DTES:
        matches = np.flatnonzero(np.isclose(model_dtes, tenor, atol=1.0e-9))
        if len(matches) != 1 or not np.any(np.isclose(dtes, tenor, atol=1.0e-9)):
            raise ValueError(f"missing {tenor}D ATM audit expiry")
        indices.append(int(matches[0]))
    decoded = _decode_dlv_only_atm(model, states, batch_size=batch_size)[:, indices]
    decoded_full = _decode_atm_volatilities(
        model, states, batch_size=batch_size
    )[:, indices]
    decoded_historical = _leverage_values(
        np.log(gross_train), np.diff(np.log(decoded), axis=0)
    )
    state_normals, independent_spot_normals = antithetic_normals(
        count=path_count, dimension=model.config.state_size, seed=seed
    )
    mean = model.equilibrium_mean.detach().cpu().double().numpy()
    next_states = mean + (states[-1] - mean) @ transition
    next_states = next_states[None] + state_normals @ factor.T
    score = (
        state_normals @ copula.state_loading
        + copula.orthogonal_std * independent_spot_normals
    ) / copula.total_std
    with torch.inference_mode():
        current_surface = _without_backbone(_decode_surface(
            model,
            torch.as_tensor(states[-1: ], dtype=model.fdtype, device=model.device),
        ))
    gross = _dlv_only_front_gross(current_surface, score)
    next_atm = _decode_dlv_only_atm(model, next_states, batch_size=batch_size)[:, indices]
    next_full_atm = _decode_atm_volatilities(
        model, next_states, batch_size=batch_size
    )[:, indices]
    change = np.log(next_atm / decoded[-1][None])
    full_change = np.log(next_full_atm / decoded_full[-1][None])
    simulated = _leverage_values(np.log(gross), change)
    native_indices = [
        int(np.flatnonzero(np.isclose(dtes, tenor, atol=1.0e-9))[0])
        for tenor in AUDIT_BUSINESS_DTES
    ]
    native_change = np.diff(np.log(native_atm[:, native_indices]), axis=0)
    endpoint_dates = observed_dates[1:]
    pre_mask = endpoint_dates < "2025-01-01"
    forward_mask = ~pre_mask
    observations = json.loads(observation_report.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "schema": "dysanos-v11-dlv-only-leverage-v1",
        "status": "complete",
        "definition": "Decoded DLV lattice unchanged; Black backbone total variance set to zero only after decoding. This is a diagnostic, not a quote-refitted market model.",
        "checkpoint": checkpoint_audit,
        "simulation_origin_date": dates[-1],
        "simulation_path_count": path_count,
        "independent_antithetic_pairs": path_count // 2,
        "business_dtes": list(AUDIT_BUSINESS_DTES),
        "historical_pre2025_correlation": observations["pre_evaluation"]["leverage"]["correlation"],
        "historical_2025_correlation": observations["forward_evaluation"]["leverage"]["correlation"],
        "decoded_dlv_only_training": {
            "correlation": decoded_historical[0].tolist(),
            "covariance": decoded_historical[1].tolist(),
            "slope": decoded_historical[2].tolist(),
        },
        "simulated_dlv_only_terminal": {
            "correlation": simulated[0].tolist(),
            "covariance": simulated[1].tolist(),
            "slope": simulated[2].tolist(),
        },
        "cross_tenor_atm_change_correlation": {
            "native_pre2025": _cross_tenor_correlation(
                native_change[pre_mask]
            ).tolist(),
            "native_2025": _cross_tenor_correlation(
                native_change[forward_mask]
            ).tolist(),
            "decoded_full_pre2025": _cross_tenor_correlation(
                np.diff(np.log(decoded_full), axis=0)
            ).tolist(),
            "decoded_dlv_only_pre2025": _cross_tenor_correlation(
                np.diff(np.log(decoded), axis=0)
            ).tolist(),
            "simulated_full_terminal": _cross_tenor_correlation(
                full_change
            ).tolist(),
            "simulated_dlv_only_terminal": _cross_tenor_correlation(
                change
            ).tolist(),
        },
        "front_dlv_support_size": int(current_surface.support_strikes.shape[-1]),
        "front_spot_log_return_quantiles": {
            str(q): float(np.quantile(np.log(gross), q))
            for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "front_spot_copula": copula.to_dict(),
        "front_spot_copula_diagnostics": copula_diagnostics,
        "transition_diagnostics": transition_diagnostics,
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(**vars(args))
    print(f"DLV-only leverage audit complete: {report['simulation_origin_date']}")


if __name__ == "__main__":
    main()
