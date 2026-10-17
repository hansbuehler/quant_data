"""Compare historical and generated DYSANOS spot/ATM-vol correlations."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy.special import ndtri

from quantdata.dysanos.flex.front_spot_copula import (
    antithetic_normals,
    fit_front_spot_copula_from_history,
    sample_correlated_front_scenarios,
)
from quantdata.dysanos.flex.presentation_hedging import (
    _decode_surface,
    exact_daily_transition,
    load_presentation_model,
)


def _atm_iv_from_normalized_calls(
    normalized_calls: np.ndarray,
    expiries_y: np.ndarray,
) -> np.ndarray:
    """Invert normalized forward-ATM Black calls analytically."""
    calls = np.asarray(normalized_calls, dtype=np.float64)
    expiries = np.asarray(expiries_y, dtype=np.float64)
    if (
        calls.ndim != 2
        or expiries.shape != (calls.shape[1],)
        or np.any(~np.isfinite(calls))
        or np.any((calls <= 0.0) | (calls >= 1.0))
        or np.any(~np.isfinite(expiries))
        or np.any(expiries <= 0.0)
    ):
        raise ValueError("ATM calls and expiries must be positive, finite and aligned")
    probabilities = np.clip(
        0.5 * (calls + 1.0),
        0.5 + np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    return 2.0 * ndtri(probabilities) / np.sqrt(expiries)[None, :]


def _decode_atm_volatilities(
    model: object,
    states: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    values = torch.as_tensor(states, dtype=model.fdtype, device="cpu")
    expiries_y = np.asarray(model.config.geometry.model_expiries_y, dtype=np.float64)
    chunks: list[np.ndarray] = []
    started = perf_counter()
    with torch.inference_mode():
        for first in range(0, len(values), batch_size):
            state_batch = values[first : first + batch_size].to(model.device)
            surface = _decode_surface(model, state_batch)
            expiries = torch.as_tensor(
                expiries_y, dtype=model.fdtype, device=model.device
            ).unsqueeze(0).expand(len(state_batch), -1)
            calls = model.sanos_decoder.price_calls_batch(
                surface,
                expiries_y=expiries,
                strikes=torch.ones_like(expiries),
            )
            chunks.append(calls.cpu().double().numpy())
            print(
                f"decoded={min(first + batch_size, len(values))}/{len(values)} "
                f"elapsed_s={perf_counter() - started:.1f}",
                flush=True,
            )
    return _atm_iv_from_normalized_calls(np.concatenate(chunks), expiries_y)


def _correlations(spot_returns: np.ndarray, log_vol_changes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.corrcoef(spot_returns, log_vol_changes[:, expiry])[0, 1]
            for expiry in range(log_vol_changes.shape[1])
        ]
    )


def _direct_sanos_history(
    *,
    history_path: Path,
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load rolling-tenor ATM vols and aligned spot returns from direct SANOS."""
    with h5py.File(history_path, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        total_variances = np.asarray(
            stream["projected_atm_total_variances"], dtype=np.float64
        )
    observations = pl.read_parquet(dataset_path / "observations.parquet")
    spot_returns = np.asarray(
        observations["forward_adjusted_spot_return_into_date"].to_list(),
        dtype=np.float64,
    )
    if len(dates) != len(spot_returns) or total_variances.shape != (
        len(dates),
        len(business_dtes),
    ):
        raise ValueError("direct SANOS history and spot observations do not align")
    annualized_volatilities = np.sqrt(
        total_variances / (business_dtes[None, :] / 252.0)
    )
    return dates, business_dtes, annualized_volatilities, spot_returns[1:]


def _antithetic_correlation_standard_errors(
    spot_returns: np.ndarray,
    log_vol_changes: np.ndarray,
) -> np.ndarray:
    """Estimate correlation errors with each antithetic pair as one unit."""
    count = len(spot_returns)
    if count % 2 or log_vol_changes.shape[0] != count:
        raise ValueError("antithetic correlation errors require aligned even samples")
    x = (spot_returns - np.mean(spot_returns)) / np.std(spot_returns, ddof=0)
    result = np.empty(log_vol_changes.shape[1], dtype=np.float64)
    half = count // 2
    for expiry in range(log_vol_changes.shape[1]):
        raw_y = log_vol_changes[:, expiry]
        y = (raw_y - np.mean(raw_y)) / np.std(raw_y, ddof=0)
        correlation = float(np.mean(x * y))
        influence = x * y - 0.5 * correlation * (x * x + y * y)
        pair_influence = 0.5 * (influence[:half] + influence[half:])
        result[expiry] = np.std(pair_influence, ddof=1) / np.sqrt(half)
    return result


def _write_figure(
    output: Path,
    *,
    business_dtes: np.ndarray,
    historical_full: np.ndarray,
    historical_recent: np.ndarray,
    simulated: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(business_dtes, historical_full, color="#202020", label="Historical")
    axis.plot(
        business_dtes,
        historical_recent,
        color="#d55e00",
        label="Historical, recent",
    )
    axis.plot(
        business_dtes,
        simulated,
        color="#0072b2",
        linestyle="--",
        label="DYSANOS generated",
    )
    axis.axhline(0.0, color="#808080", linewidth=0.8)
    axis.set_xlabel("Business days to expiry")
    axis.set_ylabel(r"Corr$(\Delta\log S,\,\Delta\log\sigma_{\rm ATM})$")
    axis.set_ylim(-1.0, 0.05)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def run_diagnostic(
    *,
    checkpoint: Path,
    target_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    output_json: Path,
    output_figure: Path,
    path_count: int,
    batch_size: int,
    recent_start_date: str,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Run the paper-model baseline diagnostic at its terminal state."""
    if path_count <= 0 or path_count % 2 or batch_size <= 0:
        raise ValueError("path_count must be positive and even; batch_size must be positive")
    _, model = load_presentation_model(checkpoint, device=device, precision=32)
    history = torch.load(target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_value = history.get("states")
    spots_value = history.get("spot_refs")
    if (
        not isinstance(dates, list)
        or not isinstance(states_value, torch.Tensor)
        or not isinstance(spots_value, torch.Tensor)
    ):
        raise ValueError("invalid Direct-200 target history")
    states = states_value.double().numpy()
    spots = spots_value.double().numpy()
    daily_transition_t, daily_factor_t, transition_diagnostics = (
        exact_daily_transition(model)
    )
    daily_transition = daily_transition_t.cpu().double().numpy()
    daily_factor = daily_factor_t.cpu().double().numpy()
    copula, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states,
        spots=spots,
        dates=dates,
        daily_transition=daily_transition,
        daily_factor=daily_factor,
        fit_before=date.max,
    )

    reconstructed_atm = _decode_atm_volatilities(
        model, states, batch_size=batch_size
    )
    direct_dates, direct_dtes, direct_atm, direct_spot_return = _direct_sanos_history(
        history_path=historical_sanos_history,
        dataset_path=historical_dataset,
    )
    if not np.array_equal(direct_dates, np.asarray(dates, dtype="U10")):
        raise ValueError("direct SANOS dates and DYSANOS state dates differ")
    reconstructed_vol_change = np.diff(np.log(reconstructed_atm), axis=0)
    recent = direct_dates[1:] >= recent_start_date
    reconstructed_full = _correlations(
        direct_spot_return, reconstructed_vol_change
    )
    reconstructed_recent = _correlations(
        direct_spot_return[recent], reconstructed_vol_change[recent]
    )
    direct_vol_change = np.diff(np.log(direct_atm), axis=0)
    direct_full_all = _correlations(direct_spot_return, direct_vol_change)
    direct_recent_all = _correlations(
        direct_spot_return[recent], direct_vol_change[recent]
    )
    business_dtes = (
        np.asarray(model.config.geometry.model_expiries_y, dtype=np.float64) * 252.0
    )
    direct_indices = np.searchsorted(direct_dtes, business_dtes)
    if np.any(direct_indices == len(direct_dtes)) or np.any(
        direct_dtes[direct_indices] != business_dtes
    ):
        raise ValueError("direct SANOS history lacks a DYSANOS audit expiry")
    historical_full = direct_full_all[direct_indices]
    historical_recent = direct_recent_all[direct_indices]

    state_normals, independent_spot_normals = antithetic_normals(
        count=path_count,
        dimension=model.config.state_size,
        seed=seed,
    )
    next_states, gross_returns = sample_correlated_front_scenarios(
        model=model,
        current_state=states[-1],
        daily_transition=daily_transition,
        daily_factor=daily_factor,
        fit=copula,
        state_normals=state_normals,
        independent_spot_normals=independent_spot_normals,
    )
    next_atm = _decode_atm_volatilities(
        model, next_states, batch_size=batch_size
    )
    simulated_spot_return = np.log(gross_returns)
    simulated_vol_change = np.log(next_atm / reconstructed_atm[-1][None, :])
    simulated = _correlations(simulated_spot_return, simulated_vol_change)
    simulated_standard_error = _antithetic_correlation_standard_errors(
        simulated_spot_return, simulated_vol_change
    )

    report: dict[str, object] = {
        "schema": "dysanos-v11-atm-spot-correlation-v1",
        "definition": "correlation of one-session log spot return with the log change in rolling-tenor forward-ATM Black implied volatility",
        "checkpoint": str(checkpoint.resolve()),
        "target_history": str(target_history.resolve()),
        "historical_sanos_history": str(historical_sanos_history.resolve()),
        "historical_dataset": str(historical_dataset.resolve()),
        "historical_surface_source": "direct per-date SANOS projected ATM term structure",
        "historical_observations": int(len(direct_spot_return)),
        "recent_start_date": recent_start_date,
        "recent_observations": int(np.sum(recent)),
        "simulation_initial_date": dates[-1],
        "simulation_path_count": path_count,
        "simulation_independent_units": path_count // 2,
        "simulation_seed": seed,
        "business_dtes": business_dtes.tolist(),
        "historical_full_correlations": historical_full.tolist(),
        "historical_recent_correlations": historical_recent.tolist(),
        "affine_reconstruction_full_correlations": reconstructed_full.tolist(),
        "affine_reconstruction_recent_correlations": reconstructed_recent.tolist(),
        "simulated_correlations": simulated.tolist(),
        "simulated_correlation_standard_errors": simulated_standard_error.tolist(),
        "transition_diagnostics": transition_diagnostics,
        "front_spot_copula": copula.to_dict(),
        "front_spot_copula_diagnostics": copula_diagnostics,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    _write_figure(
        output_figure,
        business_dtes=business_dtes,
        historical_full=historical_full,
        historical_recent=historical_recent,
        simulated=simulated,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recent-start-date", default="2025-01-01")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run_diagnostic(
        checkpoint=arguments.checkpoint,
        target_history=arguments.target_history,
        historical_sanos_history=arguments.historical_sanos_history,
        historical_dataset=arguments.historical_dataset,
        output_json=arguments.output_json,
        output_figure=arguments.output_figure,
        path_count=arguments.path_count,
        batch_size=arguments.batch_size,
        recent_start_date=arguments.recent_start_date,
        seed=arguments.seed,
        device=arguments.device,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
