"""Score strictly next-session ATM-IV forecasts from frozen DYSANOS laws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import exact_daily_transition
from quantdata.dysanos.v11.atm_spot_correlation import (
    _decode_atm_volatilities,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.factorial_statistics import (
    AUDIT_BUSINESS_DTES,
    transition_sample_indices,
)
from quantdata.dysanos.v11.native_atm_history import load_atm_history
from quantdata.dysanos.v11.predictive_scores import predictive_panel
from quantdata.dysanos.v11.tail_copula import normal_score_from_scale_mixture


def _history(path: Path) -> tuple[list[str], np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dates = payload.get("dates")
    states = payload.get("states")
    if not isinstance(dates, list) or not isinstance(states, torch.Tensor):
        raise ValueError(f"invalid fitted-state history: {path}")
    if len(dates) != len(states):
        raise ValueError("fitted-state dates and rows differ")
    return dates, states.double().numpy()


def _weekly_sampler(model: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transition, factor, _ = exact_daily_transition(model)
    return (
        model.equilibrium_mean.detach().cpu().double().numpy(),
        transition.detach().cpu().double().numpy(),
        factor.detach().cpu().double().numpy(),
    )


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    output: Path,
    forward_state_history: Path | None = None,
    daily_normal_parameters: Path | None = None,
    tail_fit_report: Path | None = None,
    paths_per_origin: int = 128,
    origin_batch_size: int = 16,
    decoder_batch_size: int = 1024,
    seed: int = 20261018,
    device: str = "cpu",
) -> dict[str, object]:
    """Draw conditionally independent antithetic forecasts for every origin."""
    if paths_per_origin <= 0 or paths_per_origin % 2 or origin_batch_size <= 0:
        raise ValueError("paths per origin must be positive/even; batches positive")
    model, checkpoint_audit = load_capped_joint_model(joint_checkpoint, device=device)
    dates, states = _history(training_state_history)
    training_count = len(dates)
    if forward_state_history is not None:
        forward_dates, forward_states = _history(forward_state_history)
        if forward_dates[0] <= dates[-1]:
            raise ValueError("forward states must begin after training states")
        dates += forward_dates
        states = np.concatenate((states, forward_states), axis=0)
    if states.shape[1] != model.config.state_size:
        raise ValueError("fitted-state dimensions do not match the decoder")
    direct_dates, direct_dtes, direct_atm, _ = load_atm_history(
        history_path=historical_sanos_history,
        dataset_path=historical_dataset,
    )
    if list(direct_dates[: len(dates)]) != dates:
        raise ValueError("direct SANOS observations and fitted states differ in date")
    model_dtes = np.asarray(model.config.geometry.model_expiries_y) * 252.0
    model_indices = []
    direct_indices = []
    for tenor in AUDIT_BUSINESS_DTES:
        model_match = np.flatnonzero(np.isclose(model_dtes, tenor, atol=1e-9))
        direct_match = np.flatnonzero(np.isclose(direct_dtes, tenor, atol=1e-9))
        if len(model_match) != 1 or len(direct_match) != 1:
            raise ValueError(f"ATM-IV audit tenor {tenor}D missing")
        model_indices.append(int(model_match[0]))
        direct_indices.append(int(direct_match[0]))
    observations = np.log(direct_atm[1 : len(dates), direct_indices])
    forecasts = np.empty(
        (len(dates) - 1, paths_per_origin, len(AUDIT_BUSINESS_DTES)),
        dtype=np.float64,
    )
    if daily_normal_parameters is None:
        law = "weekly-normal-daily-bridge"
        mean, transition, factor = _weekly_sampler(model)
    else:
        with np.load(daily_normal_parameters, allow_pickle=False) as parameters:
            daily = {name: parameters[name] for name in parameters.files}
        daily_joint = "spot_loading" in daily and "transition" in daily
        law = (
            "daily-spot-joint-normal"
            if daily_joint else "daily-joint-normal-diagonal-pilot"
        )
    tail_config = None
    if tail_fit_report is not None:
        tail_report = json.loads(tail_fit_report.read_text(encoding="utf-8"))
        tail_config = tail_report["fit"]["config"]
        if float(tail_config["stationary_log_variance"]) < 0.0:
            raise ValueError("Tail Diffusion stationary log variance is negative")
        law = law.replace("normal", "tail-stationary")
    rng = np.random.default_rng(seed)
    started = perf_counter()
    for first in range(0, len(forecasts), origin_batch_size):
        last = min(first + origin_batch_size, len(forecasts))
        current = states[first:last]
        normals_half = rng.standard_normal(
            (len(current), paths_per_origin // 2, model.config.state_size)
        )
        normals = np.concatenate((normals_half, -normals_half), axis=1)
        if tail_config is None:
            multiplier = np.ones((len(current), paths_per_origin))
        else:
            variance = float(tail_config["stationary_log_variance"])
            scale = float(tail_config["unconditional_scale"])
            latent_half = rng.normal(
                scale=np.sqrt(variance),
                size=(len(current), paths_per_origin // 2),
            )
            latent = np.concatenate((latent_half, latent_half), axis=1)
            multiplier = np.sqrt(scale * np.exp(latent - 0.5 * variance))
        if daily_normal_parameters is None:
            conditional_mean = mean + (current - mean) @ transition
            next_states = conditional_mean[:, None, :] + (
                multiplier[..., None] * normals
            ) @ factor.T
        elif daily_joint:
            spot_half = rng.standard_normal((len(current), paths_per_origin // 2))
            raw_spot = np.concatenate((spot_half, -spot_half), axis=1)
            spot_score = (
                raw_spot if tail_config is None else
                normal_score_from_scale_mixture(
                    multiplier * raw_spot,
                    stationary_log_variance=float(tail_config["stationary_log_variance"]),
                    unconditional_scale=float(tail_config["unconditional_scale"]),
                )
            )
            conditional_mean = daily["equilibrium_mean"] + (
                current - daily["equilibrium_mean"]
            ) @ daily["transition"]
            next_states = (
                conditional_mean[:, None, :]
                + (spot_score - float(daily["training_spot_score_mean"]))[..., None]
                * daily["spot_loading"]
                + (multiplier[..., None] * normals) @ daily["innovation_cholesky"].T
            )
        else:
            centered = current - daily["state_mean"]
            factor_mean = daily["factor_intercept"] + (
                centered @ daily["factor_basis"]
            ) @ daily["factor_transition"]
            conditional_mean = daily["state_mean"] + (
                factor_mean @ daily["factor_basis"].T
            )
            next_states = conditional_mean[:, None, :] + (
                (multiplier[..., None] * normals) @ daily["innovation_factor"].T
            ) @ daily["factor_basis"].T
        decoded = _decode_atm_volatilities(
            model,
            next_states.reshape(-1, model.config.state_size),
            batch_size=decoder_batch_size,
        )
        forecasts[first:last] = np.log(
            decoded[:, model_indices].reshape(
                len(current), paths_per_origin, len(model_indices)
            )
        )
        elapsed = perf_counter() - started
        remaining = elapsed * (len(forecasts) - last) / last
        print(
            f"predictive {law} origins={last}/{len(forecasts)} "
            f"elapsed={elapsed:.1f}s remaining_estimate={remaining:.1f}s",
            flush=True,
        )
    training, forward = transition_sample_indices(np.asarray(dates)) if (
        forward_state_history is not None
    ) else (np.arange(1, len(dates)), np.array([], dtype=np.int64))
    if training_count != len(training) + 1:
        raise ValueError("training split and fitted-state rows do not match")
    report: dict[str, object] = {
        "schema": "dysanos-v11-atm-iv-next-session-predictive-v1",
        "law": law,
        "checkpoint": checkpoint_audit,
        "training_state_history": str(training_state_history.resolve()),
        "observed_atm_history": str(historical_sanos_history.resolve()),
        "historical_dataset": str(historical_dataset.resolve()),
        "forward_state_history": (
            None if forward_state_history is None else str(forward_state_history.resolve())
        ),
        "tail_fit_report": (
            None if tail_fit_report is None else str(tail_fit_report.resolve())
        ),
        "tail_scale_assumption": (
            None if tail_fit_report is None else "stationary one-step latent scale at each forecast origin"
        ),
        "paths_per_origin": paths_per_origin,
        "independent_antithetic_pairs_per_origin": paths_per_origin // 2,
        "business_dtes": list(AUDIT_BUSINESS_DTES),
        "target": "next-session log forward-ATM Black implied-volatility level",
        "pre_evaluation": predictive_panel(
            observations[training - 1], forecasts[training - 1]
        ),
        "forward_evaluation": (
            None
            if len(forward) == 0
            else predictive_panel(
                observations[forward - 1], forecasts[forward - 1]
            )
        ),
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-state-history", type=Path)
    parser.add_argument("--daily-normal-parameters", type=Path)
    parser.add_argument("--tail-fit-report", type=Path)
    parser.add_argument("--paths-per-origin", type=int, default=128)
    parser.add_argument("--origin-batch-size", type=int, default=16)
    parser.add_argument("--decoder-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"predictive audit {report['law']}: {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
