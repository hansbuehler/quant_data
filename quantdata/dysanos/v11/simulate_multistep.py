"""Forward DYSANOS paths and comparable spot/ATM-IV dynamics diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from quantdata.dysanos.flex.front_spot_copula import _decode_surface
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import exact_daily_transition
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_inverse_cdf
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.conditional_sequence import (
    ConditionalInnovationModel,
    ConditionalSequenceConfig,
)
from quantdata.dysanos.v11.factorial_statistics import (
    AUDIT_BUSINESS_DTES,
    _leverage_values,
    downside_leverage,
)
from quantdata.torch import BaseEnvironment


def _pooled_path_lag_correlations(
    values: np.ndarray, *, lags: tuple[int, ...] = (1, 5, 20)
) -> dict[int, list[float]]:
    """Pool within-path lag pairs, never pairs across path boundaries."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim == 2:
        series = series[..., None]
    if series.ndim != 3 or not np.all(np.isfinite(series)):
        raise ValueError("path series must be finite with path/time/column axes")
    result: dict[int, list[float]] = {}
    for lag in lags:
        if lag <= 0 or lag >= series.shape[1] - 1:
            raise ValueError("lag leaves too few within-path pairs")
        left = series[:, :-lag].reshape(-1, series.shape[2])
        right = series[:, lag:].reshape(-1, series.shape[2])
        left = left - np.mean(left, axis=0)
        right = right - np.mean(right, axis=0)
        denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
        result[lag] = np.divide(
            np.sum(left * right, axis=0), denominator,
            out=np.full(series.shape[2], np.nan), where=denominator > 0.0,
        ).tolist()
    return result


def _path_radius_tails(
    states: np.ndarray,
    *,
    equilibrium_mean: np.ndarray,
    transition: np.ndarray,
    innovation_covariance: np.ndarray,
    deterministic_shift: np.ndarray,
) -> dict[int, dict[str, float | int]]:
    """Whiten 5/10/15/20-session state increments within each path."""
    paths, levels, dimension = states.shape
    if levels <= 21:
        raise ValueError("path radius audit requires more than 20 transitions")
    reference = float(chi2.ppf(0.99, dimension) / dimension)
    panels: dict[int, dict[str, float | int]] = {}
    for weeks in (1, 2, 3, 4):
        horizon = 5 * weeks
        power = np.eye(dimension)
        covariance = np.zeros((dimension, dimension), dtype=np.float64)
        shift = np.zeros(dimension)
        for _ in range(horizon):
            covariance += power.T @ innovation_covariance @ power
            shift += deterministic_shift @ power
            power = power @ transition
        mean = equilibrium_mean + (
            states[:, :-horizon] - equilibrium_mean
        ) @ power + shift
        residual = (states[:, horizon:] - mean).reshape(-1, dimension)
        factor = np.linalg.cholesky(0.5 * (covariance + covariance.T))
        white = solve_triangular(factor, residual.T, lower=True).T
        radius = np.mean(white * white, axis=1)
        panels[weeks] = {
            "observations": int(len(radius)),
            "gaussian_p99": reference,
            "mean": float(np.mean(radius)),
            "p95": float(np.quantile(radius, 0.95)),
            "p99": float(np.quantile(radius, 0.99)),
            "fraction_above_gaussian_p99": float(np.mean(radius > reference)),
        }
    return panels


def _conditional_scale_mixture_normal_score(
    raw_spot: torch.Tensor,
    *,
    conditional_log_mean: torch.Tensor,
    conditional_log_variance: float,
    stationary_log_variance: float,
    unconditional_scale: float,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Gaussianize the conditional common-scale spot shock on the device."""
    log_scale = (
        math.log(unconditional_scale)
        + conditional_log_mean[:, None]
        + math.sqrt(2.0 * conditional_log_variance) * nodes[None, :]
        - 0.5 * stationary_log_variance
    )
    probability = torch.sum(
        torch.special.ndtr(raw_spot[:, None] / torch.exp(0.5 * log_scale))
        * weights[None, :],
        dim=1,
    ) / math.sqrt(math.pi)
    return torch.special.ndtri(torch.clamp(probability, 1.0e-7, 1.0 - 1.0e-7))


def _atm_log_iv(
    model: object, surface: object, tenors_y: torch.Tensor
) -> torch.Tensor:
    expiries = tenors_y[None, :].expand(len(surface.support_strikes), -1)
    calls = model.sanos_decoder.price_calls_batch(
        surface, expiries_y=expiries, strikes=torch.ones_like(expiries)
    )
    eps = 32.0 * torch.finfo(calls.dtype).eps
    probabilities = torch.clamp(0.5 * (calls + 1.0), 0.5 + eps, 1.0 - eps)
    iv = 2.0 * torch.special.ndtri(probabilities) / torch.sqrt(expiries)
    if not bool(torch.all(torch.isfinite(iv) & (iv > 0.0))):
        raise RuntimeError("simulated ATM IV is not positive and finite")
    return torch.log(iv)


def _load_sequence_model(
    path: Path, *, device: str
) -> tuple[ConditionalInnovationModel, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "dysanos-v11-conditional-sequence-v1":
        raise ValueError("not a conditional-sequence checkpoint")
    config = ConditionalSequenceConfig(**payload["model_config"])
    if (
        payload.get("model_config_version") != config.version.unique_id48
        or payload.get("model_user_version") != config.user_version
    ):
        raise ValueError("conditional-sequence model version changed")
    env = BaseEnvironment(precision=32, device=device, debug=False)
    basis = payload["model_state_dict"]["white_mean_basis"]
    model = ConditionalInnovationModel(
        env, config=config, white_mean_basis=basis
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    values = {
        name: torch.as_tensor(payload[name], dtype=model.fdtype, device=model.device)
        for name in (
            "feature_mean", "feature_std", "state_mean", "state_basis",
            "initial_context", "initial_state",
        )
    }
    return model, values


def run(
    *,
    variant: str,
    joint_checkpoint: Path,
    target_history: Path,
    observation_report: Path,
    output: Path,
    weekly_one_step_report: Path | None = None,
    daily_parameters: Path | None = None,
    tail_fit_report: Path | None = None,
    sequence_checkpoint: Path | None = None,
    paths: int = 1_024,
    steps: int = 60,
    batch_size: int = 256,
    seed: int = 20261018,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Simulate a fixed-origin path ensemble with exact front-SANOS spot draws."""
    variants = {
        "weekly_normal", "weekly_tail", "daily_normal", "daily_tail",
        "lstm", "selective_ssm",
    }
    if variant not in variants or paths <= 0 or paths % 2 or steps <= 21 or batch_size <= 0 or batch_size % 2 or paths % batch_size:
        raise ValueError("invalid multi-step variant or path geometry")
    model, audit = load_capped_joint_model(joint_checkpoint, device=device)
    history = torch.load(target_history, map_location="cpu", weights_only=False)
    initial_state = history["states"][-1].to(dtype=model.fdtype, device=model.device)
    observation = json.loads(observation_report.read_text(encoding="utf-8"))
    downside_reference = observation["pre_evaluation"]["downside_leverage"]
    all_dtes = np.asarray(model.config.geometry.model_expiries_y) * 252.0
    selected = []
    for tenor in AUDIT_BUSINESS_DTES:
        matches = np.flatnonzero(np.isclose(all_dtes, tenor, atol=1.0e-9))
        if len(matches) != 1:
            raise ValueError(f"model misses {tenor}D tenor")
        selected.append(int(matches[0]))
    tenors_y = torch.as_tensor(
        np.asarray(model.config.geometry.model_expiries_y)[selected],
        dtype=model.fdtype, device=model.device,
    )
    mean = model.equilibrium_mean.detach()
    sequence_model = None
    sequence_values = None
    if variant.startswith("weekly"):
        if weekly_one_step_report is None:
            raise ValueError("weekly paths need the fitted front-spot copula")
        daily_transition, daily_factor, _ = exact_daily_transition(model)
        transition = daily_transition.detach()
        factor = daily_factor.detach()
        copula = json.loads(weekly_one_step_report.read_text(encoding="utf-8"))[
            "front_spot_copula"
        ]
        loading = torch.as_tensor(
            copula["state_loading"], dtype=model.fdtype, device=model.device
        )
        orthogonal_std = float(copula["orthogonal_std"])
        total_std = float(copula["total_std"])
        spot_loading = torch.zeros_like(mean)
        score_mean = 0.0
    else:
        if daily_parameters is None:
            raise ValueError("daily paths need full daily joint parameters")
        with np.load(daily_parameters, allow_pickle=False) as stream:
            parameters = {name: stream[name] for name in stream.files}
        mean = torch.as_tensor(
            parameters["equilibrium_mean"], dtype=model.fdtype, device=model.device
        )
        transition = torch.as_tensor(
            parameters["transition"], dtype=model.fdtype, device=model.device
        )
        factor = torch.as_tensor(
            parameters["innovation_cholesky"], dtype=model.fdtype, device=model.device
        )
        spot_loading = torch.as_tensor(
            parameters["spot_loading"], dtype=model.fdtype, device=model.device
        )
        score_mean = float(parameters["training_spot_score_mean"])
        if variant in {"lstm", "selective_ssm"}:
            if sequence_checkpoint is None:
                raise ValueError("sequence paths need their fitted model checkpoint")
            sequence_model, sequence_values = _load_sequence_model(
                sequence_checkpoint, device=device
            )
    tail_config = None
    tail_nodes = None
    tail_weights = None
    if variant.endswith("tail"):
        if tail_fit_report is None:
            raise ValueError("Tail paths need fitted log-variance parameters")
        tail_config = json.loads(tail_fit_report.read_text(encoding="utf-8"))[
            "fit"
        ]["config"]
        nodes, weights = np.polynomial.hermite.hermgauss(48)
        tail_nodes = torch.as_tensor(nodes, dtype=model.fdtype, device=model.device)
        tail_weights = torch.as_tensor(weights, dtype=model.fdtype, device=model.device)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    path_states = np.empty((paths, steps + 1, model.config.state_size), dtype=np.float32)
    path_spot = np.empty((paths, steps), dtype=np.float32)
    path_log_iv = np.empty((paths, steps + 1, len(selected)), dtype=np.float32)
    total_dlvs = 0
    at_cap = 0
    near_cap = 0
    maximum_overshoot = -math.inf
    cap = model.config.geometry.maximum_log_dlvsq
    count_dlvs = model.config.geometry.num_model_expiries * model.config.geometry.num_inner_strikes
    started = perf_counter()
    for first in range(0, paths, batch_size):
        last = first + batch_size
        state = initial_state[None, :].expand(batch_size, -1).clone()
        context = None
        if sequence_model is not None:
            assert sequence_values is not None
            context = sequence_values["initial_context"][None, :].expand(batch_size, -1, -1).clone()
        if tail_config is not None:
            variance = float(tail_config["stationary_log_variance"])
            scale = float(tail_config["unconditional_scale"])
            persistence = float(tail_config["persistence"])
            if variant == "weekly_tail":
                persistence = persistence ** (1.0 / 5.0)
            half_latent = torch.randn(
                batch_size // 2, dtype=model.fdtype, device=model.device,
                generator=generator,
            ) * math.sqrt(variance)
            latent = torch.cat((half_latent, half_latent))
        path_states[first:last, 0] = state.detach().cpu().numpy()
        with torch.inference_mode():
            surface = _decode_surface(model, state)
            log_iv = _atm_log_iv(model, surface, tenors_y)
            path_log_iv[first:last, 0] = log_iv.cpu().numpy()
            for step in range(steps):
                half_epsilon = torch.randn(
                    batch_size // 2, model.config.state_size,
                    dtype=model.fdtype, device=model.device, generator=generator,
                )
                epsilon = torch.cat((half_epsilon, -half_epsilon))
                half_eta = torch.randn(
                    batch_size // 2, dtype=model.fdtype, device=model.device,
                    generator=generator,
                )
                eta = torch.cat((half_eta, -half_eta))
                if variant.startswith("weekly"):
                    base_spot_score = (
                        epsilon @ loading + orthogonal_std * eta
                    ) / total_std
                else:
                    base_spot_score = eta
                multiplier = 1.0
                if tail_config is not None:
                    assert tail_nodes is not None and tail_weights is not None
                    conditional_log_mean = persistence * latent
                    conditional_log_variance = variance * (1.0 - persistence**2)
                    half_zeta = torch.randn(
                        batch_size // 2, dtype=model.fdtype, device=model.device,
                        generator=generator,
                    )
                    zeta = torch.cat((half_zeta, half_zeta))
                    latent = (
                        persistence * latent
                        + math.sqrt(variance * (1.0 - persistence**2)) * zeta
                    )
                    multiplier = torch.sqrt(
                        torch.as_tensor(scale, dtype=model.fdtype, device=model.device)
                        * torch.exp(latent - 0.5 * variance)
                    )
                    spot_score = _conditional_scale_mixture_normal_score(
                        multiplier * base_spot_score,
                        conditional_log_mean=conditional_log_mean,
                        conditional_log_variance=conditional_log_variance,
                        stationary_log_variance=variance,
                        unconditional_scale=scale,
                        nodes=tail_nodes,
                        weights=tail_weights,
                    )
                else:
                    spot_score = base_spot_score
                white_mean = torch.zeros_like(epsilon)
                log_scale = torch.zeros(batch_size, dtype=model.fdtype, device=model.device)
                if sequence_model is not None:
                    assert context is not None
                    white_mean, log_scale = sequence_model(context, spot_score)
                surface_white = white_mean + (
                    torch.exp(log_scale)[:, None] * epsilon * (
                        multiplier[:, None] if isinstance(multiplier, torch.Tensor) else multiplier
                    )
                )
                gross = front_week_sanos_inverse_cdf(surface, spot_score)
                state = (
                    mean + (state - mean) @ transition
                    + (spot_score - score_mean)[:, None] * spot_loading
                    + surface_white @ factor.T
                )
                raw = model.raw_states_from_states(state)[..., :count_dlvs]
                total_dlvs += raw.numel()
                at_cap += int(torch.sum(raw >= cap).item())
                near_cap += int(torch.sum(raw >= cap - 0.1).item())
                maximum_overshoot = max(maximum_overshoot, float(torch.max(raw - cap).item()))
                next_surface = _decode_surface(model, state)
                next_log_iv = _atm_log_iv(model, next_surface, tenors_y)
                path_spot[first:last, step] = torch.log(gross).cpu().numpy()
                path_log_iv[first:last, step + 1] = next_log_iv.cpu().numpy()
                path_states[first:last, step + 1] = state.cpu().numpy()
                if sequence_model is not None:
                    assert context is not None and sequence_values is not None
                    state_score = (
                        state - sequence_values["state_mean"]
                    ) @ sequence_values["state_basis"]
                    log_radius = torch.log(torch.clamp(
                        torch.mean(surface_white.square(), dim=1), min=1.0e-8
                    ))
                    next_feature = torch.cat(
                        (state_score, spot_score[:, None], log_radius[:, None], next_log_iv - log_iv),
                        dim=1,
                    )
                    next_feature = (
                        next_feature - sequence_values["feature_mean"]
                    ) / sequence_values["feature_std"]
                    context = torch.cat((context[:, 1:], next_feature[:, None, :]), dim=1)
                surface = next_surface
                log_iv = next_log_iv
                elapsed = perf_counter() - started
                completed = first * steps + (step + 1) * batch_size
                total = paths * steps
                remaining = elapsed * (total - completed) / completed
                print(
                    f"multistep {variant} paths={first + 1}-{last}/{paths} "
                    f"step={step + 1}/{steps} elapsed={elapsed:.1f}s "
                    f"remaining_estimate={remaining:.1f}s",
                    flush=True,
                )
    changes = np.diff(path_log_iv.astype(np.float64), axis=1)
    spot = path_spot.astype(np.float64)
    correlation, covariance, slope = _leverage_values(
        spot.reshape(-1), changes.reshape(-1, len(selected))
    )
    state_covariance = factor.detach().cpu().double().numpy()
    state_covariance = state_covariance @ state_covariance.T
    deterministic_shift = np.zeros(model.config.state_size)
    if not variant.startswith("weekly"):
        beta = spot_loading.detach().cpu().double().numpy()
        state_covariance += np.outer(beta, beta)
        deterministic_shift = -score_mean * beta
    report: dict[str, object] = {
        "schema": "dysanos-v11-multistep-path-statistics-v1",
        "status": "complete",
        "variant": variant,
        "checkpoint": audit,
        "initial_date": history["dates"][-1],
        "paths": paths,
        "steps": steps,
        "independent_antithetic_pairs": paths // 2,
        "spot_sampling": "each step uses the current decoded one-day SANOS front quantile",
        "tail_assumption": (
            None if tail_config is None else
            "OU scale multiplies spot and surface Gaussian shocks; conditional PIT preserves the front SANOS spot marginal"
        ),
        "business_dtes": list(AUDIT_BUSINESS_DTES),
        "leverage": {
            "correlation": correlation.tolist(),
            "covariance": covariance.tolist(),
            "slope": slope.tolist(),
        },
        "autocorrelation": {
            "atm_log_iv_changes": _pooled_path_lag_correlations(changes),
            "atm_log_iv_levels": _pooled_path_lag_correlations(path_log_iv),
            "absolute_spot_log_returns": _pooled_path_lag_correlations(np.abs(spot)),
            "absolute_atm_log_iv_changes": _pooled_path_lag_correlations(np.abs(changes)),
        },
        "downside_leverage": downside_leverage(
            spot.reshape(-1), changes.reshape(-1, len(selected)),
            spot_down_threshold=float(downside_reference["spot_down_threshold"]),
            iv_up_thresholds=np.asarray(downside_reference["iv_up_thresholds"]),
        ),
        "innovation_radius": _path_radius_tails(
            path_states.astype(np.float64),
            equilibrium_mean=mean.detach().cpu().double().numpy(),
            transition=transition.detach().cpu().double().numpy(),
            innovation_covariance=state_covariance,
            deterministic_shift=deterministic_shift,
        ),
        "decoder_cap": {
            "dlv_coordinates": total_dlvs,
            "fraction_at_hard_cap": at_cap / total_dlvs,
            "fraction_within_0_1_log_unit_of_cap": near_cap / total_dlvs,
            "maximum_log_coordinate_overshoot": maximum_overshoot,
        },
        "spot_return_quantiles": {
            str(p): float(np.quantile(spot, p)) for p in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weekly-one-step-report", type=Path)
    parser.add_argument("--daily-parameters", type=Path)
    parser.add_argument("--tail-fit-report", type=Path)
    parser.add_argument("--sequence-checkpoint", type=Path)
    parser.add_argument("--paths", type=int, default=1_024)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(**vars(args))
    print(f"multistep {report['variant']} complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
