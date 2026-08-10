"""Empirical diagnostics for the surface-first PCA--AR(1) DYSANOS driver.

The script compares five retained surface factors with the full 20-dimensional
surface innovation.  It writes one JSON report and four publication figures.
The diagnostic sample counts refer to independent simulated state paths; the
historical comparisons use every available state or weekly increment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from cdxcore.bs import bs
from cdxcore.verbose import Context
from scipy.stats import ks_2samp

from quantdata.biz_calendar import DAYS_PER_YEAR

from .torch_prep import GenDysanosExperiment, _load_generator_components


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _standardize(reference: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0)
    floor = math.sqrt(np.finfo(reference.dtype).eps) * max(
        1.0, float(np.max(scale))
    )
    scale = np.maximum(scale, floor)
    return (reference - center) / scale, (sample - center) / scale


def _sliced_wasserstein(
    reference: np.ndarray,
    sample: np.ndarray,
    *,
    rng: np.random.Generator,
    projections: int = 128,
    quantiles: int = 1_001,
) -> float:
    directions = rng.normal(size=(reference.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    probabilities = np.linspace(0.0, 1.0, quantiles)
    reference_quantiles = np.quantile(reference @ directions, probabilities, axis=0)
    sample_quantiles = np.quantile(sample @ directions, probabilities, axis=0)
    return float(np.mean(np.abs(reference_quantiles - sample_quantiles)))


def _distribution_metrics(
    reference: np.ndarray,
    sample: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(sample)):
        raise FloatingPointError("distribution diagnostic received non-finite values")
    standardized_reference, standardized_sample = _standardize(reference, sample)
    reference_covariance = np.cov(standardized_reference, rowvar=False, bias=True)
    sample_covariance = np.cov(standardized_sample, rowvar=False, bias=True)
    covariance_norm = max(
        float(np.linalg.norm(reference_covariance, ord="fro")),
        np.finfo(float).eps,
    )
    ks_values = np.asarray(
        [
            ks_2samp(
                standardized_reference[:, coordinate],
                standardized_sample[:, coordinate],
            ).statistic
            for coordinate in range(reference.shape[1])
        ]
    )
    return {
        "observations_reference": int(reference.shape[0]),
        "observations_simulated": int(sample.shape[0]),
        "dimension": int(reference.shape[1]),
        "mean_rms_in_historical_std": float(
            np.sqrt(np.mean(np.square(np.mean(standardized_sample, axis=0))))
        ),
        "mean_max_abs_in_historical_std": float(
            np.max(np.abs(np.mean(standardized_sample, axis=0)))
        ),
        "relative_covariance_frobenius": float(
            np.linalg.norm(sample_covariance - reference_covariance, ord="fro")
            / covariance_norm
        ),
        "sliced_wasserstein_in_historical_std": _sliced_wasserstein(
            standardized_reference,
            standardized_sample,
            rng=rng,
        ),
        "coordinate_ks_mean": float(np.mean(ks_values)),
        "coordinate_ks_max": float(np.max(ks_values)),
    }


def _pca_summary(changes: np.ndarray, maximum_components: int = 10) -> dict[str, object]:
    if not np.all(np.isfinite(changes)):
        raise FloatingPointError("PCA changes contain non-finite values")
    centered = changes - np.mean(changes, axis=0, keepdims=True)
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular_values)
    ratio = variance / np.sum(variance)
    cumulative = np.cumsum(ratio)
    count = min(maximum_components, ratio.size)
    return {
        "explained_variance_ratio": ratio[:count],
        "cumulative_explained_variance": cumulative[:count],
        "components_for_90_percent": int(np.searchsorted(cumulative, 0.9) + 1),
        "components_for_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
        "components": vectors[:count],
    }


def _principal_angles_degrees(
    left_components: np.ndarray,
    right_components: np.ndarray,
    count: int,
) -> np.ndarray:
    from scipy.linalg import subspace_angles

    count = min(count, left_components.shape[0], right_components.shape[0])
    return np.degrees(
        subspace_angles(
            left_components[:count].T,
            right_components[:count].T,
        )
    )


def _public_pca_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in summary.items()
        if key != "components"
    }


def _leading_pca(changes: np.ndarray, components: int = 5) -> dict[str, object]:
    """Return the leading PCA loadings without constructing a full SVD."""

    if changes.ndim != 2:
        raise ValueError(f"PCA changes must be a matrix, got shape {changes.shape}")
    if components <= 0 or components > min(changes.shape):
        raise ValueError(
            f"components must be in [1, {min(changes.shape)}], got {components}"
        )
    if not np.all(np.isfinite(changes)):
        raise FloatingPointError("PCA changes contain non-finite values")
    centered = changes - np.mean(changes, axis=0, keepdims=True)
    covariance_numerator = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_numerator)
    order = np.argsort(eigenvalues)[::-1][:components]
    selected_values = np.maximum(eigenvalues[order], 0.0)
    total_variance = float(np.sum(np.square(centered)))
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("PCA changes have zero or non-finite total variance")
    return {
        "explained_variance_ratio": selected_values / total_variance,
        "components": eigenvectors[:, order].T,
    }


def _align_eigensurface_signs(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose deterministic reference signs and align candidate signs to them."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(
            "reference and candidate PCA components must be equally shaped matrices, "
            f"got {reference.shape} and {candidate.shape}"
        )
    aligned_reference = np.array(reference, copy=True)
    aligned_candidate = np.array(candidate, copy=True)
    for component in range(reference.shape[0]):
        pivot = int(np.argmax(np.abs(aligned_reference[component])))
        if aligned_reference[component, pivot] < 0.0:
            aligned_reference[component] *= -1.0
        if np.dot(aligned_reference[component], aligned_candidate[component]) < 0.0:
            aligned_candidate[component] *= -1.0
    return aligned_reference, aligned_candidate


def _rankwise_component_angles_degrees(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"component shapes differ: {reference.shape} and {candidate.shape}"
        )
    cosine = np.sum(reference * candidate, axis=1)
    return np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))


def _stylized_statistics(
    changes: np.ndarray,
    spot_changes: np.ndarray,
    *,
    expiries: int,
    strikes: int,
) -> dict[str, float]:
    surface = changes.reshape(-1, expiries, strikes)
    low = max(0, strikes // 4)
    high = min(strikes - 1, 3 * strikes // 4)
    level = np.mean(surface, axis=(1, 2))
    skew = np.mean(surface[:, :, low] - surface[:, :, high], axis=1)
    term = np.mean(surface[:, -1], axis=1) - np.mean(surface[:, 0], axis=1)

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            return math.nan
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "level_change_std": float(np.std(level)),
        "skew_change_std": float(np.std(skew)),
        "term_change_std": float(np.std(term)),
        "spot_level_change_correlation": correlation(spot_changes, level),
        "spot_skew_change_correlation": correlation(spot_changes, skew),
        "spot_term_change_correlation": correlation(spot_changes, term),
    }


def _decoded_fixed_option_vol_changes(
    *,
    model,
    paths: np.ndarray,
    expiries: np.ndarray,
    normalized_strikes: np.ndarray,
    dt: float,
    block_size: int,
) -> np.ndarray:
    """Return one-day IV changes for fixed cash options along two-state paths."""

    import torch

    verify_shape = (paths.shape[0], 2, model.config.num_state + 1)
    if paths.shape != verify_shape:
        raise ValueError(f"paths have shape {paths.shape}, expected {verify_shape}")
    if expiries[0] <= dt:
        raise ValueError(f"first model expiry {expiries[0]} must exceed dt={dt}")

    samples = paths.shape[0]
    T = expiries.size
    N = normalized_strikes.size
    result = np.empty((samples, T * N), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, samples, block_size):
            end = min(start + block_size, samples)
            block = paths[start:end]
            count = end - start
            states = np.concatenate((block[:, 0, 1:], block[:, 1, 1:]), axis=0)
            decoder_expiries = np.concatenate(
                (
                    np.broadcast_to(expiries, (count, T)),
                    np.broadcast_to(expiries - dt, (count, T)),
                ),
                axis=0,
            )
            current_strikes = np.broadcast_to(
                normalized_strikes[None, None, :], (count, T, N)
            )
            next_strikes = current_strikes * np.exp(
                block[:, 0, 0] - block[:, 1, 0]
            )[:, None, None]
            strikes = np.concatenate((current_strikes, next_strikes), axis=0)

            torch_states = torch.as_tensor(states, dtype=model.env.fdtype, device=model.env.device)
            torch_expiries = torch.as_tensor(
                decoder_expiries, dtype=model.env.fdtype, device=model.env.device
            )
            torch_strikes = torch.as_tensor(
                strikes, dtype=model.env.fdtype, device=model.env.device
            )
            pricer = model.decoder(
                state=torch_states,
                expiries=torch_expiries,
                training=False,
            )
            calls = pricer(torch_strikes, training=False).detach().cpu().numpy()
            vols = bs.implied(
                k=strikes,
                sqrtT=np.sqrt(decoder_expiries)[:, :, None],
                prices=calls,
                on_exceed_bounds="quiet",
            )
            if not np.all(np.isfinite(vols)):
                raise FloatingPointError("fixed-option implied-volatility inversion failed")
            result[start:end] = (vols[count:] - vols[:count]).reshape(count, T * N)
    return result


def _decoded_nms_vol_levels(
    *,
    model,
    states: np.ndarray,
    expiries: np.ndarray,
    nms_grid: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Decode IV surface levels on the fixed normalized-moneyness grid."""

    import torch

    verify_shape = (states.shape[0], model.config.num_state)
    if states.shape != verify_shape:
        raise ValueError(f"states have shape {states.shape}, expected {verify_shape}")
    if nms_grid.shape[0] != expiries.size:
        raise ValueError(
            f"NMS grid has shape {nms_grid.shape}, expected first dimension {expiries.size}"
        )
    samples = states.shape[0]
    T, N = nms_grid.shape
    result = np.empty((samples, T * N), dtype=np.float64)
    torch_expiries = torch.as_tensor(
        expiries[None], dtype=model.env.fdtype, device=model.env.device
    )
    torch_nms = torch.as_tensor(
        nms_grid[None], dtype=model.env.fdtype, device=model.env.device
    )
    nms_eps = model.decoder.config.nms_eps
    model.eval()
    with torch.no_grad():
        for start in range(0, samples, block_size):
            end = min(start + block_size, samples)
            torch_states = torch.as_tensor(
                states[start:end], dtype=model.env.fdtype, device=model.env.device
            )
            pricer = model.decoder(
                state=torch_states,
                expiries=torch_expiries,
                training=False,
            )
            if nms_eps is None:
                strike_exponent = torch_nms * pricer.sqrtVars[:, :, None]
            else:
                variance_epsilon = (
                    pricer.sqrtVars[:, :1].square() / torch_expiries[:, :1]
                ) * nms_eps
                variance = pricer.sqrtVars.square()
                nms_adjustment = torch.sqrt(
                    (variance_epsilon / variance + 1.0) / (nms_eps + 1.0)
                )
                strike_exponent = (
                    torch_nms
                    * pricer.sqrtVars[:, :, None]
                    * nms_adjustment[:, :, None]
                )
            strikes = torch.exp(strike_exponent)
            calls = pricer(strikes, training=False).detach().cpu().numpy()
            numpy_strikes = strikes.detach().cpu().numpy()
            vols = bs.implied(
                k=numpy_strikes,
                sqrtT=np.sqrt(expiries)[None, :, None],
                prices=calls,
                eps=1.0e-6,
                on_exceed_bounds="quiet",
            )
            if not np.all(np.isfinite(vols)):
                raise FloatingPointError("NMS-grid implied-volatility inversion failed")
            result[start:end] = vols.reshape(end - start, T * N)
    return result


def _weekly_path_statistics(paths: np.ndarray) -> dict[str, float]:
    log_spot_change = paths[:, -1, 0] - paths[:, 0, 0]
    realized_variance = np.sum(np.square(np.diff(paths[..., 0], axis=1)), axis=1)
    surface_displacement = np.linalg.norm(paths[:, -1, 1:] - paths[:, 0, 1:], axis=1)
    return {
        "log_spot_change_mean": float(np.mean(log_spot_change)),
        "log_spot_change_std": float(np.std(log_spot_change)),
        "log_spot_change_q01": float(np.quantile(log_spot_change, 0.01)),
        "log_spot_change_q99": float(np.quantile(log_spot_change, 0.99)),
        "realized_log_variance_mean": float(np.mean(realized_variance)),
        "realized_log_variance_std": float(np.std(realized_variance)),
        "surface_displacement_mean": float(np.mean(surface_displacement)),
        "surface_displacement_std": float(np.std(surface_displacement)),
    }


def _input_iv_changes(market: dict, strike_count: int) -> np.ndarray:
    mid_vols = np.asarray(market["mid_vols"], dtype=np.float64)
    indices = np.rint(
        np.linspace(0, mid_vols.shape[-1] - 1, strike_count)
    ).astype(np.int64)
    selected = mid_vols[..., indices]
    changes = np.diff(selected, axis=0).reshape(selected.shape[0] - 1, -1)
    if not np.all(np.isfinite(changes)):
        raise FloatingPointError("market implied-volatility changes contain non-finite values")
    return changes


def run_diagnostics(
    *,
    state_samples: int,
    iv_samples: int,
    seed: int,
    output_directory: Path,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / DAYS_PER_YEAR
    factor_reports: dict[str, object] = {}
    plot_data: dict[str, object] = {}
    shared = None

    for factor_count in (5, None):
        label = "five_surface_factors" if factor_count == 5 else "full_surface_factors"
        experiment = GenDysanosExperiment(
            num_samples=state_samples,
            num_alpha=factor_count,
            use_matrix_exp=True,
            bandwidth=1.0,
            seed=seed + (5 if factor_count == 5 else 20),
        )
        grid_config, market, model, sampler = _load_generator_components(
            experiment,
            verbose=Context.quiet,
        )
        historical_surface = sampler.xstate[:, 1:].astype(np.float64)
        invariant_surface = sampler.sample_invariant_surface_states(state_samples).astype(np.float64)
        weekly_paths = sampler.free_sample_xstates(
            num_steps=6,
            num_samples=state_samples,
            dt=dt,
            bandwidth=1.0,
        ).astype(np.float64)
        historical_weekly_change = sampler.xstate[5:] - sampler.xstate[:-5]
        simulated_weekly_change = weekly_paths[:, -1] - weekly_paths[:, 0]
        rng = np.random.default_rng(seed + (105 if factor_count == 5 else 120))

        historical_pairs = np.stack(
            (sampler.xstate[:-1], sampler.xstate[1:]), axis=1
        ).astype(np.float64)
        simulated_pairs = sampler.free_sample_xstates(
            num_steps=2,
            num_samples=iv_samples,
            dt=dt,
            bandwidth=1.0,
        ).astype(np.float64)
        normalized_strikes = np.exp(np.linspace(-0.35, 0.35, 9))
        expiries = np.asarray(grid_config.grid_expiries, dtype=np.float64)
        historical_fixed_changes = _decoded_fixed_option_vol_changes(
            model=model,
            paths=historical_pairs,
            expiries=expiries,
            normalized_strikes=normalized_strikes,
            dt=dt,
            block_size=2_048,
        )
        simulated_fixed_changes = _decoded_fixed_option_vol_changes(
            model=model,
            paths=simulated_pairs,
            expiries=expiries,
            normalized_strikes=normalized_strikes,
            dt=dt,
            block_size=2_048,
        )
        input_changes = _input_iv_changes(market, normalized_strikes.size)
        input_pca = _pca_summary(input_changes)
        historical_pca = _pca_summary(historical_fixed_changes)
        simulated_pca = _pca_summary(simulated_fixed_changes)
        market_nms = np.asarray(market["nms"], dtype=np.float64)
        nms_grid = market_nms[0]
        if not np.allclose(market_nms, nms_grid[None], rtol=0.0, atol=1.0e-12):
            raise RuntimeError("historical NMS coordinates are not constant across dates")
        historical_nms_levels = np.asarray(
            market["mid_vols"], dtype=np.float64
        ).reshape(len(market.dates), -1)
        simulated_nms_levels = _decoded_nms_vol_levels(
            model=model,
            states=invariant_surface[:iv_samples],
            expiries=expiries,
            nms_grid=nms_grid,
            block_size=512,
        )
        historical_nms_pca = _leading_pca(historical_nms_levels, components=5)
        simulated_nms_pca = _leading_pca(simulated_nms_levels, components=5)
        historical_nms_components, simulated_nms_components = (
            _align_eigensurface_signs(
                historical_nms_pca["components"],
                simulated_nms_pca["components"],
            )
        )
        rankwise_nms_angles = _rankwise_component_angles_degrees(
            historical_nms_components,
            simulated_nms_components,
        )
        historical_nms_changes = np.diff(historical_nms_levels, axis=0)
        simulated_pair_nms_levels = _decoded_nms_vol_levels(
            model=model,
            states=simulated_pairs[:, :, 1:].reshape(
                2 * simulated_pairs.shape[0], model.config.num_state
            ),
            expiries=expiries,
            nms_grid=nms_grid,
            block_size=512,
        ).reshape(simulated_pairs.shape[0], 2, -1)
        simulated_nms_changes = (
            simulated_pair_nms_levels[:, 1] - simulated_pair_nms_levels[:, 0]
        )
        historical_nms_change_pca = _leading_pca(
            historical_nms_changes, components=5
        )
        simulated_nms_change_pca = _leading_pca(
            simulated_nms_changes, components=5
        )
        (
            historical_nms_change_components,
            simulated_nms_change_components,
        ) = _align_eigensurface_signs(
            historical_nms_change_pca["components"],
            simulated_nms_change_pca["components"],
        )
        rankwise_nms_change_angles = _rankwise_component_angles_degrees(
            historical_nms_change_components,
            simulated_nms_change_components,
        )
        historical_spot_changes = np.diff(sampler.xstate[:, 0])
        simulated_spot_changes = simulated_pairs[:, 1, 0] - simulated_pairs[:, 0, 0]

        factor_reports[label] = {
            "num_surface_factors": sampler.pcaxar1.num_surface_alpha,
            "num_total_factors": sampler.num_alpha,
            "surface_innovation_variance_explained": sampler.pcaxar1.surface_var_explained,
            "joint_innovation_variance_explained": sampler.pcaxar1.var_explained,
            "spot_innovation_variance_explained_by_surface_factors": sampler.pcaxar1.spot_surface_r2,
            "surface_generator_minimum_real_eigenvalue": float(
                np.min(np.linalg.eigvals(sampler.pcaxar1.k[1:, 1:]).real)
            ),
            "invariant_surface_vs_historical_states": _distribution_metrics(
                historical_surface,
                invariant_surface,
                rng=rng,
            ),
            "weekly_increment_vs_historical_weekly_increment": _distribution_metrics(
                historical_weekly_change,
                simulated_weekly_change,
                rng=rng,
            ),
            "weekly_historical_path_statistics": _weekly_path_statistics(
                np.lib.stride_tricks.sliding_window_view(
                    sampler.xstate, window_shape=6, axis=0
                ).transpose(0, 2, 1)
            ),
            "weekly_simulated_path_statistics": _weekly_path_statistics(weekly_paths),
            "iv_pca": {
                "input_surface_grid": _public_pca_summary(input_pca),
                "historical_states_fixed_cash_options": _public_pca_summary(
                    historical_pca
                ),
                "simulated_states_fixed_cash_options": _public_pca_summary(
                    simulated_pca
                ),
                "historical_vs_simulated_principal_angles_degrees": {
                    str(count): _principal_angles_degrees(
                        historical_pca["components"],
                        simulated_pca["components"],
                        count,
                    )
                    for count in (1, 3, 5)
                },
            },
            "nms_grid_iv_eigensurfaces": {
                "historical_observations": int(len(market.dates)),
                "simulated_observations": int(simulated_nms_levels.shape[0]),
                "quantity": "implied-volatility surface levels",
                "grid_shape": [int(expiries.size), int(nms_grid.shape[1])],
                "historical_explained_variance_ratio": historical_nms_pca[
                    "explained_variance_ratio"
                ],
                "simulated_explained_variance_ratio": simulated_nms_pca[
                    "explained_variance_ratio"
                ],
                "rankwise_component_angles_degrees": rankwise_nms_angles,
            },
            "nms_grid_iv_change_eigensurfaces": {
                "historical_observations": int(historical_nms_changes.shape[0]),
                "simulated_observations": int(simulated_nms_changes.shape[0]),
                "quantity": (
                    "one-day changes of implied-volatility surfaces on the "
                    "moving normalized-moneyness grid; independent of spot"
                ),
                "grid_shape": [int(expiries.size), int(nms_grid.shape[1])],
                "historical_explained_variance_ratio": historical_nms_change_pca[
                    "explained_variance_ratio"
                ],
                "simulated_explained_variance_ratio": simulated_nms_change_pca[
                    "explained_variance_ratio"
                ],
                "rankwise_component_angles_degrees": rankwise_nms_change_angles,
            },
            "historical_fixed_option_stylized_statistics": _stylized_statistics(
                historical_fixed_changes,
                historical_spot_changes,
                expiries=expiries.size,
                strikes=normalized_strikes.size,
            ),
            "simulated_fixed_option_stylized_statistics": _stylized_statistics(
                simulated_fixed_changes,
                simulated_spot_changes,
                expiries=expiries.size,
                strikes=normalized_strikes.size,
            ),
        }
        plot_data[label] = {
            "invariant_metrics": factor_reports[label]["invariant_surface_vs_historical_states"],
            "weekly_metrics": factor_reports[label]["weekly_increment_vs_historical_weekly_increment"],
            "input_pca": input_pca,
            "historical_pca": historical_pca,
            "simulated_pca": simulated_pca,
            "nms_grid": nms_grid,
            "expiries": expiries,
            "historical_nms_components": historical_nms_components,
            "simulated_nms_components": simulated_nms_components,
            "historical_nms_variance_ratio": historical_nms_pca[
                "explained_variance_ratio"
            ],
            "simulated_nms_variance_ratio": simulated_nms_pca[
                "explained_variance_ratio"
            ],
            "historical_nms_observation_count": int(historical_nms_levels.shape[0]),
            "simulated_nms_observation_count": int(simulated_nms_levels.shape[0]),
            "historical_nms_change_components": historical_nms_change_components,
            "simulated_nms_change_components": simulated_nms_change_components,
            "historical_nms_change_variance_ratio": historical_nms_change_pca[
                "explained_variance_ratio"
            ],
            "simulated_nms_change_variance_ratio": simulated_nms_change_pca[
                "explained_variance_ratio"
            ],
            "historical_nms_change_observation_count": int(
                historical_nms_changes.shape[0]
            ),
            "simulated_nms_change_observation_count": int(
                simulated_nms_changes.shape[0]
            ),
        }
        if shared is None:
            shared = {
                "first_date": str(market.dates[0]),
                "last_date": str(market.dates[-1]),
                "historical_dates": int(len(market.dates)),
                "state_dimension": int(sampler.xstate.shape[1] - 1),
                "model_expiries_years": expiries,
                "fixed_normalized_strikes": normalized_strikes,
            }

    report = {
        "definition": (
            "The 20-dimensional ML-SANOS state is fitted first as a stationary "
            "PCA--AR(1) process. Log-spot is then added with constant drift, "
            "contemporaneous surface-factor loadings, and one orthogonal factor."
        ),
        "simulation": {
            "state_samples_per_fit": state_samples,
            "fixed_option_path_samples_per_fit": iv_samples,
            "weekly_path_states": 6,
            "daily_dt_years": dt,
            "seed": seed,
        },
        "data": shared,
        "fits": factor_reports,
    }
    _write_figures(plot_data, output_directory)
    report_path = output_directory / "surface_first_diagnostics.json"
    report_path.write_text(
        json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _write_figures(plot_data: dict[str, object], output_directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantdata-matplotlib")
    import matplotlib.pyplot as plt

    labels = ("five_surface_factors", "full_surface_factors")
    display_labels = ("five surface factors", "20 surface factors")
    metric_names = (
        "mean_rms_in_historical_std",
        "relative_covariance_frobenius",
        "sliced_wasserstein_in_historical_std",
        "coordinate_ks_mean",
    )
    metric_labels = ("mean RMS", "covariance", "sliced W1", "mean KS")
    x = np.arange(len(metric_names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    for axis, source, title in zip(
        axes,
        ("invariant_metrics", "weekly_metrics"),
        ("Invariant states", "Five-day changes"),
        strict=True,
    ):
        for offset, (label, display) in enumerate(zip(labels, display_labels, strict=True)):
            values = [plot_data[label][source][name] for name in metric_names]
            axis.bar(x + (offset - 0.5) * width, values, width, label=display)
        axis.set_xticks(x, metric_labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("distance (standardized where applicable)")
    axes[1].legend(frameon=False)
    fig.savefig(output_directory / "surface_first_distances.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    for axis, label, title in zip(axes, labels, display_labels, strict=True):
        for source, display, style in (
            ("input_pca", "input IV grid", "-"),
            ("historical_pca", "fixed options, historical states", "--"),
            ("simulated_pca", "fixed options, simulated states", ":"),
        ):
            cumulative = np.asarray(
                plot_data[label][source]["cumulative_explained_variance"]
            )
            axis.plot(
                np.arange(1, cumulative.size + 1),
                cumulative,
                style,
                marker="o",
                markersize=3,
                label=display,
            )
        axis.axhline(0.9, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylim(0.0, 1.01)
        axis.set_xlabel("number of principal components")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("cumulative variance explained")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output_directory / "surface_first_iv_pca.png", dpi=220)
    plt.close(fig)

    five = plot_data["five_surface_factors"]
    full = plot_data["full_surface_factors"]
    nms_grid = np.asarray(five["nms_grid"])
    expiries = np.asarray(five["expiries"])
    if not np.allclose(nms_grid, full["nms_grid"]):
        raise RuntimeError("factor fits use different NMS grids")
    if not np.allclose(expiries, full["expiries"]):
        raise RuntimeError("factor fits use different expiry grids")
    eigensurface_rows = (
        (
            "Market",
            np.asarray(five["historical_nms_components"]),
            np.asarray(five["historical_nms_variance_ratio"]),
        ),
        (
            r"$\mathtt{num\_alpha=5}$",
            np.asarray(five["simulated_nms_components"]),
            np.asarray(five["simulated_nms_variance_ratio"]),
        ),
        (
            r"$\mathtt{num\_alpha=None}$",
            np.asarray(full["simulated_nms_components"]),
            np.asarray(full["simulated_nms_variance_ratio"]),
        ),
    )
    absolute_limit = max(
        float(np.max(np.abs(components)))
        for _, components, _ in eigensurface_rows
    )
    expiry_labels = tuple(
        "6m"
        if np.isclose(expiry, 0.5)
        else "1y"
        if np.isclose(expiry, 1.0)
        else f"{int(round(expiry * 255.0))}d"
        for expiry in expiries
    )
    x_coordinates = nms_grid[0]
    if not np.allclose(nms_grid, x_coordinates[None]):
        raise RuntimeError("NMS coordinates differ across expiries")
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(11.5, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, (row_label, components, variance_ratio) in enumerate(
        eigensurface_rows
    ):
        surfaces = components.reshape(5, expiries.size, nms_grid.shape[1])
        for component, axis in enumerate(axes[row]):
            image = axis.imshow(
                surfaces[component],
                origin="lower",
                aspect="auto",
                extent=(
                    float(x_coordinates[0]),
                    float(x_coordinates[-1]),
                    -0.5,
                    expiries.size - 0.5,
                ),
                cmap="RdBu_r",
                vmin=-absolute_limit,
                vmax=absolute_limit,
                interpolation="nearest",
                rasterized=True,
            )
            axis.axvline(0.0, color="black", linewidth=0.45, alpha=0.5)
            axis.set_title(
                f"PC {component + 1}: {100.0 * variance_ratio[component]:.1f}%",
                fontsize=9,
            )
            axis.set_xticks((-2.0, -1.0, 0.0, 1.0))
            axis.set_yticks(
                np.arange(expiries.size),
                expiry_labels,
            )
            if row == len(eigensurface_rows) - 1:
                axis.set_xlabel("NMS")
            if component == 0:
                axis.set_ylabel(f"{row_label}\nexpiry")
    if image is None:
        raise RuntimeError("no eigensurface image was generated")
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.72,
        pad=0.015,
        label="unit-norm PCA loading",
    )
    fig.savefig(
        output_directory / "surface_first_iv_eigensurfaces.png",
        dpi=220,
    )
    plt.close(fig)

    change_eigensurface_rows = (
        (
            "Market",
            np.asarray(five["historical_nms_change_components"]),
            np.asarray(five["historical_nms_change_variance_ratio"]),
        ),
        (
            r"$\mathtt{num\_alpha=5}$",
            np.asarray(five["simulated_nms_change_components"]),
            np.asarray(five["simulated_nms_change_variance_ratio"]),
        ),
        (
            r"$\mathtt{num\_alpha=None}$",
            np.asarray(full["simulated_nms_change_components"]),
            np.asarray(full["simulated_nms_change_variance_ratio"]),
        ),
    )
    change_absolute_limit = max(
        float(np.max(np.abs(components)))
        for _, components, _ in change_eigensurface_rows
    )
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(11.5, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, (row_label, components, variance_ratio) in enumerate(
        change_eigensurface_rows
    ):
        surfaces = components.reshape(5, expiries.size, nms_grid.shape[1])
        for component, axis in enumerate(axes[row]):
            image = axis.imshow(
                surfaces[component],
                origin="lower",
                aspect="auto",
                extent=(
                    float(x_coordinates[0]),
                    float(x_coordinates[-1]),
                    -0.5,
                    expiries.size - 0.5,
                ),
                cmap="RdBu_r",
                vmin=-change_absolute_limit,
                vmax=change_absolute_limit,
                interpolation="nearest",
                rasterized=True,
            )
            axis.axvline(0.0, color="black", linewidth=0.45, alpha=0.5)
            axis.set_title(
                f"PC {component + 1}: {100.0 * variance_ratio[component]:.1f}%",
                fontsize=9,
            )
            axis.set_xticks((-2.0, -1.0, 0.0, 1.0))
            axis.set_yticks(np.arange(expiries.size), expiry_labels)
            if row == len(change_eigensurface_rows) - 1:
                axis.set_xlabel("NMS")
            if component == 0:
                axis.set_ylabel(f"{row_label}\nexpiry")
    if image is None:
        raise RuntimeError("no change-eigensurface image was generated")
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.72,
        pad=0.015,
        label="unit-norm PCA loading",
    )
    fig.savefig(
        output_directory / "surface_first_iv_change_eigensurfaces.png",
        dpi=220,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-samples", type=int, default=100_000)
    parser.add_argument("--iv-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=614_203)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tex" / "generated",
    )
    arguments = parser.parse_args()
    report = run_diagnostics(
        state_samples=arguments.state_samples,
        iv_samples=arguments.iv_samples,
        seed=arguments.seed,
        output_directory=arguments.output_directory,
    )
    print(
        f"Wrote surface-first diagnostics for {', '.join(report['fits'])} "
        f"to {arguments.output_directory}"
    )


if __name__ == "__main__":
    main()
